"""ISSUE-B Critic 答案级自我批判节点测试。

覆盖三层边界：critic 节点三段式行为（预检/裁判/修订 + fail-open 全路径）、
aiops_service 的图路由决策（critic_enabled 开关与 critic_reviewed 防重入）、
SSE 事件格式化。ISSUE-C 追加复审（recheck）路径与 trace 事件落盘回归。
全部使用 fake ChatQwen，不连接真实 DashScope 或 LangGraph。
"""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from app.agent.aiops.critic import Claim, CriticVerdict
from app.config import config
from app.core.llm_usage import RawUsage, UsageAccumulator
from app.core.request_context import reset_request_context, set_request_context
from app.observability.tracing import TraceLogger


_EVIDENCE_USABLE = [
    {
        "step": "检查 CPU",
        "tool_name": "query_cpu_metrics",
        "usable": True,
        "text": "工具调用成功: cpu=83%",
    }
]
_EVIDENCE_UNUSABLE = [
    {
        "step": "检索日志",
        "tool_name": "search_log",
        "usable": False,
        "text": "该结果不能作为事实依据。",
    }
]


class _FakeStructuredChain:
    """with_structured_output fake：behavior 可以是返回值或 callable。"""

    def __init__(self, behavior: object) -> None:
        self._behavior = behavior
        self.invoked_messages: list[object] | None = None

    async def ainvoke(
        self, messages: object, config: dict[str, object] | None = None
    ) -> object:
        # critic 现在通过 ainvoke_structured_with_retry 传入 usage 采集 callbacks。
        _ = config
        self.invoked_messages = list(messages) if isinstance(messages, list) else messages
        if callable(self._behavior):
            return await self._behavior(messages) if asyncio.iscoroutinefunction(self._behavior) else self._behavior(messages)
        return self._behavior


class _FakeChatQwen:
    """按 schema 类型分派可编程行为的 ChatQwen fake。"""

    verdict_behavior: object = None
    revision_behavior: object = None

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs

    def with_structured_output(self, schema: object) -> _FakeStructuredChain:
        from app.agent.aiops.critic import CriticVerdict as _Verdict

        behavior = (
            _FakeChatQwen.verdict_behavior
            if schema is _Verdict
            else _FakeChatQwen.revision_behavior
        )
        return _FakeStructuredChain(behavior)


def _critic_module():
    return importlib.import_module("app.agent.aiops.critic")


def _patch_critic_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict_behavior: object = None,
    revision_behavior: object = None,
) -> object:
    """把 critic 模块的 ChatQwen 替换为可编程 fake。"""

    module = _critic_module()
    _FakeChatQwen.verdict_behavior = verdict_behavior
    _FakeChatQwen.revision_behavior = revision_behavior
    monkeypatch.setattr(module, "ChatQwen", _FakeChatQwen)
    return module


def _accept_verdict() -> CriticVerdict:
    return CriticVerdict(verdict="accept")


def _revise_verdict() -> CriticVerdict:
    return CriticVerdict(
        verdict="revise",
        unsupported_claims=[
            Claim(claim="CPU 使用率 83%", evidence_id=None, reason="证据块未包含该数值")
        ],
        revision_notes="删除无证据数值",
    )


@pytest.mark.asyncio
async def test_critic_skips_llm_when_no_evidence_and_strips_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无证据块时跳过 LLM 裁判，但仍执行确定性泄露剥离。"""

    def _must_not_be_called(messages: object) -> object:
        _ = messages
        raise AssertionError("critic must not call LLM without evidence blocks")

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_must_not_be_called)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "诊断完成。密钥 sk-abcd1234efg 已暴露，详见 http://internal.svc.local/report",
            "tool_evidence": [],
        }
    )

    assert result["critic_reviewed"] is True
    assert "sk-abcd1234efg" not in result["response"]
    assert "internal.svc.local" not in result["response"]
    assert "诊断完成" in result["response"]


@pytest.mark.asyncio
async def test_critic_no_evidence_clean_response_only_marks_reviewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无证据且无泄露：只标记已审查，不修改 response。"""

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_accept_verdict())

    result = await module.critic(
        {"input": "任务", "response": "干净的答案", "tool_evidence": []}
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_empty_response_marks_reviewed_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无响应可审查：直接标记已审，条件边据此放行。"""

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_accept_verdict())

    result = await module.critic({"input": "任务", "response": "", "tool_evidence": []})

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_accept_verdict_keeps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁判 accept：答案原样定稿，只补 critic_reviewed。"""

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_accept_verdict())

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%，来源证据 E1。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_revise_verdict_rewrites_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁判 revise：修订稿覆盖 response 并标记已审查。"""

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior={"response": "CPU 使用率未确认（证据不足）。"},
    )

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["critic_reviewed"] is True
    assert result["response"] == "CPU 使用率未确认（证据不足）。"


@pytest.mark.asyncio
async def test_critic_judge_failure_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁判 LLM 异常：fail-open 放行原答案（泄露剥离仍生效）。"""

    def _broken(messages: object) -> object:
        _ = messages
        raise RuntimeError("provider down")

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_broken)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "结论正常。密钥 sk-abcdef123456 泄露",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["critic_reviewed"] is True
    assert "sk-abcdef123456" not in result["response"]
    assert "结论正常" in result["response"]


@pytest.mark.asyncio
async def test_critic_judge_timeout_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁判超时：fail-open 放行原答案，不阻塞出答案。"""

    async def _slow(messages: object) -> object:
        _ = messages
        await asyncio.sleep(1)
        return _accept_verdict()

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_slow)
    monkeypatch.setattr(config, "critic_timeout_seconds", 0.05, raising=False)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_revision_failure_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修订 LLM 异常：fail-open 放行原答案。"""

    def _broken_revision(messages: object) -> object:
        _ = messages
        raise RuntimeError("revision provider down")

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior=_broken_revision,
    )

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_blank_revision_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空/空白修订稿是结构化输出失败：重试耗尽后 fail-open，不能用空串覆盖原答案。"""

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior={"response": "   "},
    )

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_rejects_expanding_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修订稿显著长于原稿：违反"只收缩"约束，拒绝采纳。"""

    original = "CPU 使用率 83%。"
    expanding = {"response": "x" * (len(original) + 500)}

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior=expanding,
    )

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": original,
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_sanitizes_revised_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修订稿自身携带泄露：定稿前必须重跑确定性剥离。"""

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior={"response": "CPU 未确认。详情见 http://internal.svc.local/debug"},
    )

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["critic_reviewed"] is True
    assert "internal.svc.local" not in result["response"]
    assert "CPU 未确认" in result["response"]


@pytest.mark.asyncio
async def test_critic_skips_revision_when_max_revisions_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """critic_max_revisions=0：裁判结果为 revise 也不修订（轮次预算耗尽）。"""

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior={"response": "should not be used"},
    )
    monkeypatch.setattr(config, "critic_max_revisions", 0, raising=False)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result == {"critic_reviewed": True}


@pytest.mark.asyncio
async def test_critic_injects_contradiction_signal_into_judge_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败工具 + 成功表述的矛盾信号必须注入裁判 prompt。"""

    captured: dict[str, object] = {}

    def _capture(messages: object) -> object:
        captured["messages"] = messages
        return _accept_verdict()

    module = _patch_critic_llm(monkeypatch, verdict_behavior=_capture)

    await module.critic(
        {
            "input": "诊断系统",
            "response": "日志检索已完成，CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE + _EVIDENCE_UNUSABLE,
        }
    )

    messages_text = str(captured["messages"])
    assert "矛盾" in messages_text
    assert "E1" in messages_text and "E2" in messages_text
    assert "可用=否" in messages_text


def test_precheck_counts_contradiction_only_with_unusable_evidence() -> None:
    """矛盾检测只在存在不可用证据块时触发。"""

    module = _critic_module()

    with_unusable = module._precheck(
        "日志检索已完成。", _EVIDENCE_USABLE + _EVIDENCE_UNUSABLE
    )
    all_usable = module._precheck(
        "日志检索已完成。", _EVIDENCE_USABLE + _EVIDENCE_USABLE
    )

    assert with_unusable.contradiction_hits == 1
    assert with_unusable.usable_evidence_count == 1
    assert with_unusable.total_evidence_count == 2
    assert all_usable.contradiction_hits == 0


def test_critic_settings_have_stable_defaults() -> None:
    """Critic 配置默认开启，回滚开关与边界值稳定。"""

    from app.config import Settings

    settings = Settings()

    assert settings.critic_enabled is True
    assert settings.critic_model == "qwen-plus"
    assert settings.critic_timeout_seconds == 8.0
    assert settings.critic_max_revisions == 1
    assert settings.critic_recheck_enabled is False


def test_state_declares_critic_reviewed_flag() -> None:
    """PlanExecuteState 必须声明 critic_reviewed（条件边防重入依赖）。"""

    from app.agent.aiops.state import PlanExecuteState

    assert "critic_reviewed" in PlanExecuteState.__annotations__


# ---------------------------------------------------------------------------
# 图路由决策
# ---------------------------------------------------------------------------


def _routing_module():
    return importlib.import_module("app.services.aiops_service")


def test_route_to_critic_when_enabled_and_unreviewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """response 已生成、未审查且开关开启：路由进 critic。"""

    from langgraph.graph import END

    module = _routing_module()
    monkeypatch.setattr(config, "critic_enabled", True, raising=False)

    assert module._should_continue({"response": "答案"}) == module.NODE_CRITIC
    assert module.NODE_CRITIC == "critic"
    assert module.NODE_CRITIC != END


def test_route_to_end_when_critic_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开关关闭：行为与接入 critic 前的旧图完全一致（直接 END）。"""

    from langgraph.graph import END

    module = _routing_module()
    monkeypatch.setattr(config, "critic_enabled", False, raising=False)

    assert module._should_continue({"response": "答案"}) == END


def test_route_to_end_when_already_reviewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """critic_reviewed=True：防止条件边重复路由进 critic。"""

    from langgraph.graph import END

    module = _routing_module()
    monkeypatch.setattr(config, "critic_enabled", True, raising=False)

    assert module._should_continue({"response": "答案", "critic_reviewed": True}) == END


def test_route_preserves_legacy_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """error_event/plan/空计划三条旧路由路径不受 critic 接入影响。"""

    from langgraph.graph import END

    module = _routing_module()
    monkeypatch.setattr(config, "critic_enabled", True, raising=False)

    assert module._should_continue({"error_event": {"type": "error"}}) == END
    assert module._should_continue({"plan": ["步骤1"]}) == module.NODE_EXECUTOR
    assert module._should_continue({"plan": []}) == END


def test_format_critic_event_reports_revision() -> None:
    """修订发生时推送 report 更新事件，客户端以最后一条 report/complete 为准。"""

    module = _routing_module()
    service = module.AIOpsService.__new__(module.AIOpsService)

    event = service._format_critic_event({"response": "修订后答案", "critic_reviewed": True})

    assert event["type"] == "report"
    assert event["stage"] == "critic_revised"
    assert event["report"] == "修订后答案"


def test_format_critic_event_status_when_not_revised() -> None:
    """未修订时只发 status 事件，不重复推送报告内容。"""

    module = _routing_module()
    service = module.AIOpsService.__new__(module.AIOpsService)

    event = service._format_critic_event({"critic_reviewed": True})
    empty_event = service._format_critic_event(None)

    assert event["type"] == "status"
    assert event["stage"] == "critic"
    assert empty_event["type"] == "status"


# ---------------------------------------------------------------------------
# ISSUE-C：复审（recheck）路径
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_recheck_accept_finalizes_revision_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复审开启：修订后复审返回 accept，提前定稿修订稿。"""

    verdicts = [_revise_verdict(), _accept_verdict()]

    def _verdict_sequence(messages: object) -> object:
        _ = messages
        return verdicts.pop(0) if verdicts else _accept_verdict()

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_verdict_sequence,
        revision_behavior={"response": "CPU 使用率未确认（证据不足）。"},
    )
    monkeypatch.setattr(config, "critic_recheck_enabled", True, raising=False)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["response"] == "CPU 使用率未确认（证据不足）。"
    assert result["critic_reviewed"] is True


@pytest.mark.asyncio
async def test_critic_recheck_failure_finalizes_current_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复审异常：定稿当前修订稿，不回退原稿也不阻塞出答案。"""

    calls = {"count": 0}

    def _verdict_then_fail(messages: object) -> object:
        _ = messages
        calls["count"] += 1
        if calls["count"] == 1:
            return _revise_verdict()
        raise RuntimeError("recheck provider down")

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_verdict_then_fail,
        revision_behavior={"response": "CPU 使用率未确认。"},
    )
    monkeypatch.setattr(config, "critic_recheck_enabled", True, raising=False)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["response"] == "CPU 使用率未确认。"
    assert result["critic_reviewed"] is True


@pytest.mark.asyncio
async def test_critic_runs_second_revision_when_recheck_still_revise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复审仍 revise 且轮次未耗尽：继续第二轮修订，最终采用第二轮修订稿。"""

    revisions = ["第一轮修订：CPU 使用率待确认。", "第二轮修订：CPU 使用率证据不足。"]

    def _revision_sequence(messages: object) -> object:
        _ = messages
        return {"response": revisions.pop(0)}

    module = _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior=_revision_sequence,
    )
    monkeypatch.setattr(config, "critic_recheck_enabled", True, raising=False)
    monkeypatch.setattr(config, "critic_max_revisions", 2, raising=False)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["response"] == "第二轮修订：CPU 使用率证据不足。"
    assert result["critic_reviewed"] is True


# ---------------------------------------------------------------------------
# ISSUE-C：trace 事件落盘
# ---------------------------------------------------------------------------


def _read_trace_events(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _patch_critic_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> object:
    """把 critic 模块的 trace_logger 指向临时文件，返回模块引用。"""

    module = _critic_module()
    monkeypatch.setattr(
        module,
        "_trace_logger",
        TraceLogger(
            trace_jsonl_path=str(tmp_path / "critic_trace.jsonl"),
            enabled=True,
            metrics_enabled=False,
        ),
    )
    return module


@pytest.mark.asyncio
async def test_critic_writes_trace_events_on_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    fake_request_context,
) -> None:
    """修订路径必须落 precheck/verdict/revise 三类 trace 事件，且不含草稿与证据原文。"""

    module = _patch_critic_trace(monkeypatch, tmp_path)
    _patch_critic_llm(
        monkeypatch,
        verdict_behavior=_revise_verdict(),
        revision_behavior={"response": "CPU 使用率未确认。"},
    )

    token = set_request_context(fake_request_context)
    try:
        await module.critic(
            {
                "input": "诊断系统",
                "response": "机密草稿标记：CPU 使用率 83%。",
                "tool_evidence": _EVIDENCE_USABLE,
            }
        )
    finally:
        reset_request_context(token)

    events = _read_trace_events(tmp_path / "critic_trace.jsonl")
    names = [event["name"] for event in events]
    assert names == [
        "agent.critic.precheck",
        "agent.critic.verdict",
        "agent.critic.revise",
    ]

    precheck_event = events[0]
    assert precheck_event["leak_hits"] == 0
    assert precheck_event["usable_evidence_count"] == 1
    assert precheck_event["total_evidence_count"] == 1
    assert precheck_event["contradiction_hits"] == 0

    verdict_event = events[1]
    assert verdict_event["verdict"] == "revise"
    assert verdict_event["unsupported_claim_count"] == 1
    assert verdict_event["model"] == config.critic_model

    revise_event = events[2]
    assert revise_event["revision_round"] == 1

    raw = (tmp_path / "critic_trace.jsonl").read_text(encoding="utf-8")
    assert "机密草稿标记" not in raw
    assert "cpu=83%" not in raw


@pytest.mark.asyncio
async def test_critic_writes_fail_open_trace_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    fake_request_context,
) -> None:
    """裁判异常必须落 fail_open trace 事件，携带 stage 与稳定 error_code。"""

    module = _patch_critic_trace(monkeypatch, tmp_path)

    def _broken(messages: object) -> object:
        _ = messages
        raise RuntimeError("provider down")

    _patch_critic_llm(monkeypatch, verdict_behavior=_broken)

    token = set_request_context(fake_request_context)
    try:
        await module.critic(
            {
                "input": "诊断系统",
                "response": "CPU 使用率 83%。",
                "tool_evidence": _EVIDENCE_USABLE,
            }
        )
    finally:
        reset_request_context(token)

    events = _read_trace_events(tmp_path / "critic_trace.jsonl")
    names = [event["name"] for event in events]
    assert names == ["agent.critic.precheck", "agent.critic.fail_open"]

    fail_event = events[1]
    assert fail_event["status"] == "error"
    assert fail_event["stage"] == "judge"
    assert fail_event["error_code"] == "LLM_PROVIDER_ERROR"


class _UsageReportingChain:
    """模拟 LangChain callback 分发：先向 config.callbacks 派发带 usage 的
    LLMResult，再返回脚本化行为（behavior 可以是返回值或抛异常的 callable）。
    """

    def __init__(self, behavior: object, *, usage: tuple[int, int]) -> None:
        self._behavior = behavior
        self._usage = usage

    async def ainvoke(
        self, messages: object, config: dict[str, object] | None = None
    ) -> object:
        if config is not None:
            input_tokens, output_tokens = self._usage
            message = SimpleNamespace(
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
            llm_result = SimpleNamespace(
                generations=[[SimpleNamespace(message=message)]],
                llm_output=None,
            )
            for handler in config.get("callbacks") or []:
                handler.on_llm_end(llm_result)
        if callable(self._behavior):
            result = self._behavior(messages)
            if isinstance(result, Exception):
                raise result
            return result
        return self._behavior


class _UsageRecordingChatQwen:
    """按 schema 分派行为与 usage 数值的 ChatQwen fake。"""

    verdict_behavior: object = None
    revision_behavior: object = None
    verdict_usage: tuple[int, int] = (0, 0)
    revision_usage: tuple[int, int] = (0, 0)

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs

    def with_structured_output(self, schema: object) -> _UsageReportingChain:
        from app.agent.aiops.critic import CriticVerdict as _Verdict

        if schema is _Verdict:
            return _UsageReportingChain(
                _UsageRecordingChatQwen.verdict_behavior,
                usage=_UsageRecordingChatQwen.verdict_usage,
            )
        return _UsageReportingChain(
            _UsageRecordingChatQwen.revision_behavior,
            usage=_UsageRecordingChatQwen.revision_usage,
        )


@pytest.mark.asyncio
async def test_critic_records_real_usage_for_judge_and_revise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """judge/revise 两次 LLM 调用各自记账累计的真实 usage，model 用 critic_model。"""

    module = _critic_module()
    _UsageRecordingChatQwen.verdict_behavior = _revise_verdict()
    _UsageRecordingChatQwen.revision_behavior = {
        "response": "CPU 使用率未确认（证据不足）。"
    }
    _UsageRecordingChatQwen.verdict_usage = (200, 40)
    _UsageRecordingChatQwen.revision_usage = (150, 60)
    monkeypatch.setattr(module, "ChatQwen", _UsageRecordingChatQwen)

    recorded: list[tuple[RawUsage | None, str]] = []

    def _spy(accumulator: object, *, model: str, ctx: object) -> None:
        _ = ctx
        usage = (
            accumulator.to_raw_usage()
            if isinstance(accumulator, UsageAccumulator)
            else None
        )
        recorded.append((usage, model))

    monkeypatch.setattr(module, "record_llm_usage", _spy)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    assert result["response"] == "CPU 使用率未确认（证据不足）。"
    assert len(recorded) == 2
    assert recorded[0] == (RawUsage(input_tokens=200, output_tokens=40), config.critic_model)
    assert recorded[1] == (RawUsage(input_tokens=150, output_tokens=60), config.critic_model)


@pytest.mark.asyncio
async def test_critic_records_usage_even_when_judge_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁判异常 fail-open 前已消耗的 usage 不能丢：finally 记账仍要执行。"""

    module = _critic_module()

    def _provider_error(messages: object) -> object:
        _ = messages
        return RuntimeError("provider down")

    _UsageRecordingChatQwen.verdict_behavior = _provider_error
    _UsageRecordingChatQwen.verdict_usage = (90, 12)
    monkeypatch.setattr(module, "ChatQwen", _UsageRecordingChatQwen)

    recorded: list[RawUsage | None] = []

    def _spy(accumulator: object, *, model: str, ctx: object) -> None:
        _ = model, ctx
        usage = (
            accumulator.to_raw_usage()
            if isinstance(accumulator, UsageAccumulator)
            else None
        )
        recorded.append(usage)

    monkeypatch.setattr(module, "record_llm_usage", _spy)

    result = await module.critic(
        {
            "input": "诊断系统",
            "response": "CPU 使用率 83%。",
            "tool_evidence": _EVIDENCE_USABLE,
        }
    )

    # fail-open 放行原答案，但 judge 已消耗的 usage 仍被记账
    assert result == {"critic_reviewed": True}
    assert recorded == [RawUsage(input_tokens=90, output_tokens=12)]
