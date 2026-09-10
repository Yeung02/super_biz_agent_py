"""ISSUE-C 灰度验证：真实 LangGraph 图的 Critic 端到端集成测试。

与 test_critic_node.py 的差异：这里构建真实 AIOpsService（conftest 已注入
MEMORY_CHECKPOINTER=memory，checkpointer 为进程内 MemorySaver），执行真实图
拓扑、真实 _should_continue 路由与真实 critic 节点；只有 planner/executor/
replanner 三个 LLM 节点函数和 critic 内的 ChatQwen 用 fake 替换。目标是灰度
开关切换下的全链路行为验证：

- critic_enabled=true：replanner 出答案后路由进 critic，修订稿透出 SSE 并进入
  complete 事件；
- critic_enabled=false：事件序列与接入 critic 前的旧图完全一致；
- tool_evidence 经 operator.add 累加后真实传入 critic 裁判 prompt。

不连接真实 DashScope、MCP、Redis 或 Milvus。
"""

from __future__ import annotations

import importlib
import uuid

import pytest

from app.agent.aiops.critic import Claim, CriticVerdict
from app.config import config


_ORIGINAL_RESPONSE = "CPU 使用率 83%，诊断完成。"
_REVISED_RESPONSE = "CPU 使用率未能确认（证据不足），其余诊断结论不变。"
_EVIDENCE_ENTRY = {
    "step": "查询 CPU 指标",
    "tool_name": "query_cpu_metrics",
    "usable": True,
    "text": "工具调用成功: cpu=83%",
}


class _FakeStructuredChain:
    """with_structured_output fake：behavior 可以是返回值或 callable。"""

    def __init__(self, behavior: object) -> None:
        self._behavior = behavior

    async def ainvoke(
        self, messages: object, config: dict[str, object] | None = None
    ) -> object:
        # critic 现在通过 ainvoke_structured_with_retry 传入 usage 采集 callbacks。
        _ = config
        if callable(self._behavior):
            behavior = self._behavior
            return behavior(messages)
        return self._behavior


class _FakeChatQwen:
    """critic 模块 ChatQwen fake：按 schema 类型分派可编程行为。"""

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


class _MustNotCallChatQwen:
    """critic 关闭时的哨兵：critic 节点不该运行，LLM 构造即失败。"""

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs
        raise AssertionError("critic LLM must not be constructed when critic_enabled=false")


async def _fake_planner(state: dict) -> dict:
    _ = state
    return {"plan": ["查询 CPU 指标"]}


async def _fake_executor(state: dict) -> dict:
    plan = state.get("plan", [])
    if not plan:
        return {}
    task = plan[0]
    return {
        "plan": plan[1:],
        "past_steps": [(task, "执行完成：cpu=83%")],
        # ISSUE-A 通道：真实图中经 operator.add 累加，供 critic 裁判核对。
        "tool_evidence": [dict(_EVIDENCE_ENTRY)],
    }


async def _fake_replanner(state: dict) -> dict:
    _ = state
    return {"response": _ORIGINAL_RESPONSE}


def _build_service(monkeypatch: pytest.MonkeyPatch, tmp_path) -> object:
    """构建真实 AIOpsService：真图真路由，只有三个 LLM 节点是 fake。"""

    # 关闭 trace/metrics 落盘，避免集成测试污染开发日志。
    monkeypatch.setattr(config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(config, "metrics_enabled", False, raising=False)
    monkeypatch.setattr(config, "trace_jsonl_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(config, "metrics_jsonl_path", str(tmp_path / "metrics.jsonl"))

    # 显式在包对象上打补丁：子模块导入会把同名包属性（如 critic）重绑成模块对象，
    # 字符串形式 monkeypatch 可能解析到子模块；这里保证 _build_graph 的
    # `from app.agent.aiops import ...` 拿到的是函数而非模块。
    import app.agent.aiops as aiops_package
    from app.agent.aiops.critic import critic as real_critic

    monkeypatch.setattr(aiops_package, "planner", _fake_planner)
    monkeypatch.setattr(aiops_package, "executor", _fake_executor)
    monkeypatch.setattr(aiops_package, "replanner", _fake_replanner)
    monkeypatch.setattr(aiops_package, "critic", real_critic)

    from app.services.aiops_service import AIOpsService

    return AIOpsService()


def _patch_real_critic_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict_behavior: object = None,
    revision_behavior: object = None,
) -> None:
    """把真实 critic 节点内的 ChatQwen 替换为可编程 fake。"""

    critic_module = importlib.import_module("app.agent.aiops.critic")
    _FakeChatQwen.verdict_behavior = verdict_behavior
    _FakeChatQwen.revision_behavior = revision_behavior
    monkeypatch.setattr(critic_module, "ChatQwen", _FakeChatQwen)


async def _collect_events(service: object) -> list[dict]:
    session_id = f"critic-it-{uuid.uuid4()}"
    return [event async for event in service.execute("诊断系统", session_id=session_id)]


@pytest.mark.asyncio
async def test_critic_enabled_end_to_end_revises_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """灰度开启：完整图跑 planner→executor→replanner→critic，修订稿进入 SSE 与 complete。"""

    service = _build_service(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def _capture_verdict(messages: object) -> object:
        captured["messages"] = messages
        return CriticVerdict(
            verdict="revise",
            unsupported_claims=[
                Claim(claim="CPU 使用率 83%", evidence_id=None, reason="证据块未包含该数值")
            ],
        )

    _patch_real_critic_llm(
        monkeypatch,
        verdict_behavior=_capture_verdict,
        revision_behavior={"response": _REVISED_RESPONSE},
    )
    monkeypatch.setattr(config, "critic_enabled", True, raising=False)

    events = await _collect_events(service)

    stages = [event.get("stage") for event in events]
    assert stages == [
        "plan_created",
        "step_executed",
        "final_report",
        "critic_revised",
        "complete",
    ]

    # 修订稿通过 SSE report 事件透出，客户端以最后一条 report/complete 为准。
    revised_events = [event for event in events if event.get("stage") == "critic_revised"]
    assert revised_events[0]["report"] == _REVISED_RESPONSE

    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["response"] == _REVISED_RESPONSE

    # 证据链经真实 state 累加传入裁判 prompt：E1 编号与证据文本必须在场。
    judge_messages_text = str(captured.get("messages"))
    assert "E1" in judge_messages_text
    assert "cpu=83%" in judge_messages_text
    assert _ORIGINAL_RESPONSE in judge_messages_text


@pytest.mark.asyncio
async def test_critic_disabled_preserves_legacy_event_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """灰度关闭：事件序列与接入 critic 前的旧图完全一致，critic LLM 不被构造。"""

    service = _build_service(monkeypatch, tmp_path)

    critic_module = importlib.import_module("app.agent.aiops.critic")
    monkeypatch.setattr(critic_module, "ChatQwen", _MustNotCallChatQwen)
    monkeypatch.setattr(config, "critic_enabled", False, raising=False)

    events = await _collect_events(service)

    stages = [event.get("stage") for event in events]
    assert stages == ["plan_created", "step_executed", "final_report", "complete"]

    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["response"] == _ORIGINAL_RESPONSE


@pytest.mark.asyncio
async def test_critic_accept_emits_status_and_keeps_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """灰度开启但裁判 accept：critic 只发 status 事件，答案保持原样。"""

    service = _build_service(monkeypatch, tmp_path)
    _patch_real_critic_llm(
        monkeypatch,
        verdict_behavior=CriticVerdict(verdict="accept"),
    )
    monkeypatch.setattr(config, "critic_enabled", True, raising=False)

    events = await _collect_events(service)

    stages = [event.get("stage") for event in events]
    assert stages == [
        "plan_created",
        "step_executed",
        "final_report",
        "critic",
        "complete",
    ]

    status_event = next(event for event in events if event.get("stage") == "critic")
    assert status_event["type"] == "status"

    complete = events[-1]
    assert complete["response"] == _ORIGINAL_RESPONSE
