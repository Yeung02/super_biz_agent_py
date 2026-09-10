"""ISSUE-014 ConversationSummarizer 历史摘要测试。

这些测试只使用 fake LLM、fake MemorySaver 和本地 trace 文件。目标是先锁定摘要契约：
只在超轮次或超预算时触发，摘要必须过滤工具原始 payload、密钥、堆栈和未验证工具错误，
失败时保留最近完整轮次且不删除原始 checkpoint。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from app.core.request_context import RequestContext
from app.core.token_budget import TokenBudgetManager
from app.observability.tracing import TraceLogger


def _summarizer_module() -> object:
    """按需导入 summarizer，让 RED 阶段表现为明确的缺失模块失败。"""

    try:
        return importlib.import_module("app.memory.summarizer")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.memory.summarizer must exist for ISSUE-014: {exc}")


def _conversation_module() -> object:
    """按需导入 ConversationManager，验证 summarizer 已接入阶段 2 门面。"""

    return importlib.import_module("app.memory.conversation_manager")


def _turn(role: str, content: str, index: int) -> object:
    module = _conversation_module()
    return module.ConversationTurn(
        role=role,
        content=content,
        timestamp=f"2026-07-07T00:00:{index:02d}+00:00",
    )


def _long_turns(count: int = 5) -> tuple[object, ...]:
    turns: list[object] = []
    for index in range(count):
        turns.append(_turn("user", f"用户目标 {index}: 排查支付链路延迟。", index * 2))
        turns.append(_turn("assistant", f"已确认事实 {index}: 只看到 API 网关耗时升高。", index * 2 + 1))
    return tuple(turns)


class _RecordingLLM:
    """同步 fake LLM，记录 prompt 并返回固定摘要。"""

    def __init__(self, answer: str = "用户目标：排查支付链路延迟。已确认事实：API 网关耗时升高。") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return type("FakeLLMResponse", (), {"content": self.answer})()


class _FailingLLM:
    """摘要失败 fake，错误原文不得进入 trace 或用户可见上下文。"""

    def invoke(self, prompt: str) -> object:
        _ = prompt
        raise RuntimeError("provider exploded with api_key=sk-secret and stacktrace")


class _FakeMemorySaver:
    """ConversationManager 集成测试用内存 checkpoint fake。"""

    def __init__(self, messages: list[object]) -> None:
        self.messages = messages
        self.deleted_threads: list[str] = []

    def get_tuple(self, config: Mapping[str, object]) -> Mapping[str, object]:
        _ = config
        return {"channel_values": {"messages": self.messages}}

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


def test_summarizer_triggers_when_turn_count_exceeds_threshold(
    fake_request_context: RequestContext,
) -> None:
    """历史轮次超过阈值时生成摘要，并返回摘要元数据。"""

    module = _summarizer_module()
    llm = _RecordingLLM()
    summarizer = module.ConversationSummarizer(
        llm=llm,
        token_budget_manager=TokenBudgetManager(),
        max_source_turns=2,
        enabled=True,
    )

    result = summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], _long_turns(4)),
        existing_summary=None,
        budget=2048,
        ctx=fake_request_context,
    )

    assert result.triggered is True
    assert result.trigger_reason == "turn_count"
    assert result.summary is not None
    assert result.metadata["summary_version"] == "v1"
    assert result.metadata["source_turn_count"] == 8
    assert llm.prompts


def test_summarizer_triggers_when_history_tokens_exceed_budget(
    fake_request_context: RequestContext,
) -> None:
    """即使轮次没超阈值，只要历史 token 超预算也必须触发摘要。"""

    module = _summarizer_module()
    llm = _RecordingLLM()
    token_budget_manager = TokenBudgetManager()
    summarizer = module.ConversationSummarizer(
        llm=llm,
        token_budget_manager=token_budget_manager,
        max_source_turns=20,
        enabled=True,
    )

    result = summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], _long_turns(2)),
        existing_summary="此前摘要。",
        budget=12,
        ctx=fake_request_context,
    )

    assert result.triggered is True
    assert result.trigger_reason == "token_budget"
    assert "此前摘要" in llm.prompts[0]


def test_summarizer_filters_tool_payload_secret_stack_and_unverified_error(
    fake_request_context: RequestContext,
) -> None:
    """摘要 prompt 和输出都不得包含工具原文、密钥、堆栈或未验证工具错误。"""

    module = _summarizer_module()
    llm = _RecordingLLM(
        "用户目标：排查告警。raw_payload api_key=sk-secret Traceback tool failed at internal URL。"
    )
    summarizer = module.ConversationSummarizer(
        llm=llm,
        token_budget_manager=TokenBudgetManager(),
        max_source_turns=1,
        enabled=True,
    )
    turns = (
        _turn("user", "请根据工具结果排查告警。", 1),
        _turn(
            "assistant",
            (
                "工具返回 raw_payload={'api_key':'sk-secret','stacktrace':'boom'}，"
                "tool_error: 数据库已确认被删除，http://internal.local/trace"
            ),
            2,
        ),
        _turn("user", "继续。", 3),
        _turn("assistant", "已确认事实：用户只授权分析告警。", 4),
    )

    result = summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], turns),
        existing_summary=None,
        budget=2048,
        ctx=fake_request_context,
    )

    prompt = llm.prompts[0]
    summary = result.summary or ""
    forbidden_fragments = (
        "raw_payload",
        "sk-secret",
        "stacktrace",
        "Traceback",
        "tool_error",
        "http://internal.local",
        "数据库已确认被删除",
    )
    for fragment in forbidden_fragments:
        assert fragment not in prompt
        assert fragment not in summary


def test_summarizer_failure_keeps_recent_turns_and_records_safe_trace(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """摘要失败不影响主请求，不删除原始 checkpoint，trace 只记录错误码。"""

    module = _summarizer_module()
    trace_path = tmp_path / "trace.jsonl"
    summarizer = module.ConversationSummarizer(
        llm=_FailingLLM(),
        token_budget_manager=TokenBudgetManager(),
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True),
        max_source_turns=1,
        enabled=True,
    )

    result = summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], _long_turns(3)),
        existing_summary=None,
        budget=2048,
        ctx=fake_request_context,
    )

    assert result.triggered is True
    assert result.summary is None
    assert result.error_code == "LLM_PROVIDER_ERROR"
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "conversation.summary" in trace_text
    assert "LLM_PROVIDER_ERROR" in trace_text
    assert "sk-secret" not in trace_text
    assert "stacktrace" not in trace_text


def test_summary_length_is_limited_by_budget(fake_request_context: RequestContext) -> None:
    """LLM 返回过长摘要时必须按摘要预算截断。"""

    module = _summarizer_module()
    llm = _RecordingLLM("确认事实。" * 200)
    summarizer = module.ConversationSummarizer(
        llm=llm,
        token_budget_manager=TokenBudgetManager(),
        max_source_turns=1,
        max_summary_tokens=20,
        enabled=True,
    )

    result = summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], _long_turns(3)),
        existing_summary=None,
        budget=2048,
        ctx=fake_request_context,
    )

    assert result.summary is not None
    assert TokenBudgetManager().estimate_tokens(result.summary).token_count <= 20


def test_conversation_manager_uses_summarizer_without_exposing_summary_in_history(
    fake_request_context: RequestContext,
) -> None:
    """ConversationManager 只在上下文中使用摘要，对外历史接口仍返回旧结构。"""

    module = _conversation_module()
    summarizer_module = _summarizer_module()
    messages: list[object] = [
        {"role": "user", "content": "旧问题 " + "x" * 80, "timestamp": "t1"},
        {"role": "assistant", "content": "旧回答 " + "y" * 80, "timestamp": "t2"},
        {"role": "user", "content": "新问题", "timestamp": "t3"},
        {"role": "assistant", "content": "新回答", "timestamp": "t4"},
    ]
    memory = _FakeMemorySaver(messages)
    summarizer = summarizer_module.ConversationSummarizer(
        llm=_RecordingLLM("用户目标：排查旧问题。"),
        token_budget_manager=TokenBudgetManager(),
        max_source_turns=1,
        enabled=True,
    )
    manager = module.ConversationManager(
        memory,
        token_budget_manager=TokenBudgetManager(),
        summarizer=summarizer,
        recent_turns=1,
    )

    context = manager.load_context("session-1", budget=40, ctx=fake_request_context)
    history = manager.get_history("session-1", fake_request_context)

    assert context.summary == "用户目标：排查旧问题。"
    assert [turn.content for turn in context.recent_messages] == ["新问题", "新回答"]
    assert "summary" not in history[0]
    assert history[0] == {"role": "user", "content": "旧问题 " + "x" * 80, "timestamp": "t1"}
    assert memory.deleted_threads == []


def test_summary_trace_contains_contract_fields(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """conversation.summary trace 必须包含 trigger/source/token/error 契约字段。"""

    module = _summarizer_module()
    trace_path = tmp_path / "trace.jsonl"
    summarizer = module.ConversationSummarizer(
        llm=_RecordingLLM("用户目标：排查支付链路延迟。"),
        token_budget_manager=TokenBudgetManager(),
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True),
        max_source_turns=1,
        enabled=True,
    )

    summarizer.summarize_if_needed(
        turns=cast(tuple[object, ...], _long_turns(3)),
        existing_summary=None,
        budget=2048,
        ctx=fake_request_context,
    )

    event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["name"] == "conversation.summary"
    assert event["trigger_reason"] == "turn_count"
    assert event["source_turn_count"] == 6
    assert event["summary_tokens"] > 0
    assert event["error_code"] is None
