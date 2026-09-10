"""ISSUE-013 ConversationManager 与 MemorySaver 边界测试。

这些测试只使用内存 fake，不访问真实 Milvus、DashScope、MCP server 或网络。
测试重点是先锁定业务门面契约：业务层不能理解 MemorySaver checkpoint 的各种内部形态，
也不能把系统消息、工具原始 payload 或非文本内容暴露给历史 API。
"""

from __future__ import annotations

import importlib
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from app.core.request_context import RequestContext
from app.core.token_budget import TokenBudgetManager
from app.observability.tracing import TraceLogger


def _conversation_module() -> object:
    """按需导入 conversation manager，让 RED 阶段表现为明确的缺失模块失败。"""

    try:
        return importlib.import_module("app.memory.conversation_manager")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.memory.conversation_manager must exist for ISSUE-013: {exc}")


def _msg(role: str, content: object, *, timestamp: str | None = None) -> dict[str, object]:
    message: dict[str, object] = {"role": role, "content": content}
    if timestamp is not None:
        message["timestamp"] = timestamp
    return message


def _checkpoint(messages: list[object], *, summary: str | None = None) -> dict[str, object]:
    channel_values: dict[str, object] = {"messages": messages}
    if summary is not None:
        channel_values["summary"] = summary
    return {"channel_values": channel_values}


class _FakeMemorySaver:
    """MemorySaver fake，支持裸 checkpoint、tuple 和 namedtuple 三种读取形态。"""

    def __init__(
        self,
        checkpoint: object | None = None,
        *,
        tuple_mode: str = "checkpoint",
        raise_on_read: bool = False,
        raise_on_clear: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.tuple_mode = tuple_mode
        self.raise_on_read = raise_on_read
        self.raise_on_clear = raise_on_clear
        self.read_configs: list[Mapping[str, object]] = []
        self.deleted_threads: list[str] = []

    def get_tuple(self, config: Mapping[str, object]) -> object | None:
        self.read_configs.append(config)
        if self.raise_on_read:
            raise RuntimeError("raw checkpoint read failure should stay internal")
        if self.tuple_mode == "namedtuple":
            CheckpointTuple = namedtuple("CheckpointTuple", ["checkpoint"])
            return CheckpointTuple(self.checkpoint)
        if self.tuple_mode == "tuple":
            return (self.checkpoint,)
        if self.tuple_mode == "none":
            return None
        return self.checkpoint

    def get(self, config: Mapping[str, object]) -> object | None:
        self.read_configs.append(config)
        if self.raise_on_read:
            raise RuntimeError("raw checkpoint read failure should stay internal")
        return self.checkpoint

    def delete_thread(self, thread_id: str) -> None:
        if self.raise_on_clear:
            raise RuntimeError("raw delete failure should stay internal")
        self.deleted_threads.append(thread_id)


class _AppendOnlyMemory:
    """提供 save_turn 适配方法的 fake，用来验证门面不会手写 LangGraph checkpoint。"""

    def __init__(self) -> None:
        self.saved_turns: list[dict[str, object]] = []

    def save_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.saved_turns.append(
            {
                "session_id": session_id,
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "metadata": dict(metadata or {}),
            }
        )


@dataclass(frozen=True)
class _FakeMessage:
    """同形 LangChain 消息 fake，避免测试提前绑定某个 SDK 版本。"""

    content: object
    timestamp: str | None = None


class SystemMessage(_FakeMessage):
    """用于确认系统消息不会出现在对外历史和业务上下文中。"""


class HumanMessage(_FakeMessage):
    """用于模拟用户消息，避免测试绑定真实 LangChain 版本。"""


class AIMessage(_FakeMessage):
    """用于模拟助手消息，避免测试绑定真实 LangChain 版本。"""


class ToolMessage(_FakeMessage):
    """用于确认工具原始 payload 不会暴露给 API。"""


def test_get_history_returns_empty_for_missing_checkpoint(
    fake_request_context: RequestContext,
) -> None:
    """空 MemorySaver 历史应稳定返回空列表，不把底层 None 暴露给 API。"""

    module = _conversation_module()
    manager = module.ConversationManager(_FakeMemorySaver(None, tuple_mode="none"))

    history = manager.get_history("session-1", fake_request_context)

    assert history == []


def test_get_history_supports_raw_checkpoint_and_filters_internal_messages(
    fake_request_context: RequestContext,
) -> None:
    """对外历史只保留用户/助手纯文本，过滤系统消息、工具结果和非文本 payload。"""

    module = _conversation_module()
    checkpoint = _checkpoint(
        [
            _msg("system", "系统提示不得外显"),
            _msg("user", "用户问题", timestamp="2026-07-07T01:00:00+00:00"),
            _msg("assistant", "助手回答", timestamp="2026-07-07T01:00:01+00:00"),
            _msg("tool", {"raw_payload": "工具原始结果不得外显"}),
            _msg("assistant", {"raw_payload": "非文本 assistant payload 也不得外显"}),
        ]
    )
    manager = module.ConversationManager(_FakeMemorySaver(checkpoint))

    history = manager.get_history("session-1", fake_request_context)

    assert history == [
        {
            "role": "user",
            "content": "用户问题",
            "timestamp": "2026-07-07T01:00:00+00:00",
        },
        {
            "role": "assistant",
            "content": "助手回答",
            "timestamp": "2026-07-07T01:00:01+00:00",
        },
    ]


def test_load_context_supports_namedtuple_checkpoint_and_keeps_recent_turns(
    fake_request_context: RequestContext,
) -> None:
    """load_context 负责最近 N 轮选择，而不是让业务层直接解析 MemorySaver。"""

    module = _conversation_module()
    messages: list[object] = [SystemMessage("系统提示不得进入业务历史")]
    for index in range(5):
        messages.append(HumanMessage(f"用户问题 {index}", timestamp=f"t{index}-u"))
        messages.append(AIMessage(f"助手回答 {index}", timestamp=f"t{index}-a"))
    manager = module.ConversationManager(
        _FakeMemorySaver(_checkpoint(messages, summary="已有安全摘要"), tuple_mode="namedtuple"),
        recent_turns=2,
    )

    context = manager.load_context("session-1", budget=None, ctx=fake_request_context)

    assert context.summary == "已有安全摘要"
    assert [turn.content for turn in context.recent_messages] == [
        "用户问题 3",
        "助手回答 3",
        "用户问题 4",
        "助手回答 4",
    ]
    assert context.history_metadata["message_count"] == 10
    assert context.history_metadata["summary_used"] is True
    assert context.history_metadata["trimmed_count"] == 6


def test_load_context_supports_tuple_checkpoint_and_applies_token_budget(
    fake_request_context: RequestContext,
) -> None:
    """预算裁剪应发生在 ConversationManager 内，返回给 Agent 的历史不能无限增长。"""

    module = _conversation_module()
    messages: list[object] = []
    for index in range(8):
        messages.append(_msg("user", f"用户问题 {index} " + "x" * 160))
        messages.append(_msg("assistant", f"助手回答 {index} " + "y" * 160))
    token_budget_manager = TokenBudgetManager()
    manager = module.ConversationManager(
        _FakeMemorySaver(_checkpoint(messages), tuple_mode="tuple"),
        token_budget_manager=token_budget_manager,
        recent_turns=8,
    )

    context = manager.load_context("session-1", budget=80, ctx=fake_request_context)

    assert 0 < len(context.recent_messages) < 16
    assert context.recent_messages[-1].content.startswith("助手回答 7")
    assert context.history_metadata["trimmed_count"] > 0


def test_load_context_returns_empty_and_traces_when_memory_read_fails(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """读取失败默认 fail-open 为空历史，并只记录安全错误码，不暴露原始异常全文。"""

    module = _conversation_module()
    trace_path = tmp_path / "trace.jsonl"
    manager = module.ConversationManager(
        _FakeMemorySaver(raise_on_read=True),
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True),
    )

    context = manager.load_context("session-1", budget=None, ctx=fake_request_context)

    assert context.recent_messages == ()
    assert context.history_metadata["load_failed"] is True
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "conversation.load" in trace_text
    assert "INTERNAL_ERROR" in trace_text
    assert "raw checkpoint read failure" not in trace_text


def test_clear_session_delegates_to_memory_saver_and_reports_failures(
    fake_request_context: RequestContext,
) -> None:
    """清理通过门面调用底层 delete_thread，失败时返回 false 而不是裸异常。"""

    module = _conversation_module()
    successful_memory = _FakeMemorySaver()
    failing_memory = _FakeMemorySaver(raise_on_clear=True)

    assert module.ConversationManager(successful_memory).clear_session(
        "session-1",
        fake_request_context,
    )
    assert successful_memory.deleted_threads == ["session-1"]
    assert module.ConversationManager(failing_memory).clear_session(
        "session-1",
        fake_request_context,
    ) is False


def test_save_turn_uses_adapter_method_instead_of_mutating_raw_checkpoint(
    fake_request_context: RequestContext,
) -> None:
    """save_turn 只调用受控适配方法，避免手工拼写 LangGraph checkpoint 内部结构。"""

    module = _conversation_module()
    memory = _AppendOnlyMemory()
    manager = module.ConversationManager(memory)

    saved = manager.save_turn(
        "session-1",
        "用户问题",
        "助手回答",
        {"trace_id": "trc_test"},
        fake_request_context,
    )

    assert saved is True
    assert memory.saved_turns == [
        {
            "session_id": "session-1",
            "user_msg": "用户问题",
            "assistant_msg": "助手回答",
            "metadata": {"trace_id": "trc_test"},
        }
    ]


def test_rag_agent_service_delegates_history_and_clear_to_conversation_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RagAgentService 的公开历史/清理方法应只委托 ConversationManager 门面。"""

    from app.services import rag_agent_service as rag_module

    class _FakeConversationManager:
        def __init__(self) -> None:
            self.history_called_with: str | None = None
            self.clear_called_with: str | None = None

        def get_history(
            self,
            session_id: str,
            ctx: object | None = None,
        ) -> list[dict[str, str]]:
            self.history_called_with = session_id
            return [{"role": "user", "content": "hello", "timestamp": "t"}]

        def clear_session(self, session_id: str, ctx: object | None = None) -> bool:
            self.clear_called_with = session_id
            return True

    service = cast(rag_module.RagAgentService, rag_module.RagAgentService.__new__(rag_module.RagAgentService))
    fake_manager = _FakeConversationManager()
    service.conversation_manager = fake_manager
    monkeypatch.setattr(rag_module.config, "conversation_manager_enabled", True, raising=False)

    assert service.get_session_history("session-1") == [
        {"role": "user", "content": "hello", "timestamp": "t"}
    ]
    assert service.clear_session("session-1") is True
    assert fake_manager.history_called_with == "session-1"
    assert fake_manager.clear_called_with == "session-1"
