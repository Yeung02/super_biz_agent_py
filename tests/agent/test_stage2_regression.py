"""ISSUE-016 阶段 2 单元和回归测试。

这些用例只锁定阶段 2 已经落地的契约：fallback、token 裁剪、ConversationManager
边界、摘要失败安全性和薄编排层回滚。测试全部使用内存 fake，不访问真实 Milvus、
DashScope、MCP server 或网络，避免把阶段验收变成集成环境验收。
"""

from __future__ import annotations

import json
import importlib
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.core.errors import (
    AppError,
    InternalAppError,
    InvalidInputError,
    LLMTimeoutError,
    ToolExecutionError,
    VectorStoreUnavailableError,
)
from app.core.fallback import FallbackEvidence, FallbackManager
from app.core.request_context import RequestContext
from app.core.token_budget import TokenBudgetManager, TokenContext
from app.memory.conversation_manager import ConversationManager
from app.observability.tracing import TraceLogger


@dataclass(frozen=True)
class _Stage2ToolResult:
    """阶段 2 回归测试用 ToolResult 同形对象，避免依赖真实 ToolManager 或 MCP。"""

    status: str
    is_error: bool
    data: dict[str, object]

    def is_evidence_usable(self) -> bool:
        """错误工具结果必须被排除在事实证据之外。"""

        return self.status == "success" and not self.is_error


@dataclass(frozen=True)
class _Stage2Chunk:
    """TokenBudgetManager 裁剪测试用 chunk 同形对象，不提前引入阶段 3 RAG 模型。"""

    chunk_id: str
    content: str
    normalized_score: float


@dataclass
class _Stage2RagService:
    """只模拟旧 RagAgentService 的 query/query_stream 边界。"""

    mode: str = "success"
    answer: str = "stage2 answer"

    def __post_init__(self) -> None:
        self.model_name = "qwen-test"
        self.system_prompt = "必须基于事实回答，不暴露内部错误。"
        self.query_calls: list[tuple[str, str]] = []

    async def query(self, question: str, session_id: str) -> str:
        self.query_calls.append((question, session_id))
        if self.mode == "milvus_fail":
            raise VectorStoreUnavailableError(
                internal_message="milvus://internal:19530 api_key=sk-secret"
            )
        if self.mode == "llm_fail":
            raise LLMTimeoutError()
        return self.answer

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        _ = question, session_id
        yield {"type": "content", "data": self.answer}
        yield {"type": "complete"}


@dataclass
class _Stage2AIOpsService:
    """只模拟旧 AIOpsService.diagnose 边界，阶段 2 不进入真实 LangGraph。"""

    async def diagnose(self, session_id: str) -> AsyncGenerator[dict[str, object], None]:
        yield {"type": "complete", "session_id": session_id, "response": "ok"}


class _ConversationBoundary:
    """编排层测试用会话门面，记录 load_context 调用而不解析真实 checkpoint。"""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []

    def load_context(
        self,
        session_id: str,
        budget: object,
        ctx: RequestContext | None = None,
    ) -> object:
        _ = budget
        self.loaded.append((session_id, ctx.trace_id if ctx is not None else ""))
        return SimpleNamespace(summary=None, recent_messages=(), history_metadata={})


@pytest.mark.parametrize(
    ("error", "scenario", "expected_fallback"),
    (
        (LLMTimeoutError(), "chat", True),
        (
            VectorStoreUnavailableError(internal_message="milvus://internal password=secret"),
            "chat",
            True,
        ),
        (
            ToolExecutionError(internal_message="tool failed with http://internal.local"),
            "aiops",
            True,
        ),
        (InvalidInputError(), "chat", False),
        (VectorStoreUnavailableError(internal_message="milvus down"), "health", False),
    ),
)
def test_stage2_fallback_matrix_locks_failure_policy_and_trace(
    tmp_path: Path,
    fake_request_context: RequestContext,
    error: AppError,
    scenario: str,
    expected_fallback: bool,
) -> None:
    """阶段 2 回归矩阵覆盖 LLM、Milvus、工具、输入错误和健康检查场景。"""

    trace_path = tmp_path / "trace.jsonl"
    manager = FallbackManager(
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)
    )
    evidence = FallbackEvidence(
        partial_answer="已确认 CPU runbook 可用，api_key=sk-secret http://internal.local",
        evidence_count=1,
        tool_results=(
            _Stage2ToolResult(
                status="error",
                is_error=True,
                data={"raw_payload": "password=secret"},
            ),
        ),
    )

    result = manager.decide(
        error,
        fake_request_context,
        scenario=scenario,
        evidence=evidence,
    )
    response_fields = manager.to_response_fields(result)
    serialized = json.dumps(response_fields, ensure_ascii=False)
    trace_events = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.fallback_used is expected_fallback
    assert trace_events[-1]["name"] == "fallback.decide"
    assert trace_events[-1]["reason_code"] == error.code
    assert trace_events[-1]["fallback_used"] is expected_fallback
    assert "sk-secret" not in serialized
    assert "http://internal.local" not in serialized
    assert "password=secret" not in serialized


def test_stage2_token_trim_snapshot_preserves_contract_order() -> None:
    """组合裁剪顺序用快照式断言锁定，防止后续阶段改坏 token 优先级。"""

    manager = TokenBudgetManager(model_context_windows={"qwen-test": 2048})
    budget = manager.allocate("rag_chat", "qwen-test").with_limits(
        input_tokens=100,
        history_tokens=30,
        rag_context_tokens=20,
        tool_result_tokens=20,
        summary_tokens=20,
    )
    context = TokenContext(
        system_prompt="系统安全约束必须保留。",
        current_question="当前用户问题必须保留。",
        debug_notes=("debug payload must drop",),
        tool_results=(
            _Stage2ToolResult(
                status="success",
                is_error=False,
                data={
                    "summary": "工具摘要",
                    "raw_payload": "x" * 1000,
                    "items": [{"index": index, "value": "y" * 40} for index in range(10)],
                },
            ),
        ),
        rag_chunks=(
            _Stage2Chunk("low", "低分证据 " + "l" * 100, 0.1),
            _Stage2Chunk("high", "高分证据", 0.95),
        ),
        history_messages=tuple(
            {"role": "user" if index % 2 == 0 else "assistant", "content": "h" * 80}
            for index in range(12)
        ),
        summary="第一句。" + "第二句。" * 80,
    )

    result = manager.trim_context(context, budget)
    snapshot = {
        "actions": [action.component for action in result.actions],
        "debug_notes": list(result.context.debug_notes),
        "system_prompt": result.context.system_prompt,
        "current_question": result.context.current_question,
        "kept_chunk_ids": [chunk.chunk_id for chunk in result.context.rag_chunks],
    }

    assert snapshot == {
        "actions": ["debug", "tool_result", "rag_chunk", "history", "summary"],
        "debug_notes": [],
        "system_prompt": "系统安全约束必须保留。",
        "current_question": "当前用户问题必须保留。",
        "kept_chunk_ids": ["high"],
    }


def test_stage2_conversation_memory_failure_supports_fail_open_and_strict_modes(
    tmp_path: Path,
    fake_request_context: RequestContext,
    fake_memory_saver: Callable[..., object],
) -> None:
    """MemorySaver 读取异常默认 fail-open，严格模式才包装为安全 AppError。"""

    trace_path = tmp_path / "trace.jsonl"
    trace_logger = TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)
    fail_open_manager = ConversationManager(
        fake_memory_saver(raise_on_read=True),
        trace_logger=trace_logger,
        fail_open_on_read_error=True,
    )

    context = fail_open_manager.load_context("session-test", budget=None, ctx=fake_request_context)

    assert context.recent_messages == ()
    assert context.history_metadata["load_failed"] is True
    with pytest.raises(InternalAppError) as exc_info:
        ConversationManager(
            fake_memory_saver(raise_on_read=True),
            trace_logger=trace_logger,
            fail_open_on_read_error=False,
        ).load_context("session-test", budget=None, ctx=fake_request_context)

    trace_text = trace_path.read_text(encoding="utf-8")
    assert exc_info.value.code == "INTERNAL_ERROR"
    assert "raw checkpoint read failure" not in exc_info.value.user_message
    assert "raw checkpoint read failure" not in trace_text
    assert "conversation.load" in trace_text


def test_stage2_summary_failure_keeps_recent_turns_and_checkpoint(
    fake_request_context: RequestContext,
    fake_memory_saver: Callable[..., object],
    fake_summarizer: Callable[..., object],
) -> None:
    """摘要失败只能降级为最近轮次，不能删除或改写原始 MemorySaver checkpoint。"""

    messages: list[dict[str, str]] = []
    for index in range(10):
        messages.append({"role": "user", "content": f"用户问题 {index}", "timestamp": f"u{index}"})
        messages.append(
            {"role": "assistant", "content": f"助手回答 {index}", "timestamp": f"a{index}"}
        )
    memory = fake_memory_saver(
        checkpoint={"channel_values": {"messages": messages}},
        tuple_mode="checkpoint",
    )
    summarizer = fake_summarizer(mode="fail")
    manager = ConversationManager(
        memory,
        summarizer=summarizer,
        recent_turns=2,
        summary_enabled=True,
    )

    context = manager.load_context("session-test", budget=512, ctx=fake_request_context)

    assert context.summary is None
    assert [turn.content for turn in context.recent_messages] == [
        "用户问题 8",
        "助手回答 8",
        "用户问题 9",
        "助手回答 9",
    ]
    assert memory.deleted_threads == []
    assert summarizer.calls == 1


@pytest.mark.asyncio
async def test_stage2_orchestrator_success_records_usage_and_keeps_legacy_fields(
    fake_request_context: RequestContext,
    fake_usage_recorder: Callable[..., object],
) -> None:
    """成功路径仍走旧 service，编排层只补 usage/conversation/fallback 兼容字段。"""

    rag_service = _Stage2RagService()
    usage_recorder = fake_usage_recorder()
    conversation = _ConversationBoundary()
    orchestrator = AgentOrchestrator(
        rag_service=rag_service,
        aiops_service=_Stage2AIOpsService(),
        fallback_manager=FallbackManager(
            trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False)
        ),
        token_budget_manager=usage_recorder,
        conversation_manager=conversation,
        trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False),
        enabled=True,
    )

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )
    api_data = result.to_api_data()

    assert rag_service.query_calls == [("CPU 怎么排查", "session-test")]
    assert conversation.loaded == [("session-test", "trc_test")]
    assert usage_recorder.usage_records == [
        {"model": "qwen-test", "input_tokens": 8, "output_tokens": 13, "estimated": True}
    ]
    assert api_data["success"] is True
    assert api_data["answer"] == "stage2 answer"
    assert api_data["errorMessage"] is None
    assert api_data["fallback_used"] is False
    assert api_data["citations"] == []


@pytest.mark.asyncio
async def test_stage2_orchestrator_milvus_failure_uses_safe_fallback(
    fake_request_context: RequestContext,
    fake_usage_recorder: Callable[..., object],
) -> None:
    """Milvus/向量库失败可 fallback，但不得暴露内部 URL、密钥或原始异常全文。"""

    orchestrator = AgentOrchestrator(
        rag_service=_Stage2RagService(mode="milvus_fail"),
        aiops_service=_Stage2AIOpsService(),
        fallback_manager=FallbackManager(
            trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False)
        ),
        token_budget_manager=fake_usage_recorder(),
        conversation_manager=_ConversationBoundary(),
        trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False),
        enabled=True,
    )

    result = await orchestrator.run_chat(
        question="知识库是否可用",
        session_id="session-test",
        ctx=fake_request_context,
    )
    serialized = json.dumps(result.to_api_data(), ensure_ascii=False)

    assert result.fallback_used is True
    assert result.reason_code == "VECTOR_STORE_UNAVAILABLE"
    assert "milvus://internal" not in serialized
    assert "sk-secret" not in serialized


@pytest.mark.asyncio
async def test_stage2_orchestrator_rollback_bypasses_new_boundaries(
    fake_request_context: RequestContext,
    fake_usage_recorder: Callable[..., object],
) -> None:
    """关闭 orchestrator_enabled 等价回旧路径：不触发 token、conversation、fallback 边界。"""

    usage_recorder = fake_usage_recorder()
    conversation = _ConversationBoundary()
    rag_service = _Stage2RagService(answer="legacy direct answer")
    orchestrator = AgentOrchestrator(
        rag_service=rag_service,
        aiops_service=_Stage2AIOpsService(),
        fallback_manager=FallbackManager(
            trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False)
        ),
        token_budget_manager=usage_recorder,
        conversation_manager=conversation,
        trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False),
        enabled=False,
    )

    result = await orchestrator.run_chat(
        question="回滚路径",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "legacy direct answer"
    assert result.fallback_used is False
    assert usage_recorder.allocate_calls == []
    assert usage_recorder.usage_records == []
    assert conversation.loaded == []


@pytest.mark.asyncio
async def test_stage2_replanner_response_failure_returns_standard_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final report generation failures must not masquerade as successful reports."""

    replanner_module = importlib.import_module("app.agent.aiops.replanner")

    class _Prompt:
        def __or__(self, other: object) -> object:
            return other

    class _FailingChain:
        async def ainvoke(
            self, payload: object, config: dict[str, object] | None = None
        ) -> object:
            # replanner 现在传入 usage 采集 callbacks；fake 容忍 config 参数。
            _ = payload, config
            raise RuntimeError("api_key=sk-secret http://internal.local")

    class _FailingLLM:
        def with_structured_output(self, schema: object) -> _FailingChain:
            _ = schema
            return _FailingChain()

    monkeypatch.setattr(replanner_module, "response_prompt", _Prompt(), raising=False)

    result = await replanner_module._generate_response(
        {
            "input": "diagnose alert",
            "plan": [],
            "past_steps": [("query logs", "raw secret token from http://internal.local")],
            "response": "",
        },
        _FailingLLM(),
    )

    serialized = json.dumps(result, ensure_ascii=False)

    assert "response" not in result
    assert result["error_event"]["type"] == "error"
    assert result["error_event"]["error"]["code"] == "LLM_PROVIDER_ERROR"
    assert "sk-secret" not in serialized
    assert "http://internal.local" not in serialized
