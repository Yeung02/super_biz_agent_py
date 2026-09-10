"""ISSUE-015 AgentOrchestrator 薄编排层测试。

这些用例只验证编排契约：验证后的请求进入旧 service、失败交给 FallbackManager、
trace 记录 start/end/error、以及关闭开关时 API 能回到旧 service 直连。测试不访问
真实 DashScope、Milvus、MCP 或网络，避免把当前 issue 变成集成测试。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from app.agent.intent import IntentLabel, IntentResult
from app.core.errors import LLMTimeoutError
from app.core.fallback import FallbackManager
from app.core.llm_usage import RawUsage
from app.core.request_context import RequestContext
from app.core.token_budget import TokenBudgetManager
from app.observability.tracing import TraceLogger

if TYPE_CHECKING:
    from pathlib import Path


def _orchestrator_module():
    """按需导入 orchestrator，让 RED 阶段表现为清晰的契约失败。"""

    try:
        return importlib.import_module("app.agent.orchestrator")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.agent.orchestrator must exist for ISSUE-015: {exc}")


@dataclass
class _FakeRagService:
    """只模拟旧 RagAgentService 的 query/query_stream 边界。"""

    mode: str = "success"
    answer: str = "orchestrated answer"

    def __post_init__(self) -> None:
        self.query_calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []
        self.model_name = "qwen-test"
        self.system_prompt = "system prompt"

    async def query(self, question: str, session_id: str) -> str:
        self.query_calls.append((question, session_id))
        if self.mode == "timeout":
            raise LLMTimeoutError()
        if self.mode == "unknown_error":
            raise RuntimeError("raw api_key=sk-secret http://internal.local")
        return self.answer

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        self.stream_calls.append((question, session_id))
        if self.mode == "timeout":
            raise LLMTimeoutError()
        yield {"type": "content", "data": "first chunk"}
        yield {"type": "complete"}


@dataclass
class _FakeAIOpsService:
    """只模拟旧 AIOpsService.diagnose 事件流。"""

    mode: str = "success"

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    async def diagnose(self, session_id: str) -> AsyncGenerator[dict[str, object], None]:
        self.calls.append(session_id)
        yield {"type": "plan", "plan": ["check alert"]}
        if self.mode == "agent_limit":
            yield {
                "type": "error",
                "error": {
                    "code": "AGENT_MAX_STEP_EXCEEDED",
                    "message": "should not leak as final text",
                    "retryable": False,
                },
            }
            return
        yield {"type": "complete", "response": "ok"}


class _FakeConversationManager:
    """记录是否被编排层调用，避免依赖真实 MemorySaver checkpoint。"""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []
        self.context = SimpleNamespace(summary=None, recent_messages=(), history_metadata={})

    def load_context(
        self,
        session_id: str,
        budget: object,
        ctx: RequestContext | None = None,
    ) -> object:
        _ = budget
        self.loaded.append((session_id, ctx.trace_id if ctx else ""))
        return self.context


class _ContextAwareRagService(_FakeRagService):
    """Fake service that exposes the Stage 2 controlled-history boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.context_calls: list[object] = []

    async def query_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> str:
        _ = ctx
        self.context_calls.append(conversation_context)
        return await self.query(question, session_id)


class _CitationAwareRagService(_ContextAwareRagService):
    """Fake service for the Stage 3B answer+citations boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.citation_calls: list[object] = []
        self.citation_payload = {
            "citation_id": "C1",
            "doc_id": "doc_cpu",
            "chunk_id": "doc_cpu#000001",
            "source_path": "aiops-docs/cpu.md",
            "file_name": "cpu.md",
            "score": 0.91,
            "content_preview": "CPU runbook",
        }

    async def query_with_citations(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> object:
        _ = ctx
        self.citation_calls.append(conversation_context)
        return SimpleNamespace(
            answer=f"citation answer for {question}/{session_id}",
            citations=(self.citation_payload,),
            usage=getattr(self, "usage_payload", None),
        )


class _FakeHttpRequest:
    """API handler 回滚测试用 request fake。"""

    def __init__(self, ctx: RequestContext) -> None:
        self.state = SimpleNamespace(ctx=ctx)

    async def is_disconnected(self) -> bool:
        return False


def _build_orchestrator(
    *,
    rag_service: _FakeRagService | None = None,
    aiops_service: _FakeAIOpsService | None = None,
    conversation_manager: _FakeConversationManager | None = None,
    trace_path: Path | None = None,
    intent_classifier: object | None = None,
):
    orchestrator_module = _orchestrator_module()
    trace_logger = TraceLogger(
        trace_jsonl_path=str(trace_path or "unused"),
        enabled=trace_path is not None,
    )
    extra_kwargs: dict[str, object] = {}
    if intent_classifier is not None:
        extra_kwargs["intent_classifier"] = intent_classifier
    return orchestrator_module.AgentOrchestrator(
        rag_service=rag_service or _FakeRagService(),
        aiops_service=aiops_service or _FakeAIOpsService(),
        fallback_manager=FallbackManager(trace_logger=trace_logger),
        token_budget_manager=TokenBudgetManager(
            model_context_windows={"qwen-test": 4096, "default": 4096},
            trace_logger=trace_logger,
        ),
        conversation_manager=conversation_manager or _FakeConversationManager(),
        trace_logger=trace_logger,
        enabled=True,
        **extra_kwargs,
    )


@pytest.mark.asyncio
async def test_run_chat_success_calls_legacy_service_and_records_trace(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """成功路径仍调用旧 RagAgentService.query，编排层只增加 trace/token/conversation 边界。"""

    rag_service = _FakeRagService()
    conversation_manager = _FakeConversationManager()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        conversation_manager=conversation_manager,
        trace_path=tmp_path / "trace.jsonl",
    )

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["name"] for event in events]

    assert result.answer == "orchestrated answer"
    assert result.fallback_used is False
    assert rag_service.query_calls == [("CPU 怎么排查", "session-test")]
    assert conversation_manager.loaded == [("session-test", "trc_test")]
    assert "orchestrator.start" in event_names
    assert "orchestrator.end" in event_names
    assert events[-1]["mode"] == "chat"
    assert events[-1]["fallback_used"] is False


@pytest.mark.asyncio
async def test_run_chat_passes_controlled_conversation_context_to_rag_service(
    fake_request_context: RequestContext,
) -> None:
    """Long-history control must reach the service boundary."""

    rag_service = _ContextAwareRagService()
    conversation_manager = _FakeConversationManager()
    conversation_manager.context = SimpleNamespace(
        summary="safe prior summary",
        recent_messages=(
            SimpleNamespace(role="user", content="recent question"),
            SimpleNamespace(role="assistant", content="recent answer"),
        ),
        history_metadata={"trimmed_count": 40},
    )
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        conversation_manager=conversation_manager,
    )

    await orchestrator.run_chat(
        question="CPU 鎬庝箞鎺掓煡",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert rag_service.context_calls == [conversation_manager.context]


@pytest.mark.asyncio
async def test_run_chat_returns_stage3b_citations_when_service_exposes_result(
    fake_request_context: RequestContext,
) -> None:
    """Stage 3B answer+citations results must reach `/api/chat` response data."""

    rag_service = _CitationAwareRagService()
    conversation_manager = _FakeConversationManager()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        conversation_manager=conversation_manager,
    )

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "citation answer for CPU 怎么排查/session-test"
    assert list(result.citations) == [rag_service.citation_payload]
    assert rag_service.query_calls == []
    assert rag_service.citation_calls == [conversation_manager.context]


@pytest.mark.asyncio
async def test_run_chat_records_real_usage_from_service_result(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """service 返回真实 usage 时，token.usage 记录 estimated=False 且 token 数为真实值。"""

    rag_service = _CitationAwareRagService()
    rag_service.usage_payload = RawUsage(input_tokens=4321, output_tokens=876)
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        trace_path=tmp_path / "trace.jsonl",
    )

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.usage == RawUsage(input_tokens=4321, output_tokens=876)
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    token_usage_events = [event for event in events if event["name"] == "token.usage"]
    assert len(token_usage_events) == 1
    usage_payload = token_usage_events[0]["usage"]
    assert usage_payload["input_tokens"] == 4321
    assert usage_payload["output_tokens"] == 876
    assert usage_payload["total_tokens"] == 5197
    assert usage_payload["estimated"] is False


@pytest.mark.asyncio
async def test_run_chat_falls_back_to_estimated_usage_without_service_usage(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """service 未返回 usage 时保持旧估算路径，token.usage 仍标记 estimated=True。"""

    rag_service = _CitationAwareRagService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        trace_path=tmp_path / "trace.jsonl",
    )

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.usage is None
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    token_usage_events = [event for event in events if event["name"] == "token.usage"]
    assert len(token_usage_events) == 1
    assert token_usage_events[0]["usage"]["estimated"] is True


class _FakeIntentClassifier:
    """恒定返回指定意图的 fake，绕过 LLM 分类。"""

    def __init__(self, label: IntentLabel) -> None:
        self._label = label

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = question, ctx
        return IntentResult(label=self._label, confidence=0.99, source="rule")


@pytest.mark.asyncio
async def test_run_chat_aiops_intent_skips_orchestrator_usage_recording(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """AIOPS 意图的 usage 由四节点记录真实值：编排层不再聚合记账，防止双计。"""

    rag_service = _FakeRagService()
    aiops_service = _FakeAIOpsService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        aiops_service=aiops_service,
        trace_path=tmp_path / "trace.jsonl",
        intent_classifier=_FakeIntentClassifier(IntentLabel.AIOPS),
    )

    result = await orchestrator.run_chat(
        question="诊断告警",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert aiops_service.calls == ["session-test"]
    assert result.answer
    assert result.usage is None
    assert rag_service.query_calls == []
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["name"] for event in events]
    assert "token.usage" not in event_names
    assert "orchestrator.usage" not in event_names


@pytest.mark.asyncio
async def test_run_chat_stream_aiops_intent_skips_usage_recording(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """AIOPS 意图的流式分支同样跳过编排层 usage 记账（节点级真实值兜底）。"""

    aiops_service = _FakeAIOpsService()
    orchestrator = _build_orchestrator(
        aiops_service=aiops_service,
        trace_path=tmp_path / "trace.jsonl",
        intent_classifier=_FakeIntentClassifier(IntentLabel.AIOPS),
    )

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="诊断告警",
            session_id="session-test",
            ctx=fake_request_context,
        )
    ]

    assert aiops_service.calls == ["session-test"]
    assert [chunk["type"] for chunk in chunks] == ["content", "complete"]
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["name"] for event in events]
    assert "token.usage" not in event_names
    assert "orchestrator.usage" not in event_names


@pytest.mark.asyncio
async def test_run_chat_timeout_returns_safe_fallback(
    fake_request_context: RequestContext,
) -> None:
    """可降级 AppError 由编排层交给 FallbackManager，输出不包含原始内部异常。"""

    orchestrator = _build_orchestrator(rag_service=_FakeRagService(mode="timeout"))

    result = await orchestrator.run_chat(
        question="CPU 怎么排查",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.fallback_used is True
    assert result.reason_code == "LLM_TIMEOUT"
    assert result.answer
    assert "sk-secret" not in result.answer
    assert "http://internal.local" not in result.answer


@pytest.mark.asyncio
async def test_run_chat_stream_converts_error_to_fallback_chunks(
    fake_request_context: RequestContext,
) -> None:
    """Chat SSE 失败路径由编排层输出 fallback 和 complete chunk，API 层只负责 SSE adapter。"""

    orchestrator = _build_orchestrator(rag_service=_FakeRagService(mode="timeout"))

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="CPU 怎么排查",
            session_id="session-test",
            ctx=fake_request_context,
        )
    ]

    assert [chunk["type"] for chunk in chunks] == ["fallback", "complete"]
    assert chunks[0]["data"]["fallback_used"] is True
    assert chunks[0]["data"]["reason_code"] == "LLM_TIMEOUT"
    assert chunks[1]["data"]["fallback_used"] is True
    assert chunks[1]["data"]["answer"]


@pytest.mark.asyncio
async def test_run_aiops_converts_error_event_to_partial_fallback(
    fake_request_context: RequestContext,
) -> None:
    """AIOps 旧 service 的 error event 要在编排层转成 fallback/done，而不是继续暴露内部错误文本。"""

    orchestrator = _build_orchestrator(
        aiops_service=_FakeAIOpsService(mode="agent_limit"),
    )

    events = [
        event
        async for event in orchestrator.run_aiops(
            session_id="session-test",
            ctx=fake_request_context,
        )
    ]

    assert [event["type"] for event in events] == ["plan", "fallback", "done"]
    assert events[1]["fallback_used"] is True
    assert events[1]["reason_code"] == "AGENT_MAX_STEP_EXCEEDED"
    assert events[2]["fallback_used"] is True
    assert "should not leak" not in json.dumps(events, ensure_ascii=False)


@pytest.mark.asyncio
async def test_chat_api_rollback_switch_uses_legacy_service_directly(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """关闭 orchestrator_enabled 时，API handler 必须回到旧 service 直连路径。"""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    async def _legacy_query(question: str, session_id: str) -> str:
        assert (question, session_id) == ("CPU 怎么排查", "session-test")
        return "legacy answer"

    class _UnexpectedOrchestrator:
        async def run_chat(self, **kwargs: object) -> object:
            _ = kwargs
            raise AssertionError("orchestrator must not be called when disabled")

    monkeypatch.setattr(chat_api.config, "orchestrator_enabled", False, raising=False)
    monkeypatch.setattr(chat_api.rag_agent_service, "query", _legacy_query)
    monkeypatch.setattr(chat_api, "agent_orchestrator", _UnexpectedOrchestrator(), raising=False)

    response = await chat_api.chat(
        ChatRequest(Id="session-test", Question="CPU 怎么排查"),
        _FakeHttpRequest(fake_request_context),
    )

    assert isinstance(response, dict)
    assert response["code"] == 200
    assert response["data"]["answer"] == "legacy answer"
