from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.config import config
from app.core.errors import RagEmptyResultError
from app.core.request_context import RequestContext
from app.rag.models import NoAnswerDecision, RagContext
from app.rag.retriever import RetrievalResult
from app.services.rag_agent_service import RagAgentService


API_CITATION = {
    "citation_id": "C1",
    "doc_id": "doc_cpu",
    "chunk_id": "doc_cpu#000001",
    "source_path": "aiops-docs/cpu.md",
    "file_name": "cpu.md",
    "score": 0.9,
    "content_preview": "CPU runbook",
}


def test_stage3b_rollback_switches_are_configured() -> None:
    assert isinstance(config.context_builder_enabled, bool)
    assert isinstance(config.citations_enabled, bool)


@pytest.mark.asyncio
async def test_query_with_citations_runs_stage3b_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    _enable_stage3b(monkeypatch)
    service = _build_service()

    result = await service.query_with_citations(
        "CPU high usage",
        "session-test",
        SimpleNamespace(summary=None, recent_messages=()),
        fake_request_context,
    )

    assert result.answer == "pipeline answer"
    assert list(result.citations) == [API_CITATION]
    assert service.rag_retriever.calls == [("CPU high usage", fake_request_context)]
    assert service.context_builder.calls[0][0] == "CPU high usage"
    assert any("RAG context" in message.content for message in service.model.messages)
    assert any("packed context" in message.content for message in service.model.messages)
    assert any("引用要求" in message.content for message in service.model.messages)
    assert service.citation_builder.sanitize_calls == [("pipeline answer", ("C1",))]


@pytest.mark.asyncio
async def test_query_stream_with_context_emits_stage3b_done_citations(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    _enable_stage3b(monkeypatch)
    service = _build_service()

    chunks = [
        chunk
        async for chunk in service.query_stream_with_context(
            "CPU high usage",
            "session-test",
            SimpleNamespace(summary=None, recent_messages=()),
            fake_request_context,
        )
    ]

    assert chunks == [
        {"type": "content", "data": "pipeline answer"},
        {
            "type": "complete",
            "data": {
                "answer": "pipeline answer",
                "citations": [API_CITATION],
            },
        },
    ]


@pytest.mark.asyncio
async def test_query_with_citations_carries_real_usage(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """模型响应带 usage_metadata 时，结果要携带真实 usage 供编排层记录。"""

    _enable_stage3b(monkeypatch)
    service = _build_service()
    service.model = _UsageReportingModel()

    result = await service.query_with_citations(
        "CPU high usage",
        "session-test",
        SimpleNamespace(summary=None, recent_messages=()),
        fake_request_context,
    )

    assert result.usage is not None
    assert result.usage.input_tokens == 2100
    assert result.usage.output_tokens == 90


@pytest.mark.asyncio
async def test_query_with_citations_raises_rag_empty_for_no_answer(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    _enable_stage3b(monkeypatch)
    decision = NoAnswerDecision(
        should_answer=False,
        reason_code="RAG_EMPTY_RESULT",
        safe_message="no evidence",
        evidence_count=0,
    )
    service = _build_service(
        retrieval=RetrievalResult(
            chunks=[],
            empty_reason="RAG_EMPTY_RESULT",
            no_answer_decision=decision,
        )
    )

    with pytest.raises(RagEmptyResultError):
        await service.query_with_citations(
            "unknown alert",
            "session-test",
            None,
            fake_request_context,
        )


def _enable_stage3b(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "new_rag_retriever_enabled", True, raising=False)
    monkeypatch.setattr(config, "context_builder_enabled", True, raising=False)
    monkeypatch.setattr(config, "citations_enabled", True, raising=False)


def _build_service(
    retrieval: RetrievalResult | None = None,
) -> RagAgentService:
    service = RagAgentService.__new__(RagAgentService)
    service.model_name = "qwen-test"
    service.system_prompt = "system prompt"
    service.model = _FakeModel()
    service.rag_retriever = _FakeRetriever(
        retrieval
        or RetrievalResult(
            chunks=[],
            candidate_count=1,
            final_count=1,
            min_score=0.35,
        )
    )
    service.context_builder = _FakeContextBuilder()
    service.citation_builder = _FakeCitationBuilder()
    service.token_budget_manager = _FakeTokenBudgetManager()
    return service


@dataclass
class _FakeRetriever:
    result: RetrievalResult

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, RequestContext | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        ctx: RequestContext | None = None,
        budget: object | None = None,
    ) -> RetrievalResult:
        _ = budget
        self.calls.append((query, ctx))
        return self.result


class _FakeContextBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object, RequestContext | None]] = []

    def build(
        self,
        query: str,
        chunks: object,
        budget: object,
        *,
        ctx: RequestContext | None = None,
    ) -> RagContext:
        self.calls.append((query, chunks, budget, ctx))
        return RagContext(context_text="packed context")


class _FakeCitationBuilder:
    def __init__(self) -> None:
        self.sanitize_calls: list[tuple[str, tuple[str, ...]]] = []

    def build(
        self,
        context: RagContext,
        answer: str,
        *,
        ctx: RequestContext | None = None,
    ) -> list[object]:
        _ = context, answer, ctx
        return [SimpleNamespace(citation_id="C1")]

    def to_api_schema(self, citations: list[object]) -> list[dict[str, object]]:
        _ = citations
        return [API_CITATION]

    def sanitize_answer_anchors(
        self,
        answer: str,
        valid_citation_ids: object,
        *,
        ctx: RequestContext | None = None,
    ) -> str:
        _ = ctx
        self.sanitize_calls.append((answer, tuple(valid_citation_ids)))
        return answer


class _FakeTokenBudgetManager:
    def allocate(
        self,
        scenario: str,
        model: str,
        ctx: RequestContext | None = None,
        *,
        current_input: str | None = None,
        system_prompt: str | None = None,
    ) -> int:
        _ = scenario, model, ctx, current_input, system_prompt
        return 1800


class _FakeModel:
    def __init__(self) -> None:
        self.messages: list[object] = []

    async def ainvoke(self, messages: list[object]) -> object:
        self.messages = messages
        return SimpleNamespace(content="pipeline answer")


class _UsageReportingModel(_FakeModel):
    """模拟 DashScope 返回 usage_metadata 的响应对象。"""

    async def ainvoke(self, messages: list[object]) -> object:
        await super().ainvoke(messages)
        return SimpleNamespace(
            content="pipeline answer",
            usage_metadata={"input_tokens": 2100, "output_tokens": 90, "total_tokens": 2190},
        )
