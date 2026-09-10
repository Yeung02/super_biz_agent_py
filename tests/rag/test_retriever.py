"""ISSUE-023/024/027 RagRetriever 检索入口测试。

这些测试只验证检索入口、向量检索 adapter、去重和错误映射，不接入旧
`retrieve_knowledge` 工具。ISSUE-024 起补充分数归一化、min_score 过滤和
no-answer 前置判断。ISSUE-027 补充默认关闭的 reranker 插口。阶段 3C 补充
LLM 查询改写、多路召回与 RRF 融合，仍然不连接真实 Milvus、DashScope、MCP、
外部 reranker 或网络。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Protocol

import pytest

from app.core.errors import VectorStoreUnavailableError
from app.core.request_context import RequestContext
from app.rag.models import RetrievedChunk
from app.rag.query_rewriter import QueryRewriteResult
from app.rag.retriever import RagRetriever


class TraceSpanLike(Protocol):
    """测试 trace span 的最小协议。"""

    name: str


@dataclass(frozen=True)
class FakeSearchResult:
    """向量检索结果 fake，字段贴近 `VectorSearchService.SearchResult`。"""

    id: str
    content: str
    score: float
    metadata: dict[str, object]
    metric: str = "L2"
    normalized_score: float | None = None


class RecordingVectorSearchService:
    """记录 RagRetriever 对向量检索服务的调用，避免连接真实 Milvus。"""

    def __init__(self, results: list[FakeSearchResult] | None = None, *, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[FakeSearchResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": dict(filters or {})})
        if self.fail:
            raise RuntimeError("raw vector error password=secret http://internal.local")
        return list(self.results)


class RecordingTraceLogger:
    """记录 RagRetriever trace 调用，验证字段而不写 JSONL。"""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.starts: list[dict[str, object]] = []
        self.ends: list[dict[str, object]] = []

    def record_event(self, name: str, ctx: RequestContext, **fields: object) -> None:
        self.events.append({"name": name, "trace_id": ctx.trace_id, **fields})

    def start_span(self, name: str, ctx: RequestContext, **fields: object) -> TraceSpanLike:
        self.starts.append({"name": name, "trace_id": ctx.trace_id, **fields})
        return _TraceSpan(name=name)

    def end_span(self, span: TraceSpanLike, **fields: object) -> None:
        self.ends.append({"name": span.name, **fields})


@dataclass(frozen=True)
class _TraceSpan:
    name: str


class ReverseReranker:
    """测试用 reranker，显式改序以证明 retriever 在 final_k 前调用插口。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "trace_id": ctx.trace_id if ctx is not None else None,
            }
        )
        return list(reversed(chunks))


class FailingReranker:
    """测试用失败 reranker，验证失败不能污染用户可见结果或破坏检索排序。"""

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        _ = query, ctx, chunks
        raise RuntimeError("reranker token=secret failed at http://internal.local")


class FakeQueryRewriter:
    """测试用 query rewriter，返回预设改写主 query 与多路变体。"""

    def __init__(
        self,
        *,
        rewritten_query: str | None = None,
        variants: list[str] | None = None,
        error_code: str | None = None,
        fail: bool = False,
    ) -> None:
        self.rewritten_query = rewritten_query
        self.variants = list(variants or [])
        self.error_code = error_code
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def rewrite(
        self,
        *,
        query: str,
        variant_count: int,
        ctx: RequestContext | None = None,
    ) -> QueryRewriteResult:
        _ = ctx
        self.calls.append({"query": query, "variant_count": variant_count})
        if self.fail:
            raise RuntimeError("rewrite failed token=secret")
        return QueryRewriteResult(
            rewritten_query=self.rewritten_query,
            variants=list(self.variants),
            error_code=self.error_code,
        )


class PerQueryVectorSearchService:
    """按 query 返回不同结果的 fake 向量检索服务，用于验证多路召回。"""

    def __init__(self, results_by_query: dict[str, list[FakeSearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[dict[str, object]] = []

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[FakeSearchResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": dict(filters or {})})
        return list(self.results_by_query.get(query, []))


def test_noop_reranker_returns_original_order(
    fake_request_context: RequestContext,
) -> None:
    from app.rag.reranker import NoopReranker

    chunks = [
        RetrievedChunk.from_search_result(
            _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
            metric="L2",
            normalized_score=0.90,
        ),
        RetrievedChunk.from_search_result(
            _result("hit-2", "second", "doc_cpu#000001", "hash-b", 0.20),
            metric="L2",
            normalized_score=0.80,
        ),
    ]

    reranked = NoopReranker().rerank(
        query="CPU",
        chunks=chunks,
        ctx=fake_request_context,
    )

    assert reranked == chunks
    assert reranked is not chunks


def test_retriever_skips_disabled_reranker_and_keeps_vector_order(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    reranker = ReverseReranker()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
                _result("hit-2", "second", "doc_cpu#000001", "hash-b", 0.20),
                _result("hit-3", "third", "doc_cpu#000002", "hash-c", 0.30),
            ]
        ),
        trace_logger=trace_logger,
        reranker=reranker,
        reranker_enabled=False,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=3, final_k=2)

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "doc_cpu#000000",
        "doc_cpu#000001",
    ]
    assert reranker.calls == []
    assert trace_logger.events[-1]["name"] == "rag.reranker.skipped"
    assert trace_logger.events[-1]["reason"] == "disabled"


def test_retriever_applies_enabled_reranker_before_final_k(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    reranker = ReverseReranker()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
                _result("hit-2", "second", "doc_cpu#000001", "hash-b", 0.20),
                _result("hit-3", "third", "doc_cpu#000002", "hash-c", 0.30),
            ]
        ),
        trace_logger=trace_logger,
        reranker=reranker,
        reranker_enabled=True,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=3, final_k=2)

    assert reranker.calls == [
        {
            "query": "CPU",
            "chunk_ids": ["doc_cpu#000000", "doc_cpu#000001", "doc_cpu#000002"],
            "trace_id": "trc_test",
        }
    ]
    assert [chunk.chunk_id for chunk in result.chunks] == [
        "doc_cpu#000002",
        "doc_cpu#000001",
    ]
    reranker_starts = [
        event for event in trace_logger.starts if event["name"] == "rag.reranker"
    ]
    reranker_ends = [event for event in trace_logger.ends if event["name"] == "rag.reranker"]
    assert reranker_starts
    assert reranker_ends
    assert reranker_ends[-1]["status"] == "ok"
    assert reranker_ends[-1]["reranker_used"] is True


def test_retriever_falls_back_to_vector_order_when_reranker_fails(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
                _result("hit-2", "second", "doc_cpu#000001", "hash-b", 0.20),
            ]
        ),
        trace_logger=trace_logger,
        reranker=FailingReranker(),
        reranker_enabled=True,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=2, final_k=2)

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "doc_cpu#000000",
        "doc_cpu#000001",
    ]
    error_events = [event for event in trace_logger.events if event["name"] == "rag.reranker.error"]
    assert error_events
    assert error_events[-1]["fallback_to_vector_order"] is True
    assert "secret" not in str(error_events[-1])
    assert "internal.local" not in str(error_events[-1])


def test_retriever_returns_chunks_and_keeps_raw_score_semantics(
    fake_request_context: RequestContext,
) -> None:
    vector_search = RecordingVectorSearchService(
        [
            _result("hit-1", "CPU runbook A", "doc_cpu#000000", "hash-a", 0.10),
            _result("hit-2", "CPU runbook B", "doc_cpu#000001", "hash-b", 0.20),
            _result("hit-3", "CPU runbook C", "doc_cpu#000002", "hash-c", 0.30),
        ]
    )
    trace_logger = RecordingTraceLogger()
    retriever = RagRetriever(vector_search_service=vector_search, trace_logger=trace_logger)

    result = retriever.retrieve(
        "CPU 使用率过高",
        ctx=fake_request_context,
        candidate_k=5,
        final_k=2,
        filters={"tenant_id": "default"},
    )

    assert vector_search.calls == [
        {"query": "CPU 使用率过高", "top_k": 5, "filters": {"tenant_id": "default"}}
    ]
    assert result.empty is False
    assert result.empty_reason is None
    assert result.query_variants == ["CPU 使用率过高"]
    assert result.candidate_count == 3
    assert result.final_count == 2
    assert [chunk.chunk_id for chunk in result.chunks] == ["doc_cpu#000000", "doc_cpu#000001"]
    assert result.chunks[0].raw_score == 0.10
    assert result.chunks[0].metric == "L2"
    assert result.chunks[0].normalized_score == pytest.approx(1 / 1.10)
    assert trace_logger.starts[0]["name"] == "rag.retrieve"
    assert trace_logger.ends[-1]["candidate_count"] == 3
    assert trace_logger.ends[-1]["final_count"] == 2
    assert trace_logger.ends[-1]["min_score"] == pytest.approx(0.35)
    assert trace_logger.ends[-1]["dropped_low_score_count"] == 0


def test_retriever_returns_empty_result_without_throwing(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService([]),
        trace_logger=trace_logger,
    )

    result = retriever.retrieve("不存在的知识", ctx=fake_request_context, candidate_k=4, final_k=2)

    assert result.empty is True
    assert result.empty_reason == "RAG_EMPTY_RESULT"
    assert result.chunks == []
    assert result.candidate_count == 0
    assert result.final_count == 0
    assert result.no_answer_decision is not None
    assert result.no_answer_decision.should_answer is False
    assert result.no_answer_decision.reason_code == "RAG_EMPTY_RESULT"
    assert result.no_answer_decision.evidence_count == 0
    assert trace_logger.ends[-1]["empty_reason"] == "RAG_EMPTY_RESULT"
    assert trace_logger.ends[-1]["dropped_low_score_count"] == 0


def test_retriever_filters_chunks_below_min_score_before_final_k(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "high confidence", "doc_cpu#000000", "hash-a", 0.25),
                _result("hit-2", "low confidence", "doc_cpu#000001", "hash-b", 3.00),
                _result("hit-3", "also high", "doc_mem#000000", "hash-c", 0.50),
            ]
        ),
        trace_logger=trace_logger,
        default_min_score=0.50,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=6, final_k=5)

    assert [chunk.chunk_id for chunk in result.chunks] == ["doc_cpu#000000", "doc_mem#000000"]
    assert [chunk.normalized_score for chunk in result.chunks] == [
        pytest.approx(0.8),
        pytest.approx(2 / 3),
    ]
    assert result.candidate_count == 3
    assert result.final_count == 2
    assert result.dropped_low_score_count == 1
    assert result.empty_reason is None
    assert result.no_answer_decision is None
    assert trace_logger.ends[-1]["min_score"] == pytest.approx(0.50)
    assert trace_logger.ends[-1]["dropped_low_score_count"] == 1


def test_retriever_low_score_candidates_create_no_answer_decision(
    fake_request_context: RequestContext,
) -> None:
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "weak evidence", "doc_cpu#000000", "hash-a", 3.00),
                _result("hit-2", "weaker evidence", "doc_cpu#000001", "hash-b", 4.00),
            ]
        ),
        default_min_score=0.50,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=4, final_k=2)

    assert result.empty is True
    assert result.empty_reason == "RAG_LOW_SCORE"
    assert result.chunks == []
    assert result.candidate_count == 2
    assert result.final_count == 0
    assert result.dropped_low_score_count == 2
    assert result.no_answer_decision is not None
    assert result.no_answer_decision.should_answer is False
    assert result.no_answer_decision.reason_code == "RAG_LOW_SCORE"
    assert result.no_answer_decision.evidence_count == 0


def test_retriever_unknown_metric_uses_conservative_score_and_does_not_raise(
    fake_request_context: RequestContext,
) -> None:
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result(
                    "hit-1",
                    "unknown score semantics",
                    "doc_cpu#000000",
                    "hash-a",
                    99.0,
                    metric="mystery_metric",
                )
            ]
        ),
        default_min_score=0.35,
    )

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=1, final_k=1)

    assert result.empty is True
    assert result.empty_reason == "RAG_LOW_SCORE"
    assert result.dropped_low_score_count == 1
    assert result.no_answer_decision is not None
    assert result.no_answer_decision.reason_code == "RAG_LOW_SCORE"


def test_retriever_deduplicates_by_chunk_id_and_content_hash(
    fake_request_context: RequestContext,
) -> None:
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(
            [
                _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
                _result("hit-2", "same chunk id", "doc_cpu#000000", "hash-b", 0.20),
                _result("hit-3", "same content hash", "doc_disk#000000", "hash-a", 0.30),
                _result("hit-4", "unique", "doc_mem#000000", "hash-c", 0.40),
            ]
        )
    )

    result = retriever.retrieve("资源告警", ctx=fake_request_context, candidate_k=4, final_k=4)

    assert [chunk.chunk_id for chunk in result.chunks] == ["doc_cpu#000000", "doc_mem#000000"]
    assert result.candidate_count == 4
    assert result.final_count == 2


def test_retriever_uses_candidate_k_for_search_and_final_k_for_output(
    fake_request_context: RequestContext,
) -> None:
    vector_search = RecordingVectorSearchService(
        [
            _result("hit-1", "first", "doc_cpu#000000", "hash-a", 0.10),
            _result("hit-2", "second", "doc_cpu#000001", "hash-b", 0.20),
        ]
    )
    retriever = RagRetriever(vector_search_service=vector_search)

    result = retriever.retrieve("CPU", ctx=fake_request_context, candidate_k=8, final_k=1)

    assert vector_search.calls[0]["top_k"] == 8
    assert [chunk.chunk_id for chunk in result.chunks] == ["doc_cpu#000000"]
    assert result.candidate_count == 2
    assert result.final_count == 1


def test_retriever_maps_vector_errors_to_safe_app_error(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    retriever = RagRetriever(
        vector_search_service=RecordingVectorSearchService(fail=True),
        trace_logger=trace_logger,
    )

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        retriever.retrieve("CPU", ctx=fake_request_context)

    assert exc_info.value.code == "VECTOR_STORE_UNAVAILABLE"
    assert exc_info.value.user_message == "知识库暂时不可用。"
    assert "password" not in exc_info.value.user_message
    assert "internal.local" not in exc_info.value.user_message
    assert trace_logger.ends[-1]["status"] == "error"
    assert trace_logger.ends[-1]["error_code"] == "VECTOR_STORE_UNAVAILABLE"


def test_retriever_multi_query_expands_variants_and_searches_each(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    rewriter = FakeQueryRewriter(
        rewritten_query="CPU 使用率过高如何排查",
        variants=["CPU 利用率高 排查步骤", "服务器 CPU 告警处理"],
    )
    vector_search = RecordingVectorSearchService(
        [_result("hit-1", "CPU runbook", "doc_cpu#000000", "hash-a", 0.10)]
    )
    retriever = RagRetriever(
        vector_search_service=vector_search,
        trace_logger=trace_logger,
        query_rewriter=rewriter,
        query_rewrite_enabled=True,
        multi_query_count=2,
    )

    result = retriever.retrieve("CPU 使用率过高", ctx=fake_request_context, candidate_k=5, final_k=4)

    # 一次 rewriter 调用同时产出主 query 与变体；每路 variant 各触发一次向量检索
    assert rewriter.calls == [{"query": "CPU 使用率过高", "variant_count": 2}]
    assert [call["query"] for call in vector_search.calls] == [
        "CPU 使用率过高",
        "CPU 使用率过高如何排查",
        "CPU 利用率高 排查步骤",
        "服务器 CPU 告警处理",
    ]
    assert result.query_variants == [
        "CPU 使用率过高",
        "CPU 使用率过高如何排查",
        "CPU 利用率高 排查步骤",
        "服务器 CPU 告警处理",
    ]
    # 4 路各召回 1 条原始候选，去重后只剩 1 个 chunk
    assert result.candidate_count == 4
    assert result.final_count == 1
    rewrite_starts = [event for event in trace_logger.starts if event["name"] == "rag.query_rewrite"]
    rewrite_ends = [event for event in trace_logger.ends if event["name"] == "rag.query_rewrite"]
    assert rewrite_starts
    assert rewrite_ends[-1]["status"] == "ok"
    assert rewrite_ends[-1]["produced_variants"] == 2


def test_retriever_rrf_promotes_chunks_hit_by_multiple_variants(
    fake_request_context: RequestContext,
) -> None:
    # 变体一返回 [X, Y]，变体二返回 [Z, X]：X 被两路共同命中，融合后必须排第一；
    # Y（仅第一路 rank 1）应排在 Z（第二路 rank 0）之后。
    vector_search = PerQueryVectorSearchService(
        {
            "原始查询": [
                _result("hit-x", "X", "doc_x#000000", "hash-x", 0.10),
                _result("hit-y", "Y", "doc_y#000000", "hash-y", 0.20),
            ],
            "变体查询": [
                _result("hit-z", "Z", "doc_z#000000", "hash-z", 0.15),
                _result("hit-x-dup", "X", "doc_x#000000", "hash-x", 0.12),
            ],
        }
    )
    rewriter = FakeQueryRewriter(rewritten_query="变体查询")
    retriever = RagRetriever(
        vector_search_service=vector_search,
        query_rewriter=rewriter,
        query_rewrite_enabled=True,
        multi_query_count=0,
    )

    result = retriever.retrieve("原始查询", ctx=fake_request_context, candidate_k=5, final_k=3)

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "doc_x#000000",
        "doc_z#000000",
        "doc_y#000000",
    ]
    # 2 路各召回 2 条原始候选；X 跨路去重后只保留一份
    assert result.candidate_count == 4
    assert result.final_count == 3


def test_retriever_rrf_keeps_best_normalized_score_instance(
    fake_request_context: RequestContext,
) -> None:
    # 同一 chunk 在两路召回中分数不同：第一路 L2=3.00（归一化 0.25，低于阈值），
    # 第二路 L2=0.50（归一化 2/3）。融合必须保留分数最高实例，否则会被 min_score
    # 误杀，多路召回失去意义。
    vector_search = PerQueryVectorSearchService(
        {
            "原始查询": [_result("hit-weak", "weak hit", "doc_x#000000", "hash-x", 3.00)],
            "变体查询": [_result("hit-strong", "strong hit", "doc_x#000000", "hash-x", 0.50)],
        }
    )
    rewriter = FakeQueryRewriter(rewritten_query="变体查询")
    retriever = RagRetriever(
        vector_search_service=vector_search,
        query_rewriter=rewriter,
        query_rewrite_enabled=True,
        multi_query_count=0,
        default_min_score=0.50,
    )

    result = retriever.retrieve("原始查询", ctx=fake_request_context, candidate_k=5, final_k=2)

    assert result.final_count == 1
    assert result.chunks[0].chunk_id == "doc_x#000000"
    assert result.chunks[0].normalized_score == pytest.approx(2 / 3)
    assert result.dropped_low_score_count == 0


def test_retriever_rewrite_failure_falls_back_to_original_query(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    rewriter = FakeQueryRewriter(fail=True)
    vector_search = RecordingVectorSearchService(
        [_result("hit-1", "CPU runbook", "doc_cpu#000000", "hash-a", 0.10)]
    )
    retriever = RagRetriever(
        vector_search_service=vector_search,
        trace_logger=trace_logger,
        query_rewriter=rewriter,
        query_rewrite_enabled=True,
        multi_query_count=2,
    )

    result = retriever.retrieve("CPU 使用率过高", ctx=fake_request_context, candidate_k=5, final_k=2)

    # 改写失败 fail-open：只检索原始 query，结果照常返回
    assert [call["query"] for call in vector_search.calls] == ["CPU 使用率过高"]
    assert result.query_variants == ["CPU 使用率过高"]
    assert result.final_count == 1
    error_events = [
        event for event in trace_logger.events if event["name"] == "rag.query_rewrite.error"
    ]
    assert error_events
    assert error_events[-1]["fallback_to_original_query"] is True
    assert "secret" not in str(error_events[-1])


def test_retriever_skips_rewriter_when_disabled(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    rewriter = FakeQueryRewriter(
        rewritten_query="不应被使用的改写",
        variants=["不应被使用的变体"],
    )
    vector_search = RecordingVectorSearchService(
        [_result("hit-1", "CPU runbook", "doc_cpu#000000", "hash-a", 0.10)]
    )
    retriever = RagRetriever(
        vector_search_service=vector_search,
        trace_logger=trace_logger,
        query_rewriter=rewriter,
        query_rewrite_enabled=False,
        multi_query_count=2,
    )

    result = retriever.retrieve("CPU 使用率过高", ctx=fake_request_context, candidate_k=5, final_k=2)

    assert rewriter.calls == []
    assert [call["query"] for call in vector_search.calls] == ["CPU 使用率过高"]
    assert result.query_variants == ["CPU 使用率过高"]
    skipped_events = [
        event for event in trace_logger.events if event["name"] == "rag.query_rewrite.skipped"
    ]
    assert skipped_events
    assert skipped_events[-1]["reason"] == "disabled"


def test_retriever_default_rewriter_is_noop_and_keeps_single_query(
    fake_request_context: RequestContext,
) -> None:
    # 未注入 rewriter 的调用方（旧 API、evaluation runner）必须保持单 query 行为，
    # 不触网、不多路。
    vector_search = RecordingVectorSearchService(
        [_result("hit-1", "CPU runbook", "doc_cpu#000000", "hash-a", 0.10)]
    )
    retriever = RagRetriever(vector_search_service=vector_search)

    result = retriever.retrieve("CPU 使用率过高", ctx=fake_request_context, candidate_k=5, final_k=2)

    assert [call["query"] for call in vector_search.calls] == ["CPU 使用率过高"]
    assert result.query_variants == ["CPU 使用率过高"]
    assert result.candidate_count == 1


def test_vector_search_service_passes_candidate_k_and_safe_metadata_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_search_module = _reload_vector_search_service_with_fake_embedding(monkeypatch)
    collection = RecordingMilvusCollection()
    monkeypatch.setattr(
        vector_search_module,
        "vector_embedding_service",
        SimpleNamespace(embed_query=lambda query: [0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        vector_search_module.milvus_manager,
        "get_collection",
        lambda: collection,
    )
    service = vector_search_module.VectorSearchService()

    results = service.search_similar_documents(
        "CPU",
        top_k=7,
        filters={"tenant_id": "default", "chunk_index": [0, 1]},
    )

    assert collection.search_kwargs["limit"] == 7
    assert collection.search_kwargs["expr"] == (
        'metadata["tenant_id"] == "default" and metadata["chunk_index"] in [0, 1]'
    )
    assert len(results) == 1
    assert results[0].metric == "L2"
    assert results[0].to_dict()["raw_score"] == 0.42


def _reload_vector_search_service_with_fake_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """用 fake embedding 模块重载向量检索服务，避免导入真实 LangChain 依赖。"""

    embedding_module = ModuleType("app.services.vector_embedding_service")
    embedding_module.vector_embedding_service = SimpleNamespace(
        embed_query=lambda query: [0.1, 0.2, 0.3],
    )
    monkeypatch.setitem(sys.modules, "app.services.vector_embedding_service", embedding_module)
    sys.modules.pop("app.services.vector_search_service", None)
    return importlib.import_module("app.services.vector_search_service")


class RecordingMilvusCollection:
    """记录 PyMilvus search kwargs，避免测试连接真实 Milvus。"""

    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] = {}

    def search(self, **kwargs: object) -> list[list[object]]:
        self.search_kwargs = dict(kwargs)
        entity = {
            "id": "doc_cpu#000000",
            "content": "CPU evidence",
            "metadata": {
                "doc_id": "doc_cpu",
                "chunk_id": "doc_cpu#000000",
                "content_hash": "hash-a",
                "source_path": "aiops-docs/cpu.md",
                "file_name": "cpu.md",
                "chunk_index": 0,
            },
        }
        return [[SimpleNamespace(entity=entity, distance=0.42)]]


def _result(
    result_id: str,
    content: str,
    chunk_id: str,
    content_hash: str,
    score: float,
    *,
    metric: str = "L2",
) -> FakeSearchResult:
    """构造带稳定 RAG metadata 的 fake search result。"""

    source_name = chunk_id.split("#", maxsplit=1)[0].removeprefix("doc_")
    return FakeSearchResult(
        id=result_id,
        content=content,
        score=score,
        metric=metric,
        metadata={
            "doc_id": chunk_id.split("#", maxsplit=1)[0],
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "tenant_id": "default",
            "version": 1,
            "source_path": f"aiops-docs/{source_name}.md",
            "file_name": f"{source_name}.md",
            "chunk_index": int(chunk_id.rsplit("#", maxsplit=1)[1]),
        },
    )
