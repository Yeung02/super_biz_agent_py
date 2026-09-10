"""可控 RAG 检索入口。

ISSUE-023 先把“调用向量检索、收集 query variants、转换 RetrievedChunk、去重、
截断 final_k、记录 trace”收敛成独立入口。ISSUE-024 在此基础上接入 score
normalization、min_score 和 no-answer 前置判断。ISSUE-027 只加入默认关闭的
reranker 插口，失败回退原始向量排序。阶段 3C 接入 LLM 查询改写与多路召回：
`multi_query` 产出多路检索 query，逐路召回后按 RRF 融合排序；改写失败时 fail-open
回退原始 query，多路召回不改变单路时的既有排序行为。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import config
from app.core.errors import (
    AppError,
    JsonValue,
    RagMetadataInvalidError,
    VectorStoreUnavailableError,
)
from app.core.request_context import RequestContext
from app.rag.models import NoAnswerDecision, RetrievedChunk, SearchResultLike, normalize_score
from app.rag.query_rewriter import (
    NoopQueryRewriter,
    QueryRewriteResult,
    QueryRewriterLike,
)
from app.rag.reranker import NoopReranker, RerankerLike

RetrievalFilterPrimitive: TypeAlias = str | int | float | bool | None
RetrievalFilterValue: TypeAlias = RetrievalFilterPrimitive | list[RetrievalFilterPrimitive]
RetrievalFilters: TypeAlias = dict[str, RetrievalFilterValue]
_EMPTY_RESULT_MESSAGE = "知识库没有找到相关依据。"
_LOW_SCORE_MESSAGE = (
    "我没有在当前知识库中找到足够可靠的依据来回答这个问题，因此不能给出确定答案。"
    "可以补充相关文档后重新提问，或换一个更具体的问题。"
)


class VectorSearchServiceLike(Protocol):
    """RagRetriever 需要的向量检索最小协议。

    使用协议而不是直接绑定全局单例，是为了让测试和后续 evaluation runner 可以注入
    fake vector search，不依赖真实 Milvus、DashScope 或网络。
    """

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: Mapping[str, RetrievalFilterValue] | None = None,
    ) -> Sequence[SearchResultLike]:
        """按 query 检索候选结果。"""


class TraceLoggerLike(Protocol):
    """RagRetriever 使用的 trace 最小协议。"""

    def record_event(self, name: str, ctx: RequestContext, **fields: JsonValue) -> None:
        """记录 trace 事件。"""

    def start_span(self, name: str, ctx: RequestContext, **fields: JsonValue) -> object:
        """开始 trace span。"""

    def end_span(self, span: object, **fields: JsonValue) -> None:
        """结束 trace span。"""


class RetrievalQuery(BaseModel):
    """一次 RAG 检索请求的内部模型。

    `candidate_k` 控制向量库召回数量，`final_k` 控制输出给后续 ContextBuilder/
    evaluation 的数量；`min_score` 使用归一化分数过滤低置信 chunk。过滤在检索入口
    前置完成，是为了让低置信结果不能伪装成可用事实证据进入后续 prompt。
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="用户查询文本")
    candidate_k: int = Field(..., ge=1, description="向量检索候选数量")
    final_k: int = Field(..., ge=0, description="最终保留 chunk 数量")
    min_score: float = Field(..., ge=0.0, le=1.0, description="归一化分数最低阈值")
    filters: RetrievalFilters = Field(default_factory=dict, description="metadata 过滤条件")
    query_variants: list[str] = Field(default_factory=list, description="实际检索 query 列表")

    @field_validator("text")
    @classmethod
    def _strip_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query text cannot be empty")
        return stripped


class RetrievalResult(BaseModel):
    """RagRetriever 输出结果。

    `empty_reason` 用稳定错误码表达空检索原因，但这里不抛 `RagEmptyResultError`，
    因为空结果在 RAG 链路中是可预期业务状态，后续 Fallback/ContextBuilder 再决定
    是否拒答或降级。
    """

    model_config = ConfigDict(extra="forbid")

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    empty_reason: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    final_count: int = Field(default=0, ge=0)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dropped_low_score_count: int = Field(default=0, ge=0)
    score_warnings: list[str] = Field(default_factory=list)
    no_answer_decision: NoAnswerDecision | None = None
    filters: RetrievalFilters = Field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """返回本次检索是否没有可用 chunk。"""

        return not self.chunks

    def to_trace_fields(self) -> dict[str, JsonValue]:
        """生成安全 trace 字段，不包含完整 chunk 正文。"""

        return {
            "query_variants": list(self.query_variants),
            "candidate_count": self.candidate_count,
            "final_count": self.final_count,
            "min_score": self.min_score,
            "dropped_low_score_count": self.dropped_low_score_count,
            "score_warnings": list(self.score_warnings),
            "empty_reason": self.empty_reason,
            "filters": cast(JsonValue, dict(self.filters)),
        }


class RagRetriever:
    """可控 RAG 检索入口。

    该类只依赖一个向量检索协议和可选 TraceLogger。这样它能被 evaluation、后续
    knowledge_tool adapter 或单元测试复用，同时不改动当前 FastAPI API 和旧工具路径。
    """

    def __init__(
        self,
        *,
        vector_search_service: VectorSearchServiceLike | None = None,
        trace_logger: TraceLoggerLike | None = None,
        default_candidate_k: int | None = None,
        default_final_k: int | None = None,
        default_min_score: float | None = None,
        query_rewrite_enabled: bool | None = None,
        query_rewriter: QueryRewriterLike | None = None,
        multi_query_count: int | None = None,
        reranker: RerankerLike | None = None,
        reranker_enabled: bool | None = None,
    ) -> None:
        if vector_search_service is None:
            from app.services.vector_search_service import vector_search_service as default_service

            vector_search_service = default_service

        self.vector_search_service = vector_search_service
        self.trace_logger = trace_logger
        self.default_candidate_k = default_candidate_k or config.rag_candidate_k
        self.default_final_k = default_final_k or config.rag_final_k
        self.default_min_score = (
            default_min_score if default_min_score is not None else config.rag_min_score
        )
        self.query_rewrite_enabled = (
            config.rag_query_rewrite_enabled
            if query_rewrite_enabled is None
            else query_rewrite_enabled
        )
        # 阶段 3C：默认注入 Noop，未显式接入 LLM 的调用方（旧 API、evaluation、单测）
        # 保持旧单 query 行为、不触网；生产链路显式注入 LlmQueryRewriter 后改写与
        # 多路召回才真正生效，失败在 retriever 内 fail-open 回退原始 query。
        self.query_rewriter = query_rewriter or NoopQueryRewriter()
        self.multi_query_count = max(
            0,
            int(
                multi_query_count
                if multi_query_count is not None
                else config.rag_multi_query_count
            ),
        )
        # reranker 是阶段 3B 的可选插口，默认使用 Noop 并由配置关闭。这样当前线上/
        # 测试行为仍保持向量检索排序；未来真实 reranker 可注入，但失败必须回退。
        self.reranker = reranker or NoopReranker()
        self.reranker_enabled = (
            config.reranker_enabled if reranker_enabled is None else reranker_enabled
        )

    def retrieve(
        self,
        query: str | RetrievalQuery,
        *,
        ctx: RequestContext | None = None,
        budget: object | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
        min_score: float | None = None,
        filters: Mapping[str, RetrievalFilterValue] | None = None,
    ) -> RetrievalResult:
        """执行一次可控 RAG 检索。

        `budget` 当前只作为阶段 2/3B 的接口占位传入，不在 ISSUE-023 消费；这样后续
        ContextBuilder 可以无破坏地接上 token 裁剪，而当前旧 API 和旧工具完全不受影响。
        """

        _ = budget
        retrieval_query = self._coerce_query(
            query,
            candidate_k=candidate_k,
            final_k=final_k,
            min_score=min_score,
            filters=filters,
        )
        span = self._start_trace(ctx, retrieval_query)
        try:
            variants = self.multi_query(retrieval_query, ctx=ctx)
            variant_raw_results = self._search_variants(variants, retrieval_query)
        except AppError as exc:
            self._end_trace_error(span, exc)
            raise
        except Exception as exc:
            error = VectorStoreUnavailableError(
                internal_message="RagRetriever vector search failed",
            )
            self._end_trace_error(span, error)
            raise error from exc

        chunks, score_warnings = self._fuse_variant_results(variant_raw_results)
        threshold_chunks, dropped_low_score_count = self.apply_threshold(
            chunks,
            min_score=retrieval_query.min_score,
        )
        reranked_chunks = self._rerank_candidates(
            retrieval_query,
            threshold_chunks,
            ctx=ctx,
        )
        final_chunks = reranked_chunks[: retrieval_query.final_k]
        candidate_count = sum(
            len(variant_results) for variant_results in variant_raw_results
        )
        empty_reason = _empty_reason(
            candidate_count=candidate_count,
            surviving_count=len(threshold_chunks),
            final_count=len(final_chunks),
        )
        no_answer_decision = _build_no_answer_decision(
            reason_code=empty_reason,
            min_score=retrieval_query.min_score,
            dropped_low_score_count=dropped_low_score_count,
        )
        result = RetrievalResult(
            chunks=final_chunks,
            query_variants=variants,
            empty_reason=empty_reason,
            candidate_count=candidate_count,
            final_count=len(final_chunks),
            min_score=retrieval_query.min_score,
            dropped_low_score_count=dropped_low_score_count,
            score_warnings=score_warnings,
            no_answer_decision=no_answer_decision,
            filters=dict(retrieval_query.filters),
        )
        self._end_trace_success(span, result)
        return result

    def rewrite_query(self, query: RetrievalQuery, *, ctx: RequestContext | None = None) -> str:
        """返回 rewrite 后 query；改写关闭或失败时回退原始 query。

        该方法只产出改写主 query（variant_count=0），供外部单独调用；`retrieve` 链路
        走 `multi_query`，一次 LLM 调用同时拿改写主 query 与多路变体。
        """

        rewrite_result = self._rewrite_with_trace(query.text, ctx=ctx, variant_count=0)
        return rewrite_result.rewritten_query or query.text

    def multi_query(
        self,
        query: RetrievalQuery,
        *,
        ctx: RequestContext | None = None,
    ) -> list[str]:
        """生成实际检索 query variants（原始 + 改写主 query + 多路变体，去重）。

        改写关闭或失败时只保留原始 query，维持旧单路召回行为。额外变体数量由
        `multi_query_count` 控制；调用方通过 `query.query_variants` 显式传入的变体
        仍会合并参与多路召回。
        """

        variant_count = self.multi_query_count if self.query_rewrite_enabled else 0
        rewrite_result = self._rewrite_with_trace(
            query.text,
            ctx=ctx,
            variant_count=variant_count,
        )
        candidates = [query.text]
        if rewrite_result.rewritten_query:
            candidates.append(rewrite_result.rewritten_query)
        candidates.extend(rewrite_result.variants)
        candidates.extend(query.query_variants)

        variants: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            variants.append(normalized)
        return variants or [query.text]

    def _rewrite_with_trace(
        self,
        query_text: str,
        *,
        ctx: RequestContext | None,
        variant_count: int,
    ) -> QueryRewriteResult:
        """调用改写器并记录 trace；任何异常都收敛为 fail-open 结果。"""

        if not self.query_rewrite_enabled:
            self._record_rewrite_skipped(ctx, reason="disabled", variant_count=variant_count)
            return QueryRewriteResult()

        span = self._start_rewrite_trace(ctx, variant_count=variant_count)
        try:
            result = self.query_rewriter.rewrite(
                query=query_text,
                variant_count=variant_count,
                ctx=ctx,
            )
        except Exception as exc:
            self._record_rewrite_error(ctx, span, exc, variant_count=variant_count)
            return QueryRewriteResult(error_code="REWRITER_FAILED")
        if result.error_code is not None:
            self._end_rewrite_trace_error(
                ctx,
                span,
                error_code=result.error_code,
                variant_count=variant_count,
            )
            return result
        self._end_rewrite_trace_success(ctx, span, result, variant_count=variant_count)
        return result

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, RetrievalFilterValue] | None = None,
    ) -> Sequence[SearchResultLike]:
        """调用向量检索服务。

        这里保留薄 adapter，是为了隔离 LangChain/Milvus 返回形态差异；RagRetriever
        后续只处理 SearchResultLike，不需要关心真实向量库对象。
        """

        return self.vector_search_service.search_similar_documents(
            query,
            top_k=top_k,
            filters=filters,
        )

    def deduplicate(self, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """按 chunk_id 或 content_hash 去重，并保留向量检索原始排序。

        向量库返回顺序代表当前基础 ranking。去重阶段先保留第一次出现的候选，
        reranker 插口在后续步骤可选改序；这样重复证据不会进入重排或 context
        packing，避免同一 chunk 被当作多条事实依据。
        """

        seen_keys: set[str] = set()
        deduplicated: list[RetrievedChunk] = []
        for chunk in chunks:
            candidate_keys = {f"chunk:{chunk.chunk_id}", f"hash:{chunk.content_hash}"}
            if candidate_keys & seen_keys:
                continue
            seen_keys.update(candidate_keys)
            deduplicated.append(chunk)
        return deduplicated

    def apply_threshold(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        min_score: float,
    ) -> tuple[list[RetrievedChunk], int]:
        """按归一化分数过滤低置信 chunk。

        阈值判断放在 `final_k` 截断之前，是为了避免“前几个候选分数很低但数量满足
        final_k”时仍被当作证据。缺失归一化分数的 chunk 也按低分丢弃，因为它无法
        证明自己的相关性，不能进入后续 ContextBuilder 或 LLM prompt。
        """

        kept_chunks: list[RetrievedChunk] = []
        dropped_count = 0
        for chunk in chunks:
            score = chunk.normalized_score
            if score is None or score < min_score:
                dropped_count += 1
                continue
            kept_chunks.append(chunk)
        return kept_chunks, dropped_count

    def _coerce_query(
        self,
        query: str | RetrievalQuery,
        *,
        candidate_k: int | None,
        final_k: int | None,
        min_score: float | None,
        filters: Mapping[str, RetrievalFilterValue] | None,
    ) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query.model_copy(
                update={
                    "candidate_k": candidate_k or query.candidate_k,
                    "final_k": final_k if final_k is not None else query.final_k,
                    "min_score": (
                        min_score if min_score is not None else query.min_score
                    ),
                    "filters": dict(filters) if filters is not None else dict(query.filters),
                }
            )
        return RetrievalQuery(
            text=query,
            candidate_k=candidate_k or self.default_candidate_k,
            final_k=final_k if final_k is not None else self.default_final_k,
            min_score=min_score if min_score is not None else self.default_min_score,
            filters=dict(filters or {}),
        )

    def _search_variants(
        self,
        variants: Sequence[str],
        query: RetrievalQuery,
    ) -> list[list[SearchResultLike]]:
        """逐个 variant 调用向量检索，保留每一路的独立排序供 RRF 融合。"""

        variant_results: list[list[SearchResultLike]] = []
        for variant in variants:
            variant_results.append(
                list(
                    self.search(
                        variant,
                        top_k=query.candidate_k,
                        filters=query.filters,
                    )
                )
            )
        return variant_results

    def _fuse_variant_results(
        self,
        variant_raw_results: Sequence[Sequence[SearchResultLike]],
    ) -> tuple[list[RetrievedChunk], list[str]]:
        """按 RRF 融合多路召回结果，转换并去重为 RetrievedChunk 列表。

        单路召回时 RRF 顺序与向量检索原始排序等价，保持旧行为不变；多路召回时多个
        query 共同命中的 chunk 获得更高融合分，避免 final_k 截断后只剩第一路结果。
        同一 chunk 在不同 variant 下分数不同，保留归一化分数最高的实例，让
        min_score 阈值按最有利的证据判断。
        """

        rrf_scores: dict[str, float] = {}
        first_seen_order: dict[str, int] = {}
        best_chunk_by_key: dict[str, RetrievedChunk] = {}
        score_warnings: list[str] = []
        order_counter = 0

        for variant_results in variant_raw_results:
            for rank, result in enumerate(variant_results):
                metric = _result_metric(result)
                normalized_score, warning = _normalized_score_for_result(result, metric=metric)
                if warning is not None:
                    score_warnings.append(warning)
                chunk = RetrievedChunk.from_search_result(
                    result,
                    metric=metric,
                    normalized_score=normalized_score,
                )
                key = _chunk_fusion_key(chunk)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if key not in first_seen_order:
                    first_seen_order[key] = order_counter
                    order_counter += 1
                existing = best_chunk_by_key.get(key)
                if (
                    existing is None
                    or _normalized_score_value(chunk) > _normalized_score_value(existing)
                ):
                    best_chunk_by_key[key] = chunk

        fused_keys = sorted(
            rrf_scores,
            key=lambda key: (-rrf_scores[key], first_seen_order[key]),
        )
        fused_chunks = [best_chunk_by_key[key] for key in fused_keys]
        return self.deduplicate(fused_chunks), score_warnings

    def _rerank_candidates(
        self,
        query: RetrievalQuery,
        chunks: Sequence[RetrievedChunk],
        *,
        ctx: RequestContext | None,
    ) -> list[RetrievedChunk]:
        original_chunks = list(chunks)
        if not self.reranker_enabled:
            self._record_reranker_skipped(
                ctx,
                reason="disabled",
                candidate_count=len(original_chunks),
            )
            return original_chunks
        if not original_chunks:
            self._record_reranker_skipped(
                ctx,
                reason="empty_candidates",
                candidate_count=0,
            )
            return original_chunks

        span = self._start_reranker_trace(ctx, candidate_count=len(original_chunks))
        try:
            reranked_chunks = self.reranker.rerank(
                query=query.text,
                chunks=list(original_chunks),
                ctx=ctx,
            )
        except Exception as exc:
            self._record_reranker_error(ctx, span, exc, candidate_count=len(original_chunks))
            return original_chunks

        if not _same_chunk_identity(original_chunks, reranked_chunks):
            # reranker 只负责排序，不能静默过滤或注入新证据。发现身份集合变化时回退，
            # 既保护原始向量检索结果，也避免后续 ContextBuilder 引用无法追踪的 chunk。
            self._record_reranker_invalid_result(
                ctx,
                span,
                candidate_count=len(original_chunks),
                returned_count=len(reranked_chunks),
            )
            return original_chunks

        self._end_reranker_trace_success(
            span,
            candidate_count=len(original_chunks),
            returned_count=len(reranked_chunks),
        )
        return list(reranked_chunks)

    def _start_trace(
        self,
        ctx: RequestContext | None,
        query: RetrievalQuery,
    ) -> object | None:
        if self.trace_logger is None or ctx is None:
            return None
        return self.trace_logger.start_span(
            "rag.retrieve",
            ctx,
            candidate_k=query.candidate_k,
            final_k=query.final_k,
            min_score=query.min_score,
            filters=dict(query.filters),
        )

    def _end_trace_success(self, span: object | None, result: RetrievalResult) -> None:
        if self.trace_logger is None or span is None:
            return
        self.trace_logger.end_span(span, status="ok", **result.to_trace_fields())

    def _end_trace_error(self, span: object | None, error: AppError) -> None:
        if self.trace_logger is None or span is None:
            return
        self.trace_logger.end_span(
            span,
            status="error",
            error_code=error.code,
            candidate_count=0,
            final_count=0,
        )

    def _record_reranker_skipped(
        self,
        ctx: RequestContext | None,
        *,
        reason: str,
        candidate_count: int,
    ) -> None:
        if self.trace_logger is None or ctx is None:
            return
        self.trace_logger.record_event(
            "rag.reranker.skipped",
            ctx,
            reason=reason,
            candidate_count=candidate_count,
            reranker_enabled=self.reranker_enabled,
        )

    def _record_rewrite_skipped(
        self,
        ctx: RequestContext | None,
        *,
        reason: str,
        variant_count: int,
    ) -> None:
        if self.trace_logger is None or ctx is None:
            return
        self.trace_logger.record_event(
            "rag.query_rewrite.skipped",
            ctx,
            reason=reason,
            variant_count=variant_count,
            query_rewrite_enabled=self.query_rewrite_enabled,
        )

    def _start_rewrite_trace(
        self,
        ctx: RequestContext | None,
        *,
        variant_count: int,
    ) -> object | None:
        if self.trace_logger is None or ctx is None:
            return None
        return self.trace_logger.start_span(
            "rag.query_rewrite",
            ctx,
            variant_count=variant_count,
        )

    def _end_rewrite_trace_success(
        self,
        ctx: RequestContext | None,
        span: object | None,
        result: QueryRewriteResult,
        *,
        variant_count: int,
    ) -> None:
        if self.trace_logger is None or span is None:
            return
        self.trace_logger.end_span(
            span,
            status="ok",
            variant_count=variant_count,
            produced_variants=len(result.variants),
            rewritten_query=result.rewritten_query,
        )

    def _end_rewrite_trace_error(
        self,
        ctx: RequestContext | None,
        span: object | None,
        *,
        error_code: str,
        variant_count: int,
    ) -> None:
        error_fields: dict[str, JsonValue] = {
            "status": "error",
            "error_code": error_code,
            "variant_count": variant_count,
            "fallback_to_original_query": True,
        }
        if self.trace_logger is not None and ctx is not None:
            self.trace_logger.record_event("rag.query_rewrite.error", ctx, **error_fields)
        if self.trace_logger is not None and span is not None:
            self.trace_logger.end_span(span, **error_fields)

    def _record_rewrite_error(
        self,
        ctx: RequestContext | None,
        span: object | None,
        exc: Exception,
        *,
        variant_count: int,
    ) -> None:
        error_fields: dict[str, JsonValue] = {
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "error_class": exc.__class__.__name__,
            "variant_count": variant_count,
            "fallback_to_original_query": True,
        }
        if self.trace_logger is not None and ctx is not None:
            self.trace_logger.record_event("rag.query_rewrite.error", ctx, **error_fields)
        if self.trace_logger is not None and span is not None:
            self.trace_logger.end_span(span, **error_fields)

    def _start_reranker_trace(
        self,
        ctx: RequestContext | None,
        *,
        candidate_count: int,
    ) -> object | None:
        if self.trace_logger is None or ctx is None:
            return None
        return self.trace_logger.start_span(
            "rag.reranker",
            ctx,
            candidate_count=candidate_count,
        )

    def _end_reranker_trace_success(
        self,
        span: object | None,
        *,
        candidate_count: int,
        returned_count: int,
    ) -> None:
        if self.trace_logger is None or span is None:
            return
        self.trace_logger.end_span(
            span,
            status="ok",
            candidate_count=candidate_count,
            returned_count=returned_count,
            reranker_used=True,
            fallback_to_vector_order=False,
        )

    def _record_reranker_error(
        self,
        ctx: RequestContext | None,
        span: object | None,
        exc: Exception,
        *,
        candidate_count: int,
    ) -> None:
        error_fields: dict[str, JsonValue] = {
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "error_class": exc.__class__.__name__,
            "candidate_count": candidate_count,
            "fallback_to_vector_order": True,
        }
        if self.trace_logger is not None and ctx is not None:
            self.trace_logger.record_event("rag.reranker.error", ctx, **error_fields)
        if self.trace_logger is not None and span is not None:
            self.trace_logger.end_span(span, **error_fields)

    def _record_reranker_invalid_result(
        self,
        ctx: RequestContext | None,
        span: object | None,
        *,
        candidate_count: int,
        returned_count: int,
    ) -> None:
        error_fields: dict[str, JsonValue] = {
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "error_class": "InvalidRerankerResult",
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "fallback_to_vector_order": True,
        }
        if self.trace_logger is not None and ctx is not None:
            self.trace_logger.record_event("rag.reranker.error", ctx, **error_fields)
        if self.trace_logger is not None and span is not None:
            self.trace_logger.end_span(span, **error_fields)


def _result_metric(result: SearchResultLike) -> str:
    metric = getattr(result, "metric", None)
    return metric.strip() if isinstance(metric, str) and metric.strip() else "L2"


# RRF 平滑常数，取业界通用值 60：单路时顺序不变，多路时共同命中结果显著前移。
_RRF_K = 60


def _chunk_fusion_key(chunk: RetrievedChunk) -> str:
    """生成多路召回融合用的 chunk 身份键。

    以稳定 chunk_id 为主、content_hash 兜底（迁移期 metadata 可能缺 chunk_id）；
    同一 chunk 的 chunk_id 与 content_hash 别名去重仍由 `deduplicate` 处理。
    """

    if chunk.chunk_id:
        return f"chunk:{chunk.chunk_id}"
    return f"hash:{chunk.content_hash}"


def _normalized_score_value(chunk: RetrievedChunk) -> float:
    return chunk.normalized_score if chunk.normalized_score is not None else -1.0


def _same_chunk_identity(
    original_chunks: Sequence[RetrievedChunk],
    reranked_chunks: Sequence[RetrievedChunk],
) -> bool:
    """判断 reranker 是否只改变排序。

    重排阶段不负责过滤证据，也不能注入新证据；否则 citation、eval 和 no-answer 的
    来源都会失去可解释性。这里用 chunk_id 与 content_hash 组合校验，兼顾稳定 ID
    和迁移期去重锚点。
    """

    original_identity = sorted(
        (chunk.chunk_id, chunk.content_hash) for chunk in original_chunks
    )
    reranked_identity = sorted(
        (chunk.chunk_id, chunk.content_hash) for chunk in reranked_chunks
    )
    return original_identity == reranked_identity


def _result_normalized_score(result: SearchResultLike) -> float | None:
    value = getattr(result, "normalized_score", None)
    return value if isinstance(value, int | float) else None


def _normalized_score_for_result(
    result: SearchResultLike,
    *,
    metric: str,
) -> tuple[float | None, str | None]:
    provided_score = _result_normalized_score(result)
    if provided_score is not None:
        # 兼容未来向量检索服务已经返回 normalized_score 的形态；仍然夹到 0-1，
        # 防止上游 bug 把 citation/API 可见分数带出契约范围。
        return _clamp_normalized_score(provided_score), None

    try:
        return normalize_score(result.score, metric), None
    except RagMetadataInvalidError:
        normalized_metric = metric.strip().upper() or "UNKNOWN"
        # 未知 metric 的 raw_score 可能是距离、相似度或供应商自定义分值。这里不能猜，
        # 否则低质量证据会被误判为高置信；用 0 分让 min_score 前置过滤处理。
        return 0.0, f"unsupported_metric:{normalized_metric}"


def _clamp_normalized_score(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _empty_reason(
    *,
    candidate_count: int,
    surviving_count: int,
    final_count: int,
) -> str | None:
    if final_count > 0:
        return None
    if candidate_count == 0:
        return "RAG_EMPTY_RESULT"
    if surviving_count == 0:
        return "RAG_LOW_SCORE"
    # final_k=0 或后续调用方主动要求不返回证据时，最终仍没有可用上下文。这里保持
    # RAG_EMPTY_RESULT，不新增对外错误码，避免当前 issue 扩大 API 契约面。
    return "RAG_EMPTY_RESULT"


def _build_no_answer_decision(
    *,
    reason_code: str | None,
    min_score: float,
    dropped_low_score_count: int,
) -> NoAnswerDecision | None:
    if reason_code is None:
        return None

    safe_message = _LOW_SCORE_MESSAGE if reason_code == "RAG_LOW_SCORE" else _EMPTY_RESULT_MESSAGE
    return NoAnswerDecision(
        should_answer=False,
        reason_code=reason_code,
        safe_message=safe_message,
        evidence_count=0,
        metadata={
            "min_score": min_score,
            "dropped_low_score_count": dropped_low_score_count,
        },
    )
