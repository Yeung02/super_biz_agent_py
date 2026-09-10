"""CitationBuilder：把内部 RAG 证据转换为 API-safe citations。

ISSUE-026 实现 citation 输出边界：从 `RagContext.used_chunks` 生成内部
`Citation`，再转换为 `docs/api_contract.md` 允许的 `citations[]` 字段。本模块
不接入检索、不生成答案，也不暴露绝对路径、完整 chunk 或 raw metadata。

在 citation 输出边界之上，`sanitize_answer_anchors` 负责校验答案正文中的
`[C*]` 引用标记：模型被要求在正文中标注证据编号，凡是无法对应已构建 citation
的标记（编造编号、指向被丢弃证据的编号）都会被剥离，保证 API 输出的标记全部
可解析。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from app.core.errors import JsonObject, JsonValue, RagMetadataInvalidError
from app.core.request_context import RequestContext
from app.rag.models import Citation, RagContext, RagMetadataValue, RetrievedChunk

DEFAULT_PREVIEW_CHARS = 200

# 答案正文中的引用标记形如 [C1]/[C12]，与 ContextBuilder 写入 context 的 anchor 同构。
_ANSWER_MARKER_PATTERN = re.compile(r"( ?)\[C(\d+)\]")


class TraceLoggerLike(Protocol):
    """CitationBuilder 使用的 trace 最小协议。"""

    def record_event(self, name: str, ctx: RequestContext, **fields: JsonValue) -> None:
        """记录结构化 trace 事件。"""


class CitationBuilder:
    """从 `RagContext.used_chunks` 构建 API 安全 citation。

    Builder 的职责只有两步：第一，按 context 中已经确认可用的 chunk 顺序分配 `C1/C2`
    并合并重复证据；第二，把内部 `Citation` 投影为对外安全字段。这样后续 API adapter
    只需要追加 `citations`，不会删除旧 `answer/errorMessage` 字段，也不会把 RAG 内部
    metadata、raw_score 或本机路径泄露给前端。
    """

    def __init__(
        self,
        *,
        trace_logger: TraceLoggerLike | None = None,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ) -> None:
        self.trace_logger = trace_logger
        self.preview_chars = max(0, int(preview_chars))

    def build(
        self,
        context: RagContext,
        answer: str,
        *,
        ctx: RequestContext | None = None,
    ) -> list[Citation]:
        """生成内部 citation，并写回 `context.citations`。

        `answer` 当前只作为未来按正文 anchor 精确筛选 citation 的扩展参数保留。工程计划
        明确本 issue 的风险是“答案未显式引用 anchor”，因此当前先按 `used_chunks` 输出
        来源列表，避免为了精确对齐答案而提前引入 LLM judge 或事实校验。
        """

        _ = answer
        citations: list[Citation] = []
        dropped_invalid_count = 0
        seen_keys: set[str] = set()

        for chunk in context.used_chunks:
            if not _is_chunk_citable(chunk):
                dropped_invalid_count += 1
                continue

            identity_keys = _chunk_identity_keys(chunk)
            if seen_keys.intersection(identity_keys):
                # 重复 chunk 合并不是错误：ContextBuilder 已经可能做过去重，但 Builder
                # 仍需自我保护，防止上游绕过 context packing 后让同一证据重复出现在 API。
                continue

            citation = self._citation_from_chunk(
                chunk,
                citation_id=f"C{len(citations) + 1}",
            )
            if not self.validate_public_fields(citation):
                dropped_invalid_count += 1
                continue

            seen_keys.update(identity_keys)
            citations.append(citation)

        context.citations = citations
        self._record_trace(citations, dropped_invalid_count=dropped_invalid_count, ctx=ctx)
        return citations

    def build_api_schema(
        self,
        context: RagContext,
        answer: str,
        *,
        ctx: RequestContext | None = None,
    ) -> list[JsonObject]:
        """构建 API-safe `citations[]`。

        该便捷入口供 API adapter 后续接入时使用；它只追加安全 citation 字段，不改变
        Chat 旧响应结构，也不会把 `Citation.metadata/raw_score/span_*` 带出内部边界。
        """

        citations = self.build(context, answer, ctx=ctx)
        return self.to_api_schema(citations)

    def sanitize_answer_anchors(
        self,
        answer: str,
        valid_citation_ids: Iterable[str],
        *,
        ctx: RequestContext | None = None,
    ) -> str:
        """校验答案正文中的 `[C*]` 标记，移除无法对应已构建 citation 的标记。

        system prompt 要求模型在正文中标注 `[C1]` 证据编号；模型可能编造不存在的
        编号，或引用了在 citation 构建阶段被丢弃（非法/去重）的证据编号。这些标记
        对用户不可解析，必须在使用前剥离，而不是把悬空标记透传到 API。清洗只做
        字符级删除，不改写其余答案内容。
        """

        valid_ids = {citation_id for citation_id in valid_citation_ids if isinstance(citation_id, str)}
        marker_count = 0
        removed_count = 0
        cited_ids: list[str] = []
        seen_cited: set[str] = set()

        def _replace(match: re.Match[str]) -> str:
            nonlocal marker_count, removed_count
            marker_count += 1
            citation_id = f"C{match.group(2)}"
            if citation_id in valid_ids:
                if citation_id not in seen_cited:
                    seen_cited.add(citation_id)
                    cited_ids.append(citation_id)
                return match.group(0)
            removed_count += 1
            # 丢弃标记时同时吃掉一个前导空格，避免留下 "依据  数据" 这类双空格。
            return ""

        sanitized = _ANSWER_MARKER_PATTERN.sub(_replace, answer)
        self._record_anchor_trace(
            sanitized,
            marker_count=marker_count,
            removed_count=removed_count,
            cited_ids=cited_ids,
            ctx=ctx,
        )
        return sanitized

    def _record_anchor_trace(
        self,
        sanitized_answer: str,
        *,
        marker_count: int,
        removed_count: int,
        cited_ids: Sequence[str],
        ctx: RequestContext | None,
    ) -> None:
        if self.trace_logger is None or ctx is None:
            return
        self.trace_logger.record_event(
            "rag.citation.anchor_check",
            ctx,
            marker_count=marker_count,
            valid_marker_count=marker_count - removed_count,
            removed_marker_count=removed_count,
            cited_ids=list(cited_ids),
            sanitized_answer_chars=len(sanitized_answer),
        )

    def assign_ids(self, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
        """按传入 chunk 顺序分配 `C1/C2`。

        这是一个纯转换入口，主要服务测试和后续 adapter；非法或重复 chunk 的处理规则与
        `build()` 一致，但不写 trace，也不需要 `RagContext`。
        """

        temp_context = RagContext(context_text="", used_chunks=list(chunks))
        return self.build(temp_context, answer="")

    def to_api_schema(self, citations: Sequence[Citation]) -> list[JsonObject]:
        """把内部 citation 转为 API 契约字段。

        如果外部调用方传入了带绝对路径或目录逃逸的内部 citation，本方法会丢弃该条，
        而不是把异常或不安全路径继续传到 HTTP/SSE 层。构建阶段已经负责 trace 统计；
        这里保持无副作用，便于 API adapter 安全复用。
        """

        api_citations: list[JsonObject] = []
        for citation in citations:
            try:
                api_citation = citation.to_api_citation(preview_chars=self.preview_chars)
            except RagMetadataInvalidError:
                continue
            api_citations.append(api_citation)
        return api_citations

    def preview(self, text: str, max_chars: int | None = None) -> str:
        """按 API 契约截断证据预览。

        不做摘要或语义改写，只做字符级截断；这样 preview 永远来自真实 chunk，且不会
        因生成式摘要引入未验证事实。完整 chunk 仅留在内部 `RetrievedChunk.content`。
        """

        limit = self.preview_chars if max_chars is None else max(0, int(max_chars))
        return text[:limit]

    def validate_public_fields(self, citation: Citation) -> bool:
        """验证 citation 是否可安全输出到 API。

        这里复用 `Citation.to_api_citation()` 的路径安全检查，而不是重新维护一套规则。
        这样 API 契约变化时只需要更新模型层安全投影，Builder 和测试不会出现双轨语义。
        """

        if not _non_empty_text(citation.doc_id) or not _non_empty_text(citation.chunk_id):
            return False
        if not _non_empty_text(citation.source_path) or not _non_empty_text(citation.file_name):
            return False
        try:
            citation.to_api_citation(preview_chars=self.preview_chars)
        except RagMetadataInvalidError:
            return False
        return True

    def _citation_from_chunk(self, chunk: RetrievedChunk, *, citation_id: str) -> Citation:
        """从 `RetrievedChunk` 构造内部 `Citation`。

        内部 citation 保留 raw_score、metric 和最小评估 metadata，方便后续 trace/eval；
        API 输出仍必须经过 `to_api_schema()`，不会直接暴露这些字段。
        """

        return Citation(
            citation_id=citation_id,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            file_name=chunk.file_name,
            normalized_score=chunk.normalized_score,
            raw_score=chunk.raw_score,
            metric=chunk.metric,
            evidence_text=self.preview(chunk.content),
            metadata=_internal_metadata_from_chunk(chunk),
        )

    def _record_trace(
        self,
        citations: Sequence[Citation],
        *,
        dropped_invalid_count: int,
        ctx: RequestContext | None,
    ) -> None:
        if self.trace_logger is None or ctx is None:
            return
        self.trace_logger.record_event(
            "rag.citation.build",
            ctx,
            citation_count=len(citations),
            dropped_invalid_count=dropped_invalid_count,
            doc_ids=_ordered_unique(citation.doc_id for citation in citations),
            chunk_ids=[citation.chunk_id for citation in citations],
        )


def _is_chunk_citable(chunk: RetrievedChunk) -> bool:
    if not chunk.success:
        return False
    if not _non_empty_text(chunk.doc_id) or not _non_empty_text(chunk.chunk_id):
        return False
    if not _non_empty_text(chunk.source_path) or not _non_empty_text(chunk.file_name):
        return False
    return True


def _non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _chunk_identity_keys(chunk: RetrievedChunk) -> set[str]:
    keys = {f"chunk:{chunk.chunk_id}"}
    if chunk.content_hash:
        keys.add(f"hash:{chunk.content_hash}")
    return keys


def _internal_metadata_from_chunk(chunk: RetrievedChunk) -> dict[str, RagMetadataValue]:
    """生成内部 citation metadata。

    只保留 eval/trace 有用且低敏的字段，不复制完整 `metadata_extra`。这样内部 schema
    仍可评估、可追踪，但不会让未来 API adapter 误把业务 raw metadata 透传出去。
    """

    return {
        "tenant_id": chunk.tenant_id,
        "content_hash": chunk.content_hash,
        "chunk_index": chunk.chunk_index,
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


__all__ = ["CitationBuilder"]
