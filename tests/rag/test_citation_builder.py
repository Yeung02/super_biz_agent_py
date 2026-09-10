"""ISSUE-026 CitationBuilder citation 输出测试。

这些测试只覆盖 `RagContext.used_chunks -> citations[]` 的内部转换边界，不接入
Chat API、真实检索、Milvus、DashScope 或 MCP。这样当前 issue 可以独立回滚，
同时锁定 API citation 不能泄露绝对路径、raw metadata 或完整 chunk 的契约。
"""

from __future__ import annotations

from typing import Protocol

from app.core.request_context import RequestContext
from app.rag.citation import CitationBuilder
from app.rag.models import RagContext, RagMetadata, RetrievedChunk


class TraceLoggerLike(Protocol):
    """测试 trace logger 的最小协议。"""

    events: list[dict[str, object]]


class RecordingTraceLogger:
    """记录 CitationBuilder trace 事件，避免测试写真实 JSONL。"""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record_event(self, name: str, ctx: RequestContext, **fields: object) -> None:
        self.events.append({"name": name, "trace_id": ctx.trace_id, **fields})


def test_citation_builder_assigns_ids_in_used_chunk_order_and_merges_duplicates(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    builder = CitationBuilder(trace_logger=trace_logger)
    first_cpu = _chunk(
        "doc_cpu#000001",
        "hash-cpu",
        "CPU 高可信证据",
        score=0.92,
    )
    duplicate_cpu = _chunk(
        "doc_cpu#000001",
        "hash-cpu-copy",
        "CPU 重复证据不应再次输出",
        score=0.88,
    )
    memory = _chunk(
        "doc_mem#000001",
        "hash-mem",
        "内存证据",
        score=0.81,
    )
    context = RagContext(
        context_text="",
        used_chunks=[first_cpu, duplicate_cpu, memory],
    )

    citations = builder.build(context, "答案未显式携带引用锚点。", ctx=fake_request_context)
    api_citations = builder.to_api_schema(citations)

    assert [citation.citation_id for citation in citations] == ["C1", "C2"]
    assert [citation.chunk_id for citation in citations] == [
        "doc_cpu#000001",
        "doc_mem#000001",
    ]
    assert context.citations == citations
    assert api_citations[0]["citation_id"] == "C1"
    assert api_citations[0]["content_preview"] == "CPU 高可信证据"
    assert api_citations[1]["citation_id"] == "C2"
    assigned = builder.assign_ids([first_cpu, memory])
    assert [citation.citation_id for citation in assigned] == ["C1", "C2"]
    assert trace_logger.events[-1] == {
        "name": "rag.citation.build",
        "trace_id": "trc_test",
        "citation_count": 2,
        "dropped_invalid_count": 0,
        "doc_ids": ["doc_cpu", "doc_mem"],
        "chunk_ids": ["doc_cpu#000001", "doc_mem#000001"],
    }


def test_citation_builder_outputs_api_schema_only_and_truncates_preview() -> None:
    builder = CitationBuilder()
    context = RagContext(
        context_text="",
        used_chunks=[
            _chunk(
                "doc_cpu#000002",
                "hash-long",
                "A" * 260,
                score=0.73,
            )
        ],
    )

    api_citations = builder.build_api_schema(context, answer="无锚点回答")

    assert api_citations == [
        {
            "citation_id": "C1",
            "doc_id": "doc_cpu",
            "chunk_id": "doc_cpu#000002",
            "source_path": "aiops-docs/cpu.md",
            "file_name": "cpu.md",
            "score": 0.73,
            "content_preview": "A" * 200,
        }
    ]
    assert "raw_score" not in api_citations[0]
    assert "metric" not in api_citations[0]
    assert "metadata" not in api_citations[0]


def test_citation_builder_drops_unsafe_paths_and_missing_identity_with_trace(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    builder = CitationBuilder(trace_logger=trace_logger)
    unsafe_path = _chunk(
        "doc_secret#000001",
        "hash-secret",
        "不应输出的绝对路径证据",
        score=0.66,
        source_path="C:/internal/secret.md",
    )
    missing_identity = _invalid_identity_chunk()
    valid = _chunk(
        "doc_disk#000001",
        "hash-disk",
        "磁盘证据",
        score=0.77,
    )
    context = RagContext(
        context_text="",
        used_chunks=[unsafe_path, missing_identity, valid],
    )

    citations = builder.build(context, "答案", ctx=fake_request_context)
    api_citations = builder.to_api_schema(citations)

    assert [citation.chunk_id for citation in citations] == ["doc_disk#000001"]
    assert api_citations[0]["source_path"] == "aiops-docs/disk.md"
    assert trace_logger.events[-1]["citation_count"] == 1
    assert trace_logger.events[-1]["dropped_invalid_count"] == 2
    assert trace_logger.events[-1]["doc_ids"] == ["doc_disk"]
    assert trace_logger.events[-1]["chunk_ids"] == ["doc_disk#000001"]


def test_sanitize_answer_anchors_keeps_valid_markers_and_strips_invalid_ones(
    fake_request_context: RequestContext,
) -> None:
    trace_logger = RecordingTraceLogger()
    builder = CitationBuilder(trace_logger=trace_logger)
    answer = "CPU 达到 92% [C1]，内存充足 [C3]。磁盘正常 [C99]。多重引用 [C2][C1]。"

    sanitized = builder.sanitize_answer_anchors(
        answer,
        ["C1", "C2"],
        ctx=fake_request_context,
    )

    assert sanitized == "CPU 达到 92% [C1]，内存充足。磁盘正常。多重引用 [C2][C1]。"
    assert trace_logger.events[-1] == {
        "name": "rag.citation.anchor_check",
        "trace_id": "trc_test",
        "marker_count": 5,
        "valid_marker_count": 3,
        "removed_marker_count": 2,
        "cited_ids": ["C1", "C2"],
        "sanitized_answer_chars": len(sanitized),
    }


def test_sanitize_answer_anchors_removes_all_markers_without_valid_ids() -> None:
    builder = CitationBuilder()
    answer = "正文 [C1] 不应保留悬空标记 [C12]。"

    sanitized = builder.sanitize_answer_anchors(answer, [])

    # 清洗删除的是"前导空格+标记"，标记后方的普通空格不属于匹配范围。
    assert sanitized == "正文 不应保留悬空标记。"


def _chunk(
    chunk_id: str,
    content_hash: str,
    content: str,
    *,
    score: float,
    source_path: str | None = None,
) -> RetrievedChunk:
    doc_id, raw_index = chunk_id.split("#", maxsplit=1)
    file_stem = doc_id.removeprefix("doc_")
    resolved_source_path = source_path or f"aiops-docs/{file_stem}.md"
    return RetrievedChunk(
        content=content,
        metadata=RagMetadata(
            doc_id=doc_id,
            chunk_id=chunk_id,
            content_hash=content_hash,
            source_path=resolved_source_path,
            file_name=resolved_source_path.rsplit("/", maxsplit=1)[-1],
            chunk_index=int(raw_index),
        ),
        normalized_score=score,
        raw_score=1.0 - score,
        metric="L2",
    )


def _invalid_identity_chunk() -> RetrievedChunk:
    """构造迁移期异常对象，模拟上游绕过 Pydantic 校验后的缺失 ID 风险。"""

    metadata = RagMetadata.model_construct(
        doc_id="",
        chunk_id="",
        content_hash="hash-missing",
        tenant_id="default",
        version=1,
        source_path="aiops-docs/missing.md",
        file_name="missing.md",
        chunk_index=0,
        extension=".md",
        created_at=None,
        updated_at=None,
        metadata_extra={},
        compat_warnings=[],
    )
    return RetrievedChunk.model_construct(
        content="缺少 doc_id/chunk_id 的证据",
        metadata=metadata,
        raw_score=0.2,
        normalized_score=0.8,
        metric="L2",
        success=True,
    )
