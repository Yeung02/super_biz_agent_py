"""ISSUE-017 RAG 内部数据模型测试。

这些测试只覆盖模型本身，不连接 Milvus、DashScope、MCP 或网络。这样可以证明
`app.rag.models` 是一个可独立回滚的阶段 3A 地基，不会提前改变业务 API 行为。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.core.errors import RagMetadataInvalidError
from app.rag.models import (
    ChunkRecord,
    Citation,
    NoAnswerDecision,
    RagContext,
    RagMetadata,
    RetrievedChunk,
    build_stable_rag_metadata,
)


def test_chunk_record_validates_required_metadata_and_keeps_trace_safe() -> None:
    metadata = RagMetadata(
        doc_id="doc_cpu",
        chunk_id="doc_cpu:000001",
        content_hash="hash-001",
        tenant_id="default",
        version=1,
        source_path="aiops-docs/cpu.md",
        file_name="cpu.md",
        chunk_index=1,
        metadata_extra={"h1": "CPU", "debug": "kept-internal"},
    )

    chunk = ChunkRecord(content="CPU 使用率持续超过 80% 时应检查热点进程。", metadata=metadata)

    assert chunk.doc_id == "doc_cpu"
    assert chunk.chunk_id == "doc_cpu:000001"
    trace_fields = chunk.to_trace_fields()
    assert trace_fields == {
        "doc_id": "doc_cpu",
        "chunk_id": "doc_cpu:000001",
        "source_path": "aiops-docs/cpu.md",
        "file_name": "cpu.md",
        "chunk_index": 1,
        "tenant_id": "default",
        "metadata_version": 1,
        "content_hash": "hash-001",
    }
    assert "content" not in trace_fields
    assert "debug" not in trace_fields


def test_from_langchain_document_converts_legacy_metadata_with_warning() -> None:
    doc = Document(
        page_content="旧知识片段",
        metadata={
            "_source": "aiops-docs/legacy.md",
            "_file_name": "legacy.md",
            "_extension": ".md",
            "h1": "旧标题",
        },
    )

    chunk = ChunkRecord.from_langchain_document(doc, fallback_chunk_id="legacy-uuid-1")

    assert chunk.source_path == "aiops-docs/legacy.md"
    assert chunk.file_name == "legacy.md"
    assert chunk.extension == ".md"
    assert chunk.chunk_index == 0
    assert chunk.metadata.metadata_extra["h1"] == "旧标题"
    assert "legacy_metadata" in chunk.metadata.compat_warnings
    assert chunk.chunk_id.startswith("legacy:")
    assert chunk.doc_id.startswith("legacy:")


def test_missing_source_metadata_raises_rag_metadata_invalid() -> None:
    doc = Document(page_content="缺少来源", metadata={"_file_name": "missing.md"})

    with pytest.raises(RagMetadataInvalidError) as exc_info:
        ChunkRecord.from_langchain_document(doc)

    assert exc_info.value.code == "RAG_METADATA_INVALID"
    assert exc_info.value.user_message == "知识库文档元数据不完整。"
    assert exc_info.value.safe_details()["missing_fields"] == ["source_path"]


def test_retrieved_chunk_requires_score_metric_and_normalized_score_range() -> None:
    chunk = RetrievedChunk(
        content="证据",
        metadata=RagMetadata(
            doc_id="doc_1",
            chunk_id="doc_1:000001",
            content_hash="hash-001",
            source_path="aiops-docs/doc.md",
            file_name="doc.md",
            chunk_index=1,
        ),
        raw_score=0.25,
        normalized_score=0.8,
        metric="L2",
    )

    assert chunk.raw_score == 0.25
    assert chunk.normalized_score == 0.8
    assert chunk.metric == "L2"
    assert chunk.to_trace_fields()["raw_score"] == 0.25

    with pytest.raises(ValidationError):
        RetrievedChunk(
            content="证据",
            metadata=chunk.metadata,
            raw_score=0.25,
            normalized_score=1.2,
            metric="L2",
        )

    with pytest.raises(ValidationError):
        RetrievedChunk(
            content="证据",
            metadata=chunk.metadata,
            raw_score=0.25,
            normalized_score=0.8,
        )


def test_citation_preconversion_returns_api_safe_fields_only() -> None:
    citation = Citation(
        citation_id="C1",
        doc_id="doc_cpu",
        chunk_id="doc_cpu:000001",
        source_path="aiops-docs/cpu.md",
        file_name="cpu.md",
        normalized_score=0.82,
        raw_score=0.1,
        metric="L2",
        evidence_text="A" * 260,
        metadata={"internal": "hidden"},
    )

    api_citation = citation.to_api_citation()

    assert api_citation == {
        "citation_id": "C1",
        "doc_id": "doc_cpu",
        "chunk_id": "doc_cpu:000001",
        "source_path": "aiops-docs/cpu.md",
        "file_name": "cpu.md",
        "score": 0.82,
        "content_preview": "A" * 200,
    }


def test_citation_rejects_absolute_or_traversal_source_path() -> None:
    with pytest.raises(RagMetadataInvalidError):
        Citation(
            citation_id="C1",
            doc_id="doc_secret",
            chunk_id="doc_secret:000001",
            source_path="../secret.md",
            file_name="secret.md",
            normalized_score=0.5,
            evidence_text="secret",
        ).to_api_citation()


def test_rag_context_and_no_answer_decision_are_structured() -> None:
    decision = NoAnswerDecision(
        should_answer=False,
        reason_code="RAG_EMPTY_RESULT",
        safe_message="知识库没有找到相关依据。",
        evidence_count=0,
    )
    context = RagContext(
        context_text="",
        used_chunks=[],
        dropped_chunks=[],
        citations=[],
        no_answer_decision=decision,
    )

    assert context.no_answer_decision is decision
    assert context.to_trace_fields() == {
        "used_chunk_count": 0,
        "dropped_chunk_count": 0,
        "citation_count": 0,
        "no_answer": True,
        "no_answer_reason": "RAG_EMPTY_RESULT",
    }


def test_stable_rag_metadata_normalizes_path_and_generates_bounded_ids() -> None:
    metadata_from_windows_path = build_stable_rag_metadata(
        tenant_id="default",
        source_path=r"AIops-Docs\CPU_HIGH_USAGE.md",
        chunk_index=3,
        chunk_text="CPU chunk",
        document_content="CPU line\r\nnext line",
    )
    metadata_from_posix_path = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu_high_usage.md",
        chunk_index=3,
        chunk_text="CPU chunk",
        document_content="CPU line\nnext line",
    )

    assert metadata_from_windows_path == metadata_from_posix_path
    assert metadata_from_windows_path["source_path"] == "aiops-docs/cpu_high_usage.md"
    assert metadata_from_windows_path["file_name"] == "cpu_high_usage.md"
    assert metadata_from_windows_path["extension"] == ".md"
    assert metadata_from_windows_path["version"] == 1
    assert metadata_from_windows_path["chunk_id"].endswith("#000003")
    assert len(metadata_from_windows_path["doc_id"]) <= 100
    assert len(metadata_from_windows_path["chunk_id"]) <= 100


def test_stable_content_hash_changes_without_changing_path_based_logical_ids() -> None:
    original = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu.md",
        chunk_index=0,
        chunk_text="old chunk",
        document_content="old document",
    )
    changed = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu.md",
        chunk_index=0,
        chunk_text="new chunk",
        document_content="new document",
    )

    assert changed["doc_id"] == original["doc_id"]
    assert changed["chunk_id"] == original["chunk_id"]
    assert changed["content_hash"] != original["content_hash"]


def test_content_hash_is_chunk_scoped_not_document_scoped() -> None:
    first = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu.md",
        chunk_index=0,
        chunk_text="same chunk",
        document_content="document version one",
    )
    second = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu.md",
        chunk_index=0,
        chunk_text="same chunk",
        document_content="document version two",
    )

    assert second["doc_id"] == first["doc_id"]
    assert second["chunk_id"] == first["chunk_id"]
    assert second["content_hash"] == first["content_hash"]
