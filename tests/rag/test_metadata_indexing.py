"""ISSUE-018 metadata indexing tests.

这些测试只验证稳定 metadata 如何进入分片与向量写入边界，不测试后续 ISSUE-019 的
幂等删除、doc-level delete 或失败任务状态，也不连接真实 Milvus、DashScope、MCP 或网络。
"""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from langchain_core.documents import Document
from pytest import MonkeyPatch

from app.core.errors import EmbeddingProviderError, VectorStoreUnavailableError
from app.rag.models import build_stable_rag_metadata
from app.services.document_splitter_service import DocumentSplitterService

EVAL_SET_PATH = Path("eval_sets/rag_cases.yaml")
CORE_AIOPS_DOCS = (
    "aiops-docs/cpu_high_usage.md",
    "aiops-docs/disk_high_usage.md",
    "aiops-docs/memory_high_usage.md",
    "aiops-docs/service_unavailable.md",
    "aiops-docs/slow_response.md",
)
RAG_DOC_ID_PATTERN = re.compile(r"^doc_[0-9a-f]{32}$")
ALLOWED_CASE_TYPES = {"answer", "low_score", "empty_retrieval", "no_answer"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


class RecordingVectorStore:
    """记录 LangChain Milvus adapter 收到的文档和 ids，避免真实向量库连接。"""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.ids: list[str] = []

    def add_documents(self, documents: Sequence[Document], *, ids: Sequence[str]) -> list[str]:
        self.documents = list(documents)
        self.ids = list(ids)
        return list(ids)


class RecordingMilvusCollection:
    """记录 delete 表达式，避免 doc-level delete 测试连接真实 Milvus。"""

    def __init__(self, *, delete_count: int = 2) -> None:
        self.delete_count = delete_count
        self.expressions: list[str] = []

    def delete(self, expr: str) -> SimpleNamespace:
        self.expressions.append(expr)
        return SimpleNamespace(delete_count=self.delete_count)

    def query(self, expr: str, output_fields: list[str] | None = None) -> list[SimpleNamespace]:
        _ = output_fields
        self.expressions.append(expr)
        return []


class FixedMetadataSplitter:
    """返回带稳定 RAG metadata 的分片，专门验证索引服务的删除和写入编排。"""

    def __init__(self, *, doc_id: str = "doc_cpu") -> None:
        self.doc_id = doc_id
        self.calls: list[dict[str, object]] = []

    def split_document(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = "default",
    ) -> list[Document]:
        self.calls.append(
            {
                "content": content,
                "file_path": file_path,
                "source_root": source_root,
                "tenant_id": tenant_id,
            }
        )
        return [
            Document(
                page_content="CPU chunk 0",
                metadata={
                    "doc_id": self.doc_id,
                    "chunk_id": f"{self.doc_id}#000000",
                    "content_hash": "hash-cpu",
                    "tenant_id": tenant_id,
                    "version": 1,
                    "source_path": "cpu.md",
                    "file_name": "cpu.md",
                    "chunk_index": 0,
                    "_source": file_path,
                },
            ),
            Document(
                page_content="CPU chunk 1",
                metadata={
                    "doc_id": self.doc_id,
                    "chunk_id": f"{self.doc_id}#000001",
                    "content_hash": "hash-cpu",
                    "tenant_id": tenant_id,
                    "version": 1,
                    "source_path": "cpu.md",
                    "file_name": "cpu.md",
                    "chunk_index": 1,
                    "_source": file_path,
                },
            ),
        ]


class EmptyMetadataSplitter:
    """Splitter fake for files that validate but produce no indexable chunks."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def split_document(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = "default",
    ) -> list[Document]:
        self.calls.append(
            {
                "content": content,
                "file_path": file_path,
                "source_root": source_root,
                "tenant_id": tenant_id,
            }
        )
        return []


class RecordingIndexStoreManager:
    """记录索引服务对向量存储的调用顺序，用内存行为覆盖真实 Milvus。"""

    def __init__(self, *, old_hashes: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.added_documents: list[Document] = []
        self.old_hashes = old_hashes or {}

    def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
        self.calls.append(("get_chunk_hashes_by_doc_id", doc_id))
        return dict(self.old_hashes)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        self.calls.append(("delete_by_chunk_ids", tuple(chunk_ids)))
        return len(chunk_ids)

    def delete_by_doc_id(self, doc_id: str) -> int:
        self.calls.append(("delete_by_doc_id", doc_id))
        return 2

    def delete_by_source(self, file_path: str) -> int:
        self.calls.append(("delete_by_source", file_path))
        return 1

    def add_documents(self, documents: list[Document]) -> list[str]:
        self.added_documents = list(documents)
        chunk_ids = [str(document.metadata["chunk_id"]) for document in documents]
        self.calls.append(("add_documents", tuple(chunk_ids)))
        return chunk_ids


def test_minimal_rag_eval_set_is_parseable_and_schema_safe() -> None:
    """校验 ISSUE-020 只交付离线 eval 数据文件，不提前引入后续 loader。"""

    cases = _load_eval_cases(EVAL_SET_PATH)
    known_doc_ids = _expected_aiops_doc_ids()
    cases_by_doc_id = dict.fromkeys(known_doc_ids.values(), 0)
    seen_case_ids: set[str] = set()
    case_types: set[str] = set()

    for case in cases:
        case_id = _required_text(case, "id")
        assert case_id not in seen_case_ids
        seen_case_ids.add(case_id)
        assert _required_text(case, "question")
        should_answer = _required_bool(case, "should_answer")
        expected_doc_ids = _required_text_list(case, "expected_doc_ids")
        expected_keywords = _required_text_list(case, "expected_keywords")
        tags = _required_text_list(case, "tags")
        case_type = _required_text(case, "case_type")
        difficulty = _required_text(case, "difficulty")

        assert case_type in ALLOWED_CASE_TYPES
        assert difficulty in ALLOWED_DIFFICULTIES
        assert tags
        case_types.add(case_type)

        if should_answer:
            assert expected_doc_ids, "可回答 case 必须声明 expected_doc_ids，避免指标误判为空命中"
            assert expected_keywords, "可回答 case 必须声明关键词，后续 runner 才能做轻量正确性检查"
        else:
            assert not expected_doc_ids, "拒答/空检索 case 不应绑定文档，避免把无证据问题算作召回成功"

        for doc_id in expected_doc_ids:
            assert RAG_DOC_ID_PATTERN.match(doc_id)
            assert doc_id in cases_by_doc_id
            cases_by_doc_id[doc_id] += 1

    assert case_types >= ALLOWED_CASE_TYPES
    assert all(3 <= count <= 5 for count in cases_by_doc_id.values())


def _load_eval_cases(path: Path) -> list[dict[str, object]]:
    """读取 YAML 并保持测试内 schema 校验，避免在 ISSUE-020 提前实现 evaluation loader。"""

    raw_cases: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw_cases, list)
    cases: list[dict[str, object]] = []
    for raw_case in raw_cases:
        assert isinstance(raw_case, dict)
        cases.append({str(key): value for key, value in raw_case.items()})
    return cases


def _expected_aiops_doc_ids() -> dict[str, str]:
    """按 ISSUE-018 稳定 ID 规则生成文档 ID，防止 eval set 手写 ID 与索引规则漂移。"""

    doc_ids: dict[str, str] = {}
    for source_path in CORE_AIOPS_DOCS:
        metadata = build_stable_rag_metadata(
            tenant_id="default",
            source_path=source_path,
            chunk_index=0,
            chunk_text="",
            document_content="",
        )
        doc_ids[source_path] = str(metadata["doc_id"])
    return doc_ids


def _required_text(case: dict[str, object], field: str) -> str:
    value = case.get(field)
    assert isinstance(value, str) and value.strip(), f"{field} 必须是非空字符串"
    return value


def _required_bool(case: dict[str, object], field: str) -> bool:
    value = case.get(field)
    assert isinstance(value, bool), f"{field} 必须是布尔值"
    return value


def _required_text_list(case: dict[str, object], field: str) -> list[str]:
    value = case.get(field)
    assert isinstance(value, list), f"{field} 必须是列表"
    result: list[str] = []
    for item in value:
        assert isinstance(item, str) and item.strip(), f"{field} 只能包含非空字符串"
        result.append(item)
    return result


def test_document_splitter_writes_stable_and_legacy_metadata(tmp_path: Path) -> None:
    service = DocumentSplitterService()
    source_root = tmp_path / "uploads"
    source_root.mkdir()
    file_path = source_root / "CPU_RunBook.MD"
    content = "# CPU\n\nCPU 使用率持续超过 80% 时应检查热点进程。"

    first_docs = service.split_document(
        content,
        file_path.as_posix(),
        source_root=source_root,
        tenant_id="default",
    )
    second_docs = service.split_document(
        content.replace("\n", "\r\n"),
        file_path.as_posix().replace("/", "\\"),
        source_root=source_root,
        tenant_id="default",
    )

    assert first_docs
    first_metadata = first_docs[0].metadata
    second_metadata = second_docs[0].metadata
    assert first_metadata["doc_id"] == second_metadata["doc_id"]
    assert first_metadata["chunk_id"] == second_metadata["chunk_id"]
    assert first_metadata["content_hash"] == second_metadata["content_hash"]
    assert first_metadata["source_path"] == "cpu_runbook.md"
    assert first_metadata["file_name"] == "cpu_runbook.md"
    assert first_metadata["chunk_index"] == 0
    assert first_metadata["tenant_id"] == "default"
    assert first_metadata["version"] == 1
    assert first_metadata["_source"] == file_path.as_posix()
    assert first_metadata["_file_name"] == "CPU_RunBook.MD"
    assert first_metadata["_extension"] == ".md"


def test_vector_store_manager_uses_chunk_id_as_primary_id(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    manager = vector_store_module.VectorStoreManager()
    fake_store = RecordingVectorStore()
    manager.vector_store = fake_store
    monkeypatch.setattr(vector_store_module.config, "stable_rag_ids_enabled", True)
    documents = [
        Document(page_content="chunk-0", metadata={"chunk_id": "doc_cpu#000000"}),
        Document(page_content="chunk-1", metadata={"chunk_id": "doc_cpu#000001"}),
    ]

    result_ids = manager.add_documents(documents)

    assert result_ids == ["doc_cpu#000000", "doc_cpu#000001"]
    assert fake_store.ids == result_ids
    assert fake_store.documents == documents


def test_vector_store_manager_keeps_uuid_rollback_path(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    manager = vector_store_module.VectorStoreManager()
    fake_store = RecordingVectorStore()
    manager.vector_store = fake_store
    monkeypatch.setattr(vector_store_module.config, "stable_rag_ids_enabled", False)
    monkeypatch.setattr(vector_store_module.uuid, "uuid4", lambda: "uuid-rollback")
    documents = [Document(page_content="chunk-0", metadata={"chunk_id": "doc_cpu#000000"})]

    result_ids = manager.add_documents(documents)

    assert result_ids == ["uuid-rollback"]
    assert fake_store.ids == ["uuid-rollback"]


def test_vector_store_manager_deletes_chunks_by_doc_id(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    manager = vector_store_module.VectorStoreManager()
    collection = RecordingMilvusCollection(delete_count=2)
    monkeypatch.setattr(milvus_module.milvus_manager, "get_collection", lambda: collection)

    deleted_count = manager.delete_by_doc_id("doc_cpu")

    assert deleted_count == 2
    assert collection.expressions == ['metadata["doc_id"] == "doc_cpu"']


def test_vector_store_manager_maps_embedding_failures(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)

    class FailingEmbeddingStore:
        def add_documents(self, documents: Sequence[Document], *, ids: Sequence[str]) -> list[str]:
            _ = documents, ids
            raise RuntimeError("DashScope embedding provider failed")

    manager = vector_store_module.VectorStoreManager()
    manager.vector_store = FailingEmbeddingStore()
    documents = [Document(page_content="chunk-0", metadata={"chunk_id": "doc_cpu#000000"})]

    with pytest.raises(EmbeddingProviderError) as exc_info:
        manager.add_documents(documents)

    assert exc_info.value.code == "EMBEDDING_PROVIDER_ERROR"


def test_index_single_file_deletes_by_doc_id_before_source_fallback(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_module = importlib.import_module("app.services.vector_index_service")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    file_path = upload_dir / "cpu.md"
    file_path.write_text("# CPU\n\nCPU usage", encoding="utf-8")
    splitter = FixedMetadataSplitter(doc_id="doc_cpu")
    store_manager = RecordingIndexStoreManager()
    monkeypatch.setattr(index_module, "document_splitter_service", splitter)
    monkeypatch.setattr(index_module, "vector_store_manager", store_manager)
    # 该用例专测 ISSUE-019 全量重建的删除顺序；增量路径的 diff 行为
    # 由 tests/rag/test_incremental_index.py 覆盖。
    monkeypatch.setattr(index_module.config, "incremental_index_enabled", False)

    service = index_module.VectorIndexService()
    result = service.index_single_file(str(file_path), allowed_root=upload_dir)

    assert result.doc_id == "doc_cpu"
    assert result.chunk_ids == ["doc_cpu#000000", "doc_cpu#000001"]
    assert result.deleted_count == 3
    assert store_manager.calls == [
        ("delete_by_doc_id", "doc_cpu"),
        ("delete_by_source", file_path.as_posix()),
        ("add_documents", ("doc_cpu#000000", "doc_cpu#000001")),
    ]


def test_index_directory_uses_allowlist_parent_for_aiops_doc_ids(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_module = importlib.import_module("app.services.vector_index_service")
    aiops_dir = tmp_path / "aiops-docs"
    aiops_dir.mkdir()
    file_path = aiops_dir / "cpu_high_usage.md"
    file_path.write_text("# CPU\n\nCPU usage is high.", encoding="utf-8")
    store_manager = RecordingIndexStoreManager()
    monkeypatch.setattr(index_module, "vector_store_manager", store_manager)
    monkeypatch.setattr(index_module.config, "allowed_upload_extensions", [".md"])
    expected = build_stable_rag_metadata(
        tenant_id="default",
        source_path="aiops-docs/cpu_high_usage.md",
        chunk_index=0,
        chunk_text="",
    )

    service = index_module.VectorIndexService()
    result = service.index_directory(str(aiops_dir), allowed_root=aiops_dir)

    assert result.success is True
    assert result.indexed_doc_ids == [expected["doc_id"]]
    assert store_manager.added_documents
    first_metadata = store_manager.added_documents[0].metadata
    assert first_metadata["source_path"] == "aiops-docs/cpu_high_usage.md"
    assert first_metadata["doc_id"] == expected["doc_id"]


def test_index_single_file_cleans_existing_chunks_when_splitter_returns_no_documents(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_module = importlib.import_module("app.services.vector_index_service")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    file_path = upload_dir / "empty.md"
    file_path.write_text("", encoding="utf-8")
    splitter = EmptyMetadataSplitter()
    store_manager = RecordingIndexStoreManager()
    monkeypatch.setattr(index_module, "document_splitter_service", splitter)
    monkeypatch.setattr(index_module, "vector_store_manager", store_manager)
    expected = build_stable_rag_metadata(
        tenant_id="default",
        source_path="uploads/empty.md",
        chunk_index=0,
        chunk_text="",
    )

    service = index_module.VectorIndexService()
    result = service.index_single_file(
        str(file_path),
        allowed_root=upload_dir,
        source_root=upload_dir.parent,
    )

    assert result.doc_id == expected["doc_id"]
    assert result.chunk_ids == []
    assert result.deleted_count == 3
    assert store_manager.calls == [
        ("delete_by_doc_id", expected["doc_id"]),
        ("delete_by_source", file_path.as_posix()),
    ]


def test_delete_by_source_maps_systemic_failures(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    def fail_get_collection() -> object:
        raise RuntimeError("milvus connection closed")

    manager = vector_store_module.VectorStoreManager()
    monkeypatch.setattr(milvus_module.milvus_manager, "get_collection", fail_get_collection)

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        manager.delete_by_source("uploads/cpu.md")

    assert exc_info.value.code == "VECTOR_STORE_UNAVAILABLE"


def test_index_directory_reports_partial_status_and_indexed_doc_ids(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_module = importlib.import_module("app.services.vector_index_service")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    good_file = upload_dir / "good.md"
    bad_file = upload_dir / "bad.md"
    good_file.write_text("# Good\n\nok", encoding="utf-8")
    bad_file.write_text("# Bad\n\nwill fail", encoding="utf-8")
    service = index_module.VectorIndexService()

    def fake_index_single_file(
        file_path: str,
        *,
        allowed_root: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> object:
        _ = allowed_root, source_root
        if Path(file_path).name == "bad.md":
            raise EmbeddingProviderError(internal_message="fake embedding failure")
        return index_module.SingleFileIndexResult(
            doc_id="doc_good",
            chunk_ids=["doc_good#000000"],
            deleted_count=1,
            source_deleted_count=0,
        )

    monkeypatch.setattr(service, "index_single_file", fake_index_single_file)

    result = service.index_directory(str(upload_dir))
    body = result.to_dict()

    assert result.success is False
    assert body["success"] is False
    assert body["status"] == "partial_success"
    assert body["partial_success"] is True
    assert body["failed_file_count"] == 1
    assert body["indexed_doc_ids"] == ["doc_good"]
    assert body["directory_path"] == str(upload_dir.resolve())
    assert body["total_files"] == 2
    assert body["success_count"] == 1
    assert body["fail_count"] == 1
    assert str(bad_file.resolve(strict=False)) in body["failed_files"]


def test_fake_vector_store_supports_idempotent_add_delete_and_search(
    fake_vector_store: object,
) -> None:
    """锁定 ISSUE-022 需要的内存向量库行为。

    后续索引测试会把该 fake 当作 `vector_store_manager` 替身使用，所以它必须同时支持
    稳定 chunk_id 写入、重复写入覆盖、doc-level delete、旧 `_source` delete 以及
    简单 search。测试先约束 fake 的语义，避免为了绕过真实 Milvus 而写出和索引链路
    不同形的测试替身。
    """

    first_version = Document(
        page_content="CPU old chunk",
        metadata={
            "doc_id": "doc_cpu",
            "chunk_id": "doc_cpu#000000",
            "source_path": "cpu.md",
            "_source": "uploads/cpu.md",
        },
    )
    second_version = Document(
        page_content="CPU new chunk",
        metadata={
            "doc_id": "doc_cpu",
            "chunk_id": "doc_cpu#000000",
            "source_path": "cpu.md",
            "_source": "uploads/cpu.md",
        },
    )
    disk_document = Document(
        page_content="Disk chunk",
        metadata={
            "doc_id": "doc_disk",
            "chunk_id": "doc_disk#000000",
            "source_path": "disk.md",
            "_source": "uploads/disk.md",
        },
    )

    assert fake_vector_store.add_documents([first_version]) == ["doc_cpu#000000"]
    assert fake_vector_store.add_documents([second_version, disk_document]) == [
        "doc_cpu#000000",
        "doc_disk#000000",
    ]
    assert [document.page_content for document in fake_vector_store.search("CPU", top_k=10)] == [
        "CPU new chunk"
    ]

    assert fake_vector_store.delete_by_doc_id("doc_cpu") == 1
    assert fake_vector_store.search("CPU", top_k=10) == []

    assert fake_vector_store.delete_by_source("uploads/disk.md") == 1
    assert fake_vector_store.search("Disk", top_k=10) == []


def test_index_directory_maps_fake_embedding_failure_to_failed_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    fake_vector_store: object,
    fake_embedding: object,
) -> None:
    """用 fake embedding 覆盖写入失败，不依赖真实 DashScope。

    ISSUE-022 要求“fake embedding 异常映射到失败文件”。这里通过 fake vector store
    的 `embedding` 钩子触发 `EmbeddingProviderError`，再断言目录索引保留旧响应字段
    `success_count/fail_count/failed_files`，避免错误直接冒泡或污染用户可见异常。
    """

    index_module = importlib.import_module("app.services.vector_index_service")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    file_path = upload_dir / "cpu.md"
    file_path.write_text("# CPU\n\nCPU usage", encoding="utf-8")
    splitter = FixedMetadataSplitter(doc_id="doc_cpu")
    fake_vector_store.embedding = fake_embedding(mode="error")
    monkeypatch.setattr(index_module, "document_splitter_service", splitter)
    monkeypatch.setattr(index_module, "vector_store_manager", fake_vector_store)

    service = index_module.VectorIndexService()
    result = service.index_directory(str(upload_dir))
    body = result.to_dict()

    assert body["success"] is False
    assert body["status"] == "failed"
    assert body["success_count"] == 0
    assert body["fail_count"] == 1
    assert body["failed_file_count"] == 1
    assert str(file_path.resolve(strict=False)) in body["failed_files"]
    assert body["failed_files"][str(file_path.resolve(strict=False))] == "向量化服务暂时不可用。"


def _reload_vector_store_manager_without_real_embedding(monkeypatch: MonkeyPatch) -> ModuleType:
    """重新导入向量存储模块，并用 fake embedding 替换会要求 DashScope key 的全局单例。"""

    embedding_module = ModuleType("app.services.vector_embedding_service")
    embedding_module.vector_embedding_service = object()
    monkeypatch.setitem(sys.modules, "app.services.vector_embedding_service", embedding_module)
    sys.modules.pop("app.services.vector_store_manager", None)
    return importlib.import_module("app.services.vector_store_manager")
