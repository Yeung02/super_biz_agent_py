"""增量向量索引测试。

覆盖 content_hash 主键级 diff：未变更零写入、变更只增删受影响 chunk、首建 legacy
兜底清理、迁移期/开关回退全量路径，以及 VectorStoreManager 新增查询与精确删除。
全部用内存 fake，不连接真实 Milvus、DashScope 或网络。
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.documents import Document
from pytest import MonkeyPatch

from app.core.errors import VectorStoreUnavailableError

DOC_ID = "doc_cpu"


class ChunkSpecSplitter:
    """按构造参数返回带稳定身份 metadata 的分片，驱动增量 diff 测试。"""

    def __init__(self, chunks: Sequence[tuple[str, str]]) -> None:
        self.chunks = list(chunks)

    def split_document(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = "default",
    ) -> list[Document]:
        _ = content, source_root, tenant_id
        file_name = Path(file_path).name
        return [
            Document(
                page_content=text,
                metadata={
                    "doc_id": DOC_ID,
                    "chunk_id": f"{DOC_ID}#{index:06d}",
                    "content_hash": content_hash,
                    "tenant_id": tenant_id,
                    "version": 1,
                    "source_path": file_name,
                    "file_name": file_name,
                    "chunk_index": index,
                    "_source": file_path,
                },
            )
            for index, (text, content_hash) in enumerate(self.chunks)
        ]


class LegacySplitter:
    """缺 doc_id/content_hash 的迁移期分片，应回退全量重建路径。"""

    def split_document(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = "default",
    ) -> list[Document]:
        _ = content, source_root, tenant_id
        return [
            Document(
                page_content="legacy chunk",
                metadata={"_source": file_path, "_file_name": Path(file_path).name},
            )
        ]


class RecordingStoreManager:
    """记录索引服务对向量存储的调用，可注入已有 chunk hash 模拟 Milvus 现状。"""

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
        chunk_ids = [
            str(document.metadata["chunk_id"])
            for document in documents
            if isinstance(document.metadata.get("chunk_id"), str)
            and document.metadata["chunk_id"].strip()
        ]
        self.calls.append(("add_documents", tuple(chunk_ids)))
        return chunk_ids


class FailingQueryStoreManager(RecordingStoreManager):
    """get_chunk_hashes 失败时必须上抛，不允许静默降级为全量重建。"""

    def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
        self.calls.append(("get_chunk_hashes_by_doc_id", doc_id))
        raise VectorStoreUnavailableError(internal_message="milvus query failed")


def _setup_service(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    splitter: object,
    store_manager: RecordingStoreManager,
) -> tuple[object, Path, Path]:
    index_module = importlib.import_module("app.services.vector_index_service")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    file_path = upload_dir / "cpu.md"
    file_path.write_text("# CPU\n\nCPU usage", encoding="utf-8")
    monkeypatch.setattr(index_module, "document_splitter_service", splitter)
    monkeypatch.setattr(index_module, "vector_store_manager", store_manager)
    return index_module, upload_dir, file_path


def test_incremental_first_index_writes_all_chunks(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """首建（old 为空）全量写入，并执行一次 delete_by_source 清理 legacy 残留。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c1", "h1")])
    store_manager = RecordingStoreManager(old_hashes={})
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.doc_id == DOC_ID
    assert result.added_count == 2
    assert result.skipped_count == 0
    assert store_manager.calls == [
        ("get_chunk_hashes_by_doc_id", DOC_ID),
        ("delete_by_source", file_path.as_posix()),
        ("add_documents", (f"{DOC_ID}#000000", f"{DOC_ID}#000001")),
    ]


def test_incremental_unchanged_file_skips_all_writes(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """内容无变化时零删除、零写入、零 embedding，只发生一次现状查询。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c1", "h1")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 0
    assert result.skipped_count == 2
    assert result.deleted_count == 0
    assert result.chunk_ids == [f"{DOC_ID}#000000", f"{DOC_ID}#000001"]
    assert store_manager.calls == [("get_chunk_hashes_by_doc_id", DOC_ID)]


def test_incremental_single_chunk_change_rewrites_only_that_chunk(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """单 chunk 内容变更：只重写该片、只按主键删除被顶替的旧 chunk。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c1-changed", "h1-new")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 1
    assert result.skipped_count == 1
    assert store_manager.calls == [
        ("get_chunk_hashes_by_doc_id", DOC_ID),
        ("delete_by_chunk_ids", (f"{DOC_ID}#000001",)),
        ("add_documents", (f"{DOC_ID}#000001",)),
    ]


def test_incremental_tail_chunk_removed_deletes_without_adds(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """尾部段落删除：只删消失 chunk，无任何新增写入。"""

    splitter = ChunkSpecSplitter([("c0", "h0")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 0
    assert result.skipped_count == 1
    assert store_manager.calls == [
        ("get_chunk_hashes_by_doc_id", DOC_ID),
        ("delete_by_chunk_ids", (f"{DOC_ID}#000001",)),
    ]


def test_incremental_middle_insertion_shifts_suffix(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """中间插入：后缀 chunk_index 平移按变更处理（一删一写），前缀 skipped。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c-new", "h-new"), ("c1", "h1")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 2
    assert result.skipped_count == 1
    assert store_manager.calls == [
        ("get_chunk_hashes_by_doc_id", DOC_ID),
        ("delete_by_chunk_ids", (f"{DOC_ID}#000001",)),
        ("add_documents", (f"{DOC_ID}#000001", f"{DOC_ID}#000002")),
    ]


def test_incremental_falls_back_to_full_rebuild_for_legacy_documents(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """缺 doc_id 的 legacy 分片回退全量路径：不查询现状、按 _source 兜底清理。"""

    splitter = LegacySplitter()
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "stale"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 1
    assert result.skipped_count == 0
    assert store_manager.calls == [
        ("delete_by_source", file_path.as_posix()),
        ("add_documents", ()),
    ]
    # Legacy 分片没有 chunk_id，add_documents 记录的 id 列表为空元组
    assert len(store_manager.added_documents) == 1


def test_incremental_disabled_falls_back_to_full_rebuild(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """开关关闭时恢复 ISSUE-019 全量路径：doc_id 删除 → source 兜底 → 全量写入。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c1", "h1")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )
    monkeypatch.setattr(index_module.config, "incremental_index_enabled", False)

    result = index_module.VectorIndexService().index_single_file(
        str(file_path), allowed_root=upload_dir
    )

    assert result.added_count == 2
    assert result.skipped_count == 0
    assert store_manager.calls == [
        ("delete_by_doc_id", DOC_ID),
        ("delete_by_source", file_path.as_posix()),
        ("add_documents", (f"{DOC_ID}#000000", f"{DOC_ID}#000001")),
    ]


def test_incremental_query_failure_raises_without_fallback(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """现状查询失败必须上抛稳定错误码，不允许静默降级放大写入量。"""

    splitter = ChunkSpecSplitter([("c0", "h0")])
    store_manager = FailingQueryStoreManager()
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        index_module.VectorIndexService().index_single_file(
            str(file_path), allowed_root=upload_dir
        )

    assert exc_info.value.code == "VECTOR_STORE_UNAVAILABLE"
    assert store_manager.calls == [("get_chunk_hashes_by_doc_id", DOC_ID)]


def test_index_directory_aggregates_incremental_counts(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """目录级结果汇总 added/skipped/deleted 计数并进入 to_dict()。"""

    splitter = ChunkSpecSplitter([("c0", "h0"), ("c1", "h1")])
    store_manager = RecordingStoreManager(
        old_hashes={f"{DOC_ID}#000000": "h0", f"{DOC_ID}#000001": "h1"}
    )
    index_module, upload_dir, file_path = _setup_service(
        monkeypatch, tmp_path, splitter, store_manager
    )
    monkeypatch.setattr(index_module.config, "allowed_upload_extensions", [".md"])

    result = index_module.VectorIndexService().index_directory(
        str(upload_dir), allowed_root=upload_dir
    )

    assert result.success is True
    assert result.success_count == 1
    assert result.added_chunk_count == 0
    assert result.skipped_chunk_count == 2
    assert result.deleted_chunk_count == 0
    result_dict = result.to_dict()
    assert result_dict["added_chunk_count"] == 0
    assert result_dict["skipped_chunk_count"] == 2
    assert result_dict["deleted_chunk_count"] == 0


# ---------------------------------------------------------------------------
# VectorStoreManager 单元测试
# ---------------------------------------------------------------------------


class QueryableMilvusCollection:
    """记录 query/delete 表达式的 Milvus collection fake。"""

    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[tuple[str, list[str] | None]] = []
        self.delete_expressions: list[str] = []

    def query(
        self, expr: str, output_fields: list[str] | None = None
    ) -> list[SimpleNamespace]:
        self.queries.append((expr, output_fields))
        return self.rows

    def delete(self, expr: str) -> SimpleNamespace:
        self.delete_expressions.append(expr)
        return SimpleNamespace(delete_count=3)


def _reload_vector_store_manager_without_real_embedding(
    monkeypatch: MonkeyPatch,
) -> ModuleType:
    """重新导入向量存储模块，避免全局 embedding 单例要求 DashScope key。"""

    embedding_module = ModuleType("app.services.vector_embedding_service")
    embedding_module.vector_embedding_service = object()
    monkeypatch.setitem(sys.modules, "app.services.vector_embedding_service", embedding_module)
    sys.modules.pop("app.services.vector_store_manager", None)
    return importlib.import_module("app.services.vector_store_manager")


def test_get_chunk_hashes_parses_rows_and_skips_dirty_entries(
    monkeypatch: MonkeyPatch,
) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    collection = QueryableMilvusCollection(
        rows=[
            SimpleNamespace(
                metadata={"chunk_id": f" {DOC_ID}#000000 ", "content_hash": "h0"}
            ),
            SimpleNamespace(metadata={"chunk_id": f"{DOC_ID}#000001"}),
            SimpleNamespace(metadata={"content_hash": "h2"}),
            SimpleNamespace(metadata="not-a-dict"),
            SimpleNamespace(
                metadata={"chunk_id": f"{DOC_ID}#000002", "content_hash": "h2"}
            ),
        ]
    )
    monkeypatch.setattr(
        milvus_module.milvus_manager, "get_collection", lambda: collection
    )

    manager = vector_store_module.VectorStoreManager()

    assert manager.get_chunk_hashes_by_doc_id(DOC_ID) == {
        f"{DOC_ID}#000000": "h0",
        f"{DOC_ID}#000002": "h2",
    }
    assert collection.queries == [(f'metadata["doc_id"] == "{DOC_ID}"', ["metadata"])]


def test_get_chunk_hashes_maps_query_failures(monkeypatch: MonkeyPatch) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    class FailingCollection:
        def query(self, expr: str, output_fields: list[str] | None = None) -> list[object]:
            _ = expr, output_fields
            raise RuntimeError("milvus query timeout")

    monkeypatch.setattr(
        milvus_module.milvus_manager, "get_collection", lambda: FailingCollection()
    )
    manager = vector_store_module.VectorStoreManager()

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        manager.get_chunk_hashes_by_doc_id(DOC_ID)

    assert exc_info.value.code == "VECTOR_STORE_UNAVAILABLE"


def test_delete_by_chunk_ids_builds_primary_key_expression(
    monkeypatch: MonkeyPatch,
) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    collection = QueryableMilvusCollection()
    monkeypatch.setattr(
        milvus_module.milvus_manager, "get_collection", lambda: collection
    )
    manager = vector_store_module.VectorStoreManager()

    deleted_count = manager.delete_by_chunk_ids(
        [f"{DOC_ID}#000001", 'legacy"quote', "  "]
    )

    assert deleted_count == 3
    assert collection.delete_expressions == [
        f'id in ["{DOC_ID}#000001", "legacy\\"quote"]'
    ]


def test_delete_by_chunk_ids_with_empty_list_skips_milvus(
    monkeypatch: MonkeyPatch,
) -> None:
    vector_store_module = _reload_vector_store_manager_without_real_embedding(monkeypatch)
    from app.core import milvus_client as milvus_module

    collection = QueryableMilvusCollection()
    monkeypatch.setattr(
        milvus_module.milvus_manager, "get_collection", lambda: collection
    )
    manager = vector_store_module.VectorStoreManager()

    assert manager.delete_by_chunk_ids([]) == 0
    assert collection.delete_expressions == []
