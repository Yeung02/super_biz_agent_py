"""VectorSearchService 混合检索（dense + BM25 hybrid_search）单元测试。

全部 mock：不连接真实 Milvus、不调用 DashScope embedding。AnnSearchRequest /
RRFRanker 使用 pymilvus 真实数据对象（纯构造不触网），以验证调用参数与
hybrid 失败回退行为。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from app.services import vector_search_service as vss_module
from app.services.vector_search_service import VectorSearchService


class FakeHit:
    def __init__(self, hit_id: str, content: str, distance: float) -> None:
        self.distance = distance
        self._entity = {
            "id": hit_id,
            "content": content,
            "metadata": {"doc_id": f"doc-{hit_id}"},
        }

    @property
    def entity(self) -> dict[str, object]:
        return SimpleNamespace(get=self._entity.get)


class FakeHybridCollection:
    """记录 hybrid_search / search 调用并返回预设结果的 fake collection。"""

    def __init__(
        self,
        *,
        hybrid_hits: list[FakeHit] | None = None,
        dense_hits: list[FakeHit] | None = None,
        hybrid_error: Exception | None = None,
    ) -> None:
        self.hybrid_hits = hybrid_hits or []
        self.dense_hits = dense_hits or []
        self.hybrid_error = hybrid_error
        self.hybrid_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        if self.hybrid_error is not None:
            raise self.hybrid_error
        return [list(self.hybrid_hits)]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [list(self.dense_hits)]


class FakeMilvusManager:
    def __init__(self, *, hybrid_supported: bool) -> None:
        self._hybrid_supported = hybrid_supported
        self.collection = FakeHybridCollection()
        # 与真实 MilvusClientManager 保持同一字段名
        self.SPARSE_VECTOR_FIELD = "sparse_vector"

    def get_collection(self) -> FakeHybridCollection:
        return self.collection

    def hybrid_supported(self) -> bool:
        return self._hybrid_supported


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3, 0.4]


def _setup(
    monkeypatch: MonkeyPatch,
    *,
    hybrid_config_enabled: bool,
    collection_supported: bool,
    rrf_k: int = 60,
    hybrid_error: Exception | None = None,
) -> tuple[FakeMilvusManager, FakeEmbeddingService]:
    manager = FakeMilvusManager(hybrid_supported=collection_supported)
    manager.collection = FakeHybridCollection(hybrid_error=hybrid_error)
    embedding = FakeEmbeddingService()

    monkeypatch.setattr(vss_module.config, "rag_hybrid_search_enabled", hybrid_config_enabled)
    monkeypatch.setattr(vss_module.config, "rag_hybrid_search_rrf_k", rrf_k)
    monkeypatch.setattr(vss_module, "milvus_manager", manager)
    monkeypatch.setattr(vss_module, "vector_embedding_service", embedding)
    return manager, embedding


def test_hybrid_search_issues_dense_and_bm25_requests(monkeypatch: MonkeyPatch) -> None:
    manager, embedding = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=True,
        rrf_k=60,
    )
    manager.collection.hybrid_hits = [
        FakeHit("doc_cpu#000001", "CPU 排查手册", 2.0 / 61.0),
    ]

    results = VectorSearchService().search_similar_documents("CPU 使用率过高", top_k=5)

    # 只发起一次 hybrid_search（不额外触发 dense search）
    assert len(manager.collection.hybrid_calls) == 1
    assert manager.collection.search_calls == []
    assert embedding.queries == ["CPU 使用率过高"]

    call = manager.collection.hybrid_calls[0]
    assert call["limit"] == 5
    assert call["output_fields"] == ["id", "content", "metadata"]
    assert call["expr"] == ""

    # 请求包含 dense（向量）与 BM25（原始文本）两路
    reqs = call["reqs"]
    assert len(reqs) == 2
    dense_req, sparse_req = reqs
    assert dense_req.data == [[0.1, 0.2, 0.3, 0.4]]
    assert dense_req.anns_field == "vector"
    assert dense_req.param["metric_type"] == "L2"
    assert sparse_req.data == ["CPU 使用率过高"]
    assert sparse_req.anns_field == "sparse_vector"
    assert sparse_req.param["metric_type"] == "BM25"

    # RRF 融合分 + 归一化分数（两路 rank0 理论上限 2/61 → 1.0）
    assert len(results) == 1
    assert results[0].metric == "RRF"
    assert results[0].score == 2.0 / 61.0
    assert results[0].normalized_score == 1.0


def test_hybrid_search_normalizes_rrf_score_to_shared_scale(monkeypatch: MonkeyPatch) -> None:
    """仅单路命中的 RRF 分数应映射为 0-1 内的相对置信度，而非原始 0.0x 小数。"""

    manager, _ = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=True,
        rrf_k=60,
    )
    # 只有 dense 路 rank0 命中：RRF 分 = 1/61
    manager.collection.hybrid_hits = [FakeHit("doc_cpu#000002", "内存泄漏", 1.0 / 61.0)]

    results = VectorSearchService().search_similar_documents("内存泄漏", top_k=3)

    assert results[0].normalized_score == pytest.approx(0.5)
    assert results[0].score == 1.0 / 61.0


def test_hybrid_search_failure_falls_back_to_dense(monkeypatch: MonkeyPatch) -> None:
    manager, _ = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=True,
        hybrid_error=RuntimeError("bm25 index not found"),
    )
    manager.collection.dense_hits = [
        FakeHit("doc_cpu#000003", "CPU 排查手册", 0.42),
    ]

    results = VectorSearchService().search_similar_documents("CPU 使用率过高", top_k=5)

    # hybrid 失败后回退 dense：一次 hybrid 失败 + 一次 dense 成功
    assert len(manager.collection.hybrid_calls) == 1
    assert len(manager.collection.search_calls) == 1
    assert results[0].id == "doc_cpu#000003"
    assert results[0].metric == "L2"
    assert results[0].score == 0.42
    # dense 回退结果不带预归一化分数，由 RagRetriever 的 normalize_score 处理
    assert results[0].normalized_score is None


def test_hybrid_disabled_by_config_uses_dense(monkeypatch: MonkeyPatch) -> None:
    manager, _ = _setup(
        monkeypatch,
        hybrid_config_enabled=False,
        collection_supported=True,
    )
    manager.collection.dense_hits = [FakeHit("doc_cpu#000004", "CPU 排查手册", 0.30)]

    results = VectorSearchService().search_similar_documents("CPU 使用率过高", top_k=5)

    assert manager.collection.hybrid_calls == []
    assert len(manager.collection.search_calls) == 1
    assert results[0].metric == "L2"


def test_hybrid_unsupported_collection_uses_dense(monkeypatch: MonkeyPatch) -> None:
    """collection 未迁移（无 sparse 字段）时即使配置打开也走 dense。"""

    manager, _ = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=False,
    )
    manager.collection.dense_hits = [FakeHit("doc_cpu#000005", "CPU 排查手册", 0.31)]

    results = VectorSearchService().search_similar_documents("CPU 使用率过高", top_k=5)

    assert manager.collection.hybrid_calls == []
    assert len(manager.collection.search_calls) == 1
    assert results[0].metric == "L2"


def test_hybrid_search_forwards_metadata_filter_expr(monkeypatch: MonkeyPatch) -> None:
    manager, _ = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=True,
    )

    VectorSearchService().search_similar_documents(
        "CPU 使用率过高",
        top_k=5,
        filters={"tenant_id": "tenant-1"},
    )

    call = manager.collection.hybrid_calls[0]
    assert call["expr"] == 'metadata["tenant_id"] == "tenant-1"'


def test_hybrid_search_dense_failure_raises_stable_error(monkeypatch: MonkeyPatch) -> None:
    """dense 路径（含 embedding）失败仍然抛稳定 VECTOR_STORE_UNAVAILABLE。"""

    manager, embedding = _setup(
        monkeypatch,
        hybrid_config_enabled=True,
        collection_supported=True,
    )

    def _fail_embed(text: str) -> list[float]:
        raise RuntimeError("dashscope quota exceeded")

    embedding.embed_query = _fail_embed

    from app.core.errors import VectorStoreUnavailableError

    try:
        VectorSearchService().search_similar_documents("CPU 使用率过高", top_k=5)
        raised = False
    except VectorStoreUnavailableError:
        raised = True

    assert raised is True
