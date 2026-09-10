"""Milvus collection safety tests for phase 3A."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch


def test_existing_collection_dimension_mismatch_does_not_drop_collection(
    monkeypatch: MonkeyPatch,
) -> None:
    from app.core import milvus_client

    dropped: list[str] = []
    loaded: list[str] = []

    class FakeCollection:
        def __init__(self, name: str) -> None:
            self.name = name
            self.schema = SimpleNamespace(
                fields=[
                    SimpleNamespace(name="id", dtype=milvus_client.DataType.VARCHAR),
                    SimpleNamespace(
                        name="vector",
                        dtype=milvus_client.DataType.FLOAT_VECTOR,
                        params={"dim": 768},
                    ),
                    SimpleNamespace(name="content", dtype=milvus_client.DataType.VARCHAR),
                    SimpleNamespace(name="metadata", dtype=milvus_client.DataType.JSON),
                ]
            )

        def release(self) -> None:
            return None

        def load(self) -> None:
            loaded.append(self.name)

    monkeypatch.setattr(milvus_client.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(milvus_client.connections, "has_connection", lambda alias: False)
    monkeypatch.setattr(milvus_client, "MilvusClient", lambda uri: object())
    monkeypatch.setattr(milvus_client.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(milvus_client.utility, "drop_collection", lambda name: dropped.append(name))
    monkeypatch.setattr(milvus_client, "Collection", FakeCollection)

    manager = milvus_client.MilvusClientManager()

    with pytest.raises(RuntimeError) as exc_info:
        manager.connect()

    assert dropped == []
    assert loaded == []
    assert "vector dimension mismatch" in str(exc_info.value)


def test_existing_collection_missing_metadata_field_fails_schema_validation(
    monkeypatch: MonkeyPatch,
) -> None:
    from app.core import milvus_client

    class FakeCollection:
        def __init__(self, name: str) -> None:
            self.name = name
            self.schema = SimpleNamespace(
                fields=[
                    SimpleNamespace(name="id", dtype=milvus_client.DataType.VARCHAR),
                    SimpleNamespace(
                        name="vector",
                        dtype=milvus_client.DataType.FLOAT_VECTOR,
                        params={"dim": milvus_client.MilvusClientManager.VECTOR_DIM},
                    ),
                    SimpleNamespace(name="content", dtype=milvus_client.DataType.VARCHAR),
                ]
            )

        def release(self) -> None:
            return None

    monkeypatch.setattr(milvus_client.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(milvus_client.connections, "has_connection", lambda alias: False)
    monkeypatch.setattr(milvus_client, "MilvusClient", lambda uri: object())
    monkeypatch.setattr(milvus_client.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(milvus_client.utility, "drop_collection", lambda name: None)
    monkeypatch.setattr(milvus_client, "Collection", FakeCollection)

    manager = milvus_client.MilvusClientManager()

    with pytest.raises(RuntimeError) as exc_info:
        manager.connect()

    assert "metadata" in str(exc_info.value)


def _fake_collection_with_fields(fields: list[SimpleNamespace]) -> type:
    class FakeExistingCollection:
        def __init__(self, name: str) -> None:
            self.name = name
            self.schema = SimpleNamespace(fields=fields)

        def release(self) -> None:
            return None

        def load(self) -> None:
            return None

    return FakeExistingCollection


def _base_existing_fields(milvus_client) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(name="id", dtype=milvus_client.DataType.VARCHAR),
        SimpleNamespace(
            name="vector",
            dtype=milvus_client.DataType.FLOAT_VECTOR,
            params={"dim": milvus_client.MilvusClientManager.VECTOR_DIM},
        ),
        SimpleNamespace(name="content", dtype=milvus_client.DataType.VARCHAR),
        SimpleNamespace(name="metadata", dtype=milvus_client.DataType.JSON),
    ]


def _connect_existing_collection(
    monkeypatch: MonkeyPatch,
    milvus_client,
    fields: list[SimpleNamespace],
) -> "milvus_client.MilvusClientManager":
    """连接到带指定 fields 的既有 collection（不触发建表）。"""

    collection_cls = _fake_collection_with_fields(fields)
    monkeypatch.setattr(milvus_client.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(milvus_client.connections, "has_connection", lambda alias: False)
    monkeypatch.setattr(milvus_client, "MilvusClient", lambda uri: object())
    monkeypatch.setattr(milvus_client.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(milvus_client.utility, "load_state", lambda name: "NotLoad")
    monkeypatch.setattr(milvus_client, "Collection", collection_cls)
    manager = milvus_client.MilvusClientManager()
    manager.connect()
    return manager


def test_existing_collection_without_sparse_field_stays_dense_compatible(
    monkeypatch: MonkeyPatch,
) -> None:
    """旧 collection（无 sparse 字段）即使打开 hybrid 配置也不报错，只降级。"""

    from app.core import milvus_client

    monkeypatch.setattr(milvus_client.config, "rag_hybrid_search_enabled", True)
    manager = _connect_existing_collection(
        monkeypatch, milvus_client, _base_existing_fields(milvus_client)
    )

    assert manager.hybrid_supported() is False


def test_existing_collection_with_sparse_field_enables_hybrid(
    monkeypatch: MonkeyPatch,
) -> None:
    from app.core import milvus_client

    fields = _base_existing_fields(milvus_client)
    fields.append(
        SimpleNamespace(
            name=milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD,
            dtype=milvus_client.DataType.SPARSE_FLOAT_VECTOR,
        )
    )
    manager = _connect_existing_collection(monkeypatch, milvus_client, fields)

    assert manager.hybrid_supported() is True


def test_existing_collection_with_mistyped_sparse_field_degrades_to_dense(
    monkeypatch: MonkeyPatch,
) -> None:
    """sparse 字段类型异常属于未知中间态，按不支持处理而不是冒险走 hybrid。"""

    from app.core import milvus_client

    fields = _base_existing_fields(milvus_client)
    fields.append(
        SimpleNamespace(
            name=milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD,
            dtype=milvus_client.DataType.FLOAT_VECTOR,
        )
    )
    manager = _connect_existing_collection(monkeypatch, milvus_client, fields)

    assert manager.hybrid_supported() is False


def test_create_collection_with_hybrid_enabled_adds_bm25_function_and_sparse_index(
    monkeypatch: MonkeyPatch,
) -> None:
    from app.core import milvus_client

    monkeypatch.setattr(milvus_client.config, "rag_hybrid_search_enabled", True)

    created: dict[str, object] = {}

    class FakeNewCollection:
        def __init__(self, name: str, schema, num_shards: int) -> None:
            created["schema"] = schema
            created["index_calls"] = []
            self.schema = schema

        def create_index(self, field_name: str, index_params: dict) -> None:
            created["index_calls"].append((field_name, index_params))

        def load(self) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr(milvus_client.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(milvus_client.connections, "has_connection", lambda alias: False)
    monkeypatch.setattr(milvus_client, "MilvusClient", lambda uri: object())
    monkeypatch.setattr(milvus_client.utility, "has_collection", lambda name: False)
    monkeypatch.setattr(milvus_client.utility, "load_state", lambda name: "NotLoad")
    monkeypatch.setattr(milvus_client, "Collection", FakeNewCollection)

    manager = milvus_client.MilvusClientManager()
    manager.connect()

    schema = created["schema"]
    field_names = [field.name for field in schema.fields]
    assert milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD in field_names

    # content 字段开启 analyzer（jieba 中文分词），BM25 function 挂在 schema 上
    content_field = next(field for field in schema.fields if field.name == "content")
    content_params = getattr(content_field, "params", {}) or {}
    assert content_params.get("enable_analyzer") is True
    assert "jieba" in str(content_params.get("analyzer_params"))
    functions = getattr(schema, "functions", [])
    assert len(functions) == 1
    assert functions[0].name == "content_bm25"
    assert functions[0].input_field_names == ["content"]
    assert functions[0].output_field_names == [
        milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD
    ]

    # dense 与 sparse 各建一条索引
    index_calls = created["index_calls"]
    assert [call[0] for call in index_calls] == [
        "vector",
        milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD,
    ]
    assert index_calls[1][1]["metric_type"] == "BM25"

    assert manager.hybrid_supported() is True


def test_create_collection_without_hybrid_keeps_legacy_schema(
    monkeypatch: MonkeyPatch,
) -> None:
    """hybrid 配置关闭时建表保持旧 schema，不引入 analyzer/function/sparse 字段。"""

    from app.core import milvus_client

    monkeypatch.setattr(milvus_client.config, "rag_hybrid_search_enabled", False)

    created: dict[str, object] = {}

    class FakeNewCollection:
        def __init__(self, name: str, schema, num_shards: int) -> None:
            created["schema"] = schema
            created["index_calls"] = []
            self.schema = schema

        def create_index(self, field_name: str, index_params: dict) -> None:
            created["index_calls"].append((field_name, index_params))

        def load(self) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr(milvus_client.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(milvus_client.connections, "has_connection", lambda alias: False)
    monkeypatch.setattr(milvus_client, "MilvusClient", lambda uri: object())
    monkeypatch.setattr(milvus_client.utility, "has_collection", lambda name: False)
    monkeypatch.setattr(milvus_client.utility, "load_state", lambda name: "NotLoad")
    monkeypatch.setattr(milvus_client, "Collection", FakeNewCollection)

    manager = milvus_client.MilvusClientManager()
    manager.connect()

    schema = created["schema"]
    field_names = [field.name for field in schema.fields]
    assert milvus_client.MilvusClientManager.SPARSE_VECTOR_FIELD not in field_names
    content_field = next(field for field in schema.fields if field.name == "content")
    content_params = getattr(content_field, "params", {}) or {}
    assert "enable_analyzer" not in content_params
    assert "analyzer_params" not in content_params
    assert getattr(schema, "functions", []) == []

    # 只建 dense 索引
    assert [call[0] for call in created["index_calls"]] == ["vector"]

    assert manager.hybrid_supported() is False
