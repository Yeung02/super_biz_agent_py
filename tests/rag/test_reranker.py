"""DashScopeReranker / DashScopeTextReRankClient 单元测试。

只验证重排器自身的参数传递、响应解析、部分结果兜底和失败抛错行为；DashScope
客户端一律注入 fake，不连接真实服务或网络。失败抛错（而不是吞掉）是有意设计：
RagRetriever 依赖异常触发 fail-open 回退原始向量排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.rag.models import RetrievedChunk
from app.rag.reranker import DashScopeReranker


@dataclass(frozen=True)
class _FakeSearchResult:
    """贴近向量检索结果的最小 fake，供 RetrievedChunk.from_search_result 使用。"""

    id: str
    content: str
    score: float
    metadata: dict[str, object]


class FakeRerankClient:
    """返回预设 rerank 结果并记录调用参数的 fake DashScope 客户端。"""

    def __init__(
        self,
        results: list[dict[str, object]] | None = None,
        *,
        status_code: int = 200,
        fail: bool = False,
    ) -> None:
        self.results = results if results is not None else []
        self.status_code = status_code
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def rerank_documents(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
        api_key: str | None,
        request_timeout: float,
    ) -> object:
        self.calls.append(
            {
                "model": model,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
                "api_key": api_key,
                "request_timeout": request_timeout,
            }
        )
        if self.fail:
            raise RuntimeError("rerank client error token=secret http://internal.local")
        output = SimpleNamespace(
            results=[
                SimpleNamespace(index=item["index"], relevance_score=item["relevance_score"])
                for item in self.results
            ]
        )
        return SimpleNamespace(status_code=self.status_code, output=output)


def _chunk(chunk_id: str, content: str) -> RetrievedChunk:
    doc_id = chunk_id.split("#", maxsplit=1)[0]
    source_name = doc_id.removeprefix("doc_")
    result = _FakeSearchResult(
        id=chunk_id,
        content=content,
        score=0.10,
        metadata={
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "content_hash": f"hash-{chunk_id}",
            "tenant_id": "default",
            "version": 1,
            "source_path": f"aiops-docs/{source_name}.md",
            "file_name": f"{source_name}.md",
            "chunk_index": int(chunk_id.rsplit("#", maxsplit=1)[1]),
        },
    )
    return RetrievedChunk.from_search_result(result, metric="L2", normalized_score=0.90)


def test_dashscope_reranker_reorders_by_relevance_score_desc() -> None:
    client = FakeRerankClient(
        results=[
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.62},
            {"index": 1, "relevance_score": 0.31},
        ]
    )
    reranker = DashScopeReranker(client=client)
    chunks = [
        _chunk("doc_cpu#000000", "first"),
        _chunk("doc_cpu#000001", "second"),
        _chunk("doc_cpu#000002", "third"),
    ]

    reranked = reranker.rerank(query="CPU 使用率过高", chunks=chunks)

    assert [chunk.chunk_id for chunk in reranked] == [
        "doc_cpu#000002",
        "doc_cpu#000000",
        "doc_cpu#000001",
    ]


def test_dashscope_reranker_passes_config_and_documents_to_client() -> None:
    client = FakeRerankClient(results=[{"index": 0, "relevance_score": 0.9}])
    reranker = DashScopeReranker(
        client=client,
        model="qwen3-rerank",
        api_key="sk-test",
        timeout_seconds=4.5,
    )
    chunks = [_chunk("doc_cpu#000000", "CPU 排查手册"), _chunk("doc_mem#000000", "内存排查手册")]

    reranker.rerank(query="CPU 告警", chunks=chunks)

    assert client.calls == [
        {
            "model": "qwen3-rerank",
            "query": "CPU 告警",
            "documents": ["CPU 排查手册", "内存排查手册"],
            "top_n": 2,
            "api_key": "sk-test",
            "request_timeout": 4.5,
        }
    ]


def test_dashscope_reranker_appends_missing_indices_in_original_order() -> None:
    # 服务端只返回了部分索引：缺失项必须按原始顺序追加尾部，保证身份集合不变，
    # 不触发 RagRetriever 的 invalid_result 回退。
    client = FakeRerankClient(results=[{"index": 1, "relevance_score": 0.9}])
    reranker = DashScopeReranker(client=client)
    chunks = [
        _chunk("doc_cpu#000000", "first"),
        _chunk("doc_cpu#000001", "second"),
        _chunk("doc_cpu#000002", "third"),
    ]

    reranked = reranker.rerank(query="CPU", chunks=chunks)

    assert [chunk.chunk_id for chunk in reranked] == [
        "doc_cpu#000001",
        "doc_cpu#000000",
        "doc_cpu#000002",
    ]


def test_dashscope_reranker_skips_invalid_and_duplicate_entries() -> None:
    client = FakeRerankClient(
        results=[
            {"index": 0, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.99},
            {"index": 7, "relevance_score": 0.9},
            {"index": 1, "relevance_score": None},
        ]
    )
    reranker = DashScopeReranker(client=client)
    chunks = [_chunk("doc_cpu#000000", "first"), _chunk("doc_cpu#000001", "second")]

    reranked = reranker.rerank(query="CPU", chunks=chunks)

    # 重复索引取首个高分项；越界索引与缺失分数条目被忽略，其余按原序兜底。
    assert [chunk.chunk_id for chunk in reranked] == [
        "doc_cpu#000000",
        "doc_cpu#000001",
    ]


def test_dashscope_reranker_keeps_original_order_for_equal_scores() -> None:
    client = FakeRerankClient(
        results=[
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.5},
        ]
    )
    reranker = DashScopeReranker(client=client)
    chunks = [_chunk("doc_cpu#000000", "first"), _chunk("doc_cpu#000001", "second")]

    reranked = reranker.rerank(query="CPU", chunks=chunks)

    assert [chunk.chunk_id for chunk in reranked] == [
        "doc_cpu#000000",
        "doc_cpu#000001",
    ]


def test_dashscope_reranker_truncates_overlong_documents() -> None:
    client = FakeRerankClient(results=[{"index": 0, "relevance_score": 0.9}])
    reranker = DashScopeReranker(client=client, max_document_chars=10)
    chunks = [_chunk("doc_cpu#000000", "a" * 25)]

    reranker.rerank(query="CPU", chunks=chunks)

    assert client.calls[0]["documents"] == ["a" * 10]


def test_dashscope_reranker_returns_empty_list_without_client_call() -> None:
    client = FakeRerankClient()
    reranker = DashScopeReranker(client=client)

    assert reranker.rerank(query="CPU", chunks=[]) == []
    assert client.calls == []


def test_dashscope_reranker_raises_on_non_ok_status() -> None:
    client = FakeRerankClient(results=[], status_code=401)
    reranker = DashScopeReranker(client=client)
    chunks = [_chunk("doc_cpu#000000", "first")]

    with pytest.raises(ValueError):
        reranker.rerank(query="CPU", chunks=chunks)


def test_dashscope_reranker_raises_on_missing_results() -> None:
    reranker = DashScopeReranker(client=SimpleNamespace(rerank_documents=lambda **kwargs: SimpleNamespace(status_code=200)))
    chunks = [_chunk("doc_cpu#000000", "first")]

    with pytest.raises(ValueError):
        reranker.rerank(query="CPU", chunks=chunks)


def test_dashscope_reranker_propagates_client_exception_for_fail_open() -> None:
    client = FakeRerankClient(fail=True)
    reranker = DashScopeReranker(client=client)
    chunks = [_chunk("doc_cpu#000000", "first")]

    with pytest.raises(RuntimeError):
        reranker.rerank(query="CPU", chunks=chunks)


def test_dashscope_reranker_defaults_model_and_timeout_from_config() -> None:
    from app.config import config

    client = FakeRerankClient(results=[{"index": 0, "relevance_score": 0.9}])
    reranker = DashScopeReranker(client=client)

    reranker.rerank(query="CPU", chunks=[_chunk("doc_cpu#000000", "first")])

    assert client.calls[0]["model"] == config.reranker_model
    assert client.calls[0]["request_timeout"] == config.reranker_timeout_seconds
