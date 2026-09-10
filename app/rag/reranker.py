"""RAG reranker 插口与 DashScope qwen3-rerank 实现。

ISSUE-027 定义重排边界和默认 NoopReranker；本模块在其上接入真实重排模型
`DashScopeReranker`（DashScope qwen3-rerank，SDK 调用方式见官方文档
https://help.aliyun.com/zh/model-studio/developer-reference/general-text-sorting-model/）。
reranker 只能改变候选 chunk 顺序；任何失败（网络/鉴权/超时/响应异常）都直接抛给
RagRetriever，由其 fail-open 回退到原始向量检索排序，避免重排能力成为基础 RAG
pipeline 的阻塞点。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol

from loguru import logger

from app.config import config
from app.core.request_context import RequestContext
from app.rag.models import RetrievedChunk

# qwen3-rerank 单条文档输入上限为 4000 token；chunk 切分上限 1600 字符通常不会
# 触达该限制，这里做客户端截断兜底，避免服务端静默截断导致排序依据不完整。
_MAX_DOCUMENT_CHARS = 2000


class RerankerLike(Protocol):
    """RagRetriever 依赖的 reranker 最小协议。

    使用协议是为了让测试和未来真实 reranker 可以通过依赖注入接入，而不需要提前
    绑定某个模型 SDK。接口显式携带 query 与 RequestContext，方便后续成本、trace
    和权限边界接入；当前 Noop 实现不会使用这些参数。
    """

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        """返回重排后的 chunk 列表。"""


class BaseReranker(ABC):
    """Reranker 抽象基类。

    阶段 3B 只要求“可插拔且默认关闭”。抽象类的职责是稳定方法签名，避免未来真实
    reranker 把过滤、检索或 citation 逻辑塞进重排层，破坏模块边界。
    """

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        """按 query 对候选 chunks 重新排序。"""


class NoopReranker(BaseReranker):
    """默认空实现，保持向量检索原始排序。

    返回新的 list 而不是原列表对象，是为了让调用方可以安全地截断或局部修改返回值，
    同时不影响上游保留的原始 vector order 回退副本。
    """

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        _ = query, ctx
        return list(chunks)


class RerankClientLike(Protocol):
    """DashScopeReranker 依赖的最小 rerank 客户端协议。

    真实实现是 dashscope SDK 的 `TextReRank.call` 薄封装；单元测试注入 fake，
    不触网。协议显式携带 model/api_key/request_timeout，让成本、鉴权和超时边界
    都留在 reranker 层配置。
    """

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
        """调用 rerank 服务，返回 DashScope TextReRank 响应形态对象。"""


class DashScopeTextReRankClient:
    """dashscope SDK TextReRank 客户端薄封装。

    官方文档的 SDK 调用示例即 `dashscope.TextReRank.call(model="qwen3-rerank",
    ...)`；SDK 会处理鉴权与请求封装。`api_key` 传空时回退 SDK 默认取值规则
    （环境变量 DASHSCOPE_API_KEY），避免空字符串被当作真实 key 导致鉴权失败。
    """

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
        from dashscope import TextReRank

        return TextReRank.call(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            api_key=api_key or None,
            request_timeout=request_timeout,
        )


class DashScopeReranker(BaseReranker):
    """基于 DashScope qwen3-rerank 的真实重排实现。

    客户端延迟构造且可注入：`client=None` 时首次调用才创建 SDK 客户端，构造本身
    不触网，方便单元测试注入 fake。重排只按 relevance_score 改变顺序，不增删
    chunk；部分索引缺失时按原始顺序追加尾部，保证身份集合与输入一致，避免触发
    RagRetriever 的 invalid_result 回退。任何异常都直接抛出，由 RagRetriever
    fail-open 回退原始向量排序。
    """

    def __init__(
        self,
        *,
        client: RerankClientLike | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_document_chars: int = _MAX_DOCUMENT_CHARS,
    ) -> None:
        self.client = client
        self.model = model or config.reranker_model
        self.api_key = api_key if api_key is not None else config.dashscope_api_key
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.reranker_timeout_seconds
        )
        self.max_document_chars = max_document_chars

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        ctx: RequestContext | None = None,
    ) -> list[RetrievedChunk]:
        _ = ctx
        if not chunks:
            return list(chunks)
        documents = [chunk.content[: self.max_document_chars] for chunk in chunks]
        response = self._get_client().rerank_documents(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
            api_key=self.api_key,
            request_timeout=self.timeout_seconds,
        )
        ranked_indices = _ranked_indices(response, total=len(chunks))
        logger.debug(
            "DashScopeReranker reranked {} chunks with model={}",
            len(chunks),
            self.model,
        )
        return [chunks[index] for index in ranked_indices]

    def _get_client(self) -> RerankClientLike:
        if self.client is None:
            self.client = DashScopeTextReRankClient()
        return self.client


def _ranked_indices(response: object, *, total: int) -> list[int]:
    """从 DashScope rerank 响应提取按相关性降序的原始索引序列。

    服务端正常返回按 relevance_score 降序的全部结果；这里仍显式排序（同分按原始
    索引升序）保证确定性。非法条目（索引越界、缺分数、重复索引）跳过；未返回的
    索引按原始顺序追加尾部，确保输出索引集合与输入完全一致。
    """

    results = _extract_results(response)
    scored: list[tuple[float, int]] = []
    for item in results:
        index = _result_index(item)
        score = _result_score(item)
        if index is None or score is None or not 0 <= index < total:
            continue
        scored.append((score, index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    ranked: list[int] = []
    seen: set[int] = set()
    for _, index in scored:
        if index in seen:
            continue
        seen.add(index)
        ranked.append(index)
    ranked.extend(index for index in range(total) if index not in seen)
    return ranked


def _extract_results(response: object) -> list[object]:
    """提取响应中的 results 列表，兼容属性与 Mapping 两种访问形态。"""

    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and status_code != 200:
        # 不携带 provider 返回的 code/message，避免内部错误细节进入日志/trace。
        raise ValueError("rerank provider returned non-OK status")
    output = getattr(response, "output", None)
    if output is None and isinstance(response, Mapping):
        output = response.get("output")
    results = getattr(output, "results", None)
    if results is None and isinstance(output, Mapping):
        results = output.get("results")
    if not isinstance(results, list):
        raise ValueError("rerank response missing output.results")
    return results


def _result_index(item: object) -> int | None:
    value = getattr(item, "index", None)
    if value is None and isinstance(item, Mapping):
        value = item.get("index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _result_score(item: object) -> float | None:
    value = getattr(item, "relevance_score", None)
    if value is None and isinstance(item, Mapping):
        value = item.get("relevance_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
