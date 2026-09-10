"""向量检索服务模块"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from loguru import logger
from pymilvus import AnnSearchRequest, Collection, RRFRanker

from app.config import config
from app.core.errors import JsonValue, VectorStoreUnavailableError
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service

_METADATA_FILTER_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")

# hybrid_search 返回的 RRF 融合分 metric 标识；RagRetriever/citation 据此区分
# 分数语义（越大越相关，而非 L2 越小越相关）。
HYBRID_RRF_METRIC = "RRF"


class SearchResult:
    """搜索结果类。

    `score` 继续保留旧字段名以兼容已存在调用方；在 ISSUE-023 之后它明确表示
    `raw_score`，`metric` 记录原始分数语义，避免后续 citation 把 L2 距离误当相似度。
    """

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Mapping[str, object],
        *,
        metric: str = "L2",
        normalized_score: float | None = None,
    ) -> None:
        self.id = id
        self.content = content
        self.score = score
        self.metadata = dict(metadata)
        self.metric = metric
        self.normalized_score = normalized_score

    def to_dict(self) -> dict[str, object]:
        """转换为字典。"""

        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "raw_score": self.score,
            "metric": self.metric,
            "normalized_score": self.normalized_score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """向量检索服务 - 负责从 Milvus 中搜索相似向量"""

    def __init__(self) -> None:
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: Mapping[str, JsonValue] | None = None,
    ) -> list[SearchResult]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回最相似的K个结果
            filters: metadata 过滤条件，供 RagRetriever 控制租户、文档等候选范围

        Returns:
            list[SearchResult]: 搜索结果列表

        Raises:
            VectorStoreUnavailableError: 搜索失败时抛出稳定错误
        """
        try:
            logger.info(f"开始搜索相似文档, 查询: {query}, topK: {top_k}")

            # 1. 将查询文本向量化
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功, 维度: {len(query_vector)}")

            # 2. 获取 collection
            collection: Collection = milvus_manager.get_collection()

            # 3. metadata 过滤表达式（dense 与 hybrid 共用）
            filter_expr = _build_metadata_filter_expr(filters)

            # 4. 混合检索：dense（语义）+ BM25（关键词）服务端 RRF 融合。
            #    配置关闭、collection 未迁移或 hybrid 调用失败时回退纯 dense。
            if self._hybrid_ready():
                hybrid_results = self._hybrid_search(
                    collection,
                    query=query,
                    query_vector=query_vector,
                    top_k=top_k,
                    filter_expr=filter_expr,
                )
                if hybrid_results is not None:
                    return hybrid_results

            return self._dense_search(
                collection,
                query_vector=query_vector,
                top_k=top_k,
                filter_expr=filter_expr,
            )

        except Exception as exc:
            logger.error(f"搜索相似文档失败: {exc.__class__.__name__}")
            # 对外和上层 RAG 只暴露稳定错误码；原始异常通过异常链保留给日志/调试，
            # 不把 str(exc) 拼进用户可见错误，避免泄露内部 URL、密钥或 Milvus 细节。
            raise VectorStoreUnavailableError(
                internal_message="Vector search failed",
            ) from exc

    @staticmethod
    def _hybrid_ready() -> bool:
        """hybrid 检索是否可用：配置开启且 collection 已迁移出 sparse 字段。"""

        return bool(config.rag_hybrid_search_enabled and milvus_manager.hybrid_supported())

    def _hybrid_search(
        self,
        collection: Collection,
        *,
        query: str,
        query_vector: list[float],
        top_k: int,
        filter_expr: str | None,
    ) -> list[SearchResult] | None:
        """执行 dense + BM25 混合检索；失败时返回 None 由调用方回退 dense。

        与 reranker/查询改写一致的 fail-open 策略：混合检索是增强能力，不能因
        BM25 索引缺失、服务端版本不支持等问题阻塞基础检索链路。
        """

        rrf_k = max(0, int(config.rag_hybrid_search_rrf_k))
        try:
            requests = [
                AnnSearchRequest(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "L2", "params": {"nprobe": 10}},
                    limit=top_k,
                ),
                # BM25 查询直接传原始文本，由 Milvus 服务端 analyzer（jieba）分词。
                AnnSearchRequest(
                    data=[query],
                    anns_field=milvus_manager.SPARSE_VECTOR_FIELD,
                    param={"metric_type": "BM25", "params": {}},
                    limit=top_k,
                ),
            ]
            search_kwargs: dict[str, object] = {
                "reqs": requests,
                "rerank": RRFRanker(rrf_k),
                "limit": top_k,
                "output_fields": ["id", "content", "metadata"],
                "expr": filter_expr or "",
            }
            results = collection.hybrid_search(**search_kwargs)
        except Exception as exc:
            logger.warning(
                "混合检索失败，回退纯 dense 检索: error_type={}", exc.__class__.__name__
            )
            return None

        # RRF 融合分归一化：两路都排第一时得分 2/(k+1)，以此为理论上限映射到 0-1，
        # 与 L2 归一化分数同一尺度，让 min_score 阈值与 citation 置信度语义保持一致。
        theoretical_max = 2.0 / (rrf_k + 1)
        search_results: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                normalized = (
                    min(max(hit.distance / theoretical_max, 0.0), 1.0)
                    if theoretical_max > 0
                    else 0.0
                )
                search_results.append(
                    SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # RRF 融合分，越大越相似
                        metadata=hit.entity.get("metadata", {}),
                        metric=HYBRID_RRF_METRIC,
                        normalized_score=normalized,
                    )
                )
        logger.info(
            "混合检索完成, 找到 {} 个相似文档 (dense+BM25, rrf_k={})",
            len(search_results),
            rrf_k,
        )
        return search_results

    @staticmethod
    def _dense_search(
        collection: Collection,
        *,
        query_vector: list[float],
        top_k: int,
        filter_expr: str | None,
    ) -> list[SearchResult]:
        """纯 dense 向量检索（原有路径，行为不变）。"""

        search_params = {
            "metric_type": "L2",  # 欧氏距离
            "params": {"nprobe": 10},
        }
        search_kwargs: dict[str, object] = {
            "data": [query_vector],
            "anns_field": "vector",
            "param": search_params,
            "limit": top_k,
            "output_fields": ["id", "content", "metadata"],
        }
        if filter_expr:
            # PyMilvus 使用 expr 做服务端过滤；表达式只由安全 key 和 JSON literal
            # 拼出，避免把上层传入的原始 filter 文本直接透传到 Milvus 查询语法。
            search_kwargs["expr"] = filter_expr
        results = collection.search(**search_kwargs)

        search_results: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                result = SearchResult(
                    id=hit.entity.get("id"),
                    content=hit.entity.get("content"),
                    score=hit.distance,  # L2 距离，越小越相似
                    metadata=hit.entity.get("metadata", {}),
                    metric=search_params["metric_type"],
                )
                search_results.append(result)

        logger.info(f"搜索完成, 找到 {len(search_results)} 个相似文档")
        return search_results


def _build_metadata_filter_expr(filters: Mapping[str, JsonValue] | None) -> str | None:
    """把安全 metadata filters 转为 Milvus expr。

    当前只支持标量相等和标量列表 in 查询，满足 ISSUE-023 的 tenant/doc 过滤入口。
    dense 与 hybrid 检索共用本函数；更复杂的范围查询、布尔组合不在当前范围，避免把
    检索入口扩成不可控 DSL。
    """

    if not filters:
        return None

    clauses: list[str] = []
    for key, value in filters.items():
        if value is None:
            continue
        if not _METADATA_FILTER_KEY_PATTERN.fullmatch(key):
            raise ValueError("metadata filter key is not allowed")

        field_expr = f'metadata["{key}"]'
        if isinstance(value, list):
            literals = [_metadata_filter_literal(item) for item in value]
            if literals:
                clauses.append(f"{field_expr} in [{', '.join(literals)}]")
            continue

        clauses.append(f"{field_expr} == {_metadata_filter_literal(value)}")

    return " and ".join(clauses) if clauses else None


def _metadata_filter_literal(value: JsonValue) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    raise ValueError("metadata filter value is not supported")


# 全局单例
vector_search_service = VectorSearchService()
