"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

import uuid

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger
from pymilvus import connections

from app.config import config
from app.core.errors import AppError, EmbeddingProviderError, VectorStoreUnavailableError
from app.services.vector_embedding_service import vector_embedding_service

# 统一使用 biz collection
COLLECTION_NAME = "biz"


class ConnectedMilvus(Milvus):
    """Milvus wrapper that registers the ORM alias used by Collection()."""

    @property
    def col(self):
        if self.alias and not connections.has_connection(self.alias):
            connections.connect(
                alias=self.alias,
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,
            )
        return Milvus.col.fget(self)

    @col.setter
    def col(self, value):
        Milvus.col.fset(self, value)


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            self.vector_store = ConnectedMilvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector",  # 向量存储到 vector 字段
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error("VectorStore 初始化失败: code=VECTOR_STORE_UNAVAILABLE")
            raise VectorStoreUnavailableError(
                internal_message=f"{e.__class__.__name__}: {e}",
            ) from e

    def _ensure_vector_store(self) -> Milvus:
        """Lazy initialize VectorStore so importing API modules never connects to Milvus."""

        if self.vector_store is None:
            self._initialize_vector_store()
        if self.vector_store is None:
            raise VectorStoreUnavailableError(
                internal_message="VectorStore initialization returned no instance",
            )
        return self.vector_store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            start_time = time.time()

            if not documents:
                return []

            ids = self._build_document_ids(documents)

            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            vector_store = self._ensure_vector_store()
            result_ids = vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids
        except AppError:
            raise
        except Exception as e:
            app_error = _map_vector_add_error(e)
            logger.error(
                "批量添加文档失败，已映射为稳定错误码: code={}, error_type={}",
                app_error.code,
                e.__class__.__name__,
            )
            raise app_error from e

    def _build_document_ids(self, documents: list[Document]) -> list[str]:
        """为 Milvus primary id 选择稳定 chunk_id 或旧 UUID。

        `auto_id=False` 要求调用方显式传入主键。ISSUE-018 默认使用分割器写入的
        `metadata["chunk_id"]`，这样同一 tenant/source/chunk_index 重复索引会得到
        稳定 ID；如果配置关闭或调用方传入的旧 Document 没有 chunk_id，则回退 UUID，
        避免一次性破坏已有内部调用。真正的幂等删除和残留清理由 ISSUE-019 负责。
        """

        if not config.stable_rag_ids_enabled:
            return self._uuid_document_ids(len(documents))

        ids: list[str] = []
        fallback_count = 0
        for document in documents:
            chunk_id = document.metadata.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id.strip():
                ids.append(chunk_id.strip())
                continue
            fallback_count += 1
            ids.append(str(uuid.uuid4()))

        if fallback_count:
            logger.warning(
                "部分文档缺少稳定 chunk_id，已回退 UUID: "
                f"fallback_count={fallback_count}, total={len(documents)}"
            )
        return ids

    @staticmethod
    def _uuid_document_ids(count: int) -> list[str]:
        """生成旧 UUID 主键，用于 `stable_rag_ids_enabled=false` 回滚路径。"""

        return [str(uuid.uuid4()) for _ in range(count)]

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            safe_file_path = _escape_milvus_string(file_path)
            expr = f'metadata["_source"] == "{safe_file_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.error(
                "删除旧数据失败: source={}, error_type={}",
                file_path,
                e.__class__.__name__,
            )
            raise VectorStoreUnavailableError(
                internal_message=f"{e.__class__.__name__}: {e}",
            ) from e

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按稳定 doc_id 删除一个文档的全部 chunk。

        ISSUE-019 把 doc_id 作为删除主路径，是为了避免同一文件重建索引时旧 chunk
        残留。旧数据仍可能只有 `_source`，因此调用方还会继续执行 `delete_by_source`
        做兼容清理；但 doc_id 删除失败不能静默吞掉，否则幂等索引会失效并污染后续
        citation/eval，所以这里映射为稳定 `VECTOR_STORE_UNAVAILABLE`。
        """

        safe_doc_id = _escape_milvus_string(doc_id)
        if not safe_doc_id:
            return 0

        try:
            from app.core.milvus_client import milvus_manager

            collection = milvus_manager.get_collection()
            expr = f'metadata["doc_id"] == "{safe_doc_id}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info("按 doc_id 删除旧 chunk: doc_id={}, deleted_count={}", doc_id, deleted_count)
            return int(deleted_count)
        except Exception as e:
            logger.error(
                "按 doc_id 删除旧 chunk 失败: doc_id={}, error_type={}",
                doc_id,
                e.__class__.__name__,
            )
            raise VectorStoreUnavailableError(
                internal_message=f"{e.__class__.__name__}: {e}",
            ) from e

    def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
        """查询 doc 下已有 chunk 的 {chunk_id: content_hash}。

        增量索引的 diff 基础：返回空 dict 表示首建，或该 doc 仅有 legacy 数据
        （legacy chunk 无 doc_id，按 doc_id 查询不可见）。迁移期脏条目（缺
        chunk_id/content_hash 或类型不对）直接跳过，不阻塞索引；查询失败映射为
        稳定 `VECTOR_STORE_UNAVAILABLE`，不静默吞错，否则增量 diff 会在错误数据
        上做判定并污染索引。
        """

        safe_doc_id = _escape_milvus_string(doc_id)
        if not safe_doc_id:
            return {}

        try:
            from app.core.milvus_client import milvus_manager

            collection = milvus_manager.get_collection()
            expr = f'metadata["doc_id"] == "{safe_doc_id}"'
            rows = collection.query(expr=expr, output_fields=["metadata"])

            chunk_hashes: dict[str, str] = {}
            for row in rows:
                metadata = getattr(row, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                chunk_id = metadata.get("chunk_id")
                content_hash = metadata.get("content_hash")
                if (
                    isinstance(chunk_id, str)
                    and chunk_id.strip()
                    and isinstance(content_hash, str)
                    and content_hash.strip()
                ):
                    chunk_hashes[chunk_id.strip()] = content_hash.strip()

            logger.info(
                "查询 doc 已有 chunk hash: doc_id={}, count={}", doc_id, len(chunk_hashes)
            )
            return chunk_hashes
        except Exception as e:
            logger.error(
                "查询 doc 已有 chunk hash 失败: doc_id={}, error_type={}",
                doc_id,
                e.__class__.__name__,
            )
            raise VectorStoreUnavailableError(
                internal_message=f"{e.__class__.__name__}: {e}",
            ) from e

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """按主键 id in [...] 精确删除指定 chunk，返回删除数量。

        增量索引只删除消失或被变更顶替的旧 chunk；相比 `delete_by_doc_id` 的
        doc 级表达式，主键精确删除不会误删未变更 chunk。空列表直接返回 0，
        不产生 Milvus 调用。
        """

        safe_ids = [
            _escape_milvus_string(chunk_id)
            for chunk_id in chunk_ids
            if isinstance(chunk_id, str) and chunk_id.strip()
        ]
        if not safe_ids:
            return 0

        try:
            from app.core.milvus_client import milvus_manager

            collection = milvus_manager.get_collection()
            id_list = ", ".join(f'"{safe_id}"' for safe_id in safe_ids)
            expr = f"id in [{id_list}]"
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(
                "按 chunk_id 精确删除旧 chunk: count={}, deleted_count={}",
                len(safe_ids),
                deleted_count,
            )
            return int(deleted_count)
        except Exception as e:
            logger.error(
                "按 chunk_id 精确删除失败: count={}, error_type={}",
                len(safe_ids),
                e.__class__.__name__,
            )
            raise VectorStoreUnavailableError(
                internal_message=f"{e.__class__.__name__}: {e}",
            ) from e

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        return self._ensure_vector_store()

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            vector_store = self._ensure_vector_store()
            docs = vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()


def _map_vector_add_error(exc: Exception) -> AppError:
    """把向量写入阶段的下游异常映射到稳定错误码。

    LangChain Milvus 的 `add_documents` 内部同时包含 embedding 和 Milvus 写入；
    当前阶段还没有更细粒度 adapter，所以只能根据异常类型与常见提供商关键字做保守分类。
    分类结果只进入内部 `AppError`，用户仍只看到契约里的安全文案，不会暴露原始异常全文。
    """

    error_text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(
        keyword in error_text
        for keyword in ("embedding", "dashscope", "embed_query", "embed_documents")
    ):
        return EmbeddingProviderError(internal_message=f"{exc.__class__.__name__}: {exc}")
    return VectorStoreUnavailableError(internal_message=f"{exc.__class__.__name__}: {exc}")


def _escape_milvus_string(value: str) -> str:
    """转义 Milvus 表达式字符串，避免 doc_id 中的引号破坏 delete 表达式。"""

    return value.strip().replace("\\", "\\\\").replace('"', '\\"')
