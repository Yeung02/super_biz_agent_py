"""Milvus 客户端工厂模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)

from app.config import config
from app.rag.models import DEFAULT_METADATA_VERSION


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2
    # 混合检索：BM25 function 输出的稀疏向量字段名，检索侧据此判断 hybrid 能力。
    SPARSE_VECTOR_FIELD: str = "sparse_vector"

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None
        # 连接时由 schema 检测填充：collection 是否带 BM25 sparse 字段。
        # 进程内缓存安全——schema 只会随人工迁移（drop + 重建 + 重新 ingest）变化。
        self._hybrid_supported: bool = False

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        try:
            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立连接
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建客户端
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)
                self._validate_existing_collection_schema()

            # 加载 collection
            self._load_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        # pymilvus 的类型标注可能不准确，实际返回 bool
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)  # type: ignore[arg-type]

    def _validate_existing_collection_schema(self) -> None:
        """Validate an existing collection without dropping user data."""

        if self._collection is None:
            raise RuntimeError("Milvus collection is not initialized")

        fields = {field.name: field for field in self._collection.schema.fields}
        expected_types = {
            "id": DataType.VARCHAR,
            "vector": DataType.FLOAT_VECTOR,
            "content": DataType.VARCHAR,
            "metadata": DataType.JSON,
        }
        missing_fields = [name for name in expected_types if name not in fields]
        if missing_fields:
            raise RuntimeError(
                "Milvus collection schema incompatible: missing fields "
                f"{missing_fields}; manual migration is required"
            )

        for field_name, expected_dtype in expected_types.items():
            actual_dtype = getattr(fields[field_name], "dtype", None)
            if actual_dtype != expected_dtype:
                raise RuntimeError(
                    "Milvus collection schema incompatible: field "
                    f"{field_name!r} has dtype {actual_dtype!r}, expected {expected_dtype!r}; "
                    "manual migration is required"
                )

        vector_params = getattr(fields["vector"], "params", {}) or {}
        existing_dim = vector_params.get("dim")
        try:
            existing_dim_value = int(existing_dim)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Milvus collection schema incompatible: vector dimension is missing; "
                "manual migration is required"
            ) from exc

        if existing_dim_value != self.VECTOR_DIM:
            raise RuntimeError(
                "Milvus collection schema incompatible: vector dimension mismatch "
                f"(existing={existing_dim_value}, expected={self.VECTOR_DIM}); "
                "automatic drop is disabled, manual migration is required"
            )

        # sparse 字段不参与硬校验：旧 collection 缺失时保持可用（hybrid 自动降级 dense），
        # 已迁移 collection 则记录 hybrid 能力供检索侧判断。
        self._hybrid_supported = self._detect_sparse_field(fields)

        logger.info(
            "Milvus collection schema validated: vector_dim={}, metadata_version={}, hybrid_supported={}",
            self.VECTOR_DIM,
            DEFAULT_METADATA_VERSION,
            self._hybrid_supported,
        )
        if config.rag_hybrid_search_enabled and not self._hybrid_supported:
            logger.warning(
                "rag_hybrid_search_enabled=true 但 collection 缺少 sparse_vector 字段，"
                "混合检索已降级为纯 dense；需 drop collection 并重新 ingest 完成迁移"
            )

    def _detect_sparse_field(self, fields: dict[str, object]) -> bool:
        """检测 collection 是否带 BM25 sparse 字段且类型正确。

        类型不对说明 schema 处于未知中间态，按不支持处理并告警，避免 hybrid
        检索打到错误字段上产生不可解释的排序。
        """

        sparse_field = fields.get(self.SPARSE_VECTOR_FIELD)
        if sparse_field is None:
            return False
        if getattr(sparse_field, "dtype", None) != DataType.SPARSE_FLOAT_VECTOR:
            logger.warning(
                "sparse 字段类型异常，混合检索已降级: expected SPARSE_FLOAT_VECTOR, got {}",
                getattr(sparse_field, "dtype", None),
            )
            return False
        return True

    def hybrid_supported(self) -> bool:
        """当前 collection 是否支持 BM25 混合检索（连接后由 schema 检测填充）。"""

        return self._hybrid_supported

    def _create_collection(self) -> None:
        """创建 biz collection"""
        hybrid_enabled = bool(config.rag_hybrid_search_enabled)

        # 定义字段
        content_field = FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=self.CONTENT_MAX_LENGTH,
        )
        if hybrid_enabled:
            # BM25 依赖 content 字段开启 analyzer；jieba 保证中文按词切分而不是逐字切分。
            content_field = FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "jieba"},
            )

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            content_field,
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        if hybrid_enabled:
            # BM25 function 输出字段：写入时只写 content，稀疏向量由 Milvus 服务端生成，
            # 写入链路（LangChain add_documents）零改动。
            fields.append(
                FieldSchema(
                    name=self.SPARSE_VECTOR_FIELD,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                )
            )

        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        if hybrid_enabled:
            schema.add_function(
                Function(
                    name="content_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["content"],
                    output_field_names=[self.SPARSE_VECTOR_FIELD],
                )
            )
            self._hybrid_supported = True

        # 创建 collection
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为 vector（及 hybrid 模式下的 sparse_vector）字段创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            "metric_type": "L2",  # 欧氏距离
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        _ = self._collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

        if not (config.rag_hybrid_search_enabled and self._hybrid_supported):
            return

        sparse_index_params = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }
        _ = self._collection.create_index(
            field_name=self.SPARSE_VECTOR_FIELD,
            index_params=sparse_index_params,
        )
        logger.info("成功为 sparse_vector 字段创建 BM25 索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        # 检查 collection 是否已加载（兼容多版本）
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            # 尝试列出 connections
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None

        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
milvus_manager = MilvusClientManager()
