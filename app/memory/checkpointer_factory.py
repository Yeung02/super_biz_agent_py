"""记忆存储底座工厂：短期记忆 checkpointer 的唯一创建入口。

生产形态（docker-compose.memory.yml）：
- 短期记忆 = Redis（LangGraph checkpoint，多副本共享、AOF 持久化）
- 长期记忆 = PostgreSQL（会话历史 + 摘要）与 Milvus（RAG 知识 + 用户画像）

checkpointer 后端由 `config.memory_checkpointer` 决定，默认 redis。`memory` 后端
仅供测试进程注入（tests/conftest.py 会在导入 app 前设置环境变量），生产配置
memory 属于配置错误：`verify_memory_storage()` 只在测试后端下跳过外部存储检查。

业务层（ConversationManager/RagAgentService/AIOpsService）一律通过
`create_checkpointer()` 获取实例，不直接 import 具体后端，保证底座可替换。
"""

from __future__ import annotations

import threading

from loguru import logger

from app.config import config


class MemoryStorageUnavailableError(RuntimeError):
    """启动期记忆存储连接验证失败（fail-fast，不做静默降级）。"""


_checkpointer_instance: object | None = None
_checkpointer_lock = threading.Lock()


def _create_redis_dual_mode_saver(redis_url: str) -> object:
    """创建双模 Redis checkpointer（async graph 执行 + sync 业务读取）。

    背景：langgraph-checkpoint-redis 的 `RedisSaver`（sync）与 `AsyncRedisSaver`
    （async）各自只实现一半接口，另一半继承基类直接抛 `NotImplementedError`。
    项目同时存在两条访问路径：
    - `graph.astream`（chat 流式 / AIOps Plan-Execute-Replan）走 async 方法；
    - `ConversationManager._read_checkpoint` / `graph.get_state` / 旧历史读取 /
      TTL cleanup 走 sync 方法。
    单用任一 saver 都会让另一条路径全量失败（曾表现为 chat/aiops INTERNAL_ERROR）。

    双模 saver 继承 AsyncRedisSaver（async 方法原生可用），并把 sync 方法委托给
    内部同步 RedisSaver。两个客户端指向同一 Redis 键空间，读写一致；`setup()` 的
    FT 索引建在 Redis 服务端，任一端执行即可共享。
    """

    from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver

    class _RedisDualModeSaver(AsyncRedisSaver):
        """async 方法继承自 AsyncRedisSaver；sync 方法委托内部同步 saver。"""

        def __init__(self, url: str) -> None:
            super().__init__(redis_url=url)
            self._sync_saver = RedisSaver(redis_url=url)

        # ---- 同步方法：委托同步 saver（签名透传，兼容上游版本差异） ----
        def setup(self, *args: object, **kwargs: object) -> object:
            # 覆盖 AsyncRedisSaver 的协程 setup：启动期 verify_memory_storage
            # 以同步方式建索引（FT 索引在 Redis 服务端，两端共享）。
            return self._sync_saver.setup(*args, **kwargs)

        def get_tuple(self, *args: object, **kwargs: object) -> object:
            return self._sync_saver.get_tuple(*args, **kwargs)

        def get(self, *args: object, **kwargs: object) -> object:
            return self._sync_saver.get(*args, **kwargs)

        def list(self, *args: object, **kwargs: object) -> object:  # noqa: A003
            return self._sync_saver.list(*args, **kwargs)

        def put(self, *args: object, **kwargs: object) -> object:
            return self._sync_saver.put(*args, **kwargs)

        def put_writes(self, *args: object, **kwargs: object) -> object:
            return self._sync_saver.put_writes(*args, **kwargs)

        def delete_thread(self, *args: object, **kwargs: object) -> object:
            # 两端 delete_thread 均为同步实现且等价，委托同步端保持旧行为。
            return self._sync_saver.delete_thread(*args, **kwargs)

    return _RedisDualModeSaver(redis_url)


def create_checkpointer() -> object:
    """按配置返回进程级共享的 checkpointer 单例。

    - `redis`：双模 Redis checkpointer（异步 graph 执行 + 同步业务读取），
      连接惰性建立（redis-py 连接池），`setup()` 建索引由
      `verify_memory_storage()` 在应用启动时显式执行；
    - `memory`：进程内 MemorySaver，仅测试进程使用。
    """

    global _checkpointer_instance

    if _checkpointer_instance is not None:
        return _checkpointer_instance

    with _checkpointer_lock:
        if _checkpointer_instance is not None:
            return _checkpointer_instance

        backend = (config.memory_checkpointer or "redis").strip().lower()

        if backend == "redis":
            _checkpointer_instance = _create_redis_dual_mode_saver(config.redis_url)
            logger.info("记忆存储底座: Redis checkpointer 双模 (async graph + sync 业务读) ({})", config.redis_url)
        elif backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            _checkpointer_instance = MemorySaver()
            logger.warning(
                "记忆存储底座: 进程内 MemorySaver（仅限测试进程；生产必须配置 redis）"
            )
        else:
            raise ValueError(
                f"未知的 memory_checkpointer 后端: {backend!r}（可选: redis | memory）"
            )

        return _checkpointer_instance


def reset_checkpointer() -> None:
    """重置单例（测试专用，用于隔离不同后端的用例）。"""

    global _checkpointer_instance
    with _checkpointer_lock:
        _checkpointer_instance = None


def verify_memory_storage() -> None:
    """启动期硬依赖检查：Redis 可达并完成 setup，PG 可达。

    - 生产（redis 后端 + fail_fast 开启）：任一失败抛
      `MemoryStorageUnavailableError`，应用拒绝启动，避免带病服务静默失忆；
    - 测试（memory 后端）：跳过外部存储检查；
    - PG 检查同时执行 schema 初始化（建表幂等），见 conversation_store。
    """

    if not config.memory_storage_fail_fast:
        logger.warning("memory_storage_fail_fast=false：跳过记忆存储启动检查（降级模式）")
        return

    backend = (config.memory_checkpointer or "redis").strip().lower()
    if backend != "redis":
        logger.info("checkpointer 后端为 {!r}，跳过 Redis/PG 启动检查", backend)
        return

    try:
        saver = create_checkpointer()
        saver.setup()
        # 启动期显式连通性验证：优先用双模 saver 内部的同步客户端（async 客户端
        # 的 ping() 返回协程，同步调用会被静默跳过，起不到 fail-fast 作用）。
        # 兼容不同版本 RedisSaver 内部客户端属性名，按候选取。
        sync_saver = getattr(saver, "_sync_saver", None) or saver
        client = getattr(sync_saver, "_redis", None) or getattr(sync_saver, "_conn", None)
        if client is not None:
            client.ping()
    except Exception as exc:
        raise MemoryStorageUnavailableError(
            f"Redis 记忆存储不可用（{config.redis_url}）: {exc.__class__.__name__}: {exc}。"
            "请先执行 docker compose -f docker-compose.memory.yml up -d"
        ) from exc

    try:
        from app.memory.conversation_store import ensure_schema

        ensure_schema()
    except Exception as exc:
        raise MemoryStorageUnavailableError(
            f"PostgreSQL 长期记忆库不可用（{config.postgres_dsn}）: "
            f"{exc.__class__.__name__}: {exc}。"
            "请先执行 docker compose -f docker-compose.memory.yml up -d"
        ) from exc

    logger.info("✅ 记忆存储验证通过: Redis checkpointer + PostgreSQL 长期记忆库")
