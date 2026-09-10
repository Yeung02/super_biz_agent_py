"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import aiops, chat, file, health
from app.config import config
from app.core.errors import VectorStoreUnavailableError
from app.core.milvus_client import milvus_manager
from app.core.request_context import RequestContextMiddleware
from app.memory.checkpointer_factory import verify_memory_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")
    # metrics 是阶段 4 的旁路观测能力，不改变 API 响应；启动时打印路径便于排障时
    # 快速确认是否开启，同时回滚时只需设置 metrics_enabled=false。
    if config.metrics_enabled:
        logger.info(f"📈 Metrics JSONL: {config.metrics_jsonl_path}")

    # 记忆存储硬依赖检查（Redis checkpointer + PG 长期记忆库）：
    # 失败直接抛 MemoryStorageUnavailableError 拒绝启动，避免带病服务静默失忆。
    # 测试进程（memory 后端）在 verify 内部自动跳过外部存储检查。
    verify_memory_storage()

    # 记忆 TTL 清理后台任务（仅生产后端运行）
    cleanup_task: asyncio.Task | None = None
    # 三路写入对账补偿后台任务（仅生产后端运行）：
    # - 补偿队列周期重试 record_turn 失败的轮次；
    # - checkpoint→PG 前缀对账兜底修复长期分歧。
    compensation_task: asyncio.Task | None = None
    reconcile_task: asyncio.Task | None = None
    if config.memory_cleanup_enabled and (
        (config.memory_checkpointer or "redis").strip().lower() == "redis"
    ):
        from app.memory.cleanup import memory_cleanup_loop

        cleanup_task = asyncio.create_task(memory_cleanup_loop())
    if config.turn_compensation_enabled and (
        (config.memory_checkpointer or "redis").strip().lower() == "redis"
    ):
        from app.memory.compensation import reconciliation_loop, turn_compensation_loop

        compensation_task = asyncio.create_task(turn_compensation_loop())
        if config.reconciliation_enabled:
            reconcile_task = asyncio.create_task(reconciliation_loop())

    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    try:
        milvus_manager.connect()
        logger.info("✅ Milvus 连接成功")
    except Exception as exc:
        app_error = VectorStoreUnavailableError(
            internal_message=f"{exc.__class__.__name__}: {exc}",
        )
        logger.error(
            "Milvus 连接失败，服务继续启动并由 /health 暴露不可用状态: code={}",
            app_error.code,
        )

    logger.info("=" * 60)

    yield

    # 关闭时执行
    if cleanup_task is not None:
        cleanup_task.cancel()
    if compensation_task is not None:
        compensation_task.cancel()
    if reconcile_task is not None:
        reconcile_task.cancel()
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    # 关闭 PG 长期记忆库连接池，释放后台维护线程与数据库连接。
    from app.memory.conversation_store import conversation_history_store

    conversation_history_store.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的Aegis Agent运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allow_origins,  # 默认保持 "*"，生产环境可通过配置收敛
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ISSUE-002: 请求上下文必须位于业务路由之前接入。middleware 只创建 ctx、追加
# trace header/body 和写 JSONL，不读取 request body，避免提前破坏上传接口兼容性。
app.add_middleware(RequestContextMiddleware)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
# `/health` 是当前代码已经暴露的旧路径，必须继续保留；API 契约和阶段 4 集成测试
# 统一使用 `/api/health`，因此这里只追加同一个 router 的兼容别名，不改变 handler
# 返回 schema，也不影响旧监控或脚本继续访问顶层健康检查。
app.include_router(health.router, prefix="/api", tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers={"Cache-Control": "no-cache"})
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
