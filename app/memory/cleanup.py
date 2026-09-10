"""会话 TTL 清理：长期记忆的生命周期收口。

按 `memory_session_ttl_days` 定期清理：
1. PG 中 updated_at 过期的会话（历史 + 摘要，delete_sessions_stale）；
2. 对应的 Redis checkpoint thread（先拿被删清单再删，保证两边一致）。

Milvus 中的 RAG 知识与用户记忆画像是长期有效的，不参与 TTL 清理。
仅在生产后端（redis）下由 main.lifespan 启动该循环；测试进程不运行。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from loguru import logger

from app.config import config
from app.memory.conversation_store import conversation_history_store


async def memory_cleanup_loop() -> None:
    """周期执行 TTL 清理；单轮失败只记日志，不退出循环。"""

    interval_seconds = max(float(config.memory_cleanup_interval_hours) * 3600.0, 60.0)
    logger.info(
        "记忆 TTL 清理任务启动: 间隔 {:.1f}h, 会话保留 {} 天",
        config.memory_cleanup_interval_hours,
        config.memory_session_ttl_days,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            # TTL 清扫是同步阻塞 IO（PG 删表 + Redis 删 thread），放 worker 线程
            # 执行，避免清理期间阻塞事件循环拖慢在线请求。
            removed = await asyncio.to_thread(run_cleanup_once)
            if removed:
                logger.info("记忆 TTL 清理完成: 清除 {} 个过期会话", len(removed))
        except asyncio.CancelledError:
            logger.info("记忆 TTL 清理任务停止")
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务不允许因单轮失败退出
            logger.warning(
                "记忆 TTL 清理单轮失败（下轮重试）: {}: {}",
                exc.__class__.__name__,
                exc,
            )


def run_cleanup_once() -> list[str]:
    """执行一轮清理，返回被删除的 session_id 列表（供测试与手动触发复用）。"""

    stale_ids = conversation_history_store.delete_sessions_stale(
        older_than=timedelta(days=config.memory_session_ttl_days),
    )
    if stale_ids:
        from app.memory.checkpointer_factory import create_checkpointer

        checkpointer = create_checkpointer()
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if callable(delete_thread):
            for session_id in stale_ids:
                try:
                    delete_thread(session_id)
                except Exception as exc:  # noqa: BLE001 - 单个 thread 删除失败不阻断
                    logger.warning(
                        "清理 Redis checkpoint thread 失败 (session_id={}): {}",
                        session_id,
                        exc,
                    )
    return stale_ids
