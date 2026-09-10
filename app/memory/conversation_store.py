"""长期记忆库：PostgreSQL 会话历史 + 会话摘要持久化。

与短期记忆（Redis checkpointer，LangGraph 执行状态）刻意分离：
- checkpoint 是执行状态，随裁剪丢弃远端轮次；
- 本库是用户可见历史、摘要与 TTL 归档的唯一持久层，由
  docker-compose.memory.yml 提供（config.postgres_dsn）。

连接管理：所有操作通过 psycopg_pool 连接池复用连接（池惰性创建，构造不建立
连接），并发操作由池分配连接，不再用进程级锁串行化。事务语义与旧实现一致：
成功 commit、异常 rollback，归还连接由池负责。

硬依赖策略：连接失败在应用启动期由 `checkpointer_factory.verify_memory_storage`
fail-fast；运行期 record/读取失败由调用方（chat API）按旧行为降级为 warning。
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import config

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT 'default',
        title TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0,
        last_message_id BIGINT NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
    ON chat_sessions(user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, id)
    """,
    # 摘要表不加会话外键：摘要可在 record_turn 之前生成（基于 checkpoint），
    # 会话清理时由 clear_session/delete_sessions_stale 手动级联删除。
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        source_message_count INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
)


class ConversationHistoryStore:
    """PostgreSQL-backed store for chat sessions, visible messages and summaries."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        pool_timeout_seconds: float | None = None,
    ) -> None:
        # 构造不建立连接（连接池与 schema 幂等初始化都惰性延迟到首次操作），保证测试
        # 进程 import 本模块不会因为 PG 不可用而在收集阶段失败。
        self.dsn = dsn or config.postgres_dsn
        self.pool_min_size = max(
            1,
            int(pool_min_size if pool_min_size is not None else config.postgres_pool_min_size),
        )
        self.pool_max_size = max(
            self.pool_min_size,
            int(pool_max_size if pool_max_size is not None else config.postgres_pool_max_size),
        )
        self.pool_timeout_seconds = max(
            1.0,
            float(
                pool_timeout_seconds
                if pool_timeout_seconds is not None
                else config.postgres_pool_timeout_seconds
            ),
        )
        # 只保护“池/ schema 各初始化一次”的临界区；并发操作本身由连接池调度，
        # 不再做进程级串行化（旧 RLock 是吞吐瓶颈）。
        self._init_lock = threading.Lock()
        self._pool: ConnectionPool | None = None
        self._schema_ready = False

    # ----- 会话与消息 -----

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        user_id: str = "default",
    ) -> None:
        """Record one user/assistant turn in a durable history table."""

        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id must not be empty")

        timestamp = datetime.now(UTC)
        title = _title_from_message(user_message)

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (
                        session_id, user_id, title, created_at, updated_at,
                        message_count, last_message_id
                    )
                    VALUES (%s, %s, %s, %s, %s, 0, 0)
                    ON CONFLICT (session_id) DO UPDATE
                        SET updated_at = EXCLUDED.updated_at,
                            user_id = EXCLUDED.user_id
                    """,
                    (clean_session_id, user_id, title, timestamp, timestamp),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, created_at)
                    VALUES (%s, 'user', %s, %s)
                    """,
                    (clean_session_id, user_message, timestamp),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, created_at)
                    VALUES (%s, 'assistant', %s, %s)
                    RETURNING id
                    """,
                    (clean_session_id, assistant_message, timestamp),
                )
                assistant_row_id = cursor.fetchone()["id"]
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = %s,
                        message_count = message_count + 2,
                        last_message_id = %s
                    WHERE session_id = %s
                    """,
                    (timestamp, assistant_row_id, clean_session_id),
                )

    def list_sessions(
        self,
        limit: int = 100,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return sessions ordered by the newest recorded turn first."""

        safe_limit = min(max(int(limit), 1), 500)
        query = """
            SELECT session_id, user_id, title, created_at, updated_at, message_count
            FROM chat_sessions
        """
        params: list[Any] = []
        if user_id is not None:
            query += " WHERE user_id = %s"
            params.append(user_id)
        query += " ORDER BY updated_at DESC, last_message_id DESC LIMIT %s"
        params.append(safe_limit)

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()

        return [_session_row(row) for row in rows]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the complete visible message history for a session."""

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": _iso(row["created_at"]),
            }
            for row in rows
        ]

    def get_history_page(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """分页读取会话消息（生产前端演进用，按 id 倒序取页后正序返回）。"""

        safe_limit = min(max(int(limit), 1), 200)
        safe_offset = max(int(offset), 0)
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS total FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                total = int(cursor.fetchone()["total"])
                cursor.execute(
                    """
                    SELECT id, role, content, created_at FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (session_id, safe_limit, safe_offset),
                )
                rows = cursor.fetchall()

        messages = [
            {
                "id": int(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "timestamp": _iso(row["created_at"]),
            }
            for row in reversed(rows)
        ]
        return {"total": total, "limit": safe_limit, "offset": safe_offset, "messages": messages}

    def clear_session(self, session_id: str) -> bool:
        """Delete one session, its visible history and its persisted summary."""

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                existed = cursor.fetchone() is not None
                cursor.execute("DELETE FROM session_summaries WHERE session_id = %s", (session_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        return existed

    def delete_sessions_stale(self, *, older_than: timedelta) -> list[str]:
        """删除 updated_at 早于阈值的会话（TTL 归档入口），返回被删 session_id 列表。"""

        cutoff = datetime.now(UTC) - older_than
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id FROM chat_sessions
                    WHERE updated_at < %s
                    """,
                    (cutoff,),
                )
                stale_ids = [row["session_id"] for row in cursor.fetchall()]
                if stale_ids:
                    cursor.execute(
                        "DELETE FROM session_summaries WHERE session_id = ANY(%s)",
                        (stale_ids,),
                    )
                    cursor.execute(
                        "DELETE FROM chat_sessions WHERE session_id = ANY(%s)",
                        (stale_ids,),
                    )
        return stale_ids

    # ----- 会话摘要持久化 -----

    def get_summary(self, session_id: str) -> dict[str, Any] | None:
        """读取持久化摘要；无摘要返回 None。"""

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT summary, source_message_count, updated_at
                    FROM session_summaries
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()

        if row is None:
            return None
        return {
            "summary": row["summary"],
            "source_message_count": int(row["source_message_count"]),
            "updated_at": _iso(row["updated_at"]),
        }

    def save_summary(self, session_id: str, summary: str, *, source_message_count: int) -> None:
        """UPSERT 会话摘要（增量生成后的持久化写回）。"""

        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO session_summaries (session_id, summary, source_message_count, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE
                        SET summary = EXCLUDED.summary,
                            source_message_count = EXCLUDED.source_message_count,
                            updated_at = EXCLUDED.updated_at
                    """,
                    (
                        session_id,
                        summary,
                        int(source_message_count),
                        datetime.now(UTC),
                    ),
                )

    # ----- 内部 -----

    def _get_pool(self) -> ConnectionPool:
        """惰性创建进程内共享连接池；首次操作或启动期检查时触发。"""

        if self._pool is not None:
            return self._pool
        with self._init_lock:
            if self._pool is None:
                # check=check_connection：归还连接时做一次 SELECT 1 健康检查，
                # 把“服务端重启/网络闪断产生的死连接”转为丢弃重建，而不是把
                # OperationalError 抛给下一次借用。max_lifetime/max_idle 走池默认
                # （1h/10min），周期性回收避免长寿命连接被中间件静默掐断。
                self._pool = ConnectionPool(
                    self.dsn,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    timeout=self.pool_timeout_seconds,
                    name="conversation-history",
                    kwargs={
                        "row_factory": dict_row,
                        "autocommit": False,
                        "connect_timeout": 5,
                    },
                    check=ConnectionPool.check_connection,
                    open=True,
                )
        return self._pool

    def _connect(self) -> AbstractContextManager[psycopg.Connection]:
        """从池中借出连接；成功 commit、异常 rollback、归还由池负责。"""

        return self._get_pool().connection()

    def _ensure_schema(self, connection: psycopg.Connection) -> None:
        if self._schema_ready:
            return
        with self._init_lock:
            if self._schema_ready:
                return
            with connection.cursor() as cursor:
                for statement in _SCHEMA_STATEMENTS:
                    cursor.execute(statement)
            self._schema_ready = True

    def close(self) -> None:
        """关闭连接池（应用停机时调用；池未创建时为空操作）。"""

        pool = self._pool
        self._pool = None
        self._schema_ready = False
        if pool is not None:
            pool.close()


def ensure_schema(dsn: str | None = None) -> None:
    """幂等建表（应用启动期 fail-fast 检查时调用）。"""

    store = ConversationHistoryStore(dsn)
    with store._connect() as connection:  # noqa: SLF001 - 同模块内复用连接上下文
        store._ensure_schema(connection)  # noqa: SLF001


def _session_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
        "message_count": int(row["message_count"]),
    }


def _title_from_message(message: str) -> str:
    compact = " ".join(message.split())
    if not compact:
        return "New conversation"
    return compact[:30] + ("..." if len(compact) > 30 else "")


def _iso(value: datetime) -> str:
    return value.isoformat()


conversation_history_store = ConversationHistoryStore()
