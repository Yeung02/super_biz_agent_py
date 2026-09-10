"""ConversationHistoryStore 单元测试（PostgreSQL 版）。

store 现在是 PG 持久层；这里用内存 fake 连接池替身实现 store 用到的最小 SQL 语义
（upsert 会话、插入消息、按 updated_at 排序、级联删除、摘要 upsert），验证的是
store 自身的 Python 逻辑（标题、时间戳、分页、TTL、fail 路径），不依赖真实 PG。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import pytest

import app.memory.conversation_store as conversation_store_module
from app.memory.conversation_store import ConversationHistoryStore


class _FakePgDatabase:
    """实现 store 所用 SQL 子集的内存库。"""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = []
        self.summaries: dict[str, dict[str, Any]] = {}
        self._next_message_id = 1

    def execute(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        normalized = " ".join(sql.split())

        if normalized.startswith("CREATE"):
            return []

        if normalized.startswith("INSERT INTO chat_sessions"):
            session_id, user_id, title, created_at, updated_at = params
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "title": title,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "message_count": 0,
                    "last_message_id": 0,
                }
            else:
                self.sessions[session_id]["user_id"] = user_id
                self.sessions[session_id]["updated_at"] = updated_at
            return []

        if normalized.startswith("INSERT INTO chat_messages"):
            session_id, content, created_at = params
            role = "user" if "'user'" in normalized else "assistant"
            message_id = self._next_message_id
            self._next_message_id += 1
            self.messages.append(
                {
                    "id": message_id,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                }
            )
            return [{"id": message_id}] if "RETURNING id" in normalized else []

        if normalized.startswith("UPDATE chat_sessions"):
            timestamp, last_message_id, session_id = params
            session = self.sessions[session_id]
            session["updated_at"] = timestamp
            session["message_count"] += 2
            session["last_message_id"] = last_message_id
            return []

        if normalized.startswith("SELECT session_id, user_id, title, created_at, updated_at, message_count"):
            rows = list(self.sessions.values())
            if "WHERE user_id = %s" in normalized:
                rows = [row for row in rows if row["user_id"] == params[0]]
                limit = params[1]
            else:
                limit = params[0]
            rows.sort(
                key=lambda row: (row["updated_at"], row["last_message_id"]),
                reverse=True,
            )
            return rows[:limit]

        if normalized.startswith("SELECT role, content, created_at FROM chat_messages"):
            session_id = params[0]
            return [
                {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
                for row in self.messages
                if row["session_id"] == session_id
            ]

        if normalized.startswith("SELECT COUNT(*)"):
            session_id = params[0]
            return [
                {"total": sum(1 for row in self.messages if row["session_id"] == session_id)}
            ]

        if normalized.startswith("SELECT id, role, content, created_at FROM chat_messages"):
            session_id, limit, offset = params
            rows = [row for row in self.messages if row["session_id"] == session_id]
            rows.sort(key=lambda row: row["id"], reverse=True)
            return rows[offset : offset + limit]

        if normalized.startswith("SELECT 1 FROM chat_sessions"):
            return [{"1": 1}] if params[0] in self.sessions else []

        if normalized.startswith("SELECT session_id FROM chat_sessions WHERE updated_at <"):
            cutoff = params[0]
            return [
                {"session_id": row["session_id"]}
                for row in self.sessions.values()
                if row["updated_at"] < cutoff
            ]

        if normalized.startswith("DELETE FROM session_summaries WHERE session_id = ANY"):
            for session_id in params[0]:
                self.summaries.pop(session_id, None)
            return []

        if normalized.startswith("DELETE FROM chat_sessions WHERE session_id = ANY"):
            for session_id in params[0]:
                self._delete_session(session_id)
            return []

        if normalized.startswith("DELETE FROM session_summaries"):
            self.summaries.pop(params[0], None)
            return []

        if normalized.startswith("DELETE FROM chat_sessions"):
            self._delete_session(params[0])
            return []

        if normalized.startswith("SELECT summary, source_message_count, updated_at FROM session_summaries"):
            summary = self.summaries.get(params[0])
            return [dict(summary)] if summary else []

        if normalized.startswith("INSERT INTO session_summaries"):
            session_id, summary, source_message_count, updated_at = params
            self.summaries[session_id] = {
                "summary": summary,
                "source_message_count": source_message_count,
                "updated_at": updated_at,
            }
            return []

        raise AssertionError(f"未覆盖的 SQL: {normalized!r}")

    def _delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.messages = [row for row in self.messages if row["session_id"] != session_id]


class _FakePgCursor:
    def __init__(self, database: _FakePgDatabase) -> None:
        self._database = database
        self._result: list[dict[str, Any]] = []

    def __enter__(self) -> _FakePgCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._result = self._database.execute(sql, params)

    def fetchone(self) -> dict[str, Any] | None:
        return self._result[0] if self._result else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._result)


class _FakePgConnection:
    def __init__(self, database: _FakePgDatabase) -> None:
        self._database = database

    def cursor(self) -> _FakePgCursor:
        return _FakePgCursor(self._database)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeConnectionPool:
    """psycopg_pool.ConnectionPool 的最小替身。

    与真实池同形：`connection()` 是上下文管理器，成功 commit、异常 rollback；
    一个池实例共享同一个内存库，对应“每个 store 一个池”的生产布局。
    """

    def __init__(self, conninfo: str, **kwargs: object) -> None:
        _ = conninfo, kwargs
        self._database = _FakePgDatabase()

    @staticmethod
    def check_connection(connection: _FakePgConnection) -> None:
        """真实池在归还连接时做 SELECT 1 健康检查；fake 恒通过。"""

        _ = connection

    @contextmanager
    def connection(self) -> Iterator[_FakePgConnection]:
        connection = _FakePgConnection(self._database)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def close(self) -> None:
        return None


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> ConversationHistoryStore:
    """绑定内存 PG fake 池的 store；每个测试获得独立数据库。"""

    monkeypatch.setattr(
        conversation_store_module,
        "ConnectionPool",
        _FakeConnectionPool,
    )
    return ConversationHistoryStore()


def test_records_and_lists_sessions_ordered_by_latest_turn(store: ConversationHistoryStore) -> None:
    store.record_turn("session-a", "How is CPU?", "CPU is healthy.")
    store.record_turn("session-a", "And memory?", "Memory is stable.")
    store.record_turn("session-b", "Disk?", "Disk usage is normal.")

    sessions = store.list_sessions()
    session_ids = [session["session_id"] for session in sessions]

    assert session_ids == ["session-b", "session-a"]
    assert sessions[0]["title"] == "Disk?"
    assert sessions[1]["message_count"] == 4


def test_get_history_returns_timestamps_per_turn(store: ConversationHistoryStore) -> None:
    store.record_turn("session-a", "How is CPU?", "CPU is healthy.")
    store.record_turn("session-a", "And memory?", "Memory is stable.")

    sessions = {s["session_id"]: s for s in store.list_sessions()}
    history = store.get_history("session-a")

    assert [item["role"] for item in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "How is CPU?"
    assert history[0]["timestamp"] == sessions["session-a"]["created_at"]
    assert history[2]["timestamp"] == sessions["session-a"]["updated_at"]


def test_list_sessions_filters_by_user(store: ConversationHistoryStore) -> None:
    store.record_turn("session-a", "Q1", "A1", user_id="alice")
    store.record_turn("session-b", "Q2", "A2", user_id="bob")

    alice_sessions = store.list_sessions(user_id="alice")

    assert [session["session_id"] for session in alice_sessions] == ["session-a"]
    assert alice_sessions[0]["user_id"] == "alice"


def test_get_history_page_returns_latest_window_ascending(store: ConversationHistoryStore) -> None:
    for index in range(4):
        store.record_turn("session-a", f"Q{index}", f"A{index}")

    page = store.get_history_page("session-a", limit=2, offset=4)

    assert page["total"] == 8
    # id 倒序取第 5-6 条（A1/Q1），返回前再反转为时间正序。
    assert [item["content"] for item in page["messages"]] == ["Q1", "A1"]


def test_clear_session_removes_history_and_summary(store: ConversationHistoryStore) -> None:
    store.record_turn("session-a", "Question", "Answer")
    store.save_summary("session-a", "summary text", source_message_count=2)

    assert store.clear_session("session-a") is True

    assert store.list_sessions() == []
    assert store.get_history("session-a") == []
    assert store.get_summary("session-a") is None
    assert store.clear_session("session-a") is False


def test_delete_sessions_stale_only_removes_expired(store: ConversationHistoryStore) -> None:
    store.record_turn("old-session", "Old question", "Old answer")

    # cutoff = now - older_than；刚记录的会话只有在 older_than 为负时才会过期。
    assert store.delete_sessions_stale(older_than=timedelta(seconds=1)) == []

    removed = store.delete_sessions_stale(older_than=-timedelta(seconds=1))

    assert removed == ["old-session"]
    assert store.list_sessions() == []


def test_summary_upsert_overwrites_previous(store: ConversationHistoryStore) -> None:
    assert store.get_summary("session-a") is None

    store.save_summary("session-a", "first summary", source_message_count=2)
    store.save_summary("session-a", "second summary", source_message_count=4)

    summary = store.get_summary("session-a")
    assert summary is not None
    assert summary["summary"] == "second summary"
    assert summary["source_message_count"] == 4


def test_record_turn_rejects_blank_session_id(store: ConversationHistoryStore) -> None:
    with pytest.raises(ValueError):
        store.record_turn("   ", "Question", "Answer")


def test_pool_is_created_lazily_and_reused(
    store: ConversationHistoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连接池惰性创建且跨操作复用：同一 store 的多次操作共享一个池实例。"""

    created: list[_FakeConnectionPool] = []
    real_init = _FakeConnectionPool.__init__

    def _tracking_init(self: _FakeConnectionPool, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]
        created.append(self)

    monkeypatch.setattr(_FakeConnectionPool, "__init__", _tracking_init)

    assert store._pool is None  # noqa: SLF001 - 构造期不建立连接

    store.record_turn("session-a", "Question", "Answer")
    first_pool = store._pool  # noqa: SLF001

    store.get_history("session-a")
    store.save_summary("session-a", "summary", source_message_count=2)

    assert first_pool is not None
    assert store._pool is first_pool  # noqa: SLF001
    assert len(created) == 1
    assert store.get_history("session-a")[0]["content"] == "Question"


def test_close_resets_pool_and_allows_reuse(store: ConversationHistoryStore) -> None:
    """close() 释放池；后续操作会基于新池继续工作（停机再拉起的语义）。"""

    store.record_turn("session-a", "Question", "Answer")
    closed_pool = store._pool  # noqa: SLF001

    store.close()

    assert store._pool is None  # noqa: SLF001
    assert closed_pool is not None

    store.record_turn("session-b", "Question", "Answer")
    assert store._pool is not None  # noqa: SLF001
    assert store._pool is not closed_pool  # noqa: SLF001
    assert [item["session_id"] for item in store.list_sessions()] == ["session-b"]
