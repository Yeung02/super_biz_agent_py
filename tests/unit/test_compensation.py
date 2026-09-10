"""三路写入对账补偿单元测试。

覆盖 TurnCompensator（内存队列重试、去重防乱序、有界、过期、失败重入队）与
run_reconciliation_once（前缀对账补写、PG 领先跳过、非前缀分歧、落单 user 不补）。
用内存 fake store / fake manager 替身，不依赖真实 PG 与 Redis。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.memory.compensation import (
    PendingTurn,
    ReconcileReport,
    TurnCompensator,
    run_reconciliation_once,
)


class _FakeStore:
    """ConversationHistoryStore 的内存替身：record/get/list 最小语义。"""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[dict[str, str]]] = {}
        self.fail_record = False
        self.fail_list = False
        self.fail_get = False
        self.record_calls: list[tuple[str, str, str, str]] = []

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        user_id: str = "default",
    ) -> None:
        if self.fail_record:
            raise RuntimeError("pg unavailable")
        self.record_calls.append((session_id, user_message, assistant_message, user_id))
        timestamp = datetime.now(UTC).isoformat()
        rows = self.messages.setdefault(session_id, [])
        rows.append({"role": "user", "content": user_message, "timestamp": timestamp})
        rows.append(
            {"role": "assistant", "content": assistant_message, "timestamp": timestamp}
        )
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "title": user_message[:50],
            }

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        if self.fail_get:
            raise RuntimeError("pg read failed")
        return list(self.messages.get(session_id, []))

    def list_sessions(self, limit: int = 100, *, user_id: str | None = None) -> list[dict[str, Any]]:
        if self.fail_list:
            raise RuntimeError("pg list failed")
        return list(self.sessions.values())[:limit]


class _FakeManager:
    """ConversationManager 替身：get_history 返回 checkpoint 视角历史。"""

    def __init__(self, history_by_session: dict[str, list[dict[str, str]]]) -> None:
        self.history_by_session = history_by_session
        self.fail_sessions: set[str] = set()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        if session_id in self.fail_sessions:
            raise RuntimeError("redis read failed")
        return list(self.history_by_session.get(session_id, []))


def _turn(session_id: str, index: int, *, queued_at: datetime | None = None) -> PendingTurn:
    return PendingTurn(
        session_id=session_id,
        user_message=f"q{index}",
        assistant_message=f"a{index}",
        user_id="default",
        queued_at=queued_at or datetime.now(UTC),
    )


def _make_compensator(store: _FakeStore, **kwargs: Any) -> TurnCompensator:
    defaults: dict[str, Any] = {"max_pending": 100, "max_age_hours": 24.0, "batch_size": 100}
    defaults.update(kwargs)
    return TurnCompensator(store, **defaults)


# ----- 队列补偿 -----


def test_flush_writes_pending_turn_in_order() -> None:
    store = _FakeStore()
    compensator = _make_compensator(store)

    compensator.enqueue("s1", "q1", "a1", user_id="u1")
    compensator.enqueue("s1", "q2", "a2", user_id="u1")
    report = compensator.flush_once()

    assert report.written == 2
    assert report.requeued == 0
    assert compensator.pending_count() == 0
    assert [row["content"] for row in store.get_history("s1")] == ["q1", "a1", "q2", "a2"]
    # user_id 随轮次透传到 record_turn。
    assert all(call[3] == "u1" for call in store.record_calls)


def test_flush_skips_turn_already_recorded() -> None:
    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")  # 后续轮次已直接写成功
    compensator = _make_compensator(store)

    compensator.enqueue("s1", "q1", "a1")
    report = compensator.flush_once()

    assert report.written == 0
    assert report.skipped_recorded == 1
    assert len(store.get_history("s1")) == 2  # 无重复写入


def test_flush_skips_stale_turn_to_preserve_order() -> None:
    """更早轮次失败、更晚轮次已落库：跳过旧轮次而不是乱序补写。"""

    store = _FakeStore()
    compensator = _make_compensator(store)

    # turn1 于 1 秒前失败入队。
    compensator.enqueue(
        "s1",
        "q1",
        "a1",
        user_id="default",
    )
    with compensator._lock:  # noqa: SLF001 - 模拟早于 turn2 的入队时间
        compensator._queue[0].queued_at = datetime.now(UTC) - timedelta(seconds=1)
    # turn2 随后直接写成功（PG 尾部时间戳晚于 turn1 入队时间）。
    store.record_turn("s1", "q2", "a2")

    report = compensator.flush_once()

    assert report.written == 0
    assert report.skipped_stale == 1
    # PG 顺序保持 [q2, a2]，不出现 [q2, a2, q1, a1] 的乱序历史。
    assert [row["content"] for row in store.get_history("s1")] == ["q2", "a2"]


def test_flush_requeues_on_store_failure() -> None:
    store = _FakeStore()
    compensator = _make_compensator(store)

    compensator.enqueue("s1", "q1", "a1")
    store.fail_record = True
    report = compensator.flush_once()

    assert report.written == 0
    assert report.requeued == 1
    assert compensator.pending_count() == 1

    # 存储恢复后下一轮冲刷成功。
    store.fail_record = False
    report = compensator.flush_once()
    assert report.written == 1
    assert compensator.pending_count() == 0


def test_flush_failure_keeps_other_sessions() -> None:
    """单会话重试失败不影响同批其他会话的补偿。"""

    store = _FakeStore()
    compensator = _make_compensator(store)

    compensator.enqueue("s1", "q1", "a1")
    compensator.enqueue("s2", "q1", "a1")

    original_record = store.record_turn

    def _fail_only_s1(session_id: str, *args: Any, **kwargs: Any) -> None:
        if session_id == "s1":
            raise RuntimeError("pg unavailable for s1")
        original_record(session_id, *args, **kwargs)

    store.record_turn = _fail_only_s1  # type: ignore[method-assign]
    report = compensator.flush_once()

    assert report.written == 1
    assert report.requeued == 1
    assert len(store.get_history("s2")) == 2
    assert compensator.pending_count() == 1  # s1 的轮次保留


def test_enqueue_is_bounded_and_drops_oldest() -> None:
    store = _FakeStore()
    compensator = _make_compensator(store, max_pending=2)

    compensator.enqueue("s1", "q1", "a1")
    compensator.enqueue("s2", "q1", "a1")
    compensator.enqueue("s3", "q1", "a1")

    assert compensator.pending_count() == 2
    report = compensator.flush_once()
    # 最旧的 s1 被丢弃，只剩 s2/s3。
    assert report.attempted == 2
    assert store.record_calls[0][0] == "s2"


def test_flush_drops_expired_turns() -> None:
    store = _FakeStore()
    compensator = _make_compensator(store, max_age_hours=1.0)

    compensator.enqueue("s1", "q-old", "a-old")
    # 手动把队列中的轮次改为 2 小时前入队。
    with compensator._lock:  # noqa: SLF001 - 测试直接操纵队列模拟过期
        compensator._queue[0].queued_at = datetime.now(UTC) - timedelta(hours=2)

    report = compensator.flush_once()

    assert report.skipped_expired == 1
    assert report.attempted == 0
    assert compensator.pending_count() == 0
    assert store.record_calls == []


def test_batch_size_limits_single_flush() -> None:
    store = _FakeStore()
    compensator = _make_compensator(store, batch_size=1)

    compensator.enqueue("s1", "q1", "a1")
    compensator.enqueue("s2", "q1", "a1")

    report = compensator.flush_once()
    assert report.attempted == 1
    assert compensator.pending_count() == 1

    report = compensator.flush_once()
    assert report.attempted == 1
    assert compensator.pending_count() == 0


# ----- checkpoint→PG 对账 -----


def _history(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user, assistant in pairs:
        rows.append({"role": "user", "content": user, "timestamp": "t"})
        rows.append({"role": "assistant", "content": assistant, "timestamp": "t"})
    return rows


def test_reconcile_appends_missing_tail() -> None:
    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")  # PG 只有第一轮
    manager = _FakeManager({"s1": _history(("q1", "a1"), ("q2", "a2"))})

    report = run_reconciliation_once(store=store, manager=manager)

    assert report.checked == 1
    assert report.repaired_sessions == 1
    assert report.appended_messages == 2
    assert [row["content"] for row in store.get_history("s1")] == ["q1", "a1", "q2", "a2"]


def test_reconcile_skips_when_pg_leading() -> None:
    """PG 领先 checkpoint（graph 执行失败但历史已记）：不回写 checkpoint。"""

    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")
    store.record_turn("s1", "q2", "a2")
    manager = _FakeManager({"s1": _history(("q1", "a1"))})  # checkpoint 只有第一轮

    report = run_reconciliation_once(store=store, manager=manager)

    assert report.checked == 1
    assert report.repaired_sessions == 0
    assert report.appended_messages == 0
    assert len(store.get_history("s1")) == 4  # PG 保持不动


def test_reconcile_reports_diverged_prefix() -> None:
    """内容前缀不匹配（非尾部缺失）：记分歧，不盲写。"""

    store = _FakeStore()
    store.record_turn("s1", "q1-modified", "a1")  # PG 内容与 checkpoint 不一致
    manager = _FakeManager({"s1": _history(("q1", "a1"), ("q2", "a2"))})

    report = run_reconciliation_once(store=store, manager=manager)

    assert report.diverged_sessions == 1
    assert report.repaired_sessions == 0
    assert len(store.get_history("s1")) == 2


def test_reconcile_skips_trailing_lone_user() -> None:
    """checkpoint 末尾落单 user（graph 未回复）：只补完整轮次。"""

    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")  # PG 已有第一轮
    manager = _FakeManager(
        {
            "s1": _history(("q1", "a1"), ("q2", "a2"))
            + [{"role": "user", "content": "q3", "timestamp": "t"}]
        }
    )

    report = run_reconciliation_once(store=store, manager=manager)

    # 第二轮补齐（2 条），落单 q3 不补（等 assistant 出现后由下一轮对账处理）。
    assert report.repaired_sessions == 1
    assert report.appended_messages == 2
    assert [row["content"] for row in store.get_history("s1")] == ["q1", "a1", "q2", "a2"]


def test_reconcile_counts_read_errors_and_continues() -> None:
    """单会话读取失败不阻断其余会话对账。"""

    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")
    store.record_turn("s2", "q1", "a1")
    manager = _FakeManager({"s1": _history(("q1", "a1"), ("q2", "a2")), "s2": _history(("q1", "a1"), ("q2", "a2"))})
    manager.fail_sessions.add("s1")

    report = run_reconciliation_once(store=store, manager=manager)

    assert report.error_sessions == 1
    assert report.repaired_sessions == 1  # s2 正常补写
    assert report.appended_messages == 2


def test_reconcile_empty_store_returns_zero_report() -> None:
    store = _FakeStore()
    manager = _FakeManager({})

    report = run_reconciliation_once(store=store, manager=manager)

    assert isinstance(report, ReconcileReport)
    assert report.checked == 0


def test_reconcile_ignores_checkpoint_read_failure_as_pg_leading() -> None:
    """Redis 读失败时门面 fail-open 返回空历史 → 视为 PG 领先，安全跳过。

    这里直接模拟 fail-open 后的空 checkpoint 视角（manager 返回 []），
    验证对账不会因此误删或误写 PG。
    """

    store = _FakeStore()
    store.record_turn("s1", "q1", "a1")
    manager = _FakeManager({"s1": []})  # Redis 不可用 → 空 checkpoint 视角

    report = run_reconciliation_once(store=store, manager=manager)

    assert report.repaired_sessions == 0
    assert len(store.get_history("s1")) == 2  # PG 不被破坏
