"""三路写入对账补偿：PG 历史写入失败的重试与 checkpoint→PG 对账。

三路写入的职责划分与本模块的补偿范围：
- checkpoint（Redis）：LangGraph graph 执行自动写入，是执行状态的唯一事实源，不补偿；
- PG 历史（record_turn）：用户可见历史的唯一持久层，失败原本只 warning → 本模块
  提供两层补偿，消除"静默丢轮次导致 PG 与 checkpoint 永久分歧"；
- 摘要（save_summary）：fail-open，失败下次请求自然重生成，无需专门队列；对账补齐
  PG 后，source_message_count 的口径（checkpoint 过滤后 turns 数）随之对齐，不再漂移。

补偿分两层：
1. 内存待重试队列（短期）：record_turn 失败的轮次进有界队列，后台周期重试，覆盖
   瞬时 PG 抖动。队列有界（溢出丢最旧）且按年龄过期；进程重启即清空——新会话首次
   写入就失败且期间重启的场景接受丢失（trace 日志保留人工介入线索）。
2. 周期对账（长期兜底）：以 checkpoint 为准，对 PG 近期活跃会话做前缀对比，补写 PG
   缺失的尾部完整轮次。PG 领先 checkpoint（graph 执行失败但历史已记）是设计接受的
   行为，不反向回写 checkpoint，避免破坏 graph 状态。

一致性语义：record_turn 单事务写 user+assistant 两条，因此 PG 缺口总是整轮缺失；
checkpoint 末尾可能存在落单的 user（graph 未回复），对账只补完整轮次，落单 user 等
assistant 出现后由下一轮对账补齐。
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger

from app.config import config
from app.memory.conversation_store import conversation_history_store

# 历史条目里对账用到的字段（role/content 与 store.get_history / manager.get_history
# 两边的输出 schema 一致，timestamp 不参与对比）。
_HistoryItem = dict[str, str]


@dataclass
class PendingTurn:
    """一条待重试的会话轮次（record_turn 失败时入队）。"""

    session_id: str
    user_message: str
    assistant_message: str
    user_id: str
    queued_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class FlushReport:
    """一轮队列消费的结果统计。"""

    attempted: int = 0
    written: int = 0
    skipped_recorded: int = 0
    skipped_stale: int = 0
    skipped_expired: int = 0
    requeued: int = 0


@dataclass
class ReconcileReport:
    """一轮对账扫描的结果统计。"""

    checked: int = 0
    repaired_sessions: int = 0
    appended_messages: int = 0
    diverged_sessions: int = 0
    error_sessions: int = 0


class TurnCompensator:
    """PG 历史写入补偿器：内存有界队列 + 尾部去重防乱序重试。

    队列按入队序（同 session 内即时间序）消费；重试前先读 PG 当前尾部做两级判断：
    - 内容去重：轮次对已在 PG 中（后续轮次直接写成功、或上次重试已写）则跳过；
    - 时间防乱序：PG 尾部时间戳晚于轮次入队时间（说明该轮失败后，更晚的轮次已
      直接写成功）则跳过——补写旧轮次会打乱 PG 顺序，丢一轮优于乱序。
    """

    def __init__(
        self,
        store: Any | None = None,
        *,
        max_pending: int | None = None,
        max_age_hours: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.store = store or conversation_history_store
        self.max_pending = max(
            1,
            int(max_pending if max_pending is not None else config.turn_compensation_max_pending),
        )
        self.max_age = timedelta(
            hours=float(
                max_age_hours
                if max_age_hours is not None
                else config.turn_compensation_max_age_hours
            ),
        )
        self.batch_size = max(
            1,
            int(batch_size if batch_size is not None else config.turn_compensation_batch_size),
        )
        self._queue: deque[PendingTurn] = deque()
        self._lock = threading.Lock()

    def enqueue(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        user_id: str = "default",
    ) -> bool:
        """记录一条失败轮次；队列满时丢最旧并记日志。"""

        turn = PendingTurn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            user_id=user_id,
        )
        with self._lock:
            self._evict_expired_locked()
            if len(self._queue) >= self.max_pending:
                dropped = self._queue.popleft()
                logger.warning(
                    "补偿队列已满（max_pending={}），丢弃最旧待重试轮次: session_id={}",
                    self.max_pending,
                    dropped.session_id,
                )
            self._queue.append(turn)
        return True

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def flush_once(self) -> FlushReport:
        """消费一批待重试轮次；仍失败的重新入队等下轮。"""

        report = FlushReport()
        batch: list[PendingTurn] = []
        with self._lock:
            self._evict_expired_locked(report)
            while len(batch) < self.batch_size and self._queue:
                batch.append(self._queue.popleft())
        report.attempted = len(batch)
        if not batch:
            return report

        remaining: list[PendingTurn] = []
        for session_id, turns in _group_by_session(batch):
            try:
                written, skipped_recorded, skipped_stale = self._flush_session(session_id, turns)
            except Exception as exc:  # noqa: BLE001 - 单会话重试失败不阻断其余会话
                logger.warning(
                    "补偿重试失败（保留待下轮）: session_id={}, error={}: {}",
                    session_id,
                    exc.__class__.__name__,
                    exc,
                )
                remaining.extend(turns)
                continue
            report.written += written
            report.skipped_recorded += skipped_recorded
            report.skipped_stale += skipped_stale

        if remaining:
            with self._lock:
                self._queue.extend(remaining)
            report.requeued = len(remaining)
        return report

    def _flush_session(
        self,
        session_id: str,
        turns: list[PendingTurn],
    ) -> tuple[int, int, int]:
        """重试单个会话的一组轮次，返回 (写入数, 已存在跳过数, 过时跳过数)。"""

        history = self.store.get_history(session_id)
        recorded = _recorded_pairs(history)
        last_pg_time = _last_timestamp(history)
        written = skipped_recorded = skipped_stale = 0
        for turn in turns:
            if (turn.user_message, turn.assistant_message) in recorded:
                skipped_recorded += 1
                continue
            if last_pg_time is not None and last_pg_time > turn.queued_at:
                # 该轮失败后已有更晚的轮次直接写成功（PG 尾部时间戳更晚）：
                # 补写会乱序，跳过（丢一轮优于乱序）。
                skipped_stale += 1
                continue
            self.store.record_turn(
                turn.session_id,
                turn.user_message,
                turn.assistant_message,
                user_id=turn.user_id,
            )
            written += 1
        return written, skipped_recorded, skipped_stale

    def _evict_expired_locked(self, report: FlushReport | None = None) -> None:
        """丢弃超过 max_age 的待重试轮次（调用方需持有锁）。"""

        if not self._queue:
            return
        now = datetime.now(UTC)
        expired = [turn for turn in self._queue if now - turn.queued_at > self.max_age]
        if expired:
            self._queue = deque(turn for turn in self._queue if now - turn.queued_at <= self.max_age)
            if report is not None:
                report.skipped_expired = len(expired)
            logger.warning(
                "补偿队列清理 {} 条过期轮次（max_age={:.0f}h）",
                len(expired),
                self.max_age.total_seconds() / 3600.0,
            )


def run_reconciliation_once(
    *,
    max_sessions: int | None = None,
    store: Any | None = None,
    manager: Any | None = None,
) -> ReconcileReport:
    """以 checkpoint 为准补齐 PG 近期活跃会话缺失的尾部轮次。

    会话清单来自 PG（list_sessions 按 updated_at 倒序）：record_turn 从未成功的新会话
    不在 PG 中，无法被对账发现——该场景由内存队列覆盖，重启丢失是已接受的限制。
    manager 参数仅用于测试注入门面替身；生产路径从 rag_agent_service 取
    ConversationManager（延迟导入避免 app.memory → app.services 循环依赖）。
    """

    report = ReconcileReport()
    resolved_store = store or conversation_history_store
    if manager is None:
        from app.services.rag_agent_service import rag_agent_service

        manager = getattr(rag_agent_service, "conversation_manager", None)
        if manager is None:
            return report
    sessions = resolved_store.list_sessions(
        limit=max_sessions if max_sessions is not None else config.reconciliation_max_sessions
    )

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        if not session_id:
            continue
        report.checked += 1
        user_id = str(session.get("user_id") or "default") or "default"
        try:
            # 门面读 checkpoint 视角历史；Redis 不可用时 fail-open 返回空列表，
            # 对账安全跳过（不会误判 PG 领先）。
            checkpoint_history = manager.get_history(session_id)
            pg_history = resolved_store.get_history(session_id)
        except Exception:  # noqa: BLE001 - 单会话读取失败不阻断对账
            report.error_sessions += 1
            continue

        if len(pg_history) >= len(checkpoint_history):
            # PG 领先或持平：graph 执行失败但历史已记是设计接受的行为，不回写。
            continue
        missing = _missing_tail(checkpoint_history, pg_history)
        if missing is None:
            # 前缀不匹配：分歧超出"尾部缺失"范畴（如内容被外部修改），留人工介入。
            report.diverged_sessions += 1
            logger.warning(
                "对账发现非前缀分歧，跳过补写（需人工核查）: session_id={}", session_id
            )
            continue
        try:
            appended = _append_missing_pairs(resolved_store, session_id, user_id, missing)
        except Exception:  # noqa: BLE001 - 单会话补写失败不阻断对账
            report.error_sessions += 1
            continue
        if appended > 0:
            report.repaired_sessions += 1
            report.appended_messages += appended
            logger.info(
                "对账补写完成: session_id={}, 追加 {} 条消息", session_id, appended
            )
    return report


async def turn_compensation_loop() -> None:
    """周期消费待重试队列；单轮失败只记日志，不退出循环。"""

    interval_seconds = max(float(config.turn_compensation_interval_seconds), 1.0)
    logger.info(
        "历史写入补偿任务启动: 间隔 {:.0f}s, 队列上限 {}, 单轮批量 {}",
        interval_seconds,
        config.turn_compensation_max_pending,
        config.turn_compensation_batch_size,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            # 队列消费是同步阻塞 IO（PG 读写），放 worker 线程执行。
            report = await asyncio.to_thread(turn_compensator.flush_once)
            if report.written or report.requeued or report.skipped_stale:
                logger.info(
                    "补偿队列消费: 写入 {}, 已存在跳过 {}, 过时跳过 {}, 过期清理 {}, 重新入队 {}",
                    report.written,
                    report.skipped_recorded,
                    report.skipped_stale,
                    report.skipped_expired,
                    report.requeued,
                )
        except asyncio.CancelledError:
            logger.info("历史写入补偿任务停止")
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务不允许因单轮失败退出
            logger.warning(
                "历史写入补偿单轮失败（下轮重试）: {}: {}",
                exc.__class__.__name__,
                exc,
            )


async def reconciliation_loop() -> None:
    """周期执行 checkpoint→PG 对账；单轮失败只记日志，不退出循环。"""

    interval_seconds = max(float(config.reconciliation_interval_seconds), 60.0)
    logger.info(
        "历史对账任务启动: 间隔 {:.0f}s, 单轮最多 {} 个会话",
        interval_seconds,
        config.reconciliation_max_sessions,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            # 对账是同步阻塞 IO（checkpoint 读 + PG 读写），放 worker 线程执行。
            report = await asyncio.to_thread(run_reconciliation_once)
            if report.repaired_sessions or report.diverged_sessions or report.error_sessions:
                logger.info(
                    "历史对账完成: 检查 {}, 修复 {}, 追加 {} 条消息, 分歧 {}, 错误 {}",
                    report.checked,
                    report.repaired_sessions,
                    report.appended_messages,
                    report.diverged_sessions,
                    report.error_sessions,
                )
        except asyncio.CancelledError:
            logger.info("历史对账任务停止")
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务不允许因单轮失败退出
            logger.warning(
                "历史对账单轮失败（下轮重试）: {}: {}",
                exc.__class__.__name__,
                exc,
            )


turn_compensator = TurnCompensator()


def _group_by_session(batch: list[PendingTurn]) -> list[tuple[str, list[PendingTurn]]]:
    """按 session 分组并保持组内入队序；组间顺序不影响正确性。"""

    grouped: dict[str, list[PendingTurn]] = {}
    for turn in batch:
        grouped.setdefault(turn.session_id, []).append(turn)
    return list(grouped.items())


def _recorded_pairs(history: list[_HistoryItem]) -> set[tuple[str, str]]:
    """提取 PG 历史中已落库的 (user, assistant) 内容对（去重依据）。"""

    pairs: set[tuple[str, str]] = set()
    for index in range(len(history) - 1):
        if history[index].get("role") == "user" and history[index + 1].get("role") == "assistant":
            pairs.add((history[index].get("content", ""), history[index + 1].get("content", "")))
    return pairs


def _last_timestamp(history: list[_HistoryItem]) -> datetime | None:
    """解析 PG 历史尾部时间戳（防乱序判断依据）；无法解析返回 None。"""

    if not history:
        return None
    raw_timestamp = history[-1].get("timestamp")
    if not raw_timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw_timestamp))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _missing_tail(
    checkpoint_history: list[_HistoryItem],
    pg_history: list[_HistoryItem],
) -> list[_HistoryItem] | None:
    """返回 checkpoint 相对 PG 缺失的尾部消息；非前缀分歧返回 None。"""

    pg_length = len(pg_history)
    if pg_length == 0:
        return list(checkpoint_history)
    for index, pg_item in enumerate(pg_history):
        if index >= len(checkpoint_history):
            return None
        checkpoint_item = checkpoint_history[index]
        if (
            checkpoint_item.get("role") != pg_item.get("role")
            or checkpoint_item.get("content") != pg_item.get("content")
        ):
            return None
    return list(checkpoint_history[pg_length:])


def _append_missing_pairs(
    store: Any,
    session_id: str,
    user_id: str,
    missing: list[_HistoryItem],
) -> int:
    """把缺失尾部中的完整 user/assistant 轮次补写进 PG，返回追加的消息数。

    末尾落单的 user（graph 已收到问题但尚未回复）不补：等 assistant 出现后由下一轮
    对账补齐，避免 PG 出现半轮历史。
    """

    appended = 0
    index = 0
    while index < len(missing):
        item = missing[index]
        if item.get("role") != "user":
            # checkpoint 侧连续 assistant（如重试轨迹）：跳过该条，等待下一对。
            index += 1
            continue
        if index + 1 >= len(missing) or missing[index + 1].get("role") != "assistant":
            # 落单 user：不补，留给下一轮对账。
            break
        user_content = item.get("content", "")
        assistant_content = missing[index + 1].get("content", "")
        if user_content and assistant_content:
            store.record_turn(
                session_id,
                user_content,
                assistant_content,
                user_id=user_id,
            )
            appended += 2
        index += 2
    return appended
