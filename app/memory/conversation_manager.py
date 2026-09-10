"""ConversationManager 与 MemorySaver 的业务边界。

MemorySaver 只负责 LangGraph checkpoint 的读写存储，checkpoint 内部结构会随
LangGraph 版本和 graph 编排变化。ConversationManager 把这种不稳定结构隔离在单一门面
中，向业务层提供可过滤、可裁剪、可追踪的历史上下文，避免 API/Agent 直接解析底层
checkpoint 造成兼容风险。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, TypeAlias, cast

from app.config import config
from app.core.errors import InternalAppError, JsonObject
from app.core.input_guard import input_guard
from app.core.request_context import RequestContext
from app.core.token_budget import (
    TokenBudget,
    TokenBudgetManager,
    token_budget_manager as default_token_budget_manager,
)
from app.memory.summarizer import ConversationSummarizer, SummaryResult
from app.observability.tracing import TraceLogger

ConversationRole: TypeAlias = Literal["user", "assistant"]
_MemoryReadError: TypeAlias = AttributeError | TypeError | ValueError | RuntimeError | KeyError
_MEMORY_READ_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
)


@dataclass(frozen=True)
class ConversationTurn:
    """对业务层安全可见的一条会话消息。

    这里只允许 user/assistant 角色，系统提示、工具消息、函数调用和非文本 payload 都会在
    进入该模型前被过滤。这样对外历史和后续 prompt 上下文不会泄漏系统约束或工具原文。
    """

    role: ConversationRole
    content: str
    timestamp: str
    metadata: JsonObject = field(default_factory=dict)

    def to_history_dict(self) -> dict[str, str]:
        """转换为旧 `/api/chat/session/{session_id}` 兼容的历史条目。"""

        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ConversationContext:
    """提供给 Agent/编排层的可控会话上下文。"""

    summary: str | None
    recent_messages: tuple[ConversationTurn, ...]
    history_metadata: JsonObject
    # 用户长期记忆画像召回结果（跨会话偏好/事实/实体），由编排层注入，
    # 默认空元组保证不启用画像时行为与旧路径完全一致。
    user_memories: tuple[str, ...] = ()


class ConversationManager:
    """业务会话上下文门面。

    本类只通过 MemorySaver 的公开读/删方法访问底层状态，并在内部容错
    `get_tuple()`、`get()`、namedtuple、普通 tuple 和裸 checkpoint 字典几种形态。
    写入方面，当前 LangGraph Agent 仍由 graph/checkpointer 自动保存 checkpoint；
    `save_turn()` 仅调用显式适配方法，不手工拼写 checkpoint，避免破坏 graph 状态。
    """

    def __init__(
        self,
        memory_saver: object,
        *,
        token_budget_manager: TokenBudgetManager | None = None,
        trace_logger: TraceLogger | None = None,
        summarizer: ConversationSummarizer | None = None,
        recent_turns: int | None = None,
        fail_open_on_read_error: bool = True,
        enabled: bool | None = None,
        summary_enabled: bool | None = None,
        summary_store: object | None = None,
    ) -> None:
        self.memory_saver = memory_saver
        # 摘要持久层（ConversationHistoryStore：get_summary/save_summary 协议）。
        # 注入后摘要增量生成并落库，未注入时保持旧行为（每次现算、不落库）。
        self.summary_store = summary_store
        self.token_budget_manager = token_budget_manager or default_token_budget_manager
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        self.summary_enabled = bool(
            summary_enabled
            if summary_enabled is not None
            else config.conversation_summary_enabled
        )
        # 默认 summarizer 使用本地抽取式兜底摘要，不访问外部模型；生产编排层或测试可以
        # 注入 ChatQwen/fake_llm。这样 ISSUE-014 的摘要接入不会让已有单测或旧 API 路径
        # 误触 DashScope，同时仍保留后续接入真实 LLM 的稳定扩展点。
        self.summarizer = summarizer or ConversationSummarizer(
            token_budget_manager=self.token_budget_manager,
            trace_logger=self.trace_logger,
            enabled=self.summary_enabled,
        )
        self.recent_turns = max(
            1,
            int(recent_turns if recent_turns is not None else config.conversation_recent_turns),
        )
        self.fail_open_on_read_error = fail_open_on_read_error
        self.enabled = bool(
            enabled if enabled is not None else config.conversation_manager_enabled
        )

    def load_context(
        self,
        session_id: str,
        budget: TokenBudget | int | None,
        ctx: RequestContext | None = None,
    ) -> ConversationContext:
        """读取并裁剪会话上下文。

        最近 N 轮选择和 token 预算裁剪都在门面内完成；调用方拿到的是业务可控上下文，
        而不是 MemorySaver 原始 checkpoint。读取失败默认 fail-open 为空历史，保护旧
        Chat API 可用性；若调用方需要强失败，可在构造时关闭 `fail_open_on_read_error`。
        """

        validated_session_id = input_guard.validate_session_id(session_id)
        try:
            checkpoint = self._read_checkpoint(validated_session_id)
        except _MEMORY_READ_ERRORS as exc:
            return self._handle_load_error(validated_session_id, ctx, exc)

        messages = _extract_messages(checkpoint)
        all_turns = _messages_to_turns(messages)
        summary = _extract_summary(checkpoint)
        persisted_summary = self._load_persisted_summary(validated_session_id)
        summary_result: SummaryResult | None = None
        if (
            persisted_summary is not None
            and len(all_turns) <= int(persisted_summary.get("source_message_count") or 0)
        ):
            # 持久化摘要已覆盖当前全部消息：直接复用，避免每个请求重复生成。
            summary = str(persisted_summary.get("summary") or summary or "")
            summary_result = None
        else:
            base_summary = (
                str(persisted_summary.get("summary"))
                if persisted_summary is not None and persisted_summary.get("summary")
                else summary
            )
            summary_result = self._summarize_history_if_needed(all_turns, base_summary, budget, ctx)
        if summary_result is not None and summary_result.summary:
            summary = summary_result.summary
            self._persist_summary(validated_session_id, summary, len(all_turns))
        recent_turns = self._select_recent_turns(all_turns)
        after_budget_turns, budget_trimmed_count = self._trim_recent_messages(
            recent_turns,
            budget,
            ctx,
        )
        trimmed_count = max(0, len(all_turns) - len(after_budget_turns))
        if budget_trimmed_count > 0:
            trimmed_count = max(trimmed_count, len(all_turns) - len(after_budget_turns))

        metadata = _history_metadata(
            raw_message_count=len(messages),
            visible_message_count=len(all_turns),
            recent_message_count=len(after_budget_turns),
            trimmed_count=trimmed_count,
            summary_used=summary is not None,
            load_failed=False,
        )
        self._record_load(
            ctx,
            status="ok",
            message_count=len(all_turns),
            summary_used=summary is not None,
            trimmed_count=trimmed_count,
        )
        return ConversationContext(
            summary=summary,
            recent_messages=after_budget_turns,
            history_metadata=metadata,
        )

    def summarize_if_needed(
        self,
        session_id: str,
        budget: TokenBudget | int | None,
        ctx: RequestContext | None = None,
    ) -> str | None:
        """按需返回已有或新生成摘要。

        摘要只服务内部上下文，不写回 MemorySaver，也不影响对外历史接口。生成失败时
        返回 None 或旧摘要，主请求继续使用最近完整轮次，符合 ISSUE-014 fail-open 要求。
        """

        return self.load_context(session_id, budget, ctx).summary

    def save_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        metadata: Mapping[str, object] | None,
        ctx: RequestContext | None = None,
    ) -> bool:
        """保存一轮业务对话。

        当前生产路径的 checkpoint 由 LangGraph 自动写入；这里仅支持显式适配器方法
        `save_turn` 或 `append_turn`。没有适配器时返回 false，而不是手工修改
        MemorySaver checkpoint，避免把阶段 2 的门面变成底层存储实现。
        """

        validated_session_id = input_guard.validate_session_id(session_id)
        save_method = _resolve_save_method(self.memory_saver)
        if save_method is None:
            self._record_save(ctx, status="skipped", success=False)
            return False
        try:
            save_method(validated_session_id, user_msg, assistant_msg, metadata)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._record_save(ctx, status="error", success=False, error_code="INTERNAL_ERROR")
            return False

        self._record_save(ctx, status="ok", success=True)
        return True

    def clear_session(self, session_id: str, ctx: RequestContext | None = None) -> bool:
        """清理底层 MemorySaver thread。

        API 层仍返回旧 `status/message/data` 字段；本方法只负责通过门面执行删除并记录
        trace。失败时返回 false，避免把底层异常原文暴露给用户。
        """

        validated_session_id = input_guard.validate_session_id(session_id)
        delete_thread = getattr(self.memory_saver, "delete_thread", None)
        if not callable(delete_thread):
            self._record_clear(ctx, status="error", success=False, error_code="INTERNAL_ERROR")
            return False
        try:
            cast(Callable[[str], object], delete_thread)(validated_session_id)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._record_clear(ctx, status="error", success=False, error_code="INTERNAL_ERROR")
            return False

        self._record_clear(ctx, status="ok", success=True)
        return True

    def get_history(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> list[dict[str, str]]:
        """返回 API 兼容的安全历史列表。

        与 `load_context()` 不同，查询历史面向用户查看，应返回全部安全可见历史，而不是
        只返回最近 N 轮；但同样必须过滤系统消息和工具原始 payload。
        """

        validated_session_id = input_guard.validate_session_id(session_id)
        try:
            checkpoint = self._read_checkpoint(validated_session_id)
        except _MEMORY_READ_ERRORS as exc:
            if self.fail_open_on_read_error:
                self._record_load(
                    ctx,
                    status="error",
                    message_count=0,
                    summary_used=False,
                    trimmed_count=0,
                    error_code="INTERNAL_ERROR",
                )
                return []
            raise InternalAppError(
                internal_message=f"{exc.__class__.__name__}",
                origin_module="app.memory.conversation_manager",
            ) from exc

        turns = _messages_to_turns(_extract_messages(checkpoint))
        self._record_load(
            ctx,
            status="ok",
            message_count=len(turns),
            summary_used=_extract_summary(checkpoint) is not None,
            trimmed_count=0,
        )
        return [turn.to_history_dict() for turn in turns]

    def _read_checkpoint(self, session_id: str) -> Mapping[str, object] | None:
        thread_config = _thread_config(session_id)
        get_tuple = getattr(self.memory_saver, "get_tuple", None)
        if callable(get_tuple):
            raw_tuple = cast(Callable[[Mapping[str, object]], object | None], get_tuple)(
                thread_config
            )
            checkpoint = _coerce_checkpoint(raw_tuple)
            if checkpoint is not None:
                return checkpoint

        get_checkpoint = getattr(self.memory_saver, "get", None)
        if callable(get_checkpoint):
            raw_checkpoint = cast(
                Callable[[Mapping[str, object]], object | None],
                get_checkpoint,
            )(thread_config)
            return _coerce_checkpoint(raw_checkpoint)
        return None

    def _select_recent_turns(
        self,
        turns: tuple[ConversationTurn, ...],
    ) -> tuple[ConversationTurn, ...]:
        max_messages = self.recent_turns * 2
        if len(turns) <= max_messages:
            return turns
        return turns[-max_messages:]

    def _trim_recent_messages(
        self,
        turns: tuple[ConversationTurn, ...],
        budget: TokenBudget | int | None,
        ctx: RequestContext | None,
    ) -> tuple[tuple[ConversationTurn, ...], int]:
        if budget is None or not self.token_budget_manager.enabled:
            return turns, 0

        trim_result = self.token_budget_manager.trim_messages(turns, budget)
        self.token_budget_manager.record_trim(trim_result, ctx, component="history")
        content = trim_result.content
        if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
            kept = tuple(item for item in content if isinstance(item, ConversationTurn))
        else:
            kept = turns
        return kept, trim_result.dropped_count

    def _summarize_history_if_needed(
        self,
        turns: tuple[ConversationTurn, ...],
        existing_summary: str | None,
        budget: TokenBudget | int | None,
        ctx: RequestContext | None,
    ) -> SummaryResult | None:
        if not self.summary_enabled:
            return None
        # ConversationSummarizer 自己负责触发条件、敏感信息清洗、失败降级和 trace。
        # Manager 只接收结果并决定是否替换内部 summary，避免摘要逻辑泄漏到历史读取门面。
        return self.summarizer.summarize_if_needed(
            turns=turns,
            existing_summary=existing_summary,
            budget=budget,
            ctx=ctx,
        )

    def _load_persisted_summary(self, session_id: str) -> Mapping[str, object] | None:
        """读取持久化摘要；读取失败 fail-open（返回 None，退回现算行为）。"""

        if self.summary_store is None:
            return None
        try:
            get_summary = getattr(self.summary_store, "get_summary", None)
            if not callable(get_summary):
                return None
            return get_summary(session_id)
        except Exception:  # noqa: BLE001 - 摘要读失败不阻断主链路
            return None

    def _persist_summary(self, session_id: str, summary: str, source_message_count: int) -> None:
        """摘要增量生成后写回持久层；写失败 fail-open（下次请求重新生成）。"""

        if self.summary_store is None:
            return
        try:
            save_summary = getattr(self.summary_store, "save_summary", None)
            if not callable(save_summary):
                return
            save_summary(session_id, summary, source_message_count=source_message_count)
        except Exception:  # noqa: BLE001 - 摘要写失败不阻断主链路
            return

    def _handle_load_error(
        self,
        session_id: str,
        ctx: RequestContext | None,
        exc: _MemoryReadError,
    ) -> ConversationContext:
        _ = session_id
        self._record_load(
            ctx,
            status="error",
            message_count=0,
            summary_used=False,
            trimmed_count=0,
            error_code="INTERNAL_ERROR",
        )
        if not self.fail_open_on_read_error:
            raise InternalAppError(
                internal_message=f"{exc.__class__.__name__}",
                origin_module="app.memory.conversation_manager",
            ) from exc
        return ConversationContext(
            summary=None,
            recent_messages=(),
            history_metadata=_history_metadata(
                raw_message_count=0,
                visible_message_count=0,
                recent_message_count=0,
                trimmed_count=0,
                summary_used=False,
                load_failed=True,
            ),
        )

    def _record_load(
        self,
        ctx: RequestContext | None,
        *,
        status: Literal["ok", "error"],
        message_count: int,
        summary_used: bool,
        trimmed_count: int,
        error_code: str | None = None,
    ) -> None:
        if ctx is None:
            return
        self.trace_logger.record_event(
            "conversation.load",
            ctx,
            status=status,
            error_code=error_code,
            message_count=message_count,
            summary_used=summary_used,
            trimmed_count=trimmed_count,
        )

    def _record_save(
        self,
        ctx: RequestContext | None,
        *,
        status: Literal["ok", "error", "skipped"],
        success: bool,
        error_code: str | None = None,
    ) -> None:
        if ctx is None:
            return
        self.trace_logger.record_event(
            "conversation.save",
            ctx,
            status=status,
            error_code=error_code,
            success=success,
        )

    def _record_clear(
        self,
        ctx: RequestContext | None,
        *,
        status: Literal["ok", "error"],
        success: bool,
        error_code: str | None = None,
    ) -> None:
        if ctx is None:
            return
        self.trace_logger.record_event(
            "conversation.clear",
            ctx,
            status=status,
            error_code=error_code,
            success=success,
        )


def _thread_config(session_id: str) -> dict[str, object]:
    return {"configurable": {"thread_id": session_id}}


def _coerce_checkpoint(raw_checkpoint: object | None) -> Mapping[str, object] | None:
    if raw_checkpoint is None:
        return None

    checkpoint_attr = getattr(raw_checkpoint, "checkpoint", None)
    if isinstance(checkpoint_attr, Mapping):
        return cast(Mapping[str, object], checkpoint_attr)

    if isinstance(raw_checkpoint, tuple) and raw_checkpoint:
        first_item = raw_checkpoint[0]
        if isinstance(first_item, Mapping):
            return cast(Mapping[str, object], first_item)

    if isinstance(raw_checkpoint, Mapping):
        nested_checkpoint = raw_checkpoint.get("checkpoint")
        if isinstance(nested_checkpoint, Mapping):
            return cast(Mapping[str, object], nested_checkpoint)
        return cast(Mapping[str, object], raw_checkpoint)

    return None


def _extract_messages(checkpoint: Mapping[str, object] | None) -> tuple[object, ...]:
    if checkpoint is None:
        return ()
    channel_values = checkpoint.get("channel_values")
    if isinstance(channel_values, Mapping):
        messages = channel_values.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, str | bytes | bytearray):
            return tuple(messages)

    messages = checkpoint.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes | bytearray):
        return tuple(messages)
    return ()


def _extract_summary(checkpoint: Mapping[str, object] | None) -> str | None:
    if checkpoint is None:
        return None
    channel_values = checkpoint.get("channel_values")
    if isinstance(channel_values, Mapping):
        summary = channel_values.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary

    summary = checkpoint.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def _messages_to_turns(messages: Sequence[object]) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    for message in messages:
        role = _message_role(message)
        if role is None:
            continue
        content = _message_content(message)
        if content is None or not content.strip():
            continue
        turns.append(
            ConversationTurn(
                role=role,
                content=content,
                timestamp=_message_timestamp(message),
            )
        )
    return tuple(turns)


def _message_role(message: object) -> ConversationRole | None:
    if isinstance(message, Mapping):
        raw_role = message.get("role", message.get("type"))
        role_text = str(raw_role).lower() if raw_role is not None else ""
    else:
        role_text = message.__class__.__name__.lower()

    if role_text in {"user", "human"} or "human" in role_text:
        return "user"
    if role_text in {"assistant", "ai"} or "assistant" in role_text or "aimessage" in role_text:
        return "assistant"
    return None


def _message_content(message: object) -> str | None:
    if isinstance(message, Mapping):
        raw_content = message.get("content")
    else:
        raw_content = getattr(message, "content", None)

    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, Sequence) and not isinstance(
        raw_content,
        str | bytes | bytearray,
    ):
        text_parts = [_text_from_content_block(block) for block in raw_content]
        joined = "".join(part for part in text_parts if part)
        return joined or None
    return None


def _text_from_content_block(block: object) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        block_type = block.get("type")
        text = block.get("text")
        if block_type == "text" and isinstance(text, str):
            return text
    return None


def _message_timestamp(message: object) -> str:
    if isinstance(message, Mapping):
        timestamp = message.get("timestamp") or message.get("created_at")
        if isinstance(timestamp, str) and timestamp.strip():
            return timestamp
    else:
        timestamp_attr = getattr(message, "timestamp", None)
        if isinstance(timestamp_attr, str) and timestamp_attr.strip():
            return timestamp_attr
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, Mapping):
            timestamp_value = additional_kwargs.get("timestamp")
            if isinstance(timestamp_value, str) and timestamp_value.strip():
                return timestamp_value
    return datetime.now(UTC).isoformat()


def _history_metadata(
    *,
    raw_message_count: int,
    visible_message_count: int,
    recent_message_count: int,
    trimmed_count: int,
    summary_used: bool,
    load_failed: bool,
) -> JsonObject:
    return {
        "raw_message_count": raw_message_count,
        "message_count": visible_message_count,
        "recent_message_count": recent_message_count,
        "trimmed_count": trimmed_count,
        "summary_used": summary_used,
        "load_failed": load_failed,
    }


def _resolve_save_method(
    memory_saver: object,
) -> Callable[[str, str, str, Mapping[str, object] | None], object] | None:
    for method_name in ("save_turn", "append_turn"):
        method = getattr(memory_saver, method_name, None)
        if callable(method):
            return cast(
                Callable[[str, str, str, Mapping[str, object] | None], object],
                method,
            )
    return None
