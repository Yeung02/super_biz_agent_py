"""受控历史摘要生成器。

ConversationSummarizer 是阶段 2 的独立边界模块：它只负责判断是否需要摘要、
清洗可摘要的历史、调用可注入 LLM 或本地兜底摘要、裁剪摘要长度和记录 trace。
它不修改 MemorySaver checkpoint，不决定 API 响应，也不把工具错误或原始 payload
当作事实证据写入长期上下文。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, TypeAlias

from app.config import config
from app.core.errors import JsonObject
from app.core.request_context import RequestContext
from app.core.token_budget import (
    TokenBudget,
    TokenBudgetManager,
    token_budget_manager as default_token_budget_manager,
)
from app.observability.tracing import TraceLogger

SummaryTriggerReason: TypeAlias = Literal["turn_count", "token_budget"]
_LLM_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    TimeoutError,
)

_SUMMARY_VERSION = "v1"
_MAX_PROMPT_HISTORY_CHARS = 12_000
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;，。]+['\"]?"
)
_SECRET_VALUE_RE = re.compile(r"\b(sk|ak)-[A-Za-z0-9_\-]{4,}\b")
_URL_RE = re.compile(r"https?://[^\s,;，。\"']+")
_STACK_RE = re.compile(
    r"(?is)(traceback|stacktrace|exception stack|调用栈|堆栈)[^。.!?\n]*(?:[。.!?]|\n|$)"
)
_TOOL_PAYLOAD_RE = re.compile(
    r"(?is)(raw_payload|full_payload|payload|tool_error|isError|工具返回|工具错误)"
    r"[^。.!?\n]*(?:[。.!?]|\n|$)"
)
_DANGEROUS_WORD_RE = re.compile(
    r"(?i)(raw_payload|full_payload|stacktrace|traceback|tool_error|api[_-]?key)"
)


class SummaryLLM(Protocol):
    """ConversationSummarizer 需要的最小同步 LLM 接口。"""

    def invoke(self, prompt: str) -> object:
        """同步生成摘要。真实 LLM 和测试 fake 都只需要实现这个方法。"""


@dataclass(frozen=True)
class SummaryResult:
    """摘要尝试结果。

    `triggered=true` 只表示满足触发条件并尝试摘要；`summary=None` 且 `error_code`
    有值表示摘要失败，但调用方仍应保留最近完整轮次继续主请求。
    """

    triggered: bool
    trigger_reason: SummaryTriggerReason | None
    summary: str | None
    metadata: JsonObject = field(default_factory=dict)
    error_code: str | None = None


class ConversationSummarizer:
    """为 ConversationManager 提供安全摘要能力。

    LLM 被设计为可注入依赖：单元测试传 fake，生产可传 ChatQwen/ChatOpenAI。未注入
    LLM 时使用本地抽取式兜底摘要，避免 ConversationManager 的普通单测或回滚路径误访问
    DashScope。无论哪种来源，摘要前后都会做安全清洗和预算裁剪。
    """

    def __init__(
        self,
        *,
        llm: SummaryLLM | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
        trace_logger: TraceLogger | None = None,
        max_source_turns: int | None = None,
        max_summary_tokens: int | None = None,
        max_prompt_history_chars: int = _MAX_PROMPT_HISTORY_CHARS,
        enabled: bool | None = None,
    ) -> None:
        self.llm = llm
        self.token_budget_manager = token_budget_manager or default_token_budget_manager
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        self.max_source_turns = max(
            1,
            int(
                max_source_turns
                if max_source_turns is not None
                else config.conversation_summary_max_source_turns
            ),
        )
        self.max_summary_tokens = max(
            1,
            int(
                max_summary_tokens
                if max_summary_tokens is not None
                else config.conversation_summary_max_tokens
            ),
        )
        self.max_prompt_history_chars = max(256, int(max_prompt_history_chars))
        self.enabled = bool(
            enabled if enabled is not None else config.conversation_summary_enabled
        )

    def summarize_if_needed(
        self,
        *,
        turns: Sequence[object],
        existing_summary: str | None,
        budget: TokenBudget | int | None,
        ctx: RequestContext | None = None,
    ) -> SummaryResult:
        """按轮次和 token 阈值决定是否摘要。

        触发条件只看“安全可见历史”的数量和估算 token。摘要失败时返回 error_code，
        但不抛出原始异常，防止主请求因为历史摘要这个辅助能力而失败。
        """

        source_turns = tuple(turns)
        source_turn_count = len(source_turns)
        trigger_reason = self._trigger_reason(source_turns, existing_summary, budget)
        metadata = self._metadata(source_turn_count, trigger_reason)
        if not self.enabled or trigger_reason is None:
            return SummaryResult(
                triggered=False,
                trigger_reason=None,
                summary=existing_summary,
                metadata=metadata,
            )

        prompt = self._build_prompt(source_turns, existing_summary, budget)
        try:
            raw_summary = self._generate_summary(prompt, source_turns, existing_summary)
            safe_summary = self._limit_summary_tokens(_sanitize_text(raw_summary))
        except _LLM_ERRORS as exc:
            error_code = _summary_error_code(exc)
            self._record_summary(
                ctx,
                status="error",
                trigger_reason=trigger_reason,
                source_turn_count=source_turn_count,
                summary_tokens=0,
                error_code=error_code,
            )
            return SummaryResult(
                triggered=True,
                trigger_reason=trigger_reason,
                summary=None,
                metadata=metadata,
                error_code=error_code,
            )

        summary_tokens = self.token_budget_manager.estimate_tokens(safe_summary).token_count
        self._record_summary(
            ctx,
            status="ok",
            trigger_reason=trigger_reason,
            source_turn_count=source_turn_count,
            summary_tokens=summary_tokens,
            error_code=None,
        )
        return SummaryResult(
            triggered=True,
            trigger_reason=trigger_reason,
            summary=safe_summary,
            metadata=metadata,
        )

    def _trigger_reason(
        self,
        turns: Sequence[object],
        existing_summary: str | None,
        budget: TokenBudget | int | None,
    ) -> SummaryTriggerReason | None:
        if len(turns) > self.max_source_turns:
            return "turn_count"
        if budget is None:
            return None

        history_budget = _history_budget_tokens(budget)
        estimated_tokens = self.token_budget_manager.estimate_tokens(
            (*turns, existing_summary or "")
        ).token_count
        if estimated_tokens > history_budget:
            return "token_budget"
        return None

    def _build_prompt(
        self,
        turns: Sequence[object],
        existing_summary: str | None,
        budget: TokenBudget | int | None,
    ) -> str:
        """构建受控摘要 prompt。

        prompt 只包含清洗后的用户/助手文本，并明确禁止总结工具错误为事实。历史内容会先按
        字符上限截断，避免摘要请求本身因为旧 payload 过大再次撑爆上下文。
        """

        safe_lines = _safe_history_lines(turns)
        history_text = "\n".join(safe_lines)
        if len(history_text) > self.max_prompt_history_chars:
            history_text = history_text[-self.max_prompt_history_chars :]
        summary_budget = _summary_budget_tokens(
            budget,
            fallback=self.max_summary_tokens,
        )
        previous_summary = _sanitize_text(existing_summary or "无")
        return "\n".join(
            (
                "请生成一段受控会话摘要，用于后续对话上下文。",
                "只保留：用户目标、已确认事实、未解决事项、用户显式偏好。",
                "不要包含工具原始 payload、密钥、内部 URL、错误堆栈、未验证工具错误。",
                "不要把工具失败文本、异常信息或未验证诊断当作业务事实。",
                f"摘要长度不超过约 {summary_budget} token。",
                f"已有摘要：{previous_summary}",
                "待摘要历史：",
                history_text,
            )
        )

    def _generate_summary(
        self,
        prompt: str,
        turns: Sequence[object],
        existing_summary: str | None,
    ) -> str:
        if self.llm is None:
            return _extractive_summary(turns, existing_summary)

        response = self.llm.invoke(prompt)
        content = _extract_llm_content(response)
        if not content.strip():
            raise ValueError("empty summary response")
        return content

    def _limit_summary_tokens(self, summary: str) -> str:
        """把摘要裁剪到配置预算内。

        TokenBudgetManager 的通用文本裁剪会附加截断标记；这里再做一次循环校验，是为了
        保证摘要预算测试和生产 trace 中的 `summary_tokens` 不被截断标记反向撑超。
        """

        text = summary.strip()
        if not text:
            return ""

        while self.token_budget_manager.estimate_tokens(text).token_count > self.max_summary_tokens:
            next_length = max(1, int(len(text) * 0.85))
            if next_length >= len(text):
                next_length = len(text) - 1
            text = text[:next_length].rstrip()
            if not text:
                return ""
        return text

    def _metadata(
        self,
        source_turn_count: int,
        trigger_reason: SummaryTriggerReason | None,
    ) -> JsonObject:
        return {
            "summary_version": _SUMMARY_VERSION,
            "source_turn_count": source_turn_count,
            "updated_at": datetime.now(UTC).isoformat(),
            "trigger_reason": trigger_reason,
        }

    def _record_summary(
        self,
        ctx: RequestContext | None,
        *,
        status: Literal["ok", "error"],
        trigger_reason: SummaryTriggerReason,
        source_turn_count: int,
        summary_tokens: int,
        error_code: str | None,
    ) -> None:
        if ctx is None:
            return
        self.trace_logger.record_event(
            "conversation.summary",
            ctx,
            status=status,
            trigger_reason=trigger_reason,
            source_turn_count=source_turn_count,
            summary_tokens=summary_tokens,
            error_code=error_code,
        )


def _safe_history_lines(turns: Sequence[object]) -> list[str]:
    lines: list[str] = []
    for turn in turns:
        role = _turn_role(turn)
        content = _turn_content(turn)
        if role is None or not content:
            continue
        safe_content = _sanitize_text(content)
        if not safe_content:
            continue
        role_label = "用户" if role == "user" else "助手"
        lines.append(f"{role_label}: {safe_content}")
    return lines


def _turn_role(turn: object) -> Literal["user", "assistant"] | None:
    if isinstance(turn, Mapping):
        raw_role = turn.get("role")
    else:
        raw_role = getattr(turn, "role", None)
    role_text = str(raw_role).lower() if raw_role is not None else ""
    if role_text in {"user", "human"}:
        return "user"
    if role_text in {"assistant", "ai"}:
        return "assistant"
    return None


def _turn_content(turn: object) -> str:
    if isinstance(turn, Mapping):
        content = turn.get("content")
    else:
        content = getattr(turn, "content", None)
    return content if isinstance(content, str) else ""


def _sanitize_text(text: str) -> str:
    """清洗摘要输入输出中的高风险内容。

    对工具 payload 和错误堆栈采取“整段移除”，而不仅是脱敏关键字段。原因是工具错误文本
    很容易被模型误写成“已确认事实”；整段删除能更好地保持事实边界。
    """

    sanitized = _TOOL_PAYLOAD_RE.sub("", text)
    sanitized = _STACK_RE.sub("", sanitized)
    sanitized = _SECRET_ASSIGNMENT_RE.sub("", sanitized)
    sanitized = _SECRET_VALUE_RE.sub("", sanitized)
    sanitized = _URL_RE.sub("", sanitized)
    sanitized = _DANGEROUS_WORD_RE.sub("", sanitized)
    return _normalize_space(sanitized)


def _normalize_space(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip(" ，,;；")
    return normalized.strip()


def _extract_llm_content(response: object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        content = response.get("content")
        if isinstance(content, str):
            return content
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)


def _extractive_summary(turns: Sequence[object], existing_summary: str | None) -> str:
    """本地兜底摘要。

    兜底摘要只抽取清洗后的最近几条用户/助手文本，不调用外部服务。它牺牲表达质量来换取
    回滚安全性，确保摘要功能开启后不会让单元测试或短期演示环境依赖真实模型服务。
    """

    lines = _safe_history_lines(turns)
    selected = lines[-6:]
    parts: list[str] = []
    if existing_summary:
        parts.append(f"已有摘要：{_sanitize_text(existing_summary)}")
    if selected:
        parts.append("近期上下文：" + "；".join(selected))
    return "。".join(part for part in parts if part)


def _history_budget_tokens(budget: TokenBudget | int) -> int:
    return max(1, budget.history_tokens if isinstance(budget, TokenBudget) else int(budget))


def _summary_budget_tokens(budget: TokenBudget | int | None, *, fallback: int) -> int:
    if budget is None:
        return fallback
    if isinstance(budget, TokenBudget):
        return max(1, budget.summary_tokens)
    return max(1, int(budget))


def _summary_error_code(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "LLM_TIMEOUT"
    return "LLM_PROVIDER_ERROR"
