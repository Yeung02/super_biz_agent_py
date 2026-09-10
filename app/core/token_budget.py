"""Token 预算、估算、裁剪和 usage 记录。

ISSUE-012 的职责边界很窄：本模块只负责把 prompt 相关输入变成可估算、可裁剪、
可追踪的上下文组件，不生成摘要正文、不改变检索排序、不决定 HTTP 响应状态。
调用方如果遇到当前用户问题或系统安全约束超限，必须走 AppError/fallback 等上层路径，
不能在这里静默截断关键输入。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, cast

from app.core.errors import JsonObject, JsonValue, RequestTooLargeError
from app.core.request_context import RequestContext
from app.observability.tracing import TraceLogger

TokenScenario: TypeAlias = Literal["rag_chat", "aiops_plan", "aiops_execute", "aiops_report"]
EstimateMethod: TypeAlias = Literal["tokenizer", "char_fallback"]
TrimComponent: TypeAlias = Literal["debug", "tool_result", "rag_chunk", "history", "summary"]
Tokenizer: TypeAlias = Callable[[str], int]

_DEFAULT_CONTEXT_WINDOW = 32_768
_TRUNCATED_SUFFIX = "...[truncated]"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s*")
_DROPPED_TOOL_KEYS = frozenset(
    (
        "debug",
        "debug_info",
        "raw",
        "raw_payload",
        "payload",
        "full_payload",
        "full_response",
        "stack",
        "stacktrace",
        "trace",
        "traceback",
    )
)
_PRIORITY_TOOL_KEYS = (
    "status",
    "error",
    "summary",
    "result",
    "data",
    "items",
    "metrics",
    "evidence",
)
_DEFAULT_BUDGET_RATIOS: dict[str, dict[str, float]] = {
    "rag_chat": {
        "input": 0.15,
        "history": 0.15,
        # 用户长期记忆画像注入槽位（从 history 划出），未召回时槽位闲置不影响其他组件。
        "memory": 0.05,
        "rag_context": 0.35,
        "tool_result": 0.10,
        "output": 0.20,
    },
    "aiops_plan": {
        "input": 0.15,
        "history": 0.05,
        "rag_context": 0.35,
        "tool_result": 0.25,
        "output": 0.20,
    },
    "aiops_execute": {
        "input": 0.20,
        "history": 0.0,
        "rag_context": 0.10,
        "tool_result": 0.50,
        "output": 0.20,
    },
    "aiops_report": {
        "input": 0.10,
        "history": 0.10,
        "rag_context": 0.20,
        "tool_result": 0.40,
        "output": 0.20,
    },
}


@dataclass(frozen=True)
class TokenEstimate:
    """一次 token 估算结果。

    `estimated=true` 说明没有可靠 tokenizer，只能用字符估算。调用方在 trace 或 usage
    中保留该标记，后续排查成本偏差时能区分真实模型 usage 和本地估算。
    """

    token_count: int
    method: EstimateMethod
    estimated: bool


@dataclass(frozen=True)
class BudgetAllocation:
    """各 prompt 组件的预算槽位。"""

    input_tokens: int
    history_tokens: int
    rag_context_tokens: int
    tool_result_tokens: int
    summary_tokens: int
    output_tokens: int
    # 用户长期记忆画像槽位；带默认值保证旧调用方按位置构造仍然兼容。
    memory_tokens: int = 0


@dataclass(frozen=True)
class TokenBudget:
    """单次请求或单个 Agent 节点的 token 预算。

    属性代理保留 `budget.history_tokens` 等直观访问方式，同时把实际值集中在
    `allocation`，后续如果需要扩展更多槽位不会破坏已有调用方。
    """

    scenario: str
    model: str
    model_context_window: int
    allocation: BudgetAllocation
    enabled: bool = True

    @property
    def input_tokens(self) -> int:
        return self.allocation.input_tokens

    @property
    def history_tokens(self) -> int:
        return self.allocation.history_tokens

    @property
    def rag_context_tokens(self) -> int:
        return self.allocation.rag_context_tokens

    @property
    def tool_result_tokens(self) -> int:
        return self.allocation.tool_result_tokens

    @property
    def summary_tokens(self) -> int:
        return self.allocation.summary_tokens

    @property
    def memory_tokens(self) -> int:
        return self.allocation.memory_tokens

    @property
    def output_tokens(self) -> int:
        return self.allocation.output_tokens

    @property
    def input_capacity(self) -> int:
        """返回除输出外的可用输入总预算。"""

        return self.model_context_window - self.output_tokens

    def with_limits(
        self,
        *,
        input_tokens: int | None = None,
        history_tokens: int | None = None,
        rag_context_tokens: int | None = None,
        tool_result_tokens: int | None = None,
        summary_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> TokenBudget:
        """返回一个只覆盖指定槽位的新预算。

        单测和局部调用经常需要用很小的预算触发裁剪；这里用不可变对象复制，避免修改全局
        config 或长生命周期 manager，保证每个 issue 的测试可独立回滚。
        """

        allocation = BudgetAllocation(
            input_tokens=_positive_or_current(input_tokens, self.input_tokens),
            history_tokens=_positive_or_current(history_tokens, self.history_tokens),
            rag_context_tokens=_positive_or_current(rag_context_tokens, self.rag_context_tokens),
            tool_result_tokens=_positive_or_current(tool_result_tokens, self.tool_result_tokens),
            summary_tokens=_positive_or_current(summary_tokens, self.summary_tokens),
            output_tokens=_positive_or_current(output_tokens, self.output_tokens),
        )
        return replace(self, allocation=allocation)


@dataclass(frozen=True)
class TrimAction:
    """一次裁剪动作的可追踪摘要。"""

    component: TrimComponent
    original_tokens: int
    trimmed_tokens: int
    dropped_count: int
    reason: str


@dataclass(frozen=True)
class TrimResult:
    """裁剪结果的统一 envelope。

    `content` 可以是消息、chunk、工具摘要或纯文本。这里使用 object 而不是 Any，是为了
    明确本模块只透传同形对象，不对后续模块的具体类型做承诺。
    """

    content: object
    trimmed: bool
    original_tokens: int
    trimmed_tokens: int
    dropped_count: int
    actions: tuple[TrimAction, ...] = ()


@dataclass(frozen=True)
class TokenContext:
    """组合上下文裁剪的输入模型。

    它不是 HTTP schema，也不依赖后续 RAG/Conversation 模型；只是让当前 issue 能测试
    内部契约要求的裁剪顺序，并给后续 Orchestrator/ConversationManager 预留稳定入口。
    """

    system_prompt: str
    current_question: str
    debug_notes: tuple[str, ...] = ()
    tool_results: tuple[object, ...] = ()
    rag_chunks: tuple[object, ...] = ()
    history_messages: tuple[object, ...] = ()
    summary: str | None = None


@dataclass(frozen=True)
class ContextTrimResult:
    """组合上下文裁剪结果。"""

    context: TokenContext
    actions: tuple[TrimAction, ...]


@dataclass(frozen=True)
class TokenUsage:
    """模型 usage 与成本估算结果。"""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool
    estimated_cost: float

    def to_dict(self) -> JsonObject:
        """转换为 trace 可写入的 JSON 对象。"""

        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "estimated_cost": self.estimated_cost,
        }


class TokenBudgetManager:
    """Token 预算管理器。

    Manager 保持纯边界职责：估算、分配、裁剪、usage trace。它不调用 LLM、不做摘要生成、
    不决定 RAG chunk 的相关性阈值，也不修改 MemorySaver checkpoint。
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        model_context_windows: Mapping[str, int] | None = None,
        budget_ratios: Mapping[str, Mapping[str, float]] | None = None,
        max_output_tokens: int | None = None,
        min_reserved_output_tokens: int | None = None,
        summary_budget: int | None = None,
        token_input_prices_per_1k: Mapping[str, float] | None = None,
        token_output_prices_per_1k: Mapping[str, float] | None = None,
        trace_logger: TraceLogger | None = None,
        enabled: bool | None = None,
    ) -> None:
        from app.config import config

        self.tokenizer = tokenizer
        self.model_context_windows = dict(
            model_context_windows
            if model_context_windows is not None
            else getattr(config, "token_model_context_windows", {"default": _DEFAULT_CONTEXT_WINDOW})
        )
        self.budget_ratios = _normalize_ratios(
            budget_ratios
            if budget_ratios is not None
            else getattr(config, "token_budget_ratios", _DEFAULT_BUDGET_RATIOS)
        )
        self.max_output_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else getattr(config, "token_max_output_tokens", 2048)
        )
        self.min_reserved_output_tokens = int(
            min_reserved_output_tokens
            if min_reserved_output_tokens is not None
            else getattr(config, "token_min_reserved_output_tokens", 512)
        )
        self.summary_budget = int(
            summary_budget
            if summary_budget is not None
            else getattr(config, "token_summary_budget", 1024)
        )
        self.token_input_prices_per_1k = dict(
            token_input_prices_per_1k
            if token_input_prices_per_1k is not None
            else getattr(config, "token_input_prices_per_1k", {})
        )
        self.token_output_prices_per_1k = dict(
            token_output_prices_per_1k
            if token_output_prices_per_1k is not None
            else getattr(config, "token_output_prices_per_1k", {})
        )
        self.enabled = bool(
            enabled if enabled is not None else getattr(config, "token_budget_enabled", True)
        )
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=getattr(config, "trace_jsonl_path", "logs/trace.jsonl"),
            enabled=bool(getattr(config, "trace_enabled", True)),
        )

    def allocate(
        self,
        scenario: str,
        model: str,
        ctx: RequestContext | None = None,
        *,
        current_input: str | None = None,
        system_prompt: str | None = None,
    ) -> TokenBudget:
        """为指定场景分配 token 预算。

        分配只读配置，不访问模型服务。若调用方传入当前问题，则立即校验硬上限；
        当前问题和系统安全约束都不能靠裁剪“悄悄变短”，否则会改变用户真实意图或安全策略。
        """

        resolved_model = model or "default"
        window = self._model_context_window(resolved_model)
        output_tokens = self._reserved_output_tokens(window)
        usable_tokens = max(1, window - output_tokens)
        ratios = self.budget_ratios.get(scenario, self.budget_ratios["rag_chat"])

        input_tokens = math.floor(usable_tokens * ratios["input"])
        history_tokens = math.floor(usable_tokens * ratios["history"])
        rag_context_tokens = math.floor(usable_tokens * ratios["rag_context"])
        tool_result_tokens = math.floor(usable_tokens * ratios["tool_result"])
        summary_tokens = min(self.summary_budget, max(0, history_tokens // 2))
        # memory 槽位允许缺省（旧配置无该键时为 0，行为等价于未启用画像）。
        memory_tokens = math.floor(usable_tokens * ratios.get("memory", 0.0))

        budget = TokenBudget(
            scenario=scenario,
            model=resolved_model,
            model_context_window=window,
            allocation=BudgetAllocation(
                input_tokens=input_tokens,
                history_tokens=history_tokens,
                rag_context_tokens=rag_context_tokens,
                tool_result_tokens=tool_result_tokens,
                summary_tokens=summary_tokens,
                output_tokens=output_tokens,
                memory_tokens=memory_tokens,
            ),
            enabled=self.enabled,
        )

        if current_input is not None:
            self.validate_current_input(current_input, hard_limit_tokens=budget.input_tokens)
        if system_prompt is not None:
            self._validate_system_prompt(system_prompt, hard_limit_tokens=budget.input_tokens)
        if ctx is not None:
            self._record_allocate(ctx, budget)
        return budget

    def estimate_tokens(self, text_or_messages: object) -> TokenEstimate:
        """估算字符串、消息列表或同形对象的 token 数。

        tokenizer 失败时只捕获预期运行时错误并降级，不裸 except；降级估算偏保守，优先保护
        上下文窗口不被低估撑爆。
        """

        text = _stringify_for_token_count(text_or_messages)
        if self.tokenizer is not None:
            try:
                token_count = max(0, int(self.tokenizer(text)))
                return TokenEstimate(
                    token_count=token_count,
                    method="tokenizer",
                    estimated=False,
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return _estimate_by_chars(text)

        return _estimate_by_chars(text)

    def validate_current_input(self, text: str, *, hard_limit_tokens: int) -> None:
        """校验当前用户输入是否超过硬上限。

        这是 ISSUE-012 的关键安全边界：当前问题承载用户真实意图，不能像历史或 RAG
        chunk 一样裁剪。超限时抛 `REQUEST_TOO_LARGE`，由 API/SSE adapter 保持旧字段兼容。
        """

        estimate = self.estimate_tokens(text)
        if estimate.token_count > max(1, hard_limit_tokens):
            raise RequestTooLargeError("请求内容过长。")

    def trim_messages(self, messages: Sequence[object], budget: TokenBudget | int) -> TrimResult:
        """按历史预算裁剪消息列表，保留系统消息和最近完整上下文。

        ConversationManager 后续会负责“旧历史转摘要”；当前 issue 只做确定性裁剪，保证
        20 轮长对话不会整体塞入 LLM。
        """

        max_tokens = _history_budget_tokens(budget)
        original_messages = tuple(messages)
        original_tokens = self.estimate_tokens(original_messages).token_count
        if original_tokens <= max_tokens:
            return TrimResult(
                content=original_messages,
                trimmed=False,
                original_tokens=original_tokens,
                trimmed_tokens=original_tokens,
                dropped_count=0,
            )

        protected = [message for message in original_messages if _message_role(message) == "system"]
        candidates = [
            message for message in original_messages if _message_role(message) != "system"
        ]
        kept: list[object] = []
        for message in reversed(candidates):
            trial = [*protected, message, *reversed(kept)]
            if self.estimate_tokens(trial).token_count <= max_tokens or not kept:
                kept.append(message)
            else:
                break

        trimmed_messages = tuple([*protected, *reversed(kept)])
        trimmed_tokens = self.estimate_tokens(trimmed_messages).token_count
        dropped_count = max(0, len(original_messages) - len(trimmed_messages))
        action = TrimAction(
            component="history",
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            reason="drop_old_history_keep_system_and_recent",
        )
        return TrimResult(
            content=trimmed_messages,
            trimmed=dropped_count > 0,
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            actions=(action,) if dropped_count > 0 else (),
        )

    def trim_chunks(self, chunks: Sequence[object], budget: TokenBudget | int) -> TrimResult:
        """按 RAG 预算裁剪 chunk。

        当前阶段没有引入 `app.rag.models`，因此只读取 `chunk_id/content_hash/content/score`
        这类同形字段。裁剪时保留高分和去重后的短证据，不改变上游检索排序实现。
        """

        max_tokens = _rag_budget_tokens(budget)
        original_chunks = tuple(chunks)
        original_tokens = self.estimate_tokens(original_chunks).token_count
        deduped = _dedupe_chunks(original_chunks)
        sorted_chunks = sorted(deduped, key=_chunk_score, reverse=True)
        kept: list[object] = []
        used_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = self.estimate_tokens(_chunk_content(chunk)).token_count
            if used_tokens + chunk_tokens <= max_tokens:
                kept.append(chunk)
                used_tokens += chunk_tokens

        trimmed_tokens = self.estimate_tokens(tuple(kept)).token_count
        dropped_count = max(0, len(original_chunks) - len(kept))
        action = TrimAction(
            component="rag_chunk",
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            reason="drop_low_score_and_duplicate_chunks",
        )
        return TrimResult(
            content=tuple(kept),
            trimmed=dropped_count > 0,
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            actions=(action,) if dropped_count > 0 else (),
        )

    def trim_tool_result(self, result: object, budget: TokenBudget | int) -> TrimResult:
        """按工具结果预算压缩成功工具输出。

        错误工具结果不能作为事实证据，本方法不会尝试从错误文本里提取上下文；成功结果
        先移除 raw/debug/payload 等高风险字段，再按 token 预算做最后截断。
        """

        max_tokens = _tool_budget_tokens(budget)
        if _is_error_tool_result(result):
            original_tokens = self.estimate_tokens(result).token_count
            return TrimResult(
                content=result,
                trimmed=False,
                original_tokens=original_tokens,
                trimmed_tokens=original_tokens,
                dropped_count=0,
            )

        raw_value = _tool_data(result)
        original_tokens = self.estimate_tokens(raw_value).token_count
        compacted, compact_trimmed, dropped_count = _compact_tool_value(_to_json_value(raw_value))
        serialized = _serialize_json(compacted)
        if self.estimate_tokens(serialized).token_count > max_tokens:
            content: object = _truncate_text_by_tokens(serialized, max_tokens)
            trimmed = True
        else:
            content = compacted
            trimmed = compact_trimmed
        trimmed_tokens = self.estimate_tokens(content).token_count
        action = TrimAction(
            component="tool_result",
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            reason="compact_tool_result_then_truncate",
        )
        return TrimResult(
            content=content,
            trimmed=trimmed,
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=dropped_count,
            actions=(action,) if trimmed else (),
        )

    def trim_text(self, text: str, budget: TokenBudget | int) -> TrimResult:
        """裁剪普通文本，供 AIOps planner/replanner 这类旧 prompt 拼接路径使用。"""

        max_tokens = _summary_budget_tokens(budget)
        original_tokens = self.estimate_tokens(text).token_count
        if original_tokens <= max_tokens:
            return TrimResult(
                content=text,
                trimmed=False,
                original_tokens=original_tokens,
                trimmed_tokens=original_tokens,
                dropped_count=0,
            )
        trimmed_text = _truncate_text_by_tokens(text, max_tokens)
        trimmed_tokens = self.estimate_tokens(trimmed_text).token_count
        action = TrimAction(
            component="summary",
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=1,
            reason="truncate_text_to_budget",
        )
        return TrimResult(
            content=trimmed_text,
            trimmed=True,
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=1,
            actions=(action,),
        )

    def trim_summary(self, summary: str | None, budget: TokenBudget | int) -> TrimResult:
        """按句子边界裁剪历史摘要文本。

        摘要正文可以被压缩，但不能在这里重新生成。若单句本身过长，则退回安全截断并保留
        truncated 标记，避免无限制扩大 prompt。
        """

        text = summary or ""
        max_tokens = _summary_budget_tokens(budget)
        original_tokens = self.estimate_tokens(text).token_count
        if original_tokens <= max_tokens:
            return TrimResult(
                content=text,
                trimmed=False,
                original_tokens=original_tokens,
                trimmed_tokens=original_tokens,
                dropped_count=0,
            )

        sentences = [sentence for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence]
        kept_sentences: list[str] = []
        for sentence in sentences:
            trial = "".join([*kept_sentences, sentence])
            if self.estimate_tokens(trial).token_count <= max_tokens:
                kept_sentences.append(sentence)
            else:
                break
        trimmed_text = "".join(kept_sentences).strip() or _truncate_text_by_tokens(text, max_tokens)
        trimmed_tokens = self.estimate_tokens(trimmed_text).token_count
        action = TrimAction(
            component="summary",
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=max(1, len(sentences) - len(kept_sentences)),
            reason="compress_summary_on_sentence_boundary",
        )
        return TrimResult(
            content=trimmed_text,
            trimmed=True,
            original_tokens=original_tokens,
            trimmed_tokens=trimmed_tokens,
            dropped_count=action.dropped_count,
            actions=(action,),
        )

    def trim_context(self, context: TokenContext, budget: TokenBudget) -> ContextTrimResult:
        """按内部契约的全局优先级裁剪组合上下文。

        顺序固定为：debug -> tool_result -> rag_chunk -> history -> summary。当前问题和系统
        安全约束只校验、不裁剪；如果它们超限，上层应返回 REQUEST_TOO_LARGE 或 fallback。
        """

        self.validate_current_input(context.current_question, hard_limit_tokens=budget.input_tokens)
        self._validate_system_prompt(context.system_prompt, hard_limit_tokens=budget.input_tokens)

        actions: list[TrimAction] = []
        trimmed_context = context
        if trimmed_context.debug_notes:
            original_tokens = self.estimate_tokens(trimmed_context.debug_notes).token_count
            action = TrimAction(
                component="debug",
                original_tokens=original_tokens,
                trimmed_tokens=0,
                dropped_count=len(trimmed_context.debug_notes),
                reason="drop_debug_notes_first",
            )
            actions.append(action)
            trimmed_context = replace(trimmed_context, debug_notes=())

        if trimmed_context.tool_results:
            per_tool_budget = max(1, budget.tool_result_tokens // len(trimmed_context.tool_results))
            tool_results: list[object] = []
            tool_actions: list[TrimAction] = []
            for tool_result in trimmed_context.tool_results:
                result = self.trim_tool_result(tool_result, per_tool_budget)
                tool_results.append(result.content)
                tool_actions.extend(result.actions)
            if tool_actions:
                actions.append(_merge_actions("tool_result", tool_actions))
            trimmed_context = replace(trimmed_context, tool_results=tuple(tool_results))

        if trimmed_context.rag_chunks:
            result = self.trim_chunks(trimmed_context.rag_chunks, budget.rag_context_tokens)
            actions.extend(result.actions)
            trimmed_context = replace(trimmed_context, rag_chunks=cast(tuple[object, ...], result.content))

        if trimmed_context.history_messages:
            result = self.trim_messages(trimmed_context.history_messages, budget.history_tokens)
            actions.extend(result.actions)
            trimmed_context = replace(
                trimmed_context,
                history_messages=cast(tuple[object, ...], result.content),
            )

        if trimmed_context.summary:
            result = self.trim_summary(trimmed_context.summary, budget.summary_tokens)
            actions.extend(result.actions)
            trimmed_context = replace(trimmed_context, summary=cast(str, result.content))

        return ContextTrimResult(context=trimmed_context, actions=tuple(actions))

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        ctx: RequestContext | None = None,
        estimated: bool = False,
    ) -> TokenUsage:
        """记录 LLM usage 和成本估算。

        非流式响应可传真实 usage；流式或 SDK 无 usage 时传估算值并标记 estimated。成本配置
        默认可以为 0，避免在价格未确认时输出伪精确费用。
        """

        safe_input_tokens = max(0, int(input_tokens))
        safe_output_tokens = max(0, int(output_tokens))
        cost = (
            safe_input_tokens * self._input_price(model)
            + safe_output_tokens * self._output_price(model)
        ) / 1000
        usage = TokenUsage(
            model=model,
            input_tokens=safe_input_tokens,
            output_tokens=safe_output_tokens,
            total_tokens=safe_input_tokens + safe_output_tokens,
            estimated=estimated,
            estimated_cost=round(cost, 8),
        )
        if ctx is not None:
            self.trace_logger.record_event("token.usage", ctx, usage=usage.to_dict())
        return usage

    def record_trim(
        self,
        result: TrimResult,
        ctx: RequestContext | None,
        *,
        component: TrimComponent,
    ) -> None:
        """记录裁剪 trace。

        裁剪动作本身不应写入原文，只记录 token 数量和 dropped_count。这样既满足排障需要，
        又不会把历史对话、工具 payload 或知识库 chunk 全量落盘。
        """

        if ctx is None or not result.trimmed:
            return
        self.trace_logger.record_event(
            "token.trim",
            ctx,
            component=component,
            original_tokens=result.original_tokens,
            trimmed_tokens=result.trimmed_tokens,
            dropped_count=result.dropped_count,
            action_count=len(result.actions),
            estimated=True,
        )

    def _record_allocate(self, ctx: RequestContext, budget: TokenBudget) -> None:
        """记录预算分配 trace，不包含 prompt 原文。"""

        self.trace_logger.record_event(
            "token.allocate",
            ctx,
            scenario=budget.scenario,
            model=budget.model,
            model_context_window=budget.model_context_window,
            input_tokens=budget.input_tokens,
            history_tokens=budget.history_tokens,
            rag_context_tokens=budget.rag_context_tokens,
            tool_result_tokens=budget.tool_result_tokens,
            summary_tokens=budget.summary_tokens,
            output_tokens=budget.output_tokens,
            enabled=budget.enabled,
        )

    def _model_context_window(self, model: str) -> int:
        return max(
            1,
            int(
                self.model_context_windows.get(
                    model,
                    self.model_context_windows.get("default", _DEFAULT_CONTEXT_WINDOW),
                )
            ),
        )

    def _reserved_output_tokens(self, window: int) -> int:
        configured = max(self.min_reserved_output_tokens, self.max_output_tokens)
        return min(configured, max(1, window - 1))

    def _validate_system_prompt(self, text: str, *, hard_limit_tokens: int) -> None:
        estimate = self.estimate_tokens(text)
        if estimate.token_count > max(1, hard_limit_tokens):
            raise RequestTooLargeError("请求内容过长。")

    def _input_price(self, model: str) -> float:
        return float(
            self.token_input_prices_per_1k.get(
                model,
                self.token_input_prices_per_1k.get("default", 0.0),
            )
        )

    def _output_price(self, model: str) -> float:
        return float(
            self.token_output_prices_per_1k.get(
                model,
                self.token_output_prices_per_1k.get("default", 0.0),
            )
        )


def _positive_or_current(value: int | None, current: int) -> int:
    return max(0, current if value is None else int(value))


def _normalize_ratios(
    ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    normalized = {scenario: dict(values) for scenario, values in ratios.items()}
    normalized.setdefault("rag_chat", dict(_DEFAULT_BUDGET_RATIOS["rag_chat"]))
    for scenario, default_values in _DEFAULT_BUDGET_RATIOS.items():
        normalized.setdefault(scenario, dict(default_values))
        for key, value in default_values.items():
            normalized[scenario].setdefault(key, value)
    return normalized


def _history_budget_tokens(budget: TokenBudget | int) -> int:
    return max(1, budget.history_tokens if isinstance(budget, TokenBudget) else int(budget))


def _rag_budget_tokens(budget: TokenBudget | int) -> int:
    return max(1, budget.rag_context_tokens if isinstance(budget, TokenBudget) else int(budget))


def _tool_budget_tokens(budget: TokenBudget | int) -> int:
    return max(1, budget.tool_result_tokens if isinstance(budget, TokenBudget) else int(budget))


def _summary_budget_tokens(budget: TokenBudget | int) -> int:
    return max(1, budget.summary_tokens if isinstance(budget, TokenBudget) else int(budget))


def _stringify_for_token_count(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
        return _serialize_json(_to_json_value(value))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return "\n".join(_stringify_for_token_count(item) for item in value)

    content_attr = getattr(value, "content", None)
    if isinstance(content_attr, str):
        return content_attr
    return str(value)


def _estimate_by_chars(text: str) -> TokenEstimate:
    if not text:
        return TokenEstimate(token_count=0, method="char_fallback", estimated=True)
    cjk_chars = sum(1 for char in text if _is_cjk_char(char))
    non_cjk_chars = max(0, len(text) - cjk_chars)
    # 中文、英文按不同经验比例累加；再和全英文估算取更大值。这样既不会低估混合文本，
    # 也不会把大段英文 JSON 按中文比例严重高估，避免无谓丢弃最近完整轮次。
    mixed_estimate = cjk_chars / 1.5 + non_cjk_chars / 4
    conservative_estimate = max(mixed_estimate, len(text) / 4)
    token_count = max(1, math.ceil(conservative_estimate))
    return TokenEstimate(token_count=token_count, method="char_fallback", estimated=True)


def _message_role(message: object) -> str:
    if isinstance(message, Mapping):
        role = message.get("role")
        return str(role).lower() if role is not None else ""
    message_type = message.__class__.__name__.lower()
    if "system" in message_type:
        return "system"
    if "human" in message_type or "user" in message_type:
        return "user"
    if "ai" in message_type or "assistant" in message_type:
        return "assistant"
    return ""


def _dedupe_chunks(chunks: Sequence[object]) -> tuple[object, ...]:
    best_by_key: dict[str, object] = {}
    for chunk in chunks:
        key = _chunk_key(chunk)
        current = best_by_key.get(key)
        if current is None or _chunk_is_better(chunk, current):
            best_by_key[key] = chunk
    return tuple(best_by_key.values())


def _chunk_is_better(candidate: object, current: object) -> bool:
    candidate_score = _chunk_score(candidate)
    current_score = _chunk_score(current)
    if candidate_score != current_score:
        return candidate_score > current_score
    return len(_chunk_content(candidate)) < len(_chunk_content(current))


def _chunk_key(chunk: object) -> str:
    for attr_name in ("chunk_id", "content_hash", "doc_id"):
        value = _read_field(chunk, attr_name)
        if isinstance(value, str) and value.strip():
            return f"{attr_name}:{value.strip()}"
    return f"content:{_chunk_content(chunk)}"


def _chunk_score(chunk: object) -> float:
    for attr_name in ("normalized_score", "score", "relevance_score"):
        value = _read_field(chunk, attr_name)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _chunk_content(chunk: object) -> str:
    for attr_name in ("content", "text", "page_content", "evidence_text"):
        value = _read_field(chunk, attr_name)
        if isinstance(value, str):
            return value
    return _stringify_for_token_count(chunk)


def _read_field(value: object, field_name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _is_error_tool_result(result: object) -> bool:
    if bool(getattr(result, "is_error", False)):
        return True
    status = getattr(result, "status", None)
    if status in ("error", "timeout", "unauthorized"):
        return True
    checker = getattr(result, "is_evidence_usable", None)
    if callable(checker):
        try:
            return not bool(checker())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return True
    return False


def _tool_data(result: object) -> object:
    if isinstance(result, Mapping):
        return result.get("data", result)
    data = getattr(result, "data", None)
    return result if data is None else data


def _compact_tool_value(value: JsonValue) -> tuple[JsonValue, bool, int]:
    if isinstance(value, dict):
        compacted: JsonObject = {}
        dropped_count = 0
        for key in _PRIORITY_TOOL_KEYS:
            if key in value:
                child, child_trimmed, child_dropped = _compact_tool_value(value[key])
                compacted[key] = child
                dropped_count += child_dropped
                if child_trimmed:
                    dropped_count += 1

        for key, child_value in value.items():
            if key in compacted:
                continue
            if key.lower() in _DROPPED_TOOL_KEYS:
                dropped_count += 1
                continue
            if len(compacted) >= 8:
                dropped_count += 1
                continue
            child, child_trimmed, child_dropped = _compact_tool_value(child_value)
            compacted[key] = child
            dropped_count += child_dropped + (1 if child_trimmed else 0)
        return compacted, dropped_count > 0, dropped_count

    if isinstance(value, list):
        max_items = 5
        compacted_items: list[JsonValue] = []
        dropped_count = max(0, len(value) - max_items)
        for item in value[:max_items]:
            child, child_trimmed, child_dropped = _compact_tool_value(item)
            compacted_items.append(child)
            dropped_count += child_dropped + (1 if child_trimmed else 0)
        return compacted_items, dropped_count > 0, dropped_count

    return value, False, 0


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_json_value(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_value(model_dump())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _to_json_value(dict_method())
    return str(value)


def _serialize_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    max_chars = max(1, math.floor(max_tokens * 1.5))
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATED_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(_TRUNCATED_SUFFIX)].rstrip() + _TRUNCATED_SUFFIX


def _merge_actions(component: TrimComponent, actions: Sequence[TrimAction]) -> TrimAction:
    """合并同一组件的多个裁剪动作，保持全局优先级列表简洁。"""

    return TrimAction(
        component=component,
        original_tokens=sum(action.original_tokens for action in actions),
        trimmed_tokens=sum(action.trimmed_tokens for action in actions),
        dropped_count=sum(action.dropped_count for action in actions),
        reason=";".join(action.reason for action in actions),
    )


def _is_cjk_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


token_budget_manager = TokenBudgetManager()
