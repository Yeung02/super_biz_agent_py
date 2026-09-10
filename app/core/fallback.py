"""Fallback 决策与响应转换。

本模块只负责“失败后能否安全降级”的纯决策，不做重试、不重新调用工具、不吞掉输入错误。
这样 API、SSE 与后续 Agent 编排都可以复用同一张策略矩阵，同时不会把下游服务异常、
内部 URL 或密钥样式文本原样暴露给用户。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from app.core.errors import AppError, JsonObject
from app.core.request_context import RequestContext

if TYPE_CHECKING:
    from app.observability.tracing import TraceLogger

FallbackScenario: TypeAlias = Literal["chat", "chat_stream", "aiops", "file", "health"]

_CONVERSATIONAL_SCENARIOS: frozenset[FallbackScenario] = frozenset(
    ("chat", "chat_stream", "aiops")
)
_NO_FALLBACK_SCENARIOS: frozenset[FallbackScenario] = frozenset(("file", "health"))
_MAX_PARTIAL_ANSWER_CHARS = 1200

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|authorization|credential)\s*=\s*([^\s,;]+)"
)
_SECRET_VALUE_RE = re.compile(r"\b(sk|ak)-[A-Za-z0-9_\-]{4,}\b")
_URL_RE = re.compile(r"https?://[^\s,;\"')\]]+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class FallbackEvidence:
    """Fallback 决策可使用的已验证证据摘要。

    这里刻意不导入真实 ToolResult，避免 FallbackManager 反向依赖 LangChain/MCP。
    调用方只传入摘要、证据数量和“同形”的工具结果即可；错误工具结果只影响 reason，
    不能被当成事实写入 partial_answer。
    """

    summary: str | None = None
    partial_answer: str | None = None
    evidence_count: int = 0
    tool_results: Sequence[object] = ()


@dataclass(frozen=True)
class FallbackPolicy:
    """单个错误码的降级策略。

    `use_partial_evidence` 用来区分“可基于已验证证据给摘要”和“必须 no-answer”的场景；
    例如 RAG_EMPTY_RESULT 不能伪造 citation，也不能把空检索包装成确定回答。
    """

    reason_code: str
    safe_message: str
    allowed_scenarios: frozenset[FallbackScenario]
    should_continue_stream: bool = False
    use_partial_evidence: bool = True


@dataclass(frozen=True)
class FallbackResult:
    """FallbackManager 的稳定输出。

    API 层只读取这个对象转换 HTTP/SSE 响应，不再自行拼接错误文案，避免同一错误码在不同
    handler 里表现不一致，也便于 trace 统一记录 fallback_used 和 reason_code。
    """

    fallback_used: bool
    reason_code: str
    safe_message: str
    partial_answer: str | None
    should_continue_stream: bool
    scenario: FallbackScenario
    evidence_count: int = 0
    citations: tuple[JsonObject, ...] = ()
    retryable: bool = False


_POLICY_MATRIX: dict[str, FallbackPolicy] = {
    "LLM_PROVIDER_ERROR": FallbackPolicy(
        reason_code="LLM_PROVIDER_ERROR",
        safe_message="模型服务暂时不可用，以下是已收集到的信息。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "LLM_TIMEOUT": FallbackPolicy(
        reason_code="LLM_TIMEOUT",
        safe_message="模型响应超时，请稍后重试。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "LLM_EMPTY_RESPONSE": FallbackPolicy(
        reason_code="LLM_EMPTY_RESPONSE",
        safe_message="模型返回为空，无法生成可靠答案。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
        use_partial_evidence=False,
    ),
    "RAG_EMPTY_RESULT": FallbackPolicy(
        reason_code="RAG_EMPTY_RESULT",
        safe_message="知识库没有找到相关依据。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
        use_partial_evidence=False,
    ),
    "VECTOR_STORE_UNAVAILABLE": FallbackPolicy(
        reason_code="VECTOR_STORE_UNAVAILABLE",
        safe_message="当前无法访问知识库。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "TOOL_TIMEOUT": FallbackPolicy(
        reason_code="TOOL_TIMEOUT",
        safe_message="实时工具不可用，以下基于已获得信息。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "TOOL_EXECUTION_ERROR": FallbackPolicy(
        reason_code="TOOL_EXECUTION_ERROR",
        safe_message="内部工具执行失败，系统已记录问题。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "AGENT_MAX_STEP_EXCEEDED": FallbackPolicy(
        reason_code="AGENT_MAX_STEP_EXCEEDED",
        safe_message="诊断未完整完成，以下是已完成步骤摘要。",
        allowed_scenarios=_CONVERSATIONAL_SCENARIOS,
    ),
    "SSE_STREAM_INTERRUPTED": FallbackPolicy(
        reason_code="SSE_STREAM_INTERRUPTED",
        safe_message="流式输出中断，以下内容可能不完整。",
        allowed_scenarios=frozenset(("chat_stream", "aiops")),
    ),
}


class FallbackManager:
    """集中执行 fallback 策略矩阵。

    Manager 不做外部 IO 和下游调用，只把 AppError、场景和已验证证据映射成稳定结果。
    这能保证关闭 `fallback_enabled` 时仍走旧结构化错误路径，也能避免 API handler 变厚。
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        from app.config import config
        from app.observability.tracing import TraceLogger

        self.enabled = config.fallback_enabled if enabled is None else enabled
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )

    def decide(
        self,
        error: AppError,
        ctx: RequestContext | None,
        *,
        scenario: FallbackScenario,
        evidence: FallbackEvidence | None = None,
    ) -> FallbackResult:
        """根据错误码、场景和证据决定是否降级。

        输入错误、文件处理和健康检查不能被 LLM fallback 掩盖；这类失败必须保留原本的
        HTTP/SSE 错误语义，避免旧前端把用户输入问题或依赖健康问题误认为成功回答。
        """

        policy = _POLICY_MATRIX.get(error.code)
        evidence_count = _count_usable_evidence(evidence)
        fallback_used = (
            self.enabled
            and scenario not in _NO_FALLBACK_SCENARIOS
            and policy is not None
            and scenario in policy.allowed_scenarios
            and error.requires_fallback()
        )
        partial_answer = (
            _select_partial_answer(evidence)
            if fallback_used and policy is not None and policy.use_partial_evidence
            else None
        )
        safe_message = policy.safe_message if policy is not None else error.user_message
        result = FallbackResult(
            fallback_used=fallback_used,
            reason_code=error.code,
            safe_message=safe_message,
            partial_answer=partial_answer,
            should_continue_stream=bool(policy.should_continue_stream) if fallback_used else False,
            scenario=scenario,
            evidence_count=evidence_count,
            retryable=error.retryable,
        )
        self._record_decision(result, ctx)
        return result

    def for_chat(
        self,
        error: AppError,
        ctx: RequestContext | None,
        *,
        evidence: FallbackEvidence | None = None,
    ) -> FallbackResult:
        """Chat 非流式场景的快捷入口，避免 handler 直接写场景字符串。"""

        return self.decide(error, ctx, scenario="chat", evidence=evidence)

    def for_aiops(
        self,
        error: AppError,
        ctx: RequestContext | None,
        *,
        evidence: FallbackEvidence | None = None,
    ) -> FallbackResult:
        """AIOps SSE 场景的快捷入口，后续编排层也可复用同一决策。"""

        return self.decide(error, ctx, scenario="aiops", evidence=evidence)

    def to_response_fields(self, result: FallbackResult) -> JsonObject:
        """把降级结果转为 Chat 旧 `data` 字段。

        非降级结果只暴露 fallback 元数据，让 API handler 继续返回 AppError envelope；降级结果
        则保持旧前端依赖的 `success/answer/errorMessage` 字段，同时新增契约要求的
        `fallback_used/reason_code/citations`。
        """

        if not result.fallback_used:
            return {"fallback_used": False, "reason_code": result.reason_code}

        answer = result.partial_answer or result.safe_message
        return {
            "success": True,
            "answer": answer,
            "errorMessage": None,
            "fallback_used": True,
            "reason_code": result.reason_code,
            "citations": [dict(citation) for citation in result.citations],
        }

    def to_sse_event(
        self,
        result: FallbackResult,
        ctx: RequestContext | None,
    ) -> dict[str, str]:
        """生成兼容旧 `event: message` 的 fallback SSE 事件。

        对外事件名仍保持 message，真正的新类型放在 JSON 的 `type=fallback`，这样旧客户端不会因
        监听事件名变化而失效，新客户端可以根据 `data.type` 做差异化 UI。
        """

        payload = _attach_trace(
            {
                "type": "fallback",
                "stage": "fallback",
                "message": result.safe_message,
                "data": result.partial_answer or result.safe_message,
                "fallback_used": result.fallback_used,
                "reason_code": result.reason_code,
                "partial_answer": result.partial_answer,
                "should_continue_stream": result.should_continue_stream,
                "citations": [dict(citation) for citation in result.citations],
            },
            ctx,
        )
        return _sse_message(payload)

    def to_sse_done_event(
        self,
        result: FallbackResult,
        ctx: RequestContext | None,
        *,
        session_id: str | None = None,
    ) -> dict[str, str]:
        """生成 fallback 后的 done 事件。

        Chat/AIOps SSE 都需要在最终 done 中暴露 fallback_used；同时保留 `data.answer`，
        兼容只读取 done.data 的旧前端实现。
        """

        answer = result.partial_answer or result.safe_message
        payload: JsonObject = {
            "type": "done",
            "stage": "fallback_done",
            "message": result.safe_message,
            "data": {
                "answer": answer,
                "fallback_used": result.fallback_used,
                "reason_code": result.reason_code,
                "citations": [dict(citation) for citation in result.citations],
            },
            "answer": answer,
            "fallback_used": result.fallback_used,
            "reason_code": result.reason_code,
            "citations": [dict(citation) for citation in result.citations],
        }
        if session_id is not None:
            payload["session_id"] = session_id
        return _sse_message(_attach_trace(payload, ctx))

    def _record_decision(self, result: FallbackResult, ctx: RequestContext | None) -> None:
        """记录 fallback 决策 trace，不写入原始异常或证据全文。"""

        if ctx is None:
            return
        self.trace_logger.record_event(
            "fallback.decide",
            ctx,
            reason_code=result.reason_code,
            fallback_used=result.fallback_used,
            evidence_count=result.evidence_count,
            partial_answer=result.partial_answer is not None,
            scenario=result.scenario,
        )


def _count_usable_evidence(evidence: FallbackEvidence | None) -> int:
    """统计可作为事实依据的证据数量，错误工具结果不计入。"""

    if evidence is None:
        return 0
    tool_evidence_count = sum(
        1
        for tool_result in evidence.tool_results
        if _is_usable_tool_result(tool_result)
    )
    count = max(0, evidence.evidence_count, tool_evidence_count)
    if count == 0 and _has_text(evidence.partial_answer or evidence.summary):
        return 1
    return count


def _select_partial_answer(evidence: FallbackEvidence | None) -> str | None:
    """选择可展示的部分答案，并做脱敏和长度控制。

    partial_answer 只能来自上游明确传入的已验证摘要，不能从 AppError.internal_message 或工具错误文本
    里提取，避免把内部异常、密钥、内网地址包装成“事实依据”。
    """

    if evidence is None or _count_usable_evidence(evidence) <= 0:
        return None
    text = evidence.partial_answer or evidence.summary
    if not _has_text(text):
        return None
    sanitized = _sanitize_user_visible_text(text)
    if not sanitized:
        return None
    if len(sanitized) > _MAX_PARTIAL_ANSWER_CHARS:
        return sanitized[:_MAX_PARTIAL_ANSWER_CHARS].rstrip() + "..."
    return sanitized


def _is_usable_tool_result(tool_result: object) -> bool:
    """按 ToolResult 同形协议判断工具结果是否能当作证据。

    真实 ToolResult 和测试 fake 都提供 `is_evidence_usable()`；兜底分支只读 `status/is_error`，
    不导入工具模块，避免 fallback 单测依赖真实 MCP 或 LangChain。
    """

    checker = getattr(tool_result, "is_evidence_usable", None)
    if callable(checker):
        try:
            return bool(checker())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    status = getattr(tool_result, "status", None)
    is_error = getattr(tool_result, "is_error", True)
    return status == "success" and is_error is False


def _sanitize_user_visible_text(text: str | None) -> str:
    """清理 fallback 可见文本中的敏感信息和控制字符。"""
    if text is None:
        return ""
    sanitized = _CONTROL_CHAR_RE.sub("", text)
    sanitized = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)
    sanitized = _SECRET_VALUE_RE.sub("<redacted>", sanitized)
    sanitized = _URL_RE.sub("<internal-url-redacted>", sanitized)
    return sanitized.strip()


def sanitize_answer_text(text: str | None) -> str:
    """答案级脱敏公开入口（ISSUE-B Critic 复用）。

    与 fallback partial_answer 使用完全相同的正则集合：不重复定义规则，保证降级
    文案与 Critic 修订稿的脱敏行为始终一致。只删不改，与 sanitize_answer_anchors
    的字符级剥离哲学相同。
    """

    return _sanitize_user_visible_text(text)


def count_sensitive_hits(text: str | None) -> int:
    """统计文本中的敏感信息命中数（密钥赋值/密钥值/内网 URL）。

    供 Critic 预检 trace 使用：命中数 > 0 说明草稿需要确定性剥离，事件里保留
    具体计数便于观测泄露频率，但不记录命中内容本身。
    """

    if not text:
        return 0
    return (
        len(_SECRET_ASSIGNMENT_RE.findall(text))
        + len(_SECRET_VALUE_RE.findall(text))
        + len(_URL_RE.findall(text))
    )


def _has_text(text: str | None) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _attach_trace(payload: JsonObject, ctx: RequestContext | None) -> JsonObject:
    if ctx is not None:
        payload["trace_id"] = ctx.trace_id
        payload["request_id"] = ctx.request_id
    return payload


def _sse_message(payload: JsonObject) -> dict[str, str]:
    return {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}
