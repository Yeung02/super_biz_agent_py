"""JSONL trace 写入与 metrics 旁路记录。

阶段 1A 的 TraceLogger 只负责可串联的本地 JSONL 事件；ISSUE-029 在不改变业务
调用点的前提下，把现有 trace 事件旁路映射到 MetricsRecorder。trace/metrics 任一
写入失败都只降级为 warning，不影响 API、SSE 或 Agent 主流程。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.core.errors import AppError, JsonObject, JsonValue
from app.core.request_context import RequestContext
from app.observability.metrics import MetricsRecorder

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)
_SAFE_TOKEN_METRIC_KEYS = frozenset(
    (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "summary_tokens",
        "history_tokens",
        "rag_context_tokens",
        "tool_result_tokens",
        "total_tokens",
    )
)


@dataclass(frozen=True)
class TraceSpan:
    """内存中的轻量 span 句柄，用于把 start/end 事件用同一个 span_id 串起来。"""

    name: str
    ctx: RequestContext
    span_id: str
    parent_span_id: str | None
    started_monotonic: float


class TraceLogger:
    """写入最小 JSONL trace event 的工具类。"""

    def __init__(
        self,
        *,
        trace_jsonl_path: str,
        enabled: bool = True,
        metrics_recorder: MetricsRecorder | None = None,
        metrics_enabled: bool | None = None,
        metrics_jsonl_path: str | None = None,
    ) -> None:
        self.trace_jsonl_path = Path(trace_jsonl_path)
        self.enabled = enabled
        self.metrics_recorder = metrics_recorder or _default_metrics_recorder(
            enabled=metrics_enabled,
            metrics_jsonl_path=metrics_jsonl_path,
        )

    def record_event(self, name: str, ctx: RequestContext, **fields: JsonValue) -> None:
        """记录一个结构化事件。

        事件字段先经过 JSON 化和敏感字段脱敏；写入失败只记录 warning。metrics
        使用同一份已脱敏事件做旁路记录，因此不会把密钥类字段直接落盘，也不会要求
        Tool/RAG/Token 模块重复写 metrics 代码。
        """

        event: JsonObject = {
            "timestamp": time.time(),
            "name": name,
            "status": fields.pop("status", "ok"),
            "span_id": str(fields.pop("span_id", _new_span_id())),
            "parent_span_id": fields.pop("parent_span_id", None),
            "latency_ms": fields.pop("latency_ms", None),
            "error_code": fields.pop("error_code", None),
            **ctx.to_trace_fields(),
        }
        for key, value in fields.items():
            event[key] = _sanitize_value(key, value)
        self._record_metrics_from_event(name, ctx, event)
        if not self.enabled:
            return
        self._write_event(event)

    def start_span(self, name: str, ctx: RequestContext, **fields: JsonValue) -> TraceSpan:
        """创建 span 并写入 `.start` 事件。"""

        span = TraceSpan(
            name=name,
            ctx=ctx,
            span_id=_new_span_id(),
            parent_span_id=_string_or_none(fields.pop("parent_span_id", None)),
            started_monotonic=time.monotonic(),
        )
        self.record_event(
            f"{name}.start",
            ctx,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            status="start",
            **fields,
        )
        return span

    def end_span(self, span: TraceSpan, **fields: JsonValue) -> None:
        """结束 span 并记录耗时。"""

        latency_ms = round((time.monotonic() - span.started_monotonic) * 1000, 3)
        self.record_event(
            f"{span.name}.end",
            span.ctx,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            latency_ms=latency_ms,
            **fields,
        )

    def record_error(self, error: AppError, ctx: RequestContext) -> None:
        """记录 AppError 的安全元数据，不写入 internal_message 原文。"""

        self.record_event(
            "request.error",
            ctx,
            status="error",
            error_code=error.code,
            http_status=error.http_status,
            retryable=error.retryable,
            fallback_required=error.fallback_required,
            origin_module=error.origin_module,
            latency_ms=ctx.latency_ms(),
        )

    def record_usage(self, ctx: RequestContext, usage: JsonObject) -> None:
        """记录模型/工具 usage；阶段 1A 仅提供通用入口，后续 token issue 再接入。"""

        self.record_event("usage", ctx, usage=usage)

    def _write_event(self, event: JsonObject) -> None:
        try:
            self.trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"trace 写入失败，已降级为不中断主流程: {exc.__class__.__name__}")

    def _record_metrics_from_event(
        self,
        name: str,
        ctx: RequestContext,
        event: JsonObject,
    ) -> None:
        """把已有 trace 事件映射为 metrics。

        ISSUE-029 的关键兼容点是“不改 API、不让 handler 变厚”。因此这里按事件名和
        已有字段做保守映射：能识别的 request/tool/rag/fallback/usage 写 metrics，
        识别不了的 debug 或内部事件仍只保留 trace。
        """

        try:
            if name in {"request.end", "request.error"}:
                self._record_request_metric(name, ctx, event)
            elif name in {"tool.end", "tool.error"}:
                self._record_tool_metric(ctx, event)
            elif name in {"rag.retrieve.end", "rag.context.build", "rag.citation.build"}:
                self._record_rag_metric(ctx, event)
            elif name == "fallback.decide":
                self._record_fallback_metric(ctx, event)
            elif name in {"usage", "token.usage", "orchestrator.usage"}:
                self._record_usage_metrics(ctx, event)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "metrics 映射失败，已降级为只写 trace: {}",
                exc.__class__.__name__,
            )

    def _record_request_metric(
        self,
        name: str,
        ctx: RequestContext,
        event: JsonObject,
    ) -> None:
        http_status = (
            _int_or_none(event.get("status_code"))
            if name == "request.end"
            else _int_or_none(event.get("http_status"))
        )
        self.metrics_recorder.record_request(
            ctx,
            status=_str_or_default(event.get("status"), "ok"),
            latency_ms=_float_or_none(event.get("latency_ms")),
            http_status=http_status,
            error_code=_str_or_none(event.get("error_code")),
            fallback_used=_bool_or_none(event.get("fallback_used")),
        )

    def _record_tool_metric(self, ctx: RequestContext, event: JsonObject) -> None:
        tool_status = _str_or_none(event.get("tool_result.status"))
        event_status = _str_or_default(event.get("status"), "ok")
        self.metrics_recorder.record_tool(
            ctx,
            tool_name=_str_or_default(event.get("tool_name"), "unknown"),
            status=tool_status or event_status,
            latency_ms=_float_or_none(event.get("latency_ms")),
            error_code=_str_or_none(event.get("error_code")),
            raw_size=_int_or_none(event.get("raw_size")),
            preview_size=_int_or_none(event.get("preview_size")),
            trimmed=_bool_or_none(event.get("trimmed")),
            evidence_usable=_bool_or_none(event.get("evidence_usable")),
            input_preview=event.get("input_preview"),
            output_preview=event.get("output_preview"),
            source=_str_or_none(event.get("source")),
        )

    def _record_rag_metric(self, ctx: RequestContext, event: JsonObject) -> None:
        final_count = _first_int(
            event.get("final_count"),
            event.get("used_chunk_count"),
            event.get("citation_count"),
        )
        self.metrics_recorder.record_rag(
            ctx,
            status=_str_or_default(event.get("status"), "ok"),
            latency_ms=_float_or_none(event.get("latency_ms")),
            candidate_count=_first_int(event.get("candidate_count"), event.get("input_chunk_count")),
            final_count=final_count,
            retrieval_count=final_count,
            citation_count=_int_or_none(event.get("citation_count")),
            dropped_count=_first_int(
                event.get("dropped_low_score_count"),
                event.get("dropped_chunk_count"),
                event.get("dropped_invalid_count"),
            ),
            empty_reason=_str_or_none(event.get("empty_reason")),
            no_answer=_bool_or_none(event.get("no_answer")),
            error_code=_str_or_none(event.get("error_code")),
        )

    def _record_fallback_metric(self, ctx: RequestContext, event: JsonObject) -> None:
        self.metrics_recorder.record_fallback(
            ctx,
            fallback_used=bool(event.get("fallback_used")),
            reason_code=_str_or_default(event.get("reason_code"), "UNKNOWN"),
            scenario=_str_or_none(event.get("scenario")),
            evidence_count=_int_or_none(event.get("evidence_count")),
            latency_ms=_float_or_none(event.get("latency_ms")),
        )

    def _record_usage_metrics(self, ctx: RequestContext, event: JsonObject) -> None:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            return
        model = _str_or_default(usage.get("model"), "unknown")
        input_tokens = _int_or_none(usage.get("input_tokens")) or 0
        output_tokens = _int_or_none(usage.get("output_tokens")) or 0
        total_tokens = _int_or_none(usage.get("total_tokens")) or input_tokens + output_tokens
        estimated = bool(usage.get("estimated"))
        self.metrics_recorder.record_llm(
            ctx,
            model=model,
            status=_str_or_default(event.get("status"), "ok"),
            latency_ms=_float_or_none(event.get("latency_ms")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated=estimated,
            error_code=_str_or_none(event.get("error_code")),
        )
        self.metrics_recorder.record_cost(
            ctx,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=_float_or_none(usage.get("estimated_cost")) or 0.0,
            estimated=estimated,
        )


def _new_span_id() -> str:
    return f"span_{uuid.uuid4().hex}"


def _default_metrics_recorder(
    *,
    enabled: bool | None,
    metrics_jsonl_path: str | None,
) -> MetricsRecorder:
    from app.config import config

    return MetricsRecorder(
        metrics_jsonl_path=metrics_jsonl_path
        or getattr(config, "metrics_jsonl_path", "logs/metrics.jsonl"),
        enabled=bool(getattr(config, "metrics_enabled", True) if enabled is None else enabled),
    )


def _sanitize_value(key: str, value: JsonValue) -> JsonValue:
    if key in _SAFE_TOKEN_METRIC_KEYS:
        return value
    if _SENSITIVE_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {child_key: _sanitize_value(child_key, child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    return value


def _string_or_none(value: JsonValue) -> str | None:
    if value is None:
        return None
    return str(value)


def _str_or_none(value: JsonValue) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _str_or_default(value: JsonValue, default: str) -> str:
    return _str_or_none(value) or default


def _int_or_none(value: JsonValue) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _first_int(*values: JsonValue) -> int | None:
    for value in values:
        converted = _int_or_none(value)
        if converted is not None:
            return converted
    return None


def _float_or_none(value: JsonValue) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _bool_or_none(value: JsonValue) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
