"""结构化 metrics JSONL 记录器。

ISSUE-029 在阶段 1A 最小 trace 的基础上补齐可用于趋势和回归分析的指标记录。
MetricsRecorder 只做本地 JSONL 写入和字段脱敏，不接 Prometheus、不建 dashboard，
也不改变任何 HTTP API 响应；写入失败时仅降级为 warning，避免可观测性组件影响主流程。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

from app.core.errors import JsonObject, JsonValue
from app.core.request_context import RequestContext

MetricType = Literal["request", "tool", "rag", "llm", "fallback", "cost"]

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
class MetricsRecord:
    """单条 metrics 记录的内部结构。

    公共 request/trace 字段由 RequestContext 统一提供，业务指标放入 values 后再
    扁平化输出。这样评估 runner 或后续趋势脚本只需按 `metric_type` 过滤 JSONL，
    同时仍能用同一个 `trace_id/request_id` 回查 trace 明细。
    """

    metric_type: MetricType
    ctx: RequestContext
    status: str
    latency_ms: float | None = None
    error_code: str | None = None
    values: JsonObject = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> JsonObject:
        """转换为可落盘 JSON 对象，并在最终边界统一脱敏。"""

        record: JsonObject = {
            "timestamp": self.timestamp,
            "metric_type": self.metric_type,
            "trace_id": self.ctx.trace_id,
            "request_id": self.ctx.request_id,
            "session_id": self.ctx.session_id,
            "tenant_id": self.ctx.tenant_id,
            "user_id": self.ctx.user_id,
            "method": self.ctx.method,
            "path": self.ctx.path,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
        }
        for key, value in self.values.items():
            record[key] = _sanitize_value(key, value)
        return record


class MetricsRecorder:
    """按请求、工具、RAG、LLM、fallback 和 cost 维度写入 metrics。

    该类保持无业务决策：调用方传入已经计算好的 latency、token、cost 或计数字段。
    MetricsRecorder 只负责补充 trace/request 关联字段、脱敏以及非阻塞写入。
    """

    def __init__(self, *, metrics_jsonl_path: str, enabled: bool = True) -> None:
        self.metrics_jsonl_path = Path(metrics_jsonl_path)
        self.enabled = enabled

    def record_request(
        self,
        ctx: RequestContext,
        *,
        status: str,
        latency_ms: float | None = None,
        http_status: int | None = None,
        error_code: str | None = None,
        fallback_used: bool | None = None,
    ) -> None:
        """记录 HTTP 请求级指标。

        请求指标只保存状态码、耗时和是否 fallback，不保存请求 body。这样既满足
        API 契约要求的 trace/request 串联，也不会把用户问题或上传内容写入 metrics。
        """

        values: JsonObject = {}
        if http_status is not None:
            values["http_status"] = http_status
        if fallback_used is not None:
            values["fallback_used"] = fallback_used
        self._record(
            MetricsRecord(
                metric_type="request",
                ctx=ctx,
                status=status,
                latency_ms=latency_ms,
                error_code=error_code,
                values=values,
            )
        )

    def record_tool(
        self,
        ctx: RequestContext,
        *,
        tool_name: str,
        status: str,
        latency_ms: float | None = None,
        error_code: str | None = None,
        raw_size: int | None = None,
        preview_size: int | None = None,
        trimmed: bool | None = None,
        evidence_usable: bool | None = None,
        input_preview: JsonValue = None,
        output_preview: JsonValue = None,
        source: str | None = None,
    ) -> None:
        """记录单次工具调用指标。

        preview 字段只允许用于排障摘要，并在写入前再次脱敏。工具错误或大 payload
        不能作为事实证据，但可以通过 metrics 统计错误率、裁剪率和耗时分布。
        """

        values: JsonObject = {"tool_name": tool_name}
        _put_if_not_none(values, "raw_size", raw_size)
        _put_if_not_none(values, "preview_size", preview_size)
        _put_if_not_none(values, "trimmed", trimmed)
        _put_if_not_none(values, "evidence_usable", evidence_usable)
        _put_if_not_none(values, "input_preview", input_preview)
        _put_if_not_none(values, "output_preview", output_preview)
        _put_if_not_none(values, "source", source)
        self._record(
            MetricsRecord(
                metric_type="tool",
                ctx=ctx,
                status=status,
                latency_ms=latency_ms,
                error_code=error_code,
                values=values,
            )
        )

    def record_rag(
        self,
        ctx: RequestContext,
        *,
        status: str,
        latency_ms: float | None = None,
        candidate_count: int | None = None,
        final_count: int | None = None,
        retrieval_count: int | None = None,
        citation_count: int | None = None,
        dropped_count: int | None = None,
        empty_reason: str | None = None,
        no_answer: bool | None = None,
        error_code: str | None = None,
    ) -> None:
        """记录 RAG 检索、context 或 citation 指标。

        metrics 只记录数量和状态，不写完整 chunk 内容、绝对路径或 raw metadata；
        这与 API citation 只暴露安全字段的约束保持一致。
        """

        resolved_retrieval_count = retrieval_count if retrieval_count is not None else final_count
        values: JsonObject = {}
        _put_if_not_none(values, "candidate_count", candidate_count)
        _put_if_not_none(values, "final_count", final_count)
        _put_if_not_none(values, "retrieval_count", resolved_retrieval_count)
        _put_if_not_none(values, "citation_count", citation_count)
        _put_if_not_none(values, "dropped_count", dropped_count)
        _put_if_not_none(values, "empty_reason", empty_reason)
        _put_if_not_none(values, "no_answer", no_answer)
        self._record(
            MetricsRecord(
                metric_type="rag",
                ctx=ctx,
                status=status,
                latency_ms=latency_ms,
                error_code=error_code,
                values=values,
            )
        )

    def record_llm(
        self,
        ctx: RequestContext,
        *,
        model: str,
        status: str,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated: bool | None = None,
        error_code: str | None = None,
    ) -> None:
        """记录 LLM usage 指标。

        真实供应商 usage 和本地估算都可进入这里，必须通过 `estimated` 标记区分，
        避免把估算 token 或成本误当精确计费数据。
        """

        values: JsonObject = {"model": model}
        _put_if_not_none(values, "input_tokens", input_tokens)
        _put_if_not_none(values, "output_tokens", output_tokens)
        _put_if_not_none(values, "total_tokens", total_tokens)
        _put_if_not_none(values, "estimated", estimated)
        self._record(
            MetricsRecord(
                metric_type="llm",
                ctx=ctx,
                status=status,
                latency_ms=latency_ms,
                error_code=error_code,
                values=values,
            )
        )

    def record_fallback(
        self,
        ctx: RequestContext,
        *,
        fallback_used: bool,
        reason_code: str,
        scenario: str | None = None,
        evidence_count: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """记录 fallback 决策指标。

        这里记录的是稳定 reason_code 和证据数量，不记录 partial_answer 正文。fallback
        文案可能包含用户上下文摘要，留在 API adapter 做安全展示即可。
        """

        values: JsonObject = {
            "fallback_used": fallback_used,
            "reason_code": reason_code,
        }
        _put_if_not_none(values, "scenario", scenario)
        _put_if_not_none(values, "evidence_count", evidence_count)
        self._record(
            MetricsRecord(
                metric_type="fallback",
                ctx=ctx,
                status="ok" if fallback_used else "skipped",
                latency_ms=latency_ms,
                error_code=None if fallback_used else reason_code,
                values=values,
            )
        )

    def record_cost(
        self,
        ctx: RequestContext,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        estimated: bool,
    ) -> None:
        """记录成本估算指标。

        价格配置默认可为 0，因此 cost 指标必须保留 estimated 标志；后续趋势分析可据此
        区分真实供应商 usage 与本地估算，不把占位价格当作真实账单。
        """

        self._record(
            MetricsRecord(
                metric_type="cost",
                ctx=ctx,
                status="ok",
                values={
                    "model": model,
                    "input_tokens": max(0, input_tokens),
                    "output_tokens": max(0, output_tokens),
                    "total_tokens": max(0, total_tokens),
                    "estimated_cost": max(0.0, estimated_cost),
                    "estimated": estimated,
                },
            )
        )

    def _record(self, record: MetricsRecord) -> None:
        if not self.enabled:
            return
        try:
            self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_jsonl_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "metrics 写入失败，已降级为不中断主流程: {}",
                exc.__class__.__name__,
            )


def _put_if_not_none(values: JsonObject, key: str, value: JsonValue) -> None:
    if value is not None:
        values[key] = value


def _sanitize_value(key: str, value: JsonValue) -> JsonValue:
    if key in _SAFE_TOKEN_METRIC_KEYS:
        return value
    if _SENSITIVE_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            child_key: _sanitize_value(child_key, child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    return value
