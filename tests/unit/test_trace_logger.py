"""TraceLogger 的 ISSUE-002 单元测试。"""

import json
from pathlib import Path

from app.core.errors import InvalidInputError
from app.core.request_context import RequestContext
from app.observability.metrics import MetricsRecorder
from app.observability.tracing import TraceLogger


def _ctx() -> RequestContext:
    return RequestContext(
        trace_id="trc_unit",
        request_id="req_unit",
        session_id="session-1",
        tenant_id="default",
        user_id="anonymous",
        deadline_ms=1000,
        feature_flags=("trace-test",),
        started_at=1.0,
        started_monotonic=1.0,
        method="GET",
        path="/unit",
        invalid_inbound_trace_header=False,
    )


def _read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_metrics(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_event_writes_jsonl_with_context_fields_and_redacts_sensitive_values(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)

    logger.record_event(
        "unit.event",
        _ctx(),
        api_key="sk-secret",
        nested={"token": "sk-nested-secret", "safe": "ok"},
    )

    [event] = _read_events(trace_path)
    assert event["name"] == "unit.event"
    assert event["trace_id"] == "trc_unit"
    assert event["request_id"] == "req_unit"
    assert event["session_id"] == "session-1"
    assert event["api_key"] == "<redacted>"
    assert event["nested"] == {"token": "<redacted>", "safe": "ok"}


def test_start_and_end_span_write_correlated_events(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)

    span = logger.start_span("unit.span", _ctx(), component="unit")
    logger.end_span(span, status="ok", output_size=3)

    events = _read_events(trace_path)
    assert [event["name"] for event in events] == ["unit.span.start", "unit.span.end"]
    assert events[0]["span_id"] == events[1]["span_id"]
    assert events[1]["status"] == "ok"
    assert events[1]["output_size"] == 3
    assert isinstance(events[1]["latency_ms"], int | float)


def test_record_error_writes_safe_error_metadata(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)
    error = InvalidInputError(user_message="问题不能为空。", internal_message="raw secret")

    logger.record_error(error, _ctx())

    [event] = _read_events(trace_path)
    assert event["name"] == "request.error"
    assert event["status"] == "error"
    assert event["error_code"] == "INVALID_INPUT"
    assert event["http_status"] == 400
    assert event["retryable"] is False
    assert event["fallback_required"] is False
    assert "raw secret" not in json.dumps(event, ensure_ascii=False)


def test_trace_write_failure_does_not_escape(tmp_path: Path) -> None:
    logger = TraceLogger(trace_jsonl_path=str(tmp_path), enabled=True)

    logger.record_event("unit.event", _ctx())


def test_metrics_recorder_writes_request_tool_rag_llm_fallback_and_cost_records(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(metrics_jsonl_path=str(metrics_path), enabled=True)

    recorder.record_request(
        _ctx(),
        status="ok",
        latency_ms=12.5,
        http_status=200,
        fallback_used=False,
    )
    recorder.record_tool(
        _ctx(),
        tool_name="retrieve_knowledge",
        status="success",
        latency_ms=6.0,
        raw_size=120,
        preview_size=30,
        trimmed=False,
        evidence_usable=True,
        input_preview={"api_key": "sk-secret"},
    )
    recorder.record_rag(
        _ctx(),
        status="ok",
        latency_ms=4.0,
        candidate_count=8,
        final_count=3,
        citation_count=2,
        empty_reason=None,
    )
    recorder.record_llm(
        _ctx(),
        model="qwen-max",
        status="ok",
        latency_ms=50.0,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    recorder.record_fallback(
        _ctx(),
        fallback_used=True,
        reason_code="LLM_TIMEOUT",
        scenario="chat",
        evidence_count=1,
    )
    recorder.record_cost(
        _ctx(),
        model="qwen-max",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost=0.0123,
        estimated=True,
    )

    records = _read_metrics(metrics_path)
    assert [record["metric_type"] for record in records] == [
        "request",
        "tool",
        "rag",
        "llm",
        "fallback",
        "cost",
    ]
    assert all(record["trace_id"] == "trc_unit" for record in records)
    assert records[1]["input_preview"] == {"api_key": "<redacted>"}
    assert records[2]["retrieval_count"] == 3
    assert records[3]["total_tokens"] == 15
    assert records[4]["fallback_used"] is True
    assert records[5]["estimated_cost"] == 0.0123


def test_metrics_write_failure_does_not_escape(tmp_path: Path) -> None:
    recorder = MetricsRecorder(metrics_jsonl_path=str(tmp_path), enabled=True)

    recorder.record_request(_ctx(), status="ok", latency_ms=1.0)


def test_trace_logger_mirrors_existing_events_to_metrics(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_recorder = MetricsRecorder(metrics_jsonl_path=str(metrics_path), enabled=True)
    logger = TraceLogger(
        trace_jsonl_path=str(trace_path),
        enabled=True,
        metrics_recorder=metrics_recorder,
    )

    logger.record_event("request.end", _ctx(), status="ok", status_code=200, latency_ms=7.0)
    span = logger.start_span("tool", _ctx(), tool_name="retrieve_knowledge")
    logger.end_span(
        span,
        status="ok",
        tool_name="retrieve_knowledge",
        **{
            "tool_result.status": "success",
            "raw_size": 20,
            "preview_size": 10,
            "evidence_usable": True,
        },
    )
    logger.record_event(
        "rag.retrieve.end",
        _ctx(),
        status="ok",
        latency_ms=8.0,
        candidate_count=5,
        final_count=2,
    )
    logger.record_event(
        "fallback.decide",
        _ctx(),
        fallback_used=True,
        reason_code="RAG_EMPTY_RESULT",
        evidence_count=0,
        scenario="chat",
    )
    logger.record_event(
        "token.usage",
        _ctx(),
        usage={
            "model": "qwen-max",
            "input_tokens": 11,
            "output_tokens": 13,
            "total_tokens": 24,
            "estimated": True,
            "estimated_cost": 0.0,
        },
    )

    records = _read_metrics(metrics_path)
    assert [record["metric_type"] for record in records] == [
        "request",
        "tool",
        "rag",
        "fallback",
        "llm",
        "cost",
    ]
    assert records[0]["http_status"] == 200
    assert records[1]["tool_name"] == "retrieve_knowledge"
    assert records[2]["retrieval_count"] == 2
    assert records[3]["reason_code"] == "RAG_EMPTY_RESULT"
    assert records[4]["total_tokens"] == 24
