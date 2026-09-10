"""RequestContextMiddleware 的 ISSUE-002 回归测试。

这些用例只验证请求级 trace/request 注入和最小响应兼容，不触发真实
Milvus、DashScope 或 MCP server。middleware 必须避免读取 request body，
否则后续上传接口会在 ISSUE-004 前就被破坏。
"""

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.request_context import RequestContextMiddleware, get_request_context
from app.observability.tracing import TraceLogger


def _build_test_app(trace_path: Path, *, trace_enabled: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RequestContextMiddleware,
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=trace_enabled),
        request_timeout_ms=1500,
    )

    @app.get("/ctx")
    async def read_context(request: Request) -> dict[str, str | None]:
        ctx = get_request_context()
        request.state.ctx = ctx.with_session("session-1")
        return {
            "ctx_trace_id": ctx.trace_id,
            "ctx_request_id": ctx.request_id,
            "tenant_id": ctx.tenant_id,
            "user_id": ctx.user_id,
            "session_id": request.state.ctx.session_id,
        }

    @app.post("/echo-body")
    async def echo_body(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"size": len(body)}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("token=sk-secret failed at http://internal.service")

    return app


def _read_trace_events(trace_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def test_middleware_propagates_valid_trace_headers_to_state_headers_body_and_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    client = TestClient(_build_test_app(trace_path))

    response = client.get(
        "/ctx",
        headers={
            "X-Trace-Id": "client-trace-1",
            "X-Request-Id": "client-request-1",
            "X-Tenant-Id": "tenant-a",
            "X-User-Id": "user-a",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "client-trace-1"
    assert response.headers["X-Request-Id"] == "client-request-1"
    assert body["trace_id"] == "client-trace-1"
    assert body["request_id"] == "client-request-1"
    assert body["ctx_trace_id"] == "client-trace-1"
    assert body["ctx_request_id"] == "client-request-1"
    assert body["tenant_id"] == "tenant-a"
    assert body["user_id"] == "user-a"
    assert body["session_id"] == "session-1"

    events = _read_trace_events(trace_path)
    assert [event["name"] for event in events] == ["request.start", "request.end"]
    assert {event["trace_id"] for event in events} == {"client-trace-1"}
    assert events[-1]["status_code"] == 200
    assert events[-1]["session_id"] == "session-1"


def test_middleware_regenerates_invalid_inbound_trace_headers(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    client = TestClient(_build_test_app(trace_path))

    response = client.get(
        "/ctx",
        headers={
            "X-Trace-Id": "bad trace with spaces",
            "X-Request-Id": "x" * 200,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["trace_id"].startswith("trc_")
    assert body["request_id"].startswith("req_")
    assert body["trace_id"] != "bad trace with spaces"
    assert body["request_id"] != "x" * 200

    start_event = _read_trace_events(trace_path)[0]
    assert start_event["invalid_inbound_trace_header"] is True


def test_middleware_uses_default_user_and_tenant_when_headers_are_missing(tmp_path: Path) -> None:
    response = TestClient(_build_test_app(tmp_path / "trace.jsonl")).get("/ctx")

    body = response.json()
    assert body["tenant_id"] == "default"
    assert body["user_id"] == "anonymous"


def test_middleware_does_not_consume_request_body(tmp_path: Path) -> None:
    response = TestClient(_build_test_app(tmp_path / "trace.jsonl")).post(
        "/echo-body",
        content=b"upload-bytes",
    )

    assert response.status_code == 200
    assert response.json()["size"] == len(b"upload-bytes")


def test_middleware_returns_safe_error_response_with_context_trace(tmp_path: Path) -> None:
    response = TestClient(_build_test_app(tmp_path / "trace.jsonl")).get(
        "/boom",
        headers={"X-Trace-Id": "client-trace-err", "X-Request-Id": "client-request-err"},
    )

    body = response.json()
    assert response.status_code == 500
    assert response.headers["X-Trace-Id"] == "client-trace-err"
    assert body["trace_id"] == "client-trace-err"
    assert body["request_id"] == "client-request-err"
    assert body["error"]["trace_id"] == "client-trace-err"
    assert body["error"]["request_id"] == "client-request-err"
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "sk-secret" not in response.text
    assert "http://internal.service" not in response.text
