"""ISSUE-031 AIOps API 集成测试。

这些用例通过真实 FastAPI TestClient 进入 `/api/aiops`，只把 AIOpsService
替换为内存 fake。这样测试覆盖 HTTP/SSE adapter、RequestContextMiddleware、
trace/request 字段和 fallback 外显契约，同时不会触发真实 LangGraph、MCP、
Milvus、DashScope 或网络调用。
"""

from __future__ import annotations

import json
from typing import Protocol, cast

from fastapi.testclient import TestClient


class _IntegrationAIOpsService(Protocol):
    """测试只依赖 fake 的公开状态，避免直接 import conftest 造成重复加载。"""

    calls: list[str]
    mode: str


def test_aiops_sse_success_stream_keeps_legacy_message_events_and_done_mirror(
    integration_aiops_client: TestClient,
    integration_aiops_service: _IntegrationAIOpsService,
) -> None:
    """正常诊断流必须保留 `event: message`，并把旧 `complete` 兼容映射为 `done`。"""

    with integration_aiops_client.stream(
        "POST",
        "/api/aiops",
        json={"session_id": "session-aiops"},
        headers={"X-Trace-Id": "trace-aiops-ok", "X-Request-Id": "request-aiops-ok"},
    ) as response:
        text = "".join(response.iter_text())

    events = _parse_sse_events(text)
    payloads = [event["payload"] for event in events]
    assert response.status_code == 200
    assert [event["event"] for event in events] == ["message"] * 7
    assert [payload["type"] for payload in payloads] == [
        "start",
        "status",
        "plan",
        "step_complete",
        "report",
        "complete",
        "done",
    ]
    assert payloads[0]["trace_id"] == "trace-aiops-ok"
    assert payloads[0]["request_id"] == "request-aiops-ok"
    assert payloads[-1]["trace_id"] == "trace-aiops-ok"
    assert payloads[-1]["request_id"] == "request-aiops-ok"
    assert payloads[-1]["fallback_used"] is False
    assert integration_aiops_service.calls == ["session-aiops"]


def test_aiops_sse_accepts_native_done_event_without_adding_complete(
    integration_aiops_client: TestClient,
    integration_aiops_service: _IntegrationAIOpsService,
) -> None:
    """服务层若已输出规范 `done`，API 不应反向制造旧 `complete` 事件。"""

    integration_aiops_service.mode = "done"
    with integration_aiops_client.stream(
        "POST",
        "/api/aiops",
        json={"session_id": "session-done"},
        headers={"X-Trace-Id": "trace-aiops-done", "X-Request-Id": "request-aiops-done"},
    ) as response:
        text = "".join(response.iter_text())

    payloads = [event["payload"] for event in _parse_sse_events(text)]
    assert response.status_code == 200
    assert [payload["type"] for payload in payloads] == ["start", "status", "plan", "done"]
    assert payloads[-1]["trace_id"] == "trace-aiops-done"
    assert payloads[-1]["request_id"] == "request-aiops-done"
    assert "complete" not in [payload["type"] for payload in payloads]
    assert integration_aiops_service.calls == ["session-done"]


def test_aiops_sse_midstream_exception_returns_safe_error_event(
    integration_aiops_client: TestClient,
    integration_aiops_service: _IntegrationAIOpsService,
) -> None:
    """SSE 建立后异常必须转为安全 error event，不泄漏原始异常、密钥或内部 URL。"""

    integration_aiops_service.mode = "raise"
    with integration_aiops_client.stream(
        "POST",
        "/api/aiops",
        json={"session_id": "session-error"},
        headers={"X-Trace-Id": "trace-aiops-error", "X-Request-Id": "request-aiops-error"},
    ) as response:
        text = "".join(response.iter_text())

    payloads = [event["payload"] for event in _parse_sse_events(text)]
    error_payload = payloads[-1]
    assert response.status_code == 200
    assert [payload["type"] for payload in payloads] == ["start", "status", "plan", "error"]
    assert error_payload["error"]["code"] == "SSE_STREAM_INTERRUPTED"
    assert error_payload["trace_id"] == "trace-aiops-error"
    assert error_payload["request_id"] == "request-aiops-error"
    assert "sk-secret" not in text
    assert "http://internal.aiops" not in text


def test_aiops_sse_tool_error_event_triggers_fallback_and_done(
    integration_aiops_client: TestClient,
    integration_aiops_service: _IntegrationAIOpsService,
) -> None:
    """可降级的工具错误必须外显 fallback，并在最终 done 中标记 fallback_used。"""

    integration_aiops_service.mode = "fallback"
    with integration_aiops_client.stream(
        "POST",
        "/api/aiops",
        json={"session_id": "session-fallback"},
        headers={"X-Trace-Id": "trace-aiops-fallback", "X-Request-Id": "request-aiops-fallback"},
    ) as response:
        text = "".join(response.iter_text())

    payloads = [event["payload"] for event in _parse_sse_events(text)]
    fallback_payload = payloads[-2]
    done_payload = payloads[-1]
    assert response.status_code == 200
    assert [payload["type"] for payload in payloads] == ["start", "fallback", "done"]
    assert fallback_payload["fallback_used"] is True
    assert fallback_payload["reason_code"] == "TOOL_TIMEOUT"
    assert fallback_payload["trace_id"] == "trace-aiops-fallback"
    assert fallback_payload["request_id"] == "request-aiops-fallback"
    assert done_payload["fallback_used"] is True
    assert done_payload["reason_code"] == "TOOL_TIMEOUT"
    assert done_payload["trace_id"] == "trace-aiops-fallback"
    assert done_payload["request_id"] == "request-aiops-fallback"
    assert "sk-secret" not in text
    assert "http://internal.aiops" not in text


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    """解析真实 TestClient 收到的 text/event-stream 片段。

    测试必须验证 HTTP 层最终输出，而不是直接消费内部 async generator 的 dict；因此这里
    只按 SSE 文本协议读取 `event:` 与 JSON `data:`，确保旧前端依赖的 message 事件也被锁住。
    """

    events: list[dict[str, object]] = []
    # SSE 规范允许 CR/LF/CRLF 行尾；TestClient 在 Windows 上返回 CRLF，
    # 先归一化为 LF 再按空行切分，否则所有事件会挤进同一个块导致 JSON 解析失败。
    normalized = text.strip().replace("\r\n", "\n")
    for raw_event in normalized.split("\n\n"):
        if not raw_event:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in raw_event.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if data_lines:
            events.append(
                {
                    "event": event_name,
                    "payload": cast(dict[str, object], json.loads("\n".join(data_lines))),
                }
            )
    return events
