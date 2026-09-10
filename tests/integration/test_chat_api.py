"""ISSUE-030 Chat API 集成测试。

这些用例通过真实 FastAPI TestClient 进入 middleware 和 router，验证 `/api/chat`、
`/api/chat_stream`、`/api/chat/clear`、`/api/chat/session/{session_id}` 的对外
schema。RAG service 使用 conftest 中的内存 fake，避免测试访问真实 LLM、Milvus、
MCP 或网络。
"""

from __future__ import annotations

import json
from typing import Protocol, cast

from fastapi.testclient import TestClient


class _IntegrationRagService(Protocol):
    """测试只依赖 fake 的公开记录字段，避免直接 import conftest 造成重复加载。"""

    query_calls: list[tuple[str, str]]
    stream_calls: list[tuple[str, str]]
    clear_calls: list[str]
    history_calls: list[str]
    history: list[dict[str, str]]
    stream_mode: str


def test_chat_accepts_legacy_and_new_field_aliases(
    integration_client: TestClient,
    integration_rag_service: _IntegrationRagService,
) -> None:
    """`Id/Question` 和 `id/question` 都必须可用，并保留旧响应字段。"""

    legacy_response = integration_client.post(
        "/api/chat",
        json={"Id": "session-legacy", "Question": "CPU 怎么排查？"},
        headers={"X-Trace-Id": "trace-chat-legacy", "X-Request-Id": "request-chat-legacy"},
    )
    new_response = integration_client.post(
        "/api/chat",
        json={"id": "session-new", "question": "内存怎么排查？"},
        headers={"X-Trace-Id": "trace-chat-new", "X-Request-Id": "request-chat-new"},
    )

    assert legacy_response.status_code == 200
    assert new_response.status_code == 200
    legacy_body = legacy_response.json()
    new_body = new_response.json()
    for body in (legacy_body, new_body):
        assert body["code"] == 200
        assert body["message"] == "success"
        assert body["data"]["success"] is True
        assert body["data"]["answer"]
        assert body["data"]["errorMessage"] is None
        assert body["trace_id"]
        assert body["request_id"]
    assert legacy_body["trace_id"] == "trace-chat-legacy"
    assert new_body["request_id"] == "request-chat-new"
    assert integration_rag_service.query_calls == [
        ("CPU 怎么排查？", "session-legacy"),
        ("内存怎么排查？", "session-new"),
    ]


def test_chat_invalid_input_returns_error_envelope(integration_client: TestClient) -> None:
    """输入校验错误必须返回 4xx、统一 error，并保留旧 data.errorMessage。"""

    response = integration_client.post(
        "/api/chat",
        json={"Id": "session-1", "Question": "   "},
        headers={"X-Trace-Id": "trace-chat-invalid", "X-Request-Id": "request-chat-invalid"},
    )

    body = response.json()
    assert response.status_code == 400
    assert body["success"] is False
    assert body["code"] == 400
    assert body["data"]["success"] is False
    assert body["data"]["answer"] is None
    assert body["data"]["errorMessage"]
    assert body["error"]["code"] == "INVALID_INPUT"
    assert body["error"]["trace_id"] == "trace-chat-invalid"
    assert body["request_id"] == "request-chat-invalid"


def test_chat_stream_success_keeps_message_event_and_typed_payloads(
    integration_client: TestClient,
) -> None:
    """SSE 成功流必须有 start/content/done，且每个 data 都保留 type。"""

    with integration_client.stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-stream", "Question": "CPU 怎么排查？"},
        headers={"X-Trace-Id": "trace-stream-ok", "X-Request-Id": "request-stream-ok"},
    ) as response:
        text = "".join(response.iter_text())

    events = _parse_sse_events(text)
    payloads = [event["payload"] for event in events]
    assert response.status_code == 200
    assert [event["event"] for event in events] == ["message", "message", "message", "message"]
    assert [payload["type"] for payload in payloads] == ["start", "content", "content", "done"]
    assert payloads[0]["trace_id"] == "trace-stream-ok"
    assert payloads[0]["request_id"] == "request-stream-ok"
    assert payloads[-1]["trace_id"] == "trace-stream-ok"
    assert payloads[-1]["request_id"] == "request-stream-ok"


def test_chat_stream_midstream_error_keeps_http_200_and_safe_error_event(
    integration_client: TestClient,
    integration_rag_service: _IntegrationRagService,
) -> None:
    """SSE 已建立后的错误只能进入流内 error event，且不能泄漏原始异常。"""

    integration_rag_service.stream_mode = "error"
    with integration_client.stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-stream", "Question": "CPU 怎么排查？"},
        headers={"X-Trace-Id": "trace-stream-error", "X-Request-Id": "request-stream-error"},
    ) as response:
        text = "".join(response.iter_text())

    payloads = [event["payload"] for event in _parse_sse_events(text)]
    error_payload = payloads[-1]
    serialized = json.dumps(error_payload, ensure_ascii=False)
    assert response.status_code == 200
    assert [payload["type"] for payload in payloads] == ["start", "content", "error"]
    assert error_payload["error"]["code"] == "SSE_STREAM_INTERRUPTED"
    assert error_payload["trace_id"] == "trace-stream-error"
    assert error_payload["request_id"] == "request-stream-error"
    assert "sk-secret" not in serialized
    assert "http://internal.stream" not in serialized


def test_clear_session_accepts_session_id_aliases_and_keeps_legacy_fields(
    integration_client: TestClient,
    integration_rag_service: _IntegrationRagService,
) -> None:
    """清理接口必须同时支持 `sessionId` 和 `session_id`，并保留 status/message/data。"""

    legacy_response = integration_client.post(
        "/api/chat/clear",
        json={"sessionId": "session-clear-legacy"},
    )
    new_response = integration_client.post(
        "/api/chat/clear",
        json={"session_id": "session-clear-new"},
    )

    for response in (legacy_response, new_response):
        body = response.json()
        assert response.status_code == 200
        assert body["success"] is True
        assert body["status"] == "success"
        assert body["message"] == "会话已清空"
        assert body["data"] is None
        assert body["trace_id"]
        assert body["request_id"]
    assert integration_rag_service.clear_calls == [
        "session-clear-legacy",
        "session-clear-new",
    ]


def test_session_history_keeps_old_top_level_fields(
    integration_client: TestClient,
    integration_rag_service: _IntegrationRagService,
) -> None:
    """历史查询必须保留旧顶层 session_id/message_count/history 字段。"""

    response = integration_client.get(
        "/api/chat/session/session-history",
        headers={"X-Trace-Id": "trace-session", "X-Request-Id": "request-session"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["session_id"] == "session-history"
    assert body["message_count"] == len(body["history"])
    assert body["history"] == integration_rag_service.history
    assert body["trace_id"] == "trace-session"
    assert body["request_id"] == "request-session"
    assert integration_rag_service.history_calls == ["session-history"]


def test_chat_records_backend_history_and_lists_sessions(
    integration_client: TestClient,
    monkeypatch,
) -> None:
    """Successful chat turns should be visible through the backend session list."""

    from app.api import chat as chat_api
    from tests.conftest import FakeHistoryStore

    # integration_client 已注入内存 FakeHistoryStore（与 PG store 同形）；
    # 这里换一个干净实例，保证断言只针对当前请求写入的数据。
    store = FakeHistoryStore()
    monkeypatch.setattr(chat_api, "conversation_history_store", store)

    chat_response = integration_client.post(
        "/api/chat",
        json={"Id": "session-db-list", "Question": "Show me backend history"},
    )
    sessions_response = integration_client.get("/api/chat/sessions")
    history_response = integration_client.get("/api/chat/session/session-db-list")

    assert chat_response.status_code == 200
    assert sessions_response.status_code == 200
    sessions_body = sessions_response.json()
    assert sessions_body["data"]["count"] == 1
    assert sessions_body["data"]["sessions"][0]["session_id"] == "session-db-list"
    assert sessions_body["data"]["sessions"][0]["message_count"] == 2

    history_body = history_response.json()
    assert history_body["history"][0]["role"] == "user"
    assert history_body["history"][0]["content"] == "Show me backend history"
    assert history_body["history"][1]["role"] == "assistant"
    assert history_body["message_count"] == 2


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    """解析 TestClient 收到的 SSE 文本，断言 event 名和 JSON data。

    测试关注的是 HTTP 层真实输出，而不是直接调用 handler 返回的 dict；因此这里解析
    标准 text/event-stream 片段，避免误把内部生成器对象当成已验证的 API 响应。
    """

    events: list[dict[str, object]] = []
    normalized_text = text.replace("\r\n", "\n")
    for raw_event in normalized_text.strip().split("\n\n"):
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
