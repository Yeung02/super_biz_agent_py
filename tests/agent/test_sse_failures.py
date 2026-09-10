"""ISSUE-009 SSE 异常和断开处理测试。

这些测试直接调用 API handler，并用 fake service 生成流事件。这样既能验证
`event: message + data.type` 的旧前端兼容，又能覆盖 `start/error/done` 的标准
payload，不需要真实浏览器、Milvus、DashScope 或 MCP server。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.request_context import RequestContext


class _FakeSseRequest:
    """SSE handler 测试用请求对象，支持模拟客户端断开。"""

    def __init__(
        self,
        ctx: RequestContext,
        *,
        disconnect_after_checks: int | None = None,
    ) -> None:
        self.state = SimpleNamespace(ctx=ctx)
        self.disconnect_after_checks = disconnect_after_checks
        self.disconnect_checks = 0

    async def is_disconnected(self) -> bool:
        self.disconnect_checks += 1
        if self.disconnect_after_checks is None:
            return False
        return self.disconnect_checks > self.disconnect_after_checks


async def _collect_sse_events(response: Any) -> list[dict[str, Any]]:
    """收集 EventSourceResponse 的测试事件。

    sse-starlette 在未进入 ASGI send 流程时会直接暴露 dict；测试环境缺少
    sse-starlette 时 conftest 的 stub 会暴露文本块。这里同时支持两种形态，避免测试
    把断言耦合到某个第三方库实现细节。
    """

    events: list[dict[str, Any]] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, dict):
            events.append(
                {
                    "event": chunk.get("event"),
                    "payload": json.loads(str(chunk.get("data", "{}"))),
                }
            )
            continue

        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        for raw_event in text.strip().split("\n\n"):
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
                events.append({"event": event_name, "payload": json.loads("\n".join(data_lines))})
    return events


@pytest.mark.asyncio
async def test_chat_stream_emits_start_content_and_done_with_trace(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """Chat SSE 首条必须是 start，结束必须是 done，且不破坏旧 message 事件。"""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    async def _fake_query_stream(question: str, session_id: str):
        _ = question, session_id
        yield {"type": "content", "data": "hello"}
        yield {"type": "complete", "data": {"answer": "hello"}}

    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _fake_query_stream)

    response = await chat_api.chat_stream(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeSseRequest(fake_request_context),
    )
    events = await _collect_sse_events(response)

    assert [event["event"] for event in events] == ["message", "message", "message"]
    assert [event["payload"]["type"] for event in events] == ["start", "content", "done"]
    assert events[0]["payload"]["trace_id"] == "trc_test"
    assert events[0]["payload"]["request_id"] == "req_test"
    assert events[-1]["payload"]["trace_id"] == "trc_test"
    assert events[-1]["payload"]["request_id"] == "req_test"


@pytest.mark.asyncio
async def test_chat_stream_maps_midstream_exception_to_safe_error_event(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """Chat SSE 中途异常只能暴露安全错误码和文案，不能返回原始异常全文。"""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    async def _failing_query_stream(question: str, session_id: str):
        _ = question, session_id
        yield {"type": "content", "data": "partial"}
        raise RuntimeError("secret token leaked from http://internal.service")

    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _failing_query_stream)

    response = await chat_api.chat_stream(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeSseRequest(fake_request_context),
    )
    events = await _collect_sse_events(response)
    error_payload = events[-1]["payload"]
    serialized = json.dumps(error_payload, ensure_ascii=False)

    assert [event["payload"]["type"] for event in events] == ["start", "content", "error"]
    assert error_payload["error"]["code"] == "SSE_STREAM_INTERRUPTED"
    assert error_payload["trace_id"] == "trc_test"
    assert error_payload["request_id"] == "req_test"
    assert "secret token" not in serialized
    assert "http://internal.service" not in serialized


@pytest.mark.asyncio
async def test_chat_stream_stops_before_downstream_when_client_already_disconnected(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """客户端已断开时不能继续拉取下游 Agent 流，避免留下后台任务。"""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    calls = 0

    async def _fake_query_stream(question: str, session_id: str):
        nonlocal calls
        _ = question, session_id
        calls += 1
        yield {"type": "content", "data": "should not send"}

    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _fake_query_stream)

    response = await chat_api.chat_stream(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeSseRequest(fake_request_context, disconnect_after_checks=0),
    )
    events = await _collect_sse_events(response)

    assert events == []
    assert calls == 0


@pytest.mark.asyncio
async def test_chat_stream_stops_downstream_when_client_disconnects_after_start(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """After emitting start, the handler must check disconnect before pulling downstream."""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    calls = 0

    async def _fake_query_stream(question: str, session_id: str):
        nonlocal calls
        _ = question, session_id
        calls += 1
        yield {"type": "content", "data": "should not send"}

    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _fake_query_stream)

    response = await chat_api.chat_stream(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeSseRequest(fake_request_context, disconnect_after_checks=1),
    )
    events = await _collect_sse_events(response)

    assert [event["payload"]["type"] for event in events] == ["start"]
    assert calls == 0


@pytest.mark.asyncio
async def test_chat_stream_request_timeout_emits_safe_error_event(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """Chat SSE must convert overall request timeout into a safe error event."""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    async def _slow_query_stream(question: str, session_id: str):
        _ = question, session_id
        await asyncio.sleep(0.03)
        yield {"type": "content", "data": "late"}

    short_ctx = replace(
        fake_request_context,
        deadline_ms=1,
        started_monotonic=time.monotonic(),
    )
    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _slow_query_stream)

    response = await chat_api.chat_stream(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeSseRequest(short_ctx),
    )
    events = await _collect_sse_events(response)
    error_payload = events[-1]["payload"]

    assert [event["payload"]["type"] for event in events] == ["start", "error"]
    assert error_payload["error"]["code"] == "LLM_TIMEOUT"
    assert "late" not in json.dumps(error_payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_aiops_stream_emits_start_and_keeps_complete_done_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """AIOps SSE 要先发 start，并同时保留旧 complete 与新 done。"""

    from app.api import aiops as aiops_api
    from app.models.aiops import AIOpsRequest

    async def _fake_diagnose(session_id: str):
        _ = session_id
        yield {"type": "plan", "stage": "plan_created", "message": "计划完成", "plan": ["检查"]}
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": "诊断流程完成",
            "diagnosis": {"status": "completed"},
        }

    monkeypatch.setattr(aiops_api.aiops_service, "diagnose", _fake_diagnose)

    response = await aiops_api.diagnose_stream(
        AIOpsRequest(session_id="session-1"),
        _FakeSseRequest(fake_request_context),
    )
    events = await _collect_sse_events(response)

    assert [event["event"] for event in events] == ["message", "message", "message", "message"]
    assert [event["payload"]["type"] for event in events] == [
        "start",
        "plan",
        "complete",
        "done",
    ]
    assert events[0]["payload"]["trace_id"] == "trc_test"
    assert events[-1]["payload"]["request_id"] == "req_test"


@pytest.mark.asyncio
async def test_aiops_stream_stops_downstream_when_client_disconnects_after_start(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """AIOps SSE must stop before pulling diagnose events after a post-start disconnect."""

    from app.api import aiops as aiops_api
    from app.models.aiops import AIOpsRequest

    calls = 0

    async def _fake_diagnose(session_id: str):
        nonlocal calls
        _ = session_id
        calls += 1
        yield {"type": "plan", "plan": ["should not send"]}

    monkeypatch.setattr(aiops_api.aiops_service, "diagnose", _fake_diagnose)

    response = await aiops_api.diagnose_stream(
        AIOpsRequest(session_id="session-1"),
        _FakeSseRequest(fake_request_context, disconnect_after_checks=1),
    )
    events = await _collect_sse_events(response)

    assert [event["payload"]["type"] for event in events] == ["start"]
    assert calls == 0


@pytest.mark.asyncio
async def test_aiops_stream_maps_midstream_exception_to_safe_error_event(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """AIOps SSE 中途异常必须返回安全 error payload，并保留 trace/request。"""

    from app.api import aiops as aiops_api
    from app.models.aiops import AIOpsRequest

    async def _failing_diagnose(session_id: str):
        _ = session_id
        raise RuntimeError("raw password=abc from http://internal.aiops")
        yield {}

    monkeypatch.setattr(aiops_api.aiops_service, "diagnose", _failing_diagnose)

    response = await aiops_api.diagnose_stream(
        AIOpsRequest(session_id="session-1"),
        _FakeSseRequest(fake_request_context),
    )
    events = await _collect_sse_events(response)
    error_payload = events[-1]["payload"]
    serialized = json.dumps(error_payload, ensure_ascii=False)

    assert [event["payload"]["type"] for event in events] == ["start", "error"]
    assert error_payload["error"]["code"] == "SSE_STREAM_INTERRUPTED"
    assert error_payload["trace_id"] == "trc_test"
    assert error_payload["request_id"] == "req_test"
    assert "password=abc" not in serialized
    assert "http://internal.aiops" not in serialized
