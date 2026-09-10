"""ISSUE-011 FallbackManager 与 fallback 策略矩阵测试。

这些测试只覆盖当前 issue 的降级决策和最小 API/SSE 接入：失败场景能稳定映射为
`reason_code`、用户可见安全文案和 `fallback_used`，输入/文件/健康检查等不可降级
场景继续走结构化错误。所有 service 都用 fake 或 monkeypatch，不访问真实外部服务。
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.errors import (
    InvalidInputError,
    LLMTimeoutError,
    RagEmptyResultError,
    SSEStreamInterruptedError,
    ToolTimeoutError,
    VectorStoreUnavailableError,
)
from app.core.request_context import RequestContext
from app.observability.tracing import TraceLogger


def _fallback_module() -> Any:
    """按需导入 fallback 模块，让 RED 阶段表现为断言失败而不是收集错误。"""

    try:
        return importlib.import_module("app.core.fallback")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.core.fallback must exist for ISSUE-011: {exc}")


def _load_api_module(monkeypatch: pytest.MonkeyPatch, module_name: str) -> ModuleType:
    """导入 API router 前替换重依赖 service，避免单测导入 LangChain/DashScope。"""

    rag_module = types.ModuleType("app.services.rag_agent_service")
    rag_module.rag_agent_service = SimpleNamespace(query=None, query_stream=None)
    aiops_module = types.ModuleType("app.services.aiops_service")
    aiops_module.aiops_service = SimpleNamespace(diagnose=None)

    monkeypatch.setitem(sys.modules, "app.services.rag_agent_service", rag_module)
    monkeypatch.setitem(sys.modules, "app.services.aiops_service", aiops_module)
    sys.modules.pop(f"app.api.{module_name}", None)
    api_package = sys.modules.get("app.api")
    if api_package is not None and hasattr(api_package, module_name):
        delattr(api_package, module_name)
    return importlib.import_module(f"app.api.{module_name}")


@dataclass(frozen=True)
class _FakeToolResult:
    """FallbackManager 测试用 ToolResult 同形对象，避免导入真实 LangChain 工具模块。"""

    status: str
    is_error: bool

    def is_evidence_usable(self) -> bool:
        return not self.is_error and self.status == "success"


class _FakeRequest:
    """API handler 测试用 request fake，只提供 state.ctx 和 SSE 断开检测。"""

    def __init__(self, ctx: RequestContext) -> None:
        self.state = SimpleNamespace(ctx=ctx)

    async def is_disconnected(self) -> bool:
        return False


async def _collect_sse_events(response: Any) -> list[dict[str, Any]]:
    """收集 EventSourceResponse 或测试 stub 的 SSE 事件。"""

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


def test_input_validation_error_does_not_trigger_fallback(
    fake_request_context: RequestContext,
) -> None:
    """输入错误不能被 fallback 隐藏，否则旧前端会把用户问题错误误判成成功回答。"""

    fallback = _fallback_module()
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    result = manager.for_chat(InvalidInputError(), fake_request_context)

    assert result.fallback_used is False
    assert result.reason_code == "INVALID_INPUT"
    assert result.safe_message == "请求参数不合法。"
    assert result.should_continue_stream is False
    assert manager.to_response_fields(result) == {"fallback_used": False, "reason_code": "INVALID_INPUT"}


def test_llm_timeout_returns_safe_partial_chat_fallback(
    fake_request_context: RequestContext,
) -> None:
    """LLM 超时可降级，但回答只能来自安全文案和已验证摘要，不能泄漏内部异常。"""

    fallback = _fallback_module()
    evidence = fallback.FallbackEvidence(
        summary="已检索到 CPU Runbook。api_key=sk-secret http://internal.local",
        evidence_count=1,
    )
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    result = manager.for_chat(LLMTimeoutError(), fake_request_context, evidence=evidence)
    fields = manager.to_response_fields(result)
    serialized = json.dumps(fields, ensure_ascii=False)

    assert result.fallback_used is True
    assert result.reason_code == "LLM_TIMEOUT"
    assert result.partial_answer is not None
    assert fields["fallback_used"] is True
    assert fields["reason_code"] == "LLM_TIMEOUT"
    assert fields["answer"] == result.partial_answer
    assert fields["errorMessage"] is None
    assert "sk-secret" not in serialized
    assert "http://internal.local" not in serialized


def test_rag_empty_result_uses_no_answer_without_citations(
    fake_request_context: RequestContext,
) -> None:
    """RAG 为空应降级为 no-answer，不能伪造 citation 或确定性结论。"""

    fallback = _fallback_module()
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    result = manager.for_chat(RagEmptyResultError(), fake_request_context)
    fields = manager.to_response_fields(result)

    assert result.fallback_used is True
    assert result.reason_code == "RAG_EMPTY_RESULT"
    assert "没有找到相关依据" in result.safe_message
    assert fields["citations"] == []
    assert fields["answer"] == result.safe_message


def test_tool_timeout_error_result_is_not_used_as_fact(
    fake_request_context: RequestContext,
) -> None:
    """工具错误结果只参与降级决策，不能被包装成事实证据或 partial_answer。"""

    fallback = _fallback_module()
    tool_result = _FakeToolResult(status="timeout", is_error=True)
    evidence = fallback.FallbackEvidence(tool_results=(tool_result,))
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    result = manager.for_chat(ToolTimeoutError(tool_name="query_cpu_metrics"), fake_request_context, evidence=evidence)
    fields = manager.to_response_fields(result)
    serialized = json.dumps(fields, ensure_ascii=False)

    assert result.fallback_used is True
    assert result.reason_code == "TOOL_TIMEOUT"
    assert result.partial_answer is None
    assert "实时工具不可用" in result.safe_message
    assert "password=abc" not in serialized
    assert "http://internal.tool" not in serialized


def test_file_and_health_scenarios_do_not_use_llm_fallback(
    fake_request_context: RequestContext,
) -> None:
    """文件/健康检查接口不能返回 LLM fallback，必须保留结构化错误语义。"""

    fallback = _fallback_module()
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    file_result = manager.decide(VectorStoreUnavailableError(), fake_request_context, scenario="file")
    health_result = manager.decide(VectorStoreUnavailableError(), fake_request_context, scenario="health")

    assert file_result.fallback_used is False
    assert file_result.reason_code == "VECTOR_STORE_UNAVAILABLE"
    assert health_result.fallback_used is False
    assert health_result.reason_code == "VECTOR_STORE_UNAVAILABLE"


def test_sse_fallback_event_and_done_payload_keep_type_and_trace(
    fake_request_context: RequestContext,
) -> None:
    """SSE fallback 必须保留旧前端依赖的 `event: message` 和 `data.type`。"""

    fallback = _fallback_module()
    manager = fallback.FallbackManager(trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False))

    result = manager.decide(
        SSEStreamInterruptedError(),
        fake_request_context,
        scenario="chat_stream",
        evidence=fallback.FallbackEvidence(summary="已经输出部分内容", evidence_count=1),
    )
    fallback_event = manager.to_sse_event(result, fake_request_context)
    done_event = manager.to_sse_done_event(result, fake_request_context, session_id="session-test")

    fallback_payload = json.loads(fallback_event["data"])
    done_payload = json.loads(done_event["data"])

    assert fallback_event["event"] == "message"
    assert fallback_payload["type"] == "fallback"
    assert fallback_payload["fallback_used"] is True
    assert fallback_payload["reason_code"] == "SSE_STREAM_INTERRUPTED"
    assert fallback_payload["trace_id"] == "trc_test"
    assert fallback_payload["request_id"] == "req_test"
    assert done_payload["type"] == "done"
    assert done_payload["fallback_used"] is True
    assert done_payload["session_id"] == "session-test"


def test_fallback_decision_records_trace_event(
    tmp_path,
    fake_request_context: RequestContext,
) -> None:
    """FallbackManager 每次决策都要写 trace，便于排查降级原因和证据数量。"""

    fallback = _fallback_module()
    trace_path = tmp_path / "trace.jsonl"
    manager = fallback.FallbackManager(
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)
    )

    manager.for_chat(
        LLMTimeoutError(),
        fake_request_context,
        evidence=fallback.FallbackEvidence(summary="partial", evidence_count=1),
    )

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event = events[-1]

    assert event["name"] == "fallback.decide"
    assert event["reason_code"] == "LLM_TIMEOUT"
    assert event["fallback_used"] is True
    assert event["evidence_count"] == 1
    assert event["partial_answer"] is True


def test_disabled_fallback_manager_keeps_structured_error_path(
    fake_request_context: RequestContext,
) -> None:
    """回滚开关关闭时，不可把可降级错误转换成成功响应。"""

    fallback = _fallback_module()
    manager = fallback.FallbackManager(
        enabled=False,
        trace_logger=TraceLogger(trace_jsonl_path="unused", enabled=False),
    )

    result = manager.for_chat(LLMTimeoutError(), fake_request_context)

    assert result.fallback_used is False
    assert result.reason_code == "LLM_TIMEOUT"


def test_chat_api_non_stream_uses_fallback_response_on_llm_timeout(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """非流式 Chat 的 fallback 对外必须是 HTTP 200 兼容响应字段。"""

    chat_api = _load_api_module(monkeypatch, "chat")
    from app.models.request import ChatRequest

    async def _timeout_query(question: str, session_id: str) -> str:
        _ = question, session_id
        raise LLMTimeoutError()

    monkeypatch.setattr(chat_api.rag_agent_service, "query", _timeout_query)

    response = asyncio.run(
        chat_api.chat(
            ChatRequest(Id="session-test", Question="CPU 怎么排查？"),
            _FakeRequest(fake_request_context),
        )
    )

    assert isinstance(response, dict)
    assert response["code"] == 200
    assert response["message"] == "success"
    assert response["data"]["success"] is True
    assert response["data"]["fallback_used"] is True
    assert response["data"]["reason_code"] == "LLM_TIMEOUT"
    assert response["data"]["errorMessage"] is None


def test_chat_stream_emits_fallback_then_done_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """Chat SSE 可降级错误应新增 fallback 事件，并用 done 暴露 fallback_used。"""

    chat_api = _load_api_module(monkeypatch, "chat")
    from app.models.request import ChatRequest

    async def _timeout_stream(question: str, session_id: str) -> AsyncGenerator[dict[str, object], None]:
        _ = question, session_id
        raise LLMTimeoutError()
        yield {}

    monkeypatch.setattr(chat_api.rag_agent_service, "query_stream", _timeout_stream)

    response = asyncio.run(
        chat_api.chat_stream(
            ChatRequest(Id="session-test", Question="CPU 怎么排查？"),
            _FakeRequest(fake_request_context),
        )
    )
    events = asyncio.run(_collect_sse_events(response))

    assert [event["payload"]["type"] for event in events] == ["start", "fallback", "done"]
    assert events[1]["payload"]["reason_code"] == "LLM_TIMEOUT"
    assert events[1]["payload"]["fallback_used"] is True
    assert events[2]["payload"]["fallback_used"] is True


def test_aiops_stream_emits_fallback_then_done_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """AIOps SSE 可降级错误应输出 fallback 事件并结束为 done，保留旧 message 事件。"""

    aiops_api = _load_api_module(monkeypatch, "aiops")
    from app.models.aiops import AIOpsRequest

    async def _timeout_diagnose(session_id: str) -> AsyncGenerator[dict[str, object], None]:
        _ = session_id
        raise LLMTimeoutError()
        yield {}

    monkeypatch.setattr(aiops_api.aiops_service, "diagnose", _timeout_diagnose)

    response = asyncio.run(
        aiops_api.diagnose_stream(
            AIOpsRequest(session_id="session-test"),
            _FakeRequest(fake_request_context),
        )
    )
    events = asyncio.run(_collect_sse_events(response))

    assert [event["payload"]["type"] for event in events] == ["start", "fallback", "done"]
    assert events[1]["payload"]["reason_code"] == "LLM_TIMEOUT"
    assert events[1]["payload"]["fallback_used"] is True
    assert events[2]["payload"]["fallback_used"] is True
