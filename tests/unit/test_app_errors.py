"""统一错误模型的回归测试。

这些测试只覆盖 ISSUE-001 的边界：错误码、HTTP 状态、用户可见 envelope、
旧前端兼容字段以及异常脱敏。RequestContext/TraceLogger 会在后续 issue 中接入，
因此这里用显式 trace_id/request_id 验证错误模型本身的输出契约。
"""

import json

from fastapi import HTTPException

from app.core.errors import AppError, InvalidInputError, ToolTimeoutError


def test_invalid_input_error_keeps_legacy_chat_fields_and_trace_ids() -> None:
    error = InvalidInputError(user_message="问题不能为空。", internal_message="raw empty input")

    response = error.to_error_response(trace_id="trc_test", request_id="req_test")

    assert response["success"] is False
    assert response["code"] == 400
    assert response["message"] == "问题不能为空。"
    assert response["data"] == {
        "success": False,
        "answer": None,
        "errorMessage": "问题不能为空。",
    }
    assert response["error"]["code"] == "INVALID_INPUT"
    assert response["error"]["message"] == "问题不能为空。"
    assert response["error"]["retryable"] is False
    assert response["error"]["fallback_required"] is False
    assert response["error"]["trace_id"] == "trc_test"
    assert response["error"]["request_id"] == "req_test"
    assert response["trace_id"] == "trc_test"
    assert response["request_id"] == "req_test"


def test_tool_timeout_error_maps_to_retryable_504_response() -> None:
    error = ToolTimeoutError(tool_name="retrieve_knowledge")

    response = error.to_error_response(trace_id="trc_tool", request_id="req_tool")

    assert response["code"] == 504
    assert response["message"] == "工具调用超时，请稍后重试。"
    assert response["error"]["code"] == "TOOL_TIMEOUT"
    assert response["error"]["retryable"] is True
    assert response["error"]["fallback_required"] is True


def test_unknown_exception_is_wrapped_without_leaking_raw_exception_text() -> None:
    raw = RuntimeError("dashscope_api_key=sk-secret failed at http://internal.service")

    error = AppError.from_exception(raw, origin_module="unit-test")
    response = error.to_error_response(trace_id="trc_safe", request_id="req_safe")

    assert response["code"] == 500
    assert response["message"] == "服务内部错误。"
    serialized = json.dumps(response, ensure_ascii=False)
    assert "sk-secret" not in serialized
    assert "http://internal.service" not in serialized
    assert "dashscope_api_key" not in serialized
    assert error.internal_message.startswith("RuntimeError:")


def test_http_exception_is_mapped_to_stable_app_error() -> None:
    raw = HTTPException(status_code=400, detail="文件名不能为空")

    error = AppError.from_exception(raw)
    response = error.to_error_response(trace_id="trc_http", request_id="req_http")

    assert response["code"] == 400
    assert response["error"]["code"] == "INVALID_INPUT"
    assert response["message"] == "文件名不能为空"


def test_json_response_uses_error_http_status_and_body() -> None:
    error = ToolTimeoutError(tool_name="retrieve_knowledge")

    response = error.to_json_response(trace_id="trc_json", request_id="req_json")
    body = json.loads(response.body)

    assert response.status_code == 504
    assert body["error"]["code"] == "TOOL_TIMEOUT"
    assert body["trace_id"] == "trc_json"
