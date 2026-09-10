"""统一应用错误模型。

本模块是 ISSUE-001 的边界产物：它只负责把内部异常映射为稳定的错误码、
HTTP status 和对外安全 envelope，不负责 RequestContext、TraceLogger 或
FallbackManager 的生命周期管理。后续 ISSUE-002 接入请求上下文后，API 层可以把
真实 trace_id/request_id 传入这些方法；在那之前这里生成兜底 ID，保证错误响应
已经满足对外契约的可追踪字段要求。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from fastapi import HTTPException
from fastapi.responses import JSONResponse

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class ErrorDefinition:
    """错误码注册信息。

    错误码表集中定义，避免不同 API handler 自己拼 status/message，导致同一个
    失败场景在前端、SSE 和后续 trace 中出现不同语义。
    """

    code: str
    http_status: int
    user_message: str
    retryable: bool
    fallback_required: bool


ERROR_DEFINITIONS: dict[str, ErrorDefinition] = {
    "INVALID_INPUT": ErrorDefinition("INVALID_INPUT", 400, "请求参数不合法。", False, False),
    "INVALID_SESSION_ID": ErrorDefinition(
        "INVALID_SESSION_ID", 400, "会话 ID 不合法。", False, False
    ),
    "REQUEST_TOO_LARGE": ErrorDefinition("REQUEST_TOO_LARGE", 413, "请求内容过长。", False, False),
    "FILE_TOO_LARGE": ErrorDefinition("FILE_TOO_LARGE", 413, "文件大小超过限制。", False, False),
    "UNSUPPORTED_FILE_TYPE": ErrorDefinition(
        "UNSUPPORTED_FILE_TYPE", 400, "不支持的文件类型。", False, False
    ),
    "INVALID_FILE_MIME": ErrorDefinition(
        "INVALID_FILE_MIME", 415, "文件内容类型与扩展名不匹配。", False, False
    ),
    "INVALID_FILE_ENCODING": ErrorDefinition(
        "INVALID_FILE_ENCODING", 400, "文件必须是 UTF-8 编码文本。", False, False
    ),
    "INVALID_DIRECTORY": ErrorDefinition(
        "INVALID_DIRECTORY", 400, "目录不存在或不允许索引。", False, False
    ),
    "PATH_TRAVERSAL_BLOCKED": ErrorDefinition(
        "PATH_TRAVERSAL_BLOCKED", 400, "路径不允许包含目录逃逸。", False, False
    ),
    "SYMLINK_NOT_ALLOWED": ErrorDefinition(
        "SYMLINK_NOT_ALLOWED", 400, "不允许索引符号链接。", False, False
    ),
    "TOOL_TIMEOUT": ErrorDefinition("TOOL_TIMEOUT", 504, "工具调用超时，请稍后重试。", True, True),
    "TOOL_EXECUTION_ERROR": ErrorDefinition(
        "TOOL_EXECUTION_ERROR", 502, "工具调用失败。", True, True
    ),
    "LLM_TIMEOUT": ErrorDefinition("LLM_TIMEOUT", 504, "模型响应超时，请稍后重试。", True, True),
    "LLM_EMPTY_RESPONSE": ErrorDefinition(
        "LLM_EMPTY_RESPONSE", 502, "模型未返回有效内容。", True, True
    ),
    "LLM_PROVIDER_ERROR": ErrorDefinition(
        "LLM_PROVIDER_ERROR", 502, "模型服务暂时不可用。", True, True
    ),
    "RAG_EMPTY_RESULT": ErrorDefinition(
        "RAG_EMPTY_RESULT", 404, "未找到足够相关的知识库内容。", False, True
    ),
    "RAG_METADATA_INVALID": ErrorDefinition(
        "RAG_METADATA_INVALID", 500, "知识库文档元数据不完整。", False, True
    ),
    "VECTOR_STORE_UNAVAILABLE": ErrorDefinition(
        "VECTOR_STORE_UNAVAILABLE", 503, "知识库暂时不可用。", True, True
    ),
    "EMBEDDING_PROVIDER_ERROR": ErrorDefinition(
        "EMBEDDING_PROVIDER_ERROR", 502, "向量化服务暂时不可用。", True, False
    ),
    "AGENT_MAX_STEP_EXCEEDED": ErrorDefinition(
        "AGENT_MAX_STEP_EXCEEDED", 409, "任务步骤过多，已停止继续执行。", False, True
    ),
    "SSE_STREAM_INTERRUPTED": ErrorDefinition(
        "SSE_STREAM_INTERRUPTED", 200, "流式响应中断。", True, True
    ),
    "INTERNAL_ERROR": ErrorDefinition("INTERNAL_ERROR", 500, "服务内部错误。", True, True),
}

_HTTP_STATUS_TO_CODE: dict[int, str] = {
    400: "INVALID_INPUT",
    413: "REQUEST_TOO_LARGE",
    415: "INVALID_FILE_MIME",
    500: "INTERNAL_ERROR",
    502: "TOOL_EXECUTION_ERROR",
    503: "VECTOR_STORE_UNAVAILABLE",
    504: "TOOL_TIMEOUT",
}

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s,;\"']+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*=\s*([^\s,;]+)")
_SECRET_VALUE_PATTERN = re.compile(r"\b(sk|ak)-[A-Za-z0-9_\-]{4,}\b")


class AppError(Exception):
    """应用内统一错误基类。

    `user_message` 是唯一允许进入 HTTP/SSE 响应的错误文案；`internal_message`
    保留给日志和后续 trace 使用。这样 API handler 可以记录诊断信息，但不会把
    堆栈、密钥、内部 URL 或原始异常全文直接暴露给用户。
    """

    def __init__(
        self,
        code: str = "INTERNAL_ERROR",
        *,
        user_message: str | None = None,
        internal_message: str | None = None,
        retryable: bool | None = None,
        fallback_required: bool | None = None,
        http_status: int | None = None,
        details: Mapping[str, JsonValue] | None = None,
        origin_module: str | None = None,
    ) -> None:
        definition = ERROR_DEFINITIONS.get(code, ERROR_DEFINITIONS["INTERNAL_ERROR"])
        self.code = definition.code
        self.http_status = http_status if http_status is not None else definition.http_status
        self.user_message = user_message or definition.user_message
        self.internal_message = internal_message or self.user_message
        self.retryable = retryable if retryable is not None else definition.retryable
        self.fallback_required = (
            fallback_required if fallback_required is not None else definition.fallback_required
        )
        self.trace_required = True
        self.details = dict(details or {})
        self.origin_module = origin_module
        super().__init__(self.internal_message)

    def is_retryable(self) -> bool:
        """返回调用方是否值得重试，供 API/SSE adapter 和后续 fallback 策略使用。"""

        return self.retryable

    def requires_fallback(self) -> bool:
        """返回该错误是否可能触发降级；具体降级文案留给后续 FallbackManager。"""

        return self.fallback_required

    def safe_details(self) -> JsonObject:
        """返回脱敏后的诊断 details。

        details 目前不默认写入对外响应，但后续 trace 或受控调试场景会复用这里的
        脱敏逻辑，避免每个模块重复实现且遗漏密钥字段。
        """

        return {key: _sanitize_detail(key, value) for key, value in self.details.items()}

    def to_error_response(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
        legacy_data: JsonObject | None = None,
    ) -> JsonObject:
        """转换为 API 契约要求的统一错误 envelope。

        `code/message/data.success/answer/errorMessage` 是旧前端仍依赖的字段，
        因此即使新增了标准 `error` 对象，也必须保留这些兼容字段。
        """

        resolved_trace_id = trace_id or _generate_id("trc")
        resolved_request_id = request_id or _generate_id("req")
        compatibility_data = legacy_data or {
            "success": False,
            "answer": None,
            "errorMessage": self.user_message,
        }
        error_body: JsonObject = {
            "code": self.code,
            "message": self.user_message,
            "retryable": self.retryable,
            "fallback_required": self.fallback_required,
            "trace_id": resolved_trace_id,
            "request_id": resolved_request_id,
        }
        return {
            "success": False,
            "code": self.http_status,
            "message": self.user_message,
            "data": compatibility_data,
            "error": error_body,
            "trace_id": resolved_trace_id,
            "request_id": resolved_request_id,
        }

    def to_json_response(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
        legacy_data: JsonObject | None = None,
    ) -> JSONResponse:
        """生成 FastAPI 可直接返回的 JSONResponse。"""

        return JSONResponse(
            status_code=self.http_status,
            content=self.to_error_response(
                trace_id=trace_id,
                request_id=request_id,
                legacy_data=legacy_data,
            ),
        )

    def to_sse_payload(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
        stage: str = "error",
    ) -> JsonObject:
        """生成兼容旧 `data.type=error` 的 SSE 错误 payload。

        Chat SSE 旧前端读取 `data` 字段，AIOps 旧前端读取 `message` 字段；两者都
        保留，同时新增标准 `error` 对象和 trace/request 字段。
        """

        response = self.to_error_response(trace_id=trace_id, request_id=request_id)
        error_body = cast(JsonObject, response["error"])
        return {
            "type": "error",
            "stage": stage,
            "message": self.user_message,
            "data": self.user_message,
            "trace_id": cast(str, response["trace_id"]),
            "request_id": cast(str, response["request_id"]),
            "error": error_body,
        }

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        origin_module: str | None = None,
    ) -> AppError:
        """把任意异常归一为 AppError。

        未分类异常一律变成 `INTERNAL_ERROR`，对外只返回稳定文案。这里刻意不把
        `str(exc)` 作为用户文案，避免泄漏堆栈、密钥、内部 URL 或下游原始错误。
        """

        if isinstance(exc, AppError):
            return exc

        if isinstance(exc, HTTPException):
            code = _HTTP_STATUS_TO_CODE.get(exc.status_code, "INTERNAL_ERROR")
            detail_message = _safe_http_detail(exc)
            return cls(
                code,
                user_message=detail_message,
                internal_message=f"HTTPException({exc.status_code}): {exc.detail}",
                http_status=exc.status_code,
                origin_module=origin_module,
            )

        return InternalAppError(
            internal_message=f"{exc.__class__.__name__}: {exc}",
            origin_module=origin_module,
        )


class InvalidInputError(AppError):
    """请求参数非法。"""

    def __init__(
        self,
        user_message: str | None = None,
        *,
        internal_message: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
        origin_module: str | None = None,
    ) -> None:
        super().__init__(
            "INVALID_INPUT",
            user_message=user_message,
            internal_message=internal_message,
            details=details,
            origin_module=origin_module,
        )


class InvalidSessionIdError(AppError):
    """会话 ID 格式非法。"""

    def __init__(
        self,
        user_message: str | None = None,
        *,
        internal_message: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(
            "INVALID_SESSION_ID",
            user_message=user_message,
            internal_message=internal_message,
            details=details,
        )


class RequestTooLargeError(AppError):
    """请求内容超过上限。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("REQUEST_TOO_LARGE", user_message=user_message)


class FileTooLargeError(AppError):
    """上传文件超过上限。"""

    def __init__(self, *, max_bytes: int | None = None) -> None:
        details: JsonObject = {}
        if max_bytes is not None:
            details["max_bytes"] = max_bytes
        super().__init__("FILE_TOO_LARGE", details=details)


class UnsupportedFileTypeError(AppError):
    """上传或索引的文件类型不在允许列表内。"""

    def __init__(
        self,
        user_message: str | None = None,
        *,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__("UNSUPPORTED_FILE_TYPE", user_message=user_message, details=details)


class InvalidFileMimeError(AppError):
    """文件 MIME 与扩展名不匹配。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("INVALID_FILE_MIME", user_message=user_message)


class InvalidFileEncodingError(AppError):
    """文件编码不是 UTF-8 文本。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("INVALID_FILE_ENCODING", user_message=user_message)


class InvalidDirectoryError(AppError):
    """目录不存在或不允许索引。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("INVALID_DIRECTORY", user_message=user_message)


class PathTraversalBlockedError(AppError):
    """路径存在目录逃逸风险。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("PATH_TRAVERSAL_BLOCKED", user_message=user_message)


class SymlinkNotAllowedError(AppError):
    """请求路径或文件是符号链接。"""

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__("SYMLINK_NOT_ALLOWED", user_message=user_message)


class ToolTimeoutError(AppError):
    """工具调用超时。"""

    def __init__(self, *, tool_name: str | None = None) -> None:
        details: JsonObject = {}
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(
            "TOOL_TIMEOUT",
            internal_message=f"Tool timed out: {tool_name or 'unknown'}",
            details=details,
        )


class ToolExecutionError(AppError):
    """工具执行失败。"""

    def __init__(
        self,
        user_message: str | None = None,
        *,
        internal_message: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(
            "TOOL_EXECUTION_ERROR",
            user_message=user_message,
            internal_message=internal_message,
            details=details,
        )


class LLMTimeoutError(AppError):
    """模型响应超时。"""

    def __init__(self) -> None:
        super().__init__("LLM_TIMEOUT")


class LLMEmptyResponseError(AppError):
    """模型返回空内容或无效内容。"""

    def __init__(self) -> None:
        super().__init__("LLM_EMPTY_RESPONSE")


class LLMProviderError(AppError):
    """模型服务商错误。"""

    def __init__(self, *, internal_message: str | None = None) -> None:
        super().__init__("LLM_PROVIDER_ERROR", internal_message=internal_message)


class RagEmptyResultError(AppError):
    """RAG 未检索到足够证据。"""

    def __init__(self) -> None:
        super().__init__("RAG_EMPTY_RESULT")


class RagMetadataInvalidError(AppError):
    """RAG 内部文档元数据缺失或语义不安全。

    该错误主要供阶段 3A 之后的索引、检索和 citation 模块内部使用。它不把原始
    metadata 直接暴露给用户，只通过 details 记录字段名等低敏诊断信息，避免
    chunk 内容、绝对路径或下游异常穿透到 HTTP/SSE 响应。
    """

    def __init__(
        self,
        user_message: str | None = None,
        *,
        internal_message: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(
            "RAG_METADATA_INVALID",
            user_message=user_message,
            internal_message=internal_message,
            details=details,
            origin_module="app.rag",
        )


class VectorStoreUnavailableError(AppError):
    """向量库暂时不可用。"""

    def __init__(self, *, internal_message: str | None = None) -> None:
        super().__init__("VECTOR_STORE_UNAVAILABLE", internal_message=internal_message)


class EmbeddingProviderError(AppError):
    """向量化服务错误。"""

    def __init__(self, *, internal_message: str | None = None) -> None:
        super().__init__("EMBEDDING_PROVIDER_ERROR", internal_message=internal_message)


class AgentMaxStepExceededError(AppError):
    """Agent 执行步骤超过业务上限。"""

    def __init__(self) -> None:
        super().__init__("AGENT_MAX_STEP_EXCEEDED")


class SSEStreamInterruptedError(AppError):
    """SSE 已建立后发生中断。"""

    def __init__(self, *, internal_message: str | None = None) -> None:
        super().__init__("SSE_STREAM_INTERRUPTED", internal_message=internal_message)


class InternalAppError(AppError):
    """未分类服务端异常。"""

    def __init__(
        self,
        *,
        internal_message: str | None = None,
        origin_module: str | None = None,
    ) -> None:
        super().__init__(
            "INTERNAL_ERROR",
            internal_message=internal_message,
            origin_module=origin_module,
        )


def _safe_http_detail(exc: HTTPException) -> str:
    definition = ERROR_DEFINITIONS.get(_HTTP_STATUS_TO_CODE.get(exc.status_code, "INTERNAL_ERROR"))
    default_message = (
        definition.user_message if definition else ERROR_DEFINITIONS["INTERNAL_ERROR"].user_message
    )
    if exc.status_code >= 500:
        return default_message
    if isinstance(exc.detail, str) and exc.detail.strip():
        return _sanitize_text(exc.detail.strip())
    return default_message


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _sanitize_detail(key: str, value: JsonValue) -> JsonValue:
    if _SENSITIVE_KEY_PATTERN.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_detail(key, item) for item in value]
    if isinstance(value, dict):
        return {
            child_key: _sanitize_detail(child_key, child_value)
            for child_key, child_value in value.items()
        }
    return value


def _sanitize_text(value: str) -> str:
    """脱敏可进入受控输出的文本片段。

    该函数用于 HTTPException 的 4xx detail 和 future trace details。它保留用户可读
    的业务文案，但替换常见密钥、token 和内部 URL。
    """

    sanitized = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1=<redacted>", value)
    sanitized = _SECRET_VALUE_PATTERN.sub("<redacted>", sanitized)
    sanitized = _URL_PATTERN.sub("<redacted-url>", sanitized)
    return sanitized
