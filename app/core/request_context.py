"""请求上下文和最小 trace middleware。

ISSUE-002 只建立每次 HTTP 请求的 `trace_id/request_id` 传播边界，不读取请求
body，也不引入认证、InputGuard、ToolManager 或 Agent 编排。这样后续 issue 可以
把同一个上下文显式传给 guard/tool/RAG，同时当前上传和旧 API 行为不会被提前改动。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from json import JSONDecodeError
from typing import TYPE_CHECKING, cast

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.errors import AppError, JsonObject

if TYPE_CHECKING:
    from app.observability.tracing import TraceLogger


_REQUEST_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "super_biz_request_context",
    default=None,
)
_SAFE_HEADER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEFAULT_PERMISSIONS = (
    "chat:read",
    "rag:query",
    "file:upload",
    "aiops:diagnose",
)


@dataclass(frozen=True)
class RequestContext:
    """一次请求内共享的只读上下文。

    `started_monotonic` 是内部计时字段，用单调时钟计算 latency/remaining，避免系统
    时间调整导致负耗时；对外 trace 仍使用 `started_at` 作为可读时间来源。
    """

    trace_id: str
    request_id: str
    session_id: str | None
    tenant_id: str
    user_id: str
    deadline_ms: int
    feature_flags: tuple[str, ...]
    started_at: float
    started_monotonic: float
    method: str
    path: str
    invalid_inbound_trace_header: bool
    permissions: tuple[str, ...] = _DEFAULT_PERMISSIONS

    @classmethod
    def from_request(cls, request: Request, *, request_timeout_ms: int) -> RequestContext:
        """从 FastAPI Request headers 创建上下文。

        header 只做轻量格式校验：非法、空值、过长或包含控制/空白字符时重新生成。
        这样既能透传上游合法 trace，又不会把恶意 header 原样写回响应和 trace。
        """

        trace_id, invalid_trace = _resolve_id_header(
            request.headers.get("X-Trace-Id"),
            prefix="trc",
        )
        request_id, invalid_request = _resolve_id_header(
            request.headers.get("X-Request-Id"),
            prefix="req",
        )
        return cls(
            trace_id=trace_id,
            request_id=request_id,
            session_id=None,
            tenant_id=_resolve_identity_header(
                request.headers.get("X-Tenant-Id"),
                default="default",
            ),
            user_id=_resolve_identity_header(
                request.headers.get("X-User-Id"),
                default="anonymous",
            ),
            deadline_ms=request_timeout_ms,
            feature_flags=_parse_feature_flags(request.headers.get("X-Feature-Flags")),
            started_at=time.time(),
            started_monotonic=time.monotonic(),
            method=request.method,
            path=request.url.path,
            invalid_inbound_trace_header=invalid_trace or invalid_request,
        )

    def with_session(self, session_id: str | None) -> RequestContext:
        """返回带业务 session 的上下文副本。

        middleware 不解析请求 body，避免消耗上传/JSON 流；API handler 在 Pydantic
        model 已经解析完成后再调用此方法补充 session，是对旧接口风险最小的做法。
        """

        normalized_session = session_id.strip() if isinstance(session_id, str) else None
        return replace(self, session_id=normalized_session or None)

    def latency_ms(self) -> float:
        """返回从 middleware 创建上下文到当前的耗时毫秒。"""

        return round((time.monotonic() - self.started_monotonic) * 1000, 3)

    def remaining_ms(self) -> int:
        """返回本请求剩余预算毫秒数；当前阶段只记录，不强制中断业务。"""

        remaining = self.deadline_ms - self.latency_ms()
        return max(0, int(remaining))

    def to_trace_fields(self) -> JsonObject:
        """转换为 TraceLogger 公共字段，避免各模块重复拼 trace 关键字段。"""

        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "method": self.method,
            "path": self.path,
            "deadline_ms": self.deadline_ms,
            "remaining_ms": self.remaining_ms(),
            "feature_flags": list(self.feature_flags),
            "invalid_inbound_trace_header": self.invalid_inbound_trace_header,
        }


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为 FastAPI 请求注入 RequestContext 并追加 trace 响应字段。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        trace_logger: TraceLogger | None = None,
        request_timeout_ms: int | None = None,
    ) -> None:
        super().__init__(app)
        if trace_logger is None or request_timeout_ms is None:
            from app.config import config
            from app.observability.tracing import TraceLogger

            trace_logger = trace_logger or TraceLogger(
                trace_jsonl_path=config.trace_jsonl_path,
                enabled=config.trace_enabled,
            )
            request_timeout_ms = request_timeout_ms or config.request_timeout_ms
        self.trace_logger = trace_logger
        self.request_timeout_ms = request_timeout_ms

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        ctx = RequestContext.from_request(
            request,
            request_timeout_ms=self.request_timeout_ms,
        )
        request.state.ctx = ctx
        token = _REQUEST_CONTEXT.set(ctx)
        self.trace_logger.record_event("request.start", ctx, status="start")

        try:
            response = await call_next(request)
        except Exception as exc:
            current_ctx = _ctx_from_request_state(request) or ctx
            app_error = AppError.from_exception(
                exc,
                origin_module="app.core.request_context",
            )
            self.trace_logger.record_error(app_error, current_ctx)
            response = app_error.to_json_response(
                trace_id=current_ctx.trace_id,
                request_id=current_ctx.request_id,
            )
            return await self._finalize_response(response, current_ctx)
        finally:
            _REQUEST_CONTEXT.reset(token)

        current_ctx = _ctx_from_request_state(request) or ctx
        self.trace_logger.record_event(
            "request.end",
            current_ctx,
            status="ok" if response.status_code < 500 else "error",
            status_code=response.status_code,
            latency_ms=current_ctx.latency_ms(),
        )
        return await self._finalize_response(response, current_ctx)

    async def _finalize_response(self, response: Response, ctx: RequestContext) -> Response:
        """追加 trace headers，并在 JSON body 中追加/覆盖 trace 字段。

        这里只读取响应体，不读取请求体；同时跳过 `text/event-stream` 等非 JSON 响应，
        避免破坏现有 SSE `event: message` 流式兼容。
        """

        response.headers["X-Trace-Id"] = ctx.trace_id
        response.headers["X-Request-Id"] = ctx.request_id

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return response

        body = await _consume_response_body(response)
        if not body:
            return response

        try:
            payload = json.loads(body)
        except (JSONDecodeError, UnicodeDecodeError):
            return _rebuild_response(response, body, ctx)

        if isinstance(payload, dict):
            traced_payload = _attach_trace_fields(cast(JsonObject, payload), ctx)
            encoded = json.dumps(traced_payload, ensure_ascii=False).encode("utf-8")
            return _rebuild_response(response, encoded, ctx)

        return _rebuild_response(response, body, ctx)


def get_request_context(request: Request | None = None) -> RequestContext:
    """获取当前请求上下文。

    优先读取 `request.state.ctx`，其次读取 contextvar。显式报错比静默新建 ID 更安全：
    后者会让同一请求内出现多个无法串联的 trace，正是 ISSUE-002 要解决的问题。
    """

    if request is not None:
        ctx = _ctx_from_request_state(request)
        if ctx is not None:
            return ctx
    ctx = _REQUEST_CONTEXT.get()
    if ctx is None:
        raise RuntimeError("RequestContext is not available")
    return ctx


def get_request_context_or_none(request: Request | None = None) -> RequestContext | None:
    """返回当前上下文；缺失时返回 None 以兼容未接 middleware 的单元测试。

    运行时主应用已经在 `app/main.py` 注册 middleware；这里的宽松入口只用于旧测试、
    直接调用 handler 或局部 FastAPI app，避免为了 ISSUE-002 破坏既有测试方式。
    """

    if request is not None:
        ctx = _ctx_from_request_state(request)
        if ctx is not None:
            return ctx
    return _REQUEST_CONTEXT.get()


def set_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    """测试或非 FastAPI adapter 可显式设置上下文。"""

    return _REQUEST_CONTEXT.set(ctx)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    """恢复 `set_request_context` 之前的上下文。"""

    _REQUEST_CONTEXT.reset(token)


def _ctx_from_request_state(request: Request) -> RequestContext | None:
    candidate = getattr(request.state, "ctx", None)
    return candidate if isinstance(candidate, RequestContext) else None


def _resolve_id_header(value: str | None, *, prefix: str) -> tuple[str, bool]:
    if value is None:
        return _generate_id(prefix), False
    stripped = value.strip()
    if _is_safe_header_value(stripped):
        return stripped, False
    return _generate_id(prefix), True


def _resolve_identity_header(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    stripped = value.strip()
    return stripped if _is_safe_header_value(stripped) else default


def _parse_feature_flags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    flags: list[str] = []
    for raw_flag in value.split(","):
        flag = raw_flag.strip()
        if _is_safe_header_value(flag):
            flags.append(flag)
    return tuple(flags)


def _is_safe_header_value(value: str) -> bool:
    return bool(_SAFE_HEADER_RE.fullmatch(value))


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


async def _consume_response_body(response: Response) -> bytes:
    if hasattr(response, "body_iterator"):
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            if isinstance(chunk, bytes):
                chunks.append(chunk)
            else:
                chunks.append(str(chunk).encode("utf-8"))
        return b"".join(chunks)
    body = getattr(response, "body", b"")
    return body if isinstance(body, bytes) else str(body).encode("utf-8")


def _attach_trace_fields(payload: JsonObject, ctx: RequestContext) -> JsonObject:
    payload["trace_id"] = ctx.trace_id
    payload["request_id"] = ctx.request_id
    error = payload.get("error")
    if isinstance(error, dict):
        error["trace_id"] = ctx.trace_id
        error["request_id"] = ctx.request_id
    return payload


def _rebuild_response(response: Response, body: bytes, ctx: RequestContext) -> Response:
    # Body 被重建后长度可能改变；让 Starlette 重新计算，避免旧 Content-Length 截断。
    # MutableHeaders 内部通常使用小写 key，直接追加 `X-Trace-Id` 会形成重复 header，
    # 因此这里按大小写无关方式先清理再写回一份规范 header。
    headers = {
        key: value
        for key, value in dict(response.headers).items()
        if key.lower() not in {"content-length", "x-trace-id", "x-request-id"}
    }
    headers["X-Trace-Id"] = ctx.trace_id
    headers["X-Request-Id"] = ctx.request_id
    return Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
        background=response.background,
    )
