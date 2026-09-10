"""
AIOps 智能运维接口
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.agent.orchestrator import agent_orchestrator
from app.config import config
from app.core.errors import AppError, JsonObject, SSEStreamInterruptedError
from app.core.fallback import FallbackManager
from app.core.input_guard import input_guard
from app.core.request_context import RequestContext, get_request_context_or_none
from app.models.aiops import AIOpsRequest
from app.observability.tracing import TraceLogger
from app.services.aiops_service import aiops_service

router = APIRouter()
_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)
_fallback_manager = FallbackManager(trace_logger=_trace_logger)


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest, http_request: Request):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 自动获取当前系统的活动告警
    - 使用 Plan-Execute-Replan 模式进行智能诊断
    - 流式返回诊断过程和结果

    **SSE 事件类型：**

    1. `status` - 状态更新
       ```json
       {
         "type": "status",
         "stage": "fetching_alerts",
         "message": "正在获取系统告警信息..."
       }
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {
         "type": "plan",
         "stage": "plan_created",
         "message": "诊断计划已制定，共 6 个步骤",
         "target_alert": {...},
         "plan": ["步骤1: ...", "步骤2: ..."]
       }
       ```

    3. `step_complete` - 步骤执行完成
       ```json
       {
         "type": "step_complete",
         "stage": "step_executed",
         "message": "步骤执行完成 (2/6)",
         "current_step": "查询系统日志",
         "result_preview": "...",
         "remaining_steps": 4
       }
       ```

    4. `report` - 最终诊断报告
       ```json
       {
         "type": "report",
         "stage": "final_report",
         "message": "最终诊断报告已生成",
         "report": "# 故障诊断报告\\n...",
         "evidence": {...}
       }
       ```

    5. `complete` - 诊断完成
       ```json
       {
         "type": "complete",
         "stage": "diagnosis_complete",
         "message": "诊断流程完成",
         "diagnosis": {...}
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "诊断过程发生错误: ..."
       }
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    **前端使用示例：**
    ```javascript
    const eventSource = new EventSource('/api/aiops');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'plan') {
        console.log('诊断计划:', data.plan);
      } else if (data.type === 'step_complete') {
        console.log('步骤完成:', data.current_step);
      } else if (data.type === 'report') {
        console.log('最终报告:', data.report);
      } else if (data.type === 'complete') {
        console.log('诊断完成');
        eventSource.close();
      }
    };
    ```

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    ctx = get_request_context_or_none(http_request)
    try:
        guarded = input_guard.validate_aiops(request, ctx)
    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.aiops")
        logger.error(f"AIOps 输入校验失败: {app_error.code}")
        return app_error.to_json_response(**_trace_kwargs(ctx))

    session_id = guarded.value.session_id
    ctx = _bind_session(http_request, session_id)
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        try:
            if await _is_client_disconnected(http_request):
                _record_sse_disconnect(ctx, route="aiops", before_start=True)
                return

            yield _sse_message(_build_stream_start_payload(session_id, ctx))
            # ISSUE-015 的编排层只包住旧 AIOpsService：打开时补 trace/fallback 边界，
            # 关闭时仍回到旧 diagnose 事件流，保证旧前端监听 type=complete/done 的逻辑不变。
            source = (
                _active_aiops_orchestrator().run_aiops(session_id=session_id, ctx=ctx)
                if config.orchestrator_enabled
                else aiops_service.diagnose(session_id=session_id)
            )

            async for event in _iterate_sse_source(
                source,
                ctx=ctx,
                request=http_request,
                route="aiops",
            ):
                if event.get("type") == "error":
                    # AIOpsService 旧实现会把底层异常文本拼到 message 中。SSE 连接
                    # 已经建立后不能改 HTTP status，因此这里保留旧 type/message 字段，
                    # 同时追加标准 error envelope，避免泄漏内部异常全文。
                    app_error = _app_error_from_sse_event(event)
                    fallback_result = _fallback_manager.for_aiops(app_error, ctx)
                    if fallback_result.fallback_used:
                        # AIOps SSE 的 fallback 仍通过 event: message 发送，避免旧客户端监听逻辑失效；
                        # 最终 done 明确带 fallback_used，方便前端区分完整报告和降级报告。
                        yield _fallback_manager.to_sse_event(fallback_result, ctx)
                        yield _fallback_manager.to_sse_done_event(
                            fallback_result,
                            ctx,
                            session_id=session_id,
                        )
                        break
                    if _has_error_envelope(event):
                        event = _with_stream_trace(dict(event), ctx)
                    else:
                        event = app_error.to_sse_payload(
                            stage=str(event.get("stage", "error")),
                            **_trace_kwargs(ctx),
                        )
                else:
                    event = _with_stream_trace(dict(event), ctx)

                event_type = event.get("type")

                # 发送事件
                yield _sse_message(event)

                if event_type == "complete":
                    done_event = dict(event)
                    done_event["type"] = "done"
                    yield _sse_message(done_event)
                    break

                # 如果是完成或错误事件，结束流
                if event_type in ["done", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except asyncio.CancelledError:
            _record_sse_disconnect(ctx, route="aiops", before_start=False)
            return
        except Exception as e:
            can_use_fallback = isinstance(e, AppError)
            app_error = (
                e
                if can_use_fallback
                else SSEStreamInterruptedError(
                    internal_message=f"{e.__class__.__name__}: {e}",
                )
            )
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {app_error.code}")
            fallback_result = (
                _fallback_manager.for_aiops(app_error, ctx) if can_use_fallback else None
            )
            if fallback_result is not None and fallback_result.fallback_used:
                # 这里不使用原始异常 message 生成文案，避免工具/MCP/模型异常中的内网地址或密钥
                # 在 SSE 已建立后绕过 HTTP 错误脱敏路径。
                yield _fallback_manager.to_sse_event(fallback_result, ctx)
                yield _fallback_manager.to_sse_done_event(
                    fallback_result,
                    ctx,
                    session_id=session_id,
                )
                return
            yield {
                "event": "message",
                "data": json.dumps(
                    app_error.to_sse_payload(stage="exception", **_trace_kwargs(ctx)),
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


def _bind_session(request: Request, session_id: str | None) -> RequestContext | None:
    """把 AIOps session 写入请求上下文，供 SSE error/complete 和 request.end 串联。"""

    ctx = get_request_context_or_none(request)
    if ctx is None:
        return None
    updated_ctx = ctx.with_session(session_id)
    request.state.ctx = updated_ctx
    return updated_ctx


def _trace_kwargs(ctx: RequestContext | None) -> dict[str, str | None]:
    if ctx is None:
        return {"trace_id": None, "request_id": None}
    return {"trace_id": ctx.trace_id, "request_id": ctx.request_id}


def _with_stream_trace(payload: JsonObject, ctx: RequestContext | None) -> JsonObject:
    """给旧 AIOps SSE payload 追加 trace 字段，不改变现有 `type`。"""

    if ctx is not None:
        payload["trace_id"] = ctx.trace_id
        payload["request_id"] = ctx.request_id
    return payload


def _build_stream_start_payload(session_id: str, ctx: RequestContext | None) -> JsonObject:
    """构建 AIOps SSE start payload，并继续使用旧 `event: message` 传输。"""

    return _with_stream_trace(
        {
            "type": "start",
            "session_id": session_id,
            "mode": "aiops",
            "created_at": datetime.now(UTC).isoformat(),
        },
        ctx,
    )


def _sse_message(payload: JsonObject) -> dict[str, str]:
    """统一生成兼容旧前端的 SSE message 事件。"""

    return {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}


def _active_aiops_orchestrator():
    """返回绑定当前 AIOps service 的编排器。

    AIOps API 的旧注入点是 `aiops_service`，测试和回滚都会替换它；编排层使用前同步引用，
    避免开启 orchestrator 后绕过 fake service 或运行时替换实例。
    """

    agent_orchestrator.bind_services(aiops_service=aiops_service)
    return agent_orchestrator


def _request_deadline_timeout_error(origin_module: str) -> AppError:
    """构造不触发 fallback 的请求级超时错误，避免对外暴露 asyncio 原始异常。"""

    return AppError(
        "LLM_TIMEOUT",
        fallback_required=False,
        internal_message="request deadline exceeded",
        origin_module=origin_module,
    )


def _app_error_from_sse_event(event: dict[str, object]) -> AppError:
    """从 AIOps service error event 中恢复 AppError。

    标准 error envelope 只信任稳定 `error.code`，不复用 message 作为 user_message；
    旧 service 的 message 可能含原始异常，因此走 AppError.from_exception 生成安全内部错误。
    """

    error = event.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            fallback_required = error.get("fallback_required")
            retryable = error.get("retryable")
            return AppError(
                code,
                fallback_required=(
                    fallback_required if isinstance(fallback_required, bool) else None
                ),
                retryable=retryable if isinstance(retryable, bool) else None,
            )
    return AppError.from_exception(
        RuntimeError(str(event.get("message", "AIOps stream error"))),
        origin_module="app.api.aiops",
    )


def _has_error_envelope(event: dict[str, object]) -> bool:
    """识别服务层已经标准化的错误事件，避免二次包成 INTERNAL_ERROR。"""

    error = event.get("error")
    return isinstance(error, dict) and isinstance(error.get("code"), str)


async def _iterate_sse_source(
    async_iterable: object,
    *,
    ctx: RequestContext | None,
    request: Request,
    route: str,
) -> AsyncGenerator[dict[str, object], None]:
    """Check disconnect and deadline before each downstream SSE pull."""

    iterator = async_iterable.__aiter__()  # type: ignore[attr-defined]
    try:
        while True:
            if await _is_client_disconnected(request):
                _record_sse_disconnect(ctx, route=route, before_start=False)
                return
            try:
                yield await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=_request_timeout_seconds(ctx),
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise _request_deadline_timeout_error("app.api.aiops") from exc
    finally:
        await _close_async_iterator(iterator)


def _request_timeout_seconds(ctx: RequestContext | None) -> float:
    timeout_ms = (
        ctx.remaining_ms()
        if ctx is not None
        else int(getattr(config, "request_timeout_ms", 60_000))
    )
    return max(timeout_ms, 0) / 1000


async def _close_async_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()


async def _is_client_disconnected(request: object) -> bool:
    """检测 SSE 客户端是否已断开，缺少该方法时保持旧 handler 兼容。"""

    checker = getattr(request, "is_disconnected", None)
    if not callable(checker):
        return False
    try:
        return bool(await checker())
    except RuntimeError:
        return False


def _record_sse_disconnect(
    ctx: RequestContext | None,
    *,
    route: str,
    before_start: bool,
) -> None:
    """记录客户端断开事件，不再尝试向已断开的连接发送错误包。"""

    if ctx is None:
        return
    _trace_logger.record_event(
        "sse.client_disconnect",
        ctx,
        status="cancelled",
        route=route,
        before_start=before_start,
    )
