"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.agent.orchestrator import agent_orchestrator
from app.config import config
from app.core.errors import (
    AppError,
    JsonObject,
    JsonValue,
    SSEStreamInterruptedError,
)
from app.core.fallback import FallbackManager
from app.core.input_guard import input_guard
from app.core.request_context import RequestContext, get_request_context_or_none
from app.memory.conversation_store import conversation_history_store
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.observability.tracing import TraceLogger
from app.services.rag_agent_service import rag_agent_service as _imported_rag_agent_service

router = APIRouter()
_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)
_fallback_manager = FallbackManager(trace_logger=_trace_logger)


class _UnavailableRagAgentService:
    """Shape-compatible placeholder for tests that import API modules with stubs."""

    async def query(self, question: str, session_id: str) -> str:
        _ = question, session_id
        raise RuntimeError("RAG service is unavailable")

    async def query_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> str:
        _ = conversation_context, ctx
        return await self.query(question, session_id)

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        _ = question, session_id
        raise RuntimeError("RAG service is unavailable")
        yield {}

    async def query_stream_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> AsyncGenerator[dict[str, object], None]:
        _ = conversation_context, ctx
        async for chunk in self.query_stream(question, session_id):
            yield chunk

    def clear_session(self, session_id: str, ctx: RequestContext | None = None) -> bool:
        _ = session_id, ctx
        return False

    def get_session_history(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> list[dict[str, str]]:
        _ = session_id, ctx
        return []


def _ensure_rag_agent_service_shape(service: object) -> object:
    if callable(getattr(service, "query", None)) and callable(
        getattr(service, "query_stream", None)
    ):
        return service
    return _UnavailableRagAgentService()


rag_agent_service = _ensure_rag_agent_service_shape(_imported_rag_agent_service)


@router.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    """快速对话接口
    {
        "code": 200,
        "message": "success",
        "data": {
            "success": true,
            "answer": "回答内容",
            "errorMessage": null
        }
    }

    Args:
        request: 对话请求

    Returns:
        统一格式的对话响应
    """
    ctx = get_request_context_or_none(http_request)
    try:
        guarded = input_guard.validate_chat(request, ctx)
        validated = guarded.value
        ctx = _bind_session(http_request, validated.session_id)
        logger.info(f"[会话 {validated.session_id}] 收到快速对话请求: {validated.question}")

        if config.orchestrator_enabled:
            # ISSUE-015 只把已完成的上下文、预算、fallback 和 trace 边界串到旧 RAG service
            # 前面；API 层仍只负责校验和响应适配，避免把编排细节泄漏成新的 HTTP 契约。
            result = await _active_chat_orchestrator().run_chat(
                question=validated.question,
                session_id=validated.session_id,
                ctx=ctx,
            )
            response_data = result.to_api_data()
        else:
            # 回滚开关必须保持旧 service 直连路径：一旦编排层接入出现兼容风险，前端仍能拿到
            # 原先的 success/answer/errorMessage 字段，不需要随部署同时改客户端。
            try:
                answer = await asyncio.wait_for(
                    rag_agent_service.query(
                        validated.question,
                        session_id=validated.session_id,
                    ),
                    timeout=_request_timeout_seconds(ctx),
                )
            except TimeoutError as exc:
                # 这里是 request deadline 被 API 层耗尽，而不是下游模型主动给出的可降级
                # LLM_TIMEOUT；保留相同错误码但关闭 fallback，避免把硬超时伪装成成功回答。
                raise _request_deadline_timeout_error("app.api.chat") from exc
            response_data = {"success": True, "answer": answer, "errorMessage": None}

        answer_text = _extract_answer_text(response_data)
        if answer_text is not None:
            await _record_conversation_turn(
                validated.session_id,
                validated.question,
                answer_text,
                ctx,
            )

        logger.info(f"[会话 {validated.session_id}] 快速对话完成")

        return {
            "code": 200,
            "message": "success",
            "data": response_data,
        }

    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.chat")
        logger.error(f"对话接口错误: {app_error.code}")
        fallback_result = _fallback_manager.for_chat(app_error, ctx)
        if fallback_result.fallback_used:
            # 非流式 Chat 的 fallback 按 API 契约返回 HTTP 200，但 data 仍保留旧前端依赖的
            # success/answer/errorMessage 字段；关闭 fallback_enabled 时会自动回到旧错误 envelope。
            return {
                "code": 200,
                "message": "success",
                "data": _fallback_manager.to_response_fields(fallback_result),
            }
        return app_error.to_json_response(**_trace_kwargs(ctx))


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """流式对话接口（基于 RAG Agent，SSE）

    返回 SSE 格式，data 字段为 JSON：

    工具调用事件:
    event: message
    data: {"type":"tool_call","data":{"tool":"工具名","status":"start|end","input":{...}}}

    内容流式事件:
    event: message
    data: {"type":"content","data":"内容块"}

    完成事件:
    event: message
    data: {"type":"done","data":{"answer":"完整答案","tool_calls":[...]}}

    Args:
        request: 对话请求

    Returns:
        SSE 事件流
    """
    ctx = get_request_context_or_none(http_request)
    try:
        guarded = input_guard.validate_chat(request, ctx)
    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.chat_stream")
        logger.error(f"流式对话输入校验失败: {app_error.code}")
        return app_error.to_json_response(**_trace_kwargs(ctx))

    validated = guarded.value
    ctx = _bind_session(http_request, validated.session_id)
    logger.info(f"[会话 {validated.session_id}] 收到流式对话请求: {validated.question}")

    async def event_generator():
        stream_answer_parts: list[str] = []
        try:
            if await _is_client_disconnected(http_request):
                _record_sse_disconnect(ctx, route="chat_stream", before_start=True)
                return

            yield _sse_message(_build_stream_start_payload(validated.session_id, ctx))
            source = (
                _active_chat_orchestrator().run_chat_stream(
                    question=validated.question,
                    session_id=validated.session_id,
                    ctx=ctx,
                )
                if config.orchestrator_enabled
                else rag_agent_service.query_stream(
                    validated.question,
                    session_id=validated.session_id,
                )
            )

            async for chunk in _iterate_sse_source(
                source,
                ctx=ctx,
                request=http_request,
                route="chat_stream",
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                # 处理调试类型消息（新增）
                if chunk_type == "debug":
                    # 调试信息，可以选择发送或忽略
                    payload: JsonObject = _with_stream_trace(
                        {
                            "type": "debug",
                            "node": chunk.get("node", "unknown"),
                            "message_type": chunk.get("message_type", "unknown"),
                        },
                        ctx,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(payload, ensure_ascii=False),
                    }
                elif chunk_type == "tool_call":
                    # 发送工具调用事件（可选，前端可以显示工具调用状态）
                    payload = _with_stream_trace(
                        {"type": "tool_call", "data": chunk_data},
                        ctx,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(payload, ensure_ascii=False),
                    }
                elif chunk_type == "search_results":
                    # 发送检索结果（可选，前端可以忽略）
                    payload = _with_stream_trace(
                        {"type": "search_results", "data": chunk_data},
                        ctx,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(payload, ensure_ascii=False),
                    }
                elif chunk_type == "content":
                    if isinstance(chunk_data, str):
                        stream_answer_parts.append(chunk_data)
                    # 发送内容块 - 关键：data 必须是 JSON 字符串
                    payload = _with_stream_trace({"type": "content", "data": chunk_data}, ctx)
                    yield {
                        "event": "message",
                        "data": json.dumps(payload, ensure_ascii=False),
                    }
                elif chunk_type == "complete":
                    # 发送完成信号。fallback 场景需要把 fallback_used/reason_code 同时提升到
                    # done 顶层和 data 内部，兼容只读 done.data 的旧前端与按顶层字段判断的新前端。
                    payload = _with_stream_trace(_build_done_stream_payload(chunk_data), ctx)
                    answer_text = _extract_answer_text(payload)
                    if answer_text is None:
                        answer_text = "".join(stream_answer_parts).strip() or None
                    if answer_text is not None:
                        await _record_conversation_turn(
                            validated.session_id,
                            validated.question,
                            answer_text,
                            ctx,
                        )
                    yield {
                        "event": "message",
                        "data": json.dumps(payload, ensure_ascii=False),
                    }
                elif chunk_type == "fallback":
                    # Orchestrator 已经完成降级决策，API 层只把安全 payload 转成旧的
                    # event: message 形态；这里不重新构造 FallbackResult，避免重复决策导致
                    # fallback_used/reason_code 在流中前后不一致。
                    payload = _with_stream_trace(_build_fallback_stream_payload(chunk_data), ctx)
                    yield _sse_message(payload)
                elif chunk_type == "error":
                    # 旧 service 可能把原始异常字符串放进 chunk_data；这里统一转成
                    # 安全 envelope，避免 SSE 已建立后把内部异常全文推给前端。
                    if isinstance(chunk_data, AppError):
                        app_error = chunk_data
                    elif isinstance(chunk_data, Exception):
                        app_error = AppError.from_exception(
                            chunk_data,
                            origin_module="app.api.chat_stream",
                        )
                    else:
                        app_error = AppError.from_exception(
                            RuntimeError(str(chunk_data)),
                            origin_module="app.api.chat_stream",
                        )
                    fallback_result = _fallback_manager.decide(
                        app_error,
                        ctx,
                        scenario="chat_stream",
                    )
                    if fallback_result.fallback_used:
                        # SSE 已建立后不能改 HTTP status；可降级错误新增 type=fallback，
                        # 再用 done 暴露 fallback_used，保留旧前端按 data.type 消费的路径。
                        yield _fallback_manager.to_sse_event(fallback_result, ctx)
                        yield _fallback_manager.to_sse_done_event(
                            fallback_result,
                            ctx,
                            session_id=validated.session_id,
                        )
                        break
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            app_error.to_sse_payload(**_trace_kwargs(ctx)),
                            ensure_ascii=False,
                        ),
                    }
                    break

            logger.info(f"[会话 {validated.session_id}] 流式对话完成")

        except asyncio.CancelledError:
            _record_sse_disconnect(ctx, route="chat_stream", before_start=False)
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
            logger.error(f"流式对话接口错误: {app_error.code}")
            fallback_result = (
                _fallback_manager.decide(app_error, ctx, scenario="chat_stream")
                if can_use_fallback
                else None
            )
            if fallback_result is not None and fallback_result.fallback_used:
                # 外层异常多发生在下游迭代中断或 timeout；降级事件必须使用安全文案，
                # 不把 internal_message、堆栈或下游原始异常写入 SSE。
                yield _fallback_manager.to_sse_event(fallback_result, ctx)
                yield _fallback_manager.to_sse_done_event(
                    fallback_result,
                    ctx,
                    session_id=validated.session_id,
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


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest, http_request: Request):
    """清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    ctx = get_request_context_or_none(http_request)
    try:
        guarded = input_guard.validate_clear(request, ctx)
        validated = guarded.value
        ctx = _bind_session(http_request, validated.session_id)
        # Redis checkpoint 删除与 PG 历史清理都是同步阻塞 IO，放 worker 线程执行。
        rag_success = await asyncio.to_thread(
            rag_agent_service.clear_session, validated.session_id, ctx
        )
        store_success = await asyncio.to_thread(
            _clear_conversation_history, validated.session_id, ctx
        )
        success = bool(rag_success or store_success)
        logger.info(f"清空会话: {validated.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None,
        )

    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.chat.clear")
        logger.error(f"清空会话错误: {app_error.code}")
        return app_error.to_json_response(**_trace_kwargs(ctx))


@router.get("/chat/sessions")
async def list_chat_sessions(http_request: Request, limit: int = 100):
    """List durable chat sessions for the frontend history panel."""

    ctx = get_request_context_or_none(http_request)
    try:
        safe_limit = min(max(int(limit), 1), 200)
        sessions = await asyncio.to_thread(
            conversation_history_store.list_sessions, limit=safe_limit
        )
        return {
            "code": 200,
            "message": "success",
            "data": {"sessions": sessions, "count": len(sessions)},
            **_trace_kwargs(ctx),
        }
    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.chat.sessions")
        logger.error(f"获取会话列表错误: {app_error.code}")
        return app_error.to_json_response(**_trace_kwargs(ctx))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str, http_request: Request) -> SessionInfoResponse:
    """查询会话历史

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    ctx = get_request_context_or_none(http_request)
    try:
        validated_session_id = input_guard.validate_session_id(session_id)
        ctx = _bind_session(http_request, validated_session_id)
        # PG 历史读和 checkpoint 回退读都是同步阻塞 IO，放 worker 线程执行。
        history = await asyncio.to_thread(
            conversation_history_store.get_history, validated_session_id
        )
        if not history:
            history = await asyncio.to_thread(
                rag_agent_service.get_session_history, validated_session_id, ctx
            )

        return SessionInfoResponse(
            session_id=validated_session_id, message_count=len(history), history=history
        )

    except Exception as e:
        app_error = AppError.from_exception(e, origin_module="app.api.chat.session")
        logger.error(f"获取会话信息错误: {app_error.code}")
        return app_error.to_json_response(**_trace_kwargs(ctx))


def _bind_session(request: Request, session_id: str | None) -> RequestContext | None:
    """把已解析出的业务 session 写回 request.state.ctx。

    middleware 不读取 body，因此只能在 handler 拿到 Pydantic model 后补充 session。
    这保持了上传兼容性，也让 request.end trace 能记录同一个会话标识。
    """

    ctx = get_request_context_or_none(request)
    if ctx is None:
        return None
    updated_ctx = ctx.with_session(session_id)
    request.state.ctx = updated_ctx
    return updated_ctx


def _extract_answer_text(response_data: Mapping[str, object]) -> str | None:
    if response_data.get("success") is False:
        return None
    for key in ("answer", "response", "message", "data"):
        value = response_data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


async def _record_conversation_turn(
    session_id: str,
    question: str,
    answer: str,
    ctx: RequestContext | None,
) -> None:
    user_id = ctx.user_id if ctx is not None and ctx.user_id else "default"
    try:
        # PG 写入是同步阻塞 IO，放 worker 线程执行，避免阻塞事件循环。
        await asyncio.to_thread(
            conversation_history_store.record_turn,
            session_id,
            question,
            answer,
            user_id=user_id,
        )
    except Exception as exc:
        trace = _trace_kwargs(ctx)
        logger.warning(
            "Failed to record conversation history: "
            f"{exc.__class__.__name__}; trace_id={trace['trace_id']}"
        )
        # PG 历史写入失败 → 进补偿队列，由后台任务周期重试（防丢轮次）；
        # 队列有界且内存态，长期分歧由周期对账任务兜底修复。
        from app.memory.compensation import turn_compensator

        turn_compensator.enqueue(
            session_id,
            question,
            answer,
            user_id=user_id,
        )

    # 长期记忆画像：对话完成后异步抽取用户偏好/事实（fire-and-forget，
    # 失败静默，不影响已返回的响应）。必须在事件循环侧调度，让抽取跑在
    # 独立后台任务而不是当前 worker 线程。
    try:
        from app.memory.user_memory import schedule_ingest

        schedule_ingest(user_id, question, answer)
    except Exception:  # noqa: BLE001 - 画像抽取绝不阻断主链路
        pass


def _clear_conversation_history(session_id: str, ctx: RequestContext | None) -> bool:
    try:
        return conversation_history_store.clear_session(session_id)
    except Exception as exc:
        trace = _trace_kwargs(ctx)
        logger.warning(
            "Failed to clear conversation history: "
            f"{exc.__class__.__name__}; trace_id={trace['trace_id']}"
        )
        return False


def _trace_kwargs(ctx: RequestContext | None) -> dict[str, str | None]:
    if ctx is None:
        return {"trace_id": None, "request_id": None}
    return {"trace_id": ctx.trace_id, "request_id": ctx.request_id}


def _with_stream_trace(payload: JsonObject, ctx: RequestContext | None) -> JsonObject:
    """给旧 `event: message` payload 追加 trace 字段，不改变 `data.type` 兼容逻辑。"""

    if ctx is not None:
        payload["trace_id"] = ctx.trace_id
        payload["request_id"] = ctx.request_id
    return payload


def _request_timeout_seconds(ctx: RequestContext | None) -> float:
    """计算非流式请求的整体 timeout。

    有 RequestContext 时优先使用剩余 deadline，确保 middleware 创建的请求预算能贯穿
    handler 和 Agent；缺少 ctx 的单元测试或脚本调用回退到配置默认值。返回秒是为了
    直接传给 `asyncio.wait_for`。
    """

    timeout_ms = ctx.remaining_ms() if ctx is not None else config.request_timeout_ms
    return max(timeout_ms, 0) / 1000


def _build_stream_start_payload(session_id: str, ctx: RequestContext | None) -> JsonObject:
    """构建 Chat SSE start payload，同时保留旧 `event: message` 发送方式。"""

    return _with_stream_trace(
        {
            "type": "start",
            "session_id": session_id,
            "model": config.rag_model,
            "created_at": datetime.now(UTC).isoformat(),
        },
        ctx,
    )


def _sse_message(payload: JsonObject) -> dict[str, str]:
    """统一生成兼容旧前端的 SSE message 事件。"""

    return {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}


def _active_chat_orchestrator():
    """返回绑定当前 RAG service 的编排器。

    API 层仍保留 `rag_agent_service` 旧注入点；编排层每次使用前同步该引用，确保
    `orchestrator_enabled=true` 时不会绕过测试 fake 或运行时替换的旧 service。
    """

    agent_orchestrator.bind_services(rag_service=rag_agent_service)
    return agent_orchestrator


def _request_deadline_timeout_error(origin_module: str) -> AppError:
    """构造不触发 fallback 的请求级超时错误，避免对外暴露 asyncio 原始异常。"""

    return AppError(
        "LLM_TIMEOUT",
        fallback_required=False,
        internal_message="request deadline exceeded",
        origin_module=origin_module,
    )


def _build_fallback_stream_payload(chunk_data: object) -> JsonObject:
    """把编排层 fallback chunk 收窄成可安全序列化的 SSE payload。

    Orchestrator 只返回内部稳定字段，不直接生成 EventSourceResponse；这里保留 API adapter
    职责，同时刻意不对任意对象调用 `str(value)`，避免未来某个下游异常对象把密钥、内网 URL
    或原始异常全文混入 SSE。
    """

    payload: JsonObject = {"type": "fallback"}
    if isinstance(chunk_data, Mapping):
        for key, value in chunk_data.items():
            payload[str(key)] = _coerce_json_value(value)
    else:
        payload["data"] = _coerce_json_value(chunk_data)

    if "data" not in payload:
        partial_answer = payload.get("partial_answer")
        safe_message = payload.get("safe_message")
        payload["data"] = (
            partial_answer
            if isinstance(partial_answer, str)
            else safe_message
            if isinstance(safe_message, str)
            else None
        )
    if "message" not in payload and isinstance(payload.get("safe_message"), str):
        payload["message"] = payload["safe_message"]
    return payload


def _build_done_stream_payload(chunk_data: object) -> JsonObject:
    """构建 Chat SSE done payload，保留旧 data 字段并提升关键 fallback 元数据。"""

    payload: JsonObject = {"type": "done", "data": _coerce_json_value(chunk_data)}
    if isinstance(chunk_data, Mapping):
        for key in ("answer", "fallback_used", "reason_code", "citations", "message"):
            if key in chunk_data:
                payload[key] = _coerce_json_value(chunk_data[key])
    return payload


def _coerce_json_value(value: object) -> JsonValue:
    """将 fallback 附加字段限制为 JSON 值，未知对象只暴露类型名。"""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_json_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_coerce_json_value(item) for item in value]
    return value.__class__.__name__


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
                raise _request_deadline_timeout_error("app.api.chat_stream") from exc
    finally:
        await _close_async_iterator(iterator)


async def _close_async_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()


async def _is_client_disconnected(request: object) -> bool:
    """检测 SSE 客户端是否已断开。

    FastAPI Request 提供 `is_disconnected()`；测试 fake 也实现同名方法。若某些旧路径
    没有该方法，则按未断开处理，避免破坏直接调用 handler 的兼容测试。
    """

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
