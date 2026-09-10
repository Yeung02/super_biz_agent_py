"""Agent 薄编排层。

ISSUE-015 的边界是“串联已经存在的边界模块”，而不是重写 Agent、RAG 或 AIOps
业务算法。本模块只负责把 RequestContext、TokenBudget、ConversationManager、
FallbackManager 和 TraceLogger 放到同一个请求入口中，然后继续调用旧 service。
这样 API handler 可以保持薄层，出现兼容风险时也能通过 `orchestrator_enabled=false`
回到旧 service 直连路径。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from app.agent.intent import (
    BaseIntentClassifier,
    ChitchatResponder,
    FailoverIntentClassifier,
    IntentLabel,
    IntentResult,
    LlmIntentClassifier,
    NoopIntentClassifier,
    RuleIntentClassifier,
)
from app.config import config
from app.core.errors import AppError, InternalAppError, JsonObject, LLMTimeoutError
from app.core.fallback import FallbackManager, FallbackResult
from app.core.llm_usage import RawUsage
from app.core.request_context import RequestContext
from app.core.token_budget import TokenBudget, TokenBudgetManager, token_budget_manager
from app.observability.tracing import TraceLogger, TraceSpan


class RagServiceProtocol(Protocol):
    """编排层依赖的 RAG service 最小协议，方便测试注入 fake。"""

    model_name: str
    system_prompt: str

    async def query(self, question: str, session_id: str) -> str:
        """旧非流式 Chat 查询入口。"""

    def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncIterable[Mapping[str, object]]:
        """旧流式 Chat 查询入口。"""


class AIOpsServiceProtocol(Protocol):
    """编排层依赖的 AIOps service 最小协议，避免暴露 LangGraph 内部结构。"""

    def diagnose(self, session_id: str = "default") -> AsyncIterable[Mapping[str, object]]:
        """旧 AIOps SSE 诊断入口。"""


class ConversationManagerProtocol(Protocol):
    """ConversationManager 的最小门面协议。"""

    def load_context(
        self,
        session_id: str,
        budget: TokenBudget | int | None,
        ctx: RequestContext | None = None,
    ) -> object:
        """读取受控会话上下文；返回值由后续阶段决定，当前 issue 只做边界调用。"""


@dataclass(frozen=True)
class ChatRunResult:
    """非流式 Chat 编排结果。

    API 层只需要把该对象转成旧 `data.success/answer/errorMessage` 字段，并追加
    `fallback_used/reason_code/citations`。编排层不直接返回 FastAPI response，避免内部
    模块泄漏为外部 HTTP 契约。
    """

    answer: str
    fallback_used: bool = False
    reason_code: str | None = None
    citations: tuple[JsonObject, ...] = ()
    # 真实 LLM usage；None 表示底层未捕获（chitchat 短路、旧 service 等），记录时回退
    # 本地估算。仅用于内部 trace/metrics，不进入 to_api_data，避免外部 HTTP 契约变化。
    usage: RawUsage | None = None

    def to_api_data(self) -> JsonObject:
        """转换为 `/api/chat` 的 data 字段，旧字段只追加不删除。"""

        data: JsonObject = {
            "success": True,
            "answer": self.answer,
            "errorMessage": None,
            "fallback_used": self.fallback_used,
            "citations": [dict(citation) for citation in self.citations],
        }
        if self.reason_code is not None:
            data["reason_code"] = self.reason_code
        return data


class AgentOrchestrator:
    """RAG Chat 与 AIOps 的薄编排入口。

    这里刻意只做五类事：预算分配、意图分流、会话上下文门面调用、旧 service 调用、
    失败降级与 trace。RAG 排序、工具选择、AIOps 规划/执行、citation 构建都不在本类
    中实现，防止编排层变厚。意图分流只决定"调用哪个旧 service"，分类失败一律
    fail-open 回 RAG 路径，保证关闭开关等价于旧对话行为。
    """

    def __init__(
        self,
        *,
        rag_service: RagServiceProtocol,
        aiops_service: AIOpsServiceProtocol,
        fallback_manager: FallbackManager | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
        conversation_manager: ConversationManagerProtocol | None = None,
        trace_logger: TraceLogger | None = None,
        intent_classifier: BaseIntentClassifier | None = None,
        chitchat_responder: ChitchatResponder | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.rag_service = rag_service
        self.aiops_service = aiops_service
        self.intent_classifier = intent_classifier or NoopIntentClassifier()
        self.chitchat_responder = chitchat_responder
        self.fallback_manager = fallback_manager or FallbackManager()
        self.token_budget_manager = token_budget_manager or token_budget_manager_default()
        self.conversation_manager = conversation_manager or _conversation_manager_from_rag(
            rag_service
        )
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        self.enabled = config.orchestrator_enabled if enabled is None else enabled

    def bind_services(
        self,
        *,
        rag_service: RagServiceProtocol | None = None,
        aiops_service: AIOpsServiceProtocol | None = None,
    ) -> None:
        """同步 API 当前持有的旧 service 实例。

        ISSUE-015 的回滚与测试注入都发生在 API adapter 层；如果测试或热修复替换了
        `chat_api.rag_agent_service`，编排层不能继续握着旧单例，否则会绕过旧 service
        直连兼容路径。该方法只更新依赖引用，不改变编排算法。
        """

        if rag_service is not None and self.rag_service is not rag_service:
            self.rag_service = rag_service
            self.conversation_manager = _conversation_manager_from_rag(rag_service)
        if aiops_service is not None and self.aiops_service is not aiops_service:
            self.aiops_service = aiops_service

    async def run_chat(
        self,
        *,
        question: str,
        session_id: str,
        ctx: RequestContext | None,
    ) -> ChatRunResult:
        """运行非流式 Chat 编排。

        成功时仍调用旧 `rag_agent_service.query`；可降级 AppError 由 FallbackManager 转为
        安全文案；不可降级错误继续抛给 API adapter，保留 HTTP 错误语义。
        """

        if not self.enabled:
            answer = await self.rag_service.query(question, session_id)
            return ChatRunResult(answer=answer)

        span = self._start_span("chat", session_id, ctx)
        fallback_used = False
        try:
            conversation_context = await self._prepare_boundaries(
                mode="chat",
                session_id=session_id,
                ctx=ctx,
                current_input=question,
            )
            intent = await self._classify_intent(question, ctx)
            self._record_intent(intent, mode="chat", session_id=session_id, ctx=ctx)
            if intent.label is IntentLabel.AIOPS:
                # aiops 意图：在对话入口直接触发诊断流，报告聚合为 answer 返回，
                # API 契约保持 success/answer/errorMessage 字段不变。
                # usage 由 planner/executor/replanner/critic 节点级记录真实值，
                # 这里不再做聚合估算，避免 metrics 双计。
                result = await asyncio.wait_for(
                    self._collect_aiops_report(session_id=session_id),
                    timeout=_request_timeout_seconds(ctx),
                )
                return result
            if intent.label is IntentLabel.CHITCHAT:
                # chitchat 意图：单次 LLM 直答短路，跳过检索与 Agent 工具循环；
                # 直答失败时继续走下方 RAG 路径（fail-open）。
                chitchat_answer = await self._answer_chitchat(question)
                if chitchat_answer is not None:
                    result = ChatRunResult(answer=chitchat_answer)
                    self._record_usage(
                        question=question,
                        answer=chitchat_answer,
                        ctx=ctx,
                        mode="chat",
                    )
                    return result
            result = await asyncio.wait_for(
                self._query_rag_result(
                    question=question,
                    session_id=session_id,
                    ctx=ctx,
                    conversation_context=conversation_context,
                ),
                timeout=_request_timeout_seconds(ctx),
            )
            self._record_usage(
                question=question,
                answer=result.answer,
                ctx=ctx,
                mode="chat",
                usage=result.usage,
            )
            return result
        except TimeoutError as exc:
            app_error = _request_deadline_timeout_error(
                origin_module="app.agent.orchestrator"
            )
            self._record_error(app_error, mode="chat", session_id=session_id, ctx=ctx)
            raise app_error from exc
        except Exception as exc:
            app_error = _normalize_orchestrator_error(exc, origin_module="app.agent.orchestrator")
            self._record_error(app_error, mode="chat", session_id=session_id, ctx=ctx)
            fallback = self.fallback_manager.for_chat(app_error, ctx)
            fallback_used = fallback.fallback_used
            if fallback.fallback_used:
                return _chat_result_from_fallback(fallback)
            raise app_error from exc
        finally:
            self._end_span(span, mode="chat", session_id=session_id, fallback_used=fallback_used)

    async def run_chat_stream(
        self,
        *,
        question: str,
        session_id: str,
        ctx: RequestContext | None,
    ) -> AsyncIterator[dict[str, object]]:
        """运行流式 Chat 编排，输出仍保持旧 service chunk 形态。

        API 层继续负责把 chunk 转为 `event: message` SSE，以保留旧前端解析路径。编排层只在
        下游失败时插入 `fallback` 与 `complete` chunk。
        """

        if not self.enabled:
            async for chunk in self.rag_service.query_stream(question, session_id):
                yield dict(chunk)
            return

        span = self._start_span("chat_stream", session_id, ctx)
        fallback_used = False
        collected_answer: list[str] = []
        try:
            conversation_context = await self._prepare_boundaries(
                mode="chat_stream",
                session_id=session_id,
                ctx=ctx,
                current_input=question,
            )
            intent = await self._classify_intent(question, ctx)
            self._record_intent(intent, mode="chat_stream", session_id=session_id, ctx=ctx)
            chitchat_answer = (
                await self._answer_chitchat(question)
                if intent.label is IntentLabel.CHITCHAT
                else None
            )
            if intent.label is IntentLabel.AIOPS:
                chunk_source: AsyncIterator[Mapping[str, object]] = (
                    self._stream_aiops_as_chat(session_id=session_id)
                )
                # aiops 流式的 usage 由各节点记录真实值，聚合估算跳过，避免双计。
                record_stream_usage = False
            elif chitchat_answer is not None:
                chunk_source = _single_answer_chunks(chitchat_answer)
                record_stream_usage = True
            else:
                chunk_source = self._stream_rag(
                    question=question,
                    session_id=session_id,
                    ctx=ctx,
                    conversation_context=conversation_context,
                )
                record_stream_usage = True
            async for raw_chunk in chunk_source:
                chunk = dict(raw_chunk)
                if chunk.get("type") == "content" and isinstance(chunk.get("data"), str):
                    collected_answer.append(cast(str, chunk["data"]))
                if chunk.get("type") == "error":
                    app_error = _app_error_from_stream_chunk(
                        chunk,
                        origin_module="app.agent.orchestrator.chat_stream",
                    )
                    self._record_error(
                        app_error,
                        mode="chat_stream",
                        session_id=session_id,
                        ctx=ctx,
                    )
                    fallback = self.fallback_manager.decide(
                        app_error,
                        ctx,
                        scenario="chat_stream",
                    )
                    fallback_used = fallback.fallback_used
                    if fallback.fallback_used:
                        yield _chat_fallback_chunk(fallback)
                        yield _chat_fallback_complete_chunk(fallback)
                    else:
                        yield {"type": "error", "data": app_error}
                    return
                yield chunk
            if collected_answer and record_stream_usage:
                self._record_usage(
                    question=question,
                    answer="".join(collected_answer),
                    ctx=ctx,
                    mode="chat_stream",
                )
        except Exception as exc:
            app_error = _normalize_stream_error(
                exc,
                origin_module="app.agent.orchestrator.chat_stream",
            )
            self._record_error(app_error, mode="chat_stream", session_id=session_id, ctx=ctx)
            fallback = self.fallback_manager.decide(app_error, ctx, scenario="chat_stream")
            fallback_used = fallback.fallback_used
            if fallback.fallback_used:
                yield _chat_fallback_chunk(fallback)
                yield _chat_fallback_complete_chunk(fallback)
            else:
                yield {"type": "error", "data": app_error}
        finally:
            self._end_span(
                span,
                mode="chat_stream",
                session_id=session_id,
                fallback_used=fallback_used,
            )

    async def run_aiops(
        self,
        *,
        session_id: str,
        ctx: RequestContext | None,
    ) -> AsyncIterator[dict[str, object]]:
        """运行 AIOps 编排，输出仍保持旧 AIOps event payload 形态。"""

        if not self.enabled:
            async for event in self.aiops_service.diagnose(session_id=session_id):
                yield dict(event)
            return

        span = self._start_span("aiops", session_id, ctx)
        fallback_used = False
        try:
            await self._prepare_boundaries(
                mode="aiops",
                session_id=session_id,
                ctx=ctx,
                current_input=None,
            )
            async for raw_event in self.aiops_service.diagnose(session_id=session_id):
                event = dict(raw_event)
                if event.get("type") == "error":
                    app_error = _app_error_from_stream_chunk(
                        event,
                        origin_module="app.agent.orchestrator.aiops",
                    )
                    self._record_error(app_error, mode="aiops", session_id=session_id, ctx=ctx)
                    fallback = self.fallback_manager.for_aiops(app_error, ctx)
                    fallback_used = fallback.fallback_used
                    if fallback.fallback_used:
                        yield _event_fallback_payload(fallback, ctx)
                        yield _event_fallback_done_payload(fallback, ctx, session_id=session_id)
                    else:
                        yield _event_error_payload(app_error, ctx)
                    return
                yield _attach_trace(dict(event), ctx)
        except Exception as exc:
            app_error = _normalize_stream_error(
                exc,
                origin_module="app.agent.orchestrator.aiops",
            )
            self._record_error(app_error, mode="aiops", session_id=session_id, ctx=ctx)
            fallback = self.fallback_manager.for_aiops(app_error, ctx)
            fallback_used = fallback.fallback_used
            if fallback.fallback_used:
                yield _event_fallback_payload(fallback, ctx)
                yield _event_fallback_done_payload(fallback, ctx, session_id=session_id)
            else:
                yield _event_error_payload(app_error, ctx)
        finally:
            self._end_span(span, mode="aiops", session_id=session_id, fallback_used=fallback_used)

    async def _prepare_boundaries(
        self,
        *,
        mode: str,
        session_id: str,
        ctx: RequestContext | None,
        current_input: str | None,
    ) -> object | None:
        """分配 token 预算并触发 ConversationManager 门面。

        这里不把历史拼进旧 service prompt，因为当前 issue 只接薄编排层；真正的上下文注入和
        RAG pipeline 由后续阶段负责。现在先记录预算和门面调用，保证边界能力有统一入口。
        load_context 内部是同步阻塞 IO（Redis checkpoint 读 + PG 摘要读写），
        放到 worker 线程执行，避免阻塞事件循环拖慢并发请求。
        """

        scenario = "rag_chat" if mode in {"chat", "chat_stream"} else "aiops_plan"
        budget = self.token_budget_manager.allocate(
            scenario,
            _model_name(self.rag_service),
            ctx,
            current_input=current_input,
            system_prompt=_system_prompt(self.rag_service) if current_input is not None else None,
        )
        if self.conversation_manager is not None:
            context = await asyncio.to_thread(
                self.conversation_manager.load_context,
                session_id,
                budget,
                ctx,
            )
            return await self._augment_with_user_memories(context, current_input, ctx)
        return None

    async def _augment_with_user_memories(
        self,
        context: object | None,
        current_input: str | None,
        ctx: RequestContext | None,
    ) -> object | None:
        """把用户长期记忆召回结果注入会话上下文；失败 fail-open 返回原上下文。"""

        if context is None or not current_input:
            return context
        from app.config import config as _config

        if not _config.user_memory_enabled:
            return context

        from app.memory.user_memory import user_memory_service

        try:
            user_id = (ctx.user_id if ctx is not None and ctx.user_id else "default")
            # Milvus 向量召回是同步阻塞 IO，放 worker 线程执行。
            memories = await asyncio.to_thread(user_memory_service.recall, user_id, current_input)
        except Exception:  # noqa: BLE001 - 记忆召回不允许阻断对话
            return context
        if not memories:
            return context
        try:
            from dataclasses import replace

            augmented = replace(context, user_memories=tuple(memories))
        except TypeError:
            return context
        if ctx is not None:
            self.trace_logger.record_event(
                "user_memory.recall",
                ctx,
                status="ok",
                memory_count=len(memories),
            )
        return augmented

    async def _query_rag(
        self,
        *,
        question: str,
        session_id: str,
        ctx: RequestContext | None,
        conversation_context: object | None,
    ) -> str:
        """Use context-aware RAG when present while preserving the legacy service API."""

        query_with_context = getattr(self.rag_service, "query_with_context", None)
        if callable(query_with_context):
            return cast(
                str,
                await query_with_context(question, session_id, conversation_context, ctx),
            )
        return await self.rag_service.query(question, session_id)

    async def _query_rag_result(
        self,
        *,
        question: str,
        session_id: str,
        ctx: RequestContext | None,
        conversation_context: object | None,
    ) -> ChatRunResult:
        """Use Stage 3B answer+citations when the service exposes it."""

        query_with_citations = getattr(self.rag_service, "query_with_citations", None)
        if callable(query_with_citations):
            result = await query_with_citations(
                question,
                session_id,
                conversation_context,
                ctx,
            )
            return _chat_result_from_rag_result(result)

        answer = await self._query_rag(
            question=question,
            session_id=session_id,
            ctx=ctx,
            conversation_context=conversation_context,
        )
        return ChatRunResult(answer=answer)

    async def _stream_rag(
        self,
        *,
        question: str,
        session_id: str,
        ctx: RequestContext | None,
        conversation_context: object | None,
    ) -> AsyncIterator[Mapping[str, object]]:
        """Use context-aware streaming when present while preserving legacy chunks."""

        stream_with_context = getattr(self.rag_service, "query_stream_with_context", None)
        if callable(stream_with_context):
            async for chunk in stream_with_context(
                question,
                session_id,
                conversation_context,
                ctx,
            ):
                yield cast(Mapping[str, object], chunk)
            return

        async for chunk in self.rag_service.query_stream(question, session_id):
            yield chunk

    async def _classify_intent(
        self,
        question: str,
        ctx: RequestContext | None,
    ) -> IntentResult:
        """意图分类；超时或异常一律 fail-open 回 rag_qa，不阻断对话链路。

        分类器是同步 LLM 调用（与 query rewriter 同形态），用 to_thread 避免阻塞
        事件循环；wait_for 只放弃等待，线程内的调用由分类器自身的 LLM 超时兜底。
        """

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.intent_classifier.classify,
                    question=question,
                    ctx=ctx,
                ),
                timeout=config.intent_timeout_seconds,
            )
        except Exception:
            return IntentResult(error_code="INTENT_CLASSIFY_FAILED")

    async def _answer_chitchat(self, question: str) -> str | None:
        """闲聊直答；响应器缺失或失败返回 None，由调用方落回 RAG 路径。"""

        if self.chitchat_responder is None:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.chitchat_responder.answer, question),
                timeout=_request_timeout_seconds(None),
            )
        except Exception:
            return None

    async def _collect_aiops_report(
        self,
        *,
        session_id: str,
    ) -> ChatRunResult:
        """把 AIOps 诊断事件流聚合为 chat answer（aiops 意图的非流式分支）。

        优先返回最终报告全文；没有报告时返回进度摘要，避免空 answer。诊断流中的
        error 事件恢复为稳定 AppError 抛出，交给 run_chat 既有的 fallback 契约。
        """

        progress: list[str] = []
        final_report = ""
        async for raw_event in self.aiops_service.diagnose(session_id=session_id):
            event = dict(raw_event)
            if event.get("type") == "error":
                raise _app_error_from_stream_chunk(
                    event,
                    origin_module="app.agent.orchestrator.chat_aiops",
                )
            text = _aiops_event_text(event)
            if text:
                progress.append(text)
            if event.get("type") == "report":
                report = event.get("report")
                if isinstance(report, str) and report:
                    final_report = report
            elif event.get("type") == "complete":
                report = _aiops_complete_report(event)
                if report:
                    final_report = final_report or report
        if final_report:
            return ChatRunResult(answer=final_report)
        if progress:
            return ChatRunResult(answer="\n".join(progress))
        return ChatRunResult(answer="诊断流程已完成，但未生成最终报告。")

    async def _stream_aiops_as_chat(
        self,
        *,
        session_id: str,
    ) -> AsyncIterator[Mapping[str, object]]:
        """把 AIOps 诊断事件翻译为旧 chat stream chunk（aiops 意图的流式分支）。

        进度/报告事件映射为 content 分片，complete 映射为 chat complete；error 事件
        原样透传，由上层 chat_stream 错误处理统一走 fallback/error 契约，保持
        `event: message` + `data.type` 的旧前端解析方式。
        """

        async for raw_event in self.aiops_service.diagnose(session_id=session_id):
            event = dict(raw_event)
            event_type = event.get("type")
            if event_type == "complete":
                yield {"type": "complete"}
                return
            if event_type == "error":
                yield event
                return
            text = _aiops_event_text(event)
            if text:
                yield {"type": "content", "data": text}
        # 事件流自然结束但没有 complete 事件时补一个，保持 chat 流契约完整。
        yield {"type": "complete"}

    def _record_intent(
        self,
        intent: IntentResult,
        *,
        mode: str,
        session_id: str,
        ctx: RequestContext | None,
    ) -> None:
        """记录意图分流结果，供流量构成与分类失败率观测使用。"""

        if ctx is None:
            return
        self.trace_logger.record_event(
            "orchestrator.intent",
            ctx,
            mode=mode,
            session_id=session_id,
            intent=intent.label.value,
            confidence=intent.confidence,
            source=intent.source,
            error_code=intent.error_code,
        )

    def _record_usage(
        self,
        *,
        question: str,
        answer: str,
        ctx: RequestContext | None,
        mode: str,
        usage: RawUsage | None = None,
    ) -> None:
        """记录 usage：真实 usage 优先，底层未捕获时回退本地估算。"""

        if ctx is None:
            return
        if usage is not None:
            # 真实 usage 覆盖 system prompt、RAG 上下文和工具结果的完整输入，
            # estimated=False 让成本指标能区分供应商数据和本地估算。
            usage_record = self.token_budget_manager.record_usage(
                model=_model_name(self.rag_service),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                ctx=ctx,
                estimated=False,
            )
        else:
            input_tokens = self.token_budget_manager.estimate_tokens(question).token_count
            output_tokens = self.token_budget_manager.estimate_tokens(answer).token_count
            usage_record = self.token_budget_manager.record_usage(
                model=_model_name(self.rag_service),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ctx=ctx,
                estimated=True,
            )
        self.trace_logger.record_event(
            "orchestrator.usage",
            ctx,
            mode=mode,
            session_id=ctx.session_id,
            usage=usage_record.to_dict(),
        )

    def _start_span(
        self,
        mode: str,
        session_id: str,
        ctx: RequestContext | None,
    ) -> TraceSpan | None:
        if ctx is None:
            return None
        return self.trace_logger.start_span(
            "orchestrator",
            ctx,
            mode=mode,
            session_id=session_id,
            fallback_used=False,
        )

    def _end_span(
        self,
        span: TraceSpan | None,
        *,
        mode: str,
        session_id: str,
        fallback_used: bool,
    ) -> None:
        if span is None:
            return
        self.trace_logger.end_span(
            span,
            mode=mode,
            session_id=session_id,
            fallback_used=fallback_used,
        )

    def _record_error(
        self,
        error: AppError,
        *,
        mode: str,
        session_id: str,
        ctx: RequestContext | None,
    ) -> None:
        """记录编排层错误，只写稳定错误码，不写原始异常全文。"""

        if ctx is None:
            return
        self.trace_logger.record_event(
            "orchestrator.error",
            ctx,
            status="error",
            mode=mode,
            session_id=session_id,
            error_code=error.code,
            fallback_used=error.requires_fallback(),
        )


def token_budget_manager_default() -> TokenBudgetManager:
    """返回全局 TokenBudgetManager，单独封装便于测试注入替换。"""

    return token_budget_manager


def _conversation_manager_from_rag(
    rag_service: RagServiceProtocol,
) -> ConversationManagerProtocol | None:
    manager = getattr(rag_service, "conversation_manager", None)
    load_context = getattr(manager, "load_context", None)
    if callable(load_context):
        return cast(ConversationManagerProtocol, manager)
    return None


def _default_intent_classifier() -> BaseIntentClassifier:
    """生产装配：开关关闭时使用 Noop，保持未启用意图识别的旧对话行为。

    直接 new AgentOrchestrator 的调用方（单测/evaluation）默认拿到 Noop，不会在
    测试环境误触 DashScope；只有生产懒加载单例按配置装配真实分类器。
    """

    if not config.intent_enabled:
        return NoopIntentClassifier()
    classifier: BaseIntentClassifier = LlmIntentClassifier()
    if config.intent_rule_fallback_enabled:
        classifier = FailoverIntentClassifier(
            primary=classifier,
            secondary=RuleIntentClassifier(),
        )
    return classifier


def _default_chitchat_responder() -> ChitchatResponder | None:
    """生产装配：闲聊直答器与意图开关同生命周期，关闭时闲聊意图自然落回 RAG。"""

    if not config.intent_enabled:
        return None
    return ChitchatResponder()


def _model_name(rag_service: RagServiceProtocol) -> str:
    model_name = getattr(rag_service, "model_name", None)
    return model_name if isinstance(model_name, str) and model_name else config.rag_model


def _system_prompt(rag_service: RagServiceProtocol) -> str:
    prompt = getattr(rag_service, "system_prompt", None)
    return prompt if isinstance(prompt, str) else ""


def _request_timeout_seconds(ctx: RequestContext | None) -> float:
    timeout_ms = ctx.remaining_ms() if ctx is not None else int(config.request_timeout_ms)
    return max(timeout_ms, 0) / 1000


def _normalize_orchestrator_error(exc: Exception, *, origin_module: str) -> AppError:
    """把编排层捕获的异常归一化为 AppError，未知异常固定为 INTERNAL_ERROR。"""

    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError()
    return InternalAppError(
        internal_message=f"{exc.__class__.__name__}",
        origin_module=origin_module,
    )


def _normalize_stream_error(exc: Exception, *, origin_module: str) -> AppError:
    """把流式迭代中断映射为对外 SSE 契约错误。

    ISSUE-015 的内部规则要求未知异常不能原样外泄；`api_contract.md` 同时要求 SSE
    中途失败对外使用 `SSE_STREAM_INTERRUPTED`。这里按外部契约优先，只暴露稳定错误码，
    并关闭 fallback，避免没有部分证据时把纯中断伪装成可恢复回答。
    """

    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError()
    return _stream_interrupted_error(origin_module=origin_module)


def _request_deadline_timeout_error(*, origin_module: str) -> AppError:
    """构造请求级 deadline 超时错误。

    该错误仍使用 `LLM_TIMEOUT` 对外 code，兼容旧前端和错误码表；但它来自 API/request
    总时限而不是模型主动返回的可降级失败，所以不触发 fallback，避免超时后再包装成成功响应。
    """

    return AppError(
        "LLM_TIMEOUT",
        fallback_required=False,
        internal_message="request deadline exceeded",
        origin_module=origin_module,
    )


def _stream_interrupted_error(*, origin_module: str) -> AppError:
    """构造不含原始异常全文的 SSE 中断错误。"""

    return AppError(
        "SSE_STREAM_INTERRUPTED",
        fallback_required=False,
        internal_message="stream interrupted by downstream exception",
        origin_module=origin_module,
    )


def _app_error_from_stream_chunk(
    chunk: Mapping[str, object],
    *,
    origin_module: str,
) -> AppError:
    """从旧 service 的 error chunk/event 中恢复稳定错误码。

    旧 service 可能把原始异常文字塞进 `data/message`；这里只信任标准 `error.code`，
    其他情况统一映射为 INTERNAL_ERROR，避免把内部 URL、密钥或堆栈原样传给用户。
    """

    data = chunk.get("data")
    if isinstance(data, AppError):
        return data
    if isinstance(data, Exception):
        return _normalize_stream_error(data, origin_module=origin_module)

    error = chunk.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        if isinstance(code, str) and code:
            return AppError(code)
    return _stream_interrupted_error(origin_module=origin_module)


def _aiops_event_text(event: Mapping[str, object]) -> str:
    """把 AIOps 进度事件转换为用户可读文本；无法转换的事件返回空串。"""

    event_type = event.get("type")
    message = event.get("message")
    message_text = message if isinstance(message, str) and message else ""
    if event_type == "plan":
        plan = event.get("plan")
        if isinstance(plan, list) and plan:
            steps = "\n".join(f"- {step}" for step in plan if isinstance(step, str))
            if steps:
                return f"诊断计划（共 {len(plan)} 步）：\n{steps}"
        return message_text
    if event_type == "step_complete":
        step = event.get("current_step")
        if isinstance(step, str) and step:
            return f"{message_text}：{step}" if message_text else step
        return message_text
    if event_type == "report":
        report = event.get("report")
        return report if isinstance(report, str) and report else message_text
    if event_type == "status":
        return message_text
    return ""


def _aiops_complete_report(event: Mapping[str, object]) -> str:
    """从 complete 事件中提取报告文本，兼容 diagnosis.report 与 response 两种形态。"""

    diagnosis = event.get("diagnosis")
    if isinstance(diagnosis, Mapping):
        report = diagnosis.get("report")
        if isinstance(report, str) and report:
            return report
    response = event.get("response")
    return response if isinstance(response, str) else ""


async def _single_answer_chunks(answer: str) -> AsyncIterator[Mapping[str, object]]:
    """把一次性回答包装为旧 chat stream 的 content+complete chunk 序列。"""

    yield {"type": "content", "data": answer}
    yield {"type": "complete"}


def _chat_result_from_fallback(result: FallbackResult) -> ChatRunResult:
    answer = result.partial_answer or result.safe_message
    return ChatRunResult(
        answer=answer,
        fallback_used=True,
        reason_code=result.reason_code,
        citations=result.citations,
    )


def _chat_result_from_rag_result(result: object) -> ChatRunResult:
    if isinstance(result, ChatRunResult):
        return result

    answer = getattr(result, "answer", "")
    citations = getattr(result, "citations", ())
    usage = getattr(result, "usage", None)
    return ChatRunResult(
        answer=answer if isinstance(answer, str) else str(answer),
        citations=_coerce_citations(citations),
        usage=usage if isinstance(usage, RawUsage) else None,
    )


def _coerce_citations(value: object) -> tuple[JsonObject, ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    citations: list[JsonObject] = []
    for item in value:
        if isinstance(item, Mapping):
            citations.append(cast(JsonObject, dict(item)))
    return tuple(citations)


def _chat_fallback_chunk(result: FallbackResult) -> dict[str, object]:
    return {
        "type": "fallback",
        "data": _fallback_data(result),
    }


def _chat_fallback_complete_chunk(result: FallbackResult) -> dict[str, object]:
    data = _fallback_data(result)
    data["answer"] = result.partial_answer or result.safe_message
    return {
        "type": "complete",
        "data": data,
    }


def _event_fallback_payload(
    result: FallbackResult,
    ctx: RequestContext | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "fallback",
        "stage": "fallback",
        "message": result.safe_message,
        "data": result.partial_answer or result.safe_message,
        **_fallback_data(result),
    }
    return _attach_trace(payload, ctx)


def _event_fallback_done_payload(
    result: FallbackResult,
    ctx: RequestContext | None,
    *,
    session_id: str,
) -> dict[str, object]:
    answer = result.partial_answer or result.safe_message
    payload: dict[str, object] = {
        "type": "done",
        "stage": "fallback_done",
        "message": result.safe_message,
        "session_id": session_id,
        "answer": answer,
        "data": {
            "answer": answer,
            "fallback_used": result.fallback_used,
            "reason_code": result.reason_code,
            "citations": [dict(citation) for citation in result.citations],
        },
        **_fallback_data(result),
    }
    return _attach_trace(payload, ctx)


def _event_error_payload(error: AppError, ctx: RequestContext | None) -> dict[str, object]:
    trace_kwargs = (
        {"trace_id": ctx.trace_id, "request_id": ctx.request_id}
        if ctx is not None
        else {"trace_id": None, "request_id": None}
    )
    return error.to_sse_payload(**trace_kwargs)


def _fallback_data(result: FallbackResult) -> dict[str, object]:
    return {
        "fallback_used": result.fallback_used,
        "reason_code": result.reason_code,
        "safe_message": result.safe_message,
        "partial_answer": result.partial_answer,
        "should_continue_stream": result.should_continue_stream,
        "citations": [dict(citation) for citation in result.citations],
    }


def _attach_trace(payload: dict[str, object], ctx: RequestContext | None) -> dict[str, object]:
    if ctx is not None:
        payload["trace_id"] = ctx.trace_id
        payload["request_id"] = ctx.request_id
    return payload


# 全局编排器延续旧 service 单例，API handler 通过配置开关决定是否使用它。
class _LazyAgentOrchestrator:
    """Delay production service wiring until the orchestrator is actually used."""

    def __init__(self) -> None:
        self._instance: AgentOrchestrator | None = None

    def _get(self) -> AgentOrchestrator:
        if self._instance is None:
            from app.services.aiops_service import aiops_service
            from app.services.rag_agent_service import rag_agent_service

            self._instance = AgentOrchestrator(
                rag_service=rag_agent_service,
                aiops_service=aiops_service,
                intent_classifier=_default_intent_classifier(),
                chitchat_responder=_default_chitchat_responder(),
            )
        return self._instance

    def bind_services(
        self,
        *,
        rag_service: RagServiceProtocol | None = None,
        aiops_service: AIOpsServiceProtocol | None = None,
    ) -> None:
        self._get().bind_services(rag_service=rag_service, aiops_service=aiops_service)

    async def run_chat(self, **kwargs: object) -> ChatRunResult:
        return await self._get().run_chat(**kwargs)  # type: ignore[arg-type]

    def run_chat_stream(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        return self._get().run_chat_stream(**kwargs)  # type: ignore[arg-type]

    def run_aiops(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        return self._get().run_aiops(**kwargs)  # type: ignore[arg-type]


# Keep the historical import name while avoiding LangChain/LangGraph import side effects
# during collection, rollback paths, and fake-only tests.
agent_orchestrator = _LazyAgentOrchestrator()
