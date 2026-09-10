"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import inspect
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from loguru import logger
from typing_extensions import TypedDict

try:
    from langchain.agents import create_agent
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        RemoveMessage,
        SystemMessage,
    )
    from langchain_qwq import ChatQwen
    from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
except ModuleNotFoundError as exc:
    _MISSING_AGENT_DEPENDENCY: ModuleNotFoundError | None = exc

    class BaseMessage:
        def __init__(self, content: str = "", **kwargs: object) -> None:
            self.content = content
            self.id = kwargs.get("id")

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class RemoveMessage(BaseMessage):
        pass

    class MemorySaver:
        def delete_thread(self, thread_id: str) -> None:
            _ = thread_id

    REMOVE_ALL_MESSAGES = "__remove_all_messages__"

    def add_messages(left: Sequence[object], right: Sequence[object]) -> tuple[object, ...]:
        return (*left, *right)

    def create_agent(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        raise RuntimeError("LangChain dependencies are required to create the RAG agent")

    class ChatQwen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs
            raise RuntimeError("langchain_qwq is required to create the RAG model")

else:
    _MISSING_AGENT_DEPENDENCY = None

from app.config import config
from app.core.errors import JsonObject, RagEmptyResultError
from app.core.llm_usage import RawUsage, UsageAccumulator, extract_message_usage
from app.core.request_context import RequestContext
from app.core.token_budget import token_budget_manager
from app.memory.conversation_manager import ConversationManager
from app.observability.tracing import TraceLogger
from app.rag.citation import CitationBuilder
from app.rag.context_builder import ContextBuilder
from app.rag.models import RagContext
from app.rag.query_rewriter import LlmQueryRewriter
from app.rag.reranker import DashScopeReranker
from app.rag.retriever import RagRetriever, RetrievalResult
try:
    from app.agent.tool_manager import ToolManager, get_tool_manager
    from app.agent.tool_selector import ToolSelector
except ModuleNotFoundError:
    class ToolManager:
        def wrap_local_tool(self, tool: object) -> object:
            return tool

        def wrap_mcp_tool(self, tool: object) -> object:
            return tool

        def to_langchain_tool(self, spec: object) -> object:
            return spec

    def get_tool_manager() -> ToolManager:
        return ToolManager()

    class ToolSelector:
        def select(self, query: str, tools: list[object]) -> list[object]:
            return tools

try:
    from app.agent.mcp_client import get_mcp_tools_with_retry
except ModuleNotFoundError:
    async def get_mcp_tools_with_retry() -> list[object]:
        raise RuntimeError("MCP client dependencies are unavailable")

try:
    from app.tools import get_current_time, retrieve_knowledge
except ModuleNotFoundError:
    class _UnavailableTool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = "unavailable test placeholder"
            self.args_schema = None

        async def ainvoke(self, args: dict[str, object]) -> str:
            _ = args
            raise RuntimeError(f"{self.name} is unavailable")

        def __call__(self, *args: object, **kwargs: object) -> str:
            _ = args, kwargs
            raise RuntimeError(f"{self.name} is unavailable")

    retrieve_knowledge = _UnavailableTool("retrieve_knowledge")
    get_current_time = _UnavailableTool("get_current_time")

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@dataclass(frozen=True)
class RagAnswerResult:
    """Internal Stage 3B Chat result used before API response adaptation."""

    answer: str
    citations: tuple[JsonObject, ...] = ()
    # 真实 LLM usage；None 表示底层未捕获（旧路径或供应商未返回），调用方回退本地估算。
    usage: RawUsage | None = None
    # 模型实际看到的 RAG 证据上下文；仅供离线 evaluation judge 等内部消费方使用，
    # 不进入 API 响应。旧调用方不感知该字段。
    context_text: str = ""


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
            # streaming=True 时 ainvoke 内部走流式，DashScope 默认不在流尾返回 usage；
            # 显式开启后最终消息才带 usage_metadata，供真实 usage 采集使用。
            stream_usage=True,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]
        # ISSUE-008 只接入工具边界，不接管 Agent 编排。关闭 tool_manager_enabled 时，
        # 初始化阶段会直接使用旧的 `self.tools + self.mcp_tools` 原始列表。
        # 使用进程级共享 ToolManager：与 AIOps Executor 复用同一注册表和请求级
        # 工具调用预算，避免每个 service 实例重复注册和包装。
        self.tool_manager = get_tool_manager()
        # 按问题筛选工具子集：name -> 已包装工具，以及按工具集缓存的 Agent 变体。
        self.tool_selector = ToolSelector()
        self._wrapped_tool_order: list[str] = []
        self._wrapped_tools_by_name: dict[str, object] = {}
        self._agent_cache: dict[frozenset[str], object] = {}

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 记忆存储底座：生产为 Redis checkpointer（多副本共享、AOF 持久化），
        # 测试进程注入 memory 后端。ConversationManager 门面不感知具体实现。
        from app.memory.checkpointer_factory import create_checkpointer

        self.checkpointer = create_checkpointer()
        # ISSUE-013: checkpointer 继续作为 LangGraph checkpoint 存储；业务历史读取、过滤、
        # 裁剪和清理统一走 ConversationManager。这样 API/Agent 不需要理解 checkpoint
        # tuple/dict/namedtuple 的不稳定结构，关闭开关时仍可回滚到旧方法。
        # summary_store 注入 PG 摘要持久层后，摘要增量生成并落库，不再每请求现算。
        from app.memory.conversation_store import conversation_history_store

        self.conversation_manager = ConversationManager(
            self.checkpointer,
            token_budget_manager=token_budget_manager,
            summary_store=conversation_history_store,
        )
        self.token_budget_manager = token_budget_manager
        self.trace_logger = TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        # 阶段 3C：检索入口接入 LLM 查询改写与多路召回；改写失败由 retriever 内部
        # fail-open 回退原始 query，不影响主问答链路。qwen3-rerank 重排在 candidate
        # 过滤后、final_k 截断前生效，失败同样 fail-open 回退向量排序。
        self.rag_retriever = RagRetriever(
            trace_logger=self.trace_logger,
            query_rewriter=LlmQueryRewriter(),
            reranker=DashScopeReranker(),
        )
        self.context_builder = ContextBuilder(
            token_budget_manager=self.token_budget_manager,
            trace_logger=self.trace_logger,
        )
        self.citation_builder = CitationBuilder(trace_logger=self.trace_logger)

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}"
        )

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）加载原始 MCP 工具。
        mcp_tools = await get_mcp_tools_with_retry()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具。默认经过 ToolManager wrapper；关闭开关时保留旧工具列表，
        # 作为 ISSUE-008 的独立回滚路径，避免 ToolNode schema 兼容问题影响现有 demo。
        all_tools = self._build_agent_tools(mcp_tools)

        # 记录已包装工具映射，供按问题筛选工具子集时复用（raw 回滚模式不筛选）。
        self._wrapped_tool_order = [
            tool.name for tool in all_tools if hasattr(tool, "name")
        ]
        self._wrapped_tools_by_name = {
            tool.name: tool for tool in all_tools if hasattr(tool, "name")
        }

        self.agent = _create_agent_with_trim(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )
        self._agent_cache[frozenset(self._wrapped_tool_order)] = self.agent

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _agent_for_question(self, question: str):
        """按问题筛选工具子集，返回绑定该子集的 Agent（按工具集缓存）。

        - 关闭 tool_selection_enabled、工具未包装（raw 回滚模式）或筛选结果为
          全量/空时，直接返回默认 Agent，行为与不筛选完全一致；
        - LangGraph Agent 的工具在创建时绑定，因此每个不同子集构建一次并缓存，
          共享同一个 model 和 checkpointer，会话状态不受影响。
        """

        if not config.tool_selection_enabled or not self._wrapped_tools_by_name:
            return self.agent

        selected = self.tool_selector.select(
            question or "",
            [self._wrapped_tools_by_name[name] for name in self._wrapped_tool_order],
        )
        selected_ids = {id(tool) for tool in selected}
        names = frozenset(
            name
            for name in self._wrapped_tool_order
            if id(self._wrapped_tools_by_name[name]) in selected_ids
        )
        if not names or names >= frozenset(self._wrapped_tool_order):
            return self.agent

        agent = self._agent_cache.get(names)
        if agent is None:
            subset_tools = [self._wrapped_tools_by_name[name] for name in self._wrapped_tool_order if name in names]
            agent = _create_agent_with_trim(
                self.model,
                tools=subset_tools,
                checkpointer=self.checkpointer,
            )
            self._agent_cache[names] = agent
            logger.info(
                "按问题筛选工具子集: {} -> {}",
                len(self._wrapped_tool_order),
                len(names),
            )
        return agent

    def _build_agent_tools(self, mcp_tools: list[object]) -> list[object]:
        """构建传给 LangChain Agent 的工具列表。

        wrapper 保留原工具 `name/description/args_schema`，但把真实调用统一导向
        ToolResult。错误 ToolResult 只会生成“不可作为事实依据”的安全文本，降低 RAG
        工具异常被模型当知识证据引用的风险。
        """

        if not config.tool_manager_enabled:
            return [*self.tools, *mcp_tools]

        wrapped_tools: list[object] = []
        for tool in self.tools:
            spec = self.tool_manager.wrap_local_tool(tool)
            wrapped_tools.append(self.tool_manager.to_langchain_tool(spec))
        for tool in mcp_tools:
            spec = self.tool_manager.wrap_mcp_tool(tool)
            wrapped_tools.append(self.tool_manager.to_langchain_tool(spec))
        return wrapped_tools

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    def _build_messages(
        self,
        question: str,
        conversation_context: object | None = None,
    ) -> list[BaseMessage]:
        """Build model messages from controlled ConversationManager output only."""

        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        if conversation_context is not None:
            summary = getattr(conversation_context, "summary", None)
            if isinstance(summary, str) and summary.strip():
                messages.append(SystemMessage(content=f"Conversation summary:\n{summary.strip()}"))

            # 用户长期记忆画像（跨会话偏好/事实）：召回失败为空元组，零影响。
            user_memories = getattr(conversation_context, "user_memories", ())
            if isinstance(user_memories, Sequence) and not isinstance(
                user_memories,
                str | bytes | bytearray,
            ):
                memory_lines = [str(item) for item in user_memories if str(item).strip()]
                if memory_lines:
                    memory_block = "\n".join(f"- {line}" for line in memory_lines)
                    messages.append(
                        SystemMessage(
                            content=(
                                "User long-term memory (cross-session preferences/facts, "
                                "use as background reference only). Each item may carry a "
                                "'recorded' date. If memory items conflict with each other, "
                                "trust the one with the latest recorded date; if they "
                                "conflict with the current conversation, the current "
                                f"conversation wins:\n{memory_block}"
                            )
                        )
                    )

            recent_messages = getattr(conversation_context, "recent_messages", ())
            if isinstance(recent_messages, Sequence) and not isinstance(
                recent_messages,
                str | bytes | bytearray,
            ):
                for turn in recent_messages:
                    content = getattr(turn, "content", None)
                    role = getattr(turn, "role", None)
                    if not isinstance(content, str) or not content.strip():
                        continue
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=question))
        return messages

    async def query_with_citations(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> RagAnswerResult:
        """Run the Stage 3B RAG pipeline when enabled and return API-safe citations."""

        if not _stage3b_rag_enabled():
            answer, usage = await self._query_with_context_and_usage(
                question, session_id, conversation_context, ctx
            )
            return RagAnswerResult(answer=answer, usage=usage)

        retrieval = self._rag_retriever().retrieve(question, ctx=ctx)
        _raise_no_answer_if_needed(retrieval)

        budget = self._token_budget_manager().allocate(
            "rag_chat",
            self.model_name,
            ctx,
            current_input=question,
            system_prompt=self.system_prompt,
        )
        context = self._context_builder().build(question, retrieval.chunks, budget, ctx=ctx)
        _raise_no_answer_if_needed(context)

        answer, usage = await self._answer_with_rag_context(
            question,
            conversation_context=conversation_context,
            context=context,
        )
        builder = self._citation_builder()
        if config.citations_enabled:
            citations = builder.build(context, answer, ctx=ctx)
            api_citations = builder.to_api_schema(citations)
        else:
            citations = []
            api_citations = []
        answer = builder.sanitize_answer_anchors(
            answer,
            (citation.citation_id for citation in citations),
            ctx=ctx,
        )
        return RagAnswerResult(
            answer=answer,
            citations=tuple(api_citations),
            usage=usage,
            context_text=context.context_text,
        )

    async def _answer_with_rag_context(
        self,
        question: str,
        *,
        conversation_context: object | None,
        context: RagContext,
    ) -> tuple[str, RawUsage | None]:
        """直连 LLM 生成带 RAG 上下文的回答，并顺带提取真实 usage。"""

        messages = self._build_messages(question, conversation_context)
        if context.context_text:
            messages.insert(
                -1,
                SystemMessage(
                    content=(
                        f"RAG context:\n{context.context_text}\n\n"
                        "引用要求:\n"
                        "- 回答中每个依据上述证据的句子，必须在句末标注对应证据编号，如 [C1] 或 [C2]\n"
                        "- 只能使用证据块中真实存在的编号，禁止编造不存在的编号\n"
                        "- 与证据无关的内容不要标注编号"
                    ),
                ),
            )
        response = await self.model.ainvoke(messages)
        return _message_text(response), extract_message_usage(response)

    def _rag_retriever(self) -> RagRetriever:
        retriever = getattr(self, "rag_retriever", None)
        if isinstance(retriever, RagRetriever):
            return retriever
        return (
            retriever
            if retriever is not None
            else RagRetriever(
                trace_logger=self._trace_logger(),
                query_rewriter=LlmQueryRewriter(),
                reranker=DashScopeReranker(),
            )
        )

    def _context_builder(self) -> ContextBuilder:
        builder = getattr(self, "context_builder", None)
        if isinstance(builder, ContextBuilder):
            return builder
        return (
            builder
            if builder is not None
            else ContextBuilder(
                token_budget_manager=self._token_budget_manager(),
                trace_logger=self._trace_logger(),
            )
        )

    def _citation_builder(self) -> CitationBuilder:
        builder = getattr(self, "citation_builder", None)
        if isinstance(builder, CitationBuilder):
            return builder
        return builder if builder is not None else CitationBuilder(trace_logger=self._trace_logger())

    def _token_budget_manager(self) -> object:
        return getattr(self, "token_budget_manager", token_budget_manager)

    def _trace_logger(self) -> TraceLogger:
        trace_logger = getattr(self, "trace_logger", None)
        if isinstance(trace_logger, TraceLogger):
            return trace_logger
        return TraceLogger(trace_jsonl_path=config.trace_jsonl_path, enabled=config.trace_enabled)

    async def query_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> str:
        """Non-streaming query that uses Stage 2 controlled conversation context."""

        answer, _usage = await self._query_with_context_and_usage(
            question, session_id, conversation_context, ctx
        )
        return answer

    async def _query_with_context_and_usage(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None,
        ctx: RequestContext | None,
    ) -> tuple[str, RawUsage | None]:
        """query_with_context 的 usage 采集版本，供 query_with_citations 复用。"""

        _ = ctx
        try:
            await self._initialize_agent()
            logger.info("[session {}] RAG Agent received context-aware query", session_id)
            agent_input = {"messages": self._build_messages(question, conversation_context)}
            # 挂 callback 只累计本次运行的 LLM 调用；checkpoint 历史消息携带的旧 usage
            # 不参与，避免多轮对话重复累计。
            accumulator = UsageAccumulator()
            config_dict = {
                "configurable": {
                    "thread_id": _agent_thread_id(session_id, conversation_context, ctx)
                },
                "recursion_limit": config.agent_recursion_limit,
                "callbacks": [accumulator],
            }
            result = await self._agent_for_question(question).ainvoke(
                input=agent_input, config=config_dict
            )
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = (
                    last_message.content if hasattr(last_message, "content") else str(last_message)
                )
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info("[session {}] Agent used tools: {}", session_id, tool_names)
                return answer, accumulator.to_raw_usage()
            logger.warning("[session {}] Agent returned empty result", session_id)
            return "", accumulator.to_raw_usage()
        except Exception as exc:
            logger.error(
                "[session {}] RAG Agent context-aware query failed: {}",
                session_id,
                exc.__class__.__name__,
            )
            raise

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）和 LangGraph recursion_limit。
            # recursion_limit 是防止 Agent 工具调用/节点调度异常循环的底层保护；
            # 不改变旧会话字段，只在 LangGraph config 顶层追加限制。
            config_dict = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": config.agent_recursion_limit,
            }

            result = await self._agent_for_question(question).ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = (
                    last_message.content if hasattr(last_message, "content") else str(last_message)
                )

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming query that uses Stage 2 controlled conversation context."""

        if _stage3b_rag_enabled():
            async for chunk in self._query_stream_with_stage3b_pipeline(
                question,
                session_id,
                conversation_context,
                ctx,
            ):
                yield chunk
            return

        async for chunk in self._query_stream_with_legacy_context(
            question,
            session_id,
            conversation_context,
            ctx,
        ):
            yield chunk

    async def _query_stream_with_stage3b_pipeline(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            result = await self.query_with_citations(
                question,
                session_id,
                conversation_context,
                ctx,
            )
            if result.answer:
                yield {"type": "content", "data": result.answer}
            yield {
                "type": "complete",
                "data": {
                    "answer": result.answer,
                    "citations": [dict(citation) for citation in result.citations],
                },
            }
        except Exception as exc:
            logger.error(
                "[session {}] RAG Agent Stage 3B stream failed: {}",
                session_id,
                exc.__class__.__name__,
            )
            raise

    async def _query_stream_with_legacy_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        _ = ctx
        try:
            await self._initialize_agent()
            logger.info("[session {}] RAG Agent received context-aware stream", session_id)
            agent_input = {"messages": self._build_messages(question, conversation_context)}
            config_dict = {
                "configurable": {
                    "thread_id": _agent_thread_id(session_id, conversation_context, ctx)
                },
                "recursion_limit": config.agent_recursion_limit,
            }

            async for token, metadata in self._agent_for_question(question).astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = (
                    metadata.get("langgraph_node", "unknown")
                    if isinstance(metadata, dict)
                    else "unknown"
                )
                message_type = type(token).__name__
                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, "content_blocks", None)
                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_content = block.get("text", "")
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name,
                                    }
            yield {"type": "complete"}
        except Exception as exc:
            logger.error(
                "[session {}] RAG Agent context-aware stream failed: {}",
                session_id,
                exc.__class__.__name__,
            )
            yield {"type": "error", "data": "处理请求时发生错误，请稍后重试。"}
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）和 LangGraph recursion_limit。
            # 流式路径同样需要该上限，否则客户端长连时 Agent 异常循环只能靠断开连接终止。
            config_dict = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": config.agent_recursion_limit,
            }

            async for token, metadata in self._agent_for_question(question).astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = (
                    metadata.get("langgraph_node", "unknown")
                    if isinstance(metadata, dict)
                    else "unknown"
                )
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, "content_blocks", None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_content = block.get("text", "")
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name,
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {"type": "error", "data": "处理请求时发生错误，请稍后重试。"}
            raise

    def get_session_history(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> list[dict[str, str]]:
        """
        获取会话历史。

        Args:
            session_id: 会话ID（即 thread_id）
            ctx: 请求上下文，用于 conversation trace；旧调用方可省略

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        if config.conversation_manager_enabled:
            return self.conversation_manager.get_history(session_id, ctx)
        return self._get_session_history_legacy(session_id)

    def _get_session_history_legacy(self, session_id: str) -> list[dict[str, str]]:
        """旧历史读取回滚路径。

        仅在 `conversation_manager_enabled=false` 时使用。正常路径必须走
        ConversationManager，避免业务层长期依赖 MemorySaver checkpoint 内部结构。
        """

        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            graph_config = {"configurable": {"thread_id": session_id}}

            # 获取该 thread 的最新检查点
            checkpoint_data = self.checkpointer.get(graph_config)

            if not checkpoint_data:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history: list[dict[str, str]] = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                if not isinstance(content, str):
                    continue

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, "timestamp", None)
                if timestamp:
                    history.append({"role": role, "content": content, "timestamp": timestamp})
                else:
                    from datetime import datetime

                    history.append(
                        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
                    )

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）
            ctx: 请求上下文，用于 conversation trace；旧调用方可省略

        Returns:
            bool: 是否成功
        """
        if config.conversation_manager_enabled:
            return self.conversation_manager.clear_session(session_id, ctx)
        return self._clear_session_legacy(session_id)

    def _clear_session_legacy(self, session_id: str) -> bool:
        """旧清理回滚路径，仅在 ConversationManager 开关关闭时使用。"""

        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
def _create_agent_with_trim(
    model: object,
    *,
    tools: list[object],
    checkpointer: object,
) -> object:
    """Attach the existing trim hook only when the installed agent factory supports it."""

    kwargs: dict[str, object] = {"tools": tools, "checkpointer": checkpointer}
    try:
        parameters = inspect.signature(create_agent).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "pre_model_hook" in parameters:
        kwargs["pre_model_hook"] = trim_messages_middleware
    return create_agent(model, **kwargs)


def _agent_thread_id(
    session_id: str,
    conversation_context: object | None,
    ctx: RequestContext | None,
) -> str:
    """Use a request-scoped graph thread when controlled context is supplied."""

    if conversation_context is None:
        return session_id
    request_id = getattr(ctx, "request_id", None)
    if isinstance(request_id, str) and request_id:
        return f"{session_id}:ctx:{request_id}"
    return f"{session_id}:ctx"


def _stage3b_rag_enabled() -> bool:
    return bool(config.new_rag_retriever_enabled and config.context_builder_enabled)


def _raise_no_answer_if_needed(result: RetrievalResult | RagContext) -> None:
    decision = result.no_answer_decision
    if decision is not None and not decision.should_answer:
        raise RagEmptyResultError()


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class _LazyRagAgentService:
    """Delay ChatQwen/LangGraph construction until the RAG service is actually used."""

    def __init__(self, *, streaming: bool = True) -> None:
        self._streaming = streaming
        self._instance: RagAgentService | None = None

    def _get(self) -> RagAgentService:
        if self._instance is None:
            self._instance = RagAgentService(streaming=self._streaming)
        return self._instance

    async def query(self, question: str, session_id: str) -> str:
        return await self._get().query(question, session_id)

    async def query_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> str:
        return await self._get().query_with_context(
            question,
            session_id,
            conversation_context,
            ctx,
        )

    async def query_with_citations(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> RagAnswerResult:
        return await self._get().query_with_citations(
            question,
            session_id,
            conversation_context,
            ctx,
        )

    def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return self._get().query_stream(question, session_id)

    def query_stream_with_context(
        self,
        question: str,
        session_id: str,
        conversation_context: object | None = None,
        ctx: RequestContext | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        query_stream_override = self.__dict__.get("query_stream")
        if callable(query_stream_override):
            # Preserve the long-standing test/runtime injection contract where
            # callers replace query_stream on the singleton without needing to
            # know about the context-aware adapter added in stage 3B.
            return query_stream_override(question, session_id)
        return self._get().query_stream_with_context(
            question,
            session_id,
            conversation_context,
            ctx,
        )

    def get_session_history(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> list[dict[str, str]]:
        return self._get().get_session_history(session_id, ctx)

    def clear_session(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> bool:
        return self._get().clear_session(session_id, ctx)

    async def cleanup(self) -> None:
        await self._get().cleanup()


# Keep the historical singleton name while avoiding import-time model construction.
rag_agent_service = _LazyRagAgentService(streaming=True)
