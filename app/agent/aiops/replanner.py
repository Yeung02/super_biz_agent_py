"""
Replanner 节点：重新规划或生成最终响应
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Dict, Any, List, Literal, cast
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from app.config import config
from app.core.errors import AgentMaxStepExceededError, JsonObject, LLMProviderError
from app.core.llm_usage import UsageAccumulator
from app.core.request_context import get_request_context_or_none
from app.core.token_budget import token_budget_manager
from app.observability.tracing import TraceLogger
from .state import PlanExecuteState
from .structured_output import ainvoke_structured_with_retry
from .utils import format_tools_description, record_llm_usage

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_qwq import ChatQwen
except ModuleNotFoundError:
    class _FallbackPrompt:
        def __or__(self, other: object) -> object:
            return other

    class ChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages: object) -> _FallbackPrompt:
            _ = messages
            return _FallbackPrompt()

    class ChatQwen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs
            raise RuntimeError("langchain_qwq is required for replanner LLM calls")

try:
    from app.tools import get_current_time, retrieve_knowledge
except ModuleNotFoundError:
    class _UnavailableTool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = "unavailable test placeholder"

    get_current_time = _UnavailableTool("get_current_time")
    retrieve_knowledge = _UnavailableTool("retrieve_knowledge")

try:
    from app.agent.mcp_client import get_mcp_client_with_retry
except ModuleNotFoundError:
    async def get_mcp_client_with_retry() -> object:
        raise RuntimeError("MCP client dependencies are unavailable")


_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)


class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(min_length=1, description="对用户的最终响应")

    @field_validator("response")
    @classmethod
    def _response_not_blank(cls, value: str) -> str:
        """空响应是结构化输出解析失败，不能静默替换为空答案。"""
        if not value.strip():
            raise ValueError("response must not be blank")
        return value


class Act(BaseModel):
    """重新规划的输出格式"""
    action: Literal["continue", "replan", "respond"] = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应"""
    )
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）"
    )


# Replanner 提示词
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：** 
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")
    request_ctx = get_request_context_or_none()
    # Replanner 会读取原始任务、剩余计划和历史执行结果，是 AIOps 链路最容易超上下文的
    # 节点。这里先分配 report 场景预算，后续只裁剪历史和工具描述，不裁剪当前用户输入。
    budget = token_budget_manager.allocate(
        "aiops_report",
        config.rag_model,
        request_ctx,
        current_input=input_text,
    )

    # ⚠️ 强制限制：如果已执行步骤过多，直接生成响应
    # 业务步骤上限与 LangGraph recursion_limit 分层控制：这里限制真实执行步骤数，
    # recursion_limit 只兜底异常调度环路。默认值保持 8，兼容旧 AIOps 前端的进度预期。
    max_steps = int(getattr(config, "agent_max_steps", 8))
    if len(past_steps) >= max_steps:
        _record_max_steps_event(current_steps=len(past_steps), max_steps=max_steps)
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制 {max_steps}，停止继续执行")
        return {
            "plan": [],
            "response": "",
            "error_event": _build_max_steps_error_event(),
        }

    # 获取可用工具列表
    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)
        if config.token_budget_enabled:
            trim_result = token_budget_manager.trim_text(
                tools_description,
                budget.tool_result_tokens,
            )
            token_budget_manager.record_trim(
                trim_result,
                request_ctx,
                component="tool_result",
            )
            tools_description = cast(str, trim_result.content)
    except Exception as exc:
        logger.warning("获取工具列表失败: {}", exc.__class__.__name__)
        tools_description = "无法获取工具列表"

    # 创建 LLM
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0
    )

    # 格式化已执行的步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])
    if config.token_budget_enabled and steps_summary:
        # 已执行步骤只作为 replanner 判断依据；超预算时裁剪旧结果，保留最近执行信息即可，
        # 不能把完整工具 payload 或长报告草稿继续注入后续 LLM 调用。
        trim_result = token_budget_manager.trim_text(steps_summary, budget.history_tokens)
        token_budget_manager.record_trim(
            trim_result,
            request_ctx,
            component="history",
        )
        steps_summary = cast(str, trim_result.content)

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        replanner_chain = replanner_prompt | llm.with_structured_output(Act)

        # Act 决策与最终报告是两次独立 LLM 交互，各自记账；解析重试的用量在
        # finally 中记录，重试耗尽降级为"继续原计划"时已消耗的 token 不丢。
        act_usage_accumulator = UsageAccumulator()
        try:
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）")
            ]

            try:
                act = await ainvoke_structured_with_retry(
                    replanner_chain,
                    {
                        "messages": messages,
                        "tools_description": tools_description,
                    },
                    schema=Act,
                    node="replanner",
                    usage_accumulator=act_usage_accumulator,
                )
            finally:
                record_llm_usage(
                    act_usage_accumulator, model=config.rag_model, ctx=request_ctx
                )

            # 统一 coercion 后一定是合法的 Act 实例；解析失败已在重试层抛出，
            # 由下方 except 按"继续执行原计划"fail-open。
            action = act.action
            new_steps = act.new_steps

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == "replan":
                # ⚠️ 强制限制：新步骤数不能超过当前剩余步骤数
                if len(new_steps) > len(plan):
                    logger.warning(
                        f"新步骤数 {len(new_steps)} > 剩余步骤数 {len(plan)}，"
                        f"强制截断为 {len(plan)} 个步骤"
                    )
                    new_steps = new_steps[:len(plan)]

                # ⚠️ 二次检查：如果已执行步骤 >= 5，禁止 replan
                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划，强制生成响应")
                    return await _generate_response(state, llm)

                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    # 替换剩余计划
                    return {"plan": new_steps}
                else:
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                return {}  # 不修改状态，继续执行

        except Exception as exc:
            logger.error("重新规划失败: {}，继续执行剩余计划", exc.__class__.__name__)
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])
    request_ctx = get_request_context_or_none()
    budget = token_budget_manager.allocate(
        "aiops_report",
        config.rag_model,
        request_ctx,
        current_input=input_text,
    )

    # 格式化执行历史
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])
    if config.token_budget_enabled and execution_history:
        # 最终报告 prompt 只需要受控的执行历史摘要。这里不生成新摘要，只做确定性截断；
        # 真实 ConversationSummarizer 留给后续 ISSUE-014。
        trim_result = token_budget_manager.trim_text(
            execution_history,
            budget.tool_result_tokens,
        )
        token_budget_manager.record_trim(
            trim_result,
            request_ctx,
            component="tool_result",
        )
        execution_history = cast(str, trim_result.content)

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应")
        ]

        # 报告生成的解析重试同样消耗 token；重试耗尽走错误事件路径前先记账。
        response_usage_accumulator = UsageAccumulator()
        try:
            response_obj = await ainvoke_structured_with_retry(
                response_gen,
                {"messages": messages},
                schema=Response,
                node="replanner.response",
                usage_accumulator=response_usage_accumulator,
            )
        finally:
            record_llm_usage(
                response_usage_accumulator, model=config.rag_model, ctx=request_ctx
            )
        # 空/空白响应会被 Response schema 拒绝并进入重试；重试耗尽仍失败时
        # 由下方 except 返回稳定错误事件，不再静默产出空答案。
        final_response = response_obj.response

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        return {"response": final_response}

    except Exception as exc:
        logger.error("生成响应失败: {}", exc.__class__.__name__)
        return {"error_event": _build_response_generation_error_event()}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)


def _record_max_steps_event(*, current_steps: int, max_steps: int) -> None:
    """记录业务步骤上限命中事件。

    Replanner 运行在 LangGraph 节点内部，不能依赖 HTTP handler 传参；因此只在当前
    request context 存在时写 trace，缺失时保持旧离线调用行为。事件不包含原始 LLM
    输出或工具结果，避免把内部诊断细节写入对外可关联的错误路径。
    """

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.max_steps",
        ctx,
        status="error",
        error_code="AGENT_MAX_STEP_EXCEEDED",
        current_steps=current_steps,
        max_steps=max_steps,
    )


def _build_max_steps_error_event() -> JsonObject:
    """Build the stable SSE error payload without calling the LLM fallback."""

    ctx = get_request_context_or_none()
    trace_kwargs = (
        {"trace_id": ctx.trace_id, "request_id": ctx.request_id}
        if ctx is not None
        else {"trace_id": None, "request_id": None}
    )
    return AgentMaxStepExceededError().to_sse_payload(**trace_kwargs)


def _build_response_generation_error_event() -> JsonObject:
    """Build a stable error event for final-report LLM failures."""

    ctx = get_request_context_or_none()
    trace_kwargs = (
        {"trace_id": ctx.trace_id, "request_id": ctx.request_id}
        if ctx is not None
        else {"trace_id": None, "request_id": None}
    )
    return LLMProviderError().to_sse_payload(**trace_kwargs)
