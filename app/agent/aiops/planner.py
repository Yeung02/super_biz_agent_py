"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Dict, Any, List, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.core.errors import RequestTooLargeError
from app.core.llm_usage import UsageAccumulator
from app.core.request_context import get_request_context_or_none
from app.core.token_budget import token_budget_manager
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState
from .structured_output import ainvoke_structured_with_retry
from .utils import format_tools_description, record_llm_usage


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        min_length=1,
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )


# Planner 提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
                - 将任务分解为逻辑上独立的步骤
                - 每个步骤应该明确使用哪些工具(如果需要工具的话)来获取信息, 最好能同时提供工具执行所需要的参数
                - 步骤之间应该有清晰的依赖关系
                - 步骤描述要具体、可操作
                - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

                示例输入："分析当前系统的性能问题"
                示例输出（假设有对应工具）：
                步骤1: 使用 get_metrics 工具收集系统的 CPU 和内存使用情况
                步骤2: 使用 query_logs 工具检查最近的错误日志
                步骤3: 使用 query_database 工具分析慢查询日志
                步骤4: 综合以上信息生成性能分析报告
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        request_ctx = get_request_context_or_none()
        # Planner 会把用户问题、工具描述和经验文档一起拼进 prompt。ISSUE-012 先在这里
        # 分配预算并校验当前问题，避免旧路径把超长输入或大段 Runbook 静默塞给 LLM。
        budget = token_budget_manager.allocate(
            "aiops_plan",
            config.rag_model,
            request_ctx,
            current_input=input_text,
        )

        # 步骤1: 查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            # retrieve_knowledge 使用 response_format="content_and_artifact"
            # ainvoke() 只返回 content（字符串），不是元组
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                if config.token_budget_enabled:
                    trim_result = token_budget_manager.trim_text(
                        context_str,
                        budget.rag_context_tokens,
                    )
                    token_budget_manager.record_trim(
                        trim_result,
                        request_ctx,
                        component="rag_chunk",
                    )
                    experience_docs = cast(str, trim_result.content)
                else:
                    experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning("查询内部文档失败: {}", e.__class__.__name__)

        # 步骤2: 获取可用工具列表
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
            # 工具描述来自内部列表，不能无限增长。这里按 tool_result 槽位裁剪描述文本，
            # 只影响 LLM 看到的工具说明长度，不删除真实工具，也不改变 ToolManager 行为。
            tool_description_trim = token_budget_manager.trim_text(
                tools_description,
                budget.tool_result_tokens,
            )
            token_budget_manager.record_trim(
                tool_description_trim,
                request_ctx,
                component="tool_result",
            )
            tools_description = cast(str, tool_description_trim.content)

        # 步骤3: 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤4: 创建 LLM 并生成计划
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划。解析失败（含空步骤列表）先带错误反馈重试，仍失败时
        # 由外层 except 降级为默认计划；空计划不再静默流入 state。
        # usage 以 callbacks 采集：结构化输出的解析重试同样消耗真实 token，
        # 重试耗尽抛出时已发生的用量也不能丢，因此记录放在 finally。
        usage_accumulator = UsageAccumulator()
        try:
            plan_result = await ainvoke_structured_with_retry(
                planner_chain,
                {
                    "messages": [("user", input_text)],
                    "tools_description": tools_description,
                    "experience_context": experience_context,
                },
                schema=Plan,
                node="planner",
                usage_accumulator=usage_accumulator,
            )
        finally:
            record_llm_usage(usage_accumulator, model=config.rag_model, ctx=request_ctx)

        # 统一 coercion 后一定是合法的 Plan 实例（steps 非空由 schema 保证）
        plan_steps = plan_result.steps

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        return {"plan": plan_steps}

    except RequestTooLargeError:
        logger.warning("Planner 输入超过 token 硬上限，交由服务层返回 REQUEST_TOO_LARGE")
        raise

    except Exception as e:
        logger.error("生成计划失败: {}", e.__class__.__name__)
        # 返回一个默认计划
        return {
            "plan": [
                "收集相关信息",
                "分析数据",
                "生成报告"
            ]
        }
