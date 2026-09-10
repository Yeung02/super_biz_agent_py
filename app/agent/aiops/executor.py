"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_tools_with_retry
from app.agent.tool_manager import get_tool_manager
from app.agent.tool_selector import ToolSelector
from app.config import config
from app.core.errors import AppError, JsonObject, ToolExecutionError
from app.core.fallback import FallbackManager
from app.core.llm_usage import UsageAccumulator
from app.core.request_context import get_request_context_or_none
from app.core.token_budget import token_budget_manager
from app.observability.tracing import TraceLogger
from app.tools import get_current_time, retrieve_knowledge

from .state import PlanExecuteState
from .utils import record_llm_usage

# Executor 步骤失败时消费 fallback 矩阵；不做 IO，只决定 past_steps 中的安全文案。
_fallback_manager = FallbackManager()
_tool_selector = ToolSelector()
_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)

_WORD_BOUNDARY_CHARS = re.compile(r"[\w]")


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        request_ctx = get_request_context_or_none()
        # Executor 当前阶段仍走旧的 LLM -> ToolNode 流程。预算只用于限制 prompt 中的
        # 步骤结果写回 past_steps，避免某个工具返回大 payload 后继续污染 replanner/report。
        budget = token_budget_manager.allocate("aiops_execute", config.rag_model, request_ctx)

        # 获取本地工具
        local_tools = [get_current_time, retrieve_knowledge]

        # 获取 MCP 工具。MCP client 内部仍保留 retry interceptor；本层只负责把
        # 原始工具交给 ToolManager 包装，不新增 MCP server 能力。
        mcp_tools = await get_mcp_tools_with_retry()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 按步骤文本筛选相关工具子集，降低无关工具 schema 干扰；无命中时回退全量。
        selected_local, selected_mcp = _select_step_tools(task, local_tools, mcp_tools)
        if len(selected_local) + len(selected_mcp) < len(local_tools) + len(mcp_tools):
            logger.info(
                f"步骤工具筛选: {len(local_tools) + len(mcp_tools)} -> "
                f"{len(selected_local) + len(selected_mcp)}"
            )

        # 合并所有工具。默认包装后再交给 bind_tools/ToolNode；若配置关闭则回到旧
        # `local_tools + mcp_tools` 行为，作为 ISSUE-008 的独立回滚开关。
        all_tools = _build_executor_tools(selected_local, selected_mcp)

        # 步骤文本只命中一个已绑定工具时，首轮 LLM 调用强制使用该工具（tool_choice）。
        forced_tool = (
            _extract_forced_tool(task, all_tools)
            if config.tool_choice_enabled
            else None
        )

        # 创建 LLM（绑定工具）
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)
        if forced_tool:
            logger.info(f"步骤指定唯一工具，强制调用: {forced_tool}")
            llm_forced = llm.bind_tools(
                all_tools, tool_choice={"type": "tool", "name": forced_tool}
            )
        else:
            llm_forced = llm.bind_tools(all_tools)
        # 工具执行后的总结调用不强制 tool_choice，否则会反复调用同一工具。
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(
                content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""
            ),
            HumanMessage(content=f"请执行以下任务: {task}"),
        ]

        # 第一步：LLM 决定是否调用工具（命中唯一工具时已被强制指定）。
        # 一次步骤最多两次 LLM 调用（工具决策 + 结果总结），共用一个 accumulator，
        # 汇总成一条 token.usage 事件；中间工具执行失败时已消耗的首次调用量
        # 也要记账，因此记录放在 finally。
        usage_accumulator = UsageAccumulator()
        usage_callbacks = {"callbacks": [usage_accumulator]}
        try:
            llm_response = await llm_forced.ainvoke(messages, config=usage_callbacks)
            logger.info(f"LLM 响应类型: {type(llm_response)}")

            # 第二步：如果有工具调用，执行工具
            if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

                # 使用 ToolNode 自动执行工具
                messages.append(llm_response)
                tool_messages = await tool_node.ainvoke({"messages": messages})

                # 第三步：将工具结果返回给 LLM 生成最终答案
                messages.extend(tool_messages["messages"])
                final_response = await llm_with_tools.ainvoke(messages, config=usage_callbacks)
                result = (
                    final_response.content
                    if hasattr(final_response, "content")
                    else str(final_response)
                )
            else:
                # 没有工具调用，直接使用 LLM 的输出
                logger.info("LLM 未调用工具，直接返回结果")
                result = (
                    llm_response.content
                    if hasattr(llm_response, "content")
                    else str(llm_response)
                )
        finally:
            record_llm_usage(usage_accumulator, model=config.rag_model, ctx=request_ctx)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")
        if config.token_budget_enabled:
            trim_result = token_budget_manager.trim_text(result, budget.tool_result_tokens)
            token_budget_manager.record_trim(
                trim_result,
                request_ctx,
                component="tool_result",
            )
            result = cast(str, trim_result.content)

        # 返回更新：移除已执行的步骤，添加执行历史
        update: dict[str, Any] = {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }
        # ISSUE-A：drain 本步骤执行期间 ToolManager 收集的工具证据，写入 state
        # 供后续 Critic 核对答案断言；采集 fail-open，失败时返回空列表不影响主路径。
        evidence_entries = _drain_step_evidence(task)
        if evidence_entries:
            update["tool_evidence"] = evidence_entries
        return update

    except Exception as e:
        logger.error("执行步骤失败: {}", e.__class__.__name__)
        update = {
            "plan": plan[1:],
            # past_steps 会进入 replanner/report 语境，不能把原始异常全文放进去。
            # 这里把失败交给 FallbackManager 按矩阵决策：文案来自稳定策略矩阵，
            # fallback_enabled=false 时同样返回矩阵外的通用安全文案；详细错误
            # 留在日志和 ToolManager trace 中，ToolResult 的 fallback_required
            # 信号也通过 trace 暴露给排障链路。
            "past_steps": [(task, _step_failure_message(e))],
        }
        # 工具可能已在步骤失败前执行成功；证据照常 drain，让 Critic 仍能核对到
        # 已获取的部分证据，而不是整步无痕迹丢失。
        evidence_entries = _drain_step_evidence(task)
        if evidence_entries:
            update["tool_evidence"] = evidence_entries
        return update


def _select_step_tools(
    task: str,
    local_tools: list[object],
    mcp_tools: list[object],
) -> tuple[list[object], list[object]]:
    """按步骤文本筛选工具子集；关闭开关或无命中时返回全量。"""

    if not config.tool_selection_enabled:
        return local_tools, mcp_tools
    return _tool_selector.select_split(task, local_tools, mcp_tools)


def _extract_forced_tool(task: str, tools: list[object]) -> str | None:
    """从步骤文本中提取被唯一指定的工具名。

    planner/replanner 的步骤是自由文本（受 prompt 约束会写明工具名）。这里用
    词边界匹配已绑定工具名：只命中一个工具时才强制调用；多命中或不命中保持
    LLM 自主选择，避免误绑错误工具。
    """

    if not task:
        return None
    lowered = task.lower()
    matched: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            continue
        for match_start, matched_text in _iter_name_occurrences(lowered, name.lower()):
            _ = match_start
            _ = matched_text
            matched.append(name)
            break
    unique = sorted(set(matched))
    if len(unique) == 1:
        return unique[0]
    return None


def _iter_name_occurrences(text: str, name: str):
    """生成所有完整词边界命中的位置，避免子串误命中（如 tool_a 命中 tool）。"""

    if not name:
        return
    start = 0
    while True:
        index = text.find(name, start)
        if index < 0:
            return
        before_ok = index == 0 or not _WORD_BOUNDARY_CHARS.match(text[index - 1])
        end = index + len(name)
        after_ok = end >= len(text) or not _WORD_BOUNDARY_CHARS.match(text[end])
        if before_ok and after_ok:
            yield index, name
        start = index + 1


def _drain_step_evidence(task: str) -> list[JsonObject]:
    """取出本步骤执行期间 ToolManager 收集的证据块，标注步骤后供 state 累加。

    证据链是 Critic 的输入而非主路径依赖，必须 fail-open：ToolManager 关闭、无请求
    上下文或 drain 异常都返回空列表，executor 的执行结果不受影响。
    """

    if not config.tool_manager_enabled:
        return []
    ctx = get_request_context_or_none()
    if ctx is None:
        return []
    try:
        blocks = get_tool_manager().drain_request_evidence(ctx.request_id)
    except Exception:
        return []
    entries = [{"step": task, **block.to_dict()} for block in blocks]
    _record_step_evidence(entries)
    return entries


def _record_step_evidence(entries: list[JsonObject]) -> None:
    """记录本步骤证据收集信号（ISSUE-C 灰度可观测性）。

    只记数量与可用块数，不落步骤文本和证据正文：与 critic.py 的 trace 原则一致，
    草稿/证据内容不进 trace。block_count=0 也是有效信号（该步无工具证据入链）。
    """

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.executor.evidence",
        ctx,
        block_count=len(entries),
        usable_count=sum(1 for entry in entries if bool(entry.get("usable"))),
    )


def _step_failure_message(exc: Exception) -> str:
    """按 fallback 矩阵生成步骤失败的安全文案。"""

    app_error = (
        exc
        if isinstance(exc, AppError)
        else ToolExecutionError(
            internal_message=f"{exc.__class__.__name__}: {exc}",
            details={"stage": "executor"},
        )
    )
    ctx = get_request_context_or_none()
    try:
        decision = _fallback_manager.decide(app_error, ctx, scenario="aiops")
        return decision.safe_message
    except Exception:
        # FallbackManager 自身异常不能让 executor 节点崩溃，退回固定安全文案。
        return "执行失败：工具或模型调用失败，系统已记录问题。"


def _build_executor_tools(local_tools: list[object], mcp_tools: list[object]) -> list[object]:
    """构建 AIOps Executor 使用的工具列表。

    Executor 仍使用现有 LLM -> ToolNode 流程，本函数只替换工具对象为 ToolManager
    wrapper。这样 `bind_tools` 获取到的 schema 与 ToolNode 实际执行的工具保持一致，
    同时工具错误会被 ToolResult 标记为不可用证据。使用进程级共享 ToolManager，
    避免每步执行重复注册和包装。
    """

    if not config.tool_manager_enabled:
        return [*local_tools, *mcp_tools]

    manager = get_tool_manager()
    wrapped_tools: list[object] = []
    for tool in local_tools:
        wrapped_tools.append(manager.to_langchain_tool(manager.wrap_local_tool(tool)))
    for tool in mcp_tools:
        wrapped_tools.append(manager.to_langchain_tool(manager.wrap_mcp_tool(tool)))
    return wrapped_tools
