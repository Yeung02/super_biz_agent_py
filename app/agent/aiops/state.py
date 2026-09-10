"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import Any, List, TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict, total=False):
    """Plan-Execute-Replan 状态"""
    
    # 用户输入（任务描述）
    input: str
    
    # 执行计划（步骤列表）
    plan: List[str]
    
    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # 每步工具证据块（operator.add 累加），ISSUE-A Critic 证据链输入。
    # 块为 JSON-safe dict：step/tool_name/usable/text；失败工具以"不可用声明"入链，
    # 供后续 Critic 节点核对答案断言。只存摘要不存完整 ToolResult，避免撑爆 checkpointer。
    tool_evidence: Annotated[List[dict[str, Any]], operator.add]
    
    # 最终响应/报告
    response: str

    # Critic 是否已完成答案审查（ISSUE-B）。非累加：critic 节点写入 True，
    # 条件边据此防止 response 存在时重复路由进 critic。
    critic_reviewed: bool

    # Standardized error event; the service layer turns it into SSE `type=error`.
    error_event: dict[str, Any]
