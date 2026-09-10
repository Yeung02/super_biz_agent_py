"""
AIOps Agent 通用工具函数
"""

from typing import List

from app.core.llm_usage import UsageAccumulator
from app.core.request_context import RequestContext
from app.core.token_budget import token_budget_manager


def format_tools_description(tools: List) -> str:
    """格式化工具列表为描述文本"""
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)


def record_llm_usage(
    accumulator: UsageAccumulator,
    *,
    model: str,
    ctx: RequestContext | None,
) -> None:
    """把节点内累计的真实 LLM usage 写入 token.usage trace。

    四个 AIOps 节点共用：planner/executor/replanner/critic 的每次 LLM 交互结束后
    调用一次。accumulator 未捕获到任何 usage（LLM 调用失败或供应商未返回）时是
    no-op——AIOps 节点没有可回退的文本估算路径，宁可少记也不写估算值。
    """

    usage = accumulator.to_raw_usage()
    if usage is None:
        return
    token_budget_manager.record_usage(
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        ctx=ctx,
        estimated=False,
    )
