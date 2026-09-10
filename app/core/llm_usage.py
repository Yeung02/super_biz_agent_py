"""LLM 真实 usage 采集。

职责边界很窄：只负责从 LangChain 响应对象中读出 token 数字，并在一次请求的
多次 LLM 调用之间累计。不计算成本、不写 trace——记录统一走
TokenBudgetManager.record_usage，这样价格换算、estimated 标记和 trace 字段
保持单一出口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ModuleNotFoundError:  # 与 rag_agent_service 相同的降级约定：无 agent 依赖时保持可导入
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class RawUsage:
    """不带模型名和价格的原始 usage。

    字段命名与 LangChain usage_metadata 对齐；价格换算留给 record_usage，
    避免采集层重复维护单价表。
    """

    input_tokens: int
    output_tokens: int


def extract_message_usage(message: object) -> RawUsage | None:
    """从 AIMessage/AIMessageChunk 等响应对象提取真实 usage。

    优先读 LangChain 标准 `usage_metadata`，兜底 OpenAI 风格
    `response_metadata["token_usage"]`。两者都没有时返回 None，调用方回退本地估算。
    """

    usage_meta = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        raw = _from_mapping(usage_meta, "input_tokens", "output_tokens")
        if raw is not None:
            return raw
    response_meta = getattr(message, "response_metadata", None)
    token_usage = response_meta.get("token_usage") if isinstance(response_meta, dict) else None
    if isinstance(token_usage, dict):
        return _from_mapping(token_usage, "prompt_tokens", "completion_tokens")
    return None


def _from_mapping(data: dict[str, Any], input_key: str, output_key: str) -> RawUsage | None:
    input_tokens = data.get(input_key)
    output_tokens = data.get(output_key)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return RawUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return None


class UsageAccumulator(BaseCallbackHandler):
    """挂在 ainvoke/astream 的 callbacks 上，跨 Agent 循环累计真实 usage。

    LangGraph checkpoint 会把历史 AIMessage 连同 usage_metadata 一起持久化，因此不能
    通过扫描 result["messages"] 汇总（多轮对话会重复累计）；callback 只统计本次运行内
    发生的 LLM 调用。on_llm_end 保持同步实现：LangChain 的同步与异步分发都会调用它。
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        _ = kwargs
        usage = self._from_llm_result(response)
        if usage is None:
            usage = self._from_generations(response)
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.call_count += 1

    def _from_llm_result(self, response: Any) -> RawUsage | None:
        llm_output = getattr(response, "llm_output", None)
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(token_usage, dict):
            return _from_mapping(token_usage, "prompt_tokens", "completion_tokens")
        return None

    def _from_generations(self, response: Any) -> RawUsage | None:
        generations = getattr(response, "generations", None)
        if not isinstance(generations, list):
            return None
        for generation_list in generations:
            if not isinstance(generation_list, list):
                continue
            for generation in generation_list:
                usage = extract_message_usage(getattr(generation, "message", None))
                if usage is not None:
                    return usage
        return None

    def to_raw_usage(self) -> RawUsage | None:
        """返回累计结果；本次运行没有捕获到任何 usage 时返回 None（调用方回退估算）。"""

        if self.call_count == 0:
            return None
        return RawUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)
