"""app.core.llm_usage 单元测试：真实 usage 的提取与累计。

测试只验证采集契约：usage_metadata / token_usage 两种来源的读取、多次 LLM 调用
累计、以及无 usage 时返回 None（调用方回退本地估算）。不访问真实 LangChain 响应。

附带验证 aiops utils 的 record_llm_usage 记账契约：累计值以 estimated=False 写入
token_budget_manager；未捕获 usage 时是 no-op，不写估算值。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.llm_usage import RawUsage, UsageAccumulator, extract_message_usage


def test_extract_message_usage_prefers_usage_metadata() -> None:
    """LangChain 标准 usage_metadata 优先于 OpenAI 风格 token_usage。"""

    message = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    )

    assert extract_message_usage(message) == RawUsage(input_tokens=100, output_tokens=50)


def test_extract_message_usage_falls_back_to_response_metadata() -> None:
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        },
    )

    assert extract_message_usage(message) == RawUsage(input_tokens=7, output_tokens=3)


def test_extract_message_usage_returns_none_without_usage() -> None:
    assert extract_message_usage(SimpleNamespace(content="hi")) is None
    assert extract_message_usage(object()) is None


def test_extract_message_usage_rejects_non_integer_usage() -> None:
    """非整数 usage 视为无效，返回 None 让调用方回退估算，而不是抛错。"""

    message = SimpleNamespace(
        usage_metadata={"input_tokens": "100", "output_tokens": 5},
    )

    assert extract_message_usage(message) is None


def _llm_result_with_message_usage(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(generations=[[generation]], llm_output=None)


def _llm_result_with_llm_output(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        generations=[[]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        },
    )


def test_accumulator_sums_multiple_llm_calls() -> None:
    """Agent 循环内的多次 LLM 调用要累计，覆盖 message 和 llm_output 两种来源。"""

    accumulator = UsageAccumulator()
    accumulator.on_llm_end(_llm_result_with_message_usage(100, 20))
    accumulator.on_llm_end(_llm_result_with_llm_output(30, 5))

    assert accumulator.to_raw_usage() == RawUsage(input_tokens=130, output_tokens=25)
    assert accumulator.call_count == 2


def test_accumulator_without_usage_returns_none() -> None:
    accumulator = UsageAccumulator()
    accumulator.on_llm_end(SimpleNamespace(generations=[[]], llm_output=None))

    assert accumulator.to_raw_usage() is None
    assert accumulator.call_count == 0


def test_accumulator_accepts_callback_kwargs() -> None:
    """LangChain 分发事件时会附带 run_id 等参数，签名必须容忍额外 kwargs。"""

    accumulator = UsageAccumulator()
    accumulator.on_llm_end(_llm_result_with_message_usage(1, 1), run_id="ignored")

    assert accumulator.to_raw_usage() == RawUsage(input_tokens=1, output_tokens=1)


class _RecordingBudgetManager:
    """捕获 record_usage 调用的最小替身。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        ctx: object | None = None,
        estimated: bool = False,
    ) -> None:
        self.calls.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ctx": ctx,
                "estimated": estimated,
            }
        )


def test_record_llm_usage_writes_real_usage_without_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIOps 节点累计的真实 usage 以 estimated=False 记账，不落估算值。"""

    from app.agent.aiops import utils as aiops_utils

    budget_manager = _RecordingBudgetManager()
    monkeypatch.setattr(aiops_utils, "token_budget_manager", budget_manager)

    accumulator = UsageAccumulator()
    accumulator.on_llm_end(_llm_result_with_message_usage(120, 30))
    accumulator.on_llm_end(_llm_result_with_llm_output(10, 5))

    aiops_utils.record_llm_usage(
        accumulator, model="qwen-critic-test", ctx=None
    )

    assert budget_manager.calls == [
        {
            "model": "qwen-critic-test",
            "input_tokens": 130,
            "output_tokens": 35,
            "ctx": None,
            "estimated": False,
        }
    ]


def test_record_llm_usage_noop_without_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未捕获任何 usage（LLM 失败/供应商未返回）时不记账，宁缺毋滥。"""

    from app.agent.aiops import utils as aiops_utils

    budget_manager = _RecordingBudgetManager()
    monkeypatch.setattr(aiops_utils, "token_budget_manager", budget_manager)

    aiops_utils.record_llm_usage(UsageAccumulator(), model="qwen-test", ctx=None)

    assert budget_manager.calls == []
