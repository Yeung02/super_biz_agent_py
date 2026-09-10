"""结构化输出统一解析重试与观测测试（structured_output.py）。

覆盖四层边界：
1. 有界重试：解析类异常带错误反馈重试一次，成功即返回；重试耗尽抛最后一次异常。
2. 异常分类：provider/网络类错误不重试，直接向上抛给节点既有 fail-open 路径。
3. 统一 coercion：实例/dict 双形态规范化；非法形态（含 schema 约束拒绝）进入重试。
4. schema 边界：Plan 非空、Act/CriticVerdict 枚举、Response/RevisedResponse 空值
   防御，防止非法决策或空答案静默流入 state。

全部使用脚本化 fake chain，不连接真实 DashScope。
"""

from __future__ import annotations

import importlib

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.agent.aiops.critic import CriticVerdict, RevisedResponse
from app.agent.aiops.planner import Plan
from app.agent.aiops.replanner import Act, Response
from app.agent.aiops.structured_output import (
    StructuredOutputCoercionError,
    ainvoke_structured_with_retry,
)
from app.config import config
from app.core.llm_usage import UsageAccumulator


class _ScriptedChain:
    """按脚本依次返回/抛出结果的 fake chain，记录每次 ainvoke 的 payload 与 config。"""

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.payloads: list[object] = []
        self.configs: list[dict[str, object] | None] = []

    async def ainvoke(
        self, payload: object, config: dict[str, object] | None = None
    ) -> object:
        self.payloads.append(payload)
        self.configs.append(config)
        if not self.script:
            raise AssertionError("script exhausted")
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


@pytest.mark.asyncio
async def test_parse_error_retries_with_feedback_and_recovers() -> None:
    """解析失败后带修复提示重试一次；成功后返回规范化实例。"""

    chain = _ScriptedChain(
        [
            OutputParserException("bad json"),
            {"steps": ["步骤1"]},
        ]
    )

    result = await ainvoke_structured_with_retry(
        chain,
        {
            "messages": [("user", "task")],
            "tools_description": "tools",
        },
        schema=Plan,
        node="test",
    )

    assert isinstance(result, Plan)
    assert result.steps == ["步骤1"]
    assert len(chain.payloads) == 2

    first, second = chain.payloads
    # 首次 payload 原样透传
    assert first == {
        "messages": [("user", "task")],
        "tools_description": "tools",
    }
    # 重试 payload 只在 messages 末尾追加修复提示，其余键不变
    assert isinstance(second, dict)
    assert second["tools_description"] == "tools"
    second_messages = second["messages"]
    assert isinstance(second_messages, list)
    assert second_messages[0] == ("user", "task")
    hint = second_messages[1]
    assert hint[0] == "user"
    assert "Plan" in hint[1]


@pytest.mark.asyncio
async def test_list_payload_receives_appended_hint() -> None:
    """critic 形态的裸消息列表 payload 同样在末尾追加修复提示。"""

    chain = _ScriptedChain(
        [
            OutputParserException("bad json"),
            {"verdict": "accept"},
        ]
    )
    messages = [("system", "judge"), ("user", "draft")]

    result = await ainvoke_structured_with_retry(
        chain,
        messages,
        schema=CriticVerdict,
        node="test",
    )

    assert result.verdict == "accept"
    first, second = chain.payloads
    assert first == messages
    assert isinstance(second, list)
    assert second[:2] == messages
    assert second[2][0] == "user"
    # 原 payload 不被修改
    assert messages == [("system", "judge"), ("user", "draft")]


@pytest.mark.asyncio
async def test_retry_exhausted_raises_last_parse_error() -> None:
    """重试耗尽后抛出最后一次解析异常，由调用节点按既有策略 fail-open。"""

    chain = _ScriptedChain(
        [
            {"steps": []},  # min_length=1 拒绝
            {"steps": []},
        ]
    )

    with pytest.raises(ValidationError):
        await ainvoke_structured_with_retry(
            chain,
            {"messages": [("user", "task")]},
            schema=Plan,
            node="test",
        )

    assert len(chain.payloads) == 2


@pytest.mark.asyncio
async def test_provider_error_is_not_retried() -> None:
    """网络/provider 类错误不重试，直接向上抛，不在重试层放大延迟。"""

    chain = _ScriptedChain([RuntimeError("provider down")])

    with pytest.raises(RuntimeError):
        await ainvoke_structured_with_retry(
            chain,
            {"messages": [("user", "task")]},
            schema=Plan,
            node="test",
        )

    assert len(chain.payloads) == 1


@pytest.mark.asyncio
async def test_retry_disabled_keeps_legacy_single_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """structured_output_retry_enabled=false 回到"解析失败直接抛出"的旧行为。"""

    monkeypatch.setattr(
        config, "structured_output_retry_enabled", False, raising=False
    )
    chain = _ScriptedChain([{"steps": []}])

    with pytest.raises(ValidationError):
        await ainvoke_structured_with_retry(
            chain,
            {"messages": [("user", "task")]},
            schema=Plan,
            node="test",
        )

    assert len(chain.payloads) == 1


@pytest.mark.asyncio
async def test_coercion_accepts_schema_instance() -> None:
    """chain 直接返回 schema 实例时原样通过。"""

    chain = _ScriptedChain([Act(action="continue")])

    result = await ainvoke_structured_with_retry(
        chain,
        {"messages": [("user", "task")]},
        schema=Act,
        node="test",
    )

    assert result.action == "continue"
    assert result.new_steps == []


@pytest.mark.asyncio
async def test_non_dict_non_instance_output_is_coercion_error_and_retried() -> None:
    """既不是实例也不是 dict 的返回形态进入重试，而不是静默走 dict 兜底。"""

    chain = _ScriptedChain(
        [
            "raw text",
            {"action": "respond"},
        ]
    )

    result = await ainvoke_structured_with_retry(
        chain,
        {"messages": [("user", "task")]},
        schema=Act,
        node="test",
    )

    assert result.action == "respond"
    assert len(chain.payloads) == 2


@pytest.mark.asyncio
async def test_dict_payload_without_messages_key_is_not_mutated() -> None:
    """模板输入不含 messages 键时无法注入提示：原样重试，不破坏 payload。"""

    chain = _ScriptedChain(
        [
            OutputParserException("bad json"),
            {"steps": ["步骤1"]},
        ]
    )
    payload = {"other": "value"}

    result = await ainvoke_structured_with_retry(
        chain,
        payload,
        schema=Plan,
        node="test",
    )

    assert result.steps == ["步骤1"]
    assert chain.payloads[0] == {"other": "value"}
    assert chain.payloads[1] == {"other": "value"}
    assert payload == {"other": "value"}


def test_plan_rejects_empty_steps() -> None:
    """空步骤列表是结构化输出失败，不能静默产出空计划。"""

    with pytest.raises(ValidationError):
        Plan(steps=[])


def test_act_rejects_unknown_action() -> None:
    """action 是三值枚举：未知值必须被 schema 拒绝，不能落入 else 当 continue。"""

    with pytest.raises(ValidationError):
        Act(action="contnue")


def test_response_rejects_empty_and_blank() -> None:
    """空/空白最终响应必须被拒绝，不能静默产出空答案。"""

    with pytest.raises(ValidationError):
        Response(response="")
    with pytest.raises(ValidationError):
        Response(response="   ")


def test_revised_response_rejects_empty_and_blank() -> None:
    """空/空白修订稿必须被拒绝，不能把原答案覆盖成空串。"""

    with pytest.raises(ValidationError):
        RevisedResponse(response="")
    with pytest.raises(ValidationError):
        RevisedResponse(response="   ")


def test_critic_verdict_rejects_unknown_value() -> None:
    """verdict 是二值枚举：未知值必须被 schema 拒绝。"""

    with pytest.raises(ValidationError):
        CriticVerdict(verdict="maybe")


@pytest.mark.asyncio
async def test_replanner_empty_response_returns_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最终响应为空串：重试耗尽后返回稳定错误事件，不再静默返回空答案。"""

    replanner_module = importlib.import_module("app.agent.aiops.replanner")

    class _Prompt:
        def __or__(self, other: object) -> object:
            return other

    class _EmptyResponseChain:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self, payload: object, config: dict[str, object] | None = None
        ) -> object:
            # replanner 现在传入 usage 采集 callbacks；fake 容忍 config 参数。
            _ = payload, config
            self.calls += 1
            return {"response": ""}

    class _LLM:
        def __init__(self) -> None:
            self.chain = _EmptyResponseChain()

        def with_structured_output(self, schema: object) -> _EmptyResponseChain:
            _ = schema
            return self.chain

    monkeypatch.setattr(
        replanner_module, "response_prompt", _Prompt(), raising=False
    )
    llm = _LLM()

    result = await replanner_module._generate_response(
        {
            "input": "diagnose alert",
            "plan": [],
            "past_steps": [("query logs", "cpu=83%")],
            "response": "",
        },
        llm,
    )

    assert "response" not in result
    assert result["error_event"]["type"] == "error"
    assert result["error_event"]["error"]["code"] == "LLM_PROVIDER_ERROR"
    # 空响应触发一次带反馈的重试
    assert llm.chain.calls == 2


def test_structured_output_coercion_error_is_parse_error() -> None:
    """coercion 失败属于解析类异常，确保进入 PARSE_ERROR_TYPES 重试分支。"""

    from app.agent.aiops.structured_output import PARSE_ERROR_TYPES

    assert StructuredOutputCoercionError in PARSE_ERROR_TYPES


@pytest.mark.asyncio
async def test_usage_accumulator_receives_callbacks_on_every_attempt() -> None:
    """解析失败重试的 LLM 调用同样经 callbacks 采集真实 usage，不能只计首次。"""

    chain = _ScriptedChain(
        [
            OutputParserException("bad json"),
            {"steps": ["步骤1"]},
        ]
    )
    accumulator = UsageAccumulator()

    result = await ainvoke_structured_with_retry(
        chain,
        {
            "messages": [("user", "task")],
            "tools_description": "tools",
        },
        schema=Plan,
        node="test",
        usage_accumulator=accumulator,
    )

    assert result.steps == ["步骤1"]
    assert len(chain.configs) == 2
    for config in chain.configs:
        assert isinstance(config, dict)
        callbacks = config.get("callbacks")
        assert isinstance(callbacks, list)
        assert callbacks == [accumulator]


@pytest.mark.asyncio
async def test_without_accumulator_keeps_legacy_invoke_signature() -> None:
    """不传 usage_accumulator 时 ainvoke 不带 config，兼容旧 fake chain 签名。"""

    chain = _ScriptedChain([{"steps": ["步骤1"]}])

    result = await ainvoke_structured_with_retry(
        chain,
        {"messages": [("user", "task")]},
        schema=Plan,
        node="test",
    )

    assert result.steps == ["步骤1"]
    assert chain.configs == [None]
