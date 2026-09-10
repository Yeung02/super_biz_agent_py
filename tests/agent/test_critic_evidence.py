"""ISSUE-A Critic 证据链单元与集成测试。

覆盖两层边界：ToolManager 按 request 收集 EvidenceBlock（成功/失败/截断/开关/
取走语义），以及 executor 每步 drain 证据并标注 step 写入 state 更新。全部使用
fake，不连接真实 DashScope、MCP 或 Milvus，也不构建真实 LangGraph 图。
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from app.agent.policies import PolicyRegistry
from app.agent.tool_manager import ToolManager
from app.config import config
from app.core.request_context import reset_request_context, set_request_context
from app.observability.tracing import TraceLogger


def _evidence_manager(**kwargs: object) -> ToolManager:
    """构造关闭 policy 的 ToolManager，让测试只关注证据收集行为。

    与 tests/unit/test_tool_manager.py 的做法一致：生产默认会按 allowlist 拦截
    未知工具，这里临时工具名只为验证证据登记，不应混入权限语义。
    """

    return ToolManager(
        policy_registry=PolicyRegistry(enabled=False),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_success_tool_call_records_usable_evidence_block(
    fake_request_context,
) -> None:
    """成功工具调用应登记 usable=True 的证据块，文本来自 as_prompt_block 的安全投影。"""

    def query_cpu(hostname: str) -> str:
        return f"{hostname} cpu=83%"

    manager = _evidence_manager()
    manager.register(manager.wrap_local_callable("query_cpu", query_cpu))

    await manager.ainvoke("query_cpu", {"hostname": "api-1"}, fake_request_context)

    blocks = manager.drain_request_evidence(fake_request_context.request_id)
    assert len(blocks) == 1
    assert blocks[0].tool_name == "query_cpu"
    assert blocks[0].usable is True
    assert "调用成功" in blocks[0].text
    assert "cpu=83%" in blocks[0].text
    # drain 是取走语义：同一请求第二次 drain 不重复返回。
    assert manager.drain_request_evidence(fake_request_context.request_id) == []


@pytest.mark.asyncio
async def test_error_tool_call_records_unusable_evidence_block(
    fake_request_context,
) -> None:
    """失败工具调用以“不可用声明”入链，错误细节不能出现在证据文本中。"""

    def broken_tool() -> str:
        raise RuntimeError("api_key=secret-token leaked")

    manager = _evidence_manager()
    manager.register(manager.wrap_local_callable("broken_tool", broken_tool))

    await manager.ainvoke("broken_tool", {}, fake_request_context)

    blocks = manager.drain_request_evidence(fake_request_context.request_id)
    assert len(blocks) == 1
    assert blocks[0].tool_name == "broken_tool"
    assert blocks[0].usable is False
    assert "不能作为事实依据" in blocks[0].text
    assert "secret-token" not in blocks[0].text


@pytest.mark.asyncio
async def test_evidence_text_is_truncated_to_configured_chars(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证据块文本必须受 critic_evidence_max_chars 约束，防止撑爆 checkpointer。"""

    monkeypatch.setattr(config, "critic_evidence_max_chars", 20, raising=False)

    def big_tool() -> str:
        return "x" * 500

    manager = _evidence_manager()
    manager.register(manager.wrap_local_callable("big_tool", big_tool))

    await manager.ainvoke("big_tool", {}, fake_request_context)

    blocks = manager.drain_request_evidence(fake_request_context.request_id)
    assert len(blocks) == 1
    assert len(blocks[0].text) <= 20


@pytest.mark.asyncio
async def test_evidence_collection_can_be_disabled(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """critic_evidence_enabled=false 时不再登记证据，作为紧急回滚开关。"""

    monkeypatch.setattr(config, "critic_evidence_enabled", False, raising=False)

    def ok_tool() -> str:
        return "ok"

    manager = _evidence_manager()
    manager.register(manager.wrap_local_callable("ok_tool", ok_tool))

    await manager.ainvoke("ok_tool", {}, fake_request_context)

    assert manager.drain_request_evidence(fake_request_context.request_id) == []


@pytest.mark.asyncio
async def test_evidence_blocks_are_isolated_by_request_id(
    fake_request_context,
) -> None:
    """证据按 request_id 隔离：一个请求的 drain 不能取走另一个请求的证据。"""

    from dataclasses import replace

    other_ctx = replace(fake_request_context, request_id="req_other")

    def ok_tool() -> str:
        return "ok"

    manager = _evidence_manager()
    manager.register(manager.wrap_local_callable("ok_tool", ok_tool))

    await manager.ainvoke("ok_tool", {}, fake_request_context)
    await manager.ainvoke("ok_tool", {}, other_ctx)

    assert len(manager.drain_request_evidence(fake_request_context.request_id)) == 1
    assert len(manager.drain_request_evidence(other_ctx.request_id)) == 1


class _FakeChatQwen:
    """捕获 bind_tools 的最小 ChatQwen fake。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    def bind_tools(self, tools: list[object], **kwargs: object) -> "_FakeChatQwen":
        _ = tools, kwargs
        return self

    async def ainvoke(
        self, messages: list[object], config: dict[str, object] | None = None
    ) -> SimpleNamespace:
        # executor 现在传入 usage 采集 callbacks；fake 容忍 config 参数。
        _ = messages, config
        return SimpleNamespace(content="执行完成", tool_calls=[])


class _BrokenLLM:
    """模拟 provider 故障，验证失败路径仍会 drain 已收集证据。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    def bind_tools(self, tools: list[object], **kwargs: object) -> "_BrokenLLM":
        _ = tools, kwargs
        return self

    async def ainvoke(
        self, messages: list[object], config: dict[str, object] | None = None
    ) -> object:
        _ = messages, config
        raise RuntimeError("provider down")


class _FakeToolNode:
    """不执行真实工具的 ToolNode fake，只保留构造签名兼容。"""

    def __init__(self, tools: list[object]) -> None:
        _ = tools

    async def ainvoke(self, payload: dict[str, object]) -> dict[str, list[object]]:
        _ = payload
        return {"messages": []}


class _FakeMcpTool:
    """最小 MCP 工具 fake，让 executor 装配阶段有远端工具可 wrap。"""

    name = "query_cpu_metrics"
    description = "查询 CPU 指标"
    args_schema = None

    async def ainvoke(self, args: dict[str, object]) -> str:
        _ = args
        return "fake metric"


def _patch_executor_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: object,
    manager: ToolManager,
) -> None:
    """把 executor 依赖替换为 fake，并强制走 ToolManager 装配路径。

    get_tool_manager 同步指向测试自建的 manager（policy 关闭），保证 executor 装配
    与 _drain_step_evidence 看到的是同一个证据收集器。
    """

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [_FakeMcpTool()]

    monkeypatch.setattr(executor_module, "ChatQwen", llm)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(
        executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry
    )
    monkeypatch.setattr(executor_module, "get_tool_manager", lambda: manager)
    monkeypatch.setattr(config, "tool_manager_enabled", True, raising=False)
    monkeypatch.setattr(config, "tool_choice_enabled", False, raising=False)


def _register_metric_tool(manager: ToolManager) -> None:
    """注册返回固定 CPU 指标的本地工具，用于制造可用证据。"""

    def query_cpu_metrics() -> str:
        return "cpu=83%"

    manager.register(
        manager.wrap_local_callable("query_cpu_metrics", query_cpu_metrics)
    )


@pytest.mark.asyncio
async def test_executor_drains_step_evidence_into_state_update(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """executor 应把本步骤工具证据 drain 出来，标注 step 后写入 tool_evidence。"""

    manager = _evidence_manager()
    _register_metric_tool(manager)
    _patch_executor_env(monkeypatch, llm=_FakeChatQwen, manager=manager)

    # 模拟本步骤内工具已执行：证据登记进 request 维度收集器。
    await manager.ainvoke("query_cpu_metrics", {}, fake_request_context)

    token = set_request_context(fake_request_context)
    try:
        result = await importlib.import_module("app.agent.aiops.executor").executor(
            {"plan": ["检查 CPU"], "past_steps": []}
        )
    finally:
        reset_request_context(token)

    entries = result.get("tool_evidence", [])
    assert len(entries) == 1
    assert entries[0]["step"] == "检查 CPU"
    assert entries[0]["tool_name"] == "query_cpu_metrics"
    assert entries[0]["usable"] is True
    assert "cpu=83%" in entries[0]["text"]
    # 旧字段行为不变。
    assert result["plan"] == []
    assert result["past_steps"][0][0] == "检查 CPU"


@pytest.mark.asyncio
async def test_executor_failure_path_still_drains_evidence(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """步骤失败（LLM 异常）时，失败前已执行的工具证据仍应进入 state。"""

    manager = _evidence_manager()
    _register_metric_tool(manager)
    _patch_executor_env(monkeypatch, llm=_BrokenLLM, manager=manager)

    await manager.ainvoke("query_cpu_metrics", {}, fake_request_context)

    token = set_request_context(fake_request_context)
    try:
        result = await importlib.import_module("app.agent.aiops.executor").executor(
            {"plan": ["查询指标"], "past_steps": []}
        )
    finally:
        reset_request_context(token)

    assert result["tool_evidence"][0]["step"] == "查询指标"
    assert result["tool_evidence"][0]["usable"] is True
    assert "内部工具执行失败" in result["past_steps"][0][1]


@pytest.mark.asyncio
async def test_executor_evidence_drain_failure_is_fail_open(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drain 自身异常必须 fail-open：不写证据字段，也不影响步骤执行结果。"""

    manager = _evidence_manager()
    _patch_executor_env(monkeypatch, llm=_FakeChatQwen, manager=manager)

    def _broken_drain(request_id: str) -> object:
        _ = request_id
        raise RuntimeError("drain broken")

    monkeypatch.setattr(manager, "drain_request_evidence", _broken_drain)

    token = set_request_context(fake_request_context)
    try:
        result = await importlib.import_module("app.agent.aiops.executor").executor(
            {"plan": ["检查 CPU"], "past_steps": []}
        )
    finally:
        reset_request_context(token)

    assert "tool_evidence" not in result
    assert result["plan"] == []
    assert result["past_steps"][0][1] == "执行完成"


@pytest.mark.asyncio
async def test_executor_skips_evidence_when_tool_manager_disabled(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool_manager_enabled=false 回滚路径：state 更新不携带 tool_evidence 字段。"""

    manager = _evidence_manager()
    _register_metric_tool(manager)
    _patch_executor_env(monkeypatch, llm=_FakeChatQwen, manager=manager)
    monkeypatch.setattr(config, "tool_manager_enabled", False, raising=False)

    token = set_request_context(fake_request_context)
    try:
        result = await importlib.import_module("app.agent.aiops.executor").executor(
            {"plan": ["检查 CPU"], "past_steps": []}
        )
    finally:
        reset_request_context(token)

    assert "tool_evidence" not in result


@pytest.mark.asyncio
async def test_executor_without_request_context_skips_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无请求上下文（旧测试/离线调用）时 drain 静默返回空，不写 tool_evidence。"""

    manager = _evidence_manager()
    _patch_executor_env(monkeypatch, llm=_FakeChatQwen, manager=manager)

    result = await importlib.import_module("app.agent.aiops.executor").executor(
        {"plan": ["检查 CPU"], "past_steps": []}
    )

    assert "tool_evidence" not in result


@pytest.mark.asyncio
async def test_executor_records_evidence_trace_event(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """ISSUE-C：executor drain 证据后写 agent.executor.evidence 事件，只记数量。"""

    trace_path = tmp_path / "executor_trace.jsonl"
    executor_module = importlib.import_module("app.agent.aiops.executor")
    monkeypatch.setattr(
        executor_module,
        "_trace_logger",
        TraceLogger(
            trace_jsonl_path=str(trace_path),
            enabled=True,
            metrics_enabled=False,
        ),
    )

    manager = _evidence_manager()
    _register_metric_tool(manager)
    _patch_executor_env(monkeypatch, llm=_FakeChatQwen, manager=manager)

    await manager.ainvoke("query_cpu_metrics", {}, fake_request_context)

    token = set_request_context(fake_request_context)
    try:
        await executor_module.executor({"plan": ["检查 CPU"], "past_steps": []})
    finally:
        reset_request_context(token)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    evidence_events = [e for e in events if e["name"] == "agent.executor.evidence"]
    assert len(evidence_events) == 1
    assert evidence_events[0]["block_count"] == 1
    assert evidence_events[0]["usable_count"] == 1

    # 步骤文本与证据正文不落 trace（与 critic trace 的内容原则一致）。
    raw = trace_path.read_text(encoding="utf-8")
    assert "cpu=83%" not in raw
    assert "检查 CPU" not in raw


def test_critic_evidence_settings_have_stable_defaults() -> None:
    """证据链配置必须进入 Settings，默认开启且截断上限稳定。"""

    from app.config import Settings

    settings = Settings()

    assert settings.critic_evidence_enabled is True
    assert settings.critic_evidence_max_chars == 400


def test_state_declares_accumulating_tool_evidence_channel() -> None:
    """PlanExecuteState 必须声明 operator.add 累加的 tool_evidence 通道。"""

    import operator
    from typing import get_args

    from app.agent.aiops.state import PlanExecuteState

    hints = PlanExecuteState.__annotations__

    assert "tool_evidence" in hints
    # 与 past_steps 一致：累加语义保证多步证据不互相覆盖。
    assert get_args(hints["tool_evidence"])[1] is operator.add
    assert get_args(hints["past_steps"])[1] is operator.add
