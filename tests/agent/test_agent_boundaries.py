"""Agent 与工具边界测试。

前半部分保留 ISSUE-008 的工具装配回归：RAG Agent 和 AIOps Executor 交给
LangChain 的工具必须是 ToolManager wrapper；关闭 `tool_manager_enabled` 时仍保留
旧工具列表回滚路径。

ISSUE-009 在同一文件继续补 Agent 执行上限：配置必须可见、AIOps graph 必须携带
`recursion_limit`，ToolManager 必须按 request 限制工具调用次数，非流式 Chat 必须
受请求整体 timeout 保护。所有测试都使用 fake，不连接真实 Milvus、DashScope 或 MCP。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.policies import PolicyRegistry
from app.config import Settings
from app.core.request_context import RequestContext, reset_request_context, set_request_context
from app.observability.tracing import TraceLogger


@dataclass
class _FakeTool:
    """最小 LangChain-like tool fake，用来模拟 MCP client 返回工具。"""

    name: str = "query_cpu_metrics"
    description: str = "查询 CPU 指标"
    args_schema: object | None = None

    async def ainvoke(self, args: dict[str, object]) -> str:
        _ = args
        return "fake metric"


class _FakeMCPClient:
    """不启动 MCP server 的 client fake。"""

    def __init__(self, tools: list[object]) -> None:
        self._tools = tools

    async def get_tools(self) -> list[object]:
        return self._tools


class _FakeChatQwen:
    """捕获 `bind_tools` 入参的 ChatQwen fake。"""

    captured_tools: list[object] = []
    captured_tool_choice: object = "unset"

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    def bind_tools(self, tools: list[object], **kwargs: object) -> _FakeChatQwen:
        self.__class__.captured_tools = tools
        # executor 会对同一工具集做两次 bind_tools（首轮带 tool_choice、总结轮不带）；
        # 只在显式传入 tool_choice 时记录，避免后一次调用覆盖首轮的强制指定。
        if "tool_choice" in kwargs:
            self.__class__.captured_tool_choice = kwargs["tool_choice"]
        return self

    async def ainvoke(
        self, messages: list[object], config: dict[str, object] | None = None
    ) -> _FakeLLMMessage:
        # executor 现在以 config={"callbacks": [UsageAccumulator]} 传入真实 usage
        # 采集器；fake 只需容忍该参数。
        _ = messages, config
        return _FakeLLMMessage(content="执行完成", tool_calls=[])


@dataclass
class _FakeLLMMessage:
    """AIOps executor 所需的最小 LLM message fake。"""

    content: str
    tool_calls: list[dict[str, object]]


class _FakeToolNode:
    """捕获 ToolNode 初始化工具，避免执行真实 LangGraph ToolNode。"""

    captured_tools: list[object] = []

    def __init__(self, tools: list[object]) -> None:
        self.__class__.captured_tools = tools

    async def ainvoke(self, payload: dict[str, object]) -> dict[str, list[object]]:
        _ = payload
        return {"messages": []}


class _FakeHttpRequest:
    """API handler 测试用请求对象，只提供当前 handler 实际访问的属性。"""

    def __init__(self, ctx: RequestContext) -> None:
        self.state = SimpleNamespace(ctx=ctx)

    async def is_disconnected(self) -> bool:
        return False


class _FakeGraph:
    """捕获 LangGraph astream config，避免构建真实 planner/executor/replanner。"""

    def __init__(self) -> None:
        self.last_config: dict[str, object] | None = None

    async def astream(
        self,
        *,
        input: dict[str, object],
        config: dict[str, object],
        stream_mode: str,
    ):
        _ = input, stream_mode
        self.last_config = config
        yield {"planner": {"plan": ["检查告警"]}}

    def get_state(self, config: dict[str, object]) -> SimpleNamespace:
        _ = config
        return SimpleNamespace(values={"response": "诊断完成"})


class _RecursionLimitGraph:
    """模拟 LangGraph recursion_limit 触发后的异常形态。"""

    async def astream(
        self,
        *,
        input: dict[str, object],
        config: dict[str, object],
        stream_mode: str,
    ):
        _ = input, config, stream_mode
        raise RuntimeError("Recursion limit of 1 reached without hitting a stop condition")
        yield {}

    def get_state(self, config: dict[str, object]) -> SimpleNamespace:
        _ = config
        return SimpleNamespace(values={})


class _SlowGraph:
    """Graph fake that only emits after the request deadline has expired."""

    async def astream(
        self,
        *,
        input: dict[str, object],
        config: dict[str, object],
        stream_mode: str,
    ):
        _ = input, config, stream_mode
        await asyncio.sleep(0.03)
        yield {"planner": {"plan": ["late step"]}}

    def get_state(self, config: dict[str, object]) -> SimpleNamespace:
        _ = config
        return SimpleNamespace(values={"response": "late"})


@pytest.mark.asyncio
async def test_rag_agent_initializes_wrapped_tools_when_tool_manager_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG Agent 初始化时应把本地和 MCP 工具统一替换成 ToolManager wrapper。"""

    from app.services import rag_agent_service as rag_module

    captured: dict[str, list[object]] = {}
    mcp_tool = _FakeTool()

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [mcp_tool]

    def _fake_create_agent(model: object, *, tools: list[object], checkpointer: object) -> object:
        _ = model, checkpointer
        captured["tools"] = tools
        return object()

    monkeypatch.setattr(rag_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(rag_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(rag_module, "create_agent", _fake_create_agent)
    monkeypatch.setattr(rag_module.config, "tool_manager_enabled", True, raising=False)

    service = rag_module.RagAgentService(streaming=False)
    await service._initialize_agent()

    tools = captured["tools"]

    assert [getattr(tool, "name", "") for tool in tools] == [
        "retrieve_knowledge",
        "get_current_time",
        "query_cpu_metrics",
    ]
    assert all(getattr(tool, "metadata", {}).get("tool_manager_wrapped") for tool in tools)


@pytest.mark.asyncio
async def test_rag_agent_keeps_raw_tools_when_tool_manager_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭 ToolManager 时必须保留旧 `local_tools + mcp_tools` 回滚路径。"""

    from app.services import rag_agent_service as rag_module

    captured: dict[str, list[object]] = {}
    mcp_tool = _FakeTool()

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [mcp_tool]

    def _fake_create_agent(model: object, *, tools: list[object], checkpointer: object) -> object:
        _ = model, checkpointer
        captured["tools"] = tools
        return object()

    monkeypatch.setattr(rag_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(rag_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(rag_module, "create_agent", _fake_create_agent)
    monkeypatch.setattr(rag_module.config, "tool_manager_enabled", False, raising=False)

    service = rag_module.RagAgentService(streaming=False)
    await service._initialize_agent()

    assert captured["tools"] == [*service.tools, mcp_tool]


@pytest.mark.asyncio
async def test_aiops_executor_binds_and_executes_wrapped_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIOps Executor 传给 `bind_tools` 和 `ToolNode` 的工具都必须经过 ToolManager。"""

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [_FakeTool()]

    monkeypatch.setattr(executor_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(executor_module.config, "tool_manager_enabled", True, raising=False)
    _FakeChatQwen.captured_tools = []
    _FakeToolNode.captured_tools = []

    result = await executor_module.executor({"plan": ["检查 CPU"], "past_steps": []})

    bind_tools = _FakeChatQwen.captured_tools
    tool_node_tools = _FakeToolNode.captured_tools

    assert result["plan"] == []
    assert [getattr(tool, "name", "") for tool in bind_tools] == [
        "get_current_time",
        "retrieve_knowledge",
        "query_cpu_metrics",
    ]
    assert [getattr(tool, "name", "") for tool in tool_node_tools] == [
        "get_current_time",
        "retrieve_knowledge",
        "query_cpu_metrics",
    ]
    assert all(getattr(tool, "metadata", {}).get("tool_manager_wrapped") for tool in bind_tools)
    assert all(
        getattr(tool, "metadata", {}).get("tool_manager_wrapped") for tool in tool_node_tools
    )


def test_agent_execution_limit_settings_have_stable_defaults() -> None:
    """ISSUE-009 的执行边界必须进入配置，不能散落成硬编码常量。"""

    settings = Settings()

    assert settings.agent_max_steps == 8
    assert settings.agent_recursion_limit == 30
    assert settings.agent_max_tool_calls == 12
    assert settings.tool_timeout_ms == 15_000
    assert settings.tool_timeout_seconds == 15


@pytest.mark.asyncio
async def test_aiops_executor_forces_unique_tool_from_step_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """步骤文本唯一命中一个工具时，首轮 LLM 调用必须带 tool_choice 强制指定。"""

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [_FakeTool()]

    monkeypatch.setattr(executor_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(executor_module.config, "tool_manager_enabled", True, raising=False)
    monkeypatch.setattr(executor_module.config, "tool_choice_enabled", True, raising=False)
    _FakeChatQwen.captured_tools = []
    _FakeChatQwen.captured_tool_choice = None

    await executor_module.executor({"plan": ["使用 query_cpu_metrics 查询 CPU"], "past_steps": []})

    assert _FakeChatQwen.captured_tool_choice == {"type": "tool", "name": "query_cpu_metrics"}


@pytest.mark.asyncio
async def test_aiops_executor_keeps_auto_choice_for_ambiguous_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """步骤未命中或多命中工具名时保持 LLM 自主选择，不强制 tool_choice。"""

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [_FakeTool()]

    monkeypatch.setattr(executor_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(executor_module.config, "tool_manager_enabled", True, raising=False)
    monkeypatch.setattr(executor_module.config, "tool_choice_enabled", True, raising=False)
    _FakeChatQwen.captured_tools = []
    _FakeChatQwen.captured_tool_choice = None

    await executor_module.executor({"plan": ["分析当前告警并给出结论"], "past_steps": []})

    assert _FakeChatQwen.captured_tool_choice is None


@pytest.mark.asyncio
async def test_aiops_executor_filters_step_tools_by_relevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按步骤文本筛选工具子集：无关 MCP 工具不进入 bind_tools/ToolNode。"""

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [
            _FakeTool(name="query_cpu_metrics", description="查询 CPU 指标"),
            _FakeTool(name="search_log", description="检索系统日志"),
        ]

    monkeypatch.setattr(executor_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(executor_module.config, "tool_manager_enabled", True, raising=False)
    monkeypatch.setattr(executor_module.config, "tool_selection_enabled", True, raising=False)
    monkeypatch.setattr(executor_module.config, "tool_choice_enabled", False, raising=False)
    _FakeChatQwen.captured_tools = []
    _FakeToolNode.captured_tools = []

    await executor_module.executor({"plan": ["检查 CPU 使用率"], "past_steps": []})

    # 默认 always_include 保留两个本地工具；search_log 与步骤无关被筛除。
    assert [getattr(tool, "name", "") for tool in _FakeChatQwen.captured_tools] == [
        "get_current_time",
        "retrieve_knowledge",
        "query_cpu_metrics",
    ]
    assert [getattr(tool, "name", "") for tool in _FakeToolNode.captured_tools] == [
        "get_current_time",
        "retrieve_knowledge",
        "query_cpu_metrics",
    ]


@pytest.mark.asyncio
async def test_aiops_executor_failure_uses_fallback_matrix_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """executor 步骤失败时文案来自 fallback 矩阵，不再是硬编码字符串。"""

    executor_module = importlib.import_module("app.agent.aiops.executor")

    async def _fake_get_mcp_tools_with_retry() -> list[object]:
        return [_FakeTool()]

    class _BrokenLLM:
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

    monkeypatch.setattr(executor_module, "ChatQwen", _BrokenLLM)
    monkeypatch.setattr(executor_module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(executor_module, "get_mcp_tools_with_retry", _fake_get_mcp_tools_with_retry)
    monkeypatch.setattr(executor_module.config, "tool_manager_enabled", True, raising=False)

    result = await executor_module.executor({"plan": ["查询指标"], "past_steps": []})

    assert result["plan"] == []
    step_text = result["past_steps"][0][1]
    # 非工具异常在 executor 边界映射为工具执行错误，走矩阵稳定文案。
    assert step_text == "内部工具执行失败，系统已记录问题。"


@pytest.mark.asyncio
async def test_aiops_execute_passes_recursion_limit_to_langgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIOps graph 执行时必须把配置化 recursion_limit 传给 LangGraph。"""

    from app.services import aiops_service as aiops_module

    fake_graph = _FakeGraph()
    service = aiops_module.AIOpsService.__new__(aiops_module.AIOpsService)
    service.graph = fake_graph
    monkeypatch.setattr(
        aiops_module,
        "config",
        SimpleNamespace(agent_recursion_limit=7),
        raising=False,
    )

    events = [event async for event in service.execute("诊断告警", session_id="session-1")]

    assert fake_graph.last_config is not None
    assert fake_graph.last_config["recursion_limit"] == 7
    assert events[-1]["type"] == "complete"


@pytest.mark.asyncio
async def test_aiops_execute_maps_recursion_limit_error_to_agent_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recursion_limit 触发后要返回稳定错误码，而不是把 LangGraph 异常原文推给用户。"""

    from app.services import aiops_service as aiops_module

    service = aiops_module.AIOpsService.__new__(aiops_module.AIOpsService)
    service.graph = _RecursionLimitGraph()
    monkeypatch.setattr(
        aiops_module,
        "config",
        SimpleNamespace(agent_recursion_limit=1),
        raising=False,
    )

    events = [event async for event in service.execute("诊断告警", session_id="session-1")]

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error"]["code"] == "AGENT_MAX_STEP_EXCEEDED"
    assert events[0]["message"] == "任务步骤过多，已停止继续执行。"
    assert "Recursion limit" not in json.dumps(events[0], ensure_ascii=False)


@pytest.mark.asyncio
async def test_aiops_replanner_max_steps_returns_structured_agent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Business step overflow must stop with AGENT_MAX_STEP_EXCEEDED."""

    replanner_module = importlib.import_module("app.agent.aiops.replanner")

    class _UnexpectedChatQwen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs
            raise AssertionError("max step overflow must not call the LLM")

    monkeypatch.setattr(replanner_module, "ChatQwen", _UnexpectedChatQwen)
    monkeypatch.setattr(replanner_module.config, "agent_max_steps", 1, raising=False)

    result = await replanner_module.replanner(
        {
            "input": "diagnose",
            "plan": ["continue"],
            "past_steps": [("step", "result")],
            "response": "",
        }
    )

    error_event = result["error_event"]

    assert error_event["type"] == "error"
    assert error_event["error"]["code"] == "AGENT_MAX_STEP_EXCEEDED"
    assert "continue" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_aiops_execute_uses_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """AIOps streaming execution must respect the overall request deadline."""

    from app.services import aiops_service as aiops_module

    short_ctx = replace(
        fake_request_context,
        deadline_ms=1,
        started_monotonic=time.monotonic(),
    )
    service = aiops_module.AIOpsService.__new__(aiops_module.AIOpsService)
    service.graph = _SlowGraph()
    monkeypatch.setattr(
        aiops_module,
        "config",
        SimpleNamespace(agent_recursion_limit=7, request_timeout_ms=60_000),
        raising=False,
    )

    token = set_request_context(short_ctx)
    try:
        events = [event async for event in service.execute("diagnose", session_id="session-1")]
    finally:
        reset_request_context(token)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error"]["code"] == "LLM_TIMEOUT"
    assert "late" not in json.dumps(events[0], ensure_ascii=False)


@pytest.mark.asyncio
async def test_tool_manager_limits_tool_calls_per_request(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
    tmp_path: Path,
) -> None:
    """ToolManager 必须按 request_id 计数，超过上限后不能继续执行真实工具。"""

    from app.agent import tool_manager as tool_manager_module

    calls: list[dict[str, object]] = []

    async def _tracked_tool() -> str:
        calls.append({})
        return "ok"

    monkeypatch.setattr(
        tool_manager_module,
        "config",
        SimpleNamespace(
            agent_max_tool_calls=1,
            tool_timeout_seconds=15,
            tool_max_result_chars=12_000,
            trace_jsonl_path=str(tmp_path / "trace.jsonl"),
            trace_enabled=False,
            tool_policy_enabled=False,
            tool_default_allowlist=[],
            tool_policy_default_allowed_tenants=["default"],
            tool_policy_default_allowed_users=["anonymous"],
        ),
    )
    manager = tool_manager_module.ToolManager(
        trace_logger=TraceLogger(
            trace_jsonl_path=str(tmp_path / "trace.jsonl"),
            enabled=False,
        ),
        policy_registry=PolicyRegistry(enabled=False),
    )
    manager.register(manager.wrap_local_callable("unit_tool", _tracked_tool))

    first = await manager.ainvoke("unit_tool", {}, fake_request_context)
    second = await manager.ainvoke("unit_tool", {}, fake_request_context)

    assert first.status == "success"
    assert second.is_error is True
    assert second.error_info is not None
    assert second.error_info.code == "AGENT_MAX_STEP_EXCEEDED"
    assert second.is_evidence_usable() is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_chat_non_streaming_uses_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
    fake_request_context: RequestContext,
) -> None:
    """非流式 Chat 需要整体 timeout，超时后返回安全错误 envelope。"""

    from app.api import chat as chat_api
    from app.models.request import ChatRequest

    async def _slow_query(question: str, session_id: str) -> str:
        _ = question, session_id
        await asyncio.sleep(0.03)
        return "late answer"

    short_ctx = replace(
        fake_request_context,
        deadline_ms=1,
        started_monotonic=time.monotonic(),
    )
    monkeypatch.setattr(chat_api.rag_agent_service, "query", _slow_query)

    response = await chat_api.chat(
        ChatRequest(Id="session-1", Question="CPU 怎么排查？"),
        _FakeHttpRequest(short_ctx),
    )
    assert hasattr(response, "body")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 504
    assert payload["error"]["code"] == "LLM_TIMEOUT"
    assert payload["message"] == "模型响应超时，请稍后重试。"
    assert "late answer" not in json.dumps(payload, ensure_ascii=False)
