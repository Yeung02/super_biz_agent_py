"""ISSUE-010 ToolManager 与 Agent 边界回归测试。

这些用例只补阶段 1B 的测试保护，不新增生产能力。测试通过 fake local tool、
fake MCP tool、fake graph 和 fake SSE client 覆盖工具错误不可作为事实证据、
Agent recursion_limit、SSE error schema 与 AIOps complete/done 兼容，避免后续
改造把工具错误、流式中断或旧前端兼容关系悄悄破坏。
"""

from __future__ import annotations

import json

import pytest

from app.agent.policies import PolicyRegistry
from app.agent.tool_manager import ToolManager, ToolResult
from app.core.request_context import RequestContext
from app.observability.tracing import TraceLogger


def _tool_manager_without_policy(**kwargs: object) -> ToolManager:
    """构造只验证工具边界的 ToolManager。

    ISSUE-010 的重点是回归保护，不重新测试权限矩阵；关闭 policy 可避免 fake 工具名
    被默认 allowlist 拦截，从而让用例聚焦 timeout、错误 envelope、裁剪和证据隔离。
    """

    return ToolManager(policy_registry=PolicyRegistry(enabled=False), **kwargs)


def _usable_evidence_blocks(results: list[ToolResult]) -> list[str]:
    """模拟后续事实证据收集，只接收 ToolResult 明确标记可用的结果。

    阶段 3B 才会实现真正 ContextBuilder；这里故意只在测试内放一个最小收集器，避免
    当前 issue 提前引入 RAG context 模块，同时锁住 1B 契约：错误工具结果不能进入事实证据链。
    """

    return [result.as_prompt_block() for result in results if result.is_evidence_usable()]


@pytest.mark.asyncio
async def test_fake_local_tool_matrix_covers_success_error_timeout_and_large_json(
    fake_local_tool,
    fake_request_context: RequestContext,
) -> None:
    """fake local tool 必须覆盖 1B 关键边界，且错误/超时结果不能作为证据。"""

    manager = _tool_manager_without_policy(
        timeout_seconds=0.01,
        max_result_chars=160,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    for mode in ("success", "error", "timeout", "large_json"):
        manager.register(manager.wrap_local_callable(f"local_{mode}", fake_local_tool(mode=mode)))

    success = await manager.ainvoke("local_success", {"service": "orders"}, fake_request_context)
    error = await manager.ainvoke("local_error", {"service": "orders"}, fake_request_context)
    timeout = await manager.ainvoke("local_timeout", {"service": "orders"}, fake_request_context)
    large_json = await manager.ainvoke("local_large_json", {}, fake_request_context)

    assert success.status == "success"
    assert success.is_evidence_usable() is True
    assert error.error_info is not None
    assert error.error_info.code == "TOOL_EXECUTION_ERROR"
    assert error.is_evidence_usable() is False
    assert timeout.error_info is not None
    assert timeout.error_info.code == "TOOL_TIMEOUT"
    assert timeout.is_evidence_usable() is False
    assert large_json.status == "success"
    assert large_json.trimmed is True
    assert isinstance(large_json.data, str)
    assert len(large_json.data) <= 160


@pytest.mark.asyncio
async def test_tool_errors_are_excluded_from_fact_evidence(
    fake_local_tool,
    fake_mcp_tool,
    fake_request_context: RequestContext,
) -> None:
    """本地异常、MCP isError 和超时只能进入诊断说明，不能进入事实证据列表。"""

    manager = _tool_manager_without_policy(
        timeout_seconds=0.01,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(
        manager.wrap_local_callable("healthy_metrics", fake_local_tool(mode="success"))
    )
    manager.register(
        manager.wrap_local_callable("broken_metrics", fake_local_tool(mode="error"))
    )
    manager.register(manager.wrap_mcp_tool("mcp_error", fake_mcp_tool(mode="is_error")))
    manager.register(
        manager.wrap_local_callable("slow_metrics", fake_local_tool(mode="timeout"))
    )

    results = [
        await manager.ainvoke("healthy_metrics", {"service": "orders"}, fake_request_context),
        await manager.ainvoke("broken_metrics", {"service": "orders"}, fake_request_context),
        await manager.ainvoke("mcp_error", {"service": "orders"}, fake_request_context),
        await manager.ainvoke("slow_metrics", {"service": "orders"}, fake_request_context),
    ]
    evidence_blocks = _usable_evidence_blocks(results)
    diagnostic_blocks = [result.as_prompt_block() for result in results if result.is_error]

    assert len(evidence_blocks) == 1
    assert "healthy evidence" in evidence_blocks[0]
    assert all("不能作为事实依据" in block for block in diagnostic_blocks)
    assert "raw secret" not in json.dumps(evidence_blocks + diagnostic_blocks, ensure_ascii=False)
    assert "mcp structured raw failure" not in json.dumps(
        evidence_blocks + diagnostic_blocks,
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_fake_graph_recursion_limit_maps_to_agent_error(
    fake_graph,
    fake_request_context: RequestContext,
) -> None:
    """fake graph 要能模拟 LangGraph recursion_limit，便于 Agent 超限测试不依赖真实图。"""

    events = [
        event
        async for event in fake_graph(mode="recursion_limit").stream(
            session_id=fake_request_context.session_id or "default",
            recursion_limit=1,
            trace_id=fake_request_context.trace_id,
            request_id=fake_request_context.request_id,
        )
    ]

    assert events == [
        {
            "type": "error",
            "error": {
                "code": "AGENT_MAX_STEP_EXCEEDED",
                "message": "任务步骤过多，已停止继续执行。",
                "retryable": False,
            },
            "trace_id": "trc_test",
            "request_id": "req_test",
            "session_id": "session-test",
        }
    ]


@pytest.mark.asyncio
async def test_fake_sse_client_locks_message_event_and_done_compatibility(
    fake_sse_client,
    fake_request_context: RequestContext,
) -> None:
    """SSE fake 要锁住旧 `event: message` 与 AIOps `complete -> done` 兼容。"""

    client = fake_sse_client(fake_request_context)
    response = client.response_from_payloads(
        [
            {"type": "start", "session_id": "session-test"},
            {"type": "complete", "diagnosis": {"status": "completed"}},
        ],
        mirror_complete_as_done=True,
    )

    events = await client.collect(response)

    assert [event["event"] for event in events] == ["message", "message", "message"]
    assert [event["payload"]["type"] for event in events] == ["start", "complete", "done"]
    assert all(event["payload"]["trace_id"] == "trc_test" for event in events)
    assert all(event["payload"]["request_id"] == "req_test" for event in events)


@pytest.mark.asyncio
async def test_fake_sse_client_builds_safe_error_event(
    fake_sse_client,
    fake_request_context: RequestContext,
) -> None:
    """SSE 中断只能暴露稳定错误码、trace/request 和安全文案。"""

    client = fake_sse_client(fake_request_context)
    response = client.response_from_error(
        code="SSE_STREAM_INTERRUPTED",
        message="流式响应中断。",
        raw_error="raw secret token from http://internal.service",
    )

    events = await client.collect(response)
    payload = events[0]["payload"]
    serialized = json.dumps(payload, ensure_ascii=False)

    assert events[0]["event"] == "message"
    assert payload["type"] == "error"
    assert payload["error"]["code"] == "SSE_STREAM_INTERRUPTED"
    assert payload["trace_id"] == "trc_test"
    assert payload["request_id"] == "req_test"
    assert "raw secret" not in serialized
    assert "http://internal.service" not in serialized
