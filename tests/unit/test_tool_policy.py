"""ISSUE-007 ToolPolicy/PolicyRegistry 单元测试。

这些测试先描述工具权限策略的独立边界：默认策略不能破坏现有 demo 工具；显式禁用、
tenant/user 不匹配必须在 ToolManager 调用真实工具前被拦截；未授权结果不能作为事实证据。
测试不接入 RAG/AIOps 主流程，避免提前实现 ISSUE-008。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.agent.policies import PolicyRegistry, ToolPolicy
from app.agent.tool_manager import ToolManager
from app.config import config
from app.observability.tracing import TraceLogger


def test_default_policy_allows_existing_demo_tools_for_default_identity(
    fake_request_context,
) -> None:
    """默认 anonymous/default 必须还能使用现有本地工具和 mock MCP 工具，避免升级后 demo 失效。"""

    registry = PolicyRegistry.from_config(config)

    assert registry.check("retrieve_knowledge", fake_request_context).allowed is True
    assert registry.check("get_current_time", fake_request_context).allowed is True
    assert registry.check("query_cpu_metrics", fake_request_context).allowed is True


@pytest.mark.asyncio
async def test_tool_manager_returns_unauthorized_result_without_calling_tool(
    fake_request_context,
) -> None:
    """ToolManager 必须在 invoke 前检查 policy，未授权时不能执行真实工具函数。"""

    called = False

    def disabled_tool() -> str:
        nonlocal called
        called = True
        return "should not run"

    policy = ToolPolicy(tool_name="disabled_tool", enabled=False)
    manager = ToolManager(
        policy_registry=PolicyRegistry([policy]),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("disabled_tool", disabled_tool))

    result = await manager.ainvoke("disabled_tool", {}, fake_request_context)

    assert called is False
    assert result.status == "unauthorized"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "UNAUTHORIZED_TOOL"
    assert result.is_evidence_usable() is False
    assert "不能作为事实依据" in result.as_prompt_block()


@pytest.mark.asyncio
async def test_tenant_mismatch_is_denied_before_tool_execution(fake_request_context) -> None:
    """tenant 不匹配时拒绝调用；错误结果只保留稳定码，不暴露允许租户列表。"""

    def tenant_tool() -> dict[str, str]:
        return {"secret_policy_detail": "should not leak"}

    policy = ToolPolicy(
        tool_name="tenant_tool",
        allowed_tenants=("tenant-a",),
        allowed_users=("anonymous",),
    )
    manager = ToolManager(
        policy_registry=PolicyRegistry([policy]),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("tenant_tool", tenant_tool))

    result = await manager.ainvoke("tenant_tool", {}, fake_request_context)
    payload = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.status == "unauthorized"
    assert result.error is not None
    assert result.error.message == "工具不可用。"
    assert "tenant-a" not in payload
    assert "secret_policy_detail" not in payload


@pytest.mark.asyncio
async def test_user_mismatch_is_denied(fake_request_context) -> None:
    """user 不匹配时也必须拒绝，给后续多用户边界留出稳定入口。"""

    context = replace(fake_request_context, user_id="user-b")
    manager = ToolManager(
        policy_registry=PolicyRegistry(
            [
                ToolPolicy(
                    tool_name="user_tool",
                    allowed_tenants=("default",),
                    allowed_users=("user-a",),
                )
            ]
        ),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("user_tool", lambda: "ok"))

    result = await manager.ainvoke("user_tool", {}, context)

    assert result.status == "unauthorized"
    assert result.is_evidence_usable() is False


@pytest.mark.asyncio
async def test_disabled_policy_registry_rolls_back_to_allow_all(fake_request_context) -> None:
    """`tool_policy_enabled=false` 的回滚语义是全允许，便于紧急恢复旧工具可见性。"""

    manager = ToolManager(
        policy_registry=PolicyRegistry(
            [ToolPolicy(tool_name="safe_tool", enabled=False)],
            enabled=False,
        ),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("safe_tool", lambda: {"status": "ok"}))

    result = await manager.ainvoke("safe_tool", {}, fake_request_context)

    assert result.status == "success"
    assert result.data == {"status": "ok"}
    assert result.is_evidence_usable() is True


@pytest.mark.asyncio
async def test_policy_max_result_chars_overrides_manager_default(fake_request_context) -> None:
    """策略中的 max_result_chars 应覆盖工具默认裁剪预算，避免单个高风险工具撑爆上下文。"""

    manager = ToolManager(
        max_result_chars=500,
        policy_registry=PolicyRegistry(
            [
                ToolPolicy(
                    tool_name="large_policy_tool",
                    max_result_chars=80,
                )
            ]
        ),
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(
        manager.wrap_local_callable(
            "large_policy_tool",
            lambda: {"items": [{"blob": "x" * 20} for _ in range(20)]},
        )
    )

    result = await manager.ainvoke("large_policy_tool", {}, fake_request_context)

    assert result.status == "success"
    assert result.trimmed is True
    assert isinstance(result.data, str)
    assert len(result.data) <= 80


@pytest.mark.asyncio
async def test_policy_trace_records_allowed_and_denied_events(
    tmp_path: Path,
    fake_request_context,
) -> None:
    """policy 决策必须写 trace，便于后续排查是工具失败还是权限拒绝。"""

    trace_path = tmp_path / "trace.jsonl"
    manager = ToolManager(
        policy_registry=PolicyRegistry(
            [
                ToolPolicy(tool_name="allowed_tool"),
                ToolPolicy(tool_name="denied_tool", enabled=False),
            ]
        ),
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True),
    )
    manager.register(manager.wrap_local_callable("allowed_tool", lambda: "ok"))
    manager.register(manager.wrap_local_callable("denied_tool", lambda: "not ok"))

    await manager.ainvoke("allowed_tool", {}, fake_request_context)
    await manager.ainvoke("denied_tool", {}, fake_request_context)

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_names = [event["name"] for event in events]

    assert "tool.policy.allowed" in event_names
    assert "tool.policy.denied" in event_names
    assert all(event["tenant_id"] == "default" for event in events)
    assert all(event["user_id"] == "anonymous" for event in events)
