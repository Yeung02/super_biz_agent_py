"""ISSUE-006 ToolManager 核心模型与包装单元测试。

这些用例只验证独立工具边界，不接入 RAG Agent、AIOps Executor、权限策略、retry
或 SSE。这样 ISSUE-006 可以单独回滚，同时保证后续 issue 接业务链路时已经有稳定的
ToolResult envelope、timeout、安全错误和裁剪行为作为基础。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.tools import tool

from app.agent.policies import PolicyRegistry
from app.agent.tool_manager import ToolManager, ToolResult
from app.observability.tracing import TraceLogger


def _tool_manager_without_policy(**kwargs: object) -> ToolManager:
    """构造不启用策略的 ToolManager，保持 ISSUE-006 测试只关注工具 envelope。

    ISSUE-007 后生产默认会按 allowlist 拦截未知工具；本文件大量使用临时工具名验证
    timeout、异常和裁剪行为，因此显式关闭 policy，避免把权限测试混进旧基础测试。
    """

    return ToolManager(policy_registry=PolicyRegistry(enabled=False), **kwargs)


def test_tool_result_error_is_never_usable_as_evidence() -> None:
    """错误结果必须从模型层就标记为不可作为事实证据，避免后续 context 拼装误用。"""

    result = ToolResult.error(
        tool_name="query_metrics",
        code="TOOL_EXECUTION_ERROR",
        message="工具调用失败。",
        retryable=True,
    )

    assert result.is_error is True
    assert result.status == "error"
    assert result.is_evidence_usable() is False
    assert "不能作为事实依据" in result.as_prompt_block()
    assert result.to_trace_fields()["evidence_usable"] is False


@pytest.mark.asyncio
async def test_local_callable_success_returns_serializable_tool_result(
    fake_request_context,
) -> None:
    """本地函数成功时应统一包装为可序列化 ToolResult，而不是把原始对象直接塞回 Agent。"""

    def summarize(hostname: str) -> dict[str, str]:
        return {"hostname": hostname, "status": "ok"}

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_local_callable("summarize", summarize))

    result = await manager.ainvoke("summarize", {"hostname": "api-1"}, fake_request_context)

    assert result.status == "success"
    assert result.is_error is False
    assert result.data == {"hostname": "api-1", "status": "ok"}
    assert result.is_evidence_usable() is True
    json.dumps(result.to_dict(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_langchain_tool_uses_ainvoke_interface(fake_request_context) -> None:
    """LangChain BaseTool 包装要保留 invoke/ainvoke 语义，供后续 ToolNode 接入复用。"""

    @tool
    def double(value: int) -> int:
        """Return doubled value."""

        return value * 2

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_langchain_tool(double))

    result = await manager.ainvoke("double", {"value": 3}, fake_request_context)

    assert result.status == "success"
    assert result.data == 6
    assert result.is_evidence_usable() is True


@pytest.mark.asyncio
async def test_local_callable_exception_maps_to_safe_tool_error(
    fake_request_context,
) -> None:
    """工具异常不能把原始异常、内部 URL 或密钥样式内容原样暴露给用户可见字段。"""

    def unsafe_tool() -> str:
        raise RuntimeError("api_key=secret-token http://internal.service.local failed")

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_local_callable("unsafe_tool", unsafe_tool))

    result = await manager.ainvoke("unsafe_tool", {}, fake_request_context)
    payload = result.to_dict()

    assert result.status == "error"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert result.is_evidence_usable() is False
    assert "secret-token" not in json.dumps(payload, ensure_ascii=False)
    assert "internal.service.local" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_async_tool_timeout_maps_to_tool_timeout(fake_request_context) -> None:
    """单工具超时在 ISSUE-006 先映射为 ToolResult，后续 issue 再接入 Agent 上限和 fallback。"""

    async def slow_tool() -> str:
        await asyncio.sleep(1)
        return "late"

    manager = _tool_manager_without_policy(
        timeout_seconds=0.01,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("slow_tool", slow_tool))

    result = await manager.ainvoke("slow_tool", {}, fake_request_context)

    assert result.status == "timeout"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_TIMEOUT"
    assert result.error.retryable is True
    assert result.is_evidence_usable() is False


@pytest.mark.asyncio
async def test_mcp_is_error_result_becomes_non_evidence_tool_result(
    fake_mcp_tool,
    fake_request_context,
) -> None:
    """MCP `isError=true` 必须转成结构化错误，不能像普通 content 一样进入事实证据链。"""

    mcp_tool = fake_mcp_tool(mode="is_error")
    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_mcp_tool("fake_mcp", mcp_tool))

    result = await manager.ainvoke("fake_mcp", {"service": "orders"}, fake_request_context)

    assert result.status == "error"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert result.is_evidence_usable() is False
    assert "fake mcp structured error" not in result.as_prompt_block()


@pytest.mark.asyncio
async def test_builtin_time_tool_failure_is_not_usable_evidence(
    fake_request_context,
) -> None:
    """Legacy local tools that return safe error text must not enter the evidence chain."""

    from app.tools.time_tool import get_current_time

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_langchain_tool(get_current_time))

    result = await manager.ainvoke(
        "get_current_time",
        {"timezone": "Not/A_Real_Timezone_Secret"},
        fake_request_context,
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.status == "error"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert result.is_evidence_usable() is False
    assert "Not/A_Real_Timezone_Secret" not in payload


@pytest.mark.asyncio
async def test_langchain_adapter_preserves_schema_and_returns_safe_prompt_block(
    fake_request_context,
) -> None:
    """LangChain adapter 必须保留工具 schema，并通过 ToolManager 生成安全 prompt 文本。

    ISSUE-008 要把 RAG/AIOps 的工具交给 LangChain Agent 和 ToolNode 消费。这个测试
    先锁定 adapter 的外形：name/description/args_schema 不能丢，否则 `bind_tools`
    无法把正确参数 schema 发给模型；返回文本必须来自 ToolResult，而不是绕过工具边界。
    """

    @tool
    def lookup_runbook(query: str) -> dict[str, str]:
        """查询运行手册。"""

        return {"query": query, "evidence": "cpu runbook"}

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    spec = manager.wrap_langchain_tool(lookup_runbook)

    wrapped = manager.to_langchain_tool(spec, ctx=fake_request_context)
    payload = await wrapped.ainvoke({"query": "cpu"})

    assert wrapped.name == lookup_runbook.name
    assert wrapped.description == lookup_runbook.description
    assert wrapped.args_schema is lookup_runbook.args_schema
    assert "调用成功" in payload
    assert "cpu runbook" in payload


@pytest.mark.asyncio
async def test_langchain_adapter_hides_mcp_is_error_content_from_evidence(
    fake_mcp_tool,
    fake_request_context,
) -> None:
    """MCP isError 经过 LangChain adapter 后仍不能把原始错误文本暴露为事实证据。"""

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    spec = manager.wrap_mcp_tool("query_cpu_metrics", fake_mcp_tool(mode="is_error"))

    wrapped = manager.to_langchain_tool(spec, ctx=fake_request_context)
    payload = await wrapped.ainvoke({"service": "orders"})

    assert wrapped.name == "query_cpu_metrics"
    assert "TOOL_EXECUTION_ERROR" in payload
    assert "不能作为事实依据" in payload
    assert "fake mcp structured error" not in payload


@pytest.mark.asyncio
async def test_large_json_result_is_trimmed_with_trace_metadata(fake_request_context) -> None:
    """大 JSON 先在工具边界裁剪，并保留 raw/preview 尺寸，避免后续 prompt 被原始 payload 撑爆。"""

    def large_tool() -> dict[str, object]:
        return {
            "summary": "many rows",
            "items": [{"value": index, "blob": "x" * 30} for index in range(30)],
        }

    manager = _tool_manager_without_policy(
        max_result_chars=180,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("large_tool", large_tool))

    result = await manager.ainvoke("large_tool", {}, fake_request_context)
    trace_fields = result.to_trace_fields()

    assert result.status == "success"
    assert result.trimmed is True
    assert isinstance(result.data, str)
    assert len(result.data) <= 180
    assert trace_fields["raw_size"] > trace_fields["preview_size"]
    assert trace_fields["trimmed"] is True


@pytest.mark.asyncio
async def test_tool_manager_records_tool_trace_events(
    tmp_path: Path,
    fake_request_context,
) -> None:
    """工具调用 trace 至少包含 start/end/error，后续接 Agent 时才能串联每次工具行为。"""

    def ok_tool() -> str:
        return "ok"

    def bad_tool() -> str:
        raise RuntimeError("boom")

    trace_path = tmp_path / "trace.jsonl"
    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True)
    )
    manager.register(manager.wrap_local_callable("ok_tool", ok_tool))
    manager.register(manager.wrap_local_callable("bad_tool", bad_tool))

    await manager.ainvoke("ok_tool", {}, fake_request_context)
    await manager.ainvoke("bad_tool", {}, fake_request_context)

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_names = [event["name"] for event in events]

    assert "tool.start" in event_names
    assert "tool.end" in event_names
    assert "tool.error" in event_names
    assert all(event["trace_id"] == fake_request_context.trace_id for event in events)


def test_dashscope_embedding_log_mask_does_not_keep_api_key_fragments() -> None:
    """Log-safe API key masking must not retain prefix/suffix fragments."""

    from app.services.vector_embedding_service import DashScopeEmbeddings

    masked = DashScopeEmbeddings._mask_api_key("sk-1234567890abcdef")

    assert masked == "<redacted>"
    assert "sk-12345" not in masked
    assert "cdef" not in masked


@pytest.mark.asyncio
async def test_output_schema_validation_failure_downgrades_to_error(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已注册输出校验器的工具返回契约外数据时，结果必须降级为不可用证据。"""

    from app.config import config as app_config

    monkeypatch.setattr(app_config, "tool_output_schema_enabled", True, raising=False)

    def dirty_tool() -> dict[str, str]:
        return {"unexpected": "shape"}

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_local_callable("dirty_tool", dirty_tool))
    manager.register_output_schema("dirty_tool", lambda data: isinstance(data, str))

    result = await manager.ainvoke("dirty_tool", {}, fake_request_context)

    assert result.status == "error"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert result.is_evidence_usable() is False
    assert result.metadata["reason"] == "output_schema_validation_failed"


@pytest.mark.asyncio
async def test_output_schema_validation_skipped_when_disabled(
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭 tool_output_schema_enabled 后校验层必须完全旁路，保留原成功结果。"""

    from app.config import config as app_config

    monkeypatch.setattr(app_config, "tool_output_schema_enabled", False, raising=False)

    def dirty_tool() -> dict[str, str]:
        return {"unexpected": "shape"}

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_local_callable("dirty_tool", dirty_tool))
    manager.register_output_schema("dirty_tool", lambda data: isinstance(data, str))

    result = await manager.ainvoke("dirty_tool", {}, fake_request_context)

    assert result.status == "success"
    assert result.is_evidence_usable() is True


@pytest.mark.asyncio
async def test_mcp_empty_content_fails_default_output_validation(
    fake_mcp_tool,
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP 成功返回但 content 为空：默认非空校验判失败，不能进入事实证据链。"""

    from app.config import config as app_config

    monkeypatch.setattr(app_config, "tool_output_schema_enabled", True, raising=False)
    monkeypatch.setattr(
        app_config, "tool_output_schema_mcp_enabled", True, raising=False
    )

    mcp_tool = fake_mcp_tool(mode="success", payload="")
    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_mcp_tool("empty_mcp", mcp_tool))

    result = await manager.ainvoke("empty_mcp", {}, fake_request_context)

    assert result.status == "error"
    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert result.is_evidence_usable() is False
    assert result.metadata["reason"] == "output_schema_validation_failed"


@pytest.mark.asyncio
async def test_mcp_empty_content_passes_when_default_check_disabled(
    fake_mcp_tool,
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭 tool_output_schema_mcp_enabled 后回到旧行为：空 content 保持成功结果。"""

    from app.config import config as app_config

    monkeypatch.setattr(
        app_config, "tool_output_schema_mcp_enabled", False, raising=False
    )

    mcp_tool = fake_mcp_tool(mode="success", payload="")
    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_mcp_tool("empty_mcp", mcp_tool))

    result = await manager.ainvoke("empty_mcp", {}, fake_request_context)

    assert result.status == "success"
    assert result.is_evidence_usable() is True


@pytest.mark.asyncio
async def test_mcp_nonempty_content_passes_default_output_validation(
    fake_mcp_tool,
    fake_request_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非空 content 的 MCP 成功结果不受默认校验影响。"""

    from app.config import config as app_config

    monkeypatch.setattr(app_config, "tool_output_schema_enabled", True, raising=False)
    monkeypatch.setattr(
        app_config, "tool_output_schema_mcp_enabled", True, raising=False
    )

    mcp_tool = fake_mcp_tool(mode="success", payload="cpu=83%")
    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_mcp_tool("metrics_mcp", mcp_tool))

    result = await manager.ainvoke("metrics_mcp", {}, fake_request_context)

    assert result.status == "success"
    assert result.is_error is False
    assert result.data == "cpu=83%"
    assert result.is_evidence_usable() is True


@pytest.mark.asyncio
async def test_output_schema_validator_exception_counts_as_failure(
    fake_request_context,
) -> None:
    """校验器自身抛异常时按校验失败处理，不能把脏数据放行进证据链。"""

    def ok_tool() -> str:
        return "ok"

    def broken_validator(data: object) -> bool:
        raise ValueError("validator bug")

    manager = _tool_manager_without_policy(
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False)
    )
    manager.register(manager.wrap_local_callable("ok_tool", ok_tool))
    manager.register_output_schema("ok_tool", broken_validator)

    result = await manager.ainvoke("ok_tool", {}, fake_request_context)

    assert result.is_error is True
    assert result.metadata["reason"] == "output_schema_validation_failed"


@pytest.mark.asyncio
async def test_local_tool_retries_transient_failure_and_succeeds(
    fake_request_context,
) -> None:
    """本地工具瞬时失败时按配置重试；重试成功后返回正常成功结果。"""

    calls: list[int] = []

    def flaky_tool() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"

    manager = _tool_manager_without_policy(
        retry_count=3,
        retry_delay_seconds=0,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("flaky_tool", flaky_tool))

    result = await manager.ainvoke("flaky_tool", {}, fake_request_context)

    assert result.status == "success"
    assert result.data == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_local_tool_retry_exhaustion_returns_stable_error(
    fake_request_context,
) -> None:
    """重试耗尽后返回稳定 TOOL_EXECUTION_ERROR，不暴露原始异常文本。"""

    calls: list[int] = []

    def always_fails() -> str:
        calls.append(1)
        raise RuntimeError("secret-token http://internal.local")

    manager = _tool_manager_without_policy(
        retry_count=2,
        retry_delay_seconds=0,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("always_fails", always_fails))

    result = await manager.ainvoke("always_fails", {}, fake_request_context)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "TOOL_EXECUTION_ERROR"
    assert len(calls) == 2
    assert "secret-token" not in json.dumps(result.to_dict(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_local_tool_timeout_is_not_retried(fake_request_context) -> None:
    """超时不进入重试循环，避免按次数放大延迟。"""

    calls: list[int] = []

    async def slow_tool() -> str:
        calls.append(1)
        await asyncio.sleep(0.05)
        return "late"

    manager = _tool_manager_without_policy(
        timeout_seconds=0.01,
        retry_count=3,
        retry_delay_seconds=0,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_local_callable("slow_tool", slow_tool))

    result = await manager.ainvoke("slow_tool", {}, fake_request_context)

    assert result.status == "timeout"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_mcp_tool_is_not_retried_by_tool_manager(
    fake_mcp_tool,
    fake_request_context,
) -> None:
    """MCP 工具由 client interceptor 重试，ToolManager 必须避免双重重试。"""

    mcp_tool = fake_mcp_tool(mode="error")
    manager = _tool_manager_without_policy(
        retry_count=3,
        retry_delay_seconds=0,
        trace_logger=TraceLogger(trace_jsonl_path="unused.jsonl", enabled=False),
    )
    manager.register(manager.wrap_mcp_tool("flaky_mcp", mcp_tool))

    result = await manager.ainvoke("flaky_mcp", {}, fake_request_context)

    assert result.status == "error"
    assert len(mcp_tool.calls) == 1


def test_get_tool_manager_returns_shared_singleton_with_default_schemas() -> None:
    """共享 ToolManager 进程级复用，并预注册已知本地工具的输出校验器。"""

    from app.agent.tool_manager import (
        get_tool_manager,
        reset_shared_tool_manager,
    )

    reset_shared_tool_manager()
    try:
        first = get_tool_manager()
        second = get_tool_manager()

        assert first is second
        assert "get_current_time" in first._output_schemas
        assert "retrieve_knowledge" in first._output_schemas
    finally:
        reset_shared_tool_manager()


def test_tool_result_trace_fields_expose_fallback_required() -> None:
    """错误 ToolResult 的 fallback_required 信号进入 trace，供排障与 fallback 决策消费。"""

    result = ToolResult.error(
        tool_name="query_metrics",
        code="TOOL_TIMEOUT",
        message="工具调用超时，请稍后重试。",
        retryable=True,
        fallback_required=True,
    )

    assert result.to_trace_fields()["fallback_required"] is True
    success = ToolResult.success("query_metrics", "ok")
    assert success.to_trace_fields()["fallback_required"] is False
