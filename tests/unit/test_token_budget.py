"""ISSUE-012 TokenBudgetManager 与 token 预算裁剪测试。

这些用例只验证 token 预算模块本身和最小 trace 副作用，不访问真实 DashScope、
Milvus、MCP server 或网络。测试先描述契约需要的公开 API，再由生产代码补齐实现，
避免把当前实现细节反向写成测试断言。
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

import pytest

from app.core.errors import RequestTooLargeError
from app.core.request_context import RequestContext
from app.observability.tracing import TraceLogger


def _token_budget_module() -> Any:
    """按需导入 token budget 模块，让 RED 阶段表现为明确断言失败。"""

    try:
        return importlib.import_module("app.core.token_budget")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.core.token_budget must exist for ISSUE-012: {exc}")


@dataclass(frozen=True)
class _FakeChunk:
    """RAG chunk 同形对象，避免当前阶段提前依赖 app.rag.models。"""

    chunk_id: str
    content: str
    normalized_score: float
    source_path: str = "docs/runbook.md"


@dataclass(frozen=True)
class _FakeToolResult:
    """ToolResult 同形对象，用于验证工具成功/失败结果的裁剪边界。"""

    tool_name: str
    status: str
    is_error: bool
    data: dict[str, object]

    def is_evidence_usable(self) -> bool:
        return not self.is_error and self.status == "success"


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def test_estimate_tokens_falls_back_when_tokenizer_fails() -> None:
    """tokenizer 不可用时要降级为字符估算，而不是中断主流程。"""

    token_budget = _token_budget_module()

    def _broken_tokenizer(text: str) -> int:
        _ = text
        raise RuntimeError("tokenizer unavailable")

    manager = token_budget.TokenBudgetManager(tokenizer=_broken_tokenizer)

    estimate = manager.estimate_tokens("CPU 使用率 high")

    assert estimate.token_count > 0
    assert estimate.method == "char_fallback"
    assert estimate.estimated is True


def test_allocate_uses_model_window_and_reserved_output(fake_request_context: RequestContext) -> None:
    """预算分配必须保留输出窗口，并按 scenario 比例切分输入槽位。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager(
        model_context_windows={"qwen-test": 1000},
        max_output_tokens=200,
        min_reserved_output_tokens=120,
    )

    budget = manager.allocate("rag_chat", "qwen-test", fake_request_context)

    assert budget.model_context_window == 1000
    assert budget.output_tokens == 200
    assert budget.input_tokens == 120
    # rag_chat 的 history 槽位划出 5% 给用户记忆画像；未召回时槽位闲置。
    assert budget.history_tokens == 120
    assert budget.memory_tokens == 40
    assert budget.rag_context_tokens == 280
    assert budget.tool_result_tokens == 80
    assert budget.summary_tokens <= budget.history_tokens


def test_trim_messages_keeps_system_and_recent_turns() -> None:
    """20 轮长历史不能全部进入 LLM，系统约束和最近完整轮次要优先保留。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager()
    messages = [_message("system", "必须基于事实回答，不要泄露内部信息。")]
    for index in range(20):
        messages.append(_message("user", f"历史问题 {index} " + "x" * 80))
        messages.append(_message("assistant", f"历史回答 {index} " + "y" * 80))

    result = manager.trim_messages(messages, 120)
    kept_messages = result.content

    assert result.trimmed is True
    assert result.dropped_count > 0
    assert len(kept_messages) < len(messages)
    assert kept_messages[0]["role"] == "system"
    assert kept_messages[-1]["content"].startswith("历史回答 19")
    assert kept_messages[-2]["content"].startswith("历史问题 19")


def test_trim_chunks_drops_low_score_and_duplicate_chunks() -> None:
    """RAG chunk 裁剪按相关性和去重进行，不改变检索排序模块本身。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager()
    chunks = [
        _FakeChunk("chunk-low", "低分内容 " + "x" * 120, 0.1),
        _FakeChunk("chunk-keep", "高分内容", 0.95),
        _FakeChunk("chunk-dup", "重复长内容 " + "y" * 80, 0.7),
        _FakeChunk("chunk-dup", "重复短内容", 0.8),
    ]

    result = manager.trim_chunks(chunks, 32)
    kept_ids = [chunk.chunk_id for chunk in result.content]

    assert result.trimmed is True
    assert "chunk-keep" in kept_ids
    assert "chunk-low" not in kept_ids
    assert kept_ids.count("chunk-dup") == 1


def test_trim_tool_result_compacts_large_json_before_truncation() -> None:
    """工具结果应先移除 debug/raw payload 等冗余字段，再按预算截断。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager()
    tool_result = _FakeToolResult(
        tool_name="query_cpu_metrics",
        status="success",
        is_error=False,
        data={
            "status": "ok",
            "summary": "CPU 峰值 91%",
            "items": [{"timestamp": index, "value": index} for index in range(20)],
            "debug": "debug text should be dropped",
            "raw_payload": "x" * 2000,
        },
    )

    result = manager.trim_tool_result(tool_result, 60)
    serialized = json.dumps(result.content, ensure_ascii=False)

    assert result.trimmed is True
    assert "CPU 峰值" in serialized
    assert "debug text should be dropped" not in serialized
    assert "raw_payload" not in serialized


def test_current_question_over_hard_limit_raises_request_too_large() -> None:
    """当前用户问题不能被静默裁剪，超过硬上限必须映射 REQUEST_TOO_LARGE。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager()

    with pytest.raises(RequestTooLargeError) as exc_info:
        manager.validate_current_input("问题" * 500, hard_limit_tokens=20)

    assert exc_info.value.code == "REQUEST_TOO_LARGE"


def test_trim_context_records_priority_order() -> None:
    """组合裁剪的动作顺序必须符合内部契约：debug -> tool -> rag -> history -> summary。"""

    token_budget = _token_budget_module()
    manager = token_budget.TokenBudgetManager()
    budget = manager.allocate("rag_chat", "qwen-max")
    context = token_budget.TokenContext(
        system_prompt="系统安全约束不得裁剪。",
        current_question="当前问题也不得裁剪。",
        debug_notes=("debug note",),
        tool_results=(
            _FakeToolResult(
                tool_name="tool",
                status="success",
                is_error=False,
                data={"summary": "tool summary", "raw_payload": "z" * 1000},
            ),
        ),
        rag_chunks=(_FakeChunk("chunk-low", "low " + "x" * 1000, 0.1),),
        history_messages=tuple(_message("user", f"old {index} " + "h" * 80) for index in range(8)),
        summary="第一句。" + "第二句。" * 80,
    )

    result = manager.trim_context(context, budget.with_limits(history_tokens=40, rag_context_tokens=20, tool_result_tokens=20, summary_tokens=20))

    assert [action.component for action in result.actions] == [
        "debug",
        "tool_result",
        "rag_chunk",
        "history",
        "summary",
    ]
    assert result.context.system_prompt == "系统安全约束不得裁剪。"
    assert result.context.current_question == "当前问题也不得裁剪。"


def test_record_usage_writes_token_usage_trace(
    tmp_path,
    fake_request_context: RequestContext,
) -> None:
    """usage/cost 记录要能进入 trace，字段稳定且不依赖真实 LLM response。"""

    token_budget = _token_budget_module()
    trace_path = tmp_path / "trace.jsonl"
    manager = token_budget.TokenBudgetManager(
        trace_logger=TraceLogger(trace_jsonl_path=str(trace_path), enabled=True),
        token_input_prices_per_1k={"qwen-test": 0.1},
        token_output_prices_per_1k={"qwen-test": 0.2},
    )

    usage = manager.record_usage(
        model="qwen-test",
        input_tokens=1000,
        output_tokens=500,
        ctx=fake_request_context,
        estimated=True,
    )

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event = events[-1]

    assert usage.total_tokens == 1500
    assert usage.estimated_cost == pytest.approx(0.2)
    assert event["name"] == "token.usage"
    assert event["usage"]["input_tokens"] == 1000
    assert event["usage"]["output_tokens"] == 500
    assert event["usage"]["estimated"] is True
