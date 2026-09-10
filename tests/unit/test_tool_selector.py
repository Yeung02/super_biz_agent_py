"""ToolSelector 按任务相关性筛选工具子集的单元测试。

只验证纯词法筛选行为，不接入 Agent、MCP 或 ToolManager。核心契约：
- 无命中回退全量工具（行为不劣于关闭状态）；
- always_include 工具无条件保留；
- 工具名直接出现在步骤文本中得分最高；
- max_tools 截断保持确定性顺序。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.tool_selector import ToolSelector, tokenize


@dataclass
class _FakeTool:
    name: str
    description: str


def test_tokenize_splits_ascii_words_and_cjk_bigrams() -> None:
    tokens = tokenize("检查 CPU 使用率 Query")

    assert "cpu" in tokens
    assert "query" in tokens
    assert "检查" in tokens
    assert "使用" in tokens
    assert "用率" in tokens


def test_select_returns_all_tools_when_nothing_matches() -> None:
    tools = [
        _FakeTool("query_cpu_metrics", "查询 CPU 指标"),
        _FakeTool("search_log", "检索日志"),
    ]
    selector = ToolSelector(max_tools=8, always_include=[])

    selected = selector.select("今天天气怎么样", tools)

    assert selected == tools


def test_select_returns_all_tools_for_empty_query() -> None:
    tools = [_FakeTool("query_cpu_metrics", "查询 CPU 指标")]
    selector = ToolSelector(max_tools=8, always_include=[])

    assert selector.select("", tools) == tools
    assert selector.select("   ", tools) == tools


def test_select_keeps_only_matching_tools() -> None:
    tools = [
        _FakeTool("query_cpu_metrics", "查询 CPU 指标"),
        _FakeTool("search_log", "检索系统日志"),
    ]
    selector = ToolSelector(max_tools=8, always_include=[])

    selected = selector.select("检查 CPU 使用率", tools)

    assert [tool.name for tool in selected] == ["query_cpu_metrics"]


def test_select_gives_highest_score_to_explicit_tool_name() -> None:
    tools = [
        _FakeTool("search_log", "检索 CPU 相关日志"),
        _FakeTool("query_cpu_metrics", "查询 CPU 指标"),
    ]
    selector = ToolSelector(max_tools=1, always_include=[])

    selected = selector.select("使用 query_cpu_metrics 查询", tools)

    # 文本中直接写出工具名的工具必须排第一，即使另一个工具描述也有 CPU。
    assert [tool.name for tool in selected] == ["query_cpu_metrics"]


def test_select_always_includes_core_tools() -> None:
    tools = [
        _FakeTool("retrieve_knowledge", "知识库检索"),
        _FakeTool("get_current_time", "获取当前时间"),
        _FakeTool("query_cpu_metrics", "查询 CPU 指标"),
    ]
    selector = ToolSelector(max_tools=8, always_include=["retrieve_knowledge"])

    selected = selector.select("检查 CPU 使用率", tools)

    # always_include 工具无条件保留（排最前），未命中的普通工具被筛除。
    assert [tool.name for tool in selected] == [
        "retrieve_knowledge",
        "query_cpu_metrics",
    ]


def test_select_caps_tools_by_max_tools_with_stable_order() -> None:
    tools = [
        _FakeTool("tool_a", "cpu cpu"),
        _FakeTool("tool_b", "cpu"),
        _FakeTool("tool_c", "cpu cpu cpu"),
    ]
    selector = ToolSelector(max_tools=2, always_include=[])

    selected = selector.select("cpu", tools)

    # 同分保持原始顺序：a/b/c 描述均含 cpu，截断为前 2 个。
    assert [tool.name for tool in selected] == ["tool_a", "tool_b"]


def test_select_split_preserves_source_groups() -> None:
    local = [_FakeTool("get_current_time", "获取当前时间")]
    mcp = [
        _FakeTool("query_cpu_metrics", "查询 CPU 指标"),
        _FakeTool("search_log", "检索日志"),
    ]
    selector = ToolSelector(max_tools=8, always_include=[])

    selected_local, selected_mcp = selector.select_split("检查 CPU", local, mcp)

    # 纯词法筛选：本地时间工具与步骤无关被筛除，MCP 只保留 CPU 指标工具。
    assert selected_local == []
    assert [tool.name for tool in selected_mcp] == ["query_cpu_metrics"]
