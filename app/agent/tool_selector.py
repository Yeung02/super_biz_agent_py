"""按任务相关性动态筛选工具子集。

该模块只做纯词法打分，不依赖 embedding 服务或向量库：工具数量在几十以内时，
词法匹配（工具名命中、名称分词、description 重叠）已经能显著收缩绑定给 LLM 的
工具 schema，同时避免为工具选择引入新的外部依赖和延迟。

筛选原则：
- 无任何命中时返回全量工具，保证行为不劣于关闭状态；
- always_include 中的核心工具无条件保留；
- 超过 max_tools 时按得分截断，同分保持原始顺序（结果确定性，便于测试）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.config import config

_CJK_RANGE = (
    "\u4e00-\u9fff"  # CJK 统一表意文字
    "\u3400-\u4dbf"  # 扩展 A
    "\uf900-\ufaff"  # 兼容表意文字
)
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(f"[{_CJK_RANGE}]+")


def tokenize(text: str) -> set[str]:
    """把文本拆成词法 token 集合。

    ASCII 连续字母数字作为一个 token（小写化）；CJK 连续段拆成字符二元组
    （bigram），这是中文词法匹配的常规做法，可避免单字过噪、又能命中双字词。
    """

    if not text:
        return set()
    lowered = text.lower()
    tokens = set(_ASCII_WORD_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class ToolSelector:
    """基于词法相关性的工具子集筛选器。"""

    def __init__(
        self,
        *,
        max_tools: int | None = None,
        always_include: Sequence[str] | None = None,
    ) -> None:
        self.max_tools = (
            max_tools
            if max_tools is not None
            else int(getattr(config, "tool_selection_max_tools", 8))
        )
        include = always_include
        if include is None:
            include = getattr(config, "tool_selection_always_include", ())
        self.always_include = frozenset(
            name.strip() for name in include if isinstance(name, str) and name.strip()
        )

    def select(self, query: str, tools: Sequence[object]) -> list[object]:
        """返回与 query 相关的工具子集；无命中时返回全量工具。"""

        if not tools:
            return list(tools)

        query_text = query or ""
        if not query_text.strip():
            return list(tools)
        query_tokens = tokenize(query_text)
        lowered_query = query_text.lower()

        scored: list[tuple[float, int, object]] = []
        for index, tool in enumerate(tools):
            name = self._tool_name(tool)
            score = self._score_tool(name, self._tool_description(tool), lowered_query, query_tokens)
            if name in self.always_include or score > 0:
                scored.append((score if name not in self.always_include else float("inf"), index, tool))

        if not scored:
            # 完全无命中时回退全量，避免词法失配导致 Agent 无工具可用。
            return list(tools)

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [tool for _, _, tool in scored]
        if self.max_tools > 0:
            selected = selected[: self.max_tools]
        return selected

    def select_split(
        self,
        query: str,
        local_tools: Sequence[object],
        mcp_tools: Sequence[object],
    ) -> tuple[list[object], list[object]]:
        """对 local/mcp 两组工具分别筛选，保留来源信息供 ToolManager 包装区分。"""

        combined = [*local_tools, *mcp_tools]
        selected = self.select(query, combined)
        selected_ids = {id(tool) for tool in selected}
        return (
            [tool for tool in local_tools if id(tool) in selected_ids],
            [tool for tool in mcp_tools if id(tool) in selected_ids],
        )

    def _score_tool(
        self,
        name: str,
        description: str | None,
        lowered_query: str,
        query_tokens: set[str],
    ) -> float:
        if not query_tokens:
            return 0.0

        score = 0.0
        lowered_name = name.lower()
        # 步骤文本里直接写出工具名（planner 被要求这样生成），权重最高。
        if lowered_name in lowered_query:
            score += 10.0

        name_tokens = set(_ASCII_WORD_RE.findall(lowered_name)) | tokenize(
            "".join(_CJK_RUN_RE.findall(lowered_name))
        )
        score += 3.0 * len(name_tokens & query_tokens)

        if description:
            desc_tokens = tokenize(description)
            score += 1.0 * len(desc_tokens & query_tokens)
        return score

    @staticmethod
    def _tool_name(tool: object) -> str:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
        return tool.__class__.__name__

    @staticmethod
    def _tool_description(tool: object) -> str | None:
        description = getattr(tool, "description", None)
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None
