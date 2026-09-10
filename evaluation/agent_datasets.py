"""Agent 轨迹评测数据集加载器。

与 `evaluation.datasets` 的边界一致：只读取并校验本地 YAML（默认
`eval_sets/agent_cases.yaml`），返回结构化 `AgentCase`。它不连接 LLM、MCP
server、Redis 或网络，也不被 FastAPI 默认 import，保证离线轨迹评测不污染线上
请求路径。

`AgentCase` 评的是 Plan-Execute-Replan 轨迹而非单次检索：
- `required_tools` / `forbidden_tools` 驱动工具选择指标；
- `max_steps_budget` 驱动步数效率指标；
- `expected_keywords` 驱动任务成功率；
- `should_complete=false` 的拒答类用例不进入成功率分母（与 RAG eval 的
  comparable 语义对齐）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LOGGER = logging.getLogger(__name__)

AgentCaseType = Literal["diagnose", "tool_query", "out_of_scope"]
AgentDifficulty = Literal["easy", "medium", "hard"]
_AGENT_CASE_FIELD_NAMES = (
    "id",
    "task",
    "should_complete",
    "required_tools",
    "forbidden_tools",
    "expected_keywords",
    "max_steps_budget",
    "case_type",
    "tags",
    "difficulty",
    "golden_answer",
)


class AgentDatasetError(ValueError):
    """Agent 轨迹 eval 数据集错误。

    与 `RagDatasetError` 一致：错误信息保持稳定、可读、可测试，不拼接底层异常
    全文，避免把本机路径或解析细节误传到评估报告。
    """


class AgentCase(BaseModel):
    """单条 Agent 轨迹评估用例。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="稳定 case id")
    task: str = Field(..., description="发给 Agent 的任务描述")
    should_complete: bool = Field(..., description="该任务是否应该被完成")
    required_tools: list[str] = Field(
        default_factory=list,
        description="轨迹中必须调用的工具名",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="轨迹中禁止调用的工具名",
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="最终答案必须覆盖的关键词",
    )
    max_steps_budget: int = Field(..., description="步数预算，用于效率指标")
    case_type: AgentCaseType = Field(..., description="用例类型")
    tags: list[str] = Field(..., description="用例标签")
    difficulty: AgentDifficulty = Field(..., description="难度")
    golden_answer: str = Field(..., description="期望行为说明")

    @field_validator("id", "task", "golden_answer")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("字段不能为空")
        return stripped

    @field_validator("required_tools", "forbidden_tools", "expected_keywords", "tags")
    @classmethod
    def _validate_text_list(cls, value: list[str]) -> list[str]:
        normalized_items: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("列表字段不能包含空字符串")
            normalized_items.append(stripped)
        return normalized_items

    @field_validator("max_steps_budget")
    @classmethod
    def _validate_steps_budget(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_steps_budget 不能小于 1")
        return value

    @model_validator(mode="after")
    def _validate_trajectory_contract(self) -> Self:
        # 工具集合互斥：同一名工具同时出现在 required 与 forbidden 时，
        # required_tool_coverage 与 forbidden_tool_violation 会互相矛盾。
        required_set = set(self.required_tools)
        forbidden_set = set(self.forbidden_tools)
        if required_set & forbidden_set:
            raise ValueError("required_tools 与 forbidden_tools 不能相交")
        if self.should_complete and not self.expected_keywords:
            raise ValueError("should_complete=true 时 expected_keywords 不能为空")
        if not self.should_complete and self.expected_keywords:
            raise ValueError("should_complete=false 时 expected_keywords 必须为空")
        if not self.should_complete and self.required_tools:
            raise ValueError("should_complete=false 时 required_tools 必须为空")
        if not self.tags:
            raise ValueError("tags 不能为空")
        return self


def load_agent_cases(path: str | Path) -> list[AgentCase]:
    """加载并校验 Agent 轨迹 eval YAML。

    与 `load_rag_cases` 一致：拒绝缺字段、重复 case id 和契约冲突的用例，让
    轨迹指标不会因为数据集含糊而得到虚假基线。
    """

    dataset_path = Path(path)
    raw_cases = _read_yaml_cases(dataset_path)
    cases: list[AgentCase] = []
    seen_case_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        case_mapping = _as_case_mapping(raw_case, dataset_path=dataset_path, index=index)
        case_id_for_error = _case_id_for_error(case_mapping, index)
        try:
            case = AgentCase.model_validate(case_mapping)
        except ValidationError as exc:
            fields = _validation_error_fields(exc)
            raise AgentDatasetError(
                f"Agent eval case '{case_id_for_error}' invalid fields: {', '.join(fields)}"
            ) from exc

        if case.id in seen_case_ids:
            raise AgentDatasetError(f"Agent eval duplicate case id: {case.id}")
        seen_case_ids.add(case.id)
        cases.append(case)

    LOGGER.info("Loaded %s agent eval cases from %s", len(cases), dataset_path)
    return cases


def _read_yaml_cases(dataset_path: Path) -> list[object]:
    if not dataset_path.exists():
        raise AgentDatasetError(f"Agent eval dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise AgentDatasetError(f"Agent eval dataset path is not a file: {dataset_path}")

    try:
        raw_data: object = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AgentDatasetError(f"Agent eval dataset cannot be read: {dataset_path}") from exc
    except yaml.YAMLError as exc:
        raise AgentDatasetError(f"Agent eval dataset YAML is invalid: {dataset_path}") from exc

    if not isinstance(raw_data, list):
        raise AgentDatasetError("Agent eval dataset root must be a YAML list")
    return list(raw_data)


def _as_case_mapping(raw_case: object, *, dataset_path: Path, index: int) -> dict[str, object]:
    if not isinstance(raw_case, Mapping):
        raise AgentDatasetError(
            f"Agent eval dataset case at index {index} must be a mapping: {dataset_path}"
        )
    return {str(key): value for key, value in raw_case.items()}


def _case_id_for_error(case_mapping: Mapping[str, object], index: int) -> str:
    raw_case_id = case_mapping.get("id")
    if isinstance(raw_case_id, str) and raw_case_id.strip():
        return raw_case_id.strip()
    return f"index-{index}"


def _validation_error_fields(error: ValidationError) -> list[str]:
    """提取稳定字段名，避免把 pydantic 原始错误全文暴露给调用方。"""

    fields: list[str] = []
    for raw_error in error.errors():
        error_mapping = cast(Mapping[str, object], raw_error)
        raw_location = error_mapping.get("loc")
        field_name = _field_name_from_location(raw_location)
        if field_name == "__root__":
            field_name = _field_name_from_model_error(error_mapping.get("msg"))
        if field_name not in fields:
            fields.append(field_name)
    return fields or ["__root__"]


def _field_name_from_location(location: object) -> str:
    if isinstance(location, tuple) and location:
        return str(location[0])
    if isinstance(location, list) and location:
        return str(location[0])
    if isinstance(location, str) and location:
        return location
    return "__root__"


def _field_name_from_model_error(message: object) -> str:
    """把模型级约束错误映射回稳定字段名。"""

    if not isinstance(message, str):
        return "__root__"
    for field_name in sorted(_AGENT_CASE_FIELD_NAMES, key=len, reverse=True):
        if field_name in message:
            return field_name
    return "__root__"

__all__ = [
    "AgentCase",
    "AgentCaseType",
    "AgentDatasetError",
    "AgentDifficulty",
    "load_agent_cases",
]
