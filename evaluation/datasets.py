"""RAG 离线评估数据集加载器。

本模块是 ISSUE-021 的边界产物：只读取并校验 `eval_sets/rag_cases.yaml` 这类
本地 YAML 文件，返回结构化 `RagCase`。它不连接向量库、不调用 LLM、不写线上
trace，也不被 FastAPI 默认 import，避免离线评估能力污染线上请求路径。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LOGGER = logging.getLogger(__name__)

CaseType = Literal["answer", "low_score", "empty_retrieval", "no_answer"]
Difficulty = Literal["easy", "medium", "hard"]
_RAG_CASE_FIELD_NAMES = (
    "id",
    "question",
    "expected_doc_ids",
    "expected_keywords",
    "should_answer",
    "case_type",
    "tags",
    "difficulty",
    "golden_answer",
)


class RagDatasetError(ValueError):
    """离线 eval 数据集错误。

    这里不用 `AppError`，原因是 evaluation 不属于 HTTP API 边界；但错误信息仍保持
    稳定、可读、可测试，不直接拼接底层异常全文，避免把本地路径、解析细节或未来
    runner 内部信息误传到报告里。
    """


class RagCase(BaseModel):
    """单条 RAG 评估用例。

    `expected_doc_ids` 是轻量检索指标的标准答案集合；no-answer/empty-retrieval
    case 必须保持为空，避免把拒答类问题误计入 Hit/Recall/MRR。`expected_keywords`
    和 `golden_answer` 先作为离线基线字段保留，后续 runner/judge 可复用，但本 issue
    不提前实现答案评分。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="稳定 case id")
    question: str = Field(..., description="用户问题")
    expected_doc_ids: list[str] = Field(..., description="期望召回的稳定 doc_id")
    expected_keywords: list[str] = Field(..., description="后续轻量答案检查关键词")
    should_answer: bool = Field(..., description="该问题是否应该基于知识库回答")
    case_type: CaseType = Field(..., description="用例类型")
    tags: list[str] = Field(..., description="用例标签")
    difficulty: Difficulty = Field(..., description="难度")
    golden_answer: str = Field(..., description="参考答案或拒答说明")

    @field_validator("id", "question", "golden_answer")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("字段不能为空")
        return stripped

    @field_validator("expected_doc_ids", "expected_keywords", "tags")
    @classmethod
    def _validate_text_list(cls, value: list[str]) -> list[str]:
        normalized_items: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("列表字段不能包含空字符串")
            normalized_items.append(stripped)
        return normalized_items

    @model_validator(mode="after")
    def _validate_answer_contract(self) -> Self:
        if self.should_answer and not self.expected_doc_ids:
            raise ValueError("should_answer=true 时 expected_doc_ids 不能为空")
        if not self.should_answer and self.expected_doc_ids:
            raise ValueError("should_answer=false 时 expected_doc_ids 必须为空")
        if self.should_answer and not self.expected_keywords:
            raise ValueError("should_answer=true 时 expected_keywords 不能为空")
        if not self.tags:
            raise ValueError("tags 不能为空")
        return self


def load_rag_cases(path: str | Path) -> list[RagCase]:
    """加载并校验 RAG eval YAML。

    loader 明确拒绝缺字段、重复 case id，以及可回答 case 缺少 expected_doc_ids 的
    情况。这样后续 Hit@K/Recall@K/MRR 不会因为数据集本身含糊而得到虚假的基线。
    """

    dataset_path = Path(path)
    raw_cases = _read_yaml_cases(dataset_path)
    cases: list[RagCase] = []
    seen_case_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        case_mapping = _as_case_mapping(raw_case, dataset_path=dataset_path, index=index)
        case_id_for_error = _case_id_for_error(case_mapping, index)
        try:
            case = RagCase.model_validate(case_mapping)
        except ValidationError as exc:
            fields = _validation_error_fields(exc)
            raise RagDatasetError(
                f"RAG eval case '{case_id_for_error}' invalid fields: {', '.join(fields)}"
            ) from exc

        if case.id in seen_case_ids:
            raise RagDatasetError(f"RAG eval duplicate case id: {case.id}")
        seen_case_ids.add(case.id)
        cases.append(case)

    LOGGER.info("Loaded %s RAG eval cases from %s", len(cases), dataset_path)
    return cases


def _read_yaml_cases(dataset_path: Path) -> list[object]:
    if not dataset_path.exists():
        raise RagDatasetError(f"RAG eval dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise RagDatasetError(f"RAG eval dataset path is not a file: {dataset_path}")

    try:
        raw_data: object = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RagDatasetError(f"RAG eval dataset cannot be read: {dataset_path}") from exc
    except yaml.YAMLError as exc:
        raise RagDatasetError(f"RAG eval dataset YAML is invalid: {dataset_path}") from exc

    if not isinstance(raw_data, list):
        raise RagDatasetError("RAG eval dataset root must be a YAML list")
    return list(raw_data)


def _as_case_mapping(raw_case: object, *, dataset_path: Path, index: int) -> dict[str, object]:
    if not isinstance(raw_case, Mapping):
        raise RagDatasetError(
            f"RAG eval dataset case at index {index} must be a mapping: {dataset_path}"
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
    """把模型级约束错误映射回稳定字段名。

    Pydantic 的 model_validator 没有天然字段位置；这里仅扫描我们自己写入的字段名，
    避免把完整错误消息透传给调用方，同时让测试和后续 runner 能定位数据问题。
    """

    if not isinstance(message, str):
        return "__root__"
    for field_name in sorted(_RAG_CASE_FIELD_NAMES, key=len, reverse=True):
        if field_name in message:
            return field_name
    return "__root__"
