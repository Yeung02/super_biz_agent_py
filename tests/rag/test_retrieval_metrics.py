"""ISSUE-021 离线 eval loader 和轻量检索指标测试。

这些测试只覆盖本地 YAML 加载和纯函数指标计算，不连接真实 Milvus、DashScope、
MCP server 或网络。这样 evaluation 目录可以作为离线评估地基独立回滚，不进入
线上 FastAPI 请求路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.datasets import RagCase, RagDatasetError, load_rag_cases
from evaluation.rag_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    mrr,
    recall_at_k,
    summarize_retrieval_metrics,
)


def test_load_rag_cases_validates_project_eval_set() -> None:
    cases = load_rag_cases(Path("eval_sets/rag_cases.yaml"))

    assert cases
    assert all(isinstance(case, RagCase) for case in cases)
    assert {case.case_type for case in cases} >= {
        "answer",
        "low_score",
        "empty_retrieval",
        "no_answer",
    }
    assert all(case.expected_doc_ids for case in cases if case.should_answer)
    assert all(not case.expected_doc_ids for case in cases if not case.should_answer)


def test_load_rag_cases_reports_missing_required_field(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-question.yaml"
    dataset_path.write_text(
        """
- id: missing_question
  expected_doc_ids: ["doc_cpu"]
  expected_keywords: ["CPU"]
  should_answer: true
  case_type: answer
  tags: ["cpu"]
  difficulty: easy
  golden_answer: "CPU answer"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RagDatasetError) as exc_info:
        load_rag_cases(dataset_path)

    assert "question" in str(exc_info.value)
    assert "missing_question" in str(exc_info.value)


def test_load_rag_cases_rejects_duplicate_case_id(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate.yaml"
    dataset_path.write_text(
        """
- id: duplicate_case
  question: "CPU 怎么排查？"
  expected_doc_ids: ["doc_cpu"]
  expected_keywords: ["CPU"]
  should_answer: true
  case_type: answer
  tags: ["cpu"]
  difficulty: easy
  golden_answer: "CPU answer"
- id: duplicate_case
  question: "磁盘怎么排查？"
  expected_doc_ids: ["doc_disk"]
  expected_keywords: ["磁盘"]
  should_answer: true
  case_type: answer
  tags: ["disk"]
  difficulty: easy
  golden_answer: "Disk answer"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RagDatasetError) as exc_info:
        load_rag_cases(dataset_path)

    assert "duplicate case id" in str(exc_info.value)
    assert "duplicate_case" in str(exc_info.value)


def test_load_rag_cases_requires_expected_doc_ids_when_should_answer(tmp_path: Path) -> None:
    dataset_path = tmp_path / "answer-without-docs.yaml"
    dataset_path.write_text(
        """
- id: answer_without_docs
  question: "CPU 怎么排查？"
  expected_doc_ids: []
  expected_keywords: ["CPU"]
  should_answer: true
  case_type: answer
  tags: ["cpu"]
  difficulty: easy
  golden_answer: "CPU answer"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RagDatasetError) as exc_info:
        load_rag_cases(dataset_path)

    assert "expected_doc_ids" in str(exc_info.value)
    assert "answer_without_docs" in str(exc_info.value)


def test_hit_rate_at_k_counts_cases_with_at_least_one_expected_hit() -> None:
    expected = [["doc_cpu"], ["doc_disk"], ["doc_memory"]]
    retrieved = [["doc_cpu", "doc_other"], ["doc_other", "doc_disk"], ["doc_other"]]

    assert hit_rate_at_k(expected, retrieved, k=1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(expected, retrieved, k=2) == pytest.approx(2 / 3)


def test_metrics_treat_chunk_ids_as_hits_for_their_parent_doc_ids() -> None:
    expected = [["doc_cpu"], ["doc_disk#000002"]]
    retrieved = [["doc_cpu#000003"], ["doc_disk#000001", "doc_disk#000002"]]

    assert hit_rate_at_k(expected, retrieved, k=1) == pytest.approx(1 / 2)
    assert hit_rate_at_k(expected, retrieved, k=2) == 1.0


def test_recall_at_k_averages_per_case_expected_doc_coverage() -> None:
    expected = [["doc_cpu", "doc_disk"], ["doc_memory"], []]
    retrieved = [["doc_cpu", "doc_other"], ["doc_memory", "doc_other"], ["doc_any"]]

    assert recall_at_k(expected, retrieved, k=2) == pytest.approx(0.75)
    assert recall_at_k(expected, retrieved, k=0) == 0.0


def test_mrr_uses_first_relevant_rank_and_ignores_empty_expected_cases() -> None:
    expected = [["doc_cpu"], ["doc_disk"], [], ["doc_memory"]]
    retrieved = [
        ["doc_other", "doc_cpu"],
        ["doc_unrelated", "doc_disk"],
        ["doc_any"],
        ["doc_memory", "doc_other"],
    ]

    assert mean_reciprocal_rank(expected, retrieved, k=2) == pytest.approx((1 / 2 + 1 / 2 + 1) / 3)
    assert mrr(expected, retrieved, k=2) == pytest.approx((1 / 2 + 1 / 2 + 1) / 3)


def test_metrics_return_zero_for_empty_retrieval_or_no_answer_only_cases() -> None:
    expected = [[], []]
    retrieved: list[list[str]] = [[], ["doc_cpu"]]

    assert hit_rate_at_k(expected, retrieved, k=5) == 0.0
    assert recall_at_k(expected, retrieved, k=5) == 0.0
    assert mean_reciprocal_rank(expected, retrieved, k=5) == 0.0


def test_metrics_reject_mismatched_case_lengths() -> None:
    with pytest.raises(ValueError) as exc_info:
        hit_rate_at_k([["doc_cpu"]], [], k=1)

    assert "same number of cases" in str(exc_info.value)


def test_summarize_retrieval_metrics_returns_aggregate_and_per_case_baseline() -> None:
    expected = [["doc_cpu"], ["doc_disk", "doc_disk_extra"], []]
    retrieved = [
        ["doc_cpu#000001", "doc_other"],
        ["doc_other", "doc_disk#000002"],
        ["doc_unrelated"],
    ]

    summary = summarize_retrieval_metrics(
        expected,
        retrieved,
        k=2,
        case_ids=["cpu_case", "disk_case", "no_answer_case"],
    )

    assert summary.case_count == 3
    assert summary.comparable_case_count == 2
    assert summary.hit_rate_at_k == 1.0
    assert summary.recall_at_k == pytest.approx(0.75)
    assert summary.mrr == pytest.approx((1 + 1 / 2) / 2)
    assert [case.case_id for case in summary.cases] == [
        "cpu_case",
        "disk_case",
        "no_answer_case",
    ]
    assert summary.cases[0].hit_at_k is True
    assert summary.cases[0].first_relevant_rank == 1
    assert summary.cases[1].recall_at_k == pytest.approx(0.5)
    assert summary.cases[1].reciprocal_rank == pytest.approx(1 / 2)
    assert summary.cases[2].comparable is False
    assert summary.cases[2].hit_at_k is False
