"""RAG 轻量检索指标。

本模块只实现可离线计算的 Hit@K、Recall@K 和 MRR。它接收检索出的 doc_id 或
chunk_id 字符串，不依赖真实 retriever、向量库、LLM 或在线 trace，便于 CI 和后续
runner 复用。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCaseMetric:
    """单条 eval case 的检索指标。

    no-answer/empty-retrieval case 也会保留在 per-case 输出里，但 `comparable=false`，
    不进入 Hit/Recall/MRR 分母。这样 baseline 报告既能追踪拒答类用例，又不会把
    “本来不该召回文档”的 case 误算成检索失败。
    """

    case_index: int
    case_id: str | None
    comparable: bool
    expected_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    hit_at_k: bool
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None


@dataclass(frozen=True)
class RetrievalMetricsSummary:
    """RAG baseline 指标汇总。

    该对象保持纯数据结构，便于 ISSUE-028 测试和后续 runner/report 复用；它不读取
    eval set、不调用 retriever，也不写报告文件，避免提前实现阶段 4 的 evaluation
    runner 和报告输出。
    """

    k: int
    case_count: int
    comparable_case_count: int
    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    cases: tuple[RetrievalCaseMetric, ...]


def hit_rate_at_k(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    k: int,
) -> float:
    """计算 Hit@K。

    no-answer/empty-retrieval case 的 expected ids 为空，不应进入分母；否则拒答类
    用例会把检索指标稀释成不可解释的数字。只要 top-k 里命中任一 expected id，
    当前 case 记为一次 hit。
    """

    comparable_cases = _comparable_cases(expected_ids_by_case, retrieved_ids_by_case)
    if not comparable_cases or k <= 0:
        return 0.0

    hit_count = 0
    for expected_ids, retrieved_ids in comparable_cases:
        if _first_relevant_rank(expected_ids, retrieved_ids, k=k) is not None:
            hit_count += 1
    return hit_count / len(comparable_cases)


def recall_at_k(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    k: int,
) -> float:
    """计算平均 Recall@K。

    每个 case 先计算 expected ids 的覆盖率，再对可回答 case 求平均。这样多答案 case
    不会因为命中一个文档就被误判为完全召回，符合 RAG eval baseline 的诊断目的。
    """

    comparable_cases = _comparable_cases(expected_ids_by_case, retrieved_ids_by_case)
    if not comparable_cases or k <= 0:
        return 0.0

    recall_sum = 0.0
    for expected_ids, retrieved_ids in comparable_cases:
        unique_expected_ids = _unique_preserving_order(expected_ids)
        top_k_retrieved_ids = list(retrieved_ids[:k])
        covered_count = sum(
            1
            for expected_id in unique_expected_ids
            if _contains_relevant_id(expected_id, top_k_retrieved_ids)
        )
        recall_sum += covered_count / len(unique_expected_ids)
    return recall_sum / len(comparable_cases)


def mean_reciprocal_rank(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    k: int | None = None,
) -> float:
    """计算 MRR。

    只看每个可回答 case 的第一个相关结果排名，未命中记为 0。`k=None` 表示使用完整
    retrieved 列表；传入 `k` 时只评估 top-k，便于和 Hit@K/Recall@K 使用同一窗口。
    """

    comparable_cases = _comparable_cases(expected_ids_by_case, retrieved_ids_by_case)
    if not comparable_cases:
        return 0.0

    reciprocal_sum = 0.0
    for expected_ids, retrieved_ids in comparable_cases:
        rank = _first_relevant_rank(expected_ids, retrieved_ids, k=k)
        if rank is not None:
            reciprocal_sum += 1.0 / rank
    return reciprocal_sum / len(comparable_cases)


def mrr(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    k: int | None = None,
) -> float:
    """`mean_reciprocal_rank` 的短别名，方便后续 runner 输出指标字段。"""

    return mean_reciprocal_rank(expected_ids_by_case, retrieved_ids_by_case, k=k)


def summarize_retrieval_metrics(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    k: int,
    case_ids: Sequence[str] | None = None,
) -> RetrievalMetricsSummary:
    """计算 aggregate 和 per-case 检索指标。

    ISSUE-028 需要把 pipeline baseline 固定成可回归的结构，而不是只返回三个浮点数。
    因此这里在保留既有 Hit@K/Recall@K/MRR 语义的基础上，额外输出每个 case 的命中
    明细。该函数仍是纯函数，不依赖真实 Milvus、DashScope、MCP 或网络。
    """

    _validate_case_lengths(expected_ids_by_case, retrieved_ids_by_case, case_ids=case_ids)

    case_metrics = tuple(
        _build_case_metric(
            case_index=index,
            case_id=_case_id_at(case_ids, index),
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            k=k,
        )
        for index, (expected_ids, retrieved_ids) in enumerate(
            zip(expected_ids_by_case, retrieved_ids_by_case, strict=True)
        )
    )
    comparable_cases = [case for case in case_metrics if case.comparable]
    if not comparable_cases or k <= 0:
        return RetrievalMetricsSummary(
            k=k,
            case_count=len(case_metrics),
            comparable_case_count=len(comparable_cases),
            hit_rate_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            cases=case_metrics,
        )

    hit_count = sum(1 for case in comparable_cases if case.hit_at_k)
    recall_sum = sum(case.recall_at_k for case in comparable_cases)
    reciprocal_sum = sum(case.reciprocal_rank for case in comparable_cases)
    comparable_count = len(comparable_cases)
    return RetrievalMetricsSummary(
        k=k,
        case_count=len(case_metrics),
        comparable_case_count=comparable_count,
        hit_rate_at_k=hit_count / comparable_count,
        recall_at_k=recall_sum / comparable_count,
        mrr=reciprocal_sum / comparable_count,
        cases=case_metrics,
    )


def _comparable_cases(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
) -> list[tuple[list[str], list[str]]]:
    _validate_case_lengths(expected_ids_by_case, retrieved_ids_by_case)

    comparable: list[tuple[list[str], list[str]]] = []
    for expected_ids, retrieved_ids in zip(
        expected_ids_by_case, retrieved_ids_by_case, strict=True
    ):
        unique_expected_ids = _unique_preserving_order(expected_ids)
        if not unique_expected_ids:
            continue
        comparable.append((unique_expected_ids, _normalize_id_list(retrieved_ids)))
    return comparable


def _build_case_metric(
    *,
    case_index: int,
    case_id: str | None,
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> RetrievalCaseMetric:
    normalized_expected_ids = tuple(_unique_preserving_order(expected_ids))
    normalized_retrieved_ids = tuple(_normalize_id_list(retrieved_ids))
    comparable = bool(normalized_expected_ids)
    if not comparable or k <= 0:
        return RetrievalCaseMetric(
            case_index=case_index,
            case_id=case_id,
            comparable=comparable,
            expected_ids=normalized_expected_ids,
            retrieved_ids=normalized_retrieved_ids,
            hit_at_k=False,
            recall_at_k=0.0,
            reciprocal_rank=0.0,
            first_relevant_rank=None,
        )

    first_rank = _first_relevant_rank(
        normalized_expected_ids,
        normalized_retrieved_ids,
        k=k,
    )
    top_k_retrieved_ids = list(normalized_retrieved_ids[:k])
    covered_count = sum(
        1
        for expected_id in normalized_expected_ids
        if _contains_relevant_id(expected_id, top_k_retrieved_ids)
    )
    return RetrievalCaseMetric(
        case_index=case_index,
        case_id=case_id,
        comparable=True,
        expected_ids=normalized_expected_ids,
        retrieved_ids=normalized_retrieved_ids,
        hit_at_k=first_rank is not None,
        recall_at_k=covered_count / len(normalized_expected_ids),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        first_relevant_rank=first_rank,
    )


def _validate_case_lengths(
    expected_ids_by_case: Sequence[Sequence[str]],
    retrieved_ids_by_case: Sequence[Sequence[str]],
    *,
    case_ids: Sequence[str] | None = None,
) -> None:
    if len(expected_ids_by_case) != len(retrieved_ids_by_case):
        raise ValueError("expected and retrieved must contain the same number of cases")
    if case_ids is not None and len(case_ids) != len(expected_ids_by_case):
        raise ValueError("case_ids must contain the same number of cases")


def _case_id_at(case_ids: Sequence[str] | None, index: int) -> str | None:
    if case_ids is None:
        return None
    normalized_case_id = case_ids[index].strip()
    return normalized_case_id or None


def _first_relevant_rank(
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    *,
    k: int | None,
) -> int | None:
    if k is not None and k <= 0:
        return None

    limit = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    for zero_based_index, retrieved_id in enumerate(retrieved_ids[:limit]):
        if any(_ids_match(expected_id, retrieved_id) for expected_id in expected_ids):
            return zero_based_index + 1
    return None


def _contains_relevant_id(expected_id: str, retrieved_ids: Sequence[str]) -> bool:
    return any(_ids_match(expected_id, retrieved_id) for retrieved_id in retrieved_ids)


def _ids_match(expected_id: str, retrieved_id: str) -> bool:
    """比较 expected id 与 retrieved id。

    eval set 当前保存 doc_id，而后续 retriever 可能直接返回 chunk_id。若 expected 是
    doc_id，`doc_id#000001` 这类 chunk_id 应视为命中同一文档；若 expected 本身是
    chunk_id，则必须精确命中，避免把同文档的其他 chunk 误算为 chunk 级召回。
    """

    if expected_id == retrieved_id:
        return True
    if "#" in expected_id:
        return False
    return retrieved_id.startswith(f"{expected_id}#")


def _unique_preserving_order(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for raw_id in ids:
        normalized_id = raw_id.strip()
        if not normalized_id or normalized_id in seen:
            continue
        seen.add(normalized_id)
        unique_ids.append(normalized_id)
    return unique_ids


def _normalize_id_list(ids: Sequence[str]) -> list[str]:
    return [raw_id.strip() for raw_id in ids if raw_id.strip()]
