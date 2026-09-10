"""Agent 轨迹轻量指标。

本模块只实现可离线计算的任务成功与轨迹质量指标，不依赖真实 LLM、MCP server、
Redis 或在线 trace，便于 CI 和 `evaluation.agent_runner` 复用。指标语义：

- `task_success`：可完成任务的关键词全覆盖（goal 层，只看最终答案）。
- `required_tool_coverage`：必须调用的工具是否都被调用（工具选择层）。
- `forbidden_tool_violation_count`：禁用工具被调用的次数（越界层）。
- `steps_within_budget`：实际步数是否在预算内（效率层）。
- `tool_failure_rate`：轨迹中失败工具调用的占比（可靠性层）。

与 RAG 检索指标一样，`should_complete=false` 的拒答类用例标记
`comparable=false`，不进入任务成功率分母，避免把"本来就不该执行"的任务误算成
执行失败。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTrajectory:
    """adapter 返回的单条 case 轨迹。

    `steps` 是按执行顺序的步骤描述；`tool_calls` 是按调用顺序的工具名（保留
    重复，供违规计数与失败率使用）；`failed_tool_calls` 是其中执行失败的子集；
    `response` 是最终答案，供关键词覆盖指标与报告使用。
    """

    steps: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    failed_tool_calls: tuple[str, ...] = ()
    response: str = ""


@dataclass(frozen=True)
class AgentTrajectoryExpectation:
    """单条 case 的轨迹期望（runner 从 `AgentCase` 构造）。

    metrics 模块保持零 pydantic 依赖：期望以轻量 dataclass 传入，测试可以直接
    构造而不需要加载 YAML。
    """

    should_complete: bool
    expected_keywords: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    steps_budget: int = 8


@dataclass(frozen=True)
class AgentCaseMetric:
    """单条 eval case 的轨迹指标。

    `comparable=false` 的用例（拒答任务或执行失败）仍保留在 per-case 输出里，
    但不进入 task_success_rate / keyword_coverage 的分母。
    """

    case_index: int
    case_id: str | None
    comparable: bool
    task_success: bool
    keyword_coverage: float
    required_tool_coverage: float
    missing_required_tools: tuple[str, ...]
    forbidden_tool_violation_count: int
    violated_forbidden_tools: tuple[str, ...]
    steps_used: int
    steps_budget: int
    steps_within_budget: bool
    tool_call_count: int
    tool_failure_rate: float


@dataclass(frozen=True)
class AgentMetricsSummary:
    """Agent 轨迹 baseline 指标汇总。

    `average_required_tool_coverage` / `average_tool_failure_rate` 在分母为空时
    为 None：没有携带工具要求或没有产生工具调用的数据集，输出 0.0 会伪装成
    "全部达标/全部成功"，None 才是"无样本"的可解释语义。
    """

    case_count: int
    comparable_case_count: int
    task_success_rate: float
    average_keyword_coverage: float
    average_required_tool_coverage: float | None
    forbidden_tool_violation_count: int
    average_steps_used: float
    steps_within_budget_rate: float
    average_tool_failure_rate: float | None
    failed_case_count: int
    cases: tuple[AgentCaseMetric, ...]


def summarize_agent_metrics(
    expectations: Sequence[AgentTrajectoryExpectation],
    trajectories: Sequence[AgentTrajectory],
    *,
    case_ids: Sequence[str] | None = None,
    failed_case_indices: Sequence[int] = (),
) -> AgentMetricsSummary:
    """计算 aggregate 和 per-case 轨迹指标。

    纯函数：输入期望与轨迹的平行列表。`failed_case_indices` 是 adapter 执行
    失败（已捕获异常）的 case 下标，这些 case 的轨迹为空、不进 comparable
    分母，但仍计入 failed_case_count 与步数/预算统计，保留诊断信号。
    """

    _validate_case_lengths(expectations, trajectories, case_ids=case_ids)
    failed_indices = set(failed_case_indices)

    case_metrics = tuple(
        _build_case_metric(
            case_index=index,
            case_id=_case_id_at(case_ids, index),
            expectation=expectation,
            trajectory=trajectory,
            failed=index in failed_indices,
        )
        for index, (expectation, trajectory) in enumerate(
            zip(expectations, trajectories, strict=True)
        )
    )
    comparable_cases = [case for case in case_metrics if case.comparable]
    comparable_count = len(comparable_cases)

    success_count = sum(1 for case in comparable_cases if case.task_success)
    keyword_sum = sum(case.keyword_coverage for case in comparable_cases)

    # required_tool_coverage 的平均只针对"携带工具要求"的 comparable case；
    # 期望为空时单 case 记 1.0（无要求即满足），但不参与平均，避免稀释。
    required_cases = [
        case
        for case, expectation in zip(case_metrics, expectations, strict=True)
        if case.comparable and expectation.required_tools
    ]
    average_required_coverage = (
        sum(case.required_tool_coverage for case in required_cases) / len(required_cases)
        if required_cases
        else None
    )

    tool_called_cases = [case for case in case_metrics if case.tool_call_count > 0]
    average_tool_failure_rate = (
        sum(case.tool_failure_rate for case in tool_called_cases) / len(tool_called_cases)
        if tool_called_cases
        else None
    )

    within_budget_count = sum(1 for case in case_metrics if case.steps_within_budget)

    return AgentMetricsSummary(
        case_count=len(case_metrics),
        comparable_case_count=comparable_count,
        task_success_rate=(success_count / comparable_count) if comparable_count else 0.0,
        average_keyword_coverage=(keyword_sum / comparable_count) if comparable_count else 0.0,
        average_required_tool_coverage=average_required_coverage,
        forbidden_tool_violation_count=sum(
            case.forbidden_tool_violation_count for case in case_metrics
        ),
        average_steps_used=(
            sum(case.steps_used for case in case_metrics) / len(case_metrics)
            if case_metrics
            else 0.0
        ),
        steps_within_budget_rate=(
            within_budget_count / len(case_metrics) if case_metrics else 0.0
        ),
        average_tool_failure_rate=average_tool_failure_rate,
        failed_case_count=len(failed_indices),
        cases=case_metrics,
    )


def _build_case_metric(
    *,
    case_index: int,
    case_id: str | None,
    expectation: AgentTrajectoryExpectation,
    trajectory: AgentTrajectory,
    failed: bool,
) -> AgentCaseMetric:
    comparable = expectation.should_complete and not failed
    steps_used = len(trajectory.steps)
    tool_calls = tuple(_normalize_tool_names(trajectory.tool_calls))
    failed_tools = set(_normalize_tool_names(trajectory.failed_tool_calls))

    keyword_coverage = _keyword_coverage(
        expectation.expected_keywords,
        trajectory.response,
    ) if comparable else 0.0
    task_success = comparable and keyword_coverage >= 1.0

    required_tools = tuple(_unique_preserving_order(expectation.required_tools))
    called_tools = set(tool_calls)
    missing_required = tuple(
        tool_name for tool_name in required_tools if tool_name not in called_tools
    )
    required_tool_coverage = (
        (len(required_tools) - len(missing_required)) / len(required_tools)
        if comparable and required_tools
        else (1.0 if comparable else 0.0)
    )

    forbidden_tools = tuple(_unique_preserving_order(expectation.forbidden_tools))
    violated = tuple(
        tool_name
        for tool_name in tool_calls
        if tool_name in set(forbidden_tools)
    )

    tool_call_count = len(tool_calls)
    tool_failure_rate = (
        sum(1 for name in tool_calls if name in failed_tools) / tool_call_count
        if tool_call_count
        else 0.0
    )

    return AgentCaseMetric(
        case_index=case_index,
        case_id=case_id,
        comparable=comparable,
        task_success=task_success,
        keyword_coverage=keyword_coverage,
        required_tool_coverage=required_tool_coverage,
        missing_required_tools=missing_required,
        forbidden_tool_violation_count=len(violated),
        violated_forbidden_tools=tuple(_unique_preserving_order(violated)),
        steps_used=steps_used,
        steps_budget=expectation.steps_budget,
        steps_within_budget=steps_used <= expectation.steps_budget,
        tool_call_count=tool_call_count,
        tool_failure_rate=tool_failure_rate,
    )


def _keyword_coverage(expected_keywords: Sequence[str], response: str) -> float:
    unique_keywords = _unique_preserving_order(expected_keywords)
    if not unique_keywords:
        return 0.0
    lowered_response = response.lower()
    covered = sum(
        1 for keyword in unique_keywords if keyword.lower() in lowered_response
    )
    return covered / len(unique_keywords)


def _validate_case_lengths(
    expectations: Sequence[AgentTrajectoryExpectation],
    trajectories: Sequence[AgentTrajectory],
    *,
    case_ids: Sequence[str] | None = None,
) -> None:
    if len(expectations) != len(trajectories):
        raise ValueError("expectations and trajectories must contain the same number of cases")
    if case_ids is not None and len(case_ids) != len(expectations):
        raise ValueError("case_ids must contain the same number of cases")


def _case_id_at(case_ids: Sequence[str] | None, index: int) -> str | None:
    if case_ids is None:
        return None
    normalized_case_id = case_ids[index].strip()
    return normalized_case_id or None


def _normalize_tool_names(tool_names: Sequence[str]) -> list[str]:
    return [name.strip() for name in tool_names if name.strip()]


def _unique_preserving_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for raw_item in items:
        normalized_item = raw_item.strip()
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        unique_items.append(normalized_item)
    return unique_items

__all__ = [
    "AgentCaseMetric",
    "AgentMetricsSummary",
    "AgentTrajectory",
    "AgentTrajectoryExpectation",
    "summarize_agent_metrics",
]
