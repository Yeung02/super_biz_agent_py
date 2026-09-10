"""Agent 轨迹离线 evaluation runner。

runner 默认使用 dry-run adapter，因此命令

`python -m evaluation.agent_runner --dataset eval_sets/agent_cases.yaml --output eval_reports`

不会连接真实 LLM、MCP server、Redis 或网络。真实轨迹需要显式选择
`--adapter aiops`（会创建 AIOpsService 并按 case 消费 SSE 事件流），防止离线
评估误触发线上依赖。

与 `evaluation.runner`（RAG）保持相同的工程约定：
- 单条 case 失败记录稳定错误码并继续后续 case；
- 报告写成 JSON/Markdown + `agent_trends.jsonl` 趋势文件；
- 写入失败抛稳定错误码，由 CLI 映射为非 0 退出。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar, cast
from uuid import uuid4

from evaluation.agent_datasets import AgentCase, AgentDatasetError, load_agent_cases
from evaluation.agent_metrics import (
    AgentCaseMetric,
    AgentMetricsSummary,
    AgentTrajectory,
    AgentTrajectoryExpectation,
    summarize_agent_metrics,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

LOGGER = logging.getLogger(__name__)
_DEFAULT_SAFE_CASE_ERROR = "评估用例执行失败，已跳过该用例的轨迹输出。"
_DEFAULT_AGENT_DATASET_PATH = "eval_sets/agent_cases.yaml"
_DEFAULT_STEPS_BUDGET = 8
_DEFAULT_REPORT_OUTPUT_DIR = "eval_reports"
_AGENT_REPORT_SCHEMA_VERSION = "agent_eval_report.v1"
_AGENT_TREND_FILE_NAME = "agent_trends.jsonl"
_ConfigValue = TypeVar("_ConfigValue")


class AgentCaseExecutionError(RuntimeError):
    """单条 case 执行失败的稳定错误。

    真实 adapter 消费 SSE `type=error` 事件时抛出；runner 捕获后保留 `code`
    写入结构化结果。底层异常只通过 exception chaining 留给日志排查。
    """

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class AgentEvaluationAdapterLike(Protocol):
    """runner 调用 Agent 执行轨迹的最小 adapter 协议。"""

    async def evaluate_case(self, case: AgentCase) -> AgentTrajectory:
        """执行单条 case 并返回轨迹（步骤/工具调用/最终答案）。"""


class AIOpsServiceLike(Protocol):
    """真实 adapter 依赖的 AIOps service 最小协议，方便测试注入 fake。"""

    def execute(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> object:
        """消费 Plan-Execute-Replan 的 SSE 事件流。"""


class DryRunAgentEvaluationAdapter:
    """默认 dry-run adapter。

    不连接 LLM、MCP 或 Redis，只返回空轨迹和安全说明文本，让默认 CLI 在任意
    开发机/CI 上跑通数据集加载、轨迹指标汇总和报告写入。
    """

    async def evaluate_case(self, case: AgentCase) -> AgentTrajectory:
        response = (
            "dry-run 未执行真实 Agent pipeline。" if case.should_complete else case.golden_answer
        )
        return AgentTrajectory(response=response)


class AIOpsServiceAgentEvaluationAdapter:
    """基于 `AIOpsService.execute` 的显式轨迹 adapter。

    只有调用方明确选择 `--adapter aiops` 或测试注入时才会使用。轨迹来源：
    1. SSE 事件流：`step_complete` 提取步骤、`complete` 提取最终答案、
       `error` 转成稳定失败；
    2. 流结束后的 `graph.get_state`：读取 `tool_evidence` 得到工具调用名与
       可用性（LangGraph 公开 API，不触碰 checkpoint 内部结构）。

    每条 case 使用独立 session_id（前缀 + case id + 随机后缀），避免污染真实
    会话的 checkpoint。
    """

    def __init__(
        self,
        service: AIOpsServiceLike | None = None,
        *,
        session_prefix: str = "agent_eval",
        case_deadline_ms: int = 300_000,
    ) -> None:
        if service is None:
            from app.services.aiops_service import AIOpsService

            service = cast(AIOpsServiceLike, AIOpsService())
        self.service = service
        self.session_prefix = session_prefix
        # 单 case deadline：真实轨迹是多步 LLM 调用（plan/execute/replan/critic），
        # 线上 request_timeout_ms（默认 60s）对 hard 用例偏紧；评测默认放宽到 5 分钟。
        self.case_deadline_ms = case_deadline_ms

    async def evaluate_case(self, case: AgentCase) -> AgentTrajectory:
        session_id = f"{self.session_prefix}_{case.id}_{uuid4().hex[:8]}"
        steps: list[str] = []
        response = ""
        # 真实 service 路径依赖 RequestContext：token budget 分配、LLM usage 记账、
        # ToolManager 证据采集都按 request_id 归档，缺失时 tool_evidence 会静默为空。
        # 离线 CLI 没有 FastAPI middleware，这里按 case 设置独立上下文（延迟导入，
        # dry-run 路径不引入 app 依赖），证据按 case 隔离、互不串扰。
        from app.core.request_context import (
            reset_request_context,
            set_request_context,
        )

        token = set_request_context(self._build_request_context(session_id))
        try:
            stream = self.service.execute(case.task, session_id)
            async for event in _iterate_events(stream):
                event_type = event.get("type")
                if event_type == "step_complete":
                    step = event.get("current_step")
                    if isinstance(step, str) and step.strip():
                        steps.append(step.strip())
                elif event_type == "complete":
                    raw_response = event.get("response")
                    if isinstance(raw_response, str):
                        response = raw_response
                elif event_type == "error":
                    raise AgentCaseExecutionError(
                        _sse_error_code(event),
                        "Agent 执行返回错误事件，该 case 未产出有效轨迹。",
                    )
        except AgentCaseExecutionError:
            raise
        except Exception as exc:
            raise AgentCaseExecutionError(
                "AGENT_EVAL_CASE_FAILED",
                _DEFAULT_SAFE_CASE_ERROR,
            ) from exc
        finally:
            reset_request_context(token)

        tool_calls, failed_tool_calls = self._read_state_tools(session_id)
        return AgentTrajectory(
            steps=tuple(steps),
            tool_calls=tool_calls,
            failed_tool_calls=failed_tool_calls,
            response=response,
        )

    def _build_request_context(self, session_id: str) -> object:
        """按 case 构造独立 RequestContext；ID 格式与线上 middleware 一致。"""

        from app.core.request_context import RequestContext

        now = time.time()
        return RequestContext(
            trace_id=f"trc_{uuid4().hex}",
            request_id=f"req_{uuid4().hex}",
            session_id=session_id,
            tenant_id="default",
            user_id="agent-eval",
            deadline_ms=self.case_deadline_ms,
            feature_flags=(),
            started_at=now,
            started_monotonic=time.monotonic(),
            method="CLI",
            path="evaluation/agent_runner",
            invalid_inbound_trace_header=False,
        )

    def _read_state_tools(
        self,
        session_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """从最终 state 的 tool_evidence 读取工具调用轨迹（fail-open）。"""

        try:
            graph = getattr(self.service, "graph", None)
            if graph is None:
                return (), ()
            state = graph.get_state({"configurable": {"thread_id": session_id}})
            values = getattr(state, "values", None) if state is not None else None
            if not isinstance(values, Mapping):
                return (), ()
            evidence = values.get("tool_evidence")
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
                return (), ()
            tool_calls: list[str] = []
            failed_tool_calls: list[str] = []
            for block in evidence:
                if not isinstance(block, Mapping):
                    continue
                tool_name = block.get("tool_name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    continue
                tool_calls.append(tool_name.strip())
                if block.get("usable") is False:
                    failed_tool_calls.append(tool_name.strip())
            return tuple(tool_calls), tuple(failed_tool_calls)
        except Exception:
            # 工具证据是轨迹指标的增强信息，读取失败不阻断主轨迹（SSE 已提供
            # steps/response）；异常类型进日志即可，不进入评估结果。
            LOGGER.warning(
                "Agent eval tool evidence read failed",
                extra={"session_id": session_id},
            )
            return (), ()


@dataclass(frozen=True)
class AgentEvaluationErrorInfo:
    """评估错误的安全输出结构。"""

    code: str
    message: str

    def to_dict(self) -> JsonObject:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class _RawCaseRun:
    case: AgentCase
    trajectory: AgentTrajectory
    latency_ms: float
    error: AgentEvaluationErrorInfo | None


@dataclass(frozen=True)
class AgentCaseEvaluationResult:
    """单条 eval case 的完整结构化结果。"""

    case_id: str
    task: str
    should_complete: bool
    required_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    steps: tuple[str, ...]
    tool_calls: tuple[str, ...]
    failed_tool_calls: tuple[str, ...]
    response: str
    trajectory_metric: AgentCaseMetric
    latency_ms: float
    error: AgentEvaluationErrorInfo | None = None

    def to_dict(self) -> JsonObject:
        """转换为后续报告可复用的结构，不包含原始异常全文。"""

        return {
            "case_id": self.case_id,
            "task": self.task,
            "should_complete": self.should_complete,
            "required_tools": list(self.required_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "expected_keywords": list(self.expected_keywords),
            "steps": list(self.steps),
            "tool_calls": list(self.tool_calls),
            "failed_tool_calls": list(self.failed_tool_calls),
            "response": self.response,
            "trajectory_metric": _case_metric_to_dict(self.trajectory_metric),
            "latency_ms": self.latency_ms,
            "error": self.error.to_dict() if self.error is not None else None,
        }


@dataclass(frozen=True)
class AgentEvaluationAggregateResult:
    """全量轨迹评估汇总指标。"""

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

    def to_dict(self) -> JsonObject:
        return {
            "case_count": self.case_count,
            "comparable_case_count": self.comparable_case_count,
            "task_success_rate": self.task_success_rate,
            "average_keyword_coverage": self.average_keyword_coverage,
            "average_required_tool_coverage": self.average_required_tool_coverage,
            "forbidden_tool_violation_count": self.forbidden_tool_violation_count,
            "average_steps_used": self.average_steps_used,
            "steps_within_budget_rate": self.steps_within_budget_rate,
            "average_tool_failure_rate": self.average_tool_failure_rate,
            "failed_case_count": self.failed_case_count,
        }


@dataclass(frozen=True)
class AgentEvaluationResult:
    """runner 顶层结果。"""

    run_id: str
    dataset_path: str | None
    adapter_name: str
    duration_ms: float
    aggregate: AgentEvaluationAggregateResult
    cases: tuple[AgentCaseEvaluationResult, ...]

    def to_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "adapter": self.adapter_name,
            "duration_ms": self.duration_ms,
            "aggregate": self.aggregate.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class AgentEvaluationReportPaths:
    """一次评估报告写入产生的文件路径。"""

    json_path: Path
    markdown_path: Path
    trend_path: Path

    def to_dict(self) -> JsonObject:
        return {
            "json_path": str(self.json_path),
            "markdown_path": str(self.markdown_path),
            "trend_path": str(self.trend_path),
        }


class AgentEvaluationReportWriteError(RuntimeError):
    """评估报告写入失败的稳定错误。"""

    code = "AGENT_EVAL_REPORT_WRITE_FAILED"
    safe_message = "Agent 评估报告写入失败，请检查输出目录权限和路径。"

    def __init__(self) -> None:
        super().__init__(self.safe_message)


class AgentEvaluationRunner:
    """执行 Agent 轨迹 eval set 的离线 runner。"""

    def __init__(
        self,
        *,
        adapter: AgentEvaluationAdapterLike | None = None,
        adapter_name: str = "dry-run",
        run_id: str | None = None,
    ) -> None:
        self.adapter = adapter or DryRunAgentEvaluationAdapter()
        self.adapter_name = adapter_name
        self.run_id = run_id or _new_run_id()

    async def run(
        self,
        cases: Sequence[AgentCase],
        *,
        dataset_path: str | Path | None = None,
    ) -> AgentEvaluationResult:
        """运行完整评估。

        单条 case 失败会被记录并继续后续 case，轨迹指标始终可输出——与 RAG
        runner 的"judge/case 失败不阻断"约定一致。
        """

        started = time.perf_counter()
        raw_runs = [await self._run_case(case) for case in cases]
        metrics_summary = summarize_agent_metrics(
            [self._expectation_of(raw.case) for raw in raw_runs],
            [raw.trajectory for raw in raw_runs],
            case_ids=[raw.case.id for raw in raw_runs],
            failed_case_indices=[
                index for index, raw in enumerate(raw_runs) if raw.error is not None
            ],
        )
        case_results = tuple(
            self._build_case_result(raw_run, metric)
            for raw_run, metric in zip(raw_runs, metrics_summary.cases, strict=True)
        )
        duration_ms = _elapsed_ms(started)
        aggregate = _build_aggregate(metrics_summary)
        dataset_path_text = str(dataset_path) if dataset_path is not None else None
        LOGGER.info(
            "Agent eval run finished",
            extra={
                "run_id": self.run_id,
                "dataset_path": dataset_path_text,
                "case_count": len(case_results),
                "duration_ms": duration_ms,
            },
        )
        return AgentEvaluationResult(
            run_id=self.run_id,
            dataset_path=dataset_path_text,
            adapter_name=self.adapter_name,
            duration_ms=duration_ms,
            aggregate=aggregate,
            cases=case_results,
        )

    async def _run_case(self, case: AgentCase) -> _RawCaseRun:
        started = time.perf_counter()
        trajectory = AgentTrajectory()
        error: AgentEvaluationErrorInfo | None = None
        try:
            trajectory = await self.adapter.evaluate_case(case)
        except AgentCaseExecutionError as exc:
            error = AgentEvaluationErrorInfo(code=exc.code, message=exc.safe_message)
        except Exception as exc:
            # 只把异常类型写入日志，结构化结果使用固定安全文案，避免密钥、内部
            # URL 或 traceback 进入评估输出；失败 case 的轨迹为空，指标会自然
            # 反映该 case 未达标，同时 runner 继续执行后续用例。
            LOGGER.warning(
                "Agent eval case failed",
                extra={
                    "run_id": self.run_id,
                    "case_id": case.id,
                    "error_class": exc.__class__.__name__,
                },
            )
            error = AgentEvaluationErrorInfo(
                code="AGENT_EVAL_CASE_FAILED",
                message=_DEFAULT_SAFE_CASE_ERROR,
            )
        return _RawCaseRun(
            case=case,
            trajectory=trajectory,
            latency_ms=_elapsed_ms(started),
            error=error,
        )

    @staticmethod
    def _expectation_of(case: AgentCase) -> AgentTrajectoryExpectation:
        return AgentTrajectoryExpectation(
            should_complete=case.should_complete,
            expected_keywords=tuple(case.expected_keywords),
            required_tools=tuple(case.required_tools),
            forbidden_tools=tuple(case.forbidden_tools),
            steps_budget=case.max_steps_budget,
        )

    @staticmethod
    def _build_case_result(
        raw_run: _RawCaseRun,
        metric: AgentCaseMetric,
    ) -> AgentCaseEvaluationResult:
        return AgentCaseEvaluationResult(
            case_id=raw_run.case.id,
            task=raw_run.case.task,
            should_complete=raw_run.case.should_complete,
            required_tools=tuple(raw_run.case.required_tools),
            forbidden_tools=tuple(raw_run.case.forbidden_tools),
            expected_keywords=tuple(raw_run.case.expected_keywords),
            steps=raw_run.trajectory.steps,
            tool_calls=raw_run.trajectory.tool_calls,
            failed_tool_calls=raw_run.trajectory.failed_tool_calls,
            response=raw_run.trajectory.response,
            trajectory_metric=metric,
            latency_ms=raw_run.latency_ms,
            error=raw_run.error,
        )


async def load_and_run_agent(
    dataset_path: str | Path,
    *,
    adapter: AgentEvaluationAdapterLike | None = None,
    adapter_name: str = "dry-run",
) -> AgentEvaluationResult:
    """加载数据集并运行 runner，供 CLI 和测试复用。"""

    cases = load_agent_cases(dataset_path)
    runner = AgentEvaluationRunner(adapter=adapter, adapter_name=adapter_name)
    return await runner.run(cases, dataset_path=dataset_path)


def write_agent_evaluation_report(
    result: AgentEvaluationResult,
    *,
    output_dir: str | Path,
    config: Mapping[str, object] | None = None,
    thresholds: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> AgentEvaluationReportPaths:
    """把一次轨迹评估结果写成 JSON、Markdown 和趋势 JSONL。

    趋势写入独立的 `agent_trends.jsonl`（与 RAG 的 `trends.jsonl` schema 不同，
    不混写，避免趋势分析工具误读字段）。写入失败抛稳定错误码，由 CLI 映射为
    非 0 退出。
    """

    report_time = generated_at or datetime.now().astimezone()
    report_payload = _build_report_payload(
        result,
        config=config or {},
        thresholds=thresholds or {},
        generated_at=report_time,
    )
    output_path = Path(output_dir)
    stem = _report_file_stem(report_time)
    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"
    trend_path = output_path / _AGENT_TREND_FILE_NAME

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            _render_markdown_report(report_payload),
            encoding="utf-8",
        )
        _append_trend(trend_path, report_payload)
    except OSError as exc:
        LOGGER.error(
            "Agent eval report write failed",
            extra={
                "run_id": result.run_id,
                "output_dir": str(output_path),
                "error_class": exc.__class__.__name__,
            },
        )
        raise AgentEvaluationReportWriteError() from exc

    LOGGER.info(
        "Agent eval report written",
        extra={
            "run_id": result.run_id,
            "json_report_path": str(json_path),
            "markdown_report_path": str(markdown_path),
            "trend_path": str(trend_path),
        },
    )
    return AgentEvaluationReportPaths(
        json_path=json_path,
        markdown_path=markdown_path,
        trend_path=trend_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    默认参数遵循离线验收命令：dry-run adapter，不触碰任何线上依赖。stdout
    输出结构化结果并追加 `reports` 字段。
    """

    parser = argparse.ArgumentParser(
        description="Run AegisOps Agent offline trajectory evaluation"
    )
    parser.add_argument(
        "--dataset",
        default=_config_value("agent_eval_dataset_path", _DEFAULT_AGENT_DATASET_PATH),
        help="Agent eval YAML path",
    )
    parser.add_argument(
        "--adapter",
        choices=("dry-run", "aiops"),
        default=_config_value("agent_eval_adapter", "dry-run"),
        help="Evaluation adapter. dry-run never touches external services.",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_REPORT_OUTPUT_DIR,
        help="Directory for JSON/Markdown reports and agent_trends.jsonl",
    )
    args = parser.parse_args(argv)

    adapter = _build_adapter(str(args.adapter))
    try:
        result = asyncio.run(
            load_and_run_agent(
                args.dataset,
                adapter=adapter,
                adapter_name=str(args.adapter),
            )
        )
    except AgentDatasetError:
        payload = {
            "success": False,
            "error": {
                "code": "AGENT_EVAL_DATASET_INVALID",
                "message": "评估数据集无效，请检查 YAML 文件路径、字段和 case id。",
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    try:
        report_paths = write_agent_evaluation_report(
            result,
            output_dir=args.output,
            config={
                "dataset": args.dataset,
                "adapter": args.adapter,
            },
        )
    except AgentEvaluationReportWriteError as exc:
        payload = {
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.safe_message,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    payload = result.to_dict()
    payload["reports"] = report_paths.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


async def _iterate_events(stream: object):
    """把 service.execute 的 async generator 转为事件迭代，兼容注入的 fake。"""

    async for event in stream:  # type: ignore[union-attr]
        if isinstance(event, Mapping):
            yield event


def _build_adapter(adapter_name: str) -> AgentEvaluationAdapterLike:
    if adapter_name == "aiops":
        return AIOpsServiceAgentEvaluationAdapter()
    return DryRunAgentEvaluationAdapter()


def _build_aggregate(
    metrics_summary: AgentMetricsSummary,
) -> AgentEvaluationAggregateResult:
    return AgentEvaluationAggregateResult(
        case_count=metrics_summary.case_count,
        comparable_case_count=metrics_summary.comparable_case_count,
        task_success_rate=metrics_summary.task_success_rate,
        average_keyword_coverage=metrics_summary.average_keyword_coverage,
        average_required_tool_coverage=metrics_summary.average_required_tool_coverage,
        forbidden_tool_violation_count=metrics_summary.forbidden_tool_violation_count,
        average_steps_used=metrics_summary.average_steps_used,
        steps_within_budget_rate=metrics_summary.steps_within_budget_rate,
        average_tool_failure_rate=metrics_summary.average_tool_failure_rate,
        failed_case_count=metrics_summary.failed_case_count,
    )


def _build_report_payload(
    result: AgentEvaluationResult,
    *,
    config: Mapping[str, object],
    thresholds: Mapping[str, object],
    generated_at: datetime,
) -> JsonObject:
    aggregate = result.aggregate.to_dict()
    case_diffs = [_case_diff(case) for case in result.cases]
    failed_cases = _failed_case_entries(result.cases)
    return {
        "schema_version": _AGENT_REPORT_SCHEMA_VERSION,
        "report_type": "agent_trajectory_evaluation",
        "run_id": result.run_id,
        "generated_at": generated_at.isoformat(),
        "config": _json_safe_mapping(config),
        "thresholds": _json_safe_mapping(thresholds),
        "dataset": {
            "path": result.dataset_path,
            "case_count": result.aggregate.case_count,
            "comparable_case_count": result.aggregate.comparable_case_count,
        },
        "adapter": result.adapter_name,
        "aggregate": aggregate,
        "failed_cases": failed_cases,
        "case_diffs": case_diffs,
        "cases": [case.to_dict() for case in result.cases],
    }


def _render_markdown_report(report_payload: JsonObject) -> str:
    """渲染面向开发回归的 Markdown 报告（巡检视图，完整数据留在 JSON）。"""

    aggregate = _mapping_value(report_payload, "aggregate")
    dataset = _mapping_value(report_payload, "dataset")
    config = _mapping_value(report_payload, "config")
    failed_cases = _list_of_mappings(report_payload.get("failed_cases"))
    case_diffs = _list_of_mappings(report_payload.get("case_diffs"))

    lines = _markdown_report_header(report_payload)
    lines.extend(_markdown_config_section(config))
    lines.extend(_markdown_dataset_section(dataset))
    lines.extend(_markdown_metrics_section(aggregate))
    lines.extend(_markdown_failed_cases_section(failed_cases))
    lines.extend(_markdown_case_diff_section(case_diffs))
    lines.append("")
    return "\n".join(lines)


def _markdown_report_header(report_payload: Mapping[str, JsonValue]) -> list[str]:
    return [
        "# AegisOps Agent Trajectory Evaluation Report",
        "",
        f"- Run ID: `{_markdown_inline(report_payload.get('run_id'))}`",
        f"- Generated At: `{_markdown_inline(report_payload.get('generated_at'))}`",
        f"- Schema: `{_markdown_inline(report_payload.get('schema_version'))}`",
        f"- Adapter: `{_markdown_inline(report_payload.get('adapter'))}`",
        "",
    ]


def _markdown_config_section(config: Mapping[str, JsonValue]) -> list[str]:
    lines = [
        "## 配置",
        "",
    ]
    if config:
        for key in sorted(config):
            lines.append(f"- `{key}`: `{_markdown_inline(config[key])}`")
    else:
        lines.append("- 无额外配置。")
    lines.append("")
    return lines


def _markdown_dataset_section(dataset: Mapping[str, JsonValue]) -> list[str]:
    return [
        "## 数据集",
        "",
        f"- Path: `{_markdown_inline(dataset.get('path'))}`",
        f"- Case Count: `{_markdown_inline(dataset.get('case_count'))}`",
        (
            "- Comparable Case Count: "
            f"`{_markdown_inline(dataset.get('comparable_case_count'))}`"
        ),
        "",
    ]


def _markdown_metrics_section(aggregate: Mapping[str, JsonValue]) -> list[str]:
    lines = [
        "## 指标汇总",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in (
        "task_success_rate",
        "average_keyword_coverage",
        "average_required_tool_coverage",
        "forbidden_tool_violation_count",
        "average_steps_used",
        "steps_within_budget_rate",
        "average_tool_failure_rate",
        "failed_case_count",
    ):
        lines.append(f"| `{key}` | `{_markdown_inline(aggregate.get(key))}` |")
    lines.append("")
    return lines


def _markdown_failed_cases_section(
    failed_cases: Sequence[Mapping[str, JsonValue]],
) -> list[str]:
    lines = ["## 失败 Case", ""]
    if failed_cases:
        lines.extend(
            [
                "| Case ID | Code | Message |",
                "| --- | --- | --- |",
            ]
        )
        for item in failed_cases:
            lines.append(
                "| "
                f"{_markdown_cell(item.get('case_id'))} | "
                f"{_markdown_cell(item.get('code'))} | "
                f"{_markdown_cell(item.get('message'))} |"
            )
    else:
        lines.append("- 无失败 case。")
    lines.append("")
    return lines


def _markdown_case_diff_section(
    case_diffs: Sequence[Mapping[str, JsonValue]],
) -> list[str]:
    lines = [
        "## Per-case Diff",
        "",
        (
            "| Case ID | Success | Steps | Budget | Missing Tools | "
            "Forbidden Violations | Keyword Coverage | Error |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in case_diffs:
        metric = _mapping_value(item, "trajectory_metric")
        lines.append(
            "| "
            f"{_markdown_cell(item.get('case_id'))} | "
            f"{_markdown_cell(metric.get('task_success'))} | "
            f"{_markdown_cell(metric.get('steps_used'))} | "
            f"{_markdown_cell(metric.get('steps_budget'))} | "
            f"{_markdown_cell(_join_json_list(item.get('missing_required_tools')))} | "
            f"{_markdown_cell(metric.get('forbidden_tool_violation_count'))} | "
            f"{_markdown_cell(metric.get('keyword_coverage'))} | "
            f"{_markdown_cell(item.get('failure_code'))} |"
        )
    return lines


def _append_trend(trend_path: Path, report_payload: JsonObject) -> None:
    aggregate = _mapping_value(report_payload, "aggregate")
    dataset = _mapping_value(report_payload, "dataset")
    trend_payload: JsonObject = {
        "timestamp": report_payload.get("generated_at"),
        "run_id": report_payload.get("run_id"),
        "report_type": report_payload.get("report_type"),
        "dataset_path": dataset.get("path"),
        "case_count": aggregate.get("case_count"),
        "comparable_case_count": aggregate.get("comparable_case_count"),
        "task_success_rate": aggregate.get("task_success_rate"),
        "average_keyword_coverage": aggregate.get("average_keyword_coverage"),
        "average_required_tool_coverage": aggregate.get("average_required_tool_coverage"),
        "forbidden_tool_violation_count": aggregate.get("forbidden_tool_violation_count"),
        "average_steps_used": aggregate.get("average_steps_used"),
        "steps_within_budget_rate": aggregate.get("steps_within_budget_rate"),
        "average_tool_failure_rate": aggregate.get("average_tool_failure_rate"),
        "failed_case_count": aggregate.get("failed_case_count"),
    }
    with trend_path.open("a", encoding="utf-8") as trend_file:
        trend_file.write(json.dumps(trend_payload, ensure_ascii=False) + "\n")


def _case_diff(case: AgentCaseEvaluationResult) -> JsonObject:
    metric = case.trajectory_metric
    return {
        "case_id": case.case_id,
        "task": case.task,
        "should_complete": case.should_complete,
        "required_tools": list(case.required_tools),
        "forbidden_tools": list(case.forbidden_tools),
        "steps": list(case.steps),
        "tool_calls": list(case.tool_calls),
        "missing_required_tools": list(metric.missing_required_tools),
        "violated_forbidden_tools": list(metric.violated_forbidden_tools),
        "trajectory_metric": _case_metric_to_dict(metric),
        "failure_code": case.error.code if case.error is not None else None,
        "error": case.error.to_dict() if case.error is not None else None,
    }


def _failed_case_entries(
    cases: Sequence[AgentCaseEvaluationResult],
) -> list[JsonObject]:
    return [
        {
            "case_id": case.case_id,
            "code": case.error.code,
            "message": case.error.message,
        }
        for case in cases
        if case.error is not None
    ]


def _report_file_stem(generated_at: datetime) -> str:
    return generated_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _sse_error_code(event: Mapping[str, object]) -> str:
    raw_code = event.get("code")
    if isinstance(raw_code, str) and raw_code.strip():
        return raw_code.strip()
    return "AGENT_EVAL_CASE_FAILED"


def _mapping_value(mapping: Mapping[str, JsonValue], key: str) -> JsonObject:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _list_of_mappings(value: JsonValue | object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _join_json_list(value: JsonValue | object) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item) for item in value)


def _markdown_inline(value: JsonValue | object) -> str:
    return str(value).replace("\n", " ").replace("`", "'")


def _markdown_cell(value: JsonValue | object) -> str:
    return _markdown_inline(value).replace("|", "\\|")


def _case_metric_to_dict(metric: AgentCaseMetric) -> JsonObject:
    return {
        "case_index": metric.case_index,
        "case_id": metric.case_id,
        "comparable": metric.comparable,
        "task_success": metric.task_success,
        "keyword_coverage": metric.keyword_coverage,
        "required_tool_coverage": metric.required_tool_coverage,
        "missing_required_tools": list(metric.missing_required_tools),
        "forbidden_tool_violation_count": metric.forbidden_tool_violation_count,
        "violated_forbidden_tools": list(metric.violated_forbidden_tools),
        "steps_used": metric.steps_used,
        "steps_budget": metric.steps_budget,
        "steps_within_budget": metric.steps_within_budget,
        "tool_call_count": metric.tool_call_count,
        "tool_failure_rate": metric.tool_failure_rate,
    }


def _json_safe_mapping(mapping: Mapping[str, object]) -> JsonObject:
    return {str(key): _json_safe_value(value) for key, value in mapping.items()}


def _json_safe_value(value: object) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _new_run_id() -> str:
    return f"agent_eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _config_value(name: str, default: _ConfigValue) -> _ConfigValue:
    """读取 app.config 中的 evaluation 配置，缺少线上依赖时使用默认值。

    `python -m evaluation.agent_runner` 是离线验收命令，不能因为开发机没有
    loguru/langchain 这类线上依赖而无法启动；只有显式选择 aiops adapter 时，
    相关 app 依赖才会被后续路径检查。
    """

    try:
        from app.config import config as app_config
    except ModuleNotFoundError:
        return default
    return cast(_ConfigValue, getattr(app_config, name, default))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentCaseEvaluationResult",
    "AgentCaseExecutionError",
    "AgentEvaluationAdapterLike",
    "AgentEvaluationAggregateResult",
    "AgentEvaluationErrorInfo",
    "AgentEvaluationReportPaths",
    "AgentEvaluationReportWriteError",
    "AgentEvaluationResult",
    "AgentEvaluationRunner",
    "AIOpsServiceAgentEvaluationAdapter",
    "AIOpsServiceLike",
    "DryRunAgentEvaluationAdapter",
    "load_and_run_agent",
    "main",
    "write_agent_evaluation_report",
]
