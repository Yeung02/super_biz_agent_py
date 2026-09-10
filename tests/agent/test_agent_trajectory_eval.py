"""Agent 轨迹评测体系测试（数据集 / 指标 / runner / 真实 adapter）。

全部使用内存 fake 与临时目录，不访问真实 LLM、MCP server、Redis 或网络，
与 `tests/rag/test_evaluation_runner.py` 的离线验收口径一致。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evaluation.agent_datasets import AgentCase, AgentDatasetError, load_agent_cases
from evaluation.agent_metrics import (
    AgentTrajectory,
    AgentTrajectoryExpectation,
    summarize_agent_metrics,
)
from evaluation.agent_runner import (
    AgentCaseExecutionError,
    AgentEvaluationRunner,
    AIOpsServiceAgentEvaluationAdapter,
    DryRunAgentEvaluationAdapter,
    load_and_run_agent,
    main,
    write_agent_evaluation_report,
)


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    dataset_path = tmp_path / "agent_cases.yaml"
    dataset_path.write_text(
        yaml.safe_dump(cases, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_path


def _raw_case(**overrides: object) -> dict:
    case = {
        "id": "case_001",
        "task": "查询当前 CPU 使用率",
        "should_complete": True,
        "required_tools": ["query_cpu_metrics"],
        "forbidden_tools": [],
        "expected_keywords": ["CPU"],
        "max_steps_budget": 5,
        "case_type": "diagnose",
        "tags": ["cpu"],
        "difficulty": "easy",
        "golden_answer": "应调用 query_cpu_metrics。",
    }
    case.update(overrides)
    return case


# ---------------------------------------------------------------------------
# 数据集加载器
# ---------------------------------------------------------------------------


def test_load_agent_cases_valid(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            _raw_case(),
            _raw_case(
                id="out_of_scope_001",
                task="写一首诗",
                should_complete=False,
                required_tools=[],
                forbidden_tools=["search_log"],
                expected_keywords=[],
                case_type="out_of_scope",
            ),
        ],
    )

    cases = load_agent_cases(dataset_path)

    assert len(cases) == 2
    assert cases[0].id == "case_001"
    assert cases[0].required_tools == ["query_cpu_metrics"]
    assert cases[1].should_complete is False


def test_load_agent_cases_duplicate_id(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path, [_raw_case(), _raw_case()])

    with pytest.raises(AgentDatasetError, match="duplicate case id"):
        load_agent_cases(dataset_path)


def test_load_agent_cases_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AgentDatasetError, match="not found"):
        load_agent_cases(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"required_tools": ["search_log"], "forbidden_tools": ["search_log"]},
            "invalid fields",
        ),
        ({"should_complete": True, "expected_keywords": []}, "expected_keywords"),
        ({"should_complete": False, "expected_keywords": ["结论"]}, "expected_keywords"),
        (
            {
                "should_complete": False,
                "expected_keywords": [],
                "required_tools": ["query_cpu_metrics"],
            },
            "invalid fields",
        ),
        ({"max_steps_budget": 0}, "max_steps_budget"),
        ({"task": "   "}, "task"),
    ],
)
def test_load_agent_cases_contract_violations(
    tmp_path: Path,
    overrides: dict,
    match: str,
) -> None:
    dataset_path = _write_dataset(tmp_path, [_raw_case(**overrides)])

    with pytest.raises(AgentDatasetError, match=match):
        load_agent_cases(dataset_path)


def test_load_agent_cases_root_must_be_list(tmp_path: Path) -> None:
    dataset_path = tmp_path / "agent_cases.yaml"
    dataset_path.write_text("id: not-a-list", encoding="utf-8")

    with pytest.raises(AgentDatasetError, match="YAML list"):
        load_agent_cases(dataset_path)


# ---------------------------------------------------------------------------
# 轨迹指标
# ---------------------------------------------------------------------------


def test_metrics_successful_case() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("CPU", "使用率"),
        required_tools=("query_cpu_metrics", "search_log"),
        forbidden_tools=("get_current_time",),
        steps_budget=5,
    )
    trajectory = AgentTrajectory(
        steps=("查 CPU 指标", "检索日志"),
        tool_calls=("query_cpu_metrics", "search_log"),
        response="CPU 使用率 85%，建议排查。",
    )

    summary = summarize_agent_metrics([expectation], [trajectory], case_ids=["case_001"])

    assert summary.case_count == 1
    assert summary.comparable_case_count == 1
    assert summary.task_success_rate == 1.0
    assert summary.average_keyword_coverage == 1.0
    assert summary.average_required_tool_coverage == 1.0
    assert summary.forbidden_tool_violation_count == 0
    assert summary.steps_within_budget_rate == 1.0
    assert summary.failed_case_count == 0

    metric = summary.cases[0]
    assert metric.task_success is True
    assert metric.missing_required_tools == ()
    assert metric.steps_used == 2


def test_metrics_partial_keyword_and_tool_miss() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("CPU", "内存"),
        required_tools=("query_cpu_metrics", "query_memory_metrics"),
        steps_budget=5,
    )
    trajectory = AgentTrajectory(
        steps=("查 CPU",),
        tool_calls=("query_cpu_metrics", "search_log"),
        response="CPU 使用率偏高。",
    )

    summary = summarize_agent_metrics([expectation], [trajectory])
    metric = summary.cases[0]

    assert metric.task_success is False
    assert metric.keyword_coverage == 0.5
    assert metric.required_tool_coverage == 0.5
    assert metric.missing_required_tools == ("query_memory_metrics",)
    assert summary.task_success_rate == 0.0
    assert summary.average_required_tool_coverage == 0.5


def test_metrics_forbidden_tool_violation_and_step_overrun() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("时间",),
        required_tools=("get_current_time",),
        forbidden_tools=("search_log",),
        steps_budget=2,
    )
    trajectory = AgentTrajectory(
        steps=("步骤1", "步骤2", "步骤3"),
        tool_calls=("get_current_time", "search_log", "search_log"),
        response="当前时间是 12:00。",
    )

    summary = summarize_agent_metrics([expectation], [trajectory])
    metric = summary.cases[0]

    assert metric.task_success is True
    assert metric.forbidden_tool_violation_count == 2
    assert metric.violated_forbidden_tools == ("search_log",)
    assert metric.steps_within_budget is False
    assert metric.steps_used == 3
    assert summary.steps_within_budget_rate == 0.0
    assert summary.forbidden_tool_violation_count == 2


def test_metrics_tool_failure_rate() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("日志",),
        required_tools=("search_log",),
        steps_budget=5,
    )
    trajectory = AgentTrajectory(
        steps=("检索日志",),
        tool_calls=("search_log", "query_cpu_metrics"),
        failed_tool_calls=("search_log",),
        response="日志检索失败。",
    )

    summary = summarize_agent_metrics([expectation], [trajectory])

    assert summary.cases[0].tool_failure_rate == 0.5
    assert summary.average_tool_failure_rate == 0.5


def test_metrics_refusal_case_not_comparable() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=False,
        forbidden_tools=("search_log",),
        steps_budget=3,
    )
    trajectory = AgentTrajectory(
        steps=("判断任务范围",),
        response="该任务超出诊断范围。",
    )

    summary = summarize_agent_metrics([expectation], [trajectory], case_ids=["refusal_001"])

    assert summary.comparable_case_count == 0
    assert summary.task_success_rate == 0.0
    assert summary.average_keyword_coverage == 0.0
    assert summary.average_required_tool_coverage is None
    assert summary.average_tool_failure_rate is None
    assert summary.cases[0].comparable is False
    # 拒答任务仍要统计禁用工具越界与步数预算。
    assert summary.cases[0].steps_within_budget is True


def test_metrics_failed_case_excluded_from_comparable() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("CPU",),
        required_tools=("query_cpu_metrics",),
        steps_budget=5,
    )
    trajectory = AgentTrajectory()

    summary = summarize_agent_metrics(
        [expectation],
        [trajectory],
        failed_case_indices=[0],
    )

    assert summary.comparable_case_count == 0
    assert summary.failed_case_count == 1
    assert summary.cases[0].comparable is False
    assert summary.cases[0].task_success is False


def test_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of cases"):
        summarize_agent_metrics(
            [AgentTrajectoryExpectation(should_complete=True)],
            [AgentTrajectory(), AgentTrajectory()],
        )


def test_metrics_keyword_matching_is_case_insensitive() -> None:
    expectation = AgentTrajectoryExpectation(
        should_complete=True,
        expected_keywords=("cpu",),
        steps_budget=3,
    )
    trajectory = AgentTrajectory(response="CPU 使用率 85%。")

    summary = summarize_agent_metrics([expectation], [trajectory])

    assert summary.cases[0].keyword_coverage == 1.0


# ---------------------------------------------------------------------------
# Runner（dry-run / fake adapter / 失败不阻断 / 报告）
# ---------------------------------------------------------------------------


async def test_runner_with_builtin_dataset(tmp_path: Path) -> None:
    """内置评测集必须能被 dry-run adapter 全量跑通（回归基线）。"""

    dataset_path = Path("eval_sets/agent_cases.yaml")
    if not dataset_path.exists():
        pytest.skip("builtin agent eval dataset not found")

    result = await load_and_run_agent(dataset_path)

    assert result.adapter_name == "dry-run"
    assert result.aggregate.case_count == len(load_agent_cases(dataset_path))
    assert result.aggregate.comparable_case_count >= 1
    assert len(result.cases) == result.aggregate.case_count
    # dry-run 无轨迹：可完成任务不应被误判为成功。
    assert result.aggregate.task_success_rate == 0.0
    assert all(case.error is None for case in result.cases)


async def test_runner_case_failure_does_not_block(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            _raw_case(id="boom"),
            _raw_case(id="ok", task="查询内存使用率", expected_keywords=["内存"]),
        ],
    )

    class _FlakyAdapter:
        async def evaluate_case(self, case: AgentCase) -> AgentTrajectory:
            if case.id == "boom":
                raise RuntimeError("boom with internal url http://10.0.0.1")
            return AgentTrajectory(
                steps=("s1",),
                tool_calls=("query_memory_metrics",),
                response="内存使用率 70%。",
            )

    result = await load_and_run_agent(
        dataset_path,
        adapter=_FlakyAdapter(),  # type: ignore[arg-type]
        adapter_name="flaky",
    )

    boom = next(case for case in result.cases if case.case_id == "boom")
    ok = next(case for case in result.cases if case.case_id == "ok")

    assert boom.error is not None
    assert boom.error.code == "AGENT_EVAL_CASE_FAILED"
    assert "10.0.0.1" not in boom.error.message
    assert ok.error is None
    assert ok.trajectory_metric.task_success is True
    assert result.aggregate.failed_case_count == 1
    assert result.aggregate.comparable_case_count == 1


async def test_runner_preserves_sse_error_code(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path, [_raw_case()])

    class _SseErrorAdapter:
        async def evaluate_case(self, case: AgentCase) -> AgentTrajectory:
            raise AgentCaseExecutionError("AGENT_MAX_STEP_EXCEEDED", "步数超限。")

    result = await load_and_run_agent(
        dataset_path,
        adapter=_SseErrorAdapter(),  # type: ignore[arg-type]
    )

    assert result.cases[0].error is not None
    assert result.cases[0].error.code == "AGENT_MAX_STEP_EXCEEDED"


async def test_runner_writes_reports(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            _raw_case(),
            _raw_case(
                id="refusal_001",
                task="写诗",
                should_complete=False,
                required_tools=[],
                forbidden_tools=["search_log"],
                expected_keywords=[],
                case_type="out_of_scope",
            ),
        ],
    )

    result = await load_and_run_agent(dataset_path)

    output_dir = tmp_path / "reports"
    paths = write_agent_evaluation_report(
        result,
        output_dir=output_dir,
        config={"dataset": str(dataset_path), "adapter": "dry-run"},
    )

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert paths.trend_path.name == "agent_trends.jsonl"
    assert paths.trend_path.exists()

    report = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "agent_eval_report.v1"
    assert report["report_type"] == "agent_trajectory_evaluation"
    assert report["aggregate"]["case_count"] == 2
    assert len(report["case_diffs"]) == 2
    assert report["failed_cases"] == []

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Agent Trajectory Evaluation Report" in markdown
    assert "task_success_rate" in markdown

    trend_lines = paths.trend_path.read_text(encoding="utf-8").strip().splitlines()
    trend = json.loads(trend_lines[-1])
    assert trend["report_type"] == "agent_trajectory_evaluation"
    assert trend["case_count"] == 2

    # 二次写入应追加趋势而不是覆盖。
    write_agent_evaluation_report(result, output_dir=output_dir)
    assert len(paths.trend_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_cli_dry_run_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dataset_path = _write_dataset(tmp_path, [_raw_case()])
    output_dir = tmp_path / "cli_reports"

    exit_code = main(["--dataset", str(dataset_path), "--output", str(output_dir)])

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["run_id"].startswith("agent_eval_")
    assert stdout["adapter"] == "dry-run"
    assert stdout["reports"]["trend_path"].endswith("agent_trends.jsonl")
    assert (output_dir / "agent_trends.jsonl").exists()


def test_cli_invalid_dataset_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--dataset", str(tmp_path / "missing.yaml")])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "AGENT_EVAL_DATASET_INVALID"


def test_cli_report_write_failure_returns_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    dataset_path = _write_dataset(tmp_path, [_raw_case()])

    def _raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise_os_error)

    exit_code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "out")])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "AGENT_EVAL_REPORT_WRITE_FAILED"
    assert "disk full" not in payload["error"]["message"]


# ---------------------------------------------------------------------------
# 真实 adapter（fake AIOpsService，验证 SSE 轨迹提取）
# ---------------------------------------------------------------------------


@dataclass
class _FakeAIOpsService:
    """模拟 AIOpsService 的公开边界：SSE execute + graph.get_state。"""

    events: list[dict]
    tool_evidence: list[dict] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)

    def execute(self, user_input: str, session_id: str = "default") -> AsyncIterator[dict]:
        from app.core.request_context import get_request_context_or_none

        self.sessions.append(session_id)
        ctx = get_request_context_or_none()
        if ctx is not None:
            self.request_ids.append(ctx.request_id)

        async def _stream() -> AsyncIterator[dict]:
            for event in self.events:
                yield event

        return _stream()

    @property
    def graph(self) -> SimpleNamespace:
        service = self

        class _Graph:
            def get_state(self, config: Mapping) -> SimpleNamespace:
                assert config["configurable"]["thread_id"] in service.sessions
                return SimpleNamespace(
                    values={"tool_evidence": list(service.tool_evidence)}
                )

        return SimpleNamespace(get_state=_Graph().get_state)


async def test_aiops_adapter_extracts_trajectory_from_sse_and_state() -> None:
    service = _FakeAIOpsService(
        events=[
            {"type": "plan", "plan": ["步骤A", "步骤B"]},
            {"type": "step_complete", "current_step": "步骤A"},
            {"type": "step_complete", "current_step": "  步骤B  "},
            {"type": "report", "report": "草稿"},
            {"type": "complete", "response": "CPU 使用率 85%。"},
        ],
        tool_evidence=[
            {"step": "步骤A", "tool_name": "query_cpu_metrics", "usable": True},
            {"step": "步骤B", "tool_name": "search_log", "usable": False},
        ],
    )
    adapter = AIOpsServiceAgentEvaluationAdapter(service)  # type: ignore[arg-type]
    case = AgentCase.model_validate(_raw_case())

    trajectory = await adapter.evaluate_case(case)

    assert trajectory.steps == ("步骤A", "步骤B")
    assert trajectory.response == "CPU 使用率 85%。"
    assert trajectory.tool_calls == ("query_cpu_metrics", "search_log")
    assert trajectory.failed_tool_calls == ("search_log",)
    # 独立 session：避免污染真实会话 checkpoint。
    assert service.sessions[0].startswith("agent_eval_case_001_")
    # 每 case 注入独立 RequestContext：tool_evidence 采集按 request_id 归档的前提。
    assert len(service.request_ids) == 1
    assert service.request_ids[0].startswith("req_")


async def test_aiops_adapter_resets_request_context_after_case() -> None:
    from app.core.request_context import get_request_context_or_none

    service = _FakeAIOpsService(events=[{"type": "complete", "response": "完成。"}])
    adapter = AIOpsServiceAgentEvaluationAdapter(service)  # type: ignore[arg-type]
    case = AgentCase.model_validate(_raw_case())

    await adapter.evaluate_case(case)

    # context 按 case 设置并在结束后恢复，避免泄漏到 runner 的后续 case 或调用方。
    assert get_request_context_or_none() is None


async def test_aiops_adapter_maps_sse_error_event() -> None:
    service = _FakeAIOpsService(
        events=[
            {"type": "step_complete", "current_step": "步骤A"},
            {"type": "error", "code": "LLM_TIMEOUT", "message": "内部超时"},
        ],
    )
    adapter = AIOpsServiceAgentEvaluationAdapter(service)  # type: ignore[arg-type]
    case = AgentCase.model_validate(_raw_case())

    with pytest.raises(AgentCaseExecutionError) as exc_info:
        await adapter.evaluate_case(case)

    assert exc_info.value.code == "LLM_TIMEOUT"


async def test_aiops_adapter_state_read_failure_is_fail_open() -> None:
    class _BrokenGraphService(_FakeAIOpsService):
        @property
        def graph(self) -> SimpleNamespace:
            raise RuntimeError("checkpoint unavailable")

    service = _BrokenGraphService(
        events=[{"type": "complete", "response": "任务完成。"}],
    )
    adapter = AIOpsServiceAgentEvaluationAdapter(service)  # type: ignore[arg-type]
    case = AgentCase.model_validate(_raw_case())

    trajectory = await adapter.evaluate_case(case)

    # 工具证据读取失败不阻断主轨迹。
    assert trajectory.response == "任务完成。"
    assert trajectory.tool_calls == ()


async def test_dry_run_adapter_never_invokes_service() -> None:
    adapter = DryRunAgentEvaluationAdapter()
    case = AgentCase.model_validate(_raw_case())
    refusal_case = AgentCase.model_validate(
        _raw_case(
            id="refusal_001",
            should_complete=False,
            required_tools=[],
            expected_keywords=[],
            case_type="out_of_scope",
        )
    )

    assert (await adapter.evaluate_case(case)).response == "dry-run 未执行真实 Agent pipeline。"
    assert (await adapter.evaluate_case(refusal_case)).response == refusal_case.golden_answer


async def test_runner_with_fake_aiops_service_end_to_end(tmp_path: Path) -> None:
    """runner + aiops adapter 端到端：SSE 轨迹驱动完整指标与报告。"""

    dataset_path = _write_dataset(
        tmp_path,
        [
            _raw_case(),
            _raw_case(
                id="time_001",
                task="现在几点",
                required_tools=["get_current_time"],
                forbidden_tools=["search_log"],
                expected_keywords=["时间"],
                max_steps_budget=2,
                case_type="tool_query",
                golden_answer="应调用 get_current_time。",
            ),
        ],
    )

    class _RoutingService(_FakeAIOpsService):
        def __init__(self) -> None:
            super().__init__(events=[])

        def execute(self, user_input: str, session_id: str = "default") -> AsyncIterator[dict]:
            if "CPU" in user_input:
                self.events = [
                    {"type": "step_complete", "current_step": "查 CPU 指标"},
                    {"type": "complete", "response": "CPU 使用率 85%，建议排查进程。"},
                ]
                self.tool_evidence = [{"tool_name": "query_cpu_metrics", "usable": True}]
            else:
                self.events = [
                    {"type": "step_complete", "current_step": "查时间"},
                    {"type": "complete", "response": "当前时间是 12:00。"},
                ]
                self.tool_evidence = [
                    {"tool_name": "get_current_time", "usable": True},
                    {"tool_name": "search_log", "usable": False},
                ]
            return super().execute(user_input, session_id)

    runner = AgentEvaluationRunner(
        adapter=AIOpsServiceAgentEvaluationAdapter(_RoutingService()),  # type: ignore[arg-type]
        adapter_name="aiops",
    )
    cases = load_agent_cases(dataset_path)
    result = await runner.run(cases, dataset_path=dataset_path)

    cpu_case = next(case for case in result.cases if case.case_id == "case_001")
    time_case = next(case for case in result.cases if case.case_id == "time_001")

    assert cpu_case.trajectory_metric.task_success is True
    assert cpu_case.trajectory_metric.required_tool_coverage == 1.0
    assert time_case.trajectory_metric.task_success is True
    assert time_case.trajectory_metric.forbidden_tool_violation_count == 1
    assert time_case.trajectory_metric.tool_failure_rate == 0.5

    assert result.aggregate.task_success_rate == 1.0
    assert result.aggregate.forbidden_tool_violation_count == 1

    paths = write_agent_evaluation_report(result, output_dir=tmp_path / "reports")
    report = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["task_success_rate"] == 1.0
    assert report["adapter"] == "aiops"
