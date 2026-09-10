"""ISSUE-033/034 LLMJudge、evaluation runner 与报告输出测试。

这些测试只使用内存 fake adapter 和 fake judge，不访问真实 Milvus、DashScope、
MCP server 或网络。这样 runner 可以作为离线/CI 工具独立验证，不进入线上 FastAPI
请求路径，也不会把 judge 成本或不稳定性引入默认回归。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.errors import RagEmptyResultError
from app.core.request_context import get_request_context_or_none
from evaluation.datasets import RagCase, load_rag_cases
from evaluation.judge import JudgeError, JudgeInput, JudgeResult, LLMJudge
from evaluation.runner import (
    EvaluationAdapterResult,
    EvaluationReportWriteError,
    RagEvaluationRunner,
    RagPipelineEvaluationAdapter,
    main,
    write_evaluation_report,
)


class _StaticAdapter:
    """按 case id 返回固定结果的 runner adapter fake。"""

    def __init__(self, results: dict[str, EvaluationAdapterResult]) -> None:
        self.results = dict(results)
        self.calls: list[str] = []

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        self.calls.append(case.id)
        return self.results.get(
            case.id,
            EvaluationAdapterResult(
                retrieved_ids=(),
                answer="",
                context="",
            ),
        )


class _FailingFirstAdapter:
    """首个 case 抛异常，后续 case 正常返回，用来验证失败继续执行。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        self.calls.append(case.id)
        if len(self.calls) == 1:
            raise RuntimeError("raw sk-secret from http://internal.eval")
        return EvaluationAdapterResult(
            retrieved_ids=tuple(case.expected_doc_ids),
            answer=case.golden_answer,
            context="safe context",
        )


class _FakeJudge:
    """可选择抛错的 judge fake，避免测试依赖真实 LLM。"""

    def __init__(self, *, fail_case_id: str | None = None) -> None:
        self.fail_case_id = fail_case_id
        self.calls: list[str] = []

    def judge(self, payload: JudgeInput) -> JudgeResult:
        self.calls.append(payload.case_id)
        if payload.case_id == self.fail_case_id:
            raise RuntimeError("raw judge password from http://internal.judge")
        return JudgeResult(
            status="scored",
            faithfulness=0.9,
            answer_correctness=0.8,
            no_answer=1.0 if not payload.should_answer else 0.0,
            reasoning="fake stable reasoning",
        )


class _FakeJudgeModel:
    """LLMJudge 解析测试用模型 fake。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return type("FakeMessage", (), {"content": self.payload})()


def _write_dataset(path: Path) -> list[RagCase]:
    path.write_text(
        """
- id: answer_case
  question: "CPU 怎么排查？"
  expected_doc_ids: ["doc_cpu"]
  expected_keywords: ["CPU"]
  should_answer: true
  case_type: answer
  tags: ["cpu"]
  difficulty: easy
  golden_answer: "先查看 CPU 指标和进程日志。"
- id: no_answer_case
  question: "公司客户名单是什么？"
  expected_doc_ids: []
  expected_keywords: []
  should_answer: false
  case_type: no_answer
  tags: ["no-answer"]
  difficulty: medium
  golden_answer: "知识库没有客户名单，应拒绝编造。"
""".strip(),
        encoding="utf-8",
    )
    return load_rag_cases(path)


def test_runner_dry_run_with_disabled_judge_returns_structured_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rag_cases.yaml"
    cases = _write_dataset(dataset_path)
    adapter = _StaticAdapter(
        {
            "answer_case": EvaluationAdapterResult(
                retrieved_ids=("doc_cpu#000001",),
                answer="先查看 CPU 指标和进程日志。",
                context="CPU context",
            ),
            "no_answer_case": EvaluationAdapterResult(
                retrieved_ids=(),
                answer="知识库没有相关依据。",
                context="",
            ),
        }
    )

    result = RagEvaluationRunner(adapter=adapter, judge=None, top_k=3).run(cases)

    assert adapter.calls == ["answer_case", "no_answer_case"]
    assert result.run_id.startswith("eval_")
    assert result.aggregate.case_count == 2
    assert result.aggregate.comparable_case_count == 1
    assert result.aggregate.hit_rate_at_k == 1.0
    assert result.aggregate.recall_at_k == 1.0
    assert result.aggregate.mrr == 1.0
    assert [case.case_id for case in result.cases] == ["answer_case", "no_answer_case"]
    assert all(case.judge is None for case in result.cases)
    assert all(case.error is None for case in result.cases)


def test_runner_records_case_error_and_continues_without_raw_exception(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    adapter = _FailingFirstAdapter()

    result = RagEvaluationRunner(adapter=adapter, judge=None, top_k=3).run(cases)

    assert adapter.calls == ["answer_case", "no_answer_case"]
    assert result.cases[0].error is not None
    assert result.cases[0].error.code == "EVAL_CASE_FAILED"
    assert result.cases[1].error is None

    payload_text = str(result.to_dict())
    assert "raw sk-secret" not in payload_text
    assert "http://internal.eval" not in payload_text
    assert "评估用例执行失败" in payload_text


def test_runner_uses_fake_judge_and_ignores_judge_failures(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    adapter = _StaticAdapter(
        {
            "answer_case": EvaluationAdapterResult(
                retrieved_ids=("doc_cpu",),
                answer="先查看 CPU 指标和进程日志。",
                context="CPU context",
            ),
            "no_answer_case": EvaluationAdapterResult(
                retrieved_ids=(),
                answer="无法基于知识库回答。",
                context="",
            ),
        }
    )
    judge = _FakeJudge(fail_case_id="no_answer_case")

    result = RagEvaluationRunner(adapter=adapter, judge=judge, top_k=3).run(cases)

    assert judge.calls == ["answer_case", "no_answer_case"]
    assert result.cases[0].judge is not None
    assert result.cases[0].judge.status == "scored"
    assert result.cases[1].judge is None
    assert result.cases[1].judge_error is not None
    assert result.cases[1].judge_error.code == "EVAL_JUDGE_FAILED"
    assert result.aggregate.hit_rate_at_k == 1.0

    payload_text = str(result.to_dict())
    assert "raw judge password" not in payload_text
    assert "http://internal.judge" not in payload_text


def test_llm_judge_uses_fixed_rubric_temperature_and_parses_scores() -> None:
    model = _FakeJudgeModel(
        """
{
  "faithfulness": 0.75,
  "answer_correctness": 0.5,
  "no_answer": 0.0,
  "reasoning": "回答部分依据上下文"
}
""".strip()
    )
    judge = LLMJudge(model_client=model, model_name="fake-judge", temperature=0.9)

    result = judge.judge(
        JudgeInput(
            case_id="answer_case",
            question="CPU 怎么排查？",
            golden_answer="先查看 CPU 指标和进程日志。",
            answer="先查看 CPU 指标。",
            context="CPU context",
            should_answer=True,
            expected_doc_ids=("doc_cpu",),
            retrieved_ids=("doc_cpu",),
        )
    )

    assert judge.temperature == 0.0
    assert result.status == "scored"
    assert result.faithfulness == pytest.approx(0.75)
    assert result.answer_correctness == pytest.approx(0.5)
    assert result.no_answer == pytest.approx(0.0)
    assert "faithfulness" in model.prompts[0]
    assert "answer_correctness" in model.prompts[0]
    assert "no_answer" in model.prompts[0]


def test_cli_enabled_judge_dependency_failure_keeps_lightweight_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "rag_cases.yaml"
    _write_dataset(dataset_path)

    def _raise_judge_error() -> None:
        raise JudgeError("JUDGE_PROVIDER_UNAVAILABLE", "LLM judge 模型依赖不可用。")

    monkeypatch.setattr("evaluation.runner.LLMJudge", _raise_judge_error)

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--judge",
            "enabled",
            "--output",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    payload = capsys.readouterr().out
    assert '"judge_enabled": true' in payload
    assert '"judge_failed_case_count": 2' in payload
    assert "LLM judge 模型依赖不可用。" in payload


def test_report_writer_creates_json_markdown_and_trend(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    adapter = _StaticAdapter(
        {
            "answer_case": EvaluationAdapterResult(
                retrieved_ids=("doc_cpu#000001", "doc_extra"),
                answer="先查看 CPU 指标和进程日志。",
                context="CPU context",
                citations=(
                    {
                        "citation_id": "C1",
                        "doc_id": "doc_cpu",
                        "chunk_id": "doc_cpu#000001",
                        "source_path": "aiops-docs/cpu.md",
                    },
                ),
            ),
            "no_answer_case": EvaluationAdapterResult(
                retrieved_ids=(),
                answer="知识库没有相关依据。",
                context="",
            ),
        }
    )
    result = RagEvaluationRunner(
        adapter=adapter,
        judge=None,
        top_k=3,
        run_id="eval_test_report",
    ).run(cases, dataset_path=tmp_path / "rag_cases.yaml")

    paths = write_evaluation_report(
        result,
        output_dir=tmp_path / "reports",
        config={
            "adapter": "dry-run",
            "judge": "disabled",
            "top_k": 3,
        },
    )

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert paths.trend_path.exists()

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rag_eval_report.v1"
    assert payload["run_id"] == "eval_test_report"
    assert payload["aggregate"]["hit_rate_at_k"] == pytest.approx(1.0)
    assert payload["dataset"]["case_count"] == 2
    assert payload["config"]["judge"] == "disabled"
    assert payload["case_diffs"][0]["case_id"] == "answer_case"
    assert payload["case_diffs"][0]["missing_expected_doc_ids"] == []
    assert payload["case_diffs"][0]["unexpected_retrieved_ids"] == ["doc_extra"]

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "# AegisOps Agent RAG Evaluation Report" in markdown
    assert "## 配置" in markdown
    assert "## 数据集" in markdown
    assert "## 指标汇总" in markdown
    assert "## Per-case Diff" in markdown
    assert "answer_case" in markdown
    assert "doc_cpu#000001" in markdown

    trend_lines = paths.trend_path.read_text(encoding="utf-8").splitlines()
    assert len(trend_lines) == 1
    trend_payload = json.loads(trend_lines[0])
    assert trend_payload["run_id"] == "eval_test_report"
    assert trend_payload["hit_rate_at_k"] == pytest.approx(1.0)
    assert trend_payload["failed_case_count"] == 0


def test_report_writer_records_failed_cases_and_judge_failures(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    result = RagEvaluationRunner(
        adapter=_FailingFirstAdapter(),
        judge=_FakeJudge(fail_case_id="no_answer_case"),
        top_k=3,
        run_id="eval_test_failed_cases",
    ).run(cases, dataset_path=tmp_path / "rag_cases.yaml")

    paths = write_evaluation_report(
        result,
        output_dir=tmp_path / "reports",
        config={"adapter": "fake", "judge": "fake"},
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    failed_cases = payload["failed_cases"]
    assert failed_cases == [
        {
            "case_id": "answer_case",
            "failure_type": "case",
            "code": "EVAL_CASE_FAILED",
            "message": "评估用例执行失败，已跳过该用例的 pipeline 输出。",
        },
        {
            "case_id": "no_answer_case",
            "failure_type": "judge",
            "code": "EVAL_JUDGE_FAILED",
            "message": "LLM judge 失败，已保留轻量检索指标。",
        },
    ]

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "## 失败 Case" in markdown
    assert "EVAL_CASE_FAILED" in markdown
    assert "EVAL_JUDGE_FAILED" in markdown
    assert "raw sk-secret" not in markdown
    assert "raw judge password" not in markdown


def test_report_writer_appends_trend_for_each_run(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    output_dir = tmp_path / "reports"

    for run_id in ("eval_test_trend_1", "eval_test_trend_2"):
        result = RagEvaluationRunner(
            adapter=_StaticAdapter({}),
            judge=None,
            top_k=3,
            run_id=run_id,
        ).run(cases, dataset_path=tmp_path / "rag_cases.yaml")
        write_evaluation_report(
            result,
            output_dir=output_dir,
            config={"adapter": "dry-run", "judge": "disabled"},
        )

    trend_path = output_dir / "trends.jsonl"
    trend_lines = trend_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in trend_lines] == [
        "eval_test_trend_1",
        "eval_test_trend_2",
    ]


def test_report_write_failure_returns_clear_error_code(tmp_path: Path) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    result = RagEvaluationRunner(
        adapter=_StaticAdapter({}),
        judge=None,
        top_k=3,
        run_id="eval_test_report_error",
    ).run(cases, dataset_path=tmp_path / "rag_cases.yaml")
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("block mkdir", encoding="utf-8")

    with pytest.raises(EvaluationReportWriteError) as exc_info:
        write_evaluation_report(
            result,
            output_dir=output_file,
            config={"adapter": "dry-run", "judge": "disabled"},
        )

    assert exc_info.value.code == "EVAL_REPORT_WRITE_FAILED"
    assert "FileExistsError" not in exc_info.value.safe_message
    assert "block mkdir" not in exc_info.value.safe_message


def test_cli_report_write_failure_returns_error_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "rag_cases.yaml"
    _write_dataset(dataset_path)
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("block mkdir", encoding="utf-8")

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--judge",
            "disabled",
            "--output",
            str(output_file),
        ]
    )

    assert exit_code == 3
    payload = capsys.readouterr().out
    assert '"code": "EVAL_REPORT_WRITE_FAILED"' in payload
    assert "FileExistsError" not in payload
    assert "block mkdir" not in payload


class _FakePipelineService:
    """pipeline adapter 测试用 fake：记录调用参数，返回固定 Stage 3B 结果。"""

    def __init__(self, *, raise_empty: bool = False) -> None:
        self.raise_empty = raise_empty
        self.calls: list[dict[str, object]] = []

    async def query_with_citations(
        self,
        question: str,
        session_id: str = "default",
        conversation_context: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.calls.append(
            {"question": question, "session_id": session_id, "ctx": ctx}
        )
        if self.raise_empty:
            raise RagEmptyResultError()
        return SimpleNamespace(
            answer="先查看 CPU 指标和进程日志 [C1]。",
            citations=(
                {
                    "citation_id": "C1",
                    "doc_id": "doc_cpu",
                    "chunk_id": "doc_cpu#000001",
                    "file_name": "cpu.md",
                    "content_preview": "CPU 排查步骤：先看 top，再看进程日志。",
                },
            ),
            context_text="证据1: CPU 排查步骤：先看 top，再看进程日志。",
        )


def _pipeline_case(case_id: str = "answer_case") -> RagCase:
    return RagCase(
        id=case_id,
        question="CPU 怎么排查？",
        expected_doc_ids=["doc_cpu"],
        expected_keywords=["CPU"],
        should_answer=True,
        case_type="answer",
        tags=["cpu"],
        difficulty="easy",
        golden_answer="先查看 CPU 指标和进程日志。",
    )


def test_pipeline_adapter_maps_result_and_isolates_sessions() -> None:
    service = _FakePipelineService()
    adapter = RagPipelineEvaluationAdapter(service)
    try:
        first = adapter.evaluate_case(_pipeline_case())
        second = adapter.evaluate_case(_pipeline_case())

        assert first.answer == "先查看 CPU 指标和进程日志 [C1]。"
        assert first.context == "证据1: CPU 排查步骤：先看 top，再看进程日志。"
        assert first.retrieved_ids == ("doc_cpu#000001",)
        assert first.citations[0]["citation_id"] == "C1"
        # 每 case 独立 session_id + 显式 ctx，避免串扰线上记忆/trace。
        assert len(service.calls) == 2
        first_call, second_call = service.calls
        assert str(first_call["session_id"]).startswith("rag_eval_answer_case_")
        assert str(second_call["session_id"]).startswith("rag_eval_answer_case_")
        assert first_call["session_id"] != second_call["session_id"]
        assert first_call["ctx"] is not None
        assert first_call["question"] == "CPU 怎么排查？"
    finally:
        adapter.close()


def test_pipeline_adapter_resets_request_context_after_case() -> None:
    adapter = RagPipelineEvaluationAdapter(_FakePipelineService())
    try:
        adapter.evaluate_case(_pipeline_case())
        # 离线 CLI 不能把 eval case 的 RequestContext 泄漏给后续 case 或进程。
        assert get_request_context_or_none() is None
    finally:
        adapter.close()


def test_pipeline_adapter_maps_refusal_path_to_stable_answer() -> None:
    adapter = RagPipelineEvaluationAdapter(_FakePipelineService(raise_empty=True))
    try:
        case = RagCase(
            id="no_answer_case",
            question="公司客户名单是什么？",
            expected_doc_ids=[],
            expected_keywords=[],
            should_answer=False,
            case_type="no_answer",
            tags=["no-answer"],
            difficulty="medium",
            golden_answer="知识库没有客户名单，应拒绝编造。",
        )

        result = adapter.evaluate_case(case)

        # 拒答是正确行为：使用线上 RAG_EMPTY_RESULT 的用户文案，judge 的
        # no_answer 指标据此评分，而不是把 case 记为 EVAL_CASE_FAILED。
        assert result.answer == RagPipelineEvaluationAdapter.REFUSAL_ANSWER
        assert result.answer == "未找到足够相关的知识库内容。"
        assert result.retrieved_ids == ()
        assert result.citations == ()
        assert get_request_context_or_none() is None
    finally:
        adapter.close()


def test_llm_judge_prompt_includes_citations_and_parses_citation_correctness() -> None:
    model = _FakeJudgeModel(
        """
{
  "faithfulness": 0.9,
  "answer_correctness": 0.8,
  "no_answer": 0.0,
  "citation_correctness": 1.0,
  "reasoning": "引用标记与证据一致"
}
""".strip()
    )
    judge = LLMJudge(model_client=model, model_name="fake-judge")

    result = judge.judge(
        JudgeInput(
            case_id="answer_case",
            question="CPU 怎么排查？",
            golden_answer="先查看 CPU 指标和进程日志。",
            answer="先查看 CPU 指标和进程日志 [C1]。",
            context="CPU context",
            should_answer=True,
            expected_doc_ids=("doc_cpu",),
            retrieved_ids=("doc_cpu#000001",),
            citations=(
                {
                    "citation_id": "C1",
                    "doc_id": "doc_cpu",
                    "chunk_id": "doc_cpu#000001",
                    "file_name": "cpu.md",
                    "content_preview": "CPU 排查步骤。",
                },
            ),
        )
    )

    assert result.citation_correctness == pytest.approx(1.0)
    prompt = model.prompts[0]
    assert "citation_correctness" in prompt
    assert "citations:" in prompt
    assert "doc_cpu" in prompt
    assert "CPU 排查步骤。" in prompt


def test_llm_judge_treats_missing_citation_correctness_as_not_evaluated() -> None:
    model = _FakeJudgeModel(
        """
{
  "faithfulness": 0.9,
  "answer_correctness": 0.8,
  "no_answer": 0.0,
  "reasoning": "旧版模型输出没有 citation 字段"
}
""".strip()
    )
    judge = LLMJudge(model_client=model, model_name="fake-judge")

    result = judge.judge(
        JudgeInput(
            case_id="answer_case",
            question="CPU 怎么排查？",
            golden_answer="先查看 CPU 指标和进程日志。",
            answer="先查看 CPU 指标。",
            context="CPU context",
            should_answer=True,
        )
    )

    # 缺字段表示"未评估"而不是 0 分，聚合阶段会跳过该 case 的该指标。
    assert result.citation_correctness is None
    assert "citations:\nnone" in model.prompts[0]


class _CitationFakeJudge:
    """按 citations 是否存在返回 citation_correctness 的 judge fake。"""

    def __init__(self) -> None:
        self.payloads: list[JudgeInput] = []

    def judge(self, payload: JudgeInput) -> JudgeResult:
        self.payloads.append(payload)
        return JudgeResult(
            status="scored",
            faithfulness=0.9,
            answer_correctness=0.8,
            no_answer=0.0,
            reasoning="fake",
            citation_correctness=0.5 if payload.citations else None,
        )


def test_runner_passes_citations_to_judge_and_aggregates_citation_correctness(
    tmp_path: Path,
) -> None:
    cases = _write_dataset(tmp_path / "rag_cases.yaml")
    adapter = _StaticAdapter(
        {
            "answer_case": EvaluationAdapterResult(
                retrieved_ids=("doc_cpu#000001",),
                answer="先查看 CPU 指标和进程日志 [C1]。",
                context="CPU context",
                citations=(
                    {
                        "citation_id": "C1",
                        "doc_id": "doc_cpu",
                        "chunk_id": "doc_cpu#000001",
                    },
                ),
            ),
            "no_answer_case": EvaluationAdapterResult(
                retrieved_ids=(),
                answer="知识库没有相关依据。",
                context="",
            ),
        }
    )
    judge = _CitationFakeJudge()

    result = RagEvaluationRunner(
        adapter=adapter,
        judge=judge,
        top_k=3,
        run_id="eval_citation_test",
    ).run(cases, dataset_path=tmp_path / "rag_cases.yaml")

    # 有 citations 的 case 拿到 citation_correctness；无 citations 的记 None，
    # 聚合只对已评估值求平均，None 不会摊薄基线。
    assert len(judge.payloads) == 2
    citations_by_case = {
        payload.case_id: payload.citations for payload in judge.payloads
    }
    assert citations_by_case["answer_case"][0]["chunk_id"] == "doc_cpu#000001"
    assert citations_by_case["no_answer_case"] == ()
    assert result.aggregate.judge_average_citation_correctness == pytest.approx(0.5)
    assert result.aggregate.to_dict()["judge_average_citation_correctness"] == (
        pytest.approx(0.5)
    )
