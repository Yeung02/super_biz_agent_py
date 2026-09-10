"""RAG 离线 evaluation runner。

runner 默认使用 dry-run adapter 和 disabled judge，因此命令
`python -m evaluation.runner --dataset eval_sets/rag_cases.yaml --judge disabled --output eval_reports`
不会连接真实 Milvus、DashScope、MCP server 或网络。ISSUE-034 在此基础上只增加离线
JSON/Markdown 报告和趋势文件，不进入线上 FastAPI 请求路径。
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar, cast
from uuid import uuid4

from evaluation.datasets import RagCase, RagDatasetError, load_rag_cases
from evaluation.judge import JudgeError, JudgeInput, JudgeResult, LLMJudge
from evaluation.rag_metrics import (
    RetrievalCaseMetric,
    RetrievalMetricsSummary,
    summarize_retrieval_metrics,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

LOGGER = logging.getLogger(__name__)
_DEFAULT_SAFE_CASE_ERROR = "评估用例执行失败，已跳过该用例的 pipeline 输出。"
_DEFAULT_SAFE_JUDGE_ERROR = "LLM judge 失败，已保留轻量检索指标。"
_DEFAULT_DATASET_PATH = "eval_sets/rag_cases.yaml"
_DEFAULT_RETRIEVAL_K = 5
_DEFAULT_JUDGE_ENABLED = False
_DEFAULT_REPORT_OUTPUT_DIR = "eval_reports"
_REPORT_SCHEMA_VERSION = "rag_eval_report.v1"
_TREND_FILE_NAME = "trends.jsonl"
_ConfigValue = TypeVar("_ConfigValue")


class EvaluationAdapterLike(Protocol):
    """runner 调用检索或完整 RAG pipeline 的最小 adapter 协议。"""

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        """执行单条 case，并返回检索 ID、答案和上下文。"""


class JudgeLike(Protocol):
    """runner 依赖的 judge 最小协议。"""

    def judge(self, payload: JudgeInput) -> JudgeResult:
        """对单条 case 的答案进行评分。"""


class RetrieverLike(Protocol):
    """`RagRetrieverEvaluationAdapter` 需要的最小检索协议。"""

    def retrieve(self, query: str) -> object:
        """按问题返回包含 chunks 字段的检索结果。"""


@dataclass(frozen=True)
class EvaluationAdapterResult:
    """adapter 返回的单条 case 输出。

    `retrieved_ids` 可以是 doc_id 或 chunk_id；轻量指标已兼容 `doc_id#chunk` 命中父
    doc_id 的语义。`answer/context/citations` 仅供 judge 和后续报告使用，不作为线上 API
    schema 暴露。
    """

    retrieved_ids: Sequence[str] = ()
    answer: str = ""
    context: str = ""
    citations: Sequence[Mapping[str, object]] = ()

    def normalized_retrieved_ids(self) -> tuple[str, ...]:
        """返回去空白后的 retrieved ids。"""

        return tuple(_normalize_text_items(self.retrieved_ids))

    def to_dict(self) -> JsonObject:
        """转换为结构化结果，保证 citations 只含 JSON-safe 字段。"""

        return {
            "retrieved_ids": list(self.normalized_retrieved_ids()),
            "answer": self.answer,
            "context": self.context,
            "citations": [_json_safe_mapping(citation) for citation in self.citations],
        }


@dataclass(frozen=True)
class EvaluationErrorInfo:
    """评估错误的安全输出结构。"""

    code: str
    message: str

    def to_dict(self) -> JsonObject:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class CaseEvaluationResult:
    """单条 eval case 的完整结构化结果。"""

    case_id: str
    question: str
    should_answer: bool
    expected_doc_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    answer: str
    context: str
    citations: tuple[JsonObject, ...]
    retrieval_metric: RetrievalCaseMetric
    latency_ms: float
    error: EvaluationErrorInfo | None = None
    judge: JudgeResult | None = None
    judge_error: EvaluationErrorInfo | None = None

    def to_dict(self) -> JsonObject:
        """转换为后续报告可复用的结构，不包含原始异常全文。"""

        return {
            "case_id": self.case_id,
            "question": self.question,
            "should_answer": self.should_answer,
            "expected_doc_ids": list(self.expected_doc_ids),
            "retrieved_ids": list(self.retrieved_ids),
            "answer": self.answer,
            "context": self.context,
            "citations": list(self.citations),
            "retrieval_metric": _case_metric_to_dict(self.retrieval_metric),
            "latency_ms": self.latency_ms,
            "error": self.error.to_dict() if self.error is not None else None,
            "judge": self.judge.to_dict() if self.judge is not None else None,
            "judge_error": (
                self.judge_error.to_dict() if self.judge_error is not None else None
            ),
        }


@dataclass(frozen=True)
class EvaluationAggregateResult:
    """全量评估汇总指标。"""

    k: int
    case_count: int
    comparable_case_count: int
    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    failed_case_count: int
    judge_case_count: int
    judge_failed_case_count: int
    judge_average_faithfulness: float | None
    judge_average_answer_correctness: float | None
    judge_average_no_answer: float | None
    judge_average_citation_correctness: float | None = None

    def to_dict(self) -> JsonObject:
        return {
            "k": self.k,
            "case_count": self.case_count,
            "comparable_case_count": self.comparable_case_count,
            "hit_rate_at_k": self.hit_rate_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "failed_case_count": self.failed_case_count,
            "judge_case_count": self.judge_case_count,
            "judge_failed_case_count": self.judge_failed_case_count,
            "judge_average_faithfulness": self.judge_average_faithfulness,
            "judge_average_answer_correctness": self.judge_average_answer_correctness,
            "judge_average_no_answer": self.judge_average_no_answer,
            "judge_average_citation_correctness": self.judge_average_citation_correctness,
        }


@dataclass(frozen=True)
class RagEvaluationResult:
    """runner 顶层结果。"""

    run_id: str
    dataset_path: str | None
    judge_enabled: bool
    top_k: int
    duration_ms: float
    aggregate: EvaluationAggregateResult
    cases: tuple[CaseEvaluationResult, ...]

    def to_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "judge_enabled": self.judge_enabled,
            "top_k": self.top_k,
            "duration_ms": self.duration_ms,
            "aggregate": self.aggregate.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class EvaluationReportPaths:
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


class EvaluationReportWriteError(RuntimeError):
    """评估报告写入失败的稳定错误。

    该错误只暴露固定 code 和安全文案；原始 `OSError` 通过 exception chaining 留给
    开发侧日志排查，避免 CLI 输出泄露本机目录结构、权限细节或底层异常全文。
    """

    code = "EVAL_REPORT_WRITE_FAILED"
    safe_message = "评估报告写入失败，请检查输出目录权限和路径。"

    def __init__(self) -> None:
        super().__init__(self.safe_message)


@dataclass(frozen=True)
class _RawCaseRun:
    case: RagCase
    adapter_result: EvaluationAdapterResult
    latency_ms: float
    error: EvaluationErrorInfo | None


class DryRunEvaluationAdapter:
    """默认 dry-run adapter。

    它不连接向量库、不调用 LLM，只返回空检索和安全拒答文本。这样默认 CLI 能在任意
    开发机/CI 上跑通数据集加载、指标汇总和结构化输出；真实 retriever 或完整 pipeline
    需要由调用方显式注入 adapter，防止离线评估误触发线上依赖。
    """

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        answer = "dry-run 未执行真实 RAG pipeline。" if case.should_answer else case.golden_answer
        return EvaluationAdapterResult(
            retrieved_ids=(),
            answer=answer,
            context="",
        )


class RagRetrieverEvaluationAdapter:
    """基于 `RagRetriever` 的显式检索 adapter。

    该 adapter 只有在调用方明确选择 `--adapter retriever` 或测试注入时才会使用；默认
    dry-run 不会创建它，避免 evaluation runner 在无 Milvus/DashScope 的环境中失败。
    """

    def __init__(self, retriever: object | None = None) -> None:
        if retriever is None:
            # 线上 API 在 lifespan 里连接 Milvus；离线 eval 进程没有 lifespan，
            # 必须在这里显式 connect，否则 get_collection() 会因未初始化全部失败。
            from app.core.milvus_client import milvus_manager

            milvus_manager.connect()

            from app.rag.retriever import RagRetriever

            retriever = RagRetriever()
        self.retriever = cast(RetrieverLike, retriever)

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        retrieval = self.retriever.retrieve(case.question)
        chunks = tuple(getattr(retrieval, "chunks", ()))
        retrieved_ids = tuple(_chunk_retrieved_id(chunk) for chunk in chunks)
        context = "\n\n".join(_chunk_content(chunk) for chunk in chunks if _chunk_content(chunk))
        return EvaluationAdapterResult(
            retrieved_ids=retrieved_ids,
            answer="",
            context=context,
        )


class RagAgentServiceLike(Protocol):
    """pipeline adapter 依赖的 RagAgentService 最小协议，方便测试注入 fake。"""

    async def query_with_citations(
        self,
        question: str,
        session_id: str = "default",
        conversation_context: object | None = None,
        ctx: object | None = None,
    ) -> object:
        """执行完整 Stage 3B RAG pipeline 并返回带 citations 的结果。"""


class RagPipelineEvaluationAdapter:
    """基于线上 `RagAgentService.query_with_citations` 的完整链路 adapter。

    覆盖 Stage 3B 全链路：LLM 查询改写 → 多路召回 → 阈值过滤 → rerank →
    token 预算分配 → context 构建 → 答案生成（引用指令）→ citation 构建 →
    锚点清洗。`retrieved_ids` 取最终 citations 的 chunk_id，因此 Hit@K/Recall
    衡量的是"期望文档进入最终答案证据链"的端到端召回，而非裸检索召回。

    只有调用方明确选择 `--adapter pipeline` 或测试注入时才会创建；它要求
    Milvus、DashScope LLM 全部可用。拒答路径（`RagEmptyResultError`）映射为
    线上 RAG_EMPTY_RESULT 的用户文案，使 judge 的 no_answer 指标可评估。
    """

    # 与 app/core/errors.py 的 RAG_EMPTY_RESULT 用户文案保持一致：评测记录的
    # 拒答文本就是 API 层真正返回给用户的文案，避免评测与线上语义分叉。
    REFUSAL_ANSWER = "未找到足够相关的知识库内容。"

    def __init__(
        self,
        service: RagAgentServiceLike | None = None,
        *,
        session_prefix: str = "rag_eval",
    ) -> None:
        if service is None:
            # 与 retriever adapter 相同的显式连接：离线 eval 进程没有 lifespan。
            from app.core.milvus_client import milvus_manager

            milvus_manager.connect()

            from app.services.rag_agent_service import RagAgentService

            service = cast(RagAgentServiceLike, RagAgentService(streaming=False))
        self.service = service
        self.session_prefix = session_prefix
        # query_with_citations 是 async，而 runner adapter 协议是 sync。这里用
        # 常驻事件循环桥接：若每 case 各 asyncio.run 一次，ChatQwen 底层 httpx
        # 连接池会绑定到已关闭的 loop，第二条 case 起报 "Event loop is closed"。
        self._loop = asyncio.new_event_loop()
        atexit.register(self.close)

    def close(self) -> None:
        """关闭常驻事件循环；由 atexit 兜底，CLI 正常退出时不泄漏。"""

        if not self._loop.is_closed():
            self._loop.close()

    def evaluate_case(self, case: RagCase) -> EvaluationAdapterResult:
        session_id = f"{self.session_prefix}_{case.id}_{uuid4().hex[:8]}"
        # 与 agent_runner 的真实 adapter 相同：离线 CLI 没有 FastAPI middleware，
        # 按 case 设置独立 RequestContext，让 token budget 分配和 trace 事件
        # 按 request 归档，case 之间互不串扰。延迟导入，dry-run 路径不引入 app 依赖。
        from app.core.errors import RagEmptyResultError
        from app.core.request_context import reset_request_context, set_request_context

        request_ctx = _build_pipeline_request_context(session_id)
        token = set_request_context(request_ctx)
        try:
            result = self._loop.run_until_complete(
                self.service.query_with_citations(
                    case.question,
                    session_id,
                    ctx=request_ctx,
                )
            )
        except RagEmptyResultError:
            # 拒答是正确行为而非失败：should_answer=false 的 case 依赖该路径，
            # judge 的 no_answer 指标据此打分；轻量检索指标自然记 0 召回。
            return EvaluationAdapterResult(
                retrieved_ids=(),
                answer=self.REFUSAL_ANSWER,
                context="",
                citations=(),
            )
        finally:
            reset_request_context(token)

        citations = _result_citations(result)
        return EvaluationAdapterResult(
            retrieved_ids=tuple(
                citation_id
                for citation_id in (_citation_retrieved_id(c) for c in citations)
                if citation_id
            ),
            answer=_result_answer(result),
            context=_result_context_text(result),
            citations=citations,
        )


@dataclass(frozen=True)
class _UnavailableJudge:
    """真实 judge 初始化失败时的占位 judge。

    这样 `--judge enabled` 缺依赖或配置错误也不会阻断轻量检索指标；每条 case 会得到
    稳定 `EVAL_JUDGE_FAILED`，符合“judge 失败不阻断”的执行计划。
    """

    error: JudgeError

    def judge(self, payload: JudgeInput) -> JudgeResult:
        _ = payload
        raise self.error


class RagEvaluationRunner:
    """执行 RAG eval set 的离线 runner。"""

    def __init__(
        self,
        *,
        adapter: EvaluationAdapterLike | None = None,
        judge: JudgeLike | None = None,
        top_k: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.adapter = adapter or DryRunEvaluationAdapter()
        self.judge = judge
        self.top_k = top_k or _config_value("eval_retrieval_k", _DEFAULT_RETRIEVAL_K)
        self.run_id = run_id or _new_run_id()

    def run(
        self,
        cases: Sequence[RagCase],
        *,
        dataset_path: str | Path | None = None,
    ) -> RagEvaluationResult:
        """运行完整评估。

        单条 case 或 judge 失败都会被记录并继续后续 case。这样轻量 Hit@K/Recall@K/MRR
        始终可输出，符合“judge 失败不阻断轻量指标”的执行计划要求。
        """

        started = time.perf_counter()
        raw_runs = [self._run_case(case) for case in cases]
        metrics_summary = summarize_retrieval_metrics(
            [raw.case.expected_doc_ids for raw in raw_runs],
            [raw.adapter_result.normalized_retrieved_ids() for raw in raw_runs],
            k=self.top_k,
            case_ids=[raw.case.id for raw in raw_runs],
        )
        case_results = tuple(
            self._build_case_result(raw_run, metric)
            for raw_run, metric in zip(raw_runs, metrics_summary.cases, strict=True)
        )
        duration_ms = _elapsed_ms(started)
        aggregate = _build_aggregate(metrics_summary, case_results)
        dataset_path_text = str(dataset_path) if dataset_path is not None else None
        LOGGER.info(
            "RAG eval run finished",
            extra={
                "run_id": self.run_id,
                "dataset_path": dataset_path_text,
                "case_count": len(case_results),
                "duration_ms": duration_ms,
            },
        )
        return RagEvaluationResult(
            run_id=self.run_id,
            dataset_path=dataset_path_text,
            judge_enabled=self.judge is not None,
            top_k=self.top_k,
            duration_ms=duration_ms,
            aggregate=aggregate,
            cases=case_results,
        )

    def _run_case(self, case: RagCase) -> _RawCaseRun:
        started = time.perf_counter()
        try:
            adapter_result = self.adapter.evaluate_case(case)
            error = None
        except Exception as exc:
            # 这里只记录异常类型给日志，结构化结果使用固定安全文案，避免把密钥、内部 URL
            # 或底层 traceback 带进 eval 输出。失败 case 的 retrieved_ids 置空，轻量指标
            # 会自然反映该 case 未命中，同时 runner 继续执行后续用例。
            LOGGER.warning(
                "RAG eval case failed",
                extra={
                    "run_id": self.run_id,
                    "case_id": case.id,
                    "error_class": exc.__class__.__name__,
                },
            )
            adapter_result = EvaluationAdapterResult()
            error = EvaluationErrorInfo(
                code="EVAL_CASE_FAILED",
                message=_DEFAULT_SAFE_CASE_ERROR,
            )
        return _RawCaseRun(
            case=case,
            adapter_result=adapter_result,
            latency_ms=_elapsed_ms(started),
            error=error,
        )

    def _build_case_result(
        self,
        raw_run: _RawCaseRun,
        metric: RetrievalCaseMetric,
    ) -> CaseEvaluationResult:
        judge_result: JudgeResult | None = None
        judge_error: EvaluationErrorInfo | None = None
        if self.judge is not None and raw_run.error is None:
            judge_result, judge_error = self._judge_case(raw_run)

        adapter_result = raw_run.adapter_result
        return CaseEvaluationResult(
            case_id=raw_run.case.id,
            question=raw_run.case.question,
            should_answer=raw_run.case.should_answer,
            expected_doc_ids=tuple(raw_run.case.expected_doc_ids),
            retrieved_ids=adapter_result.normalized_retrieved_ids(),
            answer=adapter_result.answer,
            context=adapter_result.context,
            citations=tuple(_json_safe_mapping(item) for item in adapter_result.citations),
            retrieval_metric=metric,
            latency_ms=raw_run.latency_ms,
            error=raw_run.error,
            judge=judge_result,
            judge_error=judge_error,
        )

    def _judge_case(
        self,
        raw_run: _RawCaseRun,
    ) -> tuple[JudgeResult | None, EvaluationErrorInfo | None]:
        if self.judge is None:
            return None, None
        payload = JudgeInput(
            case_id=raw_run.case.id,
            question=raw_run.case.question,
            golden_answer=raw_run.case.golden_answer,
            answer=raw_run.adapter_result.answer,
            context=raw_run.adapter_result.context,
            should_answer=raw_run.case.should_answer,
            expected_doc_ids=tuple(raw_run.case.expected_doc_ids),
            retrieved_ids=raw_run.adapter_result.normalized_retrieved_ids(),
            citations=tuple(
                _json_safe_mapping(item) for item in raw_run.adapter_result.citations
            ),
        )
        try:
            return self.judge.judge(payload), None
        except JudgeError as exc:
            safe_message = exc.safe_message
            error_code = "EVAL_JUDGE_FAILED"
        except Exception as exc:
            # fake 或第三方 judge 可能抛普通异常；输出仍保持同一个安全错误码和文案，
            # 只把异常类型写入日志，防止原始异常污染评估结果。
            LOGGER.warning(
                "RAG eval judge failed",
                extra={
                    "run_id": self.run_id,
                    "case_id": raw_run.case.id,
                    "error_class": exc.__class__.__name__,
                },
            )
            safe_message = _DEFAULT_SAFE_JUDGE_ERROR
            error_code = "EVAL_JUDGE_FAILED"
        return None, EvaluationErrorInfo(code=error_code, message=safe_message)


def load_and_run(
    dataset_path: str | Path,
    *,
    adapter: EvaluationAdapterLike | None = None,
    judge: JudgeLike | None = None,
    top_k: int | None = None,
) -> RagEvaluationResult:
    """加载数据集并运行 runner，供 CLI 和后续报告模块复用。"""

    cases = load_rag_cases(dataset_path)
    runner = RagEvaluationRunner(adapter=adapter, judge=judge, top_k=top_k)
    return runner.run(cases, dataset_path=dataset_path)


def write_evaluation_report(
    result: RagEvaluationResult,
    *,
    output_dir: str | Path,
    config: Mapping[str, object] | None = None,
    thresholds: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> EvaluationReportPaths:
    """把一次完整评估结果写成 JSON、Markdown 和趋势 JSONL。

    报告 writer 是离线 runner 的内部能力，不改变任何 HTTP API。它显式接收
    `RagEvaluationResult`，这样测试可以使用 fake adapter/judge 覆盖报告 schema，
    不需要真实 Milvus、DashScope、MCP 或网络。写入失败时抛稳定错误码，由 CLI 映射为
    非 0 退出，避免“评估成功但报告没写出来”被静默吞掉。
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
    trend_path = output_path / _TREND_FILE_NAME

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
        # 只把异常类型和 run_id 写入日志，CLI/stdout 使用稳定安全文案；这样权限错误、
        # 文件系统路径细节或底层系统消息不会成为用户可见输出。
        LOGGER.error(
            "RAG eval report write failed",
            extra={
                "run_id": result.run_id,
                "output_dir": str(output_path),
                "error_class": exc.__class__.__name__,
            },
        )
        raise EvaluationReportWriteError() from exc

    LOGGER.info(
        "RAG eval report written",
        extra={
            "run_id": result.run_id,
            "json_report_path": str(json_path),
            "markdown_report_path": str(markdown_path),
            "trend_path": str(trend_path),
        },
    )
    return EvaluationReportPaths(
        json_path=json_path,
        markdown_path=markdown_path,
        trend_path=trend_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    默认参数遵循离线验收命令：judge disabled、adapter dry-run，并把报告写入
    `eval_reports/`。stdout 仍保留原结构化结果，只追加 `reports` 字段，兼容已有调用方
    对 `judge_enabled`、`aggregate` 等字段的读取。
    """

    parser = argparse.ArgumentParser(description="Run AegisOps Agent offline RAG evaluation")
    parser.add_argument(
        "--dataset",
        default=_config_value("eval_dataset_path", _DEFAULT_DATASET_PATH),
        help="RAG eval YAML path",
    )
    parser.add_argument(
        "--judge",
        choices=("disabled", "enabled"),
        default=(
            "enabled"
            if _config_value("eval_judge_enabled", _DEFAULT_JUDGE_ENABLED)
            else "disabled"
        ),
        help="Whether to call LLMJudge; default is disabled",
    )
    parser.add_argument(
        "--adapter",
        choices=("dry-run", "retriever", "pipeline"),
        default="dry-run",
        help=(
            "Evaluation adapter. dry-run never touches external services; "
            "retriever covers retrieval only; pipeline covers the full "
            "Stage 3B chain (rewrite/retrieve/rerank/context/answer/citations)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_config_value("eval_retrieval_k", _DEFAULT_RETRIEVAL_K),
        help="Retrieval metric K",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_REPORT_OUTPUT_DIR,
        help="Directory for JSON/Markdown reports and trends.jsonl",
    )
    args = parser.parse_args(argv)

    adapter = _build_adapter(str(args.adapter))
    judge = _build_judge(str(args.judge))
    try:
        result = load_and_run(
            args.dataset,
            adapter=adapter,
            judge=judge,
            top_k=args.top_k,
        )
    except RagDatasetError as exc:
        payload = {
            "success": False,
            "error": {
                "code": "EVAL_DATASET_INVALID",
                "message": _safe_dataset_error_message(exc),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    try:
        report_paths = write_evaluation_report(
            result,
            output_dir=args.output,
            config={
                "dataset": args.dataset,
                "judge": args.judge,
                "adapter": args.adapter,
                "top_k": args.top_k,
            },
        )
    except EvaluationReportWriteError as exc:
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


def _build_adapter(adapter_name: str) -> EvaluationAdapterLike:
    if adapter_name == "retriever":
        return RagRetrieverEvaluationAdapter()
    if adapter_name == "pipeline":
        return RagPipelineEvaluationAdapter()
    return DryRunEvaluationAdapter()


def _build_judge(judge_mode: str) -> JudgeLike | None:
    if judge_mode != "enabled":
        return None
    try:
        return LLMJudge()
    except JudgeError as exc:
        return _UnavailableJudge(exc)


def _build_aggregate(
    metrics_summary: RetrievalMetricsSummary,
    case_results: Sequence[CaseEvaluationResult],
) -> EvaluationAggregateResult:
    scored_cases = [case.judge for case in case_results if case.judge is not None]
    return EvaluationAggregateResult(
        k=metrics_summary.k,
        case_count=metrics_summary.case_count,
        comparable_case_count=metrics_summary.comparable_case_count,
        hit_rate_at_k=metrics_summary.hit_rate_at_k,
        recall_at_k=metrics_summary.recall_at_k,
        mrr=metrics_summary.mrr,
        failed_case_count=sum(1 for case in case_results if case.error is not None),
        judge_case_count=len(scored_cases),
        judge_failed_case_count=sum(1 for case in case_results if case.judge_error is not None),
        judge_average_faithfulness=_average_score(
            [case.faithfulness for case in scored_cases]
        ),
        judge_average_answer_correctness=_average_score(
            [case.answer_correctness for case in scored_cases]
        ),
        judge_average_no_answer=_average_score([case.no_answer for case in scored_cases]),
        # citation_correctness 只在模型返回该字段时计入；None（缺字段/未评估）
        # 直接剔除，避免把"未评估"摊薄成 0 分基线。
        judge_average_citation_correctness=_average_score(
            [
                case.citation_correctness
                for case in scored_cases
                if case.citation_correctness is not None
            ]
        ),
    )


def _build_report_payload(
    result: RagEvaluationResult,
    *,
    config: Mapping[str, object],
    thresholds: Mapping[str, object],
    generated_at: datetime,
) -> JsonObject:
    aggregate = result.aggregate.to_dict()
    case_diffs = [_case_diff(case) for case in result.cases]
    failed_cases = _failed_case_entries(result.cases)
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_type": "rag_evaluation",
        "run_id": result.run_id,
        "generated_at": generated_at.isoformat(),
        "config": _json_safe_mapping(config),
        "thresholds": _json_safe_mapping(thresholds),
        "dataset": {
            "path": result.dataset_path,
            "case_count": result.aggregate.case_count,
            "comparable_case_count": result.aggregate.comparable_case_count,
        },
        "aggregate": aggregate,
        "failed_cases": failed_cases,
        "case_diffs": case_diffs,
        "cases": [case.to_dict() for case in result.cases],
    }


def _render_markdown_report(report_payload: JsonObject) -> str:
    """渲染面向开发回归的 Markdown 报告。

    Markdown 保留配置、数据集、指标、失败原因和 per-case diff，避免直接展开完整
    context/answer 大文本。完整结构留在 JSON 中，Markdown 作为发布前快速巡检视图。
    """

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
        "# AegisOps Agent RAG Evaluation Report",
        "",
        f"- Run ID: `{_markdown_inline(report_payload.get('run_id'))}`",
        f"- Generated At: `{_markdown_inline(report_payload.get('generated_at'))}`",
        f"- Schema: `{_markdown_inline(report_payload.get('schema_version'))}`",
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
        "hit_rate_at_k",
        "recall_at_k",
        "mrr",
        "failed_case_count",
        "judge_case_count",
        "judge_failed_case_count",
        "judge_average_faithfulness",
        "judge_average_answer_correctness",
        "judge_average_no_answer",
        "judge_average_citation_correctness",
    ):
        lines.append(f"| `{key}` | `{_markdown_inline(aggregate.get(key))}` |")
    lines.append("")
    return lines


def _markdown_failed_cases_section(failed_cases: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    lines = ["## 失败 Case", ""]
    if failed_cases:
        lines.extend(
            [
                "| Case ID | Type | Code | Message |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in failed_cases:
            lines.append(
                "| "
                f"{_markdown_cell(item.get('case_id'))} | "
                f"{_markdown_cell(item.get('failure_type'))} | "
                f"{_markdown_cell(item.get('code'))} | "
                f"{_markdown_cell(item.get('message'))} |"
            )
    else:
        lines.append("- 无失败 case。")
    lines.append("")
    return lines


def _markdown_case_diff_section(case_diffs: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    lines = [
        "## Per-case Diff",
        "",
        "| Case ID | Expected | Retrieved | Missing | Unexpected | Hit@K | Recall@K | Error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in case_diffs:
        metric = _mapping_value(item, "retrieval_metric")
        lines.append(
            "| "
            f"{_markdown_cell(item.get('case_id'))} | "
            f"{_markdown_cell(_join_json_list(item.get('expected_doc_ids')))} | "
            f"{_markdown_cell(_join_json_list(item.get('retrieved_ids')))} | "
            f"{_markdown_cell(_join_json_list(item.get('missing_expected_doc_ids')))} | "
            f"{_markdown_cell(_join_json_list(item.get('unexpected_retrieved_ids')))} | "
            f"{_markdown_cell(metric.get('hit_at_k'))} | "
            f"{_markdown_cell(metric.get('recall_at_k'))} | "
            f"{_markdown_cell(item.get('failure_code'))} |"
        )
    return lines


def _append_trend(trend_path: Path, report_payload: JsonObject) -> None:
    aggregate = _mapping_value(report_payload, "aggregate")
    dataset = _mapping_value(report_payload, "dataset")
    trend_payload: JsonObject = {
        "timestamp": report_payload.get("generated_at"),
        "run_id": report_payload.get("run_id"),
        "dataset_path": dataset.get("path"),
        "case_count": aggregate.get("case_count"),
        "comparable_case_count": aggregate.get("comparable_case_count"),
        "failed_case_count": aggregate.get("failed_case_count"),
        "judge_failed_case_count": aggregate.get("judge_failed_case_count"),
        "hit_rate_at_k": aggregate.get("hit_rate_at_k"),
        "recall_at_k": aggregate.get("recall_at_k"),
        "mrr": aggregate.get("mrr"),
        "judge_average_faithfulness": aggregate.get("judge_average_faithfulness"),
        "judge_average_answer_correctness": aggregate.get(
            "judge_average_answer_correctness"
        ),
        "judge_average_no_answer": aggregate.get("judge_average_no_answer"),
        "judge_average_citation_correctness": aggregate.get(
            "judge_average_citation_correctness"
        ),
    }
    with trend_path.open("a", encoding="utf-8") as trend_file:
        trend_file.write(json.dumps(trend_payload, ensure_ascii=False) + "\n")


def _case_diff(case: CaseEvaluationResult) -> JsonObject:
    expected_doc_ids = list(case.expected_doc_ids)
    retrieved_ids = list(case.retrieved_ids)
    missing_expected_doc_ids = [
        expected_id
        for expected_id in expected_doc_ids
        if not _contains_expected_id(expected_id, retrieved_ids)
    ]
    unexpected_retrieved_ids = [
        retrieved_id
        for retrieved_id in retrieved_ids
        if not _matches_any_expected_id(retrieved_id, expected_doc_ids)
    ]
    failure_code = None
    if case.error is not None:
        failure_code = case.error.code
    elif case.judge_error is not None:
        failure_code = case.judge_error.code

    return {
        "case_id": case.case_id,
        "question": case.question,
        "should_answer": case.should_answer,
        "expected_doc_ids": expected_doc_ids,
        "retrieved_ids": retrieved_ids,
        "missing_expected_doc_ids": missing_expected_doc_ids,
        "unexpected_retrieved_ids": unexpected_retrieved_ids,
        "retrieval_metric": _case_metric_to_dict(case.retrieval_metric),
        "failure_code": failure_code,
        "error": case.error.to_dict() if case.error is not None else None,
        "judge_error": case.judge_error.to_dict() if case.judge_error is not None else None,
    }


def _failed_case_entries(cases: Sequence[CaseEvaluationResult]) -> list[JsonObject]:
    entries: list[JsonObject] = []
    for case in cases:
        if case.error is not None:
            entries.append(
                {
                    "case_id": case.case_id,
                    "failure_type": "case",
                    "code": case.error.code,
                    "message": case.error.message,
                }
            )
        if case.judge_error is not None:
            entries.append(
                {
                    "case_id": case.case_id,
                    "failure_type": "judge",
                    "code": case.judge_error.code,
                    "message": case.judge_error.message,
                }
            )
    return entries


def _report_file_stem(generated_at: datetime) -> str:
    return generated_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _contains_expected_id(expected_id: str, retrieved_ids: Sequence[str]) -> bool:
    return any(_ids_match(expected_id, retrieved_id) for retrieved_id in retrieved_ids)


def _matches_any_expected_id(retrieved_id: str, expected_ids: Sequence[str]) -> bool:
    return any(_ids_match(expected_id, retrieved_id) for expected_id in expected_ids)


def _ids_match(expected_id: str, retrieved_id: str) -> bool:
    if expected_id == retrieved_id:
        return True
    if "#" in expected_id:
        return False
    return retrieved_id.startswith(f"{expected_id}#")


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


def _average_score(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _case_metric_to_dict(metric: RetrievalCaseMetric) -> JsonObject:
    return {
        "case_index": metric.case_index,
        "case_id": metric.case_id,
        "comparable": metric.comparable,
        "expected_ids": list(metric.expected_ids),
        "retrieved_ids": list(metric.retrieved_ids),
        "hit_at_k": metric.hit_at_k,
        "recall_at_k": metric.recall_at_k,
        "reciprocal_rank": metric.reciprocal_rank,
        "first_relevant_rank": metric.first_relevant_rank,
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


def _normalize_text_items(items: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _chunk_retrieved_id(chunk: object) -> str:
    chunk_id = getattr(chunk, "chunk_id", None)
    if isinstance(chunk_id, str) and chunk_id.strip():
        return chunk_id.strip()
    doc_id = getattr(chunk, "doc_id", None)
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()
    return ""


def _chunk_content(chunk: object) -> str:
    content = getattr(chunk, "content", None)
    return content if isinstance(content, str) else ""


def _build_pipeline_request_context(session_id: str) -> object:
    """按 case 构造独立 RequestContext；ID 格式与线上 middleware 一致。

    deadline 放宽到 120s：完整 pipeline 含查询改写、rerank 和答案生成多次
    LLM 调用，线上单请求 timeout（默认 60s）对 hard 用例偏紧。
    """

    from app.core.request_context import RequestContext

    now = time.time()
    return RequestContext(
        trace_id=f"trc_{uuid4().hex}",
        request_id=f"req_{uuid4().hex}",
        session_id=session_id,
        tenant_id="default",
        user_id="rag-eval",
        deadline_ms=120_000,
        feature_flags=(),
        started_at=now,
        started_monotonic=time.monotonic(),
        method="CLI",
        path="evaluation/runner",
        invalid_inbound_trace_header=False,
    )


def _result_answer(result: object) -> str:
    answer = getattr(result, "answer", None)
    return answer if isinstance(answer, str) else ""


def _result_context_text(result: object) -> str:
    context_text = getattr(result, "context_text", None)
    return context_text if isinstance(context_text, str) else ""


def _result_citations(result: object) -> tuple[Mapping[str, object], ...]:
    raw_citations = getattr(result, "citations", None)
    if isinstance(raw_citations, Sequence) and not isinstance(raw_citations, (str, bytes, bytearray)):
        return tuple(
            _json_safe_mapping(citation)
            for citation in raw_citations
            if isinstance(citation, Mapping)
        )
    return ()


def _citation_retrieved_id(citation: Mapping[str, object]) -> str:
    """从 API-safe citation 提取检索指标可比较的 id（chunk_id 优先）。"""

    chunk_id = citation.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id.strip():
        return chunk_id.strip()
    doc_id = citation.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()
    return ""


def _new_run_id() -> str:
    return f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _safe_dataset_error_message(error: RagDatasetError) -> str:
    # 不把异常文本原样返回给命令行输出，避免未来 loader 把本机绝对路径或解析细节
    # 混入用户可见结果。具体字段定位由 loader 单测保证；CLI 只给稳定错误码和安全文案。
    _ = error
    return "评估数据集无效，请检查 YAML 文件路径、字段和 case id。"


def _config_value(name: str, default: _ConfigValue) -> _ConfigValue:
    """读取 app.config 中的 evaluation 配置，缺少线上依赖时使用默认值。

    `python -m evaluation.runner --judge disabled` 是离线验收命令，不能因为开发机没安装
    loguru/langchain/openai 这类线上依赖而无法启动。只有调用方显式选择 retriever 或
    enabled judge 时，相关 app/LLM 依赖才会被后续路径检查。
    """

    try:
        from app.config import config as app_config
    except ModuleNotFoundError:
        return default
    return cast(_ConfigValue, getattr(app_config, name, default))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DryRunEvaluationAdapter",
    "EvaluationAdapterLike",
    "EvaluationAdapterResult",
    "EvaluationAggregateResult",
    "EvaluationErrorInfo",
    "EvaluationReportPaths",
    "EvaluationReportWriteError",
    "RagAgentServiceLike",
    "RagEvaluationResult",
    "RagEvaluationRunner",
    "RagPipelineEvaluationAdapter",
    "RagRetrieverEvaluationAdapter",
    "load_and_run",
    "main",
    "write_evaluation_report",
]
