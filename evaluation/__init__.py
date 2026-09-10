"""AegisOps Agent 离线评估工具包。

该包属于离线/CI 评估边界，不被线上 FastAPI 默认导入。ISSUE-033 暴露 runner 和
LLM judge 的结构化入口；ISSUE-034 增加 JSON/Markdown 报告。Agent 轨迹评测
（agent_* 模块）与 RAG 评测共用同一工程约定：默认 dry-run、失败不阻断、报告
落盘带稳定错误码。
"""

from __future__ import annotations

from importlib import import_module

from evaluation.agent_datasets import AgentCase, AgentDatasetError, load_agent_cases
from evaluation.agent_metrics import (
    AgentCaseMetric,
    AgentMetricsSummary,
    AgentTrajectory,
    AgentTrajectoryExpectation,
    summarize_agent_metrics,
)
from evaluation.datasets import RagCase, RagDatasetError, load_rag_cases
from evaluation.rag_metrics import (
    RetrievalCaseMetric,
    RetrievalMetricsSummary,
    hit_rate_at_k,
    mean_reciprocal_rank,
    mrr,
    recall_at_k,
    summarize_retrieval_metrics,
)

_LAZY_EXPORT_MODULES = {
    "AgentCaseEvaluationResult": "evaluation.agent_runner",
    "AgentCaseExecutionError": "evaluation.agent_runner",
    "AgentEvaluationAdapterLike": "evaluation.agent_runner",
    "AgentEvaluationAggregateResult": "evaluation.agent_runner",
    "AgentEvaluationErrorInfo": "evaluation.agent_runner",
    "AgentEvaluationReportPaths": "evaluation.agent_runner",
    "AgentEvaluationReportWriteError": "evaluation.agent_runner",
    "AgentEvaluationResult": "evaluation.agent_runner",
    "AgentEvaluationRunner": "evaluation.agent_runner",
    "AIOpsServiceAgentEvaluationAdapter": "evaluation.agent_runner",
    "DryRunAgentEvaluationAdapter": "evaluation.agent_runner",
    "load_and_run_agent": "evaluation.agent_runner",
    "write_agent_evaluation_report": "evaluation.agent_runner",
    "DryRunEvaluationAdapter": "evaluation.runner",
    "EvaluationAdapterResult": "evaluation.runner",
    "EvaluationAggregateResult": "evaluation.runner",
    "EvaluationErrorInfo": "evaluation.runner",
    "JudgeError": "evaluation.judge",
    "JudgeInput": "evaluation.judge",
    "JudgeResult": "evaluation.judge",
    "LLMJudge": "evaluation.judge",
    "RagEvaluationResult": "evaluation.runner",
    "RagEvaluationRunner": "evaluation.runner",
    "RagRetrieverEvaluationAdapter": "evaluation.runner",
    "load_and_run": "evaluation.runner",
}


def __getattr__(name: str) -> object:
    """懒加载 runner/judge 导出，避免 `python -m evaluation.runner` 出现 runpy 警告。

    package root 仍保留便捷导出，但不会在导入 `evaluation` 时提前执行 runner 模块；
    这让离线命令入口和单元测试都保持清爽，也避免默认导入真实 judge 相关依赖。
    """

    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'evaluation' has no attribute '{name}'")
    module = import_module(module_name)
    return getattr(module, name)

__all__ = [
    "AgentCase",
    "AgentCaseEvaluationResult",
    "AgentCaseExecutionError",
    "AgentCaseMetric",
    "AgentDatasetError",
    "AgentEvaluationAdapterLike",
    "AgentEvaluationAggregateResult",
    "AgentEvaluationErrorInfo",
    "AgentEvaluationReportPaths",
    "AgentEvaluationReportWriteError",
    "AgentEvaluationResult",
    "AgentEvaluationRunner",
    "AgentMetricsSummary",
    "AgentTrajectory",
    "AgentTrajectoryExpectation",
    "AIOpsServiceAgentEvaluationAdapter",
    "DryRunAgentEvaluationAdapter",
    "DryRunEvaluationAdapter",
    "EvaluationAdapterResult",
    "EvaluationAggregateResult",
    "EvaluationErrorInfo",
    "JudgeError",
    "JudgeInput",
    "JudgeResult",
    "LLMJudge",
    "RagCase",
    "RagDatasetError",
    "RagEvaluationResult",
    "RagEvaluationRunner",
    "RagRetrieverEvaluationAdapter",
    "RetrievalCaseMetric",
    "RetrievalMetricsSummary",
    "hit_rate_at_k",
    "load_and_run",
    "load_and_run_agent",
    "load_agent_cases",
    "load_rag_cases",
    "mean_reciprocal_rank",
    "mrr",
    "recall_at_k",
    "summarize_agent_metrics",
    "summarize_retrieval_metrics",
    "write_agent_evaluation_report",
]

