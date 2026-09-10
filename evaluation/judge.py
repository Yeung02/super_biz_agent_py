"""LLM judge 封装。

本模块属于 ISSUE-033 的离线评估边界：它只负责把固定 rubric、固定
temperature=0 的 judge 调用封装成可测试对象，不参与线上 FastAPI 请求路径，也不把
judge 失败作为轻量检索指标的阻断条件。默认 runner 会关闭 judge；只有显式开启或在
测试中注入 fake judge 时才会调用模型。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, TypeVar, cast


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

JudgeStatus = Literal["scored"]
_FIXED_TEMPERATURE = 0.0
_DEFAULT_MAX_CONTEXT_CHARS = 8_000
_DEFAULT_JUDGE_MODEL = "qwen-max"
_ConfigValue = TypeVar("_ConfigValue")


class JudgeModelLike(Protocol):
    """LLM judge 使用的最小模型协议。

    只要求 `invoke(prompt)`，方便测试注入内存 fake，也避免 runner 为了评估而绑定
    某个具体供应商 SDK。真实模型由 `LLMFactory` 延迟创建，默认 disabled 时不会触发。
    """

    def invoke(self, prompt: str) -> object:
        """执行同步 judge 调用并返回带 `content` 的对象或字符串。"""


class JudgeModelFactoryLike(Protocol):
    """用于创建 judge 模型的最小工厂协议。"""

    def create_chat_model(
        self,
        model: str | None = None,
        temperature: float = _FIXED_TEMPERATURE,
        streaming: bool = False,
    ) -> JudgeModelLike:
        """按固定 temperature 创建同步 judge 模型。"""


@dataclass(frozen=True)
class JudgeInput:
    """单条 case 的 judge 输入。

    `context` 可以来自检索证据或完整 pipeline adapter。这里保留 expected/retrieved ids，
    是为了让 judge prompt 能知道“答案是否有依据”，但不把原始 metadata、绝对路径或
    工具 payload 透传进去，降低离线报告泄漏内部信息的风险。
    """

    case_id: str
    question: str
    golden_answer: str
    answer: str
    context: str
    should_answer: bool
    expected_doc_ids: tuple[str, ...] = ()
    retrieved_ids: tuple[str, ...] = ()
    # 完整 pipeline adapter 传入的 API-safe citations；retriever/dry-run adapter 保持
    # 空元组，judge prompt 会把空列表渲染为 none，评分语义不受影响。
    citations: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class JudgeResult:
    """LLM judge 的稳定评分结果。

    `citation_correctness` 仅在输入包含 citations 时有意义；模型返回缺失该字段时
    保持 None，聚合阶段跳过，避免把"未评估"误记为 0 分。
    """

    status: JudgeStatus
    faithfulness: float
    answer_correctness: float
    no_answer: float
    reasoning: str
    citation_correctness: float | None = None

    def to_dict(self) -> JsonObject:
        """转换为 runner 可序列化结构，不包含模型原始返回。"""

        return {
            "status": self.status,
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "no_answer": self.no_answer,
            "citation_correctness": self.citation_correctness,
            "reasoning": self.reasoning,
        }


class JudgeError(RuntimeError):
    """judge 调用或解析失败。

    `safe_message` 是 runner 可写入结构化结果的用户可见文案；底层异常只通过
    exception chaining 留给开发调试，不能进入 `to_dict()`，避免泄漏密钥、内部 URL
    或供应商原始错误全文。
    """

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class LLMJudge:
    """固定 rubric 的 LLM judge。

    评分项与执行计划保持一致：`faithfulness` 衡量回答是否由 context 支撑，
    `answer_correctness` 衡量是否贴近 golden answer，`no_answer` 衡量拒答类 case 是否
    正确拒答。temperature 强制为 0，即使调用方传入其它值也会被忽略，保证评估尽量稳定。
    """

    def __init__(
        self,
        *,
        model_client: JudgeModelLike | None = None,
        model_factory: JudgeModelFactoryLike | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        _ = temperature
        self.model_name = model_name or _config_value("eval_judge_model", _DEFAULT_JUDGE_MODEL)
        self.temperature = _FIXED_TEMPERATURE
        self.max_context_chars = max_context_chars or _config_value(
            "eval_judge_max_context_chars",
            _DEFAULT_MAX_CONTEXT_CHARS,
        )
        self.model_client = model_client or self._create_model(model_factory)

    def judge(self, payload: JudgeInput) -> JudgeResult:
        """执行 judge 并解析为稳定分数字段。

        这里捕获模型调用异常和 JSON 解析异常，但只抛稳定 `JudgeError`；runner 会记录
        `EVAL_JUDGE_FAILED` 并继续轻量指标，避免 judge 服务波动影响 CI 基线。
        """

        prompt = self._build_prompt(payload)
        try:
            response = self.model_client.invoke(prompt)
        except Exception as exc:
            raise JudgeError("JUDGE_PROVIDER_ERROR", "LLM judge 调用失败。") from exc

        try:
            response_mapping = _parse_response_mapping(response)
            return JudgeResult(
                status="scored",
                faithfulness=_score(response_mapping, "faithfulness"),
                answer_correctness=_score(response_mapping, "answer_correctness"),
                no_answer=_score(response_mapping, "no_answer"),
                reasoning=_reasoning(response_mapping),
                citation_correctness=_optional_score(response_mapping, "citation_correctness"),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JudgeError("JUDGE_INVALID_RESPONSE", "LLM judge 返回格式无效。") from exc

    def _create_model(self, model_factory: JudgeModelFactoryLike | None) -> JudgeModelLike:
        if model_factory is None:
            try:
                from app.core.llm_factory import LLMFactory
            except ModuleNotFoundError as exc:
                # evaluation 包在 disabled/fake judge 场景下不能要求真实 LLM 依赖存在；
                # 只有显式创建真实 judge 时才把缺失依赖映射为稳定错误。
                raise JudgeError(
                    "JUDGE_PROVIDER_UNAVAILABLE",
                    "LLM judge 模型依赖不可用。",
                ) from exc
            factory = LLMFactory()
        else:
            factory = model_factory
        return factory.create_chat_model(
            model=self.model_name,
            temperature=self.temperature,
            streaming=False,
        )

    def _build_prompt(self, payload: JudgeInput) -> str:
        context = _trim_text(payload.context, self.max_context_chars)
        answer = _trim_text(payload.answer, 4_000)
        golden_answer = _trim_text(payload.golden_answer, 4_000)
        expected_ids = ", ".join(payload.expected_doc_ids) or "none"
        retrieved_ids = ", ".join(payload.retrieved_ids) or "none"
        citations = _format_citations(payload.citations)
        should_answer = "true" if payload.should_answer else "false"
        # prompt 明确要求只返回 JSON，是为了让 runner 能稳定解析；rubric 名称使用英文
        # 字段，便于后续报告、趋势和跨语言工具链复用同一 schema。
        return f"""You are an offline RAG evaluation judge. Use the fixed rubric below.

Scores must be numbers from 0.0 to 1.0.
- faithfulness: whether the answer is supported by the provided context.
- answer_correctness: whether the answer matches the golden answer.
- no_answer: for should_answer=false, score 1.0 only if the answer refuses to invent unsupported facts; for should_answer=true, score 0.0 unless the answer incorrectly refuses.
- citation_correctness: whether the answer's [C*] citation markers refer to the provided citations and the cited evidence supports the marked statements. Score 1.0 when every marker matches a provided citation and the evidence supports the statement, or when there are no citations and no markers; score 0.0 when markers reference non-existent citations or contradict the evidence.

Return only JSON with fields:
{{"faithfulness": 0.0, "answer_correctness": 0.0, "no_answer": 0.0, "citation_correctness": 0.0, "reasoning": "short reason"}}

case_id: {payload.case_id}
should_answer: {should_answer}
expected_doc_ids: {expected_ids}
retrieved_ids: {retrieved_ids}
question:
{payload.question}

golden_answer:
{golden_answer}

answer:
{answer}

citations:
{citations}

context:
{context}
"""


def _parse_response_mapping(response: object) -> Mapping[str, object]:
    raw_content = _response_content(response)
    json_text = _extract_json_text(raw_content)
    parsed = json.loads(json_text)
    if not isinstance(parsed, Mapping):
        raise TypeError("judge response must be a JSON object")
    return cast(Mapping[str, object], parsed)


def _response_content(response: object) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise TypeError("judge response has no textual content")


def _extract_json_text(raw_content: str) -> str:
    stripped = raw_content.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("missing JSON object", stripped, 0)
    return stripped[start : end + 1]


def _score(response_mapping: Mapping[str, object], field_name: str) -> float:
    value = response_mapping.get(field_name)
    if not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    return min(max(float(value), 0.0), 1.0)


def _optional_score(response_mapping: Mapping[str, object], field_name: str) -> float | None:
    """解析可选分数字段：缺失返回 None，存在但非数字仍视为格式错误。

    citation_correctness 依赖模型遵循新 prompt；老版本模型输出可能缺字段，
    聚合阶段按 None 跳过，不能把缺省当 0 分拉低基线。
    """

    if field_name not in response_mapping:
        return None
    return _score(response_mapping, field_name)


def _format_citations(citations: Sequence[Mapping[str, object]]) -> str:
    """把 API-safe citations 渲染为 judge prompt 的紧凑证据列表。

    只保留 citation_id/doc_id/file_name/preview 等安全字段；来源完整 metadata
    或绝对路径不进入 prompt，与 runner 的报告脱敏约定保持一致。
    """

    if not citations:
        return "none"
    lines: list[str] = []
    for citation in citations:
        citation_id = citation.get("citation_id")
        doc_id = citation.get("doc_id")
        file_name = citation.get("file_name")
        preview = citation.get("content_preview")
        line = f"{citation_id} (doc_id={doc_id}, file_name={file_name})"
        if isinstance(preview, str) and preview.strip():
            line += f" preview: {_trim_text(preview, 400)}"
        lines.append(line)
    return "\n".join(lines)


def _reasoning(response_mapping: Mapping[str, object]) -> str:
    raw_reasoning = response_mapping.get("reasoning")
    if not isinstance(raw_reasoning, str):
        return ""
    return _trim_text(raw_reasoning, 500)


def _trim_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return value if len(value) <= max_chars else value[:max_chars]


def _config_value(name: str, default: _ConfigValue) -> _ConfigValue:
    """读取 app.config 中的 evaluation 配置，缺依赖时使用本地默认值。

    evaluation 包必须能在没有线上运行时依赖的环境中导入和 dry-run。这里延迟导入
    `app.config`，并在缺少 loguru/langchain 等 app 依赖时回退默认值；真实 judge 模型
    只有显式开启时才会继续检查供应商依赖。
    """

    try:
        from app.config import config as app_config
    except ModuleNotFoundError:
        return default
    return cast(_ConfigValue, getattr(app_config, name, default))


__all__ = [
    "JudgeError",
    "JudgeInput",
    "JudgeModelLike",
    "JudgeResult",
    "LLMJudge",
]
