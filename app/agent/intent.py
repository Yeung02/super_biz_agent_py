"""Chat 入口意图识别与闲聊直答。

在 orchestrator 边界对用户输入做一次轻量 LLM 分类，把对话请求路由到闲聊直答、
RAG 问答或 AIOps 诊断。分类失败、超时或输出非法时 fail-open 回 rag_qa：意图识别
是入口的加速与分流能力，不能成为对话链路的阻塞点，这与 query rewriter/reranker
的回退策略保持一致。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import config
from app.core.request_context import RequestContext

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# SDK 层自动重试次数：只重试瞬时错误（连接失败、429、5xx）。解析类失败
# （非 JSON / 未知标签）在 temperature=0 下确定性复现，重试无意义，由
# fail-open 直接兜底，不走重试。
_LLM_MAX_RETRIES = 1


class IntentLabel(str, Enum):
    """Chat 入口的三个路由目标。"""

    CHITCHAT = "chitchat"
    RAG_QA = "rag_qa"
    AIOPS = "aiops"


class IntentResult(BaseModel):
    """一次意图分类的输出。

    `label` 默认 rag_qa：任何失败路径只要带着 error_code 返回，编排层就自然落回
    现有 RAG 管线，不需要额外的兜底分支。
    """

    model_config = ConfigDict(extra="forbid")

    label: IntentLabel = Field(default=IntentLabel.RAG_QA, description="识别出的意图")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="分类置信度")
    source: str = Field(
        default="llm",
        description="标签来源：llm/rule/noop，用于观测降级链路的实际命中层",
    )
    error_code: str | None = Field(default=None, description="失败错误码，None 表示成功")

    @property
    def successful(self) -> bool:
        """返回本次分类是否产出了可用标签。"""

        return self.error_code is None


class PromptLLM(Protocol):
    """意图模块需要的最小同步 LLM 接口。

    真实 LLM（ChatOpenAI）和测试 fake 都只需要实现 `invoke`；与 query rewriter
    的 RewriterLLM 协议保持同一形态，方便复用测试 fake。
    """

    def invoke(self, prompt: str) -> object:
        """同步生成文本。"""


class BaseIntentClassifier(ABC):
    """意图分类器抽象基类。

    与 BaseQueryRewriter 一样只稳定方法签名：分类器只输出标签，不携带路由、检索
    或诊断逻辑，防止边界模块变厚。
    """

    @abstractmethod
    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        """把用户输入分类为 chitchat/rag_qa/aiops 之一。"""


class NoopIntentClassifier(BaseIntentClassifier):
    """空实现：恒返 rag_qa，保持未启用意图识别时的旧对话行为。"""

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = question, ctx
        return IntentResult(source="noop")


class LlmIntentClassifier(BaseIntentClassifier):
    """通过一次轻量 LLM 调用输出意图标签。

    LLM 延迟构造且可注入：`llm=None` 时首次调用才经 LLMFactory 创建客户端，构造
    本身不触网。任何异常都收敛为 error_code，不向调用方抛出，与 LlmQueryRewriter
    的失败语义一致。
    """

    def __init__(
        self,
        *,
        llm: PromptLLM | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.llm = llm
        self.model = model or config.intent_model
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.intent_timeout_seconds
        )

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = ctx
        normalized = question.strip()
        if not normalized:
            return IntentResult(error_code="EMPTY_QUERY")
        try:
            prompt = self._build_prompt(normalized)
            response = self._get_llm().invoke(prompt)
            content = llm_content_text(response)
        except TimeoutError:
            return IntentResult(error_code="LLM_TIMEOUT")
        except Exception:
            return IntentResult(error_code="LLM_PROVIDER_ERROR")
        return _parse_intent_json(content)

    def _get_llm(self) -> PromptLLM:
        if self.llm is None:
            from app.core.llm_factory import LLMFactory

            self.llm = LLMFactory.create_chat_model(
                model=self.model,
                temperature=0.0,
                streaming=False,
                timeout=self._per_attempt_timeout(),
                max_retries=_LLM_MAX_RETRIES,
            )
        return self.llm

    def _per_attempt_timeout(self) -> float:
        """按尝试次数均分总预算并留 20% 余量，作为客户端级单次超时。

        编排层 wait_for(intent_timeout_seconds) 是整个分类步骤的硬顶；若客户端
        单次超时不缩小，首次尝试就会耗尽预算，SDK 重试永远跑不到。3s 预算 +
        1 次重试 → 单次 1.2s，最坏 2.4s < 3s，瞬时错误有一次自愈机会。
        """

        return self.timeout_seconds * 0.8 / (_LLM_MAX_RETRIES + 1)

    def _build_prompt(self, question: str) -> str:
        return "\n".join(
            [
                "你是企业运维助手入口的意图分类器。把用户输入分类为以下三类之一：",
                "- chitchat：问候、闲聊、询问助手身份或能力，不涉及具体业务诉求。",
                "- rag_qa：知识问答、文档/业务/概念咨询，需要查询知识库后回答。",
                "- aiops：故障诊断诉求，例如排查告警、定位服务异常、分析根因。",
                '只输出一行 JSON，不要解释：{"label": "chitchat|rag_qa|aiops", "confidence": 0到1的小数}',
                "示例：",
                '输入：你好，你是谁？ 输出：{"label": "chitchat", "confidence": 0.95}',
                '输入：发布流程里的灰度策略是什么？ 输出：{"label": "rag_qa", "confidence": 0.9}',
                '输入：帮我看看现在的告警，订单服务为什么超时？ 输出：{"label": "aiops", "confidence": 0.9}',
                f"输入：{question}",
                "输出：",
            ]
        )


def _parse_intent_json(content: str) -> IntentResult:
    """从 LLM 输出中提取 {label, confidence}；非法输出 fail-open 回 rag_qa。"""

    match = _JSON_OBJECT_RE.search(content)
    if match is None:
        return IntentResult(error_code="UNPARSEABLE_RESPONSE")
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return IntentResult(error_code="UNPARSEABLE_RESPONSE")
    if not isinstance(payload, dict):
        return IntentResult(error_code="UNPARSEABLE_RESPONSE")

    try:
        label = IntentLabel(str(payload.get("label")).strip().lower())
    except ValueError:
        return IntentResult(error_code="INVALID_LABEL")

    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        confidence = 0.0
    return IntentResult(label=label, confidence=min(max(float(confidence), 0.0), 1.0))


class RuleIntentClassifier(BaseIntentClassifier):
    """LLM 不可用时的规则兜底：高精度、低召回。

    设计边界：规则只放行"几乎不可能误判"的模式——归一化后与寒暄集合精确匹配
    的短输入、含明确诊断祈使词的输入；其余一律返回默认 rag_qa（与无意图识别
    行为等价）。宁可漏识别（落回 rag_qa 只是次优路由），不可误识别：知识问题
    被误判为 chitchat 会绕过检索产生幻觉，被误判为 aiops 会触发昂贵的诊断流。
    """

    # 归一化（去空白与标点、转小写）后精确匹配的寒暄/身份/社交用语。
    _CHITCHAT_EXACT = frozenset(
        {
            "你好", "你好呀", "你好啊", "您好", "您好呀", "嗨", "hi", "hello",
            "在吗", "在么", "你是谁", "你叫什么", "你能做什么", "你能干什么",
            "你会做什么", "你有什么功能", "谢谢", "多谢", "辛苦了", "再见", "拜拜",
        }
    )
    # 明确的诊断祈使词（动词+宾语结构），比单独的"告警/故障"名词精度高得多。
    _AIOPS_KEYWORDS = (
        "帮我诊断", "帮我排查", "诊断一下", "排查一下", "查一下告警",
        "帮我查告警", "看看告警", "分析根因", "定位根因", "根因分析",
        "为什么挂了", "为什么宕机",
    )

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = ctx
        normalized = _normalize_query(question)
        if not normalized:
            return IntentResult(source="rule", error_code="EMPTY_QUERY")
        if normalized in self._CHITCHAT_EXACT:
            return IntentResult(
                label=IntentLabel.CHITCHAT,
                confidence=0.9,
                source="rule",
            )
        if any(keyword in normalized for keyword in self._AIOPS_KEYWORDS):
            return IntentResult(
                label=IntentLabel.AIOPS,
                confidence=0.8,
                source="rule",
            )
        return IntentResult(source="rule")


class FailoverIntentClassifier(BaseIntentClassifier):
    """主分类器（LLM）失败后降级到备用规则分类器。

    主分类器产出可用标签则直接透传；失败（含自身崩溃）时交给规则层高精度识别。
    规则命中非默认标签（chitchat/aiops）则采用规则结果并标记 source="rule"，
    trace 可观测到降级发生；规则未命中则保留主分类器的失败详情返回——label
    天然是默认 rag_qa，fail-open 语义不变。
    """

    def __init__(
        self,
        *,
        primary: BaseIntentClassifier,
        secondary: BaseIntentClassifier,
    ) -> None:
        self.primary = primary
        self.secondary = secondary

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        try:
            primary_result = self.primary.classify(question=question, ctx=ctx)
        except Exception:
            primary_result = IntentResult(error_code="PRIMARY_CLASSIFIER_CRASH")
        if primary_result.successful:
            return primary_result
        secondary_result = self.secondary.classify(question=question, ctx=ctx)
        if secondary_result.label is not IntentLabel.RAG_QA:
            return secondary_result
        return primary_result


def _normalize_query(text: str) -> str:
    """去掉空白与常见标点并转小写，让"你好！"/"你好"命中同一规则。"""

    return re.sub(r"[\s，。！？!?,.、；;：:~～]+", "", text).lower()


class ChitchatResponder:
    """闲聊直答：单次 LLM 调用生成简短回答。

    生成失败返回 None，由编排层 fail-open 落回 RAG 路径；闲聊不检索知识库，
    避免把寒暄语送进向量检索产生无关 citation。
    """

    def __init__(
        self,
        *,
        llm: PromptLLM | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.model = model or config.rag_model

    def answer(self, question: str) -> str | None:
        normalized = question.strip()
        if not normalized:
            return None
        try:
            response = self._get_llm().invoke(_CHITCHAT_PROMPT.format(question=normalized))
            text = llm_content_text(response).strip()
        except Exception:
            return None
        return text or None

    def _get_llm(self) -> PromptLLM:
        if self.llm is None:
            from app.core.llm_factory import LLMFactory

            self.llm = LLMFactory.create_chat_model(
                model=self.model,
                temperature=0.7,
                streaming=False,
            )
        return self.llm


_CHITCHAT_PROMPT = (
    "你是 AegisOps 智能助手。用户正在寒暄或询问你的身份与能力。"
    "请用中文简洁友好地直接回答，一两句话即可，不要编造知识库内容。"
    "可以自然地提示：你能回答运维知识库问题，也能帮你诊断系统故障。\n"
    "用户：{question}"
)


def llm_content_text(response: object) -> str:
    """从 LLM 响应中提取文本，兼容 str/Mapping/对象属性三种形态。"""

    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        content = response.get("content")
        if isinstance(content, str):
            return content
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)
