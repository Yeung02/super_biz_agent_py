"""意图识别与 orchestrator 意图路由测试。

分类器/直答器用内存 fake LLM 覆盖 JSON 解析、非法输出、异常与空输入边界；
路由测试注入 fake classifier，验证 chitchat/rag_qa/aiops 三分支各自调用对应
service，以及分类失败时 fail-open 回 RAG。全程不访问 DashScope、Milvus 或网络。
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from app.agent.intent import (
    ChitchatResponder,
    FailoverIntentClassifier,
    IntentLabel,
    IntentResult,
    LlmIntentClassifier,
    NoopIntentClassifier,
    RuleIntentClassifier,
)
from app.agent.orchestrator import AgentOrchestrator
from app.core.errors import AppError
from app.core.fallback import FallbackManager
from app.core.token_budget import TokenBudgetManager
from app.observability.tracing import TraceLogger

if TYPE_CHECKING:
    from pathlib import Path

    from app.core.request_context import RequestContext


class _FakeContentLLM:
    """意图模块测试用同步 LLM fake，只实现 invoke(prompt) -> 带content对象。"""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


class _FakeClassifier:
    """按配置返回固定意图，或抛异常模拟分类器崩溃。"""

    def __init__(self, label: IntentLabel, error: Exception | None = None) -> None:
        self.label = label
        self.error = error
        self.calls: list[str] = []

    def classify(self, *, question: str, ctx: RequestContext | None = None) -> IntentResult:
        _ = ctx
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return IntentResult(label=self.label, confidence=0.9)


class _ErrorResultClassifier:
    """恒返失败结果（不抛异常），模拟 LLM 超时/供应商错误的正常失败路径。"""

    def __init__(self, error_code: str = "LLM_TIMEOUT") -> None:
        self.error_code = error_code
        self.calls = 0

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = question, ctx
        self.calls += 1
        return IntentResult(error_code=self.error_code)


class _OkClassifier:
    """恒返成功结果，验证 failover 对成功主分类器的透传。"""

    def classify(
        self,
        *,
        question: str,
        ctx: RequestContext | None = None,
    ) -> IntentResult:
        _ = question, ctx
        return IntentResult(label=IntentLabel.RAG_QA, confidence=0.99, source="llm")


class _FakeResponder:
    """按配置返回固定闲聊回答；answer_value=None 模拟直答失败。"""

    def __init__(self, answer_value: str | None = "你好，我是 AegisOps 助手") -> None:
        self.answer_value = answer_value
        self.calls: list[str] = []

    def answer(self, question: str) -> str | None:
        self.calls.append(question)
        return self.answer_value


@dataclass
class _FakeRagService:
    answer: str = "rag answer"

    def __post_init__(self) -> None:
        self.query_calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []
        self.model_name = "qwen-test"
        self.system_prompt = "system prompt"

    async def query(self, question: str, session_id: str) -> str:
        self.query_calls.append((question, session_id))
        return self.answer

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        self.stream_calls.append((question, session_id))
        yield {"type": "content", "data": "rag chunk"}
        yield {"type": "complete"}


@dataclass
class _FakeAiopsService:
    mode: str = "success"

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    async def diagnose(
        self, session_id: str = "default"
    ) -> AsyncGenerator[dict[str, object], None]:
        self.calls.append(session_id)
        if self.mode == "error":
            yield {
                "type": "error",
                "error": {"code": "AGENT_MAX_STEP_EXCEEDED", "message": "步骤过多"},
            }
            return
        yield {"type": "status", "stage": "fetching_alerts", "message": "正在获取系统告警信息"}
        yield {
            "type": "plan",
            "stage": "plan_created",
            "message": "诊断计划已制定",
            "plan": ["检查告警", "分析日志"],
        }
        yield {
            "type": "step_complete",
            "message": "步骤执行完成 (1/2)",
            "current_step": "检查告警",
        }
        yield {"type": "report", "stage": "final_report", "report": "# 诊断报告\n根因：CPU 飙高"}
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "diagnosis": {"status": "completed", "report": "complete 兜底报告"},
        }


class _FakeConversationManager:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_context(
        self,
        session_id: str,
        budget: object,
        ctx: RequestContext | None = None,
    ) -> object:
        _ = budget, ctx
        self.loaded.append(session_id)
        return SimpleNamespace(summary=None, recent_messages=(), history_metadata={})


def _build_orchestrator(
    *,
    rag_service: _FakeRagService | None = None,
    aiops_service: _FakeAiopsService | None = None,
    classifier: object | None = None,
    responder: object | None = None,
    trace_path: Path | None = None,
) -> AgentOrchestrator:
    trace_logger = TraceLogger(
        trace_jsonl_path=str(trace_path or "unused"),
        enabled=trace_path is not None,
    )
    return AgentOrchestrator(
        rag_service=rag_service or _FakeRagService(),
        aiops_service=aiops_service or _FakeAiopsService(),
        fallback_manager=FallbackManager(trace_logger=trace_logger, enabled=False),
        token_budget_manager=TokenBudgetManager(
            model_context_windows={"qwen-test": 4096, "default": 4096},
            trace_logger=trace_logger,
        ),
        conversation_manager=_FakeConversationManager(),
        trace_logger=trace_logger,
        enabled=True,
        intent_classifier=classifier,  # type: ignore[arg-type]
        chitchat_responder=responder,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 分类器边界
# ---------------------------------------------------------------------------


def test_noop_classifier_returns_rag_default() -> None:
    """未启用意图识别时 Noop 恒返 rag_qa，等价于旧对话行为。"""

    result = NoopIntentClassifier().classify(question="你好")

    assert result.label is IntentLabel.RAG_QA
    assert result.successful is True


def test_llm_classifier_parses_label_and_confidence() -> None:
    """合法 JSON 输出映射为标签与置信度，prompt 携带用户输入。"""

    llm = _FakeContentLLM(content='{"label": "aiops", "confidence": 0.88}')
    classifier = LlmIntentClassifier(llm=llm)

    result = classifier.classify(question="帮我看看告警")

    assert result.successful is True
    assert result.label is IntentLabel.AIOPS
    assert result.confidence == pytest.approx(0.88)
    assert llm.calls and "帮我看看告警" in llm.calls[0]


def test_llm_classifier_extracts_json_from_wrapped_text() -> None:
    """模型在 JSON 前后附加解释文本时仍可提取，置信度越界时收敛到 [0,1]。"""

    llm = _FakeContentLLM(content='结果如下 {"label": "chitchat", "confidence": 1.7} 请参考')
    classifier = LlmIntentClassifier(llm=llm)

    result = classifier.classify(question="你好")

    assert result.label is IntentLabel.CHITCHAT
    assert result.confidence == 1.0


@pytest.mark.parametrize(
    "content",
    ["我觉得是闲聊", '{"label": "smalltalk", "confidence": 0.5}', ""],
)
def test_llm_classifier_fails_open_on_invalid_output(content: str) -> None:
    """非法/非 JSON/未知标签输出 fail-open：带 error_code 回到默认 rag_qa。"""

    llm = _FakeContentLLM(content=content)
    classifier = LlmIntentClassifier(llm=llm)

    result = classifier.classify(question="任意输入")

    assert result.successful is False
    assert result.label is IntentLabel.RAG_QA
    assert result.error_code in {"UNPARSEABLE_RESPONSE", "INVALID_LABEL"}


def test_llm_classifier_maps_timeout_and_provider_error() -> None:
    """LLM 超时与供应商异常映射为稳定 error_code，不向调用方抛出。"""

    timeout_classifier = LlmIntentClassifier(
        llm=_FakeContentLLM(error=TimeoutError("fake timeout")),
    )
    provider_classifier = LlmIntentClassifier(
        llm=_FakeContentLLM(error=RuntimeError("fake provider error")),
    )

    assert timeout_classifier.classify(question="q").error_code == "LLM_TIMEOUT"
    assert provider_classifier.classify(question="q").error_code == "LLM_PROVIDER_ERROR"


def test_llm_classifier_empty_question_short_circuits() -> None:
    """空输入直接返回 EMPTY_QUERY，不触发 LLM 调用。"""

    llm = _FakeContentLLM(content='{"label": "chitchat", "confidence": 0.9}')
    classifier = LlmIntentClassifier(llm=llm)

    result = classifier.classify(question="   ")

    assert result.error_code == "EMPTY_QUERY"
    assert llm.calls == []


def test_llm_classifier_lazy_client_configures_sdk_retry(monkeypatch) -> None:
    """延迟创建的真实客户端带 SDK 级重试，且单次超时按尝试次数均分。

    重试必须能在编排层 wait_for 硬顶内跑完：3s 预算、1 次重试 →
    单次超时 3.0*0.8/2 = 1.2s，最坏 2.4s < 3s。
    """

    captured: dict[str, object] = {}

    def _fake_create_chat_model(**kwargs: object) -> _FakeContentLLM:
        captured.update(kwargs)
        return _FakeContentLLM(content='{"label": "chitchat", "confidence": 0.9}')

    monkeypatch.setattr(
        "app.core.llm_factory.LLMFactory.create_chat_model",
        _fake_create_chat_model,
    )
    classifier = LlmIntentClassifier()

    result = classifier.classify(question="你好")

    assert result.label is IntentLabel.CHITCHAT
    assert captured["max_retries"] == 1
    assert captured["timeout"] == pytest.approx(1.2)
    assert captured["temperature"] == 0.0


# ---------------------------------------------------------------------------
# 规则兜底分类器边界
# ---------------------------------------------------------------------------


def test_rule_classifier_greeting_hits_chitchat() -> None:
    """归一化后精确匹配的寒暄/身份用语命中 chitchat，标点与空白不影响。"""

    classifier = RuleIntentClassifier()

    assert classifier.classify(question="你好！").label is IntentLabel.CHITCHAT
    assert classifier.classify(question=" 你好呀 ").label is IntentLabel.CHITCHAT
    assert classifier.classify(question="你是谁？").label is IntentLabel.CHITCHAT
    assert classifier.classify(question="Hello").label is IntentLabel.CHITCHAT


def test_rule_classifier_diagnostic_imperative_hits_aiops() -> None:
    """含明确诊断祈使词的输入命中 aiops。"""

    classifier = RuleIntentClassifier()

    assert classifier.classify(question="帮我诊断一下系统").label is IntentLabel.AIOPS
    assert classifier.classify(question="看看告警").label is IntentLabel.AIOPS
    assert (
        classifier.classify(question="订单服务为什么挂了").label is IntentLabel.AIOPS
    )


def test_rule_classifier_defaults_to_rag_on_no_confident_match() -> None:
    """规则未命中高精度模式时返回默认 rag_qa：宁可漏识别，不可误路由。"""

    classifier = RuleIntentClassifier()

    # 问候 + 业务诉求混合输入不能误判为 chitchat（绕过检索会幻觉）。
    mixed = classifier.classify(question="你好，灰度策略是什么")
    assert mixed.label is IntentLabel.RAG_QA
    # 知识型故障咨询不含诊断祈使词，不是 aiops。
    knowledge = classifier.classify(question="发布流程报错怎么处理")
    assert knowledge.label is IntentLabel.RAG_QA
    # 空输入沿用 EMPTY_QUERY 语义。
    assert classifier.classify(question="  ").error_code == "EMPTY_QUERY"


def test_rule_classifier_marks_source() -> None:
    """规则结果携带 source=rule，供 trace 观测降级链路命中层。"""

    result = RuleIntentClassifier().classify(question="你好")

    assert result.source == "rule"


# ---------------------------------------------------------------------------
# failover 组合与生产装配
# ---------------------------------------------------------------------------


def test_failover_uses_rule_result_when_primary_fails() -> None:
    """主分类器失败（结果失败或崩溃）时降级到规则层，命中则采用规则标签。"""

    by_error = FailoverIntentClassifier(
        primary=_ErrorResultClassifier("LLM_TIMEOUT"),
        secondary=RuleIntentClassifier(),
    )
    by_crash = FailoverIntentClassifier(
        primary=_FakeClassifier(IntentLabel.CHITCHAT, error=RuntimeError("boom")),
        secondary=RuleIntentClassifier(),
    )

    rule_hit = by_error.classify(question="你好！")
    assert rule_hit.label is IntentLabel.CHITCHAT
    assert rule_hit.source == "rule"
    assert by_crash.classify(question="帮我排查一下").label is IntentLabel.AIOPS


def test_failover_preserves_primary_error_when_rules_miss() -> None:
    """规则未命中时保留主分类器失败详情：label 为默认 rag_qa，fail-open 不变。"""

    failover = FailoverIntentClassifier(
        primary=_ErrorResultClassifier("LLM_PROVIDER_ERROR"),
        secondary=RuleIntentClassifier(),
    )

    result = failover.classify(question="灰度发布策略是什么")

    assert result.label is IntentLabel.RAG_QA
    assert result.error_code == "LLM_PROVIDER_ERROR"
    assert result.source == "llm"


def test_failover_passthrough_on_primary_success() -> None:
    """主分类器成功时结果原样透传，不触发规则层。"""

    primary = _OkClassifier()
    secondary = RuleIntentClassifier()
    failover = FailoverIntentClassifier(primary=primary, secondary=secondary)

    result = failover.classify(question="你好")

    assert result.label is IntentLabel.RAG_QA
    assert result.confidence == pytest.approx(0.99)
    assert result.source == "llm"


def test_default_assembly_enables_rule_fallback(monkeypatch) -> None:
    """生产装配按开关组装降级链：意图开启时主 LLM 外包一层规则 failover。"""

    from app.agent import orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module.config, "intent_enabled", True)
    monkeypatch.setattr(orchestrator_module.config, "intent_rule_fallback_enabled", True)

    classifier = orchestrator_module._default_intent_classifier()

    assert isinstance(classifier, FailoverIntentClassifier)
    assert isinstance(classifier.primary, LlmIntentClassifier)
    assert isinstance(classifier.secondary, RuleIntentClassifier)


def test_default_assembly_respects_switches(monkeypatch) -> None:
    """规则兜底关闭时直接返回 LLM 分类器；意图总开关关闭时返回 Noop。"""

    from app.agent import orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module.config, "intent_enabled", True)
    monkeypatch.setattr(orchestrator_module.config, "intent_rule_fallback_enabled", False)
    assert isinstance(
        orchestrator_module._default_intent_classifier(), LlmIntentClassifier
    )

    monkeypatch.setattr(orchestrator_module.config, "intent_enabled", False)
    assert isinstance(
        orchestrator_module._default_intent_classifier(), NoopIntentClassifier
    )


# ---------------------------------------------------------------------------
# 闲聊直答器边界
# ---------------------------------------------------------------------------


def test_chitchat_responder_returns_answer() -> None:
    llm = _FakeContentLLM(content="你好！我可以帮你查知识库或诊断故障。")
    responder = ChitchatResponder(llm=llm)

    answer = responder.answer("你是谁")

    assert answer == "你好！我可以帮你查知识库或诊断故障。"
    assert llm.calls and "你是谁" in llm.calls[0]


def test_chitchat_responder_returns_none_on_failure() -> None:
    """直答失败返回 None 而不是异常，由编排层落回 RAG 路径。"""

    error_responder = ChitchatResponder(llm=_FakeContentLLM(error=RuntimeError("boom")))
    empty_responder = ChitchatResponder(llm=_FakeContentLLM(content="  "))

    assert error_responder.answer("你好") is None
    assert empty_responder.answer("你好") is None


# ---------------------------------------------------------------------------
# orchestrator 意图路由
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chat_aiops_intent_triggers_diagnosis(
    fake_request_context: RequestContext,
) -> None:
    """aiops 意图在对话入口直接触发诊断流，报告全文作为 answer，不调用 RAG。"""

    rag_service = _FakeRagService()
    aiops_service = _FakeAiopsService()
    classifier = _FakeClassifier(IntentLabel.AIOPS)
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        aiops_service=aiops_service,
        classifier=classifier,
    )

    result = await orchestrator.run_chat(
        question="帮我诊断一下为什么服务超时",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "# 诊断报告\n根因：CPU 飙高"
    assert result.fallback_used is False
    assert aiops_service.calls == ["session-test"]
    assert rag_service.query_calls == []
    assert classifier.calls == ["帮我诊断一下为什么服务超时"]


@pytest.mark.asyncio
async def test_run_chat_chitchat_intent_short_circuits(
    fake_request_context: RequestContext,
) -> None:
    """chitchat 意图单次直答返回，跳过 RAG 检索与 Agent 循环。"""

    rag_service = _FakeRagService()
    responder = _FakeResponder()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        classifier=_FakeClassifier(IntentLabel.CHITCHAT),
        responder=responder,
    )

    result = await orchestrator.run_chat(
        question="你好",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "你好，我是 AegisOps 助手"
    assert rag_service.query_calls == []
    assert responder.calls == ["你好"]


@pytest.mark.asyncio
async def test_run_chat_chitchat_failure_falls_back_to_rag(
    fake_request_context: RequestContext,
) -> None:
    """直答器失败（返回 None）时 fail-open 落回 RAG 路径，不产生空回答。"""

    rag_service = _FakeRagService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        classifier=_FakeClassifier(IntentLabel.CHITCHAT),
        responder=_FakeResponder(answer_value=None),
    )

    result = await orchestrator.run_chat(
        question="你好",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "rag answer"
    assert rag_service.query_calls == [("你好", "session-test")]


@pytest.mark.asyncio
async def test_run_chat_classifier_exception_fails_open_to_rag(
    fake_request_context: RequestContext,
) -> None:
    """分类器崩溃被编排层吞掉，请求继续走 RAG，意图识别不阻塞对话链路。"""

    rag_service = _FakeRagService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        classifier=_FakeClassifier(IntentLabel.AIOPS, error=RuntimeError("classifier boom")),
    )

    result = await orchestrator.run_chat(
        question="任意问题",
        session_id="session-test",
        ctx=fake_request_context,
    )

    assert result.answer == "rag answer"
    assert rag_service.query_calls == [("任意问题", "session-test")]


@pytest.mark.asyncio
async def test_run_chat_aiops_error_event_raises_stable_app_error(
    fake_request_context: RequestContext,
) -> None:
    """诊断流中的 error 事件恢复为稳定 AppError，交给既有 fallback/错误契约。"""

    aiops_service = _FakeAiopsService(mode="error")
    orchestrator = _build_orchestrator(
        aiops_service=aiops_service,
        classifier=_FakeClassifier(IntentLabel.AIOPS),
    )

    with pytest.raises(AppError) as exc_info:
        await orchestrator.run_chat(
            question="帮我诊断",
            session_id="session-test",
            ctx=fake_request_context,
        )

    assert exc_info.value.code == "AGENT_MAX_STEP_EXCEEDED"


@pytest.mark.asyncio
async def test_run_chat_stream_aiops_intent_maps_events_to_chat_chunks() -> None:
    """流式 aiops 分支把计划/进度/报告翻译为 content 分片，complete 保持 chat 契约。"""

    rag_service = _FakeRagService()
    aiops_service = _FakeAiopsService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        aiops_service=aiops_service,
        classifier=_FakeClassifier(IntentLabel.AIOPS),
    )

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="帮我诊断",
            session_id="session-test",
            ctx=None,
        )
    ]

    content_text = "".join(
        str(chunk["data"]) for chunk in chunks if chunk.get("type") == "content"
    )
    assert "诊断计划（共 2 步）" in content_text
    assert "检查告警" in content_text
    assert "# 诊断报告\n根因：CPU 飙高" in content_text
    assert chunks[-1] == {"type": "complete"}
    assert aiops_service.calls == ["session-test"]
    assert rag_service.stream_calls == []


@pytest.mark.asyncio
async def test_run_chat_stream_chitchat_single_answer_chunks() -> None:
    """流式 chitchat 分支输出单个 content + complete，不触碰 RAG。"""

    rag_service = _FakeRagService()
    orchestrator = _build_orchestrator(
        rag_service=rag_service,
        classifier=_FakeClassifier(IntentLabel.CHITCHAT),
        responder=_FakeResponder(),
    )

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="你好",
            session_id="session-test",
            ctx=None,
        )
    ]

    assert chunks == [
        {"type": "content", "data": "你好，我是 AegisOps 助手"},
        {"type": "complete"},
    ]
    assert rag_service.stream_calls == []


@pytest.mark.asyncio
async def test_run_chat_stream_rag_default_unchanged() -> None:
    """默认 Noop 分类器（等价 intent_enabled=false）时流式行为与旧 RAG 路径一致。"""

    rag_service = _FakeRagService()
    orchestrator = _build_orchestrator(rag_service=rag_service)

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="知识问题",
            session_id="session-test",
            ctx=None,
        )
    ]

    assert rag_service.stream_calls == [("知识问题", "session-test")]
    assert chunks[0] == {"type": "content", "data": "rag chunk"}
    assert chunks[-1]["type"] == "complete"


@pytest.mark.asyncio
async def test_run_chat_stream_aiops_error_yields_error_chunk() -> None:
    """流式 aiops 分支的 error 事件透传给上层，按 chat error 契约结束流。"""

    orchestrator = _build_orchestrator(
        aiops_service=_FakeAiopsService(mode="error"),
        classifier=_FakeClassifier(IntentLabel.AIOPS),
    )

    chunks = [
        chunk
        async for chunk in orchestrator.run_chat_stream(
            question="帮我诊断",
            session_id="session-test",
            ctx=None,
        )
    ]

    assert chunks and chunks[-1].get("type") == "error"
    assert not any(chunk.get("type") == "complete" for chunk in chunks)


@pytest.mark.asyncio
async def test_run_chat_records_intent_trace_event(
    tmp_path: Path,
    fake_request_context: RequestContext,
) -> None:
    """意图分流结果写入 trace，用于观测流量构成与分类失败率。"""

    trace_path = tmp_path / "trace.jsonl"
    orchestrator = _build_orchestrator(
        classifier=_FakeClassifier(IntentLabel.AIOPS),
        trace_path=trace_path,
    )

    await orchestrator.run_chat(
        question="帮我诊断",
        session_id="session-test",
        ctx=fake_request_context,
    )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    intent_events = [event for event in events if event["name"] == "orchestrator.intent"]

    assert intent_events and intent_events[0]["intent"] == "aiops"
    assert intent_events[0]["mode"] == "chat"
