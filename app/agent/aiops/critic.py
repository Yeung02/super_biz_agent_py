"""Critic 节点：答案级自我批判（ISSUE-B）。

基于 ISSUE-A 的 tool_evidence 证据链，对 replanner 生成的草稿答案做生成后核对：

1. Stage 1 确定性预检（零 LLM 成本）：复用 fallback 的脱敏正则做泄露拦截，
   并计算证据覆盖信号（可用证据数、失败工具与成功表述的矛盾数）注入裁判 prompt。
2. Stage 2 LLM 裁判（轻量模型）：逐断言核对证据块，输出结构化 CriticVerdict。
3. Stage 3 有界修订：verdict=revise 时按"只收缩不扩张"约束重写答案，修订后
   重跑泄露预检；critic_recheck_enabled 开启时每轮修订后复审，accept 提前定稿。

设计不变量：
- fail-open：节点任何异常（含 LLM 超时/解析失败）都放行当前答案——Critic 挂了
  不能阻塞出答案。确定性泄露剥离不受 fail-open 影响（安全优先于可用性）。
- 修订只收缩：只允许删除断言、降级确定性表述、追加诚实说明；修订稿超过原稿
  长度加允许余量视为引入了新内容，直接拒绝采纳。
- 有界修订：critic_max_revisions 在节点内部计数，耗尽即定稿，不回
  executor/replanner——避免与 Plan-Execute-Replan 循环叠加，保住
  agent_max_steps 与 recursion_limit 的既有语义边界。
"""

from textwrap import dedent
from typing import Any, Dict, List, Literal
from dataclasses import dataclass
import asyncio
import re
import time

from pydantic import BaseModel, Field, field_validator
from loguru import logger

from app.config import config
from app.core.fallback import count_sensitive_hits, sanitize_answer_text
from app.core.llm_usage import UsageAccumulator
from app.core.request_context import get_request_context_or_none
from app.core.token_budget import token_budget_manager
from app.observability.tracing import TraceLogger
from .state import PlanExecuteState
from .structured_output import ainvoke_structured_with_retry
from .utils import record_llm_usage

try:
    from langchain_qwq import ChatQwen
except ModuleNotFoundError:
    class ChatQwen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs
            raise RuntimeError("langchain_qwq is required for critic LLM calls")


_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)

# 失败工具与"成功表述"的矛盾检测：证据链中存在 usable=false 块、且草稿中出现
# 这些短语时，裁判 prompt 会收到显式矛盾提示。短语刻意保守（完整词组而非单词），
# 避免把"执行成功与否待确认"这类中性表述误判为矛盾。
_SUCCESS_ASSERTION_RE = re.compile(
    r"已完成|已确认|已成功|执行成功|查询成功|成功获取|已验证"
)

# 修订稿允许的额外字符余量：诚实说明（"以下信息未能获取"）会小幅增加长度，
# 但修订主体是收缩；超过余量视为违反"只收缩"约束，拒绝采纳整份修订稿。
_MAX_REVISION_GROWTH_CHARS = 300


class Claim(BaseModel):
    """草稿中缺少证据支撑的具体断言。"""

    claim: str = Field(description="草稿答案中的具体断言（含数字/结论/引用）")
    evidence_id: str | None = Field(
        default=None,
        description="对应的证据块编号（如 E1）；None 表示找不到任何支撑证据",
    )
    reason: str = Field(default="", description="判定为无证据断言的理由")


class CriticVerdict(BaseModel):
    """Stage 2 裁判输出：逐断言核对证据块后的结论。"""

    verdict: Literal["accept", "revise"] = Field(description="accept 或 revise")
    unsupported_claims: List[Claim] = Field(
        default_factory=list,
        description="无证据支撑的断言列表；verdict=revise 时必须非空",
    )
    revision_notes: str = Field(
        default="",
        description="给修订器的指令，说明每类断言应如何处理",
    )


class RevisedResponse(BaseModel):
    """Stage 3 修订输出。"""

    response: str = Field(min_length=1, description="修订后的完整答案（Markdown）")

    @field_validator("response")
    @classmethod
    def _response_not_blank(cls, value: str) -> str:
        """空修订稿是结构化输出失败：采纳它会用空串覆盖原答案。"""
        if not value.strip():
            raise ValueError("revised response must not be blank")
        return value


@dataclass(frozen=True)
class PrecheckResult:
    """Stage 1 确定性预检结果。

    leak_hits/contradiction_hits 只用于 trace 与裁判 prompt 的信号注入；
    usable_evidence_count 为 0 时节点跳过 LLM 裁判（核对任务没有输入）。
    """

    leak_hits: int
    usable_evidence_count: int
    total_evidence_count: int
    contradiction_hits: int


_JUDGE_SYSTEM_PROMPT = dedent("""
    你是答案审查裁判。对照工具证据块核对草稿答案中的事实断言。

    核对规则：
    - 证据块编号为 E1..En，每块标注"可用=是/否"
    - 可用=否 的证据块是工具失败的不可用声明，不能支撑任何事实断言
    - 草稿中的数字、时间、指标值、因果结论必须能映射到某个可用证据块
    - 无法映射到证据块的断言即为无证据断言

    判定标准：
    - 关键断言均有证据支撑，且没有失败工具被表述为成功 → verdict=accept
    - 存在无证据断言，或失败工具被表述为成功 → verdict=revise 并逐条列出
""").strip()

_REVISE_SYSTEM_PROMPT = dedent("""
    你是答案修订器。根据裁判指出的无证据断言修订草稿答案。

    只允许三类操作：
    1. 删除无证据支撑的断言（数字、结论、引用）
    2. 确定性降级：把"确认/已验证/已完成"改为"未确认/未能获取"
    3. 在结尾追加简短的诚实说明，指出哪些信息因工具失败未能获取

    严格禁止：
    - 引入证据块之外的任何新事实
    - 编造新的数字、时间或结论
    - 大幅扩写或重排原文结构
""").strip()


async def critic(state: PlanExecuteState) -> Dict[str, Any]:
    """答案级自我批判节点：预检 → 裁判 → 有界修订。

    返回的 state 更新只包含 response（修订稿，可能未变化）和 critic_reviewed；
    不触碰 plan/past_steps/tool_evidence 等其余通道。
    """

    response = state.get("response", "")
    evidence_blocks = list(state.get("tool_evidence", []))
    input_text = state.get("input", "")

    if not response:
        # 无响应可审查：直接标记已审，防止条件边反复路由进本节点。
        return {"critic_reviewed": True}

    precheck = _precheck(response, evidence_blocks)
    _record_precheck(precheck)
    # 确定性泄露剥离：不依赖 LLM，fail-open 路径也保留这道防线。
    sanitized = sanitize_answer_text(response)

    # 无任何证据块时（纯 LLM 任务/证据链关闭），跳过 LLM 裁判：核对任务没有
    # 输入，只剩确定性剥离，不值得一次 LLM 调用。
    if not evidence_blocks:
        if sanitized != response:
            return {"response": sanitized, "critic_reviewed": True}
        return {"critic_reviewed": True}

    timeout_seconds = float(getattr(config, "critic_timeout_seconds", 8.0))

    # Stage 2：LLM 裁判。任何失败（超时/异常/解析失败）都 fail-open 放行。
    started = time.monotonic()
    try:
        verdict = await asyncio.wait_for(
            _judge_response(
                input_text=input_text,
                response=response,
                evidence_blocks=evidence_blocks,
                precheck=precheck,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        _record_fail_open(stage="judge", error_code="LLM_TIMEOUT")
        return _unrevised_result(response, sanitized)
    except Exception as exc:
        logger.warning("Critic 裁判失败: {}，放行原答案", exc.__class__.__name__)
        _record_fail_open(stage="judge", error_code="LLM_PROVIDER_ERROR")
        return _unrevised_result(response, sanitized)

    _record_verdict(
        verdict,
        latency_ms=round((time.monotonic() - started) * 1000, 3),
    )

    if verdict.verdict != "revise" or not verdict.unsupported_claims:
        # accept：无证据断言为空，无需修订；泄露剥离（若有）仍然生效。
        return _unrevised_result(response, sanitized)

    # Stage 3：有界修订。
    max_revisions = max(0, int(getattr(config, "critic_max_revisions", 1)))
    recheck_enabled = bool(getattr(config, "critic_recheck_enabled", False))
    if max_revisions <= 0:
        return _unrevised_result(response, sanitized)

    current = response
    for revision_round in range(1, max_revisions + 1):
        try:
            revised = await asyncio.wait_for(
                _revise_response(response=current, verdict=verdict),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            _record_fail_open(stage="revise", error_code="LLM_TIMEOUT")
            return _final_result(current, sanitized, response)
        except Exception as exc:
            logger.warning(
                "Critic 修订失败: {}，放行当前答案", exc.__class__.__name__
            )
            _record_fail_open(stage="revise", error_code="LLM_PROVIDER_ERROR")
            return _final_result(current, sanitized, response)

        # 只收缩护栏：以原始草稿长度为基准（而非当前版本），防止多轮累计膨胀。
        if len(revised) > len(response) + _MAX_REVISION_GROWTH_CHARS:
            logger.warning(
                "修订稿长度 {} 超出允许余量（原稿 {} + {}），拒绝采纳",
                len(revised),
                len(response),
                _MAX_REVISION_GROWTH_CHARS,
            )
            return _final_result(current, sanitized, response)

        # 修订后重跑确定性泄露检查（免费防线，不依赖 critic_recheck_enabled）。
        current = sanitize_answer_text(revised)
        _record_revise(
            revision_round=revision_round,
            removed_chars=len(response) - len(current),
        )

        if not recheck_enabled:
            break

        # 复审：accept 提前定稿；复审失败不阻塞，定稿当前版本。
        try:
            verdict = await asyncio.wait_for(
                _judge_response(
                    input_text=input_text,
                    response=current,
                    evidence_blocks=evidence_blocks,
                    precheck=_precheck(current, evidence_blocks),
                ),
                timeout=timeout_seconds,
            )
            _record_verdict(verdict, latency_ms=0.0)
        except Exception as exc:
            logger.warning(
                "Critic 复审失败: {}，定稿当前版本", exc.__class__.__name__
            )
            break
        if verdict.verdict != "revise" or not verdict.unsupported_claims:
            break

    return _final_result(current, sanitized, response)


async def _judge_response(
    *,
    input_text: str,
    response: str,
    evidence_blocks: List[dict[str, Any]],
    precheck: PrecheckResult,
) -> CriticVerdict:
    """Stage 2：轻量模型 + 结构化输出核对草稿断言。"""

    llm = ChatQwen(
        model=config.critic_model,
        api_key=config.dashscope_api_key,
        temperature=0,
    )
    chain = llm.with_structured_output(CriticVerdict)

    request_ctx = get_request_context_or_none()
    evidence_text = _format_evidence_blocks(evidence_blocks)
    # 证据块在 ISSUE-A 已按块截断，但极端场景（工具预算放开）总量仍可能膨胀；
    # 沿用 aiops_report 场景的 tool_result 预算做整体裁剪，与 replanner 的
    # execution_history 裁剪同一模式。裁判输入刻意不含 past_steps 全量。
    budget = token_budget_manager.allocate(
        "aiops_report",
        config.critic_model,
        request_ctx,
        current_input=input_text,
    )
    if config.token_budget_enabled:
        trim_result = token_budget_manager.trim_text(
            evidence_text,
            budget.tool_result_tokens,
        )
        evidence_text = str(trim_result.content)

    messages = [
        ("system", _JUDGE_SYSTEM_PROMPT),
        ("user", f"原始任务: {input_text}"),
        ("user", f"工具证据块:\n{evidence_text}"),
        ("user", f"草稿答案:\n{response}"),
    ]
    if precheck.contradiction_hits > 0:
        messages.append(
            (
                "user",
                f"⚠️ 检测到 {precheck.contradiction_hits} 处失败工具与成功表述的"
                "矛盾，请重点核对这些断言是否被证据支撑。",
            )
        )

    # 裸消息列表直连 chain；解析失败（含非法 verdict 枚举值）在重试层带错误
    # 反馈重试一次，仍失败由节点主函数 fail-open 放行原答案。
    # 裁判调用的 usage 在 finally 记账：重试耗尽 fail-open 前已消耗的 token 不丢。
    judge_usage_accumulator = UsageAccumulator()
    try:
        return await ainvoke_structured_with_retry(
            chain,
            messages,
            schema=CriticVerdict,
            node="critic.judge",
            usage_accumulator=judge_usage_accumulator,
        )
    finally:
        record_llm_usage(
            judge_usage_accumulator, model=config.critic_model, ctx=request_ctx
        )


async def _revise_response(*, response: str, verdict: CriticVerdict) -> str:
    """Stage 3：按"只收缩不扩张"约束修订草稿。"""

    request_ctx = get_request_context_or_none()
    llm = ChatQwen(
        model=config.critic_model,
        api_key=config.dashscope_api_key,
        temperature=0,
    )
    chain = llm.with_structured_output(RevisedResponse)

    claims_text = "\n".join(
        f"- {item.claim}（理由: {item.reason or '无证据支撑'}）"
        for item in verdict.unsupported_claims
    )
    messages = [
        ("system", _REVISE_SYSTEM_PROMPT),
        ("user", f"草稿答案:\n{response}"),
        ("user", f"裁判指出的无证据断言:\n{claims_text}"),
        (
            "user",
            f"修订指令: {verdict.revision_notes}"
            if verdict.revision_notes
            else "请按规则修订以上断言。",
        ),
    ]

    # 空/空白修订稿会被 RevisedResponse schema 拒绝并重试；重试耗尽仍失败时
    # 抛给节点主函数 fail-open，不会用空串覆盖原答案。
    revise_usage_accumulator = UsageAccumulator()
    try:
        revised = await ainvoke_structured_with_retry(
            chain,
            messages,
            schema=RevisedResponse,
            node="critic.revise",
            usage_accumulator=revise_usage_accumulator,
        )
    finally:
        record_llm_usage(
            revise_usage_accumulator, model=config.critic_model, ctx=request_ctx
        )
    return revised.response


def _precheck(
    response: str,
    evidence_blocks: List[dict[str, Any]],
) -> PrecheckResult:
    """Stage 1：零 LLM 成本的确定性预检。"""

    leak_hits = count_sensitive_hits(response)
    usable_count = sum(1 for block in evidence_blocks if bool(block.get("usable")))
    has_unusable = any(not bool(block.get("usable")) for block in evidence_blocks)
    contradiction_hits = (
        len(_SUCCESS_ASSERTION_RE.findall(response)) if has_unusable else 0
    )
    return PrecheckResult(
        leak_hits=leak_hits,
        usable_evidence_count=usable_count,
        total_evidence_count=len(evidence_blocks),
        contradiction_hits=contradiction_hits,
    )


def _format_evidence_blocks(evidence_blocks: List[dict[str, Any]]) -> str:
    """把 state.tool_evidence 渲染为裁判可读的编号证据块文本。"""

    if not evidence_blocks:
        return "（无工具证据）"
    lines: list[str] = []
    for index, block in enumerate(evidence_blocks, 1):
        usable = "是" if bool(block.get("usable")) else "否"
        tool_name = str(block.get("tool_name", "unknown"))
        step = str(block.get("step", ""))
        text = str(block.get("text", ""))
        lines.append(f"[E{index}] 工具={tool_name} 可用={usable} 步骤={step}\n{text}")
    return "\n\n".join(lines)


def _unrevised_result(response: str, sanitized: str) -> Dict[str, Any]:
    """accept / 裁判前跳过路径的定稿：泄露剥离生效，内容不修改。"""

    if sanitized != response:
        return {"response": sanitized, "critic_reviewed": True}
    return {"critic_reviewed": True}


def _final_result(
    current: str,
    sanitized: str,
    original: str,
) -> Dict[str, Any]:
    """修订循环结束后的定稿。

    current 为零轮成功修订（等于 original）时退化为泄露剥离版，与
    _unrevised_result 行为一致；发生过修订则返回修订稿。
    """

    if current == original:
        return _unrevised_result(original, sanitized)
    return {"response": current, "critic_reviewed": True}


def _record_precheck(precheck: PrecheckResult) -> None:
    """记录预检信号；不写入草稿原文与证据文本。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.critic.precheck",
        ctx,
        leak_hits=precheck.leak_hits,
        usable_evidence_count=precheck.usable_evidence_count,
        total_evidence_count=precheck.total_evidence_count,
        contradiction_hits=precheck.contradiction_hits,
    )


def _record_verdict(verdict: CriticVerdict, *, latency_ms: float) -> None:
    """记录裁判结论与耗时；不记录断言原文，避免草稿内容进 trace。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.critic.verdict",
        ctx,
        verdict=verdict.verdict,
        unsupported_claim_count=len(verdict.unsupported_claims),
        latency_ms=latency_ms,
        model=config.critic_model,
    )


def _record_revise(*, revision_round: int, removed_chars: int) -> None:
    """记录修订轮次与收缩量（可为负：诚实说明导致的小幅增长）。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.critic.revise",
        ctx,
        revision_round=revision_round,
        removed_chars=removed_chars,
    )


def _record_fail_open(*, stage: str, error_code: str) -> None:
    """记录 fail-open：观测 Critic 故障率，评估灰度质量。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.critic.fail_open",
        ctx,
        status="error",
        error_code=error_code,
        stage=stage,
    )
