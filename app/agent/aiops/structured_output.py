"""结构化输出统一解析、有界重试与观测。

planner/replanner/critic 的 LLM 结构化输出原先解析失败后直接 fail-open（默认
计划/继续原计划/放行原答案），一次 JSON 格式抖动就会让整条链路退化，且失败只
写日志、无 trace 事件。本模块在 fail-open 之前插入最后一道修复：

1. 统一 coercion：chain 返回的 schema 实例/dict 一律规范化为 Pydantic 实例；
   非法形态抛 StructuredOutputCoercionError，与 ValidationError 一样进入重试，
   不再静默走旧 dict 兜底分支。
2. 有界重试：解析类异常（OutputParserException/ValidationError/JSONDecodeError/
   coercion 失败）触发带错误反馈的重试；网络、超时、provider 错误不重试，仍走
   各节点既有 fail-open 路径，不在这里放大延迟。
3. 观测：每次解析失败与重试成功都写 trace 事件
   （error_code=STRUCTURED_OUTPUT_PARSE_ERROR），灰度时可量化各节点解析失败率。

设计约束：
- 重试提示只包含错误类型与 schema 名，不回显模型原始输出，避免脏内容二次注入
  prompt 或进入日志。
- payload 支持两种形态：prompt 模板输入（dict 含 messages 键，planner/replanner）
  或裸消息列表（critic 直连 chain）。重试提示以 user 消息追加，不修改原 payload。
- 重试耗尽后抛出最后一次解析异常，由调用节点按既有策略 fail-open。
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from app.config import config
from app.core.llm_usage import UsageAccumulator
from app.core.request_context import get_request_context_or_none
from app.observability.tracing import TraceLogger

ModelT = TypeVar("ModelT", bound=BaseModel)

_trace_logger = TraceLogger(
    trace_jsonl_path=config.trace_jsonl_path,
    enabled=config.trace_enabled,
)


class StructuredOutputCoercionError(ValueError):
    """chain 返回值无法规范化为 schema 实例（既不是实例也不是 dict）。"""


# 视为"解析失败"可重试的异常。provider/网络/超时错误不在其中，直接向上抛出。
PARSE_ERROR_TYPES: tuple[type[Exception], ...] = (
    OutputParserException,
    ValidationError,
    json.JSONDecodeError,
    StructuredOutputCoercionError,
)


async def ainvoke_structured_with_retry(
    chain: Any,
    payload: Any,
    *,
    schema: type[ModelT],
    node: str,
    usage_accumulator: UsageAccumulator | None = None,
) -> ModelT:
    """调用结构化输出 chain 并规范化为 schema 实例；解析失败有界重试。

    重试次数由 config.structured_output_retry_count 控制（额外次数，不含首次）；
    structured_output_retry_enabled=false 时回到"解析失败直接抛出"的旧行为。

    usage_accumulator 不为 None 时以 callbacks config 传入 chain，用于采集每次
    LLM 调用（含解析失败重试）的真实 usage；为 None 时不传 config，兼容旧 fake
    chain 的 ainvoke(payload) 签名。
    """

    retry_enabled = bool(getattr(config, "structured_output_retry_enabled", True))
    extra_attempts = (
        max(0, int(getattr(config, "structured_output_retry_count", 1)))
        if retry_enabled
        else 0
    )

    last_exc: Exception | None = None
    for attempt in range(extra_attempts + 1):
        current_payload = (
            _append_repair_hint(payload, schema.__name__, last_exc)
            if attempt > 0
            else payload
        )
        try:
            if usage_accumulator is not None:
                raw = await chain.ainvoke(
                    current_payload, config={"callbacks": [usage_accumulator]}
                )
            else:
                raw = await chain.ainvoke(current_payload)
            result = _coerce(raw, schema)
            if attempt > 0:
                _record_retry_success(
                    node=node,
                    schema_name=schema.__name__,
                    attempt=attempt,
                )
            return result
        except PARSE_ERROR_TYPES as exc:
            last_exc = exc
            _record_parse_error(
                node=node,
                schema_name=schema.__name__,
                attempt=attempt,
                will_retry=attempt < extra_attempts,
            )

    assert last_exc is not None
    raise last_exc


def _coerce(raw: object, schema: type[ModelT]) -> ModelT:
    """把 chain 返回值规范化为 schema 实例。

    dict 形态交给 Pydantic 校验（非法值抛 ValidationError，可触发重试）；
    既不是实例也不是 dict 的形态抛 StructuredOutputCoercionError。
    """

    if isinstance(raw, schema):
        return raw
    if isinstance(raw, dict):
        return schema(**raw)
    raise StructuredOutputCoercionError(
        f"unparseable {schema.__name__} output: {type(raw).__name__}"
    )


def _append_repair_hint(payload: Any, schema_name: str, exc: Exception | None) -> Any:
    """把修复提示追加到 payload 的 messages 末尾；不修改原始 payload。

    提示只包含错误类型与 schema 名：Pydantic 校验错误文本可能回显模型输出片段，
    不能把它二次注入 prompt 或写进日志。
    """

    hint = (
        f"上一次输出未能解析为 {schema_name} 结构"
        f"（错误类型: {exc.__class__.__name__ if exc else 'unknown'}）。"
        "请严格按 schema 要求重新输出，不要输出任何多余文本。"
    )
    if isinstance(payload, dict):
        if "messages" not in payload:
            # 模板不含 messages 键时无法注入提示：原样重试，保留一次机会。
            return payload
        messages = list(payload.get("messages") or [])
        messages.append(("user", hint))
        return {**payload, "messages": messages}
    if isinstance(payload, list):
        return [*payload, ("user", hint)]
    return payload


def _record_parse_error(
    *,
    node: str,
    schema_name: str,
    attempt: int,
    will_retry: bool,
) -> None:
    """记录解析失败事件；不落模型输出原文，只落稳定信号。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.structured_output.parse_error",
        ctx,
        status="error",
        error_code="STRUCTURED_OUTPUT_PARSE_ERROR",
        node=node,
        schema=schema_name,
        attempt=attempt,
        will_retry=will_retry,
    )


def _record_retry_success(*, node: str, schema_name: str, attempt: int) -> None:
    """记录重试后解析成功事件，用于评估修复重试的实际收益。"""

    ctx = get_request_context_or_none()
    if ctx is None:
        return
    _trace_logger.record_event(
        "agent.structured_output.retry_success",
        ctx,
        node=node,
        schema=schema_name,
        attempt=attempt,
    )
