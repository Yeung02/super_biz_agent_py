"""ToolManager 核心模型与工具包装。

ISSUE-006 提供独立可测的工具边界：把本地函数、LangChain BaseTool 和 MCP tool
统一包装为 ToolResult，并处理 timeout、异常、MCP isError、大结果裁剪和 trace 事件。
ISSUE-007 在调用真实工具前接入 ToolPolicy/PolicyRegistry；本模块仍不接入
RAG/AIOps 主流程、不做 retry 决策、Agent 调用次数上限或 fallback 文案，避免提前实现
ISSUE-008 及之后的内容。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from langchain_core.tools import StructuredTool

from app.agent.policies import PolicyDecision, PolicyRegistry
from app.config import config
from app.core.errors import (
    AgentMaxStepExceededError,
    AppError,
    JsonObject,
    JsonValue,
    ToolExecutionError,
    ToolTimeoutError,
)
from app.core.request_context import RequestContext, get_request_context_or_none
from app.observability.tracing import TraceLogger, TraceSpan

ToolStatus: TypeAlias = Literal["success", "error", "timeout", "unauthorized"]
ToolSource: TypeAlias = Literal["local", "langchain", "mcp"]
LocalToolCallable: TypeAlias = Callable[..., object]

_SAFE_TOOL_MESSAGES: dict[str, str] = {
    "AGENT_MAX_STEP_EXCEEDED": "任务步骤过多，已停止继续执行。",
    "TOOL_TIMEOUT": "工具调用超时，请稍后重试。",
    "TOOL_EXECUTION_ERROR": "工具调用失败。",
    "UNAUTHORIZED_TOOL": "工具不可用。",
}
_TRUNCATED_SUFFIX = "...[truncated]"
_LEGACY_KNOWLEDGE_ERROR_PREFIX = "知识检索工具暂时不可用"
_LEGACY_TIME_ERROR_PREFIX = "时间查询工具暂时不可用"
_LEGACY_TOOL_ERROR_PREFIXES = (
    _LEGACY_KNOWLEDGE_ERROR_PREFIX,
    _LEGACY_TIME_ERROR_PREFIX,
)
# 单请求证据块上限。正常请求受 agent_max_tool_calls 约束（默认 12），该上限只为
# 在工具预算关闭时防止证据表无界增长，不影响正常链路。
_MAX_EVIDENCE_BLOCKS_PER_REQUEST = 256


class _ToolErrorAccessor:
    """让 `ToolResult.error(...)` 和 `result.error` 同时可用。

    契约要求 `ToolResult.error` 是工厂方法，调用方又自然会读取 `result.error` 字段。
    普通 classmethod 和实例字段在 Python 中同名会互相覆盖，因此用 descriptor 按访问对象区分：
    类上返回工厂方法，实例上返回 error_info。
    """

    def __get__(self, instance: object, owner: type[ToolResult]) -> object:
        if instance is None:
            return owner._build_error
        return cast(ToolResult, instance).error_info


@dataclass(frozen=True)
class ToolErrorInfo:
    """工具错误的安全 envelope。

    这里刻意只保存稳定错误码和安全文案，不保存原始异常全文。原始异常可能带有堆栈、
    内部 URL 或密钥样式字段；这些信息只允许通过 AppError 的 internal_message 进入受控日志，
    不能从 ToolResult 再流入 Agent prompt 或最终回答。
    """

    code: str
    message: str
    retryable: bool
    fallback_required: bool

    @classmethod
    def from_app_error(cls, error: AppError) -> ToolErrorInfo:
        """从 AppError 提取可进入 ToolResult 的安全错误字段。"""

        return cls(
            code=error.code,
            message=_SAFE_TOOL_MESSAGES.get(error.code, error.user_message),
            retryable=error.retryable,
            fallback_required=error.fallback_required,
        )

    def to_dict(self) -> JsonObject:
        """转换为 JSON-safe dict，供 trace 和后续 prompt adapter 使用。"""

        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "fallback_required": self.fallback_required,
        }


@dataclass(frozen=True)
class ToolResult:
    """内部工具调用统一结果 envelope。

    `is_error/status/evidence_usable` 是防止工具错误污染事实证据链的核心字段。后续
    ContextBuilder 或 Agent adapter 应只消费 `is_evidence_usable()` 为 true 的结果；
    错误、超时和未授权结果只能用于诊断、fallback 或向用户说明工具不可用。
    """

    tool_name: str
    status: ToolStatus
    is_error: bool
    data: JsonValue | None = None
    error_info: ToolErrorInfo | None = None
    metadata: JsonObject = field(default_factory=dict)
    trimmed: bool = False
    raw_size: int = 0
    preview: str = ""
    latency_ms: float = 0.0

    @classmethod
    def success(
        cls,
        tool_name: str,
        data: object,
        *,
        metadata: Mapping[str, JsonValue] | None = None,
        latency_ms: float = 0.0,
        max_chars: int = 12_000,
    ) -> ToolResult:
        """创建成功结果，并在工具边界完成 JSON 化和大结果裁剪。

        工具返回值可能是 SDK 对象、Pydantic 模型、tuple 或大 JSON。这里先规范化为
        JsonValue，再按字符预算裁剪，避免后续 prompt 构造时才发现上下文被工具 payload 撑爆。
        """

        normalized = _to_json_value(data)
        trimmed_data, trimmed, raw_size, preview = _trim_json_value(normalized, max_chars)
        return cls(
            tool_name=tool_name,
            status="success",
            is_error=False,
            data=trimmed_data,
            metadata=dict(metadata or {}),
            trimmed=trimmed,
            raw_size=raw_size,
            preview=preview,
            latency_ms=latency_ms,
        )

    @classmethod
    def _build_error(
        cls,
        tool_name: str,
        *,
        code: str,
        message: str,
        retryable: bool,
        status: ToolStatus = "error",
        fallback_required: bool = True,
        metadata: Mapping[str, JsonValue] | None = None,
        latency_ms: float = 0.0,
    ) -> ToolResult:
        """创建错误结果，并确保错误文本不会被误判为事实数据。"""

        safe_message = _SAFE_TOOL_MESSAGES.get(code, message)
        error = ToolErrorInfo(
            code=code,
            message=safe_message,
            retryable=retryable,
            fallback_required=fallback_required,
        )
        preview = f"{code}: {safe_message}"
        return cls(
            tool_name=tool_name,
            status=status,
            is_error=True,
            data=None,
            error_info=error,
            metadata=dict(metadata or {}),
            trimmed=False,
            raw_size=len(preview),
            preview=preview,
            latency_ms=latency_ms,
        )

    error = _ToolErrorAccessor()

    @classmethod
    def from_mcp_result(
        cls,
        result: object,
        *,
        tool_name: str,
        metadata: Mapping[str, JsonValue] | None = None,
        latency_ms: float = 0.0,
        max_chars: int = 12_000,
    ) -> ToolResult:
        """把 MCP CallToolResult 或同形 fake 转换为 ToolResult。

        MCP 的 `isError=true` 是协议级失败信号。即使 content 中有文本，也只能作为诊断线索，
        不能作为事实证据，所以这里不把 content 放入 data，也不把原始错误文本写入 prompt block。
        """

        if _read_mcp_error_flag(result):
            safe_metadata = dict(metadata or {})
            safe_metadata["mcp_is_error"] = True
            return cls.error(
                tool_name,
                code="TOOL_EXECUTION_ERROR",
                message=_SAFE_TOOL_MESSAGES["TOOL_EXECUTION_ERROR"],
                retryable=True,
                metadata=safe_metadata,
                latency_ms=latency_ms,
            )
        return cls.success(
            tool_name,
            _extract_mcp_content(result),
            metadata=metadata,
            latency_ms=latency_ms,
            max_chars=max_chars,
        )

    def is_evidence_usable(self) -> bool:
        """判断该工具结果是否允许进入事实证据链。"""

        return not self.is_error and self.status == "success"

    def as_prompt_block(self) -> str:
        """转换为后续 Agent prompt adapter 可消费的安全文本块。

        成功结果只暴露裁剪后的 preview；失败结果只暴露稳定错误码和“不可作为事实依据”的限制，
        避免模型把工具错误文本当成业务事实引用。
        """

        if self.is_evidence_usable():
            return f"工具 {self.tool_name} 调用成功，结果摘要：{self.preview}"
        error_code = self.error.code if self.error else "TOOL_EXECUTION_ERROR"
        return (
            f"工具 {self.tool_name} 调用失败，错误码 {error_code}。"
            "该结果不能作为事实依据，只能用于说明工具不可用。"
        )

    def to_trace_fields(self) -> JsonObject:
        """输出 trace 需要的工具结果字段。"""

        error_code = self.error_info.code if self.error_info else None
        return {
            "tool_name": self.tool_name,
            "tool_result.status": self.status,
            "is_error": self.is_error,
            "evidence_usable": self.is_evidence_usable(),
            "trimmed": self.trimmed,
            "raw_size": self.raw_size,
            "preview_size": len(self.preview),
            "output_preview": self.preview,
            "error_code": error_code,
            "fallback_required": bool(self.error_info.fallback_required) if self.error_info else False,
            "latency_ms": self.latency_ms,
        }

    def to_dict(self) -> JsonObject:
        """序列化为 JSON-safe dict，供测试、trace 和后续适配层复用。"""

        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "is_error": self.is_error,
            "data": self.data,
            "error": self.error_info.to_dict() if self.error_info else None,
            "metadata": self.metadata,
            "trimmed": self.trimmed,
            "raw_size": self.raw_size,
            "preview": self.preview,
            "latency_ms": self.latency_ms,
            "evidence_usable": self.is_evidence_usable(),
        }


@dataclass(frozen=True)
class EvidenceBlock:
    """request-scoped 的工具证据摘要（ISSUE-A：Critic 证据链输入）。

    只保留工具名、可用性和 prompt-safe 文本，不携带完整 ToolResult.data。文本来自
    `ToolResult.as_prompt_block()`：失败结果是"不可用声明"，错误细节不会进入证据链。
    step 标注由 executor 在 drain 时补充，ToolManager 不感知计划步骤语义。
    """

    tool_name: str
    usable: bool
    text: str

    def to_dict(self) -> JsonObject:
        """转换为 JSON-safe dict，供 executor 写入 state 和后续 Critic 消费。"""

        return {
            "tool_name": self.tool_name,
            "usable": self.usable,
            "text": self.text,
        }


@dataclass(frozen=True)
class ToolCallSpec:
    """已注册工具的最小调用描述。

    timeout 和 max_result_chars 可以来自工具自身，也可以被 ISSUE-007 的 ToolPolicy
    覆盖；最终以 invoke 前 policy 决策为准，便于单个高风险工具使用更小的超时或裁剪预算。
    """

    name: str
    target: object
    source: ToolSource
    description: str | None = None
    timeout_seconds: float | None = None
    max_result_chars: int | None = None


class ToolRegistry:
    """内存工具注册表，负责名称到 ToolCallSpec 的稳定映射。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolCallSpec] = {}

    def register(self, spec: ToolCallSpec) -> ToolCallSpec:
        """注册工具并返回 spec，便于调用方链式使用。"""

        name = spec.name.strip()
        if not name:
            raise ValueError("tool name must not be empty")
        self._tools[name] = spec
        return spec

    def get(self, tool_name: str) -> ToolCallSpec:
        """读取已注册工具；未注册时抛 KeyError，由 ToolManager 转成安全错误结果。"""

        return self._tools[tool_name]

    def names(self) -> tuple[str, ...]:
        """返回当前注册的工具名快照。"""

        return tuple(self._tools.keys())


def _is_string_payload(data: JsonValue) -> bool:
    """本地基础工具的输出契约：返回纯文本字符串。"""

    return isinstance(data, str)


def _is_nonempty_mcp_payload(data: JsonValue) -> bool:
    """MCP 工具的默认输出校验：提取内容不能为空。

    MCP 输出形状由远端 server 决定，不做形状级硬编码；但 content 提取结果为空
    （None/空串/空列表）几乎必然是异常返回，不应作为事实证据进入后续链路。
    """

    if data is None:
        return False
    if isinstance(data, str):
        return data.strip() != ""
    if isinstance(data, list):
        return len(data) > 0
    return True


# 已知本地工具的默认输出校验器。MCP 工具输出形状由远端 server 决定，不在这里
# 硬编码；无显式校验器的 MCP 工具走 _is_nonempty_mcp_payload 默认校验，
# 后续新增工具需要形状级校验时通过 register_output_schema 显式注册。
_DEFAULT_OUTPUT_SCHEMAS: dict[str, Callable[[JsonValue], bool]] = {
    "get_current_time": _is_string_payload,
    "retrieve_knowledge": _is_string_payload,
}

# 进程级共享 ToolManager。RAG Agent 与 AIOps Executor 复用同一实例，避免 executor
# 每步执行都重建注册表和重复 wrap；请求级工具调用预算按 request_id 隔离，共享后
# 语义更准确（同一请求内两条链路的工具调用计入同一预算）。
_shared_tool_manager: ToolManager | None = None


def get_tool_manager() -> ToolManager:
    """获取进程级共享 ToolManager，并注册默认输出校验器。"""

    global _shared_tool_manager
    if _shared_tool_manager is None:
        _shared_tool_manager = ToolManager()
        if bool(getattr(config, "tool_output_schema_enabled", True)):
            for tool_name, validator in _DEFAULT_OUTPUT_SCHEMAS.items():
                _shared_tool_manager.register_output_schema(tool_name, validator)
    return _shared_tool_manager


def reset_shared_tool_manager() -> None:
    """重置共享单例（测试隔离用）。"""

    global _shared_tool_manager
    _shared_tool_manager = None


class ToolManager:
    """统一工具包装和调用入口。

    该类只处理单次工具调用的边界：policy、trace、timeout、retry（本地工具）、
    输出 schema 校验、结果 envelope、异常映射和裁剪。它不决定 Agent 是否应该调用
    某个工具，也不做 fallback 决策；fallback 由 FallbackManager 按矩阵统一执行。
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = None,
        trace_logger: TraceLogger | None = None,
        policy_registry: PolicyRegistry | None = None,
        retry_count: int | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(getattr(config, "tool_timeout_seconds", 15))
        )
        self.max_result_chars = (
            max_result_chars
            if max_result_chars is not None
            else int(getattr(config, "tool_max_result_chars", 12_000))
        )
        self.max_tool_calls_per_request = int(getattr(config, "agent_max_tool_calls", 12))
        # retry 统一收口：MCP 工具已由 client interceptor 按同一配置重试，这里只对
        # local/langchain 工具重试，避免双重重试放大延迟。
        self.retry_count = (
            retry_count
            if retry_count is not None
            else max(0, int(getattr(config, "tool_retry_count", 0)))
        )
        self.retry_delay_seconds = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else float(getattr(config, "tool_retry_delay_seconds", 1.0))
        )
        self._output_schemas: dict[str, Callable[[JsonValue], bool]] = {}
        # ToolManager 可能在 RAG service 中作为长生命周期对象复用，因此计数表不能依赖
        # 局部变量。key 使用 request_id，既能跨多个 wrapper 工具共享上限，又不会把
        # session_id 这种业务标识误用为并发请求边界。
        self._request_tool_call_counts: dict[str, int] = {}
        # ISSUE-A 证据链：按 request_id 收集工具证据摘要，executor 每步 drain 一次。
        # 与 _request_tool_call_counts 共用请求边界；RAG 等无人 drain 的请求由
        # _compact_request_evidence 按同一阈值清理，长生命周期下不会无界增长。
        self._request_evidence: dict[str, list[EvidenceBlock]] = {}
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=getattr(config, "trace_jsonl_path", "logs/trace.jsonl"),
            enabled=bool(getattr(config, "trace_enabled", True)),
        )
        self.policy_registry = policy_registry or PolicyRegistry.from_config(config)

    def register(self, spec: ToolCallSpec) -> ToolCallSpec:
        """注册一个 ToolCallSpec。"""

        return self.registry.register(spec)

    def register_output_schema(
        self,
        tool_name: str,
        validator: Callable[[JsonValue], bool],
    ) -> None:
        """注册单个工具的输出校验器。

        设计文档 9.4 节要求已知工具配置 output schema、未知工具至少校验可 JSON 序列化。
        为避免为每个工具硬编码 pydantic 模型，这里接受轻量校验函数：输入是成功结果的
        `data`（已 JSON 化），返回 False 即判定校验失败。校验失败会降级为
        `TOOL_EXECUTION_ERROR` 的不可用证据结果，不会把脏数据送进事实上下文。
        """

        normalized = tool_name.strip()
        if not normalized:
            raise ValueError("output schema tool name must not be empty")
        self._output_schemas[normalized] = validator

    def wrap_local_callable(
        self,
        name: str,
        func: LocalToolCallable,
        *,
        description: str | None = None,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = None,
    ) -> ToolCallSpec:
        """把普通 Python 函数包装为可注册工具。

        本地函数使用关键字参数调用，这与现有 `@tool` 函数的业务参数保持一致；如果调用方
        需要 LangChain 的 args_schema，后续 ISSUE-008 会在 adapter 层保留 schema。
        """

        return ToolCallSpec(
            name=name,
            target=func,
            source="local",
            description=description,
            timeout_seconds=timeout_seconds,
            max_result_chars=max_result_chars,
        )

    def wrap_local_tool(
        self,
        tool: object,
        *,
        name: str | None = None,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = None,
    ) -> ToolCallSpec:
        """把项目内置工具包装为 ToolCallSpec，并在 trace 中保留 `source=local`。

        当前本地工具是 LangChain `@tool` 产物，具备 `invoke/ainvoke/args_schema`，
        但从治理视角它们仍属于 local tool。单独提供这个入口，是为了 ISSUE-008
        接入后能区分本地知识/时间工具和远端 MCP 工具，同时不破坏原始参数 schema。
        """

        tool_name = name or _read_tool_name(tool)
        description = _read_optional_string_attr(tool, "description")
        return ToolCallSpec(
            name=tool_name,
            target=tool,
            source="local",
            description=description,
            timeout_seconds=timeout_seconds,
            max_result_chars=max_result_chars,
        )

    def wrap_langchain_tool(
        self,
        tool: object,
        *,
        name: str | None = None,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = None,
    ) -> ToolCallSpec:
        """把 LangChain BaseTool 同形对象包装为 ToolCallSpec。"""

        tool_name = name or _read_tool_name(tool)
        description = _read_optional_string_attr(tool, "description")
        return ToolCallSpec(
            name=tool_name,
            target=tool,
            source="langchain",
            description=description,
            timeout_seconds=timeout_seconds,
            max_result_chars=max_result_chars,
        )

    def wrap_mcp_tool(
        self,
        tool_or_name: object,
        tool: object | None = None,
        *,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = None,
    ) -> ToolCallSpec:
        """把 MCP tool 包装为 ToolCallSpec。

        为兼容测试 fake 和后续 MCP client 返回的真实 BaseTool，这里同时支持
        `wrap_mcp_tool(tool)` 和 `wrap_mcp_tool("name", tool)` 两种形态。
        """

        target = tool if tool is not None else tool_or_name
        name = (
            cast(str, tool_or_name)
            if isinstance(tool_or_name, str) and tool is not None
            else _read_tool_name(target)
        )
        description = _read_optional_string_attr(target, "description")
        return ToolCallSpec(
            name=name,
            target=target,
            source="mcp",
            description=description,
            timeout_seconds=timeout_seconds,
            max_result_chars=max_result_chars,
        )

    def to_langchain_tool(
        self,
        spec: ToolCallSpec,
        *,
        ctx: RequestContext | None = None,
    ) -> StructuredTool:
        """把 ToolCallSpec 适配成 LangChain Agent/ToolNode 可消费的工具。

        `bind_tools` 和 `ToolNode` 依赖 `name/description/args_schema`；wrapper 只替换
        执行入口，让真实调用进入 ToolManager。成功结果返回摘要，错误结果只返回稳定
        错误码和“不可作为事实依据”的说明，避免 MCP `isError=true` 或本地异常污染证据链。
        """

        registered_spec = self.register(spec)
        args_schema = _read_args_schema(registered_spec.target)
        response_format = _read_response_format(registered_spec.target)

        async def _async_adapter(**kwargs: object) -> object:
            result = await self.ainvoke(
                registered_spec.name,
                _json_mapping(kwargs),
                _resolve_tool_context(ctx),
            )
            return _to_langchain_payload(result, response_format=response_format)

        def _sync_adapter(**kwargs: object) -> object:
            result = self.invoke(
                registered_spec.name,
                _json_mapping(kwargs),
                _resolve_tool_context(ctx),
            )
            return _to_langchain_payload(result, response_format=response_format)

        return StructuredTool.from_function(
            func=_sync_adapter,
            coroutine=_async_adapter,
            name=registered_spec.name,
            description=registered_spec.description or f"{registered_spec.name} tool",
            args_schema=args_schema,
            infer_schema=args_schema is None,
            response_format=response_format,
            return_direct=bool(getattr(registered_spec.target, "return_direct", False)),
            metadata={
                "tool_manager_wrapped": True,
                "tool_source": registered_spec.source,
                "raw_tool_name": registered_spec.name,
            },
        )

    def invoke(
        self,
        tool_name: str,
        args: Mapping[str, JsonValue],
        ctx: RequestContext,
    ) -> ToolResult:
        """同步调用工具。

        若调用方已经处于事件循环内，不能安全地阻塞等待 async 工具，因此返回结构化错误结果。
        这比抛出 RuntimeError 更适合 Agent 边界：错误结果会被标记为不可用证据。
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(tool_name, args, ctx))
        return ToolResult.error(
            tool_name=tool_name,
            code="TOOL_EXECUTION_ERROR",
            message=_SAFE_TOOL_MESSAGES["TOOL_EXECUTION_ERROR"],
            retryable=True,
            metadata={"invoke_mode": "sync_inside_event_loop"},
        )

    async def ainvoke(
        self,
        tool_name: str,
        args: Mapping[str, JsonValue],
        ctx: RequestContext,
    ) -> ToolResult:
        """异步调用工具并返回 ToolResult。

        所有出口（成功、超时、异常、未注册、超预算、未授权）都在统一出口登记
        request-scoped 证据摘要，保证 Critic 证据链能看到每个工具的可用性声明。
        """

        result = await self._ainvoke_core(tool_name, args, ctx)
        self._record_request_evidence(ctx, result)
        return result

    async def _ainvoke_core(
        self,
        tool_name: str,
        args: Mapping[str, JsonValue],
        ctx: RequestContext,
    ) -> ToolResult:
        """工具调用主体；证据登记由 ainvoke 统一收口。"""

        limit_result = self._enforce_request_tool_limit(tool_name, ctx)
        if limit_result is not None:
            return limit_result

        try:
            spec = self.registry.get(tool_name)
        except KeyError:
            return ToolResult.error(
                tool_name=tool_name,
                code="TOOL_EXECUTION_ERROR",
                message=_SAFE_TOOL_MESSAGES["TOOL_EXECUTION_ERROR"],
                retryable=True,
                metadata={"reason": "tool_not_registered"},
            )

        policy_decision = self.policy_registry.check(spec.name, ctx)
        self._record_policy_decision(ctx, policy_decision)
        if not policy_decision.allowed:
            return self._unauthorized_result(spec.name, policy_decision)

        timeout_seconds = (
            policy_decision.timeout_ms / 1000
            if policy_decision.timeout_ms is not None
            else spec.timeout_seconds or self.timeout_seconds
        )
        max_chars = (
            policy_decision.max_result_chars
            if policy_decision.max_result_chars is not None
            else spec.max_result_chars or self.max_result_chars
        )
        input_preview = _preview_json(_to_json_value(dict(args)), 512)
        span = self.trace_logger.start_span(
            "tool",
            ctx,
            tool_name=spec.name,
            source=spec.source,
            input_preview=input_preview,
            timeout_seconds=timeout_seconds,
        )
        started = time.monotonic()

        try:
            raw_result = await self._call_tool_with_retry(
                spec,
                args,
                timeout_seconds=timeout_seconds,
                ctx=ctx,
            )
            latency_ms = _elapsed_ms(started)
            if _looks_like_legacy_tool_error(raw_result):
                result = ToolResult.error(
                    spec.name,
                    code="TOOL_EXECUTION_ERROR",
                    message=_SAFE_TOOL_MESSAGES["TOOL_EXECUTION_ERROR"],
                    retryable=True,
                    metadata={
                        "source": spec.source,
                        "legacy_error_payload": True,
                    },
                    latency_ms=latency_ms,
                )
            elif spec.source == "mcp" or _looks_like_mcp_result(raw_result):
                result = ToolResult.from_mcp_result(
                    raw_result,
                    tool_name=spec.name,
                    metadata={"source": spec.source},
                    latency_ms=latency_ms,
                    max_chars=max_chars,
                )
                # MCP 结果原先完全跳过输出校验；现在无显式校验器的 MCP 工具
                # 应用"内容非空"默认校验，空返回降级为不可用证据。
                result = self._validate_output_schema(
                    spec.name, result, source=spec.source
                )
            else:
                result = ToolResult.success(
                    spec.name,
                    raw_result,
                    metadata={"source": spec.source},
                    latency_ms=latency_ms,
                    max_chars=max_chars,
                )
                result = self._validate_output_schema(
                    spec.name, result, source=spec.source
                )
        except TimeoutError:
            result = self._result_from_app_error(
                spec.name,
                ToolTimeoutError(tool_name=spec.name),
                status="timeout",
                started=started,
                metadata={"source": spec.source},
            )
        except AppError as exc:
            result = self._result_from_app_error(
                spec.name,
                exc,
                status="error",
                started=started,
                metadata={"source": spec.source},
            )
        except Exception as exc:
            app_error = ToolExecutionError(
                internal_message=f"{exc.__class__.__name__}: {exc}",
                details={"tool_name": spec.name, "source": spec.source},
            )
            result = self._result_from_app_error(
                spec.name,
                app_error,
                status="error",
                started=started,
                metadata={"source": spec.source},
            )

        self._record_tool_result(span, result)
        return result

    def trim_result(self, result: ToolResult, budget: int | None = None) -> ToolResult:
        """重新按预算裁剪成功结果。

        这是给后续 TokenBudgetManager 预留的轻量入口；ISSUE-006 只在 ToolManager 内部使用字符
        预算，不提前实现 token 估算。
        """

        if result.is_error:
            return result
        max_chars = budget if budget is not None else self.max_result_chars
        return ToolResult.success(
            result.tool_name,
            result.data,
            metadata=result.metadata,
            latency_ms=result.latency_ms,
            max_chars=max_chars,
        )

    def validate_result_schema(self, result: ToolResult) -> bool:
        """确认 ToolResult 可被 JSON 序列化。"""

        json.dumps(result.to_dict(), ensure_ascii=False)
        return True

    def drain_request_evidence(self, request_id: str) -> list[EvidenceBlock]:
        """取走该请求自上次 drain 以来积累的证据块。

        取走（pop）语义与 state 的 operator.add 累加配合：executor 每步 drain 一次，
        保证 tool_evidence 中每个工具结果只出现一份；重复 drain 返回空列表。
        """

        return self._request_evidence.pop(request_id, [])

    def _record_request_evidence(self, ctx: RequestContext, result: ToolResult) -> None:
        """把 ToolResult 摘要登记进 request-scoped 证据表（ISSUE-A）。

        证据表是 Critic 链路的输入而非主路径依赖：登记只做长度上限保护，不抛异常，
        任何失败都不能影响工具调用本身。文本复用 as_prompt_block 的安全投影，
        失败结果天然是"不可用声明"，不会把错误细节送进证据链。
        """

        if not bool(getattr(config, "critic_evidence_enabled", True)):
            return
        blocks = self._request_evidence.setdefault(ctx.request_id, [])
        if len(blocks) >= _MAX_EVIDENCE_BLOCKS_PER_REQUEST:
            return
        max_chars = max(1, int(getattr(config, "critic_evidence_max_chars", 400)))
        blocks.append(
            EvidenceBlock(
                tool_name=result.tool_name,
                usable=result.is_evidence_usable(),
                text=_truncate_text(result.as_prompt_block(), max_chars),
            )
        )
        self._compact_request_evidence()

    def _compact_request_evidence(self) -> None:
        """限制证据表规模，语义与 _compact_tool_call_counts 一致。"""

        if len(self._request_evidence) > 10_000:
            self._request_evidence.clear()

    def _record_policy_decision(
        self,
        ctx: RequestContext,
        decision: PolicyDecision,
    ) -> None:
        """记录工具权限决策。

        这里不记录允许租户/用户列表，只记录当前请求身份和稳定 reason_code。这样排障时
        能区分“工具执行失败”和“权限拒绝”，同时不会把内部策略细节暴露到 ToolResult。
        """

        event_name = "tool.policy.allowed" if decision.allowed else "tool.policy.denied"
        self.trace_logger.record_event(
            event_name,
            ctx,
            status="ok" if decision.allowed else "error",
            error_code=None if decision.allowed else "UNAUTHORIZED_TOOL",
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            **decision.to_trace_fields(),
        )

    def _unauthorized_result(
        self,
        tool_name: str,
        decision: PolicyDecision,
    ) -> ToolResult:
        """把未授权决策转换为不可作为证据的 ToolResult。

        未授权不是 Python 异常，也不应该执行真实工具。返回结构化错误可以让后续 Agent
        adapter 明确知道工具不可用，但不会得到策略白名单、租户列表或用户列表等内部细节。
        """

        return ToolResult.error(
            tool_name=tool_name,
            code="UNAUTHORIZED_TOOL",
            message=_SAFE_TOOL_MESSAGES["UNAUTHORIZED_TOOL"],
            retryable=False,
            status="unauthorized",
            fallback_required=True,
            metadata={"policy_reason": decision.reason_code},
        )

    async def _call_tool(self, spec: ToolCallSpec, args: Mapping[str, JsonValue]) -> object:
        if spec.source == "local":
            # 项目内置工具既可能是普通 Python callable，也可能是 LangChain `@tool`
            # 包装后的 BaseTool。优先走 invoke/ainvoke，可以保留原始 schema 和
            # response_format；普通函数继续使用 ISSUE-006 的关键字参数调用路径。
            local_ainvoke = getattr(spec.target, "ainvoke", None)
            if callable(local_ainvoke):
                return await _await_if_needed(local_ainvoke(dict(args)))

            local_invoke = getattr(spec.target, "invoke", None)
            if callable(local_invoke):
                return await _await_if_needed(local_invoke(dict(args)))

            if callable(spec.target):
                local_callable = cast(LocalToolCallable, spec.target)
                return await _await_if_needed(local_callable(**dict(args)))

            raise ToolExecutionError(
                internal_message="registered local tool has no invoke interface",
                details={"tool_name": spec.name},
            )

        ainvoke = getattr(spec.target, "ainvoke", None)
        if callable(ainvoke):
            return await _await_if_needed(ainvoke(dict(args)))

        invoke = getattr(spec.target, "invoke", None)
        if callable(invoke):
            return await _await_if_needed(invoke(dict(args)))

        if callable(spec.target):
            return await _await_if_needed(cast(LocalToolCallable, spec.target)(**dict(args)))

        raise ToolExecutionError(
            internal_message="registered tool has no invoke interface",
            details={"tool_name": spec.name, "source": spec.source},
        )

    async def _call_tool_with_retry(
        self,
        spec: ToolCallSpec,
        args: Mapping[str, JsonValue],
        *,
        timeout_seconds: float,
        ctx: RequestContext,
    ) -> object:
        """带超时与本地重试的工具调用。

        重试策略与设计文档 9.4 节对齐：
        - `tool_retry_count` 是总尝试次数（含首次），0 表示只尝试一次；
        - 超时（TimeoutError）与 AppError 不重试：超时按次数放大延迟，AppError 已是
          内部边界的确定性失败；
        - `source=mcp` 不重试：MCP client interceptor 已按同一份配置重试过，
          ToolManager 再重试会形成双重重试；
        - 其余异常按指数退避重试，退避基数与 MCP interceptor 共用同一配置。
        """

        attempts = max(1, self.retry_count) if spec.source != "mcp" else 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    self._call_tool(spec, args),
                    timeout=timeout_seconds,
                )
            except (TimeoutError, AppError):
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                self.trace_logger.record_event(
                    "tool.retry",
                    ctx,
                    tool_name=spec.name,
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    status="error",
                    error_code="TOOL_EXECUTION_ERROR",
                )
                await asyncio.sleep(self.retry_delay_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def _validate_output_schema(
        self,
        tool_name: str,
        result: ToolResult,
        *,
        source: ToolSource | None = None,
    ) -> ToolResult:
        """对成功结果执行已注册的输出校验；失败降级为不可用证据。

        校验器自身抛异常也视为校验失败，避免有缺陷的校验器把脏数据放行进证据链。
        无显式校验器的 MCP 工具应用"内容非空"默认校验（tool_output_schema_mcp_enabled
        可独立回滚）；本地/langchain 工具维持旧行为，JSON 序列化已由结果规范化保证。
        """

        if result.is_error:
            return result
        if not bool(getattr(config, "tool_output_schema_enabled", True)):
            return result
        validator = self._output_schemas.get(tool_name)
        if validator is None:
            if source == "mcp" and bool(
                getattr(config, "tool_output_schema_mcp_enabled", True)
            ):
                validator = _is_nonempty_mcp_payload
            else:
                return result
        try:
            valid = bool(validator(result.data))
        except Exception:
            valid = False
        if valid:
            return result
        metadata = dict(result.metadata)
        metadata["reason"] = "output_schema_validation_failed"
        return ToolResult.error(
            tool_name,
            code="TOOL_EXECUTION_ERROR",
            message=_SAFE_TOOL_MESSAGES["TOOL_EXECUTION_ERROR"],
            retryable=False,
            metadata=metadata,
            latency_ms=result.latency_ms,
        )

    def _result_from_app_error(
        self,
        tool_name: str,
        app_error: AppError,
        *,
        status: ToolStatus,
        started: float,
        metadata: Mapping[str, JsonValue],
    ) -> ToolResult:
        error_info = ToolErrorInfo.from_app_error(app_error)
        return ToolResult.error(
            tool_name=tool_name,
            code=error_info.code,
            message=error_info.message,
            retryable=error_info.retryable,
            status=status,
            fallback_required=error_info.fallback_required,
            metadata=metadata,
            latency_ms=_elapsed_ms(started),
        )

    def _record_tool_result(self, span: TraceSpan, result: ToolResult) -> None:
        fields = result.to_trace_fields()
        if result.is_error:
            self.trace_logger.record_event(
                "tool.error",
                span.ctx,
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                status="error",
                **fields,
            )
        end_fields = dict(fields)
        # TraceLogger.end_span 会用 span 的单调时钟重新计算 latency_ms；这里移除 ToolResult
        # 的调用耗时，避免同名参数重复，同时保留 error event 中的工具级耗时。
        end_fields.pop("latency_ms", None)
        self.trace_logger.end_span(
            span,
            status="error" if result.is_error else "ok",
            **end_fields,
        )

    def _enforce_request_tool_limit(
        self,
        tool_name: str,
        ctx: RequestContext,
    ) -> ToolResult | None:
        """按 request_id 限制工具总调用次数。

        这个限制位于真实工具执行之前，解决 Agent 循环调用工具时可能拖垮下游服务的问题。
        超限结果使用 `AGENT_MAX_STEP_EXCEEDED`，并显式标记为不可作为证据；这样后续
        prompt adapter 只能说明“工具预算耗尽”，不会把超限文本包装成业务事实。
        """

        max_calls = self.max_tool_calls_per_request
        if max_calls <= 0:
            return None

        request_key = ctx.request_id
        current_count = self._request_tool_call_counts.get(request_key, 0) + 1
        self._request_tool_call_counts[request_key] = current_count
        self._compact_tool_call_counts()
        self.trace_logger.record_event(
            "agent.tool_call_count",
            ctx,
            tool_name=tool_name,
            tool_call_count=current_count,
            max_tool_calls=max_calls,
            status="error" if current_count > max_calls else "ok",
            error_code="AGENT_MAX_STEP_EXCEEDED" if current_count > max_calls else None,
        )
        if current_count <= max_calls:
            return None

        error = AgentMaxStepExceededError()
        return ToolResult.error(
            tool_name=tool_name,
            code=error.code,
            message=error.user_message,
            retryable=error.retryable,
            status="error",
            fallback_required=error.fallback_required,
            metadata={
                "tool_call_count": current_count,
                "max_tool_calls": max_calls,
            },
        )

    def _compact_tool_call_counts(self) -> None:
        """限制长生命周期 ToolManager 的计数表规模。

        当前阶段没有 request.end 回调可通知 ToolManager 清理指定 request_id；为避免
        长时间运行后字典无限增长，超过保守阈值时整体清空。清空只会放宽极少数边缘请求
        的工具预算，不会影响旧 API 字段或正常回答路径。
        """

        if len(self._request_tool_call_counts) > 10_000:
            self._request_tool_call_counts.clear()


def _elapsed_ms(started_monotonic: float) -> float:
    return round((time.monotonic() - started_monotonic) * 1000, 3)


async def _await_if_needed(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _to_json_value(value: object) -> JsonValue:
    """把工具原始返回值规范化为 JSON-safe 结构。"""

    if value is None or isinstance(value, str | int | float | bool):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list | set):
        return [_to_json_value(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_value(model_dump())

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _to_json_value(dict_method())

    return str(value)


def _trim_json_value(value: JsonValue, max_chars: int) -> tuple[JsonValue, bool, int, str]:
    max_size = max(1, max_chars)
    serialized = _serialize_json(value)
    raw_size = len(serialized)
    if raw_size <= max_size:
        return value, False, raw_size, serialized
    trimmed = _truncate_text(serialized, max_size)
    return trimmed, True, raw_size, trimmed


def _serialize_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _preview_json(value: JsonValue, max_chars: int) -> str:
    return _truncate_text(_serialize_json(value), max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    max_size = max(1, max_chars)
    if len(value) <= max_size:
        return value
    if max_size <= len(_TRUNCATED_SUFFIX):
        return value[:max_size]
    return value[: max_size - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX


def _read_tool_name(tool: object) -> str:
    name = _read_optional_string_attr(tool, "name")
    if name:
        return name
    return tool.__class__.__name__


def _read_optional_string_attr(value: object, attr_name: str) -> str | None:
    candidate = getattr(value, attr_name, None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def _read_args_schema(tool: object) -> object | None:
    return getattr(tool, "args_schema", None)


def _read_response_format(tool: object) -> Literal["content", "content_and_artifact"]:
    response_format = getattr(tool, "response_format", "content")
    return "content_and_artifact" if response_format == "content_and_artifact" else "content"


def _json_mapping(values: Mapping[str, object]) -> JsonObject:
    return {str(key): _to_json_value(value) for key, value in values.items()}


def _resolve_tool_context(explicit_ctx: RequestContext | None) -> RequestContext:
    """解析工具调用上下文。

    正常 HTTP/SSE 请求中 middleware 已把 RequestContext 放入 contextvar；wrapper 在
    工具实际执行时读取即可串联同一 trace。测试、脚本或旧路径可能没有 middleware，
    此时创建临时上下文，避免 trace 缺失直接中断原有 demo 调用。
    """

    if explicit_ctx is not None:
        return explicit_ctx
    current_ctx = get_request_context_or_none()
    if current_ctx is not None:
        return current_ctx
    return _build_direct_tool_context()


def _build_direct_tool_context() -> RequestContext:
    now = time.time()
    return RequestContext(
        trace_id=f"trc_tool_{uuid.uuid4().hex}",
        request_id=f"req_tool_{uuid.uuid4().hex}",
        session_id=None,
        tenant_id="default",
        user_id="anonymous",
        deadline_ms=int(getattr(config, "request_timeout_ms", 60_000)),
        feature_flags=(),
        started_at=now,
        started_monotonic=time.monotonic(),
        method="TOOL",
        path="/internal/tool",
        invalid_inbound_trace_header=False,
    )


def _to_langchain_payload(
    result: ToolResult,
    *,
    response_format: Literal["content", "content_and_artifact"],
) -> object:
    content = result.as_prompt_block()
    if response_format == "content_and_artifact":
        # 对原本使用 `content_and_artifact` 的工具，artifact 放 ToolResult envelope。
        # 这样 ToolNode 仍拿到合法二元组，同时错误 artifact 明确标记
        # `is_error/evidence_usable=false`，不会被后续链路误当作原始证据。
        return content, result.to_dict()
    return content


def _looks_like_legacy_tool_error(raw_result: object) -> bool:
    """识别旧工具用固定安全文本表达的失败结果。

    `retrieve_knowledge` 为了保留 `tool_manager_enabled=false` 的回滚路径，raw 模式仍
    返回 `(安全错误文案, [])`；ToolManager 路径必须把它重新升格为结构化错误，
    避免这段文案被当作知识库事实证据。
    """

    if isinstance(raw_result, str):
        return raw_result.startswith(_LEGACY_TOOL_ERROR_PREFIXES)
    if isinstance(raw_result, tuple | list) and raw_result:
        first_item = raw_result[0]
        return isinstance(first_item, str) and first_item.startswith(
            _LEGACY_TOOL_ERROR_PREFIXES
        )
    return False


def _looks_like_mcp_result(value: object) -> bool:
    return hasattr(value, "isError") or hasattr(value, "is_error")


def _read_mcp_error_flag(result: object) -> bool:
    is_error = getattr(result, "isError", getattr(result, "is_error", False))
    return bool(is_error)


def _extract_mcp_content(result: object) -> JsonValue:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return _to_json_value(content)
    if isinstance(content, tuple | list):
        parts: list[JsonValue] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(_to_json_value(item))
        return parts
    return _to_json_value(content)
