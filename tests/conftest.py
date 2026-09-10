"""pytest fake fixtures for the engineering test suite.

这些夹具只服务测试进程：它们把当前仓库根目录放到 import 优先级最前面，
并提供不依赖 Milvus、DashScope、MCP server 或网络的内存 fake。这样后续 issue
可以直接复用同一套边界对象，而不会在单元测试阶段误触发真实外部服务。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from app.core.request_context import RequestContext

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_TEXT = str(ROOT_DIR)
if ROOT_DIR_TEXT in sys.path:
    sys.path.remove(ROOT_DIR_TEXT)
sys.path.insert(0, ROOT_DIR_TEXT)

# 记忆存储底座：测试进程注入进程内 checkpointer（生产默认 redis 硬依赖）。
# 必须在任何 `app.config` 导入之前生效，pytest 先加载 conftest 再收集测试模块，
# 因此放在模块顶层即可保证先于测试模块 import app。
os.environ.setdefault("MEMORY_CHECKPOINTER", "memory")


class _NoopLogger:
    """测试环境缺少 loguru 时的最小 logger 替身。

    运行时代码仍然依赖真实 loguru；这里仅在当前 Python 环境未安装该依赖时注入 stub，
    使单元测试不会因为日志库缺失而在收集阶段失败。logger 方法吞掉消息是安全的，
    因为本 issue 只验证输入边界和 fake 夹具，不断言日志副作用。
    """

    def add(self, *values: object, **kwargs: object) -> int:
        _ = values, kwargs
        return 0

    def remove(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs

    def bind(self, **kwargs: object) -> _NoopLogger:
        _ = kwargs
        return self

    def opt(self, *values: object, **kwargs: object) -> _NoopLogger:
        _ = values, kwargs
        return self

    def info(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs

    def warning(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs

    def error(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs

    def debug(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs

    def exception(self, *values: object, **kwargs: object) -> None:
        _ = values, kwargs


if importlib.util.find_spec("loguru") is None:
    loguru_module = ModuleType("loguru")
    loguru_module.logger = _NoopLogger()
    sys.modules["loguru"] = loguru_module

if importlib.util.find_spec("sse_starlette") is None:
    from starlette.responses import StreamingResponse

    class _StubEventSourceResponse(StreamingResponse):
        """测试用 SSE 响应替身，避免单测因缺少 sse-starlette 依赖无法导入 API。"""

        def __init__(self, content: object, **kwargs: object) -> None:
            _ = kwargs
            super().__init__(_encode_sse_stream(content), media_type="text/event-stream")

    async def _encode_sse_stream(content: object) -> AsyncIterator[str]:
        if isinstance(content, AsyncIterable):
            async for item in content:
                yield _format_sse_item(item)
            return
        if isinstance(content, Iterable) and not isinstance(
            content,
            str | bytes | Mapping,
        ):
            for item in content:
                yield _format_sse_item(item)
            return
        yield _format_sse_item(content)

    def _format_sse_item(item: object) -> str:
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if not isinstance(item, Mapping):
            return str(item)

        event = item.get("event")
        data = item.get("data", "")
        lines: list[str] = []
        if event is not None:
            lines.append(f"event: {event}\n")
        for data_line in str(data).splitlines() or [""]:
            lines.append(f"data: {data_line}\n")
        lines.append("\n")
        return "".join(lines)

    sse_package = ModuleType("sse_starlette")
    sse_module = ModuleType("sse_starlette.sse")
    sse_module.EventSourceResponse = _StubEventSourceResponse
    sys.modules["sse_starlette"] = sse_package
    sys.modules["sse_starlette.sse"] = sse_module


@dataclass(frozen=True)
class FakeLLMResponse:
    """LLM fake 的稳定返回结构，避免测试依赖具体供应商 SDK 对象。"""

    content: str


class FakeLLM:
    """可配置的 LLM fake，用于覆盖正常、空响应、异常和超时路径。"""

    def __init__(self, *, mode: str = "normal", answer: str = "fake answer") -> None:
        self.mode = mode
        self.answer = answer
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeLLMResponse:
        self.calls.append(prompt)
        if self.mode == "empty":
            return FakeLLMResponse(content="")
        if self.mode == "error":
            raise RuntimeError("fake llm provider error")
        if self.mode == "timeout":
            raise TimeoutError("fake llm timeout")
        return FakeLLMResponse(content=self.answer)

    def invoke(self, prompt: str) -> FakeLLMResponse:
        self.calls.append(prompt)
        if self.mode == "empty":
            return FakeLLMResponse(content="")
        if self.mode == "error":
            raise RuntimeError("fake llm provider error")
        if self.mode == "timeout":
            raise TimeoutError("fake llm timeout")
        return FakeLLMResponse(content=self.answer)


@dataclass(frozen=True)
class FakeChunk:
    """检索 fake 的最小 chunk 结构，后续 RAG/ToolManager 测试可直接断言。"""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    source_path: str


class FakeRetriever:
    """不访问 Milvus 的检索 fake，覆盖空、低分、重复 chunk 和正常结果。"""

    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[FakeChunk]:
        self.queries.append(query)
        if self.mode == "empty":
            return []
        if self.mode == "low_score":
            return [
                FakeChunk("chunk-low", "doc-low", "low confidence", 0.05, "docs/low.md"),
            ]
        if self.mode == "duplicate":
            return [
                FakeChunk("chunk-1", "doc-1", "same content", 0.92, "docs/runbook.md"),
                FakeChunk("chunk-1", "doc-1", "same content", 0.85, "docs/runbook.md"),
            ]
        return [
            FakeChunk("chunk-1", "doc-1", "normal evidence", 0.91, "docs/runbook.md"),
        ]


@dataclass(frozen=True)
class FakeMCPResult:
    """MCP fake 结果，字段名贴近 CallToolResult 但不导入真实 MCP 依赖。"""

    content: str
    isError: bool = False


class FakeMCPTool:
    """可模拟成功、异常、timeout 和 isError 的 MCP tool fake。"""

    def __init__(self, *, mode: str = "success", payload: str = "fake tool result") -> None:
        self.mode = mode
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, arguments: dict[str, object]) -> FakeMCPResult:
        self.calls.append(dict(arguments))
        if self.mode == "error":
            raise RuntimeError("fake mcp error")
        if self.mode == "timeout":
            raise TimeoutError("fake mcp timeout")
        if self.mode == "is_error":
            return FakeMCPResult(content=self.payload, isError=True)
        return FakeMCPResult(content=self.payload)


class FakeLocalTool:
    """ToolManager 测试用本地工具 fake。

    阶段 1B 要求 fake local tool 能稳定覆盖 success/error/timeout/large_json。这里不用
    真实业务工具，是为了保证 ToolManager 边界测试不会碰 Milvus、DashScope 或 MCP server；
    error 模式故意包含密钥和内部 URL 形态文本，用来验证生产代码不会把原始异常暴露给用户。
    """

    def __init__(self, *, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.mode == "error":
            raise RuntimeError("raw secret token from http://internal.local")
        if self.mode == "timeout":
            await asyncio.sleep(1)
            return "late local evidence"
        if self.mode == "large_json":
            return {
                "summary": "large payload",
                "items": [{"index": index, "blob": "x" * 40} for index in range(40)],
            }
        return {
            "service": str(kwargs.get("service", "unknown")),
            "evidence": "healthy evidence",
        }


class FakeAgentGraph:
    """Agent 边界测试用图 fake。

    真实 LangGraph 需要 planner/executor/replanner 和外部工具。ISSUE-010 只验证
    recursion_limit 错误映射和 trace/request 字段能被测试锁住，因此 fake 直接输出
    与服务层同形的事件，避免把边界测试变成集成测试。
    """

    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.last_recursion_limit: int | None = None

    async def stream(
        self,
        *,
        session_id: str,
        recursion_limit: int,
        trace_id: str,
        request_id: str,
    ) -> AsyncIterator[dict[str, object]]:
        self.last_recursion_limit = recursion_limit
        if self.mode == "recursion_limit":
            yield {
                "type": "error",
                "error": {
                    "code": "AGENT_MAX_STEP_EXCEEDED",
                    "message": "任务步骤过多，已停止继续执行。",
                    "retryable": False,
                },
                "trace_id": trace_id,
                "request_id": request_id,
                "session_id": session_id,
            }
            return

        yield {
            "type": "plan",
            "plan": ["检查告警"],
            "trace_id": trace_id,
            "request_id": request_id,
            "session_id": session_id,
        }
        yield {
            "type": "complete",
            "response": "诊断完成",
            "trace_id": trace_id,
            "request_id": request_id,
            "session_id": session_id,
        }


class FakeSSEClient:
    """SSE adapter 测试用 fake client。

    测试只关心事件外形：旧前端仍读取 `event: message` 和 `data.type`，新契约要求
    start/error/done 携带 trace_id/request_id。fake client 统一补这些字段，避免每个
    测试复制 SSE 文本解析代码，也避免误用真实 HTTP 长连接。
    """

    def __init__(self, ctx: RequestContext) -> None:
        self.ctx = ctx

    def response_from_payloads(
        self,
        payloads: Iterable[Mapping[str, object]],
        *,
        mirror_complete_as_done: bool = False,
    ) -> _FakeSSEResponse:
        events: list[dict[str, str]] = []
        for payload in payloads:
            safe_payload = self._with_trace(payload)
            events.append(self._message_event(safe_payload))
            if mirror_complete_as_done and safe_payload.get("type") == "complete":
                done_payload = dict(safe_payload)
                done_payload["type"] = "done"
                events.append(self._message_event(done_payload))
        return _FakeSSEResponse(events)

    def response_from_error(
        self,
        *,
        code: str,
        message: str,
        raw_error: str,
    ) -> _FakeSSEResponse:
        """生成安全 error event。

        `raw_error` 只用于证明测试输入里确实有敏感原文；fake 和生产 adapter 一样不把它
        放进 payload，防止后续断言误以为泄漏原始异常是可接受行为。
        """

        _ = raw_error
        payload = self._with_trace(
            {
                "type": "error",
                "stage": "error",
                "message": message,
                "data": message,
                "error": {
                    "code": code,
                    "message": message,
                    "retryable": True,
                },
            }
        )
        return _FakeSSEResponse([self._message_event(payload)])

    async def collect(self, response: _FakeSSEResponse) -> list[dict[str, object]]:
        """收集 fake SSE response，返回事件名和 JSON payload。"""

        events: list[dict[str, object]] = []
        async for chunk in response.body_iterator:
            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            event_name = "message"
            data_lines: list[str] = []
            for line in text.strip().splitlines():
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            events.append(
                {
                    "event": event_name,
                    "payload": json.loads("\n".join(data_lines)),
                }
            )
        return events

    def _with_trace(self, payload: Mapping[str, object]) -> dict[str, object]:
        safe_payload = dict(payload)
        safe_payload["trace_id"] = self.ctx.trace_id
        safe_payload["request_id"] = self.ctx.request_id
        return safe_payload

    def _message_event(self, payload: Mapping[str, object]) -> dict[str, str]:
        return {
            "event": "message",
            "data": json.dumps(dict(payload), ensure_ascii=False),
        }


class _FakeSSEResponse:
    """最小 SSE response fake，只暴露测试收集器需要的 body_iterator。"""

    def __init__(self, events: Iterable[Mapping[str, str]]) -> None:
        self._events = [dict(event) for event in events]

    @property
    async def body_iterator(self) -> AsyncIterator[str]:
        for event in self._events:
            yield _format_fake_sse_item(event)


def _format_fake_sse_item(event: Mapping[str, str]) -> str:
    """把 fake SSE 事件格式化为 text/event-stream 片段。"""

    event_name = event.get("event", "message")
    data = event.get("data", "")
    lines = [f"event: {event_name}\n"]
    for data_line in data.splitlines() or [""]:
        lines.append(f"data: {data_line}\n")
    lines.append("\n")
    return "".join(lines)


class _DocumentLike(Protocol):
    """fake vector store 需要的最小文档协议。

    这里不直接在 conftest 顶层导入 LangChain Document，是为了让测试夹具保持轻量；
    只要对象有 `page_content` 和 `metadata`，就能模拟索引链路的写入和检索。
    """

    page_content: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _StoredVectorDocument:
    """内存向量库中的一条记录。"""

    id: str
    document: _DocumentLike


class FakeVectorStore:
    """内存向量库 fake，支持 add/delete/search 并保持幂等写入。

    ISSUE-022 的测试需要把该对象当作 `VectorStoreManager` 或 LangChain vector store
    的替身使用。它既不连接 Milvus，也不调用 DashScope；如果绑定了 `FakeEmbedding`，
    会在写入时调用 embedding fake，从而覆盖 embedding 失败如何进入失败文件列表。
    """

    def __init__(self, *, embedding: FakeEmbedding | None = None) -> None:
        self.embedding = embedding
        self.records: dict[str, _StoredVectorDocument] = {}
        self.add_calls: list[list[str]] = []
        self.delete_doc_id_calls: list[str] = []
        self.delete_source_calls: list[str] = []
        self.search_queries: list[str] = []

    @property
    def documents(self) -> list[_DocumentLike]:
        """按写入顺序返回当前文档。

        旧 fake 暴露 `documents` 列表；保留这个只读属性，避免后续测试扩展时破坏既有
        使用方式，同时让重复 chunk_id 写入自然表现为“最后一次版本覆盖旧版本”。
        """

        return [record.document for record in self.records.values()]

    def add_documents(
        self,
        documents: Iterable[_DocumentLike],
        *,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """写入文档并返回稳定主键。

        和生产 `VectorStoreManager.add_documents` 一样优先使用传入 ids，其次使用
        `metadata["chunk_id"]`。相同 id 再次写入会覆盖旧记录，模拟 doc-level delete
        或 upsert 后不应产生残留 chunk 的幂等语义。
        """

        document_list = list(documents)
        resolved_ids = self._resolve_document_ids(document_list, ids=ids)
        for document in document_list:
            if self.embedding is not None:
                try:
                    self.embedding.embed_query(document.page_content)
                except Exception as exc:
                    from app.core.errors import EmbeddingProviderError

                    # fake 也要走稳定 AppError，而不是把 RuntimeError 原文抛给测试。
                    # 这样目录索引测试可以断言失败文件语义，不需要依赖真实 DashScope。
                    raise EmbeddingProviderError(
                        internal_message="Fake embedding failed during add_documents",
                    ) from exc

        for document_id, document in zip(resolved_ids, document_list, strict=True):
            self.records[document_id] = _StoredVectorDocument(id=document_id, document=document)

        self.add_calls.append(resolved_ids)
        return resolved_ids

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 `metadata["doc_id"]` 删除文档，返回删除数量。"""

        self.delete_doc_id_calls.append(doc_id)
        return self._delete_where(lambda document: _metadata_text(document, "doc_id") == doc_id)

    def delete_by_source(self, file_path: str) -> int:
        """按旧 `_source` 或新 `source_path` 删除文档，兼容迁移期索引逻辑。"""

        self.delete_source_calls.append(file_path)
        return self._delete_where(
            lambda document: _metadata_text(document, "_source") == file_path
            or _metadata_text(document, "source_path") == file_path
        )

    def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
        """返回 doc 下已有 chunk 的 {chunk_id: content_hash}，支撑增量 diff 测试。"""

        chunk_hashes: dict[str, str] = {}
        for record in self.records.values():
            if _metadata_text(record.document, "doc_id") != doc_id:
                continue
            chunk_id = _metadata_text(record.document, "chunk_id")
            content_hash = _metadata_text(record.document, "content_hash")
            if chunk_id and content_hash:
                chunk_hashes[chunk_id] = content_hash
        return chunk_hashes

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """按主键精确删除 chunk，模拟 VectorStoreManager 增量删除语义。"""

        deleted_count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.records:
                del self.records[chunk_id]
                deleted_count += 1
        return deleted_count

    def search(self, query: str, *, top_k: int = 3) -> list[_DocumentLike]:
        """执行简单文本检索。

        该 fake 不模拟向量距离，只按正文和来源 metadata 的包含关系过滤；空 query 返回
        前 top_k 条记录。这样测试能覆盖索引数据流，不会把检索排序提前做成阶段 3B。
        """

        self.search_queries.append(query)
        normalized_query = query.strip().casefold()
        if top_k <= 0:
            return []

        candidates = self.documents
        if normalized_query:
            candidates = [
                document
                for document in candidates
                if normalized_query in document.page_content.casefold()
                or normalized_query in (_metadata_text(document, "source_path") or "").casefold()
                or normalized_query in (_metadata_text(document, "_source") or "").casefold()
            ]
        return candidates[:top_k]

    def _resolve_document_ids(
        self,
        documents: Sequence[_DocumentLike],
        *,
        ids: Sequence[str] | None,
    ) -> list[str]:
        if ids is not None:
            return [str(document_id) for document_id in ids]

        resolved_ids: list[str] = []
        for index, document in enumerate(documents):
            chunk_id = _metadata_text(document, "chunk_id")
            resolved_ids.append(chunk_id or f"fake-vector-id-{len(self.records) + index}")
        return resolved_ids

    def _delete_where(self, predicate: Callable[[_DocumentLike], bool]) -> int:
        ids_to_delete = [
            document_id
            for document_id, record in self.records.items()
            if predicate(record.document)
        ]
        for document_id in ids_to_delete:
            del self.records[document_id]
        return len(ids_to_delete)


def _metadata_text(document: _DocumentLike, key: str) -> str | None:
    value = document.metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class FakeEmbedding:
    """固定维度 embedding fake，用于后续索引测试避免调用 DashScope。"""

    def __init__(self, *, mode: str = "normal", dimensions: int = 4) -> None:
        self.mode = mode
        self.dimensions = dimensions
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.mode == "error":
            raise RuntimeError("fake embedding error")
        return [0.1 for _ in range(self.dimensions)]


class FakeReranker:
    """默认按分数降序重排；错误模式用于验证后续 reranker 回退。"""

    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode

    def rerank(self, chunks: list[FakeChunk]) -> list[FakeChunk]:
        if self.mode == "error":
            raise RuntimeError("fake reranker error")
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


@dataclass(frozen=True)
class FakeUsage:
    """阶段 2 编排层 usage fake，保持 `to_dict()` 与真实 TokenUsage 同形。

    Orchestrator 单测只需要确认 usage 被记录并携带稳定字段，不应该为了这个断言依赖
    真实模型供应商返回 usage，也不应该触发任何外部计费或网络调用。
    """

    model: str
    input_tokens: int
    output_tokens: int
    estimated: bool

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "estimated_cost": 0.0,
        }


class FakeUsageRecorder:
    """AgentOrchestrator 测试用 TokenBudgetManager 同形 fake。

    它只实现编排层会调用的最小方法：预算分配、token 估算和 usage 记录。这样阶段 2
    回归测试可以验证“是否经过 token/usage 边界”，但不会把测试绑定到真实 tokenizer、
    价格配置或后续 RAG 模型。
    """

    def __init__(self, *, budget: int = 512) -> None:
        self.budget = budget
        self.allocate_calls: list[dict[str, object]] = []
        self.usage_records: list[dict[str, object]] = []

    def allocate(
        self,
        scenario: str,
        model: str,
        ctx: RequestContext | None = None,
        *,
        current_input: str | None = None,
        system_prompt: str | None = None,
    ) -> int:
        self.allocate_calls.append(
            {
                "scenario": scenario,
                "model": model,
                "trace_id": ctx.trace_id if ctx is not None else None,
                "current_input": current_input,
                "system_prompt": system_prompt,
            }
        )
        return self.budget

    def estimate_tokens(self, value: object) -> SimpleNamespace:
        text = _fake_token_text(value)
        return SimpleNamespace(token_count=len(text))

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        ctx: RequestContext | None = None,
        estimated: bool = False,
    ) -> FakeUsage:
        _ = ctx
        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated": estimated,
        }
        self.usage_records.append(record)
        return FakeUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
        )


class FakeMemorySaver:
    """ConversationManager 回归测试用 MemorySaver fake。

    fake 支持 `get_tuple/get/delete_thread/save_turn` 四类门面调用，并可显式触发读取或清理
    异常。这样测试能覆盖 MemorySaver 边界，而不需要导入真实 LangGraph checkpoint 存储。
    """

    def __init__(
        self,
        checkpoint: object | None = None,
        *,
        tuple_mode: str = "checkpoint",
        raise_on_read: bool = False,
        raise_on_clear: bool = False,
        raise_on_save: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.tuple_mode = tuple_mode
        self.raise_on_read = raise_on_read
        self.raise_on_clear = raise_on_clear
        self.raise_on_save = raise_on_save
        self.read_configs: list[Mapping[str, object]] = []
        self.deleted_threads: list[str] = []
        self.saved_turns: list[dict[str, object]] = []

    def get_tuple(self, config: Mapping[str, object]) -> object | None:
        self.read_configs.append(dict(config))
        if self.raise_on_read:
            raise RuntimeError("raw checkpoint read failure should stay internal")
        if self.tuple_mode == "none":
            return None
        if self.tuple_mode == "tuple":
            return (self.checkpoint,)
        return self.checkpoint

    def get(self, config: Mapping[str, object]) -> object | None:
        self.read_configs.append(dict(config))
        if self.raise_on_read:
            raise RuntimeError("raw checkpoint read failure should stay internal")
        return self.checkpoint

    def delete_thread(self, thread_id: str) -> None:
        if self.raise_on_clear:
            raise RuntimeError("raw delete failure should stay internal")
        self.deleted_threads.append(thread_id)

    def save_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self.raise_on_save:
            raise RuntimeError("raw save failure should stay internal")
        self.saved_turns.append(
            {
                "session_id": session_id,
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "metadata": dict(metadata or {}),
            }
        )


class FakeSummarizer:
    """ConversationManager 回归测试用摘要 fake。

    真实 ConversationSummarizer 已有单测覆盖；这里的 fake 只用于验证 manager 在摘要失败时
    是否继续使用最近轮次、是否保留原始 checkpoint，以及是否没有把摘要能力变成主链路阻断点。
    """

    def __init__(self, *, mode: str = "success", summary: str = "fake safe summary") -> None:
        self.mode = mode
        self.summary = summary
        self.calls = 0
        self.last_turn_count = 0

    def summarize_if_needed(
        self,
        *,
        turns: Sequence[object],
        existing_summary: str | None,
        budget: object | None,
        ctx: RequestContext | None = None,
    ) -> SimpleNamespace:
        _ = budget, ctx
        self.calls += 1
        self.last_turn_count = len(turns)
        if self.mode == "disabled":
            return SimpleNamespace(
                triggered=False,
                trigger_reason=None,
                summary=existing_summary,
                metadata={},
                error_code=None,
            )
        if self.mode == "fail":
            return SimpleNamespace(
                triggered=True,
                trigger_reason="turn_count",
                summary=None,
                metadata={},
                error_code="LLM_PROVIDER_ERROR",
            )
        return SimpleNamespace(
            triggered=True,
            trigger_reason="turn_count",
            summary=self.summary,
            metadata={},
            error_code=None,
        )


class FakeIntegrationRagService:
    """API 集成测试用 RAG service fake。

    这个 fake 只实现 Chat handler 会触碰的旧 service 边界：普通问答、SSE 流、
    会话清理和历史查询。ISSUE-030 的目标是验证 FastAPI handler、middleware 与
    响应 schema，而不是重新测试 LangChain、Milvus 或 DashScope，因此这里必须把
    所有外部依赖收窄为内存行为；同时记录调用入参，便于断言大小写 alias 和
    sessionId/session_id 兼容逻辑真的穿过了 API 层。
    """

    def __init__(self) -> None:
        self.answer = "集成测试回答"
        self.query_calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []
        self.clear_calls: list[str] = []
        self.history_calls: list[str] = []
        self.stream_mode = "success"
        self.history: list[dict[str, str]] = [
            {"role": "user", "content": "CPU 怎么排查？"},
            {"role": "assistant", "content": "先看进程级 CPU。"},
        ]

    async def query(self, question: str, session_id: str) -> str:
        self.query_calls.append((question, session_id))
        return f"{self.answer}: {question}/{session_id}"

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncIterator[dict[str, object]]:
        self.stream_calls.append((question, session_id))
        if self.stream_mode == "error":
            yield {"type": "content", "data": "partial"}
            raise RuntimeError("raw token=sk-secret from http://internal.stream")
        yield {"type": "content", "data": "hello "}
        yield {"type": "content", "data": "world"}
        yield {"type": "complete", "data": {"answer": "hello world", "citations": []}}

    def clear_session(self, session_id: str, ctx: RequestContext | None = None) -> bool:
        _ = ctx
        self.clear_calls.append(session_id)
        return True

    def get_session_history(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> list[dict[str, str]]:
        _ = ctx
        self.history_calls.append(session_id)
        return list(self.history)


class FakeHistoryStore:
    """内存长期记忆库 fake，与 PostgreSQL ConversationHistoryStore 同形。

    集成测试通过它替换 `app.api.chat.conversation_history_store`，保证 HTTP 层
    测试不依赖真实 PG；`user_id` 分组与摘要 upsert 语义按生产实现的最小集合模拟。
    """

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[dict[str, str]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        user_id: str = "default",
    ) -> None:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id must not be empty")

        timestamp = datetime.now(UTC).isoformat()
        title = " ".join(user_message.split())[:30] or "New conversation"
        session = self.sessions.get(clean_session_id)
        if session is None:
            session = {
                "session_id": clean_session_id,
                "user_id": user_id,
                "title": title,
                "created_at": timestamp,
                "updated_at": timestamp,
                "message_count": 0,
            }
            self.sessions[clean_session_id] = session
            self.messages[clean_session_id] = []
        session["user_id"] = user_id
        session["updated_at"] = timestamp
        self.messages[clean_session_id].append(
            {"role": "user", "content": user_message, "timestamp": timestamp}
        )
        self.messages[clean_session_id].append(
            {"role": "assistant", "content": assistant_message, "timestamp": timestamp}
        )
        session["message_count"] += 2

    def list_sessions(self, limit: int = 100, *, user_id: str | None = None) -> list[dict[str, Any]]:
        safe_limit = min(max(int(limit), 1), 500)
        sessions = [dict(item) for item in self.sessions.values()]
        if user_id is not None:
            sessions = [item for item in sessions if item["user_id"] == user_id]
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)[:safe_limit]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return [dict(item) for item in self.messages.get(session_id, [])]

    def get_history_page(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        history = self.get_history(session_id)
        return {
            "total": len(history),
            "limit": limit,
            "offset": offset,
            "messages": history[offset : offset + limit],
        }

    def clear_session(self, session_id: str) -> bool:
        existed = session_id in self.sessions
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)
        self.summaries.pop(session_id, None)
        return existed

    def delete_sessions_stale(self, *, older_than: object) -> list[str]:
        return []

    def get_summary(self, session_id: str) -> dict[str, Any] | None:
        summary = self.summaries.get(session_id)
        return dict(summary) if summary is not None else None

    def save_summary(self, session_id: str, summary: str, *, source_message_count: int) -> None:
        self.summaries[session_id] = {
            "summary": summary,
            "source_message_count": source_message_count,
            "updated_at": datetime.now(UTC).isoformat(),
        }


AIOpsFakeMode = Literal["success", "done", "raise", "fallback"]


class FakeIntegrationAIOpsService:
    """API 集成测试用 AIOps service fake。

    ISSUE-031 只验证 `/api/aiops` 的 SSE adapter 契约，不重新执行真实 LangGraph、
    MCP 工具或模型调用。这个 fake 用固定事件流模拟旧 service 边界：正常 `complete`
    流、规范 `done` 流、中途异常以及可 fallback 的工具错误。这样测试能锁住
    `event: message`、`data.type`、trace/request 和错误脱敏，同时不会依赖外部服务。
    """

    def __init__(self) -> None:
        self.mode: AIOpsFakeMode = "success"
        self.calls: list[str] = []

    async def diagnose(self, session_id: str = "default") -> AsyncIterator[dict[str, object]]:
        """按测试场景输出 AIOps SSE payload。

        fake 事件刻意贴近旧 `AIOpsService.diagnose` 的 dict 形态，让 API handler 的
        兼容逻辑接受真实输入；异常场景包含密钥和内部 URL 形态文本，用来证明生产
        adapter 不会把原始异常全文发给用户。
        """

        self.calls.append(session_id)
        if self.mode == "fallback":
            yield {
                "type": "error",
                "stage": "tool_timeout",
                "message": "raw token=sk-secret from http://internal.aiops/tool",
                "error": {
                    "code": "TOOL_TIMEOUT",
                    "message": "工具调用超时，请稍后重试。",
                    "retryable": True,
                    "fallback_required": True,
                },
            }
            return

        yield {
            "type": "status",
            "stage": "fetching_alerts",
            "message": "正在获取告警信息",
            "session_id": session_id,
        }
        yield {
            "type": "plan",
            "stage": "plan_created",
            "message": "诊断计划已制定",
            "plan": ["检查告警", "生成报告"],
            "session_id": session_id,
        }
        if self.mode == "done":
            yield {
                "type": "done",
                "stage": "diagnosis_complete",
                "message": "诊断流程完成",
                "status": "completed",
                "session_id": session_id,
                "fallback_used": False,
            }
            return
        if self.mode == "raise":
            raise RuntimeError("raw token=sk-secret from http://internal.aiops/stream")

        yield {
            "type": "step_complete",
            "stage": "step_executed",
            "message": "步骤执行完成",
            "current_step": "检查告警",
            "remaining_steps": 1,
            "session_id": session_id,
        }
        yield {
            "type": "report",
            "stage": "final_report",
            "message": "最终报告已生成",
            "report": "# 诊断报告\n系统运行正常。",
            "session_id": session_id,
        }
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": "诊断流程完成",
            "diagnosis": {"status": "completed", "report": "系统运行正常。"},
            "session_id": session_id,
            "fallback_used": False,
        }


class FakeIntegrationMilvusManager:
    """API 集成测试用 Milvus lifecycle/health fake。

    TestClient 进入主应用 lifespan 时会调用 connect/close，`/api/health` 会调用
    health_check。这里把三者都放在同一个内存 fake 上，既验证真实 FastAPI 生命周期
    没有被跳过，也保证测试不会连接本机或 CI 中不存在的 Milvus。
    """

    def __init__(self, *, healthy: bool = True, raise_on_health: bool = False) -> None:
        self.healthy = healthy
        self.raise_on_health = raise_on_health
        self.connect_calls = 0
        self.close_calls = 0
        self.health_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def health_check(self) -> bool:
        self.health_calls += 1
        if self.raise_on_health:
            raise RuntimeError("raw password=secret from http://internal.milvus")
        return self.healthy


class FakeIntegrationIndexingResult:
    """目录索引 fake 的响应对象，保持和生产 `IndexingResult` 同形。

    File API 只依赖 `success/success_count/fail_count/error_code/to_dict()` 这些字段。
    在测试夹具里显式保留这些字段，是为了验证 HTTP schema 和兼容字段，不让测试因为
    真实 Milvus、DashScope 或生产分割器不可用而偏离 ISSUE-032 的范围。
    """

    def __init__(self, directory_path: str) -> None:
        self.success = False
        self.directory_path = directory_path
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.error_message = ""
        self.error_code = ""
        self.failed_files: dict[str, str] = {}
        self.failed_file_error_codes: dict[str, str] = {}
        self.status = "pending"
        self.partial_success = False
        self.indexed_doc_ids: list[str] = []

    def add_success(self, file_path: Path) -> None:
        """记录一个成功文件，doc_id 只需稳定可断言，不模拟真实 RAG metadata。"""

        self.success_count += 1
        doc_id = f"fake-doc-{file_path.stem}"
        if doc_id not in self.indexed_doc_ids:
            self.indexed_doc_ids.append(doc_id)

    def add_failure(self, file_path: Path, *, code: str, message: str) -> None:
        """记录单文件失败，保留旧 `failed_files` map 和新增错误码 map。"""

        resolved_path = str(file_path.resolve(strict=False))
        self.fail_count += 1
        self.failed_files[resolved_path] = message
        self.failed_file_error_codes[resolved_path] = code

    def finalize_status(self) -> None:
        """生成和生产索引结果一致的 success/status/partial_success 字段。"""

        self.partial_success = self.success_count > 0 and self.fail_count > 0
        if self.fail_count == 0:
            self.status = "success"
        elif self.partial_success:
            self.status = "partial_success"
        else:
            self.status = "failed"
        self.success = self.fail_count == 0
        if self.fail_count:
            self.error_message = next(iter(self.failed_files.values()), "")
        unique_codes = set(self.failed_file_error_codes.values())
        if self.status == "failed" and len(unique_codes) == 1:
            self.error_code = next(iter(unique_codes))

    def to_dict(self) -> dict[str, object]:
        """返回 File API 需要透出的目录索引兼容字段。"""

        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": 0,
            "error_message": self.error_message,
            "errorMessage": self.error_message,
            "error_code": self.error_code,
            "failed_files": self.failed_files,
            "failed_file_error_codes": self.failed_file_error_codes,
            "status": self.status,
            "partial_success": self.partial_success,
            "failed_file_count": len(self.failed_files),
            "indexed_doc_ids": self.indexed_doc_ids,
        }


class FakeIntegrationVectorIndexService:
    """上传/目录索引集成测试用向量索引 fake。

    真实 `VectorIndexService` 会继续深入分割器、embedding provider 和 Milvus。ISSUE-032
    只要求验证 FastAPI multipart/body/query、安全矩阵、trace 和新旧路径兼容，因此 fake
    在文件系统层模拟索引结果，并复用 `InputGuard.validate_index_file` 锁定目录内单文件
    的 UTF-8、大小和 symlink 语义，避免为了测试而绕过真实安全边界。
    """

    def __init__(self) -> None:
        self.single_file_calls: list[str] = []
        self.directory_calls: list[tuple[str, str | None]] = []
        self.single_file_mode = "success"

    def index_single_file(
        self,
        file_path: str,
        *,
        allowed_root: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> object:
        """记录单文件索引调用，可按测试场景模拟下游失败。"""

        _ = allowed_root, source_root
        self.single_file_calls.append(file_path)
        if self.single_file_mode == "embedding_error":
            from app.core.errors import EmbeddingProviderError

            raise EmbeddingProviderError(
                internal_message="Fake embedding provider failure for upload integration test",
            )
        return SimpleNamespace(doc_id=f"fake-doc-{Path(file_path).stem}", chunk_ids=["fake-chunk"])

    def index_directory(
        self,
        directory_path: str | None = None,
        *,
        allowed_root: str | Path | None = None,
    ) -> FakeIntegrationIndexingResult:
        """模拟目录索引，同时保留单文件失败列表。

        这里故意只遍历一层目录并只处理允许扩展名，贴近当前生产服务的旧行为。单个文件
        失败会进入 `failed_files`，而不会抛出并中断整个目录任务，确保部分成功契约可测。
        """

        from app.config import config
        from app.core.errors import AppError
        from app.core.input_guard import input_guard

        target_path = Path(directory_path or config.upload_dir).resolve()
        root_path = Path(allowed_root).resolve() if allowed_root is not None else target_path
        self.directory_calls.append((str(target_path), str(root_path)))
        result = FakeIntegrationIndexingResult(str(target_path))
        allowed_extensions = _fake_allowed_extensions(config.allowed_upload_extensions)
        files = sorted(
            (
                file_path
                for file_path in target_path.iterdir()
                if (file_path.is_file() or file_path.is_symlink())
                and file_path.suffix.casefold() in allowed_extensions
            ),
            key=lambda item: item.name,
        )
        result.total_files = len(files)

        for file_path in files:
            try:
                input_guard.validate_index_file(
                    file_path,
                    allowed_root=root_path,
                    allowed_extensions=tuple(allowed_extensions),
                    max_bytes=config.upload_max_bytes,
                )
                result.add_success(file_path)
            except AppError as exc:
                result.add_failure(file_path, code=exc.code, message=exc.user_message)

        result.finalize_status()
        return result


def _fake_allowed_extensions(extensions: Sequence[str]) -> set[str]:
    """规范化测试用扩展名配置，和生产服务保持 `.ext` 形式。"""

    normalized: set[str] = set()
    for extension in extensions:
        clean_extension = extension.strip().casefold()
        if not clean_extension:
            continue
        normalized.add(clean_extension if clean_extension.startswith(".") else f".{clean_extension}")
    return normalized


def _fake_token_text(value: object) -> str:
    """把 fake usage 输入转换为稳定文本，只用于测试估算而非生产 tokenizer。"""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return content if isinstance(content, str) else str(dict(value))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return "\n".join(_fake_token_text(item) for item in value)
    content_attr = getattr(value, "content", None)
    return content_attr if isinstance(content_attr, str) else str(value)


@pytest.fixture
def fake_request_context() -> RequestContext:
    """构造默认 anonymous/default 上下文，避免单测依赖 FastAPI middleware。"""

    from app.core.request_context import RequestContext

    now = time.time()
    return RequestContext(
        trace_id="trc_test",
        request_id="req_test",
        session_id="session-test",
        tenant_id="default",
        user_id="anonymous",
        deadline_ms=60_000,
        feature_flags=(),
        started_at=now,
        started_monotonic=time.monotonic(),
        method="POST",
        path="/unit-test",
        invalid_inbound_trace_header=False,
    )


@pytest.fixture
def tmp_upload_dir(tmp_path: Path) -> Path:
    """测试上传根目录，隔离真实 uploads，避免污染开发文件。"""

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def tmp_index_allowlist(tmp_path: Path) -> tuple[Path, ...]:
    """测试目录索引 allowlist，仅允许当前临时根目录。"""

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    return (allowed,)


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLM]:
    """返回 LLM fake 工厂，让测试按场景选择 mode。"""

    return lambda **kwargs: FakeLLM(**kwargs)


@pytest.fixture
def fake_retriever() -> Callable[..., FakeRetriever]:
    """返回 retriever fake 工厂，避免真实向量检索。"""

    return lambda **kwargs: FakeRetriever(**kwargs)


@pytest.fixture
def fake_mcp_tool() -> Callable[..., FakeMCPTool]:
    """返回 MCP tool fake 工厂，避免启动 MCP server。"""

    return lambda **kwargs: FakeMCPTool(**kwargs)


@pytest.fixture
def fake_local_tool() -> Callable[..., FakeLocalTool]:
    """返回本地工具 fake 工厂，覆盖 success/error/timeout/large_json 工具边界。"""

    return lambda **kwargs: FakeLocalTool(**kwargs)


@pytest.fixture
def fake_graph() -> Callable[..., FakeAgentGraph]:
    """返回 Agent graph fake 工厂，避免测试依赖真实 LangGraph 工作流。"""

    return lambda **kwargs: FakeAgentGraph(**kwargs)


@pytest.fixture
def fake_sse_client() -> Callable[[RequestContext], FakeSSEClient]:
    """返回 SSE fake client 工厂，用于锁定 message/type/trace 兼容契约。"""

    return lambda ctx: FakeSSEClient(ctx)


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """提供空的内存向量库 fake。"""

    return FakeVectorStore()


@pytest.fixture
def fake_embedding() -> Callable[..., FakeEmbedding]:
    """返回 embedding fake 工厂，避免调用 DashScope。"""

    return lambda **kwargs: FakeEmbedding(**kwargs)


@pytest.fixture
def fake_reranker() -> Callable[..., FakeReranker]:
    """返回 reranker fake 工厂，供后续 RAG 测试复用。"""

    return lambda **kwargs: FakeReranker(**kwargs)


@pytest.fixture
def fake_memory_saver() -> Callable[..., FakeMemorySaver]:
    """返回 MemorySaver fake 工厂，覆盖阶段 2 会话门面异常和回滚边界。"""

    return lambda **kwargs: FakeMemorySaver(**kwargs)


@pytest.fixture
def fake_summarizer() -> Callable[..., FakeSummarizer]:
    """返回摘要 fake 工厂，避免 ConversationManager 回归测试访问真实 LLM。"""

    return lambda **kwargs: FakeSummarizer(**kwargs)


@pytest.fixture
def fake_usage_recorder() -> Callable[..., FakeUsageRecorder]:
    """返回 usage 记录 fake 工厂，隔离真实 tokenizer、模型 usage 和成本估算。"""

    return lambda **kwargs: FakeUsageRecorder(**kwargs)


@pytest.fixture
def integration_rag_service() -> FakeIntegrationRagService:
    """提供 `/api/chat` 集成测试共用的内存 RAG service。"""

    return FakeIntegrationRagService()


@pytest.fixture
def integration_aiops_service() -> FakeIntegrationAIOpsService:
    """提供 `/api/aiops` 集成测试共用的内存 AIOps service。"""

    return FakeIntegrationAIOpsService()


@pytest.fixture
def integration_milvus_manager() -> FakeIntegrationMilvusManager:
    """提供 `/api/health` 和主应用 lifespan 共用的 Milvus fake。"""

    return FakeIntegrationMilvusManager()


@pytest.fixture
def integration_vector_index_service() -> FakeIntegrationVectorIndexService:
    """提供 `/api/upload` 和目录索引集成测试共用的向量索引 fake。"""

    return FakeIntegrationVectorIndexService()


@pytest.fixture
def integration_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    integration_rag_service: FakeIntegrationRagService,
    integration_milvus_manager: FakeIntegrationMilvusManager,
    integration_vector_index_service: FakeIntegrationVectorIndexService,
) -> Iterator[TestClient]:
    """返回不会访问外部服务的真实 FastAPI TestClient。

    ISSUE-030 要验证 handler、middleware、lifespan 和响应 schema 的集成行为，因此这里
    使用 `app.main.app`，而不是临时拼一个只含 router 的小 app。为了避免 lifespan
    连接真实 Milvus、Chat 调用真实 LLM/RAG，这个 fixture 在 TestClient 启动前替换
    对应模块级单例；同时临时关闭 trace/metrics 落盘，避免集成测试污染开发日志。
    """

    _install_integration_import_stubs(
        monkeypatch,
        integration_rag_service,
        integration_vector_index_service,
    )

    from app.api import chat as chat_api
    from app.api import file as file_api
    from app.api import health as health_api
    import app.main as app_main
    from app.main import app as main_app
    from app.main import config as main_config

    monkeypatch.setattr(chat_api, "rag_agent_service", integration_rag_service)
    monkeypatch.setattr(chat_api, "conversation_history_store", FakeHistoryStore())
    # 用户记忆抽取会真实调用 LLM；集成测试只验证 HTTP 契约，必须整体关闭。
    monkeypatch.setattr(chat_api.config, "user_memory_enabled", False, raising=False)
    monkeypatch.setattr(chat_api.config, "orchestrator_enabled", False, raising=False)
    monkeypatch.setattr(chat_api.config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(chat_api.config, "metrics_enabled", False, raising=False)
    monkeypatch.setattr(chat_api.config, "trace_jsonl_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(chat_api.config, "metrics_jsonl_path", str(tmp_path / "metrics.jsonl"))

    monkeypatch.setattr(file_api, "vector_index_service", integration_vector_index_service)
    monkeypatch.setattr(file_api.config, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(file_api.config, "index_allowed_directories", [str(tmp_path / "uploads")])
    monkeypatch.setattr(file_api.config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(file_api.config, "metrics_enabled", False, raising=False)
    Path(file_api.config.upload_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app_main, "milvus_manager", integration_milvus_manager)
    monkeypatch.setattr(health_api, "milvus_manager", integration_milvus_manager)
    monkeypatch.setattr(main_config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(main_config, "metrics_enabled", False, raising=False)
    monkeypatch.setattr(main_config, "trace_jsonl_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(main_config, "metrics_jsonl_path", str(tmp_path / "metrics.jsonl"))

    # Starlette 会缓存 middleware stack；测试逐例 monkeypatch 配置和单例后需要清空缓存，
    # 否则上一例创建的 middleware 可能继续持有旧的 trace/metrics 设置。
    main_app.middleware_stack = None
    with TestClient(main_app) as client:
        yield client


@pytest.fixture
def integration_aiops_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    integration_rag_service: FakeIntegrationRagService,
    integration_aiops_service: FakeIntegrationAIOpsService,
    integration_milvus_manager: FakeIntegrationMilvusManager,
) -> Iterator[TestClient]:
    """返回启用真实 AIOps router、但不访问外部服务的 FastAPI TestClient。

    ISSUE-030 的 `integration_client` 会把 AIOps router stub 成空模块；ISSUE-031
    需要真实 `/api/aiops` 路由来验证 SSE schema。因此这里使用独立 fixture，只 stub
    与当前测试无关的 file router 和 RAG service，并在 TestClient 启动前把
    `app.api.aiops.aiops_service` 替换为内存 fake。这样既不破坏旧 Chat/Health 测试，
    也能证明 AIOps API 的真实 adapter 层被执行。
    """

    _install_aiops_integration_import_stubs(monkeypatch, integration_rag_service)

    from app.api import aiops as aiops_api
    from app.api import health as health_api
    import app.main as app_main
    from app.main import app as main_app
    from app.main import config as main_config

    monkeypatch.setattr(aiops_api, "aiops_service", integration_aiops_service)
    monkeypatch.setattr(aiops_api.config, "user_memory_enabled", False, raising=False)
    monkeypatch.setattr(aiops_api.config, "orchestrator_enabled", False, raising=False)
    monkeypatch.setattr(aiops_api.config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(aiops_api.config, "metrics_enabled", False, raising=False)
    monkeypatch.setattr(aiops_api.config, "fallback_enabled", True, raising=False)
    monkeypatch.setattr(aiops_api.config, "trace_jsonl_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(aiops_api.config, "metrics_jsonl_path", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr(aiops_api._trace_logger, "enabled", False, raising=False)
    monkeypatch.setattr(aiops_api._fallback_manager.trace_logger, "enabled", False, raising=False)

    monkeypatch.setattr(app_main, "milvus_manager", integration_milvus_manager)
    monkeypatch.setattr(health_api, "milvus_manager", integration_milvus_manager)
    monkeypatch.setattr(main_config, "trace_enabled", False, raising=False)
    monkeypatch.setattr(main_config, "metrics_enabled", False, raising=False)
    monkeypatch.setattr(main_config, "fallback_enabled", True, raising=False)
    monkeypatch.setattr(main_config, "trace_jsonl_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(main_config, "metrics_jsonl_path", str(tmp_path / "metrics.jsonl"))

    # Starlette 会缓存 middleware stack；这里和通用 integration_client 一样逐例清空，
    # 避免前一个集成测试创建的 middleware 持有旧 trace/metrics 配置或旧 fake 单例。
    main_app.middleware_stack = None
    with TestClient(main_app) as client:
        yield client


def _install_integration_import_stubs(
    monkeypatch: pytest.MonkeyPatch,
    integration_rag_service: FakeIntegrationRagService,
    integration_vector_index_service: FakeIntegrationVectorIndexService,
) -> None:
    """为主应用导入安装与 ISSUE-030 无关的轻量 stub。

    `app.main` 会一次性导入 chat/file/aiops/health。AIOps 仍用空 router 替身，避免
    缺失的 LangChain/MCP 依赖在收集阶段抢先失败；File router 从 ISSUE-032 起必须
    真实挂载，因此只替换其下游 `vector_index_service`，让测试覆盖 HTTP 层和
    InputGuard，却不连接 Milvus、DashScope 或生产分割器。
    """

    from fastapi import APIRouter

    aiops_module = ModuleType("app.api.aiops")
    aiops_module.router = APIRouter()
    rag_module = ModuleType("app.services.rag_agent_service")
    rag_module.rag_agent_service = integration_rag_service
    vector_index_module = ModuleType("app.services.vector_index_service")
    vector_index_module.vector_index_service = integration_vector_index_service

    monkeypatch.setitem(sys.modules, "app.api.aiops", aiops_module)
    monkeypatch.setitem(sys.modules, "app.services.rag_agent_service", rag_module)
    monkeypatch.setitem(sys.modules, "app.services.vector_index_service", vector_index_module)
    sys.modules.pop("app.api.file", None)
    sys.modules.pop("app.main", None)

    api_package = sys.modules.get("app.api")
    if api_package is not None:
        for attr_name in ("file", "aiops"):
            if hasattr(api_package, attr_name):
                monkeypatch.delattr(api_package, attr_name, raising=False)


def _install_aiops_integration_import_stubs(
    monkeypatch: pytest.MonkeyPatch,
    integration_rag_service: FakeIntegrationRagService,
) -> None:
    """为 AIOps 集成测试安装轻量 import stub。

    与 `_install_integration_import_stubs` 的差异是：这里不能 stub `app.api.aiops`，
    因为 ISSUE-031 要测试真实 `/api/aiops` router。file router 仍然用空 router
    隔离上传/索引依赖；RAG service 仍然用 fake，避免 app.main 导入 chat router 时
    初始化真实 LangChain/Milvus/DashScope 链路。
    """

    from fastapi import APIRouter

    file_module = ModuleType("app.api.file")
    file_module.router = APIRouter()
    rag_module = ModuleType("app.services.rag_agent_service")
    rag_module.rag_agent_service = integration_rag_service

    monkeypatch.setitem(sys.modules, "app.api.file", file_module)
    monkeypatch.setitem(sys.modules, "app.services.rag_agent_service", rag_module)
    sys.modules.pop("app.main", None)
    sys.modules.pop("app.api.aiops", None)

    api_package = sys.modules.get("app.api")
    if api_package is not None:
        for attr_name in ("file", "aiops"):
            if hasattr(api_package, attr_name):
                monkeypatch.delattr(api_package, attr_name, raising=False)


@pytest.fixture
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """屏蔽测试中的 sleep，避免 timeout 分支拖慢单元测试。"""

    async def _async_noop_sleep(delay: float) -> None:
        _ = delay

    monkeypatch.setattr(asyncio, "sleep", _async_noop_sleep)
