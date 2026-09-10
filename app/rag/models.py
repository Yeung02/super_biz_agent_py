"""RAG 内部稳定数据模型。

本文件是 ISSUE-017 的边界产物，只负责定义阶段 3A 需要复用的数据结构：
metadata、文档、chunk、检索 chunk、上下文包、内部 citation 和 no-answer 决策。
它不执行检索、不写 Milvus、不调用 LLM，也不改变任何 HTTP API 响应。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.errors import JsonValue, RagMetadataInvalidError

RagMetadataPrimitive: TypeAlias = str | int | float | bool | None
RagMetadataValue: TypeAlias = (
    RagMetadataPrimitive | list[RagMetadataPrimitive] | dict[str, RagMetadataPrimitive]
)
DEFAULT_TENANT_ID = "default"
DEFAULT_METADATA_VERSION = 1
MILVUS_PRIMARY_ID_MAX_LENGTH = 100
_LEGACY_ID_PREFIX = "legacy:"
_DOC_ID_PREFIX = "doc"
_CHUNK_ID_PREFIX = "chk"
_STABLE_HASH_LENGTH = 32
_CHUNK_INDEX_WIDTH = 6
_KNOWN_METADATA_KEYS = {
    "doc_id",
    "chunk_id",
    "chunk_index",
    "source_path",
    "file_name",
    "extension",
    "content_hash",
    "tenant_id",
    "version",
    "created_at",
    "updated_at",
    "_source",
    "_file_name",
    "_extension",
}
_SAFE_PATH_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_MULTI_SLASH_PATTERN = re.compile(r"/+")


@dataclass(frozen=True)
class _BasicMetadataFields:
    """从新旧 metadata 中解析出的基础字段。"""

    source_path: str
    tenant_id: str
    file_name: str
    chunk_index: int
    extension: str | None
    version: int


@dataclass(frozen=True)
class _IdentityMetadataFields:
    """迁移期可补齐的身份字段。"""

    doc_id: str
    chunk_id: str
    content_hash: str
    compat_warnings: list[str]


class LangChainDocumentLike(Protocol):
    """`langchain_core.documents.Document` 的最小结构协议。

    使用协议而不是直接绑定具体类，是为了让模型测试和后续 adapter 可以用轻量
    fake 对象验证转换逻辑，不需要连接真实 LangChain retriever 或向量库。
    """

    page_content: str
    metadata: Mapping[str, object]


class SearchResultLike(Protocol):
    """向量检索结果的最小结构协议。"""

    id: str
    content: str
    score: float
    metadata: Mapping[str, object]


class RagMetadata(BaseModel):
    """RAG chunk 的标准 metadata schema。

    旧系统只写 `_source/_file_name/_extension`，无法支撑 citation 和 eval。这里先把
    新旧字段收敛到统一模型；对缺失的 `doc_id/chunk_id/content_hash` 只做迁移期
    兼容补齐，并通过 `compat_warnings` 标记风险，真正稳定 ID 生成留给 ISSUE-018
    接入索引链路。
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="稳定文档 ID 或迁移期 legacy ID")
    chunk_id: str = Field(..., description="稳定 chunk ID 或迁移期 legacy ID")
    content_hash: str = Field(..., description="chunk 内容 hash")
    tenant_id: str = Field(DEFAULT_TENANT_ID, description="租户 ID")
    version: int = Field(DEFAULT_METADATA_VERSION, ge=1, description="metadata schema 版本")
    source_path: str = Field(..., description="知识库内来源路径")
    file_name: str = Field(..., description="来源文件名")
    chunk_index: int = Field(..., ge=0, description="文档内 chunk 顺序")
    extension: str | None = Field(default=None, description="文件扩展名")
    created_at: str | None = Field(default=None, description="入库创建时间")
    updated_at: str | None = Field(default=None, description="入库更新时间")
    metadata_extra: dict[str, RagMetadataValue] = Field(
        default_factory=dict,
        description="业务扩展元数据",
    )
    compat_warnings: list[str] = Field(default_factory=list, description="迁移期兼容提示")

    @field_validator(
        "doc_id",
        "chunk_id",
        "content_hash",
        "tenant_id",
        "source_path",
        "file_name",
    )
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("字段不能为空")
        return stripped

    @field_validator("extension")
    @classmethod
    def _normalize_extension(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return stripped if stripped.startswith(".") else f".{stripped}"

    @classmethod
    def from_mapping(
        cls,
        metadata: Mapping[str, object],
        *,
        content: str = "",
        fallback_chunk_id: str | None = None,
    ) -> Self:
        """从新旧 metadata 字典构造标准 metadata。

        旧 `_source` 数据仍然可能存在于 Milvus collection 中，模型层必须能读取它们，
        否则阶段 3A 后续迁移和删除逻辑无法渐进上线。这里的 legacy ID 仅用于让内部
        schema 保持完整和可追踪，不代表已经完成幂等入库。
        """

        basic_fields = _resolve_basic_metadata_fields(metadata)
        identity_fields = _resolve_identity_metadata_fields(
            metadata,
            basic_fields=basic_fields,
            content=content,
            fallback_chunk_id=fallback_chunk_id,
        )

        return cls(
            doc_id=identity_fields.doc_id,
            chunk_id=identity_fields.chunk_id,
            content_hash=identity_fields.content_hash,
            tenant_id=basic_fields.tenant_id,
            version=basic_fields.version,
            source_path=basic_fields.source_path,
            file_name=basic_fields.file_name,
            chunk_index=basic_fields.chunk_index,
            extension=basic_fields.extension,
            created_at=_metadata_text(metadata, "created_at"),
            updated_at=_metadata_text(metadata, "updated_at"),
            metadata_extra=_metadata_extra(metadata),
            compat_warnings=identity_fields.compat_warnings,
        )

    def validate_metadata(self) -> Self:
        """返回已校验的 metadata 实例，供后续模块形成统一调用入口。"""

        return self

    def to_trace_fields(self) -> dict[str, JsonValue]:
        """生成安全 trace 字段。

        trace 只记录可定位问题的 ID、路径和版本，不包含完整 chunk 内容或任意扩展
        metadata，避免日志里泄露知识库正文、绝对路径或未来工具 payload。
        """

        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "tenant_id": self.tenant_id,
            "metadata_version": self.version,
            "content_hash": self.content_hash,
        }


class DocumentRecord(BaseModel):
    """文档级记录，用于后续幂等索引、删除和 eval 聚合。"""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    tenant_id: str = DEFAULT_TENANT_ID
    source_path: str
    file_name: str
    extension: str | None = None
    version: int = Field(DEFAULT_METADATA_VERSION, ge=1)
    content_hash: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    metadata_extra: dict[str, RagMetadataValue] = Field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: ChunkRecord) -> Self:
        """从 chunk 派生文档记录。

        ISSUE-017 不负责统计真实 chunk_count；这里默认保留为 0，后续索引服务在掌握
        完整文档分片列表时再写入准确数量。
        """

        metadata = chunk.metadata
        return cls(
            doc_id=metadata.doc_id,
            tenant_id=metadata.tenant_id,
            source_path=metadata.source_path,
            file_name=metadata.file_name,
            extension=metadata.extension,
            version=metadata.version,
            content_hash=metadata.content_hash,
            metadata_extra=metadata.metadata_extra,
        )

    def to_trace_fields(self) -> dict[str, JsonValue]:
        """生成文档级 trace 字段，不包含正文或扩展 metadata。"""

        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "metadata_version": self.version,
            "chunk_count": self.chunk_count,
        }


class ChunkRecord(BaseModel):
    """RAG 文档分片记录。"""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="chunk 正文，仅供内部上下文构建使用")
    metadata: RagMetadata = Field(..., description="标准化 metadata")

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def content_hash(self) -> str:
        return self.metadata.content_hash

    @property
    def tenant_id(self) -> str:
        return self.metadata.tenant_id

    @property
    def source_path(self) -> str:
        return self.metadata.source_path

    @property
    def file_name(self) -> str:
        return self.metadata.file_name

    @property
    def chunk_index(self) -> int:
        return self.metadata.chunk_index

    @property
    def extension(self) -> str | None:
        return self.metadata.extension

    @classmethod
    def from_langchain_document(
        cls,
        doc: LangChainDocumentLike,
        *,
        fallback_chunk_id: str | None = None,
    ) -> Self:
        """从 LangChain Document 转为内部 chunk。

        该方法只读取 `page_content` 和 `metadata`，不依赖 retriever 类型。这样后续
        LangChain Document、Milvus SearchResult 或测试 fake 都可以通过薄 adapter
        复用同一个模型边界。
        """

        metadata = RagMetadata.from_mapping(
            doc.metadata,
            content=doc.page_content,
            fallback_chunk_id=fallback_chunk_id,
        )
        return cls(content=doc.page_content, metadata=metadata)

    def to_trace_fields(self) -> dict[str, JsonValue]:
        """生成 chunk trace 字段，不记录完整正文。"""

        return self.metadata.to_trace_fields()


class RetrievedChunk(ChunkRecord):
    """检索返回的 chunk。

    `raw_score` 保留向量库原始语义，`normalized_score` 才能被 API citation 当作用户
    可见相关性分数。ISSUE-017 只定义字段和校验，真正过滤、排序和 no-answer 判定
    留给后续 RagRetriever/ContextBuilder。
    """

    raw_score: float | None = Field(default=None, description="向量库原始分数或距离")
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0, description="归一化分数")
    metric: str | None = Field(default=None, description="原始分数 metric，例如 L2")
    success: bool = Field(default=True, description="该 chunk 是否可作为证据")

    @model_validator(mode="after")
    def _validate_score_semantics(self) -> Self:
        if self.raw_score is not None and not self.metric:
            raise ValueError("raw_score 存在时必须提供 metric，避免误把 L2 距离当相似度")
        return self

    @classmethod
    def from_search_result(
        cls,
        result: SearchResultLike,
        *,
        metric: str = "L2",
        normalized_score: float | None = None,
    ) -> Self:
        """从向量检索结果转为内部检索 chunk。

        默认 metric 为当前系统实际使用的 L2，但调用方仍可显式传入，避免后续替换
        向量库或检索策略时出现分数语义漂移。
        """

        metadata = RagMetadata.from_mapping(
            result.metadata,
            content=result.content,
            fallback_chunk_id=result.id,
        )
        return cls(
            content=result.content,
            metadata=metadata,
            raw_score=result.score,
            normalized_score=normalized_score,
            metric=metric,
        )

    def to_trace_fields(self) -> dict[str, JsonValue]:
        trace_fields = super().to_trace_fields()
        trace_fields.update(
            {
                "raw_score": self.raw_score,
                "normalized_score": self.normalized_score,
                "metric": self.metric,
                "success": self.success,
            }
        )
        return trace_fields


class Citation(BaseModel):
    """内部 citation schema。

    内部 citation 可以保留 raw_score、span、evidence_text 和 metadata 供 trace/eval 使用；
    `to_api_citation` 会投影为 `docs/api_contract.md` 允许的安全字段。
    """

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    doc_id: str
    chunk_id: str
    source_path: str
    file_name: str
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = None
    metric: str | None = None
    evidence_text: str = ""
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    metadata: dict[str, RagMetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_span_order(self) -> Self:
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ValueError("span_end 不能小于 span_start")
        return self

    def to_api_citation(self, *, preview_chars: int = 200) -> dict[str, JsonValue]:
        """转换为 API 契约允许的 citation 字段。

        这里会拦截绝对路径和 `..` 目录逃逸，原因是 citation 面向用户展示，不能把
        本机路径、内部目录结构、raw metadata 或完整 chunk 正文带出服务边界。
        """

        safe_source_path = _safe_relative_source_path(self.source_path)
        body: dict[str, JsonValue] = {
            "citation_id": self.citation_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_path": safe_source_path,
            "file_name": self.file_name,
        }
        if self.normalized_score is not None:
            body["score"] = self.normalized_score
        if self.evidence_text:
            body["content_preview"] = self.evidence_text[:preview_chars]
        return body

    def to_trace_fields(self) -> dict[str, JsonValue]:
        """生成 citation trace 字段，不包含完整 evidence_text 或 raw metadata。"""

        return {
            "citation_id": self.citation_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "metric": self.metric,
            "span_start": self.span_start,
            "span_end": self.span_end,
        }


class NoAnswerDecision(BaseModel):
    """RAG no-answer 决策。

    它只表达“是否有足够证据回答”和安全文案，不生成最终回答，也不决定 fallback。
    这样 ContextBuilder、FallbackManager 和 API adapter 可以各自保持薄边界。
    """

    model_config = ConfigDict(extra="forbid")

    should_answer: bool
    reason_code: str | None = None
    safe_message: str
    evidence_count: int = Field(default=0, ge=0)
    metadata: dict[str, RagMetadataValue] = Field(default_factory=dict)

    def to_trace_fields(self) -> dict[str, JsonValue]:
        return {
            "no_answer": not self.should_answer,
            "no_answer_reason": self.reason_code,
            "evidence_count": self.evidence_count,
        }


class RagContext(BaseModel):
    """已打包的 RAG 上下文结构。"""

    model_config = ConfigDict(extra="forbid")

    context_text: str
    used_chunks: list[RetrievedChunk] = Field(default_factory=list)
    dropped_chunks: list[RetrievedChunk] = Field(default_factory=list)
    anchors: dict[str, str] = Field(
        default_factory=dict,
        description="内部引用锚点到 chunk_id 的映射，例如 {'[C1]': 'doc#000001'}",
    )
    citations: list[Citation] = Field(default_factory=list)
    no_answer_decision: NoAnswerDecision | None = None

    def to_trace_fields(self) -> dict[str, JsonValue]:
        decision = self.no_answer_decision
        return {
            "used_chunk_count": len(self.used_chunks),
            "dropped_chunk_count": len(self.dropped_chunks),
            "citation_count": len(self.citations),
            "no_answer": bool(decision and not decision.should_answer),
            "no_answer_reason": decision.reason_code if decision else None,
        }


def build_stable_rag_metadata(
    *,
    tenant_id: str,
    source_path: str,
    chunk_index: int,
    chunk_text: str,
    document_content: str | None = None,
    source_root: str | Path | None = None,
) -> dict[str, RagMetadataValue]:
    """生成 ISSUE-018 要求的稳定 RAG metadata。

    这里故意只依赖租户、规范化相对路径、文档内容和 chunk 顺序这些可重复输入，
    不读取 Milvus、时间戳或 UUID。这样同一文件重复索引能得到相同逻辑 ID，而内容
    变化只通过 `content_hash` 表达版本差异，后续 ISSUE-019 再负责删除旧 chunk 和
    幂等 upsert。
    """

    normalized_tenant_id = _normalize_tenant_id(tenant_id)
    normalized_source_path = normalize_rag_source_path(source_path, source_root=source_root)
    # Keep the parameter for call-site compatibility; 3A content_hash is chunk-scoped.
    _ = document_content
    content_hash = generate_content_hash(chunk_text)
    doc_id = generate_doc_id(
        tenant_id=normalized_tenant_id,
        normalized_source_path=normalized_source_path,
    )
    chunk_id = generate_chunk_id(doc_id=doc_id, chunk_index=chunk_index)
    file_name = PurePosixPath(normalized_source_path).name
    extension = PurePosixPath(file_name).suffix or None
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "tenant_id": normalized_tenant_id,
        "version": DEFAULT_METADATA_VERSION,
        "source_path": normalized_source_path,
        "file_name": file_name,
        "chunk_index": chunk_index,
        "extension": extension,
    }


def normalize_rag_source_path(
    source_path: str | Path,
    *,
    source_root: str | Path | None = None,
) -> str:
    """把 RAG source_path 规范为安全、稳定、相对的 POSIX 路径。

    citation 和 eval 都不能依赖本机绝对路径；同时 Windows 与 POSIX 分隔符、大小写
    差异会让同一文件生成不同 ID。本函数把路径统一为小写 POSIX 相对路径，并拒绝
    绝对路径越界、空路径和 `..` 目录逃逸。若传入 `source_root`，绝对路径会先解析
    到该根目录内的相对路径；这正是目录索引通过 allowlist 后的稳定边界。
    """

    raw_text = str(source_path).strip()
    if not raw_text:
        raise RagMetadataInvalidError(
            internal_message="RAG source_path is empty",
            details={"field": "source_path"},
        )

    if source_root is not None:
        raw_text = _relative_path_from_root(raw_text, source_root)

    normalized = _normalize_relative_path_text(raw_text)
    return normalized.casefold()


def generate_content_hash(content: str) -> str:
    """生成稳定 content_hash。

    只规范 Unicode 组合形式和换行符，不裁剪首尾空白。这样 Windows/Unix 换行不会导致
    不必要的版本变化，但正文空格、段落或文字变更仍会改变 hash，便于后续索引和 eval
    判断文档版本。
    """

    return _sha256_text(_normalize_content_for_hash(content))


def generate_doc_id(*, tenant_id: str, normalized_source_path: str) -> str:
    """按 tenant 与安全相对路径生成 Milvus 可接受的稳定文档 ID。"""

    normalized_tenant_id = _normalize_tenant_id(tenant_id)
    safe_source_path = _normalize_relative_path_text(normalized_source_path).casefold()
    return _bounded_stable_id(_DOC_ID_PREFIX, f"{normalized_tenant_id}\0{safe_source_path}")


def generate_chunk_id(*, doc_id: str, chunk_index: int) -> str:
    """按文档 ID 和 chunk 顺序生成稳定 chunk ID。

    常规路径使用 `doc_id#000001`，方便人工排查和后续 citation 锚定；如果未来 doc_id
    被配置成较长格式导致超过 Milvus `id` 字段 100 字符限制，则自动降级为短 hash，
    避免入库阶段才出现不可恢复错误。
    """

    if chunk_index < 0:
        raise RagMetadataInvalidError(
            internal_message="RAG chunk_index must be non-negative",
            details={"field": "chunk_index", "chunk_index": chunk_index},
        )
    candidate = f"{doc_id}#{chunk_index:0{_CHUNK_INDEX_WIDTH}d}"
    if len(candidate) <= MILVUS_PRIMARY_ID_MAX_LENGTH:
        return candidate
    return _bounded_stable_id(_CHUNK_ID_PREFIX, f"{doc_id}\0{chunk_index}")


def normalize_score(raw_score: float | None, metric: str | None) -> float | None:
    """按 metric 把原始检索分数转换到 `0.0-1.0`。

    当前系统 Milvus 使用 L2 distance，值越小越相关，因此不能直接输出给用户。这里
    提供模型层公共函数，供 retriever、citation 和 eval 复用同一套分数语义。未知
    metric 不在模型层猜测含义，而是抛出稳定内部错误；RagRetriever 会捕获它并使用
    0 分保守降级，同时写 trace warning，避免把未知距离/相似度误当高置信证据。
    """

    if raw_score is None:
        return None
    if metric is None:
        raise RagMetadataInvalidError(details={"missing_fields": ["metric"]})

    normalized_metric = metric.strip().upper()
    if normalized_metric in {"L2", "EUCLIDEAN"}:
        safe_distance = max(raw_score, 0.0)
        return 1.0 / (1.0 + safe_distance)
    if normalized_metric in {"COSINE", "IP", "INNER_PRODUCT"}:
        return min(max(raw_score, 0.0), 1.0)
    if normalized_metric == "RRF":
        # Milvus hybrid_search 的 RRF 融合分（越大越相似）。当前固定 dense+BM25 两路、
        # k=60，理论上限 2/(k+1)；与 VectorSearchService._hybrid_search 的归一化保持
        # 同一尺度，正常路径下该分支只是兜底（service 已直接给出 normalized_score）。
        theoretical_max = 2.0 / 61.0
        return min(max(raw_score / theoretical_max, 0.0), 1.0)

    raise RagMetadataInvalidError(
        internal_message=f"Unsupported retrieval metric: {normalized_metric}",
        details={"metric": normalized_metric},
    )


def _resolve_basic_metadata_fields(metadata: Mapping[str, object]) -> _BasicMetadataFields:
    source_path = _metadata_text(metadata, "source_path") or _metadata_text(metadata, "_source")
    if source_path is None:
        raise _metadata_error(["source_path"], metadata)

    tenant_id = _metadata_text(metadata, "tenant_id") or DEFAULT_TENANT_ID
    file_name = (
        _metadata_text(metadata, "file_name")
        or _metadata_text(metadata, "_file_name")
        or PurePosixPath(source_path.replace("\\", "/")).name
    )
    if not file_name:
        raise _metadata_error(["file_name"], metadata)

    chunk_index = _metadata_int(metadata, "chunk_index", default=0)
    extension = (
        _metadata_text(metadata, "extension")
        or _metadata_text(metadata, "_extension")
        or PurePosixPath(file_name).suffix
        or None
    )
    version = _metadata_int(metadata, "version", default=DEFAULT_METADATA_VERSION)
    return _BasicMetadataFields(
        source_path=source_path,
        tenant_id=tenant_id,
        file_name=file_name,
        chunk_index=chunk_index,
        extension=extension,
        version=version,
    )


def _resolve_identity_metadata_fields(
    metadata: Mapping[str, object],
    *,
    basic_fields: _BasicMetadataFields,
    content: str,
    fallback_chunk_id: str | None,
) -> _IdentityMetadataFields:
    content_hash = _metadata_text(metadata, "content_hash")
    doc_id = _metadata_text(metadata, "doc_id")
    chunk_id = _metadata_text(metadata, "chunk_id")
    compat_warnings: list[str] = []

    if _is_legacy_metadata(metadata):
        compat_warnings.append("legacy_metadata")

    if content_hash is None:
        # 迁移期读取旧数据时仍需要 content_hash 作为去重和 trace 锚点。这里使用
        # 与稳定 ID 生成一致的内容 hash 补齐模型字段；但不把 legacy 补齐值接入
        # 索引主键，避免把旧存量数据误判为已经完成 ISSUE-018 幂等迁移。
        content_hash = generate_content_hash(content)
        compat_warnings.append("content_hash_generated_from_content")

    if doc_id is None:
        doc_key = f"{basic_fields.tenant_id}:{basic_fields.source_path}"
        doc_id = f"{_LEGACY_ID_PREFIX}{_short_hash(doc_key)}"
        compat_warnings.append("doc_id_generated_for_legacy_metadata")

    if chunk_id is None:
        legacy_chunk_anchor = fallback_chunk_id or f"{doc_id}:{basic_fields.chunk_index:06d}"
        chunk_id = f"{_LEGACY_ID_PREFIX}{legacy_chunk_anchor}"
        compat_warnings.append("chunk_id_generated_for_legacy_metadata")

    return _IdentityMetadataFields(
        doc_id=doc_id,
        chunk_id=chunk_id,
        content_hash=content_hash,
        compat_warnings=compat_warnings,
    )


def _metadata_text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _metadata_int(metadata: Mapping[str, object], key: str, *, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise _metadata_error([key], metadata)
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as exc:
        raise _metadata_error([key], metadata) from exc
    if parsed < 0:
        raise _metadata_error([key], metadata)
    return parsed


def _metadata_extra(metadata: Mapping[str, object]) -> dict[str, RagMetadataValue]:
    return {
        key: _json_safe_value(value)
        for key, value in metadata.items()
        if key not in _KNOWN_METADATA_KEYS
    }


def _json_safe_value(value: object) -> RagMetadataValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe_primitive(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_primitive(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe_primitive(child_value) for key, child_value in value.items()}
    return str(value)


def _json_safe_primitive(value: object) -> RagMetadataPrimitive:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _is_legacy_metadata(metadata: Mapping[str, object]) -> bool:
    return any(key in metadata for key in ("_source", "_file_name", "_extension"))


def _metadata_error(
    missing_fields: list[str], metadata: Mapping[str, object]
) -> RagMetadataInvalidError:
    return RagMetadataInvalidError(
        internal_message="RAG metadata missing required fields",
        details={
            "missing_fields": missing_fields,
            "metadata_keys": sorted(str(key) for key in metadata.keys()),
        },
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str) -> str:
    return _sha256_text(value)[:24]


def _bounded_stable_id(prefix: str, value: str) -> str:
    stable_id = f"{prefix}_{_sha256_text(value)[:_STABLE_HASH_LENGTH]}"
    if len(stable_id) <= MILVUS_PRIMARY_ID_MAX_LENGTH:
        return stable_id
    return f"{prefix}_{_sha256_text(stable_id)[:_STABLE_HASH_LENGTH]}"


def _normalize_tenant_id(tenant_id: str) -> str:
    normalized = tenant_id.strip() if tenant_id else DEFAULT_TENANT_ID
    return normalized or DEFAULT_TENANT_ID


def _normalize_content_for_hash(content: str) -> str:
    normalized = unicodedata.normalize("NFC", content)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def _relative_path_from_root(source_path: str, source_root: str | Path) -> str:
    root = Path(source_root).resolve()
    raw_path = Path(source_path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        relative = candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        # 这里抛 RAG_METADATA_INVALID 而不是泄露原始绝对路径，是因为路径已进入 RAG
        # metadata 生成阶段；对外错误映射仍由上层 AppError envelope 负责。
        raise RagMetadataInvalidError(
            internal_message="RAG source_path is outside source_root",
            details={"field": "source_path"},
        ) from exc
    return relative.as_posix()


def _normalize_relative_path_text(source_path: str) -> str:
    normalized = _MULTI_SLASH_PATTERN.sub("/", source_path.strip().replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(source_path)
    if (
        not normalized
        or normalized == "."
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or _SAFE_PATH_DRIVE_PATTERN.match(source_path)
        or ".." in posix_path.parts
    ):
        raise RagMetadataInvalidError(
            internal_message="RAG source_path is not a safe relative path",
            details={"field": "source_path"},
        )
    return posix_path.as_posix()


def _safe_relative_source_path(source_path: str) -> str:
    normalized = source_path.strip().replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(source_path)
    if (
        not normalized
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or _SAFE_PATH_DRIVE_PATTERN.match(source_path)
        or ".." in posix_path.parts
    ):
        raise RagMetadataInvalidError(
            internal_message="Citation source_path is not a safe relative path",
            details={"field": "source_path"},
        )
    return normalized
