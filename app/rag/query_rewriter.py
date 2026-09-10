"""RAG 查询改写与多路召回变体生成。

阶段 3C 补齐 ISSUE-023 留下的 rewrite 扩展点：`LlmQueryRewriter` 通过一次 LLM 调用
同时生成"改写主 query + N 个检索变体"，供 RagRetriever 做多路召回。改写失败时返回
带 error_code 的空结果（fail-open），由 retriever 回退原始 query；改写能力不能成为
基础检索链路的阻塞点，这与 reranker 的回退策略保持一致。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import config
from app.core.request_context import RequestContext

_MAX_VARIANT_CHARS = 256
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\s*[\.、)．]\s*")


class RewriterLLM(Protocol):
    """LlmQueryRewriter 需要的最小同步 LLM 接口。

    真实 LLM（ChatOpenAI/ChatQwen）和测试 fake 都只需要实现 `invoke`；与
    ConversationSummarizer 的 SummaryLLM 协议保持同一形态，方便复用 fake。
    """

    def invoke(self, prompt: str) -> object:
        """同步生成文本。"""


class QueryRewriteResult(BaseModel):
    """一次查询改写的输出。

    `rewritten_query` 是修正错别字、补全上下文后的主检索 query；`variants` 是不同
    角度的额外检索变体。两者都允许为空：失败时只带 `error_code` 返回，调用方回退
    原始 query，不把改写失败当成检索失败。
    """

    model_config = ConfigDict(extra="forbid")

    rewritten_query: str | None = Field(default=None, description="改写后的主查询")
    variants: list[str] = Field(default_factory=list, description="额外检索变体")
    error_code: str | None = Field(default=None, description="失败错误码，None 表示成功")

    @property
    def successful(self) -> bool:
        """返回本次改写是否产出了可用内容。"""

        return self.error_code is None and bool(self.rewritten_query or self.variants)


class QueryRewriterLike(Protocol):
    """RagRetriever 依赖的 query rewriter 最小协议。"""

    def rewrite(
        self,
        *,
        query: str,
        variant_count: int,
        ctx: RequestContext | None = None,
    ) -> QueryRewriteResult:
        """返回改写后的主 query 与额外检索变体。"""


class BaseQueryRewriter(ABC):
    """Query rewriter 抽象基类。

    抽象类的职责与 BaseReranker 一致：稳定方法签名，防止未来实现把检索、过滤或
    citation 逻辑塞进改写层，破坏模块边界。
    """

    @abstractmethod
    def rewrite(
        self,
        *,
        query: str,
        variant_count: int,
        ctx: RequestContext | None = None,
    ) -> QueryRewriteResult:
        """按 query 生成改写主 query 与额外检索变体。"""


class NoopQueryRewriter(BaseQueryRewriter):
    """默认空实现：不做任何改写，保持旧单 query 检索行为。

    RagRetriever 默认注入该实现，保证未显式接入 LLM 的调用方（旧 API、evaluation
    runner、单元测试）不会误触 DashScope。
    """

    def rewrite(
        self,
        *,
        query: str,
        variant_count: int,
        ctx: RequestContext | None = None,
    ) -> QueryRewriteResult:
        _ = query, variant_count, ctx
        return QueryRewriteResult()


class LlmQueryRewriter(BaseQueryRewriter):
    """通过一次 LLM 调用生成改写 query 与多路检索变体。

    LLM 延迟构造且可注入：`llm=None` 时首次调用才经 LLMFactory 创建客户端，构造
    本身不触网，方便单元测试注入 fake 或完全跳过。任何异常都收敛为 error_code，
    不向调用方抛出。
    """

    def __init__(
        self,
        *,
        llm: RewriterLLM | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.llm = llm
        self.model = model or config.rag_query_rewrite_model
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.rag_query_rewrite_timeout_seconds
        )

    def rewrite(
        self,
        *,
        query: str,
        variant_count: int,
        ctx: RequestContext | None = None,
    ) -> QueryRewriteResult:
        _ = ctx
        normalized_query = query.strip()
        if not normalized_query:
            return QueryRewriteResult(error_code="EMPTY_QUERY")
        count = max(0, int(variant_count))
        try:
            prompt = self._build_prompt(normalized_query, count)
            response = self._get_llm().invoke(prompt)
            content = _extract_llm_content(response)
        except TimeoutError:
            return QueryRewriteResult(error_code="LLM_TIMEOUT")
        except Exception:
            return QueryRewriteResult(error_code="LLM_PROVIDER_ERROR")
        return self._parse_response(content, count)

    def _get_llm(self) -> RewriterLLM:
        if self.llm is None:
            from app.core.llm_factory import LLMFactory

            self.llm = LLMFactory.create_chat_model(
                model=self.model,
                temperature=0.0,
                streaming=False,
                timeout=self.timeout_seconds,
            )
        return self.llm

    def _build_prompt(self, query: str, variant_count: int) -> str:
        lines = [
            "你是运维知识库的检索查询优化器，负责把用户查询改写成更适合向量检索的形式。",
            f"第 1 行输出改写后的主查询：修正错别字、补全省略的主语与上下文，保持原有语言。",
            f"第 2 到 {1 + variant_count} 行输出从不同角度改写的检索变体：同义词替换、关键词化、换一种表述角度。",
            "要求：",
            "- 每行一个查询，不要编号、不要引号、不要任何解释。",
            "- 保持与用户查询相同的语言。",
            "- 不要编造用户未提到的系统、实体或指标。",
            f"用户查询：{query}",
        ]
        return "\n".join(lines)

    def _parse_response(
        self,
        content: str,
        variant_count: int,
    ) -> QueryRewriteResult:
        candidates: list[str] = []
        seen: set[str] = set()
        for raw_line in content.splitlines():
            line = _sanitize_variant(raw_line)
            if not line or line in seen:
                continue
            seen.add(line)
            candidates.append(line)
        if not candidates:
            return QueryRewriteResult(error_code="EMPTY_RESPONSE")
        return QueryRewriteResult(
            rewritten_query=candidates[0],
            variants=candidates[1 : 1 + variant_count],
        )


def _sanitize_variant(raw_line: str) -> str:
    """清洗单行 LLM 输出为可检索 query。

    只做保守清洗：去掉编号/引号等格式残留、限制长度。超长行大概率是解释性文本
    而不是查询，直接丢弃而不是截断，避免截断产物进入向量检索。冒号结尾的行
    （如“以下是改写结果：”）是引导语而非查询，同样丢弃。
    """

    line = _NUMBERED_LINE_RE.sub("", raw_line)
    line = line.strip().strip("\"'“”‘’「」『』").strip()
    if not line or len(line) > _MAX_VARIANT_CHARS:
        return ""
    if line.endswith("：") or line.endswith(":"):
        return ""
    return line


def _extract_llm_content(response: object) -> str:
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
