"""阶段 3C LlmQueryRewriter / NoopQueryRewriter 单元测试。

只验证改写器自身的 prompt 构建、输出解析、清洗和 fail-open 行为；LLM 一律注入
fake，不连接真实 DashScope/Milvus。
"""

from __future__ import annotations

from app.rag.query_rewriter import LlmQueryRewriter, NoopQueryRewriter


class FakeLLM:
    """返回预设响应并记录 prompt 的 fake LLM。"""

    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.response


class FailingLLM:
    def invoke(self, prompt: str) -> object:
        _ = prompt
        raise RuntimeError("llm error token=secret http://internal.local")


class TimeoutLLM:
    def invoke(self, prompt: str) -> object:
        _ = prompt
        raise TimeoutError("llm timeout")


class FakeAIMessage:
    """贴近 LangChain AIMessage 的返回形态，验证 `.content` 提取。"""

    def __init__(self, content: str) -> None:
        self.content = content


def test_noop_query_rewriter_returns_empty_result() -> None:
    result = NoopQueryRewriter().rewrite(query="CPU 使用率过高", variant_count=3)

    assert result.rewritten_query is None
    assert result.variants == []
    assert result.error_code is None
    assert result.successful is False


def test_llm_rewriter_parses_numbered_lines_from_string_response() -> None:
    llm = FakeLLM(
        "1. CPU 使用率过高如何排查\n2. CPU 利用率高 排查步骤\n3. 服务器 CPU 告警 处理手册\n"
    )
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="cpu 使用率过高", variant_count=2)

    assert result.error_code is None
    assert result.successful is True
    assert result.rewritten_query == "CPU 使用率过高如何排查"
    assert result.variants == ["CPU 利用率高 排查步骤", "服务器 CPU 告警 处理手册"]
    # prompt 一次性产出主 query 与变体，并携带原始 query
    assert "cpu 使用率过高" in llm.prompts[0]


def test_llm_rewriter_extracts_content_from_aimessage_response() -> None:
    llm = FakeLLM(FakeAIMessage("1. 改写后的查询\n"))
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="原始查询", variant_count=1)

    assert result.rewritten_query == "改写后的查询"
    assert result.variants == []


def test_llm_rewriter_strips_quotes_and_dedupes_repeated_lines() -> None:
    llm = FakeLLM('1. "带引号的主查询"\n2. 重复变体\n2. 重复变体\n3. 重复变体\n')
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="原始查询", variant_count=3)

    assert result.rewritten_query == "带引号的主查询"
    assert result.variants == ["重复变体"]


def test_llm_rewriter_truncates_variants_to_requested_count() -> None:
    llm = FakeLLM("1. 主查询\n2. 变体一\n3. 变体二\n4. 变体三\n")
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="原始查询", variant_count=1)

    assert result.rewritten_query == "主查询"
    assert result.variants == ["变体一"]


def test_llm_rewriter_drops_preamble_and_overlong_lines() -> None:
    overlong = "长" * 300
    llm = FakeLLM(f"好的，以下是改写结果：\n1. 正常主查询\n2. {overlong}\n")
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="原始查询", variant_count=2)

    assert result.rewritten_query == "正常主查询"
    assert result.variants == []


def test_llm_rewriter_empty_response_maps_to_error_code() -> None:
    llm = FakeLLM("好的，以下是改写结果：\n\n")
    rewriter = LlmQueryRewriter(llm=llm)

    result = rewriter.rewrite(query="原始查询", variant_count=2)

    assert result.error_code == "EMPTY_RESPONSE"
    assert result.successful is False
    assert result.rewritten_query is None


def test_llm_rewriter_rejects_blank_query() -> None:
    rewriter = LlmQueryRewriter(llm=FakeLLM("1. 任意输出"))

    result = rewriter.rewrite(query="   ", variant_count=1)

    assert result.error_code == "EMPTY_QUERY"
    assert result.successful is False


def test_llm_rewriter_maps_provider_failure_to_error_code() -> None:
    rewriter = LlmQueryRewriter(llm=FailingLLM())

    result = rewriter.rewrite(query="原始查询", variant_count=2)

    assert result.error_code == "LLM_PROVIDER_ERROR"
    assert result.successful is False


def test_llm_rewriter_maps_timeout_to_error_code() -> None:
    rewriter = LlmQueryRewriter(llm=TimeoutLLM())

    result = rewriter.rewrite(query="原始查询", variant_count=2)

    assert result.error_code == "LLM_TIMEOUT"
    assert result.successful is False
