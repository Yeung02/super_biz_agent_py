"""Markdown 表格/代码块原子保护分割测试。

验证 DocumentSplitterService 的原子片段 (表格、围栏代码块) 不被递归字符分割器
切开；超限时只在安全边界切分且续片自含语义；稳定 ID 对 CRLF/LF 输入保持幂等。
"""

from app.services.document_splitter_service import DocumentSplitterService


def _service() -> DocumentSplitterService:
    return DocumentSplitterService()


def _build_table(row_count: int, header: str = "| metric | value |") -> str:
    lines = [header, "| --- | --- |"]
    lines.extend(f"| key{i:03d} | value-{i:03d}-payload-padding |" for i in range(row_count))
    return "\n".join(lines)


def _build_code(line_count: int, blank_every: int = 5) -> str:
    lines = []
    for i in range(line_count):
        lines.append(f"def func_{i:02d}(): value = computation_step_{i:02d}  # padding padding")
        if (i + 1) % blank_every == 0:
            lines.append("")
    return "\n".join(lines)


def test_small_table_kept_intact_in_single_chunk() -> None:
    """未超限表格必须完整落在同一个分片里，不被逐行切开。"""
    service = _service()
    table = _build_table(10)
    content = f"# Runbook\n\n## 指标\n\n前置说明文字。\n\n{table}\n\n收尾说明文字。"

    docs = service.split_markdown(content, "runbook.md")

    header_docs = [d for d in docs if "| metric | value |" in d.page_content]
    assert len(header_docs) == 1, "表格应只出现在一个分片中"
    table_doc = header_docs[0]
    for i in range(10):
        assert f"| key{i:03d} |" in table_doc.page_content


def test_oversized_table_chunks_repeat_header() -> None:
    """超限表格按行组切分；每个续片都复制表头和分隔行，保证列语义自含。"""
    service = _service()
    table = _build_table(60)
    content = f"# Runbook\n\n## 指标\n\n{table}"

    docs = service.split_markdown(content, "runbook.md")

    table_docs = [d for d in docs if "| key" in d.page_content]
    assert len(table_docs) >= 2, "超限表格应被切分为多个分片"
    for doc in table_docs:
        assert "| metric | value |" in doc.page_content, "每个表格分片都应包含表头"
        assert "| --- | --- |" in doc.page_content, "每个表格分片都应包含分隔行"
        assert len(doc.page_content) <= service.secondary_chunk_size
    # 行数据不丢失
    all_rows = "\n".join(d.page_content for d in table_docs)
    for i in range(60):
        assert f"| key{i:03d} |" in all_rows


def test_small_code_block_kept_intact() -> None:
    """未超限代码块 (含内部空行) 必须完整落在同一个分片里。"""
    service = _service()
    code = "```bash\nps aux | grep python\n\nwc -l\n```"
    content = f"# Runbook\n\n## 命令\n\n执行以下命令排查。\n\n{code}\n\n说明结束。"

    docs = service.split_markdown(content, "runbook.md")

    code_docs = [d for d in docs if "ps aux" in d.page_content]
    assert len(code_docs) == 1
    assert "wc -l" in code_docs[0].page_content
    assert "```bash" in code_docs[0].page_content


def test_oversized_code_split_refenced_and_complete() -> None:
    """超限代码块在空行边界切分；每个分片都是完整围栏块，body 内容不丢失。"""
    service = _service()
    body = _build_code(40)
    content = f"# Runbook\n\n## 脚本\n\n```python\n{body}\n```"

    docs = service.split_markdown(content, "runbook.md")
    code_docs = [d for d in docs if "func_" in d.page_content]

    assert len(code_docs) >= 2, "超限代码块应被切分为多个分片"
    reconstructed_body: list[str] = []
    for doc in code_docs:
        # prose 可能因小片合并进入分片头部，围栏不一定在第 0 行，按行定位围栏边界
        chunk_lines = doc.page_content.split("\n")
        fence_starts = [
            i for i, line in enumerate(chunk_lines) if line.strip().startswith("```")
        ]
        assert fence_starts, "分片内应存在围栏行"
        opening_idx = fence_starts[0]
        assert chunk_lines[-1].strip() == "```", "续片应以围栏结束"
        reconstructed_body.extend(chunk_lines[opening_idx + 1 : -1])
        assert len(doc.page_content) <= service.secondary_chunk_size

    assert "\n".join(reconstructed_body).strip() == body.strip(), "代码内容按顺序拼回应等于原 body"


def test_pipe_lines_inside_fence_not_treated_as_table() -> None:
    """围栏内以 | 开头的行不应被识别为表格原子片段。"""
    service = _service()
    code = "```\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```"
    content = f"# Runbook\n\n## 图示\n\n{code}\n\n结束。"

    docs = service.split_markdown(content, "runbook.md")

    pipe_docs = [d for d in docs if "| a | b |" in d.page_content]
    assert len(pipe_docs) == 1
    assert "```" in pipe_docs[0].page_content, "管道行应保留在代码块分片内"


def test_tilde_fence_protected() -> None:
    """~~~ 围栏同样受保护。"""
    service = _service()
    code = "~~~yaml\nkey: value\n\nother: value\n~~~"
    content = f"# Runbook\n\n## 配置\n\n{code}"

    docs = service.split_markdown(content, "runbook.md")

    code_docs = [d for d in docs if "key: value" in d.page_content]
    assert len(code_docs) == 1
    assert "~~~yaml" in code_docs[0].page_content


def test_nested_fence_uses_longest_marker_to_close() -> None:
    """```` 外层围栏内的 ``` 不应提前关闭围栏。"""
    service = _service()
    code = "````\ninner markdown\n\n```python\nprint('hi')\n```\n\nouter text\n````"
    content = f"# Runbook\n\n## 嵌套\n\n{code}"

    docs = service.split_markdown(content, "runbook.md")

    code_docs = [d for d in docs if "inner markdown" in d.page_content]
    assert len(code_docs) == 1
    assert "print('hi')" in code_docs[0].page_content
    assert "outer text" in code_docs[0].page_content


def test_unclosed_fence_kept_whole() -> None:
    """未闭合围栏整体作为原子片段，不跨行切割。"""
    service = _service()
    content = "# Runbook\n\n## 片段\n\n```bash\necho one\necho two\n"

    docs = service.split_markdown(content, "runbook.md")

    fence_docs = [d for d in docs if "echo one" in d.page_content]
    assert len(fence_docs) == 1
    assert "echo two" in fence_docs[0].page_content


def test_atomic_chunk_preserves_header_metadata() -> None:
    """原子片段分片应继承 Markdown 标题 metadata (h1/h2)。"""
    service = _service()
    table = _build_table(5)
    content = f"# Runbook\n\n## 指标\n\n{table}"

    docs = service.split_markdown(content, "runbook.md")

    table_doc = next(d for d in docs if "| metric | value |" in d.page_content)
    assert table_doc.metadata.get("h1") == "Runbook"
    assert table_doc.metadata.get("h2") == "指标"


def test_crlf_input_produces_same_stable_ids() -> None:
    """CRLF 与 LF 输入应产生相同 doc_id/chunk_id，保持索引幂等。"""
    service = _service()
    table = _build_table(8)
    code = "```bash\nps aux | grep python\n```"
    content = f"# Runbook\n\n## 混合\n\n说明文字。\n\n{table}\n\n{code}\n\n收尾。"

    lf_docs = service.split_markdown(content, "runbook.md")
    crlf_docs = service.split_markdown(content.replace("\n", "\r\n"), "runbook.md")

    def _ids(docs):
        return (
            docs[0].metadata["doc_id"],
            sorted(d.metadata["chunk_id"] for d in docs),
        )

    assert _ids(lf_docs) == _ids(crlf_docs)
