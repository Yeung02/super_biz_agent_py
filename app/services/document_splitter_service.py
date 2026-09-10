"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config
from app.rag.models import DEFAULT_TENANT_ID, build_stable_rag_metadata

# 围栏代码块起始标记: 3 个及以上连续反引号或波浪线
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})")


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap
        # 二级分割上限显式来自配置；兜底不低于 chunk_size，避免误配置导致上限倒挂
        self.secondary_chunk_size = max(config.chunk_secondary_size, self.chunk_size)

        # Markdown 标题分割器 (只按一级和二级标题分割，减少分片数)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器 (用于二次分割，上限为二级分割配置)
        # 默认 separators 只有 "\n\n"/"\n"/" "/""
        # 中文长段落内没有空格，会退化为硬字符切割把句子拦腰截断，损害 embedding
        # 质量，因此补充中文句读分隔符，让切分优先落在句子边界上。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.secondary_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.secondary_chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 原子保护分割 (表格/代码块不切开) + prose 递归分割
            docs_after_split = self._split_sections_with_atomic_protection(md_docs)

            # 第三阶段: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            self._apply_stable_metadata(
                final_docs,
                content=content,
                file_path=file_path,
                fallback_extension=".md",
                source_root=source_root,
                tenant_id=tenant_id,
            )

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    self._legacy_metadata(file_path, fallback_extension=Path(file_path).suffix)
                ],
            )
            # 与 Markdown 路径保持一致，合并过小分片，避免检索侧碎片化
            docs = self._merge_small_chunks(docs, min_size=300)
            self._apply_stable_metadata(
                docs,
                content=content,
                file_path=file_path,
                fallback_extension=Path(file_path).suffix,
                source_root=source_root,
                tenant_id=tenant_id,
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(
        self,
        content: str,
        file_path: str = "",
        *,
        source_root: str | Path | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if file_path.casefold().endswith(".md"):
            return self.split_markdown(
                content,
                file_path,
                source_root=source_root,
                tenant_id=tenant_id,
            )
        else:
            return self.split_text(
                content,
                file_path,
                source_root=source_root,
                tenant_id=tenant_id,
            )

    def _split_sections_with_atomic_protection(self, md_docs: list[Document]) -> list[Document]:
        """按 "prose/原子片段" 分流分割 section，保护表格和代码块不被切开。

        原子片段 (表格、围栏代码块) 不经过递归字符分割器: 未超限直接成块，超限只在
        安全边界切分; prose 走正常递归字符分割。输出保持原文顺序，保证 chunk_index
        顺序稳定，不影响稳定 ID 幂等性。
        """
        result: list[Document] = []
        for doc in md_docs:
            for kind, segment in self._extract_atomic_spans(doc.page_content):
                if not segment.strip():
                    continue
                if kind == "atomic":
                    result.extend(self._split_atomic_segment(segment, doc.metadata))
                else:
                    result.extend(
                        self.text_splitter.create_documents(
                            texts=[segment],
                            metadatas=[dict(doc.metadata)],
                        )
                    )
        return result

    def _extract_atomic_spans(self, text: str) -> list[tuple[str, str]]:
        """行级状态机扫描，把文本切成交替的 prose/原子片段。

        原子片段是围栏代码块 (``` 或 ~~~) 和连续以 `|` 开头的表格行。围栏内不识别
        表格 (shell 管道符等不误判); 围栏须由同类型且长度不小于起始标记的行关闭，
        支持 ```` 嵌套 ``` 的写法。行尾统一归一为 LF，与
        `_normalize_content_for_hash` 的幂等行为保持一致。
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        spans: list[tuple[str, str]] = []
        prose_lines: list[str] = []
        table_lines: list[str] = []
        fence_lines: list[str] = []
        fence_char = ""
        fence_len = 0

        def _flush_prose() -> None:
            if prose_lines:
                spans.append(("prose", "\n".join(prose_lines)))
                prose_lines.clear()

        def _flush_table() -> None:
            if table_lines:
                spans.append(("atomic", "\n".join(table_lines)))
                table_lines.clear()

        for line in normalized.split("\n"):
            stripped = line.strip()
            if fence_char:
                fence_lines.append(line)
                # 关闭围栏: 整行只由同类型标记字符组成且长度 >= 起始标记
                if stripped and set(stripped) == {fence_char} and len(stripped) >= fence_len:
                    spans.append(("atomic", "\n".join(fence_lines)))
                    fence_lines = []
                    fence_char = ""
                continue

            fence_match = _FENCE_OPEN_RE.match(stripped)
            if fence_match:
                _flush_prose()
                _flush_table()
                fence_char = fence_match.group(1)[0]
                fence_len = len(fence_match.group(1))
                fence_lines = [line]
                continue

            if stripped.startswith("|"):
                _flush_prose()
                table_lines.append(line)
                continue

            _flush_table()
            prose_lines.append(line)

        if fence_lines:
            # 未闭合围栏: 整体按原子片段处理，不跨行切割
            spans.append(("atomic", "\n".join(fence_lines)))
        _flush_table()
        _flush_prose()
        return spans

    def _split_atomic_segment(self, segment: str, metadata: dict) -> list[Document]:
        """分割原子片段: 未超限整块保留；超限按类型选择安全边界。"""
        if len(segment) <= self.secondary_chunk_size:
            return [Document(page_content=segment, metadata=dict(metadata))]

        lines = segment.split("\n")
        first = lines[0].strip()
        if first.startswith("|"):
            return self._split_oversized_table(lines, metadata)
        if first.startswith(("```", "~~~")):
            return self._split_oversized_code(lines, metadata)
        return self._hard_split(segment, metadata)

    def _split_oversized_table(self, lines: list[str], metadata: dict) -> list[Document]:
        """超限表格按行组切分；续片复制表头和分隔行，保证每片自含列语义。"""
        header = lines[:2]
        header_len = len("\n".join(header)) + 1
        if header_len >= self.secondary_chunk_size:
            return self._hard_split("\n".join(lines), metadata)

        chunks: list[str] = []
        current = list(header)
        current_len = header_len
        for row in lines[2:]:
            row_len = len(row) + 1
            if current_len + row_len > self.secondary_chunk_size and len(current) > len(header):
                chunks.append("\n".join(current))
                current = list(header)
                current_len = header_len
            current.append(row)
            current_len += row_len
        if current:
            chunks.append("\n".join(current))
        return [Document(page_content=chunk, metadata=dict(metadata)) for chunk in chunks]

    def _split_oversized_code(self, lines: list[str], metadata: dict) -> list[Document]:
        """超限代码块在块内空行边界切分；续片重建首尾围栏，保持完整代码块形态。"""
        opening, closing = lines[0], lines[-1]
        body = lines[1:-1]
        overhead = len(opening) + len(closing) + 2
        if overhead >= self.secondary_chunk_size:
            return self._hard_split("\n".join(lines), metadata)

        # 按空行分组; 空行保留在前一组尾部，分组按原文顺序拼回即等于原 body
        groups: list[list[str]] = []
        current: list[str] = []
        for line in body:
            if line.strip():
                if current and current[-1].strip() == "":
                    groups.append(current)
                    current = []
                current.append(line)
            elif current:
                current.append(line)
        if current:
            groups.append(current)

        if not groups:
            return [Document(page_content="\n".join(lines), metadata=dict(metadata))]

        chunks: list[str] = []
        packed: list[str] = []
        packed_len = overhead
        for group in groups:
            group_len = len("\n".join(group)) + 1
            if packed_len + group_len > self.secondary_chunk_size and packed:
                chunks.append("\n".join([opening, *packed, closing]))
                packed = []
                packed_len = overhead
            if overhead + group_len > self.secondary_chunk_size:
                # 单个分组自身超限: 组内按行硬分组
                for piece in self._split_lines_by_size(group):
                    chunks.append("\n".join([opening, *piece, closing]))
                continue
            packed.extend(group)
            packed_len += group_len
        if packed:
            chunks.append("\n".join([opening, *packed, closing]))
        return [Document(page_content=chunk, metadata=dict(metadata)) for chunk in chunks]

    def _split_lines_by_size(self, lines: list[str]) -> list[list[str]]:
        """按大小贪心分组行；单行超限退化为字符切分。"""
        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if line_len > self.secondary_chunk_size:
                if current:
                    groups.append(current)
                    current = []
                    current_len = 0
                text = line
                while len(text) > self.secondary_chunk_size:
                    groups.append([text[: self.secondary_chunk_size]])
                    text = text[self.secondary_chunk_size :]
                if text:
                    current = [text]
                    current_len = len(text) + 1
                continue
            if current_len + line_len > self.secondary_chunk_size and current:
                groups.append(current)
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            groups.append(current)
        return groups

    def _hard_split(self, text: str, metadata: dict) -> list[Document]:
        """无安全边界的兜底切分: 硬字符切割并记录警告。"""
        logger.warning(
            f"原子片段超过二级分割上限 ({self.secondary_chunk_size}) 且无安全边界，"
            f"退化为硬字符切分: {text[:40]!r}"
        )
        size = self.secondary_chunk_size
        return [
            Document(page_content=text[i : i + size], metadata=dict(metadata))
            for i in range(0, len(text), size)
        ]

    def _merge_small_chunks(
        self, documents: list[Document], min_size: int = 300
    ) -> list[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif (
                doc_size < min_size or len(current_doc.page_content) < min_size
            ) and len(current_doc.page_content) + doc_size + len("\n\n") <= self.secondary_chunk_size:
                # 任一侧过小且合并后总长不超过二级分割上限，则合并。
                # 按合并后的总长判断，避免 "\n\n" 拼接后超出上限；
                # 同时让过小的 current 也能被后续大分片吸收，不留碎片。
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def _apply_stable_metadata(
        self,
        documents: list[Document],
        *,
        content: str,
        file_path: str,
        fallback_extension: str,
        source_root: str | Path | None,
        tenant_id: str,
    ) -> None:
        """为分片写入稳定 RAG metadata，同时保留旧字段。

        `_source/_file_name/_extension` 是旧索引和删除逻辑仍依赖的兼容字段，不能删除；
        `doc_id/chunk_id/content_hash/source_path` 是 ISSUE-018 新增字段。这里统一写入，
        可以让后续 vector store 用 `chunk_id` 作为主键，同时让旧 `delete_by_source`
        在 ISSUE-019 前继续工作。
        """

        if not documents:
            return

        legacy_metadata = self._legacy_metadata(file_path, fallback_extension=fallback_extension)
        source_for_stable_id = legacy_metadata["_source"] or "unknown"
        first_content_hash = ""
        first_doc_id = ""
        for chunk_index, doc in enumerate(documents):
            stable_metadata = build_stable_rag_metadata(
                tenant_id=tenant_id,
                source_path=str(source_for_stable_id),
                source_root=source_root,
                chunk_index=chunk_index,
                chunk_text=doc.page_content,
                document_content=content,
            )
            # 先保留旧字段，再追加新字段。这样 h1/h2 等 Markdown metadata 不会丢失，
            # 且新字段优先被 RagMetadata.from_mapping 读取，避免继续走 legacy ID。
            doc.metadata.update(legacy_metadata)
            doc.metadata.update(stable_metadata)
            first_doc_id = str(stable_metadata["doc_id"])
            first_content_hash = str(stable_metadata["content_hash"])

        logger.info(
            "RAG 稳定 metadata 写入完成: "
            f"doc_id={first_doc_id}, chunk_count={len(documents)}, "
            f"content_hash_prefix={first_content_hash[:12]}, "
            f"metadata_version={documents[0].metadata.get('version')}"
        )

    @staticmethod
    def _legacy_metadata(file_path: str, *, fallback_extension: str) -> dict[str, str]:
        """生成旧 metadata 字段。

        当内部测试或临时调用没有传入文件路径时，旧实现仍会返回分片而不是报错。这里用
        `unknown` 兜底只影响这类非索引调用；真实上传/目录索引会先经过 InputGuard，
        因此始终带有可规范化的文件路径。
        """

        path_text = Path(file_path).as_posix() if file_path else "unknown"
        extension = (Path(path_text).suffix or fallback_extension or "").casefold()
        return {
            "_source": path_text,
            "_extension": extension,
            "_file_name": Path(path_text).name,
        }


# 全局单例
document_splitter_service = DocumentSplitterService()
