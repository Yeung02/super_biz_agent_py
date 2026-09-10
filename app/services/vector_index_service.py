"""向量索引服务模块"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.errors import AppError, InvalidDirectoryError, JsonValue
from app.core.input_guard import input_guard
from app.rag.models import build_stable_rag_metadata
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


@dataclass(frozen=True)
class SingleFileIndexResult:
    """单文件索引结果。

    目录索引需要汇总成功 doc_id、chunk ids 和删除数量；显式返回这些字段，可以避免
    目录层解析 Document 内部结构，也让单文件索引仍保持独立测试和独立回滚。
    """

    doc_id: str | None
    chunk_ids: list[str]
    deleted_count: int
    source_deleted_count: int
    # 增量索引计数：added 含新增与内容变更后重写的 chunk；skipped 是与已有
    # content_hash 完全一致、本次零写入零 embedding 的 chunk。
    added_count: int = 0
    skipped_count: int = 0

    @classmethod
    def empty(cls) -> "SingleFileIndexResult":
        """空文件或无可索引分片时的稳定返回。"""

        return cls(
            doc_id=None,
            chunk_ids=[],
            deleted_count=0,
            source_deleted_count=0,
            added_count=0,
            skipped_count=0,
        )


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error_message = ""
        self.error_code = ""
        self.failed_files: dict[str, str] = {}
        self.failed_file_error_codes: dict[str, str] = {}
        self.status = "pending"
        self.partial_success = False
        self.indexed_doc_ids: list[str] = []
        # 增量索引目录级汇总：来自各文件 SingleFileIndexResult 的计数累加。
        self.added_chunk_count = 0
        self.skipped_chunk_count = 0
        self.deleted_chunk_count = 0

    def increment_success_count(self, *, indexed_doc_id: str | None = None):
        """增加成功计数"""
        self.success_count += 1
        if indexed_doc_id and indexed_doc_id not in self.indexed_doc_ids:
            self.indexed_doc_ids.append(indexed_doc_id)

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str, *, code: str | None = None):
        """添加失败文件"""
        self.failed_files[file_path] = error
        if code:
            self.failed_file_error_codes[file_path] = code

    def finalize_status(self) -> None:
        """根据成功/失败数量生成稳定任务状态。

        旧 API 仍通过 `success_count/fail_count/failed_files` 表达结果；这里追加的
        `status/partial_success/failed_file_count/indexed_doc_ids` 只提供更清晰的
        失败任务诊断，不删除或重命名任何旧字段。
        """

        self.partial_success = self.success_count > 0 and self.fail_count > 0
        if self.fail_count == 0:
            self.status = "success"
        elif self.partial_success:
            self.status = "partial_success"
        else:
            self.status = "failed"
        self.success = self.fail_count == 0
        if self.fail_count:
            self.error_message = self.error_message or next(iter(self.failed_files.values()), "")
        if self.status == "failed" and self.failed_file_error_codes:
            unique_codes = set(self.failed_file_error_codes.values())
            if len(unique_codes) == 1:
                self.error_code = next(iter(unique_codes))

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> dict[str, JsonValue]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "errorMessage": self.error_message,
            "error_code": self.error_code,
            "failed_files": self.failed_files,
            "failed_file_error_codes": self.failed_file_error_codes,
            "status": self.status,
            "partial_success": self.partial_success,
            "failed_file_count": len(self.failed_files),
            "indexed_doc_ids": self.indexed_doc_ids,
            "added_chunk_count": self.added_chunk_count,
            "skipped_chunk_count": self.skipped_chunk_count,
            "deleted_chunk_count": self.deleted_chunk_count,
        }


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(self):
        """初始化向量索引服务"""
        self.upload_path = config.upload_dir
        logger.info("向量索引服务初始化完成")

    def index_directory(
        self,
        directory_path: str | None = None,
        *,
        allowed_root: str | Path | None = None,
    ) -> IndexingResult:
        """
        索引指定目录下的所有文件

        Args:
            directory_path: 目录路径（可选，默认使用配置的上传目录）

        Returns:
            IndexingResult: 索引结果
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录。HTTP API 已经做 allowlist 校验；service 层仍保留
            # exists/is_dir 判断，保证内部调用不经过 API 时也不会把无效路径交给下游。
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise InvalidDirectoryError()

            security_root, metadata_source_root = _resolve_index_roots(
                dir_path,
                explicit_allowed_root=allowed_root,
            )
            result.directory_path = str(dir_path)

            # 保持旧行为：只尝试索引支持扩展名的同级文件；同时按 ISSUE-004 增加 .markdown。
            # symlink 文件如果扩展名匹配也进入列表，并在文件级 guard 中记录失败而不是跟随读取。
            allowed_extensions = _normalize_extensions(config.allowed_upload_extensions)
            files = sorted(
                (
                    file_path
                    for file_path in dir_path.iterdir()
                    if (file_path.is_file() or file_path.is_symlink())
                    and file_path.suffix.casefold() in allowed_extensions
                ),
                key=lambda item: item.name,
            )

            result.total_files = len(files)
            logger.info(
                "rag.index.start directory_path={} total_files={}",
                str(dir_path),
                result.total_files,
            )

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.end_time = datetime.now()
                result.finalize_status()
                logger.info(
                    "rag.index.end directory_path={} status={} success_count={} fail_count={}",
                    result.directory_path,
                    result.status,
                    result.success_count,
                    result.fail_count,
                )
                return result

            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            for file_path in files:
                try:
                    guarded = input_guard.validate_index_file(
                        file_path,
                        allowed_root=security_root,
                        allowed_extensions=tuple(allowed_extensions),
                        max_bytes=config.upload_max_bytes,
                    )
                    single_file_result = self.index_single_file(
                        str(guarded.value.file_path),
                        allowed_root=security_root,
                        source_root=metadata_source_root,
                    )
                    # 兼容旧内部调用和测试替身：历史上的 index_single_file 成功时不返回对象。
                    # 新实现会返回 doc_id 便于目录任务汇总，但旧式 None 仍必须算作成功，避免破坏旧前端和已有集成。
                    indexed_doc_id = (
                        single_file_result.doc_id
                        if isinstance(single_file_result, SingleFileIndexResult)
                        else None
                    )
                    result.increment_success_count(indexed_doc_id=indexed_doc_id)
                    if isinstance(single_file_result, SingleFileIndexResult):
                        result.added_chunk_count += single_file_result.added_count
                        result.skipped_chunk_count += single_file_result.skipped_count
                        result.deleted_chunk_count += single_file_result.deleted_count
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except AppError as e:
                    result.increment_fail_count()
                    result.add_failed_file(
                        str(file_path.resolve(strict=False)),
                        e.user_message,
                        code=e.code,
                    )
                    logger.warning(
                        "rag.index.error file={} code={} fail_count={}",
                        str(file_path.resolve(strict=False)),
                        e.code,
                        result.fail_count,
                    )
                    logger.warning(f"文件索引前校验失败: {file_path.name}, code={e.code}")
                except Exception as e:
                    app_error = AppError.from_exception(
                        e,
                        origin_module="app.services.vector_index_service.index_directory",
                    )
                    result.increment_fail_count()
                    result.add_failed_file(
                        str(file_path.resolve(strict=False)),
                        app_error.user_message,
                        code=app_error.code,
                    )
                    logger.error(
                        "rag.index.error file={} code={} fail_count={}",
                        str(file_path.resolve(strict=False)),
                        app_error.code,
                        result.fail_count,
                    )
                    logger.error(f"文件索引失败: {file_path.name}, code={app_error.code}")

            result.end_time = datetime.now()
            result.finalize_status()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )
            logger.info(
                "rag.index.end directory_path={} status={} partial_success={} "
                "success_count={} fail_count={} indexed_doc_ids={} failed_file_count={} "
                "added_chunk_count={} skipped_chunk_count={} deleted_chunk_count={}",
                result.directory_path,
                result.status,
                result.partial_success,
                result.success_count,
                result.fail_count,
                result.indexed_doc_ids,
                len(result.failed_files),
                result.added_chunk_count,
                result.skipped_chunk_count,
                result.deleted_chunk_count,
            )

            return result

        except AppError as e:
            logger.warning(
                "rag.index.error directory_path={} code={} fail_count={}",
                directory_path or self.upload_path,
                e.code,
                result.fail_count,
            )
            logger.warning(f"索引目录被拒绝: code={e.code}")
            result.success = False
            result.status = "failed"
            result.partial_success = False
            result.error_message = e.user_message
            result.end_time = datetime.now()
            return result
        except Exception as e:
            app_error = AppError.from_exception(
                e,
                origin_module="app.services.vector_index_service.index_directory",
            )
            logger.error(
                "rag.index.error directory_path={} code={} fail_count={}",
                directory_path or self.upload_path,
                app_error.code,
                result.fail_count,
            )
            logger.error(f"索引目录失败: code={app_error.code}")
            result.success = False
            result.status = "failed"
            result.partial_success = False
            result.error_message = app_error.user_message
            result.end_time = datetime.now()
            return result

    def index_single_file(
        self,
        file_path: str,
        *,
        allowed_root: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> SingleFileIndexResult:
        """
        索引单个文件 (使用新的 LangChain 分割器)

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        root_path = Path(allowed_root or self.upload_path).resolve()
        metadata_source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else _metadata_source_root_for_allowed_root(root_path)
        )
        guarded = input_guard.validate_index_file(
            file_path,
            allowed_root=root_path,
            allowed_extensions=tuple(_normalize_extensions(config.allowed_upload_extensions)),
            max_bytes=config.upload_max_bytes,
        )
        path = guarded.value.file_path

        logger.info(f"开始索引文件: {path}")
        logger.info("rag.index.start file={}", str(path))

        try:
            # 1. 读取文件内容
            content = path.read_text(encoding="utf-8")
            logger.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")

            # 2. 删除该文件的旧数据（如果存在）
            normalized_path = path.as_posix()
            # ISSUE-019 要先拿到稳定 doc_id，再执行 doc-level delete。旧 `_source`
            # 删除仍保留，但必须放在 doc_id 删除之后作为历史数据兼容兜底。

            # 3. 使用新的文档分割器。ISSUE-018 要求新增的 `source_path` 是 allowlist
            # 内相对路径；这里把已经由 InputGuard 校验过的根目录传给分割器，避免
            # metadata 生成阶段把本机绝对路径写入新的 RAG 字段。旧 `_source` 仍保留
            # `normalized_path`，让 ISSUE-019 前的 `delete_by_source` 兼容逻辑继续可用。
            documents = document_splitter_service.split_document(
                content,
                normalized_path,
                source_root=metadata_source_root,
            )
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 4. 新的幂等路径要求先按 doc_id 删除旧 chunk，再按旧 `_source` 做兼容清理。
            # 如果文件为空或分割器没有产出 chunk，就不触发删除和写入，避免误删已有文档。
            if not documents:
                doc_id = (
                    _build_doc_id_for_path(path, source_root=metadata_source_root)
                    if config.stable_rag_ids_enabled
                    else None
                )
                doc_deleted_count = (
                    vector_store_manager.delete_by_doc_id(doc_id)
                    if config.stable_rag_ids_enabled and doc_id
                    else 0
                )
                source_deleted_count = vector_store_manager.delete_by_source(normalized_path)
                deleted_count = doc_deleted_count + source_deleted_count
                logger.warning(f"文件内容为空或无法分割: {file_path}")
                logger.info(
                    "rag.index.end file={} doc_id={} deleted_count={} success_count={} fail_count={}",
                    file_path,
                    doc_id,
                    deleted_count,
                    0,
                    0,
                )
                return SingleFileIndexResult(
                    doc_id=doc_id,
                    chunk_ids=[],
                    deleted_count=deleted_count,
                    source_deleted_count=source_deleted_count,
                )

            doc_id = _extract_doc_id(documents)
            chunk_ids = _extract_chunk_ids(documents)

            use_incremental = (
                config.incremental_index_enabled
                and config.stable_rag_ids_enabled
                and doc_id is not None
                and _has_valid_chunk_identities(documents)
            )

            if use_incremental:
                return self._index_single_file_incremental(
                    file_path=file_path,
                    normalized_path=normalized_path,
                    doc_id=doc_id,
                    documents=documents,
                    chunk_ids=chunk_ids,
                )

            doc_deleted_count = (
                vector_store_manager.delete_by_doc_id(doc_id)
                if config.stable_rag_ids_enabled and doc_id
                else 0
            )
            source_deleted_count = vector_store_manager.delete_by_source(normalized_path)
            added_ids = vector_store_manager.add_documents(documents)
            indexed_chunk_ids = [str(item) for item in added_ids] or chunk_ids
            deleted_count = doc_deleted_count + source_deleted_count
            logger.info(
                "文件索引完成: file={}, doc_id={}, chunks={}, deleted_count={}",
                file_path,
                doc_id,
                len(documents),
                deleted_count,
            )
            logger.info(
                "rag.index.end file={} doc_id={} deleted_count={} success_count={} fail_count={} "
                "added_chunk_count={} skipped_chunk_count={}",
                file_path,
                doc_id,
                deleted_count,
                1,
                0,
                len(documents),
                0,
            )
            return SingleFileIndexResult(
                doc_id=doc_id,
                chunk_ids=indexed_chunk_ids,
                deleted_count=deleted_count,
                source_deleted_count=source_deleted_count,
                added_count=len(documents),
                skipped_count=0,
            )

        except AppError as e:
            logger.warning(
                "rag.index.error file={} code={} success_count={} fail_count={}",
                file_path,
                e.code,
                0,
                1,
            )
            raise
        except Exception as e:
            app_error = AppError.from_exception(
                e,
                origin_module="app.services.vector_index_service.index_single_file",
            )
            logger.error(
                "rag.index.error file={} code={} success_count={} fail_count={}",
                file_path,
                app_error.code,
                0,
                1,
            )
            logger.error(f"索引文件失败: {file_path}, code={app_error.code}")
            raise app_error from e

    def _index_single_file_incremental(
        self,
        *,
        file_path: str,
        normalized_path: str,
        doc_id: str,
        documents: list[Document],
        chunk_ids: list[str],
    ) -> SingleFileIndexResult:
        """按 content_hash 做主键级 diff，只重写受影响的 chunk。

        未变更 chunk 不再重复 embedding（零写入零删除）；`chunk_id` 含顺序，
        中间插入导致的后缀平移会体现为 hash 不一致，按"一删一写"处理，语义正确。
        old 为空（首建）时仍执行一次 `delete_by_source`，清理旧 UUID 主键时代的
        legacy 残留——它们没有 doc_id，按 doc_id 查询不可见。
        """

        old_hashes = vector_store_manager.get_chunk_hashes_by_doc_id(doc_id)
        new_hashes = {
            str(document.metadata["chunk_id"]): str(document.metadata["content_hash"])
            for document in documents
        }
        added_documents = [
            document
            for document in documents
            if old_hashes.get(str(document.metadata["chunk_id"]))
            != str(document.metadata["content_hash"])
        ]
        removed_chunk_ids = [
            chunk_id
            for chunk_id, content_hash in old_hashes.items()
            if new_hashes.get(chunk_id) != content_hash
        ]

        if not added_documents and not removed_chunk_ids:
            logger.info(
                "文件内容未变化，跳过重嵌入: file={}, doc_id={}, chunks={}",
                file_path,
                doc_id,
                len(documents),
            )
            logger.info(
                "rag.index.end file={} doc_id={} deleted_count={} success_count={} fail_count={} "
                "added_chunk_count={} skipped_chunk_count={}",
                file_path,
                doc_id,
                0,
                1,
                0,
                0,
                len(documents),
            )
            return SingleFileIndexResult(
                doc_id=doc_id,
                chunk_ids=chunk_ids,
                deleted_count=0,
                source_deleted_count=0,
                added_count=0,
                skipped_count=len(documents),
            )

        chunk_deleted_count = (
            vector_store_manager.delete_by_chunk_ids(removed_chunk_ids)
            if removed_chunk_ids
            else 0
        )
        source_deleted_count = (
            vector_store_manager.delete_by_source(normalized_path)
            if not old_hashes
            else 0
        )
        added_ids = (
            vector_store_manager.add_documents(added_documents) if added_documents else []
        )
        indexed_chunk_ids = [str(item) for item in added_ids] or chunk_ids
        logger.info(
            "增量索引完成: file={}, doc_id={}, added={}, skipped={}, deleted={}",
            file_path,
            doc_id,
            len(added_documents),
            len(documents) - len(added_documents),
            chunk_deleted_count,
        )
        logger.info(
            "rag.index.end file={} doc_id={} deleted_count={} success_count={} fail_count={} "
            "added_chunk_count={} skipped_chunk_count={}",
            file_path,
            doc_id,
            chunk_deleted_count + source_deleted_count,
            1,
            0,
            len(added_documents),
            len(documents) - len(added_documents),
        )
        return SingleFileIndexResult(
            doc_id=doc_id,
            chunk_ids=indexed_chunk_ids,
            deleted_count=chunk_deleted_count + source_deleted_count,
            source_deleted_count=source_deleted_count,
            added_count=len(added_documents),
            skipped_count=len(documents) - len(added_documents),
        )


def _extract_doc_id(documents: list[Document]) -> str | None:
    """从分片 metadata 中提取稳定 doc_id。

    ISSUE-019 处在新旧 metadata 共存阶段：新文档应都有 `doc_id`，但为了保护旧内部调用，
    缺失时不在这里构造临时 ID，而是返回 None，让调用方跳过 doc-level delete 并继续执行
    旧 `_source` 删除兜底。
    """

    for document in documents:
        value = document.metadata.get("doc_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_chunk_ids(documents: list[Document]) -> list[str]:
    """从分片 metadata 中提取 chunk_id，缺失的旧分片不伪造 ID。"""

    chunk_ids: list[str] = []
    for document in documents:
        value = document.metadata.get("chunk_id")
        if isinstance(value, str) and value.strip():
            chunk_ids.append(value.strip())
    return chunk_ids


def _has_valid_chunk_identities(documents: list[Document]) -> bool:
    """校验全部分片都带合法 chunk_id/content_hash，防御迁移期混合数据进入增量 diff。

    只要有一片缺身份字段，按 doc_id 查询得到的老集合就不可信，必须回退
    ISSUE-019 的全量重建路径，避免在新旧混杂的 chunk 集上做错误 diff。
    """

    for document in documents:
        chunk_id = document.metadata.get("chunk_id")
        content_hash = document.metadata.get("content_hash")
        if not (
            isinstance(chunk_id, str)
            and chunk_id.strip()
            and isinstance(content_hash, str)
            and content_hash.strip()
        ):
            return False
    return True


def _normalize_extensions(extensions: list[str]) -> set[str]:
    """规范化扩展名配置，兼容 `.md` 和 `md` 两种写法。

    目录索引服务只需要判断文件后缀是否应进入本轮索引；具体的错误码和边界解释由
    InputGuard 负责，避免 service 层复制一套安全规则。
    """

    normalized: set[str] = set()
    for extension in extensions:
        clean_extension = extension.strip().casefold()
        if not clean_extension:
            continue
        if not clean_extension.startswith("."):
            clean_extension = f".{clean_extension}"
        normalized.add(clean_extension)
    return normalized


def _resolve_index_roots(
    dir_path: Path,
    *,
    explicit_allowed_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the guard root and the metadata root for directory indexing."""

    if explicit_allowed_root is not None:
        allowed_root = Path(explicit_allowed_root).resolve()
        return allowed_root, _metadata_source_root_for_allowed_root(allowed_root)

    for configured_root in config.index_allowed_directories:
        allowed_root = Path(configured_root).resolve()
        if _is_relative_to(dir_path, allowed_root):
            return allowed_root, _metadata_source_root_for_allowed_root(allowed_root)

    return dir_path, dir_path


def _metadata_source_root_for_allowed_root(allowed_root: Path) -> Path:
    """Keep the allowlist directory name in stable RAG source paths.

    For the default roots this yields `uploads/foo.md` and
    `aiops-docs/foo.md`, matching the eval-set IDs and API examples while
    still keeping absolute local paths out of new RAG metadata.
    """

    parent = allowed_root.parent
    return parent if parent != allowed_root else allowed_root


def _build_doc_id_for_path(path: Path, *, source_root: Path) -> str:
    """Build the stable doc_id even when a valid file produces no chunks."""

    metadata = build_stable_rag_metadata(
        tenant_id="default",
        source_path=path.as_posix(),
        source_root=source_root,
        chunk_index=0,
        chunk_text="",
    )
    return str(metadata["doc_id"])


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# 全局单例
vector_index_service = VectorIndexService()
