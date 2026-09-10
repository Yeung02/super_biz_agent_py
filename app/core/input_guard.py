"""Input validation boundary for ISSUE-003.

    该模块位于已解析的 API 模型与业务服务之间。
    它通过仅对当前任务所需的字段进行标准化处理，确保 API 处理逻辑保持精简；
    同时，在 Agent、RAG、Tool 或 AIOps 相关代码接触到无效用户输入之前，便抛出稳定的 AppError 子类异常。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from app.core.errors import (
    FileTooLargeError,
    InvalidDirectoryError,
    InvalidFileEncodingError,
    InvalidFileMimeError,
    InvalidInputError,
    InvalidSessionIdError,
    PathTraversalBlockedError,
    RequestTooLargeError,
    SymlinkNotAllowedError,
    UnsupportedFileTypeError,
)
from app.models.aiops import AIOpsRequest
from app.models.request import ChatRequest, ClearRequest

if TYPE_CHECKING:
    from app.core.request_context import RequestContext


DEFAULT_MAX_TEXT_CHARS = 8_000
DEFAULT_MAX_SESSION_CHARS = 128
DEFAULT_UPLOAD_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_ALLOWED_EXTENSIONS = (".txt", ".md", ".markdown")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "disregard previous instructions",
    "override instructions",
    "system prompt",
    "developer message",
    "jailbreak",
    "忽略之前",
    "忽略以上",
    "系统提示",
)
_MIME_BY_EXTENSION: dict[str, tuple[str, ...]] = {
    ".txt": ("text/plain",),
    ".md": ("text/markdown", "text/plain"),
    ".markdown": ("text/markdown", "text/plain"),
}

T = TypeVar("T")


@dataclass(frozen=True)
class PromptInjectionRisk:
    """针对类似提示词注入（prompt-injection-like）文本的“仅标记”结果。

    ISSUE-003 仅要求进行风险标记，而非拦截。
    若在此处实施拦截，将会静默拒绝关于提示词注入的合法诊断性提问，并改变对话语义，从而超出当前议题的范畴。
    """

    detected: bool
    markers: tuple[str, ...]


@dataclass(frozen=True)
class GuardResult(Generic[T]):
    """经验证的值，以及用于追踪或日志记录的非敏感元数据。"""

    value: T
    input_length: int
    prompt_injection_risk: bool = False
    risk_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedChatInput:
    """已归一化的聊天输入，可安全传递至智能体（Agent）层。"""

    session_id: str
    question: str


@dataclass(frozen=True)
class ValidatedAIOpsInput:
    """已归一化的 AIOps 输入，用于现有的 SSE 诊断流程。"""

    session_id: str


@dataclass(frozen=True)
class ValidatedClearInput:
    """已归一化的清除会话输入，用于旧版聊天管理 API。"""

    session_id: str


@dataclass(frozen=True)
class ValidatedFileInput:
    """已验证的上传文件，API 层可直接写入目标路径。"""

    filename: str
    extension: str
    content: bytes
    target_path: Path
    size: int


@dataclass(frozen=True)
class ValidatedDirectoryInput:
    """已验证的目录索引请求，路径已解析到 allowlist 内。"""

    directory_path: Path
    allowed_root: Path


@dataclass(frozen=True)
class ValidatedIndexFileInput:
    """已验证的目录索引单文件，供 VectorIndexService 在入库前使用。"""

    file_path: Path
    extension: str
    size: int


class InputGuard:
    """验证 API 输入，但不执行任何业务逻辑。 

    该守卫（guard）特意不调用服务、不重写问题、不执行身份验证，也不决定回退策略。
    其职责是利用稳定的错误代码实现快速失败，并保留由 Pydantic 模型（如 `Id/Question` 和 `sessionId`）已处理的既有别名（alias）行为。
    """

    def __init__(
        self,
        *,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        max_session_chars: int = DEFAULT_MAX_SESSION_CHARS,
        upload_max_bytes: int = DEFAULT_UPLOAD_MAX_BYTES,
        allowed_extensions: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_EXTENSIONS,
    ) -> None:
        self.max_text_chars = max(1, max_text_chars)
        self.max_session_chars = max(1, max_session_chars)
        self.upload_max_bytes = max(1, upload_max_bytes)
        self.allowed_extensions = _normalize_extensions(allowed_extensions)

    def validate_chat(
        self,
        request: ChatRequest,
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedChatInput]:
        """在智能体执行之前，验证并规范化聊天请求。"""

        _ = ctx
        session_id = self.validate_session_id(request.id)
        question = _normalize_text(request.question)
        if not question:
            raise InvalidInputError(
                user_message="问题不能为空。",
                internal_message="chat question is empty after trimming",
                details={"field": "question"},
                origin_module="app.core.input_guard",
            )
        if len(question) > self.max_text_chars:
            raise RequestTooLargeError(user_message="请求内容过长。")

        risk = self.detect_prompt_injection(question)
        return GuardResult(
            value=ValidatedChatInput(session_id=session_id, question=question),
            input_length=len(question),
            prompt_injection_risk=risk.detected,
            risk_markers=risk.markers,
        )

    def validate_aiops(
        self,
        request: AIOpsRequest,
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedAIOpsInput]:
        """验证 AIOps 会话，同时保留旧的默认设置。"""

        _ = ctx
        raw_session = request.session_id
        # 当前 AIOps 旧接口把缺省或空白 session_id 当作 "default"。
        # 这里保留该行为，避免依赖旧 `request.session_id or "default"` 的调用方被破坏。
        session_id = "default" if raw_session is None or not raw_session.strip() else raw_session
        normalized_session = self.validate_session_id(session_id)
        return GuardResult(
            value=ValidatedAIOpsInput(session_id=normalized_session),
            input_length=len(normalized_session),
        )

    def validate_clear(
        self,
        request: ClearRequest,
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedClearInput]:
        """在修改对话状态之前，验证清除会话的有效载荷。"""

        _ = ctx
        session_id = self.validate_session_id(request.session_id)
        return GuardResult(
            value=ValidatedClearInput(session_id=session_id),
            input_length=len(session_id),
        )

    def validate_session_id(self, session_id: str | None) -> str:
        """返回规范化的会话 ID，或引发 `INVALID_SESSION_ID` 异常。 

            之所以采用严格的允许列表（allowlist），是因为会话 ID 目前被用作内存或线程键，
            未来还可能用于文件名、追踪过滤器或存储查找。在此阶段拒绝包含斜杠、空白字符、控制字符以及类似路径遍历特征的值，
            可以避免后续组件在处理这些 ID 时再次面临相同的路径及注入风险。
        """

        normalized = session_id.strip() if isinstance(session_id, str) else ""
        if not normalized:
            raise self._invalid_session("empty session id")
        if len(normalized) > self.max_session_chars:
            raise self._invalid_session("session id is too long")
        if not _SESSION_ID_RE.fullmatch(normalized):
            raise self._invalid_session("session id contains disallowed characters")
        return normalized

    def detect_prompt_injection(self, text: str) -> PromptInjectionRisk:
        """检测提示词注入风险特征，而不拦截请求。"""

        normalized = text.casefold()
        markers = tuple(marker for marker in _PROMPT_INJECTION_MARKERS if marker in normalized)
        return PromptInjectionRisk(detected=bool(markers), markers=markers)

    def validate_upload(
        self,
        *,
        filename: str | None,
        content: bytes,
        content_type: str | None,
        upload_dir: str | Path,
        allowed_extensions: tuple[str, ...] | list[str] | None = None,
        max_bytes: int | None = None,
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedFileInput]:
        """验证上传文件边界，返回可写入的安全目标路径。

        文件 API handler 已经负责读取 multipart 和保存文件；guard 在这里集中处理
        文件名、扩展名、MIME、UTF-8 和大小限制。这样旧 `/api/upload` 与新别名
        `/api/file/upload` 能复用同一规则，避免两个入口出现兼容差异。
        """

        _ = ctx
        safe_filename = self._validate_safe_filename(filename)
        extensions = _normalize_extensions(allowed_extensions or self.allowed_extensions)
        extension = Path(safe_filename).suffix.casefold()
        if extension not in extensions:
            raise UnsupportedFileTypeError(details={"extension": extension})

        limit = max(1, max_bytes or self.upload_max_bytes)
        size = len(content)
        if size > limit:
            raise FileTooLargeError(max_bytes=limit)

        self._validate_mime(extension, content_type)
        _decode_utf8(content)

        root = Path(upload_dir).resolve()
        target_path = (root / safe_filename).resolve(strict=False)
        if not _is_relative_to(target_path, root):
            # 即使文件名已拒绝分隔符，仍保留 resolve 后的二次校验，防止 Windows
            # drive、Unicode 规范化或未来文件名规则调整带来目录逃逸。
            raise PathTraversalBlockedError()

        return GuardResult(
            value=ValidatedFileInput(
                filename=safe_filename,
                extension=extension,
                content=content,
                target_path=target_path,
                size=size,
            ),
            input_length=size,
        )

    def validate_directory(
        self,
        directory_path: str | Path | None,
        *,
        index_allowlist: tuple[str | Path, ...] | list[str | Path],
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedDirectoryInput]:
        """验证目录索引根目录是否位于 allowlist 内。

        目录索引能够批量读取本地文件，因此必须先 resolve，再确认请求目录没有离开
        配置允许的根目录；symlink 根目录直接拒绝，避免后续遍历时跟随到未知位置。
        """

        _ = ctx
        raw_directory = Path(directory_path or "uploads")
        if _has_parent_traversal(raw_directory):
            raise PathTraversalBlockedError()
        if raw_directory.is_symlink():
            raise SymlinkNotAllowedError()

        resolved_directory = raw_directory.resolve()
        if not resolved_directory.exists() or not resolved_directory.is_dir():
            raise InvalidDirectoryError()

        allowed_roots = tuple(Path(root).resolve() for root in index_allowlist)
        for allowed_root in allowed_roots:
            if _is_relative_to(resolved_directory, allowed_root):
                return GuardResult(
                    value=ValidatedDirectoryInput(
                        directory_path=resolved_directory,
                        allowed_root=allowed_root,
                    ),
                    input_length=len(str(resolved_directory)),
                )

        raise PathTraversalBlockedError()

    def validate_index_file(
        self,
        file_path: str | Path,
        *,
        allowed_root: str | Path,
        allowed_extensions: tuple[str, ...] | list[str] | set[str] | None = None,
        max_bytes: int | None = None,
        ctx: RequestContext | None = None,
    ) -> GuardResult[ValidatedIndexFileInput]:
        """验证目录索引中的单个文件。

        批量索引必须允许“单文件失败、其它文件继续”，所以该方法抛出稳定 AppError，
        由 `VectorIndexService.index_directory` 捕获后写入 `failed_files`，而不是把
        UnicodeDecodeError 或真实路径细节直接暴露给用户。
        """

        _ = ctx
        raw_file_path = Path(file_path)
        if raw_file_path.is_symlink():
            raise SymlinkNotAllowedError()
        resolved_root = Path(allowed_root).resolve()
        resolved_file_path = raw_file_path.resolve(strict=False)
        if not _is_relative_to(resolved_file_path, resolved_root):
            raise PathTraversalBlockedError()
        if not resolved_file_path.exists() or not resolved_file_path.is_file():
            raise InvalidDirectoryError()

        extensions = _normalize_extensions(allowed_extensions or self.allowed_extensions)
        extension = resolved_file_path.suffix.casefold()
        if extension not in extensions:
            raise UnsupportedFileTypeError(details={"extension": extension})

        limit = max(1, max_bytes or self.upload_max_bytes)
        size = resolved_file_path.stat().st_size
        if size > limit:
            raise FileTooLargeError(max_bytes=limit)
        _decode_utf8(resolved_file_path.read_bytes())

        return GuardResult(
            value=ValidatedIndexFileInput(
                file_path=resolved_file_path,
                extension=extension,
                size=size,
            ),
            input_length=size,
        )

    @staticmethod
    def _invalid_session(reason: str) -> InvalidSessionIdError:
        return InvalidSessionIdError(
            user_message="会话 ID 不合法。",
            details={"reason": reason},
        )

    @staticmethod
    def _validate_safe_filename(filename: str | None) -> str:
        normalized = filename.strip() if isinstance(filename, str) else ""
        if (
            not normalized
            or "/" in normalized
            or "\\" in normalized
            or Path(normalized).name != normalized
            or _CONTROL_CHAR_RE.search(normalized)
        ):
            raise PathTraversalBlockedError()
        return normalized

    @staticmethod
    def _validate_mime(extension: str, content_type: str | None) -> None:
        if not content_type:
            return
        normalized_type = content_type.split(";", 1)[0].strip().casefold()
        allowed_types = _MIME_BY_EXTENSION.get(extension, ())
        if allowed_types and normalized_type not in allowed_types:
            raise InvalidFileMimeError()


def _normalize_text(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_extensions(extensions: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for extension in extensions:
        clean_extension = extension.strip().casefold()
        if not clean_extension:
            continue
        if not clean_extension.startswith("."):
            clean_extension = f".{clean_extension}"
        normalized.append(clean_extension)
    return tuple(dict.fromkeys(normalized))


def _decode_utf8(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        # 只暴露稳定错误码和用户文案，原始 UnicodeDecodeError 细节留在内部异常链中，
        # 避免目录批量索引的 `failed_files` 泄漏底层 codec 位置或原始字节。
        raise InvalidFileEncodingError() from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_parent_traversal(path: Path) -> bool:
    return ".." in path.parts


input_guard = InputGuard()
