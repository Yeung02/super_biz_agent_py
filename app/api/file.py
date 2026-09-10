"""文件上传与目录索引 API。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Body, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.config import config
from app.core.errors import AppError, FileTooLargeError, JsonObject, JsonValue
from app.core.input_guard import input_guard
from app.core.request_context import RequestContext, get_request_context_or_none
from app.services.vector_index_service import vector_index_service

router = APIRouter()
_DOWNSTREAM_INDEX_ERROR_CODES = {"VECTOR_STORE_UNAVAILABLE", "EMBEDDING_PROVIDER_ERROR"}


class IndexDirectoryRequest(BaseModel):
    """目录索引 JSON 请求体。

    旧接口只支持 query 参数；ISSUE-004 增加 JSON body，同时保留 query 兼容。
    当二者同时出现时使用 body，避免新调用方被旧 query 默认值意外覆盖。
    """

    directory_path: str | None = None


@router.post("/file/upload")
@router.post("/upload")
async def upload_file(http_request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """上传文件并自动创建向量索引。

    handler 只负责读取 multipart、调用 InputGuard、保存文件和触发已有索引服务；
    文件名、MIME、UTF-8、大小与路径越界都由 guard 统一处理，避免 API 层堆业务规则。
    """

    ctx = get_request_context_or_none(http_request)
    try:
        # UploadFile 在部分客户端会携带 size。先做一次轻量前置判断，避免明显超限文件
        # 被完整读入内存；读完后仍交给 InputGuard 复核真实字节数，防止客户端谎报大小。
        reported_size = getattr(file, "size", None)
        if isinstance(reported_size, int) and reported_size > config.upload_max_bytes:
            raise FileTooLargeError(max_bytes=config.upload_max_bytes)

        content = await file.read()
        guarded = input_guard.validate_upload(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
            upload_dir=config.upload_dir,
            allowed_extensions=config.allowed_upload_extensions,
            max_bytes=config.upload_max_bytes,
            ctx=ctx,
        )
        validated = guarded.value
        validated.target_path.parent.mkdir(parents=True, exist_ok=True)

        if validated.target_path.exists():
            logger.info(f"文件已存在，将覆盖更新: {validated.target_path}")
            validated.target_path.unlink()

        validated.target_path.write_bytes(validated.content)
        logger.info(
            "文件上传成功: path={}, size={}, extension={}",
            validated.target_path,
            validated.size,
            validated.extension,
        )

        indexing: JsonObject = {"success": True}
        try:
            vector_index_service.index_single_file(str(validated.target_path))
            logger.info(f"向量索引创建成功: {validated.target_path}")
        except Exception as exc:
            # The file is already persisted, so keep HTTP 200 while exposing a sanitized
            # partial indexing failure for compatible retry flows.
            app_error = AppError.from_exception(exc, origin_module="app.api.file.upload.index")
            logger.error(f"上传后索引失败: {validated.target_path}, code={app_error.code}")
            indexing = {
                "success": False,
                "errorMessage": app_error.user_message,
                "error": {
                    "code": app_error.code,
                    "message": app_error.user_message,
                    "retryable": app_error.retryable,
                    "fallback_required": app_error.fallback_required,
                },
            }

        return _success_response(
            message="success" if indexing["success"] is True else "partial_success",
            data={
                "filename": validated.filename,
                "file_path": str(validated.target_path),
                "size": validated.size,
                "indexing": indexing,
            },
            ctx=ctx,
        )

    except AppError as exc:
        logger.warning(f"文件上传被拒绝: {exc.code}")
        return _error_response(exc, ctx)
    except Exception as exc:
        app_error = AppError.from_exception(exc, origin_module="app.api.file.upload")
        logger.error(f"文件上传失败: {app_error.code}")
        return _error_response(app_error, ctx)


@router.post("/file/index_directory")
@router.post("/index_directory")
async def index_directory(
    http_request: Request,
    payload: IndexDirectoryRequest | None = Body(default=None),
    directory_path: str | None = Query(default=None),
) -> JSONResponse:
    """索引指定目录下的所有支持文件。

    JSON body 是规范化入口，query 参数是旧接口兼容入口。二者共存时 body 优先，
    这是契约中对新旧调用方最小惊扰的策略。
    """

    ctx = get_request_context_or_none(http_request)
    try:
        requested_directory = (
            payload.directory_path
            if payload is not None and payload.directory_path is not None
            else directory_path
        )
        guarded = input_guard.validate_directory(
            requested_directory,
            index_allowlist=config.index_allowed_directories,
            ctx=ctx,
        )
        validated = guarded.value
        logger.info(
            "开始索引目录: path={}, allowed_root={}",
            validated.directory_path,
            validated.allowed_root,
        )

        result = vector_index_service.index_directory(
            str(validated.directory_path),
            allowed_root=validated.allowed_root,
        )

        if _should_return_indexing_error(result):
            app_error = AppError(result.error_code)
            return _error_response(app_error, ctx, legacy_data=result.to_dict())

        return _success_response(
            message="success" if result.success else "partial_success",
            data=result.to_dict(),
            ctx=ctx,
        )

    except AppError as exc:
        logger.warning(f"目录索引被拒绝: {exc.code}")
        return _error_response(exc, ctx)
    except Exception as exc:
        app_error = AppError.from_exception(exc, origin_module="app.api.file.index_directory")
        logger.error(f"索引目录失败: {app_error.code}")
        return _error_response(app_error, ctx)


def _success_response(
    *,
    message: str,
    data: dict[str, JsonValue],
    ctx: RequestContext | None,
) -> JSONResponse:
    trace = _trace_fields(ctx)
    content: JsonObject = {
        "success": True,
        "code": 200,
        "message": message,
        "data": data,
        **trace,
    }
    return JSONResponse(
        status_code=200,
        content=content,
        headers={"X-Trace-Id": trace["trace_id"], "X-Request-Id": trace["request_id"]},
    )


def _error_response(
    error: AppError,
    ctx: RequestContext | None,
    *,
    legacy_data: JsonObject | None = None,
) -> JSONResponse:
    trace = _trace_fields(ctx)
    response = error.to_json_response(**trace, legacy_data=legacy_data)
    response.headers["X-Trace-Id"] = trace["trace_id"]
    response.headers["X-Request-Id"] = trace["request_id"]
    return response


def _trace_fields(ctx: RequestContext | None) -> dict[str, str]:
    if ctx is not None:
        return {"trace_id": ctx.trace_id, "request_id": ctx.request_id}
    return {"trace_id": f"trc_{uuid.uuid4().hex}", "request_id": f"req_{uuid.uuid4().hex}"}


def _should_return_indexing_error(result: object) -> bool:
    """Promote systemic all-file index failures to the documented error envelope."""

    return (
        getattr(result, "success", False) is False
        and getattr(result, "success_count", 0) == 0
        and getattr(result, "fail_count", 0) > 0
        and getattr(result, "error_code", "") in _DOWNSTREAM_INDEX_ERROR_CODES
    )
