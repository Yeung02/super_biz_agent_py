"""Health check API."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import config
from app.core.errors import AppError, JsonObject, VectorStoreUnavailableError
from app.core.milvus_client import milvus_manager
from app.core.request_context import RequestContext, get_request_context_or_none

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Return service health without leaking backend exception details."""

    ctx = get_request_context_or_none(request)
    health_data: JsonObject = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy",
    }

    try:
        milvus_healthy = milvus_manager.health_check()
    except Exception as exc:
        logger.warning("Milvus health check failed: code=VECTOR_STORE_UNAVAILABLE")
        health_data["milvus"] = {
            "status": "error",
            "message": "Milvus check failed",
        }
        return _unhealthy_response(
            VectorStoreUnavailableError(
                internal_message=f"{exc.__class__.__name__}: {exc}",
            ),
            health_data,
            ctx,
        )

    if not milvus_healthy:
        health_data["milvus"] = {
            "status": "disconnected",
            "message": "Milvus disconnected",
        }
        return _unhealthy_response(
            VectorStoreUnavailableError(
                internal_message="Milvus health check returned false",
            ),
            health_data,
            ctx,
        )

    health_data["milvus"] = {
        "status": "connected",
        "message": "Milvus connected",
    }
    trace = _trace_fields(ctx)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "code": 200,
            "message": "service healthy",
            "data": health_data,
            **trace,
        },
        headers={"X-Trace-Id": trace["trace_id"], "X-Request-Id": trace["request_id"]},
    )


def _unhealthy_response(
    error: AppError,
    health_data: JsonObject,
    ctx: RequestContext | None,
) -> JSONResponse:
    health_data["status"] = "unhealthy"
    health_data["error"] = error.code
    trace = _trace_fields(ctx)
    response = error.to_json_response(
        trace_id=trace["trace_id"],
        request_id=trace["request_id"],
        legacy_data=health_data,
    )
    response.headers["X-Trace-Id"] = trace["trace_id"]
    response.headers["X-Request-Id"] = trace["request_id"]
    return response


def _trace_fields(ctx: RequestContext | None) -> dict[str, str]:
    if ctx is not None:
        return {"trace_id": ctx.trace_id, "request_id": ctx.request_id}
    return {"trace_id": f"trc_{uuid.uuid4().hex}", "request_id": f"req_{uuid.uuid4().hex}"}
