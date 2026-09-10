"""API handler 的统一错误响应接入测试。

这些用例只验证 ISSUE-001 的最小接入：handler 捕获异常后通过 AppError 输出安全
envelope，并保留旧前端依赖字段。不启动主应用 lifespan，避免连接真实 Milvus、
DashScope 或 MCP server。
"""

import importlib
import sys
import types
from collections.abc import AsyncGenerator
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.request_context import RequestContextMiddleware
from app.observability.tracing import TraceLogger


class FailingRagService:
    async def query(self, question: str, session_id: str) -> str:
        raise RuntimeError("dashscope_api_key=sk-secret failed at http://internal.service")


class StreamingErrorRagService:
    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        yield {
            "type": "error",
            "data": RuntimeError("token=sk-secret failed at http://internal.stream"),
        }


class StreamingDoneRagService:
    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, object], None]:
        yield {"type": "content", "data": "hello"}
        yield {"type": "complete", "data": {"answer": "hello"}}


class FailingAIOpsService:
    async def diagnose(self, session_id: str = "default") -> AsyncGenerator[dict[str, str], None]:
        raise RuntimeError("token=sk-secret failed at http://internal.aiops")
        yield {"type": "unreachable"}


class CompleteAIOpsService:
    async def diagnose(self, session_id: str = "default") -> AsyncGenerator[dict[str, str], None]:
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": "complete",
        }


class StubVectorIndexService:
    def index_directory(
        self,
        directory_path: str | None = None,
        *,
        allowed_root: str | Path | None = None,
    ) -> None:
        _ = allowed_root
        return None

    def index_single_file(self, file_path: str) -> None:
        return None


class FailingUploadVectorIndexService:
    def index_single_file(self, file_path: str) -> None:
        raise RuntimeError("token=sk-secret failed at http://internal.index")


class FailedDirectoryResult:
    success = False
    status = "failed"
    success_count = 0
    fail_count = 1
    error_code = "EMBEDDING_PROVIDER_ERROR"
    error_message = "向量化服务暂时不可用。"

    def __init__(self, directory_path: str) -> None:
        self.directory_path = directory_path
        self.failed_files = {f"{directory_path}/bad.md": "向量化服务暂时不可用。"}

    def to_dict(self) -> dict[str, object]:
        return {
            "success": False,
            "directory_path": self.directory_path,
            "total_files": 1,
            "success_count": 0,
            "fail_count": 1,
            "duration_ms": 0,
            "error_message": self.error_message,
            "errorMessage": self.error_message,
            "failed_files": self.failed_files,
            "status": self.status,
            "partial_success": False,
            "failed_file_count": 1,
            "failed_file_error_codes": dict.fromkeys(self.failed_files, self.error_code),
            "error_code": self.error_code,
        }


class AllFailedEmbeddingDirectoryService:
    def index_directory(
        self,
        directory_path: str | None = None,
        *,
        allowed_root: str | Path | None = None,
    ) -> FailedDirectoryResult:
        _ = allowed_root
        return FailedDirectoryResult(directory_path or "")


class FailingMilvusManager:
    def connect(self) -> None:
        raise RuntimeError("token=sk-secret failed at http://internal.milvus")

    def health_check(self) -> bool:
        raise RuntimeError("token=sk-secret failed at http://internal.health")

    def close(self) -> None:
        return None


def _load_api_module(monkeypatch: MonkeyPatch, module_name: str) -> ModuleType:
    """导入 API router 前替换重依赖 service，避免测试连接真实外部系统。"""

    rag_module = types.ModuleType("app.services.rag_agent_service")
    rag_module.rag_agent_service = object()
    aiops_module = types.ModuleType("app.services.aiops_service")
    aiops_module.aiops_service = object()
    vector_module = types.ModuleType("app.services.vector_index_service")
    vector_module.vector_index_service = StubVectorIndexService()

    monkeypatch.setitem(sys.modules, "app.services.rag_agent_service", rag_module)
    monkeypatch.setitem(sys.modules, "app.services.aiops_service", aiops_module)
    monkeypatch.setitem(sys.modules, "app.services.vector_index_service", vector_module)
    sys.modules.pop(f"app.api.{module_name}", None)
    return importlib.import_module(f"app.api.{module_name}")


def test_chat_handler_returns_safe_error_envelope(monkeypatch: MonkeyPatch) -> None:
    chat = _load_api_module(monkeypatch, "chat")
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    monkeypatch.setattr(chat, "rag_agent_service", FailingRagService())

    response = TestClient(app).post(
        "/api/chat",
        json={"Id": "session-1", "Question": "hello"},
    )

    body = response.json()
    assert response.status_code == 500
    assert body["success"] is False
    assert body["code"] == 500
    assert body["data"]["success"] is False
    assert body["data"]["answer"] is None
    assert body["data"]["errorMessage"] == "服务内部错误。"
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["trace_id"].startswith("trc_")
    assert "sk-secret" not in response.text
    assert "http://internal.service" not in response.text


def test_upload_http_exception_returns_error_envelope(monkeypatch: MonkeyPatch) -> None:
    file = _load_api_module(monkeypatch, "file")
    app = FastAPI()
    app.include_router(file.router, prefix="/api")

    response = TestClient(app).post(
        "/api/upload",
        files={"file": ("bad.exe", b"hello", "application/octet-stream")},
    )

    body = response.json()
    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert body["data"]["errorMessage"] == "不支持的文件类型。"
    assert "detail" not in body


def test_aiops_stream_error_event_uses_safe_envelope(monkeypatch: MonkeyPatch) -> None:
    aiops = _load_api_module(monkeypatch, "aiops")
    app = FastAPI()
    app.include_router(aiops.router, prefix="/api")
    monkeypatch.setattr(aiops, "aiops_service", FailingAIOpsService())

    with TestClient(app).stream("POST", "/api/aiops", json={"session_id": "session-1"}) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "error"' in text
    assert '"code": "SSE_STREAM_INTERRUPTED"' in text
    assert '"trace_id": "trc_' in text
    assert "sk-secret" not in text
    assert "http://internal.aiops" not in text


def test_chat_stream_error_event_uses_request_context_trace(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    chat = _load_api_module(monkeypatch, "chat")
    app = FastAPI()
    app.add_middleware(
        RequestContextMiddleware,
        trace_logger=TraceLogger(trace_jsonl_path=str(tmp_path / "trace.jsonl"), enabled=False),
    )
    app.include_router(chat.router, prefix="/api")
    monkeypatch.setattr(chat, "rag_agent_service", StreamingErrorRagService())

    with TestClient(app).stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-1", "Question": "hello"},
        headers={"X-Trace-Id": "client-trace-stream", "X-Request-Id": "client-request-stream"},
    ) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "error"' in text
    assert '"trace_id": "client-trace-stream"' in text
    assert '"request_id": "client-request-stream"' in text
    assert "sk-secret" not in text
    assert "http://internal.stream" not in text


def test_chat_stream_done_event_uses_request_context_trace(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    chat = _load_api_module(monkeypatch, "chat")
    app = FastAPI()
    app.add_middleware(
        RequestContextMiddleware,
        trace_logger=TraceLogger(trace_jsonl_path=str(tmp_path / "trace.jsonl"), enabled=False),
    )
    app.include_router(chat.router, prefix="/api")
    monkeypatch.setattr(chat, "rag_agent_service", StreamingDoneRagService())

    with TestClient(app).stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-1", "Question": "hello"},
        headers={"X-Trace-Id": "client-trace-done", "X-Request-Id": "client-request-done"},
    ) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "done"' in text
    assert '"trace_id": "client-trace-done"' in text
    assert '"request_id": "client-request-done"' in text


def test_main_import_does_not_initialize_vector_store(monkeypatch: MonkeyPatch) -> None:
    import langchain_milvus

    def fail_milvus_init(self: object, *args: object, **kwargs: object) -> None:
        _ = self, args, kwargs
        raise AssertionError("Milvus should not be initialized while importing app.main")

    monkeypatch.setattr(langchain_milvus.Milvus, "__init__", fail_milvus_init)
    for module_name in (
        "app.main",
        "app.api.aiops",
        "app.api.file",
        "app.api.health",
        "app.services.vector_index_service",
        "app.services.vector_store_manager",
        "app.tools.knowledge_tool",
        "app.tools",
    ):
        sys.modules.pop(module_name, None)
    api_package = sys.modules.get("app.api")
    if api_package is not None:
        for attr_name in ("aiops", "chat", "file", "health"):
            if hasattr(api_package, attr_name):
                delattr(api_package, attr_name)
    services_package = sys.modules.get("app.services")
    if services_package is not None:
        for attr_name in ("vector_index_service", "vector_store_manager"):
            if hasattr(services_package, attr_name):
                delattr(services_package, attr_name)

    importlib.import_module("app.main")


def test_health_failure_uses_safe_vector_store_error_envelope(monkeypatch: MonkeyPatch) -> None:
    health = importlib.import_module("app.api.health")
    app = FastAPI()
    app.include_router(health.router)
    monkeypatch.setattr(health, "milvus_manager", FailingMilvusManager())

    response = TestClient(app).get("/health")

    body = response.json()
    assert response.status_code == 503
    assert body["success"] is False
    assert body["code"] == 503
    assert body["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"
    assert body["trace_id"].startswith("trc_")
    assert "sk-secret" not in response.text
    assert "http://internal.health" not in response.text


def test_app_startup_continues_when_milvus_is_unavailable(monkeypatch: MonkeyPatch) -> None:
    app_main = importlib.import_module("app.main")
    monkeypatch.setattr(app_main, "milvus_manager", FailingMilvusManager())
    monkeypatch.setattr(app_main.health, "milvus_manager", FailingMilvusManager())

    with TestClient(app_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"


def test_upload_returns_partial_success_when_indexing_fails(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    file = _load_api_module(monkeypatch, "file")
    app = FastAPI()
    app.include_router(file.router, prefix="/api")
    monkeypatch.setattr(file.config, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(file, "vector_index_service", FailingUploadVectorIndexService())

    response = TestClient(app).post(
        "/api/upload",
        files={"file": ("notes.md", b"# hello\n", "text/markdown")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["message"] == "partial_success"
    assert body["data"]["filename"] == "notes.md"
    assert body["data"]["indexing"]["success"] is False
    assert body["data"]["indexing"]["error"]["code"] == "INTERNAL_ERROR"
    assert "sk-secret" not in response.text
    assert "http://internal.index" not in response.text


def test_index_directory_all_downstream_failures_return_error_envelope(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    file = _load_api_module(monkeypatch, "file")
    index_dir = tmp_path / "uploads"
    index_dir.mkdir()
    app = FastAPI()
    app.include_router(file.router, prefix="/api")
    monkeypatch.setattr(file.config, "index_allowed_directories", [str(tmp_path)])
    monkeypatch.setattr(file, "vector_index_service", AllFailedEmbeddingDirectoryService())

    response = TestClient(app).post(
        "/api/index_directory",
        json={"directory_path": str(index_dir)},
    )

    body = response.json()
    assert response.status_code == 502
    assert body["success"] is False
    assert body["error"]["code"] == "EMBEDDING_PROVIDER_ERROR"
    assert body["data"]["success"] is False
    assert body["data"]["directory_path"] == str(index_dir.resolve())
    assert body["data"]["failed_files"]
    assert body["trace_id"].startswith("trc_")


def test_aiops_complete_event_also_emits_done_alias(monkeypatch: MonkeyPatch) -> None:
    aiops = _load_api_module(monkeypatch, "aiops")
    app = FastAPI()
    app.include_router(aiops.router, prefix="/api")
    monkeypatch.setattr(aiops, "aiops_service", CompleteAIOpsService())

    with TestClient(app).stream("POST", "/api/aiops", json={"session_id": "session-1"}) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "complete"' in text
    assert '"type": "done"' in text
