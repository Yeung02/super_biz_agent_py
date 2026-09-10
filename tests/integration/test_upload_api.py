"""ISSUE-032 上传和目录索引 API 集成测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pytest
from fastapi.testclient import TestClient


class _IntegrationVectorIndexService(Protocol):
    """测试只读取 fake 的调用记录和模式开关，避免直接 import conftest。"""

    single_file_calls: list[str]
    directory_calls: list[tuple[str, str | None]]
    single_file_mode: str


def test_upload_success_keeps_legacy_schema_and_trace(
    integration_client: TestClient,
    integration_vector_index_service: _IntegrationVectorIndexService,
) -> None:
    """旧上传路径必须通过真实 multipart 进入 File API，并保留旧响应字段。"""

    response = integration_client.post(
        "/api/upload",
        files={"file": ("runbook.md", b"# CPU\ncheck process cpu\n", "text/markdown")},
        headers={"X-Trace-Id": "trace-upload-ok", "X-Request-Id": "request-upload-ok"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["code"] == 200
    assert body["message"] == "success"
    assert body["data"]["filename"] == "runbook.md"
    assert body["data"]["size"] == len(b"# CPU\ncheck process cpu\n")
    assert body["data"]["indexing"]["success"] is True
    assert body["trace_id"] == "trace-upload-ok"
    assert body["request_id"] == "request-upload-ok"
    assert response.headers["X-Trace-Id"] == "trace-upload-ok"
    assert len(integration_vector_index_service.single_file_calls) == 1


def test_file_upload_alias_matches_legacy_response_schema(
    integration_client: TestClient,
) -> None:
    """新 `/api/file/upload` 只能作为旧上传路径别名，响应结构不能分叉。"""

    legacy_response = _post_upload(integration_client, "/api/upload", "legacy.md")
    alias_response = _post_upload(integration_client, "/api/file/upload", "alias.md")

    legacy_body = legacy_response.json()
    alias_body = alias_response.json()
    assert legacy_response.status_code == 200
    assert alias_response.status_code == 200
    assert _shape(legacy_body) == _shape(alias_body)
    assert set(legacy_body["data"]) == set(alias_body["data"])
    assert set(legacy_body["data"]["indexing"]) == set(alias_body["data"]["indexing"])


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_status", "expected_code", "max_bytes"),
    [
        ("bad.pdf", b"plain text", "text/plain", 400, "UNSUPPORTED_FILE_TYPE", None),
        ("bad.md", b"# title\n", "application/json", 415, "INVALID_FILE_MIME", None),
        ("bad.md", b"\xff\xfe\x00", "text/markdown", 400, "INVALID_FILE_ENCODING", None),
        ("too-large.md", b"12345", "text/markdown", 413, "FILE_TOO_LARGE", 4),
    ],
)
def test_upload_security_errors_return_stable_codes(
    integration_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: bytes,
    content_type: str,
    expected_status: int,
    expected_code: str,
    max_bytes: int | None,
) -> None:
    """上传扩展名、MIME、UTF-8 和大小错误必须映射到契约错误码。"""

    if max_bytes is not None:
        from app.api import file as file_api

        monkeypatch.setattr(file_api.config, "upload_max_bytes", max_bytes)

    response = integration_client.post(
        "/api/upload",
        files={"file": (filename, content, content_type)},
        headers={"X-Trace-Id": f"trace-{expected_code}", "X-Request-Id": "request-upload-error"},
    )

    _assert_error_response(
        response,
        expected_status=expected_status,
        expected_code=expected_code,
        expected_trace=f"trace-{expected_code}",
    )


def test_index_directory_accepts_legacy_query_and_returns_trace(
    integration_client: TestClient,
    tmp_path: Path,
    integration_vector_index_service: _IntegrationVectorIndexService,
) -> None:
    """旧目录索引 query 参数必须继续可用，并保留 IndexingResult 旧字段。"""

    upload_root = _upload_root(tmp_path)
    (upload_root / "runbook.md").write_text("# CPU\ncheck process cpu\n", encoding="utf-8")

    response = integration_client.post(
        "/api/index_directory",
        params={"directory_path": str(upload_root)},
        headers={"X-Trace-Id": "trace-index-query", "X-Request-Id": "request-index-query"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["code"] == 200
    assert body["message"] == "success"
    assert body["data"]["success"] is True
    assert body["data"]["total_files"] == 1
    assert body["data"]["success_count"] == 1
    assert body["data"]["fail_count"] == 0
    assert body["data"]["failed_files"] == {}
    assert body["trace_id"] == "trace-index-query"
    assert body["request_id"] == "request-index-query"
    assert integration_vector_index_service.directory_calls == [(str(upload_root), str(upload_root))]


def test_file_index_directory_alias_accepts_json_body(
    integration_client: TestClient,
    tmp_path: Path,
) -> None:
    """新目录索引别名必须接受 JSON body，并与旧目录响应保持同形。"""

    upload_root = _upload_root(tmp_path)
    (upload_root / "body.md").write_text("# Memory\ncheck rss\n", encoding="utf-8")

    legacy_response = integration_client.post(
        "/api/index_directory",
        json={"directory_path": str(upload_root)},
    )
    alias_response = integration_client.post(
        "/api/file/index_directory",
        json={"directory_path": str(upload_root)},
    )

    assert legacy_response.status_code == 200
    assert alias_response.status_code == 200
    assert _shape(legacy_response.json()) == _shape(alias_response.json())
    assert alias_response.json()["data"]["total_files"] == 1


@pytest.mark.parametrize(
    ("path_value", "expected_codes"),
    [
        ("../outside", {"PATH_TRAVERSAL_BLOCKED"}),
        ("outside-absolute", {"PATH_TRAVERSAL_BLOCKED", "INVALID_DIRECTORY"}),
    ],
)
def test_index_directory_rejects_traversal_and_allowlist_escape(
    integration_client: TestClient,
    tmp_path: Path,
    path_value: str,
    expected_codes: set[str],
) -> None:
    """目录索引必须拒绝 `../` 和 allowlist 外目录，不进入索引服务。"""

    requested_path = path_value
    if path_value == "outside-absolute":
        outside = tmp_path / "outside"
        outside.mkdir()
        requested_path = str(outside)

    response = integration_client.post(
        "/api/file/index_directory",
        json={"directory_path": requested_path},
        headers={"X-Trace-Id": "trace-index-blocked", "X-Request-Id": "request-index-blocked"},
    )

    body = response.json()
    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] in expected_codes
    assert body["trace_id"] == "trace-index-blocked"
    assert body["request_id"] == "request-index-blocked"


def test_index_directory_rejects_symlink_root(
    integration_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录根为 symlink 时必须返回 `SYMLINK_NOT_ALLOWED`。

    Windows 普通权限通常不能创建真实 symlink；这里仅对本测试路径定点替换
    `Path.is_symlink`，仍然通过真实 FastAPI request 触发 InputGuard 的拒绝分支。
    """

    upload_root = _upload_root(tmp_path)
    symlink_root = upload_root / "link-root"
    original_is_symlink = Path.is_symlink

    def _fake_is_symlink(path: Path) -> bool:
        if path == symlink_root:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", _fake_is_symlink)

    response = integration_client.post(
        "/api/index_directory",
        json={"directory_path": str(symlink_root)},
        headers={"X-Trace-Id": "trace-index-symlink", "X-Request-Id": "request-index-symlink"},
    )

    _assert_error_response(
        response,
        expected_status=400,
        expected_code="SYMLINK_NOT_ALLOWED",
        expected_trace="trace-index-symlink",
    )


def test_index_directory_partial_success_keeps_failed_files(
    integration_client: TestClient,
    tmp_path: Path,
) -> None:
    """目录中部分文件失败时必须返回 200 partial_success 和旧 `failed_files` map。"""

    upload_root = _upload_root(tmp_path)
    (upload_root / "good.md").write_text("# OK\nvalid content\n", encoding="utf-8")
    (upload_root / "bad.md").write_bytes(b"\xff\xfe\x00")

    response = integration_client.post(
        "/api/file/index_directory",
        json={"directory_path": str(upload_root)},
        headers={"X-Trace-Id": "trace-index-partial", "X-Request-Id": "request-index-partial"},
    )

    body = response.json()
    failed_codes = cast(dict[str, str], body["data"]["failed_file_error_codes"])
    assert response.status_code == 200
    assert body["success"] is True
    assert body["message"] == "partial_success"
    assert body["data"]["success"] is False
    assert body["data"]["partial_success"] is True
    assert body["data"]["total_files"] == 2
    assert body["data"]["success_count"] == 1
    assert body["data"]["fail_count"] == 1
    assert body["data"]["failed_files"]
    assert set(failed_codes.values()) == {"INVALID_FILE_ENCODING"}
    assert body["trace_id"] == "trace-index-partial"
    assert body["request_id"] == "request-index-partial"


def _post_upload(client: TestClient, path: str, filename: str) -> ResponseLike:
    """发起上传请求，隐藏 multipart 样板，保留真实 HTTP 调用。"""

    return client.post(
        path,
        files={"file": (filename, b"# title\nbody\n", "text/markdown")},
    )


def _upload_root(tmp_path: Path) -> Path:
    """返回集成 fixture 配置的临时上传根目录。"""

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(exist_ok=True)
    return upload_root.resolve()


def _assert_error_response(
    response: ResponseLike,
    *,
    expected_status: int,
    expected_code: str,
    expected_trace: str,
) -> None:
    """断言文件/目录 API 的统一错误 envelope 和旧兼容字段。"""

    body = response.json()
    assert response.status_code == expected_status
    assert body["success"] is False
    assert body["code"] == expected_status
    assert body["data"]["success"] is False
    assert body["data"]["errorMessage"]
    assert body["error"]["code"] == expected_code
    assert body["error"]["trace_id"] == expected_trace
    assert body["trace_id"] == expected_trace
    assert "http://internal" not in response.text


def _shape(value: object) -> object:
    """把响应值转换为结构指纹，用于比较新旧别名 schema。"""

    if isinstance(value, dict):
        return {str(key): _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


class ResponseLike(Protocol):
    """测试 helper 需要的最小 TestClient response 协议。"""

    status_code: int
    text: str

    def json(self) -> dict[str, object]:
        """返回 JSON body。"""
