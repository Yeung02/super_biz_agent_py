"""ISSUE-030 Health API 集成测试。

Health 测试使用真实 FastAPI TestClient 和 RequestContextMiddleware，但 Milvus
lifecycle/health 由内存 fake 接管，避免测试依赖本机 Milvus 或网络状态。
"""

from __future__ import annotations

from typing import Protocol

from fastapi.testclient import TestClient


class _IntegrationMilvusManager(Protocol):
    """测试只读取 fake 状态字段，避免直接 import conftest 造成重复加载。"""

    healthy: bool
    health_calls: int


def test_api_health_success_returns_trace_and_legacy_envelope(
    integration_client: TestClient,
    integration_milvus_manager: _IntegrationMilvusManager,
) -> None:
    """`/api/health` 成功时必须保留 code/message/data 并带 trace/request。"""

    response = integration_client.get(
        "/api/health",
        headers={"X-Trace-Id": "trace-health-ok", "X-Request-Id": "request-health-ok"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["code"] == 200
    assert body["message"] == "service healthy"
    assert body["data"]["service"] == "AegisOps Agent"
    assert body["data"]["status"] == "healthy"
    assert body["data"]["milvus"]["status"] == "connected"
    assert body["trace_id"] == "trace-health-ok"
    assert body["request_id"] == "request-health-ok"
    assert response.headers["X-Trace-Id"] == "trace-health-ok"
    assert integration_milvus_manager.health_calls == 1


def test_api_health_failure_returns_vector_store_error_without_raw_exception(
    integration_client: TestClient,
    integration_milvus_manager: _IntegrationMilvusManager,
) -> None:
    """Milvus 不可用时返回 503 和稳定错误码，不泄漏内部异常全文。"""

    integration_milvus_manager.healthy = False
    response = integration_client.get(
        "/api/health",
        headers={"X-Trace-Id": "trace-health-fail", "X-Request-Id": "request-health-fail"},
    )

    body = response.json()
    assert response.status_code == 503
    assert body["success"] is False
    assert body["code"] == 503
    assert body["data"]["status"] == "unhealthy"
    assert body["data"]["milvus"]["status"] == "disconnected"
    assert body["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"
    assert body["error"]["trace_id"] == "trace-health-fail"
    assert body["request_id"] == "request-health-fail"
    assert "http://internal" not in response.text


def test_root_html_requires_cache_revalidation(integration_client: TestClient) -> None:
    """The SPA shell should not be reused stale after static asset updates."""

    response = integration_client.get("/")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert 'styles.css?v=aegisops-agent-history-db' in response.text
    assert 'app.js?v=aegisops-agent-history-db' in response.text
