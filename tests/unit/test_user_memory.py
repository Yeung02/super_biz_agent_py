"""用户记忆画像时间戳注入测试。

画像无实体级覆盖机制，新旧事实（如"在北京"/"在上海"）会共存。recall() 必须把
updated_at 一并召回并转成可读日期，下游 prompt 冲突规则才有判断新旧的依据；
时间字段缺失/非法时不得影响记忆行注入（fail-open）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import config
from app.memory.user_memory import UserMemoryService, _format_memory_time


# ----- _format_memory_time 容错 -----


def test_format_memory_time_converts_milliseconds_to_date() -> None:
    """毫秒时间戳应转成 YYYY-MM-DD（UTC）。"""

    assert _format_memory_time(1755000000000) == "2025-08-12"


def test_format_memory_time_returns_none_for_invalid_values() -> None:
    """None/非法字符串/非正数返回 None，不抛异常。"""

    assert _format_memory_time(None) is None
    assert _format_memory_time("not-a-number") is None
    assert _format_memory_time(0) is None
    assert _format_memory_time(-1) is None


# ----- recall() 时间戳注入 -----


class _FakeMilvusClient:
    """fake Milvus client：记录 search 参数并返回固定 hits。"""

    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits
        self.search_calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
        self.search_calls.append(kwargs)
        return [self.hits]


def _make_service(monkeypatch: pytest.MonkeyPatch, hits: list[dict[str, object]]) -> tuple[UserMemoryService, _FakeMilvusClient]:
    monkeypatch.setattr(config, "user_memory_enabled", True, raising=False)
    service = UserMemoryService()
    client = _FakeMilvusClient(hits)
    monkeypatch.setattr(service, "_ensure_collection", lambda: client)
    monkeypatch.setattr(service, "_embed", lambda text: [0.0] * 4)
    return service, client


def test_recall_requests_updated_at_and_appends_recorded_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recall() 应请求 updated_at 字段，并在记忆行尾追加 (recorded: 日期)。"""

    hits = [
        {
            "entity": {
                "content": "用户常驻上海",
                "memory_type": "fact",
                "updated_at": 1755000000000,
            }
        }
    ]
    service, client = _make_service(monkeypatch, hits)

    memories = service.recall("default", "用户在哪个城市")

    assert memories == ["[fact] 用户常驻上海 (recorded: 2025-08-12)"]
    assert "updated_at" in client.search_calls[0]["output_fields"]


def test_recall_omits_suffix_when_updated_at_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧数据或字段缺失时记忆行不带时间后缀，仍正常注入（fail-open）。"""

    hits = [
        {
            "entity": {
                "content": "用户常驻北京",
                "memory_type": "fact",
                "updated_at": None,
            }
        }
    ]
    service, _client = _make_service(monkeypatch, hits)

    memories = service.recall("default", "用户在哪个城市")

    assert memories == ["[fact] 用户常驻北京"]


# ----- prompt 注入冲突规则 -----


def test_build_messages_contains_conflict_resolution_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注入画像的 SystemMessage 必须声明冲突规则：记忆间以最新日期为准、与当前对话冲突以对话为准。"""

    from app.services import rag_agent_service as rag_module

    monkeypatch.setattr(rag_module, "ChatQwen", _FakeChatQwen)
    monkeypatch.setattr(rag_module, "get_mcp_tools_with_retry", _fake_no_mcp_tools)

    service = rag_module.RagAgentService(streaming=False)
    context = SimpleNamespace(
        summary=None,
        recent_messages=(),
        history_metadata={},
        user_memories=(
            "[fact] 用户常驻北京 (recorded: 2025-01-01)",
            "[fact] 用户常驻上海 (recorded: 2025-08-12)",
        ),
    )

    messages = service._build_messages("用户现在在哪个城市", context)

    memory_messages = [m for m in messages if "User long-term memory" in str(m.content)]
    assert len(memory_messages) == 1
    content = str(memory_messages[0].content)
    assert "latest recorded date" in content
    assert "current conversation wins" in content
    assert "(recorded: 2025-01-01)" in content
    assert "(recorded: 2025-08-12)" in content


class _FakeChatQwen:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs


async def _fake_no_mcp_tools() -> list[object]:
    return []
