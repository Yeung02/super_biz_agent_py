from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_chat_sessions_error_log_is_readable_chinese():
    chat_api = (ROOT / "app" / "api" / "chat.py").read_text(encoding="utf-8")

    assert "获取会话列表错误" in chat_api
    assert "鑾峰彇浼氳瘽鍒楄〃閿欒" not in chat_api
