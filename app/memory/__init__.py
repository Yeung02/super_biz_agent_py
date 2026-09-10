"""会话记忆边界模块。

阶段 2 从这里开始隔离业务上下文和 LangGraph MemorySaver checkpoint。外部模块应通过
ConversationManager 读取、清理和转换历史，不再直接理解底层 checkpoint 结构。
"""

from app.memory.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationTurn,
)
from app.memory.summarizer import ConversationSummarizer, SummaryResult
from app.memory.conversation_store import ConversationHistoryStore, conversation_history_store

__all__ = [
    "ConversationContext",
    "ConversationHistoryStore",
    "ConversationManager",
    "ConversationSummarizer",
    "ConversationTurn",
    "SummaryResult",
    "conversation_history_store",
]
