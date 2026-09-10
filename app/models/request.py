"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    # Runtime requiredness is enforced by InputGuard instead of Pydantic so
    # missing legacy fields can return the API-contract 400 envelope rather
    # than FastAPI's default 422 validation body.
    id: str | None = Field(default=None, description="会话 ID", alias="Id")
    question: str | None = Field(default=None, description="用户问题", alias="Question")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？"
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    # Keep `sessionId` optional at the model layer for the same reason as
    # ChatRequest: InputGuard owns INVALID_SESSION_ID mapping and legacy
    # response compatibility for missing or malformed session identifiers.
    session_id: str | None = Field(default=None, description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True
