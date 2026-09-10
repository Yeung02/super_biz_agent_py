"""响应数据模型

定义 API 响应的 Pydantic 模型
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChatResponse(BaseModel):
    """对话响应"""

    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="会话 ID")


class SessionInfoResponse(BaseModel):
    """会话信息响应"""

    # 旧接口依赖顶层 session_id/message_count/history；阶段 1A 之后的统一 envelope
    # 只允许追加字段，因此这里增加 success，不包裹旧结构，避免破坏既有前端读取路径。
    success: bool = Field(True, description="是否成功")
    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="消息数量")
    history: List[Dict[str, str]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应"""

    # Clear API 原本只返回 status/message/data。新增 success 是 API 契约要求的统一成功
    # 标记，但保留原三字段和含义不变，避免旧前端按 status 判断时被迫迁移。
    success: bool = Field(True, description="是否成功")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Any] = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
