"""用户记忆画像（长期记忆）：跨会话的偏好、事实与实体记忆。

链路：
- 写入：chat API 回答完成后 fire-and-forget 触发 `schedule_ingest`，
  轻量 LLM 从本轮问答中抽取记忆（沿用摘要器的敏感信息清洗），embedding 后
  以内容哈希为稳定主键 upsert 进 Milvus（同内容幂等覆盖）；
- 读取：编排层加载会话上下文时按当前问题向量召回 top-k，经
  `dataclasses.replace` 注入 `ConversationContext.user_memories`，
  最终作为 SystemMessage 进入模型 prompt（失败一律 fail-open 为空）。

存储复用现有 Milvus（独立 collection，标量过滤 user_id），不引入新组件。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from app.config import config
from app.observability.tracing import TraceLogger

_MEMORY_ID_RE = re.compile(r"[^A-Za-z0-9_-]")
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_@.:,-]{1,64}$")
_MAX_MEMORY_CHARS = 400
_VALID_MEMORY_TYPES = frozenset({"preference", "fact", "entity"})
_MAX_MEMORIES_PER_TURN = 5

_EXTRACT_PROMPT = """你是一个用户记忆抽取器。从下面这轮对话中抽取值得长期记住的用户信息。
只输出 JSON，不要任何解释：
{{"memories": [{{"type": "preference|fact|entity", "content": "简短陈述", "confidence": 0.0到1.0}}]}}

规则：
- preference：用户对回答方式的偏好（如"回答要简短"、"用中文"）
- fact：用户的客观事实（如"负责订单系统"、"服务部署在杭州机房"）
- entity：用户关注的对象（如"主要维护 svc-order 服务"）
- 没有值得记的信息就输出 {{"memories": []}}
- 不要记录一次性问题内容、密钥、token、URL

用户: {user_message}
助手: {assistant_message}"""


@dataclass(frozen=True)
class MemoryItem:
    """一条抽取出的用户记忆。"""

    memory_type: str
    content: str
    confidence: float


class UserMemoryService:
    """Milvus 用户记忆画像存取 + LLM 抽取门面。"""

    def __init__(self, *, trace_logger: TraceLogger | None = None) -> None:
        self.trace_logger = trace_logger or TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        self._client: Any | None = None
        self._collection_ready = False

    # ----- 召回（读路径，fail-open） -----

    def recall(self, user_id: str, query: str, top_k: int | None = None) -> list[str]:
        """按当前问题召回用户记忆，返回可直接注入 prompt 的文本行。"""

        if not config.user_memory_enabled or not query.strip():
            return []
        safe_user_id = _safe_user_id(user_id)
        if safe_user_id is None:
            return []

        try:
            client = self._ensure_collection()
            embedding = self._embed(query.strip())
            results = client.search(
                collection_name=config.user_memory_collection,
                data=[embedding],
                filter=f'user_id == "{safe_user_id}"',
                limit=max(1, int(top_k or config.user_memory_recall_top_k)),
                output_fields=["content", "memory_type", "confidence", "updated_at"],
                search_params={"metric_type": "COSINE"},
                timeout=5,
            )
            memories: list[str] = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
                    content = str(entity.get("content", "")).strip()
                    memory_type = str(entity.get("memory_type", "fact"))
                    if content:
                        recorded_at = _format_memory_time(entity.get("updated_at"))
                        # 带记录时间：画像无实体级覆盖机制，新旧事实可能共存，
                        # 时间戳是下游（prompt 冲突规则）判断新旧状态的唯一依据。
                        suffix = f" (recorded: {recorded_at})" if recorded_at else ""
                        memories.append(f"[{memory_type}] {content}{suffix}")
            return memories
        except Exception as exc:  # noqa: BLE001 - 召回失败不阻断主链路
            logger.debug("用户记忆召回失败（fail-open）: {}: {}", exc.__class__.__name__, exc)
            return []

    # ----- 抽取与写入（对话后异步，全链路静默降级） -----

    def ingest_turn(self, user_id: str, user_message: str, assistant_message: str) -> int:
        """抽取本轮对话中的用户记忆并写入 Milvus；返回写入条数。"""

        if not config.user_memory_enabled:
            return 0
        safe_user_id = _safe_user_id(user_id)
        if safe_user_id is None:
            return 0

        items = self._extract_memories(user_message, assistant_message)
        if not items:
            return 0

        try:
            client = self._ensure_collection()
            contents = [item.content for item in items]
            embeddings = self._embed_batch(contents)
            now_ms = int(time.time() * 1000)
            rows = [
                {
                    "memory_id": _memory_id(safe_user_id, item.content),
                    "user_id": safe_user_id,
                    "memory_type": item.memory_type,
                    "content": item.content,
                    "embedding": embedding,
                    "confidence": float(item.confidence),
                    "updated_at": now_ms,
                }
                for item, embedding in zip(items, embeddings, strict=True)
            ]
            client.upsert(collection_name=config.user_memory_collection, rows=rows)
            logger.info("用户记忆写入: user_id={}, count={}", safe_user_id, len(rows))
            return len(rows)
        except Exception as exc:  # noqa: BLE001 - 写入失败不影响已完成的主请求
            logger.debug("用户记忆写入失败（静默降级）: {}: {}", exc.__class__.__name__, exc)
            return 0

    # ----- LLM 抽取 -----

    def _extract_memories(self, user_message: str, assistant_message: str) -> list[MemoryItem]:
        prompt = _EXTRACT_PROMPT.format(
            user_message=user_message[:2000],
            assistant_message=assistant_message[:2000],
        )
        try:
            raw = self._invoke_extract_llm(prompt)
        except Exception as exc:  # noqa: BLE001 - 抽取失败等价于本轮无记忆
            logger.debug("用户记忆抽取失败（跳过）: {}: {}", exc.__class__.__name__, exc)
            return []
        return _parse_memory_items(raw)

    def _invoke_extract_llm(self, prompt: str) -> str:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.user_memory_extract_model,
            api_key=config.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            timeout=config.user_memory_extract_timeout_seconds,
            max_retries=1,
        )
        return str(llm.invoke(prompt).content)

    # ----- 基础设施 -----

    def _ensure_collection(self) -> Any:
        if self._collection_ready and self._client is not None:
            return self._client
        from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

        uri = f"http://{config.milvus_host}:{config.milvus_port}"
        client = MilvusClient(uri=uri)
        if not client.has_collection(config.user_memory_collection):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(
                        "memory_id", DataType.VARCHAR, max_length=64, is_primary=True
                    ),
                    FieldSchema("user_id", DataType.VARCHAR, max_length=64),
                    FieldSchema("memory_type", DataType.VARCHAR, max_length=16),
                    FieldSchema("content", DataType.VARCHAR, max_length=1024),
                    FieldSchema(
                        "embedding",
                        DataType.FLOAT_VECTOR,
                        dim=1024,
                    ),
                    FieldSchema("confidence", DataType.FLOAT),
                    FieldSchema("updated_at", DataType.INT64),
                ],
                description="user long-term memory profile",
            )
            client.create_collection(
                collection_name=config.user_memory_collection,
                schema=schema,
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        client.load_collection(config.user_memory_collection)
        self._client = client
        self._collection_ready = True
        return client

    def _embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from app.services.vector_embedding_service import embeddings

        vectors = embeddings.embed_documents(texts)
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError("embedding batch size mismatch")
        return [list(vector) for vector in vectors]


def schedule_ingest(
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """对话完成后调度异步记忆抽取；任何失败静默降级，不影响已返回的响应。"""

    if not config.user_memory_enabled:
        return
    if not user_message.strip() or not assistant_message.strip():
        return

    async def _run() -> None:
        try:
            await asyncio.to_thread(
                user_memory_service.ingest_turn,
                user_id,
                user_message,
                assistant_message,
            )
        except Exception:  # noqa: BLE001 - 后台任务兜底
            pass

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        # 无事件循环（同步调用上下文）：退化为直接执行，仍保持静默。
        user_memory_service.ingest_turn(user_id, user_message, assistant_message)


def _parse_memory_items(raw: str) -> list[MemoryItem]:
    """解析 LLM 输出的记忆 JSON；解析失败返回空（fail-open）。"""

    from app.memory.summarizer import _sanitize_text

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return []
        payload = json.loads(raw[start : end + 1])
        memories = payload.get("memories")
        if not isinstance(memories, list):
            return []
    except (ValueError, TypeError, AttributeError):
        return []

    items: list[MemoryItem] = []
    for entry in memories:
        if not isinstance(entry, dict):
            continue
        content = _sanitize_text(str(entry.get("content", ""))).strip()
        if not content:
            continue
        memory_type = str(entry.get("type", "fact"))
        if memory_type not in _VALID_MEMORY_TYPES:
            memory_type = "fact"
        try:
            confidence = min(1.0, max(0.0, float(entry.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        items.append(
            MemoryItem(
                memory_type=memory_type,
                content=content[:_MAX_MEMORY_CHARS],
                confidence=confidence,
            )
        )
        if len(items) >= _MAX_MEMORIES_PER_TURN:
            break
    return items


def _format_memory_time(raw: object) -> str | None:
    """把 updated_at（毫秒时间戳）转成可读日期；缺失/非法返回 None（不影响注入）。"""

    try:
        milliseconds = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if milliseconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None


def _safe_user_id(user_id: str) -> str | None:
    """归一化 user_id 并做 Milvus 表达式注入防护；非法返回 None。"""

    candidate = (user_id or "default").strip() or "default"
    if not _SAFE_USER_ID_RE.match(candidate):
        candidate = "default"
    return candidate


def _memory_id(user_id: str, content: str) -> str:
    normalized = " ".join(content.split()).casefold()
    digest = hashlib.sha1(f"{user_id}:{normalized}".encode("utf-8")).hexdigest()[:16]
    return _MEMORY_ID_RE.sub("", digest)


user_memory_service = UserMemoryService()
