"""知识检索工具 - 从向量数据库中检索相关信息"""

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.core.token_budget import token_budget_manager
from app.rag.context_builder import ContextBuilder
from app.rag.query_rewriter import LlmQueryRewriter
from app.rag.reranker import DashScopeReranker
from app.rag.retriever import RagRetriever
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> tuple[str, list[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        if _stage3b_tool_enabled():
            return _retrieve_with_stage3b_pipeline(query)

        logger.info(f"知识检索工具被调用: query='{query}'")

        # 从向量存储中检索相关文档
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_top_k})

        docs = retriever.invoke(query)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        # 旧 raw 工具路径仍需要返回 `(content, artifact)`，以便关闭
        # `tool_manager_enabled` 后可以独立回滚；但用户/模型可见文本必须是固定安全
        # 文案，不能拼接 str(e)。ToolManager wrapper 会识别此前缀并转成
        # ToolResult 错误，防止“检索失败”文本被当成知识库事实证据。
        return "知识检索工具暂时不可用，请稍后重试。", []


def _stage3b_tool_enabled() -> bool:
    return bool(config.new_rag_retriever_enabled and config.context_builder_enabled)


def _stage3b_retriever() -> RagRetriever:
    """阶段 3C 起检索入口接入 LLM 查询改写与多路召回，qwen3-rerank 重排。

    改写与重排失败都由 RagRetriever 内部 fail-open 回退（原始 query / 向量排序），
    不会让增强能力成为知识检索工具的阻塞点。
    """

    return RagRetriever(
        query_rewriter=LlmQueryRewriter(),
        reranker=DashScopeReranker(),
    )


def _retrieve_with_stage3b_pipeline(query: str) -> tuple[str, list[Document]]:
    retrieval = _stage3b_retriever().retrieve(query)
    if retrieval.no_answer_decision is not None and not retrieval.no_answer_decision.should_answer:
        logger.warning("Stage 3B knowledge retrieval did not find enough evidence.")
        return "没有找到相关信息。", []

    budget = token_budget_manager.allocate(
        "rag_chat",
        config.rag_model,
        current_input=query,
    )
    context = ContextBuilder().build(query, retrieval.chunks, budget)
    if context.no_answer_decision is not None and not context.no_answer_decision.should_answer:
        logger.warning("Stage 3B context packing did not keep usable evidence.")
        return "没有找到相关信息。", []
    return context.context_text, []


def format_docs(docs: list[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
