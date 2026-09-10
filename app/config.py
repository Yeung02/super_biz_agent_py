"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "AegisOps Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_ms: int = 60_000
    trace_enabled: bool = True
    trace_jsonl_path: str = "logs/trace.jsonl"
    # ISSUE-029 的 metrics 开关独立于 trace 开关：回滚时可只关闭 metrics，
    # 继续保留 1A 的最小 trace，避免为了控制日志量而丢失 request_id/trace_id 排障能力。
    metrics_enabled: bool = True
    metrics_jsonl_path: str = "logs/metrics.jsonl"
    # ISSUE-033 的 evaluation runner 是离线/CI 工具，不进入线上 FastAPI 请求路径。
    # 默认 adapter 使用 dry-run，judge 默认关闭，确保验收命令在没有 Milvus、DashScope、
    # MCP server 或网络的环境里也能跑完数据集加载和轻量检索指标。真实 judge 只有显式
    # 开启时才会创建模型，避免把评估成本和不稳定性带入默认回归。
    eval_dataset_path: str = "eval_sets/rag_cases.yaml"
    eval_retrieval_k: int = 5
    eval_judge_enabled: bool = False
    eval_judge_model: str = "qwen-max"
    eval_judge_temperature: float = 0.0
    eval_judge_max_context_chars: int = 8_000
    # Agent 轨迹评测与 RAG 评测同一离线边界：默认 dry-run adapter，不进入线上
    # FastAPI 请求路径；真实轨迹（aiops adapter 会创建 AIOpsService）只在显式
    # 指定时触发，避免离线评估误连 LLM/MCP/Redis。
    agent_eval_dataset_path: str = "eval_sets/agent_cases.yaml"
    agent_eval_adapter: str = "dry-run"
    agent_eval_steps_budget: int = 8
    # ISSUE-011 的降级总开关。默认开启以满足 API 契约；紧急回滚时设为 false，
    # API 会继续返回旧的 AppError 结构化错误，而不会删除新增代码或破坏 trace 字段。
    fallback_enabled: bool = True

    # Token 预算配置
    # ISSUE-012 只建立估算、预算和裁剪边界，不改变对外 API schema。默认开启是为了
    # 让 AIOps planner/replanner 的长经验文档和执行历史先受控；紧急回滚可设为 false，
    # 调用点会尽量使用旧文本路径，但仍保留 usage/trace 代码以便后续阶段复用。
    token_budget_enabled: bool = True
    token_model_context_windows: dict[str, int] = Field(
        default_factory=lambda: {
            "qwen-max": 32768,
            "qwen-plus": 32768,
            "default": 32768,
        }
    )
    token_max_output_tokens: int = 2048
    token_min_reserved_output_tokens: int = 512
    token_summary_budget: int = 1024
    token_budget_ratios: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "rag_chat": {
                "input": 0.15,
                "history": 0.15,
                # 用户长期记忆画像注入槽位（从 history 划出），未召回时槽位闲置。
                "memory": 0.05,
                "rag_context": 0.35,
                "tool_result": 0.10,
                "output": 0.20,
            },
            "aiops_plan": {
                "input": 0.15,
                "history": 0.05,
                "rag_context": 0.35,
                "tool_result": 0.25,
                "output": 0.20,
            },
            "aiops_execute": {
                "input": 0.20,
                "history": 0.0,
                "rag_context": 0.10,
                "tool_result": 0.50,
                "output": 0.20,
            },
            "aiops_report": {
                "input": 0.10,
                "history": 0.10,
                "rag_context": 0.20,
                "tool_result": 0.40,
                "output": 0.20,
            },
        }
    )
    # 价格默认 0，避免在供应商计费口径未确认时输出伪精确成本；需要时可用环境变量或
    # 配置文件覆盖为每千 token 价格。TokenBudgetManager 会保留 estimated 标记。
    token_input_prices_per_1k: dict[str, float] = Field(default_factory=lambda: {"default": 0.0})
    token_output_prices_per_1k: dict[str, float] = Field(default_factory=lambda: {"default": 0.0})

    # ConversationManager 配置
    # ISSUE-013 默认启用业务会话门面，把 API/Agent 与 MemorySaver checkpoint 内部结构隔离。
    # 紧急回滚时可设为 false，RagAgentService 会走旧的只读/清理方法；旧 API 字段不删除。
    conversation_manager_enabled: bool = True
    # 最近 N 轮只用于给 Agent 的上下文输入；用户查询历史接口仍返回全部安全可见历史。
    conversation_recent_turns: int = 3
    # ISSUE-014 历史摘要开关。摘要只影响 ConversationManager 提供给 Agent 的内部上下文，
    # 不默认暴露到 `/api/chat/session/{session_id}`，因此关闭开关即可回到 ISSUE-013
    # 的“仅最近 N 轮”行为，不会破坏旧 API 字段或旧前端。
    conversation_summary_enabled: bool = True
    # 这里的 source_turns 按“安全可见消息数”计数，而不是 MemorySaver 原始消息数；
    # 系统消息、工具消息和非文本 payload 已被 ConversationManager 过滤，不参与触发。
    conversation_summary_max_source_turns: int = 20
    # 摘要预算独立于历史预算，避免 LLM 生成过长摘要后再次挤占最近完整轮次。
    conversation_summary_max_tokens: int = 1024
    conversation_history_db_path: str = "data/conversation_history.sqlite3"

    # ===== 记忆存储底座（生产形态：Redis 短期 + PG 长期 + Milvus 知识/画像）=====
    # checkpointer 后端：redis 为生产默认（多副本共享、AOF 持久化）；memory 仅供
    # 测试进程注入（conftest 会设置 MEMORY_CHECKPOINTER=memory），生产配置 memory
    # 属于配置错误，启动时 verify_memory_storage 不会放行 redis 检查。
    memory_checkpointer: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    # 长期记忆库（会话历史归档 + 摘要持久化），由 docker-compose.memory.yml 提供。
    postgres_dsn: str = "postgresql://aegis:aegis_secret@localhost:5432/aegisops"
    # PG 连接池（psycopg_pool）：历史读写复用连接，避免每次操作重建 TCP+认证握手。
    # timeout 是单次操作等待可用连接的上界，超时由调用方按既有 fail-open 语义降级。
    postgres_pool_min_size: int = 2
    postgres_pool_max_size: int = 10
    postgres_pool_timeout_seconds: float = 10.0
    # 硬依赖策略：启动时 Redis/PG 连接验证失败直接 fail-fast，不做静默降级。
    memory_storage_fail_fast: bool = True
    # 会话 TTL 清理：仅生产 checkpointer=redis 时启动后台任务，按 updated_at 清理
    # 过期会话的历史、摘要与 checkpoint thread。
    memory_cleanup_enabled: bool = True
    memory_cleanup_interval_hours: float = 24.0
    memory_session_ttl_days: int = 30

    # 三路写入对账补偿：record_turn 失败进内存待重试队列（有界、按年龄过期），
    # 后台周期重试；另以 checkpoint 为准对 PG 近期活跃会话做前缀对账补写。
    turn_compensation_enabled: bool = True
    turn_compensation_interval_seconds: float = 30.0
    turn_compensation_max_pending: int = 1000
    turn_compensation_max_age_hours: float = 24.0
    turn_compensation_batch_size: int = 100
    reconciliation_enabled: bool = True
    reconciliation_interval_seconds: float = 600.0
    reconciliation_max_sessions: int = 50

    # 用户记忆画像（长期记忆，Milvus collection）。
    # 对话结束后异步抽取用户偏好/事实/实体，请求时按当前问题向量召回注入。
    user_memory_enabled: bool = True
    user_memory_collection: str = "user_memories"
    user_memory_recall_top_k: int = 5
    # 抽取走轻量模型（与 intent_model 同级），失败静默降级为不抽取，不影响主请求。
    user_memory_extract_model: str = "qwen-plus"
    user_memory_extract_timeout_seconds: float = 8.0

    # AgentOrchestrator 薄编排层开关
    # ISSUE-015 只把已经完成的 guard/context/token/conversation/fallback/trace 边界串起来，
    # 不重写 RAG 或 AIOps 算法。默认开启以便新边界生效；若线上出现兼容风险，可设为 false，
    # API handler 会回到旧 service 直连路径，保留旧响应字段和旧前端解析方式。
    orchestrator_enabled: bool = True

    # Chat 入口意图识别开关。开启后 orchestrator 对 chat 输入做一次轻量 LLM 分类：
    # chitchat 短路直答（跳过检索与 Agent 工具循环）、rag_qa 走现有 RAG 管线（默认
    # 兜底）、aiops 在对话内直接触发 AIOps 诊断流。分类失败/超时/非法输出一律
    # fail-open 回 rag_qa；紧急回滚设为 false 即回到无意图识别的旧对话行为。
    intent_enabled: bool = True
    # 意图分类走轻量模型降低入口延迟；与 rag_query_rewrite_model 同级。
    intent_model: str = "qwen-plus"
    # 分类是入口的阻塞调用，超时必须远小于请求级 deadline；超时即走 rag_qa 兜底。
    intent_timeout_seconds: float = 3.0
    # LLM 分类失败（含重试后）时是否降级到高精度规则兜底。规则只识别归一化后
    # 精确匹配的寒暄与含诊断祈使词的输入，未命中一律维持 rag_qa 兜底，与无
    # 意图识别行为等价；误关后果仅是 LLM 故障期间闲聊/诊断请求走默认 RAG 路径。
    intent_rule_fallback_enabled: bool = True

    # Agent 执行边界配置
    # ISSUE-009 将原先散落在 replanner、LangGraph 和 ToolManager 中的上限显式配置化。
    # 默认值延续当前 demo 的宽松行为：业务步骤仍为 8，recursion_limit 给 LangGraph 留出
    # planner/executor/replanner 多轮切换空间；工具和整体请求超时只在超限时返回稳定错误码，
    # 不改变正常短请求和旧前端的成功路径。
    agent_max_steps: int = 8
    agent_recursion_limit: int = 30
    agent_max_tool_calls: int = 12
    tool_timeout_ms: int = 15_000
    tool_max_result_chars: int = 12_000

    # 按任务动态筛选工具子集。开启后 executor 按步骤文本、RAG Agent 按用户问题做
    # 词法相关性打分，只把相关工具绑定给 LLM，降低无关工具的 schema 干扰和 token
    # 消耗。无任何命中时回退全量工具，保证行为不劣于关闭状态；紧急回滚设为 false。
    tool_selection_enabled: bool = True
    # 筛选后保留的最大工具数（按得分截断，保持原始顺序稳定）。
    tool_selection_max_tools: int = 8
    # 永远保留的核心工具（RAG 核心检索能力与本地基础工具），防止词法失配误伤主链路。
    tool_selection_always_include: list[str] = Field(
        default_factory=lambda: ["retrieve_knowledge", "get_current_time"]
    )
    # tool_choice 强制指定。planner/replanner 生成的步骤文本通常包含工具名；当步骤
    # 只命中一个已绑定工具时，executor 首轮 LLM 调用通过 tool_choice 强制调用该工具，
    # 避免模型跳过指定工具。多命中或不命中时保持 LLM 自主选择；回滚设为 false。
    tool_choice_enabled: bool = True
    # retry 统一收口：本地工具由 ToolManager 重试，MCP 由 client interceptor 重试，
    # 两者共用同一配置，避免双重重试（ToolManager 对 source=mcp 不重试）。
    # 0 表示不重试；默认 3 次总尝试次数与旧 MCP interceptor 行为一致。
    tool_retry_count: int = 3
    tool_retry_delay_seconds: float = 1.0
    # 工具输出 schema 校验。成功结果在进入证据链前先过已注册的输出校验器；
    # 校验失败降级为不可用证据的 ToolResult。未知工具至少保证可 JSON 序列化。
    tool_output_schema_enabled: bool = True
    # MCP 工具默认输出校验：无显式注册校验器的 MCP 工具，成功结果至少要求提取
    # 内容非空（None/空串/空 content 列表判为失败，降级为不可用证据）。MCP 输出
    # 形状由远端 server 决定，不做形状级硬编码；紧急回滚设为 false。
    tool_output_schema_mcp_enabled: bool = True
    # Critic 证据链（ISSUE-A）：ToolManager 按 request 收集工具证据摘要，executor
    # 每步 drain 写入 state.tool_evidence，供后续答案级 Critic 节点核对断言。
    # 只存 prompt-safe 摘要不影响工具主路径；紧急回滚设为 false 即停止收集。
    critic_evidence_enabled: bool = True
    # 单个证据块文本上限（字符）。控制 checkpointer 体积与后续 Critic prompt 预算。
    critic_evidence_max_chars: int = 400

    # Critic 答案级自我批判（ISSUE-B）：replanner 生成响应后、END 之前插入 critic
    # 节点，按证据链核对草稿断言并做有界修订。默认开启；关闭时图路由直接 END，
    # 行为与接入前完全一致，紧急回滚只需改这一个开关。
    critic_enabled: bool = True
    # 裁判与修订共用轻量模型：核对任务输入（证据块+草稿）远小于 replanner 的
    # 全轨迹上下文，与 intent/rag_query_rewrite 的模型同级。
    critic_model: str = "qwen-plus"
    # 单次 LLM 调用（裁判/修订）的硬超时；超时即 fail-open 放行原答案。
    critic_timeout_seconds: float = 8.0
    # 节点内部修订轮数上限；耗尽即定稿。修订不回 executor/replanner（不加图环），
    # 保住 agent_max_steps 与 recursion_limit 的既有语义边界。
    critic_max_revisions: int = 1
    # 修订后是否再做一次 LLM 复审。默认关闭以控制成本；开启时每轮修订后重新
    # 裁判，accept 则提前定稿，复审失败不阻塞（定稿当前版本）。
    critic_recheck_enabled: bool = False

    # 结构化输出解析重试（planner/replanner/critic 的 with_structured_output）。
    # 解析类异常（格式错误/schema 不匹配/coercion 失败）先带错误反馈重试一次，
    # 仍失败再走各节点既有 fail-open 降级；网络/超时/provider 错误不重试。
    # 紧急回滚设为 false 即回到"解析失败直接降级"的旧行为。
    structured_output_retry_enabled: bool = True
    # 额外重试次数（不含首次调用）。默认 1 次，控制延迟放大上限。
    structured_output_retry_count: int = 1

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    # C-002 在 ISSUE-012/ISSUE-035 中标记 README/.env 与代码不一致。当前 issue 只做
    # 最小配置补齐，不改 LLMFactory/ChatQwen 调用路径，避免 token 预算开发顺手改变模型连接行为。
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    # ISSUE-023 新 RagRetriever 默认不接管旧工具路径。开启后可由后续 knowledge_tool
    # adapter/evaluation runner 显式使用；当前 API 和旧 `retrieve_knowledge` 仍保持原行为。
    new_rag_retriever_enabled: bool = True
    # candidate/final 控制检索入口召回和输出数量；ISSUE-024 起用 rag_min_score 做
    # 归一化分数过滤。默认值来自工程设计文档，回滚时可设为 0，表示保留所有候选但
    # 仍记录 raw_score/normalized_score，避免再把 L2 distance 误当用户可见相似度。
    rag_candidate_k: int = 12
    rag_final_k: int = 4
    rag_min_score: float = 0.35
    context_builder_enabled: bool = True
    citations_enabled: bool = True
    rag_query_rewrite_enabled: bool = True
    # 额外生成的检索变体数量（不含原始 query 与改写主 query）；0 表示只做单条改写。
    rag_multi_query_count: int = 3
    # 查询改写走轻量模型降低检索链路延迟；超时或失败时 fail-open 回退原始 query。
    rag_query_rewrite_model: str = "qwen-plus"
    rag_query_rewrite_timeout_seconds: float = 8.0
    # Milvus 原生 BM25 混合检索（dense + sparse，服务端 RRF 融合）。
    # 依赖 Milvus >= 2.5（BM25 Function + jieba analyzer）且 collection 需带
    # sparse_vector 字段：旧 collection 不会自动迁移，需 drop 后重新 ingest 才能生效；
    # 运行时检测不到 sparse 字段会自动回退纯 dense 检索。
    # 默认开启：错误码/精确关键词类查询依赖 BM25 路（jieba 分词精确 term 匹配），
    # 纯 dense 对 OOV token 几乎无区分度；未迁移 collection 自动回退，无可用性风险。
    rag_hybrid_search_enabled: bool = True
    # hybrid_search 服务端 RRF 平滑常数；与 RagRetriever 多路召回的 RRF k 保持同量级。
    rag_hybrid_search_rrf_k: int = 60
    # ISSUE-027 的 reranker 插口已接入 DashScope qwen3-rerank 真实重排。默认开启与
    # 查询改写（rag_query_rewrite_enabled）保持一致；重排只改变候选顺序，失败（网络/
    # 鉴权/超时/响应异常）由 RagRetriever fail-open 回退原始向量排序。紧急回滚设为
    # false 即恢复纯向量排序，不删除代码、不影响旧 API 行为。
    reranker_enabled: bool = True
    # qwen3-rerank 单条文档输入上限 4000 token、单请求最多 500 文档；重排走轻量
    # 调用，超时保持与查询改写同量级，避免拖慢检索链路。
    reranker_model: str = "qwen3-rerank"
    reranker_timeout_seconds: float = 5.0
    # ISSUE-018 默认使用稳定 doc_id/chunk_id/content_hash 作为 RAG 逻辑 ID 基础；
    # 紧急回滚时设为 false，VectorStoreManager 会恢复旧 UUID 主键生成。旧 `_source`
    # metadata 仍会保留，因此关闭开关不会破坏当前上传、目录索引或旧前端响应字段。
    stable_rag_ids_enabled: bool = True
    # 增量索引：开启后 index_single_file 先按 doc_id 查询已有 chunk 的
    # content_hash 做主键级 diff，未变更 chunk 不再重复 embedding。
    # 紧急回滚时设为 false，恢复 ISSUE-019 的全量重建路径。
    incremental_index_enabled: bool = True

    # 文档分块配置
    # chunk_secondary_size 是 Markdown 标题分片后再切分的分片上限，显式配置
    # 替代旧的 chunk_max_size*2 隐式加倍；应保持 >= chunk_max_size。
    chunk_max_size: int = 800
    chunk_secondary_size: int = 1600
    chunk_overlap: int = 100

    # 文件上传与目录索引边界配置
    # 后端在 ISSUE-004 中作为强制边界，避免旧前端 50MB 文案或外部调用绕过真实限制。
    upload_dir: str = "uploads"
    upload_max_bytes: int = 10 * 1024 * 1024
    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".markdown"]
    )
    # 目录索引只能访问显式 allowlist。默认保留 uploads 旧行为，并允许项目内 aiops-docs
    # 作为演示知识库来源；其它路径必须先经过 InputGuard 的 resolve 与越界校验。
    index_allowed_directories: list[str] = Field(default_factory=lambda: ["uploads", "aiops-docs"])

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 工具权限策略配置
    # ISSUE-008 默认开启 ToolManager 包装，让 RAG/AIOps 工具调用经过统一 timeout、
    # 权限、裁剪和 trace 边界；如果 LangGraph ToolNode schema 兼容出现问题，可设为
    # false 回到旧的 `local_tools + mcp_tools` 原始列表，保护当前前端和 demo 行为。
    tool_manager_enabled: bool = True
    # ISSUE-007 默认开启策略，但 allowlist 必须覆盖当前 demo 的本地工具和 mock MCP 工具；
    # 否则后续 ToolManager 接入后会因为“默认拒绝未知工具”导致现有 AIOps/RAG 演示不可用。
    # 紧急回滚时可设置 tool_policy_enabled=false，ToolManager 会退回 ISSUE-006 的全允许行为。
    tool_policy_enabled: bool = True
    tool_default_allowlist: list[str] = Field(
        default_factory=lambda: [
            "retrieve_knowledge",
            "get_current_time",
            "get_current_timestamp",
            "get_region_code_by_name",
            "get_topic_info_by_name",
            "search_topic_by_service_name",
            "search_log",
            "query_cpu_metrics",
            "query_memory_metrics",
        ]
    )
    tool_policy_default_allowed_tenants: list[str] = Field(default_factory=lambda: ["default"])
    tool_policy_default_allowed_users: list[str] = Field(default_factory=lambda: ["anonymous"])

    @property
    def mcp_servers(self) -> dict[str, dict[str, str]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }

    @property
    def tool_timeout_seconds(self) -> float:
        """ToolManager 使用秒级 timeout，配置文件保留毫秒便于和 API 契约对齐。"""

        return self.tool_timeout_ms / 1000


# 全局配置实例
config = Settings()
