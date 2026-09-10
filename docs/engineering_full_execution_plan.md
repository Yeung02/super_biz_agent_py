# AegisOps Agent 工程化全局开发执行计划

## 1. 文档定位

本文档基于当前代码仓库以及以下三份契约/设计文档生成：

- `docs/agent_engineering_design.md`
- `docs/api_contract.md`
- `docs/internal_contract.md`

本文档不是阶段 1A 的局部实施计划，也不是只描述方向的路线图，而是后续开发人员可以按照顺序连续开发的全局工程化执行计划。开发顺序固定为：

```text
阶段 1A -> 阶段 1B -> 阶段 2 -> 阶段 3A -> 阶段 3B -> 阶段 4
```

执行原则：

- 以当前真实代码为基线，不把未来能力描述为当前已存在能力。
- 所有标注为“新增”的文件、目录、类当前均不存在，必须在对应 issue 中创建。
- 本计划只生成开发计划文档，不要求当前立即创建业务模块或测试文件。
- 每个 issue 的粒度按“一次独立开发、独立测试、独立提交、可独立回滚”设计。
- 不允许跳阶段实现，不允许一次性大重构，不允许删除旧 API 或旧响应字段。

Issue 编号沿用用户建议的 `ISSUE-001` 到 `ISSUE-035`，未合并、未重排。原因是现有设计文档已经按该顺序形成阶段依赖，保留编号可以降低后续任务追踪成本。

## 2. 代码基线摘要

### 2.1 当前工程形态

当前项目是 FastAPI + LangChain/LangGraph + Milvus + MCP 的 Agent Demo，具备可运行的 Chat、Chat SSE、AIOps SSE、文件上传入库、目录索引、MCP 工具接入能力，但缺少生产级边界、统一错误模型、trace、工具治理、token 预算、RAG metadata、citation、eval 和测试体系。

当前目录不是 Git worktree，`git status` 和 `git ls-files` 不可用。后续开发人员在真实仓库中执行本计划时，应在有 Git 历史的工作区按 issue 提交。

### 2.2 当前主要模块

| 模块 | 当前文件 | 当前事实 |
| --- | --- | --- |
| FastAPI 入口 | `app/main.py` | 注册 `health/chat/file/aiops` 路由，CORS 当前为 `allow_origins=["*"]`，未接入 request context middleware。 |
| Chat API | `app/api/chat.py` | `/api/chat` 返回旧字段 `code/message/data.success/answer/errorMessage`；`/api/chat_stream` 使用 `event: message` + `data.type`。 |
| AIOps API | `app/api/aiops.py` | `/api/aiops` 使用 SSE，当前 type 包括 `status/plan/step_complete/report/complete/error`。 |
| 文件 API | `app/api/file.py` | `/api/upload` 和 `/api/index_directory` 已存在；无 `/api/file/upload` 和 `/api/file/index_directory`。 |
| 请求模型 | `app/models/request.py` | `ChatRequest.id` alias 为 `Id`，`question` alias 为 `Question`，`populate_by_name=True`；`ClearRequest.session_id` alias 为 `sessionId`。 |
| AIOps 模型 | `app/models/aiops.py` | `AIOpsRequest` 只有 `session_id`，默认 `default`。 |
| RAG Agent | `app/services/rag_agent_service.py` | 通过 `create_agent` 让 LLM 条件调用 `retrieve_knowledge` 和 MCP 工具；`trim_messages_middleware` 已定义但未接入。 |
| AIOps Flow | `app/services/aiops_service.py` | LangGraph `planner -> executor -> replanner`；`graph.astream` 没有显式 `recursion_limit`。 |
| AIOps Executor | `app/agent/aiops/executor.py` | 直接构建 `ToolNode(all_tools)`，没有 ToolManager timeout、权限、裁剪和错误 envelope。 |
| AIOps Replanner | `app/agent/aiops/replanner.py` | `MAX_STEPS = 8` 是业务步骤上限，不等同于 LangGraph recursion_limit。 |
| MCP Client | `app/agent/mcp_client.py` | 有 retry interceptor，最多 3 次；失败后返回 `CallToolResult(isError=True)`；无统一 timeout/权限/schema/裁剪。 |
| 知识检索工具 | `app/tools/knowledge_tool.py` | `retrieve_knowledge` 异常时返回错误文本和空 docs，错误结果可能被当作普通上下文。 |
| 向量索引 | `app/services/vector_index_service.py` | `index_directory` 可 resolve 任意路径；仅 `*.txt` 和 `*.md`；单文件读取 UTF-8。 |
| 向量存储 | `app/services/vector_store_manager.py` | 入库 ID 使用 UUID；delete 使用 metadata `_source`；collection 固定为 `biz`。 |
| 向量检索 | `app/services/vector_search_service.py` | Milvus L2 distance 直接作为 `score` 返回，越小越相关，尚未归一化。 |
| Milvus schema | `app/core/milvus_client.py` | 字段为 `id/vector/content/metadata`，`id` 最大长度 100，`content` 最大 8000。 |
| 日志 | `app/utils/logger.py` | Loguru 文本日志，未提供 JSONL trace 或 metrics recorder。 |
| 测试 | `tests/` | 当前不存在。 |

### 2.3 当前 API 基线

| API | 当前状态 | 后续要求 |
| --- | --- | --- |
| `POST /api/chat` | 已存在 | 保留 `Id/Question`，兼容 `id/question`，追加 trace/envelope 字段。 |
| `POST /api/chat_stream` | 已存在 | 保留 `event: message` + `data.type`，可新增规范 SSE event。 |
| `POST /api/aiops` | 已存在 | 保留 `complete`，兼容映射为 `done`，追加 trace/request。 |
| `POST /api/upload` | 已存在 | 必须保留。 |
| `POST /api/file/upload` | 当前不存在 | 只能作为 `/api/upload` 的兼容别名新增。 |
| `POST /api/index_directory` | 已存在 | 必须保留。 |
| `POST /api/file/index_directory` | 当前不存在 | 只能作为 `/api/index_directory` 的兼容别名新增。 |
| `GET /api/health` | 已存在 | 保留旧 `code/message/data`，追加 trace/request。 |
| `POST /api/chat/clear` | 已存在 | 保留 `sessionId`，兼容 `session_id`。 |
| `GET /api/chat/session/{session_id}` | 已存在 | 保留顶层 `session_id/message_count/history`。 |

### 2.4 当前 RAG 基线

当前 RAG 不是稳定、确定性的 RAG pipeline，而是 LangChain Agent 自主决定是否调用 `retrieve_knowledge` 工具：

```text
POST /api/chat or /api/chat_stream
  -> app/api/chat.py
  -> RagAgentService.query/query_stream
  -> create_agent(...)
  -> LLM 条件调用 retrieve_knowledge
  -> vector_store.as_retriever(search_kwargs={"k": config.rag_top_k})
  -> Milvus collection biz
```

缺失项：

- 稳定 `doc_id/chunk_id/content_hash/tenant_id/version`。
- 索引幂等和 doc-level delete 一致性。
- score normalization。
- context packing。
- citation。
- no-answer 策略。
- reranker 插口。
- eval set、baseline 和评估报告。

## 3. 代码与文档冲突清单

| 编号 | 冲突点 | 当前代码事实 | 文档/目标要求 | 处理方案 | 计划位置 |
| --- | --- | --- | --- | --- | --- |
| C-001 | Python 版本 | `pyproject.toml` 要求 `>=3.11,<3.14` | README 写 Python 3.10+ | 阶段 4 更新 README 为 Python 3.11+，不改运行时逻辑。 | ISSUE-035 |
| C-002 | DashScope base URL | `app/config.py` 无 `dashscope_api_base`；embedding 固定 base URL | README 和 `.env` 提到 `DASHSCOPE_API_BASE` | 阶段 4 或阶段 2 前置配置修复，新增配置项并保持默认值。 | ISSUE-012, ISSUE-035 |
| C-003 | 上传扩展名 | 后端仅 `txt/md` | 前端允许 `.txt/.md/.markdown` | 阶段 1A 统一配置；若启用 `.markdown`，后端同步支持；否则返回明确错误。 | ISSUE-004 |
| C-004 | 上传大小 | 后端 10MB | 前端 50MB | 阶段 1A 以后端配置为准，并更新前端/README 文案。 | ISSUE-004, ISSUE-035 |
| C-005 | 文件路径 | 代码只有 `/api/upload` 和 `/api/index_directory` | 契约新增 `/api/file/upload` 和 `/api/file/index_directory` | 新增路径只能作为旧路径别名，不删除旧路径，响应结构必须一致。 | ISSUE-004, ISSUE-032 |
| C-006 | SSE event | 代码使用 `event: message` + `data.type` | 契约可新增规范 event `start/error/done` 等 | 新增规范 event 时必须保留 `data.type`；旧解析继续工作。 | ISSUE-009, ISSUE-030, ISSUE-031 |
| C-007 | AIOps complete | 当前 AIOps type 为 `complete` | 契约规范为 `done` | 同时发送或适配 `complete -> done`，不得删除 `complete` 兼容。 | ISSUE-009, ISSUE-031 |
| C-008 | MCP README | README 列出 `search_service_logs/list_all_services/search_historical_tickets` 等 | 当前 `cls_server.py` 和 `monitor_server.py` 未全部实现 | 阶段 4 修 README 或补 mock 工具；不在阶段 1 强行补。 | ISSUE-035 |
| C-009 | RAG score | `VectorSearchService` 返回 L2 distance 为 `score` | RAG API citation 需要 `0.0-1.0` normalized score | 阶段 3B 增加 score normalization，保留 raw_score 用于 trace。 | ISSUE-024 |
| C-010 | RAG ID | 入库 ID 为 UUID | eval/citation 需要稳定 doc/chunk ID | 阶段 3A 增加稳定生成和迁移兼容。 | ISSUE-018 |
| C-011 | 测试配置 | `pyproject.toml` 已配置 pytest coverage | 当前没有 `tests/` | 阶段 1A 建立 fake 夹具和首批单测。 | ISSUE-005 |
| C-012 | Git 状态 | 当前目录不是 Git 仓库 | 每个 issue 要能独立提交 | 执行计划保留提交顺序；实际提交在开发者真实 Git 工作区完成。 | 第 20 节 |

## 4. 执行风险与修正建议

| 风险 | 说明 | 修正建议 | 约束 |
| --- | --- | --- | --- |
| 抽象层一次性过多 | `ToolManager/AgentOrchestrator/ConversationManager/RagRetriever` 同时落地会影响主链路稳定性 | 严格按阶段接入；1A 只做边界，1B 只做工具和执行上限，2 才接薄编排 | 不允许一次性大重构 |
| 旧前端兼容破坏 | 前端依赖 `code/message/data` 和 SSE `data.type` | 所有 envelope 只能追加字段；SSE 新 event 必须保留旧解析方式 | 不删除旧字段 |
| 工具错误污染事实证据 | 当前 `retrieve_knowledge` 异常返回错误文本，可能被 LLM 当上下文 | 1B 引入 `ToolResult.is_error`；3B ContextBuilder 排除错误工具结果 | 错误结果不得作为事实证据 |
| RAG metadata 迁移风险 | Milvus collection `biz` 已有旧数据，直接变更 schema 可能破坏检索 | 3A 先兼容旧 `_source`，必要重建前备份 source docs | 阶段 3A 可回滚 |
| L2 score 语义误用 | L2 越小越相关，不能直接作为相似度 | 3B 中显式记录 `raw_score/metric`，转换为 normalized score | citation score 必须归一化 |
| LLM judge 不稳定 | judge 结果有随机性和成本 | 阶段 4 CI 先使用轻量检索指标；judge 只做报告或非强阻断 | runner 不进入线上请求路径 |
| SSE 中途失败难测 | HTTP status 已经 200，异常只能在流内表达 | 1B 增加 `start/error/done` 和 client disconnect 测试 | SSE error 必带 trace/request |
| MCP 工具边界不清 | MCP retry 已存在，但没有 timeout/权限/schema/裁剪 | 1B 只包适配层，不重写 MCP server；阶段 4 再修 README 差异 | 保留现有 MCP 能力 |

## 5. 开发总原则

1. 以当前真实代码为基线，新增能力必须写明“新增”。
2. 代码与文档冲突时，先保护当前对外兼容，再按阶段修正。
3. 每个 issue 独立开发、独立测试、独立提交、可独立回滚。
4. 不允许跳阶段实现后续能力。
5. 不允许删除旧 API。
6. 不允许删除旧响应字段。
7. 不允许一次性大重构。
8. 新增统一 envelope、trace、metrics 字段只能追加。
9. 所有失败响应和 SSE 错误必须能关联 `trace_id/request_id`。
10. 所有工具错误结果不得作为事实证据。
11. FallbackManager 只做降级决策，不做 retry。
12. TokenBudgetManager 只做预算、估算、裁剪、usage/cost 记录，不负责生成摘要正文。
13. ConversationManager 是业务会话门面，MemorySaver 只作为底层 checkpoint 存储。
14. RAG metadata 稳定后才能实现 citation 和 eval。
15. Reranker 默认关闭，失败必须回退到原始检索排序。
16. Evaluation runner 不进入线上 FastAPI 请求路径。

## 6. 全局 API 兼容约束

以下约束必须贯穿所有阶段和所有 issue：

1. `/api/upload` 必须保留，不能删除。
2. `/api/index_directory` 必须保留，不能删除。
3. `/api/file/upload` 只能作为 `/api/upload` 的兼容别名。
4. `/api/file/index_directory` 只能作为 `/api/index_directory` 的兼容别名。
5. 新旧上传/索引路径必须返回相同响应结构。
6. `/api/chat` 必须保留 `Id/Question`，同时兼容 `id/question`。
7. 同时传入大小写字段时，保持当前 Pydantic alias 行为，即 `Id/Question` 优先。
8. `POST /api/chat/clear` 必须保留 `sessionId`，同时兼容 `session_id`。
9. 旧响应字段 `code/message/data` 不允许删除。
10. `data.success/answer/errorMessage` 不允许删除。
11. `status/message/data` 不允许删除。
12. 新统一 envelope 字段只能追加，不能破坏旧前端。
13. SSE 当前 `event: message` + `data.type` 的解析方式必须兼容。
14. 规范 SSE event 可以新增，但 `data.type` 不得删除。
15. AIOps 当前 `complete` type 要兼容映射为 `done`。
16. 所有成功响应必须包含或能关联 `trace_id/request_id`。
17. 所有失败响应必须包含或能关联 `trace_id/request_id`。
18. SSE 的 `start/error/done` event 必须包含 `trace_id/request_id`。

## 7. 全局内部模块约束

| 模块/约束 | 全局要求 |
| --- | --- |
| AppError | 新增统一错误模型，所有 P0 错误必须映射到稳定 `code/http_status/retryable/fallback_required`。 |
| RequestContext | 新增 `trace_id/request_id/session_id/tenant_id/user_id/deadline_ms/feature_flags/started_at`，通过 request.state 和显式参数传播。 |
| TraceLogger | 阶段 1A 先提供最小 JSONL trace；阶段 4 扩展为完整 metrics/trace。trace 写入失败不得影响主流程。 |
| InputGuard | 统一校验 chat、aiops、clear、上传文件、目录路径、MIME、UTF-8、大小、session_id 和 prompt injection 风险标记。 |
| ToolManager | 包装本地函数、LangChain BaseTool、MCP tool；统一权限、timeout、schema、结果裁剪、trace。 |
| ToolResult | `is_error=true`、`status in ["error","timeout","unauthorized"]`、MCP `isError=true` 的结果不得作为事实证据。 |
| Agent 执行上限 | 同时控制业务步骤上限、LangGraph `recursion_limit`、最大工具调用次数、整体请求超时、单工具超时、SSE 中断。 |
| FallbackManager | 只负责降级决策和安全文案，不做 retry，不隐藏输入校验错误。 |
| TokenBudgetManager | 负责预算分配、token 估算、裁剪顺序、usage/cost 记录；当前问题和系统安全约束不得静默裁剪。 |
| ConversationManager | 负责最近 N 轮、摘要、上下文裁剪和对外历史转换；不得让业务层直接解析 MemorySaver checkpoint。 |
| ConversationSummarizer | 仅在超预算或轮次超阈值时触发；摘要不得包含工具原始 payload、密钥、堆栈或未验证事实。 |
| RAG metadata | 必须包含 `doc_id/chunk_id/content_hash/tenant_id/version/source_path/file_name`；旧 `_source` 保留兼容。 |
| 索引幂等 | 同一 tenant、source_path、content_hash 和 chunk_index 应生成稳定 ID；重复入库不重复产生残留 chunk。 |
| 删除一致性 | doc-level delete 以 `doc_id` 为主，兼容旧 `_source`；失败任务必须有状态和失败文件列表。 |
| RagRetriever | 支持 `candidate_k/final_k/min_score`，保留 raw_score 和 metric，输出 normalized score。 |
| ContextBuilder | 去重、packing、预算裁剪、prompt injection 隔离；排除错误工具结果。 |
| CitationBuilder | 内部 citation 转 API citation，只输出安全相对路径和 preview，不输出完整 chunk/绝对路径/raw metadata。 |
| No-answer | 空检索、低分检索、无可用证据、预算不足时优先拒答或 fallback，不编造。 |
| Reranker | 默认关闭；失败回退；不得成为早期交付阻塞项。 |
| Evaluation | 3A 建最小 eval baseline，4 完整 runner、LLM judge、报告和趋势；不被线上 app 默认 import。 |

## 8. 全局阶段总览

| 阶段 | 范围 | 关键产物 | 主要测试命令 |
| --- | --- | --- | --- |
| 阶段 1A | API、输入、文件、路径、错误模型、最小 trace、基础测试夹具 | `AppError`、`RequestContext`、`InputGuard`、`TraceLogger`、fake fixtures | `pytest tests/unit -q` |
| 阶段 1B | ToolManager、Agent 执行上限、MCP 工具边界、SSE 异常处理 | `ToolManager`、`ToolPolicy`、工具适配、recursion_limit、SSE error | `pytest tests/unit/test_tool_manager.py tests/agent -q` |
| 阶段 2 | Fallback、TokenBudget、ConversationManager、MemorySaver 边界 | `FallbackManager`、`TokenBudgetManager`、`ConversationManager`、`ConversationSummarizer`、`AgentOrchestrator` | `pytest tests/unit/test_fallback.py tests/unit/test_token_budget.py -q` |
| 阶段 3A | RAG metadata、索引幂等、doc/chunk ID、删除一致性、最小 eval baseline | `app/rag/models.py`、稳定 ID、幂等索引、eval set、metrics | `pytest tests/rag/test_metadata_indexing.py -q` |
| 阶段 3B | RAG Retriever、ContextBuilder、Citation、no-answer、reranker 插口 | `RagRetriever`、`ContextBuilder`、`CitationBuilder`、`NoopReranker` | `pytest tests/rag tests/unit/test_context_builder.py -q` |
| 阶段 4 | 完整可观测性、RAG 评估报告、CI 回归、指标趋势、README 更新 | `MetricsRecorder`、integration tests、`LLMJudge`、runner、README/CI 文档 | `pytest tests -q`；`python -m evaluation.runner --dataset eval_sets/rag_cases.yaml` |

## 9. 全局新增文件总表

所有条目均为“新增”，只有在对应阶段 issue 中创建。

| 阶段 | 新增文件/目录 | 说明 |
| --- | --- | --- |
| 1A | `app/core/errors.py` | 新增统一错误模型。 |
| 1A | `app/core/request_context.py` | 新增请求上下文与 middleware。 |
| 1A | `app/core/input_guard.py` | 新增输入、文件、目录路径安全校验。 |
| 1A | `app/observability/` | 新增可观测性目录。 |
| 1A | `app/observability/tracing.py` | 新增最小 JSONL trace。 |
| 1A | `tests/` | 新增测试根目录。 |
| 1A | `tests/conftest.py` | 新增 fake 夹具。 |
| 1A | `tests/unit/` | 新增单元测试目录。 |
| 1A | `tests/unit/test_app_errors.py` | 新增 AppError 单测。 |
| 1A | `tests/unit/test_request_context.py` | 新增 RequestContext 单测。 |
| 1A | `tests/unit/test_trace_logger.py` | 新增 TraceLogger 单测。 |
| 1A | `tests/unit/test_input_guard.py` | 新增 InputGuard 单测。 |
| 1B | `app/agent/tool_manager.py` | 新增 ToolManager 和 ToolResult。 |
| 1B | `app/agent/policies.py` | 新增 ToolPolicy 和 PolicyRegistry。 |
| 1B | `tests/unit/test_tool_manager.py` | 新增 ToolManager 单测。 |
| 1B | `tests/unit/test_tool_policy.py` | 新增工具策略单测。 |
| 1B | `tests/agent/` | 新增 Agent 边界测试目录。 |
| 1B | `tests/agent/test_agent_boundaries.py` | 新增 Agent 执行上限测试。 |
| 1B | `tests/agent/test_sse_failures.py` | 新增 SSE 异常测试。 |
| 2 | `app/core/fallback.py` | 新增 FallbackManager。 |
| 2 | `app/core/token_budget.py` | 新增 TokenBudgetManager。 |
| 2 | `app/memory/` | 新增 memory 业务门面目录。 |
| 2 | `app/memory/conversation_manager.py` | 新增 ConversationManager。 |
| 2 | `app/memory/summarizer.py` | 新增 ConversationSummarizer。 |
| 2 | `app/agent/orchestrator.py` | 新增薄 AgentOrchestrator。 |
| 2 | `tests/unit/test_fallback.py` | 新增 fallback 单测。 |
| 2 | `tests/unit/test_token_budget.py` | 新增 token budget 单测。 |
| 2 | `tests/unit/test_conversation_manager.py` | 新增会话管理单测。 |
| 2 | `tests/unit/test_conversation_summarizer.py` | 新增摘要触发单测。 |
| 2 | `tests/agent/test_orchestrator.py` | 新增 orchestrator 回归测试。 |
| 3A | `app/rag/` | 新增 RAG 内部模块目录。 |
| 3A | `app/rag/models.py` | 新增 RAG 数据模型。 |
| 3A | `tests/rag/` | 新增 RAG 测试目录。 |
| 3A | `tests/rag/test_rag_models.py` | 新增 RAG models 单测。 |
| 3A | `tests/rag/test_metadata_indexing.py` | 新增 metadata 和幂等入库测试。 |
| 3A | `eval_sets/` | 新增评估集目录。 |
| 3A | `eval_sets/rag_cases.yaml` | 新增最小 RAG eval set。 |
| 3A | `evaluation/` | 新增离线评估目录。 |
| 3A | `evaluation/datasets.py` | 新增 eval dataset loader。 |
| 3A | `evaluation/rag_metrics.py` | 新增 Hit/Recall/MRR 指标。 |
| 3B | `app/rag/retriever.py` | 新增 RagRetriever。 |
| 3B | `app/rag/context_builder.py` | 新增 ContextBuilder。 |
| 3B | `app/rag/citation.py` | 新增 CitationBuilder。 |
| 3B | `app/rag/reranker.py` | 新增 Reranker/NoopReranker。 |
| 3B | `tests/rag/test_retriever.py` | 新增 retriever 测试。 |
| 3B | `tests/rag/test_context_builder.py` | 新增 context packing 测试。 |
| 3B | `tests/rag/test_citation_builder.py` | 新增 citation 测试。 |
| 3B | `tests/rag/test_rag_pipeline.py` | 新增 RAG pipeline 回归测试。 |
| 4 | `app/observability/metrics.py` | 新增 MetricsRecorder。 |
| 4 | `tests/integration/` | 新增 API 集成测试目录。 |
| 4 | `tests/integration/test_chat_api.py` | 新增 Chat API 集成测试。 |
| 4 | `tests/integration/test_aiops_api.py` | 新增 AIOps API 集成测试。 |
| 4 | `tests/integration/test_upload_api.py` | 新增上传/目录索引集成测试。 |
| 4 | `tests/integration/test_health_api.py` | 新增 health/clear/session 集成测试。 |
| 4 | `tests/rag/test_retrieval_metrics.py` | 新增检索指标回归测试。 |
| 4 | `evaluation/judge.py` | 新增 LLMJudge。 |
| 4 | `evaluation/runner.py` | 新增 evaluation runner。 |
| 4 | `eval_reports/` | 新增评估报告输出目录，通常加入 `.gitignore` 或仅提交 baseline。 |

## 10. 全局修改文件总表

| 阶段 | 修改文件 | 修改目的 | 兼容风险 |
| --- | --- | --- | --- |
| 1A | `app/main.py` | 接入 RequestContextMiddleware、配置化 CORS、保留现有路由。 | 中间件不得读取并消耗上传 body。 |
| 1A | `app/config.py` | 增加输入长度、上传限制、目录 allowlist、trace、CORS 配置。 | 默认值必须保持当前可运行行为。 |
| 1A | `app/models/request.py` | 保持 alias 行为，补充字段约束或 V2 model。 | `Id/Question` 优先级不得变化。 |
| 1A | `app/models/aiops.py` | 增加 session 约束和后续扩展字段兼容。 | 当前只传 `session_id` 必须继续可用。 |
| 1A | `app/models/response.py` | 增加统一 envelope 类型或兼容响应模型。 | 不删除 `ApiResponse`、`SessionInfoResponse` 旧字段。 |
| 1A | `app/api/chat.py` | 接入错误模型、context、InputGuard、trace 字段。 | 旧响应字段和 SSE `data.type` 必须保留。 |
| 1A | `app/api/aiops.py` | 接入 context、InputGuard、SSE trace/error 兼容。 | `complete` type 不能删除。 |
| 1A | `app/api/file.py` | 接入文件/目录安全校验和别名路径。 | `/api/upload`、`/api/index_directory` 不能删除。 |
| 1A | `app/api/health.py` | 追加 trace/request 和统一错误 envelope。 | 保留 `code/message/data`。 |
| 1B | `app/agent/mcp_client.py` | 将 retry 边界暴露给 ToolManager，增加 timeout/schema 适配。 | 不能破坏现有 MCP server 连接。 |
| 1B | `app/agent/aiops/executor.py` | 接入 ToolManager 包装后的工具。 | 保留 ToolNode 适配，避免重写工作流。 |
| 1B | `app/services/rag_agent_service.py` | 工具包装、执行上限、SSE 异常处理。 | 保留旧 `query/query_stream` 方法签名。 |
| 1B | `app/services/aiops_service.py` | 添加 recursion_limit、请求超时、SSE 中断处理。 | 不改变 planner/executor/replanner 基本链路。 |
| 2 | `app/services/rag_agent_service.py` | 通过薄 orchestrator 接入 fallback、token、conversation。 | 可配置回滚到旧服务路径。 |
| 2 | `app/services/aiops_service.py` | 接入 fallback、token、conversation 边界。 | 诊断事件结构兼容旧前端。 |
| 2 | `app/agent/aiops/planner.py` | planner 经验文档加入 token 预算和错误降级。 | 不影响默认计划 fallback。 |
| 2 | `app/agent/aiops/executor.py` | 工具结果压缩和预算控制。 | 工具错误不得作为事实。 |
| 2 | `app/agent/aiops/replanner.py` | past_steps 裁剪、摘要输入、MAX_STEPS trace。 | 保留 `MAX_STEPS=8` 业务语义。 |
| 2 | `app/tools/knowledge_tool.py` | 返回结构化检索/错误结果适配。 | 旧工具入口必须仍可被 Agent 调用。 |
| 3A | `app/services/document_splitter_service.py` | 写入稳定 metadata，兼容旧 metadata。 | 不破坏现有 split 行为。 |
| 3A | `app/services/vector_index_service.py` | 幂等索引、任务状态、doc-level delete。 | 部分失败仍返回旧 `failed_files`。 |
| 3A | `app/services/vector_store_manager.py` | 稳定 ids、doc_id delete、旧 `_source` 兼容。 | Milvus `id` 长度限制 100 需控制。 |
| 3A | `app/core/milvus_client.py` | 校验 metadata schema 兼容，必要时迁移说明。 | 不随意 drop collection。 |
| 3B | `app/services/vector_search_service.py` | 输出 raw_score/metric，支持 candidate_k/filter。 | 保留旧 `search_similar_documents`。 |
| 3B | `app/tools/knowledge_tool.py` | 适配新 RagRetriever/ContextBuilder/CitationBuilder。 | 旧工具返回文本仍可用。 |
| 3B | `app/services/rag_agent_service.py` | 接入可控 RAG pipeline 和 citations。 | 非流式与流式响应旧字段保留。 |
| 3B | `app/config.py` | 增加 RAG candidate/final/min_score/reranker 配置。 | 默认值保持当前 top_k 语义。 |
| 4 | `app/utils/logger.py` | 统一日志字段、关联 trace 和 metrics。 | 文本日志仍保留。 |
| 4 | `app/main.py` | 注册可观测性生命周期和 CI 友好配置。 | 不引入线上必需外部依赖。 |
| 4 | `README.md` | 更新 Python 版本、API 兼容、测试、评估、MCP 差异。 | 不把未来能力描述成当前已完成。 |
| 4 | `pyproject.toml` | 必要时补测试 marker、coverage omit、评估依赖。 | 避免强制安装非必需 judge 依赖。 |
| 4 | `mcp_servers/README.md` | 修正 MCP 工具清单或标注 mock 范围。 | 与实际 server 函数一致。 |
| 4 | `.gitignore` | 视情况忽略 `eval_reports/`、trace JSONL、临时上传。 | 不误忽略源码和 eval baseline。 |

## 11. 阶段 1A 详细执行计划

### 11.1 阶段目标

建立 API 输入边界、统一错误模型、请求上下文、文件/路径安全、最小 trace 和基础测试夹具。阶段 1A 完成后，非法输入不进入 Agent/RAG/Tool，成功和失败响应均能关联 `trace_id/request_id`。

### 11.2 阶段范围

- 统一 `AppError` 和错误响应 envelope。
- 新增 `RequestContextMiddleware` 和 `TraceLogger` 最小 JSONL trace。
- 新增 `InputGuard` 校验 Chat、AIOps、Clear、上传文件、目录路径。
- 保留旧 API 和旧响应字段，新增 `/api/file/upload` 与 `/api/file/index_directory` 兼容别名。
- 建立 `tests/conftest.py` fake 夹具和 1A 单元测试。

### 11.3 前置依赖

无阶段前置依赖。需要当前 FastAPI 应用可 import，并且不依赖真实 Milvus/DashScope 才能运行 1A 单测。

### 11.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/core/errors.py` | 新增 | AppError、错误子类、错误响应转换。 |
| `app/core/request_context.py` | 新增 | RequestContext 和 middleware。 |
| `app/core/input_guard.py` | 新增 | 输入、文件、目录校验。 |
| `app/observability/` | 新增目录 | 可观测性模块目录。 |
| `app/observability/tracing.py` | 新增 | 最小 JSONL trace。 |
| `tests/` | 新增目录 | 测试根目录。 |
| `tests/conftest.py` | 新增 | fake LLM、retriever、MCP、ctx、temp 文件夹具。 |
| `tests/unit/test_app_errors.py` | 新增 | AppError 单测。 |
| `tests/unit/test_request_context.py` | 新增 | RequestContext 单测。 |
| `tests/unit/test_trace_logger.py` | 新增 | TraceLogger 单测。 |
| `tests/unit/test_input_guard.py` | 新增 | InputGuard 单测。 |

### 11.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/main.py` | 注册 middleware，配置化 CORS。 | 中间件不得提前消费 body。 |
| `app/config.py` | 增加输入/上传/目录/trace/CORS 配置。 | 默认值需兼容现状。 |
| `app/models/request.py` | 保留 alias，补充约束。 | `Id/Question` 优先级不得变化。 |
| `app/models/aiops.py` | session_id 约束。 | 当前空请求仍默认 `default`。 |
| `app/models/response.py` | 增加 envelope 类型。 | 不删除旧 response model。 |
| `app/api/chat.py` | 接入 AppError、InputGuard、trace。 | 保留旧响应和 SSE data.type。 |
| `app/api/aiops.py` | 接入 InputGuard、trace、SSE error。 | 保留 `complete`。 |
| `app/api/file.py` | 路径安全、MIME、别名接口。 | 保留旧路径和响应结构。 |
| `app/api/health.py` | 追加 trace/envelope。 | 保留 `code/message/data`。 |

### 11.6 Issue 拆分

### ISSUE-001：统一错误模型与错误响应

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1A |
| 任务目标 | 新增统一 `AppError` 模型和错误响应转换函数，使 API、文件、RAG、Tool、LLM 错误后续能映射到稳定错误码。 |
| 背景说明 | 当前 API 多处直接返回 `str(e)` 或抛 `HTTPException`，错误结构不一致且缺少 trace。 |
| 前置依赖 | 无。 |
| 新增文件 | 新增：`app/core/errors.py`；新增：`tests/unit/test_app_errors.py`。 |
| 修改文件 | `app/models/response.py` 增加兼容 envelope 类型；`app/api/chat.py`、`app/api/file.py`、`app/api/health.py` 只做最小错误响应适配。 |
| 涉及模块 | API layer、错误模型、response adapter。 |
| 输入 | Python `Exception`、业务主动抛出的错误参数、可选 `RequestContext`。 |
| 输出 | `AppError` 实例；兼容旧字段的错误响应 dict；HTTP status 映射。 |
| 核心开发步骤 | 1. 定义 `AppError` 字段：`code/http_status/user_message/internal_message/retryable/fallback_required/details`。<br>2. 定义子类：`InvalidInputError`、`InvalidSessionIdError`、`RequestTooLargeError`、`FileTooLargeError`、`UnsupportedFileTypeError`、`InvalidFileMimeError`、`InvalidFileEncodingError`、`InvalidDirectoryError`、`PathTraversalBlockedError`、`SymlinkNotAllowedError`、`ToolTimeoutError`、`ToolExecutionError`、`LLMProviderError`、`VectorStoreUnavailableError`、`RagEmptyResultError`、`InternalAppError`。<br>3. 实现 `from_exception`、`to_error_response`、`safe_details`。<br>4. 保留旧 `code/message/data.success/answer/errorMessage/status/message/data` 字段适配能力。<br>5. 给 API handler 提供统一 `to_http_response` 辅助函数。 |
| 错误处理要求 | 未分类异常映射为 `INTERNAL_ERROR`；用户可见 message 不包含堆栈、密钥、内部 URL；details 写入前脱敏。 |
| trace/log 要求 | 当前 issue 可接收 ctx 但不强依赖 middleware；若 ctx 存在必须把 `trace_id/request_id` 写入 error envelope。 |
| API 兼容要求 | 不删除旧字段；失败响应只能追加 `success/error/trace_id/request_id`。 |
| 测试用例 | 错误码到 HTTP status 映射；`from_exception` 包装；脱敏；旧响应字段保留；无 ctx 时也能生成响应。 |
| 测试命令 | `pytest tests/unit/test_app_errors.py -q` |
| 验收标准 | P0 错误码均有稳定 code/status/retryable/fallback；错误响应包含或能追加 trace/request；旧字段未删除。 |
| 回滚策略 | 移除 API handler 的适配调用即可回到旧错误路径；保留 `errors.py` 不被业务强依赖。 |
| 风险点 | 一次性替换所有 handler 会影响前端，首轮只接入最小路径并保持旧结构。 |
| 不做事项 | 不实现 fallback 决策；不实现 trace 写入；不改 Agent 主流程。 |

### ISSUE-002：RequestContextMiddleware 和最小 trace

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1A |
| 任务目标 | 新增请求上下文和最小 JSONL trace，使每个请求有统一 `trace_id/request_id`。 |
| 背景说明 | 当前日志无法串联一次请求的 API、Agent、RAG、Tool 调用。 |
| 前置依赖 | ISSUE-001。 |
| 新增文件 | 新增：`app/core/request_context.py`；新增：`app/observability/tracing.py`；新增：`tests/unit/test_request_context.py`；新增：`tests/unit/test_trace_logger.py`。 |
| 修改文件 | `app/main.py` 注册 middleware；`app/config.py` 增加 `trace_enabled/trace_jsonl_path/request_timeout_ms/cors_allow_origins`。 |
| 涉及模块 | FastAPI middleware、RequestContext、TraceLogger、logger。 |
| 输入 | HTTP headers：`X-Trace-Id`、`X-Request-Id`；request method/path；默认 user/tenant 配置。 |
| 输出 | `request.state.ctx`；响应 header/body/SSE 可用的 `trace_id/request_id`；JSONL trace event。 |
| 核心开发步骤 | 1. 定义 `RequestContext`：`trace_id/request_id/session_id/tenant_id/user_id/deadline_ms/feature_flags/started_at`。<br>2. 实现 header 校验，非法 header 重新生成。<br>3. 实现 middleware，只创建 ctx，不读取 body。<br>4. 实现 `get_request_context` 和 `with_session`。<br>5. 实现 `TraceLogger.record_event/start_span/end_span/record_error`，写入 JSONL。<br>6. 在响应 header 追加 `X-Trace-Id/X-Request-Id`。 |
| 错误处理要求 | 非法 trace/request header 不报错，重新生成并记录 `invalid_inbound_trace_header=true`；trace 写入失败只打 warning。 |
| trace/log 要求 | `request.start/request.end/request.error` 至少包含 `trace_id/request_id/path/method/status/latency_ms/error_code`。 |
| API 兼容要求 | 只追加 header 和 body 字段，不改变旧 status 和旧响应字段。 |
| 测试用例 | header 透传；非法 header 重生成；默认 user/tenant；trace JSONL 写入；trace 写入失败不影响业务。 |
| 测试命令 | `pytest tests/unit/test_request_context.py tests/unit/test_trace_logger.py -q` |
| 验收标准 | 任一 TestClient 请求均能得到 `trace_id/request_id`；JSONL 可按 trace_id 串联 start/end。 |
| 回滚策略 | 从 `app/main.py` 移除 middleware 注册，保留文件不影响旧逻辑。 |
| 风险点 | 中间件如果读取 body 会破坏 upload；必须只读 header/path。 |
| 不做事项 | 不做完整 metrics；不做分布式 tracing；不引入外部观测平台。 |

### ISSUE-003：InputGuard 和 API 输入约束

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1A |
| 任务目标 | 新增 `InputGuard`，统一校验 chat、aiops、clear/session 和 prompt injection 风险标记。 |
| 背景说明 | 当前 `ChatRequest`、`AIOpsRequest` 基本无长度、空值、session 格式约束。 |
| 前置依赖 | ISSUE-001、ISSUE-002。 |
| 新增文件 | 新增：`app/core/input_guard.py`；新增：`tests/unit/test_input_guard.py`。 |
| 修改文件 | `app/models/request.py`、`app/models/aiops.py`、`app/api/chat.py`、`app/api/aiops.py`。 |
| 涉及模块 | request models、API handler、InputGuard、AppError。 |
| 输入 | `ChatRequest`、`AIOpsRequest`、`ClearRequest`、path `session_id`。 |
| 输出 | `ValidatedChatInput`、`ValidatedAIOpsInput`、`GuardResult`；失败时抛 AppError。 |
| 核心开发步骤 | 1. 定义 session regex 和长度上限。<br>2. 实现 `validate_session_id`。<br>3. 实现 `validate_chat`，trim question，判断空输入和超长。<br>4. 保持 `Id/Question` 与 `id/question` alias 行为。<br>5. 实现 `validate_aiops`，空 session 使用 `default`。<br>6. 实现 `detect_prompt_injection` 只记录 risk marker，不直接拒绝普通问题。<br>7. 在 chat/aiops/clear/session handlers 入口调用。 |
| 错误处理要求 | 空输入 `INVALID_INPUT`；非法 session `INVALID_SESSION_ID`；超长 `REQUEST_TOO_LARGE`；错误不得进入 Agent。 |
| trace/log 要求 | 记录 `input.validated/input.rejected/input.length/prompt_injection_risk`，不记录完整敏感文本。 |
| API 兼容要求 | `/api/chat` 同时支持 `Id/Question` 与 `id/question`；大小写同时传入时 `Id/Question` 优先。 |
| 测试用例 | 空 question；超长 question；非法 session；alias 大小写；AIOps 默认 session；prompt injection risk marker。 |
| 测试命令 | `pytest tests/unit/test_input_guard.py -q` |
| 验收标准 | 非法输入稳定 4xx，不进入 `rag_agent_service` 或 `aiops_service`；合法旧请求仍可用。 |
| 回滚策略 | 移除 handler 入口 guard 调用即可回旧行为；模型 alias 保持不变。 |
| 风险点 | Pydantic v2 alias 行为易被 model_config 改动影响，必须用测试锁定。 |
| 不做事项 | 不做内容审查拦截；不做认证鉴权；不改业务 prompt。 |

### ISSUE-004：文件上传与 index_directory 路径安全

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1A |
| 任务目标 | 加固上传和目录索引路径安全，并新增 `/api/file/*` 兼容别名。 |
| 背景说明 | 当前上传仅校验扩展名和大小；目录索引可接受任意路径；前端和后端上传限制不一致。 |
| 前置依赖 | ISSUE-001、ISSUE-002、ISSUE-003。 |
| 新增文件 | 无，复用 `app/core/input_guard.py`；新增测试可放入 `tests/unit/test_input_guard.py`。 |
| 修改文件 | `app/api/file.py`、`app/config.py`。 |
| 涉及模块 | File API、InputGuard、VectorIndexService、配置。 |
| 输入 | multipart `file`；JSON body 或 query `directory_path`。 |
| 输出 | 安全文件路径、规范化目录路径；失败 AppError；兼容旧 `IndexingResult.to_dict()`。 |
| 核心开发步骤 | 1. 配置 `upload_dir/upload_max_bytes/allowed_extensions/index_allowlist`。<br>2. 实现安全文件名：basename、拒绝控制字符、路径分隔符、Windows 保留名。<br>3. 读取内容前后校验大小，统一 10MiB 默认值。<br>4. 校验扩展名 `.txt/.md/.markdown` 策略。<br>5. MIME sniff 和 UTF-8 decode。<br>6. 目录 `resolve()` 后必须位于 allowlist。<br>7. 拒绝 symlink 和 path traversal。<br>8. 新增 `/api/file/upload` 和 `/api/file/index_directory`，复用同一 handler。 |
| 错误处理要求 | 文件过大 `FILE_TOO_LARGE`；扩展名 `UNSUPPORTED_FILE_TYPE`；MIME `INVALID_FILE_MIME`；编码 `INVALID_FILE_ENCODING`；目录 `INVALID_DIRECTORY/PATH_TRAVERSAL_BLOCKED/SYMLINK_NOT_ALLOWED`。 |
| trace/log 要求 | 记录文件大小、扩展名、目录 allowlist 命中、失败文件列表；不记录完整文件内容。 |
| API 兼容要求 | `/api/upload` 和 `/api/index_directory` 不删除；别名路径响应结构与旧路径一致；旧 `failed_files` map 保留。 |
| 测试用例 | 非法扩展名；MIME 不匹配；非 UTF-8；文件过大；文件名路径逃逸；allowlist 外目录；symlink；path traversal；部分文件失败。 |
| 测试命令 | `pytest tests/unit/test_input_guard.py -q` |
| 验收标准 | 路径逃逸和 symlink 被拒绝；部分文件失败仍返回 `partial_success`；新旧路径响应 schema 一致。 |
| 回滚策略 | 移除别名路由和 guard 调用，恢复旧上传逻辑；保留配置默认值。 |
| 风险点 | MIME sniff 可能误伤纯文本 Markdown；先允许保守文本 MIME，并在失败消息中可诊断。 |
| 不做事项 | 不做异步任务队列；不做病毒扫描；不重建 Milvus collection。 |

### ISSUE-005：tests/conftest.py fake 夹具和 InputGuard 单元测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1A |
| 任务目标 | 建立测试基础设施，让后续阶段可在无 Milvus、无 DashScope、无 MCP 的环境下运行关键单测。 |
| 背景说明 | 当前仓库 `tests/` 不存在，但 `pyproject.toml` 已配置 pytest、coverage、pytest-asyncio。 |
| 前置依赖 | ISSUE-001 至 ISSUE-004。 |
| 新增文件 | 新增：`tests/conftest.py`；新增：`tests/unit/test_input_guard.py`；根据前序 issue 保留已新增 test 文件。 |
| 修改文件 | 无；如 pytest coverage 导致外部服务 import，可后续 issue 最小调整 `pyproject.toml`。 |
| 涉及模块 | pytest、fake fixtures、InputGuard、AppError、RequestContext。 |
| 输入 | fake ctx、临时目录、临时文件、模拟上传对象、非法参数。 |
| 输出 | 可重复运行的单元测试。 |
| 核心开发步骤 | 1. 新建 `tests/conftest.py`。<br>2. 提供 `fake_request_context`。<br>3. 提供 `tmp_upload_dir` 和 `tmp_index_allowlist`。<br>4. 提供 `fake_llm`：正常、空响应、异常、超时。<br>5. 提供 `fake_retriever`：空、低分、重复 chunk、正常。<br>6. 提供 `fake_mcp_tool`：成功、异常、timeout、isError。<br>7. 补齐 InputGuard、AppError、RequestContext、TraceLogger 单测。 |
| 错误处理要求 | 测试不依赖真实环境变量；fake 抛出的错误应可映射到 AppError。 |
| trace/log 要求 | trace 测试写入临时 JSONL，不污染真实 `logs/`。 |
| API 兼容要求 | 测试锁定旧字段和 alias 兼容。 |
| 测试用例 | fake fixtures 可 import；InputGuard 所有边界；AppError 旧字段；RequestContext header；TraceLogger JSONL。 |
| 测试命令 | `pytest tests/unit -q` |
| 验收标准 | 单元测试可在无 Milvus/DashScope/MCP 环境运行；阶段 1A 测试通过。 |
| 回滚策略 | 删除 `tests/` 不影响业务运行；但不建议回滚测试基础。 |
| 风险点 | 导入 app 单例可能触发 DashScope 或 Milvus 初始化；测试中需要 monkeypatch 或延迟 import。 |
| 不做事项 | 不写集成测试；不启动服务；不访问网络。 |

### 11.7 阶段内开发顺序

1. ISSUE-001
2. ISSUE-002
3. ISSUE-003
4. ISSUE-004
5. ISSUE-005

### 11.8 阶段验收标准

- 所有旧 API 路径仍可访问。
- `/api/file/upload` 和 `/api/file/index_directory` 如新增，响应结构与旧路径一致。
- `/api/chat` 继续支持 `Id/Question`，并兼容 `id/question`。
- `POST /api/chat/clear` 支持 `sessionId` 和 `session_id`。
- 成功和失败响应可关联 `trace_id/request_id`。
- SSE 的 `start/error/done` 如新增必须包含 `trace_id/request_id`，并保留 `data.type`。
- 上传和目录索引拒绝非法扩展名、MIME 不匹配、非 UTF-8、文件过大、路径逃逸、symlink。
- `pytest tests/unit -q` 可运行。

### 11.9 阶段测试命令

```bash
pytest tests/unit -q
pytest tests/unit/test_app_errors.py -q
pytest tests/unit/test_request_context.py -q
pytest tests/unit/test_trace_logger.py -q
pytest tests/unit/test_input_guard.py -q
```

### 11.10 阶段回滚策略

- 从 `app/main.py` 移除 middleware 可回滚 RequestContext。
- 从 API handler 移除 InputGuard 调用可回滚输入校验。
- 别名路径可单独移除，但旧路径不能删除。
- TraceLogger 可通过 `trace_enabled=false` 关闭。
- 新增测试文件不影响运行时，可保留。

### 11.11 阶段不做事项

- 不接入 ToolManager。
- 不改 Agent 工作流。
- 不改 RAG metadata。
- 不引入 ConversationManager。
- 不做完整 metrics 和 eval。

### 11.12 阶段风险点

- middleware 消耗 body。
- Pydantic alias 行为被破坏。
- MIME sniff 误伤文本文件。
- 新 envelope 破坏旧前端判断。
- pytest import 触发外部服务初始化。

## 12. 阶段 1B 详细执行计划

### 12.1 阶段目标

统一本地工具、LangChain BaseTool、MCP tool 的调用边界，并建立 Agent 执行上限和 SSE 异常处理能力。阶段 1B 完成后，工具超时、异常、未授权、MCP `isError=true` 都应结构化表达，且不会作为事实证据。

### 12.2 阶段范围

- 新增 ToolManager、ToolResult、ToolPolicy、PolicyRegistry。
- 包装本地工具、LangChain BaseTool、MCP tool。
- 接入 RAG Agent 和 AIOps Executor 的工具列表。
- 增加业务步骤上限、LangGraph recursion_limit、最大工具调用次数、整体请求超时、单工具超时。
- 完善 Chat SSE 和 AIOps SSE 异常/中断处理。

### 12.3 前置依赖

阶段 1A 全部完成，特别是 AppError、RequestContext、TraceLogger、tests/conftest.py。

### 12.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/agent/tool_manager.py` | 新增 | ToolManager、ToolResult、ToolRegistry、工具 wrapper。 |
| `app/agent/policies.py` | 新增 | ToolPolicy、PolicyRegistry。 |
| `tests/unit/test_tool_manager.py` | 新增 | 工具成功、异常、timeout、裁剪、MCP isError 测试。 |
| `tests/unit/test_tool_policy.py` | 新增 | 工具权限策略测试。 |
| `tests/agent/` | 新增目录 | Agent 边界测试。 |
| `tests/agent/test_agent_boundaries.py` | 新增 | 最大步骤、recursion_limit、工具调用次数测试。 |
| `tests/agent/test_sse_failures.py` | 新增 | SSE 中途失败和 client disconnect 测试。 |

### 12.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/config.py` | 增加工具超时、最大调用次数、recursion_limit、request timeout 配置。 | 默认值要接近当前行为，避免突然中断。 |
| `app/agent/mcp_client.py` | 给 ToolManager 提供 MCP tool 包装边界。 | 现有 retry interceptor 不能被意外绕过。 |
| `app/agent/aiops/executor.py` | 使用 ToolManager 包装工具后交给 ToolNode 或适配层执行。 | 不重写执行节点。 |
| `app/services/rag_agent_service.py` | 初始化 agent 前包装工具；query/query_stream 加执行上限。 | 保留方法签名和旧输出类型。 |
| `app/services/aiops_service.py` | `graph.astream` 设置 recursion_limit，处理超时和中断。 | 保留现有 SSE type。 |
| `app/api/chat.py` | SSE start/error/done 结构化和 client disconnect。 | 保留 `event: message` + `data.type`。 |
| `app/api/aiops.py` | AIOps SSE 异常结构化。 | 保留 `complete`。 |

### 12.6 Issue 拆分

### ISSUE-006：ToolManager 核心模型与工具包装

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1B |
| 任务目标 | 新增 ToolManager、ToolResult 和工具包装能力，统一本地函数、LangChain BaseTool、MCP tool 的输出 envelope。 |
| 背景说明 | 当前工具调用分散，异常和 MCP `isError=true` 可能以普通文本进入 Agent 上下文。 |
| 前置依赖 | 阶段 1A；尤其是 AppError、RequestContext、TraceLogger、fake_mcp_tool。 |
| 新增文件 | 新增：`app/agent/tool_manager.py`；新增：`tests/unit/test_tool_manager.py`。 |
| 修改文件 | 暂不强接业务，先提供可独立测试的核心类。 |
| 涉及模块 | ToolManager、ToolResult、TraceLogger、AppError。 |
| 输入 | 工具对象、工具名称、args、RequestContext、timeout/trim 配置。 |
| 输出 | `ToolResult`，包含 `status/is_error/data/error/metadata/trimmed/evidence_usable`。 |
| 核心开发步骤 | 1. 定义 `ToolResult` 和 `ToolErrorInfo`。<br>2. 实现 `ToolResult.success/error/from_mcp_result/is_evidence_usable/as_prompt_block/to_trace_fields`。<br>3. 定义 `ToolCallSpec` 和 `ToolRegistry`。<br>4. 实现 `wrap_local_callable`、`wrap_langchain_tool`、`wrap_mcp_tool`。<br>5. 实现同步和异步 invoke。<br>6. 实现 result preview 和大 JSON 裁剪。 |
| 错误处理要求 | timeout 映射 `TOOL_TIMEOUT`；异常映射 `TOOL_EXECUTION_ERROR`；MCP `isError=true` 转 `ToolResult.is_error=true`。 |
| trace/log 要求 | 每次调用记录 `tool.start/tool.end/tool.error`，包含 tool_name、latency、status、trimmed、preview、error_code。 |
| API 兼容要求 | 内部模块不直接改变 API；后续接入时不得改变旧响应字段。 |
| 测试用例 | 本地工具成功；本地工具异常；异步工具超时；MCP isError；大 JSON 裁剪；`is_evidence_usable=false`。 |
| 测试命令 | `pytest tests/unit/test_tool_manager.py -q` |
| 验收标准 | 所有工具结果都可序列化为 ToolResult；错误结果不能作为证据。 |
| 回滚策略 | 暂未接业务，可直接停止 import。 |
| 风险点 | LangChain BaseTool 的 sync/async 调用差异；需要兼容 `invoke/ainvoke`。 |
| 不做事项 | 不做权限策略；不接 Agent；不做 retry 决策。 |

### ISSUE-007：ToolPolicy/PolicyRegistry 工具权限策略

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1B |
| 任务目标 | 新增工具权限策略，按 tenant/user/session/tool_name 控制工具可用性。 |
| 背景说明 | 当前所有本地和 MCP 工具对 Agent 默认可见，缺少权限和未来多租户边界。 |
| 前置依赖 | ISSUE-006。 |
| 新增文件 | 新增：`app/agent/policies.py`；新增：`tests/unit/test_tool_policy.py`。 |
| 修改文件 | `app/config.py` 增加默认工具 allowlist。 |
| 涉及模块 | PolicyRegistry、ToolManager、RequestContext、config。 |
| 输入 | tool_name、RequestContext、ToolPolicy 配置。 |
| 输出 | allow/deny 决策；未授权 ToolResult 或 AppError。 |
| 核心开发步骤 | 1. 定义 `ToolPolicy`：tool_name、allowed_tenants、allowed_users、enabled、timeout_ms、max_result_chars。<br>2. 定义 `PolicyRegistry`。<br>3. 增加默认策略：anonymous/default 可用当前本地工具和 mock MCP 工具。<br>4. ToolManager invoke 前检查 policy。<br>5. 未授权转换为 `ToolResult.status="unauthorized"`。 |
| 错误处理要求 | 未授权不暴露内部策略细节；对外统一映射 `TOOL_EXECUTION_ERROR` 或内部 `UNAUTHORIZED_TOOL` trace code。 |
| trace/log 要求 | 记录 `tool.policy.allowed/denied`，包含 tool_name、tenant_id、user_id。 |
| API 兼容要求 | 默认策略必须保持现有 demo 可用，不因权限导致工具全部不可用。 |
| 测试用例 | 默认允许；显式禁用；tenant 不匹配；user 不匹配；未授权不作为证据。 |
| 测试命令 | `pytest tests/unit/test_tool_policy.py -q` |
| 验收标准 | ToolManager 调用前必经 policy；默认匿名用户仍可使用既有工具。 |
| 回滚策略 | 配置 `tool_policy_enabled=false` 回到全允许。 |
| 风险点 | 策略过严会让 AIOps 无法调用 MCP；默认配置需保守。 |
| 不做事项 | 不引入认证系统；不实现 RBAC 管理 API。 |

### ISSUE-008：AIOps/RAG 工具接入 ToolManager

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1B |
| 任务目标 | 将 RAG Agent 和 AIOps Executor 的工具列表通过 ToolManager 包装接入。 |
| 背景说明 | 当前 `RagAgentService` 和 AIOps `executor.py` 每次直接合并本地工具和 MCP 工具，缺少统一边界。 |
| 前置依赖 | ISSUE-006、ISSUE-007。 |
| 新增文件 | 无。 |
| 修改文件 | `app/services/rag_agent_service.py`、`app/agent/aiops/executor.py`、`app/agent/mcp_client.py`、`app/tools/knowledge_tool.py`。 |
| 涉及模块 | ToolManager、RagAgentService、AIOps executor、MCP client、knowledge_tool。 |
| 输入 | 本地工具 `retrieve_knowledge/get_current_time`；MCP client tools。 |
| 输出 | 经过 ToolManager 包装的 LangChain-compatible tools；工具调用 trace。 |
| 核心开发步骤 | 1. 在 ToolManager 中提供 LangChain tool adapter。<br>2. `RagAgentService._initialize_agent` 合并工具后统一 wrap。<br>3. `executor.py` 传给 `bind_tools` 和 `ToolNode` 的工具使用 wrapper。<br>4. MCP `CallToolResult(isError=True)` 转成不可用证据。<br>5. 保留原始工具构建函数作为回滚路径。 |
| 错误处理要求 | 工具错误返回 ToolResult 错误块，不进入事实上下文；必要时由 Agent 得到“工具失败”的安全说明。 |
| trace/log 要求 | 工具名、来源 local/mcp、latency、status、error_code 写 trace。 |
| API 兼容要求 | `/api/chat`、`/api/chat_stream`、`/api/aiops` 响应旧字段不变。 |
| 测试用例 | RAG 工具正常；MCP 工具 isError；工具异常；ToolNode 可消费 wrapper；错误工具不作为证据。 |
| 测试命令 | `pytest tests/unit/test_tool_manager.py tests/agent/test_agent_boundaries.py -q` |
| 验收标准 | RAG/AIOps 工具调用均经过 ToolManager；默认配置下现有 demo 工具仍可用。 |
| 回滚策略 | 配置 `tool_manager_enabled=false`，恢复旧 `all_tools = local_tools + mcp_tools`。 |
| 风险点 | LangGraph ToolNode 可能要求 BaseTool schema，adapter 需保留 name/description/args_schema。 |
| 不做事项 | 不重写 planner/executor/replanner；不实现新 MCP 工具。 |

### ISSUE-009：Agent 执行上限、recursion_limit、SSE 异常处理

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1B |
| 任务目标 | 建立 Agent 执行上限，包括业务步骤、recursion_limit、最大工具调用、整体超时、单工具超时和 SSE 中断处理。 |
| 背景说明 | 当前只有 replanner 内部 `MAX_STEPS=8`，没有 graph recursion_limit、请求超时、工具调用上限和规范 SSE error。 |
| 前置依赖 | ISSUE-006 至 ISSUE-008。 |
| 新增文件 | 新增：`tests/agent/test_agent_boundaries.py`；新增：`tests/agent/test_sse_failures.py`。 |
| 修改文件 | `app/config.py`、`app/services/aiops_service.py`、`app/services/rag_agent_service.py`、`app/api/chat.py`、`app/api/aiops.py`。 |
| 涉及模块 | LangGraph config、ToolManager、SSE adapter、RequestContext deadline。 |
| 输入 | request context deadline；Agent run config；SSE client connection。 |
| 输出 | 正常 done event；结构化 error event；fallback-ready AppError。 |
| 核心开发步骤 | 1. 配置 `agent_recursion_limit/max_tool_calls/request_timeout_ms/tool_timeout_ms`。<br>2. `AIOpsService.graph.astream` config 增加 `recursion_limit`。<br>3. ToolManager 记录并限制每个 request 的工具调用次数。<br>4. 非流式请求包裹整体 timeout。<br>5. 单工具 timeout 由 ToolManager 控制。<br>6. Chat SSE 首条发送 `start`，异常发送 `error`，结束发送 `done`，同时保留旧 `event: message` + `data.type`。<br>7. AIOps `complete` 兼容 `done`。<br>8. 处理 client disconnect，停止生成并记录 trace。 |
| 错误处理要求 | 超业务步骤 `AGENT_MAX_STEP_EXCEEDED`；超 recursion_limit 映射同类错误或 `INTERNAL_ERROR` with fallback；工具超时 `TOOL_TIMEOUT`；SSE 中断 `SSE_STREAM_INTERRUPTED`。 |
| trace/log 要求 | 记录 `agent.start/end/error`、`agent.max_steps`、`agent.recursion_limit`、`agent.tool_call_count`、`sse.client_disconnect`。 |
| API 兼容要求 | SSE 可新增规范 event，但 `data.type` 不删除；AIOps `complete` 继续兼容。 |
| 测试用例 | 最大步骤停止；recursion_limit 生效；最大工具调用次数；工具 timeout；SSE 中途失败；client disconnect。 |
| 测试命令 | `pytest tests/agent/test_agent_boundaries.py tests/agent/test_sse_failures.py -q` |
| 验收标准 | 超限场景可停止且返回结构化错误；SSE error/done 均包含 trace/request。 |
| 回滚策略 | 配置增大或关闭上限；SSE adapter 保留旧 chunk 转发路径。 |
| 风险点 | asyncio timeout 取消可能留下后台任务；需要确保 generator 退出。 |
| 不做事项 | 不做 FallbackManager；不做 TokenBudget。 |

### ISSUE-010：ToolManager 和 Agent 边界测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 1B |
| 任务目标 | 补齐 ToolManager 和 Agent 边界测试，形成阶段 1B 回归保护。 |
| 背景说明 | 1B 改动影响工具调用和 SSE，必须使用 fake LLM/MCP/tool 验证边界。 |
| 前置依赖 | ISSUE-006 至 ISSUE-009。 |
| 新增文件 | 已新增测试文件基础上补充用例；必要新增：`tests/agent/test_tool_evidence_boundary.py`。 |
| 修改文件 | `tests/conftest.py` 增强 fake 工具、fake agent、fake SSE client。 |
| 涉及模块 | pytest、pytest-asyncio、ToolManager、AIOps/RAG service adapter。 |
| 输入 | fake tool、fake graph、fake SSE event generator。 |
| 输出 | 1B 测试集稳定通过。 |
| 核心开发步骤 | 1. fake local tool 支持 success/error/timeout/large_json。<br>2. fake MCP tool 支持 `isError=true`。<br>3. fake graph 模拟 recursion_limit。<br>4. 测试 ToolResult 不可作为证据。<br>5. 测试 SSE error event schema。<br>6. 测试 `complete` 到 `done` 兼容。 |
| 错误处理要求 | 测试覆盖所有 1B 错误码和 ToolResult 错误状态。 |
| trace/log 要求 | 测试断言 trace event 至少包含 tool/agent/sse 错误。 |
| API 兼容要求 | 测试锁定 `event: message` 和 `data.type`。 |
| 测试用例 | ToolManager 全矩阵；Agent 超限；SSE 中断；AIOps complete/done；错误工具不进事实证据。 |
| 测试命令 | `pytest tests/unit/test_tool_manager.py tests/unit/test_tool_policy.py tests/agent -q` |
| 验收标准 | 阶段 1B 测试不依赖真实 MCP/Milvus/DashScope；全部通过。 |
| 回滚策略 | 测试可保留；业务回滚后测试按配置跳过 ToolManager enabled 场景。 |
| 风险点 | 过度 mock 可能脱离真实 ToolNode；保留一个 adapter shape 测试。 |
| 不做事项 | 不跑集成测试；不连接真实 MCP server。 |

### 12.7 阶段内开发顺序

1. ISSUE-006
2. ISSUE-007
3. ISSUE-008
4. ISSUE-009
5. ISSUE-010

### 12.8 阶段验收标准

- 本地工具、LangChain BaseTool、MCP tool 都能包装为 ToolResult。
- ToolResult 错误结果不得进入事实证据链。
- Agent 有业务步骤、recursion_limit、工具调用次数、整体请求超时和单工具超时。
- SSE 中途失败能发结构化 error event。
- client disconnect 被记录并停止生成。
- 旧 SSE 解析方式兼容。

### 12.9 阶段测试命令

```bash
pytest tests/unit/test_tool_manager.py -q
pytest tests/unit/test_tool_policy.py -q
pytest tests/agent -q
```

### 12.10 阶段回滚策略

- `tool_manager_enabled=false` 恢复旧工具列表。
- `tool_policy_enabled=false` 允许默认工具。
- 移除 `recursion_limit` 配置可回旧 graph 行为，但保留业务 `MAX_STEPS`。
- SSE adapter 保留旧 `event: message` 转发。

### 12.11 阶段不做事项

- 不做 TokenBudget。
- 不做 ConversationManager。
- 不做 fallback 决策矩阵。
- 不做 RAG metadata。
- 不修改 MCP server 工具实现。

### 12.12 阶段风险点

- ToolNode adapter 不兼容。
- Timeout 取消导致资源泄漏。
- 默认权限误关工具。
- MCP retry 与 ToolManager timeout 重叠导致耗时过长。

## 13. 阶段 2 详细执行计划

### 13.1 阶段目标

建立 fallback、token 预算、会话上下文和 MemorySaver 边界。阶段 2 完成后，长历史不会无限塞入 LLM，LLM/RAG/Tool 失败有可解释降级，业务代码不直接解析 MemorySaver checkpoint。

### 13.2 阶段范围

- 新增 FallbackManager 和 fallback 策略矩阵。
- 新增 TokenBudgetManager 和裁剪优先级。
- 新增 ConversationManager 与 ConversationSummarizer。
- 新增薄 AgentOrchestrator，串联 guard/context/budget/memory/fallback/trace。
- 让 RAG Chat 和 AIOps 逐步通过薄编排层接入。

### 13.3 前置依赖

阶段 1A 和阶段 1B 全部完成。

### 13.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/core/fallback.py` | 新增 | FallbackManager、FallbackResult、FallbackPolicy。 |
| `app/core/token_budget.py` | 新增 | TokenBudgetManager、TokenBudget、TrimResult。 |
| `app/memory/` | 新增目录 | 会话管理模块。 |
| `app/memory/conversation_manager.py` | 新增 | ConversationManager。 |
| `app/memory/summarizer.py` | 新增 | ConversationSummarizer。 |
| `app/agent/orchestrator.py` | 新增 | 薄 AgentOrchestrator。 |
| `tests/unit/test_fallback.py` | 新增 | fallback 单测。 |
| `tests/unit/test_token_budget.py` | 新增 | token budget 单测。 |
| `tests/unit/test_conversation_manager.py` | 新增 | conversation 单测。 |
| `tests/unit/test_conversation_summarizer.py` | 新增 | summarizer 单测。 |
| `tests/agent/test_orchestrator.py` | 新增 | orchestrator 回归测试。 |

### 13.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/config.py` | 增加 fallback、token、conversation 配置。 | 默认值不得让现有短对话失败。 |
| `app/services/rag_agent_service.py` | 接入 Orchestrator 或提供 adapter。 | 保留旧 query/query_stream。 |
| `app/services/aiops_service.py` | AIOps 事件走 fallback/token 边界。 | 保留旧事件 type。 |
| `app/agent/aiops/planner.py` | 经验文档检索和工具描述纳入 token 预算。 | 默认计划 fallback 保留。 |
| `app/agent/aiops/executor.py` | 工具结果裁剪、错误结果隔离。 | 不改 ToolNode 基本执行。 |
| `app/agent/aiops/replanner.py` | past_steps 摘要/裁剪，记录 MAX_STEPS。 | 保持 MAX_STEPS 语义。 |
| `app/tools/knowledge_tool.py` | 适配 ToolResult 和 fallback/no-answer 边界。 | 旧工具可被 Agent 调用。 |
| `app/api/chat.py` | 非流式和流式 fallback 外显。 | 保留旧 response fields。 |
| `app/api/aiops.py` | AIOps fallback event。 | 保留 complete/done 兼容。 |

### 13.6 Issue 拆分

### ISSUE-011：FallbackManager 与 fallback 策略矩阵

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 新增 FallbackManager，定义 Chat、Chat SSE、AIOps、文件/健康检查的降级策略矩阵。 |
| 背景说明 | 当前异常多为直接返回错误文本或在节点内生成简易 fallback，职责分散且可能泄漏内部异常。 |
| 前置依赖 | 阶段 1A、1B。 |
| 新增文件 | 新增：`app/core/fallback.py`；新增：`tests/unit/test_fallback.py`。 |
| 修改文件 | `app/api/chat.py`、`app/api/aiops.py`、`app/services/rag_agent_service.py`、`app/services/aiops_service.py` 做最小接入。 |
| 涉及模块 | FallbackManager、AppError、ToolResult、TraceLogger、API/SSE adapter。 |
| 输入 | AppError、RequestContext、可用证据摘要、ToolResult 列表、调用场景。 |
| 输出 | `FallbackResult`：`fallback_used/reason_code/safe_message/partial_answer/should_continue_stream`。 |
| 核心开发步骤 | 1. 定义 `FallbackResult`。<br>2. 定义策略矩阵：输入错误不 fallback；LLM/RAG/Tool 可 fallback；文件和 health 不用 LLM fallback。<br>3. 实现 `decide/for_chat/for_aiops/to_sse_event/to_response_fields`。<br>4. 接入 Chat 非流式失败路径。<br>5. 接入 Chat SSE/AIOps SSE fallback event。 |
| 错误处理要求 | FallbackManager 不做 retry；fallback 文案不包含堆栈、密钥、内部 URL、原始异常全文。 |
| trace/log 要求 | 记录 `fallback.decide`，包含 reason_code、fallback_used、evidence_count、partial_answer。 |
| API 兼容要求 | 非流式 Chat 使用 `data.fallback_used=true`；SSE 可新增 `fallback` event 且保留 `data.type`。 |
| 测试用例 | LLM timeout；RAG empty；tool timeout；vector unavailable；SSE 中途失败；输入错误不 fallback。 |
| 测试命令 | `pytest tests/unit/test_fallback.py -q` |
| 验收标准 | 可降级场景有稳定 reason_code；不可降级场景保持结构化错误；不泄漏内部异常。 |
| 回滚策略 | `fallback_enabled=false` 回旧错误响应，但保留 AppError。 |
| 风险点 | fallback 被误用隐藏输入错误；矩阵测试必须覆盖。 |
| 不做事项 | 不做 retry；不调用 LLM 生成 fallback 文案。 |

### ISSUE-012：TokenBudgetManager 与 token 预算裁剪

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 新增 TokenBudgetManager，统一预算分配、token 估算、裁剪顺序和 usage/cost 记录。 |
| 背景说明 | 当前历史、经验文档、工具结果和执行历史可无限拼入 prompt，容易超上下文。 |
| 前置依赖 | ISSUE-011。 |
| 新增文件 | 新增：`app/core/token_budget.py`；新增：`tests/unit/test_token_budget.py`。 |
| 修改文件 | `app/config.py` 增加 token 配置；`app/agent/aiops/planner.py`、`executor.py`、`replanner.py` 接预算。 |
| 涉及模块 | TokenBudgetManager、ConversationManager、ToolResult、RAG models、TraceLogger。 |
| 输入 | scenario、model、system prompt、当前问题、历史消息、summary、RAG chunks、ToolResult、配置预算。 |
| 输出 | `TokenBudget`、`BudgetAllocation`、`TrimResult`、usage/cost trace fields。 |
| 核心开发步骤 | 1. 定义预算配置：model_context_window、reserved_output_tokens、history_budget、rag_budget、tool_budget、summary_budget。<br>2. 实现 `estimate_tokens`，优先 tokenizer，失败降级字符估算。<br>3. 实现 `allocate(scenario, model, ctx)`。<br>4. 按内部契约裁剪顺序实现 messages/chunks/tool result 裁剪。<br>5. 记录 usage/cost。<br>6. 当前问题超过硬上限抛 `REQUEST_TOO_LARGE`。 |
| 错误处理要求 | 估算失败不能中断主流程；当前问题和系统安全约束不得静默裁剪。 |
| trace/log 要求 | 记录 `token.allocate/token.trim/token.usage`，包含预算、估算方法、裁剪数量、usage/cost。 |
| API 兼容要求 | 不直接改 API schema；可在 `done.usage` 或 response data 中追加 usage。 |
| 测试用例 | tokenizer 降级；长历史裁剪；RAG chunk 裁剪；工具结果裁剪；当前问题超限；裁剪优先级。 |
| 测试命令 | `pytest tests/unit/test_token_budget.py -q` |
| 验收标准 | 20 轮长对话不会全部进入 LLM；裁剪顺序符合全局内部约束。 |
| 回滚策略 | `token_budget_enabled=false` 或使用宽松预算；保留 usage 记录关闭。 |
| 风险点 | token 估算不准；要保守预留输出和系统 prompt。 |
| 不做事项 | 不生成摘要正文；不改变检索排序。 |

### ISSUE-013：ConversationManager 与 MemorySaver 边界

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 新增 ConversationManager，隔离业务上下文和底层 MemorySaver checkpoint。 |
| 背景说明 | 当前 `RagAgentService.get_session_history` 直接解析 MemorySaver checkpoint，历史控制和对外转换耦合。 |
| 前置依赖 | ISSUE-012。 |
| 新增文件 | 新增：`app/memory/conversation_manager.py`；新增：`tests/unit/test_conversation_manager.py`。 |
| 修改文件 | `app/services/rag_agent_service.py`、`app/api/chat.py`。 |
| 涉及模块 | ConversationManager、MemorySaver、TokenBudgetManager、InputGuard、TraceLogger。 |
| 输入 | session_id、RequestContext、TokenBudget、当前用户消息、assistant 输出、MemorySaver handle。 |
| 输出 | `ConversationContext`：summary、recent_messages、history_metadata；保存/清理/查询结果。 |
| 核心开发步骤 | 1. 定义 `ConversationTurn/ConversationContext`。<br>2. 实现 `load_context` 从 MemorySaver 读取并转换。<br>3. 实现最近 N 轮选择和预算裁剪。<br>4. 实现 `save_turn` 写回或调用旧 checkpointer。<br>5. 实现 `clear_session` 和 `get_history` 门面。<br>6. API 查询历史改用门面，过滤系统消息和工具原始 payload。 |
| 错误处理要求 | MemorySaver 读取失败记录 trace，可按场景返回空历史或 AppError；摘要失败不删除 checkpoint。 |
| trace/log 要求 | 记录 `conversation.load/save/clear`，包含 message_count、summary_used、trimmed_count。 |
| API 兼容要求 | `GET /api/chat/session/{session_id}` 保留顶层 `session_id/message_count/history`。 |
| 测试用例 | 空历史；长历史；最近 N 轮；底层异常；清理成功/失败；对外历史不含系统消息。 |
| 测试命令 | `pytest tests/unit/test_conversation_manager.py -q` |
| 验收标准 | 业务层不直接解析 MemorySaver checkpoint；对外历史安全稳定。 |
| 回滚策略 | `conversation_manager_enabled=false` 回旧 `RagAgentService` 方法。 |
| 风险点 | MemorySaver 内部结构不稳定；门面需容错已有 tuple/namedtuple。 |
| 不做事项 | 不做长期记忆检索；不做数据库持久化。 |

### ISSUE-014：ConversationSummarizer 历史摘要

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 新增 ConversationSummarizer，在轮次或 token 超阈值时生成受控摘要。 |
| 背景说明 | 仅保留最近 N 轮会丢失长期上下文；但直接拼全部历史会超 token。 |
| 前置依赖 | ISSUE-012、ISSUE-013。 |
| 新增文件 | 新增：`app/memory/summarizer.py`；新增：`tests/unit/test_conversation_summarizer.py`。 |
| 修改文件 | `app/memory/conversation_manager.py` 接入摘要触发。 |
| 涉及模块 | ConversationSummarizer、ConversationManager、TokenBudgetManager、LLMFactory/ChatQwen。 |
| 输入 | 历史 turns、已有 summary、token budget、RequestContext。 |
| 输出 | 受限 summary 文本和 metadata：summary_version、source_turn_count、updated_at。 |
| 核心开发步骤 | 1. 定义摘要触发条件：历史轮次超过阈值或预计 token 超预算。<br>2. 定义摘要 prompt：只保留用户目标、关键事实、未解决事项、显式偏好。<br>3. 排除工具原始 payload、错误堆栈、密钥、未验证工具错误。<br>4. 摘要失败时保留最近 N 轮并记录 trace。<br>5. 支持 fake_llm 测试。 |
| 错误处理要求 | 摘要 LLM 失败不影响主请求；不得删除原始 MemorySaver checkpoint。 |
| trace/log 要求 | 记录 `conversation.summary`，包含 trigger_reason、source_turn_count、summary_tokens、error_code。 |
| API 兼容要求 | 对外历史接口不默认暴露 summary，除非后续追加字段。 |
| 测试用例 | 轮次触发；token 触发；摘要失败降级；工具 payload 被排除；摘要内容长度受限。 |
| 测试命令 | `pytest tests/unit/test_conversation_summarizer.py -q` |
| 验收标准 | 超预算历史可生成安全摘要；摘要不包含未验证错误工具文本。 |
| 回滚策略 | `conversation_summary_enabled=false`，只保留最近 N 轮。 |
| 风险点 | 摘要可能丢关键信息；保留最近完整轮次作为补偿。 |
| 不做事项 | 不做向量长期记忆；不做用户画像。 |

### ISSUE-015：AgentOrchestrator 薄编排层接入

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 新增薄 AgentOrchestrator，串联 RequestContext、InputGuard、TokenBudget、ConversationManager、ToolManager、FallbackManager、TraceLogger。 |
| 背景说明 | 当前 API 直接调 service，后续边界能力分散接入会导致重复逻辑。 |
| 前置依赖 | ISSUE-011 至 ISSUE-014。 |
| 新增文件 | 新增：`app/agent/orchestrator.py`；新增：`tests/agent/test_orchestrator.py`。 |
| 修改文件 | `app/api/chat.py`、`app/api/aiops.py`、`app/services/rag_agent_service.py`、`app/services/aiops_service.py`。 |
| 涉及模块 | AgentOrchestrator、RagAgentService、AIOpsService、FallbackManager、TokenBudgetManager、ConversationManager。 |
| 输入 | validated request、RequestContext、调用模式 chat/chat_stream/aiops。 |
| 输出 | API response fields 或 SSE events；trace spans。 |
| 核心开发步骤 | 1. 定义 `run_chat/run_chat_stream/run_aiops` 薄接口。<br>2. 只做编排，不承载业务算法。<br>3. Chat 先通过 orchestrator 调用旧 `rag_agent_service.query`。<br>4. AIOps 先通过 orchestrator 调用旧 `aiops_service.diagnose`。<br>5. 在边界包裹 token、conversation、fallback。<br>6. 提供配置回滚到旧服务直连。 |
| 错误处理要求 | orchestrator 捕获 AppError 并交给 FallbackManager；未知异常转 InternalAppError。 |
| trace/log 要求 | 记录 `orchestrator.start/end/error`，包含 mode、session_id、fallback_used、latency。 |
| API 兼容要求 | API 输出结构由 adapter 保持旧字段；orchestrator 不直接暴露给外部。 |
| 测试用例 | chat 成功；chat fallback；stream error；aiops partial fallback；回滚开关。 |
| 测试命令 | `pytest tests/agent/test_orchestrator.py -q` |
| 验收标准 | API 可通过 orchestrator 跑通，关闭开关可回旧 service。 |
| 回滚策略 | `orchestrator_enabled=false`，API handler 直连旧 service。 |
| 风险点 | 编排层变厚；必须禁止放入 RAG 排序、工具选择等业务算法。 |
| 不做事项 | 不重写 Agent 框架；不实现 RAG pipeline。 |

### ISSUE-016：阶段 2 单元测试和回归测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 2 |
| 任务目标 | 补齐阶段 2 的单元和回归测试，确保 fallback、token、conversation、orchestrator 可独立验收。 |
| 背景说明 | 阶段 2 串联多个边界模块，必须避免看似可运行但长历史/失败路径无保护。 |
| 前置依赖 | ISSUE-011 至 ISSUE-015。 |
| 新增文件 | 根据需要补充：`tests/agent/test_stage2_regression.py`。 |
| 修改文件 | `tests/conftest.py` 增加 fake MemorySaver、fake summarizer、fake usage。 |
| 涉及模块 | fallback、token_budget、conversation、summarizer、orchestrator。 |
| 输入 | fake LLM、fake memory、长历史、错误工具结果、RAG empty。 |
| 输出 | 阶段 2 测试集稳定通过。 |
| 核心开发步骤 | 1. 为 fallback 矩阵补全参数化测试。<br>2. 为 token 裁剪顺序补快照测试。<br>3. 为 conversation manager 补 MemorySaver 异常测试。<br>4. 为 summarizer 补安全摘要测试。<br>5. 为 orchestrator 补成功、fallback、回滚开关测试。 |
| 错误处理要求 | 失败路径必须返回 AppError/FallbackResult，不得裸异常。 |
| trace/log 要求 | 测试断言关键 trace event 存在。 |
| API 兼容要求 | 回归测试锁定旧 Chat/AIOps 字段。 |
| 测试用例 | LLM fail；Milvus fail；tool fail；20 轮长对话；summary fail；回滚开关。 |
| 测试命令 | `pytest tests/unit/test_fallback.py tests/unit/test_token_budget.py tests/unit/test_conversation_manager.py tests/unit/test_conversation_summarizer.py tests/agent/test_orchestrator.py -q` |
| 验收标准 | 阶段 2 所有新增模块测试通过；不访问真实外部服务。 |
| 回滚策略 | 保留测试，业务开关回旧路径后部分测试改为配置条件。 |
| 风险点 | 测试过度依赖内部实现；优先断言契约行为。 |
| 不做事项 | 不做 RAG eval；不跑 integration。 |

### 13.7 阶段内开发顺序

1. ISSUE-011
2. ISSUE-012
3. ISSUE-013
4. ISSUE-014
5. ISSUE-015
6. ISSUE-016

### 13.8 阶段验收标准

- FallbackManager 职责清晰，只决策不 retry。
- TokenBudgetManager 有预算分配、估算、裁剪顺序、usage/cost 记录。
- ConversationManager 与 MemorySaver 边界清晰。
- ConversationSummarizer 有触发条件和安全摘要约束。
- AgentOrchestrator 是薄编排层，可配置回滚。
- 20 轮长对话不会完整塞入 LLM。

### 13.9 阶段测试命令

```bash
pytest tests/unit/test_fallback.py -q
pytest tests/unit/test_token_budget.py -q
pytest tests/unit/test_conversation_manager.py -q
pytest tests/unit/test_conversation_summarizer.py -q
pytest tests/agent/test_orchestrator.py -q
```

### 13.10 阶段回滚策略

- `fallback_enabled=false`
- `token_budget_enabled=false`
- `conversation_manager_enabled=false`
- `conversation_summary_enabled=false`
- `orchestrator_enabled=false`

关闭后 API 回旧 service 路径，旧响应字段仍保留。

### 13.11 阶段不做事项

- 不做 RAG metadata 迁移。
- 不做 citation。
- 不做 reranker。
- 不做 LLM judge。
- 不把 evaluation runner 接入线上请求。

### 13.12 阶段风险点

- 编排层变厚。
- 摘要丢失上下文。
- Token 估算不准。
- fallback 隐藏真实错误。
- MemorySaver 内部结构变化。

## 14. 阶段 3A 详细执行计划

### 14.1 阶段目标

建立稳定 RAG 数据基础：metadata schema、稳定 doc/chunk ID、content_hash、索引幂等、doc-level delete、失败任务状态和最小 eval baseline。阶段 3A 完成后，同一文档重复入库应产生稳定 ID，删除旧文档无残留，eval set 可加载并计算轻量指标。

### 14.2 阶段范围

- 新增 `app/rag/models.py`。
- 稳定生成 `doc_id/chunk_id/content_hash`。
- 修改索引与向量存储以支持幂等。
- doc-level delete 和失败任务状态。
- 最小 RAG eval set 和 metrics。

### 14.3 前置依赖

阶段 1A 必须完成；建议阶段 2 已完成。阶段 3A 不依赖 3B。

### 14.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/rag/` | 新增目录 | RAG 内部模块。 |
| `app/rag/models.py` | 新增 | RAG 数据模型和 metadata schema。 |
| `tests/rag/` | 新增目录 | RAG 测试目录。 |
| `tests/rag/test_rag_models.py` | 新增 | 模型和 ID 单测。 |
| `tests/rag/test_metadata_indexing.py` | 新增 | metadata、幂等、删除一致性测试。 |
| `eval_sets/` | 新增目录 | 评估集。 |
| `eval_sets/rag_cases.yaml` | 新增 | 最小 eval baseline。 |
| `evaluation/` | 新增目录 | 离线评估模块。 |
| `evaluation/datasets.py` | 新增 | eval set loader。 |
| `evaluation/rag_metrics.py` | 新增 | Hit@K、Recall@K、MRR。 |

### 14.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/services/document_splitter_service.py` | 增加稳定 metadata。 | 保留旧 `_source/_extension/_file_name`。 |
| `app/services/vector_index_service.py` | 幂等索引、失败任务状态、doc-level delete。 | 部分失败结构兼容旧 `IndexingResult`。 |
| `app/services/vector_store_manager.py` | 稳定 ids、delete_by_doc_id、兼容 delete_by_source。 | Milvus id 长度限制。 |
| `app/core/milvus_client.py` | 校验 metadata version，必要时文档化迁移。 | 不随意 drop collection。 |
| `app/config.py` | 增加 tenant_id 默认值、metadata version、eval 配置。 | 默认 tenant 为 `default`。 |

### 14.6 Issue 拆分

### ISSUE-017：RAG 内部数据模型 app/rag/models.py

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 新增 RAG 内部数据模型，为 metadata、retrieval、context、citation、eval 提供统一 schema。 |
| 背景说明 | 当前 RAG 只使用 LangChain Document metadata，缺少稳定内部模型。 |
| 前置依赖 | 阶段 1A；建议阶段 2。 |
| 新增文件 | 新增：`app/rag/__init__.py`；新增：`app/rag/models.py`；新增：`tests/rag/test_rag_models.py`。 |
| 修改文件 | 无业务接入，先独立定义模型。 |
| 涉及模块 | RAG models、Pydantic、AppError。 |
| 输入 | LangChain Document、Milvus search result、旧 metadata。 |
| 输出 | `DocumentRecord`、`ChunkRecord`、`RetrievedChunk`、`RagContext`、`Citation`、`NoAnswerDecision`。 |
| 核心开发步骤 | 1. 定义 metadata schema：`doc_id/chunk_id/content_hash/tenant_id/version/source_path/file_name/chunk_index`。<br>2. 定义 `DocumentRecord` 和 `ChunkRecord`。<br>3. 定义 `RetrievedChunk`，包含 raw_score、normalized_score、metric。<br>4. 定义 `Citation` 内部 schema。<br>5. 定义 `NoAnswerDecision`。<br>6. 实现旧 metadata 兼容转换。 |
| 错误处理要求 | 缺关键 metadata 时抛 `RAG_METADATA_INVALID` 或返回迁移期兼容 warning。 |
| trace/log 要求 | 模型提供 `to_trace_fields`，不包含完整 chunk 内容。 |
| API 兼容要求 | 内部 citation 到 API citation 的输出延后到 3B，不改变 API。 |
| 测试用例 | 字段校验；旧 metadata 兼容；score 字段语义；citation API 预转换；缺字段错误。 |
| 测试命令 | `pytest tests/rag/test_rag_models.py -q` |
| 验收标准 | RAG 内部 schema 稳定，后续索引和检索都能复用。 |
| 回滚策略 | 未接业务，可停止 import。 |
| 风险点 | 过早锁死 schema；保留 `metadata_extra` 兼容未来字段。 |
| 不做事项 | 不改索引；不做 retriever；不输出 citation。 |

### ISSUE-018：doc_id/chunk_id/content_hash 稳定生成

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 为文档和 chunk 生成稳定 `doc_id/chunk_id/content_hash`，替代 UUID 作为逻辑 ID 基础。 |
| 背景说明 | 当前 UUID 入库无法幂等，也无法支撑 eval 的 `expected_doc_ids`。 |
| 前置依赖 | ISSUE-017。 |
| 新增文件 | 可在 `app/rag/models.py` 中新增函数；测试在 `tests/rag/test_rag_models.py`。 |
| 修改文件 | `app/services/document_splitter_service.py`、`app/services/vector_store_manager.py`。 |
| 涉及模块 | DocumentSplitterService、VectorStoreManager、RAG models。 |
| 输入 | tenant_id、source_path、安全相对路径、文件内容、chunk_index、chunk_text。 |
| 输出 | 稳定 doc_id、chunk_id、content_hash、metadata version。 |
| 核心开发步骤 | 1. 规范 source_path 为 allowlist 内相对路径。<br>2. `content_hash = sha256(normalized_content)`。<br>3. `doc_id = stable_hash(tenant_id + source_path)`，长度控制在 Milvus id 限制内。<br>4. `chunk_id = doc_id + "#" + zero_padded_chunk_index` 或 hash 后短 ID。<br>5. 每个 Document metadata 写入新字段并保留旧字段。<br>6. VectorStoreManager add_documents 使用 chunk_id 作为 Milvus primary id。 |
| 错误处理要求 | 路径无法规范化时抛 `INVALID_DIRECTORY` 或 `RAG_METADATA_INVALID`；ID 超长时使用短 hash。 |
| trace/log 要求 | 记录 doc_id、chunk_count、content_hash 前 12 位、metadata_version。 |
| API 兼容要求 | 上传/索引响应旧字段不变，可追加 doc_id/chunk_count。 |
| 测试用例 | 同一文件重复生成相同 doc_id/chunk_id；内容变化 content_hash 变化；路径大小写/分隔符规范化；ID 长度。 |
| 测试命令 | `pytest tests/rag/test_rag_models.py tests/rag/test_metadata_indexing.py -q` |
| 验收标准 | 同一 tenant 和 source_path 重复入库 ID 稳定；不同内容 hash 可区分版本。 |
| 回滚策略 | 保留旧 UUID 生成开关 `stable_rag_ids_enabled=false`。 |
| 风险点 | Windows 路径和 POSIX 路径规范化差异；测试需覆盖。 |
| 不做事项 | 不做检索改造；不做 citation。 |

### ISSUE-019：索引幂等、doc-level delete、失败任务状态

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 让文件重复入库幂等，支持按 doc_id 删除，并记录目录索引部分失败状态。 |
| 背景说明 | 当前按 `_source` 删除旧数据，新增 UUID 文档，存在残留和不可评估风险。 |
| 前置依赖 | ISSUE-018。 |
| 新增文件 | 可新增测试：`tests/rag/test_metadata_indexing.py`。 |
| 修改文件 | `app/services/vector_index_service.py`、`app/services/vector_store_manager.py`、`app/core/milvus_client.py`。 |
| 涉及模块 | VectorIndexService、VectorStoreManager、Milvus collection、IndexingResult。 |
| 输入 | 文件路径、文档 chunks、doc_id、content_hash。 |
| 输出 | 幂等入库结果；doc-level delete count；失败文件列表和任务状态。 |
| 核心开发步骤 | 1. `index_single_file` 先计算 doc metadata。<br>2. 入库前按 doc_id 删除旧 chunks，兼容 `_source` 删除。<br>3. 使用 chunk_id 作为 Milvus id 添加。<br>4. 如果 content_hash 未变，可选择 skip 或 delete+upsert，默认先 delete+upsert 保守。<br>5. `IndexingResult` 追加 `status/partial_success/failed_file_count/indexed_doc_ids`。<br>6. 单文件失败不影响目录其他文件。 |
| 错误处理要求 | 单文件失败加入 `failed_files`；向量库不可用映射 `VECTOR_STORE_UNAVAILABLE`；embedding 失败映射 `EMBEDDING_PROVIDER_ERROR`。 |
| trace/log 要求 | 记录 `rag.index.start/end/error`、doc_id、deleted_count、success_count、fail_count。 |
| API 兼容要求 | 保留 `data.success/directory_path/total_files/success_count/fail_count/failed_files`。 |
| 测试用例 | 重复入库不重复；doc-level delete 后无旧 chunk；部分文件失败；空目录；embedding 异常。 |
| 测试命令 | `pytest tests/rag/test_metadata_indexing.py -q` |
| 验收标准 | 同一文件重复索引无残留重复；删除一致；失败状态可诊断。 |
| 回滚策略 | 保留旧 `delete_by_source`，配置关闭 doc_id delete。 |
| 风险点 | Milvus delete 表达式兼容 JSON 字段；需要 fake vector store 单测先覆盖。 |
| 不做事项 | 不做在线迁移脚本；不做任务队列。 |

### ISSUE-020：最小 RAG eval set

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 新增最小 RAG 评估集，覆盖 aiops-docs 中关键知识文档。 |
| 背景说明 | 没有 eval set 无法判断 RAG 改造是否退化。 |
| 前置依赖 | ISSUE-018，保证 expected_doc_ids 可稳定表达。 |
| 新增文件 | 新增：`eval_sets/rag_cases.yaml`。 |
| 修改文件 | 可选更新 `docs/agent_engineering_design.md` 的引用，但本 issue 不强制。 |
| 涉及模块 | eval set、RAG metadata、aiops-docs。 |
| 输入 | 手写 case：id、question、expected_doc_ids、expected_keywords、should_answer。 |
| 输出 | YAML 格式评估集。 |
| 核心开发步骤 | 1. 扫描 `aiops-docs/` 文档主题。<br>2. 每篇核心文档添加 3 到 5 个 case。<br>3. 每个 case 写 `expected_doc_ids`，依赖稳定 doc_id 生成规则。<br>4. 增加 no-answer case。<br>5. 标注 tags 和 difficulty。 |
| 错误处理要求 | eval set loader 后续应对缺字段报错；本 issue 保持 YAML 可读。 |
| trace/log 要求 | 无运行时 trace；评估 runner 后续记录。 |
| API 兼容要求 | 无 API 变更。 |
| 测试用例 | YAML 可解析；每个 case 有 id/question/expected_doc_ids；doc_id 格式合法。 |
| 测试命令 | `pytest tests/rag/test_metadata_indexing.py -q` |
| 验收标准 | eval set 覆盖正常问答、低分、空检索、no-answer。 |
| 回滚策略 | 删除 eval set 不影响业务，但会失去 baseline。 |
| 风险点 | expected_doc_ids 与 doc_id 规则不一致；需 loader 校验。 |
| 不做事项 | 不做 LLM judge；不生成报告。 |

### ISSUE-021：evaluation/datasets.py 与 evaluation/rag_metrics.py

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 新增离线评估数据加载和轻量检索指标计算。 |
| 背景说明 | 阶段 3B 和阶段 4 需要可复用的 eval 基础。 |
| 前置依赖 | ISSUE-020。 |
| 新增文件 | 新增：`evaluation/__init__.py`；新增：`evaluation/datasets.py`；新增：`evaluation/rag_metrics.py`；新增：`tests/rag/test_retrieval_metrics.py` 初版可选。 |
| 修改文件 | `pyproject.toml` 如缺 YAML 依赖再补，可先使用已安装依赖或标准库替代。 |
| 涉及模块 | evaluation、RAG models、pytest。 |
| 输入 | `eval_sets/rag_cases.yaml`；retrieved doc_ids/chunk_ids。 |
| 输出 | `RagCase` 列表；Hit@K、Recall@K、MRR。 |
| 核心开发步骤 | 1. 定义 `RagCase`。<br>2. 实现 `load_rag_cases(path)` 和字段校验。<br>3. 实现 `hit_rate_at_k`。<br>4. 实现 `recall_at_k`。<br>5. 实现 `mrr`。<br>6. 添加纯函数单测。 |
| 错误处理要求 | 缺字段、重复 case id、expected_doc_ids 为空但 should_answer=true 时明确报错。 |
| trace/log 要求 | 离线模块使用普通 logger，不写线上 trace。 |
| API 兼容要求 | 不进入线上 API import。 |
| 测试用例 | 正常加载；缺字段；重复 id；Hit@K；Recall@K；MRR；空检索。 |
| 测试命令 | `pytest tests/rag/test_retrieval_metrics.py -q` |
| 验收标准 | 可离线加载 eval set 并计算指标。 |
| 回滚策略 | evaluation 目录独立，删除不影响线上。 |
| 风险点 | 引入 PyYAML 依赖；如未安装需添加到 dev 或主依赖。 |
| 不做事项 | 不做 runner；不调用 LLM。 |

### ISSUE-022：RAG metadata 与索引测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3A |
| 任务目标 | 补齐 RAG metadata、幂等、删除一致性和 eval loader 测试。 |
| 背景说明 | 3A 是后续 citation/eval 的地基，必须用测试锁定稳定 ID。 |
| 前置依赖 | ISSUE-017 至 ISSUE-021。 |
| 新增文件 | 完善：`tests/rag/test_metadata_indexing.py`、`tests/rag/test_rag_models.py`、`tests/rag/test_retrieval_metrics.py`。 |
| 修改文件 | `tests/conftest.py` 增加 fake vector store 和 fake embedding。 |
| 涉及模块 | RAG models、vector_index_service、vector_store_manager、evaluation metrics。 |
| 输入 | 临时文档、fake vector store、eval YAML。 |
| 输出 | 阶段 3A 测试集通过。 |
| 核心开发步骤 | 1. fake vector store 支持 add/delete/search。<br>2. 测试同文件重复入库 chunk_id 稳定。<br>3. 测试内容变化 content_hash 变化。<br>4. 测试 doc-level delete。<br>5. 测试部分文件失败。<br>6. 测试 eval set loader 和 metrics。 |
| 错误处理要求 | fake embedding 异常映射到失败文件；无真实 Milvus 依赖。 |
| trace/log 要求 | 测试可断言 index trace fields。 |
| API 兼容要求 | 索引结果旧字段保留。 |
| 测试用例 | metadata 稳定性；重复入库幂等；doc-level delete；部分失败；eval loader；Hit/Recall/MRR。 |
| 测试命令 | `pytest tests/rag -q` |
| 验收标准 | 3A 所有 RAG 基础测试通过。 |
| 回滚策略 | 业务可通过配置回 UUID；测试保留用于后续修复。 |
| 风险点 | fake 与 Milvus 行为差异；阶段 4 再补集成。 |
| 不做事项 | 不做 RAG pipeline。 |

### 14.7 阶段内开发顺序

1. ISSUE-017
2. ISSUE-018
3. ISSUE-019
4. ISSUE-020
5. ISSUE-021
6. ISSUE-022

### 14.8 阶段验收标准

- `app/rag/models.py` schema 稳定。
- 同一文档重复入库 ID 稳定。
- doc-level delete 无残留。
- 目录索引部分失败状态可诊断。
- eval set 可加载。
- Hit@K、Recall@K、MRR 可离线计算。

### 14.9 阶段测试命令

```bash
pytest tests/rag/test_rag_models.py -q
pytest tests/rag/test_metadata_indexing.py -q
pytest tests/rag/test_retrieval_metrics.py -q
pytest tests/rag -q
```

### 14.10 阶段回滚策略

- `stable_rag_ids_enabled=false` 回到 UUID 入库。
- 保留旧 `_source` delete。
- 如需要重建 collection，先导出 source docs 和 metadata。
- evaluation 目录独立，可不影响线上回滚。

### 14.11 阶段不做事项

- 不做 retriever pipeline。
- 不做 context packing。
- 不做 citation 输出。
- 不做 reranker。
- 不做 LLM judge。

### 14.12 阶段风险点

- Milvus primary key 长度限制。
- JSON metadata 查询表达式兼容性。
- Windows 路径规范化。
- 旧数据迁移期间 metadata 不完整。

## 15. 阶段 3B 详细执行计划

### 15.1 阶段目标

在 3A 稳定 metadata 基础上，构建可控 RAG pipeline：RagRetriever、score normalization、min_score/no-answer、ContextBuilder、CitationBuilder、Reranker/NoopReranker。阶段 3B 完成后，RAG answer 可输出 citations，低置信度可 no-answer，reranker 默认关闭且失败回退。

### 15.2 阶段范围

- 新增 `RagRetriever`。
- score normalization、min_score 和 no-answer 前置判断。
- ContextBuilder 去重、packing、预算裁剪、prompt injection 隔离。
- CitationBuilder 内部 citation 到 API citation 转换。
- Reranker 接口和 NoopReranker。
- RAG pipeline 测试和 baseline 回归。

### 15.3 前置依赖

阶段 3A 必须完成。TokenBudgetManager 来自阶段 2，若阶段 2 未完成，则 3B 只能使用临时预算适配器，不建议跳过。

### 15.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/rag/retriever.py` | 新增 | RagRetriever 和 RetrievalResult。 |
| `app/rag/context_builder.py` | 新增 | ContextBuilder。 |
| `app/rag/citation.py` | 新增 | CitationBuilder。 |
| `app/rag/reranker.py` | 新增 | Reranker 接口和 NoopReranker。 |
| `tests/rag/test_retriever.py` | 新增 | retriever 测试。 |
| `tests/rag/test_context_builder.py` | 新增 | context packing 测试。 |
| `tests/rag/test_citation_builder.py` | 新增 | citation 测试。 |
| `tests/rag/test_rag_pipeline.py` | 新增 | pipeline 回归测试。 |

### 15.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/services/vector_search_service.py` | 支持 candidate_k、filter、raw_score/metric 输出。 | 保留旧方法签名。 |
| `app/tools/knowledge_tool.py` | 使用新 retriever/context/citation 或提供兼容适配。 | 旧 Agent 工具调用仍返回文本。 |
| `app/services/rag_agent_service.py` | Chat 回答接入 RAG pipeline 和 citations。 | 旧回答字段保留。 |
| `app/config.py` | 增加 `rag_candidate_k/rag_final_k/rag_min_score/reranker_enabled`。 | 默认关闭 reranker。 |
| `app/core/token_budget.py` | 提供 RAG context budget 适配。 | 不改变阶段 2 行为。 |

### 15.6 Issue 拆分

### ISSUE-023：RagRetriever 检索入口

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 新增 RagRetriever，封装 query 变体、向量检索、去重、候选数和最终数。 |
| 背景说明 | 当前 RAG 检索隐藏在 `retrieve_knowledge` 工具中，不可控、不可评估。 |
| 前置依赖 | 阶段 3A。 |
| 新增文件 | 新增：`app/rag/retriever.py`；新增：`tests/rag/test_retriever.py`。 |
| 修改文件 | `app/services/vector_search_service.py`、`app/config.py`。 |
| 涉及模块 | RagRetriever、VectorSearchService、RAG models、TraceLogger。 |
| 输入 | query、RequestContext、TokenBudget、candidate_k、final_k、filters。 |
| 输出 | `RetrievalResult`：chunks、query_variants、empty_reason、raw/normalized score。 |
| 核心开发步骤 | 1. 定义 `RetrievalQuery/RetrievalResult`。<br>2. `VectorSearchService` 支持 candidate_k 和 metadata filters。<br>3. RagRetriever 调用向量检索并转 `RetrievedChunk`。<br>4. 去重 chunk_id 或 content_hash。<br>5. 截断到 final_k。<br>6. 记录 query variants，rewrite 默认关闭。 |
| 错误处理要求 | 向量库异常映射 `VECTOR_STORE_UNAVAILABLE`；空结果返回 `RetrievalResult.empty`。 |
| trace/log 要求 | 记录 `rag.retrieve.start/end`，包含 candidate_count、final_count、filters、latency。 |
| API 兼容要求 | 暂不改 API；旧 `retrieve_knowledge` 可继续使用旧路径。 |
| 测试用例 | 正常检索；空检索；重复 chunk；candidate_k > final_k；vector 异常。 |
| 测试命令 | `pytest tests/rag/test_retriever.py -q` |
| 验收标准 | retriever 可被 evaluation 和 knowledge_tool 复用。 |
| 回滚策略 | `new_rag_retriever_enabled=false` 使用旧 vector_store retriever。 |
| 风险点 | 旧 LangChain Milvus 返回 Document 与自定义 SearchResult 形态不同；需 adapter。 |
| 不做事项 | 不做 citation；不做 reranker。 |

### ISSUE-024：score normalization、min_score、no-answer 前置判断

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 将 raw_score 转换为 normalized score，并基于 min_score 做 no-answer 前置判断。 |
| 背景说明 | 当前 L2 distance 越小越相关，不能直接作为 `score` 对外输出。 |
| 前置依赖 | ISSUE-023。 |
| 新增文件 | 测试补充在 `tests/rag/test_retriever.py`。 |
| 修改文件 | `app/rag/models.py`、`app/rag/retriever.py`、`app/config.py`。 |
| 涉及模块 | RetrievedChunk、RagRetriever、NoAnswerDecision。 |
| 输入 | raw_score、metric_type、min_score、empty result。 |
| 输出 | normalized_score `0.0-1.0`；NoAnswerDecision。 |
| 核心开发步骤 | 1. 为 RetrievedChunk 保留 `raw_score/metric/normalized_score`。<br>2. L2 使用可解释归一化函数，越小越接近 1。<br>3. 配置 `rag_min_score`。<br>4. 过滤低分 chunks。<br>5. 全部低分或空结果时生成 no-answer decision。<br>6. trace 记录 raw 和 normalized。 |
| 错误处理要求 | 未知 metric 不得误判相似度，降级记录 warning 并使用保守分数。 |
| trace/log 要求 | 记录 min_score、dropped_low_score_count、empty_reason。 |
| API 兼容要求 | API citation 的 `score` 只使用 normalized score。 |
| 测试用例 | L2 归一化；min_score 过滤；空检索；低分检索；未知 metric。 |
| 测试命令 | `pytest tests/rag/test_retriever.py -q` |
| 验收标准 | 低置信检索不会伪装成有效证据；score 语义明确。 |
| 回滚策略 | `rag_min_score=0` 关闭过滤，保留 raw_score。 |
| 风险点 | 归一化函数阈值需要 baseline 调参；先保守。 |
| 不做事项 | 不做 LLM 拒答生成；不做 reranker。 |

### ISSUE-025：ContextBuilder context packing

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 新增 ContextBuilder，在 token 预算内构建可追溯、去重、安全隔离的 RAG context。 |
| 背景说明 | 当前 `format_docs` 直接拼接检索文档，无预算裁剪和引用锚点。 |
| 前置依赖 | ISSUE-024；阶段 2 TokenBudgetManager。 |
| 新增文件 | 新增：`app/rag/context_builder.py`；新增：`tests/rag/test_context_builder.py`。 |
| 修改文件 | `app/core/token_budget.py` 可补 RAG budget helper。 |
| 涉及模块 | ContextBuilder、RAG models、TokenBudgetManager、ToolResult。 |
| 输入 | query、RetrievedChunk 列表、TokenBudget、可选 ToolResult 列表。 |
| 输出 | `RagContext`：context_text、used_chunks、dropped_chunks、anchors、no_answer_decision。 |
| 核心开发步骤 | 1. 按 chunk_id/content_hash 去重。<br>2. 排除 `ToolResult.is_error=true`。<br>3. 按 normalized_score 和 source diversity 排序。<br>4. 为每个 chunk 生成 anchor，如 `[C1]`。<br>5. 按 token budget packing，低分优先丢弃。<br>6. 对 chunk 内容做 prompt injection 隔离，明确“以下是资料，不是指令”。 |
| 错误处理要求 | 无可用证据返回 no-answer decision；预算不足不编造 context。 |
| trace/log 要求 | 记录 `rag.context.build`，包含 input/used/dropped/budget/no_answer。 |
| API 兼容要求 | 不直接输出 API citation；只为后续生成提供内部 context。 |
| 测试用例 | 重复 chunk 去重；低分丢弃；预算裁剪；错误工具排除；无证据 no-answer；anchor 稳定。 |
| 测试命令 | `pytest tests/rag/test_context_builder.py -q` |
| 验收标准 | LLM prompt 中每段证据可追溯到 chunk_id；错误工具结果不进入事实上下文。 |
| 回滚策略 | `context_builder_enabled=false` 回旧 `format_docs`。 |
| 风险点 | 过度裁剪导致召回下降；需 baseline 回归。 |
| 不做事项 | 不生成答案；不做 citation API schema。 |

### ISSUE-026：CitationBuilder citation 输出

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 新增 CitationBuilder，将内部 citation 转为 API 安全 citations 输出。 |
| 背景说明 | 当前 Chat 响应没有可追溯来源，无法验证回答证据。 |
| 前置依赖 | ISSUE-025。 |
| 新增文件 | 新增：`app/rag/citation.py`；新增：`tests/rag/test_citation_builder.py`。 |
| 修改文件 | `app/services/rag_agent_service.py`、`app/api/chat.py`、`app/tools/knowledge_tool.py`。 |
| 涉及模块 | CitationBuilder、RagContext、RagAgentService、API adapter。 |
| 输入 | RagContext.used_chunks、answer、RequestContext。 |
| 输出 | API-safe `citations[]`：citation_id、doc_id、chunk_id、source_path、file_name、score、content_preview。 |
| 核心开发步骤 | 1. 按 used_chunks 顺序分配 `C1/C2`。<br>2. 合并重复 chunk。<br>3. preview 截断到 200 字符。<br>4. 拒绝绝对路径和 allowlist 外路径。<br>5. 非流式 Chat 在 `data.citations` 追加。<br>6. Chat SSE `retrieval/done` 可追加 citations。 |
| 错误处理要求 | 缺 doc_id/chunk_id 时 3A 后必须报错或丢弃该 citation 并 trace；不得输出绝对路径。 |
| trace/log 要求 | 记录 `rag.citation.build`，包含 citation_count、dropped_invalid_count、doc_ids。 |
| API 兼容要求 | 只追加 `citations`；不删除 `data.answer/errorMessage`；SSE 保留 `data.type`。 |
| 测试用例 | citation_id 顺序；重复 chunk 合并；preview 截断；绝对路径拒绝；API schema 转换。 |
| 测试命令 | `pytest tests/rag/test_citation_builder.py -q` |
| 验收标准 | API citations 与 `docs/api_contract.md` schema 兼容。 |
| 回滚策略 | `citations_enabled=false` 返回空数组或不追加字段。 |
| 风险点 | LLM 答案未显式引用 anchor；先按 used_chunks 输出来源列表。 |
| 不做事项 | 不做答案事实校验；不做 LLM judge。 |

### ISSUE-027：Reranker/NoopReranker 插口

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 新增 reranker 接口和默认 NoopReranker，为未来重排预留插口。 |
| 背景说明 | 设计要求 reranker 默认关闭，不能阻塞基础 RAG pipeline。 |
| 前置依赖 | ISSUE-023。 |
| 新增文件 | 新增：`app/rag/reranker.py`；测试补充：`tests/rag/test_retriever.py`。 |
| 修改文件 | `app/rag/retriever.py`、`app/config.py`。 |
| 涉及模块 | Reranker、NoopReranker、RagRetriever、config。 |
| 输入 | query、candidate chunks、RequestContext。 |
| 输出 | 重排后的 chunks；失败回退原序。 |
| 核心开发步骤 | 1. 定义 `BaseReranker.rerank`。<br>2. 实现 `NoopReranker` 原样返回。<br>3. 配置 `reranker_enabled=false` 默认关闭。<br>4. RagRetriever 在 candidate 后、final 前调用。<br>5. reranker 异常时记录 trace 并回退。 |
| 错误处理要求 | reranker 失败不影响请求；不得吞掉原始向量检索结果。 |
| trace/log 要求 | 记录 `rag.reranker.start/end/error/skipped`。 |
| API 兼容要求 | 不改变 API；只影响内部排序。 |
| 测试用例 | 默认关闭；Noop 原序；reranker 改序；reranker 异常回退。 |
| 测试命令 | `pytest tests/rag/test_retriever.py -q` |
| 验收标准 | 默认关闭时行为与无 reranker 一致；失败可回退。 |
| 回滚策略 | 保持 `reranker_enabled=false`。 |
| 风险点 | 未来真实 reranker 成本和延迟；阶段 3B 不引入外部模型。 |
| 不做事项 | 不实现真实 reranker；不做 hybrid search。 |

### ISSUE-028：RAG pipeline 测试和 baseline 回归

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 3B |
| 任务目标 | 补齐 RAG pipeline 测试和基于 eval set 的 baseline 回归，保证 3B 不降低核心检索指标。 |
| 背景说明 | retriever/context/citation/no-answer 改造会影响回答质量，需要可重复 baseline。 |
| 前置依赖 | ISSUE-023 至 ISSUE-027。 |
| 新增文件 | 新增：`tests/rag/test_rag_pipeline.py`；完善：`tests/rag/test_retrieval_metrics.py`。 |
| 修改文件 | `evaluation/rag_metrics.py` 如需补 per-case 输出。 |
| 涉及模块 | RagRetriever、ContextBuilder、CitationBuilder、NoopReranker、evaluation metrics。 |
| 输入 | fake vector store、eval cases、query。 |
| 输出 | Pipeline result、citations、no-answer decision、baseline metrics。 |
| 核心开发步骤 | 1. 构造正常 RAG case。<br>2. 构造空检索 case。<br>3. 构造低分检索 case。<br>4. 构造重复 chunk case。<br>5. 断言 citations 输出。<br>6. 计算 Hit@K/Recall@K/MRR baseline，并记录阈值。 |
| 错误处理要求 | 空检索和低分检索返回 no-answer，不抛未分类异常。 |
| trace/log 要求 | 测试断言 retrieve/context/citation trace 事件。 |
| API 兼容要求 | Chat response 旧字段保留，citations 只追加。 |
| 测试用例 | 空检索；低分检索；重复 chunk 去重；citation 输出；no-answer；Hit@K/Recall@K/MRR。 |
| 测试命令 | `pytest tests/rag -q` |
| 验收标准 | baseline Hit@5 不低于阶段 3A 记录阈值；pipeline 测试通过。 |
| 回滚策略 | 关闭 new RAG pipeline，回旧 `retrieve_knowledge`。 |
| 风险点 | baseline 数据太小波动大；阶段 4 再扩充报告。 |
| 不做事项 | 不做 LLM judge；不做生产性能压测。 |

### 15.7 阶段内开发顺序

1. ISSUE-023
2. ISSUE-024
3. ISSUE-025
4. ISSUE-026
5. ISSUE-027
6. ISSUE-028

### 15.8 阶段验收标准

- RagRetriever 支持 candidate_k、final_k、min_score。
- score normalized 到 `0.0-1.0`，raw_score 保留 trace。
- 空检索、低分检索触发 no-answer。
- ContextBuilder 去重、packing、预算裁剪、隔离错误工具结果。
- CitationBuilder 输出 API-safe citations。
- Reranker 默认关闭，失败回退。
- RAG pipeline baseline 不下降。

### 15.9 阶段测试命令

```bash
pytest tests/rag/test_retriever.py -q
pytest tests/rag/test_context_builder.py -q
pytest tests/rag/test_citation_builder.py -q
pytest tests/rag/test_rag_pipeline.py -q
pytest tests/rag -q
```

### 15.10 阶段回滚策略

- `new_rag_retriever_enabled=false`
- `context_builder_enabled=false`
- `citations_enabled=false`
- `reranker_enabled=false`
- `rag_min_score=0`

保留旧 `retrieve_knowledge` 入口作为兼容路径。

### 15.11 阶段不做事项

- 不做 hybrid search。
- 不引入真实 reranker 模型。
- 不做 LLM judge。
- 不重建线上请求路径外的评估平台。

### 15.12 阶段风险点

- min_score 阈值调参不准。
- normalized score 误导用户。
- context packing 影响召回。
- citation 与答案未逐句对齐。
- reranker 插口过度抽象。

## 16. 阶段 4 详细执行计划

### 16.1 阶段目标

建立上线前质量闭环：完整 trace/metrics、API/AIOps/上传集成测试、RAG 评估报告、CI 回归命令、指标趋势、README/config/pyproject/MCP README 一致性更新。

### 16.2 阶段范围

- 新增 MetricsRecorder。
- 补 API、AIOps、上传/目录索引集成测试。
- 新增 LLMJudge 和 evaluation runner。
- 输出 JSON/Markdown 评估报告。
- 补 CI 回归命令、README、配置和 MCP README。

### 16.3 前置依赖

阶段 1A、1B、2、3A、3B 全部完成。

### 16.4 新增文件总表

| 文件 | 标注 | 用途 |
| --- | --- | --- |
| `app/observability/metrics.py` | 新增 | MetricsRecorder。 |
| `tests/integration/` | 新增目录 | API 集成测试。 |
| `tests/integration/test_chat_api.py` | 新增 | Chat API 测试。 |
| `tests/integration/test_aiops_api.py` | 新增 | AIOps API 测试。 |
| `tests/integration/test_upload_api.py` | 新增 | 上传/目录索引测试。 |
| `tests/integration/test_health_api.py` | 新增 | health/clear/session 测试。 |
| `evaluation/judge.py` | 新增 | LLMJudge。 |
| `evaluation/runner.py` | 新增 | evaluation runner。 |
| `eval_reports/` | 新增目录 | 报告输出。 |

### 16.5 修改文件总表

| 文件 | 修改目的 | 兼容风险 |
| --- | --- | --- |
| `app/utils/logger.py` | 统一 trace/metrics 日志字段。 | 文本日志仍需可读。 |
| `app/main.py` | 测试友好生命周期、metrics 初始化。 | 不引入线上强依赖。 |
| `app/config.py` | metrics/eval/judge 配置。 | judge 默认关闭。 |
| `README.md` | 更新运行、API、测试、评估、Python 版本。 | 不把未来能力写成当前能力。 |
| `pyproject.toml` | 补 markers、coverage、可选 eval 依赖。 | 避免破坏现有 uv/pytest。 |
| `mcp_servers/README.md` | 修正 MCP 工具清单。 | 与实际工具一致。 |
| `.gitignore` | 忽略临时报表/trace。 | 不忽略 baseline eval set。 |

### 16.6 Issue 拆分

### ISSUE-029：MetricsRecorder 与完整 trace/metrics

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 新增 MetricsRecorder，完善请求、工具、RAG、LLM、fallback、token、cost 指标记录。 |
| 背景说明 | 阶段 1A 只有最小 trace，缺少可用于趋势和回归的 metrics。 |
| 前置依赖 | 阶段 1A 至 3B。 |
| 新增文件 | 新增：`app/observability/metrics.py`。 |
| 修改文件 | `app/observability/tracing.py`、`app/utils/logger.py`、`app/main.py`、`app/config.py`。 |
| 涉及模块 | TraceLogger、MetricsRecorder、TokenBudgetManager、RAG pipeline、ToolManager。 |
| 输入 | trace events、usage、latency、token、cost、error、citation metrics。 |
| 输出 | JSONL metrics records；聚合字段；评估 runner 可读取的趋势数据。 |
| 核心开发步骤 | 1. 定义 `MetricsRecord` 字段。<br>2. 实现 `record_request/record_tool/record_rag/record_llm/record_fallback/record_cost`。<br>3. 与 TraceLogger 共享 trace_id/request_id。<br>4. 写入 `logs/metrics.jsonl` 或配置路径。<br>5. 对敏感字段脱敏。<br>6. trace 写失败不影响主流程。 |
| 错误处理要求 | metrics 写入失败只 warning；不得影响 API。 |
| trace/log 要求 | 完整 trace 至少包含 latency、status、error_code、token、cost、fallback、citation_count、retrieval_count。 |
| API 兼容要求 | 不改变 API，可追加 `usage`。 |
| 测试用例 | 成功 request；失败 request；tool metrics；rag metrics；写入失败降级；敏感字段脱敏。 |
| 测试命令 | `pytest tests/unit/test_trace_logger.py -q` |
| 验收标准 | 一次 chat 请求可串联 request、rag、tool、llm、fallback、usage。 |
| 回滚策略 | `metrics_enabled=false`，保留最小 trace。 |
| 风险点 | 日志量膨胀；配置采样或文件轮转。 |
| 不做事项 | 不接 Prometheus；不建 dashboard。 |

### ISSUE-030：API 集成测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 为 `/api/chat`、`/api/chat_stream`、`/api/health`、`/api/chat/clear`、`/api/chat/session/{session_id}` 增加集成测试。 |
| 背景说明 | 前面阶段有单测，但需要验证 FastAPI handler、middleware、response schema。 |
| 前置依赖 | ISSUE-029。 |
| 新增文件 | 新增：`tests/integration/test_chat_api.py`；新增：`tests/integration/test_health_api.py`。 |
| 修改文件 | `tests/conftest.py` 增加 app TestClient、lifespan mock、service monkeypatch。 |
| 涉及模块 | FastAPI app、chat API、health API、RequestContext、SSE adapter。 |
| 输入 | TestClient 请求、fake rag service、fake health dependency。 |
| 输出 | HTTP/SSE response schema 断言。 |
| 核心开发步骤 | 1. 提供不会连接真实 Milvus 的 app fixture。<br>2. 测 `/api/chat` 大小写字段兼容。<br>3. 测 `/api/chat_stream` start/token/error/done 和 data.type。<br>4. 测 `/api/health` 成功/失败。<br>5. 测 clear 支持 sessionId/session_id。<br>6. 测 session history 旧顶层字段。 |
| 错误处理要求 | 输入错误返回 4xx 和统一 error；SSE 内错误不改变已建立 HTTP status。 |
| trace/log 要求 | 所有集成响应断言 trace_id/request_id。 |
| API 兼容要求 | 锁定 `code/message/data`、`data.success/answer/errorMessage`、`status/message/data`。 |
| 测试用例 | `/api/chat`；`/api/chat_stream`；`/api/health`；`/api/chat/clear`；`/api/chat/session/{session_id}`。 |
| 测试命令 | `pytest tests/integration/test_chat_api.py tests/integration/test_health_api.py -q` |
| 验收标准 | Chat 和 health 集成测试通过，不访问真实外部服务。 |
| 回滚策略 | 测试可保留；如 app fixture 影响现有测试，隔离到 integration conftest。 |
| 风险点 | FastAPI lifespan 会连接 Milvus；需要 mock lifespan 或 dependency。 |
| 不做事项 | 不跑真实 LLM；不跑端到端浏览器。 |

### ISSUE-031：AIOps 集成测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 为 `/api/aiops` 增加 SSE 集成测试，覆盖 plan、step_complete、report、complete/done、error、fallback。 |
| 背景说明 | AIOps SSE 类型较多，且旧前端兼容 `complete`。 |
| 前置依赖 | ISSUE-030。 |
| 新增文件 | 新增：`tests/integration/test_aiops_api.py`。 |
| 修改文件 | `tests/conftest.py` 增加 fake aiops_service。 |
| 涉及模块 | AIOps API、AIOpsService、SSE adapter、FallbackManager。 |
| 输入 | TestClient POST `/api/aiops`，fake diagnose async generator。 |
| 输出 | SSE events schema。 |
| 核心开发步骤 | 1. fake diagnose 产生 status/plan/step_complete/report/complete。<br>2. 测 complete 和 done 兼容。<br>3. fake diagnose 抛异常，断言 error event。<br>4. fake fallback event。<br>5. 断言 start/error/done trace/request。 |
| 错误处理要求 | 异常通过 SSE error event 返回，不泄漏堆栈。 |
| trace/log 要求 | AIOps 每个 start/error/done/fallback 包含 trace/request。 |
| API 兼容要求 | `event: message` 和 `data.type` 保留；`complete` 兼容。 |
| 测试用例 | 正常诊断流；异常流；fallback；client disconnect 可用 agent 测试覆盖。 |
| 测试命令 | `pytest tests/integration/test_aiops_api.py -q` |
| 验收标准 | AIOps SSE schema 与兼容策略通过测试。 |
| 回滚策略 | 测试保留；业务回滚后 fake service 调整兼容旧事件。 |
| 风险点 | SSE parser 在 TestClient 中处理细节；封装测试 helper。 |
| 不做事项 | 不调用真实 MCP。 |

### ISSUE-032：上传/目录索引集成测试

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 为 `/api/upload`、`/api/file/upload`、`/api/index_directory`、`/api/file/index_directory` 增加集成测试。 |
| 背景说明 | 上传/索引安全和新旧路径一致性必须通过真实 FastAPI multipart/body 断言。 |
| 前置依赖 | ISSUE-030。 |
| 新增文件 | 新增：`tests/integration/test_upload_api.py`。 |
| 修改文件 | `tests/conftest.py` 增加 fake vector_index_service。 |
| 涉及模块 | File API、InputGuard、VectorIndexService、FastAPI multipart。 |
| 输入 | multipart 文件、JSON body、query 参数、临时目录。 |
| 输出 | HTTP response schema 和 trace。 |
| 核心开发步骤 | 1. 测 `/api/upload` 成功。<br>2. 测 `/api/file/upload` schema 与旧路径一致。<br>3. 测非法扩展名、MIME、非 UTF-8、过大文件。<br>4. 测 `/api/index_directory` query 兼容。<br>5. 测 `/api/file/index_directory` body。<br>6. 测 allowlist 外、symlink、path traversal、部分失败。 |
| 错误处理要求 | 安全错误返回 4xx/415/413 对应 code；部分失败 200 partial_success。 |
| trace/log 要求 | 成功、失败、部分成功均含 trace/request。 |
| API 兼容要求 | 新旧上传/索引路径响应结构一致；旧 `failed_files` 保留。 |
| 测试用例 | `/api/upload`；`/api/file/upload`；`/api/index_directory`；`/api/file/index_directory`；全部文件/目录安全矩阵。 |
| 测试命令 | `pytest tests/integration/test_upload_api.py -q` |
| 验收标准 | 上传/索引集成路径安全且兼容。 |
| 回滚策略 | 保留旧路径测试；别名测试可随别名开关跳过。 |
| 风险点 | multipart MIME 在测试中不稳定；明确指定 content_type。 |
| 不做事项 | 不连接真实 Milvus；不测大文件真实 50MB，可用配置降阈值。 |

### ISSUE-033：LLMJudge 与 evaluation runner

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 新增 LLMJudge 和 evaluation runner，支持完整 RAG 评估。 |
| 背景说明 | 3A/3B 已有轻量指标，阶段 4 需要回答质量评估和报告入口。 |
| 前置依赖 | ISSUE-028、ISSUE-029。 |
| 新增文件 | 新增：`evaluation/judge.py`；新增：`evaluation/runner.py`。 |
| 修改文件 | `app/config.py` 增加 eval/judge 配置；`pyproject.toml` 可选补依赖。 |
| 涉及模块 | evaluation datasets、rag_metrics、RagRetriever、LLMJudge、report writer。 |
| 输入 | eval dataset、retriever/pipeline adapter、可选 judge model。 |
| 输出 | per-case evaluation result；aggregate metrics；judge scores。 |
| 核心开发步骤 | 1. 定义 judge rubric：faithfulness、answer_correctness、no_answer。<br>2. Judge model temperature 固定 0。<br>3. runner 加载 dataset。<br>4. 对每个 case 执行 retrieval/pipeline。<br>5. 计算 Hit@K/Recall@K/MRR。<br>6. 可选调用 LLMJudge。<br>7. 返回结构化结果。 |
| 错误处理要求 | judge 失败不阻断轻量指标；case 失败记录 error 并继续。 |
| trace/log 要求 | evaluation 使用独立 eval run id，不写线上 request trace；记录 dataset、case_id、latency、errors。 |
| API 兼容要求 | evaluation 不进入线上 API。 |
| 测试用例 | dataset 加载；runner dry-run；judge disabled；judge fake enabled；case 失败继续。 |
| 测试命令 | `python -m evaluation.runner --dataset eval_sets/rag_cases.yaml --judge disabled` |
| 验收标准 | runner 可在无真实 judge 时完成轻量评估。 |
| 回滚策略 | evaluation 独立目录，关闭 judge 或不运行 runner。 |
| 风险点 | judge 成本和不稳定；默认 disabled。 |
| 不做事项 | 不把 judge 作为唯一 CI 阻断条件。 |

### ISSUE-034：评估报告 JSON/Markdown 输出

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 为 evaluation runner 增加 JSON 和 Markdown 报告输出、趋势记录。 |
| 背景说明 | 评估结果需要给开发和发布回归使用，而不是只打印到控制台。 |
| 前置依赖 | ISSUE-033。 |
| 新增文件 | 新增目录：`eval_reports/`；runner 生成 `eval_reports/{timestamp}.json` 和 `.md`。 |
| 修改文件 | `evaluation/runner.py`、`.gitignore`。 |
| 涉及模块 | evaluation runner、report writer、metrics trend。 |
| 输入 | per-case eval results、aggregate metrics、baseline thresholds。 |
| 输出 | JSON report、Markdown report、趋势记录。 |
| 核心开发步骤 | 1. 定义 report schema。<br>2. 输出 aggregate metrics。<br>3. 输出 per-case diff。<br>4. 输出失败 case 列表。<br>5. Markdown 报告包含配置、数据集、指标、失败原因。<br>6. 趋势文件记录 timestamp 和核心指标。 |
| 错误处理要求 | 报告目录不存在自动创建；写入失败返回明确错误码并非静默。 |
| trace/log 要求 | eval runner 日志记录 report path 和 run id。 |
| API 兼容要求 | 无线上 API 变更。 |
| 测试用例 | JSON schema；Markdown 内容；失败 case；趋势追加；报告写入失败。 |
| 测试命令 | `python -m evaluation.runner --dataset eval_sets/rag_cases.yaml --output eval_reports --judge disabled` |
| 验收标准 | 每次完整评估输出 JSON 和 Markdown 报告。 |
| 回滚策略 | 删除报告输出参数，仅打印控制台；不影响线上。 |
| 风险点 | 报告文件过多；`.gitignore` 忽略临时报告，仅提交 baseline。 |
| 不做事项 | 不做 dashboard；不上传外部系统。 |

### ISSUE-035：CI 回归命令和 README 更新

| 字段 | 内容 |
| --- | --- |
| 所属阶段 | 阶段 4 |
| 任务目标 | 整理 CI 回归命令，更新 README/config/pyproject/MCP README 不一致项。 |
| 背景说明 | 当前 README 与 pyproject、config、MCP README、上传限制存在差异。 |
| 前置依赖 | ISSUE-029 至 ISSUE-034。 |
| 新增文件 | 可新增：`.github/workflows/ci.yml`，仅当项目已有或确认使用 GitHub Actions；否则只文档化命令。 |
| 修改文件 | `README.md`、`pyproject.toml`、`mcp_servers/README.md`、`.gitignore`。 |
| 涉及模块 | 文档、CI、配置、MCP server docs。 |
| 输入 | 当前测试命令、评估命令、配置项、MCP 实际工具清单。 |
| 输出 | README 更新、CI 命令、MCP README 一致性、可选 workflow。 |
| 核心开发步骤 | 1. README Python 版本改为 3.11+。<br>2. README 说明 `/api/upload` 和 `/api/file/upload` 兼容关系。<br>3. README 说明测试命令矩阵。<br>4. README 说明 evaluation runner。<br>5. config 文档补 `dashscope_api_base` 或说明固定默认。<br>6. MCP README 工具清单与 `cls_server.py/monitor_server.py` 实际函数一致，或补 mock 工具实现计划。<br>7. pyproject 补 markers/coverage 配置。<br>8. 文档列出 CI 推荐命令。 |
| 错误处理要求 | README 不得声称未完成能力已上线；MCP README 不得列不存在工具为已实现。 |
| trace/log 要求 | 无运行时 trace。 |
| API 兼容要求 | README 必须明确所有旧 API 不删除。 |
| 测试用例 | 文档链接检查可选；运行全量测试命令和 evaluation dry-run。 |
| 测试命令 | `pytest tests -q`；`python -m evaluation.runner --dataset eval_sets/rag_cases.yaml` |
| 验收标准 | README/config/pyproject/MCP README 不一致项关闭；测试矩阵命令可执行。 |
| 回滚策略 | 文档修改可单独 revert；CI workflow 可单独禁用。 |
| 风险点 | 添加 workflow 可能因外部服务缺失失败；优先 fake/default profile。 |
| 不做事项 | 不强制真实 DashScope/Milvus/MCP 进入 CI；不删除旧接口文档。 |

### 16.7 阶段内开发顺序

1. ISSUE-029
2. ISSUE-030
3. ISSUE-031
4. ISSUE-032
5. ISSUE-033
6. ISSUE-034
7. ISSUE-035

### 16.8 阶段验收标准

- 完整 trace/metrics 覆盖 request、tool、rag、llm、fallback、token、cost。
- API 集成测试覆盖 Chat、Chat SSE、Health、Clear、Session。
- AIOps 集成测试覆盖 SSE 正常、异常、fallback。
- 上传/目录索引集成测试覆盖新旧路径和安全矩阵。
- evaluation runner 输出 JSON/Markdown。
- README/config/pyproject/MCP README 不一致项已修复或明确标注。

### 16.9 阶段测试命令

```bash
pytest tests/unit -q
pytest tests/agent -q
pytest tests/rag -q
pytest tests/integration -q
pytest tests -q
python -m evaluation.runner --dataset eval_sets/rag_cases.yaml
```

### 16.10 阶段回滚策略

- `metrics_enabled=false`
- `judge_enabled=false`
- CI workflow 可禁用。
- README/MCP README 可单独 revert。
- evaluation runner 独立，不影响线上。

### 16.11 阶段不做事项

- 不把 evaluation runner 放入线上请求路径。
- 不强制 CI 访问真实 LLM、Milvus、MCP。
- 不做可视化 dashboard。
- 不删除旧 API 文档。

### 16.12 阶段风险点

- 集成测试 lifespan 触发真实外部服务。
- LLM judge 成本和波动。
- CI 环境缺少服务依赖。
- README 容易把规划中能力写成已完成能力。

## 17. 全局测试矩阵

### 17.1 单元测试

| 测试对象 | 覆盖点 | 建议文件 | 命令 |
| --- | --- | --- | --- |
| InputGuard | 空输入、超长、非法 session、prompt injection marker、文件/目录安全 | `tests/unit/test_input_guard.py` | `pytest tests/unit/test_input_guard.py -q` |
| AppError | 错误码、HTTP status、旧字段、脱敏 | `tests/unit/test_app_errors.py` | `pytest tests/unit/test_app_errors.py -q` |
| RequestContext | header 透传、非法 header、默认 user/tenant、deadline | `tests/unit/test_request_context.py` | `pytest tests/unit/test_request_context.py -q` |
| TraceLogger | JSONL 写入、失败降级、敏感字段脱敏 | `tests/unit/test_trace_logger.py` | `pytest tests/unit/test_trace_logger.py -q` |
| ToolManager | local/BaseTool/MCP、timeout、异常、裁剪、isError | `tests/unit/test_tool_manager.py` | `pytest tests/unit/test_tool_manager.py -q` |
| FallbackManager | 策略矩阵、reason_code、安全文案 | `tests/unit/test_fallback.py` | `pytest tests/unit/test_fallback.py -q` |
| TokenBudgetManager | 估算、预算、裁剪顺序、usage/cost | `tests/unit/test_token_budget.py` | `pytest tests/unit/test_token_budget.py -q` |
| ConversationManager | MemorySaver 边界、最近 N 轮、清理、历史转换 | `tests/unit/test_conversation_manager.py` | `pytest tests/unit/test_conversation_manager.py -q` |
| RAG models | metadata、ID、score、citation 转换 | `tests/rag/test_rag_models.py` | `pytest tests/rag/test_rag_models.py -q` |
| RagRetriever | candidate/final/min_score、空检索、低分、reranker 回退 | `tests/rag/test_retriever.py` | `pytest tests/rag/test_retriever.py -q` |
| ContextBuilder | 去重、packing、预算裁剪、错误工具隔离 | `tests/rag/test_context_builder.py` | `pytest tests/rag/test_context_builder.py -q` |
| CitationBuilder | API schema、preview、路径安全、重复合并 | `tests/rag/test_citation_builder.py` | `pytest tests/rag/test_citation_builder.py -q` |

### 17.2 Agent 边界测试

| 场景 | 验证点 | 文件 | 命令 |
| --- | --- | --- | --- |
| 最大步骤停止 | `MAX_STEPS` 和业务错误映射 | `tests/agent/test_agent_boundaries.py` | `pytest tests/agent/test_agent_boundaries.py -q` |
| recursion_limit 生效 | LangGraph config 限制可触发 | `tests/agent/test_agent_boundaries.py` | 同上 |
| 最大工具调用次数 | ToolManager per-request counter | `tests/agent/test_agent_boundaries.py` | 同上 |
| 工具 timeout | 单工具超时映射 `TOOL_TIMEOUT` | `tests/agent/test_agent_boundaries.py` | 同上 |
| SSE 中途失败 | error event schema | `tests/agent/test_sse_failures.py` | `pytest tests/agent/test_sse_failures.py -q` |
| client disconnect | generator 停止和 trace | `tests/agent/test_sse_failures.py` | 同上 |

### 17.3 文件/目录测试

| 场景 | 验证点 | 文件 |
| --- | --- | --- |
| 非法扩展名 | `UNSUPPORTED_FILE_TYPE` | `tests/unit/test_input_guard.py`、`tests/integration/test_upload_api.py` |
| MIME 不匹配 | `INVALID_FILE_MIME` | 同上 |
| 非 UTF-8 | `INVALID_FILE_ENCODING` | 同上 |
| 文件过大 | `FILE_TOO_LARGE` | 同上 |
| 文件名路径逃逸 | 不写出 upload_dir | 同上 |
| allowlist 外目录 | `INVALID_DIRECTORY` 或 `PATH_TRAVERSAL_BLOCKED` | 同上 |
| symlink | `SYMLINK_NOT_ALLOWED` | 同上 |
| path traversal | `PATH_TRAVERSAL_BLOCKED` | 同上 |
| 部分文件失败 | `partial_success` 和 `failed_files` | `tests/integration/test_upload_api.py` |

### 17.4 RAG 测试

| 场景 | 验证点 | 文件 |
| --- | --- | --- |
| metadata 稳定性 | doc_id/chunk_id/content_hash | `tests/rag/test_metadata_indexing.py` |
| 重复入库幂等 | 无重复残留 | `tests/rag/test_metadata_indexing.py` |
| doc-level delete | 按 doc_id 删除无残留 | `tests/rag/test_metadata_indexing.py` |
| 空检索 | no-answer | `tests/rag/test_retriever.py` |
| 低分检索 | min_score 过滤 | `tests/rag/test_retriever.py` |
| 重复 chunk 去重 | 保留最高分或最短版本 | `tests/rag/test_context_builder.py` |
| citation 输出 | API-safe citations | `tests/rag/test_citation_builder.py` |
| no-answer | 空、低分、无证据 | `tests/rag/test_rag_pipeline.py` |
| Hit@K/Recall@K/MRR | 轻量 baseline | `tests/rag/test_retrieval_metrics.py` |

### 17.5 API 集成测试

| API | 覆盖点 | 文件 |
| --- | --- | --- |
| `/api/chat` | alias、成功、失败、fallback、trace | `tests/integration/test_chat_api.py` |
| `/api/chat_stream` | start/token/error/done、data.type | `tests/integration/test_chat_api.py` |
| `/api/aiops` | plan、step_complete、report、complete/done、error | `tests/integration/test_aiops_api.py` |
| `/api/upload` | 旧上传路径 | `tests/integration/test_upload_api.py` |
| `/api/file/upload` | 新别名路径 | `tests/integration/test_upload_api.py` |
| `/api/index_directory` | 旧目录索引路径 | `tests/integration/test_upload_api.py` |
| `/api/file/index_directory` | 新别名路径 | `tests/integration/test_upload_api.py` |
| `/api/health` | Milvus healthy/unhealthy | `tests/integration/test_health_api.py` |
| `/api/chat/clear` | sessionId/session_id | `tests/integration/test_health_api.py` |
| `/api/chat/session/{session_id}` | 旧顶层字段 | `tests/integration/test_health_api.py` |

### 17.6 完整测试命令

```bash
pytest tests/unit -q
pytest tests/agent -q
pytest tests/rag -q
pytest tests/integration -q
pytest tests -q
python -m evaluation.runner --dataset eval_sets/rag_cases.yaml
```

## 18. 全局验收清单

- [ ] 阶段 1A 到阶段 4 均按顺序完成。
- [ ] 所有新增文件均在对应 issue 中创建，并标注为新增。
- [ ] 所有修改文件都有修改目的和兼容风险说明。
- [ ] `/api/upload`、`/api/index_directory` 保留。
- [ ] `/api/file/upload`、`/api/file/index_directory` 仅作为兼容别名。
- [ ] `/api/chat` 保留 `Id/Question`，兼容 `id/question`，alias 优先级不变。
- [ ] `/api/chat/clear` 保留 `sessionId`，兼容 `session_id`。
- [ ] 旧响应字段 `code/message/data`、`data.success/answer/errorMessage`、`status/message/data` 未删除。
- [ ] SSE `event: message` + `data.type` 兼容。
- [ ] AIOps `complete` 兼容 `done`。
- [ ] 成功和失败响应均包含或能关联 `trace_id/request_id`。
- [ ] SSE `start/error/done` 均包含 `trace_id/request_id`。
- [ ] AppError 覆盖 P0 错误码。
- [ ] RequestContext 在 API、Agent、Tool、RAG 间传播。
- [ ] TraceLogger 和 MetricsRecorder 不泄漏敏感字段。
- [ ] InputGuard 覆盖输入、上传、目录路径安全。
- [ ] ToolManager 覆盖本地工具、LangChain BaseTool、MCP tool。
- [ ] ToolResult 错误结果不作为事实证据。
- [ ] Agent 执行上限全部可测。
- [ ] FallbackManager 不做 retry。
- [ ] TokenBudgetManager 裁剪顺序符合内部契约。
- [ ] ConversationManager 与 MemorySaver 职责边界清晰。
- [ ] ConversationSummarizer 摘要安全可控。
- [ ] RAG metadata 包含 doc_id、chunk_id、content_hash、tenant_id、version。
- [ ] 索引幂等和 doc-level delete 通过测试。
- [ ] RagRetriever 支持 candidate_k、final_k、min_score、score normalization。
- [ ] ContextBuilder 支持去重、packing、预算裁剪和 prompt injection 隔离。
- [ ] CitationBuilder 输出 API-safe citations。
- [ ] No-answer 策略覆盖空检索和低分检索。
- [ ] Reranker 默认关闭且失败回退。
- [ ] RAG eval set 可加载，Hit@K/Recall@K/MRR 可计算。
- [ ] evaluation runner 输出 JSON/Markdown 报告。
- [ ] README/config/pyproject 不一致项已修复或明确说明。
- [ ] MCP README 与代码不一致项已修复或明确说明。
- [ ] 全量命令 `pytest tests -q` 可运行。
- [ ] 评估命令 `python -m evaluation.runner --dataset eval_sets/rag_cases.yaml` 可运行。

## 19. 回滚与风险控制

| 层级 | 回滚方式 | 保留项 |
| --- | --- | --- |
| 1A 错误/trace/input | 关闭 `trace_enabled`，移除 InputGuard 调用，保留旧 API handler。 | 旧路径、旧响应字段。 |
| 1B ToolManager | `tool_manager_enabled=false`，恢复旧工具列表。 | MCP retry interceptor。 |
| 1B Agent 上限 | 增大或关闭 timeout/max_tool_calls/recursion_limit。 | `MAX_STEPS=8` 业务上限。 |
| 2 Fallback | `fallback_enabled=false` 回结构化错误。 | AppError。 |
| 2 TokenBudget | `token_budget_enabled=false` 使用旧 prompt。 | usage trace 可关闭。 |
| 2 Conversation | `conversation_manager_enabled=false` 回旧 MemorySaver 读取。 | Clear/session API。 |
| 3A Stable IDs | `stable_rag_ids_enabled=false` 回 UUID。 | 旧 `_source` metadata。 |
| 3A Delete | 回旧 `delete_by_source`。 | doc_id metadata 保留。 |
| 3B Retriever | `new_rag_retriever_enabled=false` 回旧 `retrieve_knowledge`。 | RAG models 不影响旧路径。 |
| 3B Citation | `citations_enabled=false` 返回空数组或不追加。 | 旧 answer 字段。 |
| 3B Reranker | `reranker_enabled=false`。 | vector 排序。 |
| 4 Metrics/Judge | `metrics_enabled=false`、`judge_enabled=false`。 | 最小 trace。 |
| 4 CI/README | 文档或 workflow 单独 revert。 | 代码不受影响。 |

## 20. 开发提交顺序建议

每个 issue 一个提交。建议提交信息：

```text
feat(errors): add app error envelope
feat(trace): add request context middleware
feat(input): add input guard
feat(file): harden upload and directory indexing
test(fixtures): add fake fixtures for unit tests
feat(tools): add tool manager
feat(tools): add tool policy registry
feat(agent): route tools through tool manager
feat(agent): enforce execution limits and sse errors
test(agent): cover tool and agent boundaries
feat(fallback): add fallback manager
feat(tokens): add token budget manager
feat(memory): add conversation manager
feat(memory): add conversation summarizer
feat(agent): add thin orchestrator
test(stage2): add fallback token memory regressions
feat(rag): add rag internal models
feat(rag): generate stable doc and chunk ids
feat(rag): make indexing idempotent
test(eval): add minimal rag eval set
feat(eval): add rag datasets and metrics
test(rag): cover metadata indexing
feat(rag): add retriever
feat(rag): normalize scores and no-answer
feat(rag): add context builder
feat(rag): add citation builder
feat(rag): add noop reranker
test(rag): add pipeline baseline regression
feat(obs): add metrics recorder
test(api): add chat and health integration tests
test(api): add aiops integration tests
test(api): add upload integration tests
feat(eval): add llm judge and runner
feat(eval): add json and markdown reports
docs(ci): update readme and regression commands
```

当前工作目录不是 Git 仓库；上述提交顺序供开发人员在真实 Git 工作区执行。

## 21. 后续维护规则

1. 任何对外 API 字段、HTTP status、SSE event、错误码、trace_id/request_id、fallback_used、citations 的变更，必须同步更新 `docs/api_contract.md` 和集成测试。
2. 任何内部模块职责、输入输出、错误处理、trace 字段的变更，必须同步更新 `docs/internal_contract.md`。
3. 新增工具必须先经过 ToolManager 和 ToolPolicy，再进入 Agent。
4. 新增 RAG metadata 字段必须保持向后兼容，不得删除旧 `_source`。
5. RAG pipeline 参数调整必须运行 `pytest tests/rag -q` 和 evaluation runner。
6. README 只能描述已完成能力；规划中能力必须标注阶段和状态。
7. MCP README 必须与 `mcp_servers/*.py` 实际工具清单一致。
8. 新增评估报告默认不提交临时报告，只提交必要 baseline。
9. 每个阶段结束必须补阶段验收记录和测试命令输出。
10. 每次修复兼容性问题，必须增加回归测试。
