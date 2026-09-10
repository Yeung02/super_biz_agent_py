# AegisOps Agent 内部模块契约文档

> 本文档定义系统内部模块之间如何协作，不定义对外 HTTP API 或 SSE event。对外契约以 `docs/api_contract.md` 为准；内部模块契约以本文档为准。内部模块变更如果影响 HTTP status、响应字段、SSE event、错误码、`trace_id/request_id`、`fallback_used`、`citations` 等对外表现，必须同步更新 `docs/api_contract.md`。

## 1. 文档范围

### 1.1 与 API 契约的边界

1. `docs/api_contract.md` 只描述对外 HTTP/SSE 契约，包括路由、请求字段、响应 envelope、错误码、SSE event、兼容策略和外显 fallback。
2. 本文档只描述内部模块契约，包括模块职责、输入输出、核心类方法、依赖、错误处理、trace、测试方式和验收标准。
3. 内部模块不得被前端或外部调用方直接依赖。内部 schema 可以比 API schema 更丰富，但对外输出必须经过 API 层或 adapter 转换。
4. 内部模块变更如果改变对外 HTTP 响应或 SSE 事件，必须同步更新 `docs/api_contract.md`，并补充对应测试断言。

### 1.2 阶段口径

| 阶段 | 目标 |
| --- | --- |
| 1A | 统一错误模型、请求上下文、输入边界、最小 trace。 |
| 1B | 统一工具调用边界、工具结果 envelope、Agent 执行上限。 |
| 2 | fallback、token 预算、可控会话上下文。 |
| 3A | 稳定 RAG 数据模型、metadata、doc/chunk/citation 内部 schema。 |
| 3B | 可控 RAG 检索、context packing、citation 输出、no-answer。 |

### 1.3 模块协作主链路

```text
API handler
  -> RequestContext
  -> TraceLogger
  -> InputGuard
  -> ConversationManager
  -> TokenBudgetManager
  -> RagRetriever / ToolManager
  -> ContextBuilder
  -> CitationBuilder
  -> LLM / Agent
  -> FallbackManager
  -> API response adapter
```

错误处理主链路：

```text
Exception
  -> AppError
  -> TraceLogger.record_error
  -> FallbackManager.decide (仅适用可降级场景)
  -> API/SSE adapter
```

## 2. 全局内部约束

### 2.1 错误和事实证据

`ToolManager` 返回的工具错误结果不能作为正常事实证据。任何 `ToolResult.is_error=true`、`ToolResult.status in ["error", "timeout", "unauthorized"]` 或 MCP `isError=true` 的结果，只能用于：

- 触发 retry、fallback 或 no-answer。
- 写入 trace 和诊断元数据。
- 在最终回答中说明“某工具调用失败”，但不得把错误文本当作业务事实引用。

### 2.2 ConversationManager 与 MemorySaver 边界

`MemorySaver` 是 LangGraph checkpoint/thread 的底层状态存储，只负责保存和读取原始 graph checkpoint。`ConversationManager` 是业务上下文门面，负责把原始历史变成可控、可裁剪、可摘要、可审计的上下文输入。

| 能力 | ConversationManager | MemorySaver |
| --- | --- | --- |
| 业务会话 ID 校验 | 负责 | 不负责 |
| 最近 N 轮选择 | 负责 | 不负责 |
| 历史摘要生成和读取 | 负责 | 不负责 |
| token 预算内裁剪 | 负责，依赖 TokenBudgetManager | 不负责 |
| LangGraph checkpoint 持久化 | 不直接实现 | 负责 |
| 原始 checkpoint 删除 | 通过门面调用 | 执行底层删除 |
| 对外历史响应格式 | 负责转换 | 不负责 |

### 2.3 TokenBudget 裁剪优先级

当输入超过预算时，必须按以下优先级裁剪，越靠前越先裁剪：

1. 调试信息、debug event、非用户可见内部备注。
2. 成功工具结果的冗余字段、大 JSON、重复列表项、原始 payload。
3. 低相关 RAG chunk，按归一化相关度从低到高裁剪。
4. 重复或近重复 RAG chunk，保留最高分或最短可引用版本。
5. 旧对话历史，优先保留摘要和最近完整轮次。
6. 历史摘要正文，按句子边界压缩到摘要预算。
7. 当前用户问题和系统安全约束不得裁剪；只能拒绝、fallback 或返回 `REQUEST_TOO_LARGE`。

### 2.4 RAG citation schema 分层

RAG citation 有两层 schema：

- 内部 schema：`app/rag/models.py::Citation`，用于模块间传递，允许包含 `span_start/span_end/raw_score/normalized_score/evidence_text/metadata` 等内部字段。
- API 输出 schema：`docs/api_contract.md` 中的 `citations[]`，只返回稳定、安全、用户可见字段。

转换规则：

| 内部字段 | API 字段 | 规则 |
| --- | --- | --- |
| `citation_id` | `citation_id` | 保持一致，按 `C1`、`C2` 递增。 |
| `doc_id` | `doc_id` | 阶段 3A 后必填。 |
| `chunk_id` | `chunk_id` | 阶段 3A 后必填。 |
| `source_path` | `source_path` | 必须是相对安全路径，不输出任意绝对路径。 |
| `file_name` | `file_name` | 从 metadata 或 source path 派生。 |
| `normalized_score` | `score` | 输出 `0.0-1.0`，不输出 L2 原始距离语义。 |
| `evidence_text` | `content_preview` | 截断到 API 契约长度，不泄露完整 chunk。 |
| `raw_score`、`metadata`、`span_*` | 不输出 | 只写 trace 或内部评估。 |

## 3. 模块契约

### 3.1 AppError

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1A |
| 是否新增 | 是 |
| 文件路径 | `app/core/errors.py` |
| 职责 | 提供统一内部错误模型；承载稳定 `code`、`http_status`、`user_message`、`internal_message`、`retryable`、`fallback_required`、`details`；把 Python 异常转换为可追踪、可映射的业务错误。 |
| 不负责什么 | 不负责直接写 HTTP response；不负责决定 fallback 文案；不负责吞掉异常；不负责记录完整 trace span。 |
| 输入 | 原始 `Exception`、模块主动抛出的业务错误参数、`RequestContext` 中的 trace/request 信息。 |
| 输出 | `AppError` 实例；`ErrorEnvelope`/dict；HTTP status 映射数据。 |
| 核心类 | `AppError`、`InvalidInputError`、`ToolTimeoutError`、`ToolExecutionError`、`LLMProviderError`、`VectorStoreUnavailableError`、`RagEmptyResultError`、`InternalAppError`。 |
| 核心方法 | `to_error_response(ctx)`、`from_exception(exc)`、`is_retryable()`、`requires_fallback()`、`safe_details()`。 |
| 依赖模块 | `RequestContext`、`TraceLogger`、FastAPI/Pydantic response adapter。 |
| 错误处理 | 未分类异常统一包装为 `INTERNAL_ERROR`；用户可见文案不得包含堆栈、密钥、内部 URL；`details` 写入前必须脱敏。 |
| trace 记录 | 每个 `AppError` 必须记录 `error.code`、`http_status`、`retryable`、`fallback_required`、`origin_module`、`trace_id`。 |
| 测试方式 | 单测错误码映射、HTTP status 映射、脱敏、`from_exception` 包装、兼容旧响应字段。 |
| 验收标准 | P0 错误都能稳定转换为统一 envelope；成功和失败路径都能关联 `trace_id/request_id`；对外错误码与 `docs/api_contract.md` 一致。 |

### 3.2 RequestContext

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1A |
| 是否新增 | 是 |
| 文件路径 | `app/core/request_context.py` |
| 职责 | 为一次请求创建和传递内部上下文；统一 `trace_id`、`request_id`、`session_id`、`tenant_id`、`user_id`、`deadline_ms`、`feature_flags`、`started_at`。 |
| 不负责什么 | 不负责鉴权系统；不负责解析业务 body；不负责直接裁剪 token；不负责持久化会话历史。 |
| 输入 | FastAPI `Request` headers/path/body 中已解析字段；服务端默认配置；调用方显式传入的 session_id。 |
| 输出 | `RequestContext` 实例；`request.state.ctx`；供内部模块使用的只读上下文。 |
| 核心类 | `RequestContext`、`RequestContextMiddleware`。 |
| 核心方法 | `from_request(request)`、`get_request_context()`、`with_session(session_id)`、`remaining_ms()`、`to_trace_fields()`。 |
| 依赖模块 | FastAPI request lifecycle、`InputGuard`、`TraceLogger`、`AppError`。 |
| 错误处理 | 非法 inbound `X-Trace-Id`/`X-Request-Id` 必须丢弃并重新生成；缺省用户使用 `anonymous`，缺省租户使用 `default`。 |
| trace 记录 | 请求开始记录 `request.start`；结束记录 `request.end`，包含 latency、status、error_code；非法 header 记录 `invalid_inbound_trace_header=true`。 |
| 测试方式 | FastAPI TestClient 中间件测试；header 透传测试；非法 header 重生成测试；默认 user/tenant 测试。 |
| 验收标准 | 每个请求都有稳定 `trace_id/request_id`；内部模块不再自行生成相互无关的 trace id；响应 header/body/SSE 可使用同一上下文值。 |

### 3.3 InputGuard

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1A |
| 是否新增 | 是 |
| 文件路径 | `app/core/input_guard.py` |
| 职责 | 统一校验用户输入、session_id、上传文件、目录路径、MIME、UTF-8、大小限制和 prompt injection 风险标记。 |
| 不负责什么 | 不负责业务理解；不负责调用 Agent；不负责清洗工具结果；不负责替代权限策略。 |
| 输入 | `ChatRequest`、`AIOpsRequest`、`ClearRequest`、multipart file metadata/content、`directory_path`、`RequestContext`。 |
| 输出 | `GuardResult`；规范化后的 `ValidatedChatInput`、`ValidatedFileInput`、`ValidatedDirectoryInput`；失败时抛 `AppError` 子类。 |
| 核心类 | `InputGuard`、`GuardResult`、`ValidatedChatInput`、`ValidatedFileInput`、`ValidatedDirectoryInput`。 |
| 核心方法 | `validate_chat(request, ctx)`、`validate_aiops(request, ctx)`、`validate_session_id(session_id)`、`validate_upload(file, ctx)`、`validate_directory(path, ctx)`、`detect_prompt_injection(text)`。 |
| 依赖模块 | `AppError`、`RequestContext`、`config`、`pathlib`、MIME sniff 工具。 |
| 错误处理 | 空输入抛 `INVALID_INPUT`；非法 session 抛 `INVALID_SESSION_ID`；超长抛 `REQUEST_TOO_LARGE`；路径逃逸抛 `PATH_TRAVERSAL_BLOCKED`；symlink 抛 `SYMLINK_NOT_ALLOWED`；MIME/编码错误抛对应错误。 |
| trace 记录 | 记录 `input.validated`、`input.rejected`、`input.length`、`prompt_injection_risk`、`file.size`、`directory.allowed_root`，不记录完整敏感输入。 |
| 测试方式 | 单测空输入、超长输入、非法 session、大小写 alias、非法扩展名、MIME 不匹配、非 UTF-8、path traversal、symlink。 |
| 验收标准 | 非法输入不进入 Agent/RAG/Tool；文件和目录边界稳定可测；错误码与 API 契约一致。 |

### 3.4 TraceLogger

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1A |
| 是否新增 | 是 |
| 文件路径 | `app/observability/tracing.py` |
| 职责 | 提供最小结构化 trace/span 写入；串联请求、输入校验、RAG、工具、LLM、fallback、token 和错误。 |
| 不负责什么 | 不负责业务决策；不负责指标聚合平台；不负责把 trace 暴露为 HTTP API；不负责保存敏感原文。 |
| 输入 | `RequestContext`、span 名称、模块事件字段、错误对象、usage/cost/citation 元数据。 |
| 输出 | JSONL trace event；可选内存 span 对象；用于日志关联的字段 dict。 |
| 核心类 | `TraceLogger`、`TraceSpan`。 |
| 核心方法 | `start_span(name, ctx, **fields)`、`end_span(span, **fields)`、`record_event(name, ctx, **fields)`、`record_error(error, ctx)`、`record_usage(ctx, usage)`。 |
| 依赖模块 | `RequestContext`、`AppError`、`app/utils/logger.py`、后续 `MetricsRecorder`。 |
| 错误处理 | trace 写入失败不得影响主流程；写入异常降级为普通 logger warning；敏感字段写入前脱敏。 |
| trace 记录 | 自身负责生成标准 trace event，字段至少包含 `trace_id`、`request_id`、`span_id`、`parent_span_id`、`name`、`status`、`latency_ms`、`error_code`。 |
| 测试方式 | 使用临时 JSONL 文件测试成功/失败 span；测试 trace 写入失败不影响业务；测试敏感字段脱敏。 |
| 验收标准 | `/api/chat` 成功、失败、fallback 至少各有一条可串联 trace；工具和 RAG 事件能按 `trace_id` 查询。 |

### 3.5 ToolManager

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1B |
| 是否新增 | 是 |
| 文件路径 | `app/agent/tool_manager.py` |
| 职责 | 统一包装本地工具、LangChain `BaseTool` 和 MCP tool；执行权限校验、timeout、重试边界、结果裁剪、schema 校验和 trace span。 |
| 不负责什么 | 不负责决定工具调用计划；不负责把工具错误当知识证据；不负责 RAG citation；不负责 fallback 文案。 |
| 输入 | 工具名、工具参数、`RequestContext`、`ToolPolicy`、timeout/retry/trim 配置。 |
| 输出 | `ToolResult`；适配为 LangChain tool 返回值；失败时返回结构化错误结果或抛 `AppError`，由调用模式决定。 |
| 核心类 | `ToolManager`、`ToolCallSpec`、`ToolRegistry`。 |
| 核心方法 | `register(tool)`、`wrap_langchain_tool(tool)`、`wrap_mcp_tool(tool)`、`invoke(tool_name, args, ctx)`、`ainvoke(tool_name, args, ctx)`、`trim_result(result, budget)`、`validate_result_schema(result)`。 |
| 依赖模块 | `ToolResult`、`AppError`、`TraceLogger`、`RequestContext`、`PolicyRegistry`、`TokenBudgetManager`。 |
| 错误处理 | timeout 映射 `TOOL_TIMEOUT`；未授权内部异常对外默认映射为 `TOOL_EXECUTION_ERROR`，如未来新增独立 HTTP 错误码必须同步更新 `docs/api_contract.md`；MCP `isError=true` 必须转换为 `ToolResult.is_error=true`；错误结果不得注入事实证据链。 |
| trace 记录 | 每次工具调用记录 `tool.start`、`tool.end` 或 `tool.error`，包含 tool_name、latency、status、trimmed、input_preview、output_preview、error_code。 |
| 测试方式 | fake 本地工具、fake MCP tool 覆盖成功、异常、timeout、未授权、大 JSON、`isError=true`、结果裁剪。 |
| 验收标准 | 所有本地/MCP 工具输出统一为 `ToolResult`；错误工具结果不会被 `ContextBuilder` 或最终回答当作正常事实证据；trace 能看到每次工具调用。 |

### 3.6 ToolResult

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 1B |
| 是否新增 | 是 |
| 文件路径 | `app/agent/tool_manager.py` |
| 职责 | 作为内部工具调用统一结果 envelope；区分正常数据、错误、超时、未授权、裁剪状态和证据可用性。 |
| 不负责什么 | 不负责执行工具；不负责决定是否 fallback；不负责对外 HTTP schema；不负责 citation 编号。 |
| 输入 | 工具原始返回值、MCP `CallToolResult`、异常对象、timeout 状态、裁剪信息。 |
| 输出 | `ToolResult` 实例；可序列化 dict；供 Agent prompt adapter 或 trace 使用的 preview。 |
| 核心类 | `ToolResult`、`ToolErrorInfo`。 |
| 核心方法 | `success(tool_name, data, metadata)`、`error(tool_name, error)`、`from_mcp_result(result)`、`as_prompt_block()`、`to_trace_fields()`、`is_evidence_usable()`。 |
| 依赖模块 | `AppError`、`TraceLogger`、`TokenBudgetManager`。 |
| 错误处理 | `is_error=true` 时必须设置 `error.code`、`error.message`、`retryable`；`is_evidence_usable()` 必须返回 false；错误文本只允许进入诊断说明，不允许进入事实上下文。 |
| trace 记录 | 输出 `tool_result.status`、`is_error`、`evidence_usable`、`trimmed`、`raw_size`、`preview_size`。 |
| 测试方式 | 构造成功/失败/MCP isError/裁剪结果，断言 `is_evidence_usable()` 和 prompt block 行为。 |
| 验收标准 | 所有工具结果都有明确 `status`；错误结果无法被误判为检索证据；大结果被裁剪后仍保留可追踪 metadata。 |

### 3.7 FallbackManager

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 2 |
| 是否新增 | 是 |
| 文件路径 | `app/core/fallback.py` |
| 职责 | 根据 `AppError`、局部证据、请求类型和上下文决定是否降级、如何返回用户安全文案、是否保留部分结果。 |
| 不负责什么 | 不负责 retry；不负责执行工具；不负责隐藏输入校验错误；不负责健康检查 fallback。 |
| 输入 | `AppError`、`RequestContext`、可用证据摘要、RAG chunks、工具结果列表、调用场景（chat/chat_stream/aiops）。 |
| 输出 | `FallbackResult`，包含 `fallback_used`、`reason_code`、`safe_message`、`partial_answer`、`should_continue_stream`。 |
| 核心类 | `FallbackManager`、`FallbackResult`、`FallbackPolicy`。 |
| 核心方法 | `decide(error, ctx, evidence)`、`for_chat(error, ctx, evidence)`、`for_aiops(error, ctx, evidence)`、`to_sse_event(result)`、`to_response_fields(result)`。 |
| 依赖模块 | `AppError`、`TraceLogger`、`RequestContext`、`ToolResult`、RAG models。 |
| 错误处理 | 输入校验类错误通常不 fallback；LLM/RAG/Tool 失败可 fallback；fallback 文案不得包含堆栈、密钥、内部 URL 或原始异常全文。 |
| trace 记录 | 记录 `fallback.decide`，包含 `reason_code`、`fallback_used`、`evidence_count`、`partial_answer` 是否存在。 |
| 测试方式 | 单测 LLM timeout、RAG empty、tool timeout、vector store unavailable、SSE 中途失败、输入错误不 fallback。 |
| 验收标准 | fallback 对外只通过 API 契约字段外显；失败恢复有稳定 reason_code；不会把未验证工具错误包装成事实回答。 |

### 3.8 TokenBudgetManager

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 2 |
| 是否新增 | 是 |
| 文件路径 | `app/core/token_budget.py` |
| 职责 | 统一估算和分配输入/输出 token 预算；裁剪历史、RAG chunk、工具结果和摘要；记录 usage/cost。 |
| 不负责什么 | 不负责检索排序本身；不负责生成摘要内容；不负责决定 HTTP status；不负责修改原始 MemorySaver checkpoint。 |
| 输入 | 模型名、请求场景、系统提示、当前问题、历史消息、摘要、RAG chunks、工具结果、配置预算。 |
| 输出 | `TokenBudget`、`TrimResult`、裁剪后的上下文组件、usage 估算。 |
| 核心类 | `TokenBudgetManager`、`TokenBudget`、`BudgetAllocation`、`TrimResult`。 |
| 核心方法 | `allocate(scenario, model, ctx)`、`estimate_tokens(text_or_messages)`、`trim_messages(messages, budget)`、`trim_chunks(chunks, budget)`、`trim_tool_result(result, budget)`、`record_usage(usage, ctx)`。 |
| 依赖模块 | `RequestContext`、`TraceLogger`、`ConversationManager`、`ToolResult`、RAG models、config。 |
| 错误处理 | 无法精确 tokenizer 时降级为字符估算；当前问题超过硬上限时抛 `REQUEST_TOO_LARGE`；预算不足时返回 no-answer/fallback 而不是静默截断关键约束。 |
| trace 记录 | 记录 `token.allocate`、`token.trim`、`token.usage`，包含各部分预算、裁剪数量、估算方式、input/output tokens。 |
| 测试方式 | 单测 tokenizer 降级、长历史裁剪、RAG chunk 裁剪、工具结果裁剪、当前问题超限、裁剪优先级。 |
| 验收标准 | 20 轮长对话不会把全部历史塞入 LLM；裁剪顺序符合本文档 2.3；trace 中能看到预算分配和裁剪结果。 |

### 3.9 ConversationManager

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 2 |
| 是否新增 | 是 |
| 文件路径 | `app/memory/conversation_manager.py` |
| 职责 | 封装业务会话上下文读取、摘要、最近轮次选择、历史保存和清理；隔离 API/Agent 与底层 `MemorySaver`。 |
| 不负责什么 | 不负责底层 checkpoint 实现；不负责长期记忆检索；不负责 token 估算算法；不负责直接对外暴露原始 LangGraph message。 |
| 输入 | `session_id`、`RequestContext`、`TokenBudget`、当前用户消息、Agent/LLM 输出、底层 MemorySaver handle。 |
| 输出 | `ConversationContext`，包含 `summary`、`recent_messages`、`history_metadata`；保存结果；清理结果。 |
| 核心类 | `ConversationManager`、`ConversationContext`、`ConversationTurn`。 |
| 核心方法 | `load_context(session_id, budget, ctx)`、`save_turn(session_id, user_msg, assistant_msg, metadata, ctx)`、`summarize_if_needed(session_id, budget, ctx)`、`clear_session(session_id, ctx)`、`get_history(session_id, ctx)`。 |
| 依赖模块 | `MemorySaver`、`TokenBudgetManager`、`ConversationSummarizer`、`InputGuard`、`TraceLogger`、`AppError`。 |
| 错误处理 | MemorySaver 读取失败记录 trace 并可返回空历史或抛 `INTERNAL_ERROR`，按调用场景决定；摘要失败不应删除原始 checkpoint；清理失败返回 false 或包装错误。 |
| trace 记录 | 记录 `conversation.load`、`conversation.summary`、`conversation.save`、`conversation.clear`，包含 message_count、summary_used、trimmed_count。 |
| 测试方式 | fake MemorySaver 测试空历史、长历史、摘要触发、清理、底层异常、预算裁剪。 |
| 验收标准 | 业务层不直接解析 MemorySaver checkpoint；返回给 Agent 的历史受预算控制；对外历史查询不泄露系统消息和工具原始 payload。 |

### 3.10 RAG models

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 3A |
| 是否新增 | 是 |
| 文件路径 | `app/rag/models.py` |
| 职责 | 定义 RAG 内部稳定数据模型，包括文档、chunk、检索结果、上下文包、citation、no-answer 决策和评估字段。 |
| 不负责什么 | 不负责执行检索；不负责写入 Milvus；不负责生成 API 响应；不负责调用 LLM。 |
| 输入 | 文档切分 metadata、Milvus search result、LangChain `Document`、检索分数、source path。 |
| 输出 | `DocumentRecord`、`ChunkRecord`、`RetrievedChunk`、`RagContext`、`Citation`、`NoAnswerDecision`。 |
| 核心类 | `DocumentRecord`、`ChunkRecord`、`RetrievedChunk`、`RagContext`、`Citation`、`NoAnswerDecision`。 |
| 核心方法 | `from_langchain_document(doc)`、`from_search_result(result)`、`normalize_score(raw_score, metric)`、`to_api_citation()`、`validate_metadata()`。 |
| 依赖模块 | Pydantic、`AppError`、vector search service、document splitter/indexing service。 |
| 错误处理 | 缺少关键 metadata 时抛 `RAG_METADATA_INVALID` 或在阶段迁移期降级填充兼容字段；分数语义必须标明 metric，避免把 L2 距离直接当相似度。 |
| trace 记录 | 记录 doc_id、chunk_id、source_path、raw_score、normalized_score、metadata_version，不记录完整 chunk 内容。 |
| 测试方式 | 单测 doc_id/chunk_id 稳定性、旧 metadata 兼容、分数归一化、API citation 转换、字段校验。 |
| 验收标准 | 同一文档重复入库产生稳定 doc/chunk 标识；citation 和 eval 都基于同一内部模型；API 输出由内部模型安全转换。 |

### 3.11 RagRetriever

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 3B |
| 是否新增 | 是 |
| 文件路径 | `app/rag/retriever.py` |
| 职责 | 提供可控 RAG 检索入口；封装 query rewrite/multi-query、向量检索、score normalization、去重、min_score/no-answer 前置判断和 reranker 插口。 |
| 不负责什么 | 不负责 context packing；不负责 citation 编号；不负责最终回答生成；不负责把工具错误作为证据。 |
| 输入 | 用户问题、`RequestContext`、`TokenBudget`、检索配置（candidate_k/final_k/min_score）、可选 filter。 |
| 输出 | `list[RetrievedChunk]`、`RetrievalResult`，包含 query variants、分数、空结果原因。 |
| 核心类 | `RagRetriever`、`RetrievalQuery`、`RetrievalResult`。 |
| 核心方法 | `retrieve(query, ctx, budget)`、`rewrite_query(query, ctx)`、`multi_query(query, ctx)`、`search(query, top_k, filters)`、`deduplicate(chunks)`、`apply_threshold(chunks)`。 |
| 依赖模块 | `app/services/vector_search_service.py`、`TokenBudgetManager`、RAG models、`TraceLogger`、可选 `Reranker`。 |
| 错误处理 | 向量库不可用抛 `VECTOR_STORE_UNAVAILABLE`；空结果返回 `RAG_EMPTY_RESULT` 或 `RetrievalResult.empty`；低分结果不得伪装成有效证据。 |
| trace 记录 | 记录 `rag.retrieve.start/end`、query variants、candidate_count、final_count、min_score、empty_reason、latency。 |
| 测试方式 | fake vector search 覆盖正常、空结果、低分、重复 chunk、向量库异常、reranker 失败回退。 |
| 验收标准 | 检索结果分数语义明确；低置信结果能触发 no-answer/fallback；输出只包含可被 ContextBuilder 使用的有效 chunks。 |

### 3.12 ContextBuilder

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 3B |
| 是否新增 | 是 |
| 文件路径 | `app/rag/context_builder.py` |
| 职责 | 在 token 预算内构建 LLM 可用 RAG context；去重、排序、截断、保留引用锚点、隔离不可用证据。 |
| 不负责什么 | 不负责执行检索；不负责生成 citation API schema；不负责 LLM 回答；不负责使用 ToolResult 错误作为证据。 |
| 输入 | `RetrievedChunk[]`、`TokenBudget`、当前问题、可用 `ToolResult[]`、`RequestContext`。 |
| 输出 | `RagContext`，包含 packed context text、used_chunks、dropped_chunks、citation anchors、no_answer_decision。 |
| 核心类 | `ContextBuilder`、`ContextBuildResult`。 |
| 核心方法 | `build(query, chunks, budget, ctx)`、`pack_chunks(chunks, budget)`、`drop_duplicates(chunks)`、`format_context(chunks)`、`decide_no_answer(chunks)`。 |
| 依赖模块 | RAG models、`TokenBudgetManager`、`ToolResult`、`TraceLogger`。 |
| 错误处理 | 所有 `ToolResult.is_error=true` 直接排除出事实上下文；预算不足时按裁剪优先级丢弃低分 chunk；无可用证据时返回 no-answer 决策而不是编造 context。 |
| trace 记录 | 记录 `rag.context.build`，包含 input_chunk_count、used_chunk_count、dropped_chunk_count、budget_used、no_answer。 |
| 测试方式 | 单测 chunk 去重、低分丢弃、预算裁剪、工具错误排除、无证据 no-answer、context 锚点稳定。 |
| 验收标准 | LLM prompt 中的 RAG context 可追溯到 `chunk_id`；错误工具结果不进入事实证据；超预算时输出可解释裁剪结果。 |

### 3.13 CitationBuilder

| 字段 | 契约 |
| --- | --- |
| 所属阶段 | 3B |
| 是否新增 | 是 |
| 文件路径 | `app/rag/citation.py` |
| 职责 | 从 `RagContext.used_chunks` 生成稳定内部 citation，并转换为 API 安全输出 schema。 |
| 不负责什么 | 不负责检索；不负责判断 chunk 相关性；不负责修改答案正文事实；不负责输出绝对路径或完整 chunk。 |
| 输入 | `RagContext`、`RetrievedChunk[]`、答案中引用锚点、`RequestContext`。 |
| 输出 | 内部 `Citation[]`；API-safe `citations[]` dict；citation trace 字段。 |
| 核心类 | `CitationBuilder`。 |
| 核心方法 | `build(context, answer, ctx)`、`assign_ids(chunks)`、`to_api_schema(citations)`、`preview(text, max_chars)`、`validate_public_fields(citation)`。 |
| 依赖模块 | RAG models、`ContextBuilder`、`TraceLogger`、`docs/api_contract.md` 中的 citation schema。 |
| 错误处理 | 缺少 `doc_id/chunk_id` 时阶段 3A 后必须报错或拒绝输出不完整 citation；source path 非安全相对路径时拒绝输出并记录 trace。 |
| trace 记录 | 记录 `rag.citation.build`，包含 citation_count、doc_ids、chunk_ids、dropped_invalid_count。 |
| 测试方式 | 单测 citation_id 顺序、重复 chunk 合并、preview 截断、绝对路径拦截、内部 schema 到 API schema 转换。 |
| 验收标准 | API `citations[]` 与 `docs/api_contract.md` 完全兼容；内部 citation 保留评估所需字段；对外不泄露 raw metadata、绝对路径或完整 chunk。 |

## 4. 跨模块验收清单

| 验收项 | 标准 |
| --- | --- |
| API/Internal 分工 | `docs/api_contract.md` 只管 HTTP/SSE；本文档只管内部模块协作。 |
| API 同步 | 内部模块变更影响 HTTP/SSE 外显时，同步更新 `docs/api_contract.md`。 |
| 错误统一 | 所有模块失败都能映射到 `AppError` 或明确的 `ToolResult.is_error`。 |
| trace 串联 | 一次请求的 guard、tool、rag、fallback、token、error 都能用同一 `trace_id` 串联。 |
| 工具证据 | 工具错误结果不会进入 RAG context 或最终事实证据。 |
| Memory 边界 | 业务代码通过 `ConversationManager` 使用历史，不直接解析 MemorySaver checkpoint。 |
| Token 裁剪 | 裁剪顺序符合本文档 2.3，当前用户问题和系统安全约束不被静默裁剪。 |
| Citation 分层 | 内部 citation schema 可评估、可追踪；API citation schema 安全、稳定、与 `docs/api_contract.md` 一致。 |
| 测试归属 | 1A/1B/2/3A/3B 各阶段模块都有单测或 fake 夹具覆盖。 |
