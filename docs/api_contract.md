# AegisOps Agent API 接口契约文档

> 本文档只定义对外 HTTP API、SSE event、请求/响应字段、错误 envelope、trace/request 传播、fallback 外显、安全协议和兼容策略。`ToolManager`、`InputGuard`、`TokenBudgetManager`、`ConversationManager`、`RagRetriever/ContextBuilder`、`evaluation/runner.py` 等内部模块不作为 HTTP API 暴露，只在本文档中体现为错误码、`trace_id`、`citations`、`fallback_used`、SSE event 等对外影响。

## 1. 文档范围

### 1.1 当前真实接口

| 接口 | 当前代码入口 | 当前状态 |
| --- | --- | --- |
| `POST /api/chat` | `app/api/chat.py::chat` | 已存在 |
| `POST /api/chat_stream` | `app/api/chat.py::chat_stream` | 已存在 |
| `POST /api/aiops` | `app/api/aiops.py::diagnose_stream` | 已存在 |
| `POST /api/upload` | `app/api/file.py::upload_file` | 已存在，必须保留 |
| `POST /api/index_directory` | `app/api/file.py::index_directory` | 已存在，必须保留 |
| `GET /api/health` | `app/api/health.py::health_check` | 已存在 |
| `POST /api/chat/clear` | `app/api/chat.py::clear_session` | 已存在 |
| `GET /api/chat/session/{session_id}` | `app/api/chat.py::get_session_info` | 已存在 |

### 1.2 阶段 1A 兼容别名

| 接口 | 关系 | 阶段 1A 要求 |
| --- | --- | --- |
| `POST /api/file/upload` | `POST /api/upload` 的规范化别名 | 可新增；新旧路径必须返回相同响应结构 |
| `POST /api/file/index_directory` | `POST /api/index_directory` 的规范化别名 | 可新增；新旧路径必须返回相同响应结构 |

`/api/upload` 是当前真实接口，必须保留；`/api/file/upload` 是阶段 1A 可新增的规范化别名。`/api/index_directory` 是当前真实接口，必须保留；`/api/file/index_directory` 是阶段 1A 可新增的规范化别名。阶段 1A 不允许直接删除旧路径，避免前端或旧调用方失效。

## 2. 全局响应约定

### 2.1 统一成功响应建议格式

阶段 1A 起，所有非 SSE API 的成功响应建议使用统一 envelope：

```json
{
  "success": true,
  "data": {},
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

为避免破坏当前前端，阶段 1A 必须保留旧响应兼容字段。当前旧接口可能仍返回：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

阶段 1A 推荐同时返回新旧字段：

```json
{
  "success": true,
  "code": 200,
  "message": "success",
  "data": {},
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

### 2.2 统一失败响应格式

所有失败响应建议使用：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "问题不能为空。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  }
}
```

阶段 1A 为兼容旧前端，可额外保留：

```json
{
  "code": 400,
  "message": "问题不能为空。",
  "data": {
    "success": false,
    "answer": null,
    "errorMessage": "问题不能为空。"
  }
}
```

兼容字段只能追加，不得删除 `success=false`、`error.code`、`error.message`、`error.retryable`、`trace_id/request_id`。

### 2.3 HTTP status 规则

| 场景 | HTTP status |
| --- | --- |
| 请求成功 | `200` |
| 流式接口成功建立 SSE 连接 | `200` |
| 请求字段非法 | `400` |
| 请求体或问题过长 | `413` |
| 文件 MIME 不被允许 | `415` |
| 下游 LLM/Embedding/Tool 失败 | `502` 或 `504` |
| 向量库不可用 | `503` |
| 健康检查依赖不可用 | `503` |
| 未分类服务端异常 | `500` |

SSE 一旦开始发送，后续业务失败不再改变 HTTP status，必须通过 `error` event 暴露。

## 3. trace_id / request_id 返回规则

1. 所有 API 成功响应都应返回 `trace_id`。
2. 所有 API 失败响应都应返回 `trace_id`。
3. 所有 API 成功和失败响应都应返回 `request_id`。
4. SSE 的 `start` event 必须返回 `trace_id` 和 `request_id`。
5. SSE 的 `error` 和 `done` event 必须返回 `trace_id` 和 `request_id`。
6. 如果请求 header 中有 `X-Trace-Id`，服务端按设计文档规则透传为 `trace_id`；如果没有，则由服务端生成。
7. 如果请求 header 中有 `X-Request-Id`，服务端按设计文档规则透传为 `request_id`；如果没有，则由服务端生成。
8. 阶段 1A 建议同时在响应 header 中返回 `X-Trace-Id` 和 `X-Request-Id`，并在 JSON body / SSE data 中返回同值。
9. 若 header 值为空、过长或包含控制字符，服务端应重新生成并在 trace 日志中记录 `invalid_inbound_trace_header=true`，不得把非法 header 原样写入响应。

## 4. citations 对外契约

`citations` 是 RAG 对外引用来源列表，属于 `/api/chat` 和 `/api/chat_stream` 的响应字段。阶段 1A 不一定实现该字段；阶段 3B 生效后必须按以下 schema 返回。阶段 1A 可返回空数组或不返回，但不得返回与此不兼容的结构。

```json
{
  "citation_id": "C1",
  "doc_id": "doc_cpu_high_usage",
  "chunk_id": "doc_cpu_high_usage#0003",
  "source_path": "aiops-docs/cpu_high_usage.md",
  "file_name": "cpu_high_usage.md",
  "score": 0.82,
  "content_preview": "CPU 使用率持续超过 80% 时..."
}
```

字段约束：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `citation_id` | string | 是 | 面向用户的引用编号，建议 `C1`、`C2` 递增 |
| `doc_id` | string | 阶段 3A 后必填 | 稳定文档 ID |
| `chunk_id` | string | 阶段 3A 后必填 | 稳定 chunk ID |
| `source_path` | string | 是 | 相对知识库根目录的安全路径，不返回任意绝对路径 |
| `file_name` | string | 是 | 文件名 |
| `score` | number | 否 | 归一化相关性分数，范围 `0.0-1.0` |
| `content_preview` | string | 否 | 截断预览，不超过 200 字符 |

## 5. fallback 对外表现

fallback 是内部失败恢复策略的外显结果，不暴露 `FallbackManager` 内部方法。

| 接口类型 | 对外表现 |
| --- | --- |
| 非流式 Chat | HTTP `200`，`data.fallback_used=true`，`data.answer` 为安全降级文案或基于已有证据的简化回答 |
| Chat SSE | 发送 `fallback` event，最终 `done` event 中 `fallback_used=true` |
| AIOps SSE | 发送 `fallback` event，若有部分证据，后续 `report`/`done` 返回部分诊断报告 |
| 文件/目录索引 | 不使用 LLM fallback；部分文件失败用 `partial_success` 和失败列表表达 |
| 健康检查 | 不触发 fallback，依赖不可用返回 `503` |

fallback 响应不得包含堆栈、密钥、内部 URL、原始异常全文。可返回稳定 `reason_code`，例如 `LLM_TIMEOUT`、`RAG_EMPTY_RESULT`、`TOOL_TIMEOUT`。

## 6. 统一错误码表

| code | HTTP status | 用户可见 message | retryable | 可能触发 fallback | 适用接口 |
| --- | --- | --- | --- | --- | --- |
| `INVALID_INPUT` | `400` | 请求参数不合法。 | false | 否 | 全部 |
| `INVALID_SESSION_ID` | `400` | 会话 ID 不合法。 | false | 否 | `/api/chat`、`/api/chat_stream`、`/api/aiops`、`/api/chat/clear`、`/api/chat/session/{session_id}` |
| `REQUEST_TOO_LARGE` | `413` | 请求内容过长。 | false | 否 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `FILE_TOO_LARGE` | `413` | 文件大小超过限制。 | false | 否 | `/api/upload`、`/api/file/upload` |
| `UNSUPPORTED_FILE_TYPE` | `400` | 不支持的文件类型。 | false | 否 | `/api/upload`、`/api/file/upload`、目录索引接口 |
| `INVALID_FILE_MIME` | `415` | 文件内容类型与扩展名不匹配。 | false | 否 | `/api/upload`、`/api/file/upload` |
| `INVALID_FILE_ENCODING` | `400` | 文件必须是 UTF-8 编码文本。 | false | 否 | `/api/upload`、`/api/file/upload`、目录索引接口 |
| `INVALID_DIRECTORY` | `400` | 目录不存在或不允许索引。 | false | 否 | `/api/index_directory`、`/api/file/index_directory` |
| `PATH_TRAVERSAL_BLOCKED` | `400` | 路径不允许包含目录逃逸。 | false | 否 | 文件和目录接口 |
| `SYMLINK_NOT_ALLOWED` | `400` | 不允许索引符号链接。 | false | 否 | 文件和目录接口 |
| `TOOL_TIMEOUT` | `504` | 工具调用超时，请稍后重试。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `TOOL_EXECUTION_ERROR` | `502` | 工具调用失败。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `LLM_TIMEOUT` | `504` | 模型响应超时，请稍后重试。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `LLM_EMPTY_RESPONSE` | `502` | 模型未返回有效内容。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `LLM_PROVIDER_ERROR` | `502` | 模型服务暂时不可用。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `RAG_EMPTY_RESULT` | `404`，默认可降级为 `200` fallback | 未找到足够相关的知识库内容。 | false | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops` |
| `RAG_METADATA_INVALID` | `500` | 知识库文档元数据不完整。 | false | 是 | RAG 索引、检索和 citation 元数据边界 |
| `VECTOR_STORE_UNAVAILABLE` | `503` | 知识库暂时不可用。 | true | 是 | `/api/chat`、`/api/chat_stream`、`/api/aiops`、文件和目录接口、`/api/health` |
| `EMBEDDING_PROVIDER_ERROR` | `502` | 向量化服务暂时不可用。 | true | 否 | 文件和目录接口 |
| `AGENT_MAX_STEP_EXCEEDED` | `409`，默认可降级为 `200` fallback | 任务步骤过多，已停止继续执行。 | false | 是 | `/api/aiops` |
| `SSE_STREAM_INTERRUPTED` | `200` 流内 `error` event | 流式响应中断。 | true | 是 | `/api/chat_stream`、`/api/aiops` |
| `INTERNAL_ERROR` | `500` | 服务内部错误。 | true | 视场景而定 | 全部 |

## 7. HTTP API 契约

### 7.1 `POST /api/chat`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 非流式 RAG Chat，一次性返回完整回答 |
| 当前代码入口 | `app/api/chat.py::chat` |
| 当前请求模型 | `app/models/request.py::ChatRequest`，字段 `id` alias 为 `Id`，`question` alias 为 `Question`，`populate_by_name=True` |
| 改造后的请求模型 | `ChatRequestV1`：推荐 `id/question`，兼容 `Id/Question` |
| 向后兼容策略 | 保留当前前端字段 `Id/Question`；兼容字段为 `id/question`；推荐新调用方使用 `id/question`；若大小写字段同时传入，为保持当前 Pydantic 行为，`Id/Question` 优先 |
| HTTP status | 成功 `200`；输入错误 `400/413`；下游错误 `502/503/504`；未分类错误 `500` |
| 是否返回 `trace_id` | 是，阶段 1A 成功和失败都返回 |
| 是否返回 `request_id` | 是，阶段 1A 成功和失败都返回 |
| 是否可能触发 `fallback_used` | 是，LLM/RAG/Tool 失败或空结果可触发 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `Id` | string | 当前前端必填 | 无 | 当前旧字段；会话 ID；建议长度 `1-128`；仅允许字母、数字、`_`、`-`、`.`、`:` |
| `Question` | string | 当前前端必填 | 无 | 当前旧字段；问题文本；trim 后不得为空 |
| `id` | string | 兼容必填 | 无 | 推荐字段；与 `Id` 同义 |
| `question` | string | 兼容必填 | 无 | 推荐字段；与 `Question` 同义 |

Chat 兼容性规则：

| 场景 | 阶段 1A 契约 |
| --- | --- |
| 当前前端字段 | `Id`、`Question` |
| 兼容字段 | `id`、`question` |
| 推荐字段 | `id`、`question` |
| 大小写字段同时传入 | `Id` 优先于 `id`，`Question` 优先于 `question`，保持当前 Pydantic alias 行为 |
| 非法 `session_id` | 返回 `400 INVALID_SESSION_ID`，不进入 Agent |
| 空 `Question` / `question` | trim 后为空返回 `400 INVALID_INPUT` |
| 超长 `Question` / `question` | 返回 `413 REQUEST_TOO_LARGE`；阶段 1A 建议默认最大 `8000` 字符，可配置 |

成功响应格式：

```json
{
  "success": true,
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "answer": "CPU 使用率持续升高时，建议先检查进程级 CPU 占用...",
    "errorMessage": null,
    "session_id": "session-123",
    "fallback_used": false,
    "citations": []
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应格式：

```json
{
  "success": false,
  "code": 400,
  "message": "问题不能为空。",
  "data": {
    "success": false,
    "answer": null,
    "errorMessage": "问题不能为空。"
  },
  "error": {
    "code": "INVALID_INPUT",
    "message": "问题不能为空。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`INVALID_INPUT`、`INVALID_SESSION_ID`、`REQUEST_TOO_LARGE`、`TOOL_TIMEOUT`、`TOOL_EXECUTION_ERROR`、`LLM_TIMEOUT`、`LLM_EMPTY_RESPONSE`、`LLM_PROVIDER_ERROR`、`RAG_EMPTY_RESULT`、`VECTOR_STORE_UNAVAILABLE`、`INTERNAL_ERROR`。

请求示例：

```json
{
  "Id": "session-123",
  "Question": "CPU 使用率过高应该如何排查？"
}
```

推荐请求示例：

```json
{
  "id": "session-123",
  "question": "CPU 使用率过高应该如何排查？"
}
```

测试断言点：

1. `Id/Question` 和 `id/question` 都可用。
2. 同时传入大小写字段时使用 `Id/Question`。
3. 空问题返回 `400 INVALID_INPUT`。
4. 非法 session 返回 `400 INVALID_SESSION_ID`。
5. 超长问题返回 `413 REQUEST_TOO_LARGE`。
6. 成功响应保留 `code/message/data.success/answer/errorMessage`，并新增 `trace_id/request_id`。
7. fallback 场景返回 `data.fallback_used=true` 且不泄漏内部异常。

### 7.2 `POST /api/chat_stream`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 流式 RAG Chat，通过 SSE 逐步返回检索、工具调用、token、fallback、完成或错误 |
| 当前代码入口 | `app/api/chat.py::chat_stream` |
| 当前请求模型 | `ChatRequest(Id, Question)` |
| 改造后的请求模型 | 与 `/api/chat` 相同，推荐 `id/question`，兼容 `Id/Question` |
| 向后兼容策略 | 当前代码使用 `event: message`，`data.type=content/done/error/tool_call/search_results/debug`；阶段 1A 规范事件名为 `start/retrieval/tool_call/token/fallback/error/done`，但 data 内必须保留 `type` 字段以兼容旧解析逻辑 |
| HTTP status | SSE 建立成功 `200`；建立前输入错误 `400/413`；建立后错误通过 `error` event |
| 是否返回 `trace_id` | 是，`start/error/done` 必须返回 |
| 是否返回 `request_id` | 是，`start/error/done` 必须返回 |
| 是否可能触发 `fallback_used` | 是 |

请求字段同 `/api/chat`。

成功响应格式：`text/event-stream`。每条消息建议：

```text
event: token
data: {"type":"token","trace_id":"trc_01JABCDEF1234567890","delta":"CPU","index":1}
```

失败响应格式：

建立 SSE 前：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "问题不能为空。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  }
}
```

建立 SSE 后：

```text
event: error
data: {"type":"error","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","error":{"code":"LLM_TIMEOUT","message":"模型响应超时，请稍后重试。","retryable":true}}
```

可能错误码：`INVALID_INPUT`、`INVALID_SESSION_ID`、`REQUEST_TOO_LARGE`、`TOOL_TIMEOUT`、`TOOL_EXECUTION_ERROR`、`LLM_TIMEOUT`、`LLM_EMPTY_RESPONSE`、`LLM_PROVIDER_ERROR`、`RAG_EMPTY_RESULT`、`VECTOR_STORE_UNAVAILABLE`、`SSE_STREAM_INTERRUPTED`、`INTERNAL_ERROR`。

请求示例：

```json
{
  "Id": "session-123",
  "Question": "根据知识库说明 CPU 告警的排查步骤。"
}
```

成功响应示例：

```text
event: start
data: {"type":"start","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","session_id":"session-123","model":"qwen-max"}

event: token
data: {"type":"token","trace_id":"trc_01JABCDEF1234567890","delta":"可以先检查","index":1}

event: done
data: {"type":"done","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","session_id":"session-123","answer":"可以先检查进程级 CPU...","citations":[],"fallback_used":false}
```

失败响应示例：

```text
event: error
data: {"type":"error","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","error":{"code":"SSE_STREAM_INTERRUPTED","message":"流式响应中断。","retryable":true}}
```

测试断言点：

1. 第一条规范事件为 `start`，且包含 `trace_id/request_id/session_id`。
2. `token` event 可多次出现，按顺序拼接为最终回答。
3. `retrieval` event 可返回 `citations`，阶段 1A 可为空数组。
4. `tool_call` event 不暴露敏感参数全文，只允许 preview。
5. fallback 场景必须先发 `fallback` event，再在 `done` 中返回 `fallback_used=true`。
6. 异常必须发 `error` event，包含统一错误结构和 `trace_id/request_id`。
7. `done` event 必须包含 `trace_id/request_id`。
8. 兼容模式下 `data.type` 必须存在。

#### `/api/chat_stream` SSE event schema

`start`：

```json
{
  "type": "start",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "session_id": "session-123",
  "model": "qwen-max",
  "created_at": "2026-07-07T09:30:00+08:00"
}
```

`retrieval`：

```json
{
  "type": "retrieval",
  "trace_id": "trc_01JABCDEF1234567890",
  "status": "completed",
  "query": "CPU 使用率过高",
  "top_k": 3,
  "citations": [
    {
      "citation_id": "C1",
      "doc_id": "doc_cpu_high_usage",
      "chunk_id": "doc_cpu_high_usage#0003",
      "source_path": "aiops-docs/cpu_high_usage.md",
      "file_name": "cpu_high_usage.md",
      "score": 0.82,
      "content_preview": "CPU 使用率持续超过 80% 时..."
    }
  ]
}
```

`tool_call`：

```json
{
  "type": "tool_call",
  "trace_id": "trc_01JABCDEF1234567890",
  "tool_name": "retrieve_knowledge",
  "status": "start",
  "input_preview": {
    "query": "CPU 使用率过高"
  }
}
```

`token`：

```json
{
  "type": "token",
  "trace_id": "trc_01JABCDEF1234567890",
  "delta": "建议先查看进程级 CPU 占用",
  "index": 1
}
```

`fallback`：

```json
{
  "type": "fallback",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "fallback_used": true,
  "reason_code": "RAG_EMPTY_RESULT",
  "message": "未找到足够相关的知识库内容，已返回通用排查建议。"
}
```

`error`：

```json
{
  "type": "error",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "error": {
    "code": "LLM_TIMEOUT",
    "message": "模型响应超时，请稍后重试。",
    "retryable": true
  }
}
```

`done`：

```json
{
  "type": "done",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "session_id": "session-123",
  "answer": "建议先查看进程级 CPU 占用，再检查近期发布和日志错误。",
  "citations": [],
  "fallback_used": false,
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 260
  }
}
```

### 7.3 `POST /api/aiops`

| 项 | 内容 |
| --- | --- |
| 接口用途 | AIOps 智能运维诊断，流式返回计划、步骤、工具调用、报告、fallback、完成或错误 |
| 当前代码入口 | `app/api/aiops.py::diagnose_stream` |
| 当前请求模型 | `app/models/aiops.py::AIOpsRequest`，仅 `session_id`，默认 `"default"` |
| 改造后的请求模型 | `AIOpsRequestV1`：保留 `session_id`，可扩展 `input/alert_id/options`，但阶段 1A 不强制新增 |
| 请求字段 | 见下表 |
| 向后兼容策略 | 当前代码返回 `event: message`，`data.type=status/plan/step_complete/report/complete/error`；阶段 1A 规范事件名为 `start/plan/step_complete/tool_call/fallback/report/error/done`，同时保留 `data.type`；旧 `complete` 应兼容映射为新 `done` |
| HTTP status | SSE 建立成功 `200`；建立前输入错误 `400`；建立后错误通过 `error` event |
| 是否返回 `trace_id` | 是，`start/error/done` 必须返回 |
| 是否返回 `request_id` | 是，`start/error/done` 必须返回 |
| 是否可能触发 `fallback_used` | 是 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `session_id` | string | 否 | `"default"` | 建议长度 `1-128`；仅允许字母、数字、`_`、`-`、`.`、`:` |

成功响应格式：`text/event-stream`。

失败响应格式：同 `/api/chat_stream`，SSE 建立后通过 `error` event 返回。

可能错误码：`INVALID_INPUT`、`INVALID_SESSION_ID`、`TOOL_TIMEOUT`、`TOOL_EXECUTION_ERROR`、`LLM_TIMEOUT`、`LLM_EMPTY_RESPONSE`、`LLM_PROVIDER_ERROR`、`RAG_EMPTY_RESULT`、`VECTOR_STORE_UNAVAILABLE`、`AGENT_MAX_STEP_EXCEEDED`、`SSE_STREAM_INTERRUPTED`、`INTERNAL_ERROR`。

请求示例：

```json
{
  "session_id": "session-123"
}
```

成功响应示例：

```text
event: start
data: {"type":"start","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","session_id":"session-123","mode":"aiops"}

event: plan
data: {"type":"plan","trace_id":"trc_01JABCDEF1234567890","message":"执行计划已制定，共 3 个步骤","plan":["查询当前告警","检查服务日志","生成诊断报告"]}

event: done
data: {"type":"done","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","session_id":"session-123","status":"completed","fallback_used":false}
```

失败响应示例：

```text
event: error
data: {"type":"error","trace_id":"trc_01JABCDEF1234567890","request_id":"req_01JABCDEF1234567890","error":{"code":"AGENT_MAX_STEP_EXCEEDED","message":"任务步骤过多，已停止继续执行。","retryable":false}}
```

测试断言点：

1. 不传 `session_id` 时使用 `"default"`。
2. 非法 `session_id` 返回 `INVALID_SESSION_ID`。
3. 第一条规范事件为 `start`。
4. 至少能识别 `plan/step_complete/report/done` 事件。
5. 工具调用通过 `tool_call` event 外显，敏感入参只返回 preview。
6. 超过步骤上限通过 `fallback` 或 `error` event 暴露，不无限流式输出。
7. `error/done` event 均包含 `trace_id/request_id`。
8. 旧 `complete` type 在阶段 1A 兼容为 `done`。

#### `/api/aiops` SSE event schema

`start`：

```json
{
  "type": "start",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "session_id": "session-123",
  "mode": "aiops",
  "created_at": "2026-07-07T09:30:00+08:00"
}
```

`plan`：

```json
{
  "type": "plan",
  "trace_id": "trc_01JABCDEF1234567890",
  "stage": "plan_created",
  "message": "执行计划已制定，共 3 个步骤",
  "plan": [
    "查询当前系统告警",
    "检查相关服务日志",
    "生成诊断报告"
  ]
}
```

`step_complete`：

```json
{
  "type": "step_complete",
  "trace_id": "trc_01JABCDEF1234567890",
  "stage": "step_executed",
  "message": "步骤执行完成 (1/3)",
  "step_index": 1,
  "total_steps": 3,
  "current_step": "查询当前系统告警",
  "result_preview": "发现 HighCPUUsage 告警仍处于活跃状态...",
  "remaining_steps": 2
}
```

`tool_call`：

```json
{
  "type": "tool_call",
  "trace_id": "trc_01JABCDEF1234567890",
  "tool_name": "get_current_time",
  "status": "success",
  "duration_ms": 32,
  "input_preview": {}
}
```

`fallback`：

```json
{
  "type": "fallback",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "fallback_used": true,
  "reason_code": "TOOL_TIMEOUT",
  "message": "部分工具调用超时，已基于已完成步骤生成诊断摘要。",
  "partial_report_available": true
}
```

`report`：

```json
{
  "type": "report",
  "trace_id": "trc_01JABCDEF1234567890",
  "stage": "final_report",
  "message": "最终报告已生成",
  "report": "# 告警分析报告\n\n## 活跃告警清单\n...",
  "evidence": {
    "alerts": 1,
    "tool_calls": 3
  },
  "citations": []
}
```

`error`：

```json
{
  "type": "error",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "stage": "error",
  "error": {
    "code": "TOOL_EXECUTION_ERROR",
    "message": "工具调用失败。",
    "retryable": true
  }
}
```

`done`：

```json
{
  "type": "done",
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890",
  "session_id": "session-123",
  "status": "completed",
  "fallback_used": false,
  "diagnosis": {
    "status": "completed",
    "report": "# 告警分析报告\n..."
  }
}
```

### 7.4 `POST /api/upload`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 上传单个文本/Markdown 文件，并自动创建向量索引 |
| 当前代码入口 | `app/api/file.py::upload_file` |
| 当前请求模型 | `multipart/form-data`，字段名 `file`，类型 `UploadFile` |
| 改造后的请求模型 | 保持 `multipart/form-data` 字段名 `file` 不变；新增 MIME、UTF-8、文件名安全和 trace |
| 向后兼容策略 | `/api/upload` 是当前真实接口，必须保留；阶段 1A 新增 `/api/file/upload` 时必须返回同结构；保留 `code/message/data.filename/file_path/size` |
| HTTP status | 成功 `200`；校验失败 `400/413/415`；索引服务失败按部分成功或 `502/503`；未分类错误 `500` |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `file` | file | 是 | 无 | multipart 字段名固定为 `file`；文件名不能为空；阶段 1A 最大 10MiB；内容必须为 UTF-8 文本 |

文件上传契约：

| 项 | 当前真实行为 | 阶段 1A 契约 |
| --- | --- | --- |
| 当前路径 | `/api/upload` | 必须保留 |
| 新增兼容别名 | 无 | `/api/file/upload` 可新增 |
| multipart 字段名 | `file` | `file`，不得改名 |
| 支持扩展名 | `.txt`、`.md` | 默认 `.txt`、`.md`、`.markdown`；如不实现 `.markdown`，必须继续返回 `UNSUPPORTED_FILE_TYPE` |
| 是否支持 `.markdown` | 当前后端不支持；前端 accept 允许 | 阶段 1A 建议支持并按 Markdown 处理 |
| 最大文件大小 | 后端 10MiB；前端当前 50MiB，存在不一致 | 以服务端配置为准，默认 10MiB；超限返回 `FILE_TOO_LARGE` |
| MIME sniff | 当前无 | 必须检查内容与扩展名；文本可接受 `text/plain`、`text/markdown`，`application/octet-stream` 仅在 UTF-8 文本校验通过时接受 |
| UTF-8 编码校验 | 当前索引时 `read_text(encoding="utf-8")`，失败较晚 | 上传保存前校验；失败返回 `INVALID_FILE_ENCODING` |
| 文件名安全 | 当前替换空格和 `\ / : * ? " < > |` | 必须取 basename、拒绝空名/控制字符/路径分隔符、规范化扩展名 |
| 文件名碰撞策略 | 当前同名文件覆盖并重建索引 | 阶段 1A 默认保留覆盖更新语义；响应建议返回 `replaced_existing` |
| 索引成功响应 | 当前 `200 code=200 message=success` | 同时返回统一 envelope 和 legacy 字段 |
| 索引失败响应 | 当前上传仍成功，仅日志记录索引失败 | 阶段 1A 建议 `200 partial_success`，`data.indexing.success=false`；文件保存失败才返回失败 envelope |
| trace_id 返回规则 | 当前无 | 成功、部分成功、失败都返回 |

成功响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "success",
  "data": {
    "filename": "cpu_high_usage.md",
    "file_path": "uploads/cpu_high_usage.md",
    "size": 2048,
    "replaced_existing": false,
    "indexing": {
      "success": true,
      "chunk_count": 6
    }
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

索引失败但上传成功响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "partial_success",
  "data": {
    "filename": "cpu_high_usage.md",
    "file_path": "uploads/cpu_high_usage.md",
    "size": 2048,
    "indexing": {
      "success": false,
      "error_code": "VECTOR_STORE_UNAVAILABLE",
      "error_message": "知识库暂时不可用。"
    }
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应示例：

```json
{
  "success": false,
  "error": {
    "code": "UNSUPPORTED_FILE_TYPE",
    "message": "不支持的文件类型。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`INVALID_INPUT`、`FILE_TOO_LARGE`、`UNSUPPORTED_FILE_TYPE`、`INVALID_FILE_MIME`、`INVALID_FILE_ENCODING`、`PATH_TRAVERSAL_BLOCKED`、`RAG_METADATA_INVALID`、`VECTOR_STORE_UNAVAILABLE`、`EMBEDDING_PROVIDER_ERROR`、`INTERNAL_ERROR`。

请求示例：

```bash
curl -X POST "http://localhost:9900/api/upload" \
  -F "file=@cpu_high_usage.md"
```

测试断言点：

1. multipart 字段名必须是 `file`。
2. `.txt/.md` 当前可上传；`.markdown` 当前真实后端会失败，阶段 1A 若启用必须成功。
3. 超过 10MiB 返回 `413 FILE_TOO_LARGE`。
4. MIME 与扩展名不匹配返回 `415 INVALID_FILE_MIME`。
5. 非 UTF-8 返回 `400 INVALID_FILE_ENCODING`。
6. 文件名包含路径分隔符或目录逃逸时不得写出 `uploads`。
7. 同名文件碰撞策略可断言为覆盖更新或返回 `replaced_existing=true`。
8. 上传成功、索引部分失败、上传失败都包含 `trace_id/request_id`。

### 7.5 `POST /api/file/upload`

| 项 | 内容 |
| --- | --- |
| 接口用途 | `/api/upload` 的阶段 1A 规范化别名 |
| 当前代码入口 | 当前不存在；阶段 1A 应复用 `app/api/file.py::upload_file` 或同等 handler |
| 当前请求模型 | 当前不存在 |
| 改造后的请求模型 | 与 `/api/upload` 完全一致：`multipart/form-data`，字段名 `file` |
| 请求字段 | 与 `/api/upload` 完全一致 |
| 字段类型/必填/默认/约束 | 与 `/api/upload` 完全一致 |
| 向后兼容策略 | 新旧路径返回相同结构；不得删除 `/api/upload` |
| 成功响应格式 | 与 `/api/upload` 完全一致 |
| 失败响应格式 | 与 `/api/upload` 完全一致 |
| HTTP status | 与 `/api/upload` 完全一致 |
| 可能错误码 | 与 `/api/upload` 完全一致 |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求示例：

```bash
curl -X POST "http://localhost:9900/api/file/upload" \
  -F "file=@cpu_high_usage.md"
```

成功响应示例、失败响应示例和测试断言点同 `/api/upload`，额外断言：同一个合法文件分别请求 `/api/upload` 和 `/api/file/upload`，除路径自身外响应 schema 必须一致。

### 7.6 `POST /api/index_directory`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 索引指定目录下的所有支持文件 |
| 当前代码入口 | `app/api/file.py::index_directory` |
| 当前请求模型 | 无 Pydantic body；`directory_path: str = None` 在 FastAPI 中是 query 参数；不传则默认 `./uploads` |
| 改造后的请求模型 | `IndexDirectoryRequest` JSON body：`directory_path`；同时兼容旧 query 参数 |
| 向后兼容策略 | `/api/index_directory` 是当前真实接口，必须保留；阶段 1A 新增 `/api/file/index_directory` 时必须返回同结构；阶段 1A 仍接受旧 query 参数 |
| HTTP status | 成功或部分成功 `200`；非法目录/路径 `400`；向量库/embedding 失败 `502/503`；未分类错误 `500` |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `directory_path` | string | 否 | `./uploads` | 阶段 1A 必须位于 allowlist 内；不得 path traversal；不得为 symlink；目录必须存在 |

目录索引安全协议：

| 项 | 当前真实行为 | 阶段 1A 契约 |
| --- | --- | --- |
| 当前路径 | `/api/index_directory` | 必须保留 |
| 新增兼容别名 | 无 | `/api/file/index_directory` 可新增 |
| `directory_path` 字段 | 当前 query 参数 | JSON body 字段，兼容 query 参数 |
| allowlist 目录 | 当前无，任意路径可 resolve | 必须配置 allowlist，建议默认仅允许项目内 `uploads/`、`aiops-docs/` |
| symlink 拒绝 | 当前无 | 请求目录或待索引文件为 symlink 时返回 `SYMLINK_NOT_ALLOWED` 或跳过并记录失败；请求根目录 symlink 必须拒绝 |
| path traversal 拒绝 | 当前无 | `../`、绝对路径逃逸、resolve 后不在 allowlist 内均返回 `PATH_TRAVERSAL_BLOCKED` 或 `INVALID_DIRECTORY` |
| 非法路径错误码 | 当前通常进入 `500` 或 result.error_message | 阶段 1A 使用 `INVALID_DIRECTORY`、`PATH_TRAVERSAL_BLOCKED`、`SYMLINK_NOT_ALLOWED` |
| 索引结果响应 | 当前 `code=200`，`message=success/partial_success`，`data=IndexingResult.to_dict()` | 保留旧字段，新增统一 envelope、trace |
| 部分文件失败 | 当前 `data.success=false`，`failed_files` 为 `{path: error}` | 保留 `failed_files` map，建议追加 `failed_file_count` 或 `files[]` 不破坏旧结构 |
| trace_id 返回规则 | 当前无 | 成功、部分成功、失败都返回 |

成功响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "directory_path": "E:/Code/AgentCode/super_biz_agent_py/uploads",
    "total_files": 2,
    "success_count": 2,
    "fail_count": 0,
    "duration_ms": 1530,
    "error_message": "",
    "failed_files": {}
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

部分文件失败响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "partial_success",
  "data": {
    "success": false,
    "directory_path": "E:/Code/AgentCode/super_biz_agent_py/uploads",
    "total_files": 3,
    "success_count": 2,
    "fail_count": 1,
    "duration_ms": 1800,
    "error_message": "",
    "failed_files": {
      "E:/Code/AgentCode/super_biz_agent_py/uploads/bad.md": "文件必须是 UTF-8 编码文本。"
    }
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应示例：

```json
{
  "success": false,
  "error": {
    "code": "PATH_TRAVERSAL_BLOCKED",
    "message": "路径不允许包含目录逃逸。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`INVALID_INPUT`、`UNSUPPORTED_FILE_TYPE`、`INVALID_FILE_ENCODING`、`INVALID_DIRECTORY`、`PATH_TRAVERSAL_BLOCKED`、`SYMLINK_NOT_ALLOWED`、`RAG_METADATA_INVALID`、`VECTOR_STORE_UNAVAILABLE`、`EMBEDDING_PROVIDER_ERROR`、`INTERNAL_ERROR`。

请求示例：

```json
{
  "directory_path": "uploads"
}
```

旧 query 兼容请求示例：

```bash
curl -X POST "http://localhost:9900/api/index_directory?directory_path=uploads"
```

测试断言点：

1. 不传 `directory_path` 时索引默认 `uploads`。
2. JSON body 和旧 query 参数都可用；若同时传入，阶段 1A 建议 JSON body 优先。
3. allowlist 外路径返回 `INVALID_DIRECTORY` 或 `PATH_TRAVERSAL_BLOCKED`。
4. `../` 路径逃逸返回 `PATH_TRAVERSAL_BLOCKED`。
5. symlink 根目录返回 `SYMLINK_NOT_ALLOWED`。
6. 部分文件失败仍返回 `200 partial_success`，并保留 `failed_files`。
7. 成功、部分成功、失败都包含 `trace_id/request_id`。

### 7.7 `POST /api/file/index_directory`

| 项 | 内容 |
| --- | --- |
| 接口用途 | `/api/index_directory` 的阶段 1A 规范化别名 |
| 当前代码入口 | 当前不存在；阶段 1A 应复用 `app/api/file.py::index_directory` 或同等 handler |
| 当前请求模型 | 当前不存在 |
| 改造后的请求模型 | 与 `/api/index_directory` 完全一致 |
| 请求字段 | 与 `/api/index_directory` 完全一致 |
| 字段类型/必填/默认/约束 | 与 `/api/index_directory` 完全一致 |
| 向后兼容策略 | 新旧路径返回相同结构；不得删除 `/api/index_directory` |
| 成功响应格式 | 与 `/api/index_directory` 完全一致 |
| 失败响应格式 | 与 `/api/index_directory` 完全一致 |
| HTTP status | 与 `/api/index_directory` 完全一致 |
| 可能错误码 | 与 `/api/index_directory` 完全一致 |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求示例：

```json
{
  "directory_path": "uploads"
}
```

成功响应示例、失败响应示例和测试断言点同 `/api/index_directory`，额外断言：同一个合法目录分别请求 `/api/index_directory` 和 `/api/file/index_directory`，除路径自身外响应 schema 必须一致。

### 7.8 `GET /api/health`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 健康检查，返回服务和 Milvus 状态 |
| 当前代码入口 | `app/api/health.py::health_check` |
| 当前请求模型 | 无请求体 |
| 改造后的请求模型 | 不变 |
| 请求字段 | 无 |
| 字段类型/必填/默认/约束 | 无 |
| 向后兼容策略 | 保留 `code/message/data.service/version/status/milvus` |
| HTTP status | 健康 `200`；Milvus 不可用 `503` |
| 是否返回 `trace_id` | 是，阶段 1A 建议返回 |
| 是否返回 `request_id` | 是，阶段 1A 建议返回 |
| 是否可能触发 `fallback_used` | 否 |

成功响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "服务运行正常",
  "data": {
    "service": "AegisOps Agent",
    "version": "1.0.0",
    "status": "healthy",
    "milvus": {
      "status": "connected",
      "message": "Milvus 连接正常"
    }
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应示例：

```json
{
  "success": false,
  "code": 503,
  "message": "服务不可用",
  "data": {
    "service": "AegisOps Agent",
    "version": "1.0.0",
    "status": "unhealthy",
    "milvus": {
      "status": "disconnected",
      "message": "Milvus 连接异常"
    },
    "error": "数据库不可用"
  },
  "error": {
    "code": "VECTOR_STORE_UNAVAILABLE",
    "message": "知识库暂时不可用。",
    "retryable": true,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`VECTOR_STORE_UNAVAILABLE`、`INTERNAL_ERROR`。

请求示例：

```bash
curl "http://localhost:9900/api/health"
```

测试断言点：

1. Milvus 连接正常时返回 `200`。
2. Milvus 不可用时返回 `503`。
3. 保留 `code/message/data`。
4. 阶段 1A 成功和失败都返回 `trace_id/request_id`。

### 7.9 `POST /api/chat/clear`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 清空指定会话历史 |
| 当前代码入口 | `app/api/chat.py::clear_session` |
| 当前请求模型 | `app/models/request.py::ClearRequest`，字段 `session_id` alias 为 `sessionId`，`populate_by_name=True` |
| 改造后的请求模型 | 保留 `session_id` 和 `sessionId`；推荐 `session_id` |
| 向后兼容策略 | 保留当前响应 `status/message/data`；阶段 1A 追加统一 envelope 和 trace |
| HTTP status | 成功 `200`；非法 session `400`；服务端异常 `500` |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `session_id` | string | 是 | 无 | 推荐字段；会话 ID；建议长度 `1-128` |
| `sessionId` | string | 是 | 无 | 兼容 alias；若两者同时传入，保留当前 alias 优先行为 |

成功响应示例：

```json
{
  "success": true,
  "status": "success",
  "message": "会话已清空",
  "data": null,
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应示例：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_SESSION_ID",
    "message": "会话 ID 不合法。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`INVALID_INPUT`、`INVALID_SESSION_ID`、`INTERNAL_ERROR`。

请求示例：

```json
{
  "session_id": "session-123"
}
```

测试断言点：

1. `session_id` 和 `sessionId` 都可用。
2. 非法 session 返回 `INVALID_SESSION_ID`。
3. 成功响应保留 `status/message/data`。
4. 成功和失败均包含 `trace_id/request_id`。

### 7.10 `GET /api/chat/session/{session_id}`

| 项 | 内容 |
| --- | --- |
| 接口用途 | 查询指定会话历史 |
| 当前代码入口 | `app/api/chat.py::get_session_info` |
| 当前请求模型 | path 参数 `session_id` |
| 改造后的请求模型 | 不变，增加 session 格式校验 |
| 向后兼容策略 | 保留当前 `session_id/message_count/history` 顶层字段；阶段 1A 可追加统一 envelope，但不得破坏旧字段 |
| HTTP status | 成功 `200`；非法 session `400`；服务端异常 `500` |
| 是否返回 `trace_id` | 是 |
| 是否返回 `request_id` | 是 |
| 是否可能触发 `fallback_used` | 否 |

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 字段约束 |
| --- | --- | --- | --- | --- |
| `session_id` | string | 是 | 无 | path 参数；建议长度 `1-128`；仅允许字母、数字、`_`、`-`、`.`、`:` |

成功响应示例：

```json
{
  "success": true,
  "session_id": "session-123",
  "message_count": 2,
  "history": [
    {
      "role": "user",
      "content": "CPU 使用率过高怎么办？",
      "timestamp": "2026-07-07T09:30:00+08:00"
    },
    {
      "role": "assistant",
      "content": "建议先查看进程级 CPU 占用。",
      "timestamp": "2026-07-07T09:30:01+08:00"
    }
  ],
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

失败响应示例：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_SESSION_ID",
    "message": "会话 ID 不合法。",
    "retryable": false,
    "trace_id": "trc_01JABCDEF1234567890",
    "request_id": "req_01JABCDEF1234567890"
  },
  "trace_id": "trc_01JABCDEF1234567890",
  "request_id": "req_01JABCDEF1234567890"
}
```

可能错误码：`INVALID_SESSION_ID`、`INTERNAL_ERROR`。

请求示例：

```bash
curl "http://localhost:9900/api/chat/session/session-123"
```

测试断言点：

1. 存在历史时返回 `message_count == len(history)`。
2. 不存在历史时返回 `200`、`message_count=0`、`history=[]`。
3. 非法 session 返回 `INVALID_SESSION_ID`。
4. 成功响应保留当前顶层字段，并追加 `trace_id/request_id`。

## 8. 文件上传和目录索引安全协议

阶段 1A 的文件与目录接口必须满足以下安全协议：

| 类别 | 协议 |
| --- | --- |
| 根目录限制 | 上传只能写入配置的 upload 根目录；目录索引只能访问 allowlist 根目录 |
| 路径解析 | 所有路径必须 `resolve()` 后再校验是否仍位于 allowlist 内 |
| path traversal | `../`、绝对路径逃逸、URL 编码逃逸、Windows drive 逃逸均拒绝 |
| symlink | 请求目录 symlink 必须拒绝；待索引文件 symlink 必须拒绝或记录为单文件失败 |
| 文件名 | 取 basename；拒绝空名、控制字符、路径分隔符、保留设备名；规范化 Unicode 和扩展名 |
| 扩展名 | 上传默认 `.txt/.md/.markdown`；目录索引默认 `.txt/.md/.markdown`，如果阶段 1A 未实现 `.markdown`，必须清楚返回 `UNSUPPORTED_FILE_TYPE` |
| MIME sniff | 扩展名、声明 MIME、内容 sniff 三者不一致时返回 `INVALID_FILE_MIME` |
| 编码 | 文本必须 UTF-8；失败返回 `INVALID_FILE_ENCODING` |
| 大小 | 上传默认最大 10MiB；目录索引单文件也应应用同一限制 |
| 部分失败 | 批量索引不得因单个文件失败丢失其他成功结果，必须返回失败文件列表 |
| trace | 成功、失败、部分成功都必须关联 `trace_id/request_id` |

## 9. 向后兼容策略

1. 旧路径 `/api/upload`、`/api/index_directory` 必须保留，不能直接删除。
2. 新路径 `/api/file/upload`、`/api/file/index_directory` 是阶段 1A 规范化别名，不是替代删除。
3. 新旧上传/索引路径在阶段 1A 应返回相同响应结构。
4. Chat 保留 `Id/Question`，同时兼容 `id/question`；推荐新调用方使用 `id/question`。
5. Chat 字段同时存在时，阶段 1A 保留当前 alias 优先行为：`Id/Question` 优先。
6. `ClearRequest` 保留 `sessionId`，同时兼容 `session_id`；推荐 `session_id`。
7. 旧响应字段 `code/message/data`、`data.success/answer/errorMessage`、`status/message/data` 在阶段 1A 不得删除。
8. 新统一 envelope 字段只能追加，避免破坏旧前端判断逻辑。
9. SSE 当前 `event: message` + `data.type` 的解析方式必须兼容；规范 event 名可以新增，但 data 内 `type` 不得删除。
10. 当前 AIOps `complete` type 应兼容映射为规范 `done`。
11. citations 阶段 1A 可为空或缺省；阶段 3B 生效后必须稳定返回 schema。
12. 错误码稳定，用户可见 message 可调整但不得改变 code 语义。

## 10. 接口覆盖清单

| 接口/契约 | 当前是否存在 | 是否新增 | 是否写入本文档 | 所属阶段 | 备注 |
| ----- | ------ | ---- | ------- | ---- | -- |
| `POST /api/chat` | 是 | 否 | 是 | 当前 + 1A 增强 | 覆盖 `Id/Question` 与 `id/question` 兼容 |
| `POST /api/chat_stream` | 是 | 否 | 是 | 当前 + 1A/1B 增强 | 覆盖规范 SSE event schema |
| `POST /api/aiops` | 是 | 否 | 是 | 当前 + 1A/1B 增强 | 覆盖 AIOps SSE event schema |
| `POST /api/upload` | 是 | 否 | 是 | 当前 + 1A 增强 | 当前真实接口，必须保留 |
| `POST /api/file/upload` | 否 | 是 | 是 | 1A | `/api/upload` 规范化别名 |
| `POST /api/index_directory` | 是 | 否 | 是 | 当前 + 1A 增强 | 当前真实接口，必须保留 |
| `POST /api/file/index_directory` | 否 | 是 | 是 | 1A | `/api/index_directory` 规范化别名 |
| `GET /api/health` | 是 | 否 | 是 | 当前 + 1A 增强 | Milvus 不可用返回 `503` |
| `POST /api/chat/clear` | 是 | 否 | 是 | 当前 + 1A 增强 | 清空会话 |
| `GET /api/chat/session/{session_id}` | 是 | 否 | 是 | 当前 + 1A 增强 | 查询会话历史 |
| 统一成功 envelope | 部分存在旧格式 | 是 | 是 | 1A | 新旧字段并存 |
| 统一失败 envelope | 否 | 是 | 是 | 1A | 保留旧错误兼容字段 |
| 错误码表 | 否 | 是 | 是 | 1A 起 | 覆盖 P0 错误码 |
| `trace_id/request_id` 规则 | 否 | 是 | 是 | 1A | header 透传或服务端生成 |
| Chat SSE event schema | 部分旧格式存在 | 是 | 是 | 1A/1B | `start/retrieval/tool_call/token/fallback/error/done` |
| AIOps SSE event schema | 部分旧格式存在 | 是 | 是 | 1A/1B | `start/plan/step_complete/tool_call/fallback/report/error/done` |
| citations schema | 否 | 是 | 是 | 3B | 阶段 1A 可为空或缺省 |
| fallback 外显契约 | 否 | 是 | 是 | 2 | 先在文档中定义外显字段 |
| 文件上传安全协议 | 部分存在 | 是 | 是 | 1A | MIME、UTF-8、大小、文件名 |
| 目录索引安全协议 | 否 | 是 | 是 | 1A | allowlist、symlink、path traversal |
| 向后兼容策略 | 部分存在 | 是 | 是 | 1A 起 | 明确不删除旧路径 |

## 11. 开发验收标准

1. 所有公开 HTTP 接口在文档中有定义。
2. 新旧上传/索引路径的兼容关系明确。
3. 所有流式接口 event schema 明确。
4. 所有错误响应都符合统一 envelope。
5. 所有响应都能关联 `trace_id`。
6. 所有响应都能关联 `request_id`。
7. 文件上传和目录索引边界可测试。
8. 文档中的字段名和代码模型一致。
9. 每个接口都有成功示例。
10. 每个接口都有失败示例。
11. 每个接口都有测试断言点。
12. `/api/upload` 和 `/api/index_directory` 不被删除。
13. `/api/file/upload` 和 `/api/file/index_directory` 若在阶段 1A 新增，必须与旧路径返回相同响应结构。
14. Chat 同时支持 `Id/Question` 与 `id/question`，并明确冲突优先级。
15. SSE `start/error/done` 必须包含 `trace_id/request_id`。
16. `citations` 字段若返回，必须符合本文档 schema。
17. fallback 场景必须通过 `fallback_used` 或 `fallback` event 对外可见。
18. 目录索引必须拒绝 path traversal 和 symlink。
19. 上传必须校验扩展名、大小、MIME、UTF-8 和安全文件名。
20. 阶段 1A 保留旧响应兼容字段，避免破坏当前前端。
