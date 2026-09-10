# AegisOps Agent

> 企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能诊断

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)

## ✨ 核心特性

- 🤖 **智能对话** - LangChain 多轮对话 + 流式输出
- 📚 **RAG 问答** - 向量检索增强，支持文档上传、自动建立向量索引、自动更新知识库
- 🔧 **AIOps 诊断** - Plan-Execute-Replan 自动故障诊断和根因分析
- 🌐 **Web 界面** - 现代化 UI，支持多种对话模式：快速问答/流式对话
- 🔌 **MCP 集成** - 日志查询和监控数据工具接入

## 🛠️ 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope (通义千问)
- **向量库**: Milvus
- **工具协议**: MCP (Model Context Protocol)

## 🚀 快速开始

### 环境要求
- Python 3.11+（`pyproject.toml` 当前约束为 `>=3.11,<3.14`）
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 安装依赖（推荐使用 uv）
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 方式 2: 使用 pip
pip install -e .

# 3. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env  # 或使用其他编辑器

# 4. 一键初始化（启动 Docker + 服务 + 上传文档）
make init

# 5. 一键启动
make start
```

#### Windows 环境（PowerShell/CMD）

如果Windows 不支持 `make` 命令，可以手动执行以下步骤以启动服务：

```powershell
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
# 创建虚拟环境
uv venv
# 激活虚拟环境
.venv\Scripts\activate
# 安装所有依赖
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 等待 Milvus 启动完成（约 5-10 秒）
timeout /t 10

# 7. 启动 MCP 服务
# 启动 CLS 日志查询服务（新开一个 PowerShell 窗口）
python mcp_servers/cls_server.py

# 启动 Monitor 监控服务（新开一个 PowerShell 窗口）
python mcp_servers/monitor_server.py

# 8. 启动 FastAPI 主服务（新开一个 PowerShell 窗口）
# 注意：日志会自动输出到 logs\app_YYYY-MM-DD.log
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 9. 上传文档到向量库（新开一个 PowerShell 窗口）
# 等待服务启动完成后执行
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）

使用启动脚本：

```powershell
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 旧路径，必须保留；上传并索引文档 |
| 文件上传别名 | POST | `/api/file/upload` | `/api/upload` 的兼容别名，响应结构保持一致 |
| 目录索引 | POST | `/api/index_directory` | 旧路径，必须保留；索引 allowlist 内目录 |
| 目录索引别名 | POST | `/api/file/index_directory` | `/api/index_directory` 的兼容别名，响应结构保持一致 |
| 健康检查 | GET | `/api/health` | 服务状态检查 |

兼容约束：旧 API 不删除，旧响应字段不删除。`/api/chat` 继续兼容 `Id/Question`，也支持推荐字段 `id/question`；SSE 继续保留 `event: message` 与 `data.type`，新增规范事件也不会移除旧解析字段。

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer

# 文件上传旧路径与兼容别名
curl -X POST "http://localhost:9900/api/upload" \
  -F "file=@aiops-docs/cpu_high_usage.md"
curl -X POST "http://localhost:9900/api/file/upload" \
  -F "file=@aiops-docs/cpu_high_usage.md"

# 目录索引旧路径与兼容别名
curl -X POST "http://localhost:9900/api/index_directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path":"aiops-docs"}'
curl -X POST "http://localhost:9900/api/file/index_directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path":"aiops-docs"}'
```

## 📁 项目结构

```
super_biz_agent_py/
├── app/                                    # 应用核心
│   ├── __init__.py                         # 包初始化（自动加载日志配置）
│   ├── main.py                             # FastAPI 应用入口
│   ├── config.py                           # 配置管理（环境变量、边界开关、评估配置、MCP 服务器配置）
│   ├── api/                                # API 路由层
│   │   ├── __init__.py
│   │   ├── chat.py                         # 对话接口（普通/SSE，保留旧响应字段）
│   │   ├── aiops.py                        # AIOps SSE 接口（诊断、fallback、done/complete 兼容）
│   │   ├── file.py                         # 文件上传与目录索引（旧路径和 /api/file/* 别名）
│   │   └── health.py                       # 健康检查（服务和 Milvus 状态）
│   ├── agent/                              # Agent 模块
│   │   ├── __init__.py
│   │   ├── mcp_client.py                   # MCP 客户端（连接和 retry interceptor）
│   │   ├── orchestrator.py                 # 薄编排层（guard/context/token/memory/fallback/trace 串联）
│   │   ├── policies.py                     # 工具权限策略
│   │   ├── tool_manager.py                 # 工具调用统一边界（timeout、权限、裁剪、trace）
│   │   └── aiops/                          # AIOps 核心逻辑
│   │       ├── __init__.py
│   │       ├── planner.py                  # 计划制定器
│   │       ├── executor.py                 # 步骤执行器
│   │       ├── replanner.py                # 重规划器
│   │       ├── state.py                    # 状态定义
│   │       └── utils.py                    # 工具函数
│   ├── core/                               # 核心边界组件
│   │   ├── __init__.py
│   │   ├── errors.py                       # AppError 与统一错误映射
│   │   ├── fallback.py                     # fallback 决策和安全文案
│   │   ├── input_guard.py                  # 输入、文件、目录和 session 边界校验
│   │   ├── llm_factory.py                  # LLM 工厂（模型管理）
│   │   ├── milvus_client.py                # Milvus 客户端
│   │   ├── request_context.py              # trace_id/request_id 请求上下文
│   │   └── token_budget.py                 # token 预算、裁剪和 usage/cost 记录
│   ├── memory/                             # 会话上下文管理
│   │   ├── __init__.py
│   │   ├── conversation_manager.py         # ConversationManager 与 MemorySaver 边界
│   │   └── summarizer.py                   # 会话摘要生成和安全过滤
│   ├── models/                             # 数据模型层
│   │   ├── __init__.py
│   │   ├── aiops.py                        # AIOps 模型
│   │   ├── document.py                     # 文档模型
│   │   ├── request.py                      # 请求模型
│   │   └── response.py                     # 响应模型
│   ├── observability/                      # 可观测性
│   │   ├── __init__.py
│   │   ├── metrics.py                      # JSONL metrics 记录
│   │   └── tracing.py                      # JSONL trace/span 记录
│   ├── rag/                                # RAG 工程化模块
│   │   ├── __init__.py
│   │   ├── citation.py                     # citation 构建和 API-safe 转换
│   │   ├── context_builder.py              # context packing、去重、no-answer
│   │   ├── models.py                       # 文档/chunk/retrieval/citation 内部模型
│   │   ├── reranker.py                     # reranker 插口（默认关闭）
│   │   └── retriever.py                    # 可控检索、分数归一化和阈值过滤
│   ├── services/                           # 业务服务层
│   │   ├── __init__.py
│   │   ├── rag_agent_service.py            # RAG Agent 服务和旧工具路径适配
│   │   ├── aiops_service.py                # AIOps 服务（计划-执行-重规划）
│   │   ├── document_splitter_service.py    # 文档分割服务
│   │   ├── vector_embedding_service.py     # 向量 embedding 服务
│   │   ├── vector_index_service.py         # 向量索引服务
│   │   ├── vector_search_service.py        # 向量检索服务
│   │   └── vector_store_manager.py         # 向量存储管理器
│   ├── tools/                              # Agent 工具集
│   │   ├── __init__.py
│   │   ├── knowledge_tool.py               # 知识库查询工具
│   │   └── time_tool.py                    # 时间工具
│   └── utils/                              # 工具类
│       ├── __init__.py
│       └── logger.py                       # 日志配置（Loguru）
├── docs/                                   # 工程化计划、API 契约和内部模块契约
├── evaluation/                             # 离线 RAG 评估 runner 和指标
│   ├── __init__.py
│   ├── datasets.py                         # eval set 加载和校验
│   ├── judge.py                            # 可选 LLM judge（默认关闭）
│   ├── rag_metrics.py                      # Hit@K / Recall@K / MRR
│   └── runner.py                           # dry-run evaluation CLI 和报告输出
├── eval_sets/                              # 评估数据集
│   └── rag_cases.yaml                      # 最小 RAG eval baseline
├── tests/                                  # 自动化测试（fake fixture，不依赖真实外部服务）
│   ├── conftest.py                         # fake LLM/RAG/MCP/vector fixtures
│   ├── unit/                               # 单元测试
│   ├── agent/                              # Agent/Tool/SSE 边界测试
│   ├── rag/                                # RAG 与 evaluation 测试
│   └── integration/                        # FastAPI 集成测试
├── scripts/                                # 辅助脚本
├── static/                                 # Web 前端（纯静态）
│   ├── index.html                          # 主页面
│   ├── app.js                              # 前端逻辑
│   └── styles.css                          # 样式表
├── mcp_servers/                            # MCP 服务器
│   ├── cls_server.py                       # CLS 日志查询服务
│   ├── monitor_server.py                   # 监控数据服务
│   └── README.md                           # MCP 服务说明
├── aiops-docs/                             # 运维知识库（Markdown 文档）
├── eval_reports/                           # evaluation 临时报告目录（默认不提交报告产物）
├── logs/                                   # 日志目录（Loguru 自动创建）
│   ├── app_YYYY-MM-DD.log                  # 按天轮转的应用日志
│   ├── trace.jsonl                         # trace JSONL（配置开启时）
│   └── metrics.jsonl                       # metrics JSONL（配置开启时）
├── uploads/                                # 上传文件临时目录
├── volumes/                                # Milvus 数据持久化目录
├── .env                                    # 环境变量配置（需手动创建）
├── .gitignore                              # Git 忽略规则
├── Makefile                                # 项目管理命令（Linux/macOS）
├── start-windows.bat                       # Windows 启动脚本
├── stop-windows.bat                        # Windows 停止脚本
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、元数据）
├── uv.lock                                 # uv 依赖锁定文件
├── pyrightconfig.json                      # Pyright 类型检查配置
└── README.md                               # 项目说明
```

## ⚙️ 配置说明

通过 `.env` 文件配置：

```bash
# 阿里云LLM DashScope 配置（必填）
# 秘钥管理： https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.61ac133ccTVQLw&tab=demohouse#/api-key
DASHSCOPE_API_KEY=your-api-key （配置你自己的秘钥）
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # 可选；不配置时使用代码中的同名默认值
DASHSCOPE_MODEL=qwen-max

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# RAG 配置
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# 文件上传与目录索引边界
UPLOAD_MAX_BYTES=10485760
ALLOWED_UPLOAD_EXTENSIONS=[".txt",".md",".markdown"]
INDEX_ALLOWED_DIRECTORIES=["uploads","aiops-docs"]
```

上传和目录索引以后端配置为准：默认支持 `.txt`、`.md`、`.markdown`，单文件上限 10 MiB；目录索引只允许访问 `uploads`、`aiops-docs` 等 allowlist 内路径。

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 核心特性
- ✅ 自动制定诊断计划（Planner）
- ✅ 智能工具调用（Executor）
- ✅ 动态调整步骤（Replanner）
- ✅ 流式输出诊断过程
- ✅ 生成结构化报告

### 快速测试

```bash
# 服务已通过 make init 自动启动
# 如需重启服务：make restart

# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test"}' \
  --no-buffer
```

### 诊断流程
```
1. Planner 制定计划 → 生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续/调整/生成报告
4. 输出诊断报告 → 根因分析 + 运维建议
```

## 📝 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Docker 容器
make down              # 停止 Docker 容器

# 代码质量
make format            # 格式化代码
make lint              # 代码检查
```

### CI 回归命令矩阵

以下命令均应避免依赖真实 Milvus、DashScope、MCP server 或网络。完整回归建议按从快到慢执行：

请先安装项目依赖并激活虚拟环境；Windows 也可以把 `pytest` 替换为 `.\.venv\Scripts\python.exe -m pytest`。

```bash
# 单元层与边界层
pytest tests/unit -q
pytest tests/agent -q
pytest tests/rag -q

# API 集成层（使用 fake fixture，不访问真实外部服务）
pytest tests/integration -q

# 全量测试与覆盖率
pytest tests -q

# 离线 RAG evaluation dry-run，默认写入 eval_reports/
python -m evaluation.runner --dataset eval_sets/rag_cases.yaml --judge disabled --output eval_reports
```

可选质量检查：

```bash
python -m ruff check app tests evaluation
python -m ruff format --check app tests evaluation
```

### Evaluation Runner

`evaluation.runner` 是离线回归工具，不进入线上 FastAPI 请求路径。默认 `--adapter dry-run` 与 `--judge disabled`，只验证数据集加载、轻量检索指标、JSON/Markdown 报告和趋势文件写入；只有显式选择 `--adapter retriever` 或 `--judge enabled` 时才会尝试使用真实检索或 LLM judge。


## 🐛 常见问题

### Windows 环境问题

#### 1. `make` 命令不可用
Windows 不支持 `make` 命令，请使用提供的批处理脚本：
```powershell
# 启动服务
.\start-windows.bat

# 停止服务
.\stop-windows.bat
```

#### 2. PowerShell 执行策略限制
如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 错误：
```powershell
# 临时允许脚本执行（管理员权限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或者使用 CMD 而不是 PowerShell
cmd
.\start-windows.bat
```

#### 3. 端口被占用（Windows）
```powershell
# 查看占用端口的进程
netstat -ano | findstr :9900

# 结束进程（替换 PID 为实际进程 ID）
taskkill /F /PID <PID>
```

### 通用问题

### API Key 错误
```bash
# 检查环境变量
cat .env | grep DASHSCOPE_API_KEY    # Linux/macOS
type .env | findstr DASHSCOPE_API_KEY  # Windows
```

### Milvus 连接失败
```bash
# 确保本机有 Docker 服务并且已经启动（可以使用 Docker Desktop）

# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus（使用 docker compose）
docker compose -f vector-database.yml restart

# 或者重启单个服务
docker compose -f vector-database.yml restart standalone
```

### 服务无法启动

**Linux/macOS:**
```bash
# 查看服务日志
tail -f logs/app_$(date +%Y-%m-%d).log  # FastAPI 主服务（Loguru 日志）
tail -f mcp_cls.log                      # CLS MCP 服务
tail -f mcp_monitor.log                  # Monitor MCP 服务

# 检查端口占用
lsof -i :9900  # FastAPI
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

**Windows:**
```powershell
# 查看服务日志（获取今天的日期）
$today = Get-Date -Format "yyyy-MM-dd"
type logs\app_$today.log  # FastAPI 主服务（Loguru 日志）
type mcp_cls.log          # CLS MCP 服务
type mcp_monitor.log      # Monitor MCP 服务

# 或者查看最新的日志文件
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# 检查端口占用
netstat -ano | findstr :9900  # FastAPI
netstat -ano | findstr :8003  # CLS MCP
netstat -ano | findstr :8004  # Monitor MCP
```

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [阿里云 DashScope](https://dashscope.aliyun.com/)
- [MCP 协议](https://modelcontextprotocol.io/)

## 📄 许可证
author： chief

MIT License
