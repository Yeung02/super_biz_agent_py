# 增量向量索引设计（Incremental Vector Indexing）

- 日期：2026-08-19
- 状态：已获用户批准
- 影响模块：`app/config.py`、`app/services/vector_store_manager.py`、`app/services/vector_index_service.py`、`tests/rag/`

## 1. 背景与问题

当前 `VectorIndexService.index_single_file` 实现的是**幂等全量重建**：每次索引都按
`doc_id` 删除旧 chunk（ISSUE-019），再把全部分片重新经 DashScope embedding 后写入。
重复索引同一目录时，即使文件内容毫无变化，每个 chunk 都会重新调用 embedding API，
浪费成本与时间。

已有基础设施：

- 稳定逻辑 ID（ISSUE-018）：`doc_id = f(tenant_id, source_path)`、
  `chunk_id = f"{doc_id}#{chunk_index:06d}"`（含顺序）。
- chunk 级 `content_hash`（CRLF→LF 归一化的内容 hash）已写入 metadata JSON。
- Milvus `biz` collection 支持 JSON 路径表达式查询与按主键删除。

缺的只是"索引前对比"这一层 diff 逻辑。

## 2. 目标与非目标

### 目标

1. 文件内容未变化时，重复索引零 embedding 调用、零删除、零写入。
2. 内容变化时，只对新增/变更 chunk 重新 embedding 写入，只删除消失/变更的旧 chunk。
3. 结果可观测：单文件与目录级返回 added/skipped/deleted 计数，`rag.index.*`
   结构化日志同步输出。
4. 可独立回滚：新增配置开关，关闭后行为与现有全量重建完全一致。

### 非目标

- 不做文件级 mtime/监听触发（增量判定仍发生在每次显式索引调用内）。
- 不做本地 manifest/SQLite 缓存（Milvus 是唯一事实源）。
- 不改动检索链路、citation、context builder。
- 不处理多租户并发写同一 doc 的竞态（现状亦如此，单一索引任务串行执行）。

## 3. 方案选型

| 方案 | 说明 | 结论 |
|------|------|------|
| A. Milvus 现状查询 + 主键级 diff | 索引前 query 已有 chunk 的 `(chunk_id, content_hash)`，与本次分片做集合 diff，按主键精确增删 | **采用** |
| B. 本地 SQLite manifest 缓存文件级 hash | 文件 hash 未变整文件跳过 | 否决：引入第二状态源，与 Milvus 漂移时有丢数据风险 |
| C. A + B 混合 | manifest 快路径 + Milvus 校验兜底 | 否决：当前规模过度设计（YAGNI） |

选 A 的理由：零新增存储、无状态漂移；单次轻量 query 成本远低于全量重嵌入。

## 4. 详细设计

### 4.1 配置开关

`app/config.py` 新增：

```python
# 增量索引：开启后 index_single_file 先按 doc_id 查询已有 chunk 的
# content_hash 做主键级 diff，未变更 chunk 不再重复 embedding。
# 紧急回滚时设为 false，恢复 ISSUE-019 的全量重建路径。
incremental_index_enabled: bool = True
```

独立于 `stable_rag_ids_enabled`，两者可分别回滚。

### 4.2 VectorStoreManager 新增方法

```python
def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
    """查询 doc 下已有 chunk 的 {chunk_id: content_hash}。

    query 表达式复用现有 metadata["doc_id"] JSON 路径语法；
    返回空 dict 表示首建或该 doc 仅有 legacy 数据。
    """

def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
    """按主键 id in [...] 精确删除指定 chunk，返回删除数量。"""
```

错误映射沿用现有约定：查询/删除失败 → `VectorStoreUnavailableError`，
不静默吞错（与 ISSUE-019 原则一致）。

### 4.3 index_single_file 的 diff 流程

在分块完成后、现有删除/写入逻辑之前插入（替换现有第 4 步）：

```text
前置条件（全部满足才走增量路径）：
  config.incremental_index_enabled
  且 config.stable_rag_ids_enabled
  且 doc_id 可从分片 metadata 提取
  且 本次分片非空

old = vector_store_manager.get_chunk_hashes_by_doc_id(doc_id)
new = {分片 metadata 的 chunk_id: content_hash}

added   = [c for c in new 分片 if old.get(chunk_id) != content_hash]   # 新增或内容变更
removed = [chunk_id for chunk_id in old
           if chunk_id 不在 new 或 new[chunk_id] != old[chunk_id]]      # 消失或被变更顶替

if not added and not removed:
    return skipped 路径（零删除、零写入、零 embedding）
else:
    delete_by_chunk_ids(removed)
    add_documents(added 对应分片)
```

diff 语义说明：

- `chunk_id` 含顺序，中间插入段落会使后缀 chunk_index 平移，这些 id 的
  content_hash 随之不同 → 自动判为"变更"（一删一写），语义正确。
- content_hash 已做 CRLF→LF 归一化，跨操作系统重复索引不会误判变更。

### 4.4 回退与兼容路径

以下任一条件成立时，走现有全量重建路径（`delete_by_doc_id` +
`delete_by_source` 兜底 + 全量 `add_documents`），行为与当前完全一致：

- `incremental_index_enabled = false`
- `stable_rag_ids_enabled = false`（稳定 ID 是增量的前提）
- 分片 metadata 缺 `doc_id`（迁移期 legacy Document）
- 分片为空（保持现状：删除该 doc 全部旧数据后返回）

`delete_by_source` 兜底的执行时机：**old 为空（首建）时仍执行一次**，因为该
路径可能残留旧 UUID 主键时代的 legacy 数据（它们没有 `doc_id`，按 doc_id 查询
不可见）；**old 非空时跳过**，此时该 doc 已是新式数据，此前重建已清理过 legacy
残留，再删 `_source` 属于多余写放大。

### 4.5 结果模型与可观测性

`SingleFileIndexResult` 新增字段（均有默认值，旧调用方不受影响）：

```python
added_count: int = 0      # 本次新增/变更写入的 chunk 数
skipped_count: int = 0    # 未变更跳过的 chunk 数
deleted_count: int        # 已有字段：本次删除的 chunk 数（含变更顶替的旧 chunk）
```

目录级 `IndexingResult` 新增汇总计数 `added_count / skipped_count /
deleted_count`，纳入 `to_dict()` 输出；`rag.index.end` 结构化日志同步输出
`added_chunk_count / skipped_chunk_count / deleted_chunk_count`。

旧字段（success/fail/status/failed_files/indexed_doc_ids 等）全部保留，
只增不改。

### 4.6 错误处理

- `get_chunk_hashes_by_doc_id` / `delete_by_chunk_ids` / `add_documents`
  失败：映射为现有稳定错误码（`VECTOR_STORE_UNAVAILABLE` /
  `EMBEDDING_PROVIDER_ERROR`），任务级标记失败；不降级为全量重试（避免
  失败时静默放大写入量）。
- 目录级：单文件增量失败进入现有 `failed_files` 清单，其余文件继续。

## 5. 测试计划

基于 `tests/conftest.py` 现有 `FakeVectorStore` 基建扩展：

| 用例 | 断言 |
|------|------|
| 首次索引（old 为空） | 全量写入，added=全部，skipped=0 |
| 内容无变化重复索引 | 零删除零写入，skipped=全部 |
| 新增段落 | 仅新/平移 chunk 写入，其余 skipped |
| 删除段落 | 仅消失 chunk 被删 |
| 中间插入（后缀平移） | 平移部分按变更处理（一删一写） |
| legacy 分片（无 doc_id） | 回退全量重建路径 |
| 开关关闭 | 回退全量重建路径，行为与现状一致 |
| query 失败 | 抛 `VectorStoreUnavailableError`，不静默降级 |
| 空文件 | 保持现状：删光返回 |

## 6. 风险与边界

- **chunk_index 平移导致"伪变更"**：中间插入会使后缀全部重嵌。属于
  content_hash 语义的正确代价，接受；文件级 hash 快路径属非目标。
- **Milvus query 表达式性能**：单 doc chunk 数量级为几十至几百，
  `metadata["doc_id"]` 等值查询无压力；不做 JSON 路径索引。
- **并发索引同一文件**：现状即串行任务，不在本次范围引入锁。
