# 增量向量索引实施计划

- 日期：2026-08-19
- Spec：`docs/superpowers/specs/2026-08-19-incremental-vector-index-design.md`
- 测试命令：`.venv\Scripts\python.exe -m pytest tests\rag -q`（全量：`.venv\Scripts\python.exe -m pytest -q`）

## Task 1：配置开关

**文件**：`app/config.py`

在 `stable_rag_ids_enabled`（L170）之后新增：

```python
# 增量索引：开启后 index_single_file 先按 doc_id 查询已有 chunk 的
# content_hash 做主键级 diff，未变更 chunk 不再重复 embedding。
# 紧急回滚时设为 false，恢复 ISSUE-019 的全量重建路径。
incremental_index_enabled: bool = True
```

**验证**：`python -c "from app.config import config; print(config.incremental_index_enabled)"` 输出 `True`。

## Task 2：VectorStoreManager 新增查询与精确删除

**文件**：`app/services/vector_store_manager.py`

新增两个方法（放在 `delete_by_doc_id` 之后）：

```python
def get_chunk_hashes_by_doc_id(self, doc_id: str) -> dict[str, str]:
    """查询 doc 下已有 chunk 的 {chunk_id: content_hash}。

    返回空 dict 表示首建或该 doc 仅有 legacy 数据（legacy 无 doc_id，查询不可见）。
    失败映射为 VectorStoreUnavailableError，不静默吞错。
    """
    # 实现：milvus_manager.get_collection().query(
    #   expr=f'metadata["doc_id"] == "{_escape_milvus_string(doc_id)}"',
    #   output_fields=["metadata"])
    # 从每条结果的 metadata JSON 中取合法的 chunk_id/content_hash 字符串对；
    # 缺字段或类型不对的条目跳过（迁移期脏数据不阻塞索引）。

def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
    """按主键 id in [...] 精确删除指定 chunk，返回删除数量。"""
    # 实现：去空、转义、拼 expr=f'id in ["{id1}", "{id2}"]'，collection.delete(expr)
    # 空 chunk_ids 直接返回 0，不产生 Milvus 调用。
```

错误映射与日志风格对齐现有 `delete_by_doc_id`（L201-231）。

**已知边界**：Milvus query 窗口默认上限 16384 条；单 doc chunk 数远低于此（10MB 文件 × 最小 300 字符合并粒度，实际几十至几百），不在本次引入分页。

## Task 3：index_single_file 增量 diff 逻辑

**文件**：`app/services/vector_index_service.py`

### 3a. `SingleFileIndexResult` 增加计数字段

```python
added_count: int = 0      # 本次新增/变更写入的 chunk 数
skipped_count: int = 0    # 未变更跳过的 chunk 数
```

frozen dataclass 末尾带默认值，`empty()` 显式置 0，旧调用点不受影响。

### 3b. 替换非空分片路径（现 L395-426）

新增模块级辅助 `_has_valid_chunk_identities(documents) -> bool`：所有分片的
`chunk_id`、`content_hash` 均为非空 str 才返回 True（防御迁移期混合数据）。

```python
doc_id = _extract_doc_id(documents)
chunk_ids = _extract_chunk_ids(documents)

use_incremental = (
    config.incremental_index_enabled
    and config.stable_rag_ids_enabled
    and doc_id is not None
    and _has_valid_chunk_identities(documents)
)

if use_incremental:
    old_hashes = vector_store_manager.get_chunk_hashes_by_doc_id(doc_id)
    new_hashes = {每片 metadata["chunk_id"]: metadata["content_hash"]}
    added_documents = [d for d in documents
                       if old_hashes.get(d.metadata["chunk_id"]) != d.metadata["content_hash"]]
    removed_chunk_ids = [cid for cid, h in old_hashes.items()
                         if new_hashes.get(cid) != h]   # 覆盖"消失"与"被变更顶替"

    if not added_documents and not removed_chunk_ids:
        # 全部未变更：零删除、零写入、零 embedding
        rag.index.end 日志（added=0, skipped=len, deleted=0）
        return SingleFileIndexResult(doc_id, chunk_ids, deleted_count=0,
                                     source_deleted_count=0, added_count=0,
                                     skipped_count=len(documents))

    chunk_deleted_count = (vector_store_manager.delete_by_chunk_ids(removed_chunk_ids)
                           if removed_chunk_ids else 0)
    # old 为空 = 首建：仍执行一次 delete_by_source 清理旧 UUID 时代 legacy 残留
    source_deleted_count = (vector_store_manager.delete_by_source(normalized_path)
                            if not old_hashes else 0)
    added_ids = vector_store_manager.add_documents(added_documents)
    # rag.index.end 日志输出 added/skipped/deleted 计数
    return SingleFileIndexResult(doc_id=doc_id,
                                 chunk_ids=indexed_chunk_ids or chunk_ids,
                                 deleted_count=chunk_deleted_count,
                                 source_deleted_count=source_deleted_count,
                                 added_count=len(added_documents),
                                 skipped_count=len(documents) - len(added_documents))

# 否则：现有全量重建路径原样保留（delete_by_doc_id → delete_by_source → add_documents），
# 返回值补 added_count=len(documents), skipped_count=0
```

**空分片分支**（现 L366-393）不动。

## Task 4：目录级汇总与可观测性

**文件**：`app/services/vector_index_service.py`

- `IndexingResult.__init__` 新增 `self.added_chunk_count = 0` /
  `self.skipped_chunk_count = 0` / `self.deleted_chunk_count = 0`。
- `index_directory` 成功分支：`isinstance(single_file_result, SingleFileIndexResult)`
  时累加三个计数（旧式 None 返回跳过）。
- `to_dict()` 增加三个键；目录级 `rag.index.end` 日志同步输出。

## Task 5：测试

### 5a. 适配现有测试（`tests/rag/test_metadata_indexing.py`）

- `RecordingIndexStoreManager` 新增：
  - `get_chunk_hashes_by_doc_id(doc_id)`：返回构造时注入的 `old_hashes`（默认 `{}`），记录调用
  - `delete_by_chunk_ids(chunk_ids)`：记录调用，返回 `len(chunk_ids)`
- `test_index_single_file_deletes_by_doc_id_before_source_fallback`（L357）加
  `monkeypatch.setattr(index_module.config, "incremental_index_enabled", False)`——
  该用例专测 ISSUE-019 全量回退路径的删除顺序。
- 其余用例（首建路径）在新默认行为下断言不变：old 空 → 不调 `delete_by_doc_id`、
  `delete_by_source` 一次、全量写入；`indexed_doc_ids`/`added_documents` 断言保持。

### 5b. 新增 `tests/rag/test_incremental_index.py`

| # | 用例 | 关键断言 |
|---|------|----------|
| 1 | 首次索引（old 空） | `delete_by_source` 调用一次、无 `delete_by_doc_id`、全量 add；added=2/skipped=0 |
| 2 | 内容无变化重复索引 | 仅一次 `get_chunk_hashes_by_doc_id`，零删除零 add；skipped=2 |
| 3 | 单 chunk 内容变更 | 只 add 变更片、只 delete 旧 chunk_id；added=1/skipped=1 |
| 4 | 段落删除（尾部消失） | removed 含消失 id，无多余 add |
| 5 | 中间插入（后缀平移） | 平移片按变更处理（一删一写），未受影响前缀 skipped |
| 6 | legacy 分片缺 doc_id | 回退全量路径（`delete_by_doc_id`+`delete_by_source`+全量 add） |
| 7 | `incremental_index_enabled=False` | 同 6 回退路径 |
| 8 | query 失败 | `VectorStoreUnavailableError` 向上抛，不降级 |
| 9 | 目录级汇总 | `to_dict()` 含三个计数且数值正确 |

### 5c. VectorStoreManager 单元测试（并入 5b 文件）

- `get_chunk_hashes_by_doc_id`：fake collection（带 `query()`，返回样例 metadata）→
  断言 query 表达式与解析结果；脏条目（缺 content_hash）被跳过
- `delete_by_chunk_ids`：断言 expr 形如 `id in ["a", "b"]`；空列表零调用；异常映射

## Task 6：全量回归

1. `.venv\Scripts\python.exe -m pytest tests\rag -q`
2. `.venv\Scripts\python.exe -m pytest -q`（确认 conftest 中
   `FakeIntegrationVectorIndexService` 等集成 fake 未受影响）
3. 修复回归至全绿。

## 执行顺序与依赖

Task 1 → 2 → 3 → 4 → 5 → 6（3 依赖 1/2；5 依赖 3/4）。
