# P3 - KV cache（逐 token 推理的显存账本）

## 学习目标
- 明白为什么没有 cache 的解码是 O(T²)、有 cache 后每步只算新 token。
- 会算 KV cache 的显存账：`2 × n_layers × n_kv_head × seq_len × head_dim × batch × bytes`
  —— 这也是 GQA 存在的理由。

## 任务
1. 实现 `KVCache`：按层 `update(k, v)` 追加并返回全量历史；`seq_len`、`reset`、
   上下文管理器（一条请求一个作用域）。
2. 通过 `pytest tests/test_p3_kv_cache.py`（含与全量重算的对拍）。

## 验收命令
```bash
pytest tests/test_p3_kv_cache.py -q
```

## 参考实现
`infer_lab/p3_kv_cache/solutions/kv_cache.py`
