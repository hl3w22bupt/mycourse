# P2 - embedding（词嵌入 + 可学习位置嵌入）

## 学习目标
- 理解 token id 如何变成向量，以及「位置信息」为什么必须显式注入。
- 认识生成时的一个隐藏坑：**KV cache 逐步解码时位置编号要接续，不能从 0 重来**。

## 任务
1. 实现 `embedding.py` 的 `Embedding`（token 表 + 位置表 + Dropout）。
2. 实现 `forward(idx, pos_offset=0)`：带缓存生成时位置要从 `pos_offset` 继续编号。
3. 通过 `pytest tests/test_p2_embedding.py`。

## 验收命令
```bash
pytest tests/test_p2_embedding.py -q
```

## 参考实现
`infer_lab/p2_embedding/solutions/embedding.py`
