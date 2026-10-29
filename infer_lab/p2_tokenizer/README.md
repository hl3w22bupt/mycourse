# P2 - tokenizer（字符级）

## 学习目标
- 理解「文本 ↔ id 序列」是推理引擎最前端的契约：编码、解码、未登录字符、pad/bos/eos。
- 为 P5 的训练数据流水线提供分词底座。

## 任务
1. 实现 `tokenizer.py` 的 `CharTokenizer`（接口与规格见模块 docstring）：
   词表构建、`encode`（unk/max_len/special）、`decode`、`pad_batch`、`save/load`。
2. 通过 `pytest tests/test_p2_tokenizer.py`。
3. 思考题：GPT-2 用 ByteLevel BPE 解决了字符级的什么问题？（提示：词表大小 vs 序列长度）

## 验收命令
```bash
pytest tests/test_p2_tokenizer.py -q
```

## 参考实现
`infer_lab/p2_tokenizer/solutions/tokenizer.py`
