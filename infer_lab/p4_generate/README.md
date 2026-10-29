# P4 - 自回归生成循环

## 学习目标
- 手写生成主循环：取最后一步 logits → 采样 → 拼回 → 继续，理解框架 `generate()` 的内核。
- 掌握 KV cache 的使用姿势：**首步喂整段 prompt，之后每步只喂 1 个新 token**。
- 理解 eos 提前停止与批量解码的关系。

## 任务
1. 实现 `generate(model, input_ids, config)`（`GenerationConfig` 已给出）。
2. 通过 `pytest tests/test_p4_generate.py`，重点两条对拍：
   - `use_cache=True` 与 `False` 输出逐 token 一致（cache 只能加速，不能改变结果）；
   - 贪心（top_k=1）与手写 argmax 展开循环一致。

## 验收命令
```bash
pytest tests/test_p4_generate.py -q
```

## 参考实现
`infer_lab/p4_generate/solutions/generate.py`
