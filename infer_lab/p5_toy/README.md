# P5 - 训练 + 生成闭环（TinyGPT）

## 学习目标
- 把 P2-P4 的零件组装成可训练的小 GPT（对标 nanoGPT 的 model.py 骨架）。
- 跑通「语料 → batch → 训练（loss 下降）→ checkpoint → 生成文本」完整闭环。

## 任务
1. 实现 `model.py` 的 `TinyGPT`（组装清单见 docstring：Embedding → n×Block → ln_f → lm_head）。
2. 实现 `data.py` 的 `CharDataset` / `train_val_split` / `get_batch`（x 与 y 右移一位）。
3. `train.py` / `generate.py` 是可直接运行的工作脚本（内部按 INFER_LAB_TARGET 装载你的实现）。

## 验收命令
```bash
make train-demo      # solution 态：loss 明显下降，输出 SAMPLE_TEXT
make generate-demo   # 加载 checkpoint 生成文本
pytest tests/test_p5_train_generate.py -q   # 秒级小训练 + loss 下降断言
```

用自己的实现跑 demo：
```bash
INFER_LAB_TARGET=user make train-demo
```
