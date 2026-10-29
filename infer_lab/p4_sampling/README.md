# P4 - 采样策略（temperature / top-k / top-p）

## 学习目标
- 三条策略各自回答一个问题：
  - **temperature**：多大胆？（logits/T，T→0 趋近贪心）
  - **top-k**：候选几个？（固定数量，不看分布形状）
  - **top-p（nucleus）**：候选凑够多大概率 mass？（自适应候选数）
- 理解为什么过滤要在 softmax 之前用 `-inf` 完成。
- 理解批量（B>1）下的按行独立：三条策略都只动最后一维，批量计算必须逐行等价于单行计算。

## 任务
1. 实现 `softmax_with_temperature`（数值稳定：先减 max）。
2. 实现 `top_k_filtering` / `top_p_filtering`（nucleus：累计概率首次覆盖 p 为止）。
3. 实现 `sample_token`：缩放 → 过滤 → softmax → 多项式采样，可复现（generator 注入）。
   - 单序列（`(V,)` / `(1, V)`）返回 python `int`；
   - 批量（`(B, V)` 且 `B > 1`）返回每行一个 id 的 `(B,)` LongTensor —— 每行用本行分布采样。
   - 温度/top-k/top-p **按行独立**：批量结果 ≡ 逐行单独计算（不许跨行平均/串 max），
     交换 batch 行序结果同步交换（`test_batch_softmax_and_filters_equal_per_row_computation` 等）。
4. 通过 `pytest tests/test_p4_sampling.py`。

## 验收命令
```bash
pytest tests/test_p4_sampling.py -q
```

## 参考实现
`infer_lab/p4_sampling/solutions/sampling.py`
