# P3 - attention（RoPE + 因果掩码 + MHA/GQA）

## 学习目标
- 手写带 RoPE 的因果自注意力，理解三个性质测试背后的数学：
  1. **因果性**：位置 i 的输出不能依赖 j>i 的输入（掩码）；
  2. **RoPE 相对性**：同一对 (q,k) 的打分只依赖相对距离 j-i；
  3. **缓存一致性**：逐 token 带缓存解码 ≡ 全量重算（批量 B>1 亦然）。
- 理解 GQA（分组查询注意力）为什么能按比例砍掉 KV cache 显存。

## 任务
1. 实现 `build_rope_cache` / `apply_rope`（旋转对编码）。
2. 实现 `repeat_kv`（GQA 的 KV 头复制）。
3. 实现 `CausalSelfAttention.forward(x, past_kv, use_cache)`：qkv 投影 → RoPE →
   因果打分 → softmax → 加权求和 → 输出投影。
   - 批量维度：`past_kv` 保持 `(B, n_kv_head, S, head_dim)`，各行输出/缓存互不串扰
     （`test_cached_decode_batch_dim_matches_full_forward_per_row` /
     `test_batch_rows_do_not_mix_in_attention`）。
4. 通过 `pytest tests/test_p3_attention.py`（四组性质测试全绿）。

## 验收命令
```bash
pytest tests/test_p3_attention.py -q
```

## 参考实现
`infer_lab/p3_attention/solutions/attention.py`
