"""P4 参考实现：自回归生成循环。

学习要点：
- 生成循环 = 「取最后位置 logits → 采样 → 拼回 → 再喂进去」的 while 循环，
  框架（HF ``generate``）里这段逻辑被封装成 logits_processor + stopping_criteria；
  手写一遍才知道采样参数到底作用在哪一步。
- KV cache 的用法只有一句话：**首步喂整段 prompt，之后每步只喂 1 个新 token**，
  past_kv 由循环负责透传。
- 批量解码：每行从**本行**的 logits 分布里采样各自的 next token —— 把一条 id
  广播给整个 batch 会让所有序列生成完全相同的走向。
  批量语义正确的金标准：**批量输出 ≡ 逐条循环输出（逐 token 全等）**，cache 开/关都成立；
  随机采样路径下每行 token 只能落在本行候选集内（候选集互异即互异走向）。
- 提前停止：batch 场景某条序列先撞 eos，其它还在继续 —— 教学版用「全部命中才停」，
  足够展示思想；生产版会配合 padding + attention_mask 处理早停序列。
"""

from __future__ import annotations

import torch

from ...p4_sampling.solutions.sampling import sample_token
from ..generate import GenerationConfig  # noqa: F401  (re-export)


def generate(model, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError("input_ids 必须是 (B, T) 二维张量")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须 >= 1")

    seq = input_ids
    past_kv = None
    finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=seq.device)

    for _ in range(config.max_new_tokens):
        res = model(
            seq if past_kv is None else seq[:, -1:],
            past_kv=past_kv,
            use_cache=config.use_cache,
        )
        logits, past_kv = res[0], res[-1]
        next_ids = sample_token(
            logits[:, -1, :],
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
        if not torch.is_tensor(next_ids):  # 单序列（B=1）路径：sample_token 返回 python int
            next_ids = torch.full((seq.shape[0],), next_ids, dtype=torch.long, device=seq.device)
        column = next_ids.to(device=seq.device, dtype=seq.dtype).reshape(seq.shape[0], 1)
        seq = torch.cat([seq, column], dim=1)
        if config.eos_id is not None:
            finished |= seq[:, -1] == config.eos_id
            if bool(finished.all()):
                break
    return seq
