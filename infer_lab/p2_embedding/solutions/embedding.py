"""P2 参考实现：词嵌入 + 可学习位置嵌入。

学习要点：
- GPT-1/2 的位置信息是「可学习的一张查询表」（区别于 P3 里的 RoPE：无需训练、外推友好）。
- 实现只有两行查表 + 相加，但有两个工程细节：
  1) ``arange`` 必须用 ``idx.device``，否则 MPS/CUDA 上直接报错；
  2) 越界要尽早失败（ValueError），而不是让 nn.Embedding 抛出难懂的 index 断言。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """token embedding + 可学习位置 embedding，输出直接进 attention 块。"""

    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size 必须 >= 1")
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """``pos_offset``：KV cache 逐步解码时，本步 token 的绝对位置起点。

        无缓存的前向（训练 / 全量重算）保持默认 0；带缓存生成第 i 步时，
        位置必须从「已缓存长度」继续编号，否则位置嵌入整体错位。
        """
        *_, seq_len = idx.shape
        if pos_offset + seq_len > self.block_size:
            raise ValueError(
                f"序列总长 {pos_offset + seq_len} 超过 block_size={self.block_size}，请截断或调大 block_size"
            )
        positions = torch.arange(pos_offset, pos_offset + seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        return self.drop(x)
