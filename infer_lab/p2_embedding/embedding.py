"""P2 - embedding：词嵌入 + 可学习位置嵌入。

TODO(学习者)：实现 ``Embedding``。

要求：
1. ``__init__(vocab_size, d_model, block_size, dropout=0.0)``：
   - ``self.tok_emb = nn.Embedding(vocab_size, d_model)``
   - ``self.pos_emb = nn.Embedding(block_size, d_model)``（可学习位置嵌入，GPT-2 风格）
   - ``self.drop = nn.Dropout(dropout)``
2. ``forward(idx)``：``idx`` 形状 ``(B, T)``；返回 ``(B, T, d_model)``；
   位置下标用 ``torch.arange(T, device=idx.device)``，不要写死 CPU。
3.``positions`` 超出 ``block_size`` 时要抛 ``ValueError``（生成超长会被这里拦住）。

验收：``pytest tests/test_p2_embedding.py`` 全绿。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 Embedding.__init__")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 Embedding.forward")
