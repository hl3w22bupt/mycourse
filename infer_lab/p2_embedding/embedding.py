"""P2 - embedding：词嵌入 + 可学习位置嵌入。

TODO(学习者)：实现 ``Embedding``。

要求：
1. ``__init__(vocab_size, d_model, block_size, dropout=0.0)``：
   - ``self.tok_emb = nn.Embedding(vocab_size, d_model)``
   - ``self.pos_emb = nn.Embedding(block_size, d_model)``（可学习位置嵌入，GPT-2 风格）
   - ``self.drop = nn.Dropout(dropout)``
   - 顺手暴露 ``self.block_size = block_size``（生成循环按它判越界）。
2. ``forward(idx, pos_offset=0)``：``idx`` 形状 ``(B, T)``；返回 ``(B, T, d_model)``；
   位置下标用 ``torch.arange(pos_offset, pos_offset + T, device=idx.device)``，不要写死 CPU。
   ``pos_offset`` 是 KV cache 逐步解码时本步 token 的绝对位置起点：无缓存前向
   （训练 / 全量重算）保持默认 0，带缓存第 i 步必须从「已缓存长度」继续编号，
   否则位置嵌入整体错位（P3/P5 的缓存生成依赖它）。
3. ``pos_offset + T`` 超出 ``block_size`` 时要抛 ``ValueError``（生成超长会被这里拦住）。

对照：参考实现见 ``solutions/embedding.py``。

验收：``pytest tests/test_p2_embedding.py`` 全绿。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 Embedding.__init__")

    def forward(self, idx: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 Embedding.forward")
