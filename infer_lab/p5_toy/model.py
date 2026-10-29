"""P5 - TinyGPT：把 P2/P3 的零件组装成一个可训练的小 GPT。

TODO(学习者)：实现 ``TinyGPT``。``TinyGPTConfig`` 已给出，不用改。

组装清单（对照 nanoGPT 的 model.py，少什么补什么）：
1. ``__init__``：
   - ``self.emb = Embedding(vocab_size, d_model, block_size, dropout)``（复用 P2）
   - ``self.blocks = nn.ModuleList([Block(...) for _ in range(n_layer)])``
   - ``self.ln_f = nn.LayerNorm(d_model)``
   - ``self.lm_head = nn.Linear(d_model, vocab_size, bias=False)``
2. ``Block``：pre-LN 结构
   ``x = x + attn(ln1(x))``；``x = x + mlp(ln2(x))``；
   MLP = Linear(d, 4d) → GELU → Linear(4d, d) → Dropout。
3. ``forward(idx, targets=None, past_kv=None, use_cache=False)``：
   - ``past_kv`` 是每层一个 ``(k, v)`` 元组的 list（长度 n_layer），None 表示无缓存；
   - 返回 ``(logits, loss, past_kv_new)``：
     ``targets=None`` 时 loss 为 None；有 targets 时交叉熵（预测平移一位后的 token）；
     ``use_cache=True`` 时收集每层新 KV 组成 list 返回。
4. ``num_params(non_embedding=True)``：统计参数量（学习用打印）。

验收：``pytest tests/test_p5_train_generate.py`` 全绿；
``make train-demo``（solution 态）loss 应明显下降。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TinyGPTConfig:
    vocab_size: int
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int | None = 2   # GQA；None 表示 MHA
    block_size: int = 64
    dropout: float = 0.0


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 TinyGPT.__init__")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list | None = None,
        use_cache: bool = False,
    ):
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 TinyGPT.forward")

    def num_params(self, non_embedding: bool = True) -> int:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 TinyGPT.num_params")
