"""P5 参考实现：TinyGPT —— miniGPT 式的组装层。

对照阅读建议：
- nanoGPT ``model.py`` 的 ``Block``/``GPT``：结构与这里一致，工业版多了
  flash-attention、混合精度、DDP、fused AdamW 等工程外衣。
- 位置编码：本仓库走「可学习位置嵌入（P2）+ RoPE（P3 attention 内部）」的组合，
  与 nanoGPT 纯可学习位置嵌入略有不同 —— 目的是把 P3 的 RoPE 真正用进闭环。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...p2_embedding.solutions.embedding import Embedding
from ...p3_attention.solutions.attention import CausalSelfAttention
from ..model import TinyGPTConfig  # noqa: F401  (re-export 供 train 脚本使用)


class Block(nn.Module):
    """pre-LN transformer 块：attention 残差 + MLP 残差。"""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            n_embd=config.d_model,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    """最小可训练 GPT：embedding → n_layer × Block → LayerNorm → lm_head。"""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.emb = Embedding(config.vocab_size, config.d_model, config.block_size, config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list | None = None,
        use_cache: bool = False,
    ):
        x = self.emb(idx, pos_offset=0 if past_kv is None else past_kv[0][0].shape[-2])
        new_kvs: list = []
        for i, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[i]
            x, new_kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, (new_kvs if use_cache else None)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.tok_emb.weight.numel()
            n -= self.emb.pos_emb.weight.numel()
        return n
