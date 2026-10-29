"""P3 参考实现：RoPE + 因果多头注意力（MHA / GQA）+ KV cache 接入。

学习要点：
- RoPE 的核心思想：把「位置」编码成对 q/k 向量的**旋转**，两个 token 打分只依赖
  **相对距离**（q_i·k_j 只跟 i-j 有关），于是天生支持长度外推，也省掉了位置嵌入表。
- GQA（Grouped-Query Attention）：query 头多、KV 头少（如 8 个 q 头共享 2 组 KV）。
  推理时 KV cache 显存按 n_kv_head/n_head 比例缩小 —— 这是 Llama-2/3、Mistral
  的标准配置。n_kv_head == n_head 时退化成标准 MHA。
- 因果掩码：pre-training/生成都是「只看过去」。有 KV cache 时，掩码只需对**当前
  这一小步**的 T_new 个 query，挡住列下标 > offset + t 的 key（offset = 过去长度）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算每个位置、每个频率分量上的旋转角 cos/sin。"""
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须是偶数才能做旋转对")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    theta = torch.arange(seq_len, dtype=torch.float32, device=device)[:, None] * inv_freq[None, :]
    return theta.cos(), theta.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, n_head, T, head_dim) 的 q 或 k 施加旋转。"""
    d_half = x.shape[-1] // 2
    # (T, d/2) -> (1, 1, T, d/2)，对 batch 与 head 维广播
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 (B, n_kv_head, T, head_dim) 复制 n_rep 份给每个 query 头。"""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return x.reshape(b, n_kv * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    """带 RoPE 的因果自注意力，支持 MHA（n_kv_head=None）与 GQA。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = n_head
        self.n_kv_head = n_head if n_kv_head is None else n_kv_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head 必须能被 n_kv_head 整除")
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, (self.n_head + 2 * self.n_kv_head) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        n_q, n_kv, d = self.n_head, self.n_kv_head, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([n_q * d, n_kv * d, n_kv * d], dim=-1)
        q = q.view(b, t, n_q, d).transpose(1, 2)  # (B, n_head, T, d)
        k = k.view(b, t, n_kv, d).transpose(1, 2)  # (B, n_kv_head, T, d)
        v = v.view(b, t, n_kv, d).transpose(1, 2)

        # RoPE：q、k 同步旋转（v 不转）。有缓存时只对新增位置旋转，角度要加 offset。
        offset = 0 if past_kv is None else past_kv[0].shape[-2]
        cos, sin = build_rope_cache(offset + t, d, device=x.device)
        cos, sin = cos[offset:], sin[offset:]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # 追加 KV cache：推理时每步只算 1 个 token，cache 避免重算全部历史。
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v) if use_cache else None

        # GQA：KV 头复制给每个 query 头，再按 MHA 打分
        k_full, v_full = repeat_kv(k, n_q // n_kv), repeat_kv(v, n_q // n_kv)

        att = (q @ k_full.transpose(-2, -1)) / (d**0.5)  # (B, n_head, T_q, S)
        # 因果掩码：query 第 i 行（全局位置 offset+i）只能看 key 列 0..offset+i
        q_pos = torch.arange(offset, offset + t, device=x.device)[:, None]
        k_pos = torch.arange(0, k.shape[-2], device=x.device)[None, :]
        att = att.masked_fill(k_pos > q_pos, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v_full  # (B, n_head, T, d)
        out = out.transpose(1, 2).contiguous().view(b, t, n_q * d)
        return self.resid_dropout(self.out_proj(out)), new_kv
