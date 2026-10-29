"""P3 - attention：RoPE 旋转位置编码 + 因果多头注意力（MHA/GQA）+ KV cache 接入。

TODO(学习者)：实现以下四个部分。

1. ``build_rope_cache(seq_len, head_dim, base=10000.0, device=None)``
   返回 ``(cos, sin)``，形状都是 ``(seq_len, head_dim // 2)``：
   - 频率 ``inv_freq = 1 / base ** (arange(0, head_dim, 2) / head_dim)``
   - 角度 ``theta = arange(seq_len)[:, None] * inv_freq[None, :]``
   - ``cos, sin = theta.cos(), theta.sin()``

2. ``apply_rope(x, cos, sin)``：``x`` 形状 ``(B, n_head, T, head_dim)``；
   前一半维度与后一半维度视为一对，旋转对：
   ``x1, x2 = x[..., :d/2], x[..., d/2:]``；
   ``out = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)``。
   （cos/sin 广播到 ``(1, 1, T, head_dim//2)``。）

3. ``repeat_kv(x, n_rep)``：GQA 里把 KV 头复制 n_rep 份，``(B, n_kv_head, T, d) → (B, n_kv_head*n_rep, T, d)``；
   ``n_rep == 1`` 时原样返回。

4. ``CausalSelfAttention(n_embd, n_head, n_kv_head=None, dropout=0.0, bias=True)``：
   - ``n_kv_head=None`` 视为 MHA（等于 n_head）；``n_kv_head < n_head`` 即 GQA；
   - ``qkv_proj`` 一次线性投影出 ``q, k, v``；
   - forward 签名：``forward(x, past_kv=None, use_cache=False) -> (out, new_kv)``；
     ``past_kv = (k_past, v_past)`` 形状 ``(B, n_kv_head, S, head_dim)``，为 None 表示无缓存；
   - 注意力打分：``q @ k^T / sqrt(head_dim)``，加因果掩码（当前位置只能看 ≤ 当前位置，
     有缓存时列下标要加上过去长度 offset），softmax 后乘 ``v``。

验收：``pytest tests/test_p3_attention.py`` 全绿（含因果性 / RoPE 相对性 / GQA / 缓存对拍四组性质测试）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 build_rope_cache")


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 apply_rope")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 repeat_kv")


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CausalSelfAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CausalSelfAttention.forward")
