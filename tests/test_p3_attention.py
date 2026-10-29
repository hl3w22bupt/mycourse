"""P3 attention 测试：RoPE 性质 / 因果性 / GQA / KV cache 对拍。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.testing import load_impl


@pytest.fixture()
def mod():
    return load_impl("p3_attention", "attention")


# ---------- RoPE ----------

def test_rope_cache_shapes(mod):
    cos, sin = mod.build_rope_cache(seq_len=6, head_dim=8)
    assert cos.shape == (6, 4) and sin.shape == (6, 4)
    # 位置 0 不旋转
    assert torch.allclose(cos[0], torch.ones(4)) and torch.allclose(sin[0], torch.zeros(4))


def test_rope_relative_position_property(mod):
    """RoPE 核心性质：同一对 (q, k) 摆到不同位置，打分只依赖相对距离 j-i。"""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    cos, sin = mod.build_rope_cache(8, 8)

    def score(i: int, j: int) -> float:
        qi = mod.apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = mod.apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return float((qi * kj).sum())

    for dist in (-3, -1, 0, 1, 2, 4):
        # 固定相对距离 dist=j-i，把 (q,k) 沿序列合法滑动，打分必须不变
        lo, hi = max(0, -dist), 8 - max(0, dist)
        vals = [score(i, i + dist) for i in range(lo, hi)]
        assert len(vals) >= 3
        assert all(abs(v - vals[0]) < 1e-4 for v in vals), (
            f"相对距离 {dist} 下打分应平移不变：{vals}"
        )


def test_rope_preserves_norm(mod):
    """旋转是正交变换：向量长度不变。"""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 5, 8)
    cos, sin = mod.build_rope_cache(5, 8)
    xr = mod.apply_rope(x, cos, sin)
    assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


# ---------- repeat_kv (GQA) ----------

def test_repeat_kv_identity_and_expansion(mod):
    x = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 2, 3, 2)
    assert mod.repeat_kv(x, 1) is x
    r = mod.repeat_kv(x, 2)
    assert r.shape == (2, 4, 3, 2)
    assert torch.equal(r[:, 0], x[:, 0]) and torch.equal(r[:, 1], x[:, 0])
    assert torch.equal(r[:, 2], x[:, 1]) and torch.equal(r[:, 3], x[:, 1])


# ---------- CausalSelfAttention ----------

def test_attention_output_shape_and_kv_cache_shape(mod):
    torch.manual_seed(0)
    attn = mod.CausalSelfAttention(n_embd=16, n_head=4, n_kv_head=2).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out, kv = attn(x, use_cache=True)
    assert out.shape == (1, 5, 16)
    k, v = kv
    assert k.shape == (1, 2, 5, 4) and v.shape == (1, 2, 5, 4)  # n_kv_head=2, head_dim=4


def test_causal_property_future_tokens_do_not_leak(mod):
    torch.manual_seed(0)
    attn = mod.CausalSelfAttention(n_embd=16, n_head=4).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out1, _ = attn(x)
        x_perturbed = x.clone()
        x_perturbed[0, 4:] += 100.0  # 只扰动「未来」
        out2, _ = attn(x_perturbed)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), "位置 i 的输出不能依赖 j>i 的输入"


def test_gqa_with_full_kv_heads_equals_mha(mod):
    """n_kv_head == n_head 时 GQA 配置应与 MHA 完全一致。"""
    torch.manual_seed(0)
    mha = mod.CausalSelfAttention(n_embd=16, n_head=4, n_kv_head=None).eval()
    gqa = mod.CausalSelfAttention(n_embd=16, n_head=4, n_kv_head=4).eval()
    gqa.load_state_dict(mha.state_dict())
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        o1, _ = mha(x)
        o2, _ = gqa(x)
    assert torch.allclose(o1, o2, atol=1e-6)


def test_incremental_cached_decode_matches_full_forward(mod):
    """KV cache 核心性质：逐 token 带缓存解码 ≡ 全量前向。"""
    torch.manual_seed(0)
    attn = mod.CausalSelfAttention(n_embd=16, n_head=4, n_kv_head=2).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        full, _ = attn(x)
        outs, past = [], None
        for t in range(5):
            o, past = attn(x[:, t : t + 1], past_kv=past, use_cache=True)
            outs.append(o)
        incremental = torch.cat(outs, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_gqa_uses_fewer_kv_params_than_mha(mod):
    mha = mod.CausalSelfAttention(n_embd=64, n_head=8, n_kv_head=None)
    gqa = mod.CausalSelfAttention(n_embd=64, n_head=8, n_kv_head=2)
    assert sum(p.numel() for p in gqa.parameters()) < sum(p.numel() for p in mha.parameters())
