"""P2 embedding 测试：形状 / 位置信息 / 越界 / 梯度。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.testing import load_impl


def build_emb():
    """构造被测 Embedding（在测试体内调用，保证骨架态以 xfail 呈现「待实现」）。"""
    mod = load_impl("p2_embedding", "embedding")
    torch.manual_seed(0)
    return mod.Embedding(vocab_size=50, d_model=16, block_size=32)


def test_output_shape():
    emb = build_emb()
    out = emb(torch.randint(0, 50, (2, 7)))
    assert out.shape == (2, 7, 16)


def test_same_token_different_position_differs():
    emb = build_emb()
    emb.eval()
    idx = torch.tensor([[5, 5, 5]])
    out = emb(idx)
    assert not torch.allclose(out[0, 0], out[0, 1]), "位置嵌入应让同 token 不同位置有所区分"


def test_exceeding_block_size_raises():
    emb = build_emb()
    with pytest.raises(ValueError):
        emb(torch.randint(0, 50, (1, 33)))  # block_size=32


def test_gradients_flow_to_token_table():
    emb = build_emb()
    out = emb(torch.tensor([[1, 2, 3]]))
    out.sum().backward()
    assert emb.tok_emb.weight.grad is not None
    assert emb.pos_emb.weight.grad is not None


def test_device_follows_input():
    mod = load_impl("p2_embedding", "embedding")
    m = mod.Embedding(vocab_size=10, d_model=8, block_size=16)
    out = m(torch.zeros(1, 4, dtype=torch.long))  # CPU 上 arange 不应写死 device
    assert out.shape == (1, 4, 8)
