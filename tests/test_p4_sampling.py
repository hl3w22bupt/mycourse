"""P4 采样测试：temperature / top-k / top-p / 完整采样链。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.testing import load_impl


@pytest.fixture()
def mod():
    return load_impl("p4_sampling", "sampling")


LOGITS = torch.tensor([1.0, 5.0, 3.0, 2.0, 0.5])


def test_softmax_sums_to_one_and_orders(mod):
    probs = mod.softmax_with_temperature(LOGITS, temperature=1.0)
    assert probs.shape == LOGITS.shape
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs.argmax() == 1


def test_low_temperature_sharpens_distribution(mod):
    hot = mod.softmax_with_temperature(LOGITS, temperature=0.1)
    cold = mod.softmax_with_temperature(LOGITS, temperature=2.0)
    assert hot.max() > cold.max(), "低温应让分布更尖"


def test_non_positive_temperature_raises(mod):
    with pytest.raises(ValueError):
        mod.softmax_with_temperature(LOGITS, 0.0)
    with pytest.raises(ValueError):
        mod.softmax_with_temperature(LOGITS, -1.0)


def test_top_k_keeps_only_k_largest(mod):
    filtered = mod.top_k_filtering(LOGITS, top_k=2)
    keep = torch.isfinite(filtered)
    assert keep.sum().item() == 2
    assert keep.tolist() == [False, True, True, False, False]


def test_top_k_none_or_too_large_is_noop(mod):
    assert torch.equal(mod.top_k_filtering(LOGITS, None), LOGITS)
    assert torch.equal(mod.top_k_filtering(LOGITS, 0), LOGITS)
    assert torch.equal(mod.top_k_filtering(LOGITS, 100), LOGITS)


def test_top_p_nucleus_adapts_to_distribution_shape(mod):
    # 尖锐分布：前 1 项就覆盖 p=0.6 → 只留 1 个候选
    sharp = torch.tensor([10.0, 0.0, 0.0, 0.0])
    f = mod.top_p_filtering(sharp, 0.6)
    assert torch.isfinite(f).sum().item() == 1
    # 平缓分布：需要多项才能凑够 p → 候选更多
    flat = torch.tensor([1.0, 1.0, 1.0, 1.0])
    f2 = mod.top_p_filtering(flat, 0.6)
    assert torch.isfinite(f2).sum().item() >= 2


def test_top_p_invalid_raises(mod):
    with pytest.raises(ValueError):
        mod.top_p_filtering(LOGITS, 0.0)
    with pytest.raises(ValueError):
        mod.top_p_filtering(LOGITS, 1.5)


def test_sample_token_top_k_1_is_argmax(mod):
    for _ in range(10):
        assert mod.sample_token(LOGITS, temperature=1.0, top_k=1) == 1


def test_sample_token_respects_filters(mod):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(123)
    picks = {mod.sample_token(LOGITS, temperature=1.0, top_k=2, generator=gen) for _ in range(200)}
    assert picks <= {1, 2}, f"top_k=2 时只可能采出 1/2，实际 {picks}"


def test_sample_token_reproducible_with_same_seed(mod):
    a = torch.Generator().manual_seed(7)
    b = torch.Generator().manual_seed(7)
    s1 = [mod.sample_token(LOGITS, 0.9, top_k=4, generator=a) for _ in range(50)]
    s2 = [mod.sample_token(LOGITS, 0.9, top_k=4, generator=b) for _ in range(50)]
    assert s1 == s2
