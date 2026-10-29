"""P4 采样测试：temperature / top-k / top-p / 完整采样链。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.testing import load_impl


@pytest.fixture()
def mod():
    return load_impl("p4_sampling", "sampling")


LOGITS = torch.tensor([1.0, 5.0, 3.0, 2.0, 0.5])

# 批量：三行各自的 argmax 不同（1 / 3 / 4），且每行 top-2 候选集互不相同
BATCH_LOGITS = torch.tensor(
    [
        [1.0, 5.0, 3.0, 2.0, 0.5],  # 行 0：argmax=1，top-2={1,2}
        [5.0, 1.0, 2.0, 3.0, 0.5],  # 行 1：argmax=0，top-2={0,3}
        [0.5, 1.0, 2.0, 3.0, 9.0],  # 行 2：argmax=4，top-2={3,4}
    ]
)


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


# ---- 批量 (B>1, V)：sample_token 必须逐行独立采样（Bug cmvtfut20003om9sff0svg7kj）----


def test_sample_token_batch_returns_per_row_ids(mod):
    """(B>1, V) 输入返回 (B,) 张量，top_k=1 时每行取**各自**的 argmax。"""
    picked = mod.sample_token(BATCH_LOGITS, temperature=1.0, top_k=1)
    assert torch.is_tensor(picked), "批量输入应返回每行一个 id 的张量，而不是 python int"
    assert picked.shape == (BATCH_LOGITS.shape[0],)
    assert picked.tolist() == [1, 0, 4], "top_k=1 时每行必须取本行的 argmax"


def test_sample_token_single_sequence_stays_int(mod):
    """单序列（(V,) 与 (1, V)）保持向后兼容：返回 python int。"""
    assert isinstance(mod.sample_token(LOGITS, top_k=1), int)
    assert isinstance(mod.sample_token(LOGITS.unsqueeze(0), top_k=1), int)


def test_sample_token_batch_respects_filters_per_row(mod):
    """批量过滤也按行生效：每行只能从**本行**的 top-k 候选里采。"""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(321)
    draws = [mod.sample_token(BATCH_LOGITS, 1.0, top_k=2, generator=gen) for _ in range(200)]
    allowed_per_row = [{1, 2}, {0, 3}, {3, 4}]
    for row, allowed in enumerate(allowed_per_row):
        picks = {int(d[row]) for d in draws}
        assert picks <= allowed, f"第 {row} 行只可能从本行 top-2 {allowed} 采出，实际 {picks}"


def test_sample_token_batch_reproducible_with_same_seed(mod):
    a = torch.Generator().manual_seed(11)
    b = torch.Generator().manual_seed(11)
    s1 = mod.sample_token(BATCH_LOGITS, 0.9, top_k=4, generator=a)
    s2 = mod.sample_token(BATCH_LOGITS, 0.9, top_k=4, generator=b)
    assert torch.equal(s1, s2), "同 seed 的批量采样必须可复现"


# ---- 批量一等公民加固：温度/top-k/top-p 全部按样本（行）独立 ----


def test_batch_softmax_and_filters_equal_per_row_computation(mod):
    """温度缩放 softmax 与两条过滤在批量下必须逐行等价于「单独算这一行」。

    任何把行间信息混进来的实现（按 batch 平均、跨行 max/topk）都会在此挂掉。
    """
    for temperature in (0.5, 1.7):
        probs = mod.softmax_with_temperature(BATCH_LOGITS, temperature)
        for row in range(BATCH_LOGITS.shape[0]):
            alone = mod.softmax_with_temperature(BATCH_LOGITS[row], temperature)
            assert torch.allclose(probs[row], alone, atol=1e-6), (
                f"softmax_with_temperature 第 {row} 行（T={temperature}）与单行计算不一致"
            )
    for top_k in (1, 3, None):
        filtered = mod.top_k_filtering(BATCH_LOGITS, top_k)
        for row in range(BATCH_LOGITS.shape[0]):
            assert torch.equal(filtered[row], mod.top_k_filtering(BATCH_LOGITS[row], top_k)), (
                f"top_k_filtering 第 {row} 行（k={top_k}）与单行计算不一致"
            )
    for top_p in (0.5, 0.9):
        filtered = mod.top_p_filtering(BATCH_LOGITS, top_p)
        for row in range(BATCH_LOGITS.shape[0]):
            assert torch.equal(filtered[row], mod.top_p_filtering(BATCH_LOGITS[row], top_p)), (
                f"top_p_filtering 第 {row} 行（p={top_p}）与单行计算不一致"
            )


def test_batch_top_p_candidate_count_adapts_per_row(mod):
    """top-p 的自适应候选数按行独立：同一批量里尖锐行 1 个候选就够、平缓行要多个。"""
    batch = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],  # 尖锐行：p=0.6 第 1 项就覆盖 → 只留 1 个候选
            [1.0, 1.0, 1.0, 1.0],  # 平缓行：要多项凑够 p → 候选更多
        ]
    )
    f = mod.top_p_filtering(batch, 0.6)
    assert torch.isfinite(f[0]).sum().item() == 1, "尖锐行 top-p=0.6 应只保留 1 个候选"
    assert torch.isfinite(f[1]).sum().item() >= 2, "平缓行 top-p=0.6 应保留多个候选"


def test_sample_token_batch_row_permutation_invariance(mod):
    """每行输出只由本行 logits 决定：交换 batch 行序，贪心结果同步交换。"""
    picked = mod.sample_token(BATCH_LOGITS, temperature=1.0, top_k=1)
    perm = [2, 0, 1]
    picked_perm = mod.sample_token(BATCH_LOGITS[perm], temperature=1.0, top_k=1)
    assert picked_perm.tolist() == picked[perm].tolist(), (
        "行置换后结果必须跟着置换 —— 否则说明行与行之间有串扰"
    )


def test_sample_token_batch_stochastic_stays_in_own_row_support(mod):
    """随机采样路径同样按样本独立：各行只能采出**本行**候选集里的 id。

    候选集互不相交，把任何一行的 id 广播/串到别行都会立刻暴露。
    温度、top-k、top-p 各种组合下都要成立。
    """
    batch = torch.tensor(
        [
            [-5.0, 5.0, 5.0, -5.0, -5.0],  # 行 0 候选 {1, 2}
            [5.0, -5.0, -5.0, 5.0, -5.0],  # 行 1 候选 {0, 3}
        ]
    )
    allowed_per_row = [{1, 2}, {0, 3}]
    for kwargs in (
        {"temperature": 1.0},
        {"temperature": 0.3},
        {"temperature": 2.5, "top_k": 2},
        {"temperature": 1.2, "top_p": 0.9},
        {"temperature": 1.0, "top_k": 2, "top_p": 0.9},
    ):
        gen = torch.Generator().manual_seed(2024)
        draws = [mod.sample_token(batch, generator=gen, **kwargs) for _ in range(120)]
        for row, allowed in enumerate(allowed_per_row):
            picks = {int(d[row]) for d in draws}
            assert picks <= allowed, f"{kwargs}: 第 {row} 行只可能采出本行候选 {allowed}，实际 {picks}"
            assert len(picks) >= 2, f"{kwargs}: 第 {row} 行 120 次采样应覆盖本行多个候选（证明确实在采样）"


def test_sample_token_batch_sharp_row_deterministic_flat_row_varies(mod):
    """同一批量同一组采样参数：尖锐行过滤后只剩 1 个候选 → 恒定；平缓行 → 有变化。"""
    batch = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],  # 尖锐行
            [1.0, 1.0, 1.0, 1.0],  # 平缓行
        ]
    )
    gen = torch.Generator().manual_seed(7)
    draws = [mod.sample_token(batch, temperature=1.0, top_p=0.6, generator=gen) for _ in range(60)]
    assert {int(d[0]) for d in draws} == {0}, "尖锐行 top-p=0.6 只剩 1 个候选，必须恒采本行 argmax=0"
    assert len({int(d[1]) for d in draws}) >= 2, "平缓行 top-p=0.6 候选多个，采样应有变化（不能被钳成同列）"
