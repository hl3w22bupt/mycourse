"""P3 KVCache 测试：追加 / 读长度 / 复位 / 上下文管理。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.testing import load_impl


@pytest.fixture()
def cls():
    mod = load_impl("p3_kv_cache", "kv_cache")
    return mod.KVCache


def test_update_appends_and_returns_full_history(cls):
    kv = cls(n_layers=2)
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    k_all, v_all = kv.update(0, k1, v1)
    assert k_all.shape[-2] == 3
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    k_all, v_all = kv.update(0, k2, v2)
    assert k_all.shape[-2] == 4
    assert torch.equal(k_all[:, :, :3], k1) and torch.equal(k_all[:, :, 3:], k2)
    assert torch.equal(v_all[:, :, :3], v1) and torch.equal(v_all[:, :, 3:], v2)


def test_layers_are_independent(cls):
    kv = cls(n_layers=2)
    k0 = torch.randn(1, 1, 2, 2)
    kv.update(0, k0, k0)
    assert kv.get(1) is None
    assert kv.get(0)[0].shape[-2] == 2


def test_seq_len_and_empty_state(cls):
    kv = cls(n_layers=1)
    assert kv.seq_len == 0
    k = torch.randn(1, 1, 5, 2)
    kv.update(0, k, k)
    assert kv.seq_len == 5 and len(kv) == 5


def test_reset_clears_everything(cls):
    kv = cls(n_layers=2)
    k = torch.randn(1, 1, 3, 2)
    kv.update(0, k, k)
    kv.update(1, k, k)
    kv.reset()
    assert kv.seq_len == 0 and kv.get(0) is None and kv.get(1) is None


def test_context_manager_scopes_a_request(cls):
    kv = cls(n_layers=1)
    k = torch.randn(1, 1, 3, 2)
    kv.update(0, k, k)
    with kv as scoped:
        assert scoped is kv
        assert kv.seq_len == 0  # 进入即复位：一条请求一个干净作用域
        kv.update(0, k, k)
        assert kv.seq_len == 3
    assert kv.seq_len == 0  # 退出即清


def test_out_of_range_layer_raises(cls):
    kv = cls(n_layers=2)
    k = torch.randn(1, 1, 1, 1)
    with pytest.raises(IndexError):
        kv.update(5, k, k)


# ---- 批量维度正确性：追加拼在时间维（-2），batch 各行的历史互不串行 ----


def test_batch_dim_append_is_along_time_not_batch(cls):
    """B>1 时 update 追加必须拼在时间维：形状 (B,H,S1+T,D)，而不是把 batch 拼起来。"""
    kv = cls(n_layers=1)
    k1 = torch.arange(3 * 2 * 4 * 4, dtype=torch.float32).reshape(3, 2, 4, 4)
    v1 = -k1
    k_all, v_all = kv.update(0, k1, v1)
    assert k_all.shape == (3, 2, 4, 4) and v_all.shape == (3, 2, 4, 4)

    k2 = torch.arange(3 * 2 * 1 * 4, dtype=torch.float32).reshape(3, 2, 1, 4) + 100.0
    v2 = -k2
    k_all, v_all = kv.update(0, k2, v2)
    assert k_all.shape == (3, 2, 5, 4), f"第二次 update 后时间维应为 4+1=5，实际 {tuple(k_all.shape)}"
    assert kv.seq_len == 5
    for b in range(3):
        assert torch.equal(k_all[b, :, :4], k1[b]), f"行 {b} 的历史段不能被改写或串行"
        assert torch.equal(k_all[b, :, 4:], k2[b]), f"行 {b} 的新增段必须拼在本行历史之后"
        assert torch.equal(v_all[b, :, :4], v1[b]) and torch.equal(v_all[b, :, 4:], v2[b])


def test_batch_rows_keep_their_own_history(cls):
    """batch 各行的缓存内容互不污染：每行缓存 == 该行单独累积的缓存。"""
    kv = cls(n_layers=1)
    torch.manual_seed(0)
    steps = [torch.randn(4, 2, t, 3) for t in (3, 1, 2)]  # 三步：prompt 3 + 逐 token 1/1
    for s in steps:
        kv.update(0, s, s.clone())

    k_all, v_all = kv.get(0)
    assert k_all.shape == (4, 2, 6, 3)
    for b in range(4):
        # 行 b 的缓存必须等于「只用行 b 的数据」累积出的缓存 —— 行间零串扰
        single = cls(n_layers=1)
        for s in steps:
            single.update(0, s[b : b + 1], s[b : b + 1].clone())
        k_single, v_single = single.get(0)
        assert torch.equal(k_all[b], k_single[0]), f"行 {b} 的 K 缓存与单独累积不一致（跨行串扰）"
        assert torch.equal(v_all[b], v_single[0]), f"行 {b} 的 V 缓存与单独累积不一致（跨行串扰）"
