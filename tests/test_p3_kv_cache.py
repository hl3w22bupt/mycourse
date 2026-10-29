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
