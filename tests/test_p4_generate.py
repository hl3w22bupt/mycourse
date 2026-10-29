"""P4 生成循环测试：对拍（cache on/off、贪心 ≡ argmax 展开）、eos 停止、参数校验。"""

from __future__ import annotations

import pytest
import torch

from infer_lab.p5_toy.solutions.model import TinyGPT, TinyGPTConfig
from infer_lab.testing import load_impl


class ScriptedModel:
    """测试桩：正常输出偏向 token 7；生成步数达到阈值后只给 eos。"""

    def __init__(self, vocab_size: int, eos_id: int, stop_after: int) -> None:
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.stop_after = stop_after
        self.calls = 0

    def __call__(self, idx, past_kv=None, use_cache=False):
        self.calls += 1
        logits = torch.full((idx.shape[0], idx.shape[1], self.vocab_size), -5.0)
        logits[..., 7] = 5.0
        if self.calls > self.stop_after:
            logits[..., :] = -50.0
            logits[..., self.eos_id] = 50.0
        return logits, (past_kv if use_cache else None)


@pytest.fixture()
def gen_mod():
    return load_impl("p4_generate", "generate")


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    model = TinyGPT(TinyGPTConfig(vocab_size=40, d_model=32, n_layer=1, n_head=2, n_kv_head=2, block_size=32))
    return model.eval()


def test_config_defaults(gen_mod):
    cfg = gen_mod.GenerationConfig()
    assert cfg.max_new_tokens == 32 and cfg.use_cache is True and cfg.eos_id is None


def test_output_length(gen_mod, tiny_model):
    cfg = gen_mod.GenerationConfig(max_new_tokens=6, top_k=1)
    out = gen_mod.generate(tiny_model, torch.tensor([[1, 2, 3]]), cfg)
    assert out.shape == (1, 9)


def test_greedy_matches_manual_argmax_loop(gen_mod, tiny_model):
    torch.manual_seed(0)
    cfg = gen_mod.GenerationConfig(max_new_tokens=6, top_k=1)
    out = gen_mod.generate(tiny_model, torch.tensor([[1, 2, 3, 4, 5]]), cfg)

    manual = [1, 2, 3, 4, 5]
    cur = torch.tensor([manual], dtype=torch.long)
    with torch.no_grad():
        for _ in range(6):
            logits, _, _ = tiny_model(cur)
            manual.append(int(logits[0, -1].argmax()))
            cur = torch.tensor([manual], dtype=torch.long)
    assert out[0].tolist() == manual


def test_use_cache_true_matches_false(gen_mod, tiny_model):
    torch.manual_seed(0)
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    with_cache = gen_mod.generate(tiny_model, ids, gen_mod.GenerationConfig(max_new_tokens=8, top_k=1, use_cache=True))
    without = gen_mod.generate(tiny_model, ids, gen_mod.GenerationConfig(max_new_tokens=8, top_k=1, use_cache=False))
    assert torch.equal(with_cache, without), "KV cache 只能加速，不能改变输出"


def test_eos_early_stops(gen_mod):
    model = ScriptedModel(vocab_size=12, eos_id=0, stop_after=3)
    cfg = gen_mod.GenerationConfig(max_new_tokens=10, top_k=1, eos_id=0)
    out = gen_mod.generate(model, torch.tensor([[9, 9]]), cfg)
    assert out.shape[1] == 2 + 4, "生成 3 个正常 token 后第 4 步命中 eos，应提前停止"


def test_rejects_bad_inputs(gen_mod, tiny_model):
    with pytest.raises(ValueError):
        gen_mod.generate(tiny_model, torch.tensor([1, 2, 3]), gen_mod.GenerationConfig())
    with pytest.raises(ValueError):
        gen_mod.generate(tiny_model, torch.tensor([[1]]), gen_mod.GenerationConfig(max_new_tokens=0))


# ---- 批量 (B>1)：逐行采样、eos 全停语义、cache 对拍（Bug cmvtfut20003om9sff0svg7kj）----


class RowBiasedModel:
    """测试桩：batch 各行偏好不同 token —— 逐行采样时各行新 token 必然不同。"""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, idx, past_kv=None, use_cache=False):
        logits = torch.full((idx.shape[0], idx.shape[1], self.vocab_size), -5.0)
        logits[:, -1, 3] = 5.0  # 默认所有行偏好 token 3
        logits[1:, -1, 3] = -50.0
        logits[1:, -1, 1] = 5.0  # 第 1 行改为偏好 token 1
        return logits, (past_kv if use_cache else None)


class StaggeredEosModel:
    """测试桩：行 0 在第 row0_stop 步起只给 eos，行 1 更晚 —— 两行不同时撞 eos。"""

    def __init__(self, vocab_size: int, eos_id: int, row0_stop: int, row1_stop: int) -> None:
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.row0_stop = row0_stop
        self.row1_stop = row1_stop
        self.calls = 0

    def __call__(self, idx, past_kv=None, use_cache=False):
        self.calls += 1
        step = self.calls - 1
        logits = torch.full((2, idx.shape[1], self.vocab_size), -5.0)
        logits[:, :, 7] = 5.0
        for row, stop in [(0, self.row0_stop), (1, self.row1_stop)]:
            if step >= stop:
                logits[row, :, :] = -50.0
                logits[row, :, self.eos_id] = 50.0
        return logits, (past_kv if use_cache else None)


def test_batch_rows_follow_their_own_distribution(gen_mod):
    """B>1 时每行必须采出**各自**的 id，不得把单条 id 广播给整个 batch。"""
    model = RowBiasedModel(vocab_size=12)
    cfg = gen_mod.GenerationConfig(max_new_tokens=4, top_k=1)
    out = gen_mod.generate(model, torch.tensor([[0, 0], [0, 0]]), cfg)
    assert out.shape == (2, 6)
    assert out[0, 2:].tolist() == [3, 3, 3, 3], "第 0 行应逐行采出自己的偏好 token 3"
    assert out[1, 2:].tolist() == [1, 1, 1, 1], "第 1 行应逐行采出自己的偏好 token 1"


def test_batch_eos_stops_only_when_all_rows_hit(gen_mod):
    """某行先撞 eos 不能提前停：要等全部 batch 行都命中才停。"""
    model = StaggeredEosModel(vocab_size=12, eos_id=0, row0_stop=2, row1_stop=4)
    cfg = gen_mod.GenerationConfig(max_new_tokens=10, top_k=1, eos_id=0)
    out = gen_mod.generate(model, torch.tensor([[9, 9], [8, 8]]), cfg)
    # 行 0 第 2 步、行 1 第 4 步才撞 eos → 生成 5 步后全停
    assert out.shape[1] == 2 + 5, f"应等最晚撞 eos 的行（第 4 步）才停，实际长度 {out.shape[1]}"
    # 新 token 从下标 2 开始：行 0 在第 0/1 步正常采 7，第 2/3/4 步是 eos
    assert out[0, 2:4].tolist() == [7, 7], "行 0 撞 eos 前应继续采出正常 token"
    assert out[0, 4:].tolist() == [0, 0, 0], "行 0 撞 eos 后的步都应是 eos"
    # 行 1 第 0..3 步正常采 7，第 4 步才撞 eos
    assert out[1, 2:6].tolist() == [7, 7, 7, 7], "行 1 在自己撞 eos 前应继续采出正常 token"
    assert out[1, 6] == 0


def test_batch_use_cache_true_matches_false(gen_mod, tiny_model):
    """批量 + KV cache：只加速、不改输出。"""
    torch.manual_seed(0)
    ids = torch.tensor([[3, 1, 4, 1, 5], [9, 8, 7, 6, 5]])
    with_cache = gen_mod.generate(tiny_model, ids, gen_mod.GenerationConfig(max_new_tokens=8, top_k=1, use_cache=True))
    without = gen_mod.generate(tiny_model, ids, gen_mod.GenerationConfig(max_new_tokens=8, top_k=1, use_cache=False))
    assert torch.equal(with_cache, without), "批量下 KV cache 同样只能加速，不能改变输出"
