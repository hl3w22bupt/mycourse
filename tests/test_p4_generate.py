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
