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


# ---- 批量一等公民：批量 ≡ 逐条循环逐 token 对拍 + 按行独立采样 ----


class RowPreferenceStubModel:
    """确定性测试桩：行 r 第 s 步的偏好 token 固定为 ``(s + row_offset + r) % vocab``。

    关键性质：每行的 logits 只依赖 (行号, 步数)，与 batch 维度无关 ——
    单独跑第 r 行（实例取 ``row_offset=r``）和把它放进批量跑（实例 row_offset=0），
    该行每一步 logits 完全一致，「批量输出 == 逐条循环输出」的逐 token 对拍
    因此才成立（贪心可精确判等）。
    """

    def __init__(self, vocab_size: int, row_offset: int = 0) -> None:
        self.vocab_size = vocab_size
        self.row_offset = row_offset
        self.step = 0

    def __call__(self, idx, past_kv=None, use_cache=False):
        b = idx.shape[0]
        logits = torch.full((b, idx.shape[1], self.vocab_size), -8.0)
        for row in range(b):
            logits[row, :, (self.step + self.row_offset + row) % self.vocab_size] = 6.0
        self.step += 1
        return logits, (past_kv if use_cache else None)


class DisjointSupportStubModel:
    """测试桩：batch 各行的候选集互不相交（行 0 → {2,3}，行 1 → {5,6}）。

    随机采样路径下，任何一行都不可能采出别行候选集里的 id ——
    单条 id 广播给整个 batch 的缺陷会让别的行「说出」本行候选之外的 token，立即暴露。
    """

    def __init__(self, vocab_size: int, supports: list[list[int]]) -> None:
        self.vocab_size = vocab_size
        self.supports = supports

    def __call__(self, idx, past_kv=None, use_cache=False):
        b = idx.shape[0]
        logits = torch.full((b, idx.shape[1], self.vocab_size), -8.0)
        for row, toks in enumerate(self.supports):
            for j, tok in enumerate(toks):
                logits[row, :, tok] = 6.0 - 0.1 * j
        return logits, (past_kv if use_cache else None)


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "nocache"])
def test_batch_equals_per_row_loop_stub_greedy(gen_mod, use_cache):
    """批量生成 ≡ 逐条循环生成（确定性桩，逐 token 全等）：KV cache 开关两种姿态都要成立。"""
    batch = 4
    model = RowPreferenceStubModel(vocab_size=16)
    prompt = torch.arange(batch * 3, dtype=torch.long).reshape(batch, 3) % 16
    cfg = gen_mod.GenerationConfig(max_new_tokens=5, top_k=1, use_cache=use_cache)

    batch_out = gen_mod.generate(model, prompt, cfg)

    loop_rows = []
    for row in range(batch):
        single = gen_mod.generate(
            RowPreferenceStubModel(vocab_size=16, row_offset=row), prompt[row : row + 1], cfg
        )
        loop_rows.append(single[0])
    loop_out = torch.stack(loop_rows)

    assert batch_out.shape == loop_out.shape
    assert torch.equal(batch_out, loop_out), (
        f"批量生成与逐条循环必须逐 token 全等（use_cache={use_cache}）\n批量:\n{batch_out}\n逐条:\n{loop_out}"
    )


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "nocache"])
def test_batch_equals_per_row_loop_real_model_diverse_prompts(gen_mod, tiny_model, use_cache):
    """真模型 + 多样 prompt：批量 B>1 与逐条循环逐 token 全等（贪心），cache 开关均成立。"""
    torch.manual_seed(0)
    prompts = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [20, 21, 5, 7, 9, 11, 13, 3],
            [0, 39, 1, 38, 2, 37, 3, 36],
        ]
    )
    cfg = gen_mod.GenerationConfig(max_new_tokens=6, top_k=1, use_cache=use_cache)
    batch_out = gen_mod.generate(tiny_model, prompts, cfg)

    loop_rows = [
        gen_mod.generate(tiny_model, prompts[row : row + 1], cfg)[0] for row in range(prompts.shape[0])
    ]
    loop_out = torch.stack(loop_rows)

    assert batch_out.shape == (4, 8 + 6)
    assert torch.equal(batch_out, loop_out), (
        f"批量与逐条循环不一致（use_cache={use_cache}）\n批量:\n{batch_out}\n逐条:\n{loop_out}"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 1.0},
        {"temperature": 0.4},
        {"temperature": 2.0, "top_k": 2},
        {"temperature": 1.0, "top_p": 0.9},
        {"temperature": 1.5, "top_k": 2, "top_p": 0.9},
    ],
    ids=["plain", "lowT", "T+k", "p", "T+k+p"],
)
def test_batch_stochastic_rows_stay_in_own_supports(gen_mod, overrides):
    """随机采样路径的按行独立：每行新 token 只能落在本行候选集内，且各行走向互不相同。"""
    model = DisjointSupportStubModel(vocab_size=12, supports=[[2, 3], [5, 6], [8, 9]])
    prompt = torch.zeros(3, 2, dtype=torch.long)
    cfg = gen_mod.GenerationConfig(max_new_tokens=6, **overrides)

    out = gen_mod.generate(model, prompt, cfg)
    new = out[:, 2:]

    supports = [{2, 3}, {5, 6}, {8, 9}]
    for row, allowed in enumerate(supports):
        picks = set(new[row].tolist())
        assert picks <= allowed, (
            f"{overrides}: 第 {row} 行的新 token 必须全部落在本行候选 {allowed} 内，实际 {picks}"
        )
    # 三行候选集互不相交 → 各行走向必然互异；若出现两行全同即是被广播/串行
    for a in range(3):
        for b in range(a + 1, 3):
            assert not torch.equal(new[a], new[b]), f"{overrides}: 行 {a} 与行 {b} 走向完全相同（疑似广播）"


def test_batch_temperature_top_p_are_applied_per_row(gen_mod):
    """同一批量同一组采样参数：尖锐行被 top-p 收敛成确定走向，平缓行保持随机 —— 各行互不干扰。"""
    model = DisjointSupportStubModel(vocab_size=12, supports=[[2], [5, 6]])
    prompt = torch.zeros(2, 2, dtype=torch.long)
    cfg = gen_mod.GenerationConfig(max_new_tokens=8, temperature=1.0, top_p=0.9)

    out = gen_mod.generate(model, prompt, cfg)
    new = out[:, 2:]
    assert set(new[0].tolist()) == {2}, "尖锐行（单候选）top-p 过滤后只剩 1 个候选，走向必须恒定"
    assert len(set(new[1].tolist())) >= 2, "平缓行（双候选）在 8 步里应采出多于 1 种 token"
    assert set(new[1].tolist()) <= {5, 6}, "平缓行也只能采本行候选"
