"""骨架态守护测试：所有 TODO 桩必须显式抛「待实现」的 NotImplementedError。

这条测试与 INFER_LAB_TARGET 无关（永远直接导入用户骨架文件）：
- 骨架态：保证学习者会收到明确的「待实现」提示，而不是静默的错误结果；
- solution 态：防止有人误删骨架或把桩改成「假装实现」。
验收口径：骨架态用例被收集并明确提示待实现（make test-user 全部 XFAIL「待实现」）。
"""

from __future__ import annotations

import importlib
from dataclasses import fields

import pytest
import torch

CASES = [
    "p0_env:check_env",
    "p0_env:format_report",
    "p2_tokenizer:build",
    "p2_tokenizer:encode",
    "p2_tokenizer:decode",
    "p2_tokenizer:pad_batch",
    "p2_tokenizer:save",
    "p2_tokenizer:load",
    "p2_tokenizer:vocab_size",
    "p2_embedding:forward",
    "p3_attention:build_rope_cache",
    "p3_attention:apply_rope",
    "p3_attention:repeat_kv",
    "p3_attention:forward",
    "p3_kv_cache:update",
    "p3_kv_cache:get",
    "p3_kv_cache:seq_len",
    "p3_kv_cache:reset",
    "p3_kv_cache:context_manager",
    "p4_sampling:softmax_with_temperature",
    "p4_sampling:top_k_filtering",
    "p4_sampling:top_p_filtering",
    "p4_sampling:sample_token",
    "p4_generate:generate",
    "p5_toy:model_forward",
    "p5_toy:num_params",
    "p5_toy:dataset_init",
    "p5_toy:vocab_size",
    "p5_toy:train_val_split",
    "p5_toy:get_batch",
    "p1_framework:init",
    "p1_framework:load",
    "p1_framework:complete",
    "p1_framework:chat",
    "p1_framework:render_chat_prompt",
]


def _thunk(stage: str, op: str):
    """返回一个零参 callable：构造与调用都包在里面，整体跑进 pytest.raises。"""

    def thunk():
        if stage == "p0_env":
            m = importlib.import_module("infer_lab.p0_env.env_check")
            return m.format_report({}) if op == "format_report" else m.check_env()
        if stage == "p2_tokenizer":
            m = importlib.import_module("infer_lab.p2_tokenizer.tokenizer")
            tok = m.CharTokenizer({"a": 4})
            return {
                "build": lambda: m.CharTokenizer.build("ab"),
                "encode": lambda: tok.encode("a"),
                "decode": lambda: tok.decode([4]),
                "pad_batch": lambda: tok.pad_batch([[4], [4, 5]]),
                "save": lambda: tok.save("/tmp/__never__.json"),
                "load": lambda: m.CharTokenizer.load("/tmp/__never__.json"),
                "vocab_size": lambda: tok.vocab_size,
            }[op]()
        if stage == "p2_embedding":
            m = importlib.import_module("infer_lab.p2_embedding.embedding")
            emb = m.Embedding(10, 8, 16)
            return emb.forward(torch.zeros(1, 2, dtype=torch.long))
        if stage == "p3_attention":
            m = importlib.import_module("infer_lab.p3_attention.attention")
            if op == "build_rope_cache":
                return m.build_rope_cache(4, 4)
            if op == "apply_rope":
                return m.apply_rope(torch.zeros(1, 1, 2, 4), torch.zeros(2, 2), torch.zeros(2, 2))
            if op == "repeat_kv":
                return m.repeat_kv(torch.zeros(1, 2, 2, 2), 2)
            attn = m.CausalSelfAttention(n_embd=8, n_head=2)
            return attn.forward(torch.zeros(1, 2, 8))
        if stage == "p3_kv_cache":
            m = importlib.import_module("infer_lab.p3_kv_cache.kv_cache")
            kv = m.KVCache(2)
            k = torch.zeros(1, 1, 1, 1)
            return {
                "update": lambda: kv.update(0, k, k),
                "get": lambda: kv.get(0),
                "seq_len": lambda: kv.seq_len,
                "reset": lambda: kv.reset(),
                "context_manager": lambda: kv.__enter__(),
            }[op]()
        if stage == "p4_sampling":
            m = importlib.import_module("infer_lab.p4_sampling.sampling")
            return {
                "sample_token": lambda: m.sample_token(torch.zeros(4)),
                "softmax_with_temperature": lambda: m.softmax_with_temperature(torch.zeros(4), 1.0),
                "top_k_filtering": lambda: m.top_k_filtering(torch.zeros(4), 2),
                "top_p_filtering": lambda: m.top_p_filtering(torch.zeros(4), 0.9),
            }[op]()
        if stage == "p4_generate":
            m = importlib.import_module("infer_lab.p4_generate.generate")

            class _M:
                def __call__(self, idx, past_kv=None, use_cache=False):
                    return torch.zeros(1, 1, 4), None

            return m.generate(_M(), torch.zeros(1, 2, dtype=torch.long), m.GenerationConfig())
        if stage == "p5_toy":
            model = importlib.import_module("infer_lab.p5_toy.model")
            data = importlib.import_module("infer_lab.p5_toy.data")
            if op in ("model_forward", "num_params"):
                cfg = model.TinyGPTConfig(
                    vocab_size=8, d_model=8, n_layer=1, n_head=2, n_kv_head=2, block_size=4
                )
                net = model.TinyGPT(cfg)
                if op == "model_forward":
                    return net.forward(torch.zeros(1, 2, dtype=torch.long))
                return net.num_params()
            if op == "dataset_init":
                return data.CharDataset("ab", 4)
            ds = data.CharDataset.__new__(data.CharDataset)
            return {
                "vocab_size": lambda: ds.vocab_size,
                "train_val_split": lambda: data.train_val_split(torch.arange(10), 0.1),
                "get_batch": lambda: data.get_batch(torch.arange(10), 4, 2, "cpu"),
            }[op]()
        if stage == "p1_framework":
            m = importlib.import_module("infer_lab.p1_framework.engine")
            if op == "init":
                return m.FrameworkEngine()
            if op == "render_chat_prompt":
                return m.render_chat_prompt([{"role": "user", "content": "hi"}])
            eng = m.FrameworkEngine()
            return {
                "load": lambda: eng.load(),
                "complete": lambda: eng.complete("hi"),
                "chat": lambda: eng.chat([{"role": "user", "content": "hi"}]),
            }[op]()
        raise AssertionError(f"未知桩用例 {stage}:{op}")  # pragma: no cover

    return thunk


@pytest.mark.parametrize("case", CASES)
def test_stub_raises_not_implemented_with_hint(case):
    stage, op = case.split(":")
    with pytest.raises(NotImplementedError) as excinfo:
        _thunk(stage, op)()
    assert "待实现" in str(excinfo.value), f"{case} 的桩应提示「待实现」，实际消息: {excinfo.value}"


def test_generation_config_is_complete_dataclass():
    """GenerationConfig 是给定的（非 TODO）：字段与默认值必须齐备。"""
    m = importlib.import_module("infer_lab.p4_generate.generate")
    cfg = m.GenerationConfig()
    names = {f.name for f in fields(cfg)}
    assert {"max_new_tokens", "temperature", "top_k", "top_p", "eos_id", "use_cache"} <= names
    assert cfg.max_new_tokens == 32 and cfg.use_cache is True
