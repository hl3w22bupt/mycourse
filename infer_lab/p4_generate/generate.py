"""P4 - 自回归生成循环（把 P2-P3 的零件串成一条解码流水线）。

TODO(学习者)：实现 ``GenerationConfig`` 与 ``generate``。

约定模型接口（duck typing，TinyGPT 与测试桩都按这个约定实现）：
    res = model(idx, past_kv=past_kv, use_cache=use_cache)
    logits, past_kv = res[0], res[-1]     # 兼容 (logits, kv) 与 (logits, loss, kv)
    - 首步：idx 为完整 prompt (B, T0)，past_kv=None
    - 后续：idx 为上一步采出的 1 个 token (B, 1)，past_kv 接续上一步

``GenerationConfig`` 字段（dataclass）：
    max_new_tokens=32, temperature=1.0, top_k=None, top_p=None,
    eos_id=None, use_cache=True

``generate(model, input_ids, config) -> torch.Tensor``：
1. 校验 ``input_ids`` 是 2 维 ``(B, T)``、``max_new_tokens >= 1``；
2. 循环 ``max_new_tokens`` 次：取最后一步 logits → ``sample_token`` 采样 →
   拼回序列；全部 batch 命中 ``eos_id`` 时提前停止；
3. 返回完整序列 ``(B, T0 + 实际生成数)``（不包含 eos 之后的填充）。

验收：``pytest tests/test_p4_generate.py`` 全绿
（核心性质：use_cache=True 与 False 逐步对拍一致；贪心模式 ≡ 手写 argmax 循环）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_id: int | None = None
    use_cache: bool = True


def generate(model, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 generate")
