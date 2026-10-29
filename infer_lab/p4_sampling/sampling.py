"""P4 - 采样策略：temperature / top-k / top-p（nucleus）。

TODO(学习者)：实现以下函数。全部只动最后一维 logits，输入输出形状一致。

1. ``softmax_with_temperature(logits, temperature)``
   - ``temperature <= 0`` 抛 ``ValueError``；
   - 除以 temperature 后做数值稳定的 softmax（减 max）。

2. ``top_k_filtering(logits, top_k)``
   - ``top_k`` 为 None 或 ``<= 0`` 表示不过滤，原样返回；
   - 只保留分数最大的 top_k 个位置，其余置 ``-inf``；
   - ``top_k >= 词表大小`` 等价于不过滤。

3. ``top_p_filtering(logits, top_p)``
   - ``top_p`` 为 None 或 ``>= 1.0`` 表示不过滤；
   - 按概率从大到小排序，累计概率首次覆盖 top_p 为止（含当前位），
     其余位置置 ``-inf``；
   - ``0 < top_p <= 1``，越界抛 ``ValueError``。

4. ``sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None)``
   - 依次做 temperature 缩放 → top-k → top-p 过滤 → softmax → 按概率采 1 个 id；
   - 返回 python ``int``（最后一个维度上的选择）。

对照：参考实现见 ``solutions/sampling.py``；过滤用 ``-inf``（不是 0），
必须发生在 softmax 之前。

验收：``pytest tests/test_p4_sampling.py`` 全绿。
"""

from __future__ import annotations

import torch


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 softmax_with_temperature")


def top_k_filtering(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 top_k_filtering")


def top_p_filtering(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 top_p_filtering")


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 sample_token")
