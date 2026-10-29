"""P5 - 数据侧：把玩具语料变成训练 batch。

TODO(学习者)：实现 ``CharDataset`` 与 ``get_batch``。

要求：
1. ``CharDataset(text, block_size)``
   - 用 P2 的 ``CharTokenizer.build(text)`` 建词表；
   - ``self.ids = tokenizer.encode(text)``（整段语料的 id 序列，1-D ``torch.long``）；
   - 暴露 ``tokenizer``、``vocab_size`` 属性（``train.py`` 还会读 ``ids`` 与
     ``block_size``，照参考实现一并存成属性即可）。
2. ``train_val_split(ids, val_frac=0.1)``：按比例切出 train/val 两个 1-D ``torch.long`` 张量
   （``val_frac`` 不在 (0,1) 抛 ``ValueError``；val 至少 1 个 token）。
3. ``get_batch(ids_tensor, block_size, batch_size, device, generator=None)``
   - 随机采 batch_size 个起点，各取长度 ``block_size + 1`` 的窗口；
   - ``x, y = window[:, :-1], window[:, 1:]``（预测下一位），搬到 ``device``；
   - 返回 ``(x, y)``（语料切不满一个窗口时抛 ``ValueError``）。

对照：参考实现见 ``solutions/data.py``（注意它 import 的是 ``solutions`` 的 tokenizer，
你自己的实现应 import ``p2_tokenizer.tokenizer.CharTokenizer``）。

验收：``pytest tests/test_p5_train_generate.py`` 全绿。
"""

from __future__ import annotations

import torch

from ..p2_tokenizer.tokenizer import CharTokenizer


class CharDataset:
    def __init__(self, text: str, block_size: int) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CharDataset.__init__")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CharDataset.vocab_size")


def train_val_split(ids: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 train_val_split")


def get_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 get_batch")
