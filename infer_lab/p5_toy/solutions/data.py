"""P5 参考实现：数据侧（语料 → 训练 batch）。

学习要点：
- 语言建模的监督信号是「自身右移一位」：``x = ids[t:t+B]``，``y = ids[t+1:t+B+1]``。
  不需要人工标注，任何文本流都能变成训练数据 —— 这就是 next-token prediction。
- 随机起点采样（而非顺序切 batch）：小语料上等效于无限数据增广，
  也避免模型按位置背 batch。
"""

from __future__ import annotations

import torch

from ...p2_tokenizer.solutions.tokenizer import CharTokenizer


class CharDataset:
    """整段语料的字符级视图：一次 encode，训练时按窗口随机切。"""

    def __init__(self, text: str, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("block_size 必须 >= 2（至少留出 x 和 y 各一位）")
        self.block_size = block_size
        self.tokenizer = CharTokenizer.build(text)
        self.ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def train_val_split(ids: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < val_frac < 1:
        raise ValueError("val_frac 必须在 (0, 1) 区间")
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


def get_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) < block_size + 1:
        raise ValueError("语料太短，不够切一个 block_size+1 的窗口")
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,), generator=generator)
    windows = torch.stack([ids[s : s + block_size + 1] for s in starts])
    x, y = windows[:, :-1], windows[:, 1:]
    return x.to(device), y.to(device)
