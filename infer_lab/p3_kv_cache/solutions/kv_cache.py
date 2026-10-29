"""P3 参考实现：KV cache。

学习要点：
- 为什么需要：无缓存的自回归解码，每生成 1 个 token 都要把整段历史重算一遍，
  复杂度 O(T^2)；KV cache 把每个 token 的 K/V 存下来，每步只算新 token 的
  q/k/v，历史部分直接查表 —— 代价是显存随序列长度线性增长。
- 显存账本（这就是 GQA 存在的理由）：``2 * n_layers * n_kv_head * seq_len * head_dim
  * batch * bytes``。KV 头数减半，账本直接减半。
"""

from __future__ import annotations

import torch


class KVCache:
    """按层保管的 KV 缓存。每层一个 ``(k, v)`` 槽位，推理时逐层 update。"""

    def __init__(self, n_layers: int) -> None:
        if n_layers < 1:
            raise ValueError("n_layers 必须 >= 1")
        self.n_layers = n_layers
        self._slots: list[list[torch.Tensor | None]] = [[None, None] for _ in range(n_layers)]

    # ---------- 读写 ----------
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx={layer_idx} 超出层数 {self.n_layers}")
        slot = self._slots[layer_idx]
        slot[0] = k if slot[0] is None else torch.cat([slot[0], k], dim=-2)
        slot[1] = v if slot[1] is None else torch.cat([slot[1], v], dim=-2)
        return slot[0], slot[1]

    def get(self, layer_idx: int):
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx={layer_idx} 超出层数 {self.n_layers}")
        k, v = self._slots[layer_idx]
        return None if k is None else (k, v)

    # ---------- 状态 ----------
    @property
    def seq_len(self) -> int:
        k = self._slots[0][0]
        return 0 if k is None else k.shape[-2]

    def reset(self) -> None:
        self._slots = [[None, None] for _ in range(self.n_layers)]

    # 上下文管理器：一条生成请求一个作用域，退出即清缓存
    def __enter__(self) -> "KVCache":
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        self.reset()

    def __len__(self) -> int:
        return self.seq_len
