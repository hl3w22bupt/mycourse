"""P3 - KV cache：逐 token 推理的显存账本。

TODO(学习者)：实现 ``KVCache``。

要求：
1. ``__init__(n_layers)``：为每层建一个空槽位（list[(k, v)]，初始全 None）。
2. ``update(layer_idx, k, v)``：把本步新算出的 k、v（形状 ``(B, n_head, T_new, d)``）
   追加到该层缓存，返回**全量** ``(k_all, v_all)``（形状 ``(B, n_head, S_total, d)``），
   直接可喂给注意力打分。
3. ``get(layer_idx)``：读该层 ``(k, v)``，未写入时返回 ``None``。
4. ``seq_len`` 属性：当前缓存长度（第 0 层 key 的时间维；全空时为 0）。
5. ``reset()``：清空所有层 —— 新的一条生成请求开始前必须 reset，否则串请求。
6. 上下文管理器协议：``with kv:`` 等价于进入时 ``reset()``，方便「一条请求一个作用域」。
7. ``__len__``：等价于 ``seq_len``（``len(kv)`` 直接当缓存长度用）。
   ``update``/``get`` 的 ``layer_idx`` 越界时抛 ``IndexError``。

对照：参考实现见 ``solutions/kv_cache.py``。

验收：``pytest tests/test_p3_kv_cache.py`` 全绿（核心性质：带缓存逐 token 解码 ≡ 全量重算）。
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.__init__")

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.update")

    def get(self, layer_idx: int):
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.get")

    @property
    def seq_len(self) -> int:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.seq_len")

    def reset(self) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.reset")

    def __enter__(self) -> "KVCache":
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.__enter__")

    def __exit__(self, *exc) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 KVCache.__exit__")
