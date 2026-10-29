"""P2 - tokenizer：字符级分词器（对标 GPT-2 BPE 的最小入门版）。

TODO(学习者)：按下面的接口要求实现 ``CharTokenizer``。

要求：
1. 类常量/特殊 token：``SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}``，
   普通字符的 id 从 ``len(SPECIAL_TOKENS)`` 开始，按文本中首次出现顺序分配。
2. ``build(text)``：类方法，扫一遍文本建词表并返回实例。
3. ``encode(text, add_special=False, max_len=None)``：文本 → id 列表；
   未登录字符映射到 ``<unk>``；``add_special=True`` 时首尾加 ``<bos>``/``<eos>``；
   ``max_len`` 截断（截断后仍按 add_special 规则包 special token）。
4. ``decode(ids, skip_special=True)``：id 列表 → 文本；
   ``skip_special=True`` 时跳过 special token，否则用其字面量占位。
5. ``pad_batch(sequences)``：把若干 id 序列 pad 到同长，返回 ``(padded: list[list[int]], lengths: list[int])``。
6. ``vocab_size`` 属性；``save(path)`` / ``load(path)`` 用 JSON 存取词表。

验收：``pytest tests/test_p2_tokenizer.py`` 全绿。
"""

from __future__ import annotations


# 特殊 token 的固定 id（这是规格约定，不是 TODO）：0..3 预留，普通字符从 4 开始编号
SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


class CharTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CharTokenizer.__init__")

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 CharTokenizer.build")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 vocab_size")

    @property
    def pad_id(self) -> int:
        raise NotImplementedError("待实现：pad_id")

    @property
    def unk_id(self) -> int:
        raise NotImplementedError("待实现：unk_id")

    @property
    def bos_id(self) -> int:
        raise NotImplementedError("待实现：bos_id")

    @property
    def eos_id(self) -> int:
        raise NotImplementedError("待实现：eos_id")

    def encode(self, text: str, add_special: bool = False, max_len: int | None = None) -> list[int]:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 encode")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 decode")

    def pad_batch(self, sequences: list[list[int]]) -> tuple[list[list[int]], list[int]]:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 pad_batch")

    def save(self, path: str) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 save")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 load")
