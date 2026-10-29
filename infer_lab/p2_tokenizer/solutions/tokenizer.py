"""P2 参考实现：字符级 tokenizer。

学习要点：
- 真实 GPT 用 BPE/ByteLevelBPE，教学版用「字符 = token」：词表小、可完全看懂，
  保留了「文本 ↔ id 序列」这一推理引擎最前端的契约。
- id 空间设计：``0..3`` 固定给特殊 token，普通 token 从 4 开始 —— 训练/采样时
  需要屏蔽 pad、提前停止需要 eos，都依赖这些固定 id。
"""

from __future__ import annotations

import json

from ..tokenizer import SPECIAL_TOKENS  # 单一事实来源：规格见骨架文件


class CharTokenizer:
    """字符级分词器：词表 = 特殊 token + 语料中出现的全部字符（按首现顺序）。"""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = dict(vocab)
        self._id2tok = {i: t for t, i in self._vocab.items()}
        for tok, idx in SPECIAL_TOKENS.items():
            self._vocab.setdefault(tok, idx)
            self._id2tok.setdefault(idx, tok)

    # ---------- 构造 ----------
    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        vocab = dict(SPECIAL_TOKENS)
        next_id = len(SPECIAL_TOKENS)
        for ch in text:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1
        return cls(vocab)

    # ---------- 基本信息 ----------
    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<pad>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<unk>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<bos>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<eos>"]

    # ---------- 编解码 ----------
    def encode(self, text: str, add_special: bool = False, max_len: int | None = None) -> list[int]:
        ids = [self._vocab.get(ch, self.unk_id) for ch in text]
        if max_len is not None:
            if max_len < 1:
                raise ValueError("max_len 必须 >= 1")
            keep = max(max_len - (2 if add_special else 0), 0)
            ids = ids[:keep]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(SPECIAL_TOKENS.values())
        out = []
        for i in ids:
            if i in special_ids and skip_special:
                continue
            out.append(self._id2tok.get(i, "<unk>"))
        return "".join(out)

    # ---------- 批处理 ----------
    def pad_batch(self, sequences: list[list[int]]) -> tuple[list[list[int]], list[int]]:
        if not sequences:
            return [], []
        longest = max(len(s) for s in sequences)
        padded = [s + [self.pad_id] * (longest - len(s)) for s in sequences]
        lengths = [len(s) for s in sequences]
        return padded, lengths

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls({tok: int(idx) for tok, idx in vocab.items()})

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"CharTokenizer(vocab_size={self.vocab_size})"
