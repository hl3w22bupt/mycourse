"""P2 tokenizer 测试：词表 / 编解码 / 特殊 token / pad / 持久化。"""

from __future__ import annotations

from infer_lab.p2_tokenizer.tokenizer import SPECIAL_TOKENS
from infer_lab.testing import load_impl


def build_tok():
    """构造被测 tokenizer（在测试体内调用，保证骨架态以 xfail 呈现「待实现」）。"""
    mod = load_impl("p2_tokenizer", "tokenizer")
    return mod.CharTokenizer.build("hello world")


def test_special_tokens_fixed_ids():
    assert SPECIAL_TOKENS == {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}


def test_vocab_contains_specials_plus_unique_chars():
    tok = build_tok()
    # "hello world" → 8 个不同字符 + 4 个特殊 token
    assert tok.vocab_size == len(set("hello world")) + 4


def test_encode_decode_roundtrip():
    tok = build_tok()
    text = "hello world"
    ids = tok.encode(text)
    assert ids and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_unknown_char_maps_to_unk():
    tok = build_tok()
    ids = tok.encode("helloΩ")  # Ω 不在词表
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "hello"


def test_add_special_and_max_len():
    tok = build_tok()
    ids = tok.encode("hello world", add_special=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert len(ids) == len("hello world") + 2

    clipped = tok.encode("hello world", add_special=True, max_len=5)
    assert clipped == [tok.bos_id, *tok.encode("hel"), tok.eos_id]


def test_pad_batch_aligns_lengths_and_keeps_original_length():
    tok = build_tok()
    seqs = [tok.encode("hello"), tok.encode("hi")]
    padded, lengths = tok.pad_batch(seqs)
    assert lengths == [5, 2]
    assert all(len(p) == 5 for p in padded)
    assert padded[1] == tok.encode("hi") + [tok.pad_id] * 3


def test_save_load_roundtrip(tmp_path):
    tok = build_tok()
    path = tmp_path / "tok.json"
    tok.save(str(path))
    tok2 = type(tok).load(str(path))
    assert tok2.vocab == tok.vocab
    assert tok2.encode("hello world") == tok.encode("hello world")
