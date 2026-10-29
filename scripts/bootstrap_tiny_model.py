#!/usr/bin/env python3
"""P1 离线自建 tiny 模型：用 transformers 构建一个真 GPT-2 并在玩具语料上快速微调。

为什么需要它：
- P1 的演示模型 `sshleifer/tiny-gpt2` 需要从 HuggingFace Hub 下载；在
  huggingface.co 不可达、镜像也不稳定的网络里，`make smoke` 会卡在模型下载上。
- 本脚本全程离线：`GPT2Config` 从零构建权重（seed 固定、可复现），再用
  transformers 的标准 API 在 `infer_lab.p5_toy.corpus.TOY_CORPUS` 上微调几百步，
  得到一个能生成玩具句子的真 GPT-2，保存到 `models_cache/tiny-gpt2-toy/`。
- 这也顺便演示了「框架训练循环」—— 与 P5 的手搓循环正好构成对照组。

用法：
    make bootstrap    # 幂等：目录已存在则跳过
    .venv/bin/python scripts/bootstrap_tiny_model.py --out-dir models_cache/tiny-gpt2-toy --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "models_cache" / "tiny-gpt2-toy"


def build_tokenizer(corpus: str, out_dir: Path):
    """构建 WordLevel 分词器并保存为 HF 格式。

    关键点：预分词要把标点隔离成独立 token（``Punctuation(isolated)``），
    否则语料里的 ``run.`` 不在词表、整段训练数据变成大量 [UNK]，
    模型学到的就是「下一步输出 UNK」。
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(behavior="isolated")]
    )
    words, seen = ["[PAD]", "[UNK]"], set()
    for w, _ in pre.pre_tokenize_str(corpus):
        if w not in seen:
            seen.add(w)
            words.append(w)
    vocab = {w: i for i, w in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[PAD]",
        eos_token="[PAD]",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out_dir))
    return tok, len(vocab)


def main() -> int:
    p = argparse.ArgumentParser(description="离线自建 tiny-gpt2（transformers 训练循环）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--block-size", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="忽略已有目录，重建")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    marker = out_dir / "config.json"
    if marker.exists() and not args.force:
        print(f"BOOTSTRAP_SKIP 已存在 {out_dir}")
        return 0

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from infer_lab.common.device import pick_device, seed_everything
    from infer_lab.p5_toy.corpus import TOY_CORPUS

    seed_everything(args.seed)
    device = pick_device("auto")

    print(f"BOOTSTRAP_TOKENIZER -> {out_dir}")
    tok, vocab_size = build_tokenizer(TOY_CORPUS, out_dir)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        bos_token_id=0,   # 与 tokenizer 的 [PAD] 对齐，避免 GPT2 默认 50256 越界告警
        eos_token_id=0,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)
    print(f"BOOTSTRAP_MODEL params={sum(p.numel() for p in model.parameters())} device={device}")

    ids = tok(TOY_CORPUS, return_tensors="pt")["input_ids"][0]
    unk_ratio = float((ids == tok.unk_token_id).float().mean())
    print(f"BOOTSTRAP_CORPUS tokens={len(ids)} unk_ratio={unk_ratio:.4f}")
    if unk_ratio > 0.01:
        raise SystemExit("语料 UNK 占比过高，tokenizer 构建有 bug（标点应被隔离成独立 token）")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    generator = torch.Generator().manual_seed(args.seed)
    first = last = None
    for step in range(args.steps + 1):
        starts = torch.randint(0, len(ids) - args.block_size - 1, (args.batch_size,), generator=generator)
        x = torch.stack([ids[s : s + args.block_size] for s in starts]).to(device)
        # 注意：transformers 的 labels 会在内部自动右移一位（logits[t] 对 labels[t+1]），
        # 所以这里 labels 直接给 x，【不要】像手搓循环（P5）那样提前构造 y=x 右移一位 ——
        # 双重位移会让模型悄悄学会「预测后两个 token」。这是框架版与手搓版的经典差异。
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(loss.detach())
        if step % 100 == 0:
            print(f"BOOTSTRAP_STEP {step} loss {float(loss.detach()):.4f}")
        last = float(loss.detach())

    model.eval()
    model.config.save_pretrained(str(out_dir))  # 连同微调后的权重一起落盘
    model.save_pretrained(str(out_dir))
    print(f"BOOTSTRAP_LOSS {first:.4f} -> {last:.4f}")
    print(f"BOOTSTRAP_OK saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
