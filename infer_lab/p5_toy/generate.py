"""P5 - 生成 CLI：加载 checkpoint，用 P4 的生成循环产出文本。

用法：
    make generate-demo
    .venv/bin/python -m infer_lab.p5_toy.generate --ckpt runs/toy/ckpt.pt \
        --prompt "the little engine" --max-new-tokens 80
"""

from __future__ import annotations

import argparse

import torch

from ..common.device import pick_device
from ..p4_generate.solutions.generate import GenerationConfig, generate as autoregress
from .train import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="infer-lab P5 生成 demo")
    p.add_argument("--ckpt", default="runs/toy/ckpt.pt")
    p.add_argument("--prompt", default="the little engine")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load_checkpoint(args.ckpt, device)

    ids = tokenizer.encode(args.prompt)
    room = model.block_size - len(ids)
    if room < 1:
        raise SystemExit(
            f"prompt 已有 {len(ids)} 个 token，超过 block_size={model.block_size}，请缩短 prompt"
        )
    cfg = GenerationConfig(
        max_new_tokens=min(args.max_new_tokens, room),
        temperature=args.temperature,
        top_k=args.top_k,
    )
    with torch.no_grad():
        out = autoregress(model, torch.tensor([ids], dtype=torch.long, device=device), cfg)
    text = tokenizer.decode(out[0].tolist())
    print(f"PROMPT {args.prompt}")
    print(f"GENERATION {text}")


if __name__ == "__main__":
    main()
