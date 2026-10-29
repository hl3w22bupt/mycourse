"""P5 - 训练循环 CLI（可直接运行的工作代码，模型/数据来自当前 target）。

用法：
    make train-demo                      # solution 参考实现，秒级出 loss 下降
    INFER_LAB_TARGET=user make train-demo  # 用你自己的实现跑（未完成会抛 NotImplementedError）

输出约定（方便冒烟与日志归档）：
    STEP <i> loss <x.xxxx>            每隔 log_every 步一行
    LOSS_INITIAL <x.xxxx> / LOSS_FINAL <x.xxxx>
    SAMPLE_TEXT <生成的一段文本>
    CHECKPOINT <路径>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..common.device import pick_device, seed_everything
from ..testing import load_impl
from .corpus import TOY_CORPUS


def run_training(
    steps: int = 200,
    d_model: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    block_size: int = 64,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int = 42,
    device_pref: str = "auto",
    log_every: int = 40,
    out_dir: str | Path = "runs/toy",
    log=print,
) -> dict:
    """跑一次完整的小数据训练，返回含 loss 曲线与产物路径的 dict。

    独立成函数（而非只在 main 里）是为了让 pytest 能用更小的配置秒级验证
    「loss 确实下降」。
    """
    seed_everything(seed)
    device = pick_device(device_pref)

    data_mod = load_impl("p5_toy", "data")
    model_mod = load_impl("p5_toy", "model")

    ds = data_mod.CharDataset(TOY_CORPUS, block_size)
    train_ids, val_ids = data_mod.train_val_split(ds.ids, val_frac=0.1)

    config = model_mod.TinyGPTConfig(
        vocab_size=ds.vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )
    model = model_mod.TinyGPT(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: list[float] = []
    val_losses: list[float] = []
    for step in range(steps + 1):
        if step % 20 == 0:  # 周期性看一眼验证集，观察是否过拟合
            model.eval()
            with torch.no_grad():
                vx, vy = data_mod.get_batch(val_ids, block_size, batch_size, device)
                _, vloss, _ = model(vx, targets=vy)
            val_losses.append(float(vloss))
            model.train()
        x, y = data_mod.get_batch(train_ids, block_size, batch_size, device)
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0:
            log(f"STEP {step} loss {loss:.4f} val {val_losses[-1]:.4f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocab": ds.tokenizer.vocab,
            "losses": losses,
            "val_losses": val_losses,
        },
        ckpt_path,
    )
    ds.tokenizer.save(str(out_dir / "tokenizer.json"))

    improved = losses[-1] < losses[0]
    log(f"LOSS_INITIAL {losses[0]:.4f}")
    log(f"LOSS_FINAL {losses[-1]:.4f}")
    log(f"LOSS_IMPROVED {'yes' if improved else 'no'}")
    log(f"CHECKPOINT {ckpt_path}")

    # 训练完当场生成一小段，闭环「训练 → 生成」
    sample = generate_from_model(model, ds.tokenizer, "the little engine", 60, device)
    log(f"SAMPLE_TEXT {sample}")

    return {
        "losses": losses,
        "val_losses": val_losses,
        "initial": losses[0],
        "final": losses[-1],
        "improved": improved,
        "checkpoint": str(ckpt_path),
        "sample_text": sample,
        "num_params": model.num_params(),
        "device": str(device),
    }


def generate_from_model(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    """用 P4 的生成循环产出文本（TinyGPT 返回 3 元组，generate 兼容）。

    注意：可学习位置嵌入决定了模型上限 ``model.block_size``，
    生成总长（prompt+新 token）不能超过它 —— 这里自动 clamp。
    """
    from ..p4_generate.solutions.generate import GenerationConfig, generate as autoregress

    ids = tokenizer.encode(prompt)
    room = model.block_size - len(ids)
    if room < 1:
        raise ValueError(
            f"prompt 已有 {len(ids)} 个 token，超过 block_size={model.block_size}，请缩短 prompt"
        )
    cfg = GenerationConfig(
        max_new_tokens=min(max_new_tokens, room), temperature=0.8, top_k=20
    )
    out = autoregress(model.eval(), torch.tensor([ids], dtype=torch.long, device=device), cfg)
    return tokenizer.decode(out[0].tolist())


def load_checkpoint(path: str | Path, device) -> tuple:
    """载入 checkpoint，返回 (model, tokenizer)。供 generate CLI 与测试复用。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    model_mod = load_impl("p5_toy", "model")
    config = model_mod.TinyGPTConfig(**payload["config"])
    model = model_mod.TinyGPT(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    tokenizer_mod = load_impl("p2_tokenizer", "tokenizer")
    tokenizer = tokenizer_mod.CharTokenizer(payload["vocab"])
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="infer-lab P5 玩具训练 demo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--out-dir", default="runs/toy")
    args = p.parse_args()

    run_training(
        steps=args.steps,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_pref=args.device,
        log_every=args.log_every,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
