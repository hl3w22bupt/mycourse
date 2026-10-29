"""P5 闭环测试：小数据训练 loss 下降 → 生成非空文本 → checkpoint 复用。

用极小配置（CPU 上秒级）验证「训练-生成」整条链路真的能跑通。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

import infer_lab.p5_toy.train as train_script
from infer_lab.testing import load_impl

ROOT = Path(__file__).resolve().parents[1]


def _tiny_kwargs(out_dir):
    return dict(
        steps=60,
        d_model=32,
        n_layer=1,
        n_head=2,
        block_size=32,
        batch_size=8,
        device_pref="cpu",
        log_every=30,
        out_dir=str(out_dir),
        log=lambda *_: None,  # 测试静默
    )


def test_training_loss_decreases_and_sample_nonempty(tmp_path):
    # train.py 本身是工作脚本，内部经 load_impl 装载 model/data（user 态会抛待实现）
    result = train_script.run_training(**_tiny_kwargs(tmp_path))

    assert result["improved"] is True
    assert result["final"] < result["initial"] - 0.3, (
        f"loss 应明显下降：initial={result['initial']:.3f} final={result['final']:.3f}"
    )
    assert Path(result["checkpoint"]).exists()
    assert isinstance(result["sample_text"], str) and len(result["sample_text"]) > 10
    assert result["num_params"] > 0


def test_get_batch_windows_shift_by_one():
    data = load_impl("p5_toy", "data")
    ids = torch.arange(100, dtype=torch.long)
    x, y = data.get_batch(ids, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:]), "y 必须是 x 右移一位（next-token prediction）"


def test_train_val_split_frac():
    data = load_impl("p5_toy", "data")
    train_ids, val_ids = data.train_val_split(torch.arange(1000, dtype=torch.long), val_frac=0.1)
    assert len(train_ids) == 900 and len(val_ids) == 100
    assert torch.equal(train_ids, torch.arange(900, dtype=torch.long))


def test_checkpoint_roundtrip_and_generate_cli(tmp_path):
    result = train_script.run_training(**_tiny_kwargs(tmp_path))
    ckpt = result["checkpoint"]

    model, tokenizer = train_script.load_checkpoint(ckpt, torch.device("cpu"))
    assert model is not None and tokenizer.vocab_size > 4

    # CLI 生成 demo（DoD：make generate-demo 可复现）
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "infer_lab.p5_toy.generate",
            "--ckpt",
            ckpt,
            "--prompt",
            "the little engine",
            "--max-new-tokens",
            "40",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"generate CLI 失败:\n{r.stdout}\n{r.stderr}"
    assert "GENERATION " in r.stdout
    sample = r.stdout.split("GENERATION ", 1)[1].strip()
    assert len(sample) >= 10, f"生成文本过短: {sample!r}"
