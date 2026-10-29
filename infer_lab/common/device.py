"""跨阶段共用的小工具。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """固定 python / numpy / torch 随机源，保证 demo 与测试可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - 仅在有 CUDA 的机器上触发
        torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    """按 auto > cuda > mps > cpu 的顺序选择设备。

    学习项目默认 auto：有 GPU（CUDA/MPS）就用，否则退回 CPU。
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
