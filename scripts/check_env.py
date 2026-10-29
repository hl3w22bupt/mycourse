#!/usr/bin/env python3
"""P0 环境自检入口：``make check-env``。

依赖齐备退出码 0；缺核心依赖（torch/numpy）退出码 1 并给出安装提示。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer_lab.testing import load_impl  # noqa: E402


def main() -> int:
    mod = load_impl("p0_env", "env_check")  # solution 态跑参考实现；user 态跑你的 TODO
    report = mod.check_env()
    print(mod.format_report(report))
    if not report.get("ok"):
        print("\n缺少核心依赖，请先执行：make install")
        return 1
    print("\n环境 OK，可以开始 P1-P5 的学习。入口见 README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
