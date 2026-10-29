"""P0 环境自检测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from infer_lab.testing import load_impl

ROOT = Path(__file__).resolve().parents[1]


def test_check_env_returns_full_report():
    mod = load_impl("p0_env", "env_check")
    report = mod.check_env()
    for key in ("python", "torch", "numpy", "cuda", "mps", "framework_deps", "ok"):
        assert key in report, f"报告缺少字段 {key}"
    assert isinstance(report["cuda"], bool) and isinstance(report["mps"], bool)


def test_check_env_core_deps_present_in_test_env():
    mod = load_impl("p0_env", "env_check")
    report = mod.check_env()
    assert report["ok"] is True, "测试环境应当装好 torch/numpy"
    assert report["torch"] and report["numpy"]


def test_format_report_readable():
    mod = load_impl("p0_env", "env_check")
    text = mod.format_report(mod.check_env())
    assert "python:" in text and "torch:" in text
    assert text.count("\n") >= 5


def test_check_env_cli_exit_zero():
    """DoD 验收命令：`make check-env`（solution 态）退出码为 0。"""
    env = {**os.environ, "INFER_LAB_TARGET": "solution"}  # 子进程锁 solution 态
    r = subprocess.run(
        [sys.executable, "scripts/check_env.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert r.returncode == 0, f"check-env 失败:\n{r.stdout}\n{r.stderr}"
    assert "python:" in r.stdout
