"""P0 参考实现：环境自检。

学习要点：
- 探测依赖用 ``importlib.util.find_spec``（不真正 import，代价小）；
- 版本号用 ``importlib.metadata.version`` 读取，比 ``pkg_resources`` 轻；
- torch 的硬件能力在 macOS 上看 ``torch.backends.mps``，NVIDIA 机器看 ``torch.cuda``。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
from typing import Any

CORE_DEPS = ("torch", "numpy")
FRAMEWORK_DEPS = ("transformers", "fastapi")


def _version_of(dist_name: str) -> str | None:
    """已安装返回版本号，否则 None（不抛异常，方便报表输出）。"""
    if importlib.util.find_spec(dist_name) is None:
        return None
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return None


def check_env() -> dict:
    """采集当前 Python 环境信息，供 ``make check-env`` 打印。"""
    import torch  # 核心依赖，直接 import 拿能力位

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": _version_of("torch"),
        "numpy": _version_of("numpy"),
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()),
        "framework_deps": {name: _version_of(name) for name in FRAMEWORK_DEPS},
    }
    report["ok"] = report["torch"] is not None and report["numpy"] is not None
    return report


def format_report(report: dict) -> str:
    """渲染成 ``key: value`` 多行文本，方便人读、也方便 CI 里 cat 出来。"""
    lines = [
        f"python: {report.get('python')}",
        f"torch: {report.get('torch')}",
        f"numpy: {report.get('numpy')}",
        f"cuda: {report.get('cuda')}",
        f"mps: {report.get('mps')}",
    ]
    for name, ver in (report.get("framework_deps") or {}).items():
        lines.append(f"{name}: {ver}")
    lines.append(f"ok: {report.get('ok')}")
    return "\n".join(lines)
