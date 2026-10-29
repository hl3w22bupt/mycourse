"""P0 - 环境与工程基线：环境自检。

TODO(学习者)：实现 ``check_env`` 与 ``format_report``。

要求：
1. ``check_env()`` 返回 dict，至少包含：
   - ``python``：字符串，如 ``"3.13.2"``
   - ``torch`` / ``numpy``：版本字符串；包缺失时为 ``None``
   - ``cuda`` / ``mps``：布尔值（当前机器是否可用）
   - ``framework_deps``：dict，键 ``transformers`` / ``fastapi``，值为版本或 ``None``
   - ``ok``：布尔值，torch 与 numpy 都可用才为 True
2. ``format_report(report)`` 把 dict 渲染成逐行 ``key: value`` 的可读文本。

验收命令：``make check-env``（应打印报告且退出码为 0）。
"""

from __future__ import annotations


def check_env() -> dict:
    """采集当前 Python 环境信息。"""
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 check_env")


def format_report(report: dict) -> str:
    """把 check_env 的结果渲染成多行文本。"""
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 format_report")
