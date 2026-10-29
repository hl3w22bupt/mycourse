"""测试装货助手：决定单测跑「TODO 骨架」还是「solutions 参考实现」。

- ``INFER_LAB_TARGET=solution``（默认）：导入 ``infer_lab/<stage>/solutions/<mod>.py``，
  全量测试应当全绿。
- ``INFER_LAB_TARGET=user``：导入 ``infer_lab/<stage>/<mod>.py``（TODO 骨架）。
  未实现的用例会因 ``NotImplementedError`` 被 conftest 标成 xfail，原因写明「待实现」。
"""

from __future__ import annotations

import importlib
import os

TARGET_ENV = "INFER_LAB_TARGET"
SOLUTION = "solution"
USER = "user"


def current_target() -> str:
    return os.environ.get(TARGET_ENV, SOLUTION).strip().lower()


def is_user_target() -> bool:
    return current_target() == USER


def load_impl(stage: str, module: str):
    """按当前 target 装载模块。

    ``load_impl("p3_attention", "attention")``
    → solution 态返回 ``infer_lab.p3_attention.solutions.attention``
    → user 态返回 ``infer_lab.p3_attention.attention``
    """
    if is_user_target():
        return importlib.import_module(f"infer_lab.{stage}.{module}")
    return importlib.import_module(f"infer_lab.{stage}.solutions.{module}")
