"""pytest 全局配置：双模式测试底座。

- ``INFER_LAB_TARGET=solution``（默认）：``make test``，跑参考实现，应当全绿。
- ``INFER_LAB_TARGET=user``：``make test-user``，导入 TODO 骨架。
  未实现（抛 ``NotImplementedError``）的用例自动标 XFAIL，原因写明「待实现」；
  其余已通过的用例仍然必须通过 —— 这样骨架态与实现态共用同一套测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_lab.testing import is_user_target  # noqa: E402


def _chain_has_not_implemented(exc: BaseException | None) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, NotImplementedError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if call.when != "call" or not is_user_target():
        return
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None and _chain_has_not_implemented(call.excinfo.value):
        rep.outcome = "skipped"
        rep.wasxfail = "待实现（TODO 骨架态）：完成该模块后请改跑 `make test` 验证"
