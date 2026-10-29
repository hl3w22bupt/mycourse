"""solutions ↔ 骨架 TODO 对齐守护：「照做」与「对照」不漂移的自动化保证。

每个 TODO 模块都是一对文件——学习者照着写的骨架 ``infer_lab/<stage>/<mod>.py``
和用来对照的参考实现 ``infer_lab/<stage>/solutions/<mod>.py``。本测试把「对齐」
变成可判定口径（与 docs/solutions-guide.md 的对照清单一一对应）：

1. 成对存在：每个骨架文件都有 solution 文件（反之亦然），且各有专属单测文件；
2. 接口一致：solution 必须暴露骨架声明的全部公开符号，且**签名逐参数一致**
   （参数名 / 顺序 / 种类 / 默认值）——否则学习者按骨架写出的实现过不了
   同一套测试，对照也失去意义；solution 可以多出骨架没写的辅助符号；
3. solution 干净：参考实现里不允许残留 ``NotImplementedError`` 桩；
4. 胶水文件定位清晰：``metrics.py``/``server.py``/``train.py`` 等直接可用的
   胶水**有意不设 solution**（它们经 ``load_impl`` 按 target 装载 TODO 模块），
   这里守护它们不被误改成 TODO 桩。
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (stage, module)：骨架 infer_lab/<stage>/<module>.py ↔ solutions/<module>.py 成对
MODULE_PAIRS = {
    "p0_env": ("env_check",),
    "p1_framework": ("engine",),
    "p2_tokenizer": ("tokenizer",),
    "p2_embedding": ("embedding",),
    "p3_attention": ("attention",),
    "p3_kv_cache": ("kv_cache",),
    "p4_sampling": ("sampling",),
    "p4_generate": ("generate",),
    "p5_toy": ("data", "model"),
}

# 直接可用的胶水文件：按 target 装载 TODO 模块，故意不设 solution（不是遗漏）
GLUE_FILES = (
    "p1_framework/metrics.py",
    "p1_framework/server.py",
    "p5_toy/corpus.py",
    "p5_toy/train.py",
    "p5_toy/generate.py",
)

# 每个模块的专属单测文件（骨架态与 solution 态共用同一套）
TEST_FILES = {
    "p0_env": "test_p0_env.py",
    "p1_framework": "test_p1_framework.py",
    "p2_tokenizer": "test_p2_tokenizer.py",
    "p2_embedding": "test_p2_embedding.py",
    "p3_attention": "test_p3_attention.py",
    "p3_kv_cache": "test_p3_kv_cache.py",
    "p4_sampling": "test_p4_sampling.py",
    "p4_generate": "test_p4_generate.py",
    "p5_toy": "test_p5_train_generate.py",
}


def _load(stage: str, module: str, *, solution: bool):
    path = (
        f"infer_lab.{stage}.solutions.{module}" if solution else f"infer_lab.{stage}.{module}"
    )
    return importlib.import_module(path)


def _is_public_symbol(obj) -> bool:
    """骨架里「学习者要实现的东西」：函数 / 类 / 规格常量，排除 import 进来的三方件。

    - 带 ``infer_lab`` ``__module__`` 的一律算（含跨模块 re-export 的 dataclass）；
    - 不带 ``__module__`` 的纯数据常量（如 ``SPECIAL_TOKENS`` 这类规格字典）也算，
      模块 / 可调用对象（``torch``、``Path`` 等 import 件）不算。
    """
    if inspect.ismodule(obj):
        return False
    module_of = getattr(obj, "__module__", "")
    if module_of.startswith("infer_lab"):
        return True
    return not callable(obj)


def _drop_self(sig: inspect.Signature) -> inspect.Signature:
    params = [p for name, p in sig.parameters.items() if name not in ("self", "cls")]
    return inspect.Signature(params, return_annotation=sig.return_annotation)


def _signature_of(owner, name: str) -> inspect.Signature:
    attr = inspect.getattr_static(owner, name)
    if isinstance(attr, property):
        return _drop_self(inspect.signature(attr.fget))
    return _drop_self(inspect.signature(attr))


def _assert_signatures_match(where: str, skel_sig: inspect.Signature, sol_sig: inspect.Signature):
    skel = list(skel_sig.parameters.values())
    sol = list(sol_sig.parameters.values())
    assert [p.name for p in skel] == [p.name for p in sol], (
        f"{where}: 参数名/顺序不一致 —— 骨架 {[p.name for p in skel]} vs "
        f"参考实现 {[p.name for p in sol]}（学习者照骨架写会跟对照实现对不上）"
    )
    for p_sk, p_so in zip(skel, sol):
        assert p_sk.kind == p_so.kind, f"{where}: 参数 {p_sk.name} 种类不一致"
        if p_sk.default is inspect.Parameter.empty:
            assert p_so.default is inspect.Parameter.empty, (
                f"{where}: 参数 {p_sk.name} 骨架无默认值，参考实现不该加默认值"
            )
        else:
            assert p_so.default == p_sk.default, (
                f"{where}: 参数 {p_sk.name} 默认值不一致 —— 骨架 {p_sk.default!r} vs "
                f"参考实现 {p_so.default!r}"
            )


def _iter_skeleton_api(mod):
    """产出 (名字, 对象, 说明) 需要对齐的骨架公开 API。"""
    for name, obj in sorted(vars(mod).items()):
        if name.startswith("_") or name == "annotations" or not _is_public_symbol(obj):
            continue  # annotations 是 __future__ 导入的哨兵，不是学习契约
        if inspect.isfunction(obj):
            yield name, obj, "function"
        elif inspect.isclass(obj):
            yield name, obj, "class"
        else:
            yield name, obj, "constant"


def _iter_class_api(cls):
    """类内声明需要对齐的公开成员（方法 / property），按定义顺序稳定输出。"""
    for name, obj in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(obj, property):
            yield name, "property"
        elif inspect.isfunction(obj):
            yield name, "method"


@pytest.mark.parametrize(
    "stage,module", [(s, m) for s, mods in sorted(MODULE_PAIRS.items()) for m in mods]
)
def test_solution_file_exists_alongside_skeleton(stage: str, module: str):
    skeleton = ROOT / "infer_lab" / stage / f"{module}.py"
    solution = ROOT / "infer_lab" / stage / "solutions" / f"{module}.py"
    assert skeleton.exists(), f"缺骨架文件 {skeleton}"
    assert solution.exists(), f"缺参考实现 {solution}（学习者无处对照）"


@pytest.mark.parametrize(
    "stage", sorted(MODULE_PAIRS)
)
def test_each_module_has_dedicated_test_file(stage: str):
    test_path = ROOT / "tests" / TEST_FILES[stage]
    assert test_path.exists(), f"{stage} 缺专属单测 {test_path}"


@pytest.mark.parametrize(
    "stage,module", [(s, m) for s, mods in sorted(MODULE_PAIRS.items()) for m in mods]
)
def test_solution_covers_skeleton_api_with_matching_signatures(stage: str, module: str):
    skel = _load(stage, module, solution=False)
    sol = _load(stage, module, solution=True)

    for name, obj, kind in _iter_skeleton_api(skel):
        assert hasattr(sol, name), (
            f"infer_lab/{stage}/solutions/{module}.py 缺少骨架声明的 {name} "
            f"（学习者照骨架实现后无处对照）"
        )
        mirror = getattr(sol, name)
        if kind == "function":
            _assert_signatures_match(
                f"solutions/{module}.{name}", inspect.signature(obj), inspect.signature(mirror)
            )
        elif kind == "class":
            # 类本身：__init__ 签名一致（dataclass 复用骨架定义时为同一对象，天然一致）
            if mirror is not obj:
                _assert_signatures_match(
                    f"solutions/{module}.{name}.__init__",
                    _signature_of(obj, "__init__"),
                    _signature_of(mirror, "__init__"),
                )
            # 类成员：骨架声明的每个公开方法 / property 都要对齐
            for member, m_kind in _iter_class_api(obj):
                assert hasattr(mirror, member), (
                    f"solutions/{module}.{name} 缺少骨架声明的方法 {member}"
                )
                if mirror is not obj:
                    _assert_signatures_match(
                        f"solutions/{module}.{name}.{member}",
                        _signature_of(obj, member),
                        _signature_of(mirror, member),
                    )
        else:  # constant：值一致（solution 通常直接 import 骨架的规格常量）
            sol_obj = getattr(sol, name)
            assert sol_obj == obj, (
                f"solutions/{module}.{name} 与骨架常量值不一致：{sol_obj!r} != {obj!r}"
            )


@pytest.mark.parametrize(
    "stage,module", [(s, m) for s, mods in sorted(MODULE_PAIRS.items()) for m in mods]
)
def test_solution_has_no_todo_stub_and_carries_learning_docstring(stage: str, module: str):
    path = ROOT / "infer_lab" / stage / "solutions" / f"{module}.py"
    source = path.read_text(encoding="utf-8")
    assert "raise NotImplementedError" not in source, (
        f"{path} 是参考实现，不允许残留待实现桩（骨架态守护见 test_skeleton_todos.py）"
    )
    doc = importlib.import_module(f"infer_lab.{stage}.solutions.{module}").__doc__ or ""
    assert "参考实现" in doc, f"{path} 的模块 docstring 应说明「参考实现」身份与学习要点"


@pytest.mark.parametrize("relpath", GLUE_FILES)
def test_glue_files_stay_glue_not_todo(relpath: str):
    """胶水文件必须保持「直接可用」：不写 TODO 桩，也不设 solutions 副本。"""
    source = (ROOT / "infer_lab" / relpath).read_text(encoding="utf-8")
    assert "raise NotImplementedError" not in source, (
        f"infer_lab/{relpath} 是胶水代码，不允许出现待实现桩"
    )
    assert "TODO(学习者" not in source, f"infer_lab/{relpath} 是胶水代码，不该布置 TODO"
    solution_counterpart = ROOT / "infer_lab" / relpath.replace(
        "/", "/solutions/"
    ) if "/solutions/" not in relpath else None
    assert solution_counterpart is None or not solution_counterpart.exists(), (
        f"infer_lab/{relpath} 是胶水代码，不应有 solutions 副本（按 target 装载即可）"
    )
