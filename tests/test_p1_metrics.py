"""P1 指标记录模块测试：不依赖网络与真模型，用假 complete_fn 验证口径、结构与落盘。"""

from __future__ import annotations

import time

import pytest

from infer_lab.p1_framework import metrics as M


def fake_complete(prompt: str, max_tokens: int) -> dict:
    """桩补全函数：与 /v1/completions 返回口径一致（token 数固定）。"""
    return {"text": " hello world", "prompt_tokens": 4, "completion_tokens": 3}


def fixed_rows() -> list[dict]:
    """三条手工指标行：延迟 100/50/100 ms，补全 token 10/20/30，便于口算校验。"""
    return [
        {"index": 0, "prompt_tokens": 4, "completion_tokens": 10, "latency_ms": 100.0,
         "tokens_per_s": 100.0},
        {"index": 1, "prompt_tokens": 4, "completion_tokens": 20, "latency_ms": 50.0,
         "tokens_per_s": 400.0},
        {"index": 2, "prompt_tokens": 4, "completion_tokens": 30, "latency_ms": 100.0,
         "tokens_per_s": 300.0},
    ]


def test_build_protocol_pins_comparison_contract():
    p = M.build_protocol("the little engine could", 24, 5)
    assert p["prompt"] == "the little engine could"
    assert p["max_tokens"] == 24 and p["requests"] == 5
    # greedy 是两边可比的前提：采样不确定，数字与文本都没法对拍
    assert "greedy" in p["sampling"]
    assert "HTTP" in p["latency_view"]  # 延迟口径：客户端视角（含往返）


def test_measure_request_row_fields_are_consistent():
    row = M.measure_request(fake_complete, "hi", 3, index=0)
    assert row["index"] == 0
    assert row["prompt_tokens"] == 4 and row["completion_tokens"] == 3
    assert row["latency_ms"] >= 0
    # tokens_per_s 必须能由本行字段复算出来（口径自洽）
    if row["latency_ms"] > 0:
        assert row["tokens_per_s"] == round(3 / (row["latency_ms"] / 1000.0), 2)
    else:
        assert row["tokens_per_s"] is None


def test_summarize_math():
    s = M.summarize(fixed_rows())
    assert s["requests"] == 3
    assert s["prompt_tokens"] == 4
    assert s["completion_tokens_per_request"] == 10  # 取首行口径
    assert s["completion_tokens_total"] == 60
    assert s["latency_ms_avg"] == round((100 + 50 + 100) / 3, 2)
    assert s["latency_ms_p50"] == 100.0  # 偶数取中间均值：这里 3 条取中位 100
    assert s["latency_ms_min"] == 50.0 and s["latency_ms_max"] == 100.0
    # 聚合吞吐 = 总补全 token / 总延迟秒 = 60 / 0.25
    assert s["tokens_per_s"] == 240.0


def test_summarize_rejects_empty_rows():
    with pytest.raises(ValueError):
        M.summarize([])


def test_build_record_fills_framework_and_reserves_handrolled_slot():
    rec = M.build_record(
        protocol=M.build_protocol("hi", 8, 3),
        engine_info={
            "model": "models_cache/tiny-gpt2-toy",
            "engine": "transformers",
            "device": "mps",
            "param_count": 123456,
            "library_versions": {"torch": "2.15.0", "transformers": "5.21.1"},
        },
        rows=fixed_rows(),
        sample_text=" hello world",
        now="2026-10-29T00:00:00Z",
    )
    assert rec["schema_version"] == M.METRICS_SCHEMA_VERSION
    assert rec["updated_at"] == "2026-10-29T00:00:00Z"

    fw = rec["framework"]
    assert fw["engine"] == "transformers"
    assert fw["model"] == "models_cache/tiny-gpt2-toy"
    assert fw["param_count"] == 123456
    assert fw["device"] == "mps"
    assert fw["sample_text"] == " hello world"
    assert len(fw["requests"]) == 3
    assert fw["summary"]["requests"] == 3

    # 手搓侧是预留槽位：字段齐、值全空、带回填口径说明
    slot = rec["handrolled"]
    for field in M.SIDE_FIELDS:
        assert field in slot and slot[field] is None, f"handrolled.{field} 应保持预留为空"
    assert "framework.protocol" in slot["todo"] or "同口径" in slot["todo"]


def test_build_record_rejects_empty_rows():
    with pytest.raises(ValueError):
        M.build_record(
            protocol=M.build_protocol("hi", 8, 0), engine_info={}, rows=[], sample_text=""
        )


def test_build_record_reserves_batch_section_with_todo():
    """schema v2：smoke 新写的记录也要有 batch 预留节（回填脚本负责填两侧）。"""
    rec = M.build_record(
        protocol=M.build_protocol("hi", 8, 1),
        engine_info={"model": "m"},
        rows=fixed_rows()[:1],
        sample_text="x",
    )
    batch = rec["batch"]
    assert batch["framework"] is not None and batch["handrolled"] is not None
    for side in (batch["framework"], batch["handrolled"]):
        assert side["summary"] is None and side["todo"], "batch 两侧应保持预留"
    assert batch["framework"] is not batch["handrolled"]


def test_batch_row_and_summary_math():
    """批量指标行口径自洽：tokens_per_s 能由延迟复算；speedup 以 B=1 为基线。"""
    calls = {"n": 0}

    def batch_fn(prompt: str, max_tokens: int, batch_size: int) -> dict:
        calls["n"] += 1
        time.sleep(0.002)  # 让延迟进入可测区间（纯内存桩快到 round 后会是 0.0ms）
        return {"completion_tokens_total": batch_size * max_tokens, "sequences": batch_size}

    row = M.measure_batch_run(batch_fn, "hi", 10, batch_size=4)
    assert row["batch_size"] == 4
    assert row["completion_tokens_total"] == 40
    assert row["latency_ms"] > 0
    assert row["tokens_per_s"] == round(40 / (row["latency_ms"] / 1000.0), 2)

    with pytest.raises(ValueError):
        M.measure_batch_run(
            lambda **kw: {"completion_tokens_total": 0}, "hi", 10, batch_size=1
        )

    rows = [
        {"batch_size": 1, "completion_tokens_total": 10, "latency_ms": 100.0, "tokens_per_s": 100.0},
        {"batch_size": 4, "completion_tokens_total": 40, "latency_ms": 50.0, "tokens_per_s": 800.0},
    ]
    s = M.summarize_batch(rows)
    assert s["batch_sizes"] == [1, 4]
    by_b = {r["batch_size"]: r for r in s["rows"]}
    assert by_b[4]["speedup_vs_b1"] == 8.0  # 800 / 100
    assert by_b[4]["tokens_per_s_per_sequence"] == 200.0  # 800 / 4
    assert s["best_batch_size"] == 4 and s["best_tokens_per_s"] == 800.0

    with pytest.raises(ValueError):
        M.summarize_batch([])
    with pytest.raises(ValueError):
        M.summarize_batch(rows[1:])  # 缺 B=1 基线行


def test_build_batch_protocol_pins_inprocess_view():
    p = M.build_batch_protocol("hi", 8, [1, 4], repeats=3)
    assert p["batch_sizes"] == [1, 4]
    assert "in-process" in p["latency_view"], "批量对拍两边都不走 HTTP，才可比"
    assert "greedy" in p["sampling"]
    assert "最小延迟" in p["latency_aggregation"]


def filled_handrolled_side() -> dict:
    """一份字段齐全、数字健康的手搓侧回填数据（测试夹具）。"""
    return {
        "engine": "infer-lab handrolled (P2-P5 TinyGPT)",
        "model": "runs/toy/ckpt.pt",
        "param_count": 99584,
        "device": "mps",
        "library_versions": {"torch": "2.15.0"},
        "sample_text": " run. the little engine ",
        "requests": fixed_rows(),
        "summary": M.summarize(fixed_rows()),
        "protocol_delta": {"latency_view": "in-process(无 HTTP 往返)"},
        "backfill_script": "scripts/measure_handrolled.py",
    }


def fresh_record() -> dict:
    return M.build_record(
        protocol=M.build_protocol("hi", 8, 3),
        engine_info={"model": "models_cache/tiny-gpt2-toy", "param_count": 105920},
        rows=fixed_rows(),
        sample_text=" hello world",
        now="2026-10-29T00:00:00Z",
    )


def test_backfill_handrolled_fills_slot_and_keeps_framework_intact():
    rec = fresh_record()
    out = M.backfill_handrolled(rec, filled_handrolled_side(), now="2026-10-29T12:00:00Z")
    assert out["schema_version"] == M.METRICS_SCHEMA_VERSION  # v1 记录长出 v2 特性 → 盖版本章
    slot = out["handrolled"]
    for field in M.SIDE_FIELDS:
        assert slot[field] == filled_handrolled_side()[field]
    assert slot["backfilled_at"] == "2026-10-29T12:00:00Z"
    assert slot["protocol_delta"]["latency_view"].startswith("in-process")
    # framework 槽位原样保留（回填脚本无权改动框架侧基线）
    assert out["framework"] == rec["framework"]
    # 原记录不被原地修改
    assert rec["handrolled"]["summary"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(engine=None),
        lambda s: s.update(requests=None),
        lambda s: s.update(summary=None),
        lambda s: s.update(param_count=0),
        lambda s: s["summary"].update(tokens_per_s=0),
    ],
)
def test_backfill_handrolled_rejects_unhealthy_data(mutate):
    side = filled_handrolled_side()
    mutate(side)
    with pytest.raises(ValueError):
        M.backfill_handrolled(fresh_record(), side)


def test_write_record_preserves_backfill_on_smoke_rewrite(tmp_path):
    """make smoke 重写记录时不得冲掉已回填的 handrolled / batch 数据。"""
    path = tmp_path / "inference-baseline.json"
    backfilled = M.backfill_handrolled(fresh_record(), filled_handrolled_side())
    backfilled = M.attach_batch_section(
        backfilled,
        M.build_batch_section(
            protocol=M.build_batch_protocol("hi", 8, [1]),
            framework_side={"engine": "transformers", "model": "m", "batch_sizes": [1],
                            "rows": [{"batch_size": 1, "latency_ms": 1.0,
                                      "completion_tokens_total": 8, "tokens_per_s": 8000.0}],
                            "summary": {"best_tokens_per_s": 1.0}},
            handrolled_side=None,
        ),
    )
    M.write_record(backfilled, path)

    smoke_new = fresh_record()  # 新一次 smoke：framework 刷新、handrolled/batch 全空
    smoke_new["framework"]["summary"]["tokens_per_s"] = 999.0
    M.write_record(smoke_new, path)

    reloaded = M.load_record(path)
    assert reloaded["handrolled"] == backfilled["handrolled"], "回填数据不能被 smoke 冲掉"
    assert reloaded["batch"]["framework"]["summary"] == {"best_tokens_per_s": 1.0}, (
        "batch 节已回填侧也不能被冲掉"
    )
    assert reloaded["framework"]["summary"]["tokens_per_s"] == 999.0, "框架侧基线正常刷新"
    assert reloaded["schema_version"] == M.METRICS_SCHEMA_VERSION


def test_write_record_can_skip_preserve(tmp_path):
    path = tmp_path / "inference-baseline.json"
    M.write_record(M.backfill_handrolled(fresh_record(), filled_handrolled_side()), path)
    M.write_record(fresh_record(), path, preserve_backfill=False)
    assert M.load_record(path)["handrolled"]["summary"] is None


def test_record_round_trip(tmp_path):
    rec = M.build_record(
        protocol=M.build_protocol("hi", 8, 1),
        engine_info={"model": "m"},
        rows=fixed_rows()[:1],
        sample_text="x",
        now="2026-10-29T00:00:00Z",
    )
    path = tmp_path / "nested" / "inference-baseline.json"  # 父目录不存在也要能落盘
    assert M.write_record(rec, path) == path
    assert M.load_record(path) == rec
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_default_record_path_points_into_repo_docs():
    path = M.default_record_path()
    assert path.is_absolute()
    assert path.parts[-3:] == ("docs", "metrics", "inference-baseline.json")
