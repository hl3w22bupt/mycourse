"""P1 指标记录模块测试：不依赖网络与真模型，用假 complete_fn 验证口径、结构与落盘。"""

from __future__ import annotations

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
