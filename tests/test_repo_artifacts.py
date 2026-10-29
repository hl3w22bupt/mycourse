"""仓库学习产物守护：路线图、对比笔记、各阶段 README、指标记录位必须存在且内容达标。

对应目标验收口径：
- ROADMAP 分阶段（P0..P5）且每阶段有 README（目标/任务/验收命令）；
- 仓库内含「框架部署 vs 手搓引擎」对比笔记（结构、KV cache、延迟路径、控制粒度、推荐顺序、
  批量生成的坑）；
- 指标记录位 docs/metrics/inference-baseline.json：framework 槽位已填（make smoke 写入）、
  handrolled 槽位按同口径回填（scripts/measure_handrolled.py）、顶层 batch 节两侧齐全。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE_DIRS = {
    "P0": "infer_lab/p0_env",
    "P1": "infer_lab/p1_framework",
    "P2": "infer_lab/p2_tokenizer",
    "P2b": "infer_lab/p2_embedding",
    "P3": "infer_lab/p3_attention",
    "P3b": "infer_lab/p3_kv_cache",
    "P4": "infer_lab/p4_sampling",
    "P4b": "infer_lab/p4_generate",
    "P5": "infer_lab/p5_toy",
}


def test_roadmap_exists_and_covers_all_stages():
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    for stage in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        assert stage in text, f"ROADMAP 缺少阶段 {stage}"
    # 勾稽表：路线图阶段 ↔ 代码目录
    for d in STAGE_DIRS.values():
        assert d in text, f"ROADMAP 勾稽表缺少目录 {d}"


def test_every_stage_has_readme_with_goal_tasks_and_commands():
    for stage, d in STAGE_DIRS.items():
        readme = ROOT / d / "README.md"
        assert readme.exists(), f"{stage} 缺少 README: {readme}"
        content = readme.read_text(encoding="utf-8")
        for keyword in ["学习目标", "任务", "验收"]:
            assert keyword in content, f"{stage} README 缺少「{keyword}」小节"


def test_comparison_note_exists_and_covers_required_dimensions():
    note = ROOT / "docs" / "comparison-framework-vs-handrolled.md"
    assert note.exists(), "缺少对比笔记 docs/comparison-framework-vs-handrolled.md"
    text = note.read_text(encoding="utf-8")
    for keyword in ["KV cache", "延迟", "控制粒度", "推荐顺序", "批量生成的坑"]:
        assert keyword in text, f"对比笔记缺少维度「{keyword}」"


def test_readme_learning_entry_exists():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for keyword in ["快速开始", "make test", "ROADMAP.md"]:
        assert keyword in text, f"README 学习入口缺少 {keyword}"


def test_metrics_record_has_framework_side_and_backfilled_handrolled_slot():
    """指标记录位：framework 槽位有真实数字，handrolled 槽位按同口径回填。"""
    path = ROOT / "docs" / "metrics" / "inference-baseline.json"
    assert path.exists(), (
        "缺少指标记录 docs/metrics/inference-baseline.json，"
        "请跑 `make smoke`（冒烟通过后自动写入框架侧指标）"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 2, "handrolled 回填 + batch 节是 schema v2 特性"

    # 对拍协议必须在记录里：两边可比的前提是口径一致
    protocol = record["protocol"]
    assert "prompt" in protocol and "max_tokens" in protocol and "requests" in protocol
    assert "greedy" in protocol["sampling"], "对拍协议必须是 greedy，保证输出可复现"

    fw = record["framework"]
    assert fw["model"] and fw["device"], "framework 槽位缺元信息（model/device）"
    assert isinstance(fw["param_count"], int) and fw["param_count"] > 0
    assert len(fw["requests"]) == protocol["requests"] >= 1
    summary = fw["summary"]
    assert summary["tokens_per_s"] and summary["tokens_per_s"] > 0, "吞吐必须是正数"
    assert summary["latency_ms_p50"] > 0

    # ---- 手搓侧：已回填，且与 framework 槽位同口径 ----
    slot = record["handrolled"]
    assert slot["summary"], "handrolled 槽位应已按 framework 同口径回填"
    assert isinstance(slot["param_count"], int) and slot["param_count"] > 0
    assert slot["model"] and slot["device"] and slot["sample_text"]
    assert len(slot["requests"]) == protocol["requests"], "回填必须沿用对拍协议的请求数"
    for row in slot["requests"]:
        assert row["completion_tokens"] == protocol["max_tokens"], (
            "回填必须沿用对拍协议的 max_tokens（每请求补全 token 数）"
        )
    hs = slot["summary"]
    assert hs["tokens_per_s"] and hs["tokens_per_s"] > 0, "手搓侧吞吐必须是正数"
    assert hs["latency_ms_p50"] > 0
    assert slot["protocol_delta"], "口径差异（in-process / tokenizer 粒度）必须写在回填数据旁"
    assert slot["backfill_script"] == "scripts/measure_handrolled.py", "回填必须可复现（脚本留名）"


def test_metrics_record_has_two_sided_batch_section():
    """顶层 batch 节：B>1 批量对拍两侧齐全，speedup 以 B=1 为基线且 ≥1。"""
    path = ROOT / "docs" / "metrics" / "inference-baseline.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    batch = record.get("batch")
    assert batch, "缺少顶层 batch 节（B>1 批量对拍），请跑 scripts/measure_handrolled.py"
    assert "in-process" in batch["protocol"]["latency_view"], "批量对拍两边都不走 HTTP 才可比"
    sizes = batch["protocol"]["batch_sizes"]
    assert sizes and sizes[0] == 1, "批量对拍必须含 B=1 基线"

    for side_name in ("framework", "handrolled"):
        side = batch[side_name]
        assert side["summary"], f"batch.{side_name} 侧应已测量（scripts/measure_handrolled.py）"
        assert [r["batch_size"] for r in side["rows"]] == sizes, (
            f"batch.{side_name} 测量尺寸必须与协议一致"
        )
        by_b = {r["batch_size"]: r for r in side["summary"]["rows"]}
        speedups = [by_b[b]["speedup_vs_b1"] for b in sizes]
        assert speedups[0] == 1.0
        assert all(s >= 1.0 for s in speedups), (
            f"batch.{side_name} 吞吐随 B 增大不应低于单序列基线：{speedups}"
        )
    # 批量解码的意义：两边整批吞吐都应随 B 显著放大（1.5× 起步）
    for side_name in ("framework", "handrolled"):
        top = batch[side_name]["summary"]["rows"][-1]
        assert top["speedup_vs_b1"] >= 1.5, (
            f"batch.{side_name} 最大 batch 的吞吐增益应 ≥1.5×，实测 {top['speedup_vs_b1']}×"
        )
