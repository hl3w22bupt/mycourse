"""仓库学习产物守护：路线图、对比笔记、各阶段 README 必须存在且内容达标。

对应目标验收口径：
- ROADMAP 分阶段（P0..P5）且每阶段有 README（目标/任务/验收命令）；
- 仓库内含「框架部署 vs 手搓引擎」对比笔记（结构、KV cache、延迟路径、控制粒度、推荐顺序）。
"""

from __future__ import annotations

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
    for keyword in ["KV cache", "延迟", "控制粒度", "推荐顺序"]:
        assert keyword in text, f"对比笔记缺少维度「{keyword}」"


def test_readme_learning_entry_exists():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for keyword in ["快速开始", "make test", "ROADMAP.md"]:
        assert keyword in text, f"README 学习入口缺少 {keyword}"
