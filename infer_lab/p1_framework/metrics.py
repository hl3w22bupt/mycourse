"""P1 框架侧推理指标记录（胶水模块，直接可用，不是 TODO）。

目的：给「框架版 vs 手搓版」对拍留下**同一口径**的数字，而不是各跑各的。

- ``make smoke`` 冒烟断言通过后，用固定的对拍协议（greedy 采样、同 prompt、
  同 max_tokens、同请求数）再打 N 次补全请求，把延迟 / 吞吐 / token 用量
  写进 ``docs/metrics/inference-baseline.json`` 的 ``framework`` 槽位；
- ``handrolled`` 槽位**预留为空**（各字段 None + ``todo`` 说明）：等 P2-P5
  手搓引擎能以同协议出数后按字段回填，两边即可逐字段对比。

口径约定（同时写进 record 的 ``protocol``，改动需同步 P1 README）：
- 延迟为**客户端视角**（含 HTTP 往返），毫秒；
- ``tokens_per_s = completion_tokens / 延迟秒``；
- 采样固定 greedy（``temperature=0``），让两边的生成文本都确定、可复现；
- 指标请求发生在冒烟断言之后 —— **只有冒烟通过的运行才配留下记录**。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

METRICS_SCHEMA_VERSION = 1

RECORD_DESCRIPTION = (
    "推理指标对拍记录：`framework` 槽位由 `make smoke` 自动写入（框架部署基线），"
    "`handrolled` 槽位预留，待手搓引擎（P2-P5）按同协议出数后回填，供逐字段对比。"
)

HANDROLLED_TODO = (
    "预留对比位：手搓引擎按 framework.protocol 同口径（greedy、同 prompt、同 max_tokens、"
    "同请求数）跑 N 次补全后，把各字段填到这里；不要动 framework 槽位。"
)

# framework / handrolled 两个槽位共用的字段集（对比时逐字段对齐）
SIDE_FIELDS = (
    "engine",
    "model",
    "param_count",
    "device",
    "library_versions",
    "sample_text",
    "requests",
    "summary",
)

CompleteFn = Callable[..., dict]


def default_record_path(root: Path | None = None) -> Path:
    """指标记录的默认落点：``<repo>/docs/metrics/inference-baseline.json``。"""
    root = root or Path(__file__).resolve().parents[2]
    return root / "docs" / "metrics" / "inference-baseline.json"


def build_protocol(prompt: str, max_tokens: int, requests: int) -> dict:
    """对拍协议：两边（框架/手搓）必须用完全相同的口径出数，否则没有可比性。"""
    return {
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "requests": int(requests),
        "sampling": "greedy(temperature=0)",
        "latency_view": "client-side(含 HTTP 往返)",
        "tokens_per_s": "completion_tokens / latency_s",
    }


def measure_request(
    complete_fn: CompleteFn,
    prompt: str,
    max_tokens: int,
    *,
    index: int = 0,
) -> dict:
    """发一次补全请求并计时，返回单条指标行（不含文本，文本只留首条样本）。

    ``complete_fn(prompt=..., max_tokens=...) -> {"text", "prompt_tokens",
    "completion_tokens"}``：HTTP 客户端或测试桩都可注入。
    """
    start = time.perf_counter()
    result = complete_fn(prompt=prompt, max_tokens=max_tokens)
    # 先落定延迟精度，再算吞吐：保证记录里 tokens_per_s 能由 latency_ms 原样复算（口径自洽）
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    completion_tokens = int(result["completion_tokens"])
    return {
        "index": int(index),
        "prompt_tokens": int(result["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tokens_per_s": (
            round(completion_tokens / (latency_ms / 1000.0), 2) if latency_ms > 0 else None
        ),
    }


def summarize(rows: list[dict]) -> dict:
    """把多条指标行压成一个对比友好的摘要（均值 / p50 / 聚合吞吐）。"""
    if not rows:
        raise ValueError("summarize 至少需要一条指标行")
    latencies = sorted(float(r["latency_ms"]) for r in rows)
    total_latency_ms = sum(latencies)
    total_completion = sum(int(r["completion_tokens"]) for r in rows)
    mid = len(latencies) // 2
    p50 = float(latencies[mid]) if len(latencies) % 2 else (latencies[mid - 1] + latencies[mid]) / 2
    return {
        "requests": len(rows),
        "prompt_tokens": int(rows[0]["prompt_tokens"]),
        "completion_tokens_per_request": int(rows[0]["completion_tokens"]),
        "completion_tokens_total": total_completion,
        "latency_ms_avg": round(total_latency_ms / len(rows), 2),
        "latency_ms_p50": round(p50, 2),
        "latency_ms_min": round(latencies[0], 2),
        "latency_ms_max": round(latencies[-1], 2),
        "tokens_per_s": round(total_completion / (total_latency_ms / 1000.0), 2)
        if total_latency_ms > 0
        else None,
    }


def _empty_side() -> dict:
    """手搓侧的预留槽位：字段齐、值全空，外加一行 todo 说明回填口径。"""
    side = {field: None for field in SIDE_FIELDS}
    side["todo"] = HANDROLLED_TODO
    return side


def build_record(
    *,
    protocol: dict,
    engine_info: dict,
    rows: list[dict],
    sample_text: str,
    now: str | None = None,
) -> dict:
    """组装对拍记录：``framework`` 填满，``handrolled`` 留空槽位。"""
    if not rows:
        raise ValueError("build_record 至少需要一条指标行（冒烟通过后才调用）")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "description": RECORD_DESCRIPTION,
        "updated_at": now
        or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "protocol": dict(protocol),
        "framework": {
            "engine": engine_info.get("engine") or "transformers",
            "model": engine_info.get("model"),
            "param_count": engine_info.get("param_count"),
            "device": engine_info.get("device"),
            "library_versions": engine_info.get("library_versions"),
            "sample_text": sample_text,
            "requests": rows,
            "summary": summarize(rows),
        },
        "handrolled": _empty_side(),
    }


def write_record(record: dict, path: Path) -> Path:
    """落盘（UTF-8、缩进 2、结尾换行），返回写入路径。目录不存在则创建。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_record(path: Path) -> dict:
    """读回记录（守护测试与对拍脚本用）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
