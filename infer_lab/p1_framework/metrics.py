"""P1 框架侧推理指标记录（胶水模块，直接可用，不是 TODO）。

目的：给「框架版 vs 手搓版」对拍留下**同一口径**的数字，而不是各跑各的。

- ``make smoke`` 冒烟断言通过后，用固定的对拍协议（greedy 采样、同 prompt、
  同 max_tokens、同请求数）再打 N 次补全请求，把延迟 / 吞吐 / token 用量
  写进 ``docs/metrics/inference-baseline.json`` 的 ``framework`` 槽位；
- ``handrolled`` 槽位默认**预留为空**（各字段 None + ``todo`` 说明）：等 P2-P5
  手搓引擎能以同协议出数后，用 ``scripts/measure_handrolled.py`` 按字段回填，
  两边即可逐字段对比；回填之后再次 ``make smoke`` **不会冲掉手搓侧数据**
  （见 ``write_record`` 的 preserve 逻辑）；
- 顶层 ``batch`` 节（schema v2 新增）：B>1 批量生成对拍 —— 同一 prompt 复制
  B 行、greedy、各生成 ``max_tokens`` 个 token，两边都 in-process 计时，
  专门回答「批量解码到底赚不赚 / 赚多少」。

口径约定（同时写进 record 的 ``protocol``，改动需同步 P1 README）：
- 单请求延迟为**客户端视角**（含 HTTP 往返），毫秒；batch 节为 **in-process**
  （两边都不走 HTTP，互相可比）；
- ``tokens_per_s = completion_tokens / 延迟秒``；
- 采样固定 greedy（框架 ``temperature=0`` / 手搓 ``top_k=1``，二者等价），
  让两边的生成文本都确定、可复现；
- 指标请求发生在冒烟断言之后 —— **只有冒烟通过的运行才配留下记录**。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

METRICS_SCHEMA_VERSION = 2

RECORD_DESCRIPTION = (
    "推理指标对拍记录：`framework` 槽位由 `make smoke` 自动写入（框架部署基线），"
    "`handrolled` 槽位由 `scripts/measure_handrolled.py` 按同协议回填（手搓引擎），"
    "顶层 `batch` 节是 B>1 批量生成对拍（两边 in-process 计时），供逐字段对比。"
)

HANDROLLED_TODO = (
    "预留对比位：手搓引擎按 framework.protocol 同口径（greedy、同 prompt、同 max_tokens、"
    "同请求数）跑 N 次补全后，把各字段填到这里；不要动 framework 槽位。"
    "回填方式：make train-demo && .venv/bin/python scripts/measure_handrolled.py"
)

BATCH_TODO = (
    "预留对比位：B>1 批量生成对拍（同 prompt 复制 B 行、greedy、各生成 max_tokens 个 token、"
    "两边 in-process 计时），跑 scripts/measure_handrolled.py 自动回填两侧。"
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

# batch 节里框架 / 手搓两侧共用的字段集
BATCH_SIDE_FIELDS = ("engine", "model", "batch_sizes", "rows", "summary")

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


# ---------- B>1 批量生成对拍（schema v2 新增的顶层 batch 节） ----------


def build_batch_protocol(
    prompt: str, max_tokens: int, batch_sizes: list[int], repeats: int = 1
) -> dict:
    """批量对拍协议：同 prompt 复制 B 行、greedy、各生成 max_tokens 个 token。

    与单请求协议（``build_protocol``）的三处刻意差异：
    - 计时 **in-process**（两边都不走 HTTP，才能公平比「批量解码」本身的收益）；
    - 吞吐按**整批**算：``B × max_tokens / 延迟秒`` —— 这才是 batch 的意义；
    - 计时口径 ``repeats``：每个 batch 形状先热身、再重复计时取**最小延迟**
      （新形状首跑带内核编译开销，最小值最接近真实算力）。
    """
    return {
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "batch_sizes": [int(b) for b in batch_sizes],
        "sampling": "greedy(框架 do_sample=False / 手搓 top_k=1)",
        "latency_view": "in-process(无 HTTP 往返)",
        "latency_aggregation": f"每尺寸热身后重复 {int(repeats)} 次取最小延迟",
        "tokens_per_s": "batch 总补全 token / 延迟秒（整批吞吐）",
    }


def measure_batch_run(
    batch_generate_fn: CompleteFn,
    prompt: str,
    max_tokens: int,
    *,
    batch_size: int,
) -> dict:
    """计时一次 B 行批量生成，返回单条批量指标行。

    ``batch_generate_fn(prompt=..., max_tokens=..., batch_size=...) ->
    {"completion_tokens_total", ...}``：真引擎或测试桩都可注入。
    """
    start = time.perf_counter()
    result = batch_generate_fn(prompt=prompt, max_tokens=max_tokens, batch_size=batch_size)
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    total = int(result["completion_tokens_total"])
    if total <= 0:
        raise ValueError("批量指标行至少要生成 1 个 token")
    return {
        "batch_size": int(batch_size),
        "completion_tokens_total": total,
        "latency_ms": latency_ms,
        "tokens_per_s": round(total / (latency_ms / 1000.0), 2) if latency_ms > 0 else None,
    }


def summarize_batch(rows: list[dict]) -> dict:
    """把各 batch 尺寸的指标行压成摘要：单行延迟与整批吞吐随 B 的变化。

    ``speedup_vs_b1`` 是这个节最值得看的一个数字：整批吞吐(B) / 整批吞吐(B=1)，
    >1 说明批量解码确实摊薄了每 token 的固定开销。
    """
    if not rows:
        raise ValueError("summarize_batch 至少需要一条指标行")
    by_b = {int(r["batch_size"]): r for r in rows}
    if 1 not in by_b:
        raise ValueError("summarize_batch 需要 batch_size=1 的基线行（计算 speedup 用）")
    base_tps = by_b[1]["tokens_per_s"]
    out: dict = {"batch_sizes": sorted(by_b), "rows": []}
    for b in sorted(by_b):
        r = by_b[b]
        out["rows"].append(
            {
                "batch_size": b,
                "latency_ms": r["latency_ms"],
                "completion_tokens_total": r["completion_tokens_total"],
                "tokens_per_s": r["tokens_per_s"],
                "tokens_per_s_per_sequence": (
                    round(r["tokens_per_s"] / b, 2) if r["tokens_per_s"] else None
                ),
                "speedup_vs_b1": (
                    round(r["tokens_per_s"] / base_tps, 2)
                    if base_tps and r["tokens_per_s"]
                    else None
                ),
            }
        )
    best = max(out["rows"], key=lambda r: r["tokens_per_s"] or 0.0)
    out["best_tokens_per_s"] = best["tokens_per_s"]
    out["best_batch_size"] = best["batch_size"]
    return out


def build_batch_section(
    *,
    protocol: dict,
    framework_side: dict | None,
    handrolled_side: dict | None,
) -> dict:
    """组装顶层 batch 节：两侧都出数才完整，缺的一侧保持预留。"""
    section: dict = {
        "description": (
            "B>1 批量生成对拍：同一 prompt 复制 B 行、greedy、各生成 max_tokens 个 token，"
            "两边 in-process 计时 —— 回答「批量解码赚不赚、赚多少」。"
        ),
        "protocol": dict(protocol),
        "framework": framework_side if framework_side is not None else _empty_batch_side(),
        "handrolled": handrolled_side if handrolled_side is not None else _empty_batch_side(),
    }
    return section


def _empty_batch_side() -> dict:
    side = {field: None for field in BATCH_SIDE_FIELDS}
    side["todo"] = BATCH_TODO
    return side


def is_side_filled(side: dict | None) -> bool:
    """槽位是否已回填真实数据（summary 非空，且单请求侧 requests / 批量侧 rows 非空）。"""
    if not side or side.get("summary") is None:
        return False
    return bool(side.get("requests") or side.get("rows"))


# ---------- 回填与落盘 ----------


def backfill_handrolled(record: dict, side: dict, *, now: str | None = None) -> dict:
    """把手搓侧数据回填进对拍记录，返回新 record（不改 framework 槽位）。

    校验先行：字段不全 / 数字不健康（param_count、吞吐非正）一律拒绝，
    避免把「半份数据」当成可对拍基线写进仓库。
    """
    missing = [f for f in SIDE_FIELDS if side.get(f) is None]
    if missing:
        raise ValueError(f"handrolled 槽位缺字段：{missing}")
    if not isinstance(side["param_count"], int) or side["param_count"] <= 0:
        raise ValueError("handrolled.param_count 必须是正整数")
    if not is_side_filled(side):
        raise ValueError("handrolled.requests/summary 必须有真实数据")
    tps = side["summary"].get("tokens_per_s")
    if not isinstance(tps, (int, float)) or tps <= 0:
        raise ValueError("handrolled.summary.tokens_per_s 必须是正数")

    record = dict(record)
    filled = {f: side[f] for f in SIDE_FIELDS}
    # 口径差异要写在回填数据旁边（如手搓侧 in-process 无 HTTP），否则逐字段对比会误读
    for extra in ("protocol_delta", "backfilled_at", "backfill_script", "notes"):
        if side.get(extra) is not None:
            filled[extra] = side[extra]
    filled["backfilled_at"] = filled.get("backfilled_at") or now or _utcnow()
    record["handrolled"] = filled
    return _stamp_schema_v2(record)


def attach_batch_section(record: dict, section: dict) -> dict:
    """把 B>1 批量对拍节挂到记录顶层（schema v2）。"""
    record = dict(record)
    record["batch"] = section
    return _stamp_schema_v2(record)


def _stamp_schema_v2(record: dict) -> dict:
    """v1 记录一旦长出 v2 特性（handrolled 回填 / batch 节），版本与描述同步升级。"""
    record = dict(record)
    record["schema_version"] = METRICS_SCHEMA_VERSION
    record["description"] = RECORD_DESCRIPTION
    return record


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_record(
    *,
    protocol: dict,
    engine_info: dict,
    rows: list[dict],
    sample_text: str,
    now: str | None = None,
) -> dict:
    """组装对拍记录：``framework`` 填满，``handrolled`` / ``batch`` 留空槽位。"""
    if not rows:
        raise ValueError("build_record 至少需要一条指标行（冒烟通过后才调用）")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "description": RECORD_DESCRIPTION,
        "updated_at": now or _utcnow(),
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
        "batch": build_batch_section(
            protocol=build_batch_protocol(
                protocol["prompt"], protocol["max_tokens"], batch_sizes=[]
            ),
            framework_side=None,
            handrolled_side=None,
        ),
    }


def write_record(record: dict, path: Path, *, preserve_backfill: bool = True) -> Path:
    """落盘（UTF-8、缩进 2、结尾换行），返回写入路径。目录不存在则创建。

    ``preserve_backfill=True``（默认）：``make smoke`` 重写记录时，若目标文件里
    已有回填好的 ``handrolled`` / ``batch`` 数据，则原样保留 —— 刷新框架侧基线
    不许把手搓侧对拍数据冲掉。
    """
    path = Path(path)
    if preserve_backfill and path.exists():
        try:
            existing = load_record(path)
        except (OSError, ValueError, KeyError):
            existing = None
        if existing:
            if is_side_filled(existing.get("handrolled")) and not is_side_filled(
                record.get("handrolled")
            ):
                record = dict(record)
                record["handrolled"] = existing["handrolled"]
            existing_batch = existing.get("batch")
            if isinstance(existing_batch, dict) and (
                is_side_filled(existing_batch.get("handrolled"))
                or is_side_filled(existing_batch.get("framework"))
            ):
                new_batch = record.get("batch")
                if not (
                    isinstance(new_batch, dict)
                    and (
                        is_side_filled(new_batch.get("handrolled"))
                        or is_side_filled(new_batch.get("framework"))
                    )
                ):
                    record = dict(record)
                    record["batch"] = existing_batch
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_record(path: Path) -> dict:
    """读回记录（守护测试与对拍脚本用）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
