#!/usr/bin/env python3
"""手搓引擎（P2-P5）指标回填脚本：按 framework 同口径出数，回填对拍记录。

回答一个问题：**同一条 prompt、同样 greedy 生成 24 个 token，手搓版多快？**
以及批量版才有答案的一问：**B>1 批量解码让吞吐翻了几倍？**

流程：
1. 读现有对拍记录（``docs/metrics/inference-baseline.json``）的 ``protocol``，
   手搓侧严格沿用（同 prompt / 同 max_tokens / 同请求数 / greedy）；
2. 加载 P5 训练出的手搓 checkpoint（``runs/toy/ckpt.pt``，缺了就提示先
   ``make train-demo``），B=1 计时打 N 次补全 → 回填 ``handrolled`` 槽位；
3. B>1 批量对拍：同一 prompt 复制 B 行、greedy、各生成 max_tokens 个 token，
   手搓引擎与框架引擎（transformers，in-process）各测一遍 → 写顶层 ``batch`` 节；
4. 落盘。**永不改动 ``framework`` 槽位**（那是 `make smoke` 写的框架侧基线）。

口径说明（同时写进记录的 ``handrolled.protocol_delta``，避免逐字段对比时误读）：
- 手搓引擎没有 HTTP 服务，延迟为 **in-process**；framework 槽位是客户端视角
  （含 HTTP 往返）—— 绝对值不可直接比，只比「同侧结构」与 batch 节（两边都
  in-process，可比）；
- greedy 的实现：框架 ``do_sample=False``（temperature=0），手搓 ``top_k=1``
  （分布退化成点质量，温度不起作用），二者等价、输出确定；
- tokenizer 粒度不同：framework WordLevel 同 prompt = 4 tokens，手搓字符级
  = 23 tokens。吞吐按**补全 token 数**算，两边都是 24 个 token，可比。

用法（huggingface.co 直连不通时 batch 节的框架侧自动走 hf-mirror 镜像）：
    make train-demo            # 先有手搓 checkpoint
    .venv/bin/python scripts/measure_handrolled.py
    # 或 make metrics-handrolled

退出码：0 = 回填成功；非 0 = 任一步失败（记录保持原样）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from infer_lab.common.device import pick_device, seed_everything  # noqa: E402
from infer_lab.p1_framework.metrics import (  # noqa: E402
    build_batch_protocol,
    build_batch_section,
    backfill_handrolled,
    attach_batch_section,
    default_record_path,
    load_record,
    measure_batch_run,
    measure_request,
    summarize,
    summarize_batch,
    write_record,
)
from infer_lab.p4_generate.generate import GenerationConfig  # noqa: E402
from infer_lab.p4_generate.solutions.generate import generate as handrolled_generate  # noqa: E402

DEFAULT_CKPT = ROOT / "runs" / "toy" / "ckpt.pt"
DEFAULT_FRAMEWORK_MODEL = ROOT / "models_cache" / "tiny-gpt2-toy"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
FALLBACK_HF_ENDPOINT = "https://hf-mirror.com"


def display_path(path: str | Path) -> str:
    """指标记录里的路径统一写成仓库相对路径（绝对路径含本机目录，不便入库）。"""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:  # 不在仓库内：原样记录
        return str(p)


# ---------- 手搓侧（P2-P5） ----------


def load_handrolled_engine(ckpt_path: Path, device_pref: str = "auto"):
    """载入手搓 checkpoint，返回 (model, tokenizer, device)。缺 checkpoint 给出可执行指引。"""
    if not ckpt_path.exists():
        raise SystemExit(
            f"手搓 checkpoint 不存在：{ckpt_path}\n"
            "先跑 `make train-demo` 产出（秒级），再执行本脚本回填指标。"
        )
    device = pick_device(device_pref)
    from infer_lab.p5_toy.train import load_checkpoint

    model, tokenizer = load_checkpoint(str(ckpt_path), device)
    model.eval()
    return model, tokenizer, device


def handrolled_complete_fn(model, tokenizer, device, *, seed: int):
    """B=1 补全函数：greedy（top_k=1）生成 max_tokens 个 token。

    与 framework.complete 的返回口径对齐：{"text", "prompt_tokens", "completion_tokens"}。
    """

    def complete(prompt: str, max_tokens: int) -> dict:
        seed_everything(seed)  # greedy 下不影响结果，但保持与框架侧「每次置种子」同姿态
        ids = tokenizer.encode(prompt)
        room = model.block_size - len(ids)
        if room < max_tokens:
            raise SystemExit(
                f"prompt {len(ids)} tokens + max_tokens {max_tokens} 超过手搓模型 "
                f"block_size={model.block_size}，缩短 prompt 或重训更大 block_size"
            )
        cfg = GenerationConfig(max_new_tokens=max_tokens, temperature=1.0, top_k=1)
        with torch.no_grad():
            out = handrolled_generate(
                model, torch.tensor([ids], dtype=torch.long, device=device), cfg
            )
        completion = out[0][len(ids):].tolist()
        return {
            "text": tokenizer.decode(completion),
            "prompt_tokens": len(ids),
            "completion_tokens": len(completion),
        }

    return complete


def handrolled_batch_fn(model, tokenizer, device, *, seed: int):
    """B 行批量生成函数：同一 prompt 复制 B 行，greedy，各生成 max_tokens 个 token。"""

    def batch_generate(prompt: str, max_tokens: int, batch_size: int) -> dict:
        seed_everything(seed)
        ids = tokenizer.encode(prompt)
        cfg = GenerationConfig(max_new_tokens=max_tokens, temperature=1.0, top_k=1)
        batch = torch.tensor([ids] * batch_size, dtype=torch.long, device=device)
        with torch.no_grad():
            out = handrolled_generate(model, batch, cfg)
        total = int(out.shape[1] - batch.shape[1]) * int(out.shape[0])
        return {
            "completion_tokens_total": total,
            "sequences": int(out.shape[0]),
        }

    return batch_generate


# ---------- 框架侧（仅 batch 节用：in-process 直调 transformers） ----------


def load_framework_model(model_dir: Path, device):
    """加载 bootstrap 出的 tiny-gpt2（in-process 批量对拍用）；缺失则先离线自建。"""
    if not model_dir.exists():
        print(f"MEASURE_BOOTSTRAP {model_dir} 不存在，先离线自建 tiny-gpt2 ……", flush=True)
        env_note = (
            f"（HF_ENDPOINT={FALLBACK_HF_ENDPOINT}，全离线）"
            if not _reachable("https://huggingface.co")
            else ""
        )
        print(f"MEASURE_HF{env_note}", flush=True)
        r = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), "scripts/bootstrap_tiny_model.py"],
            cwd=ROOT,
        )
        if r.returncode != 0:
            raise SystemExit(f"bootstrap 失败（exit={r.returncode}），batch 节框架侧无法测量")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device).eval()
    return model, tok


def _reachable(url: str, timeout: float = 5.0) -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except OSError:
        return False


def framework_batch_fn(model, tok, device):
    """框架批量生成函数：do_sample=False（greedy）+ eos_token_id=None 关早停。

    关早停是为了与手搓侧同口径：手搓 ``GenerationConfig`` 默认 ``eos_id=None``
    （<eos> 从不出现在玩具语料词表里），两边每行都恰好生成 max_tokens 个 token。
    """

    def batch_generate(prompt: str, max_tokens: int, batch_size: int) -> dict:
        torch.manual_seed(42)
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        batch = input_ids.repeat(batch_size, 1)
        with torch.no_grad():
            out = model.generate(
                batch,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tok.pad_token_id or 0,
            )
        total = int(out.shape[1] - batch.shape[1]) * int(out.shape[0])
        return {"completion_tokens_total": total, "sequences": int(out.shape[0])}

    return batch_generate


# ---------- 主流程 ----------


def measure_side(
    batch_fn,
    prompt: str,
    max_tokens: int,
    batch_sizes: list[int],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> list[dict]:
    """对一个批量生成函数按各 batch 尺寸计时，返回每尺寸一行（取重复测量的最小延迟）。

    为什么不能只测一次：MPS/新硬件后端对**每个新 batch 形状**都有一次性的
    内核编译 / 图缓存开销（实测能把首跑拖慢一个数量级），且小模型单次耗时
    本身抖动大。所以每个尺寸先热身 ``warmup`` 次，再计时 ``repeats`` 次，
    取**最小延迟** —— 微基准惯例：最小值最接近真实算力，系统噪音只会更慢。
    """
    rows = []
    for b in batch_sizes:
        for _ in range(warmup):
            batch_fn(prompt=prompt, max_tokens=max_tokens, batch_size=b)
        attempts = [
            measure_batch_run(batch_fn, prompt, max_tokens, batch_size=b)
            for _ in range(repeats)
        ]
        best = min(attempts, key=lambda r: r["latency_ms"])
        rows.append(best)
        print(
            f"MEASURE_BATCH_ROW {best}（{repeats} 次取最小，"
            f"全部延迟 ms：{[a['latency_ms'] for a in attempts]}）",
            flush=True,
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="infer-lab 手搓引擎指标回填（framework 同口径）")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--framework-model", default=str(DEFAULT_FRAMEWORK_MODEL))
    p.add_argument("--record", default=str(default_record_path()))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
                   help="批量对拍的 B 序列（逗号分隔，必须含 1 作基线）")
    p.add_argument("--repeats", type=int, default=3,
                   help="每个 batch 尺寸重复计时次数（取最小延迟，抗系统噪音）")
    p.add_argument("--no-framework-batch", action="store_true",
                   help="跳过 batch 节框架侧测量（无网/无 bootstrap 模型时）")
    args = p.parse_args()

    record_path = Path(args.record)
    if not record_path.exists():
        raise SystemExit(f"对拍记录不存在：{record_path}，先跑 `make smoke` 生成框架侧基线")
    record = load_record(record_path)
    protocol = record["protocol"]
    prompt, max_tokens, requests = protocol["prompt"], int(protocol["max_tokens"]), int(
        protocol["requests"]
    )
    print(f"MEASURE_PROTOCOL {protocol}", flush=True)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    if not batch_sizes or batch_sizes[0] != 1:
        raise SystemExit("--batch-sizes 必须以 1 开头（speedup 基线）")

    seed_everything(args.seed)
    model, tokenizer, device = load_handrolled_engine(Path(args.ckpt), args.device)
    param_count = int(sum(p_.numel() for p_ in model.parameters()))
    print(
        f"MEASURE_HANDROLLED_ENGINE params={param_count} device={device} "
        f"vocab={tokenizer.vocab_size} block_size={model.block_size}",
        flush=True,
    )

    # ---- ① B=1：与 framework 槽位同协议（同 prompt/max_tokens/请求数/greedy）----
    complete = handrolled_complete_fn(model, tokenizer, device, seed=args.seed)
    # 热身一次（首次前向有惰性初始化 / 内核编译开销，不入计时），顺手取确定样本文本
    sample_text = complete(prompt, max_tokens)["text"]
    print(f"MEASURE_SAMPLE_TEXT {sample_text!r}", flush=True)
    rows = []
    for i in range(requests):
        row = measure_request(complete, prompt, max_tokens, index=i)
        print(f"MEASURE_REQUEST {row}", flush=True)
        rows.append(row)

    handrolled_side = {
        "engine": "infer-lab handrolled (P2-P5 TinyGPT)",
        "model": display_path(args.ckpt),
        "param_count": param_count,
        "device": str(device),
        "library_versions": {"torch": torch.__version__},
        "sample_text": sample_text,
        "requests": rows,
        "summary": summarize(rows),
        "protocol_delta": {
            "latency_view": "in-process(无 HTTP 往返；framework 槽位为客户端视角含 HTTP)",
            "sampling": "greedy ≡ top_k=1（分布退化成点质量，温度不起作用）",
            "tokenizer": "字符级：同 prompt = "
            f"{rows[0]['prompt_tokens']} tokens（framework WordLevel = 4）",
            "model_source": "make train-demo 的 P5 手搓训练产物（与 framework 侧 tiny-gpt2 "
            "同为玩具语料同规模训练，但权重互不通用）",
        },
        "backfill_script": "scripts/measure_handrolled.py",
    }
    record = backfill_handrolled(record, handrolled_side)
    print(f"MEASURE_HANDROLLED_SUMMARY {handrolled_side['summary']}", flush=True)

    # ---- ② B>1 批量对拍（两边都 in-process，互相可比）----
    handrolled_rows = measure_side(
        handrolled_batch_fn(model, tokenizer, device, seed=args.seed),
        prompt,
        max_tokens,
        batch_sizes,
        repeats=args.repeats,
    )
    handrolled_batch = {
        "engine": "infer-lab handrolled (P2-P5 TinyGPT)",
        "model": display_path(args.ckpt),
        "batch_sizes": batch_sizes,
        "rows": handrolled_rows,
        "summary": summarize_batch(handrolled_rows),
    }

    framework_batch = None
    if not args.no_framework_batch:
        fw_model, fw_tok = load_framework_model(Path(args.framework_model), device)
        framework_rows = measure_side(
            framework_batch_fn(fw_model, fw_tok, device),
            prompt,
            max_tokens,
            batch_sizes,
            repeats=args.repeats,
        )
        framework_batch = {
            "engine": "transformers",
            "model": display_path(args.framework_model),
            "batch_sizes": batch_sizes,
            "rows": framework_rows,
            "summary": summarize_batch(framework_rows),
        }
        print(f"MEASURE_FRAMEWORK_BATCH_SUMMARY {framework_batch['summary']}", flush=True)
    else:
        print("MEASURE_FRAMEWORK_BATCH_SKIPPED --no-framework-batch", flush=True)

    print(f"MEASURE_HANDROLLED_BATCH_SUMMARY {handrolled_batch['summary']}", flush=True)
    record = attach_batch_section(
        record,
        build_batch_section(
            protocol=build_batch_protocol(prompt, max_tokens, batch_sizes, args.repeats),
            framework_side=framework_batch,
            handrolled_side=handrolled_batch,
        ),
    )

    write_record(record, record_path)
    print(f"MEASURE_SAVED {record_path}（handrolled 槽位 + batch 节已回填；framework 槽位未动）",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
