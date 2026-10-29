#!/usr/bin/env python3
"""P1 冒烟脚本：一键验证「框架部署路径」端到端可用，并留下框架侧指标记录。

流程：起 uvicorn 服务（子进程）→ 轮询 /health → POST /v1/completions →
断言生成文本非空 → 再打一发 /v1/chat/completions →
按对拍协议（greedy、同 prompt、同 max_tokens）计时打 N 次补全 →
把框架侧指标写进 docs/metrics/inference-baseline.json 的 ``framework`` 槽位
（已回填的 ``handrolled`` 槽位 / ``batch`` 节原样保留，不会被本次重写冲掉）→ 关服务。

用法（huggingface.co 直连不通时会自动走 hf-mirror 镜像）：
    make smoke
    # 等价于
    HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/smoke_framework.py \
        --model sshleifer/tiny-gpt2 --port 8001

退出码：0 = 冒烟通过且指标记录已写入；非 0 = 任一步失败。
输出带 SMOKE_ 前缀便于归档。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_lab.p1_framework.metrics import (  # noqa: E402
    build_protocol,
    build_record,
    default_record_path,
    measure_request,
    write_record,
)

DEFAULT_MODEL = str(ROOT / "models_cache" / "tiny-gpt2-toy")
# huggingface.co 直连不通的环境用镜像（国内环境常见），已在 Makefile 里默认设置
FALLBACK_HF_ENDPOINT = "https://hf-mirror.com"


def ensure_model(model: str) -> None:
    """默认的本地 bootstrap 模型缺失时先离线构建（幂等，秒级~分钟级）。"""
    if Path(model).exists():
        return
    print(f"SMOKE_BOOTSTRAP {model} 不存在，先离线自建 tiny-gpt2 ……", flush=True)
    r = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "scripts/bootstrap_tiny_model.py"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        raise SystemExit(f"bootstrap 失败（exit={r.returncode}），无法继续冒烟")


def _post(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_health(base_url: str, deadline_s: float = 240.0) -> None:
    """首次会下载模型权重，超时给足（tiny 模型走镜像通常 < 1 分钟）。"""
    print(f"SMOKE_WAIT_FOR_HEALTH {base_url}", flush=True)
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            body = _get(f"{base_url}/health")
            if body.get("status") == "ok":
                print(f"SMOKE_HEALTH {body}", flush=True)
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"服务 {deadline_s}s 内未就绪：{base_url}")


# ---------- 指标记录（框架侧基线，供手搓版对拍） ----------

def fetch_engine_info(base_url: str) -> dict:
    """读引擎元信息（device / 参数量 / 版本）；端点缺席时字段留空，不阻塞冒烟。"""
    try:
        return _get(f"{base_url}/v1/engine/info")
    except (urllib.error.URLError, OSError):
        print("SMOKE_METRICS_INFO_UNAVAILABLE /v1/engine/info 不可用，元信息字段留空", flush=True)
        return {}


def collect_metric_rows(
    base_url: str, prompt: str, max_tokens: int, requests: int
) -> tuple[list[dict], str]:
    """按对拍协议（greedy）计时打 N 次补全，返回（指标行列表, 首条生成文本样本）。"""
    texts: list[str] = []

    def complete_fn(prompt: str, max_tokens: int) -> dict:
        body = _post(
            f"{base_url}/v1/completions",
            # greedy（temperature=0）：两边（框架/手搓）输出确定，数字与文本才可比
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        )
        usage = body["usage"]
        texts.append(body["choices"][0]["text"])
        return {
            "text": body["choices"][0]["text"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }

    rows = []
    for i in range(requests):
        row = measure_request(complete_fn, prompt, max_tokens, index=i)
        print(f"SMOKE_METRICS_REQUEST {row}", flush=True)
        rows.append(row)
    sample = texts[0] if texts else ""
    return rows, sample


def main() -> int:
    p = argparse.ArgumentParser(description="infer-lab P1 框架部署冒烟（含指标记录）")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--prompt", default="the little engine could")  # 语料内短语，保证 in-domain
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--metric-requests", type=int, default=5,
                   help="冒烟通过后按对拍协议计时打的补全请求数")
    p.add_argument("--metrics-out", default="",
                   help="指标记录落点（默认 docs/metrics/inference-baseline.json）；传 '' 跳过记录")
    args = p.parse_args()

    # hf 直连探测失败时补一个镜像端点（仅回退 Hub 模型时才会用到）
    env = dict(os.environ)
    if not env.get("HF_ENDPOINT"):
        import urllib.request as _u

        try:
            _u.urlopen("https://huggingface.co/api/models", timeout=5)
        except OSError:
            env["HF_ENDPOINT"] = FALLBACK_HF_ENDPOINT
            print(f"SMOKE_HF_FALLBACK {FALLBACK_HF_ENDPOINT}", flush=True)

    ensure_model(args.model)
    base = f"http://{args.host}:{args.port}"
    server = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "infer_lab.p1_framework.server",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server_log_tail: list[str] = []
    try:
        wait_for_health(base)

        completion = _post(
            f"{base}/v1/completions",
            {
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.3,
                "top_k": 50,
                "top_p": 0.95,
            },
        )
        text = completion["choices"][0]["text"]
        print(f"SMOKE_COMPLETION_USAGE {completion['usage']}", flush=True)
        print(f"SMOKE_COMPLETION_TEXT {text!r}", flush=True)
        if not text.strip():
            print("SMOKE_FAIL 生成文本为空", flush=True)
            return 1

        chat = _post(
            f"{base}/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": "You are a tiny GPT-2 study buddy."},
                    {"role": "user", "content": "Say hello."},
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0.3,
            },
        )
        chat_text = chat["choices"][0]["message"]["content"]
        print(f"SMOKE_CHAT_TEXT {chat_text!r}", flush=True)
        if not chat_text.strip():
            print("SMOKE_FAIL chat 生成文本为空", flush=True)
            return 1

        # ---- 指标记录位：只有冒烟断言全部通过的运行才写框架侧基线 ----
        if args.metrics_out:
            try:
                protocol = build_protocol(args.prompt, args.max_tokens, args.metric_requests)
                print(f"SMOKE_METRICS_PROTOCOL {protocol}", flush=True)
                rows, sample_text = collect_metric_rows(
                    base, args.prompt, args.max_tokens, args.metric_requests
                )
                record = build_record(
                    protocol=protocol,
                    engine_info=fetch_engine_info(base),
                    rows=rows,
                    sample_text=sample_text,
                )
                out_path = write_record(record, Path(args.metrics_out))
                print(f"SMOKE_METRICS_SUMMARY {record['framework']['summary']}", flush=True)
                print(
                    f"SMOKE_METRICS_SAVED {out_path}"
                    "（framework 槽位已刷新；已回填的 handrolled / batch 数据原样保留）",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - 记录失败必须让冒烟失败，避免「无记录假通过」
                print(f"SMOKE_FAIL 指标记录失败：{exc!r}", flush=True)
                return 1

        print("SMOKE_OK 框架部署端到端通过（起服务 → 生成非空 → chat 通道 → 指标留档）", flush=True)
        return 0
    finally:
        server.terminate()
        try:
            out, _ = server.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            out, _ = server.communicate()
        if out:
            server_log_tail = out.strip().splitlines()[-8:]
            print("SMOKE_SERVER_LOG_TAIL", flush=True)
            for line in server_log_tail:
                print(f"  {line}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
