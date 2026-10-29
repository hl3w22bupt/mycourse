"""B>1 批量生成 bug 修复核查脚本。

针对两处缺陷的行为化核查（不锁定具体修复形态，只看行为契约）：
  缺陷1：``sample_token`` 输入 (B>1, V) logits 时 ``int(idx[..., 0])`` 抛
         ``ValueError: only one element tensors can be converted to Python scalars``；
  缺陷2：``generate`` 把单个 next_id 广播给全 batch（``torch.full((B,1), next_id)``），
         batch 内各条序列本应各自采样各自的下一个 token。

三个检查节：
  1. sample_token 批量输入 (B, V) 不再抛 ValueError（修复后返回批量结果或契约化报错）；
  2. 批量 generate（贪心、各行偏好不同）不崩溃，且各条序列的新 token 互异 —— 判定无广播；
  3. 批量 generate 输出与「逐条循环生成」逐行全等 —— 判定批量语义正确。

用法：``.venv/bin/python scripts/verify_b1_batch.py``（退出码 0 = 全部通过）。
"""

from __future__ import annotations

import sys

import torch

from infer_lab.p4_generate.generate import GenerationConfig
from infer_lab.p4_generate.solutions.generate import generate
from infer_lab.p4_sampling.solutions.sampling import sample_token

VOCAB = 8
BATCH = 4


class RowDistinctStubModel(torch.nn.Module):
    """确定性测试桩：行 r 在第 s 步的偏好 token 固定为 (s + row_offset + r) % VOCAB。

    关键性质：**每行的 logits 只依赖行号 r 与步数 s，与 batch 维度无关** ——
    单独跑第 b 行（实例取 row_offset=b）和把它放进批量跑（实例 row_offset=0），
    该行每一步的 logits 完全一致，因此「批量输出 == 逐条输出」的对拍才有意义
    （top_k=1 贪心下逐 token 可比）。
    broadcast 缺陷在此桩下必然暴露：各行贪心偏好互不相同，
    若 generate 用同一个 next_id 广播，输出各行将完全相同。
    """

    def __init__(self, vocab: int = VOCAB, row_offset: int = 0) -> None:
        super().__init__()
        self.vocab = vocab
        self.row_offset = row_offset
        self.step = 0

    def forward(self, idx, past_kv=None, use_cache=True):  # noqa: ARG002
        b = idx.shape[0]
        logits = torch.zeros(b, idx.shape[1], self.vocab)
        for row in range(b):
            logits[row, :, (self.step + self.row_offset + row) % self.vocab] = 5.0
        self.step += 1
        return (logits, past_kv) if past_kv is not None else (logits, None)


GREEDY = GenerationConfig(max_new_tokens=4, temperature=1.0, top_k=1, use_cache=True)


def check_sample_token_batch() -> bool:
    """缺陷1：sample_token 输入 (B>1, V) 不应抛 ValueError。"""
    torch.manual_seed(0)
    logits = torch.randn(BATCH, VOCAB)
    try:
        out = sample_token(logits, temperature=1.0)
    except ValueError as e:
        print(f"[FAIL] 节1 sample_token((B={BATCH},V)) 仍抛 ValueError: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 节1 sample_token((B={BATCH},V)) 抛出非 ValueError 异常 {type(e).__name__}: {e}")
        return False
    ids = out if isinstance(out, (list, tuple)) else None
    if ids is None and torch.is_tensor(out) and out.dim() >= 1:
        ids = out.reshape(-1).tolist()
    if ids is None:
        print(f"[WARN] 节1 返回单值 {out!r} —— 批量输入未按行采样（若契约如此，需 generate 内部逐行调用）")
        return True
    if len(ids) != BATCH:
        print(f"[FAIL] 节1 返回 {len(ids)} 个 id，期望 {BATCH}")
        return False
    # 贪心下各行独立确定性：给各行构造偏好互异的 one-hot logits，采样必等于各自偏好
    pref = torch.zeros(BATCH, VOCAB)
    for row in range(BATCH):
        pref[row, (row + 1) % VOCAB] = 5.0
    greedy = sample_token(pref, temperature=1.0, top_k=1)
    greedy_ids = greedy if isinstance(greedy, (list, tuple)) else list(
        torch.as_tensor(greedy).reshape(-1).tolist()
    )
    expect = [(row + 1) % VOCAB for row in range(BATCH)]
    ok = greedy_ids == expect
    print(f"[{'PASS' if ok else 'FAIL'}] 节1 批量返回 {BATCH} 个 id；贪心逐行采样={greedy_ids} 期望={expect}")
    return ok


def check_generate_batch_no_broadcast() -> bool:
    """缺陷2：批量 generate 各行新 token 互异（broadcast 必使其全同）。"""
    model = RowDistinctStubModel()
    prompt = torch.arange(BATCH * 3, dtype=torch.long).reshape(BATCH, 3) % VOCAB
    try:
        out = generate(model, prompt, GREEDY)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 节2 批量 generate(B={BATCH}) 崩溃: {type(e).__name__}: {e}")
        return False
    new_ids = out[:, 3:]
    rows_identical = all(
        torch.equal(new_ids[0], new_ids[i]) for i in range(1, BATCH)
    )
    if rows_identical:
        print(f"[FAIL] 节2 各行新 token 完全相同 -> 单 token 广播全 batch 缺陷仍在:\n{new_ids}")
        return False
    print(f"[PASS] 节2 批量 generate 不崩溃，各行新 token 互异:\n{new_ids}")
    return True


def check_batch_equals_loop() -> bool:
    """批量 generate 输出逐行 == 逐条循环 generate 输出。"""
    torch.manual_seed(0)
    model = RowDistinctStubModel()
    prompt = torch.arange(BATCH * 3, dtype=torch.long).reshape(BATCH, 3) % VOCAB

    model.step = 0
    try:
        batch_out = generate(model, prompt, GREEDY)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 节3 批量 generate 崩溃，无法对拍: {type(e).__name__}: {e}")
        return False

    loop_out = []
    for row in range(BATCH):
        m = RowDistinctStubModel(row_offset=row)  # 单独跑行 r 的偏好序列须与批量中该行一致
        single = generate(m, prompt[row : row + 1], GREEDY)
        loop_out.append(single[0])
    loop_out = torch.stack(loop_out)

    ok = torch.equal(batch_out, loop_out)
    if ok:
        print(f"[PASS] 节3 批量输出与逐条循环逐行全等 (shape={tuple(batch_out.shape)})")
    else:
        diff = (batch_out != loop_out).nonzero().tolist()
        print(f"[FAIL] 节3 批量输出 != 逐条循环，差异位置 {diff}\n批量:\n{batch_out}\n逐条:\n{loop_out}")
    return ok


def main() -> int:
    results = [
        ("节1 sample_token (B>1,V)", check_sample_token_batch()),
        ("节2 generate 批量无广播", check_generate_batch_no_broadcast()),
        ("节3 批量==逐条", check_batch_equals_loop()),
    ]
    print("\n=== 核查汇总 ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(ok for _, ok in results)
    print(f"结论: {'通过' if all_ok else '不通过'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
