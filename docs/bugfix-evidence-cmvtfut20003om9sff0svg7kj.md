# 修复证据包：Bug `cmvtfut20003om9sff0svg7kj`

> infer-lab 参考实现批量生成崩溃：`sample_token` 对 (B>1,V) logits 抛 ValueError，
> `generate` 无法批量解码。
> 状态：**unconfirmed → 待目标大师核实本证据包后置 resolved/closed**（本文不代替判定）。
> 证据采集时间：2026-10-29 12:00–12:20 UTC（终验收口节点在本功能线分支上真机复验）。

## 0. 一页结论

| 项 | 内容 |
|----|------|
| Bug id | `cmvtfut20003om9sff0svg7kj`（project=mycourse，severity=medium，source=user，报告于 2026-10-29T11:06:53Z） |
| 缺陷本质 | 两处叠加：① `sample_token` 把 `(B>1,V)` 的采样结果硬转 python 标量 → `ValueError`（崩溃）；② `generate` 用单条 id 广播整个 batch（不崩但语义错误，所有序列同走向） |
| 修复 commit | `25aa828`（主修复）+ `ca81c6a`（一等公民加固），分支 `myrd/infer-lab-goal-cmvtaqxy7000pm9sfuz8zvm3i`；与 PR #2 的 `784db8e` 内容等价 |
| 判别力（本次复验） | 修复后测试放回修复前代码 `c7459b6`：**19 failed / 19 passed**（全部失败集中在批量用例） |
| 行为复现（本次复验） | `scripts/verify_b1_batch.py`：修复前 3 节全 FAIL（`ValueError`）、修复后 3 节全 PASS（exit 0） |
| 回归 | 修复时 160 passed；终验收口后 **191 passed**（solution）/ 117 passed + 74 xfailed（骨架态，无回归） |
| 修复红利 | 批量解码成为一等公民 → B>1 指标可测（`inference-baseline.json` batch 节）+ 对比笔记「批量生成的坑」一节 |

## 1. 缺陷复述（与 bug 单逐点对应）

报告基线：分支 `myrd/infer-lab-goal-cmvtaqxy7000pm9sfuz8zvm3i` @ `c7459b6`。

| # | bug 单描述 | 修复前代码（`c7459b6`） | 触发条件 |
|---|-----------|------------------------|----------|
| 1 | `sample_token()` 第 61 行 `return int(idx[..., 0])` 对 (B>1,V) 抛 `ValueError: only one element tensors can be converted to Python scalars` | `infer_lab/p4_sampling/solutions/sampling.py` 无条件把 multinomial 输出 `(B,1)` 转成 python `int` | 任何 B>1 的采样；单序列 `(V,)`/`(1,V)` 正常，故 153 个既有测试全绿抓不到 |
| 2 | `generate()` 第 38–44 行把单个标量 `next_id` 经 `torch.full` 广播给所有 batch 行 | `infer_lab/p4_generate/solutions/generate.py`：`torch.full((seq.shape[0], 1), next_id)` | 修好 #1 后必现：batch 内所有序列逐 token 相同，与 docstring 宣称的「各序列独立走向、全部命中 eos 才停」矛盾 |

bug 单给出的复现命令（`sample_token(torch.randn(2,50), top_k=50)` 抛错）在修复前可复现，
修复后同一输入返回形状 `(2,)` 的 LongTensor。

## 2. 修复落点（diff 范围 `c7459b6..ca81c6a`）

### `25aa828` 主修复（9 文件 +154/−14）

```python
# infer_lab/p4_sampling/solutions/sampling.py —— 契约扩展：按输入形状分派返回类型
  idx = torch.multinomial(probs, num_samples=1, generator=generator)
+ if logits.dim() >= 2 and logits.shape[0] > 1:
+     return idx[..., 0]   # (B,1) → (B,)：每行各自的 id
  return int(idx[..., 0])  # 单序列保持 python int（向后兼容）

# infer_lab/p4_generate/solutions/generate.py —— 广播缺陷：按行拼列
- next_id = sample_token(logits[:, -1, :], ...)
- seq = torch.cat([seq, torch.full((seq.shape[0], 1), next_id, ...)], dim=1)
+ next_ids = sample_token(logits[:, -1, :], ...)
+ if not torch.is_tensor(next_ids):          # B=1 路径仍是 python int
+     next_ids = torch.full((seq.shape[0],), next_ids, dtype=torch.long, ...)
+ column = next_ids.reshape(seq.shape[0], 1) # 每行用本行采出的 id
+ seq = torch.cat([seq, column], dim=1)
```

同步更新：骨架 docstring 的契约说明（`sampling.py`/`generate.py`）、两个模块 README、
`docs/solutions-guide.md`、新增批量测试。

### `ca81c6a` 一等公民加固（13 文件 +366/−9）

把「批量语义正确」升级为金标准性质测试：**批量输出 ≡ 逐条循环输出（逐 token 全等）**
（真实模型 + 确定性桩、`use_cache` 开/关双姿态）、随机采样每行只落在本行候选集、
温度/top-p 逐行生效，外加 KV cache 批量维度与按样本独立采样的 P3/P3b 用例。

## 3. 判别力实证（本次终验复核）

把**修复后的测试文件**放回**修复前代码**（`git worktree` @ `c7459b6`）运行：

```
$ INFER_LAB_TARGET=solution pytest tests/test_p4_sampling.py tests/test_p4_generate.py -q
19 failed, 19 passed
```

失败全部集中在批量用例（示例）：

```
FAILED test_sample_token_batch_returns_per_row_ids            ← 缺陷1 直接复现
FAILED test_batch_rows_follow_their_own_distribution          ← 缺陷2 判别器（广播必使全行相同）
FAILED test_batch_equals_per_row_loop_real_model_diverse_prompts[cache|nocache] ← 金标准对拍
FAILED test_batch_use_cache_true_matches_false
FAILED test_batch_eos_stops_only_when_all_rows_hit
...
```

19 个通过的是非批量用例（单序列向后兼容断言等）——失败/通过的切分线正好是
「批量 vs 单序列」，证明测试**真实区分**修复前后，不是恒红/恒绿的摆设。

## 4. 行为复现（本次终验复核）

`scripts/verify_b1_batch.py`（确定性桩 `RowDistinctStubModel`：行 r 第 s 步偏好
`(s+r) % V`，top_k=1 贪心可逐 token 预测）：

| 代码 | 节1 `sample_token((4,V))` | 节2 批量 `generate(B=4)` 各行互异 | 节3 批量 ≡ 逐条逐 token 全等 | 退出码 |
|------|---------------------------|----------------------------------|------------------------------|--------|
| 修复前 `c7459b6` | FAIL：`ValueError: only one element tensors can be converted to Python scalars` | FAIL：同上崩溃 | FAIL：崩溃无法对拍 | 1 |
| 修复后（本分支 HEAD） | PASS：返回 `(4,)` 逐行 argmax | PASS：各行新 token 互异 | PASS：`torch.equal` 全等 | 0 |

## 5. 全量回归（本次终验复核）

| 姿态 | 命令 | 结果 |
|------|------|------|
| 参考实现态 | `make test`（`INFER_LAB_TARGET=solution pytest -q`） | **191 passed**（演进：修复时 160 → 加固后 179 → 终验收口后 191，无回归） |
| 骨架态 | `make test-user`（`INFER_LAB_TARGET=user pytest -q`） | 117 passed + 74 xfailed（xfail 原因均为「待实现」，守护不回归） |
| 端到端 | `make e2e`（check-env → test → train-demo → generate-demo） | 全绿：loss 3.6221 → 0.1387，生成正常 |
| P1 冒烟 | `make smoke` | 通过，framework 槽位刷新且已回填数据保留（见 §6） |

## 6. 修复红利：批量从「会崩」到「可测、可对比」（终验收口新增）

修复前批量路径崩溃，B>1 的性能与语义都无法度量；修复后本轮收口把它变成了交付物：

1. **B>1 指标对拍**（`docs/metrics/inference-baseline.json` 顶层 `batch` 节，
   `scripts/measure_handrolled.py` 产出）：同 prompt 复制 B 行、greedy、各生成 24 token，
   两边 in-process 计时、每尺寸热身后重复 3 次取最小延迟 ——
   - 手搓引擎：B=1→16 整批吞吐 **421 → 7156 tokens/s（17.25×）**，整批延迟 ≈53ms 基本不变；
   - 框架（transformers）：B=1→16 **855 → 10841 tokens/s（12.47×）**；
   - 「批量摊薄每步固定开销」第一次有了本仓库自己的实测数字。
2. **手搓侧基线回填**（同文件 `handrolled` 槽位）：与 `framework` 槽位同协议
   （同 prompt/max_tokens/5 请求/greedy），B=1 吞吐 394.8 tokens/s，
   口径差（in-process 无 HTTP、字符级 tokenizer）写在 `protocol_delta`；
   `make smoke` 复写保护已验证（framework 刷新、handrolled/batch 逐字节保留）。
3. **经验沉淀**：[comparison-framework-vs-handrolled.md](comparison-framework-vs-handrolled.md)
   新增「**批量生成的坑**」一节 —— 标量聚合是 batch 杀手、广播让 bug 静默、
   单序列全绿 ≠ 批量正确、eos 早停语义、cache 批量维度、批量计时别被首跑骗。

## 7. 核实清单（供目标大师判定 resolved/closed）

- [ ] `git show 25aa828` / `git show ca81c6a`：两处修复分别落在 bug 单指认的两处缺陷代码行上；
- [ ] `git diff c7459b6..ca81c6a -- infer_lab/p4_sampling/solutions/sampling.py infer_lab/p4_generate/solutions/generate.py`：契约扩展 + 去广播；
- [ ] `.venv/bin/python scripts/verify_b1_batch.py`：退出码 0（三节 PASS）；
- [ ] 判别力：`git worktree add /tmp/prefix c7459b6` 后把当前 `tests/test_p4_*.py` 放进去跑 → 19 failed；
- [ ] `make test` 191 passed、`make test-user` 117 passed + 74 xfailed、`make e2e` 全绿；
- [ ] `docs/metrics/inference-baseline.json`：`handrolled` 槽位已回填、`batch` 节两侧齐全（守护测试 `tests/test_repo_artifacts.py` 常绿）。

> 关联记录：直通车节点首轮核查见
> [bugfix-verification-b1-batch.md](bugfix-verification-b1-batch.md)（修复当时点：
> 6/7 判别、160 passed、PR #2 状态）；本证据包为终验时点在功能线分支上的复验与收口。
