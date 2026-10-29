# 核查记录：B>1 批量生成缺陷修复（bug 修复 run `cmvtfvq97003rm9sfuwj4slje`）

> 核查时间：2026-10-29 11:07–11:35 UTC。核查人：直通车验证节点。
> 结论先行：**四点全部通过**，无遗留兜底项。

## 0. 结论总表

| # | 核实点 | 结论 | 关键证据 |
|---|--------|------|----------|
| ① | run completed 且 PR 已创建 | ✅ 通过 | run `completed`（11:31:44Z），dev/review/pr 三节点全 completed；[PR #2](https://github.com/hl3w22bupt/mycourse/pull/2) open |
| ② | diff 覆盖两处缺陷 | ✅ 通过 | commit `784db8e`：`sample_token` 批量分支 + `generate` 按行拼列 |
| ③ | 新增 B>1 批量单测有效 | ✅ 通过 | 新增 7 个测试；放回修复前代码 6/7 FAIL（判别有效） |
| ④ | 全量回归 + 批量复现 | ✅ 通过 | 修复后 `pytest` 160 passed（基线 153）；复现脚本三节 PASS（修复前三节 FAIL） |

## 1. 缺陷与修复对照（核实点②）

修复 commit：`784db8e fix(infer-lab): 批量生成缺陷修复——sample_token 支持 (B>1,V) 逐行采样 + generate 不再广播单条 id`
（基于功能线分支 `c7459b6` 之上，diff 范围 `c7459b6..784db8e`：9 文件 +154/-14）

| 缺陷 | 修复前位置与行为 | 修复方式 |
|------|------------------|----------|
| 1. `sample_token` 对 (B>1,V) logits 抛 `ValueError: only one element tensors can be converted to Python scalars` | `infer_lab/p4_sampling/solutions/sampling.py:61` 无条件 `return int(idx[..., 0])`，(B>1,V) 下 `idx[..., 0]` 是多元素张量 | 契约扩展：`(B,V)` 且 `B>1` 返回每行一个 id 的 `(B,)` LongTensor；`(V,)`/`(1,V)` 保持 python `int`（向后兼容）。骨架 docstring / README / solutions-guide 同步更新契约 |
| 2. `generate` 把单条 id 广播给全 batch | `infer_lab/p4_generate/solutions/generate.py:44` `torch.full((seq.shape[0], 1), next_id)` —— 各序列被强制相同走向 | `sample_token` 批量返回 `(B,)` → `reshape(B,1)` 按行拼列；仅单序列路径（返回 int）才 `torch.full` 展开。批量下每行从**本行**分布采样 |

判定依据：两处修复都落在缺陷代码行上；`B=1` 路径行为不变（有向后兼容测试锁定）；eos「全 batch 命中才停」语义保留并有新测试覆盖。

## 2. 批量单测判别力（核实点③）

新增 7 个测试（`tests/test_p4_sampling.py` +4、`tests/test_p4_generate.py` +3）。

**判别力验证**：把修复后的测试文件放回**修复前**代码（`c7459b6`）上运行——

```
6 failed, 17 passed
FAILED test_sample_token_batch_returns_per_row_ids          (ValueError 复现)
FAILED test_sample_token_batch_respects_filters_per_row
FAILED test_sample_token_batch_reproducible_with_same_seed
FAILED test_batch_rows_follow_their_own_distribution        (广播判别器)
FAILED test_batch_eos_stops_only_when_all_rows_hit
FAILED test_batch_use_cache_true_matches_false
```

唯一通过的是 `test_sample_token_single_sequence_stays_int`（向后兼容断言，修复前本就该过）→ 断言真实区分修复前后，**判别有效**。其中 `test_batch_rows_follow_their_own_distribution` 用两行偏好互异的测试桩直接锁定广播缺陷（各行新 token 必须互异）。

## 3. 全量回归与批量复现（核实点④）

| 阶段 | 命令 | 结果 |
|------|------|------|
| 修复前基线 | `INFER_LAB_TARGET=solution pytest -q`（`c7459b6`） | 153 passed |
| 修复后全量 | 同上（`784db8e`） | **160 passed**（= 153 + 新增 7，无回归） |
| 修复前复现 | `.venv/bin/python scripts/verify_b1_batch.py` | 三节全 FAIL：节1/节2 崩于 `ValueError`，节3 无法对拍 |
| 修复后复现 | 同上 | **三节全 PASS**（退出码 0） |

复现脚本 `scripts/verify_b1_batch.py` 三节内容（确定性测试桩 `RowDistinctStubModel`，行 r 第 s 步偏好 `(s+r) % V`，`top_k=1` 贪心可逐 token 预测）：

1. `sample_token((4,V))` 不再抛 ValueError，贪心逐行采样 = 各行 argmax；
2. 批量 `generate(B=4)` 不崩溃且各行新 token 互异（广播缺陷必使全行相同）；
3. 批量输出与逐条循环生成**逐行全等**：

```
批量: [[0,1,2,3,0,1,2,3...] 行 r 偏好序列 (s+r)%V
逐条: 同上 —— torch.equal 全等
```

> 备注：首轮脚本 FAIL 系核查脚本自身桩缺陷（`logits[:, :, ch]` 漏行索引导致各行并列 top-1，采样随机；且逐条对拍未带行偏移），修正桩后复跑定性。被测代码无此问题。

## 4. 评审与 PR 状态（核实点①）

- run `cmvtfvq97003rm9sfuwj4slje`：`completed` @ 2026-10-29T11:31:44Z，`pull → dev → review → pr` 全 completed。
- 评审结论（review 节点）：两组测试与提交声明一致（160 passed / 105 passed + 55 xfailed），e2e 通过（loss 下降、生成闭环正常），无危险模式（无 eval/exec、无 shell=True、无硬编码密钥，服务默认仅绑 127.0.0.1）。
- [PR #2](https://github.com/hl3w22bupt/mycourse/pull/2) `open`：head `myrd/run-cmvtfvq97003rm9sfuwj4slje`（= `784db8e`）→ base `main`。
- 关联：[PR #1](https://github.com/hl3w22bupt/mycourse/pull/1) `open`（功能线分支 `myrd/infer-lab-goal-cmvtaqxy7000pm9sfuz8zvm3i` → `main`）。run 分支 = 功能线分支 `c7459b6` + 修复 `784db8e`，两分支同源，合入互不冲突。

## 5. 证据路径与复现命令

| 证据 | 路径 / 命令 |
|------|-------------|
| 修复 diff | `git diff c7459b6..784db8e`（远端分支 `myrd/run-cmvtfvq97003rm9sfuwj4slje`） |
| 批量复现脚本 | `scripts/verify_b1_batch.py`（修复前 FAIL / 修复后 PASS） |
| 判别力验证 | 修复后 tests 放回 `c7459b6` 代码：6 failed, 17 passed |
| 全量回归 | `INFER_LAB_TARGET=solution .venv/bin/python -m pytest -q` → 160 passed |
| 平台侧 run 详情 | workflow `cmvqcronn0000m9tvn79cnf2k` / run `cmvtfvq97003rm9sfuwj4slje` |
