# demo 运行记录（端到端验收留档）

> 本文件由「端到端串联验收」节点在真实机器上执行各命令后留档，作为
> 需求验收标准 ③「框架路径本地完成端到端演示并留有运行记录」的证据。
>
> - 机器：macOS 26.3.1（Apple Silicon, arm64），MPS 可用
> - 环境：Python 3.13.12，torch 2.15.0，numpy 2.5.4，transformers 5.21.1，fastapi 0.162.0
> - 日期：2026-10-29
> - 所有命令均在仓库根目录、干净状态（删除 runs/ 与 models_cache/ 后）执行，可复现。

## 1. P0 环境自检 —— `make check-env`

```
$ make check-env
python: 3.13.12
torch: 2.15.0
numpy: 2.5.4
cuda: False
mps: True
transformers: 5.21.1
fastapi: 0.162.0
ok: True

环境 OK，可以开始 P1-P5 的学习。入口见 README.md
```

## 2. 全量测试（solutions 参考实现态）—— `make test`

```
$ make test
98 passed in 2.86s
```

骨架态（TODO 未填，用同一套测试切到 user 目标）——未实现用例全部 XFAIL 并提示「待实现」：

```
$ make test-user
50 passed, 48 xfailed in 1.47s
# XFAIL 示例：
# tests/test_p5_train_generate.py::test_training_loss_decreases_and_sample_nonempty
#   - 待实现（TODO 骨架态）：完成该模块后请改跑 `make test` 验证
```

## 3. P5 训练 demo —— `make train-demo`

```
$ make train-demo
STEP 0 loss 3.6221 val 3.6311
STEP 40 loss 1.7997 val 1.8058
STEP 80 loss 0.7723 val 0.8093
STEP 120 loss 0.3304 val 0.3253
STEP 160 loss 0.2051 val 0.1560
STEP 200 loss 0.1387 val 0.1348
LOSS_INITIAL 3.6221
LOSS_FINAL 0.1387
LOSS_IMPROVED yes
CHECKPOINT runs/toy/ckpt.pt
SAMPLE_TEXT the little engine could run. the little engine could write. a ti
```

loss 从 3.62 降到 0.14（≈ ln(vocab_size) 起步），验证集同步下降，无过拟合。

## 4. P5 生成 demo —— `make generate-demo`

```
$ make generate-demo
PROMPT the little engine
GENERATION the little engine could write. a tiny model can run. a tiny mode
```

## 5. P1 框架部署冒烟 —— `make smoke`（起服务 + 请求 + 断言一条龙）

首次运行自动离线自建模型（幂等，之后直接复用）：

```
SMOKE_BOOTSTRAP models_cache/tiny-gpt2-toy 不存在，先离线自建 tiny-gpt2 ……
BOOTSTRAP_CORPUS tokens=1344 unk_ratio=0.0000
BOOTSTRAP_STEP 0 loss 3.7844
BOOTSTRAP_STEP 100 loss 0.1847
BOOTSTRAP_STEP 200 loss 0.0795
BOOTSTRAP_STEP 300 loss 0.0618
BOOTSTRAP_LOSS 3.7844 -> 0.0618
BOOTSTRAP_OK saved to models_cache/tiny-gpt2-toy
```

服务冒烟（transformers 真实推理路径，OpenAI 兼容协议）：

```
SMOKE_WAIT_FOR_HEALTH http://127.0.0.1:8001
SMOKE_HEALTH {'status': 'ok', 'model': 'models_cache/tiny-gpt2-toy'}
SMOKE_COMPLETION_USAGE {'prompt_tokens': 4, 'completion_tokens': 24, 'total_tokens': 28}
SMOKE_COMPLETION_TEXT 'write . a tiny model can learn . a tiny model can grow . a tiny model can run . a tiny model can'
SMOKE_CHAT_TEXT '. . . the engine step by step . we build the model day by day . attention is all you need . attention'
SMOKE_OK 框架部署端到端通过（起服务 → 生成非空 → chat 通道）
```

说明：默认模型 `models_cache/tiny-gpt2-toy` 由 `make bootstrap` 用 transformers
（`GPT2Config` 从零构建 + 框架训练循环微调 300 步）离线产出，**不依赖访问
HuggingFace**；该环境 huggingface.co 不可达、hf-mirror 的文件下载会 308 回源，
因此离线自建是本仓库保证「一键可复现」的关键设计。

## 6. 端到端串联 —— `make e2e`

```
$ make e2e
（check-env 报告 → 98 tests passed → train-demo loss 3.6221 → 0.1387 → generate-demo）
✅ e2e 完成：环境自检 / 全量测试 / 训练 demo / 生成 demo 全部通过。
   P1 框架部署演示请另跑：make smoke（起服务+冒烟一条龙）
```

## 7. 本次验收期间修复的回归清单

| # | 回归 | 修复 |
|---|------|------|
| 1 | KV cache 路径位置嵌入未加 offset → 带缓存生成与全量重算不一致 | `Embedding.forward(idx, pos_offset)`；TinyGPT 传缓存长度（P3/P4 对拍测试锁定） |
| 2 | solution 相对导入指向自身/桩（`.model` 应为 `..model`） | 修正 import；solution 跨模块一律引用 solutions |
| 3 | transformers `labels` 内部已右移，训练循环再手动右移 → 双重位移，生成复读 | bootstrap 改为 `labels=x`，并在对比笔记中记录该差异 |
| 4 | 词表与预分词不一致 → 语料大面积 `[UNK]` | 词表从同一预分词器输出构建 + UNK 占比断言 |
| 5 | huggingface.co 不可达且镜像 308 回源 → P1 冒烟卡在模型下载 | `make bootstrap` 离线自建 tiny-gpt2，冒烟全离线可跑 |

## 8. 建仓验证记录（干净环境从零装依赖 → 测试可执行）

> 按任务「按路线图建仓：……测试可执行」的要求，在**全新虚拟环境**（删除 `.venv/`、
> `runs/`、`models_cache/` 后）从零执行依赖安装与全部验收命令，证明仓库
> clone 后即可复现。与前述记录不同的另一台/另一版环境：

- 环境：Python 3.11.15（uv 管理），macOS arm64（MPS 可用）
- 安装：`uv venv .venv --python 3.11` → `uv pip install -e ".[framework,dev]"`
- 装得版本：torch 2.15.0 / numpy 2.4.6 / transformers 5.21.1 / fastapi 0.162.0 / pytest
- 日期：2026-10-29

```
$ make test        # solutions 参考实现态
98 passed in 7.62s

$ make test-user   # TODO 骨架态
50 passed, 48 xfailed in 1.58s   # xfail 原因均为「待实现」

$ make e2e         # check-env → test → train-demo → generate-demo
STEP 0 loss 3.6221 val 3.6311
STEP 200 loss 0.1387 val 0.1348
LOSS_IMPROVED yes
CHECKPOINT runs/toy/ckpt.pt
SAMPLE_TEXT the little engine could run. the little engine could write. a ti
GENERATION the little engine could think. the little engine could think. th
✅ e2e 完成：环境自检 / 全量测试 / 训练 demo / 生成 demo 全部通过。

$ make smoke       # P1 框架部署：bootstrap 离线自建 → 起服务 → 冒烟断言
BOOTSTRAP_LOSS 3.7844 -> 0.0619
SMOKE_COMPLETION_TEXT 'write . a tiny model can learn . a tiny model can grow . a tiny model can run . a tiny model can'
SMOKE_OK 框架部署端到端通过（起服务 → 生成非空 → chat 通道）
```

结论：路线图勾稽表（ROADMAP.md）9 项全部核验通过，双模式测试、训练/生成闭环与
P1 框架部署冒烟在干净环境下均可执行。

## 9. P1 指标记录位 —— `make smoke` 冒烟通过后写入框架侧指标（2026-10-29 追加）

「一条命令拉起本地推理服务并冒烟通过，留框架侧指标记录位供对比」节点：`make smoke`
在原有「起服务 → 生成断言 → chat 通道」之后，按对拍协议（greedy、同 prompt、同
`max_tokens`、5 次请求）计时补打请求，把框架侧延迟 / 吞吐写进
[metrics/inference-baseline.json](metrics/inference-baseline.json) 的 `framework` 槽位；
`handrolled` 槽位预留，待手搓引擎（P2-P5）按同口径回填。

干净状态（`models_cache/` 不存在）真机运行，一条命令全离线走完 bootstrap → 服务 → 冒烟 → 留档。

首跑（干净状态，自动触发离线自建，幂等）：

```
$ make smoke
SMOKE_BOOTSTRAP models_cache/tiny-gpt2-toy 不存在，先离线自建 tiny-gpt2 ……
BOOTSTRAP_MODEL params=105920 device=mps
BOOTSTRAP_CORPUS tokens=1344 unk_ratio=0.0000
BOOTSTRAP_LOSS 3.7844 -> 0.0614
BOOTSTRAP_OK saved to models_cache/tiny-gpt2-toy
SMOKE_OK 框架部署端到端通过（起服务 → 生成非空 → chat 通道 → 指标留档）
```

复跑（模型已缓存，生成最终入库的记录；每次冒烟都会刷新 `framework` 槽位）：

```
$ make smoke
SMOKE_HEALTH {'status': 'ok', 'model': 'models_cache/tiny-gpt2-toy'}
SMOKE_COMPLETION_TEXT 'write . a tiny model can learn . a tiny model can grow . a tiny model can run . a tiny model can'
SMOKE_CHAT_TEXT '. . . the engine step by step . we build the model day by day . attention is all you need . attention'
SMOKE_METRICS_PROTOCOL {'prompt': 'the little engine could', 'max_tokens': 24, 'requests': 5, 'sampling': 'greedy(temperature=0)', 'latency_view': 'client-side(含 HTTP 往返)', ...}
SMOKE_METRICS_REQUEST {'index': 0, 'prompt_tokens': 4, 'completion_tokens': 24, 'latency_ms': 53.52, 'tokens_per_s': 448.43}
SMOKE_METRICS_REQUEST {'index': 1, 'prompt_tokens': 4, 'completion_tokens': 24, 'latency_ms': 44.5, 'tokens_per_s': 539.33}
SMOKE_METRICS_REQUEST {'index': 2, 'prompt_tokens': 4, 'completion_tokens': 24, 'latency_ms': 44.94, 'tokens_per_s': 534.05}
SMOKE_METRICS_REQUEST {'index': 3, 'prompt_tokens': 4, 'completion_tokens': 24, 'latency_ms': 40.07, 'tokens_per_s': 598.95}
SMOKE_METRICS_REQUEST {'index': 4, 'prompt_tokens': 4, 'completion_tokens': 24, 'latency_ms': 35.05, 'tokens_per_s': 684.74}
SMOKE_METRICS_SUMMARY {'requests': 5, 'prompt_tokens': 4, 'completion_tokens_per_request': 24, 'completion_tokens_total': 120, 'latency_ms_avg': 43.62, 'latency_ms_p50': 44.5, 'latency_ms_min': 35.05, 'latency_ms_max': 53.52, 'tokens_per_s': 550.26}
SMOKE_METRICS_SAVED docs/metrics/inference-baseline.json（framework 槽位已写入；handrolled 槽位预留，待手搓引擎对拍回填）
SMOKE_OK 框架部署端到端通过（起服务 → 生成非空 → chat 通道 → 指标留档）
```

- 环境：Python 3.11.15（uv 管理），torch 2.15.0 / transformers 5.21.1 / fastapi 0.162.0，
  macOS arm64（MPS），全离线。
- 记录要点：105,920 参数 tiny-gpt2 @ mps，单请求补全 24 token，p50 延迟 ≈44.5ms、
  聚合吞吐 ≈550 tokens/s（客户端视角，含 HTTP 往返）。
- 新增测试：`tests/test_p1_metrics.py`（口径与结构，桩注入）+
  `tests/test_repo_artifacts.py` 守护「framework 已填 / handrolled 预留」。
