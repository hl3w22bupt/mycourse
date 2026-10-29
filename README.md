# infer-lab · LLM 推理引擎学习仓库

以学习为目的的 LLM 推理引擎实践项目，对标 **nanoGPT / minGPT** 的「小而完整」：
少量模块即可跑通 GPT 训练与推理，重在**读懂、复现、改写**。

- **先框架**（P1）：用 transformers 把「训练好的模型 → OpenAI 兼容服务 → 生成」跑通，建立基线；
- **后手搓**（P2-P5）：自建 tokenizer / embedding / attention(RoPE, GQA) / KV cache / 采样 /
  生成循环 / TinyGPT 训练闭环，与框架基线对拍验证正确性。

完整阶段规划见 **[ROADMAP.md](ROADMAP.md)**；框架版 vs 手搓版的深度对比见
**[docs/comparison-framework-vs-handrolled.md](docs/comparison-framework-vs-handrolled.md)**；
逐模块「骨架 ↔ 参考实现 ↔ 单测」对照清单见
**[docs/solutions-guide.md](docs/solutions-guide.md)**（由 `tests/test_solutions_alignment.py`
自动守护，接口漂移会挂测试）。

## 快速开始

```bash
# 0) 环境（Python ≥ 3.10，建议建 venv）
make install          # 安装 torch/numpy + transformers/fastapi + pytest
make check-env        # P0：环境自检报告

# 1) 全量测试（solutions 参考实现态，应全绿）
make test

# 2) 一键复现训练 + 生成闭环（秒级）
make train-demo       # 玩具语料训练，loss 明显下降
make generate-demo    # 加载 checkpoint 生成文本

# 3) P1 框架部署：单条命令起 OpenAI 兼容服务 + 冒烟 + 指标留档
make bootstrap        # 离线自建 tiny-gpt2（transformers 构建 + 玩具语料微调，幂等）
make smoke            # 起服务 → 生成非空断言 → 写框架侧指标记录 → 关服务（全离线可跑）
# 指标记录落在 docs/metrics/inference-baseline.json：framework 槽位自动写入，
# handrolled 槽位预留 —— 手搓引擎（P2-P5）对拍后按同口径回填
# 或手动：make serve   然后另开终端 curl http://127.0.0.1:8000/v1/completions ...

# 4) 端到端串联（环境 → 测试 → 训练 → 生成）
make e2e
```

> 网络说明：P1 演示默认使用 `make bootstrap` 离线自建的 tiny-gpt2
> （`models_cache/tiny-gpt2-toy`，105K 参数），**不需要访问 HuggingFace**。
> 想用官方权重时 `make serve TINY_MODEL=sshleifer/tiny-gpt2`，直连不通的环境
> 脚本会自动改走 `HF_ENDPOINT=https://hf-mirror.com` 镜像。

## 学习入口：两条路径

**路径 A · 用户视角（框架基线）**：读 [P1 README](infer_lab/p1_framework/README.md) →
`make smoke` → 对照 [对比笔记](docs/comparison-framework-vs-handrolled.md)。

**路径 B · 引擎视角（手搓主线）**：按 ROADMAP 顺序，每个模块三步走：

```bash
# ① 读该模块的 TODO（docstring 里有接口规格）
$EDITOR infer_lab/p3_attention/attention.py
# ② 用你的实现跑测试（user 态：未完成的用例 XFAIL 提示「待实现」）
INFER_LAB_TARGET=user .venv/bin/python -m pytest tests/test_p3_attention.py -q
# ③ 对照参考实现
$EDITOR infer_lab/p3_attention/solutions/attention.py
```

| 阶段 | 你要实现的文件 | 验收命令 |
|------|----------------|----------|
| P0 | `infer_lab/p0_env/env_check.py` | `make check-env` |
| P1 | `infer_lab/p1_framework/engine.py` | `make smoke` |
| P2 | `infer_lab/p2_tokenizer/tokenizer.py` | `pytest tests/test_p2_tokenizer.py` |
| P2b | `infer_lab/p2_embedding/embedding.py` | `pytest tests/test_p2_embedding.py` |
| P3 | `infer_lab/p3_attention/attention.py` | `pytest tests/test_p3_attention.py` |
| P3b | `infer_lab/p3_kv_cache/kv_cache.py` | `pytest tests/test_p3_kv_cache.py` |
| P4 | `infer_lab/p4_sampling/sampling.py` | `pytest tests/test_p4_sampling.py` |
| P4b | `infer_lab/p4_generate/generate.py` | `pytest tests/test_p4_generate.py` |
| P5 | `infer_lab/p5_toy/model.py` + `data.py` | `INFER_LAB_TARGET=user make train-demo` |

每个阶段目录都有自己的 `README.md`（学习目标 / 任务 / 验收命令）；
每个 TODO 桩都有 `solutions/` 参考实现，两者的公开接口逐参数一致
（`tests/test_solutions_alignment.py` 守护），照骨架写的代码可直接与参考实现互相替换。

## 命令总表

| 命令 | 作用 |
|------|------|
| `make install` | 安装全部依赖 |
| `make check-env` | P0 环境自检 |
| `make test` | 全量测试（solution 态，应全绿） |
| `make test-user` | 骨架态测试（TODO 未填 → XFAIL「待实现」） |
| `make train-demo` | P5 训练 demo（loss 下降 + 存 checkpoint） |
| `make generate-demo` | P5 生成 demo（加载 checkpoint） |
| `make bootstrap` | P1 离线自建 tiny-gpt2（幂等） |
| `make serve` | P1 起本地 OpenAI 兼容服务 |
| `make smoke` | P1 一键冒烟（起服务→请求→断言→收尾）+ 写框架侧指标记录 |
| `make e2e` | 一键串联：check-env → test → train-demo → generate-demo |

## 仓库结构

```
├── README.md                # 本文件（学习入口）
├── ROADMAP.md               # 分阶段路线图 + 勾稽表
├── Makefile                 # 全部命令入口
├── infer_lab/
│   ├── p0_env/              # P0 环境自检
│   ├── p1_framework/        # P1 框架部署（transformers + OpenAI 兼容服务）
│   ├── p2_tokenizer/        # P2 字符级 tokenizer
│   ├── p2_embedding/        # P2 词嵌入 + 位置嵌入
│   ├── p3_attention/        # P3 RoPE + 因果注意力（MHA/GQA）
│   ├── p3_kv_cache/         # P3 KV cache
│   ├── p4_sampling/         # P4 temperature/top-k/top-p
│   ├── p4_generate/         # P4 自回归生成循环
│   └── p5_toy/              # P5 TinyGPT 训练 + 生成闭环
├── tests/                   # 双模式测试（solution 全绿 / user 态 XFAIL 待实现）
│                            # + 骨架/产物守护（test_skeleton_todos / test_repo_artifacts）
│                            # + solutions↔骨架对齐守护（test_solutions_alignment）
├── scripts/                 # check-env / smoke 脚本
└── docs/                    # 对比笔记 + solutions 对照手册 + demo 运行记录 + 指标记录位（docs/metrics/）
```

## 约定与 FAQ

- **双模式测试**：默认 `make test` 跑参考实现；`INFER_LAB_TARGET=user` 时同一套测试切到
  你的 TODO 骨架 —— 未实现的用例 XFAIL（原因「待实现」），其余仍必须通过。
- **out of scope**：分布式训练、超大规模模型、生产级服务优化（量化调度、continuous batching）、
  自研算子内核。本项目只要「小而完整、随时可跑」。
- **生成有随机性**：采样受 seed 控制；`train-demo`/`generate-demo` 固定 seed=42，可复现。
- **冒烟没网？** P1 默认走 `make bootstrap` 的离线自建模型，无需下载；官方权重
  `sshleifer/tiny-gpt2` 只是可选项。
- **指标记录位怎么用？** `make smoke` 冒烟通过后自动把框架侧延迟 / 吞吐 /
  token 用量写进 [docs/metrics/inference-baseline.json](docs/metrics/inference-baseline.json)
  的 `framework` 槽位；`handrolled` 槽位预留，等 P2-P5 手搓引擎能按同口径
  （greedy、同 prompt / max_tokens / 请求数）出数后回填，两边逐字段对比。
  详见 [P1 README](infer_lab/p1_framework/README.md)。
