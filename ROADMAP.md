# infer-lab 分阶段学习路线图

> 需求基准：《LLM 推理引擎学习项目》——范围对标 nanoGPT/minGPT 的「小而完整」，
> 部署策略「**先框架、后手搓**」，任何时点仓库都有可运行版本。
>
> **不在范围内**（out of scope）：分布式/多机训练、超大规模模型、生产级推理服务优化
> （量化调度、continuous batching 等）、自研算子内核。

## 总览

| 阶段 | 名称 | 一句话目标 | 代码目录 | 阶段 README | 验收命令 |
|------|------|-----------|----------|-------------|----------|
| P0 | 环境与工程基线 | 环境可复现、命令有入口 | `infer_lab/p0_env` | [README](infer_lab/p0_env/README.md) | `make check-env` |
| P1 | 框架部署与推理基线 | transformers 拉起 OpenAI 兼容服务 | `infer_lab/p1_framework` | [README](infer_lab/p1_framework/README.md) | `make smoke` |
| P2 | tokenizer | 文本 ↔ id 序列契约 | `infer_lab/p2_tokenizer` | [README](infer_lab/p2_tokenizer/README.md) | `pytest tests/test_p2_tokenizer.py` |
| P2b | embedding | 词向量 + 位置信息 | `infer_lab/p2_embedding` | [README](infer_lab/p2_embedding/README.md) | `pytest tests/test_p2_embedding.py` |
| P3 | attention | RoPE + 因果掩码 + GQA | `infer_lab/p3_attention` | [README](infer_lab/p3_attention/README.md) | `pytest tests/test_p3_attention.py` |
| P3b | KV cache | 逐 token 解码的显存账本 | `infer_lab/p3_kv_cache` | [README](infer_lab/p3_kv_cache/README.md) | `pytest tests/test_p3_kv_cache.py` |
| P4 | 采样 | temperature / top-k / top-p | `infer_lab/p4_sampling` | [README](infer_lab/p4_sampling/README.md) | `pytest tests/test_p4_sampling.py` |
| P4b | 生成循环 | 自回归解码主循环 | `infer_lab/p4_generate` | [README](infer_lab/p4_generate/README.md) | `pytest tests/test_p4_generate.py` |
| P5 | 训练+生成闭环 | TinyGPT 小数据训练出文本 | `infer_lab/p5_toy` | [README](infer_lab/p5_toy/README.md) | `make train-demo` |

## 各阶段详情

### P0 环境与工程基线
- **学习目标**：可复现环境；学会先自检再动手。
- **产出物**：`scripts/check_env.py` + `env_check` 模块（TODO → solution）、Makefile 命令面。
- **DoD**：`make check-env` 打印报告且退出码 0；`pytest tests/test_p0_env.py` 全绿。

### P1 框架部署与推理基线（先框架）
- **学习目标**：用成熟框架跑通「加载 → 推理 → 服务化」，建立可对拍的基线。
- **产出物**：`FrameworkEngine`（transformers 后端）+ FastAPI OpenAI 兼容服务 + 冒烟脚本
  + `make bootstrap` 离线自建 tiny-gpt2（框架训练循环演示，不依赖网络）
  + 指标记录位（`metrics.py` + `GET /v1/engine/info`：冒烟通过后写
  `docs/metrics/inference-baseline.json` 的 `framework` 槽位，`handrolled` 槽位留待 P5 后
  由 `make metrics-handrolled` 按同口径回填）。
- **DoD**：`make serve` + `make smoke` 单机可复现，冒烟生成文本非空，指标记录
  `framework` 槽位有数；
  `pytest tests/test_p1_framework.py` 协议用例全绿（桩引擎，无需真模型）。

### P2-P4 手搓核心（后手搓）
- **学习目标**：逐模块手写推理引擎核心，每个模块都有「性质测试」锁正确性。
- **产出物**：tokenizer / embedding / attention(RoPE, GQA) / KV cache / 采样 / 生成循环。
- **DoD**：对应单测全绿；三条核心性质（因果掩码、RoPE 相对性、缓存对拍全量重算）成立。

### P5 训练 + 生成闭环
- **学习目标**：把手搓零件组装成 TinyGPT，玩具语料上完成训练与生成。
- **产出物**：TinyGPT 模型 + 数据流水线 + 训练/生成 CLI + checkpoint。
- **DoD**：`make train-demo` loss 明显下降（≥200 步内）并生成样例文本；
  `make generate-demo` 从 checkpoint 复现生成。

## 路线图 ↔ 代码骨架勾稽表（对照清单）

- [x] P0 → `infer_lab/p0_env`（`env_check.py` 桩 + `solutions/env_check.py`）—— 验收：`make check-env`
- [x] P1 → `infer_lab/p1_framework`（`engine.py` 桩 + `server.py`/`metrics.py` 胶水 + `solutions/engine.py`；指标记录落 `docs/metrics/inference-baseline.json`）—— 验收：`make smoke`
- [x] P2 → `infer_lab/p2_tokenizer`（`tokenizer.py` 桩 + solution）—— 验收：`pytest tests/test_p2_tokenizer.py`
- [x] P2b → `infer_lab/p2_embedding`（`embedding.py` 桩 + solution）—— 验收：`pytest tests/test_p2_embedding.py`
- [x] P3 → `infer_lab/p3_attention`（`attention.py` 桩 + solution）—— 验收：`pytest tests/test_p3_attention.py`
- [x] P3b → `infer_lab/p3_kv_cache`（`kv_cache.py` 桩 + solution）—— 验收：`pytest tests/test_p3_kv_cache.py`
- [x] P4 → `infer_lab/p4_sampling`（`sampling.py` 桩 + solution）—— 验收：`pytest tests/test_p4_sampling.py`
- [x] P4b → `infer_lab/p4_generate`（`generate.py` 桩 + solution）—— 验收：`pytest tests/test_p4_generate.py`
- [x] P5 → `infer_lab/p5_toy`（`model.py`/`data.py` 桩 + `corpus.py` + `train.py`/`generate.py` 工作脚本 + solution）—— 验收：`make train-demo`

无悬空项：每个阶段都有对应目录、README、验收命令；每个桩模块都有 `solutions/` 参考实现
与 `tests/` 用例（默认跑 solution，`INFER_LAB_TARGET=user` 切骨架态）。
模块级「骨架 ↔ 参考实现 ↔ 单测」对照清单见
[docs/solutions-guide.md](docs/solutions-guide.md)；勾稽与对齐由两套守护测试自动把关：
`tests/test_repo_artifacts.py`（文档/指标产物）与 `tests/test_solutions_alignment.py`
（成对存在、公开接口与签名逐参数一致、solution 无残留桩、胶水不设 solution）。
以上勾稽项已全部在真实环境逐条核验通过（全量测试全绿、骨架态未实现用例全部 XFAIL
「待实现」、`make e2e` 与 `make smoke` 端到端可跑），运行记录见
[docs/demo-run-log.md](docs/demo-run-log.md)。

## 推荐学习路径

1. `make check-env` → 确认环境（P0）。
2. `make smoke` → 先玩框架基线，对「一条 prompt 进、一段文本出」有体感（P1）。
3. 按 P2 → P2b → P3 → P3b → P4 → P4b 顺序手搓，每个模块：读 docstring 的 TODO →
   实现 → `pytest tests/test_<模块>.py -q` 跑绿；卡住或做完都翻
   [docs/solutions-guide.md](docs/solutions-guide.md) 的对照清单找参考实现比对。
4. `INFER_LAB_TARGET=user make train-demo` → 用**你自己的**实现跑训练闭环（P5）。
5. 读 [docs/comparison-framework-vs-handrolled.md](docs/comparison-framework-vs-handrolled.md)
   把手搓版与框架版逐维度对拍，回头重看 P1 的 `model.generate` 参数。
