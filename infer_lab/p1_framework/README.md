# P1 - 框架部署与推理基线（transformers + OpenAI 兼容服务）

## 学习目标
- 用成熟框架把「模型加载 → 推理 → 服务化」端到端跑通，建立可工作的基线。
- 理解 OpenAI 兼容协议的最小请求/响应面（`/v1/completions`、`/v1/chat/completions`、`/v1/models`）。
- 知道 `model.generate` 里藏了什么（KV cache、采样、停止条件），为 P3/P4 手搓做对照。

## 任务
1. 实现 `engine.py` 的 `FrameworkEngine`（懒加载 transformers，`complete`/`chat`）。
2. `server.py` 直接可用（胶水层，无需改动）：`make serve` 拉起服务。
3. `make smoke` 一条命令完成「起服务 → 探活 → 生成非空 → 收尾」。

> **离线友好**：默认模型 `models_cache/tiny-gpt2-toy` 由 `make bootstrap` 离线自建——
> 用 transformers 从 `GPT2Config` 构建 + 在玩具语料上微调 300 步（顺带演示框架训练循环），
> 不依赖任何网络。若目录缺失，`make serve` 会自动先 bootstrap；想用官方权重也可以
> `make serve TINY_MODEL=sshleifer/tiny-gpt2`（需要能访问 HuggingFace，脚本会用
> `HF_ENDPOINT` 镜像兜底）。

## 验收命令
```bash
make smoke    # 退出码 0，输出 SMOKE_OK，且生成文本非空 + 指标记录已写入
pytest tests/test_p1_framework.py -q   # 协议用例（桩引擎，不需要真模型）
pytest tests/test_p1_metrics.py -q     # 指标口径与记录结构（桩注入，不需要真模型）
```

想换真实模型/更高吞吐，可另起 vLLM（接口同为 OpenAI 兼容）：
```bash
pip install vllm && vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

## 指标记录位（框架侧基线，供对拍）

`make smoke` 在冒烟断言**全部通过**后，按对拍协议（greedy 采样、同 prompt、同
`max_tokens`、默认 5 次请求）再计时补打几发补全，把框架侧指标写进
[docs/metrics/inference-baseline.json](../../docs/metrics/inference-baseline.json)：

| 字段 | 含义 |
|------|------|
| `protocol` | 对拍口径：prompt / max_tokens / 请求数 / greedy / 延迟视角（客户端含 HTTP 往返） |
| `framework.summary` | 平均 / p50 延迟、聚合吞吐 `tokens_per_s`、token 用量 |
| `framework.requests` | 每请求的 `prompt_tokens` / `completion_tokens` / `latency_ms` / `tokens_per_s` |
| `framework.sample_text` | 首条 greedy 生成样本（两边的文本可以直接比对） |
| `handrolled` | 手搓引擎（P2-P5）按同口径回填的对应字段（`make metrics-handrolled`；未回填时各字段 null + `todo` 说明）。`protocol_delta` 写明与框架侧的口径差（in-process 无 HTTP、字符级 tokenizer） |

实现要点：

- **只有冒烟通过的运行才写记录**：生成断言失败直接退出、不落盘，避免留下脏基线；
- 采样固定 greedy（`temperature=0`）、延迟为客户端视角 —— 口径写进 `protocol` 字段，
  两边（框架 / 手搓）用同一协议出数才有可比性；
- 元信息（device / 参数量 / torch & transformers 版本）来自服务端
  `GET /v1/engine/info`（infer-lab 扩展端点，非 OpenAI 协议字段；引擎没暴露的属性自动缺席）；
- 记录随仓库提交留档，`tests/test_repo_artifacts.py` 有守护测试锁定
  「framework 槽位已填、handrolled 槽位按同口径回填、batch 节两侧齐全」的形态（schema v2）。

对拍回填方式：`make train-demo && make metrics-handrolled`（`scripts/measure_handrolled.py`）
按 `protocol` 同口径跑同样的请求，自动把结果回填进 `handrolled` 槽位（含
`protocol_delta` 口径差说明），并在同 prompt / greedy 下对拍 B=1/2/4/8/16 的整批吞吐
写进顶层 `batch` 节。**不要动 `framework` 槽位**——那是框架基线，重跑 `make smoke`
会自动刷新它；且刷新时 `write_record` 会保留已回填的 `handrolled` / `batch` 数据。

## 参考实现
`infer_lab/p1_framework/solutions/engine.py`
