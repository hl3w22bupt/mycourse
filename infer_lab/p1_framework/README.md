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
make smoke    # 退出码 0，输出 SMOKE_OK，且生成文本非空
pytest tests/test_p1_framework.py -q   # 协议用例（桩引擎，不需要真模型）
```

想换真实模型/更高吞吐，可另起 vLLM（接口同为 OpenAI 兼容）：
```bash
pip install vllm && vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

## 参考实现
`infer_lab/p1_framework/solutions/engine.py`
