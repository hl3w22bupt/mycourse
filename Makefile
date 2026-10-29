# infer-lab 一键复现入口
# 用法：make <target>   （不带参数执行 `make help` 查看全部目标）
PY       ?= .venv/bin/python
PIP      ?= .venv/bin/python -m pip
HF_ENDPOINT ?= https://hf-mirror.com   # huggingface.co 直连不通时自动走镜像，可覆盖
TINY_MODEL ?= models_cache/tiny-gpt2-toy   # P1 默认模型（离线自建，可换任意 hub 模型名）
METRICS_OUT ?= docs/metrics/inference-baseline.json   # P1 冒烟写入的框架侧指标记录位

.PHONY: help install check-env bootstrap test test-user test-all \
        train-demo generate-demo serve smoke e2e clean

help:
	@echo "infer-lab 常用命令："
	@echo "  make install        安装依赖（含 framework/dev extras）"
	@echo "  make check-env      P0 环境自检"
	@echo "  make test           全量测试（solutions 参考实现态，应全绿）"
	@echo "  make test-user      骨架态测试（TODO 未填 → 全部 xfail『待实现』）"
	@echo "  make train-demo     P5 玩具语料训练 demo（秒级，loss 下降）"
	@echo "  make generate-demo  加载 checkpoint 生成文本 demo"
	@echo "  make bootstrap      P1 离线自建 tiny-gpt2（默认模型，幂等）"
	@echo "  make serve          P1 拉起 OpenAI 兼容推理服务（transformers 后端）"
	@echo "  make smoke          P1 冒烟：起服务→请求→断言非空生成→写框架侧指标记录→关服务"
	@echo "  make e2e            一键复现：check-env → test → train-demo → generate-demo"

install:
	$(PIP) install -e ".[framework,dev]"

check-env:
	$(PY) scripts/check_env.py

test:
	INFER_LAB_TARGET=solution $(PY) -m pytest -q

# 骨架态：把测试目标切到 TODO 桩。未实现用例全部 xfail（原因：待实现），其余用例仍须通过。
test-user:
	INFER_LAB_TARGET=user $(PY) -m pytest -q

train-demo:
	$(PY) -m infer_lab.p5_toy.train --steps 200 --d-model 64 --n-layer 2 \
		--block-size 64 --batch-size 16 --device auto \
		--out-dir runs/toy --log-every 40

generate-demo:
	$(PY) -m infer_lab.p5_toy.generate --ckpt runs/toy/ckpt.pt \
		--prompt "the little engine" --max-new-tokens 80 \
		--temperature 0.8 --top-k 20

bootstrap:
	HF_ENDPOINT=$(HF_ENDPOINT) $(PY) scripts/bootstrap_tiny_model.py --out-dir $(TINY_MODEL)

serve: bootstrap
	HF_ENDPOINT=$(HF_ENDPOINT) $(PY) -m infer_lab.p1_framework.server \
		--model $(TINY_MODEL) --host 127.0.0.1 --port 8000

smoke:
	HF_ENDPOINT=$(HF_ENDPOINT) $(PY) scripts/smoke_framework.py \
		--model $(TINY_MODEL) --port 8001 --metrics-out $(METRICS_OUT)

e2e: check-env test train-demo generate-demo
	@echo ""
	@echo "✅ e2e 完成：环境自检 / 全量测试 / 训练 demo / 生成 demo 全部通过。"
	@echo "   P1 框架部署演示请另跑：make smoke（起服务+冒烟一条龙）"
