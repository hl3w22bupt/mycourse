# P0 - 环境与工程基线

## 学习目标
- 建立可复现的本地环境：锁版本依赖、统一入口命令（Makefile）。
- 学会用「环境自检」脚本先回答"我能不能跑"，再开始学习。

## 任务
1. `make install` 安装依赖（torch/numpy 为核心，transformers/fastapi 为 P1 可选）。
2. 实现 `env_check.py` 中的 `check_env` / `format_report`（TODO 见模块 docstring）。
3. 让 `make check-env` 打印环境报告且退出码为 0。

## 验收命令
```bash
make check-env          # 打印 python/torch/numpy/cuda/mps 报告
pytest tests/test_p0_env.py -q   # P0 用例全绿
```

## 参考实现
`infer_lab/p0_env/solutions/env_check.py`
