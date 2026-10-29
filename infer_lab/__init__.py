"""infer-lab：以学习为目的的 LLM 推理引擎实验仓库。

两条主线（先框架、后手搓）：
- P1 框架路径：``infer_lab.p1_framework`` —— transformers 后端 + OpenAI 兼容服务。
- 手搓路径：``infer_lab.p2_*`` → ``infer_lab.p5_toy`` —— tokenizer / embedding /
  attention(RoPE, GQA) / KV cache / 采样 / 生成循环 / 玩具训练闭环。

每个模块都有：
- TODO 骨架（学习者在 ``infer_lab/<stage>/<mod>.py`` 里照 TODO 实现）；
- 参考实现（``infer_lab/<stage>/solutions/<mod>.py``）；
- 单测（``tests/``，默认跑参考实现，``INFER_LAB_TARGET=user`` 切到骨架态）。
"""

__version__ = "0.1.0"
