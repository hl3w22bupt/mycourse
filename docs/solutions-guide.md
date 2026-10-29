# solutions 对照手册：逐模块「照做 → 验收 → 对照」

> 这份清单回答两个问题：**照做**——我在哪个文件里写代码、跑哪条命令验收；
> **对照**——我的实现和参考实现差在哪、哪些性质必须成立。
> 逐模块的骨架 ↔ solution ↔ 单测对应关系由 `tests/test_solutions_alignment.py`
> 自动守护（接口漂移会直接挂测试），本手册是它的可读版。

## 1. 三类文件，各司其职

| 角色 | 位置 | 谁来写 | 说明 |
|------|------|--------|------|
| **TODO 骨架** | `infer_lab/<stage>/<module>.py` | 学习者照 docstring 的 TODO 实现 | 所有公开接口都先以 `raise NotImplementedError("待实现：…")` 占位 |
| **参考实现** | `infer_lab/<stage>/solutions/<module>.py` | 仓库提供，供对照 | docstring 写「学习要点」；默认 `make test` 跑的就是它 |
| **胶水工作代码** | 如 `p1_framework/server.py`、`p5_toy/train.py` | 不用写，直接可跑 | 内部经 `load_impl` 按当前 target 装载骨架/参考实现，所以**有意不设 solutions 副本** |

胶水清单（见到它们不要找 solution，也不要在里面布置 TODO）：
`p1_framework/metrics.py`、`p1_framework/server.py`、`p5_toy/corpus.py`、
`p5_toy/train.py`、`p5_toy/generate.py`。

## 2. 模块对照清单（骨架 ↔ solution ↔ 单测）

| 阶段 | 照做：骨架文件 | 对照：参考实现 | 单测 | 核心性质 / 考点 | 验收命令 |
|------|----------------|----------------|------|-----------------|----------|
| P0 | `infer_lab/p0_env/env_check.py` | `solutions/env_check.py` | `tests/test_p0_env.py` | 依赖缺失不崩（返回 `None`），`ok` 只看 torch+numpy | `make check-env` |
| P1 | `infer_lab/p1_framework/engine.py` | `solutions/engine.py` | `tests/test_p1_framework.py`、`tests/test_p1_metrics.py` | 返回 text 剥掉 prompt；懒加载；greedy 时不传采样参数 | `make smoke` |
| P2 | `infer_lab/p2_tokenizer/tokenizer.py` | `solutions/tokenizer.py` | `tests/test_p2_tokenizer.py` | 编解码往返；未登录 → `<unk>`；`max_len` 截断后仍包 special | `pytest tests/test_p2_tokenizer.py` |
| P2b | `infer_lab/p2_embedding/embedding.py` | `solutions/embedding.py` | `tests/test_p2_embedding.py` | 同 token 不同位置可区分；`arange` 跟 `idx.device`；越界抛 `ValueError` | `pytest tests/test_p2_embedding.py` |
| P3 | `infer_lab/p3_attention/attention.py` | `solutions/attention.py` | `tests/test_p3_attention.py` | **因果性**；**RoPE 相对性**；**GQA**；**缓存对拍 ≡ 全量重算** | `pytest tests/test_p3_attention.py` |
| P3b | `infer_lab/p3_kv_cache/kv_cache.py` | `solutions/kv_cache.py` | `tests/test_p3_kv_cache.py` | `update` 返回全量可打分；`with kv:` 进出即复位；越界 `IndexError` | `pytest tests/test_p3_kv_cache.py` |
| P4 | `infer_lab/p4_sampling/sampling.py` | `solutions/sampling.py` | `tests/test_p4_sampling.py` | 过滤用 `-inf` 且在 softmax 之前；top-k=1 ≡ argmax；同 seed 可复现；批量 `(B>1,V)` 返回每行一个 id 的张量（单序列保持 python int） | `pytest tests/test_p4_sampling.py` |
| P4b | `infer_lab/p4_generate/generate.py` | `solutions/generate.py` | `tests/test_p4_generate.py` | **use_cache=True ≡ False**（含批量）；贪心 ≡ 手写 argmax 循环；每行采**各自**的 id（不广播）；全 batch 命中 eos 提前停 | `pytest tests/test_p4_generate.py` |
| P5 | `infer_lab/p5_toy/model.py` + `data.py` | `solutions/model.py` + `solutions/data.py` | `tests/test_p5_train_generate.py` | 窗口右移一位当标签；loss 下降；checkpoint 存取往返 | `INFER_LAB_TARGET=user make train-demo` |

## 3. 照做五步法（每个模块都一样）

```bash
# ① 读骨架 docstring 的 TODO（那里是唯一规格：接口、口径、易错点都写全了）
$EDITOR infer_lab/p3_attention/attention.py
# ② user 态跑单测：没实现的用例 XFAIL「待实现」，已实现的必须过
INFER_LAB_TARGET=user .venv/bin/python -m pytest tests/test_p3_attention.py -q
# ③ 实现你的版本
# ④ 用你的实现跑绿同一套测试（这一步过了 = 达标）
INFER_LAB_TARGET=user .venv/bin/python -m pytest tests/test_p3_attention.py -q   # 无 XFAIL
# ⑤ 对照参考实现：看学习要点注释，重点比对你没写的防御性细节
$EDITOR infer_lab/p3_attention/solutions/attention.py
# 最后用 solution 态全量回归，确认没改坏任何共享件
make test
```

对照时优先看三样东西：**边界校验**（越界/非法参数要早失败）、**device 处理**
（`arange`/张量别写死 CPU）、**口径自洽**（返回值要能由记录的字段复算）。

## 4. 骨架规格已经补齐的易错点（照做时容易被坑的地方）

以下口径此前只写在参考实现/单测里，现已回写进骨架 docstring——照骨架实现即可通过单测：

1. **P1 engine**：`__init__` 默认模型是本地 `models_cache/tiny-gpt2-toy`（`make bootstrap`
   的产物），不是 Hub 模型；可选元信息属性 `engine_kind` / `param_count` / `library_versions`
   供 `GET /v1/engine/info` 与指标记录用。
2. **P2 tokenizer**：需要 `vocab` 属性（词表副本）——P5 训练脚本存 checkpoint 时要整表写入。
3. **P2b embedding**：`forward` 带 `pos_offset=0`——KV cache 逐步解码时位置要从已缓存长度
   继续编号，否则 P3/P5 带缓存生成整体错位。
4. **P3 attention**：输出投影 `out_proj`（打分加权求和 reshape 回 `(B, T, n_embd)` 后再过线性）。
5. **P3b KVCache**：`len(kv) == seq_len`；`update`/`get` 越界抛 `IndexError`。
6. **P5 model/data**：`TinyGPT` 暴露 `config` 与 `block_size`（生成脚本按它 clamp 总长）；
   `CharDataset` 暴露 `tokenizer` / `vocab_size` / `ids` / `block_size`。

## 5. 对齐由谁守护

`tests/test_solutions_alignment.py` 四条口径，任何一条被破坏都会挂测试：

1. **成对存在**：每个骨架文件都有 solution，每个模块都有专属单测文件；
2. **接口一致**：solution 必须暴露骨架声明的全部公开符号，且签名逐参数一致
   （参数名/顺序/种类/默认值）——保证「照骨架写的代码」与「参考实现」可互相替换跑同一套测试；
3. **solution 干净**：参考实现不允许残留 `raise NotImplementedError` 桩；
4. **胶水定位**：胶水文件不许出现待实现桩或 TODO，也不设 solutions 副本。
