# 对比笔记：框架部署（P1）vs 手搓引擎（P2-P5）

> 这份笔记回答一个问题：**同一条 prompt 变成一段文本，框架版和手搓版分别发生了什么？**
> 结论先行：两者在数学上做的是同一件事（tokenizer → embedding → attention → 采样 → 解码），
> 差异全部在「封装边界」上 —— 而这正是学习价值所在。

## 1. 结构对比

| 维度 | 框架版（transformers，P1） | 手搓版（infer-lab，P2-P5） |
|------|---------------------------|---------------------------|
| 模块划分 | `AutoTokenizer` / `AutoModelForCausalLM` / `generate()` 三件套 | tokenizer → embedding → attention → KV cache → sampling → 生成循环，六层显式拆开 |
| 调用面 | 一次 `model.generate(**inputs, ...)` 出全部结果 | 循环里每步「前向一层 → 采样一个 token」 |
| 代码规模 | 千行级（含设备管理、量化、flash-attn 分支） | 每模块几十行，行行可读 |
| 学习视角 | 黑盒：参数生效但看不见过程 | 白盒：每个张量形状都能 print 出来看 |

## 2. KV cache 对比

| 维度 | 框架版 | 手搓版 |
|------|--------|--------|
| 触发方式 | `use_cache=True`（generate 默认开启），返回 `past_key_values` | 自己实现 `KVCache` / `past_kv` 元组，逐层 `update` |
| 存储布局 | `DynamicCache`/tuple 嵌套结构（版本间变过多次） | 每层一个 `(k, v)`，形状 `(B, n_kv_head, S, head_dim)`，一目了然 |
| 位置编码 | 框架内部自动维护 `cache_position`/RoPE offset | 必须自己把 offset 传给位置嵌入（`pos_offset`）与 RoPE —— **这是手搓最容易错的点** |
| 显存账本 | 只能查文档估算 | `2 × n_layers × n_kv_head × S × head_dim × B × bytes` 直接可算，GQA 的收益一算便知 |

## 3. 延迟路径对比

**框架版一次请求的路径**：
`HTTP → FastAPI/pydantic 校验 → tokenizer → generate()（内部循环：embedding → n×block → logits →
采样 → 停止判定）→ decode → HTTP`。延迟大头在 generate 的黑盒循环里，可用
`performance_metrics`/TTFT 记录器观测，但循环本身改不了。

**手搓版同一路径**：每个环节都是你自己写的那几行。能亲手做的延迟实验：
1. `use_cache=False` 逐 token 重新前向整段历史 → 生成 100 token 的耗时随长度平方增长；
2. 切 `use_cache=True` → 每步只前向 1 个 token，耗时线性；
3. 把 `n_kv_head` 从 4 减到 2（GQA）→ cache 变小、显存下降（速度差异在 CPU/小模型上不明显，
   显存账本直接可算）。

> P3/P4 的测试把这三件事写成了「对拍」：**缓存版与全量重算版输出必须逐 token 一致** ——
> 加速不能改变结果，这是推理引擎正确性的底线。
>
> 实测数字有留档：`make smoke` 冒烟通过后按对拍协议（greedy、同 prompt、同 max_tokens、
> 5 次请求）把框架版延迟 / 吞吐写进
> [metrics/inference-baseline.json](metrics/inference-baseline.json) 的 `framework` 槽位；
> `handrolled` 槽位由 `make metrics-handrolled`（`scripts/measure_handrolled.py`）按
> **同口径**（同 prompt / 同 max_tokens / 同请求数 / greedy）回填，两边即可逐字段对比。
> 注意一处口径差：框架侧延迟含 HTTP 往返，手搓侧是 in-process（无服务），绝对值不可直接
> 比，差异写死在记录的 `handrolled.protocol_delta` 里；真正可比的批量数字见顶层
> `batch` 节（两边都 in-process，见第 6 节）。

## 4. 控制粒度对比

| 控制点 | 框架版 | 手搓版 |
|--------|--------|--------|
| 采样参数 | `temperature/top_k/top_p` 进 `generate()` | 自己的 `sample_token(logits, ...)`，想加 repetition penalty / min_p 只需改一处 |
| 停止条件 | `eos_token_id` / `stopping_criteria` 列表 | 循环里一行 `finished |= seq[:, -1] == eos_id`，想加最大长度、字符串停止符都直接改 |
| 中途干预 | logits_processor（注册式，较绕） | logits 就在手里，直接改 |
| 流式输出 | `TextStreamer` 回调 | 每步循环体就是天然的流式挂点 |

框架版赢在「快、稳、兼容 OpenAI 协议」；手搓版赢在「每一步都可解释、可改造」。

## 5. 推荐顺序（照做即可）

1. **P0**：`make check-env` 确认环境。
2. **P1 先跑框架基线**：`make bootstrap`（离线自建 tiny-gpt2：transformers 构建 + 框架训练
   循环微调 300 步，本身就是「框架版训练」的对照组）→ `make smoke`。记下冒烟输出里的
   `usage`（prompt_tokens/completion_tokens）——之后手搓版要能对上这个语义；
   冒烟还会把框架侧延迟 / 吞吐自动写进
   [metrics/inference-baseline.json](metrics/inference-baseline.json)。
3. **P2 tokenizer** → `pytest tests/test_p2_tokenizer.py -q` 绿。
4. **P2b embedding** → 重点读懂 `pos_offset` 注释（生成错位的头号坑）。
5. **P3 attention**：先过 `build_rope_cache`/`apply_rope`，再写 `CausalSelfAttention`；
   四条性质测试就是验收标准（因果、RoPE 相对性、GQA≈MHA、缓存对拍）。
6. **P3b KVCache** → 显存账本会算了，GQA 的意义就清楚了。
7. **P4 采样 → 生成循环**：跑通 `use_cache` on/off 对拍。
8. **P5 闭环**：`INFER_LAB_TARGET=user make train-demo`，loss 下降、生成出像样的句子即毕业；
   随后 `make metrics-handrolled` 把手搓版按同口径回填进指标记录（含 B>1 批量对拍）。
9. **回头重读 P1**：再看 `model.generate` 的参数表，你会发现每个参数都能对应到手搓版的某几行。

## 6. 批量生成的坑（广播 / 维度语义）

> 这一节来自一次真实缺陷（批量生成崩溃 + 静默广播，修复证据见
> [bugfix-evidence-cmvtfut20003om9sff0svg7kj.md](bugfix-evidence-cmvtfut20003om9sff0svg7kj.md)）：
> 单序列全绿的代码，一开 batch 就崩 —— 或者更糟：**不崩，但 batch 里所有序列生成同一段文本**。

两版批量解码的维度约定先对齐：

| 环节 | 框架版 `model.generate` | 手搓版 `generate` |
|------|------------------------|-------------------|
| 输入 | `(B, T)`，B>1 必须配 `attention_mask`（不等长要 pad） | `(B, T)` 裸张量，教学版约定等长 batch |
| 每步 logits | `(B, vocab)`，HF 内部取 last position | 手动 `logits[:, -1, :]` |
| 采样 | logits_processor 全程带 batch 维 | `sample_token` 必须支持 `(B,V) → (B,)` |
| 早停 | 逐行 `unfinished_sequences` 跟踪 + pad 填充 | 教学版「全部行命中 eos 才停」 |

- **坑 1：标量聚合是 batch 杀手**。`int(idx[..., 0])` 把 multinomial 输出硬转 python
  标量 —— 输入 `(B>1, V)` 时 `idx` 是 `(B, 1)`，`int()` 直接抛
  `ValueError: only one element tensors can be converted to Python scalars`。
  维度语义上：温度/top-k/top-p 过滤只动**最后一维**，天然按行独立、不会错；真正的
  batch 杀手是**聚合成本值**那一步。修复后的契约：单序列返回 `int`，
  `(B>1, V)` 返回 `(B,)` LongTensor —— 调用方要同时接住两种返回类型。
- **坑 2：广播让 bug 静默**。修好坑 1 之后还有更隐蔽的一层：`torch.full((B, 1), next_id)`
  把一个标量 next token 广播给整个 batch —— 不报错、loss 正常、输出「看着像生成的」，
  但 batch 内所有序列逐 token 相同。判别器是金标准对拍：**批量输出 ≡ 逐条循环输出
  （逐 token 全等）**，且测试要用「各行偏好互异」的确定性桩 —— 广播缺陷必然使全行
  相同，一测即红（`tests/test_p4_generate.py` 的
  `test_batch_rows_follow_their_own_distribution` 干的就是这件事）。
- **坑 3：单序列全绿 ≠ 批量正确**。B=1 时两条路径恰好重合（标量），上述两个缺陷
  全都抓不到。批量是一等公民：新特性从第一天起就要有 B>1 用例，`scripts/verify_b1_batch.py`
  可以随时复检（修复前 3 节全 FAIL、修复后全 PASS）。
- **坑 4：eos 早停语义不同**。框架版逐行跟踪 `unfinished`，先停的行被 pad 填充；
  手搓教学版「全部命中才停」。等长 batch（同 prompt 复制 B 份）下「批量 ≡ 逐条」严格
  成立；不等长 batch 手搓版没做 padding，语义会和框架版分叉 —— 想补齐就去做
  padding + attention_mask，那是生产版的第一课。
- **坑 5：KV cache 的批量维度**。cache 形状 `(B, n_kv_head, S, head_dim)`，B 必须全程
  一致；`pos_offset` 与 cache 长度在批量路径下要同一套对齐（`tests/test_p3_kv_cache.py`
  有批量维度专项用例）。
- **坑 6：批量计时别被首跑骗**。每个新 batch 形状有一次性内核编译开销（实测能把首跑
  拖慢一个数量级），且小模型单次耗时抖动大 —— `scripts/measure_handrolled.py` 每个
  尺寸先热身、重复 3 次取**最小延迟**。实测形状留在
  [metrics/inference-baseline.json](metrics/inference-baseline.json) 的 `batch` 节：
  B=1→16 整批吞吐 手搓 421→7156 tokens/s（**17.25×**）、框架 855→10841 tokens/s
  （**12.47×**），而整批延迟几乎不变（手搓 ≈53ms、框架 ≈28–35ms）——
  这就是「批量摊薄每步固定开销」的实测形状，也是为什么生产引擎都要 continuous batching。

## 7. 常见坑（两版都会踩）

- **labels 位移语义不同（框架版专属大坑）**：手搓版里 `x`/`y` 要自己右移一位
  （P5 `get_batch` 的 `y = window[:, 1:]`），cross_entropy 直接对齐；而 transformers 的
  `model(input_ids=x, labels=x)` 会在**内部再右移一位** —— 外面再手动移一次就变成
  双重位移，模型悄悄学会「预测后两个 token」，loss 照样收敛、生成全是复读
  （`scripts/bootstrap_tiny_model.py` 的注释就是为这个坑立的碑）。
- **位置错位**：缓存路径忘了给位置嵌入/RoPE 加 offset → 生成文本重复、胡言乱语（embedding 的
  `pos_offset` 与 attention 的 `offset` 必须同时处理）。
- **词表与分词不一致**：建词表时剥掉标点、分词时却不剥（或反之），训练数据会大面积变
  `[UNK]`，模型学到的下一个 token 就是 UNK。词表必须从**同一个预分词器**的实际输出构建。
- **过滤时机**：top-k/top-p 在 softmax 之后做就晚了（要在 logits 上置 `-inf`）。
- **cache 串请求**：服务端复用 KVCache 忘了 `reset()`，上一位用户的上下文漏给下一位。
- **`-inf` 与温度**：过滤后的 `-inf` 除以极小温度仍是 `-inf`，softmax 后概率为 0，这是对的；
  但 `temperature=0` 会直接除零 —— 贪心请走 `top_k=1`（或框架版 `do_sample=False`）。
- **批量路径**：标量聚合 / 广播 / eos 早停三连坑，专列一节 —— 见上文第 6 节。
