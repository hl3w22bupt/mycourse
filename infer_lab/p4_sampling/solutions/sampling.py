"""P4 参考实现：temperature / top-k / top-p 采样。

学习要点（三条策略各自回答一个问题）：
- temperature：**多大胆？** logits/T。T→0 趋近贪心（确定性），T>1 更发散。
  实现上 T 很小时先减 max 再除，防止上溢。
- top-k：**候选几个？** 只留分数前 k 个。缺点是 k 固定，不看分布形状。
- top-p（nucleus）：**候选多大一堆？** 留到「累计概率 ≥ p」的最小集合，
  分布尖锐时候选少、分布平缓时候选多 —— 自适应候选数。
- 过滤一律用 ``-inf`` 而不是 0：要在 softmax **之前**置零概率。
- 批量（B>1）：三条过滤策略都只动最后一维、天然按行独立；``sample_token``
  对 ``(B, V)`` 输入返回每行一个 id 的 ``(B,)`` 张量 —— 各行用**本行**分布采样。
"""

from __future__ import annotations

import torch


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0（要贪心请用 temperature→0 的极小值或 top_k=1）")
    scaled = logits / temperature
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values  # 数值稳定
    return torch.softmax(scaled, dim=-1)


def top_k_filtering(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    vocab = logits.shape[-1]
    k = min(top_k, vocab)
    thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]  # 第 k 大的分数
    return logits.masked_fill(logits < thresh, float("-inf"))


def top_p_filtering(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 区间（=1.0 表示不过滤）")
    if top_p == 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # 移除累计概率**超过** top_p 的位置；「首次达到 top_p 的那个」必须保留
    mask = cumulative - probs > top_p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int | torch.Tensor:
    """完整采样链：temperature → top-k → top-p → softmax → 多项式采样。

    返回值按输入形状分两种（向后兼容）：
    - 单序列 ``(...)`` 或 ``(1, V)``：返回 python ``int``（最后一个维度上的选择）；
    - 批量 ``(B, V)`` 且 ``B > 1``：返回形状 ``(B,)`` 的 ``LongTensor``，
      每行用**本行**过滤后的分布独立采样 —— 各行 id 可以不同。
    """
    filtered = top_p_filtering(top_k_filtering(logits, top_k), top_p)
    probs = softmax_with_temperature(filtered, temperature)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    if logits.dim() >= 2 and logits.shape[0] > 1:
        return idx[..., 0]  # (B, 1) → (B,)：每行各自的 id
    return int(idx[..., 0])
