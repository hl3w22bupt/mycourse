"""P1 参考实现：transformers 推理引擎封装。

学习要点：
- 框架路径的「推理」只有四步：tokenizer 编码 → model.generate → 解码 → 剥掉 prompt。
  KV cache、采样过滤、停止条件全都藏在 ``generate`` 的几十个参数里 —— 这正是
  P3/P4 要手搓的东西，学习时可以两相对照。
- transformers v4/v5 兼容：``do_sample``+``top_k/top_p`` 的组合在两个大版本间
  语义一致；``pad_token_id`` 显式指定避免 GPT-2 类模型的告警。
"""

from __future__ import annotations

import os

import torch

DEFAULT_LOCAL_MODEL = "models_cache/tiny-gpt2-toy"   # `make bootstrap` 的产物（离线可复现）
DEFAULT_HUB_MODEL = "sshleifer/tiny-gpt2"            # 本地没有时回退到 Hub（需网络）


def _pick_device(preference: str | None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_chat_prompt(messages: list[dict]) -> str:
    """最小 chat 模板：``Role: content`` 逐行拼接，以 ``Assistant:`` 收尾等待续写。"""
    lines = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


class FrameworkEngine:
    """懒加载的 transformers 引擎：import 秒开，首次 ``load``/推理才拉模型。"""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self._requested_name = model_name
        self._device_pref = device
        self._seed = seed
        self.tokenizer = None
        self.model = None
        self.device = _pick_device(device)

    @property
    def model_name(self) -> str:
        return self._requested_name

    # ---- 引擎元信息（指标记录 /v1/engine/info 用，见 metrics.py）----
    @property
    def engine_kind(self) -> str:
        return "transformers"

    @property
    def param_count(self) -> int | None:
        """模型参数量；懒加载未触发时为 ``None``（该端点不主动拉模型）。"""
        if self.model is None:
            return None
        return int(sum(p.numel() for p in self.model.parameters()))

    @property
    def library_versions(self) -> dict:
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__}

    def _resolve_source(self) -> str:
        """本地 bootstrap 模型缺失时回退 Hub 模型（有网环境可直接用官方 tiny-gpt2）。"""
        if os.path.exists(self._requested_name) or os.path.isdir(self._requested_name):
            return self._requested_name
        print(f"ENGINE_FALLBACK {self._requested_name} 不存在，改用 Hub 模型 {DEFAULT_HUB_MODEL}")
        return DEFAULT_HUB_MODEL

    def load(self) -> None:
        """加载 tokenizer 与模型（幂等：二次调用直接返回）。"""
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = self._resolve_source()
        torch.manual_seed(self._seed)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        self.model = AutoModelForCausalLM.from_pretrained(source)
        self.model.to(self.device).eval()

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> dict:
        self.load()
        torch.manual_seed(self._seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4) if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return {
            "text": text,
            "prompt_tokens": prompt_len,
            "completion_tokens": int(completion_ids.shape[0]),
        }

    def chat(self, messages: list[dict], **kwargs) -> dict:
        return self.complete(render_chat_prompt(messages), **kwargs)
