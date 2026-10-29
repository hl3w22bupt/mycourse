"""P1 - 框架推理引擎封装（transformers 后端）。

TODO(学习者)：实现 ``FrameworkEngine``。接口保持与真实服务解耦，
``server.py`` 只认 ``complete`` / ``chat`` / ``model_name`` 三个方法 ——
这样测试可以用桩引擎替换，不必加载真模型。

要求：
1. ``__init__(model_name="models_cache/tiny-gpt2-toy", device=None, seed=42)``：
   只存参数不加载（懒加载，import 快、测试友好）。默认模型是 ``make bootstrap``
   离线自建的本地目录；本地没有时可回退 Hub 模型（如 ``sshleifer/tiny-gpt2``）。
2. ``load()``：``AutoTokenizer`` + ``AutoModelForCausalLM``；
   device 顺序 cuda → mps → cpu；模型放 eval 模式。
3. ``complete(prompt, max_new_tokens=32, temperature=0.8, top_k=50, top_p=0.95)``
   → dict：``{"text", "prompt_tokens", "completion_tokens"}``。
   用 ``model.generate(do_sample=temperature>0, ...)``；返回的 text 要**剥掉 prompt**。
4. ``chat(messages, ...)``：把 OpenAI 风格 messages（``[{"role","content"}]``）
   模板化成单条 prompt（``Role: content`` 逐行 + ``Assistant:`` 结尾）后走 ``complete``。
5. ``model_name`` 属性。

验收：``pytest tests/test_p1_framework.py`` 全绿（用桩引擎测协议，不需要真模型）；
``make smoke`` 用真模型端到端验证。
"""

from __future__ import annotations


class FrameworkEngine:
    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 FrameworkEngine.__init__")

    def load(self) -> None:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 FrameworkEngine.load")

    @property
    def model_name(self) -> str:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 model_name")

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> dict:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 FrameworkEngine.complete")

    def chat(self, messages: list[dict], **kwargs) -> dict:
        raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 FrameworkEngine.chat")


def render_chat_prompt(messages: list[dict]) -> str:
    """最小 chat 模板：tiny-gpt2 没有 chat template，手工拼。"""
    raise NotImplementedError("待实现：请按模块 docstring 的 TODO 完成 render_chat_prompt")
