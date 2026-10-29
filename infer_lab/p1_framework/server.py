"""P1 - OpenAI 兼容推理服务（FastAPI）。

这是**直接可跑的胶水层**（不是 TODO）：路由只依赖引擎的
``complete`` / ``chat`` / ``model_name`` 三个方法，引擎本体在
``engine.py``（那是 TODO）。

启动：
    make serve
    # 等价于
    .venv/bin/python -m infer_lab.p1_framework.server --model sshleifer/tiny-gpt2 --port 8000

冒烟：
    make smoke
"""

from __future__ import annotations

import argparse
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..testing import load_impl


# ---------- 请求模型（OpenAI 兼容字段的一个实用子集） ----------
class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


def _cmpl_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def create_app(engine=None) -> FastAPI:
    """构建 FastAPI 应用。``engine`` 缺省时按当前 target 装载引擎实现。"""
    if engine is None:
        engine = load_impl("p1_framework", "engine").FrameworkEngine()

    app = FastAPI(title="infer-lab P1 OpenAI-compatible server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_name}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": engine.model_name, "object": "model", "owned_by": "infer-lab"}],
        }

    def _payload(text: str, prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        result = engine.complete(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        return {
            "id": _cmpl_id(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.model_name,
            "choices": [
                {"index": 0, "text": result["text"], "finish_reason": "length", "logprobs": None}
            ],
            "usage": _payload(result["text"], result["prompt_tokens"], result["completion_tokens"]),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        result = engine.chat(
            [m.model_dump() for m in req.messages],
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        return {
            "id": _cmpl_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "length",
                    "logprobs": None,
                }
            ],
            "usage": _payload(result["text"], result["prompt_tokens"], result["completion_tokens"]),
        }

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="infer-lab P1 OpenAI 兼容服务")
    p.add_argument("--model", default="models_cache/tiny-gpt2-toy",
                   help="本地 bootstrap 模型目录，或任意 Hub 模型名")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="cpu / mps / cuda，默认自动")
    args = p.parse_args()

    engine = load_impl("p1_framework", "engine").FrameworkEngine(
        model_name=args.model, device=args.device
    )
    import uvicorn

    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
