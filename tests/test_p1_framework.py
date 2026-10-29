"""P1 框架服务测试：用桩引擎验证 OpenAI 兼容协议（不下载真模型）。"""

from __future__ import annotations

import pytest

from infer_lab.p1_framework.solutions.engine import render_chat_prompt

fastapi_testclient = pytest.importorskip("fastapi.testclient")


class StubEngine:
    """与 FrameworkEngine 同接口的桩：返回固定文本，记录调用参数。"""

    def __init__(self):
        self.model_name_value = "stub-tiny-gpt2"
        self.last_call: dict = {}

    @property
    def model_name(self) -> str:
        return self.model_name_value

    def load(self) -> None:  # pragma: no cover - 桩无需加载
        pass

    def complete(self, prompt: str, **kwargs) -> dict:
        self.last_call = {"prompt": prompt, **kwargs}
        return {"text": " stub output text", "prompt_tokens": 4, "completion_tokens": 3}

    def chat(self, messages: list[dict], **kwargs) -> dict:
        return self.complete(render_chat_prompt(messages), **kwargs)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from infer_lab.p1_framework.server import create_app

    engine = StubEngine()
    return TestClient(create_app(engine)), engine


def test_health(client):
    tc, _ = client
    r = tc.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "stub-tiny-gpt2"}


def test_list_models(client):
    tc, _ = client
    r = tc.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and body["data"][0]["id"] == "stub-tiny-gpt2"


def test_completions_openai_shape(client):
    tc, _ = client
    r = tc.post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 16, "temperature": 0.5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "stub-tiny-gpt2"
    assert body["choices"][0]["text"] == " stub output text"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}


def test_completions_forwards_sampling_params(client):
    tc, engine = client
    tc.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": 8, "temperature": 0.2, "top_k": 5, "top_p": 0.9},
    )
    call = engine.last_call
    assert call["max_new_tokens"] == 8
    assert call["temperature"] == 0.2
    assert call["top_k"] == 5
    assert call["top_p"] == 0.9


def test_chat_completions_messages_are_templated(client):
    tc, engine = client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi there"},
            ],
            "max_tokens": 12,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == " stub output text"
    assert engine.last_call["prompt"] == (
        "System: be brief\nUser: hi there\nAssistant:"
    )


def test_missing_prompt_is_422(client):
    tc, _ = client
    r = tc.post("/v1/completions", json={"max_tokens": 8})
    assert r.status_code == 422


def test_chat_prompt_renderer():
    prompt = render_chat_prompt(
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
    )
    assert "User: 你好" in prompt and prompt.endswith("Assistant:")
