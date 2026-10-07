"""Live HTTP round-trip for every adapter in `src/agent/llm/`.

A local socket server plays the provider: it records the request each
adapter actually sent (headers, tool schema, messages) and answers with
that provider's real response shape. This exercises the parts unit
tests skip — `_build_body` → transport → retries → `_parse`.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from agent.llm.anthropic import AnthropicProvider
from agent.llm.gemini import GeminiProvider
from agent.llm.ollama import OllamaProvider
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.normalizer import NormalizedResponse
from agent.llm.provider import FinishReason, Message, ToolSpec
from agent.llm.runtime import get_llm_registry, reset_llm_registry

SPEC = ToolSpec(
    name="note_tool",
    description="Store a short note",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

# Canned responses, one per provider's native wire format.
OPENAI_PAYLOAD = {
    "choices": [{
        "finish_reason": "tool_calls",
        "message": {
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "note_tool",
                             "arguments": "{\"text\": \"hello\"}"},
            }],
        },
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7},
}
ANTHROPIC_PAYLOAD = {
    "content": [
        {"type": "text", "text": "noting that"},
        {"type": "tool_use", "id": "toolu_1", "name": "note_tool",
         "input": {"text": "hello"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 5, "output_tokens": 7},
}
GEMINI_PAYLOAD = {
    "candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [
            {"functionCall": {"name": "note_tool", "args": {"text": "hello"}}}
        ]},
    }],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7},
}
OLLAMA_PAYLOAD = {
    "model": "llama3",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "note_tool", "arguments": {"text": "hello"}}}
        ],
    },
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 5,
    "eval_count": 7,
}


class _Handler(BaseHTTPRequestHandler):
    requests: List[Dict[str, Any]] = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {"_unparsed": raw.decode("utf-8", "replace")}
        type(self).requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "x_api_key": self.headers.get("x-api-key", ""),
            "body": body,
        })

        payload = None
        if self.path.endswith("/chat/completions"):
            payload = OPENAI_PAYLOAD
        elif self.path.endswith("/v1/messages"):
            payload = ANTHROPIC_PAYLOAD
        elif ":generateContent" in self.path:
            payload = GEMINI_PAYLOAD
        elif self.path.endswith("/api/chat"):
            payload = OLLAMA_PAYLOAD

        if payload is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "not found"}')
            return
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):  # keep test output clean
        return


@pytest.fixture()
def fake_provider_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _config(base_url: str, **extra) -> Dict[str, Any]:
    # Fail fast in tests: one attempt, short timeout.
    return {"api_key": "test-key", "base_url": base_url,
            "timeout": 5.0, "max_retries": 0, **extra}


@pytest.mark.asyncio
async def test_openai_adapter_round_trip(fake_provider_url):
    provider = OpenAICompatibleProvider(_config(fake_provider_url))
    await provider.start()
    try:
        resp = await provider.chat(
            [Message(role="user", content="hi")],
            model="vendor/custom", tools=[SPEC],
        )
    finally:
        await provider.close()

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [
        ("note_tool", {"text": "hello"})
    ]
    assert resp.usage["total_tokens"] == 12

    sent = _Handler.requests[-1]
    assert sent["authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "vendor/custom"
    assert sent["body"]["tools"][0]["function"]["name"] == "note_tool"
    assert sent["body"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_anthropic_adapter_round_trip(fake_provider_url):
    provider = AnthropicProvider(_config(fake_provider_url))
    await provider.start()
    try:
        resp = await provider.chat(
            [Message(role="user", content="hi")],
            model="claude-test", tools=[SPEC],
        )
    finally:
        await provider.close()

    assert resp.content == "noting that"
    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [
        ("note_tool", {"text": "hello"})
    ]

    sent = _Handler.requests[-1]
    assert sent["x_api_key"] == "test-key"
    # Anthropic's tool schema uses input_schema, not parameters.
    assert sent["body"]["tools"][0]["input_schema"]["required"] == ["text"]
    assert sent["body"]["max_tokens"] > 0


@pytest.mark.asyncio
async def test_gemini_adapter_round_trip(fake_provider_url):
    provider = GeminiProvider(_config(fake_provider_url))
    await provider.start()
    try:
        resp = await provider.chat(
            [Message(role="user", content="hi")],
            model="gemini-test", tools=[SPEC],
        )
    finally:
        await provider.close()

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [
        ("note_tool", {"text": "hello"})
    ]

    sent = _Handler.requests[-1]
    assert sent["path"].endswith("/models/gemini-test:generateContent")
    decls = sent["body"]["tools"][0]["function_declarations"]
    assert decls[0]["name"] == "note_tool"


@pytest.mark.asyncio
async def test_ollama_adapter_round_trip(fake_provider_url):
    provider = OllamaProvider(_config(base_url=fake_provider_url, api_key=""))
    await provider.start()
    try:
        resp = await provider.chat(
            [Message(role="user", content="hi")],
            model="llama3", tools=[SPEC],
        )
    finally:
        await provider.close()

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [
        ("note_tool", {"text": "hello"})
    ]
    assert resp.usage["prompt_tokens"] == 5

    sent = _Handler.requests[-1]
    assert sent["path"].endswith("/api/chat")
    assert sent["body"]["tools"][0]["function"]["name"] == "note_tool"
    assert sent["body"]["options"]["num_predict"] > 0


@pytest.mark.asyncio
async def test_registry_completes_through_real_adapter(fake_provider_url):
    """Adapters → normalizer, with a real adapter doing real HTTP."""
    reset_llm_registry()
    try:
        registry = get_llm_registry()
        registry.install_provider(
            "custom",
            OpenAICompatibleProvider(_config(fake_provider_url)),
            api_key="test-key",
            model="vendor/custom",
            metadata={"provider": "openai-compatible", "context_window": 8192},
        )
        provider = registry.providers["custom"]
        await provider.start()
        try:
            out = await registry.complete(
                [Message(role="user", content="hi")], tools=[SPEC],
            )
        finally:
            await provider.close()

        assert isinstance(out, NormalizedResponse)
        assert out.finish_reason == FinishReason.TOOL_CALLS.value
        assert [(c.name, c.arguments) for c in out.tool_calls] == [
            ("note_tool", {"text": "hello"})
        ]
        # provider is the adapter's own name, not the install key.
        assert out.provider == "openai-compatible"
        assert out.model == "vendor/custom"
        assert out.usage.prompt_tokens == 5
        assert out.usage.completion_tokens == 7
        assert out.latency_ms >= 0
    finally:
        reset_llm_registry()
