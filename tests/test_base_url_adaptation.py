"""Base URLs adapt to whatever the user pastes.

Two layers:

  1. `normalize_base_url` / `plan_runtime` — a bare host, a full endpoint
     ("/chat/completions"), or a versioned prefix ("/v1") all become a
     URL the adapter can build its own path on, and that normalized value
     is what gets persisted.
  2. `OpenAICompatibleProvider.chat` — if the server still answers 404,
     walk the candidate API roots and stick with the one that works.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Tuple

import pytest

import agent.llm.runtime as runtime_mod
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.provider import Message, ProviderError
from agent.llm.runtime import (
    configure_runtime_provider,
    load_runtime_config,
    normalize_base_url,
    plan_runtime,
)

OK_PAYLOAD = {
    "choices": [{
        "finish_reason": "stop",
        "message": {"content": "ok"},
    }],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
}


# ======================================================================
# 1. normalize_base_url: one paste, any provider
# ======================================================================

@pytest.mark.parametrize("pasted, family, expected", [
    # A bare host: scheme added, /v1 filled in for OpenAI-compatible.
    ("api.openai.com", "openai-compatible",
     "https://api.openai.com/v1"),
    ("https://api.openai.com", "openai-compatible",
     "https://api.openai.com/v1"),
    # A pasted endpoint: the suffix comes off; no /v1 is re-added,
    # because the pasted path already said where the endpoint lives.
    ("https://openrouter.ai/api/v1/chat/completions", "openai-compatible",
     "https://openrouter.ai/api/v1"),
    ("http://localhost:8000/chat/completions", "openai-compatible",
     "http://localhost:8000"),
    ("api.together.xyz/v1/completions", "openai-compatible",
     "https://api.together.xyz/v1"),
    # Already usable stays as-is.
    ("https://my.corp/llm/v1", "openai-compatible",
     "https://my.corp/llm/v1"),
    # Anthropic: the adapter posts /v1/messages itself.
    ("https://api.anthropic.com/v1", "anthropic",
     "https://api.anthropic.com"),
    ("api.anthropic.com", "anthropic", "https://api.anthropic.com"),
    ("https://proxy.corp/api/v1/messages", "anthropic",
     "https://proxy.corp/api"),
    # Gemini: the adapter posts /{version}/models/... itself.
    ("https://generativelanguage.googleapis.com/v1beta", "gemini",
     "https://generativelanguage.googleapis.com"),
    ("https://generativelanguage.googleapis.com/v1", "gemini",
     "https://generativelanguage.googleapis.com"),
    # Ollama: the adapter posts /api/chat; /v1 is its OpenAI-compat layer.
    ("localhost:11434", "ollama", "https://localhost:11434"),
    ("http://localhost:11434/v1", "ollama", "http://localhost:11434"),
    # Query/fragment noise is dropped; empty input round-trips.
    ("https://api.openai.com/v1?key=x#frag", "openai-compatible",
     "https://api.openai.com/v1"),
    ("", "openai-compatible", ""),
])
def test_normalize_base_url_shapes(pasted: str, family: str, expected: str):
    assert normalize_base_url(pasted, family) == expected


def test_normalize_base_url_is_idempotent():
    once = normalize_base_url("api.openai.com/v1/chat/completions",
                              "openai-compatible")
    assert normalize_base_url(once, "openai-compatible") == once


# ======================================================================
# 2. plan_runtime / connect: the normalized value is what gets stored
# ======================================================================

def test_plan_runtime_normalizes_and_explains_itself():
    spec = plan_runtime(
        "openrouter.ai/api/v1/chat/completions", "sk-x", "openai/auto",
        family="openai-compatible",
    )

    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.provider_config.base_url == spec.base_url
    assert any("adapted" in note for note in spec.notes), spec.notes


def test_plan_runtime_detects_family_then_adapts():
    spec = plan_runtime("api.anthropic.com/v1", "sk-ant-x", "claude-x")

    assert spec.family == "anthropic"
    assert spec.base_url == "https://api.anthropic.com"


class FakeRegistry:
    def __init__(self):
        self.providers = {}
        self._api_keys = {}
        self._current_provider = None
        self._current_model = None

    def install_provider(self, name, provider, api_key, model, metadata=None):
        self.providers[name] = provider
        self._api_keys[name] = api_key
        self._current_provider = name
        self._current_model = model


async def _fake_discover(provider, model, family):
    return {
        "context_window": 8192, "max_output": 4096,
        "supports_tools": True, "supports_streaming": False,
        "supports_vision": False, "supports_json_mode": False,
        "description": "", "source": "fallback",
    }


def test_connect_persists_the_normalized_base_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # No network in this test: skip capability discovery.
    monkeypatch.setattr(runtime_mod, "discover_capabilities", _fake_discover)

    configure_runtime_provider(
        FakeRegistry(), "openrouter.ai/api/v1/chat/completions",
        "sk-x", "openai/auto",
    )

    saved = load_runtime_config()
    assert saved["base_url"] == "https://openrouter.ai/api/v1"
    assert saved["provider"] == "openai-compatible"


# ======================================================================
# 3. live: the adapter walks past a wrong "/v1" and sticks to the winner
# ======================================================================

@contextmanager
def fake_server(
    routes: Dict[str, Tuple[int, Dict[str, Any]]],
) -> Iterator[Tuple[str, List[str]]]:
    """Serve `routes`; anything else 404s. Yields (url, hit log)."""
    hits: List[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            hits.append(self.path)
            status, payload = routes.get(self.path, (404, {"error": "nope"}))
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):  # keep test output clean
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}", hits
    finally:
        server.shutdown()
        server.server_close()


def _config(base_url: str, **extra) -> Dict[str, Any]:
    # Fail fast in tests: one attempt, short timeout.
    return {"api_key": "test-key", "base_url": base_url,
            "timeout": 5.0, "max_retries": 0, **extra}


def _messages() -> List[Message]:
    return [Message(role="user", content="hi")]


@pytest.mark.asyncio
async def test_missing_v1_falls_back_then_remembers_it():
    with fake_server({
        "/chat/completions": (404, {"error": "nope"}),
        "/v1/chat/completions": (200, OK_PAYLOAD),
    }) as (url, hits):
        provider = OpenAICompatibleProvider(_config(url))
        await provider.start()
        try:
            resp = await provider.chat(_messages(), model="m")
            assert resp.content == "ok"
            assert hits == ["/chat/completions", "/v1/chat/completions"]

            # The working root is remembered: the next call goes
            # straight there instead of probing again.
            await provider.chat(_messages(), model="m")
            assert hits == [
                "/chat/completions", "/v1/chat/completions",
                "/v1/chat/completions",
            ]
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_gateway_under_api_v1_is_reached():
    with fake_server({
        "/api/v1/chat/completions": (200, OK_PAYLOAD),
    }) as (url, hits):
        provider = OpenAICompatibleProvider(_config(url))
        await provider.start()
        try:
            resp = await provider.chat(_messages(), model="m")
            assert resp.content == "ok"
            assert hits == [
                "/chat/completions",
                "/v1/chat/completions",
                "/api/v1/chat/completions",
            ]
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_only_404_triggers_adaptation():
    with fake_server({
        "/chat/completions": (500, {"error": "boom"}),
        "/v1/chat/completions": (500, {"error": "boom"}),
        "/api/v1/chat/completions": (500, {"error": "boom"}),
    }) as (url, hits):
        provider = OpenAICompatibleProvider(_config(url))
        await provider.start()
        try:
            with pytest.raises(ProviderError) as exc:
                await provider.chat(_messages(), model="m")
            # A server error is a real failure — no wandering to
            # other bases, and the status survives for the caller.
            assert exc.value.status == 500
            assert hits == ["/chat/completions"]
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_pasted_endpoint_via_plan_runtime_works_first_try():
    with fake_server({
        "/chat/completions": (200, OK_PAYLOAD),
    }) as (url, hits):
        spec = plan_runtime(
            f"{url}/chat/completions", "test-key", "m",
            family="openai-compatible",
        )
        assert spec.base_url == url, "the pasted endpoint is stripped"

        provider = OpenAICompatibleProvider(_config(spec.base_url))
        await provider.start()
        try:
            resp = await provider.chat(_messages(), model="m")
            assert resp.content == "ok"
            assert hits == ["/chat/completions"]
        finally:
            await provider.close()
