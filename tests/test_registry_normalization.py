"""The adapters → normalizer → loop boundary.

`registry.complete()` must hand the loop a `NormalizedResponse` no matter
what shape the adapter returned, and must keep raising `ProviderError`
on failure (the loop's error path is unchanged).
"""
from __future__ import annotations

import pytest

from agent.llm.normalizer import NormalizedResponse
from agent.llm.provider import LLMResponse, Message, ProviderError, ToolCall
from agent.llm.runtime import get_llm_registry, reset_llm_registry


class _StubProvider:
    """Adapter double that returns the messy shapes real providers do."""

    name = "stub"
    started = True

    def __init__(self, response: LLMResponse):
        self._response = response

    def is_configured(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def chat(self, messages, *, model=None, **kwargs):
        return self._response


@pytest.fixture(autouse=True)
def clean_registry():
    reset_llm_registry()
    yield
    reset_llm_registry()


def _connect(response: LLMResponse) -> None:
    registry = get_llm_registry()
    registry.install_provider(
        "custom", _StubProvider(response), api_key="k", model="vendor/custom",
        metadata={"provider": "openai-compatible", "context_window": 8192},
    )


@pytest.mark.asyncio
async def test_complete_returns_a_normalized_response():
    _connect(LLMResponse(content="hi", model="vendor/custom", provider="stub"))

    out = await get_llm_registry().complete([Message.user("hello")])

    assert isinstance(out, NormalizedResponse)
    # Legacy call sites read .content; canonical field is .text.
    assert out.content == out.text == "hi"
    assert out.provider == "stub"
    assert out.model == "vendor/custom"
    assert out.error is None


@pytest.mark.asyncio
async def test_adapter_output_is_canonicalized():
    _connect(LLMResponse(
        content="",
        finish_reason="tool_use",           # Anthropic's spelling
        tool_calls=[
            ToolCall(id="", name="read", arguments={"filePath": "a.txt"}),
            ToolCall(id="x", name="", arguments={}),   # no name → dropped
        ],
        usage={"input_tokens": 7, "output_tokens": 5},  # Anthropic's keys
    ))

    out = await get_llm_registry().complete([Message.user("hello")])

    assert out.finish_reason == "tool_calls"
    assert [tc.name for tc in out.tool_calls] == ["read"]
    assert out.tool_calls[0].id, "missing tool-call ids are synthesized"
    assert out.tool_calls[0].arguments == {"filePath": "a.txt"}
    assert out.usage.prompt_tokens == 7
    assert out.usage.completion_tokens == 5
    assert out.usage.total_tokens == 12
    # Loop-facing fields stay readable.
    assert getattr(out, "content", None) == ""
    assert len(list(getattr(out, "tool_calls", []))) == 1


@pytest.mark.asyncio
async def test_no_model_still_raises_provider_error():
    with pytest.raises(ProviderError):
        await get_llm_registry().complete([Message.user("hello")])
