"""Provider-layer prompt caching: stable prefixes, markers, and accounting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop, LoopContext
from agent.llm.anthropic import AnthropicProvider
from agent.llm.anthropic import _normalize_usage as anthropic_usage
from agent.llm.gemini import GeminiProvider
from agent.llm.normalizer import NormalizedResponse, NormalizedUsage, ResponseNormalizer
from agent.llm.ollama import OllamaProvider
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.provider import Message, ToolSpec
from agent.llm.runtime import MODEL_METADATA
from agent.tui.app import _usage_add, _usage_empty, _usage_snapshot

SPEC_A = ToolSpec(name="alpha", description="first")
SPEC_Z = ToolSpec(name="zulu", description="last")


def test_anthropic_splits_stable_system_and_marks_tools():
    body = AnthropicProvider({"api_key": "k"})._build_body(
        [
            Message.system("stable system"),
            Message.user("hello"),
            Message.system("transient execution state"),
        ],
        "claude-test", 100, [SPEC_Z, SPEC_A], None, False,
    )

    system = body["system"]
    assert [block["text"] for block in system] == [
        "stable system",
        "transient execution state",
    ]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[1]

    assert [tool["name"] for tool in body["tools"]] == ["alpha", "zulu"]
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    # Conversation-prefix breakpoint lands on the final message block.
    last_block = body["messages"][-1]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_cache_can_be_disabled():
    body = AnthropicProvider(
        {"api_key": "k", "prompt_cache": False}
    )._build_body(
        [Message.system("stable"), Message.user("hello")],
        "claude-test", 100, [SPEC_A], None, False,
    )

    assert body["system"] == "stable"
    assert "cache_control" not in body["tools"][0]
    assert body["messages"][-1]["content"] == "hello"


def test_anthropic_usage_counts_cache_as_input():
    usage = anthropic_usage({
        "input_tokens": 5,
        "output_tokens": 7,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
    })

    assert usage["prompt_tokens"] == 125
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 132
    assert usage["cached_tokens"] == 100
    assert usage["cache_creation_tokens"] == 20


def test_all_adapters_use_stable_tool_order_and_avoid_unsupported_markers():
    openai = OpenAICompatibleProvider(
        {"api_key": "k", "prompt_cache_key": "session-1"}
    )._build_body(
        [Message.user("hi")], "m", 100, [SPEC_Z, SPEC_A], None, False
    )
    assert [t["function"]["name"] for t in openai["tools"]] == ["alpha", "zulu"]
    assert openai["prompt_cache_key"] == "session-1"
    assert "cache_control" not in str(openai["tools"])

    gemini = GeminiProvider(
        {"api_key": "k", "cached_content": "cachedContents/test"}
    )._build_body([Message.user("hi")], 100, [SPEC_Z, SPEC_A], None)
    decls = gemini["tools"][0]["function_declarations"]
    assert [d["name"] for d in decls] == ["alpha", "zulu"]
    assert gemini["cachedContent"] == "cachedContents/test"
    assert "cache_control" not in str(gemini["tools"])

    ollama = OllamaProvider()._build_body(
        [Message.user("hi")], "m", 100, [SPEC_Z, SPEC_A], None, False
    )
    assert [t["function"]["name"] for t in ollama["tools"]] == ["alpha", "zulu"]
    assert "cache_control" not in str(ollama["tools"])


def test_gemini_system_instruction_keeps_stable_part_first():
    body = GeminiProvider({"api_key": "k"})._build_body(
        [
            Message.system("stable"),
            Message.user("hi"),
            Message.system("transient"),
        ],
        100, None, None,
    )
    parts = body["system_instruction"]["parts"]
    assert [part["text"] for part in parts] == ["stable", "transient"]


def test_normalizer_preserves_anthropic_cache_fields():
    usage = ResponseNormalizer._normalize_usage({
        "input_tokens": 5,
        "output_tokens": 7,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
    })
    assert usage.prompt_tokens == 125
    assert usage.cached_tokens == 100
    assert usage.cache_creation_tokens == 20
    assert usage.billable_input == 25


def test_loop_records_cache_tokens_cost_and_savings():
    loop = object.__new__(AgentLoop)
    loop.context = LoopContext()
    loop.llm = SimpleNamespace(get_current_model=lambda: "cache-test")

    previous = MODEL_METADATA.get("cache-test")
    MODEL_METADATA["cache-test"] = {"cost_input": 10.0, "cost_output": 20.0}
    try:
        loop._record_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cached_tokens": 600,
            "cache_creation_tokens": 100,
        })
    finally:
        if previous is None:
            MODEL_METADATA.pop("cache-test", None)
        else:
            MODEL_METADATA["cache-test"] = previous

    assert loop.context.cached_tokens == 600
    assert loop.context.cache_creation_tokens == 100
    assert loop.context.cache_savings == pytest.approx(600 / 1_000_000 * 10.0)
    # 400 billable input tokens + 100 output tokens.
    assert loop.context.cost == pytest.approx(
        (400 / 1_000_000 * 10.0) + (100 / 1_000_000 * 20.0)
    )


def test_streaming_helper_keeps_cache_usage():
    class StreamLLM:
        async def stream_with_tools(self, **kwargs):
            yield NormalizedResponse(
                text="hello",
                usage=NormalizedUsage(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                    cached_tokens=60,
                    cache_creation_tokens=20,
                ),
            )

    loop = object.__new__(AgentLoop)
    loop.enable_streaming = True
    loop.llm = StreamLLM()
    loop.event_handlers = {}

    import asyncio

    response = asyncio.run(
        loop._call_llm_maybe_streaming(
            messages=[], tool_schemas=[], max_tokens=10
        )
    )
    assert response.usage["cached_tokens"] == 60
    assert response.usage["cache_creation_tokens"] == 20


def test_tui_usage_snapshot_and_add_track_cache():
    snap = _usage_snapshot(SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        tokens_used=110,
        cached_tokens=70,
        cache_creation_tokens=5,
        cache_savings=0.25,
    ))
    assert snap["cached_tokens"] == 70
    assert snap["cache_creation_tokens"] == 5
    assert snap["cache_savings"] == 0.25

    total = _usage_empty()
    _usage_add(total, snap)
    _usage_add(total, snap)
    assert total["cached_tokens"] == 140
    assert total["cache_creation_tokens"] == 10
    assert total["cache_savings"] == pytest.approx(0.5)
