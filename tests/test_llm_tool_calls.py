"""The new `src/agent/llm/` package, verified end to end.

Three layers, in the order the data flows:

  1. outbound  — ToolSpec → each adapter's native request schema
  2. inbound   — each provider's native response payload → ToolCall
  3. loop      — registry → normalizer → AgentLoop actually executes
                 the tool the model asked for
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.anthropic import AnthropicProvider
from agent.llm.gemini import GeminiProvider
from agent.llm.ollama import OllamaProvider
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.provider import (
    FinishReason,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
)
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


@pytest.fixture(autouse=True)
def clean_registry():
    reset_llm_registry()
    yield
    reset_llm_registry()


def _messages() -> list:
    return [Message(role="user", content="hi")]


# ======================================================================
# 1. OUTBOUND: tool schema → provider-specific request body
# ======================================================================

def test_openai_request_keeps_openai_tool_shape():
    provider = OpenAICompatibleProvider({"api_key": "k"})
    body = provider._build_body(_messages(), "m", 0.1, 100, [SPEC], None, False)

    assert body["tools"][0]["type"] == "function"
    fn = body["tools"][0]["function"]
    assert fn["name"] == "note_tool"
    assert fn["parameters"]["required"] == ["text"]
    assert body["tool_choice"] == "auto"


def test_anthropic_request_uses_input_schema():
    provider = AnthropicProvider({"api_key": "k"})
    body = provider._build_body(_messages(), "m", 0.1, 100, [SPEC], None, False)

    # Anthropic wants {"name", "description", "input_schema"} — not "parameters".
    assert "parameters" not in body["tools"][0]
    assert body["tools"][0]["name"] == "note_tool"
    assert body["tools"][0]["input_schema"]["required"] == ["text"]
    assert body["tool_choice"] == {"type": "auto"}


def test_gemini_request_wraps_function_declarations():
    provider = GeminiProvider({"api_key": "k"})
    body = provider._build_body(_messages(), 0.1, 100, [SPEC], None)

    decls = body["tools"][0]["function_declarations"]
    assert decls[0]["name"] == "note_tool"
    assert decls[0]["parameters"]["required"] == ["text"]
    # tool_choice "auto" is Gemini's default, so no tool_config is sent.
    assert "tool_config" not in body

    body = provider._build_body(_messages(), 0.1, 100, [SPEC], "required")
    assert body["tool_config"]["function_calling_config"]["mode"] == "ANY"


def test_ollama_request_uses_openai_tool_shape():
    provider = OllamaProvider()
    body = provider._build_body(_messages(), "m", 0.1, 100, [SPEC], None, False)

    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "note_tool"


# ======================================================================
# 2. INBOUND: native response payload → ToolCall
# ======================================================================

def test_openai_response_parses_tool_calls():
    provider = OpenAICompatibleProvider({"api_key": "k"})
    resp = provider._parse({
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
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }, "vendor/custom")

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
        ("call_1", "note_tool", {"text": "hello"})
    ]
    assert resp.usage["total_tokens"] == 8


def test_anthropic_response_parses_tool_use_blocks():
    provider = AnthropicProvider({"api_key": "k"})
    resp = provider._parse({
        "content": [
            {"type": "text", "text": "let me note that"},
            {"type": "tool_use", "id": "toolu_1", "name": "note_tool",
             "input": {"text": "hello"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 9, "output_tokens": 4},
    }, "vendor/custom")

    assert resp.content == "let me note that"
    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
        ("toolu_1", "note_tool", {"text": "hello"})
    ]
    assert resp.usage["prompt_tokens"] == 9


def test_gemini_response_parses_function_call():
    provider = GeminiProvider({"api_key": "k"})
    resp = provider._parse({
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [
                {"functionCall": {"name": "note_tool", "args": {"text": "hello"}}}
            ]},
        }],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
    }, "vendor/custom")

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    call = resp.tool_calls[0]
    assert (call.name, call.arguments) == ("note_tool", {"text": "hello"})
    assert call.id, "a tool-call id is always present"
    assert resp.usage["total_tokens"] == 9


def test_ollama_response_parses_tool_calls_and_recovers_text_form():
    provider = OllamaProvider()
    resp = provider._parse({
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "note_tool", "arguments": {"text": "hello"}}}
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 6,
        "eval_count": 3,
    }, "vendor/custom")

    assert resp.finish_reason == FinishReason.TOOL_CALLS.value
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [
        ("note_tool", {"text": "hello"})
    ]
    assert resp.usage["prompt_tokens"] == 6

    # Local models sometimes print the call as JSON in the text instead.
    recovered = provider._parse({
        "message": {"role": "assistant",
                    "content": '{"tool": "note_tool", "parameters": {"text": "hi"}}'},
        "done": True,
        "done_reason": "stop",
    }, "vendor/custom")
    assert [(c.name, c.arguments) for c in recovered.tool_calls] == [
        ("note_tool", {"text": "hi"})
    ]


# ======================================================================
# 3. LOOP: registry → normalizer → tool actually executes
# ======================================================================

class ScriptedProvider:
    """Stand-in adapter: turn 1 asks for a tool, turn 2 reports back."""

    name = "scripted"
    started = True

    def __init__(self):
        self.calls = 0
        self.seen_tools = None

    def is_configured(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def chat(self, messages, *, model=None, **kwargs):
        self.calls += 1
        self.seen_tools = kwargs.get("tools")
        if self.calls == 1:
            return LLMResponse(
                content="calling the tool",
                model=model or "",
                provider=self.name,
                finish_reason=FinishReason.TOOL_CALLS.value,
                tool_calls=[ToolCall(id="call-1", name="note_tool",
                                     arguments={"text": "hello"})],
                usage={"prompt_tokens": 4, "completion_tokens": 6},
            )
        return LLMResponse(
            content="note stored",
            model=model or "",
            provider=self.name,
            finish_reason=FinishReason.STOP.value,
            usage={"prompt_tokens": 10, "completion_tokens": 3},
        )


class FakeContextManager:
    def __init__(self):
        self.messages = []

    async def add_message(self, role, content, metadata=None, **kwargs):
        self.messages.append(SimpleNamespace(role=role, content=content,
                                             metadata=metadata or {}))

    async def add_system_message(self, content, pinned=True, **kwargs):
        await self.add_message("system", content, {"pinned": pinned})

    async def add_assistant_message(self, content, metadata=None, **kwargs):
        await self.add_message("assistant", content, metadata or {})


class ExecAgent:
    def __init__(self):
        self.context_manager = FakeContextManager()
        self.workspace = None
        self.permission_manager = None
        self.executed = []

    async def execute_tool(self, tool_name, params):
        self.executed.append((tool_name, params))
        return {"success": True, "content": "note stored", "tool": tool_name}


class FakeToolRegistry:
    """Supplies the schema the loop forwards to the provider."""

    tools = {"note_tool": object()}

    def get_schemas(self):
        return [{
            "type": "function",
            "function": {
                "name": "note_tool",
                "description": "Store a short note",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }]

    def is_read_only(self, name):
        return True


@pytest.mark.asyncio
async def test_loop_executes_tool_call_through_registry():
    provider = ScriptedProvider()
    registry = get_llm_registry()
    registry.install_provider(
        "custom", provider, api_key="k", model="vendor/custom",
        metadata={"provider": "openai-compatible", "context_window": 8192},
    )

    agent = ExecAgent()
    loop = AgentLoop(
        agent, registry, FakeToolRegistry(), None,
        {"enable_intent_classification": False, "enable_planning": False},
    )
    result = await loop.run("store a note")

    assert result["success"] is True, result
    # The tool the model asked for really ran, with the arguments it sent.
    assert agent.executed == [("note_tool", {"text": "hello"})]
    assert [c.name for c in loop.completed_tool_calls] == ["note_tool"]
    # Two model turns: tool call, then the final answer.
    assert provider.calls == 2
    assert result["response"] == "note stored"
    # The loop's tool schemas reached the provider unchanged.
    assert provider.seen_tools[0]["function"]["name"] == "note_tool"
    # Usage flowed back through the normalizer into loop accounting.
    assert loop.context.llm_calls == 2
    assert loop.context.input_tokens == 14
    assert loop.context.output_tokens == 9
    assert loop.context.tokens_used == 23
