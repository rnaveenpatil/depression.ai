"""Tool-call history survives the context ↔ provider boundary.

The context layer stores tool calls as JSON-safe dicts for persistence;
the adapters need `ToolCall` objects and each provider's own wire shape.
This used to crash: `Message.to_dict()` raised
`AttributeError: 'dict' object has no attribute 'id'` on the second turn
of any tool-calling conversation, because `metadata["tool_calls"]` holds
dicts.

Layers covered here:
  1. Message coercion      — dicts → ToolCall, junk dropped, round-trips
  2. Adapters              — a full tool history builds a body per provider
  3. Context layer         — add_tool_call/add_tool_result → get_model_messages
                             → every adapter
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

import pytest

from agent.context.runtime import add_tool_call, add_tool_result, get_model_messages
from agent.llm.anthropic import AnthropicProvider
from agent.llm.gemini import GeminiProvider
from agent.llm.ollama import OllamaProvider
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.provider import Message, ToolCall, ToolSpec

SPEC = ToolSpec(
    name="note_tool",
    description="Store a short note",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

# Exactly what context.runtime.add_tool_call() writes into metadata.
STORED_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "note_tool", "arguments": "{\"text\": \"hello\"}"},
}


def _tool_history() -> List[Message]:
    """A second-turn conversation: ask → tool call → tool result."""
    return [
        Message(role="system", content="be terse"),
        Message(role="user", content="store a note"),
        Message(role="assistant", content=None, tool_calls=[STORED_CALL]),
        Message(role="tool", content='{"success": true}',
                name="note_tool", tool_call_id="call_1"),
    ]


# ======================================================================
# 1. MESSAGE COERCION
# ======================================================================

def test_stored_tool_call_dicts_become_tool_calls():
    msg = _tool_history()[2]

    assert all(isinstance(tc, ToolCall) for tc in msg.tool_calls)
    call = msg.tool_calls[0]
    # JSON-string arguments are parsed back into a dict.
    assert (call.id, call.name, call.arguments) == (
        "call_1", "note_tool", {"text": "hello"},
    )


def test_message_dict_round_trip_is_json_safe():
    msg = _tool_history()[2]
    out = msg.to_dict()

    # Serialization must never raise (persistence path), and the wire
    # shape must be the OpenAI one: content null, arguments as a string.
    json.dumps(out)
    assert out["content"] is None
    assert out["tool_calls"][0]["type"] == "function"
    assert out["tool_calls"][0]["function"]["name"] == "note_tool"
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {
        "text": "hello"
    }


def test_plain_and_json_string_argument_forms_are_accepted():
    msg = Message(role="assistant", content=None, tool_calls=[
        {"name": "note_tool", "arguments": {"text": "a"}},
        {"name": "other_tool", "arguments": "{\"text\": \"b\"}"},
        {"tool": "third_tool", "parameters": {"text": "c"}},
    ])

    assert [(tc.name, tc.arguments) for tc in msg.tool_calls] == [
        ("note_tool", {"text": "a"}),
        ("other_tool", {"text": "b"}),
        ("third_tool", {"text": "c"}),
    ]
    assert all(tc.id for tc in msg.tool_calls), "ids are synthesized"


def test_unusable_tool_call_entries_are_dropped_not_fatal():
    msg = Message(role="assistant", content="hi", tool_calls=[
        "junk", {}, None, 42, {"id": "call_1"},  # none of these name a tool
        {"name": "keep_me", "arguments": {}},
    ])

    assert [tc.name for tc in msg.tool_calls] == ["keep_me"]
    json.dumps(msg.to_dict())


def test_tool_result_message_keeps_pairing_fields():
    msg = _tool_history()[3]
    out = msg.to_dict()

    assert out["role"] == "tool"
    assert out["name"] == "note_tool"
    assert out["tool_call_id"] == "call_1"
    assert out["content"] == '{"success": true}'


# ======================================================================
# 2. ADAPTERS: the full history builds a provider-correct body
# ======================================================================

def test_openai_body_carries_tool_calls_and_tool_results():
    body = OpenAICompatibleProvider({"api_key": "k"})._build_body(
        _tool_history(), "m", 0.1, 100, [SPEC], None, False
    )
    msgs = body["messages"]

    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "store a note"}

    assistant = msgs[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == "call_1"
    # OpenAI wants arguments as a JSON *string*.
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "text": "hello"
    }

    tool = msgs[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"
    assert tool["name"] == "note_tool"


def test_anthropic_body_uses_tool_use_and_tool_result_blocks():
    body = AnthropicProvider({"api_key": "k"})._build_body(
        _tool_history(), "m", 0.1, 100, [SPEC], None, False
    )

    # System turns are hoisted out of `messages`.
    assert body.get("system") == "be terse"
    msgs = body["messages"]

    assert msgs[0] == {"role": "user", "content": "store a note"}
    assert msgs[1]["content"][0] == {
        "type": "tool_use", "id": "call_1",
        "name": "note_tool", "input": {"text": "hello"},
    }
    # Tool results come back as a user turn of tool_result blocks.
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "call_1"


def test_gemini_body_uses_function_call_and_function_response():
    body = GeminiProvider({"api_key": "k"})._build_body(
        _tool_history(), 0.1, 100, [SPEC], None
    )

    assert body["system_instruction"]["parts"][0]["text"] == "be terse"
    contents = body["contents"]

    assert contents[0]["role"] == "user"
    model = contents[1]
    assert model["role"] == "model"
    assert model["parts"] == [
        {"functionCall": {"name": "note_tool", "args": {"text": "hello"}}}
    ]
    response = contents[2]["parts"][0]["functionResponse"]
    assert response["name"] == "note_tool"
    assert response["response"] == {"success": True}


def test_ollama_body_keeps_dict_arguments_and_flat_tool_role():
    body = OllamaProvider()._build_body(
        _tool_history(), "m", 0.1, 100, [SPEC], None, False
    )
    msgs = body["messages"]

    assistant = msgs[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ""
    # Ollama wants arguments as a dict, not a JSON string.
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "text": "hello"
    }
    # Ollama pairs results by order — no tool_call_id.
    assert msgs[3] == {"role": "tool", "content": '{"success": true}'}


# ======================================================================
# 3. CONTEXT LAYER: real add_tool_call/add_tool_result → every adapter
# ======================================================================

class RecordingContextManager:
    """The bits of ContextManager that context.runtime touches."""

    def __init__(self):
        self.messages: List[Any] = []

    async def add_message(self, role, content, metadata=None, **kwargs):
        self.messages.append(SimpleNamespace(
            role=role, content=content, metadata=metadata or {},
        ))


@pytest.mark.asyncio
async def test_context_tool_history_builds_a_body_in_every_adapter():
    cm = RecordingContextManager()
    call = ToolCall(id="call_1", name="note_tool", arguments={"text": "hello"})
    await add_tool_call(cm, "calling the tool", [call])
    await add_tool_result(cm, call, {"success": True})

    messages = get_model_messages(cm, 40_000)

    assistant = next(m for m in messages if m.tool_calls)
    assert assistant.role == "assistant"
    # Stored as dicts, presented to adapters as ToolCall objects.
    assert all(isinstance(tc, ToolCall) for tc in assistant.tool_calls)
    assert [(tc.id, tc.name, tc.arguments) for tc in assistant.tool_calls] == [
        ("call_1", "note_tool", {"text": "hello"})
    ]

    tool = next(m for m in messages if m.role == "tool")
    assert tool.tool_call_id == "call_1"
    assert json.loads(tool.content) == {"success": True}

    # The full history builds a body in each adapter without raising.
    openai_body = OpenAICompatibleProvider({"api_key": "k"})._build_body(
        messages, "m", 0.1, 100, [SPEC], None, False
    )
    assert openai_body["messages"][-1]["role"] == "tool"

    anthropic_body = AnthropicProvider({"api_key": "k"})._build_body(
        messages, "m", 0.1, 100, [SPEC], None, False
    )
    assert anthropic_body["messages"][-1]["content"][0]["type"] == "tool_result"

    gemini_body = GeminiProvider({"api_key": "k"})._build_body(
        messages, 0.1, 100, [SPEC], None
    )
    assert any(
        "functionCall" in part
        for entry in gemini_body["contents"] if entry["role"] == "model"
        for part in entry["parts"]
    )

    ollama_body = OllamaProvider()._build_body(
        messages, "m", 0.1, 100, [SPEC], None, False
    )
    assert ollama_body["messages"][-1]["role"] == "tool"
