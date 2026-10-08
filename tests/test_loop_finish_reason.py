"""Regression tests: the agent must not silently accept an *incomplete* LLM
turn (truncated output or a provider-side error) as its final answer.

Before the fix, ``AgentLoop._think`` only retried when the response was
completely empty.  A turn with ``finish_reason="length"`` (provider hit
max_tokens) or ``finish_reason="error"`` (Gemini MALFORMED_FUNCTION_CALL)
that contained partial assistant text and no tool call was accepted as the
final answer -- so the agent "sometimes answered without calling a tool".
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import FinishReason, LLMResponse, ToolCall
from agent.llm.runtime import reset_llm_registry


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch, tmp_path):
    for key in (
        "DEPRESSION_PROVIDER",
        "DEPRESSION_BASE_URL",
        "DEPRESSION_API_KEY",
        "DEPRESSION_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_llm_registry()
    yield
    reset_llm_registry()


class ReadContextManager:
    """Minimal fake of the real ContextManager (see test_loop_speedups)."""

    def __init__(self):
        self.messages = []

    async def add_message(self, role, content, metadata=None, **kwargs):
        self.messages.append(
            SimpleNamespace(role=role, content=content, metadata=metadata or {})
        )

    async def add_system_message(self, content, pinned=True, **kwargs):
        await self.add_message("system", content, {"pinned": pinned})

    async def add_assistant_message(self, content, metadata=None, **kwargs):
        await self.add_message("assistant", content, metadata or {})


class ExecAgent:
    def __init__(self):
        self.context_manager = ReadContextManager()
        self.workspace = None
        self.permission_manager = None

    async def execute_tool(self, tool_name, params):
        return {"success": True, "content": "hello world file content", "tool": tool_name}


class FirstTurnIncompleteLLM:
    """Turn 1: partial text, no tool call, non-STOP finish reason.
    Turn 2: the tool call the model should have emitted.
    """

    def __init__(self, first_finish_reason: str, first_content: str = "Let me check"):
        self.first_finish_reason = first_finish_reason
        self.first_content = first_content
        self.calls = 0
        self.max_tokens_seen = []

    def get_current_model(self):
        return "test/model"

    async def complete_with_tools(self, messages, **kwargs):
        self.calls += 1
        self.max_tokens_seen.append(kwargs.get("max_tokens"))
        if self.calls == 1:
            return LLMResponse(
                content=self.first_content,
                model="test/model",
                provider="test",
                finish_reason=self.first_finish_reason,
            )
        return LLMResponse(
            content="",
            model="test/model",
            provider="test",
            finish_reason=FinishReason.TOOL_CALLS.value,
            tool_calls=[ToolCall(id="1", name="read", arguments={"filePath": "a.txt"})],
        )


class StopTextLLM:
    """A legitimate plain-text final answer (no tool call)."""

    def __init__(self):
        self.calls = 0

    def get_current_model(self):
        return "test/model"

    async def complete_with_tools(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(
            content="Paris is the capital of France.",
            model="test/model",
            provider="test",
            finish_reason=FinishReason.STOP.value,
        )


def _loop(llm):
    return AgentLoop(
        ExecAgent(),
        llm,
        SimpleNamespace(tools={}),
        None,
        {"enable_intent_classification": False, "enable_planning": False},
    )


@pytest.mark.asyncio
async def test_truncated_turn_is_retried_and_tool_runs():
    """finish_reason=length must NOT end the loop with partial text."""
    llm = FirstTurnIncompleteLLM(FinishReason.LENGTH.value)
    result = await _loop(llm).run("read a.txt")

    assert llm.calls == 2, "truncated turn should be retried exactly once"
    assert result["success"] is True, result
    assert result["tool_calls"] == 1, result
    assert result["response"] == "hello world file content", result


@pytest.mark.asyncio
async def test_truncated_turn_retry_escalates_output_budget():
    """The retry after a truncation gets a larger output budget."""
    llm = FirstTurnIncompleteLLM(FinishReason.LENGTH.value)
    await _loop(llm).run("read a.txt")

    assert llm.max_tokens_seen[0] is not None
    assert llm.max_tokens_seen[1] > llm.max_tokens_seen[0]


@pytest.mark.asyncio
async def test_provider_error_turn_is_retried():
    """Gemini MALFORMED_FUNCTION_CALL normalises to 'error' -> retry."""
    llm = FirstTurnIncompleteLLM(FinishReason.ERROR.value)
    result = await _loop(llm).run("read a.txt")

    assert llm.calls == 2, "errored turn should be retried exactly once"
    assert result["tool_calls"] == 1, result


@pytest.mark.asyncio
async def test_legitimate_text_answer_is_not_retried():
    """A normal STOP turn with text stays a single, final answer."""
    llm = StopTextLLM()
    result = await _loop(llm).run("what is the capital of France?")

    assert llm.calls == 1, "a completed text answer must not be retried"
    assert result["response"] == "Paris is the capital of France.", result
