"""Focused tests for current AgentLoop behavior."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse, ToolCall
from agent.llm.runtime import configure_runtime_provider, get_llm_registry, reset_llm_registry


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch, tmp_path):
    for key in ("DEPRESSION_PROVIDER", "DEPRESSION_BASE_URL", "DEPRESSION_API_KEY", "DEPRESSION_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_llm_registry()
    yield
    reset_llm_registry()


def test_runtime_provider_fails_fast():
    registry = get_llm_registry()
    provider = configure_runtime_provider(registry, "https://llm.example/v1", "dummy-key", "vendor/custom")
    assert provider.timeout == 30.0
    assert provider.max_retries == 2


class MockToolRegistry:
    """Minimal ToolRegistry mock with all methods AgentLoop expects."""
    
    def __init__(self, tools=None):
        self.tools = tools or {}
    
    def get_schemas(self):
        return []
    
    def select_for_task(self, intent):
        return []
    
    def list_tools(self):
        return list(self.tools.keys())
    
    def has_tool(self, name):
        return name in self.tools
    
    def is_read_only(self, name):
        return True
    
    def get_category(self, name):
        return "inspect"


@pytest.mark.asyncio
async def test_project_context_is_cached():
    class WS:
        project_dir = Path(".")
        list_count = 0

        async def list_files(self, max_files=200):
            self.list_count += 1
            return []

    class Agent:
        workspace = WS()
        context_manager = SimpleNamespace(messages=[])

    class LLM:
        def get_current_model(self):
            return "test/model"

    loop = AgentLoop(Agent(), LLM(), MockToolRegistry(), None, {})
    first = await loop._get_project_context()
    second = await loop._get_project_context()
    assert first == second
    assert Agent.workspace.list_count == 1


class ReadLLM:
    def __init__(self, with_content=False):
        self.calls = 0
        self.with_content = with_content

    def get_current_model(self):
        return "test/model"

    async def complete_with_tools(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="preparing read" if self.with_content else "",
                model="test/model",
                provider="test",
                tool_calls=[ToolCall(id="1", name="read", arguments={"filePath": "a.txt"})],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        return LLMResponse(content="final summary", model="test/model", provider="test")

    async def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(content="final summary", model="test/model", provider="test")


class ReadContextManager:
    """
    Minimal fake of the real ContextManager.

    Mirrors the API the loop actually calls:
        add_message(role, content, metadata)
        add_system_message(content, pinned)
        add_assistant_message(content)
    """

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


@pytest.mark.asyncio
async def test_single_read_tool_uses_fast_path():
    llm = ReadLLM()
    loop = AgentLoop(
        ExecAgent(), llm, MockToolRegistry({"read": object()}), None,
        {"enable_intent_classification": False, "enable_planning": False, "require_demo_offer": False},
    )
    result = await loop.run("read a.txt")
    assert result["success"] is True, result
    assert result["response"] == "hello world file content"
    assert result["tool_calls"] == 1
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_read_with_intermediate_content_continues():
    """
    When the model emits assistant text AND a tool call on the same turn,
    the fast path is skipped (fast path requires no assistant text) and
    the loop runs a second LLM turn for the final answer.
    """
    llm = ReadLLM(with_content=True)
    loop = AgentLoop(
        ExecAgent(), llm, MockToolRegistry({"read": object()}), None,
        {"enable_intent_classification": False, "enable_planning": False, "require_demo_offer": False},
    )
    result = await loop.run("read a.txt")
    assert result["success"] is True, result
    assert result["response"] == "final summary"
    assert llm.calls == 2