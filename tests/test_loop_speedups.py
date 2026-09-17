"""Focused tests for current AgentLoop behavior."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse, ToolCall, get_llm_registry, reset_llm_registry
from agent.llm.runtime import configure_runtime_provider


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


@pytest.mark.asyncio
async def test_project_context_is_cached():
    class WS:
        project_dir = Path(".")
        list_count = 0
        git_count = 0

        async def list_files(self, max_files=200):
            self.list_count += 1
            return []

        async def get_git_info(self):
            self.git_count += 1
            return {"is_git_repo": False}

    class Agent:
        workspace = WS()
        context_manager = SimpleNamespace(messages=[])

    class LLM:
        def get_current_model(self):
            return "test/model"

    registry = SimpleNamespace(tools={})
    loop = AgentLoop(Agent(), LLM(), registry, None, {})
    first = await loop._get_project_context()
    second = await loop._get_project_context()
    assert first == second
    assert Agent.workspace.list_count == 1
    assert Agent.workspace.git_count == 1


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


class ExecAgent:
    context_manager = SimpleNamespace(messages=[])
    workspace = None
    permission_manager = None

    async def execute_tool(self, tool_name, params):
        return {"success": True, "content": "hello world file content", "tool": tool_name}


@pytest.mark.asyncio
async def test_single_read_tool_uses_fast_path():
    llm = ReadLLM()
    loop = AgentLoop(ExecAgent(), llm, SimpleNamespace(tools={}), None, {})
    result = await loop.run("read a.txt")
    assert result["success"] is True, result
    assert result["response"] == "hello world file content"
    assert result["tool_calls"] == 1
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_read_with_intermediate_content_continues():
    llm = ReadLLM(with_content=True)
    loop = AgentLoop(ExecAgent(), llm, SimpleNamespace(tools={}), None, {})
    result = await loop.run("read a.txt")
    assert result["success"] is True, result
    assert result["response"] == "final summary"
    assert llm.calls == 2
