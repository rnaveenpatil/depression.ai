"""Tests for the latency fixes in the agent loop and runtime provider.

Covers:
  - runtime provider fails fast (30s timeout, 2 retries) instead of 120s x 3
  - project context (file list + git info) is cached, not rebuilt per think
  - single successful tool call round-trips without a 2nd LLM call
  - the loop still continues normally when the fast-path precondition isn't met
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import (
    LLMResponse,
    ToolCall,
    get_llm_registry,
    reset_llm_registry,
)
from agent.llm.runtime import configure_runtime_provider

RUNTIME_ENVS = (
    "DEPRESSION_PROVIDER",
    "DEPRESSION_BASE_URL",
    "DEPRESSION_API_KEY",
    "DEPRESSION_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_runtime_env(monkeypatch, tmp_path):
    for key in RUNTIME_ENVS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_llm_registry()
    yield
    reset_llm_registry()


def test_runtime_provider_fails_fast():
    registry = get_llm_registry()
    provider = configure_runtime_provider(
        registry, "https://llm.example/v1", "tui-secret-key", "vendor/custom-agent"
    )
    assert provider.timeout == 30.0
    assert provider.max_retries == 2


class _CtxManager:
    def __init__(self):
        self.messages = []

    async def add_message(self, **kwargs):
        self.messages.append(SimpleNamespace(role=kwargs["role"], content=kwargs.get("content", ""), metadata=kwargs.get("metadata", {})))

    async def add_system_message(self, content, pinned=False):
        self.messages.append(SimpleNamespace(role="system", content=content, metadata={}, pinned=pinned))

    async def add_assistant_message(self, content):
        self.messages.append(SimpleNamespace(role="assistant", content=content, metadata={}))


def _make_loop(agent, llm):
    registry = SimpleNamespace(tools={})
    return AgentLoop(agent=agent, llm=llm, tool_registry=registry, planner=None, config={})


async def test_project_context_cached():
    class WS:
        def __init__(self):
            self.project_dir = Path(".")
            self.list_count = 0
            self.git_count = 0

        async def list_files(self, max_files=200):
            self.list_count += 1
            return []

        async def get_git_info(self):
            self.git_count += 1
            return {"is_git_repo": False}

    class AG:
        def __init__(self):
            self.workspace = WS()

    class LLM:
        def get_current_model(self):
            return "gpt-4o"

    loop = _make_loop(AG(), LLM())
    ctx1 = await loop._get_project_context()
    ctx2 = await loop._get_project_context()
    assert ctx1 == ctx2
    assert loop._project_ctx is not None
    assert ctx1.keys() == {"project_path", "files", "git_info"}


class _ReadLLM:
    def __init__(self, with_content=False):
        self.calls = 0
        self.with_content = with_content
        self.model = "vendor/custom-agent"

    def get_current_model(self):
        return self.model

    async def complete_with_tools(self, messages, tools=None, temperature=0.7, max_tokens=2000, **kwargs):
        self.calls += 1
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        if self.calls == 1:
            content = "preparing read" if self.with_content else ""
            return LLMResponse(
                content=content,
                model=self.model,
                provider="custom",
                tool_calls=[ToolCall(id="call-1", name="read", arguments={"filePath": "a.txt"})],
                usage=usage,
            )
        return LLMResponse(content="final summary from model", model=self.model, provider="custom", usage=usage)

    async def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(content="final summary from model", model=self.model, provider="custom")


class _ExecAgent:
    def __init__(self):
        self.context_manager = _CtxManager()
        self.workspace = None
        self.permission_manager = None

    async def execute_tool(self, tool_name, params):
        return {"success": True, "content": "hello world file content", "tool": tool_name}


async def test_single_tool_read_uses_fast_path():
    llm = _ReadLLM()
    loop = _make_loop(_ExecAgent(), llm)

    result = await loop.run(query="read a.txt")

    assert result["success"] is True
    assert result["response"] == "hello world file content"
    assert result["tool_calls"] == 1
    assert llm.calls == 1


async def test_loop_still_continues_when_fast_path_not_applicable():
    llm = _ReadLLM(with_content=True)
    loop = _make_loop(_ExecAgent(), llm)

    result = await loop.run(query="read a.txt")

    assert result["success"] is True
    assert result["response"] == "final summary from model"
    assert llm.calls == 2