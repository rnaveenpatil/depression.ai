"""End-to-end wiring tests for the TUI -> terminal -> LLM provider chain.

Covers the exact paths the agentic workflow depends on:
  - persisted runtime connection survives agent bootstrapping
  - the custom OpenAI-compatible provider carries the TUI's API key
  - the agent loop fires on_tool_executed and tracks model cost
  - .env writes resolve to the current working directory at call time
  - both console entry points are importable
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import ToolCall, get_llm_registry, reset_llm_registry
from agent.llm.runtime import configure_runtime_provider
from agent.utils.env_manager import set_env_var

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


def _minimal_agent(registry):
    from agent.agent.dual_agent import AgentRole, BaseAgent

    agent = BaseAgent(
        role=AgentRole.BUILD,
        config={},
        session=None,
        context_manager=None,
        permission_manager=None,
        workspace=None,
        database=None,
    )
    agent.llm = registry
    return agent


async def test_boot_order_keeps_runtime_provider():
    """run.py configures the runtime provider before agent init; agent init
    must NOT clobber it back to the built-in NVIDIA defaults."""
    registry = get_llm_registry()
    configure_runtime_provider(
        registry, "https://llm.example/v1", "tui-secret-key", "vendor/custom-agent"
    )

    agent = _minimal_agent(registry)
    await agent._resolve_role_model()

    assert registry.get_current_provider() == "custom"
    assert registry.get_current_model() == "vendor/custom-agent"

    info = registry.get_model_info(registry.get_current_model())
    provider = registry.providers[info["provider"]]
    assert type(provider).__name__ == "OpenAICompatibleProvider"
    assert provider.api_key == "tui-secret-key"
    assert provider.base_url == "https://llm.example/v1"


class _FakeCM:
    def __init__(self):
        self.messages = []

    async def add_message(self, **kwargs):
        self.messages.append(SimpleNamespace(**kwargs))


class _FakeAgent:
    def __init__(self):
        self.context_manager = _FakeCM()

    async def execute_tool(self, tool_name, params):
        return {"success": True, "output": "ok", "tool": tool_name}


class _FakeLLM:
    def __init__(self, model):
        self._model = model

    def get_current_model(self):
        return self._model


async def test_loop_fires_on_tool_executed_and_tracks_cost():
    loop = AgentLoop(
        agent=_FakeAgent(),
        llm=_FakeLLM("gpt-4o"),
        tool_registry=None,
        planner=None,
        config={},
    )
    seen = []

    async def handler(data):
        seen.append(data)

    loop.add_event_handler("on_tool_executed", handler)
    await loop._act([ToolCall(id="call-1", name="read", arguments={})])

    assert len(seen) == 1
    assert seen[0]["tool"] == "read"
    assert seen[0]["tool_call_id"] == "call-1"
    assert seen[0]["result"]["success"] is True
    assert len(loop.completed_tool_calls) == 1

    # cost uses catalog pricing: gpt-4o = 2.50/M input, 10.00/M output
    assert loop._compute_cost({"prompt_tokens": 1_000_000, "completion_tokens": 100_000}) == pytest.approx(3.50)


def test_env_write_targets_calltime_cwd(tmp_path):
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    before = None
    if repo_env.exists():
        before = repo_env.read_text()

    set_env_var("DEPRESSION_API_KEY", "tmp-only-key")

    env_file = Path.cwd() / ".env"
    assert env_file.exists()
    assert "tmp-only-key" in env_file.read_text()

    # the original bug wrote test secrets to the command's import-time cwd
    if before is not None:
        assert "tmp-only-key" not in repo_env.read_text()
    else:
        assert not repo_env.exists()


def test_console_entry_points_importable():
    import agent.main
    import agent.tui.run

    assert callable(agent.main.main)
    assert callable(agent.tui.run.main)