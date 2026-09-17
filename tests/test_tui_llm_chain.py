"""Focused tests for the current runtime-provider and loop wiring."""
from __future__ import annotations

from types import SimpleNamespace

from agent.agent.dual_agent import AgentRole, BaseAgent
from agent.agent.loop import AgentLoop
from agent.llm.provider import ToolCall, get_llm_registry, reset_llm_registry
from agent.llm.runtime import configure_runtime_provider


def test_runtime_provider_is_selected_by_the_registry():
    reset_llm_registry()
    registry = get_llm_registry()
    provider = configure_runtime_provider(registry, "https://llm.example/v1", "dummy-key", "vendor/custom")
    assert registry.get_current_provider() == "custom"
    assert registry.get_current_model() == "vendor/custom"
    assert provider.base_url == "https://llm.example/v1"
    assert provider.api_key == "dummy-key"
    reset_llm_registry()


def test_role_model_resolution_keeps_user_selected_model():
    reset_llm_registry()
    registry = get_llm_registry()
    configure_runtime_provider(registry, "https://llm.example/v1", "dummy-key", "vendor/custom")
    agent = BaseAgent(
        role=AgentRole.BUILD,
        config={}, session=None, context_manager=None,
        permission_manager=None, workspace=None, database=None,
    )
    agent.llm = registry
    import asyncio
    asyncio.run(agent._resolve_role_model())
    assert agent._fallback_chain == ["vendor/custom"]
    reset_llm_registry()


class FakeAgent:
    context_manager = SimpleNamespace()

    async def execute_tool(self, name, params):
        return {"success": True, "tool": name, "output": "ok"}


class FakeLLM:
    def get_current_model(self):
        return "test/model"


async def test_loop_emits_tool_event():
    loop = AgentLoop(FakeAgent(), FakeLLM(), None, None, {})
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
