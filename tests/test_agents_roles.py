"""Tests for the current Plan/Build agent API.

These tests mirror the current implementation: plan mode is read-only and
build mode can execute mutation through the common tool interface.  Small
fakes keep the tests independent of network services and real credentials.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.dual_agent import AgentCoordinator, AgentRole, BaseAgent, BuildAgent, PlanAgent
from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse, ToolCall
from agent.permissions.manager import PermissionManager


class FakeRegistry:
    def __init__(self, names=("read", "write", "bash", "git", "filesystem")):
        self.tools = {name: SimpleNamespace(name=name) for name in names}
        self.executed = []

    def has_tool(self, name):
        return name in self.tools

    async def execute(self, name, params):
        self.executed.append((name, params))
        return {"success": True, "tool": name, "content": params.get("content", "ok")}


class FakeSession:
    id = "test-session"

    def add_user_message(self, *args, **kwargs):
        pass

    def add_assistant_message(self, *args, **kwargs):
        pass

    def add_tool_message(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass


class FakeContextManager:
    async def add_user_message(self, *args, **kwargs):
        pass

    async def add_assistant_message(self, *args, **kwargs):
        pass

    async def add_tool_output(self, *args, **kwargs):
        pass

    async def needs_compaction(self):
        return False

    async def get_context(self):
        return {}


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get_current_model(self):
        return "test/model"

    def get_current_provider(self):
        return "test"

    async def complete(self, messages=None, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="done", model="test/model", provider="test")

    async def complete_with_tools(self, messages=None, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="done", model="test/model", provider="test")


def _agent(role):
    session = FakeSession()
    pm = PermissionManager({"mode": "manual", "*": "allow", "default": "allow"}, ui=None, input_handler=None)
    agent = BaseAgent(
        role=role,
        config={},
        session=session,
        context_manager=FakeContextManager(),
        permission_manager=pm,
        workspace=None,
        database=None,
    )
    agent.tool_registry = FakeRegistry()
    agent._initialized = True
    return agent


@pytest.mark.asyncio
async def test_plan_agent_rejects_mutating_tools_even_if_registry_contains_them():
    agent = _agent(AgentRole.PLAN)
    result = await agent.execute_tool("write", {"filePath": "x.txt", "content": "blocked"})
    assert result["success"] is False
    assert result["permission_denied"] is True
    assert "read-only" in result["error"]
    assert agent.tool_registry.executed == []


@pytest.mark.asyncio
async def test_plan_agent_allows_read_tools():
    agent = _agent(AgentRole.PLAN)
    result = await agent.execute_tool("read", {"filePath": "x.txt"})
    assert result["success"] is True
    assert agent.tool_registry.executed == [("read", {"filePath": "x.txt"})]


@pytest.mark.asyncio
async def test_build_agent_can_execute_mutating_tools():
    agent = _agent(AgentRole.BUILD)
    result = await agent.execute_tool("write", {"filePath": "x.txt", "content": "ok"})
    assert result["success"] is True
    assert agent.tool_registry.executed[0][0] == "write"


@pytest.mark.asyncio
async def test_process_query_requires_initialization():
    agent = _agent(AgentRole.BUILD)
    agent._initialized = False
    with pytest.raises(Exception, match="used before initialize"):
        await agent.process_query("hello")


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_and_returns_final_response():
    agent = _agent(AgentRole.BUILD)
    llm = ScriptedLLM([
        LLMResponse(
            content="",
            model="test/model",
            provider="test",
            tool_calls=[ToolCall(id="1", name="write", arguments={"content": "ok"})],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
        LLMResponse(content="finished", model="test/model", provider="test"),
    ])
    loop = AgentLoop(
        agent=agent,
        llm=llm,
        tool_registry=agent.tool_registry,
        planner=None,
        config={"enable_planning": False, "enable_intent_classification": False},
    )
    agent.llm = llm
    agent.loop = loop
    result = await loop.run("write something")
    assert result["success"] is True
    assert result["response"] == "finished"
    assert result["tool_calls"] == 1


def test_coordinator_mode_switching():
    plan = PlanAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    build = BuildAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    coordinator = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    assert coordinator.set_mode("plan") is True
    assert coordinator.current_mode == "plan"
    assert coordinator.set_mode("build") is True
    assert coordinator.current_mode == "build"
    assert coordinator.set_mode("invalid") is False
