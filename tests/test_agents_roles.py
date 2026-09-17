"""Tests for the current Plan/Build agent and execution-loop API."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.dual_agent import AgentCoordinator, AgentRole, BaseAgent, BuildAgent, PlanAgent
from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse, ToolCall
from agent.permissions.manager import PermissionManager


class FakeRegistry:
    def __init__(self, names=("read", "write", "bash", "git", "filesystem")):
        self.tools = {
            name: SimpleNamespace(
                name=name,
                description=f"fake {name} tool",
                parameters={"type": "object"},
            )
            for name in names
        }
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
    def __init__(self):
        self.messages = []

    async def add_message(self, role, content, metadata=None, **kwargs):
        self.messages.append(
            SimpleNamespace(role=role, content=content, metadata=metadata or {})
        )

    async def add_system_message(self, content, pinned=True, **kwargs):
        await self.add_message("system", content, {"pinned": pinned})

    async def add_user_message(self, content, **kwargs):
        await self.add_message("user", content, kwargs)

    async def add_assistant_message(self, content, **kwargs):
        await self.add_message("assistant", content, kwargs)

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
    pm = PermissionManager(
        {"mode": "manual", "*": "allow", "default": "allow"},
        ui=None,
        input_handler=None,
    )
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
    """
    A single successful tool call with readable output takes the loop's
    fast path: the tool result is returned directly and no second LLM
    turn is consumed. This documents that behavior.
    """
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
        config={
            "enable_planning": False,
            "enable_intent_classification": False,
            "enable_qa_verification": False,
        },
    )
    agent.llm = llm
    agent.loop = loop
    result = await loop.run("write something")
    assert result["success"] is True, result
    # Fast path: the tool's `content` ("ok") is returned directly.
    assert result["response"] == "ok"
    assert result["tool_calls"] == 1
    # Only one LLM turn was needed.
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_agent_loop_continues_when_tool_output_has_no_readable_text():
    """
    When the tool result has no readable output field, the fast path is
    skipped and the loop asks the LLM for a final answer.
    """
    agent = _agent(AgentRole.BUILD)

    class OpaqueRegistry(FakeRegistry):
        async def execute(self, name, params):
            self.executed.append((name, params))
            # No content / output / result / text field → fast path skips.
            return {"success": True, "tool": name}

    agent.tool_registry = OpaqueRegistry()
    llm = ScriptedLLM([
        LLMResponse(
            content="",
            model="test/model",
            provider="test",
            tool_calls=[ToolCall(id="1", name="write", arguments={"content": "x"})],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
        LLMResponse(content="finished", model="test/model", provider="test"),
    ])
    loop = AgentLoop(
        agent=agent,
        llm=llm,
        tool_registry=agent.tool_registry,
        planner=None,
        config={
            "enable_planning": False,
            "enable_intent_classification": False,
            "enable_qa_verification": False,
        },
    )
    agent.llm = llm
    agent.loop = loop
    result = await loop.run("write something")
    assert result["success"] is True, result
    assert result["response"] == "finished"
    assert result["tool_calls"] == 1
    assert llm.calls == 2


def test_coordinator_mode_switching():
    plan = PlanAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    build = BuildAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    coordinator = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    assert coordinator.set_mode("plan") is True
    assert coordinator.current_mode == "plan"
    assert coordinator.set_mode("build") is True
    assert coordinator.current_mode == "build"
    assert coordinator.set_mode("invalid") is False