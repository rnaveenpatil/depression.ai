"""Regression tests for accuracy, stopping, and planning improvements."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse, ToolCall


class FakeContext:
    def __init__(self):
        self.messages = []

    async def add_message(self, role, content, metadata=None, pinned=False, **kwargs):
        self.messages.append(
            SimpleNamespace(
                role=role, content=content, metadata=metadata or {}, pinned=pinned
            )
        )

    async def add_system_message(self, content, pinned=True, **kwargs):
        await self.add_message("system", content, {"pinned": pinned}, pinned=pinned)

    async def add_assistant_message(self, content, metadata=None, **kwargs):
        await self.add_message("assistant", content, metadata or {})


class FakeAgent:
    def __init__(self, tool_impl):
        self.context_manager = FakeContext()
        self.workspace = None
        self.permission_manager = None
        self.tool_impl = tool_impl
        self.executed = []

    async def execute_tool(self, name, args):
        self.executed.append((name, args))
        return self.tool_impl(name, args)


class FakeRegistry:
    def __init__(self):
        self.tools = {
            name: object() for name in ("read", "edit", "write", "bash", "glob")
        }
        self.mutation_epoch = 0

    def list_tools(self):
        return list(self.tools)

    def has_tool(self, name):
        return name in self.tools

    def is_read_only(self, name):
        return name in ("read", "glob")

    def get_schemas(self):
        return [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in self.tools
        ]

    def select_for_task(self, intent):
        return self.get_schemas()


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.model = "test/model"

    def get_current_model(self):
        return self.model

    def get_current_provider(self):
        return "test"

    async def complete_with_tools(self, messages=None, **kwargs):
        self.calls += 1
        if not self.responses:
            return LLMResponse(content="done", model=self.model, provider="test")
        return self.responses.pop(0)

    async def complete(self, messages=None, **kwargs):
        self.calls += 1
        if not self.responses:
            return LLMResponse(content="done", model=self.model, provider="test")
        return self.responses.pop(0)


def tool_response(*calls):
    return LLMResponse(
        content="", model="test/model", provider="test", tool_calls=list(calls)
    )


def text_response(text):
    return LLMResponse(content=text, model="test/model", provider="test")


def make_loop(responses, tool_impl=None, **config):
    def default_tool(name, args):
        return {"success": True, "tool": name, "content": "ok"}

    agent = FakeAgent(tool_impl or default_tool)
    loop = AgentLoop(
        agent,
        ScriptedLLM(responses),
        FakeRegistry(),
        None,
        {
            "enable_intent_classification": False,
            "enable_planning": False,
            "enable_streaming": False,
            "require_demo_offer": False,
            "verification_mode": "batch",
            **config,
        },
    )
    return loop, agent


@pytest.mark.asyncio
async def test_root_cause_gate_blocks_edit_until_evidence_is_observed():
    loop, agent = make_loop([text_response("done")])
    loop.context.intent = "bugfix"
    edit = ToolCall(
        id="e1", name="edit", arguments={"filePath": "a.py", "old": "x", "new": "y"}
    )

    refused = await loop._act([edit])
    assert refused[0]["success"] is False
    assert refused[0]["result"]["needs_evidence"] is True
    assert agent.executed == []

    await loop._act([ToolCall(id="r1", name="read", arguments={"filePath": "a.py"})])
    assert loop.context.metadata["evidence_observed"] is True

    done = await loop._act([
        ToolCall(
            id="e2",
            name="edit",
            arguments={"filePath": "a.py", "old": "x", "new": "y"},
        )
    ])
    assert done[0]["result"]["success"] is True
    assert ("edit", {"filePath": "a.py", "old": "x", "new": "y"}) in agent.executed


@pytest.mark.asyncio
async def test_goal_completion_can_succeed_after_iteration_limit():
    loop, _agent = make_loop(
        [
            tool_response(
                ToolCall(id="r1", name="read", arguments={"filePath": "a.py"})
            ),
            text_response("The file defines a function; nothing else is needed."),
        ]
    )

    result = await loop.run("explain a.py", max_turns=1)

    assert result["success"] is True, result
    assert loop.context.metadata.get("goal_achieved") is True


def test_checkpoint_records_actions_discoveries_and_decisions():
    loop, _agent = make_loop([text_response("done")])
    loop.context.add_action({
        "tool": "edit",
        "params": {"path": "a.py"},
        "result": {"success": True},
        "time": 0.1,
    })
    loop.context.metadata["discoveries"] = ["read:a.py", "glob:**/*.py"]
    loop.context.metadata["decisions"] = ["Use the existing helper"]

    checkpoint = loop._task_checkpoint()

    assert checkpoint["completed_tools"] == ["edit"]
    assert checkpoint["modified_files"] == ["a.py"]
    assert checkpoint["discoveries"] == ["read:a.py", "glob:**/*.py"]
    assert checkpoint["decisions"] == ["Use the existing helper"]


def test_simple_read_only_tasks_skip_planning_llm():
    loop, _agent = make_loop([text_response("done")])
    assert loop._is_simple_task("what does a.py do?", "question") is True
    assert loop._is_simple_task("refactor all modules", "refactor") is False


@pytest.mark.asyncio
async def test_state_context_recommends_search_before_edit():
    loop, _agent = make_loop([text_response("done")])
    loop.context.intent = "bugfix"

    state = await loop._get_state_context()

    assert state["evidence_observed"] is False
    assert "before editing" in state["recommended_next_step"]
