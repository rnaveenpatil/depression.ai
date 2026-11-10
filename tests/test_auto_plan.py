"""Auto mode: one plan call, then one build loop."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent.dual_agent import AgentCoordinator


class FakeContextManager:
    def __init__(self):
        self.user_messages = []
        self.assistant_messages = []

    async def add_user_message(self, text, context=None):
        self.user_messages.append(text)

    async def add_assistant_message(self, text):
        self.assistant_messages.append(text)


def make_plan(goal: str):
    return SimpleNamespace(
        id="p1",
        goal=goal,
        tasks=[
            SimpleNamespace(id="t1", description="Add the parser", subtasks=[]),
            SimpleNamespace(id="t2", description="Verify the parser", subtasks=[]),
        ],
    )


def make_coordinator(planner, build_calls, build_response="done"):
    async def build_process_query(query, context=None, record_user=True, initial_plan=None):
        build_calls.append({
            "query": query,
            "record_user": record_user,
            "initial_plan": initial_plan,
        })
        return {"success": True, "response": build_response, "tool_calls": 2}

    plan_agent = SimpleNamespace(planner=planner, loop=None)
    build_agent = SimpleNamespace(
        loop=None,
        context_manager=FakeContextManager(),
        session=None,
        process_query=build_process_query,
    )
    return AgentCoordinator(
        plan_agent=plan_agent, build_agent=build_agent, config={}
    )


@pytest.mark.asyncio
async def test_auto_mode_plans_once_then_runs_a_single_build_loop():
    planner_calls = []
    build_calls = []

    class FakePlanner:
        async def create_plan(self, goal, context=None, constraints=None):
            planner_calls.append(goal)
            return make_plan(goal)

    coordinator = make_coordinator(FakePlanner(), build_calls)

    result = await coordinator.process_query(
        "Add a parser and then verify it across all files",
        mode="auto",
        auto_execute=True,
    )

    assert len(planner_calls) == 1
    assert len(build_calls) == 1
    assert build_calls[0]["initial_plan"] is not None
    assert build_calls[0]["record_user"] is False
    assert result["success"] is True
    assert "Add the parser" in result["plan"]
    assert result["execution"] == "done"


@pytest.mark.asyncio
async def test_auto_mode_always_plans_without_keyword_classification():
    planner_calls = []
    build_calls = []

    class FakePlanner:
        async def create_plan(self, goal, context=None, constraints=None):
            planner_calls.append(goal)
            return make_plan(goal)

    coordinator = make_coordinator(FakePlanner(), build_calls)

    result = await coordinator.process_query(
        "What is Python?", mode="auto", auto_execute=True,
    )

    assert planner_calls == ["What is Python?"]
    assert build_calls[0]["initial_plan"] is not None
    assert result["plan"] is not None
    assert result["success"] is True


@pytest.mark.asyncio
async def test_plan_mode_returns_structured_plan_without_executing():
    build_calls = []

    class FakePlanner:
        async def create_plan(self, goal, context=None, constraints=None):
            return make_plan(goal)

    coordinator = make_coordinator(FakePlanner(), build_calls)

    result = await coordinator.process_query(
        "Add a parser", mode="plan", auto_execute=True,
    )

    assert build_calls == []
    assert result["mode"] == "plan"
    assert result["execution"] is None
    assert "- [ ] Add the parser" in result["plan"]
