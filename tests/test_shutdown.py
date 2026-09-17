"""Shutdown behavior for the current agent implementation."""
from __future__ import annotations

import signal

import pytest

from agent.agent.dual_agent import AgentCoordinator, AgentRole, BuildAgent, PlanAgent
from agent.main import CLIAgent


class CountingTool:
    name = "test-tool"

    async def shutdown(self):
        self.calls += 1

    def __init__(self):
        self.calls = 0


def test_cli_signal_handler_stops_cleanly():
    cli = CLIAgent()
    cli.ui = None
    with pytest.raises(KeyboardInterrupt):
        cli._on_signal(signal.SIGINT, None)
    assert cli.running is False
    assert cli._signals == 1


@pytest.mark.asyncio
async def test_base_agent_shutdown_is_idempotent():
    agent = PlanAgent(
        config={}, session=None, context_manager=None,
        permission_manager=None, workspace=None, database=None,
    )
    tool = CountingTool()
    agent.tool_registry = type("Registry", (), {"tools": {tool.name: tool}})()
    await agent.shutdown()
    await agent.shutdown()
    assert agent._shutdown_done is True
    assert agent.is_shutting_down is True
    assert tool.calls == 1


@pytest.mark.asyncio
async def test_coordinator_shutdown_is_idempotent():
    plan = PlanAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    build = BuildAgent(config={}, session=None, context_manager=None, permission_manager=None, workspace=None, database=None)
    plan_tool = CountingTool()
    build_tool = CountingTool()
    plan.tool_registry = type("Registry", (), {"tools": {plan_tool.name: plan_tool}})()
    build.tool_registry = type("Registry", (), {"tools": {build_tool.name: build_tool}})()
    coordinator = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    await coordinator.shutdown()
    await coordinator.shutdown()
    assert coordinator._shutdown_done is True
    assert plan_tool.calls == 1
    assert build_tool.calls == 1
