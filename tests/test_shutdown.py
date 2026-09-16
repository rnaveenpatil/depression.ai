"""Shutdown regressions.

Covers the Ctrl+C fix: signal handlers must not leak dangling coroutines, and
BaseAgent / AgentCoordinator shutdown must be idempotent so the TUI and the
normal exit path can both run it without double-teardown.
"""
from __future__ import annotations

import signal

import pytest

from agent.agent.dual_agent import AgentCoordinator, AgentRole
from agent.main import CLIAgent

from tests.test_agents_roles import _make_agent, _make_workspace

_NOTES = "hello world\n"


def _write_notes(tmp_path):
    (tmp_path / "notes.txt").write_text(_NOTES)


class _CountingLLM:
    def __init__(self):
        self.reconnects = 0

    async def reconnect_all(self):
        self.reconnects += 1

    def get_current_model(self):
        return "vendor/test-model"

    def get_current_provider(self):
        return "test"


def test_cli_signal_handler_raises_instead_of_dangling_task():
    """First ^C cancels the loop via KeyboardInterrupt (no never-awaited coroutine)."""
    cli = CLIAgent()
    cli.ui = None
    with pytest.raises(KeyboardInterrupt):
        cli._on_signal(signal.SIGINT, None)
    assert cli.running is False
    assert cli._signals == 1


@pytest.mark.asyncio
async def test_base_agent_shutdown_is_idempotent(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    _write_notes(tmp_path)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [], with_planner=True)
    llm = _CountingLLM()
    plan.llm = llm

    await plan.shutdown()
    assert plan._shutdown_done is True
    assert llm.reconnects == 1
    await plan.shutdown()  # no-op second call
    assert llm.reconnects == 1


@pytest.mark.asyncio
async def test_coordinator_shutdown_is_idempotent(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    _write_notes(tmp_path)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [], with_planner=True)
    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)
    plan.llm = _CountingLLM()
    build.llm = _CountingLLM()

    coordinator = AgentCoordinator(
        plan_agent=plan,
        build_agent=build,
        config={},
    )

    await coordinator.shutdown()
    assert coordinator._shutdown_done is True
    assert plan.llm.reconnects == 1 and build.llm.reconnects == 1
    await coordinator.shutdown()  # no-op second call
    assert plan.llm.reconnects == 1 and build.llm.reconnects == 1