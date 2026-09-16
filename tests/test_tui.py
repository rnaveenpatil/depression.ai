"""TUI (Textual) interaction tests.

Verifies the two TUI regressions users hit in the terminal demo:
  1. Typing characters into the prompt actually lands in the Input widget.
  2. Permission 'ask' verdicts surface an in-UI ALLOW/DENY modal (instead of
     the old silent stdin-confirm path that auto-denied every request).
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent.permissions.manager import PermissionRequest, PermissionVerdict, RiskLevel
from agent.tui.app import DepressionApp
from agent.tui.app import PermissionModal
from textual.widgets import Button, Input


@pytest.mark.asyncio
async def test_tui_prompt_accepts_typed_characters():
    """Keystrokes must land in the #prompt Input (focus is set on mount and
    restored after every agent run / permission modal)."""
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        assert inp.has_focus, "prompt must be focused on mount"
        await pilot.press("h", "e", "l", "l", "o", " ", "p", "a", "t", "i", "l")
        assert inp.value == "hello patil"


@pytest.mark.asyncio
async def test_tui_permission_request_renders_allow_deny_modal():
    """An 'ask' verdict must be answered in the UI: pressing ALLOW returns True,
    DENY returns False — never an untraceable silent denial."""
    req = PermissionRequest(
        tool="terminal", action="execute",
        params={"command": "git push"}, request_id="t1",
    )
    verdict = PermissionVerdict.ask("default ask for terminal", risk=RiskLevel.MEDIUM)

    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        worker = app.run_worker(
            app._tui_confirm(req, verdict), name="test-confirm-allow")
        await pilot.pause()
        await pilot.pause()
        modal = next(
            (s for s in app.screen_stack if isinstance(s, PermissionModal)), None)
        assert modal is not None, "permission modal must be pushed"
        assert modal.query_one("#perm-allow", Button).has_focus, (
            "allow should be focused for medium risk")
        await pilot.pause()
        await pilot.press("enter")
        assert await asyncio.wait_for(worker.wait(), 5) is True


@pytest.mark.asyncio
async def test_tui_permission_deny_button_returns_false():
    req = PermissionRequest(
        tool="delete", action="file", params={"path": "/etc/passwd"}, request_id="t2",
    )
    verdict = PermissionVerdict.ask("delete outside workspace", risk=RiskLevel.HIGH)

    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        worker = app.run_worker(
            app._tui_confirm(req, verdict), name="test-confirm-deny")
        await pilot.pause()
        await pilot.pause()
        modal = next(
            (s for s in app.screen_stack if isinstance(s, PermissionModal)), None)
        assert modal is not None
        assert modal.query_one("#perm-deny", Button).has_focus, (
            "deny should be focused for high risk")
        await pilot.pause()
        await pilot.press("enter")
        assert await asyncio.wait_for(worker.wait(), 5) is False

@pytest.mark.asyncio
async def test_tui_submit_while_busy_keeps_typed_text():
    """Submitting while the agent is busy must NOT wipe the typed message —
    it stays in the box and can be resubmitted when the task finishes."""
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.focus()
        await pilot.press("k", "e", "e", "p", " ", "m", "y", " ", "t", "e", "x", "t")

        app.busy = True  # simulate an in-flight agent task
        await pilot.press("enter")
        await pilot.pause()

        assert inp.value == "keep my text", f"typed text was dropped: {inp.value!r}"
        assert app.busy is True


@pytest.mark.asyncio
async def test_tui_sidebar_click_keeps_prompt_focus_and_typing():
    """Clicking sidebar tabs / buttons must not steal focus from #prompt —
    typed characters keep landing in the input box."""
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        assert inp.has_focus

        await pilot.click("#connect")
        await pilot.pause()
        await pilot.click("#aws-save")
        await pilot.pause()

        assert inp.has_focus, "focus must stay on the prompt input"
        await pilot.press("s", "t", "i", "l", "l", " ", "t", "y", "p", "i", "n", "g")
        assert inp.value == "still typing"


@pytest.mark.asyncio
async def test_tui_wiring_neutralizes_blocking_stdin_fallback():
    """The TUI must never let a permission ask fall back to a blocking CLI
    prompt (prompt_toolkit on the Textual-owned stdin). The modal callback is
    wired AND the input_handler fallback is neutered on both agents."""
    from agent.agent.dual_agent import AgentRole, AgentCoordinator
    from agent.permissions.manager import PermissionManager
    from tests.test_agents_roles import _make_workspace, _make_agent

    d = Path(tempfile.mkdtemp(prefix="wire2_"))
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    ws = _make_workspace(d)
    await ws.initialize()
    plan, _ = await _make_agent(ws, AgentRole.PLAN,
                                [{"label": "p", "content": "plan", "tool_calls": []}],
                                with_planner=False)
    build, _ = await _make_agent(ws, AgentRole.BUILD,
                                 [{"label": "g", "content": "go", "tool_calls": []}],
                                 with_planner=False)
    coord = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    app = DepressionApp(coordinator=coord)
    app._wire_permission_confirmation()
    for agent in (coord.plan_agent, coord.build_agent):
        pm: PermissionManager = agent.permission_manager
        assert pm.confirm_callback is not None
        assert pm.input_handler is None, "blocking stdin fallback must be neutered"

    # A failed/absent callback must dead-end at a non-blocking DENY (no hang).
    from agent.permissions.manager import PermissionRequest, PermissionVerdict
    pm = build.permission_manager
    calls = {"ask": None}

    async def _boom(request, verdict):
        calls["ask"] = request.tool
        raise RuntimeError("no active worker")

    pm.set_confirm_callback(_boom)
    allowed = await pm._ask_user(
        PermissionRequest(tool="bash", action="execute", params={"command": "echo hi"}),
        PermissionVerdict.ask("test"),
    )
    assert allowed is False
    assert calls["ask"] == "bash"


@pytest.mark.asyncio
async def test_tui_permission_modal_end_to_end_allows_tool_run():
    """Full real path: agent asks permission inside the run worker → the
    ALLOW/DENY modal appears and is RESPONSIVE → pressing the default button
    allows and the bash tool actually executes."""
    from agent.agent.dual_agent import AgentRole, AgentCoordinator
    from agent.permissions.manager import PermissionManager
    from tests.test_agents_roles import _make_workspace, _make_agent

    d = Path(tempfile.mkdtemp(prefix="e2epm_"))
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    ws = _make_workspace(d)
    await ws.initialize()

    script = [
        {"label": "e2e-pm", "content": "running", "tool_calls": [
            {"name": "bash", "arguments": {"command": "printf e2e-panel-ok > out.txt"}}]},
        {"label": "e2e-pm-done", "content": "Done.", "tool_calls": []},
    ]
    plan, _ = await _make_agent(
        ws, AgentRole.PLAN, [{"label": "p", "content": "plan", "tool_calls": []}],
        with_planner=False)
    build, _ = await _make_agent(ws, AgentRole.BUILD, script, with_planner=False)
    build.permission_manager = PermissionManager(
        {"mode": "manual", "*": "ask", "default": "allow"}, ui=None, input_handler=None)
    coord = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    app = DepressionApp(coordinator=coord)
    app._wire_permission_confirmation()

    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.value = "create a flag file"
        await pilot.press("enter")
        # let the agent worker reach the permission ask
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, PermissionModal):
                break
        assert isinstance(app.screen, PermissionModal), "ask must surface the modal"
        await pilot.press("enter")  # default: ALLOW (medium risk)
        for _ in range(30):
            await pilot.pause(0.05)
            if not isinstance(app.screen, PermissionModal):
                break
        assert not isinstance(app.screen, PermissionModal)
        assert (d / "out.txt").read_text() == "e2e-panel-ok", "tool must have run"


@pytest.mark.asyncio
async def test_tui_permission_modal_key_shortcuts():
    """y/a allow, n/d/escape deny — the panel responds even without focus."""
    for key, expected in (("y", True), ("a", True), ("n", False),
                          ("d", False), ("escape", False)):
        app = DepressionApp(coordinator=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            result = {}
            async def _push():
                result["allowed"] = await app.push_screen_wait(
                    PermissionModal(
                        PermissionRequest(tool="bash", action="execute",
                                          params={"command": "ls"}),
                        PermissionVerdict.ask("demo")))
            app.run_worker(_push, exclusive=False)
            await pilot.pause(0.2)
            assert isinstance(app.screen, PermissionModal)
            await pilot.press(key)
            await pilot.pause(0.2)
            assert result.get("allowed") == expected, f"{key!r} -> expected {expected}"


@pytest.mark.asyncio
async def test_tui_permission_confirm_responds_outside_worker():
    """The permission panel must stay interactive even when the ask arrives
    outside a Textual worker (previously push_screen_wait raised
    NoActiveWorker and the panel could not be answered)."""
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        task = asyncio.ensure_future(app._tui_confirm(
            PermissionRequest(tool="bash", action="execute",
                              params={"command": "ls"}),
            PermissionVerdict.ask("demo")))
        await pilot.pause(0.3)
        assert isinstance(app.screen, PermissionModal), "modal must appear"
        await pilot.click("#perm-allow")
        assert await task is True
        await pilot.pause(0.1)
        assert not isinstance(app.screen, PermissionModal)
