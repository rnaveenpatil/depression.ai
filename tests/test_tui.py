"""Tests for the current Textual TUI behavior."""
from __future__ import annotations

import asyncio

import pytest

from agent.permissions.manager import PermissionRequest, PermissionVerdict, RiskLevel
from agent.session.session import Session
from agent.tools.todo import TodoTool
from agent.tui.app import DepressionApp, PermissionModal
from rich.text import Text
from textual.widgets import Button, Input


@pytest.mark.asyncio
async def test_prompt_accepts_typed_characters():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        assert inp.has_focus
        await pilot.press("h", "e", "l", "l", "o")
        assert inp.value == "hello"


@pytest.mark.asyncio
async def test_permission_modal_default_allow_for_medium_risk():
    modal = PermissionModal(
        PermissionRequest(tool="bash", action="execute", params={"command": "echo ok"}),
        PermissionVerdict.ask("test", risk=RiskLevel.MEDIUM),
    )
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        pilot.app.push_screen(modal)
        await pilot.pause()
        assert modal.query_one("#perm-allow", Button).has_focus
        await pilot.press("enter")
        await pilot.pause()
        assert modal._done is True


@pytest.mark.asyncio
async def test_permission_modal_default_deny_for_high_risk():
    modal = PermissionModal(
        PermissionRequest(tool="delete", action="file", params={"path": "x"}),
        PermissionVerdict.ask("test", risk=RiskLevel.HIGH),
    )
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        pilot.app.push_screen(modal)
        await pilot.pause()
        assert modal.query_one("#perm-deny", Button).has_focus
        await pilot.press("enter")
        await pilot.pause()
        assert modal._done is True


@pytest.mark.asyncio
async def test_busy_visual_restores():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        app._set_busy_visual(True)
        await pilot.pause()
        assert inp.has_class("busy")
        assert inp.placeholder == app._BUSY_PLACEHOLDER
        app._set_busy_visual(False)
        await pilot.pause()
        assert not inp.has_class("busy")
        assert inp.placeholder == "ask the agent…"


@pytest.mark.asyncio
async def test_write_sanitizes_control_characters():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._write("clean\x1b[31mred\x07\x00\x08text", "system")
        await pilot.pause()
        rendered = " ".join(str(node.render()) for node in app.query_one("#transcript").children)
        assert "clean" in rendered and "red" in rendered and "text" in rendered
        assert "\x1b" not in rendered
        assert "\x07" not in rendered


# ----------------------------------------------------------------------
# TODO PANEL
# ----------------------------------------------------------------------


class _FakeCoordinator:
    """Minimal stand-in exposing the session lookup path the panel relies on."""

    def __init__(self, session=None):
        self.plan_agent = None
        self.build_agent = None
        if session is not None:
            self.build_agent = type("A", (), {"session": session})()

    def get_status(self):
        return {}


async def _render_todo_panel(session: Session, items):
    """Mount the app with a live session, seed todos, return the rendered markup."""

    async def seed(tool):
        for params in items:
            await tool.execute(params)

    return await _render_todo_panel_seeded(session, seed)


async def _render_todo_panel_seeded(session: Session, seed):
    """
    Mount the app with a live session, let the caller seed the todo store
    (it receives the tool so it can capture generated ids), return markup.
    """
    tool = TodoTool(session)
    await seed(tool)

    app = DepressionApp(coordinator=_FakeCoordinator(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        # The panel polls the shared store every 0.5s.
        await asyncio.sleep(0.6)
        await pilot.pause()
        panel = app.query_one("#panel-todo")
        try:
            return panel._last_render
        finally:
            TodoTool.forget_session(session)


@pytest.mark.asyncio
async def test_todo_panel_renders_empty_state():
    session = Session()
    markup = await _render_todo_panel(session, [])
    assert "no tasks yet" in markup


@pytest.mark.asyncio
async def test_todo_panel_shows_status_glyphs_and_counter():
    session = Session()
    markup = await _render_todo_panel(
        session,
        [
            {"action": "add", "title": "alpha", "priority": 1},
            {"action": "add", "title": "beta", "priority": 2},
            {"action": "add", "title": "gamma", "priority": 3},
        ],
    )
    # Header counter reflects the todo totals.
    assert "0/3 done" in markup
    # Every task title reaches the panel.
    assert "alpha" in markup and "beta" in markup and "gamma" in markup
    # All three are still pending.
    assert markup.count("○") == 3


@pytest.mark.asyncio
async def test_todo_panel_marks_completed_task_with_strike():
    async def seed(tool):
        first = await tool.execute({"action": "add", "title": "alpha"})
        await tool.execute({"action": "add", "title": "beta"})
        await tool.execute(
            {"action": "update", "todo_id": first["todo_id"], "status": "done"}
        )

    markup = await _render_todo_panel_seeded(Session(), seed)
    # Counter counts the completed item.
    assert "1/2 done" in markup
    # Done uses the filled glyph and is struck through.
    assert "●" in markup and "strike" in markup
    assert "alpha" in markup
    # The pending sibling keeps the hollow glyph and no strike.
    assert markup.count("○") == 1


@pytest.mark.asyncio
async def test_todo_panel_orders_in_progress_first():
    async def seed(tool):
        # alpha has the higher priority, so a plain priority/age sort would
        # place it first. Marking beta in_progress must override that.
        await tool.execute({"action": "add", "title": "alpha", "priority": 1})
        beta = await tool.execute({"action": "add", "title": "beta", "priority": 5})
        await tool.execute(
            {"action": "update", "todo_id": beta["todo_id"], "status": "in_progress"}
        )

    markup = await _render_todo_panel_seeded(Session(), seed)
    assert "▶" in markup
    assert markup.index("beta") < markup.index("alpha")
    assert "0/2 done" in markup


@pytest.mark.asyncio
async def test_todo_panel_escapes_markup_in_title():
    session = Session()
    title = "[bold red]not markup[/]"
    markup = await _render_todo_panel(session, [{"action": "add", "title": title}])
    # '[' is escaped so Rich treats the tag as literal text rather than markup.
    assert r"\[" in markup
    assert title in Text.from_markup(markup).plain


@pytest.mark.asyncio
async def test_todo_panel_truncates_long_title():
    session = Session()
    long_title = "x" * 80
    markup = await _render_todo_panel(session, [{"action": "add", "title": long_title}])
    assert "x" * 32 not in markup
    assert "…" in markup


@pytest.mark.asyncio
async def test_todo_panel_refreshes_after_mount():
    """The panel polls the shared store, so later tool writes still show up."""
    session = Session()

    async def seed(tool):
        await tool.execute({"action": "add", "title": "alpha"})

    app = DepressionApp(coordinator=_FakeCoordinator(session))
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one("#panel-todo")
            assert "no tasks yet" in panel._last_render

            await TodoTool(session).execute({"action": "add", "title": "late arrival"})
            await asyncio.sleep(0.6)
            await pilot.pause()
            assert "late arrival" in panel._last_render
    finally:
        TodoTool.forget_session(session)
