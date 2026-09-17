"""Tests for the current Textual TUI behavior."""
from __future__ import annotations

import pytest

from agent.permissions.manager import PermissionRequest, PermissionVerdict, RiskLevel
from agent.tui.app import DepressionApp, PermissionModal
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
