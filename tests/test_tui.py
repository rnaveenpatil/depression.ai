"""Tests for the current Textual TUI behavior."""
from __future__ import annotations

import asyncio

import pytest

from agent.permissions.manager import PermissionRequest, PermissionVerdict, RiskLevel
from agent.tui.app import DepressionApp, PermissionModal
from textual.widgets import Button, TextArea


@pytest.mark.asyncio
async def test_prompt_accepts_typed_characters():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", TextArea)
        assert inp.has_focus
        await pilot.press("h", "e", "l", "l", "o")
        assert inp.text == "hello"


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
        inp = app.query_one("#prompt", TextArea)
        app._set_busy_visual(True)
        await pilot.pause()
        assert inp.has_class("busy")
        app._set_busy_visual(False)
        await pilot.pause()
        assert not inp.has_class("busy")


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


@pytest.mark.asyncio
async def test_error_line_escapes_brackets_exactly_once():
    """Brackets in dynamic text must render literally, not as literal backslashes.

    ``_error``/``_system``/``_queued`` funnel through ``_write``, which already
    escapes ``[`` for markup. Escaping a second time upstream would leak
    ``\\[INFO]`` into the transcript instead of ``[INFO]``.
    """
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._error("[INFO] tool failed")
        await pilot.pause()
        rendered = " ".join(str(node.render()) for node in app.query_one("#transcript").children)
        assert "[INFO] tool failed" in rendered
        assert "\\[INFO]" not in rendered
        assert "\\\\[" not in rendered


@pytest.mark.asyncio
async def test_system_line_escapes_brackets_exactly_once():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._system("model [gpt-4o] ready")
        await pilot.pause()
        rendered = " ".join(str(node.render()) for node in app.query_one("#transcript").children)
        assert "[gpt-4o] ready" in rendered
        assert "\\[gpt-4o]" not in rendered


# ---------------------------------------------------------------------------
# Tool-call card rendering: truncation must be discoverable, never silent.
# ---------------------------------------------------------------------------


def _plain(markup: str) -> str:
    """Strip Textual markup tags so assertions read the visible text."""
    import re

    return re.sub(r"\[/?[^\]]*\]", "", markup)


def _search_widget(n_matches: int, count: int | None = None) -> "ToolCallWidget":
    from agent.tui.widgets.tool_call import ToolCallWidget

    widget = ToolCallWidget("grep", {"pattern": "loop"})
    widget._result = {
        "count": n_matches if count is None else count,
        "pattern": "loop",
        "matches": [{"path": f"f{i}.py", "text": "hit"} for i in range(n_matches)],
    }
    return widget


def test_search_preview_hints_at_hidden_matches():
    """Collapsed cards advertise that more rows exist."""
    widget = _search_widget(9)
    collapsed = _plain(widget._render_search(widget._result))

    assert "press ctrl+e" in collapsed
    assert "f8.py" not in collapsed, "rows beyond the preview window leaked through"


def test_search_expand_is_strictly_larger_and_drops_the_hint():
    """Expanding reveals strictly more rows and stops advertising `e`."""
    widget = _search_widget(9)
    collapsed = _plain(widget._render_search(widget._result))

    widget._expanded = True
    expanded = _plain(widget._render_search(widget._result))

    assert "f8.py" in expanded, "expansion failed to reveal the hidden rows"
    assert expanded.count("\n") > collapsed.count("\n")
    assert "press ctrl+e" not in expanded


def test_search_reports_matches_withheld_by_the_tool_payload():
    """When the tool itself clipped `matches`, say so instead of lying."""
    widget = _search_widget(3, count=25)
    widget._expanded = True

    rendered = _plain(widget._render_search(widget._result))
    assert "not in payload" in rendered


def test_long_tool_error_is_preserved_and_flagged():
    """A 3k-char error must not be silently cut to 800 chars."""
    from agent.tui.widgets.tool_call import ToolCallWidget

    widget = ToolCallWidget("bash", {"command": "boom"})
    err = "E" * 3000

    collapsed = widget._render_error(err)
    assert "press ctrl+e" in collapsed
    assert "more char" in collapsed
    assert len(_plain(collapsed)) < 2000, "collapsed error should stay compact"

    widget._expanded = True
    expanded = widget._render_error(err)
    assert len(_plain(expanded)) > 2500, "expanded error truncated the payload"


def test_aws_error_path_uses_discoverable_truncation():
    """The AWS renderer must not clip errors behind the user's back."""
    from agent.tui.widgets.tool_call import ToolCallWidget

    widget = ToolCallWidget("mcp__aws__s3_list_objects", {})
    result = {"success": False, "error": "AccessDenied: " + "z" * 1200}

    rendered = _plain(widget._render_aws(result))
    assert "AccessDenied" in rendered
    assert "press ctrl+e" in rendered


def test_web_body_expands_to_full_text():
    """`e` on a fetched page shows the whole body, not a 180-char teaser."""
    from agent.tui.widgets.tool_call import ToolCallWidget

    widget = ToolCallWidget("webfetch", {"url": "https://x.test"})
    widget._result = {
        "status": 200,
        "url": "https://x.test",
        "content": "line\n" * 60,
    }

    assert "press ctrl+e" in _plain(widget._render_web(widget._result))

    widget._expanded = True
    expanded = _plain(widget._render_web(widget._result))
    assert "press ctrl+e" not in expanded
    assert expanded.count("line") >= 60


@pytest.mark.asyncio
async def test_modal_tab_does_not_cycle_app_mode():
    modal = PermissionModal(
        PermissionRequest(tool="bash", action="execute", params={"command": "echo ok"}),
        PermissionVerdict.ask("test", risk=RiskLevel.MEDIUM),
    )
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        before = app.current_mode
        app.push_screen(modal)
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert app.current_mode == before


@pytest.mark.asyncio
async def test_onboarding_runs_inside_a_worker(monkeypatch):
    from textual.worker import get_current_worker

    seen = []

    async def fake_push(app, **kwargs):
        seen.append(get_current_worker())
        return None

    monkeypatch.setattr(
        "agent.tui.onboarding.welcome.push_welcome_if_new_user",
        fake_push,
    )
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if seen:
                break

    assert seen, "onboarding did not run"


@pytest.mark.asyncio
async def test_concurrent_stream_tokens_create_one_widget():
    from agent.tui.events import AgentEvent

    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.gather(
            app._on_token(AgentEvent("on_token", {"delta": "a"})),
            app._on_token(AgentEvent("on_token", {"delta": "b"})),
        )
        await pilot.pause()
        widgets = [
            w for w in app.query("*") if type(w).__name__ == "StreamingOutput"
        ]
        assert len(widgets) == 1


def test_stop_phrase_requests_graceful_loop_stop():
    from types import SimpleNamespace

    class _FakeLoop:
        stopped = False

        def stop(self):
            self.stopped = True

    fake_loop = _FakeLoop()
    coordinator = SimpleNamespace(
        plan_agent=SimpleNamespace(loop=fake_loop),
        build_agent=None,
    )
    app = DepressionApp(coordinator=coordinator)
    app.busy = True

    app.submit_prompt_text("ok stop")

    assert fake_loop.stopped is True
    assert app._prompt_queue == []


def test_default_turn_timeout_is_one_hour(monkeypatch):
    monkeypatch.delenv("TUI_TURN_TIMEOUT", raising=False)
    app = DepressionApp(coordinator=None)
    assert app.turn_timeout == 3600.0
