"""TUI slash-command discovery: filtering, keyboard behavior, and safety."""
from __future__ import annotations

import pytest

from agent.tui.app import (
    AgentCommandProvider,
    DepressionApp,
    PromptArea,
    _slash_command_matches,
)


def test_slash_menu_is_backed_by_actual_commands():
    names = {name for name, _ in AgentCommandProvider.COMMANDS}
    for query in ("/", "/c", "/mo", "/mcp", "/rn", "/h"):
        assert {name for name, _ in _slash_command_matches(query)} <= names
    assert _slash_command_matches("/mcp")[0][0] == "/mcp"
    assert _slash_command_matches("mcp") == []
    assert _slash_command_matches("/mcp argument") == []
    assert _slash_command_matches("/doesnotexist") == []
    assert _slash_command_matches("/")  # Opens immediately.


@pytest.mark.asyncio
async def test_slash_menu_filters_as_user_types():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("/")
        immediate = (app._command_menu_open, prompt.text, app._command_suggestion_matches)
        await pilot.pause()
        assert app._command_menu_open, repr((immediate, prompt.text, app._command_suggestion_matches, app._command_suppressed_text))
        assert app.query_one("#command-menu").display is True
        prompt.set_prompt_text("/mcp")
        await pilot.pause()
        assert app._command_suggestion_matches[0][0] == "/mcp"
        prompt.set_prompt_text("ordinary question")
        await pilot.pause()
        assert not app._command_menu_open


@pytest.mark.asyncio
async def test_tab_inserts_command_without_executing():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("/mcp")
        await pilot.pause()
        app.action_tab_or_mode()
        await pilot.pause()
        assert prompt.text == "/mcp"
        assert not app._command_menu_open
        assert app._active_panel == "llm"


@pytest.mark.asyncio
async def test_enter_on_command_suggestion_dispatches_original_command():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("/mcp")
        await pilot.pause()
        assert app._accept_command_suggestion(execute=True)
        await pilot.pause()
        assert prompt.text == ""
        assert app._active_panel == "mcp"


@pytest.mark.asyncio
async def test_argument_command_is_never_run_without_arguments():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("/run")
        await pilot.pause()
        assert app._accept_command_suggestion(execute=True)
        await pilot.pause()
        assert prompt.text == "/run "
        assert not app._command_menu_open


@pytest.mark.asyncio
async def test_escape_dismisses_menu_without_interrupting_agent():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("/m")
        await pilot.pause()
        assert app._command_menu_open
        assert app.check_action("interrupt", ()) is False
        assert not app._command_menu_open
        assert prompt.text == "/m"
        # Next character allows suggestions to open again.
        prompt.set_prompt_text("/mc")
        await pilot.pause()
        assert app._command_menu_open


@pytest.mark.asyncio
async def test_existing_file_reference_and_multiline_input_remain_available():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptArea)
        prompt.set_prompt_text("explain")
        prompt.insert_file_reference("src/agent/tui/app.py")
        await pilot.pause()
        assert "@src/agent/tui/app.py" in prompt.text
        assert not app._command_menu_open
