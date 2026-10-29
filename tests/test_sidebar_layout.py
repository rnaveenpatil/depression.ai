"""Focused UI tests: compact sidebar and mode-specific MCP paste fields."""
import pytest
from textual.widgets import Button, Input, TextArea
from agent.tui.app import DepressionApp
from agent.tui.widgets.mcp_panel import MCPPanel
from agent.tui.widgets.sidebar import Sidebar, SidebarTab


@pytest.mark.asyncio
async def test_navigation_uses_two_tabs_per_row_and_keeps_panel_fields_visible():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test(size=(120, 35)) as pilot:
        app.action_toggle_sidebar()
        await pilot.pause()
        sidebar = app.query_one("#sidebar", Sidebar)
        assert sidebar.is_open
        assert len(sidebar.query(".side-tab-row")) == 4
        assert len(sidebar.query(".side-tab")) == 8
        app._slash("/connect")
        await pilot.pause()
        assert app.query_one("#base-url", Input).display
        assert app.query_one("#api-key", Input).display
        assert app.query_one("#model-id", Input).display


@pytest.mark.asyncio
async def test_mcp_modes_show_only_relevant_fields_and_preserve_drafts():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test(size=(120, 35)) as pilot:
        app._slash("/mcp")
        await pilot.pause()
        panel = app.query_one("#panel-mcp", MCPPanel)
        assert app.focused is app.query_one("#mcp-url", Input)
        assert panel.query_one("#mcp-remote-fields").display
        assert not panel.query_one("#mcp-local-fields").display
        assert not panel.query_one("#mcp-json-fields").display
        app.query_one("#mcp-url", Input).value = "https://example.com/mcp"
        panel.set_entry_mode("local")
        await pilot.pause()
        assert panel.query_one("#mcp-local-fields").display
        assert not panel.query_one("#mcp-remote-fields").display
        app.query_one("#mcp-command", Input).value = "npx"
        assert panel.connection_values()["transport"] == "stdio"
        assert panel.connection_values()["url"] == ""
        panel.set_entry_mode("json")
        await pilot.pause()
        assert panel.query_one("#mcp-json-fields").display
        assert not panel.query_one("#mcp-manual-fields").display
        panel.query_one("#mcp-paste", TextArea).text = '{"mcpServers": {}}'
        assert panel.connection_values()["json"].startswith("{")
        assert panel.connection_values()["command"] == ""
        panel.set_entry_mode("http")
        assert app.query_one("#mcp-url", Input).value == "https://example.com/mcp"


@pytest.mark.asyncio
async def test_mcp_connection_details_are_collapsed_by_default():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test(size=(100, 30)) as pilot:
        app._slash("/mcp")
        await pilot.pause()
        panel = app.query_one("#panel-mcp", MCPPanel)
        assert not panel.query_one("#mcp-details").display
        panel._toggle_details()
        await pilot.pause()
        assert panel.query_one("#mcp-details").display
        panel._toggle_details()
        assert not panel.query_one("#mcp-details").display


@pytest.mark.asyncio
@pytest.mark.parametrize("screen_width", [80, 100, 120])
async def test_sidebar_buttons_have_height_for_visible_labels(screen_width):
    """Catch Textual's one-row Button clipping (seen as green lines)."""
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test(size=(screen_width, 35)) as pilot:
        app._show_panel("llm")
        await pilot.pause()
        sidebar = app.query_one("#sidebar", Sidebar)
        assert sidebar.is_open
        for tab in sidebar.query(".side-tab"):
            assert isinstance(tab, SidebarTab)
            assert tab.region.height == 1, (tab.id, tab.region)
            assert tab.region.width >= 8, (tab.id, tab.region)
        assert app.query_one("#side-content").region.height >= 10
        assert app.query_one("#base-url", Input).display


@pytest.mark.asyncio
async def test_one_line_sidebar_tabs_switch_panels_with_click_and_keyboard():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test(size=(120, 35)) as pilot:
        app._show_panel("llm")
        await pilot.pause()
        tab = app.query_one("#tab-mcp", SidebarTab)
        await pilot.click("#tab-mcp")
        await pilot.pause()
        assert app._active_panel == "mcp"
        assert tab.has_class("active")
        app.query_one("#tab-llm", SidebarTab).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app._active_panel == "llm"
