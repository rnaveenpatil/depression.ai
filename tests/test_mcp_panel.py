"""Focused tests for the TUI MCP panel and tool-registry cleanup."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from agent.tui.app import DepressionApp
from agent.tui.widgets.mcp_panel import MCPPanel
from agent.tools.registry import ToolRegistry


class DummyClient:
    enabled = True

    def __init__(self):
        self.reload_count = 0
        self.servers_cfg = [
            SimpleNamespace(
                name="example", transport=SimpleNamespace(value="stdio")
            )
        ]
        self.transports = {
            "example": SimpleNamespace(state=SimpleNamespace(value="ready"))
        }
        self.tools = {"mcp__example__read": object()}
        self.resources = {}
        self.prompts = {}

    def list_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Read a safe example",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self.tools
        ]

    async def reload(self):
        self.reload_count += 1


@pytest.mark.asyncio
async def test_mcp_panel_displays_shared_client_and_lists_servers():
    client = DummyClient()
    coordinator = SimpleNamespace(
        build_agent=SimpleNamespace(mcp_client=client),
        plan_agent=None,
    )
    app = DepressionApp(coordinator=coordinator, skip_onboarding=True)
    async with app.run_test() as pilot:
        app._slash("/mcp")
        await pilot.pause()
        panel = app.query_one("#panel-mcp", MCPPanel)
        assert app._active_panel == "mcp"
        assert panel.get_client() is client
        assert "1/1" in str(panel.query_one("#mcp-status", Static).render())
        assert "mcp__example__read" in str(
            panel.query_one("#mcp-tools", Static).render()
        )


@pytest.mark.asyncio
async def test_reconnect_does_not_change_config_or_register_stale_tools(monkeypatch):
    client = DummyClient()
    registry = ToolRegistry()
    registry.register_external(
        name="mcp__stale__old",
        handler=lambda **kwargs: None,
    )
    coordinator = SimpleNamespace(
        build_agent=SimpleNamespace(
            mcp_client=client,
            tool_registry=registry,
            loop=None,
        ),
        plan_agent=None,
    )
    app = DepressionApp(coordinator=coordinator, skip_onboarding=True)
    async with app.run_test() as pilot:
        app._slash("/mcp")
        app._reconnect_mcp()
        await pilot.pause()
        for _ in range(20):
            await pilot.pause()
            if client.reload_count:
                break
        assert client.reload_count == 1
        assert registry.has_tool("mcp__example__read")
        assert not registry.has_tool("mcp__stale__old")
        assert not registry.is_read_only("mcp__example__read")
        assert registry.is_mutating("mcp__example__read")


@pytest.mark.asyncio
async def test_mcp_reconnect_is_disabled_while_agent_busy():
    client = DummyClient()
    app = DepressionApp(
        coordinator=SimpleNamespace(
            build_agent=SimpleNamespace(mcp_client=client),
            plan_agent=None,
        ),
        skip_onboarding=True,
    )
    async with app.run_test() as pilot:
        app._slash("/mcp")
        app.busy = True
        app.query_one("#panel-mcp", MCPPanel).refresh_status()
        assert app.query_one("#mcp-reconnect", Button).disabled
        app._reconnect_mcp()
        await pilot.pause()
        assert client.reload_count == 0
