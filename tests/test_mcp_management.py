"""Focused tests for MCP server management and deletion."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from textual.widgets import Button, OptionList

from agent.mcp.client import MCPClient, MCPState
from agent.tui.app import DepressionApp
from agent.tui.mcp_config_ui import (
    delete_mcp_server,
    list_mcp_servers,
    save_mcp_server,
)
from agent.tui.widgets.mcp_panel import MCPPanel


def _saved(tmp_path, name: str, token: str):
    config = tmp_path / "config.json"
    env = tmp_path / "env"
    save_mcp_server(
        {
            "name": name,
            "url": "https://example.com/mcp",
            "transport": "streamable-http",
            "api_key": token,
        },
        config_path=config,
        env_path=env,
    )
    return config, env


def test_delete_server_removes_only_its_owned_secrets(tmp_path):
    config, env = _saved(tmp_path, "demo", "demo-secret")
    save_mcp_server(
        {
            "name": "other",
            "url": "https://example.com/other",
            "transport": "streamable-http",
            "api_key": "other-secret",
        },
        config_path=config,
        env_path=env,
    )

    result = delete_mcp_server("demo", config_path=config, env_path=env)

    document = json.loads(config.read_text())
    names = [server["name"] for server in document["mcp"]["servers"]]
    assert names == ["other"]
    assert result["secrets_removed"] == ["DEPRESSION_MCP_DEMO_API_KEY"]
    env_text = env.read_text()
    assert "DEPRESSION_MCP_DEMO_API_KEY" not in env_text
    assert "DEPRESSION_MCP_OTHER_API_KEY=other-secret" in env_text


def test_delete_keeps_secret_still_referenced_by_another_server(tmp_path):
    config, env = _saved(tmp_path, "demo", "shared-secret")
    save_mcp_server(
        {
            "json": json.dumps({
                "mcpServers": {
                    "other": {
                        "url": "https://example.com/other",
                        "headers": {
                            "Authorization": "${DEPRESSION_MCP_DEMO_API_KEY}"
                        },
                    }
                }
            })
        },
        config_path=config,
        env_path=env,
    )

    result = delete_mcp_server("demo", config_path=config, env_path=env)

    assert result["secrets_removed"] == []
    assert "DEPRESSION_MCP_DEMO_API_KEY=shared-secret" in env.read_text()


def test_delete_enabled_preset_disables_it(tmp_path):
    config = tmp_path / "config.json"
    env = tmp_path / "env"
    config.write_text(json.dumps({
        "mcp": {
            "enabled": True,
            "servers": [],
            "presets": {"git": {"enabled": True, "repo_path": "."}},
        }
    }))
    config.chmod(0o600)
    assert list_mcp_servers(config_path=config)[0]["name"] == "git"

    result = delete_mcp_server("git", config_path=config, env_path=env)

    assert result["kind"] == "preset"
    assert json.loads(config.read_text())["mcp"]["presets"]["git"]["enabled"] is False
    assert list_mcp_servers(config_path=config) == []


@pytest.mark.asyncio
async def test_disconnect_server_only_removes_that_servers_state():
    class Transport:
        state = MCPState.READY

        def __init__(self):
            self.disconnected = False

        async def disconnect(self):
            self.disconnected = True

    client = MCPClient({"enabled": True, "servers": []})
    first, second = Transport(), Transport()
    client.transports = {"first": first, "second": second}
    client.tools = {
        "mcp__first__read": SimpleNamespace(server="first"),
        "mcp__second__read": SimpleNamespace(server="second"),
    }
    client.resources = {"first:file": object(), "second:file": object()}
    client.prompts = {"first:prompt": object(), "second:prompt": object()}

    assert await client.disconnect_server("first") is True

    assert first.disconnected is True
    assert second.disconnected is False
    assert set(client.transports) == {"second"}
    assert set(client.tools) == {"mcp__second__read"}
    assert set(client.resources) == {"second:file"}
    assert set(client.prompts) == {"second:prompt"}


@pytest.mark.asyncio
async def test_mcp_panel_lists_servers_and_delete_is_available(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mcp": {"enabled": True, "servers": [
        {"name": "alpha", "transport": "stdio", "command": "python"},
        {"name": "beta", "transport": "stdio", "command": "python"},
    ]}}))
    monkeypatch.setattr(
        "agent.tui.mcp_config_ui.USER_CONFIG_PATH", config,
    )

    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        app._slash("/mcp")
        await pilot.pause()
        panel = app.query_one("#panel-mcp", MCPPanel)
        listing = panel.query_one("#mcp-server-list", OptionList)
        assert listing.option_count == 2
        assert panel.selected_server_name() in {"alpha", "beta"}
        assert not panel.query_one("#mcp-server-delete", Button).disabled
        assert panel.query_one("#mcp-server-connect", Button).disabled
        assert panel.query_one("#mcp-server-disconnect", Button).disabled
