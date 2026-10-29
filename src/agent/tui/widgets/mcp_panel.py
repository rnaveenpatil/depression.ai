"""Read-only MCP overview for the Textual sidebar.

Server lifecycle actions are dispatched by DepressionApp. The panel never
executes an MCP tool, changes permissions, or displays auth credentials.
"""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from agent.tui.theme import GREEN, MUTED, TEXT, PANEL, BORDER


class MCPPanel(Vertical):
    DEFAULT_CSS = f"""
    MCPPanel {{
        height: auto;
        padding: 0 1;
        background: transparent;
    }}
    MCPPanel .mcp-heading {{
        height: auto;
        color: {GREEN};
        text-style: bold;
        margin-bottom: 1;
    }}
    MCPPanel .mcp-detail {{
        height: auto;
        color: {TEXT};
        margin-bottom: 1;
    }}
    MCPPanel .mcp-muted {{
        height: auto;
        color: {MUTED};
        margin-bottom: 1;
    }}
    MCPPanel #mcp-actions {{
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }}
    MCPPanel #mcp-actions Button {{
        width: 1fr;
        min-width: 0;
        height: 3;
        border: round {BORDER};
        background: {PANEL};
        color: {GREEN};
    }}
    """

    def __init__(self, host_app: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._host_app = host_app

    def compose(self) -> ComposeResult:
        yield Static("▌ MCP SERVERS", classes="mcp-heading", markup=False)
        yield Static("Checking MCP…", id="mcp-status", classes="mcp-detail", markup=False)
        yield Static("", id="mcp-servers", classes="mcp-detail", markup=False)
        yield Static("", id="mcp-tools", classes="mcp-muted", markup=False)
        with Horizontal(id="mcp-actions"):
            yield Button("REFRESH", id="mcp-refresh")
            yield Button("RECONNECT", id="mcp-reconnect")
        yield Static(
            "Configure servers in ~/.agent/config.json under 'mcp', "
            "then restart depression.ai. Reconnect retries currently configured servers.",
            classes="mcp-muted",
            markup=False,
        )

    def on_mount(self) -> None:
        self.refresh_status()

    def get_client(self) -> Any:
        """Find the shared MCP client without creating one or changing settings."""
        coordinator = getattr(self._host_app, "coordinator", None)
        for role in ("build_agent", "plan_agent"):
            agent = getattr(coordinator, role, None)
            client = getattr(agent, "mcp_client", None)
            if client is not None:
                return client
        return None

    def refresh_status(self) -> None:
        """Display configured and live server state without leaking credentials."""
        client = self.get_client()
        if not self.is_mounted:
            return

        reconnect = self.query_one("#mcp-reconnect", Button)
        reconnect.disabled = (
            client is None or not bool(getattr(client, "enabled", False))
            or bool(getattr(self._host_app, "busy", False))
        )
        status = self.query_one("#mcp-status", Static)
        servers = self.query_one("#mcp-servers", Static)
        tools = self.query_one("#mcp-tools", Static)

        if client is None or not getattr(client, "enabled", False):
            status.update("MCP disabled or not initialized")
            servers.update("No active MCP connection.")
            tools.update("Enable MCP in your config and restart.")
            return

        configurations = list(getattr(client, "servers_cfg", ()) or ())
        transports = getattr(client, "transports", {}) or {}
        tool_list = list(getattr(client, "tools", {}).keys())
        connected = 0
        lines = []
        for cfg in configurations[:16]:
            name = str(getattr(cfg, "name", "unnamed"))
            transport = transports.get(name)
            state = getattr(getattr(transport, "state", None), "value", None)
            state = str(state or "disconnected")
            if state in ("connected", "ready"):
                connected += 1
            transport_kind = getattr(getattr(cfg, "transport", None), "value", "unknown")
            lines.append(f"• {name} ({transport_kind}): {state}")

        status.update(
            f"{connected}/{len(configurations)} configured server(s) connected"
        )
        if len(configurations) > 16:
            lines.append(f"… and {len(configurations) - 16} more")
        servers.update("\n".join(lines) if lines else "No MCP servers configured.")

        resources = len(getattr(client, "resources", {}) or {})
        prompts = len(getattr(client, "prompts", {}) or {})
        visible_names = tool_list[:8]
        summary = f"{len(tool_list)} tools · {resources} resources · {prompts} prompts"
        if visible_names:
            summary += "\n" + "\n".join(f"  {name}" for name in visible_names)
        if len(tool_list) > len(visible_names):
            summary += f"\n  … and {len(tool_list) - len(visible_names)} more"
        tools.update(summary)


__all__ = ["MCPPanel"]
