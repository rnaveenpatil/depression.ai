"""Read-only MCP overview for the Textual sidebar.

Server lifecycle actions are dispatched by DepressionApp. The panel never
executes an MCP tool, changes permissions, or displays auth credentials.
"""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, Input, Select, TextArea

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
    MCPPanel .mcp-label {{ color: {MUTED}; height: 1; margin-top: 1; }}
    MCPPanel Input, MCPPanel Select, MCPPanel TextArea {{
        border: round {BORDER}; background: {PANEL}; color: {TEXT}; width: 100%;
    }}
    MCPPanel Input, MCPPanel Select {{ height: 3; }}
    MCPPanel #mcp-paste {{ height: 6; }}
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
        yield Static("Paste URL, API token, command or JSON, then save.", classes="mcp-muted", markup=False)
        yield Static("Server name", classes="mcp-label")
        yield Input(id="mcp-name", placeholder="my-server")
        yield Static("Transport", classes="mcp-label")
        yield Select([
            ("HTTP", "streamable-http"), ("SSE", "sse"),
            ("Cloud", "cloud"), ("Local command", "stdio"),
        ], id="mcp-transport", value="streamable-http", allow_blank=False)
        yield Static("MCP URL", classes="mcp-label")
        yield Input(id="mcp-url", placeholder="https://example.com/mcp")
        yield Static("API token (optional)", classes="mcp-label")
        yield Input(id="mcp-token", password=True, placeholder="Paste API token")
        yield Static("Local executable (stdio)", classes="mcp-label")
        yield Input(id="mcp-command", placeholder="npx / uvx / python")
        yield Static("Arguments (stdio)", classes="mcp-label")
        yield Input(id="mcp-args", placeholder="-y @package/server")
        yield Static("Or paste MCP server JSON", classes="mcp-label")
        yield TextArea("", id="mcp-paste", soft_wrap=True, show_line_numbers=False)
        yield Button("SAVE & CONNECT", id="mcp-save")
        yield Static("", id="mcp-save-status", classes="mcp-muted", markup=False)
        yield Static("Checking MCP…", id="mcp-status", classes="mcp-detail", markup=False)
        yield Static("", id="mcp-servers", classes="mcp-detail", markup=False)
        yield Static("", id="mcp-tools", classes="mcp-muted", markup=False)
        with Horizontal(id="mcp-actions"):
            yield Button("REFRESH", id="mcp-refresh")
            yield Button("RECONNECT", id="mcp-reconnect")
        yield Static(
            "Paste one Claude-style mcpServers JSON object, or enter a URL. "
            "Existing servers remain. Reload reconnects configured servers.",
            classes="mcp-muted",
            markup=False,
        )

    def on_mount(self) -> None:
        self.refresh_status()

    def connection_values(self) -> dict[str, str]:
        return {
            "name": self.query_one("#mcp-name", Input).value,
            "transport": str(self.query_one("#mcp-transport", Select).value),
            "url": self.query_one("#mcp-url", Input).value,
            "api_key": self.query_one("#mcp-token", Input).value,
            "command": self.query_one("#mcp-command", Input).value,
            "args": self.query_one("#mcp-args", Input).value,
            "json": self.query_one("#mcp-paste", TextArea).text,
        }

    def set_save_status(self, message: str) -> None:
        self.query_one("#mcp-save-status", Static).update(message)

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
        self.query_one("#mcp-save", Button).disabled = bool(
            getattr(self._host_app, "busy", False)
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
