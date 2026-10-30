"""Compact, paste-first MCP connection panel with collapsible connection details."""
from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, Input, TextArea, OptionList
from textual.widgets.option_list import Option

from agent.tui.mcp_config_ui import list_mcp_servers
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
    MCPPanel .mcp-label {{
        color: {MUTED};
        height: 1;
        margin-top: 1;
    }}
    MCPPanel Input, MCPPanel TextArea {{
        border: round {BORDER};
        background: {PANEL};
        color: {TEXT};
        width: 100%;
    }}
    MCPPanel Input {{ height: 3; }}
    MCPPanel Input:focus, MCPPanel TextArea:focus {{
        border: round {GREEN};
    }}
    MCPPanel #mcp-mode-row {{
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }}
    MCPPanel #mcp-mode-row Button {{
        width: 1fr;
        min-width: 0;
        height: 3;
        padding: 0;
        border: round {BORDER};
        background: {PANEL};
        color: {MUTED};
    }}
    MCPPanel #mcp-mode-row Button.active {{
        border: round {GREEN};
        color: {GREEN};
        text-style: bold;
    }}
    MCPPanel #mcp-json-fields, MCPPanel #mcp-manual-fields,
    MCPPanel #mcp-remote-fields, MCPPanel #mcp-local-fields,
    MCPPanel #mcp-details {{
        height: auto;
        width: 100%;
    }}
    MCPPanel #mcp-json-fields, MCPPanel #mcp-local-fields,
    MCPPanel #mcp-details {{
        display: none;
    }}
    MCPPanel #mcp-paste {{ height: 7; }}
    MCPPanel #mcp-save {{
        height: 3;
        width: 100%;
        border: round {GREEN};
        background: {PANEL};
        color: {GREEN};
        text-style: bold;
        margin: 1 0;
    }}
    MCPPanel #mcp-details-toggle {{
        width: 100%;
        height: 3;
        border: round {BORDER};
        background: transparent;
        color: {MUTED};
        margin-top: 1;
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
    MCPPanel #mcp-server-list {{
        height: auto;
        max-height: 8;
        border: round {BORDER};
        background: {PANEL};
        color: {TEXT};
        margin-bottom: 1;
    }}
    MCPPanel #mcp-server-list:focus {{
        border: round {GREEN};
    }}
    MCPPanel #mcp-server-actions {{
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }}
    MCPPanel #mcp-server-actions Button {{
        width: 1fr;
        min-width: 0;
        height: 3;
        border: round {BORDER};
        background: {PANEL};
        color: {GREEN};
    }}
    MCPPanel #mcp-server-actions Button:disabled {{
        color: {MUTED};
        border: round {BORDER};
    }}
    """

    def __init__(self, host_app: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._host_app = host_app
        self._entry_mode = "http"
        self._details_expanded = False
        self._server_list_signature: tuple[tuple[str, str], ...] = ()

    def compose(self) -> ComposeResult:
        yield Static("▌ CONNECT MCP", classes="mcp-heading", markup=False)
        yield Static(
            "Paste connection details directly. No suggestions or presets.",
            classes="mcp-muted", markup=False,
        )
        with Horizontal(id="mcp-mode-row"):
            yield Button("HTTP", id="mcp-mode-http", classes="mcp-mode active")
            yield Button("SSE", id="mcp-mode-sse", classes="mcp-mode")
            yield Button("LOCAL", id="mcp-mode-local", classes="mcp-mode")
            yield Button("JSON", id="mcp-mode-json", classes="mcp-mode")
        with Vertical(id="mcp-manual-fields"):
            yield Static("Server name", classes="mcp-label")
            yield Input(id="mcp-name", placeholder="my-server")
            with Vertical(id="mcp-remote-fields"):
                yield Static("MCP server URL", classes="mcp-label")
                yield Input(id="mcp-url", placeholder="https://example.com/mcp")
                yield Static("API token (optional)", classes="mcp-label")
                yield Input(
                    id="mcp-token", password=True,
                    placeholder="Paste token; stored privately",
                )
            with Vertical(id="mcp-local-fields"):
                yield Static("Executable", classes="mcp-label")
                yield Input(id="mcp-command", placeholder="npx, uvx or python")
                yield Static("Arguments", classes="mcp-label")
                yield Input(id="mcp-args", placeholder="-y @package/server")
        with Vertical(id="mcp-json-fields"):
            yield Static(
                "Paste one MCP server JSON object or mcpServers config.",
                classes="mcp-muted", markup=False,
            )
            yield TextArea(
                "", id="mcp-paste",
                soft_wrap=True, show_line_numbers=False,
            )
        yield Button("SAVE & CONNECT", id="mcp-save")
        yield Static(
            "", id="mcp-save-status",
            classes="mcp-detail", markup=False,
        )
        yield Static(
            "Connection status: not checked", id="mcp-status",
            classes="mcp-muted", markup=False,
        )
        yield Static("▌ MANAGE SERVERS", classes="mcp-heading", markup=False)
        yield OptionList(id="mcp-server-list")
        with Horizontal(id="mcp-server-actions"):
            yield Button("CONNECT", id="mcp-server-connect")
            yield Button("DISCONNECT", id="mcp-server-disconnect")
            yield Button("DELETE", id="mcp-server-delete")
        yield Static(
            "Select a server to manage it.", id="mcp-server-status",
            classes="mcp-muted", markup=False,
        )
        yield Button("CONNECTION DETAILS  ▾", id="mcp-details-toggle")
        with Vertical(id="mcp-details"):
            yield Static(
                "", id="mcp-servers",
                classes="mcp-detail", markup=False,
            )
            yield Static(
                "", id="mcp-tools",
                classes="mcp-muted", markup=False,
            )
            with Horizontal(id="mcp-actions"):
                yield Button("REFRESH", id="mcp-refresh")
                yield Button("RECONNECT", id="mcp-reconnect")

    def on_mount(self) -> None:
        self.set_entry_mode(self._entry_mode)
        self.refresh_status()

    def set_entry_mode(self, mode: str) -> None:
        """Show only relevant paste inputs; never wipe drafts."""
        if mode not in ("http", "sse", "local", "json"):
            return
        self._entry_mode = mode
        if not self.is_mounted:
            return
        self.query_one("#mcp-manual-fields", Vertical).display = mode != "json"
        self.query_one("#mcp-json-fields", Vertical).display = mode == "json"
        self.query_one("#mcp-remote-fields", Vertical).display = mode in ("http", "sse")
        self.query_one("#mcp-local-fields", Vertical).display = mode == "local"
        for name in ("http", "sse", "local", "json"):
            button = self.query_one(f"#mcp-mode-{name}", Button)
            button.set_class(name == mode, "active")
        self.refresh(layout=True)

    @on(Button.Pressed, ".mcp-mode")
    def _select_mode(self, event: Button.Pressed) -> None:
        self.set_entry_mode(event.button.id.removeprefix("mcp-mode-"))
        self.focus_entry_field()

    @on(Button.Pressed, "#mcp-details-toggle")
    def _toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self.query_one("#mcp-details", Vertical).display = self._details_expanded
        self.query_one("#mcp-details-toggle", Button).label = (
            "CONNECTION DETAILS  ▴" if self._details_expanded
            else "CONNECTION DETAILS  ▾"
        )
        if self._details_expanded:
            self.refresh_status()

    def focus_entry_field(self) -> None:
        target = (
            "#mcp-paste" if self._entry_mode == "json"
            else "#mcp-command" if self._entry_mode == "local"
            else "#mcp-url"
        )
        widget = self.query_one(target)
        widget.focus()
        widget.scroll_visible(animate=False)

    def connection_values(self) -> dict[str, str]:
        """Use only the current entry mode to avoid stale hidden inputs."""
        return {
            "name": self.query_one("#mcp-name", Input).value.strip(),
            "transport": {
                "http": "streamable-http",
                "sse": "sse",
                "local": "stdio",
                "json": "streamable-http",
            }[self._entry_mode],
            "url": (
                self.query_one("#mcp-url", Input).value.strip()
                if self._entry_mode in ("http", "sse") else ""
            ),
            "api_key": (
                self.query_one("#mcp-token", Input).value.strip()
                if self._entry_mode in ("http", "sse") else ""
            ),
            "command": (
                self.query_one("#mcp-command", Input).value.strip()
                if self._entry_mode == "local" else ""
            ),
            "args": (
                self.query_one("#mcp-args", Input).value
                if self._entry_mode == "local" else ""
            ),
            "json": (
                self.query_one("#mcp-paste", TextArea).text
                if self._entry_mode == "json" else ""
            ),
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

    def _management_entries(self, client: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in list_mcp_servers():
            name = str(item.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            entries.append({
                "name": name,
                "enabled": bool(item.get("enabled", True)),
            })
        configurations = list(getattr(client, "servers_cfg", ()) or ())
        for cfg in configurations:
            name = str(getattr(cfg, "name", "") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            entries.append({"name": name, "enabled": True})
        return entries

    def _server_state(self, client: Any, name: str, enabled: bool) -> str:
        if not enabled:
            return "disabled"
        if client is None:
            return "not initialized"
        if not bool(getattr(client, "enabled", False)):
            return "MCP disabled"
        transport = (getattr(client, "transports", {}) or {}).get(name)
        state = getattr(getattr(transport, "state", None), "value", None)
        return str(state or "disconnected")

    def selected_server_name(self) -> str | None:
        try:
            option = self.query_one(
                "#mcp-server-list", OptionList
            ).highlighted_option
            return str(option.id) if option is not None and option.id else None
        except Exception:
            return None

    def set_management_status(self, message: str) -> None:
        self.query_one("#mcp-server-status", Static).update(message)

    def _refresh_management_buttons(self, client: Any) -> None:
        name = self.selected_server_name()
        busy = bool(getattr(self._host_app, "busy", False))
        client_ready = (
            client is not None and bool(getattr(client, "enabled", False))
        )
        transports = getattr(client, "transports", {}) or {}
        enabled = True
        if name:
            for entry in self._management_entries(client):
                if entry["name"] == name:
                    enabled = bool(entry["enabled"])
                    break
        state = self._server_state(client, name, enabled) if name else ""

        connect = self.query_one("#mcp-server-connect", Button)
        disconnect = self.query_one("#mcp-server-disconnect", Button)
        delete = self.query_one("#mcp-server-delete", Button)
        connect.disabled = (
            busy or not client_ready or not name or not enabled
            or state in ("connected", "ready", "connecting", "reconnecting")
        )
        disconnect.disabled = (
            busy or not client_ready or not name or not enabled
            or name not in transports
        )
        delete.disabled = busy or not name

    def _refresh_server_management(self, client: Any) -> None:
        option_list = self.query_one("#mcp-server-list", OptionList)
        entries = self._management_entries(client)
        signature = tuple(
            (entry["name"], self._server_state(
                client, entry["name"], bool(entry["enabled"])
            ))
            for entry in entries
        )
        if signature != self._server_list_signature:
            selected = self.selected_server_name()
            option_list.clear_options()
            if entries:
                options = []
                for entry in entries:
                    state = self._server_state(
                        client, entry["name"], bool(entry["enabled"])
                    )
                    options.append(Option(
                        f"• {entry['name']}  [{state}]",
                        id=entry["name"],
                    ))
                option_list.add_options(options)
                names = [entry["name"] for entry in entries]
                if selected in names:
                    option_list.highlighted = names.index(selected)
                elif option_list.highlighted is None:
                    option_list.highlighted = 0
            self._server_list_signature = signature
        self._refresh_management_buttons(client)

    @on(OptionList.OptionHighlighted, "#mcp-server-list")
    def _server_highlighted(self) -> None:
        self._refresh_management_buttons(self.get_client())

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
        self._refresh_server_management(client)

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
