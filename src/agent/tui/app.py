"""
Depression.AI TUI - Ultimate Terminal User Interface

A next-generation futuristic TUI featuring:
- Large animated Depression.AI hero branding
- Futuristic glass/neon workspace styling
- Real-time streaming with markdown rendering
- Interactive tool execution visualization
- Session management with persistence
- Command palette (Ctrl+P)
- LLM provider selection panel
- Live metrics and status
- Keyboard-driven navigation
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import (
    Static,
    Input,
    ListView,
    ListItem,
    Label,
    Button,
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual import on, work
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget

from agent.tui.theme import OPENCODE_CSS, apply_opencode_theme
from agent.tui.views.header import HeaderBar
from agent.tui.views.status_bar import StatusBar
from agent.tui.views.sidebar import Sidebar
from agent.tui.views.chat import ChatView
from agent.tui.llm_providers import LLMModel, get_llm_config, PROVIDER_MODELS
from agent.tui.widgets.llm_panel import LLMProviderPanel


class CommandPalette(ModalScreen[str]):
    """Command palette for fast keyboard-first navigation."""
    DEFAULT_CSS = """
    CommandPalette { align: center middle; }
    #palette-container { width: 60; max-width: 80; height: auto; max-height: 24; background: $bg-panel; border: solid $border-focused; padding: 0; }
    #palette-input { height: 3; background: $input-bg; border-bottom: solid $border-focused; padding: 0 1; }
    #palette-list { height: auto; max-height: 18; overflow-y: auto; }
    .palette-item { height: 1; padding: 0 1; color: $text; }
    .palette-item:hover { background: $bg-hover; color: $secondary; }
    .palette-item .shortcut { color: $text-dim; float: right; }
    """
    COMMANDS = [
        ("/help", "Show help", "?"), ("/model", "Switch model", "m"),
        ("/llm", "LLM provider panel", "l"), ("/plan", "Switch to Plan mode", "1"),
        ("/build", "Switch to Build mode", "2"), ("/auto", "Switch to Auto mode", "3"),
        ("/session", "Manage sessions", "s"), ("/sessions", "List all sessions", ""),
        ("/clear", "Clear chat", "Ctrl+L"), ("/tools", "List tools", "t"),
        ("/config", "Show config", ""), ("/status", "Show status", ""),
        ("/context", "Show context", ""), ("/compact", "Compact context", ""),
        ("/export", "Export session", ""), ("/quit", "Exit", "Ctrl+D"),
    ]
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.filtered_commands = list(self.COMMANDS); self.selected_index = 0
    def compose(self) -> ComposeResult:
        with Widget(id="palette-container"):
            yield Input(placeholder=" Type a command...", id="palette-input")
            with ListView(id="palette-list"):
                for cmd, desc, shortcut in self.filtered_commands:
                    yield ListItem(Label(f"{cmd}  {desc}"), Label(f"[shortcut]{shortcut}[/]" if shortcut else "", classes="shortcut"))
    @on(Input.Changed, "#palette-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        self.filtered_commands = [(cmd, desc, sc) for cmd, desc, sc in self.COMMANDS if query in cmd.lower() or query in desc.lower()]
        try:
            view = self.query_one("#palette-list", ListView); view.clear()
            for cmd, desc, shortcut in self.filtered_commands:
                view.append(ListItem(Label(f"{cmd}  {desc}"), Label(f"[shortcut]{shortcut}[/]" if shortcut else "", classes="shortcut")))
        except Exception: pass
    @on(ListView.Selected, "#palette-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        if 0 <= event.index < len(self.filtered_commands): self.dismiss(self.filtered_commands[event.index][0])
    @on(Input.Submitted, "#palette-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        if self.filtered_commands: self.dismiss(self.filtered_commands[0][0])
    def on_key(self, event) -> None:
        if event.key == "escape": self.dismiss(None)
        elif event.key == "up":
            self.selected_index = max(0, self.selected_index - 1)
        elif event.key == "down":
            self.selected_index = min(max(0, len(self.filtered_commands) - 1), self.selected_index + 1)


class HelpScreen(ModalScreen[str]):
    """Help screen."""
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    #help-container { width: 70; height: auto; max-height: 30; background: $bg-panel; border: solid $border-focused; padding: 1 2; }
    .help-title { text-style: bold; color: $primary; height: 1; margin: 0 0 1 0; }
    .help-section { color: $secondary; text-style: bold; height: 1; margin: 1 0 0 0; }
    .help-row { height: 1; color: $text; }
    .help-key { color: $accent; text-style: bold; }
    """
    def compose(self) -> ComposeResult:
        with Widget(id="help-container"):
            yield Static(" ◆ DEPRESSION.AI — KEYBOARD SHORTCUTS ", classes="help-title")
            yield Static(" ─── Navigation ─── ", classes="help-section")
            yield Static(" [help-key]Tab[/]          Switch Plan/Build mode", classes="help-row")
            yield Static(" [help-key]Ctrl+P[/]        Open command palette", classes="help-row")
            yield Static(" [help-key]Ctrl+L[/]        Clear screen", classes="help-row")
            yield Static(" [help-key]Ctrl+C[/]        Cancel / clear input", classes="help-row")
            yield Static(" [help-key]Ctrl+D[/]        Exit", classes="help-row")
            yield Static(" ─── Input ─── ", classes="help-section")
            yield Static(" [help-key]Enter[/]         Send message", classes="help-row")
            yield Static(" [help-key]Alt+Enter[/]     New line in input", classes="help-row")
            yield Static(" [help-key]Shift+Enter[/]   New line in input", classes="help-row")
            yield Static(" [help-key]Tab[/]           Autocomplete", classes="help-row")
            yield Static(" [help-key]Ctrl+Space[/]    Force completion", classes="help-row")
            yield Static(" ─── Commands ─── ", classes="help-section")
            yield Static(" [help-key]/help[/]          Show this help", classes="help-row")
            yield Static(" [help-key]/llm[/]           Open LLM provider panel", classes="help-row")
            yield Static(" [help-key]/plan[/]          Switch to Plan mode", classes="help-row")
            yield Static(" [help-key]/build[/]         Switch to Build mode", classes="help-row")
            yield Static(" [help-key]/auto[/]          Switch to Auto mode", classes="help-row")
            yield Static(" [help-key]/model[/]         Switch model", classes="help-row")
            yield Static(" [help-key]/session[/]       Manage sessions", classes="help-row")
            yield Static(" [help-key]/clear[/]         Clear chat", classes="help-row")
            yield Static(" [help-key]/tools[/]         List tools", classes="help-row")
            yield Static(" [help-key]/quit[/]          Exit", classes="help-row")
            yield Static(" ─── Press any key to close ─── ", classes="help-section")
    def on_key(self, event) -> None: self.dismiss(None)


class DepressionTUI(App):
    """Depression.AI — futuristic web-style terminal workspace."""
    TITLE = "DEPRESSION.AI"
    SUB_TITLE = "Agentic Workspace"
    CSS = OPENCODE_CSS
    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Command Palette", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True), Binding("ctrl+d", "quit", "Exit", show=True),
        Binding("ctrl+c", "cancel", "Cancel", show=True), Binding("tab", "switch_mode", "Switch Mode", show=True),
        Binding("f1", "show_help", "Help", show=True), Binding("f2", "toggle_sidebar", "Sidebar", show=True),
        Binding("escape", "escape", "Back", show=False),
    ]
    current_mode = reactive[str]("build"); agent_status = reactive[str]("idle")
    model_name = reactive[str]("—"); tokens_used = reactive[int](0); cost = reactive[float](0.0); session_id = reactive[str]("—")

    def __init__(self, agent_coordinator=None, config: dict = None, project_dir: str = None, model_override: str = None,
                 provider_override: str = None, yolo: bool = False, no_sidebar: bool = False, session_id: str = None, **kwargs):
        super().__init__(**kwargs); self.coordinator = agent_coordinator; self.config = config or {}; self.project_dir = project_dir
        self.model_override = model_override; self.provider_override = provider_override; self.yolo = yolo; self.no_sidebar = no_sidebar
        self.session_id_override = session_id; self.sidebar_visible = not no_sidebar; self._init_task = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Widget(id="main-container"):
            yield Sidebar(id="sidebar", on_llm_connect=self._on_llm_connect)
            yield ChatView(id="chat-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        apply_opencode_theme(self); self.title = " ◈ DEPRESSION.AI "; self.sub_title = f" {self.current_mode.upper()}  /  WORKSPACE "
        self._load_sidebar_data()
        try: self.query_one("#sidebar", Sidebar).display = self.sidebar_visible
        except Exception: pass
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system("**Welcome to Depression.AI.**\n\nA futuristic agentic workspace for local development, cloud control, planning, execution and verification.\n\nPress **Ctrl+P** for commands · **Tab** to change mode · **F1** for help.")
        chat.add_divider()
        self.query_one("#status-bar", StatusBar).update_all(status="idle", model=self.model_name, tokens=self.tokens_used, cost=self.cost)
        self._init_task = self.run_worker(self._init_agent_async(), exclusive=True, group="agent-init", thread=True)
        try: self.query_one("#chat-input", Input).focus()
        except Exception: pass

    def _load_sidebar_data(self) -> None:
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.set_sessions([{"id": "current", "name": "Current Workspace", "time": "now", "active": True}])
        sidebar.set_tools([{"name": n, "enabled": True} for n in ["terminal", "filesystem", "git", "search", "web", "patch", "browser", "task", "diagnostics"]])
        selected = get_llm_config().get_selected_model_info()
        if selected:
            self.model_name = f"{selected.provider}/{selected.short_name}"; self.query_one("#header", HeaderBar).set_model(self.model_name); self.query_one("#status-bar", StatusBar).set_model(self.model_name)

    def _on_llm_connect(self, model: LLMModel) -> None:
        chat = self.query_one("#chat-panel", ChatView); status = self.query_one("#status-bar", StatusBar)
        self.model_name = f"{model.provider}/{model.short_name}"; self.query_one("#header", HeaderBar).set_model(self.model_name); status.set_model(self.model_name)
        get_llm_config().selected_model = model.name
        chat.add_system(f"Connected to **{model.display_name}** ({model.provider})\nBase URL: `{model.base_url}`\nContext: {model.context_window // 1000}K tokens")
        if self.coordinator: self._apply_llm_to_coordinator(model)

    def _apply_llm_to_coordinator(self, model: LLMModel) -> None:
        try:
            llm_config = get_llm_config(); api_key = llm_config.get_api_key(model.provider); base_url = llm_config.get_base_url(model.provider)
            for agent in [self.coordinator.plan_agent, self.coordinator.build_agent]:
                if hasattr(agent, "llm") and agent.llm:
                    if hasattr(agent.llm, "api_key"): agent.llm.api_key = api_key
                    if hasattr(agent.llm, "base_url"): agent.llm.base_url = base_url
                    if hasattr(agent.llm, "model"): agent.llm.model = model.name
            self.query_one("#chat-panel", ChatView).add_system(f"Agent updated to use {model.display_name}")
        except Exception as e: self.query_one("#chat-panel", ChatView).add_error(f"Failed to update agent: {e}")

    async def action_command_palette(self) -> None:
        result = await self.push_screen_wait(CommandPalette());
        if result: await self._execute_command(result)
    async def action_show_help(self) -> None: await self.push_screen_wait(HelpScreen())
    def action_clear_screen(self) -> None: self.query_one("#chat-panel", ChatView).clear_messages()
    def action_cancel(self) -> None:
        if self.agent_status in ("thinking", "acting"):
            self.agent_status = "idle"; status = self.query_one("#status-bar", StatusBar); status.set_status("idle"); status.set_tool("")
    def action_switch_mode(self) -> None:
        modes = ["plan", "build", "auto"]; self.current_mode = modes[(modes.index(self.current_mode) + 1) % len(modes)] if self.current_mode in modes else "build"
        if self.coordinator: self.coordinator.set_mode(self.current_mode)
        self.query_one("#header", HeaderBar).set_mode(self.current_mode); self.sub_title = f" {self.current_mode.upper()}  /  WORKSPACE "; self.query_one("#status-bar", StatusBar).set_status("idle")
        self.query_one("#chat-panel", ChatView).add_system(f"Switched to {self.current_mode.upper()} mode")
    def action_toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible; self.query_one("#sidebar", Sidebar).display = self.sidebar_visible
    def action_escape(self) -> None: self.query_one("#chat-panel", ChatView).clear_input()

    async def _execute_command(self, command: str) -> None:
        chat = self.query_one("#chat-panel", ChatView)
        if command == "/quit": self.exit(); return
        if command == "/clear": chat.clear_messages(); return
        if command == "/help": await self.action_show_help(); return
        if command in ("/plan", "/build", "/auto"):
            self.current_mode = command[1:]; self.query_one("#header", HeaderBar).set_mode(self.current_mode); chat.add_system(f"Switched to {self.current_mode.upper()} mode"); return
        if command in ("/model", "/llm"): self.query_one("#sidebar", Sidebar).switch_to_llm(); chat.add_system("Opening LLM provider panel..."); return
        if command == "/status": chat.add_system(f"Mode: {self.current_mode}\nModel: {self.model_name}\nTokens: {self.tokens_used:,}\nCost: ${self.cost:.4f}\nSession: {self.session_id[:8]}"); return
        if command == "/tools": self.query_one("#sidebar", Sidebar).active_tab = "tools"; chat.add_system("Showing tools in sidebar"); return
        if command.startswith("/session"): chat.add_system("Session commands: /session new, /session list, /session load <id>, /session save"); return
        chat.add_error(f"Unknown command: {command}")

    @on(Input.Changed, "#chat-input")
    def _on_input_changed(self, event: Input.Changed) -> None: pass

    @on(Input.Submitted, "#chat-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None: self._send_message()

    def _send_message(self) -> None:
        chat = self.query_one("#chat-panel", ChatView); text = chat.get_input_text().strip()
        if not text: return
        chat.add_message("user", text); chat.clear_input(); self._process_message(text)

    @work(exclusive=True, group="message-processor")
    async def _process_message(self, text: str) -> None:
        status = self.query_one("#status-bar", StatusBar); chat = self.query_one("#chat-panel", ChatView); self.agent_status = "thinking"; status.set_status("thinking")
        try:
            if self.coordinator:
                chat.add_thinking("Analyzing...")
                result = await asyncio.wait_for(self.coordinator.process_query(text, mode=self.current_mode, auto_execute=True), timeout=120)
                chat.remove_thinking()
                if result.get("success"):
                    response = result.get("execution") or result.get("response", ""); chat.add_message("agent", response)
                    if "tokens" in result: self.tokens_used = result["tokens"]; status.set_tokens(result["tokens"])
                    if "cost" in result: self.cost = result["cost"]; status.set_cost(result["cost"])
                    coord_status = self.coordinator.get_status(); plan_model = coord_status.get("plan_agent", {}).get("model", "—"); build_model = coord_status.get("build_agent", {}).get("model", "—"); self.set_model(f"Plan: {plan_model} | Build: {build_model}")
                else: chat.add_error(result.get("error", "Unknown error"))
            else:
                await asyncio.sleep(0.5); chat.add_message("agent", f"Received: {text}\n\nThis is a demo response. Connect an agent coordinator for full functionality.\n\nCurrent mode: **{self.current_mode}**\nUse /help for available commands.")
        except asyncio.TimeoutError: chat.add_error("Request timed out after 120 seconds.")
        except Exception as e: chat.add_error(f"Error: {str(e)}")
        finally: self.agent_status = "idle"; status.set_status("idle"); status.set_tool("")

    def set_coordinator(self, coordinator) -> None: self.coordinator = coordinator
    def set_model(self, model: str) -> None:
        self.model_name = model; self.query_one("#header", HeaderBar).set_model(model); self.query_one("#status-bar", StatusBar).set_model(model)
    def set_session(self, session_id: str) -> None:
        self.session_id = session_id; self.query_one("#header", HeaderBar).set_session(session_id)
    def update_metrics(self, tokens: int = 0, cost: float = 0.0) -> None:
        self.tokens_used = tokens; self.cost = cost; self.query_one("#status-bar", StatusBar).update_all(tokens=tokens, cost=cost)
    def show_tool_call(self, tool_name: str, params: dict = None, result: Any = None, success: bool = True, duration: float = None) -> None:
        self.query_one("#chat-panel", ChatView).add_tool_call(tool_name, params, result, success, duration)
    def stream_response(self, text: str) -> None: self.query_one("#chat-panel", ChatView).append_stream(text)

    async def action_quit(self) -> None:
        if self.coordinator:
            try: await self.coordinator.shutdown()
            except Exception: pass
        if hasattr(self, 'session_manager') and self.session_manager:
            try: await self.session_manager.save_current_session()
            except Exception: pass
        self.exit()

    async def _init_agent_async(self) -> None:
        try:
            from agent.config.loader import load_config
            from agent.utils.platform import get_data_dir, get_cache_dir
            from agent.storage.database import Database
            from agent.storage.cache import Cache
            from agent.project.workspace import WorkspaceManager
            from agent.session.session import SessionManager
            from agent.context.manager import ContextManager
            from agent.llm.provider import get_llm_registry
            from agent.permissions.manager import PermissionManager
            from agent.agent.dual_agent import create_dual_agent_system
            APP_NAME = "depression"
            config_dict = load_config(cache=True)
            if self.model_override: config_dict.setdefault("llm", {})["model"] = self.model_override
            if self.provider_override: config_dict.setdefault("llm", {})["provider"] = self.provider_override
            if self.yolo: config_dict.setdefault("permissions", {})["auto_approve"] = True
            database = Database(str(get_data_dir(APP_NAME) / "agent.db")); await database.initialize()
            cache = Cache(cache_dir=str(get_cache_dir(APP_NAME)))
            project_dir = self.project_dir or config_dict.get("workspace", {}).get("path", ".")
            workspace = WorkspaceManager(workspace_dir=str(get_data_dir(APP_NAME)), project_dir=project_dir); await workspace.initialize()
            session_manager = SessionManager(database=database)
            session = await session_manager.load_session(self.session_id_override) if self.session_id_override else await session_manager.get_or_create_session()
            llm_registry = get_llm_registry(); context_manager = ContextManager(workspace=workspace, session=session, config=config_dict.get("context", {}), llm=llm_registry); await context_manager.initialize()
            permission_manager = PermissionManager(config=config_dict.get("permissions", {}), ui=None)
            coordinator = await create_dual_agent_system(config=config_dict, session=session, context_manager=context_manager, permission_manager=permission_manager, workspace=workspace, database=database, ui=None, cache=cache)
            self.call_from_thread(self._on_agent_ready, coordinator, session, workspace, session_manager)
        except Exception as e: self.call_from_thread(self._on_agent_error, str(e))

    def _on_agent_ready(self, coordinator, session, workspace, session_manager) -> None:
        self.coordinator = coordinator; self.session_manager = session_manager; self.workspace = workspace
        status = coordinator.get_status(); self.set_model(f"Plan: {status.get('plan_agent', {}).get('model', '—')} | Build: {status.get('build_agent', {}).get('model', '—')}"); self.set_session(session.id)
        sidebar = self.query_one("#sidebar", Sidebar)
        try: sidebar.set_tools([{"name": n, "enabled": True} for n in coordinator.plan_agent.tool_registry.list_tools()])
        except Exception: pass
        try:
            files = [{"name": str(f.relative_to(workspace.get_project_dir())), "is_dir": f.is_dir(), "size": self._human_size(f.stat().st_size) if f.is_file() else ""} for f in workspace.list_files()]
            sidebar.set_files(files[:100])
        except Exception: pass
        self._load_sessions_list(); chat = self.query_one("#chat-panel", ChatView); chat.add_system(f"Agent connected! Session: {session.id[:8]}"); chat.add_divider()

    def _on_agent_error(self, error: str) -> None:
        chat = self.query_one("#chat-panel", ChatView); chat.add_error(f"Agent initialization failed: {error}\nRunning in demo mode."); chat.add_divider()
    def _load_sessions_list(self) -> None:
        try:
            if hasattr(self, 'session_manager') and self.session_manager: self.query_one("#sidebar", Sidebar).set_sessions([{"id": "current", "name": "Current Session", "time": "now", "active": True}])
        except Exception: pass
    @staticmethod
    def _human_size(size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024: return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    def action_toggle_llm_panel(self) -> None:
        self.query_one("#sidebar", Sidebar).switch_to_llm(); self.query_one("#chat-panel", ChatView).add_system("Opened LLM provider panel")
