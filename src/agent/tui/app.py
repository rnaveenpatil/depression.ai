"""
Depression.AI TUI - polished terminal workspace.

The visual system intentionally follows the simplicity and density of modern
agent CLIs while keeping a distinctive animated angel identity.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static, Input, ListView, ListItem, Label, Button
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


class CommandPalette(ModalScreen[str]):
    DEFAULT_CSS = """
    CommandPalette { align: center middle; background: #000000 55%; }
    #palette-container { width: 64; max-width: 85%; height: auto; max-height: 24; background: $bg-panel; border: round $border-focused; }
    #palette-input { height: 3; background: $input-bg; border: none; border-bottom: solid $border; padding: 0 1; }
    #palette-list { height: auto; max-height: 18; }
    .palette-item { height: 2; padding: 0 2; color: $text; }
    .palette-item:hover { background: $bg-hover; color: $secondary; }
    """
    COMMANDS = [
        ("/help", "Show help", "?"), ("/model", "Switch model", "m"), ("/llm", "LLM provider panel", "l"),
        ("/plan", "Switch to Plan mode", "1"), ("/build", "Switch to Build mode", "2"), ("/auto", "Switch to Auto mode", "3"),
        ("/session", "Manage sessions", "s"), ("/sessions", "List all sessions", ""), ("/clear", "Clear chat", "Ctrl+L"),
        ("/tools", "List tools", "t"), ("/config", "Show config", ""), ("/status", "Show status", ""),
        ("/context", "Show context", ""), ("/compact", "Compact context", ""), ("/export", "Export session", ""), ("/quit", "Exit", "Ctrl+D"),
    ]
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.filtered_commands = list(self.COMMANDS); self.selected_index = 0
    def compose(self) -> ComposeResult:
        with Widget(id="palette-container"):
            yield Input(placeholder="Command...", id="palette-input")
            with ListView(id="palette-list"):
                for cmd, desc, shortcut in self.filtered_commands:
                    yield ListItem(Label(f"{cmd}  {desc}"), Label(shortcut, classes="shortcut"))
    @on(Input.Changed, "#palette-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower(); self.filtered_commands = [(c,d,s) for c,d,s in self.COMMANDS if query in c.lower() or query in d.lower()]
        view = self.query_one("#palette-list", ListView); view.clear()
        for cmd, desc, shortcut in self.filtered_commands: view.append(ListItem(Label(f"{cmd}  {desc}"), Label(shortcut, classes="shortcut")))
    @on(ListView.Selected, "#palette-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        if 0 <= event.index < len(self.filtered_commands): self.dismiss(self.filtered_commands[event.index][0])
    @on(Input.Submitted, "#palette-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        if self.filtered_commands: self.dismiss(self.filtered_commands[0][0])
    def on_key(self, event) -> None:
        if event.key == "escape": self.dismiss(None)


class HelpScreen(ModalScreen[str]):
    DEFAULT_CSS = """
    HelpScreen { align: center middle; background: #000000 55%; }
    #help-container { width: 70; height: auto; max-height: 30; background: $bg-panel; border: round $border-focused; padding: 1 2; }
    .help-title { text-style: bold; color: $primary; height: 1; }
    .help-section { color: $secondary; text-style: bold; height: 1; margin: 1 0 0 0; }
    .help-row { height: 1; color: $text; }
    .help-key { color: $accent; text-style: bold; }
    """
    def compose(self) -> ComposeResult:
        with Widget(id="help-container"):
            yield Static("DEPRESSION.AI  ·  KEYBOARD", classes="help-title")
            yield Static("Navigation", classes="help-section")
            yield Static("[help-key]Tab[/]  switch mode    [help-key]Ctrl+P[/]  command palette    [help-key]F2[/]  sidebar", classes="help-row")
            yield Static("Input", classes="help-section")
            yield Static("[help-key]Enter[/] send    [help-key]Shift+Enter[/] newline    [help-key]Ctrl+C[/] cancel    [help-key]Ctrl+D[/] exit", classes="help-row")
            yield Static("Commands", classes="help-section")
            yield Static("/llm  /model  /plan  /build  /auto  /session  /tools  /context  /compact  /clear", classes="help-row")
    def on_key(self, event) -> None: self.dismiss(None)


class DepressionTUI(App):
    TITLE = "DEPRESSION.AI"
    SUB_TITLE = "Agentic Workspace"
    CSS = OPENCODE_CSS
    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Command Palette", show=True), Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+d", "quit", "Exit", show=True), Binding("ctrl+c", "cancel", "Cancel", show=True), Binding("tab", "switch_mode", "Switch Mode", show=True),
        Binding("f1", "show_help", "Help", show=True), Binding("f2", "toggle_sidebar", "Sidebar", show=True), Binding("escape", "escape", "Back", show=False),
    ]
    current_mode = reactive[str]("build"); agent_status = reactive[str]("idle"); model_name = reactive[str]("—"); tokens_used = reactive[int](0); cost = reactive[float](0.0); session_id = reactive[str]("—")

    def __init__(self, agent_coordinator=None, config: dict = None, project_dir: str = None, model_override: str = None, provider_override: str = None, yolo: bool = False, no_sidebar: bool = False, session_id: str = None, **kwargs):
        super().__init__(**kwargs); self.coordinator = agent_coordinator; self.config = config or {}; self.project_dir = project_dir
        self.model_override = model_override; self.provider_override = provider_override; self.yolo = yolo; self.no_sidebar = no_sidebar; self.session_id_override = session_id; self.sidebar_visible = not no_sidebar; self._init_task = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Widget(id="main-container"):
            yield ChatView(id="chat-panel")
            yield Sidebar(id="sidebar", on_llm_connect=self._on_llm_connect)
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        apply_opencode_theme(self); self.title = "DEPRESSION.AI"; self.sub_title = self.current_mode.upper()
        self._load_sidebar_data()
        self.query_one("#sidebar", Sidebar).display = self.sidebar_visible
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system("**DEPRESSION.AI**  ·  local + cloud agent\n\nType a task to begin.  **Tab** changes mode · **Ctrl+P** opens commands · **F2** toggles the workspace sidebar.")
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
            chat = self.query_one("#chat-panel", ChatView); chat.add_system(f"Agent updated to use {model.display_name}")
        except Exception as e: self.query_one("#chat-panel", ChatView).add_error(f"Failed to update agent: {e}")

    async def action_command_palette(self) -> None:
        result = await self.push_screen_wait(CommandPalette());
        if result: await self._execute_command(result)
    async def action_show_help(self) -> None: await self.push_screen_wait(HelpScreen())
    def action_clear_screen(self) -> None: self.query_one("#chat-panel", ChatView).clear_messages()
    def action_cancel(self) -> None:
        if self.agent_status in ("thinking", "acting"): self.agent_status = "idle"; self.query_one("#status-bar", StatusBar).set_status("idle"); self.query_one("#status-bar", StatusBar).set_tool("")
    def action_switch_mode(self) -> None:
        modes = ["plan", "build", "auto"]; self.current_mode = modes[(modes.index(self.current_mode) + 1) % len(modes)] if self.current_mode in modes else "build"
        if self.coordinator: self.coordinator.set_mode(self.current_mode)
        self.query_one("#header", HeaderBar).set_mode(self.current_mode); self.sub_title = self.current_mode.upper(); self.query_one("#status-bar", StatusBar).set_status("idle"); self.query_one("#chat-panel", ChatView).add_system(f"Switched to {self.current_mode.upper()} mode")
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
        if command in ("/model", "/llm"): self.query_one("#sidebar", Sidebar).switch_to_llm(); chat.add_system("LLM panel opened in the right sidebar."); return
        if command == "/status": chat.add_system(f"Mode: {self.current_mode}\nModel: {self.model_name}\nTokens: {self.tokens_used:,}\nCost: ${self.cost:.4f}\nSession: {self.session_id[:8]}"); return
        if command == "/tools": self.query_one("#sidebar", Sidebar).active_tab = "tools"; chat.add_system("Tools shown in the right sidebar."); return
        if command.startswith("/session"): chat.add_system("Session management is available from the sidebar."); return
        chat.add_system(f"Unknown command: {command}")

    async def _init_agent_async(self):
        if self.coordinator is None: return
        try:
            self.agent_status = "thinking"
            if self.model_override or self.provider_override:
                cfg = get_llm_config(); selected = cfg.get_selected_model_info()
                if selected: self._apply_llm_to_coordinator(selected)
            self.agent_status = "idle"
        except Exception as e:
            self.agent_status = "error"; self.query_one("#chat-panel", ChatView).add_error(str(e))
