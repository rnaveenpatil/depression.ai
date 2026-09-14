"""
Depression.AI TUI - Futuristic terminal workspace.

The shell intentionally feels like a compact web application rendered inside
the terminal: a large animated hero, persistent workspace navigation, clean
chat canvas, command palette, live model state, and keyboard-first controls.
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
from agent.tui.widgets.llm_panel import LLMProviderPanel


class CommandPalette(ModalScreen[str]):
    """Command launcher styled like a lightweight web-app command menu."""

    DEFAULT_CSS = """
    CommandPalette { align: center middle; background: #070711 80%; }
    #palette-container {
        width: 64; max-width: 86%; height: auto; max-height: 24;
        background: #11111d; border: round #8f7cff; padding: 0;
    }
    #palette-input {
        height: 3; background: #141422; border-bottom: solid #2c2b4a; padding: 0 1;
    }
    #palette-list { height: auto; max-height: 18; overflow-y: auto; }
    .palette-item { height: 2; padding: 0 2; color: #c9c6df; }
    .palette-item:hover { background: #20203a; color: #54d9ff; }
    .palette-item .shortcut { color: #5d5a77; }
    """

    COMMANDS = [
        ("/help", "Show help", "?"),
        ("/model", "Switch model", "m"),
        ("/llm", "Open model workspace", "l"),
        ("/plan", "Switch to Plan mode", "1"),
        ("/build", "Switch to Build mode", "2"),
        ("/auto", "Switch to Auto mode", "3"),
        ("/session", "Manage sessions", "s"),
        ("/sessions", "List all sessions", ""),
        ("/clear", "Clear chat", "Ctrl+L"),
        ("/tools", "Inspect tools", "t"),
        ("/config", "Show configuration", ""),
        ("/status", "Show live status", ""),
        ("/context", "Inspect context", ""),
        ("/compact", "Compact context", ""),
        ("/export", "Export session", ""),
        ("/quit", "Exit", "Ctrl+D"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        with Container(id="palette-container"):
            yield Input(placeholder="Search commands…", id="palette-input")
            with ListView(id="palette-list"):
                for command, description, shortcut in self.COMMANDS:
                    yield ListItem(
                        Label(f"{command}  —  {description}"),
                        Label(shortcut, classes="shortcut"),
                        classes="palette-item",
                    )

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()

    @on(Input.Submitted, "#palette-input")
    def _submit(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def on_key(self, event) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    """Keyboard-first help overlay."""

    DEFAULT_CSS = """
    HelpScreen { align: center middle; background: #070711 80%; }
    #help-container { width: 72; max-width: 88%; height: auto; max-height: 30; background: #11111d; border: round #8f7cff; padding: 1 2; }
    .help-title { color: #ffffff; text-style: bold; margin: 0 0 1 0; }
    .help-section { color: #54d9ff; text-style: bold; margin: 1 0 0 0; }
    .help-row { height: 1; color: #c9c6df; }
    .help-key { color: #ffc857; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-container"):
            yield Static("◈  DEPRESSION.AI  /  KEYBOARD MAP", classes="help-title")
            yield Static("── Workspace", classes="help-section")
            yield Static(" [help-key]Tab[/]          Cycle Plan → Build → Auto", classes="help-row")
            yield Static(" [help-key]Ctrl+P[/]        Open command palette", classes="help-row")
            yield Static(" [help-key]F1[/]            Open this help", classes="help-row")
            yield Static(" [help-key]F2[/]            Toggle workspace sidebar", classes="help-row")
            yield Static(" [help-key]Ctrl+L[/]        Clear chat", classes="help-row")
            yield Static(" [help-key]Ctrl+C[/]        Cancel active work", classes="help-row")
            yield Static(" [help-key]Ctrl+D[/]        Exit", classes="help-row")
            yield Static("── Input", classes="help-section")
            yield Static(" [help-key]Enter[/]         Send message", classes="help-row")
            yield Static(" [help-key]Alt+Enter[/]     New line", classes="help-row")
            yield Static(" [help-key]Tab[/]           Autocomplete", classes="help-row")
            yield Static("── Commands", classes="help-section")
            yield Static(" [help-key]/plan[/]          Planning agent", classes="help-row")
            yield Static(" [help-key]/build[/]         Execution agent", classes="help-row")
            yield Static(" [help-key]/auto[/]          Automatic mode", classes="help-row")
            yield Static(" [help-key]/llm[/]           Model workspace", classes="help-row")
            yield Static(" [help-key]/session[/]       Session management", classes="help-row")
            yield Static(" [help-key]/tools[/]         Tool inventory", classes="help-row")
            yield Static(" [help-key]/quit[/]          Exit", classes="help-row")
            yield Static("── Press any key to close", classes="help-section")

    def on_key(self, event) -> None:
        self.dismiss(None)


class DepressionTUI(App):
    """Depression.AI — futuristic web-style terminal workspace."""

    TITLE = "DEPRESSION.AI"
    SUB_TITLE = "Agentic Workspace"
    CSS = OPENCODE_CSS

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Command Palette", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+d", "quit", "Exit", show=True),
        Binding("ctrl+c", "cancel", "Cancel", show=True),
        Binding("tab", "switch_mode", "Switch Mode", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("f2", "toggle_sidebar", "Sidebar", show=True),
        Binding("escape", "escape", "Back", show=False),
    ]

    current_mode = reactive[str]("build")
    agent_status = reactive[str]("idle")
    model_name = reactive[str]("—")
    tokens_used = reactive[int](0)
    cost = reactive[float](0.0)
    session_id = reactive[str]("—")

    def __init__(self, agent_coordinator=None, config: dict = None, project_dir: str = None,
                 model_override: str = None, provider_override: str = None,
                 yolo: bool = False, no_sidebar: bool = False, session_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.coordinator = agent_coordinator
        self.config = config or {}
        self.project_dir = project_dir
        self.model_override = model_override
        self.provider_override = provider_override
        self.yolo = yolo
        self.no_sidebar = no_sidebar
        self.session_id_override = session_id
        self.sidebar_visible = not no_sidebar
        self._init_task = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Widget(id="main-container"):
            yield Sidebar(id="sidebar", on_llm_connect=self._on_llm_connect)
            yield ChatView(id="chat-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        apply_opencode_theme(self)
        self.title = " ◈ DEPRESSION.AI "
        self.sub_title = f" {self.current_mode.upper()}  /  WORKSPACE "
        self._load_sidebar_data()

        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.display = self.sidebar_visible
        except Exception:
            pass

        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system(
            "**Welcome to Depression.AI.**\n\n"
            "A futuristic agentic workspace for local development, cloud control, "
            "planning, execution and verification.\n\n"
            "Press **Ctrl+P** for commands · **Tab** to change mode · **F1** for help."
        )
        chat.add_divider()

        status = self.query_one("#status-bar", StatusBar)
        status.update_all(status="idle", model=self.model_name, tokens=self.tokens_used, cost=self.cost)

        self._init_task = self.run_worker(
            self._init_agent_async(), exclusive=True, group="agent-init", thread=True
        )
        try:
            self.query_one("#chat-input", Input).focus()
        except Exception:
            pass

    def _load_sidebar_data(self) -> None:
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.set_sessions([{"id": "current", "name": "Current Workspace", "time": "now", "active": True}])
        sidebar.set_tools([
            {"name": "terminal", "enabled": True}, {"name": "filesystem", "enabled": True},
            {"name": "git", "enabled": True}, {"name": "search", "enabled": True},
            {"name": "web", "enabled": True}, {"name": "patch", "enabled": True},
            {"name": "browser", "enabled": True}, {"name": "task", "enabled": True},
            {"name": "diagnostics", "enabled": True},
        ])
        llm_config = get_llm_config()
        selected = llm_config.get_selected_model_info()
        if selected:
            self.model_name = f"{selected.provider}/{selected.short_name}"
            self.query_one("#header", HeaderBar).set_model(self.model_name)
            self.query_one("#status-bar", StatusBar).set_model(self.model_name)

    def _on_llm_connect(self, model: LLMModel) -> None:
        chat = self.query_one("#chat-panel", ChatView)
        status = self.query_one("#status-bar", StatusBar)
        self.model_name = f"{model.provider}/{model.short_name}"
        self.query_one("#header", HeaderBar).set_model(self.model_name)
        status.set_model(self.model_name)
        llm_config = get_llm_config()
        llm_config.selected_model = model.name
        chat.add_system(
            f"Connected to **{model.display_name}** ({model.provider})\n"
            f"Base URL: `{model.base_url}`\nContext: {model.context_window // 1000}K tokens"
        )
        if self.coordinator:
            self._apply_llm_to_coordinator(model)

    def _apply_llm_to_coordinator(self, model: LLMModel) -> None:
        try:
            llm_config = get_llm_config()
            api_key = llm_config.get_api_key(model.provider)
            base_url = llm_config.get_base_url(model.provider)
            for agent in [self.coordinator.plan_agent, self.coordinator.build_agent]:
                if hasattr(agent, "llm") and agent.llm:
                    if hasattr(agent.llm, "api_key"): agent.llm.api_key = api_key
                    if hasattr(agent.llm, "base_url"): agent.llm.base_url = base_url
                    if hasattr(agent.llm, "model"): agent.llm.model = model.name
            self.query_one("#chat-panel", ChatView).add_system(f"Agent updated to use {model.display_name}")
        except Exception as e:
            self.query_one("#chat-panel", ChatView).add_error(f"Failed to update agent: {e}")

    async def action_command_palette(self) -> None:
        result = await self.push_screen_wait(CommandPalette())
        if result:
            await self._execute_command(result)

    async def action_show_help(self) -> None:
        await self.push_screen_wait(HelpScreen())

    def action_clear_screen(self) -> None:
        self.query_one("#chat-panel", ChatView).clear_messages()

    def action_cancel(self) -> None:
        if self.agent_status in ("thinking", "acting"):
            self.agent_status = "idle"
            status = self.query_one("#status-bar", StatusBar)
            status.set_status("idle")
            status.set_tool("")

    def action_switch_mode(self) -> None:
        modes = ["plan", "build", "auto"]
        current_idx = modes.index(self.current_mode) if self.current_mode in modes else 1
        self.current_mode = modes[(current_idx + 1) % len(modes)]
        if self.coordinator:
            self.coordinator.set_mode(self.current_mode)
        self.query_one("#header", HeaderBar).set_mode(self.current_mode)
        self.sub_title = f" {self.current_mode.upper()}  /  WORKSPACE "

    def action_toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible
        self.query_one("#sidebar", Sidebar).display = self.sidebar_visible

    def action_escape(self) -> None:
        try:
            self.pop_screen()
        except Exception:
            pass

    @on(Input.Submitted, "#chat-input")
    async def _on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_message("user", text)
        self.agent_status = "thinking"
        chat.add_thinking()
        status = self.query_one("#status-bar", StatusBar)
        status.set_status("thinking")
        await self._run_agent(text)

    async def _run_agent(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatView)
        try:
            if self.coordinator is None:
                chat.remove_thinking()
                chat.add_error("Agent coordinator is not connected.")
                return
            result = await self._dispatch_to_coordinator(text)
            chat.remove_thinking()
            if isinstance(result, str) and result.strip():
                chat.add_message("agent", result)
            elif result is not None:
                chat.add_message("agent", str(result))
        except asyncio.CancelledError:
            chat.remove_thinking()
            chat.add_error("Operation cancelled.")
        except Exception as e:
            chat.remove_thinking()
            chat.add_error(f"Agent error: {e}")
        finally:
            self.agent_status = "idle"
            self.query_one("#status-bar", StatusBar).set_status("idle")

    async def _dispatch_to_coordinator(self, text: str) -> Any:
        """Use the coordinator's public async entry point when available."""
        if hasattr(self.coordinator, "run"):
            result = self.coordinator.run(text)
            if hasattr(result, "__await__"):
                return await result
            return result
        if hasattr(self.coordinator, "execute"):
            result = self.coordinator.execute(text)
            if hasattr(result, "__await__"):
                return await result
            return result
        raise RuntimeError("Coordinator has no run/execute entry point")

    async def _execute_command(self, command: str) -> None:
        if not command.startswith("/"):
            await self._run_agent(command)
            return
        parts = command.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if name == "/help": await self.action_show_help()
        elif name == "/clear": self.action_clear_screen()
        elif name == "/plan": self.current_mode = "plan"; self.action_switch_mode()
        elif name == "/build": self.current_mode = "build"; self.action_switch_mode()
        elif name == "/auto": self.current_mode = "auto"; self.action_switch_mode()
        elif name == "/llm": self.query_one("#sidebar", Sidebar).switch_to_llm()
        elif name == "/quit": self.exit()
        elif name == "/status": self.query_one("#status-bar", StatusBar).refresh()
        elif name == "/tools": self.query_one("#sidebar", Sidebar).active_tab = "tools"
        elif name == "/sessions": self.query_one("#sidebar", Sidebar).active_tab = "sessions"
        elif name == "/model": self.query_one("#sidebar", Sidebar).switch_to_llm()
        elif name == "/session": self.query_one("#sidebar", Sidebar).active_tab = "sessions"
        elif name in ("/context", "/compact", "/config", "/export"):
            await self._run_agent(command)
        else:
            await self._run_agent(command)

    def on_unmount(self) -> None:
        if self._init_task:
            try: self._init_task.cancel()
            except Exception: pass
