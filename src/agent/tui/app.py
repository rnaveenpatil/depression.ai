"""
Depression.AI TUI - Ultimate Terminal User Interface

A next-generation cyberpunk TUI featuring:
- Multi-theme support (cyberpunk, tron, matrix, synthwave)
- Animated gradient headers and scanline effects
- Real-time streaming with markdown rendering
- Interactive tool execution visualization
- Session management with persistence
- Command palette (Ctrl+P)
- LLM provider selection panel
- Live metrics and status
- Particle/background effects
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


# ======================================================================
# COMMAND PALETTE SCREEN
# ======================================================================

class CommandPalette(ModalScreen[str]):
    """A command palette modal for quick access to commands."""

    DEFAULT_CSS = """
    CommandPalette {
        align: center middle;
    }

    #palette-container {
        width: 60;
        max-width: 80;
        height: auto;
        max-height: 24;
        background: $bg-panel;
        border: solid $border-focused;
        padding: 0;
    }

    #palette-input {
        height: 3;
        background: $bg-input;
        border-bottom: solid $border-focused;
        padding: 0 1;
    }

    #palette-list {
        height: auto;
        max-height: 18;
        overflow-y: auto;
    }

    .palette-item {
        height: 1;
        padding: 0 1;
        color: $text-secondary;
    }

    .palette-item:hover {
        background: $bg-hover;
        color: $accent-primary;
    }

    .palette-item .shortcut {
        color: $text-dim;
        float: right;
    }
    """

    COMMANDS = [
        ("/help", "Show help", "?"),
        ("/model", "Switch model", "m"),
        ("/llm", "LLM provider panel", "l"),
        ("/plan", "Switch to Plan mode", "1"),
        ("/build", "Switch to Build mode", "2"),
        ("/auto", "Switch to Auto mode", "3"),
        ("/session", "Manage sessions", "s"),
        ("/sessions", "List all sessions", ""),
        ("/clear", "Clear chat", "Ctrl+L"),
        ("/tools", "List tools", "t"),
        ("/config", "Show config", ""),
        ("/status", "Show status", ""),
        ("/context", "Show context", ""),
        ("/compact", "Compact context", ""),
        ("/export", "Export session", ""),
        ("/quit", "Exit", "Ctrl+D"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filtered_commands = list(self.COMMANDS)
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        with Widget(id="palette-container"):
            yield Input(
                placeholder=" Type a command...",
                id="palette-input",
            )
            with ListView(id="palette-list"):
                for cmd, desc, shortcut in self.filtered_commands:
                    yield ListItem(
                        Label(f"{cmd}  {desc}"),
                        Label(f"[shortcut]{shortcut}[/]" if shortcut else "", classes="shortcut"),
                    )

    @on(Input.Changed, "#palette-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        """Filter commands based on input."""
        query = event.value.lower()
        self.filtered_commands = [
            (cmd, desc, sc)
            for cmd, desc, sc in self.COMMANDS
            if query in cmd.lower() or query in desc.lower()
        ]
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the command list."""
        try:
            list_view = self.query_one("#palette-list", ListView)
            list_view.clear()
            for cmd, desc, shortcut in self.filtered_commands:
                list_view.append(
                    ListItem(
                        Label(f"{cmd}  {desc}"),
                        Label(f"[shortcut]{shortcut}[/]" if shortcut else "", classes="shortcut"),
                    )
                )
        except Exception:
            pass

    @on(ListView.Selected, "#palette-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        """Handle command selection."""
        if 0 <= event.index < len(self.filtered_commands):
            cmd = self.filtered_commands[event.index][0]
            self.dismiss(cmd)

    @on(Input.Submitted, "#palette-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in the input."""
        if self.filtered_commands:
            cmd = self.filtered_commands[0][0]
            self.dismiss(cmd)

    def on_key(self, event) -> None:
        """Handle key events."""
        if event.key == "escape":
            self.dismiss(None)
        elif event.key == "up":
            if self.selected_index > 0:
                self.selected_index -= 1
                try:
                    self.query_one("#palette-list").index = self.selected_index
                except Exception:
                    pass
        elif event.key == "down":
            if self.selected_index < len(self.filtered_commands) - 1:
                self.selected_index += 1
                try:
                    self.query_one("#palette-list").index = self.selected_index
                except Exception:
                    pass


# ======================================================================
# HELP SCREEN
# ======================================================================

class HelpScreen(ModalScreen[str]):
    """A help screen showing keyboard shortcuts and commands."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-container {
        width: 70;
        height: auto;
        max-height: 30;
        background: $bg-panel;
        border: solid $border-focused;
        padding: 1 2;
    }

    .help-title {
        text-style: bold;
        color: $accent-primary;
        height: 1;
        margin: 0 0 1 0;
    }

    .help-section {
        color: $neon-cyan;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    .help-row {
        height: 1;
        color: $text-secondary;
    }

    .help-key {
        color: $neon-yellow;
        text-style: bold;
    }

    .help-desc {
        color: $text-muted;
    }
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

    def on_key(self, event) -> None:
        self.dismiss(None)


# ======================================================================
# MAIN TUI APP
# ======================================================================

class DepressionTUI(App):
    """
    Depression.AI - Ultimate Terminal User Interface

    A cyberpunk-themed TUI for the depression.ai coding agent.
    """

    TITLE = "DEPRESSION.AI"
    SUB_TITLE = "Neural Coding Agent"

    CSS = OPENCODE_CSS

    # Key bindings
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

    # Reactive state
    current_mode = reactive[str]("build")
    agent_status = reactive[str]("idle")
    model_name = reactive[str]("—")
    tokens_used = reactive[int](0)
    cost = reactive[float](0.0)
    session_id = reactive[str]("—")

    def __init__(
        self,
        agent_coordinator=None,
        config: dict = None,
        project_dir: str = None,
        model_override: str = None,
        provider_override: str = None,
        yolo: bool = False,
        no_sidebar: bool = False,
        session_id: str = None,
        **kwargs,
    ):
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
        """Build the main layout."""
        # Header
        yield HeaderBar(id="header")

        # Main content area
        with Widget(id="main-container"):
            # Sidebar with LLM connect callback
            yield Sidebar(id="sidebar", on_llm_connect=self._on_llm_connect)

            # Chat area
            yield ChatView(id="chat-panel")

        # Status bar
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize after mounting."""
        # Apply opencode theme
        apply_opencode_theme(self)
        
        self.title = " ◆ DEPRESSION.AI "
        self.sub_title = f" {self.current_mode.upper()} MODE "

        # Set up the sidebar
        self._load_sidebar_data()

        # Sidebar visible by default (opencode style)
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.display = True
        except Exception:
            pass

        # Add welcome message
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system(
            "Welcome to Depression.AI Neural Coding Agent.\n"
            "Type a message or press Ctrl+P for the command palette.\n"
            "Press F1 for help, F2 for sidebar, Tab to switch mode."
        )
        chat.add_divider()

        # Update status bar
        status = self.query_one("#status-bar", StatusBar)
        status.update_all(
            status="idle",
            model=self.model_name,
            tokens=self.tokens_used,
            cost=self.cost,
        )

        # Start agent initialization in background (threaded)
        self._init_task = self.run_worker(self._init_agent_async(), exclusive=True, group="agent-init", thread=True)

        # Focus the input
        try:
            self.query_one("#chat-input", Input).focus()
        except Exception:
            pass

    def _load_sidebar_data(self) -> None:
        """Load sidebar data."""
        sidebar = self.query_one("#sidebar", Sidebar)

        # Default sessions
        sidebar.set_sessions([
            {"id": "current", "name": "Current Session", "time": "now", "active": True},
        ])

        # Default tools
        sidebar.set_tools([
            {"name": "terminal", "enabled": True},
            {"name": "filesystem", "enabled": True},
            {"name": "git", "enabled": True},
            {"name": "search", "enabled": True},
            {"name": "web", "enabled": True},
            {"name": "patch", "enabled": True},
            {"name": "browser", "enabled": True},
            {"name": "task", "enabled": True},
            {"name": "diagnostics", "enabled": True},
        ])

        # Restore LLM selection from saved config
        llm_config = get_llm_config()
        selected = llm_config.get_selected_model_info()
        if selected:
            self.model_name = f"{selected.provider}/{selected.short_name}"
            self.query_one("#header", HeaderBar).set_model(self.model_name)
            self.query_one("#status-bar", StatusBar).set_model(self.model_name)

    def _on_llm_connect(self, model: LLMModel) -> None:
        """Handle LLM provider connection."""
        chat = self.query_one("#chat-panel", ChatView)
        status = self.query_one("#status-bar", StatusBar)

        # Update model display
        self.model_name = f"{model.provider}/{model.short_name}"
        self.query_one("#header", HeaderBar).set_model(self.model_name)
        status.set_model(self.model_name)

        # Store in config
        llm_config = get_llm_config()
        llm_config.selected_model = model.name

        # Notify in chat
        chat.add_system(
            f"Connected to **{model.display_name}** ({model.provider})\n"
            f"Base URL: `{model.base_url}`\n"
            f"Context: {model.context_window // 1000}K tokens"
        )

        # Update the coordinator if available
        if self.coordinator:
            self._apply_llm_to_coordinator(model)

    def _apply_llm_to_coordinator(self, model: LLMModel) -> None:
        """Apply LLM settings to the agent coordinator."""
        try:
            llm_config = get_llm_config()
            api_key = llm_config.get_api_key(model.provider)
            base_url = llm_config.get_base_url(model.provider)

            # Update the LLM provider in the coordinator's agents
            for agent in [self.coordinator.plan_agent, self.coordinator.build_agent]:
                if hasattr(agent, "llm") and agent.llm:
                    if hasattr(agent.llm, "api_key"):
                        agent.llm.api_key = api_key
                    if hasattr(agent.llm, "base_url"):
                        agent.llm.base_url = base_url
                    if hasattr(agent.llm, "model"):
                        agent.llm.model = model.name

            chat = self.query_one("#chat-panel", ChatView)
            chat.add_system(f"Agent updated to use {model.display_name}")
        except Exception as e:
            chat = self.query_one("#chat-panel", ChatView)
            chat.add_error(f"Failed to update agent: {e}")

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------

    async def action_command_palette(self) -> None:
        """Open the command palette."""
        result = await self.push_screen_wait(CommandPalette())
        if result:
            await self._execute_command(result)

    async def action_show_help(self) -> None:
        """Show the help screen."""
        await self.push_screen_wait(HelpScreen())

    def action_clear_screen(self) -> None:
        """Clear the chat."""
        chat = self.query_one("#chat-panel", ChatView)
        chat.clear_messages()

    def action_cancel(self) -> None:
        """Cancel current operation."""
        if self.agent_status in ("thinking", "acting"):
            self.agent_status = "idle"
            status = self.query_one("#status-bar", StatusBar)
            status.set_status("idle")
            status.set_tool("")

    def action_switch_mode(self) -> None:
        """Switch between Plan and Build modes (like Tab in opencode)."""
        modes = ["plan", "build", "auto"]
        current_idx = modes.index(self.current_mode) if self.current_mode in modes else 1
        next_idx = (current_idx + 1) % len(modes)
        self.current_mode = modes[next_idx]

        # Update coordinator mode
        if self.coordinator:
            self.coordinator.set_mode(self.current_mode)

        # Update header
        header = self.query_one("#header", HeaderBar)
        header.set_mode(self.current_mode)

        # Update subtitle
        self.sub_title = f" {self.current_mode.upper()} MODE "

        # Update status bar
        status = self.query_one("#status-bar", StatusBar)
        status.set_status("idle")

        # Notify in chat
        chat = self.query_one("#chat-panel", ChatView)
        mode_desc = {
            "plan": "PLAN mode (read-only analysis)",
            "build": "BUILD mode (full execution)",
            "auto": "AUTO mode (Plan → Build loop)",
        }
        chat.add_system(f"Switched to {mode_desc.get(self.current_mode, self.current_mode.upper())}")

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        sidebar = self.query_one("#sidebar", Sidebar)
        self.sidebar_visible = not self.sidebar_visible
        sidebar.display = self.sidebar_visible

    def action_escape(self) -> None:
        """Handle escape key."""
        # Close any overlays, or clear input
        chat = self.query_one("#chat-panel", ChatView)
        chat.clear_input()

    # ------------------------------------------------------------------
    # COMMAND EXECUTION
    # ------------------------------------------------------------------

    async def _execute_command(self, command: str) -> None:
        """Execute a slash command."""
        chat = self.query_one("#chat-panel", ChatView)

        if command == "/quit":
            self.exit()
            return

        if command == "/clear":
            chat.clear_messages()
            return

        if command == "/help":
            await self.action_show_help()
            return

        if command == "/plan":
            self.current_mode = "plan"
            self.query_one("#header", HeaderBar).set_mode("plan")
            chat.add_system("Switched to PLAN mode (read-only analysis)")
            return

        if command == "/build":
            self.current_mode = "build"
            self.query_one("#header", HeaderBar).set_mode("build")
            chat.add_system("Switched to BUILD mode (full execution)")
            return

        if command == "/auto":
            self.current_mode = "auto"
            self.query_one("#header", HeaderBar).set_mode("auto")
            chat.add_system("Switched to AUTO mode (Plan → Build loop)")
            return

        if command == "/model":
            # Switch to LLM tab in sidebar
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.switch_to_llm()
            chat.add_system("Opening LLM provider panel in sidebar...")
            return

        if command == "/llm":
            # Switch to LLM tab in sidebar
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.switch_to_llm()
            chat.add_system("Opening LLM provider panel...")
            return

        if command == "/status":
            status_text = (
                f"Mode: {self.current_mode}\n"
                f"Model: {self.model_name}\n"
                f"Tokens: {self.tokens_used:,}\n"
                f"Cost: ${self.cost:.4f}\n"
                f"Session: {self.session_id[:8]}"
            )
            chat.add_system(status_text)
            return

        if command == "/tools":
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.active_tab = "tools"
            chat.add_system("Showing tools in sidebar")
            return

        if command.startswith("/session"):
            parts = command.split()
            if len(parts) < 2:
                chat.add_system("Session commands: /session new, /session list, /session load <id>, /session save")
                return
            subcmd = parts[1]
            if subcmd == "new":
                if hasattr(self, 'session_manager') and self.session_manager:
                    try:
                        session = await self.session_manager.create_session()
                        self.set_session(session.id)
                        chat.add_system(f"Created new session: {session.id[:8]}")
                    except Exception as e:
                        chat.add_error(f"Failed to create session: {e}")
                else:
                    chat.add_error("Session manager not available")
            elif subcmd == "list":
                chat.add_system("Session listing not yet implemented")
            elif subcmd == "load" and len(parts) > 2:
                if hasattr(self, 'session_manager') and self.session_manager:
                    try:
                        session = await self.session_manager.load_session(parts[2])
                        self.set_session(session.id)
                        chat.add_system(f"Loaded session: {session.id[:8]}")
                    except Exception as e:
                        chat.add_error(f"Failed to load session: {e}")
                else:
                    chat.add_error("Session manager not available")
            elif subcmd == "save":
                if hasattr(self, 'session_manager') and self.session_manager:
                    try:
                        await self.session_manager.save_current_session()
                        chat.add_system("Session saved")
                    except Exception as e:
                        chat.add_error(f"Failed to save session: {e}")
                else:
                    chat.add_error("Session manager not available")
            else:
                chat.add_system("Session commands: /session new, /session list, /session load <id>, /session save")
            return

        # Unknown command
        chat.add_error(f"Unknown command: {command}")

    # ------------------------------------------------------------------
    # INPUT HANDLING
    # ------------------------------------------------------------------

    @on(Input.Changed, "#chat-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        pass

    @on(Input.Submitted, "#chat-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in the input - send the message."""
        self._send_message()

    def _send_message(self) -> None:
        """Send the current message."""
        chat = self.query_one("#chat-panel", ChatView)
        text = chat.get_input_text().strip()

        if not text:
            return

        # Add user message
        chat.add_message("user", text)
        chat.clear_input()

        # Process the message
        self._process_message(text)

    @work(exclusive=True, group="message-processor")
    async def _process_message(self, text: str) -> None:
        """Process a user message."""
        status = self.query_one("#status-bar", StatusBar)
        chat = self.query_one("#chat-panel", ChatView)

        # Update status
        self.agent_status = "thinking"
        status.set_status("thinking")

        try:
            # If we have a coordinator, use it
            if self.coordinator:
                # Show thinking
                chat.add_thinking("Analyzing...")

                result = await asyncio.wait_for(
                    self.coordinator.process_query(
                        text,
                        mode=self.current_mode,
                        auto_execute=True,
                    ),
                    timeout=120,
                )

                # Remove thinking indicator
                # (In a real app, we'd remove the last thinking message)

                if result.get("success"):
                    response = result.get("execution") or result.get("response", "")
                    chat.add_message("agent", response)

                    # Update metrics
                    if "tokens" in result:
                        self.tokens_used = result["tokens"]
                        status.set_tokens(result["tokens"])
                    if "cost" in result:
                        self.cost = result["cost"]
                        status.set_cost(result["cost"])
                    
                    # Update status bar with model info
                    if self.coordinator:
                        coord_status = self.coordinator.get_status()
                        plan_model = coord_status.get("plan_agent", {}).get("model", "—")
                        build_model = coord_status.get("build_agent", {}).get("model", "—")
                        self.set_model(f"Plan: {plan_model} | Build: {build_model}")
                else:
                    chat.add_error(result.get("error", "Unknown error"))
            else:
                # No coordinator - echo back (demo mode)
                await asyncio.sleep(0.5)
                chat.add_message(
                    "agent",
                    f"Received: {text}\n\n"
                    "This is a demo response. Connect an agent coordinator for full functionality.\n\n"
                    f"Current mode: **{self.current_mode}**\n"
                    f"Use /help for available commands."
                )

        except asyncio.TimeoutError:
            chat.add_error("Request timed out after 120 seconds.")
        except Exception as e:
            chat.add_error(f"Error: {str(e)}")
        finally:
            self.agent_status = "idle"
            status.set_status("idle")
            status.set_tool("")

    # ------------------------------------------------------------------
    # AGENT INTEGRATION
    # ------------------------------------------------------------------

    def set_coordinator(self, coordinator) -> None:
        """Set the agent coordinator."""
        self.coordinator = coordinator

    def set_model(self, model: str) -> None:
        """Update the model name."""
        self.model_name = model
        self.query_one("#header", HeaderBar).set_model(model)
        self.query_one("#status-bar", StatusBar).set_model(model)

    def set_session(self, session_id: str) -> None:
        """Update the session ID."""
        self.session_id = session_id
        self.query_one("#header", HeaderBar).set_session(session_id)

    def update_metrics(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Update token/cost metrics."""
        self.tokens_used = tokens
        self.cost = cost
        self.query_one("#status-bar", StatusBar).update_all(tokens=tokens, cost=cost)

    def show_tool_call(
        self,
        tool_name: str,
        params: dict = None,
        result: Any = None,
        success: bool = True,
        duration: float = None,
    ) -> None:
        """Display a tool call in the chat."""
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_tool_call(tool_name, params, result, success, duration)

    def stream_response(self, text: str) -> None:
        """Stream a response token by token."""
        chat = self.query_one("#chat-panel", ChatView)
        chat.append_stream(text)

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def action_quit(self) -> None:
        """Graceful shutdown."""
        if self.coordinator:
            try:
                await self.coordinator.shutdown()
            except Exception:
                pass
        if hasattr(self, 'session_manager') and self.session_manager:
            try:
                await self.session_manager.save_current_session()
            except Exception:
                pass
        self.exit()

    async def _init_agent_async(self) -> None:
        """Initialize the agent coordinator asynchronously within the app lifecycle."""
        try:
            from agent.config.loader import load_config, get_config
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

            # Load config
            config_dict = load_config(cache=True)

            # Apply CLI overrides
            if self.model_override:
                config_dict.setdefault("llm", {})["model"] = self.model_override
            if self.provider_override:
                config_dict.setdefault("llm", {})["provider"] = self.provider_override
            if self.yolo:
                config_dict.setdefault("permissions", {})["auto_approve"] = True

            # Initialize storage
            db_path = str(get_data_dir(APP_NAME) / "agent.db")
            cache_dir = str(get_cache_dir(APP_NAME))

            database = Database(db_path)
            await database.initialize()

            cache = Cache(cache_dir=cache_dir)

            # Initialize workspace
            project_dir = self.project_dir or config_dict.get("workspace", {}).get("path", ".")
            workspace = WorkspaceManager(
                workspace_dir=str(get_data_dir(APP_NAME)),
                project_dir=project_dir,
            )
            await workspace.initialize()

            # Initialize session
            session_manager = SessionManager(database=database)
            if self.session_id_override:
                session = await session_manager.load_session(self.session_id_override)
            else:
                session = await session_manager.get_or_create_session()

            # Initialize context
            llm_registry = get_llm_registry()
            context_manager = ContextManager(
                workspace=workspace,
                session=session,
                config=config_dict.get("context", {}),
                llm=llm_registry,
            )
            await context_manager.initialize()

            # Initialize permissions
            permission_manager = PermissionManager(
                config=config_dict.get("permissions", {}),
                ui=None,
            )

            # Create dual agent system
            coordinator = await create_dual_agent_system(
                config=config_dict,
                session=session,
                context_manager=context_manager,
                permission_manager=permission_manager,
                workspace=workspace,
                database=database,
                ui=None,
                cache=cache,
            )

            # Wire coordinator into TUI (on main thread)
            self.call_from_thread(self._on_agent_ready, coordinator, session, workspace, session_manager)

        except Exception as e:
            # If initialization fails, show error in chat
            self.call_from_thread(self._on_agent_error, str(e))

    def _on_agent_ready(self, coordinator, session, workspace, session_manager) -> None:
        """Called when agent is ready (on main thread)."""
        self.coordinator = coordinator
        self.session_manager = session_manager
        self.workspace = workspace

        # Set model info
        status = coordinator.get_status()
        plan_model = status.get("plan_agent", {}).get("model", "—")
        build_model = status.get("build_agent", {}).get("model", "—")
        self.set_model(f"Plan: {plan_model} | Build: {build_model}")
        self.set_session(session.id)

        # Load sidebar data
        sidebar = self.query_one("#sidebar", Sidebar)
        tools = []
        for tool_name in coordinator.plan_agent.tool_registry.list_tools():
            tools.append({"name": tool_name, "enabled": True})
        sidebar.set_tools(tools)

        # Load file list
        try:
            files = []
            for f in workspace.list_files():
                files.append({
                    "name": str(f.relative_to(workspace.get_project_dir())),
                    "is_dir": f.is_dir(),
                    "size": self._human_size(f.stat().st_size) if f.is_file() else "",
                })
            sidebar.set_files(files[:100])
        except Exception:
            pass

        # Load sessions
        self._load_sessions_list()

        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system(f"Agent connected! Session: {session.id[:8]}")
        chat.add_divider()

    def _on_agent_error(self, error: str) -> None:
        """Called when agent initialization fails."""
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_error(f"Agent initialization failed: {error}\nRunning in demo mode.")
        chat.add_divider()

    def _load_sessions_list(self) -> None:
        """Load session list in sidebar."""
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            if hasattr(self, 'session_manager') and self.session_manager:
                sessions = [
                    {"id": "current", "name": "Current Session", "time": "now", "active": True},
                ]
                sidebar.set_sessions(sessions)
        except Exception:
            pass

    @staticmethod
    def _human_size(size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def action_toggle_llm_panel(self) -> None:
        """Toggle LLM panel in sidebar."""
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.switch_to_llm()
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system("Opened LLM provider panel")
