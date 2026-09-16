"""Terminal-native agentic interface."""
from __future__ import annotations

import asyncio
import re
import threading
from typing import Any, Optional

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Select, Static

from agent.agent.dual_agent import AgentCoordinator
from agent.llm.provider import get_llm_registry
from agent.llm.runtime import configure_runtime_provider, load_runtime_config
from agent.utils.env_manager import get_aws_credentials, set_aws_credentials

from agent.tui.events import AgentEvent, EventBridge
from agent.tui.theme import (
    GREEN, GREEN_DIM, GREEN_GLOW, AMBER, ERROR,
    TEXT, MUTED, DIM, BG, PANEL, RAISED, BORDER,
    MATRIX_THEME,
)
from agent.tui.widgets.aws_panel import AWSPanel
from agent.tui.widgets.thinking import ThinkingIndicator
from agent.tui.widgets.todo_panel import TodoPanel
from agent.tui.widgets.tool_call import ToolCallWidget


BANNER_LINES = [
    "  ██████╗ ███████╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗",
    "  ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║",
    "  ██║  ██║█████╗  ██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║",
    "  ██║  ██║██╔══╝  ██╔═══╝ ██╔═══╝ ██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║",
    "  ██████╔╝███████╗██║     ██║     ███████╗███████║███████║██║╚██████╔╝██║ ╚████║",
    "  ╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
MODE_ICONS = {"plan": "◇", "build": "◆"}
MODE_ORDER = ["build", "plan"]

DEFAULT_TURN_TIMEOUT = 900.0


# ======================================================================
# LLM PANEL
# ======================================================================

class LLMPanel(Vertical):
    DEFAULT_CSS = f"""
    LLMPanel {{
        width: 100%; height: auto;
        padding: 0 1;
        background: {BG};
    }}
    LLMPanel .label {{ color: {MUTED}; margin-top: 1; }}
    LLMPanel Input {{
        background: {PANEL}; border: round {BORDER};
        color: {TEXT}; margin: 0 0 1 0; height: 3; width: 100%;
    }}
    LLMPanel Input:focus {{ border: round {GREEN}; }}
    LLMPanel Button {{
        width: 1fr; min-width: 12;
        background: transparent; border: round {GREEN};
        color: {GREEN}; margin-top: 1; height: 3; text-style: bold;
    }}
    LLMPanel Button:hover, LLMPanel Button:focus {{
        background: {RAISED}; color: {GREEN_GLOW}; border: round {GREEN_GLOW};
    }}
    LLMPanel .system {{ color: {MUTED}; margin-bottom: 1; }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ LLM CONNECTION[/]", markup=True)
        yield Static("Base URL", classes="label")
        yield Input(value=self._app.cfg["base_url"], id="base-url",
                    placeholder="https://api.example.com/v1")
        yield Static("API Key", classes="label")
        yield Input(value=self._app.cfg["api_key"], password=True,
                    id="api-key", placeholder="API key")
        yield Static("Model ID", classes="label")
        yield Input(value=self._app.cfg["model"], id="model-id",
                    placeholder="provider/model-id")
        yield Button("CONNECT", id="connect")
        yield Button("DISCOVER /models", id="discover")
        yield Static("", id="llm-status", classes="system")


# ======================================================================
# CONTEXT PANEL
# ======================================================================

class ContextPanel(Vertical):
    DEFAULT_CSS = f"""
    ContextPanel {{
        width: 100%; height: auto;
        padding: 0 1;
        background: {BG};
    }}
    ContextPanel .ctx-value {{ color: {GREEN_GLOW}; height: 1; }}
    ContextPanel .ctx-bar   {{ height: 1; margin: 0 0 1 0; }}
    ContextPanel .label     {{ color: {MUTED}; }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ CONTEXT WINDOW[/]", markup=True)
        yield Static("", id="ctx-model", classes="ctx-value")
        yield Static("", id="ctx-window", classes="ctx-value")
        yield Static("", id="ctx-bar", classes="ctx-bar")
        yield Static("", id="ctx-tokens", classes="ctx-value")
        yield Static("", id="ctx-cost", classes="ctx-value")
        yield Static("", id="ctx-msgs", classes="ctx-value")
        yield Static("updates after each agent reply",
                     id="ctx-note", classes="label")

    def on_mount(self) -> None:
        self.refresh_values()

    def refresh_values(self) -> None:
        try:
            model = self._app.cfg.get("model") or "—"
            model = str(model).replace("[", r"\[")
            self.query_one("#ctx-model", Static).update(
                f"[{MUTED}]model[/]   [{TEXT}]{model}[/]")
            self.query_one("#ctx-window", Static).update(
                f"[{MUTED}]window[/]  [{TEXT}]{self._app.context_window:,}[/]")
            used = self._app.tokens_used
            window = max(1, self._app.context_window)
            pct = min(100.0, (used / window) * 100.0)
            bar_w = 30
            filled = int(bar_w * pct / 100.0)
            bar = "█" * filled + "░" * (bar_w - filled)
            color = GREEN if pct < 60 else AMBER if pct < 85 else ERROR
            self.query_one("#ctx-bar", Static).update(f"[{color}]{bar}[/]")
            self.query_one("#ctx-tokens", Static).update(
                f"[{MUTED}]tokens[/]  [{TEXT}]{used:,} / {window:,}  "
                f"([{color}]{pct:.1f}%[/])[/]")
            self.query_one("#ctx-cost", Static).update(
                f"[{MUTED}]cost[/]    [{TEXT}]${self._app.cost:.4f}[/]")
            self.query_one("#ctx-msgs", Static).update(
                f"[{MUTED}]turns[/]   [{TEXT}]{self._app.message_count}[/]")
        except Exception:
            pass


# ======================================================================
# HELP PANEL
# ======================================================================

class HelpPanel(Vertical):
    DEFAULT_CSS = f"""
    HelpPanel {{
        width: 100%; height: auto;
        padding: 0 1;
        background: {BG};
    }}
    HelpPanel .help-section {{ color: {GREEN}; text-style: bold; margin: 1 0 0 0; }}
    HelpPanel .help-row     {{ color: {TEXT}; }}
    """

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ HELP[/]", markup=True)
        yield Static("Commands", classes="help-section")
        yield Static("/connect   open LLM panel", classes="help-row")
        yield Static("/model     open LLM panel + discover models", classes="help-row")
        yield Static("/models    discover models", classes="help-row")
        yield Static("/aws       open AWS panel", classes="help-row")
        yield Static("/context   open context panel", classes="help-row")
        yield Static("/todo      open todo panel", classes="help-row")
        yield Static("/plan      switch mode → plan", classes="help-row")
        yield Static("/build     switch mode → build", classes="help-row")
        yield Static("/clear     clear transcript", classes="help-row")
        yield Static("/quit      exit", classes="help-row")
        yield Static("Keys", classes="help-section")
        yield Static("Tab        cycle mode (build ↔ plan)", classes="help-row")
        yield Static("Esc        interrupt running agent (also denies modal)", classes="help-row")
        yield Static("e          expand / collapse the last tool block", classes="help-row")
        yield Static("Ctrl+L     clear transcript", classes="help-row")
        yield Static("Ctrl+C     interrupt if busy, quit if idle", classes="help-row")
        yield Static("Ctrl+Q     quit", classes="help-row")
        yield Static("Ctrl+D     quit", classes="help-row")
        yield Static("y/a/n/d    allow / deny in the modal", classes="help-row")


# ======================================================================
# PERMISSION MODAL
# ======================================================================

class PermissionModal(ModalScreen[bool]):
    DEFAULT_CSS = f"""
    PermissionModal {{
        align: center middle;
        background: {RAISED} 78%;
    }}
    #perm-card {{
        width: 68%; min-width: 46; min-height: 12; max-height: 20;
        border: thick {GREEN}; background: {PANEL};
        padding: 1 2; layout: vertical;
    }}
    #perm-body {{ height: 1fr; overflow-y: auto; }}
    #perm-buttons {{ height: 3; layout: horizontal; margin-top: 1; }}
    #perm-allow {{
        width: 1fr; height: 3; background: transparent;
        border: round {GREEN}; color: {GREEN}; text-style: bold; margin-right: 1;
    }}
    #perm-allow:focus {{ background: {GREEN}; color: {BG}; }}
    #perm-deny {{
        width: 1fr; height: 3; background: transparent;
        border: round {ERROR}; color: {ERROR}; text-style: bold;
    }}
    #perm-deny:focus {{ background: {ERROR}; color: {BG}; }}
    """

    BINDINGS = [
        Binding("y", "allow", "Allow", priority=True),
        Binding("a", "allow", "Allow", priority=True),
        Binding("n", "deny",  "Deny",  priority=True),
        Binding("d", "deny",  "Deny",  priority=True),
        Binding("escape", "deny", "Deny", priority=True),
        Binding("enter", "default_action", "Select", priority=True),
    ]

    def __init__(self, request: Any, verdict: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self._request = request
        self._verdict = verdict
        self._done = False

    def _esc(self, text: Any) -> str:
        if text is None:
            return ""
        return str(text).replace("[", r"\[")

    def compose(self) -> ComposeResult:
        req, v = self._request, self._verdict
        risk = getattr(v, "risk", "safe")
        if hasattr(risk, "value"):
            risk = risk.value
        risk_color = ERROR if risk in ("high", "critical") else GREEN
        lines = [
            f"[bold {GREEN}]▌ PERMISSION REQUEST[/]",
            f"[{risk_color}]risk: {self._esc(str(risk).upper())}[/]",
            f"[bold {GREEN}]{self._esc(req.tool)}.{self._esc(req.action)}[/]",
        ]
        if getattr(v, "reason", None):
            lines.append(f"[{MUTED}]{self._esc(v.reason)}[/]")
        if getattr(req, "params", None):
            for k, val in list(req.params.items())[:4]:
                sval = str(val)
                if len(sval) > 60:
                    sval = sval[:57] + "…"
                lines.append(f"[{DIM}]{self._esc(k)}: {self._esc(sval)}[/]")
        lines.append("")
        lines.append(f"[{MUTED}][A] allow   [D] deny   (enter = default)[/]")

        with Vertical(id="perm-card"):
            yield Static("\n".join(lines), id="perm-body", markup=True)
            with Horizontal(id="perm-buttons"):
                yield Button("ALLOW [A]", id="perm-allow")
                yield Button("DENY  [D]", id="perm-deny")

    def on_mount(self) -> None:
        self._focus_default()
        self.set_timer(0.05, self._focus_default)

    def _focus_default(self) -> None:
        try:
            risk = getattr(self._verdict, "risk", "safe")
            if hasattr(risk, "value"):
                risk = risk.value
            btn_id = ("perm-allow"
                      if str(risk) in ("safe", "low", "medium")
                      else "perm-deny")
            self.query_one(f"#{btn_id}", Button).focus()
        except Exception:
            pass

    def _finish(self, allowed: bool) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.dismiss(allowed)
        except Exception:
            pass

    def action_allow(self) -> None: self._finish(True)
    def action_deny(self) -> None:  self._finish(False)

    def action_default_action(self) -> None:
        focused = self.focused
        if isinstance(focused, Button) and focused.id == "perm-deny":
            self._finish(False)
        else:
            self._finish(True)

    @on(Button.Pressed, "#perm-allow")
    def _allow_btn(self, event: Button.Pressed) -> None: self._finish(True)

    @on(Button.Pressed, "#perm-deny")
    def _deny_btn(self, event: Button.Pressed) -> None: self._finish(False)


# ======================================================================
# APP
# ======================================================================

class DepressionApp(App):
    TITLE = "depression.ai"

    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}

    #body {{ height: 1fr; layout: horizontal; }}
    #main-col {{ width: 1fr; height: 1fr; layout: vertical; }}

    #banner-area {{ height: 8; padding: 1 2 0 2; background: {BG}; }}
    #banner {{ height: 6; width: 100%; }}
    #banner-sub {{ height: 1; color: {GREEN_DIM}; padding: 0 0 0 2; }}

    #transcript {{
        height: 1fr; width: 100%;
        padding: 1 2 0 2;
        scrollbar-background: {BG};
        scrollbar-color: {BORDER};
    }}
    .user       {{ color: {GREEN_GLOW}; margin-bottom: 1; }}
    .agent      {{ color: {TEXT}; margin-bottom: 1; }}
    .agent-head {{ color: {GREEN}; text-style: bold; }}
    .system     {{ color: {MUTED}; margin-bottom: 1; }}
    .error      {{ color: {ERROR}; margin-bottom: 1; }}
    .queued     {{ color: {AMBER}; margin-bottom: 1; }}

    #sidebar {{
        width: 46; min-width: 46; max-width: 46;
        height: 1fr; background: {BG};
        border-left: solid {BORDER}; layout: vertical;
    }}
    #side-title {{
        height: 1; padding: 0 1;
        background: {BG}; color: {GREEN}; text-style: bold;
        border-bottom: solid {BORDER};
    }}
    #side-tabs {{
        height: 1; layout: horizontal;
        background: {BG}; border-bottom: solid {BORDER};
    }}
    .side-tab {{
        width: 1fr; min-width: 8; height: 1;
        background: transparent; border: none;
        color: {DIM}; text-style: bold;
    }}
    .side-tab:hover {{ color: {MUTED}; }}
    .side-tab.active {{ color: {GREEN}; }}

    #side-content {{
        height: 1fr; overflow-y: auto; overflow-x: hidden;
        background: {BG}; padding: 1 0;
    }}
    #panel-llm, #panel-aws, #panel-context, #panel-todo, #panel-help {{
        width: 100%; height: auto;
    }}

    #prompt-wrap {{
        dock: bottom; height: auto;
        background: {BG};
        border-left: thick {GREEN};
        padding: 1 0 0 2; margin: 0 0 0 2;
    }}
    #prompt-row {{ height: 3; layout: horizontal; }}
    #prompt-sign {{
        width: 2; height: 3; color: {GREEN};
        content-align: left middle; text-style: bold;
    }}
    #prompt {{
        width: 1fr; height: 3;
        background: {BG}; border: none; color: {TEXT}; padding: 0;
    }}
    #prompt:focus {{ border: none; }}
    #prompt.busy {{ color: {AMBER}; }}
    #mode-chip {{ height: 1; padding: 0 0 0 2; color: {MUTED}; background: {BG}; }}
    #hint {{ height: 1; padding: 0 0 0 2; color: {DIM}; background: {BG}; margin-bottom: 0; }}

    #thinking-bar {{ height: 1; background: {BG}; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+c", "cancel", "Cancel", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=False),
        Binding("ctrl+l", "clear", "Clear", priority=True),
        Binding("tab", "cycle_mode", "Mode", priority=True),
        Binding("e", "toggle_expand_tool", "Expand", priority=False),
    ]

    current_mode: reactive[str] = reactive("build")
    tokens_used: reactive[int] = reactive(0)
    cost: reactive[float] = reactive(0.0)
    message_count: reactive[int] = reactive(0)
    context_window: reactive[int] = reactive(200_000)
    queue_depth: reactive[int] = reactive(0)

    def __init__(
        self,
        coordinator: Optional[AgentCoordinator] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.coordinator = coordinator
        self.cfg = load_runtime_config()
        self.aws = get_aws_credentials()

        self.busy = False
        self._banner_phase = 0
        self._active_panel = "llm"
        self._agent_worker = None
        self._prompt_queue: list[str] = []
        self._draining = False
        self._is_shutting_down = False

        import os as _os
        try:
            self.turn_timeout = float(
                _os.environ.get("TUI_TURN_TIMEOUT", DEFAULT_TURN_TIMEOUT)
            )
        except Exception:
            self.turn_timeout = DEFAULT_TURN_TIMEOUT

        self._events = EventBridge()
        self._event_handlers_registered = False
        self._active_tool_widget: Optional[ToolCallWidget] = None

    # ------------------------------------------------------------------
    # COMPOSE
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="main-col"):
                with Vertical(id="banner-area"):
                    yield Static(self._banner_markup(), id="banner", markup=True)
                    yield Static(
                        f"  [{GREEN_DIM}]✧  a  t  e  r  m  i  n  a  l  "
                        f"c  o  d  i  n  g  a  g  e  n  t  ✧[/]",
                        id="banner-sub", markup=True,
                    )
                yield VerticalScroll(id="transcript")
                yield ThinkingIndicator(id="thinking-bar")

            with Vertical(id="sidebar"):
                yield Static("▌ SIDEBAR", id="side-title", markup=True)
                with Horizontal(id="side-tabs"):
                    yield Button("CONNECT", id="tab-llm",
                                 classes="side-tab active")
                    yield Button("AWS", id="tab-aws", classes="side-tab")
                    yield Button("CTX", id="tab-context", classes="side-tab")
                    yield Button("TODO", id="tab-todo", classes="side-tab")
                    yield Button("HELP", id="tab-help", classes="side-tab")
                with VerticalScroll(id="side-content"):
                    yield LLMPanel(self, id="panel-llm")
                    yield AWSPanel(self, id="panel-aws")
                    yield ContextPanel(self, id="panel-context")
                    yield TodoPanel(session=self._session(), id="panel-todo")
                    yield HelpPanel(id="panel-help")

        with Vertical(id="prompt-wrap"):
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-sign")
                yield Input(placeholder="ask the agent…", id="prompt")
            yield Static(self._mode_chip(), id="mode-chip", markup=True)
            yield Static(
                "tab mode  ·  esc interrupt  ·  e expand  ·  ctrl+c exit  ·  "
                "/connect /model /aws /context /todo /help",
                id="hint", markup=True,
            )

        yield Footer()

    def _session(self) -> Any:
        if self.coordinator is None:
            return None
        for attr in ("plan_agent", "build_agent"):
            agent = getattr(self.coordinator, attr, None)
            if agent is not None and getattr(agent, "session", None) is not None:
                return agent.session
        return None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.register_theme(MATRIX_THEME)
        self.theme = "matrix"

        self._show_panel("llm")

        for w in self.query("#sidebar Button, .side-tab"):
            try:
                w.can_focus = False
            except Exception:
                pass

        self._wire_events()

        self._system(
            "welcome. type a prompt to run the agent — "
            "/connect to set LLM, /model to discover, /aws for AWS, "
            "/context for tokens, /todo for tasks, /help for all commands."
        )

        self.query_one("#prompt", Input).focus()
        self.set_interval(1.50, self._tick_banner)

        self._refresh_aws_status()

    def _wire_events(self) -> None:
        if self._event_handlers_registered:
            return
        if self.coordinator is None:
            return
        try:
            self._events.set_ui_loop(asyncio.get_running_loop())
            self._events.attach(self.coordinator)
            self._events.subscribe("on_tool_executed", self._on_tool_event)
            self._event_handlers_registered = True
        except Exception as exc:
            self._error(f"event wiring failed: {exc}")

    async def _on_tool_event(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        if not self.is_running:
            return

        data = event.payload or {}
        tool = str(data.get("tool") or "tool")
        params = data.get("params") or {}
        result = data.get("result") or {}
        cached = bool(data.get("cached"))
        duration = data.get("execution_time")

        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return

        widget = ToolCallWidget(tool=tool, params=params)
        await transcript.mount(widget)
        widget.set_running(execution_time=duration)
        widget.set_result(
            result if isinstance(result, dict) else {"success": True, "result": result},
            execution_time=duration,
            cached=cached,
        )
        self._active_tool_widget = widget
        self.call_after_refresh(lambda: transcript.scroll_end(animate=False))

        if tool in ("aws",) or tool.startswith("mcp__aws__"):
            try:
                self.query_one("#panel-aws", AWSPanel).refresh_mcp_status()
            except Exception:
                pass

        try:
            self._refresh_context_panel()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PERMISSION CONFIRMATION
    # ------------------------------------------------------------------

    def install_permission_callback(self, callback) -> None:
        if self.coordinator is None:
            return
        for attr in ("plan_agent", "build_agent"):
            agent = getattr(self.coordinator, attr, None)
            if agent is None:
                continue
            pm = getattr(agent, "permission_manager", None)
            if pm is not None:
                try:
                    pm.input_handler = None
                    pm.set_confirm_callback(callback)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # BANNER / MODE CHIP
    # ------------------------------------------------------------------

    def _banner_markup(self) -> str:
        greens = [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM]
        return "\n".join(
            f"[bold {greens[i]}]{line}[/]"
            for i, line in enumerate(BANNER_LINES)
        )

    def _mode_chip(self) -> str:
        icon = MODE_ICONS.get(self.current_mode, "◆")
        model = self.cfg.get("model") or "no model selected"
        if len(model) > 28:
            model = model[:28] + "…"
        model = str(model).replace("[", r"\[")

        prefix = ""
        if self.busy:
            spin = SPINNER_FRAMES[self.message_count % len(SPINNER_FRAMES)]
            prefix = f"[{AMBER}]{spin}[/] "

        pct = (self.tokens_used / max(1, self.context_window)) * 100.0
        ctx = f"[{MUTED}]ctx {pct:.0f}%[/]"

        queue = ""
        if self.queue_depth > 0:
            queue = f"  [{DIM}]·[/]  [{AMBER}]⧗ {self.queue_depth} queued[/]"

        return (
            f"{prefix}[{GREEN}]{icon} {self.current_mode.upper()}[/]  "
            f"[{DIM}]·[/]  [{TEXT}]{model}[/]  "
            f"[{DIM}]·[/]  {ctx}{queue}"
        )

    _BUSY_PLACEHOLDER = "agent running… type to queue · esc to interrupt"

    def _set_busy_visual(self, busy: bool) -> None:
        try:
            inp = self.query_one("#prompt", Input)
            if busy:
                inp.placeholder = self._BUSY_PLACEHOLDER
                inp.add_class("busy")
            else:
                inp.placeholder = "ask the agent…"
                inp.remove_class("busy")
        except Exception:
            pass
        try:
            thinking = self.query_one("#thinking-bar", ThinkingIndicator)
            if busy:
                thinking.start()
            else:
                thinking.stop()
        except Exception:
            pass

    def _refresh_mode_chip(self) -> None:
        try:
            self.query_one("#mode-chip", Static).update(self._mode_chip())
        except Exception:
            pass

    def _refresh_context_panel(self) -> None:
        try:
            self.query_one("#panel-context", ContextPanel).refresh_values()
        except Exception:
            pass

    def _refresh_aws_status(self) -> None:
        try:
            self.query_one("#panel-aws", AWSPanel).refresh_mcp_status()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PANEL SWITCHING
    # ------------------------------------------------------------------

    def _show_panel(self, which: str) -> None:
        self._active_panel = which
        for name in ("llm", "aws", "context", "todo", "help"):
            try:
                self.query_one(f"#panel-{name}").display = (name == which)
            except Exception:
                pass
            try:
                btn = self.query_one(f"#tab-{name}")
                btn.remove_class("active")
                if name == which:
                    btn.add_class("active")
            except Exception:
                pass
        if which == "context":
            self._refresh_context_panel()
        if which == "aws":
            self._refresh_aws_status()

    @on(Button.Pressed, ".side-tab")
    def _on_tab(self, event: Button.Pressed) -> None:
        self._show_panel(event.button.id.replace("tab-", ""))
        self._refocus_prompt()

    def _tick_banner(self) -> None:
        if self._is_shutting_down:
            return
        self._banner_phase = (self._banner_phase + 1) % 3
        greens = [
            [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM],
            [GREEN, GREEN_GLOW, GREEN, GREEN_DIM, GREEN, GREEN_DIM],
            [GREEN, GREEN, GREEN_GLOW, GREEN_DIM, GREEN_DIM, GREEN],
        ][self._banner_phase]
        rows = "\n".join(
            f"[bold {greens[i]}]{line}[/]"
            for i, line in enumerate(BANNER_LINES)
        )
        try:
            self.query_one("#banner", Static).update(rows)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # TRANSCRIPT
    # ------------------------------------------------------------------

    _CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def _sanitize(self, text: str) -> str:
        clean = self._CTRL_RE.sub("", str(text))
        # Textual markup treats [ as the start of a tag. Escape only the
        # opening bracket; a bare ] is a literal in Textual's parser.
        # Do NOT use markup_escape — some Textual versions escape both
        # brackets in a way the parser rejects on round-trip.
        return clean.replace("[", r"\[")

    def _write(self, text: str, cls: str = "agent") -> None:
        if self._is_shutting_down:
            return
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        transcript.mount(Static(self._sanitize(text), classes=cls, markup=True))
        self.call_after_refresh(
            lambda: transcript.scroll_end(animate=False)
        )

    def _system(self, text: str) -> None:
        self._write(f"· {text}", "system")

    def _error(self, text: str) -> None:
        self._write(f"× {text}", "error")

    def _queued(self, text: str) -> None:
        self._write(f"⧗ {text}", "queued")

    def _user(self, text: str) -> None:
        if self._is_shutting_down:
            return
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        safe = str(text).replace("[", r"\[")
        transcript.mount(
            Static(
                f"[bold {GREEN}]›[/] [bold {TEXT}]{safe}[/]",
                classes="user", markup=True,
            )
        )
        self.message_count = self.message_count + 1
        self.call_after_refresh(
            lambda: transcript.scroll_end(animate=False)
        )

    def _agent_head(self) -> None:
        self._write(f"[bold {GREEN}]◆ depression.ai[/]", "agent-head")

    def _refocus_prompt(self) -> None:
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # INPUT / COMMANDS
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            self._slash(text)
            return

        if self.busy or self._draining:
            self._prompt_queue.append(text)
            self.queue_depth = len(self._prompt_queue)
            self._refresh_mode_chip()
            preview = text if len(text) <= 60 else text[:60] + "…"
            self._queued(
                f"queued ({len(self._prompt_queue)} ahead): {preview}"
            )
            return

        self._agent_worker = self._run_agent(text)

    def _slash(self, text: str) -> None:
        command = text.split()[0].lower()
        if command == "/aws":
            self._show_panel("aws")
            try:
                self.query_one("#aws-key", Input).focus()
            except Exception:
                pass
        elif command == "/connect":
            self._show_panel("llm")
            try:
                self.query_one("#base-url", Input).focus()
            except Exception:
                pass
        elif command in ("/models", "/model"):
            self._show_panel("llm")
            self._discover_models()
        elif command == "/context":
            self._show_panel("context")
        elif command == "/todo":
            self._show_panel("todo")
        elif command == "/help":
            self._show_panel("help")
        elif command == "/clear":
            try:
                self.query_one("#transcript", VerticalScroll).remove_children()
            except Exception:
                pass
        elif command == "/plan":
            self._switch_mode("plan")
        elif command == "/build":
            self._switch_mode("build")
        elif command in ("/quit", "/exit"):
            self._is_shutting_down = True
            self.exit()
        else:
            self._error(f"unknown command: {command}")

    def _switch_mode(self, mode: str) -> None:
        if mode not in MODE_ORDER:
            return
        self.current_mode = mode
        if self.coordinator is not None:
            try:
                self.coordinator.set_mode(mode)
            except Exception:
                pass
        self._refresh_mode_chip()
        self._system(f"mode → {mode}")

    # ------------------------------------------------------------------
    # AGENT RUNNER
    # ------------------------------------------------------------------

    @work(exclusive=True, group="agent")
    async def _run_agent(self, text: str) -> None:
        self.busy = True
        self._set_busy_visual(True)
        self._refresh_mode_chip()
        self._user(text)

        if self.coordinator is None:
            self._error("agent coordinator is not initialized")
            self.busy = False
            self._set_busy_visual(False)
            self._refresh_mode_chip()
            self._schedule_drain()
            return

        try:
            result = await asyncio.wait_for(
                self.coordinator.process_query(
                    text,
                    mode=self.current_mode,
                    auto_execute=True,
                ),
                timeout=self.turn_timeout,
            )

            if result.get("success"):
                output = (
                    result.get("execution")
                    or result.get("response")
                    or "task completed."
                )
                self._agent_head()
                self._write(str(output), "agent")

                ctx = result.get("context") or {}
                tokens = (
                    ctx.get("tokens_used")
                    or result.get("tokens_used")
                    or 0
                )
                if tokens:
                    self.tokens_used = int(tokens)
                cost = ctx.get("cost") or 0.0
                try:
                    self.cost = float(cost)
                except Exception:
                    pass
                self._refresh_context_panel()
                self._refresh_mode_chip()
            else:
                self._error(str(result.get("error", "agent request failed")))
        except asyncio.TimeoutError:
            mins = self.turn_timeout / 60.0
            self._error(
                f"agent timed out after {mins:.0f} min with no result. "
                "Press Esc if you want to keep waiting, or raise "
                "TUI_TURN_TIMEOUT in the environment."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error(f"agent error: {exc}")
        finally:
            self.busy = False
            self._set_busy_visual(False)
            self._refresh_mode_chip()
            self._refocus_prompt()
            self._schedule_drain()

    def _schedule_drain(self) -> None:
        if self._is_shutting_down:
            return
        self._draining = True
        self.call_after_refresh(self._drain_prompt_queue)

    def _drain_prompt_queue(self) -> None:
        self._draining = False
        if self._is_shutting_down:
            return
        if self.busy or not self._prompt_queue:
            self._refresh_mode_chip()
            return
        nxt = self._prompt_queue.pop(0)
        self.queue_depth = len(self._prompt_queue)
        self._refresh_mode_chip()
        self._agent_worker = self._run_agent(nxt)

    def is_agent_busy(self) -> bool:
        if self.busy or self._draining:
            return True
        worker = self._agent_worker
        return worker is not None and getattr(worker, "is_running", False)

    def cancel_current_agent(self) -> None:
        worker = self._agent_worker
        if worker is not None and getattr(worker, "is_running", False):
            worker.cancel()
        self._prompt_queue.clear()
        self.queue_depth = 0
        self._draining = False
        self.busy = False
        self._set_busy_visual(False)
        self._refresh_mode_chip()

    # ------------------------------------------------------------------
    # LLM PANEL ACTIONS
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#connect")
    def connect_llm(self) -> None:
        base_url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        api_key = key_input.value.strip()
        model = self.query_one("#model-id", Input).value.strip()
        status = self.query_one("#llm-status", Static)
        if not base_url or not api_key or not model:
            status.update("base URL, API key and model are required")
            return
        try:
            configure_runtime_provider(
                get_llm_registry(), base_url, api_key, model
            )
            self.cfg = load_runtime_config()
            status.update(f"connected · {model}")
            self._refresh_mode_chip()
            self._refresh_context_panel()
            self._system(f"LLM connected: {base_url} · {model}")
        except Exception as exc:
            status.update(f"error: {exc}")

    @on(Button.Pressed, "#discover")
    def discover_llm(self) -> None:
        self._discover_models()

    @work(exclusive=True, group="discover")
    async def _discover_models(self) -> None:
        url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        key = key_input.value.strip()
        status = self.query_one("#llm-status", Static)
        if not url:
            status.update("enter a base URL first")
            return
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                headers = (
                    {"Authorization": f"Bearer {key}"} if key else {}
                )
                response = await client.get(f"{url}/models", headers=headers)
                response.raise_for_status()
                models = response.json().get("data", [])
            ids = [str(m.get("id")) for m in models if m.get("id")]
            if ids:
                self.query_one("#model-id", Input).value = ids[0]
                status.update(
                    f"discovered {len(ids)} model(s); selected {ids[0]}"
                )
                self._system("models: " + ", ".join(ids[:12]))
            else:
                status.update("/models returned no model IDs")
        except Exception as exc:
            status.update(f"discovery failed: {exc}; enter model manually")

    # ------------------------------------------------------------------
    # AWS PANEL ACTIONS
    # ------------------------------------------------------------------

    @on(Input.Changed, "#aws-key")
    def _on_aws_key_change(self, event: Input.Changed) -> None:
        try:
            self.query_one("#panel-aws", AWSPanel).refresh_strength()
        except Exception:
            pass

    @on(Button.Pressed, "#aws-save")
    def save_aws(self) -> None:
        key = self.query_one("#aws-key", Input).value.strip()
        secret_input = self.query_one("#aws-secret", Input)
        secret = secret_input.value.strip()
        region = str(self.query_one("#aws-region", Select).value or "")
        panel = self.query_one("#panel-aws", AWSPanel)

        if not key or not secret or not region:
            panel.set_status("access key, secret, and region are required")
            return

        panel.start_radar()
        try:
            set_aws_credentials(key, secret, region)
            self.aws = get_aws_credentials()
            panel.refresh_values()
            panel.set_status(f"saved to .env · {region}")
            panel.start_scan(passes=2)
            panel.start_trace(ok=True)

            note = ""
            try:
                from agent.mcp.aws_config import install_aws_preset_into_config
                mcp_cfg = None
                if self.coordinator is not None:
                    for attr in ("plan_agent", "build_agent"):
                        agent = getattr(self.coordinator, attr, None)
                        client = getattr(agent, "mcp_client", None) if agent else None
                        if client is not None and hasattr(client, "config"):
                            mcp_cfg = client.config
                            break
                if mcp_cfg is not None:
                    if install_aws_preset_into_config(mcp_cfg, region=region):
                        note = " · MCP config updated (restart to apply)"
            except Exception as exc:
                note = f" · MCP config refresh skipped: {exc}"

            panel.refresh_mcp_status()
            self._system(
                f"AWS credentials saved to .env ({region}){note}; "
                "secret is not displayed."
            )
        except Exception as exc:
            panel.set_status(f"error: {exc}")
            panel.start_trace(ok=False)

    # ------------------------------------------------------------------
    # KEYS
    # ------------------------------------------------------------------

    def action_interrupt(self) -> None:
        if isinstance(self.screen, PermissionModal):
            self.screen.action_deny()
            return
        worker = self._agent_worker
        running = worker is not None and getattr(worker, "is_running", False)
        if not self.busy and not running:
            return
        if running:
            worker.cancel()
        if self._prompt_queue:
            self._system(f"dropped {len(self._prompt_queue)} queued prompt(s)")
            self._prompt_queue.clear()
            self.queue_depth = 0
        self._draining = False
        self._system("interrupted by ESC")
        self.busy = False
        self._set_busy_visual(False)
        self._refresh_mode_chip()

    def action_cancel(self) -> None:
        if isinstance(self.screen, PermissionModal):
            self.screen.action_deny()
            return
        if self.busy:
            self.action_interrupt()
            self._system("cancelled — press Ctrl+C again to quit")
            return
        self._is_shutting_down = True
        self.exit()

    def action_quit(self) -> None:
        self._is_shutting_down = True
        self.exit()

    def action_clear(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).remove_children()
        except Exception:
            pass

    def action_toggle_expand_tool(self) -> None:
        widget = self._active_tool_widget
        if widget is None:
            return
        try:
            widget.toggle_expand()
            transcript = self.query_one("#transcript", VerticalScroll)
            self.call_after_refresh(lambda: transcript.scroll_end(animate=False))
        except Exception:
            pass

    def action_cycle_mode(self) -> None:
        i = (
            MODE_ORDER.index(self.current_mode)
            if self.current_mode in MODE_ORDER else 0
        )
        self._switch_mode(MODE_ORDER[(i + 1) % len(MODE_ORDER)])


__all__ = ["DepressionApp", "PermissionModal"]