"""depression.ai — matrix green TUI, wired to the real AgentCoordinator."""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Footer, Input, Select, Static
from textual import on, work

from agent.agent.dual_agent import AgentCoordinator
from agent.llm.provider import get_llm_registry
from agent.llm.runtime import configure_runtime_provider, load_runtime_config
from agent.utils.env_manager import get_aws_credentials, set_aws_credentials


GREEN       = "#00ff66"
GREEN_DIM   = "#00aa44"
GREEN_FAINT = "#005522"
GREEN_GLOW  = "#88ffbb"
AMBER       = "#ffcc44"
ERROR       = "#ff4466"
TEXT        = "#aaffcc"
MUTED       = "#3d8c5c"
DIM         = "#1a5c33"
BG          = "#000000"
PANEL       = "#031008"
RAISED      = "#061a0f"
BORDER      = "#0a3d20"


THEME = Theme(
    name="matrix",
    primary=GREEN, secondary=GREEN_DIM, accent=GREEN_GLOW,
    warning=AMBER, error=ERROR, success=GREEN,
    surface=PANEL, panel=RAISED, boost="#0d2a17",
    foreground=TEXT, background=BG, dark=True,
)


BANNER_LINES = [
    "  ██████╗ ███████╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗",
    "  ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║",
    "  ██║  ██║█████╗  ██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║",
    "  ██║  ██║██╔══╝  ██╔═══╝ ██╔═══╝ ██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║",
    "  ██████╔╝███████╗██║     ██║     ███████╗███████║███████║██║╚██████╔╝██║ ╚████║",
    "  ╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
MODE_ICONS     = {"plan": "◇", "build": "◆", "auto": "⟡"}
MODE_ORDER     = ["plan", "build", "auto"]

AWS_REGIONS = [
    "ap-south-1", "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1", "ap-southeast-1",
    "ap-southeast-2", "ap-northeast-1", "ca-central-1", "sa-east-1",
]


class LLMPanel(Static):
    DEFAULT_CSS = f"""
    LLMPanel {{
        width: 100%; height: auto;
        layout: vertical;
        padding: 0 1;
        background: {BG};
    }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static("▌ LLM CONNECTION", classes="panel-title")
        yield Static("Base URL", classes="label")
        yield Input(value=self._app.cfg["base_url"], id="base-url",
                    placeholder="https://api.example.com/v1")
        yield Static("API Key", classes="label")
        yield Input(value="••••••••" if self._app.cfg["api_key"] else "",
                    password=True, id="api-key", placeholder="API key")
        yield Static("Model ID", classes="label")
        yield Input(value=self._app.cfg["model"], id="model-id",
                    placeholder="provider/model-id")
        yield Button("CONNECT", id="connect")
        yield Button("DISCOVER /models", id="discover")
        yield Static("", id="llm-status", classes="system")


class AWSPanel(Static):
    DEFAULT_CSS = f"""
    AWSPanel {{
        width: 100%; height: auto;
        layout: vertical;
        padding: 0 1;
        background: {BG};
    }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static("▌ AWS CREDENTIALS", classes="panel-title")
        yield Static("Access Key ID", classes="label")
        yield Input(value=self._app.aws.get("access_key") or "", id="aws-key")
        yield Static("Secret Access Key", classes="label")
        yield Input(value="••••••••" if self._app.aws.get("secret_key") else "",
                    password=True, id="aws-secret")
        yield Static("Region", classes="label")
        yield Select([(r, r) for r in AWS_REGIONS],
                     value=self._app.aws.get("region") or "ap-south-1",
                     id="aws-region")
        yield Button("SAVE AWS", id="aws-save")
        yield Static("", id="aws-status", classes="system")


class ContextPanel(Static):
    DEFAULT_CSS = f"""
    ContextPanel {{
        width: 100%; height: auto;
        layout: vertical;
        padding: 0 1;
        background: {BG};
    }}
    .ctx-value {{ color: {GREEN_GLOW}; height: 1; }}
    .ctx-bar   {{ height: 1; margin: 0 0 1 0; }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static("▌ CONTEXT WINDOW", classes="panel-title")
        yield Static("", id="ctx-model",  classes="ctx-value")
        yield Static("", id="ctx-window", classes="ctx-value")
        yield Static("", id="ctx-bar",    classes="ctx-bar")
        yield Static("", id="ctx-tokens", classes="ctx-value")
        yield Static("", id="ctx-cost",   classes="ctx-value")
        yield Static("", id="ctx-msgs",   classes="ctx-value")
        yield Static("", id="ctx-note",   classes="label")

    def on_mount(self) -> None:
        self.refresh_values()

    def refresh_values(self) -> None:
        try:
            self.query_one("#ctx-model", Static).update(
                f"[{MUTED}]model[/]   [{TEXT}]{self._app.cfg.get('model') or '—'}[/]")
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
            self.query_one("#ctx-note", Static).update(
                "updates after each agent reply")
        except Exception:
            pass


class HelpPanel(Static):
    DEFAULT_CSS = f"""
    HelpPanel {{
        width: 100%; height: auto;
        layout: vertical;
        padding: 0 1;
        background: {BG};
    }}
    .help-section {{ color: {GREEN}; text-style: bold; margin: 1 0 0 0; }}
    .help-row     {{ color: {TEXT}; }}
    """

    def compose(self) -> ComposeResult:
        yield Static("▌ HELP", classes="panel-title")
        yield Static("Commands", classes="help-section")
        yield Static("/connect   open LLM panel", classes="help-row")
        yield Static("/aws       open AWS panel", classes="help-row")
        yield Static("/context   open context panel", classes="help-row")
        yield Static("/models    discover models", classes="help-row")
        yield Static("/plan      switch mode → plan", classes="help-row")
        yield Static("/build     switch mode → build", classes="help-row")
        yield Static("/auto      switch mode → auto", classes="help-row")
        yield Static("/clear     clear transcript", classes="help-row")
        yield Static("/quit      exit", classes="help-row")
        yield Static("Keys", classes="help-section")
        yield Static("Tab        cycle mode", classes="help-row")
        yield Static("Esc        interrupt session / deny permission",
                     classes="help-row")
        yield Static("Ctrl+L     clear transcript", classes="help-row")
        yield Static("Ctrl+C     cancel or exit", classes="help-row")
        yield Static("y / n      allow / deny in the permission modal",
                     classes="help-row")


class PermissionModal(ModalScreen[bool]):
    """In-UI allow/deny prompt for tool permission requests."""

    DEFAULT_CSS = f"""
    PermissionModal {{
        align: center middle;
        background: {RAISED} 78%;
    }}
    #perm-card {{
        width: 68%;
        min-width: 46;
        min-height: 12;
        max-height: 20;
        border: thick {GREEN};
        background: {PANEL};
        padding: 1 2;
        layout: vertical;
    }}
    #perm-body {{
        height: 1fr;
        overflow-y: auto;
    }}
    #perm-buttons {{
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }}
    #perm-allow {{
        width: 1fr;
        height: 3;
        background: transparent;
        border: round {GREEN};
        color: {GREEN};
        text-style: bold;
        margin-right: 1;
    }}
    #perm-allow:focus {{
        background: {GREEN};
        color: {BG};
    }}
    #perm-deny {{
        width: 1fr;
        height: 3;
        background: transparent;
        border: round {ERROR};
        color: {ERROR};
        text-style: bold;
    }}
    #perm-deny:focus {{
        background: {ERROR};
        color: {BG};
    }}
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

    def compose(self) -> ComposeResult:
        req, v = self._request, self._verdict
        risk = getattr(v, "risk", "safe")
        if hasattr(risk, "value"):
            risk = risk.value
        risk_color = ERROR if risk in ("high", "critical") else GREEN
        lines = [
            f"[bold {GREEN}]▌ PERMISSION REQUEST[/]",
            f"[{risk_color}]risk: {str(risk).upper()}[/]",
            f"[bold {GREEN}]{req.tool}.{req.action}[/]",
        ]
        if getattr(v, "reason", None):
            lines.append(f"[{MUTED}]{v.reason}[/]")
        if getattr(req, "params", None):
            for k, val in list(req.params.items())[:4]:
                sval = str(val)
                if len(sval) > 60:
                    sval = sval[:57] + "…"
                lines.append(f"[{DIM}]{k}: {sval}[/]")
        lines.append("")
        lines.append(f"[{MUTED}][A] allow   [D] deny   (enter = default)[/]")

        with Vertical(id="perm-card"):
            yield Static("\n".join(lines), id="perm-body")
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
            btn_id = ("perm-allow" if str(risk) in ("safe", "low", "medium")
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

    def action_allow(self) -> None:
        self._finish(True)

    def action_deny(self) -> None:
        self._finish(False)

    def action_default_action(self) -> None:
        focused = self.focused
        if isinstance(focused, Button) and focused.id == "perm-deny":
            self._finish(False)
        else:
            self._finish(True)

    @on(Button.Pressed, "#perm-allow")
    def _allow_btn(self, event: Button.Pressed) -> None:
        self._finish(True)

    @on(Button.Pressed, "#perm-deny")
    def _deny_btn(self, event: Button.Pressed) -> None:
        self._finish(False)


class DepressionApp(App):
    TITLE = "depression.ai"

    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}

    #body {{
        height: 1fr;
        layout: horizontal;
    }}

    #main-col {{
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }}
    #banner-area {{
        height: 8;
        padding: 1 2 0 2;
        background: {BG};
    }}
    #banner {{ height: 6; width: 100%; }}
    #banner-sub {{ height: 1; color: {GREEN_DIM}; padding: 0 0 0 2; }}

    #transcript {{
        height: 1fr;
        width: 100%;
        padding: 1 2 0 2;
        scrollbar-background: {BG};
        scrollbar-color: {BORDER};
    }}

    .user       {{ color: {GREEN_GLOW}; margin-bottom: 1; }}
    .agent      {{ color: {TEXT}; margin-bottom: 1; }}
    .agent-head {{ color: {GREEN}; text-style: bold; }}
    .system     {{ color: {MUTED}; margin-bottom: 1; }}
    .error      {{ color: {ERROR}; margin-bottom: 1; }}

    #sidebar {{
        width: 46;
        min-width: 46;
        max-width: 46;
        height: 1fr;
        background: {BG};
        border-left: solid {BORDER};
        layout: vertical;
    }}
    #side-title {{
        height: 1;
        padding: 0 1;
        background: {BG};
        color: {GREEN};
        text-style: bold;
        border-bottom: solid {BORDER};
    }}
    #side-tabs {{
        height: 1;
        layout: horizontal;
        background: {BG};
        border-bottom: solid {BORDER};
    }}
    .side-tab {{
        width: 1fr;
        min-width: 8;
        height: 1;
        background: transparent;
        border: none;
        color: {DIM};
        text-style: bold;
    }}
    .side-tab:hover {{ color: {MUTED}; }}
    .side-tab.active {{ color: {GREEN}; }}

    #side-content {{
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
        background: {BG};
        padding: 1 0;
    }}

    #panel-llm, #panel-aws, #panel-context, #panel-help {{
        width: 100%;
        height: auto;
    }}

    .panel-title {{ color: {GREEN}; text-style: bold; margin: 0 0 1 0; }}
    .label {{ color: {MUTED}; margin-top: 1; }}

    Input {{
        background: {PANEL};
        border: round {BORDER};
        color: {TEXT};
        margin: 0 0 1 0;
        height: 3;
        width: 100%;
    }}
    Input:focus {{ border: round {GREEN}; }}

    Select {{
        background: {PANEL};
        border: round {BORDER};
        margin: 0 0 1 0;
        width: 100%;
    }}

    #sidebar Button {{
        width: 1fr;
        min-width: 12;
        background: transparent;
        border: round {GREEN};
        color: {GREEN};
        margin-top: 1;
        height: 3;
        text-style: bold;
    }}
    #sidebar Button:hover, #sidebar Button:focus {{
        background: {RAISED};
        color: {GREEN_GLOW};
        border: round {GREEN_GLOW};
    }}

    #prompt-wrap {{
        dock: bottom;
        height: auto;
        min-height: 5;
        background: {BG};
        border-left: thick {GREEN};
        padding: 1 0 0 2;
        margin: 0 0 0 2;
    }}
    #prompt-row {{ height: 3; layout: horizontal; }}
    #prompt-sign {{
        width: 2; height: 3;
        color: {GREEN};
        content-align: left middle;
        text-style: bold;
    }}
    #prompt {{
        width: 1fr;
        height: 3;
        background: {BG};
        border: none;
        color: {TEXT};
        padding: 0;
    }}
    #prompt:focus {{ border: none; }}
    #mode-chip {{
        height: 1;
        padding: 0 0 0 2;
        color: {MUTED};
        background: {BG};
    }}
    #hint {{
        height: 1;
        padding: 0 0 0 2;
        color: {DIM};
        background: {BG};
        margin-bottom: 1;
    }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",    "Quit",      priority=True),
        Binding("ctrl+d", "quit",    "Quit",      priority=True),
        Binding("ctrl+c", "cancel",  "Cancel",    priority=True),
        Binding("escape", "interrupt", "Interrupt Session", priority=False),
        Binding("ctrl+l", "clear",   "Clear"),
        Binding("tab",    "cycle_mode", "Mode",   priority=True),
    ]

    current_mode:   reactive[str]   = reactive("build")
    tokens_used:    reactive[int]   = reactive(0)
    cost:           reactive[float] = reactive(0.0)
    message_count:  reactive[int]   = reactive(0)
    context_window: reactive[int]   = reactive(200_000)

    def __init__(self, coordinator: Optional[AgentCoordinator] = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.coordinator = coordinator
        self.cfg = load_runtime_config()
        self.aws = get_aws_credentials()
        self.busy = False
        self._spinner_idx = 0
        self._spinner_on = False
        self._banner_phase = 0
        self._active_panel = "llm"
        self._agent_worker = None
        self._pending_permission: Optional["asyncio.Future[bool]"] = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="main-col"):
                with Vertical(id="banner-area"):
                    yield Static(self._banner_markup(), id="banner")
                    yield Static(
                        f"  [{GREEN_DIM}]✧  a  t  e  r  m  i  n  a  l  "
                        f"c  o  d  i  n  g  a  g  e  n  t  ✧[/]",
                        id="banner-sub",
                    )
                yield VerticalScroll(id="transcript")

            with Vertical(id="sidebar"):
                yield Static("▌ SIDEBAR", id="side-title")
                with Horizontal(id="side-tabs"):
                    yield Button("CONNECT", id="tab-llm",
                                 classes="side-tab active")
                    yield Button("AWS", id="tab-aws", classes="side-tab")
                    yield Button("CTX", id="tab-context", classes="side-tab")
                    yield Button("HELP", id="tab-help", classes="side-tab")
                with VerticalScroll(id="side-content"):
                    yield LLMPanel(self, id="panel-llm")
                    yield AWSPanel(self, id="panel-aws")
                    yield ContextPanel(self, id="panel-context")
                    yield HelpPanel(id="panel-help")

        with Vertical(id="prompt-wrap"):
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-sign")
                yield Input(placeholder="ask the agent…", id="prompt")
            yield Static(self._mode_chip(), id="mode-chip")
            yield Static(
                "tab mode  ·  /connect /aws /context /help",
                id="hint")

        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "matrix"
        self._show_panel("llm")
        self._wire_permission_confirmation()

        for w in self.query("#sidebar Button, .side-tab"):
            try:
                w.can_focus = False
            except Exception:
                pass

        self._system(
            "welcome. type a prompt to run the agent — "
            "/connect to set LLM, /aws for AWS, /context for tokens, /help for all commands."
        )
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.10, self._tick_spinner)
        self.set_interval(1.50, self._tick_banner)

        import os as _os
        if _os.environ.get("DEPRESSION_DEBUG_DUMP"):
            def _dump_debug():
                import asyncio as _aio
                with open(_os.environ.get("DEPRESSION_DEBUG_DUMP"), "w") as _f:
                    _f.write("FOCUS %r\n" % (self.focused,))
                    for _t in _aio.all_tasks():
                        if _t is _aio.current_task() or _t.done():
                            continue
                        _f.write(f"TASK {_t.get_name()!r}\n")
                        for _fr in _t.get_stack():
                            _f.write(f"  {_fr.filename}:{_fr.lineno} {_fr.name}\n")
                        _f.write("---\n")
            self.set_interval(2.0, _dump_debug)

    # ── permission wiring ─────────────────────────────────────────────
    def _wire_permission_confirmation(self) -> None:
        if not self.coordinator:
            return
        for agent in (getattr(self.coordinator, "plan_agent", None),
                      getattr(self.coordinator, "build_agent", None)):
            pm = getattr(agent, "permission_manager", None)
            if pm is not None:
                pm.set_confirm_callback(self._tui_confirm)
                pm.input_handler = None

    async def _tui_confirm(self, request: Any, verdict: Any) -> bool:
        """Show the permission modal, safe across threads.

        The PermissionManager may be awaited from a worker thread (tool
        execution usually runs via to_thread/run_in_executor). Pushing a
        modal from a worker thread paints it but never makes it the active
        screen on the main loop — so clicks and keys fall through to the
        main screen behind it. Marshal onto the main loop when needed.
        """
        if self._pending_permission is not None:
            return False

        loop = asyncio.get_event_loop()
        on_main = (threading.current_thread() is threading.main_thread())

        if not on_main:
            fut: "asyncio.Future[bool]" = loop.create_future()

            def _kick() -> None:
                asyncio.ensure_future(
                    self._open_modal_on_main(request, verdict, fut))

            loop.call_soon_threadsafe(_kick)
            try:
                return bool(await asyncio.wait_for(
                    asyncio.shield(fut), timeout=120.0))
            except asyncio.TimeoutError:
                self._system("permission request timed out — denied")
                return False

        fut = loop.create_future()
        return await self._open_modal_on_main(request, verdict, fut)

    async def _open_modal_on_main(self, request: Any, verdict: Any,
                                  fut: "asyncio.Future[bool]") -> bool:
        self._pending_permission = fut
        self._system(
            f"🔐 permission request — {request.tool}.{request.action} "
            "(A=allow · D=deny)")

        modal = PermissionModal(request, verdict)
        try:
            result = await self.push_screen_wait(modal)
            allowed = bool(result)
        except Exception:
            allowed = False
        finally:
            self._pending_permission = None
            if not fut.done():
                fut.set_result(allowed)
            self._refocus_prompt()

        return allowed

    def _refocus_prompt(self) -> None:
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    def _banner_markup(self) -> str:
        greens = [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM]
        return "\n".join(f"[bold {greens[i]}]{line}[/]"
                         for i, line in enumerate(BANNER_LINES))

    def _mode_chip(self) -> str:
        icon = MODE_ICONS.get(self.current_mode, "◆")
        model = self.cfg.get("model") or "no model selected"
        prefix = ""
        if self._spinner_on:
            spin = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            prefix = f"[{AMBER}]{spin}[/] "
        pct = (self.tokens_used / max(1, self.context_window)) * 100.0
        ctx = f"[{MUTED}]ctx {pct:.0f}%[/]"
        return (f"{prefix}[{GREEN}]{icon} {self.current_mode.upper()}[/]  "
                f"[{DIM}]·[/]  [{TEXT}]{model}[/]  "
                f"[{DIM}]·[/]  {ctx}")

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

    def _show_panel(self, which: str) -> None:
        self._active_panel = which
        for name in ("llm", "aws", "context", "help"):
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

    @on(Button.Pressed, ".side-tab")
    def _on_tab(self, event: Button.Pressed) -> None:
        self._show_panel(event.button.id.replace("tab-", ""))
        self._refocus_prompt()

    def _tick_spinner(self) -> None:
        if not self._spinner_on:
            return
        self._spinner_idx += 1
        self._refresh_mode_chip()

    def _tick_banner(self) -> None:
        self._banner_phase = (self._banner_phase + 1) % 3
        greens = [
            [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM],
            [GREEN, GREEN_GLOW, GREEN, GREEN_DIM, GREEN, GREEN_DIM],
            [GREEN, GREEN, GREEN_GLOW, GREEN_DIM, GREEN_DIM, GREEN],
        ][self._banner_phase]
        rows = "\n".join(f"[bold {greens[i]}]{line}[/]"
                         for i, line in enumerate(BANNER_LINES))
        try:
            self.query_one("#banner", Static).update(rows)
        except Exception:
            pass

    def _write(self, text: str, cls: str = "agent") -> None:
        self.query_one("#transcript", VerticalScroll).mount(
            Static(text, classes=cls))
        self.call_after_refresh(
            lambda: self.query_one("#transcript", VerticalScroll)
                        .scroll_end(animate=False))

    def _system(self, text: str) -> None:
        self._write(f"· {text}", "system")

    def _error(self, text: str) -> None:
        self._write(f"× {text}", "error")

    def _user(self, text: str) -> None:
        self._write(f"[bold {GREEN}]›[/] [bold {TEXT}]{text}[/]", "user")
        self.message_count = self.message_count + 1

    def _agent_head(self) -> None:
        self._write(f"[bold {GREEN}]◆ depression.ai[/]", "agent-head")

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.busy and not text.startswith("/"):
            self._error("agent is busy — press Esc to interrupt, then resubmit")
            return
        event.input.value = ""
        if text.startswith("/"):
            self._slash(text)
            return
        self._agent_worker = self._run_agent(text)

    def _slash(self, text: str) -> None:
        command = text.split()[0].lower()
        if command == "/aws":
            self._show_panel("aws")
            self.query_one("#aws-key", Input).focus()
        elif command == "/connect":
            self._show_panel("llm")
            self.query_one("#base-url", Input).focus()
        elif command in ("/models", "/model"):
            self._show_panel("llm")
            self._discover_models()
        elif command == "/context":
            self._show_panel("context")
        elif command == "/help":
            self._show_panel("help")
        elif command == "/clear":
            self.query_one("#transcript", VerticalScroll).remove_children()
        elif command == "/plan":
            self.current_mode = "plan"
            if self.coordinator:
                self.coordinator.set_mode("plan")
            self._refresh_mode_chip()
            self._system("mode → plan")
        elif command == "/build":
            self.current_mode = "build"
            if self.coordinator:
                self.coordinator.set_mode("build")
            self._refresh_mode_chip()
            self._system("mode → build")
        elif command == "/auto":
            self.current_mode = "auto"
            if self.coordinator:
                self.coordinator.set_mode("auto")
            self._refresh_mode_chip()
            self._system("mode → auto")
        elif command in ("/quit", "/exit"):
            self.exit()
        else:
            self._error(f"unknown command: {command}")

    @work(exclusive=True, group="agent")
    async def _run_agent(self, text: str) -> None:
        self.busy = True
        self._spinner_on = True
        self._refresh_mode_chip()
        self._user(text)

        if not self.coordinator:
            self._error("agent coordinator is not initialized")
            self.busy = False
            self._spinner_on = False
            self._refresh_mode_chip()
            return

        try:
            result = await asyncio.wait_for(
                self.coordinator.process_query(
                    text, mode=self.current_mode,
                    auto_execute=True),
                timeout=300,
            )
            if result.get("success"):
                output = (result.get("execution")
                          or result.get("response")
                          or "task completed.")
                self._agent_head()
                self._write(str(output), "agent")

                tokens = (result.get("tokens") or result.get("total_tokens")
                          or result.get("context", {}).get("tokens_used"))
                if tokens:
                    self.tokens_used = int(tokens)
                cost = (result.get("cost")
                        or result.get("context", {}).get("cost"))
                if cost is not None:
                    try:
                        self.cost = float(cost)
                    except Exception:
                        pass
                self._refresh_context_panel()
                self._refresh_mode_chip()
            else:
                self._error(str(result.get("error", "agent request failed")))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error(f"agent error: {exc}")
        finally:
            self.busy = False
            self._spinner_on = False
            self._refresh_mode_chip()
            self._refocus_prompt()

    @on(Button.Pressed, "#connect")
    def connect_llm(self) -> None:
        base_url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        api_key = (self.cfg["api_key"]
                   if key_input.value == "••••••••"
                   else key_input.value.strip())
        model = self.query_one("#model-id", Input).value.strip()
        if not base_url or not api_key or not model:
            self.query_one("#llm-status", Static).update(
                "base URL, API key and model are required")
            return
        try:
            configure_runtime_provider(get_llm_registry(), base_url, api_key, model)
            self.cfg = load_runtime_config()
            self.query_one("#llm-status", Static).update(f"connected · {model}")
            self._refresh_mode_chip()
            self._refresh_context_panel()
            self._system(f"LLM connected: {base_url} · {model}")
        except Exception as exc:
            self.query_one("#llm-status", Static).update(f"error: {exc}")

    @on(Button.Pressed, "#discover")
    def discover_llm(self) -> None:
        self._discover_models()

    @work(exclusive=True, group="discover")
    async def _discover_models(self) -> None:
        url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        key = (self.cfg["api_key"]
               if key_input.value == "••••••••"
               else key_input.value.strip())
        if not url:
            self.query_one("#llm-status", Static).update("enter a base URL first")
            return
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {key}"} if key else {})
                response.raise_for_status()
                models = response.json().get("data", [])
            ids = [str(m.get("id")) for m in models if m.get("id")]
            if ids:
                self.query_one("#model-id", Input).value = ids[0]
                self.query_one("#llm-status", Static).update(
                    f"discovered {len(ids)} model(s); selected {ids[0]}")
                self._system("models: " + ", ".join(ids[:12]))
            else:
                self.query_one("#llm-status", Static).update(
                    "/models returned no model IDs")
        except Exception as exc:
            self.query_one("#llm-status", Static).update(
                f"discovery failed: {exc}; enter model manually")

    @on(Button.Pressed, "#aws-save")
    def save_aws(self) -> None:
        key = self.query_one("#aws-key", Input).value.strip()
        secret_input = self.query_one("#aws-secret", Input)
        secret = (self.aws.get("secret_key") or ""
                  if secret_input.value == "••••••••"
                  else secret_input.value.strip())
        region = str(self.query_one("#aws-region", Select).value)
        if not key or not secret:
            self.query_one("#aws-status", Static).update(
                "access key and secret are required")
            return
        try:
            set_aws_credentials(key, secret, region)
            self.aws = get_aws_credentials()
            self.query_one("#aws-status", Static).update(
                f"saved to .env · {region}")
            self._system(
                f"AWS credentials saved to .env ({region}); secret is not displayed.")
        except Exception as exc:
            self.query_one("#aws-status", Static).update(f"error: {exc}")

    def is_agent_busy(self) -> bool:
        if self.busy:
            return True
        worker = self._agent_worker
        return worker is not None and getattr(worker, "is_running", False)

    def cancel_current_agent(self) -> None:
        worker = self._agent_worker
        if worker is not None and getattr(worker, "is_running", False):
            worker.cancel()
        self.busy = False
        self._spinner_on = False
        self._refresh_mode_chip()

    def action_interrupt(self) -> None:
        if isinstance(self.screen, PermissionModal):
            self.screen.action_deny()
            return
        worker = self._agent_worker
        if worker is not None and getattr(worker, "is_running", False):
            worker.cancel()
            self._system("interrupted by ESC")
        self.busy = False
        self._spinner_on = False
        self._refresh_mode_chip()

    def action_cancel(self) -> None:
        if isinstance(self.screen, PermissionModal):
            self.screen.action_deny()
            return
        if self.busy:
            self.action_interrupt()
            self._system("cancelled — press Ctrl+C again to quit")
            return
        self.exit()

    def action_clear(self) -> None:
        self.query_one("#transcript", VerticalScroll).remove_children()

    def action_cycle_mode(self) -> None:
        i = (MODE_ORDER.index(self.current_mode)
             if self.current_mode in MODE_ORDER else 1)
        self.current_mode = MODE_ORDER[(i + 1) % len(MODE_ORDER)]
        if self.coordinator:
            try:
                self.coordinator.set_mode(self.current_mode)
            except Exception:
                pass
        self._refresh_mode_chip()


__all__ = ["DepressionApp"]