"""Terminal-native agentic interface."""
from __future__ import annotations

import asyncio
import difflib
import importlib.util
import json
import os as _os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from agent.agent.dual_agent import AgentCoordinator
from agent.llm.runtime import (
    configure_runtime_provider,
    get_llm_registry,
    load_runtime_config,
    plan_runtime,
)
from agent.utils.env_manager import (
    get_aws_credentials,
)
from agent.utils.shell import is_long_running, run_command

from agent.tui.events import AgentEvent, EventBridge
from agent.tui.theme import (
    GREEN, GREEN_DIM, GREEN_GLOW, AMBER, ERROR,
    TEXT, MUTED, DIM, BG, PANEL, RAISED, BORDER,
    MATRIX_THEME,
)
from agent.tui.widgets.aws_panel import AWSPanel
from agent.tui.widgets.empty_banner import EmptyBanner
from agent.tui.widgets.output_view import OutputView
from agent.tui.widgets.plan_view import PlanView
from agent.tui.widgets.profile_panel import ProfilePanel
from agent.tui.widgets.sidebar import Sidebar, SidebarTab
from agent.tui.widgets.mcp_panel import MCPPanel
from agent.tui.widgets.thinking import ThinkingIndicator
from agent.tui.widgets.tool_call import ToolCallWidget


# UI component classes, their CSS, and command utilities are unchanged.
from agent.tui.app_components import (
    SPINNER_FRAMES,
    MODE_ICONS,
    MODE_ORDER,
    DEFAULT_TURN_TIMEOUT,
    DEFAULT_AWS_REGION,
    _PANEL_NAMES,
    _DIFF_TOOLS,
    _PROMPT_MIN_ROWS,
    _PROMPT_MAX_ROWS,
    _STOP_REQUESTS,
    _is_stop_request,
    _esc,
    _fmt,
    _shorten_path,
    _get,
    _open_file_with_default_app,
    StreamingOutput,
    PromptArea,
    TextPromptModal,
    ConfirmModal,
    _PickerModal,
    HistorySearchModal,
    FilePickerModal,
    SessionPickerModal,
    TranscriptSearchModal,
    AgentCommandProvider,
    _SLASH_ARGUMENT_COMMANDS,
    _SLASH_SUGGESTION_LIMIT,
    _slash_command_matches,
    LLMPanel,
    ContextPanel,
    TodoPanel,
    _probe_loop_context,
    _usage_empty,
    _usage_float,
    _usage_int,
    _usage_snapshot,
    _usage_add,
    _usage_agents,
    _query_usage,
    _session_usage,
    _query_elapsed,
    LLMCostPanel,
    HelpPanel,
    PermissionModal,
)


# ======================================================================
# APP
# ======================================================================

class DepressionApp(App):
    TITLE = "depression.ai"

    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}

    #body {{ height: 1fr; layout: horizontal; }}
    #main-col {{ width: 1fr; height: 1fr; layout: vertical; }}

    #transcript-wrap {{
        height: 1fr;
        width: 100%;
        layout: vertical;
    }}
    #transcript {{
        height: 1fr;
        width: 100%;
        padding: 1 2 0 2;
        scrollbar-background: {BG};
        scrollbar-color: {BORDER};
    }}
    #empty-banner {{
        height: 1fr;
        width: 100%;
        display: block;
    }}

    .user       {{ color: {GREEN_GLOW}; margin-bottom: 1; }}
    .agent      {{ color: {TEXT}; margin-bottom: 1; }}
    .agent-head {{ color: {GREEN}; text-style: bold; }}
    .system     {{ color: {MUTED}; margin-bottom: 1; }}
    .error      {{ color: {ERROR}; margin-bottom: 1; }}
    .queued     {{ color: {AMBER}; margin-bottom: 1; }}

    #prompt-wrap {{
        dock: bottom; height: auto;
        background: {BG};
        border-left: thick {GREEN};
        padding: 1 0 0 2; margin: 0 0 0 2;
    }}
    #prompt-row {{ height: auto; layout: horizontal; }}
    #command-menu {{
        display: none;
        height: auto;
        max-height: 10;
        border: round {BORDER};
        background: {PANEL};
        padding: 0 1;
        margin: 0 1 1 0;
    }}
    #command-menu-title {{
        height: 1;
        color: {GREEN};
        text-style: bold;
    }}
    #command-options {{
        height: auto;
        max-height: 7;
        background: {PANEL};
        color: {TEXT};
        border: none;
        scrollbar-size-vertical: 1;
    }}
    #command-options > .option-list--option-highlighted {{
        background: {GREEN_DIM};
        color: {BG};
        text-style: bold;
    }}
    #prompt-sign {{
        width: 2; height: auto;
        color: {GREEN};
        content-align: left top; text-style: bold;
    }}
    #prompt {{
        width: 1fr;
        height: auto;
        background: {BG}; border: none; color: {TEXT}; padding: 0;
    }}
    #prompt:focus {{ border: none; }}
    #prompt.busy {{ color: {AMBER}; }}
    #mode-chip {{ height: 1; padding: 0 0 0 2; color: {MUTED}; background: {BG}; }}
    #hint {{ height: 1; padding: 0 0 0 2; color: {DIM}; background: {BG}; }}
    #token-bar {{ height: 1; padding: 0 0 0 2; background: {BG}; }}
    #thinking-bar {{ height: 1; background: {BG}; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+c", "cancel", "Cancel", priority=True),
        Binding("escape", "interrupt", "Interrupt", priority=False),
        Binding("ctrl+l", "clear", "Clear", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+r", "history_search", "History", priority=True),
        Binding("ctrl+f", "search_transcript", "Search", priority=True),
        Binding("ctrl+space", "complete_file", "Files", priority=True),
        Binding("f2", "toggle_auto_scroll", "Follow", priority=True),
        Binding("tab", "tab_or_mode", "Mode", priority=True),
        Binding("ctrl+e", "toggle_expand_tool", "Expand", priority=False),
    ]

    COMMANDS = {AgentCommandProvider}

    current_mode: reactive[str] = reactive("build")
    tokens_used: reactive[int] = reactive(0)
    cost: reactive[float] = reactive(0.0)
    message_count: reactive[int] = reactive(0)
    context_window: reactive[int] = reactive(1_000_000)
    queue_depth: reactive[int] = reactive(0)

    live_input_tokens: reactive[int] = reactive(0)
    live_output_tokens: reactive[int] = reactive(0)
    live_total_tokens: reactive[int] = reactive(0)

    # [Spinner fix] Tick a dedicated reactive so the spinner animates even
    # when no messages are being written.
    spinner_tick: reactive[int] = reactive(0)

    def __init__(
        self,
        coordinator: Optional[AgentCoordinator] = None,
        skip_onboarding: bool = False,
        session_manager: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.coordinator = coordinator
        self.skip_onboarding = skip_onboarding
        self.session_manager = session_manager
        self.cfg = load_runtime_config()
        self.aws = get_aws_credentials()

        self._user_name: str = ""
        self._prompt_history: List[str] = []
        self._history_index = -1
        self._history_draft = ""
        self._command_menu_open = False
        self._command_suggestion_matches: List[tuple[str, str]] = []
        self._command_suggestion_index = 0
        self._command_suppressed_text: Optional[str] = None
        self._last_prompt = ""
        self._last_output = ""
        self._transcript_log: List[Dict[str, Any]] = []
        self._auto_scroll = True
        self._search_cursor = 0
        self._session_allowed_tools: set[str] = set()
        self._turn_started = 0.0

        self.busy = False
        self._active_panel = "llm"
        self._agent_worker = None
        self._user_shell_worker = None
        self._prompt_queue: list[str] = []
        self._draining = False
        self._is_shutting_down = False
        self._has_messages = False

        self._plan_view: Optional[PlanView] = None
        self._plan_signature: Optional[tuple] = None
        self._streaming_widget: Optional[StreamingOutput] = None
        self._stream_lock: Optional[asyncio.Lock] = None

        try:
            self.turn_timeout = float(
                _os.environ.get("TUI_TURN_TIMEOUT", DEFAULT_TURN_TIMEOUT)
            )
        except Exception:
            self.turn_timeout = DEFAULT_TURN_TIMEOUT

        self._events = EventBridge()
        self._event_handlers_registered = False
        self._active_tool_widget: Optional[ToolCallWidget] = None

        self._mounted = False

    # ------------------------------------------------------------------
    # COMPOSE
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="main-col"):
                with Vertical(id="transcript-wrap"):
                    yield EmptyBanner(id="empty-banner")
                    yield VerticalScroll(id="transcript")
                yield ThinkingIndicator(id="thinking-bar")

            yield Sidebar(
                panels=[
                    ("llm", LLMPanel(self, id="panel-llm")),
                    ("aws", AWSPanel(self, id="panel-aws")),
                    ("mcp", MCPPanel(self, id="panel-mcp")),
                    ("profile", ProfilePanel(id="panel-profile")),
                    ("context", ContextPanel(self, id="panel-context")),
                    ("llmcost", LLMCostPanel(self, id="panel-llmcost")),
                    ("todo", TodoPanel(self, id="panel-todo")),
                    ("help", HelpPanel(id="panel-help")),
                ],
                id="sidebar",
            )

        with Vertical(id="prompt-wrap"):
            with Vertical(id="command-menu"):
                yield Static(
                    "COMMANDS  ·  ↑↓ select  ·  Tab insert  ·  Enter run  ·  Esc close",
                    id="command-menu-title",
                    markup=False,
                )
                yield OptionList(id="command-options")
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-sign")
                yield PromptArea(id="prompt")
            yield Static(self._mode_chip(), id="mode-chip", markup=True)
            yield Static(
                f"[{DIM}]enter submit · shift+enter newline · ctrl+p commands · "
                f"type / for commands · ctrl+r history · ctrl+space @files · "
                f"f2 follow · ctrl+b sidebar · ctrl+q quit[/]",
                id="hint", markup=True,
            )
            yield Static(self._token_bar(), id="token-bar", markup=True)

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

        self._set_active_panel("llm")

        # Compact sidebar labels are keyboard-accessible; do not disable focus.
        # Editable LLM/MCP/AWS controls remain focusable as before.

        try:
            self.query_one("#transcript", VerticalScroll).display = False
        except Exception:
            pass

        self._wire_events()

        self._load_identity_from_env()

        try:
            self.query_one("#prompt", PromptArea).focus()
        except Exception:
            pass

        self._refresh_aws_status()
        self._refresh_token_bar()
        self._refresh_llmcost_panel()
        self._refresh_setup_status()
        self._apply_sidebar_layout()

        # [Spinner fix] Animate the mode chip independently of message_count.
        self.set_interval(0.25, self._tick_spinner)

        self._mounted = True

        self.run_worker(self._maybe_onboard(), group="onboarding")

    def on_resize(self, event: Any) -> None:
        self._apply_sidebar_layout()

    def _apply_sidebar_layout(self) -> None:
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.apply_responsive(self.size.width, self.size.height)
        except Exception:
            pass

    def _tick_spinner(self) -> None:
        if not self.busy:
            return
        self.spinner_tick = (self.spinner_tick + 1) % 10_000
        self._refresh_mode_chip()

    def _load_identity_from_env(self) -> None:
        try:
            name = (
                _os.environ.get("DEPRESSION_USER_NAME")
                or _os.environ.get("DEPRESSION_USER_EMAIL")
                or ""
            )
        except Exception:
            name = ""
        if name:
            self._user_name = name
            try:
                self.sub_title = name
            except Exception:
                pass
            self._refresh_mode_chip()
            self._refresh_banner_welcome()
            self._refresh_setup_status()

    def _refresh_banner_welcome(self) -> None:
        try:
            banner = self.query_one("#empty-banner", EmptyBanner)
            banner.set_welcome(self._user_name)
        except Exception:
            pass

    def _is_real_user_profile(self, profile: Any) -> bool:
        """Check if profile is a real signed-in user (not placeholder/guest)."""
        try:
            if not profile or not getattr(profile, "signed_in", False):
                return False
            if getattr(profile, "metadata", {}).get("local"):
                return False
            uid = getattr(profile, "uid", "") or ""
            email = getattr(profile, "email", "") or ""
            name = getattr(profile, "name", "") or ""
            # Detect placeholder/fake profiles
            if uid.startswith("firebase-uid-") or uid == "local-guest":
                return False
            if email in ("user@gmail.com", "") or name in ("Real User", "Local user", ""):
                return False
            # Real profile should have tokens saved
            from agent.tui.onboarding.profile import ProfileStore
            store = ProfileStore()
            tokens = store.load_tokens()
            if not tokens or not tokens.get("refresh_token"):
                return False
            return True
        except Exception:
            return False

    async def _maybe_onboard(self) -> None:
        if self.skip_onboarding:
            from agent.tui.onboarding.profile import guest_profile, ProfileStore
            store = ProfileStore()
            if store.load() is None:
                profile = guest_profile()
                store.save(profile)
                self._sync_identity(profile)
            else:
                profile = store.load()
                if profile:
                    self._sync_identity(profile)
            return

        # Let the onboarding gate decide (it also honors
        # DEPRESSION_ONBOARDING=force/reset). It returns the stored profile for
        # returning users, so identity still gets synced on every launch.
        from agent.tui.onboarding.welcome import push_welcome_if_new_user

        prompt = None
        try:
            prompt = self.query_one("#prompt", PromptArea)
            prompt.disabled = True
        except Exception:
            prompt = None

        try:
            profile = await push_welcome_if_new_user(
                self, context_manager=self._context_manager()
            )
            if profile is not None:
                self._sync_identity(profile)
        finally:
            if prompt is not None:
                try:
                    prompt.disabled = False
                except Exception:
                    pass
            self._refocus_prompt()

    def _context_manager(self) -> Any:
        try:
            coordinator = getattr(self, "coordinator", None)
            for attr in ("plan_agent", "build_agent"):
                agent = getattr(coordinator, attr, None)
                context_manager = getattr(agent, "context_manager", None)
                if context_manager is not None:
                    return context_manager
        except Exception:
            pass
        return None

    def _sync_identity(self, profile: Any) -> None:
        name = ""
        try:
            name = getattr(profile, "display_name", "") or ""
        except Exception:
            name = ""
        if not name:
            name = (
                _os.environ.get("DEPRESSION_USER_NAME")
                or _os.environ.get("DEPRESSION_USER_EMAIL")
                or ""
            )
        self._user_name = name

        try:
            self.sub_title = name or "depression.ai"
        except Exception:
            pass

        try:
            self._refresh_aws_status()
        except Exception:
            pass
        try:
            self._refresh_llmcost_panel()
        except Exception:
            pass
        try:
            self._refresh_mode_chip()
        except Exception:
            pass
        self._refresh_banner_welcome()
        self._refresh_setup_status()
        try:
            self.query_one("#panel-profile", ProfilePanel).refresh_profile()
        except Exception:
            pass

    def _wire_events(self) -> None:
        if self._event_handlers_registered:
            return
        if self.coordinator is None:
            return
        try:
            self._events.set_ui_loop(asyncio.get_running_loop())
            self._events.attach(self.coordinator)
            self._events.subscribe("on_tool_executed", self._on_tool_event)
            self._events.subscribe("on_plan_updated", self._on_plan_updated)
            self._events.subscribe("on_plan_created", self._on_plan_created)
            self._events.subscribe("on_plan_rendered", self._on_plan_rendered)
            self._events.subscribe("on_plan_task_complete", self._on_plan_task_complete)
            self._events.subscribe("on_plan_task_failed", self._on_plan_task_failed)
            # [Streaming] Live model output.
            self._events.subscribe("on_token", self._on_token)
            self._events.subscribe("on_token_end", self._on_token_end)
            self._event_handlers_registered = True
        except Exception as exc:
            self._error(f"event wiring failed: {exc}")

    # ------------------------------------------------------------------
    # CHAT BAR FOCUS SAFETY
    # ------------------------------------------------------------------

    def _refocus_prompt(self) -> None:
        def _do() -> None:
            try:
                prompt = self.query_one("#prompt", PromptArea)
                if not prompt.disabled and not prompt.has_focus:
                    prompt.focus()
            except Exception:
                pass

        try:
            self.call_after_refresh(_do)
            self.set_timer(0.05, _do)
        except Exception:
            _do()

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        # Modals own their keys; app shortcuts must not fire behind them.
        if isinstance(self.screen, ModalScreen) and action in {
            "tab_or_mode",
            "toggle_expand_tool",
            "toggle_sidebar",
            "clear",
        }:
            return False
        # Tab must navigate editable sidebar controls, not switch the
        # agent's mode or accept a slash suggestion behind their back.
        if action == "tab_or_mode":
            focused = self.focused
            if focused is not None and focused is not self.query_one("#prompt", PromptArea):
                if isinstance(focused, (Input, TextArea, Select)):
                    return False
        return True

    def on_key(self, event) -> None:
        if not self._mounted:
            return
        try:
            if isinstance(self.screen, PermissionModal):
                return
            if self._is_shutting_down:
                return
            prompt = self.query_one("#prompt", PromptArea)
            if prompt.disabled or prompt.has_focus:
                return
            # A sidebar connection field owns its keystrokes and paste.
            # Previously the app moved printable keys back to the main
            # prompt whenever that prompt was not focused.
            focused = self.focused
            if focused is not None and isinstance(
                focused, (Input, TextArea, Select, Button, OptionList)
            ):
                return

            key = event.key
            if key is None:
                return

            if len(key) == 1 and key.isprintable() and key != " ":
                if key in self.active_bindings:
                    return
                prompt.focus()
                prompt.insert(key)
                event.stop()
                return
            if key == "space":
                prompt.focus()
                prompt.insert(" ")
                event.stop()
                return
            if key in ("backspace", "delete", "left", "right", "home", "end",
                       "up", "down"):
                prompt.focus()
                event.stop()
                return
        except Exception:
            pass

    # ------------------------------------------------------------------
    # STREAMING
    # ------------------------------------------------------------------

    async def _on_token(self, event: AgentEvent) -> None:
        # [Streaming fix] Do NOT gate on self.is_running. Tokens can arrive
        # while the app is still wiring up its first frame. The only valid
        # gate is a shutdown check.
        if self._is_shutting_down:
            return

        data = event.payload or {}
        delta = data.get("delta") or ""
        if not delta:
            return

        self._show_transcript()

        # Ensure a streaming widget exists. Tokens can arrive as concurrent
        # tasks, so guard the check/create/mount sequence with a lock.
        if self._stream_lock is None:
            self._stream_lock = asyncio.Lock()
        async with self._stream_lock:
            if (
                self._streaming_widget is None
                or not self._streaming_widget.is_mounted
            ):
                try:
                    transcript = self.query_one("#transcript", VerticalScroll)
                except Exception:
                    return
                widget = StreamingOutput()
                try:
                    await transcript.mount(widget)
                except Exception:
                    return
                self._streaming_widget = widget

        # append_token batches repaints and scrolls, so this stays cheap even
        # when the provider streams tokens faster than the UI can render them.
        self._streaming_widget.append_token(delta)

    async def _on_token_end(self, event: AgentEvent) -> None:
        # [Streaming fix] Same rationale as _on_token — no is_running gate.
        # The structured OutputView will replace the streaming widget in
        # _render_output once the turn's result comes back.
        if self._is_shutting_down:
            return
        widget = self._streaming_widget
        if widget is not None and widget.is_mounted:
            widget.flush()

    # ------------------------------------------------------------------
    # AGENT EVENTS
    # ------------------------------------------------------------------

    async def _on_tool_event(self, event: AgentEvent) -> None:
        # [UI fix] Do not gate on self.is_running — the first tool event can
        # arrive before the app fully settles, and dropping it loses UI state.
        if self._is_shutting_down:
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

        self._show_transcript()

        widget = ToolCallWidget(tool=tool, params=params)
        await transcript.mount(widget)
        widget.set_running(execution_time=duration)

        if isinstance(result, dict) and tool in _DIFF_TOOLS:
            result = self._shape_diff_payload(tool, params, result)

        widget.set_result(
            result if isinstance(result, dict) else {"success": True, "result": result},
            execution_time=duration,
            cached=cached,
        )
        self._active_tool_widget = widget
        self._scroll_transcript_end()

        self._sync_tokens_from_loop()
        self._refresh_llmcost_panel()

        if tool in ("todo", "todowrite", "todoread"):
            self._refresh_todo_from_store(force=True)

        if tool == "mcp" or tool.startswith("mcp__"):
            self._refresh_mcp_status()

        if tool in ("aws",) or tool.startswith("mcp__aws__"):
            try:
                self.query_one("#panel-aws", AWSPanel).refresh_mcp_status()
            except Exception:
                pass

        try:
            self._refresh_context_panel()
        except Exception:
            pass

    @staticmethod
    def _shape_diff_payload(tool: str, params: dict, result: dict) -> dict:
        """Normalize supplied snapshots without inventing file changes.

        Read/list results often include 'content' or 'output'. Such values
        cannot be interpreted as a post-edit snapshot without a real before.
        """
        r = dict(result)
        if tool == "filesystem" and params.get("action") not in (
            "write", "append", "edit", "delete",
        ):
            return r
        if isinstance(r.get("results"), list):
            # Multi-file patch snapshots live in individual result entries.
            return r
        before = next((
            r[key] for key in ("before", "content_before", "old_content")
            if isinstance(r.get(key), str)
        ), None)
        after = next((
            r[key] for key in ("after", "content_after", "new_content")
            if isinstance(r.get(key), str)
        ), None)
        if before is not None and after is not None:
            r["before"] = before
            r["after"] = after
            r["diff_source"] = "tool"
        else:
            r["diff_source"] = "tool:none"
        return r

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------

    async def _on_plan_created(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        data = event.payload or {}
        tasks = data.get("tasks") or []
        if not tasks:
            return

        entries = []
        for idx, t in enumerate(tasks):
            if not isinstance(t, dict):
                continue
            desc = t.get("description") or t.get("content") or ""
            if not desc:
                continue
            entries.append({
                "content": str(desc)[:200],
                "status": "in_progress" if idx == 0 else "pending",
                "priority": t.get("priority"),
            })

        if not entries:
            return

        self._show_transcript()
        await self._upsert_plan(entries)
        self._update_todo_panel(entries)

    async def _on_plan_updated(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        data = event.payload or {}
        entries = data.get("entries") or []
        render_inline = bool(data.get("render", True))
        if not entries:
            return

        normalised = self._normalise_plan_entries(entries)
        if not normalised:
            return

        self._update_todo_panel(normalised)

        if not render_inline:
            return
        self._show_transcript()
        await self._upsert_plan(normalised)

    async def _on_plan_rendered(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        data = event.payload or {}
        entries = data.get("entries") or []
        if not entries:
            return
        normalised = self._normalise_plan_entries(entries)
        if not normalised:
            return
        self._update_todo_panel(normalised)

    async def _on_plan_task_complete(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        self._refresh_todo_from_store(force=True)

    async def _on_plan_task_failed(self, event: AgentEvent) -> None:
        if self._is_shutting_down:
            return
        data = event.payload or {}
        task_id = str(data.get("task_id") or "")
        if task_id:
            try:
                from agent.tools.todo import TodoTool
                session = self._session()
                sid = TodoTool._session_key(session)
                store = TodoTool._strong_keys.get(sid) or {}
                item = store.get(task_id)
                if item is not None:
                    item.status = "failed"
            except Exception:
                pass
        self._refresh_todo_from_store(force=True)

    def _refresh_todo_from_store(self, force: bool = False) -> None:
        try:
            self.query_one("#panel-todo", TodoPanel).refresh_from_store(force=force)
        except Exception:
            pass

    @staticmethod
    def _normalise_plan_entries(entries: list) -> list:
        out = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            status = str(e.get("status") or "pending").lower()
            if status == "done":
                status = "completed"
            content = (
                e.get("content") or e.get("description") or e.get("title") or ""
            )
            content = str(content)[:200]
            if not content:
                continue
            out.append({
                "content": content,
                "status": status,
                "priority": e.get("priority"),
            })
        return out

    def _update_todo_panel(self, entries: list) -> None:
        try:
            self.query_one("#panel-todo", TodoPanel).set_entries(entries)
        except Exception:
            pass

    async def _upsert_plan(self, normalised: list) -> None:
        sig = tuple((e["content"][:80], e["status"]) for e in normalised)
        if sig == self._plan_signature:
            return
        self._plan_signature = sig

        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return

        if self._plan_view is None:
            try:
                self._plan_view = PlanView(entries=normalised)
                await transcript.mount(self._plan_view)
            except Exception:
                self._plan_view = None
                try:
                    await transcript.mount(
                        Static(self._render_plan_fallback(normalised),
                               classes="agent", markup=True)
                    )
                except Exception:
                    pass
        else:
            try:
                self._plan_view.set_entries(normalised)
            except Exception:
                pass

        self._scroll_transcript_end()

    @staticmethod
    def _render_plan_fallback(entries: list) -> str:
        icons = {
            "completed": f"[{GREEN}]✓[/]",
            "in_progress": f"[{AMBER}]▸[/]",
            "pending": f"[{DIM}]○[/]",
            "failed": f"[{ERROR}]✗[/]",
            "skipped": f"[{DIM}]—[/]",
        }
        lines = [f"[bold {GREEN}]◇ PLAN[/]"]
        for e in entries:
            icon = icons.get(e["status"], f"[{DIM}]○[/]")
            content = _esc(e["content"])
            if e["status"] == "completed":
                lines.append(f"  {icon} [{MUTED}]{content}[/]")
            elif e["status"] == "in_progress":
                lines.append(f"  {icon} [bold {TEXT}]{content}[/]")
            else:
                lines.append(f"  {icon} [{TEXT}]{content}[/]")
        return "\n".join(lines)

    def _plan_from_text(self, text: str) -> list:
        try:
            from agent.tui.onboarding.plan_parser import (
                parse_plan_from_text, looks_like_plan,
            )
        except Exception:
            return []
        if not looks_like_plan(text):
            return []
        plan = parse_plan_from_text(text)
        if plan is None or not plan.is_valid:
            return []
        return plan.to_list()

    # ------------------------------------------------------------------
    # TOKEN COUNTER
    # ------------------------------------------------------------------

    def _sync_tokens_from_loop(self) -> None:
        # [Token-count fix] Use the shared probe so we always find the
        # live loop context if one exists.
        ctx = _probe_loop_context(self.coordinator)
        if ctx is None:
            return
        try:
            self.live_input_tokens = int(
                _get(ctx, "input_tokens", "prompt_tokens", default=0) or 0
            )
            self.live_output_tokens = int(
                _get(ctx, "output_tokens", "completion_tokens", default=0) or 0
            )
            total = int(
                _get(ctx, "tokens_used", "total_tokens", default=0) or 0
            )
            if not total:
                total = self.live_input_tokens + self.live_output_tokens
            self.live_total_tokens = total

            # Also keep the app-level reactive counters in sync so the
            # ContextPanel and the token bar agree.
           # if total:
           #    self.tokens_used = total
            #cost = float(_get(ctx, "cost", default=0.0) or 0.0)
           # if cost:
            #    self.cost = cost
        except Exception:
            pass
        self._refresh_token_bar()
        self._refresh_context_panel()

    def _token_bar(self) -> str:
        ti = self.live_input_tokens
        to = self.live_output_tokens
        tt = self.live_total_tokens or (ti + to)
        return (
            f"[{DIM}]tokens[/]  "
            f"[{MUTED}]in[/] [{TEXT}]{_fmt(ti)}[/]  "
            f"[{DIM}]·[/]  "
            f"[{MUTED}]out[/] [{TEXT}]{_fmt(to)}[/]  "
            f"[{DIM}]·[/]  "
            f"[{MUTED}]total[/] [{GREEN}]{_fmt(tt)}[/]"
        )

    def _refresh_token_bar(self) -> None:
        try:
            self.query_one("#token-bar", Static).update(self._token_bar())
        except Exception:
            pass

    def _refresh_llmcost_panel(self) -> None:
        try:
            self.query_one("#panel-llmcost", LLMCostPanel).refresh_values()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # TRANSCRIPT / BANNER VISIBILITY
    # ------------------------------------------------------------------

    def _show_transcript(self) -> None:
        if self._has_messages:
            return
        self._has_messages = True
        try:
            self.query_one("#empty-banner", EmptyBanner).display = False
        except Exception:
            pass
        try:
            self.query_one("#transcript", VerticalScroll).display = True
        except Exception:
            pass

    def _clear_transcript(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).remove_children()
        except Exception:
            pass
        self._plan_view = None
        self._plan_signature = None
        self._streaming_widget = None
        self._transcript_log.clear()
        self._search_cursor = 0
        self._has_messages = False
        self._auto_scroll = True
        try:
            self.query_one("#empty-banner", EmptyBanner).display = True
        except Exception:
            pass
        try:
            self.query_one("#transcript", VerticalScroll).display = False
        except Exception:
            pass
        self.live_input_tokens = 0
        self.live_output_tokens = 0
        self.live_total_tokens = 0
        self._refresh_token_bar()

    # ------------------------------------------------------------------
    # PERMISSION CONFIRMATION
    # ------------------------------------------------------------------

    def is_tool_allowed_for_session(self, tool: str) -> bool:
        return str(tool or "") in self._session_allowed_tools

    def allow_tool_for_session(self, tool: str) -> None:
        name = str(tool or "").strip()
        if name:
            self._session_allowed_tools.add(name)

    def run_slash_command(self, command: str) -> None:
        self.submit_prompt_text(str(command or ""))

    # ------------------------------------------------------------------
    # INLINE SLASH COMMANDS
    # ------------------------------------------------------------------

    @on(TextArea.Changed, "#prompt")
    def _prompt_text_changed(self, event: TextArea.Changed) -> None:
        """Handle actual typing and paste events from the prompt."""
        self._update_command_suggestions(event.text_area.text)

    def _update_command_suggestions(self, text: str) -> None:
        """Update local command matches, without running any slash command."""
        if text == self._command_suppressed_text:
            self._hide_command_suggestions()
            return
        self._command_suppressed_text = None

        matches = _slash_command_matches(text)
        if not matches:
            self._hide_command_suggestions()
            return

        self._command_suggestion_matches = matches
        self._command_suggestion_index = 0
        try:
            menu = self.query_one("#command-menu", Vertical)
            options = self.query_one("#command-options", OptionList)
            options.clear_options()
            options.add_options([
                Option(f"{command:<12}  {description}", id=command)
                for command, description in matches
            ])
            options.styles.height = len(matches)
            options.highlighted = 0
            menu.display = True
            self._command_menu_open = True
        except Exception as exc:
            self._command_menu_open = False
            # Visible in Textual / pytest diagnostics; do not swallow UI failures.
            import logging
            logging.getLogger(__name__).warning(
                "Slash command menu update failed: %s", exc, exc_info=True
            )

    def _hide_command_suggestions(self, *, suppress: bool = False) -> None:
        """Dismiss suggestions without changing the user's draft."""
        if suppress:
            try:
                self._command_suppressed_text = self.query_one("#prompt", PromptArea).text
            except Exception:
                self._command_suppressed_text = None
        self._command_menu_open = False
        self._command_suggestion_matches = []
        self._command_suggestion_index = 0
        try:
            self.query_one("#command-menu", Vertical).display = False
        except Exception:
            pass

    def _move_command_suggestion(self, step: int) -> None:
        if not self._command_menu_open or not self._command_suggestion_matches:
            return
        count = len(self._command_suggestion_matches)
        self._command_suggestion_index = (
            self._command_suggestion_index + step
        ) % count
        try:
            self.query_one("#command-options", OptionList).highlighted = (
                self._command_suggestion_index
            )
        except Exception:
            pass

    def _accept_command_suggestion(
        self, command: Optional[str] = None, *, execute: bool = False,
    ) -> bool:
        """Fill with Tab; Enter/click executes safe built-in slash dispatch.

        Argument-taking commands are always filled, never run with a
        missing argument. A bare slash also only fills a suggestion.
        """
        matches = self._command_suggestion_matches
        if not self._command_menu_open or not matches:
            return False
        available = {name for name, _ in matches}
        selected = command if command in available else matches[
            self._command_suggestion_index
        ][0]
        try:
            prompt = self.query_one("#prompt", PromptArea)
            draft = prompt.text
            # Enter executes the highlighted command immediately, even
            # when the user has only typed "/". Commands needing an
            # argument still fill the prompt without running.
            should_run = execute and selected not in _SLASH_ARGUMENT_COMMANDS
            replacement = selected + (
                " " if selected in _SLASH_ARGUMENT_COMMANDS else ""
            )
            self._hide_command_suggestions()
            self._command_suppressed_text = replacement if not should_run else ""
            prompt.set_prompt_text("" if should_run else replacement)
            if should_run:
                self.submit_prompt_text(selected)
            else:
                prompt.focus()
            return True
        except Exception:
            return False

    @on(OptionList.OptionSelected, "#command-options")
    def _command_option_selected(self, event: OptionList.OptionSelected) -> None:
        selected = getattr(event.option, "id", None)
        if selected:
            self._accept_command_suggestion(str(selected), execute=True)


    # ------------------------------------------------------------------
    # PROMPT HISTORY / QUICK PICKERS
    # ------------------------------------------------------------------

    def _remember_prompt(self, text: str) -> None:
        value = (text or "").strip()
        if not value or value.startswith("/"):
            return
        if not self._prompt_history or self._prompt_history[-1] != value:
            self._prompt_history.append(value)
            if len(self._prompt_history) > 200:
                del self._prompt_history[:-200]
        self._history_index = -1
        self._history_draft = ""

    def navigate_prompt_history(self, direction: int) -> Optional[str]:
        if not self._prompt_history:
            return None
        try:
            current = self.query_one("#prompt", PromptArea).text
        except Exception:
            current = ""
        if self._history_index == -1:
            if direction > 0:
                return None
            self._history_draft = current
            self._history_index = len(self._prompt_history) - 1
        else:
            self._history_index += direction
        if self._history_index < 0:
            self._history_index = 0
        if self._history_index >= len(self._prompt_history):
            self._history_index = -1
            return self._history_draft
        return self._prompt_history[self._history_index]

    def action_history_search(self) -> None:
        if not self._prompt_history:
            self._system("prompt history is empty")
            return

        def selected(value: Optional[str]) -> None:
            if value is None:
                return
            try:
                self.query_one("#prompt", PromptArea).set_prompt_text(value)
                self.query_one("#prompt", PromptArea).focus()
            except Exception:
                pass

        self.push_screen(HistorySearchModal(self._prompt_history), selected)

    def _project_root(self) -> Path:
        if self.coordinator is not None:
            for attr in ("plan_agent", "build_agent"):
                agent = getattr(self.coordinator, attr, None)
                workspace = getattr(agent, "workspace", None) if agent else None
                root = getattr(workspace, "project_dir", None) if workspace else None
                if root:
                    return Path(str(root))
        return Path.cwd()

    def _project_files(self, limit: int = 2500) -> List[str]:
        root = self._project_root()
        skip = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "src/agent/depression",
        }
        files: List[str] = []
        try:
            for dirpath, dirnames, filenames in _os.walk(root):
                rel_dir = _os.path.relpath(dirpath, root)
                dirnames[:] = [
                    d for d in dirnames
                    if d not in skip and not d.startswith(".tox")
                ]
                for name in filenames:
                    if name.startswith(".") and name not in (".env.example",):
                        continue
                    rel = name if rel_dir == "." else _os.path.join(rel_dir, name)
                    files.append(rel.replace(_os.sep, "/"))
                    if len(files) >= limit:
                        return sorted(files)
        except Exception:
            pass
        return sorted(files)

    def action_complete_file(self) -> None:
        self.run_worker(self._open_file_picker(), group="file-complete")

    async def _open_file_picker(self) -> None:
        # Walking a large repository can take a noticeable moment; keep it
        # off the UI loop so the TUI stays responsive.
        files = await asyncio.to_thread(self._project_files)
        if not files:
            self._error("no project files found")
            return

        def selected(value: Optional[str]) -> None:
            if value is None:
                return
            try:
                prompt = self.query_one("#prompt", PromptArea)
                prompt.insert_file_reference(value)
                prompt.focus()
            except Exception:
                pass

        self.push_screen(FilePickerModal(files), selected)

    def action_search_transcript(self) -> None:
        if not self._transcript_log:
            self._system("transcript is empty")
            return

        def selected(value: Optional[str]) -> None:
            if value:
                self._search_transcript(value)

        self.push_screen(TranscriptSearchModal(), selected)

    def _search_transcript(self, query: str) -> None:
        needle = query.strip().lower()
        if not needle:
            return
        total = len(self._transcript_log)
        for offset in range(total):
            idx = (self._search_cursor + offset) % total
            entry = self._transcript_log[idx]
            if needle not in str(entry.get("text", "")).lower():
                continue
            widget = entry.get("widget")
            self._search_cursor = idx + 1
            self._auto_scroll = False
            try:
                widget.scroll_visible(animate=True)
            except Exception:
                pass
            self._system(f"match {idx + 1}/{total}: {query}")
            return
        self._system(f"no transcript match: {query}")

    def action_toggle_auto_scroll(self) -> None:
        self._auto_scroll = not self._auto_scroll
        state = "on" if self._auto_scroll else "paused"
        self._system(f"transcript follow {state}")
        if self._auto_scroll:
            self._scroll_transcript_end()

    def on_mouse_scroll_up(self, event: Any) -> None:
        if self._auto_scroll:
            self._auto_scroll = False
            self._system("transcript follow paused — F2 to resume")

    # ------------------------------------------------------------------
    # PERMISSION CONFIRMATION HELPERS
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
    # MODE CHIP
    # ------------------------------------------------------------------

    def _mode_chip(self) -> str:
        icon = MODE_ICONS.get(self.current_mode, "◆")
        model = self.cfg.get("model") or "no model selected"
        if len(model) > 28:
            model = model[:28] + "…"
        model = _esc(model)

        prefix = ""
        if self.busy:
            spin = SPINNER_FRAMES[self.spinner_tick % len(SPINNER_FRAMES)]
            prefix = f"[{AMBER}]{spin}[/] "
            if self._turn_started:
                elapsed = max(0.0, time.time() - self._turn_started)
                mins, secs = divmod(int(elapsed), 60)
                prefix += f"[{MUTED}]{mins:02d}:{secs:02d}[/]  "

        pct = (self.tokens_used / max(1, self.context_window)) * 100.0
        ctx = f"[{MUTED}]ctx {pct:.0f}%[/]"

        queue = ""
        if self.queue_depth > 0:
            queue = f"  [{DIM}]·[/]  [{AMBER}]⧗ {self.queue_depth} queued[/]"

        welcome = ""
        if self._user_name:
            short = self._user_name
            if len(short) > 24:
                short = short[:23] + "…"
            welcome = f"  [{DIM}]·[/]  [{GREEN}]welcome {_esc(short)}[/]"

        return (
            f"{prefix}[{GREEN}]{icon} {self.current_mode.upper()}[/]  "
            f"[{DIM}]·[/]  [{TEXT}]{model}[/]  "
            f"[{DIM}]·[/]  {ctx}{welcome}{queue}"
        )

    def _set_busy_visual(self, busy: bool) -> None:
        try:
            inp = self.query_one("#prompt", PromptArea)
            if busy:
                inp.add_class("busy")
            else:
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

    def _refresh_mcp_status(self) -> None:
        try:
            self.query_one("#panel-mcp", MCPPanel).refresh_status()
        except Exception:
            pass

    @on(Button.Pressed, "#mcp-refresh")
    def _mcp_refresh(self) -> None:
        self._refresh_mcp_status()

    @on(Button.Pressed, "#mcp-reconnect")
    def _mcp_reconnect(self) -> None:
        self._reconnect_mcp()

    @on(Button.Pressed, "#mcp-save")
    def _mcp_save_pressed(self) -> None:
        self._save_mcp_connection()

    @work(exclusive=True, group="mcp-save")
    async def _save_mcp_connection(self) -> None:
        """Save pasted MCP connection safely; attempt live hookup if possible."""
        panel = self.query_one("#panel-mcp", MCPPanel)
        if self.busy or self._is_shutting_down:
            panel.set_save_status("Stop the running agent before changing MCP servers.")
            return
        button = self.query_one("#mcp-save", Button)
        button.disabled = True
        try:
            from agent.tui.mcp_config_ui import save_mcp_server
            saved = save_mcp_server(panel.connection_values())
            panel.set_save_status(f"Saved {saved['name']} ({saved['transport']}). Connecting…")
            client = panel.get_client()
            if client is None:
                if self.coordinator is None:
                    panel.set_save_status("Saved privately. Restart to activate MCP servers.")
                    return
                from agent.mcp.client import MCPClient
                client = MCPClient(saved["config"])
                for role in ("plan_agent", "build_agent"):
                    agent = getattr(self.coordinator, role, None)
                    if agent is not None:
                        agent.mcp_client = client
            else:
                client.update_config(saved["config"])
            # Preserve unrelated live servers. Reconnect only the saved server.
            if hasattr(client, "disconnect_server") and hasattr(client, "connect_server"):
                await client.disconnect_server(saved["name"])
                try:
                    await client.connect_server(saved["name"])
                except Exception as exc:
                    panel.set_save_status(f"Saved, but connection failed: {exc}")
                self._sync_mcp_tools(client)
            else:
                await self._reload_mcp_client(client)
            connected = saved["name"] in getattr(client, "transports", {})
            if connected:
                panel.set_save_status(f"Connected: {saved['name']}")
            else:
                panel.set_save_status(
                    f"Saved {saved['name']}; server connection unsuccessful. "
                    "Check URL, command, and credentials."
                )
        except ValueError as exc:
            panel.set_save_status(f"Invalid connection: {exc}")
        except Exception as exc:
            panel.set_save_status(f"Could not save/connect: {exc}")
        finally:
            button.disabled = False
            panel.refresh_status()

    @work(exclusive=True, group="mcp-reconnect")
    async def _reconnect_mcp(self) -> None:
        """Reconnect configured MCP servers only when the agent is idle.

        Never changes configuration, grants permissions, or invokes MCP tools.
        New tools are registered conservatively as mutating by default.
        """
        if self.busy or self._is_shutting_down:
            self._system("MCP reconnect unavailable while the agent is running.")
            return
        panel = self.query_one("#panel-mcp", MCPPanel)
        button = self.query_one("#mcp-reconnect", Button)
        button.disabled = True
        try:
            # Re-read the actual saved configuration on every reconnect.
            # This also permits recovery when MCP was disabled at startup.
            from agent.config.loader import ConfigLoader
            from agent.mcp.client import MCPClient
            cfg = ConfigLoader().load(validate=False).to_dict(
                include_secrets=True
            ).get("mcp", {}) or {}
            if not cfg.get("enabled"):
                message = "MCP is disabled in saved config; save an MCP server first."
                panel.set_management_status(message)
                self._system(message)
                return

            client = panel.get_client()
            if client is None:
                client = MCPClient(cfg)
                if self.coordinator is None:
                    self._system("No active agent to attach MCP tools to.")
                    return
                for role in ("plan_agent", "build_agent"):
                    agent = getattr(self.coordinator, role, None)
                    if agent is not None:
                        agent.mcp_client = client
            else:
                client.update_config(cfg)
            self._system("Reconnecting configured MCP servers…")
            # Repair missing/down connections without interrupting healthy ones.
            if hasattr(client, "connect_server") and hasattr(client, "disconnect_server"):
                wanted = {server.name for server in client.servers_cfg}
                for name in list(client.transports):
                    if name not in wanted:
                        await client.disconnect_server(name)
                for server in client.servers_cfg:
                    transport = client.transports.get(server.name)
                    state = getattr(getattr(transport, "state", None), "value", "")
                    if state not in ("ready", "connected"):
                        try:
                            await client.connect_server(server.name)
                        except Exception as exc:
                            client.connection_errors[server.name] = str(exc)
                self._sync_mcp_tools(client)
            else:
                await self._reload_mcp_client(client)
            servers = list(client.servers_cfg)
            connected = len(client.transports)
            errors = getattr(client, "connection_errors", {}) or {}
            if not servers:
                message = "MCP enabled but no servers configured."
            elif connected == len(servers):
                message = f"MCP connected: {connected}/{len(servers)} servers."
            else:
                failures = "; ".join(
                    f"{name}: {detail}" for name, detail in list(errors.items())[:3]
                )
                message = (
                    f"MCP connected: {connected}/{len(servers)} servers. "
                    f"Failures: {failures or 'check server configuration'}"
                )
            panel.set_management_status(message)
            self._system(message)
        except Exception as exc:
            message = f"MCP reconnect failed: {exc}"
            panel.set_management_status(message)
            self._error(message)
        finally:
            self._refresh_mcp_status()
            if not self.busy and not self._is_shutting_down:
                button.disabled = False

    @on(Button.Pressed, "#mcp-server-connect")
    def _mcp_server_connect_pressed(self) -> None:
        self._connect_mcp_server()

    @work(exclusive=True, group="mcp-manage")
    async def _connect_mcp_server(self) -> None:
        panel = self.query_one("#panel-mcp", MCPPanel)
        name = panel.selected_server_name()
        if self.busy or self._is_shutting_down:
            panel.set_management_status("Stop the running agent first.")
            return
        if not name:
            panel.set_management_status("Select a server first.")
            return
        client = panel.get_client()
        if client is None or not getattr(client, "enabled", False):
            panel.set_management_status(
                "MCP is disabled. Enable it and restart, or use Save & Connect."
            )
            return

        panel.set_management_status(f"Connecting {name}…")
        try:
            changed = await client.connect_server(name)
            self._sync_mcp_tools(client)
            message = (
                f"Connected MCP server: {name}" if changed
                else f"MCP server already connected: {name}"
            )
            panel.set_management_status(message)
            self._system(message)
        except ValueError as exc:
            panel.set_management_status(f"Connect failed: {exc}")
        except Exception as exc:
            panel.set_management_status(
                f"Connect failed: {exc}"
            )
            self._error(f"MCP connect failed for {name}: {exc}")
        finally:
            panel.refresh_status()

    @on(Button.Pressed, "#mcp-server-disconnect")
    def _mcp_server_disconnect_pressed(self) -> None:
        self._disconnect_mcp_server()

    @work(exclusive=True, group="mcp-manage")
    async def _disconnect_mcp_server(self) -> None:
        panel = self.query_one("#panel-mcp", MCPPanel)
        name = panel.selected_server_name()
        if self.busy or self._is_shutting_down:
            panel.set_management_status("Stop the running agent first.")
            return
        if not name:
            panel.set_management_status("Select a server first.")
            return
        client = panel.get_client()
        if client is None or not getattr(client, "enabled", False):
            panel.set_management_status("MCP is disabled or not initialized.")
            return

        panel.set_management_status(f"Disconnecting {name}…")
        try:
            changed = await client.disconnect_server(name)
            self._sync_mcp_tools(client)
            message = (
                f"Disconnected MCP server: {name}" if changed
                else f"MCP server was not connected: {name}"
            )
            panel.set_management_status(message)
            self._system(message)
        except Exception as exc:
            panel.set_management_status(
                f"Disconnect failed: {exc}"
            )
            self._error(
                f"MCP disconnect failed for {name}: {exc}"
            )
        finally:
            panel.refresh_status()

    @on(Button.Pressed, "#mcp-server-delete")
    def _mcp_server_delete_pressed(self) -> None:
        panel = self.query_one("#panel-mcp", MCPPanel)
        name = panel.selected_server_name()
        if self.busy or self._is_shutting_down:
            panel.set_management_status("Stop the running agent first.")
            return
        if not name:
            panel.set_management_status("Select a server first.")
            return

        def confirmed(yes: bool) -> None:
            if yes:
                self.run_worker(
                    self._delete_mcp_server(name),
                    group="mcp-manage", exclusive=True,
                )

        self.push_screen(
            ConfirmModal(
                "DELETE MCP SERVER",
                f"Remove '{name}' from the MCP config? "
                "Its auto-generated secrets will also be removed.",
            ),
            confirmed,
        )

    async def _delete_mcp_server(self, name: str) -> None:
        panel = self.query_one("#panel-mcp", MCPPanel)
        if self.busy or self._is_shutting_down:
            panel.set_management_status("Stop the running agent first.")
            return
        try:
            from agent.tui.mcp_config_ui import delete_mcp_server

            result = delete_mcp_server(name)
            client = panel.get_client()
            if client is not None:
                await client.disconnect_server(name)
                client.update_config(result["config"])
                self._sync_mcp_tools(client)

            message = f"Deleted MCP {result['kind']}: {name}"
            if result.get("cleanup_error"):
                message += f" (secret cleanup failed: {result['cleanup_error']})"
            panel.set_management_status(message)
            self._system(message)
        except ValueError as exc:
            panel.set_management_status(f"Delete failed: {exc}")
        except Exception as exc:
            panel.set_management_status(
                f"Delete failed: {exc}"
            )
            self._error(f"MCP delete failed for {name}: {exc}")
        finally:
            panel.refresh_status()

    async def _reload_mcp_client(self, client: Any) -> None:
        """Reload MCP and sync Build tools without changing agent permissions."""
        await client.reload()
        self._sync_mcp_tools(client)

    def _sync_mcp_tools(self, client: Any) -> None:
        """Mirror live MCP tools into every agent registry, removing stale entries."""
        coordinator = self.coordinator
        discovered = {
            item["function"]["name"]: item["function"]
            for item in client.list_tools()
        }
        # Permission defaults must remain conservative for unknown servers.
        # Never infer read-only safety from a user-controlled tool name.
        for role in ("build_agent", "plan_agent"):
            agent = getattr(coordinator, role, None)
            registry = getattr(agent, "tool_registry", None)
            if registry is None:
                continue
            for name in list(registry.list_tools(include_disabled=True)):
                if name.startswith("mcp__") and name not in discovered:
                    registry.unregister_tool(name)
            for name, schema in discovered.items():
                async def _call(_name: str = name, **kwargs: Any) -> Any:
                    return await client.call_tool(_name, kwargs)
                registry.register_external(
                    name=name,
                    handler=_call,
                    description=str(schema.get("description") or ""),
                    parameters=schema.get("parameters") or {
                        "type": "object", "properties": {},
                    },
                    read_only=False,
                    mutating=True,
                    category="cloud" if name.startswith("mcp__aws__") else "misc",
                )
            loop = getattr(agent, "loop", None)
            if loop is not None:
                loop._system_prompt_cache = None
                loop._system_prompt_fp = None

    def _refresh_aws_status(self) -> None:
        try:
            self.query_one("#panel-aws", AWSPanel).refresh_mcp_status()
        except Exception:
            pass

    def _refresh_setup_status(self) -> None:
        try:
            llm_ready = bool(
                str(self.cfg.get("model") or "").strip()
                and str(self.cfg.get("base_url") or "").strip()
            )
            aws_ready = bool(
                self.aws.get("access_key")
                or self.aws.get("secret_key")
                or _os.environ.get("AWS_ACCESS_KEY_ID")
            )
            profile_ready = bool(self._user_name)
            self.query_one("#empty-banner", EmptyBanner).set_setup(
                llm=llm_ready, aws=aws_ready, profile=profile_ready,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PANEL SWITCHING
    # ------------------------------------------------------------------

    def _set_active_panel(self, which: str) -> None:
        self._active_panel = which
        for name in _PANEL_NAMES:
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
        if which == "mcp":
            self._refresh_mcp_status()
        if which == "llmcost":
            self._refresh_llmcost_panel()
        if which == "profile":
            try:
                self.query_one("#panel-profile", ProfilePanel).refresh_profile()
            except Exception:
                pass

    def _show_panel(self, which: str) -> None:
        self._set_active_panel(which)
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            if not sidebar.is_open:
                sidebar.toggle()
        except Exception:
            pass

    @on(SidebarTab.Pressed, ".side-tab")
    def _on_tab(self, event: SidebarTab.Pressed) -> None:
        panel = event.tab.id.replace("tab-", "")
        self._set_active_panel(panel)
        # Connection panels should focus their first editable field.
        # Do not unexpectedly force the user back to the chat input.
        try:
            if panel == "mcp":
                self.query_one("#panel-mcp", MCPPanel).focus_entry_field()
            elif panel == "llm":
                self.query_one("#base-url", Input).focus()
            else:
                self._refocus_prompt()
        except Exception:
            self._refocus_prompt()

    # ------------------------------------------------------------------
    # TRANSCRIPT WRITES
    # ------------------------------------------------------------------

    _CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def _sanitize(self, text: str) -> str:
        clean = self._CTRL_RE.sub("", str(text))
        return clean.replace("[", r"\[")

    def _strip_ctrl(self, text: str) -> str:
        return self._CTRL_RE.sub("", str(text))

    def _scroll_transcript_end(self) -> None:
        if not self._auto_scroll:
            return
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        try:
            transcript.scroll_end(animate=False)
        except Exception:
            return

    def _log_transcript(self, kind: str, text: str, widget: Any) -> None:
        self._transcript_log.append({
            "kind": kind, "text": str(text), "widget": widget,
        })

    def _write(self, text: str, cls: str = "agent", markup: bool = False) -> None:
        if self._is_shutting_down:
            return
        self._show_transcript()
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        safe = self._strip_ctrl(text) if markup else self._sanitize(text)
        widget = Static(safe, classes=cls, markup=True)
        try:
            transcript.mount(widget)
        except Exception:
            widget = Static(str(text), classes=cls, markup=False)
            transcript.mount(widget)
        self._log_transcript(cls, text, widget)
        self._scroll_transcript_end()

    def _system(self, text: str) -> None:
        self._write(f"· {text}", "system")

    def _error(self, text: str) -> None:
        self._write(f"× {text}", "error")

    def _queued(self, text: str) -> None:
        self._write(f"⧗ {text}", "queued")

    def _user(self, text: str) -> None:
        if self._is_shutting_down:
            return
        self._show_transcript()
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        safe = _esc(text)
        stamp = time.strftime("%H:%M")
        widget = Static(
            f"[{DIM}]{stamp}[/] [bold {GREEN}]›[/] [bold {TEXT}]{safe}[/]",
            classes="user", markup=True,
        )
        try:
            transcript.mount(widget)
        except Exception:
            widget = Static(f"{stamp} › {text}", classes="user", markup=False)
            transcript.mount(widget)
        self._log_transcript("user", text, widget)
        self.message_count = self.message_count + 1
        self._scroll_transcript_end()

    # ------------------------------------------------------------------
    # INPUT / COMMANDS
    # ------------------------------------------------------------------

    def submit_prompt_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        if text.startswith("/"):
            self._slash(text)
            return

        if (self.busy or self._draining) and _is_stop_request(text):
            self._request_agent_stop()
            return

        if self.busy or self._draining:
            self._prompt_queue.append(text)
            self.queue_depth = len(self._prompt_queue)
            self._refresh_mode_chip()
            preview = text if len(text) <= 60 else text[:60] + "…"
            self._queued(f"queued ({len(self._prompt_queue)} ahead): {preview}")
            return

        self._begin_query()
        self._agent_worker = self._run_agent(text)

    def _begin_query(self) -> None:
        self.live_input_tokens = 0
        self.live_output_tokens = 0
        self.live_total_tokens = 0
        self._refresh_token_bar()
        try:
            self.query_one("#panel-llmcost", LLMCostPanel).mark_query_start()
        except Exception:
            pass

    def _slash(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/aws":
            self._show_panel("aws")
            try:
                self.query_one("#aws-key", Input).focus()
            except Exception:
                pass
        elif command == "/mcp":
            self._show_panel("mcp")
            try:
                self.query_one("#panel-mcp", MCPPanel).focus_entry_field()
            except Exception:
                pass
        elif command == "/connect":
            self._show_panel("llm")
            try:
                self.query_one("#base-url", Input).focus()
            except Exception:
                pass
        elif command == "/model":
            self._show_panel("llm")
            try:
                self.query_one("#model-id", Input).focus()
            except Exception:
                pass
        elif command == "/models":
            self._show_panel("llm")
            self._discover_models()
        elif command == "/profile":
            self._show_panel("profile")
        elif command == "/context":
            self._show_panel("context")
        elif command == "/llmcost":
            self._show_panel("llmcost")
        elif command == "/todo":
            self._show_panel("todo")
            self._dump_todos()
        elif command == "/sessions":
            self._open_sessions()
        elif command == "/copy":
            self._copy_last_output()
        elif command == "/export":
            self._export_transcript(arg)
        elif command == "/retry":
            self._retry_last()
        elif command == "/continue":
            self._continue_last()
        elif command == "/search":
            if arg.strip():
                self._search_transcript(arg)
            else:
                self.action_search_transcript()
        elif command == "/doctor":
            self._doctor()
        elif command == "/help":
            self._show_panel("help")
        elif command == "/run":
            self._run_shell(arg)
        elif command == "/shot":
            self._open_shot(arg)
        elif command == "/clear":
            self._clear_transcript()
        elif command == "/plan":
            self._switch_mode("plan")
        elif command == "/build":
            self._switch_mode("build")
        elif command == "/auto":
            self._switch_mode("auto")
        elif command in ("/stop", "/interrupt"):
            self._request_agent_stop()
        elif command in ("/quit", "/exit"):
            self._is_shutting_down = True
            self.exit()
        else:
            self._error(f"unknown command: {command}")

    # ------------------------------------------------------------------
    # SESSIONS
    # ------------------------------------------------------------------

    def _open_sessions(self) -> None:
        self.run_worker(self._load_sessions(), group="sessions")

    async def _load_sessions(self) -> None:
        if self.session_manager is None:
            self._error("session manager is unavailable")
            return
        try:
            sessions = await self.session_manager.list_sessions()
        except Exception as exc:
            self._error(f"could not list sessions: {exc}")
            return
        if not sessions:
            self._system("no saved sessions")
            return
        self.push_screen(SessionPickerModal(sessions), self._session_selected)

    def _session_selected(self, result: Any) -> None:
        if not result:
            return
        action, session_id = result
        if action == "open":
            self.run_worker(self._resume_session(session_id), group="sessions")
        elif action == "rename":
            self._prompt_rename_session(session_id)
        elif action == "delete":
            self._confirm_delete_session(session_id)

    def _session_title(self, session: Any) -> str:
        return str(getattr(session, "name", None) or "session")

    def _prompt_rename_session(self, session_id: str) -> None:
        async def load_and_prompt() -> None:
            try:
                session = await self.session_manager.load_session(session_id)
            except Exception as exc:
                self._error(f"could not load session: {exc}")
                return
            if session is None:
                self._error("session not found")
                return

            def renamed(name: Optional[str]) -> None:
                if name:
                    self.run_worker(
                        self._rename_session(session, name), group="sessions",
                    )

            self.push_screen(
                TextPromptModal("RENAME SESSION", self._session_title(session)),
                renamed,
            )

        self.run_worker(load_and_prompt(), group="sessions")

    async def _rename_session(self, session: Any, name: str) -> None:
        try:
            session.name = name
            ok = await self.session_manager.save_session(session)
            if ok:
                self._system(f"session renamed: {name}")
            else:
                self._error("session rename failed")
        except Exception as exc:
            self._error(f"session rename failed: {exc}")

    def _confirm_delete_session(self, session_id: str) -> None:
        def confirmed(yes: bool) -> None:
            if yes:
                self.run_worker(
                    self._delete_session(session_id), group="sessions",
                )

        self.push_screen(
            ConfirmModal(
                "DELETE SESSION",
                "This removes the saved session transcript. Continue?",
            ),
            confirmed,
        )

    async def _delete_session(self, session_id: str) -> None:
        if self.session_manager is None:
            return
        try:
            ok = await self.session_manager.delete_session(session_id)
            if ok:
                self._system(f"deleted session {session_id[:8]}")
            else:
                self._error("session delete failed")
        except Exception as exc:
            self._error(f"session delete failed: {exc}")

    async def _resume_session(self, session_id: str) -> None:
        if self.session_manager is None or self.coordinator is None:
            self._error("sessions are unavailable")
            return
        try:
            session = await self.session_manager.load_session(session_id)
        except Exception as exc:
            self._error(f"could not load session: {exc}")
            return
        if session is None:
            self._error("session not found")
            return

        for attr in ("plan_agent", "build_agent"):
            agent = getattr(self.coordinator, attr, None)
            if agent is None:
                continue
            agent.session = session
            manager = getattr(agent, "context_manager", None)
            if manager is not None and hasattr(manager, "load_from_session"):
                try:
                    await manager.load_from_session(session)
                except Exception:
                    pass

        history = getattr(session, "history", None)
        count = len(history.all()) if hasattr(history, "all") else len(history or [])
        self._clear_transcript()
        self.message_count = count
        self._last_output = ""
        self._last_prompt = ""
        self._show_transcript()
        self._system(
            f"resumed session {self._session_title(session)} "
            f"({count} message(s)); prior context loaded"
        )
        self._refresh_context_panel()
        self._refresh_token_bar()
        self._refresh_llmcost_panel()
        self._refresh_mode_chip()
        self._refocus_prompt()

    # ------------------------------------------------------------------
    # TRANSCRIPT ACTIONS
    # ------------------------------------------------------------------

    @work(exclusive=True, group="clipboard")
    async def _copy_last_output(self) -> None:
        if not self._last_output:
            self._system("nothing to copy yet")
            return
        from agent.utils.clipboard import copy_text

        text = self._last_output
        # Native clipboard tools can take seconds (or hit their timeout) on
        # a misconfigured display server. Never run them on the UI loop.
        outcome = await asyncio.to_thread(copy_text, text)
        if outcome == "unavailable":
            # OSC 52 must be written from the UI thread.
            try:
                self.copy_to_clipboard(text)
                outcome = "terminal"
            except Exception:
                outcome = "unavailable"
        chars = len(text)
        if outcome == "copied":
            self._system(f"copied last reply ({chars:,} chars) to the clipboard")
        elif outcome == "terminal":
            self._system(
                "sent clipboard request to terminal (OSC 52); "
                "paste availability depends on terminal support"
            )
        else:
            self._error(
                "clipboard unavailable; install wl-clipboard (Wayland) "
                "or xclip/xsel (X11), or use a terminal supporting OSC 52"
            )

    def _export_transcript(self, path: str = "") -> None:
        if not self._transcript_log:
            self._system("transcript is empty")
            return
        target = (path or "").strip().strip("'\"")
        if target:
            out = Path(target).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = self._project_root() / f"depression-transcript-{stamp}.md"
        if not out.is_absolute():
            out = self._project_root() / out
        lines = [
            "# depression.ai transcript",
            "",
            f"- exported: {datetime.now().isoformat(timespec='seconds')}",
            f"- model: {self.cfg.get('model') or '—'}",
            f"- mode: {self.current_mode}",
            "",
        ]
        for entry in self._transcript_log:
            kind = str(entry.get("kind") or "agent")
            text = str(entry.get("text") or "")
            if kind == "user":
                lines.extend(["## You", "", text, ""])
            elif kind == "agent":
                lines.extend(["## depression.ai", "", text, ""])
            else:
                lines.extend([f"> {text}", ""])
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines), encoding="utf-8")
            self._system(f"exported transcript → {out}")
        except Exception as exc:
            self._error(f"export failed: {exc}")

    # ------------------------------------------------------------------
    # RETRY / CONTINUE / DIAGNOSTICS
    # ------------------------------------------------------------------

    def _retry_last(self) -> None:
        if self.is_agent_busy():
            self._error("agent is busy; stop or wait before retrying")
            return
        if not self._last_prompt:
            self._system("no previous prompt to retry")
            return
        prompt = self._last_prompt
        self._system("retrying the last prompt")
        self.submit_prompt_text(prompt)

    def _continue_last(self) -> None:
        if self.is_agent_busy():
            self._error("agent is busy; stop or wait before continuing")
            return
        if not self._last_prompt:
            self._system("nothing to continue")
            return
        self._system("continuing from the previous work")
        self.submit_prompt_text(
            "Continue the previous task from where it stopped. "
            "Review the transcript, do not start over, and finish the "
            "remaining work."
        )

    def _timeout_summary(self, timeout: float) -> str:
        if self.coordinator is None:
            return ""
        for attr in ("plan_agent", "build_agent"):
            agent = getattr(self.coordinator, attr, None)
            loop = getattr(agent, "loop", None) if agent else None
            if loop is None or not hasattr(loop, "_format_stop_summary"):
                continue
            try:
                summary = str(loop._format_stop_summary())
            except Exception:
                continue
            if summary:
                minutes = timeout / 60.0
                first = f"Timed out after {minutes:.0f} min."
                if summary.startswith("Stopped as requested."):
                    summary = summary.replace(
                        "Stopped as requested.", first, 1,
                    )
                else:
                    summary = first + "\n" + summary
                return summary
        return ""

    def _doctor(self) -> None:
        self._show_panel("help")
        self._system("diagnostics")
        checks: list[tuple[str, bool, str]] = []
        checks.append(("python", sys.version_info >= (3, 10),
                       platform.python_version()))
        try:
            root = self._project_root()
            writable = _os.access(root, _os.W_OK)
        except Exception:
            root, writable = Path.cwd(), False
        checks.append(("workspace", writable, f"{root}"))
        model = str(self.cfg.get("model") or "")
        base = str(self.cfg.get("base_url") or "")
        checks.append(("llm config", bool(model and base),
                       f"{model or '—'} @ {base or '—'}"))
        api_key = str(self.cfg.get("api_key") or "")
        checks.append(("api key", bool(api_key),
                       "set" if api_key else "missing (Ollama may not need one)"))
        aws = bool(self.aws.get("access_key"))
        checks.append(("aws credentials", aws,
                       self.aws.get("region") or DEFAULT_AWS_REGION))
        checks.append(("session manager", self.session_manager is not None,
                       "available" if self.session_manager else "unavailable"))
        checks.append(("coordinator", self.coordinator is not None,
                       "ready" if self.coordinator else "unavailable"))
        checks.append(("git", shutil.which("git") is not None,
                       shutil.which("git") or "not found"))
        checks.append(("ripgrep", shutil.which("rg") is not None,
                       shutil.which("rg") or "not found"))
        playwright = importlib.util.find_spec("playwright") is not None
        checks.append(("playwright", playwright,
                       "installed" if playwright else "optional"))
        checks.append(("turn timeout", True, f"{self.turn_timeout / 60:.0f} min"))
        for name, ok, detail in checks:
            mark = "✓" if ok else "×"
            self._write(
                f"  {mark} {name:<18} {detail}",
                "system" if ok else "error",
            )

    # ------------------------------------------------------------------
    # SHELL / IMAGES
    # ------------------------------------------------------------------

    def _open_shot(self, path: str) -> None:
        if not path:
            self._error("usage: /shot <path-to-image>")
            return
        p = path.strip().strip("'\"")
        if not _os.path.exists(p):
            self._error(f"not found: {p}")
            return
        if _open_file_with_default_app(p):
            self._system(f"opened {p}")
        else:
            self._error(f"could not open {p}; open it manually")

    def _dump_todos(self) -> None:
        try:
            from agent.tools.todo import TodoTool
        except Exception:
            self._system("TodoTool not available")
            return
        session = self._session()
        sid = TodoTool._session_key(session)
        store = TodoTool._strong_keys.get(sid) or {}
        if not store:
            self._system("(todo list is empty)")
            return
        self._system("todo list:")
        for item in store.values():
            status = getattr(item, "status", "pending")
            title = getattr(item, "title", "?")
            self._write(f"  [{status}] {title}", "system")

    def _run_shell(self, cmd: str) -> None:
        if not cmd:
            self._error("usage: /run <command>")
            return
        self._system(f"$ {cmd}")
        self._user_shell_worker = self._run_shell_async(cmd)

    @work(exclusive=True, group="user-shell")
    async def _run_shell_async(self, cmd: str) -> None:
        """Stream /run output without letting the child own the TUI terminal."""
        long_running = is_long_running(cmd)
        if long_running:
            self._system(
                f"long-running command ({long_running}); use /stop to terminate"
            )

        def on_output(stream: str, line: str) -> None:
            self._write(
                f"  {line}", "error" if stream == "stderr" else "agent"
            )

        try:
            result = await run_command(
                cmd,
                timeout=None if long_running else 30.0,
                on_output=on_output,
            )
        except asyncio.CancelledError:
            self._system("run stopped")
            raise
        except Exception as exc:
            self._error(f"run failed: {exc}")
            return

        if result.timed_out:
            self._error("run timed out after 30 seconds")
            return
        if result.truncated:
            self._system("output truncated")
        self._system(f"exit {result.exit_code}")

    def _request_agent_stop(self) -> None:
        """Ask the running loop to stop gracefully and summarize its work."""
        shell_worker = self._user_shell_worker
        shell_running = (
            shell_worker is not None
            and getattr(shell_worker, "is_running", False)
        )
        agent_running = self.busy or self._draining
        if not agent_running and not shell_running:
            self._system("nothing is running")
            return
        if shell_running:
            shell_worker.cancel()
            self._user_shell_worker = None
            self._system("shell command stopped")
        if not agent_running:
            return
        self._prompt_queue.clear()
        self.queue_depth = 0
        self._system("stop requested — wrapping up with a summary…")
        stopped = False
        if self.coordinator is not None:
            for attr in ("plan_agent", "build_agent"):
                agent = getattr(self.coordinator, attr, None)
                loop = getattr(agent, "loop", None) if agent else None
                if loop is not None and hasattr(loop, "stop"):
                    try:
                        loop.stop()
                        stopped = True
                    except Exception:
                        pass
        if not stopped:
            self.cancel_current_agent()
        self._refresh_mode_chip()

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
        self._turn_started = time.time()
        self._remember_prompt(text)
        self._last_prompt = text
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

        self._plan_view = None
        self._plan_signature = None
        self._streaming_widget = None

        started = time.time()
        try:
            result = await asyncio.wait_for(
                self.coordinator.process_query(
                    text,
                    mode=self.current_mode,
                    # Plan mode is planning-only; build/auto may auto-execute.
                    auto_execute=(self.current_mode != "plan"),
                ),
                timeout=self.turn_timeout,
            )

            self._sync_tokens_from_loop()

            if result.get("success"):
                output = (
                    result.get("execution")
                    or result.get("response")
                    or "task completed."
                )
                await self._render_output(str(output), started)

                try:
                    text_plan = self._plan_from_text(str(output))
                    if text_plan:
                        normalised = self._normalise_plan_entries(text_plan)
                        if normalised:
                            await self._upsert_plan(normalised)
                            self._update_todo_panel(normalised)
                except Exception:
                    pass

                ctx = result.get("context") or {}
                tokens = ctx.get("tokens_used") or result.get("tokens_used") or 0
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
                output = result.get("execution") or result.get("response")
                if output:
                    await self._render_output(str(output), started)
                self._error(str(result.get("error", "agent request failed")))
        except asyncio.TimeoutError:
            mins = self.turn_timeout / 60.0
            summary = self._timeout_summary(self.turn_timeout)
            if summary:
                await self._render_output(summary, started)
            self._error(
                f"agent timed out after {mins:.0f} min. "
                "Partial work is preserved above; use /continue to resume, "
                "/retry to rerun, or raise TUI_TURN_TIMEOUT."
            )
            self._preserve_prompt(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error(f"agent error: {exc}")
            self._preserve_prompt(text)
        finally:
            # Remove the streaming widget (its content is now part of the
            # structured OutputView).
            if self._streaming_widget is not None and self._streaming_widget.is_mounted:
                try:
                    await self._streaming_widget.remove()
                except Exception:
                    pass
            self._streaming_widget = None

            self.busy = False
            self._set_busy_visual(False)
            self._refresh_mode_chip()
            self._refocus_prompt()
            self._refresh_token_bar()
            try:
                self.query_one("#panel-llmcost", LLMCostPanel).mark_query_end()
                self._refresh_llmcost_panel()
            except Exception:
                pass
            self._schedule_drain()

    def _preserve_prompt(self, text: str) -> None:
        try:
            prompt = self.query_one("#prompt", PromptArea)
            if not prompt.text.strip():
                prompt.set_prompt_text(text)
            prompt.focus()
        except Exception:
            pass

    async def _render_output(self, text: str, started: float) -> None:
        if self._is_shutting_down:
            return
        self._show_transcript()
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return

        elapsed = time.time() - started
        stamp = time.strftime("%H:%M:%S")
        meta = f"{stamp} · {elapsed:.2f}s"
        self._last_output = str(text)
        try:
            view = OutputView(text=text, header="◆ depression.ai", meta=meta)
            await transcript.mount(view)
            self._log_transcript("agent", str(text), view)
        except Exception:
            self._write(text, "agent")
        self._scroll_transcript_end()

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
        self._begin_query()
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
        if not base_url or not model:
            status.update("base URL and model ID are required (API key optional locally)")
            return
        try:
            provider = configure_runtime_provider(
                get_llm_registry(), base_url, api_key, model
            )
            self.cfg = load_runtime_config()
            name = getattr(provider, "name", None) or type(provider).__name__
            status.update(f"connected · {_esc(model)}")
            try:
                self.query_one("#llm-provider", Static).update(
                    f"[{MUTED}]provider[/] [{TEXT}]{_esc(str(name))}[/]"
                )
            except Exception:
                pass
            self._refresh_mode_chip()
            self._refresh_context_panel()
            self._refresh_llmcost_panel()
            self._refresh_setup_status()
            self._system(f"LLM connected: {base_url} · {model} · {name}")
        except Exception as exc:
            status.update(f"error: {_esc(exc)}")

    @on(Button.Pressed, "#discover")
    def discover_llm(self) -> None:
        self._discover_models()

    @on(Button.Pressed, "#llm-test")
    def test_llm_button(self) -> None:
        self._test_llm()

    @work(exclusive=True, group="llm-test")
    async def _test_llm(self) -> None:
        url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key = self.query_one("#api-key", Input).value.strip()
        model = self.query_one("#model-id", Input).value.strip()
        status = self.query_one("#llm-status", Static)
        if not url:
            status.update("enter a base URL first")
            return
        status.update("testing connection…")
        started = time.time()
        try:
            base = plan_runtime(url, key, model).base_url or url
            async with httpx.AsyncClient(timeout=15) as client:
                headers = ({"Authorization": f"Bearer {key}"} if key else {})
                response = await client.get(f"{base}/models", headers=headers)
                response.raise_for_status()
            elapsed = time.time() - started
            status.update(f"connection ok · {elapsed * 1000:.0f} ms")
            try:
                spec = plan_runtime(url, key, model)
                self.query_one("#llm-provider", Static).update(
                    f"[{MUTED}]provider[/] [{TEXT}]{_esc(spec.family)}[/]"
                )
            except Exception:
                pass
        except Exception as exc:
            status.update(f"connection failed: {_esc(exc)}")

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
            base = plan_runtime(url, key, "").base_url or url
            async with httpx.AsyncClient(timeout=20) as client:
                headers = ({"Authorization": f"Bearer {key}"} if key else {})
                response = await client.get(f"{base}/models", headers=headers)
                response.raise_for_status()
                models = response.json().get("data", [])
            ids = [str(m.get("id")) for m in models if m.get("id")]
            if ids:
                status.update(f"discovered {len(ids)} model(s); paste the desired ID into Model ID")
                self._system("models: " + ", ".join(ids[:12]))
            else:
                status.update("/models returned no model IDs")
        except Exception as exc:
            status.update(f"discovery failed: {_esc(exc)}")

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
        panel = self.query_one("#panel-aws", AWSPanel)
        panel.start_radar()
        result = panel.save_credentials()
        if not result.get("ok"):
            panel.set_status(f"error: {_esc(result.get('error') or 'save failed')}")
            panel.start_trace(ok=False)
            return
        self.aws = get_aws_credentials()
        panel.refresh_values()
        panel.set_status(
            f"saved to project .env · {self.aws.get('region') or DEFAULT_AWS_REGION}"
        )
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
                res = install_aws_preset_into_config(
                    mcp_cfg, region=self.aws.get("region")
                )
                if isinstance(res, dict):
                    if res.get("installed"):
                        note = " · MCP config updated (restart to apply)"
                elif res:
                    note = " · MCP config updated (restart to apply)"
        except Exception as exc:
            note = f" · MCP config refresh skipped: {_esc(exc)}"
        panel.refresh_mcp_status()
        self._refresh_setup_status()
        self._system(
            f"AWS credentials saved to project .env "
            f"({self.aws.get('region') or DEFAULT_AWS_REGION}){note}"
        )

    def on_aws_changed(self) -> None:
        if self.coordinator is not None:
            for attr in ("plan_agent", "build_agent"):
                agent = getattr(self.coordinator, attr, None)
                if agent is None:
                    continue
                loop = getattr(agent, "loop", None)
                if loop is not None:
                    try:
                        loop._system_prompt_cache = None
                        loop._system_prompt_fp = None
                        loop._project_ctx = None
                        loop._project_ctx_ts = 0.0
                    except Exception:
                        pass
        try:
            if self.coordinator is not None:
                for attr in ("plan_agent", "build_agent"):
                    agent = getattr(self.coordinator, attr, None)
                    client = getattr(agent, "mcp_client", None) if agent else None
                    if client is None:
                        continue
                    if hasattr(client, "reload"):
                        try:
                            asyncio.create_task(client.reload())
                        except Exception:
                            pass
                    break
        except Exception:
            pass
        self._refresh_aws_status()

    # ------------------------------------------------------------------
    # KEYS
    # ------------------------------------------------------------------

    def action_tab_or_mode(self) -> None:
        if self._command_menu_open:
            self._accept_command_suggestion(execute=False)
            return
        try:
            prompt = self.query_one("#prompt", PromptArea)
            if not prompt.disabled and not prompt.has_focus:
                prompt.focus()
                return
        except Exception:
            pass
        self.action_cycle_mode()

    def action_interrupt(self) -> None:
        if self._command_menu_open and not isinstance(self.screen, ModalScreen):
            self._hide_command_suggestions(suppress=True)
            return
        if isinstance(self.screen, PermissionModal):
            self.screen.action_deny()
            return
        worker = self._agent_worker
        running = worker is not None and getattr(worker, "is_running", False)
        shell_worker = self._user_shell_worker
        shell_running = (
            shell_worker is not None
            and getattr(shell_worker, "is_running", False)
        )
        if not self.busy and not running and not shell_running:
            return
        if shell_running:
            shell_worker.cancel()
            self._user_shell_worker = None
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
        try:
            self.query_one("#panel-llmcost", LLMCostPanel).mark_query_end()
        except Exception:
            pass
        self._refocus_prompt()

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

    def action_toggle_sidebar(self) -> None:
        try:
            self.query_one("#sidebar", Sidebar).toggle()
        except Exception:
            pass

    def action_clear(self) -> None:
        self._clear_transcript()

    def action_toggle_expand_tool(self) -> None:
        widget = self._active_tool_widget
        if widget is None:
            return
        try:
            widget.toggle_expand()
            self._scroll_transcript_end()
        except Exception:
            pass

    def action_cycle_mode(self) -> None:
        i = (
            MODE_ORDER.index(self.current_mode)
            if self.current_mode in MODE_ORDER else 0
        )
        self._switch_mode(MODE_ORDER[(i + 1) % len(MODE_ORDER)])


__all__ = [
    "DepressionApp",
    "PermissionModal",
    "PromptArea",
    "TodoPanel",
    "StreamingOutput",
]