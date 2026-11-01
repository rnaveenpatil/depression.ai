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


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
MODE_ICONS = {"plan": "◇", "build": "◆", "auto": "◈"}
MODE_ORDER = ["build", "plan", "auto"]

DEFAULT_TURN_TIMEOUT = 3600.0
DEFAULT_AWS_REGION = "us-east-1"

_PANEL_NAMES = ("llm", "aws", "mcp", "profile", "context", "llmcost", "todo", "help")

_DIFF_TOOLS = {"write", "edit", "apply_patch", "filesystem", "patch"}

_PROMPT_MIN_ROWS = 1
_PROMPT_MAX_ROWS = 10

_STOP_REQUESTS = {
    "stop", "stop it", "stop now", "ok stop", "okay stop", "please stop",
    "cancel", "cancel it", "abort", "abort it", "halt", "wait",
    "enough", "that's enough", "thats enough", "quit",
}


def _is_stop_request(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9' ]+", " ", str(text).lower())
    return " ".join(normalized.split()) in _STOP_REQUESTS


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"


def _shorten_path(path: str, max_len: int = 32) -> str:
    if not path:
        return "—"
    if len(path) <= max_len:
        return path
    head_len = max(4, max_len - 24)
    tail_len = max_len - head_len - 1
    return f"{path[:head_len]}…{path[-tail_len:]}"


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a value from a dict-like or attribute-like object."""
    if obj is None:
        return default
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default


def _open_file_with_default_app(path: str) -> bool:
    """Open a file with the OS default application. Best effort."""
    try:
        system = platform.system().lower()
        if system == "darwin":
            cmd = ["open", path]
        elif system == "windows":
            cmd = ["cmd", "/c", "start", "", path]
        else:
            if subprocess.run(["which", "xdg-open"], capture_output=True).returncode == 0:
                cmd = ["xdg-open", path]
            else:
                cmd = ["gio", "open", path]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False


# ======================================================================
# STREAMING OUTPUT WIDGET
# ======================================================================

class StreamingOutput(Vertical):
    """
    Live text output. Appended to as `on_token` events arrive.
    When the turn completes, the app replaces this widget with the
    structured OutputView.
    """

    DEFAULT_CSS = f"""
    StreamingOutput {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {GREEN_DIM};
    }}
    StreamingOutput > Static {{
        height: auto;
        width: 100%;
    }}
    """

    # Coalesce tokens instead of repainting the whole Static on every one.
    # Streaming providers can emit hundreds of tokens per second; updating
    # the widget each time makes the UI quadratic in reply length and
    # starves key handling, which reads as the TUI freezing.
    FLUSH_INTERVAL = 0.05

    def __init__(self, header: str = "◆ depression.ai", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._header = header
        # [Streaming fix] Keep a running buffer + a running *escaped* string
        # so we don't re-join / re-escape the whole transcript on every token.
        self._raw: list[str] = []
        self._escaped: list[str] = []
        self._pending: list[str] = []
        self._flush_timer: Any = None
        self._body: Optional[Static] = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold {GREEN}]{_esc(self._header)}[/] [{DIM}]streaming…[/]",
            markup=True,
        )
        yield Static(
            f"[{BORDER}]" + "─" * 56 + "[/]",
            markup=True,
        )
        yield Static("", id="stream-body", markup=True)

    def on_mount(self) -> None:
        try:
            self._body = self.query_one("#stream-body", Static)
        except Exception:
            self._body = None

    def on_unmount(self) -> None:
        self._cancel_flush_timer()
        self._body = None

    def append_token(self, delta: str) -> None:
        if not delta:
            return
        self._raw.append(delta)
        # Escape only the incoming delta; buffer it until the next repaint.
        self._pending.append(delta.replace("[", r"\["))
        if self._flush_timer is None:
            try:
                self._flush_timer = self.set_timer(
                    self.FLUSH_INTERVAL, self._flush
                )
            except Exception:
                self._flush()

    def _cancel_flush_timer(self) -> None:
        timer = self._flush_timer
        self._flush_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

    def _flush(self) -> None:
        self._flush_timer = None
        if not self._pending:
            return
        self._escaped.extend(self._pending)
        self._pending.clear()
        if self._body is not None:
            try:
                self._body.update("".join(self._escaped))
            except Exception:
                pass
        # Follow the stream here rather than on every token.
        try:
            self.app._scroll_transcript_end()
        except Exception:
            pass

    def flush(self) -> None:
        """Render any buffered tokens immediately."""
        self._cancel_flush_timer()
        self._flush()

    def get_text(self) -> str:
        return "".join(self._raw)


# ======================================================================
# MULTILINE PROMPT
# ======================================================================

class PromptArea(TextArea):
    """Soft-wrapping, auto-growing multiline prompt."""

    DEFAULT_CSS = f"""
    PromptArea {{
        height: auto;
        min-height: {_PROMPT_MIN_ROWS};
        max-height: {_PROMPT_MAX_ROWS + 2};
        width: 1fr;
        background: {BG};
        color: {TEXT};
        border: none;
        padding: 0;
        scrollbar-size-vertical: 1;
        scrollbar-background: {BG};
        scrollbar-color: {BORDER};
    }}
    PromptArea:focus {{ border: none; }}
    PromptArea .text-area--cursor {{ background: {GREEN}; color: {BG}; }}
    PromptArea.busy {{ color: {AMBER}; }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "",
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            **kwargs,
        )

    def on_mount(self) -> None:
        # TextArea was configured with soft_wrap=True; do not override it.
        self._resize()

    def _resize(self) -> None:
        try:
            text = self.text
            width = max(1, self.content_size.width)
            wrapped = 0
            for raw in text.split("\n"):
                if not raw:
                    wrapped += 1
                    continue
                wrapped += max(1, (len(raw) + width - 1) // width)
            rows = max(_PROMPT_MIN_ROWS, min(_PROMPT_MAX_ROWS, wrapped))
            self.styles.height = rows
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._resize()

    def set_prompt_text(self, text: str) -> None:
        """Replace the prompt text and move the cursor to the end."""
        self.text = text or ""
        self._resize()
        # Programmatic updates don't always emit TextArea.Changed.
        self.app._update_command_suggestions(self.text)
        try:
            loc = self.document.get_location_from_index(len(self.text))
            self.cursor_location = loc
        except Exception:
            pass

    def insert_file_reference(self, path: str) -> None:
        """Insert @path at the cursor, replacing an unfinished @ token."""
        text = self.text
        try:
            idx = self.document.get_index_from_location(self.cursor_location)
        except Exception:
            idx = len(text)
        before = text[:idx]
        start = idx
        at = before.rfind("@")
        if at >= 0 and (at == 0 or before[at - 1].isspace()):
            token = before[at:]
            if not any(c.isspace() for c in token):
                start = at
        ref = f"@{path} "
        self.set_prompt_text(text[:start] + ref + text[idx:])
        try:
            loc = self.document.get_location_from_index(start + len(ref))
            self.cursor_location = loc
        except Exception:
            pass

    async def _on_key(self, event) -> None:
        key = event.key
        if getattr(self.app, "_command_menu_open", False):
            if key in ("up", "down"):
                event.stop()
                event.prevent_default()
                self.app._move_command_suggestion(-1 if key == "up" else 1)
                return
            if key == "escape":
                event.stop()
                event.prevent_default()
                self.app._hide_command_suggestions(suppress=True)
                return
            if key == "tab":
                event.stop()
                event.prevent_default()
                self.app._accept_command_suggestion(execute=False)
                return
            if key == "enter":
                event.stop()
                event.prevent_default()
                if self.app._accept_command_suggestion(execute=True):
                    return

        if key in ("up", "down"):
            try:
                at_edge = (
                    self.cursor_at_first_line if key == "up"
                    else self.cursor_at_last_line
                )
            except Exception:
                at_edge = False
            if at_edge:
                handler = getattr(self.app, "navigate_prompt_history", None)
                value = handler(-1 if key == "up" else 1) if handler else None
                if value is not None:
                    event.stop()
                    event.prevent_default()
                    self.set_prompt_text(value)
                    return

        if key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            self._resize()
            return

        if key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text
            self.text = ""
            self._resize()
            try:
                handler = getattr(self.app, "submit_prompt_text", None)
                if handler is not None:
                    handler(text)
            except Exception:
                pass
            return


# ======================================================================
# SMALL MODALS
# ======================================================================

class TextPromptModal(ModalScreen[Optional[str]]):
    """A one-line text prompt used for renaming sessions."""

    DEFAULT_CSS = f"""
    TextPromptModal {{ align: center middle; background: {RAISED} 78%; }}
    #tp-card {{
        width: 64%; min-width: 44; height: auto;
        border: thick {GREEN}; background: {PANEL}; padding: 1 2;
    }}
    #tp-title {{ color: {GREEN}; text-style: bold; margin-bottom: 1; }}
    #tp-input {{
        background: {BG}; border: round {BORDER}; color: {TEXT}; height: 3;
    }}
    #tp-input:focus {{ border: round {GREEN}; }}
    #tp-buttons {{ height: 3; layout: horizontal; margin-top: 1; }}
    #tp-buttons Button {{
        width: 1fr; height: 3; background: transparent;
        border: round {GREEN}; color: {GREEN}; margin-right: 1;
    }}
    #tp-buttons Button:focus {{ background: {GREEN}; color: {BG}; }}
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title: str, value: str = "", placeholder: str = "",
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._title = title
        self._value = value
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="tp-card"):
            yield Static(_esc(self._title), id="tp-title", markup=True)
            yield Input(self._value, placeholder=self._placeholder, id="tp-input")
            with Horizontal(id="tp-buttons"):
                yield Button("SAVE", id="tp-save")
                yield Button("CANCEL", id="tp-cancel")

    def on_mount(self) -> None:
        inp = self.query_one("#tp-input", Input)
        inp.focus()
        try:
            inp.cursor_position = len(inp.value)
        except Exception:
            pass

    def _save(self) -> None:
        value = self.query_one("#tp-input", Input).value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#tp-input")
    def _submitted(self) -> None:
        self._save()

    @on(Button.Pressed, "#tp-save")
    def _save_btn(self) -> None:
        self._save()

    @on(Button.Pressed, "#tp-cancel")
    def _cancel_btn(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Small yes/no confirmation dialog."""

    DEFAULT_CSS = f"""
    ConfirmModal {{ align: center middle; background: {RAISED} 78%; }}
    #cf-card {{
        width: 58%; min-width: 40; height: auto;
        border: thick {ERROR}; background: {PANEL}; padding: 1 2;
    }}
    #cf-title {{ color: {ERROR}; text-style: bold; margin-bottom: 1; }}
    #cf-body {{ color: {TEXT}; margin-bottom: 1; }}
    #cf-buttons {{ height: 3; layout: horizontal; }}
    #cf-buttons Button {{
        width: 1fr; height: 3; background: transparent;
        border: round {BORDER}; color: {TEXT}; margin-right: 1;
    }}
    #cf-yes {{ border: round {ERROR}; color: {ERROR}; }}
    #cf-yes:focus {{ background: {ERROR}; color: {BG}; }}
    """

    BINDINGS = [
        Binding("escape", "no", "No", priority=True),
        Binding("n", "no", "No", priority=True),
        Binding("y", "yes", "Yes", priority=True),
    ]

    def __init__(self, title: str, body: str = "", **kwargs: Any):
        super().__init__(**kwargs)
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="cf-card"):
            yield Static(_esc(self._title), id="cf-title", markup=True)
            if self._body:
                yield Static(_esc(self._body), id="cf-body", markup=True)
            with Horizontal(id="cf-buttons"):
                yield Button("YES", id="cf-yes")
                yield Button("NO", id="cf-no")

    def on_mount(self) -> None:
        try:
            self.query_one("#cf-no", Button).focus()
        except Exception:
            pass

    def action_yes(self) -> None: self.dismiss(True)
    def action_no(self) -> None: self.dismiss(False)

    @on(Button.Pressed, "#cf-yes")
    def _yes_btn(self) -> None: self.dismiss(True)

    @on(Button.Pressed, "#cf-no")
    def _no_btn(self) -> None: self.dismiss(False)


class _PickerModal(ModalScreen[Optional[str]]):
    """Shared filter/list picker used by history and file completion."""

    DEFAULT_CSS = f"""
    _PickerModal {{ align: center middle; background: {RAISED} 78%; }}
    _PickerModal #pick-card {{
        width: 76%; min-width: 46; height: 72%;
        border: thick {GREEN}; background: {PANEL}; padding: 1 2;
    }}
    _PickerModal #pick-title {{ color: {GREEN}; text-style: bold; }}
    _PickerModal #pick-filter {{
        background: {BG}; border: round {BORDER}; color: {TEXT};
        height: 3; margin: 1 0;
    }}
    _PickerModal #pick-filter:focus {{ border: round {GREEN}; }}
    _PickerModal OptionList {{
        height: 1fr; background: {BG}; color: {TEXT};
        border: round {BORDER};
    }}
    _PickerModal OptionList > .option-list--option-highlighted {{
        background: {GREEN_GLOW}; color: {BG};
    }}
    _PickerModal .pick-help {{ color: {MUTED}; height: 1; margin-top: 1; }}
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title: str, items: List[str], placeholder: str = "filter",
                 help_text: str = "type to filter · enter select · esc close",
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._title = title
        self._items = list(items)
        self._placeholder = placeholder
        self._help_text = help_text
        self._filtered = list(self._items)

    def compose(self) -> ComposeResult:
        with Vertical(id="pick-card"):
            yield Static(_esc(self._title), id="pick-title", markup=True)
            yield Input(placeholder=self._placeholder, id="pick-filter")
            yield OptionList(id="pick-results")
            yield Static(self._help_text, classes="pick-help")

    def on_mount(self) -> None:
        self._refresh("")
        try:
            self.query_one("#pick-filter", Input).focus()
        except Exception:
            pass

    def _refresh(self, query: str) -> None:
        q = query.strip().lower()
        self._filtered = [item for item in self._items if q in item.lower()]
        options = [Option(item, id=str(i)) for i, item in enumerate(self._filtered)]
        if not options:
            options = [Option("(no matches)", id="__none__", disabled=True)]
        try:
            self.query_one("#pick-results", OptionList).clear_options()
            self.query_one("#pick-results", OptionList).add_options(options)
        except Exception:
            pass

    def _selected(self) -> Optional[str]:
        try:
            opt = self.query_one("#pick-results", OptionList).highlighted_option
        except Exception:
            return None
        if opt is None or opt.id == "__none__":
            return None
        try:
            return self._filtered[int(str(opt.id))]
        except Exception:
            return None

    def _choose(self, value: Optional[str]) -> None:
        self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Changed, "#pick-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._refresh(event.value)

    @on(Input.Submitted, "#pick-filter")
    def _filter_submitted(self) -> None:
        value = self._selected()
        if value is None and self._filtered:
            value = self._filtered[0]
        self._choose(value)

    @on(OptionList.OptionSelected, "#pick-results")
    def _option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "__none__":
            return
        try:
            self._choose(self._filtered[int(str(event.option.id))])
        except Exception:
            pass


class HistorySearchModal(_PickerModal):
    def __init__(self, prompts: List[str], **kwargs: Any):
        super().__init__(
            "PROMPT HISTORY",
            items=list(reversed(prompts)),
            placeholder="search history",
            help_text="enter select · esc close · ctrl+r reopen",
            **kwargs,
        )


class FilePickerModal(_PickerModal):
    def __init__(self, files: List[str], **kwargs: Any):
        super().__init__(
            "ATTACH A FILE",
            items=files,
            placeholder="filter project files",
            help_text="enter insert @path · esc close",
            **kwargs,
        )


class SessionPickerModal(ModalScreen[Optional[tuple[str, str]]]):
    """List sessions and let the user open, rename, or delete one."""

    DEFAULT_CSS = f"""
    SessionPickerModal {{ align: center middle; background: {RAISED} 78%; }}
    #sp-card {{
        width: 76%; min-width: 48; height: 72%;
        border: thick {GREEN}; background: {PANEL}; padding: 1 2;
    }}
    #sp-title {{ color: {GREEN}; text-style: bold; }}
    #sp-list {{
        height: 1fr; margin-top: 1; background: {BG}; color: {TEXT};
        border: round {BORDER};
    }}
    #sp-list > .option-list--option-highlighted {{
        background: {GREEN_GLOW}; color: {BG};
    }}
    #sp-buttons {{ height: 3; layout: horizontal; margin-top: 1; }}
    #sp-buttons Button {{
        width: 1fr; height: 3; background: transparent;
        border: round {GREEN}; color: {GREEN}; margin-right: 1;
    }}
    #sp-buttons Button:focus {{ background: {GREEN}; color: {BG}; }}
    #sp-delete {{ border: round {ERROR}; color: {ERROR}; }}
    #sp-help {{ color: {MUTED}; height: 1; margin-top: 1; }}
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", priority=True),
        Binding("o", "open_selected", "Open", priority=True),
        Binding("r", "rename_selected", "Rename", priority=True),
        Binding("d", "delete_selected", "Delete", priority=True),
    ]

    def __init__(self, sessions: List[Dict[str, Any]], **kwargs: Any):
        super().__init__(**kwargs)
        self._sessions = list(sessions)

    def compose(self) -> ComposeResult:
        with Vertical(id="sp-card"):
            yield Static("SESSIONS", id="sp-title", markup=True)
            yield OptionList(id="sp-list")
            with Horizontal(id="sp-buttons"):
                yield Button("OPEN", id="sp-open")
                yield Button("RENAME", id="sp-rename")
                yield Button("DELETE", id="sp-delete")
            yield Static(
                "enter open · r rename · d delete · esc close",
                id="sp-help",
            )

    def on_mount(self) -> None:
        options: List[Option] = []
        for s in self._sessions:
            sid = str(s.get("id") or "")
            if not sid:
                continue
            name = str(s.get("name") or "session")
            updated = s.get("updated_at") or 0
            when = ""
            try:
                when = datetime.fromtimestamp(float(updated)).strftime("%m-%d %H:%M")
            except Exception:
                when = "—"
            current = " *" if s.get("current") else ""
            options.append(Option(
                f"{name}{current}  ·  {when}  ·  {sid[:8]}", id=sid,
            ))
        if not options:
            options = [Option("(no saved sessions)", id="__none__", disabled=True)]
        lst = self.query_one("#sp-list", OptionList)
        lst.add_options(options)
        lst.focus()

    def _selected_id(self) -> Optional[str]:
        try:
            opt = self.query_one("#sp-list", OptionList).highlighted_option
        except Exception:
            return None
        if opt is None or opt.id == "__none__":
            return None
        return str(opt.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_open_selected(self) -> None:
        sid = self._selected_id()
        if sid:
            self.dismiss(("open", sid))

    def action_rename_selected(self) -> None:
        sid = self._selected_id()
        if sid:
            self.dismiss(("rename", sid))

    def action_delete_selected(self) -> None:
        sid = self._selected_id()
        if sid:
            self.dismiss(("delete", sid))

    @on(OptionList.OptionSelected, "#sp-list")
    def _list_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id != "__none__":
            self.dismiss(("open", str(event.option.id)))

    @on(Button.Pressed, "#sp-open")
    def _open_btn(self) -> None: self.action_open_selected()

    @on(Button.Pressed, "#sp-rename")
    def _rename_btn(self) -> None: self.action_rename_selected()

    @on(Button.Pressed, "#sp-delete")
    def _delete_btn(self) -> None: self.action_delete_selected()


class TranscriptSearchModal(TextPromptModal):
    def __init__(self, **kwargs: Any):
        super().__init__(
            "SEARCH TRANSCRIPT",
            placeholder="text to find",
            **kwargs,
        )


# ======================================================================
# COMMAND PALETTE
# ======================================================================

class AgentCommandProvider(Provider):
    """Expose slash commands in Textual's ctrl+p command palette."""

    COMMANDS = [
        ("/connect", "Configure the LLM connection"),
        ("/model", "Paste your model ID into the LLM panel"),
        ("/models", "Discover models from the provider"),
        ("/sessions", "Open saved sessions"),
        ("/retry", "Retry the last prompt"),
        ("/continue", "Continue after a timeout or stop"),
        ("/copy", "Copy the last agent reply"),
        ("/export", "Export the transcript as Markdown"),
        ("/search", "Search the transcript"),
        ("/doctor", "Run local diagnostics"),
        ("/aws", "Open AWS credentials"),
        ("/mcp", "Show MCP server status and tools"),
        ("/profile", "Open profile"),
        ("/context", "Open context usage"),
        ("/llmcost", "Open token and API cost"),
        ("/todo", "Open the todo list"),
        ("/run", "Run a shell command (streams until /stop if long-running)"),
        ("/shot", "Open an image with the OS viewer"),
        ("/plan", "Switch to plan mode"),
        ("/build", "Switch to build mode"),
        ("/auto", "Switch to auto mode"),
        ("/clear", "Clear the transcript"),
        ("/help", "Open help"),
        ("/quit", "Quit"),
        ("/exit", "Quit (alias)"),
        ("/stop", "Stop the running task"),
        ("/interrupt", "Stop the running task (alias)"),
    ]

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for slash, help_text in self.COMMANDS:
            name = slash.lstrip("/")
            score = matcher.match(name)
            if score <= 0:
                continue
            yield Hit(
                score,
                f"{slash}  ·  {help_text}",
                lambda s=slash: self.app.run_slash_command(s),
                text=slash,
                help=help_text,
            )


# ======================================================================
# INLINE SLASH-COMMAND MATCHING (same catalog as Ctrl+P)
# ======================================================================

_SLASH_ARGUMENT_COMMANDS = frozenset({"/run", "/shot", "/search", "/export"})
_SLASH_SUGGESTION_LIMIT = 7


def _slash_command_matches(
    text: str, limit: int = _SLASH_SUGGESTION_LIMIT,
) -> List[tuple[str, str]]:
    """Rank actual slash commands by prefix, substring, then fuzzy match.

    No LLM call and no speculative suggestions: every result is an
    executable command in AgentCommandProvider.COMMANDS.
    """
    if not re.fullmatch(r"/[a-zA-Z0-9_-]*", text or ""):
        return []
    needle = text[1:].casefold()
    found: List[tuple[int, int, str, str]] = []
    for order, (command, description) in enumerate(AgentCommandProvider.COMMANDS):
        word = command.lstrip("/").casefold()
        if not needle or word.startswith(needle):
            rank = 0
        elif needle in word:
            rank = 1
        else:
            chars = iter(word)
            if not all(any(char == c for c in chars) for char in needle):
                continue
            rank = 2
        found.append((rank, order, command, description))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(command, description) for _, _, command, description in found[:max(1, limit)]]


# ======================================================================
# LLM PANEL
# ======================================================================

class LLMPanel(Vertical):
    DEFAULT_CSS = f"""
    LLMPanel {{
        width: 100%; height: auto;
        padding: 0 1; background: {BG};
    }}
    LLMPanel .label {{ color: {MUTED}; margin-top: 1; }}
    LLMPanel Input {{
        background: {PANEL}; border: round {BORDER};
        color: {TEXT}; margin: 0 0 1 0; height: 3; width: 100%;
    }}
    LLMPanel Input:focus {{ border: round {GREEN}; }}
    LLMPanel Select {{
        background: {PANEL}; border: round {BORDER};
        color: {TEXT}; margin: 0 0 1 0; height: 3; width: 100%;
    }}
    LLMPanel Select:focus {{ border: round {GREEN}; }}
    LLMPanel Button {{
        width: 1fr; min-width: 12;
        background: transparent; border: round {GREEN};
        color: {GREEN}; margin-top: 1; height: 3; text-style: bold;
    }}
    LLMPanel Button:hover, LLMPanel Button:focus {{
        background: {RAISED}; color: {GREEN_GLOW}; border: round {GREEN_GLOW};
    }}
    LLMPanel #connect {{ text-style: bold; }}
    LLMPanel #llm-actions {{ height: auto; }}
    LLMPanel #llm-actions Button {{ width: 1fr; min-width: 0; margin-top: 0; }}
    LLMPanel .system {{ margin-bottom: 1; }}
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
        yield Button("TEST CONNECTION", id="llm-test")
        yield Static("", id="llm-status", classes="system")
        yield Static("", id="llm-provider", classes="system")


# ======================================================================
# CONTEXT PANEL
# ======================================================================

class ContextPanel(Vertical):
    DEFAULT_CSS = f"""
    ContextPanel {{
        width: 100%; height: auto;
        padding: 0 1; background: {BG};
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
        self.set_interval(1.0, self.refresh_values)

    def _live_ctx(self) -> Any:
        # [Token-count fix] Same probe as LLMCostPanel so we see live tokens
        # instead of only the post-turn snapshot.
        return _probe_loop_context(self._app.coordinator)

    def refresh_values(self) -> None:
        try:
            model = _esc(self._app.cfg.get("model") or "—")
            self.query_one("#ctx-model", Static).update(
                f"[{MUTED}]model[/]   [{TEXT}]{model}[/]")
            self.query_one("#ctx-window", Static).update(
                f"[{MUTED}]window[/]  [{TEXT}]{self._app.context_window:,}[/]")

            live = self._live_ctx()
            used = 0
            if live is not None:
                used = int(_get(live, "tokens_used", "total_tokens", default=0) or 0)
                if not used:
                    used = (
                        int(_get(live, "input_tokens", "prompt_tokens", default=0) or 0)
                        + int(_get(live, "output_tokens", "completion_tokens", default=0) or 0)
                    )
            if not used:
                used = int(self._app.tokens_used or 0)

            # The live loop token counter is cumulative billing usage, not
            # current prompt occupancy. Display retained context separately.
            active = None
            coordinator = getattr(self._app, "coordinator", None)
            agent = getattr(coordinator, "build_agent", None)
            if getattr(coordinator, "current_mode", None) == "plan":
                agent = getattr(coordinator, "plan_agent", None)
            manager = getattr(agent, "context_manager", None)
            if manager is not None:
                try:
                    active = int(manager.total_tokens())
                except Exception:
                    pass
            window = max(1, self._app.context_window)
            pct = min(100.0, ((active if active is not None else used) / window) * 100.0)
            bar_w = 30
            filled = int(bar_w * pct / 100.0)
            bar = "█" * filled + "░" * (bar_w - filled)
            color = GREEN if pct < 60 else AMBER if pct < 85 else ERROR
            self.query_one("#ctx-bar", Static).update(f"[{color}]{bar}[/]")
            self.query_one("#ctx-tokens", Static).update(
                f"[{MUTED}]context[/] [{TEXT}]{(active if active is not None else used):,} / {window:,} "
                f"([{color}]{pct:.1f}%[/])  total used: {used:,}[/]")

            cost = 0.0
            if live is not None:
                cost = float(_get(live, "cost", default=0.0) or 0.0)
            if not cost:
                cost = float(self._app.cost or 0.0)
            self.query_one("#ctx-cost", Static).update(
                f"[{MUTED}]cost[/]    [{TEXT}]${cost:.4f}[/]")
            self.query_one("#ctx-msgs", Static).update(
                f"[{MUTED}]turns[/]   [{TEXT}]{self._app.message_count}[/]")
        except Exception:
            pass


# ======================================================================
# TODO PANEL
# ======================================================================

class TodoPanel(Vertical):
    """Sidebar Todo panel."""

    DEFAULT_CSS = f"""
    TodoPanel {{
        width: 100%; height: auto;
        padding: 0 1; background: {BG};
    }}
    TodoPanel .hint {{ color: {MUTED}; margin-top: 1; }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref
        self._entries: list[dict] = []
        self._store_sig: tuple = ()
        self._entries_from_store = False

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ TODO[/]", markup=True)
        yield Static(f"[{MUTED}]— no active plan —[/]",
                     id="todo-body", markup=True)
        yield Static("updates as the agent works",
                     classes="hint")

    def on_mount(self) -> None:
        self.refresh_from_store(force=True)
        self.set_interval(1.0, self.refresh_from_store)

    def set_entries(self, entries: list[dict]) -> None:
        # [Todo fix] Accept empty lists so the panel clears correctly.
        self._entries = list(entries or [])
        self._entries_from_store = False
        # Remember the store state the pushed entries correspond to so the
        # poll below does not immediately overwrite a model-authored
        # checklist with an older store bucket.
        self._store_sig = self._current_store_signature()
        self._paint_entries()

    def refresh_from_store(self, force: bool = False) -> None:
        """Refresh from the shared TodoTool store (polled every 0.5s)."""
        store = self._read_store()
        sig = self._signature(store)
        if not force and sig == self._store_sig:
            return
        self._store_sig = sig
        if not store:
            if self._entries_from_store:
                self._entries = []
                self._entries_from_store = False
                self._paint_entries()
            return
        self._entries = [self._store_entry(item) for item in store.values()]
        self._entries_from_store = True
        self._paint_entries()

    def _read_store(self) -> dict:
        try:
            from agent.tools.todo import TodoTool
            session = self._app._session() if self._app is not None else None
            sid = TodoTool._session_key(session)
            return TodoTool._strong_keys.get(sid) or {}
        except Exception:
            return {}

    @staticmethod
    def _signature(store: dict) -> tuple:
        return tuple(
            (
                getattr(item, "id", ""),
                getattr(item, "status", ""),
                getattr(item, "title", ""),
                getattr(item, "updated_at", 0),
            )
            for item in store.values()
        )

    def _current_store_signature(self) -> tuple:
        return self._signature(self._read_store())

    @staticmethod
    def _store_entry(item: Any) -> dict:
        status = str(getattr(item, "status", "pending") or "pending").lower()
        if status == "done":
            status = "completed"
        return {
            "content": str(getattr(item, "title", "") or ""),
            "status": status,
            "priority": getattr(item, "priority", 3),
        }

    def _paint_entries(self) -> None:
        try:
            if not self._entries:
                self.query_one("#todo-body", Static).update(
                    f"[{MUTED}]— no active plan —[/]"
                )
                return
            self.query_one("#todo-body", Static).update(
                self._format_entries(self._entries)
            )
        except Exception:
            pass

    @staticmethod
    def _format_entries(entries: list[dict]) -> str:
        icons = {
            "completed": f"[{GREEN}]✓[/]",
            "in_progress": f"[{AMBER}]▸[/]",
            "pending": f"[{DIM}]○[/]",
            "failed": f"[{ERROR}]✗[/]",
            "blocked": f"[{ERROR}]■[/]",
            "cancelled": f"[{DIM}]✕[/]",
            "skipped": f"[{DIM}]—[/]",
        }
        out = []
        for e in entries:
            content = str(e.get("content") or e.get("description") or "")
            content = _esc(content)
            status = str(e.get("status") or "pending").lower()
            icon = icons.get(status, f"[{DIM}]○[/]")
            if status == "completed":
                out.append(f"  {icon} [{MUTED}]{content}[/]")
            elif status == "in_progress":
                out.append(f"  {icon} [bold {TEXT}]{content}[/]")
            elif status in ("failed", "blocked"):
                out.append(f"  {icon} [{ERROR}]{content}[/]")
            elif status == "cancelled":
                out.append(f"  {icon} [{DIM}]{content}[/]")
            else:
                out.append(f"  {icon} [{TEXT}]{content}[/]")
        return "\n".join(out)


# ======================================================================
# LLM COST PANEL
# ======================================================================

def _probe_loop_context(coordinator: Any) -> Any:
    """
    Best-effort lookup of the live AgentLoop.context for whatever agent is
    currently active on the coordinator.

    Probes, in order:
      0. auto mode: the most recently started plan/build loop
      1. coordinator.current_agent            (attribute)
      2. coordinator.get_current_agent()      (method)
      3. coordinator.build_agent              (attribute)
      4. coordinator.plan_agent               (attribute)
      5. agent.loop.context
      6. agent.context_manager (with usage attrs)
    """
    if coordinator is None:
        return None

    # Auto mode runs plan and build in sequence. `get_current_agent()`
    # always returns the build agent there, which is stale while the plan
    # phase is running. Pick the loop that started most recently instead.
    if getattr(coordinator, "current_mode", None) == "auto":
        best_ctx = None
        best_start = float("-inf")
        for attr in ("plan_agent", "build_agent"):
            candidate = getattr(coordinator, attr, None)
            loop = getattr(candidate, "loop", None) if candidate is not None else None
            ctx = getattr(loop, "context", None) if loop is not None else None
            if ctx is None:
                continue
            try:
                start = float(getattr(ctx, "start_time", 0.0) or 0.0)
            except Exception:
                start = 0.0
            if start >= best_start:
                best_start, best_ctx = start, ctx
        if best_ctx is not None:
            return best_ctx

    agent = None

    # 1. direct attribute
    agent = getattr(coordinator, "current_agent", None)

    # 2. method
    if agent is None:
        fn = getattr(coordinator, "get_current_agent", None)
        if callable(fn):
            try:
                agent = fn()
            except Exception:
                agent = None

    # 3/4. explicit agents
    if agent is None:
        for attr in ("build_agent", "plan_agent"):
            candidate = getattr(coordinator, attr, None)
            if candidate is not None:
                agent = candidate
                break

    if agent is None:
        return None

    loop = getattr(agent, "loop", None)
    if loop is not None:
        ctx = getattr(loop, "context", None)
        if ctx is not None:
            return ctx

    # 6. fallback: context_manager with usage attributes
    cm = getattr(agent, "context_manager", None)
    if cm is not None:
        if any(
            hasattr(cm, k) for k in
            ("tokens_used", "input_tokens", "output_tokens", "llm_calls")
        ):
            return cm

    return None


def _usage_empty() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_used": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "cost": 0.0,
        "duration": 0.0,
    }


def _usage_float(obj: Any, *names: str) -> float:
    for name in names:
        value = _get(obj, name)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _usage_int(obj: Any, *names: str) -> int:
    for name in names:
        value = _get(obj, name)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def _usage_snapshot(obj: Any, *, now: Optional[float] = None) -> Dict[str, Any]:
    """
    Read one usage-bearing object (AgentLoop.context or Agent.context).

    Loop contexts expose `actions_taken`; session contexts expose a
    `tools_used` map. Both are normalised to the same shape.
    """
    snap = _usage_empty()
    if obj is None:
        return snap

    snap["input_tokens"] = _usage_int(obj, "input_tokens", "prompt_tokens")
    snap["output_tokens"] = _usage_int(obj, "output_tokens", "completion_tokens")
    snap["tokens_used"] = (
        _usage_int(obj, "tokens_used", "total_tokens")
        or snap["input_tokens"] + snap["output_tokens"]
    )
    snap["llm_calls"] = _usage_int(obj, "llm_calls", "api_calls")
    snap["cost"] = _usage_float(obj, "cost", "total_cost")

    actions = _get(obj, "actions_taken")
    if isinstance(actions, list):
        snap["tool_calls"] = len(actions)
    else:
        tools_used = _get(obj, "tools_used")
        if isinstance(tools_used, dict) and tools_used:
            total = 0
            for value in tools_used.values():
                try:
                    total += int(value)
                except Exception:
                    continue
            snap["tool_calls"] = total
        else:
            snap["tool_calls"] = _usage_int(obj, "tool_calls", "total_tool_calls")
        snap["duration"] = _usage_float(obj, "duration")

    start = _usage_float(obj, "start_time")
    if now is not None and start:
        snap["duration"] = max(0.0, now - start)

    return snap


def _usage_add(total: Dict[str, Any], part: Dict[str, Any]) -> None:
    for key in total:
        if key == "cost" or key == "duration":
            total[key] += float(part.get(key, 0.0) or 0.0)
        else:
            total[key] += int(part.get(key, 0) or 0)


def _usage_agents(coordinator: Any) -> List[Any]:
    """Return distinct plan/build agents, falling back to the active agent."""
    if coordinator is None:
        return []

    agents: List[Any] = []
    seen: set = set()
    for attr in ("plan_agent", "build_agent"):
        agent = getattr(coordinator, attr, None)
        if agent is not None and id(agent) not in seen:
            seen.add(id(agent))
            agents.append(agent)
    if agents:
        return agents

    agent = getattr(coordinator, "current_agent", None)
    if agent is None:
        fn = getattr(coordinator, "get_current_agent", None)
        if callable(fn):
            try:
                agent = fn()
            except Exception:
                agent = None
    return [agent] if agent is not None else []


def _query_usage(coordinator: Any, query_start_ts: float) -> Dict[str, Any]:
    """
    Usage for the current top-level query.

    Every AgentLoop resets its context when a run starts, so summing the
    contexts whose `start_time` is at/after the query marker captures all
    phases of an auto run (plan + build) without leaking the previous turn.
    """
    total = _usage_empty()
    if coordinator is None:
        return total

    if not query_start_ts:
        return _usage_snapshot(_probe_loop_context(coordinator))

    found = False
    for agent in _usage_agents(coordinator):
        loop = getattr(agent, "loop", None)
        ctx = getattr(loop, "context", None) if loop is not None else None
        if ctx is None:
            continue
        start = _usage_float(ctx, "start_time")
        if start and start < query_start_ts - 0.001:
            continue
        _usage_add(total, _usage_snapshot(ctx))
        found = True

    if not found:
        ctx = _probe_loop_context(coordinator)
        start = _usage_float(ctx, "start_time")
        if ctx is not None and (not start or start >= query_start_ts - 0.001):
            _usage_add(total, _usage_snapshot(ctx))
    return total


def _session_usage(coordinator: Any, now: Optional[float] = None) -> Dict[str, Any]:
    """
    Session totals across both agents.

    `Agent.context` only receives a run's usage after that run completes, so
    add the live loop context while a query is in flight. `_query_started_at`
    is set for the duration of `_run_query_pipeline`, which avoids counting
    the previous run twice.
    """
    total = _usage_empty()
    if coordinator is None:
        return total

    current = time.time() if now is None else now
    for agent in _usage_agents(coordinator):
        _usage_add(total, _usage_snapshot(getattr(agent, "context", None)))

        query_started = _usage_float(agent, "_query_started_at")
        if not query_started:
            continue
        loop = getattr(agent, "loop", None)
        ctx = getattr(loop, "context", None) if loop is not None else None
        if ctx is None:
            continue
        start = _usage_float(ctx, "start_time")
        if start and start < query_started - 0.001:
            continue
        _usage_add(total, _usage_snapshot(ctx, now=current))
    return total


def _query_elapsed(
    active: bool,
    start_ts: float,
    frozen: float,
    now: Optional[float] = None,
) -> float:
    if active and start_ts:
        current = time.time() if now is None else now
        return max(0.0, current - start_ts)
    return max(0.0, frozen)


class LLMCostPanel(Vertical):
    DEFAULT_CSS = f"""
    LLMCostPanel {{
        width: 100%; height: auto;
        padding: 0 1; background: {BG};
    }}
    LLMCostPanel .label  {{ color: {MUTED}; height: 1; }}
    LLMCostPanel .value  {{ color: {TEXT}; height: 1; }}
    LLMCostPanel .head   {{ color: {GREEN}; text-style: bold; height: 1; margin: 1 0 0 0; }}
    LLMCostPanel .accent {{ color: {GREEN_GLOW}; height: 1; }}
    LLMCostPanel .dim    {{ color: {DIM}; height: 1; }}
    """

    def __init__(self, app_ref: "DepressionApp", **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref
        self._query_start_ts: float = 0.0
        self._query_active: bool = False
        self._frozen_dur: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ LLM COST[/]", markup=True)
        yield Static("dir", classes="label")
        yield Static("", id="cost-dir", classes="value")
        yield Static("data", classes="label")
        yield Static("", id="cost-data", classes="value")
        yield Static("model", classes="label")
        yield Static("", id="cost-model", classes="value")
        yield Static("provider", classes="label")
        yield Static("", id="cost-provider", classes="value")
        yield Static("session", classes="label")
        yield Static("", id="cost-session", classes="value")
        yield Static("── this query ──", classes="head")
        yield Static("", id="cost-q-in",    classes="value")
        yield Static("", id="cost-q-out",   classes="value")
        yield Static("", id="cost-q-total", classes="accent")
        yield Static("", id="cost-q-calls", classes="value")
        yield Static("", id="cost-q-tools", classes="value")
        yield Static("", id="cost-q-dur",   classes="value")
        yield Static("", id="cost-q-tps",   classes="dim")
        yield Static("── session ──", classes="head")
        yield Static("", id="cost-s-in",    classes="value")
        yield Static("", id="cost-s-out",   classes="value")
        yield Static("", id="cost-s-total", classes="accent")
        yield Static("", id="cost-s-calls", classes="value")
        yield Static("", id="cost-s-tools", classes="value")
        yield Static("", id="cost-s-turns", classes="value")
        yield Static("", id="cost-s-tps",   classes="dim")

    def on_mount(self) -> None:
        self.refresh_values()
        self.set_interval(1.0, self.refresh_values)

    def _coordinator_status(self) -> dict:
        try:
            coord = getattr(self._app, "coordinator", None)
            if coord is None:
                return {}
            fn = getattr(coord, "get_status", None)
            if callable(fn):
                st = fn() or {}
                if isinstance(st, dict):
                    return st
        except Exception:
            pass
        return {}

    def _loop_context(self) -> Any:
        return _probe_loop_context(getattr(self._app, "coordinator", None))

    @staticmethod
    def _num(obj: Any, *names: str) -> int:
        if obj is None:
            return 0
        for n in names:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return 0

    @staticmethod
    def _flt(obj: Any, *names: str) -> float:
        if obj is None:
            return 0.0
        for n in names:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _str(obj: Any, *names: str, default: str = "—") -> str:
        if obj is None:
            return default
        for n in names:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v:
                return str(v)
        return default

    def mark_query_start(self) -> None:
        self._query_start_ts = time.time()
        self._query_active = True
        self._frozen_dur = 0.0
        self.refresh_values()

    def mark_query_end(self) -> None:
        if self._query_active and self._query_start_ts:
            self._frozen_dur = time.time() - self._query_start_ts
        self._query_active = False
        self.refresh_values()

    def refresh_values(self) -> None:
        coordinator = getattr(self._app, "coordinator", None)
        status = self._coordinator_status()

        project_dir = self._str(status, "project_dir", "workspace_dir", "cwd", default="")
        data_dir = self._str(status, "data_dir", "home_dir", default="")
        self._update("#cost-dir", f"[{TEXT}]{_esc(_shorten_path(project_dir))}[/]")
        self._update("#cost-data", f"[{TEXT}]{_esc(_shorten_path(data_dir))}[/]")

        model = self._str(
            status, "model", "model_name",
            default=(
                str(getattr(self._app, "cfg", {}).get("model") or "—")
                if isinstance(getattr(self._app, "cfg", None), dict) else "—"
            ),
        )
        provider = self._str(status, "provider", "provider_name", default="—")
        session_id = self._str(status, "session_id", "session",
                               "current_session_id", default="—")
        if len(model) > 30:
            model = model[:29] + "…"
        if len(session_id) > 16:
            session_id = session_id[:16]
        self._update("#cost-model", f"[{TEXT}]{_esc(model)}[/]")
        self._update("#cost-provider", f"[{TEXT}]{_esc(provider)}[/]")
        self._update("#cost-session", f"[{TEXT}]{_esc(session_id)}[/]")

        # ---- this query ----
        query = _query_usage(coordinator, self._query_start_ts)
        if not any(
            query[k] for k in
            ("input_tokens", "output_tokens", "tokens_used", "llm_calls", "cost")
        ):
            query["input_tokens"] = self._num(status, "query_input_tokens")
            query["output_tokens"] = self._num(status, "query_output_tokens")
            query["tokens_used"] = (
                self._num(status, "query_tokens")
                or query["input_tokens"] + query["output_tokens"]
            )
            query["llm_calls"] = self._num(status, "query_llm_calls")
            query["tool_calls"] = self._num(status, "query_tool_calls")
            query["cost"] = self._flt(status, "query_cost")

        qi = int(query["input_tokens"])
        qo = int(query["output_tokens"])
        qt = int(query["tokens_used"])
        qc = int(query["llm_calls"])
        qtools = int(query["tool_calls"])
        qdur = _query_elapsed(
            self._query_active, self._query_start_ts, self._frozen_dur,
        )
        tps = (qo / qdur) if qdur > 0 and qo else 0.0

        self._update("#cost-q-in", f"[{MUTED}]in[/]      [{TEXT}]{_fmt(qi)}[/]")
        self._update("#cost-q-out", f"[{MUTED}]out[/]     [{TEXT}]{_fmt(qo)}[/]")
        self._update("#cost-q-total", f"[{MUTED}]total[/]   [{GREEN}]{_fmt(qt)}[/]")
        self._update("#cost-q-calls", f"[{MUTED}]api calls[/]  [{AMBER}]{_fmt(qc)}[/]")
        self._update("#cost-q-tools", f"[{MUTED}]tool calls[/] [{TEXT}]{_fmt(qtools)}[/]")
        self._update("#cost-q-dur", f"[{MUTED}]duration[/]  [{TEXT}]{qdur:.1f}s[/]")
        self._update("#cost-q-tps", f"[{MUTED}]tok/s[/]     [{TEXT}]{tps:.1f}[/]")

        # ---- session ----
        # Agent.context holds session-accumulated counters. The active run's
        # loop context is added while it is in flight, then folded into the
        # session totals when the run returns.
        session = _session_usage(coordinator)
        si = int(session["input_tokens"])
        so = int(session["output_tokens"])
        st = int(session["tokens_used"])
        sc = int(session["llm_calls"])
        stools = int(session["tool_calls"])
        sdur = float(session["duration"])

        if not _usage_agents(coordinator) and status:
            si = self._num(status, "input_tokens", "prompt_tokens",
                           "total_input_tokens")
            so = self._num(status, "output_tokens", "completion_tokens",
                           "total_output_tokens")
            st = self._num(status, "tokens_used", "total_tokens") or (si + so)
            sc = self._num(status, "llm_calls", "api_calls",
                           "total_llm_calls")
            tools_used = status.get("tools_used") or status.get("tool_calls")
            if isinstance(tools_used, dict):
                stools = sum(int(v) for v in tools_used.values())
            elif isinstance(tools_used, (int, float)):
                stools = int(tools_used)
            if not stools:
                stools = self._num(status, "total_tool_calls", "tool_calls_total")

        if st == 0:
            st = int(getattr(self._app, "tokens_used", 0) or 0)
            si = si or int(getattr(self._app, "live_input_tokens", 0) or 0)
            so = so or int(getattr(self._app, "live_output_tokens", 0) or 0)

        # User turns come from the TUI, not from plan/build agent turn
        # counters (auto mode counts internal phases as turns).
        sturns = int(getattr(self._app, "message_count", 0) or 0)
        if not sturns and status:
            sturns = self._num(status, "user_turns", "message_count")

        stps = (so / sdur) if sdur > 0 and so else 0.0

        self._update("#cost-s-in", f"[{MUTED}]in[/]      [{TEXT}]{_fmt(si)}[/]")
        self._update("#cost-s-out", f"[{MUTED}]out[/]     [{TEXT}]{_fmt(so)}[/]")
        self._update("#cost-s-total", f"[{MUTED}]total[/]   [{GREEN}]{_fmt(st)}[/]")
        self._update("#cost-s-calls", f"[{MUTED}]api calls[/]  [{AMBER}]{_fmt(sc)}[/]")
        self._update("#cost-s-tools", f"[{MUTED}]tool calls[/] [{TEXT}]{_fmt(stools)}[/]")
        self._update("#cost-s-turns", f"[{MUTED}]turns[/]     [{TEXT}]{_fmt(sturns)}[/]")
        self._update("#cost-s-tps", f"[{MUTED}]tok/s[/]     [{TEXT}]{stps:.1f}[/]")

    def _update(self, selector: str, markup: str) -> None:
        try:
            self.query_one(selector, Static).update(markup)
        except Exception:
            pass


# ======================================================================
# HELP PANEL
# ======================================================================

class HelpPanel(Vertical):
    DEFAULT_CSS = f"""
    HelpPanel {{
        width: 100%; height: auto;
        padding: 0 1; background: {BG};
    }}
    HelpPanel .help-section {{ color: {GREEN}; text-style: bold; margin: 1 0 0 0; }}
    HelpPanel .help-row     {{ color: {TEXT}; }}
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {GREEN}]▌ HELP[/]", markup=True)
        yield Static("Commands", classes="help-section")
        yield Static("/connect   open LLM panel", classes="help-row")
        yield Static("/model     edit LLM model ID", classes="help-row")
        yield Static("/models    discover models", classes="help-row")
        yield Static("/aws       open AWS panel", classes="help-row")
        yield Static("/mcp       open MCP status and reconnect panel", classes="help-row")
        yield Static("/profile   open profile panel", classes="help-row")
        yield Static("/context   open context panel", classes="help-row")
        yield Static("/llmcost   open token + api cost panel", classes="help-row")
        yield Static("/todo      open todo panel", classes="help-row")
        yield Static("/sessions  open, rename or delete saved sessions", classes="help-row")
        yield Static("/retry     retry the last prompt", classes="help-row")
        yield Static("/continue  continue after a timeout or stop", classes="help-row")
        yield Static("/copy      copy the last agent reply", classes="help-row")
        yield Static("/export    export the transcript as Markdown", classes="help-row")
        yield Static("/search    search the transcript", classes="help-row")
        yield Static("/doctor    run local diagnostics", classes="help-row")
        yield Static("/run CMD   run a shell command (long-running output streams; /stop ends it)", classes="help-row")
        yield Static("/shot PATH open a screenshot with the OS viewer", classes="help-row")
        yield Static("/plan      switch mode → plan", classes="help-row")
        yield Static("/build     switch mode → build", classes="help-row")
        yield Static("/auto      switch mode → auto (plan → build loop)", classes="help-row")
        yield Static("/clear     clear transcript", classes="help-row")
        yield Static("/quit      exit", classes="help-row")
        yield Static("CLI Flags", classes="help-section")
        yield Static("depression --nosignin   start without Google sign-in / onboarding", classes="help-row")
        yield Static("Keys", classes="help-section")
        yield Static("Enter      submit prompt", classes="help-row")
        yield Static("Shift+Ent  newline inside the prompt", classes="help-row")
        yield Static("Ctrl+J     newline (fallback)", classes="help-row")
        yield Static("Up / Down  browse prompt history", classes="help-row")
        yield Static("Ctrl+R     search prompt history", classes="help-row")
        yield Static("Ctrl+P     command palette", classes="help-row")
        yield Static("Ctrl+Space complete @file references", classes="help-row")
        yield Static("Ctrl+F     search the transcript", classes="help-row")
        yield Static("F2         pause / resume transcript auto-scroll", classes="help-row")
        yield Static("Tab        focus prompt / cycle mode", classes="help-row")
        yield Static("Ctrl+B     show / hide the sidebar", classes="help-row")
        yield Static("Esc        interrupt running agent", classes="help-row")
        yield Static("Ctrl+E     expand / collapse the last tool block", classes="help-row")
        yield Static("Ctrl+L     clear transcript", classes="help-row")
        yield Static("Ctrl+C     interrupt if busy, quit if idle", classes="help-row")
        yield Static("Ctrl+Q     quit", classes="help-row")


# ======================================================================
# PERMISSION MODAL
# ======================================================================

class PermissionModal(ModalScreen[Optional[str]]):
    """Ask for permission and return once/session/deny."""

    DEFAULT_CSS = f"""
    PermissionModal {{
        align: center middle;
        background: {RAISED} 78%;
    }}
    #perm-card {{
        width: 76%; min-width: 48; height: 78%;
        border: thick {GREEN}; background: {PANEL};
        padding: 1 2; layout: vertical;
    }}
    #perm-body {{
        height: 1fr; width: 100%;
        background: {BG}; border: round {BORDER};
        padding: 0 1; scrollbar-size-vertical: 1;
    }}
    #perm-buttons {{ height: 3; layout: horizontal; margin-top: 1; }}
    #perm-allow {{
        width: 1fr; height: 3; background: transparent;
        border: round {GREEN}; color: {GREEN}; text-style: bold; margin-right: 1;
    }}
    #perm-allow:focus {{ background: {GREEN}; color: {BG}; }}
    #perm-allow-session {{
        width: 1fr; height: 3; background: transparent;
        border: round {AMBER}; color: {AMBER}; text-style: bold; margin-right: 1;
    }}
    #perm-allow-session:focus {{ background: {AMBER}; color: {BG}; }}
    #perm-deny {{
        width: 1fr; height: 3; background: transparent;
        border: round {ERROR}; color: {ERROR}; text-style: bold;
    }}
    #perm-deny:focus {{ background: {ERROR}; color: {BG}; }}
    """

    BINDINGS = [
        Binding("y", "allow", "Allow once", priority=True),
        Binding("a", "allow", "Allow once", priority=True),
        Binding("s", "allow_session", "Allow tool", priority=True),
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

    @staticmethod
    def _details(request: Any) -> str:
        params = getattr(request, "params", None) or {}
        if not isinstance(params, dict):
            return str(params)

        command = (
            params.get("command") or params.get("cmd")
            or params.get("script") or params.get("shell")
        )
        if command:
            return f"$ {command}"

        old = next((params.get(k) for k in (
            "old_string", "oldString", "old_content", "before", "oldText",
        ) if isinstance(params.get(k), str)), None)
        new = next((params.get(k) for k in (
            "new_string", "newString", "new_content", "content", "after",
            "newText",
        ) if isinstance(params.get(k), str)), None)
        path = params.get("path") or params.get("file") or params.get("filepath")

        if old is not None and new is not None:
            diff = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(),
                fromfile="before", tofile="after", lineterm="", n=3,
            ))
            body = "\n".join(diff[:160])
            if path:
                body = f"{path}\n{body}"
            return body or "(no changes)"

        if path and new is not None:
            lines = new.splitlines()
            body = "\n".join(lines[:160])
            more = "" if len(lines) <= 160 else "\n…"
            return f"{path}\n{body}{more}"

        try:
            return json.dumps(params, indent=2, default=str)[:8000]
        except Exception:
            return str(params)[:8000]

    def compose(self) -> ComposeResult:
        req, v = self._request, self._verdict
        risk = getattr(v, "risk", "safe")
        if hasattr(risk, "value"):
            risk = risk.value
        risk_color = ERROR if risk in ("high", "critical") else GREEN
        lines = [
            f"[bold {GREEN}]▌ PERMISSION REQUEST[/]",
            f"[{risk_color}]risk: {_esc(str(risk).upper())}[/]",
            f"[bold {GREEN}]{_esc(req.tool)}.{_esc(req.action)}[/]",
        ]
        if getattr(v, "reason", None):
            lines.append(f"[{MUTED}]{_esc(v.reason)}[/]")
        details = self._details(req)
        if details:
            lines.append("")
            lines.append(f"[{BORDER}]" + "─" * 56 + "[/]")
            lines.append(f"[{TEXT}]{_esc(details)}[/]")
        lines.append("")
        lines.append(
            f"[{MUTED}][A] allow once   [S] allow this tool for the session   "
            f"[D] deny[/]"
        )

        with Vertical(id="perm-card"):
            with VerticalScroll(id="perm-body"):
                yield Static("\n".join(lines), markup=True)
            with Horizontal(id="perm-buttons"):
                yield Button("ALLOW ONCE [A]", id="perm-allow")
                yield Button("ALLOW TOOL [S]", id="perm-allow-session")
                yield Button("DENY [D]", id="perm-deny")

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

    def _finish(self, result: str) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.dismiss(result)
        except Exception:
            pass

    def action_allow(self) -> None:
        self._finish("once")

    def action_allow_session(self) -> None:
        self._finish("session")

    def action_deny(self) -> None:
        self._finish("deny")

    def action_default_action(self) -> None:
        focused = self.focused
        if isinstance(focused, Button) and focused.id == "perm-deny":
            self._finish("deny")
        elif isinstance(focused, Button) and focused.id == "perm-allow-session":
            self._finish("session")
        else:
            self._finish("once")

    @on(Button.Pressed, "#perm-allow")
    def _allow_btn(self, event: Button.Pressed) -> None:
        self._finish("once")

    @on(Button.Pressed, "#perm-allow-session")
    def _allow_session_btn(self, event: Button.Pressed) -> None:
        self._finish("session")

    @on(Button.Pressed, "#perm-deny")
    def _deny_btn(self, event: Button.Pressed) -> None:
        self._finish("deny")


