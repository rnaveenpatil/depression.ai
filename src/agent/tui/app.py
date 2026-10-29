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
from agent.tui.widgets.sidebar import Sidebar
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

    def __init__(self, header: str = "◆ depression.ai", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._header = header
        # [Streaming fix] Keep a running buffer + a running *escaped* string
        # so we don't re-join / re-escape the whole transcript on every token.
        self._raw: list[str] = []
        self._escaped: list[str] = []
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

    def append_token(self, delta: str) -> None:
        if not delta:
            return
        self._raw.append(delta)
        # Escape only the incoming delta; append to the escaped list.
        self._escaped.append(delta.replace("[", r"\["))
        if self._body is not None:
            try:
                self._body.update("".join(self._escaped))
            except Exception:
                pass

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
        ("/model", "Open the LLM panel and discover models"),
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
        ("/run", "Run a shell command"),
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
        yield Static("Discovered Models", classes="label")
        yield Select(
            [], prompt="discover models to choose",
            allow_blank=True, id="model-select",
        )
        yield Button("CONNECT", id="connect")
        yield Horizontal(
            Button("TEST", id="llm-test"),
            Button("MODELS", id="discover"),
            id="llm-actions",
        )
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
        self.set_interval(0.5, self.refresh_from_store)

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
        self.set_interval(0.5, self.refresh_values)

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
        yield Static("/model     open LLM panel + discover models", classes="help-row")
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
        yield Static("/run CMD   run a shell command and show output", classes="help-row")
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
    context_window: reactive[int] = reactive(200_000)
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

        # Only panel tabs should be skipped during keyboard navigation.
        # Panel controls (AWS save, MCP reconnect, etc.) must remain focusable.
        for w in self.query("#sidebar .side-tab"):
            try:
                w.can_focus = False
            except Exception:
                pass

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
        self.set_interval(0.12, self._tick_spinner)

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

        self._streaming_widget.append_token(delta)
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
            self._scroll_transcript_end()
        except Exception:
            pass

    async def _on_token_end(self, event: AgentEvent) -> None:
        # [Streaming fix] Same rationale as _on_token — no is_running gate.
        # The structured OutputView will replace the streaming widget in
        # _render_output once the turn's result comes back.
        if self._is_shutting_down:
            return

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
        r = dict(result)

        before = None
        for key in ("before", "content_before", "old_content"):
            v = r.get(key)
            if isinstance(v, str):
                before = v
                break

        after = None
        for key in ("after", "content", "new_content", "text", "output"):
            v = r.get(key)
            if isinstance(v, str):
                after = v
                break

        if before is not None:
            r["before"] = before
        if after is not None:
            r["after"] = after

        if before is None and after is not None:
            r["diff_source"] = "tool:add-only"
        elif before is None and after is None:
            r["diff_source"] = "tool:none"
        else:
            r["diff_source"] = "tool"
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
            should_run = (
                execute
                and selected not in _SLASH_ARGUMENT_COMMANDS
                and len(draft.strip()) > 1
            )
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
        files = self._project_files()
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
        client = panel.get_client()
        if client is None or not getattr(client, "enabled", False):
            self._system("MCP is disabled; enable it in the config and restart.")
            return
        button = self.query_one("#mcp-reconnect", Button)
        button.disabled = True
        try:
            self._system("Reconnecting configured MCP servers…")
            await client.reload()

            # MCPClient.reload() discovers tools, but the Build agent's
            # registry originally receives them only during initialization.
            coordinator = self.coordinator
            build = getattr(coordinator, "build_agent", None)
            registry = getattr(build, "tool_registry", None)
            if registry is not None:
                discovered = {
                    tool["function"]["name"]: tool["function"]
                    for tool in client.list_tools()
                }
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

                # Rebuild pinned system prompts after schema changes.
                loop = getattr(build, "loop", None)
                if loop is not None:
                    loop._system_prompt_cache = None
                    loop._system_prompt_fp = None

            self._system("MCP reconnect finished; check server status for failures.")
        except Exception as exc:
            self._error("MCP reconnect failed: " + type(exc).__name__)
        finally:
            self._refresh_mcp_status()
            if not self.busy and not self._is_shutting_down:
                button.disabled = False

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

    @on(Button.Pressed, ".side-tab")
    def _on_tab(self, event: Button.Pressed) -> None:
        self._set_active_panel(event.button.id.replace("tab-", ""))
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
        elif command == "/connect":
            self._show_panel("llm")
            try:
                self.query_one("#base-url", Input).focus()
            except Exception:
                pass
        elif command in ("/models", "/model"):
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

    def _copy_last_output(self) -> None:
        if not self._last_output:
            self._system("nothing to copy yet")
            return
        try:
            self.copy_to_clipboard(self._last_output)
            chars = len(self._last_output)
            self._system(f"copied last reply ({chars:,} chars) to the clipboard")
        except Exception as exc:
            self._error(f"clipboard copy failed: {exc}")

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
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self._write(f"  {line}", "agent")
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self._write(f"  {line}", "error")
            self._system(f"exit {proc.returncode}")
        except Exception as exc:
            self._error(f"run failed: {exc}")

    def _request_agent_stop(self) -> None:
        """Ask the running loop to stop gracefully and summarize its work."""
        if not (self.busy or self._draining):
            self._system("nothing is running")
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
        if not base_url or not api_key or not model:
            status.update("base URL, API key and model are required")
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

    @on(Select.Changed, "#model-select")
    def _model_selected(self, event: Select.Changed) -> None:
        value = event.value
        if value is Select.BLANK or value is None:
            return
        try:
            self.query_one("#model-id", Input).value = str(value)
            self.query_one("#llm-status", Static).update(
                f"selected model · {_esc(str(value))}"
            )
        except Exception:
            pass

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
                try:
                    self.query_one("#model-select", Select).set_options(
                        [(mid, mid) for mid in ids]
                    )
                except Exception:
                    pass
                status.update(
                    f"discovered {len(ids)} model(s); choose one from the list"
                )
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
            f"saved to ~/.agent/env · {self.aws.get('region') or DEFAULT_AWS_REGION}"
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
            f"AWS credentials saved to ~/.agent/env "
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