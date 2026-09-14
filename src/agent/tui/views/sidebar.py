"""Futuristic web-app style sidebar for Depression.AI."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, ListView, ListItem, Label
from textual.reactive import reactive
from textual import on

from agent.tui.widgets.llm_panel import LLMProviderPanel


SIDEBAR_BG = "#0b0b16"
PANEL = "#11111d"
PANEL_HOVER = "#20203a"
PANEL_SELECTED = "#252047"
BORDER = "#272640"
CYAN = "#54d9ff"
VIOLET = "#8f7cff"
GREEN = "#62f5b0"
PINK = "#ff8bdc"
GOLD = "#ffc857"
TEXT = "#f3f1ff"
SECONDARY = "#c9c6df"
MUTED = "#77738e"
DIM = "#5d5a77"


class SessionItem(ListItem):
    """Compact session card."""

    DEFAULT_CSS = f"""
    SessionItem {{ height: 4; margin: 0 0 1 0; padding: 0 1; background: {PANEL}; }}
    SessionItem:hover {{ background: {PANEL_HOVER}; }}
    SessionItem.active {{ background: {PANEL_SELECTED}; border-left: tall {CYAN}; }}
    .session-name {{ color: {TEXT}; text-style: bold; width: 1fr; overflow: hidden; }}
    .session-meta {{ color: {DIM}; width: 1fr; }}
    .session-state {{ color: {GREEN}; width: auto; }}
    """

    def __init__(self, session_id: str, name: str, time: str = "", active: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.session_name = name
        self.time_str = time or "now"
        if active:
            self.add_class("active")

    def compose(self) -> ComposeResult:
        yield Label(f"◈  {self.session_name}", classes="session-name")
        yield Label(f"    {self.session_id[:8]}  ·  {self.time_str}", classes="session-meta")
        yield Label("●", classes="session-state")


class ToolItem(ListItem):
    """Tool capability row."""

    DEFAULT_CSS = f"""
    ToolItem {{ height: 2; padding: 0 1; }}
    ToolItem:hover {{ background: {PANEL_HOVER}; }}
    .tool-icon {{ width: 3; color: {GOLD}; }}
    .tool-name {{ width: 1fr; color: {SECONDARY}; }}
    .tool-status {{ width: auto; color: {GREEN}; }}
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.tool_enabled = enabled

    def compose(self) -> ComposeResult:
        yield Label("◇", classes="tool-icon")
        yield Label(self.tool_name, classes="tool-name")
        yield Label("●" if self.tool_enabled else "○", classes="tool-status")


class FileItem(ListItem):
    """Project file row."""

    DEFAULT_CSS = f"""
    FileItem {{ height: 2; padding: 0 1; }}
    FileItem:hover {{ background: {PANEL_HOVER}; }}
    .file-icon {{ width: 3; color: {CYAN}; }}
    .file-name {{ width: 1fr; color: {SECONDARY}; overflow: hidden; }}
    .file-size {{ width: auto; color: {DIM}; }}
    """

    def __init__(self, name: str, is_dir: bool = False, size: str = "", **kwargs):
        super().__init__(**kwargs)
        self.file_name = name
        self.is_dir = is_dir
        self.file_size = size

    def compose(self) -> ComposeResult:
        yield Label("▾" if self.is_dir else "·", classes="file-icon")
        yield Label(self.file_name, classes="file-name")
        if self.file_size:
            yield Label(self.file_size, classes="file-size")


class Sidebar(Widget):
    """Persistent navigation sidebar with sessions, tools, files and LLM."""

    DEFAULT_CSS = f"""
    Sidebar {{ width: 36; min-width: 28; background: {SIDEBAR_BG}; border-right: solid {BORDER}; layout: vertical; }}
    #sidebar-brand {{ height: 4; padding: 0 2; background: {SIDEBAR_BG}; border-bottom: solid {BORDER}; }}
    #sidebar-brand-title {{ color: {TEXT}; text-style: bold; }}
    #sidebar-brand-sub {{ color: {DIM}; }}
    #sidebar-tabs {{ height: 3; layout: horizontal; background: {PANEL}; border-bottom: solid {BORDER}; }}
    .sidebar-tab-btn {{ width: 1fr; height: 3; background: transparent; border: none; color: {MUTED}; text-style: bold; }}
    .sidebar-tab-btn:hover {{ background: {PANEL_HOVER}; color: {TEXT}; }}
    .sidebar-tab-btn.active {{ background: {PANEL_SELECTED}; color: {VIOLET}; border-bottom: tall {VIOLET}; }}
    #sidebar-content {{ height: 1fr; padding: 1; }}
    #sidebar-sessions, #sidebar-tools, #sidebar-files, #sidebar-llm {{ height: 1fr; overflow-y: auto; }}
    """

    active_tab = reactive[str]("sessions")

    def __init__(self, on_llm_connect: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.sessions: List[Dict] = []
        self.tools: List[Dict] = []
        self.files: List[Dict] = []
        self._on_llm_connect = on_llm_connect
        self._llm_panel = None

    def compose(self) -> ComposeResult:
        with Widget(id="sidebar-brand"):
            yield Static("◈  DEPRESSION.AI", id="sidebar-brand-title")
            yield Static("agentic workspace  /  local + cloud", id="sidebar-brand-sub")
        with Widget(id="sidebar-tabs"):
            yield Button("◈\nCHAT", id="tab-sessions", classes="sidebar-tab-btn active")
            yield Button("◇\nTOOLS", id="tab-tools", classes="sidebar-tab-btn")
            yield Button("⌁\nFILES", id="tab-files", classes="sidebar-tab-btn")
            yield Button("◆\nMODEL", id="tab-llm", classes="sidebar-tab-btn")
        with Widget(id="sidebar-content"):
            yield ListView(id="sidebar-sessions")
            yield ListView(id="sidebar-tools")
            yield ListView(id="sidebar-files")
            yield Widget(id="sidebar-llm")

    def on_mount(self) -> None:
        if self.sessions: self._populate_sessions()
        if self.tools: self._populate_tools()
        if self.files: self._populate_files()

    @on(Button.Pressed, ".sidebar-tab-btn")
    def _on_tab_pressed(self, event: Button.Pressed) -> None:
        tab_id = event.button.id.replace("tab-", "")
        self.active_tab = tab_id
        for btn in self.query(".sidebar-tab-btn"):
            btn.remove_class("active")
        event.button.add_class("active")
        content = self.query_one("#sidebar-content")
        for child in content.children:
            child.display = False
        self.query_one(f"#sidebar-{tab_id}").display = True
        if tab_id == "llm" and self._llm_panel is None:
            self._init_llm_panel()

    def _init_llm_panel(self) -> None:
        try:
            container = self.query_one("#sidebar-llm")
            self._llm_panel = LLMProviderPanel(id="llm-panel", on_connect=self._on_llm_connect)
            container.mount(self._llm_panel)
        except Exception:
            pass

    def _populate_sessions(self) -> None:
        try:
            view = self.query_one("#sidebar-sessions", ListView)
            view.clear()
            for s in self.sessions:
                view.append(SessionItem(s.get("id", ""), s.get("name", "Unnamed"), s.get("time", ""), s.get("active", False)))
        except Exception:
            pass

    def _populate_tools(self) -> None:
        try:
            view = self.query_one("#sidebar-tools", ListView)
            view.clear()
            for t in self.tools:
                view.append(ToolItem(t.get("name", ""), t.get("enabled", True)))
        except Exception:
            pass

    def _populate_files(self) -> None:
        try:
            view = self.query_one("#sidebar-files", ListView)
            view.clear()
            for f in self.files:
                view.append(FileItem(f.get("name", ""), f.get("is_dir", False), f.get("size", "")))
        except Exception:
            pass

    def set_sessions(self, sessions: List[Dict]) -> None:
        self.sessions = sessions
        self._populate_sessions()

    def set_tools(self, tools: List[Dict]) -> None:
        self.tools = tools
        self._populate_tools()

    def set_files(self, files: List[Dict]) -> None:
        self.files = files
        self._populate_files()

    def get_llm_panel(self) -> Optional[LLMProviderPanel]:
        if self._llm_panel is None:
            self._init_llm_panel()
        return self._llm_panel

    def switch_to_llm(self) -> None:
        try:
            self.query_one("#tab-llm", Button).press()
        except Exception:
            pass
