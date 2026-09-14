"""Compact right-side workspace rail for Depression.AI."""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, ListView, ListItem, Label
from textual.reactive import reactive
from textual import on
from agent.tui.widgets.llm_panel import LLMProviderPanel


class SessionItem(ListItem):
    DEFAULT_CSS = """
    SessionItem { height: 2; padding: 0 1; }
    SessionItem:hover, SessionItem.active { background: $boost; color: $text; }
    .session-name { width: 1fr; color: $text; }
    .session-state { width: 2; color: $success; }
    """
    def __init__(self, session_id: str, name: str, time: str = "", active: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.session_id, self.session_name = session_id, name
        if active:
            self.add_class("active")

    def compose(self) -> ComposeResult:
        yield Label(f"{self.session_name}", classes="session-name")
        yield Label("●", classes="session-state")


class ToolItem(ListItem):
    DEFAULT_CSS = """
    ToolItem { height: 2; padding: 0 1; }
    ToolItem:hover { background: $boost; }
    .tool-name { width: 1fr; color: $text; }
    .tool-status { width: 2; color: $success; }
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_name, self.tool_enabled = name, enabled

    def compose(self) -> ComposeResult:
        yield Label(self.tool_name, classes="tool-name")
        yield Label("●" if self.tool_enabled else "○", classes="tool-status")


class Sidebar(Widget):
    """A narrow terminal rail. It is intentionally not a web-app dashboard."""
    DEFAULT_CSS = """
    Sidebar { width: 34; min-width: 30; background: $panel; border-left: solid $border; layout: vertical; }
    #rail-title { height: 3; padding: 1 1 0 1; color: $text; text-style: bold; border-bottom: solid $border; }
    #rail-tabs { height: 3; layout: horizontal; border-bottom: solid $border; }
    .rail-tab { width: 1fr; height: 3; background: transparent; border: none; color: $text-muted; }
    .rail-tab:hover { background: $boost; color: $text; }
    .rail-tab.active { color: $secondary; border-bottom: tall $secondary; }
    #sidebar-content { height: 1fr; padding: 0 1; }
    #sidebar-sessions, #sidebar-tools, #sidebar-files, #sidebar-llm { height: 1fr; }
    #sidebar-llm { padding: 0; }
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
        yield Static("WORKSPACE", id="rail-title")
        with Widget(id="rail-tabs"):
            yield Button("CHAT", id="tab-sessions", classes="rail-tab active")
            yield Button("TOOLS", id="tab-tools", classes="rail-tab")
            yield Button("FILES", id="tab-files", classes="rail-tab")
            yield Button("MODEL", id="tab-llm", classes="rail-tab")
        with Widget(id="sidebar-content"):
            yield ListView(id="sidebar-sessions")
            yield ListView(id="sidebar-tools")
            yield ListView(id="sidebar-files")
            yield Widget(id="sidebar-llm")

    def on_mount(self) -> None:
        self._show("sessions")
        self._populate_sessions()
        self._populate_tools()
        self._populate_files()

    @on(Button.Pressed, ".rail-tab")
    def _on_tab(self, event: Button.Pressed) -> None:
        self._show(event.button.id.replace("tab-", ""))

    def _show(self, tab: str) -> None:
        self.active_tab = tab
        for button in self.query(".rail-tab"):
            button.remove_class("active")
        try:
            self.query_one(f"#tab-{tab}", Button).add_class("active")
        except Exception:
            pass

        content = self.query_one("#sidebar-content")
        for child in content.children:
            child.display = False
        try:
            self.query_one(f"#sidebar-{tab}").display = True
        except Exception:
            pass
        if tab == "llm" and self._llm_panel is None:
            self._init_llm_panel()

    def _init_llm_panel(self) -> None:
        try:
            container = self.query_one("#sidebar-llm")
            self._llm_panel = LLMProviderPanel(
                id="llm-panel", on_connect=self._on_llm_connect
            )
            container.mount(self._llm_panel)
        except Exception:
            pass

    def _populate_sessions(self) -> None:
        try:
            view = self.query_one("#sidebar-sessions", ListView)
            view.clear()
            for s in self.sessions:
                view.append(
                    SessionItem(
                        s.get("id", ""),
                        s.get("name", "Unnamed"),
                        s.get("time", ""),
                        s.get("active", False),
                    )
                )
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
                view.append(ListItem(Label(f.get("name", ""))))
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
        self._show("llm")

    def switch_to_tools(self) -> None:
        self._show("tools")

    def switch_to_sessions(self) -> None:
        self._show("sessions")
