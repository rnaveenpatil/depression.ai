"""Compact terminal rail — hidden by default, toggled with F2."""
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
    SessionItem:hover, SessionItem.active { background: #1a1820; }
    .session-name { width: 1fr; color: #e8e3f0; }
    .session-state { width: 2; color: #7ef7c0; }
    """
    def __init__(self, session_id: str, name: str, time: str = "",
                 active: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.session_id, self.session_name = session_id, name
        if active:
            self.add_class("active")

    def compose(self) -> ComposeResult:
        yield Label(self.session_name, classes="session-name")
        yield Label("●", classes="session-state")


class ToolItem(ListItem):
    DEFAULT_CSS = """
    ToolItem { height: 1; padding: 0 1; }
    .tool-name { width: 1fr; color: #8a849a; }
    .tool-status { width: 2; color: #7ef7c0; }
    """
    def __init__(self, name: str, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_name, self.tool_enabled = name, enabled

    def compose(self) -> ComposeResult:
        yield Label(f" · {self.tool_name}", classes="tool-name")
        yield Label("●" if self.tool_enabled else "○", classes="tool-status")


class Sidebar(Widget):
    """Contextual rail: sessions / tools / files / model."""

    DEFAULT_CSS = """
    Sidebar {
        width: 40; min-width: 34;
        background: #0e0d12;
        border-left: solid #2a2735;
        layout: vertical;
    }
    #rail-title {
        height: 1; padding: 0 1;
        color: #8a849a; text-style: bold;
        background: #08070c;
        border-bottom: solid #2a2735;
    }
    #rail-tabs {
        height: 1; layout: horizontal; background: #08070c;
        border-bottom: solid #2a2735;
    }
    .rail-tab {
        width: 1fr; height: 1;
        background: transparent; border: none;
        color: #524d60; text-style: bold;
    }
    .rail-tab:hover { color: #8a849a; }
    .rail-tab.active { color: #ff9bc7; }
    #sidebar-content {
        height: 1fr;
        layout: vertical;
    }
    #sidebar-sessions, #sidebar-tools, #sidebar-files, #sidebar-llm {
        height: 1fr;
        width: 100%;
    }
    """

    active_tab = reactive("sessions")

    TABS = [("sessions", "chat"), ("tools", "tools"),
            ("files", "files"), ("llm", "model")]

    def __init__(self, on_llm_connect: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.sessions: List[Dict] = []
        self.tools:    List[Dict] = []
        self.files:    List[Dict] = []
        self._on_llm_connect = on_llm_connect
        self._llm_panel = None

    def compose(self) -> ComposeResult:
        yield Static("WORKSPACE", id="rail-title")
        with Widget(id="rail-tabs"):
            for key, label in self.TABS:
                cls = "rail-tab active" if key == "sessions" else "rail-tab"
                yield Button(label.upper(), id=f"tab-{key}", classes=cls)
        with Widget(id="sidebar-content"):
            yield ListView(id="sidebar-sessions")
            yield ListView(id="sidebar-tools")
            yield ListView(id="sidebar-files")
            yield Widget(id="sidebar-llm")

    def on_mount(self) -> None:
        self._populate_sessions()
        self._populate_tools()
        self._init_llm_panel()
        self._show("sessions")

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

    def _init_llm_panel(self) -> None:
        if self._llm_panel is not None:
            return
        try:
            container = self.query_one("#sidebar-llm")
            self._llm_panel = LLMProviderPanel(id="llm-panel",
                                               on_connect=self._on_llm_connect)
            container.mount(self._llm_panel)
        except Exception:
            pass

    def _populate_sessions(self) -> None:
        try:
            view = self.query_one("#sidebar-sessions", ListView)
            view.clear()
            for s in self.sessions:
                view.append(SessionItem(s.get("id", ""), s.get("name", "Unnamed"),
                                        s.get("time", ""), s.get("active", False)))
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
                view.append(ListItem(Label(f" · {f.get('name', '')}")))
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
        return self._llm_panel

    def switch_to_llm(self) -> None:
        self._show("llm")

    def switch_to_tools(self) -> None:
        self._show("tools")

    def switch_to_sessions(self) -> None:
        self._show("sessions")