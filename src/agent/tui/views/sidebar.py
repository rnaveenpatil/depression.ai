"""
Sidebar View

Four-tab sidebar with:
- Sessions: List of saved sessions with timestamps
- Tools: List of available tools with status
- Files: Project file tree browser
- LLM: Provider selection and API key management
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, ListView, ListItem, Label
from textual.reactive import reactive
from textual import on, work

from agent.tui.widgets.llm_panel import LLMProviderPanel
from agent.tui.llm_providers import LLMModel


# Inline colors
SIDEBAR_BG = "#080810"
PANEL = "#12121e"
PANEL_HOVER = "#1c1c30"
PANEL_SELECTED = "#143250"
BORDER = "#283250"
CYBER_BLUE = "#00c8ff"
TEXT_PRIMARY = "#f0f0ff"
TEXT_SECONDARY = "#c8c8dc"
TEXT_MUTED = "#8c8ca0"
TEXT_DIM = "#505064"
NEON_ORANGE = "#ffa500"
NEON_GREEN = "#00ff80"
NEON_CYAN = "#00ffff"


# ======================================================================
# SESSION ITEM
# ======================================================================

class SessionItem(ListItem):
    """A session list item."""

    DEFAULT_CSS = f"""
    SessionItem {{
        height: 3;
        padding: 0 1;
    }}

    SessionItem:hover {{
        background: {PANEL_HOVER};
    }}

    SessionItem.active {{
        background: {PANEL_SELECTED};
        border-left: tall {CYBER_BLUE};
    }}

    .session-name {{
        color: {TEXT_PRIMARY};
        overflow: hidden;
    }}

    .session-meta {{
        color: {TEXT_DIM};
    }}
    """

    def __init__(self, session_id: str, name: str, time: str = "", active: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.session_name = name
        self.time_str = time or "now"
        self.is_active = active
        if active:
            self.add_class("active")

    def compose(self) -> ComposeResult:
        yield Label(self.session_name, classes="session-name")
        yield Label(f"{self.session_id[:8]} · {self.time_str}", classes="session-meta")


# ======================================================================
# TOOL ITEM
# ======================================================================

class ToolItem(ListItem):
    """A tool list item."""

    DEFAULT_CSS = f"""
    ToolItem {{
        height: 1;
        padding: 0 1;
    }}

    ToolItem:hover {{
        background: {PANEL_HOVER};
    }}

    .tool-icon {{
        width: 3;
        color: {NEON_ORANGE};
    }}

    .tool-name {{
        width: 1fr;
        color: {TEXT_SECONDARY};
    }}

    .tool-status {{
        width: auto;
        color: {NEON_GREEN};
    }}
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.tool_enabled = enabled

    def compose(self) -> ComposeResult:
        icon = "⚙" if self.tool_enabled else "○"
        color = NEON_GREEN if self.tool_enabled else TEXT_DIM
        yield Label(f"[{NEON_ORANGE}]⚙[/]", classes="tool-icon")
        yield Label(self.tool_name, classes="tool-name")
        yield Label(f"[{color}]{icon}[/]", classes="tool-status")


# ======================================================================
# FILE ITEM
# ======================================================================

class FileItem(ListItem):
    """A file tree item."""

    DEFAULT_CSS = f"""
    FileItem {{
        height: 1;
        padding: 0 1;
    }}

    FileItem:hover {{
        background: {PANEL_HOVER};
    }}

    .file-icon {{
        width: 2;
    }}

    .file-name {{
        width: 1fr;
        color: {TEXT_SECONDARY};
    }}

    .file-size {{
        width: auto;
        color: {TEXT_DIM};
    }}
    """

    def __init__(self, name: str, is_dir: bool = False, size: str = "", **kwargs):
        super().__init__(**kwargs)
        self.file_name = name
        self.is_dir = is_dir
        self.file_size = size

    def compose(self) -> ComposeResult:
        icon = "📁" if self.is_dir else "📄"
        color = NEON_CYAN if self.is_dir else TEXT_SECONDARY
        yield Label(f"[{color}]{icon}[/]", classes="file-icon")
        yield Label(self.file_name, classes="file-name")
        if self.file_size:
            yield Label(self.file_size, classes="file-size")


# ======================================================================
# SIDEBAR
# ======================================================================

class Sidebar(Widget):
    """The sidebar with tabs for sessions, tools, files, and LLM providers."""

    DEFAULT_CSS = f"""
    Sidebar {{
        width: 34;
        background: {SIDEBAR_BG};
        border-right: solid {BORDER};
        layout: vertical;
    }}

    #sidebar-tabs {{
        height: 3;
        layout: horizontal;
        background: {PANEL};
        border-bottom: solid {BORDER};
    }}

    .sidebar-tab-btn {{
        width: 1fr;
        height: 100%;
        background: transparent;
        border: none;
        color: {TEXT_MUTED};
        content-align: center middle;
    }}

    .sidebar-tab-btn:hover {{
        background: {PANEL_HOVER};
        color: {TEXT_PRIMARY};
    }}

    .sidebar-tab-btn.active {{
        background: {PANEL_SELECTED};
        color: {CYBER_BLUE};
        text-style: bold;
        border-bottom: tall {CYBER_BLUE};
    }}

    #sidebar-content {{
        height: 1fr;
        overflow-y: auto;
    }}

    #sidebar-sessions {{
        height: 100%;
        overflow-y: auto;
    }}

    #sidebar-tools {{
        height: 100%;
        display: none;
        overflow-y: auto;
    }}

    #sidebar-files {{
        height: 100%;
        display: none;
        overflow-y: auto;
    }}

    #sidebar-llm {{
        height: 100%;
        display: none;
        overflow-y: auto;
    }}

    /* ListView styling */
    ListView {{
        background: transparent;
        border: none;
    }}

    ListView > ListItem {{
        background: transparent;
    }}

    ListView > ListItem:hover {{
        background: {PANEL_HOVER};
    }}

    ListView > ListItem.active {{
        background: {PANEL_SELECTED};
        border-left: tall {CYBER_BLUE};
    }}
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
        # Tab buttons
        with Widget(id="sidebar-tabs"):
            yield Button(" ◈ Sessions ", id="tab-sessions", classes="sidebar-tab-btn active")
            yield Button(" ⚙ Tools ", id="tab-tools", classes="sidebar-tab-btn")
            yield Button(" 📁 Files ", id="tab-files", classes="sidebar-tab-btn")
            yield Button(" ◆ LLM ", id="tab-llm", classes="sidebar-tab-btn")

        # Content area
        with Widget(id="sidebar-content"):
            # Sessions panel
            yield ListView(id="sidebar-sessions")

            # Tools panel
            yield ListView(id="sidebar-tools")

            # Files panel
            yield ListView(id="sidebar-files")

            # LLM panel - will be mounted dynamically
            yield Widget(id="sidebar-llm")

    def on_mount(self) -> None:
        """Initialize after mounting."""
        # Populate initial sessions
        if self.sessions:
            self._populate_sessions()
        # Populate initial tools
        if self.tools:
            self._populate_tools()
        # Populate initial files
        if self.files:
            self._populate_files()

    @on(Button.Pressed, ".sidebar-tab-btn")
    def _on_tab_pressed(self, event: Button.Pressed) -> None:
        """Handle tab button presses."""
        tab_id = event.button.id.replace("tab-", "")
        self.active_tab = tab_id

        # Update tab styling
        for btn in self.query(".sidebar-tab-btn"):
            btn.remove_class("active")
        event.button.add_class("active")

        # Show/hide content
        content = self.query_one("#sidebar-content")
        for child in content.children:
            child.display = False

        target = self.query_one(f"#sidebar-{tab_id}")
        target.display = True

        # Initialize LLM panel when first shown
        if tab_id == "llm" and self._llm_panel is None:
            self._init_llm_panel()

    def _init_llm_panel(self) -> None:
        """Initialize the LLM panel lazily."""
        try:
            llm_container = self.query_one("#sidebar-llm")
            self._llm_panel = LLMProviderPanel(
                id="llm-panel",
                on_connect=self._on_llm_connect,
            )
            llm_container.mount(self._llm_panel)
        except Exception:
            pass

    def _populate_sessions(self) -> None:
        """Populate sessions list view."""
        try:
            list_view = self.query_one("#sidebar-sessions", ListView)
            list_view.clear()
            for s in self.sessions:
                list_view.append(
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
        """Populate tools list view."""
        try:
            list_view = self.query_one("#sidebar-tools", ListView)
            list_view.clear()
            for t in self.tools:
                list_view.append(
                    ToolItem(t.get("name", ""), t.get("enabled", True))
                )
        except Exception:
            pass

    def _populate_files(self) -> None:
        """Populate files list view."""
        try:
            list_view = self.query_one("#sidebar-files", ListView)
            list_view.clear()
            for f in self.files:
                list_view.append(
                    FileItem(
                        f.get("name", ""),
                        f.get("is_dir", False),
                        f.get("size", ""),
                    )
                )
        except Exception:
            pass

    def set_sessions(self, sessions: List[Dict]) -> None:
        """Update the sessions list."""
        self.sessions = sessions
        self._populate_sessions()

    def set_tools(self, tools: List[Dict]) -> None:
        """Update the tools list."""
        self.tools = tools
        self._populate_tools()

    def set_files(self, files: List[Dict]) -> None:
        """Update the files list."""
        self.files = files
        self._populate_files()

    def get_llm_panel(self) -> Optional[LLMProviderPanel]:
        """Get the LLM provider panel widget."""
        if self._llm_panel is None:
            self._init_llm_panel()
        return self._llm_panel

    def switch_to_llm(self) -> None:
        """Switch to the LLM tab."""
        try:
            btn = self.query_one("#tab-llm", Button)
            self._on_tab_pressed(type("Event", (), {"button": btn})())
        except Exception:
            pass


# ======================================================================
# SESSION ITEM
# ======================================================================

class SessionItem(ListItem):
    """A session list item."""

    DEFAULT_CSS = f"""
    SessionItem {{
        height: 3;
        padding: 0 1;
    }}

    SessionItem:hover {{
        background: {PANEL_HOVER};
    }}

    SessionItem.active {{
        background: {PANEL_SELECTED};
        border-left: tall {CYBER_BLUE};
    }}

    .session-name {{
        color: {TEXT_PRIMARY};
        overflow: hidden;
    }}

    .session-meta {{
        color: {TEXT_DIM};
    }}
    """

    def __init__(self, session_id: str, name: str, time: str = "", active: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.session_name = name
        self.time_str = time or "now"
        self.is_active = active
        if active:
            self.add_class("active")

    def compose(self) -> ComposeResult:
        yield Label(self.session_name, classes="session-name")
        yield Label(f"{self.session_id[:8]} · {self.time_str}", classes="session-meta")


# ======================================================================
# TOOL ITEM
# ======================================================================

class ToolItem(ListItem):
    """A tool list item."""

    DEFAULT_CSS = f"""
    ToolItem {{
        height: 1;
        padding: 0 1;
    }}

    ToolItem:hover {{
        background: {PANEL_HOVER};
    }}

    .tool-icon {{
        width: 3;
        color: {NEON_ORANGE};
    }}

    .tool-name {{
        width: 1fr;
        color: {TEXT_SECONDARY};
    }}

    .tool-status {{
        width: auto;
        color: {NEON_GREEN};
    }}
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.tool_enabled = enabled

    def compose(self) -> ComposeResult:
        icon = "⚙" if self.tool_enabled else "○"
        color = "neon-green" if self.tool_enabled else "text-dim"
        yield Label(f"[neon-orange]⚙[/]", classes="tool-icon")
        yield Label(self.tool_name, classes="tool-name")
        yield Label(f"[{color}]{icon}[/]", classes="tool-status")


# ======================================================================
# FILE ITEM
# ======================================================================

class FileItem(ListItem):
    """A file tree item."""

    DEFAULT_CSS = f"""
    FileItem {{
        height: 1;
        padding: 0 1;
    }}

    FileItem:hover {{
        background: {PANEL_HOVER};
    }}

    .file-icon {{
        width: 2;
    }}

    .file-name {{
        width: 1fr;
        color: {TEXT_SECONDARY};
    }}

    .file-size {{
        width: auto;
        color: {TEXT_DIM};
    }}
    """

    def __init__(self, name: str, is_dir: bool = False, size: str = "", **kwargs):
        super().__init__(**kwargs)
        self.file_name = name
        self.is_dir = is_dir
        self.file_size = size

    def compose(self) -> ComposeResult:
        icon = "📁" if self.is_dir else "📄"
        color = NEON_CYAN if self.is_dir else TEXT_SECONDARY
        yield Label(f"[{color}]{icon}[/]", classes="file-icon")
        yield Label(self.file_name, classes="file-name")
        if self.file_size:
            yield Label(self.file_size, classes="file-size")


# ======================================================================
# SIDEBAR
# ======================================================================

class Sidebar(Widget):
    """The sidebar with tabs for sessions, tools, files, and LLM providers."""

    DEFAULT_CSS = f"""
    Sidebar {{
        width: 34;
        background: {SIDEBAR_BG};
        border-right: solid {BORDER};
        layout: vertical;
    }}

    #sidebar-tabs {{
        height: 3;
        layout: horizontal;
        background: {PANEL};
        border-bottom: solid {BORDER};
    }}

    .sidebar-tab-btn {{
        width: 1fr;
        height: 100%;
        background: transparent;
        border: none;
        color: {TEXT_MUTED};
        content-align: center middle;
    }}

    .sidebar-tab-btn:hover {{
        background: {PANEL_HOVER};
        color: {TEXT_PRIMARY};
    }}

    .sidebar-tab-btn.active {{
        background: {PANEL_SELECTED};
        color: {CYBER_BLUE};
        text-style: bold;
        border-bottom: tall {CYBER_BLUE};
    }}

    #sidebar-content {{
        height: 1fr;
        overflow-y: auto;
    }}

    #sidebar-sessions {{
        height: 100%;
    }}

    #sidebar-tools {{
        height: 100%;
        display: none;
    }}

    #sidebar-files {{
        height: 100%;
        display: none;
    }}

    #sidebar-llm {{
        height: 100%;
        display: none;
    }}
    """

    active_tab = reactive[str]("sessions")

    def __init__(self, on_llm_connect: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.sessions: List[Dict] = []
        self.tools: List[Dict] = []
        self.files: List[Dict] = []
        self._on_llm_connect = on_llm_connect

    def compose(self) -> ComposeResult:
        # Tab buttons
        with Widget(id="sidebar-tabs"):
            yield Button(" ◈ Sessions ", id="tab-sessions", classes="sidebar-tab-btn active")
            yield Button(" ⚙ Tools ", id="tab-tools", classes="sidebar-tab-btn")
            yield Button(" 📁 Files ", id="tab-files", classes="sidebar-tab-btn")
            yield Button(" ◆ LLM ", id="tab-llm", classes="sidebar-tab-btn")

        # Content area
        with Widget(id="sidebar-content"):
            # Sessions panel
            with ListView(id="sidebar-sessions"):
                for s in self.sessions:
                    yield SessionItem(
                        s.get("id", ""),
                        s.get("name", "Unnamed"),
                        s.get("time", ""),
                        s.get("active", False),
                    )

            # Tools panel
            with ListView(id="sidebar-tools"):
                for t in self.tools:
                    yield ToolItem(
                        t.get("name", ""),
                        t.get("enabled", True),
                    )

            # Files panel
            with ListView(id="sidebar-files"):
                for f in self.files:
                    yield FileItem(
                        f.get("name", ""),
                        f.get("is_dir", False),
                        f.get("size", ""),
                    )

            # LLM panel
            yield LLMProviderPanel(
                id="sidebar-llm",
                on_connect=self._on_llm_connect,
            )

    @on(Button.Pressed, ".sidebar-tab-btn")
    def _on_tab_pressed(self, event: Button.Pressed) -> None:
        """Handle tab button presses."""
        tab_id = event.button.id.replace("tab-", "")
        self.active_tab = tab_id

        # Update tab styling
        for btn in self.query(".sidebar-tab-btn"):
            btn.remove_class("active")
        event.button.add_class("active")

        # Show/hide content
        content = self.query_one("#sidebar-content")
        for child in content.children:
            child.display = False

        target = self.query_one(f"#sidebar-{tab_id}")
        target.display = True

    def set_sessions(self, sessions: List[Dict]) -> None:
        """Update the sessions list."""
        self.sessions = sessions
        try:
            list_view = self.query_one("#sidebar-sessions", ListView)
            list_view.clear()
            for s in sessions:
                list_view.append(
                    SessionItem(
                        s.get("id", ""),
                        s.get("name", "Unnamed"),
                        s.get("time", ""),
                        s.get("active", False),
                    )
                )
        except Exception:
            pass

    def set_tools(self, tools: List[Dict]) -> None:
        """Update the tools list."""
        self.tools = tools
        try:
            list_view = self.query_one("#sidebar-tools", ListView)
            list_view.clear()
            for t in tools:
                list_view.append(
                    ToolItem(t.get("name", ""), t.get("enabled", True))
                )
        except Exception:
            pass

    def set_files(self, files: List[Dict]) -> None:
        """Update the files list."""
        self.files = files
        try:
            list_view = self.query_one("#sidebar-files", ListView)
            list_view.clear()
            for f in files:
                list_view.append(
                    FileItem(
                        f.get("name", ""),
                        f.get("is_dir", False),
                        f.get("size", ""),
                    )
                )
        except Exception:
            pass

    def get_llm_panel(self) -> Optional[LLMProviderPanel]:
        """Get the LLM provider panel widget."""
        try:
            return self.query_one("#sidebar-llm", LLMProviderPanel)
        except Exception:
            return None

    def switch_to_llm(self) -> None:
        """Switch to the LLM tab."""
        try:
            btn = self.query_one("#tab-llm", Button)
            self._on_tab_pressed(type("Event", (), {"button": btn})())
        except Exception:
            pass