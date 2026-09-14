"""
Header Bar View

Displays:
- Logo/title with gradient text
- Current mode indicator (Plan/Build/Auto)
- Model name
- Session ID
- Time
"""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


# Inline colors (matching ULTIMATE_CSS)
PANEL = "#12121e"
BORDER = "#283250"
CYBER_BLUE = "#00c8ff"
TEXT_MUTED = "#8c8ca0"
TEXT_DIM = "#505064"


class HeaderBar(Widget):
    """The top header bar."""

    DEFAULT_CSS = f"""
    HeaderBar {{
        height: 3;
        background: {PANEL};
        dock: top;
        layout: horizontal;
        padding: 0 1;
    }}

    #header-logo {{
        width: auto;
        content-align: left middle;
        color: {CYBER_BLUE};
        text-style: bold;
    }}

    #header-separator {{
        width: 1;
        color: {BORDER};
        content-align: center middle;
    }}

    #header-mode {{
        width: auto;
        content-align: center middle;
        text-style: bold;
        padding: 0 1;
    }}

    #header-model {{
        width: 1fr;
        content-align: center middle;
        color: {TEXT_MUTED};
    }}

    #header-session {{
        width: auto;
        content-align: right middle;
        color: {TEXT_DIM};
    }}

    #header-time {{
        width: auto;
        content-align: right middle;
        color: {TEXT_DIM};
        padding: 0 0 0 1;
    }}
    """

    mode = reactive[str]("build")
    model = reactive[str]("—")
    session_id = reactive[str]("—")

    MODE_COLORS = {
        "plan": "#00ffff",
        "build": "#00ff80",
        "auto": "#b464ff",
    }

    MODE_ICONS = {
        "plan": "◈",
        "build": "◆",
        "auto": "⟡",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time = datetime.now()

    def compose(self) -> ComposeResult:
        yield Static(
            " ◆ DEPRESSION.AI ",
            id="header-logo",
        )
        yield Static("│", id="header-separator")
        yield Static(
            self._mode_display(),
            id="header-mode",
        )
        yield Static(
            self.model,
            id="header-model",
        )
        yield Static(
            f"ses:{self.session_id[:8]}",
            id="header-session",
        )
        yield Static(
            datetime.now().strftime("%H:%M"),
            id="header-time",
        )

    def _mode_display(self) -> str:
        icon = self.MODE_ICONS.get(self.mode, "◆")
        color = self.MODE_COLORS.get(self.mode, "#00c8ff")
        return f" [{color}]{icon} {self.mode.upper()}[/]"

    def set_mode(self, mode: str) -> None:
        """Update the current mode."""
        self.mode = mode
        try:
            self.query_one("#header-mode").update(self._mode_display())
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        """Update the model name."""
        self.model = model
        try:
            self.query_one("#header-model").update(model)
        except Exception:
            pass

    def set_session(self, session_id: str) -> None:
        """Update the session ID."""
        self.session_id = session_id
        try:
            self.query_one("#header-session").update(f"ses:{session_id[:8]}")
        except Exception:
            pass

    def update_time(self) -> None:
        """Update the clock."""
        try:
            self.query_one("#header-time").update(
                datetime.now().strftime("%H:%M")
            )
        except Exception:
            pass