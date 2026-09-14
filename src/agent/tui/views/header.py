"""
Depression.AI futuristic hero header.

The header is intentionally more like a small web-app hero than a
traditional terminal banner: large product branding, an animated angel
mascot, mode/model context, and a restrained glass/neon visual language.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


# Futuristic dark-glass palette. Kept local to this component so the hero is
# visually distinctive without changing the public TUI API.
BG = "#070711"
PANEL = "#10101d"
PANEL_2 = "#151528"
BORDER = "#292947"
TEXT = "#f4f2ff"
MUTED = "#9290aa"
DIM = "#55536d"
CYAN = "#62ddff"
VIOLET = "#b78cff"
PINK = "#ff86d5"
GREEN = "#6fffc0"


class HeaderBar(Widget):
    """Large animated Depression.AI hero/header.

    Public methods are compatible with the previous HeaderBar so the rest of
    the TUI does not need to know about the visual redesign.
    """

    DEFAULT_CSS = f"""
    HeaderBar {{
        height: 10;
        min-height: 8;
        background: {BG};
        dock: top;
        layout: vertical;
        border-bottom: solid {BORDER};
        padding: 0 2;
    }}

    #hero-stage {{
        height: 7;
        width: 1fr;
        background: {PANEL};
        border: round {BORDER};
        layout: horizontal;
        padding: 0 2;
    }}

    #angel {{
        width: 15;
        height: 7;
        color: {VIOLET};
        content-align: center middle;
        text-style: bold;
    }}

    #hero-brand {{
        width: 1fr;
        height: 7;
        content-align: center middle;
        color: {TEXT};
        text-style: bold;
    }}

    #hero-context {{
        width: 26;
        height: 7;
        content-align: center middle;
        color: {MUTED};
        border-left: solid {BORDER};
        padding: 0 1;
    }}

    #hero-meta {{
        height: 2;
        width: 1fr;
        layout: horizontal;
        padding: 0 2;
        color: {DIM};
    }}

    #header-mode {{ width: 18; color: {GREEN}; text-style: bold; content-align: left middle; }}
    #header-model {{ width: 1fr; color: {MUTED}; content-align: center middle; }}
    #header-session {{ width: 24; color: {DIM}; content-align: right middle; }}
    #header-time {{ width: 8; color: {DIM}; content-align: right middle; }}
    """

    mode = reactive[str]("build")
    model = reactive[str]("—")
    session_id = reactive[str]("—")

    MODE_COLORS = {
        "plan": CYAN,
        "build": GREEN,
        "auto": VIOLET,
    }

    MODE_ICONS = {
        "plan": "◈",
        "build": "◆",
        "auto": "⟡",
    }

    # The mascot is deliberately pure Unicode/ASCII so it works offline and
    # does not require an image protocol, sixel, kitty graphics, or assets.
    ANGEL_FRAMES = (
        "   ✦   \n  ╱│╲  \n ✧(◡)✧ \n  ╲│╱  \n   ♡   ",
        "  ✧ ✧  \n   ╲│╱  \n ✦(◡)✦ \n   ╱│╲  \n    ♡   ",
        "   ✦   \n  ╲│╱  \n ✧(◡)✧ \n  ╱│╲  \n   ♡   ",
        "  ✧ ✧  \n   ╱│╲  \n ✦(◡)✦ \n   ╲│╱  \n    ♡   ",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._angel_index = 0
        self._timer = None

    def compose(self) -> ComposeResult:
        with Widget(id="hero-stage"):
            yield Static(self.ANGEL_FRAMES[0], id="angel")
            yield Static(self._brand_art(), id="hero-brand")
            yield Static(self._context_text(), id="hero-context")

        with Widget(id="hero-meta"):
            yield Static(self._mode_display(), id="header-mode")
            yield Static(self.model, id="header-model")
            yield Static(f"ses:{self.session_id[:8]}", id="header-session")
            yield Static(datetime.now().strftime("%H:%M"), id="header-time")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.55, self._animate_angel)

    def _animate_angel(self) -> None:
        self._angel_index = (self._angel_index + 1) % len(self.ANGEL_FRAMES)
        try:
            self.query_one("#angel", Static).update(self.ANGEL_FRAMES[self._angel_index])
        except Exception:
            pass

    def _brand_art(self) -> str:
        return (
            "✦  D E P R E S S I O N . A I  ✦\n"
            f"{MUTED}agentic intelligence • local + cloud • human controlled{TEXT}"
        )

    def _context_text(self) -> str:
        return "◌ ONLINE\n\nPLAN  →  BUILD\n\nVERIFY  →  DONE"

    def _mode_display(self) -> str:
        icon = self.MODE_ICONS.get(self.mode, "◆")
        color = self.MODE_COLORS.get(self.mode, CYAN)
        return f"[{color}]{icon} {self.mode.upper()}[/]"

    def set_mode(self, mode: str) -> None:
        """Update the current execution mode."""
        self.mode = mode
        try:
            self.query_one("#header-mode", Static).update(self._mode_display())
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        """Update the active model name."""
        self.model = model
        try:
            self.query_one("#header-model", Static).update(model)
        except Exception:
            pass

    def set_session(self, session_id: str) -> None:
        """Update the current session id."""
        self.session_id = session_id
        try:
            self.query_one("#header-session", Static).update(f"ses:{session_id[:8]}")
        except Exception:
            pass

    def update_time(self) -> None:
        """Refresh the clock."""
        try:
            self.query_one("#header-time", Static).update(datetime.now().strftime("%H:%M"))
        except Exception:
            pass
