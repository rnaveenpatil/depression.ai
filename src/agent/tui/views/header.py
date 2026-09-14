"""Polished terminal-style Depression.AI header with an animated angel princess mascot."""

from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive

BG = "#070711"
BORDER = "#272640"
TEXT = "#f4f2ff"
MUTED = "#77738e"
CYAN = "#54d9ff"
VIOLET = "#b78cff"
PINK = "#ff86d5"
GREEN = "#62f5b0"


class HeaderBar(Widget):
    """Minimal OpenCode-inspired chrome with a distinctive Depression.AI identity."""

    DEFAULT_CSS = f"""
    HeaderBar {{ height: 7; background: {BG}; border-bottom: solid {BORDER}; layout: vertical; }}
    #hero-row {{ height: 5; width: 1fr; layout: horizontal; content-align: center middle; padding: 0 2; }}
    #angel-left, #angel-right {{ width: 12; height: 5; color: {VIOLET}; content-align: center middle; text-style: bold; }}
    #hero-brand {{ width: 1fr; height: 5; content-align: center middle; color: {TEXT}; text-style: bold; }}
    #hero-mode {{ width: 18; height: 5; content-align: center middle; color: {GREEN}; }}
    #hero-meta {{ height: 2; width: 1fr; layout: horizontal; padding: 0 2; color: {MUTED}; }}
    #header-mode {{ width: 18; color: {GREEN}; text-style: bold; }}
    #header-model {{ width: 1fr; color: {MUTED}; content-align: center middle; }}
    #header-session {{ width: 24; color: {MUTED}; content-align: right middle; }}
    #header-time {{ width: 8; color: {MUTED}; content-align: right middle; }}
    """

    mode = reactive[str]("build")
    model = reactive[str]("—")
    session_id = reactive[str]("—")

    MODE_COLORS = {"plan": CYAN, "build": GREEN, "auto": VIOLET}
    MODE_ICONS = {"plan": "◇", "build": "◆", "auto": "⟡"}

    # Original Unicode art avoids terminal graphics dependencies. The character
    # is intentionally an original angelic-princess mascot, not a Disney asset.
    ANGEL_FRAMES = (
        "  ✦  \n ╭♡╮ \n(◕‿◕)\n ╰┼╯ \n  ♡  ",
        " ✧ ✧ \n  ╭♡╮\n (◕‿◕)\n  ╰┼╯\n   ♡  ",
        "  ✦  \n ╭♡╮ \n(◕‿◕)\n ╰┼╯ \n  ✧  ",
        " ✧ ✧ \n  ╭♡╮\n (◕‿◕)\n  ╰┼╯\n   ♡  ",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._angel_index = 0

    def compose(self) -> ComposeResult:
        with Widget(id="hero-row"):
            yield Static(self.ANGEL_FRAMES[0], id="angel-left")
            yield Static(self._brand_art(), id="hero-brand")
            yield Static(self.ANGEL_FRAMES[2], id="angel-right")
            yield Static(self._mode_display(), id="hero-mode")
        with Widget(id="hero-meta"):
            yield Static(self._mode_display(), id="header-mode")
            yield Static(self.model, id="header-model")
            yield Static(f"ses:{self.session_id[:8]}", id="header-session")
            yield Static(datetime.now().strftime("%H:%M"), id="header-time")

    def on_mount(self) -> None:
        self.set_interval(0.55, self._animate_angel)

    def _animate_angel(self) -> None:
        self._angel_index = (self._angel_index + 1) % len(self.ANGEL_FRAMES)
        frame = self.ANGEL_FRAMES[self._angel_index]
        try:
            self.query_one("#angel-left", Static).update(frame)
            self.query_one("#angel-right", Static).update(self.ANGEL_FRAMES[(self._angel_index + 2) % len(self.ANGEL_FRAMES)])
        except Exception:
            pass

    def _brand_art(self) -> str:
        return f"✦  D E P R E S S I O N . A I  ✦\n[{MUTED}]agentic intelligence · local + cloud[/]"

    def _mode_display(self) -> str:
        icon = self.MODE_ICONS.get(self.mode, "◆")
        color = self.MODE_COLORS.get(self.mode, CYAN)
        return f"[{color}]{icon} {self.mode.upper()}[/]"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        try:
            self.query_one("#header-mode", Static).update(self._mode_display())
            self.query_one("#hero-mode", Static).update(self._mode_display())
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        self.model = model
        try: self.query_one("#header-model", Static).update(model)
        except Exception: pass

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id
        try: self.query_one("#header-session", Static).update(f"ses:{session_id[:8]}")
        except Exception: pass
