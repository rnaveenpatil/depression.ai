"""Compact terminal chrome for Depression.AI."""
from __future__ import annotations
from datetime import datetime
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive

BG = "#000000"
BORDER = "#242424"
TEXT = "#f4f4f4"
MUTED = "#777777"
CYAN = "#6bdcff"
VIOLET = "#c29cff"
GREEN = "#75f0a8"


class HeaderBar(Widget):
    """Small CLI header: brand, mascot, mode and model. No dashboard chrome."""

    DEFAULT_CSS = f"""
    HeaderBar {{ height: 3; background: {BG}; border-bottom: solid {BORDER}; layout: horizontal; }}
    #mascot {{ width: 7; height: 3; color: {VIOLET}; content-align: center middle; }}
    #brand {{ width: 22; height: 3; color: {TEXT}; content-align: left middle; text-style: bold; }}
    #header-mode {{ width: 14; height: 3; color: {GREEN}; content-align: left middle; }}
    #header-model {{ width: 1fr; height: 3; color: {MUTED}; content-align: right middle; }}
    #header-session {{ width: 14; height: 3; color: {MUTED}; content-align: right middle; }}
    #header-time {{ width: 7; height: 3; color: {MUTED}; content-align: right middle; }}
    """

    mode = reactive[str]("build")
    model = reactive[str]("no model")
    session_id = reactive[str]("—")

    MODE_ICONS = {"plan": "◇", "build": "◆", "auto": "⟡"}
    ANGEL_FRAMES = ("✦♡✦", "♡✦♡", "✦♡✦", "♡✦♡")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(self.ANGEL_FRAMES[0], id="mascot")
        yield Static("DEPRESSION.AI", id="brand")
        yield Static(self._mode_display(), id="header-mode")
        yield Static(self.model, id="header-model")
        yield Static(f"ses:{self.session_id[:8]}", id="header-session")
        yield Static(datetime.now().strftime("%H:%M"), id="header-time")

    def on_mount(self) -> None:
        self.set_interval(0.7, self._animate)

    def _animate(self) -> None:
        self._frame = (self._frame + 1) % len(self.ANGEL_FRAMES)
        try:
            self.query_one("#mascot", Static).update(self.ANGEL_FRAMES[self._frame])
            self.query_one("#header-time", Static).update(datetime.now().strftime("%H:%M"))
        except Exception:
            pass

    def _mode_display(self) -> str:
        return f"{self.MODE_ICONS.get(self.mode, '◆')} {self.mode.upper()}"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        try:
            self.query_one("#header-mode", Static).update(self._mode_display())
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        self.model = model or "no model"
        try:
            self.query_one("#header-model", Static).update(self.model)
        except Exception:
            pass

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id or "—"
        try:
            self.query_one("#header-session", Static).update(f"ses:{self.session_id[:8]}")
        except Exception:
            pass
