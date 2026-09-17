"""
Collapsible sidebar container.

The sidebar is hidden by default (width 0). Ctrl+B toggles it to a
fixed width. The transition is instant, but the caller can animate the
`width` reactive if desired — Textual layouts handle the reflow.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Static


GREEN = "#00ff66"
GREEN_GLOW = "#88ffbb"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BG = "#000000"
BORDER = "#0a3d20"


class Sidebar(Vertical):
    DEFAULT_CSS = f"""
    Sidebar {{
        width: 0;
        min-width: 0;
        max-width: 46;
        height: 1fr;
        background: {BG};
        border-left: solid {BORDER};
        layout: vertical;
        transition: width 200ms in_out_cubic;
    }}
    Sidebar.open {{
        width: 46;
        min-width: 46;
    }}
    Sidebar #side-title {{
        height: 1;
        padding: 0 1;
        background: {BG};
        color: {GREEN};
        text-style: bold;
        border-bottom: solid {BORDER};
    }}
    Sidebar #side-tabs {{
        height: 1;
        layout: horizontal;
        background: {BG};
        border-bottom: solid {BORDER};
    }}
    Sidebar .side-tab {{
        width: 1fr;
        min-width: 8;
        height: 1;
        background: transparent;
        border: none;
        color: {DIM};
        text-style: bold;
    }}
    Sidebar .side-tab:hover {{ color: {MUTED}; }}
    Sidebar .side-tab.active {{ color: {GREEN}; }}
    Sidebar #side-content {{
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
        background: {BG};
        padding: 1 0;
    }}
    """

    is_open: reactive[bool] = reactive(False)

    def __init__(self, panels: list, **kwargs):
        super().__init__(**kwargs)
        self._panel_widgets = panels  # [(name, widget), ...]

    def compose(self) -> ComposeResult:
        yield Static("▌ SIDEBAR", id="side-title", markup=True)
        with Horizontal(id="side-tabs"):
            for i, (name, _widget) in enumerate(self._panel_widgets):
                label = name.upper()[:8]
                yield Button(label, id=f"tab-{name}",
                             classes="side-tab" + (" active" if i == 0 else ""))
        with VerticalScroll(id="side-content"):
            for _name, widget in self._panel_widgets:
                yield widget

    def watch_is_open(self, open_: bool) -> None:
        try:
            if open_:
                self.add_class("open")
            else:
                self.remove_class("open")
        except Exception:
            pass

    def toggle(self) -> None:
        self.is_open = not self.is_open