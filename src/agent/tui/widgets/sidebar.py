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
    Sidebar.compact.open {{
        width: 34;
        min-width: 34;
        max-width: 34;
    }}
    Sidebar.overlay {{
        position: absolute;
        height: 1fr;
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
        height: auto;
        background: {BG};
        border-bottom: solid {BORDER};
        padding: 0 1;
    }}
    Sidebar .side-tab-row {{
        height: 1;
        width: 100%;
        layout: horizontal;
    }}
    Sidebar .side-tab {{
        width: 1fr;
        min-width: 0;
        height: 1;
        background: transparent;
        border: none;
        color: {MUTED};
        text-style: bold;
        padding: 0 1;
    }}
    Sidebar .side-tab:hover {{
        background: {BORDER};
        color: {TEXT};
    }}
    Sidebar .side-tab:focus {{
        background: {BORDER};
        color: {GREEN_GLOW};
    }}
    Sidebar .side-tab.active {{
        background: {BORDER};
        color: {GREEN};
        text-style: bold;
    }}
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
        self._screen_width = 120
        self._compact = False
        self._overlay = False

    def compose(self) -> ComposeResult:
        yield Static("▌ SIDEBAR", id="side-title", markup=True)
        # Two tabs per row: four compact rows instead of eight full rows.
        # This gives pasteable connection forms much more usable height.
        with Vertical(id="side-tabs"):
            short_labels = {"llmcost": "COST", "context": "CONTEXT"}
            for i in range(0, len(self._panel_widgets), 2):
                with Horizontal(classes="side-tab-row"):
                    for idx, (name, _widget) in enumerate(self._panel_widgets[i:i + 2]):
                        label = short_labels.get(name, name.upper())
                        yield Button(
                            label, id=f"tab-{name}",
                            classes="side-tab" + (
                                " active" if i + idx == 0 else ""
                            ),
                        )
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
        self._apply_size()

    def apply_responsive(self, width: int, height: int) -> None:
        """Compact on mid-width terminals and overlay on narrow ones."""
        self._screen_width = max(1, int(width))
        compact = self._screen_width < 110
        overlay = self._screen_width < 90
        if compact != self._compact or overlay != self._overlay:
            self._compact = compact
            self._overlay = overlay
            try:
                if compact:
                    self.add_class("compact")
                else:
                    self.remove_class("compact")
                if overlay:
                    self.add_class("overlay")
                else:
                    self.remove_class("overlay")
            except Exception:
                pass
        self._apply_size()

    def _apply_size(self) -> None:
        try:
            if not self.is_open:
                self.styles.width = 0
                self.styles.min_width = 0
                self.styles.max_width = 46
                self.styles.offset = (0, 0)
                self.styles.height = "1fr"
                return
            side = 34 if self._compact else 46
            if self._overlay:
                side = min(40, max(28, self._screen_width - 10))
                self.styles.width = side
                self.styles.min_width = side
                self.styles.max_width = side
                self.styles.offset = (max(0, self._screen_width - side), 0)
            else:
                self.styles.width = side
                self.styles.min_width = side
                self.styles.max_width = 46
                self.styles.offset = (0, 0)
            self.styles.height = "1fr"
            self.refresh(layout=True)
        except Exception:
            pass

    def toggle(self) -> None:
        self.is_open = not self.is_open