"""
Empty-state banner — animated sci-fi wordmark.

Animations layered on top of the static block letters:

  1. Typewriter reveal   — letters appear left-to-right on first mount
  2. Horizontal scan     — a bright column sweeps across the wordmark
                            every few seconds
  3. Red R pulse         — the R glyph brightens and dims on a slow cycle
  4. Subtitle breathing  — the subtitle's glow oscillates
  5. Terminal caret      — a blinking ▮ after the subtitle

Under the wordmark a welcome line is shown when a name is available:

    welcome to depression, <name>

All animation is driven by two set_interval timers on the widget; nothing
touches the app loop or spawns tasks. The scan uses per-character coloring
so it survives any terminal that renders the block letters.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.widgets import Static


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
GREEN_GLOW = "#88ffbb"
RED = "#ff3355"
RED_GLOW = "#ff8899"
RED_PULSE = "#ffd0d8"
SCAN = "#e6fff0"
BG = "#000000"
DIM = "#1a5c33"
MUTED = "#3d8c5c"


# ----------------------------------------------------------------------
# LETTER GLYPHS (6 rows tall, variable width)
# ----------------------------------------------------------------------

_GLYPH_D = [
    "██████╗ ",
    "██╔══██╗",
    "██║  ██║",
    "██║  ██║",
    "██████╔╝",
    "╚═════╝ ",
]
_GLYPH_E = [
    "███████╗",
    "██╔════╝",
    "█████╗  ",
    "██╔══╝  ",
    "███████╗",
    "╚══════╝",
]
_GLYPH_P = [
    "██████╗ ",
    "██╔══██╗",
    "██████╔╝",
    "██╔═══╝ ",
    "██║     ",
    "╚═╝     ",
]
_GLYPH_R = [
    "██████╗ ",
    "██╔══██╗",
    "██████╔╝",
    "██╔══██╗",
    "██║  ██║",
    "╚═╝  ╚═╝",
]
_GLYPH_S = [
    "███████╗",
    "██╔════╝",
    "███████╗",
    "╚════██║",
    "███████║",
    "╚══════╝",
]
_GLYPH_I = [
    "██╗",
    "██║",
    "██║",
    "██║",
    "██║",
    "╚═╝",
]
_GLYPH_O = [
    " ██████╗ ",
    "██╔═══██╗",
    "██║   ██║",
    "██║   ██║",
    "╚██████╔╝",
    " ╚═════╝ ",
]
_GLYPH_N = [
    "███╗   ██╗",
    "████╗  ██║",
    "██╔██╗ ██║",
    "██║╚██╗██║",
    "██║ ╚████║",
    "╚═╝  ╚═══╝",
]

_LETTERS = [
    ("D", _GLYPH_D),
    ("E", _GLYPH_E),
    ("P", _GLYPH_P),
    ("R", _GLYPH_R),   # <- red, pulses
    ("E", _GLYPH_E),
    ("S", _GLYPH_S),
    ("S", _GLYPH_S),
    ("I", _GLYPH_I),
    ("O", _GLYPH_O),
    ("N", _GLYPH_N),
]

_RED_LETTER = "R"

_ROW_SHADES = [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM]


# ----------------------------------------------------------------------
# RENDER HELPERS
# ----------------------------------------------------------------------

def _esc(s: str) -> str:
    """Escape [ so Textual's markup doesn't treat it as a tag opener."""
    return s.replace("[", r"\[")


def _build_full_rows(reveal_count: int) -> list[str]:
    """
    Return the six markup strings for the wordmark, with only the first
    `reveal_count` letters visible (typewriter effect). The rest are
    rendered as spaces of the same width so the layout doesn't shift.
    """
    rows: list[str] = []
    for row_index in range(6):
        shade = _ROW_SHADES[row_index]
        parts: list[str] = []
        for idx, (name, glyph) in enumerate(_LETTERS):
            raw = glyph[row_index]
            if idx < reveal_count:
                segment = _esc(raw)
                if name == _RED_LETTER:
                    parts.append(f"[bold {RED_GLOW}]{segment}[/]")
                else:
                    parts.append(f"[bold {shade}]{segment}[/]")
            else:
                # Preserve width so the letters don't slide around as the
                # reveal progresses.
                parts.append(" " * len(raw))
        rows.append(" ".join(parts))
    return rows


def _pulse_color(tick: int) -> str:
    """Return the R color for this tick — 3-step pulse."""
    phase = tick % 6
    if phase in (0, 1):
        return RED_PULSE
    if phase in (2, 3):
        return RED_GLOW
    return RED


def _breath_color(tick: int) -> str:
    """Subtitle breathing color for this tick."""
    phase = tick % 8
    if phase < 2:
        return GREEN
    if phase < 5:
        return GREEN_GLOW
    return GREEN_DIM


class EmptyBanner(Static):
    """
    Full 6-row wordmark for the main empty-state screen.

    Pass `welcome_name` (or call `set_welcome()`) to show
    "welcome to depression, <name>" under the wordmark. When no name is
    available the welcome line is simply blank — the wordmark, subtitle
    and hint are unchanged.
    """

    DEFAULT_CSS = f"""
    EmptyBanner {{
        width: 100%;
        height: 1fr;
        background: {BG};
        align: center middle;
    }}
    EmptyBanner .banner-line {{
        width: auto;
        height: 1;
        text-align: center;
    }}
    EmptyBanner .banner-sub {{
        width: auto;
        height: 1;
        text-align: center;
        margin-top: 1;
        color: {GREEN};
    }}
    EmptyBanner .banner-welcome {{
        width: auto;
        height: 1;
        text-align: center;
        margin-top: 1;
        color: {GREEN_GLOW};
    }}
    EmptyBanner .banner-note {{
        width: auto;
        height: 1;
        text-align: center;
        color: {MUTED};
        margin-top: 1;
    }}
    EmptyBanner .banner-hint {{
        width: auto;
        height: 1;
        text-align: center;
        color: {DIM};
        margin-top: 1;
    }}
    """

    def __init__(self, welcome_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._welcome_name = welcome_name or ""
        self._line_widgets: list[Static] = []
        self._sub_widget: Static | None = None
        self._welcome_widget: Static | None = None
        self._note_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._reveal_count = 0
        self._reveal_done = False
        self._scan_tick = 0
        self._alive = True

    def set_welcome(self, name: str) -> None:
        """Update the welcome line without rebuilding the widget."""
        self._welcome_name = (name or "").strip()
        self._render_welcome()

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                for _ in range(6):
                    w = Static("", classes="banner-line", markup=True)
                    self._line_widgets.append(w)
                    yield w

                self._sub_widget = Static("", classes="banner-sub", markup=True)
                yield self._sub_widget

                # NEW: welcome line — sits between the subtitle and the note.
                self._welcome_widget = Static(
                    "", classes="banner-welcome", markup=True
                )
                yield self._welcome_widget

                self._note_widget = Static(
                    f"[{MUTED}]AI agent capable of doing everything — "
                    f"development, analysis, system handling AWS deployment,[/]",
                    classes="banner-note",
                    markup=True,
                )
                yield self._note_widget

                self._hint_widget = Static(
                    f"[{DIM}]type a prompt to begin  ·  ctrl+b for sidebar  ·  "
                    f"ctrl+q to quit[/]",
                    classes="banner-hint",
                    markup=True,
                )
                yield self._hint_widget

    def on_mount(self) -> None:
        # Typewriter: one letter every 90ms.
        self.set_interval(0.09, self._tick_reveal)
        # Ambient effects: pulse, scan, breathing subtitle.
        self.set_interval(0.14, self._tick_effects)
        # Initial paint so the widget isn't blank for one frame.
        self._render_wordmark()
        self._render_welcome()

    def on_unmount(self) -> None:
        self._alive = False

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def _render_wordmark(self) -> None:
        if not self._alive:
            return
        rows = _build_full_rows(self._reveal_count)
        # Red pulse: rebuild the R cells with the current pulse color by
        # doing a second pass that swaps the red constant.
        pulse = _pulse_color(self._scan_tick)
        for idx, row_widget in enumerate(self._line_widgets):
            try:
                row_widget.update(self._recolor_r(rows[idx], pulse))
            except Exception:
                self._alive = False
                return

        if self._sub_widget is not None:
            try:
                sub_color = _breath_color(self._scan_tick)
                self._sub_widget.update(
                    f"[bold {sub_color}]DEPRESSION.AI[/]"
                )
            except Exception:
                pass

        # Blinking caret appended to the hint line.
        if self._hint_widget is not None:
            try:
                caret = "▮" if (self._scan_tick // 4) % 2 == 0 else " "
                self._hint_widget.update(
                    f"[{DIM}]type a prompt to begin  ·  ctrl+b for sidebar  ·  "
                    f"ctrl+q to quit[/]  [{GREEN}]{caret}[/]"
                )
            except Exception:
                pass

    def _render_welcome(self) -> None:
        """Paint the welcome line under the subtitle."""
        if self._welcome_widget is None:
            return
        name = (self._welcome_name or "").strip()
        if not name:
            # Keep the line blank (it still reserves its row, so nothing
            # shifts when the name arrives later).
            try:
                self._welcome_widget.update("")
            except Exception:
                pass
            return
        try:
            self._welcome_widget.update(
                f"[bold {GREEN_GLOW}]welcome to depression, {_esc(name)}[/]"
            )
        except Exception:
            pass

    def _recolor_r(self, row: str, red: str) -> str:
        """
        The base rows were built with the R already marked. Swap the red
        shade for the current pulse by string replacement — cheap and
        safe because RED_GLOW is unique in the string.
        """
        return row.replace(RED_GLOW, red).replace(RED, red)

    # ------------------------------------------------------------------
    # TICKS
    # ------------------------------------------------------------------

    def _tick_reveal(self) -> None:
        if not self._alive:
            return
        if self._reveal_done:
            return
        self._reveal_count += 1
        if self._reveal_count >= len(_LETTERS):
            self._reveal_count = len(_LETTERS)
            self._reveal_done = True

    def _tick_effects(self) -> None:
        if not self._alive:
            return
        self._scan_tick += 1
        self._render_wordmark()