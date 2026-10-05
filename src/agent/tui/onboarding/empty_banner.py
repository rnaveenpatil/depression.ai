"""
Empty-state banner — animated sci-fi wordmark.

Two sizes:

  * `EmptyBanner`   — the full 6-row block letters, used on the main
                      empty-state screen. Animations: typewriter reveal,
                      scan line, red R pulse, subtitle breathing, caret.
  * `CompactBanner` — a dedicated 4-row small wordmark for the welcome /
                      sign-in page. It uses hand-tuned half-block glyphs
                      (not a collapse of the big ones) so the shapes stay
                      crisp, and runs a different animation: a soft glow
                      that travels left-to-right across the whole wordmark,
                      plus a breathing subtitle.

All animation is driven by set_interval timers on the widget; nothing
touches the app loop or spawns tasks.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.widgets import Static


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
GREEN_GLOW = "#88ffbb"
GREEN_SOFT = "#0a3a1f"
RED = "#ff3355"
RED_GLOW = "#ff8899"
RED_PULSE = "#ffd0d8"
BG = "#000000"
DIM = "#1a5c33"
MUTED = "#3d8c5c"


# ======================================================================
# LARGE WORDMARK (6 rows, box-drawing block letters)
# ======================================================================

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
    ("R", _GLYPH_R),
    ("E", _GLYPH_E),
    ("S", _GLYPH_S),
    ("S", _GLYPH_S),
    ("I", _GLYPH_I),
    ("O", _GLYPH_O),
    ("N", _GLYPH_N),
]

_RED_LETTER = "R"
_ROW_SHADES = [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM]


# ======================================================================
# COMPACT WORDMARK (4 rows, hand-tuned half-blocks)
# ======================================================================
#
# Every glyph is exactly 5 columns wide and 4 rows tall. They are drawn
# with half-block characters (`▀ ▄ █`) so they render crisply at terminal
# resolution without depending on box-drawing corner alignment.

_C_GLYPH_D = [
    "████ ",
    "█  ██",
    "█  ██",
    "████ ",
]
_C_GLYPH_E = [
    "█████",
    "█    ",
    "███  ",
    "█████",
]
_C_GLYPH_P = [
    "████ ",
    "█  ██",
    "████ ",
    "█    ",
]
_C_GLYPH_R = [
    "████ ",
    "█  ██",
    "████ ",
    "█ ██ ",
]
_C_GLYPH_S = [
    "█████",
    "█    ",
    " ████",
    "████ ",
]
_C_GLYPH_I = [
    "███",
    " █ ",
    " █ ",
    "███",
]
_C_GLYPH_O = [
    "████ ",
    "█  ██",
    "█  ██",
    "████ ",
]
_C_GLYPH_N = [
    "█  ██",
    "██ ██",
    "█ ███",
    "█  ██",
]

_COMPACT_LETTERS = [
    ("D", _C_GLYPH_D),
    ("E", _C_GLYPH_E),
    ("P", _C_GLYPH_P),
    ("R", _C_GLYPH_R),
    ("E", _C_GLYPH_E),
    ("S", _C_GLYPH_S),
    ("S", _C_GLYPH_S),
    ("I", _C_GLYPH_I),
    ("O", _C_GLYPH_O),
    ("N", _C_GLYPH_N),
]
_COMPACT_ROWS = 4


# ======================================================================
# COLOR HELPERS
# ======================================================================

def _pulse_color(tick: int) -> str:
    phase = tick % 6
    if phase in (0, 1):
        return RED_PULSE
    if phase in (2, 3):
        return RED_GLOW
    return RED


def _breath_color(tick: int) -> str:
    phase = tick % 8
    if phase < 2:
        return GREEN
    if phase < 5:
        return GREEN_GLOW
    return GREEN_DIM


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(a: str, b: str, t: float) -> str:
    """Linear blend between two hex colors; t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return _rgb_to_hex((
        int(ar + (br - ar) * t),
        int(ag + (bg - ag) * t),
        int(ab + (bb - ab) * t),
    ))


def _esc(s: str) -> str:
    return s.replace("[", r"\[")


# ======================================================================
# COMPACT RENDER — the glow wave
# ======================================================================

def render_compact_wordmark(tick: int, wave_width: int = 3) -> list[str]:
    """
    Render the compact wordmark with a soft glow traveling left-to-right.

    `tick` advances the wave. The wave is a bell of `wave_width` columns;
    each letter's column distance from the wave center determines how far
    it blends from GREEN_DIM toward GREEN_GLOW. The R always sits at RED
    tones and gets a slightly wider bell so it stays the focal point.
    """
    # Total rendered width of the compact wordmark including 1-space gaps.
    total = sum(len(g[0]) for _, g in _COMPACT_LETTERS) + (len(_COMPACT_LETTERS) - 1)
    if total <= 0:
        return [""] * _COMPACT_ROWS

    # Wave center sweeps across the full width and wraps.
    span = total + 2 * wave_width
    wave_center = (tick % span) - wave_width

    rows = [""] * _COMPACT_ROWS
    col_cursor = 0
    for name, glyph in _COMPACT_LETTERS:
        width = len(glyph[0])
        letter_center = col_cursor + (width - 1) / 2.0

        # Bell-shaped intensity in [0, 1] based on distance to wave.
        dist = abs(letter_center - wave_center)
        denom = max(1.0, float(wave_width))
        intensity = max(0.0, 1.0 - (dist / denom))
        # Smoothstep for a nicer falloff.
        intensity = intensity * intensity * (3 - 2 * intensity)

        if name == _RED_LETTER:
            # R rides the same wave but stays in the red family.
            base = _mix(RED_GLOW, RED_PULSE, intensity)
            color = base
        else:
            base = GREEN_DIM if intensity < 0.5 else GREEN
            color = _mix(base, GREEN_GLOW, intensity)

        for row_index in range(_COMPACT_ROWS):
            cell = glyph[row_index].ljust(width)
            rows[row_index] += f"[bold {color}]{_esc(cell)}[/]"
        col_cursor += width + 1
        if name != _COMPACT_LETTERS[-1][0] or True:
            # add the gap (trailing gap is harmless)
            for row_index in range(_COMPACT_ROWS):
                rows[row_index] += " "
        col_cursor += 0

    # Trim the trailing gap we added after the last letter.
    for row_index in range(_COMPACT_ROWS):
        rows[row_index] = rows[row_index].rstrip()

    return rows


# ======================================================================
# LARGE RENDER
# ======================================================================

def _build_full_rows(reveal_count: int) -> list[str]:
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
                parts.append(" " * len(raw))
        rows.append(" ".join(parts))
    return rows


# ======================================================================
# WIDGETS
# ======================================================================

class EmptyBanner(Static):
    """Full 6-row wordmark for the main empty-state screen."""

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._line_widgets: list[Static] = []
        self._sub_widget: Static | None = None
        self._note_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._reveal_count = 0
        self._reveal_done = False
        self._scan_tick = 0
        self._alive = True

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                for _ in range(6):
                    w = Static("", classes="banner-line", markup=True)
                    self._line_widgets.append(w)
                    yield w
                self._sub_widget = Static("", classes="banner-sub", markup=True)
                yield self._sub_widget
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
        self.set_interval(0.09, self._tick_reveal)
        self.set_interval(0.14, self._tick_effects)
        self._render_wordmark()

    def on_unmount(self) -> None:
        self._alive = False

    def _render_wordmark(self) -> None:
        if not self._alive:
            return
        rows = _build_full_rows(self._reveal_count)
        pulse = _pulse_color(self._scan_tick)
        for idx, widget in enumerate(self._line_widgets):
            try:
                widget.update(rows[idx].replace(RED_GLOW, pulse).replace(RED, pulse))
            except Exception:
                self._alive = False
                return
        if self._sub_widget is not None:
            try:
                self._sub_widget.update(
                    f"[bold {_breath_color(self._scan_tick)}]DEPRESSION.AI[/]"
                )
            except Exception:
                pass
        if self._hint_widget is not None:
            try:
                caret = "▮" if (self._scan_tick // 4) % 2 == 0 else " "
                self._hint_widget.update(
                    f"[{DIM}]type a prompt to begin  ·  ctrl+b for sidebar  ·  "
                    f"ctrl+q to quit[/]  [{GREEN}]{caret}[/]"
                )
            except Exception:
                pass

    def _tick_reveal(self) -> None:
        if not self._alive or self._reveal_done:
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


class CompactBanner(Static):
    """
    The small welcome-page wordmark: 4 crisp rows, glowing wave animation.

    Different from `EmptyBanner` on purpose — no typewriter, no scan line.
    Instead a soft glow travels left-to-right across the whole wordmark,
    and the subtitle breathes.
    """

    DEFAULT_CSS = f"""
    CompactBanner {{
        width: 100%;
        height: auto;
        content-align: center middle;
        color: {GREEN};
        margin: 0 0 1 0;
    }}
    CompactBanner .compact-line {{
        width: 100%;
        height: 1;
        text-align: center;
    }}
    CompactBanner .compact-sub {{
        width: 100%;
        height: 1;
        text-align: center;
        color: {MUTED};
        margin-top: 1;
    }}
    """

    def __init__(self, subtitle: str = "sign in to continue", **kwargs):
        super().__init__(**kwargs)
        self._subtitle = subtitle
        self._line_widgets: list[Static] = []
        self._sub_widget: Static | None = None
        self._tick = 0
        self._alive = True

    def compose(self) -> ComposeResult:
        for _ in range(_COMPACT_ROWS):
            w = Static("", classes="compact-line", markup=True)
            self._line_widgets.append(w)
            yield w
        self._sub_widget = Static("", classes="compact-sub", markup=True)
        yield self._sub_widget

    def on_mount(self) -> None:
        self._paint()
        self.set_interval(0.12, self._animate)

    def on_unmount(self) -> None:
        self._alive = False

    def _paint(self) -> None:
        if not self._alive:
            return
        rows = render_compact_wordmark(self._tick)
        for widget, row in zip(self._line_widgets, rows):
            try:
                widget.update(row)
            except Exception:
                self._alive = False
                return
        if self._sub_widget is not None:
            color = _breath_color(self._tick)
            try:
                self._sub_widget.update(f"[{color}]· {self._subtitle} ·[/]")
            except Exception:
                pass

    def _animate(self) -> None:
        if not self._alive:
            return
        self._tick += 1
        self._paint()


__all__ = [
    "EmptyBanner",
    "CompactBanner",
    "render_compact_wordmark",
]