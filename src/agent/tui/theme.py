"""
Theme tokens for the TUI.

Single source of truth. All colors used by the app and widgets come
from here so the palette can be swapped in one place.
"""

from __future__ import annotations

from textual.theme import Theme


GREEN       = "#00ff66"
GREEN_DIM   = "#00aa44"
GREEN_FAINT = "#005522"
GREEN_GLOW  = "#88ffbb"
AMBER       = "#ffcc44"
ERROR       = "#ff4466"
TEXT        = "#aaffcc"
MUTED       = "#3d8c5c"
DIM         = "#1a5c33"
BG          = "#000000"
PANEL       = "#031008"
RAISED      = "#061a0f"
BORDER      = "#0a3d20"

RED_DIFF    = "#ff4466"
RED_DIFF_DIM = "#7a1f30"
GREEN_DIFF  = "#00ff66"
GREEN_DIFF_DIM = "#0a5c2a"


MATRIX_THEME = Theme(
    name="matrix",
    primary=GREEN,
    secondary=GREEN_DIM,
    accent=GREEN_GLOW,
    warning=AMBER,
    error=ERROR,
    success=GREEN,
    surface=PANEL,
    panel=RAISED,
    boost="#0d2a17",
    foreground=TEXT,
    background=BG,
    dark=True,
)


__all__ = [
    "GREEN", "GREEN_DIM", "GREEN_FAINT", "GREEN_GLOW",
    "AMBER", "ERROR", "TEXT", "MUTED", "DIM",
    "BG", "PANEL", "RAISED", "BORDER",
    "RED_DIFF", "RED_DIFF_DIM", "GREEN_DIFF", "GREEN_DIFF_DIM",
    "MATRIX_THEME",
]