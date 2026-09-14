"""
Neon Progress Bar Widget

A progress bar with cyberpunk neon glow:
- Gradient fill from cyan to purple
- Animated shimmer effect
- Percentage display
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class NeonProgressBar(Widget):
    """A neon-styled progress bar."""

    DEFAULT_CSS = """
    NeonProgressBar {
        height: 1;
        width: 100%;
    }
    """

    progress = reactive[float](0.0)
    label = reactive[str]("")

    def __init__(self, progress: float = 0.0, label: str = "", **kwargs):
        super().__init__(**kwargs)
        self.progress = max(0.0, min(1.0, progress))
        self.label = label

    def render(self) -> str:
        width = self.size.width if self.size.width > 0 else 40
        bar_width = max(0, width - 8)

        filled = int(bar_width * self.progress)
        empty = bar_width - filled

        bar = "█" * filled + "░" * empty
        pct = f"{self.progress * 100:.0f}%"

        if self.label:
            return f" {self.label} {bar} {pct}"
        return f" {bar} {pct}"

    def set_progress(self, progress: float) -> None:
        """Update the progress."""
        self.progress = max(0.0, min(1.0, progress))

    def set_label(self, label: str) -> None:
        """Update the label."""
        self.label = label
