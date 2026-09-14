"""
Animated Spinner Widget

Displays animated spinners with multiple styles:
- Braille (default)
- Dots
- Arc
- Pulse
"""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from textual import on, work


class SpinnerWidget(Widget):
    """An animated spinner widget."""

    DEFAULT_CSS = """
    SpinnerWidget {
        height: 1;
        width: auto;
        color: $accent-primary;
    }
    """

    SPINNER_STYLES = {
        "braille": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "arc": ["◜", "◠", "◝", "◞", "◡", "◟"],
        "pulse": ["○", "◎", "●", "◎", "○"],
        "wave": [" ", "·", "•", "○", "◎", "●", "◎", "○", "•", "·"],
    }

    spinning = reactive[bool](False)
    message = reactive[str]("")

    def __init__(
        self,
        message: str = "",
        style: str = "braille",
        speed: float = 0.08,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.message = message
        self.spinner_style = style
        self.speed = speed
        self._frame = 0
        self._frames = self.SPINNER_STYLES.get(style, self.SPINNER_STYLES["braille"])

    def compose(self) -> ComposeResult:
        yield Static(self._get_frame(), id="spinner-text")

    def _get_frame(self) -> str:
        char = self._frames[self._frame % len(self._frames)]
        if self.message:
            return f" {char} {self.message}"
        return f" {char}"

    @work(exclusive=True, group="spinner")
    async def _animate(self) -> None:
        """Animation loop."""
        while self.spinning:
            self._frame += 1
            try:
                text_widget = self.query_one("#spinner-text", Static)
                text_widget.update(self._get_frame())
            except Exception:
                break
            await asyncio.sleep(self.speed)

    def start(self, message: str = "") -> None:
        """Start the spinner."""
        if message:
            self.message = message
        self.spinning = True
        self._animate()

    def stop(self) -> None:
        """Stop the spinner."""
        self.spinning = False
        self._frame = 0

    def update_message(self, message: str) -> None:
        """Update the spinner message."""
        self.message = message
        try:
            text_widget = self.query_one("#spinner-text", Static)
            text_widget.update(self._get_frame())
        except Exception:
            pass
