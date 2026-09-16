"""
Thinking indicator - animated status while the agent is reasoning.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

AMBER = "#ffcc44"
GREEN = "#00ff66"
MUTED = "#3d8c5c"

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_PHASES = [
    "thinking",
    "reasoning",
    "planning",
    "considering",
    "analyzing",
    "reflecting",
]


class ThinkingIndicator(Static):
    DEFAULT_CSS = f"""
    ThinkingIndicator {{
        height: 1;
        width: 100%;
        padding: 0 1;
        color: {AMBER};
    }}
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._i = 0
        self._active = False

    def on_mount(self) -> None:
        self.display = False
        self.set_interval(0.10, self._tick)

    def start(self) -> None:
        self._active = True
        self.display = True
        self._i = 0
        self._tick()

    def stop(self) -> None:
        self._active = False
        self.display = False

    def _tick(self) -> None:
        if not self._active:
            return
        self._i += 1
        spin = _SPINNER[self._i % len(_SPINNER)]
        phase = _PHASES[(self._i // 20) % len(_PHASES)]
        dots = "." * ((self._i // 5) % 4)
        self.update(f"[{AMBER}]{spin}[/] [{MUTED}]{phase}{dots}[/]")