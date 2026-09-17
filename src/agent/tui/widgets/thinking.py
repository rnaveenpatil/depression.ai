"""
Thinking indicator — a shrinking dot trail that pulses while the agent
is reasoning. Deliberately not a spinner; the visual rhythm is
dim → bright → dim across five characters, so the eye lands on the
centre dot. One shared clock drives every instance.
"""

from __future__ import annotations

from textual.widgets import Static

AMBER = "#ffcc44"
GREEN = "#00ff66"
MUTED = "#3d8c5c"
DIM = "#1a5c33"

# Frame sequence for the trail. Each string is 9 chars wide.
_TRAIL = [
    "·   ·   ·",
    "·   •   ·",
    "·  • •  ·",
    "•  ●  •",
    "●  •  ●",
    "• ● ● •",
    "· • • ·",
    "·  •  ·",
    "·   ·   ·",
]

_PHASES = (
    "thinking",
    "reasoning",
    "planning",
    "considering",
    "analysing",
    "reflecting",
    "weighing",
)


class ThinkingIndicator(Static):
    DEFAULT_CSS = f"""
    ThinkingIndicator {{
        height: 1;
        width: 100%;
        padding: 0 2;
        background: transparent;
        color: {AMBER};
    }}
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._i = 0
        self._active = False

    def on_mount(self) -> None:
        self.display = False
        self.set_interval(0.14, self._tick)

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
        trail = _TRAIL[self._i % len(_TRAIL)]
        phase = _PHASES[(self._i // 12) % len(_PHASES)]
        # Slow trailing ellipsis so it reads as "still going", not "stuck".
        dots = "." * ((self._i // 6) % 4)
        self.update(f"[{MUTED}]{trail}[/]  [{AMBER}]{phase}{dots}[/]")