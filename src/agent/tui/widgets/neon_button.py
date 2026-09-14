"""
Neon Button Widget

A button with cyberpunk neon glow effects:
- Glowing border on hover
- Pulsing animation on focus
- Color-coded variants (primary, success, danger)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Button
from textual.reactive import reactive


class NeonButton(Button):
    """A neon-styled button."""

    DEFAULT_CSS = """
    NeonButton {
        background: $bg-panel;
        border: solid $border-default;
        color: $text-primary;
        padding: 0 2;
        height: 3;
        min-width: 10;
        content-align: center middle;
        text-style: bold;
    }

    NeonButton:hover {
        background: $bg-hover;
        border: solid $accent-primary;
        color: $accent-primary;
    }

    NeonButton:focus {
        border: solid $neon-cyan;
        color: $neon-cyan;
    }

    NeonButton:active {
        background: $bg-selected;
    }

    NeonButton.neon-primary {
        border: solid $accent-primary;
        color: $accent-primary;
    }

    NeonButton.neon-success {
        border: solid $neon-green;
        color: $neon-green;
    }

    NeonButton.neon-danger {
        border: solid $neon-red;
        color: $neon-red;
    }

    NeonButton.neon-warning {
        border: solid $neon-yellow;
        color: $neon-yellow;
    }
    """

    variant = reactive[str]("default")

    def __init__(self, label: str = "", variant: str = "default", **kwargs):
        super().__init__(label=label, **kwargs)
        self.variant = variant
        if variant != "default":
            self.add_class(f"neon-{variant}")
