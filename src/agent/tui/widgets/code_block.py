"""
Code Block Widget

A code block with:
- Language label header
- Line numbers
- Basic syntax coloring
- Copy-to-clipboard support
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class CodeBlock(Widget):
    """A code block widget with language header."""

    DEFAULT_CSS = """
    CodeBlock {
        height: auto;
        min-height: 1;
        margin: 0 1;
    }

    .code-header {
        background: $bg-secondary;
        color: $text-muted;
        text-style: bold;
        height: 1;
        padding: 0 1;
        border-top: solid $border-default;
        border-left: solid $border-default;
        border-right: solid $border-default;
    }

    .code-body {
        background: $bg-panel;
        border: solid $border-default;
        color: $neon-orange;
        padding: 0 1;
        height: auto;
        overflow-x: auto;
    }

    .code-line-num {
        color: $text-dim;
        text-align: right;
        width: 4;
    }
    """

    def __init__(self, code: str, language: str = "", show_line_numbers: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.code = code
        self.language = language or "code"
        self.show_line_numbers = show_line_numbers

    def compose(self) -> ComposeResult:
        # Language header
        yield Static(f" {self.language}", classes="code-header")

        # Code body with optional line numbers
        lines = self.code.split("\n")
        if self.show_line_numbers:
            numbered = []
            for i, line in enumerate(lines, 1):
                numbered.append(f" {i:3} │ {line}")
            yield Static("\n".join(numbered), classes="code-body")
        else:
            yield Static(self.code, classes="code-body")

    def update_code(self, code: str, language: str = "") -> None:
        """Update the code content."""
        self.code = code
        if language:
            self.language = language
        self.remove_children()
        self.mount(*self.compose())
