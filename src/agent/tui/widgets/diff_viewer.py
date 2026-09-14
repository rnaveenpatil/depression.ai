"""
Diff Viewer Widget

Renders unified diffs with cyberpunk neon styling:
- Added lines in matrix green
- Removed lines in neon red
- Headers in cyber blue
- Context in muted gray
"""

from __future__ import annotations

import difflib
from typing import Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class DiffViewer(Widget):
    """A diff viewer with neon styling."""

    DEFAULT_CSS = """
    DiffViewer {
        height: auto;
        min-height: 1;
        padding: 0;
    }

    .diff-file-header {
        background: $bg-secondary;
        color: $accent-primary;
        text-style: bold;
        height: 1;
        padding: 0 1;
        margin: 0 0 0 0;
    }

    .diff-hunk-header {
        color: $neon-cyan;
        height: 1;
        padding: 0 1;
    }

    .diff-added {
        background: rgb(0, 40, 20);
        color: $neon-green;
        height: 1;
        padding: 0 1;
    }

    .diff-removed {
        background: rgb(40, 0, 10);
        color: $neon-red;
        height: 1;
        padding: 0 1;
    }

    .diff-context {
        color: $text-muted;
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, old: str = "", new: str = "", filename: str = "", **kwargs):
        super().__init__(**kwargs)
        self.old = old
        self.new = new
        self.filename = filename

    def compose(self) -> ComposeResult:
        if self.filename:
            yield Static(f" {self.filename}", classes="diff-file-header")

        if not self.old and not self.new:
            yield Static(" No changes", classes="diff-context")
            return

        diff = list(difflib.unified_diff(
            self.old.splitlines(keepends=True),
            self.new.splitlines(keepends=True),
            fromfile="a/" + (self.filename or "original"),
            tofile="b/" + (self.filename or "modified"),
            n=3,
        ))

        if not diff:
            yield Static(" No differences", classes="diff-context")
            return

        for line in diff:
            line = line.rstrip("\n")
            if line.startswith("+++") or line.startswith("---"):
                yield Static(f" {line}", classes="diff-file-header")
            elif line.startswith("@@"):
                yield Static(f" {line}", classes="diff-hunk-header")
            elif line.startswith("+"):
                yield Static(f" {line}", classes="diff-added")
            elif line.startswith("-"):
                yield Static(f" {line}", classes="diff-removed")
            else:
                yield Static(f" {line}", classes="diff-context")

    def update_diff(self, old: str, new: str, filename: str = "") -> None:
        """Update the diff content."""
        self.old = old
        self.new = new
        if filename:
            self.filename = filename
        self.remove_children()
        self.mount(*self.compose())
