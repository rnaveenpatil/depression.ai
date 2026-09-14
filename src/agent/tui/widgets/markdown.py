"""
Markdown Renderer Widget

Renders markdown text with cyberpunk-themed styling:
- Headers with neon accents
- Code blocks with syntax-aware coloring
- Lists with animated bullets
- Blockquotes with glowing borders
- Links with hover effects
"""

from __future__ import annotations

import re
from typing import Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual import on


# Inline colors
CYBER_BLUE = "#00c8ff"
NEON_CYAN = "#00ffff"
NEON_PURPLE = "#b464ff"
NEON_ORANGE = "#ffa500"
NEON_GREEN = "#00ff80"
TEXT_PRIMARY = "#f0f0ff"
TEXT_MUTED = "#8c8ca0"
TEXT_DIM = "#505064"
BORDER = "#283250"
PANEL = "#12121e"


class MarkdownViewer(Widget):
    """A markdown viewer with cyberpunk styling."""

    DEFAULT_CSS = f"""
    MarkdownViewer {{
        height: auto;
        min-height: 1;
        padding: 0 1;
    }}

    .md-h1 {{
        color: {CYBER_BLUE};
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }}

    .md-h2 {{
        color: {NEON_CYAN};
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }}

    .md-h3 {{
        color: {NEON_PURPLE};
        text-style: bold;
        height: 1;
    }}

    .md-paragraph {{
        height: auto;
        color: {TEXT_PRIMARY};
    }}

    .md-code-block {{
        background: {PANEL};
        border: solid {BORDER};
        padding: 0 1;
        color: {NEON_ORANGE};
        height: auto;
        margin: 0 1;
    }}

    .md-code-inline {{
        background: {PANEL};
        color: {NEON_ORANGE};
        padding: 0 1;
    }}

    .md-blockquote {{
        border-left: tall {CYBER_BLUE};
        color: {TEXT_MUTED};
        padding: 0 0 0 1;
        text-style: italic;
    }}

    .md-list-item {{
        height: 1;
        color: {TEXT_PRIMARY};
    }}

    .md-list-bullet {{
        color: {NEON_CYAN};
    }}

    .md-divider {{
        color: {BORDER};
        height: 1;
    }}

    .md-link {{
        color: {CYBER_BLUE};
        text-style: underline;
    }}

    .md-strong {{
        text-style: bold;
    }}

    .md-emphasis {{
        text-style: italic;
    }}
    """

    def __init__(self, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self._lines: list[Static] = []

    def compose(self) -> ComposeResult:
        if self.content:
            yield from self._render_content(self.content)
        else:
            yield Static("", classes="md-paragraph")

    def _render_content(self, text: str) -> ComposeResult:
        lines = text.split("\n")
        in_code_block = False
        code_lang = ""
        code_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    code_text = "\n".join(code_lines)
                    yield Static(
                        f" {code_lang}\n{code_text}",
                        classes="md-code-block",
                    )
                    code_lines = []
                    code_lang = ""
                    in_code_block = False
                else:
                    code_lang = stripped[3:].strip() or "code"
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            if not stripped:
                continue

            if stripped.startswith("### "):
                yield Static(stripped[4:], classes="md-h3")
            elif stripped.startswith("## "):
                yield Static(stripped[3:], classes="md-h2")
            elif stripped.startswith("# "):
                yield Static(stripped[2:], classes="md-h1")
            elif stripped in ("---", "***", "___"):
                yield Static("─" * 60, classes="md-divider")
            elif stripped.startswith("> "):
                yield Static(f" {stripped[2:]}", classes="md-blockquote")
            elif re.match(r"^[\-\*\+] ", stripped):
                bullet = stripped[0]
                content = stripped[2:]
                yield Static(
                    f" {self._colorize('cyan', '•')} {content}",
                    classes="md-list-item",
                )
            elif re.match(r"^\d+\. ", stripped):
                match = re.match(r"^(\d+)\. (.*)", stripped)
                if match:
                    yield Static(
                        f" {self._colorize('cyan', match.group(1) + '.')} {match.group(2)}",
                        classes="md-list-item",
                    )
            else:
                rendered = self._inline_format(stripped)
                yield Static(rendered, classes="md-paragraph")

    def _inline_format(self, text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", text)
        text = re.sub(r"\*(.+?)\*", r"[italic]\1[/italic]", text)
        text = re.sub(
            r"`(.+?)`",
            lambda m: f"[on {PANEL} {NEON_ORANGE}]{m.group(1)}[/]",
            text,
        )
        return text

    @staticmethod
    def _colorize(color: str, text: str) -> str:
        color_map = {
            "cyan": "#00ffff",
            "pink": "#ff0080",
            "green": "#00ff80",
            "orange": "#ffa500",
            "purple": "#b464ff",
            "yellow": "#ffff00",
            "blue": "#00c8ff",
        }
        hex_color = color_map.get(color, "#ffffff")
        return f"[{hex_color}]{text}[/]"

    @staticmethod
    def _color(name: str) -> str:
        return MarkdownViewer._colorize(name, "")

    def update_content(self, content: str) -> None:
        self.content = content
        self.remove_children()
        self.mount(*self._render_content(content))