"""
Structured output view for agent replies.

Turns the assistant's Markdown-ish output into a set of Textual widgets
so the transcript reads like a report, not a wall of text:

    ◆ depression.ai                         3.2s
    ────────────────────────────────────────────────
    ## Summary
    Fixed the auth redirect in welcome.py.

    ## Changes
      • onboarding/welcome.py       +42 −18
      • onboarding/identity.py       +6 −2

    ## Plan
      ✓ parse config
      ✓ render template
      ▸ run tests
      ○ report results

    ## Table
      ┌───────────────┬──────────┬────────┐
      │ step          │ status   │ time   │
      ├───────────────┼──────────┼────────┤
      │ parse         │ done     │ 12ms   │
      │ render        │ done     │ 4ms    │
      └───────────────┴──────────┴────────┘

Recognized:
    # ## / ### section headers
    * / - / • bullets
    - [ ] / - [x] task list items
    1. numbered items
    | a | b | tables (Markdown)
    ```lang ... ``` code blocks
    --- horizontal rule
    **bold** `code` *italic* inline formatting
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
GREEN_GLOW = "#88ffbb"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


# ----------------------------------------------------------------------
# Markdown-ish parsing
# ----------------------------------------------------------------------

_H_RE = re.compile(r"^(#{1,6})\s+(.*)")
_BULLET_RE = re.compile(r"^(\s*)[-*•]\s+(.*)")
_TASK_RE = re.compile(r"^(\s*)[-*•]\s+\[([ xX])\]\s+(.*)")
_NUM_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^\s*```(\w*)")
_TABLE_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def _inline_format(text: str) -> str:
    s = text.replace("[", r"\[")

    s = re.sub(
        r"`([^`]+)`",
        lambda m: f"[{AMBER}]{m.group(1)}[/]",
        s,
    )
    s = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: f"[bold {GREEN_GLOW}]{m.group(1)}[/]",
        s,
    )
    s = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        lambda m: f"[italic {TEXT}]{m.group(1)}[/]",
        s,
    )
    return s


# ----------------------------------------------------------------------
# Block model
# ----------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # "h" | "p" | "ul" | "task" | "ol" | "code" | "table" | "hr"
    text: str = ""
    level: int = 0
    items: Optional[List[str]] = None
    task_states: Optional[List[bool]] = None
    lang: str = ""
    rows: Optional[List[List[str]]] = None
    header: Optional[List[str]] = None


def _parse_blocks(text: str) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        m = _FENCE_RE.match(line)
        if m:
            lang = m.group(1) or ""
            i += 1
            buf: List[str] = []
            while i < n and not _FENCE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            if i < n:
                i += 1
            blocks.append(Block(kind="code", text="\n".join(buf), lang=lang))
            continue

        if _HR_RE.match(line):
            blocks.append(Block(kind="hr"))
            i += 1
            continue

        if _TABLE_RE.match(line):
            table_lines: List[str] = []
            while i < n and _TABLE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            header, rows = _parse_table(table_lines)
            if header is not None:
                blocks.append(Block(kind="table", header=header, rows=rows))
                continue
            for tl in table_lines:
                blocks.append(Block(kind="p", text=tl))
            continue

        m = _H_RE.match(line)
        if m:
            blocks.append(
                Block(kind="h", level=len(m.group(1)), text=m.group(2).strip())
            )
            i += 1
            continue

        # Task list (- [ ] / - [x]) must be checked before plain bullets.
        if _TASK_RE.match(line):
            items: List[str] = []
            states: List[bool] = []
            while i < n and _TASK_RE.match(lines[i]):
                mm = _TASK_RE.match(lines[i])
                states.append(mm.group(2).lower() == "x")
                items.append(mm.group(3).rstrip())
                i += 1
            blocks.append(Block(kind="task", items=items, task_states=states))
            continue

        if _BULLET_RE.match(line):
            items = []
            while i < n and _BULLET_RE.match(lines[i]):
                items.append(_BULLET_RE.match(lines[i]).group(2).rstrip())
                i += 1
            blocks.append(Block(kind="ul", items=items))
            continue

        if _NUM_RE.match(line):
            items = []
            while i < n and _NUM_RE.match(lines[i]):
                items.append(_NUM_RE.match(lines[i]).group(2).rstrip())
                i += 1
            blocks.append(Block(kind="ol", items=items))
            continue

        buf = [line.rstrip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                _H_RE.match(nxt)
                or _BULLET_RE.match(nxt)
                or _NUM_RE.match(nxt)
                or _FENCE_RE.match(nxt)
                or _HR_RE.match(nxt)
                or _TABLE_RE.match(nxt)
                or _TASK_RE.match(nxt)
            ):
                break
            buf.append(nxt.rstrip())
            i += 1
        blocks.append(Block(kind="p", text="\n".join(buf)))

    return blocks


def _split_row(line: str) -> List[str]:
    inner = line.strip().strip("|")
    return [c.strip() for c in inner.split("|")]


def _parse_table(
    lines: List[str],
) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    if len(lines) < 2:
        return None, None
    header = _split_row(lines[0])
    sep = _split_row(lines[1])
    if not all(re.fullmatch(r":?-{2,}:?", c or "") for c in sep):
        return None, None
    rows = [_split_row(l) for l in lines[2:]]
    return header, rows


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _render_header(text: str, level: int) -> str:
    if level <= 1:
        return f"[bold {GREEN_GLOW}]{_inline_format(text)}[/]"
    if level == 2:
        return f"[bold {GREEN}]{_inline_format(text)}[/]"
    if level == 3:
        return f"[bold {TEXT}]{_inline_format(text)}[/]"
    return f"[{MUTED}]{_inline_format(text)}[/]"


def _render_table(header: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    lines = [f"[{BORDER}]{top}[/]"]

    header_cells = []
    for i, c in enumerate(header):
        padded = _esc(c.ljust(widths[i]))
        header_cells.append(f"[bold {GREEN_GLOW}]{padded}[/]")
    lines.append(
        f"[{BORDER}]│[/] "
        + f" [{BORDER}]│[/] ".join(header_cells)
        + f" [{BORDER}]│[/]"
    )

    lines.append(f"[{BORDER}]{mid}[/]")

    for r in rows:
        cells = []
        for i, w in enumerate(widths):
            cell = r[i] if i < len(r) else ""
            cells.append(f"[{TEXT}]{_esc(cell.ljust(w))}[/]")
        lines.append(
            f"[{BORDER}]│[/] "
            + f" [{BORDER}]│[/] ".join(cells)
            + f" [{BORDER}]│[/]"
        )

    lines.append(f"[{BORDER}]{bot}[/]")
    return "\n".join(lines)


def _render_block(b: Block) -> str:
    if b.kind == "h":
        return _render_header(b.text, b.level)
    if b.kind == "hr":
        return f"[{BORDER}]" + "─" * 56 + "[/]"
    if b.kind == "code":
        body_lines = b.text.splitlines() or [""]
        head = f"[{DIM}]{_esc(b.lang)}[/]" if b.lang else f"[{DIM}][/]"
        out = [head]
        for ln in body_lines:
            out.append(f"[{TEXT}] {_esc(ln)}[/]")
        out.append(f"[{DIM}]```[/]")
        return "\n".join(out)
    if b.kind == "ul":
        return "\n".join(
            f" [{GREEN}]•[/] {_inline_format(item)}" for item in (b.items or [])
        )
    if b.kind == "task":
        items = b.items or []
        states = b.task_states or [False] * len(items)
        lines = []
        for item, done in zip(items, states):
            if done:
                lines.append(
                    f" [{GREEN}]✓[/] [{MUTED}]{_inline_format(item)}[/]"
                )
            else:
                lines.append(f" [{DIM}]○[/] {_inline_format(item)}")
        return "\n".join(lines)
    if b.kind == "ol":
        return "\n".join(
            f" [{GREEN}]{i + 1}.[/] {_inline_format(item)}"
            for i, item in enumerate(b.items or [])
        )
    if b.kind == "table":
        return _render_table(b.header or [], b.rows or [])
    return _inline_format(b.text)


# ----------------------------------------------------------------------
# Widget
# ----------------------------------------------------------------------


class OutputView(Vertical):
    """
    A single structured agent reply.

    ``header`` is a short line shown above the body (e.g. "◆ depression.ai"),
    ``meta`` is right-aligned (elapsed time, tokens, etc.).
    """

    DEFAULT_CSS = f"""
    OutputView {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {GREEN_DIM};
    }}
    OutputView > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(
        self,
        text: str = "",
        header: str = "◆ depression.ai",
        meta: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._text = text or ""
        self._header = header
        self._meta = meta

    def compose(self) -> ComposeResult:
        head_line = f"[bold {GREEN}]{_esc(self._header)}[/]"
        if self._meta:
            head_line += f" [{DIM}]{_esc(self._meta)}[/]"
        yield Static(head_line, markup=True)

        yield Static(f"[{BORDER}]" + "─" * 56 + "[/]", markup=True)

        for b in _parse_blocks(self._text):
            markup = _render_block(b)
            if markup:
                yield Static(markup, markup=True)


__all__ = ["OutputView", "Block", "_parse_blocks", "_render_block"]