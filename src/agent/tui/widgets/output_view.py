"""
Structured output view for agent replies.

Turns the assistant's Markdown-ish output into a set of Textual widgets
so the transcript reads like a report, not a wall of text.

Improvements over the previous version:
    * Display-width aware. Uses wcwidth when available, so CJK, emoji,
      and combining marks no longer break table alignment.
    * Task lists (- [ ] / - [x]) render with a checkbox.
    * Robust Markdown parser: tolerates `|---|` and `|---|---|`
      separators, pipes inside inline code, and zero-width chars.
    * Block coalescing. Consecutive paragraphs are grouped into one
      Static widget so a 200-block reply doesn't spawn 200 widgets.
    * Correct table layout. Columns, separators, and padding line up
      regardless of content width.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
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


# ----------------------------------------------------------------------
# Display width
# ----------------------------------------------------------------------

try:
    from wcwidth import wcswidth as _wcswidth  # type: ignore

    def _display_width(s: str) -> int:
        if not s:
            return 0
        w = _wcswidth(s)
        if w >= 0:
            return w
        return len(s)

    def _char_width(c: str) -> int:
        w = _wcswidth(c)
        return w if w >= 0 else 1
except Exception:
    def _display_width(s: str) -> int:
        """Fallback: 2 for wide/emoji, 0 for combining/zero-width, else 1."""
        total = 0
        for c in s:
            total += _char_width(c)
        return total

    def _char_width(c: str) -> int:
        o = ord(c)
        # Zero-width: combining marks, ZWJ, ZWNJ, ZWSP, format chars
        if unicodedata.combining(c):
            return 0
        if o in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD):
            return 0
        # Wide East Asian + common emoji ranges
        if (
            0x1100 <= o <= 0x115F
            or 0x2E80 <= o <= 0xA4CF
            or 0xAC00 <= o <= 0xD7A3
            or 0xF900 <= o <= 0xFAFF
            or 0xFE30 <= o <= 0xFE4F
            or 0xFF00 <= o <= 0xFF60
            or 0xFFE0 <= o <= 0xFFE6
            or 0x1F300 <= o <= 0x1FAFF
            or 0x1F000 <= o <= 0x1F02F
            or 0x1F900 <= o <= 0x1F9FF
            or 0x1F600 <= o <= 0x1F64F
        ):
            return 2
        return 1


def _pad(s: str, width: int, align: str = "left") -> str:
    """Pad a string to a target display width."""
    w = _display_width(s)
    pad = max(0, width - w)
    if align == "right":
        return " " * pad + s
    if align == "center":
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right
    return s + " " * pad


def _truncate_to_width(s: str, width: int, ellipsis: str = "…") -> str:
    """Truncate a string to a display width, appending an ellipsis."""
    if _display_width(s) <= width:
        return s
    ell_w = _display_width(ellipsis)
    budget = max(0, width - ell_w)
    out = []
    used = 0
    for c in s:
        w = _char_width(c)
        if used + w > budget:
            break
        out.append(c)
        used += w
    return "".join(out) + ellipsis


_ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(s: str) -> str:
    if not s:
        return ""
    return _CTRL_RE.sub("", _ZW_RE.sub("", s))


def _esc(text: Any) -> str:
    """Escape brackets so markup in raw text renders literally."""
    if text is None:
        return ""
    return _clean(str(text)).replace("[", r"\[")


# ----------------------------------------------------------------------
# Inline formatting
# ----------------------------------------------------------------------

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")


def _inline_format(text: str) -> str:
    """Convert a subset of inline Markdown to Textual markup."""
    s = _clean(text).replace("[", r"\[")
    s = _INLINE_CODE_RE.sub(lambda m: f"[{AMBER}]{m.group(1)}[/]", s)
    s = _BOLD_RE.sub(lambda m: f"[bold {GREEN_GLOW}]{m.group(1)}[/]", s)
    s = _ITALIC_RE.sub(lambda m: f"[italic {TEXT}]{m.group(1)}[/]", s)
    return s


# ----------------------------------------------------------------------
# Block model
# ----------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # "h" | "p" | "ul" | "ol" | "task" | "code" | "table" | "hr"
    text: str = ""
    level: int = 0
    items: Optional[List[str]] = None
    task_states: Optional[List[bool]] = None
    lang: str = ""
    rows: Optional[List[List[str]]] = None
    header: Optional[List[str]] = None


_H_RE = re.compile(r"^(#{1,6})\s+(.*)")
_TASK_RE = re.compile(r"^\s*[-*•]\s+\[([ xX])\]\s+(.*)")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*)")
_NUM_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^\s*```(\w*)")
_TABLE_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def _parse_table_row(line: str) -> List[str]:
    """
    Split a `| a | b |` line into cells, ignoring pipes inside backticks.
    """
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]

    cells: List[str] = []
    buf: List[str] = []
    in_code = False
    for c in inner:
        if c == "`":
            in_code = not in_code
            buf.append(c)
            continue
        if c == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(c)
    cells.append("".join(buf).strip())
    return cells


def _is_separator_row(line: str) -> bool:
    cells = _parse_table_row(line)
    if not cells:
        return False
    for c in cells:
        c = c.strip()
        if not c:
            return False
        if not re.fullmatch(r":?-{1,}:?", c):
            return False
    return True


def _parse_table(
    lines: List[str],
) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    if len(lines) < 2:
        return None, None
    if not _is_separator_row(lines[1]):
        return None, None
    header = _parse_table_row(lines[0])
    rows = [_parse_table_row(l) for l in lines[2:]]
    return header, rows


def _parse_blocks(text: str) -> List[Block]:
    text = _clean(text)
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

        if _TASK_RE.match(line):
            items: List[str] = []
            states: List[bool] = []
            while i < n and _TASK_RE.match(lines[i]):
                mm = _TASK_RE.match(lines[i])
                states.append(mm.group(1).lower() == "x")
                items.append(mm.group(2).rstrip())
                i += 1
            blocks.append(Block(kind="task", items=items, task_states=states))
            continue

        if _BULLET_RE.match(line):
            items = []
            while i < n and _BULLET_RE.match(lines[i]):
                items.append(_BULLET_RE.match(lines[i]).group(1).rstrip())
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
                or _TASK_RE.match(nxt)
                or _BULLET_RE.match(nxt)
                or _NUM_RE.match(nxt)
                or _FENCE_RE.match(nxt)
                or _HR_RE.match(nxt)
                or _TABLE_RE.match(nxt)
            ):
                break
            buf.append(nxt.rstrip())
            i += 1
        blocks.append(Block(kind="p", text="\n".join(buf)))

    return _coalesce_paragraphs(blocks)


def _coalesce_paragraphs(blocks: List[Block]) -> List[Block]:
    """Merge consecutive 'p' blocks so long replies mount fewer widgets."""
    out: List[Block] = []
    for b in blocks:
        if b.kind == "p" and out and out[-1].kind == "p":
            out[-1].text = out[-1].text + "\n" + b.text
        else:
            out.append(b)
    return out


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
    """
    Render a Markdown table with correct display-width alignment.

    Column widths are computed from display width of the widest cell.
    Cells longer than `max_cell_width` are truncated with an ellipsis.
    """
    if not header:
        return ""

    max_cell_width = 40

    widths = [_display_width(h) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            w = min(_display_width(cell), max_cell_width)
            if i < len(widths):
                widths[i] = max(widths[i], w)
            else:
                widths.append(w)

    # Clamp each width to max_cell_width.
    widths = [min(w, max_cell_width) for w in widths]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    lines: List[str] = [f"[{BORDER}]{top}[/]"]

    def _row(cells: List[str], style: str) -> str:
        rendered: List[str] = []
        for i, w in enumerate(widths):
            raw = cells[i] if i < len(cells) else ""
            raw = _truncate_to_width(raw, w)
            padded = _pad(raw, w)
            rendered.append(f"[{style}]{_esc(padded)}[/]")
        return (
            f"[{BORDER}]│[/] "
            + f" [{BORDER}]│[/] ".join(rendered)
            + f" [{BORDER}]│[/]"
        )

    lines.append(_row(header, f"bold {GREEN_GLOW}"))
    lines.append(f"[{BORDER}]{mid}[/]")
    for r in rows:
        lines.append(_row(r, TEXT))
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
        out = []
        for item, done in zip(items, states):
            if done:
                out.append(
                    f" [{GREEN}]✓[/] [{MUTED}]{_inline_format(item)}[/]"
                )
            else:
                out.append(f" [{DIM}]○[/] {_inline_format(item)}")
        return "\n".join(out)
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
    """A single structured agent reply."""

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