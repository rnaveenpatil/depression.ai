"""
Diff view widget - renders a unified-style line diff with red/green
highlighting, matching the pattern used by modern coding agents.

Computes line-level changes from (before, after) text and renders:
    @@ -a,b +c,d @@      amber header
    - removed line       red
    + added line         green
      unchanged line     muted
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from rich.markup import escape
from typing import Any, List, Optional, Sequence, Tuple

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


RED = "#ff4466"
RED_DIM = "#7a1f30"
GREEN = "#00ff66"
GREEN_DIM = "#0a5c2a"
AMBER = "#ffcc44"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
TEXT = "#aaffcc"
BORDER = "#0a3d20"


@dataclass
class DiffLine:
    kind: str            # "add" | "del" | "ctx" | "hunk"
    text: str
    old_no: Optional[int] = None
    new_no: Optional[int] = None


# Even valid 50 KB tool snapshots can contain thousands of tiny lines.
# Bound worst-case diff processing, and show a clear omission notice instead.
_MAX_PREVIEW_BYTES = 80 * 1024
_MAX_PREVIEW_LINES = 3500


def compute_line_diff(before: str, after: str, context: int = 3) -> List[DiffLine]:
    """Return a line diff using SequenceMatcher instead of quadratic DP memory."""
    a = before.splitlines()
    b = after.splitlines()
    raw: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=a, b=b, autojunk=True,
    ).get_opcodes():
        if tag == "equal":
            raw.extend(("ctx", value) for value in a[i1:i2])
        else:
            if tag in ("replace", "delete"):
                raw.extend(("del", value) for value in a[i1:i2])
            if tag in ("replace", "insert"):
                raw.extend(("add", value) for value in b[j1:j2])

    # Assign line numbers and trim unchanged runs to `context` lines.
    out: List[DiffLine] = []
    old_no = new_no = 1
    idx = 0
    while idx < len(raw):
        kind, text = raw[idx]

        if kind == "ctx":
            # Find the length of this context run.
            end = idx
            while end < len(raw) and raw[end][0] == "ctx":
                end += 1
            run = raw[idx:end]
            if len(run) <= context * 2:
                for _, t in run:
                    out.append(DiffLine("ctx", t, old_no, new_no))
                    old_no += 1
                    new_no += 1
            else:
                for _, t in run[:context]:
                    out.append(DiffLine("ctx", t, old_no, new_no))
                    old_no += 1
                    new_no += 1
                skipped = len(run) - context * 2
                out.append(
                    DiffLine("hunk", f"… {skipped} unchanged line"
                              f"{'s' if skipped != 1 else ''} …")
                )
                old_no += skipped
                new_no += skipped
                for _, t in run[-context:]:
                    out.append(DiffLine("ctx", t, old_no, new_no))
                    old_no += 1
                    new_no += 1
            idx = end
            continue

        if kind == "del":
            out.append(DiffLine("del", text, old_no, None))
            old_no += 1
        else:
            out.append(DiffLine("add", text, None, new_no))
            new_no += 1
        idx += 1

    return out


def render_diff(
    before: str,
    after: str,
    path: str = "",
    max_lines: int = 400,
) -> str:
    """Return Textual markup for a diff, or empty string if unchanged."""
    if before == after:
        return ""

    # Do not freeze the Textual event loop for pathological inputs.
    if (len(before.encode("utf-8")) > _MAX_PREVIEW_BYTES
            or len(after.encode("utf-8")) > _MAX_PREVIEW_BYTES
            or len(before.splitlines()) > _MAX_PREVIEW_LINES
            or len(after.splitlines()) > _MAX_PREVIEW_LINES):
        label = f" {escape(path)}:" if path else ""
        return f"[{AMBER}]Diff preview omitted{label} file is too large; inspect with git diff.[/]"

    lines = compute_line_diff(before, after)
    if not lines:
        return ""

    header_parts = []
    if path:
        header_parts.append(f"[bold {AMBER}]{escape(path)}[/]")
    added = sum(1 for dl in lines if dl.kind == "add")
    removed = sum(1 for dl in lines if dl.kind == "del")
    header_parts.append(f"[{GREEN}]+{added}[/] [{RED}]-{removed}[/]")
    out = ["  " + "  ".join(header_parts)]

    max_lines = max(1, int(max_lines))
    for line in lines[:max_lines]:
        if line.kind == "hunk":
            out.append(f"[{AMBER}]  {line.text}[/]")
            continue
        old = f"{line.old_no:>4}" if line.old_no is not None else "    "
        new = f"{line.new_no:>4}" if line.new_no is not None else "    "
        body = escape(line.text)
        if line.kind == "del":
            out.append(f"[{RED_DIM}]{old}[/] [{RED_DIM}]{new}[/] [{RED}]- {body}[/]")
        elif line.kind == "add":
            out.append(f"[{GREEN_DIM}]{old}[/] [{GREEN_DIM}]{new}[/] [{GREEN}]+ {body}[/]")
        else:
            out.append(f"[{DIM}]{old}[/] [{DIM}]{new}[/] [{MUTED}]  {body}[/]")

    if len(lines) > max_lines:
        out.append(f"[{AMBER}]  … {len(lines) - max_lines} more lines[/]")

    return "\n".join(out)


class DiffView(Vertical):
    """A collapsible diff block for a single file change."""

    DEFAULT_CSS = f"""
    DiffView {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {BORDER};
    }}
    DiffView > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(self, path: str, before: str, after: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._path = path
        self._before = before
        self._after = after

    def compose(self) -> ComposeResult:
        markup = render_diff(self._before, self._after, path=self._path)
        if not markup:
            yield Static(f"[{MUTED}]no change[/]", markup=False)
        else:
            yield Static(markup, markup=True)