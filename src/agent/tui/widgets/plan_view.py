"""
Inline plan renderer for the transcript.

Renders a plan as a single bordered block that looks like this:

    ▌ plan                                    3/7 done
    ────────────────────────────────────────────────
    ●  inspect the repo layout                    (done)
    ●  read config and env                        (done)
    ●  identify the auth module                   (done)
    ▶  patch onboarding/welcome.py                (running)
    ○  wire the /profile panel
    ○  run the TUI smoke test
    ○  write docs

The block is a single Static widget, so it can be re-rendered in place
when the plan updates instead of appending a new block every time.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


_STATUS_GLYPH = {
    "pending":     ("○", MUTED),
    "in_progress": ("▶", AMBER),
    "running":     ("▶", AMBER),
    "completed":   ("●", GREEN_DIM),
    "done":        ("●", GREEN_DIM),
    "blocked":     ("◼", ERROR),
    "cancelled":   ("✕", DIM),
}


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


def _normalize_status(raw: Any) -> str:
    s = str(raw or "pending").strip().lower()
    if s in ("done", "complete", "completed", "finished"):
        return "completed"
    if s in ("in_progress", "in-progress", "running", "active"):
        return "in_progress"
    if s in ("blocked", "stuck", "waiting"):
        return "blocked"
    if s in ("cancelled", "canceled", "skipped"):
        return "cancelled"
    return "pending"


class PlanView(Vertical):
    """
    A single inline plan block. Call `set_entries([...])` to re-render.

    Each entry is a dict with `content` (str), `status` (str), and
    optional `priority`.
    """

    DEFAULT_CSS = f"""
    PlanView {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {GREEN_DIM};
    }}
    PlanView > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(self, entries: List[Dict[str, Any]] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._entries: List[Dict[str, Any]] = list(entries or [])
        self._body: Static | None = None

    def compose(self) -> ComposeResult:
        self._body = Static(self._build_markup(), markup=True)
        yield self._body

    # ------------------------------------------------------------------

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = list(entries or [])
        if self._body is not None:
            try:
                self._body.update(self._build_markup())
            except Exception:
                pass

    def merge_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
        Merge by content: existing entries keep their position; new
        entries are appended; changed statuses are updated in place.
        """
        by_content = {e.get("content", ""): e for e in self._entries}
        merged: List[Dict[str, Any]] = []
        for e in entries:
            key = e.get("content", "")
            if key in by_content:
                old = by_content[key]
                merged.append({
                    "content": key,
                    "status": e.get("status", old.get("status", "pending")),
                    "priority": e.get("priority", old.get("priority")),
                })
                by_content.pop(key, None)
            else:
                merged.append(dict(e))
        for leftover in self._entries:
            key = leftover.get("content", "")
            if key in by_content:
                merged.append(leftover)
        self.set_entries(merged)

    # ------------------------------------------------------------------

    def _build_markup(self) -> str:
        if not self._entries:
            return f"[{DIM}]no plan yet[/]"

        total = len(self._entries)
        done = sum(
            1 for e in self._entries
            if _normalize_status(e.get("status")) == "completed"
        )
        running = sum(
            1 for e in self._entries
            if _normalize_status(e.get("status")) == "in_progress"
        )

        # ── header ───────────────────────────────────────────────
        counter = f"[{MUTED}]{done}/{total} done[/]"
        if running:
            counter += f"  [{AMBER}]{running} running[/]"
        header = f"[bold {GREEN}]▌ plan[/]   {counter}"

        # ── body ─────────────────────────────────────────────────
        lines = [header]

        # Column widths so glyphs and titles line up.
        for e in self._entries:
            status = _normalize_status(e.get("status"))
            glyph, color = _STATUS_GLYPH.get(status, ("○", MUTED))
            content = _esc(str(e.get("content") or "").strip())

            if status == "completed":
                lines.append(
                    f"  [{color}]{glyph}[/]  [{DIM}][strike]{content}[/strike][/]"
                )
            elif status == "in_progress":
                lines.append(
                    f"  [{color}]{glyph}[/]  [{GREEN_GLOW}]{content}[/]"
                )
            elif status == "blocked":
                lines.append(
                    f"  [{color}]{glyph}[/]  [{ERROR}]{content}[/]"
                )
            elif status == "cancelled":
                lines.append(
                    f"  [{color}]{glyph}[/]  [{DIM}][strike]{content}[/strike][/]"
                )
            else:
                lines.append(f"  [{color}]{glyph}[/]  [{TEXT}]{content}[/]")

        return "\n".join(lines)


__all__ = ["PlanView"]