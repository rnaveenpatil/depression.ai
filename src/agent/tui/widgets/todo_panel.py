"""
Todo panel — status-driven list with distinct colours per state.

Rendering:
    ▶  in-progress     amber   (#ffcc44)
    ○  pending         muted   (#3d8c5c)
    ●  done            green   (#00ff66), struck through
    ◼  blocked         red     (#ff4466)
    ✕  cancelled       dim     (#1a5c33)

Header shows `X/Y done`. Polls the shared TodoTool store every 0.5s.

The colour of the DONE glyph is bright green (not dim) so it stands out
against pending items. The task title for done items is dimmed so the
list reads as "these are behind us".
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


_GLYPH = {
    "pending":     ("○", MUTED),
    "in_progress": ("▶", AMBER),
    "done":        ("●", GREEN),
    "completed":   ("●", GREEN),
    "blocked":     ("◼", ERROR),
    "cancelled":   ("✕", DIM),
}


class TodoPanel(Vertical):
    DEFAULT_CSS = f"""
    TodoPanel {{
        height: auto;
        width: 100%;
        padding: 0 1;
    }}
    TodoPanel > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(self, session: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._session = session
        self._store: Dict[str, Any] = {}
        self._header: Static | None = None
        self._body: Static | None = None
        self._last_render = ""

    def compose(self) -> ComposeResult:
        self._header = Static(
            f"[bold {GREEN}]▌ TODO[/]", markup=True
        )
        self._body = Static("", markup=True)
        yield self._header
        yield self._body

    def on_mount(self) -> None:
        self._resolve_store()
        self.set_interval(0.5, self._refresh)

    def _resolve_store(self) -> None:
        try:
            from agent.tools.todo import TodoTool
            sid = TodoTool._session_key(self._session)
            self._store = TodoTool._strong_keys.get(sid, {})
        except Exception:
            self._store = {}

    def _refresh(self) -> None:
        if self._body is None:
            return
        self._resolve_store()

        items: List[Any] = sorted(
            self._store.values(),
            key=lambda i: (i.status != "in_progress", i.priority, i.created_at),
        )

        if not items:
            markup = f"[{DIM}]no tasks yet[/]"
        else:
            done = sum(1 for i in items if i.status in ("done", "completed"))
            total = len(items)
            lines = [f"[{MUTED}]{done}/{total} done[/]", ""]
            for it in items:
                glyph, color = _GLYPH.get(it.status, ("○", MUTED))
                title = str(it.title).replace("[", r"\[")
                if len(title) > 32:
                    title = title[:29] + "…"
                if it.status in ("done", "completed"):
                    lines.append(
                        f"[{color}]{glyph}[/] [{DIM}][strike]{title}[/strike][/]"
                    )
                elif it.status == "in_progress":
                    lines.append(f"[{color}]{glyph}[/] [{GREEN_GLOW}]{title}[/]")
                else:
                    lines.append(f"[{color}]{glyph}[/] [{TEXT}]{title}[/]")
            markup = "\n".join(lines)

        if markup != self._last_render:
            self._last_render = markup
            self._body.update(markup)