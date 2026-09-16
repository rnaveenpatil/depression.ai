"""
Todo panel - live task checklist from the session's TodoTool store.

States:
    pending      ○   (muted)
    in_progress  ▶   (amber)
    done         ✓   (green)
    blocked      ✗   (error)
    cancelled    –   (dim)
"""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from agent.tools.todo import TodoTool

GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"

_STATUS_GLYPH = {
    "pending": "○",
    "in_progress": "▶",
    "done": "✓",
    "blocked": "✗",
    "cancelled": "–",
}

_STATUS_COLOR = {
    "pending": MUTED,
    "in_progress": AMBER,
    "done": GREEN_DIM,
    "blocked": ERROR,
    "cancelled": DIM,
}


class TodoPanel(Static):
    """Renders the current session's todo items, refreshed live."""

    DEFAULT_CSS = f"""
    TodoPanel {{
        width: 100%;
        height: auto;
        padding: 0 1;
        color: {TEXT};
    }}
    """

    def __init__(self, session: Any = None, **kwargs: Any):
        super().__init__("", **kwargs)
        self._session = session
        self._tool = TodoTool(session)

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh_todos)
        self.refresh_todos()

    def refresh_todos(self) -> None:
        self.update(self._render())

    def _render(self) -> str:
        items = self._tool._list().get("items", [])
        if not items:
            return (
                f"[{MUTED}]no tasks yet — the agent adds todos as it plans.[/]"
            )
        lines = []
        for it in items:
            status = str(it.get("status", "pending") or "pending")
            glyph = _STATUS_GLYPH.get(status, "○")
            color = _STATUS_COLOR.get(status, MUTED)
            title = str(it.get("title", "")).replace("[", r"\[")
            lines.append(f"  [{color}]{glyph}[/] [{GREEN}]{title}[/]")
        return "\n".join(lines)