"""
Todo panel — status-driven list with distinct colours per state.

Rendering:
    ▶  in-progress     amber
    ○  pending         muted
    ●  done            green, struck through
    ◼  blocked         red
    ✕  cancelled       dim

Header shows `X/Y done`. Two refresh paths, so the panel is never blank
when there's data to show:

    1. Polls the shared TodoTool store every 0.5s.
    2. `set_items([...])` — the app calls this directly from
       `_on_plan_updated` so a fresh plan is rendered immediately without
       waiting for the next tick.

The store resolution:
    * Try the live session's bucket (via app._session()).
    * If that's empty, walk every bucket and take the newest non-empty
      one. This is what fixes the "silently blank" panel when the session
      key used by the writer and reader happened to differ.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


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

    def __init__(self, app_ref: Any = None, session: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref
        self._session = session
        self._store: Dict[str, Any] = {}
        # Items pushed directly by the app take priority over the store
        # until the store catches up.
        self._pushed_items: List[Any] = []
        self._header: Optional[Static] = None
        self._body: Optional[Static] = None
        self._last_render = ""

    def compose(self) -> ComposeResult:
        self._header = Static(f"[bold {GREEN}]▌ TODO[/]", markup=True)
        self._body = Static("", markup=True)
        yield self._header
        yield self._body

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.5, self._refresh)

    # ------------------------------------------------------------------
    # Public API — called by the app
    # ------------------------------------------------------------------

    def set_items(self, items: List[Any]) -> None:
        """
        Replace the panel's items with a list pushed by the app.

        Items are dicts with keys: title, status, priority (optional).
        The panel keeps them until the store poll finds a newer set.
        """
        self._pushed_items = list(items or [])
        self._refresh()

    def clear(self) -> None:
        self._pushed_items = []
        self._refresh()

    # ------------------------------------------------------------------
    # Session / store resolution
    # ------------------------------------------------------------------

    def _current_session(self) -> Any:
        if self._app is not None:
            try:
                live = self._app._session()  # type: ignore[attr-defined]
                if live is not None:
                    return live
            except Exception:
                pass
        return self._session

    def _resolve_store(self) -> Dict[str, Any]:
        """Return the best available bucket from the TodoTool store."""
        try:
            from agent.tools import todo as todo_mod  # local import
            TodoTool = getattr(todo_mod, "TodoTool")
        except Exception:
            self._store = {}
            return {}

        session = self._current_session()
        bucket: Dict[str, Any] = {}
        try:
            sid = TodoTool._session_key(session)
            bucket = TodoTool._strong_keys.get(sid) or {}
        except Exception:
            bucket = {}

        if not bucket:
            # Fall back to the newest non-empty bucket.
            candidates = []
            try:
                for k, v in (TodoTool._strong_keys or {}).items():
                    if v:
                        candidates.append(v)
            except Exception:
                candidates = []
            if candidates:
                # Choose the bucket whose newest item is newest.
                def _newest(b: Dict[str, Any]) -> float:
                    return max(
                        (getattr(i, "created_at", 0) or 0) for i in b.values()
                    ) if b else 0.0
                bucket = max(candidates, key=_newest)

        self._store = bucket or {}
        return self._store

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._body is None:
            return

        try:
            items = self._gather_items()
        except Exception as exc:
            self._paint(f"[{ERROR}]todo error: {_esc(exc)}[/]")
            return

        if not items:
            self._paint(f"[{DIM}]no tasks yet[/]")
            return

        # Sort: in-progress first, then pending by priority, then done.
        def _sort_key(i: Any) -> tuple:
            status = self._get(i, "status", "pending")
            rank = {"in_progress": 0, "pending": 1, "blocked": 2}.get(status, 3)
            priority = self._priority(i)
            created = self._get(i, "created_at", 0) or 0
            return (rank, priority, created)

        items.sort(key=_sort_key)

        done = sum(
            1 for i in items
            if self._get(i, "status", "") in ("done", "completed")
        )
        total = len(items)
        lines = [f"[{MUTED}]{done}/{total} done[/]", ""]

        for it in items:
            status = self._get(it, "status", "pending")
            glyph, color = _GLYPH.get(status, ("○", MUTED))
            title = _esc(self._get(it, "title", "") or "")
            if len(title) > 32:
                title = title[:29] + "…"
            if status in ("done", "completed"):
                lines.append(
                    f"[{color}]{glyph}[/] [{DIM}][strike]{title}[/strike][/]"
                )
            elif status == "in_progress":
                lines.append(f"[{color}]{glyph}[/] [{GREEN_GLOW}]{title}[/]")
            else:
                lines.append(f"[{color}]{glyph}[/] [{TEXT}]{title}[/]")

        self._paint("\n".join(lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_items(self) -> List[Any]:
        """Pushed items win; otherwise read the store."""
        if self._pushed_items:
            return list(self._pushed_items)
        store = self._resolve_store()
        return list(store.values()) if store else []

    @staticmethod
    def _get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    @staticmethod
    def _priority(item: Any) -> int:
        p = TodoPanel._get(item, "priority", 3)
        if hasattr(p, "value"):
            p = p.value
        try:
            return int(p)
        except Exception:
            return 3

    def _paint(self, markup: str) -> None:
        # NOTE: must not be named `_render` — Textual's Widget._render()
        # is an internal hook called with no arguments and returning a
        # Visual, so shadowing it crashes every repaint of this panel.
        if markup == self._last_render:
            return
        self._last_render = markup
        if self._body is not None:
            self._body.update(markup)