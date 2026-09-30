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

NOTE
----
The session is resolved *lazily on every refresh* rather than being
captured at construction time. `DepressionApp.compose()` builds this
panel before `on_mount` runs, and at that point `coordinator.build_agent`
/ `plan_agent` may not yet have a `.session` attached. Caching the
session up-front produced a `_session_key` that never matched the one
used by `TodoTool` / the plan-update handler, so the panel read from an
empty bucket and rendered nothing.
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
        # Prefer the app reference so we can re-resolve the live session
        # on each tick. `session` is kept as a fallback for standalone use.
        self._app = app_ref
        self._session = session
        self._store: Dict[str, Any] = {}
        self._header: Optional[Static] = None
        self._body: Optional[Static] = None
        self._last_render = ""

    def compose(self) -> ComposeResult:
        self._header = Static(
            f"[bold {GREEN}]▌ TODO[/]", markup=True
        )
        self._body = Static("", markup=True)
        yield self._header
        yield self._body

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.5, self._refresh)

    # ------------------------------------------------------------------
    # Session / store resolution
    # ------------------------------------------------------------------

    def _current_session(self) -> Any:
        """Ask the app for the live session each time; fall back to the
        snapshot captured at construction."""
        if self._app is not None:
            try:
                live = self._app._session()  # type: ignore[attr-defined]
                if live is not None:
                    return live
            except Exception:
                pass
        return self._session

    def _resolve_store(self) -> Dict[str, Any]:
        from agent.tools.todo import TodoTool

        session = self._current_session()
        sid = TodoTool._session_key(session)
        store = TodoTool._strong_keys.get(sid)

        # If the exact session-key lookup misses, fall back to the most
        # recently used bucket. This handles the case where the panel is
        # polling before the agent has attached its session, and prevents
        # the "silent empty" mode.
        if not store:
            try:
                candidates = [
                    v for v in TodoTool._strong_keys.values() if v
                ]
                if len(candidates) == 1:
                    store = candidates[0]
            except Exception:
                store = None

        self._store = store or {}
        return self._store

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._body is None:
            return
        try:
            self._resolve_store()
            items: List[Any] = sorted(
                self._store.values(),
                key=lambda i: (
                    getattr(i, "status", "pending") != "in_progress",
                    getattr(i, "priority", 3),
                    getattr(i, "created_at", 0),
                ),
            )
        except Exception as exc:
            # Surface the failure instead of dying silently inside the
            # interval callback.
            self._paint(f"[{ERROR}]todo error: {_esc(exc)}[/]")
            return

        if not items:
            self._paint(f"[{DIM}]no tasks yet[/]")
            return

        done = sum(
            1 for i in items
            if getattr(i, "status", "") in ("done", "completed")
        )
        total = len(items)
        lines = [f"[{MUTED}]{done}/{total} done[/]", ""]
        for it in items:
            status = getattr(it, "status", "pending")
            glyph, color = _GLYPH.get(status, ("○", MUTED))
            title = _esc(getattr(it, "title", "") or "")
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

    def _paint(self, markup: str) -> None:
        # NOTE: must not be named `_render` — Textual's Widget._render()
        # is an internal hook called with no arguments and returning a
        # Visual, so shadowing it crashes every repaint of this panel.
        if markup == self._last_render:
            return
        self._last_render = markup
        if self._body is not None:
            self._body.update(markup)