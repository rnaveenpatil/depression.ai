"""
Tool call widget - one block per tool invocation.

States:
    running  ▶ tool name + args         with spinner
    done     ✓ tool name + duration     with result summary
    error    ✗ tool name + error        with error text
    cached   ⚡ tool name + duration    served from cache
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from agent.tui.widgets.diff_view import DiffView, render_diff


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Tools whose results should be rendered as a diff.
_DIFF_TOOLS = {"write", "edit", "apply_patch", "filesystem", "patch"}

# Maximum preview length for generic results.
_PREVIEW_CHARS = 1200


class ToolCallWidget(Vertical):
    DEFAULT_CSS = f"""
    ToolCallWidget {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {BORDER};
    }}
    ToolCallWidget > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(self, tool: str, params: Dict[str, Any], **kwargs: Any):
        super().__init__(**kwargs)
        self._tool = tool
        self._params = params or {}
        self._state = "running"
        self._spin = 0
        self._result: Optional[Dict[str, Any]] = None
        self._execution_time: Optional[float] = None
        self._cached = False
        self._header_static: Optional[Static] = None
        self._body_static: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._header_static = Static(self._header_markup(), markup=True)
        self._body_static = Static("", markup=True)
        yield self._header_static
        yield self._body_static

    def on_mount(self) -> None:
        self.set_interval(0.10, self._tick)

    def _tick(self) -> None:
        if self._state != "running":
            return
        self._spin += 1
        if self._header_static is not None:
            self._header_static.update(self._header_markup())

    # ------------------------------------------------------------------

    def _args_preview(self) -> str:
        interesting = (
            "command", "path", "filePath", "query", "pattern", "url",
            "action", "ref", "selector",
        )
        for key in interesting:
            if key in self._params and self._params[key] not in (None, ""):
                val = str(self._params[key])
                if len(val) > 80:
                    val = val[:77] + "…"
                return val.replace("[", r"\[")
        if not self._params:
            return ""
        try:
            s = json.dumps(self._params, default=str)
        except Exception:
            s = str(self._params)
        if len(s) > 80:
            s = s[:77] + "…"
        return s.replace("[", r"\[")

    def _header_markup(self) -> str:
        name = self._tool
        args = self._args_preview()
        args_part = f"  [{DIM}]{args}[/]" if args else ""

        if self._state == "running":
            glyph = SPINNER[self._spin % len(SPINNER)]
            return f"[{AMBER}]{glyph}[/] [{GREEN}]{name}[/]{args_part}"
        if self._state == "cached":
            t = f"  [{DIM}]({self._execution_time:.2f}s cached)[/]" if self._execution_time else ""
            return f"[{GREEN_DIM}]⚡[/] [{GREEN}]{name}[/]{args_part}{t}"
        if self._state == "done":
            t = f"  [{DIM}]({self._execution_time:.2f}s)[/]" if self._execution_time is not None else ""
            return f"[{GREEN}]✓[/] [{GREEN}]{name}[/]{args_part}{t}"
        if self._state == "error":
            return f"[{ERROR}]✗[/] [{GREEN}]{name}[/]{args_part}"
        return name

    def _render_result(self) -> str:
        result = self._result or {}
        if not isinstance(result, dict):
            return f"  [{TEXT}]{str(result)[:800]}[/]"

        if not result.get("success", True):
            err = result.get("error") or "tool failed"
            return f"  [{ERROR}]{str(err)[:800]}[/]"

        # Diff-style rendering for file-modifying tools.
        if self._tool in _DIFF_TOOLS and result.get("path"):
            before = result.get("before") or result.get("content_before") or ""
            after = (
                result.get("after")
                or result.get("content_after")
                or result.get("new_content")
                or ""
            )
            if before and after:
                markup = render_diff(before, after, path=str(result.get("path")))
                if markup:
                    return "  " + markup.replace("\n", "\n  ")

        # Fallback: a compact preview of the interesting fields.
        preview = (
            result.get("content")
            or result.get("output")
            or result.get("text")
            or result.get("stdout")
            or result.get("result")
        )
        if preview is None:
            preview = {
                k: v for k, v in result.items()
                if k not in ("success",) and not isinstance(v, (bytes,))
            }
        if isinstance(preview, (dict, list)):
            try:
                preview = json.dumps(preview, default=str)
            except Exception:
                preview = str(preview)
        preview = str(preview).strip()
        if not preview:
            return "  " + f"[{MUTED}]ok[/]"
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS] + f"\n[{AMBER}]… truncated[/]"
        preview = preview.replace("[", r"\[")
        return "  " + preview.replace("\n", "\n  ")

    # ------------------------------------------------------------------

    def set_running(self, execution_time: Optional[float] = None) -> None:
        self._state = "running"
        self._execution_time = execution_time

    def set_result(
        self,
        result: Dict[str, Any],
        execution_time: Optional[float] = None,
        cached: bool = False,
    ) -> None:
        self._result = result
        self._execution_time = execution_time
        self._cached = cached
        if not isinstance(result, dict):
            self._state = "done"
        elif result.get("success", True):
            self._state = "cached" if cached else "done"
        else:
            self._state = "error"

        if self._header_static is not None:
            self._header_static.update(self._header_markup())
        if self._body_static is not None:
            self._body_static.update(self._render_result())

    def add_diff(self, path: str, before: str, after: str) -> None:
        """Attach an explicit before/after diff below the header."""
        if self._body_static is None:
            return
        if before == after:
            return
        child = DiffView(path=path, before=before, after=after)
        self.mount(child)