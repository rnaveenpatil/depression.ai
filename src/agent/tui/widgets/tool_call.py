"""
Tool call widget - per-tool rendering with live state transitions.

States: pending → running → completed | error

Renderers are chosen by tool name:
    bash/terminal   → $ command + stdout (10 lines, expandable)
    read            → "Loaded <path>" + byte count
    write/edit      → inline diff
    glob/grep       → pattern + match count
    webfetch        → status + preview
    generic/mcp     → output truncated to 3 lines

Every piece of dynamic content is escaped with markup_escape so brackets
in tool output (JSON like data[0], logs like [INFO], file paths like
foo[1].txt) do not trip Textual's markup parser.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.markup import escape as markup_escape
from textual.widgets import Static

from agent.tui.widgets.diff_view import render_diff


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_DIFF_TOOLS = {"write", "edit", "apply_patch", "filesystem", "patch"}
_BASH_TOOLS = {"bash", "terminal"}
_READ_TOOLS = {"read"}
_SEARCH_TOOLS = {"grep", "glob", "search"}
_WEB_TOOLS = {"webfetch", "websearch", "web"}

_BASH_PREVIEW_LINES = 10
_GENERIC_PREVIEW_LINES = 3


def _esc(text: Any) -> str:
    """Escape a value for safe embedding in a Textual markup string."""
    return markup_escape(str(text))


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
        self._state = "pending"
        self._spin = 0
        self._result: Optional[Dict[str, Any]] = None
        self._execution_time: Optional[float] = None
        self._cached = False
        self._expanded = False
        self._header_static: Optional[Static] = None
        self._body_static: Optional[Static] = None
        self._alive = True

    def compose(self) -> ComposeResult:
        self._header_static = Static(self._header_markup(), markup=True)
        self._body_static = Static("", markup=True)
        yield self._header_static
        yield self._body_static

    def on_mount(self) -> None:
        self.set_interval(0.10, self._tick)

    def on_unmount(self) -> None:
        self._alive = False
        self._header_static = None
        self._body_static = None

    def _tick(self) -> None:
        if not self._alive:
            return
        if self._state not in ("pending", "running"):
            return
        self._spin += 1
        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def set_running(self, execution_time: Optional[float] = None) -> None:
        if not self._alive:
            return
        self._state = "running"
        self._execution_time = execution_time
        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False

    def set_result(
        self,
        result: Dict[str, Any],
        execution_time: Optional[float] = None,
        cached: bool = False,
    ) -> None:
        if not self._alive:
            return
        self._result = result
        self._execution_time = execution_time
        self._cached = cached

        if not isinstance(result, dict):
            self._state = "completed"
        elif result.get("running"):
            self._state = "background"
        elif result.get("success", True):
            self._state = "cached" if cached else "completed"
        else:
            self._state = "error"

        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False
                return
        if self._body_static is not None:
            try:
                self._body_static.update(self._render_result())
            except Exception:
                self._alive = False

    def toggle_expand(self) -> None:
        if not self._alive:
            return
        self._expanded = not self._expanded
        if self._body_static is not None:
            try:
                self._body_static.update(self._render_result())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------

    def _target(self) -> str:
        for key in ("command", "filePath", "path", "pattern", "query", "url"):
            val = self._params.get(key)
            if val:
                s = str(val)
                return s if len(s) <= 80 else s[:77] + "…"
        return ""

    def _header_markup(self) -> str:
        name = _esc(self._tool)
        target = _esc(self._target())
        target_part = f"  [{DIM}]{target}[/]" if target else ""

        if self._state in ("pending", "running"):
            glyph = SPINNER[self._spin % len(SPINNER)]
            return f"[{AMBER}]{glyph}[/] [{GREEN}]{name}[/]{target_part}"
        if self._state == "background":
            pid = _esc(self._result.get("pid") if isinstance(self._result, dict) else "?")
            return f"[{AMBER}]◉[/] [{GREEN}]{name}[/]{target_part}  [{DIM}](pid {pid})[/]"
        if self._state == "cached":
            t = f"  [{DIM}]({self._execution_time:.2f}s cached)[/]" if self._execution_time else ""
            return f"[{GREEN_DIM}]⚡[/] [{GREEN}]{name}[/]{target_part}{t}"
        if self._state == "completed":
            t = f"  [{DIM}]({self._execution_time:.2f}s)[/]" if self._execution_time is not None else ""
            return f"[{GREEN}]✓[/] [{GREEN}]{name}[/]{target_part}{t}"
        return f"[{ERROR}]✗[/] [{GREEN}]{name}[/]{target_part}"

    # ------------------------------------------------------------------
    # BODY — per-tool renderers
    # ------------------------------------------------------------------

    def _render_result(self) -> str:
        result = self._result or {}
        if not isinstance(result, dict):
            return f"  [{TEXT}]{_esc(result)[:800]}[/]"

        if not result.get("success", True):
            err = result.get("error") or "tool failed"
            return f"  [{ERROR}]{_esc(err)[:800]}[/]"

        if self._tool in _DIFF_TOOLS:
            rendered = self._render_diff(result)
            if rendered:
                return rendered
        if self._tool in _BASH_TOOLS:
            return self._render_bash(result)
        if self._tool in _READ_TOOLS:
            return self._render_read(result)
        if self._tool in _SEARCH_TOOLS:
            return self._render_search(result)
        if self._tool in _WEB_TOOLS:
            return self._render_web(result)

        return self._render_generic(result)

    def _render_diff(self, result: Dict[str, Any]) -> str:
        before = result.get("before") or result.get("content_before") or ""
        after = (
            result.get("after")
            or result.get("content_after")
            or result.get("new_content")
            or ""
        )
        if not before or not after:
            return ""
        path = str(result.get("path") or "")
        markup = render_diff(before, after, path=path)
        if not markup:
            return ""
        return "  " + markup.replace("\n", "\n  ")

    def _render_bash(self, result: Dict[str, Any]) -> str:
        command = _esc(result.get("command") or self._target())
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        exit_code = result.get("exit_code")

        lines = [f"  [{DIM}]$ {command}[/]"]
        output = (stdout + ("\n" + stderr if stderr else "")).strip()
        if not output:
            lines.append(f"  [{MUTED}](no output)[/]")
        else:
            out_lines = output.splitlines()
            show = out_lines if self._expanded else out_lines[:_BASH_PREVIEW_LINES]
            for ln in show:
                lines.append(f"  [{TEXT}]{_esc(ln)}[/]")
            if not self._expanded and len(out_lines) > _BASH_PREVIEW_LINES:
                hidden = len(out_lines) - _BASH_PREVIEW_LINES
                lines.append(f"  [{AMBER}]… {hidden} more line{'s' if hidden != 1 else ''} · press 'e' to expand[/]")
        if exit_code not in (None, 0):
            lines.append(f"  [{ERROR}]exit {exit_code}[/]")
        return "\n".join(lines)

    def _render_read(self, result: Dict[str, Any]) -> str:
        path = _esc(result.get("path") or self._target())
        size = result.get("size")
        truncated = result.get("truncated")
        content = result.get("content") or ""
        size_str = f" ({size} bytes)" if size is not None else ""
        header = f"  [{DIM}]Loaded {path}{size_str}[/]"
        if truncated:
            header += f"  [{AMBER}](truncated)[/]"
        if not content:
            return header
        lines = content.splitlines()
        show = lines if self._expanded else lines[:5]
        body = "\n".join(f"  [{TEXT}]{_esc(ln)}[/]" for ln in show)
        if not self._expanded and len(lines) > 5:
            hidden = len(lines) - 5
            body += f"\n  [{AMBER}]… {hidden} more line{'s' if hidden != 1 else ''} · press 'e' to expand[/]"
        return header + "\n" + body

    def _render_search(self, result: Dict[str, Any]) -> str:
        count = result.get("count", 0)
        pattern = _esc(result.get("pattern") or self._target())
        if count == 0:
            return f"  [{MUTED}]no matches for '{pattern}'[/]"
        matches = result.get("matches") or []
        lines = [f"  [{TEXT}]{count} match{'es' if count != 1 else ''} for '{pattern}'[/]"]
        for m in matches[:5]:
            if isinstance(m, dict):
                path = _esc(m.get("path", ""))
                text = _esc(m.get("text", "")[:80])
                lines.append(f"  [{DIM}]{path}[/]  [{MUTED}]{text}[/]")
            else:
                lines.append(f"  [{DIM}]{_esc(m)}[/]")
        if count > 5:
            lines.append(f"  [{AMBER}]… {count - 5} more matches[/]")
        return "\n".join(lines)

    def _render_web(self, result: Dict[str, Any]) -> str:
        status = result.get("status", "")
        url = _esc(result.get("url") or self._target())
        content = result.get("content") or ""
        header = f"  [{DIM}]HTTP {status} · {url}[/]" if status else f"  [{DIM}]{url}[/]"
        if not content:
            return header
        preview = _esc(content.strip().replace("\n", " ")[:200])
        return header + f"\n  [{TEXT}]{preview}[/]"

    def _render_generic(self, result: Dict[str, Any]) -> str:
        preview = (
            result.get("content")
            or result.get("output")
            or result.get("text")
            or result.get("result")
        )
        if preview is None:
            preview = {k: v for k, v in result.items() if k not in ("success",)}
        if isinstance(preview, (dict, list)):
            try:
                preview = json.dumps(preview, default=str)
            except Exception:
                preview = str(preview)
        preview = str(preview).strip()
        if not preview:
            return f"  [{MUTED}]ok[/]"
        lines = preview.splitlines()
        show = lines if self._expanded else lines[:_GENERIC_PREVIEW_LINES]
        body = "\n".join(f"  [{TEXT}]{_esc(ln)}[/]" for ln in show)
        if not self._expanded and len(lines) > _GENERIC_PREVIEW_LINES:
            hidden = len(lines) - _GENERIC_PREVIEW_LINES
            body += f"\n  [{AMBER}]… {hidden} more line{'s' if hidden != 1 else ''} · press 'e' to expand[/]"
        return body