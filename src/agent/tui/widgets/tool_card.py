"""Tool execution card — princess CLI styling."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class ToolCard(Widget):
    """Compact card showing a tool invocation and its outcome."""

    DEFAULT_CSS = """
    ToolCard {
        height: auto;
        min-height: 1;
        padding: 0 2;
        margin: 0 0 1 0;
    }
    .tool-header {
        height: 1;
        color: #ffd28a;
        text-style: bold;
    }
    .tool-params {
        color: #8a849a;
        padding: 0 0 0 2;
        height: auto;
        max-height: 6;
        overflow-y: auto;
    }
    .tool-result {
        color: #e8e3f0;
        padding: 0 0 0 2;
        height: auto;
        max-height: 12;
        overflow-y: auto;
    }
    .tool-error { color: #ff6b8a; }
    """

    running = reactive(False)

    def __init__(self, tool_name: str,
                 params: Optional[Dict[str, Any]] = None,
                 result: Optional[Any] = None,
                 success: bool = True,
                 duration: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.params = params or {}
        self.result = result
        self.success = success
        self.duration = duration
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        if self.running:
            icon, color = "◉", "#b794f6"
        elif self.success:
            icon, color = "✓", "#7ef7c0"
        else:
            icon, color = "✗", "#ff6b8a"

        dur = f" [{self.duration:.2f}s]" if self.duration is not None else ""
        yield Static(
            f"[#ffd28a]⚙ {self.tool_name}[/] "
            f"[{color}]{icon}[/]"
            f"[#524d60]{dur}[/]",
            classes="tool-header",
        )

        if self.params:
            lines = []
            for k, v in list(self.params.items())[:8]:
                vs = str(v)
                if len(vs) > 80:
                    vs = vs[:77] + "…"
                lines.append(f"  [#7ee7ff]{k}[/] = [#8a849a]{vs}[/]")
            yield Static("\n".join(lines), classes="tool-params")

        if self.result is not None:
            rs = str(self.result)
            if len(rs) > 500:
                rs = rs[:500] + "\n… (truncated)"
            cls = "tool-result tool-error" if not self.success else "tool-result"
            yield Static(rs, classes=cls)

    def mark_complete(self, success: bool, duration: float) -> None:
        self.success = success
        self.duration = duration
        self.running = False
        self.remove_children()
        self.mount(*self.compose())

    def mark_running(self) -> None:
        self.running = True
        self._start_time = time.time()
        self.remove_children()
        self.mount(*self.compose())