"""
Tool Execution Card Widget

Displays a tool call with:
- Tool name in neon orange
- Parameters in a collapsible section
- Result with success/error coloring
- Execution duration
- Animated spinner while running
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class ToolCard(Widget):
    """A card displaying tool execution details."""

    DEFAULT_CSS = """
    ToolCard {
        height: auto;
        min-height: 1;
        padding: 0 0 0 0;
        margin: 0 0 0 0;
    }

    .tool-header {
        height: 1;
        padding: 0 1;
        background: $bg-panel;
        border-left: tall $neon-orange;
    }

    .tool-name {
        color: $neon-orange;
        text-style: bold;
    }

    .tool-status-success {
        color: $neon-green;
    }

    .tool-status-error {
        color: $neon-red;
    }

    .tool-status-running {
        color: $neon-purple;
    }

    .tool-duration {
        color: $text-dim;
    }

    .tool-params {
        color: $text-muted;
        padding: 0 0 0 2;
        height: auto;
        max-height: 6;
        overflow-y: auto;
    }

    .tool-param-key {
        color: $neon-cyan;
    }

    .tool-param-value {
        color: $text-secondary;
    }

    .tool-result {
        color: $text-secondary;
        padding: 0 0 0 2;
        height: auto;
        max-height: 10;
        overflow-y: auto;
    }

    .tool-result-error {
        color: $neon-red;
    }
    """

    running = reactive[bool](False)

    def __init__(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        duration: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.params = params or {}
        self.result = result
        self.success = success
        self.duration = duration
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        # Status icon
        if self.running:
            status_icon = "◉"
            status_class = "tool-status-running"
        elif self.success:
            status_icon = "✓"
            status_class = "tool-status-success"
        else:
            status_icon = "✗"
            status_class = "tool-status-error"

        # Duration string
        dur_str = ""
        if self.duration is not None:
            dur_str = f" ({self.duration:.2f}s)"

        # Header
        yield Static(
            f" ⚙ [tool-name]{self.tool_name}[/] "
            f"[{status_class}]{status_icon}[/]"
            f"[tool-duration]{dur_str}[/]",
            classes="tool-header",
        )

        # Parameters
        if self.params:
            param_lines = []
            for k, v in list(self.params.items())[:8]:
                vs = str(v)
                if len(vs) > 80:
                    vs = vs[:77] + "…"
                param_lines.append(
                    f" [tool-param-key]{k}[/] = [tool-param-value]{vs}[/]"
                )
            yield Static("\n".join(param_lines), classes="tool-params")

        # Result
        if self.result is not None:
            result_str = str(self.result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n… (truncated)"
            result_class = "tool-result-error" if not self.success else "tool-result"
            yield Static(result_str, classes=result_class)

    def mark_complete(self, success: bool, duration: float) -> None:
        """Mark the tool call as complete."""
        self.success = success
        self.duration = duration
        self.running = False
        self.remove_children()
        self.mount(*self.compose())

    def mark_running(self) -> None:
        """Mark as currently running."""
        self.running = True
        self._start_time = time.time()
        self.remove_children()
        self.mount(*self.compose())
