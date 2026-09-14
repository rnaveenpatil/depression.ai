"""
Status Bar View

The bottom status bar showing:
- Agent status (idle/thinking/acting)
- Token count
- Cost
- Current model
- Active tool
- Keyboard shortcuts hint
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


# Inline colors
DEEP_VOID = "#06060c"
TEXT_MUTED = "#8c8ca0"
TEXT_DIM = "#505064"
NEON_ORANGE = "#ffa500"
NEON_GREEN = "#00ff80"
NEON_RED = "#ff3232"
NEON_PURPLE = "#b464ff"
NEON_CYAN = "#00ffff"
NEON_YELLOW = "#ffff00"


class StatusBar(Widget):
    """The bottom status bar."""

    DEFAULT_CSS = f"""
    StatusBar {{
        height: 1;
        background: #06060c;
        dock: bottom;
        layout: horizontal;
        padding: 0 1;
    }}

    #status-indicator {{
        width: auto;
        content-align: left middle;
        padding: 0 1 0 0;
    }}

    #status-model {{
        width: 1fr;
        content-align: left middle;
        color: #8c8ca0;
    }}

    #status-tokens {{
        width: auto;
        content-align: center middle;
        color: #505064;
        padding: 0 1;
    }}

    #status-cost {{
        width: auto;
        content-align: center middle;
        color: #505064;
        padding: 0 1;
    }}

    #status-tool {{
        width: auto;
        content-align: center middle;
        color: #ffa500;
        padding: 0 1;
    }}

    #status-shortcuts {{
        width: auto;
        content-align: right middle;
        color: #505064;
    }}
    """

    status = reactive[str]("idle")
    model = reactive[str]("—")
    tokens = reactive[int](0)
    cost = reactive[float](0.0)
    active_tool = reactive[str]("")

    STATUS_COLORS = {
        "idle": "#8c8ca0",
        "thinking": "#b464ff",
        "planning": "#00ffff",
        "acting": "#ffa500",
        "observing": "#ffff00",
        "success": "#00ff80",
        "error": "#ff3232",
    }

    STATUS_ICONS = {
        "idle": "○",
        "thinking": "◉",
        "planning": "◈",
        "acting": "⚡",
        "observing": "👁",
        "success": "✓",
        "error": "✗",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(self._indicator_text(), id="status-indicator")
        yield Static(self.model, id="status-model")
        yield Static(self._tokens_text(), id="status-tokens")
        yield Static(self._cost_text(), id="status-cost")
        yield Static(self._tool_text(), id="status-tool")
        yield Static(" Tab:switch │ ?:help │ Ctrl+P:palette ", id="status-shortcuts")

    def _indicator_text(self) -> str:
        icon = self.STATUS_ICONS.get(self.status, "○")
        color = self.STATUS_COLORS.get(self.status, "#8c8ca0")
        return f" [{color}]{icon} {self.status}[/]"

    def _tokens_text(self) -> str:
        if self.tokens >= 1_000_000:
            return f" ⚡ {self.tokens / 1_000_000:.1f}M tok"
        elif self.tokens >= 1_000:
            return f" ⚡ {self.tokens / 1_000:.1f}K tok"
        return f" ⚡ {self.tokens} tok"

    def _cost_text(self) -> str:
        if self.cost >= 1.0:
            return f" 💰 ${self.cost:.2f}"
        elif self.cost >= 0.01:
            return f" 💰 ${self.cost:.3f}"
        return f" 💰 ${self.cost:.4f}"

    def _tool_text(self) -> str:
        if self.active_tool:
            return f" ⚙ {self.active_tool}"
        return ""

    def set_status(self, status: str) -> None:
        """Update the agent status."""
        self.status = status
        try:
            self.query_one("#status-indicator").update(self._indicator_text())
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        """Update the model."""
        self.model = model
        try:
            self.query_one("#status-model").update(model)
        except Exception:
            pass

    def set_tokens(self, tokens: int) -> None:
        """Update the token count."""
        self.tokens = tokens
        try:
            self.query_one("#status-tokens").update(self._tokens_text())
        except Exception:
            pass

    def set_cost(self, cost: float) -> None:
        """Update the cost."""
        self.cost = cost
        try:
            self.query_one("#status-cost").update(self._cost_text())
        except Exception:
            pass

    def set_tool(self, tool: str) -> None:
        """Update the active tool."""
        self.active_tool = tool
        try:
            self.query_one("#status-tool").update(self._tool_text())
        except Exception:
            pass

    def update_all(
        self,
        status: str = None,
        model: str = None,
        tokens: int = None,
        cost: float = None,
        tool: str = None,
    ) -> None:
        """Update multiple fields at once."""
        if status is not None:
            self.set_status(status)
        if model is not None:
            self.set_model(model)
        if tokens is not None:
            self.set_tokens(tokens)
        if cost is not None:
            self.set_cost(cost)
        if tool is not None:
            self.set_tool(tool)