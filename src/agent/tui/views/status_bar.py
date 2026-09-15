"""Minimal one-line status bar — claude-code style."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class StatusBar(Widget):
    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: #0e0d12;
        dock: bottom;
        layout: horizontal;
        padding: 0 2;
    }
    #status-indicator { width: auto; padding: 0 1 0 0; }
    #status-model     { width: 1fr; color: #8a849a; }
    #status-tokens    { width: auto; color: #524d60; padding: 0 1; }
    #status-cost      { width: auto; color: #524d60; padding: 0 1; }
    #status-tool      { width: auto; color: #7ee7ff; padding: 0 1; }
    #status-shortcuts { width: auto; color: #524d60; }
    """

    status      = reactive("idle")
    model       = reactive("—")
    tokens      = reactive(0)
    cost        = reactive(0.0)
    active_tool = reactive("")

    STATUS_COLORS = {
        "idle": "#8a849a", "thinking": "#b794f6", "planning": "#7ee7ff",
        "acting": "#ffd28a", "observing": "#7ee7ff",
        "success": "#7ef7c0", "error": "#ff6b8a",
    }
    STATUS_ICONS = {
        "idle": "○", "thinking": "◉", "planning": "◈", "acting": "⚡",
        "observing": "◌", "success": "✓", "error": "✗",
    }

    def compose(self) -> ComposeResult:
        yield Static(self._indicator_text(), id="status-indicator")
        yield Static(self.model, id="status-model")
        yield Static(self._tokens_text(), id="status-tokens")
        yield Static(self._cost_text(), id="status-cost")
        yield Static("", id="status-tool")
        yield Static(" tab:mode · ctrl+p:commands · f1:help · f2:rail ",
                     id="status-shortcuts")

    def _indicator_text(self) -> str:
        icon  = self.STATUS_ICONS.get(self.status, "○")
        color = self.STATUS_COLORS.get(self.status, "#8a849a")
        return f"[{color}]{icon} {self.status}[/]"

    def _tokens_text(self) -> str:
        if self.tokens >= 1_000_000:
            return f"⚡ {self.tokens/1_000_000:.1f}M"
        if self.tokens >= 1_000:
            return f"⚡ {self.tokens/1_000:.1f}K"
        return f"⚡ {self.tokens}"

    def _cost_text(self) -> str:
        if self.cost >= 1.0:
            return f"${self.cost:.2f}"
        if self.cost >= 0.01:
            return f"${self.cost:.3f}"
        return f"${self.cost:.4f}"

    def _tool_text(self) -> str:
        return f"⚙ {self.active_tool}" if self.active_tool else ""

    def _set(self, id_: str, text: str) -> None:
        try:
            self.query_one(id_).update(text)
        except Exception:
            pass

    def set_status(self, status: str) -> None:
        self.status = status
        self._set("#status-indicator", self._indicator_text())

    def set_model(self, model: str) -> None:
        self.model = model
        self._set("#status-model", model)

    def set_tokens(self, tokens: int) -> None:
        self.tokens = tokens
        self._set("#status-tokens", self._tokens_text())

    def set_cost(self, cost: float) -> None:
        self.cost = cost
        self._set("#status-cost", self._cost_text())

    def set_tool(self, tool: str) -> None:
        self.active_tool = tool
        self._set("#status-tool", self._tool_text())

    def update_all(self, status=None, model=None, tokens=None, cost=None, tool=None) -> None:
        if status is not None: self.set_status(status)
        if model  is not None: self.set_model(model)
        if tokens is not None: self.set_tokens(tokens)
        if cost   is not None: self.set_cost(cost)
        if tool   is not None: self.set_tool(tool)