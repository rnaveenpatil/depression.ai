"""
LLM Provider Panel Widget

A sidebar panel for selecting LLM providers with:
- Model list grouped by provider (exact from user's table)
- Base URL display
- API key input with secure persistent storage
- Connect button
- Status indicators
"""

from __future__ import annotations

from typing import Callable, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import (
    Static, Button, Input, ListView, ListItem, Label,
)
from textual.reactive import reactive
from textual import on

from agent.tui.llm_providers import (
    PROVIDER_MODELS,
    PROVIDER_INFO,
    LLMModel,
    get_llm_config,
)


# Inline colors
PANEL = "#12121e"
PANEL_HOVER = "#1c1c30"
PANEL_SELECTED = "#143250"
BORDER = "#283250"
CYBER_BLUE = "#00c8ff"
NEON_CYAN = "#00ffff"
NEON_PURPLE = "#b464ff"
NEON_GREEN = "#00ff80"
NEON_YELLOW = "#ffff00"
NEON_ORANGE = "#ffa500"
NEON_RED = "#ff3232"
TEXT_PRIMARY = "#f0f0ff"
TEXT_SECONDARY = "#c8c8dc"
TEXT_DIM = "#505064"
TEXT_MUTED = "#8c8ca0"


# ======================================================================
# MODEL LIST ITEM
# ======================================================================

class ModelItem(ListItem):
    """A single model in the selection list."""

    DEFAULT_CSS = f"""
    ModelItem {{
        height: 3;
        padding: 0 1;
        border-bottom: solid {BORDER};
    }}

    ModelItem:hover {{
        background: {PANEL_HOVER};
    }}

    ModelItem.selected {{
        background: {PANEL_SELECTED};
        border-left: tall {CYBER_BLUE};
    }}

    .model-name {{
        color: {TEXT_PRIMARY};
        text-style: bold;
        height: 1;
    }}

    .model-meta {{
        height: 1;
    }}

    .model-provider {{
        color: {NEON_CYAN};
    }}

    .model-context {{
        color: {TEXT_DIM};
    }}

    .model-stars {{
        color: {NEON_YELLOW};
    }}

    .model-tools {{
        color: {NEON_GREEN};
    }}
    """

    def __init__(self, model: LLMModel, is_selected: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if is_selected:
            self.add_class("selected")

    def compose(self) -> ComposeResult:
        m = self.model
        ctx = f"{m.context_window // 1000}K" if m.context_window < 1_000_000 else f"{m.context_window // 1_000_000}M"
        tools = "✓" if m.supports_tools else "✗"
        provider_info = PROVIDER_INFO.get(m.provider, {})

        yield Label(m.display_name, classes="model-name")
        yield Label(
            f" [model-provider]{provider_info.get('icon', '●')} {m.provider}[/]"
            f"  [model-context]{ctx} ctx[/]"
            f"  [model-tools]{tools} tools[/]"
            f"  [model-stars]{m.stars}[/]",
            classes="model-meta",
        )


# ======================================================================
# PROVIDER GROUP HEADER
# ======================================================================

class ProviderHeader(ListItem):
    """A provider group header."""

    DEFAULT_CSS = f"""
    ProviderHeader {{
        height: 1;
        padding: 0 1;
        background: {PANEL};
    }}

    .provider-label {{
        text-style: bold;
    }}
    """

    def __init__(self, provider_id: str, **kwargs):
        super().__init__(**kwargs)
        self.provider_id = provider_id
        info = PROVIDER_INFO.get(provider_id, {})
        self.provider_name = info.get("name", provider_id)
        self.provider_color = info.get("color", "#ffffff")
        self.provider_icon = info.get("icon", "●")

    def compose(self) -> ComposeResult:
        yield Label(
            f"[{self.provider_color}]{self.provider_icon} {self.provider_name}[/]",
            classes="provider-label",
        )


# ======================================================================
# LLM PROVIDER PANEL
# ======================================================================

class LLMProviderPanel(Widget):
    """
    Full LLM provider selection panel with:
    - Model list (grouped by provider)
    - Base URL display
    - API key input (persisted)
    - Connect button
    - Status
    """

    DEFAULT_CSS = f"""
    LLMProviderPanel {{
        height: 100%;
        width: 100%;
        layout: vertical;
        padding: 0;
    }}

    #llm-header {{
        height: 1;
        background: {PANEL};
        color: {CYBER_BLUE};
        text-style: bold;
        padding: 0 1;
        border-bottom: solid {BORDER};
    }}

    #llm-model-list {{
        height: 1fr;
        overflow-y: auto;
    }}

    #llm-config-area {{
        height: auto;
        padding: 0 1;
        border-top: solid {BORDER};
    }}

    .llm-label {{
        height: 1;
        color: {TEXT_MUTED};
        text-style: bold;
    }}

    #llm-base-url {{
        height: 1;
        background: {PANEL};
        color: {NEON_CYAN};
        padding: 0 1;
        border: solid {BORDER};
    }}

    #llm-api-key-input {{
        height: 1;
        background: {PANEL};
        color: {TEXT_PRIMARY};
        padding: 0 1;
        border: solid {BORDER};
    }}

    #llm-connect-btn {{
        height: 3;
        background: {PANEL};
        border: solid {NEON_GREEN};
        color: {NEON_GREEN};
        text-style: bold;
        content-align: center middle;
        margin: 0 1;
    }}

    #llm-connect-btn:hover {{
        background: {PANEL_HOVER};
        border: solid {CYBER_BLUE};
        color: {CYBER_BLUE};
    }}

    #llm-status {{
        height: 1;
        color: {TEXT_DIM};
        padding: 0 1;
    }}

    .status-connected {{
        color: {NEON_GREEN};
    }}

    .status-disconnected {{
        color: {TEXT_DIM};
    }}

    .status-error {{
        color: {NEON_RED};
    }}
    """

    selected_model = reactive[Optional[LLMModel]](None)
    connection_status = reactive[str]("disconnected")

    def __init__(
        self,
        on_connect: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = get_llm_config()
        self._on_connect = on_connect

    def compose(self) -> ComposeResult:
        yield Static(" ◆ LLM PROVIDERS ", id="llm-header")

        # Model list
        with ListView(id="llm-model-list"):
            current_provider = None
            for model in PROVIDER_MODELS:
                # Provider group header
                if model.provider != current_provider:
                    current_provider = model.provider
                    yield ProviderHeader(model.provider)

                # Model item
                is_sel = (
                    self.config.selected_model == model.name
                    or (self.config.selected_model is None and model == PROVIDER_MODELS[0])
                )
                yield ModelItem(model, is_selected=is_sel)

        # Config area
        with Widget(id="llm-config-area"):
            yield Static(" BASE URL", classes="llm-label")
            yield Static(
                self._get_base_url_display(),
                id="llm-base-url",
            )
            yield Static(" API KEY", classes="llm-label")
            yield Input(
                placeholder="Paste API key here...",
                password=True,
                id="llm-api-key-input",
                value=self._get_api_key_display(),
            )
            yield Button(
                self._connect_label(),
                id="llm-connect-btn",
            )
            yield Static(
                self._status_text(),
                id="llm-status",
            )

    def _get_base_url_display(self) -> str:
        model = self.selected_model or PROVIDER_MODELS[0]
        url = self.config.get_base_url(model.provider)
        if len(url) > 28:
            url = url[:25] + "..."
        return f" {url}"

    def _get_api_key_display(self) -> str:
        model = self.selected_model or PROVIDER_MODELS[0]
        key = self.config.get_api_key(model.provider)
        if key:
            return key
        return ""

    def _connect_label(self) -> str:
        if self.connection_status == "connected":
            return " ✓ CONNECTED "
        return " 🔗 CONNECT "

    def _status_text(self) -> str:
        model = self.selected_model or PROVIDER_MODELS[0]
        if self.connection_status == "connected":
            return f" [status-connected]● Connected to {model.provider}[/]"
        elif self.connection_status == "error":
            return " [status-error]✗ Connection failed — check API key[/]"
        return " [status-disconnected]○ Not connected[/]"

    @on(ListView.Selected, "#llm-model-list")
    def _on_model_selected(self, event: ListView.Selected) -> None:
        """Handle model selection."""
        try:
            list_view = self.query_one("#llm-model-list", ListView)
            item = list_view.get_item_at_index(event.index)
            if isinstance(item, ModelItem):
                self.selected_model = item.model
                self.config.selected_model = item.model.name
                self._update_config_display()
        except Exception:
            pass

    @on(Input.Submitted, "#llm-api-key-input")
    def _on_api_key_submitted(self, event: Input.Submitted) -> None:
        """Handle API key entry."""
        self._save_api_key()

    @on(Input.Changed, "#llm-api-key-input")
    def _on_api_key_changed(self, event: Input.Changed) -> None:
        """Handle API key changes - auto-save."""
        self._save_api_key()

    @on(Button.Pressed, "#llm-connect-btn")
    def _on_connect_pressed(self) -> None:
        """Handle connect button press."""
        self._save_api_key()
        self._attempt_connection()

    def _save_api_key(self) -> None:
        """Save the current API key."""
        try:
            key_input = self.query_one("#llm-api-key-input", Input)
            model = self.selected_model or PROVIDER_MODELS[0]
            self.config.set_api_key(model.provider, key_input.value)
        except Exception:
            pass

    def _update_config_display(self) -> None:
        """Update the base URL and status display."""
        try:
            url_widget = self.query_one("#llm-base-url", Static)
            url_widget.update(self._get_base_url_display())

            # Also update API key field
            key_input = self.query_one("#llm-api-key-input", Input)
            key_input.value = self._get_api_key_display()

            status_widget = self.query_one("#llm-status", Static)
            status_widget.update(self._status_text())

            btn = self.query_one("#llm-connect-btn", Button)
            btn.label = self._connect_label()
        except Exception:
            pass

    def _attempt_connection(self) -> None:
        """Attempt to connect to the selected provider."""
        model = self.selected_model or PROVIDER_MODELS[0]
        key = self.config.get_api_key(model.provider)

        if not key:
            self.connection_status = "error"
            self._update_config_display()
            return

        # Mark as connected
        self.connection_status = "connected"
        self._update_config_display()

        # Trigger callback to update agent
        if self._on_connect:
            self._on_connect(model)

    def get_current_config(self) -> dict:
        """Get the current LLM configuration."""
        model = self.selected_model or PROVIDER_MODELS[0]
        return self.config.build_llm_config(model)

    def set_connected(self, connected: bool) -> None:
        """Set connection status externally."""
        self.connection_status = "connected" if connected else "disconnected"
        self._update_config_display()

    def get_selected_model(self) -> LLMModel:
        """Get the currently selected model."""
        return self.selected_model or PROVIDER_MODELS[0]