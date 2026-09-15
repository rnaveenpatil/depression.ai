"""Terminal-native model picker + LLM connection panel."""
from __future__ import annotations

from typing import Callable, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Input, ListView, ListItem, Label
from textual.reactive import reactive
from textual import on

from agent.tui.llm_providers import (
    PROVIDER_MODELS, PROVIDER_INFO, LLMModel, get_llm_config,
)


class ModelItem(ListItem):
    """One selectable model row."""
    DEFAULT_CSS = """
    ModelItem { height: 2; padding: 0 1; background: transparent; }
    ModelItem:hover, ModelItem:focus { background: #1a1820; }
    ModelItem.selected { background: #1a1820; }
    .model-name { color: #e8e3f0; text-style: bold; height: 1; }
    .model-meta { color: #8a849a; height: 1; }
    """

    def __init__(self, model: LLMModel, is_selected: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if is_selected:
            self.add_class("selected")

    def compose(self) -> ComposeResult:
        m = self.model
        ctx = (f"{m.context_window // 1000}K" if m.context_window < 1_000_000
               else f"{m.context_window // 1_000_000}M")
        provider = PROVIDER_INFO.get(m.provider, {}).get("name", m.provider)
        yield Label(m.display_name, classes="model-name")
        yield Label(f"{provider}  ·  {ctx}  ·  {m.stars}", classes="model-meta")


class ProviderHeader(ListItem):
    """A non-selectable provider section header."""
    DEFAULT_CSS = """
    ProviderHeader { height: 1; padding: 0 1; background: transparent; }
    .provider-label { color: #ff9bc7; text-style: bold; }
    """

    def __init__(self, provider: str, **kwargs):
        super().__init__(**kwargs)
        self.provider = provider

    def compose(self) -> ComposeResult:
        info = PROVIDER_INFO.get(self.provider, {})
        yield Label(f"── {info.get('name', self.provider).upper()}",
                    classes="provider-label")


class LLMProviderPanel(Widget):
    """Model picker + endpoint/API-key form. Sits inside the sidebar."""

    DEFAULT_CSS = """
    LLMProviderPanel {
        height: 100%;
        width: 100%;
        layout: vertical;
    }
    #llm-header {
        height: 1;
        padding: 0 1;
        color: #8a849a;
        text-style: bold;
        border-bottom: solid #2a2735;
    }
    #llm-model-list {
        height: 1fr;
        min-height: 6;
        border-bottom: solid #2a2735;
    }
    #llm-config-area {
        height: auto;
        min-height: 14;
        padding: 1;
        background: #131118;
    }
    .llm-label {
        height: 1;
        color: #8a849a;
        text-style: bold;
    }
    #llm-selected-model {
        height: 1;
        color: #7ee7ff;
        margin-bottom: 1;
    }
    #llm-base-url-input, #llm-api-key-input {
        height: 3;
        background: #0e0d12;
        border: round #2a2735;
        color: #e8e3f0;
        padding: 0 1;
        margin-bottom: 1;
    }
    #llm-base-url-input:focus, #llm-api-key-input:focus {
        border: round #ff9bc7;
    }
    #llm-connect-btn {
        height: 3;
        margin-top: 1;
        background: transparent;
        border: round #b794f6;
        color: #b794f6;
        text-style: bold;
    }
    #llm-connect-btn:hover, #llm-connect-btn:focus {
        background: #1a1820;
        color: #ffffff;
        border: round #ff9bc7;
    }
    #llm-status {
        height: 1;
        margin-top: 1;
        color: #8a849a;
    }
    .status-connected { color: #7ef7c0; }
    .status-error { color: #ff6b8a; }
    """

    selected_model    = reactive(None)
    connection_status = reactive("disconnected")

    def __init__(self, on_connect: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = get_llm_config()
        self._on_connect = on_connect
        self.selected_model = (self.config.get_selected_model_info()
                               or PROVIDER_MODELS[0])

    def compose(self) -> ComposeResult:
        yield Static("MODEL / CONNECTION", id="llm-header")

        with ListView(id="llm-model-list"):
            current_provider = None
            for model in PROVIDER_MODELS:
                if model.provider != current_provider:
                    current_provider = model.provider
                    yield ProviderHeader(current_provider)
                yield ModelItem(model, model.name == self.selected_model.name)

        with Widget(id="llm-config-area"):
            yield Static("SELECTED", classes="llm-label")
            yield Static(self.selected_model.display_name,
                         id="llm-selected-model")

            yield Static("BASE URL", classes="llm-label")
            yield Input(
                value=self.config.get_base_url(self.selected_model.provider),
                placeholder="https://api.example.com/v1",
                id="llm-base-url-input",
            )

            yield Static("API KEY", classes="llm-label")
            existing_key = self.config.get_api_key(self.selected_model.provider)
            yield Input(
                value="••••••••" if existing_key else "",
                password=True,
                placeholder="paste api key here and press enter",
                id="llm-api-key-input",
            )

            yield Button("CONNECT  ↵", id="llm-connect-btn")
            yield Static(self._status_text(), id="llm-status")

    def _status_text(self) -> str:
        provider = self.selected_model.provider
        if self.connection_status == "connected":
            return f"[#7ef7c0]● connected · {provider}[/]"
        if self.connection_status == "error":
            return "[#ff6b8a]× connection failed · check key/url[/]"
        has_key = bool(self.config.get_api_key(provider))
        marker = "saved" if has_key else "required"
        return f"[#8a849a]○ {provider} · key {marker}[/]"

    def _refresh(self) -> None:
        try:
            self.query_one("#llm-selected-model", Static).update(
                self.selected_model.display_name)
            self.query_one("#llm-base-url-input", Input).value = \
                self.config.get_base_url(self.selected_model.provider)
            existing = self.config.get_api_key(self.selected_model.provider)
            self.query_one("#llm-api-key-input", Input).value = \
                "••••••••" if existing else ""
            self.query_one("#llm-status", Static).update(self._status_text())
        except Exception:
            pass

    @on(ListView.Selected, "#llm-model-list")
    def _on_model_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ModelItem):
            self.selected_model = item.model
            self.config.selected_model = item.model.name
            self.connection_status = "disconnected"
            self._refresh()

    @on(Button.Pressed, "#llm-connect-btn")
    def _on_connect_pressed(self) -> None:
        self._do_connect()

    @on(Input.Submitted, "#llm-api-key-input")
    def _on_api_key_submitted(self, event: Input.Submitted) -> None:
        self._do_connect()

    @on(Input.Submitted, "#llm-base-url-input")
    def _on_base_url_submitted(self, event: Input.Submitted) -> None:
        try:
            self.query_one("#llm-api-key-input", Input).focus()
        except Exception:
            pass

    def _do_connect(self) -> None:
        provider = self.selected_model.provider
        base = self.query_one("#llm-base-url-input", Input)
        key_input = self.query_one("#llm-api-key-input", Input)

        # If the field still shows the placeholder dots, reuse the stored key.
        if key_input.value == "••••••••":
            key = self.config.get_api_key(provider)
        else:
            key = key_input.value.strip()

        url = base.value.strip().rstrip("/")

        if not key or not url:
            self.connection_status = "error"
            self._refresh()
            return

        self.config.set_base_url(provider, url)
        self.config.set_api_key(provider, key)
        self.config.selected_model = self.selected_model.name

        # Try to push into the runtime registry if it exists.
        try:
            from agent.llm.provider import get_llm_registry
            registry = get_llm_registry()
            registry.set_api_key(provider, key)
            registry.set_model(self.selected_model.name)
        except Exception:
            pass

        self.connection_status = "connected"
        self._refresh()
        if self._on_connect:
            try:
                self._on_connect(self.selected_model)
            except Exception:
                pass

    def get_current_config(self):
        return self.config.build_llm_config(self.selected_model)

    def set_connected(self, connected: bool) -> None:
        self.connection_status = "connected" if connected else "disconnected"
        self._refresh()

    def get_selected_model(self):
        return self.selected_model