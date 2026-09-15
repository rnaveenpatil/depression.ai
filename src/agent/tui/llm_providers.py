"""Terminal-native model picker + LLM connection panel.

Uses plain ScrollableContainer + Static rows instead of ListView so the model
list always renders inside the narrow sidebar.
"""
from __future__ import annotations

from typing import Callable, Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Button, Input
from textual.reactive import reactive
from textual import on

from agent.tui.llm_providers import (
    PROVIDER_MODELS, PROVIDER_INFO, LLMModel, get_llm_config,
)


class ModelButton(Button):
    """One model row — a Button so it's clickable and focusable."""
    DEFAULT_CSS = """
    ModelButton {
        height: 2;
        width: 100%;
        background: transparent;
        border: none;
        color: #e8e3f0;
        text-align: left;
        padding: 0 1;
        content-align: left middle;
    }
    ModelButton:hover { background: #1a1820; color: #ff9bc7; }
    ModelButton:focus { background: #1a1820; color: #ff9bc7; }
    ModelButton.-selected { background: #1a1820; color: #7ee7ff; }
    """

    def __init__(self, model: LLMModel, selected: bool = False, **kwargs):
        label = f"{model.display_name}  ·  {model.stars}"
        super().__init__(label=label, **kwargs)
        self.model = model
        if selected:
            self.add_class("-selected")


class ProviderLabel(Static):
    DEFAULT_CSS = """
    ProviderLabel {
        height: 1;
        width: 100%;
        color: #ff9bc7;
        text-style: bold;
        padding: 0 1;
        background: transparent;
    }
    """

    def __init__(self, provider: str):
        info = PROVIDER_INFO.get(provider, {})
        super().__init__(f"── {info.get('name', provider).upper()}")


class LLMProviderPanel(Widget):
    """Model picker + endpoint/API-key form."""

    DEFAULT_CSS = """
    LLMProviderPanel {
        height: 100%;
        width: 100%;
        layout: vertical;
        background: #0e0d12;
    }
    #llm-header {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: #8a849a;
        text-style: bold;
        background: #08070c;
        border-bottom: solid #2a2735;
    }
    #llm-model-scroll {
        height: 1fr;
        width: 100%;
        background: #0e0d12;
        border-bottom: solid #2a2735;
    }
    #llm-config {
        height: 15;
        width: 100%;
        padding: 0 1;
        background: #131118;
    }
    .llm-label {
        height: 1;
        width: 100%;
        color: #8a849a;
        text-style: bold;
        margin-top: 1;
    }
    #llm-selected-model {
        height: 1;
        width: 100%;
        color: #7ee7ff;
    }
    #llm-base-url-input, #llm-api-key-input {
        height: 3;
        width: 100%;
        background: #0e0d12;
        border: round #2a2735;
        color: #e8e3f0;
        padding: 0 1;
    }
    #llm-base-url-input:focus, #llm-api-key-input:focus {
        border: round #ff9bc7;
    }
    #llm-connect-btn {
        height: 3;
        width: 100%;
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
        width: 100%;
        color: #8a849a;
        margin-top: 1;
    }
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

        with VerticalScroll(id="llm-model-scroll"):
            current_provider = None
            for model in PROVIDER_MODELS:
                if model.provider != current_provider:
                    current_provider = model.provider
                    yield ProviderLabel(current_provider)
                yield ModelButton(
                    model,
                    selected=(model.name == self.selected_model.name),
                    id=f"model-{model.name.replace('/', '-').replace('.', '-')}",
                )

        with Widget(id="llm-config"):
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
                placeholder="paste api key and press enter",
                id="llm-api-key-input",
            )

            yield Button("CONNECT  ↵", id="llm-connect-btn")
            yield Static(self._status_text(), id="llm-status")

    def _status_text(self) -> str:
        provider = self.selected_model.provider
        if self.connection_status == "connected":
            return f"[#7ef7c0]● connected · {provider}[/]"
        if self.connection_status == "error":
            return "[#ff6b8a]× failed · check key/url[/]"
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
            # Update selected marker on model buttons.
            for btn in self.query(ModelButton):
                if btn.model.name == self.selected_model.name:
                    btn.add_class("-selected")
                else:
                    btn.remove_class("-selected")
        except Exception:
            pass

    @on(Button.Pressed)
    def _on_button_pressed(self, event: Button.Pressed) -> None:
        # Model row click
        if isinstance(event.button, ModelButton):
            self.selected_model = event.button.model
            self.config.selected_model = event.button.model.name
            self.connection_status = "disconnected"
            self._refresh()
            return
        # Connect button
        if event.button.id == "llm-connect-btn":
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