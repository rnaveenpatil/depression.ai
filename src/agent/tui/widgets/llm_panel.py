"""Compact OpenCode-inspired LLM switcher for the right workspace sidebar."""
from __future__ import annotations
from typing import Callable, Optional
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Input, ListView, ListItem, Label
from textual.reactive import reactive
from textual import on
from agent.tui.llm_providers import PROVIDER_MODELS, PROVIDER_INFO, LLMModel, get_llm_config


class ModelItem(ListItem):
    DEFAULT_CSS = """
    ModelItem { height: 3; padding: 0 1; background: transparent; }
    ModelItem:hover { background: $bg-hover; }
    ModelItem.selected { background: $bg-selected; border-left: tall $secondary; }
    .model-name { color: $text; text-style: bold; height: 1; }
    .model-meta { color: $text-muted; height: 1; }
    """
    def __init__(self, model: LLMModel, is_selected: bool = False, **kwargs):
        super().__init__(**kwargs); self.model = model
        if is_selected: self.add_class("selected")
    def compose(self) -> ComposeResult:
        m = self.model
        ctx = f"{m.context_window // 1000}K" if m.context_window < 1_000_000 else f"{m.context_window // 1_000_000}M"
        icon = PROVIDER_INFO.get(m.provider, {}).get("icon", "·")
        yield Label(m.display_name, classes="model-name")
        yield Label(f"{icon} {m.provider}  ·  {ctx}  ·  {m.stars}", classes="model-meta")


class ProviderHeader(ListItem):
    DEFAULT_CSS = """
    ProviderHeader { height: 1; padding: 0 1; background: transparent; }
    .provider-label { color: $text-muted; text-style: bold; }
    """
    def __init__(self, provider: str, **kwargs):
        super().__init__(**kwargs); self.provider = provider
    def compose(self) -> ComposeResult:
        info = PROVIDER_INFO.get(self.provider, {})
        yield Label(f"{info.get('icon', '·')} {info.get('name', self.provider)}", classes="provider-label")


class LLMProviderPanel(Widget):
    """Model picker + endpoint + persistent API-key connection panel."""
    DEFAULT_CSS = """
    LLMProviderPanel { height: 100%; width: 100%; layout: vertical; }
    #llm-header { height: 2; padding: 0 1; color: $primary; text-style: bold; border-bottom: solid $border; }
    #llm-model-list { height: 1fr; }
    #llm-config-area { height: 13; padding: 1; border-top: solid $border; background: $bg-panel; }
    .llm-label { height: 1; color: $text-muted; }
    #llm-base-url-input, #llm-api-key-input { height: 3; background: $input-bg; border: round $input-border; color: $text; padding: 0 1; }
    #llm-base-url-input:focus, #llm-api-key-input:focus { border: round $input-focus; }
    #llm-connect-btn { height: 3; margin-top: 1; background: transparent; border: round $secondary; color: $secondary; text-style: bold; }
    #llm-connect-btn:hover { background: $bg-hover; color: $text-bright; }
    #llm-status { height: 1; margin-top: 1; color: $text-muted; }
    .status-connected { color: $success; }
    .status-error { color: $error; }
    """
    selected_model = reactive[Optional[LLMModel]](None)
    connection_status = reactive[str]("disconnected")

    def __init__(self, on_connect: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs); self.config = get_llm_config(); self._on_connect = on_connect
        self.selected_model = self.config.get_selected_model_info() or PROVIDER_MODELS[0]

    def compose(self) -> ComposeResult:
        yield Static("LLM  /  CONNECT", id="llm-header")
        with ListView(id="llm-model-list"):
            current = None
            for model in PROVIDER_MODELS:
                if model.provider != current:
                    current = model.provider; yield ProviderHeader(current)
                yield ModelItem(model, model.name == self.selected_model.name)
        with Widget(id="llm-config-area"):
            yield Static("MODEL", classes="llm-label")
            yield Static(self.selected_model.display_name, id="llm-selected-model", classes="llm-label")
            yield Static("BASE URL", classes="llm-label")
            yield Input(value=self.config.get_base_url(self.selected_model.provider), id="llm-base-url-input", placeholder="https://api.example.com/v1")
            yield Static("API KEY", classes="llm-label")
            yield Input(value="" if not self.config.get_api_key(self.selected_model.provider) else "••••••••", password=True, id="llm-api-key-input", placeholder="Paste API key")
            yield Button("CONNECT", id="llm-connect-btn")
            yield Static(self._status_text(), id="llm-status")

    def _status_text(self) -> str:
        p = self.selected_model.provider
        if self.connection_status == "connected": return f"[status-connected]● connected · {p}[/]"
        if self.connection_status == "error": return "[status-error]× connection failed[/]"
        return f"○ {p} · key {'saved' if self.config.get_api_key(p) else 'required'}"

    def _refresh(self) -> None:
        try:
            self.query_one("#llm-selected-model", Static).update(self.selected_model.display_name)
            self.query_one("#llm-base-url-input", Input).value = self.config.get_base_url(self.selected_model.provider)
            self.query_one("#llm-api-key-input", Input).value = "" if not self.config.get_api_key(self.selected_model.provider) else "••••••••"
            self.query_one("#llm-status", Static).update(self._status_text())
        except Exception: pass

    @on(ListView.Selected, "#llm-model-list")
    def _on_model_selected(self, event: ListView.Selected) -> None:
        item = self.query_one("#llm-model-list", ListView).get_item_at_index(event.index)
        if isinstance(item, ModelItem):
            self.selected_model = item.model; self.config.selected_model = item.model.name; self.connection_status = "disconnected"; self._refresh()

    @on(Button.Pressed, "#llm-connect-btn")
    def _on_connect(self) -> None:
        provider = self.selected_model.provider
        base_input = self.query_one("#llm-base-url-input", Input)
        key_input = self.query_one("#llm-api-key-input", Input)
        key = self.config.get_api_key(provider) if key_input.value == "••••••••" else key_input.value.strip()
        base_url = base_input.value.strip().rstrip("/")
        if not key:
            self.connection_status = "error"; self._refresh(); return
        self.config.set_base_url(provider, base_url)
        self.config.set_api_key(provider, key)
        self.config.selected_model = self.selected_model.name
        self.connection_status = "connected"; self._refresh()
        if self._on_connect: self._on_connect(self.selected_model)

    def get_current_config(self) -> dict:
        return self.config.build_llm_config(self.selected_model)

    def set_connected(self, connected: bool) -> None:
        self.connection_status = "connected" if connected else "disconnected"; self._refresh()

    def get_selected_model(self) -> LLMModel:
        return self.selected_model
