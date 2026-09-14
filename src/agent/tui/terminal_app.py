"""Terminal-first Depression.AI TUI.

The existing agent lifecycle is reused, but presentation is deliberately
terminal-native: transcript + prompt are the whole workspace. Model/connection
configuration appears only as a temporary command screen.
"""
from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Widget
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static
from textual import on

from agent.tui.app import DepressionTUI
from agent.tui.theme import apply_opencode_theme
from agent.tui.views.chat import ChatView
from agent.tui.views.header import HeaderBar
from agent.tui.views.status_bar import StatusBar
from agent.tui.llm_providers import LLMModel, PROVIDER_MODELS, get_llm_config


class ModelPicker(ModalScreen[Optional[LLMModel]]):
    """Small keyboard-first model picker; no permanent model panel."""

    DEFAULT_CSS = """
    ModelPicker { align: center middle; background: #000000 70%; }
    #model-box { width: 72; max-width: 92%; height: 80%; max-height: 34; background: #0b0b0b; border: solid #242424; padding: 1 2; }
    #model-title { height: 2; color: #75f0a8; text-style: bold; }
    #model-hint { height: 1; color: #777777; margin-bottom: 1; }
    #model-search { height: 3; background: #000000; border: solid #242424; }
    #model-list { height: 1fr; margin-top: 1; background: #0b0b0b; }
    .model-row { height: 2; padding: 0 1; }
    .model-row:hover { background: #181818; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = list(PROVIDER_MODELS)
        self.filtered = list(self.models)

    def compose(self) -> ComposeResult:
        with Vertical(id="model-box"):
            yield Static("SELECT MODEL", id="model-title")
            yield Static("↑↓ navigate   Enter select   Esc close   /models", id="model-hint")
            yield Input(placeholder="Filter models…", id="model-search")
            yield ListView(id="model-list")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#model-search", Input).focus()

    def _refresh_list(self) -> None:
        view = self.query_one("#model-list", ListView)
        view.clear()
        last_provider = None
        for model in self.filtered:
            if model.provider != last_provider:
                view.append(ListItem(Label(f"── {model.provider.upper()} ──"), classes="model-row"))
                last_provider = model.provider
            view.append(ListItem(Label(f"  {model.display_name:<34} {model.context_window:,}"), classes="model-row"))

    @on(Input.Changed, "#model-search")
    def _filter(self, event: Input.Changed) -> None:
        query = event.value.lower().strip()
        self.filtered = [m for m in self.models if query in m.display_name.lower() or query in m.provider.lower() or query in m.name.lower()]
        self._refresh_list()

    @on(ListView.Selected, "#model-list")
    def _selected(self, event: ListView.Selected) -> None:
        label = event.item.query_one(Label).renderable
        for model in self.filtered:
            if model.display_name in str(label):
                self.dismiss(model)
                return

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class ConnectScreen(ModalScreen[Optional[tuple[str, str]]]):
    """Temporary LLM connection form."""

    DEFAULT_CSS = """
    ConnectScreen { align: center middle; background: #000000 70%; }
    #connect-box { width: 70; max-width: 92%; height: auto; background: #0b0b0b; border: solid #242424; padding: 1 2; }
    #connect-title { height: 2; color: #75f0a8; text-style: bold; }
    .connect-label { height: 1; color: #777777; margin-top: 1; }
    .connect-value { height: 3; color: #e5e5e5; }
    .connect-input { height: 3; background: #000000; border: solid #242424; }
    #connect-hint { height: 2; color: #777777; margin-top: 1; }
    """

    def __init__(self, model: LLMModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.cfg = get_llm_config()

    def compose(self) -> ComposeResult:
        with Vertical(id="connect-box"):
            yield Static("CONNECT LLM", id="connect-title")
            yield Static("Provider", classes="connect-label")
            yield Static(self.model.provider, classes="connect-value")
            yield Static("Model", classes="connect-label")
            yield Static(self.model.display_name, classes="connect-value")
            yield Static("API key", classes="connect-label")
            yield Input(value=self.cfg.get_api_key(self.model.provider), password=True, placeholder="Paste API key", id="api-key", classes="connect-input")
            yield Static("Base URL", classes="connect-label")
            yield Input(value=self.cfg.get_base_url(self.model.provider), placeholder="https://…", id="base-url", classes="connect-input")
            yield Static("Enter connect   Esc cancel", id="connect-hint")

    def on_mount(self) -> None:
        self.query_one("#api-key", Input).focus()

    @on(Input.Submitted, "#api-key")
    def _api_key_submitted(self, event: Input.Submitted) -> None:
        self.query_one("#base-url", Input).focus()

    @on(Input.Submitted, "#base-url")
    def _connect(self, event: Input.Submitted) -> None:
        key = self.query_one("#api-key", Input).value.strip()
        url = self.query_one("#base-url", Input).value.strip()
        if not key:
            self.query_one("#api-key", Input).focus()
            return
        self.dismiss((key, url))

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class CommandMenu(ModalScreen[Optional[str]]):
    """Minimal Ctrl+P palette, intentionally styled like a terminal command menu."""

    DEFAULT_CSS = """
    CommandMenu { align: center middle; background: #000000 60%; }
    #command-box { width: 62; max-width: 90%; height: auto; background: #0b0b0b; border: solid #242424; padding: 1 2; }
    #command-input { height: 3; background: #000000; border: solid #242424; }
    #command-list { height: auto; max-height: 16; background: #0b0b0b; margin-top: 1; }
    .command-row { height: 2; padding: 0 1; }
    .command-row:hover { background: #181818; }
    """

    COMMANDS = [
        ("/models", "Select LLM"), ("/connect", "Connect current model"),
        ("/plan", "Plan mode"), ("/build", "Build mode"), ("/auto", "Auto mode"),
        ("/tools", "Show tools"), ("/sessions", "Show sessions"),
        ("/status", "Show status"), ("/clear", "Clear transcript"),
        ("/help", "Keyboard help"), ("/quit", "Exit"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filtered = list(self.COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical(id="command-box"):
            yield Input(placeholder="Type a command…", id="command-input")
            yield ListView(id="command-list")

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#command-input", Input).focus()

    def _refresh(self) -> None:
        view = self.query_one("#command-list", ListView)
        view.clear()
        for command, description in self.filtered:
            view.append(ListItem(Label(f"{command:<14} {description}"), classes="command-row"))

    @on(Input.Changed, "#command-input")
    def _filter(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        self.filtered = [x for x in self.COMMANDS if q in x[0].lower() or q in x[1].lower()]
        self._refresh()

    @on(Input.Submitted, "#command-input")
    def _submit(self, event: Input.Submitted) -> None:
        if self.filtered:
            self.dismiss(self.filtered[0][0])

    @on(ListView.Selected, "#command-list")
    def _selected(self, event: ListView.Selected) -> None:
        index = list(self.query_one("#command-list", ListView).children).index(event.item)
        if 0 <= index < len(self.filtered):
            self.dismiss(self.filtered[index][0])

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class TerminalDepressionTUI(DepressionTUI):
    """OpenCode-like terminal shell around the existing agent runtime."""

    BINDINGS = [
        Binding("ctrl+p", "command_menu", "Commands", show=False),
        Binding("f2", "model_picker", "Models", show=False),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
        Binding("ctrl+d", "quit", "Exit", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("tab", "switch_mode", "Mode", show=False),
        Binding("f1", "show_help", "Help", show=False),
    ]

    def __init__(self, *args, **kwargs):
        kwargs["no_sidebar"] = True
        super().__init__(*args, **kwargs)
        self.sidebar_visible = False

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Widget(id="main-container"):
            yield ChatView(id="chat-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        apply_opencode_theme(self)
        self.title = "DEPRESSION.AI"
        self.sub_title = "TERMINAL AGENT"
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system("**DEPRESSION.AI**  ·  terminal agent\n\nType a task.  `/models` selects a model · `/connect` configures it · `!` is reserved for shell commands.")
        chat.add_divider()
        self.query_one("#status-bar", StatusBar).update_all(status="starting", model=self.model_name, tokens=0, cost=0.0)
        # Reuse the proven agent initialization from the base app, but do not
        # mount its dashboard/sidebar.
        self._init_task = self.run_worker(self._init_agent_async(), exclusive=True, group="agent-init", thread=False)
        self.set_interval(0.2, self._focus_prompt)

    def _focus_prompt(self) -> None:
        try:
            self.query_one("#chat-input", Input).focus()
        except Exception:
            pass

    async def action_command_menu(self) -> None:
        command = await self.push_screen_wait(CommandMenu())
        if command:
            await self._execute_terminal_command(command)

    async def action_model_picker(self) -> None:
        model = await self.push_screen_wait(ModelPicker())
        if model:
            await self._choose_model(model)

    async def action_show_help(self) -> None:
        self.query_one("#chat-panel", ChatView).add_system(
            "**Commands**  `/models` `/connect` `/plan` `/build` `/auto` `/tools` `/sessions` `/status` `/clear` `/help` `/quit`\n\n"
            "**Keys**  Ctrl+P command menu · F2 model picker · Tab cycle mode · Ctrl+L clear · Ctrl+D exit"
        )

    async def _choose_model(self, model: LLMModel) -> None:
        self.model_name = f"{model.provider}/{model.short_name}"
        cfg = get_llm_config()
        cfg.selected_model = model.name
        self.query_one("#header", HeaderBar).set_model(self.model_name)
        self.query_one("#status-bar", StatusBar).set_model(self.model_name)
        key = cfg.get_api_key(model.provider)
        if key:
            self._apply_llm_to_coordinator(model)
            self.query_one("#chat-panel", ChatView).add_system(f"✓ Selected **{model.display_name}**")
        else:
            self.query_one("#chat-panel", ChatView).add_system(f"Selected **{model.display_name}**. Run `/connect` to add its API key.")

    async def _connect_current_model(self) -> None:
        model = get_llm_config().get_selected_model_info()
        if model is None:
            model = await self.push_screen_wait(ModelPicker())
            if model is None:
                return
        result = await self.push_screen_wait(ConnectScreen(model))
        if not result:
            return
        key, url = result
        cfg = get_llm_config()
        cfg.set_api_key(model.provider, key)
        cfg.set_base_url(model.provider, url)
        cfg.selected_model = model.name
        self.model_name = f"{model.provider}/{model.short_name}"
        self.query_one("#header", HeaderBar).set_model(self.model_name)
        self.query_one("#status-bar", StatusBar).set_model(self.model_name)
        if self.coordinator:
            self._apply_llm_to_coordinator(model)
        self.query_one("#chat-panel", ChatView).add_system(f"✓ Connected **{model.display_name}**")

    async def _execute_terminal_command(self, command: str) -> None:
        chat = self.query_one("#chat-panel", ChatView)
        if command == "/models":
            await self.action_model_picker()
        elif command == "/connect":
            await self._connect_current_model()
        elif command in ("/plan", "/build", "/auto"):
            self.current_mode = command[1:]
            self.query_one("#header", HeaderBar).set_mode(self.current_mode)
            if self.coordinator:
                try:
                    self.coordinator.set_mode(self.current_mode)
                except Exception:
                    pass
            chat.add_system(f"Switched to **{self.current_mode.upper()}** mode")
        elif command == "/status":
            chat.add_system(f"Mode: {self.current_mode} · Model: {self.model_name} · Tokens: {self.tokens_used:,} · Cost: ${self.cost:.4f}")
        elif command == "/tools":
            chat.add_system("Tools: terminal · filesystem · git · search · web · patch · browser · task · diagnostics")
        elif command == "/sessions":
            chat.add_system(f"Session: `{self.session_id}`")
        elif command == "/clear":
            chat.clear_messages()
        elif command == "/help":
            await self.action_show_help()
        elif command == "/quit":
            await self.action_quit()

    def _send_message(self) -> None:
        chat = self.query_one("#chat-panel", ChatView)
        text = chat.get_input_text().strip()
        if not text:
            return
        if text.startswith("/"):
            chat.clear_input()
            self.run_worker(self._execute_terminal_command(text.split()[0].lower()), exclusive=False)
            return
        super()._send_message()
