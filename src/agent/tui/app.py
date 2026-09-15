"""depression.ai — matrix green terminal agent with LLM + AWS connect panels."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Static, Input, Markdown, Button, Select
from textual import on, work

from agent.utils.env_manager import (
    load_env_file,
    write_env_file,
    set_provider_credentials,
    get_provider_credentials,
    set_aws_credentials,
    get_aws_credentials,
    set_selected_model,
    get_selected_model,
)


# ── Matrix green palette ──────────────────────────────────────────────────
GREEN       = "#00ff66"
GREEN_DIM   = "#00aa44"
GREEN_FAINT = "#005522"
GREEN_GLOW  = "#88ffbb"
AMBER       = "#ffcc44"
ERROR       = "#ff4466"
TEXT        = "#aaffcc"
MUTED       = "#3d8c5c"
DIM         = "#1a5c33"
BG          = "#000000"
PANEL       = "#031008"
RAISED      = "#061a0f"
BORDER      = "#0a3d20"


THEME = Theme(
    name="matrix",
    primary=GREEN, secondary=GREEN_DIM, accent=GREEN_GLOW,
    warning=AMBER, error=ERROR, success=GREEN,
    surface=PANEL, panel=RAISED, boost="#0d2a17",
    foreground=TEXT, background=BG, dark=True,
)


BANNER_LINES = [
    "  ██████╗ ███████╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗",
    "  ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║",
    "  ██║  ██║█████╗  ██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║",
    "  ██║  ██║██╔══╝  ██╔═══╝ ██╔═══╝ ██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║",
    "  ██████╔╝███████╗██║     ██║     ███████╗███████║███████║██║╚██████╔╝██║ ╚████║",
    "  ╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
]


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
MODE_ICONS     = {"Build": "◆", "Plan": "◇", "Auto": "⟡"}
MODE_ORDER     = ["Build", "Plan", "Auto"]


PROVIDERS = [
    ("NVIDIA",     "https://integrate.api.nvidia.com/v1",
     ["Nemotron 3 Ultra 550B", "Nemotron 3 Super 120B",
      "Nemotron 3.5 Lightning 30B", "Nemotron 3 Nano 30B"]),
    ("DeepSeek",   "https://api.deepseek.com",
     ["DeepSeek V4 Pro", "DeepSeek V4 Flash"]),
    ("MiniMax",    "https://api.minimax.io/v1", ["MiniMax M2.5"]),
    ("Mistral",    "https://api.mistral.ai/v1",
     ["Mistral Medium 3.5", "Devstral"]),
    ("Anthropic",  "https://api.anthropic.com",
     ["Claude Sonnet", "Claude Opus"]),
    ("Google",     "https://generativelanguage.googleapis.com/v1beta/openai",
     ["Gemini Flash", "Gemini Pro"]),
    ("OpenAI",     "https://api.openai.com/v1", ["GPT-5.x"]),
    ("Groq",       "https://api.groq.com/openai/v1",
     ["GPT-OSS 120B", "GPT-OSS 20B"]),
    ("Moonshot",   "https://api.moonshot.ai/v1",
     ["Kimi K2.x", "Kimi K3"]),
    ("Z.AI",       "https://api.z.ai/api/paas/v4", ["GLM-5.x"]),
    ("OpenRouter", "https://openrouter.ai/api/v1", ["OpenRouter Coding"]),
]

AWS_REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1",
    "ap-south-1", "ap-southeast-1", "ap-southeast-2",
    "ap-northeast-1", "sa-east-1", "ca-central-1",
]


# ── Config: LLM (uses .env file) ──────────────────────────────────────────
def load_config() -> dict:
    """Load LLM config from .env file."""
    env_vars = load_env_file()
    selected_model = get_selected_model() or ""
    provider, model = "", ""
    if selected_model and "/" in selected_model:
        provider, model = selected_model.split("/", 1)
    return {
        "provider": provider or env_vars.get("DEPRESSION_PROVIDER", ""),
        "model": model or env_vars.get("DEPRESSION_MODEL", ""),
        "base_url": env_vars.get("DEPRESSION_BASE_URL", ""),
        "api_key": env_vars.get("DEPRESSION_API_KEY", ""),
    }


def save_config(cfg: dict) -> None:
    """Save LLM config to .env file."""
    provider = cfg.get("provider", "")
    model = cfg.get("model", "")
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")
    
    if provider and model:
        set_selected_model(f"{provider}/{model}")
    
    if provider:
        set_provider_credentials(provider, api_key=api_key, base_url=base_url)
    
    env_vars = load_env_file()
    if provider:
        env_vars["DEPRESSION_PROVIDER"] = provider
    if model:
        env_vars["DEPRESSION_MODEL"] = model
    if base_url:
        env_vars["DEPRESSION_BASE_URL"] = base_url
    if api_key:
        env_vars["DEPRESSION_API_KEY"] = api_key
    write_env_file(env_vars)


def load_aws() -> dict:
    """Load AWS config from .env file."""
    creds = get_aws_credentials()
    return {
        "access_key": creds.get("access_key", ""),
        "secret_key": creds.get("secret_key", ""),
        "region": creds.get("region", "us-east-1"),
    }


def save_aws(cfg: dict) -> None:
    """Save AWS config to .env file."""
    set_aws_credentials(
        access_key=cfg.get("access_key"),
        secret_key=cfg.get("secret_key"),
        region=cfg.get("region", "us-east-1"),
    )


# ── Sidebar panels ────────────────────────────────────────────────────────
class LLMConnectPanel(Static):
    """Panel: paste base URL + API key for the selected model."""

    DEFAULT_CSS = f"""
    LLMConnectPanel {{
        width: 100%; height: auto;
        padding: 1 1;
        background: {BG};
    }}
    .panel-title {{
        height: 1; color: {GREEN}; text-style: bold;
        margin: 0 0 1 0;
    }}
    .panel-label {{
        height: 1; color: {MUTED}; text-style: bold;
        margin: 1 0 0 0;
    }}
    .panel-value {{
        height: 1; color: {GREEN_GLOW};
    }}
    .panel-input {{
        height: 3; width: 100%;
        background: {PANEL};
        border: round {BORDER};
        color: {TEXT};
        padding: 0 1;
    }}
    .panel-input:focus {{ border: round {GREEN}; }}
    .panel-btn {{
        height: 3; width: 100%;
        margin-top: 1;
        background: transparent;
        border: round {GREEN};
        color: {GREEN};
        text-style: bold;
    }}
    .panel-btn:hover, .panel-btn:focus {{
        background: {RAISED}; color: {GREEN_GLOW};
        border: round {GREEN_GLOW};
    }}
    .panel-status {{
        height: 1; color: {MUTED};
        margin-top: 1;
    }}
    .panel-hint {{
        height: auto; color: {DIM};
        margin-top: 1;
    }}
    """

    def __init__(self, app_ref, **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static("▌ CONNECT LLM", classes="panel-title")

        yield Static("Provider / Model", classes="panel-label")
        provider = self._app.selected_provider or "(pick from /models)"
        model = self._app.selected_model or ""
        yield Static(f"{provider}  {model}".strip(), classes="panel-value")

        yield Static("Base URL", classes="panel-label")
        yield Input(
            value=self._app.cfg.get("base_url", ""),
            placeholder="https://api.example.com/v1",
            id="llm-base-url",
            classes="panel-input",
        )

        yield Static("API Key", classes="panel-label")
        yield Input(
            value="••••••••" if self._app.cfg.get("api_key") else "",
            password=True,
            placeholder="paste your api key",
            id="llm-api-key",
            classes="panel-input",
        )

        yield Button("CONNECT  ↵", id="llm-connect-btn", classes="panel-btn")
        yield Static(self._status_text(), id="llm-status",
                     classes="panel-status")
        yield Static("Press Enter in the API key field to connect.\n"
                     "Config saved to ~/.config/depression/llm.json",
                     classes="panel-hint")

    def _status_text(self) -> str:
        if self._app.cfg.get("api_key"):
            return f"[{GREEN}]● connected · {self._app.cfg.get('provider','')}[/]"
        return f"[{MUTED}]○ not connected[/]"

    def refresh_values(self) -> None:
        try:
            self.query_one(".panel-value", Static).update(
                f"{self._app.selected_provider}  {self._app.selected_model}".strip()
                or "(pick from /models)"
            )
            self.query_one("#llm-base-url", Input).value = \
                self._app.cfg.get("base_url", "")
            self.query_one("#llm-api-key", Input).value = \
                "••••••••" if self._app.cfg.get("api_key") else ""
            self.query_one("#llm-status", Static).update(self._status_text())
        except Exception:
            pass

    @on(Button.Pressed, "#llm-connect-btn")
    def _btn(self) -> None:
        self._submit()

    @on(Input.Submitted, "#llm-api-key")
    def _enter_key(self, event: Input.Submitted) -> None:
        self._submit()

    @on(Input.Submitted, "#llm-base-url")
    def _enter_url(self, event: Input.Submitted) -> None:
        try:
            self.query_one("#llm-api-key", Input).focus()
        except Exception:
            pass

    def _submit(self) -> None:
        url = self.query_one("#llm-base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#llm-api-key", Input)
        key = (self._app.cfg.get("api_key", "")
               if key_input.value == "••••••••" else key_input.value.strip())

        if not url or not key:
            self._app._show_error("base URL and API key are required")
            return

        self._app.cfg.update({
            "provider": self._app.selected_provider,
            "model": self._app.selected_model,
            "base_url": url,
            "api_key": key,
        })
        save_config(self._app.cfg)

        # Update LLM registry with the new credentials
        try:
            from agent.llm.provider import get_llm_registry
            reg = get_llm_registry()
            reg.set_api_key(self._app.selected_provider.lower(), key)
            reg.set_model(self._app.selected_model)
            # Also set base URL if the provider supports it
            if hasattr(reg.providers.get(self._app.selected_provider.lower()), 'base_url'):
                reg.providers[self._app.selected_provider.lower()].base_url = url
        except Exception as e:
            self._app._show_error(f"Failed to update LLM registry: {e}")
            return

        self._app._refresh_mode_chip()
        self._app._show_system(f"connected · {self._app.selected_provider} · {self._app.selected_model}")
        self._app._show_system(f"base url · {url}")
        self.refresh_values()


class AWSConnectPanel(Static):
    """Panel: AWS access key / secret / region dropdown."""

    DEFAULT_CSS = f"""
    AWSConnectPanel {{
        width: 100%; height: auto;
        padding: 1 1;
        background: {BG};
    }}
    """

    def __init__(self, app_ref, **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref

    def compose(self) -> ComposeResult:
        yield Static("▌ AWS CREDENTIALS", classes="panel-title")

        yield Static("Access Key ID", classes="panel-label")
        yield Input(
            value=self._app.aws.get("access_key", ""),
            placeholder="AKIA…",
            id="aws-access-key",
            classes="panel-input",
        )

        yield Static("Secret Access Key", classes="panel-label")
        yield Input(
            value="••••••••••••" if self._app.aws.get("secret_key") else "",
            password=True,
            placeholder="paste your secret key",
            id="aws-secret-key",
            classes="panel-input",
        )

        yield Static("Region", classes="panel-label")
        yield Select(
            [(r, r) for r in AWS_REGIONS],
            value=self._app.aws.get("region", "us-east-1"),
            id="aws-region",
            allow_blank=False,
        )

        yield Button("SAVE  ↵", id="aws-save-btn", classes="panel-btn")
        yield Static(self._status_text(), id="aws-status",
                     classes="panel-status")
        yield Static("Saved to ~/.config/depression/aws.json",
                     classes="panel-hint")

    def _status_text(self) -> str:
        if self._app.aws.get("access_key") and self._app.aws.get("secret_key"):
            return f"[{GREEN}]● saved · {self._app.aws.get('region','us-east-1')}[/]"
        return f"[{MUTED}]○ no credentials[/]"

    def refresh_values(self) -> None:
        try:
            self.query_one("#aws-access-key", Input).value = \
                self._app.aws.get("access_key", "")
            self.query_one("#aws-secret-key", Input).value = \
                "••••••••••••" if self._app.aws.get("secret_key") else ""
            self.query_one("#aws-region", Select).value = \
                self._app.aws.get("region", "us-east-1")
            self.query_one("#aws-status", Static).update(self._status_text())
        except Exception:
            pass

    @on(Button.Pressed, "#aws-save-btn")
    def _btn(self) -> None:
        self._submit()

    @on(Input.Submitted, "#aws-secret-key")
    def _enter_secret(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        access = self.query_one("#aws-access-key", Input).value.strip()
        secret_input = self.query_one("#aws-secret-key", Input)
        secret = (self._app.aws.get("secret_key", "")
                  if secret_input.value == "••••••••••••"
                  else secret_input.value.strip())
        region = self.query_one("#aws-region", Select).value or "us-east-1"

        if not access or not secret:
            self._app._show_error("AWS access key and secret are required")
            return

        self._app.aws.update({
            "access_key": access,
            "secret_key": secret,
            "region": region,
        })
        save_aws(self._app.aws)

        # Push into environment for boto3 to pick up automatically.
        os.environ["AWS_ACCESS_KEY_ID"] = access
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        os.environ["AWS_DEFAULT_REGION"] = region

        self._app._show_system(f"AWS credentials saved · region {region}")
        self.refresh_values()


# ── Sidebar container ─────────────────────────────────────────────────────
class Sidebar(Static):
    """Right rail that swaps between panels."""

    DEFAULT_CSS = f"""
    Sidebar {{
        width: 46; min-width: 40; height: 1fr;
        background: {BG};
        border-left: solid {BORDER};
        layout: vertical;
        padding: 0;
    }}
    #sidebar-title {{
        height: 1; padding: 0 1;
        background: {BG};
        color: {GREEN};
        text-style: bold;
        border-bottom: solid {BORDER};
    }}
    #sidebar-tabs {{
        height: 1; layout: horizontal;
        background: {BG};
        border-bottom: solid {BORDER};
    }}
    .side-tab {{
        width: 1fr; height: 1;
        background: transparent; border: none;
        color: {DIM}; text-style: bold;
    }}
    .side-tab:hover {{ color: {MUTED}; }}
    .side-tab.active {{ color: {GREEN}; }}
    #sidebar-content {{
        height: 1fr;
        overflow-y: auto;
        background: {BG};
    }}
    """

    def __init__(self, app_ref, **kwargs):
        super().__init__(**kwargs)
        self._app = app_ref
        self.active_tab = "llm"

    def compose(self) -> ComposeResult:
        yield Static("▌ SIDEBAR", id="sidebar-title")
        with Horizontal(id="sidebar-tabs"):
            yield Button("CONNECT", id="tab-llm", classes="side-tab active")
            yield Button("AWS",     id="tab-aws", classes="side-tab")
            yield Button("HELP",    id="tab-help", classes="side-tab")
        with VerticalScroll(id="sidebar-content"):
            yield LLMConnectPanel(self._app, id="panel-llm")
            yield AWSConnectPanel(self._app, id="panel-aws")
            yield HelpPanel(id="panel-help")

    def on_mount(self) -> None:
        self._show("llm")

    @on(Button.Pressed, ".side-tab")
    def _tab(self, event: Button.Pressed) -> None:
        self._show(event.button.id.replace("tab-", ""))

    def _show(self, tab: str) -> None:
        self.active_tab = tab
        for b in self.query(".side-tab"):
            b.remove_class("active")
        try:
            self.query_one(f"#tab-{tab}", Button).add_class("active")
        except Exception:
            pass
        content = self.query_one("#sidebar-content")
        for child in content.children:
            child.display = False
        try:
            self.query_one(f"#panel-{tab}").display = True
        except Exception:
            pass


class HelpPanel(Static):
    DEFAULT_CSS = f"""
    HelpPanel {{
        width: 100%; height: auto;
        padding: 1 1;
        background: {BG};
    }}
    .help-section {{
        height: 1; color: {GREEN}; text-style: bold;
        margin: 1 0 0 0;
    }}
    .help-row {{ height: 1; color: {TEXT}; }}
    .help-key {{ color: {GREEN_GLOW}; }}
    """

    def compose(self) -> ComposeResult:
        yield Static("▌ HELP", classes="panel-title")
        yield Static("Commands", classes="help-section")
        yield Static("/models   pick a model", classes="help-row")
        yield Static("/connect  open LLM panel", classes="help-row")
        yield Static("/aws      open AWS panel", classes="help-row")
        yield Static("/clear    clear transcript", classes="help-row")
        yield Static("/help     this panel", classes="help-row")
        yield Static("/quit     exit", classes="help-row")
        yield Static("Keys", classes="help-section")
        yield Static("Tab       cycle mode", classes="help-row")
        yield Static("Ctrl+P    model picker", classes="help-row")
        yield Static("Ctrl+C    exit", classes="help-row")
        yield Static("Ctrl+L    clear transcript", classes="help-row")


# ── Model picker overlay ──────────────────────────────────────────────────
class ModelPickerModal(Static):
    DEFAULT_CSS = f"""
    ModelPickerModal {{
        display: none;
        width: 100%; height: 100%;
        background: {BG} 96%;
        color: {TEXT};
        padding: 2 4;
        overflow-y: auto;
        layer: overlay;
    }}
    ModelPickerModal.-open {{ display: block; }}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._flat: List[Tuple[str, str]] = []
        self._lines: List[str] = []
        self._row_line: List[int] = []
        self._lines.append(f"[bold {GREEN}]  select a model[/]")
        self._lines.append("")
        for provider, _url, models in PROVIDERS:
            self._lines.append(f"[bold {GREEN_DIM}]── {provider}[/]")
            for m in models:
                self._row_line.append(len(self._lines))
                self._flat.append((provider, m))
                self._lines.append(f"  [{TEXT}]{m}[/]")
        self._lines.append("")
        self._lines.append(f"[{DIM}]click a model  ·  esc to close[/]")

    def render(self) -> str:
        return "\n".join(self._lines)

    def on_click(self, event: Click) -> None:
        idx = event.y - 2
        if idx in self._row_line:
            i = self._row_line.index(idx)
            provider, model = self._flat[i]
            app = self.app
            if hasattr(app, "_select_model"):
                app._select_model(provider, model)
            if hasattr(app, "_close_model_picker"):
                app._close_model_picker()


# ── App ───────────────────────────────────────────────────────────────────
class DepressionApp(App):
    TITLE = "depression.ai"

    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}

    #topbar {{
        height: 1; background: {BG}; color: {DIM};
        padding: 0 2; dock: top;
    }}
    #topbar-left {{ width: 1fr; color: {GREEN_DIM}; }}
    #topbar-right {{ width: auto; color: {DIM}; }}

    #body {{ height: 1fr; layout: horizontal; }}
    #main-col {{ width: 1fr; height: 1fr; layout: vertical; }}

    #hero {{
        height: auto; width: 100%;
        padding: 1 2 0 2;
        layout: vertical;
    }}
    #banner {{ height: 7; width: 100%; }}
    #banner-sub {{
        height: 1; color: {GREEN_DIM};
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }}

    #transcript {{
        height: 1fr; background: {BG}; color: {TEXT};
        padding: 1 2 0 2;
        scrollbar-background: {BG};
        scrollbar-color: {BORDER};
    }}
    .msg-user       {{ height: auto; color: {TEXT}; margin: 0 0 1 0; }}
    .msg-agent-head {{ height: 1; color: {GREEN}; text-style: bold; }}
    .msg-agent-body {{ height: auto; margin: 0 0 1 0; }}
    .msg-sys        {{ height: auto; color: {MUTED}; margin: 0 0 1 0; }}
    .msg-err        {{ height: auto; color: {ERROR}; margin: 0 0 1 0; }}

    #prompt-wrap {{
        dock: bottom;
        height: auto; min-height: 5;
        background: {BG};
        border-left: thick {GREEN};
        padding: 1 0 0 2;
        margin: 0 0 0 2;
    }}
    #prompt-row  {{ height: 3; layout: horizontal; background: {BG}; }}
    #prompt-sign {{
        width: 2; height: 3;
        color: {GREEN}; content-align: left middle;
    }}
    #prompt-input {{
        width: 1fr; height: 3;
        background: {BG}; border: none;
        color: {TEXT}; padding: 0;
    }}
    #prompt-input:focus {{ border: none; }}
    #mode-chip {{ height: 1; padding: 0 0 0 2; color: {MUTED}; background: {BG}; }}
    #shortcut-hint {{
        height: 1; padding: 0 0 0 2;
        color: {DIM}; background: {BG}; margin-bottom: 1;
    }}

    #footer {{
        height: 1; background: {BG}; color: {DIM};
        padding: 0 2; dock: bottom;
    }}
    #footer-left {{ width: 1fr; color: {MUTED}; }}
    #footer-right {{ width: auto; color: {MUTED}; }}

    Button {{ background: transparent; border: none; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",   "Quit",   priority=True),
        Binding("ctrl+d", "quit",   "Quit",   priority=True),
        Binding("ctrl+c", "cancel", "Cancel", priority=True),
        Binding("ctrl+l", "clear",  "Clear"),
        Binding("ctrl+p", "open_picker", "Picker"),
        Binding("tab",    "cycle_mode",  "Mode",  priority=True),
        Binding("escape", "escape", "Back"),
    ]

    current_mode: reactive[str] = reactive("Build")

    def __init__(self, coordinator=None, **kwargs):
        super().__init__(**kwargs)
        self.coordinator = coordinator
        self.cfg = load_config()
        self.aws = load_aws()
        self.selected_provider: str = self.cfg.get("provider", "")
        self.selected_model: str = self.cfg.get("model", "")
        self._picker_open = False
        self._spinner_on = False
        self._spinner_idx = 0
        self._banner_glow = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("depression.ai", id="topbar-left")
            yield Static("", id="topbar-right")

        with Horizontal(id="body"):
            with Vertical(id="main-col"):
                with Vertical(id="hero"):
                    yield Static(self._banner_markup(), id="banner")
                    yield Static(
                        f"  [{GREEN_DIM}]✧  a  t  e  r  m  i  n  a  l  "
                        f"c  o  d  i  n  g  a  g  e  n  t  ✧[/]",
                        id="banner-sub",
                    )

                yield VerticalScroll(id="transcript")

                with Vertical(id="prompt-wrap"):
                    with Horizontal(id="prompt-row"):
                        yield Static("›", id="prompt-sign")
                        yield Input(
                            placeholder='ask anything…  "fix broken tests"',
                            id="prompt-input",
                        )
                    yield Static(self._mode_chip(), id="mode-chip")
                    yield Static("tab agents    ctrl+p commands    /connect /aws /models",
                                 id="shortcut-hint")

            yield Sidebar(self, id="sidebar")

        with Horizontal(id="footer"):
            yield Static(self._footer_left(), id="footer-left")
            yield Static("depression.ai 0.1.0", id="footer-right")

        yield ModelPickerModal(id="picker-overlay")

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "matrix"
        self.query_one("#prompt-input", Input).focus()
        self.set_interval(0.10, self._tick_spinner)
        self.set_interval(1.50, self._tick_banner)

    # ── markup ────────────────────────────────────────────────────────
    def _banner_markup(self) -> str:
        greens = [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM]
        return "\n".join(f"[bold {greens[i]}]{line}[/]"
                         for i, line in enumerate(BANNER_LINES))

    def _mode_chip(self) -> str:
        icon = MODE_ICONS.get(self.current_mode, "◆")
        model = (f"{self.selected_provider} {self.selected_model}"
                 if self.selected_provider else "no model selected")
        prefix = ""
        if self._spinner_on:
            spin = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            prefix = f"[{AMBER}]{spin}[/] "
        return (f"{prefix}[{GREEN}]{icon} {self.current_mode}[/]  "
                f"[{DIM}]·[/]  [{TEXT}]{model}[/]")

    def _refresh_mode_chip(self) -> None:
        try:
            self.query_one("#mode-chip", Static).update(self._mode_chip())
        except Exception:
            pass

    def _session_head(self) -> str:
        from datetime import datetime
        return f"New session — {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"

    def _footer_left(self) -> str:
        try:
            cwd = os.getcwd()
        except Exception:
            cwd = "."
        return f"{cwd}:main"

    # ── animations ────────────────────────────────────────────────────
    def _tick_spinner(self) -> None:
        if not self._spinner_on:
            return
        self._spinner_idx += 1
        self._refresh_mode_chip()

    def _tick_banner(self) -> None:
        self._banner_glow = (self._banner_glow + 1) % 3
        greens = [
            [GREEN_GLOW, GREEN, GREEN, GREEN, GREEN_DIM, GREEN_DIM],
            [GREEN, GREEN_GLOW, GREEN, GREEN_DIM, GREEN, GREEN_DIM],
            [GREEN, GREEN, GREEN_GLOW, GREEN_DIM, GREEN_DIM, GREEN],
        ][self._banner_glow]
        rows = "\n".join(f"[bold {greens[i]}]{line}[/]"
                         for i, line in enumerate(BANNER_LINES))
        try:
            self.query_one("#banner", Static).update(rows)
        except Exception:
            pass

    # ── sidebar switching ─────────────────────────────────────────────
    def show_sidebar(self, panel: str) -> None:
        try:
            self.query_one("#sidebar", Sidebar)._show(panel)
        except Exception:
            pass

    # ── prompt submission ─────────────────────────────────────────────
    @on(Input.Submitted, "#prompt-input")
    def _on_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#prompt-input", Input).value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_slash(text)
            return
        self._show_user(text)
        self._respond(text)

    def _handle_slash(self, text: str) -> None:
        cmd = text.split()[0].lower()
        if cmd in ("/models", "/model"):
            self.action_open_picker()
        elif cmd == "/connect":
            self.show_sidebar("llm")
        elif cmd == "/aws":
            self.show_sidebar("aws")
        elif cmd == "/help":
            self.show_sidebar("help")
        elif cmd == "/clear":
            self.action_clear()
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            self._show_error(f"unknown command: {cmd}")

    # ── transcript ────────────────────────────────────────────────────
    def _show_user(self, text: str) -> None:
        t = self.query_one("#transcript", VerticalScroll)
        t.mount(Static(f"[bold {GREEN}]❯[/] [bold {TEXT}]{text}[/]",
                       classes="msg-user"))
        t.scroll_end(animate=False)

    def _show_system(self, text: str) -> None:
        t = self.query_one("#transcript", VerticalScroll)
        t.mount(Static(f"[{MUTED}]· {text}[/]", classes="msg-sys"))
        t.scroll_end(animate=False)

    def _show_error(self, text: str) -> None:
        t = self.query_one("#transcript", VerticalScroll)
        t.mount(Static(f"[bold {ERROR}]× {text}[/]", classes="msg-err"))
        t.scroll_end(animate=False)

    def _show_agent(self, text: str) -> None:
        t = self.query_one("#transcript", VerticalScroll)
        t.mount(Static(f"[bold {GREEN}]◆ depression.ai[/]",
                       classes="msg-agent-head"))
        t.mount(Markdown(text, classes="msg-agent-body"))
        t.scroll_end(animate=False)

    # ── picker ────────────────────────────────────────────────────────
    def action_open_picker(self) -> None:
        try:
            self.query_one("#picker-overlay", ModelPickerModal).add_class("-open")
            self._picker_open = True
        except Exception:
            pass

    def _close_model_picker(self) -> None:
        try:
            self.query_one("#picker-overlay", ModelPickerModal).remove_class("-open")
            self._picker_open = False
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            pass

    def _select_model(self, provider: str, model: str) -> None:
        self.selected_provider = provider
        self.selected_model = model
        default_url = next((u for p, u, _ in PROVIDERS if p == provider), "")
        if not self.cfg.get("base_url") or self.cfg.get("provider") != provider:
            self.cfg["base_url"] = default_url
        self.cfg["provider"] = provider
        self.cfg["model"] = model
        save_config(self.cfg)
        self._refresh_mode_chip()
        self._show_system(f"selected {provider} · {model}")
        self._show_system("open /connect to paste your api key")

        # refresh the LLM panel with the new provider/model
        try:
            self.query_one("#panel-llm", LLMConnectPanel).refresh_values()
        except Exception:
            pass
        self.show_sidebar("llm")

    # ── respond ───────────────────────────────────────────────────────
    @work(exclusive=True, group="msg")
    async def _respond(self, text: str) -> None:
        self._spinner_on = True
        self._refresh_mode_chip()
        try:
            if self.coordinator:
                result = await asyncio.wait_for(
                    self.coordinator.process_query(
                        text, mode=self.current_mode.lower(),
                        auto_execute=True),
                    timeout=180,
                )
                if result.get("success"):
                    self._show_agent(
                        result.get("execution") or result.get("response", ""))
                else:
                    self._show_error(result.get("error", "unknown error"))
            else:
                if self.cfg.get("api_key"):
                    self._show_agent(
                        f"_({self.cfg['provider']}/{self.cfg['model']} is "
                        f"connected but no coordinator is running.)_\n\n"
                        f"you said: **{text}**"
                    )
                else:
                    self._show_agent(
                        f"_no model connected — open /connect to paste your key._\n\n"
                        f"you said: **{text}**"
                    )
        except asyncio.TimeoutError:
            self._show_error("timed out after 180s")
        except Exception as e:
            self._show_error(str(e))
        finally:
            self._spinner_on = False
            self._refresh_mode_chip()

    # ── actions ───────────────────────────────────────────────────────
    def action_cycle_mode(self) -> None:
        self.current_mode = MODE_ORDER[
            (MODE_ORDER.index(self.current_mode) + 1) % len(MODE_ORDER)
        ]
        self._refresh_mode_chip()

    def action_cancel(self) -> None:
        if self._picker_open:
            self._close_model_picker()
            return
        self.exit()

    def action_escape(self) -> None:
        if self._picker_open:
            self._close_model_picker()

    def action_clear(self) -> None:
        t = self.query_one("#transcript", VerticalScroll)
        t.remove_children()


if __name__ == "__main__":
    DepressionApp().run()