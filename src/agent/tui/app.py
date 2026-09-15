"""Textual UI for the real Depression.AI agent.

The TUI is presentation only: every prompt is sent to AgentCoordinator and
all provider/AWS credentials entered in the panels are persisted in .env.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Select, Static
from textual import on, work

from agent.agent.dual_agent import AgentCoordinator
from agent.llm.provider import get_llm_registry
from agent.llm.runtime import configure_runtime_provider, load_runtime_config
from agent.utils.env_manager import get_aws_credentials, set_aws_credentials


AWS_REGIONS = [
    "ap-south-1", "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1", "ap-southeast-1",
    "ap-southeast-2", "ap-northeast-1", "ca-central-1", "sa-east-1",
]


class DepressionApp(App):
    """Agent-connected Textual application."""

    CSS = """
    Screen { background: #000; color: #aaffcc; }
    #body { height: 1fr; }
    #transcript { width: 1fr; padding: 1 2; }
    #sidebar { width: 34; border-left: round #0a3d20; padding: 1; }
    .panel-title { color: #00ff66; text-style: bold; margin-bottom: 1; }
    .label { color: #3d8c5c; margin-top: 1; }
    Input { background: #031008; border: round #0a3d20; color: #aaffcc; margin: 0 0 1 0; }
    Input:focus { border: round #00ff66; }
    Select { background: #031008; border: round #0a3d20; margin-bottom: 1; }
    Button { width: 100%; background: #061a0f; border: round #00ff66; color: #00ff66; margin-top: 1; }
    #prompt { height: 3; border: round #00ff66; margin: 1 2; }
    .user { color: #88ffbb; margin-bottom: 1; }
    .agent { color: #aaffcc; margin-bottom: 1; }
    .system { color: #3d8c5c; margin-bottom: 1; }
    .error { color: #ff4466; margin-bottom: 1; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, coordinator: Optional[AgentCoordinator] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.coordinator = coordinator
        self.cfg = load_runtime_config()
        self.aws = get_aws_credentials()
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield VerticalScroll(id="transcript")
            with Vertical(id="sidebar"):
                yield Static("▌ LLM CONNECTION", classes="panel-title")
                yield Static("Base URL", classes="label")
                yield Input(value=self.cfg["base_url"], id="base-url",
                            placeholder="https://api.example.com/v1")
                yield Static("API Key", classes="label")
                yield Input(value="••••••••" if self.cfg["api_key"] else "",
                            password=True, id="api-key", placeholder="API key")
                yield Static("Model ID", classes="label")
                yield Input(value=self.cfg["model"], id="model-id",
                            placeholder="provider/model-id")
                yield Button("CONNECT", id="connect")
                yield Button("DISCOVER /models", id="discover")
                yield Static("", id="llm-status", classes="system")
                yield Static("▌ AWS CREDENTIALS", classes="panel-title")
                yield Static("Access Key ID", classes="label")
                yield Input(value=self.aws.get("access_key") or "", id="aws-key")
                yield Static("Secret Access Key", classes="label")
                yield Input(value="••••••••" if self.aws.get("secret_key") else "",
                            password=True, id="aws-secret")
                yield Static("Region", classes="label")
                yield Select([(r, r) for r in AWS_REGIONS],
                             value=self.aws.get("region") or "ap-south-1", id="aws-region")
                yield Button("SAVE AWS", id="aws-save")
                yield Static("", id="aws-status", classes="system")
        yield Input(placeholder="ask the agent…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._system("TUI connected to the AgentCoordinator. Configure LLM and AWS from the sidebar.")
        self.query_one("#prompt", Input).focus()

    def _write(self, text: str, cls: str = "agent") -> None:
        self.query_one("#transcript", VerticalScroll).mount(Static(text, classes=cls))
        self.call_after_refresh(lambda: self.query_one("#transcript", VerticalScroll).scroll_end(animate=False))

    def _system(self, text: str) -> None:
        self._write(text, "system")

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.busy:
            return
        if text.startswith("/"):
            self._slash(text)
            return
        self._run_agent(text)

    def _slash(self, text: str) -> None:
        command = text.split()[0].lower()
        if command in ("/aws",):
            self.query_one("#aws-key", Input).focus()
        elif command in ("/connect",):
            self.query_one("#base-url", Input).focus()
        elif command in ("/models", "/model"):
            self._discover_models()
        elif command == "/clear":
            self.query_one("#transcript", VerticalScroll).remove_children()
        elif command == "/plan":
            if self.coordinator:
                self.coordinator.set_mode("plan")
            self._system("Mode: Plan")
        elif command == "/build":
            if self.coordinator:
                self.coordinator.set_mode("build")
            self._system("Mode: Build")
        elif command in ("/quit", "/exit"):
            self.exit()
        else:
            self._system("Commands: /connect /aws /models /plan /build /clear /quit")

    @work(exclusive=True, group="agent")
    async def _run_agent(self, text: str) -> None:
        self.busy = True
        self._write(f"› {text}", "user")
        if not self.coordinator:
            self._write("Agent coordinator is not initialized.", "error")
            self.busy = False
            return
        try:
            result = await asyncio.wait_for(
                self.coordinator.process_query(
                    text, mode=self.coordinator.current_mode, auto_execute=True),
                timeout=300,
            )
            if result.get("success"):
                output = result.get("execution") or result.get("response") or "Task completed."
                self._write(str(output), "agent")
            else:
                self._write(str(result.get("error", "Agent request failed")), "error")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._write(f"Agent error: {exc}", "error")
        finally:
            self.busy = False

    @on(Button.Pressed, "#connect")
    def connect_llm(self) -> None:
        base_url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        api_key = self.cfg["api_key"] if key_input.value == "••••••••" else key_input.value.strip()
        model = self.query_one("#model-id", Input).value.strip()
        if not base_url or not api_key or not model:
            self.query_one("#llm-status", Static).update("base URL, API key and model are required")
            return
        try:
            configure_runtime_provider(get_llm_registry(), base_url, api_key, model)
            self.cfg = load_runtime_config()
            self.query_one("#llm-status", Static).update(f"connected · custom · {model}")
            self._system(f"LLM connected: {base_url} · {model}")
        except Exception as exc:
            self.query_one("#llm-status", Static).update(f"error: {exc}")

    @on(Button.Pressed, "#discover")
    def discover_llm(self) -> None:
        self._discover_models()

    @work(exclusive=True, group="discover")
    async def _discover_models(self) -> None:
        url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        key_input = self.query_one("#api-key", Input)
        key = self.cfg["api_key"] if key_input.value == "••••••••" else key_input.value.strip()
        if not url:
            self.query_one("#llm-status", Static).update("enter a base URL first")
            return
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"} if key else {})
                response.raise_for_status()
                models = response.json().get("data", [])
            ids = [str(m.get("id")) for m in models if m.get("id")]
            if ids:
                self.query_one("#model-id", Input).value = ids[0]
                self.query_one("#llm-status", Static).update(f"discovered {len(ids)} model(s); selected {ids[0]}")
                self._system("Models: " + ", ".join(ids[:12]))
            else:
                self.query_one("#llm-status", Static).update("/models returned no model IDs")
        except Exception as exc:
            self.query_one("#llm-status", Static).update(f"discovery failed: {exc}; enter model manually")

    @on(Button.Pressed, "#aws-save")
    def save_aws(self) -> None:
        key = self.query_one("#aws-key", Input).value.strip()
        secret_input = self.query_one("#aws-secret", Input)
        secret = self.aws.get("secret_key") or "" if secret_input.value == "••••••••" else secret_input.value.strip()
        region = str(self.query_one("#aws-region", Select).value)
        if not key or not secret:
            self.query_one("#aws-status", Static).update("access key and secret are required")
            return
        try:
            set_aws_credentials(key, secret, region)
            self.aws = get_aws_credentials()
            self.query_one("#aws-status", Static).update(f"saved to .env · {region}")
            self._system(f"AWS credentials saved to .env ({region}); secret is not displayed.")
        except Exception as exc:
            self.query_one("#aws-status", Static).update(f"error: {exc}")


__all__ = ["DepressionApp"]
