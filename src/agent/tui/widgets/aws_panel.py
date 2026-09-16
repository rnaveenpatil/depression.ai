"""
AWS credentials panel with animated save feedback and MCP status.

Animations:
    - scan-line sweep on successful save
    - rotating region radar while saving
    - key strength meter (length-based)
    - breathing status dot when credentials are present
    - endpoint trace line on verify

Status:
    - MCP status line reflects whether the AWS MCP server is connected
    - Falls back to the AWS CLI banner when MCP is unavailable
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Select, Static


GREEN = "#00ff66"
GREEN_GLOW = "#88ffbb"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"
PANEL = "#031008"
BG = "#000000"

AWS_REGIONS = [
    "ap-south-1", "ap-south-2",
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-central-2",
    "eu-north-1", "eu-south-1", "eu-south-2",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4",
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
    "ca-central-1", "ca-west-1",
    "sa-east-1",
    "me-central-1", "me-south-1",
    "af-south-1",
    "il-central-1",
]

_RADAR = ["◐", "◓", "◑", "◒"]
_SCAN_WIDTH = 34
_STRENGTH_CHARS = "▰"


class AWSPanel(Vertical):
    DEFAULT_CSS = f"""
    AWSPanel {{
        height: auto;
        width: 100%;
        padding: 0 1;
        background: {BG};
    }}
    AWSPanel .title-row {{
        height: 1;
        width: 100%;
    }}
    AWSPanel .label {{
        color: {MUTED};
        margin-top: 1;
    }}
    AWSPanel Input {{
        background: {PANEL};
        border: round {BORDER};
        color: {TEXT};
        margin: 0 0 1 0;
        height: 3;
        width: 100%;
    }}
    AWSPanel Input:focus {{ border: round {GREEN}; }}
    AWSPanel Select {{
        background: {PANEL};
        border: round {BORDER};
        margin: 0 0 1 0;
        width: 100%;
    }}
    AWSPanel Button {{
        width: 1fr;
        min-width: 12;
        background: transparent;
        border: round {GREEN};
        color: {GREEN};
        margin-top: 1;
        height: 3;
        text-style: bold;
    }}
    AWSPanel Button:hover, AWSPanel Button:focus {{
        background: #061a0f;
        color: {GREEN_GLOW};
        border: round {GREEN_GLOW};
    }}
    AWSPanel .strength {{
        height: 1;
        width: 100%;
    }}
    AWSPanel .trace {{
        height: 1;
        width: 100%;
    }}
    AWSPanel .status {{
        height: auto;
        width: 100%;
        color: {MUTED};
        margin-top: 1;
    }}
    AWSPanel .mcp-status {{
        height: auto;
        width: 100%;
        color: {MUTED};
        margin-top: 1;
    }}
    """

    def __init__(self, app_ref: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self._app = app_ref

        self._scan_pos = 0
        self._scan_active = False
        self._scan_passes = 0
        self._radar_i = 0
        self._radar_active = False
        self._breath_i = 0
        self._trace_i = 0
        self._trace_active = False
        self._trace_ok = True

        self._title: Optional[Static] = None
        self._strength: Optional[Static] = None
        self._trace: Optional[Static] = None
        self._status: Optional[Static] = None
        self._region_static: Optional[Static] = None
        self._mcp_status: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._title = Static("", markup=True)
        yield self._title

        yield Static("Access Key ID", classes="label")
        yield Input(value=self._app.aws.get("access_key") or "", id="aws-key")

        yield Static("Secret Access Key", classes="label")
        yield Input(
            value="••••••••" if self._app.aws.get("secret_key") else "",
            password=True,
            id="aws-secret",
        )

        self._strength = Static("", classes="strength")
        yield self._strength

        self._region_static = Static("Region", classes="label")
        yield self._region_static
        yield Select(
            [(r, r) for r in AWS_REGIONS],
            value=self._app.aws.get("region") or "ap-south-1",
            id="aws-region",
            allow_blank=False,
        )

        yield Button("SAVE AWS", id="aws-save")

        self._trace = Static("", classes="trace")
        yield self._trace

        self._status = Static("", id="aws-status", classes="status")
        yield self._status

        self._mcp_status = Static("", classes="mcp-status")
        yield self._mcp_status

    def on_mount(self) -> None:
        self._render_title()
        self._render_strength()
        self.refresh_mcp_status()
        self.set_interval(0.12, self._tick)

    # ------------------------------------------------------------------
    # animations
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self._breath_i += 1
        dirty = False

        if self._radar_active:
            self._radar_i += 1
            self._render_region_label()
            dirty = True

        if self._scan_active:
            self._scan_pos += 1
            if self._scan_pos >= _SCAN_WIDTH + 10:
                self._scan_pos = 0
                self._scan_passes -= 1
                if self._scan_passes <= 0:
                    self._scan_active = False
                    self._render_title()
            else:
                self._render_title()
            dirty = True

        if self._trace_active:
            self._trace_i += 1
            if self._trace_i >= 24:
                self._trace_active = False
                self._render_trace_final()
            else:
                self._render_trace_running()
            dirty = True

        if self._breath_i % 8 == 0 and not (self._scan_active or self._radar_active):
            self._render_title()
            dirty = True

        if not dirty:
            return

    # ------------------------------------------------------------------
    # render helpers
    # ------------------------------------------------------------------

    def _render_title(self) -> None:
        if self._title is None:
            return
        has_creds = bool(
            self._app.aws.get("access_key") and self._app.aws.get("secret_key")
        )
        if has_creds:
            phase = "●" if (self._breath_i // 8) % 2 == 0 else "○"
            dot = f"[{GREEN}]{phase}[/]"
        else:
            dot = f"[{DIM}]○[/]"

        if self._scan_active:
            cells = ["─"] * _SCAN_WIDTH
            for offset in range(3):
                pos = self._scan_pos - offset
                if 0 <= pos < _SCAN_WIDTH:
                    cells[pos] = "━" if offset == 0 else "─"
            bar = "".join(cells)
            mid = self._scan_pos
            line = (
                f"[{GREEN_DIM}]{bar[:mid]}[/]"
                f"[{GREEN_GLOW}]{bar[mid:mid+1]}[/]"
                f"[{GREEN_DIM}]{bar[mid+1:]}[/]"
            )
            self._title.update(
                f"{dot} [bold {GREEN}]▌ AWS CREDENTIALS[/]  {line}"
            )
            return

        self._title.update(f"{dot} [bold {GREEN}]▌ AWS CREDENTIALS[/]")

    def _render_region_label(self) -> None:
        if self._region_static is None:
            return
        if self._radar_active:
            glyph = _RADAR[self._radar_i % len(_RADAR)]
            self._region_static.update(
                f"[{MUTED}]Region[/]  [{AMBER}]{glyph}[/] [{DIM}]saving…[/]"
            )
        else:
            self._region_static.update(f"[{MUTED}]Region[/]")

    def _render_strength(self) -> None:
        if self._strength is None:
            return
        try:
            key = self.query_one("#aws-key", Input).value.strip()
        except Exception:
            key = ""
        length = len(key)
        tiers = min(5, max(0, length // 4))
        bar = _STRENGTH_CHARS * tiers + "▱" * (5 - tiers)
        color = GREEN if tiers >= 4 else AMBER if tiers >= 2 else MUTED
        self._strength.update(
            f"[{MUTED}]access key length[/]  [{color}]{bar}[/]  "
            f"[{DIM}]{length} chars[/]"
        )

    def _render_trace_running(self) -> None:
        if self._trace is None:
            return
        span = 20
        head = min(self._trace_i, span)
        tail = "─" * head
        remainder = "·" * (span - head)
        glyph = "▶" if self._trace_ok else "✕"
        color = GREEN if self._trace_ok else ERROR
        region = self._current_region()
        self._trace.update(
            f"[{color}]◉[/] [{color}]{tail}{glyph}[/][{DIM}]{remainder}[/] "
            f"[{MUTED}]→ {region}.amazonaws.com[/]"
        )

    def _render_trace_final(self) -> None:
        if self._trace is None:
            return
        region = self._current_region()
        if self._trace_ok:
            self._trace.update(
                f"[{GREEN}]◉[/] [{GREEN}]────────────────▶[/] "
                f"[{MUTED}]{region}.amazonaws.com[/]  [{GREEN}]ready[/]"
            )
        else:
            self._trace.update(
                f"[{ERROR}]◉[/] [{ERROR}]───────────────✕[/] "
                f"[{MUTED}]{region}.amazonaws.com[/]  [{ERROR}]failed[/]"
            )

    def _current_region(self) -> str:
        try:
            val = self.query_one("#aws-region", Select).value
            return str(val) if val else "ap-south-1"
        except Exception:
            return "ap-south-1"

    # ------------------------------------------------------------------
    # MCP status
    # ------------------------------------------------------------------

    def refresh_mcp_status(self) -> None:
        """Reflect whether the AWS MCP server is available."""
        if self._mcp_status is None:
            return

        line = self._compute_mcp_status()
        self._mcp_status.update(line)

    def _compute_mcp_status(self) -> str:
        mcp_tools = 0
        try:
            from agent.mcp.aws_config import build_aws_cli_fallback_status
            fallback = build_aws_cli_fallback_status()
        except Exception:
            fallback = {"usable": False, "cli_available": False,
                        "credentials_present": False}

        try:
            if self._app.coordinator is not None:
                for attr in ("plan_agent", "build_agent"):
                    agent = getattr(self._app.coordinator, attr, None)
                    client = getattr(agent, "mcp_client", None) if agent else None
                    if client is None:
                        continue
                    tools = client.list_tools() if hasattr(client, "list_tools") else []
                    mcp_tools = sum(
                        1 for t in tools
                        if t["function"]["name"].startswith("mcp__aws__")
                    )
                    if mcp_tools:
                        break
        except Exception:
            mcp_tools = 0

        if mcp_tools:
            return (
                f"[{MUTED}]MCP[/]   [{GREEN}]connected[/]  "
                f"[{DIM}]({mcp_tools} AWS tool"
                f"{'s' if mcp_tools != 1 else ''})[/]"
            )
        if fallback.get("usable"):
            return (
                f"[{MUTED}]MCP[/]   [{AMBER}]unavailable[/]  "
                f"[{DIM}]· using AWS CLI fallback[/]"
            )
        if not fallback.get("credentials_present"):
            return (
                f"[{MUTED}]MCP[/]   [{DIM}]no credentials[/]  "
                f"[{DIM}]· add keys above[/]"
            )
        return (
            f"[{MUTED}]MCP[/]   [{AMBER}]unavailable[/]  "
            f"[{DIM}]· install uvx or aws CLI for fallback[/]"
        )

    # ------------------------------------------------------------------
    # public API used by the app
    # ------------------------------------------------------------------

    def start_scan(self, passes: int = 2) -> None:
        self._scan_active = True
        self._scan_passes = passes
        self._scan_pos = 0

    def start_radar(self, duration_ticks: int = 30) -> None:
        self._radar_active = True
        self._radar_i = 0
        def _stop() -> None:
            self._radar_active = False
            self._render_region_label()
        self.set_timer(duration_ticks * 0.12, _stop)

    def start_trace(self, ok: bool = True) -> None:
        self._trace_ok = ok
        self._trace_active = True
        self._trace_i = 0

    def refresh_strength(self) -> None:
        self._render_strength()

    def set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.update(text)

    def refresh_values(self) -> None:
        try:
            self.query_one("#aws-key", Input).value = (
                self._app.aws.get("access_key") or ""
            )
            secret_input = self.query_one("#aws-secret", Input)
            secret_input.value = (
                "••••••••" if self._app.aws.get("secret_key") else ""
            )
            region = self._app.aws.get("region") or "ap-south-1"
            if region in AWS_REGIONS:
                self.query_one("#aws-region", Select).value = region
        except Exception:
            pass
        self._render_title()
        self._render_strength()
        self.refresh_mcp_status()