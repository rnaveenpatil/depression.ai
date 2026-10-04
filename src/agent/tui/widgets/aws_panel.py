"""
AWS credentials panel with animated save feedback and MCP status.

Animations:
    - scan-line sweep on successful save
    - rotating region radar while saving
    - key strength meter (length-based)
    - breathing status dot when credentials are present
    - endpoint trace line on verify

Secret field contract:
    * The secret Input is `password=True` (so characters are masked on
      screen) and is PREFILLED with the stored secret, exactly like the LLM
      panel's API-key Input. A pasted key therefore stays in the field after
      saving instead of vanishing. An earlier build wrote a literal
      "••••••••" into .value and cleared the field on every save, which made
      a real paste look like it was rejected.
    * Saving reports whether the value actually changed: the status line
      reads "secret replaced · saved HH:MM:SS" when the typed value differs
      from the stored one, "secret unchanged" otherwise.

Storage contract:
    * Credentials are written via EnvManager to the PROJECT env file
      (find_env_file(), mode 0600) — the checkout owns its own
      credentials.
    * EnvManager mirrors them into os.environ and notifies subscribers,
      so the MCP client, aws_helper, and terminal subprocesses see the
      change without a restart.
    * After a successful save, this panel fires `app.on_aws_changed()` so
      the running agent reloads MCP with the new credentials.
    * The panel VERIFIES the write landed on disk before reporting
      success: both keys must be present AND must equal the values just
      submitted. Otherwise save_credentials() returns {"ok": False, ...}
      with the exact reason.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Select, Static

from agent.utils.env_manager import EnvManager


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

DEFAULT_REGION = "us-east-1"

AWS_REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "ap-south-1", "ap-south-2",
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

# The secret field uses a real Input placeholder (see _secret_placeholder),
# NOT a literal value written into .value: a literal value is indistinguishable
# from a real secret, so after a save the field looked identical whether the
# new secret was accepted or the old one was still on disk.


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


# ----------------------------------------------------------------------
# Helper: get the canonical env file path regardless of EnvManager version
# ----------------------------------------------------------------------

def _global_env_path() -> Optional[str]:
    """
    Return the absolute path of the global env file, or None if we can't
    determine it. Tries several import names so it works across versions.
    """
    try:
        from agent.utils import env_manager as _em  # type: ignore
    except Exception:
        return None

    for name in ("GLOBAL_ENV_PATH", "GLOBAL_ENV_FILE", "ENV_FILE_PATH"):
        p = getattr(_em, name, None)
        if p is not None:
            return str(p)

    # Older builds may expose it via a helper.
    for fn_name in ("global_env_path", "default_env_path"):
        fn = getattr(_em, fn_name, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                continue

    # Last resort: derive from EnvManager.
    try:
        em = EnvManager.get()
        for attr in ("path", "env_path", "global_env_path"):
            p = getattr(em, attr, None)
            if p is not None:
                return str(p)
    except Exception:
        pass

    # Final fallback: ~/.agent/env
    return os.path.expanduser("~/.agent/env")


def _written_env_path() -> Optional[str]:
    """
    Return the env file that AWS credentials are actually WRITTEN to.

    ``EnvManager.save_aws_credentials()`` persists creds to
    ``find_env_file()`` (the project-scoped ``.env``), so verification must
    target that same file. Checking ``~/.agent/env`` instead produced false
    failures (creds saved fine but "env file was not created" / "missing
    AWS_ACCESS_KEY_ID"), and false successes whenever a stale
    ``AWS_ACCESS_KEY_ID=`` line was left behind in the global file.
    """
    try:
        from agent.utils.env_manager import find_env_file  # type: ignore

        return str(find_env_file())
    except Exception:
        return _global_env_path()


def _parse_env_text(text: str) -> Dict[str, str]:
    """Parse raw ``KEY=VALUE`` env-file text into a dict."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


class AWSPanel(Vertical):
    DEFAULT_CSS = f"""
    AWSPanel {{
        height: auto;
        width: 100%;
        padding: 0 1;
        background: {BG};
    }}
    AWSPanel .label {{ color: {MUTED}; margin-top: 1; }}
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
    AWSPanel .strength {{ height: 1; width: 100%; }}
    AWSPanel .trace {{ height: 1; width: 100%; }}
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
        self._saved_at: Optional[str] = None
        self._secret_replaced: bool = False

        self._env_cb = self._on_env_changed

    # ------------------------------------------------------------------
    # COMPOSE
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        self._title = Static("", markup=True)
        yield self._title

        yield Static("Access Key ID", classes="label")
        yield Input(value=self._stored_access_key(), id="aws-key")

        yield Static("Secret Access Key", classes="label")
        yield Input(
            value=self._stored_secret(),
            password=True,
            id="aws-secret",
        )

        self._strength = Static("", classes="strength")
        yield self._strength

        self._region_static = Static("Region", classes="label")
        yield self._region_static
        yield Select(
            [(r, r) for r in AWS_REGIONS],
            value=self._stored_region() or DEFAULT_REGION,
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

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._sync_from_env()
        try:
            EnvManager.get().subscribe(self._env_cb)
        except Exception:
            pass

        self._render_title()
        self._render_strength()
        self.refresh_mcp_status()
        self.set_interval(0.12, self._tick)

    def on_unmount(self) -> None:
        try:
            EnvManager.get().unsubscribe(self._env_cb)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # CANONICAL READS (always from EnvManager)
    # ------------------------------------------------------------------

    def _read_canonical(self) -> Dict[str, Optional[str]]:
        """Read the AWS creds straight from EnvManager."""
        try:
            creds = EnvManager.get().get_aws_credentials() or {}
        except Exception:
            creds = {}
        return {
            "access_key": creds.get("access_key"),
            "secret_key": creds.get("secret_key"),
            "region": creds.get("region"),
        }

    def _stored_access_key(self) -> str:
        return self._read_canonical().get("access_key") or ""

    def _stored_secret_present(self) -> bool:
        return bool(self._read_canonical().get("secret_key"))

    def _stored_secret(self) -> str:
        """The stored secret, shown in the field like the API-key panel.

        `password=True` still masks the characters on screen; keeping the
        real value in the field means a pasted key stays visible (masked)
        instead of being wiped on save, and matches how the LLM panel's
        API-key Input behaves.
        """
        return self._read_canonical().get("secret_key") or ""

    def _stored_region(self) -> str:
        return self._read_canonical().get("region") or ""

    # ------------------------------------------------------------------
    # ENV SYNC
    # ------------------------------------------------------------------

    def _sync_from_env(self) -> None:
        """Copy EnvManager's view of AWS creds into the app state."""
        creds = self._read_canonical()
        if not isinstance(getattr(self._app, "aws", None), dict):
            return
        if creds.get("access_key"):
            self._app.aws["access_key"] = creds["access_key"]
        if creds.get("secret_key"):
            self._app.aws["secret_key"] = creds["secret_key"]
        if creds.get("region"):
            self._app.aws["region"] = creds["region"]

    def _on_env_changed(self, snapshot: dict) -> None:
        """EnvManager notified us that creds changed (e.g. another panel)."""
        try:
            if not isinstance(getattr(self._app, "aws", None), dict):
                return
            if snapshot.get("access_key"):
                self._app.aws["access_key"] = snapshot["access_key"]
            if snapshot.get("secret_key"):
                self._app.aws["secret_key"] = snapshot["secret_key"]
            if snapshot.get("region"):
                self._app.aws["region"] = snapshot["region"]
        except Exception:
            return
        try:
            self.refresh_values()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ANIMATIONS
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
    # RENDER HELPERS
    # ------------------------------------------------------------------

    def _render_title(self) -> None:
        if self._title is None:
            return
        has_creds = self._stored_secret_present() and bool(self._stored_access_key())
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
            return str(val) if val else DEFAULT_REGION
        except Exception:
            return DEFAULT_REGION

    # ------------------------------------------------------------------
    # MCP STATUS
    # ------------------------------------------------------------------

    def refresh_mcp_status(self) -> None:
        if self._mcp_status is None:
            return
        self._mcp_status.update(self._compute_mcp_status())

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
                        if t.get("function", {}).get("name", "").startswith("mcp__aws__")
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
    # PUBLIC API
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
        """
        Redraw the panel from the canonical store (EnvManager), not from
        the app's cached snapshot. Fixes the case where the panel keeps
        showing empty fields after a successful save because the app's
        local copy never got updated.
        """
        creds = self._read_canonical()
        access = creds.get("access_key") or ""
        secret = creds.get("secret_key") or ""
        secret_present = bool(secret)
        region = creds.get("region") or DEFAULT_REGION

        try:
            self.query_one("#aws-key", Input).value = access
            # Keep the secret in the field (masked by password=True) instead
            # of blanking it, so a pasted key stays visible after save —
            # same behaviour as the LLM panel's API-key Input.
            self.query_one("#aws-secret", Input).value = secret
            if region in AWS_REGIONS:
                self.query_one("#aws-region", Select).value = region
        except Exception:
            pass

        # Keep the app's snapshot in sync too.
        if isinstance(getattr(self._app, "aws", None), dict):
            if access:
                self._app.aws["access_key"] = access
            if secret_present:
                self._app.aws["secret_key"] = creds.get("secret_key")
            if region:
                self._app.aws["region"] = region

        self._render_title()
        self._render_strength()
        self._render_saved_at()
        self.refresh_mcp_status()

    def _render_saved_at(self) -> None:
        """Show when the secret was last written, and whether it changed.

        The secret field is password-masked and is cleared after every save,
        so this line is the only visible confirmation that a paste was
        actually accepted (the old mask-as-value made every save look
        identical).
        """
        if self._status is None:
            return
        if not self._stored_secret_present():
            self._status.update(f"[{AMBER}]no secret saved yet[/]")
            return
        stamp = self._saved_at or "at an unknown time"
        verb = "secret replaced" if self._secret_replaced else "secret unchanged"
        self._status.update(f"[{GREEN}]●[/] {verb} · saved {stamp}")

    # ------------------------------------------------------------------
    # SAVE — write, verify, then fire the reload hook
    # ------------------------------------------------------------------

    def save_credentials(self) -> Dict[str, Any]:
        """
        Persist AWS creds to ~/.agent/env, verify the write, notify the
        agent, and refresh the panel.

        Returns:
            {"ok": bool, "error": Optional[str], "path": Optional[str]}
        """
        # --- read fields ---
        try:
            access_key = self.query_one("#aws-key", Input).value.strip()
        except Exception:
            access_key = ""

        try:
            secret_input = self.query_one("#aws-secret", Input)
            raw_secret = secret_input.value.strip()
        except Exception:
            raw_secret = ""

        region = self._current_region()

        # --- resolve the secret ---
        # The field is prefilled with the stored secret, so a non-empty field
        # is only a *change* when it differs from what is on disk. Compare
        # against the canonical value to decide whether to report the
        # secret as replaced.
        stored_secret = self._read_canonical().get("secret_key") or ""
        secret_replaced = bool(raw_secret) and raw_secret != stored_secret
        if raw_secret:
            secret_key = raw_secret
        else:
            secret_key = stored_secret
            if not secret_key:
                secret_key = (
                    (getattr(self._app, "aws", {}) or {}).get("secret_key") or ""
                )

        if not access_key:
            return {"ok": False, "error": "Access key is required.", "path": None}
        if not secret_key:
            return {
                "ok": False,
                "error": "Secret access key is required.",
                "path": None,
            }

        # --- persist via EnvManager ---
        try:
            em = EnvManager.get()
        except Exception as e:
            return {
                "ok": False,
                "error": f"EnvManager unavailable: {e}",
                "path": None,
            }

        saved = False
        last_error = ""

        # Preferred path: EnvManager.save_aws_credentials (updates
        # os.environ + notifies subscribers).
        saver = getattr(em, "save_aws_credentials", None)
        if callable(saver):
            try:
                saver(
                    access_key=access_key,
                    secret_key=secret_key,
                    region=region,
                )
                saved = True
            except Exception as e:
                last_error = str(e)

        # Fallback: module-level set_aws_credentials helper.
        if not saved:
            try:
                from agent.utils.env_manager import set_aws_credentials
                set_aws_credentials(access_key, secret_key, region)
                reload_fn = getattr(em, "reload", None)
                if callable(reload_fn):
                    try:
                        reload_fn()
                    except Exception:
                        pass
                saved = True
            except Exception as e:
                last_error = last_error or str(e)

        if not saved:
            return {
                "ok": False,
                "error": last_error or "unknown error saving credentials",
                "path": None,
            }

        # --- verify the write actually landed on disk ---
        # EnvManager.save_aws_credentials() writes to the PROJECT .env, so
        # verify THAT file (falling back to the legacy global path only when
        # the project path cannot be resolved).
        env_path = _written_env_path() or _global_env_path()
        if env_path:
            try:
                if not os.path.isfile(env_path):
                    return {
                        "ok": False,
                        "error": f"env file was not created at {env_path}",
                        "path": env_path,
                    }
                with open(env_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                if "AWS_ACCESS_KEY_ID=" not in text:
                    return {
                        "ok": False,
                        "error": f"{env_path} missing AWS_ACCESS_KEY_ID",
                        "path": env_path,
                    }
                if "AWS_SECRET_ACCESS_KEY=" not in text:
                    return {
                        "ok": False,
                        "error": f"{env_path} missing AWS_SECRET_ACCESS_KEY",
                        "path": env_path,
                    }
                # Confirm the value on disk is the one just submitted. The
                # presence checks above passed even when a stale secret was
                # left behind by a failed write.
                written = _parse_env_text(text)
                if written.get("AWS_ACCESS_KEY_ID", "") != access_key:
                    return {
                        "ok": False,
                        "error": f"{env_path} still holds a different access key",
                        "path": env_path,
                    }
                if secret_replaced and written.get(
                    "AWS_SECRET_ACCESS_KEY", ""
                ) != secret_key:
                    return {
                        "ok": False,
                        "error": f"{env_path} still holds a different secret",
                        "path": env_path,
                    }
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"verification failed: {e}",
                    "path": env_path,
                }

        self._saved_at = datetime.now().strftime("%H:%M:%S")
        self._secret_replaced = secret_replaced

        # --- update the app's local cache ---
        if isinstance(getattr(self._app, "aws", None), dict):
            self._app.aws["access_key"] = access_key
            self._app.aws["secret_key"] = secret_key
            self._app.aws["region"] = region

        # --- fire reload hook so the running agent respawns MCP ---
        try:
            hook = getattr(self._app, "on_aws_changed", None)
            if callable(hook):
                hook()
        except Exception:
            pass

        # --- refresh visuals ---
        self.refresh_values()
        self.refresh_mcp_status()
        self.start_scan(passes=2)
        self.start_radar(duration_ticks=30)
        self.start_trace(ok=True)

        return {"ok": True, "error": None, "path": env_path}