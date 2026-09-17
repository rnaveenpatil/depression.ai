"""
Tool call card — one bordered card per tool invocation.

Card anatomy:
    ┌─ tool_name  ·  target                  ⏱ 1.23s
    │  (body: command / output / diff)
    └─

States drive the left border color and the trailing chip:
    running   → amber left pulse, "running" chip
    done      → green left border, duration chip
    error     → red left border, "error" chip
    cached    → dim green, "cached" chip
    background→ amber left border, "pid NNNN" chip

AWS tools get their own treatment: an amber left border, a header that
reads `aws · <service>`, a service-colored access pulse while running,
and a one-line result summary above the JSON body.

Every dynamic string is escaped with _esc() so markup brackets in tool
output (JSON [0], logs [INFO], shell [x]) are rendered literally.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from agent.tui.widgets.diff_view import render_diff


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"
RUNNING = "#ffd27a"

# AWS accent palette. Each service gets one distinct color so the
# transcript is scannable at a glance.
AWS_BASE = "#ff9500"
_AWS_SERVICE_COLORS: Dict[str, str] = {
    "s3":      "#ff9500",   # orange
    "ec2":     "#ff6b35",   # deeper orange
    "ecs":     "#e879f9",   # magenta
    "eks":     "#c084fc",   # purple
    "lambda":  "#fbbf24",   # yellow
    "iam":     "#ef4444",   # red
    "rds":     "#22d3ee",   # cyan
    "dynamodb": "#a3e635",  # lime
    "sqs":     "#60a5fa",   # blue
    "sns":     "#f472b6",   # pink
    "cloudformation": "#94a3b8",  # slate
    "cloudwatch": "#38bdf8",      # sky
    "logs":    "#818cf8",   # indigo
    "route53": "#f59e0b",   # amber
    "apigateway": "#10b981",# emerald
    "kms":     "#fcd34d",   # pale gold
    "ssm":     "#67e8f9",   # light cyan
    "secretsmanager": "#f87171",  # light red
    "ecr":     "#a78bfa",   # violet
}

_AWS_FALLBACK_COLOR = AWS_BASE

_DIFF_TOOLS = {"write", "edit", "apply_patch", "filesystem", "patch"}
_BASH_TOOLS = {"bash", "terminal"}
_READ_TOOLS = {"read"}
_SEARCH_TOOLS = {"grep", "glob", "search"}
_WEB_TOOLS = {"webfetch", "websearch", "web"}

_BASH_PREVIEW_LINES = 8
_GENERIC_PREVIEW_LINES = 4

_PULSE = ("▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎")
_AWS_SPIN = ("◐", "◓", "◑", "◒")


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _esc(text: Any) -> str:
    """Escape a value for safe embedding inside a Textual markup string."""
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


# MCP AWS tools look like mcp__aws__s3_list_objects or mcp__aws__call_aws.
_MCP_AWS_PREFIX = "mcp__aws__"


def _extract_aws_service(tool: str, params: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    Identify the AWS service and operation for a tool call.

    Returns (service, operation, source) where source is one of
    "mcp", "tool", or "cli", or None if this isn't an AWS call.
    service and operation are lowercase strings; service may be "" if
    only the operation is known.
    """
    tool_l = (tool or "").lower()

    # 1. MCP AWS tool: mcp__aws__<something>
    if tool_l.startswith(_MCP_AWS_PREFIX):
        rest = tool_l[len(_MCP_AWS_PREFIX):]
        # Known MCP AWS tool names: call_aws, s3_list_objects, ec2_describe_instances, ...
        if rest.startswith("call_aws"):
            op = str(params.get("operation") or "").strip().lower()
            if not op:
                return ("", "", "mcp")
            parts = op.split(None, 1)
            service = parts[0]
            operation = parts[1] if len(parts) > 1 else ""
            return (service, operation, "mcp")
        # s3_list_objects → service=s3, operation=list-objects
        m = re.match(r"^([a-z0-9]+)_(.+)$", rest)
        if m:
            service = m.group(1)
            operation = m.group(2).replace("_", "-")
            return (service, operation, "mcp")
        return (rest, "", "mcp")

    # 2. Native aws tool: aws or aws_*
    if tool_l == "aws" or tool_l.startswith("aws_"):
        # Prefer explicit service / operation params
        service = str(params.get("service") or "").strip().lower()
        operation = str(params.get("operation") or "").strip().lower()
        if service:
            return (service, operation, "tool")
        # mcp_tool param may name the real op
        mcp_tool = str(params.get("mcp_tool") or "").lower()
        if mcp_tool.startswith(_MCP_AWS_PREFIX):
            return _extract_aws_service(mcp_tool, params)
        # args list may carry the operation
        args = params.get("args") or []
        if isinstance(args, list) and args:
            first = str(args[0]).lower()
            return (first, "", "tool")
        return ("", "", "tool")

    # 3. Shell command starting with `aws`
    if tool_l in ("bash", "terminal", "shell"):
        cmd = str(params.get("command") or params.get("cmd") or "").strip()
        if not cmd:
            return None
        # Strip leading env assignments.
        toks = cmd.split()
        i = 0
        while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
            i += 1
        if i >= len(toks):
            return None
        if toks[i].rsplit("/", 1)[-1] != "aws":
            return None
        rest = toks[i + 1:]
        # Skip global flags like --profile x --region y --output z
        service = ""
        op = ""
        j = 0
        while j < len(rest):
            tok = rest[j]
            if tok.startswith("--"):
                j += 2
                continue
            service = tok.lower()
            j += 1
            break
        if j < len(rest):
            op = rest[j].lower()
        return (service, op, "cli")

    return None


def _service_color(service: str) -> str:
    if not service:
        return _AWS_FALLBACK_COLOR
    return _AWS_SERVICE_COLORS.get(service, _AWS_FALLBACK_COLOR)


def _summarize_aws_result(service: str, operation: str, result: Dict[str, Any]) -> str:
    """
    Produce a one-line summary for common AWS shapes. Falls back to
    `<service> <operation>` when the shape isn't recognized.
    """
    head = f"{service} {operation}".strip() if service else (operation or "aws")

    # Try to find a list in the parsed JSON.
    data = result.get("json") if isinstance(result, dict) else None
    if data is None and isinstance(result, dict):
        # Some paths stash the raw payload under `raw` or `result`
        for key in ("raw", "result"):
            cand = result.get(key)
            if isinstance(cand, dict):
                data = cand
                break

    if isinstance(data, dict):
        # Common "count" keys
        for key in (
            "Buckets", "Reservations", "Instances", "Functions",
            "Clusters", "Services", "Tasks", "Tables", "Topics",
            "Queues", "DBInstances", "Volumes", "Snapshots",
            "Users", "Roles", "Policies", "Aliases", "Distributions",
        ):
            if key in data and isinstance(data[key], list):
                return f"{head} → {len(data[key])} {key.lower()}"

        # Single-resource describes
        for key, label in (
            ("Account", "account"),
            ("UserId", "user"),
            ("Arn", "resource"),
            ("InstanceId", "instance"),
            ("BucketName", "bucket"),
        ):
            if key in data:
                return f"{head} → {label} {data[key]}"

    # Look for a top-level count field on the result itself
    if isinstance(result, dict):
        for key in ("count", "total", "size"):
            v = result.get(key)
            if isinstance(v, int):
                return f"{head} → {v}"

    return head


# ----------------------------------------------------------------------
# WIDGET
# ----------------------------------------------------------------------

class ToolCallWidget(Vertical):
    DEFAULT_CSS = f"""
    ToolCallWidget {{
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick {BORDER};
    }}
    ToolCallWidget.running    {{ border-left: thick {AMBER}; }}
    ToolCallWidget.done       {{ border-left: thick {GREEN_DIM}; }}
    ToolCallWidget.error      {{ border-left: thick {ERROR}; }}
    ToolCallWidget.cached     {{ border-left: thick {BORDER}; }}
    ToolCallWidget.background {{ border-left: thick {AMBER}; }}

    /* AWS card: amber left border in every state, so it stands out. */
    ToolCallWidget.aws        {{ border-left: thick {AWS_BASE}; }}
    ToolCallWidget.aws.running {{ border-left: thick {AWS_BASE}; }}
    ToolCallWidget.aws.done    {{ border-left: thick {AWS_BASE}; }}
    ToolCallWidget.aws.cached  {{ border-left: thick {AWS_BASE}; }}
    ToolCallWidget.aws.error   {{ border-left: thick {ERROR}; }}

    ToolCallWidget > Static {{
        height: auto;
        width: 100%;
    }}
    """

    def __init__(self, tool: str, params: Dict[str, Any], **kwargs: Any):
        super().__init__(**kwargs)
        self._tool = tool
        self._params = params or {}
        self._state = "pending"
        self._pulse_i = 0
        self._result: Optional[Dict[str, Any]] = None
        self._execution_time: Optional[float] = None
        self._cached = False
        self._expanded = False
        self._header_static: Optional[Static] = None
        self._body_static: Optional[Static] = None
        self._alive = True

        # AWS metadata computed once per widget.
        self._aws = _extract_aws_service(self._tool, self._params)
        self._is_aws = self._aws is not None
        if self._is_aws:
            service, operation, source = self._aws
            self._aws_service = service or ""
            self._aws_operation = operation or ""
            self._aws_source = source
        else:
            self._aws_service = ""
            self._aws_operation = ""
            self._aws_source = ""

    def compose(self) -> ComposeResult:
        self._header_static = Static(self._header_markup(), markup=True)
        self._body_static = Static("", markup=True)
        yield self._header_static
        yield self._body_static

    def on_mount(self) -> None:
        self.add_class("running")
        if self._is_aws:
            self.add_class("aws")
        self.set_interval(0.09, self._tick)

    def on_unmount(self) -> None:
        self._alive = False
        self._header_static = None
        self._body_static = None

    def _tick(self) -> None:
        if not self._alive:
            return
        if self._state not in ("pending", "running", "background"):
            return
        self._pulse_i += 1
        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def _set_state_class(self, state: str) -> None:
        for cls in ("running", "done", "error", "cached", "background"):
            try:
                self.remove_class(cls)
            except Exception:
                pass
        try:
            self.add_class(state)
        except Exception:
            pass

    def set_running(self, execution_time: Optional[float] = None) -> None:
        if not self._alive:
            return
        self._state = "running"
        self._execution_time = execution_time
        self._set_state_class("running")
        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False

    def set_result(
        self,
        result: Dict[str, Any],
        execution_time: Optional[float] = None,
        cached: bool = False,
    ) -> None:
        if not self._alive:
            return
        self._result = result
        self._execution_time = execution_time
        self._cached = cached

        if not isinstance(result, dict):
            self._state = "done"
        elif result.get("running"):
            self._state = "background"
        elif result.get("success", True):
            self._state = "cached" if cached else "done"
        else:
            self._state = "error"

        self._set_state_class(self._state)

        if self._header_static is not None:
            try:
                self._header_static.update(self._header_markup())
            except Exception:
                self._alive = False
                return
        if self._body_static is not None:
            try:
                self._body_static.update(self._render_result())
            except Exception:
                self._alive = False

    def toggle_expand(self) -> None:
        if not self._alive:
            return
        self._expanded = not self._expanded
        if self._body_static is not None:
            try:
                self._body_static.update(self._render_result())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------

    def _target(self) -> str:
        for key in ("command", "filePath", "path", "pattern", "query", "url"):
            val = self._params.get(key)
            if val:
                s = str(val)
                return s if len(s) <= 76 else s[:73] + "…"
        return ""

    def _chip(self) -> str:
        if self._state in ("pending", "running"):
            pulse = _PULSE[self._pulse_i % len(_PULSE)]
            return f"[{RUNNING}]{pulse}[/] [{MUTED}]running[/]"
        if self._state == "background":
            pid = _esc(self._result.get("pid") if isinstance(self._result, dict) else "?")
            return f"[{AMBER}]◉[/] [{MUTED}]pid {pid}[/]"
        if self._state == "cached":
            t = f"{self._execution_time:.2f}s" if self._execution_time is not None else ""
            return f"[{GREEN_DIM}]⚡ cached {t}[/]"
        if self._state == "done":
            t = f"{self._execution_time:.2f}s" if self._execution_time is not None else ""
            return f"[{GREEN_DIM}]✓ {t}[/]"
        return f"[{ERROR}]✗ failed[/]"

    def _aws_header(self) -> str:
        """
        Header for AWS cards. Shows the service and operation, plus a
        service-colored access spinner while running.
        """
        color = _service_color(self._aws_service)
        spin = _AWS_SPIN[self._pulse_i % len(_AWS_SPIN)] if self._state in ("pending", "running") else "●"

        if self._aws_service:
            head = f"{spin} [bold {color}]aws[/] [{DIM}]·[/] [bold {color}]{_esc(self._aws_service)}[/]"
        else:
            head = f"{spin} [bold {color}]working on aws account[/]"

        if self._aws_operation:
            head += f"  [{TEXT}]{_esc(self._aws_operation)}[/]"

        # Region / profile / identity, if present in params.
        region = self._params.get("region") or self._params.get("aws_region")
        profile = self._params.get("profile") or self._params.get("aws_profile")
        extras = []
        if region:
            extras.append(f"region={_esc(region)}")
        if profile:
            extras.append(f"profile={_esc(profile)}")
        if self._aws_source == "cli":
            extras.append("via cli")
        elif self._aws_source == "mcp":
            extras.append("via mcp")

        chip = self._chip()
        right = f"   {chip}"
        if extras:
            right = "   " + f"[{DIM}]" + "  ·  ".join(extras) + f"[/]  {chip}"

        return head + right

    def _header_markup(self) -> str:
        if self._is_aws:
            return self._aws_header()

        name = _esc(self._tool)
        target = _esc(self._target())
        left = f"[bold {GREEN}]{name}[/]"
        if target:
            left += f"  [{DIM}]·[/]  [{TEXT}]{target}[/]"
        chip = self._chip()
        return f"{left}   {chip}"

    # ------------------------------------------------------------------
    # BODY — per-tool renderers
    # ------------------------------------------------------------------

    def _render_result(self) -> str:
        result = self._result or {}

        # AWS bodies get a summary strip on top, then the normal render.
        if self._is_aws:
            return self._render_aws(result)

        if not isinstance(result, dict):
            return f"[{TEXT}]{_esc(result)[:800]}[/]"

        if not result.get("success", True):
            err = result.get("error") or "tool failed"
            return f"[{ERROR}]{_esc(err)[:800]}[/]"

        if self._tool in _DIFF_TOOLS:
            rendered = self._render_diff(result)
            if rendered:
                return rendered
        if self._tool in _BASH_TOOLS:
            return self._render_bash(result)
        if self._tool in _READ_TOOLS:
            return self._render_read(result)
        if self._tool in _SEARCH_TOOLS:
            return self._render_search(result)
        if self._tool in _WEB_TOOLS:
            return self._render_web(result)

        return self._render_generic(result)

    def _render_aws(self, result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return f"[{TEXT}]{_esc(result)[:800]}[/]"

        if not result.get("success", True):
            err = result.get("error") or "aws call failed"
            color = _service_color(self._aws_service)
            return (
                f"[{color}]┃[/] [{ERROR}]{_esc(err)[:800]}[/]"
            )

        summary = _summarize_aws_result(
            self._aws_service, self._aws_operation, result
        )
        color = _service_color(self._aws_service)
        lines = [f"[{color}]┃[/] [{TEXT}]{_esc(summary)}[/]"]

        # If the caller asked for the CLI command, show it under the summary.
        cmd = result.get("command")
        if cmd and self._aws_source == "cli":
            lines.append(f"[{DIM}]$ {_esc(cmd)}[/]")

        # Body: prefer JSON, then a text fallback, then the full dict.
        parsed = result.get("json")
        if parsed is None and isinstance(result.get("raw"), dict):
            parsed = result.get("raw")

        body_text = ""
        if parsed is not None:
            try:
                body_text = json.dumps(parsed, indent=2, default=str)
            except Exception:
                body_text = str(parsed)
        else:
            for key in ("stdout", "output", "content", "text", "result"):
                v = result.get(key)
                if isinstance(v, str) and v.strip():
                    body_text = v.strip()
                    break

        if body_text:
            lines.append("")
            body_lines = body_text.splitlines()
            show = body_lines if self._expanded else body_lines[:6]
            for ln in show:
                lines.append(f"[{TEXT}]{_esc(ln)}[/]")
            if not self._expanded and len(body_lines) > 6:
                hidden = len(body_lines) - 6
                lines.append(
                    f"[{AMBER}]… {hidden} more line"
                    f"{'s' if hidden != 1 else ''} · press e[/]"
                )
        else:
            lines.append(f"[{MUTED}]no output[/]")

        return "\n".join(lines)

    def _render_diff(self, result: Dict[str, Any]) -> str:
        before = result.get("before") or result.get("content_before") or ""
        after = (
            result.get("after")
            or result.get("content_after")
            or result.get("new_content")
            or ""
        )
        if not before or not after:
            return ""
        path = str(result.get("path") or "")
        markup = render_diff(before, after, path=path)
        if not markup:
            return ""
        return markup

    def _render_bash(self, result: Dict[str, Any]) -> str:
        command = _esc(result.get("command") or self._target())
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        exit_code = result.get("exit_code")

        lines = [f"[{DIM}]$[/] [{TEXT}]{command}[/]"]
        output = (stdout + ("\n" + stderr if stderr else "")).strip()
        if not output:
            lines.append(f"[{MUTED}](no output)[/]")
        else:
            out_lines = output.splitlines()
            show = out_lines if self._expanded else out_lines[:_BASH_PREVIEW_LINES]
            for ln in show:
                lines.append(f"[{TEXT}]{_esc(ln)}[/]")
            if not self._expanded and len(out_lines) > _BASH_PREVIEW_LINES:
                hidden = len(out_lines) - _BASH_PREVIEW_LINES
                lines.append(
                    f"[{AMBER}]… {hidden} more line"
                    f"{'s' if hidden != 1 else ''} · press e[/]"
                )
        if exit_code not in (None, 0):
            lines.append(f"[{ERROR}]exit {exit_code}[/]")
        return "\n".join(lines)

    def _render_read(self, result: Dict[str, Any]) -> str:
        path = _esc(result.get("path") or self._target())
        size = result.get("size")
        truncated = result.get("truncated")
        content = result.get("content") or ""
        size_str = f"[{DIM}] {size} bytes[/]" if size is not None else ""
        header = f"[{MUTED}]loaded[/] [{TEXT}]{path}[/]{size_str}"
        if truncated:
            header += f"  [{AMBER}]truncated[/]"
        if not content:
            return header
        lines = content.splitlines()
        show = lines if self._expanded else lines[:4]
        body = "\n".join(f"[{TEXT}]{_esc(ln)}[/]" for ln in show)
        if not self._expanded and len(lines) > 4:
            hidden = len(lines) - 4
            body += (
                f"\n[{AMBER}]… {hidden} more line"
                f"{'s' if hidden != 1 else ''} · press e[/]"
            )
        return header + "\n" + body

    def _render_search(self, result: Dict[str, Any]) -> str:
        count = result.get("count", 0)
        pattern = _esc(result.get("pattern") or self._target())
        if count == 0:
            return f"[{MUTED}]no matches for[/] [{TEXT}]'{pattern}'[/]"
        matches = result.get("matches") or []
        lines = [f"[{GREEN}]{count}[/] [{MUTED}]match{'es' if count != 1 else ''} for[/] [{TEXT}]'{pattern}'[/]"]
        for m in matches[:4]:
            if isinstance(m, dict):
                path = _esc(m.get("path", ""))
                text = _esc(m.get("text", "")[:70])
                lines.append(f"[{DIM}]{path}[/]  [{MUTED}]{text}[/]")
            else:
                lines.append(f"[{DIM}]{_esc(m)}[/]")
        if count > 4:
            lines.append(f"[{AMBER}]… {count - 4} more matches · press e[/]")
        return "\n".join(lines)

    def _render_web(self, result: Dict[str, Any]) -> str:
        status = result.get("status", "")
        url = _esc(result.get("url") or self._target())
        content = result.get("content") or ""
        if status:
            head = f"[{GREEN}]{status}[/]  [{DIM}]{url}[/]"
        else:
            head = f"[{DIM}]{url}[/]"
        if not content:
            return head
        preview = _esc(content.strip().replace("\n", " ")[:180])
        return head + f"\n[{TEXT}]{preview}[/]"

    def _render_generic(self, result: Dict[str, Any]) -> str:
        preview = (
            result.get("content")
            or result.get("output")
            or result.get("text")
            or result.get("result")
        )
        if preview is None:
            preview = {k: v for k, v in result.items() if k not in ("success",)}
        if isinstance(preview, (dict, list)):
            try:
                preview = json.dumps(preview, default=str)
            except Exception:
                preview = str(preview)
        preview = str(preview).strip()
        if not preview:
            return f"[{MUTED}]ok[/]"
        lines = preview.splitlines()
        show = lines if self._expanded else lines[:_GENERIC_PREVIEW_LINES]
        body = "\n".join(f"[{TEXT}]{_esc(ln)}[/]" for ln in show)
        if not self._expanded and len(lines) > _GENERIC_PREVIEW_LINES:
            hidden = len(lines) - _GENERIC_PREVIEW_LINES
            body += (
                f"\n[{AMBER}]… {hidden} more line"
                f"{'s' if hidden != 1 else ''} · press e[/]"
            )
        return body