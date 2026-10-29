"""Safe, paste-friendly MCP server configuration for the Textual UI.

Only explicit Save actions write settings. Connections are stored in the user
config; tokens and environment values are stored in the project's private .env
file, never in config JSON. Keep .env ignored by Git.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

from agent.config.loader import USER_CONFIG_PATH
from agent.utils.env_manager import find_env_file, write_env_file, write_private_file


_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRANSPORTS = {"streamable-http", "sse", "cloud", "stdio"}


def _parse_pasted_server(raw: str) -> Dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Pasted MCP configuration must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Paste a JSON server object")

    if "mcpServers" in data or "servers" in data:
        servers = data.get("mcpServers", data.get("servers"))
        if isinstance(servers, dict):
            if len(servers) != 1:
                raise ValueError("Paste one MCP server at a time")
            name, entry = next(iter(servers.items()))
        elif isinstance(servers, list):
            if len(servers) != 1:
                raise ValueError("Paste one MCP server at a time")
            entry = servers[0]
            name = entry.get("name", "") if isinstance(entry, dict) else ""
        else:
            raise ValueError("'mcpServers' must be a name-to-server mapping")
        if not isinstance(entry, dict):
            raise ValueError("The selected MCP server must be a JSON object")
        return {"name": name, **entry}
    return data


def _transport(raw: str, entry: Dict[str, Any]) -> str:
    kind = str(entry.get("transport") or entry.get("type") or raw or "").lower()
    if kind in ("http", "streamable_http"):
        kind = "streamable-http"
    if not kind:
        kind = "stdio" if entry.get("command") else "streamable-http"
    if kind not in _TRANSPORTS:
        raise ValueError("Transport must be HTTP, SSE, cloud, or stdio")
    return kind


def _safe_url(value: str) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("MCP URL must not contain credentials or a fragment")
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValueError("Paste a complete http(s) MCP URL")
        if parsed.scheme == "http" and parsed.hostname.lower() not in (
            "localhost", "127.0.0.1", "::1",
        ):
            raise ValueError("Use HTTPS for non-local MCP servers")
    except ValueError:
        raise
    return url


def _safe_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a single-line string")
    return value


def _parse_args(value: Any) -> list[str]:
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            raise ValueError("MCP arguments must be a list of strings")
        return [_safe_value(v, "Argument") for v in value]
    if not value:
        return []
    value = str(value).strip()
    if value.startswith("["):
        try:
            return _parse_args(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError("Arguments must be a JSON array or shell-style words") from exc
    try:
        return shlex.split(value)
    except ValueError as exc:
        raise ValueError("Invalid arguments: unmatched quote") from exc


def save_mcp_server(
    values: Dict[str, str],
    *,
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> dict:
    """Validate and persist a single server, preserving all unrelated config.

    Supports manual URL/token/stdio fields and Claude-style mcpServers JSON.
    Never interprets a pasted string as a shell command.
    """
    config_path = config_path or USER_CONFIG_PATH
    env_path = env_path or find_env_file()
    pasted = _parse_pasted_server(str(values.get("json") or ""))

    name = str(pasted.get("name") or values.get("name") or "").strip()
    if not _NAME.fullmatch(name):
        raise ValueError("Server name: use 1–64 letters, numbers, '-' or '_'")

    if config_path.exists():
        try:
            document = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot safely edit existing MCP config JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("Existing config must be a JSON object")
    else:
        document = {}

    current = document.get("mcp") or {}
    if not isinstance(current, dict):
        raise ValueError("Existing 'mcp' section must be a JSON object")
    existing_servers = current.get("servers") or []
    if not isinstance(existing_servers, list):
        raise ValueError("Existing 'mcp.servers' must be a list")
    for server in existing_servers:
        if not isinstance(server, dict):
            raise ValueError("Existing MCP server entry must be an object")

    previous = next((s for s in existing_servers if s.get("name") == name), {})
    if not pasted and not any(
        str(values.get(key) or "").strip() for key in ("url", "command")
    ):
        raise ValueError("Paste an MCP URL, a stdio command, or server JSON")

    transport = _transport(str(values.get("transport") or "streamable-http"), pasted)
    if pasted.get("command") and not pasted.get("transport") and not pasted.get("type"):
        transport = "stdio"
    if pasted.get("url") and not pasted.get("transport") and not pasted.get("type"):
        transport = "streamable-http"

    entry = {k: v for k, v in previous.items() if k not in (
        "transport", "url", "command", "args", "env", "headers",
        "api_key", "api_key_env",
    )}
    entry.update({"name": name, "transport": transport, "enabled": True})

    if transport == "stdio":
        cmd = _safe_value(
            str(pasted.get("command") or values.get("command") or ""), "Command"
        ).strip()
        if not cmd:
            raise ValueError("Paste the MCP server executable (e.g. npx)")
        if any(ch.isspace() for ch in cmd):
            raise ValueError("Command must be an executable; put flags in Arguments")
        entry["command"] = cmd
        entry["args"] = _parse_args(
            pasted.get("args", values.get("args") or "")
        )
    else:
        url = pasted.get("url") or values.get("url")
        entry["url"] = _safe_url(url)
        if transport == "cloud":
            entry["transport"] = "cloud"

    namespace = "DEPRESSION_MCP_" + re.sub(r"[^A-Z0-9_]", "_", name.upper())
    secrets: Dict[str, str] = {}

    def private(value: Any, label: str, suffix: str) -> str:
        text = _safe_value(value, label)
        if text.startswith(") and text.endswith(") and _ENV_NAME.fullmatch(text[2:-1]):
            return text
        env_key = f"{namespace}_{suffix}"
        secrets[env_key] = text
        return " + env_key + "

    # Preset or pasted stdio env vars are private, even if they contain tokens.
    env_data = pasted.get("env", {})
    if env_data and not isinstance(env_data, dict):
        raise ValueError("MCP 'env' must be a JSON object")
    if transport == "stdio":
        env = dict(previous.get("env") or {})
        for env_name, env_val in env_data.items():
            if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
                raise ValueError("Invalid environment variable name")
            env[env_name] = private(env_val, "Environment value", "ENV_" + env_name.upper())
        if env:
            entry["env"] = env

    # API key can be entered separately, or pasted as a plain JSON 'api_key'.
    token = str(pasted.get("api_key") or values.get("api_key") or "").strip()
    if token:
        if transport == "stdio":
            raise ValueError("Use the pasted JSON 'env' object for stdio secrets")
        entry["api_key_env"] = namespace + "_API_KEY"
        if token.startswith(") and token.endswith(") and _ENV_NAME.fullmatch(token[2:-1]):
            entry["api_key_env"] = token[2:-1]
        else:
            secrets[entry["api_key_env"]] = _safe_value(token, "API token")
    elif previous.get("api_key_env") and transport != "stdio":
        entry["api_key_env"] = previous["api_key_env"]

    headers = pasted.get("headers", {})
    if headers and not isinstance(headers, dict):
        raise ValueError("MCP headers must be a JSON object")
    if transport != "stdio":
        safe_headers = dict(previous.get("headers") or {})
        for header, value in headers.items():
            if not isinstance(header, str) or not re.fullmatch(r"[A-Za-z0-9-]+", header):
                raise ValueError("Invalid MCP header name")
            safe_headers[header] = private(value, "Header", "HEADER_" + header.upper().replace("-", "_"))
        if safe_headers:
            entry["headers"] = safe_headers
    if "Authorization" in (entry.get("headers") or {}):
        # Authorization header always wins over separate API key in MCPClient.
        entry.pop("api_key_env", None)

    updated_servers = [
        server for server in existing_servers if server.get("name") != name
    ]
    updated_servers.append(entry)
    updated_mcp = {**current, "enabled": True, "servers": updated_servers}
    document["mcp"] = updated_mcp

    if secrets:
        write_env_file(secrets, env_path, strict=True)
    write_private_file(
        config_path, json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        mode=0o600,
    )
    for key, value in secrets.items():
        os.environ[key] = value

    # Return sanitized server details, never credentials.
    return {"name": name, "transport": transport, "config": updated_mcp}
