"""Safe, paste-friendly MCP server configuration for the Textual UI.

Only explicit Save actions write settings. Connections are stored in the user
config; tokens and environment values are stored in the project .env file,
never in the repository or config JSON.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

from agent.config.loader import USER_CONFIG_PATH
from agent.utils.credential_store import credential_env_path, save_mcp_secrets
from agent.utils.env_manager import write_private_file


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


def list_mcp_servers(
    *,
    config_path: Path | None = None,
) -> list[dict]:
    """Return configured explicit servers and enabled presets, without secrets."""
    config_path = config_path or USER_CONFIG_PATH
    if not config_path.exists():
        return []
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(document, dict):
        return []
    current = document.get("mcp") or {}
    if not isinstance(current, dict):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    servers = current.get("servers") or []
    if isinstance(servers, list):
        for server in servers:
            if not isinstance(server, dict):
                continue
            name = str(server.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append({
                "name": name,
                "kind": "server",
                "enabled": bool(server.get("enabled", True)),
                "transport": str(server.get("transport") or ""),
            })

    presets = current.get("presets") or {}
    if isinstance(presets, dict):
        for key, preset in presets.items():
            if not isinstance(preset, dict) or not preset.get("enabled", False):
                continue
            name = str(preset.get("name") or key).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "kind": "preset", "enabled": True})
    return out


def _server_secret_keys(entry: dict, namespace: str) -> set[str]:
    keys: set[str] = set()
    api_key_env = entry.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env.startswith(namespace + "_"):
        keys.add(api_key_env)
    for field in ("env", "headers"):
        values = entry.get(field)
        if not isinstance(values, dict):
            continue
        for value in values.values():
            if not isinstance(value, str):
                continue
            if value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                if key.startswith(namespace + "_"):
                    keys.add(key)
    return keys


def delete_mcp_server(
    name: str,
    *,
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> dict:
    """Delete a configured server or disable its matching preset.

    Only auto-generated ``DEPRESSION_MCP_<SERVER>_*`` secrets are removed,
    and only when no remaining config entry references them.
    """
    config_path = config_path or USER_CONFIG_PATH
    env_path = env_path or credential_env_path()
    name = str(name or "").strip()
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid MCP server name")
    if not config_path.exists():
        raise ValueError("MCP server not found")

    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot safely edit existing MCP config JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("Existing config must be a JSON object")

    current = document.get("mcp") or {}
    if not isinstance(current, dict):
        raise ValueError("Existing 'mcp' section must be a JSON object")
    servers = current.get("servers") or []
    if not isinstance(servers, list):
        raise ValueError("Existing 'mcp.servers' must be a list")

    removed = next(
        (s for s in servers if isinstance(s, dict) and s.get("name") == name),
        None,
    )
    kind = "server"
    secret_keys: set[str] = set()

    if removed is not None:
        updated_servers = [
            server for server in servers
            if not (isinstance(server, dict) and server.get("name") == name)
        ]
        updated_mcp = {**current, "servers": updated_servers}
        document["mcp"] = updated_mcp
        namespace = "DEPRESSION_MCP_" + re.sub(
            r"[^A-Z0-9_]", "_", name.upper()
        )
        secret_keys = _server_secret_keys(removed, namespace)
    else:
        presets = current.get("presets") or {}
        if not isinstance(presets, dict):
            raise ValueError("MCP server not found")
        preset_key = next(
            (
                key for key, preset in presets.items()
                if isinstance(preset, dict)
                and preset.get("enabled", False)
                and str(preset.get("name") or key) == name
            ),
            None,
        )
        if preset_key is None:
            raise ValueError("MCP server not found")
        kind = "preset"
        updated_presets = dict(presets)
        updated_presets[preset_key] = {
            **presets[preset_key], "enabled": False,
        }
        updated_mcp = {**current, "presets": updated_presets}
        document["mcp"] = updated_mcp

    remaining = json.dumps(document, ensure_ascii=False)
    removable = sorted(
        key for key in secret_keys if ("${" + key + "}") not in remaining
    )
    write_private_file(
        config_path, json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        mode=0o600,
    )

    cleanup_error = None
    if removable:
        try:
            save_mcp_secrets(
                {key: "" for key in removable},
                env_path=env_path,
                remove_empty=True,
            )
        except Exception as exc:
            cleanup_error = type(exc).__name__

    return {
        "name": name,
        "kind": kind,
        "config": updated_mcp,
        "secrets_removed": [] if cleanup_error else removable,
        "cleanup_error": cleanup_error,
    }


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
    env_path = env_path or credential_env_path()
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
        if text.startswith("${") and text.endswith("}") and _ENV_NAME.fullmatch(text[2:-1]):
            return text
        env_key = f"{namespace}_{suffix}"
        secrets[env_key] = text
        return "${" + env_key + "}"

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
        if token.startswith("${") and token.endswith("}") and _ENV_NAME.fullmatch(token[2:-1]):
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
        save_mcp_secrets(secrets, env_path=env_path)
    write_private_file(
        config_path, json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        mode=0o600,
    )

    # Return sanitized server details, never credentials.
    return {"name": name, "transport": transport, "config": updated_mcp}
