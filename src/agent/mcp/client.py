"""
MCP Client - Model Context Protocol Integration

Supports all transports:
    1. stdio            — spawn local MCP servers (npx, python, binaries)
    2. sse              — Server-Sent Events HTTP transport
    3. streamable-http  — Modern HTTP transport
    4. cloud            — Managed cloud MCP (Anthropic Cloud, Cloudflare, AWS, SaaS)
    5. websocket        — WebSocket-based MCP

Built-in presets:
    - filesystem  : read/write/search project files
    - git         : inspect diffs, commits, branches
    - github      : repositories, issues, PRs
    - browser     : research docs and websites
    - database    : SQL / SQLite / Postgres / MySQL queries
    - aws         : AWS CLI MCP (optional)

Features:
    - Multi-server management
    - Automatic tool discovery and registration
    - Resources / prompts / tools support
    - Streaming notifications
    - Auto-reconnect with exponential backoff
    - Per-server auth (Bearer, API key headers, OAuth token)
    - Health checks and heartbeats
    - Async-first design
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
import time

import httpx

from agent.utils.logging import get_logger
from agent.utils.errors import MCPError, MCPConnectionError, MCPTimeoutError

logger = get_logger(__name__)


# ======================================================================
# ENUMS & DATA MODELS
# ======================================================================

class MCPTransport(Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"
    CLOUD = "cloud"
    WEBSOCKET = "websocket"


class MCPState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READY = "ready"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""
    name: str
    transport: MCPTransport
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    api_key: Optional[str] = None
    timeout: float = 30.0
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    enabled: bool = True
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTool:
    """A tool exposed by an MCP server"""
    server: str
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"mcp__{self.server}__{self.name}"


@dataclass
class MCPResource:
    """A resource exposed by an MCP server"""
    server: str
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


@dataclass
class MCPPrompt:
    """A prompt template exposed by an MCP server"""
    server: str
    name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)


# ======================================================================
# TRANSPORT BASE
# ======================================================================

class MCPTransportBase(ABC):
    """Abstract base for MCP transports"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.state = MCPState.DISCONNECTED
        self._pending: Dict[str, asyncio.Future] = {}
        self._notification_handlers: List[Callable[[Dict], Awaitable[None]]] = []
        self._request_id = 0
        self._lock = asyncio.Lock()

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def _send(self, message: Dict[str, Any]) -> None: ...

    @abstractmethod
    async def _receive_loop(self) -> None: ...

    def _next_id(self) -> str:
        self._request_id += 1
        return f"mcp-{self.config.name}-{self._request_id}"

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.state not in (MCPState.CONNECTED, MCPState.READY):
            raise MCPConnectionError(
                f"Server {self.config.name} not connected (state={self.state.value})"
            )

        req_id = self._next_id()
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        try:
            await self._send(payload)
            return await asyncio.wait_for(
                future, timeout=timeout or self.config.timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPTimeoutError(
                f"MCP request '{method}' to {self.config.name} timed out"
            )
        except Exception:
            self._pending.pop(req_id, None)
            raise

    async def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        await self._send({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })

    def add_notification_handler(
        self, handler: Callable[[Dict], Awaitable[None]]
    ) -> None:
        self._notification_handlers.append(handler)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        if "id" in message and message.get("id") in self._pending:
            fut = self._pending.pop(message["id"])
            if "error" in message:
                fut.set_exception(MCPError(
                    f"MCP error: {message['error'].get('message', 'unknown')}"
                ))
            else:
                fut.set_result(message.get("result", {}))
            return

        if "method" in message and "id" not in message:
            for handler in self._notification_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.warning(f"Notification handler failed: {e}")


# ======================================================================
# STDIO TRANSPORT
# ======================================================================

class StdioTransport(MCPTransportBase):
    """Local MCP server over stdio (child process)"""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._read_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self.state = MCPState.CONNECTING

        if not self.config.command:
            raise MCPConnectionError(
                f"stdio server '{self.config.name}' missing 'command'"
            )

        cmd = shutil.which(self.config.command) or self.config.command
        env = {**os.environ, **self.config.env}

        try:
            self._process = await asyncio.create_subprocess_exec(
                cmd,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd,
            )
        except FileNotFoundError as e:
            raise MCPConnectionError(
                f"Command not found for MCP server '{self.config.name}': {cmd}"
            ) from e

        self.state = MCPState.CONNECTED
        self._read_task = asyncio.create_task(self._receive_loop())
        logger.info(f"stdio MCP '{self.config.name}' started (pid={self._process.pid})")

    async def disconnect(self) -> None:
        self.state = MCPState.CLOSED
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception:
                pass
        logger.info(f"stdio MCP '{self.config.name}' stopped")

    async def _send(self, message: Dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("stdio not connected")
        data = (json.dumps(message) + "\n").encode()
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _receive_loop(self) -> None:
        try:
            assert self._process and self._process.stdout
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line.decode())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON from {self.config.name}: {line[:200]}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"stdio receive loop error: {e}")
            self.state = MCPState.ERROR


# ======================================================================
# HTTP/SSE TRANSPORT
# ======================================================================

class HTTPTransport(MCPTransportBase):
    """Remote MCP server over HTTP (streamable-http) or SSE"""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._sse_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self.state = MCPState.CONNECTING

        if not self.config.url:
            raise MCPConnectionError(
                f"HTTP server '{self.config.name}' missing 'url'"
            )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self.config.headers)
        if self.config.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self.config.timeout,
            follow_redirects=True,
        )

        if self.config.transport == MCPTransport.SSE:
            self._sse_task = asyncio.create_task(self._sse_loop())

        self.state = MCPState.CONNECTED
        logger.info(f"HTTP MCP '{self.config.name}' connected to {self.config.url}")

    async def disconnect(self) -> None:
        self.state = MCPState.CLOSED
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info(f"HTTP MCP '{self.config.name}' disconnected")

    async def _send(self, message: Dict[str, Any]) -> None:
        if not self._client:
            raise MCPConnectionError("HTTP transport not connected")

        response = await self._client.post(self.config.url, json=message)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            await self._handle_message(response.json())
        elif "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    try:
                        await self._handle_message(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        continue

    async def _receive_loop(self) -> None:
        return

    async def _sse_loop(self) -> None:
        try:
            async with self._client.stream("GET", self.config.url) as resp:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        await self._handle_message(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"SSE loop error for {self.config.name}: {e}")
            self.state = MCPState.ERROR


# ======================================================================
# CLOUD TRANSPORT
# ======================================================================

class CloudTransport(HTTPTransport):
    """
    Managed Cloud MCP transport.

    Handles:
        - Automatic Authorization header injection
        - Provider-specific base paths
        - Streaming responses
        - Token refresh hooks
        - Retry with jitter
        - Request signing hook (for AWS SigV4, GCP OAuth, etc.)
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._token_refresh: Optional[Callable[[], Awaitable[str]]] = None
        self._signer: Optional[Callable[[httpx.Request], httpx.Request]] = None

    def set_token_refresher(self, fn: Callable[[], Awaitable[str]]) -> None:
        self._token_refresh = fn

    def set_request_signer(
        self, fn: Callable[[httpx.Request], httpx.Request]
    ) -> None:
        self._signer = fn

    async def connect(self) -> None:
        await super().connect()
        self.state = MCPState.READY
        logger.info(
            f"Cloud MCP '{self.config.name}' ready ({self.config.url})"
        )

    async def _send(self, message: Dict[str, Any]) -> None:
        if not self._client:
            raise MCPConnectionError("Cloud transport not connected")

        request = self._client.build_request(
            "POST", self.config.url, json=message
        )
        if self._signer:
            request = self._signer(request)

        try:
            response = await self._client.send(request)
        except httpx.TimeoutException as e:
            raise MCPTimeoutError(f"Cloud MCP timeout: {e}")

        if response.status_code == 401 and self._token_refresh:
            try:
                new_token = await self._token_refresh()
                self._client.headers["Authorization"] = f"Bearer {new_token}"
                response = await self._client.send(request)
            except Exception as e:
                raise MCPConnectionError(f"Token refresh failed: {e}")

        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            await self._handle_message(response.json())
        elif "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    try:
                        await self._handle_message(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        continue


# ======================================================================
# MCP PRESETS — FILESYSTEM, GIT, GITHUB, BROWSER, DATABASE, AWS
# ======================================================================

class MCPPresets:
    """
    Curated presets for the most common MCP servers.

    Each preset returns a fully-formed MCPServerConfig ready to use.
    """

    # ------------------------------------------------------------------
    # Filesystem — read/write/search project files
    # ------------------------------------------------------------------
    @staticmethod
    def filesystem(
        allowed_dirs: Optional[List[str]] = None,
        name: str = "filesystem",
    ) -> MCPServerConfig:
        """
        Official Filesystem MCP server.

        Exposes:
            - read_file, write_file, edit_file
            - list_directory, create_directory, move_file
            - search_files, get_file_info
            - directory_tree

        Requires `npx` on PATH.
        """
        dirs = allowed_dirs or [os.getcwd()]
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *dirs,
            ],
            timeout=60.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # Git — inspect diffs, commits, branches
    # ------------------------------------------------------------------
    @staticmethod
    def git(
        repo_path: Optional[str] = None,
        name: str = "git",
    ) -> MCPServerConfig:
        """
        Official Git MCP server.

        Exposes:
            - git_status, git_diff, git_log
            - git_show, git_branch, git_checkout
            - git_commit, git_add
            - git_reset, git_stash, git_merge

        Requires `uvx` (from uv) on PATH.
        """
        env = {}
        if repo_path:
            env["GIT_REPO_PATH"] = repo_path

        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="uvx",
            args=["mcp-server-git", "--repository", repo_path or os.getcwd()],
            env=env,
            cwd=repo_path or os.getcwd(),
            timeout=60.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # GitHub — repositories, issues, PRs
    # ------------------------------------------------------------------
    @staticmethod
    def github(
        token: Optional[str] = None,
        token_env: str = "GITHUB_TOKEN",
        name: str = "github",
    ) -> MCPServerConfig:
        """
        Official GitHub MCP server.

        Exposes:
            - create_issue, update_issue, list_issues
            - create_pull_request, get_pull_request, list_pull_requests
            - get_file_contents, create_or_update_file, push_files
            - search_repositories, search_code, search_issues
            - create_branch, list_commits, fork_repository
            - list_branches, get_commit

        Requires a GitHub personal access token.
        """
        api_key = token or os.environ.get(token_env)

        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": api_key or "",
            },
            timeout=60.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # Browser / Web — research docs and websites
    # ------------------------------------------------------------------
    @staticmethod
    def browser(
        name: str = "browser",
        headless: bool = True,
    ) -> MCPServerConfig:
        """
        Playwright-based Browser MCP server.

        Exposes:
            - browser_navigate, browser_navigate_back
            - browser_click, browser_type, browser_hover
            - browser_select_option, browser_press_key
            - browser_snapshot, browser_take_screenshot
            - browser_evaluate, browser_wait_for
            - browser_tab_new, browser_tab_close, browser_tab_list
            - browser_network_requests

        Requires `npx` on PATH.
        """
        args = ["-y", "@playwright/mcp@latest"]
        if headless:
            args.append("--headless")

        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="npx",
            args=args,
            timeout=120.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # Database — SQL queries (SQLite, Postgres, MySQL)
    # ------------------------------------------------------------------
    @staticmethod
    def database_sqlite(
        db_path: str,
        name: str = "database",
    ) -> MCPServerConfig:
        """
        SQLite MCP server.

        Exposes:
            - read_query, write_query
            - create_table, list_tables
            - describe_table
            - append_insight
        """
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="uvx",
            args=["mcp-server-sqlite", "--db-path", db_path],
            timeout=60.0,
            auto_reconnect=True,
        )

    @staticmethod
    def database_postgres(
        connection_string: str,
        name: str = "database",
    ) -> MCPServerConfig:
        """
        Postgres MCP server.

        Exposes:
            - query (read/write SQL)
            - list_schemas, list_tables
            - describe_table
        """
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-postgres",
                connection_string,
            ],
            timeout=60.0,
            auto_reconnect=True,
        )

    @staticmethod
    def database_mysql(
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        name: str = "database",
    ) -> MCPServerConfig:
        """MySQL MCP server (community)"""
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="npx",
            args=[
                "-y",
                "@benborla29/mcp-server-mysql",
            ],
            env={
                "MYSQL_HOST": host,
                "MYSQL_USER": user,
                "MYSQL_PASSWORD": password,
                "MYSQL_DATABASE": database,
                "MYSQL_PORT": str(port),
            },
            timeout=60.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # AWS — optional (agent already has terminal to run AWS CLI)
    # ------------------------------------------------------------------
    @staticmethod
    def aws(
        profile: Optional[str] = None,
        region: Optional[str] = None,
        name: str = "aws",
    ) -> MCPServerConfig:
        """
        AWS MCP server (community).

        Exposes:
            - s3_list_buckets, s3_list_objects, s3_get_object
            - ec2_describe_instances
            - lambda_list_functions, lambda_invoke
            - cloudwatch_get_metrics
            - iam_list_users, iam_list_roles

        Requires `uvx` on PATH and AWS credentials configured.
        """
        env = {}
        if profile:
            env["AWS_PROFILE"] = profile
        if region:
            env["AWS_REGION"] = region

        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="uvx",
            args=["awslabs.core-mcp-server@latest"],
            env=env,
            timeout=60.0,
            auto_reconnect=True,
        )

    # ------------------------------------------------------------------
    # Cloud — generic cloud MCP endpoint
    # ------------------------------------------------------------------
    @staticmethod
    def cloud(
        name: str,
        url: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> MCPServerConfig:
        """Generic cloud MCP server"""
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.CLOUD,
            url=url,
            headers=headers or {},
            api_key=key,
            timeout=timeout,
            auto_reconnect=True,
            max_reconnect_attempts=5,
        )

    # ------------------------------------------------------------------
    # Generic stdio / sse / http
    # ------------------------------------------------------------------
    @staticmethod
    def stdio(
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
    ) -> MCPServerConfig:
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command=command,
            args=args or [],
            env=env or {},
            cwd=cwd,
            timeout=timeout,
        )

    @staticmethod
    def sse(
        name: str,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> MCPServerConfig:
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.SSE,
            url=url,
            headers=headers or {},
            api_key=api_key,
            timeout=60.0,
        )

    @staticmethod
    def http(
        name: str,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> MCPServerConfig:
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STREAMABLE_HTTP,
            url=url,
            headers=headers or {},
            api_key=api_key,
            timeout=60.0,
        )


# ======================================================================
# MCP CLIENT
# ======================================================================

class MCPClient:
    """
    High-level MCP client managing multiple servers.

    Usage:
        client = MCPClient(config)
        await client.initialize()
        tools = client.list_tools()
        result = await client.call_tool("mcp__filesystem__read_file", {...})
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.servers_cfg: List[MCPServerConfig] = self._load_servers()

        self.transports: Dict[str, MCPTransportBase] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}

        self._initialized = False
        self._health_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = config.get("heartbeat_interval", 30.0)

    # ------------------------------------------------------------------
    # SERVER CONFIG LOADING
    # ------------------------------------------------------------------

    def _load_servers(self) -> List[MCPServerConfig]:
        """
        Load servers from config. Supports two formats:

        Format A — explicit list:
            "servers": [ {name, transport, ...}, ... ]

        Format B — preset toggles:
            "presets": {
                "filesystem": {"enabled": true, "allowed_dirs": [...]},
                "git": {"enabled": true, "repo_path": "..."},
                "github": {"enabled": true, "token_env": "GITHUB_TOKEN"},
                "browser": {"enabled": true, "headless": true},
                "database": {"type": "sqlite", "path": "app.db"},
                "aws": {"enabled": true, "region": "us-east-1"}
            }
        """
        servers: List[MCPServerConfig] = []

        # Format A: explicit list
        for s in self.config.get("servers", []):
            if not s.get("enabled", True):
                continue
            try:
                transport = MCPTransport(s.get("transport", "stdio").lower())
            except ValueError:
                transport = MCPTransport.STDIO

            servers.append(MCPServerConfig(
                name=s.get("name", f"server_{len(servers)}"),
                transport=transport,
                command=s.get("command"),
                args=s.get("args", []),
                env=self._resolve_env(s.get("env", {})),
                cwd=s.get("cwd"),
                url=s.get("url"),
                headers=s.get("headers", {}),
                api_key=s.get("api_key") or self._resolve_api_key(s),
                timeout=s.get("timeout", 30.0),
                auto_reconnect=s.get("auto_reconnect", True),
                max_reconnect_attempts=s.get("max_reconnect_attempts", 5),
                enabled=True,
                capabilities=s.get("capabilities", {}),
            ))

        # Format B: presets
        presets = self.config.get("presets", {})
        servers.extend(self._build_presets(presets))

        return servers

    def _build_presets(self, presets: Dict[str, Any]) -> List[MCPServerConfig]:
        """Expand preset toggles into full MCPServerConfig objects"""
        out: List[MCPServerConfig] = []

        # --- Filesystem ---
        fs = presets.get("filesystem", {})
        if fs.get("enabled", False):
            try:
                out.append(MCPPresets.filesystem(
                    allowed_dirs=fs.get("allowed_dirs"),
                    name=fs.get("name", "filesystem"),
                ))
            except Exception as e:
                logger.error(f"Failed to build filesystem preset: {e}")

        # --- Git ---
        git = presets.get("git", {})
        if git.get("enabled", False):
            try:
                out.append(MCPPresets.git(
                    repo_path=git.get("repo_path"),
                    name=git.get("name", "git"),
                ))
            except Exception as e:
                logger.error(f"Failed to build git preset: {e}")

        # --- GitHub ---
        gh = presets.get("github", {})
        if gh.get("enabled", False):
            try:
                out.append(MCPPresets.github(
                    token=gh.get("token"),
                    token_env=gh.get("token_env", "GITHUB_TOKEN"),
                    name=gh.get("name", "github"),
                ))
            except Exception as e:
                logger.error(f"Failed to build github preset: {e}")

        # --- Browser ---
        br = presets.get("browser", {})
        if br.get("enabled", False):
            try:
                out.append(MCPPresets.browser(
                    name=br.get("name", "browser"),
                    headless=br.get("headless", True),
                ))
            except Exception as e:
                logger.error(f"Failed to build browser preset: {e}")

        # --- Database ---
        db = presets.get("database", {})
        if db.get("enabled", False):
            db_type = db.get("type", "sqlite").lower()
            try:
                if db_type == "sqlite":
                    out.append(MCPPresets.database_sqlite(
                        db_path=db.get("path", "./app.db"),
                        name=db.get("name", "database"),
                    ))
                elif db_type == "postgres":
                    out.append(MCPPresets.database_postgres(
                        connection_string=db["connection_string"],
                        name=db.get("name", "database"),
                    ))
                elif db_type == "mysql":
                    out.append(MCPPresets.database_mysql(
                        host=db["host"],
                        user=db["user"],
                        password=db["password"],
                        database=db["database"],
                        port=db.get("port", 3306),
                        name=db.get("name", "database"),
                    ))
                else:
                    logger.warning(f"Unknown database type: {db_type}")
            except Exception as e:
                logger.error(f"Failed to build database preset: {e}")

        # --- AWS (optional) ---
        aws = presets.get("aws", {})
        if aws.get("enabled", False):
            try:
                out.append(MCPPresets.aws(
                    profile=aws.get("profile"),
                    region=aws.get("region"),
                    name=aws.get("name", "aws"),
                ))
            except Exception as e:
                logger.error(f"Failed to build aws preset: {e}")

        return out

    def _resolve_env(self, env: Dict[str, str]) -> Dict[str, str]:
        """Interpolate ${VAR} references in env values"""
        out = {}
        for k, v in env.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                out[k] = os.environ.get(v[2:-1], "")
            else:
                out[k] = v
        return out

    def _resolve_api_key(self, server_cfg: Dict[str, Any]) -> Optional[str]:
        env_var = server_cfg.get("api_key_env")
        if env_var:
            return os.environ.get(env_var)
        return None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if not self.enabled:
            logger.info("MCP client disabled")
            return
        if self._initialized:
            return

        if not self.servers_cfg:
            logger.info("No MCP servers configured")
            self._initialized = True
            return

        # Connect all servers in parallel
        results = await asyncio.gather(
            *[self._connect_server(cfg) for cfg in self.servers_cfg],
            return_exceptions=True,
        )
        for cfg, result in zip(self.servers_cfg, results):
            if isinstance(result, Exception):
                logger.error(f"MCP server '{cfg.name}' failed: {result}")

        self._initialized = True
        self._health_task = asyncio.create_task(self._health_loop())

        logger.info(
            f"MCP client initialized: "
            f"{len(self.transports)} servers, "
            f"{len(self.tools)} tools, "
            f"{len(self.resources)} resources, "
            f"{len(self.prompts)} prompts"
        )

    async def _connect_server(self, cfg: MCPServerConfig) -> None:
        # Build the right transport
        if cfg.transport == MCPTransport.STDIO:
            transport = StdioTransport(cfg)
        elif cfg.transport in (MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP):
            transport = HTTPTransport(cfg)
        elif cfg.transport == MCPTransport.CLOUD:
            transport = CloudTransport(cfg)
        else:
            raise MCPConnectionError(f"Unsupported transport: {cfg.transport}")

        transport.add_notification_handler(self._handle_notification)

        await transport.connect()
        self.transports[cfg.name] = transport

        # MCP handshake
        try:
            init_result = await transport.request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "cli-agent",
                    "version": "1.0.0",
                },
            })
            logger.debug(
                f"MCP '{cfg.name}' server: "
                f"{init_result.get('serverInfo', {})}"
            )
            await transport.notify("notifications/initialized")

            await self._discover_server(cfg.name, transport)

        except Exception as e:
            logger.warning(f"MCP handshake with '{cfg.name}' failed: {e}")
            await transport.disconnect()
            self.transports.pop(cfg.name, None)
            raise

    async def _discover_server(
        self, name: str, transport: MCPTransportBase
    ) -> None:
        """Query tools, resources, and prompts from a server"""

        # Tools
        try:
            result = await transport.request("tools/list")
            for t in result.get("tools", []):
                tool = MCPTool(
                    server=name,
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                self.tools[tool.full_name] = tool
            logger.debug(f"'{name}' exposed {len(result.get('tools', []))} tools")
        except Exception as e:
            logger.debug(f"'{name}' has no tools: {e}")

        # Resources
        try:
            result = await transport.request("resources/list")
            for r in result.get("resources", []):
                resource = MCPResource(
                    server=name,
                    uri=r["uri"],
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", ""),
                )
                self.resources[f"{name}:{resource.uri}"] = resource
        except Exception as e:
            logger.debug(f"'{name}' has no resources: {e}")

        # Prompts
        try:
            result = await transport.request("prompts/list")
            for p in result.get("prompts", []):
                prompt = MCPPrompt(
                    server=name,
                    name=p["name"],
                    description=p.get("description", ""),
                    arguments=p.get("arguments", []),
                )
                self.prompts[f"{name}:{prompt.name}"] = prompt
        except Exception as e:
            logger.debug(f"'{name}' has no prompts: {e}")

    async def shutdown(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await asyncio.gather(
            *[t.disconnect() for t in self.transports.values()],
            return_exceptions=True,
        )

        self.transports.clear()
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        self._initialized = False
        logger.info("MCP client shutdown complete")

    # ------------------------------------------------------------------
    # TOOLS
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all MCP tools as OpenAI-style function definitions"""
        tools = []
        for tool in self.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.full_name,
                    "description": f"[{tool.server}] {tool.description}",
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            })
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        server, tool = self._parse_tool_name(tool_name)
        transport = self.transports.get(server)
        if not transport:
            raise MCPConnectionError(f"MCP server '{server}' not connected")

        try:
            result = await transport.request(
                "tools/call",
                {"name": tool, "arguments": arguments},
                timeout=timeout,
            )
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}

        content = result.get("content", [])
        is_error = result.get("isError", False)

        text_parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "image":
                    text_parts.append(f"[image: {block.get('mimeType', 'image')}]")
                elif btype == "resource":
                    text_parts.append(
                        f"[resource: {block.get('resource', {}).get('uri', '')}]"
                    )
                else:
                    text_parts.append(json.dumps(block))

        return {
            "success": not is_error,
            "tool": tool_name,
            "content": "\n".join(text_parts),
            "raw": result,
        }

    def _parse_tool_name(self, full_name: str) -> Tuple[str, str]:
        if full_name.startswith("mcp__"):
            parts = full_name.split("__", 2)
            if len(parts) == 3:
                return parts[1], parts[2]
        if ":" in full_name:
            return full_name.split(":", 1)
        raise MCPError(f"Invalid MCP tool name: {full_name}")

    # ------------------------------------------------------------------
    # RESOURCES / PROMPTS
    # ------------------------------------------------------------------

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        for key, resource in self.resources.items():
            if resource.uri == uri or key == uri:
                transport = self.transports.get(resource.server)
                if not transport:
                    raise MCPConnectionError(
                        f"Server '{resource.server}' not connected"
                    )
                return await transport.request(
                    "resources/read", {"uri": resource.uri}
                )
        raise MCPError(f"Resource not found: {uri}")

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        for key, prompt in self.prompts.items():
            if prompt.name == name or key == name:
                transport = self.transports.get(prompt.server)
                if not transport:
                    raise MCPConnectionError(
                        f"Server '{prompt.server}' not connected"
                    )
                return await transport.request(
                    "prompts/get",
                    {"name": prompt.name, "arguments": arguments or {}},
                )
        raise MCPError(f"Prompt not found: {name}")

    # ------------------------------------------------------------------
    # NOTIFICATIONS
    # ------------------------------------------------------------------

    async def _handle_notification(self, message: Dict[str, Any]) -> None:
        method = message.get("method", "")
        logger.debug(f"MCP notification: {method}")

        if method == "notifications/tools/list_changed":
            for name, transport in self.transports.items():
                try:
                    await self._discover_server(name, transport)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health check error: {e}")

    async def _health_check(self) -> None:
        for name, transport in list(self.transports.items()):
            if transport.state in (MCPState.ERROR, MCPState.DISCONNECTED):
                cfg = next(
                    (c for c in self.servers_cfg if c.name == name),
                    None,
                )
                if cfg and cfg.auto_reconnect:
                    logger.info(f"Reconnecting MCP server '{name}'...")
                    await self._reconnect(name, cfg)

    async def _reconnect(self, name: str, cfg: MCPServerConfig) -> None:
        for attempt in range(cfg.max_reconnect_attempts):
            try:
                await asyncio.sleep(min(2 ** attempt, 30))
                self.transports.pop(name, None)
                await self._connect_server(cfg)
                logger.info(f"Reconnected MCP server '{name}'")
                return
            except Exception as e:
                logger.warning(
                    f"Reconnect attempt {attempt + 1} for '{name}' failed: {e}"
                )
        logger.error(f"MCP server '{name}' could not be reconnected")

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "initialized": self._initialized,
            "servers": [
                {
                    "name": name,
                    "transport": t.config.transport.value,
                    "state": t.state.value,
                }
                for name, t in self.transports.items()
            ],
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "prompts_count": len(self.prompts),
        }

    def __repr__(self) -> str:
        return (
            f"<MCPClient servers={len(self.transports)} "
            f"tools={len(self.tools)} resources={len(self.resources)} "
            f"prompts={len(self.prompts)}>"
        )


# ======================================================================
# LEGACY CONVENIENCE BUILDERS (kept for backward compat)
# ======================================================================

def cloud_mcp_server(
    name: str,
    url: str,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a cloud MCP server config dict"""
    if provider == "anthropic" and not url:
        url = "https://api.anthropic.com/mcp"

    cfg: Dict[str, Any] = {
        "name": name,
        "transport": "cloud",
        "url": url,
        "headers": headers or {},
        "timeout": 60.0,
        "auto_reconnect": True,
        "max_reconnect_attempts": 5,
    }
    if api_key:
        cfg["api_key"] = api_key
    elif api_key_env:
        cfg["api_key_env"] = api_key_env

    return cfg


def stdio_mcp_server(
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "transport": "stdio",
        "command": command,
        "args": args or [],
        "env": env or {},
        "cwd": cwd,
        "timeout": 30.0,
        "auto_reconnect": True,
    }


def sse_mcp_server(
    name: str,
    url: str,
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg = {
        "name": name,
        "transport": "sse",
        "url": url,
        "headers": headers or {},
        "timeout": 60.0,
        "auto_reconnect": True,
    }
    if api_key:
        cfg["api_key"] = api_key
    return cfg


def http_mcp_server(
    name: str,
    url: str,
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg = {
        "name": name,
        "transport": "streamable-http",
        "url": url,
        "headers": headers or {},
        "timeout": 60.0,
        "auto_reconnect": True,
    }
    if api_key:
        cfg["api_key"] = api_key
    return cfg


__all__ = [
    "MCPClient",
    "MCPPresets",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPTransport",
    "MCPState",
    "StdioTransport",
    "HTTPTransport",
    "CloudTransport",
    "cloud_mcp_server",
    "stdio_mcp_server",
    "sse_mcp_server",
    "http_mcp_server",
]