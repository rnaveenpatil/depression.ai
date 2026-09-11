"""
CLI Commands Module - Complete Command System
Handles all slash commands including model selection, session management,
configuration, and advanced features.

NOTE: Model information is fetched dynamically from the LLM provider registry
in src/agent/llm/. No models are declared here.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from agent.utils.logging import get_logger
from agent.cli.ui import UI, Colors

logger = get_logger(__name__)


class CommandCategory(Enum):
    """Categories of commands"""
    GENERAL = "General"
    SESSION = "Session"
    MODEL = "Model"
    CONFIG = "Configuration"
    TOOLS = "Tools"
    PERMISSIONS = "Permissions"
    PROJECT = "Project"
    CONTEXT = "Context"
    PLUGINS = "Plugins"
    DEBUG = "Debug"
    HELP = "Help"


@dataclass
class Command:
    """Represents a CLI command"""
    name: str
    description: str
    usage: str
    aliases: List[str] = field(default_factory=list)
    category: CommandCategory = CommandCategory.GENERAL
    handler: Optional[Callable] = None
    requires_args: bool = False
    hidden: bool = False
    examples: List[str] = field(default_factory=list)

    def matches(self, input_str: str) -> bool:
        """Check if the input matches this command"""
        cmd = input_str.split()[0].lower().lstrip('/')
        return cmd == self.name.lower() or cmd in [a.lower() for a in self.aliases]


class CommandProcessor:
    """
    Advanced command processor.

    All model information is retrieved dynamically from the LLM provider
    registry (agent.llm.provider). This file contains NO hardcoded models.
    """

    def __init__(
        self,
        agent: Any,
        ui: UI,
        session_manager: Any,
        config: Dict[str, Any],
        llm_registry: Any = None,
    ):
        self.agent = agent
        self.ui = ui
        self.session_manager = session_manager
        self.config = config

        # LLM registry provides dynamic model listing/switching
        # (see src/agent/llm/provider.py -> LLMProviderRegistry)
        self.llm_registry = llm_registry or self._resolve_llm_registry()

        # Command registry
        self.commands: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}

        # Command history
        self.history: List[str] = []
        self.max_history = 1000

        # State
        self.running = True
        self.model_history: List[str] = []

        # Register all commands
        self._register_commands()

        logger.info(f"Command processor initialized with {len(self.commands)} commands")

    # ------------------------------------------------------------------
    # Registry resolution
    # ------------------------------------------------------------------

    def _resolve_llm_registry(self) -> Any:
        """Resolve the LLM registry from the agent, if available."""
        if self.agent and hasattr(self.agent, "llm_registry"):
            return self.agent.llm_registry
        # Fallback: lazily import the global registry
        try:
            from agent.llm.provider import get_llm_registry
            return get_llm_registry()
        except Exception as e:
            logger.warning(f"Could not resolve LLM registry: {e}")
            return None

    # ------------------------------------------------------------------
    # Command registration
    # ------------------------------------------------------------------

    def _register_commands(self) -> None:
        """Register all available commands"""

        # ===== GENERAL =====
        self._add_command(Command(
            name="help",
            description="Show help information for all commands or a specific command",
            usage="/help [command]",
            aliases=["h", "?"],
            category=CommandCategory.HELP,
            handler=self.cmd_help,
            examples=["/help", "/help model", "/help session"],
        ))

        self._add_command(Command(
            name="exit",
            description="Exit the agent gracefully",
            usage="/exit [--save]",
            aliases=["quit", "q", "bye"],
            category=CommandCategory.GENERAL,
            handler=self.cmd_exit,
            examples=["/exit", "/exit --save"],
        ))

        self._add_command(Command(
            name="clear",
            description="Clear the terminal screen",
            usage="/clear",
            aliases=["cls"],
            category=CommandCategory.GENERAL,
            handler=self.cmd_clear,
        ))

        self._add_command(Command(
            name="status",
            description="Show agent status and metrics",
            usage="/status [--verbose]",
            aliases=["info", "st"],
            category=CommandCategory.GENERAL,
            handler=self.cmd_status,
            examples=["/status", "/status --verbose"],
        ))

        self._add_command(Command(
            name="version",
            description="Show agent version",
            usage="/version",
            aliases=["ver"],
            category=CommandCategory.GENERAL,
            handler=self.cmd_version,
        ))

        # ===== MODEL COMMANDS =====
        self._add_command(Command(
            name="model",
            description="Manage and select models (fetched from llm/ providers)",
            usage="/model [list|set|info|current|test|params|compare] [args]",
            aliases=["m"],
            category=CommandCategory.MODEL,
            handler=self.cmd_model,
            examples=[
                "/model list",
                "/model list --provider openai",
                "/model list --capability vision",
                "/model set <model_id>",
                "/model info <model_id>",
                "/model current",
                "/model test",
                "/model params temperature=0.7",
                "/model compare <model_a> <model_b>",
            ],
        ))

        self._add_command(Command(
            name="models",
            description="Quick list of available models",
            usage="/models [provider]",
            aliases=["ml"],
            category=CommandCategory.MODEL,
            handler=self.cmd_models_list,
        ))

        self._add_command(Command(
            name="provider",
            description="Switch LLM provider (openai|anthropic|local|...)",
            usage="/provider <name>",
            aliases=["prov"],
            category=CommandCategory.MODEL,
            handler=self.cmd_provider,
            examples=["/provider openai", "/provider anthropic", "/provider local"],
        ))

        self._add_command(Command(
            name="temperature",
            description="Set model temperature (0.0 - 2.0)",
            usage="/temperature <value>",
            aliases=["temp"],
            category=CommandCategory.MODEL,
            handler=self.cmd_temperature,
            examples=["/temperature 0.7", "/temperature 1.0"],
        ))

        self._add_command(Command(
            name="maxtokens",
            description="Set maximum output tokens",
            usage="/maxtokens <number>",
            aliases=["mt"],
            category=CommandCategory.MODEL,
            handler=self.cmd_max_tokens,
            examples=["/maxtokens 4096", "/maxtokens 8192"],
        ))

        self._add_command(Command(
            name="modelparams",
            description="Show or set all model parameters",
            usage="/modelparams [key=value ...]",
            aliases=["mp"],
            category=CommandCategory.MODEL,
            handler=self.cmd_model_params,
            examples=[
                "/modelparams",
                "/modelparams temperature=0.5 max_tokens=4096 top_p=0.9",
            ],
        ))

        self._add_command(Command(
            name="apikey",
            description="Set or update API key for a provider",
            usage="/apikey <provider> [key]",
            aliases=["key"],
            category=CommandCategory.MODEL,
            handler=self.cmd_apikey,
            examples=["/apikey openai sk-...", "/apikey anthropic"],
        ))

        # ===== SESSION COMMANDS =====
        self._add_command(Command(
            name="session",
            description="Manage sessions",
            usage="/session [list|new|load|save|delete|info] [session_id]",
            aliases=["sess"],
            category=CommandCategory.SESSION,
            handler=self.cmd_session,
            examples=[
                "/session list",
                "/session new",
                "/session load abc123",
                "/session save",
                "/session info",
                "/session delete abc123",
            ],
        ))

        self._add_command(Command(
            name="new",
            description="Create a new session",
            usage="/new [--name <name>]",
            aliases=["n"],
            category=CommandCategory.SESSION,
            handler=self.cmd_new_session,
        ))

        self._add_command(Command(
            name="history",
            description="Show conversation history",
            usage="/history [--limit N] [--search query]",
            aliases=["hist"],
            category=CommandCategory.SESSION,
            handler=self.cmd_history,
            examples=["/history", "/history --limit 50", "/history --search 'bug fix'"],
        ))

        self._add_command(Command(
            name="reset",
            description="Reset the current session",
            usage="/reset [--confirm]",
            category=CommandCategory.SESSION,
            handler=self.cmd_reset,
        ))

        # ===== CONFIGURATION =====
        self._add_command(Command(
            name="config",
            description="View or modify configuration",
            usage="/config [get|set|list|reset] [key] [value]",
            aliases=["cfg"],
            category=CommandCategory.CONFIG,
            handler=self.cmd_config,
            examples=[
                "/config list",
                "/config get llm.model",
                "/config set llm.model <model_id>",
                "/config reset",
            ],
        ))

        self._add_command(Command(
            name="set",
            description="Quick config setter",
            usage="/set <key> <value>",
            category=CommandCategory.CONFIG,
            handler=self.cmd_set_config,
        ))

        self._add_command(Command(
            name="theme",
            description="Change UI theme",
            usage="/theme <dark|light|auto>",
            category=CommandCategory.CONFIG,
            handler=self.cmd_theme,
        ))

        # ===== TOOLS =====
        self._add_command(Command(
            name="tools",
            description="List all available tools",
            usage="/tools [list|info|enable|disable] [tool_name]",
            aliases=["t"],
            category=CommandCategory.TOOLS,
            handler=self.cmd_tools,
            examples=[
                "/tools",
                "/tools info terminal",
                "/tools enable web",
                "/tools disable git",
            ],
        ))

        self._add_command(Command(
            name="tool",
            description="Execute a tool directly",
            usage="/tool <tool_name> <json_params>",
            category=CommandCategory.TOOLS,
            handler=self.cmd_execute_tool,
            examples=['/tool terminal {"command": "ls -la"}'],
        ))

        # ===== PERMISSIONS =====
        self._add_command(Command(
            name="permissions",
            description="Manage permissions",
            usage="/permissions [list|allow|deny|reset|mode] [rule]",
            aliases=["perms", "perm"],
            category=CommandCategory.PERMISSIONS,
            handler=self.cmd_permissions,
            examples=[
                "/permissions list",
                "/permissions allow terminal:*",
                "/permissions deny filesystem:write:/etc/*",
                "/permissions mode auto",
                "/permissions reset",
            ],
        ))

        self._add_command(Command(
            name="yolo",
            description="Toggle auto-approve mode (use with caution!)",
            usage="/yolo [on|off]",
            category=CommandCategory.PERMISSIONS,
            handler=self.cmd_yolo,
        ))

        # ===== PROJECT =====
        self._add_command(Command(
            name="project",
            description="Project management",
            usage="/project [info|scan|files|git|cd] [path]",
            aliases=["proj"],
            category=CommandCategory.PROJECT,
            handler=self.cmd_project,
            examples=[
                "/project info",
                "/project scan",
                "/project files",
                "/project git",
                "/project cd /path/to/project",
            ],
        ))

        self._add_command(Command(
            name="cd",
            description="Change working directory",
            usage="/cd <path>",
            category=CommandCategory.PROJECT,
            handler=self.cmd_cd,
            examples=["/cd ../other-project", "/cd ~/projects/myapp"],
        ))

        self._add_command(Command(
            name="pwd",
            description="Show current directory",
            usage="/pwd",
            category=CommandCategory.PROJECT,
            handler=self.cmd_pwd,
        ))

        self._add_command(Command(
            name="files",
            description="List project files",
            usage="/files [--tree] [--ext <extension>]",
            category=CommandCategory.PROJECT,
            handler=self.cmd_files,
            examples=["/files", "/files --tree", "/files --ext .py"],
        ))

        # ===== CONTEXT =====
        self._add_command(Command(
            name="context",
            description="Manage context",
            usage="/context [show|clear|compact|stats]",
            aliases=["ctx"],
            category=CommandCategory.CONTEXT,
            handler=self.cmd_context,
            examples=[
                "/context show",
                "/context clear",
                "/context compact",
                "/context stats",
            ],
        ))

        self._add_command(Command(
            name="compact",
            description="Compact the current context",
            usage="/compact [--aggressive]",
            category=CommandCategory.CONTEXT,
            handler=self.cmd_compact,
        ))

        # ===== PLUGINS =====
        self._add_command(Command(
            name="plugins",
            description="Manage plugins",
            usage="/plugins [list|load|unload|info] [plugin_name]",
            aliases=["plugin"],
            category=CommandCategory.PLUGINS,
            handler=self.cmd_plugins,
            examples=[
                "/plugins list",
                "/plugins load my-plugin",
                "/plugins unload my-plugin",
                "/plugins info my-plugin",
            ],
        ))

        # ===== DEBUG =====
        self._add_command(Command(
            name="debug",
            description="Debug information",
            usage="/debug [on|off|state|logs|metrics]",
            category=CommandCategory.DEBUG,
            handler=self.cmd_debug,
            hidden=True,
        ))

        self._add_command(Command(
            name="metrics",
            description="Show performance metrics",
            usage="/metrics [--reset]",
            category=CommandCategory.DEBUG,
            handler=self.cmd_metrics,
        ))

        self._add_command(Command(
            name="logs",
            description="Show recent logs",
            usage="/logs [--level <level>] [--lines N]",
            category=CommandCategory.DEBUG,
            handler=self.cmd_logs,
        ))

        self._add_command(Command(
            name="tokens",
            description="Show token usage",
            usage="/tokens [--reset]",
            category=CommandCategory.DEBUG,
            handler=self.cmd_tokens,
        ))

        self._add_command(Command(
            name="cost",
            description="Show cost estimates",
            usage="/cost [--session|--total|--reset]",
            category=CommandCategory.DEBUG,
            handler=self.cmd_cost,
        ))

        # ===== EXPORT / IMPORT =====
        self._add_command(Command(
            name="export",
            description="Export session or config",
            usage="/export <session|config|context> [path]",
            category=CommandCategory.SESSION,
            handler=self.cmd_export,
            examples=["/export session", "/export config ~/backup.json"],
        ))

        self._add_command(Command(
            name="import",
            description="Import session or config",
            usage="/import <session|config> <path>",
            category=CommandCategory.SESSION,
            handler=self.cmd_import,
        ))

    def _add_command(self, command: Command) -> None:
        """Add a command to the registry"""
        self.commands[command.name] = command
        for alias in command.aliases:
            self.aliases[alias] = command.name

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def process_command(self, user_input: str) -> Optional[str]:
        """Process a command from user input"""
        self.history.append(user_input)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        parts = user_input.strip().split()
        if not parts:
            return None

        cmd_name = parts[0].lower().lstrip('/')
        args = parts[1:]

        if cmd_name in self.aliases:
            cmd_name = self.aliases[cmd_name]

        command = self.commands.get(cmd_name)
        if not command:
            self.ui.print_error(f"Unknown command: /{cmd_name}")
            self.ui.print_info("Type /help to see available commands")
            return None

        try:
            return await command.handler(args)
        except Exception as e:
            logger.error(f"Command '{cmd_name}' failed: {e}", exc_info=True)
            self.ui.print_error(f"Command failed: {e}")
            return None

    # ==================================================================
    # GENERAL HANDLERS
    # ==================================================================

    async def cmd_help(self, args: List[str]) -> None:
        if args:
            cmd_name = args[0].lower().lstrip('/')
            if cmd_name in self.aliases:
                cmd_name = self.aliases[cmd_name]
            command = self.commands.get(cmd_name)
            if not command:
                self.ui.print_error(f"Unknown command: /{cmd_name}")
                return

            self.ui.print_header(f"Help: /{command.name}")
            self.ui.print_info(f"\nDescription: {command.description}")
            self.ui.print_info(f"Usage: {command.usage}")
            if command.aliases:
                self.ui.print_info(f"Aliases: {', '.join('/' + a for a in command.aliases)}")
            if command.examples:
                self.ui.print_info("\nExamples:")
                for example in command.examples:
                    self.ui.print_info(f"  {example}")
            return

        self.ui.print_header("Available Commands")
        categories: Dict[CommandCategory, List[Command]] = {}
        for cmd in self.commands.values():
            if cmd.hidden:
                continue
            categories.setdefault(cmd.category, []).append(cmd)

        for category in CommandCategory:
            if category not in categories or category == CommandCategory.HELP:
                continue
            self.ui.print_info(f"\n📁 {category.value}:")
            for cmd in sorted(categories[category], key=lambda c: c.name):
                aliases_str = f" ({', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
                self.ui.print_info(f"  /{cmd.name}{aliases_str}")
                self.ui.print_info(f"      {cmd.description}")

        self.ui.print_info("\n💡 Type /help <command> for detailed help")
        self.ui.print_info("💡 Use ↑/↓ arrows to navigate history")

    async def cmd_exit(self, args: List[str]) -> str:
        save = "--save" in args
        if save:
            self.ui.print_info("Saving session...")
            await self.session_manager.save_current_session()
        self.ui.print_info("👋 Goodbye!")
        self.running = False
        return "exit"

    async def cmd_clear(self, args: List[str]) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')
        self.ui.print_header("CLI Agent")

    async def cmd_status(self, args: List[str]) -> None:
        verbose = "--verbose" in args or "-v" in args
        status = self.agent.get_status() if self.agent else {}

        self.ui.print_header("Agent Status")

        self.ui.print_info(f"\n📊 Status:")
        self.ui.print_info(f"  State: {status.get('status', 'unknown')}")
        self.ui.print_info(f"  Session: {status.get('session_id', 'N/A')[:16]}...")
        self.ui.print_info(f"  Turns: {status.get('turn_count', 0)}")
        self.ui.print_info(f"  Tasks Completed: {status.get('tasks_completed', 0)}")

        model_info = self.config.get('llm', {})
        self.ui.print_info(f"\n🤖 Model:")
        self.ui.print_info(f"  Provider: {model_info.get('provider', 'unknown')}")
        self.ui.print_info(f"  Model: {model_info.get('model', 'unknown')}")

        metrics = status.get('metrics', {})
        self.ui.print_info(f"\n⚡ Performance:")
        self.ui.print_info(f"  Queries: {metrics.get('total_queries', 0)}")
        self.ui.print_info(f"  Tokens: {metrics.get('total_tokens', 0):,}")
        self.ui.print_info(f"  Avg Response: {metrics.get('avg_response_time', 0):.2f}s")
        self.ui.print_info(f"  Success Rate: {metrics.get('success_rate', 1.0) * 100:.1f}%")

        if verbose:
            tools_used = status.get('tools_used', {})
            if tools_used:
                self.ui.print_info(f"\n🔧 Tools Used:")
                for tool, count in sorted(tools_used.items(), key=lambda x: -x[1]):
                    self.ui.print_info(f"  {tool}: {count}")
            uptime = status.get('uptime', 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            self.ui.print_info(f"\n⏱️  Uptime: {hours}h {minutes}m {seconds}s")

    async def cmd_version(self, args: List[str]) -> None:
        self.ui.print_info("CLI Agent v1.0.0")
        self.ui.print_info(f"Python: {sys.version.split()[0]}")

    # ==================================================================
    # MODEL HANDLERS (dynamic; everything pulled from LLM registry)
    # ==================================================================

    async def cmd_model(self, args: List[str]) -> None:
        """
        /model [list|set|info|current|test|params|compare] ...
        All model data comes from self.llm_registry — nothing hardcoded.
        """
        if not args:
            await self._show_current_model()
            self.ui.print_info("\nUsage: /model [list|set|info|current|test|params|compare]")
            self.ui.print_info("Run '/model list' to see all available models")
            return

        sub = args[0].lower()
        rest = args[1:]

        dispatch = {
            "list": self._list_models,
            "ls": self._list_models,
            "set": self._set_model,
            "use": self._set_model,
            "info": self._model_info,
            "show": self._model_info,
            "current": self._show_current_model,
            "cur": self._show_current_model,
            "test": self._test_model,
            "params": self._model_params,
            "compare": self._compare_models,
        }

        handler = dispatch.get(sub)
        if handler:
            await handler(rest)
        else:
            # Assume it's a model ID
            await self._set_model(args)

    async def _list_models(self, args: List[str]) -> None:
        """List all models available in the LLM registry"""
        if not self.llm_registry:
            self.ui.print_error("LLM registry not available")
            return

        provider_filter = None
        capability_filter = None
        available_only = False

        i = 0
        while i < len(args):
            a = args[i]
            if a in ("--provider", "-p") and i + 1 < len(args):
                provider_filter = args[i + 1].lower()
                i += 2
            elif a in ("--capability", "-c") and i + 1 < len(args):
                capability_filter = args[i + 1].lower()
                i += 2
            elif a in ("--available", "-a"):
                available_only = True
                i += 1
            else:
                i += 1

        self.ui.print_header("📚 Available Models")

        # Ask registry for all models (dict: provider -> list of model dicts)
        try:
            all_models = await self._fetch_models_from_registry()
        except Exception as e:
            self.ui.print_error(f"Failed to fetch models: {e}")
            return

        if not all_models:
            self.ui.print_warning("No models found in registry. Configure a provider first.")
            return

        current_model = self.config.get('llm', {}).get('model', '')

        provider_icons = {
            "openai": "🟢",
            "anthropic": "🟣",
            "local": "🔵",
            "google": "🔴",
            "mistral": "🟠",
            "cohere": "🟤",
        }

        shown_any = False
        for provider, models in sorted(all_models.items()):
            if provider_filter and provider.lower() != provider_filter:
                continue

            filtered = []
            for m in models:
                if capability_filter and capability_filter not in [c.lower() for c in m.get("capabilities", [])]:
                    continue
                if available_only and not m.get("available", True):
                    continue
                filtered.append(m)

            if not filtered:
                continue

            shown_any = True
            icon = provider_icons.get(provider.lower(), "⚪")
            self.ui.print_info(f"\n{icon} {provider.upper()} ({len(filtered)} models)")
            self.ui.print_separator()

            for m in filtered:
                model_id = m.get("id") or m.get("model")
                is_current = "✅ " if model_id == current_model else "   "
                available = m.get("available", True)
                status_icon = "" if available else " ⚠️ (unavailable)"

                self.ui.print_info(f"{is_current}{m.get('name', model_id)}{status_icon}")
                self.ui.print_info(f"      ID: {model_id}")

                if m.get("description"):
                    self.ui.print_info(f"      {m['description']}")

                ctx = m.get("context_window")
                out = m.get("max_output")
                if ctx or out:
                    ctx_k = f"{ctx // 1000}K" if ctx else "?"
                    out_k = f"{out // 1000}K" if out else "?"
                    self.ui.print_info(f"      Context: {ctx_k} | Output: {out_k}")

                caps = m.get("capabilities", [])
                if caps:
                    self.ui.print_info(f"      Capabilities: {', '.join(caps)}")

                cost_in = m.get("cost_input")
                cost_out = m.get("cost_output")
                if cost_in is not None and cost_out is not None:
                    if cost_in == 0 and cost_out == 0:
                        self.ui.print_info(f"      Cost: FREE (local)")
                    else:
                        self.ui.print_info(
                            f"      Cost: ${cost_in}/M in, ${cost_out}/M out"
                        )

        if not shown_any:
            self.ui.print_warning("No models match your filters.")

    async def _set_model(self, args: List[str]) -> None:
        """Switch to a different model"""
        if not args:
            self.ui.print_error("Usage: /model set <model_id>")
            self.ui.print_info("Run '/model list' to see available models")
            return

        model_id = args[0]

        if not self.llm_registry:
            self.ui.print_error("LLM registry not available")
            return

        # Ask the registry to switch
        try:
            result = await self._switch_model_in_registry(model_id)
        except Exception as e:
            self.ui.print_error(f"Failed to switch model: {e}")
            return

        if not result.get("success", False):
            self.ui.print_error(result.get("error", f"Model '{model_id}' not available"))
            self.ui.print_info("Run '/model list' to see available models")
            return

        old_model = self.config.get("llm", {}).get("model", "")
        self.config.setdefault("llm", {})["model"] = model_id
        provider = result.get("provider", self.config["llm"].get("provider"))
        self.config["llm"]["provider"] = provider

        if self.agent:
            # Push to agent if it exposes a hook
            if hasattr(self.agent, "set_model"):
                await self.agent.set_model(model_id, provider)

        self.model_history.append(model_id)

        self.ui.print_success(f"✅ Switched model: {old_model or 'none'} → {model_id}")
        self.ui.print_info(f"   Provider: {provider}")

    async def _model_info(self, args: List[str]) -> None:
        """Show detailed info for a specific model"""
        if not args:
            self.ui.print_error("Usage: /model info <model_id>")
            return

        model_id = args[0]

        try:
            info = await self._fetch_model_info(model_id)
        except Exception as e:
            self.ui.print_error(f"Failed to fetch model info: {e}")
            return

        if not info:
            self.ui.print_error(f"Model '{model_id}' not found in registry")
            return

        self.ui.print_header(f"🔍 Model: {info.get('name', model_id)}")
        self.ui.print_info(f"\nID:          {model_id}")
        self.ui.print_info(f"Provider:    {info.get('provider', 'unknown')}")
        self.ui.print_info(f"Description: {info.get('description', 'N/A')}")

        if info.get("context_window"):
            self.ui.print_info(f"Context:     {info['context_window']:,} tokens")
        if info.get("max_output"):
            self.ui.print_info(f"Max Output:  {info['max_output']:,} tokens")

        caps = info.get("capabilities", [])
        if caps:
            self.ui.print_info(f"Capabilities: {', '.join(caps)}")

        cost_in = info.get("cost_input")
        cost_out = info.get("cost_output")
        if cost_in is not None and cost_out is not None:
            if cost_in == 0 and cost_out == 0:
                self.ui.print_info("Pricing:     FREE (local)")
            else:
                self.ui.print_info(f"Pricing:     ${cost_in}/M input, ${cost_out}/M output")

        if info.get("recommended_for"):
            self.ui.print_info(f"Best for:    {', '.join(info['recommended_for'])}")

    async def _show_current_model(self) -> None:
        """Display the currently selected model"""
        llm_cfg = self.config.get("llm", {})
        model_id = llm_cfg.get("model", "not set")
        provider = llm_cfg.get("provider", "not set")

        self.ui.print_header("🤖 Current Model")
        self.ui.print_info(f"Model:    {model_id}")
        self.ui.print_info(f"Provider: {provider}")

        if self.llm_registry:
            try:
                info = await self._fetch_model_info(model_id)
                if info:
                    self.ui.print_info(f"Context:  {info.get('context_window', 0):,} tokens")
            except Exception:
                pass

        # Non-default params
        params = llm_cfg.get("params", {})
        if params:
            self.ui.print_info("\nParameters:")
            for k, v in params.items():
                self.ui.print_info(f"  {k}: {v}")

    async def _test_model(self, args: List[str]) -> None:
        """Send a small test query to the current model"""
        prompt = " ".join(args) if args else "Say 'OK' in one word."
        self.ui.print_info(f"🧪 Testing model with: '{prompt}'")

        if not self.agent or not getattr(self.agent, "llm", None):
            self.ui.print_error("Agent LLM not available")
            return

        try:
            start = time.time()
            # Use the agent's LLM provider
            result = await self.agent.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            elapsed = time.time() - start
            content = getattr(result, "content", str(result))
            self.ui.print_success(f"✅ Response ({elapsed:.2f}s): {content}")
        except Exception as e:
            self.ui.print_error(f"Test failed: {e}")

    async def _model_params(self, args: List[str]) -> None:
        """Show or set model parameters"""
        llm_cfg = self.config.setdefault("llm", {})
        params = llm_cfg.setdefault("params", {})

        if not args:
            self.ui.print_header("⚙️  Model Parameters")
            if not params:
                self.ui.print_info("(no parameters set)")
            else:
                for k, v in params.items():
                    self.ui.print_info(f"  {k}: {v}")
            self.ui.print_info("\nUsage: /modelparams key=value key2=value2")
            return

        for arg in args:
            if "=" not in arg:
                self.ui.print_warning(f"Ignoring malformed arg: {arg}")
                continue
            key, value = arg.split("=", 1)
            parsed = self._parse_param_value(value)
            params[key.strip()] = parsed
            self.ui.print_info(f"  {key} = {parsed}")

        self.ui.print_success("Parameters updated")

    async def _compare_models(self, args: List[str]) -> None:
        """Compare two models side by side"""
        if len(args) < 2:
            self.ui.print_error("Usage: /model compare <model_a> <model_b>")
            return

        a_id, b_id = args[0], args[1]
        info_a = await self._fetch_model_info(a_id)
        info_b = await self._fetch_model_info(b_id)

        if not info_a or not info_b:
            self.ui.print_error("One or both models not found")
            return

        self.ui.print_header(f"⚖️  {info_a.get('name', a_id)}  vs  {info_b.get('name', b_id)}")

        def row(label, key, fmt=lambda x: x):
            va = fmt(info_a.get(key, "-"))
            vb = fmt(info_b.get(key, "-"))
            self.ui.print_info(f"  {label:<14} | {va:<28} | {vb}")

        row("Provider", "provider")
        row("Context", "context_window", lambda v: f"{v:,}" if isinstance(v, int) else v)
        row("Max Output", "max_output", lambda v: f"{v:,}" if isinstance(v, int) else v)
        row("Input $/M", "cost_input", lambda v: f"${v}" if isinstance(v, (int, float)) else v)
        row("Output $/M", "cost_output", lambda v: f"${v}" if isinstance(v, (int, float)) else v)
        row("Capabilities", "capabilities", lambda v: ", ".join(v) if isinstance(v, list) else v)

    async def cmd_models_list(self, args: List[str]) -> None:
        """Quick model list"""
        await self._list_models(args)

    async def cmd_provider(self, args: List[str]) -> None:
        """Switch provider"""
        if not args:
            current = self.config.get("llm", {}).get("provider", "unknown")
            self.ui.print_info(f"Current provider: {current}")

            providers = await self._list_providers_from_registry()
            if providers:
                self.ui.print_info("Available providers:")
                for p in providers:
                    marker = " (current)" if p == current else ""
                    self.ui.print_info(f"  - {p}{marker}")
            return

        provider_name = args[0].lower()

        if not self.llm_registry:
            self.ui.print_error("LLM registry not available")
            return

        try:
            ok = await self._set_provider_in_registry(provider_name)
        except Exception as e:
            self.ui.print_error(f"Failed to switch provider: {e}")
            return

        if not ok:
            self.ui.print_error(f"Provider '{provider_name}' not available")
            return

        self.config.setdefault("llm", {})["provider"] = provider_name
        self.ui.print_success(f"✅ Provider switched to: {provider_name}")

    async def cmd_temperature(self, args: List[str]) -> None:
        if not args:
            current = self.config.get("llm", {}).get("params", {}).get("temperature", "not set")
            self.ui.print_info(f"Current temperature: {current}")
            return

        try:
            value = float(args[0])
        except ValueError:
            self.ui.print_error("Temperature must be a number")
            return

        if not 0.0 <= value <= 2.0:
            self.ui.print_error("Temperature must be between 0.0 and 2.0")
            return

        self.config.setdefault("llm", {}).setdefault("params", {})["temperature"] = value
        self.ui.print_success(f"Temperature set to {value}")

    async def cmd_max_tokens(self, args: List[str]) -> None:
        if not args:
            current = self.config.get("llm", {}).get("params", {}).get("max_tokens", "not set")
            self.ui.print_info(f"Current max tokens: {current}")
            return

        try:
            value = int(args[0])
        except ValueError:
            self.ui.print_error("Max tokens must be an integer")
            return

        if value < 1:
            self.ui.print_error("Max tokens must be >= 1")
            return

        self.config.setdefault("llm", {}).setdefault("params", {})["max_tokens"] = value
        self.ui.print_success(f"Max tokens set to {value}")

    async def cmd_model_params(self, args: List[str]) -> None:
        await self._model_params(args)

    async def cmd_apikey(self, args: List[str]) -> None:
        """Set API key for a provider (also saved to secure store)"""
        if not args:
            self.ui.print_error("Usage: /apikey <provider> [key]")
            return

        provider = args[0].lower()

        if len(args) >= 2:
            key = args[1]
            self.config.setdefault("llm", {}).setdefault("api_keys", {})[provider] = key
            if self.llm_registry and hasattr(self.llm_registry, "set_api_key"):
                self.llm_registry.set_api_key(provider, key)
            self.ui.print_success(f"API key set for {provider}")
        else:
            # Masked display
            existing = self.config.get("llm", {}).get("api_keys", {}).get(provider)
            if existing:
                masked = existing[:6] + "..." + existing[-4:] if len(existing) > 10 else "***"
                self.ui.print_info(f"{provider} API key: {masked}")
            else:
                self.ui.print_warning(f"No API key configured for {provider}")
                self.ui.print_info(f"Set it via: export {provider.upper()}_API_KEY=...")
                self.ui.print_info(f"Or: /apikey {provider} <key>")

    # ==================================================================
    # SESSION HANDLERS
    # ==================================================================

    async def cmd_session(self, args: List[str]) -> None:
        if not args:
            await self._session_info()
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "list" or sub == "ls":
            await self._session_list()
        elif sub == "new":
            await self.cmd_new_session(rest)
        elif sub == "load":
            await self._session_load(rest)
        elif sub == "save":
            await self._session_save()
        elif sub == "delete" or sub == "rm":
            await self._session_delete(rest)
        elif sub == "info":
            await self._session_info()
        else:
            self.ui.print_error(f"Unknown session subcommand: {sub}")

    async def _session_list(self) -> None:
        self.ui.print_header("📋 Sessions")
        try:
            sessions = await self.session_manager.list_sessions()
            if not sessions:
                self.ui.print_info("No sessions found")
                return
            current = getattr(self.session_manager, "current_session_id", None)
            for s in sessions:
                marker = "✅ " if s.get("id") == current else "   "
                self.ui.print_info(
                    f"{marker}{s.get('id', '?')[:12]} | "
                    f"{s.get('name', 'unnamed')} | "
                    f"{s.get('messages', 0)} msgs | "
                    f"{s.get('updated_at', '')}"
                )
        except Exception as e:
            self.ui.print_error(f"Failed to list sessions: {e}")

    async def _session_load(self, args: List[str]) -> None:
        if not args:
            self.ui.print_error("Usage: /session load <session_id>")
            return
        session_id = args[0]
        try:
            session = await self.session_manager.load_session(session_id)
            if not session:
                self.ui.print_error(f"Session '{session_id}' not found")
                return
            self.ui.print_success(f"Loaded session: {session_id}")
        except Exception as e:
            self.ui.print_error(f"Failed to load session: {e}")

    async def _session_save(self) -> None:
        try:
            await self.session_manager.save_current_session()
            self.ui.print_success("Session saved")
        except Exception as e:
            self.ui.print_error(f"Failed to save session: {e}")

    async def _session_delete(self, args: List[str]) -> None:
        if not args:
            self.ui.print_error("Usage: /session delete <session_id>")
            return
        if "--confirm" not in args:
            self.ui.print_warning(f"Add --confirm to delete session: {args[0]}")
            return
        try:
            await self.session_manager.delete_session(args[0])
            self.ui.print_success(f"Deleted session: {args[0]}")
        except Exception as e:
            self.ui.print_error(f"Failed to delete: {e}")

    async def _session_info(self) -> None:
        self.ui.print_header("📋 Current Session")
        try:
            info = await self.session_manager.get_current_info()
            for k, v in info.items():
                self.ui.print_info(f"  {k}: {v}")
        except Exception as e:
            self.ui.print_error(f"Failed to get info: {e}")

    async def cmd_new_session(self, args: List[str]) -> None:
        name = None
        if "--name" in args:
            idx = args.index("--name")
            if idx + 1 < len(args):
                name = args[idx + 1]
        try:
            session = await self.session_manager.create_session(name=name)
            self.ui.print_success(f"Created new session: {session.id}")
        except Exception as e:
            self.ui.print_error(f"Failed to create session: {e}")

    async def cmd_history(self, args: List[str]) -> None:
        limit = 20
        search = None

        i = 0
        while i < len(args):
            if args[i] == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "--search" and i + 1 < len(args):
                search = args[i + 1].lower()
                i += 2
            else:
                i += 1

        try:
            history = await self.session_manager.get_history()
            if search:
                history = [h for h in history if search in str(h).lower()]
            history = history[-limit:]
            self.ui.print_header(f"📜 History (last {len(history)})")
            for i, msg in enumerate(history, 1):
                role = msg.get("role", "?") if isinstance(msg, dict) else "?"
                content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
                if len(content) > 200:
                    content = content[:200] + "..."
                self.ui.print_info(f"  [{i}] {role}: {content}")
        except Exception as e:
            self.ui.print_error(f"Failed to get history: {e}")

    async def cmd_reset(self, args: List[str]) -> None:
        if "--confirm" not in args:
            self.ui.print_warning("This will reset the current session. Add --confirm to proceed.")
            return
        try:
            await self.session_manager.reset_current()
            self.ui.print_success("Session reset")
        except Exception as e:
            self.ui.print_error(f"Failed to reset: {e}")

    # ==================================================================
    # CONFIG HANDLERS
    # ==================================================================

    async def cmd_config(self, args: List[str]) -> None:
        if not args:
            await self._config_list()
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "list" or sub == "ls":
            await self._config_list()
        elif sub == "get":
            await self._config_get(rest)
        elif sub == "set":
            await self._config_set(rest)
        elif sub == "reset":
            await self._config_reset(rest)
        else:
            self.ui.print_error(f"Unknown config subcommand: {sub}")

    async def _config_list(self) -> None:
        self.ui.print_header("⚙️  Configuration")
        self._print_config_dict(self.config)

    def _print_config_dict(self, d: Dict, indent: int = 0) -> None:
        for k, v in d.items():
            prefix = "  " * indent
            if isinstance(v, dict):
                self.ui.print_info(f"{prefix}{k}:")
                self._print_config_dict(v, indent + 1)
            else:
                self.ui.print_info(f"{prefix}{k}: {v}")

    async def _config_get(self, args: List[str]) -> None:
        if not args:
            self.ui.print_error("Usage: /config get <key>")
            return
        key = args[0]
        value = self._get_nested(self.config, key)
        if value is None:
            self.ui.print_warning(f"Key '{key}' not found")
        else:
            self.ui.print_info(f"{key} = {value}")

    async def _config_set(self, args: List[str]) -> None:
        if len(args) < 2:
            self.ui.print_error("Usage: /config set <key> <value>")
            return
        key, value = args[0], " ".join(args[1:])
        self._set_nested(self.config, key, self._parse_param_value(value))
        self.ui.print_success(f"{key} = {self._get_nested(self.config, key)}")

    async def _config_reset(self, args: List[str]) -> None:
        if "--confirm" not in args:
            self.ui.print_warning("Add --confirm to reset configuration")
            return
        try:
            from agent.config.loader import load_config
            self.config.clear()
            self.config.update(load_config())
            self.ui.print_success("Configuration reset")
        except Exception as e:
            self.ui.print_error(f"Failed to reset: {e}")

    async def cmd_set_config(self, args: List[str]) -> None:
        await self._config_set(args)

    async def cmd_theme(self, args: List[str]) -> None:
        if not args:
            current = self.config.get("ui", {}).get("theme", "dark")
            self.ui.print_info(f"Current theme: {current}")
            self.ui.print_info("Options: dark, light, auto")
            return
        theme = args[0].lower()
        if theme not in ("dark", "light", "auto"):
            self.ui.print_error("Theme must be: dark, light, or auto")
            return
        self.config.setdefault("ui", {})["theme"] = theme
        if hasattr(self.ui, "set_theme"):
            self.ui.set_theme(theme)
        self.ui.print_success(f"Theme set to: {theme}")

    # ==================================================================
    # TOOLS HANDLERS
    # ==================================================================

    async def cmd_tools(self, args: List[str]) -> None:
        if not args or args[0] in ("list", "ls"):
            await self._tools_list()
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "info" and rest:
            await self._tool_info(rest[0])
        elif sub in ("enable", "disable") and rest:
            await self._tool_toggle(sub, rest[0])
        else:
            self.ui.print_error(f"Unknown tools subcommand: {sub}")

    async def _tools_list(self) -> None:
        self.ui.print_header("🔧 Available Tools")
        try:
            tools = self.agent.tool_registry.tools if self.agent else {}
            if not tools:
                self.ui.print_info("No tools registered")
                return
            for name, tool in sorted(tools.items()):
                enabled = getattr(tool, "enabled", True)
                status = "✅" if enabled else "⛔"
                self.ui.print_info(f"  {status} {name}")
                self.ui.print_info(f"      {getattr(tool, 'description', 'no description')}")
        except Exception as e:
            self.ui.print_error(f"Failed to list tools: {e}")

    async def _tool_info(self, name: str) -> None:
        try:
            tools = self.agent.tool_registry.tools if self.agent else {}
            tool = tools.get(name)
            if not tool:
                self.ui.print_error(f"Tool '{name}' not found")
                return
            self.ui.print_header(f"🔧 Tool: {name}")
            self.ui.print_info(f"Description: {getattr(tool, 'description', 'N/A')}")
            params = getattr(tool, "parameters", None)
            if params:
                self.ui.print_info(f"Parameters:\n{json.dumps(params, indent=2)}")
        except Exception as e:
            self.ui.print_error(f"Failed to get tool info: {e}")

    async def _tool_toggle(self, action: str, name: str) -> None:
        try:
            tools = self.agent.tool_registry.tools if self.agent else {}
            tool = tools.get(name)
            if not tool:
                self.ui.print_error(f"Tool '{name}' not found")
                return
            tool.enabled = (action == "enable")
            self.ui.print_success(f"Tool '{name}' {action}d")
        except Exception as e:
            self.ui.print_error(f"Failed to {action} tool: {e}")

    async def cmd_execute_tool(self, args: List[str]) -> None:
        if len(args) < 2:
            self.ui.print_error("Usage: /tool <tool_name> <json_params>")
            return
        tool_name = args[0]
        raw_params = " ".join(args[1:])
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError as e:
            self.ui.print_error(f"Invalid JSON params: {e}")
            return
        if not self.agent:
            self.ui.print_error("Agent not available")
            return
        try:
            result = await self.agent.execute_tool(tool_name, params)
            self.ui.print_info(json.dumps(result, indent=2, default=str))
        except Exception as e:
            self.ui.print_error(f"Tool execution failed: {e}")

    # ==================================================================
    # PERMISSIONS HANDLERS
    # ==================================================================

    async def cmd_permissions(self, args: List[str]) -> None:
        pm = getattr(self.agent, "permission_manager", None)
        if not pm:
            self.ui.print_error("Permission manager not available")
            return

        if not args or args[0] in ("list", "ls"):
            await self._permissions_list(pm)
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "allow" and rest:
            pm.add_allow_rule(rest[0])
            self.ui.print_success(f"Added allow rule: {rest[0]}")
        elif sub == "deny" and rest:
            pm.add_deny_rule(rest[0])
            self.ui.print_success(f"Added deny rule: {rest[0]}")
        elif sub == "reset":
            pm.reset_rules()
            self.ui.print_success("Permission rules reset")
        elif sub == "mode" and rest:
            mode = rest[0].lower()
            if mode in ("auto", "manual", "deny"):
                pm.set_mode(mode)
                self.ui.print_success(f"Permission mode: {mode}")
            else:
                self.ui.print_error("Mode must be: auto, manual, or deny")
        else:
            self.ui.print_error(f"Unknown permissions subcommand: {sub}")

    async def _permissions_list(self, pm) -> None:
        self.ui.print_header("🔒 Permission Rules")
        self.ui.print_info(f"  Enabled: {getattr(pm, 'enabled', True)}")
        self.ui.print_info(f"  Mode:    {getattr(pm, 'mode', 'manual')}")
        rules = getattr(pm, "rules", [])
        if not rules:
            self.ui.print_info("  (no explicit rules)")
        for r in rules:
            self.ui.print_info(f"  - {r}")

    async def cmd_yolo(self, args: List[str]) -> None:
        pm = getattr(self.agent, "permission_manager", None)
        if not pm:
            self.ui.print_error("Permission manager not available")
            return

        if not args:
            current = getattr(pm, "auto_approve", False)
            self.ui.print_info(f"Auto-approve: {current}")
            self.ui.print_info("Usage: /yolo [on|off]")
            return

        state = args[0].lower()
        if state in ("on", "true", "1"):
            pm.auto_approve = True
            self.ui.print_warning("⚠️  Auto-approve ENABLED — all tool calls will execute without prompts!")
        elif state in ("off", "false", "0"):
            pm.auto_approve = False
            self.ui.print_success("Auto-approve DISABLED")
        else:
            self.ui.print_error("Usage: /yolo [on|off]")

    # ==================================================================
    # PROJECT HANDLERS
    # ==================================================================

    async def cmd_project(self, args: List[str]) -> None:
        if not args:
            await self._project_info()
            return
        sub = args[0].lower()
        rest = args[1:]
        if sub == "info":
            await self._project_info()
        elif sub == "scan":
            await self._project_scan()
        elif sub == "files":
            await self.cmd_files(rest)
        elif sub == "git":
            await self._project_git()
        elif sub == "cd" and rest:
            await self.cmd_cd(rest)
        else:
            self.ui.print_error(f"Unknown project subcommand: {sub}")

    async def _project_info(self) -> None:
        self.ui.print_header("📁 Project Info")
        try:
            ws = getattr(self.agent, "workspace", None)
            if not ws:
                self.ui.print_warning("Workspace not available")
                return
            info = await ws.get_info() if hasattr(ws, "get_info") else {}
            for k, v in (info or {}).items():
                self.ui.print_info(f"  {k}: {v}")
        except Exception as e:
            self.ui.print_error(f"Failed to get project info: {e}")

    async def _project_scan(self) -> None:
        self.ui.print_info("Scanning project...")
        try:
            ws = getattr(self.agent, "workspace", None)
            if ws and hasattr(ws, "scan"):
                result = await ws.scan()
                self.ui.print_success(f"Scan complete: {result}")
            else:
                self.ui.print_warning("Scanning not available")
        except Exception as e:
            self.ui.print_error(f"Scan failed: {e}")

    async def _project_git(self) -> None:
        try:
            ws = getattr(self.agent, "workspace", None)
            if ws and hasattr(ws, "get_git_info"):
                info = await ws.get_git_info()
                self.ui.print_header("🔀 Git Info")
                for k, v in (info or {}).items():
                    self.ui.print_info(f"  {k}: {v}")
            else:
                self.ui.print_warning("Git info not available")
        except Exception as e:
            self.ui.print_error(f"Failed: {e}")

    async def cmd_cd(self, args: List[str]) -> None:
        if not args:
            self.ui.print_error("Usage: /cd <path>")
            return
        try:
            path = os.path.expanduser(args[0])
            os.chdir(path)
            if self.agent and hasattr(self.agent, "workspace"):
                await self.agent.workspace.set_project_dir(os.getcwd())
            self.ui.print_success(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            self.ui.print_error(f"Failed to change directory: {e}")

    async def cmd_pwd(self, args: List[str]) -> None:
        self.ui.print_info(os.getcwd())

    async def cmd_files(self, args: List[str]) -> None:
        ext_filter = None
        tree = "--tree" in args

        if "--ext" in args:
            i = args.index("--ext")
            if i + 1 < len(args):
                ext_filter = args[i + 1]

        try:
            ws = getattr(self.agent, "workspace", None)
            if ws and hasattr(ws, "list_files"):
                files = await ws.list_files(ext=ext_filter, tree=tree)
                for f in files:
                    self.ui.print_info(f"  {f}")
            else:
                self.ui.print_warning("File listing not available")
        except Exception as e:
            self.ui.print_error(f"Failed to list files: {e}")

    # ==================================================================
    # CONTEXT HANDLERS
    # ==================================================================

    async def cmd_context(self, args: List[str]) -> None:
        cm = getattr(self.agent, "context_manager", None)
        if not cm:
            self.ui.print_error("Context manager not available")
            return

        if not args:
            args = ["show"]

        sub = args[0].lower()
        if sub == "show":
            await self._context_show(cm)
        elif sub == "clear":
            await cm.clear()
            self.ui.print_success("Context cleared")
        elif sub == "compact":
            await cm.compact()
            self.ui.print_success("Context compacted")
        elif sub == "stats":
            await self._context_stats(cm)
        else:
            self.ui.print_error(f"Unknown context subcommand: {sub}")

    async def _context_show(self, cm) -> None:
        self.ui.print_header("📖 Current Context")
        try:
            ctx = await cm.get_context() if hasattr(cm, "get_context") else {}
            text = json.dumps(ctx, indent=2, default=str)
            if len(text) > 2000:
                text = text[:2000] + "\n... (truncated)"
            self.ui.print_info(text)
        except Exception as e:
            self.ui.print_error(f"Failed to show context: {e}")

    async def _context_stats(self, cm) -> None:
        self.ui.print_header("📊 Context Stats")
        try:
            stats = await cm.get_stats() if hasattr(cm, "get_stats") else {}
            for k, v in (stats or {}).items():
                self.ui.print_info(f"  {k}: {v}")
        except Exception as e:
            self.ui.print_error(f"Failed: {e}")

    async def cmd_compact(self, args: List[str]) -> None:
        cm = getattr(self.agent, "context_manager", None)
        if not cm:
            self.ui.print_error("Context manager not available")
            return
        aggressive = "--aggressive" in args
        try:
            await cm.compact(aggressive=aggressive)
            self.ui.print_success("Context compacted")
        except Exception as e:
            self.ui.print_error(f"Failed: {e}")

    # ==================================================================
    # PLUGINS HANDLERS
    # ==================================================================

    async def cmd_plugins(self, args: List[str]) -> None:
        loader = getattr(self.agent, "plugin_loader", None)
        if not loader:
            self.ui.print_error("Plugin loader not available")
            return

        if not args or args[0] in ("list", "ls"):
            await self._plugins_list(loader)
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "load" and rest:
            await loader.load_plugin(rest[0])
            self.ui.print_success(f"Loaded plugin: {rest[0]}")
        elif sub == "unload" and rest:
            await loader.unload_plugin(rest[0])
            self.ui.print_success(f"Unloaded plugin: {rest[0]}")
        elif sub == "info" and rest:
            info = await loader.get_plugin_info(rest[0])
            self.ui.print_info(json.dumps(info, indent=2, default=str))
        else:
            self.ui.print_error(f"Unknown plugins subcommand: {sub}")

    async def _plugins_list(self, loader) -> None:
        self.ui.print_header("🧩 Plugins")
        try:
            plugins = await loader.list_plugins() if hasattr(loader, "list_plugins") else []
            if not plugins:
                self.ui.print_info("No plugins loaded")
                return
            for p in plugins:
                self.ui.print_info(f"  - {p}")
        except Exception as e:
            self.ui.print_error(f"Failed: {e}")

    # ==================================================================
    # DEBUG HANDLERS
    # ==================================================================

    async def cmd_debug(self, args: List[str]) -> None:
        if not args:
            self.ui.print_info("Usage: /debug [on|off|state|logs|metrics]")
            return
        sub = args[0].lower()
        if sub == "on":
            self.config.setdefault("debug", {})["enabled"] = True
            self.ui.print_success("Debug mode ON")
        elif sub == "off":
            self.config.setdefault("debug", {})["enabled"] = False
            self.ui.print_success("Debug mode OFF")
        elif sub == "state":
            state = self.agent.get_status() if self.agent else {}
            self.ui.print_info(json.dumps(state, indent=2, default=str))
        elif sub == "logs":
            await self.cmd_logs([])
        elif sub == "metrics":
            await self.cmd_metrics([])

    async def cmd_metrics(self, args: List[str]) -> None:
        if "--reset" in args:
            if self.agent:
                for k in self.agent.metrics:
                    self.agent.metrics[k] = 0
            self.ui.print_success("Metrics reset")
            return
        self.ui.print_header("📊 Metrics")
        metrics = self.agent.metrics if self.agent else {}
        for k, v in metrics.items():
            self.ui.print_info(f"  {k}: {v}")

    async def cmd_logs(self, args: List[str]) -> None:
        level = "INFO"
        lines = 50

        i = 0
        while i < len(args):
            if args[i] == "--level" and i + 1 < len(args):
                level = args[i + 1].upper()
                i += 2
            elif args[i] == "--lines" and i + 1 < len(args):
                try:
                    lines = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        try:
            from agent.utils.logging import get_recent_logs
            logs = get_recent_logs(level=level, limit=lines)
            self.ui.print_header(f"📜 Logs ({level}, last {lines})")
            for entry in logs:
                self.ui.print_info(f"  {entry}")
        except Exception as e:
            self.ui.print_error(f"Failed to fetch logs: {e}")

    async def cmd_tokens(self, args: List[str]) -> None:
        if "--reset" in args:
            if self.agent and hasattr(self.agent, "context"):
                self.agent.context.tokens_used = 0
            self.ui.print_success("Token counter reset")
            return
        used = getattr(getattr(self.agent, "context", None), "tokens_used", 0) if self.agent else 0
        self.ui.print_info(f"Tokens used this session: {used:,}")

    async def cmd_cost(self, args: List[str]) -> None:
        if "--reset" in args:
            if self.agent and hasattr(self.agent, "context"):
                self.agent.context.cost = 0.0
            self.ui.print_success("Cost counter reset")
            return
        cost = getattr(getattr(self.agent, "context", None), "cost", 0.0) if self.agent else 0.0
        self.ui.print_info(f"Estimated session cost: ${cost:.4f}")

    # ==================================================================
    # EXPORT / IMPORT
    # ==================================================================

    async def cmd_export(self, args: List[str]) -> None:
        if not args:
            self.ui.print_error("Usage: /export <session|config|context> [path]")
            return
        what = args[0].lower()
        path = args[1] if len(args) > 1 else f"./export_{what}_{int(time.time())}.json"

        try:
            if what == "session":
                data = await self.session_manager.export_session()
            elif what == "config":
                data = self.config
            elif what == "context":
                cm = getattr(self.agent, "context_manager", None)
                data = await cm.get_context() if cm else {}
            else:
                self.ui.print_error(f"Unknown export target: {what}")
                return

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self.ui.print_success(f"Exported {what} to {path}")
        except Exception as e:
            self.ui.print_error(f"Export failed: {e}")

    async def cmd_import(self, args: List[str]) -> None:
        if len(args) < 2:
            self.ui.print_error("Usage: /import <session|config> <path>")
            return
        what, path = args[0].lower(), args[1]
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if what == "session":
                await self.session_manager.import_session(data)
            elif what == "config":
                self.config.update(data)
            else:
                self.ui.print_error(f"Unknown import target: {what}")
                return
            self.ui.print_success(f"Imported {what} from {path}")
        except Exception as e:
            self.ui.print_error(f"Import failed: {e}")

    # ==================================================================
    # LLM REGISTRY BRIDGE (dynamic — no models defined here)
    # ==================================================================

    async def _fetch_models_from_registry(self) -> Dict[str, List[Dict]]:
        """
        Ask the LLM registry for all available models.
        Expected return format:
            { "openai": [ {id, name, description, capabilities, ...}, ... ], ... }
        """
        if not self.llm_registry:
            return {}

        # Preferred async interface
        if hasattr(self.llm_registry, "list_all_models_async"):
            return await self.llm_registry.list_all_models_async()

        # Fallback sync
        if hasattr(self.llm_registry, "list_all_models"):
            return self.llm_registry.list_all_models()

        # Last resort: iterate providers
        result: Dict[str, List[Dict]] = {}
        if hasattr(self.llm_registry, "providers"):
            for name, provider in self.llm_registry.providers.items():
                if hasattr(provider, "list_models"):
                    models = provider.list_models()
                    if asyncio.iscoroutine(models):
                        models = await models
                    result[name] = models
        return result

    async def _fetch_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed info for one model from the registry"""
        if not self.llm_registry:
            return None
        if hasattr(self.llm_registry, "get_model_info"):
            info = self.llm_registry.get_model_info(model_id)
            if asyncio.iscoroutine(info):
                info = await info
            return info
        return None

    async def _switch_model_in_registry(self, model_id: str) -> Dict:
        """Ask the registry to switch model. Returns {'success': bool, ...}"""
        if not self.llm_registry:
            return {"success": False, "error": "No registry"}
        if hasattr(self.llm_registry, "set_model"):
            result = self.llm_registry.set_model(model_id)
            if asyncio.iscoroutine(result):
                result = await result
            return result or {"success": True}
        return {"success": False, "error": "Registry does not support set_model"}

    async def _list_providers_from_registry(self) -> List[str]:
        if not self.llm_registry:
            return []
        if hasattr(self.llm_registry, "list_providers"):
            result = self.llm_registry.list_providers()
            if asyncio.iscoroutine(result):
                result = await result
            return result or []
        if hasattr(self.llm_registry, "providers"):
            return list(self.llm_registry.providers.keys())
        return []

    async def _set_provider_in_registry(self, provider_name: str) -> bool:
        if not self.llm_registry:
            return False
        if hasattr(self.llm_registry, "set_provider"):
            result = self.llm_registry.set_provider(provider_name)
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        return provider_name in getattr(self.llm_registry, "providers", {})

    # ==================================================================
    # UTILITIES
    # ==================================================================

    def _get_nested(self, d: Dict, key: str) -> Any:
        """Get a nested dict value using dot notation"""
        parts = key.split(".")
        cur = d
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur

    def _set_nested(self, d: Dict, key: str, value: Any) -> None:
        """Set a nested dict value using dot notation"""
        parts = key.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def _parse_param_value(self, value: str) -> Any:
        """Parse a string into int/float/bool/str"""
        v = value.strip()
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.lower() in ("none", "null"):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return value


# Required import for iscoroutine checks
import asyncio  # noqa: E402