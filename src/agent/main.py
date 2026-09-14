#!/usr/bin/env python3
"""
CLI Agent - Main Entry Point

Bootstraps the entire agent:
    1. Parse CLI arguments
    2. Load layered config
    3. Set up logging
    4. Initialize storage (Database + Cache)
    5. Set up workspace
    6. Create session (or resume)
    7. Initialize permissions
    8. Initialize context
    9. Connect MCP servers
    10. Build the Agent
    11. Run interactive or non-interactive mode
    12. Graceful shutdown
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------
# EARLY PATH SETUP (so `agent.*` imports work when running from repo root)
# ----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ----------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------
from agent.utils.logging import setup_logging, get_logger
from agent.utils.errors import (
    AgentError, ConfigError, format_error, handle_exception,
)
from agent.utils.platform import (
    get_platform, get_config_dir, get_cache_dir, get_data_dir,
    is_windows, terminal_size,
)

from agent.config.loader import load_config, get_config
from agent.config.config import Config
from agent.cli.ui import UI, get_ui, Icons, Palette, Colors
from agent.cli.input import InputHandler
from agent.cli.commands import CommandProcessor

from agent.storage.database import Database
from agent.storage.cache import Cache

from agent.session.session import SessionManager, Session
from agent.context.manager import ContextManager
from agent.permissions.manager import PermissionManager
from agent.project.workspace import WorkspaceManager
from agent.mcp.client import MCPClient
from agent.agent.dual_agent import create_dual_agent_system, AgentCoordinator


logger = get_logger(__name__)

APP_NAME = "depression"
APP_VERSION = "1.0.0"


# ======================================================================
# CLI AGENT APPLICATION
# ======================================================================

class CLIAgent:
    """Top-level application orchestrator."""

    def __init__(self):
        self.running = False
        self.interactive = True

        # Subsystems (created in initialize)
        self.ui: Optional[UI] = None
        self.input_handler: Optional[InputHandler] = None
        self.command_processor: Optional[CommandProcessor] = None
        self.database: Optional[Database] = None
        self.cache: Optional[Cache] = None
        self.workspace: Optional[WorkspaceManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.session: Optional[Session] = None
        self.context_manager: Optional[ContextManager] = None
        self.permission_manager: Optional[PermissionManager] = None
        self.mcp_client: Optional[MCPClient] = None
        self.agent: Optional[Agent] = None

        # Config
        self.config_dict: dict = {}
        self.config_obj = None

        # Runtime
        self.args: Optional[argparse.Namespace] = None
        self.start_time = datetime.now()

        # Signal handling
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ------------------------------------------------------------------
    # SIGNALS
    # ------------------------------------------------------------------

    def _on_signal(self, signum, frame):
        if self.ui:
            self.ui.print_warning(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        if self.agent_coordinator:
            try:
                asyncio.create_task(self.agent_coordinator.shutdown())
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=APP_NAME,
            description="Advanced agentic AI CLI - your intelligent terminal assistant",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  depression                          Start interactive mode
  depression "fix the login bug"      Run a single query
  depression -p /path/to/project      Use a specific project
  depression --model gpt-4o           Override the model
  depression --provider groq          Use a specific provider
  depression --yolo                   Auto-approve everything (careful!)
  depression --session <id>           Resume a session
  depression --new-session            Force a fresh session
  depression --verbose                Increase verbosity
  depression --init-config            Write a default config file
""",
        )

        # Positional
        parser.add_argument("query", nargs="?", help="Initial query or task")

        # Project / workspace
        parser.add_argument("-p", "--project", help="Project directory (default: cwd)")
        parser.add_argument("--workspace", help="Workspace directory for agent state")

        # Model / provider
        parser.add_argument("-m", "--model", help="Override the model (e.g. openai/gpt-oss-120b)")
        parser.add_argument(
            "--provider",
            help="Override the provider (openai, anthropic, groq, nvidia, xai, local, ...)",
        )
        parser.add_argument("--temperature", type=float, help="Fixed temperature (disables adaptive)")

        # Behavior
        parser.add_argument("--non-interactive", action="store_true", help="Run one query and exit")
        parser.add_argument("--yolo", action="store_true", help="Auto-approve all tool calls")
        parser.add_argument(
            "--permission-mode",
            choices=["manual", "auto", "deny"],
            help="Override permission mode",
        )
        parser.add_argument(
            "--no-permissions",
            action="store_true",
            help="Disable permission checks (equivalent to --yolo)",
        )

        # Verbosity
        parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
        parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

        # Session
        parser.add_argument("--session", help="Session ID to resume")
        parser.add_argument("--new-session", action="store_true", help="Force a new session")

        # Config
        parser.add_argument("--config", help="Path to a specific config file")
        parser.add_argument("--init-config", action="store_true", help="Write a default config and exit")

        # Limits
        parser.add_argument("--max-turns", type=int, default=50, help="Max iterations per query (agent loop)")
        parser.add_argument("--max-iterations", type=int, default=3, help="Max plan-execute iterations (coordinator)")
        parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

        # Extras
        parser.add_argument("--plugins", nargs="*", help="Plugin names to load")
        parser.add_argument("--no-mcp", action="store_true", help="Disable MCP")
        parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")

        return parser.parse_args()

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    async def initialize(self, args: argparse.Namespace) -> None:
        self.args = args

        # 1. Logging ---------------------------------------------------
        level = "INFO"
        if args.verbose == 1:
            level = "DEBUG"
        elif args.verbose >= 2:
            level = "DEBUG"
        elif args.quiet:
            level = "ERROR"

        # Config dir for the log file
        log_dir = get_data_dir(APP_NAME) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "agent.log"

        setup_logging(
            level=level,
            log_file=str(log_file),
            console=not args.quiet,
        )

        # 2. UI --------------------------------------------------------
        self.ui = get_ui()
        width = min(terminal_size()[0], 120)
        self.ui.width = width

        if not args.quiet:
            self.ui.print_banner(
                subtitle="Agentic AI CLI",
                version=f"v{APP_VERSION}",
            )

        # 3. Config ----------------------------------------------------
        if not args.quiet:
            self.ui.print_status("Loading configuration…")

        try:
            self.config_dict = load_config(
                config_path=args.config,
                init_config=args.init_config,
                cache=True,
            )
            self.config_obj = get_config()
        except ConfigError as e:
            self.ui.print_error(f"Config error: {e}")
            raise

        if args.init_config:
            self.ui.print_success("Default configuration written.")
            return

        # Apply CLI overrides on top of the loaded config
        self._apply_cli_overrides(args)

        # Re-create config_obj from updated dict so overrides take effect
        self.config_obj = Config.from_dict(self.config_dict)
        self.config_obj.validate()

        # Update global config in the loader module so resolve_llm uses the overridden values
        import agent.config.loader as config_loader
        config_loader._global_config = self.config_obj

        if not args.quiet:
            self.ui.print_status("Configuration loaded.")

        # 4. Storage ---------------------------------------------------
        storage_cfg = self.config_dict.get("storage", {}) or {}
        db_path = args.workspace or storage_cfg.get("path", str(get_data_dir(APP_NAME) / "agent.db"))
        cache_dir = storage_cfg.get("cache_dir", str(get_cache_dir(APP_NAME)))

        if not args.quiet:
            self.ui.print_status(f"Opening database at {db_path}…")

        self.database = Database(db_path)
        await self.database.initialize()

        self.cache = Cache(
            cache_dir=cache_dir,
            max_disk_bytes=storage_cfg.get("max_cache_size", 500 * 1024 * 1024),
            enabled=storage_cfg.get("enable_cache", True),
        )

        # 5. Workspace -------------------------------------------------
        ws_cfg = self.config_dict.get("workspace", {}) or {}
        project_dir = args.project or ws_cfg.get("path", os.getcwd())
        workspace_dir = args.workspace or ws_cfg.get("workspace_dir", str(get_data_dir(APP_NAME)))

        self.workspace = WorkspaceManager(
            workspace_dir=workspace_dir,
            project_dir=project_dir,
            config=ws_cfg,
        )
        await self.workspace.initialize()

        if not args.quiet:
            info = await self.workspace.get_info()
            self.ui.print_status(
                f"Workspace ready: {info.get('project_name')} "
                f"({info.get('total_files', 0)} files)"
            )

        # 6. Session ---------------------------------------------------
        session_cfg = self.config_dict.get("session", {}) or {}
        self.session_manager = SessionManager(
            database=self.database,
            config=session_cfg,
        )

        if args.session:
            self.session = await self.session_manager.load_session(args.session)
            if not self.session:
                self.ui.print_warning(f"Session {args.session} not found; creating a new one.")
                self.session = await self.session_manager.create_session()
        elif args.new_session:
            self.session = await self.session_manager.create_session()
        else:
            self.session = await self.session_manager.get_or_create_session()

        if not args.quiet:
            self.ui.print_success(f"Session: {self.session.id[:8]}  ({self.session.name})")

        # 7. Permissions ----------------------------------------------
        # Support both "permission" (opencode) and "permissions" (legacy)
        perm_cfg = dict(self.config_dict.get("permissions", {}) or {})
        opencode_perm = self.config_dict.get("permission")
        if isinstance(opencode_perm, dict):
            # merge opencode permission map under `permission` key for RulesPolicy
            perm_cfg["permission"] = {**perm_cfg.get("permission", {}), **opencode_perm}
        elif isinstance(opencode_perm, str):
            perm_cfg["permission"] = {"*": opencode_perm}
        # opencode global permission string shorthand {"permission":"allow"}
        if args.no_permissions or args.yolo:
            perm_cfg["auto_approve"] = True
            perm_cfg["mode"] = "auto"
        if args.permission_mode:
            perm_cfg["mode"] = args.permission_mode
        # pass workspace root for external_directory checks
        perm_cfg.setdefault("cwd", str(self.workspace.get_project_dir()) if self.workspace else os.getcwd())
        perm_cfg.setdefault("workspace_dir", str(self.workspace.get_project_dir()) if self.workspace else os.getcwd())

        self.permission_manager = PermissionManager(
            config=perm_cfg,
            ui=self.ui,
            input_handler=None,  # set below
        )

        # 8. Input handler --------------------------------------------
        self.input_handler = InputHandler(
            command_processor=None,      # set below
            base_dir=str(self.workspace.get_project_dir()),
        )
        self.permission_manager.input_handler = self.input_handler

        # 9. Context ---------------------------------------------------
        ctx_cfg = self.config_dict.get("context", {}) or {}
        # Pass LLM registry through so the compactor can summarize
        from agent.llm.provider import get_llm_registry
        llm_registry = get_llm_registry(config=self.config_dict)

        self.context_manager = ContextManager(
            workspace=self.workspace,
            session=self.session,
            config=ctx_cfg,
            llm=llm_registry,
        )
        await self.context_manager.initialize()

        # 10. MCP ------------------------------------------------------
        mcp_cfg = dict(self.config_dict.get("mcp", {}) or {})
        if args.no_mcp:
            mcp_cfg["enabled"] = False

        if mcp_cfg.get("enabled"):
            if not args.quiet:
                self.ui.print_status("Connecting MCP servers…")
            self.mcp_client = MCPClient(mcp_cfg)
            try:
                await self.mcp_client.initialize()
                if not args.quiet:
                    status = self.mcp_client.get_status()
                    self.ui.print_status(
                        f"MCP: {status['servers'].__len__()} server(s), "
                        f"{status['tools_count']} tool(s)"
                    )
            except Exception as e:
                logger.warning(f"MCP init failed: {e}")
                self.ui.print_warning(f"MCP unavailable: {e}")

        # 11. Agent ----------------------------------------------------
        if not args.quiet:
            self.ui.print_status("Initializing dual agent system (Plan + Build)…")

        self.agent_coordinator = await create_dual_agent_system(
            config=self.config_dict,
            session=self.session,
            context_manager=self.context_manager,
            permission_manager=self.permission_manager,
            workspace=self.workspace,
            database=self.database,
            ui=self.ui,
            input_handler=self.input_handler,
            mcp_client=self.mcp_client,
            cache=self.cache,
            max_turns=args.max_turns,
            timeout=args.timeout,
        )
        # For backwards compatibility, expose plan_agent as self.agent
        self.agent = self.agent_coordinator.plan_agent

        # 12. Command processor ---------------------------------------
        self.command_processor = CommandProcessor(
            agent=self.agent,
            ui=self.ui,
            session_manager=self.session_manager,
            config=self.config_dict,
            llm_registry=llm_registry,
        )
        self.input_handler.command_processor = self.command_processor

        # 13. Load plugins from CLI -----------------------------------
        if args.plugins:
            try:
                for name in args.plugins:
                    await self.agent.plugin_loader.load_plugin(name)
            except Exception as e:
                self.ui.print_warning(f"Plugin load failed: {e}")

        # 14. Wire live output rendering (OpenCode: stream + tool live + permission cards + status)
        for agent in (self.agent_coordinator.plan_agent, self.agent_coordinator.build_agent):
            if hasattr(agent, 'loop') and agent.loop:
                # Live tool render — matches opencode tool call live view
                async def _on_tool(data, _ui=self.ui):
                    try:
                        _ui.print_tool_call(
                            tool_name=data.get('tool', 'tool'),
                            params=data.get('params'),
                            result=data.get('result'),
                            success=data.get('result', {}).get('success', True) if isinstance(data.get('result'), dict) else True,
                            duration=data.get('execution_time')
                        )
                        # Update status bar per tool call (tokens/cost live)
                        st = agent.get_status()
                        _ui.set_status(model=st.get('model') or '—', tokens=st.get('tokens_used', 0), cost=st.get('cost', 0.0), session=st.get('session_id','')[:8])
                    except Exception:
                        pass
                agent.loop.add_event_handler('on_tool_executed', _on_tool)
                # Permission risk rendering is handled via PermissionManager._render_prompt -> UI box
        status = self.agent.get_status()
        self.ui.set_status(
            model=status.get("model") or "—",
            tokens=status.get("tokens_used", 0),
            cost=status.get("cost", 0.0),
            session=status.get("session_id", "")[:8],
        )

        self.interactive = not args.non_interactive and not args.query
        self.running = True

    def _apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """Apply CLI flags on top of the loaded config."""
        cfg = self.config_dict
        cfg.setdefault("llm", {})
        if args.model:
            cfg["llm"]["model"] = args.model
            # If model includes provider prefix (e.g., nvidia/model), don't override provider
            if "/" in args.model and not args.provider:
                provider_prefix = args.model.split("/")[0]
                cfg["llm"]["provider"] = provider_prefix
        if args.provider and (not args.model or "/" not in args.model):
            cfg["llm"]["provider"] = args.provider
        if args.temperature is not None:
            cfg["llm"].setdefault("params", {})["temperature"] = args.temperature
        if args.no_permissions or args.yolo:
            cfg.setdefault("permissions", {})["auto_approve"] = True
            cfg["permissions"]["mode"] = "auto"
        if args.verbose:
            cfg.setdefault("logging", {})["level"] = "DEBUG"

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    async def run(self) -> None:
        if not self.running:
            return

        # If a query was passed on the command line, run it once
        if self.args and self.args.query:
            await self._run_single_query(self.args.query)
            return

        if self.interactive:
            await self._run_interactive()
        else:
            # Non-interactive with no query → nothing to do
            self.ui.print_info("No query provided. Use `depression --help`.")

    async def _run_single_query(self, query: str) -> None:
        self.ui.print_user_message(query)
        spinner = None
        if not self.args.quiet:
            spinner = self.ui.start_spinner("Processing…")

        try:
            # Use auto mode for backward compatibility, but allow mode selection
            result = await asyncio.wait_for(
                self.agent_coordinator.process_query(
                    query,
                    mode="auto",
                    auto_execute=True,  # Execute plan automatically in auto mode
                    max_iterations=self.args.max_iterations,
                ),
                timeout=120,  # Reduced timeout: 2 minutes instead of 5
            )
        except asyncio.TimeoutError:
            if spinner:
                spinner.stop()
            self.ui.print_error(f"Query timed out after {self.args.timeout}s")
            return

        if spinner:
            spinner.stop()

        if result.get("success"):
            if result.get("plan"):
                self.ui.print_info("📋 Plan created, executing...")
            self.ui.print_agent_message(result.get("execution") or result.get("response", ""))
        else:
            self.ui.print_error(result.get("error", "unknown error"))

        # Update status
        status = self.agent_coordinator.get_status()
        current_mode = status.get("current_mode", "build")
        self.ui.set_status(
            model=f"Mode: {current_mode} | Plan: {status['plan_agent']['model']} | Build: {status['build_agent']['model']}",
            tokens=0,
            cost=0.0,
            session=(self.session.id if self.session else "")[:8],
        )

    async def _run_interactive(self) -> None:
        # Welcome
        status = self.agent.get_status()
        self.ui.print_welcome(
            version=f"v{APP_VERSION}",
            model=status.get("model") or "—",
        )
        self.ui.print_shortcuts()
        self.ui.print_info("Press Ctrl+P to switch between Plan/Build mode, or use /plan and /build commands")

        while self.running:
            try:
                # Show current mode in prompt
                mode = self.agent_coordinator.current_mode
                prompt = f"🧠 [{mode}] "
                user_input = await self.input_handler.get_input(prompt=prompt)
            except EOFError:
                self.ui.print_info("\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                self.ui.print_info("")
                continue

            if user_input is None:
                break

            text = user_input.strip()
            if not text:
                continue

            # Handle mode switching commands
            if text == "/plan":
                self.agent_coordinator.set_mode("plan")
                self.ui.print_info("Switched to Plan mode (read-only analysis)")
                self._refresh_status_bar()
                continue
            elif text == "/build":
                self.agent_coordinator.set_mode("build")
                self.ui.print_info("Switched to Build mode (full execution)")
                self._refresh_status_bar()
                continue
            elif text == "/auto":
                self.ui.print_info("Auto mode: Plan -> Build loop")
                # In auto mode, we process with auto_execute=True
                # The actual processing happens in _handle_query
                self._refresh_status_bar()
                continue

            # Slash command
            if text.startswith("/"):
                try:
                    result = await self.command_processor.process_command(text)
                except Exception as e:
                    self.ui.print_error(f"Command failed: {e}")
                    continue

                if result == "exit":
                    break
                self._refresh_status_bar()
                continue

            # Regular query
            await self._handle_query(text)
            self._refresh_status_bar()

    async def _handle_query(self, text: str) -> None:
        self.ui.print_user_message(text)

        spinner = None
        if not self.args.quiet:
            mode = self.agent_coordinator.current_mode
            spinner = self.ui.start_spinner(f"Processing ({mode})...")

        start = time.time()
        try:
            # Use current mode for processing
            result = await asyncio.wait_for(
                self.agent_coordinator.process_query(
                    text, 
                    mode=self.agent_coordinator.current_mode,
                    auto_execute=True,  # Auto-execute plan if in plan mode
                ),
                timeout=self.args.timeout,
            )
        except asyncio.TimeoutError:
            if spinner:
                spinner.stop()
            self.ui.print_error(f"Query timed out after {self.args.timeout}s")
            return
        except KeyboardInterrupt:
            if spinner:
                spinner.stop()
            self.ui.print_warning("Interrupted.")
            return

        if spinner:
            spinner.stop()

        if result.get("success"):
            if result.get("plan"):
                self.ui.print_info("📋 Plan created, executing...")
            self.ui.print_agent_message(result.get("execution") or result.get("response", ""))
            if self.args.verbose:
                iterations = result.get("iterations", 1)
                self.ui.print_status(
                    f"Plan + Execute ({iterations} iteration{'s' if iterations > 1 else ''}) • {time.time() - start:.2f}s"
                )
        else:
            self.ui.print_error(result.get("error", "unknown error"))

    def _refresh_status_bar(self) -> None:
        if not self.agent_coordinator:
            return
        status = self.agent_coordinator.get_status()
        self.ui.set_status(
            model=f"Plan: {status['plan_agent']['model']} | Build: {status['build_agent']['model']}",
            tokens=0,
            cost=0.0,
            session=(self.session.id if self.session else "")[:8],
        )

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        self.ui.print_info("Shutting down…")

        # Persist everything first
        try:
            if self.session_manager:
                await self.session_manager.save_current_session()
        except Exception as e:
            logger.debug(f"Session save failed during shutdown: {e}")

        # Agent shutdown (own tools, MCP, plugins)
        if self.agent_coordinator:
            try:
                await self.agent_coordinator.shutdown()
            except Exception as e:
                logger.debug(f"Agent coordinator shutdown failed: {e}")

        # MCP
        if self.mcp_client:
            try:
                await self.mcp_client.shutdown()
            except Exception:
                pass

        # Database
        if self.database:
            try:
                await self.database.close()
            except Exception:
                pass

        # Cache GC (best effort)
        if self.cache:
            try:
                await self.cache.gc_expired()
            except Exception:
                pass

        uptime = (datetime.now() - self.start_time).total_seconds()
        self.ui.print_success(
            f"Goodbye! Session: {self.session.id[:8] if self.session else '-'} • "
            f"Uptime: {int(uptime)}s"
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> int:
    """Entry point."""
    app = CLIAgent()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args = app.parse_arguments()

        loop.run_until_complete(app.initialize(args))
        loop.run_until_complete(app.run())
        loop.run_until_complete(app.shutdown())

    except KeyboardInterrupt:
        print("\n👋 Interrupted.")
        return 130
    except ConfigError as e:
        print(f"❌ Config error: {e}", file=sys.stderr)
        return 2
    except AgentError as e:
        print(f"❌ {format_error(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {format_error(e)}", file=sys.stderr)
        if app.args and getattr(app.args, "verbose", 0) >= 1:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        try:
            loop = asyncio.get_event_loop()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())