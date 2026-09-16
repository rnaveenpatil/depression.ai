"""Fully bootstrapped Textual entry point."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> None:
    from agent.main import CLIAgent
    from agent.llm.provider import get_llm_registry
    from agent.llm.runtime import load_runtime_config, configure_runtime_provider

    cli = CLIAgent()
    args = cli.parse_arguments()
    args.non_interactive = True
    args.query = None
    args.quiet = True

    # Configure the persisted TUI endpoint before Plan/Build are constructed.
    runtime = load_runtime_config()
    if all(runtime.values()):
        configure_runtime_provider(get_llm_registry(), runtime["base_url"],
                                   runtime["api_key"], runtime["model"])

    # Signal state shared with the boot coroutine. Registered with the event
    # loop (via add_signal_handler) so app.exit() runs as a normal loop
    # callback and the ExitApp message is actually processed.
    sig = {"app": None, "count": 0}

    def _on_signal(signum):
        sig["count"] += 1
        if sig["count"] >= 3:
            # Hard escape hatch: repeated ^C always gets us out.
            os._exit(128 + signum)
        if cli.ui:
            cli.ui.print_warning(f"\nReceived signal {signum}, shutting down...")
        cli.running = False
        app = sig["app"]
        if app is None:
            return  # still booting; the app hasn't mounted yet
        if signum == signal.SIGTERM:
            app.exit()
            return
        # SIGINT: cancel an in-flight agent request first, quit when idle.
        if app.is_agent_busy():
            app.cancel_current_agent()
        else:
            app.exit()

    async def boot() -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, partial(_on_signal, signal.SIGINT))
        loop.add_signal_handler(signal.SIGTERM, partial(_on_signal, signal.SIGTERM))
        await cli.initialize(args)
        from agent.tui.app import DepressionApp
        # Never fall back to a blocking CLI prompt in the TUI: a missed
        # permission ask must dead-end at a non-blocking DENY (the modal is
        # wired once the app mounts), otherwise prompt_toolkit/in.getpass
        # dispatch a thread that reads the Textual-owned stdin forever and the
        # loop reports 'executor did not finish joining its threads' on exit.
        for _agent in (getattr(cli.agent_coordinator, "plan_agent", None),
                       getattr(cli.agent_coordinator, "build_agent", None)):
            _pm = getattr(_agent, "permission_manager", None)
            if _pm is not None:
                _pm.input_handler = None
        app = DepressionApp(coordinator=cli.agent_coordinator)
        sig["app"] = app
        try:
            await app.run_async()
        finally:
            # Always run the real shutdown path; app.exit()/quit unwind here.
            await cli.shutdown()

    asyncio.run(boot())


if __name__ == "__main__":
    main()