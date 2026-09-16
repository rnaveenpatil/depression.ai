"""Fully bootstrapped Textual entry point."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from concurrent.futures import Future
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

    runtime = load_runtime_config()
    if all(runtime.values()):
        configure_runtime_provider(get_llm_registry(), runtime["base_url"],
                                   runtime["api_key"], runtime["model"])

    sig = {"app": None, "count": 0}

    def _on_signal(signum):
        sig["count"] += 1
        if sig["count"] >= 3:
            os._exit(128 + signum)
        if cli.ui:
            cli.ui.print_warning(f"\nReceived signal {signum}, shutting down...")
        cli.running = False
        app = sig["app"]
        if app is None:
            return
        if signum == signal.SIGTERM:
            app.exit()
            return
        if app.is_agent_busy():
            app.cancel_current_agent()
        else:
            app.exit()

    async def boot() -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, partial(_on_signal, signal.SIGINT))
        loop.add_signal_handler(signal.SIGTERM, partial(_on_signal, signal.SIGTERM))

        await cli.initialize(args)

        from agent.tui.app import DepressionApp, PermissionModal

        # The app's legacy on_mount hook installs a callback using
        # asyncio.get_event_loop(). That is the worker loop when permission
        # checks originate from an agent worker, so the modal can be pushed on
        # the wrong loop and become visible but unresponsive. Keep ownership
        # of the callback here, where the Textual app instance is available.
        class TUIDepressionApp(DepressionApp):
            def _wire_permission_confirmation(self) -> None:
                return

        for _agent in (getattr(cli.agent_coordinator, "plan_agent", None),
                       getattr(cli.agent_coordinator, "build_agent", None)):
            _pm = getattr(_agent, "permission_manager", None)
            if _pm is not None:
                _pm.input_handler = None

        app = TUIDepressionApp(coordinator=cli.agent_coordinator)
        sig["app"] = app

        permission_pending = {"future": None}

        async def tui_confirm(request, verdict) -> bool:
            """Bridge permission requests from agent workers to Textual UI."""
            if permission_pending["future"] is not None:
                return False

            result_future: Future = Future()
            permission_pending["future"] = result_future

            def show_modal() -> None:
                if result_future.done():
                    return

                modal = PermissionModal(request, verdict)

                def finished(result) -> None:
                    if not result_future.done():
                        result_future.set_result(bool(result))

                try:
                    app.push_screen(modal, callback=finished)
                except Exception:
                    if not result_future.done():
                        result_future.set_result(False)

            try:
                # Textual owns the UI loop. call_from_thread is the safe
                # boundary when the agent permission check is running in a
                # worker thread; direct execution is safe on the UI thread.
                if threading.current_thread() is threading.main_thread():
                    show_modal()
                else:
                    app.call_from_thread(show_modal)

                wrapped = asyncio.wrap_future(result_future)
                return bool(await asyncio.wait_for(wrapped, timeout=120.0))
            except asyncio.TimeoutError:
                if not result_future.done():
                    result_future.set_result(False)
                try:
                    app.call_from_thread(app.pop_screen)
                except Exception:
                    pass
                return False
            except asyncio.CancelledError:
                if not result_future.done():
                    result_future.set_result(False)
                raise
            finally:
                permission_pending["future"] = None

        # Install the bridge only after the coordinator exists and immediately
        # before Textual starts. This avoids the initialize/on_mount lifecycle
        # race that previously left every ASK verdict waiting on the wrong loop.
        for _agent in (getattr(cli.agent_coordinator, "plan_agent", None),
                       getattr(cli.agent_coordinator, "build_agent", None)):
            _pm = getattr(_agent, "permission_manager", None)
            if _pm is not None:
                _pm.set_confirm_callback(tui_confirm)
                _pm.input_handler = None

        try:
            await app.run_async()
        finally:
            await cli.shutdown()


if __name__ == "__main__":
    main()
