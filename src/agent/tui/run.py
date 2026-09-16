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
from typing import Any

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
        configure_runtime_provider(
            get_llm_registry(),
            runtime["base_url"],
            runtime["api_key"],
            runtime["model"],
        )

    async def boot() -> None:
        loop = asyncio.get_running_loop()

        await cli.initialize(args)

        from agent.tui.app import DepressionApp, PermissionModal

        app = DepressionApp(coordinator=cli.agent_coordinator)

        # One-at-a-time permission bridge, resolved on the UI loop.
        pending: dict[str, Any] = {"fut": None}

        async def tui_confirm(request: Any, verdict: Any) -> bool:
            if pending["fut"] is not None:
                return False

            fut: asyncio.Future[bool] = loop.create_future()
            pending["fut"] = fut

            def show_modal() -> None:
                if fut.done():
                    return
                modal = PermissionModal(request, verdict)

                def finished(result: Any) -> None:
                    if not fut.done():
                        fut.set_result(bool(result))

                try:
                    app.push_screen(modal, callback=finished)
                except Exception:
                    if not fut.done():
                        fut.set_result(False)

            try:
                # `call_from_thread` is the only safe way to touch Textual
                # state from a non-UI thread; on the UI thread it must run
                # inline.
                if threading.current_thread() is threading.main_thread():
                    show_modal()
                else:
                    app.call_from_thread(show_modal)

                return bool(await asyncio.wait_for(
                    asyncio.shield(fut), timeout=120.0))
            except asyncio.TimeoutError:
                if not fut.done():
                    fut.set_result(False)
                try:
                    app.call_from_thread(app.pop_screen)
                except Exception:
                    pass
                return False
            except asyncio.CancelledError:
                if not fut.done():
                    fut.set_result(False)
                raise
            finally:
                pending["fut"] = None

        # Install the callback before the app runs so every ASK verdict
        # lands here, not in the legacy threading fallback.
        app.install_permission_callback(tui_confirm)

        try:
            await app.run_async()
        finally:
            await cli.shutdown()

    asyncio.run(boot())


if __name__ == "__main__":
    main()