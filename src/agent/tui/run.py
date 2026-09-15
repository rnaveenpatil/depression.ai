"""Fully bootstrapped Textual entry point."""
from __future__ import annotations

import asyncio
import sys
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

    async def boot() -> None:
        await cli.initialize(args)
        from agent.tui.app import DepressionApp
        app = DepressionApp(coordinator=cli.agent_coordinator)
        try:
            await app.run_async()
        finally:
            await cli.shutdown()

    asyncio.run(boot())


if __name__ == "__main__":
    main()
