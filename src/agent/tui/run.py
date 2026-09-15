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

    cli = CLIAgent()
    args = cli.parse_arguments()
    # TUI owns presentation; suppress the legacy Rich CLI banner/output.
    args.non_interactive = True
    args.query = None
    args.quiet = True

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
