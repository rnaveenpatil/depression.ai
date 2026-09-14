"""Launcher for the terminal-first Depression.AI TUI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="depression-tui", description="Depression.AI terminal agent")
    parser.add_argument("-p", "--project", help="Project directory")
    parser.add_argument("-m", "--model", help="Override model")
    parser.add_argument("--provider", help="Override provider")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve all tools")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--no-sidebar", action="store_true", help="Compatibility option")
    args = parser.parse_args()

    from agent.tui.terminal_app import TerminalDepressionTUI

    TerminalDepressionTUI(
        project_dir=args.project,
        model_override=args.model,
        provider_override=args.provider,
        yolo=args.yolo,
        no_sidebar=True,
        session_id=args.session,
    ).run()


if __name__ == "__main__":
    main()
