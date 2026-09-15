"""TUI entry point for Depression.AI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="depression-tui",
        description="Depression.AI — terminal agentic workspace",
    )
    parser.add_argument("-p", "--project", help="Project directory")
    parser.add_argument("-m", "--model",   help="Override model")
    parser.add_argument("--provider",      help="Override provider")
    parser.add_argument("--yolo",   action="store_true", help="Auto-approve tools")
    parser.add_argument("--session",       help="Resume session ID")
    parser.add_argument("--no-sidebar", action="store_true", help="Compatibility")
    args = parser.parse_args()

    from agent.tui.app import DepressionApp

    DepressionApp().run()


if __name__ == "__main__":
    main()