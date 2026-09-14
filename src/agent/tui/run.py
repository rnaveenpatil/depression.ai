"""
TUI Entry Point

Launches the futuristic TUI for depression.ai.

Usage:
    python -m agent.tui.run              # Launch TUI
    python -m agent.tui.run --no-sidebar # Launch without sidebar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path setup
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main():
    """Launch the TUI."""
    parser = argparse.ArgumentParser(
        prog="depression-tui",
        description="Depression.AI — Opencode-style Neural Coding Agent TUI",
    )
    parser.add_argument("-p", "--project", help="Project directory")
    parser.add_argument("-m", "--model", help="Override model")
    parser.add_argument("--provider", help="Override provider")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve all tools")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--no-sidebar", action="store_true", help="Hide sidebar")

    args = parser.parse_args()

    # Import here to avoid import issues when module is used as library
    from agent.tui.app import DepressionTUI

    # Create the app with args passed for initialization
    app = DepressionTUI(
        project_dir=args.project,
        model_override=args.model,
        provider_override=args.provider,
        yolo=args.yolo,
        no_sidebar=args.no_sidebar,
        session_id=args.session,
    )

    # Run the TUI (agent initialization happens in on_mount)
    app.run()


if __name__ == "__main__":
    main()
