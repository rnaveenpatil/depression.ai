"""Launcher for the terminal-first Depression.AI TUI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from textual import work
from textual.widgets import Input

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


class TerminalLauncherMixin:
    """Adapt the legacy agent bootstrap to the sidebar-free terminal shell."""

    def on_mount(self) -> None:
        from agent.tui.theme import apply_opencode_theme
        from agent.tui.views.chat import ChatView
        from agent.tui.views.status_bar import StatusBar

        apply_opencode_theme(self)
        self.title = "DEPRESSION.AI"
        self.sub_title = "TERMINAL AGENT"
        chat = self.query_one("#chat-panel", ChatView)
        chat.add_system("**DEPRESSION.AI**  ·  terminal agent\n\nType a task.  `/models` selects a model · `/connect` configures it · `!` is reserved for shell commands.")
        chat.add_divider()
        self.query_one("#status-bar", StatusBar).update_all(status="starting", model=self.model_name, tokens=0, cost=0.0)
        self._init_task = self.run_worker(self._init_agent_async(), exclusive=True, group="agent-init", thread=True)
        self.set_interval(0.2, self._focus_prompt)

    def _focus_prompt(self) -> None:
        try:
            self.query_one("#chat-input", Input).focus()
        except Exception:
            pass

    def _on_agent_ready(self, coordinator, session, workspace, session_manager):
        from agent.tui.views.chat import ChatView
        from agent.tui.views.header import HeaderBar

        self.coordinator = coordinator
        self.session_manager = session_manager
        self.workspace = workspace
        status = coordinator.get_status()
        plan = status.get("plan_agent", {}).get("model", "—")
        build = status.get("build_agent", {}).get("model", "—")
        self.set_model(f"Plan: {plan} | Build: {build}")
        self.set_session(session.id)
        self.query_one("#chat-panel", ChatView).add_system(f"✓ Agent connected · session `{session.id[:8]}`")
        self.query_one("#chat-panel", ChatView).add_divider()
        self.query_one("#header", HeaderBar).set_mode(self.current_mode)


class TerminalApp(TerminalLauncherMixin):
    pass


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

    # Mixin-first subclass keeps the new terminal layout while replacing the
    # old sidebar-dependent startup hook and preserving the agent lifecycle.
    class PatchedTerminalTUI(TerminalLauncherMixin, TerminalDepressionTUI):
        pass

    PatchedTerminalTUI(
        project_dir=args.project,
        model_override=args.model,
        provider_override=args.provider,
        yolo=args.yolo,
        no_sidebar=True,
        session_id=args.session,
    ).run()


if __name__ == "__main__":
    main()
