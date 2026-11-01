"""Structural regression checks for the extracted TUI components."""
from agent.tui import app
from agent.tui import app_components as components


def test_extracted_widgets_keep_public_app_imports():
    for name in (
        "StreamingOutput", "PromptArea", "PermissionModal",
        "TextPromptModal", "ConfirmModal", "SessionPickerModal",
        "AgentCommandProvider", "LLMPanel", "ContextPanel",
        "TodoPanel", "LLMCostPanel", "HelpPanel",
    ):
        assert getattr(app, name) is getattr(components, name)


def test_extracted_visuals_are_not_rewritten():
    for name in (
        "StreamingOutput", "PromptArea", "PermissionModal",
        "TextPromptModal", "ConfirmModal", "LLMCostPanel", "HelpPanel",
    ):
        cls = getattr(app, name)
        assert cls.__module__ == "agent.tui.app_components"
    assert app.DepressionApp.__module__ == "agent.tui.app"
    assert app.PermissionModal.DEFAULT_CSS == components.PermissionModal.DEFAULT_CSS
    assert app.AgentCommandProvider.COMMANDS == components.AgentCommandProvider.COMMANDS
    assert app._slash_command_matches("/st") == components._slash_command_matches("/st")
