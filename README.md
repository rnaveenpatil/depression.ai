# depression

**AI agent for everything — development, deployment, and analysis.**

A terminal-native agentic AI assistant. Sits in your terminal as a Textual TUI and
drives a **Plan + Build** dual-agent system with full tool access, MCP support,
and visual permission prompts.

## Features

- **Textual TUI** — interactive terminal UI with plan/build mode switching (`Ctrl+P`), AWS credentials panel, thinking indicator, todo panel, live tool-call cards, and diff view
- **Dual-Agent Architecture** — Plan agent (read-only analysis) + Build agent (full tool execution)
- **User-Configured LLM Runtime** — connect any OpenAI-compatible endpoint at runtime (base URL + API key + model); built-in provider adapters for Anthropic, Groq, NVIDIA, and local models
- **Tool Calling** — terminal, filesystem, git, patch, search, web, todo, process, browser automation (Playwright), AWS
- **MCP Support** — Model Context Protocol client with multiple servers
- **Session Management** — persistent sessions stored in SQLite; resume with `/session`
- **Snapshot / Undo** — git-stash-based snapshots for safe experimentation
- **Subagents & Plugins** — parallel task execution and `@syntax` subagents, plugin loader
- **Permission System** — per-tool approval rules with interactive modal prompts

## Installation

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
pip install -e .
```

Optional extras:

```bash
# Browser automation tool (Playwright)
pip install -e ".[browser]"
playwright install chromium

# Development (pytest, ruff, mypy)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Launch the TUI
depression

# Equivalent forms
python -m agent.tui
python -m agent.tui.run
```

Before first use you must connect an LLM endpoint (see [Runtime LLM Configuration](#runtime-llm-configuration)
below). The TUI also offers an **LLM CONNECTION** panel to set base URL, API key,
and model interactively.

## Runtime LLM Configuration

The agent is deliberately provider-agnostic: there are **no hardcoded API keys**.
Configure a runtime provider via environment variables or the in-TUI connection
panel. Copy `.env.example` to `.env`:

```bash
# User runtime LLM connection (never commit real secrets)
DEPRESSION_PROVIDER=custom
DEPRESSION_BASE_URL=https://api.example.com/v1
DEPRESSION_API_KEY=
DEPRESSION_MODEL=

# AWS credentials entered by /aws panel
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=ap-south-1
```

## Configuration

Layered config (project overrides user): `.agent/config.json` in your project,
or `~/.agent/config.json` globally.

```json
{
  "llm": {
    "provider": "custom",
    "model": "your-model-id"
  }
}
```

Write a default config with `depression --init-config` (CLI bootstrap) or edit
`.agent/config.json` directly.

## Slash Commands

Available inside the TUI input box:

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/plan` | Switch to Plan mode (read-only analysis) |
| `/build` | Switch to Build mode (full execution) |
| `/auto` | Automatic Plan -> Build loop |
| `/snapshot [message]` | Create a git-stash snapshot |
| `/undo` | Restore the last snapshot |
| `/model` · `/provider` | View / switch model or provider |
| `/session` | Show / resume sessions |
| `/compact` | Compact conversation context |
| `/tools` · `/permissions` | List tools and permission rules |
| `/plugins` | Load plugins |
| `/metrics` · `/cost` · `/tokens` | Usage metrics and cost |
| `/config` · `/set` | View / set config values |
| `/exit` | Quit (also `Ctrl+Q`) |

## TUI

- **Mode toggle**: `Ctrl+P` cycles Plan → Build
- **Permissions**: tool requests surface as interactive modal prompts (accept / deny)
- **AWS panel**: manage AWS credentials from inside the UI
- **Live panes**: thinking indicator, todo list, tool-call cards, diff view, sidebar

## CLI Flags

The `depression` entry point parses the following flags (used in CLI bootstrap
and overridable before the TUI starts):

```bash
depression -p /path/to/project      # project directory
depression -m MODEL                 # override model
depression --provider PROVIDER      # override provider
depression --yolo                   # auto-approve all tool calls
depression --session <id>           # resume session
depression --new-session            # force a fresh session
depression --max-turns N            # agent loop limit
depression --no-mcp                 # disable MCP
depression -v / -q                  # verbose / quiet
depression --skip-tui               # (reserved) CLI-only mode
```

## Project Structure

```
src/
└── agent/
    ├── agent/           # Plan + Build dual-agent orchestration, planner, loop
    ├── cli/             # Classic CLI bootstrap (argparse, prompt-toolkit UI, slash commands)
    ├── config/          # Layered configuration loader
    ├── context/         # Context manager, compaction, file indexing
    ├── llm/             # Runtime LLM registry + provider adapters, rate limiter
    ├── mcp/             # Model Context Protocol client + AWS config
    ├── permissions/     # Tool permission policy & approval
    ├── plugins/         # Plugin loader (builtin)
    ├── project/         # Workspace management
    ├── session/         # Session persistence
    ├── storage/         # SQLite database + cache
    ├── tools/           # Built-in tools (terminal, git, filesystem, web, browser, AWS, ...)
    ├── tui/             # Textual terminal UI (app, widgets, theme)
    └── utils/           # Logging, platform, env helpers, redaction
```

## Development

```bash
pip install -e ".[dev]"
pytest                       # run test suite
ruff check src/              # lint
ruff format --check src/     # formatting
mypy src/                    # type checking
```

## License

ISC License.