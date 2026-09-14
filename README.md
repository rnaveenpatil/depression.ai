# depression

AI-powered CLI agent for depression.ai - your intelligent terminal assistant

## Installation

```bash
# Install from source
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
pip install -e .
```

## Quick Start

```bash
# Start interactive mode
depression

# Run a single query
depression "run flutter build web in kks"

# Use a specific model
depression --model google/gemini-2.5-flash-lite "explain this code"

# Use a specific project directory
depression -p /path/to/project "refactor this code"
```

## Features

- **Dual-Agent Architecture**: Plan agent (read-only analysis) + Build agent (full tool access)
- **Multi-Model Support**: Gemini, DeepSeek, and more
- **Tool Calling**: Terminal, filesystem, git, search, web, and more
- **Session Management**: Persistent sessions with history
- **Snapshot/Undo**: Git-based snapshots for safe experimentation
- **Subagents**: Parallel task execution with @syntax

## Configuration

Create `.agent/config.json` in your project:

```json
{
  "llm": {
    "provider": "google",
    "model": "google/gemini-2.5-flash-lite"
  }
}
```

Or use environment variables:
```bash
export GOOGLE_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/snapshot [message]` | Create snapshot (git stash) |
| `/undo` | Undo last snapshot |
| `/plan` | Switch to plan mode |
| `/build` | Switch to build mode |

## Commands

```bash
depression                          # Start interactive mode
depression "your query"             # Run single query
depression -p /path/to/project      # Use specific project
depression --model MODEL            # Override model
depression --provider PROVIDER      # Use specific provider
depression --yolo                   # Auto-approve all tools
depression --session <id>           # Resume session
depression --new-session            # Force fresh session
depression --init-config            # Write default config
depression --verbose                # Verbose output
depression --quiet                  # Quiet mode
depression --no-mcp                 # Disable MCP
depression --help                   # Show help
```

## Project Structure

```
src/
├── agent/
│   ├── agent/           # Core agent logic
│   ├── cli/             # CLI interface
│   ├── config/          # Configuration
│   ├── context/         # Context management
│   ├── llm/             # LLM providers
│   ├── mcp/             # Model Context Protocol
│   ├── permissions/     # Permission system
│   ├── project/         # Workspace management
│   ├── tools/           # Built-in tools
│   └── main.py          # Entry point
```

## License

ISC License - see LICENSE file for details