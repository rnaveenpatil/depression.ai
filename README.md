# depression.ai

**An open-source, general-purpose terminal harness for LLM agents.**

`depression.ai` connects the model you choose to your real terminal — files, commands, Git, the web, MCP servers and AWS — with planning, permissions and verification built around it.

It is **not** another “AI that does everything.” It is the harness that lets a model work inside the CLI safely and repeatably.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/tui.png" alt="depression.ai terminal interface" width="900">
</p>

<p align="center">
  <a href="https://youtu.be/en5f4pOvfL0"><b>▶ Watch the walkthrough demo</b></a>
</p>

---

## Contents

- [What is depression.ai?](#what-is-depressionai)
- [Terminal interface](#terminal-interface)
- [Quick start](#quick-start)
- [Add your credentials in the TUI](#add-your-credentials-in-the-tui)
- [How it works](#how-it-works)
- [What you can use it for](#what-you-can-use-it-for)
- [Advantages](#advantages)
- [Supported models](#supported-models)
- [Usage and cost tracking](#usage-and-cost-tracking)
- [CLI and TUI reference](#cli-and-tui-reference)
- [Install options](#install-options)
- [Privacy and security](#privacy-and-security)
- [Team](#team)
- [Demo](#demo)
- [License](#license)

---

## What is depression.ai?

In simple words:

> **You type a request. The harness gives your chosen model the context, tools and guardrails it needs to do real work in your terminal.**

A model on its own can only return text. A **harness** is the layer around the model that:

- reads the relevant project and session context,
- turns your request into a plan,
- exposes tools such as files, terminal, Git, web, MCP and AWS,
- asks permission before risky actions,
- checks whether the work actually succeeded,
- keeps a session you can inspect and resume.

`depression.ai` is that harness. The model is interchangeable.

### What it is

- A **general-purpose CLI/TUI harness** for tool-using models.
- **Open source** and inspectable.
- **Model-agnostic** — use a cloud API or a local model.
- **Local-first** — the agent, tools, sessions and permissions run on your machine.
- **Cross-platform** — Linux, macOS or Windows, on a laptop, VM, container or server.
- **Permission-aware** — sensitive actions require approval.
- **Verification-aware** — it does not just claim a change worked; it checks where possible.

### What it is not

- It is **not a model** and does not include one.
- It is **not a hosted service** and does not require a Team Depression account.
- It is **not magic** — results depend on the model, tools, credentials and permissions you provide.
- It is **not a replacement for AWS IAM** or your operating system’s security controls.

---

## Terminal interface

The main interface is a terminal UI built with **Textual**. You work with the agent interactively, without leaving the CLI.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/User%20Interface%20%281%29.png" alt="depression.ai start screen" width="900">
</p>

<p align="center"><em>Start screen — type a prompt to begin, or run <code>depression --nosignin</code> to enter without Google sign-in.</em></p>

Everything stays in one workspace:

- Interactive conversation
- Plan / Build / Auto mode switching
- LLM connection configuration
- AWS credential configuration
- Tool-call cards
- Permission prompts
- Todo tracking
- Thinking indicators
- Diff inspection
- Session management
- Project context
- Agent execution feedback

---

## Quick start

### Requirements

- A terminal
- **Python 3.10+**
- `pip`
- An API key for a cloud model **or** a local OpenAI-compatible model server
- Optional: AWS credentials for AWS tasks
- Optional: Playwright for browser automation

No GPU is required for cloud models. A GPU is only useful if you run a local model.

### Install and run

```bash
pip install depression.ai
depression
```

To skip the optional Google sign-in screen:

```bash
depression --nosignin
```

You can also start with a project and a model:

```bash
depression --nosignin --project /path/to/project --model your-model
```

On first run, connect your model in the **LLM CONNECTION** panel. That is the only required setup.

---

## Add your credentials in the TUI

You do **not** need to edit configuration files to get started. The TUI includes panels for the credentials most people need.

### 1. LLM credentials

Open the panel with:

```text
/connect
```

or:

```text
/model
```

Enter:

| Field | What to provide |
|---|---|
| **Base URL** | Your provider’s OpenAI-compatible endpoint, for example `https://api.example.com/v1` |
| **API Key** | Your provider API key |
| **Model ID** | The exact model name, for example `provider/model-name` |

Then:

- Press **CONNECT** to use the model immediately and save the connection.
- Press **DISCOVER /models** to ask the endpoint for available model IDs.
- Check the status line for errors.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/LLM%20Connection%20Configuration.png" alt="LLM connection panel" width="900">
</p>

<p align="center"><em>The LLM CONNECTION panel — Base URL, API key and model are entered and saved from inside the TUI.</em></p>

For a local server such as Ollama, use:

```text
Base URL : http://localhost:11434/v1
API Key  : local
Model ID : your-local-model
```

### 2. AWS credentials

Open the panel with:

```text
/aws
```

Enter:

| Field | What to provide |
|---|---|
| **Access Key ID** | Your AWS access key ID |
| **Secret Access Key** | Your AWS secret access key |
| **Region** | The AWS region to use |

Press **SAVE**.

The TUI:

- stores the credentials in the local environment file with restricted permissions,
- makes them available to AWS tools and subprocesses,
- refreshes the AWS MCP configuration when supported,
- verifies that the credentials were written before reporting success.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/AWS%20Authentication%20And%20Account%20Access.png" alt="AWS credentials panel" width="900">
</p>

<p align="center"><em>The AWS credentials panel — access key, secret and region are configured without leaving the terminal.</em></p>

> **AWS recommendation:** use short-lived credentials or an IAM role where possible. If you use an IAM user access key, grant only the permissions your workflow needs.

### 3. Google sign-in is optional

Google sign-in is **not required** to use the harness. It only personalizes the local profile.

Use `depression --nosignin` to start with a local guest profile, or sign in later from:

```text
/profile
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/User%20Profile.png" alt="depression.ai signed-in user profile panel" width="900">
</p>

<p align="center"><em>Signed-in profile — Google identity, verification status and login metadata.</em></p>

The agent, tools, sessions and permissions work without Google sign-in.

### 4. Environment variables for automation

The TUI is the easiest path, but environment variables are supported for CI, servers and scripted setups:

```bash
export DEPRESSION_BASE_URL="https://api.example.com/v1"
export DEPRESSION_API_KEY="your-api-key"
export DEPRESSION_MODEL="your-model"

export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-south-1"

depression --nosignin
```

Never commit real credentials or `.env` files to Git.

---

## How it works

```text
Your request
     │
     ▼
┌──────────────────────────────┐
│ Context                      │
│ project files, session, tools│
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Plan                         │
│ understand the task          │
│ break it into steps          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Build                        │
│ choose and call tools        │
│ read, edit, run, inspect     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Permissions                  │
│ ask before risky actions     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Verification                 │
│ check the result             │
│ fix or report honestly       │
└──────────────┬───────────────┘
               ▼
            Result
```

### The loop

1. **Understand** — read the request and relevant project context.
2. **Plan** — decide what needs to happen.
3. **Act** — call tools such as terminal, filesystem, Git, web, MCP or AWS.
4. **Observe** — inspect tool output and adapt.
5. **Verify** — check that the requested result exists.
6. **Report** — tell you what changed and what did not.

### Plan, Build and Auto modes

| Mode | Purpose |
|---|---|
| **Plan** | Inspect and plan without executing the plan |
| **Build** | Execute with tools |
| **Auto** | Plan, then execute, and re-plan if needed |

Switch modes in the TUI:

```text
/plan
/build
/auto
```

### Permissions

Risky actions go through an approval prompt. You can see the exact tool call before it runs.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/Tool%20Permission%20and%20%20Execution%20Confirmation.png" alt="Tool permission prompt" width="900">
</p>

<p align="center"><em>Tool permission prompt — inspect the action before approving it.</em></p>

You can also stop a running turn:

```text
/stop
```

or press `Esc` to interrupt.

---

## What you can use it for

`depression.ai` is general purpose, but it is still a terminal tool. It is useful when work needs to happen on a real machine.

### Software work

- Understand an unfamiliar codebase
- Find the cause of a build or test failure
- Make focused edits and patches
- Run tests and checks
- Review a diff
- Refactor without changing behaviour

### System work

- Inspect project structure
- Check running processes and ports
- Diagnose failing commands
- Inspect the local environment
- Prepare a project for deployment

### Git work

- Review changes
- Explain a diff
- Create a patch
- Prepare a clean commit
- Undo with snapshots where supported

### Web and research

- Fetch documentation
- Search the web
- Inspect a page with browser automation
- Compare information across sources

### MCP and integrations

- Connect MCP servers
- Use external tools through the same permission model
- Extend the harness with plugins

### AWS work

- Inspect configured AWS resources
- Investigate deployment problems
- Work with the AWS CLI and supported AWS tools
- Operate only within the permissions of your AWS identity

### Example requests

```text
Analyze this project and explain its architecture.

Find the cause of this failing test and fix it.

Review my Git diff and summarize the risks.

Find the process using port 8000.

Check why this command is failing.

Inspect my AWS environment and find the deployment problem.

Prepare this project for deployment.
```

The harness does not guarantee the result. A weak model, a missing tool, a denied permission or an invalid credential can still fail.

---

## Advantages

| Advantage | Why it matters |
|---|---|
| **Model-agnostic** | Use a cloud API or a local model; change models without changing the harness |
| **Terminal-native** | Works where developers already work — no separate IDE or web app required |
| **Open source** | Inspect, audit, fork and modify the code |
| **Any machine** | Python 3.10+ and a terminal are enough; no dedicated infrastructure |
| **Local-first** | Agent execution, tools, permissions and sessions run on your machine |
| **No account required** | Start with `--nosignin` and your own model credentials |
| **Credential setup in the TUI** | Connect LLM and AWS credentials without editing files |
| **Permission controls** | Review risky actions before they run |
| **Verification** | The harness checks results instead of only trusting model text |
| **Plan + Build** | Separates reasoning from execution |
| **Sessions and recovery** | Resume work and inspect what happened |
| **MCP support** | Bring external tools into the same workflow |
| **AWS-aware** | Work with cloud resources under your existing IAM permissions |
| **Lightweight install** | `pip install depression.ai` |

### Why use a harness instead of a plain chatbot?

A chatbot can explain what to do. A harness can:

- open the actual files,
- run the actual command,
- observe the actual error,
- make the actual change,
- run the actual verification,
- report what actually happened.

That difference matters when the task is not just a question — it is a change that has to work on a real system.

---

## Supported models

`depression.ai` supports:

- **OpenAI-compatible APIs** — most cloud gateways and local servers
- **Anthropic**
- **Google Gemini**
- **Ollama** and other local OpenAI-compatible servers

The model does the reasoning. The harness provides the execution environment.

If you use a remote model, the prompt and relevant context are sent to that provider. Check the provider’s privacy and retention policy before sending sensitive information.

---

## Usage and cost tracking

The TUI tracks token usage and API cost per query and per session. Open the live panel with `/llmcost` to see input/output tokens, total cost, API calls, tool calls and query duration.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/LLM%20Usage%20and%20Cost%20Tracking.png" alt="depression.ai LLM usage and cost tracking panel" width="900">
</p>

<p align="center"><em>LLM usage and cost tracking — tokens, API calls, tool calls and query duration at a glance.</em></p>

---

## CLI and TUI reference

### Start the TUI

```bash
depression
```

### Common flags

| Flag | Purpose |
|---|---|
| `--nosignin` | Skip Google sign-in and use a local guest profile |
| `-p`, `--project PATH` | Set the project directory |
| `-m`, `--model MODEL` | Select a model |
| `--provider PROVIDER` | Select a provider |
| `--new-session` | Start a new session |
| `--session ID` | Resume a session |
| `--no-mcp` | Disable MCP |
| `--max-turns N` | Limit agent turns |
| `--timeout SECONDS` | Set the CLI query timeout |

### TUI commands

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/Command%20Reference%20Interface.png" alt="depression.ai command reference and help panel" width="900">
</p>

<p align="center"><em>Command reference — available TUI commands, CLI flags and keyboard shortcuts.</em></p>

| Command | Action |
|---|---|
| `/connect` | Open the LLM connection panel |
| `/model` | Open the LLM panel and discover models |
| `/models` | Discover models from the configured endpoint |
| `/aws` | Open the AWS credentials panel |
| `/profile` | Open the profile panel |
| `/context` | Show context usage |
| `/llmcost` | Show token and API cost usage |
| `/todo` | Show the todo panel |
| `/plan` | Switch to Plan mode |
| `/build` | Switch to Build mode |
| `/auto` | Switch to Auto mode |
| `/run CMD` | Run a shell command and show output |
| `/stop` | Ask the running agent to stop and summarize |
| `/clear` | Clear the transcript |
| `/quit` | Exit |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Submit prompt |
| `Shift+Enter` / `Ctrl+J` | New line in the prompt |
| `Tab` | Focus prompt or cycle mode |
| `Ctrl+B` | Show or hide the sidebar |
| `Esc` | Interrupt the running agent |
| `Ctrl+L` | Clear the transcript |
| `Ctrl+C` | Interrupt if busy, quit if idle |
| `Ctrl+Q` | Quit |

---

## Install options

### From PyPI

```bash
pip install depression.ai
depression
```

### From source

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
pip install -e .
depression
```

### Development

```bash
pip install -e ".[dev]"
pytest
```

### Optional browser automation

```bash
pip install "depression.ai[browser]"
playwright install
```

### Platform notes

- **Linux:** any modern terminal.
- **macOS:** Terminal or iTerm2.
- **Windows:** Windows Terminal or WSL.
- **Remote machines:** SSH into the machine and run the CLI there.

No GPU is required unless you choose to run a local model that needs one.

---

## Privacy and security

### What stays local

The harness and its tool actions run on your machine:

- Agent execution
- Filesystem and terminal actions
- Git operations
- Sessions and local project context
- Credential storage
- Permission handling
- MCP server communication you configure

### Credentials

- LLM and AWS credentials can be entered through the TUI.
- Credentials are stored locally on your machine.
- Credentials are used only to connect to the services you configure.
- Team Depression does not receive your credentials.
- Never commit API keys, AWS secrets, tokens or `.env` files to Git.

### Remote models

If you use a remote LLM, the request and the context needed to answer it leave your machine and go to that provider. This can include prompts, selected project content and tool results.

> The agent stays local. The model request does not. Read your provider’s policy before sending sensitive information to a remote model.

### Local models

With a local OpenAI-compatible server, inference stays on your machine or network.

### Google sign-in

Google sign-in is optional. It is used for local profile personalization and is not required for the agent, tools or sessions.

### Open source and auditability

The source is available for inspection. No software can guarantee perfect safety: review the code, permissions, tools, AWS policies and model provider settings before using it with sensitive systems.

---

## Project structure

```text
src/
└── agent/
    ├── agent/        # Plan, Build and loop logic
    ├── cli/          # CLI input and UI
    ├── config/       # Configuration
    ├── context/      # Context management
    ├── llm/          # Model providers and runtime
    ├── mcp/          # MCP integration
    ├── permissions/  # Permission handling
    ├── plugins/      # Plugin system
    ├── project/      # Project and Git helpers
    ├── session/      # Session state
    ├── storage/      # Local storage
    ├── tools/        # Tool implementations
    ├── tui/          # Textual terminal UI
    └── utils/        # Shared utilities
```

---

## Project status

`depression.ai` is an actively developed open-source project. The current release is a beta.

Working today:

- Terminal TUI and CLI
- Plan, Build and Auto modes
- Tool execution and permissions
- Filesystem, terminal, Git, web and search tools
- MCP support
- AWS workflows
- Persistent sessions and snapshots
- TUI credential setup for LLM and AWS
- Local model support through OpenAI-compatible servers

---

## Team

> A team that builds ON ETHICS

### Academic institution and project context

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/mit_logo.png" alt="Moodlakatte Institute of Technology (MIT), Kundapura" width="220">
</p>

<p align="center">
  <b>Moodlakatte Institute of Technology (MIT), Kundapura</b><br>
  Department of Information Science & Engineering (ISE)
</p>

`depression.ai` is being developed as an **academic major project** by final-year Information Science & Engineering (ISE) students of Moodlakatte Institute of Technology (MIT), Kundapura.

The project combines AI agents, software engineering, system automation, cloud operations and LLM integration.

**College**

- **Moodlakatte Institute of Technology (MIT), Kundapura**
- **Department:** Information Science & Engineering (ISE)
- **Project type:** Academic major project
- **Team:** Team Depression

**College LinkedIn**

- https://www.linkedin.com/company/mitkundapura/posts/

### R Naveen Patil

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/patil.png" alt="R Naveen Patil" width="180" height="180">
</p>

**Lead Developer — AI Agents, Cloud & Infrastructure**

R Naveen Patil is the main developer of `depression.ai`. He built the core architecture, agent workflow, LLM integration, system tools, cloud integration and overall project.

**Profiles**

- Email: `rnaveenpatil@gmail.com`
- [LinkedIn](https://www.linkedin.com/in/r-naveen-patil-a45403292/)
- [GitHub](https://github.com/rnaveenpatil)
- [Portfolio](https://rnaveenpatil-resume.netlify.app)

### Team members

<table>
<tr>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/rahul.jpeg" alt="Rahul Jadav" width="120" height="120">

### Rahul Jadav

Email: `rjadav214@gmail.com`

[GitHub](https://github.com/Rahul-Jadav-0)

[LinkedIn](https://www.linkedin.com/in/subzero91/)

</td>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/meghasree.png" alt="K G Meghashree Naik" width="120" height="120">

### K G Meghashree Naik

Email: `meghashree.kg13@gmail.com`

[GitHub](https://github.com/Meghashree-13)

[LinkedIn](https://www.linkedin.com/in/k-g-meghashree/)

</td>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/sumantha.jpeg" alt="SUMANTHA MS" width="120" height="120">

### SUMANTHA MS

Email: `mssumanth836@gmail.com`

[LinkedIn](https://www.linkedin.com/in/sumantha-ms-235ba2295/)

</td>

</tr>
</table>

---

## Demo

<p align="center">
  <a href="https://youtu.be/en5f4pOvfL0">
    <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/youtube-demo-thumbnail.png" alt="Watch the depression.ai demo" width="900">
  </a>
</p>

<p align="center">
  <b><a href="https://youtu.be/en5f4pOvfL0">▶ Watch the walkthrough on YouTube</a></b>
</p>

---

## Repository

- GitHub: https://github.com/rnaveenpatil/depression.ai
- Issues: https://github.com/rnaveenpatil/depression.ai/issues

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**depression.ai** — an open-source general-purpose terminal harness for LLM agents.

Built by **Team Depression**.

</div>
