# depression

### `You A-Z. AI depression.ai`

## 👥 Team Depression

> **A team that builds first, breaks things second, and fixes them before anyone notices.**

### 🧑‍💻 R Naveen Patil

<img src="https://media.licdn.com/dms/image/v2/D5603AQFPvuEfXgu3kw/profile-displayphoto-crop_800_800/B56ZvjDsyIIMAI-/0/1769040958923?e=1792627200&v=beta&t=88VVSkWxkIxiOZRLuEm3y38JJdodm61VwcNRfDHZgGQ" width="180" height="180" style="border-radius:50%;">

**Lead Developer • AI & Agentic Systems • Cloud & Infrastructure**

R Naveen Patil is the primary developer behind the `depression.ai` framework, responsible for the core architecture, agentic workflow, LLM integration, system tooling and overall development of the project.

**Profiles**

* 📧 `rnaveenpatil@gmail.com`
* 🔗 [LinkedIn](https://www.linkedin.com/in/r-naveen-patil-a45403292/)
* 💻 [GitHub](https://github.com/rnaveenpatil)
* 🌐 [Portfolio](https://rnaveenpatil-resume.netlify.app)

---

### 🤝 Teammates

<table>
<tr>
<td align="center" width="33%">

<img src="https://media.licdn.com/dms/image/v2/D5603AQHETfTo9AbZhg/profile-displayphoto-crop_800_800/B56Z0cSxd9H0AI-/0/1774296177050?e=1792627200&v=beta&t=BI2xg5WhBCmr2IT9QCdM5bQaKlySbx3vaGbmwSt1UNs" width="110" height="110" style="border-radius:50%;">

### Rahul Jadav

📧 `rjadav214@gmail.com`

[GitHub](https://github.com/Rahul-Jadav-0)

[LinkedIn](https://www.linkedin.com/in/subzero91/)

</td>

<td align="center" width="33%">

<img src="https://media.licdn.com/dms/image/v2/D5603AQGRWiuUJxvRAg/profile-displayphoto-crop_800_800/B56ZoKX7S5I4AQ-/0/1761110638482?e=1792627200&v=beta&t=3bO-r6Uze3TfRaILZaYJciFqfbSPLvgQhCtH-17XJx4" width="110" height="110" style="border-radius:50%;">

### K G Meghashree Naik

📧 `meghashree.kg13@gmail.com`

[GitHub](https://github.com/Meghashree-13)

[LinkedIn](https://www.linkedin.com/in/k-g-meghashree/)

</td>

<td align="center" width="33%">

### SUMANTHA MS

📧 `mssumanth836@gmail.com`

[LinkedIn](https://www.linkedin.com/in/sumantha-ms-235ba2295/)

</td>
</tr>
</table>

---

# 🧠 About depression.ai

**depression.ai** is an agentic AI framework developed by **Team Depression**.

It is a terminal-native AI workspace designed to assist with the complete workflow from **development and analysis to system operations and cloud deployment**.

```text
                 ┌─────────────────────┐
                 │        USER          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   depression.ai     │
                 │      TUI / CLI      │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       ┌──────────────┐           ┌──────────────┐
       │  PLAN AGENT  │           │ BUILD AGENT  │
       │  Read / Plan │           │ Execute      │
       └──────┬───────┘           └──────┬───────┘
              │                          │
              └───────────┬──────────────┘
                          ▼
                 ┌─────────────────────┐
                 │    TOOL SYSTEM      │
                 ├─────────────────────┤
                 │ Terminal            │
                 │ Filesystem          │
                 │ Git                 │
                 │ Patch               │
                 │ Search              │
                 │ Web                 │
                 │ Browser             │
                 │ AWS                 │
                 │ MCP                 │
                 └─────────────────────┘
```

## 🚀 What Can It Do?

### 💻 Development

Work directly with software projects through the terminal.

* Analyze existing projects
* Read and understand source code
* Create files
* Modify existing files
* Apply patches
* Run commands
* Debug development workflows
* Work with Git
* Manage project structure

### 🖥️ System Operations

The agent can interact with the local development environment through configured tools.

Examples:

```text
"Check my project structure"

"Find why this application is failing"

"Install the required dependencies"

"Run the tests and fix the errors"

"Find all unused files"

"Prepare this project for deployment"
```

The agent determines the required actions from the task rather than relying only on predefined command keywords.

---

# 🧩 Plan + Build Architecture

`depression.ai` uses two complementary agent modes.

### PLAN

The Plan Agent focuses on understanding the problem before execution.

```text
User Request
     ↓
Project Inspection
     ↓
Analysis
     ↓
Task Breakdown
     ↓
Execution Plan
```

### BUILD

The Build Agent performs the required operations using available tools.

```text
Approved Plan
     ↓
Tool Selection
     ↓
Execution
     ↓
Verification
     ↓
Result
```

This separation helps distinguish **reasoning/planning** from **actual system modifications**.

---

# ☁️ AWS Integration

AWS is integrated into the agent workflow so cloud infrastructure can be handled from the same terminal-native environment.

Instead of switching between multiple terminals, dashboards and command-line utilities, the user can communicate the intended task to the agent.

```text
User
 │
 │ "Analyze my AWS environment"
 ▼
depression.ai
 │
 ├── AWS credentials
 ├── AWS tools
 ├── AWS CLI workflows
 └── Agent reasoning
 │
 ▼
AWS Environment
```

Depending on the tools available, AWS permissions and configured credentials, the agent can assist with:

* AWS resource inspection
* Infrastructure analysis
* Resource management
* Deployment workflows
* Cloud operations
* AWS CLI operations
* Monitoring and troubleshooting workflows
* Creating, modifying or removing supported resources
* Application deployment workflows

### 🔐 AWS Permissions Matter

`depression.ai` does not bypass AWS security.

The operations available to the agent depend on the **AWS credentials and IAM permissions** provided by the user.

For example, if the configured IAM identity cannot delete an EC2 resource, the agent cannot legitimately perform that deletion.

---

# 🔌 Connecting Your AWS Account

AWS credentials can be configured through the TUI or the environment.

Example environment configuration:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-south-1"
```

Then start:

```bash
depression
```

The TUI provides an AWS configuration panel where the AWS environment can be connected and used by the agent.

> **Never commit AWS credentials, secret keys or `.env` files to Git.**

---

# 🤖 LLM Integration

The framework separates the **agent runtime** from the **LLM provider**.

The user can configure the model at runtime using:

```text
Provider
Base URL
API Key
Model
```

Example:

```text
Provider: custom
Base URL: https://api.example.com/v1
API Key: ****************
Model: your-model
```

For a locally hosted model:

```text
Provider: custom
Base URL: http://localhost:11434/v1
API Key: local
Model: your-model
```

This allows the same agent framework to work with remote APIs as well as locally hosted model servers when they expose a compatible interface.

---

# 🧠 Model Requirements

API compatibility alone does not guarantee good agent performance.

A model used with `depression.ai` should ideally support:

* Chat/text generation
* Long enough context
* Strong instruction following
* Code understanding
* Structured output
* Tool/function calling where available
* Multi-turn conversations
* Reliable reasoning
* Consistent responses

For local deployment, hardware resources also matter.

```text
Local GPU
   ↓
Model Runtime
   ↓
Local LLM
   ↓
depression.ai
   ↓
Tools + Agent
```

Larger models generally require more VRAM/RAM and compute resources.

---

# 🔒 Local & Self-Hosted AI

The existing architecture keeps the **agent framework and tool execution on the user's system** while the model can be supplied through an API.

Team Depression is extending this architecture toward **local LLM execution**.

The target architecture is:

```text
┌─────────────────────────────────────────┐
│              ORGANIZATION               │
│                                         │
│  ┌─────────────┐     ┌───────────────┐ │
│  │ depression  │────▶│  Local LLM    │ │
│  │    .ai      │     │ GPU Server    │ │
│  └──────┬──────┘     └───────────────┘ │
│         │                               │
│         ├── Filesystem                  │
│         ├── Terminal                    │
│         ├── Git                         │
│         ├── Internal Tools              │
│         └── Infrastructure              │
│                                         │
└─────────────────────────────────────────┘
```

This architecture is intended to support environments where sensitive engineering documents, source code, internal correspondence and other confidential information need to remain within controlled infrastructure.

---

# 🛠️ Tool System

The framework provides the agent with tools for interacting with the working environment.

Current tool categories include:

```text
Terminal
Filesystem
Git
Patch
Search
Web
Todo
Process
Browser Automation
AWS
MCP
```

The LLM decides which available tools are relevant to the current task.

---

# 🔗 MCP Support

`depression.ai` includes Model Context Protocol support for connecting additional tool servers.

This allows the framework to extend beyond its built-in capabilities without requiring every external integration to be hard-coded directly into the core agent.

---

# 🔐 Permission System

Agentic execution requires controlled access to the user's system.

`depression.ai` therefore provides permission handling for tool operations.

Example:

```text
filesystem.execute [medium]

Allow? [y/N]
```

This allows the user to review potentially sensitive or destructive operations before they are executed.

---

# 💾 Sessions & Recovery

The framework includes persistent sessions and project state handling.

Features include:

* Persistent sessions
* Session resume
* SQLite-backed session storage
* Snapshots
* Git-stash-based undo
* Diff inspection
* Todo tracking
* Tool execution history

Useful commands include:

```text
/session
/snapshot
/undo
/compact
```

---

# ⌨️ TUI Commands

```text
/help
/plan
/build
/auto
/snapshot [message]
/undo
/model
/provider
/session
/compact
/tools
/permissions
/plugins
/metrics
/cost
/tokens
/config
/set
/exit
```

### Mode Switching

```text
Ctrl + P
```

Switch between:

```text
PLAN ↔ BUILD
```

---

# 🖥️ CLI

```bash
depression
```

Project:

```bash
depression -p /path/to/project
```

Select model:

```bash
depression -m MODEL
```

Select provider:

```bash
depression --provider PROVIDER
```

Start a new session:

```bash
depression --new-session
```

Resume a session:

```bash
depression --session <id>
```

Disable MCP:

```bash
depression --no-mcp
```

Maximum turns:

```bash
depression --max-turns N
```

---

# 📦 Installation

Clone the project:

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
```

Install:

```bash
pip install -e .
```

Optional development installation:

```bash
pip install -e ".[dev]"
```

Start:

```bash
depression
```

---

# ⚡ Quick Start

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
pip install -e .
depression
```

Then configure the LLM through the TUI.

```text
LLM CONNECTION
────────────────────────────
Provider    : custom
Base URL    : ...
API Key     : ...
Model       : ...
────────────────────────────
```

Configure AWS when required through the AWS panel or supported environment configuration.

---

# 🎯 Example Agent Tasks

```text
Analyze this project and explain its architecture.

Find the cause of this build error and fix it.

Review the dependencies and identify unnecessary packages.

Create a deployment plan for this application.

Run the tests and fix the failing tests.

Check the Git changes and prepare a clean commit.

Analyze my AWS environment.

Investigate the deployment failure.

Prepare this application for AWS deployment.
```

The agent determines the required sequence of actions based on the task, available tools and connected model.

---

# 🏭 Future Direction: Sovereign / On-Prem AI

The architecture of `depression.ai` is also being adapted toward **self-hosted AI workbenches** for organizations that cannot send confidential information to public AI services.

Potential deployment environment:

```text
                    ON-PREMISE NETWORK
┌──────────────────────────────────────────────────────┐
│                                                      │
│   User Workstation                                  │
│          │                                           │
│          ▼                                           │
│   ┌──────────────┐                                  │
│   │ depression.ai│                                  │
│   └──────┬───────┘                                  │
│          │                                           │
│          ▼                                           │
│   ┌──────────────┐                                  │
│   │ Local Model  │◄──── GPU Server                  │
│   │   Runtime    │                                  │
│   └──────────────┘                                  │
│          │                                           │
│          ▼                                           │
│   Internal Files / Tools / Infrastructure            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

The objective is to allow organizations to run agentic workflows using **their own infrastructure and locally hosted open-weight models**, subject to the capabilities and security controls of the deployment environment.

---

# 📁 Project Structure

```text
src/
├── agent/
├── cli/
├── config/
├── context/
├── llm/
├── mcp/
├── permissions/
├── plugins/
├── project/
├── session/
├── storage/
├── tools/
├── tui/
└── utils/
```

---

# 🧪 Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check src/
```

Format check:

```bash
ruff format --check src/
```

Type checking:

```bash
mypy src/
```

---

# 📞 Team Depression — Contact Information

## R Naveen Patil

**Lead Developer**

* Email: `rnaveenpatil@gmail.com`
* LinkedIn: https://www.linkedin.com/in/r-naveen-patil-a45403292/
* GitHub: https://github.com/rnaveenpatil
* Portfolio: https://rnaveenpatil-resume.netlify.app

## Rahul Jadav

**Teammate**

* Email: `rjadav214@gmail.com`
* Phone: `8296537642`
* LinkedIn: https://www.linkedin.com/in/subzero91/
* GitHub: https://github.com/Rahul-Jadav-0

## K G Meghashree Naik

**Teammate**

* Email: `meghashree.kg13@gmail.com`
* Phone: `8431898976`
* LinkedIn: https://www.linkedin.com/in/k-g-meghashree/
* GitHub: https://github.com/Meghashree-13

## SUMANTHA MS

**Teammate**

* Email: `mssumanth836@gmail.com`
* Phone: `9481864481`
* LinkedIn: https://www.linkedin.com/in/sumantha-ms-235ba2295/

---

<div align="center">

### `TEAM DEPRESSION`

**Build. Break. Fix. Repeat.**

### `You A-Z. AI depression.ai`

</div>

---

## License

This project is released under the **ISC License**.

---

<div align="center">

**depression.ai — Developed by Team Depression**

</div>
