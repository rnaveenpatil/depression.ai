# depression.ai

<p align="center"><strong>YOUR TERMINAL. YOUR TOOLS. YOUR MODEL.</strong></p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-2ea44f">
  <img alt="Terminal-native" src="https://img.shields.io/badge/Experience-Terminal--native-101010">
</p>

<p align="center"><em>Your A–Z AI, from a single terminal.</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/tui.png" alt="depression.ai TUI" width="900">
</p>

<p align="center"><b>An open-source, local-first AI agent harness for coding, system work, and cloud operations.</b></p>

<p align="center">
  <a href="#-quick-start">Get started</a> ·
  <a href="#-terminal-native-tui">Explore the UI</a> ·
  <a href="#-llm-integration">Connect your model</a> ·
  <a href="#-aws-integration">AWS</a> ·
  <a href="#-security-privacy--data-boundary">Security</a>
</p>

<p align="center">
  <b>▶ <a href="https://youtu.be/en5f4pOvfL0">Watch the walkthrough demo on YouTube</a></b>
</p>

---

## Start here — in simple words

**Describe the job. Choose your model. Approve the tools.**

`depression.ai` runs an AI agent in your terminal. It can inspect project files, plan edits, call tools, learn from tool errors and verify results instead of only returning text.

| You ask | The agent can help with |
| :-- | :-- |
| “Why won't this app start?” | Investigate a failure, edit the relevant code, rerun checks |
| “Build a feature” | Plan, implement, and test the changes |
| “Inspect my AWS environment” | Use available tools within your AWS IAM permissions |
| “Use a local model” | Connect a compatible local endpoint without a cloud LLM API |

> **The model reasons; the harness runs tools under your configured permissions.** Results depend on the connected model and the actual tool responses.

* **It can code and fix bugs** — read files, edit code, run tests.
* **It can work on your system** — run commands, inspect processes, use Git.
* **It can work with AWS** — inspect and manage cloud resources your keys allow.
* **It asks before risky actions** — you stay in control.
* **Credentials stay on your device** — API keys and AWS keys are stored locally and never sent to Team Depression.
* **Local-model option** — a compatible local LLM can work without remote inference; web and cloud tools still need connectivity.
* **Google login is optional** — it only personalizes the agent (for example, your name); that profile data stays on your device.
* **Your chats stay private** — Team Depression does not store your conversation history, and chats are not linked to your Gmail.
* **You choose the AI model** — use a cloud API or a local model.
* **No account is required** — install, run `depression --nosignin`, connect a model and start.
* **Works across platforms** — Python 3.10+ on Linux, macOS and Windows, including VMs and servers.

**Quick start:**

```bash
pip install depression.ai
depression --nosignin
```

Open the **LLM Connection** panel (`/connect`), paste your **Base URL**, **Model ID**, and **API key** if needed, then press **CONNECT**. No model suggestion menu is required.

---

# 👥 Team Depression

> **A team that builds ON ETHICS**

## 🎓 Academic Institution & Project Context

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/mit_logo.png" alt="Moodlakatte Institute of Technology (MIT), Kundapura" width="220">
</p>

<p align="center">
  <b>Moodlakatte Institute of Technology (MIT), Kundapura</b><br>
  Department of Information Science & Engineering (ISE)
</p>

`depression.ai` is being developed as an **academic major project** by **final-year Information Science & Engineering (ISE) students** of Moodlakatte Institute of Technology (MIT), Kundapura.

The project combines AI agents, software engineering, system automation, cloud operations and LLM integration.

**College**

* 🏫 **Moodlakatte Institute of Technology (MIT), Kundapura**
* 🎓 **Department:** Information Science & Engineering (ISE)
* 📚 **Project Type:** Academic Major Project
* 👨‍🎓 **Team:** TEAM DEPRESSION

**College LinkedIn**

* 🔗 https://www.linkedin.com/company/mitkundapura/posts/

---

## 🧑‍💻 R Naveen Patil

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/patil.png" alt="R Naveen Patil" width="180" height="180">
</p>

**Lead Developer • AI Agents • Cloud & Infrastructure**

R Naveen Patil is the main developer of `depression.ai`. He built the core architecture, agent workflow, LLM integration, system tools, cloud integration and overall project.

**Profiles**

* 📧 `rnaveenpatil@gmail.com`
* 🔗 [LinkedIn](https://www.linkedin.com/in/r-naveen-patil-a45403292/)
* 💻 [GitHub](https://github.com/rnaveenpatil)
* 🌐 [Portfolio](https://rnaveenpatil-resume.netlify.app)

---

## 🤝 Team Members

<table>
<tr>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/rahul.jpeg" alt="Rahul Jadav" width="120" height="120">

### Rahul Jadav

📧 `rjadav214@gmail.com`

[GitHub](https://github.com/Rahul-Jadav-0)

[LinkedIn](https://www.linkedin.com/in/subzero91/)

</td>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/meghasree.png" alt="K G Meghashree Naik" width="120" height="120">

### K G Meghashree Naik

📧 `meghashree.kg13@gmail.com`

[GitHub](https://github.com/Meghashree-13)

[LinkedIn](https://www.linkedin.com/in/k-g-meghashree/)

</td>

<td align="center" width="25%">

<img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/sumantha.jpeg" alt="SUMANTHA MS" width="120" height="120">

### SUMANTHA MS

📧 `mssumanth836@gmail.com`

[LinkedIn](https://www.linkedin.com/in/sumantha-ms-235ba2295/)

</td>

</tr>
</table>

---

# 🚀 What Is depression.ai?

**depression.ai** is an AI agent that runs in your terminal. It is built by **Team Depression**.

It is for people who want an AI that can work on a real computer — real projects, real tools and real cloud systems — not just answer questions in a chat window.

Think of it as an **AI workspace for a developer or technical operator**:

```text
                    USER
                      │
                      ▼
              depression.ai
                AI Agent
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Reason       Plan        Context
          │           │           │
          └───────────┼───────────┘
                      ▼
                Tool Selection
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Filesystem     Terminal        Git
        │             │             │
        ├─────────────┼─────────────┤
        ▼             ▼             ▼
      Browser        MCP           AWS
                      │
                      ▼
                 Verification
                      │
                      ▼
                    RESULT
```

It handles multi-step jobs such as:

- Understanding an unfamiliar codebase
- Finding bugs and failures
- Planning changes
- Creating and editing files
- Running commands and tests
- Debugging problems
- Reviewing Git changes
- Using browser automation
- Searching and checking information
- Working with MCP tools
- Inspecting and operating supported AWS environments
- Preparing apps for deployment
- Checking that the work actually succeeded

The **LLM does the thinking**. `depression.ai` gives the model what it needs to act: context, tools, permissions, sessions, execution and recovery.

> **Important:** results depend on both the framework and the model. A weak model, or one that does not call tools reliably, can still produce poor results even when the framework provides the right tools.

---

# 🧠 It Acts, Not Just Chats

A normal chatbot mostly returns text.

`depression.ai` follows an action loop:

```text
Understand
   ↓
Inspect
   ↓
Plan
   ↓
Choose Tool
   ↓
Execute
   ↓
Observe Result
   ↓
Re-evaluate
   ↓
Continue / Correct
   ↓
Verify
   ↓
Report
```

This makes it useful when the answer is not just a paragraph of text, but a **change that has to be made and checked on a real system**.

For example:

```text
"Find why my application is failing and fix it."

            ↓

Inspect project
            ↓
Read relevant files
            ↓
Understand configuration
            ↓
Run diagnostic commands
            ↓
Identify failure
            ↓
Modify required files
            ↓
Run tests / verification
            ↓
Inspect the result
            ↓
Report what changed
```

The exact behaviour depends on the model, available tools, permissions and environment.

---

# 🧠 About depression.ai

**depression.ai** is an AI agent framework made and released by **Team Depression**.

It is a terminal workspace that helps with the whole workflow:

```text
Development
     ↓
Analysis
     ↓
System Operations
     ↓
Testing & Debugging
     ↓
Git & Project Management
     ↓
AWS Operations
     ↓
Deployment
```

The model does the reasoning. `depression.ai` provides the execution environment, tools, permissions, project context and session management.

It is not just a set of fixed commands. The agent can inspect the environment, understand your request, decide what to do, run the available tools and check the result.

---

# 🖥️ Terminal-Native TUI

The main screen is a terminal UI (built with **Textual**) where you work with the agent interactively.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/User%20Interface%20%281%29.png" alt="depression.ai start screen" width="900">
</p>

<p align="center"><em>Start screen — type a prompt to begin, or run <code>depression --nosignin</code> to enter without Google sign-in.</em></p>

Everything happens in one workspace:

* Interactive conversation
* Plan / Build / Auto mode switching
* LLM connection configuration
* AWS configuration
* Tool-call cards
* Permission prompts
* Todo tracking
* Thinking indicators
* Diff inspection
* Session management
* Project context
* Agent execution feedback

The goal is one consistent workflow: **understand → act → observe → verify**.

### Paste-first connection panels

| Panel | What you enter |
| :-- | :-- |
| **LLM** (`/connect`) | Base URL, Model ID, API key (if required) |
| **MCP** (`/mcp`) | HTTP/SSE URL and token, local command, or JSON |
| **AWS** (`/aws`) | Authorized AWS credentials and region |

The sidebar keeps connection controls editable while additional server details remain collapsed.

---

# 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                ┌─────────────────────┐
                │   depression.ai     │
                │       TUI / CLI     │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌───────────────┐     ┌───────────────┐
        │   PLAN AGENT  │     │  BUILD AGENT  │
        │               │     │               │
        │ Read / Analyze│     │ Execute Tools │
        │ Plan Tasks    │     │ Modify System │
        └───────┬───────┘     └───────┬───────┘
                │                     │
                └──────────┬──────────┘
                           ▼
                  ┌─────────────────┐
                  │   TOOL SYSTEM   │
                  ├─────────────────┤
                  │ Terminal        │
                  │ Filesystem      │
                  │ Git             │
                  │ Patch           │
                  │ Search          │
                  │ Web             │
                  │ Browser         │
                  │ AWS             │
                  │ MCP             │
                  │ Todo / Process  │
                  └─────────────────┘
```

---

# 🧩 Plan + Build Architecture

`depression.ai` splits thinking and doing into two modes.

## PLAN

The Plan agent studies the task before changing anything.

```text
User Request
     ↓
Project Inspection
     ↓
Context Gathering
     ↓
Problem Analysis
     ↓
Task Breakdown
     ↓
Execution Plan
```

During planning, the agent can read project files, understand dependencies, inspect the environment and decide the steps needed.

## BUILD

The Build agent carries out the plan with the available tools.

```text
Approved / Selected Plan
          ↓
     Tool Selection
          ↓
       Execution
          ↓
      Verification
          ↓
        Result
```

This keeps **reasoning and execution** separate.

---

# 💻 Development Capabilities

`depression.ai` can work directly on software projects using its tools.

### Project Understanding

* Analyze existing projects
* Inspect project structure
* Read source code
* Understand dependencies
* Search through files
* Identify configuration problems
* Analyze build errors

### Code Operations

* Create files
* Modify existing files
* Apply patches
* Refactor code
* Debug applications
* Run tests
* Verify changes
* Work with Git

### Example

```text
User:
"Analyze this project and find why the application is failing."

Agent:
    ↓
Inspect project
    ↓
Read configuration
    ↓
Inspect source code
    ↓
Run relevant commands
    ↓
Identify failure
    ↓
Create fix
    ↓
Run verification
    ↓
Report result
```

The agent decides what to do based on the task, the tools available and the model you connect.

---

# 🖥️ System Operations

The agent can work with your local development environment using the tools you enable.

Example tasks:

```text
"Check my project structure."

"Find why this application is failing."

"Install the required dependencies."

"Run the tests and fix the errors."

"Find unused files."

"Check the running processes."

"Prepare this project for deployment."

"Inspect the Git changes."

"Create a clean patch for this issue."
```

Risky system actions go through the permission system.

---

# ☁️ AWS Integration

AWS support lets the agent work with cloud infrastructure from the same terminal.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/AWS%20Authentication%20And%20Account%20Access.png" alt="depression.ai AWS credentials panel" width="900">
</p>

<p align="center"><em>AWS CREDENTIALS panel — configure the Access Key ID, Secret Access Key and Region from the TUI.</em></p>

The agent connects your request to AWS through its tools:

```text
User Request
     ↓
depression.ai
     ↓
Agent Reasoning
     ↓
AWS Tools / AWS CLI
     ↓
AWS Environment
```

What it can do depends on your tools, AWS permissions and credentials. It can help with:

* AWS environment inspection
* Infrastructure analysis
* Resource management
* AWS CLI workflows
* Deployment workflows
* Cloud operations
* Monitoring and troubleshooting workflows
* Creating supported resources
* Modifying supported resources
* Removing supported resources
* Application deployment workflows

What is actually allowed depends on your AWS credentials, IAM permissions and configured tools.

---

# 🔐 AWS Security & Permissions

`depression.ai` does **not bypass AWS security controls**.

It uses the AWS identity and permissions you provide.

For example:

```text
AWS IAM Permissions
        ↓
depression.ai
        ↓
Allowed AWS Operations
```

If your AWS identity is not allowed to do something, the agent cannot do it.

So AWS keeps controlling access through normal authentication and IAM rules.

---

# 🔌 Connecting an AWS Account

You can set AWS credentials in the TUI or with supported environment variables.

Example:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-south-1"
```

Then start:

```bash
depression
```

You can also set up AWS from the TUI when your version supports it.

### AWS Access Key Setup

> **Important:** AWS recommends using temporary credentials and IAM roles instead of long-term access keys whenever possible. Use an IAM user access key only when your use case actually requires programmatic long-term credentials. For `depression.ai` local development, an IAM user with only the permissions required by your workflows can be used.

An AWS access key contains two values:

```text
Access Key ID
Secret Access Key
```

You need both values. AWS shows the **secret access key only once**, when the key is created, so save it then.

### Step 1 — Sign in to AWS

Open the AWS Management Console and sign in with an identity that has permission to manage IAM users and credentials.

### Step 2 — Open IAM

From the AWS Management Console:

```text
AWS Console
    ↓
IAM
```

Then open **Users**.

### Step 3 — Create an IAM User

If you do not already have a suitable IAM user:

```text
IAM
  ↓
Users
  ↓
Create user
```

Enter a username, for example:

```text
depression-agent
```

Give the user only the permissions required for the AWS operations that `depression.ai` needs.

AWS recommends managing permissions through groups and policies and following the principle of least privilege. Do **not** give administrator-level permissions just because the agent is easier to configure that way.

For example, if the agent only needs to inspect S3, grant the minimum S3 permissions required instead of attaching `AdministratorAccess`.

### Step 4 — Open the User's Security Credentials

After creating or selecting the IAM user:

```text
IAM
  ↓
Users
  ↓
depression-agent
  ↓
Security credentials
```

Find the **Access keys** section.

### Step 5 — Create the Access Key

Choose:

```text
Create access key
```

AWS may first show an **Access key best practices & alternatives** page. Review the alternatives. If your workflow genuinely requires an access key, continue with the appropriate programmatic-access use case.

Add an optional description such as:

```text
depression.ai local development
```

Then choose:

```text
Create access key
```

### Step 6 — Save the Credentials Immediately

AWS will display:

```text
Access Key ID
Secret Access Key
```

Save both securely.

> **The secret access key cannot be retrieved again after the creation screen.** If you lose it, create a replacement access key rather than expecting AWS to show the old secret again.

Do **not** paste the secret key into:

* GitHub
* README files
* source code
* public issue trackers
* screenshots
* Discord/Slack/WhatsApp messages
* public `.env` files
* public cloud logs

### Step 7 — Configure `depression.ai`

For local development, the credentials can be supplied through the supported TUI configuration or environment variables.

Example:

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="ap-south-1"
```

Then start the agent:

```bash
depression
```

The values above are examples only. Replace them with your actual credentials.

### Step 8 — Verify the AWS Credentials

If AWS CLI is installed, verify the identity before using the agent:

```bash
aws sts get-caller-identity
```

A successful response should identify the AWS account and IAM identity associated with the credentials.

You can then test the AWS integration used by your workflow.

### AWS Credential Flow

```text
AWS Console
     ↓
IAM
     ↓
IAM User
     ↓
Attach Required Permissions
     ↓
Security Credentials
     ↓
Create Access Key
     ↓
Access Key ID + Secret Access Key
     ↓
depression.ai
     ↓
AWS SDK / AWS CLI
     ↓
AWS Resources
```

### Least-Privilege Example

Do not start with:

```text
AdministratorAccess
```

unless there is a genuine administrative requirement and you understand the security consequences.

Prefer a policy that grants only the operations required by the agent.

For example:

```text
depression.ai
     ↓
IAM User
     ↓
Required AWS Policy
     ↓
Only required services/actions
```

If the agent only needs to inspect resources, use read-only permissions where practical.

If the agent needs to create, modify or delete resources, grant only the specific write/delete actions required for those workflows.

### Rotating an Access Key

If you need to replace a key:

```text
Create new access key
        ↓
Update depression.ai / environment
        ↓
Verify the new key
        ↓
Deactivate old key
        ↓
Confirm old key is unused
        ↓
Delete old key
```

Do not immediately delete an old key that may still be used by an application. AWS supports having up to two access keys for an IAM user, which allows a controlled rotation process.

### Security Rules

> **Never commit AWS credentials to Git.**

Add sensitive files to `.gitignore`:

```gitignore
.env
.env.*
*.pem
*.key
.aws/
```

If an AWS secret is accidentally exposed:

```text
Stop using the exposed credential
        ↓
Deactivate the access key
        ↓
Create a replacement key if required
        ↓
Check CloudTrail / access history
        ↓
Review the IAM permissions
        ↓
Remove the secret from the repository/history
```

A leaked AWS secret should be treated as compromised. Do not simply delete the visible line from the latest commit and assume the credential is safe.

> **Never commit AWS access keys, secret keys, `.env` files or other credentials to GitHub.**

For production deployments, prefer IAM roles, temporary credentials, workload identity or another AWS-supported short-lived credential mechanism instead of embedding long-lived access keys in applications or servers.

For the latest AWS guidance, see the official AWS IAM documentation on access keys and IAM users.

---

# 🤖 LLM Integration

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/LLM%20Connection%20Configuration.png" alt="depression.ai LLM connection configuration panel" width="900">
</p>

<p align="center"><em>LLM CONNECTION panel — configure Base URL, API Key and Model ID from inside the TUI.</em></p>

The agent and the model are separate, so you can change the model without changing the agent.

The model does the thinking while `depression.ai` provides:

```text
LLM
 ↓
Agent
 ↓
Context
 ↓
Tools
 ↓
Permissions
 ↓
Execution
```

You can connect a model with:

```text
Provider
Base URL
API Key
Model
```

Example:

```text
Provider : custom
Base URL : https://api.example.com/v1
API Key  : ****************
Model    : your-model
```

This lets `depression.ai` work with any compatible remote model API.

### Usage & cost tracking

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/LLM%20Usage%20and%20Cost%20Tracking.png" alt="depression.ai LLM usage and cost tracking panel" width="900">
</p>

The TUI tracks token usage and API cost per query and per session. Open the live panel with `/llmcost` (or `/cost`) to see input/output tokens, total cost, API calls, tool calls and query duration.

---

# 🧠 OpenAI-Compatible & API-Based Models

You are not locked to one model provider. `depression.ai` can use any compatible API endpoint.

Example:

```text
depression.ai
      │
      ▼
┌──────────────────┐
│ Compatible API   │
│ Endpoint         │
└────────┬─────────┘
         │
         ▼
       LLM
```

A compatible endpoint speaks the API format expected by the provider adapter.

However:

> **A compatible API does not guarantee the model will be a good agent.**

Agent work needs more than text generation.

---

# 🧠 Model Requirements

A model used with `depression.ai` should ideally have:

* Chat / text generation
* Sufficient context length
* Strong instruction following
* Code understanding
* Multi-turn conversation support
* Reliable reasoning
* Structured output capability
* Tool / function calling where supported
* Consistent responses

Model quality matters because the model must understand the task and decide which tools to use.

---

# 🏠 Local LLM Support

A major goal of `depression.ai` is running models **locally or on your own servers**.

You can run a local model server and connect it to the same agent.

For example:

```text
Local GPU
    ↓
Model Server
    ↓
Compatible API
    ↓
depression.ai
    ↓
Agent
    ↓
Tools
```

A local endpoint can look like this:

```text
Provider : custom
Base URL : http://localhost:11434/v1
API Key  : local
Model    : your-local-model
```

The exact URL, API format and model name depend on the local model server you use.

---

# ⚡ GPU-Based Local AI

With local deployment, the model runs on hardware you control.

```text
┌─────────────────────────────────────────┐
│             LOCAL GPU SERVER             │
│                                         │
│        ┌──────────────────────┐         │
│        │     GPU / VRAM       │         │
│        └──────────┬───────────┘         │
│                   ↓                     │
│        ┌──────────────────────┐         │
│        │    Local Model       │         │
│        │      Server          │         │
│        └──────────┬───────────┘         │
│                   ↓                     │
│             Local LLM API              │
└───────────────────┬─────────────────────┘
                    ↓
             depression.ai
                    ↓
          Agents + Tools + Data
```

GPU memory, RAM and compute needs depend on model size, quantization, context length, number of users and workload.

---

# 🔒 Local & Self-Hosted AI

`depression.ai` already keeps the **agent framework and tools on your system**; the LLM can come from an API.

The next step is to let the model itself run inside your own network.

```text
┌────────────────────────────────────────────────┐
│              ORGANIZATION NETWORK              │
│                                                │
│  ┌───────────────┐                             │
│  │ User / TUI    │                             │
│  └───────┬───────┘                             │
│          ↓                                     │
│  ┌────────────────┐                            │
│  │ depression.ai  │                            │
│  │ Agent          │                            │
│  └───────┬────────┘                            │
│          ↓                                     │
│  ┌────────────────┐                            │
│  │ Local LLM      │                            │
│  │ GPU Server     │                            │
│  └───────┬────────┘                            │
│          ↓                                     │
│  ┌────────────────────────────────────────┐    │
│  │ Internal Files / Tools / Infrastructure│    │
│  └────────────────────────────────────────┘    │
│                                                │
└────────────────────────────────────────────────┘
```

This is for environments where sensitive documents, source code, internal messages, financial data, designs or other confidential information must stay inside controlled infrastructure.

---

# 🧠 Multiple Local Models

A self-hosted setup can run several open-weight models for different jobs.

```text
                 depression.ai
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
     Code Model    General Model   Reasoning
          │            │            │
          └────────────┼────────────┘
                       ↓
                  GPU Server
```

Different tasks need different model strengths.

For example:

| Task               | Potential Model Requirement       |
| ------------------ | --------------------------------- |
| Code generation    | Strong code understanding         |
| Code review        | Long context + reasoning          |
| Document analysis  | Large context                     |
| Planning           | Strong instruction following      |
| Tool execution     | Reliable tool calling             |
| General assistance | General-purpose instruction model |

The model-selection layer can evolve to choose a model based on the task.

---

# 🛠️ Tool System

The agent uses tools to work on your environment.

Current tool categories:

```text
┌─────────────────────────┐
│       TOOL SYSTEM       │
├─────────────────────────┤
│ Terminal                │
│ Filesystem              │
│ Git                     │
│ Patch                   │
│ Search                  │
│ Web                     │
│ Todo                    │
│ Process                 │
│ Browser Automation      │
│ AWS                     │
│ MCP                     │
└─────────────────────────┘
```

The model chooses which tools fit the task.

So one agent can handle many workflows without a separate app for each one.

---

# 🔗 MCP Support

`depression.ai` supports the **Model Context Protocol (MCP)**.

MCP lets you connect extra tool servers to the agent.

```text
depression.ai
      │
      ├── Built-in Tools
      │
      └── MCP
           ├── Server 1
           ├── Server 2
           ├── Server 3
           └── ...
```

You can add new capabilities without hard-coding every integration into the core.

### Connect an MCP server

1. Enter `/mcp` and select **HTTP**, **SSE**, **LOCAL**, or **JSON**.
2. Paste the matching server details (URL and optional token, executable and arguments, or a compatible JSON snippet).
3. Choose **SAVE & CONNECT**. Expand **CONNECTION DETAILS** to troubleshoot.

Only connect servers you trust. MCP tools can perform sensitive actions; permission checks remain in effect.

---

# 🔐 Permission System

An AI agent can change your files, system or infrastructure.

So `depression.ai` asks for permission for tool actions.

Example:

```text
filesystem.execute [medium]

Allow? [y/N]
```

You can review sensitive or destructive actions before they run.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/Tool%20Permission%20and%20%20Execution%20Confirmation.png" alt="depression.ai tool permission and execution confirmation modal" width="900">
</p>

<p align="center"><em>Permission request modal — high-risk tool actions must be explicitly allowed or denied.</em></p>

How it works:

```text
Agent wants to execute action
          ↓
Permission Check
          ↓
┌─────────┴─────────┐
│                   │
▼                   ▼
Allowed           Ask User
                    │
                 y / N
                    │
                    ▼
                Execute
```

---

# 💾 Sessions & Recovery

`depression.ai` saves sessions and project state.

Features:

* Persistent sessions
* Session resume
* SQLite-backed session storage
* Snapshots
* Git-stash-based undo
* Diff inspection
* Todo tracking
* Tool execution history

Useful commands:

```text
/session
/snapshot
/undo
/compact
```

You can continue work across multiple interactions with the relevant state kept.

---

# 📋 Todo & Task Tracking

Complex tasks involve many steps.

The TUI tracks them so you can follow progress.

```text
TODO
────────────────────────
✓ Inspect project
✓ Identify dependency issue
→ Modify configuration
○ Run tests
○ Verify deployment
────────────────────────
```

This breaks the overall job into smaller steps.

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

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/Command%20Reference%20Interface.png" alt="depression.ai command reference and help panel" width="900">
</p>

### Mode Switching

```text
Tab
```

Switch between:

```text
PLAN ↔ BUILD ↔ AUTO
```

* **PLAN** — read-only analysis and task planning.
* **BUILD** — executes tools and applies changes.
* **AUTO** — runs Plan and Build in a loop until the task succeeds or the iteration limit is reached (`--max-iterations`).

You can also switch directly with `/plan`, `/build` and `/auto`.

---

# 🖥️ CLI

Start the application:

```bash
depression
```

Skip the Google sign-in / onboarding screen and enter as a local guest:

```bash
depression --nosignin
```

This is useful for offline use, CI/demo environments, or when you simply want to reach the prompt immediately. A local guest profile is stored on disk and the Google sign-in screen is bypassed; you can still sign in later from the profile panel.

Specify a project:

```bash
depression -p /path/to/project
```

Select a model:

```bash
depression -m MODEL
```

Select a provider:

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

Set maximum turns:

```bash
depression --max-turns N
```

---

# 🚪 Start Without Sign-In (`--nosignin`)

Google sign-in is **optional**. Use `--nosignin` to bypass the welcome/onboarding screen and go straight into the TUI:

```bash
depression --nosignin
```

On first run this creates a local guest profile (`local-guest`) on your machine. No Google account, email address or OAuth flow is required.

Combine it with other options:

```bash
depression --nosignin -p /path/to/project
depression --nosignin --provider groq --model <model>
```

What you should know:

* The guest profile is stored locally in the agent data directory.
* The agent, tools, sessions and permissions work the same as when signed in.
* Gmail-specific features (mail/calendar) stay disabled until you sign in.
* You can sign in with Google later from the `/profile` panel.
* Signing in is **not** required to use `depression.ai`.

---

# 📦 Installation

Install with a single command — no clone or build step required:

```bash
pip install depression.ai
```

The package is lightweight and runs on any Linux, macOS or Windows machine, whether virtual or physical.

To inspect or modify the architecture, clone the repository:

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
```

For development:

```bash
pip install -e ".[dev]"
```

Start:

```bash
depression
```

---

# 🧩 Compatibility

`depression.ai` is lightweight and runs on **any machine with a terminal** that can run **Python 3.10+** and **pip**:

* Linux, macOS or Windows (Windows Terminal / WSL)
* Physical machines, virtual machines, containers or remote servers
* Local laptops or cloud instances — no GPU or dedicated infrastructure required

### Plug-and-play setup

1. Install with a single command:

   ```bash
   pip install depression.ai
   ```

2. Start the agent:

   ```bash
   depression
   # or skip sign-in:
   depression --nosignin
   ```

3. Add your LLM API credentials in the LLM Connection panel.
4. Start working with the agent.

You do not need to share data to get started. Your credentials and sessions stay on your machine, and sign-in is optional. Install, run the command, connect your API credentials, and start working on any machine.

---

# ⚡ Quick Start

**Requirements:** a terminal, **Python 3.10+** and **pip** — any physical or virtual machine works, with no GPU required.

```bash
git clone https://github.com/rnaveenpatil/depression.ai.git
cd depression.ai
pip install -e .
depression
```

To skip the Google sign-in / onboarding screen and go straight to the TUI:

```bash
depression --nosignin
```

After starting the TUI, configure the LLM:

```text
LLM CONNECTION
────────────────────────────
Provider    : custom
Base URL    : ...
API Key     : ...
Model       : ...
────────────────────────────
```

For AWS work, set your AWS credentials and region in the same config.

---

# 🔧 Environment Configuration

You can use environment variables where supported:

```bash
export DEPRESSION_PROVIDER=custom
export DEPRESSION_BASE_URL=https://api.example.com/v1
export DEPRESSION_API_KEY="your-api-key"
export DEPRESSION_MODEL="your-model"

export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-south-1"
```

Local model example:

```bash
export DEPRESSION_PROVIDER=custom
export DEPRESSION_BASE_URL=http://localhost:11434/v1
export DEPRESSION_API_KEY=local
export DEPRESSION_MODEL="your-local-model"
```

---

# ⚙️ Configuration

Config can be saved in:

```text
.agent/config.json
```

or:

```text
~/.agent/config.json
```

Create a config file:

```bash
depression --init-config
```

---

# 🎯 Example Agent Tasks

### Development

```text
Analyze this project and explain its architecture.

Find the cause of this build error and fix it.

Review the dependencies and identify unnecessary packages.

Run the tests and fix the failing tests.

Refactor this module without changing its public API.
```

### System

```text
Check my project structure.

Find the process using this port.

Inspect the current environment.

Check why this command is failing.

Prepare this project for deployment.
```

### Git

```text
Review my Git changes.

Explain the current diff.

Create a clean patch for this change.

Prepare the project for a commit.
```

### AWS

```text
Analyze my AWS environment.

Investigate the deployment failure.

Check the configured AWS environment.

Prepare this application for AWS deployment.

Analyze the infrastructure configuration.
```

What the agent can do depends on the tools, model, credentials and permissions.

---

# 🔄 Agent Execution Flow

A typical task looks like this:

```text
                    USER REQUEST
                         │
                         ▼
                 ┌──────────────┐
                 │ Context Load │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Plan Agent   │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Task Planning│
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Build Agent  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Tool Calling │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Verification │
                 └──────┬───────┘
                        ↓
                       RESULT
```

---

# 🧱 Why the Framework Is Different

`depression.ai` is a **general-purpose AI agent framework**, not just a coding assistant.

### What makes it different

* **Credentials stay on your device** — the agent uses API keys and AWS keys locally, only for the services you configure. Team Depression never receives them.
* **No data is sent to Team Depression** — your conversations, project files, terminal output and usage data are not sent to us.
* **Works offline** — with a local model, the agent runs without an internet connection.
* **Conversations stay private** — we do not store your chat history, and your chats are not linked to your Google account.
* **Google login is optional and only for personalization** — it provides your name/profile locally; your Gmail content is never read.
* **Sensitive actions need permission** — you approve risky tool actions before they run.
* **Open source and modifiable** — inspect the code, fork it and adapt the agent, tools, models or workflows to your needs.
* **Lightweight and optimized** — a small Python package that runs on any machine, virtual or physical, without heavy infrastructure.
* **Single-command install** — `pip install depression.ai`.

One agent can use:

```text
LLM
 │
 ├── Development
 ├── Filesystem
 ├── Terminal
 ├── Git
 ├── System Operations
 ├── Browser Automation
 ├── MCP
 └── AWS
```

So one workspace can go from:

```text
Idea
 ↓
Analysis
 ↓
Development
 ↓
Testing
 ↓
System Handling
 ↓
Cloud Operations
 ↓
Deployment
```

---

# 🔒 Security, Privacy & Data Boundary

`depression.ai` is built to keep your work **local and secure first**.

The agent and its tool actions run on your machine:

```text
USER MACHINE
────────────────────────────────────
depression.ai
Agent
Project Context
Filesystem
Terminal
Git
Sessions
Permissions
Configured Tools
AWS Integration
────────────────────────────────────
```

### What stays local

**Team Depression does not receive or remotely inspect your project files, terminal output, source code or local system data just because you installed the package or signed in.**

Local parts include:

- Agent execution
- Tool execution
- Filesystem interaction
- Terminal commands
- Git operations
- Session and conversation state
- Credentials, API keys and profile data
- Permission handling
- Local project context

### Credentials and data boundary

- **Credentials are stored on your device.** LLM API keys, AWS access keys, tokens and session data are kept locally in the agent's data/config directory. The agent uses them only on your machine to connect to the provider or service you configure. They are **never sent to Team Depression**, and no one at Team Depression can access them.
- **Nothing is sent to Team Depression.** Your credentials, conversations, project files, terminal output and usage data are not sent to Team Depression.
- **Works offline.** With a local model, the agent can run without an internet connection. The project is open source, so you can inspect exactly what it does.
- **Project data stays local.** Project files, terminal output, source code and tool results stay on your machine. Only the prompt/context you send to a configured remote LLM leaves the machine.
- **Conversations stay private.** Team Depression does not store your conversation history. Locally saved sessions remain on your device, and your conversations are not linked to your Gmail account.
- **Sign-in is optional.** `depression --nosignin` runs the full agent with a local guest profile and requires no Google account.
- **Nothing is sold, shared or disclosed.** Your credentials, project data, conversations and login details are not published, sold or shared with any third party. Data sent to a remote LLM provider is governed by that provider's policy (see below).
- **Google/Gmail login is only for personalization.** If you choose to sign in, Google provides basic profile information (such as your name and email) so the agent can personalize your experience. That profile data is stored locally on your device, is not disclosed by Team Depression to anyone, and your Gmail content is never read. If you use `--nosignin`, no Google login data is created at all.

### Remote LLM providers

If you use a **remote LLM API**, the information needed to answer your request may leave your machine and go to that provider.

Depending on the task, this can include:

- Your prompt
- Relevant conversation/context
- Selected project content
- Tool results
- Code or configuration required for the model to reason about the task

So:

> **A remote LLM does not make the whole framework remote. The agent stays local, but model requests and the data in them follow the provider's policies.**

Read your provider's privacy and data-retention policy before sending sensitive information to a remote model.

### Local LLM mode

You can also connect a local, OpenAI-compatible model server in the **LLM Connection** panel.

For example:

```text
User Machine
     ↓
depression.ai
     ↓
Local LLM Server
     ↓
Local GPU / CPU
```

With a local model, inference stays on your machine or network instead of going to a public API.

### Gmail / Google login

Google/Gmail login is **not the agent backend**.

It is optional. If you sign in, Google shares basic profile information (such as your name and email) so the agent can personalize your experience. This profile data is stored locally on your device and is not sent to Team Depression.

Signing in does **not** give Team Depression remote access to your terminal, project directory or local files.

<p align="center">
  <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/User%20Profile.png" alt="depression.ai signed-in user profile panel" width="900">
</p>

<p align="center"><em>Signed-in profile — Google identity, verification status and login metadata.</em></p>

Sign-in is optional. Start with `depression --nosignin` to enter the TUI immediately with a local guest profile (`local-guest`); you can sign in with Google later from the profile panel. Gmail-specific features remain disabled for guest profiles.

Team Depression does not disclose your Google/Gmail login data or personalization details to anyone. They are stored locally, and Team Depression does not publish, sell or share them with third parties. Sign-in itself is handled by Google/Firebase. Your Gmail content is never accessed, your conversation history is not stored by Team Depression, and your conversations are not linked to your Gmail account. If you use `--nosignin`, no Google login data is created at all.

Login and agent execution are separate:

```text
Google Login
     │
     └── Product authentication / usage identification

depression.ai Agent
     │
     └── Local agent + tool execution
```

Signing in with Google does not make your project data available to Team Depression.

### Conversation storage

- **Chats are not stored by Team Depression.** Team Depression does not keep your conversation history.
- **Local sessions stay on your device.** If you save or resume a session, that data remains on your machine.
- **Chats are not linked to Gmail.** Your conversations are not connected to your Google/Gmail account.
- **No conversation data is disclosed.** Your chats are not published, sold or shared with anyone.

### Open source & auditability

The project is open source. Anyone can inspect and audit the code.

Be realistic about security: **no software can guarantee perfect safety**. Review the source, permissions, connected tools, AWS policies and LLM provider settings before using it with sensitive systems.

### Security principles

`depression.ai` focuses on:

- Local-first agent execution
- Offline operation with local models
- Credentials and profile data stored on your device
- No conversation data sent to or stored by Team Depression
- Explicit permission handling for sensitive operations
- Sensitive-file protection
- Secret redaction / scanning mechanisms where implemented
- Least-privilege AWS access
- User-controlled LLM configuration
- Separation between authentication and agent execution
- Open-source code for inspection

**Never commit API keys, AWS secrets, passwords, tokens, private keys or `.env` files to Git.**

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

The code is organized around the agent, LLM integration, context handling, tools, permissions, sessions, plugins, MCP and TUI.

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

# 📌 Project Status

`depression.ai` is an actively developed AI agent framework by **Team Depression**.

The current version provides:

```text
✓ Terminal-native agent
✓ Plan + Build architecture
✓ Tool execution
✓ Filesystem operations
✓ Terminal operations
✓ Git workflows
✓ MCP support
✓ Permission controls
✓ Persistent sessions
✓ Snapshot / undo workflows
✓ TUI
✓ LLM setup while running
✓ AWS-oriented workflows
```

Future plans include:

```text
→ Stronger local LLM support
→ GPU-based inference
→ Self-hosted deployment
→ Multiple local model support
→ Task-based model selection
→ Controlled on-premise deployments
→ Enterprise / industrial AI workflows
```

---

# 🔗 Repository

**GitHub:**
https://github.com/rnaveenpatil/depression.ai

---

<div align="center">

### `TEAM DEPRESSION`

**Build. Break. Fix. Repeat.**

### `Your A-Z AI depression.ai`

**depression.ai — Developed by Team Depression**

</div>

---

# 🎬 Demo

<p align="center">
  <a href="https://youtu.be/en5f4pOvfL0">
    <img src="https://raw.githubusercontent.com/rnaveenpatil/depression.ai/main/assets/youtube-demo-thumbnail.png" alt="▶ Play the depression.ai demo on YouTube" width="900">
  </a>
</p>

<p align="center">
  <b><a href="https://youtu.be/en5f4pOvfL0">▶ Watch the complete walkthrough on YouTube</a></b>
</p>

See the complete setup and agent workflow in the official `depression.ai` demo — install with one command, connect your API credentials and start working with the agent on any machine.

---

# 📜 License

This project is released under the **MIT License** (see [LICENSE](LICENSE)). Developed as an academic project at **Moodlakatte Institute of Technology, Kundapura**.
