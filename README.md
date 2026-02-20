# OpenLegion 🪖

> **Autonomous AI agent fleets — isolated, auditable, and production-ready.**
> Every agent runs in its own Docker container. API keys never leave the vault.
> Chat via Telegram, Discord, Slack, or WhatsApp. Built-in cost controls. 100+ LLM providers.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-638%20passing-brightgreen.svg)]()
[![LiteLLM](https://img.shields.io/badge/LLM-100%2B%20providers-orange.svg)](https://litellm.ai)
[![Docker](https://img.shields.io/badge/isolation-Docker%20%2B%20microVM-blue.svg)]()

**The AI agent framework built for builders who can't afford a security incident.**

[Quick Start](#quick-start) · [Full Setup Guide](QUICKSTART.md) · [Why Not OpenClaw?](#why-not-openclaw) · [Docs](docs/)

---

> 📹 **[Watch the 90-second demo →](#demo)**
> *Setup to autonomous agent fleet running in Telegram in under 3 minutes.*

---

## Demo

[INSERT 90-SECOND SCREEN RECORDING HERE]

> `openlegion setup` → `openlegion start` → agents running in Telegram with
> live cost tracking. No configuration files edited by hand.

## Table of Contents

- [Quick Start](#quick-start)
- [Why Not OpenClaw?](#why-not-openclaw)
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Mesh Host](#mesh-host)
- [Agent Architecture](#agent-architecture)
- [Memory System](#memory-system)
- [Triggering & Automation](#triggering--automation)
- [Cost Tracking & Budgets](#cost-tracking--budgets)
- [Security Model](#security-model)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [MCP Tool Support](#mcp-tool-support)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)

---

## Quick Start

**Requirements:** Python 3.10+, Docker (running), an LLM API key ([Anthropic](https://console.anthropic.com/) / [Moonshot](https://platform.moonshot.cn/) / [OpenAI](https://platform.openai.com/api-keys))

**macOS / Linux:**

```bash
git clone https://github.com/openlegion-ai/openlegion.git && cd openlegion
./install.sh                     # checks deps, creates venv, makes CLI global
openlegion setup                 # API key, project description, team template
openlegion start                 # launch agents and start chatting
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
powershell -ExecutionPolicy Bypass -File install.ps1
openlegion setup
openlegion start
```

> First install downloads ~70 packages and takes 2-3 minutes. Subsequent installs are fast.
>
> **Need help?** See the **[full setup guide](QUICKSTART.md)** for platform-specific instructions and troubleshooting.

```bash
# Add more agents later
openlegion agent add researcher

# Check status
openlegion status

# Run in background
openlegion start -d
openlegion chat researcher   # connect from another terminal
openlegion stop              # clean shutdown
```

---

## Why Not OpenClaw?

OpenClaw is the most popular personal AI assistant framework — 200K+ GitHub stars,
brilliant for single-user use. For production workloads and team deployments, it
has documented problems:

- **42,000+ exposed instances** with no authentication (Bitsight, Feb 2026)
- **341 malicious skills** found stealing user data (Koi Security / The Hacker News)
- **CVE-2026-25253**: one-click remote code execution
- No per-agent cost controls — runaway spend is a real risk
- No deterministic routing — a CEO agent (LLM) decides what runs next
- API keys stored directly in agent config

OpenLegion was designed from day one assuming agents will be compromised.

| | OpenClaw | OpenLegion |
|---|---|---|
| **API key storage** | Agent config files | Vault proxy — agents never see keys |
| **Agent isolation** | Process-level | Docker container per agent + microVM option |
| **Cost controls** | None | Per-agent daily + monthly budget caps |
| **Multi-agent routing** | LLM CEO agent | Deterministic YAML DAG workflows |
| **LLM providers** | Broad | 100+ via LiteLLM with health-tracked failover |
| **Test coverage** | Minimal | 638 tests including full Docker E2E |
| **Codebase size** | 430,000+ lines | ~11,000 lines — auditable in a day |

---

## What It Does

OpenLegion is an open-source **autonomous AI agent framework** for running multi-agent
fleets in isolated Docker containers. Each agent gets its own memory, tools, schedule,
and budget — coordinated through deterministic YAML workflows with no LLM routing layer.

Chat with your agent fleet via **Telegram**, **Discord**, or CLI. Agents act autonomously
via cron schedules, webhooks, heartbeat monitoring, and file watchers — without being
prompted.

**638 tests passing** across **~11,000 lines** of application code.
**Fully auditable in a day.**
No LangChain. No Redis. No Kubernetes. No CEO agent. MIT License.

1. **Security by architecture** — every agent runs in an isolated Docker container
   (microVM when available). API keys live in the credential vault — agents call
   through a proxy and never handle credentials directly. Defense-in-depth with
   5 security layers.

2. **Production-grade cost control** — per-agent LLM token tracking with enforced
   daily and monthly budget caps at the vault layer. Agents physically cannot spend
   what you haven't authorized. View live spend with `/costs` in the REPL.

3. **Deterministic multi-agent orchestration** — YAML-defined DAG workflows with
   step dependencies, conditions, retries, and failure handlers. No LLM decides
   what runs next. Predictable, debuggable, auditable.

4. **Acts autonomously** — cron schedules, heartbeat probes, webhook triggers, and file watchers let agents work without being prompted.

5. **Self-improves** — agents learn from tool failures and user corrections, injecting past learnings into future sessions.

6. **Self-extends** — agents write their own Python skills at runtime and hot-reload them. Agents can also spawn sub-agents for specialized work.

7. **Multi-channel** — connect agents to Telegram, Discord, Slack, and WhatsApp. Also accessible via CLI and API.

8. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly budget enforcement.

9. **Runs deterministic workflows** — YAML-defined DAG workflows chain agents in sequence with conditions, retries, and failure handlers.

10. **Fails over across providers** — configurable model failover chains cascade across LLM providers with per-model health tracking and exponential cooldown.

---

## Architecture

OpenLegion's architecture separates concerns across three trust zones:
untrusted external input, sandboxed agent containers, and a trusted mesh host
that holds credentials and coordinates the fleet. All inter-agent communication
flows through the mesh — no agent has direct network access or peer-to-peer
connections.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           User Interface                                │
│                                                                         │
│   CLI (click)          Webhooks            Cron Scheduler               │
│   - setup              - POST /webhook/    - "0 9 * * 1-5"             │
│   - start (REPL)         hook/{id}         - "every 30m"               │
│   - stop / status      - Trigger agents    - Heartbeat pattern          │
│   - agent add/list                                                      │
└──────────────┬──────────────────┬──────────────────┬────────────────────┘
               │                  │                  │
               ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Mesh Host (FastAPI)                              │
│                         Port 8420 (default)                              │
│                                                                         │
│  ┌────────────┐ ┌─────────┐ ┌───────────┐ ┌────────────────────────┐   │
│  │ Blackboard │ │ PubSub  │ │  Message   │ │   Credential Vault     │   │
│  │ (SQLite)   │ │         │ │  Router    │ │   (API Proxy)          │   │
│  │            │ │ Topics, │ │            │ │                        │   │
│  │ Key-value, │ │ subs,   │ │ Permission │ │ LLM, Anthropic,       │   │
│  │ versioned, │ │ notify  │ │ enforced   │ │ OpenAI, Apollo,        │   │
│  │ TTL, GC    │ │         │ │ routing    │ │ Hunter, Brave Search   │   │
│  └────────────┘ └─────────┘ └───────────┘ └────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Orchestrator │ │  Permission  │ │  Container   │ │    Cost      │   │
│  │              │ │  Matrix      │ │  Manager     │ │   Tracker    │   │
│  │ DAG executor,│ │              │ │              │ │              │   │
│  │ step deps,   │ │ Per-agent    │ │ Docker life- │ │ Per-agent    │   │
│  │ conditions,  │ │ ACLs, globs, │ │ cycle, nets, │ │ token/cost,  │   │
│  │ retry/fail   │ │ default deny │ │ volumes      │ │ budgets      │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
               │
               │  Docker Network (bridge / host)
               │
     ┌─────────┼──────────┬──────────────────────┐
     ▼         ▼          ▼                      ▼
┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐
│ Agent A │ │ Agent B │ │ Agent C │  ...  │ Agent N │
│ :8401   │ │ :8402   │ │ :8403   │       │ :840N   │
└─────────┘ └─────────┘ └─────────┘       └─────────┘
  Each agent: isolated Docker container, own /data volume,
  own memory DB, own workspace, 512MB RAM, 0.5 CPU cap
```

### Trust Zones

| Level | Zone | Description |
|-------|------|-------------|
| 0 | Untrusted | External input (webhooks, user prompts). Sanitized before reaching agents. |
| 1 | Sandboxed | Agent containers. Isolated filesystem, no external network, no credentials. |
| 2 | Trusted | Mesh host. Holds credentials, manages containers, routes messages. |

---

## Mesh Host

The mesh host is the central coordination layer. It runs on the host machine
as a single FastAPI process.

### Blackboard (Shared State Store)

SQLite-backed key-value store with versioning, TTL, and garbage collection.

| Namespace | Purpose | Example |
|-----------|---------|---------|
| `tasks/*` | Task assignments | `tasks/research_abc123` |
| `context/*` | Shared agent context | `context/prospect_acme` |
| `signals/*` | Inter-agent signals | `signals/research_complete` |
| `history/*` | Append-only audit log | `history/action_xyz` |

### Credential Vault (API Proxy)

Agents never hold API keys. All external API calls route through the mesh.
The vault loads credentials from `OPENLEGION_CRED_*` environment variables
and supports multiple providers. Budget limits are enforced before dispatching
LLM calls and token usage is recorded after each response.

### Model Failover

Configurable failover chains cascade across LLM providers transparently.
`ModelHealthTracker` applies exponential cooldown per model (transient errors:
60s → 300s → 1500s, billing/auth errors: 1h). Permanent errors (400, 404)
don't cascade. Streaming failover is supported — if a connection fails mid-stream,
the next model in the chain picks up.

### Permission Matrix

Every inter-agent operation is checked against per-agent ACLs:

```json
{
  "researcher": {
    "can_message": ["orchestrator"],
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"],
    "blackboard_read": ["tasks/*", "context/*"],
    "blackboard_write": ["context/prospect_*"],
    "allowed_apis": ["llm", "brave_search"]
  }
}
```

### Orchestrator (Workflow DAG Executor)

Workflows are defined in YAML and executed as directed acyclic graphs:

```yaml
name: prospect_pipeline
trigger: new_prospect
timeout: 600
steps:
  - id: research
    agent: research
    task_type: research_prospect
    input_from: trigger.payload
  - id: qualify
    agent: qualify
    task_type: qualify_lead
    depends_on: [research]
    condition: "research.result.score >= 5"
```

### Container Manager

Each agent runs in an isolated Docker container with:
- **Image**: `openlegion-agent:latest` (Python 3.12, system tools, Playwright, Chromium, Camoufox)
- **Network**: Bridge with port mapping (macOS/Windows) or host network (Linux)
- **Volume**: `openlegion_data_{agent_id}` mounted at `/data`
- **Resources**: 512MB RAM limit, 50% CPU quota
- **Security**: `no-new-privileges`, runs as non-root `agent` user (UID 1000)

---

## Agent Architecture

Each agent container runs a FastAPI server with endpoints for task assignment,
chat, status, capabilities, and results.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Container                           │
│                                                              │
│  FastAPI Server (:8400)                                      │
│    POST /task    POST /chat    POST /chat/reset               │
│    GET /status   GET /result   GET /capabilities              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                     AgentLoop                         │    │
│  │                                                       │    │
│  │  Task Mode: bounded 20-iteration loop                 │    │
│  │  Chat Mode: conversational with tool use              │    │
│  │                                                       │    │
│  │  Both: LLM call → tool execution → context mgmt      │    │
│  └──┬──────────┬──────────┬──────────┬──────────┬───────┘    │
│     │          │          │          │          │             │
│  ┌──▼───┐  ┌──▼───┐  ┌──▼──────┐ ┌─▼──────┐ ┌─▼─────────┐ │
│  │ LLM  │  │ Mesh │  │ Skill   │ │Work-   │ │ Context   │ │
│  │Client│  │Client│  │Registry │ │space   │ │ Manager   │ │
│  │(mesh │  │(HTTP)│  │(builtins│ │Manager │ │(token     │ │
│  │proxy)│  │      │  │+custom) │ │(/data/ │ │tracking,  │ │
│  └──────┘  └──────┘  └─────────┘ │workspace│ │compact)   │ │
│                                   └─────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Task Mode (`execute_task`)

Accepts a `TaskAssignment` from the orchestrator. Runs a bounded loop
(max 20 iterations) of decide → act → learn. Returns a `TaskResult` with
structured output and optional blackboard promotions.

### Chat Mode (`chat`)

Accepts a user message. On the first message, loads workspace context
(AGENTS.md, SOUL.md, USER.md, MEMORY.md) into the system prompt and
searches memory for relevant facts. Executes tool calls in a bounded loop
(max 30 rounds) and runs context compaction when needed.

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `exec` | Shell command execution with timeout |
| `read_file` | Read file contents from `/data` |
| `write_file` | Write/append file in `/data` |
| `list_files` | List/glob files in `/data` |
| `http_request` | HTTP GET/POST/PUT/DELETE/PATCH |
| `browser_navigate` | Open URL, extract page text (basic/stealth/advanced backends) |
| `browser_snapshot` | Accessibility tree snapshot with element refs (e1, e2, ...) |
| `browser_screenshot` | Capture page screenshot |
| `browser_click` | Click element by ref or CSS selector |
| `browser_type` | Fill input by ref or CSS selector |
| `browser_evaluate` | Run JavaScript in page |
| `memory_search` | Hybrid search across workspace files and structured DB |
| `memory_save` | Save fact to workspace and structured memory DB |
| `memory_recall` | Semantic search over structured facts with category filtering |
| `web_search` | Search the web via DuckDuckGo (no API key) |
| `list_agents` | Discover other agents in the fleet |
| `read_shared_state` | Read from the shared blackboard |
| `write_shared_state` | Write to the shared blackboard |
| `list_shared_state` | Browse blackboard entries by prefix |
| `publish_event` | Publish event to mesh pub/sub |
| `save_artifact` | Save deliverable file and register on blackboard |
| `set_cron` | Schedule a recurring job |
| `set_heartbeat` | Enable autonomous monitoring with probes |
| `list_cron` / `remove_cron` | Manage scheduled jobs |
| `create_skill` | Write a new Python skill at runtime |
| `list_custom_skills` | List all custom skills the agent has created |
| `reload_skills` | Hot-reload all skills |
| `spawn_agent` | Spawn an ephemeral sub-agent |
| `read_agent_history` | Read another agent's conversation logs |

Custom skills are Python functions decorated with `@skill`, auto-discovered
from the agent's `skills_dir` at startup. Agents can also create new skills
at runtime and hot-reload them.

Agents also support **[MCP (Model Context Protocol)](#mcp-tool-support)** — any
MCP-compatible tool server can be plugged in via config, giving agents access to
databases, filesystems, APIs, and more without writing custom skills.

---

## Memory System

Five layers give agents persistent, self-improving memory:

```
Layer 5: Context Manager          ← Manages the LLM's context window
  │  Monitors token usage
  │  Proactive flush at 60% capacity
  │  Auto-compacts at 70% capacity
  │  Extracts facts before discarding messages
  │
Layer 4: Learnings                ← Self-improvement through failure tracking
  │  learnings/errors.md         (tool failures with context)
  │  learnings/corrections.md   (user corrections and preferences)
  │  Auto-injected into system prompt each session
  │
Layer 3: Workspace Files          ← Durable, human-readable storage
  │  AGENTS.md, SOUL.md, USER.md  (loaded into system prompt)
  │  MEMORY.md                    (curated long-term facts)
  │  HEARTBEAT.md                 (autonomous monitoring rules)
  │  memory/YYYY-MM-DD.md         (daily session logs)
  │  BM25 search across all files
  │
Layer 2: Structured Memory DB     ← Hierarchical vector database
  │  SQLite + sqlite-vec + FTS5
  │  Facts with embeddings (KNN similarity search)
  │  Auto-categorization with category-scoped search
  │  3-tier retrieval: categories → scoped facts → flat fallback
  │  Reinforcement scoring with access-count boost + recency decay
  │
Layer 1: Salience Tracking        ← Prioritizes important facts
     Access count, decay score, last accessed timestamp
     High-salience facts auto-surface in initial context
```

### Write-Then-Compact Pattern

Before the context manager discards messages, it:

1. Asks the LLM to extract important facts from the conversation
2. Stores facts in both `MEMORY.md` and the structured memory DB
3. Summarizes the conversation
4. Replaces message history with: summary + last 4 messages

Nothing is permanently lost during compaction.

### Cross-Session Memory

Facts saved with `memory_save` are stored in both the workspace (daily log)
and the structured SQLite database. After a reset or restart, `memory_recall`
retrieves them via semantic search:

```
Session 1: User says "My cat's name is Whiskerino"
           Agent saves to daily log + structured DB

  ═══ Chat Reset ═══

Session 2: User asks "What is my cat's name?"
           Agent recalls "Whiskerino" via memory_recall
```

---

## Triggering & Automation

Agents act autonomously through trigger mechanisms running in the mesh host
(not inside containers, so they survive container restarts).

### Cron Scheduler

Persistent cron jobs that dispatch messages to agents on a schedule. Agents
can schedule their own jobs using the `set_cron` tool.

Supports 5-field cron expressions (`minute hour dom month dow`), interval
shorthand (`every 30m`, `every 2h`), and state persisted to `config/cron.json`.

### Heartbeat System

Cost-efficient autonomous monitoring. Heartbeat jobs run cheap deterministic
probes first — disk usage, pending signals, pending tasks — and only dispatch
to the agent (costing LLM tokens) when probes detect something actionable.

This 5-stage architecture (scheduler → probes → policy → escalation → action)
makes always-on agents economically viable.

### Webhook Endpoints

Named webhook URLs that dispatch payloads to agents:

```bash
curl -X POST http://localhost:8420/webhook/hook/hook_a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{"event": "push", "repo": "myproject"}'
```

### File Watchers

Poll directories for new/modified files matching glob patterns. Uses polling
(not inotify) for Docker volume compatibility.

```yaml
# config/watchers.yaml
watchers:
  - path: "/data/inbox"
    pattern: "*.csv"
    agent: "researcher"
    message: "New prospect list uploaded: {filename}. Begin research."
```

---

## Cost Tracking & Budgets

Every LLM call is tracked at the Credential Vault layer. Per-agent budgets
prevent runaway spend. View costs from the interactive REPL (`/costs`) or
configure budgets in `config/agents.yaml`:

```yaml
agents:
  researcher:
    budget:
      daily_usd: 5.00
      monthly_usd: 100.00
```

When an agent exceeds its budget, the vault rejects LLM calls with an error
instead of forwarding them to the provider.

---

## Security Model

Defense-in-depth with five layers:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Runtime isolation | **Docker Sandbox microVMs** when available; falls back to Docker containers | Agent escape, kernel exploits |
| Container hardening | Non-root user, no-new-privileges, memory/CPU limits | Privilege escalation, resource abuse |
| Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs | Unauthorized data access |
| Input validation | Path traversal prevention, safe condition eval (no `eval()`), token budgets, iteration limits | Injection, runaway loops |

### Dual Runtime Backend

OpenLegion supports two isolation levels:

| | Docker Containers (default) | Docker Sandbox microVMs |
|---|---|---|
| **Isolation** | Shared kernel, namespace separation | Own kernel per agent (hypervisor) |
| **Escape risk** | Kernel exploit could escape | Hypervisor boundary — much harder |
| **Performance** | Native speed | Near-native (Rosetta 2 on Apple Silicon) |
| **Requirements** | Any Docker install | Docker Desktop 4.58+ |
| **Enable** | `openlegion start` | `openlegion start --sandbox` |

**Docker containers** (default) run agents as non-root with `no-new-privileges`, 512MB memory limit, 50% CPU cap, and no host filesystem access. This is secure for most use cases.

**Docker Sandbox microVMs** give each agent its own Linux kernel via Apple Virtualization.framework (macOS) or Hyper-V (Windows). Even if an agent achieves code execution, it's trapped inside a lightweight VM with no visibility into other agents or the host. Use this when running untrusted code or when compliance requires hypervisor isolation.

```bash
# Default: container isolation (works everywhere)
openlegion start

# Maximum security: microVM isolation (Docker Desktop 4.58+ required)
openlegion start --sandbox
```

> **Check compatibility:** Run `docker sandbox version` — if it returns a version number, your Docker Desktop supports sandboxes. If not, update Docker Desktop to 4.58+.

---

## CLI Reference

```
openlegion
├── setup                                # Guided setup wizard
├── start [--config PATH] [-d] [--sandbox]  # Start runtime + interactive REPL
├── stop                                 # Stop all containers
├── chat <name> [--port PORT]            # Connect to a running agent
├── status [--port PORT]                 # Show agent status
│
├── agent
│   ├── add [name]                       # Add a new agent
│   ├── list                             # List configured agents
│   └── remove <name> [--yes]            # Remove an agent
│
└── channels
    ├── add [telegram|discord|slack|whatsapp]  # Connect a messaging channel
    ├── list                                    # Show configured channels
    └── remove <name>                           # Disconnect a channel
```

### Interactive REPL Commands

```
@agent <message>     Send message to a specific agent
/use <agent>         Switch active agent
/agents              List all running agents
/add                 Add a new agent (hot-adds to running system)
/status              Show agent health
/broadcast <msg>     Send message to all agents
/costs               Show today's LLM spend
/reset               Clear conversation with active agent
/quit                Exit and stop runtime
```

### Team Templates

Templates are offered during `openlegion setup`:

| Template | Agents | Description |
|----------|--------|-------------|
| `starter` | assistant | Single general-purpose agent |
| `sales` | researcher, qualifier, outreach | Sales pipeline team |
| `devteam` | pm, engineer, reviewer | Software development team |
| `content` | researcher, writer, editor | Content creation team |

---

## Configuration

### `PROJECT.md` — Fleet-Wide Context

Shared across all agents. Loaded into every agent's system prompt.

```markdown
# PROJECT.md

## What We're Building
SaaS platform for automated lead qualification

## Current Priority
Ship the email personalization pipeline this week

## Hard Constraints
- Budget: $50/day total across all agents
- No cold outreach to .edu or .gov domains
```

### `config/mesh.yaml` — Framework Settings

```yaml
mesh:
  host: "0.0.0.0"
  port: 8420

llm:
  default_model: "openai/gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  max_tokens: 4096
  temperature: 0.7
```

### `config/agents.yaml` — Agent Definitions

Created automatically by `openlegion setup` or `openlegion agent add`.

```yaml
agents:
  researcher:
    role: "research"
    model: "openai/gpt-4o-mini"
    skills_dir: "./skills/research"
    system_prompt: "You are a research specialist..."
    browser_backend: "stealth"          # basic (default), stealth, or advanced
    resources:
      memory_limit: "512m"
      cpu_limit: 0.5
    budget:
      daily_usd: 5.00
      monthly_usd: 100.00
```

### `config/permissions.json` — Agent Permissions

Per-agent access control with glob patterns for blackboard paths and
explicit allowlists for messaging, pub/sub, and API access.

### `config/workflows/*.yaml` — Workflow Definitions

DAG-based workflows with step dependencies, conditions, timeouts,
retry policies, and failure handlers.

### `.env` — API Keys

Managed automatically by `openlegion setup` and `openlegion channels add`. You can also edit directly:

```bash
OPENLEGION_CRED_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_CRED_MOONSHOT_API_KEY=sk-...
OPENLEGION_CRED_OPENAI_API_KEY=sk-...
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC...
OPENLEGION_CRED_DISCORD_BOT_TOKEN=MTIz...
OPENLEGION_CRED_SLACK_BOT_TOKEN=xoxb-...
OPENLEGION_CRED_SLACK_APP_TOKEN=xapp-...
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...

# Log format: "json" (default) or "text" (human-readable)
OPENLEGION_LOG_FORMAT=text
```

### Connecting Channels

```bash
openlegion channels add telegram    # prompts for bot token from @BotFather
openlegion channels add discord     # prompts for bot token
openlegion channels add slack       # prompts for bot + app tokens
openlegion channels add whatsapp    # prompts for access token + phone ID
openlegion channels list            # check what's connected
openlegion channels remove telegram # disconnect
```

On next `openlegion start`, a pairing code appears — send it to your bot to link.

---

## MCP Tool Support

OpenLegion supports the **[Model Context Protocol (MCP)](https://modelcontextprotocol.io)** —
the emerging standard for LLM tool interoperability. Any MCP-compatible tool server
can be plugged into an agent via config, with tools automatically discovered and
exposed to the LLM alongside built-in skills.

### Configuration

Add `mcp_servers` to any agent in `config/agents.yaml`:

```yaml
agents:
  researcher:
    role: "research"
    model: "openai/gpt-4o-mini"
    mcp_servers:
      - name: filesystem
        command: mcp-server-filesystem
        args: ["/data"]
      - name: database
        command: mcp-server-sqlite
        args: ["--db", "/data/research.db"]
```

Each server is launched as a subprocess inside the agent container using stdio
transport. Tools are discovered automatically via the MCP protocol and appear
in the LLM's tool list alongside built-in skills.

### How It Works

1. Agent container reads `MCP_SERVERS` from environment (set by the runtime)
2. `MCPClient` launches each server subprocess via stdio transport
3. MCP protocol handshake discovers available tools and their schemas
4. Tools are registered in `SkillRegistry` with OpenAI function-calling format
5. LLM tool calls route through `MCPClient.call_tool()` to the correct server
6. Name conflicts with built-in skills are resolved by prefixing (`mcp_{server}_{tool}`)

### Server Config Options

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Server identifier (used for logging and conflict prefixes) |
| `command` | string | Command to launch the server |
| `args` | list | Command-line arguments (optional) |
| `env` | dict | Environment variables for the server process (optional) |

See the full **[MCP Integration Guide](docs/mcp.md)** for advanced usage,
custom server setup, and troubleshooting.

---

## Testing

```bash
# Unit and integration tests (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py

# E2E tests (requires Docker + API key)
pytest tests/test_e2e.py tests/test_e2e_chat.py \
  tests/test_e2e_memory.py tests/test_e2e_triggering.py

# Everything
pytest tests/
```

### Test Coverage

| Category | Tests | What's Tested |
|----------|-------|---------------|
| Built-in Tools | 54 | exec, file, browser (incl. backend tiers), memory, mesh tools, discovery |
| Workspace | 41 | File scaffold, loading, BM25 search, daily logs, learnings |
| Slack Channel | 21 | Socket Mode, thread routing, pairing, command translation |
| WhatsApp Channel | 21 | Cloud API, webhook verification, message chunking |
| Channels (base) | 21 | Abstract channel, commands, per-user routing, chunking |
| Mesh | 28 | Blackboard, PubSub, MessageRouter, permissions |
| Memory Store | 34 | SQLite ops, vector search, categories, hierarchical search, tool outcomes |
| Orchestrator | 25 | Workflows, conditions, retries, failures |
| Cron | 25 | Cron expressions, intervals, dispatch, persistence |
| Agent Loop | 21 | Task execution, tool calling, cancellation, tool memory |
| Context Manager | 17 | Token estimation (tiktoken + model-aware), compaction, flushing |
| Runtime Backend | 20 | DockerBackend, SandboxBackend, browser_backend, detection, selection |
| Transport | 18 | HttpTransport, SandboxTransport, resolve_url |
| Skills | 19 | Discovery, execution, injection, MCP integration |
| Chat | 18 | Chat mode, workspace integration, cross-session |
| Failover | 15 | Health tracking, chain cascade, cooldown |
| Credentials | 13 | Vault, API proxy, provider detection |
| Integration | 13 | Multi-component mesh operations |
| MCP Client | 10 | Tool discovery, routing, conflicts, lifecycle |
| Costs | 10 | Usage recording, budgets, vault integration |
| CLI | 8 | Agent add/list/remove, chat, setup |
| Types | 8 | Pydantic validation, serialization |
| MCP E2E | 7 | Real MCP protocol with live server subprocess |
| Webhooks | 7 | Add/remove, persistence, dispatch |
| File Watchers | 7 | Polling, dispatch, pattern matching |
| Memory Tools | 6 | memory_search, memory_save, memory_recall |
| Memory Integration | 6 | Vector search, cross-task recall, salience |
| E2E | 17 | Container health, workflow, chat, memory, triggering |
| Queue Modes | 7 | Steer, collect, followup queue strategies |
| **Total** | **638** | |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | HTTP servers (mesh + agent) |
| uvicorn | ASGI server |
| httpx | Async HTTP client |
| pydantic | Data validation |
| litellm | Multi-provider LLM interface (100+ providers) |
| sqlite-vec | Vector search in SQLite |
| pyyaml | YAML config parsing |
| click | CLI framework |
| docker | Docker API client |
| python-dotenv | `.env` file loading |
| playwright | Browser automation — basic tier (in container only) |
| camoufox | Anti-detect browser — stealth tier (in container only) |
| mcp | MCP tool server client (in container only, optional) |
| slack-bolt | Slack channel adapter (optional) |

Dev: pytest, pytest-asyncio, ruff.

No LangChain. No Redis. No Kubernetes. No web UI.

---

## Project Structure

```
src/
├── cli.py                              # CLI entry point
├── agent/
│   ├── __main__.py                     # Container entry
│   ├── loop.py                         # Execution loop (task + chat)
│   ├── skills.py                       # Skill registry + discovery
│   ├── mcp_client.py                   # MCP server lifecycle + tool routing
│   ├── memory.py                       # Hierarchical memory (SQLite + sqlite-vec + FTS5)
│   ├── workspace.py                    # Persistent workspace + BM25 search
│   ├── context.py                      # Context manager (token tracking, compaction)
│   ├── llm.py                          # LLM client (routes through mesh proxy)
│   ├── mesh_client.py                  # Mesh HTTP client
│   ├── server.py                       # Agent FastAPI server
│   └── builtins/
│       ├── exec_tool.py                # Shell execution
│       ├── file_tool.py                # File I/O (read, write, list)
│       ├── http_tool.py                # HTTP requests
│       ├── browser_tool.py             # Playwright automation
│       ├── memory_tool.py              # Memory search, save, recall
│       └── mesh_tool.py               # Shared state, fleet awareness, artifacts
├── host/
│   ├── mesh.py                         # Blackboard, PubSub, MessageRouter
│   ├── orchestrator.py                 # Workflow DAG executor
│   ├── permissions.py                  # Permission matrix
│   ├── credentials.py                  # Credential vault + API proxy
│   ├── failover.py                     # Model health tracking + failover chains
│   ├── runtime.py                      # RuntimeBackend ABC + Docker/Sandbox backends
│   ├── transport.py                    # Transport ABC + Http/Sandbox transports
│   ├── server.py                       # Mesh FastAPI server
│   ├── cron.py                         # Cron scheduler
│   ├── webhooks.py                     # Named webhook endpoints
│   ├── watchers.py                     # File watchers (polling)
│   ├── costs.py                        # Cost tracking + budgets (SQLite)
│   ├── health.py                       # Health monitor + auto-restart
│   └── lanes.py                        # Per-agent task queues
├── shared/
│   ├── types.py                        # All Pydantic models (the contract)
│   └── utils.py                        # ID generation, logging
├── channels/
│   ├── base.py                         # Abstract channel with unified UX
│   ├── telegram.py                     # Telegram adapter
│   ├── discord.py                      # Discord adapter
│   ├── slack.py                        # Slack adapter (Socket Mode)
│   ├── whatsapp.py                     # WhatsApp Cloud API adapter
│   └── webhook.py                      # Workflow trigger webhook adapter
└── templates/
    ├── starter.yaml                    # Single-agent template
    ├── sales.yaml                      # Sales pipeline team
    ├── devteam.yaml                    # Dev team template
    └── content.yaml                    # Content creation team

config/
├── mesh.yaml                           # Framework settings
├── agents.yaml                         # Agent definitions (per-project)
├── permissions.json                    # Per-agent ACLs
└── workflows/                          # Workflow YAML definitions
```

---

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Messages, not method calls | Agents communicate through HTTP/JSON. Never shared memory or direct invocation. |
| The mesh is the only door | No agent has network access except through the mesh. No agent holds credentials. |
| Private by default, shared by promotion | Agents keep knowledge private. Facts are explicitly promoted to the blackboard. |
| Explicit failure handling | Every workflow step declares what happens on failure. No silent error swallowing. |
| Small enough to audit | ~11,000 total lines. The entire codebase is auditable in a day. |
| Skills over features | New capabilities are agent skills, not mesh or orchestrator code. |
| SQLite for all state | Single-file databases. No external services. WAL mode for concurrent reads. |
| Zero vendor lock-in | LiteLLM supports 100+ providers. Markdown workspace files. No proprietary formats. |

---

## License

**MIT** — see [LICENSE](LICENSE).

OpenLegion is free and open-source. Self-host it, fork it, build on it.

---

## Related Projects & Comparisons

Looking for alternatives? OpenLegion is often compared to:

- **OpenClaw** — personal AI assistant, 200K+ stars, not designed for production security
- **nanobot** — ultra-lightweight Python agent (~4K lines), limited multi-agent support
- **ZeroClaw** — Rust-based AI agent runtime, extreme resource efficiency, early-stage
- **NanoClaw** — container-isolated, Claude-only, no cost tracking
- **LangChain Agents** — feature-rich but complex, heavy framework overhead
- **CrewAI** — multi-agent framework, no built-in container isolation or cost controls
- **AutoGen** — Microsoft's multi-agent framework, requires Azure/OpenAI, no self-hosting

OpenLegion differs from all of these in combining **fleet orchestration,
Docker isolation, credential vaulting, and cost enforcement** in a single
~11,000 line auditable codebase.

**Keywords:** autonomous AI agents, multi-agent framework, LLM agent orchestration,
self-hosted AI agents, Docker AI agents, OpenClaw alternative, AI agent security,
agent cost tracking, Telegram AI bot, Python AI agent framework
