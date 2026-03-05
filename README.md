<p align="center">
  <img width="450" alt="openlegion-logo-new" src="https://github.com/user-attachments/assets/08912b04-8df1-4473-b679-6bbac0c3ae2f" />
</p>
<h3 align="center">
  <b>The AI agent framework built for builders who can't afford a security incident.</b>
</h3>
<div align="center">
   
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests: 1744](https://img.shields.io/badge/tests-1744%20passing-brightgreen)](https://github.com/openlegion-ai/openlegion/actions/workflows/test.yml)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/mXNkjpDvvr)
[![Twitter](https://img.shields.io/badge/Twitter-@openlegion-1DA1F2?logo=x&logoColor=white)](https://x.com/openlegion)
[![LiteLLM](https://img.shields.io/badge/LLM-100%2B%20providers-orange.svg)](https://litellm.ai)
[![Docker](https://img.shields.io/badge/isolation-Docker%20%2B%20microVM-blue.svg)]()
   
</div>

> **Autonomous AI agent fleets — isolated, auditable, and production-ready.**
> Every agent runs in its own Docker container. API keys never leave the vault.
> Chat via Telegram, Discord, Slack, or WhatsApp. Built-in cost controls. 100+ LLM providers.

[Quick Start](#quick-start) · [Full Setup Guide](QUICKSTART.md) · [Why Not OpenClaw?](#why-not-openclaw) · [Docs](docs/)

---

## Demo

https://github.com/user-attachments/assets/643f7e71-b8e7-4960-bc01-afa31b45b107

> `openlegion start` → inline setup → multiple agents running.
> Live cost tracking. No configuration files edited by hand.
> Connect Telegram, WhatsApp, Slack, and Discord.

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
openlegion start                 # inline setup on first run, then launch agents
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
powershell -ExecutionPolicy Bypass -File install.ps1
openlegion start
```

> First install downloads ~70 packages and takes 2-3 minutes. Subsequent installs are fast.
>
> **Need help?** See the **[full setup guide](QUICKSTART.md)** for platform-specific instructions and troubleshooting.

```bash
# Add more agents from the REPL
/add

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
| **Test coverage** | Minimal | 1744 tests including full Docker E2E |
| **Codebase size** | 430,000+ lines | ~25,000 lines — auditable in a day |

---

## What It Does

OpenLegion is an **autonomous AI agent framework** for running multi-agent
fleets in isolated Docker containers. Each agent gets its own memory, tools, schedule,
and budget — coordinated through deterministic YAML workflows with no LLM routing layer.

Chat with your agent fleet via **Telegram**, **Discord**, **Slack**, **WhatsApp**, or CLI. Agents act autonomously
via cron schedules, webhooks, heartbeat monitoring, and file watchers — without being
prompted.

**1744 tests passing** across **~25,000 lines** of application code.
**Fully auditable in a day.**
No LangChain. No Redis. No Kubernetes. No CEO agent. BSL License.

1. **Security by architecture** — every agent runs in an isolated Docker container
   (microVM when available). API keys live in the credential vault — agents call
   through a proxy and never handle credentials directly. Defense-in-depth with
   6 security layers.

2. **Production-grade cost control** — per-agent LLM token tracking with enforced
   daily and monthly budget caps at the vault layer. Agents physically cannot spend
   what you haven't authorized. View live spend with `/costs` in the REPL.

3. **Deterministic multi-agent orchestration** — YAML-defined DAG workflows with
   step dependencies, conditions, retries, and failure handlers. No LLM decides
   what runs next. Predictable, debuggable, auditable.

4. **Acts autonomously** — cron schedules, heartbeat probes, webhook triggers, and file watchers let agents work without being prompted.

5. **Self-aware and self-improving** — agents understand their own permissions, budget, fleet topology, and system architecture via auto-generated `SYSTEM.md` and live runtime context. They learn from tool failures and user corrections, injecting past learnings into future sessions.

6. **Self-extends** — agents write their own Python skills at runtime and hot-reload them. Agents can also spawn sub-agents for specialized work.

7. **Multi-channel** — connect agents to Telegram, Discord, Slack, and WhatsApp. Also accessible via CLI and API.

8. **Real-time dashboard** — web-based fleet observability with consolidated navigation, slide-over chat panels, keyboard command palette, grouped request traces, live event streaming, streaming broadcast with real-time per-agent responses, LLM prompt/response previews, agent management, agent identity editor (personality, instructions, preferences, heartbeat rules, memory, activity logs, learnings), cost charts, cron management, and embedded KasmVNC viewer for persistent browser agents.

9. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly budget enforcement.

10. **Runs deterministic workflows** — YAML-defined DAG workflows chain agents in sequence with conditions, retries, and failure handlers.

11. **Fails over across providers** — configurable model failover chains cascade across LLM providers with per-model health tracking and exponential cooldown.

12. **Token-level streaming** — real-time token-by-token LLM responses across CLI, dashboard, Telegram, Discord, and Slack with progressive message editing and graceful non-streaming fallback.

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
│   - chat / status                                                       │
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
  own memory DB, own workspace, 384MB RAM, 0.15 CPU
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
Project agents' blackboard access is automatically scoped to `projects/{name}/*` —
agents use natural keys (e.g. `tasks/research_abc123`) while the MeshClient
transparently namespaces them under the project. Standalone agents have no
blackboard access.

| Namespace | Purpose | Example |
|-----------|---------|---------|
| `tasks/*` | Task assignments | `tasks/research_abc123` |
| `context/*` | Shared agent context | `context/prospect_acme` |
| `signals/*` | Inter-agent signals | `signals/research_complete` |
| `history/*` | Append-only audit log | `history/action_xyz` |

### Credential Vault (API Proxy)

Agents never hold API keys. All external API calls route through the mesh.
The vault uses a two-tier prefix system: `OPENLEGION_SYSTEM_*` for LLM
provider keys (never agent-accessible) and `OPENLEGION_CRED_*` for agent-tier
tool/service keys. Budget limits are enforced before dispatching LLM calls
and token usage is recorded after each response.

### Model Failover

Configurable failover chains cascade across LLM providers transparently.
`ModelHealthTracker` applies exponential cooldown per model (transient errors:
60s → 300s → 1500s, billing/auth errors: 1h). Streaming failover is supported — if a connection fails mid-stream,
the next model in the chain picks up.

### Permission Matrix

Every inter-agent operation is checked against per-agent ACLs:

```json
{
  "researcher": {
    "can_message": ["orchestrator"],
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"],
    "blackboard_read": ["projects/myproject/*"],
    "blackboard_write": ["projects/myproject/*"],
    "allowed_apis": ["llm", "brave_search"],
    "allowed_credentials": ["brightdata_*"]
  }
}
```

Blackboard patterns use the `projects/{name}/*` namespace. When an agent joins a
project, it receives read/write access to that namespace. Standalone agents get
empty blackboard permissions.

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

Agent containers are slim — no browser. Browsing is handled by a shared browser service container (Camoufox + KasmVNC).

**Agent container:**
- **Image**: `openlegion-agent:latest` (Python 3.12, system tools — no browser)
- **Network**: Bridge with port mapping (macOS/Windows) or host network (Linux)
- **Volume**: `openlegion_data_{agent_id}` mounted at `/data` (agent names with spaces/special chars are sanitized)
- **Resources**: 384MB RAM, 0.15 CPU (agents are I/O-bound — waiting on LLM APIs)
- **Security**: `no-new-privileges`, runs as non-root `agent` user (UID 1000)
- **Port**: 8400 (FastAPI)

**Browser service container** (shared across all agents):
- **Image**: `openlegion-browser:latest` (Camoufox stealth browser + KasmVNC)
- **Resources**: 2GB RAM, 1 CPU, 512MB shared memory
- **Ports**: 8500 (browser API), 6080 (KasmVNC web client)
- **Capacity**: up to 5 concurrent browser sessions

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
│    GET /workspace  GET|PUT /workspace/{file}                  │
│    GET /heartbeat-context                                     │
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
(PROJECT.md if in a project, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md, SYSTEM.md) into the system prompt,
injects a live Runtime Context block (permissions, budget, fleet, cron),
and searches memory for relevant facts. Executes tool calls in a bounded loop
(max 30 rounds per turn, 200 total rounds per session) and runs context compaction when needed.

### Tool Loop Detection

Both modes include automatic detection of stuck tool-call loops. A sliding
window tracks recent `(tool_name, params_hash, result_hash)` tuples and
escalates through three levels:

| Level | Trigger | Action |
|-------|---------|--------|
| **Warn** | 2nd identical call | System message: "Try a different approach" |
| **Block** | 4th identical call | Tool skipped, error returned to agent |
| **Terminate** | 9th call with same params | Loop terminated with failure status |

`memory_search` is exempt since repeated searches are legitimate. Detection uses SHA-256 hashes of
canonicalized parameters and results over a 15-call sliding window.

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `exec` | Shell command execution with timeout |
| `read_file` | Read file contents from `/data` |
| `write_file` | Write/append file in `/data` |
| `list_files` | List/glob files in `/data` |
| `http_request` | HTTP GET/POST/PUT/DELETE/PATCH |
| `browser_navigate` | Open URL, extract page text via shared browser service |
| `browser_snapshot` | Accessibility tree snapshot with element refs (e1, e2, ...) |
| `browser_screenshot` | Capture page screenshot |
| `browser_click` | Click element by ref or CSS selector |
| `browser_type` | Fill input by ref or CSS selector (supports `$CRED{}` handles) |
| `browser_evaluate` | Run JavaScript in page |
| `browser_reset` | Reset browser session (profile preserved) |
| `browser_solve_captcha` | Manual CAPTCHA detection and solving |
| `memory_search` | Hybrid search across workspace files and structured DB |
| `memory_save` | Save fact to workspace and structured memory DB |
| `web_search` | Search the web via DuckDuckGo (no API key) |
| `notify_user` | Send notification to user across all connected channels |
| `list_agents` | Discover agents in your project (standalone agents see only themselves) |
| `read_shared_state` | Read from the shared blackboard |
| `write_shared_state` | Write to the shared blackboard |
| `list_shared_state` | Browse blackboard entries by prefix |
| `publish_event` | Publish event to mesh pub/sub |
| `subscribe_event` | Subscribe to a pub/sub topic at runtime |
| `watch_blackboard` | Watch blackboard keys matching a glob pattern |
| `claim_task` | Atomically claim a task from the shared blackboard |
| `save_artifact` | Save deliverable file and register on blackboard |
| `update_workspace` | Update identity files (SOUL.md, INSTRUCTIONS.md, USER.md, HEARTBEAT.md) |
| `set_cron` | Schedule a recurring job (set `heartbeat=true` for autonomous wakeups) |
| `list_cron` / `remove_cron` | Manage scheduled jobs |
| `create_skill` | Write a new Python skill at runtime |
| `reload_skills` | Hot-reload all skills |
| `spawn_agent` | Spawn an ephemeral sub-agent in a new container |
| `spawn_subagent` | Spawn a lightweight in-container subagent for parallel subtasks |
| `list_subagents` | List active subagents and their status |
| `wait_for_subagent` | Wait for a subagent to complete and return its result |
| `vault_generate_secret` | Generate and store a random secret (returns opaque handle) |
| `vault_capture_from_page` | Capture text from browser element and store as credential |
| `vault_list` | List credential names (names only, never values) |
| `introspect` | Query own runtime state: permissions, budget, fleet, cron, health |
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
  │  INSTRUCTIONS.md, SOUL.md, USER.md  (loaded into system prompt)
  │  SYSTEM.md                    (auto-generated architecture guide + runtime snapshot)
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
and the structured SQLite database. After a reset or restart, `memory_search`
retrieves them via hybrid search:

```
Session 1: User says "My cat's name is Whiskerino"
           Agent saves to daily log + structured DB

  ═══ Chat Reset ═══

Session 2: User asks "What is my cat's name?"
           Agent recalls "Whiskerino" via memory_search
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

When a heartbeat fires, the agent receives enriched context: its HEARTBEAT.md
rules, recent daily logs, probe alerts, and actual pending signal/task content
— all in a single message. If HEARTBEAT.md is the default scaffold, no recent
activity exists, and no probes triggered, the dispatch is skipped entirely
(zero LLM cost).

This 5-stage architecture (scheduler → probes → context → policy → action)
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

File watchers are configured programmatically via the `FileWatcher.watch()` method,
specifying a directory path, glob pattern, target agent, and message template
(supports `{filepath}` and `{filename}` placeholders).

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

Defense-in-depth with six layers:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Runtime isolation | **Docker Sandbox microVMs** when available; falls back to Docker containers | Agent escape, kernel exploits |
| Container hardening | Non-root user, no-new-privileges, memory/CPU limits | Privilege escalation, resource abuse |
| Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs | Unauthorized data access |
| Input validation | Path traversal prevention, SSRF blocking, safe condition eval (no `eval()`), token budgets, iteration limits, rate limiting | Injection, runaway loops, network abuse |
| Unicode sanitization | Invisible character stripping at five choke points (user input, tool results, workspace, mesh tools, dashboard) | Prompt injection via hidden Unicode |

### Dual Runtime Backend

OpenLegion supports two isolation levels:

| | Docker Containers (default) | Docker Sandbox microVMs |
|---|---|---|
| **Isolation** | Shared kernel, namespace separation | Own kernel per agent (hypervisor) |
| **Escape risk** | Kernel exploit could escape | Hypervisor boundary — much harder |
| **Performance** | Native speed | Near-native (Rosetta 2 on Apple Silicon) |
| **Requirements** | Any Docker install | Docker Desktop 4.58+ |
| **Enable** | `openlegion start` | `openlegion start --sandbox` |

**Docker containers** (default) run agents as non-root with `no-new-privileges`, 384MB memory limit, 0.15 CPU cap, and no host filesystem access. Browser operations are handled by a shared browser service container (2GB RAM, 1 CPU). This is secure for most use cases.

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
openlegion [--verbose/-v] [--quiet/-q] [--json]
├── start [--config PATH] [-d] [--sandbox]     # Start runtime + interactive REPL (inline setup on first run)
├── stop                                       # Stop all containers
├── chat [name] [--port PORT]                  # Connect to a running agent
├── status [--port PORT] [--wide/-w] [--watch N] [--json]  # Show agent status
└── version [--verbose/-v]                     # Show version and environment info
```

> Agent management, credentials, blackboard, cron, workflows, projects, and channels
> are managed via **REPL commands** (below) inside a running session, or via the
> **web dashboard** at `http://localhost:8420`.

### Interactive REPL Commands

```
@agent <message>                     Send message to a specific agent
/use <agent>                         Switch active agent
/agents                              List all running agents
/add                                 Add a new agent (hot-adds to running system)
/agent [edit|view]                   Agent overview, config editing, workspace files
/edit [name]                         Edit agent settings (model, browser, budget)
/remove [name]                       Remove an agent
/restart [name]                      Restart an agent container
/status                              Show agent health
/broadcast <msg>                     Send message to all agents
/steer <msg>                         Inject message into busy agent's context
/history [agent]                     Show recent conversation messages
/costs                               Show today's LLM spend + context usage + model health
/blackboard [list|get|set|del]       View/edit shared blackboard entries
/queue                               Show agent task queue status
/workflow [list|run]                 List or trigger workflows
/cron [list|del|pause|resume|run]    Manage cron jobs
/project [list|use|info]              Manage multi-project namespaces
/credential [add|list|remove]        Manage API credentials
/debug [trace]                       Show recent request traces
/logs [--level LEVEL]                Show recent runtime logs
/addkey <svc> [key]                  Add an API credential to the vault
/removekey [name]                    Remove a credential from the vault
/reset                               Clear conversation with active agent
/quit                                Exit and stop runtime

Aliases: /exit = /quit, /agents = /status, /traces = /debug
```

### Team Templates

Templates are offered during first-run setup (via `openlegion start`):

| Template | Agents | Description |
|----------|--------|-------------|
| `starter` | assistant | Single general-purpose agent |
| `sales` | researcher, qualifier, outreach | Sales pipeline team |
| `devteam` | pm, engineer, reviewer | Software development team |
| `content` | researcher, writer, editor | Content creation team |
| `deep-research` | researcher, analyst | Deep research and analysis team |
| `monitor` | monitor | Autonomous monitoring agent |

---

## Configuration

### `PROJECT.md` — Per-Project Context

Each project has its own `PROJECT.md` stored in `config/projects/{name}/project.md`.
It is mounted into project member agents' containers and loaded into their system
prompts. Standalone agents (not in a project) do not receive any PROJECT.md.

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
  embedding_model: "text-embedding-3-small"   # "none" to disable vector search

collaboration: true                           # allow agents to message each other (default: true for new agents)
```

### `config/agents.yaml` — Agent Definitions

Created automatically by `openlegion start` (inline setup) or the `/add` REPL command.

```yaml
agents:
  researcher:
    role: "research"
    model: "openai/gpt-4.1-mini"
    skills_dir: "./skills/researcher"
    initial_instructions: "You are a research specialist..."
    thinking: "medium"                   # off (default), low, medium, or high
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

Managed automatically by `openlegion start` (setup wizard) and the `/addkey` REPL command. You can also edit directly. Uses a two-tier prefix system:

```bash
# System tier — LLM provider keys (never accessible by agents)
OPENLEGION_SYSTEM_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_SYSTEM_OPENAI_API_KEY=sk-...
OPENLEGION_SYSTEM_MOONSHOT_API_KEY=sk-...

# Agent tier — tool/service keys (access controlled per-agent)
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...

# Channel tokens
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

Channels are configured via the setup wizard during `openlegion start`, or by
adding the appropriate tokens to `.env` directly:

```bash
# Telegram
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC...

# Discord
OPENLEGION_CRED_DISCORD_BOT_TOKEN=MTIz...

# Slack (both required)
OPENLEGION_CRED_SLACK_BOT_TOKEN=xoxb-...
OPENLEGION_CRED_SLACK_APP_TOKEN=xapp-...

# WhatsApp (both required)
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...
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
    model: "openai/gpt-4.1-mini"
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
| Built-in Tools | 145 | exec, file, browser (Camoufox, screenshots, reset/recovery), memory, mesh, vault, introspect, path traversal, discovery |
| Dashboard | 104 | Fleet management, blackboard, costs, traces, queues, cron, settings, config, streaming broadcast, workspace proxy |
| Workspace | 68 | File scaffold, loading, BM25 search, daily logs, learnings, heartbeat, identity files, SYSTEM.md |
| Agent Loop | 67 | Task execution, tool calling, cancellation, tool memory, chat helpers, daily log enrichment, task logging |
| Credentials | 67 | Vault, API proxy, provider detection, two-tier system, credential lifecycle |
| CLI | 80 | Agent add/list/edit/remove, chat, REPL commands, cron management, version |
| Channels (base) | 47 | Abstract channel, commands, per-user routing, chunking, steer, debug, addkey normalization, parallel broadcast |
| Cron | 44 | Cron expressions, intervals, dispatch, persistence, enriched heartbeat, skip-LLM, concurrent mutations |
| Mesh | 42 | Blackboard, PubSub, MessageRouter, permissions |
| Integration | 42 | Multi-component mesh operations, notifications |
| Sanitization | 38 | Invisible Unicode stripping, bidi overrides, tag chars, zero-width |
| Discord Channel | 36 | Slash commands, message routing, pairing, chunking, embed formatting |
| Memory Store | 34 | SQLite ops, vector search, categories, hierarchical search, tool outcomes |
| Context Manager | 34 | Token estimation (tiktoken + model-aware), compaction, flushing, flush reset |
| Runtime Backend | 34 | DockerBackend, SandboxBackend, extra_env, name sanitization, detection, VNC allocation |
| Projects | 47 | Multi-project CRUD, config, agent membership, blackboard key scoping, cross-project permission isolation |
| Events | 31 | Event streaming, filtering, WebSocket, notification events |
| Traces | 30 | Trace recording, grouping, summaries, prompt preview extraction |
| Orchestrator | 32 | Workflows, conditions, retries, failures |
| Transcript | 24 | Transcript formatting, safety, round-trip fidelity |
| WhatsApp Channel | 22 | Cloud API, webhook verification, message chunking, non-text reply |
| Agent Server | 21 | Workspace API, heartbeat-context endpoint, content sanitization, file allowlist |
| Slack Channel | 21 | Socket Mode, thread routing, pairing, command translation |
| Skills | 20 | Discovery, execution, injection, MCP integration |
| Setup Wizard | 20 | Quickstart, full setup, API key validation, templates, inline setup, two-tier credentials |
| Marketplace | 20 | Install, manifest parsing, validation, path traversal, remove |
| Transport | 18 | HttpTransport, SandboxTransport, resolve_url |
| Dashboard Workspace | 17 | Workspace proxy endpoints, filename validation, transport forwarding, sanitization |
| Vault | 16 | Credential storage, budget enforcement, failover |
| Failover | 15 | Health tracking, chain cascade, cooldown |
| Loop Detector | 14 | Agent loop detection and intervention |
| Lanes | 13 | Per-agent FIFO task queues |
| Costs | 13 | Usage recording, budgets, vault integration, budget overrun warnings |
| Chat | 12 | Chat mode, streaming, workspace integration |
| Subagent | 11 | Spawn, depth/concurrent limits, TTL timeout, skill cloning, memory isolation |
| MCP Client | 10 | Tool discovery, routing, conflicts, lifecycle |
| Chat Workspace | 10 | Cross-session memory, corrections, learnings |
| Health Monitor | 8 | Ephemeral cleanup, TTL expiry, event emission, restart with missing config, parallel health checks |
| Types | 8 | Pydantic validation, serialization |
| MCP E2E | 7 | Real MCP protocol with live server subprocess |
| Webhooks | 7 | Add/remove, persistence, dispatch |
| File Watchers | 7 | Polling, dispatch, pattern matching |
| Memory Tools | 6 | memory_search, memory_save |
| Memory Integration | 6 | Vector search, cross-task recall, salience |
| E2E | 17 | Container health, workflow, chat, memory, triggering |
| **Total** | **1744** | |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | HTTP servers (mesh + agent + browser service) |
| uvicorn | ASGI server |
| httpx | Async HTTP client |
| pydantic | Data validation |
| litellm | Multi-provider LLM interface (100+ providers) |
| sqlite-vec | Vector search in SQLite |
| pyyaml | YAML config parsing |
| click | CLI framework |
| docker | Docker API client |
| python-dotenv | `.env` file loading |
| camoufox | Stealth browser automation (in browser service container only) |
| mcp | MCP tool server client (in agent container only, optional) |
| slack-bolt | Slack channel adapter (optional) |

Dev: pytest, pytest-asyncio, pytest-cov, ruff.

No LangChain. No Redis. No Kubernetes. Real-time web dashboard. Optional channels: `python-telegram-bot`, `discord.py`, `slack-bolt`.

---

## Project Structure

```
src/
├── cli/
│   ├── main.py                         # Click commands and entry point
│   ├── config.py                       # Config loading, Docker helpers, agent management
│   ├── runtime.py                      # RuntimeContext — full lifecycle management
│   ├── repl.py                         # REPLSession — interactive command dispatch
│   ├── channels.py                     # ChannelManager — messaging channel lifecycle
│   └── formatting.py                   # Tool display, styled output, response rendering
├── agent/
│   ├── __main__.py                     # Container entry
│   ├── loop.py                         # Execution loop (task + chat)
│   ├── loop_detector.py                # Tool loop detection (warn/block/terminate)
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
│       ├── browser_tool.py             # Browser automation via shared Camoufox service
│       ├── web_search_tool.py          # Web search via DuckDuckGo
│       ├── memory_tool.py              # Memory search and save
│       ├── mesh_tool.py                # Shared state, fleet awareness, artifacts
│       ├── vault_tool.py               # Credential vault operations
│       ├── skill_tool.py               # Runtime skill creation + hot-reload
│       ├── introspect_tool.py          # Live runtime state queries
│       └── subagent_tool.py            # Spawn in-process subagents
├── host/
│   ├── server.py                       # Mesh FastAPI server
│   ├── mesh.py                         # Blackboard, PubSub, MessageRouter
│   ├── orchestrator.py                 # Workflow DAG executor
│   ├── permissions.py                  # Permission matrix
│   ├── credentials.py                  # Credential vault + API proxy
│   ├── failover.py                     # Model health tracking + failover chains
│   ├── runtime.py                      # RuntimeBackend ABC + Docker/Sandbox backends
│   ├── transport.py                    # Transport ABC + Http/Sandbox transports
│   ├── containers.py                   # Backward-compat alias for DockerBackend
│   ├── cron.py                         # Cron scheduler + heartbeats
│   ├── webhooks.py                     # Named webhook endpoints
│   ├── watchers.py                     # File watchers (polling)
│   ├── costs.py                        # Cost tracking + budgets (SQLite)
│   ├── health.py                       # Health monitor + auto-restart
│   ├── lanes.py                        # Per-agent FIFO task queues
│   ├── traces.py                       # Request tracing + grouped summaries
│   └── transcript.py                   # Provider-specific transcript sanitization
├── shared/
│   ├── types.py                        # All Pydantic models (the contract)
│   ├── utils.py                        # ID generation, logging, sanitization
│   └── trace.py                        # Trace ID generation + correlation
├── browser/
│   ├── server.py                       # Browser service FastAPI server
│   ├── service.py                      # Camoufox session management
│   ├── stealth.py                      # Anti-detection configuration
│   └── redaction.py                    # Credential redaction for browser content
├── channels/
│   ├── base.py                         # Abstract channel with unified UX
│   ├── telegram.py                     # Telegram adapter
│   ├── discord.py                      # Discord adapter
│   ├── slack.py                        # Slack adapter (Socket Mode)
│   ├── whatsapp.py                     # WhatsApp Cloud API adapter
│   └── webhook.py                      # Workflow trigger webhook adapter
├── dashboard/
│   ├── server.py                       # Dashboard FastAPI router + API
│   ├── events.py                       # EventBus for real-time streaming
│   ├── templates/index.html            # Dashboard UI (Alpine.js + Tailwind)
│   └── static/                         # CSS + JS assets
├── setup_wizard.py                    # Guided setup wizard
├── marketplace.py                     # Skill marketplace (git-based install/remove)
└── templates/
    ├── starter.yaml                    # Single-agent template
    ├── sales.yaml                      # Sales pipeline team
    ├── devteam.yaml                    # Dev team template
    ├── content.yaml                    # Content creation team
    ├── deep-research.yaml              # Deep research and analysis team
    └── monitor.yaml                    # Autonomous monitoring agent

config/
├── mesh.yaml                           # Framework settings
├── agents.yaml                         # Agent definitions (per-project)
├── permissions.json                    # Per-agent ACLs
├── projects/                           # Multi-project namespaces
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
| Small enough to audit | ~25,000 total lines. The entire codebase is auditable in a day. |
| Skills over features | New capabilities are agent skills, not mesh or orchestrator code. |
| SQLite for all state | Single-file databases. No external services. WAL mode for concurrent reads. |
| Zero vendor lock-in | LiteLLM supports 100+ providers. Markdown workspace files. No proprietary formats. |

---

## License

OpenLegion.ai is source-available under the Business Source License 1.1 (BSL).

You may view, modify, and self-host the software.

You may NOT offer it as a competing hosted or SaaS product.

See [LICENSE](LICENSE) for details.

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
~25,000 line auditable codebase.

**Keywords:** autonomous AI agents, multi-agent framework, LLM agent orchestration,
self-hosted AI agents, Docker AI agents, OpenClaw alternative, AI agent security,
agent cost tracking, Telegram AI bot, Python AI agent framework
