# OpenLegion

A fleet platform for autonomous LLM agents in Python.

OpenLegion runs LLM-powered agents in isolated Docker containers. Each agent is
an independent worker with its own memory, skills, schedule, and budget. The user
manages the fleet directly — there is no CEO agent or centralized router. Agents
coordinate through shared state (blackboard + PROJECT.md) and deterministic YAML
workflows, not through conversations with each other.

**318 tests passing** across **~9,200 lines** of application code.
Zero framework dependencies. Fully auditable.

---

## Table of Contents

- [Quick Start](#quick-start)
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
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/openlegion/openlegion.git && cd openlegion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Guided setup: API key, project description, first agent, Docker image
openlegion setup

# 3. Start the runtime and chat with your agents
openlegion start
```

That's it. `setup` walks you through everything once. `start` launches your agents and drops you into an interactive chat.

```bash
# Add more agents later
openlegion agent add researcher

# Check status
openlegion status

# Run in background (e.g. for servers)
openlegion start -d
openlegion chat researcher   # connect from another terminal
openlegion stop              # clean shutdown
```

### Requirements

- Python 3.12+
- Docker Engine running and accessible
- An LLM API key (OpenAI, Anthropic, or Groq)

---

## What It Does

1. **Runs LLM agents in isolated Docker containers** — each agent has its own
   filesystem, memory database, resource limits, and security boundary.

2. **Fleet model, not hierarchy** — each agent is independently useful. No CEO
   agent routes tasks. The user talks to agents directly via the interactive REPL.

3. **Shared alignment via PROJECT.md** — a single document loaded into every
   agent's system prompt defines what the fleet is building, the current priority,
   and hard constraints. Alignment without centralized control.

4. **Provides built-in tools** — shell execution, file I/O, HTTP requests, browser
   automation (Playwright/Chromium), web search, and persistent memory.

5. **Persists memory across sessions** — workspace files, BM25 search, context
   compaction, and daily session logs ensure agents remember.

6. **Acts autonomously** — cron schedules, heartbeat probes (cheap deterministic
   checks before LLM escalation), webhook triggers, and file watchers let agents
   work without being prompted.

7. **Self-improving** — agents learn from tool failures and user corrections,
   injecting past learnings into future sessions to avoid repeating mistakes.

8. **Self-extending** — agents can write their own Python skills at runtime and
   hot-reload them. Agents can also spawn ephemeral sub-agents for specialized work.

9. **Multi-channel** — connect agents to Telegram and Discord so users can chat
   from any device. Agents are also accessible via CLI and API.

7. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly
   budget enforcement.

8. **Deterministic workflows** — YAML-defined DAG workflows chain agents in
   sequence. No LLM decides the flow — the user defines it.

---

## Why No CEO Agent

OpenLegion uses a **fleet model**, not a hierarchy. There is no CEO agent
that routes tasks to sub-agents.

| Aspect | CEO Agent Model | Fleet Model |
|--------|----------------|-------------|
| Routing cost | $0.01-0.05 per task (LLM call) | $0 (direct command or YAML) |
| Routing reliability | Probabilistic (LLM can hallucinate) | Deterministic (user or workflow) |
| Failure mode | CEO down = fleet down | One agent down = one agent down |
| User visibility | Opaque (user talks to CEO) | Direct (user talks to workers) |
| Adding agents | Update CEO prompt + agent config | Just agent config |
| Scaling | CEO prompt grows with every agent | Each agent is independent |

Agents coordinate through four patterns:

1. **Blackboard handoff** — Agent A writes data, Agent B reads it later. No conversation.
2. **YAML workflows** — Steps run in defined order. Deterministic. No LLM routing.
3. **Broadcast** — User sends one message to all agents (`/broadcast` in REPL). Each processes independently.
4. **Pub/Sub signals** — Agent publishes an event. Subscribed agents react on next heartbeat.

---

## Architecture

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
               │  Docker Bridge Network (openlegion_internal)
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
as a single FastAPI process and provides these core services:

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
The vault loads credentials from environment variables prefixed with
`OPENLEGION_CRED_` and supports multiple providers (OpenAI, Anthropic,
Apollo, Hunter, Brave Search). The vault also enforces budget limits before
dispatching LLM calls and records token usage after each response.

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

Glob patterns (`fnmatch`) control blackboard access. Unrecognized agents
fall back to a default template, then to deny-all.

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

The orchestrator resolves step dependencies, evaluates conditions safely
(no `eval()`), handles retries and failures, and promotes results to the
blackboard after each step completes.

### Container Manager

Each agent runs in an isolated Docker container with:
- **Image**: `openlegion-agent:latest` (Python 3.12, system tools, Playwright, Chromium)
- **Network**: `openlegion_internal` bridge (can only reach mesh host)
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
(max 20 iterations) of decide → act → learn. Each iteration calls the LLM,
executes tool calls, stores learned facts, and manages the context window.
Returns a `TaskResult` with structured output and optional blackboard promotions.

### Chat Mode (`chat`)

Accepts a user message. On the first message of a session, loads workspace
context (AGENTS.md, SOUL.md, USER.md, MEMORY.md, daily logs) into the system
prompt and auto-searches workspace memory. On every message, executes tool
calls in a bounded loop (max 10 rounds), runs context compaction if needed,
and logs the turn to the daily session log.

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `exec` | Shell command execution with timeout |
| `read_file` | Read file contents from `/data` |
| `write_file` | Write/append file in `/data` |
| `list_files` | List/glob files in `/data` |
| `http_request` | HTTP GET/POST/PUT/DELETE/PATCH |
| `browser_navigate` | Open URL, extract page text |
| `browser_screenshot` | Capture page screenshot |
| `browser_click` | Click element by CSS selector |
| `browser_type` | Fill input by CSS selector |
| `browser_evaluate` | Run JavaScript in page |
| `memory_search` | BM25 search over workspace files |
| `memory_save` | Append entry to daily session log |
| `list_agents` | Discover other agents in the fleet |
| `read_shared_state` | Read from the shared blackboard |
| `write_shared_state` | Write to the shared blackboard |
| `list_shared_state` | Browse blackboard entries by prefix |
| `publish_event` | Publish event to mesh pub/sub |
| `save_artifact` | Save deliverable file and register on blackboard |
| `set_cron` | Schedule a recurring job for the agent |
| `set_heartbeat` | Enable cost-efficient autonomous monitoring with probes |
| `list_cron` | List the agent's scheduled cron jobs |
| `remove_cron` | Remove a scheduled cron job |
| `create_skill` | Write a new Python skill at runtime (self-extending) |
| `reload_skills` | Hot-reload all skills including newly created ones |
| `list_custom_skills` | List custom skills the agent has created |
| `spawn_agent` | Spawn an ephemeral sub-agent for specialized work |
| `read_agent_history` | Read another agent's conversation logs |
| `web_search` | Search the web via DuckDuckGo (no API key) |

Custom skills are Python functions decorated with `@skill`, auto-discovered
from the agent's `skills_dir` at startup. Agents can also create new skills
at runtime using `create_skill` and hot-reload them.

---

## Memory System

Four layers give agents persistent, self-improving memory:

```
Layer 4: Context Manager          ← Manages the LLM's context window
  │  Monitors token usage
  │  Auto-compacts at 70% capacity
  │  Flushes extracted facts to MEMORY.md before discarding
  │
Layer 3: Learnings                ← Self-improvement through failure tracking
  │  learnings/errors.md         (tool failures with context)
  │  learnings/corrections.md   (user corrections and preferences)
  │  Auto-injected into system prompt each session
  │  Auto-rotation when files grow large
  │
Layer 2: Workspace Files          ← Durable, human-readable storage
  │  AGENTS.md, SOUL.md, USER.md  (loaded into system prompt)
  │  MEMORY.md                    (curated long-term facts)
  │  HEARTBEAT.md                 (autonomous monitoring rules)
  │  memory/YYYY-MM-DD.md         (daily session logs)
  │  BM25 search across all files (zero dependencies, ~60 lines)
  │
Layer 1: MemoryStore              ← Structured vector database
     SQLite + sqlite-vec
     Facts with embeddings (KNN similarity search)
     Action logs with timestamps
     Salience tracking with decay
```

### Write-Then-Compact Pattern

Before the context manager discards messages, it:

1. Asks the LLM to extract important facts from the conversation
2. Appends those facts to `MEMORY.md` (durable storage)
3. Summarizes the conversation
4. Replaces message history with: summary + last 4 messages

Nothing is permanently lost during compaction.

### Cross-Session Memory

Verified by E2E tests with real Docker containers and real LLM calls:

```
Session 1: User says "My cat's name is Whiskerino"
           Agent saves to daily log via memory_save

  ═══ Chat Reset (new session) ═══

Session 2: User asks "What is my cat's name?"
           Agent recalls "Whiskerino" from workspace memory
```

---

## Triggering & Automation

Agents act autonomously through three trigger mechanisms, all running in the
mesh host (not inside containers, so they survive container restarts):

### Cron Scheduler

Persistent cron jobs that dispatch messages to agents on a schedule. Agents
can schedule their own cron jobs autonomously using the `set_cron` tool, or
you can manage them via the mesh API (`POST /mesh/cron`).

Supports 5-field cron expressions (`minute hour dom month dow`), interval
shorthand (`every 30m`, `every 2h`, `every 1d`), and state persisted to
`config/cron.json`.

### Heartbeat System

Cost-efficient autonomous monitoring. Heartbeat jobs (`set_heartbeat` tool)
run cheap deterministic probes first — disk usage, pending blackboard signals,
pending tasks — and only dispatch to the agent (costing LLM tokens) when
probes detect something actionable. Define autonomous rules in `HEARTBEAT.md`.

This 5-stage architecture (scheduler → probes → policy → escalation → action)
makes always-on agents economically viable.

### Webhook Endpoints

Named webhook URLs that dispatch payloads to agents. Managed via the mesh
API or `config/webhooks.json`:

```bash
# External services POST to webhook URLs, agent processes the payload
curl -X POST http://localhost:8420/webhook/hook/hook_a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{"event": "push", "repo": "myproject"}'
```

### File Watchers

Poll directories for new/modified files matching glob patterns. Uses polling
(not inotify) for Docker volume compatibility. Suppresses dispatch for files
that already exist on first scan.

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

When an agent exceeds its budget, the Credential Vault rejects LLM calls
with an error message instead of forwarding them to the provider.

Storage: SQLite with WAL mode. Tracks prompt tokens, completion tokens,
model, cost per call. Queryable by agent, model, and time period
(today/week/month).

---

## Security Model

Defense-in-depth with five layers:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Runtime isolation | **Docker Sandbox microVMs** (hypervisor-level, separate kernel per agent) when available; falls back to Docker containers | Agent escape, kernel exploits, cross-agent compromise |
| Container hardening | Non-root user, no-new-privileges, memory/CPU limits, filesystem isolation | Privilege escalation, resource abuse |
| Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs; configurable isolation vs. collaboration | Unauthorized data access, agent impersonation |
| Input validation | Path traversal prevention, safe condition eval (no `eval()`), token budgets, iteration limits | Injection, runaway loops, context overflow |

### Dual Runtime Backend

OpenLegion auto-detects the best isolation available:

- **Docker Sandbox (microVM)**: Each agent runs in its own virtual machine with a separate kernel. Even if an agent is compromised via internet access or code execution, the hypervisor boundary prevents host access. `docker sandbox rm` completely destroys all agent state. Stable on macOS (virtualization.framework) and Windows (Hyper-V); experimental on Linux.
- **Docker Container (fallback)**: Standard container isolation with hardening. Used when Docker Sandbox is unavailable. Shared host kernel but namespace/cgroup isolation. A startup warning is displayed when this fallback is active.

The runtime backend is detected and selected automatically at startup. The agent code is identical in both modes -- only the transport and lifecycle management differ.

---

## CLI Reference

```
openlegion
├── setup                                # Guided setup wizard (API key, project, agent, Docker)
├── start [--config PATH] [-d]           # Start runtime + interactive REPL (-d for background)
├── stop                                 # Stop all containers
├── chat <name> [--port PORT]            # Connect to a running agent (background mode)
├── status [--port PORT]                 # Show agent status
│
└── agent
    ├── add [name]                       # Add a new agent (interactive wizard)
    ├── list                             # List configured agents + status
    └── remove <name> [--yes]            # Remove an agent
```

### Interactive REPL Commands (inside `start`)

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
Created during `openlegion setup` or edit directly with any text editor.

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

### `config/mesh.yaml` — Framework Settings (tracked in git)

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

### `config/agents.yaml` — Agent Definitions (gitignored, per-project)

Created automatically by `openlegion setup` or `openlegion agent add`.
Each project has its own agents — this file is not tracked in git.

```yaml
agents:
  researcher:
    role: "research"
    model: "openai/gpt-4o-mini"
    skills_dir: "./skills/research"
    system_prompt: "You are a research specialist..."
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

### `config/cron.json` — Scheduled Jobs

Persisted cron job definitions. Agents manage their own via the `set_cron` tool.

### `config/webhooks.json` — Webhook Endpoints

Persisted webhook endpoint definitions. Managed via the mesh API.

### `.env` — API Keys and Settings

```bash
OPENLEGION_CRED_OPENAI_API_KEY=sk-...
OPENLEGION_CRED_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...

# Log format: "json" (default, structured) or "text" (human-readable)
OPENLEGION_LOG_FORMAT=text
```

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
| Types & Models | 15 | Pydantic validation, serialization |
| Memory Store | 22 | SQLite ops, vector search, salience decay |
| Workspace | 18 | File scaffold, loading, BM25 search, daily logs |
| Context Manager | 8 | Token estimation, compaction, flushing, pruning |
| Memory Tools | 6 | memory_search, memory_save, discovery |
| Skills | 15 | Discovery, execution, exec/file/http tools |
| Agent Loop | 12 | Task execution, tool calling, cancellation |
| Chat | 20 | Chat mode, workspace integration, cross-session |
| Mesh | 18 | Blackboard, PubSub, MessageRouter, permissions |
| Orchestrator | 10 | Workflows, conditions, retries, failures |
| Credentials | 8 | Vault, API proxy, provider detection |
| Integration | 8 | Multi-component mesh operations |
| CLI | 6 | Agent add/list/remove, chat, setup |
| Cron | 18 | Cron expressions, intervals, dispatch, persistence |
| Cost Tracking | 10 | Usage recording, budgets, vault integration |
| Transport | 15 | HttpTransport, SandboxTransport, resolve_url |
| Runtime Backend | 18 | DockerBackend, SandboxBackend, detection, selection |
| Channels | 21 | Base channel, commands, per-user routing, chunking |
| Webhooks | 7 | Add/remove, persistence, dispatch |
| File Watchers | 7 | Polling, dispatch, pattern matching |
| E2E: Workflow | 4 | Container health, registration, full workflow |
| E2E: Chat | 5 | Chat, exec tool, file tools, reset |
| E2E: Memory | 4 | Cross-session recall, workspace, search |
| E2E: Triggering | 4 | Cron dispatch, webhook dispatch, cost tracking |
| **Total** | **318** | |

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
| playwright | Browser automation (in container only) |

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
│   ├── memory.py                       # Vector memory store (SQLite + sqlite-vec)
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
│       ├── memory_tool.py              # Memory search/save
│       └── mesh_tool.py               # Shared state, fleet awareness, artifacts
├── host/
│   ├── mesh.py                         # Blackboard, PubSub, MessageRouter
│   ├── orchestrator.py                 # Workflow DAG executor
│   ├── permissions.py                  # Permission matrix
│   ├── credentials.py                  # Credential vault + API proxy + cost integration
│   ├── runtime.py                      # RuntimeBackend ABC + DockerBackend + SandboxBackend
│   ├── transport.py                    # Transport ABC + HttpTransport + SandboxTransport
│   ├── containers.py                   # Backward-compat alias (ContainerManager = DockerBackend)
│   ├── server.py                       # Mesh FastAPI server
│   ├── cron.py                         # Cron scheduler
│   ├── webhooks.py                     # Named webhook endpoints
│   ├── watchers.py                     # File watchers (polling)
│   ├── costs.py                        # Cost tracking + budgets (SQLite)
│   ├── health.py                       # Health monitor + auto-restart
│   └── lanes.py                        # Per-agent task queues (serial execution)
├── shared/
│   ├── types.py                        # All Pydantic models
│   └── utils.py                        # ID generation, logging, OPENLEGION_LOG_FORMAT
├── channels/
│   └── webhook.py                      # Workflow trigger webhook adapter
└── templates/
    ├── starter.yaml                    # Single-agent template
    ├── sales.yaml                      # Sales pipeline team template
    ├── devteam.yaml                    # Dev team template
    └── content.yaml                    # Content creation team template

config/
├── mesh.yaml                           # Framework settings (tracked)
├── agents.yaml                         # Agent definitions (gitignored, per-project)
├── permissions.json                    # Per-agent ACLs
└── workflows/                          # Workflow YAML definitions

skills/                                 # Example custom skills (API proxy stubs)
```

---

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Messages, not method calls | Agents communicate through HTTP/JSON. Never shared memory or direct invocation. |
| The mesh is the only door | No agent has network access except through the mesh. No agent holds credentials. |
| Private by default, shared by promotion | Agents keep knowledge private. Relevant facts are explicitly promoted to the blackboard. |
| Capabilities, not identities | Workflows address capabilities ("whoever can score leads"), enabling hot-swap. |
| Explicit failure handling | Every workflow step declares what happens on failure. No silent error swallowing. |
| Small enough to audit | No single module exceeds ~800 lines. The entire codebase is auditable in a day. |
| Skills over features | New capabilities are agent skills — never mesh/orchestrator code growth. |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dual runtime (sandbox/container) | MicroVM isolation when available (compromised agent = delete and forget). Container fallback on Linux. Agent code unchanged. |
| Credentials on mesh only | Agents can't leak keys. Single point of credential management and rotation. |
| Markdown workspace files | Human-readable, git-versionable, editable by hand. No proprietary format lock-in. |
| BM25 from scratch | Zero dependencies. Works immediately. ~60 lines. Upgradeable to hybrid vector search later. |
| Write-then-compact | Flush facts to MEMORY.md before discarding messages. No information loss during context management. |
| SQLite for all state | Single-file database. No external services needed. WAL mode for concurrent reads. |
| LiteLLM for LLM calls | Supports 100+ providers with a single interface. No vendor lock-in. |
| No web UI | CLI and messaging channels. A dashboard doesn't make agents smarter. |

---

## Roadmap

| Phase | Status | What |
|-------|--------|------|
| Phase 0: Core Runtime | **Complete** | Agent loop, mesh, orchestrator, containers, tools, CLI |
| Phase 1: Memory + Context | **Complete** | Workspace files, BM25 search, context compaction, cross-session memory |
| Phase 2: Triggering + Cost Tracking | **Complete** | Cron scheduler, webhooks, file watchers, cost tracking, budget enforcement |
| Phase 3: Fleet Coordination | **Complete** | Shared state tools, goal awareness, team templates, delivery tools, PROJECT.md alignment |
| Phase 4: Production Hardening | **Complete** | Graceful shutdown, health monitor + auto-restart, lane queues, structured logging |
| Phase 5: Fleet Model | **Complete** | Removed CEO/delegation pattern, broadcast, PROJECT.md shared context, simplified CLI (9 commands) |
| Phase 6: Self-Improving Agents | **Complete** | Failure tracking, correction learning, learnings injection, heartbeat with deterministic probes |
| Phase 7: Agent Autonomy | **Complete** | Skill authoring (agents write their own tools), hot-reload, dynamic agent spawning, inter-agent history |
| Phase 8: Messaging Channels | **Complete** | Telegram + Discord channel adapters, @-mention routing, platform-specific chunking |

---

## License

See [LICENSE](LICENSE) for details.
