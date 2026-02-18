# OpenLegion

A container-isolated, memory-aware multi-agent runtime in Python.

OpenLegion orchestrates LLM-powered agents that run in isolated Docker containers,
coordinate through a central mesh host, persist memory across sessions, and act
autonomously via scheduled triggers. It is built for production workloads where
multiple agents collaborate on multi-step tasks — such as turning a CSV of companies
into a pipeline of qualified leads — without human intervention.

**237 tests passing** (220 unit/integration + 17 end-to-end) across **~5,400 lines**
of application code. Zero framework dependencies. Fully auditable.

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
git clone <repo-url> && cd openlegion
pip install -e ".[dev]"

# 2. Set your API key
openlegion config set-key openai sk-your-key-here

# 3. One-command setup: creates agent, builds Docker image, starts chatting
openlegion quickstart
```

Or step by step:

```bash
# Create an agent
openlegion agent create researcher

# Build the Docker image (first time only)
docker build -t openlegion-agent:latest -f Dockerfile.agent .

# Start chatting
openlegion agent chat researcher
```

### Requirements

- Python 3.12+
- Docker Engine running and accessible
- An LLM API key (OpenAI, Anthropic, or Groq)

---

## What It Does

1. **Runs LLM agents in isolated Docker containers** — each agent has its own
   filesystem, memory database, resource limits, and security boundary.

2. **Coordinates multi-agent workflows** — a shared blackboard, pub/sub messaging,
   and DAG-based workflow orchestration are built in from day one.

3. **Provides built-in tools** — shell execution, file I/O, HTTP requests, browser
   automation (Playwright/Chromium), and persistent memory.

4. **Persists memory across sessions** — workspace files, BM25 search, context
   compaction, and daily session logs ensure agents remember.

5. **Acts autonomously** — cron schedules, webhook triggers, and file watchers
   let agents work without being prompted.

6. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly
   budget enforcement.

7. **Enforces permissions** — per-agent ACLs for messaging, blackboard access,
   pub/sub topics, and API access. Default deny.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           User Interface                                │
│                                                                         │
│   CLI (click)          Webhooks            Cron Scheduler               │
│   - quickstart         - POST /webhook/    - "0 9 * * 1-5"             │
│   - agent chat           hook/{id}         - "every 30m"               │
│   - start/stop         - Trigger agents    - Heartbeat pattern          │
│   - cron/webhook/costs                                                  │
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

Custom skills are Python functions decorated with `@skill`, auto-discovered
from the agent's `skills_dir` at startup.

---

## Memory System

Three layers give agents persistent memory across sessions:

```
Layer 3: Context Manager          ← Manages the LLM's context window
  │  Monitors token usage
  │  Auto-compacts at 70% capacity
  │  Flushes extracted facts to MEMORY.md before discarding
  │
Layer 2: Workspace Files          ← Durable, human-readable storage
  │  AGENTS.md, SOUL.md, USER.md  (loaded into system prompt)
  │  MEMORY.md                    (curated long-term facts)
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

Persistent cron jobs that dispatch messages to agents on a schedule.

```bash
# Standard cron expression
openlegion cron add researcher -s "0 9 * * 1-5" -m "Morning research check"

# Interval shorthand
openlegion cron add researcher -s "every 2h" -m "Check HEARTBEAT.md and execute pending tasks"
```

Supports 5-field cron expressions (`minute hour dom month dow`), interval
shorthand (`every 30m`, `every 2h`, `every 1d`), and the heartbeat pattern
where an agent checks a `HEARTBEAT.md` file and `OK`-only responses are
suppressed. State is persisted to `config/cron.json`.

### Webhook Endpoints

Named webhook URLs that dispatch payloads to agents:

```bash
openlegion webhook add --agent researcher --name github-push
# Output: URL: http://localhost:8420/webhook/hook/hook_a1b2c3d4

# External services POST to it, agent processes the payload
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
prevent runaway spend.

```bash
# View spend
openlegion costs                                  # All agents, today
openlegion costs --agent researcher --period month # One agent, monthly

# Budget config in mesh.yaml
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

Defense-in-depth with four layers:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Container isolation | Docker filesystem, network, resources, no-new-privileges, non-root | Agent escape, resource abuse, privilege escalation |
| Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs | Unauthorized data access, agent impersonation |
| Input validation | Path traversal prevention, safe condition eval (no `eval()`), token budgets, iteration limits | Injection, runaway loops, context overflow |

---

## CLI Reference

```
openlegion
├── quickstart                        # One-command setup wizard
├── start [--config PATH]             # Start mesh + all agents + cron
├── stop                              # Stop all containers
├── trigger <workflow> <payload>      # Trigger workflow
├── status [--port PORT]              # Agent status
├── agent
│   ├── create <name>                 # Interactive agent creation
│   ├── chat <name> [--port]          # Interactive REPL
│   └── list                          # List agents + status
├── config
│   └── set-key <provider> <key>      # Save API key to .env
├── cron
│   ├── add <agent> -s <schedule> -m <message>  # Add scheduled job
│   ├── list                          # List all jobs
│   ├── run <job_id>                  # Manual trigger
│   ├── pause <job_id>                # Pause a job
│   ├── resume <job_id>               # Resume a paused job
│   └── remove <job_id>              # Delete a job
├── webhook
│   ├── add --agent <name> --name <hook-name>   # Create webhook
│   ├── list                          # List all webhooks
│   ├── test <hook_id> [-p <payload>] # Send test payload
│   └── remove <hook_id>             # Delete webhook
└── costs
    [--agent <name>] [--period today|week|month]  # View LLM spend
```

Chat REPL commands: `/reset`, `/status`, `/quit`, `/help`.

---

## Configuration

### `config/mesh.yaml` — System Configuration

```yaml
mesh:
  host: "0.0.0.0"
  port: 8420

llm:
  default_model: "openai/gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  max_tokens: 4096
  temperature: 0.7

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

Persisted cron job definitions. Managed via CLI (`openlegion cron`).

### `config/webhooks.json` — Webhook Endpoints

Persisted webhook endpoint definitions. Managed via CLI (`openlegion webhook`).

### `.env` — API Keys

```bash
OPENLEGION_CRED_OPENAI_API_KEY=sk-...
OPENLEGION_CRED_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...
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
| CLI | 6 | Agent create/list, config commands |
| Cron | 18 | Cron expressions, intervals, dispatch, persistence |
| Cost Tracking | 10 | Usage recording, budgets, vault integration |
| Webhooks | 7 | Add/remove, persistence, dispatch |
| File Watchers | 7 | Polling, dispatch, pattern matching |
| E2E: Workflow | 4 | Container health, registration, full workflow |
| E2E: Chat | 5 | Chat, exec tool, file tools, reset |
| E2E: Memory | 4 | Cross-session recall, workspace, search |
| E2E: Triggering | 4 | Cron dispatch, webhook dispatch, cost tracking |
| **Total** | **237** | |

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
├── cli.py                              # CLI entry point (730 lines)
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
│       └── memory_tool.py              # Memory search/save
├── host/
│   ├── mesh.py                         # Blackboard, PubSub, MessageRouter
│   ├── orchestrator.py                 # Workflow DAG executor
│   ├── permissions.py                  # Permission matrix
│   ├── credentials.py                  # Credential vault + API proxy + cost integration
│   ├── containers.py                   # Docker container manager
│   ├── server.py                       # Mesh FastAPI server
│   ├── cron.py                         # Cron scheduler
│   ├── webhooks.py                     # Named webhook endpoints
│   ├── watchers.py                     # File watchers (polling)
│   └── costs.py                        # Cost tracking + budgets (SQLite)
├── shared/
│   ├── types.py                        # All Pydantic models
│   └── utils.py                        # ID generation, logging
└── channels/
    └── webhook.py                      # Workflow trigger webhook adapter

config/
├── mesh.yaml                           # System config (agents, models, ports)
├── permissions.json                    # Per-agent ACLs
├── cron.json                           # Scheduled jobs
├── webhooks.json                       # Webhook endpoints
└── workflows/                          # Workflow YAML definitions

tests/                                  # 25 test files, 237 tests
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
| Agents in Docker containers | True isolation. One agent crash doesn't affect others. Resource limits enforced by the kernel. |
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
| Phase 3: Multi-Agent Pipelines | Planned | Agent-to-agent delegation, team templates, delivery tools, pipeline workflows |
| Phase 4: Production Hardening | Planned | Lane queues (serial execution), health monitor, auto-restart, structured logging, graceful shutdown |
| Phase 5: Messaging Channels | Planned | Telegram bot, channel abstraction, per-user session management |

---

## License

See [LICENSE](LICENSE) for details.
