# OpenLegion

An open-source platform for running autonomous LLM agents in isolated Docker containers.

Each agent gets its own memory, tools, schedule, and budget. You talk to agents directly
through a CLI, Telegram, or Discord. Agents coordinate through shared state and YAML
workflows — no routing layer, no framework overhead.

**424 tests passing** across **~11,000 lines** of application code.
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

`setup` walks you through everything once. `start` launches your agents and drops you into an interactive chat.

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

### Requirements

- Python 3.12+
- Docker Engine running and accessible
- An LLM API key (OpenAI, Anthropic, or Groq)

---

## What It Does

1. **Runs agents in isolated Docker containers** — each agent has its own filesystem, memory database, resource limits, and security boundary.

2. **Provides 25+ built-in tools** — shell execution, file I/O, HTTP requests, browser automation (Playwright/Chromium), web search, structured memory, and fleet coordination.

3. **Remembers across sessions** — hierarchical memory with vector search, BM25 keyword search, auto-categorization, and workspace files. Facts survive resets and restarts.

4. **Acts autonomously** — cron schedules, heartbeat probes, webhook triggers, and file watchers let agents work without being prompted.

5. **Self-improves** — agents learn from tool failures and user corrections, injecting past learnings into future sessions.

6. **Self-extends** — agents write their own Python skills at runtime and hot-reload them. Agents can also spawn sub-agents for specialized work.

7. **Multi-channel** — connect agents to Telegram and Discord. Also accessible via CLI and API.

8. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly budget enforcement.

9. **Runs deterministic workflows** — YAML-defined DAG workflows chain agents in sequence with conditions, retries, and failure handlers.

10. **Fails over across providers** — configurable model failover chains cascade across LLM providers with per-model health tracking and exponential cooldown.

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
| `browser_navigate` | Open URL, extract page text |
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
| `reload_skills` | Hot-reload all skills |
| `spawn_agent` | Spawn an ephemeral sub-agent |
| `read_agent_history` | Read another agent's conversation logs |

Custom skills are Python functions decorated with `@skill`, auto-discovered
from the agent's `skills_dir` at startup. Agents can also create new skills
at runtime and hot-reload them.

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

OpenLegion auto-detects the best isolation available:

- **Docker Sandbox (microVM)**: Each agent runs in its own virtual machine with a separate kernel. Even if compromised, the hypervisor boundary prevents host access.
- **Docker Container (fallback)**: Standard container isolation with hardening. Used when Docker Sandbox is unavailable. A startup warning is displayed when this fallback is active.

---

## CLI Reference

```
openlegion
├── setup                                # Guided setup wizard
├── start [--config PATH] [-d]           # Start runtime + interactive REPL
├── stop                                 # Stop all containers
├── chat <name> [--port PORT]            # Connect to a running agent
├── status [--port PORT]                 # Show agent status
│
└── agent
    ├── add [name]                       # Add a new agent
    ├── list                             # List configured agents
    └── remove <name> [--yes]            # Remove an agent
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

```bash
OPENLEGION_CRED_OPENAI_API_KEY=sk-...
OPENLEGION_CRED_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...

# Log format: "json" (default) or "text" (human-readable)
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
| Built-in Tools | 48 | exec, file, browser, memory, mesh tools, discovery |
| Workspace | 41 | File scaffold, loading, BM25 search, daily logs, learnings |
| Mesh | 28 | Blackboard, PubSub, MessageRouter, permissions |
| Memory Store | 26 | SQLite ops, vector search, categories, hierarchical search, salience decay |
| Orchestrator | 25 | Workflows, conditions, retries, failures |
| Cron | 25 | Cron expressions, intervals, dispatch, persistence |
| Channels | 21 | Base channel, commands, per-user routing, chunking |
| Transport | 18 | HttpTransport, SandboxTransport, resolve_url |
| Runtime Backend | 18 | DockerBackend, SandboxBackend, detection, selection |
| Context Manager | 17 | Token estimation, compaction, flushing, pruning |
| Agent Loop | 15 | Task execution, tool calling, cancellation |
| Failover | 15 | Health tracking, chain cascade, cooldown |
| Skills | 14 | Discovery, execution, injection |
| Integration | 13 | Multi-component mesh operations |
| Credentials | 13 | Vault, API proxy, provider detection |
| Chat | 18 | Chat mode, workspace integration, cross-session |
| Costs | 10 | Usage recording, budgets, vault integration |
| Types | 8 | Pydantic validation, serialization |
| CLI | 8 | Agent add/list/remove, chat, setup |
| Webhooks | 7 | Add/remove, persistence, dispatch |
| File Watchers | 7 | Polling, dispatch, pattern matching |
| Memory Tools | 6 | memory_search, memory_save, memory_recall |
| Memory Integration | 6 | Vector search, cross-task recall, salience |
| E2E | 17 | Container health, workflow, chat, memory, triggering |
| **Total** | **424** | |

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
| Small enough to audit | No module exceeds ~800 lines. The entire codebase is auditable in a day. |
| Skills over features | New capabilities are agent skills, not mesh or orchestrator code. |
| SQLite for all state | Single-file databases. No external services. WAL mode for concurrent reads. |
| Zero vendor lock-in | LiteLLM supports 100+ providers. Markdown workspace files. No proprietary formats. |

---

## License

See [LICENSE](LICENSE) for details.
