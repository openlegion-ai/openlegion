# Architecture

OpenLegion is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers (or Sandbox microVMs), coordinated through a central mesh host.

## System Overview

```
User (CLI REPL / Telegram / Discord / Webhook)
  -> Mesh Host (FastAPI :8420) -- routes messages, enforces permissions, proxies APIs
    -> Agent Containers (FastAPI :8400 each) -- isolated execution with private memory
```

## Trust Zones

Three trust zones govern all inter-component communication:

| Level | Zone | Description |
|-------|------|-------------|
| 0 | **Untrusted** | External input (webhooks, user prompts). Sanitized before reaching agents. |
| 1 | **Sandboxed** | Agent containers. Isolated filesystem, no external network, no credentials. |
| 2 | **Trusted** | Mesh host. Holds credentials, manages containers, routes messages. |

Everything between zones is HTTP + JSON with Pydantic contracts defined in `src/shared/types.py`.

## Fleet Model

OpenLegion uses a **fleet model, not hierarchy**. There is no CEO agent that routes or delegates. Users talk to agents directly. Agents coordinate through:

- **Blackboard** -- shared key-value state store (SQLite WAL)
- **PubSub** -- event-driven notifications between agents
- **Workflows** -- deterministic YAML-defined DAG execution

## Component Map

### Mesh Host (`src/host/`)

The mesh host runs on the user's machine as a single FastAPI process. It is the trusted coordinator.

| Module | Responsibility |
|--------|---------------|
| `mesh.py` | Blackboard (shared state), PubSub, MessageRouter |
| `server.py` | FastAPI app factory with all mesh HTTP endpoints |
| `orchestrator.py` | DAG workflow executor with safe condition evaluation |
| `runtime.py` | RuntimeBackend ABC with Docker and Sandbox implementations |
| `transport.py` | Transport ABC with HTTP and Sandbox implementations |
| `credentials.py` | Credential vault and LLM API proxy |
| `permissions.py` | Per-agent ACL enforcement with glob patterns |
| `health.py` | Health monitor with auto-restart policy |
| `costs.py` | Per-agent LLM cost tracking and budget enforcement |
| `cron.py` | Cron scheduler with heartbeat support |
| `lanes.py` | Per-agent FIFO task queues |
| `failover.py` | Model health tracking and failover chains |
| `webhooks.py` | Named webhook endpoints |
| `watchers.py` | File watcher with polling |

### Agent (`src/agent/`)

Each agent runs in an isolated Docker container with its own FastAPI server.

| Module | Responsibility |
|--------|---------------|
| `__main__.py` | Container entry point; reads env config, wires components, starts server |
| `loop.py` | Bounded execution loop: task mode (20 iterations) and chat mode |
| `skills.py` | Skill discovery and registry; `@skill` decorator system |
| `mcp_client.py` | MCP server lifecycle management and tool routing |
| `memory.py` | SQLite + sqlite-vec + FTS5 hierarchical memory store |
| `workspace.py` | Persistent markdown workspace (MEMORY.md, daily logs, learnings) |
| `context.py` | Context window management with write-then-compact pattern |
| `llm.py` | LLM client that routes all calls through mesh API proxy |
| `mesh_client.py` | HTTP client for agent-to-mesh communication |
| `server.py` | Agent-side FastAPI server (/task, /chat, /status) |

### Built-in Tools (`src/agent/builtins/`)

| Module | Tools Provided |
|--------|---------------|
| `exec_tool.py` | Shell command execution |
| `file_tool.py` | File read/write/list scoped to /data |
| `http_tool.py` | HTTP requests |
| `browser_tool.py` | 3-tier browser: Playwright Chromium, Camoufox stealth, Bright Data CDP |
| `mesh_tool.py` | Blackboard, PubSub, fleet awareness, artifacts, cron, heartbeat, spawn |
| `memory_tool.py` | Memory search, save, recall |
| `vault_tool.py` | Credential-blind vault operations |
| `skill_tool.py` | Runtime skill creation and hot-reload |
| `web_search_tool.py` | DuckDuckGo web search (no API key) |

### Channels (`src/channels/`)

| Module | Platform |
|--------|----------|
| `base.py` | Abstract base with @mention routing, /commands, message chunking |
| `telegram.py` | Telegram Bot API adapter |
| `discord.py` | Discord Bot adapter |
| `slack.py` | Slack adapter (Socket Mode via slack-bolt) |
| `whatsapp.py` | WhatsApp Cloud API adapter |
| `webhook.py` | Generic webhook-to-workflow adapter |

### Shared (`src/shared/`)

| Module | Purpose |
|--------|---------|
| `types.py` | **THE contract** -- all Pydantic models shared between host and agents |
| `utils.py` | ID generation, structured logging setup |

## Data Flow

### Task Execution

```
User -> CLI/Channel -> Mesh Router -> Agent /task endpoint
  Agent: load context -> LLM call (via mesh proxy) -> tool execution -> iterate
  Agent -> POST result to mesh -> Mesh Router -> User
```

### Chat Mode

```
User -> CLI/Channel -> Mesh Router -> Agent /chat endpoint
  Agent: load workspace context + memory search -> LLM call -> tool rounds
  Agent -> streaming/complete response -> User
```

### Workflow Execution

```
PubSub event / Cron trigger -> Orchestrator
  Orchestrator: parse DAG -> topological sort -> execute steps
  Each step: assign task to agent via /task -> wait for result
  Step result feeds into next step's input
```

## Runtime Backends

| Backend | Isolation Level | Requirements |
|---------|----------------|-------------|
| `DockerBackend` | Container (shared kernel) | Any Docker install |
| `SandboxBackend` | MicroVM (own kernel) | Docker Desktop 4.58+ |

Both implement `RuntimeBackend` ABC so the rest of the system is isolation-agnostic. The backend is selected at startup via `--sandbox` flag.

## Key Invariants

1. **Agents never hold API keys** -- all calls route through the mesh credential vault
2. **No `eval()` on untrusted input** -- workflow conditions use a regex-based safe parser
3. **Permission checks before every cross-boundary operation** -- default deny
4. **Path traversal protection** -- agent file operations confined to `/data`
5. **Bounded execution** -- 20 iterations for tasks, 30 tool rounds for chat
6. **Write-then-compact** -- facts are flushed to memory before discarding context
7. **Tool-call message grouping** -- assistant(tool_calls) and tool(results) are never separated in context trimming
