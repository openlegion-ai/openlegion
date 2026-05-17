<p align="center">
  <img width="450" alt="openlegion-logo-new" src="https://github.com/user-attachments/assets/08912b04-8df1-4473-b679-6bbac0c3ae2f" />
</p>
<h3 align="center">
  <b>The AI agent framework built for builders who can't afford a security incident.</b>
</h3>
<div align="center">
   
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests: 5800+](https://img.shields.io/badge/tests-5800%2B%20passing-brightgreen)](https://github.com/openlegion-ai/openlegion/actions/workflows/test.yml)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/mXNkjpDvvr)
[![Twitter](https://img.shields.io/badge/Twitter-@openlegion-1DA1F2?logo=x&logoColor=white)](https://x.com/openlegion)
[![LiteLLM](https://img.shields.io/badge/LLM-100%2B%20providers-orange.svg)](https://litellm.ai)
[![Docker](https://img.shields.io/badge/isolation-Docker%20%2B%20Sandbox-blue.svg)]()
   
</div>

> **Autonomous AI agent fleets — isolated, auditable, and production-ready.**
> Every agent runs in its own Docker container. API keys never leave the vault.
> Chat via Telegram, Discord, Slack, or WhatsApp. Built-in cost controls. 100+ LLM providers.

[Quick Start](#quick-start) · [Full Setup Guide](QUICKSTART.md) · [Why Not OpenClaw?](#why-not-openclaw) · [Docs](docs/)

---

## Demo

https://github.com/user-attachments/assets/8bd3fe95-5734-474d-92f0-40616daf91ad

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

**Requirements:** Python 3.10+, Docker (running), at least one LLM provider key (Anthropic, OpenAI, Gemini, Moonshot, Deepseek, xAI, Groq, Minimax, Zai, or Ollama — the setup wizard walks you through it; existing Anthropic Claude CLI or OpenAI Codex CLI logins can be imported).

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

> **Windows note:** Docker Desktop (not Docker Engine) is required on Windows. WSL2 must be enabled. See Docker's [WSL2 backend guide](https://docs.docker.com/desktop/wsl/) if containers fail to start.

> First install downloads dependencies into a venv; this may take several minutes the first time. Subsequent installs are fast.
>
> **First run:** On the very first `openlegion start`, Docker builds the `openlegion-agent:latest` and `openlegion-browser:latest` images from the `Dockerfile.agent` and `Dockerfile.browser` in the repo root. The browser image is significantly larger (Camoufox + KasmVNC + Openbox + Xvnc) and can take several minutes with no progress output — this is normal. Subsequent starts are fast.
>
> **Background mode:** `openlegion start -d` polls for startup for up to 90 seconds. If a Docker image build is needed on first run, this timeout may be exceeded — wait for the build to finish and re-run `openlegion start -d`.
>
> **First run also creates:** `config/agents.yaml`, `config/permissions.json`, `config/mesh.yaml`, agent volumes, and an `operator` agent that you didn't define — that's a built-in fleet-management agent (lighter resource caps, excluded from cost/quota math). See [CLI Reference](#cli-reference) for `openlegion reset` if you want to wipe state and start over.
>
> **Need help?** See the **[full setup guide](QUICKSTART.md)** for platform-specific instructions and troubleshooting.

### Common commands

```bash
# Start (interactive REPL); use /add inside the REPL to register more agents
openlegion start

# Run in background
openlegion start -d
openlegion chat <agent_name>   # connect from another terminal to an agent you created
openlegion stop                # clean shutdown
openlegion reset               # destructive: wipe config/, data/, skills/* (keeps .env)
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
| **Multi-agent routing** | LLM CEO agent | Fleet model — blackboard + pub/sub coordination |
| **LLM providers** | Broad | 100+ via LiteLLM with health-tracked failover |
| **Test coverage** | Minimal | 5800+ tests across 155 test files including full Docker E2E |
| **Codebase size** | 430,000+ lines | ~77,000 lines in `src/` — still auditable in a day |

---

## What It Does

OpenLegion is an **autonomous AI agent framework** for running multi-agent
fleets in isolated Docker containers. Each agent gets its own memory, tools, schedule,
and budget — coordinated through blackboard shared state and pub/sub events with no LLM routing layer.

Chat with your agent fleet via **Telegram**, **Discord**, **Slack**, **WhatsApp**, or CLI. Agents act autonomously
via cron schedules, webhooks, and heartbeat monitoring — without being
prompted.

**5800+ tests passing** across 155 test files.
**Fully auditable in a day.**
No LangChain. No Redis. No Kubernetes. No CEO agent. BSL License.

1. **Security by architecture** — every agent runs in an isolated Docker container
   (microVM when available). API keys live in the credential vault — agents call
   through a proxy and never handle credentials directly. Defense-in-depth with
   6 security layers.

2. **Production-grade cost control** — per-agent LLM token tracking with enforced
   daily and monthly budget caps at the vault layer. Agents physically cannot spend
   what you haven't authorized. View live spend with `/costs` in the REPL.

3. **Acts autonomously** — cron schedules, heartbeat probes, and webhook triggers let agents work without being prompted.

4. **Self-aware and self-improving** — agents understand their own permissions, budget, fleet topology, and system architecture via auto-generated `SYSTEM.md` and live runtime context. They learn from tool failures and user corrections, injecting past learnings into future sessions.

5. **Self-extends** — agents write their own Python skills at runtime and hot-reload them. Agents can also spawn sub-agents for specialized work.

6. **Multi-channel** — connect agents to Telegram, Discord, Slack, and WhatsApp. Also accessible via CLI and API.

7. **Real-time dashboard** — web-based fleet observability with consolidated navigation, slide-over chat panels, keyboard command palette, grouped request traces, live event streaming, streaming broadcast with real-time per-agent responses, LLM prompt/response previews, agent management, agent settings editor (personality, instructions, preferences, heartbeat rules, memory, activity logs, learnings), cost charts, cron management, and embedded KasmVNC viewer for persistent browser agents.

8. **Tracks and caps spend** — per-agent LLM cost tracking with daily and monthly budget enforcement.

9. **Fails over across providers** — configurable model failover chains cascade across LLM providers with per-model health tracking and exponential cooldown.

10. **Token-level streaming** — real-time token-by-token LLM responses across CLI, dashboard, Telegram, Discord, and Slack with progressive message editing and graceful non-streaming fallback.

---

## Architecture

OpenLegion's architecture separates concerns across three trust zones:
untrusted external input, sandboxed agent containers, and a trusted mesh host
that holds credentials and coordinates the fleet. All inter-agent communication
flows through the mesh. Agents do not contact each other directly — no direct peer-to-peer
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
│  │ Key-value, │ │ subs,   │ │ Permission │ │ LLM, image_gen,        │   │
│  │ versioned, │ │ notify  │ │ enforced   │ │ Apollo, Hunter,        │   │
│  │ TTL, GC    │ │         │ │ routing    │ │ Brave Search           │   │
│  └────────────┘ └─────────┘ └───────────┘ └────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
│  │  Permission  │ │  Container   │ │    Cost      │                    │
│  │  Matrix      │ │  Manager     │ │   Tracker    │                    │
│  │              │ │              │ │              │                    │
│  │ Per-agent    │ │ Docker life- │ │ Per-agent    │                    │
│  │ ACLs, globs, │ │ cycle, nets, │ │ token/cost,  │                    │
│  │ default deny │ │ volumes      │ │ budgets      │                    │
│  └──────────────┘ └──────────────┘ └──────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
               │
               │  Docker Network (bridge by default; host opt-in via
               │  OPENLEGION_HOST_NETWORK=1 / _BROWSER_ALLOW_HOST_NETWORK=1)
               │
     ┌─────────┼──────────┬──────────────────────┐
     ▼         ▼          ▼                      ▼
┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐
│ Agent A │ │ Agent B │ │ Agent C │  ...  │ Agent N │
│ :8400   │ │ :8400   │ │ :8400   │       │ :8400   │
└─────────┘ └─────────┘ └─────────┘       └─────────┘
  Each agent: isolated Docker container, own /data volume,
  own memory DB, own workspace, 384MB RAM, 0.15 CPU. FastAPI
  listens on :8400 *inside* the container; host port is allocated
  dynamically by the runtime. The built-in `operator` agent runs
  with lighter caps (128MB / 0.05 CPU).
```

### Trust Zones

| Level | Zone | Description |
|-------|------|-------------|
| 0 | Untrusted | External input (webhooks, user prompts). Sanitized before reaching agents. |
| 1 | Sandboxed | Agent containers. Isolated filesystem, no credentials. External network access gated through SSRF-protected mesh proxy — restricted Docker bridge with NAT egress; private/CGNAT/IPv4-mapped/6to4/Teredo ranges blocked by `http_tool.py`. The shared browser container has its own iptables egress filter (set up by entrypoint with `NET_ADMIN`, then dropped) — that is the authoritative SSRF control for browser-initiated traffic. |
| 2 | Trusted | Mesh host. Holds credentials, manages containers, routes messages. |

---

## Mesh Host

The mesh host is the central coordination layer. It runs on the host machine
as a single FastAPI process.

### Blackboard (Shared State Store)

SQLite-backed key-value store with versioning, TTL, and garbage collection.
Team agents' blackboard access is automatically scoped to `projects/{name}/*` —
agents use natural keys (e.g. `tasks/research_abc123`) while the MeshClient
transparently namespaces them under the team. Solo agents have no
blackboard access.

The on-disk prefix is `projects/{name}/*` — that's a backend storage
namespace, not a domain term, and renaming it is intentionally out of
scope for the project→team rename (the change would invalidate every
existing blackboard write).

| Namespace | Purpose | Example |
|-----------|---------|---------|
| `tasks/*` | Task assignments | `tasks/research_abc123` |
| `context/*` | Shared agent context | `context/prospect_acme` |
| `signals/*` | Inter-agent signals | `signals/research_complete` |
| `history/*` | Append-only audit log | `history/action_xyz` |

These prefixes are conventions, not enforced schemas — agents can write any key that matches their `blackboard_write` glob.

### Credential Vault (API Proxy)

Agents never hold API keys. All external API calls route through the mesh.
The vault uses a two-tier prefix system: `OPENLEGION_SYSTEM_*` for LLM
provider keys (never agent-accessible) and `OPENLEGION_CRED_*` for agent-tier
tool/service keys. Budget limits are enforced before dispatching LLM calls
and token usage is recorded after each response.

### Model Failover

Configurable failover chains cascade across LLM providers transparently.
`ModelHealthTracker` applies exponential cooldown per model (transient errors:
60s → 300s → 1500s, billing/auth errors: 1h). Streaming failover is supported — if streaming fails mid-response (including empty/zero-length responses that indicate upstream provider failure),
the next model in the chain retries the full request from the start.

### Permission Matrix

Every inter-agent operation is checked against per-agent ACLs. The shape — agents call the blackboard with **natural keys** (e.g. `read_blackboard("tasks/foo")`) and `MeshClient` transparently namespaces them under the active team, so the patterns below are matched against the resolved key (`projects/myteam/tasks/foo`):

```json
{
  "researcher": {
    "can_message": ["*"],
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"],
    "blackboard_read": ["projects/myproject/*"],
    "blackboard_write": ["projects/myproject/*"],
    "allowed_apis": ["llm", "brave_search"],
    "allowed_credentials": ["brightdata_*"],
    "browser_actions": null
  }
}
```

Matching is **exact match (or `*`)** for `can_message`, `can_publish`, and `can_subscribe`, and **glob (`fnmatch`)** for `blackboard_read`, `blackboard_write`, and `allowed_credentials`.

`browser_actions` semantics: `null` (default) = all known actions allowed; `["*"]` = explicit allow-all; specific list (e.g. `["browser_navigate", "browser_screenshot"]`) = narrow allowlist; `[]` = deny all browser use even when `can_use_browser` is true.

Blackboard patterns use the `projects/{name}/*` namespace. When an agent joins a
team, it receives read/write access to that namespace. Solo agents get
empty blackboard permissions.

### Container Manager

Agent containers are slim — no browser. Browsing is handled by a shared browser service container (Camoufox + KasmVNC).

**Agent container:**
- **Image**: `openlegion-agent:latest` (Python 3.12, system tools — no browser)
- **Network**: Bridge with port mapping (macOS/Windows) or host network (Linux)
- **Volume**: `openlegion_data_{agent_id}` mounted at `/data` (agent names with spaces/special chars are sanitized)
- **Resources**: 384MB RAM, 0.15 CPU (agents are I/O-bound — waiting on LLM APIs). The built-in `operator` agent runs at 128MB / 0.05 CPU.
- **Security**: `no-new-privileges`, `cap_drop=[ALL]`, `read_only=True`, `tmpfs=/tmp`, non-root UID 1000
- **Port**: 8400 (FastAPI, inside the container; host port allocated dynamically)

**Browser service container** (shared across all agents):
- **Image**: `openlegion-browser:latest` (Camoufox stealth browser + KasmVNC)
- **Resources**: 2–8GB RAM, 1–2 CPU, 512MB–2GB shared memory — scaled by `OPENLEGION_MAX_AGENTS` plan tier
- **Ports**: 8500 (browser API) is the only exposed port. Per-agent KasmVNC instances run internally on 6100..6163 and are reverse-proxied by the mesh at `/agent-vnc/{agent_id}/...` (no direct port exposed to the host).
- **Capacity**: 1, 5, or 10 concurrent browser sessions on the standard plan tiers; absolute cap 64 via `OPENLEGION_BROWSER_MAX_CONCURRENT` (legacy alias `MAX_BROWSERS`). Restart the browser service to apply a change.

### Browser Capabilities

Beyond the basic navigation/screenshot/click tools, the browser service ships with:

- **CAPTCHA solving.** Optional 2captcha or capsolver provider configured per-fleet via `CAPTCHA_SOLVER_KEY` + `CAPTCHA_SOLVER_PROVIDER`. Solver credentials (`CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD`) are env-only by design — they bypass the `OPENLEGION_CRED_*` vault and are stripped from `config/settings.json` at load (`_ENV_ONLY_FLAGS` in `src/browser/flags.py`). Auto-solve runs after `browser_navigate`; behavioral / persistent challenges escalate to `request_captcha_help` which posts a card to the dashboard for the user to clear via the live VNC viewer. Disabled fleet-wide with `CAPTCHA_DISABLED=1`.
- **Per-agent + per-tenant solver cost caps.** `CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH` and `CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH` enforce monthly spend with 50/80/100% threshold alerts. Per-tenant rollups available at `/dashboard/api/billing/captcha-rollup` (requires a dashboard session cookie and the `X-Requested-With` CSRF header).
- **Fingerprint health monitoring.** A rolling per-agent rejection window detects when a fingerprint is "burned" (>50% rejection over the last 10 events across Cloudflare / DataDome / PerimeterX / Imperva / Akamai BMP signals); subsequent CAPTCHA envelopes carry `fingerprint_burn=True` and a `retry_with_fresh_profile` hint. Operator clears state manually after profile rotation.
- **JS-challenge detection.** Vendor-specific selectors detect Cloudflare 1xxx / Under Attack / Press & Hold and similar interstitials before the agent attempts to extract content.
- **Mobile emulation profiles.** `BROWSER_DEVICE_PROFILE` env var (per-agent or fleet-wide) selects a mobile UA + viewport + touch profile when sites gate on desktop fingerprints. Configured via env, not the dashboard.
- **Session continuity (opt-in).** `BROWSER_SESSION_PERSISTENCE_ENABLED=1` enables a per-agent storage-state sidecar so cookies and localStorage survive container restarts. Default-off; operator/curl-only management via `/dashboard/api/agents/{id}/session`.
- **Two-stage workspace upload.** `browser_upload_file` reads from the agent's `/data` and uploads via a stage-then-apply protocol with idempotency keys and a tmpfs partial reaper, so a half-completed upload can never end up attached to a form. Per-file cap 50 MB (`OPENLEGION_UPLOAD_STAGE_MAX_MB`), max 5 files per call.

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

### Task Mode (`POST /task`)

Accepts a `TaskAssignment` for task execution. Runs a bounded loop
(max 20 iterations) of decide → act → learn. Returns a `TaskResult` with
structured output and optional blackboard promotions.

### Chat Mode (`POST /chat`)

Accepts a user message. On the first message, loads bootstrap workspace files
into the system prompt — TEAM.md (team members only; the bootstrap loader
retains a read-only fallback for stray `PROJECT.md` files from pre-rename
workspaces), SYSTEM.md, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md — injects
a live Runtime Context
block (permissions, budget, fleet, cron), and searches memory for relevant facts.
Executes tool calls in a bounded loop with three caps from `loop.py`:
`CHAT_MAX_TOOL_ROUNDS=30` per turn, `CHAT_MAX_TOTAL_ROUNDS=200` total before
auto-compaction kicks in, and `_MAX_SESSION_CONTINUES=5` auto-continuations
(after which the session halts with an error rather than continuing forever).

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
| `run_command` | Shell command execution with timeout |
| `read_file` | Read file contents from `/data` |
| `write_file` | Write/append file in `/data` |
| `list_files` | List/glob files in `/data` |
| `http_request` | HTTP GET/POST/PUT/DELETE/PATCH |
| `browser_navigate` | Open URL, extract page text via shared browser service. Auto-detects CAPTCHAs and may auto-solve or surface a help envelope. |
| `browser_get_elements` | Accessibility tree snapshot with element refs (e1, e2, ...) |
| `browser_find_text` | Locate elements by visible/accessible name (Unicode case-fold match) |
| `browser_screenshot` | Capture page screenshot |
| `browser_click` | Click element by ref or CSS selector |
| `browser_click_xy` | Click at viewport-relative pixel coordinates (canvas / non-accessible widgets) |
| `browser_type` | Fill input by ref or CSS selector |
| `browser_fill_form` | Fill multiple labeled form fields in one call |
| `browser_hover` | Hover over element to trigger dropdowns/tooltips |
| `browser_scroll` | Scroll page up/down or scroll element into view |
| `browser_wait_for` | Wait for CSS selector to appear/disappear |
| `browser_press_key` | Press keyboard key or shortcut (Escape, Enter, Control+a) |
| `browser_go_back` / `browser_go_forward` | Navigate browser history |
| `browser_open_tab` | Open a URL in a new tab (becomes the active page) |
| `browser_switch_tab` | List open tabs or switch to a specific tab |
| `browser_upload_file` | Upload workspace files to a file-input element (1-5 files) |
| `browser_download` | Click a ref to trigger a download and save it as an artifact (≤50MB) |
| `browser_inspect_requests` | List recent network request URLs (redacted; no bodies or headers) |
| `browser_reset` | Reset browser session (profile preserved) |
| `browser_detect_captcha` | CAPTCHA detection (usually not needed — `browser_navigate` auto-detects) |
| `browser_solve_captcha` | Explicitly request a CAPTCHA solve on the current page |
| `request_captcha_help` | Hand a behavioral / persistent CAPTCHA to the user via the dashboard viewer |
| `request_browser_login` | Navigate browser to a URL and send a VNC login card to the user for manual credential entry |
| `generate_image` | Generate an image via Gemini or DALL-E 3 and save as an artifact |
| `memory_search` | Hybrid search across workspace files and structured DB |
| `memory_save` | Save fact to workspace and structured memory DB |
| `web_search` | Search the web via DuckDuckGo (HTML scrape — no API key, but subject to occasional rate limits / CAPTCHAs) |
| `notify_user` | Send notification to user across all connected channels |
| `list_agents` | Discover agents on your team (solo agents see only themselves) |
| `read_blackboard` | Read from the shared blackboard |
| `write_blackboard` | Write to the shared blackboard |
| `list_blackboard` | Browse blackboard entries by prefix |
| `publish_event` | Publish event to mesh pub/sub |
| `subscribe_event` | Subscribe to a pub/sub topic at runtime |
| `hand_off` | Hand a work item to another agent via structured coordination protocol |
| `check_inbox` | Check for pending work items handed off by other agents |
| `update_status` | Update the status of an in-progress work item visible to coordinators |
| `complete_task` | Mark a coordination work item as complete with a result |
| `watch_blackboard` | Watch blackboard keys matching a glob pattern |
| `claim_task` | Atomically claim a task from the shared blackboard |
| `save_artifact` | Save deliverable file and register on blackboard |
| `update_workspace` | Update identity files (SOUL.md, INSTRUCTIONS.md, USER.md, HEARTBEAT.md) |
| `set_cron` | Schedule a recurring job (set `heartbeat=true` for autonomous wakeups) |
| `list_cron` / `remove_cron` | Manage scheduled jobs |
| `create_skill` | Write a new Python skill at runtime |
| `reload_skills` | Hot-reload all skills |
| `spawn_fleet_agent` | Spawn an ephemeral sub-agent in a new container |
| `spawn_subagent` | Spawn a lightweight in-container subagent for parallel subtasks |
| `list_subagents` | List active subagents and their status |
| `wait_for_subagent` | Wait for a subagent to complete and return its result |
| `vault_generate_secret` | Generate and store a random secret (returns opaque handle) |
| `vault_list` | List credential names (names only, never values) |
| `wallet_get_address` | Get Ethereum/Solana wallet address for an agent (requires `[wallet]` extras) |
| `wallet_get_balance` | Get wallet balance (ETH or SOL) (requires `[wallet]` extras) |
| `wallet_read_contract` | Read data from an Ethereum smart contract (requires `[wallet]` extras) |
| `wallet_transfer` | Transfer ETH or SOL to an address (requires `[wallet]` extras) |
| `wallet_execute` | Execute an Ethereum smart contract function (requires `[wallet]` extras) |
| `get_system_status` | Query own runtime state: permissions, budget, fleet, cron, health |
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
  │  Bootstrap files loaded into the first-message system prompt:
  │    TEAM.md (team members only; legacy `PROJECT.md` read-only fallback), SYSTEM.md, INSTRUCTIONS.md,
  │    SOUL.md, USER.md, MEMORY.md
  │  Other workspace files:
  │    HEARTBEAT.md             (autonomous monitoring rules)
  │    INTERFACE.md, AGENTS.md  (channel + roster context)
  │    memory/YYYY-MM-DD.md     (daily session logs)
  │  FTS5 keyword search across files
  │
Layer 2: Structured Memory DB     ← Hierarchical vector database
  │  SQLite + sqlite-vec + FTS5
  │  Hybrid search: 0.7 vector similarity + 0.3 FTS5 keyword
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
4. Replaces message history with: summary + last 3–4 messages (role-aware, preserving message alternation invariant)

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
Cron jobs can also dispatch in **tool-mode** (`tool_name` + `tool_params`), invoking a built-in tool directly without an LLM round — useful for cheap deterministic monitoring. For example, set_cron with `tool_name="http_request"` and `tool_params={"url": "...", "method": "GET"}` polls an endpoint on a schedule without spending tokens.

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

Named webhook URLs that dispatch payloads to agents. Create one from the
dashboard (System → Automation) or via the mesh API; the URL it returns is
what you POST to. Payloads are sanitized and capped at 1MB.

```bash
# Replace hook_a1b2c3d4 with the ID returned when you created the hook.
curl -X POST http://localhost:8420/webhook/hook/hook_a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{"event": "push", "repo": "myproject"}'
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

CAPTCHA solver spend is tracked **separately** from LLM spend (per-agent and per-tenant USD caps with 50/80/100% threshold alerts) — see [Browser Capabilities](#browser-capabilities). The two budgets do not share a pool.

---

## Security Model

Defense-in-depth with six layers:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Runtime isolation | Docker containers (default); Docker Sandbox microVMs with `--sandbox` (Docker Desktop 4.58+ required) | Agent escape, kernel exploits |
| Container hardening | Non-root user, no-new-privileges, memory/CPU limits | Privilege escalation, resource abuse |
| Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs | Unauthorized data access |
| Input validation | Path traversal prevention, SSRF blocking, safe condition eval (no `eval()`), token budgets, iteration limits, rate limiting | Injection, runaway loops, network abuse |
| Unicode sanitization | Invisible character stripping at ~110 call sites across 17 source files, covering all external input boundaries | Prompt injection via hidden Unicode |

### Dual Runtime Backend

OpenLegion supports two isolation levels:

| | Docker Containers (default) | Docker Sandbox microVMs |
|---|---|---|
| **Isolation** | Shared kernel, namespace separation | Own kernel per agent (hypervisor) |
| **Escape risk** | Kernel exploit could escape | Hypervisor boundary — much harder |
| **Performance** | Native speed | Near-native (Apple Virtualization.framework on macOS / Hyper-V on Windows) |
| **Requirements** | Any Docker install | Docker Desktop 4.58+ |
| **Enable** | `openlegion start` | `openlegion start --sandbox` |

**Docker containers** (default) run agents as non-root with `no-new-privileges`, 384MB memory limit, 0.15 CPU cap, and no host filesystem access. Browser operations are handled by a shared browser service container (2–8GB RAM scaled by fleet size, 1 CPU). This is secure for most use cases.

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
├── start [--config PATH] [-d] [--sandbox] [--port PORT]   # Start runtime + interactive REPL (inline setup on first run)
├── stop                                                   # Stop the runtime + agent containers
├── chat [name] [--port PORT]                              # Connect to a running agent
├── status [--port PORT] [--wide/-w] [--watch N] [--json]  # Show agent status
├── teams [--port PORT] [--json]                           # List active teams (alias: ``projects``)
├── team <team_id> [--port PORT] [--json]                  # Show one team (members, blockers, task counts) (alias: ``project``)
├── tasks [--agent X] [--team Y | --project Y] [--status S] [--port PORT] [--json]   # List durable task records
├── pending [--port PORT] [--json]                         # List pending actions awaiting confirmation
├── confirm <nonce> [--port PORT]                          # Confirm a pending action
├── cancel <nonce> [--port PORT]                           # Cancel a pending action
├── reset [-y]                                             # DESTRUCTIVE: stop everything and wipe config/, data/, skills/* (keeps .env)
├── version [--verbose/-v]                                 # Show version and environment info
└── wallet                                                 # Manage agent wallets (derives EVM + Solana from one master seed)
    ├── init                                               # Generate the master wallet seed (shown once; HTTP 410 thereafter)
    └── show [agent_id]                                    # Show wallet addresses
```

> Agent management, credentials, blackboard, cron, and channels
> are managed via **REPL commands** (below) inside a running session, or via the
> **web dashboard** at `http://localhost:8420` (default port; change with `--port` flag or `mesh.port` in `config/mesh.yaml`).

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
/cron [list|del|pause|resume|run]    Manage cron jobs
/project [list|use|info]              Manage multi-team namespaces
/credential [add|list|remove]        Manage API credentials
/traces [id]                         Show recent request traces
/logs [--level LEVEL]                Show recent runtime logs
/addkey <svc> [key]                  Add an API credential to the vault
/removekey [name]                    Remove a credential from the vault
/reset                               Clear conversation with active agent
/quit                                Exit and stop runtime

Aliases: /exit = /quit, /agents = /status, /debug = /traces
```

### Team Templates

Templates are offered during first-run setup (via `openlegion start`):

| Template | Agents | Description |
|----------|--------|-------------|
| `starter` | assistant | Single general-purpose agent |
| `sales` | researcher, qualifier, outreach | Sales pipeline team |
| `devteam` | pm, engineer, reviewer | Software development team |
| `content` | researcher, writer, editor | Content creation team |
| `deep-research` | scout, analyst, writer | Deep research and analysis team |
| `monitor` | watcher, analyst | Autonomous monitoring agent |
| `competitive-intel` | scout | Market and competitor analysis |
| `lead-enrichment` | enricher, formatter | Lead data enrichment |
| `price-intelligence` | crawler, analyst | Price monitoring and analysis |
| `review-ops` | monitor, responder | Review and feedback management |
| `social-listening` | monitor, writer | Social media monitoring |
| `opportunity-finder` | gap-scout, evaluator, modeler | Market opportunity discovery |
| `research` | researcher | General-purpose research agent |

---

## Configuration

### `TEAM.md` — Per-Team Context

Each team has its own `TEAM.md` stored in
`config/projects/{name}/team.md`. (The on-disk dir stays
`config/projects/` during PR 2 of the project→team rename;
the legacy filename `project.md` still resolves as a fallback.)
The file is mounted into team member agents' containers and loaded into
their system prompts. Solo agents (not on a team) do not receive any
TEAM.md.

```markdown
# TEAM.md

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
    model: "openai/gpt-4o-mini"
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
OPENLEGION_CRED_WHATSAPP_APP_SECRET=...        # X-Hub-Signature-256 verification (production)

# Log format: "json" (default — production / structured) or "text" (human-readable).
# If you see JSON logs locally and want plain text, set this to "text".
OPENLEGION_LOG_FORMAT=text

# Plan limits (0 = unlimited). HTTP 403 once exceeded.
# OPENLEGION_MAX_AGENTS=0
# OPENLEGION_MAX_TEAMS=0   # legacy alias: OPENLEGION_MAX_PROJECTS
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

# WhatsApp (both required; APP_SECRET required for production webhook signature verification)
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...
OPENLEGION_CRED_WHATSAPP_APP_SECRET=...      # X-Hub-Signature-256 verification
```

On next `openlegion start`, a pairing code appears — send it to your bot to link.

---

## MCP Tool Support

OpenLegion supports the **[Model Context Protocol (MCP)](https://modelcontextprotocol.io)** —
the emerging standard for LLM tool interoperability. Any MCP-compatible tool server
can be plugged into an agent via config, with tools automatically discovered and
exposed to the LLM alongside built-in skills.

> **Note:** MCP support is an optional dependency and is NOT installed by `./install.sh`. From the cloned repo, activate the venv and run `pip install -e '.[mcp]'`. Without it, agents with `mcp_servers` configured will log an import error and skip MCP tool loading at startup. Wallet tools likewise require the optional `[wallet]` group (`web3`, `eth-account`, `mnemonic`, `solders`, `solana`).

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

Roughly **5800+ test cases across 155 test files** (`find tests -name '*.py' | xargs grep -c '^def test_'`). Coverage includes every module under `src/` — `tests/test_FOO.py` maps to `src/.../FOO.py` (see CLAUDE.md for the full mapping). The four `tests/test_e2e*.py` files require Docker and a real LLM key; everything else runs in CI in under a few minutes per shard.

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
│       ├── subagent_tool.py            # Spawn in-process subagents
│       ├── coordination_tool.py        # Structured inter-agent coordination (hand_off, check_inbox, update_status, complete_task)
│       ├── image_gen_tool.py           # Image generation via Gemini or DALL-E 3
│       └── wallet_tool.py              # Wallet operations (Ethereum + Solana)
├── host/
│   ├── server.py                       # Mesh FastAPI server
│   ├── mesh.py                         # Blackboard, PubSub, MessageRouter
│   ├── permissions.py                  # Permission matrix
│   ├── credentials.py                  # Credential vault + API proxy
│   ├── failover.py                     # Model health tracking + failover chains
│   ├── runtime.py                      # RuntimeBackend ABC + Docker/Sandbox backends
│   ├── transport.py                    # Transport ABC + Http/Sandbox transports
│   ├── cron.py                         # Cron scheduler + heartbeats
│   ├── webhooks.py                     # Named webhook endpoints
│   ├── costs.py                        # Cost tracking + budgets (SQLite)
│   ├── health.py                       # Health monitor + auto-restart
│   ├── lanes.py                        # Per-agent FIFO task queues
│   ├── traces.py                       # Request tracing + grouped summaries
│   ├── transcript.py                   # Provider-specific transcript sanitization
│   ├── wallet.py                       # WalletService — Ethereum + Solana operations
│   └── api_keys.py                     # Named API key management (salted SHA-256 hashes)
├── shared/
│   ├── types.py                        # All Pydantic models (the contract)
│   ├── utils.py                        # ID generation, logging, sanitization
│   ├── trace.py                        # Trace ID generation + correlation
│   ├── models.py                       # Model cost / context window registry (LiteLLM-backed)
│   ├── redaction.py                    # Central credential / URL redactor
│   └── operator_playbooks.py           # Built-in operator agent prompts
├── browser/
│   ├── __main__.py                     # Container entry (KasmVNC + Openbox + FastAPI)
│   ├── server.py                       # Browser service FastAPI server
│   ├── service.py                      # Camoufox session management (per-agent profiles)
│   ├── captcha.py                      # CAPTCHA solver core (2captcha, capsolver)
│   ├── captcha_policy.py               # Per-site classifier (auto-solve vs hand-off)
│   ├── captcha_cost_counter.py         # Per-agent + per-tenant solver cost rollups
│   ├── js_challenge.py                 # JS-challenge / vendor-fingerprint detection
│   ├── session_persistence.py          # Storage-state sidecar (opt-in continuity)
│   ├── profile_schema.py               # Profile schema versioning + uBO migration
│   ├── flags.py                        # Centralized browser flag registry
│   ├── ref_handle.py                   # RefHandle / ShadowHop element resolver
│   ├── canary.py                       # Stealth canary probe
│   ├── recorder.py                     # Behavior recorder
│   ├── stealth.py                      # Anti-detection configuration
│   ├── timing.py                       # Human-like timing jitter
│   └── redaction.py                    # Credential redaction for browser content
├── channels/
│   ├── base.py                         # Abstract channel with unified UX
│   ├── telegram.py                     # Telegram adapter
│   ├── discord.py                      # Discord adapter
│   ├── slack.py                        # Slack adapter (Socket Mode)
│   └── whatsapp.py                     # WhatsApp Cloud API adapter
├── dashboard/
│   ├── server.py                       # Dashboard FastAPI router + API
│   ├── events.py                       # EventBus for real-time streaming
│   ├── auth.py                         # Session cookie verification (CSRF + HMAC)
│   ├── notifications.py                # Persistent notifications store (SQLite)
│   ├── telemetry.py                    # SPA telemetry event sink
│   ├── platform_success.py             # Per-tenant success scoring
│   ├── conversations.py                # Chat-panel conversation state
│   ├── templates/index.html            # Dashboard UI (Alpine.js + Tailwind via CDN)
│   └── static/                         # CSS + JS assets + avatars
├── setup_wizard.py                    # Guided setup wizard
├── marketplace.py                     # Skill marketplace (git-based install/remove)
└── templates/
    ├── starter.yaml                    # Single-agent template
    ├── sales.yaml                      # Sales pipeline team
    ├── devteam.yaml                    # Dev team template
    ├── content.yaml                    # Content creation team
    ├── deep-research.yaml              # Deep research and analysis team
    ├── monitor.yaml                    # Autonomous monitoring agent
    ├── competitive-intel.yaml          # Competitive intelligence team
    ├── lead-enrichment.yaml            # Lead data enrichment
    ├── price-intelligence.yaml         # Price monitoring and analysis
    ├── review-ops.yaml                 # Review and feedback management
    ├── social-listening.yaml           # Social media monitoring
    ├── opportunity-finder.yaml         # Market opportunity discovery
    └── research.yaml                   # General-purpose researcher

config/
├── mesh.yaml                           # Framework settings
├── agents.yaml                         # Agent definitions (per-team)
├── permissions.json                    # Per-agent ACLs
└── teams/                              # Multi-team namespaces (pre-rename ``config/projects/`` resolves via a startup-migrator symlink)
```

---

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Messages, not method calls | Agents communicate through HTTP/JSON. Never shared memory or direct invocation. |
| The mesh is the only door | No agent has network access except through the mesh. No agent holds credentials. |
| Private by default, shared by promotion | Agents keep knowledge private. Facts are explicitly promoted to the blackboard. |
| Explicit failure handling | Domain-specific exceptions propagated with context. No silent error swallowing. |
| Small enough to audit | ~77,000 lines in `src/`. The entire codebase is still auditable in a day. |
| Skills over features | New capabilities are agent skills, not mesh code. |
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
~77,000-line auditable codebase.

**Keywords:** autonomous AI agents, multi-agent framework, LLM agent orchestration,
self-hosted AI agents, Docker AI agents, OpenClaw alternative, AI agent security,
agent cost tracking, Telegram AI bot, Python AI agent framework
