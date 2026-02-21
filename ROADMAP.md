# OpenLegion Roadmap

Prioritized for public launch. Ordered by adoption impact — informed by competitive analysis across OpenClaw, MemU, NanoBot, NanoClaw, HermitClaw, and ZeroClaw. Tier 4 priorities driven by deep agent-level comparative analysis of OpenClaw (Feb 2026).

**Design principle:** Ship what users need to adopt, then what power users demand, then what separates us from everyone else. Never compromise security boundaries for convenience.

---

## Tier 1: Adoption Critical (Going Public)

These directly affect whether new users adopt and stay. Every competitor we analyzed nails at least two of these.

### ~~1.1 SOUL.md Persona Injection~~ ✅ Done (Session 8)

Implemented. See Completed section.

### ~~1.2 MCP Tool Support~~ ✅ Done (Session 9)

Implemented. See Completed section.

### ~~1.3 Setup Wizard Polish~~ ✅ Done (Session 10)

Implemented. See Completed section.

---

## Tier 1.5: Security Hardening

The blind credential management feature (Session 7) exposed a critical gap: agents self-report their identity on mesh requests. A rogue LLM can spoof `agent_id` to bypass permissions. These items close that gap before we go public with vault features.

### ~~1.5a Mesh Auth Tokens~~ ✅ Done (Session 8)

Implemented. See Completed section.

### ~~1.5b Vault Security Hardening~~ ✅ Done (Session 8)

Implemented. See Completed section.

---

## Tier 2: Reliability + Scale

Production hardening for users who push agents hard. These prevent failures, reduce costs, and unlock power-user workflows.

### ~~2.1 Silent Reply Token~~ ✅ Done (Session 8)

Implemented. See Completed section.

### ~~2.2 Provider-Specific Transcript Hygiene~~ ✅ Done (Session 11)

Implemented. See Completed section.

### ~~2.3 Channel Expansion (WhatsApp + Slack)~~ ✅ Done (Session 12)

Implemented. See Completed section.

### ~~2.4 Message Queue Modes (Steer / Collect / Followup)~~ ✅ Done (Session 12)

Implemented. See Completed section.

### ~~2.5 Token Estimation + Proactive Compaction~~ ✅ Done (Session 13)

Implemented. See Completed section.

### ~~2.6 Tool Memory (MemU-Inspired)~~ ✅ Done (Session 14)

Implemented. See Completed section.

---

## Tier 3: Competitive Edge

Features that differentiate OpenLegion from every other runtime. These are what make users choose us over alternatives.

### ~~3.1 Browser Tiers (Stealth + Advanced)~~ ✅ Done (Session 15)

Implemented. See Completed section.

### ~~3.2 Skill Marketplace~~ ✅ Done (Session 18)

Implemented. See Completed section.

### ~~3.3 Subagent Spawning (In-Container)~~ ✅ Done (Session 18)

Implemented. See Completed section.

### ~~3.4 Labeled Screenshots~~ ✅ Done (Session 18)

Implemented. See Completed section.

### ~~3.5 Prompt Injection Sanitization~~ ✅ Done (Session 16)

Implemented. See Completed section.

---

## Tier 4: Agent Intelligence

Identified through deep comparative analysis of OpenClaw's individual agent capabilities. These improve every agent in the fleet — highest user-facing impact per engineering hour. Prioritized above observability because agents that reason better, recover faster, and stream responses affect every single user interaction.

### ~~4.1 CLI Refactor~~ ✅ Done (Session 19–20)

Split `cli.py` (~1968 lines) into `runtime.py`, `repl.py`, `channels.py`, `formatting.py`, `config.py`. Unified `agent edit` command, grouped REPL help, live-apply edits, log noise cleanup.

### ~~4.2 Token-Level LLM Streaming~~ ✅ Done (Session 21)

Implemented. See Completed section.

### ~~4.3 Tool Loop Detection~~ ✅ Done (Session 22)

Implemented. See Completed section.

### 4.4 Extended Thinking / Reasoning Support

No support for Claude's extended thinking or OpenAI's reasoning effort parameters. OpenClaw supports configurable thinking levels (off/low/medium/high) with auto-fallback to lower levels when unsupported. Extended thinking produces dramatically better results on complex tasks (planning, debugging, multi-step reasoning) at the cost of more tokens.

**Impact:** Significant quality improvement on complex tasks. Zero-effort capability boost — just passing a parameter through.

**Changes:**
- Add `thinking` parameter to `LLMClient.chat()` for Anthropic models:
  - When model starts with `anthropic/claude` and thinking is enabled, add `thinking: {"type": "enabled", "budget_tokens": N}` to the API request
  - Budget tokens configurable, default 8192
- Add `reasoning_effort` parameter for OpenAI o-series models:
  - When model contains `/o3` or `/o4`, add `reasoning_effort: "medium"` (or configured level)
- Configurable per-agent in `agents.yaml`: `thinking: "medium"` (maps to budget: low=4096, medium=8192, high=16384). Default: off (cost savings).
- Auto-strip thinking blocks from response before tool-call parsing (Anthropic returns thinking in a separate content block)
- Mesh proxy passes through unchanged — no credential changes needed
- Context manager accounts for thinking tokens in usage tracking

### 4.5 Multi-Chunk Compaction

Current `_summarize_compact()` in `src/agent/context.py` makes a single LLM call to summarize the entire context. For long conversations (50K+ tokens), important details are lost in the single-pass summary. OpenClaw splits context into chunks, summarizes each independently, then merges — producing higher-fidelity summaries.

**Impact:** Better long-conversation quality. Prevents information loss during compaction — critical for multi-hour agent sessions.

**Changes:**
- When estimated context exceeds 40K tokens at compaction time:
  - Split messages into chunks of ~15K tokens each (respecting tool-call grouping boundaries from `_trim_context()`)
  - Summarize each chunk independently with existing summary prompt (can run as parallel `asyncio.gather()` LLM calls)
  - Merge summaries with instruction: "Merge these partial summaries into a single cohesive summary. Preserve all decisions, TODOs, open questions, constraints, and key facts."
  - Replace conversation with `[merged_summary] + last_4_messages`
- Below 40K tokens: keep current single-call approach (simpler, sufficient for shorter contexts)
- Add retry (2 attempts with 2s backoff) for compaction LLM calls before falling through to hard prune
- Log compaction quality metric: `ratio = len(summary) / len(original_context)`

### 4.6 Agent-Level Model Fallback

The mesh has fleet-level model failover (`src/host/failover.py`), but individual agents retry the same model 3 times on failure. OpenClaw rotates to a completely different model on failure, with configurable fallback chains per agent. When Claude is down, the agent should transparently switch to GPT-4o rather than failing 3 times and giving up.

**Impact:** Higher agent uptime during provider outages. Cost savings by avoiding 3 retries to a dead endpoint.

**Changes:**
- Add `fallback_models` list to agent config in `agents.yaml` (e.g., `fallback_models: ["anthropic/claude-sonnet-4-6", "openai/gpt-4o"]`)
- In `_llm_call_with_retry()`: on 2nd retry, if `fallback_models` is configured, try the next model in the chain instead of the same model
- Propagate the actual model used back in `LLMResponse.model` so cost tracking attributes correctly (already happens via mesh failover, extend to agent-level)
- Log model switches: `logger.warning("Falling back from %s to %s", primary, fallback)`

### 4.7 Web Search Upgrade + Content Extraction

DuckDuckGo is rate-limited and returns low-quality snippets. No content extraction tool exists — `http_request` returns raw HTML, wasting context tokens. OpenClaw supports 3 search providers (Brave, Perplexity, Grok) with result caching and has readability-based web content extraction. This is our weakest built-in capability.

**Impact:** Directly affects every web research task. Better search + extraction = better agent research quality across all use cases.

**Changes:**
- **Search providers** (in `src/agent/builtins/web_search_tool.py`):
  - Add Brave Search API as primary provider — API key resolved via vault (`mesh_client.vault_resolve("brave_search_api_key")`), falls back to DuckDuckGo if no key configured
  - Add Tavily as optional alternative (same vault pattern)
  - Provider selection: check vault for keys in order (Brave → Tavily → DuckDuckGo fallback)
- **`web_fetch` tool** (new file `src/agent/builtins/web_fetch_tool.py`):
  - Fetches URL, extracts readable content via `trafilatura` (already handles HTML-to-text well, pure Python, no browser needed)
  - Returns clean markdown text, capped at 50K chars
  - SSRF protection: block private IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x, ::1) before connecting
  - Timeout: 30 seconds
  - Content-type check: only process text/html responses
- **Search result caching**: simple `dict[str, (float, list)]` with 15-minute TTL, cleared on agent restart. Key = query string, value = (timestamp, results). Check before hitting search API.
- Add `trafilatura` to agent container dependencies

---

## Tier 5: Observability Dashboard

The moat is multi-agent orchestration — but multi-agent systems are opaque by default. The REPL shows one conversation at a time. Logs go to a file. There's no way to watch the fleet, see how agents think, or trace a request through the system. This tier adds real-time observability via a web dashboard served from the existing mesh server.

**Architecture decision:** Lightweight SPA served by FastAPI at `http://localhost:8420/dashboard`. No Node.js, no build step, no separate server. Alpine.js (~15KB) for reactivity, Tailwind CSS via CDN for styling, native WebSocket for real-time events. Zero new Python dependencies. `pip install openlegion` includes the dashboard — nothing else to install or build.

**Directory structure:**
```
src/dashboard/
  __init__.py
  server.py           # FastAPI router: WebSocket endpoint + static file mount
  events.py           # Event bus: collect from mesh components, buffer, broadcast to WS clients
  static/
    js/app.js         # Alpine.js dashboard application
    js/websocket.js   # WS client with auto-reconnect
    css/dashboard.css # Tailwind utilities + custom component styles
  templates/
    index.html        # Single page — loads Alpine + Tailwind + Chart.js from CDN
```

### 5.1 Request Tracing

Foundation for the dashboard. Without trace IDs, events are disconnected fragments.

- Generate `trace_id` (UUID4) at every dispatch entry point: REPL message send, channel inbound, workflow step, cron trigger
- Propagate via HTTP header (`X-Trace-Id`) through transport → agent → LLM proxy → response
- Agent includes `trace_id` in tool calls and results
- Store trace events in SQLite table: `(trace_id, timestamp, source, event_type, agent_id, data_json)`
- Ring buffer: keep last 1000 traces, auto-prune older
- REPL `/debug <trace_id>` shows the full request lifecycle as a timeline
- Mesh endpoint: `GET /mesh/traces?agent=<name>&limit=50` for dashboard queries

### 5.2 Event Bus

Real-time event stream that powers the dashboard. Components emit events; the bus broadcasts to connected WebSocket clients.

**Event types:**
| Event | Source | Data |
|-------|--------|------|
| `agent_state` | Health monitor / agent `/status` | state, context_pct, tasks_completed |
| `message_sent` | REPL / channel dispatch | trace_id, from (user/agent), to, preview |
| `message_received` | Transport response | trace_id, agent_id, response preview, token count |
| `tool_start` | Agent stream events | trace_id, agent_id, tool_name, input summary |
| `tool_result` | Agent stream events | trace_id, agent_id, tool_name, output summary |
| `llm_call` | Credential vault proxy | trace_id, agent_id, model, prompt_tokens, completion_tokens, cost_usd |
| `blackboard_write` | Blackboard | trace_id, agent_id, key, value preview, written_by |
| `health_change` | Health monitor | agent_id, old_status, new_status |

**Implementation:**
- `EventBus` class in `src/dashboard/events.py` with `emit(event)` and `subscribe(ws)`
- Ring buffer of last 500 events (dashboard can catch up on connect)
- Components call `event_bus.emit()` at key points — minimal instrumentation
- WebSocket endpoint at `/ws/events` with optional filter params (`?agents=bot1,bot2&types=tool_start,tool_result`)
- Bus is created in `RuntimeContext` and injected into components that emit events

### 5.3 Web Dashboard

The actual UI. Single-page app served at `/dashboard` by the mesh FastAPI server.

**Panels:**

1. **Agents Overview** — All agents as cards. Each shows: name, state (idle/thinking/tool), model, context fill %, daily spend. Cards pulse when active. Click to drill down.

2. **Live Event Feed** — Scrolling timeline of events across all agents. Color-coded by type. Filterable by agent and event type. Click a trace ID to see the full chain.

3. **Agent Detail** (drill-down) — Per-agent view:
   - Current state and context window gauge
   - Conversation stream (like watching the agent think)
   - Tool call history with inputs/outputs
   - Memory stats (facts stored, categories)
   - Cost breakdown (today, this week)

4. **Blackboard Viewer** — Live key-value table of shared state. Highlights recent writes. Shows which agent wrote each entry.

5. **Cost Dashboard** — Per-agent spend chart (Chart.js). Daily/weekly view. Budget utilization bars. Token usage breakdown by model.

6. **Trace Inspector** — Select a trace ID, see the full journey: user message → dispatch → agent receives → LLM call → tool calls → response → cost recorded. Waterfall timeline visualization.

**Tech stack (all via CDN, no build step):**
- Alpine.js — reactive DOM binding, component state
- Tailwind CSS — utility-first styling, dark mode via `class` strategy
- Chart.js — cost and token charts
- Native WebSocket — real-time event stream

**Served by:**
- `src/dashboard/server.py` creates a FastAPI `APIRouter`
- `StaticFiles` mount for `static/`
- Jinja2 template for `index.html` (injects mesh URL for WebSocket connection)
- Included in `create_mesh_app()` from `src/host/server.py`

### 5.4 Orchestrator Polling Elimination

**Problem:** `_wait_for_task_result` in `orchestrator.py` polls for task completion. Session 2 added push-based `asyncio.Future` for the main dispatch path, but the orchestrator's DAG step completion still polls.

**Fix:**
- Wire DAG step completion through `asyncio.Future` or `asyncio.Event`
- Agent posts result → mesh resolves the future → orchestrator proceeds immediately
- Pairs well with event bus — step completion emits a trace event

### 5.5 Cron-Triggered Subagents

Cron jobs spawn subagents instead of blocking main agent:
- `spawn: true` flag in cron config
- Main agent stays available for chat
- Results announced when done
- Dashboard shows subagent activity in parent's detail view

---

## Backlog

Lower priority items grouped by theme. Implement when convenient or when a specific need arises.

### Agent Resilience

**Error Classification** — Distinguish transient errors (retry), auth errors (rotate credentials), billing errors (fail fast with user notice), and permanent errors (fail immediately) in `_llm_call_with_retry()`. Parse `Retry-After` headers on 429s. Currently all retryable errors get the same exponential backoff treatment — a 429 with `Retry-After: 60` should wait 60s, not 1s.

**Tool Result Truncation Before Compaction** — Before triggering full compaction at 70% context, try truncating individual tool results that exceed 20K tokens (e.g., a large `exec` stdout or `read_file` result). Cheaper than full compaction and preserves more conversation structure. OpenClaw does this as an intermediate recovery step.

**Compaction Retry** — If the LLM call in `_summarize_compact()` fails, currently falls through to hard prune (keeps first + last 4 messages). Add 2 retry attempts with 2s backoff before resorting to hard prune. Hard prune is destructive — worth retrying.

### Memory Quality

**Search Result Diversity (MMR)** — Add Maximal Marginal Relevance to `memory.py:search()` to avoid returning near-duplicate hits. After initial retrieval, re-rank results penalizing similarity to already-selected results. Lambda=0.7 (70% relevance, 30% diversity). OpenClaw has this as a configurable option.

**Embedding Provider Flexibility** — Currently hardcoded to OpenAI `text-embedding-3-small` (1536 dims). Allow configuring the embedding model and dimension per agent. Enables: local embedding models for privacy, Voyage for higher quality, Gemini for cost savings. Requires schema migration for different vector dimensions.

**Session Memory Indexing** — Index past task transcripts so agents learn from previous sessions. After each task completes, extract key facts and store in memory DB with source=`session:{task_id}`. OpenClaw has this as an experimental feature — directionally correct for long-running agents.

### Agent Customization

**Per-Agent Tool Policies** — Currently all agents in a fleet share the same tool set (minus permission gating on mesh operations). Allow per-agent tool allowlists/blocklists in `agents.yaml`: `tools: {allow: [exec, read_file, write_file], deny: [browser_*]}`. More granular than the current permission matrix which only gates mesh operations. OpenClaw has a 7-layer tool policy pipeline.

**TOOLS.md Workspace File** — A `TOOLS.md` file in the agent workspace where users can write custom tool usage instructions (e.g., "always use `exec` with `timeout=120` for build commands", "prefer `web_fetch` over `http_request` for reading web pages"). Loaded into the system prompt alongside AGENTS.md and SOUL.md. Cheap to implement, powerful for agent customization.

### Ecosystem

**Bundled Skill Library** — OpenClaw ships 60+ bundled skills (GitHub, Notion, Slack, Spotify, email, 1Password, Apple Reminders, etc.). Our marketplace infrastructure exists but the library is sparse. Prioritize building commonly-requested integration skills. The `@skill` decorator + vault credential pattern makes this straightforward. Start with: GitHub (issues/PRs), email (IMAP/SMTP via vault), calendar, and note-taking (Obsidian/Notion).

**Apply-Patch Tool** — A `apply_patch` tool for multi-file unified diffs. For coding-focused agents, applying a single patch is significantly more efficient than multiple individual `write_file` calls and preserves better context for the LLM. Accept standard unified diff format, validate before applying, support dry-run mode.

### Infrastructure

**Cross-Channel Broadcasting** — Mirror agent responses across channels. PubSub topic per-agent for response events. Channels subscribe to agents they're interested in. Opt-in per channel in `mesh.yaml`.

**Per-Agent Directory Configuration** — Per-agent SOUL.md and workspace paths configurable in `mesh.yaml`. `soul_md` field per agent pointing to a custom SOUL.md path. `workspace_dir` override per agent (default remains `/data`). Auto-scaffold SOUL.md on `openlegion setup` with personality prompt.

**Skill Registry Cleanup** — Replace `_skill_staging` module-level mutable global with class-level registry or explicit registration API. Skills register directly onto a `SkillRegistry` instance instead of through a global.

---

## Completed

### Session 22 (Tool Loop Detection)
- [x] **4.3 Tool Loop Detection** — `ToolLoopDetector` class in `src/agent/loop_detector.py` with SHA-256 hash-based sliding window (15 entries) and 3-level escalation.
- [x] `_hash_json()` — SHA-256 of canonically-serialised JSON, truncated to 16 hex chars (matches `MemoryStore._compute_params_hash` algorithm).
- [x] `check_before()` returns `"ok"` / `"warn"` (>=2 prior identical) / `"block"` (>=4) / `"terminate"` (>=9 same tool+params regardless of result). `would_terminate()` for pre-scan without duplicate logging.
- [x] Integrated into all 3 execution modes: `execute_task()` (task mode), `_chat_inner()` (non-streaming chat), `_chat_stream_inner()` (streaming chat).
- [x] Terminate pre-scan (`_check_tool_loop_terminate`) runs BEFORE appending assistant message to context — prevents orphaned tool_calls without matching tool results.
- [x] Exempt tools: `memory_search`, `memory_recall` (legitimate to re-search after new facts stored).
- [x] Detector resets on `execute_task()` start and `reset_chat()`.
- [x] 14 unit tests in `tests/test_loop_detector.py`, 8 integration tests in `tests/test_loop.py`. 891 total passing. PR #75 merged.

### Session 21 (Token-Level LLM Streaming)
- [x] **4.2 Token-Level LLM Streaming** — True token-by-token streaming from LLM providers through the mesh proxy, through agents, to all consumers.
- [x] `LLMClient.chat_stream()` async generator in `src/agent/llm.py` — streams via SSE from `/mesh/api/stream`, yields `text_delta` and `done` events, raises on error for caller fallback.
- [x] `AgentLoop._chat_stream_inner()` refactored — iterates `llm.chat_stream()` directly in the generator, yielding `text_delta` events immediately. Graceful fallback to non-streaming `_llm_call_with_retry()` on any streaming failure. `any_text_streamed` flag prevents doubled content on partial-streaming fallback.
- [x] **Telegram** progressive text streaming — debounced message editing (500ms), streaming_msg cleanup on tool rounds, overflow handling for >4096 char responses.
- [x] **Discord** progressive text streaming — debounced `message.edit()` (500ms) under typing indicator, overflow handling for >1900 char responses.
- [x] **Slack** progressive text streaming — debounced `chat.update` (500ms), captures `ts` + `channel` from `say()` response, overflow handling.
- [x] **Dashboard** SSE streaming chat endpoint (`/api/agents/{id}/chat/stream`) + ReadableStream JS parser with `text_delta` accumulation and `done` event replacement.
- [x] Credential vault `stream_llm()` done event includes `model` field for accurate cost attribution.
- [x] EventBus `text_delta` filtering — per-token events excluded from EventBus ring buffer (prevents flooding), dashboard streaming uses direct SSE instead.
- [x] Dashboard JS error response parsing hardened (try/catch for non-JSON errors).
- [x] 861 tests passing. PR #72 merged.

### Session 18 (Tier 3 Features — Labeled Screenshots, Skill Marketplace, Subagent Spawning)
- [x] **3.4 Labeled Screenshots** — `browser_screenshot(labeled=True)` overlays numbered red labels on interactive elements using Pillow. Auto-calls `browser_snapshot` if page refs are empty. Draws red bounding boxes + white number labels. Graceful fallback when Pillow not installed. `Pillow` added to `Dockerfile.agent`.
- [x] **3.2 Skill Marketplace** — `openlegion skill install/list/remove` CLI commands for git-based skill packages. Skills validated with existing AST checker (`_validate_skill_code`). `SKILL.md` manifest with YAML front matter (name, version, description required). Stored at `skills/_marketplace/` (gitignored), mounted read-only into containers. `SkillRegistry` auto-discovers marketplace skills. Version pinning via `--ref` flag.
- [x] **3.3 Subagent Spawning** — `spawn_subagent` tool creates lightweight `AgentLoop` instances in the same process. Each subagent gets isolated memory (`:memory:` SQLite) and workspace (`/data/workspace/subagents/<id>/`). Shared LLM/mesh clients (stateless httpx). Results written to blackboard at `subagent_results/{parent_id}/{subagent_id}`. `list_subagents` shows active status. Safety: max 3 concurrent, max depth 2 (no grandchildren), configurable TTL (default 300s), unsafe skills removed from subagent registry.
- [x] 31 new tests (5 labeled screenshots, 16 marketplace, 10 subagent). 712 total passing.
- [x] PRs #39–#41 merged.

### Session 17 (Codebase Audit + Cleanup)
- [x] **4-pass audit** — Code Hygiene, Functional Correctness, Documentation Coverage, UX/DX across all `src/` modules.
- [x] **3 bug fixes**: `AgentHealth.status` stuck on "restarting" after successful restart; `HealthMonitor._try_restart` race where `consecutive_failures` reset before readiness confirmed; `SkillValidator` missing `ast.Subscript` in forbidden-call check allowing `os["system"]()` bypass.
- [x] **1 security fix**: Prompt injection sanitization gap — `tool_history` and workspace search snippets in chat mode bypassed all three sanitization layers. Both paths now call `sanitize_for_prompt()`.
- [x] **Chat mode deduplication** — Extracted 5 shared helpers (`_prepare_chat_turn`, `_build_tool_call_entries`, `_execute_chat_tool_call`, `_compact_chat_context`, `_resolve_content`) from `_chat_inner` and `_chat_stream_inner`, eliminating ~280 lines of duplication.
- [x] **PairingManager extraction** — Replaced raw `_paired` dict in `Channel` base class with `PairingManager` class encapsulating load/save/access-control logic.
- [x] **Silent exception cleanup** — 30+ bare `except Exception: pass` blocks across 16 files converted to `logger.debug()` or narrowed exception types. Covers: `cli.py` (7), `browser_tool.py` (5), `runtime.py` (7), `cron.py` (3), `memory.py` (1), `health.py` (1), `loop.py` (1), `workspace.py` (2), `context.py` (1), `transport.py` (1), `base.py` (1).
- [x] **Documentation updates** — Test count 638→698, line count ~11K→~14K, added missing REPL commands (`/steer`, `/addkey`), rebuilt test coverage table with per-file counts.
- [x] PRs #31–#38 merged. 681 unit/integration tests passing.

### Session 16 (Prompt Injection Sanitization)
- [x] **3.5 Prompt Injection Sanitization** — `sanitize_for_prompt()` in `src/shared/utils.py` strips invisible Unicode characters at three choke points covering all paths from untrusted text to LLM context.
- [x] **Core function**: Category-based stripping (Cc, Cf, Co, Cs, Cn) with safe-list exceptions for TAB/LF/CR, ZWNJ/ZWJ (Persian/Arabic/emoji), VS15/VS16 (emoji presentation). Extra strip set for VS1-14, VS17-256, Combining Grapheme Joiner, Hangul fillers, Object Replacement. U+2028/U+2029 normalized to `\n`.
- [x] **Layer 1 — User Input** (`src/agent/server.py`): All three chat endpoints (`/chat`, `/chat/steer`, `/chat/stream`) sanitize before dispatch.
- [x] **Layer 2 — Tool Results** (`src/agent/loop.py`): All 6 `json.dumps`/`str()` points sanitized in `execute_task()`, `_chat_inner()`, `_chat_stream_inner()` (success + error paths).
- [x] **Layer 3 — System Prompt Context** (`src/agent/loop.py`): Goals, bootstrap content, learnings, tool history, memory facts, related memory, assignment context, and workspace search snippets all sanitized.
- [x] **Review-discovered fixes**: `tool_history` was missing sanitization in `_build_chat_system_prompt()` (chat mode); workspace search snippets in memory auto-load (`_chat_inner`, `_chat_stream_inner`) bypassed all three layers. Both fixed.
- [x] Zero false positives: Arabic, Hebrew, CJK, Devanagari, emoji with ZWJ sequences, ZWNJ for Persian, all preserved. Idempotent.
- [x] 38 unit tests in `tests/test_sanitize.py` + 5 integration tests in `tests/test_loop.py`. 681 total passing.

### Session 15 (Browser Tiers)
- [x] **3.1 Browser Tiers (Stealth + Advanced)** — Three-tier browser backend system controlled by `browser_backend` in agents.yaml.
- [x] **Basic** (default): Playwright + Chromium — unchanged from before.
- [x] **Stealth**: Camoufox anti-detect Firefox browser (`AsyncCamoufox`) for bot-protected sites. Lazy import so basic-mode agents never load it.
- [x] **Advanced**: Bright Data Scraping Browser via CDP (`pw.chromium.connect_over_cdp`). Credential resolved from vault via `mesh_client.vault_resolve("brightdata_cdp_url")` — agent never sees raw secrets.
- [x] Backend dispatch in `_get_page()` reads `BROWSER_BACKEND` env var, delegates to `_launch_basic()`, `_launch_stealth()`, or `_launch_advanced()`.
- [x] Config pipeline: `browser_backend` in agents.yaml → `runtime.py` passes as `BROWSER_BACKEND` env var → `browser_tool.py` reads at launch.
- [x] Resource lifecycle: `browser_cleanup()` properly shuts down Playwright process, Camoufox context manager, and all browser/context/page objects. Called in agent lifespan shutdown.
- [x] Concurrency safety: `asyncio.Lock` prevents dual browser launches from concurrent tool calls.
- [x] `mesh_client` propagated to all 6 browser skill functions for vault access in advanced mode.
- [x] Camoufox installed in `Dockerfile.agent` (`pip install camoufox` + `python -m camoufox fetch`).
- [x] 8 new tests across `test_builtins.py` (6) and `test_runtime.py` (2). 638 total passing.

### Session 14 (Tool Memory)
- [x] **2.6 Tool Memory (MemU-Inspired)** — Dedicated `tool_outcomes` SQLite table with SHA-256 params hashing for deduplication and auto-pruning (50 per tool).
- [x] `store_tool_outcome()` and `get_tool_history()` on `MemoryStore` — sync methods with optional filtering by tool name and params hash.
- [x] `_learn()` records structured success outcomes; `_record_failure()` records failures with full argument tracking.
- [x] **Chat mode gap fix** — `_chat_inner` and `_chat_stream_inner` now call `_learn()` on successful tool execution (previously silently discarded).
- [x] **`_record_failure` SQLite fix** — failures now write to both workspace markdown and SQLite `tool_outcomes` table.
- [x] `_build_tool_history_context()` injects `## Recent Tool History` into both task and chat system prompts.
- [x] 14 new tests across `test_memory.py` (8) and `test_loop.py` (6). 630 total passing.

### Session 13 (Token Estimation + Context Warnings)
- [x] **2.5 Token Estimation + Proactive Compaction** — Model-aware token estimation replacing the rough 4-chars/token heuristic.
- [x] **tiktoken for OpenAI models** — accurate token counting via cached tiktoken encodings with try/except fallback.
- [x] **3.5 chars/token for Anthropic** — better heuristic based on Anthropic guidance. Unknown models fall back to 4 chars/token.
- [x] **`MODEL_CONTEXT_WINDOWS` dict** (12 models) — auto-detects context window size from model name. `ContextManager` uses model-based lookup; explicit `max_tokens` overrides.
- [x] **80% context warning** — `context_warning()` returns warning string injected into system prompts for both chat mode (`_build_chat_system_prompt`) and task mode (`execute_task`). Agents told to wrap up or save facts.
- [x] **`AgentStatus` context fields** — `context_tokens`, `context_max`, `context_pct` (all optional with defaults for backward compat).
- [x] **CLI `/costs` context display** — per-agent context window usage shown after spend summary.
- [x] **Channel `/status` context display** — context percentage shown alongside state in channel status responses.
- [x] 14 new tests across `test_context.py`, `test_loop.py`, `test_chat.py`. 616 total passing.

### Session 12 (Channel Expansion — Slack + WhatsApp)
- [x] **2.3 Channel Expansion** — Two new channel adapters following the established `Channel` ABC pattern.
- [x] **Slack** (`src/channels/slack.py`, ~304 lines) — `SlackChannel` using `slack-bolt` AsyncApp + AsyncSocketModeHandler (Socket Mode, no public URL). Thread-aware routing via composite user key `user_id:thread_ts`. Pairing (`!start <code>`), access control (`!allow`/`!revoke`), `!`-to-`/` command translation. 3000-char message chunking.
- [x] **WhatsApp** (`src/channels/whatsapp.py`, ~324 lines) — `WhatsAppChannel` using httpx (no new dependency) against WhatsApp Cloud API. Webhook-based: FastAPI `APIRouter` with GET verification challenge + POST incoming messages. Text-only initially (media logged and skipped). 4096-char message chunking.
- [x] CLI integration: `openlegion channels add slack/whatsapp` with token prompts. `_start_channels()` returns `(pairing_instructions, webhook_routers)` tuple for app mounting.
- [x] `slack-bolt>=1.18` added to `channels` optional deps.
- [x] 42 new tests (21 Slack, 21 WhatsApp). 579 total passing.

### Session 11 (Transcript Hygiene)
- [x] **2.2 Provider-Specific Transcript Hygiene** — `_sanitize_for_provider(messages, provider)` in `src/agent/loop.py`. Provider-specific repairs: merge consecutive same-role messages, ensure tool-call pairing, fix tool IDs for Gemini/Mistral format requirements. Applied in-memory only — stored transcript unchanged.

### Session 10 (Setup Wizard Polish + Install Hardening)
- [x] **1.3 Setup Wizard Polish** — Extracted `SetupWizard` class into `src/setup_wizard.py` (~260 lines). `run_full()` 4-step guided setup with existing config detection, API key validation via `litellm.acompletion()`, step headers, ASCII summary card. `run_quickstart(model)` zero-prompt single-agent setup. Validation uses cheapest model per provider (Haiku, GPT-4.1-mini, etc.).
- [x] `openlegion quickstart` CLI command with `--model` option
- [x] `install.sh` hardened: broken venv detection/recreation, python3-venv check, sudo guard, Docker permission detection, cross-platform support (macOS `getent` fallback, Windows `Scripts/` paths, platform-aware hints)
- [x] Removed user-generated config files (`config/mesh.yaml`, `config/permissions.json`, `PROJECT.md`) from git tracking. Added `SOUL.md`, `config/workflows/`, `*.log` to `.gitignore`.
- [x] 18 new tests in `tests/test_setup_wizard.py` (513 total passing)

### Session 9 (MCP Tool Support + System Documentation)
- [x] **1.2 MCP Tool Support** — `MCPClient` class manages MCP server lifecycles via stdio transport inside agent containers. Tool discovery via `list_tools()`, call routing via `call_tool()`. Name conflict resolution: builtin wins, MCP tool prefixed with `mcp_{server}_{name}`. Graceful failure: one server crash doesn't affect others. Full pipeline: `config/agents.yaml` → `runtime.py` → `MCP_SERVERS` env var → agent container.
- [x] `SkillRegistry` integration — MCP tools registered alongside builtins, routed in `execute()`, included in `get_tool_definitions()` with full JSON Schema pass-through.
- [x] Config pipeline — `mcp_servers` param added to `RuntimeBackend.start_agent()` ABC + both Docker/Sandbox backends. Health monitor preserves MCP config across restarts.
- [x] `mcp` dependency added to `Dockerfile.agent` and `pyproject.toml` optional deps.
- [x] Echo MCP test fixture (`tests/fixtures/echo_mcp_server.py`) with echo/add/fail tools.
- [x] 22 new tests: 10 unit (mocked SDK), 7 E2E (real stdio transport), 5 SkillRegistry integration. 495 total passing.
- [x] **Comprehensive `docs/` directory** — 11 documentation guides covering architecture, security, agent tools, MCP, memory, workflows, configuration, channels, triggering, and development. All verified against actual codebase for parameter accuracy, file paths, and behavioral claims.
- [x] README updated with MCP section, test count badges (495), dependency table, project structure.

### Session 8 (Adoption + Security Hardening)
- [x] **1.1 SOUL.md Persona Injection** — Fallback chain: agent workspace SOUL.md → project `/app/SOUL.md` → default scaffold. Injected into chat system prompt via `workspace.get_bootstrap_content()`. Subject to 4K cap.
- [x] **1.5a Mesh Auth Tokens** — Per-agent `secrets.token_urlsafe(32)` generated at container startup. Injected as `MESH_AUTH_TOKEN` env var. All mesh requests include `Authorization: Bearer` header. Timing-safe comparison via `hmac.compare_digest`. All 14 agent-facing endpoints protected.
- [x] **1.5b Vault Security Hardening** — `_redact_credentials()` pattern-based redaction on `browser_navigate`, `browser_snapshot`, and `browser_evaluate` output. `_credential_filled_refs` tracking in `browser_type`. Rate limiting on `/mesh/vault/resolve` (5/60s). Audit logging on vault resolve.
- [x] **2.1 Silent Reply Token** — `SILENT_REPLY_TOKEN = "__SILENT__"` detected in all 4 code paths (chat normal, chat max-rounds, stream normal, stream max-rounds). Channels suppress empty responses.
- [x] `/addkey` wired through to Telegram and Discord channel constructors
- [x] 24 new tests (473 total passing)

### Session 7 (Blind Credential Management)
- [x] `can_manage_vault` permission on AgentPermissions + PermissionMatrix enforcement
- [x] Hot-reload credential vault: `add_credential`, `resolve_credential`, `list_credential_names`, `has_credential`
- [x] `_persist_to_env` shared helper (extracted from cli.py)
- [x] 4 mesh vault endpoints: store, list, status, resolve (permission-gated)
- [x] 4 mesh_client vault methods
- [x] `vault_generate_secret` + `vault_capture_from_page` — credential-blind tools (returns $CRED{} handles, never values)
- [x] `vault_list` + `vault_status` tools
- [x] `$CRED{name}` handle resolution in `browser_type` (types credential, returns `[credential]`)
- [x] `_redact_credentials()` pattern-based redaction in `browser_snapshot`
- [x] `/addkey` command in channels (consumed at channel layer, never dispatched to agent)
- [x] `/addkey` command in CLI REPL (inline or hidden-input prompt)
- [x] 37 new tests across 6 test files (449 total passing)
- [x] **Security note:** Protects against accidental leaks. Does NOT yet protect against adversarial LLM (see Tier 1.5)

### Session 6 (E2E Bug Fixes)
- [x] `browser_snapshot`: Playwright removed `page.accessibility.snapshot()` — added `aria_snapshot()` YAML parser with old API fallback (PR #10)
- [x] `memory_save` → `memory_recall` disconnect: facts saved to daily log were invisible to structured DB search. Now stores to both workspace and SQLite memory DB (PR #10)
- [x] `_get_fact()`: LEFT JOIN with categories table for correct category name resolution (PR #10)
- [x] 22 new tests covering all three fixes

### Session 5 (Roadmap 1.5 — Hierarchical Memory)
- [x] Categories table with auto-categorization (vector similarity + `categorize_fn` callback)
- [x] 3-tier `search_hierarchical()`: category-level → scoped fact → flat fallback
- [x] Continuous reinforcement scoring replaces static `SALIENCE_BOOST`
- [x] New `memory_recall` skill for structured fact queries with category filtering
- [x] `memory_search` queries both workspace and memory DB
- [x] `memory_store` injection into skills system
- [x] 15 new tests (test_memory.py + test_builtins.py)

### Session 4 (Roadmap 1.4 — Model Failover)
- [x] `src/host/failover.py` — ModelHealthTracker + FailoverChain
- [x] Configurable failover chains in mesh.yaml (cascade across providers)
- [x] Exponential cooldown: transient 60s→300s→1500s, billing/auth 1h
- [x] Permanent errors (400, 404) don't cascade — prevents silent config masking
- [x] Streaming failover (retry on connection, stay on model once streaming)
- [x] Cost tracking attributed to actual responding model
- [x] `/costs` shows model health summary
- [x] `GET /mesh/model-health` diagnostic endpoint
- [x] Mid-stream failure health tracking
- [x] 21 new tests (test_failover.py + test_credentials.py integration)

### Session 3 (Roadmap 1.1–1.3)
- [x] Proactive memory flush at 60% context usage (PR #5)
- [x] Bootstrap content capping — 40K chars total, per-file limits (PR #5)
- [x] Daily logs excluded from bootstrap, on-demand via memory_search
- [x] Reference-based browser via accessibility tree (PR #6)
- [x] browser_snapshot tool with structured refs
- [x] browser_click/browser_type accept ref param
- [x] 7 runtime bug fixes (PR #4)

### Session 2 (Runtime Hardening)
- [x] httpx client lifecycle (per-loop client pooling)
- [x] Push-based orchestrator (asyncio.Future, no polling)
- [x] LLM retry with exponential backoff
- [x] Skill registry isolation fix
- [x] PubSub SQLite persistence
- [x] Concurrent agent startup
- [x] SSE streaming (agent → CLI/TG)
- [x] Telegram slash command routing
- [x] Thread-safe dispatch loop (dedicated asyncio loop per thread)
- [x] litellm tools= param fix
- [x] Telegram HTML formatting
- [x] Playwright browser path fix (non-root access)
- [x] CHAT_MAX_TOOL_ROUNDS 10 → 30
- [x] "Don't narrate" system prompt directive
- [x] Browser efficiency prompt rules
- [x] Startup UX (pairing code at bottom)
- [x] CLI tool output (step counters, input summaries, result hints)
- [x] Telegram streaming tool progress (editable progress message)
- [x] /costs filtered to active agents only
- [x] Per-loop httpx clients (fixes cross-thread event loop errors)

### Session 1 (Initial Release)
- [x] Core agent loop with tool calling
- [x] Container isolation (Docker + Sandbox)
- [x] Mesh host with blackboard, pub/sub, permissions
- [x] Credential vault + LLM API proxy
- [x] Browser automation (Playwright)
- [x] Memory system (SQLite + sqlite-vec + FTS5)
- [x] Workspace persistence (MEMORY.md, daily logs, learnings)
- [x] CLI REPL with multi-agent support
- [x] Telegram channel with pairing
- [x] Cron scheduler with heartbeat support
- [x] DAG workflow orchestrator
- [x] Health monitor with auto-restart
- [x] Cost tracking + budget enforcement

---

## Competitive Positioning

| Capability | OpenLegion | OpenClaw | ZeroClaw | NanoBot | NanoClaw | MemU | HermitClaw |
|---|---|---|---|---|---|---|---|
| Container isolation | **Docker + Sandbox** | Local process | None | None | Docker | N/A | None |
| Multi-agent | **Fleet (mesh)** | Single + sub-agents | Single | Single | Claude Code | N/A | Single |
| Browser automation | **3-tier: basic + stealth + CDP** | Playwright (3 modes: sandbox/host/node) | None | None | Playwright | N/A | None |
| Memory | **Hierarchical vec+BM25 + salience decay** | SQLite vec + hybrid search (no hierarchy) | SQLite basic | None | Claude memory | **Hierarchical** | Generative Agents |
| Model support | OpenAI + Anthropic via mesh proxy | **15+ providers (Anthropic, OpenAI, Gemini, Bedrock, Ollama, Together, etc.)** | Multi | Multi | Claude only | N/A | Single |
| LLM streaming | **Token-level streaming (all channels)** | **True token-level streaming** | Native | SSE | Claude native | N/A | None |
| Model failover | **Health + cascade (fleet-level)** | Auth rotation + model fallback + thinking fallback | None | None | None | N/A | None |
| Extended thinking | Not yet | **Configurable levels (off/low/med/high)** | None | None | Claude native | N/A | None |
| Credential mgmt | **Blind vault + $CRED handles** | Env vars (visible to agent) | None | None | None | N/A | None |
| Tool loop detection | **SHA-256 hash-based + 3-level escalation (warn/block/terminate)** | SHA-256 hash-based + circuit breaker | None | None | None | N/A | None |
| Error recovery | 3x retry with backoff | **Multi-layer: auth rotation → model fallback → thinking fallback → compaction → truncation** | None | Basic | None | N/A | None |
| Self-authoring tools | **Yes (create_skill + AST validation)** | No | No | No | No | N/A | No |
| Channels | 5 (CLI, TG, Discord, Slack, WA) | **12+ (WA, TG, Slack, Discord, Signal, iMessage, etc.)** | **17+** | 9 | 2 (WA, Web) | N/A | 1 (CLI) |
| MCP support | **Yes (stdio)** | Yes | Yes | Yes | Partial | N/A | No |
| Web search | DuckDuckGo only | **Brave + Perplexity + Grok (cached)** | None | None | None | N/A | None |
| Cost tracking | **Per-agent SQLite** | Provider API via usage accumulator | None | None | None | N/A | None |
| Prompt injection defense | **Unicode sanitization at 3 choke points** | None | None | None | None | N/A | None |
| Context mgmt | **Write-then-compact (60%/70%/80% thresholds)** | Reactive compaction on overflow only | None | None | Claude native | N/A | None |
| Compaction quality | Single-pass summary | **Multi-chunk split → summarize → merge** | None | None | N/A | N/A | None |
| Setup DX | **Guided + quickstart** | Good | **Excellent** | **Excellent** | Simple | N/A | Simple |
| Skill ecosystem | **Marketplace + MCP + self-authoring** | **60+ bundled skills** | **17+ tools** | Community | MCP | N/A | None |
| Observability | **Dashboard planned (Tier 5)** | Logs only | None | None | None | N/A | None |

### What We Win On (vs OpenClaw)

These are genuine architectural advantages that OpenClaw cannot easily replicate:

1. **Security isolation** — Agents in containers with blind credential vault. OpenClaw runs agents in-process; credentials are visible to agent code. Our `$CRED{name}` handles, vault rate limiting, and credential redaction are unique.
2. **Memory sophistication** — Hierarchical 3-tier search, salience decay with access-count boosting, auto-categorization, and tool outcome tracking. OpenClaw has flat vector+FTS with no hierarchy or salience.
3. **Write-then-compact context management** — Proactive fact extraction at 60% before any trimming happens. OpenClaw only compacts reactively on overflow, risking information loss.
4. **Self-authoring tools** — `create_skill` with AST validation and forbidden-import checking. Agents extend their own capabilities at runtime. No competitor has this.
5. **Prompt injection defense** — Unicode sanitization at 3 choke points. OpenClaw has none.
6. **Labeled screenshots** — Red-numbered overlays on interactive elements for precise browser interaction.

### What OpenClaw Wins On (our gaps to close)

These are the items driving Tier 4 priority:

1. ~~**Token-level streaming** (→ 4.2) — Closed. Token-level streaming implemented across CLI, dashboard, Telegram, Discord, and Slack.~~
2. ~~**Tool loop detection** (→ 4.3) — Closed. SHA-256 hash-based detection with 3-level escalation (warn/block/terminate) and sliding window.~~
3. **Extended thinking** (→ 4.4) — Configurable reasoning levels vs our no support. Free quality boost on complex tasks.
4. **Multi-chunk compaction** (→ 4.5) — Split/summarize/merge vs our single-pass. Better long-conversation fidelity.
5. **Error recovery depth** (→ 4.6 + backlog) — 5-layer recovery vs our 3x retry on same model. Auth rotation, model fallback, thinking fallback.
6. **Web search quality** (→ 4.7) — 3 providers with caching vs our single DuckDuckGo. Plus readability content extraction.
7. **Bundled skill library** (→ backlog) — 60+ integrations vs our sparse marketplace. GitHub, email, calendar, notes.
8. **Provider breadth** — 15+ LLM providers vs our model-agnostic-but-only-2-configured setup. Less urgent since mesh proxy can route to any provider.

### Our Moat

Container-isolated multi-agent orchestration with blind credential vault, fleet coordination, hierarchical memory, and self-authoring tools. No competitor combines all five. OpenClaw is closest on individual agent capability but fundamentally cannot match security isolation (in-process agents see credentials) or fleet coordination (no mesh, no blackboard, no DAG workflows).

### Next Differentiators

1. **Agent intelligence** (Tier 4) — Close the remaining individual agent gaps (extended thinking, compaction, model fallback, web search). Token-level streaming and tool loop detection gaps already closed.
2. **Real-time observability** (Tier 5) — No competitor offers fleet-level visibility into multi-agent reasoning, tool usage, and collaboration. Served from the existing mesh server — zero additional infrastructure.
