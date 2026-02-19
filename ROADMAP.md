# OpenLegion Roadmap

Prioritized improvements based on runtime hardening, competitive analysis (OpenClaw, MemU), and real-world testing. Ordered by impact — each tier builds on the previous.

**Design principle:** Go beyond what competitors ship. OpenClaw's context management is good; ours should be better. MemU's memory retrieval is smart; ours should be smarter AND simpler.

---

## Tier 1: Critical Path (Do Next)

These items have the highest impact on agent reliability and efficiency. Each one addresses a failure mode observed in real usage or a proven pattern from competitors.

### 1.1 Proactive Memory Flush (Write-Before-Compact)

**Problem:** Our current compaction (`context.py`) calls `_flush_to_memory` during the compact cycle — if the LLM call fails or misses facts, information is silently lost. OpenClaw runs a dedicated silent flush turn BEFORE compaction triggers, ensuring durable memory.

**Design — improve on OpenClaw:**

OpenClaw flushes to flat `MEMORY.md`. We go further: flush to **structured memory categories** (see 1.5 below) so flushed facts are immediately searchable, not just appended text.

```
Context at 60% → trigger proactive flush
  1. Agent loop detects threshold (new FLUSH_THRESHOLD = 0.6)
  2. Injects silent system message: "Summarize key facts from this conversation"
  3. Response written to structured memory (categories + items)
  4. NO_REPLY flag suppresses user-visible output
  5. Continue normally — compaction at 70% now discards safely
Context at 70% → standard compaction (safe — facts already saved)
```

**Changes:**

| File | What |
|---|---|
| `src/agent/context.py` | Add `FLUSH_THRESHOLD = 0.6`. New `proactive_flush()` method that runs a silent LLM extraction turn. Returns structured facts (key, value, category). Called from `maybe_compact()` when between 60-70%. |
| `src/agent/loop.py` | After each tool round in chat mode, call `context_manager.maybe_compact()` (already happens). The flush is transparent to the loop. |
| `src/agent/memory.py` | Add `store_facts_batch(facts: list[dict])` for efficient multi-fact insertion from flush. |
| `src/agent/workspace.py` | `flush_to_memory()` writes both to MEMORY.md (human-readable) AND calls `store_facts_batch()` (searchable). |

**Why this is better than OpenClaw:** They flush to flat files. We flush to structured, searchable memory with categories. Flushed facts are immediately retrievable by the agent, not buried in a growing text file.

### 1.2 Bootstrap Content Capping + On-Demand Logs

**Problem:** We inject workspace content (MEMORY.md, daily logs, learnings, AGENTS.md, SOUL.md) into EVERY system prompt. No size limits. A user with a 50KB MEMORY.md wastes ~12K tokens before the agent even starts. OpenClaw caps at 150K chars total / 20K per file and makes daily logs tool-accessible only.

**Design — improve on OpenClaw:**

OpenClaw uses static char caps. We use **relevance-based injection**: only inject workspace content that's relevant to the current conversation.

```
Always injected (capped):
  - AGENTS.md (max 8K chars) — operating instructions
  - SOUL.md (max 4K chars) — identity
  - USER.md (max 4K chars) — user preferences
  - MEMORY.md (max 16K chars) — curated long-term facts

Never injected (on-demand via tools):
  - Daily logs (memory/YYYY-MM-DD.md)
  - Learnings (errors.md, corrections.md)
  - Old workspace files

Total bootstrap cap: 40K chars (~10K tokens)
```

**Changes:**

| File | What |
|---|---|
| `src/agent/workspace.py` | Add `MAX_BOOTSTRAP_CHARS = 40_000` and per-file caps. New `get_bootstrap_content() -> str` that enforces limits, truncating with `... (truncated, use memory_search for full content)`. |
| `src/agent/loop.py` | Replace direct workspace file reads in `_build_chat_system_prompt()` with `workspace.get_bootstrap_content()`. Remove daily log injection. |
| `src/agent/builtins/memory_tool.py` | Enhance `memory_search` to also search workspace files (daily logs, learnings). Agent can pull specific logs when needed. |

**Why this is better than OpenClaw:** They use static char caps. We use smart caps with a fallback path — the agent can always pull full content via tools if the truncated bootstrap isn't enough.

### 1.3 Reference-Based Browser + Accessibility Tree (Combined)

**Problem:** Our agent guesses CSS selectors and writes fragile JavaScript. This is the single biggest efficiency gap — browser tasks use 2-3x more tool rounds than necessary. Previously listed as two separate items; combining them since the accessibility tree IS the source for references.

**Design:**

```
browser_snapshot() → Playwright accessibility tree → structured refs
  [e1] button "Sign In"
  [e2] input[type=email] "Email address"
  [e3] link "Forgot password?"
  [e4] heading "Welcome back"

browser_click(ref="e1")  ← direct ref, no CSS guessing
browser_type(ref="e2", text="user@example.com")
```

**Implementation:**

| File | What |
|---|---|
| `src/agent/builtins/browser_tool.py` | New `browser_snapshot` tool using `page.accessibility.snapshot()` (Playwright built-in). Returns structured list with refs. Store ref→locator mapping in module-level dict, cleared on navigation. Update `browser_click` and `browser_type` to accept optional `ref` param — if provided, use stored locator instead of CSS selector. |
| `src/agent/loop.py` | Update system prompt: "Always call browser_snapshot after navigating to see page structure. Use refs (e1, e2, ...) in click/type actions instead of CSS selectors." |

**Ref storage:**
```python
_page_refs: dict[str, Locator] = {}  # "e1" → Playwright Locator

async def browser_snapshot(...) -> dict:
    tree = await page.accessibility.snapshot()
    _page_refs.clear()
    elements = _flatten_tree(tree)  # recursive walk
    for i, elem in enumerate(elements):
        ref = f"e{i+1}"
        _page_refs[ref] = page.get_by_role(elem["role"], name=elem["name"])
    return {"elements": [{"ref": f"e{i+1}", **elem} for i, elem in enumerate(elements)]}
```

**Expected impact:** ~50% fewer browser tool rounds. Eliminates CSS selector guessing and JS injection trial-and-error.

### 1.4 Model Failover Chain

**Problem:** We retry the same model 3 times with exponential backoff. If the model is down, rate-limited, or the API key is exhausted, the agent is dead. OpenClaw does two-stage failover with auth rotation and model fallback.

**Design — improve on OpenClaw:**

OpenClaw's failover is per-provider. We make it per-agent with user-configurable chains:

```yaml
# mesh.yaml
defaults:
  model: anthropic/claude-sonnet-4-5-20250929
  fallbacks:
    - anthropic/claude-haiku-4-5-20251001   # cheaper, faster
    - openai/gpt-4.1-mini                    # different provider
  fallback_strategy: cascade  # cascade | round-robin
```

**Changes:**

| File | What |
|---|---|
| `src/host/credentials.py` | Add `ModelFailover` class. Tracks per-model health (success/failure counts, cooldown timestamps). `get_model(requested) -> str` returns the requested model if healthy, else walks the fallback chain. Billing errors (402, 429 with long retry-after) get 1h cooldown; transient errors (500, 502, 503) get 1m→5m→25m exponential cooldown. |
| `src/host/server.py` | LLM proxy endpoint checks `model_failover.get_model()` before forwarding. Logs fallback decisions. |
| `src/agent/loop.py` | No changes needed — failover is transparent at the proxy layer. |
| Config | Add `defaults.fallbacks` to mesh.yaml schema. |

**Why this is better than OpenClaw:** Their failover is provider-side only. Ours is mesh-level — works across providers, respects per-agent budgets, and logs every fallback decision for `/costs` visibility.

### 1.5 Hierarchical Memory with Categories (MemU-Inspired)

**Problem:** Our current memory is flat — all facts in one SQLite table, searched by vector+BM25. Works for small memories but degrades as facts accumulate. MemU uses a three-layer hierarchy (categories → items → resources) with LLM-driven categorization.

**Design — take the best of MemU, keep our simplicity:**

We don't need MemU's full workflow system or PostgreSQL. We add a lightweight category layer on top of our existing SQLite store.

```
Categories (auto-organized by LLM):
  "user_preferences" → [fact1, fact2, fact3]
  "project_afkcrypto" → [fact4, fact5]
  "browser_patterns" → [fact6, fact7]

Search flow:
  1. Vector search on category summaries (fast, top-3 categories)
  2. Vector+BM25 search within matched categories (focused, high precision)
  3. Sufficiency check: "Do these results answer the query?" If yes, stop. If no, expand to all categories.
```

**Schema additions to `memory.py`:**
```sql
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,          -- LLM-generated description
    embedding BLOB,                  -- vector for category-level search
    item_count INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Add category_id to existing facts table
ALTER TABLE facts ADD COLUMN category_id INTEGER REFERENCES categories(id);
```

**Changes:**

| File | What |
|---|---|
| `src/agent/memory.py` | Add `categories` table. New `_auto_categorize(fact) -> int` calls LLM to assign category (or creates new one). Update `store_fact()` to auto-categorize. New `search_hierarchical(query)`: search categories first, then items within top categories, with sufficiency check. Update category summaries incrementally when facts change (MemU's patch pattern). |
| `src/agent/loop.py` | Replace `memory.search()` calls with `memory.search_hierarchical()` in chat mode. Task mode keeps flat search (simpler, faster for short tasks). |

**Retrieval with sufficiency checking (MemU's best idea):**
```python
async def search_hierarchical(self, query: str, max_results: int = 5) -> list:
    # Tier 1: Find relevant categories
    cat_results = await self._search_categories(query, top_k=3)

    # Tier 2: Search within those categories
    facts = await self._search_within_categories(query, cat_results, max_results)

    # Sufficiency check: do we have enough?
    if len(facts) >= max_results or not cat_results:
        return facts

    # Tier 3: Expand to all categories if insufficient
    all_facts = await self.search(query, max_results)  # existing flat search
    return _merge_deduplicate(facts, all_facts)[:max_results]
```

**Why this is better than MemU:** They require LLM calls at every retrieval tier (expensive). We use vector search for tier 1-2 (fast) and only fall back to LLM ranking when results are ambiguous. We also keep the flat search as a fallback — categories are an optimization, not a requirement.

**Reinforcement tracking (adopt from MemU):**
- Already have `access_count` and `decay_score` in our facts table
- Add `last_accessed` timestamp (we have this)
- Boost recently-accessed facts in search ranking: `score *= 1 + log(access_count) * recency_factor`
- This replaces our current static `SALIENCE_BOOST = 1.5` with a continuous function

---

## Tier 2: High Value, Medium Effort

### 2.1 Message Queue Modes (Steer / Collect / Followup)

**Problem:** Our `LaneManager` is strict FIFO. If an agent is mid-task (2-minute browser operation), all messages queue silently. User has no way to redirect or cancel. OpenClaw supports mid-stream message injection.

**Design:**

| Mode | Behavior | Use Case |
|---|---|---|
| **followup** (default) | Queue message, process after current task | Normal sequential chat |
| **steer** | Inject into current running context | "Stop, do this instead" |
| **collect** | Batch multiple queued messages into one turn | Rapid-fire user messages |

**Changes:**

| File | What |
|---|---|
| `src/host/lanes.py` | Add `mode` param to `enqueue()`. Add `_steer_message` field to `QueuedTask`. For steer mode: if worker is active, inject message into agent's conversation via transport POST `/chat/steer`. For collect mode: coalesce queued messages before dispatching. |
| `src/agent/server.py` | New `POST /chat/steer` endpoint that appends a user message to the current conversation mid-stream. Sets a flag that the loop checks after each tool round. |
| `src/agent/loop.py` | Check for steered messages after each tool execution in chat mode. If present, append to messages and continue the loop with new context. |
| `src/channels/telegram.py` | Messages sent while agent is working use steer mode. Shows "Message received, redirecting..." feedback. |

### 2.2 Browser Tiers (Stealth + Advanced)

**Problem:** Basic Playwright + headless Chromium is detected and blocked by most bot detection (Cloudflare, DataDome, PerimeterX). Need tiered backends.

| Tier | Stack | Cost | Anti-Detection |
|---|---|---|---|
| **Basic** (default) | Playwright + Chromium (current) | Free | Low |
| **Stealth** | [Camoufox](https://github.com/nichochar/open-operator) + fingerprint evasion | Free | Medium |
| **Advanced** | [Bright Data Scraping Browser](https://brightdata.com/products/scraping-browser) via CDP | Paid | High — residential proxies, CAPTCHA solving |

**Implementation:**
- Add `browser_backend` to `config/mesh.yaml`
- `openlegion setup` step: "Choose browser backend"
- `browser_tool.py:_get_page()` reads config, launches appropriate backend
- Bright Data connects via CDP URL (no local install): `browser.connect_over_cdp(brightdata_url)`
- Credential vault stores Bright Data API key

### 2.3 SOUL.md Persona Injection

**Problem:** Agents have generic personalities. OpenClaw already supports SOUL.md for agent identity customization. Users expect this.

**Design:**
- `SOUL.md` in project root → injected into all agents' chat system prompt
- Per-agent override: `skills/{agent_name}/SOUL.md`
- Subject to bootstrap cap (max 4K chars)
- `openlegion setup` optional step: "Describe your agent's personality"

**Changes:**
- `workspace.py`: Load SOUL.md in `get_bootstrap_content()`
- `loop.py`: Already handled via bootstrap injection from 1.2

### 2.4 Silent Reply Token

**Problem:** Heartbeat/cron acknowledgments generate filler text ("Everything looks good!"), wasting one LLM round-trip and cluttering channels.

**Design:**
- Define `SILENT_REPLY_TOKEN = "__SILENT__"`
- System prompt: "When you have nothing substantive to say (e.g., heartbeat check with no issues), return only `__SILENT__`"
- Agent loop detects token, returns empty response
- Channels suppress empty responses
- Saves ~$0.01 per heartbeat (adds up with frequent cron)

### 2.5 Provider-Specific Transcript Hygiene

**Problem:** Different LLM providers have strict, incompatible requirements for tool-call message formatting. We rely on litellm but our `_trim_context` can produce sequences that violate provider rules. OpenClaw sanitizes per-provider.

**Rules by provider:**

| Provider | Requirements |
|---|---|
| **Anthropic** | Strict user/assistant alternation. Tool calls must pair with results. No orphaned tool_use blocks. |
| **OpenAI** | More lenient. Handles orphaned calls. Image format restrictions. |
| **Google/Gemini** | Alphanumeric-only tool IDs (no hyphens/underscores). Strict alternation. |
| **Mistral** | 9-char alphanumeric tool IDs. |

**Changes:**
- `src/agent/loop.py`: Add `_sanitize_for_provider(messages, provider)` called before every LLM call
- Repairs: merge consecutive same-role messages, ensure tool-call pairing, fix tool IDs
- Applied in-memory only — stored transcript unchanged

### 2.6 Token Estimation + Proactive Compaction

**Problem:** Our 4-chars-per-token estimate is rough. We trigger compaction reactively at 70%. Should estimate accurately and compact proactively.

**Design:**
- Use `tiktoken` for OpenAI models, character-based estimate for others
- Show token usage in `/costs`: "Context: 45K/128K tokens (35%)"
- Trigger proactive flush (1.1) at 60%, compaction at 70%
- System prompt warns agent when approaching limit: "Context is 80% full. Wrap up or save important facts."

---

## Tier 3: Polish + Future

### 3.1 CLI God-Function Refactor

`cli.py:_start_interactive` is ~200 lines. Split into:
- `RuntimeContext` — shared state (transport, router, orchestrator)
- `REPLSession` — input/output, command dispatch
- `ChannelManager` — channel lifecycle

### 3.2 Labeled Screenshots

When visual context is needed (canvas elements, complex layouts):
- `browser_screenshot(labeled=True)` → overlay numbered labels on interactive elements
- Viewport culling: only label visible elements
- Requires multimodal LLM support (Claude, GPT-4V)

### 3.3 Request Tracing

Trace IDs on all mesh requests:
- Generate trace ID at dispatch entry point
- Propagate through transport → agent → LLM proxy → response
- `/debug <trace_id>` shows request lifecycle

### 3.4 Cross-Channel Broadcasting

Mirror agent responses across channels:
- PubSub topic per-agent for response events
- Channels subscribe to agents they're interested in
- Opt-in per channel in `mesh.yaml`

### 3.5 Prompt Injection Sanitization

Strip Unicode control/format characters from runtime strings before prompts:
- `sanitize_for_prompt(text)` utility
- Apply to: user messages, blackboard data, workspace content, tool outputs

### 3.6 Cron-Triggered Subagents

Cron jobs spawn subagents instead of blocking main agent:
- `spawn: true` flag in cron config
- Main agent stays available for chat
- Results announced when done

### 3.7 Skill Marketplace

Community-contributed skill packages:
- Skills as git repos with `SKILL.md` metadata
- `openlegion skill install <repo>` CLI command
- Version pinning, sandboxed execution

### 3.8 Subagent Spawning (In-Container)

Delegate long-running tasks to child sessions within the same container:
- `spawn_subagent` built-in tool
- Isolated conversation context, shared filesystem
- Results auto-posted back to parent
- Depth limit (max 3)

---

## Completed

### Session 2 (Runtime Hardening)
- [x] httpx client lifecycle (per-loop client pooling)
- [x] Push-based orchestrator (asyncio.Future, no polling)
- [x] LLM retry with exponential backoff
- [x] Skill registry isolation fix
- [x] PubSub SQLite persistence
- [x] Concurrent agent startup
- [x] SSE streaming (agent → transport → CLI)
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

| Capability | OpenLegion | OpenClaw | MemU |
|---|---|---|---|
| Container isolation | Docker + Sandbox | Local process | N/A (memory only) |
| Browser automation | Playwright (CSS selectors) | Playwright (integrated) | N/A |
| Reference-based browser | Planned (1.3) | Not explicit | N/A |
| Memory architecture | Flat SQLite + vec + BM25 | File-based (MEMORY.md) | Hierarchical categories |
| Memory categories | Planned (1.5) | None | Yes (LLM-driven) |
| Sufficiency checking | Planned (1.5) | None | Yes (multi-tier) |
| Context compaction | Reactive (70%) | Write-before-compact | N/A |
| Proactive flush | Planned (1.1) | Yes | N/A |
| Model failover | Planned (1.4) | Yes (two-stage) | N/A |
| Message steering | Planned (2.1) | Yes (steer/collect) | N/A |
| Multi-agent | Fleet model (mesh) | Single agent + bindings | N/A |
| Streaming | SSE (agent → CLI/TG) | Three event streams | N/A |
| Cost tracking | Per-agent SQLite | Provider API | N/A |
| Bootstrap capping | Planned (1.2) | 150K chars | N/A |

**Our edge after Tier 1:** Hierarchical memory with sufficiency checking (better than both), reference-based browser (unique), container isolation (unique), fleet coordination (unique), proactive flush to structured memory (better than OpenClaw's flat files).
