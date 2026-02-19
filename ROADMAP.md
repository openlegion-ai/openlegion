# OpenLegion Roadmap

Prioritized improvements discovered during runtime hardening and competitive research (OpenClaw, OpenHands). Ordered by impact and dependency.

---

## Tier 1: High Impact, Near-Term

### Browser Tiers

Currently all agents use basic Playwright + headless Chromium in the container. Many sites detect and block headless browsers. Add tiered browser backends selectable during `openlegion setup`:

| Tier | Stack | Cost | Anti-Detection |
|---|---|---|---|
| **Basic** (default) | Playwright + Chromium (current) | Free | Low — blocked by most bot detection |
| **Stealth** | [Steelbrowser](https://github.com/nichochar/open-operator) + [Camoufox](https://github.com/nichochar/open-operator) | Free | Medium — fingerprint evasion, realistic headers |
| **Advanced** | [Bright Data Scraping Browser](https://brightdata.com/products/scraping-browser) | Paid | High — residential proxies, CAPTCHA solving, unlocker |

**Implementation:**
- Add `browser_backend` field to `config/mesh.yaml` (`basic` | `stealth` | `brightdata`)
- `openlegion setup` Step 1b: "Choose browser backend" with descriptions
- `Dockerfile.agent` installs Camoufox when stealth is selected
- `browser_tool.py:_get_page()` reads config and launches the appropriate backend
- Bright Data connects via CDP to their remote browser (no local install)
- Credential vault handles Bright Data API keys

### Reference-Based Browser Interaction

The biggest efficiency gap vs OpenClaw. Our agent guesses CSS selectors and writes fragile JavaScript. Their agent gets labeled refs and uses them directly.

**Design:**
- New `browser_snapshot` tool → returns structured page elements with refs (`e1`, `e2`, `e3`)
- Uses Playwright's accessibility tree or `_snapshotForAI()` (Playwright 1.40+)
- Refs are stored per-page, reused across actions
- Update `browser_click` and `browser_type` to accept refs: `browser_click(ref="e5")` alongside CSS selectors
- System prompt updated: "Use browser_snapshot to see the page structure, then use refs in click/type actions"

**Expected impact:** ~50% fewer browser tool rounds, eliminates JS injection trial-and-error

### Subagent Spawning

Currently `LaneManager` queues messages per-agent serially. If an agent is doing a 2-minute browser task, chat messages queue behind it. OpenClaw solves this by spawning background subagents.

**Design:**
- New `spawn_subagent` built-in tool: agent can delegate a task to a child session
- Child runs in the same container but with its own conversation context
- Results auto-posted back to parent's conversation
- Depth limit (max 3) to prevent runaway spawning
- Parent agent stays responsive to chat while child works
- Add `sessions_list` tool to check subagent status

**Architecture:**
```
User message → Agent (chat mode, responsive)
                 ↓ spawn_subagent("create protonmail account")
               Subagent (background, isolated context)
                 ↓ result
               Agent receives result, reports to user
```

### Response Display UX

Current CLI output is minimal (tool names + `...`). Telegram gets raw markdown. Need structured, readable output across all surfaces.

**CLI improvements:**
- Rich tool output: show tool name, key input params, and abbreviated result on one line
- Progress indicator: `[3/30 tools]` counter so user knows how far along the agent is
- Collapsible sections: tool details hidden by default, expandable
- Color coding: tool calls in dim, errors in red, final response in bold
- Fix REPL input/output interleaving (proper async input with readline or prompt_toolkit)

**Telegram improvements:**
- Structured message formatting: use Telegram's native formatting (bold, code blocks, lists)
- Tool progress: send a single "working..." message, edit it as tools execute (Telegram `editMessageText`)
- Final response as a clean separate message
- Inline buttons for common actions (`/status`, `/reset`)

**Discord improvements:**
- Embed-based responses with fields for tool outputs
- Thread-based: long tool chains in a thread, final answer in channel

---

## Tier 2: Medium Impact

### Accessibility Tree Extraction

Complement the reference-based browser with CDP-based accessibility tree for richer page understanding.

- Establish CDP session alongside Playwright page
- Call `Accessibility.getFullAXTree` for structured DOM view
- Return interactive elements (buttons, inputs, links) with roles, names, and states
- Cap at 2000 nodes to fit context window
- Falls back to `inner_text("body")` on CDP failure

### Labeled Screenshots

When the LLM needs visual context (complex layouts, canvas elements), overlay numbered labels on interactive elements.

- `browser_screenshot(labeled=True)` → annotates up to 150 elements with ref numbers
- Viewport culling: only label visible elements
- Requires multimodal LLM support (already works with Claude, GPT-4V)

### Silent Reply Token

When the agent has nothing substantive to say (e.g., acknowledging a heartbeat), return a special token instead of generating filler text. Saves one LLM round-trip.

- Define `SILENT_REPLY_TOKEN = "__SILENT__"`
- Agent loop detects it and skips the response
- Channels suppress the message
- System prompt: "When you have nothing to add, return `__SILENT__` instead of filler"

### SOUL.md Persona Injection

Let users define agent personality via a markdown file, injected into the system prompt.

- `SOUL.md` in project root → injected into chat system prompt
- Per-agent override: `skills/{agent}/SOUL.md`
- Supports personality traits, communication style, domain expertise
- `openlegion setup` optional step: "Describe your agent's personality"

### CLI God-Function Refactor

`cli.py:_start_interactive` is a ~200-line function that sets up everything. Split into composable components:

- `RuntimeContext` class: holds all shared state (transport, router, orchestrator, etc.)
- `REPLSession` class: handles input/output, command dispatch
- `ChannelManager` class: manages channel lifecycle
- Each with clear `start()` / `stop()` lifecycle

### Token Estimation

Before sending to the LLM, estimate token count to:
- Warn when approaching context limits
- Trigger compaction proactively (not reactively)
- Show in `/costs` output

---

## Tier 3: Lower Priority / Future

### Request Tracing

Add trace IDs to all mesh requests for debugging:
- Generate trace ID at dispatch entry point
- Propagate through transport → agent → LLM proxy → response
- Log with trace ID for correlation
- `/debug <trace_id>` command to show request lifecycle

### Cross-Channel Broadcasting

When an agent responds to a Telegram message, optionally mirror the response to CLI and other channels.
- `PubSub` topic per-agent for response events
- Channels subscribe to agents they're interested in
- Configurable: opt-in per channel in `mesh.yaml`

### Prompt Injection Sanitization

Strip Unicode control characters and format characters from runtime strings before embedding in prompts (OpenClaw pattern).
- `sanitize_for_prompt(text)` utility
- Apply to: user messages, blackboard data, workspace content, tool outputs
- Prevents injection via crafted filenames, directory names, or tool responses

### Cron-Triggered Subagents

Allow cron jobs to spawn subagents instead of blocking the main agent:
- Cron config: `spawn: true` flag
- Spawns a subagent session for the cron task
- Main agent stays available for chat
- Results announced when done

### Skill Marketplace

Community-contributed skill packages:
- Skills as git repos with `SKILL.md` metadata
- `openlegion skill install <repo>` CLI command
- Version pinning and dependency resolution
- Sandboxed execution (skills run inside agent container)

---

## Completed (this session)

- [x] httpx client lifecycle (lazy singleton + close)
- [x] Push-based orchestrator (asyncio.Future, no polling)
- [x] LLM retry with exponential backoff
- [x] Skill registry isolation fix
- [x] PubSub SQLite persistence
- [x] Concurrent agent startup
- [x] SSE streaming (agent → transport → CLI)
- [x] Telegram slash command routing
- [x] Thread-safe dispatch loop (fixes asyncio cross-loop errors)
- [x] litellm tools= param fix
- [x] Telegram HTML formatting
- [x] Playwright browser path fix (non-root access)
- [x] CHAT_MAX_TOOL_ROUNDS 10 → 30
- [x] "Don't narrate" system prompt directive
- [x] Browser efficiency prompt rules
- [x] Startup UX (pairing code at bottom)
