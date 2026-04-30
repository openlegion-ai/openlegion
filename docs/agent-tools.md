# Agent Tools Reference

Agents interact with their environment through **skills** -- Python functions registered via the `@skill` decorator. Skills are auto-discovered at startup from built-in modules and custom skill directories.

## Built-in Tools

### Shell Execution

| Tool | Parameters | Description |
|------|-----------|-------------|
| `run_command` | `command`, `workdir`, `timeout` | Execute shell commands with full Linux environment |

### File Operations

All file operations are scoped to `/data` inside the container. Path traversal is blocked.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `read_file` | `path`, `offset`, `limit` | Read file contents with optional pagination |
| `write_file` | `path`, `content`, `append` | Write or append to a file (creates directories) |
| `list_files` | `path`, `pattern`, `recursive` | List files with optional glob pattern matching (recursive: boolean, default false) |

### HTTP

| Tool | Parameters | Description |
|------|-----------|-------------|
| `http_request` | `url`, `method` (default "GET"), `headers` (default {}), `body` (default "", string), `timeout` (default 30) | Make HTTP requests (GET/POST/PUT/DELETE/PATCH). Supports `$CRED{name}` handles in URL, headers, and body for credential-blind API calls. Resolved credentials are redacted from responses. SSRF protection blocks requests to private/internal addresses (loopback, link-local, reserved ranges) including redirect targets. **Note:** `body` must be a string — serialize JSON manually before passing it (e.g. `json.dumps(data)`). |

### Browser Automation

All agents share a single **browser service container** running Camoufox (a stealth Firefox fork) with KasmVNC. The browser service runs separately from agent containers — agents send browser commands via HTTP to the shared service. KasmVNC serves a live browser view on port 6080 (accessible via the dashboard).

Each agent gets a deterministic viewport from a weighted pool (1920×1080 default) and an OS-level device profile selected by `BROWSER_DEVICE_PROFILE` (`desktop-windows` default, `desktop-macos`, `mobile-ios`, `mobile-android`). Mobile profiles deliberately pair a Safari-shaped UA with the Firefox engine — a known fingerprint mismatch retained for site-compatibility reasons. On-disk profiles persist (cookies, fonts, uBlock state); the newer `storage_state` cookie/origin sidecar (cross-restart cookie continuity) is opt-in via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default off).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `browser_navigate` | `url`, `wait_ms` (default 1000, capped at 10000), `wait_until` (`domcontentloaded` default / `load` / `networkidle` / `commit`), `snapshot_after` (default false), `referer` | Open URL, wait, extract page text. `referer` unset → service auto-picks; `""` → explicit no-referer; URL → caller override. Returns `{success, data: {url, title, body}, captcha?, snapshot?}` — `body` is a 1000-char preview when `snapshot_after=true`, 5000 chars otherwise. Auto-detects CAPTCHA after load and may auto-solve, charging cost (see CAPTCHA solving below); the `captcha` field appears only when a CAPTCHA was detected and not solved. |
| `browser_get_elements` | `filter` (`actionable` / `inputs` / `headings` / `landmarks`), `from_ref`, `diff_from_last` (default false), `frame` (URL substring or `f-xxxxxxxx` frame_id), `include_frames` (default true) | Accessibility tree snapshot with element refs (e1, e2, ...). Returns structured text, not a visual image. Capped at 200 elements; iframe nesting capped at 3 levels. `diff_from_last=true` returns only what changed since the prior snapshot. |
| `browser_screenshot` | `full_page` (default false), `format` (`webp` default / `png`), `quality` (1-100, default 75), `scale` (0.5-1.0, default 1.0) | Take screenshot, return visual image. Default format is **WebP**; falls back to PNG if WebP encoding fails. Exempt from loop detector. |
| `browser_click` | `ref` or `selector`, `force` (default false), `snapshot_after` (default false), `timeout_ms` (default 10000, max 30000), `frame` | Click element by accessibility ref or CSS selector. `force` bypasses actionability checks (auto-applied for disabled button/link refs). Modal close-button clicks fall back to Escape if the modal persists. Returns `{success, data: {ref, ...}, captcha?}` — auto re-detects CAPTCHA after click. |
| `browser_click_xy` | `x`, `y` (viewport-relative CSS pixels) | Coordinate-based click for canvas/SVG widgets without a11y nodes. Pre-checks the target via `elementFromPoint` and walks the ancestor chain — returns an `invalid_input` envelope with `actual_element` + `masked_by` when an overlay would intercept the click. Rejects bools, NaN/Inf, and out-of-viewport coords. |
| `browser_type` | `ref` or `selector`, `text`, `clear` (default true), `fast` (default false), `snapshot_after` (default false), `frame` | Type into input field. `clear=false` appends rather than replacing. `fast=true` sets the value directly without keystrokes. |
| `browser_hover` | `ref` or `selector` | Hover over an element to trigger dropdowns/tooltips. |
| `browser_scroll` | `direction` (default `down`), `amount` (default 0, max 10000 px), `ref` | Scroll page up/down or scroll element into view. `amount=0` is treated as one viewport height. |
| `browser_wait_for` | `selector`, `state` (`visible` default / `attached` / `hidden` / `detached`), `timeout_ms` (default 10000, max 30000) | Wait for a CSS selector to reach a state. |
| `browser_find_text` | `query` (1-500 chars), `scroll` (default true) | Case-insensitive text search across the current snapshot. Returns up to 50 matches in snapshot order; `scroll=true` auto-scrolls to the first match. Useful for recovering when refs change between snapshots. |
| `browser_fill_form` | `fields` (array of `{label, value, submit_after?}`, max 50), `submit_after` (default false) | Bulk fill multiple fields. Label max 500 chars; value max 10000 chars. **Partial-success protocol:** when a CAPTCHA appears mid-flow the tool returns `partial_success` with `submitted: bool` and `remaining` echoed verbatim so the caller can resume after solving. Per-field error codes: `not_fillable` / `timeout` / `detached` / `hidden` / `disabled` / `other`. |
| `browser_press_key` | `key` (≤50 chars) | Press a keyboard key or shortcut (e.g. `Escape`, `Enter`, `Control+a`). Maps Playwright key names to xdotool. |
| `browser_open_tab` | `url` (http/https only), `snapshot_after` (default false) | Open a new tab and navigate. Scheme is validated agent-side. |
| `browser_switch_tab` | `tab_index` (default -1) | List open tabs or switch to a specific tab. Default `-1` lists tabs without switching. |
| `browser_go_back` | -- | Navigate back in browser history. |
| `browser_go_forward` | -- | Navigate forward in browser history. |
| `browser_reset` | -- | Reset browser session (profile preserved). |
| `browser_inspect_requests` | `include_blocked` (default false), `limit` (default 50, max 200) | Read-only network request log. Per-entry: `url` (redacted), `method`, `resource_type`, `ts`, `status`, `blocked_by_adblock`, `user_cancelled`, `failed_network`. Buffer maxlen 200 per agent. Blocked by `BROWSER_NETWORK_INSPECT_DISABLED=true`. |
| `browser_upload_file` | `ref`, `paths` (1-5 items), `idempotency_key` | Upload local files to a file-input element. Two-stage protocol — agent stages bytes via mesh, browser receives by handle. Per-file cap 50 MB (`OPENLEGION_UPLOAD_STAGE_MAX_MB`), max 5 files per call. |
| `browser_download` | `ref`, `timeout_ms` (default 30000, max 180000) | Click a download trigger and ingest the resulting file as an artifact. Returns `{success, data: {artifact_name, size_bytes, mime_type}}`. Disabled fleet-wide by `BROWSER_DOWNLOADS_DISABLED=true`. |
| `browser_detect_captcha` | -- | Run CAPTCHA detection on the current page. Recognises reCAPTCHA v2 checkbox/invisible, v3, enterprise-v2/v3; hCaptcha; Cloudflare Turnstile + interstitial (auto/behavioral); PerimeterX press-hold; DataDome behavioral; and JS-challenge vendors (Akamai, Kasada, FingerprintJS, Imperva, F5). Usually not needed — `browser_navigate` and `browser_click` auto-detect. Returns the full CAPTCHA envelope (see below). |
| `browser_solve_captcha` | `hint`, `retry_previous` (default false), `target_ref` | Explicit-solve when auto-solve was skipped. `hint` must be a solvable kind from the detect list — behavioral kinds (`cf-interstitial-auto`, `cf-interstitial-behavioral`, `px-press-hold`, `datadome-behavioral`, `js-challenge-*`) are rejected; route those to `request_captcha_help` instead. `retry_previous=true` re-checks the prior solve once after 500 ms. `target_ref` is accepted for forward compatibility but currently logged and ignored. |
| `request_captcha_help` | `service` (≤128 chars), `description` (≤500 chars) | Emit a VNC handoff card asking the user to solve a CAPTCHA manually. Use for behavioral-only kinds (CF Under Attack, Press & Hold, JS-challenge vendors), persistent rejections, or after `rate_limited` / `cost_cap` outcomes from `browser_solve_captcha`. |
| `request_browser_login` | `url`, `service`, `description`, `agent_id` (default = self) | Navigate the browser to a login page and emit a VNC login card prompting manual login. `agent_id` lets an operator agent push a login into another agent's profile — cookies persist in the target's profile. |

**CAPTCHA solving.** `browser_navigate` and `browser_click` auto-detect CAPTCHAs and may invoke the configured solver (`CAPTCHA_SOLVER_PROVIDER` + `CAPTCHA_SOLVER_KEY`, env-only — these bypass the `OPENLEGION_CRED_*` vault). All envelopes carry `{captcha_found, kind, solver_attempted, solver_outcome, solver_confidence, next_action}`. `solver_outcome` values: `solved` / `timeout` / `rejected` / `injection_failed` / `no_solver` / `unsupported` / `skipped_behavioral` / `rate_limited` / `cost_cap` / `captcha_during_solve`. Solves are rate-limited per agent (`CAPTCHA_RATE_LIMIT_PER_HOUR`, default 20) and cost-capped per agent (`CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH`) and per tenant (`CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH`). All cost values are stored in **millicents (1/100,000 USD)**, not cents — `PRICING_CENTS` is a back-compat alias for `PRICING_MILLICENTS`.

**Fingerprint burn.** After every solve, the service polls vendor-specific rejection selectors (Cloudflare 1xxx, DataDome, PerimeterX, Imperva, Akamai BMP, F5) plus branded rejection text for 10s. When ≥50% of the last 10 outcomes are rejections, subsequent envelopes carry `fingerprint_burn=True` and `next_action="retry_with_fresh_profile"`. **There is no auto-rotation** — operators clear the burn manually via `POST /api/agents/{id}/fingerprint-health/reset` after rotating the profile.

**Permission gating.** `AgentPermissions.browser_actions: list[str] | None` controls per-action access. `None` (default) or `["*"]` allows all; a specific list opts in to only listed actions; `[]` denies everything (equivalent to `can_use_browser=False`). Valid action names live in `KNOWN_BROWSER_ACTIONS` (26 actions covering all `browser_*` tools plus `request_browser_login` and `request_captcha_help`). `import_cookies` is **not** in `KNOWN_BROWSER_ACTIONS` — it is operator-only.

**Operator kill switches.** `BROWSER_DOWNLOADS_DISABLED`, `BROWSER_NETWORK_INSPECT_DISABLED`, `BROWSER_COOKIE_IMPORT_DISABLED`, and `CAPTCHA_DISABLED` flip the corresponding tools to a forbidden envelope at runtime.

**Note:** All browser tools are `parallel_safe=False`. Subagents share the parent agent's BrowserManager (`inst.lock`), so the constraint is per `agent_id`, not per Python process — subagents under the same parent should not attempt concurrent browser operations.

### Memory

| Tool | Parameters | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `category` (default ""), `max_results` (default 5) | Hybrid search across workspace files and the structured memory DB. Uses vector similarity (sqlite-vec) with FTS5 full-text fallback when the embedding model is unavailable. Provide `category` to restrict the search to the fact database filtered to that category. Returns `{results: [{key, value, category, confidence, access_count, source}], count}`. |
| `memory_save` | `content` | Save a fact to workspace daily log and structured memory |

### Mesh / Fleet

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_agents` | -- | Discover agents in your project (standalone agents see only themselves) |
| `read_blackboard` | `key` | Read from the project-scoped blackboard. Keys are auto-namespaced under `projects/{name}/` — agents use natural keys like `tasks/abc` |
| `write_blackboard` | `key`, `value` | Write a JSON value to the project-scoped blackboard |
| `list_blackboard` | `prefix` (default "") | Browse project blackboard entries by prefix. Returns key names, authors, timestamps, and 200-char value previews. |
| `publish_event` | `topic`, `data` | Publish event to mesh pub/sub |
| `subscribe_event` | `topic` | Subscribe to a pub/sub topic at runtime. Events arrive as steer messages between tool rounds. |
| `watch_blackboard` | `pattern` | Watch blackboard keys matching a glob pattern. Notifications arrive when matching keys are written. |
| `claim_task` | `key`, `claim_value` | Atomically claim a task from the blackboard (compare-and-swap). Prevents duplicate work. |
| `save_artifact` | `name`, `content`, `description` (default "") | Save deliverable to workspace and register on project blackboard at `artifacts/{agent_id}/{name}` |
| `notify_user` | `message` | Send a notification to the user across all connected channels (CLI, Telegram, Discord, Slack, etc.) |
| `read_agent_history` | `agent_id` | Read another agent's conversation logs (permission-checked) |
| `get_agent_profile` | `agent_id` | Read an agent's collaboration interface, inputs, outputs, and current status |

**Project isolation:** Blackboard tools (`read_blackboard`, `write_blackboard`, `list_blackboard`, `save_artifact`) are only available to agents assigned to a project. Standalone agents cannot access the blackboard — calls return an error explaining they must be added to a project first.

**Project context:** Agents assigned to a project automatically receive a `PROJECT.md` file mounted read-only in their workspace. This file contains the project description and shared context, visible to the agent from its first turn without any tool call.

### Workspace

| Tool | Parameters | Description |
|------|-----------|-------------|
| `update_workspace` | `filename`, `content` | Update a writable workspace file (`SOUL.md`, `INSTRUCTIONS.md`, `USER.md`, `HEARTBEAT.md`, or `INTERFACE.md`) to persist learnings across sessions |

### Scheduling & Automation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `set_cron` | `schedule`, `tool_name` (default ""), `tool_params` (default "{}"), `message` (default ""), `heartbeat` (default false) | Schedule a recurring job (cron expression or interval). Three modes: **tool-mode** — set `tool_name` (and optionally `tool_params` as a JSON string) to invoke a tool directly on each tick without any LLM involved; **message-mode** — set `message` to dispatch the text to the LLM on each tick; **heartbeat-mode** — set `heartbeat=true` to update your autonomous wakeup schedule. `tool_name` and `message` are mutually exclusive. |
| `list_cron` | -- | List scheduled jobs |
| `remove_cron` | `job_id` | Remove a scheduled job |

### Web Search

| Tool | Parameters | Description |
|------|-----------|-------------|
| `web_search` | `query`, `max_results` | Search via DuckDuckGo (no API key needed) |

### Self-Extension

| Tool | Parameters | Description |
|------|-----------|-------------|
| `create_skill` | `name`, `code` | Write a new Python skill at runtime |
| `reload_skills` | -- | Hot-reload all skills from disk |
| `spawn_fleet_agent` | `role`, `system_prompt`, `ttl` | Spawn an ephemeral sub-agent in a new container (default TTL: 3600s) |

### Subagents (In-Container)

Lightweight subagents that run inside the same process as the parent agent, sharing LLM and mesh clients but with their own memory and workspace.

**Limits:** Max 3 concurrent subagents, max depth 2 (no grandchildren), default TTL 300s (max TTL 600s), max 10 iterations per subagent. Subagents cannot use `create_skill`, `reload_skills`, `spawn_subagent`, or `wait_for_subagent` (prevents recursion and nesting). Results are written to blackboard at `subagent_results/{parent_id}/{subagent_id}`.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `spawn_subagent` | `task`, `role`, `ttl_seconds` | Spawn a lightweight subagent for parallel subtask execution |
| `list_subagents` | -- | List active subagents spawned by this agent and their status |
| `wait_for_subagent` | `subagent_id`, `timeout` | Wait for a subagent to complete and return its result |

### Credential Vault

Agents never see credential values. All operations return opaque `$CRED{name}` handles. Agents can only access credentials allowed by their `allowed_credentials` patterns in `config/permissions.json`. System credentials (LLM provider API keys) are never accessible to agents.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_generate_secret` | `name`, `length` (default 32), `charset` (default "urlsafe"; values: `urlsafe` / `hex` / `alphanumeric`) | Generate a random secret and store it (returns opaque `$CRED{name}` handle only — the actual value is never returned) |
| `vault_list` | -- | List credential names the agent can access (names only, filtered by permissions) |
| `request_credential` | `name`, `description`, `service` (default "") | Ask the user to supply a credential via the dashboard. The value is stored in the vault and the agent receives a `$CRED{name}` handle. Use this when a service requires a key the user must provide manually. |

### Wallet

Agents can interact with EVM and Solana blockchains through the wallet signing service. Private keys never enter agent containers — the mesh holds the master seed and signs transactions server-side. Wallet access requires `can_use_wallet: true` in `config/permissions.json`.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `wallet_get_address` | `chain` | Returns your wallet address on a specific chain. `chain`: one of `evm:ethereum`, `evm:base`, `evm:arbitrum`, `evm:polygon`, `evm:sepolia`, `solana:mainnet`, `solana:devnet`. |
| `wallet_get_balance` | `chain`, `token` | Check wallet balance in human-readable form. `token`: `native` (default) for ETH/SOL, or a token contract address for tokens like USDC. |
| `wallet_read_contract` | `chain`, `contract`, `function`, `args` | Read onchain data without sending a transaction. EVM: call a contract read function (e.g. `balanceOf(address)`). Solana: read account data. `function` (EVM only): Solidity function signature. `args` (EVM only): array of string arguments matching the signature. |
| `wallet_transfer` | `chain`, `to`, `amount`, `token` | Send tokens to an address — ETH, SOL, USDC, or any token. `to`: recipient address. `amount`: decimal string (e.g. `'0.1'`). `token`: `native` (default) or token contract address. For complex operations use `wallet_execute`. |
| `wallet_execute` | `chain`, `contract`, `function`, `args`, `value`, `transaction` | Call a smart contract or sign a protocol transaction (swaps, approvals, mints, staking, lending). EVM: provide `contract`, `function` (Solidity signature), `args`, and optional `value` (native token to send, default `'0'`). Solana: provide `transaction` (base64 unsigned tx from a protocol API). |

### System Introspection

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_system_status` | `section` (default "all"; values: `permissions` / `budget` / `fleet` / `cron` / `health` / `all`) | Query live runtime state |

Agents receive system awareness through three layers:

1. **SYSTEM.md** — Generated at startup and refreshed every 5 minutes. Contains a static architecture guide (mesh concepts, context window mechanics, tool cost model, common errors) plus a compact snapshot of permissions and fleet. Loaded into the system prompt via workspace bootstrap.
2. **Runtime Context** — A compact block injected into the system prompt on each turn (5-minute cache). Shows live budget numbers, permission patterns, fleet roster, and cron schedule.
3. **`get_system_status` tool** — On-demand access to fresh data when agents need exact numbers mid-conversation.

The `section` parameter accepts: `permissions`, `budget`, `fleet`, `cron`, `health`, or `all` (default). Fleet data is filtered by `can_message` permissions — agents only see teammates they can interact with.

### Multi-Agent Coordination

Higher-level tools from `coordination_tool.py` that implement a structured work-handoff protocol over the blackboard. Prefer these over raw blackboard writes for inter-agent task management.

Protocol layout: `tasks/{agent_id}/{id}` for inboxes, `output/{agent_id}/{id}` for output data, `status/{agent_id}` for state. Handoff TTL is 24 hours.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `hand_off` | `to`, `summary`, `data` (default "", JSON string) | Send work to another agent. Validates the target agent ID, writes a task entry to the target's inbox, and wakes the target agent. |
| `check_inbox` | -- | Read your own task inbox. Returns `{tasks: [{key, from, summary, status, output_key?, ts?}], count}`. Completed (`status == "done"`) tasks are omitted. |
| `update_status` | `state` (enum: `idle` / `working` / `blocked` / `done`), `summary` (default "") | Broadcast your current state to the `status/{agent_id}` blackboard key so teammates can see what you're doing. |
| `complete_task` | `task_key` | Mark a task done and remove it from the blackboard. Ownership check enforced — you can only complete tasks in your own inbox (`tasks/{own_id}/...`). |

### Image Generation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `generate_image` | `prompt`, `size` (default "square"; values: `square` / `landscape` / `portrait`), `filename` (default ""), `provider` (default "", values: `gemini` / `openai`) | Generate an image from a text prompt. Routes through the mesh credential proxy — agents never hold API keys. Saves the PNG to `/data/workspace/artifacts/` and registers it on the project blackboard at `artifacts/{filename}` if in a project. Returns a multimodal `_image` block for inline display. Default provider is Gemini (falls back to OpenAI DALL-E 3 if `provider="openai"` is specified). |

### Operator-Only Tools

The following tools are only available to the **operator agent** (when `ALLOWED_TOOLS` is set in the agent's environment). They are not accessible to user-created agents.

#### Fleet Templates (`fleet_tool.py`)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_templates` | -- | List available fleet templates |
| `apply_template` | `template`, `model` (default "") | Apply a fleet template to deploy a predefined set of agents. Requires user-origin provenance check. |

#### Fleet & Project Management (`operator_tools.py`)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `propose_edit` | `agent_id`, `field` (enum: `instructions` / `soul` / `model` / `role` / `heartbeat` / `thinking` / `budget` / `permissions`), `value` | Propose a config change for an agent. Returns a `change_id` that must be confirmed. |
| `confirm_edit` | `change_id` | Apply a previously proposed edit. |
| `save_observations` | `fleet_summary`, `agents_attention` (default []), `cost_trend`, `notes` (default "") | Persist fleet health observations for human review. |
| `read_agent_history` | `agent_id`, `period` (default "today"; values: `today` / `yesterday` / `week`) | Read an agent's conversation history with period filtering (operator version). |
| `create_agent` | `name`, `role`, `model` (default ""), `instructions`, `soul` (default "") | Create a new agent. |
| `list_projects` | -- | List all projects. |
| `get_project` | `project_name` | Get details for a project. |
| `create_project` | `name`, `description`, `agent_ids` (default []) | Create a new project and optionally add agents. |
| `add_agents_to_project` | `project_name`, `agent_ids` | Add one or more agents to a project. |
| `remove_agents_from_project` | `project_name`, `agent_ids` | Remove one or more agents from a project. |
| `update_project_context` | `project_name`, `context` | Update the shared context text (`PROJECT.md`) for a project. |

**Permission ceiling:** The operator cannot grant `can_spawn=true` or `can_use_wallet=true` to agents. Budget limits: daily $0.01–$1000, monthly $0.10–$30000.

## MCP Tools

Agents can also use tools from external MCP (Model Context Protocol) servers. These are configured per-agent in `config/agents.yaml` and discovered automatically at startup. See [MCP Integration](mcp.md) for details.

## Custom Skills

Agents can create and load custom skills at runtime:

```python
from src.agent.skills import skill

@skill(
    name="analyze_csv",
    description="Parse and analyze a CSV file",
    parameters={
        "path": {"type": "string", "description": "Path to CSV file"},
        "query": {"type": "string", "description": "Analysis question"},
    },
)
async def analyze_csv(path: str, query: str, *, workspace_manager=None) -> dict:
    # Implementation here
    return {"result": "analysis output"}
```

### Auto-Injected Dependencies

Skills can request these keyword-only arguments (auto-injected by SkillRegistry):

| Argument | Type | Provides |
|----------|------|----------|
| `mesh_client` | `MeshClient` | Access to mesh APIs (blackboard, pub/sub) |
| `workspace_manager` | `WorkspaceManager` | Access to workspace files |
| `memory_store` | `MemoryStore` | Access to structured memory DB |
