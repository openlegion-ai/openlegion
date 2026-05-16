# Configuration

OpenLegion uses YAML and JSON files in the `config/` directory. Config files are created during `openlegion start` (inline setup) and can be edited directly.

## Config Files

| File | Format | Purpose |
|------|--------|---------|
| `config/agents.yaml` | YAML | Agent definitions (role, model, skills, resources) |
| `config/mesh.yaml` | YAML | Mesh host settings, LLM defaults, channel config |
| `config/permissions.json` | JSON | Per-agent ACL matrix |
| `config/cron.json` | JSON | Scheduled job state (auto-managed) |
| `config/projects/` | Directory | Per-team data (`team.md` / legacy `project.md`, members). Directory name stays `projects/` through PR 2 of the project→team rename. |
| `config/settings.json` | JSON | Dashboard-managed runtime settings: browser speed/delay/timeout, execution limits (`max_iterations`, `chat_max_tool_rounds`, etc.), default budgets, and a `browser_flags` dict that overrides any flag in [`src/browser/flags.py:KNOWN_FLAGS`](../src/browser/flags.py) (precedence sits between per-agent overrides and env vars). Read by the browser service via `src/browser/flags.py`. The dashboard also bridges the CAPTCHA solver provider/key fields into the host process env (via `os.environ[…]` in `src/host/runtime.py`) so the browser container picks them up. **The four `_ENV_ONLY_FLAGS` (`CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD`) are stripped from this file at load with a one-time warning that names the stripped keys** — settings.json is plaintext on disk with no chmod / encryption, so solver secrets must come from env vars only. See [Browser Flag Precedence](#browser-flag-precedence). |
| `config/api_keys.json` | JSON | Named API keys for mesh authentication. Shape: `{"keys": {key_id: {name, key_hash, created_at, last_used_at}}}`. Key IDs are `"ak_" + token_hex(6)`; the raw key is returned **once** at creation (`POST /api/api-keys`) and never persisted — only the salted SHA-256 hash (`sha256(key_id + raw_key)`) is stored. Name capped at 128 chars. The legacy `OPENLEGION_API_KEY` env var (single key) is also accepted as a back-compat fallback. |
| `data/captcha_costs.json` | JSON | Runtime CAPTCHA spend ledger (chmod `0o600`). Per-agent monthly buckets in millicents (1/100,000 USD); persisted as a periodic snapshot from in-memory state on the 60s metrics tick. Override path with `CAPTCHA_COST_COUNTER_PATH`. State is current-month only — older windows defer to the planned SQLite snapshots. Restart loses at most one tick of spend. |
| `config/network.yaml` | YAML | Network settings (`no_proxy` exclusion list for proxy mode). |
| `.env` | dotenv | API keys and credentials |

## `config/agents.yaml`

Defines every agent in the fleet.

**Reserved agent IDs.** `mesh`, `operator`, and `canary-probe` are reserved (`src/shared/types.py:RESERVED_AGENT_IDS`). `mesh` and `operator` are internal trust-zone identifiers; `canary-probe` is the dedicated profile used by the stealth canary scanner (`src/browser/canary.py`). Attempts to create an agent with any of these IDs are rejected. The CLI also rejects `operator` from team membership (`src/cli/config.py`).

```yaml
agents:
  researcher:
    role: Research assistant that finds and analyzes information
    model: anthropic/claude-haiku-4-5-20251001
    skills_dir: ./skills/researcher
    system_prompt: |
      You are the 'researcher' agent in a multi-agent fleet.
      Your specialty is finding information and writing reports.
    resources:
      memory_limit: 1g
      cpu_limit: 0.5
    budget:
      daily_usd: 5.00
      monthly_usd: 100.00
    thinking: medium
    mcp_servers:
      - name: filesystem
        command: mcp-server-filesystem
        args: ["/data"]
```

### Agent Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Short description of the agent's purpose |
| `model` | string | No | LLM model in `provider/model` format. Falls back to `llm.default_model` in mesh.yaml, then `openai/gpt-4o-mini` if neither is set. |
| `skills_dir` | string | No | Path to custom skills directory |
| `system_prompt` | string | No | Custom system prompt. Auto-generated if omitted. Also accepted as `instructions` — first non-empty wins (`src/cli/config.py`). Distinct from the top-level `initial_instructions` (below): `system_prompt`/`instructions` seed the runtime system prompt, `initial_instructions` seeds the `INSTRUCTIONS.md` workspace file on first boot. |
| `resources.memory_limit` | string | No | Reserved for future use. Currently hardcoded by the runtime for security — `384m` for workers, `128m` for the operator agent (selected by `ALLOWED_TOOLS` env). |
| `resources.cpu_limit` | float | No | Reserved for future use. Currently hardcoded by the runtime for security — `0.15` for workers, `0.05` for the operator agent. |
| `budget.daily_usd` | float | No | Daily spend cap in USD (default: `10.00`) |
| `budget.monthly_usd` | float | No | Monthly spend cap in USD (default: `200.00`) |
| `soul` | string | No | Seeds `SOUL.md` on first boot — defines the agent's personality and behavioral guidelines |
| `initial_instructions` | string | No | Seeds `INSTRUCTIONS.md` on first boot. Distinct from `system_prompt` — this sets the agent's operating instructions file |
| `thinking` | string | No | Extended thinking/reasoning mode: `off` (default), `low`, `medium`, or `high`. Anthropic models use thinking budgets (5K/10K/25K tokens). OpenAI o-series models use `reasoning_effort`. Ignored for unsupported models |
| `mcp_servers` | list | No | External MCP tool servers. See [MCP Integration](mcp.md) |
| `initial_interface` | string | No | Seeds `INTERFACE.md` on first boot. Defines the agent's public collaboration contract — what inputs it accepts, what outputs it produces, and what topics it subscribes to. Readable by other agents via `get_agent_profile`. |

### Model Format

Models use `provider/model_id` format, routed through [LiteLLM](https://docs.litellm.ai/):

```yaml
# Anthropic
model: anthropic/claude-haiku-4-5-20251001
model: anthropic/claude-sonnet-4-6

# OpenAI
model: openai/gpt-4.1-mini
model: openai/gpt-4.1

# Google
model: gemini/gemini-2.5-flash

# Other supported providers
model: deepseek/deepseek-chat
model: xai/grok-2
model: groq/llama-3.3-70b-versatile
model: zai/glm-4-plus
```

## `config/mesh.yaml`

Controls the mesh host, LLM defaults, and channel configuration.

```yaml
mesh:
  host: 0.0.0.0
  port: 8420

llm:
  default_model: anthropic/claude-haiku-4-5-20251001
  embedding_model: text-embedding-3-small
  failover:
    anthropic/claude-sonnet-4-6:
      - anthropic/claude-haiku-4-5-20251001
      - openai/gpt-4.1-mini
    openai/gpt-4o:
      - openai/gpt-4o-mini

channels:
  telegram:
    enabled: true
    default_agent: assistant
  discord:
    enabled: false
    default_agent: assistant

collaboration: true
```

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mesh.host` | string | `0.0.0.0` | Bind address for mesh server |
| `mesh.port` | integer | `8420` | Mesh server port |
| `llm.default_model` | string | -- | Default model for agents without explicit model |
| `llm.embedding_model` | string | *auto* | Model for memory embeddings. Auto-detected from the default LLM provider via `_PROVIDER_EMBEDDING_DEFAULTS` in `src/cli/runtime.py`: only the prefixes `openai/`, `gpt-`, `o1`, `o3`, `o4` resolve to `text-embedding-3-small`; everything else (Anthropic, Google, DeepSeek, xAI, Groq, …) falls through to `"none"`. Must produce 1536-dim vectors. Set to `"none"` to disable vector search (FTS5 keyword search still works). |
| `llm.failover` | dict[string, list[string]] | -- | Per-model failover chains, typed as `dict[str, list[str]]` in `src/host/credentials.py`. Map each primary `provider/model` to an ordered fallback list. The legacy `{primary: ..., fallback: ...}` single-pair shape is **not** recognized and is silently ignored. |
| `collaboration` | boolean | `true` | Allow inter-agent messaging |

### Channel Configuration

See [Channels](channels.md) for full setup instructions.

| Field | Type | Description |
|-------|------|-------------|
| `channels.telegram.enabled` | boolean | Enable Telegram bot |
| `channels.telegram.default_agent` | string | Default agent for Telegram users |
| `channels.telegram.allowed_users` | list[int] | Optional: restrict by Telegram user ID |
| `channels.discord.enabled` | boolean | Enable Discord bot |
| `channels.discord.default_agent` | string | Default agent for Discord users |
| `channels.discord.allowed_guilds` | list[int] | Optional: restrict by Discord server ID |
| `channels.slack.enabled` | boolean | Enable Slack bot (Socket Mode) |
| `channels.slack.default_agent` | string | Default agent for Slack users |
| `channels.whatsapp.enabled` | boolean | Enable WhatsApp bot (Cloud API) |
| `channels.whatsapp.default_agent` | string | Default agent for WhatsApp users |

## `config/permissions.json`

Per-agent access control lists. Default policy is **deny** -- if not listed, it's blocked. If `config/permissions.json` is missing entirely, **every agent gets deny-all**. An optional `"default"` agent entry acts as a template fallback for any agent not otherwise listed (see `src/host/permissions.py`).

```json
{
  "permissions": {
    "researcher": {
      "can_message": [],
      "can_publish": ["research_complete"],
      "can_subscribe": ["new_lead"],
      "blackboard_read": ["projects/sales/*"],
      "blackboard_write": ["projects/sales/*"],
      "allowed_apis": ["llm", "brave_search"],
      "allowed_credentials": ["myservice_*"]
    },
    "writer": {
      "can_message": ["*"],
      "can_publish": ["*"],
      "can_subscribe": ["*"],
      "blackboard_read": ["projects/content/*"],
      "blackboard_write": ["projects/content/*"],
      "allowed_apis": ["llm"],
      "allowed_credentials": ["*"]
    }
  }
}
```

Blackboard patterns use the `projects/{name}/*` namespace (on-disk prefix retained through PR 2 of the project→team rename). When an agent joins a team via `openlegion team add-agent`, it automatically receives read/write access to that team's namespace. Solo agents (not on any team) get empty blackboard permissions. The `MeshClient` on the agent side transparently prefixes all blackboard keys with the team namespace, so agents use natural keys like `tasks/research_01` while data is stored under `projects/sales/tasks/research_01`.

### Permission Fields

| Field | Type | Description |
|-------|------|-------------|
| `can_message` | list[string] | Agent names this agent can send messages to |
| `can_publish` | list[string] | PubSub topics this agent can publish to |
| `can_subscribe` | list[string] | PubSub topics this agent can subscribe to |
| `blackboard_read` | list[string] | Glob patterns for readable blackboard keys |
| `blackboard_write` | list[string] | Glob patterns for writable blackboard keys |
| `allowed_apis` | list[string] | External APIs accessible through the vault proxy |
| `allowed_credentials` | list[string] | Glob patterns for accessible credential names. Matched with `fnmatch` (case-insensitive). `["*"]` grants access to all agent-tier credentials; `[]` denies all. System credentials (LLM provider keys) are always blocked regardless of patterns. |
| `can_use_browser` | boolean | Whether this agent can use the shared browser service. Default: `false`. |
| `browser_actions` | list[string] \| null | Per-action gate inside the browser surface, applied only when `can_use_browser=true`. Three forms with non-obvious semantics: `null` (default — field omitted) **and** `["*"]` both grant **all** known browser actions including any added in future phases. `[]` denies every action (equivalent to `can_use_browser=false`). Any specific list (e.g. `["navigate", "snapshot", "screenshot"]`) is an **allowlist** — only listed actions are permitted; everything else is denied. The action validator is `KNOWN_BROWSER_ACTIONS` in `src/host/permissions.py` (currently 26 entries: `navigate`, `snapshot`, `click`, `type`, `hover`, `screenshot`, `reset`, `focus`, `status`, `detect_captcha`, `scroll`, `wait_for`, `press_key`, `go_back`, `go_forward`, `switch_tab`, `upload_file`, `download`, `find_text`, `open_tab`, `fill_form`, `click_xy`, `inspect_requests`, `solve_captcha`, `request_captcha_help`, `request_browser_login`); typo'd names get HTTP 400 at the mesh gate. |
| `can_spawn` | boolean | Whether this agent can spawn ephemeral fleet agents via `spawn_fleet_agent`. Default: `false`. |
| `can_manage_cron` | boolean | Whether this agent can create, update, and delete cron jobs. Default: `false`. |
| `can_use_wallet` | boolean | Whether this agent can access the wallet signing service. Default: `false`. |
| `wallet_allowed_chains` | list[string] | Chains this agent can transact on (e.g., `["ethereum", "base"]`). `["*"]` allows all chains. Default: `[]`. |
| `wallet_spend_limit_per_tx_usd` | float | Max USD value per transaction. `0` uses the global default. |
| `wallet_spend_limit_daily_usd` | float | Daily aggregate transaction limit in USD. `0` uses the global default. |
| `wallet_rate_limit_per_hour` | integer | Max transactions per hour. `0` uses the global default. |
| `wallet_allowed_contracts` | list[string] | Contract addresses this agent can interact with. Empty list allows all. |
| `can_manage_fleet` | boolean | Control-plane: create/register/deregister agents via mesh endpoints. Default: `false` for workers, `true` for the operator agent. |
| `can_manage_teams` (alias `can_manage_projects`) | boolean | Control-plane: create/archive teams and edit membership. Default: `false` for workers, `true` for operator. The legacy field name is mirrored via a permanent model-validator so config files written under either name keep working. |
| `can_edit_agent_config` | boolean | Control-plane: propose/confirm edits to another agent's instructions/soul/model/etc. Default: `false` for workers, `true` for operator. |
| `can_view_fleet_metrics` | boolean | Control-plane: access `/mesh/system/metrics` and `/mesh/agents/{id}/metrics`. Default: `false` for workers, `true` for operator. |
| `can_route_tasks` | boolean | Control-plane: create durable orchestration task records (Task 6 surface). Default: `false` for workers, `true` for operator. |
| `can_request_user_credentials` | boolean | Allow `request_credential` and `request_browser_login` tools. Default: `false` for workers, `true` for operator. |

### Glob Patterns

- `*` matches everything
- `projects/myteam/*` matches all keys under the `myteam` namespace
- `tasks/*` matches `tasks/abc123`, `tasks/research_01`, etc.
- `context/prospect_*` matches `context/prospect_acme`, etc.

Note: Blackboard permissions are managed automatically when agents join/leave teams. You generally don't need to edit blackboard patterns by hand.

## `.env` — Credentials

API keys and secrets. Never committed to git. Uses a two-tier prefix system:

- **`OPENLEGION_SYSTEM_`** — System tier. LLM provider keys used by the mesh proxy internally. Never accessible by agents.
- **`OPENLEGION_CRED_`** — Agent tier. Tool and service keys accessible based on `allowed_credentials` patterns.

```bash
# ── System-tier credentials (LLM providers) ──────────────────
OPENLEGION_SYSTEM_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_SYSTEM_OPENAI_API_KEY=sk-...
# OPENLEGION_SYSTEM_GEMINI_API_KEY=...
# OPENLEGION_SYSTEM_OPENAI_API_BASE=https://your-proxy.example.com/v1

# ── Agent-tier credentials (tools / services) ────────────────
# OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...
# OPENLEGION_CRED_APOLLO_API_KEY=...
# Channel Bot Tokens (optional)
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC-...
OPENLEGION_CRED_DISCORD_BOT_TOKEN=MTIz...
OPENLEGION_CRED_SLACK_BOT_TOKEN=xoxb-...
OPENLEGION_CRED_SLACK_APP_TOKEN=xapp-...
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...
# Required for WhatsApp inbound webhook signature verification (X-Hub-Signature-256).
# Without it, signature verification is skipped with a WARNING.
OPENLEGION_CRED_WHATSAPP_VERIFY_TOKEN=...
```

All credentials in the `OPENLEGION_SYSTEM_*` and `OPENLEGION_CRED_*` tiers are loaded by the credential vault (`src/host/credentials.py`). Agents never see values directly -- they make API calls through the mesh proxy, which injects credentials server-side.

**CAPTCHA solver credentials bypass the vault.** The four secrets used by the browser CAPTCHA pipeline — `CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD` — are listed in `flags._ENV_ONLY_FLAGS` and read directly from the process environment by the browser service. They are NOT routed through the agent-tier `OPENLEGION_CRED_*` vault, do NOT appear in the credentials UI, and are explicitly stripped from `config/settings.json:browser_flags` at load with a one-time warning (settings.json is plaintext on disk). The dashboard's `POST /api/captcha-solver` endpoint writes them via `os.environ[…]` directly. Provide them as plain environment variables (e.g. via `.env`).

**Channel credential fallback:** Channel bot tokens (Telegram, Discord, Slack, WhatsApp) are resolved with a four-tier fallback chain: `mesh.yaml` channel config field (e.g. `channels.telegram.bot_token`) → `OPENLEGION_SYSTEM_<NAME>` → `OPENLEGION_CRED_<NAME>` → legacy unprefixed `<NAME>` (e.g., `TELEGRAM_BOT_TOKEN`). The `OPENLEGION_CRED_` prefix is recommended for channel tokens.

**Important:** LLM provider keys **must** use the `OPENLEGION_SYSTEM_` prefix. The mesh proxy only looks for provider keys in the system tier. A provider key stored with `OPENLEGION_CRED_` will be treated as an agent-tier credential and will not be used for LLM calls.

**Key/value validation.** Credential names must match `^[A-Za-z_][A-Za-z0-9_]*$`; values reject embedded newline (`\n`) and carriage-return (`\r`) bytes (`src/host/credentials.py`). The per-agent `.agent.env` files generated for `SandboxBackend` agents are written with `chmod 0o600` (`src/host/runtime.py`).

## `config/cron.json`

Auto-managed by the cron scheduler. You can edit it directly, but it's usually managed through agent tools (`set_cron`) or the mesh API.

```json
{
  "jobs": [
    {
      "id": "cron_abc123",
      "agent": "researcher",
      "schedule": "0 9 * * 1-5",
      "message": "Check for new leads and update the pipeline",
      "timezone": "UTC",
      "enabled": true,
      "suppress_empty": true,
      "heartbeat": false,
      "last_run": "2026-02-20T09:00:05.123456+00:00",
      "run_count": 42,
      "error_count": 0
    }
  ]
}
```

See [Triggering & Automation](triggering.md) for schedule syntax and heartbeat configuration.

## Environment Variables

Beyond credentials, these environment variables affect runtime behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENLEGION_LOG_FORMAT` | `json` | Log output format: `json` (structured) or `text` (human-readable) |
| `MCP_SERVERS` | -- | JSON string of MCP server configs (set automatically in containers) |
| `MESH_AUTH_TOKEN` | -- | Agent auth token (set automatically in containers) |
| `MESH_HOST` | -- | Mesh host URL (set automatically in containers) |
| `AGENT_ID` | -- | Agent identifier (set automatically in containers) |
| `THINKING` | `off` | Extended thinking/reasoning mode (set automatically from `thinking` in agents.yaml) |
| `PROJECT_NAME` | -- | Team this agent belongs to (set automatically for team members). Env-var name is retained through PR 2 of the project→team rename. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for memory vector search (set automatically from `llm.embedding_model` in mesh.yaml). Set to `"none"` to disable vector search |
| `OPENLEGION_MAX_AGENTS` | `0` | Plan limit: maximum agents to start. `0` means unlimited. If set to N > 0, only the first N agents are started. **Also drives plan-aware browser-service sizing** (`src/host/runtime.py:start_browser_service`) — the browser container's memory, SHM, CPU quota, and `MAX_BROWSERS` cap are selected by the `max_agents <= 1` / `<= 5` / `> 5` branches: ≤1 → Basic (2GB / 512m / 1.0 CPU / 1 browser), ≤5 → Growth (4GB / 1g / 1.5 CPU / N browsers), >5 → Pro (8GB / 2g / 2.0 CPU / `min(N, 10)` browsers). Note: `OPENLEGION_MAX_AGENTS=0` (the default, meaning "unlimited agents") falls into the `<= 1` branch and selects the **Basic** browser-service tier — counterintuitive but intentional for self-host installs that haven't declared a plan. |
| `OPENLEGION_MAX_TEAMS` (alias `OPENLEGION_MAX_PROJECTS`) | -- | Plan limit. Three-state semantics: **unset** (env var absent) → teams are unlimited; **set to `0`** → teams are disabled entirely, `POST /mesh/teams` returns HTTP 403; **set to N > 0** → only N teams are allowed. The new var name is the canonical read; the legacy one is preserved as a permanent zero-cost fallback for existing deploys. |
| `OPENLEGION_TEAM_SCOPE_MODE` (alias `OPENLEGION_PROJECT_SCOPE_MODE`) | `enforce` | Team-scope rollout kill switch. Accepts `"warn"` or `"enforce"` (case-insensitive via `.lower()`). Invalid values default to `"enforce"` with a WARNING. **Read once at module import — restart the mesh to change.** Legacy var name preserved as a permanent fallback. |
| `OPENLEGION_TEAM_MIGRATION_RENAME_DB` | `1` | DB column rename `tasks.project_id` → `tasks.team_id` on startup. Default-on as of PR 3 of the project→team rename; set to `0` to opt out for emergency rollback. The orchestration layer introspects the live column at init via PRAGMA so either schema shape works without source changes. |
| `OPENLEGION_ORCHESTRATION_TASKS_V2` | `1` | Toggle the durable orchestration v2 tasks store. `"1"` enables; `"0"` falls back to the legacy blackboard task path. **Read once at module import — restart the mesh to flip.** |
| `OPENLEGION_ORCHESTRATION_TASKS_DB` | `data/tasks.db` | Override the SQLite path for the orchestration v2 tasks store. |
| `OPENLEGION_API_KEY` | -- | Legacy single API key for back-compat with the named-key system (`config/api_keys.json`). Compared with `hmac.compare_digest`. Prefer creating named keys via `POST /api/api-keys`. |
| `OPENLEGION_APP_URL` | -- | App URL used by the dashboard's external SSO link (read by `src/dashboard/server.py`). |
| `OPENLEGION_HOST_NETWORK` | `0` | Use Docker host networking for agent containers instead of bridge network. Enabled only when the literal lowercase value is one of `"1"`, `"true"`, `"yes"` (after `.strip()`) — `True`, `YES`, etc. are rejected. Not recommended — disables network isolation. Also propagates to the browser service, which **additionally requires** `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1` (see below). |
| `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK` | `0` | **Hard gate** for running the browser container in host-network mode. Same case-sensitive literal match as `OPENLEGION_HOST_NETWORK` — only `"1"`/`"true"`/`"yes"` lowercase are accepted. When `OPENLEGION_HOST_NETWORK=1` and this flag is unset, `start_browser_service` raises `RuntimeError` at boot. The egress iptables filter cannot install in the host network namespace, so host mode strips the authoritative SSRF control. Set this to acknowledge the regression and run anyway (INSECURE — the browser will reach the host's private networks). |
| `OPENLEGION_SYSTEM_WALLET_MASTER_SEED` | -- | BIP-39 mnemonic (24 words) for HD wallet key derivation. Required to enable wallet features. Generate with `openlegion wallet init`. |
| `OPENLEGION_WALLET_LIMIT_PER_TX_USD` | `10.0` | Default per-transaction USD cap applied when an agent's `wallet_spend_limit_per_tx_usd` is `0`. |
| `OPENLEGION_WALLET_LIMIT_DAILY_USD` | `100.0` | Default daily aggregate USD cap applied when an agent's `wallet_spend_limit_daily_usd` is `0`. |
| `OPENLEGION_WALLET_RATE_LIMIT_PER_HOUR` | `10` | Default transactions-per-hour cap applied when an agent's `wallet_rate_limit_per_hour` is `0`. |
| `BROWSER_OS` | `windows` | OS fingerprint for Camoufox browser: `windows`, `macos`, or `linux`. Windows is recommended (≈70% desktop market share; Linux is a datacenter signal). |
| `BROWSER_LOCALE` | `en-US` | BCP-47 locale tag for browser fingerprint (e.g. `en-US`, `de-DE`). |
| `BROWSER_UA_VERSION` | -- | Override Firefox version in User-Agent string (e.g. `138.0`). Useful when Camoufox's bundled Firefox is too old for sites that enforce minimum browser versions (e.g. Shopify). Uses Camoufox's native config system. Ignored when a non-default `BROWSER_DEVICE_PROFILE` pins its own UA. |
| `BROWSER_DEVICE_PROFILE` | `desktop-windows` | Device emulation profile for Camoufox. One of `desktop-windows` (default), `desktop-macos`, `mobile-ios`, `mobile-android`. Controls UA, viewport, device-pixel-ratio, `is_mobile`, `has_touch`, and the navigator-override init script. Per-agent overrides supported via the operator settings layer. See **Device Profiles** below. |
| `BROWSER_PROXY_URL` | -- | Proxy URL for browser traffic (HTTP/HTTPS only, e.g. `http://proxy:8080`). SOCKS5 is not supported. Residential proxies recommended. |
| `BROWSER_PROXY_USER` | -- | Proxy authentication username. |
| `BROWSER_PROXY_PASS` | -- | Proxy authentication password. |
| `OPENLEGION_BROWSER_MAX_CONCURRENT` | *autodetected* | Per-service cap on simultaneous Camoufox browser instances (clamped to `[1, 64]`). When unset, autodetected from container memory by `_max_from_memory(total_mb)` in `src/browser/__main__.py`: `(total_mb - _HEADROOM_MB) // _MEM_PER_BROWSER_MB` where `_HEADROOM_MB=3072` and `_MEM_PER_BROWSER_MB=450`. Falls back to `_FALLBACK_DEFAULT=5` only when memory detection fails (typically non-Linux dev environments). **Startup-only** — runtime reconfig is unsupported (would need to bound the acquire semaphore mid-flight). The operator settings layer (`config/settings.json:browser_flags`) cannot override it either — `_resolve_max_browsers` is read once at process launch. Restart the browser service to change this. The legacy name `MAX_BROWSERS` is still honored as a fallback for older deployments. |
| `BROWSER_AUTH_TOKEN` | -- | Bearer token the mesh uses to call the browser service (`/browser/*` endpoints). When `MESH_AUTH_TOKEN` is set and `BROWSER_AUTH_TOKEN` is missing, the browser service raises `RuntimeError` at startup unless `BROWSER_AUTH_INSECURE=1` is set. |
| `BROWSER_AUTH_INSECURE` | -- | Set to `1` to disable browser-service bearer auth (dev only). Logged as a WARNING at startup. |
| `BROWSER_EGRESS_ALLOWLIST` | -- | Comma-separated CIDR allowlist for the browser container's iptables egress filter. By default the entrypoint REJECTs all RFC1918 / loopback / link-local / CGNAT / IANA-reserved ranges (the authoritative SSRF control for browser-initiated traffic); use this to punch through specific destinations (e.g. `BROWSER_EGRESS_ALLOWLIST=10.0.0.5/32` for a private proxy). When `BROWSER_PROXY_URL` is a literal RFC1918 / loopback / link-local / reserved IP, the runtime errors at startup unless this flag covers it. |
| `BROWSER_EGRESS_DISABLE` | -- | Disable the browser egress filter entirely (no iptables rules installed). Removes the SSRF guarantee; intended for operator debugging only. |
| `OPENLEGION_BROWSER_ALLOW_ENGINE_MISMATCH` | -- | Truthy string → allow UA / engine fingerprint mismatch (e.g. claiming Safari while running Camoufox's Gecko engine). Off by default; toggle only when intentionally serving a mobile profile to a site that doesn't probe engine-level signals (`src/browser/stealth.py`). |
| `OPENLEGION_CAPTCHA_SOLVER_PROVIDER` | -- | Alias bridge for `CAPTCHA_SOLVER_PROVIDER`. The dashboard writes this from `config/settings.json:captcha_solver_provider` into `os.environ` at boot (`src/host/runtime.py`); the browser container reads the canonical name. |
| `OPENLEGION_CAPTCHA_SOLVER_KEY` | -- | Alias bridge for `CAPTCHA_SOLVER_KEY`. Same env-bridge pattern as above (`src/host/runtime.py`, `src/dashboard/server.py`). |
| `OPENLEGION_SYSTEM_PROXY` | -- | System-wide outbound HTTP proxy URL for all agent traffic. Managed via the dashboard proxy settings page. |
| `HTTP_PROXY` / `HTTPS_PROXY` | -- | Per-agent proxy URLs. Auto-injected into agent containers by the runtime when a per-agent proxy is configured. Read by the agent-side `http_request` tool. |
| `OPENLEGION_TOOL_TIMEOUT` | `300` | Per-tool execution timeout in seconds (hard ceiling). |
| `OPENLEGION_MAX_ITERATIONS` | `20` | Maximum agent loop iterations per task (clamped 1–100). Overrides the default at the agent level. |
| `OPENLEGION_CHAT_MAX_TOOL_ROUNDS` | `30` | Maximum tool rounds per chat turn (clamped 1–200). |
| `OPENLEGION_CHAT_MAX_TOTAL_ROUNDS` | `200` | Maximum total chat rounds before session auto-continuation (clamped 1–1000). |
| `OPENLEGION_SETTINGS_PATH` | `config/settings.json` | Path to the operator settings file consulted by `src/browser/flags.py`. Override for tests / containerized deployments. |
| `OPENLEGION_UBLOCK_XPI` | `/opt/openlegion/extensions/uBlock0.xpi` | Path to the uBlock Origin XPI installed into agent browser profiles by the schema-v3 migration. Override for tests. |
| `OPENLEGION_REDACTION_URL_QUERY_ALLOW` | -- | Comma-separated list of URL query parameter names that the unified redactor (`src/shared/redaction.py`) should NOT redact. Use to keep specific identifiers visible in browser logs / artifacts when an integration intentionally puts non-secret context in the query string. |

The mesh port is configured in `config/mesh.yaml` (`mesh.port`), not via environment variable.

## Browser Service Flags

The browser service centralizes its environment-variable surface in [`src/browser/flags.py:KNOWN_FLAGS`](../src/browser/flags.py). Every read goes through a typed accessor (`get_str`, `get_bool`, `get_int`, `get_float`) which coerces and validates the value.

### Browser Flag Precedence

Highest to lowest:

1. **Per-agent override** — registered at runtime via `flags.set_agent_override(agent_id, name, value)`. Surfaced through the dashboard flags panel (per-template tuning).
2. **Operator settings** — `config/settings.json` under the `browser_flags` key. Plaintext on disk; the four `_ENV_ONLY_FLAGS` (CAPTCHA solver creds) are stripped at load with a warning.
3. **Environment variable** — the canonical name listed in the tables below (case-sensitive).
4. **Hardcoded default** at the call site.

A malformed value at any layer logs a warning and falls through to the next, so a broken per-agent override cannot mask a valid env-var fallback.

```json
// config/settings.json — operator-wide overrides for any flag in KNOWN_FLAGS.
// All values are stored as strings and coerced via typed accessors
// (`get_bool` / `get_int` / `get_float`) at read time, so quote bools/ints.
{
  "browser_flags": {
    "BROWSER_DEVICE_PROFILE": "desktop-windows",
    "BROWSER_DOWNLOADS_DISABLED": "false",
    "CAPTCHA_RATE_LIMIT_PER_HOUR": "20"
  }
}
```

### CAPTCHA Solver

CAPTCHA solver flags drive the metered, breaker-protected solver pipeline in `src/browser/captcha.py` + `src/browser/service.py`. **The four secret-bearing flags marked _env-only_ are stripped from `config/settings.json` at load time** (`flags._ENV_ONLY_FLAGS`) — provide them as plain environment variables only.

| Variable | Default | Description |
|---|---|---|
| `CAPTCHA_SOLVER_PROVIDER` | -- | Primary provider: `2captcha` or `capsolver`. Unset = solver disabled. |
| `CAPTCHA_SOLVER_KEY` | -- | **Env-only.** API key for the primary provider. |
| `CAPTCHA_SOLVER_PROVIDER_SECONDARY` | -- | Failover provider (§11.8). Same value set as primary. |
| `CAPTCHA_SOLVER_KEY_SECONDARY` | -- | **Env-only.** API key for the failover provider. |
| `CAPTCHA_DISABLED` | `false` | Fleet-wide kill switch. Short-circuits BEFORE health/breaker/rate-limit/cost-cap; returns `solver_outcome="no_solver"` envelope and `next_action="request_captcha_help"`. Per-agent override supported. Re-evaluated on every solve attempt — no restart needed. |
| `CAPTCHA_RATE_LIMIT_PER_HOUR` | `20` | Per-agent solve rate limit (range 0–10000). Set `0` to disable. |
| `CAPTCHA_RECAPTCHA_V3_MIN_SCORE` | `0.7` | Minimum reCAPTCHA v3 score to accept (range 0.1–0.9). |

### CAPTCHA Per-Type Timeouts

All values in milliseconds. Defaults reflect provider documentation guidance; override per type for slow networks.

| Variable | Default |
|---|---|
| `CAPTCHA_TIMEOUT_RECAPTCHA_V2_CHECKBOX_MS` | `120000` |
| `CAPTCHA_TIMEOUT_RECAPTCHA_V2_INVISIBLE_MS` | `120000` |
| `CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS` | `60000` |
| `CAPTCHA_TIMEOUT_RECAPTCHA_ENTERPRISE_V2_MS` | `120000` |
| `CAPTCHA_TIMEOUT_RECAPTCHA_ENTERPRISE_V3_MS` | `60000` |
| `CAPTCHA_TIMEOUT_HCAPTCHA_MS` | `120000` |
| `CAPTCHA_TIMEOUT_TURNSTILE_MS` | `180000` |
| `CAPTCHA_TIMEOUT_CF_INTERSTITIAL_TURNSTILE_MS` | `180000` |

### CAPTCHA Solver Proxy

Optional dedicated proxy used for solver tasks (independent of `BROWSER_PROXY_URL`). `2captcha` accepts `{http, socks4, socks5}`; `capsolver` accepts the full `{http, https, socks4, socks5}`. The `_proxyless` task variant is used when these are unset.

| Variable | Default | Description |
|---|---|---|
| `CAPTCHA_SOLVER_PROXY_TYPE` | -- | `http`, `https`, `socks4`, or `socks5`. |
| `CAPTCHA_SOLVER_PROXY_ADDRESS` | -- | Proxy host. |
| `CAPTCHA_SOLVER_PROXY_PORT` | -- | Proxy port. |
| `CAPTCHA_SOLVER_PROXY_LOGIN` | -- | **Env-only.** Proxy username. |
| `CAPTCHA_SOLVER_PROXY_PASSWORD` | -- | **Env-only.** Proxy password. |

### CAPTCHA Pacing

Gaussian-jittered delays between solves to avoid burst traffic patterns. All milliseconds.

| Variable | Default | Description |
|---|---|---|
| `CAPTCHA_PACING_MS_MIN` | `3000` | Lower clamp on the pacing distribution. |
| `CAPTCHA_PACING_MS_MAX` | `12000` | Upper clamp. |
| `CAPTCHA_SOLVE_PACING_MU_MS` | `6000` | Gaussian mean. |
| `CAPTCHA_SOLVE_PACING_SIGMA_MS` | `2500` | Gaussian standard deviation. |

### CAPTCHA Cost Caps & Site Policy

Cost caps are **opt-in** — unset = no cap. All amounts are USD; the runtime ledger stores millicents internally (1 millicent = 1/100,000 USD). Per-tenant cap thresholds (50 / 80 / 100% of cap) emit `tenant_spend_threshold` events through the dashboard EventBus, fired once per crossing per month.

| Variable | Default | Description |
|---|---|---|
| `CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH` | -- (no cap when unset) | Per-agent monthly USD cap. When exceeded, solver short-circuits with `skipped="cost_cap"`. |
| `CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH` | -- (no cap when unset) | Per-tenant monthly USD cap (tenant = team membership from `config/projects/`). Drives 50/80/100% threshold alerts. |
| `CAPTCHA_COST_COUNTER_PATH` | `data/captcha_costs.json` | Path to the persisted cost counter snapshot. chmod `0o600`. |
| `OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS` | -- | Comma-separated; force normal solver flow on hosts otherwise classified `unsolvable` by `src/browser/captcha_policy.py` (e.g. `challenges.cloudflare.com`, `humansecurity.com`, `captcha-delivery.com`). Read once at module import — restart browser service to apply changes. |
| `OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS` | -- | Comma-separated; force escalation-only on hosts the solver would otherwise attempt. Read once at module import — restart browser service to apply changes. |
| `BROWSER_CAPTCHA_REDETECT_ENABLED` | `true` | Gate the MutationObserver-based post-action captcha re-detection on `click` / `type` / `press_key` / `fill_form`. |

### Browser Operator Kill Switches

Default-off feature gates for high-trust browser surfaces. When tripped, the corresponding endpoint returns a `403 forbidden` envelope with `next_action="request_browser_login"` (or equivalent escalation).

| Variable | Default | Effect |
|---|---|---|
| `BROWSER_DOWNLOADS_DISABLED` | `false` | `/mesh/browser/download` returns forbidden envelope; agent download tool fails closed. |
| `BROWSER_NETWORK_INSPECT_DISABLED` | `false` | `inspect_requests` action returns forbidden envelope. |
| `BROWSER_COOKIE_IMPORT_DISABLED` | `false` | Operator cookie/session-import endpoint disabled. |
| `BROWSER_CANARY_ENABLED` | `false` | Opt-in; enables the stealth canary scanner (`src/browser/canary.py`). Uses the reserved `canary-probe` agent ID. |

### Browser Snapshot, Screenshot & Ad-blocker

| Variable | Default | Description |
|---|---|---|
| `BROWSER_SNAPSHOT_FORMAT` | `v2` | a11y snapshot rendering: `v1` (legacy verbose) or `v2` (compact, current default after release gate). |
| `BROWSER_SCREENSHOT_FORMAT` | `webp` | `webp` or `png`. WebP encoding falls back to PNG on failure. |
| `BROWSER_SCREENSHOT_QUALITY` | `75` | WebP quality 1–100. |
| `BROWSER_RESOLUTION_POOL` | `true` | Per-agent deterministic viewport pool (1280×720 → 1920×1080) keyed by SHA-256 of the agent ID. Disable for fixed 1920×1080. |
| `BROWSER_ENABLE_ADBLOCK` | `true` | Gates the uBlock Origin install during the schema-v3 profile migration. |
| `BROWSER_PLATFORM_TIMING_ENABLED` | `true` | Per-platform pre-navigation timing jitter. Disable to remove the small think-time delay before each nav (e.g. for deterministic test runs). |

### Browser Behavior Recorder

| Variable | Default | Description |
|---|---|---|
| `BROWSER_RECORD_BEHAVIOR` | `0` | Set `1` to enable the §5.3 behavior recorder. Records hosts only — never full URLs. |

### File Upload / Download Staging

The two-stage upload protocol (mesh-side `upload-stage` → `upload_apply`) keeps large bodies out of agent address space. The mesh stages bytes into a tmpfs-backed dir and the browser service receives them via an internal endpoint.

| Variable | Default | Description |
|---|---|---|
| `OPENLEGION_UPLOAD_STAGE_DIR` | `/tmp/openlegion-upload-stage` | Mesh-side staging directory. |
| `OPENLEGION_UPLOAD_STAGE_TTL_S` | `60` | Orphan staging-file TTL in seconds (clamped ≥ 5). `.partial` files use 5×TTL. |
| `OPENLEGION_UPLOAD_STAGE_MAX_MB` | `50` | Per-file upload byte cap (clamped ≥ 1). |
| `OPENLEGION_UPLOAD_RECV_DIR` | `/tmp/upload-recv` | Browser-side receive directory. |
| `BROWSER_DOWNLOAD_DIR` | `/tmp/downloads` | Browser-side download directory. |
| `BROWSER_DOWNLOAD_TTL_S` | `60` | Stale download GC TTL in seconds. |

### Session Continuity (§20)

Opt-in persistence of `BrowserContext.storage_state()` across container restarts. **The sidecar contains live session tokens** — if leaked, those tokens grant account takeover on whatever sites the agent is logged into. The module bakes in NO time-based expiry; operators are responsible for rotating sidecars on a cadence appropriate for their threat model.

| Variable | Default | Description |
|---|---|---|
| `BROWSER_SESSION_PERSISTENCE_ENABLED` | `false` | **Opt-in.** Enable per-agent storage_state snapshots. |
| `BROWSER_SESSION_PERIODIC_SNAPSHOT_S` | `300` | Periodic snapshot interval in seconds (range 60–3600). Snapshots run on the 60s metrics tick when this elapsed. Lower = better RPO at the cost of disk writes. |
| `BROWSER_SESSION_DIR` | `data/sessions` | Directory for per-agent sidecars (`<agent_id>.json`, chmod `0o600`). |

### Mobile / Device Profile

See [Browser Device Profiles](#browser-device-profiles) below for behavioral details.

| Variable | Default | Description |
|---|---|---|
| `BROWSER_DEVICE_PROFILE` | `desktop-windows` | One of `desktop-windows`, `desktop-macos`, `mobile-ios`, `mobile-android`. Controls UA, viewport, DPR, `is_mobile`, `has_touch`, navigator-override init script. |

## Browser Device Profiles

The shared Camoufox browser ships four device-emulation profiles, selected via `BROWSER_DEVICE_PROFILE`:

| Profile | UA family | Viewport | DPR | `is_mobile` | `has_touch` | `platform` | When to use |
|---|---|---|---|---|---|---|---|
| `desktop-windows` (default) | Firefox / Win64 | per-agent pool (1280×720 → 1920×1080) | 1.0 | false | false | `Win32` | Default for most sites; ≈70% real desktop market share. |
| `desktop-macos` | Firefox / macOS | per-agent pool | 2.0 | false | false | `MacIntel` | Targeting US/Western-Europe consumer sites that risk-score Mac users more leniently. |
| `mobile-ios` | Mobile Safari 17.5 / iPhone 14 Pro | 393×852 | 3.0 | true | true | `iPhone` | Site serves a better mobile experience; geography skewed mobile (mobile-first markets); detection systems calibrated for desktop bots. |
| `mobile-android` | Chrome 124 / Pixel 8 / Android 14 | 412×915 | 2.625 | true | true | `Linux armv8l` | Same as `mobile-ios`, alternative shape. Ships `navigator.userAgentData` with `mobile=true` (Mobile Safari does not). |

### When to choose mobile

- **Geography**: target audiences in markets where >70% of real traffic is mobile (most of South / Southeast Asia, Latin America, sub-Saharan Africa, mobile-only social platforms).
- **Site behavior**: target site serves a richer or simpler experience to mobile UAs (login flows, content pagination, ad load), or the desktop site is more aggressively bot-protected than the mobile site.
- **Detection profile**: some bot-detection vendors calibrate primarily on desktop fingerprint clusters; a clean mobile fingerprint can side-step desktop-only heuristics.

### Tradeoffs

- **Desktop-only forms reject mobile UAs**: enterprise SaaS and admin consoles often refuse mobile UAs entirely. Don't pick a mobile profile for those.
- **Different content**: many sites serve fewer features (less DOM, simplified menus) to mobile UAs — the agent's snapshot/find_text pipeline still works but the surface is smaller.
- **Camoufox compatibility**: Camoufox is built on Firefox, so mobile profiles ship a UA-engine mismatch (UA claims WebKit/Blink, engine is still Gecko). We mitigate at the surface layer (UA string, viewport, DPR, `is_mobile`, `has_touch`, `navigator.userAgentData`, `navigator.maxTouchPoints`, `navigator.platform`) but cannot spoof every nested API to fully match the claimed device. Sites that probe deep WebGL renderer strings, codec quirks, or vendor-specific CSS feature flags can still tell something is off. For high-accuracy mobile spoofing, a Chromium-based stack would be required; this implementation gives operators a usable mobile fingerprint for sites that gate on the easy-to-check signals.
- **Client Hints (`Sec-CH-UA-*`)**: Firefox doesn't send these and Camoufox can't synthesize them. Sites that key on Client Hints will see no mobile-specific Client Hints even with `mobile-android` selected.

### Per-agent override

`BROWSER_DEVICE_PROFILE` follows the standard [Browser Flag Precedence](#browser-flag-precedence). An operator can default the fleet to `desktop-windows` while pinning specific agents to `mobile-ios` for sites where mobile works better. Unknown values log a warning and fall back to `desktop-windows`.

```json
// config/settings.json — operator-wide default
{
  "browser_flags": {
    "BROWSER_DEVICE_PROFILE": "desktop-windows"
  }
}
```

```python
# Per-agent override (typically wired from the dashboard flags panel).
# NOTE: `flags.set_agent_override` mutates **in-process state** inside the
# browser service. Calling it from a separate Python process — e.g. a
# standalone script on the host — has no effect on the running container.
# Operators set per-agent overrides via the dashboard "Browser Flags"
# panel; the snippet below is for in-process dev / test wiring only.
from src.browser import flags
flags.set_agent_override("agent-mobile-scout", "BROWSER_DEVICE_PROFILE", "mobile-ios")
```
