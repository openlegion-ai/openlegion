# Configuration

OpenLegion uses YAML and JSON files in the `config/` directory. Config files are created during `openlegion start` (inline setup) and can be edited directly.

## Config Files

| File | Format | Purpose |
|------|--------|---------|
| `config/agents.yaml` | YAML | Agent definitions (role, model, skills, resources) |
| `config/mesh.yaml` | YAML | Mesh host settings, LLM defaults, channel config |
| `config/permissions.json` | JSON | Per-agent ACL matrix |
| `config/cron.json` | JSON | Scheduled job state (auto-managed) |
| `config/projects/` | Directory | Per-project data (project.md, members) |
| `config/settings.json` | JSON | Dashboard-managed runtime settings: browser speed/delay/timeout, execution limits (`max_iterations`, `chat_max_tool_rounds`, etc.), and default budgets. Written by the dashboard and injected as env vars into agent containers at startup. |
| `config/network.yaml` | YAML | Network settings (`no_proxy` exclusion list for proxy mode). |
| `.env` | dotenv | API keys and credentials |

## `config/agents.yaml`

Defines every agent in the fleet.

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
| `system_prompt` | string | No | Custom system prompt. Auto-generated if omitted. Also accepted as `instructions` |
| `resources.memory_limit` | string | No | Reserved for future use. Currently hardcoded to `384m` by the runtime for security. |
| `resources.cpu_limit` | float | No | Reserved for future use. Currently hardcoded to `0.15` by the runtime for security. |
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
  max_tokens: 4096
  temperature: 0.7
  failover:
    primary: anthropic/claude-sonnet-4-6
    fallback: openai/gpt-4.1-mini

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
| `llm.embedding_model` | string | *auto* | Model for memory embeddings. Auto-detected from default LLM provider (OpenAI → `text-embedding-3-small`; Anthropic, Google, DeepSeek, and all others → `"none"`). Must produce 1536-dim vectors. Set to `"none"` to disable vector search (FTS5 keyword search still works) |
| `llm.max_tokens` | integer | `4096` | Max output tokens per completion |
| `llm.temperature` | float | `0.7` | Sampling temperature |
| `llm.failover.primary` | string | -- | Primary model for failover routing |
| `llm.failover.fallback` | string | -- | Fallback model when primary fails |
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

Per-agent access control lists. Default policy is **deny** -- if not listed, it's blocked.

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

Blackboard patterns use the `projects/{name}/*` namespace. When an agent joins a project via `openlegion project add-agent`, it automatically receives read/write access to that project's namespace. Standalone agents (not in any project) get empty blackboard permissions. The `MeshClient` on the agent side transparently prefixes all blackboard keys with the project namespace, so agents use natural keys like `tasks/research_01` while data is stored under `projects/sales/tasks/research_01`.

### Permission Fields

| Field | Type | Description |
|-------|------|-------------|
| `can_message` | list[string] | Agent names this agent can send messages to |
| `can_publish` | list[string] | PubSub topics this agent can publish to |
| `can_subscribe` | list[string] | PubSub topics this agent can subscribe to |
| `blackboard_read` | list[string] | Glob patterns for readable blackboard keys |
| `blackboard_write` | list[string] | Glob patterns for writable blackboard keys |
| `allowed_apis` | list[string] | External APIs accessible through the vault proxy |
| `allowed_credentials` | list[string] | Glob patterns for accessible credential names. `["*"]` grants access to all agent-tier credentials; `[]` denies all. System credentials (LLM provider keys) are always blocked regardless of patterns. |
| `can_use_browser` | boolean | Whether this agent can use the shared browser service. Default: `false`. |
| `can_spawn` | boolean | Whether this agent can spawn ephemeral fleet agents via `spawn_fleet_agent`. Default: `false`. |
| `can_manage_cron` | boolean | Whether this agent can create, update, and delete cron jobs. Default: `false`. |
| `can_use_wallet` | boolean | Whether this agent can access the wallet signing service. Default: `false`. |
| `wallet_allowed_chains` | list[string] | Chains this agent can transact on (e.g., `["ethereum", "base"]`). `["*"]` allows all chains. Default: `[]`. |
| `wallet_spend_limit_per_tx_usd` | float | Max USD value per transaction. `0` uses the global default. |
| `wallet_spend_limit_daily_usd` | float | Daily aggregate transaction limit in USD. `0` uses the global default. |
| `wallet_rate_limit_per_hour` | integer | Max transactions per hour. `0` uses the global default. |
| `wallet_allowed_contracts` | list[string] | Contract addresses this agent can interact with. Empty list allows all. |

### Glob Patterns

- `*` matches everything
- `projects/myproject/*` matches all keys under the `myproject` namespace
- `tasks/*` matches `tasks/abc123`, `tasks/research_01`, etc.
- `context/prospect_*` matches `context/prospect_acme`, etc.

Note: Blackboard permissions are managed automatically when agents join/leave projects. You generally don't need to edit blackboard patterns by hand.

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
```

All credentials are loaded by the credential vault (`src/host/credentials.py`). Agents never see values directly -- they make API calls through the mesh proxy, which injects credentials server-side.

**Channel credential fallback:** Channel bot tokens (Telegram, Discord, Slack, WhatsApp) are resolved with a four-tier fallback chain: `mesh.yaml` channel config field (e.g. `channels.telegram.bot_token`) → `OPENLEGION_SYSTEM_<NAME>` → `OPENLEGION_CRED_<NAME>` → legacy unprefixed `<NAME>` (e.g., `TELEGRAM_BOT_TOKEN`). The `OPENLEGION_CRED_` prefix is recommended for channel tokens.

**Important:** LLM provider keys **must** use the `OPENLEGION_SYSTEM_` prefix. The mesh proxy only looks for provider keys in the system tier. A provider key stored with `OPENLEGION_CRED_` will be treated as an agent-tier credential and will not be used for LLM calls.

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
| `VNC_PORT` | `6080` | KasmVNC web port for browser viewing (set automatically in containers) |
| `MCP_SERVERS` | -- | JSON string of MCP server configs (set automatically in containers) |
| `MESH_AUTH_TOKEN` | -- | Agent auth token (set automatically in containers) |
| `MESH_HOST` | -- | Mesh host URL (set automatically in containers) |
| `AGENT_ID` | -- | Agent identifier (set automatically in containers) |
| `THINKING` | `off` | Extended thinking/reasoning mode (set automatically from `thinking` in agents.yaml) |
| `PROJECT_NAME` | -- | Project this agent belongs to (set automatically for project members) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for memory vector search (set automatically from `llm.embedding_model` in mesh.yaml). Set to `"none"` to disable vector search |
| `OPENLEGION_MAX_AGENTS` | `0` | Plan limit: maximum agents to start. `0` means unlimited. If set to N > 0, only the first N agents are started. |
| `OPENLEGION_MAX_PROJECTS` | -- | Plan limit: maximum projects allowed. If unset, projects are unlimited. If set to `0`, projects are disabled entirely. If set to N > 0, only N projects are allowed. |
| `OPENLEGION_HOST_NETWORK` | `0` | Use Docker host networking for agent containers instead of bridge network. Set to `1`, `true`, or `yes` (case-insensitive) to enable. Not recommended — disables network isolation. |
| `OPENLEGION_SYSTEM_WALLET_MASTER_SEED` | -- | BIP-39 mnemonic (24 words) for HD wallet key derivation. Required to enable wallet features. Generate with `openlegion wallet init`. |
| `BROWSER_OS` | `windows` | OS fingerprint for Camoufox browser: `windows`, `macos`, or `linux`. Windows is recommended (≈70% desktop market share; Linux is a datacenter signal). |
| `BROWSER_LOCALE` | `en-US` | BCP-47 locale tag for browser fingerprint (e.g. `en-US`, `de-DE`). |
| `BROWSER_UA_VERSION` | -- | Override Firefox version in User-Agent string (e.g. `138.0`). Useful when Camoufox's bundled Firefox is too old for sites that enforce minimum browser versions (e.g. Shopify). Uses Camoufox's native config system. |
| `BROWSER_PROXY_URL` | -- | Proxy URL for browser traffic (HTTP/HTTPS only, e.g. `http://proxy:8080`). SOCKS5 is not supported. Residential proxies recommended. |
| `BROWSER_PROXY_USER` | -- | Proxy authentication username. |
| `BROWSER_PROXY_PASS` | -- | Proxy authentication password. |
| `OPENLEGION_BROWSER_MAX_CONCURRENT` | `5` | Per-service cap on simultaneous Camoufox browser instances (clamped to `[1, 64]`). **Startup-only** — runtime reconfig is unsupported (would need to bound the acquire semaphore mid-flight). Restart the browser service to change this. The legacy name `MAX_BROWSERS` is still honored as a fallback for older deployments. |
| `OPENLEGION_SYSTEM_PROXY` | -- | System-wide outbound HTTP proxy URL for all agent traffic. Managed via the dashboard proxy settings page. |
| `HTTP_PROXY` / `HTTPS_PROXY` | -- | Per-agent proxy URLs. Auto-injected into agent containers by the runtime when a per-agent proxy is configured. Read by the agent-side `http_request` tool. |
| `OPENLEGION_TOOL_TIMEOUT` | `300` | Per-tool execution timeout in seconds (hard ceiling). |
| `OPENLEGION_MAX_ITERATIONS` | `20` | Maximum agent loop iterations per task (clamped 1–100). Overrides the default at the agent level. |
| `OPENLEGION_CHAT_MAX_TOOL_ROUNDS` | `30` | Maximum tool rounds per chat turn (clamped 1–200). |
| `OPENLEGION_CHAT_MAX_TOTAL_ROUNDS` | `200` | Maximum total chat rounds before session auto-continuation (clamped 1–1000). |

The mesh port is configured in `config/mesh.yaml` (`mesh.port`), not via environment variable.
