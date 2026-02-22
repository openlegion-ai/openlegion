# Configuration

OpenLegion uses YAML and JSON files in the `config/` directory. All config files are created during `openlegion setup` and can be edited directly.

## Config Files

| File | Format | Purpose |
|------|--------|---------|
| `config/agents.yaml` | YAML | Agent definitions (role, model, skills, resources) |
| `config/mesh.yaml` | YAML | Mesh host settings, LLM defaults, channel config |
| `config/permissions.json` | JSON | Per-agent ACL matrix |
| `config/cron.json` | JSON | Scheduled job state (auto-managed) |
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
      memory_limit: 512m
      cpu_limit: 0.5
    budget:
      daily_usd: 5.00
      monthly_usd: 100.00
    browser_backend: stealth
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
| `model` | string | No | LLM model in `provider/model` format. Falls back to `llm.default_model` in mesh.yaml |
| `skills_dir` | string | No | Path to custom skills directory |
| `system_prompt` | string | No | Custom system prompt. Auto-generated if omitted |
| `resources.memory_limit` | string | No | Docker memory limit (default: `512m`) |
| `resources.cpu_limit` | float | No | CPU quota, 0.5 = 50% (default: `0.5`) |
| `budget.daily_usd` | float | No | Daily spend cap in USD |
| `budget.monthly_usd` | float | No | Monthly spend cap in USD |
| `browser_backend` | string | No | Browser tier: `basic` (default), `stealth` (Camoufox), or `advanced` (Bright Data CDP) |
| `thinking` | string | No | Extended thinking/reasoning mode: `off` (default), `low`, `medium`, or `high`. Anthropic models use thinking budgets (5K/10K/25K tokens). OpenAI o-series models use `reasoning_effort`. Ignored for unsupported models |
| `mcp_servers` | list | No | External MCP tool servers. See [MCP Integration](mcp.md) |

### Model Format

Models use `provider/model_id` format, routed through [LiteLLM](https://docs.litellm.ai/):

```yaml
# Anthropic
model: anthropic/claude-haiku-4-5-20251001
model: anthropic/claude-sonnet-4-6

# OpenAI
model: openai/gpt-4o-mini
model: openai/gpt-4o

# Google
model: gemini/gemini-2.0-flash

# Other supported providers
model: deepseek/deepseek-chat
model: xai/grok-2
model: groq/llama-3.3-70b-versatile
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
    fallback: openai/gpt-4o-mini

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
| `llm.embedding_model` | string | *auto* | Model for memory embeddings. Auto-detected from default LLM provider (OpenAI → `text-embedding-3-small`, others → `none`). Must produce 1536-dim vectors. Set to `"none"` to disable vector search (FTS5 keyword search still works) |
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
      "can_message": ["orchestrator"],
      "can_publish": ["research_complete"],
      "can_subscribe": ["new_lead"],
      "blackboard_read": ["tasks/*", "context/*"],
      "blackboard_write": ["context/prospect_*"],
      "allowed_apis": ["llm", "brave_search"]
    },
    "writer": {
      "can_message": ["*"],
      "can_publish": ["*"],
      "can_subscribe": ["*"],
      "blackboard_read": ["*"],
      "blackboard_write": ["artifacts/*"],
      "allowed_apis": ["llm"]
    }
  }
}
```

### Permission Fields

| Field | Type | Description |
|-------|------|-------------|
| `can_message` | list[string] | Agent names this agent can send messages to |
| `can_publish` | list[string] | PubSub topics this agent can publish to |
| `can_subscribe` | list[string] | PubSub topics this agent can subscribe to |
| `blackboard_read` | list[string] | Glob patterns for readable blackboard keys |
| `blackboard_write` | list[string] | Glob patterns for writable blackboard keys |
| `allowed_apis` | list[string] | External APIs accessible through the vault proxy |

### Glob Patterns

- `*` matches everything
- `tasks/*` matches `tasks/abc123`, `tasks/research_01`, etc.
- `context/prospect_*` matches `context/prospect_acme`, etc.

## `.env` — Credentials

API keys and secrets. Never committed to git.

```bash
# LLM Provider Keys (at least one required)
OPENLEGION_CRED_ANTHROPIC_API_KEY=sk-ant-...
OPENLEGION_CRED_OPENAI_API_KEY=sk-...
OPENLEGION_CRED_GEMINI_API_KEY=...

# Channel Bot Tokens (optional)
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC-...
OPENLEGION_CRED_DISCORD_BOT_TOKEN=MTIz...
OPENLEGION_CRED_SLACK_BOT_TOKEN=xoxb-...
OPENLEGION_CRED_SLACK_APP_TOKEN=xapp-...
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...

# External API Keys (optional)
OPENLEGION_CRED_BRAVE_SEARCH_API_KEY=BSA...
```

All `OPENLEGION_CRED_*` variables are loaded by the credential vault (`src/host/credentials.py`). Agents never see these values directly -- they make API calls through the mesh proxy, which injects credentials server-side.

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
| `BROWSER_BACKEND` | `basic` | Browser tier: `basic`, `stealth`, or `advanced` (set automatically in containers) |
| `MCP_SERVERS` | -- | JSON string of MCP server configs (set automatically in containers) |
| `MESH_AUTH_TOKEN` | -- | Agent auth token (set automatically in containers) |
| `MESH_HOST` | -- | Mesh host URL (set automatically in containers) |
| `AGENT_ID` | -- | Agent identifier (set automatically in containers) |
| `THINKING` | `off` | Extended thinking/reasoning mode (set automatically from `thinking` in agents.yaml) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for memory vector search (set automatically from `llm.embedding_model` in mesh.yaml). Set to `"none"` to disable vector search |

The mesh port is configured in `config/mesh.yaml` (`mesh.port`), not via environment variable.
