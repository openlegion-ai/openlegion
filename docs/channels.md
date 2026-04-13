# Channels

Channels bridge external messaging platforms to the OpenLegion mesh, providing the same multi-agent chat experience across CLI, Telegram, Discord, Slack, and WhatsApp.

## Overview

Every channel provides a unified interface:

- **Per-user active agent** -- each user has a "current" agent they're chatting with
- **@agent mentions** -- `@researcher find info about Acme Corp` routes to a specific agent
- **Slash commands** -- `/use`, `/agents`, `/status`, `/broadcast`, `/costs`, `/reset`, `/help`
- **Agent labels** -- every response is prefixed with `[agent_name]` so users know who's talking
- **Notifications** -- cron and heartbeat results push to all connected users

## Commands

### Channel Commands (available on all platforms)

| Command | Description |
|---------|-------------|
| `@agent <msg>` | Send message to a specific agent |
| `/use <agent>` | Switch your active agent |
| `/agents` | List all available agents |
| `/status` | Show agent health and task counts |
| `/broadcast <msg>` | Send a message to all agents |
| `/steer <msg>` | Inject message into busy agent's context (when configured) |
| `/debug [trace_id]` | Show recent traces or trace detail (when configured) |
| `/costs` | Show today's LLM spend per agent |
| `/addkey <service> <key>` | Add an API credential to the vault |
| `/reset` | Clear conversation with active agent |
| `/help` | Show command help |

### REPL-Only Commands (not available in external channels)

These commands require interactive prompts or local system access and are only available in the CLI REPL (`openlegion start`) or detached chat (`openlegion chat`):

| Command | Description |
|---------|-------------|
| `/add` | Add a new agent (interactive prompts) |
| `/agent [edit\|view]` | Agent overview, config editing, workspace file access |
| `/edit [name]` | Edit agent settings (model, budget, thinking) |
| `/remove [name]` | Remove an agent |
| `/restart [name]` | Restart an agent container |
| `/history [agent]` | Show recent conversation messages |
| `/blackboard [list\|get\|set\|del]` | View/edit shared blackboard entries |
| `/queue` | Show agent task queue status |
| `/cron [list\|del\|pause\|resume\|run]` | Manage cron jobs |
| `/project [list\|use\|info]` | Manage multi-project namespaces |
| `/credential [add\|list\|remove]` | Manage API credentials |
| `/removekey [name]` | Remove a credential from the vault |
| `/logs [--level LEVEL]` | Show recent runtime logs |
| `/traces [trace_id]` | Alias for `/debug` |

## CLI REPL

The default interface. Starts automatically with `openlegion start`.

```bash
openlegion start          # Interactive mode
openlegion start -d       # Detached (background)
openlegion chat <agent>   # Connect to running agent (detached mode)
```

The CLI REPL supports all channel commands above plus the REPL-only commands listed in the table. It also provides token-level streaming responses with tool-use progress indicators and tab completion for agent names and subcommands.

## Env Var Lookup

All channel tokens are resolved via a three-tier lookup in this order:

1. `OPENLEGION_SYSTEM_<NAME>` — host-level (recommended for operators; system-tier credentials are never exposed to agents)
2. `OPENLEGION_CRED_<NAME>` — agent-accessible credential tier
3. Bare env var (e.g. `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN`) — convenience for quick local setup

For production deployments, prefer the `OPENLEGION_SYSTEM_*` form so that token values never enter the agent credential vault.

Examples for Telegram:
```bash
# Preferred (host-level):
OPENLEGION_SYSTEM_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# Also accepted:
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

The same pattern applies to Discord (`DISCORD_BOT_TOKEN`), Slack (`SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`), and WhatsApp (`WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`).

## Telegram

### Setup

Set your bot token in `.env`:

```bash
OPENLEGION_CRED_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

Enable in `config/mesh.yaml`:

```yaml
channels:
  telegram:
    enabled: true
    default_agent: assistant
```

### Pairing

Telegram uses a pairing code for security. On first start, a code is printed to the console:

```
Telegram pairing code: ABC123
Send /start ABC123 to the bot to pair.
```

The first user to send `/start ABC123` becomes the **owner**. The owner can then allow other users:

| Command | Description |
|---------|-------------|
| `/start <code>` | Pair with the bot (first user becomes owner) |
| `/allow <user_id>` | Owner: allow a Telegram user |
| `/revoke <user_id>` | Owner: revoke a user's access |
| `/paired` | Owner: list paired users |

After pairing, the bot sends a help summary with all available commands. Unauthorized users receive a one-time access denial message with their user ID.

Pairing state is stored in `config/telegram_paired.json`.

### Features

- Token-level streaming with progressive message editing (debounced at 500ms)
- Tool progress indicators (numbered tool list with checkmarks)
- Markdown formatting converted to Telegram HTML
- Messages chunked at 4000 characters with overflow handling
- Per-user agent tracking (each Telegram user has their own active agent)

## Discord

### Setup

Set your bot token in `.env` (see [Env Var Lookup](#env-var-lookup) for all accepted forms):

```bash
OPENLEGION_CRED_DISCORD_BOT_TOKEN=MTIz...
```

Enable in `config/mesh.yaml`:

```yaml
channels:
  discord:
    enabled: true
    default_agent: assistant
    allowed_guilds: [987654321]  # Optional: restrict to specific servers
```

### Bot Permissions

When creating the Discord bot in the [Developer Portal](https://discord.com/developers/applications), enable:

- **Message Content Intent** -- required for reading `!` prefix fallback messages
- **Bot permissions**: Send Messages, Read Message History, Add Reactions
- **OAuth2 scopes**: `bot`, `applications.commands` (required for slash commands)

### Slash Commands

All commands are registered as native Discord slash commands via `CommandTree`. They appear in Discord's autocomplete when users type `/`. Commands sync automatically on bot startup -- guild-specific when `allowed_guilds` is set (instant), global otherwise (may take up to an hour for Discord to propagate).

The `!` prefix is still supported as a fallback via regular messages (e.g. `!agents`, `!start <code>`).

**Note:** `/addkey` is intentionally not registered as a slash command because parameter values are visible in Discord's command UI. Use `!addkey <service> <key>` instead.

### Pairing

Discord uses the same pairing pattern as Telegram:

| Command | Description |
|---------|-------------|
| `/start <code>` | Pair with the bot (first user becomes owner) |
| `/allow <user_id>` | Owner: allow a Discord user |
| `/revoke <user_id>` | Owner: revoke a user's access |
| `/paired` | Owner: list paired users |

After pairing, the bot sends a help summary with all available commands. Unauthorized users receive a one-time access denial message with their user ID. Security commands (`/start`, `/allow`, `/revoke`, `/paired`) respond ephemerally -- only visible to the invoker.

Pairing state is stored in `config/discord_paired.json`.

### Features

- Native Discord slash commands with autocomplete
- `!` prefix fallback for all commands via regular messages
- Token-level streaming with progressive message editing (debounced at 500ms)
- Tool progress summary prepended to streaming responses
- Typing indicators during dispatch
- Messages chunked at 1900 characters (Discord limit) with overflow handling
- Per-user agent tracking
- Optional guild (server) allowlisting

## Slack

### Setup

Set your tokens in `.env` (see [Env Var Lookup](#env-var-lookup) for all accepted forms):

```bash
OPENLEGION_CRED_SLACK_BOT_TOKEN=xoxb-...
OPENLEGION_CRED_SLACK_APP_TOKEN=xapp-...
```

Enable in `config/mesh.yaml`:

```yaml
channels:
  slack:
    enabled: true
    default_agent: assistant
```

### Requirements

Slack uses **Socket Mode** (no public URL needed). In the [Slack API dashboard](https://api.slack.com/apps):

1. Enable **Socket Mode** and generate an app-level token (`xapp-...`)
2. Add **Bot Token Scopes**: `chat:write`, `app_mentions:read`, `channels:history`, `im:history`
3. Enable **Event Subscriptions**: `message.channels`, `message.im`, `app_mention`

### Pairing

Slack uses the same pairing pattern as other channels:

| Command | Description |
|---------|-------------|
| `/start <code>` | Pair with the bot (first user becomes owner) |
| `/allow <user_id>` | Owner: allow a Slack user |
| `/revoke <user_id>` | Owner: revoke a user's access |
| `/paired` | Owner: list paired users |

After pairing, the bot sends a help summary with all available commands. Unauthorized users receive a one-time access denial message with their user ID.

### Features

- Token-level streaming with progressive message editing via `chat.update` (debounced at 500ms)
- Tool progress summary prepended to streaming responses
- Thread-aware routing (each thread maps to its own agent context)
- `!`-prefix commands still accepted (translated to `/` internally)
- Messages chunked at 3000 characters with overflow handling
- Per-user agent tracking via composite `user_id:thread_ts` key

## WhatsApp

### Setup

Set your tokens in `.env`. All channel tokens support a three-tier lookup (see [Env Var Lookup](#env-var-lookup)):

```bash
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...
```

In production you must also set the app secret for webhook signature verification (see [Security](#security) below):

```bash
WHATSAPP_APP_SECRET=<your-app-secret-from-meta-dashboard>
```

Enable in `config/mesh.yaml`:

```yaml
channels:
  whatsapp:
    enabled: true
    default_agent: assistant
```

### Requirements

WhatsApp uses the **Cloud API** with webhook-based message delivery. In the [Meta Developer Portal](https://developers.facebook.com/):

1. Create a WhatsApp Business app
2. Generate a permanent access token
3. Configure the webhook URL: `https://your-server:8420/channels/whatsapp/webhook`
4. Subscribe to `messages` webhook field

### Security

**Production deployments must set `WHATSAPP_APP_SECRET`.** Without it, `X-Hub-Signature-256` webhook signature verification is disabled — any HTTP client can inject arbitrary messages. When `MESH_AUTH_TOKEN` is set (i.e., production mode) and `WHATSAPP_APP_SECRET` is absent, startup raises a `RuntimeError`.

Set the app secret to the value shown in the Meta dashboard under your WhatsApp app → App Settings → App Secret:

```bash
WHATSAPP_APP_SECRET=abc123...
```

The secret is read directly from the `WHATSAPP_APP_SECRET` environment variable (not via the `OPENLEGION_*` credential prefix).

### Verify Token

The webhook verification token is auto-generated as a random `secrets.token_hex(16)` value if none is configured. To use a deterministic, replay-safe token instead (recommended for reproducible deployments), set it explicitly:

```bash
OPENLEGION_SYSTEM_WHATSAPP_VERIFY_TOKEN=my-stable-verify-token
# or
OPENLEGION_CRED_WHATSAPP_VERIFY_TOKEN=my-stable-verify-token
```

The token configured here must match what you enter in the Meta Developer Portal webhook settings.

### Pairing

WhatsApp uses the same pairing pattern:

| Command | Description |
|---------|-------------|
| `/start <code>` | Pair with the bot (first user becomes owner) |
| `/allow <phone>` | Owner: allow a phone number |
| `/revoke <phone>` | Owner: revoke access |
| `/paired` | Owner: list paired users |

After pairing, the bot sends a help summary with all available commands. Unauthorized users receive a one-time access denial message with their phone number.

### Features

- Text messages only. Non-text messages (images, audio, documents, etc.) receive a reply only when pairing is already complete **and** the sender is an allowed user. Before pairing is complete (no owner set yet), non-text messages are silently dropped.
- Messages chunked at 4096 characters
- Per-user agent tracking by phone number
- Webhook verification challenge handled automatically

## Webhooks

HTTP webhooks for programmatic integration. Incoming payloads are dispatched to agents as tasks.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook/hook/<hook_id>` | Dispatch a JSON payload to the configured agent |

### Usage

```bash
# Send a payload to a webhook
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme Corp", "source": "website"}'
```

Each webhook is configured with a target agent. When a payload arrives, it is dispatched to that agent as a task.

## Writing a Custom Channel

Subclass `Channel` from `src/channels/base.py`:

```python
from src.channels.base import Channel

class MyChannel(Channel):
    async def start(self) -> None:
        # Connect to your platform, start listening for messages
        pass

    async def stop(self) -> None:
        # Graceful shutdown
        pass

    async def send_notification(self, text: str) -> None:
        # Push cron/heartbeat results to users
        pass
```

The base class provides `handle_message()` which handles all command parsing, @mention routing, and agent dispatch. Your subclass only needs to bridge the platform's message transport.

### Callback Functions

The channel receives these callbacks at construction:

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `dispatch_fn` | `(agent, message) -> str` | Route message to agent |
| `stream_dispatch_fn` | `(agent, message) -> AsyncIterator[dict]` | Token-level streaming dispatch (yields `text_delta`, `tool_start`, `tool_result`, `done` events) |
| `list_agents_fn` | `() -> dict` | List available agents |
| `status_fn` | `(agent) -> dict \| None` | Agent health info |
| `costs_fn` | `() -> list[dict]` | Today's spend per agent |
| `reset_fn` | `(agent) -> bool` | Clear agent conversation |
| `addkey_fn` | `(service, key) -> None` | Store credential in vault |
| `steer_fn` | `(agent, msg) -> None` | Inject message into agent's context |
| `debug_fn` | `(trace_id \| None) -> list[dict]` | Retrieve recent traces or trace detail |

## Source Files

| File | Role |
|------|------|
| `src/channels/base.py` | Abstract `Channel` class with command handling |
| `src/channels/telegram.py` | Telegram bot adapter |
| `src/channels/discord.py` | Discord bot adapter |
| `src/channels/slack.py` | Slack adapter (Socket Mode via slack-bolt) |
| `src/channels/whatsapp.py` | WhatsApp Cloud API adapter |
| `src/cli/` | CLI package (uses same dispatch pattern) |
