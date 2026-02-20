# Channels

Channels bridge external messaging platforms to the OpenLegion mesh, providing the same multi-agent chat experience across CLI, Telegram, and Discord.

## Overview

Every channel provides a unified interface:

- **Per-user active agent** -- each user has a "current" agent they're chatting with
- **@agent mentions** -- `@researcher find info about Acme Corp` routes to a specific agent
- **Slash commands** -- `/use`, `/agents`, `/status`, `/broadcast`, `/costs`, `/reset`, `/help`
- **Agent labels** -- every response is prefixed with `[agent_name]` so users know who's talking
- **Notifications** -- cron and heartbeat results push to all connected users

## Commands

All channels support the same command set:

| Command | Description |
|---------|-------------|
| `@agent <msg>` | Send message to a specific agent |
| `/use <agent>` | Switch your active agent |
| `/agents` | List all available agents |
| `/status` | Show agent health and task counts |
| `/broadcast <msg>` | Send a message to all agents |
| `/costs` | Show today's LLM spend per agent |
| `/addkey <service> <key>` | Add an API credential to the vault |
| `/reset` | Clear conversation with active agent |
| `/help` | Show command help |

## CLI REPL

The default interface. Starts automatically with `openlegion start`.

```bash
openlegion start          # Interactive mode
openlegion start -d       # Detached (background)
openlegion chat <agent>   # Connect to running agent (detached mode)
```

The CLI REPL supports the full command set above plus streaming responses with tool-use progress indicators.

## Telegram

### Setup

```bash
openlegion channels add telegram
```

This prompts for your bot token. Alternatively, set it in `.env`:

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

Pairing state is stored in `config/telegram_paired.json`.

### Features

- Typing indicators while agents process
- Streaming responses with tool progress updates
- Markdown formatting converted to Telegram HTML
- Messages chunked at 4000 characters
- Per-user agent tracking (each Telegram user has their own active agent)

## Discord

### Setup

```bash
openlegion channels add discord
```

This prompts for your bot token. Alternatively, set it in `.env`:

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

- **Message Content Intent** -- required for reading messages
- **Bot permissions**: Send Messages, Read Message History, Add Reactions

### Pairing

Discord uses the same pairing pattern as Telegram, but with `!start` instead of `/start`:

| Command | Description |
|---------|-------------|
| `!start <code>` | Pair with the bot |
| `!allow <user_id>` | Owner: allow a Discord user |
| `!revoke <user_id>` | Owner: revoke a user's access |

Pairing state is stored in `config/discord_paired.json`.

### Features

- Typing indicators during dispatch
- Messages chunked at 1900 characters (Discord limit)
- Per-user agent tracking
- Optional guild (server) allowlisting

## Slack

### Setup

```bash
openlegion channels add slack
```

This prompts for a bot token and app-level token. Alternatively, set them in `.env`:

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
| `!start <code>` | Pair with the bot |
| `!allow <user_id>` | Owner: allow a Slack user |
| `!revoke <user_id>` | Owner: revoke a user's access |

### Features

- Thread-aware routing (each thread maps to its own agent context)
- `!`-prefix commands translated to `/` internally
- Messages chunked at 3000 characters
- Per-user agent tracking via composite `user_id:thread_ts` key

## WhatsApp

### Setup

```bash
openlegion channels add whatsapp
```

This prompts for your access token and phone number ID. Alternatively, set them in `.env`:

```bash
OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN=EAAx...
OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID=1234...
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

### Pairing

WhatsApp uses the same pairing pattern:

| Command | Description |
|---------|-------------|
| Send pairing code | First user to send the code becomes owner |
| `/allow <phone>` | Owner: allow a phone number |
| `/revoke <phone>` | Owner: revoke access |

### Features

- Text messages only (media logged and skipped)
- Messages chunked at 4096 characters
- Per-user agent tracking by phone number
- Webhook verification challenge handled automatically

## Webhooks

HTTP webhooks for programmatic integration and workflow triggering.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook/hook/<hook_id>` | Trigger a workflow with JSON payload |

### Usage

```bash
# Trigger a workflow
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme Corp", "source": "website"}'
```

Webhooks connect to the workflow engine. Each webhook maps to a workflow trigger topic. See [Workflows](workflows.md) for details.

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
| `stream_dispatch_fn` | `(agent, message) -> AsyncIterator[dict]` | Streaming dispatch (SSE events) |
| `list_agents_fn` | `() -> dict` | List available agents |
| `status_fn` | `(agent) -> dict \| None` | Agent health info |
| `costs_fn` | `() -> list[dict]` | Today's spend per agent |
| `reset_fn` | `(agent) -> bool` | Clear agent conversation |
| `addkey_fn` | `(service, key) -> None` | Store credential in vault |

## Source Files

| File | Role |
|------|------|
| `src/channels/base.py` | Abstract `Channel` class with command handling |
| `src/channels/telegram.py` | Telegram bot adapter |
| `src/channels/discord.py` | Discord bot adapter |
| `src/channels/slack.py` | Slack adapter (Socket Mode via slack-bolt) |
| `src/channels/whatsapp.py` | WhatsApp Cloud API adapter |
| `src/channels/webhook.py` | HTTP webhook adapter |
| `src/cli.py` | CLI REPL (uses same dispatch pattern) |
