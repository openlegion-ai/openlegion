# Dashboard

Real-time web dashboard for fleet observability and management.

## Overview

The dashboard is served at `http://localhost:8420/dashboard` (or whatever port the mesh is configured on). It provides a live view of your agent fleet across six tabs for monitoring, configuration, and debugging.

No additional setup is required -- the dashboard starts automatically with `openlegion start`.

## Panels

### Agents

Overview of all registered agents showing health status, activity state (idle/thinking/tool), daily cost, token usage, and restart count. Click any agent card to drill down into its detail view with cost breakdowns, budget bars, workspace file editor, and recent events. Also includes agent configuration management — view and edit each agent's model, browser backend, role, system prompt, and daily budget. Changes that require a restart (model, browser) are flagged.

### Activity

Two sub-views toggled via a tab bar at the top of the panel:

**Traces** (default) — Grouped request traces showing the full lifecycle of each request through the system. Each trace row shows the triggering event type, participating agents, a prompt preview (extracted from the first LLM call's user message), event count, total duration, error badge, and relative time. Click a trace to expand an inline event timeline with a vertical waterfall: each event shows its type, source, agent, duration, detail text, error message (if any), and metadata fields (model, tokens, prompt/response previews). Traces auto-refresh every 10 seconds and on relevant WebSocket events (debounced). LLM call events include prompt and response preview fields extracted from the mesh API proxy.

**Live Events** — Real-time event feed streamed via WebSocket. Events include LLM calls, tool executions, text streaming deltas, messages sent/received, blackboard writes, agent state changes, and health changes. Filter by event type using the chip toggles. Events are capped at 500 in the browser; older events are dropped. Click any event row to expand an inline detail panel showing all available data fields with type-specific formatting (e.g. model, token breakdown, cost, and prompt/response previews for `llm_call`; full untruncated arguments for `tool_start`; complete message text for `message_sent`). The same expandable rows appear in the agent detail view's Recent Events section. Note: per-token `text_delta` events are delivered via the direct streaming chat endpoint (not the WebSocket event bus) to avoid flooding the event buffer.

### Blackboard

Browse, search, write, and delete shared state entries. Filter by key prefix (e.g., `tasks/`, `context/`, `signals/`). Click a value to expand/collapse. New entries are highlighted with a flash animation when written by agents in real-time. History namespace entries (`history/*`) cannot be deleted.

### Costs

Per-agent LLM spend with period selector (today/week/month). Bar chart shows cost and token usage side-by-side. Budget status bars show daily spend vs. configured limits. Cost data refreshes automatically when `llm_call` events arrive.

### Automation

Manage scheduled jobs. View schedule, last run time, run count, error count, and heartbeat status. Actions: Run (trigger immediately), Pause, Resume, Edit schedule, Delete. Auto-refreshes every 10 seconds while the tab is active.

### System

Environment overview showing configured credentials (names only, never values), pub/sub subscriptions, model pricing tables, and available browser backends.

## Agent Management

### Edit Agent Config

1. Switch to the **Agents** tab
2. Click **Edit** on an agent card
3. Modify model, browser backend, role, system prompt, or daily budget
4. Click **Save** -- a toast confirms which fields were updated
5. If the change requires a restart (model or browser), click **Restart**

### Restart Agent

Click the **Restart** button on any agent card. A confirmation dialog prevents accidental restarts. The agent is stopped and restarted with its current configuration. The agents panel updates automatically when the agent comes back online.

### Update Budget

From the agent detail view (click an agent in Agents), budget bars show current daily and monthly usage. Budget can also be updated via the Agents tab edit form.

## Workspace Editing

The agent detail view includes a workspace file editor for viewing and editing an agent's core workspace files directly from the dashboard.

### Editable Files

| File | Purpose |
|------|---------|
| `SOUL.md` | Agent personality and behavioral guidelines |
| `HEARTBEAT.md` | Autonomous monitoring rules |
| `USER.md` | User preferences and working style |
| `AGENTS.md` | Fleet context (other agents, their roles) |
| `MEMORY.md` | Curated long-term facts |

### Usage

1. Click an agent card to open the detail view
2. The **Workspace Files** panel shows all editable files with their sizes
3. Click a filename to open the inline editor
4. Edit the content in the monospace textarea
5. Click **Save** to write changes (content is sanitized for invisible Unicode before saving)
6. Click **Cancel** to discard changes

Files with size 0 are shown with an "empty" indicator. The editor uses the same `sanitize_for_prompt()` sanitization applied at all other trust boundaries.

The dashboard proxies workspace operations through the mesh transport layer to the agent's container — files are read from and written to the agent's `/data/workspace` volume.

## Chat

### Streaming Chat

Click an agent card to open the chat modal. Messages stream in real-time with token-level updates via SSE (`POST /dashboard/api/agents/{id}/chat/stream`). The response renders progressively as tokens arrive, with a pulsing cursor indicating active streaming.

### Tool Call Display

When an agent calls tools during a response, each tool appears as an inline pill inside the message bubble:
- **Running** — spinning indicator with tool name
- **Done** — green checkmark with truncated output preview (200 chars max)

Tool calls appear above the text response, in the order they were executed.

### Chat History

Conversation history persists per agent across modal open/close — reopening the same agent shows the full conversation. Use the **Clear** button in the chat header to reset history for that agent. History is stored in browser memory and resets on page reload.

### Keyboard Shortcuts

- **Enter** — send message
- **Escape** — close chat modal

### Abort

Closing the chat modal (click outside, X button, or Escape) while a response is streaming cancels the in-flight request. The partial response is preserved in history.

## Broadcast

Send a message to all agents simultaneously using the broadcast bar below the agent grid. Each agent processes the message independently. Responses display inline with expand/collapse for long replies (200+ characters).

## Blackboard Operations

### Write Entry

1. Click **+ New** to open the write form
2. Enter a key (e.g., `context/my_data`) and JSON value
3. Click **Save**

### Delete Entry

Click **Del** on any row. A confirmation dialog prevents accidental deletion. History namespace entries are protected and cannot be deleted.

## Real-Time Updates

The dashboard connects to the mesh via WebSocket at `/ws/events`. Events are streamed in real-time with optional agent and type filters. The connection indicator in the top-right shows live/disconnected status. On disconnect, the WebSocket client automatically reconnects with exponential backoff.

## API Endpoints

All dashboard API endpoints are prefixed with `/dashboard/api/`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/` | Serve dashboard HTML |
| `GET` | `/dashboard/api/agents` | Agent overview with health and costs |
| `POST` | `/dashboard/api/agents` | Create a new agent |
| `GET` | `/dashboard/api/agents/{id}` | Agent detail with spend and budget |
| `DELETE` | `/dashboard/api/agents/{id}` | Remove an agent |
| `GET` | `/dashboard/api/agents/{id}/config` | Agent configuration |
| `PUT` | `/dashboard/api/agents/{id}/config` | Update agent configuration |
| `POST` | `/dashboard/api/agents/{id}/chat/stream` | SSE streaming chat (token-level) |
| `POST` | `/dashboard/api/agents/{id}/steer` | Update agent system prompt live |
| `POST` | `/dashboard/api/agents/{id}/restart` | Restart an agent |
| `PUT` | `/dashboard/api/agents/{id}/budget` | Update agent budget |
| `GET` | `/dashboard/api/agents/{id}/workspace` | List agent workspace files |
| `GET` | `/dashboard/api/agents/{id}/workspace/{file}` | Read workspace file content |
| `PUT` | `/dashboard/api/agents/{id}/workspace/{file}` | Write workspace file content |
| `GET` | `/dashboard/api/blackboard` | List blackboard entries |
| `PUT` | `/dashboard/api/blackboard/{key}` | Write blackboard entry |
| `DELETE` | `/dashboard/api/blackboard/{key}` | Delete blackboard entry |
| `GET` | `/dashboard/api/costs` | Cost data with optional period |
| `GET` | `/dashboard/api/traces` | Recent trace events |
| `GET` | `/dashboard/api/traces/{id}` | Trace detail |
| `GET` | `/dashboard/api/queues` | Queue status per agent |
| `GET` | `/dashboard/api/cron` | List cron jobs |
| `POST` | `/dashboard/api/cron/{id}/run` | Trigger cron job |
| `POST` | `/dashboard/api/cron/{id}/pause` | Pause cron job |
| `POST` | `/dashboard/api/cron/{id}/resume` | Resume cron job |
| `GET` | `/dashboard/api/settings` | Environment settings |
| `POST` | `/dashboard/api/broadcast` | Send message to all agents |
| `GET` | `/dashboard/api/messages` | Recent message log |
| `GET` | `/dashboard/api/workflows` | Workflow definitions |
| `WS` | `/ws/events` | Real-time event stream |

## Source Files

| File | Role |
|------|------|
| `src/dashboard/server.py` | FastAPI router with all API endpoints |
| `src/dashboard/templates/index.html` | Dashboard HTML (Alpine.js + Tailwind) |
| `src/dashboard/static/js/app.js` | Dashboard application logic |
| `src/dashboard/static/js/websocket.js` | WebSocket client with reconnect |
| `src/dashboard/static/css/dashboard.css` | Custom styles |
| `src/dashboard/events.py` | EventBus for real-time event distribution |
