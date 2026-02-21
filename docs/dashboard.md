# Dashboard

Real-time web dashboard for fleet observability and management.

## Overview

The dashboard is served at `http://localhost:8420/dashboard` (or whatever port the mesh is configured on). It provides a live view of your agent fleet with nine panels for monitoring, configuration, and debugging.

No additional setup is required -- the dashboard starts automatically with `openlegion start`.

## Panels

### Fleet

Overview of all registered agents showing health status, activity state (idle/thinking/tool), daily cost, token usage, and restart count. Click any agent card to drill down into its detail view with cost breakdowns, budget bars, and recent events.

### Events

Real-time event feed streamed via WebSocket. Events include LLM calls, tool executions, messages sent/received, blackboard writes, cost updates, and health changes. Filter by event type using the chip toggles. Events are capped at 500 in the browser; older events are dropped.

### Agents

Agent configuration management. View and edit each agent's model, browser backend, role, system prompt, and daily budget. Changes that require a restart (model, browser) are flagged. Use the Restart button to apply changes. Agent configs are auto-loaded when switching to this tab.

### Blackboard

Browse, search, write, and delete shared state entries. Filter by key prefix (e.g., `tasks/`, `context/`, `signals/`). Click a value to expand/collapse. New entries are highlighted with a flash animation when written by agents in real-time. History namespace entries (`history/*`) cannot be deleted.

### Costs

Per-agent LLM spend with period selector (today/week/month). Bar chart shows cost and token usage side-by-side. Budget status bars show daily spend vs. configured limits. Cost data refreshes automatically when `cost_update` events arrive.

### Traces

Request trace timeline. The left panel lists recent trace events; clicking one shows its full event chain in the right panel with a waterfall timing visualization. Traces cover the full request path: dispatch, LLM calls, tool executions, blackboard writes, and pub/sub events.

### Queues

Live view of per-agent task queue depth, pending/collected counts, and busy/idle status. Auto-refreshes every 5 seconds while the tab is active.

### Cron

Manage scheduled jobs. View schedule, last run time, run count, error count, and heartbeat status. Actions: Run (trigger immediately), Pause, Resume. Auto-refreshes every 10 seconds while the tab is active.

### Settings

Environment overview showing configured credentials (names only, never values), pub/sub subscriptions, model pricing tables, and available browser backends.

## Agent Management

### Edit Agent Config

1. Switch to the **Agents** tab
2. Click **Edit** on an agent card
3. Modify model, browser backend, role, system prompt, or daily budget
4. Click **Save** -- a toast confirms which fields were updated
5. If the change requires a restart (model or browser), click **Restart**

### Restart Agent

Click the **Restart** button on any agent card. A confirmation dialog prevents accidental restarts. The agent is stopped and restarted with its current configuration. The fleet panel updates automatically when the agent comes back online.

### Update Budget

From the agent detail view (click an agent in Fleet), budget bars show current daily and monthly usage. Budget can also be updated via the Agents tab edit form.

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
| `GET` | `/dashboard/api/agents` | Fleet overview with health and costs |
| `GET` | `/dashboard/api/agents/{id}` | Agent detail with spend and budget |
| `GET` | `/dashboard/api/agents/{id}/config` | Agent configuration |
| `PUT` | `/dashboard/api/agents/{id}/config` | Update agent configuration |
| `POST` | `/dashboard/api/agents/{id}/restart` | Restart an agent |
| `PUT` | `/dashboard/api/agents/{id}/budget` | Update agent budget |
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
