# Dashboard

Real-time web dashboard for fleet observability and management.

## Overview

The dashboard is served at `http://localhost:8420/dashboard` (or whatever port the mesh is configured on). It provides a live view of your agent fleet across three main tabs with a consolidated navigation bar, slide-over chat panels, and a keyboard command palette.

No additional setup is required -- the dashboard starts automatically with `openlegion start`.

## Navigation

The dashboard uses a consolidated three-tab layout:

| Tab | What It Shows |
|-----|--------------|
| **Fleet** | Agent cards, agent detail views, configuration editing |
| **Activity** | Traces, live events, blackboard, costs, and automation |
| **System** | Credentials, pub/sub, model pricing |

A command palette (**Cmd+K** / **Ctrl+K**) provides quick access to agents, actions, and navigation. The search button in the nav bar also opens it.

## Project Management

The dashboard supports multi-project namespaces for organizing agents into isolated groups.

### Project Switcher

A tab bar at the top of the Fleet view shows all projects plus an "All Agents" view. Click a project tab to filter the agent grid to that project's members. The "All Agents" tab shows every agent with project badges. Each tab displays a member count.

### Create Project

Click the **+** button next to the project tabs to create a new project. Enter a name and optional description. Projects are stored in `config/projects/{name}/`.

### Project Members

When a project is selected:
- An "Add member" dropdown lists unassigned agents
- Each member card shows a "Remove" button
- Adding/removing a member auto-restarts the agent so the new scope takes effect

### Delete Project

Click the **Delete Project** button (requires confirmation). Members become standalone agents.

### PROJECT.md

When a project is selected, a PROJECT.md banner appears above the agent grid. Edit it to set shared context for all project members. The content is pushed to running member agents on save. When no project is selected (All Agents view), the banner is hidden.

## Fleet Tab

Overview of all registered agents showing health status, activity state (idle/thinking/tool), daily cost, token usage, and restart count. Click any agent card to drill down into its detail view with cost breakdowns, budget bars, workspace file editor, and recent events. Also includes agent configuration management -- view and edit each agent's model, role, system prompt, and daily budget. Changes that require a restart (model) are flagged.

All agents show an embedded KasmVNC viewer in their detail view, providing a live view of the agent's browser session.

## Activity Tab

Sub-views toggled via a tab bar at the top of the panel:

**Traces** (default) — Grouped request traces showing the full lifecycle of each request through the system. Each trace row shows the triggering event type, participating agents, a prompt preview (extracted from the first LLM call's user message), event count, total duration, error badge, and relative time. Click a trace to expand an inline event timeline with a vertical waterfall: each event shows its type, source, agent, duration, detail text, error message (if any), and metadata fields (model, tokens, prompt/response previews). Traces auto-refresh every 10 seconds and on relevant WebSocket events (debounced). LLM call events include prompt and response preview fields extracted from the mesh API proxy.

**Live Events** — Real-time event feed streamed via WebSocket. Events include LLM calls, tool executions, text streaming deltas, messages sent/received, blackboard writes, agent state changes, and health changes. Filter by event type using the chip toggles. Events are capped at 500 in the browser; older events are dropped. Click any event row to expand an inline detail panel showing all available data fields with type-specific formatting (e.g. model, token breakdown, cost, and prompt/response previews for `llm_call`; full untruncated arguments for `tool_start`; complete message text for `message_sent`). The same expandable rows appear in the agent detail view's Recent Events section. Note: per-token `text_delta` events are delivered via the direct streaming chat endpoint (not the WebSocket event bus) to avoid flooding the event buffer.

**Blackboard** — Browse, search, write, and delete shared state entries. Entries display as expandable card rows with agent avatars, color-coded namespace badges (tasks, context, signals, goals, artifacts, history), a value summary, and relative timestamps. Namespace filter buttons show per-namespace entry counts. Click any row to expand an inline detail panel with full JSON, version number, author, and exact timestamp. Filter by key prefix (e.g., `tasks/`, `context/`, `signals/`) or by writing agent. New entries are highlighted with a flash animation when written by agents in real-time. History namespace entries (`history/*`) cannot be deleted.

**Costs** — Per-agent LLM spend with period selector (today/week/month). Bar chart shows cost and token usage side-by-side. Budget status bars show daily spend vs. configured limits. Cost data refreshes automatically when `llm_call` events arrive.

**Automation** — Manage scheduled jobs. View schedule, last run time, run count, error count, and heartbeat status. Actions: Run (trigger immediately), Pause, Resume, Edit schedule, Delete. Auto-refreshes every 10 seconds while the tab is active.

## System Tab

Environment overview showing configured credentials with tier labels (system or agent, names only, never values), pub/sub subscriptions, and model pricing tables. Add new credentials from a dropdown of LLM providers, known agent tools (Brave Search, Apollo, Hunter), or custom service names.

## Agent Management

### Edit Agent Config

1. Switch to the **Fleet** tab
2. Click **Edit** on an agent card
3. Modify model, role, daily budget, or credential access patterns
4. Click **Save** -- a toast confirms which fields were updated
5. If the change requires a restart (model), click **Restart**

Credential access uses comma-separated glob patterns (e.g. `*`, `brave_search_*`, `myapp_*`). An empty field revokes all vault access. System credentials (LLM provider API keys) are always blocked regardless of patterns.

### Restart Agent

Click the **Restart** button on any agent card. A confirmation dialog prevents accidental restarts. The agent is stopped and restarted with its current configuration. The agents panel updates automatically when the agent comes back online.

### Update Budget

From the agent detail view (click an agent in Agents), budget bars show current daily and monthly usage. Budget can also be updated via the Fleet tab edit form.

## Agent Identity Panel

The agent detail view features a tabbed **Agent Identity** panel for viewing and editing an agent's identity, instructions, and behavioral context. The panel appears above the spend and budget sidebar since identity is the primary concern when drilling into an agent.

### Tabs

| Tab | Contents | Description |
|-----|----------|-------------|
| **Identity** | `SOUL.md` (4K cap), `INSTRUCTIONS.md` (8K cap) | Personality, tone, operating procedures, domain knowledge |
| **Memory** | `MEMORY.md` (16K cap), `USER.md` (4K cap), `HEARTBEAT.md` (no cap) | Long-term facts, user preferences, autonomous heartbeat rules |
| **Config** | Model, role, budget, credential access | Agent configuration (model changes trigger restart) |
| **Logs** | Activity + Learnings (read-only) | Daily session logs and recorded errors/corrections |
| **Tools** | Capabilities list (read-only) | Available tools and skill definitions |

Each file card shows an access badge: **Shared** (teal) for files both you and the agent can edit (`SOUL.md`, `INSTRUCTIONS.md`, `USER.md`, `HEARTBEAT.md`), **Auto** (gray) for system-managed files (`MEMORY.md`). Customized files show a description subtitle; default files show a CTA prompt.

### Usage

1. Click an agent card to open the detail view
2. The **Agent Identity** panel shows 5 tabs — Identity is selected by default
3. File cards with default/scaffold content show a "default" pill and a "Customize" button with a friendly prompt
4. Once customized, a description subtitle and character budget bar appear (indigo → amber at 80% → red at 95%)
5. Click **Edit** (or **Customize** for default files) to open the inline editor
6. Edit the content in the monospace textarea — the budget bar updates live. Save with **Ctrl+S** / **⌘S** or the Save button
7. Click **Save** to write changes (content is sanitized for invisible Unicode)
8. The **Logs** tab shows daily activity logs and learnings (errors in red, corrections in amber), both read-only
9. The **Config** tab shows model, role, budget, credential access, and a **Remove Agent** action at the bottom

The dashboard proxies workspace operations through the mesh transport layer to the agent's container — files are read from and written to the agent's `/data/workspace` volume.

## Chat

### Slide-Over Chat Panel

Click the chat button on an agent card to open a slide-over panel on the right side of the dashboard. Messages stream in real-time with token-level updates via SSE (`POST /dashboard/api/agents/{id}/chat/stream`). The response renders progressively as tokens arrive, with a pulsing cursor indicating active streaming.

The slide-over panel can be minimized to a pill at the bottom of the screen and restored by clicking it.

### Tool Call Display

When an agent calls tools during a response, each tool appears as an inline pill inside the message bubble:
- **Running** — spinning indicator with tool name
- **Done** — green checkmark with truncated output preview (200 chars max)

Tool calls appear above the text response, in the order they were executed.

### Chat History

Conversation history persists per agent across panel open/close — reopening the same agent shows the full conversation. Use the **Clear** button in the chat header to reset history for that agent. History is stored in browser memory and resets on page reload.

### Keyboard Shortcuts

- **Enter** — send message
- **Escape** — close chat panel
- **Cmd+K** / **Ctrl+K** — open command palette

### Notifications

When an agent calls `notify_user()`, the notification appears inline in the agent's chat panel with amber styling and a "NOTIFICATION" label, in addition to the existing toast and activity feed entry. Behavior adapts to panel state:

- **Panel open and visible** — notification appends to the conversation and auto-scrolls
- **Panel minimized** — an amber unread badge appears on the minimized header showing the count; expanding the panel clears the badge
- **Panel not open** — a new panel opens automatically if a slot is available; if all slots are occupied, the toast and history entry are sufficient

Notifications never steal keyboard focus or force-unminimize a panel the user deliberately collapsed.

### Abort

Closing the chat panel while a response is streaming cancels the in-flight request. The partial response is preserved in history.

## Broadcast

Send a message to multiple agents simultaneously using the broadcast bar below the agent grid. When a project is selected, the broadcast targets only that project's members. When viewing "All Agents", it targets every agent. Standalone agents (not in any project) are included only in the "All Agents" broadcast. Each agent processes the message independently. Responses display inline with expand/collapse for long replies (200+ characters).

## Blackboard Operations

### Write Entry

1. Click **+ New** to open the write form
2. Enter a key (e.g., `context/my_data`) and JSON value
3. Click **Save**

### Delete Entry

Click **Del** on any entry row. History namespace entries are protected and cannot be deleted.

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
| `GET` | `/dashboard/api/agents/{id}/status` | Agent status from container |
| `GET` | `/dashboard/api/agents/{id}/capabilities` | Agent capabilities and tools |
| `POST` | `/dashboard/api/agents/{id}/chat` | Non-streaming chat (request/response) |
| `POST` | `/dashboard/api/agents/{id}/chat/stream` | SSE streaming chat (token-level) |
| `POST` | `/dashboard/api/agents/{id}/steer` | Update agent system prompt live |
| `POST` | `/dashboard/api/agents/{id}/reset` | Reset agent conversation history |
| `POST` | `/dashboard/api/agents/{id}/restart` | Restart an agent |
| `PUT` | `/dashboard/api/agents/{id}/budget` | Update agent budget |
| `GET` | `/dashboard/api/agents/{id}/permissions` | Agent credential and API permissions |
| `PUT` | `/dashboard/api/agents/{id}/permissions` | Update credential access patterns |
| `GET` | `/dashboard/api/agents/{id}/workspace` | List agent workspace files (with cap, is_default) |
| `GET` | `/dashboard/api/agents/{id}/workspace/{file}` | Read workspace file content |
| `PUT` | `/dashboard/api/agents/{id}/workspace/{file}` | Write workspace file content |
| `GET` | `/dashboard/api/agents/{id}/workspace-logs?days=N` | Read daily logs (read-only, default 3 days) |
| `GET` | `/dashboard/api/agents/{id}/workspace-learnings` | Read errors and corrections (read-only) |
| `GET` | `/dashboard/api/blackboard` | List blackboard entries |
| `PUT` | `/dashboard/api/blackboard/{key}` | Write blackboard entry |
| `DELETE` | `/dashboard/api/blackboard/{key}` | Delete blackboard entry |
| `POST` | `/dashboard/api/credentials` | Add a credential to the vault |
| `DELETE` | `/dashboard/api/credentials/{name}` | Remove a credential |
| `GET` | `/dashboard/api/costs/{agent_id}` | Cost data for a specific agent |
| `GET` | `/dashboard/api/costs` | Cost data with optional period |
| `GET` | `/dashboard/api/projects` | List all projects with members |
| `POST` | `/dashboard/api/projects` | Create a new project |
| `DELETE` | `/dashboard/api/projects/{name}` | Delete a project |
| `POST` | `/dashboard/api/projects/{name}/members` | Add agent to project (auto-restarts agent) |
| `DELETE` | `/dashboard/api/projects/{name}/members/{agent}` | Remove agent from project (auto-restarts agent) |
| `GET` | `/dashboard/api/project?project={name}` | Read project's PROJECT.md (requires project param) |
| `PUT` | `/dashboard/api/project?project={name}` | Update project's PROJECT.md (requires project param) |
| `GET` | `/dashboard/api/traces` | Recent trace events |
| `GET` | `/dashboard/api/traces/{id}` | Trace detail |
| `GET` | `/dashboard/api/queues` | Queue status per agent |
| `GET` | `/dashboard/api/cron` | List cron jobs |
| `POST` | `/dashboard/api/cron` | Create a cron job |
| `POST` | `/dashboard/api/cron/{id}/run` | Trigger cron job |
| `PUT` | `/dashboard/api/cron/{id}` | Update cron job schedule |
| `POST` | `/dashboard/api/cron/{id}/pause` | Pause cron job |
| `POST` | `/dashboard/api/cron/{id}/resume` | Resume cron job |
| `DELETE` | `/dashboard/api/cron/{id}` | Delete cron job |
| `GET` | `/dashboard/api/settings` | Environment settings |
| `POST` | `/dashboard/api/credentials/validate` | Validate a credential (check if set) |
| `GET` | `/dashboard/api/model-health` | Model health and failover status |
| `POST` | `/dashboard/api/channels/{type}/connect` | Connect a messaging channel |
| `POST` | `/dashboard/api/channels/{type}/disconnect` | Disconnect a messaging channel |
| `POST` | `/dashboard/api/broadcast` | Send message to all agents |
| `POST` | `/dashboard/api/broadcast/stream` | SSE streaming broadcast to all agents |
| `GET` | `/dashboard/api/messages` | Recent message log |
| `GET` | `/dashboard/api/workflows` | Workflow definitions |
| `POST` | `/dashboard/api/workflows/{name}/run` | Trigger a workflow by name |
| `GET` | `/dashboard/api/webhooks` | List configured webhooks |
| `POST` | `/dashboard/api/webhooks` | Create a webhook endpoint |
| `DELETE` | `/dashboard/api/webhooks/{name}` | Delete a webhook |
| `POST` | `/dashboard/api/webhooks/{name}/test` | Send test payload to webhook |
| `GET` | `/dashboard/api/logs` | Runtime logs (query: lines, level) |
| `WS` | `/ws/events` | Real-time event stream |

## Accessibility

The dashboard includes several accessibility features:

- **ARIA roles** — Tab containers use `role="tablist"` / `role="tab"` with `aria-selected`. The chat modal uses `role="dialog"` with `aria-modal` and `aria-label`.
- **Keyboard navigation** — Escape closes the chat modal. Focus management within modal dialogs.
- **Reduced motion** — A `prefers-reduced-motion` media query disables animations and transitions for users who prefer reduced motion.
- **Color contrast** — Stat labels, action buttons, and queue status indicators use colors that meet accessibility contrast guidelines.
- **Mobile responsive** — Navigation shows icons-only on narrow screens (< 640px).

## Source Files

| File | Role |
|------|------|
| `src/dashboard/server.py` | FastAPI router with all API endpoints |
| `src/dashboard/templates/index.html` | Dashboard HTML (Alpine.js + Tailwind) |
| `src/dashboard/static/js/app.js` | Dashboard application logic |
| `src/dashboard/static/js/websocket.js` | WebSocket client with reconnect |
| `src/dashboard/static/css/dashboard.css` | Custom styles |
| `src/dashboard/events.py` | EventBus for real-time event distribution |
