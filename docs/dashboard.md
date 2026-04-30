# Dashboard

Real-time web dashboard for fleet observability and management.

## Overview

The dashboard is served at `http://localhost:8420/dashboard` (or whatever port the mesh is configured on). It provides a live view of your agent fleet across three main tabs with a consolidated navigation bar, slide-over chat panels, and a keyboard command palette.

No additional setup is required -- the dashboard starts automatically with `openlegion start`. In self-hosted and local dev mode, the dashboard is open to anyone who can reach port 8420. In hosted mode (subdomain deployments), SSO authentication is required; see [Authentication](#authentication) for details.

## Navigation

The dashboard uses a consolidated three-tab layout:

| Tab | What It Shows |
|-----|--------------|
| **Fleet** | Agent cards (operator card prepended in standalone view), agent detail views, configuration editing |
| **Activity** | Traces, live events, blackboard, costs, and automation |
| **System** | 11 sub-tabs: Activity / Costs / Automation / Integrations / API Keys / Wallet / Network / Storage / Operator / Browser / Settings — see [System Tab](#system-tab) |

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

When the shared browser service is running, an embedded KasmVNC viewer appears in each agent's detail view, providing a live view of the browser session. If the browser service has not started or is unavailable, the VNC viewer is not shown.

### Operator Agent Rendering

The operator is a system agent that builds and manages your workforce. It is rendered differently from regular agents:

- In the **standalone fleet view** (no project selected), the operator card is prepended to the grid as the first card with a `system` badge.
- Inside a project view, the operator card is **not** rendered — operator is never a project member. The backend rejects `POST /api/projects` and `POST /api/projects/{name}/members` requests that include the operator (`HTTP 400`).
- Clicking the operator card routes to **System → Operator** (not the standard agent detail panel).
- The operator is **excluded from quota math, fleet cost/token totals, and broadcasts** — only "real" agents count against `OPENLEGION_MAX_AGENTS` and receive broadcast messages.
- The standard agent detail panel for operator (if reached via deep link) shows a banner directing the user to the Operator system sub-tab; **Heartbeat Pause** is hidden.

## Activity Tab

Sub-views toggled via a tab bar at the top of the panel:

**Traces** (default) — Grouped request traces showing the full lifecycle of each request through the system. Each trace row shows the triggering event type, participating agents, a prompt preview (extracted from the first LLM call's user message), event count, total duration, error badge, and relative time. Click a trace to expand an inline event timeline with a vertical waterfall: each event shows its type, source, agent, duration, detail text, error message (if any), and metadata fields (model, tokens, prompt/response previews). Traces auto-refresh every 10 seconds and on relevant WebSocket events (debounced). LLM call events include prompt and response preview fields extracted from the mesh API proxy.

**Live Events** — Real-time event feed streamed via WebSocket. Events include LLM calls, tool executions, text streaming deltas, messages sent/received, blackboard writes, agent state changes, and health changes. Filter by event type using the chip toggles. Events are capped at 500 in the browser; older events are dropped. Click any event row to expand an inline detail panel showing all available data fields with type-specific formatting (e.g. model, token breakdown, cost, and prompt/response previews for `llm_call`; full untruncated arguments for `tool_start`; complete message text for `message_sent`). The same expandable rows appear in the agent detail view's Recent Events section. Note: per-token `text_delta` events are delivered via the direct streaming chat endpoint (not the WebSocket event bus) to avoid flooding the event buffer.

**Blackboard** — Browse, search, write, and delete shared state entries. Entries display as expandable card rows with agent avatars, color-coded namespace badges (tasks, context, signals, goals, artifacts, history), a value summary, and relative timestamps. Namespace filter buttons show per-namespace entry counts. Click any row to expand an inline detail panel with full JSON, version number, author, and exact timestamp. Filter by key prefix (e.g., `tasks/`, `context/`, `signals/`) or by writing agent. New entries are highlighted with a flash animation when written by agents in real-time. History namespace entries (`history/*`) cannot be deleted.

**Costs** — Per-agent LLM spend with period selector (today/week/month). Bar chart shows cost and token usage side-by-side. Budget status bars show daily spend vs. configured limits. Cost data refreshes automatically when `llm_call` events arrive.

**Automation** — Manage scheduled jobs. View schedule, last run time, run count, error count, and heartbeat status. Actions: Run (trigger immediately), Pause, Resume, Edit schedule, Delete. Auto-refreshes every 10 seconds while the tab is active.

## System Tab

The System tab is split into 11 sub-tabs along the top:

| Sub-tab | What It Shows |
|---------|--------------|
| **Activity** | Trace stream + live event log (same backing as Activity tab, scoped here for system-level audit) |
| **Costs** | Per-agent LLM spend with period selector and budget bars |
| **Automation** | Cron jobs and webhook endpoints |
| **Integrations** | Configured credentials with tier labels (system or agent, names only, never values), pub/sub subscriptions, and model pricing. Add credentials from a dropdown of LLM providers, known agent tools (Brave Search, Apollo, Hunter), or custom service names |
| **API Keys** | Named external API keys (`/api/external-api-keys`) for inbound integrations |
| **Wallet** | Wallet seed init, addresses (Ethereum + Solana), RPC endpoints, per-agent wallet enablement |
| **Network** | Fleet-wide and per-agent proxy configuration (`GET/PUT /api/network/proxy`, `PUT /api/agents/{id}/proxy`). See [Proxy Configuration](#proxy-configuration) |
| **Storage** | Agent SQLite databases with purge buttons. Each row shows DB id, size, oldest timestamp, and a `purgeable` flag |
| **Operator** | Operator agent settings (model picker, heartbeat editor) and the operator audit log ("Change Log"), backed by `/api/operator-audit`. See [Operator Sub-tab](#operator-sub-tab) |
| **Browser** | Live browser metrics fleet table, interaction speed and delay sliders, idle-timeout, and CAPTCHA solver provider/key. See [Browser Sub-tab](#browser-sub-tab) |
| **Settings** | Default model, runtime logs viewer, and miscellaneous toggles |

### Operator Sub-tab

Operator-only system-agent control panel. Shows a status card with health indicator, a model picker (searchable dropdown over the same `availableModels` list used in the agent edit form), and a heartbeat-schedule editor (number + unit, e.g. `15m`, `1h`). Heartbeat **Pause** is intentionally hidden — the operator is not subject to operator-level pause controls.

Below the status card, a **Change Log** table renders the operator audit feed (`GET /api/operator-audit?per_page=20&page=N`). Each entry shows actor, action, target, and timestamp. The list paginates client-side via `auditPage`.

### Browser Sub-tab

| Section | Backing endpoint(s) | Notes |
|---------|--------------------|-------|
| Live Browser Health (fleet table) | WS `browser_metrics` events | Per-agent rows: click rate (last 100), clicks/min, snapshot p50/p95, nav timeouts, last-update age. Stale rows fade after the 30-min eviction window |
| Interaction Speed slider | `/api/browser-settings` | Speed multiplier with presets (Off / Light / Moderate / Heavy / Maximum) |
| Delay Between Actions slider | `/api/browser-settings` | Random pause after each browser action (0–10s) |
| Idle Timeout | `/api/system-settings` | Container idle timeout in minutes (5–120, default 30); restart required |
| CAPTCHA Solver | `/api/captcha-solver` | Provider dropdown (none / 2captcha / capsolver) + API key field. Stored masked; restart required |

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

## Agent Settings Panel

The agent detail view features a tabbed **Agent Settings** panel for viewing and editing an agent's identity, instructions, and behavioral context. The panel appears above the spend and budget sidebar since identity is the primary concern when drilling into an agent.

### Tabs

| Tab | Contents | Description |
|-----|----------|-------------|
| **Config** | Model, role, budget, credential access | Agent configuration (model changes trigger restart). Includes a **Remove Agent** action at the bottom |
| **Identity** | `SOUL.md` (4K cap), `INSTRUCTIONS.md` (12K cap) | Personality, tone, operating procedures, domain knowledge |
| **Memory** | `MEMORY.md` (16K cap), `USER.md` (4K cap), `HEARTBEAT.md` (no cap) | Long-term facts, user preferences, autonomous heartbeat rules |
| **Activity** | Per-agent activity stream from `/api/agents/{id}/activity` | Recent agent events scoped to this agent |
| **Logs** | Daily logs + Learnings (read-only) | Daily session logs and recorded errors/corrections |
| **Capabilities** | Tools list (read-only) | Available tools and skill definitions from `/api/agents/{id}/capabilities` |
| **Files** | `/data` listing for the agent | Browse files written by the agent into its persistent volume; proxied via `/api/agents/{id}/files` |

Each file card shows an access badge: **Shared** (teal) for files both you and the agent can edit (`SOUL.md`, `INSTRUCTIONS.md`, `USER.md`, `HEARTBEAT.md`), **Auto** (gray) for system-managed files (`MEMORY.md`). Customized files show a description subtitle; default files show a CTA prompt.

### Usage

1. Click an agent card to open the detail view
2. The **Agent Settings** panel shows 7 tabs — Config is selected by default
3. File cards with default/scaffold content show a "default" pill and a "Customize" button with a friendly prompt
4. Once customized, a description subtitle and character budget bar appear (indigo → amber at 80% → red at 95%)
5. Click **Edit** (or **Customize** for default files) to open the inline editor
6. Edit the content in the monospace textarea — the budget bar updates live. Save with **Ctrl+S** / **⌘S** or the Save button
7. Click **Save** to write changes (content is sanitized for invisible Unicode)
8. The **Logs** tab shows daily activity logs and learnings (errors in red, corrections in amber), both read-only
9. The **Config** tab shows model, role, budget, credential access, and a **Remove Agent** action at the bottom

The dashboard proxies workspace operations through the mesh transport layer to the agent's container — files are read from and written to the agent's `/data/workspace` volume.

## Cookie / Session Import (Operator-Only)

An inline collapsible card on the agent detail panel lets operators paste cookie or storage-state payloads into an agent's Firefox profile, useful for transferring an authenticated session captured manually. The card is **operator-only** and **hidden on the operator agent itself** (operator does not run a browser session).

- Two input modes: **Playwright** (storage-state JSON) or **Netscape** (TSV cookie jar)
- POSTs to `/api/agents/{id}/browser/import_cookies` with rate limit 10/hour per (operator, agent)
- A fleet kill switch (`OPENLEGION_DISABLE_COOKIE_IMPORT=1`) disables the endpoint
- Card state lives in Alpine `x-data` only — it is **never persisted to localStorage** (cookie text is high-trust). Inputs are cleared on success
- Imported cookies are stored UNENCRYPTED in `cookies.sqlite` inside the agent profile; the card surfaces an inline warning banner

## Chat

### Slide-Over Chat Panel

Click the chat button on an agent card to open a slide-over panel on the right side of the dashboard. Messages stream in real-time with token-level updates via SSE (`POST /dashboard/api/agents/{id}/chat/stream`). The response renders progressively as tokens arrive, with a pulsing cursor indicating active streaming.

The slide-over panel can be minimized to a pill at the bottom of the screen and restored by clicking it.

### Tool Call Display

When an agent calls tools during a response, each tool appears as an inline pill inside the message bubble:
- **Running** — spinning indicator with tool name
- **Done** — green checkmark with truncated output preview (200 chars max)

Tool calls appear above the text response, in the order they were executed.

### Browser Login Handoff Card

When an agent calls `request_browser_login`, a violet-accented handoff card appears in chat with the service name, login URL, and **Open Browser** / **Mark complete** / **Cancel** buttons. The card is rendered both in the requesting agent's chat and in the operator chat; state (completed/cancelled) syncs across both surfaces via the `browser_login_completed` / `browser_login_cancelled` WS events.

### CAPTCHA Help Handoff Card

When an agent calls `request_captcha_help`, an **amber-accented** handoff card mirrors the browser-login card structure with **Open Browser** / **Mark complete** / **Cancel** buttons. The card is rendered in both the requesting agent's chat and the operator chat. State syncs across both surfaces via the `browser_captcha_help_completed` / `browser_captcha_help_cancelled` WS events — clicking "Mark complete" in either chat marks the corresponding card complete in the other.

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

### Credit Exhaustion

When an agent's LLM call fails due to depleted credits (HTTP 402), a styled card appears in the chat:
- Shows "Credits Depleted" with the error details
- "Top Up Credits" button links to the app's credits page
- "Use Own API Key" button navigates to Settings → API Keys
- Prevents duplicate cards via deduplication on the last 3 messages

## Proxy Configuration

Agent proxy settings are in the agent config panel under **Network**:
- **System proxy** (inherit) — uses the fleet-wide proxy
- **Custom proxy** — per-agent HTTP/HTTPS proxy URL with optional credentials
- **No proxy** (direct) — bypasses all proxies

When a custom proxy is configured, a green indicator shows the current proxy host. Click "Change" to modify. Only HTTP and HTTPS proxies are supported — SOCKS5 is not available.

Proxy changes auto-restart the agent and reset the browser session. In the System → Network page, per-agent proxy edits also trigger an automatic restart.

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

The dashboard connects to the mesh via WebSocket at `/ws/events`. Events are streamed in real-time. The connection indicator in the top-right shows live/disconnected status, and a countdown timer (`wsReconnectIn`) is shown during reconnect. On disconnect, the WebSocket client automatically reconnects with exponential backoff.

### Event Types

`DashboardEvent.type` is a `Literal[...]` of 26 values (`src/shared/types.py`):

```
agent_state, message_sent, message_received, tool_start, tool_result, text_delta,
llm_call, blackboard_write, health_change, notification, workspace_updated,
heartbeat_complete, cron_change, chat_user_message, chat_done, chat_reset,
credit_exhausted, credential_request, credential_stored,
browser_login_request, browser_login_completed, browser_login_cancelled,
browser_captcha_help_request, browser_captcha_help_completed, browser_captcha_help_cancelled,
browser_metrics, browser_nav_probe
```

The **WebSocket subscription is unfiltered on the wire** — the SPA receives every event and applies type/agent filters in JavaScript. The Live Events filter chips operate over a subset (`agent_state`, `message_*`, `tool_*`, `text_delta`, `llm_call`, `blackboard_write`, `health_change`, `notification`, `workspace_updated`, `heartbeat_complete`, `cron_change`, `credit_exhausted`, `credential_request`, `browser_login_*`, `browser_captcha_help_*`); `browser_metrics` and `browser_nav_probe` are not filter-chip-exposed but still arrive on the same socket.

### `tenant_spend_threshold` Discriminator

Per-tenant CAPTCHA-spend threshold alerts (50/80/100% of monthly cap) are **not** a separate WS event type. They ride as `browser_metrics` events with `data.type === "tenant_spend_threshold"` as the discriminator. Subscribers expecting a `tenant_spend_threshold` literal will silently miss every alert — match on `evt.type === "browser_metrics" && evt.data?.type === "tenant_spend_threshold"`.

## Authentication

### Dev vs Hosted Mode

The dashboard operates in two modes:

- **Dev / self-hosted mode** — when `/opt/openlegion/.access_token` does not exist (the default for local installs), all requests are allowed. No cookie or SSO is required.
- **Hosted mode** — when `/opt/openlegion/.subdomain` exists (subdomain deployments via the OpenLegion cloud), the `ol_session` cookie is required on every request. Missing cookie returns **HTTP 401** ("Authentication required"); expired or invalid-signature cookie returns **HTTP 403** ("session expired" / "invalid signature").

### Session Cookie (`ol_session`)

In hosted mode, the Caddy reverse proxy runs a `forward_auth` gate at `/__auth/callback`. The SSO flow is:

1. The app generates an HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. The user is redirected to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. The auth gate verifies the HMAC, sets the `ol_session` cookie (one-time-use — replay attacks are blocked), and redirects to the dashboard
4. Caddy `forward_auth` verifies the cookie on every subsequent request

The cookie is valid for up to 24 hours (enforced by the engine, independent of the auth gate's issued expiry). The cookie key is derived from the access token on disk via HMAC-SHA256.

### CSRF Protection

All state-changing endpoints (POST, PUT, DELETE, PATCH) require the `X-Requested-With` header. Browsers block this custom header on cross-origin requests (CORS preflight), preventing CSRF attacks on cookie-authenticated sessions. GET, HEAD, and OPTIONS are exempt.

```bash
# Example: include X-Requested-With on state-changing dashboard API calls.
# Cron POST takes `agent` (not `agent_id`).
curl -X POST http://localhost:8420/dashboard/api/cron \
  -H "X-Requested-With: XMLHttpRequest" \
  -H "Content-Type: application/json" \
  -d '{"agent": "researcher", "schedule": "every 1h", "message": "Check leads"}'
```

### VNC Proxy

The VNC reverse proxy at `/vnc/` rejects agent Bearer tokens. Only `ol_session` cookie authentication (dashboard auth) is accepted for VNC access, preventing agents from directly reading the shared browser screen.

## CAPTCHA Rollup CSV

The `/api/billing/captcha-rollup` endpoint exports per-tenant CAPTCHA-solver spend as CSV. It has **no UI** — operators reach it directly via curl. Required query params: `tenant` (project slug) and `period` (`daily` | `weekly` | `monthly`, default `monthly`).

Column schema (one header row, then per-agent rows sorted by agent_id, then a synthetic tenant-total row):

| Column | Description |
|--------|-------------|
| `period_start` | ISO-8601 UTC timestamp marking the start of the requested period |
| `agent_id` | Agent that incurred the cost (or `__tenant_total__` for the synthetic last row) |
| `millicents` | Spend in millicents (1 millicent = 1/100,000 USD) |
| `dollars` | Spend in dollars (5 decimal places) |
| `data_scope` | `monthly_actual` for `period=monthly`; `current_month_aggregate` for `period=daily` or `period=weekly` |

Why two `data_scope` values: in-memory CAPTCHA-cost state is **current-month-only**. For `daily` and `weekly` requests, the export still surfaces month-to-date per-agent buckets (with `data_scope=current_month_aggregate` flagging that the data is not period-correct) rather than returning an empty CSV. Older periods would require persisted snapshots, deferred per the §11.10 SQLite trim plan.

The final synthetic `__tenant_total__` row sums every agent in the tenant for the requested period — same `period_start` and `data_scope` as the per-agent rows.

Example:

```bash
curl -X GET "http://localhost:8420/dashboard/api/billing/captcha-rollup?tenant=acme&period=monthly" \
  -H "X-Requested-With: XMLHttpRequest"
```

## API Endpoints

All dashboard API endpoints are prefixed with `/dashboard/api/`. The SPA root is served at `GET /dashboard/` (HTML, not an API endpoint). The dashboard exposes ~120 endpoints; tables below cover the ones grouped by feature. State-changing endpoints (POST/PUT/DELETE/PATCH) require the `X-Requested-With` header; see [Authentication](#authentication).

A few endpoints have **no UI surface** and are reachable by curl only. They are flagged inline as `(operator-curl only — no UI)`.

**Agents**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/agents` | Agent overview with health and costs |
| `POST` | `/dashboard/api/agents` | Create a new agent |
| `GET` | `/dashboard/api/agents/{id}` | Agent detail with spend and budget |
| `DELETE` | `/dashboard/api/agents/{id}` | Remove an agent |
| `GET` | `/dashboard/api/agents/{id}/config` | Agent configuration |
| `PUT` | `/dashboard/api/agents/{id}/config` | Update agent configuration |
| `GET` | `/dashboard/api/agents/{id}/status` | Agent status from container |
| `GET` | `/dashboard/api/agents/{id}/capabilities` | Agent capabilities and tools |
| `POST` | `/dashboard/api/agents/{id}/restart` | Restart an agent |
| `PUT` | `/dashboard/api/agents/{id}/budget` | Update agent budget |
| `GET` | `/dashboard/api/agents/{id}/permissions` | Agent credential and API permissions |
| `PUT` | `/dashboard/api/agents/{id}/permissions` | Update credential access patterns |
| `GET` | `/dashboard/api/agents/{id}/activity` | Agent activity events |
| `POST` | `/dashboard/api/restart-agents` | Restart all agent containers and the browser service. Pushes `OPENLEGION_CAPTCHA_SOLVER_PROVIDER`/`_KEY` from settings into the browser container env, and re-pushes per-agent proxy config |

**Chat**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/agents/{id}/chat` | Non-streaming chat (request/response) |
| `POST` | `/dashboard/api/agents/{id}/chat/stream` | SSE streaming chat (token-level) |
| `GET` | `/dashboard/api/agents/{id}/chat/history` | Retrieve conversation history for agent |
| `POST` | `/dashboard/api/agents/{id}/steer` | Update agent system prompt live |
| `POST` | `/dashboard/api/agents/{id}/reset` | Reset agent conversation history |
| `POST` | `/dashboard/api/broadcast` | Send message to all agents (request body filters by `project` name or `standalone: true`; operator excluded) |
| `POST` | `/dashboard/api/broadcast/stream` | SSE streaming broadcast (same filters as above) |

**Workspace**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/agents/{id}/workspace` | List agent workspace files (with cap, is_default) |
| `GET` | `/dashboard/api/agents/{id}/workspace/{file}` | Read workspace file content |
| `PUT` | `/dashboard/api/agents/{id}/workspace/{file}` | Write workspace file content |
| `GET` | `/dashboard/api/agents/{id}/workspace-logs?days=N` | Read daily logs (read-only, default 3 days) |
| `GET` | `/dashboard/api/agents/{id}/workspace-learnings` | Read errors and corrections (read-only) |

**Artifacts & Uploads**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/agents/{id}/artifacts` | List agent artifacts |
| `GET` | `/dashboard/api/agents/{id}/artifacts/{name}` | Download an artifact |
| `DELETE` | `/dashboard/api/agents/{id}/artifacts/{name}` | Delete an artifact |
| `GET` | `/dashboard/api/agents/{id}/files` | List files in agent data volume |
| `GET` | `/dashboard/api/agents/{id}/files/{path}` | Read a file from agent data volume |
| `GET` | `/dashboard/api/uploads` | List uploads |
| `POST` | `/dashboard/api/uploads/{name}` | Upload a file |
| `GET` | `/dashboard/api/uploads/{name}/download` | Download an uploaded file |
| `DELETE` | `/dashboard/api/uploads/{name}` | Delete an uploaded file |

**Blackboard**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/blackboard` | List blackboard entries |
| `PUT` | `/dashboard/api/blackboard/{key}` | Write blackboard entry |
| `DELETE` | `/dashboard/api/blackboard/{key}` | Delete blackboard entry |

**Credentials**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/credentials` | Add a credential to the vault |
| `DELETE` | `/dashboard/api/credentials/{name}` | Remove a credential |
| `POST` | `/dashboard/api/credentials/validate` | Validate a credential (check if set) |
| `POST` | `/dashboard/api/credentials/agent` | Add an agent-tier credential |
| `GET` | `/dashboard/api/credentials/{name}/value` | Retrieve masked credential value |
| `POST` | `/dashboard/api/credentials/upload-env` | Bulk-import credentials from an .env file |

**External API Keys**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/external-api-keys` | List named external API keys |
| `POST` | `/dashboard/api/external-api-keys` | Add an external API key |
| `DELETE` | `/dashboard/api/external-api-keys/{key_id}` | Remove an external API key |

**Costs & Budgets**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/costs/{agent_id}` | Cost data for a specific agent |
| `GET` | `/dashboard/api/costs` | Cost data with optional period |

**Projects**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/projects` | List all projects with members |
| `POST` | `/dashboard/api/projects` | Create a new project |
| `DELETE` | `/dashboard/api/projects/{name}` | Delete a project |
| `POST` | `/dashboard/api/projects/{name}/members` | Add agent to project (auto-restarts agent) |
| `DELETE` | `/dashboard/api/projects/{name}/members/{agent}` | Remove agent from project (auto-restarts agent) |
| `GET` | `/dashboard/api/project?project={name}` | Read project's PROJECT.md |
| `PUT` | `/dashboard/api/project?project={name}` | Update project's PROJECT.md |

**Automation (Cron)**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/cron` | List cron jobs |
| `POST` | `/dashboard/api/cron` | Create a cron job |
| `POST` | `/dashboard/api/cron/{id}/run` | Trigger cron job immediately |
| `PUT` | `/dashboard/api/cron/{id}` | Update cron job fields |
| `POST` | `/dashboard/api/cron/{id}/pause` | Pause cron job |
| `POST` | `/dashboard/api/cron/{id}/resume` | Resume cron job |
| `DELETE` | `/dashboard/api/cron/{id}` | Delete cron job |

**Webhooks**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/webhooks` | List configured webhooks |
| `POST` | `/dashboard/api/webhooks` | Create a webhook endpoint |
| `DELETE` | `/dashboard/api/webhooks/{hook_id}` | Delete a webhook |
| `PATCH` | `/dashboard/api/webhooks/{hook_id}` | Update webhook configuration (name, agent, instructions, signature) |
| `POST` | `/dashboard/api/webhooks/{hook_id}/test` | Send test payload to webhook |

**Channels**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/channels` | List connected messaging channels |
| `POST` | `/dashboard/api/channels/{type}/connect` | Connect a messaging channel |
| `POST` | `/dashboard/api/channels/{type}/disconnect` | Disconnect a messaging channel |
| `GET` | `/dashboard/api/comms/activity` | Messaging channel activity log |
| `GET` | `/dashboard/api/messages` | Recent message log |

**Network / Proxy**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/network/proxy` | Get fleet-wide proxy configuration |
| `PUT` | `/dashboard/api/network/proxy` | Set fleet-wide proxy configuration |
| `PUT` | `/dashboard/api/agents/{id}/proxy` | Set per-agent proxy override |

**Wallet**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/wallet/init` | Initialize wallet (generates seed; shown once) |
| `GET` | `/dashboard/api/wallet/seed` | Retrieve wallet seed (HTTP 410 after first reveal) |
| `GET` | `/dashboard/api/wallet/addresses` | List wallet addresses (Ethereum + Solana) |
| `GET` | `/dashboard/api/wallet/rpc` | Get configured RPC endpoints |
| `PUT` | `/dashboard/api/wallet/rpc` | Update RPC endpoints |
| `POST` | `/dashboard/api/wallet/enable/{agent_id}` | Enable wallet access for an agent |

**Storage**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/storage` | Storage overview |
| `GET` | `/dashboard/api/storage/databases` | List agent SQLite databases |
| `POST` | `/dashboard/api/storage/databases/{db_id}/purge` | Purge a database |

**Audit**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/traces` | Recent trace events |
| `GET` | `/dashboard/api/traces/{id}` | Trace detail |
| `GET` | `/dashboard/api/audit` | Agent audit log |
| `GET` | `/dashboard/api/operator-audit` | Operator-level audit log |
| `GET` | `/dashboard/api/queues` | Queue status per agent |

**System & Settings**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/settings` | Bundled environment payload — `credentials`, `has_llm_credentials`, `available_provider_models` (Ollama + OpenLegion gateway discovery), `model_costs`, `plan_limits`, `app_url`, `pubsub_subscriptions`, `has_byok_keys`, `credit_proxy_configured`, `default_model` |
| `GET` | `/dashboard/api/system-settings` | System-level settings |
| `POST` | `/dashboard/api/system-settings` | Update system-level settings |
| `GET` | `/dashboard/api/browser-settings` | Browser service settings |
| `POST` | `/dashboard/api/browser-settings` | Update browser service settings |
| `POST` | `/dashboard/api/default-model` | Set the default LLM model in mesh.yaml |
| `GET` | `/dashboard/api/model-health` | Model health and failover status — per-model success/failure counts, cooldown status, and active model in the failover chain |
| `GET` | `/dashboard/api/agent-templates` | Available agent fleet templates |
| `GET` | `/dashboard/api/fleet/templates` | Fleet template list (alternate endpoint) |
| `GET` | `/dashboard/api/logs` | Runtime logs (query: lines, level) |

**Browser — Control & Handoffs**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/browser/{agent_id}/focus` | Focus browser window for agent (raises X11 WID) |
| `POST` | `/dashboard/api/browser/{agent_id}/control` | Toggle operator control of the browser window — pauses agent X11 input |
| `POST` | `/dashboard/api/browser/{agent_id}/reset` | Reset browser session for agent (close and relaunch with current config) |
| `POST` | `/dashboard/api/browser-login/complete` | Complete a browser login flow |
| `POST` | `/dashboard/api/browser-login/cancel` | Cancel a browser login flow |
| `POST` | `/dashboard/api/browser-captcha-help/complete` | Complete a CAPTCHA help handoff |
| `POST` | `/dashboard/api/browser-captcha-help/cancel` | Cancel a CAPTCHA help handoff |
| `POST` | `/dashboard/api/agents/{agent_id}/browser/import_cookies` | Operator-only cookie/session import (Playwright JSON or Netscape TSV); 10/hr rate limit per (operator, agent) |

**Browser — Settings & Metrics**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/browser-settings` | Browser service settings (speed multiplier, delay) |
| `POST` | `/dashboard/api/browser-settings` | Update browser service settings |
| `GET` | `/dashboard/api/agents/{agent_id}/browser/metrics` | Per-agent browser-metrics history snapshot |
| `GET` | `/dashboard/api/captcha-solver` | Get configured CAPTCHA solver provider + masked key |
| `POST` | `/dashboard/api/captcha-solver` | Set CAPTCHA solver provider + key (restart required) |
| `DELETE` | `/dashboard/api/captcha-solver` | Remove CAPTCHA solver configuration |

**Browser — Operator-Only (no UI)**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/agents/{agent_id}/session` | Inspect persisted storage state for agent (operator-curl only — no UI) |
| `DELETE` | `/dashboard/api/agents/{agent_id}/session` | Clear persisted storage state for agent (operator-curl only — no UI) |
| `GET` | `/dashboard/api/agents/{agent_id}/fingerprint-health` | Per-agent fingerprint rejection-window summary + burn flag (operator-curl only — no UI) |
| `POST` | `/dashboard/api/agents/{agent_id}/fingerprint-health/reset` | Clear fingerprint burn flag after profile rotation (operator-curl only — no UI) |
| `GET` | `/dashboard/api/billing/captcha-rollup?tenant=…&period=daily\|weekly\|monthly` | CSV export of per-tenant CAPTCHA-solver spend (operator-curl only — no UI). See [CAPTCHA Rollup CSV](#captcha-rollup-csv) for column schema |

> Mobile emulation profiles are env-only — set `BROWSER_DEVICE_PROFILE` to a profile name (e.g. `iphone_15`, `pixel_8`); there is no dashboard surface and no per-agent override endpoint.

**WebSocket**

| Method | Path | Description |
|--------|------|-------------|
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
