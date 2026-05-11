# Dashboard

Real-time web dashboard for fleet observability and management.

## Overview

The dashboard is served at `http://localhost:8420/dashboard` (or whatever port the mesh is configured on). It provides a live view of your agent fleet across four top-nav tabs with a consolidated navigation bar, slide-over chat panels, and a keyboard command palette.

No additional setup is required — the dashboard starts automatically with `openlegion start`. In self-hosted and local dev mode, the dashboard is open to anyone who can reach port 8420. In hosted mode (subdomain deployments), SSO authentication is required; see [Authentication](#authentication) for details.

## Navigation

### Top-Nav Tabs

The dashboard has **four** top-level tabs. Tab IDs are frozen for URL stability (deep links, persisted preferences, dashboard endpoint paths reference them) while the user-facing labels have diverged — per CLAUDE.md #14, renaming the IDs would break links so only the labels were updated.

| Tab ID (frozen) | User Label | What It Shows |
|----------------|------------|--------------|
| `chat` | **Chat** | Multi-agent chat surface — slide-over panels, streaming, command palette entry. Default tab on boot |
| `workplace` | **Work** | Kanban / Needs-You / Activity feed / Pending Actions — the Board surface. See [Work Tab](#work-tab) |
| `fleet` | **Team** | Agent grid with operator card prepended in standalone view, drill-down agent detail (config, identity, capabilities, files). See [Team Tab](#team-tab) |
| `system` | **Settings** | 11 sub-tabs: Activity / Costs / Automation / Integrations / API Keys / Wallet / Network / Storage / Operator / Browser / Settings. See [System Tab](#system-tab) |

The defaults — `tabs` array and `activeTab: 'chat'` — live in `src/dashboard/static/js/app.js` (top of the `dashboard()` factory). Routes parse the URL hash into `{ tab, systemTab, agentId, identityTab, homeTab, activityView }`.

A command palette (**Cmd+K** / **Ctrl+K**) provides quick access to agents, actions, and navigation. The search button in the nav bar also opens it.

### Top-Nav Bell + "Needs Attention" Badge

- **Notifications bell** (top-right) — driven by `/api/notifications`. Subtle gray dot when unread items exist; click to open the dropdown. See [Notifications](#notifications-system).
- **"N agents need attention" badge** — wired to `fleetDigest?.agents_attention?.length`, populated from the operator's `OBSERVATIONS.md` aggregation. Renders only when the count is non-zero.

## Project Management

The dashboard supports multi-project namespaces for organizing agents into isolated groups.

### Project Switcher

A tab bar at the top of the Team view shows all projects plus an "All Agents" view. Click a project tab to filter the agent grid to that project's members. The "All Agents" tab shows every agent with project badges. Each tab displays a member count.

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

## Team Tab

Overview of all registered agents showing health status, activity state (idle/thinking/tool), daily cost, token usage, and restart count. Click any agent card to drill down into its detail view with cost breakdowns, budget bars, workspace file editor, and recent events. Also includes agent configuration management — view and edit each agent's model, role, system prompt, and daily budget. Changes that require a restart (model) are flagged. Cards render activity via the `agentActivityLabel(agent)` JS helper which fuses `last_active` with `last_task_event_ts`.

When the shared browser service is running, an embedded KasmVNC viewer appears in each agent's detail view, providing a live view of the browser session. If the browser service has not started or is unavailable, the VNC viewer is not shown.

### Operator Agent Rendering

The operator is a system agent that builds and manages your workforce. It is rendered differently from regular agents:

- In the **standalone fleet view** (no project selected), the operator card is **prepended** to the grid as the first card with a `system` badge.
- Inside a project view, the operator card is **not** rendered — operator is never a project member. The backend rejects `POST /api/projects` and `POST /api/projects/{name}/members` requests that include the operator (`ValueError → HTTP 400`).
- Clicking the operator card routes to **Settings → Operator** (not the standard agent detail panel).
- The operator is **excluded from quota math, fleet cost/token totals, and broadcasts** — only "real" agents count against `OPENLEGION_MAX_AGENTS` and receive broadcast messages.
- The standard agent detail panel for operator (if reached via deep link) shows a banner directing the user to the Operator system sub-tab; **Heartbeat Pause** is hidden.

## Work Tab

The Work tab is the Board surface — Phase 2 Board UX overhaul. The route id is `workplace` (label "Work"); endpoints, JS state vars, and URL paths all keep the legacy name `workplace`.

### Sub-Routes (`homeTab`)

Two sub-routes, switched by the `homeTab` Alpine var (default `kanban`):

| `homeTab` | Surface |
|-----------|---------|
| `kanban` (default) | Kanban board — Needs You + Stuck tasks + 4-column kanban (Pending / Working / Blocked / Done) with × cancel on every card. Empty sections hide. URL: `/home` |
| `activity` | Single-scroll activity feed — Just delivered + Happening now + In progress, with pinned blockers and "Recently delivered" inline artifact preview. URL: `/home/activity` |

Legacy `homeTab` values `main` and `tasks` (Phase 3) both resolve to `kanban` for back-compat. A separate `workplaceTab` (`feed` / `project-status` / `task-board` / `team-outputs`) survives for deep-link compat but the new structure routes through `homeTab`.

### Surfaces

- **Sticky "Needs You" panel** — aggregates pending actions, credential requests, browser-login handoffs, CAPTCHA handoffs, and blockers across the fleet. Pinned to the top of the kanban view.
- **Activity feed** — pinned blockers + "Recently delivered" with inline artifact preview, "Read full" toggle, and "Copy" button. The `recentlyDeliveredItems` array is memoised in `loadWorkplaceOutputs` so Alpine doesn't recompute on every reactive read.
- **Per-task artifact preview cache** — keyed by `task_id`, full `/api/workplace/tasks/{id}` response cached so "Read full" doesn't re-fetch.
- **Onboarding intent chips** — prepend (not overwrite) drafts so users don't lose typed text when they click a suggestion.
- **Activity translation toggle** (`showTechDetail`) — when false (default), implementation events (`blackboard_write`, `llm_call`, `message_received`) are hidden and engineer event types are run through `formatActivityForUser` for plain-English summaries. Persisted to localStorage.

### Per-Section Skeleton Loaders + Retry Banners

Each Board section (`projects`, `tasks`, `blockers`, `outputs`, `pending`, `feed`) has its own loading + error bucket:

- `workplaceSectionLoading.<section>: bool` — drives skeleton placeholders during in-flight fetch.
- `workplaceErrors.<section>: string` — drives the "Couldn't load — Retry" banner. Click clears the error and re-runs the load.

Previously, failures were swallowed (`console.error`) and left the user staring at an empty panel. The retry pattern keeps recovery one click away.

## System Tab

The System tab is split into **11 sub-tabs** along the top. Default sub-tab: `activity`. State var: `systemTab` in `app.js`.

| Sub-tab | What It Shows |
|---------|--------------|
| **Activity** (default) | Trace stream + live event log + Blackboard browser. Routes to `/activity/events` and `/activity/logs` deep links via `activityView` |
| **Costs** | Per-agent LLM spend with period selector (today/week/month) and budget bars |
| **Automation** | Cron jobs + webhook endpoints. View schedule, last run time, run count, error count. Actions: Run / Pause / Resume / Edit / Delete |
| **Integrations** | Configured credentials with tier labels (system or agent, names only — never values), pub/sub subscriptions, model pricing |
| **API Keys** | Named external API keys (`/api/external-api-keys`) for inbound integrations |
| **Wallet** | Wallet seed init, addresses (Ethereum + Solana), RPC endpoints, per-agent wallet enablement |
| **Network** | Fleet-wide and per-agent proxy configuration. See [Proxy Configuration](#proxy-configuration) |
| **Storage** | Agent SQLite databases with purge buttons. Each row shows DB id, size, oldest timestamp, `purgeable` flag |
| **Operator** | Operator agent settings (model picker, heartbeat editor) and the operator audit log ("Change Log"). See [Operator Sub-tab](#operator-sub-tab) |
| **Browser** | Live browser metrics fleet table, interaction speed/delay sliders, idle timeout, CAPTCHA solver provider/key. See [Browser Sub-tab](#browser-sub-tab) |
| **Settings** | Default model, runtime logs viewer, miscellaneous toggles |

A **Fleet pulse card** sits at the top of the System tab, rendering the operator's `OBSERVATIONS.md` aggregation.

### Activity Sub-tab (System → Activity)

Three sub-views toggled via the `activityView` state var:

**Traces** (default) — Grouped request traces showing the full lifecycle of each request through the system. Each trace row shows the triggering event type, participating agents, a prompt preview, event count, total duration, error badge, and relative time. Click to expand an inline event timeline with a vertical waterfall. Traces auto-refresh every 10 seconds and on relevant WebSocket events (debounced).

**Live Events** — Real-time event feed streamed via WebSocket. Events include LLM calls, tool executions, text streaming deltas, messages sent/received, blackboard writes/deletes, agent state changes, and health changes. Filter by event type using chip toggles. Events are capped at 500 in the browser. Click any row to expand a type-specific detail panel. Note: per-token `text_delta` events are delivered via the direct streaming chat endpoint (not the WebSocket event bus) to avoid flooding the buffer.

**Blackboard** — Browse, search, write, and delete shared state entries. Entries display as expandable card rows with agent avatars, color-coded namespace badges (tasks / context / signals / goals / artifacts / history), value summary, and relative timestamps. Filter by key prefix or by writing agent. New entries are highlighted with a flash animation. History namespace entries (`history/*`) cannot be deleted.

### Operator Sub-tab

Operator-only system-agent control panel. Status card with health indicator, a model picker (searchable dropdown over `availableModels`), and a heartbeat-schedule editor (number + unit, e.g. `15m`, `1h`). Heartbeat **Pause** is intentionally hidden — the operator is not subject to operator-level pause controls.

Below the status card, a **Change Log** table renders the operator audit feed (`GET /api/operator-audit?per_page=20&page=N&include_archived=…`). Each entry shows actor, action, target, and timestamp. The list paginates client-side via `auditPage`.

- **`include_archived` toggle** — defaults off. When enabled, archived rows surface alongside active rows. Backed by the `audit_log.archived` column.
- **"Archive entries older than"** control — 7 / 30 / 90 days (default 30). Calls `POST /api/operator-audit/archive` with `{ "before_date": "<ISO 8601>" }`, which the dashboard proxies to `POST /mesh/audit/archive` over loopback so the operator-or-internal permission tier (not just the dashboard cookie) gates the write and the audit-of-audit row is recorded by the mesh. Soft-archive flips rows to `archived=1`.
- Query optimisation: the mesh-side `audit_log` table carries a composite index `idx_audit_log_active(archived, id DESC)` so the default filter is cheap.

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

1. Switch to the **Team** tab
2. Click **Edit** on an agent card
3. Modify model, role, daily budget, or credential access patterns
4. Click **Save** — a toast confirms which fields were updated
5. If the change requires a restart (model), click **Restart**

Credential access uses comma-separated glob patterns (e.g. `*`, `brave_search_*`, `myapp_*`). An empty field revokes all vault access. System credentials (LLM provider API keys) are always blocked regardless of patterns.

### Restart Agent

Click the **Restart** button on any agent card. A confirmation dialog prevents accidental restarts. The agent is stopped and restarted with its current configuration. The fleet panel updates automatically when the agent comes back online — the SPA listens for `agent_restarting` (renders a pulsing "Restarting" indicator) and clears it on `agent_restarted` or an `agent_state` event with `restart_failed`.

### Update Budget

From the agent detail view (click an agent in Team), budget bars show current daily and monthly usage. Budget can also be updated via the Team tab edit form.

## Agent Settings Panel

The agent detail view features a tabbed **Agent Settings** panel for viewing and editing an agent's identity, instructions, and behavioral context. The panel appears above the spend and budget sidebar since identity is the primary concern when drilling into an agent.

### Tabs

The panel has **7 tabs** defined in `_IDENTITY_TABS` (`app.js`):

| Tab ID | Label | Contents | Description |
|--------|-------|----------|-------------|
| `config` | **Config** | Model, role, budget, credential access | Agent configuration (model changes trigger restart). Includes a **Remove Agent** action at the bottom |
| `identity` | **Identity** | `SOUL.md` (4K cap), `INSTRUCTIONS.md` (12K cap), `INTERFACE.md` (4K cap) | Personality, tone, operating procedures, public collaboration contract |
| `memory` | **Memory** | `MEMORY.md` (16K cap), `USER.md` (4K cap), `HEARTBEAT.md` (no cap) | Long-term facts, user preferences, autonomous heartbeat rules |
| `activity` | **Activity** | Per-agent activity stream from `/api/agents/{id}/activity` | Recent agent events scoped to this agent |
| `logs` | **Logs** | Daily logs + Learnings (read-only) | Daily session logs and recorded errors/corrections |
| `capabilities` | **Tools** | Tools list (read-only) | Available tools and skill definitions from `/api/agents/{id}/capabilities` |
| `files` | **Files** | `/data` listing for the agent | Browse files written by the agent; proxied via `/api/agents/{id}/files` |

Each file card shows an access badge: **Shared** (teal) for files both you and the agent can edit (`SOUL.md`, `INSTRUCTIONS.md`, `USER.md`, `HEARTBEAT.md`, `INTERFACE.md`), **Auto** (gray) for system-managed files (`MEMORY.md`). Customized files show a description subtitle; default files show a CTA prompt.

### Usage

1. Click an agent card to open the detail view
2. The **Agent Settings** panel shows 7 tabs — Config is selected by default
3. File cards with default/scaffold content show a "default" pill and a "Customize" button
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
- Card state lives in Alpine `x-data` only — it is **never persisted to localStorage** (cookie text is high-trust; persistence would leave session material in browser storage longer than the in-page lifetime). Inputs are cleared on success
- Imported cookies are stored UNENCRYPTED in `cookies.sqlite` inside the agent profile; the card surfaces an inline warning banner

## Chat

### Slide-Over Chat Panel

Click the chat button on an agent card to open a slide-over panel on the right side of the dashboard. Messages stream in real-time with token-level updates via SSE (`POST /dashboard/api/agents/{id}/chat/stream`). The response renders progressively as tokens arrive, with a pulsing cursor indicating active streaming.

The slide-over panel can be minimized to a pill at the bottom of the screen and restored by clicking it.

### Tool Call Display

When an agent calls tools during a response, each tool appears as an inline pill inside the message bubble — a spinning indicator with the tool name while running, swapped for a green checkmark plus a truncated 200-char output preview on completion. Tool calls appear above the text response, in execution order.

### Browser Login Handoff Card

When an agent calls `request_browser_login`, a violet-accented handoff card appears in chat with the service name, login URL, and **Open Browser** / **Mark complete** / **Cancel** buttons. The card is rendered both in the requesting agent's chat and in the operator chat; state syncs across both surfaces via the `browser_login_completed` / `browser_login_cancelled` WS events.

### CAPTCHA Help Handoff Card

When an agent calls `request_captcha_help`, an **amber-accented** handoff card mirrors the browser-login card structure with **Open Browser** / **Mark complete** / **Cancel** buttons. The card is rendered in both the requesting agent's chat and the operator chat. State syncs across both surfaces via the `browser_captcha_help_completed` / `browser_captcha_help_cancelled` WS events.

### Chat History

Conversation history persists per agent across panel open/close — reopening the same agent shows the full conversation. Use the **Clear** button in the chat header to reset history for that agent. History is stored in browser memory and resets on page reload. "Load older →" pagination uses a `_chatVisibleLimit` per-agent counter (default 50, +50 per click) to avoid rendering thousands of historical messages at once.

### Conversations API

Persistent conversation state across reloads is backed by three endpoints (used by the Chat tab to remember which agents had open panels):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/conversations` | List active conversations |
| `POST` | `/api/conversations/{agent_id}/open` | Mark a conversation open |
| `POST` | `/api/conversations/{agent_id}/close` | Mark a conversation closed |

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

Closing the chat panel while a response is streaming cancels the in-flight SSE request (the streaming generator on the server is bound to the request's disconnect signal, so closing the panel closes the SSE channel and aborts mid-stream). The partial response is preserved in history.

### Credit Exhaustion

When an agent's LLM call fails due to depleted credits (HTTP 402), a styled card appears in the chat:
- Shows "Credits Depleted" with the error details
- "Top Up Credits" button links to the app's credits page
- "Use Own API Key" button navigates to Settings → API Keys
- Deduplicates against the last 3 messages so rapid-fire failures don't stack identical cards

## Notifications System

Phase 2 Board UX overhaul added a persistent notifications store (separate from transient toasts and the Needs-You badge). A notification represents a past event the user should know about; the bell badge survives page reloads and cross-device viewing.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/notifications` | Top 10 notifications, unread first then by `ts DESC` |
| `POST` | `/dashboard/api/notifications/{notification_id}/read` | Mark a single notification read |
| `POST` | `/dashboard/api/notifications/read-all` | Mark every unread notification read |

### Storage

SQLite-backed via `src/dashboard/notifications.py` (`NotificationStore`). Schema:

```
dashboard_notifications(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT,                 -- optional originating agent (NULL = system)
    ts           REAL NOT NULL,        -- Unix epoch seconds
    kind         TEXT NOT NULL,        -- short tag from _KNOWN_KINDS
    title        TEXT NOT NULL,        -- one-line headline
    body         TEXT,                 -- optional longer body
    read_at      REAL,                 -- Unix epoch when read (NULL = unread)
    payload_json TEXT                  -- optional JSON for click-through targets
)
```

WAL mode, `busy_timeout=30000`. Composite index `idx_notifications_unread_ts(read_at, ts DESC)` optimises the unread-first read pattern.

### Frozen Kinds

`_KNOWN_KINDS = {"delivered", "approval", "alert", "info", "blocker", "credential"}`. Unknown kinds are **accepted** (the log warns) so a producer typo never drops an event — the bell falls back to a generic icon. Add a value to `_KNOWN_KINDS` rather than coining ad-hoc kinds; the bell renders an icon per kind.

### Live Updates

The mesh emits `notification_added` immediately after each `NotificationStore.add` call (via `_notifications_producer`) so the bell badge updates without waiting for the 60-second poll cycle. Subscribers listen for the WS event and refetch `/api/notifications` for the top page.

## Telemetry System

Frontend telemetry (wizard step transitions, tour activation analysis) is persisted to a process-local SQLite table for hypothesis testing — does onboarding move first-visit activation?

### Endpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/telemetry` | Record one telemetry event |

### Storage

SQLite-backed via `src/dashboard/telemetry.py` (`DashboardTelemetry`). Schema:

```
dashboard_telemetry(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    session_id  TEXT NOT NULL,
    event_name  TEXT NOT NULL,
    props_json  TEXT NOT NULL DEFAULT '{}'
)
```

WAL mode. Indexes on `(event_name, ts DESC)` and `(session_id, ts DESC)`. Operators read the table directly via `sqlite3 data/telemetry.db 'select ...'` — there is no BI export yet.

### Caps & Limits

- **Retention cap** — `_MAX_EVENTS = 100_000` rows. `_maybe_trim` runs on every insert and trims the oldest excess.
- **Per-event-name length cap** — 64 chars.
- **Per-props size cap** — 4096 bytes of JSON-serialised props.
- **Per-session rate limit** — `RATE_LIMIT_EVENTS_PER_MIN = 60`. Sliding 60s window; over-budget requests get `429` with a `retry_after_ms` hint.
- **Bucket sweep** — when more than 1024 active session buckets exist, stale buckets (no events in the window) are reaped on the next admit call.

## Proxy Configuration

Agent proxy settings are in the agent config panel under **Network**:
- **System proxy** (inherit) — uses the fleet-wide proxy
- **Custom proxy** — per-agent HTTP/HTTPS proxy URL with optional credentials
- **No proxy** (direct) — bypasses all proxies

When a custom proxy is configured, a green indicator shows the current proxy host. Click "Change" to modify. Only HTTP and HTTPS proxies are supported — SOCKS5 is not available.

Proxy changes auto-restart the agent and reset the browser session. In the System → Network page, per-agent proxy edits also trigger an automatic restart.

## Broadcast

Send a message to multiple agents simultaneously using the broadcast bar below the agent grid. When a project is selected, the broadcast targets only that project's members. When viewing "All Agents", it targets every agent. Standalone agents (not in any project) are included only in the "All Agents" broadcast. The operator is always excluded from broadcasts. Each agent processes the message independently. Responses display inline with expand/collapse for long replies (200+ characters).

## Blackboard Operations

### Write Entry

1. Click **+ New** to open the write form
2. Enter a key (e.g., `context/my_data`) and JSON value
3. Click **Save**

### Delete Entry

Click **Del** on any entry row. History namespace entries are protected and cannot be deleted. Deletes emit `blackboard_delete` events so other connected dashboards reflect the removal live (mirror of `blackboard_write`).

## Onboarding — Tutorial & Wizard

Two distinct onboarding surfaces run in sequence for first-visit users. They are mutually exclusive: the wizard mounts only after the tutorial closes (or was already seen).

### New-User Tutorial (replaces "What's new" tour)

The legacy "What's new" tour was repurposed into a new-user tutorial. State var `newUserTutorial.step`:

- `0` — closed (default)
- `1..3` — modal step visible

Gating in `_maybeStartTutorial`:
1. `newUserTutorial.step === 0`
2. `wizard.step === 'idle'`
3. `localStorage.olSeenTutorial !== 'true'`
4. First-visit detector returns true (empty fleet excluding operator, no real user message in operator history)

The flag persists to `localStorage.olSeenTutorial = 'true'` on completion or any exit (skip, dismiss, reaching step 3, Escape). Migration: users who saw the legacy "What's new" tour (with `olSeenWhatsNew === 'true'`) are not new users; their `olSeenTutorial` flag is set automatically on first load to suppress the tutorial.

### Empty-Fleet Wizard

State machine `wizard.step`:

```
idle → ask → confirming → building → first-output
                      ↘ build_failed (terminal sad-state, renders dot index 3)
```

Wizard state shape: `wizard: { step, plan, startedAt, lastChip }`. Persisted to `localStorage.ol_wizard`; unknown values on read reset to `idle`. The wizard mounts only when `step !== 'idle'` and only inside the Chat tab. Non-empty fleets force `step = 'idle'`. An `abandon` telemetry event fires on page unload if a wizard run is in flight (`step !== 'idle'` and `step !== 'first-output'`).

## Real-Time Updates

The dashboard connects to the mesh via WebSocket at `/ws/events`. Events are streamed in real-time. The connection indicator in the top-right shows live/disconnected status, and a countdown timer (Alpine `wsReconnectIn`, mirrored from the JS `DashboardWebSocket.reconnectIn`) is shown during reconnect. On disconnect, the client reconnects with exponential backoff (`websocket.js`).

### Event Types

`DashboardEvent.type` is a `Literal[…]` of **50 values** (`src/shared/types.py:552-643`). They group into nine families:

| Family | Events |
|--------|--------|
| **Agent runtime** | `agent_state`, `tool_start`, `tool_result`, `text_delta`, `llm_call`, `health_change`, `workspace_updated`, `heartbeat_complete` |
| **Chat** | `chat_user_message`, `chat_done`, `chat_reset`, `message_sent`, `message_received`, `credit_exhausted` |
| **Notifications** | `notification`, `notification_added` |
| **Blackboard** | `blackboard_write`, `blackboard_delete` |
| **Automation** | `cron_change` |
| **Credentials / handoffs** | `credential_request`, `credential_request_cancelled`, `credential_stored`, `browser_login_request`, `browser_login_completed`, `browser_login_cancelled`, `browser_captcha_help_request`, `browser_captcha_help_completed`, `browser_captcha_help_cancelled` |
| **Browser metrics** | `browser_metrics`, `browser_nav_probe` |
| **Tasks** | `task_created`, `task_status_changed`, `task_outcome`, `task_artifact_added` |
| **Pending actions** | `pending_action_created`, `pending_action_resolved`, `pending_action_expired` |
| **Operator action receipts** | `operator_action_receipt`, `operator_action_receipt_undone`, `operator_action_receipt_superseded` |
| **Agent lifecycle** | `agent_archived`, `agent_unarchived`, `agent_restarting`, `agent_restarted`, `agent_config_updated` |
| **Project CRUD** | `project_created`, `project_updated`, `project_deleted`, `project_archived`, `project_unarchived` |

Adding a new event-name string anywhere in `src/` without adding the matching `Literal` value will cause `DashboardEvent` to raise `ValidationError`, which the emit-site `try/except` swallows at debug level — the event silently disappears. `tests/test_types.py::test_every_emit_string_in_src_matches_a_dashboard_event_literal` is the regex-sweep guard against that drift.

The **WebSocket subscription is unfiltered on the wire** — the server pushes every event and the SPA filters in JavaScript. The Live Events filter chips operate over a subset (the agent-runtime / chat / notifications / blackboard / credentials families); other families (browser metrics, lifecycle, project CRUD) still arrive on the same socket and are routed to their respective panels.

### `tenant_spend_threshold` Discriminator

Per-tenant CAPTCHA-spend threshold alerts (50/80/100% of monthly cap) are **not** a separate WS event type. They ride as `browser_metrics` events with `data.type === "tenant_spend_threshold"` as the discriminator. Subscribers expecting a `tenant_spend_threshold` literal will silently miss every alert — match on `evt.type === "browser_metrics" && evt.data?.type === "tenant_spend_threshold"`.

### EventBus Implementation

`src/dashboard/events.py:EventBus`. Key properties:

- **Ring buffer** — `deque(maxlen=BUFFER_SIZE)` (500 events) for replay-on-reconnect via `recent_events(before_seq=…)`. Each event carries a monotonic `_seq` counter so subscribers can avoid double-delivery.
- **Thread-safe emit** — `threading.Lock` protects `_seq` increment, buffer append, and listener snapshot. Cross-thread emission uses `call_soon_threadsafe(asyncio.ensure_future, coro)` to hop onto the bound loop.
- **In-process listeners** — `add_listener(cb)` / `remove_listener(cb)` for module-level aggregators (e.g. per-platform success rollup) that observe events without going through a WebSocket. Listeners run synchronously on the emit caller's stack; exceptions are caught and logged at debug level so a buggy aggregator can't break broadcast.
- **Dead-client reaping** — `_broadcast` removes subscribers whose `send_text` raises.

## Authentication

### Dev vs Hosted Mode

Mode detection lives in `src/dashboard/auth.py` and keys off **two** files on disk:

- **Dev / self-hosted mode** — when `/opt/openlegion/.access_token` does not exist (the default for local installs), all requests are allowed. No cookie or SSO is required.
- **Hosted mode** — when `/opt/openlegion/.subdomain` exists (subdomain deployments via the OpenLegion cloud), authentication is required. With both `/opt/openlegion/.subdomain` and `/opt/openlegion/.access_token` present, the `ol_session` cookie is verified on every request.
- **Hosted-but-unconfigured** — `/opt/openlegion/.subdomain` present but `/opt/openlegion/.access_token` missing returns an error ("Dashboard authentication required (access token not configured)") rather than open access.

Missing cookie returns **HTTP 401** ("Authentication required"); expired or invalid-signature cookie returns **HTTP 403** ("session expired" / "invalid signature").

### Session Cookie (`ol_session`)

The cookie is **issued by the Caddy sidecar's auth gate, not by engine code.** The engine only verifies. The flow:

1. The app generates an HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. The user is redirected to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. The **auth gate** (deployed via cloud-init, runs in front of Caddy) verifies the HMAC, **sets the `ol_session` cookie** (one-time-use — replay is blocked), and redirects to the dashboard
4. Caddy's `forward_auth` re-checks the cookie on every subsequent request; the dashboard's `verify_session_cookie` is the engine-side defense-in-depth check

The cookie key is derived from the access token on disk via `HMAC-SHA256(token, "ol-cookie-signing")`. Maximum cookie lifetime — even if the auth gate issues a longer-lived cookie — is `COOKIE_MAX_AGE = 24 * 3600` (24 hours), enforced engine-side with a **5-minute clock skew tolerance** on the upper bound. Cookies with `expiry` more than `COOKIE_MAX_AGE + 300` seconds in the future are rejected.

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

### Content Security Policy

The dashboard HTML is served with a strict CSP:

```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net;
style-src 'self' 'unsafe-inline';
connect-src 'self';
frame-src 'self';
object-src 'none'
```

`unsafe-inline` + `unsafe-eval` on `script-src` is required for Alpine.js + Tailwind via CDN (no build step). The two CDN hosts are explicitly allowed. `connect-src 'self'` blocks exfiltration via `fetch`/`XHR` to third-party hosts. `object-src 'none'` kills `<object>`/`<embed>` plugin loads. Templates render with Jinja2 `autoescape=True` to defend against the small surface where the SPA injects server-rendered values.

### VNC Proxy

The per-agent VNC reverse proxy lives at **`/agent-vnc/{agent_id}/{path}`** on the **mesh host** (`src/host/server.py`) — `GET` for HTTP, `WebSocket` for the noVNC websockify channel. It rejects agent Bearer tokens; only `ol_session` cookie authentication (dashboard auth) is accepted, preventing agents from directly reading the shared browser screen. The browser service routes each agent's display to a dedicated KasmVNC port (6100..6163) and the proxy resolves to the per-agent port via the display allocator.

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

In-memory CAPTCHA-cost state is **current-month-only**. For `daily` and `weekly` requests, the export still surfaces month-to-date per-agent buckets (with `data_scope=current_month_aggregate` flagging that the data is not period-correct) rather than returning an empty CSV. Older periods would require persisted snapshots, deferred per the §11.10 SQLite trim plan.

The final synthetic `__tenant_total__` row sums every agent in the tenant for the requested period — same `period_start` and `data_scope` as the per-agent rows.

Example:

```bash
curl -X GET "http://localhost:8420/dashboard/api/billing/captcha-rollup?tenant=acme&period=monthly" \
  -H "X-Requested-With: XMLHttpRequest" \
  -H "Cookie: ol_session=<value>"   # required in hosted mode
```

## API Endpoints

All dashboard API endpoints are prefixed with `/dashboard/api/`. The SPA root is served at `GET /dashboard/` (HTML, not an API endpoint). The dashboard router exposes **143 endpoints** registered via `@api_router.*` in `src/dashboard/server.py`. State-changing endpoints (POST/PUT/DELETE/PATCH) require the `X-Requested-With` header; see [Authentication](#authentication).

A handful of endpoints have **no UI surface** and are reachable by curl only. They are flagged inline as `(operator-curl only — no UI)`.

The tables below are not exhaustive — they cover the user-facing surface area grouped by prefix. The authoritative list lives in `src/dashboard/server.py` (search `@api_router.`).

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

**Chat & Conversations**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/agents/{id}/chat` | Non-streaming chat (request/response) |
| `POST` | `/dashboard/api/agents/{id}/chat/stream` | SSE streaming chat (token-level) |
| `GET` | `/dashboard/api/agents/{id}/chat/history` | Retrieve conversation history for agent |
| `POST` | `/dashboard/api/agents/{id}/steer` | Update agent system prompt live |
| `POST` | `/dashboard/api/agents/{id}/reset` | Reset agent conversation history |
| `GET` | `/dashboard/api/conversations` | List active conversations (open chat panels) |
| `POST` | `/dashboard/api/conversations/{agent_id}/open` | Mark a conversation open |
| `POST` | `/dashboard/api/conversations/{agent_id}/close` | Mark a conversation closed |
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

**Work (Board) Surface — `/api/workplace/*`**

12 endpoints power the Work tab. Together they cover the Kanban, Needs-You, Activity feed, and pending-action review surfaces. State-changing routes require the CSRF header.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/workplace/projects` | Projects with rollup status for the Project Status panel |
| `GET` | `/dashboard/api/workplace/tasks` | All tasks (kanban columns: Pending / Working / Blocked / Done) |
| `GET` | `/dashboard/api/workplace/tasks/{task_id}` | Single task detail with artifacts (powers "Read full" toggle) |
| `GET` | `/dashboard/api/workplace/tasks/{task_id}/events` | Per-task event timeline |
| `POST` | `/dashboard/api/workplace/tasks/{task_id}/cancel` | Cancel a task (× button on cards) |
| `POST` | `/dashboard/api/workplace/tasks/{task_id}/outcome` | Set the outcome rating on a delivered task |
| `GET` | `/dashboard/api/workplace/blockers` | Active blockers across the fleet |
| `GET` | `/dashboard/api/workplace/outputs` | "Recently delivered" outputs with artifact previews |
| `GET` | `/dashboard/api/workplace/feed` | Combined activity feed (capped at `workplaceFeedCap=200`) |
| `GET` | `/dashboard/api/workplace/pending` | Pending operator review queue |
| `POST` | `/dashboard/api/workplace/pending/{nonce}/cancel` | Cancel a pending action |
| `POST` | `/dashboard/api/changes/undo/{undo_token}` | Undo a soft-edit change inside its TTL window |

**Notifications**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/notifications` | Top 10 notifications, unread first then `ts DESC` |
| `POST` | `/dashboard/api/notifications/{notification_id}/read` | Mark a single notification read |
| `POST` | `/dashboard/api/notifications/read-all` | Mark every unread notification read |

**Telemetry**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/telemetry` | Record a frontend telemetry event |

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
| `DELETE` | `/dashboard/api/blackboard/{key}` | Delete blackboard entry (emits `blackboard_delete`) |

**Credentials**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/api/credentials` | Add a credential to the vault |
| `DELETE` | `/dashboard/api/credentials/{name}` | Remove a credential |
| `POST` | `/dashboard/api/credentials/validate` | Validate a credential (check if set) |
| `POST` | `/dashboard/api/credentials/agent` | Add an agent-tier credential |
| `GET` | `/dashboard/api/credentials/{name}/value` | Retrieve masked credential value |
| `POST` | `/dashboard/api/credentials/upload-env` | Bulk-import credentials from an .env file |
| `POST` | `/dashboard/api/credential-request/{request_id}/cancel` | Cancel a single credential-request handoff by ID (distinct from the legacy `complete`/`cancel` pair) |

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
| `GET` | `/dashboard/api/operator-audit` | Operator-level audit log (paginate via `page`/`per_page`; toggle archived rows via `include_archived`) |
| `POST` | `/dashboard/api/operator-audit/archive` | Soft-archive operator audit entries older than `before_date`; proxies to mesh `POST /mesh/audit/archive` over loopback so the operator-or-internal tier gates the write |
| `GET` | `/dashboard/api/queues` | Queue status per agent |

**Dashboard Aggregation**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/dashboard/platform-success` | Per-tenant success scoring snapshot — backed by `platform_success.py` |

**System & Settings**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/api/settings` | Bundled environment payload — `credentials`, `has_llm_credentials`, `available_provider_models` (Ollama + OpenLegion gateway discovery), `model_costs`, `plan_limits`, `app_url`, `pubsub_subscriptions`, `has_byok_keys`, `credit_proxy_configured`, `default_model` |
| `GET` | `/dashboard/api/system-settings` | System-level settings |
| `POST` | `/dashboard/api/system-settings` | Update system-level settings |
| `GET` | `/dashboard/api/browser-settings` | Browser service settings (speed multiplier, delay) |
| `POST` | `/dashboard/api/browser-settings` | Update browser service settings |
| `POST` | `/dashboard/api/default-model` | Set the default LLM model in mesh.yaml |
| `GET` | `/dashboard/api/model-health` | Model health and failover status — per-model success/failure counts, cooldown status, active model in chain |
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
| `POST` | `/dashboard/api/browser-login-request/{request_id}/cancel` | Cancel a single browser-login request by ID |
| `POST` | `/dashboard/api/browser-captcha-help-request/{request_id}/cancel` | Cancel a single CAPTCHA-help request by ID |
| `POST` | `/dashboard/api/agents/{agent_id}/browser/import_cookies` | Operator-only cookie/session import (Playwright JSON or Netscape TSV); 10/hr rate limit per (operator, agent) |

**Browser — Metrics & Solver**

| Method | Path | Description |
|--------|------|-------------|
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
| `GET` | `/dashboard/api/billing/captcha-rollup?tenant=…&period=daily\|weekly\|monthly` | CSV export of per-tenant CAPTCHA-solver spend (operator-curl only — no UI). See [CAPTCHA Rollup CSV](#captcha-rollup-csv) |

> Mobile emulation profiles are env-only — set `BROWSER_DEVICE_PROFILE` to a profile name (e.g. `iphone_15`, `pixel_8`); there is no dashboard surface and no per-agent override endpoint.

**Static Assets**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard/static/{file_path:path}` | Static asset proxy with `?v=…` cache-busting querystring (the SPA template renders `ASSET_VERSION`) |

**WebSocket**

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws/events` | Real-time event stream (unfiltered on wire; SPA filters in JS) |

## Accessibility

The dashboard includes several accessibility features:

- **ARIA roles** — Tab containers use `role="tablist"` / `role="tab"` with `aria-selected`. The chat modal uses `role="dialog"` with `aria-modal` and `aria-label`. The tutorial modal captures focus and restores it on close.
- **Keyboard navigation** — Escape closes the chat modal and the tutorial modal. Focus management within modal dialogs.
- **Reduced motion** — A `prefers-reduced-motion` media query disables animations and transitions for users who prefer reduced motion.
- **Color contrast** — Stat labels, action buttons, and queue status indicators use colors that meet accessibility contrast guidelines.
- **Mobile responsive** — Navigation shows icons-only on narrow screens (< 640px).

## Source Files

| File | Role |
|------|------|
| `src/dashboard/server.py` | FastAPI router with all 143 API endpoints, CSP + CSRF wiring, VNC URL injection |
| `src/dashboard/templates/index.html` | Dashboard HTML (Alpine.js SPA template + Tailwind via CDN). Renders with `autoescape=True` |
| `src/dashboard/static/js/app.js` | Dashboard application logic — top-nav tabs, Work surface, wizard / tutorial state, identity tabs |
| `src/dashboard/static/js/websocket.js` | WebSocket client with exponential-backoff reconnect (`reconnectIn` countdown) |
| `src/dashboard/static/css/dashboard.css` | Custom styles |
| `src/dashboard/events.py` | EventBus for real-time event distribution (ring buffer, lock-protected emit, in-process listener API) |
| `src/dashboard/auth.py` | `ol_session` cookie verification, hosted-mode detection via `/opt/openlegion/.subdomain`, 24h `COOKIE_MAX_AGE` + 5m skew |
| `src/dashboard/notifications.py` | Persistent SQLite notifications store backing the top-nav bell |
| `src/dashboard/telemetry.py` | Frontend telemetry sink with `_MAX_EVENTS=100_000` retention + 60/min per-session rate limit |
| `src/dashboard/platform_success.py` | Per-tenant success scoring (backs `/api/dashboard/platform-success`) |
