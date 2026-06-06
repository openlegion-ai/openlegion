# Triggering & Automation

OpenLegion agents can be triggered automatically through cron schedules, heartbeat monitoring, webhooks, and pub/sub events.

## Cron Jobs

Scheduled tasks that dispatch messages to agents at regular intervals. The cron scheduler runs in the mesh host (not agent containers), so schedules survive container restarts.

The scheduler loop ticks every `TICK_INTERVAL = 5` seconds (`src/host/cron.py:83`). 5-field cron jobs are rate-limited to fire at most once per minute via a `last_run` dedup guard, and each job runs under a per-job `asyncio.Lock` — if a previous run is still in flight when the next tick arrives, the new run is skipped rather than overlapping. When an agent is removed, all of its jobs are purged automatically via `remove_agent_jobs(agent_id)`.

### Schedule Syntax

Two formats are supported:

**5-field cron expressions** (minute granularity):

```
┌───────── minute (0-59)
│ ┌─────── hour (0-23)
│ │ ┌───── day of month (1-31)
│ │ │ ┌─── month (1-12)
│ │ │ │ ┌─ day of week (0-6, 0=Sunday)
│ │ │ │ │
* * * * *
```

Examples:
```
0 9 * * 1-5       # 9:00 AM weekdays
*/15 * * * *       # Every 15 minutes
0 0 1 * *          # First day of every month
30 14 * * 0        # 2:30 PM every Sunday
```

**Interval shorthand**:

```
every 30m          # Every 30 minutes
every 2h           # Every 2 hours
every 1d           # Every day
every 5s           # Every 5 seconds (for testing)
```

**Note:** 6-field cron (with seconds) is not supported. Use `every Ns` for sub-minute intervals.

### Creating Cron Jobs

**Via agent tool** (agents create their own schedules):

```
Agent: I'll set up a daily check.
→ set_cron(schedule="0 9 * * 1-5", message="Check for new leads")
```

**Via mesh API**:

```bash
curl -X POST http://localhost:8420/mesh/cron \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  -d '{
    "agent_id": "researcher",
    "schedule": "every 2h",
    "message": "Review pending tasks and take action"
  }'
```

> **Note:** Mesh API endpoints require authentication via the `MESH_AUTH_TOKEN` header. Agents receive this token automatically; external callers need to provide it.

### Managing Cron Jobs

**List jobs:**
```bash
curl -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  http://localhost:8420/mesh/cron
```

**Remove a job:**
```bash
curl -X DELETE -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  http://localhost:8420/mesh/cron/<job_id>
```

**Update a job:**
```bash
curl -X PUT -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8420/mesh/cron/<job_id> \
  -d '{"schedule": "every 1h", "message": "Updated message"}'
```

The updatable fields (`_UPDATABLE_FIELDS`) are: `schedule`, `message`, `enabled`, `suppress_empty`, `tool_name`, and `tool_params`. All other fields (e.g., `agent`, `heartbeat`) are immutable after creation. To enable or disable a job without deleting it, set `enabled: true` or `enabled: false`.

**Via agent tools:**
- `list_cron()` -- list all jobs for the agent
- `remove_cron(job_id)` -- remove a job

### Empty Response Suppression

By default, cron jobs suppress empty or trivial agent responses. The complete set of suppressed response strings is: `""` (empty), `"ok"`, `"heartbeat_ok"`, `"nothing to do"`, `"no updates"`. This prevents notification spam when an agent has nothing to report. Disable with `suppress_empty: false`.

## Heartbeats

Heartbeats are a cost-efficient form of autonomous monitoring. They run **cheap deterministic probes first**, and only invoke the LLM when probes find actionable items.

The default heartbeat schedule is `"every 15m"` (`DEFAULT_HEARTBEAT_SCHEDULE`, `src/host/cron.py:84`). The **operator agent's heartbeat is forced to `"every 15m"` regardless of `mesh.heartbeat_schedule`** — see `src/cli/runtime.py:973`.

### How Heartbeats Work

```
Cron tick
  → Run deterministic probes (no LLM, zero cost)
  → Fetch agent context (HEARTBEAT.md + daily logs via /heartbeat-context)
  → All clean + default HEARTBEAT.md + no activity? → Skip entirely (zero cost)
  → Otherwise → Build enriched message with:
      Rules (HEARTBEAT.md) + Recent activity (daily logs)
      + Probe alerts + Pending signal/task details
    → Dispatch to agent
    → Agent takes action using tools
```

### Built-in Probes

| Probe | What It Checks | Triggers When |
|-------|---------------|---------------|
| `disk_usage` | Agent data volume usage | > 85% full |
| `pending_signals` | Blackboard signals for the agent | Any pending signals |
| `pending_tasks` | Blackboard tasks for the agent | Any pending tasks |

### Creating a Heartbeat

**Via agent tool:**

```
Agent: I'll monitor the system every 30 minutes.
→ set_cron(schedule="every 30m", heartbeat=true)
```

The three probes (`disk_usage`, `pending_signals`, `pending_tasks`) run automatically -- you don't need to specify them, and you cannot add custom probes. Define your escalation rules in `HEARTBEAT.md` instead.

**Via mesh API** (add `heartbeat: true`):

```bash
curl -X POST http://localhost:8420/mesh/cron \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  -d '{
    "agent_id": "researcher",
    "schedule": "every 30m",
    "message": "heartbeat",
    "heartbeat": true
  }'
```

### HEARTBEAT.md

Each agent can have a `HEARTBEAT.md` file in its workspace that defines autonomous monitoring rules. The heartbeat system auto-loads HEARTBEAT.md content and includes it directly in the heartbeat message — the agent doesn't need to spend a tool call reading it.

```markdown
# Autonomous Rules

## When disk usage is high
- Archive old daily logs from memory/
- Compact MEMORY.md by removing stale entries

## When pending signals arrive
- Process each signal and respond to the sender
- Clear signals after processing

## When pending tasks exist
- Execute tasks in priority order
- Report completion to the blackboard
```

### Editing Heartbeat Schedules

Heartbeat schedules can be changed via:
1. **Dashboard Automations tab** — edit the cron schedule directly
2. **Operator agent** — via `edit_agent(agent_id, field="heartbeat_schedule", value=...)` with a cron expression (e.g. `0 */8 * * *`) or interval (e.g. `every 6h`)

`heartbeat_schedule` is a **soft** field on `edit_agent` — it applies immediately with a 5-minute Undo receipt and retargets the live cron job in lockstep with the YAML write.

Both methods sync the schedule to the cron scheduler immediately — no restart required. The dashboard displays update in real-time.

### Enriched Heartbeat Messages

When a heartbeat fires, the agent receives a single message with all the context it needs to act:

| Section | Content | Source |
|---------|---------|--------|
| **Your Heartbeat Rules** | Custom HEARTBEAT.md content | Agent's `/heartbeat-context` endpoint |
| **Your Recent Activity** | Last 2 days of daily logs (capped at 4000 chars) | Agent's `/heartbeat-context` endpoint (transport cap 8000 chars); 4000-char display cap applied by cron scheduler |
| **Probe Alerts** | Triggered probe results with details | Deterministic probes |
| **Pending Signals** | Actual blackboard signal content (up to 5 items) | Blackboard `signals/{agent}` |
| **Pending Tasks** | Actual blackboard task content (up to 5 items) | Blackboard `tasks/{agent}` |

This replaces the previous pattern where agents had to waste tool calls reading HEARTBEAT.md and querying the blackboard themselves.

### Skip-LLM Optimization

Heartbeats skip the LLM dispatch entirely (zero cost) when all four conditions are met:

1. **Not manually triggered** — manual runs (e.g. `/cron run`, dashboard "Run now") always dispatch so the user sees a result
2. **HEARTBEAT.md is default** — the file is empty (after stripping whitespace) or its content is **exactly** `# Heartbeat Rules` (exact equality, not prefix matching). A file containing `# Heartbeat Rules` followed by any additional content is considered customized and will be sent to the LLM.
3. **No recent activity** — daily logs are empty
4. **No probes triggered** — disk usage normal, no pending signals or tasks

This makes always-on heartbeats economically viable even at high frequencies. See `src/host/cron.py:543`.

#### Forcing the LLM to run every tick

Pipeline-kicker agents — those whose entire job is to wake up on a schedule and decide what to do next — typically have no probes registered and ship with an empty `HEARTBEAT.md`. Under the four-condition skip-LLM check above they would never actually wake. Set `force_llm: true` on the cron job to bypass the skip and dispatch to the LLM on every tick regardless of the other predicates:

```bash
curl -X PUT http://localhost:8420/mesh/cron/<job_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  -d '{"force_llm": true}'
```

You're opting out of the cost optimization — pick `force_llm: true` only for agents that genuinely need to think every tick. `force_llm` is in `_UPDATABLE_FIELDS` so you can flip it at runtime; it persists across mesh restarts via `config/cron.json`.

## Tool-Mode Cron

In addition to dispatching a message to an agent (message mode) or running a heartbeat (heartbeat mode), cron jobs can invoke a tool directly — **without any LLM call**. This is useful for fully deterministic periodic operations where no reasoning is needed.

Set `tool_name` (and optionally `tool_params` as a JSON-encoded dict) when creating a job:

**Via mesh API:**

```bash
curl -X POST http://localhost:8420/mesh/cron \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MESH_AUTH_TOKEN" \
  -d '{
    "agent_id": "researcher",
    "schedule": "every 1h",
    "message": "",
    "tool_name": "web_search",
    "tool_params": "{\"query\": \"openlegion news\"}"
  }'
```

When `tool_name` is set, the cron scheduler calls the tool directly via the agent's invoke endpoint, bypassing the LLM entirely. The tool result is recorded in traces and, if non-empty and `suppress_empty` is not false, logged. The `message` field is ignored when `tool_name` is present.

Both `tool_name` and `tool_params` are in the `_UPDATABLE_FIELDS` set and can be changed via `PUT /mesh/cron/<job_id>`.

## Webhooks

External systems can dispatch messages to agents via named webhooks.

```bash
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme Corp", "source": "website"}'
```

The webhook payload is included in the message dispatched to the configured agent. Request bodies are capped at **1 MB** (`_MAX_WEBHOOK_BODY = 1_048_576`, `src/host/webhooks.py:156`); oversized requests are rejected with HTTP 413 (Content-Length pre-check followed by a raw-body post-check). The payload JSON is truncated to **3000 chars** when assembled into the dispatch message and run through `sanitize_for_prompt()` before reaching the agent (`src/host/webhooks.py:26-32`).

### Creating Webhooks

Webhook creation is a **dashboard-only** flow. The endpoint is `POST /api/webhooks` on the dashboard router (`src/dashboard/server.py:6659`); it uses dashboard SSO cookie auth (`ol_session`), not `MESH_AUTH_TOKEN`. There is no `/mesh/webhooks` endpoint.

In practice you create webhooks from the dashboard System → Integrations tab. The request body is `{"name": "<label>", "agent": "<agent_id>", "instructions"?: "<extra>", "secret"?: "<existing-secret>"}`; the server stores `require_signature = bool(body.get("secret"))` and assigns a random 32-byte hex secret (`secrets.token_hex(32)`) if none was supplied. The created hook's secret is returned **once** in the response — store it immediately.

To rotate the secret on an existing hook, `PATCH /api/webhooks/{hook_id}` with `{"regenerate_secret": true}`.

### Webhook Signature Verification

Webhooks can optionally require HMAC-SHA256 signature verification. When a webhook is created with a secret (`require_signature` becomes true), callers must include the signature in the `x-webhook-signature` header (HTTP headers are case-insensitive; not to be confused with WhatsApp's `X-Hub-Signature-256`):

```bash
# Send a signed payload
BODY='{"company": "Acme"}'
SIG=$(echo -n "$BODY" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}')
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -H "x-webhook-signature: $SIG" \
  -d "$BODY"
```

Unsigned requests to a signature-required webhook are rejected with HTTP 401.

## Pub/Sub Events

Agents can publish events to notify other agents.

### Publishing Events

**From agent tools:**
```
Agent: Research is done, notifying the team.
→ publish_event(topic="research_complete", data={"prospect": "Acme Corp", "score": 8})
```

The matching subscribe-side tool is `subscribe_event(topic, callback_message)` (`src/agent/builtins/mesh_tool.py:284`).

### Permission Requirements

Pub/sub is permission-controlled. Agents need explicit ACLs:

```json
{
  "researcher": {
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"]
  }
}
```

## Agent-to-Agent Coordination

Pub/sub is a fan-out signal — for **direct** agent-to-agent work handoffs use the structured coordination protocol in `src/agent/builtins/coordination_tool.py`:

| Tool | Purpose |
|------|---------|
| `hand_off(to, summary, data)` | Writes `tasks/{to}/{handoff_id}` to the blackboard, then `wake_agent(to, msg, origin=...)`. Propagates `MessageOrigin` so completion routes back to the original channel/user. |
| `check_inbox()` | Pull pending handoffs/signals/tasks for the current agent. |
| `update_status(state, task_id?)` | Report progress on an active task. With 2+ active tasks and no `task_id`, returns an `ambiguous_task` payload so the LLM can pick a `task_id` without a follow-up `check_inbox` call. |
| `complete_task(task_key)` | Mark a task complete; releases the auto-notify back to the originator. |

Unprocessed handoffs expire after `_HANDOFF_TTL = 86_400` seconds (24h) as a safety net (`coordination_tool.py:33`).

### MessageRouter

Cross-agent calls go through `MessageRouter` (`src/host/mesh.py:737`), which resolves the target string to a container URL:

- **Agent ID** — exact match against the agent registry
- **`capability:<name>`** — first agent whose declared capabilities include `<name>`

Permissions are enforced on every routed call; unknown targets return `{"error": "Delivery failed..."}` after a single retry.

### Lanes

Each agent has a FIFO task lane (`src/host/lanes.py`) with two dispatch modes:

| Mode | Behavior |
|------|----------|
| `followup` (default) | Queue the message; process after the current task finishes. |
| `steer` | Inject the message into the agent's active conversation between tool rounds. Rate-limited to `_STEER_WAKEUP_MAX = 10` injections per `_STEER_WAKEUP_WINDOW = 3600` seconds (per-agent). |

When a lane completes a task that originated from a channel handoff (`auto_notify=True`), the result is forwarded back to the originating channel/user with a `[agent_name]` prefix. The send is capped at `_NOTIFY_FORWARD_TIMEOUT = 30` seconds (`lanes.py:30`).

## Notification Routing

Cron and heartbeat results are recorded in traces but **not** automatically pushed to channels. This prevents notification spam from routine checks that have nothing to report.

When an agent has something important to communicate, it uses the `notify_user` tool to explicitly push a message to all connected channels:

```
Agent: The daily report is ready.
-> notify_user(message="Daily report generated and saved to /data/reports/2026-02-22.pdf")
```

This sends the notification to:
1. **CLI REPL** -- printed to terminal
2. **Telegram** -- sent to all paired users
3. **Discord** -- sent to configured channels
4. **Slack** -- sent to configured channels
5. **WhatsApp** -- sent to paired users

Agents can use `notify_user` at any time -- during cron jobs, heartbeats, or regular tasks.

## Source Files

| File | Role |
|------|------|
| `src/host/cron.py` | Cron scheduler, heartbeat probes, interval/cron parsing |
| `src/host/server.py` | Cron management API |
| `src/host/webhooks.py` | Webhook router, HMAC verification, 1 MB body cap |
| `src/host/mesh.py` | Pub/Sub system and `MessageRouter` |
| `src/host/lanes.py` | Per-agent FIFO task queues (followup/steer) |
| `src/agent/builtins/mesh_tool.py` | Agent-side `set_cron`, `list_cron`, `remove_cron`, `publish_event`, `subscribe_event` tools |
| `src/agent/builtins/coordination_tool.py` | `hand_off`, `check_inbox`, `update_status`, `complete_task` |
| `src/dashboard/server.py` | Webhook creation API (`POST /api/webhooks`) |
| `config/cron.json` | Persisted job state |
| `config/webhooks.json` | Persisted webhook config |
