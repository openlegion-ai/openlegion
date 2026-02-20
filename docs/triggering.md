# Triggering & Automation

OpenLegion agents can be triggered automatically through cron schedules, heartbeat monitoring, webhooks, and pub/sub events.

## Cron Jobs

Scheduled tasks that dispatch messages to agents at regular intervals. The cron scheduler runs in the mesh host (not agent containers), so schedules survive container restarts.

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
  -d '{
    "agent_id": "researcher",
    "schedule": "every 2h",
    "message": "Review pending tasks and take action"
  }'
```

### Managing Cron Jobs

**List jobs:**
```bash
curl http://localhost:8420/mesh/cron
```

**Remove a job:**
```bash
curl -X DELETE http://localhost:8420/mesh/cron/<job_id>
```

**Via agent tools:**
- `list_cron()` -- list all jobs for the agent
- `remove_cron(job_id)` -- remove a job

### Empty Response Suppression

By default, cron jobs suppress empty or trivial agent responses (e.g., "ok", "nothing to do"). This prevents notification spam when an agent has nothing to report. Disable with `suppress_empty: false`.

## Heartbeats

Heartbeats are a cost-efficient form of autonomous monitoring. They run **cheap deterministic probes first**, and only invoke the LLM when probes find actionable items.

### How Heartbeats Work

```
Cron tick
  → Run deterministic probes (no LLM, zero cost)
    → All clean? → Skip, do nothing
    → Probe triggered? → Dispatch to agent with probe results
      → Agent reads HEARTBEAT.md for rules
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
→ set_heartbeat(schedule="every 30m")
```

The probes (disk_usage, pending_signals, pending_tasks) run automatically -- you don't need to specify them. Define your escalation rules in `HEARTBEAT.md` instead.

**Via mesh API** (add `heartbeat: true`):

```bash
curl -X POST http://localhost:8420/mesh/cron \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "researcher",
    "schedule": "every 30m",
    "message": "heartbeat",
    "heartbeat": true
  }'
```

### HEARTBEAT.md

Each agent can have a `HEARTBEAT.md` file in its workspace that defines autonomous monitoring rules. When a heartbeat probe triggers, the agent receives the probe results and is instructed to check `HEARTBEAT.md` for its response rules.

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

## Webhooks

External systems can trigger workflows via HTTP webhooks.

### Triggering a Workflow

```bash
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme Corp", "source": "website"}'
```

The webhook payload is passed as `trigger.payload` to the workflow's first step.

### Workflow Cron Integration

Cron jobs can trigger workflows directly instead of messaging agents:

```json
{
  "id": "cron_abc123",
  "agent": "orchestrator",
  "schedule": "0 9 * * 1-5",
  "message": "",
  "workflow": "daily_pipeline",
  "workflow_payload": "{\"date\": \"today\"}"
}
```

When `workflow` is set, the cron job calls `workflow_trigger_fn` instead of dispatching to an agent.

## Pub/Sub Events

Agents can publish events that trigger workflows or notify other agents.

### Publishing Events

**From agent tools:**
```
Agent: Research is done, notifying the team.
→ publish_event(topic="research_complete", data={"prospect": "Acme Corp", "score": 8})
```

### Subscribing to Events

Workflows subscribe automatically via their `trigger` field:

```yaml
# config/workflows/prospect_pipeline.yaml
name: prospect_pipeline
trigger: new_prospect   # Listens for "new_prospect" events
steps:
  - id: research
    agent: researcher
    task_type: research_prospect
    input_from: trigger.payload
```

When any agent publishes to `new_prospect`, this workflow starts automatically.

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

## Notification Routing

When cron jobs, heartbeats, or workflows produce results, notifications are pushed to connected channels:

1. **CLI REPL** -- printed to terminal
2. **Telegram** -- sent to all paired users
3. **Discord** -- sent to configured channels

Empty responses are suppressed by default to avoid spam.

## Source Files

| File | Role |
|------|------|
| `src/host/cron.py` | Cron scheduler, heartbeat probes, interval/cron parsing |
| `src/host/server.py` | Webhook endpoints, cron management API |
| `src/host/orchestrator.py` | Workflow executor (triggered by pub/sub and webhooks) |
| `src/host/mesh.py` | PubSub system for event-driven triggering |
| `src/agent/builtins/mesh_tool.py` | Agent-side `set_cron`, `set_heartbeat`, `list_cron`, `remove_cron` tools |
| `config/cron.json` | Persisted job state |
