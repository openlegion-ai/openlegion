# Workflows

OpenLegion orchestrates multi-agent work through **deterministic YAML-defined DAG workflows**. No LLM decides what runs next -- the execution order is explicit, predictable, and auditable.

## Overview

Workflows are directed acyclic graphs (DAGs) where each step assigns a task to a specific agent. Steps can depend on other steps, have conditions, retry policies, and failure handlers.

```yaml
# config/workflows/prospect_pipeline.yaml
name: prospect_pipeline
trigger: new_prospect
timeout: 600
steps:
  - id: research
    agent: researcher
    task_type: research_prospect
    input_from: trigger.payload

  - id: qualify
    agent: qualifier
    task_type: qualify_lead
    depends_on: [research]
    condition: "research.result.score >= 5"

  - id: outreach
    agent: outreach
    task_type: draft_email
    depends_on: [qualify]
    condition: "qualify.result.qualified == true"
```

## Workflow Structure

### Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | -- | Unique workflow identifier |
| `trigger` | string | Yes | -- | PubSub topic that starts the workflow |
| `timeout` | integer | No | 600 | Global timeout in seconds |
| `steps` | list | Yes | -- | Ordered list of step definitions |
| `on_complete` | string | No | -- | PubSub topic to publish when workflow finishes |

### Step Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | -- | Unique step identifier within the workflow |
| `agent` | string | No* | -- | Agent to assign the task to |
| `capability` | string | No* | -- | Alternative to `agent` -- match by capability |
| `task_type` | string | Yes | -- | Task type identifier for the agent |
| `input_from` | string | No | -- | Input data source (see Data Flow below) |
| `depends_on` | list[string] | No | `[]` | Step IDs that must complete first |
| `condition` | string | No | -- | Comparison expression to evaluate before running |
| `on_failure` | string | No | `abort` | Failure strategy: `retry`, `skip`, `abort`, or `fallback` |
| `max_retries` | integer | No | 3 | Max retry attempts (when `on_failure: retry`) |
| `fallback_agent` | string | No | -- | Agent to use when `on_failure: fallback` |
| `timeout` | integer | No | 120 | Per-step timeout in seconds |
| `requires_approval` | boolean | No | false | Pause for human approval before executing |
| `approval_timeout` | integer | No | 3600 | Seconds to wait for approval |

*Either `agent` or `capability` must be provided.

## Execution Model

### DAG Resolution

1. Orchestrator parses the YAML into a step graph
2. Topological sort determines execution order
3. Steps with no dependencies run first (potentially in parallel)
4. Each step waits for all `depends_on` steps to complete
5. Conditions are evaluated before running each step

### Condition Evaluation

Conditions use a **safe regex-based parser** (no `eval()`). Each condition is a single comparison expression:

```yaml
condition: "research.result.score >= 5"
condition: "qualify.result.qualified == true"
condition: "research.result.category == 'enterprise'"
```

Supported operators: `==`, `!=`, `>`, `<`, `>=`, `<=`

Variable access uses dot-notation: `step_id.result.field`

**Note:** Only single comparisons are supported. Boolean operators (`and`, `or`, `not`) are not implemented. If you need complex conditions, split them across multiple steps.

If a condition has invalid syntax, the step is **skipped** with a warning (the workflow continues).

### Data Flow

The `input_from` field supports three input sources:

**Step results** (dot-notation):
```yaml
steps:
  - id: research
    agent: researcher
    task_type: find_info
    # Result: {"score": 8, "summary": "..."}

  - id: report
    agent: writer
    task_type: write_report
    depends_on: [research]
    input_from: research.result
    # Receives: {"score": 8, "summary": "..."} as input
```

**Trigger payload:**
```yaml
input_from: trigger.payload
# Receives the event data that started the workflow
```

**Blackboard keys:**
```yaml
input_from: blackboard://context/market_data
# Reads directly from the shared blackboard
```

## Failure Handling

Five strategies are available via `on_failure`:

### Retry (default: 3 attempts with exponential backoff)

```yaml
steps:
  - id: api_call
    agent: researcher
    task_type: fetch_data
    on_failure: retry
    max_retries: 5
    # Retries with exponential backoff: 1s, 2s, 4s, 8s, 16s
```

### Skip (continue workflow, mark step as complete with skipped flag)

```yaml
steps:
  - id: optional_enrichment
    agent: researcher
    task_type: enrich_data
    on_failure: skip
    # Workflow continues even if this step fails
```

### Abort (stop the workflow immediately)

```yaml
steps:
  - id: critical_step
    agent: researcher
    task_type: validate
    on_failure: abort
    # Default behavior -- workflow stops on failure
```

### Fallback (route to a different agent)

```yaml
steps:
  - id: primary
    agent: researcher
    task_type: research
    on_failure: fallback
    fallback_agent: backup_researcher
    # If primary fails, the task is re-assigned to backup_researcher
```

### Dependency Failure (automatic)

When a step's dependency has failed, the step is automatically skipped without execution. The orchestrator sets `status="skipped"` with an error noting the dependency failure. This prevents wasted work on steps whose inputs are unavailable.

```yaml
steps:
  - id: research
    agent: researcher
    task_type: fetch_data
    # If this fails and on_failure is "abort"...

  - id: report
    agent: writer
    task_type: write_report
    depends_on: [research]
    # ...this step is auto-skipped with status="skipped"
```

This behavior is not configurable — it applies automatically to any step whose dependency has a failed result. It differs from `on_failure: skip` (which marks the *failed* step itself as complete with a skipped flag).

## Triggering Workflows

### Via PubSub

```python
# From an agent tool
await publish_event("new_prospect", {"company": "Acme Corp"})
```

### Via Webhook

```bash
curl -X POST http://localhost:8420/webhook/hook/<hook_id> \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme Corp"}'
```

### Via Cron

Configure a cron job that publishes to the workflow's trigger topic. See [Triggering & Automation](triggering.md).

## Source Code

| File | Responsibility |
|------|---------------|
| `src/host/orchestrator.py` | DAG parser, executor, condition evaluator, retry logic |
| `src/shared/types.py` | `WorkflowStep` and `WorkflowDefinition` Pydantic models |
| `config/workflows/*.yaml` | Workflow definitions |
| `src/host/mesh.py` | PubSub that triggers workflows |
