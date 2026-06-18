# Session Observability — durable, correlated session reconstruction

**Date:** 2026-06-18
**Status:** Proposed (plan only — no code yet)

## Motivation

Today there is no reliable way to take a past human-rooted interaction and reconstruct, after the fact, the three layers that matter:

1. **Intent** — what the human was trying to do (their verbatim words).
2. **Action** — what the agent(s) actually did (tool calls, handoffs, task transitions).
3. **Outcome** — how it ended (result, failures with reasons, cost).

This is a general engine capability, not a niche need. Once it exists, multiple consumers benefit: operator self-serve debugging, audit/compliance, per-session cost attribution, support ("what happened in my session?"), quality/eval datasets, and a **closed improvement loop** (operator uses the platform normally → an engineer reconstructs sessions from the trail → diagnoses friction/bugs → fixes). The closed loop is the first consumer that motivated this doc, but the design must serve the engine as a whole, for every operator.

## Current state (audit, 2026-06-18)

### What already works (the backbone)
- `data/tasks.db` `tasks` + `task_events` (`src/host/orchestration.py:308-345`) is durable and queryable: originating user (`origin_kind/channel/user`), the handoff/rework DAG (`parent_task_id` + `previous_task_id`), full status machine, failure reason (`blocker_note`), deliverable (`result_summary`), operator rating (`outcome`), and a per-transition `task_events` audit row.
- A correlation id `trace_id` (`tr_<hex12>`, `src/shared/trace.py:52`) already propagates via the `X-Trace-Id` header + `current_trace_id` contextvar across the agent↔mesh boundary on the **CLI**, lane, and cron paths. The plumbing exists.

### Gaps that block reconstruction
1. **The dashboard mints no `trace_id`** (`src/dashboard/server.py:3611-3619`, stream variant `3657`). The primary UI's sessions produce LLM traces with an empty correlation id — uncorrelatable end-to-end. This is a genuine defect for every operator, independent of dogfooding.
2. **No common key across stores.** `traces.db` keys on `trace_id`; `tasks.db` on `task_id`/origin; `costs.db` (`usage`, `src/host/costs.py:142-154`) on `agent`+`timestamp`; per-agent chat transcripts on nothing durable. No `JOIN` assembles a session — only fuzzy timestamp+agent correlation, ambiguous under concurrency.
3. **Verbatim intent has no central, durable home.** The user's actual words live only in container-local `chat_transcript.jsonl` (`src/agent/workspace.py:1051-1099`), which rotates (drops oldest half), archives on `/chat/reset`, and is lost if the container/volume is wiped. Central stores keep only a normalized task title and a <=120-char redacted `prompt_preview`.
4. **Agent-side work is not traced.** `TraceStore.record` is called only from `src/host/`; agent tool calls, loop iterations, and handoffs never become first-class trace events.
5. **Costs can't be attributed to a task/session** — `usage` rows carry only `agent`+`model`+`timestamp`.
6. **Process logs are ephemeral and bare.** `setup_logging()` (`src/shared/utils.py:210-228`) defaults to JSON but writes to stdout only; ~1 of thousands of log calls carries `trace_id`. Collection means `docker logs` per container, dies with the container.
7. **The live dashboard EventBus is an in-memory ring buffer** (`src/dashboard/events.py`) — not a reconstruction source.

## Design

### The session model (key decision)
`trace_id` is currently **per-turn** (one inbound user message → its fan-out). A *session* is a *conversation* = many turns. Proposed model:
- Keep per-turn `trace_id` as the **atom** of correlation.
- Define a **session as a derived grouping** of turns sharing a conversation (agent + origin chain root), assembled by the reader — **no new stored `session_id` column initially** (reversible, no migration). Promote to a stored `session_id` only if the derived grouping proves insufficient (e.g. cross-agent conversations that the origin chain can't express).

### Correlation key
`trace_id` becomes the single key stamped on every durable store so a session is a `JOIN`, not archaeology. Redaction posture is unchanged — `trace_id` is an opaque token, not sensitive; verbatim-intent capture must reuse the existing `sanitize_for_prompt`/redaction boundary before storage.

## Phases

Each phase is a normal worktree → PR → CI cycle. Tests run in a subagent per repo convention.

### Phase 1 — Correlate the entry points
- Mint `trace_id` at the dashboard chat + chat/stream entry (mirror the CLI's `new_trace_id()`/`current_trace_id.set()`); add the same at channel `async_dispatch` so messaging surfaces correlate too.
- Stamp `trace_id` on `tasks`, `usage` (costs), and chat-transcript rows.
- **Outcome:** every human-rooted action across every surface becomes joinable. Fixes the dashboard-untraceable defect.

### Phase 2 — Capture intent durably
- Persist the verbatim inbound user message centrally, keyed by `trace_id` + `MessageOrigin`, through the existing redaction boundary, with retention aligned to tasks (90d).
- **Outcome:** operators keep their actual words through container wipes/resets.

### Phase 3 — The reader (the consumed artifact)
- `openlegion session <trace_id>` and `openlegion sessions --since <when> [--user X]` assemble one timeline: intent → agent actions (from traces + task_events) → outcome (status/blocker_note/result_summary) → cost rollup. `--json` for machine consumption.
- Later: a dashboard session-timeline view reusing the same assembler.
- **Outcome:** self-serve debugging + cost attribution for any operator; the closed-loop engineer reads this directly.

### Phase 4 — Depth
- Emit agent-side trace events (`tool_call`, `handoff`, loop `iteration`) under the inherited `trace_id` from `src/agent/`.
- Add `trace_id`/`agent_id`/`task_id` to the structured log formatter via contextvar; optional off-box log shipping hook.
- **Outcome:** full "what did the agent actually do" detail + production-grade observability.

## Non-goals / guardrails
- No change to the redaction/credential boundary — verbatim capture goes through `sanitize_for_prompt`/`deep_redact`.
- No new module-level globals (Constraint #8); correlation flows via the existing contextvars.
- Respect existing retention/rotation semantics; do not silently extend retention.
- Derived session grouping before any stored `session_id` (reversibility).

## Open questions
- Q1: Derived session grouping vs. stored `session_id` — start derived; revisit after Phase 3 dogfooding.
- Q2: Should verbatim-intent capture live in a new central store or extend an existing one (e.g. a `messages` table vs. annotating `tasks`)? Decide at Phase 2 design.
- Q3: How much full-prompt/response retention is acceptable vs. the current <=500-char previews, given redaction + storage cost? Decide at Phase 4.
