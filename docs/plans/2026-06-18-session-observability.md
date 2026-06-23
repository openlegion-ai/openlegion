# Session Observability — durable, correlated session reconstruction

**Date:** 2026-06-18
**Status:** Phase 1 SHIPPED in PR #1149 (CI green; not yet merged). Phases 2–4 detailed below, not yet implemented.

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

### Phase 1 — Correlate the entry points — SHIPPED (PR #1149, CI green, not yet merged)
What shipped:
- A `trace_id` is now minted at the dashboard `/chat` + `/chat/stream` entries and at the channel/webhook dispatch path, via a single `_resolve_dispatch_trace_id` helper in `src/cli/runtime.py`. The helper **never clobbers an existing trace** — `current_trace_id.get() or new_trace_id()` semantics — so a turn that already carries an inbound `X-Trace-Id` keeps it, and only a bare context (lane wakes, cron, webhook) gets a fresh id. This fixes the dashboard-untraceable defect (gap #1) for every operator.
- A nullable `trace_id` column was added to three durable stores, each behind an idempotent PRAGMA-guarded migration (the `ALTER TABLE … ADD COLUMN` + duplicate-column swallow idiom already used by `TraceStore`):
  - `tasks` (`src/host/orchestration.py`)
  - `usage` (`src/host/costs.py`) — costs are now attributable to a session (closes gap #5)
  - transcript rows (`src/agent/workspace.py`)
- The columns are **populated** by seeding `current_trace_id` from the inbound `X-Trace-Id` header at **4 mesh endpoints**, so the contextvar is live when those stores write.
- **Outcome delivered:** every human-rooted action across every surface now shares a single join key. A session is now (almost) a `JOIN`, not archaeology. Phases 2–4 build the verbatim-intent store, the reader, and agent-side depth on top of this key.

Carryover into later phases: the reader (Phase 3) and the intent-store join (Phase 2) only become fully correct once #1149 **merges** — until then `trace_id` is non-null only on instances running the PR build.

### Phase 2 — Capture intent durably
Persist the verbatim inbound user message centrally, keyed by `trace_id` + `MessageOrigin`, so intent survives container wipes/resets/deploys (gap #3). Today the user's words live only in container-local `chat_transcript.jsonl`, which rotates and dies with the container — exactly the layer most likely lost when you go to investigate a hosted box after a deploy.

**Store.** New `IntentStore` in `src/host/intent.py`, following the `TraceStore` idiom (`src/host/traces.py:28-100`): single persistent connection via `open_db` (`src/shared/sqlite_helpers.py:8`), `PRAGMA journal_mode=WAL`, `CREATE TABLE IF NOT EXISTS` + index. Schema:
```
intent(id PK, trace_id TEXT, timestamp REAL, origin_kind TEXT, origin_channel TEXT,
       origin_user TEXT, agent TEXT, message TEXT, meta_json TEXT)
```
Indexes on `(trace_id)` and `(timestamp)`.

**Redaction at storage.** Mirror `TraceStore.record`'s centralized redaction (`traces.py:90-92`): run `redact_text_with_urls` (`src/shared/redaction.py:359`) on `message` **inside** the insert method, so no callsite can regress (rationale tag H16). `sanitize_for_prompt` (`src/shared/utils.py:145`) is already applied upstream at every surface; redaction-for-storage is the additional layer that keeps a credential-in-chat from persisting in plaintext.

**Capture point.** The central chokepoint is `_direct_dispatch` (`src/cli/runtime.py:781-862`), where `trace_id` is guaranteed minted (`:805`) and `origin` + `message` + `agent` are all in scope. An existing `TraceStore.record(... event_type="chat", detail=message[:200])` already fires there (`:807-811`) but **truncates to 200 chars** — intent capture stores the FULL verbatim message alongside it. Recommendation: capture at the dispatch layer (one touchpoint covering dashboard + channels + webhooks). Webhooks are machine-origin (`system_note=True`, `src/host/webhooks.py:260`) — still capture, but stamp `origin_kind` accordingly so the reader can distinguish human vs. machine triggers.

**Wiring.** Instantiate inside `create_app` exactly like `summaries_store` (`src/host/server.py:887-894`), env override `OPENLEGION_INTENT_DB` → `data/intent.db`, attach as `app.intent_store`.

**Retention.** Mirror tasks' 90-day window: a `retention_until` column + a rate-limited `_safe_reap` like `WorkSummariesStore` (`src/host/summaries.py:462-497`, 60s min interval). (The 90-day window matches the session/forensic semantics, vs. traces' shorter rolling `_maybe_gc_old`.)

**Tests:** store insert/query by `trace_id`; redaction-at-storage (a secret in the message never persists raw); retention reap idempotency; dispatch capture wiring (a dashboard turn lands a row with the right origin).

**Outcome:** operators keep their actual words, centrally and durably — the intent layer survives container churn. This is the phase that makes hosted-VPS reconstruction trustworthy after deploys.

### Phase 3 — The reader (the consumed artifact)
The operator-facing tool that assembles a session from the stores. This is what an engineer (or the closed-loop) actually reads.

**Commands** in `src/cli/main.py`, following the existing pattern (global `--json` `:88-94`; exemplar `tasks` command `:718-756`; `--port`):
- `openlegion session <trace_id>` — full timeline for one session.
- `openlegion sessions --since <when> [--user X] [--agent X] [--json]` — recent sessions, one summary row each.

**Data access — read host SQLite directly, read-only.** Every existing read command goes through mesh HTTP (`_mesh_get` `:603`), but the reader should instead open the host DBs read-only (`sqlite3.connect("file:...?mode=ro", uri=True)`). Rationale, called out deliberately because it diverges from convention: a forensic reader must work **even when the mesh is down or wedged** — which is exactly when you debug — and `_mesh_get` hard-exits with "Mesh is not running" (`:610`). On a hosted VPS over SSH you already have filesystem access, so direct read is both simpler and more robust. Add a canonical `DATA_DIR = PROJECT_ROOT / "data"` to `src/cli/config.py` (none exists today; DB paths are bare strings in `runtime.py`/`server.py`).

**Assembly by `trace_id`** (correct once #1149 merges so `tasks`/`usage` carry the column):
- **Intent** (verbatim) — `intent.db` (Phase 2).
- **Actions** — `traces.db` via `TraceStore.get_trace(trace_id)` (`traces.py:137`): dispatch, `llm_call`, blackboard, pubsub, plus Phase-4 `tool_call`/`handoff`/`iteration`.
- **Outcome** — `tasks.db WHERE trace_id=?`: status, `blocker_note`, `result_summary`, and the `parent_task_id`/`previous_task_id` DAG.
- **Cost** — `usage WHERE trace_id=?`: per-session token/$ rollup.

Render one chronological timeline; `--json` for machine consumers. `sessions --since` builds on a `list_trace_summaries`-style query (`traces.py:153`, `trigger_preview`) joined with intent.

**Known limitation (document it):** chat transcripts are container-local (per-agent), so the host-side timeline = intent + traces + tasks + costs. Full per-turn transcript text needs a separate per-container fetch (future: `mesh_client.get_agent_history`).

Later: a dashboard session-timeline view reusing the same assembler.

**Tests:** assembly correctness against seeded DBs; read-only open; resilience when the mesh process is down.

**Outcome:** self-serve debugging + per-session cost attribution for any operator; the SSH-on-box tool for "what went wrong on this VPS?"; the closed-loop engineer reads this directly.

### Phase 4 — Depth (agent-side traces + log correlation)
Make "what did the agent actually do" first-class, and put correlation IDs on every log line.

**Agent-side trace emission.** Today `TraceStore.record` is called only from `src/host/` — the agent never writes traces. Reuse the proven pattern by which `llm_call` events already reach the trace from agent activity: the agent calls a mesh endpoint with `x-trace-id`, and the mesh records on its behalf (`src/host/server.py:2053-2063`).
- Add `POST /mesh/traces` ingest endpoint (operator/internal gated), reading `x-trace-id` and calling `trace_store.record(...)` — mirroring the `llm_call` ingest.
- Add a `record_trace(...)` method to `src/agent/mesh_client.py` (none today; model it on `publish_event`/`notify_user`, which POST with `_trace_headers()` `:143-146`). Make it best-effort / non-blocking — never stall the loop.
- Emit at: `tool_call` — `src/agent/loop.py` `_run_tool` (`:3519`, the single `tools.execute` dispatch `:3542` covering every tool); `handoff` — `src/agent/builtins/coordination_tool.py` right after `create_task` (`:404`, task_id known); loop `iteration` — `_mark_iteration` (`:523`).
- (Rejected alternative: the agent writing its own SQLite + host merge — breaks container isolation, which is the whole reason traces are host-only.)

**Log correlation.** In `StructuredFormatter.format` (`src/shared/utils.py:181-195`) inject `current_trace_id`/`current_task_id` (`src/shared/trace.py:35/48`) and a new `current_agent_id` contextvar (add to `trace.py`, set once at agent boot) onto every line, keeping the existing `extra_data` merge winning. Mirror into `TextFormatter`. This instantly correlates every log line in both host and agent processes (both call `setup_logging`).

**Off-box log shipping.** Optional hook only — out of MVP scope; flagged so logs can survive container removal in production.

**Tests:** ingest endpoint auth + record; formatter contextvar injection; agent emits `tool_call`/`handoff` under the inherited `trace_id`.

**Outcome:** full agent-action detail in the timeline + production-grade, correlated logs.

## Usage (reader)
The operator-facing entry point (ships in Phase 3):
- `openlegion session <trace_id>` → prints the assembled timeline: the **intent** line (verbatim, redacted) → a chronological **action** timeline (dispatch, LLM calls, tool calls, handoffs, task transitions) → the **outcome** (status, `blocker_note`, `result_summary`) → a **cost** rollup. `--json` emits the structured object.
- `openlegion sessions --since today [--user X] [--agent X]` → one summary row per recent session (trace_id, when, who, trigger preview, outcome, cost) to find the `trace_id` worth drilling into.

On a hosted box: SSH in (per the provisioner deploy model) and run the same commands against the on-box `data/*.db`; the reader is read-only and works even when the mesh is down. The full operator/maintainer usage guide lives at `docs/internal/session-observability.md`.

## Non-goals / guardrails
- No change to the redaction/credential boundary — verbatim capture goes through `sanitize_for_prompt`/`deep_redact`.
- No new module-level globals (Constraint #8); correlation flows via the existing contextvars.
- Respect existing retention/rotation semantics; do not silently extend retention.
- Derived session grouping before any stored `session_id` (reversibility).

## Open questions
- Q1: Derived session grouping vs. stored `session_id` — start derived; revisit after Phase 3 dogfooding.
- Q2: Should verbatim-intent capture live in a new central store or extend an existing one (e.g. a `messages` table vs. annotating `tasks`)? Decide at Phase 2 design.
- Q3: How much full-prompt/response retention is acceptable vs. the current <=500-char previews, given redaction + storage cost? Decide at Phase 4.
