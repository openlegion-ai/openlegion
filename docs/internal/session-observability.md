# Session Observability

> **Status:** partly built. **Phase 1** (trace_id correlation) shipped in PR #1149.
> Everything else on this page is the documented *design* — see the
> [Availability / phase status](#availability--phase-status) table for what works
> today. Design doc: [`../plans/2026-06-18-session-observability.md`](../plans/2026-06-18-session-observability.md) (plan #1147).

## What it is & why

Session Observability lets a maintainer reconstruct any past **human-rooted**
interaction end to end:

**intent** (what the user actually asked) → **action** (the tool calls,
handoffs, and task transitions the fleet took) → **outcome** (result, failure
reason, status) → **cost** (per-call token usage rolled up).

Primary uses:

- **Debugging a live or hosted fleet** — "the operator said it was done but
  nothing happened" / "this task failed and I can't tell why." Pull the whole
  session and read the timeline instead of grepping logs across containers.
- **Audit** — who started what, when, through which channel, and what the fleet
  did in response.
- **Per-session cost attribution** — total token spend for one conversation,
  not just a daily aggregate.
- **The improvement loop** — *use it normally → reconstruct what happened →
  fix the gap.* The reconstruction is the feedback signal.

## Concepts

### `trace_id` — the correlation atom

A `trace_id` is minted once per **turn** (one human message and the fleet work
it triggers). Format: `tr_<hex12>` (e.g. `tr_9f3a1c0b7e21`). Every record that
belongs to that turn — trace events, tasks, usage rows — carries the same
`trace_id`, so it is the single key you join on across stores.

### `MessageOrigin` — who/what started it

Each turn has a `MessageOrigin` (see `src/shared/types.py`) describing the
source: `kind` (e.g. human vs. system), `channel` (cli / telegram / discord /
slack / whatsapp / webhook / dashboard), and `user`. Session Observability is
scoped to human-rooted origins.

### Session = conversation = grouping of turns

A "session" is a conversation — i.e. a group of turns that belong together.
**It is derived today, not a stored id**: turns are grouped by origin
(user + channel) and adjacency, not by a persisted `session_id`. In practice you
investigate by `trace_id` (one turn) and the reader stitches the surrounding
conversation around it.

### Which store holds what

| Store | Holds | Keyed / filtered by |
|---|---|---|
| `data/traces.db` | trace events (the action timeline) | `trace_id` |
| `data/tasks.db` | tasks: `status`, `blocker_note`, `result_summary`, the DAG, `origin_user` | `trace_id` (Phase 1 column), task id |
| `data/costs.db` | per-call token usage / cost | `trace_id` (Phase 1 column on `usage`) |
| `data/intent.db` | verbatim user intent | `trace_id` (**Phase 2 — planned**) |

> **Transcripts are container-local.** Full agent transcripts live inside each
> agent container, not on the host. The host-side reconstruction is built from
> traces + tasks + costs + intent — enough to see intent → action → outcome →
> cost, but not the agent's full internal transcript.

## Using the reader

> **Phase 3 — planned.** The `openlegion session` / `openlegion sessions`
> commands below are the documented design. They are **not available today**;
> until Phase 3 ships, reconstruct sessions with raw SQL (see
> [Finding the data manually](#finding-the-data-manually)).

### One session, full timeline — `openlegion session <trace_id>`

```bash
openlegion session tr_9f3a1c0b7e21
```

Sketch of the output shape:

```
SESSION tr_9f3a1c0b7e21
  origin   human · telegram · user=alice
  started  2026-06-18 14:02:11Z

INTENT
  "pull last week's signups and email me a CSV"

ACTION  (chronological)
  14:02:11  operator        → received message
  14:02:13  operator        tool: hand_off(agent=analyst, task=t_4412)
  14:02:13  task t_4412     created (analyst)            status=queued
  14:02:40  analyst         tool: run_command("psql ... > signups.csv")
  14:03:02  analyst         tool: file.read(signups.csv)
  14:03:05  task t_4412     status=done

OUTCOME
  task t_4412   status=done
  result        "Exported 1,284 signups to signups.csv"
  blocker_note  —

COST
  4 calls · 38,210 in / 2,940 out tokens · $0.21
```

When a task fails, the outcome block surfaces the failure:

```
OUTCOME
  task t_4412   status=failed
  blocker_note  "psql: connection refused (DB unreachable)"
  result        —
```

### Recent sessions — `openlegion sessions --since <when> [filters]`

```bash
openlegion sessions --since 24h
openlegion sessions --since 2026-06-17 --user alice
openlegion sessions --since 7d --agent analyst
openlegion sessions --since 24h --json        # machine-readable
```

Output is a list of recent sessions (one row per session) with its `trace_id`,
origin, start time, final status, and rolled-up cost — use it to find the
`trace_id` you then drill into with `openlegion session`.

## Debugging a hosted VPS

This is the workflow it exists for. Per the provisioner deploy model, each client
runs on its own box.

1. **SSH into the client box.** (Same access path the provisioner uses for
   `git pull` + `systemctl restart openlegion`.)
2. **Run the reader against the on-box `data/*.db`.** Point it at the live data
   directory on that box.

   ```bash
   openlegion session tr_9f3a1c0b7e21    # (Phase 3 — planned)
   ```

The reader opens the SQLite stores **read-only**, so it **works even when the
mesh is down** — which is exactly when forensics matter most. You do not need to
restart, and you cannot make things worse by reading.

> Until the Phase 3 reader ships, do the same investigation on-box with the raw
> SQL below — `sqlite3` is also read-only when you only `SELECT`.

### Getting a `trace_id` to investigate

- **From the dashboard** — a turn's `trace_id` is surfaced in the UI (the
  dashboard mints/propagates it correctly as of Phase 1).
- **From `sessions --since`** (Phase 3) — list recent sessions and copy the id.
- **From raw SQL** — query `tasks` / `usage` for the user/time window and read
  the `trace_id` column (below).

## Finding the data manually

For when you want raw SQL instead of (or before) the reader. The DB files live
under `data/` on the host:

- `data/traces.db` — trace events, keyed by `trace_id`.
- `data/tasks.db` — `tasks` table: filter `WHERE trace_id = ?`, then read
  `status`, `blocker_note`, `result_summary` (and the DAG / `origin_user`).
- `data/costs.db` — `usage` table: filter `WHERE trace_id = ?` to roll up cost.
- `data/intent.db` — verbatim intent (**Phase 2 — planned**).

Example — outcome + cost for one turn:

```bash
sqlite3 data/tasks.db \
  "SELECT id, status, blocker_note, result_summary
     FROM tasks
    WHERE trace_id = 'tr_9f3a1c0b7e21';"

sqlite3 data/costs.db \
  "SELECT count(*) AS calls,
          sum(input_tokens)  AS tok_in,
          sum(output_tokens) AS tok_out
     FROM usage
    WHERE trace_id = 'tr_9f3a1c0b7e21';"
```

(Open with `sqlite3 -readonly` if you want to be certain you can't mutate the
live DB.)

## Limitations

- **Transcripts are container-local.** The host timeline is reconstructed from
  traces + tasks + costs + intent, not from full agent transcripts (which never
  leave the container).
- **Retention windows.** Tasks and intent are kept ~90 days; traces ~7 days. A
  session older than the trace window will have intent/outcome but a thinned
  action timeline.
- **Session id is derived, not stored** — sessions are grouped from turns by
  origin + adjacency, not a persisted `session_id`.
- **`trace_id` minting** was inconsistent on the dashboard path before Phase 1;
  that is fixed as of #1149. Turns from before that fix may not correlate
  cleanly.

## Availability / phase status

| Capability | Phase | Status |
|---|---|---|
| `trace_id` correlation across tasks + usage (`trace_id` columns) | Phase 1 | **Shipped — PR #1149** |
| Verbatim intent capture (`data/intent.db`) | Phase 2 | Planned |
| `openlegion session` / `openlegion sessions` reader CLI | Phase 3 | Planned |
| Agent-side tool/handoff traces + log correlation | Phase 4 | Planned |

> Commands marked **(Phase 3 — planned)** above do **not** work today. Until
> they ship, use the [raw SQL](#finding-the-data-manually) path. Phase 1 is the
> only shipped piece — it makes the correlation key (`trace_id`) present in the
> data so the later phases can read it.

---

See also: design doc
[`../plans/2026-06-18-session-observability.md`](../plans/2026-06-18-session-observability.md).
