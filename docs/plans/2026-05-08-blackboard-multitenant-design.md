# Blackboard Multi-Tenant Scoping — Design Doc (PR-O'.2)

## TL;DR

Blackboard keys today are flat (`tasks/foo`) and project-bound enforcement is unimplemented; PR-O'.1 telemetry now counts cross-project accesses but does not gate them. This document FREEZES six implementation decisions for PR-O'.3 (the enforcement migration), so that the implementer carries no open design questions. The chosen shape — metadata column + lazy stamping + dedicated `OPENLEGION_BLACKBOARD_SCOPE_MODE` flag — preserves all existing fleet-template instructions while gating multi-tenant isolation behind a soak-window pre-condition.

## Status

**Locked design awaiting PR-O'.3 implementation.**
**Last updated:** 2026-05-08.
**Audited against:** `main` at commit `f78915e`.
**Pre-condition for advancing past Phase 1 of the rollout sequence:** PR-O'.1 telemetry (`/mesh/system/metrics → blackboard_cross_project_total`) shows zero cross-project crossings for 7 consecutive 24h windows. Operator's daily `OBSERVATIONS.md` snapshots are the source of truth for the windowing.

## Context

### What PR-O'.1 already shipped

PR-O'.1 (#853) shipped pure observability against the existing flat-key blackboard. It added:

- `_blackboard_xproject_count: dict[str, int] = {"read": 0, "write": 0}` (process-lifetime counter, `src/host/server.py:451-465`)
- `_caller_projects(agent_id)` — returns the project memberships visible to a worker, with operator/`mesh` returning the empty-set sentinel that means "fleet-global, branch elsewhere" (`src/host/server.py:507-530`)
- `_is_blackboard_cross_project(caller, writer)` — returns True only when both parties are project-bound workers in disjoint project sets; returns False for operator/`mesh`, missing writer, shared project, or unbound workers (`src/host/server.py:533-559`)
- Telemetry hooks at the four blackboard endpoints (read/put/delete/claim, `src/host/server.py:1065-1216`), gated on `not _is_internal_caller(request)`
- Surface on `/mesh/system/metrics` as `blackboard_cross_project_total` (`src/host/server.py:3488`)

The telemetry counter is NOT a denial — it does not raise 403. It is purely a soak-window measurement that quantifies whether enforcement would break real fleets in practice.

### What is NOT yet enforced

The four blackboard endpoints (`list_blackboard`, `read_blackboard`, `write_blackboard`, `delete_blackboard_entry`, `claim_blackboard`) currently let any agent with the matching glob permission read or write any key, regardless of project boundary. Project membership influences only:

- Agent rosters (`OPENLEGION_PROJECT_SCOPE_MODE`, already at `enforce`)
- The `_is_blackboard_cross_project` telemetry hook (counts but does not gate)

The `entries` table itself has NO project column today (`src/host/mesh.py:60-70`):

```sql
CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    written_by TEXT NOT NULL,   -- agent_id only; project not stored
    workflow_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    ttl INTEGER,
    version INTEGER DEFAULT 1
);
```

This means multi-tenant cross-pollution today is structural, not just policy: even if we wanted to enforce, we'd be inferring project from `written_by` at every read, which is fragile (the writer may have been removed; their project membership may have changed since the write).

### Why "do nothing" isn't an option for multi-tenant

The product roadmap places engine instances behind per-tenant subdomains (one Hetzner VPS per tenant via the provisioner). Within a single tenant, multiple **projects** are a soft isolation: distinct fleets sharing one engine instance. Today that softness is total — `tasks/foo` written by project A is readable by project B if the agent's permission glob covers `tasks/*`.

Three concrete failure modes that flip from "theoretical" to "incident" the day a tenant runs two unrelated projects on one engine:

1. **Brief leakage.** A `briefs/q3-launch` written by the marketing project is read by the research project's writer agent because both projects ship with the same content-team template and inherit the same `briefs/*` read glob. This is silent — no error, just confused output.
2. **Task hijacking.** A `tasks/research-foo` written by project A is `claim`-ed (CAS write) by an agent in project B. CAS only checks version; project boundary is not part of the contract.
3. **Audit ambiguity.** Operator views the audit log and sees `agent_x wrote tasks/launch-summary` — but `agent_x` is in two projects today, so which fleet's launch is this? Without a project stamp at write time, the question is unanswerable retroactively.

Doing nothing is acceptable only as long as one tenant runs one project. The product roadmap says we are 4-8 weeks from the first multi-project tenant.

## Goals

- Define the on-disk shape that lets the four blackboard endpoints enforce per-project scope without rewriting the 13 existing fleet templates' INSTRUCTIONS.md.
- Define the rollout sequence so operators can flip enforcement on with high confidence (telemetry-gated).
- Define the rollback path so a botched enforcement flip is recoverable in <5 minutes by reverting the flag, not by re-running migrations.
- Lock six implementation decisions hard enough that PR-O'.3 is purely execution.

## Non-goals

- **Cross-tenant isolation.** This document is about projects within a tenant. Tenants are separated at the engine-instance / VPS level (provisioner concern, not engine concern).
- **Permission glob redesign.** Globs stay flat; we are NOT introducing `proj:research/tasks/*` style globs.
- **History namespace migration.** `history/*` is fleet-global by design (audit trail) and stays fleet-global under the new column.
- **Workflow-id-based scoping.** `workflow_id` is per-task, not per-project; a task may span projects via subagents. Out of scope.
- **Per-key ACLs.** The unit of isolation is the project membership, not a per-key list.
- **Migration tooling for cross-tenant moves.** Out of scope; tenants don't share an engine instance.

## Decisions (locked)

### Decision 1: Key namespace shape — METADATA column

**Recommendation.** We will add a new column `written_by_project TEXT` to the `entries` table. Keys remain flat (`tasks/foo`, `briefs/q3-launch`, etc.). The project boundary is a metadata filter layered on top of the existing permission check at the four endpoint sites — NOT encoded in the key itself.

**Rationale.**

1. **Zero migration to agent instructions.** All 13 fleet templates' INSTRUCTIONS.md reference flat keys. Search a few:
   - `src/templates/content.yaml` — agents reference `briefs/*`, `tasks/*`
   - `src/templates/deep-research.yaml` — `research/*`, `tasks/*`
   - `src/templates/competitive-intel.yaml` — `competitors/*`, `tasks/*`

   Switching to a `proj:{name}/tasks/foo` prefix would require rewriting every `INSTRUCTIONS.md` block, every `permissions:` glob list, and every example in agent souls. Multiply by 13 templates and ~40 globs each. Invasive, error-prone, and ships behavioral risk to every existing user fleet on next template re-apply.
2. **Permission globs stay flat.** `permissions.can_read_blackboard(agent_id, key)` already takes the literal key. With prefix scoping, every glob would have to gain a per-project prefix. With metadata, the existing glob still matches the literal key; the project filter is a separate layered check downstream of the permission check.
3. **Operator stays fleet-global by identity.** With prefix, `proj:operator/...` is awkward — operator does not "own" a project, and writing to `proj:research/...` from operator code would require operator to declare a project context per call. With metadata, operator simply writes with `written_by_project=NULL` and is exempt from the read filter.
4. **Telemetry layer already proves the design.** `_is_blackboard_cross_project` already does this layered check via `_caller_projects(written_by)` lookup. Promoting that telemetry hook to an enforcement hook is structurally identical; we are just replacing "increment counter" with "raise 403 when in `enforce` mode". The proof-of-concept already lives in the codebase.

**Alternative considered.** **Key prefix approach** — every key becomes `proj:{name}/{original_key}`. Project is part of the key string itself.

**Why not.** Three concrete costs:
- Rewrites every fleet template's INSTRUCTIONS.md and permission globs (~520 lines of YAML and prose, error-prone).
- Forces the operator's bootstrap writes into one of: a synthetic `proj:operator/` namespace (operator is not a project), a `global/` carve-out (decision 2 explicitly rejects this), or a per-call project parameter (every operator-side mesh client call now needs a project arg).
- Self-documenting in `sqlite3` shell — the only point in its favor — is solvable with a `SELECT key, written_by_project FROM entries` query and is mitigated by the dashboard read pane (next decision's point).

The metadata approach loses self-documentation in raw SQL but loses no actual safety. Mitigation: include `written_by_project` in the dashboard's blackboard-view template render and in the `BlackboardEntry` Pydantic model so it surfaces everywhere a human reads keys.

### Decision 2: Fleet-global namespace — NO `global/*` carve-out

**Recommendation.** We will NOT introduce a `global/` key prefix. With the metadata column from Decision 1, "global" is implicit: any entry with `written_by_project IS NULL` is fleet-global by definition. The operator and `mesh` always write `NULL`; that is the only way to produce a `NULL`-stamped entry under enforcement.

**Rationale.**

1. **Identity, not key shape, defines scope.** Operator is fleet-global because operator is operator — not because operator writes to a magic prefix. The `_caller_projects("operator") == set()` sentinel already captures this in the telemetry layer (`src/host/server.py:517-522`).
2. **No new permission glob to maintain.** A `global/` prefix would need its own line in every template's permission list. `NULL`-stamping is invisible to globs.
3. **Mirrors the operator-or-internal permission tier.** `_require_operator_or_internal` is already the pattern for "this caller bypasses scope checks". Reusing it for blackboard scope checks (Decision 6) keeps the architectural surface area minimal.

**Alternative considered.** Reserve `global/` as a fleet-global key prefix. Operator handoffs and shared bootstrap data go there explicitly.

**Why not.** Three concrete costs:
- Doubles the permission surface — every template now needs `read: ["global/*", "tasks/*", ...]`.
- The operator can still write outside `global/*` (e.g. seeding a project's `tasks/...`), so the prefix doesn't cleanly map to "always operator". You'd need both the prefix AND identity-based bypass — pick one.
- Conflicts with the metadata model. If we keep both, every read site has to combine two checks: "is the key in `global/*`?" OR "is `written_by_project IS NULL`?". Two truths is one too many.

Identity is the canonical fleet-global signal. The key prefix is redundant.

### Decision 3: Back-compat window — 30 days WARN-MODE, gated on 7 consecutive zero-crossing days

**Recommendation.** Before flipping `OPENLEGION_BLACKBOARD_SCOPE_MODE` from `off` to `enforce`, two windowed gates must pass in order:

1. **Pre-warn gate (telemetry-only).** PR-O'.1's `blackboard_cross_project_total` counter must show **zero net accumulation across 7 consecutive 24h windows**. Operators read this from `system_metrics` snapshots in their daily `OBSERVATIONS.md` (operator already saves observations daily; this is just one more line to track). 7 days is enough to cover a normal weekly fleet rhythm.
2. **Warn-mode soak.** When the pre-warn gate passes, flip to `warn` for **30 days**. In this window, the four endpoints log every would-be denial (project-disjoint access against a stamped entry) without raising — the call still succeeds. PR description for the flip-to-`enforce` change must include the count-by-day breakdown of accumulated would-be denials.

If either gate produces a non-zero count, do NOT advance. Investigate the source (likely a forgotten cron job, a multi-project subagent, or a mistakenly-shared template), fix it, restart the gate.

**Rationale.**

1. **Two distinct signals.** Pre-warn (`xproject_count`) measures cross-project access against the ALREADY-STAMPED writer. Warn-mode logs measure denials under the new logic — they will catch new edge cases (e.g. lazy-stamped entries with unexpected project assignments) that pre-warn cannot see. Both windows are required because they answer different questions.
2. **30 days is a real soak window.** Most agent fleets have weekly cadence (cron, scheduled handoffs); 30 days covers four cycles of monthly rituals and surfaces edge cases that 7 days cannot.
3. **Operators already track daily.** No new infrastructure — the operator already saves daily observations. Adding `blackboard_cross_project_total` to that snapshot is a one-line change to the operator's heartbeat playbook (out of scope for this doc; tracked under PR-O'.3).

**Alternative considered.** A single 14-day warn-mode window (no pre-warn gate; flip straight from `off` to `warn`).

**Why not.**
- Flipping `off → warn` blindly burns the warn-mode log with the same noise the telemetry counter already shows — we'd rediscover known crossings via a different surface, which costs operator attention without adding signal.
- 14 days is too short to cover monthly cron jobs. We'd ship false-positive confidence and discover a quarterly batch job at the worst possible moment (post-`enforce` flip).
- The two-stage gate decouples "is the design correct" (pre-warn) from "is the migration correct" (warn-mode). Collapsing them confuses the rollback signal.

#### Escalation when the gate doesn't close

If 60 days elapse without 7 consecutive zero-crossing days:

1. **Engineering reviews the cross-project trace.** Use the existing
   `blackboard_cross_project_total` per-day deltas (from operator
   `OBSERVATIONS.md` snapshots) to identify which fleets are crossing.
2. **Categorize the crossings.** Are they:
   - **Legitimate.** The fleet template inherently shares state across
     projects (operator handoffs, fleet-global configs). → Restructure
     templates to use the operator-global path; or accept fleet as
     non-multi-tenant-eligible.
   - **Bug.** An agent is reading where it shouldn't. → Fix the agent
     prompt; verify the crossing stops.
   - **Acceptable noise.** Cron-driven probes, canary-probe traffic.
     → Whitelist the agent IDs from the counter (carve out a
     `_record_blackboard_xproject` skip for known-good actors).
3. **If still no closing window after intervention,** an engineering-
   approved manual override flag (`OPENLEGION_BLACKBOARD_SKIP_GATE=1`)
   may flip to `warn` mode without the 7-day requirement. The override
   is single-use, audited in commit logs, and requires a written
   justification on the deployment.

#### Automated streak tracking

To avoid manual operator counting and misalignment risk, ship the
streak counter as a system_metrics field in PR-O'.3:

- **`blackboard_xproject_clean_streak_days: int`** — number of
  consecutive 24h windows with `_blackboard_xproject_count == {read: 0,
  write: 0}` since last reset.
- Reset to 0 on any cross-project access in the current 24h window.
- Surfaced on `/mesh/system/metrics` and rendered on the dashboard
  Board's "Needs you" panel as a Phase 2 readiness banner: "Multi-tenant
  enforcement gate: 4 / 7 clean days." Operator no longer counts.

**Implementation:** the streak counter lives alongside
`_blackboard_xproject_count` in `src/host/server.py`. On each
day-rollover (existing pattern from `_record_denial`), if the
day's reads + writes were both 0, increment the streak; otherwise
reset to 0.

### Decision 4: Migration path — LAZY (not bulk re-tag)

**Recommendation.** We will NOT bulk-tag existing entries. Instead, on every write to an existing key after Phase 1 of the rollout (Decision 5):

- If the key's existing row has `written_by_project IS NULL`, stamp it with the **current writer's project** (looked up via `_caller_projects(agent_id)`).
- If `written_by_project` is already non-NULL, leave it (next-write semantics: the project of the last writer wins, identical to how `written_by` already overwrites on every write).

Reads of `NULL`-stamped keys are PERMITTED unconditionally during the back-compat window — this is the "legacy zone". After the `enforce` flip + an additional 90-day soak, a sweeper job MAY null-out (i.e. delete) any entries still at `written_by_project IS NULL` to force a re-stamp on next write. The 90-day sweeper is a follow-up PR, NOT part of PR-O'.3.

**Rationale.**

1. **Bulk re-tag is ambiguous for legacy keys.** For a key whose `written_by` is `agent_alpha`:
   - If `agent_alpha` is now in projects {A, B}, which one do we stamp? Picking arbitrarily is wrong; a migration that says "we picked one for you" undermines trust.
   - If `agent_alpha` was deleted, there is no project to look up.
   - If `agent_alpha` was in project C at write time but is now in project D, the historical project stamp would be incorrect.

   A bulk tag is forced to encode a guess. Lazy tagging encodes a fact: the project of whoever next touches the key.
2. **Bulk re-tag requires downtime or transactional risk.** With ~10K entries in a busy fleet, a single UPDATE under SQLite WAL is cheap, but the lookup-per-row to compute `_caller_projects(written_by)` is not — it walks `config/projects/*.yaml` for each row. Doing this online means the migration is interleaved with live writes, which is hard to reason about without coarse locks. Lazy avoids the entire question.
3. **Read-allow on NULL keys is the safety net.** During the warn/enforce transition, no agent loses access to anything they could read before — they just may not be able to KEEP reading newly-written keys outside their project. Existing operations continue.
4. **The 90-day sweeper is a forcing function, not a hard cutover.** Operators can choose to never run it. Some legacy entries will remain `NULL`-stamped indefinitely; that is acceptable as long as `enforce` mode treats `NULL` as "fleet-global, allow read" — which it must by Decision 2's logic.

**Alternative considered.** Bulk re-tag at migration time. For each existing entry, look up `_caller_projects(written_by)`, pick one (or pick all and store as a comma-separated list), and write the stamp.

**Why not.**
- Forces a decision the migration cannot correctly make for the ambiguous cases (multi-project writer, deleted writer, churned membership).
- Adds operational risk: the migration must run before the column is enforced anywhere, which means a downtime window or careful interleaving with live writes. SQLite ALTER TABLE is fast; bulk UPDATE with per-row lookups is not.
- Provides no advantage over lazy: under `enforce`, `NULL` is treated as fleet-global (read-allowed), so the bulk-tagged-or-not distinction is invisible to readers until they try to mutate the key, at which point lazy stamps it correctly.

#### Risk: Cold-NULL leakage

A project-private key written before migration (e.g. `briefs/q3-launch` 
written by an agent in project A) stays at `written_by_project IS NULL`
indefinitely if nobody re-writes it. Under `enforce` mode, the design
treats `NULL` as fleet-global → **the key is readable cross-project until
re-written**. For long-lived briefs / configs that nobody ever touches
again, this is a silent privacy hole.

**Mitigations** (pick one or stack):

1. **Optional partial bulk-tag at Phase 1 for unambiguous keys.**
   For each existing key whose `written_by` agent is currently in
   exactly one project, stamp `written_by_project` at migration time.
   Multi-project writers and deleted writers stay NULL. This is
   defensible — we're only making the inference where it's safe.

2. **Promote the 90-day sweeper from Phase 5 to Phase 4 (mandatory).**
   After enforce flips, all NULL-stamped keys are deleted (or marked
   for re-write) within 90 days. Operators have to keep what matters
   alive by writing to it.

3. **Accept the leakage as a feature of the back-compat window.**
   Document that legacy entries are fleet-global by design and
   require explicit re-write to scope. Suitable for tenants where
   pre-migration data is non-sensitive.

**Recommendation:** Mitigation 1 (partial bulk-tag for unambiguous keys)
combined with Mitigation 2 (mandatory sweeper). The combination
captures most pre-migration intent at Phase 1, and forces explicit
action for ambiguous cases.

### Decision 5: Flag name — `OPENLEGION_BLACKBOARD_SCOPE_MODE` with values `off | warn | enforce`, default `off`

**Recommendation.** Introduce a NEW environment variable, `OPENLEGION_BLACKBOARD_SCOPE_MODE`, with three valid values:

- `off` — schema migration applies (column added, lazy stamping active), but NO scope check is run at the four endpoint sites. Identical observable behavior to today, except writes start stamping. **Default.**
- `warn` — scope check runs; cross-project disjoint access against a stamped entry logs a structured warning to the mesh log AND increments a new counter `blackboard_scope_warn_total` on `/mesh/system/metrics`, but the call SUCCEEDS (returns 200/2xx as today).
- `enforce` — scope check runs; cross-project disjoint access raises HTTP 403 with category `scope`. `_record_denial("scope")` is called.

The flag is read once at module import (mirroring `_ORCHESTRATION_TASKS_V2` at `src/host/server.py:433-435`). Runtime flips require a mesh restart. This is intentional: enforcement changes are infrequent, deliberate, and should produce a clean log boundary.

**Rationale.**

1. **Distinct from `OPENLEGION_PROJECT_SCOPE_MODE`.** The existing flag governs **agent rosters** (which agents are visible to which callers in `/mesh/agents`-style listings) and is already at `enforce`. The blackboard scope rollout is a different blast radius, a different soak window, and a different recovery path. Conflating the two flags means a single rollback (e.g. blackboard regression) drags rosters back to `warn`, which is wrong.
2. **Three values, not two.** `off` is required because the migration ships before warn-mode is safe. `warn` is required as the gate before `enforce` (Decision 3). `enforce` is the destination state.
3. **Default `off` is the right zero-LOC-impact ship state.** PR-O'.3 lands the column, the lazy stamping, and the flag in the SAME PR — but operators see no behavior change until they flip the flag. This is how `_ORCHESTRATION_TASKS_V2` shipped (default off in early commits, default on later).

**Alternative considered.** Reuse `OPENLEGION_PROJECT_SCOPE_MODE` for blackboard scope as well.

**Why not.**
- `OPENLEGION_PROJECT_SCOPE_MODE` is at `enforce` today for rosters; introducing blackboard scope to the same flag means the moment the migration ships, blackboard scope is at `enforce` for everyone — no soak window, no warn-mode, no rollback path that doesn't also break rosters.
- Different rollout cadences need different flags. We learned this lesson with `OPENLEGION_ORCHESTRATION_TASKS_V2` shipping as a separate flag from the rest of the orchestration cutover.
- A single flag couples two rollback paths. If blackboard `enforce` produces an outage, rolling back to `warn` should NOT regress agent-roster scope. Two flags, two rollbacks.

### Decision 6: Operator's permissions — fleet-global by identity

**Recommendation.** The operator and the `mesh`-internal pseudo-id bypass blackboard scope checks unconditionally, in all three flag modes. The bypass is done at the same site as the existing `_is_internal_caller(request)` check (`src/host/server.py:1090, 1117, 1154, 1193`) and the `_caller_projects("operator") == set()` sentinel.

This means:

- Operator can read any key, regardless of `written_by_project`.
- Operator can write any key; their write stamps `written_by_project = NULL` (the fleet-global sentinel).
- `mesh`-internal writes (cron-triggered, internal bookkeeping) likewise stamp `NULL`.
- Subagents that inherit operator's identity (rare; only when operator dispatches a subagent and the subagent reuses operator's bearer) inherit the fleet-global bypass. Subagents launched from a project-bound parent do NOT — they inherit the parent's project (see Open Questions).

**Rationale.**

1. **Architectural consistency.** Operator is already fleet-global for agent rosters via `_PROJECT_SCOPE_MODE`, fleet-global for many `_require_operator_or_internal` endpoints, and fleet-global in the `_caller_projects` sentinel. Continuing the pattern for blackboard scope is the only sensible choice; introducing a per-project operator would require redesigning the operator's identity from the ground up.
2. **Operator handoffs require fleet-global writes.** When operator drops a brief into `briefs/q3-launch` for the marketing project, that write must be readable by the marketing project's writer agent. With Decision 4's lazy-stamping rule (next writer wins), the marketing writer's first write to that brief stamps it with the marketing project; until then it's `NULL`-stamped (fleet-global, readable). This is the desired flow.
3. **`mesh`-internal is loopback-only.** The `_is_internal_caller(request)` check requires the request to come from a localhost IP AND carry the `x-mesh-internal: 1` header. This is enforced at `src/host/server.py:314` and is the only way to claim `mesh` identity. We are not loosening this gate.

**Alternative considered.** Make operator and `mesh` project-scoped: each could be associated with a "synthetic" project they own (`__operator__` and `__mesh__`).

**Why not.**
- Synthetic projects are a layer of indirection with no readability benefit. Operator's writes do not "belong to" a project; they cross all projects by design.
- Synthetic projects complicate the `_caller_projects` sentinel — today the empty set means "fleet-global, branch elsewhere"; under synthetic projects, the empty set would have to mean "no projects, but you are this synthetic one", which is the wrong shape.
- The `_is_internal_caller` check is already the canonical "trusted bypass" — adding a second mechanism is redundant.

## Schema migration

### Exact ALTER TABLE

Added in `src/host/mesh.py:Blackboard._init_schema`, **after** the existing `CREATE TABLE IF NOT EXISTS entries (...)` and using the same migration pattern as the `audit_log.undoable` / `audit_log.archived` columns (`src/host/mesh.py:107-123`):

```python
existing_entry_cols = {
    row[1]
    for row in self.db.execute("PRAGMA table_info(entries)").fetchall()
}
if "written_by_project" not in existing_entry_cols:
    try:
        self.db.execute(
            "ALTER TABLE entries ADD COLUMN written_by_project TEXT"
        )
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
```

Default value is `NULL` (the SQLite default for an added column without `DEFAULT`). This is intentional: existing rows become `NULL`-stamped, which Decision 4 treats as the legacy/fleet-global zone.

### Index changes

We will add ONE composite index for the `WHERE written_by_project = ?` filter that endpoint enforcement queries will run:

```sql
CREATE INDEX IF NOT EXISTS idx_entries_project ON entries(written_by_project, key);
```

Rationale: the four endpoints all do `read(key)` (PK lookup, no index gain) followed by an in-Python project comparison. The index is for the rare admin/audit path that wants to enumerate entries by project (e.g. dashboard "show me everything project A wrote"). Composite on `(written_by_project, key)` covers both `WHERE written_by_project = ?` and `WHERE written_by_project = ? AND key LIKE ?`.

The index is small overhead today (NULLs are well-compacted) and pays off when the dashboard surfaces per-project blackboard views.

### Backfill strategy: lazy

NO bulk migration. The column is added with NULL default; existing rows stay NULL until next write. On next write to an existing key:

```python
# In Blackboard.write and Blackboard.write_if_version, after the upsert:
# (pseudo-code; actual placement after existing ON CONFLICT block)
caller_projects = _caller_projects(written_by)
if len(caller_projects) == 1:
    # Single-project worker — stamp unambiguously.
    project = next(iter(caller_projects))
    self.db.execute(
        "UPDATE entries SET written_by_project = ? "
        "WHERE key = ? AND written_by_project IS NULL",
        (project, key),
    )
elif len(caller_projects) == 0:
    # Operator, mesh, or unbound worker — leave NULL (fleet-global).
    pass
else:
    # Multi-project worker — UNDEFINED; see Open Question 3.
    # Tentative: leave NULL until OQ-3 is resolved.
    pass
```

The lazy update is gated on `written_by_project IS NULL` so it only fires for legacy rows. Subsequent writes by the same project re-stamp normally (the `INSERT ... ON CONFLICT DO UPDATE` already in `Blackboard.write` is extended to also overwrite `written_by_project = excluded.written_by_project`).

Note: the lazy stamp runs even when the flag is `off`, because the schema migration is decoupled from the enforcement flag. This is intentional — operators can flip to `warn` confident that recently-written keys are correctly stamped.

### `BlackboardEntry` Pydantic model

Add to `src/shared/types.py:BlackboardEntry`:

```python
class BlackboardEntry(BaseModel):
    key: str
    value: dict[str, Any]
    written_by: str
    written_by_project: str | None = None  # NEW; NULL = fleet-global
    workflow_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int | None = None
    version: int = 1
```

Defaulting to `None` is back-compat: existing serialization callers (channel adapters, dashboard event payloads) that don't know about the field continue to work; new readers (the dashboard's blackboard pane) can render it.

## Endpoint changes

The four blackboard endpoints in `src/host/server.py` gain a layered scope check:

- `GET /mesh/blackboard/` (list_blackboard, line 1065-1074) — list returns are filtered, not denied (see below)
- `GET /mesh/blackboard/{key:path}` (read_blackboard, line 1076-1094) — gains scope check after permission check
- `PUT /mesh/blackboard/{key:path}` (write_blackboard, line 1096-1138) — gains scope check on write to an existing project-stamped entry
- `DELETE /mesh/blackboard/{key:path}` (delete_blackboard_entry, line 1140-1164) — gains scope check on delete of an existing project-stamped entry
- `POST /mesh/blackboard/claim` (claim_blackboard, line 1177-1216) — gains scope check on CAS write

Five sites total (the doc consistently calls these "the four blackboard endpoints" because list-by-prefix is a soft surface — see below). Each promotes the existing PR-O'.1 telemetry hook from "increment counter" to "log warn or raise 403, depending on flag mode".

### Pseudocode for the layered check

The existing PR-O'.1 telemetry hook is the foundation. The new check is a strict superset:

```python
# At the top of host/server.py, mirror the existing flag pattern.
_BLACKBOARD_SCOPE_MODE = os.environ.get(
    "OPENLEGION_BLACKBOARD_SCOPE_MODE", "off"
).lower()
if _BLACKBOARD_SCOPE_MODE not in {"off", "warn", "enforce"}:
    logger.warning(
        "Invalid OPENLEGION_BLACKBOARD_SCOPE_MODE=%r; defaulting to 'off'",
        _BLACKBOARD_SCOPE_MODE,
    )
    _BLACKBOARD_SCOPE_MODE = "off"

_blackboard_scope_warn_count: int = 0


def _record_blackboard_scope_warn() -> None:
    global _blackboard_scope_warn_count
    _blackboard_scope_warn_count += 1


def _check_blackboard_scope(
    caller: str,
    key: str,
    entry: BlackboardEntry | None,
    request: Request,
    op: str,  # "read" | "write"
) -> None:
    """Layered scope check. Lifts the PR-O'.1 telemetry hook to enforcement.

    Order of operations is critical:
      1. Internal caller bypass (loopback + x-mesh-internal).
      2. Operator / mesh identity bypass (Decision 6).
      3. Missing entry (write-create) is allowed; new keys cannot be cross-project.
      4. NULL-stamped entry (legacy zone) is allowed (Decision 4).
      5. Disjoint project sets → telemetry counter, then warn-or-enforce.
    """
    if _is_internal_caller(request):
        return
    if caller in {"operator", "mesh"}:
        return
    if entry is None:
        return
    if entry.written_by_project is None:
        return
    caller_projects = _caller_projects(caller)
    if not caller_projects:
        # Unbound worker — let through (Decision 1's "no false-positive on
        # standalone agents" rule from _is_blackboard_cross_project).
        return
    if entry.written_by_project in caller_projects:
        return
    # Cross-project access against a stamped entry.
    _record_blackboard_xproject(op)  # PR-O'.1 counter, retained
    if _BLACKBOARD_SCOPE_MODE == "warn":
        _record_blackboard_scope_warn()
        logger.warning(
            "blackboard scope warn: caller=%s op=%s key=%s "
            "caller_projects=%s entry_project=%s",
            caller, op, key, sorted(caller_projects),
            entry.written_by_project,
        )
        return
    if _BLACKBOARD_SCOPE_MODE == "enforce":
        _record_denial("scope")
        raise HTTPException(
            403,
            f"Agent {caller} cannot access {key} "
            f"(project boundary)",
        )
    # mode == "off": no-op (telemetry already recorded above)
```

### Per-endpoint integration

```python
@app.get("/mesh/blackboard/{key:path}")
async def read_blackboard(key, agent_id, request):
    agent_id = _resolve_agent_id(agent_id, request)
    await _check_rate_limit("blackboard_read", agent_id)
    if not permissions.can_read_blackboard(agent_id, key):
        _record_denial("permission")
        raise HTTPException(403, f"Agent {agent_id} cannot read {key}")
    entry = blackboard.read(key)
    if not entry:
        raise HTTPException(404, f"Key not found: {key}")
    _check_blackboard_scope(agent_id, key, entry, request, op="read")  # NEW
    return entry.model_dump(mode="json")
```

PUT and DELETE: read existing entry first (already done for telemetry at lines 1118, 1155), pass it to `_check_blackboard_scope` BEFORE the mutation. CAS: same pattern at line 1194.

### `list_blackboard` is filtered, not denied

The list endpoint is special. A `LIST briefs/*` from a project-A agent that returns a project-B entry should NOT 403 the whole call (that breaks every existing template). Instead:

- In `enforce` mode, `Blackboard.list_by_prefix` filters out entries with `written_by_project NOT IN caller_projects AND written_by_project IS NOT NULL`.
- In `warn` mode, `list_by_prefix` returns everything (current behavior); `_record_blackboard_scope_warn()` fires per cross-project entry that would have been filtered.
- In `off` mode, `list_by_prefix` returns everything; PR-O'.1 telemetry already fires.

The filter is implemented as a parameter to `list_by_prefix(prefix, allowed_projects: frozenset | None)`, where `None` means "no filter, fleet-global caller". The endpoint computes `allowed_projects` from `_caller_projects(agent_id) | {None}` (callers always see their own project AND NULL-stamped legacy/global entries).

### `system_metrics` additions

Surface the new counter alongside the PR-O'.1 counter:

```python
"blackboard_cross_project_total": dict(_blackboard_xproject_count),  # existing
"blackboard_scope_mode": _BLACKBOARD_SCOPE_MODE,                     # NEW
"blackboard_scope_warn_total": _blackboard_scope_warn_count,         # NEW
```

This mirrors `scope_warn_total` / `project_scope_mode` (`src/host/server.py:3480-3481`).

## Flag rollout sequence

Five phases. Each phase has an entry condition, an exit condition, and a rollback path.

### Phase 0: ship telemetry (DONE)

- **Status:** complete (PR #853, PR-O'.1).
- **Behavior:** `_blackboard_xproject_count` ticks on cross-project access. No behavior change.
- **Exit condition:** counter is observable on `/mesh/system/metrics`.

### Phase 1: ship migration + flag at `off` (PR-O'.3 first commit)

- **Entry condition:** Phase 0 complete.
- **Behavior:**
  - `entries.written_by_project` column added (lazy ALTER TABLE).
  - `idx_entries_project` index added.
  - `Blackboard.write` / `write_if_version` start stamping `written_by_project` for single-project writers.
  - `OPENLEGION_BLACKBOARD_SCOPE_MODE=off` (default) — no scope check at endpoints.
  - `BlackboardEntry.written_by_project` field present and serialized.
- **Exit condition:** PR-O'.3 lands, all tests green, observable in dashboard read pane.
- **Rollback path:** revert PR-O'.3. Schema column persists (harmless; SQLite is forgiving). Code paths revert.

### Phase 2: gated on 7 consecutive zero-crossing days; flip to `warn`

- **Entry condition:**
  - PR-O'.1 counter shows zero accumulation for 7 consecutive 24h windows. Operator's `OBSERVATIONS.md` carries the daily snapshot.
  - At least 14 days have elapsed since Phase 1 to give lazy stamping time to fill in active keys.
- **Behavior:**
  - Operator sets `OPENLEGION_BLACKBOARD_SCOPE_MODE=warn` in the engine's `.env`.
  - Mesh restart picks up the flag.
  - `_check_blackboard_scope` now logs would-be denials AND increments `blackboard_scope_warn_total`. Calls still succeed.
  - List filtering is NOT applied (pure log, no behavior change to list returns).
- **Exit condition:** 30 consecutive days with the warn-counter increment-rate trending to zero. PR description for the `enforce` flip carries the per-day breakdown.
- **Rollback path:** flip flag back to `off`. Mesh restart. Schema unchanged.

### Phase 3: 30-day warn-mode soak

- **Entry condition:** Phase 2 active.
- **Behavior:** continuous monitoring of `blackboard_scope_warn_total`. Operator escalates any non-zero day to engineering for triage (likely a forgotten cron job or a multi-project agent's specific blackboard pattern that should be added to a global glob).
- **Exit condition:** 30 days with monotone-decreasing-or-zero warn count, PR description ready with breakdown.
- **Rollback path:** flip to `off`. Mesh restart.

### Phase 4: flip to `enforce`

- **Entry condition:** Phase 3 exit.
- **Behavior:**
  - Operator sets `OPENLEGION_BLACKBOARD_SCOPE_MODE=enforce`. Mesh restart.
  - Cross-project read/write/delete/claim returns HTTP 403 with category `scope`. `_record_denial("scope")` ticks.
  - List filtering active: cross-project entries silently dropped from `list_by_prefix` returns.
- **Rollback path (FAST, <5 min):** flip flag to `warn`. Mesh restart. No schema change, no data migration. Logged 403s become logged warnings.
- **Rollback path (SLOWER):** flip to `off`. Same restart, fully reversible.

### Phase 5 (follow-up PR, not part of PR-O'.3): legacy sweeper

- **Entry condition:** Phase 4 stable for 90 days.
- **Behavior:** optional sweeper deletes `entries WHERE written_by_project IS NULL AND key NOT LIKE 'history/%'`. Forces re-stamp on next write.
- **Note:** this is a CHOICE, not a requirement. Some legacy entries may stay NULL-stamped indefinitely; under Decision 6, that means fleet-global, which is fine.

## Risk + rollback

### What breaks if we get this wrong

1. **False-positive 403s on production fleets.** A legitimate cross-project access (e.g. a research project's analyst reading the marketing project's brief because the user wired it that way explicitly) becomes a denial. Surfaces as agents stuck mid-task with `403 scope` errors in their tool result.
2. **Silent data loss in list filtering.** A `LIST briefs/*` returns fewer rows than expected because some are filtered. Hard to debug — the agent's prompt sees a shorter list and continues, with no error.
3. **Lazy stamping mis-attribution.** A multi-project worker writes an entry; we leave `written_by_project = NULL` (per Decision 4's tentative path for OQ-3). Under `enforce`, the entry stays fleet-global, which may be MORE permissive than intended — a project-specific brief tagged as fleet-global is readable by every project on the engine.
4. **Telemetry blind spot during Phase 1.** Lazy stamping is racing the warn-mode flip. If we flip too fast, recently-written entries are stamped but old keys are still NULL, and the warn-counter undercounts. Mitigation: 14-day Phase 1 minimum (above).

### Recovery steps

| Failure | Recovery | RTO |
|---|---|---|
| False-positive 403 storm under `enforce` | Flip flag to `warn` via `.env` edit + mesh restart | <5 min |
| Silent list filter dropping rows | Flip flag to `warn` (filter only active in `enforce`) | <5 min |
| Mis-stamped entries discovered post-flip | Operator (fleet-global) overwrites the affected key, lazy stamping re-applies; or operator runs a one-off SQL `UPDATE entries SET written_by_project = NULL WHERE key = ?` | <5 min per key |
| Schema corruption (column add failed silently on legacy DB) | `PRAGMA table_info(entries)` confirms column; if missing, revert to PR-O'.3 base commit, redeploy | 15-30 min |
| Telemetry shows persistent non-zero warn count | Halt Phase 3 advance; engineering triage the source (cron, subagent, multi-project agent) | days |

The dominant rollback path is "flip the flag and restart" — sub-5-minute recovery for any enforcement issue. The schema column is safe to keep across rollbacks (NULL default; readers tolerate it).

## Open questions for review

These are NOT design decisions — the design above is locked. These are implementation specifics that PR-O'.3's author should either resolve in-PR or surface for a quick sync, but they do NOT block the doc landing.

### OQ-1: Timezone for "day" boundary on telemetry windowing

The 7-consecutive-zero-day gate (Decision 3) needs a day definition. PR-K' already uses `int(time.time() // 86400)` for the `_denial_counter` reset (`src/host/server.py:500`), which is UTC-day. Operator's `OBSERVATIONS.md` snapshots are saved per the operator's heartbeat schedule (currently `hourly` per Decision 3 in PR-L'). For consistency with the existing denial counter, **recommend UTC-day**. Open because some operators in the field may want local-time alignment for human-readable reports — TBD with operator voice during PR-O'.3 review.

### OQ-2: Block writes that would create new cross-project entries before the migration column lands

During the deploy of PR-O'.3 itself, there's a brief window where the code is half-applied: some mesh processes have the column, others don't. Should we hold the deploy in a maintenance window (no writes), or accept that during the deploy window we may have inconsistent stamps? **Recommend accepting the inconsistency** — the lazy-stamping rule (next-writer wins) self-heals on next write, and Phase 1 is `off`-mode so no enforcement is at stake. Open because the engine is single-process today, but provisioner-side rolling deploys (per the SSH-based deploy in `provisioner/app/services/ssh.py:run_update`) may make this real later.

### OQ-3: Subagents and multi-project workers — do they inherit the parent's project scope?

Decision 6 hand-waves: "Subagents launched from a project-bound parent inherit the parent's project." But the subagent infrastructure (`src/agent/builtins/subagent_tool.py`) creates an in-process subagent with a synthetic agent_id; that agent_id is NOT in any project's `members` list, so `_caller_projects(subagent_id)` returns `set()`, which under the current `_is_blackboard_cross_project` rule means "unbound worker → don't count". Two options for PR-O'.3:

- **Option A (recommended):** subagents inherit the parent's project context via a thread-local set during subagent execution. `_caller_projects(subagent_id)` is overridden to return the parent's projects.
- **Option B:** subagents are "project-less" and their writes stamp `NULL`. Subagents that need to write project-scoped keys must use the parent's mesh client identity (already the case for many tools).

Option A is cleaner (subagents are an extension of the parent, not their own actor). Option B is simpler (no contextvar plumbing). Recommend resolving in PR-O'.3 with whichever the implementer finds least invasive.

Multi-project workers (an agent in projects {A, B}) is a related case. Under Decision 4's lazy stamping, we leave them at `NULL`, which is wrong — a multi-project agent's writes ARE project-bound. PR-O'.3 should pick: (a) reject the write with a clearer error, (b) require the caller to declare an explicit project on the write, or (c) stamp with the alphabetically-first project as a deterministic fallback. Recommend (b) for new writes (force the agent to be explicit) once the stamping logic is in place; for the lazy backfill, leave NULL and accept the under-classification.

## References

### Code references (verified against `main` at `f78915e`)

- `src/host/server.py:451-465` — `_blackboard_xproject_count` and `_record_blackboard_xproject` (PR-O'.1 telemetry counter).
- `src/host/server.py:468-505` — `_DENIAL_CATEGORIES` and `_record_denial` (the parallel pattern; the new `scope` denial reuses this).
- `src/host/server.py:507-530` — `_caller_projects` (operator/mesh empty-set sentinel; lookup of worker memberships from `config/projects/*.yaml`).
- `src/host/server.py:533-559` — `_is_blackboard_cross_project` (the telemetry hook this design lifts to enforcement).
- `src/host/server.py:438-448` — `_scope_warn_count` (the parallel pattern for the agent-roster scope flag, mirrored by the new blackboard scope warn counter).
- `src/host/server.py:1065-1216` — the four (five-call-site) blackboard endpoints: `list_blackboard`, `read_blackboard`, `write_blackboard`, `delete_blackboard_entry`, `claim_blackboard`. Lines 1090, 1117-1122, 1154-1159, 1193-1198 are the existing PR-O'.1 telemetry hooks.
- `src/host/server.py:3480-3496` — `system_metrics` exposure of `_scope_warn_count`, `_PROJECT_SCOPE_MODE`, `blackboard_cross_project_total`, `tool_denials_24h`. The new `blackboard_scope_mode` and `blackboard_scope_warn_total` fields slot in here.
- `src/host/server.py:314` — `_is_internal_caller` (loopback + `x-mesh-internal: 1`).
- `src/host/mesh.py:60-70` — `entries` table schema (target of the `ALTER TABLE ADD COLUMN written_by_project`).
- `src/host/mesh.py:107-138` — existing column-add migration pattern (`audit_log.undoable` / `audit_log.archived`); reuse for the new column.
- `src/host/mesh.py:150-244` — `Blackboard.write` and `Blackboard.write_if_version` (lazy stamping insertion sites).
- `src/host/mesh.py:266-288` — `Blackboard.list_by_prefix` (target of the optional `allowed_projects` filter).
- `src/shared/types.py:259-269` — `BlackboardEntry` Pydantic model (target of the new `written_by_project` field).
- `src/host/server.py:433-435` — `_ORCHESTRATION_TASKS_V2` flag pattern (model for `_BLACKBOARD_SCOPE_MODE`).

### Roadmap references

- `docs/plans/2026-05-08-post-board-roadmap.md` — parent roadmap; §"Phase 3 — Multi-tenant blackboard scoping" defines the three slices. This doc IS slice 2 (PR-O'.2).

### Prior art (commit-level)

- PR #853 (PR-O'.1) — telemetry counter + helpers. Commit landed onto `main` ahead of this design doc; structurally identical to the enforcement promotion path.
- PR #851 (PR-F) and PR #850 (PR-G) — recent Board-redesign PRs that touched dashboard rendering of blackboard entries; relevant when surfacing `written_by_project` in the dashboard read pane (out of scope for PR-O'.3, in scope for the dashboard polish that follows).
- The `OPENLEGION_PROJECT_SCOPE_MODE` flag and `_scope_warn_count` (no specific PR, predates this roadmap) — direct architectural precedent for the new flag and counter.

### Reviewers

- Engineer authoring PR-O'.3 (the implementer)
- Multi-tenant SaaS lead (rollout cadence)
- Security review (the `enforce` flip is a privilege boundary tightening; same review burden as a new permission rule)
- Operator-experience lead (the soak-window gate puts daily reading work on the operator)

The design is FROZEN. Reviewers may push back on the open questions (timezone, deploy-window write blocking, subagent inheritance) without unblocking the doc — those are PR-O'.3 implementation specifics.
