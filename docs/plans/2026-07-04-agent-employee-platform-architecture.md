# Agent-Employee Platform — Target Architecture & Implementation Plan

**Date:** 2026-07-04
**Status:** Proposed (design brainstorm → implementation blueprint)
**Author:** Principal-engineering review (Claude Code)
**Supersedes coordination model in:** `docs/architecture.md` (fleet/blackboard sections), the
per-agent lane/heartbeat design, and the project→team migration shims.

> **Mandate for this rewrite.** There are **no users and no production data.** We therefore
> carry **zero backwards compatibility**. Every migration path, rename shim, and lazy
> `ADD COLUMN` is debt to **delete**, not compatibility to preserve. The target is a clean,
> well-architected codebase with **no legacy or unused code**. Where this doc says "remove," it
> means delete the code and its tests, not deprecate.

---

## 1. North Star

An organization **hires a team of agent employees**. They collaborate on a shared body of work,
are **governed** (identity, money, trust, audit) by a central mesh kernel, **earn autonomy**
through a track record so the human reviews *by exception* rather than per-task, and **hibernate**
when idle so a small VPS hosts a large org. As models strengthen, we **loosen policy — never
rebuild structure.**

Two design goals, in priority order:
1. **Most effective agentic team** — output quality and teamwork come first.
2. **Top-notch security** — held firm at the boundaries that protect the customer's secrets and
   cross-team blast radius; **deliberately and namedly relaxed** *within* a team where it buys
   significant teamwork gains (§4).

---

## 2. The design principle that prevents rebuilding

Every element of the platform is exactly one of three layers. **Rebuild risk comes from putting a
thing in the wrong layer.**

| Layer | What it is | Rule |
|---|---|---|
| **1. Durable mechanism** | True regardless of model strength — would be true for human employees too: identity, secret custody, mediated comms, audit, economics, permissions, human oversight. | Get right now. Expensive to change later. |
| **2. Disposable policy** | Compensation for *current* model weakness: iteration bounds, loop detectors, heartbeat suppression, coordination-verb shapes, prompt scaffolding, templates. | Must be swappable without touching layers 1 or 3. **Never weld a model-weakness assumption into a durable layer.** |
| **3. Accumulated state** | Memory DBs, workspaces, task history, audit logs, track records. | The real lock-in. Code is rewritten in a weekend; two years of agent memory in the wrong schema is not. Maximum paranoia on formats. |

This is the Bitter Lesson applied to platform design: the platform's job shifts, as models
improve, from **choreographing** agents to **governing** them. Governance *is* the durable layer.
The platform must not be the ceiling on the agents' intelligence.

**Corollary — team-scope is durable; orchestration is disposable.** Orchestration (who hands off
to whom, how work is sequenced) is choreography that improves for free as models improve — and
increasingly gets done *by the agents themselves*. Scoping (trust, budget, knowledge boundaries)
is structure that cannot be retrofitted once state accumulates in the wrong walls. **We make
`team` the first-class structural unit precisely so orchestration is allowed to be mediocre and
iterate safely behind good walls.**

---

## 3. Target architecture

```
Org  (one customer, one VPS, single-tenant)
 │
 ├─ Governance kernel  (the mesh — our "OS")
 │    identity · vault/secret custody · mediation + audit · economics
 │    (team envelope → agent budget) · autonomy policy engine · oversight UI
 │
 ├─ Team  (FIRST-CLASS: trust boundary · budget envelope · drive · threads · lead · review unit)
 │    ├─ Agent = Personnel File (durable, portable)  +  Container (disposable body, hibernates)
 │    │         + Track Record (accumulated outcomes)
 │    ├─ Team Drive (git, mesh-hosted, reviewed source-of-truth)
 │    ├─ Team Scratch (shared rw volume, live collaboration — deliberate tradeoff §4)
 │    └─ Team Threads + Team Memory (decisions, handover docs)
 │
 └─ Human  (chief-of-staff operator · Team Room dashboard · exception-based review)
```

- **Container = disposable body; Personnel File = the durable person; mesh = law + payroll; team =
  the unit of trust and work; policy = the only thing that changes as models get smarter.**
- **Solo agent = team-of-one.** One code path for everything; no separate "solo" branch.

---

## 4. Security tradeoff ledger (explicit)

Per the mandate, security is held firm where it protects secrets and cross-team blast radius, and
deliberately relaxed *within* a team where output justifies it.

### Held firm — non-negotiable (relaxing buys no teamwork, only leaks)
- **Host ↔ agent container wall.** Agents run arbitrary code by design; the container is the *only*
  enforced boundary between an injected agent and the mesh vault/wallet/other volumes. Untouched.
- **Vault custody** — agents never hold keys; all provider/API calls proxy through the mesh.
- **Cross-team isolation** — hard wall. One team's compromise never reaches another team's drive,
  threads, memory, or budget.
- **Tenancy** — single-tenant per VPS. Unchanged.
- **Audit** — every consequential action is journaled and attributable.

### Deliberately relaxed — *intra-team only*, in exchange for teamwork
- **Shared `/team/scratch` volume (marquee tradeoff).** Real-time shared filesystem for teammates,
  like coworkers sharing a drive. Cost: cross-agent content is *not* mesh-mediated on that volume,
  so an injected teammate can plant files others read/run. Mitigations that keep this bounded:
  (a) **intra-team only** — never crosses the team wall; (b) **excluded for any agent holding
  wallet or credential grants** — those agents get git-only, no shared scratch; (c) content pulled
  from scratch into an LLM context is **provenance-tagged "teammate" and sanitized** like channel
  input; (d) it is **scratch, not source-of-truth** — durable deliverables live in the git Team
  Drive where they are reviewed before integration. Net: we accept "one team = one blast radius,"
  which is exactly how a human team with a shared drive already works.
- **Real-time teammate messaging** (ask verb, threads) enters context as semi-trusted (tagged +
  sanitized) rather than being forced through a task handoff. Minimal security cost, large
  teamwork gain.

### Closed regardless (cheap, doesn't cost output)
- **Agent FastAPI server gets real bearer auth** (currently checks only a forgeable
  `x-mesh-internal` header — audit C1 second leg). The mesh already mints a per-agent token; the
  agent server simply must *verify* it. Closes host-network-mode exposure. Small, clean.

---

## 5. Piece-by-piece verdict (what we keep, refactor, remove)

### Keep — already correct for the long game (do not "improve")
- **Memory stores raw text canonically; embeddings are disposable derived data** — provider/dim
  swap rebuilds only vector tables, preserving facts (`src/agent/memory.py`). The single most
  important accumulated-state decision, already right.
- **Bounds are policy, not constants** — `src/shared/limits.py` resolves per-agent → env → default
  with clamps. Correct layer separation.
- **Budget enforcement pre-flight at the LLM proxy** (`src/host/credentials.py`) — our economics
  chokepoint. Everything in the autonomy roadmap hangs off this.
- **Containers, mesh mediation, task state machine, SQLite-WAL-only storage** — durable-layer,
  sound. No Redis/broker/microservices — the single mediator process is our kernel and should stay
  singular.
- **Two runtime backends behind `RuntimeBackend` ABC** (`DockerBackend` / `SandboxBackend` microVM)
  — the isolation-upgrade path, cleanly abstracted. Keep both.

### Refactor
- **`team` from bolt-on → first-class entity.** Today it is a blackboard key prefix + read-only
  `TEAM.md` mount + `list_agents` filter. Becomes a real store with budget envelope, drive,
  threads, policy, and lead.
- **Team budgets** (`src/host/costs.py`): today **dead code** — `set_team_budget` has zero callers
  and `_team_budgets` is an in-memory dict, so `get_team_spend` always returns "no budget"
  (Appendix A.C1). Replace with a **durable, pre-flight enforced team envelope** (mirror the
  persisted per-agent `agent_budgets.json` pattern); the lead allocates per-agent budgets within it.
- **Lanes** (`src/host/lanes.py`): in-memory queues that strand work on restart → **rehydrate from
  the durable `tasks` table on boot**; add a **second lane** per agent (deep-work + interactive) so
  a 4h task no longer head-of-line-blocks a 30-second question.
- **One model per agent → per-call model tiering.** The proxy + failover already accept a per-call
  `model` override; add caller-side policy so coordination traffic (agenda ticks, thread/standup
  digests, summaries) runs on cheap models and deep work on strong ones. Makes the "employees talk"
  layer affordable. **Not purely caller-side (Appendix A.C4):** `_enforce_model_pin`
  (`server.py:1326`) 403s any model outside the agent's `allowed_models`, and heartbeat/summary
  model choice happens *inside* the container (`default_model`) — enabling tiering also requires
  widening the pin and threading `model=` through the agent-side call sites.
- **Agent identity → unified Personnel File.** Today identity is smeared across `agents.yaml` +
  `permissions.json` + Docker volume + cron table with **no export/import**. Collapse into one
  versioned, portable bundle (config, permissions, cron, workspace, memory, learnings, track
  record) with export/import — enables DR, host migration, cloning a great agent, and clean
  offboarding-with-handover.
- **Heartbeat → agenda loop.** Replace the cost-driven suppression posture (LLM-skip on idle,
  "respond HEARTBEAT_OK," "do NOT increase your own frequency") with a **budget-governed workday**:
  the agent reviews its plate (tasks, inbox, mentions, goals) and *may create work toward its
  goals*. Deterministic probes stay (good cheap pre-checks); the *posture* inverts from "wake only
  if forced" to "work your agenda until done or budget-capped." Budget is the governor.
- **Blackboard demoted** from primary coordination substrate → **small ephemeral signals only**
  (status, claims, pointers). Durable collaborative work moves to the Team Drive.

### Remove (delete code + tests — no users, no compat)
- **All project→team shims** (pervasive — `src/shared/types.py`, `mesh_client.py`,
  `operator_tools.py`, `coordination_tool.py`, `cli/*`, `orchestration.py`, `mesh.py`,
  **`permissions.py`**, and the **dashboard SPA** `static/js/app.js` + `templates/index.html`):
  `can_manage_projects` validator, dual `tasks.project_id` dict-key emission,
  `target_kind="project"`, blackboard `projects/{name}/` prefix, the internal `_*_project` CLI
  helpers + `project_id` kwargs, `config/projects` symlink. Rename to **team** natively; one name,
  one namespace. **Corrections (Appendix A.B1):** there are **no agent-facing `*_project` *tools***
  (operator tools are already `create_team`/`add_agents_to_team`); the real work is renaming ~7
  internal `_*_project` helpers in `cli/config.py` + ~25 call sites. The `/mesh/projects/*` HTTP
  aliases are **already removed**. The `project_id` dashboard-event key and `projects/` blackboard
  prefix are **load-bearing for the current SPA and live blackboard data** — renaming needs
  coordinated frontend edits and (for any existing blackboard rows) a re-key, not a code-only drop.
- **`src/host/team_migration.py`** (PROJECT.md→TEAM.md + symlink startup migrator) — delete.
- **`src/host/orchestration_migration.py`** (legacy blackboard-task → tasks-table migrator) —
  delete.
- **All lazy schema-migration archaeology.** `orchestration.py` `PRAGMA table_info` + `ALTER`
  chains and `memory.py` idempotent `ADD COLUMN` paths collapse into **clean canonical `CREATE
  TABLE` at `schema_version = 1`**. Adopt an explicit `PRAGMA user_version` going forward.
  **Two hazards (Appendix A.B4):** (a) `memory.py:_reconcile_embedding_dim` is NOT archaeology —
  it is live embedding-provider-switch logic and must survive; and the collapse must explicitly add
  `category_id` to the `facts` CREATE TABLE (today it exists only via `ALTER`). (b) Deleting
  `team_migration.py` while hard-setting `orchestration._team_col = "team_id"` breaks any DB still
  on the `project_id` column — under clean-slate (no data) this is fine, but the canonical schema
  must be the *only* path (no dual-column detection left behind).
- **Legacy shared-browser display `:99` / port `:6080`** — **already dead code** (executable path
  deleted in a prior PR; `display_allocator.py:37`). Remaining work is comment/docstring/test-prose
  cleanup only (Appendix A.B5). Downgraded from "remove surface" to "delete stale prose."
- **Heartbeat-suppression scaffolding** — the `force_llm` / `is_default_heartbeat` skip logic and
  the "HEARTBEAT_OK / don't increase frequency" prompt copy, once the agenda loop lands.
- **Blackboard pushback-reissue pattern** — the "recipient blocks → creator re-hand_offs a new
  task" dance in `coordination_tool.py`, once the ask verb + threads exist.
- **`pending_actions` (narrow delete-confirmation store)** — absorbed into the general **action-tier
  policy engine** (§6, Phase 5), then removed.
- **`_MAX_SESSION_CONTINUES = 5`** hardcoded constant → move into `limits.py` for consistency.

### Net-new (nothing today provides these)
- **Team Drive** (git) + **Team Scratch** (shared volume) + **Team Threads**.
- **`ask_teammate`** verb — the hallway question.
- **Earned-autonomy governance**: action-tier policy engine + per-agent track record + human
  review-by-exception.
- **Agent hibernation** — stop idle containers; wake on demand.
- **Leads** — accountability owners (influence, not privilege).

---

## 6. Implementation phases

Each phase stands alone, ships runnable, and lists **benefit / build / remove / refactor / security
note**. Phases are ordered so durable-layer cleanup precedes feature work — we never build on a
schema we're about to delete.

### Phase 0 — Clean slate + durable-layer insurance
*Benefit:* removes every rebuild landmine before features land on top; leaves a codebase with no
migration cruft.
- **Build:** `PRAGMA user_version` schema versioning; **protocol version header** on the
  mesh↔agent contract (`src/shared/types.py` + transport) with reject-on-mismatch (closes the
  unguarded rolling-upgrade window); **Personnel File** export/import bundle; **lane rehydration**
  from `tasks` on boot; **per-call model tiering** policy hook.
- **Remove:** `team_migration.py`, `orchestration_migration.py`, all project→team shims, all lazy
  `ADD COLUMN`/`PRAGMA table_info` migration chains (→ clean canonical schemas), `:99/:6080` legacy
  browser surface.
- **Refactor:** collapse task/memory/orchestration schemas to canonical v1.
- **Security:** protocol versioning + agent-server bearer auth (§4) land here.

### Phase 1 — Team as first-class + kernel consolidation
*Benefit:* the structural unit everything else scopes to.
- **Build:** first-class `Team` entity (store, membership, metadata); **enforced team budget
  envelope** with per-agent sub-allocation; solo-agent = team-of-one code path.
- **Refactor:** `costs.py` team budgets → durable + pre-flight enforced; permissions → team policy
  + per-agent deltas; the mesh's team wiring becomes the authority (not the blackboard prefix).
- **Remove:** in-memory team-budget aggregation.
- **Security:** team = the durable trust + budget boundary; cross-team wall formalized here.

### Phase 2 — Collaboration substrate
*Benefit:* the biggest single unlock for real teamwork — kills the 256KB/6k-char serialization tax
and lets teammates actually share work.
- **Build:** **Team Drive** (mesh-hosted git; clone into private `/data`; push/pull mesh-mediated,
  permission-checked, quota-enforced, rejectable; review-before-integrate as a first-class flow);
  **Team Scratch** shared volume (§4 tradeoff, wallet/cred agents excluded); **Team Threads**
  (durable conversation objects, human-visible in dashboard — also fixes the "transcripts trapped
  in containers" observability gap for inter-agent reasoning); **provenance tier** (teammate
  content tagged + sanitized on context entry); **`ask_teammate`** (mesh-mediated, loads recipient
  expertise, bypasses the deep-work lane, returns inline, rate-limited, billed to asker).
- **Refactor:** blackboard → signals only.
- **Remove:** pushback-reissue dance; artifact-shuttling-through-256KB-blackboard patterns.
- **Security:** the one deliberate relaxation lives here; everything else stays mediated.

### Phase 3 — The workday
*Benefit:* turns automation nodes into employees with initiative — the difference between "runs when
pinged" and "owns outcomes."
- **Build:** **agenda loop** (review plate → prioritize → may self-create goal-directed work);
  **dual lanes** (deep-work + interactive); **budget-governed initiative** (autonomy bounded by the
  enforced budget, not by muzzling the heartbeat); richer self-scheduling.
- **Refactor:** heartbeat → agenda loop; deterministic probes retained as cheap pre-checks.
- **Remove:** LLM-skip suppression logic + babysitting prompt copy.
- **Security:** none reduced; budget remains the hard governor.

### Phase 4 — Org model
*Benefit:* accountability and knowledge continuity — teams that integrate work and don't lose
memory when an agent leaves.
- **Build:** **Lead** role (runs standup thread, reviews Team Drive merges, owns team goals,
  default human contact — **influence, not privilege: zero extra permission ceiling**); **hiring
  wizard v2** (draft team goals → derive job descriptions → create agents; `role` no longer frozen;
  templates become starting resumes); **onboarding** (read TEAM.md/GOALS.md, introduce in thread,
  probationary first task) + **offboarding-with-handover** (distill memory/workspace to a handover
  doc in the Team Drive before deletion — makes volume deletion safe, fixes data-remnance from the
  right direction); **Team Room** dashboard (who's doing what, thread activity, plate per agent).
- **Amend constitution:** **Constraint #1** → "no *router* hierarchy" (a lead is an accountability
  owner, not a message router; users still talk to any agent directly). **Constraint #12** →
  clarify leads gain no permission carve-out.
- **Security:** compromised lead is socially loud but no more technically dangerous than any worker.

### Phase 5 — Governance at scale (the "better than humans" keystone)
*Benefit:* removes the human as the throughput ceiling. A superhuman team cannot bottleneck on a
human reviewing everything; trust must be earned per-agent and oversight must shift to
by-exception.
- **Build:** **action-tier policy engine** (every consequential action classified:
  reversible-internal → external-visible → irreversible → financial; generalizes `pending_actions`,
  wallet caps, undo receipts into one gate); **per-agent track record** (accumulated
  accepted/rework/rejected outcomes + summary ratings — raw material already collected, just
  composed and made durable in the Personnel File); **earned-autonomy policy** = f(action tier,
  track record, budget) — a new hire's external emails need lead approval; after N accepted
  deliverables they don't; the human reviews exceptions + samples; **positive feedback push**
  (extend `feedback_push.py` beyond rework/rejected — reinforcement is half of learning);
  **hibernation** (stop idle containers, persist volume, cold-wake on task/ask/mention/cron — a
  small box now hosts a large org whose working set is the currently-active few).
- **Remove:** narrow `pending_actions` store (absorbed).
- **Security:** approval thresholds are *policy* (tunable per org); the gate mechanism, track
  record, and audit trail are *durable*. **This is the piece that lets the same architecture govern
  today's models and 2028's without rebuilding — as models strengthen, loosen policy; the audit
  trail says when that's justified.**

### Deferred until it hurts (do not pre-build)
Per-team DB sharding (team boundary already gives the shard key); multi-host teams; cross-org agent
mobility. External datastores remain a non-goal — SQLite-WAL + single mediator is the right
complexity budget.

---

## 7. Scale posture

First hard wall today is **host RAM/OOM** (~15 agents on 16GB, per the sizing math in
`runtime.py`), then mesh-side shared-SQLite write contention. The answer is **hibernation (Phase
5)**, not re-architecture: hired ≠ running. Sharding waits until contention actually bites; the
team boundary makes it mechanical when it does.

---

## 8. Decisions (ratified 2026-07-04)

Status: **RATIFIED** except items 1 and 4, which stand at their safe defaults pending an explicit call.

1. Security tradeoff ledger (§4) — shared `/team/scratch` volume: **not yet ratified.** Default:
   proceed **git-Drive-first**; raw shared scratch stays deferred/opt-in (safer default), so Phase 2
   is not blocked. Revisit before shipping raw scratch.
2. ✅ **RATIFIED — leads approved, no privilege.** Amend **Constraint #1** to "no *router* hierarchy"
   (accountability leads permitted; users still address any agent directly); **Constraint #12**
   unchanged (leads get NO permission ceiling; the operator-still-gated test stays intact).
   Unblocks Phase 4.
3. ✅ **RATIFIED — cost-philosophy inversion approved.** Replace heartbeat suppression with a
   budget-governed agenda loop + a **designated cheap-model tier** for coordination traffic. The
   coordination/work **spend-split must land first** (B2) before the LLM-skip is removed. Unblocks
   Phase 3.
4. Destructive-cleanup mandate — **authorized in principle** (no users/data); each rename/schema
   collapse still lands atomically with its pinned-test updates (B5) and completes its C.1 deletion.
5. ✅ **RATIFIED — team-of-one stays self-scoped.** The solo→team-of-one unification is a code-path
   merge only; a team-of-one's blackboard/context is scoped to itself, preserving today's isolation
   posture. **This removes the B6 security risk entirely** — no previously-isolated solo agent gains
   shared reachability.
6. ✅ **Delivery: separate PR per unit.** Each logical unit lands as its own GitHub PR off `main`
   (green + regression-tested + lint-clean), reviewed incrementally, rather than one accumulating
   branch.
7. ✅ **RATIFIED (2026-07-05) — C.3-b: goals live in the Team store.** The Phase-1 `TeamStore`
   (SQLite, `data/teams.db`) is the canonical home for BOTH goal kinds: team-level goals
   (`north_star` / `success_criteria` — absorbed from `metadata.yaml` as columns on the `teams`
   row) and per-agent standing goals (an `agent_goals` table **keyed by `agent_id` alone**,
   since membership is strictly one-team-per-agent — goals follow the agent across team moves,
   fixing today's orphaning where `teams/{old}/goals/{agent}` strands on reassignment). The
   blackboard `goals/{agent_id}` key path is DELETED (writer `set_agent_goals`, reader
   `loop._fetch_goals`, dashboard reader, and the `permissions.py` goals carve-outs — the
   anti-injection `goals/` write-block and the self-read exception become dead policy once no
   prompt-injected surface is named "goals," so they go too). Read path becomes
   `GET /mesh/agents/{id}/goals` (self-or-operator); write stays operator-only.
   *Rationale:* (a) Layer-3 placement — goals are accumulated state; the blackboard is slated
   for demotion to ephemeral signals (Phase 2) and `metadata.yaml` is absorbed by the store
   anyway, so the store is the only home consistent with the roadmap; (b) solo = team-of-one
   (ratified #5, same phase) makes store-held goals total — every agent has a governance home;
   (c) Phase-3 agenda loop (per-agent read) and Phase-4 wizard/lead (team goals → job
   descriptions) become plain queries on one DB; Personnel-File export picks up agent goals by
   id. *Declined sub-option:* "field on the agent record" (`agents.yaml`) — goals are written
   by LLM tools at runtime and `agents.yaml` carries the documented B-pre #2 lost-update race;
   config is not accumulated state. *Out of scope:* the operator's business-goals document
   (`GOALS.json`/`GOALS.md` via `manage_goals`) is a human-facing org-outcomes doc owned by the
   operator workflow — a different layer, kept as-is (boundary documented like C.3-d).

---

## Appendix D — Implementation log (live)

Chronological record of what has actually landed on the branch, and any plan
corrections discovered during implementation. Keeps this doc the source of truth
as code lands.

- **Env + baseline.** Project installed (`pip install -e .[dev]`); green baseline
  established (`test_orchestration/lanes/health/costs` = 304 passed).
- **✅ Landed — archive→health-monitor deregistration (B-pre #1 / prerequisite for B3
  hibernation).** `archive_agent_endpoint` now calls `health_monitor.unregister(agent_id)`
  before stopping the container, mirroring the delete path — the poller no longer
  auto-restarts an intentionally-stopped container within ~90s. Regression test added
  (`tests/test_archive_health_dereg.py`, 2 tests green).
- **✎ Plan correction — B4 `blocker_note` piece is ALREADY handled; do NOT build it.**
  A budget-exceeded `RuntimeError` on the task path already reaches the task exception
  handler (`loop.py:1795`) → `status="failed", error=str(e)`, and the mesh already
  promotes `error`→`blocker_note` on failed transitions (`server.py:6486-6489`). So a
  dedicated "promote budget block to blocker_note" change would be **redundant code**.
  The remaining real B4 work is only the **team-envelope semantics** (unset/`0` =
  unlimited), not blocker_note plumbing. Chat-turn budget errors surface directly in the
  chat stream (no task, no blocker_note needed).
- **✅ Landed — mesh↔agent protocol version handshake (Phase-0 insurance).**
  `X-Protocol-Version` added to `src/shared/trace.py` (`protocol_headers` +
  `protocol_compatible`); emitted by the mesh transport on every mesh→agent hop; the
  agent server rejects (426) only when `x-mesh-internal` AND an incompatible major
  version are both present (missing = fail-open, `/status` exempt) — non-breaking by
  construction. One complete emit→check pair, no unread header. 11 tests; touched suites
  green (223 passed).
- **✅ Landed — lane rehydration (PR #1183).** `Tasks.list_pending()` (pending-only, no
  double-exec) + `LaneManager.rehydrate_pending()` (best-effort, at-least-once, tracked
  detached enqueues) + a boot startup hook with an env-overridable settle delay. Verified
  transient-unreachable-safe (`_direct_dispatch` returns `"(no response)"`, never marks the
  task failed). Tests in `test_orchestration.py` + `test_lanes.py`.
- **✅ Landed — personnel-file export v1 (PR #1184).** `GET /mesh/agents/{id}/export`
  (operator-only, read-only): bundles config + permission ACL + cron + best-effort workspace
  markdown. Memory DB (embeddings) + import deferred to follow-ups. Tests in
  `test_agent_export.py`.

- **✅ Landed — atomic project→team rename (PR #1185, the linchpin).** One name, one
  namespace, exactly per A.1/A.2/B5/C.2: `can_manage_projects` + alias validator deleted;
  `tasks.team_id` everywhere (dict key, kwargs, event payloads — no dual emission);
  blackboard `teams/{name}/` prefix (client scoping, host enforcement, ACL patterns, REPL);
  `target_kind="team"`; `TEAM_NAME`/`TEAM_MD_PATH` container envs; `config/teams/` +
  `team.md` scaffold; all `_*_project` CLI helpers renamed; SPA reads `team_id`/`a.team`
  (frozen tab IDs untouched); schema collapse to canonical v1 (`PRAGMA user_version = 1`)
  in `orchestration.py` + `memory.py` (**`_reconcile_embedding_dim` kept** per A.0 #8);
  both migrators deleted with their tests; ~35 test files updated in lockstep; CLAUDE.md
  constraint #9 rewritten as COMPLETE. Full suite green (7716+), ruff clean.
  **Implementation corrections discovered:**
  1. `mesh.py`'s audit-log `undoable`/`archived` ALTER chain was the same archaeology —
     collapsed in the same PR (beyond A.1's manifest).
  2. A second `_PROJECT_TO_TEAM_EVENT` shim lived in `dashboard/server.py` (A.1 implied
     only the host-side one) — both deleted.
  3. The blackboard re-key must cover the persisted **`watchers` table patterns**, not just
     `entries` — registration auto-watch patterns were `projects/{team}/tasks/{id}/*`.
  4. Container-env aliases `PROJECT_NAME` / `PROJECT_MD_PATH` existed and were removed
     (not in A.1's manifest).
  5. The re-key UPDATE in `Blackboard._init_schema` is itself a small one-shot shim —
     delete it in a follow-up once dev instances have cycled (no users, so soon).
  6. Dev-container-only test failures exist (root user + forced HTTPS_PROXY break
     `test_builtins.py` chmod/http_tool tests locally); they pass in CI — don't chase them.
### PR ledger (as of 2026-07-04)
| PR | Unit | CI |
|---|---|---|
| #1180 | archive → health-monitor deregistration (+ restart re-register) | green |
| #1181 | mesh↔agent protocol version handshake | green |
| #1182 | this plan doc | — |
| #1183 | lane rehydration | green |
| #1184 | personnel-file export v1 | green |
| #1185 | atomic project→team rename + schema collapse (the linchpin) | green |

All five branch off `main`; a local integration merge of all four code branches is **conflict-free
and green (908 passed)**. Reviewed via a full pre-merge pass (findings + fixes recorded above).

- **✅ Landed — post-merge adversarial review of the merged stack (2b08fdd..main).**
  Independent multi-agent review (6 finder dimensions, 3-lens adversarial verification per
  finding) of #1180/#1181/#1183/#1184/#1185. Six findings confirmed 3/3, one 2/3; all fixed
  in one follow-up PR:
  1. The blackboard project→team re-key ran ungated on EVERY boot — a post-rename
     `projects/…` key (now ordinary user data) would be silently re-keyed at next restart,
     and via `UPDATE OR REPLACE` could destroy a newer `teams/…` sibling. Now gated on the
     pre-executescript `PRAGMA user_version` (0 = pre-rename DB → migrate once).
  2. CLI `/restart` never re-registered the health monitor after an archive deregistration
     (#1180's follow-up fixed only the dashboard restart path) — mirrored the same guard.
  3. Archive racing an in-flight `_try_restart`: `unregister` can't reach the coroutine
     mid-`start_agent`; the restarted container survived archive. `_try_restart` now
     re-checks `self.agents` post-start and rolls the container back.
  4. The lane-rehydration startup hook ran `rehydrate_pending` on uvicorn's loop, creating
     lane queues/workers on the WRONG loop (live wakes hop to `dispatch_loop`, mutating the
     queue cross-thread). The hook now hops via `run_coroutine_threadsafe` like every other
     enqueue call site.
  5. `_direct_dispatch` converted transport error dicts (unreachable agent, 426 skew) into a
     successful `"(no response)"` turn — ok-status trace + junk auto-notify to the
     originating human channel, repeated per restart for rehydrated tasks of
     unreachable/deleted assignees. Error dicts now record an `error` trace and return
     `SILENT_REPLY_TOKEN` (task stays `pending` for at-least-once recovery). This also
     closes the 2/3 finding: a protocol-skew 426 no longer masquerades as a completed turn.
  6. Settle-window double dispatch: a task created+live-woken while the rehydrate sweep was
     pending was still `pending` in SQLite → enqueued twice. `rehydrate_pending` now skips
     rows with `created_at >= ` LaneManager construction time.
  Everything else survived adversarial verification — notably the rename's auth/scoping
  hunks (`_caller_teams` / `_is_team_member` / publish-subscribe prefix gates), the schema
  collapse, the re-key's collision semantics on genuinely pre-rename DBs, and the protocol
  handshake emit/check pair are clean.
- **✅ Landed — per-call model tiering hook (Phase-0 residual a).** New optional
  `llm.utility_model` config (unset = byte-identical behavior). Both A.C4 blockers closed:
  (1) `_enforce_model_pin` widened — the deployment-configured utility model is always
  acceptable via the existing `_bare()` prefix-insensitive compare (operator-controlled
  config, not agent-choosable; `is_model_compatible` still runs), read through the same
  `_load_config()` the pin already uses; (2) agent-side `model=` threaded at the utility
  call sites — `context.py` `_summarize_text` (all `_summarize_compact` chunk/fold calls
  funnel through it), `_extract_and_store_facts`, `_maybe_consolidate_memory`, and
  `loop.py` `execute_heartbeat` — via `llm.utility_model_kwargs()` (`{}` when off, so the
  call shape is unchanged). Container wiring: `LLM_UTILITY_MODEL` injected at the
  runtime-backend level (`runtime.py`, both Docker + Sandbox env builds, read fresh from
  config per start) so every start path (boot, REPL /restart, health-monitor restart,
  dashboard restart) is covered without per-caller plumbing; `LLMClient` picks it up
  mirroring the `MESH_AUTH_TOKEN` ctor pattern. Spend split + broader tiering policy
  deliberately NOT included (this is the hook only). Tests in `test_llm_model_pin.py`,
  `test_llm.py`, `test_context.py`, `test_loop.py`, `test_runtime.py`.
- **✅ Landed — mesh→agent bearer auth on the agent server (Phase-0 residual (b), §4
  closed-regardless / audit C1 second leg).** The agent server (:8400) now verifies
  `Authorization: Bearer <MESH_AUTH_TOKEN>` on every request via an outermost middleware
  (`_install_mesh_auth_guard`, constant-time compare; env unset = fail-open for dev/tests —
  tokenless production is already blocked by the trust-tier boot gate). **B7 honored:
  enforcement and caller wiring landed in the SAME change** — `Transport._resolve_headers`
  attaches the per-agent token (the transport binds the runtime's LIVE `auth_tokens` dict
  via `bind_tokens`, so every register path — initial start, /restart, health-monitor and
  dashboard restarts, /mesh/register — plus restart token rotation is covered by one bind;
  `register_token` exists for granular use); the mesh's three raw-httpx agent calls
  (/history fallback, INTERFACE.md fallback, artifacts/ingest stream) attach it via
  `_agent_bearer_headers`; the detached CLI (`openlegion chat`) fetches it from the new
  loopback-internal-only `GET /mesh/agents/{id}/token`. **`GET /status` stays exempt**
  (sole exemption — reachability probes: `is_reachable`, `wait_for_agent`, health poller,
  CLI status). No in-container self-callers of :8400 exist (verified). Tests in
  `tests/test_agent_bearer_auth.py` (20); touched suites green.



- **✅ Landed — Phase-1 unit 1: TeamStore + full consumer repoint (C.1 row 1 complete).**
  Ratified §8 #7 (C.3-b: goals live in the Team store) FIRST, then built the store around
  it. `src/host/teams.py`: SQLite `data/teams.db` (env `OPENLEGION_TEAMS_DB`), canonical
  v1 (`PRAGMA user_version = 1`, per-op WAL conns, `:memory:` shared-conn for tests) —
  teams (metadata + north_star/success_criteria + settings + B4 budget-envelope columns +
  Phase-2 drive/thread pointers), `team_members` (one-team-per-agent via `agent_id`
  PRIMARY KEY), `agent_goals` (keyed by agent alone; populated by the goals-repoint
  follow-up). The store owns the `config/teams/{id}/` scaffold (team.md + workflows/);
  `metadata.yaml` is GONE — `_load_teams()`, all `_*_team` CLI helpers, `cfg["teams"]`/
  `cfg["_agent_teams"]`, and the `TeamMetadata` model deleted; every consumer repointed
  (server.py ~67 sites, dashboard ~35, runtime/repl/cli, `MessageRouter` takes a
  `team_resolver` callable, captcha `_tenant_for` reads the DB read-only via
  `file:...?mode=ro`). Endpoint semantics preserved exactly (status codes, response
  shapes, archived counting, ACL rewiring order, cron summary-job lifecycle).
  **Adversarial review (4 finder dimensions + 3 test-diff subreviews) — 6 findings
  confirmed and fixed pre-PR:** (1) reserved ids (`operator`/`mesh`/`default`/
  `canary-probe`) were creatable as team ids — validator regressed vs main's
  `_validate_agent_name` delegation [medium]; (2) `re.match` accepted trailing newline →
  `fullmatch`; (3) multi-statement mutators (add_member/delete_team/remove_agent) were
  non-atomic across the mesh + CLI's second process handle → `BEGIN IMMEDIATE` `_txn`;
  (4) delete→recreate silently adopted a stale `team.md` into new members' prompts →
  scaffold always overwrites; (5) browser-zone `_tenant_for` instantiated the store
  (writable conn + DDL) → read-only URI + plain SELECT seam; (6) team.md brief/context
  writes could 500 after a failed scaffold (DB row now the existence truth) → store
  `team_md_path` + parent mkdir. Plus one coverage gap closed (create-with-existing-
  member ACL strip was untested). **Accepted with documentation:** `MessageRouter`
  team lookups now cost two short-lived SQLite reads per routed message (sub-ms,
  operator-rate traffic; Phase-2 Threads reworks messaging anyway); membership gates
  read LIVE state — members created-with-team are prefix-restricted immediately
  (tightening) and team-delete un-gates ex-members immediately instead of at restart
  (matches intended solo semantics). **Explicit call-out per ratified #4 (no-compat
  mandate):** there is NO YAML→SQLite import — pre-existing `config/teams/*/
  metadata.yaml` teams do not carry over; clean-slate deploys only.

- **✅ Landed — Phase-1 unit 1b: per-agent standing goals repointed to the Team store
  (ratified #7 / C.3-b complete; blackboard `goals/` path DELETED).** New mesh endpoints
  `GET/PUT/DELETE /mesh/agents/{id}/goals` (GET = self-or-operator-or-internal; writes
  operator/internal only, validation mirrors the tool: ≤5 goals, ≤300 chars each after
  `sanitize_for_prompt`, `operator` target rejected, empty list clears, unknown target
  404 naming available agents). `MeshClient` gained `get_my_goals` /
  `set_agent_goals` / `clear_agent_goals`; the operator `set_agent_goals` tool keeps its
  full validation + return shape but calls the endpoints (the list_agents team-resolution
  step is gone — goals are keyed by agent alone); `AgentLoop._fetch_goals` reads
  `get_my_goals` (TTL cache / sentinel / stale-on-failure / prompt shape unchanged;
  `goals: []` maps to None so heartbeat-skip still gates on goals); the dashboard
  GET **and PUT** repointed to `teams_store.get/set/clear_agent_goals` (one goals home —
  leaving the dashboard write on the blackboard would have been a dual-home). The
  `permissions.py` goals carve-outs (self-read exception + `goals/` write-block) deleted
  as dead policy — keys named `goals/...` are ordinary blackboard keys now (pinned by
  the rewritten `test_permissions.py` class). Personnel-file export bundles the store
  record as `goals` (null when unset). New `tests/test_agent_goals_endpoint.py` covers
  the auth matrix + validation. **Adversarial review: 2 low findings, both fixed
  pre-PR:** (1) `DELETE .../goals` on an unknown/typo'd agent id silently reported
  success (the old tool's registry pre-check covered the clear branch too) → DELETE now
  404s naming the roster + rejects the `operator` target like PUT; (2) the LLM-driven
  write path lost the audit trail the old blackboard write produced (the human dashboard
  path logs `edit_goals`/`clear_goals`; the mesh path logged nothing) → both mesh writes
  now `log_audit` with before/after values (`provenance="mesh"`). Cleared as
  non-findings: dev-mode `X-Agent-ID` posture (identical to sibling endpoints, fail-
  closed under enforce), the now-permitted teammate write to a key named
  `teams/X/goals/Y` (inert — nothing reads such keys into prompts anymore), and
  `_fetch_goals` semantics (verified behavior-identical at all five prompt consumers).

- **✅ Landed — Phase-1 unit 2: durable pre-flight team budget envelope (B4 complete;
  in-memory `_team_budgets` deleted).** The envelope lives on the team row
  (`budget_daily_usd`/`budget_monthly_usd`, columns since unit 1); costs.py only READS
  it via `set_team_store()` wiring. New `CostTracker.team_envelope_check(agent, model)`
  runs at the LLM-proxy chokepoint (`credentials.execute_api_call`) immediately after
  the per-agent preflight — aggregate member spend + estimated call cost vs envelope,
  distinct "Team budget exceeded for team 'X'" error (surfaces on `tasks.blocker_note`
  via the existing failed-transition promotion; no new plumbing per the earlier B4 plan
  correction). **THE B4 SEMANTICS FLIP, pinned both ways in tests:** unset/NULL/0
  envelope = UNLIMITED (`test_zero_envelope_is_unlimited`,
  `test_zero_envelope_does_not_block` at the proxy) while the per-agent ledger's
  0-blocks-everything contract is explicitly UNCHANGED
  (`test_zero_agent_budget_still_blocks`). Envelope checks fail OPEN on store read
  errors (an additional governor must not take down the LLM path; the per-agent budget
  still applies) and are skipped entirely on the OAuth path (inherits `_needs_budget`).
  Surfaces: `PUT /mesh/teams/{id}/budget` (operator-or-internal; 0→NULL normalized so
  "unlimited" has one stored shape; caps $10k daily/$100k monthly),
  `manage_team(action="set_budget")` operator tool + `MeshClient.set_team_budget`;
  `get_team_spend` rewritten store-backed (unknown team keeps the historical error-dict
  contract so introspect's `"error" not in` guard is unchanged; known team now returns
  limits — None = unlimited — making `/mesh/costs/team/{id}` and the introspect
  `team_budget` block real for the first time). Old `set_team_budget`/`_team_budgets`
  deleted with their tests (C.1 discipline). **B-pre #3 folded in:** the
  no-settings-file default is now $50/$200 via `DEFAULT_DAILY_BUDGET_USD`/
  `DEFAULT_MONTHLY_BUDGET_USD` constants single-sourced into the dashboard's
  `_SYSTEM_SETTINGS_DEFAULTS` (was a silent $10 enforced vs $50 advertised).
  Known accepted race: the proxy's budget lock is per-agent, so two members' concurrent
  calls can both pass one envelope check — same post-hoc exposure class as the daily
  ledger; the governor converges on the next call. Tests: `tests/test_team_budget.py`
  (endpoint/tool/proxy E2E) + rewritten `test_costs.py` team classes.
  **Adversarial review: 4 findings (1 low-medium, 3 low), all fixed pre-PR:**
  (1) `image_gen` spend counted against the envelope but was never gated by it (its
  preflight branch only ran per-agent `check_budget`) — an exhausted team could keep
  spending real dollars through image generation; the envelope check now runs on that
  branch too (fixed-cost service → gates on already-consumed headroom, no estimate);
  (2) `NaN`/`Infinity` passed `_parse_limit`'s comparisons, silently storing NULL
  (= unlimited) then 500ing on response render — non-finite values now 400 (pinned via
  raw-body JSON, since `json.loads` accepts the non-standard literals);
  (3) three residual hardcoded `$10` daily-default literals survived the B-pre #3
  single-sourcing (`cli/runtime.py` config-budget apply, `cli/repl.py` /restart apply,
  two dead dashboard fallbacks) — all now use the constants;
  (4) `manage_team(set_budget)` is full-replace, so updating one limit silently cleared
  the other — the tool description now says to supply BOTH fields every call.
  Cleared as non-findings: envelope semantics on every path (0/NULL/negative =
  unlimited, exactly-equal allowed, consistent with per-agent `<=`), SQL
  parameterization, OAuth skip, single prod construction site for the wiring, and the
  `get_team_spend` None-limit shape (no `.toFixed`/`:.2f` consumer exists).

### YOU ARE HERE → next phase
Foundation (#1180/#1181/#1183/#1184) and the rename (#1185) are merged. Phase 0's removal
column is fully done. Next, in order:
1. **Phase-0 residuals** (two small independent PRs):
   (a) **per-call model tiering hook** — widen `_enforce_model_pin` + thread `model=`
   through agent-side heartbeat/summary call sites (A.C4). Together with the
   **coordination-vs-work spend split** this GATES Phase 3 (B2 / ratified #3).
   (b) **agent-server bearer auth** (§4 closed-regardless / B7) — wire the mesh→agent
   token through `transport._resolve_headers` + every direct caller in the same change;
   keep `GET /status` exempt.
2. **Phase 1 — team as first-class**: real Team store (delete the `_load_teams()` YAML
   glob per C.1 row 1), durable pre-flight team budget envelope (B4 semantics: unset/0 =
   unlimited), solo = team-of-one (ratified #5: self-scoped blackboard). Decide C.3-b
   (goals home) first. Fold in: cap TEAM.md via `BOOTSTRAP_CAPS` (A.2 hazard) and the
   B-pre #3 explicit budget default. Personnel-file *import* needs B-pre #2's config
   file-lock first.
3. **Then Phases 2–5** as sequenced in §6 (decide C.3-a and C.3-e before Phase 2;
   B1 reframe for Phase 3's "dual lanes"; B3 scope for Phase 5 hibernation).

**Handoff note:** this doc is the source of truth — a fresh session can continue from here without
this session's chat history. Read §5 (keep/refactor/remove), Appendices A–C (surgical manifest +
regression register + dead-code ledger), and §8 (ratified decisions) before starting the rename.

---

## Appendix A — Codebase reconciliation & surgical manifest

Verified against the tree by four parallel code-audit passes. **This appendix is authoritative on
any conflict with §5/§6** — the body describes intent; this describes the exact code. Line numbers
are point-in-time; treat symbol names as the durable anchor.

### A.0 Plan corrections (things the body assumed that the code contradicts)

| # | Body claim | Reality |
|---|---|---|
| 1 | Goals writer is dormant/unshipped | **Shipped.** `set_agent_goals` (`operator_tools.py:3084-3196`) writes `goals/{agent_id}` (or `projects/{team}/goals/{id}`) — the exact key `loop.py:_fetch_goals` (`:1840`) reads. |
| 2 | Only per-agent goals exist | **Team-level goals also exist:** `set_team_goal` (`operator_tools.py:2138-2205`) → team metadata north-star; plus operator-self `manage_goals`. |
| 3 | `pending_actions` + edit confirms are one gate | **Two separate mechanisms.** `pending_actions` is delete-confirm only (`server.py:7509,7573`). Config edits dropped propose/confirm in PR #927 (`types.py:92`) → immediate-apply + undo-receipt (`change_history.py`). The action-tier engine unifies *both*. |
| 4 | Per-agent outcome history must be built | **Already aggregated:** `count_outcomes_since` / `count_failed_status_since` (`orchestration.py:1125-1185`) feed the operator heartbeat. Earned-autonomy layers *persistence/scoring* on top. |
| 5 | Hibernation is net-new | **Primitive exists:** `stop_agent(remove_data=False)` keeps the volume (`runtime.py:950`); the archive endpoint (`server.py:7417`) already stops-without-remove. Only **auto-restart** is missing. |
| 6 | `:99/:6080` browser surface to remove | **Executable path already deleted** (`display_allocator.py:37`). Prose/test-comment cleanup only. |
| 7 | `/mesh/projects/*` + `*_project` tool aliases to delete | HTTP aliases **already removed** (`server.py:5269`); no agent-facing `*_project` *tools* exist — only internal `_*_project` CLI helpers + two `project_id` kwargs. |
| 8 | `_reconcile_embedding_dim` is migration archaeology | **No — live provider-switch logic** (`memory.py:122-169`). Must survive the schema collapse. |
| 9 | Heartbeat budget ~5 iterations | `HEARTBEAT_MAX_ITERATIONS = 12` (`loop.py:64`); the 5-cap is dead prose. |
| 10 | Per-call model tiering is caller-side-only | Also gated by `_enforce_model_pin` 403 (`server.py:1326`) + model chosen inside the container for heartbeat/summary paths. |

### A.1 Removal manifest (delete code + tests; anchors, not just line numbers)

- **`can_manage_projects`** — field `types.py:435` + `_unify_manage_teams_alias` validator
  `types.py:447-455`; **live consumer** `permissions.py:177` (rewrite to `can_manage_teams`); docs
  `configuration.md:211`; ~6 test files reference it (`test_permissions.py:1090` asserts non-persistence — re-frame).
- **`tasks.project_id` dict-key** — emitted by `orchestration.py:_row_to_dict:693` + ~25 event
  payloads; **read back by** `server.py` (6055, 6089, dashboard-event dual keys 618-625, …),
  `cli/main.py:734`, `coordination_tool.py:418`, `operator_tools.py:1025/2219`, and the **SPA**
  (`app.js:4822/6408`, `index.html:4414/4504`). Rename column + key + all readers in lockstep.
- **`target_kind="project"`** — `server.py:7512` + JS readers `app.js:3433/3437/3938`. Self-contained rename.
- **`projects/{name}/` blackboard prefix** — `mesh_client.py:_scope_key:106-118` / `_scope_topic:120-130`
  / list-strip `320-327`; host enforcement `server.py:1928-1940,1983-1995,2699-2708`;
  `mesh.py:790-806`; `repl.py:847`. **Data note:** existing rows are keyed `projects/…`; a rename
  needs a re-key pass, not just code.
- **`_*_project` CLI helpers** — `cli/config.py` `_create_project:805`, `_delete_project:845`,
  `_archive_project:965`, `_add_agent_to_project:1028`, `_remove_agent_from_project:1055`,
  `_get_agent_project:796`; ~25 call sites in `server.py` + `dashboard/server.py`. Rename to team.
- **`config/projects` symlink** — created only in `team_migration.py:88,103`; fallback readers
  `cli/config.py:70-73`, `captcha_cost_counter.py:557/581/619` must hard-point at `config/teams/`.
- **`team_migration.py`** (whole file, 336 lines) — sole caller `server.py:846-850`; test
  `tests/test_team_migration.py`.
- **`orchestration_migration.py`** (whole file, 209 lines) — sole caller `server.py:952-969`; test
  `tests/test_orchestration_migration.py`.
- **Schema collapse** — `orchestration.py`: delete the `_team_col` detection (`362-384`), the
  `outcome_columns` ADD-COLUMN loop (`385-433`), and `_ensure_outcome_set_at_column` (`435-470`);
  fold all columns into the canonical `CREATE TABLE` (`299-361`). `memory.py`: delete the
  ADD-COLUMN block (`254-283`) but **add `category_id` to the `facts` CREATE TABLE** (`173-186`) and
  **keep `_reconcile_embedding_dim` (`122-169`)**.
- **`:99/:6080`** — comment/docstring cleanup in `display_allocator.py:3-7,34-37,258-261,62-64` and
  stale test comments `test_browser_service.py:296,438`.

### A.2 New-build attach points (with the non-obvious difficulty)

- **Protocol version header** — set via a new sibling to `trace_headers()` (`shared/trace.py`)
  merged in `host/transport.py:_resolve_headers:29-39` (mesh→agent) and `mesh_client._trace_headers:143`
  (agent→mesh). Reject via a new `@app.middleware("http")` on the agent (`server.py:203`, mirror
  `_install_body_size_limit`) **scoped to `x-mesh-internal` calls** and on the mesh (`server.py:801`).
  *Difficulty:* first-party callers (dashboard, CLI, provisioner health-check) must send it or be
  exempted, or a version skew locks them out during rolling deploys. Header, **not** a Pydantic field.
- **Personnel File** — assemble the inverse of `create_custom_agent` (`server.py:4191-4393`):
  `agents.yaml` entry + `permissions.json` row + `cron.json` rows filtered by `job.agent` + the
  `openlegion_data_{name}` volume (workspace + memory DB + `learnings/`). *Difficulty:* the heavy
  payload (memory **sqlite with embeddings**) lives in the volume, reachable only via agent `/files`
  or `docker cp`; embeddings are model-specific (re-embed on cross-model import — the canonical text
  survives, §Keep); volume + `costs`/`traces`/`tasks.assignee` rows are keyed on sanitized agent id,
  so import-under-new-name = a re-key.
- **Lane rehydration** — hook right after `set_tasks_store` (`cli/runtime.py:1633`); drive from
  `Tasks.list_inbox(assignee, include_terminal=False)` (`orchestration.py:1048`). *Difficulty:* the
  lane message text, `mode`, `auto_notify`, and the caller `future` are **never persisted** — you
  can only reconstruct a fire-and-forget followup with a message synthesized from
  `title`/`description`. Drive off `PENDING_STATUSES = {pending}` only (`orchestration.py:83`), not
  all non-terminal, or you risk **double-executing** a `working` task (no dispatched-to-lane flag exists).
- **Per-call model tiering** — `llm.chat/chat_stream/embed` already take `model=` (`llm.py:282,337,476`);
  proxy reads `params["model"]` (`credentials.py:1351`); failover resolves on it (`failover.py:133`).
  *Difficulty (A.C4):* `_enforce_model_pin` (`server.py:1326`) 403s off-allowlist models — widen the
  pin or exempt an internal tiering path; and heartbeat/summary model choice is **inside the
  container** (`default_model`), so those paths need agent-side `model=` in `context.py:_summarize_*`
  and `execute_heartbeat`, not a mesh-only change.
- **Team first-class** — today "team" = `TeamMetadata` YAML (`types.py:752`) globbed by
  `_load_projects` (`cli/config.py:777`) + in-memory `_agent_projects` + the blackboard prefix + ACL
  patterns + the `TEAM.md` mount (`runtime.py:509-531`). Introduce a real store (mirror
  `WorkSummariesStore`/`ConnectorStore`) keyed by team id, absorbing metadata + the (currently dead)
  budget + drive/thread pointers, and back `/mesh/teams/*` with it. *Difficulty:* `_load_projects`
  is an O(dirs) glob on hot paths (`server.py:5344`, list_agents `3341`); no team id exists (identity
  = directory name); `TEAM.md` is **uncapped** in the prompt (`workspace.py:117` omits it from
  `BOOTSTRAP_CAPS`) — a growing Team Drive doc would flood every member's context.
- **Team Drive / Scratch volumes** — attach in the same `volumes` dict (`runtime.py:501-561`); thread
  team id through `env_overrides`. *Difficulty:* **no per-team volume lifecycle owner exists** — team
  create/delete never touch Docker volumes; needs a new `RuntimeBackend.ensure_team_volume` /
  `remove_team_volume` owned by the new team store. The **SandboxBackend microVM has no shared-volume
  analog** (per-agent dir copies) — Team Drive is Docker-backend-only unless a sync layer is added.
- **`ask_teammate` + Threads** — dormant infra: `POST /mesh/message` → `MessageRouter.route`
  (`mesh.py:788`) can route synchronously to a peer's `/message`; the deep-work bypass is
  `LaneManager.enqueue(mode="steer")` (`lanes.py:337-374`). *Difficulty:* `MessageRouter.message_log`
  is an in-memory `deque` (`mesh.py:759`) — **Threads needs a new durable store** (extend
  `dashboard/conversations.py` or a sibling). Steer intentionally drops `task_id` (`lanes.py:276`),
  so an ask-via-steer won't auto-close a task.
- **Hibernation** — build on archive (`server.py:7417` already `stop_agent(remove_data=False)`); add
  auto-restart = `start_agent` + `wait_for_agent` (`runtime.py:980`, ~0.5s poll to 30s) triggered by
  the next task/ask/mention/cron.
- **Dual lanes** — every structure in `LaneManager` is keyed by `agent` alone (`lanes.py:127-132`);
  add a parallel interactive queue/worker keyed `(agent, lane_kind)` + a `lane` param on `enqueue`
  (`:257`) and `_ensure_lane` (`:244`); fan `get_status`/`remove_lane` over both.
- **Action-tier engine** absorbs: `pending_actions` delete-confirm (`server.py:7509,7573`),
  undo-receipt TTL tiers (`change_history.py` + `HARD/SOFT_EDIT_FIELDS` `types.py:73/97`), wallet
  `_check_policy` (`wallet.py:1098-1138`), captcha cost caps (`captcha.py:66-77`). Track-record
  source = `orchestration.count_outcomes_since` + `summaries` ratings + `feedback_push`
  (extend beyond `_ACTIONABLE_OUTCOMES=("rework","rejected")` `feedback_push.py:31` to push positives).

### A.3 Ordering constraints these findings impose on §6

1. **Phase 0 schema collapse and the project→team rename must land together** — you cannot delete
   `team_migration.py` while leaving `_team_col` detection, and you cannot rename the blackboard
   prefix without the SPA edits in the same change. Treat "project→team native rename" as one atomic
   Phase-0 PR spanning backend + frontend + schema.
2. **Lane rehydration (Phase 0) depends on task-status semantics** — ship the `{pending}`-only drive
   with a new "dispatched" marker if you later want to recover `working` tasks safely.
3. **Team-first (Phase 1) must precede Team Drive/Scratch (Phase 2)** — the volume lifecycle owner is
   the new team store; there is nowhere to hang volume create/delete until it exists.
4. **Model tiering (Phase 0/3) needs the pin widened first** — otherwise every tiered call 403s.

---

## Appendix B — Regression risk register

Verified against the tree by two targeted regression-audit passes. Ranked by severity. Each entry:
the change, the invariant it threatens, the confirmed failure, and the required mitigation. **These
are the "don't break things that matter" gates — several change how a phase must be implemented.**

### B1 — 🔴 CRITICAL: "Dual lanes" as written would corrupt agent state
- **Invariant:** the agent runtime is **single-lane by explicit design**. A chat arriving during a
  task is shunted to the steer queue with the code comment *"prevents concurrent state corruption
  (shared loop_detector, state, flush_triggered)"* (`loop.py:3026-3028`). Execution is mutually
  excluded by `_chat_lock` (`loop.py:426`, held at `3078`/`5094`), the `state != "idle"` guards
  (`server.py:210`, `loop.py:2357`), and the `current_task is not None` redirect (`loop.py:3029`).
- **Failure if built naively:** a true second parallel worker calling `/chat` while a task runs
  would interleave two turns over the **same** shared mutable state — `_chat_messages`
  (`loop.py:421`), `state` (`:413`), `_loop_detector` (`:446`, `.reset()` per turn), `current_task`,
  and concurrent memory-DB writes + context compaction. Result: corrupted conversation, cross-reset
  loop detection, double memory writes. This is the single most dangerous item in the plan.
- **Mitigation (reframes Phase 3):** the "interactive lane" must be **steer-style injection into the
  running turn** (the mechanism that already exists, `lanes.py:337-374`), NOT a second parallel
  execution. True parallelism inside one agent requires per-lane splitting of `_chat_messages` /
  `state` / `_loop_detector` / memory — out of scope. **Rewrite the Phase-3 "dual lanes" item to
  "priority steer lane"**: an interactive request preempts/injects, it does not run concurrently.

### B2 — 🔴 CRITICAL: Removing heartbeat suppression can starve real work of budget
- **Invariant:** idle heartbeats skip the LLM (`cron.py:543-554`) precisely because spend is a
  **single flat per-agent daily ledger** with **no coordination-vs-work separation** (`costs.py`
  `track`/`preflight_check` key on `agent` only). Budget is enforced **pre-flight** — a call that
  would exceed the cap is *blocked*, not warned (`credentials.py:1388`).
- **Failure:** make every 15-min heartbeat a full agenda-loop LLM call (Phase 3) and those ticks bill
  the same daily cap. An agent can **exhaust its daily budget on agenda ticks and then have real task
  work hard-blocked** with a "Budget exceeded" error. With the effective default of **$10/day** (see
  B7), this bites fast.
- **Mitigation (sequencing):** the **cheap-model tier for coordination traffic MUST land with or
  before** the suppression removal, AND introduce a **spend split** (separate coordination sub-budget
  from work budget, or exempt cheap-tier coordination from the work cap). Do not remove the LLM-skip
  until both exist. Keep a deterministic-probe fast path so truly-idle ticks still cost ~nothing.

### B3 — 🔴 CRITICAL: Hibernation fights three existing subsystems
- **Invariant/failures (all confirmed):**
  1. **Health monitor** auto-restarts any unreachable container within ~90s (`health.py:349-445`,
     `MAX_FAILURES=3`, `POLL_INTERVAL=30`) and has **no "intentionally stopped" awareness**. A
     hibernated container is restarted against its will. *(Latent bug found: the archive path
     `server.py:7387-7428` also fails to `health_monitor.unregister` — only delete/remove paths do,
     `server.py:7680` — so archived agents may already be getting silently restarted.)*
  2. **Direct hot path:** CLI/dashboard call the agent container **directly** (`repl.py:1665`,
     `dashboard/server.py:2907`); a stopped container → connection refused → user sees an error. No
     mesh-side wake-then-forward exists.
  3. **Cron dispatch** to a stopped container is caught and swallowed as an error-count bump
     (`cron.py:687-692`) — no wake, no retry.
- **Mitigation:** hibernation is **not** "just stop the container." It needs (a) a hibernated-state
  flag the health monitor respects (or `unregister` like the delete path), (b) a **wake-then-forward
  shim** in front of both the direct CLI/dashboard path and cron dispatch, (c) fixing the archive
  deregistration bug first (it's the same mechanism). Scope Phase 5 hibernation accordingly.

### B4 — 🟠 HIGH: Enforcing team budgets can block an entire team on a 0/unset default
- **Invariant:** budget `0` means **block everything**, not unlimited (`costs.py:277,298`); only a
  *truly-missing* per-agent key falls back to defaults (`costs.py:273`). `get_team_spend` returns
  `{"error": "No team budget configured"}` with **no `daily_limit` key** when unset (`costs.py:377`).
- **Failure:** a new enforced team envelope that defaults to `0`, or whose unset path reads
  `daily_limit` off that error dict, either **blocks every team member's LLM call** or **KeyErrors**.
- **Mitigation:** define envelope semantics explicitly — **unset/`0` = unlimited**, ship generous
  defaults, surface the block on `tasks.blocker_note` (today budget-exceeded surfaces only as a proxy
  error string, *not* promoted to `blocker_note` — wire that promotion so team-budget blocks aren't
  silent).

### B5 — 🟠 HIGH: The project→team rename breaks specific pinned tests + must not touch frozen IDs
- **Confirmed pins that break** on shim removal: `test_team_migration.py:203,313` (`project_id`
  column), `test_mesh.py:1075-1116` (`projects/{name}/` prefix), `test_dashboard.py:3182-3251`
  (`/api/projects/*`), `:5565,5574` (`project_id` emission), `can_manage_projects` in
  `test_edit_soft_endpoint.py` + `test_operator_internet_access.py`.
- **Hard trap:** the four tab IDs `chat/fleet/workplace/system` are **frozen** (Constraint #5) and
  pinned by `test_dashboard_ui.py:314-322`. A blanket "rename project/fleet→team" sweep that touches
  `id: 'fleet'`/`id: 'workplace'` **breaks URL stability and the test**. Rename **labels and the
  `project` domain term only** — never the tab IDs.
- **Also:** `test_dashboard_ui.py:2389-2410` pins CLAUDE.md module rows + constraint text. Every
  removal touching CLAUDE.md must update the pinning test in the same PR (per the repo's own
  "`make test` even for docs-only changes" rule).
- **Mitigation:** treat these tests as **part of the rename PR** — update in lockstep, and add an
  explicit "do not touch tab IDs" guard to the rename checklist.

### B6 — 🟠 HIGH: "Solo agent = team-of-one" loosens solo-agent isolation
- **Invariant:** solo/standalone agents are **blocked from the blackboard entirely**
  (`mesh_tool.py:96-99` `_STANDALONE_ERROR`, enforced on every op) and get **no TEAM.md**
  (`workspace.py:693`). This is a real isolation posture, not an accident.
- **Failure:** unifying every solo agent into a team-of-one flips `is_standalone → False`, **granting
  blackboard read/write + a team-context surface** to agents that are deliberately isolated today —
  an expansion of cross-agent reachability on a blackboard that is "shared by design" (audit H10).
- **Mitigation:** intended or not, this is a **security decision** (added to §8 open decisions). If
  unifying the code path, keep a team-of-one's blackboard scoped to itself so the isolation is
  preserved in behavior even as the code unifies.

### B7 — 🟡 MEDIUM: Agent-server bearer auth breaks every direct caller unless wired first
- **Confirmed:** the agent server is unauthenticated; **no caller sends a token** to it, and the
  per-agent token in `runtime.auth_tokens` is the **agent→mesh** direction, **not wired into the
  mesh→agent transport** (`transport.py:29-39` sets only `x-mesh-internal` + trace). Enforcing bearer
  auth breaks the mesh transport, CLI (`repl.py`), dashboard, and health checks (`is_reachable`).
- **Mitigation:** wire a mesh→agent token through `_resolve_headers` (and every direct caller) **in
  the same change** that enforces it; keep `GET /status` exempt for reachability probes. Sequence
  behind the ICC-off network fix (which already carries the load), so this is defense-in-depth, not a
  flag day.

### B8 — 🟡 MEDIUM: Blackboard "demotion" must not evict non-artifact riders
- **Confirmed riders that MUST keep a home:** `goals/{agent}` (`loop.py:1840`), `inbox/{agent}/
  task_event/` back-edges (`coordination_tool.py:553`), `claim_task` CAS (`mesh.py:196`),
  `signals/{agent}` (`cron.py:716`), `status/*` + template working namespaces
  (`tasks/*,reviews/*,drafts/*,…` across `src/templates/*.yaml`). **Only `output/*` and `artifacts/*`
  are deliverable-shaped** — those are all that move to the Team Drive.
- **Mitigation:** scope "demote to signals" to **moving `output/*`/`artifacts/*` payloads only**;
  everything else stays. Even `hand_off` only moves its `data` blob — its task row + inbox event
  remain.

### B9 — 🟡 MEDIUM: Lane rehydration can double-execute without a claim
- **Confirmed:** there is **no `dispatched`/`claimed` flag** on the task row and no idempotency key on
  `create_task`. `list_inbox(include_terminal=False)` returns `pending` **and** `working`/`accepted`/
  `blocked`.
- **Mitigation:** drive rehydration off `PENDING_STATUSES = {pending}` **only** and **CAS-claim**
  (`write_if_version`) before dispatch, so a crash-loop restart can't re-run the same task. (Already
  in Appendix A.3 #2; restated here as the regression it prevents.)

### B-pre — Pre-existing issues found while auditing (NOT caused by the plan, worth fixing)
_(unchanged list below)_

---

## Appendix C — Dead-code avoidance: replace-pairs & keep/kill decisions

The destructive-cleanup mandate (delete, don't deprecate) is only *achieved* if every "replace X
with Y" finishes by deleting X. A phased rollout is where cruft hides — old and new coexist during a
phase, and if the phase ends without the removal, the old path lingers as redundant legacy. **Each
item below is a removal that must complete inside its phase; treat "old side still present" as an
unfinished phase, not a follow-up.**

### C.1 Replace-pairs — the OLD side MUST be deleted when the NEW lands
| Old (delete) | New (replaces it) | Phase | Cruft risk if not deleted |
|---|---|---|---|
| `_load_projects()` YAML glob (`cli/config.py:777`) + `_*_project` helpers | Team store (real entity/id) | 1 | Two team "stores"; glob left as a fallback |
| Blackboard `output/*` + `artifacts/*` payload writers in `hand_off` (`coordination_tool.py:305`) | Team Drive artifact store | 2 | Handoff data written to two places |
| `inbox/{agent}/task_event/` back-edge feed + `check_inbox` blackboard read (`coordination_tool.py:553`) | Team Threads event feed | 2 | **Decide C.3-a first.** Two event feeds if threads only *adds* |
| `MessageRouter.message_log` in-memory `deque` (`mesh.py:759`) | Durable Threads store | 2 | Dead deque shadowing the real store |
| Heartbeat suppression path (`force_llm`/`is_default` skip, `cron.py:543-554`) + `execute_heartbeat` special-casing | Single agenda loop | 3 | **Two heartbeat code paths** if agenda is added alongside |
| `pending_actions` delete-confirm store (`pending_actions.py`, `server.py:7509/7573`) | Action-tier policy engine | 5 | Two approval systems |
| Per-agent `goals/{agent_id}` blackboard key | Team-first goals surface (if unified) | 1/3 | **Decide C.3-b.** Two goal mechanisms |
| Single-lane-only `LaneManager` keying (`lanes.py:127-132`) | Priority steer lane (NOT parallel — see B1) | 3 | If a 2nd parallel worker is added *and* steer stays, two injection paths |

### C.2 Rename-completions — the term must die everywhere, atomically (per B5)
`project` → `team` across `mesh_client._scope_key`, permission ACL patterns, `_agent_projects`,
`target_kind`, `orchestration._team_col`, `tasks.project_id` dict-key + all SPA readers, config dir +
`project.md` scaffold. **Leaving any `projects/` prefix or `project_id` alias "for safety" is exactly
the legacy cruft to avoid** — there are no external consumers, so there is no safety to preserve.
Exception that must NOT be renamed: the frozen tab IDs `chat/fleet/workplace/system` (B5).

### C.3 Explicit keep/kill decisions (ambiguity itself breeds cruft)
Each must be decided *before* its phase, or the codebase ends with a "maybe-used" path:
- **(a) Threads vs the inbox event feed** — does Team Threads *replace* `inbox/{agent}/task_event/`,
  or coexist? Recommend **replace** (one event surface). If replace, delete the blackboard back-edge
  path and repoint `check_inbox`.
- **(b) Goals home** — team-first goals field vs per-agent `goals/{agent_id}` blackboard key.
  Recommend **one canonical store** (team entity owns goals; per-agent goals become a field on the
  agent record), delete the blackboard key path.
- **(c) `change_history` undo receipts vs the action-tier engine** — undo (revert an applied change)
  and approval (gate before applying) are genuinely different; recommend **keep both but under one
  policy surface** so they don't look like two competing gate systems. Document the distinction.
- **(d) `subagent_tool.py` (in-process ephemeral subagents)** — distinct purpose from hiring a
  teammate (bounded fan-out within one task, isolated memory). Recommend **keep**, but state the
  boundary explicitly in docs so it isn't mistaken for redundant-with-teams cruft.
- **(e) `SandboxBackend`** — it has **no shared-volume analog**, so it can't run Team Drive/Scratch
  (B-pre #6). Decide: **invest** in a microVM sync layer, or **delete the backend** and commit to
  Docker. Do not leave it half-supporting the flagship feature — that is the definition of divergent
  legacy code.

### C.4 Net result if C.1–C.3 are honored
After the plan, the codebase has **one** team store, **one** coordination event surface, **one**
heartbeat/agenda loop, **one** approval surface, **one** goals home, **one** namespace (`team`), and
**zero** migration/shim/lazy-migration code. If any phase ships without completing its C.1 deletion
or C.3 decision, that is the cruft — so the phase-exit checklist is: *"what old path did this replace,
and is it gone?"*

### B-pre list (referenced above)
1. **Archive path health-monitor leak** — `archive_agent_endpoint` never `unregister`s from the
   health monitor (`server.py:7387-7428` vs delete `:7680`), so archived agents are candidates for
   the ~90s auto-restart. Latent today; blocks hibernation until fixed.
2. **Config write lost-update race** — non-atomic load→mutate→save in `cli/config.py:246-260`
   (documented). The Personnel-File import path would widen the window; fix with a file lock or
   atomic temp+rename before adding import.
3. **`config/settings.json` absent → $10/day effective budget default** (`costs.py:36`), while the
   dashboard's saved default is $50/200. Silent, surprisingly low cap; make the default explicit.
4. **`MessageRouter.message_log` is an in-memory `deque(10000)`** (`mesh.py:759`) — inter-agent
   message history is lost on restart and unbounded-per-window. The Threads store should replace it.
5. **Monotonic port allocation never reclaims** (`runtime.py` `_next_port`) — a long-lived mesh with
   agent churn drifts upward. Low severity; cosmetic until very long uptimes.
6. **SandboxBackend feature divergence** — microVM backend uses per-agent dir copies, so Team Drive/
   Scratch, live `TEAM.md` push, and any shared-volume feature have **no analog**. Either commit to a
   sync layer or explicitly scope teams to `DockerBackend` and document the sandbox limitation.
```
