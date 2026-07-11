# Agent-Employee Platform — Target Architecture & Implementation Plan

**Date:** 2026-07-04
**Status:** Proposed (design brainstorm → implementation blueprint)
**Author:** Principal-engineering review
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
  the durable `tasks` table on boot**; ~~add a **second lane** per agent (deep-work + interactive)~~
  → **priority steer path** per agent (interactive requests steer-inject into the running turn with
  a reply back-edge — never a concurrent turn; reframed per B1, ratified §8 #10) so a 4h task no
  longer head-of-line-blocks a 30-second question.
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
- **Refactor:** blackboard → signals only. ✅ **DONE (unit 4)** — `hand_off` data + `save_artifact`
  registration moved to the Team Drive; `output/*` + `artifacts/*` writers and ACL grants gone.
- **Remove:** pushback-reissue dance; ~~artifact-shuttling-through-256KB-blackboard patterns~~ ✅
  **DONE (unit 4)** — the blackboard `output/*`/`artifacts/*` payload homes are deleted (grep-zero).
- **Security:** the one deliberate relaxation lives here; everything else stays mediated.

### Phase 3 — The workday
*Benefit:* turns automation nodes into employees with initiative — the difference between "runs when
pinged" and "owns outcomes."
- **Build:** **agenda loop** (review plate → prioritize → may self-create goal-directed work);
  **priority steer lane** (interactive requests preempt via steer-injection into the running turn
  with a reply back-edge — reframed from "dual lanes" per B1, ratified §8 #10; never a second
  parallel turn); **budget-governed initiative** (autonomy bounded by the
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
   is not blocked. Revisit before shipping raw scratch. **Reconfirmed 2026-07-07 at Phase-2 entry:**
   the Drive ships git-first; no raw scratch is designed or built this phase without an explicit
   user decision.
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
8. ✅ **RATIFIED (2026-07-07) — C.3-a: Team Threads REPLACE the inbox back-edge feed.** When the
   durable thread store lands (Phase 2), the blackboard `inbox/{agent}/task_event/` keys, their
   host-side producers (`_write_task_event_back_edge` / `_wake_operator_for_human_chain` blackboard
   writes), the `check_inbox` blackboard read, and the `permissions.py` inbox self-read carve-out
   are all DELETED (C.1 row 3 completes in-phase). The host still produces the same events at the
   same trigger points — they are recorded in the thread store instead, and `check_inbox` serves
   its `events[]` from there with the same envelope fields (`kind/task_id/recipient/title/status/
   ts/summary/error/blocker_note`), the same actionable-over-informational retention (25-event cap,
   `task_failed`/`task_blocked` never dropped), and the same sanitization. Wake semantics are
   storage-independent and survive unchanged (actionable-kind wakes, L9 creator binding, per-task
   and operator-storm rate limits). *Rationale:* (a) the feed is a closed loop with ONE producer
   (host) and ONE consumer (`check_inbox`) — verified: ChainWatcher re-derives everything from the
   durable tasks table and never touches `inbox/` — so replace is surgical, and coexist would be
   exactly the two-event-surfaces cruft C.1 names; (b) Layer-3 placement — task events are
   accumulated state currently trapped in TTL'd keys (7d/24h, silently lost at expiry and on any
   blackboard GC), while a durable thread row also becomes the first human-visible surface for
   inter-agent traffic (the §6 observability gap); (c) the tool contract is preserved — the only
   consumer of `events[]` is the LLM via `check_inbox`'s return shape, so storage can swap without
   disturbing prompts or the pinned envelope tests (they repoint, not rewrite). B8's rider list is
   honored either way: `claim_task` CAS, `signals/*`, `status/*`, and template working namespaces
   stay on the blackboard.
9. ✅ **RATIFIED (2026-07-07) — C.3-e: KEEP the SandboxBackend; the Team Drive is transport-level,
   not volume-level.** The Drive as built this phase is mesh-hosted git served over the existing
   agent→mesh HTTP channel (smart-HTTP endpoints on the mesh, per-agent bearer auth), with each
   agent's clone in its own private `/data`. It mounts NO shared volume — so B-pre #6's "no
   shared-volume analog" gap never touches the flagship feature, and the invest-vs-delete binary
   dissolves: nothing is half-supported. Facts this rests on (verified in-tree): both backends
   inject `MESH_URL` + `MESH_AUTH_TOKEN` (sandbox VMs reach the host via `host.docker.internal`);
   `git` is already in the agent image; the mesh is a host process, so bare repos live on the host
   FS owned by the TeamStore via the new `RuntimeBackend.ensure_team_volume`/`remove_team_volume`
   seam (shared host-dir implementation today; the ABC seam is where a future backend would place
   storage elsewhere). This preserves §5's Keep verdict (the backend is the isolation-upgrade
   path; opt-in, self-disabling to Docker). *Residual divergence is confined to raw shared
   scratch* — which is deferred and unratified per #1; IF scratch is ever ratified, that decision
   MUST resolve the sandbox story (sync layer vs Docker-only scoping) at the same time. Known
   caveat recorded: sandbox-VM→host HTTP egress is wired but exercised by no test; the Drive lands
   with DockerBackend as the tested path, matching the backend's existing best-effort posture.
10. ✅ **RATIFIED (2026-07-07) — B1: the Phase-3 "interactive lane" is a PRIORITY STEER LANE, not
    parallel execution.** The §6 Phase-3 "dual lanes (deep-work + interactive)" bullet is rewritten;
    Appendix A.2's "Dual lanes" attach-point sketch (a parallel queue keyed `(agent, lane_kind)`)
    is superseded and must NOT be built. The agent runtime is single-lane by explicit design —
    `_chat_lock` serializes full turns (`loop.py:425`, held at `3080`/`5094`), and one shared set of
    `_chat_messages` / `state` / `_loop_detector` / memory-write paths underlies every turn — so a
    true second parallel worker corrupts state (B1, the plan's most dangerous item). The priority
    mechanism is **steer-style injection into the running turn**, which `ask_teammate` already
    exercises end-to-end: `LaneManager.try_steer` (`lanes.py:490` — injection probe, no followup
    fallback, no wakeup-rate consumption) → `POST /chat/steer` → `inject_steer` → `_steer_queue` →
    mid-turn drains (`loop.py:3897-3920` after each tool round, `:4162-4180` at final answer, next
    turn's `_prepare_chat_context` as the catch-all). AskBroker's `_deliver_ask`
    (`server.py:1877-1933`) is the reference busy/idle fork: BUSY → steer-inject (no task row, no
    `task_id` — Constraint-#6-correct) with resolution via an answer back-edge; IDLE → a normal
    followup lane turn. The Phase-3 unit generalizes that fork to interactive chat and closes the
    three verified gaps plain chat has today: (a) the dashboard chat path calls the agent `/chat`
    DIRECTLY (`dashboard/server.py:3643-3715`), bypassing the lane and blocking on `_chat_lock`
    behind a long turn until its 120s timeout; (b) a steered human message gets no reply back-edge
    (asks solve this with `answer_ask`; chat today returns only "Steered: message injected"); (c)
    the steer-after-last-drain race (an injection landing after a turn's final drain waits silently
    for the next turn). What true parallelism would require — per-lane splitting of
    `_chat_messages`/`state`/`_loop_detector`/memory — is explicitly out of scope for this plan.
11. ✅ **RATIFIED (2026-07-07) — B2: the coordination-vs-work spend split is MODEL-KEYED at the LLM
    proxy; it lands BEFORE the heartbeat-suppression removal (ordering per §8 #3).** Design of
    record, grounded in code recon:
    - **Classification (mesh-authoritative, never container-supplied):** a call is *coordination*
      iff the requested model equals the deployment-configured `llm.utility_model` — read mesh-side
      via the same `_load_config()` seam `_enforce_model_pin` already uses
      (`server.py:1388-1406`), never from container headers (the container is untrusted). This is
      the "designated cheap-model tier" §8 #3 ratified: the agent-side coordination call sites
      (heartbeat/agenda tick `loop.py:2562`, summarize/extract/consolidate `context.py:604/776/1086`)
      already send that model via `utility_model_kwargs()`, so classification needs no new agent
      plumbing. *Declined alternative:* window-based attribution (mesh opens a "coordination
      window" around cron heartbeat dispatch, like the ask billing window) — fuzzier boundaries
      (turn duration unknown, interleaved traffic), more moving parts, and it would misclassify
      utility-model summarization inside work turns; model-keyed is exact for the traffic class.
    - **Ledger:** new `kind` column on the `usage` table (`'work'` default, `'coordination'`),
      following the existing `trace_id` introspection-migration precedent (`costs.py:177-182`).
      Work-ledger reads used for ENFORCEMENT — `preflight_check`, `check_budget`,
      `_check_budget_post_hoc`, and `team_envelope_check`'s `_members_spend_totals` — filter
      `kind='work'`; REPORTING surfaces (dashboard `/api/costs`, `get_team_spend`, introspect)
      stay spend-inclusive with coordination broken out (money is money; only the governors split).
    - **Governor:** coordination calls skip the per-agent work preflight AND the team envelope, and
      are instead gated by a per-agent daily coordination cap resolved via the `limits.py` float
      pattern (`ask_bill_cap_usd` precedent, `limits.py:150-174`): `OPENLEGION_COORDINATION_DAILY_CAP_USD`,
      default $2.00/day, clamped; `0` = coordination tier blocked (an operator kill-switch that
      restores probe-only ticks). A blocked coordination call fails with a DISTINCT
      "Coordination budget exceeded" error — work traffic is never touched, which is the entire
      point of B2. Exposure accepted: an agent deliberately running work on the utility model taps
      the coordination cap — bounded at the cap, on the cheap model, per day; equivalent-severity
      to the existing accepted per-agent-lock envelope race.
    - **Interactions decided now:** (a) ask-billing precedence — coordination classification WINS
      over an open ask window: utility-model calls never redirect to the asker (keeps the asker's
      cap for the work they asked for; recipient's coordination ledger absorbs its own compaction);
      (b) OAuth path unchanged (`bill=False` — tracked at zero cost, no ledger, no split);
      (c) failover substitution: classification keys on the REQUESTED model (the honest
      coordination signal), not the post-failover substitute.
    - **`llm.utility_model` unset = no coordination tier exists.** Idle-tick starvation is closed
      by the deterministic-probe fast path either way (retained per §6 Phase 3 / B2 mitigation —
      truly-empty plates never reach the LLM). Goal-directed initiative escalation (a tick with
      NO actionable items where the agent would work purely toward standing goals) additionally
      REQUIRES the coordination tier: without a designated cheap model, speculative initiative
      ticks would bill the work ledger on the strong model — the exact B2 starvation vector — so
      unset utility_model keeps those ticks probe-only. Actionable items (pending tasks, inbox
      events, probe alerts) always escalate — that is work responding to work, and pre-B2 behavior
      already dispatched the LLM on activity/probes. Deployment guidance: set `llm.utility_model`
      to enable initiative.
12. ✅ **RATIFIED (2026-07-11) — Phase-4 lead-goals reading: "owns team goals" is ACCOUNTABILITY,
    not a write path.** "Lead owns team goals" (§6 Phase 4) is read consistently with "influence,
    not privilege — zero extra permission ceiling": the lead gets goal accountability plus
    proposal/curation duties (goal staleness surfaced on its plate; proposals go to the team
    thread or the operator), NOT a goals write path. Code recon confirms the invariant this
    preserves: BOTH goal kinds are operator-write-only today — team goals
    (`POST /mesh/teams/{name}/goal` + the operator-only `set_team_goal` tool, inline
    operator-or-internal gates) and per-agent standing goals (`_require_goals_writer`, exactly
    two call sites, and NO agent-facing goals-write tool exists at all). The standing gate stands
    unchanged: any goals agent-write carve-out (team or standing) is a NEW decision requiring
    explicit ratification — nothing in Phase 4 may add an agent-reachable goals write. Also
    recorded out of scope: §5's "the lead allocates per-agent budgets within it" (team-budget
    row) IS a lead write path and is NOT built in Phase 4 — it waits for Phase 5's
    earned-autonomy machinery or its own ratification.
13. ✅ **RATIFIED (2026-07-11) — lead drive-review duty is a RECORDED ADVISORY VERDICT; merge
    execution stays operator-tier.** Code recon: approve→merge is ONE operator-only action today —
    `POST .../reviews/{id}/merge` and `.../reject` are both gated `_require_operator_or_internal`,
    there is no "approved" state (statuses `open|merging|merged|rejected|superseded`), and no
    agent tool or mesh-client method can reach either (pinned by test_team_drive.py's
    `test_merge_reject_are_operator_only`). "Lead reviews Team Drive merges" therefore lands
    additively: advisory verdict fields on `drive_reviews` (`lead_verdict` approve/reject + note
    + timestamp), an agent-reachable record-verdict surface gated to the caller being that team's
    lead (verified identity == `teams.lead_agent_id`), open reviews surfaced on the lead's plate
    via the existing `list_reviews(team_id, status="open")`, and the verdict shown to the
    operator (review listing / dashboard / merge response). The verdict has ZERO enforcement
    effect — the merge/reject gates stay untouched and the pinned test keeps passing unmodified;
    §6's security note holds (a compromised lead is socially loud, technically a worker).
    Recording an opinion is influence-shaped, consistent with amended Constraint #12.
14. ✅ **RATIFIED (2026-07-11) — lead is TEAM DATA, not an identity tier: `teams.lead_agent_id`,
    operator-only assignment; standup posts are HOST-PUBLISHED.** Shape: a nullable
    `lead_agent_id` column on the `teams` row — matching every existing per-team singleton
    (`drive_ref`, `thread_ref`, `north_star`, budget columns; the §3 diagram already lists `lead`
    as a team attribute) — with a `set_lead()` that validates real membership; assign/unassign is
    operator-only (the same inline gate every sibling team-metadata write uses). Integrity rule:
    every path that removes the lead from its team (remove_member / remove_agent / delete_team /
    offboarding) clears the pointer. "Default human contact" is a dashboard default ONLY — never
    message routing. Standup: recon confirms the team channel thread is WRITE-LESS today and
    thread writers are host-side only (a pinned invariant that SURVIVES): a standup is an
    ordinary cron MESSAGE job on the lead's own container (followup-lane `/chat` turn), and a new
    `CronJob.post_to_channel` field tells the cron executor — host-side, in-process with the
    ThreadStore — to post the turn's returned text into the team channel thread on the lead's
    behalf (mirroring the three existing host-side writers; bootstrap + boot-reconcile mirror
    `ensure_summary_job` / `_reconcile_work_summary_jobs`). NO agent-facing thread-posting
    endpoint is added. Recorded residual: teammates are not notified of channel posts (no
    agent-side thread read surface exists); pushing channel traffic into agent turns is a future
    decision, not Phase 4. Constitution amendment text (lands with unit 1, in CLAUDE.md plus its
    pinning test per B5): **Constraint #1** → "Fleet model, no *router* hierarchy. No CEO agent
    and no routing through a lead: users talk to any agent directly; agents coordinate through
    the blackboard. A team lead is an accountability owner (standup, advisory review verdicts,
    goal stewardship) — never a message router and never a permission tier." **Constraint #12** →
    append "Team leads gain no permission carve-out — lead is team data (§8 #14), not an identity
    tier; the operator remains the only trust-tier bypass."
15. ✅ **RATIFIED (2026-07-11) — offboarding-with-handover hooks at ARCHIVE time (recon-corrected);
    all three delete surfaces converge on it.** Recon correction to §6's "before deletion": the
    host reads agent workspace ONLY over HTTP through the running container (the export bundle's
    `GET /workspace` proxy; there is no direct FS path), and archive STOPS the container — so the
    handover window closes at archive, not delete. Offboard flow for team members with a live
    container: (1) a handover TURN on the still-live agent via the existing followup-lane
    system-note wake — the agent distills its own SOUL/MEMORY/INSTRUCTIONS/learnings into a
    handover doc (recon correction: this is a normal turn on the agent's own model billed to the
    work ledger — one-off, bounded by the existing preflight; no force-utility-model chat surface
    exists and none is built for a one-shot turn); (2) a mesh-side raw snapshot fallback (the
    existing export-agent bundle: config + ACL + cron jobs + workspace markdown best-effort +
    goals); (3) both committed to the Team Drive via IN-PROCESS `team_drive.commit_file` (author
    = the departing agent; needs no token — mirrors the artifact endpoint's own internal call)
    under a dedicated `handovers/{agent_id}/` path; (4) THEN archive. Delete keeps its existing
    archived-precondition + human-origin confirm chain. Unification (recon found the three delete
    surfaces asymmetric): the dashboard `DELETE /api/agents/{id}` (destroys the volume
    immediately, and skips `teams_store.remove_agent` — a live dangling-membership/goals-row bug)
    and the CLI `/remove` (never removes the volume at all) BOTH converge on the same
    offboard→archive→delete discipline, fixing both pre-existing bugs in-unit. Limitations
    recorded: solo/teamless agents have no Team Drive — export surface only, no handover commit;
    the binary memory DB (embeddings) stays out of the bundle (already a recorded deferred
    follow-up — MEMORY.md carries the distilled facts). Onboarding rides the same primitives:
    first-boot ritual (read TEAM.md + team goals; introduce itself in the team thread via the
    SAME host-published channel-post primitive as §8 #14; probationary first task created by the
    lead if present else the operator — creator attribution only, no privilege implication).
16. ✅ **RATIFIED (2026-07-11) — hiring wizard v2 is OPERATOR-TIER; B-pre #2's file-lock is an
    in-unit prerequisite (recon-widened); "role unfrozen" is a plan correction.** (a) Recon
    widened B-pre #2: `config/permissions.json` is already atomic per individual write
    (tempfile+rename) — the REAL exposure is `config/agents.yaml`, where EVERY writer is a bare
    truncate-and-write (no lock, no atomic rename) and the one existing `_creation_lock` is
    asyncio-only, covering two of FIVE creation entry points (dashboard creates, the CLI REPL
    thread, and the setup-wizard process are all unguarded). Fix, built FIRST inside the hiring
    unit: a sidecar `fcntl.flock` advisory lock held across each helper's full load→mutate→save
    (in-repo precedent: browser/profile_schema.py) + tempfile+`os.replace` atomic writes for
    agents.yaml, and the same lock wrapping permissions.json's load→mutate→save (closing its own
    documented lost-update caveat). (b) Plan correction — `role` is ALREADY operator-editable
    (it sits in `SOFT_EDIT_FIELDS`; edit_agent, the dashboard config PUT, and the CLI wizard all
    write it); the ONLY freeze is the fleet-template `agent_overrides` allowlist
    (`_ALLOWED_OVERRIDE_FIELDS` excludes it; one pinned test). "Unfreezing" = add `role` to that
    allowlist + update the pinned test + the two tool docstrings. Role is descriptive/coordination
    text only (no tool gating reads it) — zero permission surface. Staleness recorded: role edits
    do not hot-reload (restart refreshes; the pinned skip-hot-reload test stands); the
    health-monitor rebuild re-registers WITHOUT `role=`, leaving the mesh roster cache stale —
    fixed in-unit (one line + a test). (c) Wizard v2 keeps wizard v1's shape — the dashboard
    drives the OPERATOR AGENT'S own operator-tier tools (the wizard UI never calls create APIs):
    team goals → derived job descriptions → `create_agent`/`apply_template`; templates become
    starting resumes (their instructions/soul/interface seed the new hire's workspace). No new
    privilege tier, no new agent-reachable creation surface. (d) Discovered dead config: the
    template `resources` key round-trips into agents.yaml with NO reader anywhere — DELETED
    in-unit (destructive-cleanup mandate: write-only config is cruft; per-agent container limits,
    if ever wanted, are their own feature).
17. ✅ **RATIFIED (2026-07-11) — action-tier policy engine: ONE mesh-side gate, static
    classification, the pending_actions store EVOLVES into its held-actions ledger.** New
    `src/host/policy.py` (`ActionPolicyEngine`): every consequential **mesh-terminating** action
    declares a static `(action_kind, tier)` — `reversible-internal → external-visible →
    irreversible → financial` — and the gate is one shared pre-check called at each endpoint
    after the existing permission check + rate limit (the `_check_rate_limit`/`_record_denial`
    placement). Decisions: `allow | allow_audit | hold | deny`; every decision audited.
    Phase-5 classified actions: `agent_delete`/`team_delete` (irreversible),
    `wallet_transfer`/`wallet_execute` (financial), `notify_user` (external-visible — recon
    found `POST /mesh/notify` has NO permission check today; the tier gate becomes its first),
    `connector_call` (external-visible), config edits (reversible-internal, staying on
    ChangeHistory undo receipts). C.1 row 6 completes as: the store mechanics SURVIVE (nonce PK,
    payload digest, TTL + opportunistic reap, `BEGIN IMMEDIATE` single-use consume, origin gate,
    cancel — battle-tested) evolved under the engine with a `tier` column and a pluggable
    executor registry keyed on `action_kind`; the NARROWNESS is what dies (the delete-only
    producers/confirm hard-coding in `confirm_config_change` and `_apply_pending_delete`'s
    hand-coded dispatch). One approval system at exit, grep-verified. The `pending_action_*`
    WS event family + Needs-you panel + inline chat card are REUSED (no SPA event churn).
    C.3-c honored and documented in-module: approval (gate before) and undo (revert after) are
    different axes under this one policy surface; undo is never grafted onto
    irreversible/financial actions. Policy config: `config/policy.yaml`, human/dashboard-write
    ONLY (ConnectorStore posture), mtime reload, NO agent-facing write tool — the operator
    *agent* cannot loosen its own governance. Constraint #12 absolute: thresholds are operator
    policy; no lead/agent carve-outs. Recorded residuals (explicit): in-container `http_request`
    + `run_command` egress cannot be mesh-gated (the container is the boundary); browser
    per-action writes stay on `can_browser_action` (per-click holds unusable); per-remote-tool
    read/write distinction on MCP connectors deferred (all connector calls classify
    external-visible).
18. ✅ **RATIFIED (2026-07-11) — durable per-agent track record: an append-only ledger written
    at rating time (recon correction: the raw material is NOT durable today).** §6's "raw
    material already collected, just composed" is wrong on durability — rated `tasks` rows reap
    at 90 days, `work_summaries` at 30; a live-query composition silently loses history. New
    `src/host/track_record.py` (`data/track_record.db`, WAL, canonical v1, **never reaped** —
    Layer-3): one `outcome_events` row appended host-side in the same code paths that already
    call `feedback_push` — task `set_outcome` (all four enum values) and summary `set_rating`
    (its three values; enum asymmetry PRESERVED, counts reported by source+value, no invented
    unified scale, no numeric scores — none exist anywhere in the codebase). `drive_reviews`
    rows join as labeled datapoints: `lead_verdict` vs final `merged|rejected` status measures
    lead verdict-accuracy; submitter + status measures submitter delivery quality — auto-merge
    trust (#20) is a property of the *(lead, submitter) pair*. **Rating-trust rule:** autonomy
    scoring uses objective signals (merged-without-revert, rejected-after-approve, terminal task
    status) and HUMAN ratings at full weight; operator-*agent* ratings (the internal
    `POST /mesh/tasks/{id}/outcome` path) are EXCLUDED from autonomy scoring — agents grading
    agents must not feed the trust ladder — while remaining inputs to `feedback_push` learning.
    No backfill (no users; the ledger starts at deploy). Read surface
    `GET /mesh/agents/{id}/track-record` (self-or-operator); composed into `_build_agent_bundle`
    as a `track_record` section — the Personnel File finally carries it, and the git-backed
    `handovers/{agent}/` commits (the most durable store in the system) inherit it.
19. ✅ **RATIFIED (2026-07-11) — earned-autonomy policy = f(action tier, track record, budget);
    defaults preserve today's behavior; probation is an opt-in preset; held actions are released
    by HUMANS ONLY.** `policy.evaluate()` reads TrackRecordStore counts + existing budget reads
    (`check_budget`, `team_envelope_check`, wallet caps — pure reads). Default posture (B4
    precedent — never ship a default that blocks existing deployments): irreversible holds (as
    delete already does), financial stays capped by wallet policy, external-visible =
    `allow_audit`. A documented **probation preset** the operator enables makes low-track-record
    agents' external-visible/financial actions `hold` until N accepted outcomes (§6's example,
    opt-in). The human-origin confirm gate is PRESERVED for every held action (recon: the
    confirm chain requires verified human origin on both the live request and the stored
    proposal — by construction no agent, lead included, can release a hold; this invariant
    survives absorption as top-tier policy, not scattered hardcode). Lead involvement is
    advisory only: a lead-plate probe surfaces held actions pending on teammates (mirrors the
    lead-review probe) and a recommend surface records approve/reject + note on the held action,
    shown on the human's Needs-you card — zero enforcement (declined alternative, recorded:
    lead releases holds via an ask round — rejected as the first agent-verdict-releases-action
    gate, violating Constraint #12's construction). By-exception review = the held-action queue
    plus a cheap read view of recent `allow_audit` decisions (audit-log filter on the
    dashboard), not new hold machinery. **Scope fence for this cycle:** the earned-autonomy
    *executor* covers the reversible-internal tier only (drive merges, #20); external-visible
    actions stay human/policy-held until the track record has real data.
20. ✅ **RATIFIED (2026-07-11) — kernel-executed auto-merge consuming lead verdicts (supersedes
    the zero-enforcement clause of §8 #13 at the KERNEL layer only).** The lead's drive-review
    verdict remains advisory **at the permission layer** — the verdict endpoint's lead-only
    gate, the merge/reject endpoints' `_require_operator_or_internal` gates, and the pinned
    operator-only tests are all UNTOUCHED. What changes: the governance kernel may act on the
    verdict. A host-side in-process consumer fires when `record_lead_verdict` records `approve`
    on an open review; if earned-autonomy policy (#19) clears the *(lead, submitter)* pair, the
    mesh itself executes the merge through the existing internal merge path (the same
    internal-caller pattern as the `OL_DRIVE_PRIVILEGED=1` review-merge env). Reuses drive.py's
    claim-first atomic merge — `head_sha` pinning already 409s a post-approval branch advance,
    exactly the property auto-merge needs. Guardrails are policy, `limits.py`-tunable,
    conservative defaults: trust floor (~5 human-executed merges of lead-approved reviews for
    the pair, zero rejected-after-approve in the window, before the first auto-merge); sampling
    (20% of auto-merges flagged for human post-review, decaying to 5%); trust decay (one revert
    or rejected-after-approve returns the pair to human-merge); per-day rate cap; every
    auto-merge emits an undo receipt + an operator-chat notification (the ChainWatcher
    `POST /chat/note` delivery path). A drive merge is genuinely reversible-internal (git revert
    exists) — the right FIRST consumer for earned autonomy.
21. ✅ **RATIFIED (2026-07-11) — lead budget allocation within the human envelope (activates the
    item §8 #12 reserved for Phase 5).** A lead-reachable allocation surface, gated exactly like
    the verdict endpoint (verified caller == `teams.lead_agent_id` — team data, not a permission
    tier): Σ(per-agent allocations) ≤ team envelope, clamped via `limits.py`, every reallocation
    audited. The surface can NEVER raise the envelope — top-ups stay human-only forever. This is
    the second lead-gated agent-reachable surface (after the verdict endpoint), acknowledged
    explicitly; Constraint #12 is untouched (allocation within a human-set envelope is
    stewardship of team data, not a permission ceiling change). Composition payoff with the #22
    ladder: budget-blocked task → `blocker_note` → ladder reaches the lead → lead reallocates
    headroom → task retried — self-healing inside the human's money wall.
22. ✅ **RATIFIED (2026-07-11) — delivery loops on already-legal primitives; goals agent-write
    carve-out DECLINED (again).** The unratified §8 #1-adjacent standing gate stays closed: no
    agent — lead included — writes team or standing goals; if decomposition quality ever
    demonstrably needs durable sub-goals, a lead-scoped write is a NEW ratification. Instead:
    (a) **goal-coverage probe** — a cheap deterministic lead-plate check (team goals set but
    fewer than N open non-terminal tasks advancing them, via `tasks_store.list_team`) escalates
    the lead's agenda turn to decompose under-covered goals into tasks via the already-legal
    `hand_off`. Pure layer-2: cron probe + prompt scaffolding. (b) **blocked-task escalation
    ladder** — rungs on existing delivery primitives: (1) re-drive the assignee with the
    `blocker_note` (`deliver_chat`/`try_steer`); (2) escalate to the task creator (followup
    turn); (3) the lead's plate (extend the lead-duty cron probe with a blocked-tasks feed) —
    the lead reassigns via `hand_off`, answers, or reallocates budget (#21); (4) HUMAN, only for
    credential / envelope-exhausted / irreversible-tier blockers (routed into the durable
    `help_requests.py` Needs-you registry) plus a max-age fallback (~48h). Rungs 1–3 retry
    within budget indefinitely and bill the nudged agent's work ledger; the existing single
    stall nudge (`chain_watcher._maybe_nudge_stall`) becomes rung 4; rung climbs rate-limited
    via the watcher's existing claim semantics.
23. ✅ **RATIFIED (2026-07-11) — positive feedback push lands in its OWN file.** `accepted`
    outcomes with non-empty feedback text push to a new `learnings/wins.md` — never the
    corrections file, whose praise-exclusion design intent survives verbatim — surfaced via a
    new bounded `## What worked (keep doing)` section in `get_learnings_context` (same size
    caps/rotation). `acknowledged` still pushes nothing; rework/rejected behavior unchanged.
    The #18 rating-trust rule applies (operator-agent praise reaches learnings, never the trust
    ladder).
24. ✅ **RATIFIED (2026-07-11) — hibernation: the B3 scoping design (the standing gate is now
    satisfied; hibernation may be built as the LAST Phase-5 unit).** New durable state
    `hibernated` in agents.yaml status (`active | archived | hibernated`): hibernated = in
    service, container stopped, volume persisted, AUTO-WAKES on demand; archived keeps meaning
    out-of-service, never auto-woken. **Prerequisite fixes land first** (live bugs regardless):
    (i) boot `_start_agents()` must skip archived/hibernated agents — today it resurrects
    archived containers on every mesh restart; (ii) `_reconcile_heartbeats()` gains the archived
    filter its sibling reconciles already have; (iii) the heartbeat `pending_tasks` probe is
    repointed from the legacy blackboard `tasks/{agent}` prefix to the durable tasks store —
    today a pending `hand_off` task CANNOT trigger the heartbeat safety net (the probe reads the
    wrong store), and `hand_off` to a stopped agent reports full success while the dispatch
    fails silently. Three-subsystem resolution: (1) health monitor — hibernate = `unregister`
    (the archive-proven mechanism), wake = re-register, no new monitor states; (2) cron —
    hibernated agents KEEP their cron jobs; ticks become mesh-probe-only (skip the container
    `/heartbeat-context` call, which fails soft toward FEWER wakes today); actionable plate →
    cold-wake → dispatch, empty plate → zero container contact; (3) delivery — ONE
    wake-then-forward seam at the transport layer: `HttpTransport` gains an injected
    `ensure_running_fn` (no-op when running; cold-wake + `wait_for_agent` when hibernated;
    archived unchanged), uniformly covering lane dispatch AND the direct-bypass paths the
    Phase-4 busy-fork cannot help with (CLI REPL idle stream, dashboard idle stream branch,
    cron fns) — a hibernated agent is by definition idle, so those always take the direct path.
    Idle→hibernate: new per-agent `last_activity_ts` (stamped at lane-dispatch end/steer/chat
    end); a mesh-side sweep hibernates agents with busy=false, empty queue, no `working` tasks,
    no open asks, idle ≥ `limits.hibernate_idle_minutes`. **Default OFF** (unset/0 = disabled,
    B4 semantics); the operator agent is exempt (the human's front door never cold-starts).
    Cold-wake latency ~2–5s accepted (plain `docker run` + FastAPI boot; no image build in the
    hot path — verified). Hardening in-scope (hibernation makes these routine): `__SILENT__`
    filtered at channels + dashboard chat; heartbeat `"Error: …"` strings rejected by
    `usable_agent_reply` (stops a stopped lead's standup posting raw errors to the team
    channel); dispatch connect-failures emit an observability event (task status semantics
    deliberately unchanged — lane rehydration's transient-unreachable-safety depends on them;
    B9 stays out of scope).

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

- **✅ Landed — Phase-1 unit 3: solo agent = team-of-one (ratified §8 #5; B6 risk closed
  by construction).** Pure code-path merge: for SCOPING surfaces a worker's effective
  team = its real team, else its own agent id; messaging/visibility keep REAL membership
  — **(i) reachability is unchanged**: `_caller_teams`, `_is_team_member`, the
  MessageRouter cross-team block, summaries auth, dashboard "Solo" listings/broadcast
  targeting, and the REPL "Solo:" display are all untouched. What changed: (1) workers
  ALWAYS get `TEAM_NAME` (`team_of(agent) or agent_id`) at container start
  (`cli/runtime.py`) with an agent-side fallback in `agent/__main__.py` covering the
  mesh-side create/template start paths, plus the loop's introspect team-sync deriving
  the same effective team — a worker can never run unscoped; the operator alone keeps
  `team_name=None`. `TEAM_MD_PATH` still only for real teams (solo agents have no
  TEAM.md — the workspace's existing optional-file handling covers it). (2) The entire
  agent-side standalone layer is DELETED: `MeshClient.is_standalone`, mesh_tool's
  `_STANDALONE_ERROR` + all seven guards + the `save_artifact` registration skip,
  subagent_tool's result-passing skips, workspace's `_SYSTEM_MD_PREAMBLE_STANDALONE` +
  the `generate_system_md(is_standalone=)` param, `heartbeat_context.is_standalone`,
  cron's standalone heartbeat-rules branch (one 5-rule set — the blackboard rule is
  accurate: it's their private board), AND (beyond the unit spec, which missed it)
  `loop.py`'s whole `is_standalone` surface: the `_BLACKBOARD_TOOLS` exclusion set, the
  standalone task/chat/heartbeat prompt branches, and the roster-fetch skips. (3)
  Host-side self-scope enforcement: the pubsub publish/subscribe team-prefix gates now
  prefix-lock EVERY non-operator/non-`mesh` caller to `teams/{effective}/` (a solo
  worker no longer skips the gate); registration auto-watch/auto-subscribe scope to
  `teams/{agent_id}/…` for solo workers. (4) Permission invariant "self always; team
  while member": `_add_agent_permissions` grants `teams/{name}/*` in the base defaults
  (operator excluded — it keeps `["*"]`), join keeps it alongside the team pattern,
  leave restores it (a leaver lands in a working private namespace, never the old
  empty-ACL lockout), and **(iii) `_ensure_all_agent_permissions` backfills the self
  pattern at boot** for teamless workers with BOTH blackboard fields empty (existing
  defaults mechanism, not a shim; agents with any pattern untouched). (5) **(ii)
  Cross-namespace collision guard** (makes self-scoping sound): team-create paths (mesh
  `POST /mesh/teams` + dashboard `POST /api/teams`) 400 on a name matching an existing
  agent (agents.yaml + live registry); agent-create paths (`create_custom_agent`, CLI
  `_create_agent`, `_create_agent_from_template`, `_apply_template` per-slot upfront +
  the mesh `fleet/apply` upfront sweep) 400/ValueError on `team_exists`. **(iv)
  `scope_kind="solo"` in summaries/cron is KEPT** — a reporting dimension, not a code
  fork. Isolation pins in `tests/test_solo_team_of_one.py` (17, HTTP-level): solo A
  read/writes `teams/{A}/…` but 403s on `teams/{B}/…` and a real team's namespace; a
  team member 403s on solo A's namespace; the operator trust tier still reaches in;
  solo publish/subscribe outside own prefix 403; collision guard both directions;
  join→leave lands on exactly the self pattern; boot backfill; solo registration
  watches/subscriptions scoped. **Judgment calls:** (a) skills_tool — "standalone sees
  the full catalog" became `_is_team_of_one` (`team_name == agent_id`), which preserves
  today's outcome exactly and is SOUND only because of the collision guard; (b) the
  operator now gets the full SYSTEM.md preamble + team-style prompt rules (it ran the
  standalone variants before) — intended, the coordination text is accurate for its
  full access; (c) `params["team"]` passthrough finding: a solo client sends
  `?team={agent_id}` to `/mesh/agents`; the host previously returned `{}` for unknown
  teams — it now resolves a pseudo-team matching a registered/configured agent to a
  team-of-one roster (self + operator appended, exactly what the pre-merge unscoped
  path returned for solo callers); scoping by ANOTHER agent's pseudo-team stays `{}`
  under enforce. No other host path validates the pseudo-team (tasks `team_id` is
  free-string; `get_team_spend`/introspect budget read real membership only). (d)
  Teamless multi-agent constellations (e.g. Basic-plan template fleets that skip
  `create_team`) lose SHARED blackboard riders — each member now writes its private
  namespace; durable-task handoff (`hand_off` → tasks/lanes, inline brief in the wake
  message) is unaffected, and that is the ratified posture (isolation over teamless
  sharing). Similarly, an operator→solo `hand_off` `data` blob lands unscoped
  (`output/operator/…`) and is unreadable by the solo recipient — equivalent to
  pre-merge (the read tool was blocked outright); the inline brief remains the
  functioning channel. CLAUDE.md (teams.py row + blackboard tradeoff bullet) and
  docs/architecture.md + docs/security.md updated in lockstep.
  **Adversarial review (security + correctness lenses) — 5 findings, all fixed pre-PR:**
  (1) [security, MEDIUM] fresh no-template creates carried `blackboard_read: ["*"]`
  from `_DEFAULT_AGENT_COORDINATION_PERMS` — with the client-side guards deleted, a
  never-teamed solo held HOST-level fleet-wide reads (strictly wider than a team
  member, whose join strips `*`; the isolation test fixture masked it by hand-writing
  narrowed ACLs). Fixed three ways: the default no longer ships a read wildcard, the
  create merge strips `*` from workers' blackboard fields (defense vs future
  templates), and the boot backfill narrows the untouched pre-#5 default shapes
  (["*"] / ["*"]+self) while never touching human-customized lists; fixture rewritten
  to pin the REAL create-path posture. (2) [correctness, MEDIUM] ephemeral `spawn-*`
  agents (no permissions.json entry → deny-all default record) got working blackboard
  TOOLS but 403-only ACLs. Fixed structurally: the self namespace is now a
  RESOLUTION-TIME carve-out (`PermissionMatrix._is_own_namespace` — an agent can
  always touch `teams/{its-own-id}/`, sound because of the collision guard, zero
  shared reach), covering spawn agents, legacy rows, and mid-rewrite windows by
  construction; the boot backfill became additive ("self always" without narrowing
  custom patterns), also closing (3) [MEDIUM-LOW] legacy teamless template agents
  with non-empty pre-#5 ACLs that the empty-only backfill skipped. (4) [MEDIUM]
  membership ACL rewires were disk-only — no `permissions.reload()` on any of the
  nine mesh/dashboard membership endpoints, so a leaver kept live old-team access
  until mesh restart; all nine now refresh the live matrix (pinned by a live-matrix
  HTTP test). (5) [LOW] health-monitor/dashboard restarts rebuilt env without
  TEAM_NAME → silent scope flip to the self namespace on restart; fixed at the
  RUNTIME-BACKEND level (`set_team_env_provider`, the LLM_UTILITY_MODEL pattern) so
  every start path — boot, REPL/health/dashboard restarts, spawn — resolves the
  effective team identically; the per-caller env plumbing in cli/runtime was deleted.
  Accepted with documentation: unified prompt text says "other agents" to a
  team-of-one whose board reaches nobody (misdirects the LLM, no data risk — Phase 3's
  agenda-loop prompt rewrite is the right home for per-audience text); pubsub gates,
  pseudo-team roster resolution, operator carve-outs, and hand_off riders all
  verified clean by both reviewers.

- **✅ Landed — Phase-1 unit 4: TEAM.md capped via `BOOTSTRAP_CAPS` (A.2 hazard closed).**
  `_MAX_TEAM = 8_000` (the brief endpoint caps sections at 2,000 chars → ~four full
  sections before truncation); TEAM.md added to `BOOTSTRAP_CAPS` and truncated at
  injection with the standard notice. Bonus cleanup: `get_bootstrap_content`'s inline
  caps dict (a drifted duplicate of the public constant) deleted — the loop now reads
  `BOOTSTRAP_CAPS`, pinned by `test_bootstrap_caps_single_source` so the advertised
  and enforced caps can never diverge again. The `update_workspace` over-cap warning
  now fires for TEAM.md too (additive — only consumer of the mapping). Review:
  proportionate direct sweep (single consumer verified, cap+no-truncation+single-source
  pins added); no findings.

- **✅ Landed — Phase-2 pre-decisions ratified (§8 #8 C.3-a replace, §8 #9 C.3-e keep, §8 #1
  reconfirmed git-Drive-first).** Recorded BEFORE any Phase-2 storage design, per the C.3-b
  precedent. Grounded in a four-way code recon (runtime backends, messaging/conversation stores,
  coordination/blackboard flows, lanes/billing/provenance); load-bearing facts are cited inline in
  the §8 entries. Recon corrections worth recording now:
  1. **✎ Plan correction — "ask enters via the steer lane and returns inline" is not how steer
     works.** `mode="steer"` returns only a status string — `_handle_steer` inspects
     `result["injected"]` and discards any reply; `/chat/steer` returns `{"injected", "agent_state"}`
     (lanes.py:438-475, agent/server.py:527-534). The ONLY reply-returning mechanism is the followup
     dispatch future (`enqueue` → `task.future` → `/chat` response). Unit 3 must therefore pair
     steer-style delivery with an explicit answer back-edge (an `answer_ask`-style callback resolving
     a mesh-held future) for busy recipients, and may use the followup turn's own response for idle
     recipients. Steer's intentional `task_id` drop (no auto-close, Constraint-#6-correct) is
     confirmed and preserved either way.
  2. **✎ Plan correction — there is no 256KB blackboard artifact tax.** Artifact bytes never
     transit the blackboard: `save_artifact` writes content to the agent workspace and registers a
     metadata POINTER at `artifacts/{agent}/{name}`; ingest cap is 50MB (env-tunable), read cap 2MB,
     and `hand_off`'s `data` blob (the only real payload writer, `output/{agent}/ho_*`, 24h TTL) is
     uncapped. The unit-4 move is therefore pointer-rework + payload-home change, not a
     serialization-tax fix. §6's "256KB/6k-char serialization tax" framing survives only as the
     6k-char brief cap, which is real (`_MAX_BRIEF_CHARS`).
  3. **Cost attribution baseline for unit 3:** billing keys strictly on the proxy caller
     (`CostTracker.track(agent, ...)`); no bill-to-other mechanism exists anywhere — "billed to the
     asker" requires a mesh-authoritative attribution seam (never container-supplied headers; the
     container is untrusted).
- **✅ Landed — Phase-2 unit 1: Team Drive (mesh-hosted git; ratified §8 #1 git-Drive-first,
  NO scratch volume).** One bare repo per team at `data/team_drives/{team_id}.git`
  (env `OPENLEGION_TEAM_DRIVES_DIR`), served over git smart HTTP on the mesh
  (`/mesh/teams/{id}/drive/info/refs|git-upload-pack|git-receive-pack`) so every byte is
  mesh-mediated: verified-bearer auth, REAL-membership wall (cross-team 403, solo 403 —
  the drive is NOT part of the team-of-one namespace), `_RATE_LIMITS["drive"]`=(240,60),
  `drive_push_max_mb` (64) + `drive_quota_mb` (512) in `limits.py` (quota pre-checked
  before receive-pack — bounded one-push overshoot by design; short-TTL size cache on
  the app closure). `Content-Encoding: gzip` POST bodies are inflated with a
  decompressed-size cap (zip-bomb 413). Key decisions: (1) storage = CONCRETE
  `ensure_team_volume`/`remove_team_volume` on the RuntimeBackend ABC (host-dir git;
  shared by Docker AND Sandbox backends — C.3-e is thereby MOOT for the Drive: the
  transport is HTTP, no shared volume/microVM sync layer needed); lifecycle = TeamStore
  (`set_drive_provisioner` wired in cli/runtime beside `set_team_env_provider`, boot
  backfill non-destructive, create-path wipe-then-init so delete→recreate never adopts a
  stale drive, provision failure leaves `drive_ref` NULL without failing create; the
  drive endpoints self-heal via `ensure_drive`). (2) main is integrate-only: a
  pre-receive hook (0755, /bin/sh) rejects `refs/heads/main` unless
  `OL_DRIVE_PRIVILEGED=1`, which the mesh env-injects ONLY for operator-tier callers +
  the merge path; workers push feature branches. (3) Review-before-integrate is
  first-class: `drive_reviews` in teams.db (canonical v1, same executescript,
  user_version stays 1) with same-branch resubmit→supersede; merge is mesh-side
  `git merge-tree --write-tree` (requires git ≥ 2.38, checked at call time) +
  two-parent `commit-tree` + `update-ref --stdin` CAS (conflict AND lost-CAS → 409);
  merge/reject are `_require_operator_or_internal`. (4) Agent surface =
  `team_drive` tool (clone→/data/drive, pull, branch, sync [never main],
  submit_review, list_reviews, log, status) with Constraint-#10 failure envelopes;
  auth via per-invocation `-c http.extraHeader` (token never in .git/config, scrubbed
  from all tool output); teammate review text sanitized on context entry. Subprocesses:
  minimal env (PATH/HOME only + the privilege flag), own session, 120s kill-pg timeout,
  option-injection blocked by full-ref args + branch-name grammar. Plan corrections
  discovered: A.2's "attach in the volumes dict / thread team id through env_overrides"
  is obsolete for the Drive (no mount exists — transport-level design); the global 8 MB
  mesh body cap needed a per-route carve-out for receive-pack
  (`_body_cap_for_path` → push cap + 1 MB slack). No dashboard UI (rides unit 2);
  merge/reject reachable via mesh endpoints (dashboard/operator-tool surface deferred).

- **✅ Landed — Phase-2 unit 1 adversarial-review fixes (three finders concurred).** Applied
  on the same branch after the unit landed:
  1. **HIGH — unforgeable review + atomic merge.** `drive_reviews` gains a `head_sha` column
     (same canonical-v1 executescript, `user_version` stays 1 — no migration); `create_review`
     records `git rev-parse refs/heads/{branch}` at submit and the row exposes `head_sha`/
     `head_sha_short` (operator sees what they approve). Merge is now claim-first and atomic:
     `claim_review_for_merge` flips `open→merging` in a `BEGIN IMMEDIATE` txn BEFORE any git
     side effect — a lost claim (already merged/rejected/merging, or a racing second merge)
     409s and runs NO git, so two merges can't both push and no stray empty merge commit can
     land. The recorded `head_sha` is re-verified against the live branch tip at merge time:
     a post-approval branch advance → 409 "branch changed — resubmit" (closes the
     worker-advances-after-approval TOCTOU); the EXACT recorded sha is merged (not the live
     ref). `finalize_merge` (`merging→merged`) and `revert_merge_claim` (`merging→open` on any
     git failure) bracket the git section; `reject` acts on `open` only, so a concurrent
     reject can never flip a `merging` row to `rejected` while main is being integrated (the
     git/DB divergence race). Closes: concurrent merge+reject can't leave main integrated with
     a rejected row; double-merge can't push a stray empty commit; post-approval advance → 409.
  2. **MED — hermetic mesh git env.** `_subprocess_env` now carries `GIT_CONFIG_NOSYSTEM=1`
     and `GIT_CONFIG_GLOBAL=/dev/null` (both RPC + plumbing paths), matching the agent tool —
     ambient host `core.hooksPath`/`receive.*` can't disable main protection.
  3. **MED — quota/RAM concurrency.** A per-repo `asyncio.Lock` (keyed by team_id, on
     `app.state`, NOT a module global) is held across the size-check → receive-pack →
     cache-invalidate window, so a concurrent-push overshoot is bounded to a single push
     (the "bounded by one push" comment is now actually true, not aspirational). A bounded
     drive-category `Semaphore(8)` (also on `app.state`) wraps upload-pack + receive-pack to
     bound mesh RAM. **Per-op RAM ceiling** is ~`drive_push_max_mb` of buffered body per
     in-flight request, so the drive's worst case at cap is `8 × (push cap + slack)`; a full
     streaming-response refactor is a deliberate DEFERRED follow-up (documented, out of scope
     here).
  4. **MED — self-heal off the event loop.** `_drive_repo` is async and offloads the
     `ensure_drive` (re-provision) call via `run_in_executor`, matching branch_exists/
     repo_size; all five drive endpoints await it.
  5. **MED — git-version gate + test portability.** RESOLUTION: on the host (git 2.43) both
     `--name-only` and `-z` work, but `git merge-tree`'s `--name-only` was added in **git
     2.40** while `--write-tree`/`-z` are the 2.38 floor. Rather than bump the gate to 2.40,
     we **dropped `--name-only`** and parse the `-z` NUL-delimited output directly
     (`_parse_merge_tree_z`), which also isolates conflicted FILENAMES from the trailing
     informational-message prose section (fixes the conflict-info cosmetic). The gate stays
     `git_supports_merge_tree() >= (2,38)`. The merge/conflict tests carry a module-level
     `@_requires_merge_tree` skipif so older CI hosts SKIP rather than hard-fail.
  6. **LOW — merge of a deleted branch.** Re-checked at merge time (only main is
     delete-protected) → clean 409 "branch was deleted; resubmit", not a raw 500.
  7. **LOW — feature-branch griefing.** `receive.denyDeletes=true` +
     `receive.denyNonFastForwards=true` set idempotently at provision (in
     `_install_hook_and_config`) so a worker can't delete or force-push over another member's
     in-review branch; main keeps its hook; tags are deliberately unrestricted (a worker tag
     can't touch main). Normal fast-forward pushes still succeed.
  **Test gaps closed:** malformed receive-pack → clean DriveError→500 (uvloop early-exit
  branch, not shadowed by the body-cap middleware); endpoint push-cap 413 in the 1–2 MB
  slack band ("per-push cap" message); pre-receive worker delete/force-push of main + tag
  posture; strengthened assertions (cross-team clone → 403 specifically; operator push →
  main ref moved server-side; quota reject → rejected commit absent from the bare repo;
  token absent from `<clone>/.git/config` after a real clone). Drive suite 53→69 tests.

- **✅ Landed — Phase-2 unit 2: Team Threads (C.3-a executed; message_log + inbox back-edge
  deleted).** `ThreadStore` (`src/host/threads.py`, `data/threads.db`, env
  `OPENLEGION_THREADS_DB`, canonical v1) with kinds `channel` / `task` / `dm`;
  `scope_id` = effective team scope (solo = agent id, same convention as blackboard
  prefixes + summaries). BOTH C.1 rows completed in-unit: (row 4) the router's
  `message_log` deque is deleted — `MessageRouter.route()` records a `dm` thread row
  (type-tagged body + capped payload JSON) and `GET /api/messages` is store-backed;
  (row 3) the blackboard `inbox/{agent}/task_event/` writes, the `check_inbox`
  blackboard read, and the `permissions.py` self-inbox carve-out are deleted —
  back-edge events are `kind='event'` rows on the lazy per-task thread, served via new
  `GET /mesh/agents/{id}/task-events` (goals-GET auth matrix: self-or-operator-or-
  internal, unknown agent 404 naming the roster) and consumed by
  `mesh_client.list_inbox_events()`. Wake semantics, eligibility, L9 creator binding,
  and the per-task + operator-storm rate limits were untouched (storage-independent by
  construction — only the write call swapped). The former TTL split survives as
  `list_events_for` query windows (7d actionable / 24h informational); event rows reap
  after 90d, plain messages are durable. Team channel: created at the mesh
  team-create endpoint + boot backfill for NULL `thread_ref` (`TeamStore.set_thread_ref`
  is now written); both delete paths ARCHIVE the team's threads (audit trail). Human
  visibility: minimal read-only Threads panel under the `workplace` tab
  (`/api/threads*`), live via the new `thread_message` DashboardEvent. Implementation
  decisions worth recording:
  1. **✎ Naming deviation:** the spec's `MeshClient.list_task_events()` name was already
     taken by the per-task audit-history reader (`/mesh/tasks/{id}/events`, used by
     `await_task_event`) — the new inbox-feed reader is `list_inbox_events()` instead.
  2. `check_inbox`'s envelope is field-identical (incl. the 25-cap and
     actionable-retention ordering); only `key` changed value shape
     (`task_event/{task_id}` instead of the old blackboard key) since it was
     informational, never parsed.
  3. Window filtering happens at read time, so observability (the task thread) keeps
     the full 90-day event record while check_inbox stays bounded.
  4. **Adversarial-review fix — event multiplicity regression.** The old blackboard
     back-edge was an UPSERT per (recipient, task): one event per task, latest
     transition wins, window re-classified on overwrite. The initial append model
     served EVERY transition for 7 days. Caught in adversarial review; fixed via
     read-side newest-per-task dedupe in `list_events_for` (dedupe key = the
     task-scoped thread id, window classification on the surviving row only, SQL
     LIMIT 500 newest-first) — restoring overwrite semantics exactly.
  5. **✎ Deviation:** the boot backfill skips ARCHIVED teams (deliberate — their
     threads are archived history; a channel is created/restored only for live teams).
  6. **REJECTED finding:** no blackboard→threads migration for in-flight
     `inbox/*/task_event/*` events, per the ratified no-compat mandate (§8 #4 and the
     Phase-1 "clean-slate deploys only" precedent) — a deploy mid-window drops
     undelivered back-edge events once, by design.
  7. **ACCEPTED with documentation:** (a) the dashboard tier's cookie-optional
     self-hosted posture now fronts thread CONTENT via `/api/threads*` — a
     pre-existing gate (flagged for the security ledger, not a Phase-2 regression);
     (b) DM thread `scope_id` is first-writer-wins (sender's team at first message) —
     observability-only, but `archive_scope` can miss operator-initiated DM threads.

- **✅ Landed — Phase-2 unit 3: ask_teammate (mesh-brokered inline Q&A, asker-billed,
  steer-delivered).** `src/host/asks.py` `AskBroker` on `app.state`; tools `ask_teammate` /
  `answer_ask`; endpoints `POST /mesh/ask` + `POST /mesh/ask/{id}/answer` (rate buckets `ask`
  (20/min) + `ask_answer`). Design decisions of record:
  - **Busy/idle resolution (recon correction 1 honored).** Delivery probes via the new
    `LaneManager.try_steer` (injection WITHOUT the followup fallback). BUSY → steer interjection
    riding the current turn — no task row, no task_id (Constraint-#6-correct, nothing
    auto-closes), never a second parallel turn (B1); resolution comes only from the `answer_ask`
    back-edge resolving the mesh-held future. IDLE → a normal followup lane turn of the ask
    instruction (the recipient's own loop with its workspace/SOUL/INSTRUCTIONS is what "loads
    recipient expertise"); resolves with FIRST of `answer_ask`, the turn's own inline response
    (uncooperative-but-answering turns still resolve), timeout.
  - **Billing window (recon correction 3 honored — mesh-authoritative).**
    `credentials.set_bill_resolver(broker)`: while a window is open for the verified proxy
    caller, budget preflight (per-agent AND team envelope) runs against the ASKER and usage rows
    land on the asker. The window is keyed exclusively off broker state (asker/recipient pair the
    mesh itself validated) — container headers can never open, extend, or redirect it; a
    malicious asker is bounded by rate × cap. Opened by the lane's new `QueuedTask.on_start`
    when the IDLE-path dispatch actually starts (queued unrelated work never bills the asker);
    closed on future resolution +5s grace or when billed spend crosses `limits.ask_bill_cap_usd`
    (default $0.50 — after the cap the recipient pays, blocking asker-funded runaways AND
    recipient-side cost-dumping). Accepted carve-out: BUSY-path interjections are NOT re-billed —
    the marginal tokens ride the recipient's current (recipient-billed) task.
  - **In-memory broker is acceptable.** An ask is a live RPC (seconds–minutes); a mesh restart
    kills the asker's pending HTTP call → the tool surfaces a Constraint-#10 failure envelope, a
    late `answer_ask` gets the non-fatal unknown-ask envelope. Nothing durable is lost: the Q&A
    also posts to the team thread store. Thread posting is OPTIONAL wiring (`set_thread_store`,
    agreed unit-2 API `ensure_dm_thread`/`post_message`) — one line at unit-2 integration:
    `self.ask_broker.set_thread_store(self.thread_store)` in `cli/runtime.py`.
  - **Pushback-reissue dance prompt copy removed (§5 Remove).** `hand_off`/`check_inbox`/
    `update_status` descriptions, operator heartbeat (sentinel `heartbeat_v7_ask_teammate`) and
    playbook (`playbook_v7_ask_teammate` addendum) now direct: answer a `task_blocked` question
    inline via `ask_teammate` so the worker resumes the SAME task — never re-hand_off a
    duplicate; workers ask the creator BEFORE blocking. The blocked-status machinery itself
    (state machine, blocker_note, back-edges) is untouched.
  - **Security posture:** question/answer sanitized + capped (4k/8k) on the mesh side; asker
    return is provenance-tagged "teammate" and framed as semi-trusted input, not instructions
    (§4); worker→operator asks 403 (Task-2e posture — no worker-injected synchronous prompt in
    the operator's privileged loop); cross-team block mirrors MessageRouter; every failure path
    returns `answered=False` + directive `error` + `recovery_hint` (Constraint #10).
- **✅ Landed — Phase-2 unit 4: blackboard → signals only (C.1 row 2 closed).** The two blackboard
  PAYLOAD writers moved to the Team Drive (B8 held: ONLY these two flows changed). (1) `hand_off`'s
  `data` blob → `mesh_client.commit_drive_artifact` → new `POST /mesh/teams/{id}/drive/artifacts`
  (member-or-operator, `drive` rate bucket, `drive_artifact_max_mb`=8 cap, quota-guarded), committed
  to the SENDER's drive main under `handoffs/{sender}/{id}.json`; `artifact_ref` is now
  `drive://{team}/{path}@{short_sha}`. Solo/teamless senders (incl. the operator, `team_name` None,
  and team-of-one where `team_name==agent_id`) have no drive, so the payload folds INLINE into the
  task brief under `## Handoff Data` (6k cap) with `artifact_refs=[]` — the only channel that ever
  worked for those flows. The post-commit failure envelope is `drive_write_failed` (was
  `output_write_failed`, Constraint #10 shape preserved EXACTLY; the loop's `_HANDOFF_FAILURE_FLAGS`
  tracks the rename). (2) `save_artifact` stage-2 → the same drive endpoint with `kind="artifact"`
  under `artifacts/{sender}/{name}`; oversize or solo/teamless degrades gracefully to
  `saved=True, registered=False` + a note with the workspace path (same contract as the old
  registration-failure path). Discovery is a drive listing (`team_drive('log')`) instead of
  `list_blackboard("artifacts/")`. Read path for recipients + the dashboard drill-in:
  `GET /mesh/teams/{id}/drive/file?path=&ref=` (`drive.read_file` = `cat-file`; RAW content, caller
  sanitizes — the dashboard `_resolve_artifact` sanitizes drive text before render). **Direct-commit
  to main is DELIBERATE** (recorded here per the spec): artifacts are deliverable REGISTRATION, not
  reviewed source — review-before-integrate governs agent-pushed feature BRANCHES (receive-pack + the
  pre-receive hook); `commit_file` is mesh-authored plumbing (read-tree/update-index/commit-tree/CAS
  update-ref) that never runs a push, so it bypasses the hook by construction while still recording
  the sender as the commit author. **C.4 phase-exit answer** ("what old path did this replace, and is
  it gone?"): the replaced paths are the blackboard `output/{sender}/*` + `global/output/{sender}/*`
  (hand_off) and `artifacts/{agent}/{name}` (save_artifact) writers — BOTH gone (grep-zero:
  `write_blackboard(output/…)`/`(artifacts/…)` = 0 writers; `_HANDOFF_TTL` deleted; `output_write_failed`
  removed from live code; every template YAML `blackboard_read`/`blackboard_write` `output/*`+`artifacts/*`
  entry removed; the create-default `_DEFAULT_AGENT_COORDINATION_PERMS` and `_OPERATOR_PERMISSION_CEILING`
  `blackboard_write` no longer grant them — so the operator ceiling now REJECTS an `artifacts/*` grant).
  Blackboard SIGNAL namespaces (`tasks/*`, `status/*`, `context/*`, `signals/*`, `claim_task` CAS,
  template `reviews/*`/`drafts/*`/`leads/*`/… working namespaces, the unit-2 inbox/threads path) STAY
  untouched. Note: cross-team handoffs commit to the SENDER's drive, which a different-team recipient
  cannot read directly (the drive wall is REAL membership) — the operator reads all; same-team is the
  common case; this is an accepted narrowing recorded for adversarial review.

- **✅ Landed — Phase-3 pre-decisions ratified (§8 #10 B1 priority-steer reframe, §8 #11 B2
  model-keyed spend split).** Recorded BEFORE any Phase-3 build, per the C.3-a/b precedent.
  Grounded in a four-way code recon (heartbeat/cron dispatch, lanes/steer/chat delivery,
  cost/budget preflight stack, goals/plate/self-scheduling surfaces); load-bearing facts are cited
  inline in the §8 entries. Recon corrections worth recording now:
  1. **✎ Plan correction — the suppression path is TWO sites, not one.** §5/§6 name only the cron
     skip; the agent side has its own goal-gated skip (`loop.py:2370-2378`: empty HEARTBEAT.md AND
     no goals → `{"skipped": "no_heartbeat_rules"}`), plus `force_llm` plumbing spanning
     `CronJob.force_llm` (`cron.py:83`), `_UPDATABLE_FIELDS` (`cron.py:343`), the `x-force-llm`
     header (`agent/server.py:469-479`), and the `execute_heartbeat(force_llm=)` param. The unit-2
     deletion must cover both sites and the plumbing.
  2. There is no `heartbeat_context.py` — the deterministic probes are cron-side
     (`_run_heartbeat_probes`, `cron.py:691-735`: disk, `signals/{agent}`, `tasks/{agent}`) plus
     the agent `GET /heartbeat-context` endpoint (`is_default_heartbeat`/`has_recent_activity`).
     These are the "cheap pre-checks" §6 retains.
  3. **Standing goals are operator-write-only** (`_require_goals_writer`, `server.py:8928-8936`) —
     "may self-create goal-directed work" means self-created TASKS toward operator-set goals, not
     self-edited goals; any goal-self-write carve-out would be a separate, explicit unit-4 decision.
  4. No first-class self-task tool exists; `hand_off`-to-self is unguarded today (gate is
     `can_message`, default `["*"]`) and is the natural seam for self-created work.
  5. `costs.db` has no `PRAGMA user_version` — it uses a file-local introspection-`ALTER` pattern
     (the `trace_id` precedent, `costs.py:177-182`, pinned by
     `test_migration_idempotent_on_legacy_db`); the B2 `kind` column follows that file-local
     pattern rather than the canonical-v1 executescript pattern.
  6. Budget is ALREADY agent-visible (`introspect(section="budget")` → `check_budget` +
     `get_team_spend`, surfaced in the heartbeat prompt's runtime-context block) — unit 4's
     "budget-aware prioritization" is prompt + data-shape work, not a new endpoint.

- **✅ Landed — Phase-3 unit 1: B2 coordination-vs-work spend split (#1202; §8 #11 implemented
  as written).** The `usage` table gains `kind TEXT NOT NULL DEFAULT 'work'` (CREATE TABLE + the
  `trace_id`-style introspection migration — legacy rows default to work, preserving semantics);
  enforcement reads (`preflight_check`, `check_budget`, `_check_budget_post_hoc`,
  `_members_spend_totals`/team envelope) filter `kind='work'` while `get_spend` stays
  spend-inclusive by default. Classification at the proxy: `_bare_model` prefix-insensitive
  compare of the REQUESTED model against the deployment `llm.utility_model`, resolved fresh per
  call through the new `CredentialVault.set_utility_model_provider` seam wired in
  `create_mesh_app` to the pin's own `_deployment_utility_model` reader. Coordination calls skip
  the work preflight, the team envelope, AND the ask-window billing redirect (`note_billed_cost`
  structurally unreachable — `bill_agent != agent_id` can never hold), and are gated by
  `limits.coordination_daily_cap_usd()` (`OPENLEGION_COORDINATION_DAILY_CAP_USD`, default $2.00,
  clamp [0,100], **0 valid = tier blocked** kill-switch) with a distinct "Coordination budget
  exceeded" error in the work path's envelope shape. Both proxy paths covered (sync
  `execute_api_call` fork image_gen/coordination/work; streaming `stream_llm` same fork after the
  OAuth early-returns, distinct SSE error frame, `kind` on the final track — the streaming
  preflight site was NOT named in §8 #11's anchors and was caught in-build). Introspect's budget
  section gains a `coordination` sub-dict (unit 4 consumes it). Tests:
  `tests/test_coordination_split.py` (28) incl. work-exhausted-coordination-allowed at the proxy,
  reverse isolation, envelope exclusion, ask-precedence, legacy-DB migration idempotency; existing
  ledger pins unchanged. Full suite 8162 passed / 15 known dev-container failures / ruff clean.
  **Implementation deviations (2, recorded):** the seam is wired in `server.py:create_mesh_app`
  (where `set_bill_resolver` actually lives — the A.2-era `cli/runtime.py` guess was wrong), and
  `set_utility_model_provider` is a vault METHOD, not a module function (Constraint #8 — no new
  module-level globals).

- **✅ Landed — Phase-3 unit 2: plate-gated agenda loop (#1206; §6 unit 2 as ratified, C.1 row 5
  deleted in-phase).** The cron scheduler now computes the plate mesh-side:
  `actionable = bool(triggered probes) or has_recent_activity or not is_default_heartbeat`
  (custom HEARTBEAT.md rules are always actionable); a non-actionable tick dispatches only when
  the agent has operator-set standing goals AND the deployment has `llm.utility_model` configured
  (goal-only escalation) — otherwise it returns without reaching the LLM (truly-empty plate =
  zero tokens). New scheduler ctor seams `utility_model_fn`/`goals_fn` are wired in
  `cli/runtime.py` from the same deployment-config read the proxy classifier uses (mesh-side,
  never container-supplied). Replace-pair deletions: `CronJob.force_llm` (+ its
  `_UPDATABLE_FIELDS` entry + the four-condition skip), the agent-side `x-force-llm` header,
  `execute_heartbeat`'s `force_llm` param, the goal-gated `no_heartbeat_rules` skip, and the
  `HEARTBEAT_OK`/`_is_heartbeat_empty`/`_HEADING_OR_EMPTY_RE` empty-answer machinery; the
  heartbeat prompt and the 7 bundled templates rewritten to agenda copy ("end the checkup"
  replaces the sentinel). Legacy `cron.json` rows containing `force_llm` load safely (unknown
  fields are filtered against the dataclass fields on read). Agenda wakes carry no task_id
  (Constraint #6) and run on the utility model via the existing `utility_model_kwargs()` path.
  Tests: new `TestPlateGate` in test_cron.py; `TestCronForceLlm` and the sentinel/empty-heartbeat
  tests deleted with the code they covered. Full local suite 8215 passed / 26 skipped / 0 failed;
  grep-zero for `force_llm`/`x-force-llm`/`HEARTBEAT_OK` in src/.

- **✅ Landed — Phase-3 unit 3: priority steer lane (#1208; §8 #10 implemented, two recorded
  deviations).** `LaneManager.deliver_chat` (src/host/lanes.py) generalizes the exact
  `AskBroker._deliver_ask` busy/idle fork to interactive chat: busy → `try_steer_and_wait`
  (steer-inject + wait on the turn's reply), idle → `_handle_followup` lane turn; never a second
  parallel turn (B1 — `_chat_lock` stays single-lane), steered/followup chat carries no task_id
  (Constraint #6). Reply back-edge: `SteerMessage` gains `wait_reply`/`timeout`; `/chat/steer`
  with `wait_reply=True` routes to `AgentLoop.inject_steer_and_wait`, which rides an
  `asyncio.Future` on the steer-queue entry; `_drain_steer_messages` collects folded-in futures
  and `_resolve_turn_steer_futures` — a `finally` sweep in both `chat()` and `chat_stream()`
  before the lock releases — settles them with the turn's own final response text (callers get
  the agent's actual reply, not the old injection ack). The same sweep closes the
  steer-after-last-drain race: still-queued entries with a live future resolve `None` → the mesh
  dispatches a fresh followup (entry consumed, no double delivery); plain entries re-queue for
  the next turn's catch-all; caller-side timeout after injection → "still processing"
  placeholder, never a duplicate followup. Task turns never drain chat steers mid-turn (existing
  task-busy guard catches the followup). Dashboard `/api/agents/{id}/chat` now goes through
  `deliver_chat` instead of POSTing the agent's `/chat` directly (no more 120s `_chat_lock`
  block); trace-id/origin/intent preserved. New clamped limit `steer_reply_timeout_seconds`
  (default 150s). Fire-and-forget steer callers (asks, wakes, REPL, channels) byte-for-byte
  unchanged. **Deviations (2, recorded):** the reply future is held agent-side on the steer-queue
  entry — the mesh waits on one HTTP call, the same RPC-wait pattern `/mesh/ask` uses — rather
  than mesh-held (a mesh-held future would have required a new agent→mesh callback endpoint);
  and only the non-streaming dashboard chat endpoint was re-routed per the unit's gap list
  (channel paths were already lane-aware; the dashboard's STREAMING endpoint and broadcast still
  call the agent endpoints directly — the original "already lane-aware" claim here was wrong and
  was corrected by the end-of-phase review; they are single-lane-safe after #1213's heartbeat
  lock fix but not steer-routed — recorded Phase-4 follow-up). Tests: pinned test_lanes.py (68)
  and test_ask_teammate.py (51, `TestLanePrimitives` untouched) green; 10 new back-edge/race
  tests in test_chat.py; new dashboard busy/idle routing class. Full local suite 8297 passed /
  26 skipped / 0 failed.

- **✅ Landed — Phase-3 unit 4: budget-governed initiative (#1210; §6 unit 4 as ratified — prompt
  + data-shape work only, NO new mesh endpoints, per the recon correction).** The shared
  `_format_runtime_context` formatter in src/agent/loop.py (the single path task/chat/agenda
  prompts all use) renders introspect budget's `coordination` sub-dict
  (`{daily_used, daily_limit}` — added by unit 1 for exactly this consumer) as a labeled line
  under the work-budget line, gated on presence; the agenda turn surfaces it automatically
  through the existing Runtime Context block. The agenda Operating Rules (loop.py) and the
  cron-side agenda rules gate goal-driven `hand_off`-to-self work on plate capacity + the dual
  budget lines; **standing goals remain operator-write-only** (`_require_goals_writer` untouched;
  no agent-side goals write path, tool, or endpoint was added — a self-write carve-out stays a
  NEW decision requiring explicit ratification). `set_cron`'s "Only change if the USER explicitly
  asks" heartbeat muzzle is replaced with budget-governed cadence policy (tighten the interval
  when the plate is full and coordination budget has headroom, loosen when thin or tight —
  consistent with unit 2's deletion of the old cron suppression rule). Tests: coordination-budget
  breakout + omission-when-absent, agenda-prompt goal-driven self-tasking + budget surfacing,
  cron rule copy, set_cron copy pins (existing pinned substrings kept intact). Full local suite
  8303 passed / 0 failed.

- **✅ Landed — Phase-3 end-of-phase review + fixes (#1213).** Consolidated adversarial review of
  all Phase-3 PRs (#1202/#1206/#1208/#1210) on the merged tree — three independent passes (unit-3
  concurrency deep-dive with repro scripts, unit-4 + cross-unit integration, phase-wide invariant
  sweep). Five confirmed findings, all fixed in #1213:
  1. **CRITICAL — heartbeat single-lane (B1):** `execute_heartbeat` never acquired `_chat_lock`
     (only checked `.locked()` at entry), so direct `/chat`//`/chat/stream` callers — including
     the dashboard's actual streaming UI, REPL, broadcast, and lane-worker followups — could run
     a second concurrent LLM turn during an agenda tick (reproduced; unit 2 made agenda ticks
     full turns, turning a negligible window into a routine one). The agenda turn now holds
     `_chat_lock` for its duration (skip-if-busy entry semantics preserved; release in `finally`).
  2. **CRITICAL — steer-reply integrity:** heartbeat drains folded `wait_reply` chat futures into
     `_turn_steer_futures` but only `chat()`/`chat_stream()` run the resolving sweep — a
     dashboard message steered during an agenda tick hung its caller the full 150s and could
     later be answered with an UNRELATED turn's reply (both modes reproduced). Fixed semantic: a
     chat message is never answered by an agenda turn — heartbeat-mode drains resolve live reply
     futures `None` and consume the entry (mesh promptly dispatches a dedicated followup); an
     end-of-turn sweep covers late arrivals; plain fire-and-forget steers keep agenda injection.
  3. **MAJOR — budget flags:** the coordination line now renders `[EXCEEDED]` (used ≥ limit > 0)
     and `[BLOCKED: coordination kill-switch]` (limit ≤ 0) — "$2.00/$2.00" no longer reads as
     headroom.
  4. **MAJOR — inert-tier truthfulness:** with no `llm.utility_model` nothing can classify as
     coordination (all spend is work), yet introspect still emitted the `coordination` block and
     the self-tasking copy pointed at it. Introspect now omits the block when the tier is inert;
     both copy sites state that self-created work ALWAYS spends work budget.
  5. **MAJOR — stale docs:** docs/triggering.md still taught the deleted `force_llm`/skip-LLM
     mechanism (its curl example silently no-ops post-#1206); rewritten as plate-gated agenda
     dispatch (customized HEARTBEAT.md = the supported always-run path).
  **Recorded residuals (no code change):** dashboard streaming chat + broadcast are direct-call
  (single-lane-safe post-#1213, steer-routing = Phase-4 follow-up); `_HEARTBEAT_TOOLS` excludes
  `hand_off` while shared agenda copy mentions it — only the operator role uses that restricted
  set and operators cannot hold standing goals, so inert (pre-existing); prompt-copy substring
  pins are advisory by nature — the enforcing layer is the LLM-proxy preflight. One reviewer
  finding (missing unit-4 landed entry) was already fixed by #1211 before the review completed.
  Regression tests for every fix; full local suite 8313 passed / 26 skipped / 0 failed.

- **✅ Landed — Phase-4 pre-decisions ratified (§8 #12–#16) + build order.** Recon pass first
  (seven parallel read-only sweeps: drive-review privileges, goals/membership schemas, role
  freeze + pinned tests, creation paths + config race, deletion path + handover surfaces,
  dashboard/plate exposure, standup/thread-posting mechanics) grounded the §6 Phase-4 sketch in
  code; the §8 entries record the corrected designs. Highlights the appendices don't carry:
  approve→merge is one operator-only action with no "approved" state → lead verdicts are
  additive advisory metadata (§8 #13); the team channel thread is WRITE-LESS today → standups
  and onboarding introductions are host-published, preserving the writers-host-side-only
  invariant (§8 #14); the handover window closes at ARCHIVE, not delete (container-stop kills
  workspace reads) → offboarding hooks pre-archive and the three asymmetric delete surfaces
  (mesh nonce path / dashboard immediate-wipe / CLI no-volume-removal) converge (§8 #15);
  `role` was never generally frozen — only the fleet-override allowlist — and role edits never
  hot-reload (§8 #16); B-pre #2's real target is `config/agents.yaml`, where every writer is a
  bare truncate-write and `_creation_lock` covers two of five entry points (§8 #16). Build
  order: unit 1 Lead role core (+ Constraint #1/#12 amendments + the host-side channel-post
  primitive) → unit 2 hiring wizard v2 (file-lock FIRST) → unit 3 onboarding +
  offboarding-with-handover → unit 4 Team Room (Team Hub sub-tab; in-memory plate snapshot on
  the CronScheduler + one dashboard endpoint — the four frozen top-nav tab IDs are untouched)
  → unit 5 (small) streaming/broadcast steer-routing (the recorded Phase-3 residual;
  re-deferred with a recorded gate if the design balloons) → phase exit + consolidated
  end-of-phase review (the Phase-3 pattern that caught two CRITICALs).

- **✅ Landed — Phase-4 unit 1: Lead role core (#1218; §8 #12/#13/#14 implemented as ratified —
  zero permission elevation).** Nullable `teams.lead_agent_id` (the per-team-singleton column
  shape) with `set_lead` membership validation (operator rejected) and SAME-transaction integrity
  clears on `remove_member`/`remove_agent`/`add_member`-eviction; `led_team` reverse lookup.
  Assign/unassign is operator-or-internal (`PUT`/`DELETE /mesh/teams/{name}/lead` + dashboard
  equivalents); listings surface the lead; Team Hub shows a Lead badge + controls (default-contact
  is UI copy only). Advisory verdicts: `drive_reviews` gains `lead_verdict`/`_note`/`_at`;
  `POST .../reviews/{id}/verdict` is agent-reachable but LEAD-ONLY (non-lead member, other team's
  lead, and operator-as-itself all 403; open reviews only; notes sanitized + capped) — the
  merge/reject endpoints and their operator gates have ZERO changed lines and the pinned
  `test_merge_reject_are_operator_only` is unmodified; the `team_drive` tool gains
  `record_verdict` and `list_reviews` surfaces verdicts sanitized. Lead plate: a mesh-side
  `lead_reviews_fn` probe (wired like `goals_fn`) surfaces pending verdicts on the lead's agenda —
  one cheap lookup for non-leads. Standup: `CronJob.post_to_channel` (team id) makes the executor
  post a message-job's non-empty dispatch response into the team channel thread host-side (the
  writers-host-side-only invariant survives); the field is excluded from `_UPDATABLE_FIELDS` and
  UNREACHABLE from every agent-facing cron surface (explicit-field extraction on create; pinned by
  dedicated tests); `ensure_standup_job` (dedup by team, repoints on lead change, per-team
  `settings.standup_schedule`, default 09:30 vs the 09:00 summary) + `_reconcile_standup_jobs`
  boot reconcile; lead assign/unassign/team-delete sync live. Constitution amendments landed:
  Constraint #1 → "no *router* hierarchy" wording, Constraint #12 → lead carve-out sentence, with
  a new CLAUDE.md pinning test. **Recorded deviation:** live standup-cron sync hooks the dedicated
  lead endpoints + team delete; a lead cleared via the team-membership endpoints is pruned by the
  boot reconcile. 92 new tests; full local suite 8448 passed / 26 skipped / 0 failed.

- **✅ Landed — Phase-4 unit 2: hiring wizard v2 (#1220; §8 #16 implemented in its ratified order —
  file-lock FIRST).** B-pre #2 closed (recon-widened target): a sidecar `config/.config.lock`
  acquired with blocking `fcntl.flock(LOCK_EX)` on a fresh fd per outermost acquisition
  (same-thread reentrant via a threading.local depth counter — nested helpers like
  `_remove_agent` → `_remove_team_blackboard_permissions` would otherwise self-deadlock;
  cross-thread/cross-process exclusion is kernel-enforced). EVERY `config/agents.yaml` and
  `config/permissions.json` load→mutate→save runs under it — the eleven cli/config.py helpers
  (`_ensure_operator_agent` split into locked wrapper + unlocked body rather than a mass
  re-indent) plus host/server.py `_apply_pending_change` (both branches), the dashboard
  agent-delete, and setup_wizard.py (its interactive confirm stays OUTSIDE the lock). agents.yaml
  writes are now atomic (`atomic_write_text`: tempfile+fsync+os.replace) — previously every
  writer was a bare truncate-and-write. Lock tests were verified against negative controls
  (neutered lock → tests fail). Role unfrozen exactly as corrected: `role` added to
  `_ALLOWED_OVERRIDE_FIELDS` (+ `_apply_template` now honors the override — it was silently
  ignored before, a second latent gap recon missed); the pinned rejection test deliberately
  flipped to a positive end-to-end test + a still-unknown-field negative; fleet_tool
  docstrings/schema updated; health.py `_try_restart` now passes `role=` on re-register (roster
  staleness bug fixed, regression-tested). Dead `resources` template key deleted from
  cli/config.py round-trips, all 13 templates, and `AgentConfig` — grep-zero. Hiring surface in
  the wizard-v1 shape (operator-tier, UI never calls create APIs): Team Hub "Hire teammate"
  button seeds the operator chat via the existing `sendChatTo` pipeline (no new wizard states);
  the `team_build` operator playbook gains the v2 flow — job descriptions derived from team
  goals, templates as starting resumes via `apply_template(agent_overrides={"role": ...})`,
  propose-then-confirm, `add_agents_to_team` after creation. Full local suite 8462 passed /
  26 skipped / 0 failed.

- **✅ Landed — Phase-4 unit 3: onboarding + offboarding-with-handover (#1222; §8 #15 implemented
  as recon-corrected — handover at the live-container window, all three delete surfaces
  converge).** `_offboard_agent` (never raises, manifest-returning): bounded handover TURN on the
  still-live agent (followup lane, system-note, new `limits.offboard_handover_timeout_seconds`
  default 180s) → `handovers/{agent}/{date}-handover.md`; always-attempted snapshot (the export
  bundle via the shared `_build_agent_bundle` refactor) → `{date}-snapshot.json`; both committed
  in-process via `commit_file`, author = the departing agent; solo/teamless agents skip drive
  commits cleanly. New `POST /mesh/agents/{id}/offboard` (operator-or-internal, operator target
  rejected) = offboard → extracted `_archive_agent_core`; `manage_agent` gains
  `action="offboard"`. ORDER INVARIANT (order-proof test per surface): the offboard attempt runs
  strictly BEFORE `stop_agent(remove_data=True)` on the mesh confirm-delete path, the dashboard
  `DELETE /api/agents/{id}` (the surface where the container is typically live — the real
  handover path), and the CLI `/remove` (bridged via the dispatch loop). Two pre-existing bugs
  fixed in-unit: the dashboard delete now cleans team membership/goals/lead pointer under the
  config lock (was leaving dangling rows), and the CLI `/remove` now wipes the data volume
  (was leaking it). **Review fix (pre-merge):** the lane dispatcher returns the literal
  "(no response)" as an unreachable-agent SUCCESS string — both new consumers (handover doc,
  onboarding intro post) now reject it explicitly, so a dead container degrades to no document
  instead of committing/posting the sentinel as agent-authored text (regression-tested).
  Onboarding: on team join of a running agent, a fire-and-forget intro turn whose reply is
  posted HOST-SIDE into the team channel (writers stay host-side only), then a probationary
  first-task nudge to the lead (operator fallback) — no auto-created task rows; join never fails
  on onboarding hiccups. CLAUDE.md gains Known Constraint #13 (offboard-before-delete).
  Full local suite 8496 passed / 26 skipped / 0 failed.

- **✅ Landed — Phase-4 unit 4: Team Room dashboard (#1224; §6 "who's doing what, thread activity,
  plate per agent" — read-only, zero new write paths or agent-reachable endpoints).** Plate
  snapshot: `CronScheduler._last_plate` records each heartbeat tick's gate computation as a
  BYPRODUCT — `{checked_at, triggered_probes, has_recent_activity, is_default_heartbeat,
  actionable, has_goals, utility_model_configured, dispatched}` — on every tick path
  (actionable dispatch / goals-only dispatch / gated return); manual triggers never write; zero
  extra container calls (pinned by a call-count test); `get_last_plate` is the sole reader and
  agent deletion drops the entry. Composed read: `GET /api/teams/{name}/room` assembles team
  meta + lead, members with busy/idle (lane status), current working task, plate snapshot, and
  recent team threads — all in-process from the injected live objects, every source
  None-guarded, 404 unknown team, auth rides the dashboard router (hosted-mode 401 tested).
  UI: new `room` Team Hub sub-tab (now the team default — the old `work` default was unpinned):
  member cards with Lead badge, busy dot, current task and a plate line; team-goal header;
  scoped thread-activity pane reusing the Workplace Threads read-only pattern with its own
  state; WS refresh rides the existing `queue_changed`/`thread_message` debounce (no new
  polling); the four frozen top-nav tab IDs untouched and now pinned by a dedicated test.
  28 new tests. Full local suite 8524 passed / 26 skipped / 0 failed.

- **✅ Landed — Phase-4 unit 5: streaming/broadcast steer-routing (#1227; the recorded Phase-3
  residual, closed without the escape hatch).** Both surfaces reroute onto the EXISTING
  `LaneManager.deliver_chat` — zero changes to lanes.py, src/agent/loop.py, app.js, channels,
  REPL, or mesh wake. Streaming chat forks busy-aware at the dashboard endpoint: idle keeps the
  direct SSE proxy untouched (true token streaming; the advisory-check race window degrades to
  today's bounded blocking, documented in-code); busy never opens the direct stream — the message
  steers into the RUNNING turn and the reply relays over the existing SSE vocabulary (a
  `text_delta` status preamble, the turn's final text as one `text_delta` chunk — a steered
  reply cannot token-stream by construction — then `done`; no new event types, no frontend
  changes). Broadcast: the found contract (await every reply, concurrent fan-out) is preserved
  exactly — only the per-recipient delivery mechanism changed to `deliver_chat` (busy → steer
  into the running turn, idle → followup lane turn), with concurrency proven by a barrier test
  that would deadlock if dispatch serialized. B1 pinned: a dedicated test asserts the busy path
  never calls the agent's `/chat`//`/chat/stream` directly. Two existing broadcast tests
  re-pointed from transport mocks to `deliver_chat` mocks preserving their original intent.
  Full local suite 8534 passed / 26 skipped / 0 failed.

- **✅ Landed — Phase-4 exit (this PR).** All five units + pre-decisions merged as separate green
  PRs off `main` (#1217/#1218/#1220/#1222/#1224/#1227 + doc-recording PRs + the #1226
  test-isolation hardening). Exit checks on the merged tree (2026-07-11): the `resources`
  template/config key is grep-zero in src/ (round-trips deleted, §8 #16d); the "role is fixed by
  the template" freeze copy is grep-zero (role sits in `_ALLOWED_OVERRIDE_FIELDS`); the dashboard
  delete path calls `teams_store.remove_agent` (dangling-membership bug closed) and the CLI
  `/remove` wipes the data volume (leak closed); offboard-before-delete holds on all three
  delete surfaces (order-proof tests). Wizard v1 was extended, not replaced — no C.1 pair
  applied this phase beyond the deletions above. The phase closes with the consolidated
  adversarial review of ALL Phase-4 PRs (Phase-3 pattern) — findings, if any, land as a
  follow-up fix PR recorded in this ledger.

- **✅ Landed — Phase-4 end-of-phase review + fixes (#1229).** Consolidated adversarial review of
  all Phase-4 PRs on the merged tree — three independent passes (lifecycle/permission/concurrency
  deep-dive with repro scripts; units-2/4/5 correctness + cross-unit integration; phase-wide
  invariant sweep + doc accuracy). The invariant sweep PASSED every family with evidence: both
  goal kinds operator-write-only across every enumerated write path, lead ceiling zero (merge/
  reject handler bodies byte-identical, pinned test body diff-verified unchanged), thread writers
  host-side-only with `post_to_channel` unreachable at every layer, Constraints #1/#5/#6/#13
  conformant, all commits single-authored, ruff clean. Eight confirmed findings, all fixed in
  #1229 with negative-control-verified regression tests:
  1. **CRITICAL — permissions.json lock coverage:** seven PRE-EXISTING endpoints (mesh skills/
     internet/browser + dashboard skills/permissions/wallet-enable) still did bare
     load→mutate→save outside the unit-2 `_config_lock` (repro: 2–3 of 30 concurrent writes
     survive — a lost update can silently undo a permission grant OR revocation). The unit-2
     landed entry's "EVERY load→mutate→save runs under it" claim was FALSE for these seven —
     corrected here; all seven now hold the lock across the full sequence.
  2. **MAJOR — usable-reply gate:** the lane dispatcher has THREE non-success return shapes
     (`__SILENT__`, "(no response)", "dispatch_error: …"); the unit-3 pre-merge fix denylisted
     two at two sites, and the standup post gate rejected none (an unreachable lead would post
     the literal silent token into the team channel daily). One shared `usable_agent_reply`
     predicate now guards all three host-side writers.
  3. **MAJOR — standup cron leaks:** team archive and mesh confirm-delete removed the summary
     job but not the standup job (daily full LLM turn on the former lead + deleted-team channel
     resurrection until next boot). Both surfaces now remove it.
  4. **MAJOR — health auto-restart stale role:** `_try_restart` re-stamped the registry's cached
     role instead of the freshly-loaded config's — the unit-2 regression test proved
     propagation, not source freshness; `edit_agent` role changes were silently reverted by
     crash rebuilds. Fresh config role now wins.
  5. **MAJOR — lead lifecycle:** offboarding a team's lead left `lead_agent_id` dangling (ghost
     lead in the Team Room; boot reconcile recreated the standup job for the archived lead).
     Offboard now clears leadership (+ standup sync + audit); the reconcile skips archived
     leads. Plain archive stays reversible/untouched.
  6. **MAJOR — create-team onboarding gap:** initial members of a new team never got the
     onboarding wake (only later joins did); both create loops now fire it.
  7. **MINOR batch:** `_ConfigLock` fd leak on flock failure; in-flight standup re-checks job
     liveness before posting; busy-path SSE status note ends with a paragraph break; CLAUDE.md
     Constraint #13 clarified (CLI seam precondition) + content-pinned.
  8. **MINOR — RefMoved retry:** offboard drive commits retried (3 attempts) so a concurrent
     main commit can't silently drop a departing agent's handover/snapshot.
  One reviewer finding (plan doc missing the unit-5 landed entry) was already fixed by #1228
  before the review completed — dismissed with verification.
  **Recorded residuals (deliberate, no code change):** offboarding a BUSY agent reliably times
  out the handover turn (FIFO followup queues behind the in-flight task, wait ≤600s) — the raw
  snapshot still commits, so data loss is partial (no distilled narrative); a steer-shaped
  handover is future design work. The config flock blocks the event loop under CROSS-PROCESS
  contention — accepted (critical sections are ms-scale; contention requires a concurrent CLI/
  setup-wizard process). No per-agent offboard/delete serialization beyond the commit retry
  (dual-delete TOCTOU window remains; the garbage-commit half is closed by the reply gate).
  Volume-before-config teardown ordering can leave a config-listed agent with no volume on a
  mid-teardown failure (pre-existing deliberate H12 ordering). `/api/broadcast/stream` remains
  direct-call (the shipped UI fans out client-side to the CONVERTED per-agent stream endpoint;
  only direct API callers are affected — recorded gate, same shape as the original unit-5
  residual). A plain-ARCHIVED (not offboarded) lead keeps its Lead badge in the Team Room —
  truthful display; the operator can unassign. Full local suite 8556 passed / 26 skipped /
  0 failed.

- **✅ Landed — Phase-5 pre-decisions ratified (§8 #17–#24) + build order.** Recon fan-out
  (five parallel read-only passes: pending_actions surface, consequential-action inventory,
  track-record raw material, earned-autonomy mechanics, B3 subsystem behavior) preceded the
  ratification; its corrections are folded into the §8 entries above. The load-bearing ones:
  1. **Durability inversion** — rated `tasks` rows reap at 90d, `work_summaries` at 30d, while
     `audit_log` and Team Drive git history never reap; the track record must be a rating-time
     append-only ledger (#18), not a live-query composition.
  2. **`hand_off` to a stopped agent reports full success today** — `/mesh/wake` 200s
     fire-and-forget, the background dispatch swallows the connect failure
     (`SILENT_REPLY_TOKEN`), the task sits `pending` forever, and the heartbeat safety net the
     code comments assume does NOT fire: the `pending_tasks` probe reads the legacy blackboard
     `tasks/{agent}` prefix, not the durable tasks store (#24 prereq iii).
  3. **Archive is leakier than B3 recorded** — boot `_start_agents()` resurrects archived
     agents' containers on every mesh restart (no status filter) and `_reconcile_heartbeats()`
     lacks the archived filter its siblings have (#24 prereqs i–ii).
  4. **`POST /mesh/notify` has no permission check at all** — any agent can broadcast to every
     paired human on every channel (#17 closes this).
  5. **The confirm chain is human-origin-only by construction** (live request AND stored
     proposal) — no agent, lead included, can release a hold; preserved verbatim under #19.
  6. Minor: `__SILENT__` leaks raw into channel replies + dashboard chat for stopped agents;
     CLI `openlegion confirm`/`cancel` send no auth headers (mock-tested only, likely broken
     live); two stale comments point approvals UI at a System-panel removed in PR #1044; dead
     `pending_actions` kwarg in `create_dashboard_router`.
  A companion review doc (`docs/plans/2026-07-11-hands-off-teams-phase5.md`) records the
  before/after system review, the human-intervention map, and the permanent-touchpoints list;
  its decisions are THESE §8 entries (one numbering, one source of truth). An earlier draft of
  that review was PR #1231, closed unmerged in favor of this PR.
  **Build order (each unit a separate PR off `main`, full-suite gate per merge; personal
  pre-merge diff review on permission-surface and data-loss-critical units):**
  - **U0 — hardening prereqs** (#24 i–iii + recon minor items 6): archive boot-resurrection
    fix, heartbeat-reconcile archived filter, pending_tasks probe repoint, `__SILENT__` leak
    filters, heartbeat error-string reply gate, CLI confirm auth fix, stale comments, dead
    kwarg.
  - **U1 — track record** (#18): store + rating-trust rule + Personnel-File composition + read
    endpoint.
  - **U2 — positive feedback push** (#23).
  - **U3 — policy engine core** (#17 + #19 skeleton): tier registry, `evaluate()`,
    `config/policy.yaml`, held-actions store evolution absorbing delete-confirm (C.1 row 6
    completes HERE), notify/connectors/wallet classified `allow_audit` (no behavior change by
    default). *[diff review]*
  - **U4 — auto-merge consumer** (#20). *[diff review]*
  - **U5 — earned-autonomy completion** (#19): probation preset, lead advisory
    probe/recommend, recent-autonomous-actions audit view. *[diff review]*
  - **U6 — delivery loops** (#22): escalation ladder + goal-coverage probe.
  - **U7 — lead budget allocation** (#21). *[diff review]*
  - **U8 — hibernation** (#24). *[diff review]*
  Order rationale: U1 first because every autonomy decision consumes it; U3 ships with
  behavior-preserving defaults so U4 (the highest single-leverage unlock) lands early; U5–U7
  close the daily-path loops; U8 last — it advances cost/scale, not hands-off, and its
  prerequisites land in U0. Branch prefix `feat/p5gov-*` (`feat/phase5-*` is taken by an
  unrelated browser work stream).

- **✅ Landed — U0 hardening prereqs (#1239).** All eight U0 items per the build order above
  (§8 #24 prereqs i–iii + the recon minor items): archived agents survive a mesh restart
  (boot skip + heartbeat-reconcile filter, each pinned); the heartbeat pending-tasks probe
  gains a durable-tasks-store source (`pending_tasks_fn` seam →
  `Tasks.count_pending_for_assignee`, the H5 backlog-cap counter reused); `__SILENT__` /
  `"(no response)"` / `dispatch_error:` shapes are gated by `usable_agent_reply` at the two
  human-facing leaks (channels suppress, dashboard chat substitutes a readable fallback);
  `heartbeat_dispatch` failures now carry the `dispatch_error:` prefix the reply gate rejects;
  CLI `confirm`/`cancel` route through the dashboard pending proxy (the bare mesh POST could
  never pass operator-or-internal + human-origin); stale approvals-UI comments corrected; dead
  `create_dashboard_router(pending_actions=)` kwarg deleted. **Implementation corrections
  discovered:** (1) the blackboard `tasks/*` namespace is NOT legacy — 20+ template ACLs,
  `claim_task`/`watch_blackboard` docs, operator-ceiling defaults, and the SPA all use it, so
  the probe keeps BOTH sources (durable store authoritative, blackboard scan additional);
  full removal waits until the template work-queue pattern migrates to the durable store.
  (2) A second pre-existing test-ordering flake found: `test_channels.py
  TestTelegramStop::test_stop_calls_shutdown_and_logs` fails whenever `test_cli_commands.py`
  precedes it in one invocation (caplog captures nothing) — reproduced on clean main; treat
  as known flake until the logging-state leak is fixed. Full suite 8615 passed / 26 skipped /
  sole failure = that flake, green on isolated rerun.

### PR ledger — Phase 1 (as of 2026-07-07)
| PR | Unit | CI |
|---|---|---|
| #1186 | post-merge review fixes for the #1180–#1185 stack | green |
| #1187 | per-call model tiering hook (Phase-0 residual a) | green |
| #1188 | mesh→agent bearer auth (Phase-0 residual b) | green |
| #1189 | TeamStore — team as first-class entity (unit 1) | green |
| #1190 | goals → Team store, blackboard goals/ path deleted (unit 1b) | green |
| #1191 | team budget envelope, B4 semantics (unit 2) | green |
| #1192 | solo = team-of-one (unit 3) | green |
| — | TEAM.md cap (unit 4) + this doc update | this PR |

### PR ledger — Phase 2 (as of 2026-07-07)
| PR | Unit | CI |
|---|---|---|
| #1194 | Phase-2 pre-decisions ratified (C.3-a replace, C.3-e keep, §8 #1 reconfirmed) | green |
| #1195 | Team Drive — mesh-hosted git, smart-HTTP, review-before-integrate, quota (unit 1, incl. review-fix commit) | green |
| #1196 | Team Threads — durable store replacing message_log deque + inbox back-edge feed (unit 2) | green |
| #1197 | ask_teammate — mesh-brokered inline Q&A, asker-billed, steer-delivered (unit 3) | green |
| #1198 | blackboard → signals only — hand_off data + artifacts move to the Team Drive (unit 4) | green |

### PR ledger — Phase 3 (as of 2026-07-11)
| PR | Unit | CI |
|---|---|---|
| #1200 | Phase-3 pre-decisions ratified (B1 → §8 #10, B2 → §8 #11) + build order | merged |
| #1202 | B2 coordination-vs-work spend split at the LLM proxy (unit 1) | merged |
| #1206 | plate-gated agenda loop replaces heartbeat suppression (unit 2) | merged |
| #1208 | priority steer lane — busy chat steers the running turn with reply back-edge (unit 3) | merged |
| #1210 | budget-governed initiative — budget-aware agenda context + goal-driven self-tasking (unit 4) | merged |
| #1213 | end-of-phase review fixes — heartbeat single-lane, steer-reply integrity, truthful budget surface | merged |

*CI note:* workflow runs on app-authored PRs in this repo require the maintainer's one-click
approval and did not auto-run; every unit was landed on a green full local suite (the exact
`make test` command CI runs) + a post-rebase touched-suite verification. Phase-1/2 units and
Phase-3 units 1–2 additionally got per-unit adversarial review (one cross-unit integration bug
was caught this way in Phase 2: unit 4's artifact endpoints called the now-async `_drive_repo`
without `await` after unit 1's review-fix — fixed pre-merge). Phase-3 units 3–4 landed on quick
per-unit gates; the phase closes with a consolidated adversarial review of ALL Phase-3 PRs —
findings, if any, land as a follow-up fix PR recorded in this ledger.

### PR ledger — Phase 4 (as of 2026-07-11)
| PR | Unit | CI |
|---|---|---|
| #1217 | Phase-4 pre-decisions ratified (§8 #12–#16) + build order | merged |
| #1218 | Lead role core — designation, advisory verdicts, host-published standup (unit 1) | merged |
| #1220 | hiring wizard v2 — config file-lock (B-pre #2), role unfrozen, goals-driven hiring (unit 2) | merged |
| #1222 | onboarding + offboarding-with-handover — three delete surfaces converge (unit 3) | merged |
| #1224 | Team Room — plate snapshots, composed room read, default Team Hub sub-tab (unit 4) | merged |
| #1226 | test-isolation hardening — set_cron schema security pin vs registry pollution | merged |
| #1229 | end-of-phase review fixes — permissions lock coverage, reply gate, standup leaks, lead | merged |
| #1227 | streaming/broadcast steer-routing — busy-aware SSE fork + broadcast via deliver_chat (unit 5) | merged |

### PR ledger — Phase 5 (as of 2026-07-11)
| PR | Unit | CI |
|---|---|---|
| #1232 | Phase-5 pre-decisions ratified (§8 #17–#24) + build order + companion review doc | merged |
| #1231 | earlier draft of the companion review — closed unmerged, superseded by #1232 | closed |
| #1239 | U0 — hardening prereqs (§8 #24 i–iii + recon minor items) | merged |

### YOU ARE HERE → Phase 5

**Phase 3 (the workday) is COMPLETE.** All four units landed as separate green PRs off `main`,
each completing its C.1 deletion in-phase. The C.4 phase-exit check ran on the merged tree
(2026-07-11): grep-zero in `src/` for `force_llm`, `x-force-llm`, `HEARTBEAT_OK`,
`no_heartbeat_rules`, `_is_heartbeat_empty`, `_HEADING_OR_EMPTY_RE`; `is_default_heartbeat`
survives only as plate-gate/message input (by design), never as skip logic.
- **B2 spend split** (#1202) — coordination vs work is model-keyed at the LLM proxy (requested
  model == deployment `llm.utility_model`), `kind` column on `usage`, enforcement reads filter
  `kind='work'`, coordination has its own daily cap with a 0-valid kill-switch (§8 #11).
- **agenda loop** (#1206) — plate-gated agenda dispatch REPLACED heartbeat suppression (C.1 row
  5 closed): actionable plates always escalate, goals-only plates escalate only with a configured
  utility model, truly-empty plates never reach the LLM.
- **priority steer lane** (#1208) — busy chat steers the running turn with a real reply back-edge
  (never a parallel turn, B1/§8 #10); dashboard chat rides the lane path; the
  steer-after-last-drain race is closed deterministically.
- **budget-governed initiative** (#1210) — agenda context surfaces work + coordination budget;
  goal-driven `hand_off`-to-self gated on plate capacity + budget headroom; `set_cron` cadence is
  budget-governed, not user-gated. Standing goals stayed operator-write-only.

§8 #1 stands reconfirmed: git-Drive-first, **raw shared `/team/scratch` is NOT built** (deferred,
unratified) — do not build it without an explicit user decision; any future scratch ratification
MUST resolve the SandboxBackend story at the same time (recorded in §8 #9).

**Phase 4 (org model) is COMPLETE.** Pre-decisions ratified as **§8 #12–#16** (#1217), all five
units landed as separate green PRs off `main`, exit checks green (see the Phase-4 exit entry):
- **Lead role core** (#1218) — `teams.lead_agent_id` (team data, zero permission elevation),
  operator-only assignment with same-transaction integrity clears; lead-only ADVISORY drive-review
  verdicts (merge/reject gates untouched); lead-duty plate probe; host-published standup via
  `CronJob.post_to_channel` (unreachable from every agent-facing cron surface); Constraint #1/#12
  amendments landed with their pinning test.
- **hiring wizard v2** (#1220) — B-pre #2 CLOSED first (sidecar `config/.config.lock` flock +
  atomic agents.yaml writes covering every config writer); `role` unfrozen in the fleet-override
  allowlist (+ the latent `_apply_template` ignore-gap); health-monitor re-register roster
  staleness fixed; dead `resources` key deleted; Team Hub goals-driven hire entry in the
  wizard-v1 shape (operator-tier, UI never calls create APIs).
- **onboarding + offboarding-with-handover** (#1222) — offboard = bounded handover turn on the
  live agent + export-bundle snapshot, committed in-process to `handovers/{agent}/` (author =
  departing agent) STRICTLY before volume destruction on all three delete surfaces (order-proof
  tests); dashboard-delete dangling-membership bug and CLI volume-leak bug fixed; onboarding
  intro host-published + lead/operator first-task nudge; Known Constraint #13.
- **Team Room** (#1224) — read-only: plate snapshots as a heartbeat-gate byproduct, one composed
  `GET /api/teams/{name}/room`, default `room` Team Hub sub-tab; frozen top-nav IDs untouched.
- **streaming/broadcast steer-routing** (#1227) — the Phase-3 residual closed: busy streaming
  chat steers the running turn (reply as one SSE chunk over the existing event vocabulary), idle
  keeps true token streaming; broadcast fans out per-recipient through `deliver_chat` preserving
  its await-all concurrent contract; B1 pinned; loop.py/lanes.py/frontend untouched.
The consolidated end-of-phase adversarial review of all Phase-4 PRs runs at phase close
(Phase-3 pattern) — findings land as a fix PR recorded in the Phase-4 ledger.

**Phase 5 (governance at scale) is IN PROGRESS** — pre-decisions ratified as **§8 #17–#24**
(2026-07-11) with the U0–U8 build order recorded in the Phase-5 landed entry above: action-tier
policy engine (#17), durable track record + rating-trust rule (#18), earned autonomy with
human-only hold release + lead advisory (#19), kernel-executed auto-merge consuming lead
verdicts (#20), lead budget allocation within the human envelope (#21 — activates the item #12
reserved), delivery loops + goals carve-out declined again (#22), positive feedback push (#23),
and the **B3 hibernation scoping design (#24 — the standing B3 gate is SATISFIED; hibernation
builds last, U8, after its prerequisite fixes land in U0)**. The companion review doc
`2026-07-11-hands-off-teams-phase5.md` maps the human-intervention points and permanent
touchpoints. Standing gates carried forward: a goals agent-write carve-out (team or standing)
remains UNRATIFIED (declined again in #22) — do not build one without an explicit user
decision; Personnel-file *import* is unblocked (B-pre #2 fixed) but unscheduled; raw shared
`/team/scratch` stays unratified (§8 #1, sandbox story per §8 #9); Constraint #12 absolute —
leads/agents gain no permission carve-out, policy thresholds are OPERATOR policy.

**Handoff note:** this doc is the source of truth — a fresh session can continue from here without
this session's chat history. Read §5 (keep/refactor/remove), Appendices A–C, §8 (ratified
decisions), and the Phase-1/2/3/4 landed entries above (they record implementation corrections,
recorded deviations, and review findings the appendices don't).

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
- **Dual lanes** — ✎ **SUPERSEDED by B1 (ratified §8 #10): do NOT build this sketch.** The
  parallel interactive queue/worker keyed `(agent, lane_kind)` described here would create a second
  concurrent turn and corrupt shared agent state (see B1). The Phase-3 build is a **priority steer
  lane**: `try_steer` injection into the running turn + a reply back-edge, generalizing the
  `ask_teammate` busy/idle fork (`server.py:1877-1933`). Kept for the record of what was considered.
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
| ~~Blackboard `output/*` + `artifacts/*` payload writers in `hand_off` (`coordination_tool.py:305`)~~ ✅ **DONE (unit 4)** — writers deleted, grep-zero; moved to Team Drive `POST /drive/artifacts` | Team Drive artifact store | 2 | Handoff data written to two places |
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
   atomic temp+rename before adding import. ✅ **FIXED (Phase-4 unit 2, #1220)** — recon widened
   the target to `config/agents.yaml` (bare truncate-writes everywhere); the sidecar
   `config/.config.lock` flock + atomic writes now cover every agents.yaml/permissions.json
   writer (§8 #16a).
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
