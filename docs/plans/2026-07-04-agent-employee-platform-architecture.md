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
- **Team budgets** (`src/host/costs.py`): in-memory advisory aggregation → **durable, pre-flight
  enforced team envelope**; the lead allocates per-agent budgets within it.
- **Lanes** (`src/host/lanes.py`): in-memory queues that strand work on restart → **rehydrate from
  the durable `tasks` table on boot**; add a **second lane** per agent (deep-work + interactive) so
  a 4h task no longer head-of-line-blocks a 30-second question.
- **One model per agent → per-call model tiering.** The proxy + failover already accept a per-call
  `model` override; add caller-side policy so coordination traffic (agenda ticks, thread/standup
  digests, summaries) runs on cheap models and deep work on strong ones. Makes the "employees talk"
  layer affordable.
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
  `operator_tools.py`, `coordination_tool.py`, `cli/*`, `orchestration.py`, `mesh.py`, …):
  `can_manage_projects` validator, dual `tasks.project_id` emission, `target_kind="project"`,
  blackboard `projects/{name}/` prefix, `*_project` tool aliases, `config/projects` symlink. Rename
  to **team** natively everywhere; one name, one namespace.
- **`src/host/team_migration.py`** (PROJECT.md→TEAM.md + symlink startup migrator) — delete.
- **`src/host/orchestration_migration.py`** (legacy blackboard-task → tasks-table migrator) —
  delete.
- **All lazy schema-migration archaeology.** `orchestration.py` `PRAGMA table_info` + `ALTER`
  chains and `memory.py` idempotent `ADD COLUMN` paths collapse into **clean canonical `CREATE
  TABLE` at `schema_version = 1`**. Adopt an explicit `PRAGMA user_version` going forward so future
  migrations are principled instead of introspection-guessed.
- **Legacy shared-browser display `:99` / port `:6080`** — superseded by per-agent `:100..:163`.
  Remove the dead surface.
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

## 8. Open decisions for the lead

1. Ratify the **security tradeoff ledger** (§4) — specifically the shared `/team/scratch` volume and
   the wallet/credential-agent exclusion from it.
2. Ratify amending **Constraint #1** ("no router hierarchy," leads permitted) and **Constraint #12**
   (leads get no permission ceiling).
3. Ratify the **cost-philosophy inversion** (budget-governed autonomy replacing activity
   suppression) and the **designated cheap-model tier** for coordination traffic.
4. Confirm the **destructive-cleanup mandate** (delete all migration/shim/lazy-migration code; reset
   to canonical v1 schemas) is authorized given no users/data.
```
