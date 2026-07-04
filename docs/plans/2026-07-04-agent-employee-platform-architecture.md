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

## 8. Open decisions for the lead

1. Ratify the **security tradeoff ledger** (§4) — specifically the shared `/team/scratch` volume and
   the wallet/credential-agent exclusion from it.
2. Ratify amending **Constraint #1** ("no router hierarchy," leads permitted) and **Constraint #12**
   (leads get no permission ceiling).
3. Ratify the **cost-philosophy inversion** (budget-governed autonomy replacing activity
   suppression) and the **designated cheap-model tier** for coordination traffic.
4. Confirm the **destructive-cleanup mandate** (delete all migration/shim/lazy-migration code; reset
   to canonical v1 schemas) is authorized given no users/data.

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
```
