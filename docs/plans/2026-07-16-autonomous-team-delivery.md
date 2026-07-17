# Autonomous Team Delivery — Architecture & Roadmap

**Date:** 2026-07-16 · **Updated:** 2026-07-17
**Author:** bic
**Status:** SIGNED OFF with an **effectiveness-first amendment** — Phase 1 (leadership + goal +
observability loop) has **shipped**; the Phase 0 safety substrate shipped **partially** (see
§7 Implementation status below). The original proposal is preserved unedited from §0 onward as
the design of record.
**Surface:** the operator ↔ team goal-delivery loop across `src/host/` (teams, cron, runtime,
costs, policy, drive, orchestration, chain_watcher), `src/agent/` (loop, builtins), and
`src/shared/operator_playbooks.py`.
**Mandate:** decide the architecture that lets *the human talk to the operator → the operator
creates teams → the fleet autonomously does the work* (usually a business goal), with the human
supplying input/credentials/approvals **only when genuinely needed** — human-in-the-loop as the
exception, not the default.

> Line numbers below were verified during the 2026-07-16 review (a code-verified engineering pass +
> an independent adversarial second-pass review). Treat them as anchors; they may drift.

---

## 0. TL;DR

The three-layer model is right, but its **realization** must change, and a **safety substrate must
land first**:

1. **Mesh Steward** — a deterministic, host-side, *team-scoped* state machine (detection, leases,
   circuit breakers, signal routing). No LLM, no container.
2. **Team Lead** — a **dedicated, purpose-built coordinator agent** (own prompt/identity/reserved
   reasoning capacity) under the *ordinary-member* security model. **Not** a random promoted worker.
   Mandatory for non-trivial teams; solo/small teams self-lead with a promotion threshold.
3. **Operator** — the single *logical* human interface, backed by a durable exception queue +
   failover + sharded supervision so it is not a single *physical* bottleneck.

**Blunt conclusion (endorsed on the second-pass review):** fixing null leadership alone just makes the current
machinery run more often — *"without stronger safety, composition, and acceptance layers, it may
only make the system fail faster and more expensively."* Therefore **Phase 0 = safety substrate**,
before any unattended external action.

---

## 1. Problem: what breaks the autonomous loop today

Verified current-state findings (severity-ordered):

1. **The whole stewardship machinery is lead-gated, and the operator can't install a lead.**
   Goal-coverage, blocked-task escalation (rung 3), review verdicts, and standups all begin with
   `led_team(agent)` and return nothing for non-leads (`runtime.py:257,2199,2213,2264`;
   standup `runtime.py:2619`). `create_team` leaves `lead_agent_id` NULL (`teams.py:157,266`;
   `server.py:6639`); `set_lead` rejects the operator (`teams.py:470`). A lead *endpoint* exists
   (`server.py:7934`) but the operator has **no tool** for it (`cli/config.py:1806`; no `MeshClient`
   setter). ⇒ operator-built teams get **zero** stewardship; a leaderless team's goal drift reaches
   nobody.

2. **A promoted worker can't reliably steward even if appointed.** The roster has no `is_lead`
   field (`server.py:4401`); normal worker prompts fetch per-agent standing goals, not the team
   `north_star`/`success_criteria` (`loop.py:2013,2042`); team goals appear only in a one-time
   onboarding message (`server.py:6716`); and **a busy agent's heartbeat is skipped**
   (`loop.py:2538`) — so a worker-lead loses its stewardship lane exactly when it is doing work.

3. **No safety substrate for unattended operation.**
   - Team budgets **default to unlimited** (`teams.py:13`; `server.py:9752`); the LLM operator can
     replace/clear the envelope with **no verified-human provenance** (`operator_tools.py:2435`);
     team-envelope enforcement **fails open** on a store read error (`costs.py:470`); coordination
     spend sits outside envelopes on a separate $2/day cap (`costs.py:349`; `limits.py:240`).
   - Generic `http_request` (POST/PUT/PATCH/DELETE, `loop_exempt`, `http_tool.py:383,393`) and
     browser actions **bypass the action-policy engine**, which classifies only 6 action kinds
     (`policy.py:162,165,174`). ⇒ agents can publish/email/purchase/delete externally with no gate.
   - `request_credential` swallows mesh failures and still returns `success: true`
     (`vault_tool.py:153`) ⇒ autonomous work can park silently forever.

4. **`set_goal` doesn't propagate.** It persists + emits `team_updated` (`teams.py:393`;
   `server.py:7893`) but does not wake the team or update worker prompts — a redirect of a busy team
   is inert to in-flight work.

5. **No composition or acceptance.** `complete_task` carries no result (`coordination_tool.py:769`);
   `await_task_event` waits on one task only (`operator_tools.py:1162,1304`); `dependencies` are
   stored but not enforced (`orchestration.py:752,1046`); `success_criteria` are strings never
   evaluated — the only runtime use checks "field present + ≥N open tasks" (`runtime.py:264,276`).
   ⇒ the fleet can be **busy but not effective**.

6. **Weak circuit breakers.** The loop detector catches identical calls **within one turn** and
   **resets each heartbeat** (`loop_detector.py:28`; `loop.py:2550`); parent-chain depth is capped
   but **sibling fan-out is not** (`orchestration.py:138,1046`); default task/round limits are
   explicitly *not* cost controls (`limits.py`). The operator monitor playbook is orphaned dead code
   (`operator_playbooks.py:496,597`).

**Correctly nuanced (don't overstate):** monitoring is not fully pull-only — every active agent gets
a heartbeat and the operator has a 15-min schedule (`runtime.py:2452`), but it is **capped**
(~8 tool calls / 3 snapshots / 3 drill-downs, `config.py:1970,1999,2014`) and skipped when busy, so
it **samples rather than supervises** a fleet. The chain watcher does deliver terminal
operator-rooted outcomes; verification currently runs *after* the user is notified
(`runtime.py:1622,1810`).

---

## 2. Architecture decision

### 2.1 Three responsibilities, corrected realization

- **Mesh Steward — deterministic host state machine, heartbeat-independent.** Owns: liveness/leases;
  deadline/retry/budget/dependency/queue checks; missing-plan / unowned-criterion / stale-progress /
  goal-version-mismatch detection; durable signal routing to the current coordinator or the
  operator; invariant enforcement + circuit breakers. It **detects and routes**; it does **not**
  decide strategy or judge ambiguous outcomes. Runs a real sweep over every active team — *not* just
  removing `led_team` gates.

- **Team Lead — a dedicated coordinator agent, ordinary-member security.** Purpose-built prompt +
  explicit lead identity (`is_lead`, durable goal-version contract) + reserved reasoning capacity
  separate from worker execution + team task/result/artifact/dependency visibility + a least-
  privilege coordinator toolset + a health lease with automatic replacement + a self-approval
  prohibition. **Mandatory for non-trivial teams; a solo worker may self-lead in an explicit solo
  mode; the mesh promotes/spawns a dedicated coordinator once size/fan-out/risk crosses a
  threshold.** Realized as an *ordinary member* (no privileged agent type); idle cost controlled by
  hibernation. Rationale: §1 findings 1–2 show a random worker-hat can't be trusted to steward.

- **Operator — single logical interface, not single physical bottleneck.** Keep one operator
  identity + conversation, but back it with a **durable prioritized supervisory/exception queue,
  lease-based failover, checkpointed state, and horizontally shardable supervisory workers.** Leads
  handle routine per-team management; the operator consumes cross-team exceptions.

This preserves Constraint #1 (no *router* hierarchy — the coordinator plans and stewards; work still
flows peer-to-peer; it is not a message router) and Constraint #12 (operator remains the only
trust-tier bypass).

### 2.2 Guiding principle — autonomous resolver + typed human fallback

Every decision point gets an autonomous default within a guardrail, and a **typed** escalation only
when the guardrail is hit:

| Decision point | Autonomous default | Human only when… |
|---|---|---|
| Budget / spend | operator reallocates **within** an immutable human cap | the **cap** is hit |
| Deliverable acceptance | lead verdict + `success_criteria` eval + trust ladder | eval ambiguous / low-confidence |
| Irreversible / external action | broker auto-allows reversible; holds irreversible | **irreversible / high-risk** tiers |
| Missing credential | typed `credential` request | always (cannot be synthesized) |
| Strategic fork | operator decides within the stated goal | genuine goal ambiguity |

This table **is** the "needs-human" contract; building it is what delivers "human only when needed."

---

## 3. Phased roadmap

### Phase 0 — Safety substrate  ·  gate: no unattended external mutation before this ships
- Finite **deny-by-default** team + fleet budgets; an **immutable verified-human ceiling** distinct
  from an operator-controllable working budget; **remove fail-open** enforcement (`costs.py:470`);
  provider-side caps as an independent backstop.
- A **consequential-action broker** in front of HTTP/browser/connector/wallet: read/write split,
  destination + action allowlists, dry-run/preview, idempotency + side-effect receipts, tiers
  (irreversible / financial / external-visible / reputation-legal). Route generic
  `http_request`/browser mutations through it.
- Global emergency pause + scoped team/workflow pause + scoped revocation.
- Unified typed **needs-human taxonomy** (approval scope, expiry, resume token, alternatives tried,
  exact requested action) + dedupe.
- Fix false-success credential requests (`vault_tool.py:153`); durable audit + acknowledgements +
  dead-letter monitoring.

### Phase 1 — Team invariants + stewardship  (closes the loop, safely)
- **Atomic `create_team`**: create/attach members, assign-or-spawn a qualified coordinator, install
  its lease, set goal v1, enqueue its planning turn. A non-solo team never commits leaderless.
- Coordinator role legibility (`is_lead` + durable goal contract) + explicit coordinator toolset +
  operator `set_team_lead` / spawn-coordinator tool. Lead lease / health / **automatic failover**
  (the steward, not the failed lead, replaces it); self-approval prohibition.
- **Goal-as-event**: versioned goal records + `goal_changed` + coordinator wake + plan
  reconciliation + member acknowledgement; supersede stale old-goal work.
- Team-scoped steward sweep live for every active team; solo self-lead + promotion threshold.

### Phase 2 — Composition + acceptance  (⇐ "autonomous delivery" becomes a defensible claim only here)
- **Typed result-carrying completion**: `complete_task` writes summary + artifacts + metrics +
  evidence + external-action receipts + residual risks.
- **Dependency enforcement + fan-in/join** (all / quorum / first-success) + deadlines + cancellation
  propagation + idempotency keys + retry/fan-out budgets + result aggregation; result-rich
  `workflow_snapshot`.
- **`success_criteria` evaluator + acceptance state machine**: compile criteria into objective
  validators where possible; independent reviewer agents for ambiguous/high-risk work; verify
  **before** declaring success (reverse today's notify-then-verify order).

### Phase 3 — Portfolio autonomy
- Operator durable exception queue + HA execution; cross-team dependencies + resource/budget
  arbitration; within-cap autonomous budget allocation; strategic/spend/credential approval
  resumptions; goal-level outcome memory + retrospectives (canaried, reversible).

### Phase 4 — Controlled unattended rollout
- Shadow → internal low-risk → canary teams with tight caps. Chaos tests: dead lead, dead operator,
  credential expiry, partial event failure, budget-store outage, task explosion. Graduated action
  autonomy by independently-verified track record; explicit SLOs + rollback criteria.

---

## 4. Top unattended risks (must be covered before rollout)
Runaway spend (unlimited defaults + operator-mutable envelope + fail-open); doom loops / task
explosion (per-turn-only detector, unbounded sibling fan-out); "busy but wrong" goal misalignment
(no criteria evaluation, goals not in worker prompts); irreversible external actions (unbrokered
HTTP/browser); silent failure (false-success credentials, leaderless teams, verify-after-deliver);
rogue / overloaded / failed lead (new critical dependency — needs leases, failover, self-approval
prohibition, esp. since `auto_merge` can integrate an approved review host-side, `auto_merge.py:188`);
operator SPOF (sampling caps prove a single container can't supervise a large fleet).

## 5. Open questions to confirm at sign-off
1. **Coordinator cost model** — dedicated container per non-trivial team (hibernated when idle) vs.
   a lighter reserved-capacity lane on an existing member. Threshold definition (size/fan-out/risk).
2. **Immutable human ceiling mechanics** — how "verified-human" provenance is proven (dashboard-only
   write? signed action?) given the operator is an LLM.
3. **Action-broker scope for v1** — full HTTP/browser read/write classification is large; do we ship
   an allowlist-first minimal broker in Phase 0 and expand, or block unattended external writes
   entirely until the full broker lands?
4. **Sequencing preference** — safety-first (this doc) vs. leadership-loop-first (faster demo, but
   the "fail faster and more expensively" risk the second-pass review flagged).

## 6. Scope of this document
This began as a design proposal for **sign-off only**. It has since been signed off with the
effectiveness-first amendment recorded in §7, and the first increments have shipped as scoped PRs
(worktrees, `ruff check`, full-suite gate, squash-merge, committed as bic). Everything from §0
through §5 is preserved as the original design of record; §7 tracks what is actually built.

---

## 7. Implementation status (2026-07-17)

**Direction amendment — effectiveness first.** The pre-revamp system already ran agents behind two
real-world guardrails: **per-agent/team spend caps** and **Docker/microVM isolation + SSRF egress
filtering**. Given that floor, we chose to prioritise closing the *delivery loop* (leadership +
goal propagation + observability) over building the full Phase-0 action broker up front. Blocking
individual outbound actions and an immutable human ceiling remain planned; they are follow-ups, not
prerequisites, for the current usage. This resolves §5 open questions 3 (action-broker scope —
ship the loop first, broker later) and 4 (sequencing — leadership-loop-first).

### Shipped

**Phase 1 — team invariants + stewardship loop** (`#1266`–`#1276`):
- Every non-solo team gets a lead: auto-appointed when a team reaches ≥2 non-operator members (on
  create and on add), and **automatically re-appointed on every departure path** (remove /
  offboard / move-eviction / delete). Operator `set_team_lead` tool added.
- **Goal-as-propagation**: `set_team_goal` persists north star + success criteria, mirrors them
  into a section-scoped `## Goal` block in `TEAM.md`, and pushes so running members pick up the
  redirect on their **next turn**. `update_team_context` / brief writers are section-scoped and no
  longer clobber the goal.
- **Operator readback + acceptance signal**: `inspect_teams` returns goal/criteria/budget;
  `inspect_team_spend` reports per-team spend; `inspect_agents(profile)` adds budget headroom +
  track record; `assess_team_progress` bundles success criteria with the team's actual completed
  outputs for an evidence-grounded verdict. Dashboard **Team Room** surfaces north star, criteria,
  probe names, and per-member current + blocked work.
- Runtime journey guard `test_operator_team_journey_composes_end_to_end` exercises the composed
  loop (create → lead → goal → propagate → budget → context-update → departure → re-appoint).

**Phase 0 — partial** (`#1263`, `#1264`):
- `request_credential` returns an honest failure envelope instead of false success (`vault_tool.py`).
- Configurable **fail-closed** posture for the team-budget envelope
  (`OPENLEGION_TEAM_ENVELOPE_FAIL_CLOSED`, default fail-open to preserve today's behaviour).

### Deferred (planned, not yet built)
- Consequential-action broker in front of HTTP/browser/connector/wallet (Phase 0 core).
- Immutable verified-human spend ceiling distinct from the operator-controllable working budget.
- Global emergency pause / scoped revocation kill switch.
- Dedicated purpose-built coordinator agent (today's lead is a promoted ordinary member).
- Phase 2 composition/acceptance (typed result-carrying completion, dependency fan-in/join,
  `success_criteria` evaluator) and Phases 3–4.

### Verified boundary
The loop is proven at the **mesh/integration layer** (endpoint-level journey test, hand-reviewed
diffs, full-suite gate green on each PR). The **Docker-container + live-LLM layer** — a real worker
reading the goal from its live prompt and acting on it — is validated by a local `openlegion start`
run, not by the automated suite.
