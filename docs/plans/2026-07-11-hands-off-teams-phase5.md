# Hands-Off Teams — Phase-5+ Review, Ratifications & Build Order

**Date:** 2026-07-11
**Status:** Ratified (extends `2026-07-04-agent-employee-platform-architecture.md` — its Phase 5 plus three delivery loops it does not name)
**Author:** System review with user ratification
**Parent plan:** `docs/plans/2026-07-04-agent-employee-platform-architecture.md` (referenced below as "the parent plan"; its §8 decisions ledger ends at #16 — this doc continues the numbering at #17)

> **Scope of this doc.** A review of how the system coordinated *before* the team system, a map
> of every human-intervention point that remains *after* Phases 0–4, and the ratified path to
> the parent plan's own north star: the human reviews **by exception**, not per-task. It records
> three new decisions (#17–#19) and a prioritized build order. It changes no code by itself.

---

## 1. Goal statement

**User-friendliness = the team gets everything done for the project/business without human
intervention, except where a human is genuinely irreplaceable.** The parent plan's Phase 5
("governance at scale") provides the substrate — action tiers, track record, earned autonomy,
hibernation — but three *delivery loops* that actually remove the human from the daily path are
not in its written scope (§6 below). This doc ratifies and sequences both.

"Hands-off" does not mean removing human walls. It means converting each wall from a
**per-action gate** into a **policy-governed exception path**, with the audit trail deciding when
policy loosens ("as models strengthen, loosen policy — never rebuild structure").

---

## 2. System review: before the team system → now

Pre-team history is not in git (the tree was squashed at the project→team rename, #1185); it is
reconstructed from the parent plan's §5/Appendix A–C ledgers and the archived design docs
(`docs/plans/archive/`).

| Concern | BEFORE (pre 2026-07) | NOW (Phases 0–4, PRs #1180–#1229) |
|---|---|---|
| Grouping unit | "project" bolt-on: YAML glob + blackboard key prefix + `TEAM.md` mount | first-class `TeamStore` (`data/teams.db`): identity, membership, goals, budget, lead |
| Deliverables / handoff data | blackboard `output/*` + `artifacts/*` (256KB cap, 6k serialization tax) | Team Drive — mesh-hosted git, review-before-integrate, quota |
| Message history | in-memory `message_log` deque(10000), lost on restart | durable `ThreadStore` (`data/threads.db`): channel / task / dm threads |
| Back-edge events | blackboard `inbox/{agent}/task_event/` | Team Threads event feed (`GET /mesh/agents/{id}/task-events`) |
| Standing goals | blackboard `goals/{agent_id}` | TeamStore per-agent goals + team `north_star`/`success_criteria` |
| Team budget | dead code (in-memory dict, zero callers) | enforced durable envelope + coordination/work spend split (B2) |
| Heartbeat | cost suppression ("HEARTBEAT_OK", never take initiative) | plate-gated agenda loop; goal-driven initiative within budget |
| Teammate question | full task handoff + pushback-reissue dance | `ask_teammate` inline Q&A, asker-billed via mesh-held window |
| Interactive during task | bare steer queue, no reply | priority steer lane with reply back-edge (`deliver_chat`) |
| Lead / org model | none | lead as team data: standup, advisory review verdicts, duty probe |
| Agent lifecycle | delete lost memory (data remnance); no export | Personnel-File export; onboarding wake; offboarding-with-handover |
| Observability | transcripts trapped in containers | Team Room dashboard; chain outcome delivery; work summaries |

The structural insight that drove the rebuild (parent plan §2): state was accumulating in the
wrong walls. Phases 0–4 fixed the walls. What remains is entirely **policy**: who may act without
a human, when, and with what recourse.

---

## 3. Human-intervention map (verified 2026-07-11)

Every enforced human/operator wall, classified. "Earned-autonomy target" = this doc schedules its
conversion to a policy-governed path; "permanent" = deliberately never automated.

| # | Wall | Evidence | Classification |
|---|---|---|---|
| 1 | Team Drive `main` merge/reject is operator-or-internal; the lead's verdict has **zero enforcement effect** — nothing consumes an `approve` to merge | `drive.py:70` (pre-receive hook), `server.py:6615` / `:6691` (gates), `server.py:6720` (docstring: "ZERO enforcement effect") | **Earned-autonomy target** (#17, U4). The central dead-end: a pure-agent team cannot integrate its own reviewed work. |
| 2 | Credentials/OAuth: agents raise a durable "Needs you" ask (`help_requests.py`); only a human supplies a secret or reconnects a revoked grant; workers additionally need `can_request_user_credentials` to even ask | `server.py:3383`, `credentials.py` (vault tiers, `ConnectionRefreshError`) | **Permanent** — genuinely irreplaceable. |
| 3 | Budget exhaustion hard-blocks the LLM proxy; no agent can raise any cap | `credentials.py:4097` ("Budget exceeded"), `costs.py:205` (`set_budget`, operator/dashboard-fed) | **Split**: envelope + top-ups permanent; *allocation within* the envelope becomes a lead surface (#18, U7). |
| 4 | Privileged confirms require `origin_kind="human"` (deletes, model switches) | `pending_actions.py:346` | **Permanent at the top tier** — survives *inside* the future policy engine as irreversible/financial-tier policy, not hardcode (U3/U8). |
| 5 | Goals writes are operator/dashboard-only; no trigger turns a set goal into a task DAG | `server.py` `_require_goals_writer`; heartbeat goal-escalation requires goals already set + a configured `utility_model` | **Split**: goal *writes* stay human (permanent, #19); goal *decomposition into tasks* becomes a lead-plate loop (U6). |
| 6 | Team/member/lead management operator-only; onboarding nudge is influence-only (never creates a task) | `server.py:6248–6289` team endpoints | **Keep for now** — org-chart changes are low-frequency; revisit only if it measurably bottlenecks. |
| 7 | Blocked tasks have no automated unblocker: the chain watcher delivers ONE advisory stall nudge to the chain-root human and never re-drives | `chain_watcher.py:171` (`_maybe_nudge_stall`), `coordination_tool.py` (`blocker_note` to creator) | **Earned-autonomy target** (U5, escalation ladder). |
| 8 | Fingerprint burn: manual operator reset only, no auto-rotation | `browser/service.py` (reset endpoint; "no auto-rotation per §22") | **Permanent** — deliberate; leave alone. |

Autonomy loops already running without a human (for orientation): plate-gated heartbeat agenda +
goal-driven self-`hand_off`; task chains with auto-close; `ask_teammate` with asker-pays billing;
standups; daily work summaries; negative-outcome `feedback_push`; chain outcome delivery;
offboarding handover. The machine runs — it just parks at walls 1, 3, 5, 7.

---

## 4. Permanent human touchpoints (by design — never automated)

1. **Credentials and OAuth** — supplying a secret, reconnecting a revoked grant, scoping access.
2. **Money in** — budget envelope creation and every top-up. Agents only ever move money *within*
   the envelope (#18); they can never raise it.
3. **Initial goal-setting** — team `north_star`/`success_criteria` and standing goals stay
   human/operator-written (#19). Agents propose in threads; they do not write goals.
4. **The irreversible + financial top tier** — agent/team deletion confirms
   (`origin_kind="human"`), wallet spends above caps, anything undo receipts cannot revert. The
   requirement survives as top-tier *policy* inside the action-tier engine, not as scattered
   hardcode.
5. **Fingerprint burn reset** — deliberate manual recovery.
6. **Exception review and sampled audits** — not a wall but the *new form* of oversight: the
   dashboard's job shifts from approving actions to auditing samples and exceptions.

---

## 5. Decisions (ratified 2026-07-11; numbering continues the parent plan's §8)

17. ✅ **RATIFIED — kernel-executed auto-merge (supersedes the zero-enforcement clause of §8 #13).**
    The lead's drive-review verdict remains advisory **at the permission layer** — the verdict
    endpoint's lead-only gate, the merge/reject endpoints' `_require_operator_or_internal` gates,
    and the pinned operator-only tests are all UNTOUCHED. What changes: the **governance kernel
    may act on the verdict.** A host-side, in-process consumer fires when `record_lead_verdict`
    (`teams.py:765`) records `approve` on an open review; if earned-autonomy policy clears the
    *(lead, submitter)* pair, the mesh itself executes the merge through the existing internal
    merge path (the same internal-caller pattern as the `OL_DRIVE_PRIVILEGED=1` review-merge env,
    `drive.py:70`). Guardrails are policy, `limits.py`-tunable, conservative defaults:
    - **Trust floor:** ~5 human-executed merges of lead-approved reviews for the pair, zero
      rejected-after-approve in the window, before the first auto-merge.
    - **Sampling:** 20% of auto-merges flagged for human post-review, decaying to 5%.
    - **Trust decay:** one revert or rejected-after-approve returns the pair to human-merge.
    - **Rate cap:** per-day auto-merge cap; every auto-merge emits an undo receipt + an
      operator-chat notification (the ChainWatcher `POST /chat/note` delivery path).
    A drive merge is genuinely reversible-internal (git revert exists) — the right first consumer
    for earned autonomy.

18. ✅ **RATIFIED — lead budget allocation within the human envelope (activates the item §8 #12
    reserved for Phase 5).** A lead-reachable allocation surface, gated exactly like the verdict
    endpoint (verified caller == `teams.lead_agent_id` — team data, not a permission tier):
    Σ(per-agent allocations) ≤ team envelope, clamped via `limits.py`, every reallocation audited.
    The surface can **never raise the envelope** — top-ups stay human-only forever. This is the
    second lead-gated agent-reachable surface (after the verdict endpoint), acknowledged
    explicitly; Constraint #12 is untouched.

19. ✅ **RATIFIED — goals agent-write carve-out DECLINED (again).** The unratified §8 #12/#16
    standing gate stays closed: no agent — lead included — writes team or standing goals. Goal
    **decomposition** rides the already-legal `hand_off` verb instead: a deterministic
    goal-coverage probe on the lead's heartbeat plate (U6) escalates an agenda turn to decompose
    under-covered goals into tasks. Goals remain the human's steering wheel; no new
    prompt-injection-adjacent write surface into teammates' persistent context is opened. If
    decomposition quality ever demonstrably needs durable sub-goals, a lead-scoped write is a NEW
    ratification — do not build one without it.

**Scope fence for this cycle:** earned autonomy covers the **reversible-internal** tier only
(drive merges). External-visible actions (emails, posts) stay human/lead-approved until the track
record has real data — that is the parent plan's own Phase-5 example, and it comes after.

**Rating-trust rule (feeds U1/U3):** autonomy scoring uses **objective signals** (merged without
revert, rejected-after-approve, terminal task status) and **human ratings** at full weight.
Operator-*agent* ratings (the internal `POST /mesh/tasks/{task_id}/outcome`, `server.py:10018`,
exists precisely so the operator agent can score completions) are **excluded from autonomy
scoring** initially — agents grading agents must not feed the trust ladder — while remaining
inputs to `feedback_push` learning.

---

## 6. Gap check: written Phase 5 vs. the hands-off goal

| Needed for hands-off | In the parent plan's written Phase 5? |
|---|---|
| Action-tier policy engine, per-agent track record, earned-autonomy f(), positive feedback push, hibernation | **Yes** — verbatim scope. |
| Auto-merge consuming the lead verdict | **In spirit, not in letter** — §8 #13 froze the verdict at zero enforcement, "revisit with earned autonomy". Named here as U4 under #17. |
| Goal-coverage decomposition trigger | **No** — in no phase. Named here as U6 under #19. |
| Blocked-task escalation ladder / auto-recovery | **No** — the watcher's single advisory human nudge is the end of the written design. Named here as U5. |
| Lead budget allocation | **Named but deferred** (§8 #12). Activated here as U7 under #18. |

---

## 7. Build order (prioritized by leverage-per-risk)

Each unit is a separate PR off `main` (per §8 #6). Reuse targets are named — none of these units
should introduce a parallel mechanism where one exists.

- **U1 — Durable per-agent track record** (foundation; no ratification; medium).
  Compose from existing signals: `orchestration.count_outcomes_since` / `count_failed_status_since`
  (`orchestration.py:1039`/`:1088`, already aggregated for the operator heartbeat),
  `drive_reviews` rows as labeled datapoints (`lead_verdict` vs. final `merged|rejected` status
  measures **lead verdict-accuracy**; submitter + status measures **submitter delivery quality**
  — auto-merge trust is a property of the *pair*), `WorkSummariesStore` ratings, task outcomes.
  Persist in `teams.db` (sibling of `drive_reviews`) following the `WorkSummariesStore` shape;
  add to the Personnel-File bundle (`_build_agent_bundle`). Rolling windows; budget-block
  incidents included.
- **U2 — Positive feedback push** (small; no ratification).
  Extend `feedback_push.py:31` `_ACTIONABLE_OUTCOMES` beyond `("rework", "rejected")` to push
  accepted/acknowledged reinforcement. Apply the rating-trust rule above.
- **U3 — Action-tier policy engine, minimal-first** (large, ships thin; mechanism in written
  scope).
  Tier registry (`reversible-internal → external-visible → irreversible → financial`) + policy
  f(tier, track record, budget) → `auto | lead-approve | human-approve | deny`. Thresholds via
  the `limits.py` float pattern (`ask_bill_cap_usd` precedent). `lead-approve` routes to the
  lead's plate via the `lead_reviews_fn` probe pattern (`cron.py:146`). Ships with exactly one
  consumer (U4); absorbs `pending_actions` / wallet caps / undo tiers later (U8, parent plan C.1).
- **U4 — Auto-merge consumer (#17)** (medium).
  As specified in #17. Reuses `drive.py`'s claim-first atomic merge — `head_sha` pinning already
  409s a post-approval branch advance, exactly the property auto-merge needs.
- **U5 — Blocked-task escalation ladder** (medium; no ratification — nudges and lead-created
  tasks are already-legal influence).
  Rungs, each on an existing delivery primitive: (1) re-drive the assignee with the
  `blocker_note` via `LaneManager.deliver_chat`/`try_steer` (`lanes.py:544`/`:490`); (2) escalate
  to the task creator (followup turn); (3) put it on the lead's plate (extend the lead-duty cron
  probe with a blocked-tasks feed) — the lead reassigns via `hand_off`, answers, or reallocates
  budget (U7); (4) **human, only for** credential / envelope-exhausted / irreversible-tier
  blockers (routed into the durable `help_requests.py` "Needs you" registry) **plus a max-age
  fallback (~48h)**. Rungs 1–3 retry within budget indefinitely; the existing single stall nudge
  (`chain_watcher.py:171`) becomes rung 4. Rate-limit rung climbs using the watcher's existing
  claim semantics; rungs 1–3 bill the nudged agent's work ledger.
- **U6 — Goal-coverage probe (#19 path)** (small-medium).
  Cheap deterministic lead-plate check: team goals set but fewer than N open non-terminal tasks
  advancing them (`tasks_store.list_team`) → escalate the lead's agenda turn with "decompose the
  team goals into tasks and hand them off." Pure layer-2: cron probe + prompt scaffolding +
  existing `hand_off`.
- **U7 — Lead budget allocation (#18)** (medium).
  As specified in #18. Composition payoff with U5: budget-blocked task → `blocker_note` → ladder
  reaches the lead → lead reallocates headroom → task retried. Self-healing inside the human's
  money wall.
- **U8 — Policy-engine completion, then hibernation** (last).
  Absorb `pending_actions` (the `origin_kind="human"` check, `pending_actions.py:346`, becomes
  top-tier policy) and wallet caps per parent plan C.1, deleting the old stores in-phase.
  Hibernation ships last: it advances cost/scale, not hands-off, and still needs B3's undesigned
  legs (health-monitor restart awareness, wake-then-forward shim).

**Order rationale:** U1 first because every autonomy decision consumes it; U2 rides along; U3
ships thin so U4 (the highest single-leverage unlock) lands early; U5–U7 close the remaining
daily-path loops; U8 pays down the absorption debt and only then spends effort on scale.

---

## 8. What "done" looks like

A team with goals set, credentials provisioned, and an envelope funded runs an indefinite loop
with zero human actions on the daily path: heartbeats put goal gaps on the lead's plate → the
lead decomposes into tasks → workers deliver to the Drive → the lead reviews and records verdicts
→ trusted pairs auto-merge (sampled) → blocked work climbs the ladder and self-heals within the
envelope → summaries, standups, and track records accumulate. The human's remaining surface:
set/adjust goals, fund the envelope, supply credentials, confirm the irreversible tier, and audit
exceptions + samples. Every relaxation is a `limits.py` number backed by an audit trail — loosen
policy, never rebuild structure.
