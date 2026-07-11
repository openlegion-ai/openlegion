# Hands-Off Teams — Phase-5+ Review and Human-Intervention Map

**Date:** 2026-07-11
**Status:** Companion review to `2026-07-04-agent-employee-platform-architecture.md` (the
"parent plan"). **All decisions live in the parent plan's §8 (#17–#24)** — this doc records the
system review and intervention map that motivated them, and changes no code by itself. The
Phase-5 build order (U0–U8) is recorded in the parent plan's Phase-5 landed entry.

> **Scope.** A review of how the system coordinated *before* the team system, a map of every
> human-intervention point that remains *after* Phases 0–4, and the path to the parent plan's
> own north star: the human reviews **by exception**, not per-task. "Hands-off" does not mean
> removing human walls. It means converting each wall from a **per-action gate** into a
> **policy-governed exception path**, with the audit trail deciding when policy loosens ("as
> models strengthen, loosen policy — never rebuild structure").

---

## 1. System review: before the team system → now

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
wrong walls. Phases 0–4 fixed the walls. What remains is entirely **policy**: who may act
without a human, when, and with what recourse.

---

## 2. Human-intervention map (verified 2026-07-11)

Every enforced human/operator wall, classified. "Earned-autonomy target" = Phase 5 schedules its
conversion to a policy-governed path; "permanent" = deliberately never automated.

| # | Wall | Evidence | Classification |
|---|---|---|---|
| 1 | Team Drive `main` merge/reject is operator-or-internal; the lead's verdict has **zero enforcement effect** — nothing consumes an `approve` to merge | `drive.py` pre-receive hook; merge/reject gates; verdict docstring | **Earned-autonomy target** (§8 #20, U4). The central dead-end: a pure-agent team cannot integrate its own reviewed work. |
| 2 | Credentials/OAuth: agents raise a durable "Needs you" ask; only a human supplies a secret or reconnects a revoked grant; workers additionally need `can_request_user_credentials` to even ask | `help_requests.py`, `credentials.py` (vault tiers) | **Permanent** — genuinely irreplaceable. |
| 3 | Budget exhaustion hard-blocks the LLM proxy; no agent can raise any cap | `credentials.py` ("Budget exceeded"), `costs.py:set_budget` (operator/dashboard-fed) | **Split**: envelope + top-ups permanent; *allocation within* the envelope becomes a lead surface (§8 #21, U7). |
| 4 | Privileged confirms require `origin_kind="human"` (deletes today) | `pending_actions.py` consume origin gate | **Permanent at the top tier** — survives *inside* the policy engine as irreversible/financial-tier policy, not hardcode (§8 #17/#19, U3). |
| 5 | Goals writes are operator/dashboard-only; no trigger turns a set goal into a task DAG | `_require_goals_writer`; heartbeat goal-escalation requires goals already set + a configured `utility_model` | **Split**: goal *writes* stay human (permanent, §8 #22); goal *decomposition into tasks* becomes a lead-plate loop (§8 #22, U6). |
| 6 | Team/member/lead management operator-only; onboarding nudge is influence-only (never creates a task) | team endpoints in `server.py` | **Keep for now** — org-chart changes are low-frequency; revisit only if it measurably bottlenecks. |
| 7 | Blocked tasks have no automated unblocker: the chain watcher delivers ONE advisory stall nudge to the chain-root human and never re-drives | `chain_watcher._maybe_nudge_stall`; `blocker_note` to creator | **Earned-autonomy target** (§8 #22 ladder, U6). |
| 8 | Fingerprint burn: manual operator reset only, no auto-rotation | `browser/service.py` (reset endpoint) | **Permanent** — deliberate; leave alone. |

Autonomy loops already running without a human (for orientation): plate-gated heartbeat agenda +
goal-driven self-`hand_off`; task chains with auto-close; `ask_teammate` with asker-pays
billing; standups; daily work summaries; negative-outcome `feedback_push`; chain outcome
delivery; offboarding handover. The machine runs — it just parks at walls 1, 3, 5, 7.

---

## 3. Permanent human touchpoints (by design — never automated)

1. **Credentials and OAuth** — supplying a secret, reconnecting a revoked grant, scoping access.
2. **Money in** — budget envelope creation and every top-up. Agents only ever move money
   *within* the envelope (§8 #21); they can never raise it.
3. **Initial goal-setting** — team `north_star`/`success_criteria` and standing goals stay
   human/operator-written (§8 #22). Agents propose in threads; they do not write goals.
4. **The irreversible + financial top tier** — agent/team deletion confirms
   (`origin_kind="human"`), wallet spends above caps, anything undo receipts cannot revert. The
   requirement survives as top-tier *policy* inside the action-tier engine, not as scattered
   hardcode.
5. **Fingerprint burn reset** — deliberate manual recovery.
6. **Exception review and sampled audits** — not a wall but the *new form* of oversight: the
   dashboard's job shifts from approving actions to auditing samples and exceptions.

---

## 4. Gap check: written Phase 5 vs. the hands-off goal

| Needed for hands-off | In the parent plan's written Phase 5? |
|---|---|
| Action-tier policy engine, per-agent track record, earned-autonomy f(), positive feedback push, hibernation | **Yes** — verbatim scope (§8 #17–#19, #23, #24). |
| Auto-merge consuming the lead verdict | **In spirit, not in letter** — §8 #13 froze the verdict at zero enforcement, "revisit with earned autonomy". Ratified as §8 #20 (U4). |
| Goal-coverage decomposition trigger | **No** — in no phase. Ratified as part of §8 #22 (U6). |
| Blocked-task escalation ladder / auto-recovery | **No** — the watcher's single advisory human nudge was the end of the written design. Ratified as part of §8 #22 (U6). |
| Lead budget allocation | **Named but deferred** (§8 #12). Activated as §8 #21 (U7). |

---

## 5. What "done" looks like

A team with goals set, credentials provisioned, and an envelope funded runs an indefinite loop
with zero human actions on the daily path: heartbeats put goal gaps on the lead's plate → the
lead decomposes into tasks → workers deliver to the Drive → the lead reviews and records
verdicts → trusted pairs auto-merge (sampled) → blocked work climbs the ladder and self-heals
within the envelope → summaries, standups, and track records accumulate. The human's remaining
surface: set/adjust goals, fund the envelope, supply credentials, confirm the irreversible tier,
and audit exceptions + samples. Every relaxation is a `limits.py` number backed by an audit
trail — loosen policy, never rebuild structure.
