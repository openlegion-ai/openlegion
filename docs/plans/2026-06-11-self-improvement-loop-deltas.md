# Self-improvement loop — issue #1012 re-scoped to four surgical deltas

**Status:** READY — all code refs validated against `ddabc6a2` (2026-06-11).
**Supersedes:** the 3-phase roadmap in issue #1012 as written. ~70% of that
roadmap shipped through other PRs between 2026-05-31 and 2026-06-11 (#1053
blocker notes, #1060–1063 memory overhaul, #1105 rating→learning loop +
recovery wakes, #1106 rating history, #1110 watch mode). This plan ships the
*remaining* deltas, redesigned around the mid-2026 research consensus.
**Does NOT use:** any code from PR #1026 (contributor has not signed the CLA;
the PR's architecture is also rejected on merits — see PR 0).

---

## 0. Design principles (research grounding)

Four independent lines of evidence contraindicate the issue's core mechanism
("agent rewrites its own INSTRUCTIONS.md from its own failure logs via an
inline LLM pass"). Every delta below is shaped by them:

1. **ACE / context collapse** (ICLR 2026): whole-file LLM rewrites of a
   playbook erode detail and bias toward brevity. → All self-edits are
   **delta-shaped** (append a dated bullet, never rewrite), and the prompts
   say so explicitly.
2. **Misevolution** (ICLR 2026): unvalidated memory accumulation degrades
   behavior even on frontier models. → Edits are gated on **external outcome
   signals** (the `accepted/rework/rejected` ratings from #1105), not the
   agent's self-judgment of failure.
3. **Zombie Agents** (Feb 2026): failure logs and peer-writable channels are
   injection vectors into *standing* instructions. → New standing-instruction
   surfaces (per-agent goals) are **operator-write-only**, enforced in the
   permission layer, and all new prompt inputs pass `sanitize_for_prompt()`.
4. **Reflection literature**: reflection helps only against ground truth.
   → No new "reflection module"; we feed failure evidence into the loops
   that already exist (heartbeat prompt, consolidation pass, corrections
   file) and let measurement decide whether more is needed.

Also: no credible production evidence of safe autonomous goal *formation* —
so initiative ships as **operator-set goals, agent-chosen tactics** (the
read path for which is already fully built and dormant).

KISS test applied throughout: every delta is wiring or prompt text on an
existing mechanism. Zero new modules, zero new stores, one new tool, zero
new endpoints, one new permission carve-out pair.

---

## PR 0 — hygiene (no code)

1. **Close PR #1026** ("agent self-reflection module"). Reasons, in the close
   message: (a) CLA unsigned — unmergeable regardless of content; (b) a
   parallel `reflection.py` + `reflection_log.jsonl` + review endpoint
   duplicates `learnings/errors.md` + `get_learnings_context()` +
   consolidation, which already exist; (c) its operator-approval queue
   duplicates what the operator *agent* is for. Invite re-engagement after
   CLA if desired. Do not read or reference its diff in any implementation
   work (clean-room discipline).
2. **Decline the CONTRIBUTORS.md proposal** in the #1012 thread. Note the
   account-pattern flag privately: the proposal credits `@whitehat-bot`, a
   different handle than the commenting account `zsxh1990` — classic
   automated reputation-farming signature.
3. **Post a status comment on #1012** marking Phases 1/3 substantially
   superseded (cite PRs #1053/#1060–1063/#1105/#1106/#1110), and link this
   plan as the re-scope. Keeps the thread from attracting more full-roadmap
   PRs.

---

## PR 1 — Delta A: failure-aware loops (Phase-1 remainder)

Branch: `feat/failure-aware-loops`. All changes in `src/agent/`.

### A1. Inject tool history into the heartbeat prompt

The helper exists (`AgentLoop._build_tool_history_context`,
`src/agent/loop.py:1931-1940`, limit `_TOOL_HISTORY_LIMIT = 10` at
`loop.py:62`) and is already injected into task (`loop.py:2004-2006`) and
chat (`loop.py:4725-4727`) prompts — but **not** the heartbeat prompt, the
one place self-review happens.

**Edit:** in `execute_heartbeat`, insert between section 4 (Learnings,
`loop.py:2328-2332`) and section 5 (Fleet roster, `loop.py:2334-2340`):

```python
# 4.5 Recent tool outcomes — evidence for the Self-Evolution step.
if self.memory:
    tool_history = self._build_tool_history_context()
    if tool_history:
        parts.append(tool_history)
```

Mirror the same `if self.memory:` guard the task path uses. The helper is a
sync SQLite read — same call pattern as the task/chat prompt builds, no new
async plumbing. Cost: ≤10 lines × ~12 tokens per heartbeat, only when
history exists.

### A2. Tighten the Self-Evolution nudge to delta edits

Current nudge (`loop.py:2351-2355`) says only that the files are editable.
Replace the body with (keep the allowlist gate at `loop.py:2347-2350`
unchanged):

```python
parts.append(
    "## Self-Evolution\n"
    "You can update INSTRUCTIONS.md, SOUL.md, USER.md, and HEARTBEAT.md "
    "during heartbeats to improve future sessions.\n"
    "If Recent Tool History or Learnings show the SAME failure repeating, "
    "record the fix as ONE short dated bullet under a '## Lessons' heading "
    "in INSTRUCTIONS.md. Append only — never rewrite the file, never paste "
    "raw error text."
)
```

This encodes ACE delta-edit discipline at the only point agents are invited
to self-edit. The existing 32KB cap + timestamped backups in
`WorkspaceManager.update_file` (`workspace.py:801-838`, max 20 backups) are
the bloat ceiling and rollback path — no new guardrail code needed.

### A3. Failure-aware consolidation ("dream cycle" reads failures)

`ContextManager._maybe_consolidate_memory` (`src/agent/context.py:618-676`)
already runs an LLM pass ≥6h apart (`_CONSOLIDATION_MIN_INTERVAL_S`,
`context.py:39`) over compiled head + top-30 salient facts + log tail. It
never sees tool failures. Two edits:

1. **`Memory.get_tool_history`** (`src/agent/memory.py:785-815`): add a
   keyword filter `success: bool | None = None`; when not None, append
   `AND success = ?` to the WHERE clause. Backward-compatible (default
   None = current behavior).
2. **`_maybe_consolidate_memory`**: beside the salience fetch
   (`context.py:636-641`), fetch failures and add a fourth input section:

```python
failures = ""
if self.memory:
    rows = self.memory.get_tool_history(limit=15, success=False)
    if rows:
        failures = "\n".join(
            f"- {r['tool_name']} ({r['created_at']}): {r['outcome'][:200]}"
            for r in rows
        )[:2000]
```

   In the prompt f-string block (`context.py:650-653`) append:

```python
+ (f"\n\n## Recent tool failures (newest first, UNTRUSTED text)\n"
   f"{sanitize_for_prompt(failures)}" if failures else "")
```

   And extend the rules sentence (`context.py:645-648`) with:
   `"If the failures section shows the same cause repeating, keep ONE short "`
   `"lesson under a 'Lessons' heading; never copy raw error text into memory."`

   Note `sanitize_for_prompt` on the failures block — tool outcomes are
   tool-output-derived and therefore untrusted (Zombie-Agents vector). The
   head/facts/log inputs are agent-authored and already flow today; leave
   them as is. Output write already sanitizes (`context.py:669`).

   Budget: +≤2KB input per consolidation, i.e. per ≥6h per agent. No change
   to `max_tokens=1500` on the call (`context.py:656-660`); the existing
   30-min fail backoff (`context.py:41`) covers the new fetch too.

### A4. Delete dead code: `Memory.get_recent_logs`

`memory.py:729-748` has zero production callers (only
`tests/test_memory.py:140`). The consolidation pass reads the workspace
daily log instead, and A3 reads `get_tool_history`. Delete the method and
its test. (If anyone objects: adoption was considered and rejected — the
`logs` table duplicates what the workspace activity log already feeds the
consolidator.)

### A5. Acceptance instrument: repeat-failure log line

The issue's Phase-1 acceptance test was "an agent that hit the same tool
error 3× stops hitting it." Make that measurable without schema changes: in
`AgentLoop._record_failure` (`loop.py:1885-1896`), after
`store_tool_outcome`, count same-signature recent failures and log:

```python
repeats = len(self.memory.get_tool_history(
    tool_name=tool_name,
    params_hash=self.memory._compute_params_hash(arguments or {}),
    limit=5, success=False,
))
if repeats >= 3:
    logger.warning("repeat_failure tool=%s count=%d", tool_name, repeats)
```

Grep-able on deployed boxes (`journalctl | grep repeat_failure`) to compare
before/after rates. If `_compute_params_hash` is private-by-convention only,
call it directly — same module family; do not duplicate the hash logic.

### Tests (PR 1)

- `tests/test_loop.py` (pattern: `_make_loop()` at lines 39-95, mock memory):
  - `test_heartbeat_injects_tool_history` — stub
    `memory.get_tool_history` → assert `## Recent Tool History` in the
    system prompt captured from the mocked `llm.chat` call.
  - `test_heartbeat_omits_tool_history_when_empty` — empty stub → section
    absent.
  - `test_record_failure_logs_repeat_warning` — 3 same-hash failures →
    `caplog` contains `repeat_failure`.
- `tests/test_compiled_memory.py` (pattern: `_make_workspace` +
  `AsyncMock` LLM, see `TestConsolidateMemory` at line 211):
  - `test_consolidation_includes_failures_section` — memory stub returns
    failures → prompt contains `## Recent tool failures`.
  - `test_consolidation_skips_failures_section_when_clean` — no failures →
    section absent, prompt otherwise unchanged.
- `tests/test_memory.py`: replace the `get_recent_logs` test with
  `test_get_tool_history_success_filter`.

---

## PR 2 — Delta C: handoff pushback as prompt-layer (no new verb)

Branch: `fix/coordination-pushback-docs`. Single file:
`src/agent/builtins/coordination_tool.py`. Rationale: the transport already
exists end-to-end — `update_status(state="blocked")` persists a
`blocker_note` (`coordination_tool.py:701-705` →
`orchestration.py:1580-1587`), the mesh writes a `task_blocked` back-edge to
`inbox/{creator}/task_event/` and wakes the creator with L9 creator-binding
(`src/host/server.py:5577-5666`, wake binding `5690-5696`), the creator sees
it in `check_inbox` `events[]` (`coordination_tool.py:541-601`), and
human-rooted chains additionally fire an operator recovery wake
(`server.py:5474-5575`). What's missing is that **no worker is ever told
this protocol exists**: the `check_inbox` description never mentions
`events[]`, and only the operator's heartbeat template
(`src/cli/config.py:1778-1783`) explains the back-edge. Tool descriptions
are re-read by every agent every turn — fixing them needs no sentinel
roll-forward and reaches all deployed fleets on container restart.

### C1. `update_status` description (`coordination_tool.py:604-615`) — append:

> `blocked` is your pushback channel. If a handed-off task is malformed,
> missing inputs, or outside your role, call
> `update_status(state="blocked", summary="<the specific question or
> missing input>", task_id=...)` instead of guessing. Your summary is
> delivered back to the task's creator as `blocker_note`, and the operator
> is alerted on user-facing chains. NEVER mark work done that you did not
> do, and NEVER silently drop a task you can't execute.

### C2. `check_inbox` description (`coordination_tool.py:508-521`) — append:

> The result also includes `events[]` — `task_failed` / `task_blocked`
> notices for tasks YOU created via hand_off. For each one, read
> `blocker_note` / `error`: if the recipient asked a question, answer it
> with a corrected `hand_off` (new brief, same goal); if it failed
> terminally, decide whether to retry, reroute, or report the failure
> upstream. Do not ignore these events.

### C3. `hand_off` description (`coordination_tool.py:115-141`) — append one line:

> If the recipient can't execute your task they will block it with a
> question — you'll see it in `check_inbox()` `events[]`; answer and
> re-hand-off rather than waiting.

Token cost: ~+150 tokens per agent turn (tool schemas). Keep wording exactly
as tight as above; do not add examples.

**Explicit non-goal:** no `reject_task` verb. The one semantic blocked-with-
question can't express is *terminal refusal by the assignee* — today that
disposition belongs to the operator (recovery wake → `manage_task`
retry/reroute/cancel), which is the correct trust boundary. Revisit only if
post-deploy measurement (see §Measurement) shows blocked-tasks rotting
because creators won't re-hand-off.

### Tests (PR 2)

Descriptions are data, not logic — pin the contract, not the prose:
- `tests/test_coordination_tool.py`: `test_update_status_description_mentions_pushback`
  and `test_check_inbox_description_mentions_events` — assert the substrings
  `"pushback"` / `"events[]"` appear in the registered tool descriptions
  (guards against accidental truncation in future edits; mirrors how
  `test_operator_still_gated_surfaces_not_in_bypass_grep` pins prose
  contracts elsewhere).

---

## PR 3 — Delta B: operator-set per-agent goals (Phase-2 smallest slice)

Branch: `feat/operator-agent-goals`. The read path is 100% built and
dormant: every agent reads blackboard key `goals/{agent_id}`
(`AgentLoop._fetch_goals`, `loop.py:1752-1764`, 5-min fetch cache
`_GOALS_TTL` at `loop.py:60`) and injects `## Your Current Goals` into task
(`loop.py:1770-1772`), chat (`loop.py:4646-4647`), and heartbeat
(`loop.py:2300-2302`) prompts — and goals defeat the empty-heartbeat skip
(`loop.py:2262-2270`). **Nothing in `src/` writes the key.** This PR adds
the writer plus the permission hardening the new surface requires.

### B1. Scoping contract (the part that's easy to get wrong)

`_fetch_goals` reads via `read_blackboard` → `_scope_key`
(`mesh_client.py:106-118`): team agents read
`projects/{team}/goals/{agent_id}`; solo agents read raw `goals/{agent_id}`.
The operator is fleet-global (no team), so it must write with an explicit
project override — exactly the resolution `hand_off` already performs
(`coordination_tool.py:229-262`): `mesh_client.list_agents()` registry
entries carry `project` and `scope == "global"` hints. Reuse that recipe
verbatim.

### B2. New operator tool: `set_agent_goals` (in `operator_tools.py`)

Deliberately **separate from `manage_goals`** (operator-workspace GOALS.json
files, merge-write classification at `operator_tools.py:2516-2524`) — the
stores, dependencies (`workspace_manager` vs `mesh_client`), and failure
modes are disjoint; one tool with two backends would be the confusing kind
of "simple".

```python
@tool(
    name="set_agent_goals",
    description=(
        "Assign standing goals to a worker agent. Goals appear in that "
        "agent's every prompt (tasks, chats, heartbeats) under '## Your "
        "Current Goals' and make its idle heartbeats pursue them instead "
        "of sleeping. Replaces the agent's whole goal list (max 5 goals, "
        "each one sentence). Pass goals=[] to clear. This is for WORKER "
        "direction — your own fleet/business goals live in manage_goals."
    ),
    operator_only=True,
)
async def set_agent_goals(
    agent_id: str, goals: list[str], *, mesh_client=None, **_kw,
) -> dict:
```

Behavior, in order:
1. Validate: `agent_id != "operator"` (operator goals = GOALS.json; refuse
   with a pointer to `manage_goals`); `len(goals) <= 5`; each goal a
   non-empty `str` ≤ 300 chars after `.strip()`.
2. Resolve scope via `mesh_client.list_agents()` — 404-style error listing
   available agents if absent (mirror `coordination_tool.py:234-237`);
   `project = target_info.get("project")`.
3. `goals == []` → `mesh_client.delete_blackboard(f"goals/{agent_id}",
   project=project)` and return `{"cleared": True, "agent_id": agent_id}`.
4. Else write, **no TTL** (persists until changed; blackboard entries
   without `ttl` never expire — `src/host/mesh.py:270-281`):

```python
await mesh_client.write_blackboard(
    f"goals/{agent_id}",
    {"goals": goals, "set_by": "operator",
     "updated_at": datetime.now(timezone.utc).isoformat()},
    project=project,
)
return {"set": True, "agent_id": agent_id, "count": len(goals),
        "note": "Takes effect on the agent's next prompt build "
                "(<=5 min cache)."}
```

The dict pretty-prints through `format_dict` + `sanitize_for_prompt` at
every injection site — no renderer changes needed.

Inject `mesh_client` the same way the other mesh-touching operator tools
receive it (kwarg injection at registry call time — follow
`workflow_snapshot`'s pattern in the same file).

### B3. `mesh_client.delete_blackboard` gains `project=`

`mesh_client.py:255-269` currently supports only `global_scope`. Mirror
`write_blackboard`'s three-way scoping (`mesh_client.py:236-241`) exactly:
`global_scope` → raw; `project is not None` → `projects/{project}/{key}`;
else `_scope_key`. ~4 lines, backward-compatible.

### B4. Permission hardening: goals are operator-write-only, self-readable

Two targeted edits in `src/host/permissions.py`, both justified by the
Zombie-Agents finding (goals are standing instructions injected into every
prompt — the highest-value injection target this PR creates):

1. **Write guard** in `can_write_blackboard` (`permissions.py:248-261`),
   after the `_is_trusted` check and global carve-outs, before the pattern
   match:

```python
# Goals are standing instructions injected into the target agent's
# every prompt. Only the operator (endpoint carve-out, server.py:1749)
# or the mesh itself may write them — a teammate's projects/{team}/*
# write wildcard must NOT cover a peer's goals key (prompt-injection
# channel into persistent context).
tail = key.split("/", 2)[2] if key.startswith("projects/") else key
if tail.startswith("goals/"):
    return False
```

   The operator path is unaffected: `PUT /mesh/blackboard/{key}` short-
   circuits on `_caller_is_operator` before consulting this function
   (`server.py:1749-1755`).

2. **Self-read carve-out** in `can_read_blackboard`, beside the existing
   self-inbox carve-out (`permissions.py:243-246`), so goal delivery never
   depends on per-template ACL variance (template agents have narrow
   `blackboard_read` lists, e.g. `devteam.yaml:74`; team membership ACLs
   are rewritten on join/leave at `cli/config.py:1040-1092`):

```python
# An agent may always read its OWN goals key (raw or team-scoped form).
if key == f"goals/{agent_id}" or (
    key.startswith("projects/") and key.endswith(f"/goals/{agent_id}")
):
    return True
```

### B5. Out of scope (explicit)

- **Not** in `_OPERATOR_HEARTBEAT_TOOLS` (`cli/config.py:1731-1752`):
  goal *assignment* is a deliberate planning act, not an unsupervised
  heartbeat act. The operator sets goals during user-engaged turns or
  recovery wakes (full toolset). Revisit with measurement.
- **No agent-formed goals** (issue Phase-2 "observation → goal"): no
  production evidence this is safe; the operator-gate is the design.
- **No dashboard surface**: the dashboard can read the key via existing
  blackboard endpoints if wanted later; not this PR.
- **No `edit_agent` field**: goals are runtime state, not config — keeping
  them out of the edit/undo-receipt machinery is intentional.

### Tests (PR 3)

- `tests/test_operator_tools.py` (pattern: `ALLOWED_TOOLS` monkeypatch +
  mocked mesh_client, see `test_edit_agent_soft_field_calls_edit_soft_immediately`
  at line 421):
  - `test_set_agent_goals_writes_scoped_key` — registry stub with
    `project: "alpha"` → assert `write_blackboard` called with
    `key="goals/researcher"`, `project="alpha"`, no `ttl`.
  - `test_set_agent_goals_solo_agent_unscoped` — registry stub without
    `project` → `project=None`.
  - `test_set_agent_goals_clear_deletes_key` — `goals=[]` →
    `delete_blackboard` with matching scope.
  - `test_set_agent_goals_rejects_operator_target`,
    `test_set_agent_goals_caps_count_and_length`,
    `test_set_agent_goals_unknown_agent_lists_available`.
- `tests/test_permissions.py`:
  - `test_worker_cannot_write_peer_goals_key` — team agent with
    `projects/alpha/*` write ACL denied on `projects/alpha/goals/peer`.
  - `test_agent_can_always_read_own_goals_key` — raw and scoped forms,
    with an empty `blackboard_read` ACL.
  - `test_agent_cannot_read_other_agents_goals_via_carveout`.
- `tests/test_loop.py`: existing
  `test_heartbeat_runs_when_empty_rules_but_goals_exist` (line 2420)
  already covers the read side — verify it still passes unchanged.

---

## PR 4 — Delta D: operator agent-retro heartbeat step (meta-layer)

Branch: `feat/operator-heartbeat-v6-agent-retro`. Prompt-layer only, riding
the existing sentinel roll-forward machinery.

### D1. Append `heartbeat_v6_agent_retro` to `HEARTBEAT_SENTINELS`

`src/shared/types.py:101-106` — append to the END of the tuple (contract
comment at `types.py:93-100`). Both consumers then roll existing
deployments forward automatically: config-side field refresh
(`cli/config.py:1944-2014`) and workspace-side template overwrite
(`workspace.py:394-460` — HEARTBEAT.md is overwrite-style refresh, gated on
a prior sentinel being present and the new one absent).

### D2. New step in `_OPERATOR_HEARTBEAT` (`cli/config.py:1768-1910`)

Add the v6 marker beside the existing markers (`config.py:1769-1772`) and
append, after the rating-cadence step (`config.py:1841-1854`):

> **Agent retro (at most ONE agent per heartbeat).** If the Fleet Health
> digest or rating history shows an agent with `rework >= 3` or
> `rejected > 0` in the window: drill in with `inspect_agents` and read
> the repeated feedback. If the SAME correction keeps appearing, do not
> edit anything from the heartbeat — instead `notify_user` with a one-line
> proposed instruction delta, e.g. "research-1 keeps omitting source links
> (3x rework). Proposed INSTRUCTIONS.md addition: 'Always end reports with
> a Sources section.' Reply to approve, or tell me to apply it." Apply via
> `edit_agent(field="instructions")` only in a normal (non-heartbeat) turn,
> as ONE appended dated bullet — never a rewrite. Skip this step entirely
> when no agent crosses the thresholds.

This respects the existing heartbeat tool allowlist — `edit_agent` is
deliberately absent from `_OPERATOR_HEARTBEAT_TOOLS`
(`cli/config.py:1731-1752`) and **stays absent**: unsupervised instruction
edits are exactly what the Misevolution result warns about. The recovery-
wake brief (`server.py:5547-5557`) already authorizes `edit_agent` in full
chat turns "if systemic"; this step generates the *proposal* signal on a
cadence instead of only on failures.

### Tests (PR 4)

- `tests/test_operator_config.py`: update the sentinel-pinning tests
  (pattern: `test_core_carries_latest_playbook_sentinel` at line 641 — the
  heartbeat equivalents pin `HEARTBEAT_SENTINELS[-1]`); add
  `test_heartbeat_v6_mentions_agent_retro` asserting the new step text is
  in `_OPERATOR_HEARTBEAT`.
- `tests/test_workspace.py` `TestHeartbeatVersionedRefresh` (line 1230):
  existing tests reference `HEARTBEAT_SENTINELS[-1]` — verify they pick up
  v6 and that a v5-sentinel HEARTBEAT.md gets refreshed to the v6 template.

---

## Sequencing, rollout, measurement

**Order:** PR 0 → PR 1 → PR 2 → PR 3 → PR 4. Each PR is independently
shippable and revertable; none depends on another's code (PR 4's retro step
reads signals that exist today). Standard discipline per CLAUDE.md: every
PR built in an isolated worktree, `gh pr create`, wait for CI
(`gh pr checks --watch`), no direct merges to main. Run the fast suite
(`pytest tests/ --ignore=tests/test_e2e*.py -x`) in a subagent per PR.

**Deploy:** standard path — SSH the client boxes, `git pull`,
`systemctl restart openlegion` (config/ is gitignored, survives). PR 2's
docstrings and PR 4's roll-forward take effect on restart with no manual
steps. PR 3 requires no migration (new key namespace).

**Measurement gates (2 weeks on cake + kovarastudio before any follow-up):**

| Signal | Instrument | Expect |
|---|---|---|
| Repeat failures | `grep repeat_failure` in journal (A5) | rate falls after A1–A3 |
| Silent chain breaks | `chain_breaks_24h_count` on `/mesh/system/metrics` | falls after PR 2 |
| Pushback adoption | `task_blocked` back-edge frequency (blackboard `inbox/*/task_event/*`, audit log) | rises from ~0; blocked tasks get re-hand-offs, not rot |
| Goal-driven initiative | heartbeat skip rate (`no_heartbeat_rules`) for agents with goals set | drops to ~0 for those agents |
| Retro proposals | `notify_user` proposals citing rework patterns | ≥1 sensible proposal per real rework cluster; no proposal spam |
| Instruction bloat | INSTRUCTIONS.md sizes across fleet (32KB cap distance) | growth ≤ a few hundred bytes/week/agent |

**Deferred until the gates say otherwise** (do NOT build now): `reject_task`
verb; agent-formed goals; reflection on successes; a dedicated reflection
module; dashboard goals UI; `edit_agent` in the heartbeat allowlist;
delta-ifying the consolidation rewrite itself (ACE concern noted — watch for
compiled-head detail loss in the wild first).

## Risk register

| Risk | Mitigation |
|---|---|
| Self-edit feedback loop (agent edits itself worse) | Delta edits only (A2 wording); 32KB cap + 20 timestamped backups = rollback; external-signal gating for operator-driven edits (D2) |
| Prompt injection into standing goals | Operator-write-only enforced in permission layer (B4); `sanitize_for_prompt` at every injection site (already present) |
| Injection via failure logs into compiled memory | `sanitize_for_prompt` on the failures section (A3); "never copy raw error text" rule in both prompts |
| Heartbeat token creep | A1 ≤ ~120 tokens; C1–C3 ≤ ~150 tokens; both flat, not compounding; cron rule #1 ("be economical") unchanged |
| Operator retro spam | One agent per heartbeat, threshold-gated, propose-don't-apply (D2) |
| Goals key scope mismatch (silent no-op) | B1 reuses hand_off's proven resolution; B4 read carve-out removes ACL variance; tool result states the 5-min cache so the operator doesn't misread a delay as failure |
| CLA / provenance | PR 0 clean-room note; no #1026 code read or reused |
