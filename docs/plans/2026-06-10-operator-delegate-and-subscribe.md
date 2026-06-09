# Operator orchestration: delegate-and-subscribe (no more silent stalls)

**Status:** REVISED after principal-engineer + Codex review (2026-06-10). Phase 0 implemented in the same PR as this revision. Phases 1–3 ready to implement.
**One-line:** Stop the operator from *blocking* to watch a pipeline. Make watching a durable, origin-aware **system** behavior that **pushes** progress + a **guaranteed terminal outcome** to the user's channel. The operator delegates and releases; it never holds a turn open to babysit a poll loop.

> **Review note (2026-06-10).** The original draft mislocated the root cause and overstated how much of Phase 1 is "just wiring." Both were corrected against the actual code (and an independent Codex pass). The *product direction* — no silent death; delegate-and-release — survived review unchanged. The corrections are folded in below; the superseded claims are called out where they mattered.

---

## 0. TL;DR for the reviewer

- **Symptom (reproduced on cake):** user asks the operator to hand off work through the team; the operator "stops out of nowhere" when a pipeline stage takes a while.
- **Root cause (CORRECTED):** a streaming hop with a **120s idle timeout that receives no keepalive** kills the turn. It is **NOT** the mesh→agent `transport.py` hop (originally blamed) — that hop is kept alive by the agent's 15s SSE keepalive. The unprotected leg is **dashboard→browser**: `transport.stream_request` strips the agent's `": keepalive"` comments (forwards only `data:` lines), and the dashboard's own SSE generator emits no keepalive — so during a long silent tool call the browser sees zero bytes and its **120s `AbortController`** (`app.js:8610-8619`) fires. That disconnect propagates back (browser→dashboard→agent), cancelling the operator's coroutine → `await_task_event EXIT cancelled (likely tool timeout)` → turn torn down. Then **nothing re-wakes the operator** → stranded.
- **Why "we improved it" didn't help:** PR #1068 made `await_task_event` poll task status (good — hop 1 worked). But the poll runs inside a blocking tool call, and the *browser idle abort* (not any server-side cap) kills it once a single silent window exceeds 120s. #1068 didn't touch the streaming leg.
- **The fix has two layers:**
  1. **Phase 0 (bleeder-stopper, IMPLEMENTED here):** forward the agent's keepalive end-to-end so no streaming hop goes byte-silent. This stops the silent death for **every** long quiet tool (operator `await_task_event`, but also `execute_code`, `run_command`, browser ops), not just this one path. Zero security surface.
  2. **Phase 1+ (the real product):** even with the connection kept alive, holding a synchronous chat turn open for a 10-minute multi-hop pipeline is wrong — it monopolizes the operator, burns LLM tokens in a poll loop, and dies on operator restart (which we saw on cake). So decouple watching from the turn: **delegate-and-subscribe + push**, with a hard guarantee that every user-initiated chain ends in a user-facing outcome.
- **Honesty correction:** Phase 1 is **not** "just wiring." Only one reusable primitive exists end-to-end (`_handle_notify_origin` for the chat channels). The chain-level state, the proactive push trigger, and **dashboard delivery** (the exact channel the repro used) must be built. See §4.

---

## 1. Evidence (the cake trace — ground truth)

Operator on cake (`anthropic/claude-opus-4-8`), user-initiated dashboard handoff to test the end-to-end flow:

```
16:53:00  hand_off → trend-scout              task A = task_1a974ca62a0d   (origin = dashboard human)
16:53:04  await_task_event task A timeout_s=180 poll=3s
16:54:19  EXIT terminal_status status=done     ← hop 1: 75s, < 120s, WORKED
16:54:34  await_task_event task B timeout_s=240 poll=3s   (task_225a797e90fe, next hop)
16:56:34  EXIT cancelled, last_status_seen='working',
          reason='await_task_event cancelled (likely tool timeout)'   ← killed at ~120s
          → turn torn down. Operator idle for 9+ minutes. No re-wake. No user report.
```

The cancel arrives at ~120s of *silence on the browser leg*, not at any server-side tool ceiling (`_TOOL_TIMEOUT=900`). Hop 1 (75s < 120s) survived; hop 2 (silent > 120s) did not.

---

## 2. Root cause (corrected against the code)

The streaming topology for a dashboard chat is one logical pipe across three legs:

```
browser  ──fetch+ReadableStream──▶  dashboard (mounted on mesh :8420)  ──transport.stream_request──▶  agent :8400
   ▲ 120s AbortController                  │ strips ": keepalive", forwards only data:        │ emits ": keepalive" every 15s
   └──────────────── resets on ANY chunk ──┘ (no keepalive of its own)                         └── (server.py chat_stream)
```

- **Agent→dashboard leg is protected.** The agent's `chat_stream` generator yields `": keepalive\n\n"` every 15s while a tool runs silently (`src/agent/server.py`). Receiving that line resets httpx's read-idle clock at `aiter_lines()`, so the dashboard→agent hop never idles out — even for a 240s tool. **This is why the originally-blamed `transport.py timeout=120` is NOT the killer.**
- **Dashboard→browser leg is unprotected.** `transport.stream_request` forwards only lines starting with `data: ` and **drops** the agent's `": keepalive"` comments (`src/host/transport.py`). The dashboard's `api_chat_stream` / `api_broadcast_stream` generators emit no keepalive of their own (`src/dashboard/server.py`). So during a silent tool the browser receives **zero bytes**.
- **The browser aborts at 120s.** `app.js:8610-8619` arms a 120s `AbortController` that resets on *every* `reader.read()` chunk; with no chunks it fires. The abort closes the fetch → dashboard generator is torn down → its `transport.stream_request` context closes → the agent sees a client disconnect → Starlette cancels the agent's stream coroutine → `CancelledError` lands inside `await_task_event`.

**Two compounding problems remain even if you keep the pipe alive:**
1. **Synchronous watch can't scale.** A multi-hop, multi-minute pipeline is unbounded async work; a chat turn is a bounded synchronous primitive. No keepalive makes block-watching *correct* — it just stops it *crashing*.
2. **No resumption.** Once a turn ends (crash, restart, or normal release), nothing re-engages the operator on pipeline progress. The back-edge `task_event` + wake chain did not re-wake it.

---

## 3. Product vision (the north star — unchanged by review)

The operator should behave like a **project manager**, not a security guard:
- A good PM says "On it," delegates, sets up status reporting, and **goes away** — surfacing only at **milestones, completion, or a blocker**.
- Today's operator is the guard whose shift is **120 seconds**; when it ends, the feed goes unwatched and nobody tells the user.

**Cardinal rule:** silent death is the one experience we make structurally impossible. Every user-initiated pipeline ends in a user-facing outcome — **done, failed, or stalled** — pushed to the channel the request came from.

**Experience, before → after:**
> Before: "Do X" → 9-min frozen spinner → silence → user re-asks → operator has forgotten.
> After: "Do X" → instant "On it — trend-scout researching; I'll keep you posted" → "✅ Research done → drafting" → "✅ Draft → polishing" → "🎉 Done, here's the result." If jammed: "⚠️ Stuck at stage 2, need your call on X."

---

## 4. Architecture: delegate-and-subscribe + push

Three responsibilities, cleanly split:

1. **Operator (LLM, judgment):** hand off → attach the user's `MessageOrigin` to the chain root → reply "On it, I'll report back" → **end the turn**. Re-engaged only for *synthesis* (final result) or *judgment* (a stall needs intervention).
2. **System (durable, cheap, no LLM):** a chain-level watcher keyed on `root_task_id` observes `task_status_changed` events and **pushes** to the origin channel — milestone on progress, result on completion, nudge on stall/failure. Survives operator restarts.
3. **Guarantee layer:** every user-originated root chain MUST reach a terminal user notification, exactly once.

`await_task_event` is **demoted**, not deleted: a short, bounded "confirm the first agent picked it up" within a single turn — capped well under the keepalive-survivable window, graceful return. Never the end-to-end watch.

### What actually exists vs what must be built (verified against code)

| Piece | Status | Evidence |
|---|---|---|
| System-side, no-LLM, origin-targeted delivery to **chat** channels | **EXISTS — the one true reuse** | `_handle_notify_origin` (`src/cli/runtime.py:889`) → `channel.send_to_user` (telegram/discord/slack/whatsapp). Already wired as the lane's `notify_fn`. |
| Delivery to the **dashboard** origin | **MISSING — must build** | `_channel_map` has no `dashboard`; `_handle_notify_origin` drops it (`runtime.py:914-920`). Dashboard users are reached via a *separate* `EventBus`/notification path, and the per-chat SSE is **closed** once the operator releases — so terminal delivery must go through the persistent bell store (`src/dashboard/notifications.py`) + `/ws/events`, not the chat stream. **The cake repro is a dashboard chat, so this is on the critical path.** |
| Chain-level state (know when a multi-hop chain is terminal) | **MISSING — must build** | `lanes.py` is per-single-task; `auto_notify`/`_back_edge_fn` fire on the immediate handoff only. No `root_task_id` grouping anywhere. |
| Proactive "no-orphan" push from `chain_breaks_24h` | **MISSING — pull-only** | `chain_breaks_24h` (`src/host/orchestration.py:961`) is "observability-only, NO enforcement," called only by `GET /mesh/system/metrics`. Nothing runs it on a timer. A background watcher (or cron target) is new. |
| Origin on the `task_status_changed` event payload | **PARTIAL** | Origin is persisted on the task row (`orchestration.py` origin_kind/channel/user) but **not** on the event payload — a watcher must `get(task_id)` and walk `parent_task_id` to the root. |

**Conclusion:** Phase 1 builds a small but real durable subsystem: a `chain_watch` table, a restart-recoverable watcher, exactly-once terminal delivery, and a dashboard-capable origin→channel delivery function. Only the chat-channel delivery is genuine reuse.

### Security constraint (load-bearing — do NOT regress)

Per-hop origin is deliberately **downgraded** for worker-originated calls (`_validated_origin`, `src/host/server.py:458-466,506-513`), and the back-edge wake **deliberately skips distant origins where `origin_user != creator`** (`src/host/server.py:5351-5374`). That is an intentional trust boundary: a deep worker must not be able to address an arbitrary human as if it were them.

**Therefore the watcher must NOT trust per-hop origin propagation and must NOT punch a hole in that skip.** Instead:
- The authoritative **root origin is captured ONCE at chain creation** (the user-originated `hand_off`/`create_task`, where the origin is first-party and trusted) and stored on the `chain_watch` row keyed by `root_task_id`.
- All terminal/milestone delivery reads the origin **from the `chain_watch` row**, never from a mid-chain task's (possibly downgraded) origin, and never re-opens the back-edge skip.
- This keeps the existing boundary fully intact while still letting the *system* (not a worker) deliver to the original human.

### Primitives referenced
- `src/host/orchestration.py` `update_status` — central status-transition point; emits `task_status_changed` + `task_completed_without_handoff`. The watcher's hook.
- `src/cli/runtime.py:889 _handle_notify_origin` → `channel.send_to_user` — chat-channel delivery (reuse).
- `src/dashboard/notifications.py` (persistent bell) + `/ws/events` — dashboard delivery (extend).
- `src/host/user_notifications.py` — observation log (PULL; does not deliver). Useful for audit, not for push.
- `MessageOrigin` propagation (Constraint #4) for the *creation* stamp only.

---

## 5. Roadmap (re-sequenced: net before trapeze)

> **Ordering rule (review fix):** the operator behavior change (stop block-watching, acknowledge-and-release) is the **LAST** step, gated on a green end-to-end watcher *including dashboard delivery*. Changing operator behavior before the watcher reliably delivers would replace "silent death after 120s" with "silent death immediately" — strictly worse.

### Phase 0 — Bleeder-stopper: keepalive forwarding (IMPLEMENTED in this PR)
**Goal:** no streaming hop goes byte-silent; long quiet tool calls stop cancelling the turn. General fix, zero security surface, independently shippable.
**Files:** `src/host/transport.py`, `src/dashboard/server.py` (both stream generators), `src/cli/runtime.py`.
**Changes (done):**
- `transport.stream_request` forwards the agent's `": keepalive"` SSE comments as a `{"type": "keepalive"}` liveness sentinel instead of dropping them. (True end-to-end liveness: if the agent loop wedges, keepalives stop and downstream correctly times out.)
- Dashboard `api_chat_stream` + `api_broadcast_stream` re-emit the sentinel as a real `": keepalive"` SSE comment to the browser (resets `app.js`'s 120s `AbortController`); never a data event, never counted toward completion.
- CLI `runtime.py` stream consumer drops the sentinel so it can't contaminate a channel's assembled response. (REPL already ignores unknown event types.)
**Explicitly NOT done (and why):**
- **Do NOT swallow `CancelledError`** in `await_task_event`. The cancel originates from a client disconnect — by the time it fires the channel is already gone, so returning a value is futile; and the loop intentionally re-raises chat cancellation (`loop.py:3858-3860`). The current re-raise is the safe behavior.
- **Do NOT add an await re-arm loop.** Keeping the operator spinning in a poll loop under the keepalive window just monopolizes it and burns tokens — Phase 1 removes block-watching entirely.
**Acceptance:** a 200s+ silent tool call no longer cancels the turn; the browser stays connected (keepalive comments visible on the wire). Unit tests: transport forwards keepalive sentinels; both dashboard generators re-emit them as comments (added).
**Note:** Phase 0 stops the *crash*. It does NOT make watching scale across many hops/minutes — Phase 1 does. Do not stop here.

### Phase 1 — The guarantee: delegate-and-release + durable terminal outcome
**Goal:** every user-initiated pipeline ALWAYS produces a final user-facing message (done/failed/stalled), exactly once; the operator stops block-watching.

> **Status: 1a–1d IMPLEMENTED.** 1a–1c (the guarantee net) shipped + verified live on cake; **1d (operator delegate-and-release) now shipped** — the operator hands off and ends the turn instead of block-watching. `await_task_event` is demoted to a short (<120s, default 45/cap 90) in-turn pickup confirmation, and `_OPERATOR_CORE` carries an explicit "Delegate and release" rule. Net-before-trapeze ordering honored: 1d landed only after the watcher was proven delivering on cake.
> The net ships as: a `chain_deliveries` exactly-once ledger + `Tasks.{list_watchable_human_roots, chain_terminal_verdict, claim_chain_delivery}` (`src/host/orchestration.py`); a periodic, restart-safe `ChainWatcher` with settle/debounce against the in-flight-handoff race (`src/host/chain_watcher.py`); and runtime wiring that delivers a guaranteed terminal outcome to the dashboard bell (durable, deliver-then-claim) plus a best-effort paired-channel push — targeting **only** the root's first-party human origin (`src/cli/runtime.py`). The operator still block-watches for now; the net is purely additive so it cannot regress current behavior. 1d removes block-watching once this is proven on cake.

- **1a (gate — design + schema, tests first):**
  - Decide the root-origin trust model per §4 security constraint: `chain_watch` row carries the trusted root origin set at chain creation; delivery never trusts per-hop origin and never re-opens the back-edge skip.
  - Define **"chain terminal"** for fan-out: no non-terminal task remains under `root_task_id` (not merely "a `done` leaf with no child" — that's `chain_breaks`'s weaker per-leaf notion).
  - `chain_watch` schema (SQLite, WAL): `root_task_id` PK, root origin (kind/channel/user), last-notified status, created_at, terminal_notified flag. **Unique key `(root_task_id, terminal_kind)` for exactly-once.**
  - Audit/guarantee child←parent origin inheritance through every hop *for attribution only* (root lookup), since per-hop origin is downgraded.
- **1b (watcher):** a restart-recoverable background watcher subscribed to `update_status` transitions (the `EventBus` is in-memory/ephemeral — the watcher must **re-scan open `chain_watch` rows on boot** to resume/close orphans). On chain terminal, fire exactly-once terminal delivery (idempotent via the unique key; safe against `update_status` re-fires and restart replays).
- **1c (delivery):** unified system-side delivery covering **both** chat channels (`_handle_notify_origin`) **and** dashboard (persistent bell row + `/ws/events`). Test telegram and dashboard paths.
- **1d (operator behavior — LAST) — DONE:** `_OPERATOR_CORE` "Routing Work" now carries a **Delegate and release** rule (hand off → tell the user → stop; don't chain `await_task_event` to babysit a pipeline; the watcher delivers the outcome). `await_task_event` is demoted: description reframed to a quick in-turn pickup check, cap lowered 270→90 / default 240→45 (kept under the 120s streaming idle so it always returns cleanly). Shipped after 1a–1c were proven live on cake.

**Acceptance (the headline test):** a 3-hop pipeline where each hop takes >120s and the whole thing spans multiple turns/minutes ⇒ operator replies instantly, ends its turn, and the user (dashboard AND a paired chat channel) receives exactly one terminal result (or failure/stall) message — with the operator NEVER stranded, including across a mesh restart mid-pipeline.

### Phase 2 — Ambient progress: per-transition milestones
**Goal:** the "watch it move through the pipeline" feel, cheaply.
**Decision (confirmed):** **terminal-only for v1.** Per-stage milestone pings re-open a *deliberately removed* behavior (PR-3 of the Work-tab rewrite removed the per-task-completion bell producer; CLAUDE.md `notifications.py` note). Do NOT reintroduce per-stage dings until the terminal path is proven and the noise/throttle policy is agreed. When added: templated milestones on **stage completion only**, debounced/coalesced; LLM only for the final synthesis.
**Acceptance:** user sees milestone pings as stages complete; fast chains don't flood the channel.

### Phase 3 — Polish: stall watchdog + live pipeline card
- **Stall watchdog — DONE.** The `ChainWatcher` now nudges the user once when a chain is *parked*: non-terminal but with nothing `working` (stuck in `blocked`/`pending`/`accepted`) and no progress for `stall_after_s` (default 600s). This precisely covers the lane-watchdog blind spot — the lane watchdog only times out *actively-dispatched* (`working`) tasks into `failed`; a chain parked in a waiting state was the silent hole. Store: `chain_stall_state(root)` (last-progress ts iff parked) + `claim_chain_stall(root)` (separate `chain_stall_notices` ledger — a chain can get a stall nudge AND, later, a terminal delivery). Delivery: `_deliver_chain_outcome` `kind="stall"` → an `alert` bell ("⏳ Taking longer than expected … want me to check in?"). Advisory claim-then-deliver (the terminal delivery is the real guarantee). The "stall" promise was restored to the operator prompt + `await_task_event` description now that it's backed. `working`-but-slow chains are NOT nudged (they're progressing; a genuinely hung one is the lane watchdog's job).
- **Work-tab live pipeline card — remaining.** At-a-glance status of in-flight user-originated chains (dashboard UI work).
**Acceptance:** a parked chain pings the user within ~N minutes (✅ tested); the Work tab shows live pipeline state (pending).

---

## 6. Decisions (confirmed 2026-06-10) + remaining open questions

**Confirmed by the user ("confirm and proceed; do not break our security posture"):**
- **D1.** Phase 0 ships first as a standalone keepalive fix (this PR). ✅
- **D2.** Phase 2 is **terminal-only for v1** — no per-stage dings until proven (avoids re-litigating the removed bell producer).
- **D3.** Dashboard terminal delivery goes through the **persistent bell + `/ws/events`**, not the per-chat SSE.
- **D4.** Phase 1 is accepted as a small durable subsystem (chain_watch table + watcher + dashboard delivery + origin-trust model), not a wiring task.
- **D5 (security).** The watcher captures the root origin once at chain creation and delivers system-side; it does **not** loosen `_validated_origin`'s downgrade or the back-edge `origin_user != creator` skip.

**Still open (decide early in Phase 1):**
- **Q1.** `chain_watch` table vs reuse `task_events` for chain state. (Recommend: dedicated table — durable, survives restarts, clean unique key.)
- **Q2.** Progress cadence/throttle policy for Phase 2 (debounce window; templated copy).
- **Q3.** Concurrency: multiple simultaneous pipelines per user/channel — threading/attribution so the user can tell them apart (`root_task_id` in the row + a human-readable chain label).
- **Q4.** Backwards-compat: chains created without a captured root origin — fall back to a dashboard bell entry keyed by the originating session, and lean on the watcher's boot re-scan to close orphans.
- **Q5 (DECISION — surfaced in the #1088 post-merge review).** The watcher covers **operator-rooted** chains only. A *direct* user→worker→sub-agent chat produces a root created by an untrusted worker, whose `kind="human"` origin is correctly downgraded to `agent` by `_validated_origin` — so it is not watched (delivering on it would trust a forgeable origin). Per CLAUDE.md Constraint #1 ("users talk to agents directly") this topology is common. Covering it **safely** needs a trusted server-side chain registration at the dashboard `/chat` entry (the mesh knows the inbound chat was human there) rather than loosening the downgrade. **Open question for the user: is operator-rooted coverage sufficient for v1, or do we want the trusted-registration follow-up (Phase 1.5)?** Until then, direct-worker chains keep today's behavior (no terminal push) — purely a non-regression gap.

---

## 7. Non-goals / scope guards

- Not rewriting the lane/task system — hook `update_status` events.
- Not making `await_task_event` block longer; it's demoted to short in-turn confirmations.
- Not building a generic workflow engine — just reliable, origin-aware **watch + push** for user-initiated chains.
- Not loosening any origin/permission boundary (D5).
- Keep the operator's *judgment* role (synthesis, intervention) LLM-driven; keep *plumbing* (progress pings, terminal delivery) system-driven/templated.

---

## 8. Verify on cake after implementation

- **Phase 0:** trigger a long silent tool over a dashboard chat (e.g. an `await_task_event`/`run_command` > 120s). Confirm the browser SSE stays connected (keepalive comments on the wire) and the turn is no longer cancelled; `docker logs openlegion_operator` shows no `await_task_event EXIT cancelled (likely tool timeout)` from the idle path.
- **Phase 1:** re-run the exact failing flow — multi-hop team pipeline where a stage takes >120s. Confirm: operator replies instantly and ends its turn; user receives a final result/failure/stall message exactly once on the dashboard (and a paired chat channel); operator never strands (no 9-minute idle gap); survives an operator/mesh restart mid-pipeline.
- Deploy method (this repo → cake): merge to `main`, then on `root@cake.engine.openlegion.ai`: `cd /opt/openlegion/repo && git checkout . && git pull && .venv/bin/pip install -e '.[channels,wallet]' && docker build -t openlegion-agent:latest -f Dockerfile.agent . && systemctl restart openlegion`. The engine rebuilds the browser image at startup (~90s) before the mesh binds `:8420`. Do NOT read `.env`/secrets during verification — health + functional checks only.

---

## 9. Appendix — exact code locations (corrected)

| Concern | Location |
|---|---|
| Browser 120s idle abort (the real cliff) | `src/dashboard/static/js/app.js:8608-8620` (`AbortController`, `_resetStreamTimeout` on each chunk) |
| Agent keepalive (protects only agent→dashboard) | `src/agent/server.py` `chat_stream` `event_generator` — `yield ": keepalive\n\n"` every 15s |
| Keepalive stripped / now forwarded | `src/host/transport.py` `stream_request` (`aiter_lines`; forwards `data:`, now also `:` comments as `{"type":"keepalive"}`) |
| Dashboard stream generators (re-emit keepalive) | `src/dashboard/server.py` `api_chat_stream`, `api_broadcast_stream` |
| Blocking watch tool | `src/agent/builtins/operator_tools.py` `await_task_event` (~1136–1329); `CancelledError` handler re-raises (~1287 — keep as-is) |
| Per-tool hard ceiling (NOT the killer) | `src/agent/loop.py:44` `_TOOL_TIMEOUT=900`; wrap ~3105 `asyncio.wait_for(..., _TOOL_TIMEOUT)` |
| Status transition + events (watcher hook) | `src/host/orchestration.py` `update_status`; `task_status_changed`; `task_completed_without_handoff` |
| Silent-termination detector (pull-only) | `src/host/orchestration.py:961 chain_breaks_24h` (only caller: `GET /mesh/system/metrics`) |
| Chat-channel delivery (reuse) | `src/cli/runtime.py:889 _handle_notify_origin` → `channel.send_to_user` |
| Dashboard delivery (extend) | `src/dashboard/notifications.py` (bell) + `/ws/events`; `EventBus` is in-memory (`src/dashboard/events.py`) |
| Origin downgrade + back-edge skip (do NOT regress) | `src/host/server.py:458-466,506-513` (`_validated_origin`); `src/host/server.py:5351-5374` (`origin_user != creator` skip) |
| Origin stamp (dashboard) | `src/dashboard/server.py` `api_chat_stream` (`MessageOrigin(kind="human", channel="dashboard")`) |

**Reconciliation note.** This revision merges a principal-engineer review with an independent Codex pass. Both independently concluded: the root cause was mislocated (it's the browser leg, not `transport.py`), "reuse not new machinery" was overstated, and `CancelledError` must not be swallowed. Codex additionally pinned the exact browser-abort site (`app.js:8608-8620`) and surfaced the origin-downgrade / back-edge-skip security collision (`server.py:458-466,5351-5374`) that the §4 trust model now addresses.

**Related prior art:** `docs/plans/2026-05-02-operator-orchestration-roadmap.md`, `docs/plans/2026-06-09-operator-memory-context-overhaul.md`. PR #1068 (await_task_event polling — partial fix this supersedes for the watch path).
