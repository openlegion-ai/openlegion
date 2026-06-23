# Chat-native delivery + bell removal

**Date:** 2026-06-11
**Status:** SHIPPED — archived. Chat-native delivery + bell removal shipped/deployed/live-verified (PRs #1137, #1139, #1140, #1143).

## Verified root causes (current main, `65524416`)

1. **Bell-only delivery for dashboard origins.** `RuntimeContext._deliver_chain_outcome`
   (src/cli/runtime.py:976) writes the `NotificationStore` row — that write IS the
   deliver-then-claim durability point (`ChainWatcher._process_root` claims only on `True`,
   src/host/chain_watcher.py:164-190) — then emits `notification_added` (bell badge only) and
   skips the channel push for `channel == "dashboard"`. No chat bubble ever.
2. **Async operator turns are invisible.** Lane wakes (verification wake
   `_maybe_wake_operator_verification` runtime.py:1126, recovery wake
   `_wake_operator_for_human_chain` host/server.py, back-edge wakes, admin-action wakes) dispatch
   via `_direct_dispatch` → agent `POST /chat`; the reply persists to the transcript with **no
   event emitted**. The chat UI reloads history only on user actions (send / tab switch / WS
   reconnect) — hence "I only got notified after I sent a message."
3. **Wake prompts persist as `role="user"`** (`_prepare_chat_turn` →
   `workspace.append_chat_message("user", ...)`, src/agent/loop.py) and the user template renders
   every user row as the human's own bubble. Confirmed live on cake.
4. **No watching affordance.** The Work tab's live pipeline card exists; the chat thread shows
   nothing while a chain is in flight.

Load-bearing facts (verified against code, incl. spot-checks during review):

- The chat transcript (`chat_transcript.jsonl`, workspace.py) is **display-only** — LLM context is
  the in-memory `_chat_messages` list, restored from memory-DB checkpoints (loop.py:2838-2849),
  never from the transcript. New roles / extra fields on transcript rows cannot corrupt context.
- **`append_chat_message` swallows ALL write failures at debug level** (workspace.py:1067-1076)
  — any endpoint acking "written" must use a raising write path or the ack lies.
- **`HttpTransport.request` never raises for the failures that matter** — HTTP errors, timeouts,
  connect failures return `{"error": ...}` dicts (transport.py:137-145), HTTP errors with
  `status_code`. Success/failure must be judged on the returned dict + a positive endpoint ack.
- **HttpTransport keeps one AsyncClient per event loop** (`_clients` keyed by `id(loop)`,
  transport.py:91-102) — awaiting `transport.request` on the chain watcher's own loop is safe.
  The watcher runs its own loop in a daemon thread (runtime.py:1611-1621) and `_run_deliver`
  already supports an async deliver fn (`iscoroutine`, chain_watcher.py:191-194).
- The dashboard main chat template hides unknown roles (no fallback); existing `system` role
  renders as a de-emphasized divider; `notification` role renders the amber bubble both from
  history load and the live `notification` WS event (2s dedupe window). NOTE (second pass): a
  worker-chat template variant has a catch-all that renders unknown roles unstyled — verify both
  templates render `system` rows acceptably during PR 1.
- The agent server has no auth (network isolation: icc=false + loopback-only port publish;
  `x-mesh-internal` gates origin-kind trust only); `POST /message` is the pattern for a no-LLM
  durable-write endpoint. `/chat/note` adds no new trust surface beyond existing `/chat`.
- Bell producer inventory: P1 approval, P2/P3 credential requests, P4 health alerts,
  P5 credit_exhausted all have alternate surfaces. **Bell-only:** P6 `connection_refresh_failed`,
  P7 quarantine remediation text (written DIRECTLY by HealthMonitor, health.py:175-187 — not via
  the producer, whose health branch covers only degraded/unhealthy/failed), P8 chain outcomes.
  The desktop Browser-Notification hook (`_maybeFireBrowserNotification`,
  `_browserNotifyKinds = ['approval','credential','alert','blocker']`) is fed exclusively by
  bell rows — chain-`done` (kind `delivered`) never desktop-pinged even today (accident).
- `GET /api/workplace/pipelines` omits `origin`/`updated_at`; the payload **filters out wholly
  terminal chains immediately** (status-based, not claim-based) while the outcome bubble lands
  only after settle (30s) + sweep — a 30-60s dead window PR 4 must bridge client-side.
- Lane enqueue call-site audit: system-composed wakes = blackboard watch notifications
  (**mode="steer"**, falls back to followup when idle), `hand_off` wakes, recovery wake,
  back-edge wakes, admin-action wakes (reroute/retry/cancel), verification wake, cron dispatches,
  webhook dispatches (via `async_dispatch` funnel). Human-relayed lane traffic = channel inbound
  + steer paths. Recovery/verification wakes deliberately stamp `origin.kind="human"` (delivery
  routing), so **origin kind cannot distinguish system wakes — an explicit flag must ride the
  dispatch**. The flag must survive THREE paths: followup, the steer→followup fallback inside
  `_handle_steer` (lanes.py:333-351, currently drops all kwargs), and the agent-side busy
  steer-queue (`loop.chat` redirects to `_steer_queue` when `current_task` is set, loop.py:2973;
  drained entries persist as `[steer]` **user** rows at loop.py:3428/3760/4017/5069).

## Design principle

**The chat watches, not the LLM.** No re-opened await loops, no held lanes, no token spend on
progress UI. Real task state drives a frontend chip; real operator turns (verification, recovery)
render live; the transcript becomes the durable delivery record, replacing the bell.

---

## PR 1 — System wakes stop rendering as the user

Thread an explicit `system_note` flag (one name everywhere) from enqueue to transcript
persistence; persist those inbound messages with transcript role `"system"` (existing
de-emphasized divider rendering) while the in-memory LLM message stays `role="user"`.

- `src/host/lanes.py`: `enqueue(..., system_note: bool = False)` → `QueuedTask.system_note` →
  `dispatch_kwargs` only when True. **Must also thread through the steer arm**: `_handle_steer`'s
  three followup fallbacks (no-steer_fn, idle-agent wakeup, steer-error) currently drop all
  kwargs — carry `system_note` (and while there, keep the existing behavior for other kwargs
  unchanged).
- `src/cli/runtime.py`: `_direct_dispatch` accepts `system_note=False`, sends header
  `x-system-wake: 1` (precedent: `x-task-id`, `X-Origin`). `dispatch` / `async_dispatch` funnels
  gain the param (cron + webhooks route through them).
- `src/agent/server.py` `POST /chat`: read the header — trusted only when `x-mesh-internal`
  present (same gate as origin kind) — pass `system_note=True` into `loop.chat(...)`.
  `/chat/stream` is NOT touched (no system wake rides the stream path).
- `src/agent/loop.py`:
  - `_prepare_chat_turn`: when `system_note`, persist transcript row as
    `append_chat_message("system", user_message)`; in-memory append unchanged. Also **skip
    correction-recording (`looks_like_correction`) and the first-message memory auto-search**
    for system notes — wake boilerplate must not pollute the corrections store or seed memory.
  - **Busy path**: `_steer_queue` entries become `(text, system_note)` tuples (internal queue +
    `inject_steer` gains the flag, default False); the four drain persist sites write
    role `"system"` (no `[steer]` prefix) for flagged entries, `[steer]` user rows as today
    otherwise. Without this, a verification wake during a running task still renders as a user
    bubble — the most common real case.
- Flag at system-composed call sites: blackboard watch notify (steer), `wake_agent` (hand_off),
  `_wake_operator_for_human_chain`, back-edge originator wake, `_try_wake_agent`
  (reroute/retry/cancel), `_maybe_wake_operator_verification`, `cron_dispatch`, webhook dispatch.
  Human-relayed paths (channel inbound, user steer endpoints) stay unflagged. Human-triggered
  system-composed steers (credential saved/cancelled etc.) keep their `[steer]` rendering — out
  of scope.
- UI: long `system` rows get a line-clamp + click-to-expand in the system template; verify the
  worker-chat catch-all template variant renders system rows acceptably (second-pass catch).
- Historical `user`-role wake rows on existing deployments stay as-is (display-only; no
  migration).

Tests: lanes threading incl. steer-fallback kwarg carry, dispatch header emission, /chat
header→kwarg trust gate, transcript role persist (flag on/off), busy-path steer-queue tuple
drain, corrections-skip, template clamp presence.

## PR 2 — Durable chat delivery + live async turns (bell replacement, load-bearing)

- **New agent endpoint `POST /chat/note`** (`src/agent/server.py`, body `{"message": str}`,
  bounded ~2000 chars mirroring `_NOTIFY_MAX_LEN`): appends a `"notification"`-role transcript
  row via a **raising write path** — `append_chat_message` swallows failures by design, so add
  `raise_on_error=True` (or equivalent) and return `{"ok": true}` only on a successful write.
  The ack must not lie; it is the new durability point. Renders identically to `notify_user`
  rows on history load — zero new frontend.
- **`_deliver_chain_outcome` reroute** (src/cli/runtime.py:976): convert to **async def**
  (`_run_deliver` supports coroutines; per-loop transport client makes awaiting on the watcher
  loop safe) and replace the `NotificationStore.add` durability write with
  `await transport.request("operator", "POST", "/chat/note", ...)` carrying `f"{title}\n{body}"`.
  **Success = returned dict has no `"error"` key AND carries the endpoint's positive ack** —
  transport never raises for HTTP/timeout/connect failures. Failure → log + `return False`
  (watcher leaves the root unclaimed, retries next sweep — claim machinery untouched, pinned by
  `test_chain_watcher.py::test_failed_delivery_is_retried_not_lost`). **`status_code == 404`
  logs at ERROR with an explicit "agent image predates /chat/note — rebuild required" message**
  (the git-pull-without-image-rebuild deploy trap would otherwise retry silently forever).
  On success emit the existing **`notification`** DashboardEvent (`agent="operator"`) → live
  amber bubble + toast. Keep: kind branches (done/stall/failed), 1500-char bound, channel push
  for non-dashboard origins, verification wake. Drop: `notification_added` emit for chain
  outcomes.
  - Note target is **operator** unconditionally — second pass confirmed
    `list_watchable_human_roots` is operator-rooted by construction; no fronting-worker case
    exists today.
- **ChainWatcher startup gate**: re-gate from `_notification_store is not None` to transport
  availability.
- **Live render of async turns**: in `_direct_dispatch`, after a successful dispatch, emit
  `chat_done` with `data={"source": "dispatch"}` and **no `response` field**. JS handler changes
  (required, the current handler is wrong for this): for `source === "dispatch"`, skip the
  remote-bubble finalize branch entirely and call `_loadChatHistory(agent)` **without** deleting
  `_chatFetchedAt[agent]` (the current handler deliberately bypasses the 5s debounce — a blanket
  bypass on every lane dispatch fleet-wide would stampede history fetches in every connected
  session).
- **Desktop ping bridge (3 lines of JS, closes the PR2→PR3 window):**
  `_maybeFireBrowserNotification` additionally fires on live `notification` WS events
  (synthesized `{kind:'delivered', title}` row) so chain outcomes desktop-ping once bell rows
  stop minting. Full fan-in lands in PR 3.

Tests: `/chat/note` endpoint (success ack; raising-write failure → non-ok); rewritten
`TestDeliverChainOutcomeBellKind` against the note write + `notification` emit, error-dict →
`False`, 404 → ERROR log; `chat_done` dispatch-source emit + JS handler guard (template test).

**Deploy note:** PRs 1–2 touch `src/agent/**` → agent image rebuild
(`docker build -f Dockerfile.agent .`) + restart, not just git pull. Ship both in one deploy
window.

## PR 3 — Bell removal end-to-end (+ the two reroutes)

Delete:

- `src/dashboard/notifications.py` (store + `_KNOWN_KINDS`), runtime construction/wiring
  (`_notification_store`, dashboard router param + auto-init, HealthMonitor injection),
  endpoints `GET /api/notifications` + read/read-all, `_emit_notification_added` +
  `_notifications_producer` (P1–P5 have alternate surfaces), `notification_added` literal in
  `src/shared/types.py`, JS bell (state, 60s poll, WS handler, fetch/mark/icons), both bell
  markup blocks, bell test surface (`tests/test_dashboard_notifications.py` whole file,
  bell-markup UI tests, quarantine-bell test).

Reroute (the two genuinely bell-only signals) via a slim event listener using the PR-2 note
path — operator transcript note + `notification` event:

- **P6 `connection_refresh_failed`** → "OAuth connection '{service}' is dead — reconnect it on
  the Connectors page."
- **P7 quarantine**: the listener needs an explicit `health_change` → `current == "quarantined"`
  branch (today's bell row comes from HealthMonitor's direct store write, NOT the producer);
  note carries the remediation text. `HealthMonitor` drops its `notifications_store` param.
- **Threading + delivery semantics (second-pass catch):** EventBus callbacks are sync on the
  emitter's thread — marshal the note POST via `run_coroutine_threadsafe` onto the dispatch loop
  and reuse the bounded-retry pattern (`_channel_push_with_retry` numbers). After retries
  exhaust, the signal is accepted-lost to the transcript (fleet badges / Connectors page remain)
  — explicit decision, see flagged list.

Rewire desktop notifications (don't delete): `_maybeFireBrowserNotification` moves from bell
rows to a fan-in over the underlying WS events — `pending_action_created` (kind approval),
`credential_request` / `browser_login_request` (credential), `health_change` degraded+quarantined
(alert), and `notification` (delivered) — synthesizing the `{kind, id, title}` row the hook
expects (live `notification` events carry only `message`; no kind/id — synthesize). This
preserves today's approval/credential/alert coverage and ADDS completion pings (new behavior,
flagged).

Docs: update CLAUDE.md (`src/dashboard/notifications.py` row, kinds note). Existing
`dashboard_notifications.db` files orphaned — harmless; remove manually on deploy.

## PR 4 — In-chat watching chip + playbook v6

- **Backend (one line):** pass `origin` and `updated_at` through in `/api/workplace/pipelines`.
- **Chat UI chip:** operator thread, pipelines with `origin.channel === "dashboard"`: pinned
  compact element — pulsing dot, root title, `stage k/n · <assignee> <status> · <age>`; amber +
  "stalled" when flagged; click → Work tab drill-in; cap 3 chips + "+N more". Binds to existing
  `workplacePipelines` state + WS task-event debounce refresh; also load on chat-tab
  entry/initial mount (chat is the default tab).
- **Dead-window bridge (second-pass catch):** the payload drops a chain the moment it's wholly
  terminal, but the outcome bubble lands after settle(30s)+sweep. When a tracked chip's root
  leaves the payload, keep a client-side "finishing…" ghost state, cleared by the matching
  `notification` event or a 90s timeout — no backend change.
- **Playbook v6** (`playbook_v6_chat_delivery`, established sentinel machinery; mind the
  `_OPERATOR_CORE` size-cap test): after delegating, say exactly what the system now does —
  *progress shows live in this chat (watch chip + stage updates), and the outcome plus my
  verification are posted here automatically*. Never reference the bell; v5's
  no-unbacked-promises rule stands.

Tests: pipelines payload passthrough; chip render/hide + ghost-state; playbook sentinel tests
pin `PLAYBOOK_SENTINELS[-1]` dynamically.

---

## Sequencing, verification, rollout

- Order: PR 1 → PR 2 → PR 3 → PR 4 (3 depends on 2's note path; 2's desktop-ping bridge closes
  the 2→3 gap; 4 last for copy). Worktrees for all code; tests in subagents; retarget stacked
  PRs to main via REST before merging a base; CI waits check `fail` explicitly.
- Cake live verification (through the real browser path — `/chat/stream` lesson from #1112):
  (a) wake → transcript row role `system` incl. the BUSY case (wake during a running task must
  not produce a `[steer]` user bubble); (b) chain done → operator transcript `notification` row
  + live `notification` event + claim recorded; operator container stopped → delivery returns
  False, root re-delivered next sweep after restart; stale-image 404 drill → ERROR log present;
  (c) bell endpoints 404, no `notification_added` emits; desktop fan-in fires per kind;
  (d) pipelines payload carries `origin`; chip visible during an in-flight probe chain; ghost
  state bridges to the bubble; (e) verification turn appears live without any user message.
- Fleet rollout: agent-image rebuild required (PRs 1–2); other 4 boxes get everything on their
  next normal deploy — deploy MUST include the image rebuild or chain delivery 404-loops (the
  ERROR log is the tripwire).

## Flagged decisions (defaults chosen — confirm or veto)

1. Outcome notes always land in the **operator** thread (forced today: watchable roots are
   operator-rooted by construction).
2. Human-triggered system-composed steers keep `[steer]` user-row rendering — out of scope.
3. Historical wake rows already persisted as `user` are not migrated.
4. Desktop browser notifications are kept via event fan-in; **chain completions now desktop-ping
   for the first time** (success never pinged before — kind accident). New behavior.
5. Non-dashboard (telegram/etc.) chains: if the channel push exhausts its 3 retries, the only
   durable record is the operator-thread note — weaker than the old server-side bell row with
   unread state. Accepted as the cost of bell removal.
6. **Server-side unread state dies with the bell.** Unread is now per-browser-session; an
   away-for-a-day user reloads into bubbles with no unread badge. Transcript presence + desktop
   ping is the signal.
7. P6/P7 reroutes are bounded-retry then accepted-lost if the operator container is down (bell
   was host-side SQLite and ~never failed); fleet badges / Connectors page remain the fallback
   surface.
8. Webhook/cron inbound messages render as dim `system` dividers (previously fake user bubbles).
   Webhook bodies are external input rendered in an "internal authority" style — content is
   sanitized at the /chat boundary; accepted.

## Review log (2026-06-11)

Two-pass pre-implementation review. Self-pass caught: busy-path steer-queue gap, watcher
loop/transport affinity (resolved by per-loop clients + async deliver), chat_done cross-session
finalize risk. Independent fresh-context second pass (Codex quota-blocked; disclosed substitute)
additionally caught: transport error-dict contract (no raise), swallowed transcript-write
failures behind the ack, steer→followup fallback kwarg drop, chat_done debounce bypass, desktop
notification fan-in being wrong on both ends, P7 quarantine direct-write (not producer), old-image
404 trap, chip dead window, correction-store pollution by wake text. All incorporated above;
spot-verified in code before adoption (workspace.py:1067-1076, lanes.py:333-351, app.js
chat_done handler, transport.py:137-145).
