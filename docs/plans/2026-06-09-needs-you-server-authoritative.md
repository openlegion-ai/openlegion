# Plan: make "Needs you" a trustworthy, server-authoritative list of resolvable blockers

**Status:** ACTIVE — draft for review. (Supersedes the chat-scrape parts of PR #1083.)

## Problem (restated)
"Needs you" must mean: something is **blocking** the user's agents, the user is
**notified**, **told exactly how to resolve it**, and **can resolve it**. If an
item isn't blocking-and-resolvable, it should not appear at all. An empty panel
must reliably mean "nothing needs you."

## Why the current implementation falls short
The panel mixes server-authoritative sources (pending actions ← `/mesh/pending`;
blockers ← `/api/workplace/blockers`) with **volatile client state**:
- credential / browser_login / captcha items are scraped from
  `chatHistories['operator']`. Those cards are **synthesized client-side from
  live WS events** (`app.js:4418-4419`) — they are NOT in the agent transcript.
  `_loadChatHistory('operator')` rebuilds the array from the server transcript
  and preserves only `user`/`agent`/`notification` roles, so the request cards
  are **dropped** on reload / tab-switch / `openChat('operator')` (which our own
  "Sign in" button calls). On reload they are gone entirely (the mesh
  `help_requests` registry is in-memory with no reconnect replay).
- worker_dm (← `chatUnread`) is a notification, not a blocking+resolvable event.

Result: a blocked agent's credential request can silently vanish from the panel
while still open — the dead-end, in a worse form.

## Decisions (confirmed with product owner 2026-06-09)
1. **Source of truth:** expose the mesh open-requests registry via an endpoint;
   source the panel's request items from it (like pending/blockers); fix the
   registry to clear on resolve so the feed has no ghosts.
2. **Scope:** remove worker-DM from "Needs you" (keep the bell/unread dot) and
   delete the dead blocker branch.

## Integration points (verified)
- Registry: `help_requests` dict + `_record_help_request` / `_cancel_help_request`,
  `src/host/server.py` ~907-934, 2743-2787. In-memory, 256-LRU, NOT persisted.
- Auth/listing pattern to mirror: `GET /mesh/pending`, `src/host/server.py:8337`.
- Credential save: `api_add_agent_credential`, `src/dashboard/server.py:4198-4254`
  (already steers the agent; does NOT pop the registry record).
- Browser-login/captcha complete: `src/dashboard/server.py:4256`, `:4283`
  (steers; no request_id path; does NOT pop the record).
- request_id-scoped cancel (mesh, pops record): `src/host/server.py:2789, 2836, 2875`.
- DashboardEvent literals: `src/shared/types.py:760-770`.
- Frontend builder: `needsYouItems` getter, `src/dashboard/static/js/app.js:3077`.
- Browser viewer to reuse for login/captcha: `toggleBrowser` `app.js:9238`,
  standalone viewer markup `index.html:2661`.

## Sequenced work (each step independently testable)

### Step 1 — Backend: open-requests feed
- Add `GET /mesh/help-requests` (operator-or-internal auth, mirror `/mesh/pending`)
  returning open records: `{request_id, kind, agent_id, service, name,
  description, created_at}`. Reap evicted/expired before returning.
- Add dashboard proxy `GET /api/help-requests`.
- Tests: empty by default; a `credential_request` shows up; cancelled drops off.

### Step 2 — Backend: registry clears on resolve (no ghosts)
- Thread `request_id` into the resolution calls so the server pops precisely by id
  (also fixes the pre-existing `(agent_id, service)` collision bug):
  - credential save: accept optional `request_id`; after `add_credential`, call a
    new mesh `POST /mesh/credential-request/{id}/resolve` that pops the record and
    emits `credential_request_resolved` (new literal in `types.py`). Keep
    `credential_stored` for back-compat card sync.
  - browser-login / captcha complete: add request_id-scoped mesh resolve endpoints
    that pop + emit `*_completed`; dashboard complete handler calls them.
- Back-compat: legacy `(agent_id, service)` paths still work (pop-by-match fallback).
- Tests: after save/complete the record is absent from `/mesh/help-requests`;
  cancel still pops; double-resolve is a no-op (no spurious steer).

### Step 3 — Frontend: drive credential items from the feed
- Add `needsYouRequests` array; load in `loadWorkplace` poll + refresh on WS
  events (`credential_request`, `*_completed`, `*_cancelled`, `credential_stored`),
  reconciled by `request_id`.
- Build credential items from `needsYouRequests` (not chat scrape). Keep the inline
  paste-and-save form; add `request_id` to the POST body.
- Tests: feed drives the item; saving removes it after the resolve round-trips.

### Step 4 — Frontend: login/captcha resolution decoupled from the chat card  ⚠ riskiest
- These need the live VNC. Since items are no longer tied to a chat card, the
  action must NOT depend on a card being in memory. **Proposed:** reuse the
  existing browser viewer (`toggleBrowser`/standalone viewer) focused to the
  request's `agent_id`, with a "Complete login" / "Done" action wired to the
  request_id resolve endpoint. (Alt: embed a compact VNC iframe in the expanded
  panel item.) — SUB-DECISION TO CONFIRM before building this step.
- `_getVncUrl(agentId)` agent-scoping fix from #1083 is a prerequisite (keep it).
- Tests: action opens VNC for the correct agent; complete pops the item.

### Step 5 — Frontend: remove non-fitting items
- Delete the `worker_dm` branch (bell/unread dot remains the surface for DMs).
- Delete the dead `blocker` branch and, if fully unused afterward, `_humanizeBlocker`.
- Tests: those kinds never appear; bell still counts DMs.

### Step 6 — Cleanup
- Remove the now-dead chat-scrape code in `needsYouItems`; drop `seenServices`
  dedupe (the feed is already de-duplicated server-side).
- Update tests; refresh the PR description.

## Risks / load-bearing assumptions
- **Durability:** `help_requests` is in-memory; a mesh restart drops open requests
  from the feed AND strands the blocked agent (no steer will ever come). The feed
  inherits this. *Optional follow-up:* persist the registry (SQLite, like other
  state) so restart doesn't strand agents. Out of scope for the must-fix unless
  you want it now.
- **LRU cap (256):** under heavy load old open requests are evicted and never
  shown. Acceptable short-term; note it.
- **Poll vs WS:** need both (poll = completeness, WS = latency); reconcile by
  `request_id` to avoid flicker/dupes.
- **Step 4** is the only piece with real UI uncertainty; everything before it is
  mechanical and independently shippable (credential-only is a complete win on its own).

## Relationship to PR #1083
Keep: inline credential form, SVG icons, imperative copy, `_getVncUrl(agentId)` fix.
Supersede: chat-scrape source, worker_dm, blocker branch, flash-to-chat-card as the
*primary* login/captcha resolution. Recommend evolving #1083 rather than a fresh PR.

## Independent review — Codex reconciliation (revisions OVERRIDE the above)
Codex reviewed this plan (yes-with-changes). Adopted changes:

- **Durability is now MUST-FIX, not optional.** Persist the registry to SQLite
  (WAL), matching the engine's existing stores. A plain in-memory dict cannot
  back "empty = nothing needs you" across a mesh restart. Persisting also retires
  the silent **256-LRU eviction** (`src/host/server.py:919-925`) that drops open
  requests with no event. (Note: the blocked *agent* surviving restart is a
  separate concern; persisting the request at least keeps the panel honest and
  lets the user re-trigger.)
- **Enrich records at insert.** `_record_help_request` stores only name/service
  (`server.py:2586-2656`); `description` rides only the WS event. Store
  `description` (and `created_at`) on the record so the REST feed can render
  what/why.
- **Generic resolve, by id only.** Replace the three per-kind resolve/complete
  families with one `POST /mesh/help-requests/{request_id}/resolve` that
  ATOMICALLY claims/pops by id, THEN dispatches the kind-specific side effect
  (steer / credential write) only if the claim won. already-resolved = no-op.
  This fixes the save/cancel double-steer race and the (agent_id, service)
  collision in one move. Drop the pop-by-match fallback entirely.
- **Error ≠ empty.** The feed proxy must NOT return `{items: []}` on backend
  failure (the pending proxy's `server.py:7596` smell). Surface an explicit
  error/unknown panel state so an outage never reads as "nothing needs you".
- **Wire freshness.** Fetch the feed on initial load, **WS reconnect**
  (`app.js:1260-1310` currently skips workplace), **tab focus**, and poll;
  reconcile WS deltas by request_id. Add help-request events to the workplace WS
  reducer (`app.js:4370-4378`) if WS drives latency.
- **Sequencing correction.** Steps 1-3 do NOT ship independently. A mixed-source
  panel (credential feed-sourced, login/captcha still chat-scraped) is worse than
  either — a vanished login item is indistinguishable from a resolved one. All
  three kinds become feed-sourced together; Step 4's VNC resolution lands in the
  SAME release.
- **Step 4 hardening.** `toggleBrowser` uses `this.selectedAgent`
  (`app.js:9307`) and the standalone iframe uses `agentDetail.vnc_url`
  (`index.html:2664-2726`) — a card for a non-selected agent opens the WRONG
  browser. Pin the viewer to the request's `agent_id` first. And the Needs-You
  path must use exact-agent VNC only: `_getVncUrl` must fail VISIBLY when the
  target agent has no vnc_url, never fall back to another agent's browser.
- **Type literal.** Add any new resolved event to `DashboardEvent`
  (`src/shared/types.py:760-770`) before emitting, or reuse `credential_stored`
  (carries request_id) and skip a new literal.
- **Doc nit:** browser viewer markup is `src/dashboard/templates/index.html`
  (~2664-2726), not `static/`.

### Revised sequence
1. Backend: persist registry (SQLite) + enrich records (description, created_at);
   retire LRU eviction. Testable in isolation.
2. Backend: `GET /mesh/help-requests` feed + dashboard proxy (error ≠ empty).
3. Backend: generic `POST /mesh/help-requests/{id}/resolve` (atomic claim →
   side effect); migrate credential-save and login/captcha-complete to call it;
   keep legacy cancel working.
4. Frontend: `needsYouRequests` from the feed; build ALL THREE kinds from it;
   wire load/reconnect/tab/poll + WS reducer; error state.
5. Frontend: credential inline-save (send request_id); login/captcha resolution
   via agent-pinned browser viewer + resolve endpoint; exact-agent VNC only.
6. Frontend: remove worker_dm + dead blocker branch; drop chat-scrape +
   seenServices; keep icons/copy/_getVncUrl(agentId).
7. Tests + PR description.
