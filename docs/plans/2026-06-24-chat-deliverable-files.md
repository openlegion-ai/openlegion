# Chat-native deliverable files — "your agent's files live in chat"

**Date:** 2026-06-24
**Status:** Phase 1 in progress
**Origin:** Live diagnosis on cake. The `map` agent ran a lead-gen job and
produced real deliverables (`leads_master.csv`, `leads.db`, a sync script,
an 11.6 MB `map.db`). The user asked, verbatim, *"where is that
sheet_sync_appscript.gs"*. The agent **invented** a "files/attachments panel
of this session" that does not exist, then pasted the file contents inline.
That degrades for a 2 KB script and is impossible for the 97 KB CSV / 950 KB
DB / 11.6 MB DB.

## Problem

The egress *backend* already shipped (PR #1037): the agent `/files/{path}`
endpoint is rooted at the whole `/data` volume, and the dashboard
`GET /api/agents/{id}/file-download/{path}` streams any file up to 64 MB as a
`Content-Disposition` attachment. What is missing is any bridge from **chat**
(where the user lives) to it:

1. The file browser is buried — the **Files** sub-tab of a per-agent detail
   view under the **Teams** tab. From chat the user must leave Chat → Teams →
   pick the agent → open detail → switch Config→Files → navigate `/data`.
   Nothing in chat links there.
2. Chat renders agent messages as plain markdown (`renderMarkdown`,
   `app.js`) — no file chip, no download affordance.
3. The agent has no truthful way to hand a file over, so it improvises:
   hallucinates a panel, or inlines bytes. `/chat/note` is plain text;
   `task_artifact_added` routes to the task drawer, not chat.

## CPO framing — what's best for the user

The user's mental model is set by every consumer chat product: **a file made
for you shows up as a thing you click, in the conversation.** They never ask
"where is it." The worst failure here was a *trust* failure — the agent lied
about how to retrieve the work.

So the design has one hard constraint: **whatever surfaces a file must be
backed by a real file, so it can never lie.** That rules out heuristic
prose-parsing as the foundation and points at ground-truth: the agent's
actual `workspace/artifacts/` directory.

This is not a new paradigm — it completes the chat-native-delivery direction
already shipped (#1139 chain outcomes → chat, #1143 watch chip). A file is
just the richest outcome.

## Phases

### Phase 1 — Deliverables affordance in chat (this PR)

Frontend + a prompt change. No schema change, no agent image rebuild for the
dashboard half.

- **Chat-panel "Files" affordance** in BOTH chat surfaces (operator chat,
  keyed `'operator'`; side-chat, keyed `activeChatId`). A button shows the
  deliverable count and toggles an inline panel listing the agent's
  `workspace/artifacts/` entries (name + size) each with a one-click
  **Download** wired to the existing `downloadAgentFile(agentId, path)`.
  Ground-truth backed (lists real files) → never lies. Covers exactly the
  files the map agent produced, works with or without a task context.
- **Truth-telling prompt change** (`workspace.py` system-prompt body): save
  user-facing deliverables to `artifacts/`; the user retrieves them as
  files/cards in the chat; never paste large files inline; never describe a
  retrieval location/UI that isn't real. Ships in the same PR so the agent
  stops describing a panel that doesn't exist. (Agent-side → needs an image
  rebuild to take effect on a box; the dashboard affordance does not.)

### Phase 2 — Auto-cards (deferred)

Deliverables appear as download cards **automatically** the moment they're
produced, without the user opening the panel. Route an artifact event into
`chatHistories[agentId]` (the `operator_action_receipt` handler at
`app.js:4646` is the precedent pattern). Open question first: the existing
`task_artifact_added` event carries an opaque task `ref` (`orchestration.py`
:1828) and requires a task context — confirm/establish a path from `ref` to a
downloadable file path before wiring it, or emit a new file-scoped event when
an agent writes to `artifacts/`.

### Horizon — send to a destination

For many users the file is a means, not the end (the map user was trying to
get leads into a Google Sheet). A one-click **"Send to Google Sheets / Drive /
email"** on the deliverable card is the deepest UX win; it rides the
in-progress OAuth integrations connect flow. Download is the universal
fallback that always works; "send to destination" is the evolution. Phase 1
cards are built with room for a "Send to…" action to drop in later.

## Non-goals / deliberately deferred

- An explicit `share_file` agent tool — auto-surfacing all of `artifacts/`
  lets the *user* pick, which is safer for "I don't know what I'm looking
  for" than trusting the model to hand over the right one file. Add only if
  auto-surface proves noisy.
- Prose-detection chips (linkify filenames the agent types) — heuristic, so
  "magic that fails silently." Optional polish, never load-bearing.
