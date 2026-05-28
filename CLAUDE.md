# CLAUDE.md — engine

## Overview

OpenLegion Engine is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. Users interact via CLI REPL, messaging channels (Telegram/Discord/Slack/WhatsApp/Webhook), or a web dashboard.

## Architecture

### System Layout

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8500) — per-agent Camoufox + per-agent Xvnc/Openbox/unclutter on displays :100..:163 paired with KasmVNC ports 6100..6163
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). All cross-zone communication is HTTP + JSON with Pydantic contracts from `src/shared/types.py`.

### Entry Points

| Entry Point | File | What It Starts |
|---|---|---|
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands: start, stop, status, chat, wallet, version |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | FastAPI on :8500 (per-agent X stack spawned lazily) |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Responsibilities

Only modules with non-obvious quirks worth flagging are listed. The rest are visible by reading the tree.

| Path | Responsibility |
|---|---|
| `src/shared/types.py` | THE cross-component contract (Pydantic). `HARD_EDIT_FIELDS` / `SOFT_EDIT_FIELDS` drive the undo-receipt TTL split (5 min soft / 30 min hard); all edits apply immediately, TTL is just the Undo window. `DashboardEvent.type` enumerates WebSocket event names. |
| `src/shared/utils.py` | `sanitize_for_prompt()`, `setup_logging()` |
| `src/shared/redaction.py` | Central credential/URL redactor (`redact_url`, `deep_redact`) |
| `src/agent/loop.py` | Agent execution loop (task + chat). Auto-closes handed-off tasks when `x-task-id` rides the wake chain. |
| `src/agent/server.py` | Agent FastAPI server. `_FILE_CAPS` enforced on workspace writes; `_WORKSPACE_ALLOWLIST` gates reads/writes. |
| `src/agent/llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `src/agent/context.py` | Context window management (write-then-compact); empty summary falls back to hard prune |
| `src/agent/workspace.py` | Persistent markdown workspace. Bootstrap files include TEAM.md (legacy `PROJECT.md` resolves read-only), SOUL/USER/INSTRUCTIONS/MEMORY/INTERFACE/HEARTBEAT. |
| `src/agent/mesh_client.py` | Agent-side HTTP client. `wake_agent` / `create_task` accept optional `origin: MessageOrigin` and merge `origin_header(origin)` into headers. |
| `src/agent/builtins/coordination_tool.py` | `hand_off` / `check_inbox` / `update_status` / `complete_task`. `hand_off` propagates `MessageOrigin` and forwards `task_id` so recipient's loop auto-closes. `check_inbox` surfaces back-edge `events[]` from `inbox/{agent}/task_event/`. |
| `src/agent/builtins/file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `src/agent/builtins/http_tool.py` | HTTP with CRED handles, SSRF protection, cross-origin auth header stripping |
| `src/agent/builtins/skill_tool.py` | Self-authoring with AST validation (`_FORBIDDEN_IMPORTS` / `_FORBIDDEN_CALLS` / `_FORBIDDEN_ATTRS`) |
| `src/agent/builtins/fleet_tool.py` | Operator-only `list_templates` / `apply_template`. Validates upfront; create loop is per-slot, not atomic (Constraint #3). |
| `src/agent/builtins/operator_tools.py` | Operator-only orchestration. `edit_agent` is the unified edit surface (all fields apply immediately + undo receipt). `read_agent_config` is its inverse. `_OPERATOR_PERMISSION_CEILING` blocks `can_spawn` / `can_use_wallet`. |
| `src/host/server.py` | Mesh FastAPI app — 109 endpoints, all permission-checked. `_RATE_LIMITS` enforces per-category limits. `_require_operator_or_internal` is a permission tier between standard agent auth and loopback-only `x-mesh-internal`. `pending_actions` is now delete-confirmations only. |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Container hardening enforced here. |
| `src/host/credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy. OpenAI OAuth support. |
| `src/host/permissions.py` | Per-agent ACL (glob patterns, deny-all default). `browser_actions: list[str] \| None` narrows per-action surface (`None` = all allowed, `[]` = deny all). |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter, `audit_log` with `undoable` + `archived` |
| `src/host/lanes.py` | `LaneManager` per-agent FIFO task queues (followup/steer modes). `QueuedTask.task_id` field threads through `_direct_dispatch` to make auto-close work (Constraint #6). |
| `src/host/cron.py` | Persistent cron scheduler. `_UPDATABLE_FIELDS` frozenset allowlist. `CronScheduler.ensure_summary_job` / `find_summary_job` bootstrap per-team daily summary jobs at `DEFAULT_SUMMARY_SCHEDULE = "0 9 * * *"`. |
| `src/host/summaries.py` | `WorkSummariesStore` (SQLite WAL, 30-day reap). One row per `(scope_kind, scope_id, period_start)` — `scope_kind ∈ {team, solo}`. Ratings (`accepted`/`acknowledged`/`rework`) lock after `RATING_EDIT_WINDOW_SECONDS=86400`. |
| `src/browser/service.py` | BrowserManager with per-agent Camoufox + X11 WID tracking. Fingerprint health monitor with burn detection; operator clears burn manually. |
| `src/browser/captcha.py` | 2captcha + capsolver. Behavioral kinds rejected via `request_captcha_help` handoff. Millicent cost accounting, per-agent + per-tenant caps, kill switch, circuit breaker. |
| `src/browser/flags.py` | Centralized flag loader. `_ENV_ONLY_FLAGS` strips solver creds from `config/settings.json` at load (env-only by design). |
| `src/browser/stealth.py` | Anti-bot fingerprint. `BROWSER_DEVICE_PROFILE` rewrites UA, but underlying Camoufox engine unchanged — server-side TLS/JA3 may still read desktop. |
| `src/browser/server.py` | Raises on startup if `MESH_AUTH_TOKEN` set but `BROWSER_AUTH_TOKEN` missing |
| `src/browser/session_persistence.py` | Session continuity; opt-in via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default false) |
| `src/dashboard/server.py` | FastAPI router + ~141 endpoints + VNC URL injection. Alpine.js SPA, Jinja autoescape, CSP, CSRF via `X-Requested-With`. Four top-nav tabs (`chat`/`workplace`/`fleet`/`system`) — IDs frozen for URL stability (Constraint #5). Work tab lands directly on summary cards (PR 3 of Work-tab rewrite retired `/api/workplace/outputs` + `/api/workplace/feed` and the legacy team-status / team-outputs / activity-feed sub-surfaces). |
| `src/dashboard/notifications.py` | Persistent notifications store (`dashboard_notifications` SQLite, WAL). Frozen `_KNOWN_KINDS = {delivered, approval, alert, info, blocker, credential}` for the top-nav bell. `delivered` is accepted by the store (historical rows still render) but PR 3 of the Work-tab rewrite removed the only producer that minted them on `task_status_changed`→`done` — the bell no longer dings per task completion; operator authors notifications explicitly via `notify_user` for genuinely user-facing events. Producers in `dashboard/server.py:_notifications_producer` are now: `approval` (pending_action_created), `alert` (health_change degradation, credit_exhausted), `credential` (credential_request, browser_login_request). |
| `src/dashboard/telemetry.py` | Telemetry sink (`dashboard_telemetry` table) with `_MAX_EVENTS=100_000` retention cap and per-session rate limit. |
| `src/channels/whatsapp.py` | WhatsApp Cloud API (`X-Hub-Signature-256` verification, warns when disabled) |
| `src/templates/` | YAML fleet templates (starter, content, deep-research, devteam, monitor, sales, etc.) |

## Cross-Repo Integration

### Engine is standalone

The engine has NO direct dependencies on app/ or provisioner/. No imports, no calls, no shared code. Integration is handled by provisioner and app externally.

### Provisioner → Engine

Provisioner manages engine instances via Docker/systemd on Hetzner VPS:
- Deploys code via `git clone` in cloud-init. Live update path in `provisioner/app/services/ssh.py:run_update()` runs git pull + Docker rebuild + `systemctl restart openlegion` over SSH.
- Writes `.env` via SSH (base64 encoded to prevent injection).
- Health checks by SSH-ing to localhost and hitting `GET /mesh/agents` with `x-mesh-internal: 1`.
- Starts/stops via `systemctl restart openlegion`.

### App → Engine (SSO)

1. App generates HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. Redirects user to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. Auth gate (deployed via cloud-init) verifies HMAC, sets `ol_session` cookie (one-time-use, replay protection)
4. Caddy reverse proxy uses `forward_auth` to check cookie on every request

### Exposed Endpoints

- `/mesh/agents` — health check target (provisioner hits via SSH + localhost)
- `/__auth/callback` — SSO callback (handled by auth gate behind Caddy, not engine code)
- Dashboard UI on :8420 — user-facing after SSO
- `/agent-vnc/{agent_id}/{path}` — per-agent reverse proxy to that agent's KasmVNC port
- `/ws/events` — WebSocket for real-time dashboard updates

## Patterns & Conventions

- `@skill` decorator for agent capabilities. `setup_logging("component.module")` for loggers.
- All untrusted text passes through `sanitize_for_prompt()` at input boundaries before reaching the LLM.
- Async by default (FastAPI + asyncio); blocking calls wrapped in `run_in_executor`.

### Config & Environment
- `.env` loaded via python-dotenv at CLI startup.
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` — LLM provider keys (mesh-only). Agent creation validates that the chosen model's provider has credentials configured — `create_agent` / `apply_template` reject with HTTP 400 if e.g. you ask for `openai/gpt-4o-mini` but only the Anthropic key is set.
- `OPENLEGION_CRED_<NAME>` — agent-tier credentials (distinct from SYSTEM tier above).
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_TEAMS` — plan limits.
- `OPENLEGION_BROWSER_MAX_CONCURRENT` (legacy alias `MAX_BROWSERS`) — per-service Camoufox cap. Startup-only; restart browser service to apply.

### State
- All SQLite with WAL mode — blackboard, memory, costs, cron, traces. No Redis, no external databases.

### Security Boundaries

- **Agents never hold API keys.** All LLM/API calls go through the mesh credential vault.
- **AST validation gates skill self-authoring** — no `eval`/`exec`/`open` on untrusted input.
- **Permission checks default-deny** on every mesh endpoint; rate limits on state-mutating endpoints.
- **SSRF protection** = DNS pin + IP blocklist (RFC1918, loopback, CGNAT, 6to4, Teredo, IPv4-mapped IPv6) + browser-container iptables egress filter (see `docker/browser-entrypoint.sh`; operator override via `BROWSER_EGRESS_ALLOWLIST`).
- **All untrusted text sanitized** via `sanitize_for_prompt()` at input boundaries.
- **Path traversal protection** — two-stage check in `file_tool.py` (`lstat()` for symlinks), workspace uses `resolve` + `is_relative_to`.
- **CSRF** via required `X-Requested-With` header on dashboard state-changing endpoints.
- **Container hardening** — non-root UID 1000, `cap_drop=[ALL]`, `no-new-privileges`, `read_only=True`, `tmpfs=/tmp`, mem 384m, 0.15 CPU, `pids_limit=256`. Browser container is similar but has a writable `/home/browser`.
- **Chain-break observability counter.** `chain_breaks_24h_count` on `/mesh/system/metrics` counts `done` tasks per assignee with no child task referencing them via `parent_task_id` and no outcome set, within the trailing 24h window. Surfaces silent workflow terminations (agent finished work without handing off to a successor) to the operator's heartbeat without manual `workflow_snapshot` polling. Paired with the `task_completed_without_handoff` `DashboardEvent` literal emitted by `Tasks.update_status` at the `done` transition. Observability-only — NO enforcement. Operator's own tasks are excluded at the ``/mesh/system/metrics`` layer (matches the existing convention used by ``outcome_rejected_24h_count`` / ``execution_failures_24h_count`` / ``stale_tasks_24h_count``); ``Tasks.chain_breaks_24h`` itself returns the unfiltered per-assignee count.
- **Model allowlist validation at config-write time.** `create_agent` (POST `/mesh/agents/create`), `edit_agent` (PUT `/mesh/agents/{id}/edit-soft` when `field == "model"`), and `apply_template` (POST `/mesh/templates/apply` — top-level `model` override + each `agent_overrides[].model` + each slot's template-default model) reject HTTP 400 when `CredentialVault.is_model_compatible(model)` returns false — same gate the LLM proxy runs at call time. Surfaces OAuth-allowlist rejections (e.g. the OpenAI OAuth subset is `{gpt-5, gpt-5-mini, gpt-5-pro, gpt-5.3-codex}` — see `_OAUTH_ALLOWED_MODELS` in `credentials.py`) BEFORE the bad config persists. Mirrored on the CLI path in `cli/config.py:_create_agent_from_template` so `apply_template` doesn't silently mint dead-on-arrival agents from either side. Single source of truth: `credentials.py:is_model_compatible` returning `(bool, reason_str | None)`.
- **Failure reason surfacing.** `tasks.blocker_note` is the non-success status-reason column — covers `blocked` (recoverable, original semantic) AND `failed` (terminal failure, Bug 3 "silent model rejection"). `cancelled` deliberately stays None (manual user action, no error to surface); `done` clears the column. The mesh endpoint `/mesh/tasks/{task_id}/status` promotes `body["error"]` to `blocker_note` (truncated 500 chars) on failed transitions when no explicit `blocker_note` was passed — covers the agent loop's `set_task_status(task_id, "failed", error=...)` auto-close path. The lane manager passes `blocker_note=` directly on quarantine + timeout transitions. The dashboard task drill-in modal renders the note as a coloured banner (amber for `blocked`, red for `failed`). `workflow_snapshot` surfaces `blocker_note` per stage for operator heartbeat inspection.

## Dependencies & Infrastructure

- **Runtime**: Python 3.10+, FastAPI, asyncio.
- **Isolation**: Docker containers per agent, bridge network (`openlegion_agents`). `Dockerfile.agent` and `Dockerfile.browser` in repo root.
- **Browser**: Shared container running one Camoufox + Xvnc + Openbox + unclutter stack per agent (displays 100..163, KasmVNC ports 6100..6163).
- **Dashboard**: Alpine.js SPA — no React, no build step (Tailwind via CDN).
- **CI**: GitHub Actions — ruff lint + pytest on Python 3.11 and 3.12.
- **Dependencies**: `pyproject.toml` uses minimum version bounds (`>=`), no lock file. See `pyproject.toml` for the dep list.

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard.
2. **Skills over features.** New agent capabilities go in as `@skill`-decorated functions, not loop changes.
3. **`apply_template` is per-slot, not atomic.** Mesh validates upfront, but the create loop is not transactional — a mid-loop failure leaves earlier-created agents in place. Verify the returned `created` list matches the requested slot set.
4. **`MessageOrigin` propagation.** `wake_agent` and `create_task` both accept optional `origin: MessageOrigin`. New cross-agent paths producing work for another agent should read `current_origin` once and forward it to both calls — otherwise the receiving lane worker can't auto-notify the originating channel/user on handoff completion.
5. **Tab IDs are frozen for URL stability.** The four top-nav tab IDs (`chat`, `fleet`, `workplace`, `system`) appear in URL paths, JS state, and dashboard endpoints. Labels diverge (`fleet`→"Teams", `workplace`→"Work", `system`→"Settings") but renaming the IDs would break deep-links and persisted prefs.
6. **Auto-close requires task_id plumbing.** Handed-off tasks auto-transition to terminal status only when the wake chain carries `x-task-id` (via `wake_agent(task_id=...)`, threaded through `LaneManager.QueuedTask.task_id` and the `_direct_dispatch` header). Wake calls without a task_id (heartbeats, manual chats) won't auto-close — intentional. Back-edge events go to `inbox/{agent}/task_event/{id}` with a 7-day TTL; surfaced via `check_inbox`.
7. **LLM tool-calling roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. `_trim_context` merges summary into first user message to preserve this invariant.
8. **Module-level globals.** `_skill_staging` in `skills.py` (threading-lock protected), `_client` in `http_tool.py` (connection pool). Avoid adding more.
9. **Project→team rename completed 2026-05.** Remaining shims: `MeshClient.*_project` proxies + `AgentPermissions.can_manage_projects` validator (internal callers, deferred); `tasks.project_id` dict-key emission alongside `team_id` (pending external-consumer audit). Internal namespaces unchanged: blackboard `projects/{name}/` prefix, `target_kind="project"` on `pending_actions`. `src/host/team_migration.py` startup migrator scheduled for removal next release.
10. **Coordination-tool failure envelopes.** Errors from `hand_off` / `update_status` / `complete_task` wrap exceptions via `_failed_transition_envelope` so the LLM sees `handed_off=False` + a directive `error` ("MUST NOT report success") + `recovery_hint`. Sentinel keys merge AFTER caller `extras` to prevent shadowing. Without this shape, LLMs silently report success when handoffs fail post-commit.
11. **Wizard state machine: `idle | ask | confirming | building | first-output | build_failed`.** Persisted to `localStorage.ol_wizard`; resets to `idle` on unknown values. Wizard mounts only when `step !== 'idle'`. Mutually exclusive with the "What's new" tour (existing-fleet users only, in-memory state).
12. **Operator trust-tier carve-out.** `_caller_is_operator` short-circuits 14 coordination/management gates (`can_message`, `can_publish`, `can_subscribe`, `can_use_api`, `can_manage_cron`, `can_manage_fleet`, `can_route_tasks`, `can_spawn`, blackboard read/write/list/watch/claim/delete). Still gated like any worker: `can_use_wallet*`, `can_access_credential`, `can_manage_vault`, `can_browser_action`. `test_operator_still_gated_surfaces_not_in_bypass_grep` pins the boundary. Boot fail-closed: empty `auth_tokens` under `enforce` mode → `SystemExit` (X-Agent-ID would be forgeable).

## Git Workflow

- **MANDATORY: Use worktrees for ALL code changes.** Every subagent that touches code MUST use `isolation: "worktree"`. Concurrent agents without worktrees overwrite each other.
- **Never `pip install` from a worktree.** Hijacks the global `openlegion` entry point.
- **Never merge directly to main.** Always `gh pr create`, merge through GitHub.
- **Wait for CI before merging.** `gh pr checks <number> --watch`.
- **Branch naming:** `feat/`, `fix/`, `refactor/`, `docs/`, etc.
- **Commit style:** descriptive subject, body explains "why". No Co-Authored-By trailers.

## Work Patterns

- **Use subagents to parallelize independent work.** Dispatch concurrent subagents rather than working sequentially.
- **Always run tests in a subagent.** Keeps test output out of the main context window.
- **Use Explore subagents for broad codebase research.** Reserve direct Glob/Grep for targeted, known-location searches.

## Testing

```bash
# Unit + integration (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Single test file
pytest tests/test_loop.py -x -v
```

- Mock LLM responses, not the loop. See `tests/test_loop.py:_make_loop()`.
- `AsyncMock` for async methods, SQLite in-memory or `tmp_path` for DB paths.
- E2E tests skip gracefully without Docker + API key.

## Review State

2026-05-28 — Work tab rewrite, PR 1 of 3. Operator goals tracking — `manage_goals` operator-only tool + `GOALS.json` (structured, dashboard reads) / `GOALS.md` (rendered for humans) + `GET /api/workplace/goals` dashboard endpoint. Purely additive backend foundation; no UI change. Goals are workplace-wide user-stated business outcomes (distinct from per-team `set_team_goal` north stars). Max 10 goals; sanitization on name/note inputs; per-goal `updated_at` preserved on partial updates so progress_note edits don't bump the timestamp twice. Workspace allowlist gained `GOALS.json` + `GOALS.md`; the operator's HEARTBEAT.md now re-reads `GOALS.json` fresh each cycle (workspace-as-source-of-truth) and flags stale goals (no related task activity in 14 days) in the summary's "What I'm watching" section without auto-retiring them.

2026-05-28 — Work tab rewrite, PR 2 of 3. Visible product cutover. Deleted the team-management-console UI surface from the Work tab and replaced it with summary cards + Goals chip strip + "Tell Operator" composer; the per-task 👍/➖/👎 buttons were dropped from the Recently Delivered cards and replaced by the operator's heartbeat-driven `rate_delivery` tool (which calls `POST /mesh/tasks/{task_id}/outcome` — the same path the human drill-in modal hits — so the rework-spawn + memory-write machine loop is preserved). The dashboard drill-in modal still renders rating buttons. Heartbeat now grades up to 10 oldest unrated done tasks per cycle, defaulting to `acknowledged` (no memory write) when uncertain. `_OPERATOR_HEARTBEAT` gained the `<!-- heartbeat_v3_rate_delivery -->` sentinel so existing operator agents refresh their HEARTBEAT.md on next startup.

2026-05-29 — Work-tab polish + cold-start ownership move. Removes the standalone "Tell operator" textarea (redundant with the Chat tab — steering now flows through inline summary rework feedback OR chat), replaces the 👍 ➖ 👎 emoji on summary cards with three Lucide-style inline SVG icons (accept / acknowledge / rework — same emerald/gray/amber colors, same handlers, plus `data-testid` per-button for test stability), and shifts cold-start goal-seeding ownership to the operator. `_OPERATOR_HEARTBEAT` gains a "Goal seeding (cold start)" section: when `GOALS.json` is empty AND at least one team has active agents, the operator calls `notify_user` ONCE per week asking what business outcomes to track, recording the ask in `save_observations` to avoid re-pestering — guessing from a heartbeat is explicitly disallowed; the primary seeding path is chat-time `manage_goals` when the user states outcomes or after team creation reveals measurable goals. `HEARTBEAT_SENTINELS` adds `heartbeat_v4_goal_seeding` so v3 operators refresh their `HEARTBEAT.md` on next startup. Drops the `compose_work_summary` code-leak from the "No summaries yet" empty state (was naming an internal skill at non-technical users). Cleans 9 stale `kanban`/`activity` comments left over from earlier PRs. Deletes ~45 lines of `submitTellOperator` JS + the `tellOperatorText` / `tellOperatorInflight` / `tellOperatorConfirmation` Alpine state. 11 new test assertions; 6566 passed / 49 skipped / 0 failed on the broad sweep.

2026-05-28 — Work tab rewrite, PR 3 of 3. Backend cleanup + docs + late PR-2 regression fix. Deletes the two endpoints whose only JS consumers PR 2 removed: `GET /api/workplace/outputs` and `GET /api/workplace/feed` (the helper `_summarize_task_event` goes with them). Drops the `kind="delivered"` producer from `_notifications_producer` — the bell no longer dings per worker completion; `delivered` itself stays in `_KNOWN_KINDS` so historical rows render. Sunsets the orphaned Alpine state (`workplaceTab`, `workplaceTabs`) and the two `<template x-if="false && workplaceTab === '...'">` deep-link-back-compat blocks left over from PR 2. Adds one end-to-end test (`tests/test_rate_delivery_e2e.py`) that drives `rate_delivery` → `MeshClient.set_task_outcome` → mesh `/mesh/tasks/{id}/outcome` → `Tasks.create_rework_task` against an in-process ASGI mesh app, closing the Codex r1 nice-to-have gap from PR 2 where the rework-spawn path was only mock-covered. `tests/test_workplace_feed.py` deleted; `test_dashboard_notifications.py` updated to assert no `delivered` row fires; `test_dashboard.py::test_workplace_outputs_returns_done_tasks` removed. `docs/dashboard.md` Work-tab section rewritten (sub-routes / surfaces / section buckets all stale post-PR-2) and `/api/workplace/projects` endpoint reference corrected to the post-rename `/api/workplace/teams`. Also lands a stranded PR-2 fix: `_HEARTBEAT_TOOLS` in `src/agent/loop.py` was supposed to gain `rate_delivery` + `manage_goals` during PR 2's Codex-driven follow-up commit but the edit never persisted to disk; without them the heartbeat instructions told operator to call tools the loop denies. The PR 3 second-pass Codex review caught the still-broken contract and the fix lands here.
