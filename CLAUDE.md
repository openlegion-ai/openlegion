# CLAUDE.md — engine

OpenLegion Engine is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. Users interact via CLI REPL, messaging channels (Telegram/Discord/Slack/WhatsApp/Webhook), or a web dashboard.

Deeper docs: `docs/README.md` (architecture, security, configuration, mcp, memory, development). Active design docs live in `docs/plans/` (dated) — check there before re-deriving a design that may already be decided.

## Commands & Testing

```bash
make test                        # fast suite: pytest minus e2e files (same test set as CI)
make lint                        # ruff check + ruff format
pytest tests/test_loop.py -x -v  # single file
```

- ALWAYS run tests in a subagent — keeps test output out of the main context window.
- Mock LLM responses, not the loop — see `tests/test_loop.py:_make_loop()`. Use `AsyncMock` for async methods; SQLite in-memory or `tmp_path` for DB paths.
- E2E tests (`tests/test_e2e*.py`) skip gracefully without Docker + an API key.
- Run `make test` even for docs-only changes — some tests assert CLAUDE.md content (e.g. `tests/test_dashboard_ui.py` pins module rows and constraint names).
- Ruff: py310 target, line length 120 (config in `pyproject.toml`). CI: GitHub Actions — ruff + pytest on 3.11/3.12, sharded 3 ways with `--dist=loadfile`.

## Git Workflow

- **MANDATORY: Use worktrees for ALL code changes.** Every subagent that touches code MUST use `isolation: "worktree"`. Concurrent agents without worktrees overwrite each other.
- **Never `pip install` from a worktree.** Hijacks the global `openlegion` entry point.
- **Never merge directly to main.** Always `gh pr create`, merge through GitHub.
- **Wait for CI before merging.** `gh pr checks <number> --watch`.
- **Branch naming:** `feat/`, `fix/`, `refactor/`, `docs/`, etc.
- **Commit style:** descriptive subject, body explains "why". No Co-Authored-By trailers.

## Work Patterns

- Dispatch concurrent subagents for independent work rather than working sequentially.
- Use Explore subagents for broad codebase research; reserve direct Glob/Grep for targeted, known-location searches.

## Architecture

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8500) — per-agent Camoufox + Xvnc/Openbox stacks
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). All cross-zone communication is HTTP + JSON with Pydantic contracts from `src/shared/types.py`.

Stack: Python 3.10+, FastAPI, asyncio. Docker isolation (`Dockerfile.agent` / `Dockerfile.browser`, bridge network `openlegion_agents`). Dashboard is an Alpine.js SPA — no React, no build step. Deps in `pyproject.toml` (`>=` bounds, no lock file).

### Entry Points

| Entry Point | File | What It Starts |
|---|---|---|
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands (start/stop/chat/status/tasks/wallet/… — see `main.py` for the full set) |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | FastAPI on :8500 (per-agent X stack spawned lazily) |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Quirks

Only modules with non-obvious behavior are listed. The rest are visible by reading the tree.

| Path | Quirk |
|---|---|
| `src/shared/types.py` | THE cross-component contract (Pydantic). `HARD_EDIT_FIELDS` / `SOFT_EDIT_FIELDS` drive the undo-receipt TTL split — all edits apply immediately, TTL is just the Undo window. `DashboardEvent.type` enumerates WebSocket event names. |
| `src/shared/utils.py`, `redaction.py` | `sanitize_for_prompt()`, `setup_logging()`; central credential/URL redactor (`redact_url`, `deep_redact`). |
| `src/agent/loop.py` | Agent execution loop (task + chat). Auto-closes handed-off tasks when `x-task-id` rides the wake chain. |
| `src/agent/server.py` | `_FILE_CAPS` enforced on workspace writes; `_WORKSPACE_ALLOWLIST` gates reads/writes. |
| `src/agent/context.py` | Context window management (write-then-compact); empty summary falls back to hard prune. |
| `src/agent/workspace.py` | Persistent markdown workspace. Bootstrap files: TEAM.md, SOUL/USER/INSTRUCTIONS/MEMORY/INTERFACE/HEARTBEAT. |
| `src/agent/mesh_client.py` | `wake_agent` / `create_task` accept optional `origin: MessageOrigin` and merge it into headers (Constraint #4). |
| `src/agent/builtins/coordination_tool.py` | `hand_off` propagates origin and forwards `task_id` so the recipient's loop auto-closes. `check_inbox` surfaces back-edge `events[]` from the Team Threads store (`mesh_client.list_inbox_events` → `GET /mesh/agents/{id}/task-events`). Failure envelopes: Constraint #10. The optional `data` payload commits to the SENDER's Team Drive (`handoffs/{sender}/{id}.json`, `artifact_ref` = `drive://{team}/{path}@{sha}`) — solo/teamless senders fold it inline into the brief (6k cap); the old blackboard `output/*` write is GONE (Phase-2 unit 4). The post-commit failure envelope is `drive_write_failed` (was `output_write_failed`; the loop's `_HANDOFF_FAILURE_FLAGS` tracks the rename). |
| `src/agent/builtins/file_tool.py` | Two-stage path-traversal protection (`lstat()` for symlink safety). |
| `src/agent/builtins/http_tool.py` | CRED handles, SSRF protection, cross-origin auth header stripping. |
| `src/agent/builtins/tool_authoring.py` | Self-authoring with AST validation (`_FORBIDDEN_IMPORTS` / `_FORBIDDEN_CALLS` / `_FORBIDDEN_ATTRS`) — hygiene, not a sandbox (see Deliberate Tradeoffs). |
| `src/agent/builtins/fleet_tool.py` | Operator-only `list_templates` / `apply_template`. Validates upfront; create loop is per-slot, not atomic (Constraint #3). |
| `src/agent/builtins/operator_tools.py` | Operator-only orchestration. `edit_agent` is the unified edit surface (immediate apply + undo receipt); `read_agent_config` is its inverse. `_OPERATOR_PERMISSION_CEILING` blocks `can_use_wallet` (the sole operator-ungrantable flag; `can_spawn` is now a default-on capability). |
| `src/host/server.py` | Mesh FastAPI app — every endpoint permission-checked. `_RATE_LIMITS` enforces per-category limits. `_require_operator_or_internal` sits between standard agent auth and loopback-only `x-mesh-internal`. `pending_actions` is delete-confirmations only. |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Container hardening enforced here. MCP servers come from the wired `ConnectorStore` at every start — `start_agent` has no `mcp_servers` param. |
| `src/host/connectors.py` | `ConnectorStore` (`config/connectors.json`) — THE single source of MCP servers: the `Connector` union (`MCPConnector` stdio variant \| `HttpConnector` remote variant, discriminated on `transport`) + `agents: ["*"] \| [ids]`; an agent-specific server is just a connector assigned to one agent. `snapshot_for_agent` serializes **stdio records only** into `MCP_SERVERS` — an `HttpConnector` (incl. its `auth`) must never enter a container (pinned by test); remote calls go through the mesh-side `MCPGateway` (`src/host/mcp_gateway.py` — per-call streamable-HTTP sessions, vault-resolved auth, resolved-IP SSRF blocklist + no redirects, 256 KiB result cap, sampling/elicitation rejected; `/mesh/connectors/tools\|call` are assignment-gated for EVERY agent incl. the operator — NOT in the trust-tier bypass, pinned by HTTP-level test). OAuth connect for remote connectors: `/dashboard/integrations/mcp/{name}/connect\|callback` — discovery (RFC 9728→8414) + DCR (RFC 7591, public-client fallback, NO BYO client-id) via `src/host/mcp_oauth.py` (same SSRF posture: discovered URLs are server-controlled); the connection blob embeds `token_endpoint`/`client_id`(/`client_secret`) so `ensure_connection_token` refreshes WITHOUT a registry entry (blob wins over registry; public clients omit `client_secret`; no-refresh-token connections accepted — expiry → probe `needs_auth` → reconnect). Edits apply on restart only — EXCEPT http auth KEY ROTATION (same kind: resolves per call on the mesh, no restart; switching auth MODE or the first Connect bind marks assigned agents dirty — tools register at agent boot; no per-agent cred gate on `auth.cred` — the token is mesh-held). The dashboard Connectors page (sub-tab ID `integrations` is frozen; label is "Connectors") returns `affected_agents` and prompts; it never auto-restarts. Mtime auto-reload picks up hand-edited files. Pending-restart is a generation derivation (immune to the edit-during-container-build race). A missing/denied `$CRED` drops that one connector with an error log — never blocks agent boot. Connector writes are human/dashboard-only. |
| `src/host/credentials.py` | Two-tier vault (SYSTEM_*/CRED_*) + LLM API proxy + OAuth. `is_model_compatible` is the single source of truth for model-allowlist checks (OAuth model subsets are env-overridable — read the code, not stale lists). B2 spend split (Phase-3 unit 1): an LLM call whose REQUESTED model matches the deployment `llm.utility_model` (prefix-insensitive, mesh-config-keyed via `set_utility_model_provider` — never container headers) is *coordination* — tracked `kind='coordination'`, always billed to the caller (skips the ask-window redirect), exempt from the work preflight + team envelope, gated only by `limits.coordination_daily_cap_usd` (`OPENLEGION_COORDINATION_DAILY_CAP_USD`, default 2.0, 0 = tier blocked); enforcement reads in `costs.py` filter `kind='work'`, reporting stays spend-inclusive. |
| `src/host/permissions.py` | Per-agent ACL (glob patterns, deny-all default). `browser_actions: list[str] \| None` narrows per-action surface (`None` = all allowed, `[]` = deny all). |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter (cross-team block via injected `team_resolver`; routed traffic recorded to the injected `thread_store` as `dm` thread rows — the old in-memory `message_log` deque is GONE), `audit_log` with `undoable` + `archived`. |
| `src/host/threads.py` | `ThreadStore` — durable Team Threads (SQLite `data/threads.db`, env override `OPENLEGION_THREADS_DB`, canonical v1). Kinds: `channel` (one per team, created at team create + boot backfill, `teams.thread_ref` points at it; delete_team ARCHIVES, never deletes), `task` (lazy, back-edge events land here as `kind='event'` rows with `recipient`), `dm` (per sorted agent pair). `scope_id` = effective team scope (solo = agent id). REPLACED both the router `message_log` deque and the blackboard `inbox/{agent}/task_event/` feed (C.3-a): the old TTL split is now `list_events_for` query windows (7d actionable / 24h informational); events reaped after 90d, plain messages durable. Read surfaces: `GET /mesh/agents/{id}/task-events` (self-or-operator-or-internal → `check_inbox`), dashboard `/api/threads*` + store-backed `/api/messages`. `post_message` emits `thread_message`. Writers are host-side only (router, back-edge, team create) — no agent-posting endpoint. |
| `src/host/teams.py` | `TeamStore` — THE team authority (SQLite `data/teams.db`, env override `OPENLEGION_TEAMS_DB`): identity, metadata, goals, budget columns, membership (one team per agent, enforced by the `team_members.agent_id` PK — `add_member` evicts the old team and returns it), per-agent standing goals. Owns the `config/teams/{id}/` scaffold (`team.md` + `workflows/`); `metadata.yaml` is GONE — zero readers/writers. `team.md` stays a plain bind-mounted file. Membership changes rewire blackboard ACLs at the ENDPOINT layer (`_add/_remove_team_blackboard_permissions` in `cli/config.py`), not in the store. **Solo agent = team-of-one** (ratified 2026-07, plan §8 #5): a teamless worker's effective scope is its own id — `TEAM_NAME` falls back to the agent id, its ACL always holds the self pattern `teams/{agent_id}/*` ("self always; team while member" — join keeps it, leave restores it, boot backfills empty-ACL solos), and the pubsub prefix gate locks solo workers to their own prefix. Reachability rules (messaging, `_caller_teams`, cross-team block, listings) still use REAL membership; only the operator runs unscoped. Team and agent names share ONE namespace — every create path rejects collisions in both directions. Pure-DB mode (`teams_dir=None` / `:memory:`) for tests and the browser-side tenant lookup. |
| `src/host/drive.py` | Team Drive (Phase-2 unit 1): one bare git repo per team under `data/team_drives/{id}.git` (env `OPENLEGION_TEAM_DRIVES_DIR`), served via smart HTTP on `/mesh/teams/{id}/drive/*` — NO shared volume anywhere (ratified §8 #1). Access wall is REAL team membership (solo team-of-one gets a directive 403; operator/internal pass). `refs/heads/main` is integrate-only: the pre-receive hook rejects it unless the mesh sets `OL_DRIVE_PRIVILEGED=1` (operator-tier callers + the review-merge path ONLY). Reviews live in `teams.db:drive_reviews` (same-branch resubmit supersedes). Merge is claim-first + atomic: the row transitions `open→merging` (BEGIN IMMEDIATE) BEFORE any git side effect (a lost claim 409s and runs no git — no double-merge, no stray empty commit); the reviewed tip is pinned as `head_sha` at submit and re-verified against the live branch at merge (post-approval advance or deleted branch → 409 resubmit, main untouched); the EXACT pinned commit is merged via mesh-side `merge-tree --write-tree -z` (git ≥ 2.38; `--name-only` avoided to keep the floor at 2.38, NUL-split isolates conflicted filenames) + `update-ref --stdin` CAS — conflicts and lost CAS both 409, any git failure reverts `merging→open`. `reject` acts on `open` only (a concurrent reject can't flip a merging row). All mesh git subprocesses are hermetic (`GIT_CONFIG_NOSYSTEM` + `GIT_CONFIG_GLOBAL=/dev/null`). Quota (`drive_quota_mb`) is pre-checked under a per-repo `asyncio.Lock` (on `app.state`) serializing check→push→invalidate so a concurrent-push overshoot is bounded to ONE push; a category `Semaphore` (also `app.state`, cap 8) bounds concurrent packs held in mesh RAM (streaming refactor deferred). Feature branches carry `receive.denyDeletes`/`denyNonFastForwards` (anti-griefing; tags deliberately unrestricted); receive-pack has a per-route body-cap carve-out and gzip bodies inflate against the DECOMPRESSED cap. Lifecycle is TeamStore's (`set_drive_provisioner` + boot backfill; create wipes stale dirs, backfill adopts); storage is `RuntimeBackend.ensure/remove_team_volume` — concrete on the ABC, shared by both backends. Agent side: `drive_tool.py` `team_drive` (clone→`/data/drive`; `sync` never pushes main; token rides per-invocation `http.extraHeader`, never `.git/config`). Phase-2 unit 4 adds a DIRECT-COMMIT path for deliverable registration (NOT reviewed source): `POST /drive/artifacts` (`commit_file` = read-tree/update-index/commit-tree/CAS-update-ref straight to main under `{handoffs\|artifacts}/{sender}/{name}`, author=sender, bypasses the pre-receive hook because it never pushes; member-or-operator, `drive_artifact_max_mb` cap 8, quota-guarded) + `GET /drive/file?path=&ref=` (`read_file` = `cat-file`; content raw, caller sanitizes). Both back `hand_off` data payloads + `save_artifact` registration. |
| `src/host/lanes.py` | `LaneManager` per-agent FIFO task queues (followup/steer). `QueuedTask.task_id` threads through `_direct_dispatch` to make auto-close work (Constraint #6). `try_steer` = injection probe without followup fallback; `QueuedTask.on_start` fires at dispatch (not enqueue) — both exist for the ask verb. |
| `src/host/asks.py` | `AskBroker` — in-memory mesh-held registry for `ask_teammate` inline Q&A (restart = failure envelope at the asker; by design — the Q&A also posts to the thread store when wired via `set_thread_store`, one-line unit-2 integration). Busy recipient → steer-inject (NO task_id — no auto-close, Constraint #6; never a parallel turn, B1); idle → followup lane turn whose own response is the answer fallback. Billing is MESH-AUTHORITATIVE: `credentials.set_bill_resolver` makes the recipient's LLM preflight + usage rows target the ASKER while the window is open (opened by the lane `on_start`, closed on resolution +5s grace or at the `limits.ask_bill_cap_usd` cap; busy-path interjections are never re-billed). `answer_ask` is single-use, verified-recipient-only. Worker→operator asks are 403 (Task-2e posture). |
| `src/host/chain_watcher.py` | Delegate-and-subscribe machinery: durable chain watcher, stall watchdog, opt-in milestone pings. Delivers only to the chain-root first-party human origin. |
| `src/host/cron.py` | Persistent cron scheduler. `_UPDATABLE_FIELDS` frozenset allowlist. Bootstraps per-team daily summary jobs (`ensure_summary_job`). Heartbeats are a **plate-gated agenda dispatch** (Phase-3 unit 2): deterministic probes stay the cheap pre-check; actionable items (probe alerts / pending tasks / recent activity / custom HEARTBEAT.md rules) always escalate, a goals-only plate escalates only when `llm.utility_model` is configured (`utility_model_fn`/`goals_fn` read mesh-side), and a truly-empty plate never reaches the LLM. The old `force_llm`/HEARTBEAT_OK suppression path is GONE. |
| `src/host/summaries.py` | `WorkSummariesStore` — one row per `(scope_kind, scope_id, period_start)`, `scope_kind ∈ {team, solo}`. Ratings lock after a 24h edit window. `feedback_push.py` closes the rating→learning loop. |
| `src/browser/service.py` | Per-agent Camoufox + X11 WID tracking. Fingerprint burn detection; operator clears burn manually. `display_allocator.py` owns the display/VNC-port ranges. |
| `src/browser/captcha.py` | 2captcha + capsolver. Behavioral kinds rejected via `request_captcha_help` handoff. Millicent cost accounting, per-agent + per-tenant caps, kill switch, circuit breaker. |
| `src/browser/flags.py` | Centralized flag loader. `_ENV_ONLY_FLAGS` strips solver creds from `config/settings.json` at load (env-only by design). |
| `src/browser/stealth.py` | `BROWSER_DEVICE_PROFILE` rewrites UA, but the underlying Camoufox engine is unchanged — server-side TLS/JA3 may still read desktop. |
| `src/browser/server.py` | Raises on startup if `MESH_AUTH_TOKEN` set but `BROWSER_AUTH_TOKEN` missing. Session persistence is opt-in via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default false). |
| `src/dashboard/server.py` | Alpine.js SPA, Jinja autoescape, CSP, CSRF via `X-Requested-With`, VNC URL injection. Four top-nav tabs with frozen IDs (Constraint #5). The notifications bell was REMOVED (2026-06-11 chat-native delivery): chain outcomes land in the operator chat as `notification`-role transcript rows via agent `POST /chat/note` (the ChainWatcher's deliver-then-claim durability point — the ack must never lie) + live `notification` events; formerly bell-only signals (`connection_refresh_failed`, quarantine) reroute via `runtime._system_signal_producer`; desktop pings fan in from the underlying WS events in `onWsEvent`. |
| `src/dashboard/telemetry.py` | Telemetry sink (`dashboard_telemetry` table) with retention cap + per-session rate limit. |
| `src/channels/whatsapp.py` | WhatsApp Cloud API (`X-Hub-Signature-256` verification, warns when disabled). |
| `src/templates/` | YAML fleet templates (starter, devteam, deep-research, monitor, sales, …). |

## Cross-Repo Integration

Engine is standalone — NO imports, calls, or shared code with app/ or provisioner/. Integration happens externally:

- **Provisioner → Engine**: manages instances via Docker/systemd on Hetzner VPS. Deploys via `git clone` cloud-init; live updates run git pull + Docker rebuild + `systemctl restart openlegion` over SSH (`provisioner/app/services/ssh.py:run_update()`). Writes `.env` via SSH (base64-encoded). Health-checks `GET /mesh/agents` with `x-mesh-internal: 1` from localhost.
- **App → Engine (SSO)**: app generates `HMAC-SHA256(access_token, "{subdomain}:{expiry}")`, redirects to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`; the auth gate (deployed via cloud-init, not engine code) verifies and sets a one-time `ol_session` cookie; Caddy `forward_auth` checks it on every request.
- **Exposed surfaces**: `/mesh/agents` (health), `/__auth/callback` (SSO), dashboard on :8420, `/agent-vnc/{agent_id}/{path}` (per-agent KasmVNC proxy), `/ws/events` (dashboard WebSocket).

## Patterns & Conventions

- `@tool` decorator for agent capabilities. `setup_logging("component.module")` for loggers.
- Pass ALL untrusted text through `sanitize_for_prompt()` at input boundaries before it reaches the LLM.
- Async by default (FastAPI + asyncio); wrap blocking calls in `run_in_executor`.
- All state is SQLite with WAL mode — blackboard, memory, costs, cron, traces. No Redis, no external databases.

### Config & Environment

- `.env` loaded via python-dotenv at CLI startup.
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` — LLM provider keys (mesh-only). `create_agent` / `apply_template` reject HTTP 400 if the chosen model's provider has no credentials configured.
- `OPENLEGION_CRED_<NAME>` — agent-tier credentials (distinct from SYSTEM tier).
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_TEAMS` — plan limits.
- `OPENLEGION_BROWSER_MAX_CONCURRENT` (legacy alias `MAX_BROWSERS`) — per-service Camoufox cap. Startup-only; restart browser service to apply.

## Security Boundaries

Security-first posture with deliberate UX tradeoffs (next section). Before changing anything security-shaped, read `docs/security-audit-2026-05-29.md` and `docs/security-remediation-review-2026-05-29.md`.

- **Agents never hold API keys.** All LLM/API calls go through the mesh credential vault and proxy (`src/agent/llm.py` routes through the mesh).
- **Permission checks default-deny** on every mesh endpoint; rate limits on state-mutating endpoints.
- **Container hardening is the real agent sandbox**: non-root UID 1000, `cap_drop=[ALL]`, `no-new-privileges`, `read_only=True`, `tmpfs=/tmp`, memory/CPU/pids limits — enforced in `runtime.py` (operator container gets tighter limits than workers; browser container is similar but with a writable `/home/browser`).
- **SSRF**: agent HTTP = `http_tool` DNS pin + IP blocklist (RFC1918, loopback, CGNAT, 6to4, Teredo, IPv4-mapped IPv6). Browser traffic = the container iptables egress filter (`docker/browser-entrypoint.sh`, operator override via `BROWSER_EGRESS_ALLOWLIST`) is the AUTHORITATIVE anti-rebinding layer; the mesh-side `_resolve_and_pin()` check covers only `navigate`/`open_tab` and is best-effort (M20).
- **Model allowlist at config-write time**: `create_agent`, `edit_agent` (model field), and `apply_template` (every model field, including per-slot template defaults) reject HTTP 400 when `credentials.py:is_model_compatible` fails — the same gate the LLM proxy runs at call time. Mirrored on the CLI path (`cli/config.py:_create_agent_from_template`).
- **Failure reasons must surface**: `tasks.blocker_note` carries the status reason for `blocked` (recoverable) and `failed` (terminal); `cancelled` deliberately stays None; `done` clears it. The mesh status endpoint promotes `body["error"]` to `blocker_note` on failed transitions; the dashboard renders it as a banner; `workflow_snapshot` exposes it per stage.
- **Chain-break observability**: `chain_breaks_24h_count` on `/mesh/system/metrics` counts done tasks with no successor task and no outcome (silent workflow terminations), paired with the `task_completed_without_handoff` event. Observability only — NO enforcement; operator's own tasks are excluded at the metrics layer.
- **Path traversal**: two-stage check in `file_tool.py`; workspace uses `resolve` + `is_relative_to`. **CSRF**: required `X-Requested-With` header on dashboard state-changing endpoints.

## Deliberate Tradeoffs — do NOT "fix" these

Intentional product decisions that can look like security gaps. Do not revert or "harden" them without an explicit user decision:

- **Internet on by default for new agents.** Every create path persists `can_use_internet: True` via the base defaults in `cli/config._add_agent_permissions` (mirroring how that base flips `can_use_browser` to True). The `types.py` field default stays `False` so the deny-all fallback for an unknown agent stays restrictive — don't flip the field default. Worker internet tools (`http_request`/`web_search`) were already default-ungated; the flag mainly drives the operator's agent-side tool filter + the dashboard badge. The Operator Settings toggle / dashboard permission edit can still set it `False` per-agent.
- **Dashboard permission edits have NO operator ceiling** (`PUT /api/agents/{id}/permissions`). That path is the human's deliberate escalation surface; `_OPERATOR_PERMISSION_CEILING` applies only to the LLM/mesh edit path. Don't add a ceiling or a config toggle to the dashboard path (audit H1).
- **The blackboard is shared by design** (audit H10). Don't naively enforce per-team key prefixes — it breaks the default agent and template coordination. Team scoping goes through team wiring. Solo agents are NOT blocked from the board anymore — they're self-scoped to a private `teams/{agent_id}/` team-of-one namespace (ratified #5, code-path merge only; no shared reachability gained). **The blackboard is signals-only** — `output/*` and `artifacts/*` payload flows moved to the Team Drive (2026-07, Phase 2 unit 4): `hand_off` data commits to the sender's drive and `save_artifact` registers there; the create-default + operator-ceiling `blackboard_write` no longer grant those namespaces, and no template ACL does. Working signal namespaces (`tasks/*`, `status/*`, `context/*`, `signals/*`, `claim_task` CAS, template `reviews/*`/`drafts/*`/…) STAY.
- **AST validation on self-authored tools is hygiene, not a sandbox.** Agents already have `run_command` (in-container code execution by design); the Docker container is the boundary. Marketplace tools (`/app/marketplace_tools`) load WITHOUT AST validation — trusted because the dir is operator-populated and read-only. If a remote/agent-reachable install path is ever added, pin installs to a verified commit SHA (M1, H15).
- **Operator trust-tier carve-out** (Constraint #12). `_caller_is_operator` short-circuits the coordination/management gates (messaging, pubsub, cron, fleet, task routing, spawn, blackboard ops) by design. Still gated like any worker: `can_use_wallet*`, `can_access_credential`, `can_manage_vault`, `can_browser_action`. `test_operator_still_gated_surfaces_not_in_bypass_grep` pins the boundary.

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly; agents coordinate through the blackboard.
2. **Tools over features.** New agent capabilities go in as `@tool`-decorated functions, not loop changes.
3. **`apply_template` is per-slot, not atomic.** Mesh validates upfront, but a mid-loop failure leaves earlier-created agents in place — verify the returned `created` list matches the requested slot set.
4. **Propagate `MessageOrigin`.** New cross-agent paths producing work for another agent must read `current_origin` once and forward it to both `wake_agent` and `create_task` — otherwise the receiving lane worker can't auto-notify the originating channel/user on handoff completion.
5. **Tab IDs are frozen for URL stability.** The four top-nav tab IDs (`chat`, `fleet`, `workplace`, `system`) appear in URL paths, JS state, and dashboard endpoints. Labels may diverge (`fleet`→"Teams", `workplace`→"Work", `system`→"Settings") but never rename the IDs.
6. **Auto-close requires task_id plumbing.** Handed-off tasks auto-transition to terminal status only when the wake chain carries `x-task-id`. Wakes without one (heartbeats, manual chats) won't auto-close — intentional. Back-edge events land on the task's thread (`ThreadStore`, 7d actionable / 24h informational serving windows), surfaced via `check_inbox`.
7. **LLM tool-calling roles must alternate** (`user → assistant(tool_calls) → tool(result) → assistant`). `_trim_context` merges the summary into the first user message to preserve this invariant.
8. **Avoid new module-level globals.** Existing exceptions: `_tool_staging` in `tools.py` (lock-protected), `_client` in `http_tool.py` (connection pool).
9. **Project→team rename is COMPLETE (2026-07).** One name, one namespace: `can_manage_teams`, `tasks.team_id`, blackboard `teams/{name}/` prefix, `target_kind="team"`, `TEAM_NAME`/`TEAM_MD_PATH` envs, `config/teams/` + `team.md`. The startup migrators (`team_migration.py`, `orchestration_migration.py`) and every lazy `ADD COLUMN` chain were deleted — schemas are canonical v1 (`PRAGMA user_version = 1`); do not re-introduce `project_*` aliases or migration shims.
10. **Coordination-tool failure envelopes.** Errors from `hand_off` / `update_status` / `complete_task` wrap via `_failed_transition_envelope` so the LLM sees `handed_off=False` + a directive `error` + `recovery_hint`; sentinel keys merge AFTER caller `extras`. Without this shape, LLMs silently report success when handoffs fail post-commit.
11. **Wizard state machine** `idle | ask | confirming | building | first-output | build_failed`, persisted to `localStorage.ol_wizard`; resets to `idle` on unknown values; mounts only when `step !== 'idle'`; mutually exclusive with the "What's new" tour.
12. **Operator trust-tier carve-out** — see "Deliberate Tradeoffs" above for the bypassed vs. still-gated surfaces (code comments reference this as Constraint #12, e.g. `mesh_tool.py`). Boot fail-closed: empty `auth_tokens` under `enforce` mode → `SystemExit` (X-Agent-ID would be forgeable).
