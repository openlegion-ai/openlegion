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
| `src/agent/workspace.py` | Persistent markdown workspace. Bootstrap files: TEAM.md (a legacy `PROJECT.md` is renamed to TEAM.md at startup by `team_migration.py`), SOUL/USER/INSTRUCTIONS/MEMORY/INTERFACE/HEARTBEAT. |
| `src/agent/mesh_client.py` | `wake_agent` / `create_task` accept optional `origin: MessageOrigin` and merge it into headers (Constraint #4). |
| `src/agent/builtins/coordination_tool.py` | `hand_off` propagates origin and forwards `task_id` so the recipient's loop auto-closes. `check_inbox` surfaces back-edge `events[]` from `inbox/{agent}/task_event/`. Failure envelopes: Constraint #10. |
| `src/agent/builtins/file_tool.py` | Two-stage path-traversal protection (`lstat()` for symlink safety). |
| `src/agent/builtins/http_tool.py` | CRED handles, SSRF protection, cross-origin auth header stripping. |
| `src/agent/builtins/tool_authoring.py` | Self-authoring with AST validation (`_FORBIDDEN_IMPORTS` / `_FORBIDDEN_CALLS` / `_FORBIDDEN_ATTRS`) — hygiene, not a sandbox (see Deliberate Tradeoffs). |
| `src/agent/builtins/fleet_tool.py` | Operator-only `list_templates` / `apply_template`. Validates upfront; create loop is per-slot, not atomic (Constraint #3). |
| `src/agent/builtins/operator_tools.py` | Operator-only orchestration. `edit_agent` is the unified edit surface (immediate apply + undo receipt); `read_agent_config` is its inverse. `_OPERATOR_PERMISSION_CEILING` blocks `can_spawn` / `can_use_wallet`. |
| `src/host/server.py` | Mesh FastAPI app — every endpoint permission-checked. `_RATE_LIMITS` enforces per-category limits. `_require_operator_or_internal` sits between standard agent auth and loopback-only `x-mesh-internal`. `pending_actions` is delete-confirmations only. |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Container hardening enforced here. MCP servers come from the wired `ConnectorStore` at every start — `start_agent` has no `mcp_servers` param. |
| `src/host/connectors.py` | `ConnectorStore` (`config/connectors.json`) — THE single source of MCP servers: the `Connector` union (`MCPConnector` stdio variant \| `HttpConnector` remote variant, discriminated on `transport`) + `agents: ["*"] \| [ids]`; an agent-specific server is just a connector assigned to one agent. `snapshot_for_agent` serializes **stdio records only** into `MCP_SERVERS` — an `HttpConnector` (incl. its `auth`) must never enter a container (pinned by test); remote calls go through the mesh-side `MCPGateway` (`src/host/mcp_gateway.py` — per-call streamable-HTTP sessions, vault-resolved auth, resolved-IP SSRF blocklist + no redirects, 256 KiB result cap, sampling/elicitation rejected; `/mesh/connectors/tools\|call` are assignment-gated for EVERY agent incl. the operator — NOT in the trust-tier bypass, pinned by HTTP-level test). Edits apply on restart only — EXCEPT http auth-only edits (`upsert` returns restart-relevance; remote auth resolves per call on the mesh; no per-agent cred gate on `auth.cred` — the token is mesh-held). The dashboard Connectors page (sub-tab ID `integrations` is frozen; label is "Connectors") returns `affected_agents` and prompts; it never auto-restarts. Mtime auto-reload picks up hand-edited files. Pending-restart is a generation derivation (immune to the edit-during-container-build race). A missing/denied `$CRED` drops that one connector with an error log — never blocks agent boot. Connector writes are human/dashboard-only. |
| `src/host/credentials.py` | Two-tier vault (SYSTEM_*/CRED_*) + LLM API proxy + OAuth. `is_model_compatible` is the single source of truth for model-allowlist checks (OAuth model subsets are env-overridable — read the code, not stale lists). |
| `src/host/permissions.py` | Per-agent ACL (glob patterns, deny-all default). `browser_actions: list[str] \| None` narrows per-action surface (`None` = all allowed, `[]` = deny all). |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter, `audit_log` with `undoable` + `archived`. |
| `src/host/lanes.py` | `LaneManager` per-agent FIFO task queues (followup/steer). `QueuedTask.task_id` threads through `_direct_dispatch` to make auto-close work (Constraint #6). |
| `src/host/chain_watcher.py` | Delegate-and-subscribe machinery: durable chain watcher, stall watchdog, opt-in milestone pings. Delivers only to the chain-root first-party human origin. |
| `src/host/cron.py` | Persistent cron scheduler. `_UPDATABLE_FIELDS` frozenset allowlist. Bootstraps per-team daily summary jobs (`ensure_summary_job`). |
| `src/host/summaries.py` | `WorkSummariesStore` — one row per `(scope_kind, scope_id, period_start)`, `scope_kind ∈ {team, solo}`. Ratings lock after a 24h edit window. `feedback_push.py` closes the rating→learning loop. |
| `src/browser/service.py` | Per-agent Camoufox + X11 WID tracking. Fingerprint burn detection; operator clears burn manually. `display_allocator.py` owns the display/VNC-port ranges. |
| `src/browser/captcha.py` | 2captcha + capsolver. Behavioral kinds rejected via `request_captcha_help` handoff. Millicent cost accounting, per-agent + per-tenant caps, kill switch, circuit breaker. |
| `src/browser/flags.py` | Centralized flag loader. `_ENV_ONLY_FLAGS` strips solver creds from `config/settings.json` at load (env-only by design). |
| `src/browser/stealth.py` | `BROWSER_DEVICE_PROFILE` rewrites UA, but the underlying Camoufox engine is unchanged — server-side TLS/JA3 may still read desktop. |
| `src/browser/server.py` | Raises on startup if `MESH_AUTH_TOKEN` set but `BROWSER_AUTH_TOKEN` missing. Session persistence is opt-in via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default false). |
| `src/dashboard/server.py` | Alpine.js SPA, Jinja autoescape, CSP, CSRF via `X-Requested-With`, VNC URL injection. Four top-nav tabs with frozen IDs (Constraint #5). |
| `src/dashboard/notifications.py` | Top-nav bell store; `_KNOWN_KINDS` is frozen. No automatic per-task-completion notifications — the operator authors genuinely user-facing ones via `notify_user`. |
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

- **Operator internet access.** `can_use_internet` defaults to False for workers, but the operator agent is granted it by default (`_ensure_operator_agent`). Intentional UX tradeoff.
- **Dashboard permission edits have NO operator ceiling** (`PUT /api/agents/{id}/permissions`). That path is the human's deliberate escalation surface; `_OPERATOR_PERMISSION_CEILING` applies only to the LLM/mesh edit path. Don't add a ceiling or a config toggle to the dashboard path (audit H1).
- **The blackboard is shared by design** (audit H10). Don't naively enforce per-team key prefixes — it breaks the default agent and template coordination. Team scoping goes through team wiring.
- **AST validation on self-authored tools is hygiene, not a sandbox.** Agents already have `run_command` (in-container code execution by design); the Docker container is the boundary. Marketplace tools (`/app/marketplace_tools`) load WITHOUT AST validation — trusted because the dir is operator-populated and read-only. If a remote/agent-reachable install path is ever added, pin installs to a verified commit SHA (M1, H15).
- **Operator trust-tier carve-out** (Constraint #12). `_caller_is_operator` short-circuits the coordination/management gates (messaging, pubsub, cron, fleet, task routing, spawn, blackboard ops) by design. Still gated like any worker: `can_use_wallet*`, `can_access_credential`, `can_manage_vault`, `can_browser_action`. `test_operator_still_gated_surfaces_not_in_bypass_grep` pins the boundary.

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly; agents coordinate through the blackboard.
2. **Tools over features.** New agent capabilities go in as `@tool`-decorated functions, not loop changes.
3. **`apply_template` is per-slot, not atomic.** Mesh validates upfront, but a mid-loop failure leaves earlier-created agents in place — verify the returned `created` list matches the requested slot set.
4. **Propagate `MessageOrigin`.** New cross-agent paths producing work for another agent must read `current_origin` once and forward it to both `wake_agent` and `create_task` — otherwise the receiving lane worker can't auto-notify the originating channel/user on handoff completion.
5. **Tab IDs are frozen for URL stability.** The four top-nav tab IDs (`chat`, `fleet`, `workplace`, `system`) appear in URL paths, JS state, and dashboard endpoints. Labels may diverge (`fleet`→"Teams", `workplace`→"Work", `system`→"Settings") but never rename the IDs.
6. **Auto-close requires task_id plumbing.** Handed-off tasks auto-transition to terminal status only when the wake chain carries `x-task-id`. Wakes without one (heartbeats, manual chats) won't auto-close — intentional. Back-edge events go to `inbox/{agent}/task_event/{id}` (7-day TTL), surfaced via `check_inbox`.
7. **LLM tool-calling roles must alternate** (`user → assistant(tool_calls) → tool(result) → assistant`). `_trim_context` merges the summary into the first user message to preserve this invariant.
8. **Avoid new module-level globals.** Existing exceptions: `_tool_staging` in `tools.py` (lock-protected), `_client` in `http_tool.py` (connection pool).
9. **Project→team rename (2026-05) shims remain**: `can_manage_projects` validator, `tasks.project_id` emitted alongside `team_id` (pending external-consumer audit). Internal namespaces unchanged: blackboard `projects/{name}/` prefix, `target_kind="project"` on `pending_actions`. `src/host/team_migration.py` is a startup migrator pending removal.
10. **Coordination-tool failure envelopes.** Errors from `hand_off` / `update_status` / `complete_task` wrap via `_failed_transition_envelope` so the LLM sees `handed_off=False` + a directive `error` + `recovery_hint`; sentinel keys merge AFTER caller `extras`. Without this shape, LLMs silently report success when handoffs fail post-commit.
11. **Wizard state machine** `idle | ask | confirming | building | first-output | build_failed`, persisted to `localStorage.ol_wizard`; resets to `idle` on unknown values; mounts only when `step !== 'idle'`; mutually exclusive with the "What's new" tour.
12. **Operator trust-tier carve-out** — see "Deliberate Tradeoffs" above for the bypassed vs. still-gated surfaces (code comments reference this as Constraint #12, e.g. `mesh_tool.py`). Boot fail-closed: empty `auth_tokens` under `enforce` mode → `SystemExit` (X-Agent-ID would be forgeable).
