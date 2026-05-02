# Operator Orchestration Roadmap

**Goal:** Give operators a coherent control plane for project-scoped fleets — durable task records, hard project isolation, structured routing, dashboard + CLI surfaces — so a human can drive a multi-agent team end-to-end without inspecting blackboard keys.

**Core invariant:** A worker only ever sees its own project's agents and tasks. The operator and trusted internal callers (mesh, dashboard, CLI manager) remain fleet-global.

---

## Tasks

| Task | Description | PR |
|---|---|---|
| Task 2c | Server-side channel pairing recheck on every origin downgrade. | #819 |
| Task 2d | SQLite-backed pending-action store (replaces in-memory dict). | #821 |
| Task 2e | Block worker → operator synchronous wakes. | #823 |
| Task 3 | Split ``can_spawn`` into a control-plane permission set. | #824 |
| Task 4 | Cryptographic operator authentication on register. | #826 |
| Task 5 | Server-side project isolation (warn → enforce). | #828 |
| Task 6 | Durable orchestration task records behind ``OPENLEGION_ORCHESTRATION_TASKS_V2``. | #830 |
| Task 7 | Operator product tools for projects / teams / tasks. | #831 |
| Task 8 | Structured routing fields on ``AgentConfig``. | #832 |
| Task 9 | Dashboard Workplace tab + CLI ``tasks`` / ``pending`` surfaces. | #834 |
| Rollout | Flip the two opt-in flags to default-on; auto-run migration. | this PR |

---

## Feature flags

| Flag | Default | Behavior when default | Off-switch (rollback) |
|---|---|---|---|
| ``OPENLEGION_PROJECT_SCOPE_MODE`` | ``enforce`` | Workers see only own-project members on ``/mesh/agents`` and project-scoped blackboard ACLs. Operators + internal callers remain fleet-global. | Set to ``warn`` to restore legacy fleet-wide visibility while still emitting structured ``scope-warn`` log lines. |
| ``OPENLEGION_ORCHESTRATION_TASKS_V2`` | ``1`` | Mesh constructs a ``Tasks`` SQLite store; coordination tool routes ``hand_off`` / ``check_inbox`` / ``update_status`` / ``complete_task`` through ``/mesh/tasks*``; legacy blackboard task migration auto-runs once at startup. | Set to ``0`` to disable v2: ``/mesh/tasks*`` returns 503, coordination falls back to the legacy blackboard-dict path, migration does not auto-run. |
| ``OPENLEGION_STRICT_HUMAN_CONFIRMATION`` | ``1`` (already on) | Pending actions strictly require a human confirmation envelope. | Set to ``0`` only for break-glass debugging. |

Both flags read the env var once at module import — runtime flips require a mesh restart.

---

## Rollout sequence

**Shipped official: defaults flipped on in ``feat/orchestration-rollout-official``.**

The roadmap shipped each piece behind an opt-in env flag so we could soak before flipping defaults. After landing Tasks 2c through 9 the user decided to skip the staged rollout and ship everything live in one PR. This PR:

1. Flips ``OPENLEGION_PROJECT_SCOPE_MODE`` default ``warn`` → ``enforce``.
2. Flips ``OPENLEGION_ORCHESTRATION_TASKS_V2`` default ``0`` → ``1``.
3. Auto-runs ``migrate_blackboard_to_tasks`` once at mesh startup (idempotent — subsequent restarts are no-ops).

### Rollback plan

The off-switches still work for emergency rollback:

- ``OPENLEGION_PROJECT_SCOPE_MODE=warn`` — restores legacy fleet-wide ``/mesh/agents`` visibility for workers; the structured ``scope-warn`` log lines and ``scope_warn_total`` metric stay live so operators can still measure what would have been denied.
- ``OPENLEGION_ORCHESTRATION_TASKS_V2=0`` — disables the v2 path; ``/mesh/tasks*`` returns 503; coordination falls back to the legacy blackboard-dict path; the auto-run migration is skipped (existing legacy keys stay where they are).

Restart the mesh after flipping either env var.

---

## Reality check by task

### Task 5 — project scope isolation

- Worker calls to ``/mesh/agents`` filter to own-project members + always-global operator under ``enforce``.
- ``warn`` mode emits a structured ``scope-warn`` log line and bumps ``scope_warn_total`` on every call that *would* have been denied — operators read these to size impact before flipping the env var.
- Operator + internal (loopback ``x-mesh-internal: 1``) callers are exempt from filtering by design — dashboard, CLI manager, health monitor all rely on fleet-wide visibility.

### Task 6 — durable orchestration task records

Storage layer: ``Tasks`` SQLite table keyed by ``task_id``. Endpoints under ``/mesh/tasks*`` (``create``, ``get``, ``list_inbox``, ``list_project``, ``update_status``, ``reroute``, ``cancel``, ``add_artifact``, ``reap_expired``). Permission gates by ``can_route_tasks`` + per-project membership. Coordination tool probes ``/mesh/orchestration/status`` once and caches the result.

#### Migration

The blackboard → tasks migration (``src/host/orchestration_migration.py``) auto-runs **once at mesh startup** when the v2 flag is on and a ``tasks_store`` was constructed. It is idempotent — keyed on the legacy ``handoff_id`` → new ``task_id`` — so subsequent restarts find nothing to migrate and are no-ops. Migration failures are logged but never crash startup; the legacy keys stay in place and operators can re-run the helper manually via the standalone import.

The auto-run skips entirely when the v2 flag is off (someone deliberately disabled v2; respect that).

### Task 7 — operator product tools

Operator skills layer over the Task 6 endpoints: ``list_project_status``, ``list_agent_queue``, ``get_team_outputs``, ``reroute_task``, ``retry_task``. All return ``{"error": ...}`` referencing ``OPENLEGION_ORCHESTRATION_TASKS_V2`` when the flag is off so operators get a clean signal instead of garbled data.

### Task 8 — structured routing fields

``AgentConfig`` carries ``role``, ``project``, ``capabilities`` so the dashboard Workplace tab and the operator product tools can group agents by role/project without parsing free-form ``instructions``.

### Task 9 — dashboard + CLI surfaces

- Workplace tab on the dashboard (peer of Chat / Agents / System): board view of tasks grouped by project / assignee, pending-action review pane.
- CLI: ``openlegion tasks`` and ``openlegion pending`` print the same data via ``/dashboard/api/workplace/*``.
- Both degrade to a clean empty state with a hint to flip ``OPENLEGION_ORCHESTRATION_TASKS_V2=1`` when the flag is off.

---

## Test plan

- ``tests/test_rollout_defaults.py`` — verifies default-on for both flags + that the off-switches still work.
- ``tests/test_orchestration_migration.py`` — extended with auto-run-at-startup, idempotence, and failure-doesn't-crash coverage.
- Existing ``tests/test_orchestration_endpoints.py`` and ``tests/test_operator_product_tools.py`` already explicitly set the env vars; updated as needed.
- Existing ``tests/test_mesh.py`` (Task 5 scope tests) already explicitly set ``OPENLEGION_PROJECT_SCOPE_MODE`` for both warn and enforce paths.
