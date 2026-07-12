"""HTTP endpoint tests for orchestration tasks v2 (Task 6).

Boots the mesh app, drives the ``/mesh/tasks*`` routes through
``ASGITransport``, and verifies the permission gates and 503
fall-through.
"""

from __future__ import annotations

import importlib

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions, MessageOrigin


def _reload_server(monkeypatch, *, tasks_db: str):
    """Reload ``src.host.server`` after pinning the tasks DB path.

    The tasks-store DB path is read at module import via env var so a
    test that wants to point it at ``tmp_path`` must reload the module
    after ``setenv``.
    """
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _build_app(tmp_path, server_module, *, perms_map, agents=None):
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, agents or {})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    return app, blackboard


def _seed_teams(app):
    """Two teams: research with [scout, analyst]; ops with [tracker]."""
    app.teams_store.create_team("research")
    app.teams_store.add_member("research", "scout")
    app.teams_store.add_member("research", "analyst")
    app.teams_store.create_team("ops")
    app.teams_store.add_member("ops", "tracker")


@pytest.fixture
def v2_app(tmp_path, monkeypatch):
    """Build a fresh v2-enabled mesh app for each test."""
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    # Pin the track-record ledger DB into tmp_path too — otherwise every
    # test in this module (and every OTHER test file that doesn't
    # override it) shares one on-disk data/track_record.db and accrues
    # rows across the whole test session, making count assertions flaky.
    monkeypatch.setenv("OPENLEGION_TRACK_RECORD_DB", str(tmp_path / "track_record.db"))
    # Same isolation for the held-actions store — without this pin every
    # mesh app built in one pytest process shares one cwd
    # data/pending_actions.db, so the queue-capacity and recommend/hold
    # assertions below cross-contaminate with other test files.
    monkeypatch.setenv("OPENLEGION_PENDING_ACTIONS_DB", str(tmp_path / "pending_actions.db"))
    perms_map = {
        "operator": {"can_route_tasks": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
        "tracker":  {"can_route_tasks": False, "can_message": []},
    }
    app, bb = _build_app(
        tmp_path, server_module,
        perms_map=perms_map,
        agents={
            "scout": "http://scout:8400",
            "analyst": "http://analyst:8400",
            "tracker": "http://tracker:8400",
            "operator": "http://operator:8400",
        },
    )
    yield app, server_module, tmp_path
    bb.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_TRACK_RECORD_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_PENDING_ACTIONS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_create_task_success(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={
                "assignee": "analyst",
                "title": "research handoff",
                "description": "dig into X",
            },
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["assignee"] == "analyst"
    assert body["creator"] == "scout"
    assert body["status"] == "pending"
    assert body["id"].startswith("task_")


@pytest.mark.asyncio
async def test_create_task_requires_can_message_to_assignee(v2_app):
    """``POST /mesh/tasks`` is gated on ``can_message(caller, assignee)``.

    The fixture's ``tracker`` has ``can_message=[]`` (deny all), so even
    though ``can_route_tasks`` is the legacy back-compat field that some
    callers still grant, lacking the per-target ``can_message`` entry is
    the actual blocker.
    """
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # tracker has can_message=[] — cannot message scout (or anyone).
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "scout", "title": "no permission"},
            headers={"X-Agent-ID": "tracker"},
        )
    assert r.status_code == 403
    assert "can_message" in r.json().get("detail", "")


@pytest.mark.asyncio
async def test_create_task_default_collab_works_out_of_box(tmp_path, monkeypatch):
    """Headline architectural test: a worker with default collab-mode
    perms (``can_message=["*"]``) can create a task for a peer
    out-of-the-box — no ``can_route_tasks`` toggle needed.

    This is the test that demonstrates pipelines work without operator
    intervention. Mirrors the defaults applied by
    ``cli/config.py::_add_agent_permissions``.
    """
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    perms_map = {
        # No can_route_tasks set anywhere; just collab-mode defaults.
        "alpha": {"can_message": ["*"]},
        "bravo": {"can_message": ["*"]},
    }
    app, bb = _build_app(
        tmp_path, server_module,
        perms_map=perms_map,
        agents={
            "alpha": "http://alpha:8400",
            "bravo": "http://bravo:8400",
        },
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/tasks",
                json={"assignee": "bravo", "title": "out of the box"},
                headers={"X-Agent-ID": "alpha"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["assignee"] == "bravo"
        assert body["creator"] == "alpha"
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_worker_can_create_task_for_operator(tmp_path, monkeypatch):
    """Worker→operator task creation is the intended async escalation path.

    ``/mesh/wake`` explicitly blocks worker→operator synchronous wakes
    (workers can be steered into privileged actions by anyone able to
    message them). The task queue is the right channel: it's async,
    operator processes it on heartbeat. Codex review of PR #954
    flagged this design intent — pin it with a regression test so a
    future "mirror the wake block" patch doesn't silently break the
    standard worker-completion path.
    """
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    perms_map = {
        # Default collab-mode worker permissions.
        "worker": {"can_message": ["*"]},
        # Operator is permissioned the way ``_ensure_operator_agent``
        # produces it post-PR#954 (no can_route_tasks needed).
        "operator": {"can_message": ["*"]},
    }
    app, bb = _build_app(
        tmp_path, server_module,
        perms_map=perms_map,
        agents={
            "worker": "http://worker:8400",
            "operator": "http://operator:8400",
        },
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/tasks",
                json={
                    "assignee": "operator",
                    "title": "Escalating: blocker on stage 3",
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["assignee"] == "operator"
        assert body["creator"] == "worker"
        assert body["status"] == "pending"
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_create_task_invalid_assignee(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "../../etc", "title": "evil"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_get_task_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/tasks/missing",
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_task_visible_to_creator_assignee_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout creates a task for analyst.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "x"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Creator can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "scout"})
        assert r.status_code == 200
        # Assignee can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "analyst"})
        assert r.status_code == 200
        # Operator can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "operator"})
        assert r.status_code == 200
        # Outsider cannot — L16: returns a uniform 404 (not 403) so the
        # endpoint can't be used as an existence oracle.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "tracker"})
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_inbox_visibility_assignee_only(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout creates two tasks for analyst.
        for title in ("task1", "task2"):
            await c.post(
                "/mesh/tasks",
                json={"assignee": "analyst", "title": title},
                headers={"X-Agent-ID": "scout"},
            )
        # Assignee can read own inbox.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2
        # Operator can read any inbox.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Other worker cannot.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_project_list_requires_membership(v2_app, monkeypatch):
    app, _, tmp_path = v2_app
    _seed_teams(app)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout (member of research) creates a task in research.
        await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "scoped", "team_id": "research"},
            headers={"X-Agent-ID": "scout"},
        )
        # Member can list project tasks.
        r = await c.get(
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1
        # Non-member is denied.
        r = await c.get(
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Operator gets through.
        r = await c.get(
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_status_update_creator_assignee_operator_outsider(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "transition"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Outsider denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Assignee can transition pending → working.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "working"
        # Invalid transition → 400.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "pending"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_reroute_requires_operator_or_internal(v2_app):
    """Reroute is now operator-only (administrative recovery action).

    Workers — even with the legacy ``can_route_tasks=True`` grant — must
    not be able to reroute tasks. Only operator or ``x-mesh-internal``
    callers succeed.
    """
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "routed"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Operator can reroute.
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "tracker", "reason": "load balance"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Worker without can_route_tasks denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "analyst"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 403
        assert "operator-only" in r.json().get("detail", "")
        # Worker WITH legacy can_route_tasks=True ALSO denied — reroute
        # is operator-only now (the gate no longer consults the field).
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "scout"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 403
        assert "operator-only" in r.json().get("detail", "")


@pytest.mark.asyncio
async def test_retry_requires_operator_or_internal(v2_app):
    """Retry is operator-only — workers with ``can_route_tasks`` are
    rejected. Sibling contract to reroute."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Seed a task, then force-fail it via internal call so retry
        # is eligible.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "retry target"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Drive it to failed via assignee status updates.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "failed"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        # Worker WITH legacy can_route_tasks=True (scout) is rejected.
        r = await c.post(
            f"/mesh/tasks/{tid}/retry",
            json={},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 403
        assert "operator-only" in r.json().get("detail", "")
        # Operator succeeds.
        r = await c.post(
            f"/mesh/tasks/{tid}/retry",
            json={},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_cancel_creator_or_assignee_or_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "doomed"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Outsider denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "no"},
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Creator can cancel.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "user changed mind"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_failed_status_promotes_error_to_blocker_note(v2_app):
    """Bug 3 fix: ``POST /mesh/tasks/{id}/status`` with status=failed +
    ``error`` body field (the shape ``mesh_client.set_task_status``
    sends from auto-close paths) must promote ``error`` to the
    persisted ``blocker_note`` column when no explicit blocker_note was
    given. Without this, agent-loop failures left ``blocker_note=NULL``
    and the dashboard rendered a bare "failed task" pill with no
    reason."""
    app, server_module, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "promotion check"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        # Auto-close shape — error field, no blocker_note.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={
                "status": "failed",
                "error": "config_error: model 'openai/gpt-4o-mini' not in OAuth allowlist",
            },
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "failed"
        assert body["blocker_note"] is not None
        assert "config_error" in body["blocker_note"]
        assert "OAuth allowlist" in body["blocker_note"]


@pytest.mark.asyncio
async def test_done_status_persists_result_summary(v2_app):
    """``POST /mesh/tasks/{id}/status`` with a ``result.summary`` body
    persists the worker's deliverable onto the task row (via
    ``result_summary``) so ``await_task_event`` / GET surface it — not
    only the origin-gated back-edge inbox event."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "summary check"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "done", "result": {"summary": "done-summary"}},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["result_summary"] == "done-summary"
        # GET reflects the persisted value too.
        r = await c.get(
            f"/mesh/tasks/{tid}",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["result_summary"] == "done-summary"


@pytest.mark.asyncio
async def test_failed_status_explicit_blocker_note_wins_over_error(v2_app):
    """Explicit ``blocker_note`` in the body wins over ``error`` — the
    promotion only fires when blocker_note is missing/empty."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "precedence"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={
                "status": "failed",
                "error": "should NOT override",
                "blocker_note": "explicit note wins",
            },
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["blocker_note"] == "explicit note wins"


@pytest.mark.asyncio
async def test_failed_status_error_truncated_to_500_chars(v2_app):
    """Huge ``error`` strings (e.g. LLM tracebacks) are bounded to 500 chars
    in the promoted ``blocker_note`` (now via the central
    ``normalize_blocker_note`` choke point) to avoid runaway bloat."""
    app, _, _ = v2_app
    # A long, non-secret-shaped message so the assertion targets the length
    # bound, not the redactor (which would shrink a token-shaped run).
    huge = "verbose traceback line. " * 300
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "huge"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "failed", "error": huge},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        note = r.json()["blocker_note"]
        assert note is not None
        assert len(note) <= 500
        assert note.startswith("verbose traceback line.")


@pytest.mark.asyncio
async def test_events_endpoint_returns_audit_history(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "audited"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.get(
            f"/mesh/tasks/{tid}/events",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        kinds = [e["event_kind"] for e in r.json()["events"]]
        assert kinds == ["created", "status_changed"]


# ── Task 9 — /mesh/pending/* endpoints ─────────────────────────────


@pytest.mark.asyncio
async def test_pending_list_endpoint_returns_open_nonces(v2_app):
    """``GET /mesh/pending`` returns rows from the SQLite store."""
    app, _, _ = v2_app
    pa = app.pending_actions
    pa.store(
        nonce="n-test", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/pending", headers={"X-Mesh-Internal": "1"})
        assert r.status_code == 200
        rows = r.json()["pending"]
        assert any(p["nonce"] == "n-test" for p in rows)


@pytest.mark.asyncio
async def test_pending_cancel_endpoint_deletes_row(v2_app):
    """``POST /mesh/pending/{nonce}/cancel`` deletes the row."""
    app, _, _ = v2_app
    pa = app.pending_actions
    pa.store(
        nonce="n-cancel", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n-cancel/cancel",
            headers={"X-Agent-ID": "operator", "X-Mesh-Internal": "1"},
        )
        assert r.status_code == 200
        assert r.json()["nonce"] == "n-cancel"
    # Row gone — peek confirms.
    assert pa.peek("n-cancel") is None


@pytest.mark.asyncio
async def test_pending_cancel_unknown_returns_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/missing-nonce/cancel",
            headers={"X-Agent-ID": "operator", "X-Mesh-Internal": "1"},
        )
        assert r.status_code == 404


# ── Held-actions generalization: executor registry (plan §8 #17, C.1 row 6) ──


def _human_internal_headers(agent_id: str = "operator") -> dict:
    """Internal-caller headers with a human-kind X-Origin — the shape a
    dashboard confirm click actually sends."""
    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    return {
        "X-Agent-ID": agent_id,
        "X-Mesh-Internal": "1",
        "X-Origin": origin.to_header_value(),
    }


@pytest.mark.asyncio
async def test_pending_list_endpoint_surfaces_tier(v2_app):
    """GET /mesh/pending includes the tier column on each row."""
    app, _, _ = v2_app
    app.pending_actions.store(
        nonce="n-tier", actor="operator", target_kind="wallet",
        target_id="scout", action_kind="wallet_transfer", payload={"x": 1},
        tier="financial",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/pending", headers={"X-Mesh-Internal": "1"})
    assert r.status_code == 200
    row = next(p for p in r.json()["pending"] if p["nonce"] == "n-tier")
    assert row["tier"] == "financial"


@pytest.mark.asyncio
async def test_confirm_unregistered_action_kind_returns_400(v2_app):
    """A pending row whose ``action_kind`` has no registered executor
    (a stray/corrupt row, or a future producer that forgot to register
    one) is refused loudly rather than silently mis-dispatched."""
    app, _, _ = v2_app
    app.pending_actions.store(
        nonce="n-stray", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="not_a_registered_kind",
        payload={"x": 1}, origin_kind="human",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/config/confirm",
            json={"change_id": "n-stray"},
            headers=_human_internal_headers(),
        )
    assert r.status_code == 400
    assert "not_a_registered_kind" in r.text


@pytest.mark.asyncio
async def test_confirm_dispatches_via_executor_registry(v2_app):
    """A custom executor registered on ``app.pending_executors`` is
    reached through the SAME ``/mesh/config/confirm`` endpoint the
    delete flow uses — proves the dispatch is a real registry lookup,
    not a hard-coded delete/{team,agent} branch."""
    app, _, _ = v2_app
    calls = []

    async def _custom_executor(record):
        calls.append(record["nonce"])
        return {"custom": True, "payload": record["payload"]}

    app.pending_executors["custom_kind"] = _custom_executor
    app.pending_actions.store(
        nonce="n-custom", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="custom_kind",
        payload={"hello": "world"}, origin_kind="human",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/config/confirm",
            json={"change_id": "n-custom"},
            headers=_human_internal_headers(),
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"custom": True, "payload": {"hello": "world"}}
    assert calls == ["n-custom"]
    # Single-use: a second confirm attempt finds the row already consumed.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r2 = await c.post(
            "/mesh/config/confirm",
            json={"change_id": "n-custom"},
            headers=_human_internal_headers(),
        )
    assert r2.status_code == 400


@pytest.mark.asyncio
async def test_delete_proposal_still_evicts_oldest_at_cap(v2_app):
    """Regression: the DELETE producers keep their evict-oldest-at-cap
    behavior (operator-initiated surface; newest proposal winning is
    correct there) — only the new policy hold producers fail closed."""
    from src.host.server import _MAX_PENDING

    app, _, _ = v2_app
    pa = app.pending_actions
    app.teams_store.create_team("doomed")
    app.teams_store.set_status("doomed", "archived")
    # The store is tmp_path-isolated per test (OPENLEGION_PENDING_ACTIONS_DB
    # pin in v2_app) — no pre-clean or post-purge needed anymore.
    # pre-0 gets the shortest TTL -> smallest expires_at -> eviction target.
    pa.store(
        nonce="pre-0", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="delete", payload={},
        origin_kind="human", ttl=60,
    )
    for i in range(1, _MAX_PENDING):
        pa.store(
            nonce=f"pre-{i}", actor="operator", target_kind="agent",
            target_id="alpha", action_kind="delete", payload={},
            origin_kind="human", ttl=300,
        )
    assert len(pa.list_pending()) == _MAX_PENDING
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/doomed/propose-delete",
            headers=_human_internal_headers(),
        )
    assert r.status_code == 200, r.text
    new_nonce = r.json()["change_id"]
    # Evicted exactly the oldest; stored the new proposal; cap held.
    assert len(pa.list_pending()) == _MAX_PENDING
    assert pa.peek("pre-0") is None
    for i in range(1, _MAX_PENDING):
        assert pa.peek(f"pre-{i}") is not None
    assert pa.peek(new_nonce) is not None


@pytest.mark.asyncio
async def test_confirm_agent_origin_403_for_non_delete_kind(v2_app):
    """The human-origin confirm gate is unconditional — it fires for ANY
    action kind, not just delete, before the executor is even looked up."""
    app, _, _ = v2_app
    app.pending_executors["custom_kind"] = lambda record: {"ok": True}
    app.pending_actions.store(
        nonce="n-agent-origin", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="custom_kind", payload={},
    )
    agent_origin = MessageOrigin(kind="agent", channel="", user="").to_header_value()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/config/confirm",
            json={"change_id": "n-agent-origin"},
            headers={
                "X-Agent-ID": "operator", "X-Mesh-Internal": "1",
                "X-Origin": agent_origin,
            },
        )
    assert r.status_code == 403
    # Row untouched — a legitimate human confirm can still land.
    assert app.pending_actions.peek("n-agent-origin") is not None


# ── Lead advisory recommendation (plan §8 #19) ──────────────────────
#
# POST /mesh/pending/{nonce}/recommend — gated exactly like the drive-
# verdict endpoint's lead-only pattern: the verified caller must equal
# the TEAM LEAD of the agent who PROPOSED the held action. ZERO
# enforcement: recommending never touches confirm/cancel/consume.


def _store_wallet_hold(app, nonce: str, proposer: str) -> None:
    app.pending_actions.store(
        nonce=nonce, actor="operator", target_kind="wallet",
        target_id=proposer, action_kind="wallet_transfer",
        payload={"agent_id": proposer, "chain": "eth", "to": "0xabc", "amount": "1"},
    )


@pytest.mark.asyncio
async def test_recommend_lead_records_and_surfaces(v2_app):
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve", "note": "looks fine"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recorded"] is True
    assert body["pending"]["lead_recommendation"] == "approve"
    assert body["pending"]["lead_recommendation_note"] == "looks fine"
    assert body["pending"]["lead_recommendation_by"] == "scout"
    # Surfaces through the list endpoint too.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r2 = await c.get("/mesh/pending", headers={"X-Mesh-Internal": "1"})
    row = next(p for p in r2.json()["pending"] if p["nonce"] == "n1")
    assert row["lead_recommendation"] == "approve"


@pytest.mark.asyncio
async def test_recommend_emits_pending_action_updated_event(tmp_path, monkeypatch):
    """Phase-5 review finding: recording a recommendation must push a
    ``pending_action_updated`` WS event so an already-rendered inline card
    refreshes live instead of only on a full dashboard reload."""
    from src.dashboard.events import EventBus

    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    monkeypatch.setenv("OPENLEGION_TRACK_RECORD_DB", str(tmp_path / "track_record.db"))
    monkeypatch.setenv("OPENLEGION_PENDING_ACTIONS_DB", str(tmp_path / "pending_actions.db"))
    bus = EventBus()
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    for aid in ("scout", "analyst"):
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, can_route_tasks=True)
    router = MessageRouter(permissions, {"scout": "http://s:8400", "analyst": "http://a:8400"})
    app = server_module.create_mesh_app(
        blackboard=blackboard, pubsub=PubSub(), router=router,
        permissions=permissions, event_bus=bus,
    )
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "reject", "note": "hold off"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 200, r.text
    updates = [e for e in bus._buffer if e["type"] == "pending_action_updated"]
    assert len(updates) == 1, bus._buffer
    data = updates[0]["data"]
    assert data["nonce"] == "n1"
    assert data["lead_recommendation"] == "reject"
    assert data["lead_recommendation_note"] == "hold off"
    assert data["lead_recommendation_by"] == "scout"


@pytest.mark.asyncio
async def test_recommend_non_lead_teammate_403(v2_app):
    """The proposer itself is NOT the lead — recommending is still
    denied (analyst proposed this hold and is not research's lead)."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "analyst", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 403
    assert app.pending_actions.peek("n1")["lead_recommendation"] is None


@pytest.mark.asyncio
async def test_recommend_non_lead_operator_403(v2_app):
    """A non-lead operator 403s too — mirrors the drive-verdict
    endpoint's own operator-gets-no-carve-out posture."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "operator", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_recommend_other_team_lead_403(v2_app):
    """A different team's lead has no standing on this hold."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    app.teams_store.set_lead("ops", "tracker")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "tracker", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_recommend_teamless_proposer_409(v2_app):
    """The proposing agent isn't on any team — no lead to route to."""
    app, _, _ = v2_app
    _seed_teams(app)
    _store_wallet_hold(app, "n1", "solo-agent")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_recommend_team_with_no_lead_409(v2_app):
    app, _, _ = v2_app
    _seed_teams(app)  # research has no lead assigned
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_recommend_operator_proposed_hold_409(v2_app):
    """team_delete/agent_delete rows have no agent proposer at all —
    routes to nobody, 409s directively rather than guessing."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.create_team("doomed")
    app.teams_store.set_status("doomed", "archived")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        propose = await c.post(
            "/mesh/teams/doomed/propose-delete",
            headers=_human_internal_headers(),
        )
    nonce = propose.json()["change_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/mesh/pending/{nonce}/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_recommend_unknown_nonce_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/does-not-exist/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_recommend_expired_nonce_404(v2_app):
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    app.pending_actions.store(
        nonce="n-exp", actor="operator", target_kind="wallet",
        target_id="analyst", action_kind="wallet_transfer",
        payload={"agent_id": "analyst"}, ttl=-1,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n-exp/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_recommend_invalid_value_400(v2_app):
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "maybe"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_recommend_note_length_capped(v2_app):
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    _store_wallet_hold(app, "n1", "analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve", "note": "x" * 501},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_recommend_reject_does_not_block_confirm(v2_app):
    """ZERO enforcement pin: a reject recommendation must not stop the
    human confirm path from executing the held action."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    calls = []

    async def _custom_executor(record):
        calls.append(record["nonce"])
        return {"executed": True}

    app.pending_executors["wallet_transfer"] = _custom_executor
    app.pending_actions.store(
        nonce="n1", actor="operator", target_kind="wallet",
        target_id="analyst", action_kind="wallet_transfer",
        payload={"agent_id": "analyst"}, origin_kind="human",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        rec = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "reject"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert rec.status_code == 200
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        confirm = await c.post(
            "/mesh/pending/n1/confirm",
            headers=_human_internal_headers(),
        )
    assert confirm.status_code == 200, confirm.text
    assert calls == ["n1"]


@pytest.mark.asyncio
async def test_recommend_approve_does_not_auto_execute(v2_app):
    """ZERO enforcement pin: an approve recommendation must not itself
    release the hold — the executor only runs on an explicit confirm."""
    app, _, _ = v2_app
    _seed_teams(app)
    app.teams_store.set_lead("research", "scout")
    calls = []

    async def _custom_executor(record):
        calls.append(record["nonce"])
        return {"executed": True}

    app.pending_executors["wallet_transfer"] = _custom_executor
    app.pending_actions.store(
        nonce="n1", actor="operator", target_kind="wallet",
        target_id="analyst", action_kind="wallet_transfer",
        payload={"agent_id": "analyst"}, origin_kind="human",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        rec = await c.post(
            "/mesh/pending/n1/recommend",
            json={"recommendation": "approve"},
            headers={"X-Agent-ID": "scout", "X-Mesh-Internal": "1"},
        )
    assert rec.status_code == 200
    assert calls == []  # not executed
    assert app.pending_actions.peek("n1") is not None  # still pending


# ── Hotfix: _extract_verified_agent_id honors internal callers ──────
#
# In production (auth tokens configured), the dashboard cancel proxy
# hits ``/mesh/tasks/{id}/cancel`` over loopback with
# ``x-mesh-internal: 1`` + ``X-Agent-ID: operator`` but NO Bearer token
# (Bearer tokens are server-side secrets the dashboard doesn't have).
# The previous ``_extract_verified_agent_id`` rejected those callers
# with 401 "Missing authentication token". Trust the loopback boundary
# instead.


@pytest.fixture
def v2_app_with_auth(tmp_path, monkeypatch):
    """v2 mesh app with auth tokens configured (production-like)."""
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    perms_map = {
        "operator": {"can_route_tasks": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
    }
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, {
        "scout": "http://scout:8400",
        "analyst": "http://analyst:8400",
        "operator": "http://operator:8400",
    })
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        # Auth tokens configured — public callers must Bearer-auth.
        auth_tokens={
            "operator": "tok_operator",
            "scout":    "tok_scout",
            "analyst":  "tok_analyst",
        },
    )
    yield app, server_module
    blackboard.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_internal_caller_skips_bearer_requirement(v2_app_with_auth):
    """Loopback + ``x-mesh-internal: 1`` + ``X-Agent-ID`` succeeds
    without a Bearer token. This is the dashboard cancel proxy path.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Internal caller creates a task on behalf of operator…
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "internal-create"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        # …and then cancels it the same way (no Bearer token).
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "ops decided no"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_task_via_dashboard_proxy_succeeds_in_production_auth_mode(
    v2_app_with_auth,
):
    """Regression for the kanban Cancel button. With auth tokens
    configured, a loopback + ``x-mesh-internal`` POST to
    ``/mesh/tasks/{id}/cancel`` (no Bearer) must NOT 401.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Seed via a Bearer-authed worker so the task exists with a
        # creator we can verify the cancel against.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "kanban-cancel-target"},
            headers={
                "X-Agent-ID": "scout",
                "Authorization": "Bearer tok_scout",
            },
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        # Dashboard proxies the cancel: loopback + x-mesh-internal,
        # NO Authorization header.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "user clicked Cancel"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code != 401, (
            "Dashboard cancel proxy must not require Bearer when caller "
            "is loopback + x-mesh-internal"
        )
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_public_caller_without_bearer_still_rejected(v2_app_with_auth):
    """The fix only exempts internal callers — a public caller
    without a Bearer token still gets 401.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "public-no-bearer"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_public_caller_with_valid_bearer_succeeds(v2_app_with_auth):
    """Sanity check: existing Bearer auth path still works."""
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "with-bearer"},
            headers={
                "X-Agent-ID": "scout",
                "Authorization": "Bearer tok_scout",
            },
        )
        assert r.status_code == 200, r.text


# ── GET /mesh/tasks/workflow/{root_task_id} (operator workflow awareness) ──


@pytest.mark.asyncio
async def test_workflow_snapshot_endpoint_operator_only(v2_app):
    """Non-operator callers get 403; the snapshot is operator-tier
    workflow awareness, not worker-readable."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        # Scout creates a kickoff task so the root exists.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "kickoff"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]

        # Worker tries to read — 403.
        r = await c.get(
            f"/mesh/tasks/workflow/{tid}",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_workflow_snapshot_endpoint_returns_snapshot_for_operator(v2_app):
    """Operator sees the chain rooted at the kickoff task."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        # Scout creates kickoff task.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "kickoff"},
            headers={"X-Agent-ID": "scout"},
        )
        root_id = r.json()["id"]
        # Scout creates a child of the kickoff.
        r = await c.post(
            "/mesh/tasks",
            json={
                "assignee": "analyst",
                "title": "next stage",
                "parent_task_id": root_id,
            },
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200, r.text

        # Operator reads the snapshot.
        r = await c.get(
            f"/mesh/tasks/workflow/{root_id}",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["root"] == root_id
        ids = [s["task_id"] for s in body["stages"]]
        assert root_id in ids
        assert body["summary"]["total"] == 2


@pytest.mark.asyncio
async def test_workflow_snapshot_endpoint_returns_404_for_missing_root(v2_app):
    """Unknown root id returns 404 so the operator can distinguish a
    typo from an empty chain."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.get(
            "/mesh/tasks/workflow/task_does_not_exist",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 404


# ── Bug 1 post-write verify tests ────────────────────────────────


def _patch_create_for_verify(tasks_store, fake_record):
    """Replace ``tasks_store.create`` so it returns a chosen record.

    The endpoint compares the record returned from ``store.create`` against
    the incoming request body. Injecting a divergent record here is the
    cleanest way to exercise the endpoint's verify branch — it lets the
    test pretend the store had a corruption bug and stored mistyped
    values, then asserts the verify catches it before the row reaches
    the agent's ``hand_off`` envelope path.
    """
    def fake_create(**_kwargs):
        return fake_record
    tasks_store.create = fake_create


@pytest.mark.asyncio
async def test_post_write_verify_500_on_null_record(v2_app):
    """If ``Tasks.create`` ever violates its contract and returns None the
    endpoint must 500 rather than leak null JSON to the agent."""
    app, _, _ = v2_app
    _patch_create_for_verify(app.tasks_store, fake_record=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "ghost"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 500, r.text
    assert "no record" in r.text.lower() or "contract" in r.text.lower()


@pytest.mark.asyncio
async def test_post_write_verify_500_on_assignee_mismatch(v2_app):
    """A stored row with a divergent ``assignee`` must trip the verify
    branch and surface the field name in the 500 detail."""
    app, _, _ = v2_app
    fake_row = {
        "id": "task_fake",
        "assignee": "wrong",
        "creator": "scout",
        "team_id": None,
        "parent_task_id": None,
        "status": "pending",
    }
    _patch_create_for_verify(app.tasks_store, fake_record=fake_row)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "mismatch"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 500, r.text
    assert "assignee" in r.text.lower()


@pytest.mark.asyncio
async def test_post_write_verify_500_on_status_mismatch(v2_app):
    """A stored row with a divergent ``status`` (e.g. ``failed`` instead
    of the expected ``pending``) must surface as a 500 with the field
    name in the detail."""
    app, _, _ = v2_app
    fake_row = {
        "id": "task_fake",
        "assignee": "analyst",
        "creator": "scout",
        "team_id": None,
        "parent_task_id": None,
        "status": "failed",
    }
    _patch_create_for_verify(app.tasks_store, fake_record=fake_row)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "bad status"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 500, r.text
    assert "status" in r.text.lower()


@pytest.mark.asyncio
async def test_post_write_verify_500_on_creator_mismatch(v2_app):
    """A stored row with a divergent ``creator`` (header-forge bug) must
    surface as a 500 with 'creator' in the detail."""
    app, _, _ = v2_app
    fake_row = {
        "id": "task_fake",
        "assignee": "analyst",
        "creator": "impersonator",
        "team_id": None,
        "parent_task_id": None,
        "status": "pending",
    }
    _patch_create_for_verify(app.tasks_store, fake_record=fake_row)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "creator drift"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 500, r.text
    assert "creator" in r.text.lower()


@pytest.mark.asyncio
async def test_assignee_whitespace_stripped(v2_app):
    """An assignee with surrounding whitespace must be normalized in the
    stored row so SQLite ``=`` lookups (``list_task_inbox``) match."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "  analyst  ", "title": "stripme"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["assignee"] == "analyst"


@pytest.mark.asyncio
async def test_assignee_whitespace_normalization_logged(v2_app, caplog):
    """The whitespace normalization must emit a WARNING-level log so
    operators see when stale prompts are emitting padded ids."""
    import logging

    app, _, _ = v2_app
    caplog.set_level(logging.WARNING)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "  analyst  ", "title": "logme"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 200, r.text
    assert any(
        "normalized assignee" in rec.getMessage().lower()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    ), [rec.getMessage() for rec in caplog.records]


# ── PR 2 (Work tab cutover) — mesh outcome endpoint ─────────────────


async def _create_and_complete(client, *, assignee: str = "analyst") -> str:
    """Helper — create a task, mark it done. Returns the task id."""
    r = await client.post(
        "/mesh/tasks",
        json={"assignee": assignee, "title": "completed work"},
        headers={"X-Agent-ID": "scout"},
    )
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    # Mark working, then done (transition rules require working → done).
    r = await client.post(
        f"/mesh/tasks/{tid}/status",
        json={"status": "working"},
        headers={"X-Agent-ID": assignee},
    )
    assert r.status_code == 200, r.text
    r = await client.post(
        f"/mesh/tasks/{tid}/status",
        json={"status": "done"},
        headers={"X-Agent-ID": assignee},
    )
    assert r.status_code == 200, r.text
    return tid


@pytest.mark.asyncio
async def test_outcome_endpoint_operator_accepts(v2_app):
    """Operator can rate a completed task with ``accepted`` (empty feedback OK)."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        tid = await _create_and_complete(c)
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={"outcome": "accepted"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["task"]["outcome"] == "accepted"


@pytest.mark.asyncio
async def test_outcome_endpoint_writes_track_record_event_as_operator_agent(v2_app):
    """The mesh ``/mesh/tasks/{id}/outcome`` path is agent-reachable (plan
    §8 #17's ``_require_operator_or_internal``) — its track-record write
    (§8 #18) must be tagged ``rater_kind="operator_agent"`` so the
    rating-trust rule can exclude it from autonomy scoring."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        tid = await _create_and_complete(c)
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={"outcome": "accepted"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
    events = app.track_record_store.recent_events("analyst")
    assert len(events) == 1
    event = events[0]
    assert event["source"] == "task_outcome"
    assert event["ref_id"] == tid
    assert event["outcome"] == "accepted"
    assert event["rater_kind"] == "operator_agent"
    # _create_and_complete's task creation doesn't set team_id explicitly
    # (a bare create call), so the recorded event carries whatever the
    # task row carries — None here, mirroring "assignee"'s team-agnostic
    # source of truth (the task row, not team membership).
    assert event["team_id"] is None
    counts = app.track_record_store.counts_for_agent("analyst")
    assert counts == {"task_outcome": {"accepted": 1}}
    # Rating-trust rule: an operator-agent-rated event is excluded from
    # the autonomy-safe view even though it's present in plain counts.
    autonomy_counts = app.track_record_store.counts_for_agent(
        "analyst", rater_kinds=("human", "system"),
    )
    assert autonomy_counts == {}


@pytest.mark.asyncio
async def test_outcome_endpoint_rework_spawns_followup(v2_app):
    """``rework`` outcome auto-spawns a follow-up task and surfaces its id."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        tid = await _create_and_complete(c)
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={
                "outcome": "rework",
                "feedback": "tighten the intro paragraph",
            },
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["task"]["outcome"] == "rework"
        assert "rework_task_id" in body
        assert body["rework_assignee"] == "analyst"


@pytest.mark.asyncio
async def test_outcome_endpoint_rejects_unknown_outcome(v2_app):
    """Unknown outcomes get 400, not 200."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        tid = await _create_and_complete(c)
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={"outcome": "loved_it"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 400
        assert "outcome must be one of" in r.text


@pytest.mark.asyncio
async def test_outcome_endpoint_requires_feedback_for_rework(v2_app):
    """rework / rejected require non-empty feedback (HTTP 400)."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        tid = await _create_and_complete(c)
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={"outcome": "rework", "feedback": ""},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 400
        assert "feedback is required" in r.text


@pytest.mark.asyncio
async def test_outcome_endpoint_returns_404_for_missing_task(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.post(
            "/mesh/tasks/task_does_not_exist/outcome",
            json={"outcome": "accepted"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_outcome_endpoint_non_terminal_returns_409(v2_app):
    """Tasks must be terminal before they can be rated."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "still working"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Task is pending — not terminal. Try to rate it.
        r = await c.post(
            f"/mesh/tasks/{tid}/outcome",
            json={"outcome": "accepted"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 409


# ── H5: endpoint-level caps ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_task_pending_cap_returns_400(tmp_path, monkeypatch):
    """Past the per-assignee pending cap, ``POST /mesh/tasks`` returns
    400 (resource cap) — normal volume under the cap succeeds."""
    monkeypatch.setenv("OPENLEGION_MAX_PENDING_TASKS_PER_AGENT", "3")
    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    perms_map = {
        "alpha": {"can_message": ["*"]},
        "bravo": {"can_message": ["*"]},
    }
    app, bb = _build_app(
        tmp_path, server_module, perms_map=perms_map,
        agents={"alpha": "http://alpha:8400", "bravo": "http://bravo:8400"},
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            # 3 pending tasks for bravo — all allowed.
            for i in range(3):
                r = await c.post(
                    "/mesh/tasks",
                    json={"assignee": "bravo", "title": f"t{i}"},
                    headers={"X-Agent-ID": "alpha"},
                )
                assert r.status_code == 200, r.text
            # 4th over the cap → 400.
            r = await c.post(
                "/mesh/tasks",
                json={"assignee": "bravo", "title": "overflow"},
                headers={"X-Agent-ID": "alpha"},
            )
            assert r.status_code == 400, r.text
            assert "pending" in r.json().get("detail", "").lower()
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        monkeypatch.delenv("OPENLEGION_MAX_PENDING_TASKS_PER_AGENT", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_create_task_rate_limit_allows_normal_volume(v2_app):
    """The task_create rate bucket is generous (~300/min) — a normal
    burst of a dozen creates never trips it."""
    app, _, _ = v2_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        for i in range(12):
            r = await c.post(
                "/mesh/tasks",
                json={"assignee": "analyst", "title": f"burst{i}"},
                headers={"X-Agent-ID": "scout"},
            )
            assert r.status_code == 200, r.text


# ── B5: GET /mesh/tasks/{task_id}/run ─────────────────────────────


@pytest.mark.asyncio
async def test_task_run_is_operator_only(v2_app):
    app, _, _ = v2_app
    rec = app.tasks_store.create(
        creator="operator", assignee="analyst", title="deep work",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            f"/mesh/tasks/{rec['id']}/run", headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_task_run_unknown_id_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/tasks/task_doesnotexist/run",
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_task_run_returns_record_events_and_execution(v2_app):
    app, _, _ = v2_app
    rec = app.tasks_store.create(
        creator="operator", assignee="analyst", title="deep audit",
        description="full brief", thinking="high",
    )
    app.tasks_store.update_status(rec["id"], "working", actor="analyst")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            f"/mesh/tasks/{rec['id']}/run", headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["task"]["id"] == rec["id"]
    assert body["task"]["thinking"] == "high"
    assert body["task"]["status"] == "working"
    # No trace store wired in this fixture — execution summary is zeroed
    # but present and well-formed.
    assert body["execution"]["llm_calls"] == 0
    assert body["execution"]["tokens_used"] == 0
    # Timeline includes the created + status-change events.
    kinds = [e["event_kind"] for e in body["events"]]
    assert "created" in kinds


@pytest.mark.asyncio
async def test_task_run_aggregates_traces_in_window(tmp_path, monkeypatch):
    """LLM-call traces for the assignee inside the task window roll up
    into llm_calls / tokens_used / models."""
    from src.host.traces import TraceStore

    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"analyst": "http://analyst:8400"})
    traces = TraceStore(str(tmp_path / "traces.db"))
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        trace_store=traces,
    )
    try:
        rec = app.tasks_store.create(
            creator="operator", assignee="analyst", title="deep audit",
        )
        traces.record(
            trace_id="tr1", source="mesh.api_proxy", agent="analyst",
            event_type="llm_call", detail="llm/chat", duration_ms=900,
            status="ok", meta={"model": "anthropic/claude-x", "tokens_used": 1200},
        )
        traces.record(
            trace_id="tr1", source="mesh.api_proxy", agent="analyst",
            event_type="llm_call", detail="llm/chat", duration_ms=400,
            status="error", error="rate limited",
            meta={"model": "anthropic/claude-x", "tokens_used": 300},
        )
        # Streamed call: the proxy records an ``llm_stream`` row at
        # stream start AND an ``llm_call`` row (tokens, streaming=True)
        # at completion. Only the completion row may count — counting
        # both kinds doubled llm_calls for streamed traffic.
        traces.record(
            trace_id="tr3", source="mesh.api_proxy", agent="analyst",
            event_type="llm_stream", detail="llm/chat",
            meta={"model": "anthropic/claude-x"},
        )
        traces.record(
            trace_id="tr3", source="mesh.api_proxy", agent="analyst",
            event_type="llm_call", detail="llm/chat", duration_ms=700,
            status="ok",
            meta={
                "model": "anthropic/claude-x", "tokens_used": 500,
                "streaming": True,
            },
        )
        # Different agent in the same window — must NOT count.
        traces.record(
            trace_id="tr2", source="mesh.api_proxy", agent="scout",
            event_type="llm_call", detail="llm/chat",
            meta={"model": "other", "tokens_used": 999},
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get(
                f"/mesh/tasks/{rec['id']}/run",
                headers={"X-Agent-ID": "operator"},
            )
        assert r.status_code == 200
        execution = r.json()["execution"]
        assert execution["llm_calls"] == 3
        assert execution["tokens_used"] == 2000
        assert execution["models"] == ["anthropic/claude-x"]
        assert len(execution["trace_errors"]) == 1
        assert "rate limited" in execution["trace_errors"][0]["error"]
    finally:
        blackboard.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


# ── Session observability (Phase 1): X-Trace-Id seed → stamped task row ──
#
# Store-level tests (test_orchestration.py) already verify ``Tasks.create``
# stamps ``trace_id`` when the contextvar is pre-set. These tests close the
# end-to-end gap: the HTTP endpoint must seed ``current_trace_id`` from the
# inbound ``X-Trace-Id`` header so the row the store writes carries the
# originating per-turn trace. If that seed regresses, the row silently gets
# a NULL trace_id (no crash) and sessions can no longer be JOINed.


@pytest.mark.asyncio
async def test_create_task_stamps_trace_id_from_header(v2_app):
    """``POST /mesh/tasks`` with ``X-Trace-Id`` → the created task row's
    ``trace_id`` equals the header value (endpoint → contextvar → store
    → row chain)."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "traced handoff"},
            headers={"X-Agent-ID": "scout", "X-Trace-Id": "tr_test00000001"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["trace_id"] == "tr_test00000001"


@pytest.mark.asyncio
async def test_create_task_sequential_trace_attribution(v2_app):
    """Sequential attribution: a ``POST /mesh/tasks`` WITH ``X-Trace-Id``
    followed by one WITHOUT stamp the first's trace then NULL. No manual
    contextvar clearing — the second row's NULL is the request-scoping
    contract end-to-end (the old negative test had to ``set(None)`` by hand,
    which masked whatever the real per-request behavior was).

    Honesty note (mutation-verified): the mesh app's Starlette
    ``BaseHTTPMiddleware`` isolates contextvars per request, so even a
    conditional set + no reset could not actually bleed across requests here
    — reverting the reset does NOT fail this test. The unconditional set +
    token reset in the endpoint is defense-in-depth (explicit request-scoping
    that does not depend on the middleware), and this test guards the
    attribution contract rather than trapping a raw leak."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r1 = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "traced handoff"},
            headers={"X-Agent-ID": "scout", "X-Trace-Id": "tr_first0000001"},
        )
        assert r1.status_code == 200, r1.text
        assert r1.json()["trace_id"] == "tr_first0000001"
        r2 = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "untraced handoff"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r2.status_code == 200, r2.text
    assert r2.json()["trace_id"] is None


@pytest.mark.asyncio
async def test_retry_clone_inherits_original_trace_id(v2_app):
    """``POST /mesh/tasks/{id}/retry`` clones the failed task into a fresh
    pending one that continues the SAME session — the clone must inherit
    the original's ``trace_id`` even when the retry request itself carries
    no header (the endpoint seeds from ``original.get('trace_id')``)."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Create the original WITH a trace, then drive it to failed.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "retry target"},
            headers={"X-Agent-ID": "scout", "X-Trace-Id": "tr_retry0000001"},
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        assert r.json()["trace_id"] == "tr_retry0000001"
        for status in ("working", "failed"):
            r = await c.post(
                f"/mesh/tasks/{tid}/status",
                json={"status": status},
                headers={"X-Agent-ID": "analyst"},
            )
            assert r.status_code == 200, r.text
        # Retry as operator, deliberately WITHOUT an X-Trace-Id header —
        # the clone must still inherit the original's trace.
        r = await c.post(
            f"/mesh/tasks/{tid}/retry",
            json={},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        clone = r.json()["clone"]
        assert clone["id"] != tid
        assert clone["trace_id"] == "tr_retry0000001"


# ── Session observability (Phase 4): agent-side trace ingest endpoint ──
#
# Agents cannot write traces.db (host-only by container isolation), so they
# POST tool_call/handoff/iteration events to ``/mesh/traces`` and the mesh
# records them under the inbound ``X-Trace-Id`` — mirroring the llm_call
# ingest in the API proxy. These tests pin: (1) a record lands with
# source="agent" under the header trace; (2) no header → accepted no-op,
# nothing stored (no orphan rows with a NULL/empty trace); (3) under auth the
# recorded ``agent`` comes from the verified token, not the spoofable body.


def _build_traces_app(tmp_path, server_module, *, auth_tokens=None):
    """Mesh app wired with a TraceStore, for ``/mesh/traces`` ingest tests."""
    from src.host.traces import TraceStore

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"analyst": "http://analyst:8400"})
    traces = TraceStore(str(tmp_path / "traces.db"))
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        trace_store=traces,
        auth_tokens=auth_tokens,
    )
    return app, blackboard, traces


@pytest.mark.asyncio
async def test_traces_ingest_records_under_header_trace_id(tmp_path, monkeypatch):
    """``POST /mesh/traces`` with ``X-Trace-Id`` records an agent-sourced
    event the session reader can join by trace_id. Redaction happens inside
    ``TraceStore.record`` — a URL with embedded credentials is stripped."""
    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    app, bb, traces = _build_traces_app(tmp_path, server_module)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/traces",
                json={
                    "agent_id": "analyst",
                    "event_type": "tool_call",
                    "detail": "http_request via https://u:p4ss@api.example.com/x",
                    "duration_ms": 42,
                    "status": "ok",
                    "meta": {"foo": "bar"},
                },
                headers={"X-Agent-ID": "analyst", "X-Trace-Id": "tr_agent0000001"},
            )
        assert r.status_code == 200, r.text
        assert r.json()["recorded"] is True
        rows = traces.get_trace("tr_agent0000001")
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == "agent"
        assert row["agent"] == "analyst"
        assert row["event_type"] == "tool_call"
        assert row["duration_ms"] == 42
        # H16: embedded credentials redacted at storage.
        assert "p4ss" not in row["detail"]
        assert "api.example.com" in row["detail"]
    finally:
        bb.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_traces_ingest_without_header_noops(tmp_path, monkeypatch):
    """No ``X-Trace-Id`` → accepted no-op: nothing is stored. There is no
    trace to correlate, and we never want an orphan row with an empty trace."""
    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    app, bb, traces = _build_traces_app(tmp_path, server_module)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/traces",
                json={"agent_id": "analyst", "event_type": "iteration"},
                headers={"X-Agent-ID": "analyst"},
            )
        assert r.status_code == 200, r.text
        assert r.json()["recorded"] is False
        assert traces.list_recent(limit=10) == []
    finally:
        bb.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_traces_ingest_coerces_bad_duration(tmp_path, monkeypatch):
    """A malformed/negative ``duration_ms`` must not drop the whole trace —
    it is coerced (bad → 0) and clamped (negative → 0) rather than raising
    into the no-op path."""
    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    app, bb, traces = _build_traces_app(tmp_path, server_module)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r_bad = await c.post(
                "/mesh/traces",
                json={"agent_id": "a", "event_type": "iteration", "duration_ms": "oops"},
                headers={"X-Agent-ID": "a", "X-Trace-Id": "tr_dur00000001"},
            )
            r_neg = await c.post(
                "/mesh/traces",
                json={"agent_id": "a", "event_type": "iteration", "duration_ms": -5},
                headers={"X-Agent-ID": "a", "X-Trace-Id": "tr_dur00000002"},
            )
        assert r_bad.json()["recorded"] is True
        assert r_neg.json()["recorded"] is True
        assert traces.get_trace("tr_dur00000001")[0]["duration_ms"] == 0
        assert traces.get_trace("tr_dur00000002")[0]["duration_ms"] == 0
    finally:
        bb.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_traces_ingest_agent_id_from_token_not_body(tmp_path, monkeypatch):
    """Under auth the recorded ``agent`` is the verified token identity — a
    spoofed body ``agent_id`` is ignored (``_resolve_agent_id`` derives it
    from the Bearer token)."""
    server_module = _reload_server(monkeypatch, tasks_db=str(tmp_path / "tasks.db"))
    app, bb, traces = _build_traces_app(
        tmp_path, server_module,
        auth_tokens={"analyst": "tok_analyst", "scout": "tok_scout"},
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/traces",
                json={"agent_id": "scout", "event_type": "tool_call", "detail": "x"},
                headers={
                    "X-Agent-ID": "scout",  # spoof attempt
                    "Authorization": "Bearer tok_analyst",
                    "X-Trace-Id": "tr_authagent001",
                },
            )
        assert r.status_code == 200, r.text
        rows = traces.get_trace("tr_authagent001")
        assert len(rows) == 1
        assert rows[0]["agent"] == "analyst"  # token wins over body/header
    finally:
        bb.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)
