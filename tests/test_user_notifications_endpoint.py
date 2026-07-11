"""End-to-end tests for the agent→user notification observation log (Bug 1).

Drives the real mesh app:
  - an agent's ``POST /mesh/notify`` logs a row,
  - the operator reads it back via ``GET /mesh/user-notifications``,
  - non-operator workers are denied (operator-only PULL surface),
  - a logging failure does not break the notify response.

Plus the metrics-layer ``inbox_stale_count`` surface (Bug 6).
"""

from __future__ import annotations

import importlib

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


def _reload_server(monkeypatch, tmp_path):
    """Reload ``src.host.server`` with DB paths pinned to ``tmp_path``."""
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    monkeypatch.setenv(
        "OPENLEGION_USER_NOTIFICATIONS_DB", str(tmp_path / "user_notifs.db"),
    )
    # Isolate the held-actions store — the notify-hold tests below store
    # rows, and without this pin every mesh app in one pytest process
    # shares a single cwd data/pending_actions.db across test files.
    monkeypatch.setenv(
        "OPENLEGION_PENDING_ACTIONS_DB", str(tmp_path / "pending_actions.db"),
    )
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _build_app(tmp_path, server_module, *, notify_fn=None):
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid in ("operator", "scout", "analyst"):
        permissions.permissions[aid] = AgentPermissions(agent_id=aid)
    router = MessageRouter(permissions, {})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        notify_fn=notify_fn,
    )
    return app, blackboard


@pytest.fixture
def app_ctx(tmp_path, monkeypatch):
    captured: list[tuple[str, str]] = []

    async def _notify(agent_id: str, message: str) -> None:
        captured.append((agent_id, message))

    server_module = _reload_server(monkeypatch, tmp_path)
    app, bb = _build_app(tmp_path, server_module, notify_fn=_notify)
    yield app, server_module, captured
    bb.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_USER_NOTIFICATIONS_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_PENDING_ACTIONS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_notify_logs_and_operator_reads_back(app_ctx):
    """Agent notify_user("foo") → operator reads an entry containing foo."""
    app, _, captured = app_ctx
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        notify = await c.post(
            "/mesh/notify",
            json={"agent_id": "scout", "message": "stage 2 blocked on foo creds"},
            headers={"X-Agent-ID": "scout"},
        )
        assert notify.status_code == 200, notify.text
        assert notify.json() == {"sent": True}
        # Human channel still received it.
        assert captured == [("scout", "stage 2 blocked on foo creds")]

        read = await c.get(
            "/mesh/user-notifications",
            headers={"X-Agent-ID": "operator"},
        )
    assert read.status_code == 200, read.text
    notifs = read.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["from"] == "scout"
    assert "foo" in notifs[0]["message"]


@pytest.mark.asyncio
async def test_user_notifications_is_operator_gated(app_ctx):
    """A non-operator agent is denied the read surface (403)."""
    app, _, _ = app_ctx
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        await c.post(
            "/mesh/notify",
            json={"agent_id": "scout", "message": "secret peer report"},
            headers={"X-Agent-ID": "scout"},
        )
        denied = await c.get(
            "/mesh/user-notifications",
            headers={"X-Agent-ID": "analyst"},
        )
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_notify_succeeds_when_logging_fails(app_ctx):
    """A logging failure must NOT break the notify response (best-effort)."""
    app, _, captured = app_ctx

    def _boom(*_a, **_kw):
        raise RuntimeError("disk full")

    # Sabotage the log's record() — notify must still return sent: True.
    app.user_notification_log.record = _boom
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        resp = await c.post(
            "/mesh/notify",
            json={"agent_id": "scout", "message": "still delivers"},
            headers={"X-Agent-ID": "scout"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"sent": True}
    assert captured == [("scout", "still delivers")]


@pytest.mark.asyncio
async def test_metrics_surface_inbox_stale_count(app_ctx):
    """An aged operator-assigned task surfaces as metrics.inbox_stale_count."""
    import time

    app, _, _ = app_ctx
    rec = app.tasks_store.create(creator="scout", assignee="operator", title="triage me")
    # Age it past 24h so count_stale_since picks it up.
    with app.tasks_store._conn() as conn:
        conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (time.time() - 25 * 3600, rec["id"]),
        )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        metrics = await c.get(
            "/mesh/system/metrics",
            headers={"X-Agent-ID": "operator"},
        )
    assert metrics.status_code == 200, metrics.text
    body = metrics.json()
    assert "inbox_stale_count" in body
    assert body["inbox_stale_count"] == 1
    # The per-agent stale surface still excludes operator.
    assert "operator" not in body.get("stale_tasks_24h_count", {})


# ── Action-tier policy gate on /mesh/notify (plan §8 #17) ────────────


def _human_headers(agent_id: str = "operator") -> dict:
    from src.shared.types import MessageOrigin

    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    return {"X-Agent-ID": agent_id, "X-Origin": origin.to_header_value()}


def _agent_headers(agent_id: str = "operator") -> dict:
    from src.shared.types import MessageOrigin

    origin = MessageOrigin(kind="agent", channel="", user="")
    return {"X-Agent-ID": agent_id, "X-Origin": origin.to_header_value()}


class _RecordingBus:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def emit(self, event_type, agent="", data=None):
        self.events.append((event_type, agent, dict(data or {})))


def _build_hold_ctx(tmp_path, monkeypatch, policy_yaml: str):
    """Reload the server with a policy.yaml applied, build an app with a
    recording notify_fn + event bus. Returns (app, captured, bus)."""
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(policy_yaml)
    monkeypatch.setenv("OPENLEGION_POLICY_CONFIG", str(policy_path))
    server_module = _reload_server(monkeypatch, tmp_path)

    captured: list[tuple[str, str]] = []

    async def _notify(agent_id: str, message: str) -> None:
        captured.append((agent_id, message))

    bus = _RecordingBus()
    app, bb = _build_app(tmp_path, server_module, notify_fn=_notify)
    app.event_bus = bus
    # Re-wire the pending_actions store's bus (created before our fixture's
    # bus existed) so the pending_action_* events used below are captured.
    app.pending_actions.set_event_bus(bus)
    return app, bb, captured, bus


@pytest.mark.asyncio
async def test_notify_default_writes_one_policy_audit_row(app_ctx):
    """Default (no yaml) external_visible decision is allow_audit: the
    notification still goes out AND one policy_decision audit row lands,
    on top of the existing observation-log write."""
    app, _, captured = app_ctx
    blackboard = _find_closure_var(app, "blackboard")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        resp = await c.post(
            "/mesh/notify",
            json={"agent_id": "scout", "message": "fyi only"},
            headers={"X-Agent-ID": "scout"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"sent": True}
    assert captured == [("scout", "fyi only")]
    entries = blackboard.get_audit_log(action="policy_decision")["entries"]
    assert len(entries) == 1
    assert entries[0]["target"] == "scout"
    assert entries[0]["field"] == "notify_user"


@pytest.mark.asyncio
async def test_notify_hold_not_delivered_and_queued(tmp_path, monkeypatch):
    app, bb, captured, bus = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            resp = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "needs approval"},
                headers={"X-Agent-ID": "scout"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sent"] is False
        assert body["queued_for_approval"] is True
        assert "change_id" in body
        assert captured == []  # NOT delivered
        created = [e for e in bus.events if e[0] == "pending_action_created"]
        assert len(created) == 1
        assert created[0][2]["action_kind"] == "notify_user"
        assert created[0][2]["tier"] == "external_visible"
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_notify_hold_confirm_delivers_exactly_once(tmp_path, monkeypatch):
    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            propose = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "please approve"},
                headers={"X-Agent-ID": "scout"},
            )
            nonce = propose.json()["change_id"]
            digest = propose.json()["payload_digest"]
            confirm1 = await c.post(
                "/mesh/config/confirm",
                json={"change_id": nonce, "payload_digest": digest},
                headers=_human_headers("operator"),
            )
            assert confirm1.status_code == 200, confirm1.text
            assert confirm1.json()["sent"] is True
            assert captured == [("scout", "please approve")]
            # Single-use nonce — a second confirm must not re-deliver.
            confirm2 = await c.post(
                "/mesh/config/confirm",
                json={"change_id": nonce, "payload_digest": digest},
                headers=_human_headers("operator"),
            )
        assert confirm2.status_code == 400
        assert captured == [("scout", "please approve")]
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_notify_hold_cancel_never_delivers(tmp_path, monkeypatch):
    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            propose = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "cancel me"},
                headers={"X-Agent-ID": "scout"},
            )
            nonce = propose.json()["change_id"]
            cancel = await c.post(
                f"/mesh/pending/{nonce}/cancel",
                headers={"X-Agent-ID": "operator"},
            )
        assert cancel.status_code == 200
        assert captured == []
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_notify_hold_agent_origin_confirm_403(tmp_path, monkeypatch):
    """Human-origin invariant extended to notify holds: an agent-origin
    confirm attempt is refused, and the notification is never delivered."""
    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            propose = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "sneaky"},
                headers={"X-Agent-ID": "scout"},
            )
            nonce = propose.json()["change_id"]
            digest = propose.json()["payload_digest"]
            confirm = await c.post(
                "/mesh/config/confirm",
                json={"change_id": nonce, "payload_digest": digest},
                headers=_agent_headers("operator"),
            )
        assert confirm.status_code == 403
        assert captured == []
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_notify_hold_queue_full_rejected_fail_closed(tmp_path, monkeypatch):
    """With the pending store at _MAX_PENDING, a hold-decision notify is
    REFUSED (429) — never delivered, never stored, and no existing row
    is evicted (fail-closed; eviction is a delete-producer behavior
    only)."""
    from src.host.server import _MAX_PENDING

    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    pa = app.pending_actions
    try:
        # The store is tmp_path-isolated (OPENLEGION_PENDING_ACTIONS_DB
        # pin in ``_reload_server``) — counts are exact, no post-purge.
        for i in range(_MAX_PENDING):
            pa.store(
                nonce=f"pre-notify-{i}", actor="operator", target_kind="agent",
                target_id="alpha", action_kind="delete", payload={},
                origin_kind="human",
            )
        assert len(pa.list_pending()) == _MAX_PENDING
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            resp = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "queue is full"},
                headers={"X-Agent-ID": "scout"},
            )
        assert resp.status_code == 429
        assert "Approval queue full" in resp.text
        assert captured == []  # not delivered
        assert len(pa.list_pending()) == _MAX_PENDING  # not stored
        # No eviction: every prefilled row survives.
        for i in range(_MAX_PENDING):
            assert pa.peek(f"pre-notify-{i}") is not None
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_notify_deny_policy_returns_403_and_records_denial(tmp_path, monkeypatch):
    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: deny\n",
    )
    try:
        # ``pending_actions`` uses a fixed on-disk path (not tmp_path-scoped),
        # shared across the whole test session — compare a before/after
        # count rather than assuming a pristine store.
        before = len(app.pending_actions.list_pending())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            resp = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "never sent"},
                headers={"X-Agent-ID": "scout"},
            )
        assert resp.status_code == 403
        assert captured == []
        assert len(app.pending_actions.list_pending()) == before
    finally:
        bb.close()


def _find_closure_var(app, name: str):
    """Locate a variable captured in some endpoint's closure, searching
    one level deep through any closed-over helper function too (e.g.
    ``_check_rate_limit``, which is itself a free variable of every
    endpoint that calls it, and which in turn closes over ``_rate_ts``/
    ``_RATE_LIMITS`` — locals inside ``create_mesh_app``, never exposed
    on ``app``)."""

    def _scan(func, depth: int) -> object | None:
        names = getattr(getattr(func, "__code__", None), "co_freevars", ())
        closure = getattr(func, "__closure__", None) or ()
        for n, cell in zip(names, closure):
            if n == name:
                return cell.cell_contents
        if depth > 0:
            for _n, cell in zip(names, closure):
                val = cell.cell_contents
                if callable(val) and getattr(val, "__closure__", None):
                    found = _scan(val, depth - 1)
                    if found is not None:
                        return found
        return None

    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        found = _scan(endpoint, depth=1)
        if found is not None:
            return found
    raise AssertionError(f"could not locate closure var {name!r} on app")


@pytest.mark.asyncio
async def test_notify_rate_limit_enforced_before_policy(tmp_path, monkeypatch):
    """Exhausting the notify rate bucket 429s even under a hold policy —
    proving rate-limit runs BEFORE policy.evaluate (gate order: permission
    check [none exists here] -> rate limit -> policy -> act)."""
    app, bb, captured, _ = _build_hold_ctx(
        tmp_path, monkeypatch, "version: 1\ntiers:\n  external_visible: hold\n",
    )
    try:
        rate_ts = _find_closure_var(app, "_rate_ts")
        rate_limits = _find_closure_var(app, "_RATE_LIMITS")
        limit, _window = rate_limits["notify"]
        import time as _time

        before = len(app.pending_actions.list_pending())
        rate_ts["notify:scout"].extend([_time.time()] * limit)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            resp = await c.post(
                "/mesh/notify",
                json={"agent_id": "scout", "message": "should be rate limited"},
                headers={"X-Agent-ID": "scout"},
            )
        assert resp.status_code == 429
        assert captured == []
        # Never reached the policy/hold branch at all.
        assert len(app.pending_actions.list_pending()) == before
    finally:
        bb.close()
