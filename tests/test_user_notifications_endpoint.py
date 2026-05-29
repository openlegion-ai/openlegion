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
