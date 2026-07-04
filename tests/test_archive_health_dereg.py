"""Archive must deregister the agent from the health monitor.

Regression guard: the archive endpoint intentionally stops an agent's
container, but historically it did NOT call ``health_monitor.unregister``
(only the delete path did). The health poller would then see the
intentional stop as a failure and auto-restart the container within
~90s, fighting the archive. This mirrors the deregistration the delete
path already performs, and is a prerequisite for agent hibernation.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _build_app(tmp_path, monkeypatch, *, agent="scout"):
    import src.cli.config as cli_config
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    # Stub config I/O so the endpoint doesn't touch real config files.
    monkeypatch.setattr(cli_config, "_load_config", lambda: {"agents": {agent: {}}})
    monkeypatch.setattr(cli_config, "_archive_agent", lambda agent_id: None)

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {})
    health = MagicMock()
    container = MagicMock()

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        auth_tokens={"operator": "tok-op"},
        health_monitor=health,
        container_manager=container,
    )
    return app, bb, health, container


def test_archive_deregisters_from_health_monitor(tmp_path, monkeypatch):
    app, bb, health, container = _build_app(tmp_path, monkeypatch)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/agents/scout/archive",
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"archived": True, "agent_id": "scout"}
        # The core guarantee: health monitoring is torn down so the poller
        # can't auto-restart the intentionally-stopped container.
        health.unregister.assert_called_once_with("scout")
        container.stop_agent.assert_called_once_with("scout")
    finally:
        bb.close()


def test_archive_survives_health_unregister_failure(tmp_path, monkeypatch):
    """A health-deregister failure must not break archive (best-effort)."""
    app, bb, health, container = _build_app(tmp_path, monkeypatch)
    health.unregister.side_effect = RuntimeError("boom")
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/agents/scout/archive",
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 200, resp.text
        # Still attempted the deregister and still stopped the container.
        health.unregister.assert_called_once_with("scout")
        container.stop_agent.assert_called_once_with("scout")
    finally:
        bb.close()
