"""Agent personnel-file export (read-only, operator-only).

``GET /mesh/agents/{id}/export`` bundles the durable host-side identity —
config + permission ACL + cron/heartbeat schedule — plus best-effort workspace
markdown from the running container. Operator-gated; a stopped agent still
yields the host-side bundle (workspace ``None``).
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _build_app(tmp_path, monkeypatch, *, transport=None, cron_scheduler=None, auth_tokens=None):
    import src.cli.config as cli_config
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    monkeypatch.setattr(
        cli_config,
        "_load_config",
        lambda: {"agents": {"scout": {"role": "researcher", "model": "openai/gpt-4o-mini"}}},
    )
    # Isolate the track-record ledger DB into tmp_path — otherwise every
    # test file that doesn't override it shares one on-disk
    # data/track_record.db and count assertions get flaky.
    monkeypatch.setenv("OPENLEGION_TRACK_RECORD_DB", str(tmp_path / "track_record.db"))

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {})

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        auth_tokens=auth_tokens or {"operator": "tok-op"},
        cron_scheduler=cron_scheduler,
        transport=transport,
    )
    return app, bb


_OP = {"Authorization": "Bearer tok-op"}


def test_export_bundles_config_permissions_cron(tmp_path, monkeypatch):
    cron = MagicMock()
    cron.list_jobs.return_value = [
        {"id": "j1", "agent": "scout", "schedule": "every 15m", "heartbeat": True},
        {"id": "j2", "agent": "other", "schedule": "0 9 * * *", "heartbeat": False},
    ]
    app, bb = _build_app(tmp_path, monkeypatch, cron_scheduler=cron)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["bundle_version"] == 1
        assert body["agent_id"] == "scout"
        assert body["config"]["role"] == "researcher"
        # cron filtered to THIS agent only
        assert [j["id"] for j in body["cron_jobs"]] == ["j1"]
        # permissions serialized (deny-all default is still a dict)
        assert isinstance(body["permissions"], dict)
        # no transport wired → workspace omitted, host-side bundle intact
        assert body["workspace"] is None
        # no standing goals set → null, matching the optional-section style
        assert body["goals"] is None
        # no track record events → empty (but present) section (§8 #18)
        assert body["track_record"] == {"counts": {}, "recent": []}
    finally:
        bb.close()


def test_export_includes_standing_goals_record(tmp_path, monkeypatch):
    """Ratified #7 / C.3-b: the personnel file carries the Team-store goals."""
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.teams_store.set_agent_goals(
            "scout", ["Find 10 qualified leads per week."], set_by="operator",
        )
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        assert resp.status_code == 200, resp.text
        goals = resp.json()["goals"]
        assert goals["goals"] == ["Find 10 qualified leads per week."]
        assert goals["set_by"] == "operator"
        assert goals["updated_at"]
    finally:
        bb.close()


def test_export_gathers_workspace_from_running_agent(tmp_path, monkeypatch):
    transport = MagicMock()

    async def _request(agent_id, method, path, **kwargs):
        if path == "/workspace":
            return {"files": [{"name": "SOUL.md"}, {"name": "MEMORY.md"}]}
        if path == "/workspace/SOUL.md":
            return {"filename": "SOUL.md", "content": "# soul\nbe helpful"}
        if path == "/workspace/MEMORY.md":
            return {"filename": "MEMORY.md", "content": "# memory\nlearned X"}
        return {}

    transport.request = AsyncMock(side_effect=_request)
    app, bb = _build_app(tmp_path, monkeypatch, transport=transport)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        assert resp.status_code == 200, resp.text
        ws = resp.json()["workspace"]
        assert ws == {"SOUL.md": "# soul\nbe helpful", "MEMORY.md": "# memory\nlearned X"}
    finally:
        bb.close()


def test_export_survives_unreachable_agent(tmp_path, monkeypatch):
    transport = MagicMock()
    transport.request = AsyncMock(side_effect=RuntimeError("connection refused"))
    app, bb = _build_app(tmp_path, monkeypatch, transport=transport)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        # Host-side bundle still returns; workspace best-effort → None.
        assert resp.status_code == 200, resp.text
        assert resp.json()["workspace"] is None
        assert resp.json()["config"]["role"] == "researcher"
    finally:
        bb.close()


def test_export_unknown_agent_404(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).get("/mesh/agents/ghost/export", headers=_OP)
        assert resp.status_code == 404, resp.text
    finally:
        bb.close()


def test_export_requires_operator(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        # A worker bearer (not the operator) is rejected.
        resp = TestClient(app).get(
            "/mesh/agents/scout/export",
            headers={"Authorization": "Bearer tok-worker"},
        )
        assert resp.status_code in (401, 403), resp.text
    finally:
        bb.close()


# =============================================================================
# Durable track record (plan §8 #18) — Personnel-File section
# =============================================================================


def test_export_includes_track_record_counts_and_recent(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.track_record_store.record(
            source="task_outcome", ref_id="task_1", outcome="accepted",
            rater_kind="human", agent_id="scout", rated_by="operator",
        )
        app.track_record_store.record(
            source="task_outcome", ref_id="task_2", outcome="rework",
            rater_kind="operator_agent", agent_id="scout", rated_by="operator",
        )
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        assert resp.status_code == 200, resp.text
        tr = resp.json()["track_record"]
        assert tr["counts"] == {"task_outcome": {"accepted": 1, "rework": 1}}
        assert len(tr["recent"]) == 2
        assert tr["recent"][0]["ref_id"] == "task_2"  # newest first
    finally:
        bb.close()


def test_export_track_record_absent_when_read_fails(tmp_path, monkeypatch):
    """Bundle build must not fail when the store is unreachable — the
    ``track_record`` section degrades to None (mirrors ``workspace``)."""
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(
            app.track_record_store, "counts_for_agent",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")),
        )
        resp = TestClient(app).get("/mesh/agents/scout/export", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert resp.json()["track_record"] is None
    finally:
        bb.close()


# =============================================================================
# GET /mesh/agents/{id}/track-record (plan §8 #18 read surface)
# =============================================================================


_SCOUT = {"Authorization": "Bearer tok-scout"}


def test_track_record_endpoint_self_read_allowed(tmp_path, monkeypatch):
    app, bb = _build_app(
        tmp_path, monkeypatch, auth_tokens={"operator": "tok-op", "scout": "tok-scout"},
    )
    try:
        app.track_record_store.record(
            source="task_outcome", ref_id="task_1", outcome="accepted",
            rater_kind="human", agent_id="scout", rated_by="operator",
        )
        resp = TestClient(app).get("/mesh/agents/scout/track-record", headers=_SCOUT)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["agent_id"] == "scout"
        assert body["counts"] == {"task_outcome": {"accepted": 1}}
        assert len(body["recent"]) == 1
    finally:
        bb.close()


def test_track_record_endpoint_other_agent_403(tmp_path, monkeypatch):
    app, bb = _build_app(
        tmp_path, monkeypatch,
        auth_tokens={"operator": "tok-op", "scout": "tok-scout", "writer": "tok-writer"},
    )
    try:
        resp = TestClient(app).get(
            "/mesh/agents/scout/track-record",
            headers={"Authorization": "Bearer tok-writer"},
        )
        assert resp.status_code == 403, resp.text
    finally:
        bb.close()


def test_track_record_endpoint_operator_allowed(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/track-record", headers=_OP)
        assert resp.status_code == 200, resp.text
    finally:
        bb.close()


def test_track_record_endpoint_unknown_agent_404(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).get("/mesh/agents/ghost/track-record", headers=_OP)
        assert resp.status_code == 404, resp.text
    finally:
        bb.close()


def test_track_record_endpoint_autonomy_counts_exclude_operator_agent(tmp_path, monkeypatch):
    """Pins the rating-trust rule (plan §8 #18): an operator-agent-rated
    event shows up in ``counts`` but never in ``autonomy_counts``."""
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.track_record_store.record(
            source="task_outcome", ref_id="task_1", outcome="accepted",
            rater_kind="human", agent_id="scout", rated_by="operator",
        )
        app.track_record_store.record(
            source="task_outcome", ref_id="task_2", outcome="accepted",
            rater_kind="operator_agent", agent_id="scout", rated_by="operator",
        )
        resp = TestClient(app).get("/mesh/agents/scout/track-record", headers=_OP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["counts"] == {"task_outcome": {"accepted": 2}}
        assert body["autonomy_counts"] == {"task_outcome": {"accepted": 1}}
    finally:
        bb.close()
