"""Tests for LifecycleStore + the internal lifecycle_event endpoint + the
session reader's wall-clock interleaving of external infra-event markers.

Stretch goal of the session-observability work: the root cause of the real
incident was an EXTERNAL event (the provisioner restarting the host) that the
engine could not observe. These markers are emitted out-of-band via the
internal-only ``POST /mesh/system/lifecycle_event`` endpoint and interleaved by
wall-clock into the ``openlegion session`` timeline.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import pytest
from click.testing import CliRunner
from httpx import ASGITransport, AsyncClient

from src.cli import cli
from src.cli.session_reader import _fetch_lifecycle, assemble_session, render_session
from src.host.intent import IntentStore
from src.host.lifecycle import LifecycleStore
from src.host.traces import TraceStore

TRACE = "tr_lifecyclexx"


# ── LifecycleStore unit ──────────────────────────────────────


class TestLifecycleStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "lifecycle.db")
        self.store = LifecycleStore(db_path=self.db_path)

    def teardown_method(self):
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_record_returns_row_and_persists(self):
        row = self.store.record(kind="host_restart", detail="provisioner update")
        assert row["kind"] == "host_restart"
        assert row["detail"] == "provisioner update"
        assert row["timestamp"] > 0
        recent = self.store.list_recent()
        assert len(recent) == 1
        assert recent[0]["kind"] == "host_restart"

    def test_explicit_timestamp_honoured(self):
        when = time.time() - 3600
        self.store.record(kind="deploy", timestamp=when)
        rows = self.store.list_recent()
        assert rows[0]["timestamp"] == when

    def test_list_between_window(self):
        # Synthetic small timestamps fall below the 90-day retention cutoff, so
        # the first-insert GC (mirrors IntentStore) would reap them. Disable GC
        # for this pure range test — retention is covered by its own test below.
        store = LifecycleStore(
            db_path=os.path.join(self._tmpdir, "between.db"),
            max_age_hours=None,
        )
        try:
            store.record(kind="a", timestamp=100.0)
            store.record(kind="b", timestamp=200.0)
            store.record(kind="c", timestamp=300.0)
            rows = store.list_between(150.0, 250.0)
            assert [r["kind"] for r in rows] == ["b"]
        finally:
            store.close()

    def test_list_recent_newest_first(self):
        # GC disabled — synthetic timestamps below the retention cutoff would
        # otherwise be reaped on the first insert. Ordering is the unit here.
        store = LifecycleStore(
            db_path=os.path.join(self._tmpdir, "order.db"),
            max_age_hours=None,
        )
        try:
            store.record(kind="first", timestamp=10.0)
            store.record(kind="second", timestamp=20.0)
            rows = store.list_recent()
            assert [r["kind"] for r in rows] == ["second", "first"]
        finally:
            store.close()

    def test_kind_and_detail_length_capped(self):
        self.store.record(kind="x" * 200, detail="d" * 5000)
        row = self.store.list_recent()[0]
        assert len(row["kind"]) == 64
        assert len(row["detail"]) <= 2000

    def test_detail_redacted_at_storage(self):
        secret = "sk-ant-api03ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.store.record(kind="deploy", detail=f"key {secret}")
        raw = self.store._conn.execute("SELECT detail FROM lifecycle_events").fetchone()[0]
        assert secret not in raw
        assert "[REDACTED]" in raw

    def test_meta_redacted_at_storage(self):
        secret = "sk-ant-api03ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.store.record(kind="deploy", meta={"note": f"creds {secret}"})
        raw = self.store._conn.execute("SELECT meta_json FROM lifecycle_events").fetchone()[0]
        assert secret not in raw

    def test_time_based_gc_drops_old_rows(self):
        store = LifecycleStore(
            db_path=os.path.join(self._tmpdir, "gc.db"),
            max_age_hours=1,
        )
        try:
            store._conn.execute(
                "INSERT INTO lifecycle_events (timestamp, kind, detail) VALUES (?, ?, ?)",
                (time.time() - 7200, "ancient", ""),
            )
            store._conn.commit()
            assert store.list_recent()
            store._last_age_gc = -300.0
            store.record(kind="fresh")
            kinds = [r["kind"] for r in store.list_recent()]
            assert "ancient" not in kinds
            assert "fresh" in kinds
        finally:
            store.close()

    def test_idempotent_schema_init(self):
        self.store.record(kind="x")
        self.store.close()
        store2 = LifecycleStore(db_path=self.db_path)
        try:
            assert store2.list_recent()
            store2.record(kind="y")
        finally:
            self.store = store2  # teardown closes it


# ── reader: wall-clock interleaving ──────────────────────────


def _seed_session_with_restart(data_dir: str) -> float:
    """Seed one trace whose actions straddle a host-restart marker.

    Returns the restart timestamp so the test can assert ordering.
    """
    intent = IntentStore(db_path=os.path.join(data_dir, "intent.db"))
    traces = TraceStore(db_path=os.path.join(data_dir, "traces.db"))
    lifecycle = LifecycleStore(db_path=os.path.join(data_dir, "lifecycle.db"))

    t0 = time.time()
    intent.record(
        trace_id=TRACE,
        origin_kind="human",
        origin_channel="dashboard",
        origin_user="alice",
        agent="alpha",
        message="long running workflow",
    )
    # Two actions bracketing the restart. Write timestamps directly so the
    # window is deterministic.
    traces._conn.execute(
        "INSERT INTO traces (trace_id, timestamp, source, agent, event_type, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (TRACE, t0, "mesh", "alpha", "chat", "dispatch"),
    )
    traces._conn.execute(
        "INSERT INTO traces (trace_id, timestamp, source, agent, event_type, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (TRACE, t0 + 100, "mesh", "alpha", "llm_call", "resumed"),
    )
    traces._conn.commit()

    restart_ts = t0 + 50
    lifecycle.record(kind="host_restart", detail="provisioner update", timestamp=restart_ts)
    # A marker WAY outside the window must NOT be pulled in.
    lifecycle.record(kind="deploy", detail="unrelated", timestamp=t0 - 100000)

    intent.close()
    traces.close()
    lifecycle.close()
    return restart_ts


def test_reader_interleaves_lifecycle_by_wallclock(tmp_path):
    restart_ts = _seed_session_with_restart(str(tmp_path))
    session = assemble_session(str(tmp_path), TRACE)
    assert session["found"]
    # Only the in-window marker is pulled in.
    assert len(session["lifecycle"]) == 1
    assert session["lifecycle"][0]["kind"] == "host_restart"
    assert session["lifecycle"][0]["timestamp"] == restart_ts

    rendered = render_session(session)
    assert "infra:host_restart" in rendered
    # The restart line must land BETWEEN the two action lines (chat → restart
    # → llm_call), proving wall-clock interleaving rather than appending.
    lines = rendered.splitlines()
    idx_chat = next(i for i, ln in enumerate(lines) if "chat" in ln)
    idx_restart = next(i for i, ln in enumerate(lines) if "infra:host_restart" in ln)
    idx_resumed = next(i for i, ln in enumerate(lines) if "llm_call" in ln)
    assert idx_chat < idx_restart < idx_resumed


def test_reader_no_lifecycle_db_degrades(tmp_path):
    # No lifecycle.db at all — reader must not crash and yields no markers.
    intent = IntentStore(db_path=os.path.join(str(tmp_path), "intent.db"))
    intent.record(trace_id=TRACE, agent="alpha", message="hi")
    intent.close()
    session = assemble_session(str(tmp_path), TRACE)
    assert session["lifecycle"] == []
    assert "infra:" not in render_session(session)


def test_fetch_lifecycle_missing_db_returns_empty(tmp_path):
    assert _fetch_lifecycle(tmp_path, 0.0, time.time()) == []


# ── CLI: session --recent ────────────────────────────────────


def test_session_recent_lists_sessions(tmp_path):
    intent = IntentStore(db_path=os.path.join(str(tmp_path), "intent.db"))
    for i in range(3):
        intent.record(
            trace_id=f"tr_recent{i}",
            origin_kind="human",
            origin_user="alice",
            agent="alpha",
            message=f"request {i}",
        )
    intent.close()

    res = CliRunner().invoke(cli, ["session", "--recent", "2", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    # Newest first, capped at 2.
    assert "tr_recent2" in res.output
    assert "tr_recent1" in res.output
    assert "tr_recent0" not in res.output


def test_session_recent_default_count_and_json(tmp_path):
    intent = IntentStore(db_path=os.path.join(str(tmp_path), "intent.db"))
    intent.record(trace_id="tr_only", origin_user="bob", agent="beta", message="hello")
    intent.close()

    res = CliRunner().invoke(cli, ["session", "--recent", "--json", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    import json as _json

    payload = _json.loads(res.output)
    assert payload["sessions"][0]["trace_id"] == "tr_only"


def test_session_without_trace_or_recent_errors(tmp_path):
    res = CliRunner().invoke(cli, ["session", "--data-dir", str(tmp_path)])
    assert res.exit_code == 1
    assert "trace_id" in res.output


# ── endpoint: internal-only + happy path ─────────────────────


def _build_min_mesh(tmp_path):
    import src.host.server as server_module
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix

    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    lifecycle = LifecycleStore(db_path=str(tmp_path / "lifecycle.db"))
    app = server_module.create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        lifecycle_store=lifecycle,
    )
    app.state._test_bb = bb
    app.state._test_lifecycle = lifecycle
    return app


@pytest.mark.asyncio
async def test_lifecycle_endpoint_records_internal(tmp_path):
    app = _build_min_mesh(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/system/lifecycle_event",
                headers={"x-mesh-internal": "1"},
                json={"kind": "host_restart", "detail": "systemctl restart openlegion"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["recorded"] is True
        assert body["event"]["kind"] == "host_restart"
        # Persisted to the store.
        rows = app.state._test_lifecycle.list_recent()
        assert rows and rows[0]["kind"] == "host_restart"
    finally:
        app.state._test_bb.close()
        app.state._test_lifecycle.close()


@pytest.mark.asyncio
async def test_lifecycle_endpoint_rejects_non_internal(tmp_path):
    app = _build_min_mesh(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # No x-mesh-internal header → not an internal caller → 403.
            resp = await client.post(
                "/mesh/system/lifecycle_event",
                json={"kind": "host_restart"},
            )
        assert resp.status_code == 403
        assert app.state._test_lifecycle.list_recent() == []
    finally:
        app.state._test_bb.close()
        app.state._test_lifecycle.close()


@pytest.mark.asyncio
async def test_lifecycle_endpoint_requires_kind(tmp_path):
    app = _build_min_mesh(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/system/lifecycle_event",
                headers={"x-mesh-internal": "1"},
                json={"detail": "no kind"},
            )
        assert resp.status_code == 400
    finally:
        app.state._test_bb.close()
        app.state._test_lifecycle.close()


# ── App-construction smoke: lifecycle_store wiring ───────────────
#
# Regression for the ship-blocker where ``_start_mesh_server`` passed
# ``lifecycle_store=`` to BOTH ``create_mesh_app`` (which accepts it) and
# ``create_dashboard_router`` (which does NOT) → ``TypeError`` at mesh
# startup. The dashboard never consumes lifecycle markers — the
# ``openlegion session`` reader reads ``lifecycle.db`` directly — so the
# kwarg belongs to the mesh factory only. These two asserts pin that
# split so the spurious kwarg can't be reintroduced.


def test_create_mesh_app_accepts_lifecycle_store(tmp_path):
    import src.host.server as server_module
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix

    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    lifecycle = LifecycleStore(db_path=str(tmp_path / "lifecycle.db"))
    try:
        # Must construct without TypeError.
        app = server_module.create_mesh_app(
            blackboard=bb,
            pubsub=pubsub,
            router=router,
            permissions=perms,
            lifecycle_store=lifecycle,
        )
        assert app is not None
    finally:
        bb.close()
        lifecycle.close()


def test_create_dashboard_router_does_not_accept_lifecycle_store(tmp_path):
    """The dashboard factory must NOT take ``lifecycle_store``.

    The runtime wires it only into ``create_mesh_app``; passing it to the
    dashboard router was the original ``TypeError`` ship-blocker. This
    asserts the signature stays free of it so the bug can't regress.
    """
    from unittest.mock import MagicMock

    from src.dashboard.server import create_dashboard_router

    registry: dict[str, str] = {}
    common = dict(
        blackboard=MagicMock(),
        health_monitor=None,
        cost_tracker=MagicMock(),
        trace_store=None,
        event_bus=None,
        agent_registry=registry,
        mesh_port=8420,
    )

    # The legitimate call (no lifecycle_store) constructs fine.
    router = create_dashboard_router(**common)
    assert router is not None

    # Passing lifecycle_store would raise the original TypeError.
    lifecycle = LifecycleStore(db_path=str(tmp_path / "lifecycle.db"))
    try:
        with pytest.raises(TypeError):
            create_dashboard_router(**common, lifecycle_store=lifecycle)
    finally:
        lifecycle.close()
