"""Tests for the ThreadStore (Team Threads, Phase-2 unit 2).

Covers the store itself (schema, ensure_* idempotency, caps, the
7d/24h event query windows, the 90-day reaper, archive semantics) plus
the team-channel lifecycle at the mesh endpoint layer (create →
thread_ref, boot backfill, delete → archive).
"""

from __future__ import annotations

import importlib
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.threads import (
    EVENT_QUERY_LIMIT,
    EVENT_RETENTION_SECONDS,
    MAX_BODY_CHARS,
    MAX_PAYLOAD_BYTES,
    ThreadNotFound,
    ThreadStore,
)


@pytest.fixture()
def store():
    s = ThreadStore(":memory:")
    yield s
    s.close()


def _age_message(store: ThreadStore, message_id: int, age_seconds: float) -> None:
    """Backdate a message row so window/reap tests don't sleep."""
    with store._conn() as conn:
        conn.execute(
            "UPDATE thread_messages SET created_at = ? WHERE id = ?",
            (time.time() - age_seconds, message_id),
        )


class TestSchema:
    def test_canonical_v1_user_version(self, store):
        with store._conn() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    def test_disk_backed_schema(self, tmp_path):
        s = ThreadStore(str(tmp_path / "threads.db"))
        th = s.create_thread("alpha", "channel", title="x")
        assert s.get_thread(th["id"]) is not None
        # Re-open — schema init is idempotent and rows persist.
        s2 = ThreadStore(str(tmp_path / "threads.db"))
        assert s2.get_thread(th["id"]) is not None


class TestThreadCrud:
    def test_create_and_get(self, store):
        th = store.create_thread("team-a", "channel", title="#team-a", created_by="operator")
        assert th["scope_id"] == "team-a"
        assert th["kind"] == "channel"
        assert th["archived"] is False
        assert store.get_thread(th["id"])["title"] == "#team-a"

    def test_invalid_kind_rejected(self, store):
        with pytest.raises(ValueError, match="kind must be one of"):
            store.create_thread("team-a", "standup")

    def test_empty_scope_rejected(self, store):
        with pytest.raises(ValueError, match="scope_id"):
            store.create_thread("", "channel")

    def test_ensure_channel_idempotent(self, store):
        a = store.ensure_channel("team-a")
        b = store.ensure_channel("team-a")
        assert a["id"] == b["id"] == "channel:team-a"
        assert len(store.list_threads(scope_id="team-a")) == 1

    def test_ensure_task_thread_idempotent(self, store):
        a = store.ensure_task_thread("team-a", "task_123", title="do it")
        b = store.ensure_task_thread("team-a", "task_123", title="ignored on second call")
        assert a["id"] == b["id"] == "task:task_123"
        assert b["title"] == "do it"
        assert b["task_id"] == "task_123"

    def test_ensure_dm_thread_pair_is_unordered(self, store):
        a = store.ensure_dm_thread("scout", "scout", "analyst")
        b = store.ensure_dm_thread("scout", "analyst", "scout")
        assert a["id"] == b["id"] == "dm:analyst:scout"

    def test_list_threads_filters(self, store):
        store.ensure_channel("team-a")
        store.ensure_task_thread("team-a", "t1")
        store.ensure_dm_thread("solo-agent", "solo-agent", "peer")
        assert {t["kind"] for t in store.list_threads(scope_id="team-a")} == {"channel", "task"}
        assert [t["id"] for t in store.list_threads(kind="dm")] == ["dm:peer:solo-agent"]
        with pytest.raises(ValueError, match="unknown thread kind"):
            store.list_threads(kind="bogus")

    def test_archive_scope_hides_but_keeps_rows(self, store):
        store.ensure_channel("team-a")
        store.ensure_task_thread("team-a", "t1")
        store.ensure_channel("team-b")
        assert store.archive_scope("team-a") == 2
        assert store.list_threads(scope_id="team-a") == []
        archived = store.list_threads(scope_id="team-a", include_archived=True)
        assert len(archived) == 2
        assert all(t["archived"] for t in archived)
        # Other scopes untouched.
        assert len(store.list_threads(scope_id="team-b")) == 1


class TestMessages:
    def test_post_and_list(self, store):
        th = store.ensure_channel("team-a")
        m1 = store.post_message(th["id"], "scout", body="hello")
        m2 = store.post_message(th["id"], "analyst", body="hi")
        rows = store.list_messages(th["id"])
        assert [r["id"] for r in rows] == [m1["id"], m2["id"]]  # oldest-first
        assert rows[0]["body"] == "hello"

    def test_post_bumps_thread_updated_at(self, store):
        th = store.ensure_channel("team-a")
        before = th["updated_at"]
        time.sleep(0.01)
        store.post_message(th["id"], "scout", body="x")
        assert store.get_thread(th["id"])["updated_at"] > before

    def test_unknown_thread_rejected(self, store):
        with pytest.raises(ThreadNotFound):
            store.post_message("th_missing", "scout", body="x")

    def test_invalid_message_kind_rejected(self, store):
        th = store.ensure_channel("team-a")
        with pytest.raises(ValueError, match="kind must be one of"):
            store.post_message(th["id"], "scout", kind="note")

    def test_body_cap_truncates_with_notice(self, store):
        th = store.ensure_channel("team-a")
        m = store.post_message(th["id"], "scout", body="x" * (MAX_BODY_CHARS + 500))
        assert len(m["body"]) == MAX_BODY_CHARS
        assert m["body"].endswith("[truncated]")
        stored = store.list_messages(th["id"])[0]
        assert len(stored["body"]) == MAX_BODY_CHARS

    def test_payload_cap_replaces_with_marker(self, store):
        th = store.ensure_channel("team-a")
        big = {"blob": "y" * (MAX_PAYLOAD_BYTES + 1)}
        m = store.post_message(th["id"], "scout", payload=big)
        assert m["payload"]["truncated"] is True
        stored = store.list_messages(th["id"])[0]
        assert stored["payload"]["truncated"] is True
        assert stored["payload"]["original_bytes"] > MAX_PAYLOAD_BYTES

    def test_list_messages_before_pages_back(self, store):
        th = store.ensure_channel("team-a")
        for i in range(5):
            m = store.post_message(th["id"], "scout", body=f"m{i}")
            _age_message(store, m["id"], 100 - i)  # oldest = m0
        page = store.list_messages(th["id"], limit=2)
        assert [r["body"] for r in page] == ["m3", "m4"]
        older = store.list_messages(th["id"], before=page[0]["created_at"], limit=2)
        assert [r["body"] for r in older] == ["m1", "m2"]

    def test_list_messages_limit_capped_at_200(self, store):
        th = store.ensure_channel("team-a")
        assert store.list_messages(th["id"], limit=10_000) == []

    def test_recent_messages_across_threads_excludes_events(self, store):
        a = store.ensure_dm_thread("scout", "scout", "analyst")
        b = store.ensure_task_thread("scout", "t1")
        store.post_message(a["id"], "scout", recipient="analyst", body="dm")
        store.post_message(b["id"], "mesh", recipient="scout", kind="event", payload={"kind": "task_completed"})
        rows = store.recent_messages(kind="message", limit=10)
        assert [r["body"] for r in rows] == ["dm"]

    def test_thread_message_event_emitted(self):
        emitted = []

        class _Bus:
            def emit(self, event_type, agent="", data=None):
                emitted.append((event_type, agent, data))

        s = ThreadStore(":memory:", event_bus=_Bus())
        th = s.ensure_channel("team-a")
        s.post_message(th["id"], "scout", recipient="analyst", body="hi")
        assert emitted and emitted[-1][0] == "thread_message"
        assert emitted[-1][1] == "scout"
        assert emitted[-1][2]["thread_id"] == th["id"]
        s.close()


class TestEventWindows:
    """The former blackboard TTL split (7d actionable / 24h informational)
    is now a pair of query windows in ``list_events_for``."""

    def _post_event(self, store, task_id, kind, recipient="scout", age=0.0):
        th = store.ensure_task_thread("scout", task_id)
        m = store.post_message(
            th["id"], "mesh", recipient=recipient, kind="event",
            payload={"kind": kind, "task_id": task_id, "status": "x", "ts": int(time.time())},
        )
        if age:
            _age_message(store, m["id"], age)
        return m

    def test_fresh_events_of_both_classes_served(self, store):
        self._post_event(store, "t1", "task_completed")
        self._post_event(store, "t2", "task_failed")
        kinds = {e["kind"] for e in store.list_events_for("scout")}
        assert kinds == {"task_completed", "task_failed"}

    def test_informational_expires_after_24h(self, store):
        self._post_event(store, "t1", "task_completed", age=86_400 + 60)
        self._post_event(store, "t2", "task_cancelled", age=86_400 + 60)
        assert store.list_events_for("scout") == []

    def test_actionable_survives_24h_but_not_7d(self, store):
        self._post_event(store, "t1", "task_failed", age=86_400 + 60)
        self._post_event(store, "t2", "task_blocked", age=604_800 + 60)
        kinds = [e["kind"] for e in store.list_events_for("scout")]
        assert kinds == ["task_failed"]

    def test_recipient_scoping(self, store):
        self._post_event(store, "t1", "task_failed", recipient="scout")
        self._post_event(store, "t2", "task_failed", recipient="analyst")
        assert {e["task_id"] for e in store.list_events_for("scout")} == {"t1"}

    def test_envelope_carries_payload_plus_row_refs(self, store):
        m = self._post_event(store, "t1", "task_failed")
        (ev,) = store.list_events_for("scout")
        assert ev["kind"] == "task_failed"
        assert ev["task_id"] == "t1"
        assert ev["id"] == m["id"]
        assert ev["thread_id"] == "task:t1"
        assert ev["created_at"] is not None

    def test_newest_first_ordering(self, store):
        self._post_event(store, "t_old", "task_failed", age=3600)
        self._post_event(store, "t_new", "task_failed")
        assert [e["task_id"] for e in store.list_events_for("scout")] == ["t_new", "t_old"]


class TestEventDedupe:
    """Read-side newest-per-task dedupe. The old blackboard back-edge was
    an UPSERT per (recipient, task): one event per task, latest transition
    wins, window re-classified on overwrite. ``list_events_for`` restores
    those overwrite semantics on read."""

    def _post_event(self, store, task_id, kind, recipient="scout", age=0.0):
        th = store.ensure_task_thread("scout", task_id)
        m = store.post_message(
            th["id"], "mesh", recipient=recipient, kind="event",
            payload={"kind": kind, "task_id": task_id, "status": "x", "ts": int(time.time())},
        )
        if age:
            _age_message(store, m["id"], age)
        return m

    def test_blocked_then_done_serves_one_informational(self, store):
        """A later informational transition (task_completed after
        task_blocked) silences the stale actionable event."""
        self._post_event(store, "t1", "task_blocked", age=3600)
        done = self._post_event(store, "t1", "task_completed")
        events = store.list_events_for("scout")
        assert [e["kind"] for e in events] == ["task_completed"]
        assert events[0]["id"] == done["id"]

    def test_blocked_then_done_nothing_after_24h(self, store):
        """The surviving informational row ages out at 24h AND keeps
        shadowing the task's stale actionable transition."""
        self._post_event(store, "t1", "task_blocked", age=86_400 + 3600)
        self._post_event(store, "t1", "task_completed", age=86_400 + 60)
        assert store.list_events_for("scout") == []

    def test_failed_twice_serves_one_task_failed(self, store):
        self._post_event(store, "t1", "task_failed", age=60)
        newest = self._post_event(store, "t1", "task_failed")
        events = store.list_events_for("scout")
        assert [e["kind"] for e in events] == ["task_failed"]
        assert events[0]["id"] == newest["id"]  # newest transition wins

    def test_blocked_working_blocked_serves_one(self, store):
        """Only terminal/blocked transitions write events — a
        blocked→working→blocked cycle appends two task_blocked rows;
        exactly one is served."""
        self._post_event(store, "t1", "task_blocked", age=120)
        newest = self._post_event(store, "t1", "task_blocked")
        events = store.list_events_for("scout")
        assert [e["kind"] for e in events] == ["task_blocked"]
        assert events[0]["id"] == newest["id"]

    def test_dedupe_is_per_task(self, store):
        self._post_event(store, "t1", "task_failed", age=60)
        self._post_event(store, "t1", "task_failed")
        self._post_event(store, "t2", "task_blocked")
        assert {e["task_id"] for e in store.list_events_for("scout")} == {"t1", "t2"}

    def test_non_task_thread_events_not_deduped(self, store):
        """Fallback dedupe key is the message id when the thread isn't
        task-scoped — no cross-message collapsing."""
        ch = store.ensure_channel("scout")
        for _ in range(2):
            store.post_message(
                ch["id"], "mesh", recipient="scout", kind="event",
                payload={"kind": "task_failed", "task_id": "tX"},
            )
        assert len(store.list_events_for("scout")) == 2

    def test_hard_query_limit_bounds_result(self, store):
        for i in range(EVENT_QUERY_LIMIT + 5):
            self._post_event(store, f"t{i}", "task_failed")
        assert len(store.list_events_for("scout")) == EVENT_QUERY_LIMIT


class TestArchivedRevival:
    """ensure_* / post_message revive archived rows — a recreated team
    (or fresh DM traffic after a team delete) resurfaces the thread."""

    def test_ensure_channel_revives_archived(self, store):
        store.ensure_channel("team-a")
        store.archive_scope("team-a")
        assert store.list_threads(scope_id="team-a") == []
        th = store.ensure_channel("team-a")
        assert th["archived"] is False
        assert [t["id"] for t in store.list_threads(scope_id="team-a")] == ["channel:team-a"]

    def test_ensure_task_and_dm_revive_archived(self, store):
        store.ensure_task_thread("team-a", "t1")
        store.ensure_dm_thread("team-a", "scout", "analyst")
        store.archive_scope("team-a")
        assert store.ensure_task_thread("team-a", "t1")["archived"] is False
        assert store.ensure_dm_thread("team-a", "scout", "analyst")["archived"] is False

    def test_post_message_unarchives_thread(self, store):
        """DM traffic after a team delete makes the conversation visible
        again."""
        th = store.ensure_dm_thread("team-a", "scout", "analyst")
        store.archive_scope("team-a")
        assert store.list_threads(kind="dm") == []
        store.post_message(th["id"], "scout", recipient="analyst", body="still here?")
        (live,) = store.list_threads(kind="dm")
        assert live["id"] == th["id"]
        assert live["archived"] is False


class TestReaper:
    def test_reap_drops_only_old_events(self, store):
        th = store.ensure_task_thread("scout", "t1")
        old_ev = store.post_message(th["id"], "mesh", recipient="scout", kind="event", payload={"kind": "task_failed"})
        _age_message(store, old_ev["id"], EVENT_RETENTION_SECONDS + 60)
        fresh_ev = store.post_message(
            th["id"], "mesh", recipient="scout", kind="event", payload={"kind": "task_failed"},
        )
        old_msg = store.post_message(th["id"], "scout", body="durable")
        _age_message(store, old_msg["id"], EVENT_RETENTION_SECONDS + 60)

        assert store.reap_expired() == 1
        ids = [r["id"] for r in store.list_messages(th["id"], limit=200)]
        assert old_ev["id"] not in ids
        assert fresh_ev["id"] in ids
        # Plain messages are durable — no reap this phase.
        assert old_msg["id"] in ids

    def test_safe_reap_rate_limited(self, store):
        th = store.ensure_task_thread("scout", "t1")
        ev = store.post_message(th["id"], "mesh", recipient="scout", kind="event", payload={"kind": "task_failed"})
        store._safe_reap()  # stamps the interval
        _age_message(store, ev["id"], EVENT_RETENTION_SECONDS + 60)
        store._safe_reap()  # inside the min interval — no delete
        with store._conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM thread_messages").fetchone()[0] == 1


# ── Team channel lifecycle at the mesh endpoint layer ──────────────────


@pytest.fixture
def team_thread_app(tmp_path, monkeypatch):
    """Mesh app with operator auth + real Team/Thread stores (mirrors
    tests/test_teams.py's ``team_app`` fixture)."""
    monkeypatch.chdir(tmp_path)
    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(json.dumps({"permissions": {
        "agent1": {"blackboard_read": [], "blackboard_write": []},
    }}))
    import yaml as _yaml

    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(_yaml.dump({"agents": {
        "agent1": {"role": "a"},
        "operator": {"role": "operator"},
    }}))
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

    import src.host.server as server_module

    importlib.reload(server_module)

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.teams import TeamStore

    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    teams_store = TeamStore(db_path=":memory:")
    thread_store = ThreadStore(":memory:")
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        teams_store=teams_store,
        thread_store=thread_store,
        auth_tokens={"operator": "op-token"},
    )
    yield app, teams_store, thread_store
    blackboard.close()
    importlib.reload(server_module)


_OP_HEADERS = {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


class TestTeamChannelLifecycle:
    @pytest.mark.asyncio
    async def test_team_create_makes_channel_and_thread_ref(self, team_thread_app):
        app, teams_store, thread_store = team_thread_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/mesh/teams", json={"name": "growth"}, headers=_OP_HEADERS)
        assert r.status_code == 200, r.text
        team = teams_store.get_team("growth")
        assert team["thread_ref"] == "channel:growth"
        ch = thread_store.get_thread("channel:growth")
        assert ch is not None and ch["kind"] == "channel" and ch["scope_id"] == "growth"

    @pytest.mark.asyncio
    async def test_team_delete_archives_threads(self, team_thread_app):
        app, teams_store, thread_store = team_thread_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/mesh/teams", json={"name": "growth"}, headers=_OP_HEADERS)
            r = await c.delete("/mesh/teams/growth", headers=_OP_HEADERS)
        assert r.status_code == 200, r.text
        # Archived, NOT deleted — audit trail.
        assert thread_store.list_threads(scope_id="growth") == []
        archived = thread_store.list_threads(scope_id="growth", include_archived=True)
        assert [t["id"] for t in archived] == ["channel:growth"]

    @pytest.mark.asyncio
    async def test_team_recreate_after_delete_revives_channel(self, team_thread_app):
        """Delete team → recreate same name → the channel is live again
        and thread_ref points at it (no archived-thread resurrection bug)."""
        app, teams_store, thread_store = team_thread_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/mesh/teams", json={"name": "growth"}, headers=_OP_HEADERS)
            await c.delete("/mesh/teams/growth", headers=_OP_HEADERS)
            r = await c.post("/mesh/teams", json={"name": "growth"}, headers=_OP_HEADERS)
        assert r.status_code == 200, r.text
        assert teams_store.get_team("growth")["thread_ref"] == "channel:growth"
        ch = thread_store.get_thread("channel:growth")
        assert ch is not None and ch["archived"] is False
        assert [t["id"] for t in thread_store.list_threads(scope_id="growth")] == ["channel:growth"]

    def test_boot_backfill_for_null_thread_ref(self, tmp_path, monkeypatch):
        """Teams created before this unit (thread_ref NULL) get a channel
        at app construction."""
        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.teams import TeamStore

        teams_store = TeamStore(db_path=":memory:")
        teams_store.create_team("legacy-team")
        assert teams_store.get_team("legacy-team")["thread_ref"] is None
        thread_store = ThreadStore(":memory:")
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        try:
            server_module.create_mesh_app(
                blackboard=blackboard,
                pubsub=PubSub(),
                router=MessageRouter(PermissionMatrix(), {}),
                permissions=PermissionMatrix(),
                teams_store=teams_store,
                thread_store=thread_store,
                auth_tokens={"operator": "tok"},
            )
            assert teams_store.get_team("legacy-team")["thread_ref"] == "channel:legacy-team"
            assert thread_store.get_thread("channel:legacy-team") is not None
        finally:
            blackboard.close()
            importlib.reload(server_module)

    def test_boot_backfill_survives_one_bad_team(self, tmp_path):
        """The per-team try/except keeps one failing team from abandoning
        the rest of the fleet's channel backfill."""
        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.teams import TeamStore

        teams_store = TeamStore(db_path=":memory:")
        teams_store.create_team("bad-team")
        teams_store.create_team("good-team")
        thread_store = ThreadStore(":memory:")
        orig_ensure = thread_store.ensure_channel

        def _flaky(team_id):
            if team_id == "bad-team":
                raise RuntimeError("boom")
            return orig_ensure(team_id)

        thread_store.ensure_channel = _flaky
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        try:
            server_module.create_mesh_app(
                blackboard=blackboard,
                pubsub=PubSub(),
                router=MessageRouter(PermissionMatrix(), {}),
                permissions=PermissionMatrix(),
                teams_store=teams_store,
                thread_store=thread_store,
                auth_tokens={"operator": "tok"},
            )
            assert teams_store.get_team("good-team")["thread_ref"] == "channel:good-team"
            assert teams_store.get_team("bad-team")["thread_ref"] is None
        finally:
            blackboard.close()
            importlib.reload(server_module)


# ── GET /mesh/agents/{id}/task-events auth matrix ──────────────────────


@pytest.fixture
def events_app(tmp_path):
    import src.host.server as server_module

    importlib.reload(server_module)
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix

    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {
        "operator": "http://op:8400",
        "scout": "http://scout:8400",
        "analyst": "http://analyst:8400",
    })
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    thread_store = ThreadStore(":memory:")
    th = thread_store.ensure_task_thread("scout", "t1")
    thread_store.post_message(
        th["id"], "mesh", recipient="scout", kind="event",
        payload={"kind": "task_failed", "task_id": "t1", "recipient": "analyst",
                 "title": "x", "status": "failed", "ts": 1, "error": "boom"},
    )
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        thread_store=thread_store,
        auth_tokens={
            "operator": "op-token",
            "scout": "scout-token",
            "analyst": "analyst-token",
            # Authed but NOT in the router registry — exercises the
            # unknown-agent 404 self-read path.
            "ghost": "ghost-token",
        },
    )
    yield app
    blackboard.close()
    importlib.reload(server_module)


def _bearer(agent: str, token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "X-Agent-ID": agent}


class TestTaskEventsEndpointAuth:
    """Mirrors the goals GET auth matrix: self-or-operator-or-internal."""

    @pytest.mark.asyncio
    async def test_self_read_allowed(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/scout/task-events", headers=_bearer("scout", "scout-token"))
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["agent_id"] == "scout"
        assert data["count"] == 1
        assert data["events"][0]["kind"] == "task_failed"

    @pytest.mark.asyncio
    async def test_operator_read_allowed(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/scout/task-events", headers=_bearer("operator", "op-token"))
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_internal_read_allowed(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/scout/task-events", headers={"x-mesh-internal": "1"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_peer_read_denied(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/scout/task-events", headers=_bearer("analyst", "analyst-token"))
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_unknown_agent_404_names_roster_for_operator(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/ghost/task-events", headers=_bearer("operator", "op-token"))
        assert r.status_code == 404
        assert "Available:" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_unknown_agent_404_names_roster_for_internal(self, events_app):
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/ghost/task-events", headers={"x-mesh-internal": "1"})
        assert r.status_code == 404
        assert "Available:" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_unknown_agent_bare_404_for_non_operator(self, events_app):
        """Roster disclosure is operator/internal-only — an authed but
        unregistered agent's self-read gets a bare 404."""
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            r = await c.get("/mesh/agents/ghost/task-events", headers=_bearer("ghost", "ghost-token"))
        assert r.status_code == 404
        detail = r.json()["detail"]
        assert "Available" not in detail
        assert "scout" not in detail


class TestMessageRateLimit:
    """C-fix: direct messaging gets its own rate bucket — a runaway
    message loop 429s instead of hammering the router + thread store."""

    @pytest.mark.asyncio
    async def test_message_bucket_429_when_exceeded(self, events_app):
        # _RATE_LIMITS is the same dict object exposed on app.state —
        # tighten the bucket so the test doesn't need 300 requests.
        events_app.state.rate_limits["message"] = (2, 60)
        body = {"from_agent": "operator", "to": "ghost", "type": "query", "payload": {}}
        async with AsyncClient(transport=ASGITransport(app=events_app), base_url="http://t") as c:
            for _ in range(2):
                r = await c.post("/mesh/message", json=body, headers=_bearer("operator", "op-token"))
                assert r.status_code == 200, r.text
            r = await c.post("/mesh/message", json=body, headers=_bearer("operator", "op-token"))
        assert r.status_code == 429
        assert "message" in r.json()["detail"]


class TestCheckInboxMultiplicity:
    """End-to-end multiplicity pin (endpoint → check_inbox): repeated
    transitions on one task reach the agent as ONE event per task, and
    actionable events are never dropped by the agent-side 25-cap."""

    @pytest.mark.asyncio
    async def test_operator_sees_one_event_per_task_actionable_uncapped(self, tmp_path):
        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix

        thread_store = ThreadStore(":memory:")
        n_tasks = 30  # above the agent-side 25-cap
        for i in range(n_tasks):
            th = thread_store.ensure_task_thread("scout", f"t{i}")
            for kind, status in (("task_blocked", "blocked"), ("task_failed", "failed")):
                thread_store.post_message(
                    th["id"], "mesh", recipient="operator", kind="event",
                    payload={"kind": kind, "task_id": f"t{i}", "status": status,
                             "ts": int(time.time()), "title": f"work {i}"},
                )
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        app = server_module.create_mesh_app(
            blackboard=blackboard,
            pubsub=PubSub(),
            router=MessageRouter(PermissionMatrix(), {}),
            permissions=PermissionMatrix(),
            thread_store=thread_store,
            auth_tokens={"operator": "op-token"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.get("/mesh/agents/operator/task-events", headers=_OP_HEADERS)
            assert r.status_code == 200, r.text
            served = r.json()["events"]
            # ONE event per task; the newest transition (task_failed) wins.
            assert len(served) == n_tasks
            assert all(e["kind"] == "task_failed" for e in served)

            from unittest.mock import AsyncMock, MagicMock

            from src.agent.builtins.coordination_tool import check_inbox

            mc = MagicMock()
            mc.agent_id = "operator"
            mc.list_task_inbox = AsyncMock(return_value=[])
            mc.list_inbox_events = AsyncMock(return_value=served)
            result = await check_inbox(mesh_client=mc)
            # Actionable events are NEVER dropped by the 25-cap...
            assert result["event_count"] == n_tasks
            assert result["events_truncated"] is False
            # ...and multiplicity stays one-per-task through the chain.
            assert len({e["task_id"] for e in result["events"]}) == n_tasks
        finally:
            blackboard.close()
            thread_store.close()
            importlib.reload(server_module)
