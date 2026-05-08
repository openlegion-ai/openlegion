"""Tests for the persistent per-session opened-conversations store + API.

Backs the multi-user safety fix: the previous in-memory ``set[str]`` was
shared across every dashboard session in the same Python process, so
one user opening a chat surfaced that worker on every other concurrent
session. The store keys every row on a per-session identifier (a hash
of the ``ol_session`` cookie) so two sessions never see each other's
opened workers, and persists rows so opened state survives engine
restarts.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI

from src.dashboard.conversations import OpenedConversationsStore

# Reuse the dashboard test scaffolding so we exercise the real router
# alongside the pure-store unit tests.
from tests.test_dashboard import _CSRFTestClient, _make_components, _teardown

# ── Pure-store unit tests ────────────────────────────────────────────


class TestOpenedConversationsStore:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "conversations.db")
        self.store = OpenedConversationsStore(db_path=self._db_path)

    def teardown_method(self) -> None:
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_open_persists_for_session(self):
        self.store.open("session-a", "writer")
        assert self.store.list_for_session("session-a") == ["writer"]

    def test_close_removes_from_session(self):
        self.store.open("session-a", "writer")
        self.store.close_conversation("session-a", "writer")
        assert self.store.list_for_session("session-a") == []

    def test_close_unknown_is_noop(self):
        # close on an empty bucket should not raise (matches the prior
        # ``set.discard`` semantics used by the in-memory implementation).
        self.store.close_conversation("session-a", "writer")
        assert self.store.list_for_session("session-a") == []

    def test_two_sessions_dont_leak(self):
        self.store.open("session-a", "writer")
        # Session B's view must be empty even though session A opened a
        # conversation in the same process.
        assert self.store.list_for_session("session-b") == []
        # Adding a conversation under session B must not affect session A.
        self.store.open("session-b", "researcher")
        assert self.store.list_for_session("session-a") == ["writer"]
        assert self.store.list_for_session("session-b") == ["researcher"]

    def test_open_is_idempotent(self):
        self.store.open("session-a", "writer")
        self.store.open("session-a", "writer")
        # No PK violation, no duplicate rows.
        assert self.store.list_for_session("session-a") == ["writer"]

    def test_list_for_session_sorted(self):
        self.store.open("session-a", "delta")
        self.store.open("session-a", "alpha")
        self.store.open("session-a", "charlie")
        # Lexicographic order so the messenger's worker order is stable
        # across reloads (matches the prior ``sorted(...)`` call site).
        assert self.store.list_for_session("session-a") == ["alpha", "charlie", "delta"]

    def test_persistence_across_store_recreate(self):
        self.store.open("session-a", "writer")
        self.store.close()
        # Recreate with the same db_path — rows must survive.
        self.store = OpenedConversationsStore(db_path=self._db_path)
        assert self.store.list_for_session("session-a") == ["writer"]

    def test_open_rejects_empty_inputs(self):
        import pytest
        with pytest.raises(ValueError):
            self.store.open("", "writer")
        with pytest.raises(ValueError):
            self.store.open("session-a", "")

    def test_list_for_session_empty_session_id_returns_empty(self):
        # Defensive: never accidentally enumerate every session.
        assert self.store.list_for_session("") == []


# ── HTTP-level cross-session test ────────────────────────────────────


class TestEndpointSessionScope:
    """End-to-end: two distinct ``ol_session`` cookies → distinct lists.

    Exercises the full router path through ``_conversations_session_id``
    (which hashes the cookie) and the SQLite-backed store. Auth runs in
    dev mode here (no access token on disk), so any non-empty cookie
    value passes verification but still differentiates the buckets.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.components["agent_registry"]["operator"] = "http://localhost:8499"
        from src.dashboard.server import create_dashboard_router
        router = create_dashboard_router(**self.components, mesh_port=8420)
        app = FastAPI()
        app.include_router(router)
        self.app = app

    def teardown_method(self) -> None:
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_endpoint_uses_session_cookie(self):
        client_a = _CSRFTestClient(self.app, cookies={"ol_session": "cookie-A"})
        client_b = _CSRFTestClient(self.app, cookies={"ol_session": "cookie-B"})

        # Session A opens "alpha" — must NOT appear in session B's list.
        resp = client_a.post("/dashboard/api/conversations/alpha/open")
        assert resp.status_code == 200

        a_listing = client_a.get("/dashboard/api/conversations").json()
        b_listing = client_b.get("/dashboard/api/conversations").json()
        a_ids = [w["agent_id"] for w in a_listing["workers"]]
        b_ids = [w["agent_id"] for w in b_listing["workers"]]
        assert a_ids == ["alpha"]
        assert b_ids == []

        # Session B opens "beta" — A's list still only has alpha.
        resp = client_b.post("/dashboard/api/conversations/beta/open")
        assert resp.status_code == 200

        a_ids = [w["agent_id"] for w in client_a.get("/dashboard/api/conversations").json()["workers"]]
        b_ids = [w["agent_id"] for w in client_b.get("/dashboard/api/conversations").json()["workers"]]
        assert a_ids == ["alpha"]
        assert b_ids == ["beta"]

        # Session A closes its conversation — B is untouched.
        resp = client_a.post("/dashboard/api/conversations/alpha/close")
        assert resp.status_code == 200
        a_ids = [w["agent_id"] for w in client_a.get("/dashboard/api/conversations").json()["workers"]]
        b_ids = [w["agent_id"] for w in client_b.get("/dashboard/api/conversations").json()["workers"]]
        assert a_ids == []
        assert b_ids == ["beta"]
