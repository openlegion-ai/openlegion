"""Tests for the workplace activity-feed endpoint added in PR 5.

The feed is the new default landing for the workplace tab; it joins
``task_events`` back to ``tasks`` and renders a humanized summary per
event so the operator can see what the fleet is doing without inspecting
columns or drilling into individual records.

Covers: empty state when v2 off, happy path with shape assertions,
project filter, limit clamping, descending order, and the summary text
for each event_kind we support.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI

# Reuse the same component-builder + CSRF client used by the rest of the
# dashboard suite so this file stays in lockstep with that fixture.
from tests.test_dashboard import _CSRFTestClient, _make_components, _teardown


def _make_client_with_stores(components):
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return _CSRFTestClient(app)


class TestWorkplaceFeed:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        from src.host.orchestration import Tasks
        self.tasks_store = Tasks(
            db_path=os.path.join(self._tmpdir, "tasks.db"),
        )
        self.components["tasks_store"] = self.tasks_store

    def teardown_method(self):
        try:
            self.tasks_store.close()
        except Exception:
            pass
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        os.environ.pop("OPENLEGION_ORCHESTRATION_TASKS_V2", None)

    def _client(self, *, v2: bool):
        if v2:
            os.environ["OPENLEGION_ORCHESTRATION_TASKS_V2"] = "1"
        else:
            os.environ["OPENLEGION_ORCHESTRATION_TASKS_V2"] = "0"
        return _make_client_with_stores(self.components)

    # ── Disabled / empty state ────────────────────────────────────

    def test_feed_empty_state_when_v2_off(self):
        client = self._client(v2=False)
        resp = client.get("/dashboard/api/workplace/feed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["feed"] == []

    def test_feed_empty_when_no_events(self):
        client = self._client(v2=True)
        resp = client.get("/dashboard/api/workplace/feed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["feed"] == []

    # ── Shape + summary ──────────────────────────────────────────

    def test_feed_returns_created_event(self):
        client = self._client(v2=True)
        rec = self.tasks_store.create(
            creator="operator", assignee="writer-1",
            title="Draft Q4 plan", project_id="growth",
        )
        resp = client.get("/dashboard/api/workplace/feed")
        assert resp.status_code == 200
        data = resp.json()
        feed = data["feed"]
        assert len(feed) == 1
        ev = feed[0]
        assert ev["event_type"] == "created"
        assert ev["task_id"] == rec["id"]
        assert ev["task_title"] == "Draft Q4 plan"
        assert ev["project_id"] == "growth"
        assert ev["actor"] == "operator"
        assert ev["assignee"] == "writer-1"
        assert "writer-1" in ev["summary"]
        assert "Draft Q4 plan" in ev["summary"]
        assert "growth" in ev["summary"]

    def test_feed_summarizes_status_done(self):
        client = self._client(v2=True)
        rec = self.tasks_store.create(
            creator="op", assignee="writer", title="Task A",
        )
        self.tasks_store.update_status(rec["id"], "working", actor="writer")
        self.tasks_store.update_status(rec["id"], "done", actor="writer")

        resp = client.get("/dashboard/api/workplace/feed")
        feed = resp.json()["feed"]
        # Most recent first — done event should be at the top.
        kinds = [(e["event_type"], e.get("task_status")) for e in feed]
        assert kinds[0][0] == "status_changed"
        assert "completed" in feed[0]["summary"]

    def test_feed_summarizes_blocked_with_note(self):
        client = self._client(v2=True)
        rec = self.tasks_store.create(
            creator="op", assignee="reviewer-1", title="Edit copy",
        )
        self.tasks_store.update_status(rec["id"], "working", actor="reviewer-1")
        self.tasks_store.update_status(
            rec["id"], "blocked", actor="reviewer-1",
            blocker_note="needs SEO data",
        )

        resp = client.get("/dashboard/api/workplace/feed")
        feed = resp.json()["feed"]
        assert "blocked" in feed[0]["summary"]
        assert "needs SEO data" in feed[0]["summary"]

    # ── Filtering + ordering ─────────────────────────────────────

    def test_feed_filters_by_project_id(self):
        client = self._client(v2=True)
        self.tasks_store.create(
            creator="op", assignee="a", title="growth-task",
            project_id="growth",
        )
        self.tasks_store.create(
            creator="op", assignee="a", title="ops-task",
            project_id="ops",
        )

        resp = client.get(
            "/dashboard/api/workplace/feed?project_id=growth",
        )
        feed = resp.json()["feed"]
        assert all(e["project_id"] == "growth" for e in feed)
        titles = {e["task_title"] for e in feed}
        assert "growth-task" in titles
        assert "ops-task" not in titles

    def test_feed_ordered_descending_by_timestamp(self):
        client = self._client(v2=True)
        rec_a = self.tasks_store.create(
            creator="op", assignee="a", title="first",
        )
        rec_b = self.tasks_store.create(
            creator="op", assignee="a", title="second",
        )
        self.tasks_store.update_status(rec_a["id"], "working", actor="a")

        resp = client.get("/dashboard/api/workplace/feed")
        feed = resp.json()["feed"]
        # The status_changed on rec_a happened last, so it's first.
        assert feed[0]["task_id"] == rec_a["id"]
        assert feed[0]["event_type"] == "status_changed"
        # Timestamps must be monotonically non-increasing.
        for i in range(len(feed) - 1):
            assert feed[i]["timestamp"] >= feed[i + 1]["timestamp"]
        # The two created events for rec_a / rec_b are still in there.
        ids = {e["task_id"] for e in feed if e["event_type"] == "created"}
        assert ids == {rec_a["id"], rec_b["id"]}

    # ── Limit clamping ────────────────────────────────────────────

    def test_feed_limit_clamped_to_500(self):
        """A bogus limit doesn't blow up — server caps to 500."""
        client = self._client(v2=True)
        # Just one task so we're really only checking the response shape;
        # the cap matters at the SQL layer and shouldn't error.
        self.tasks_store.create(creator="op", assignee="a", title="t")
        resp = client.get("/dashboard/api/workplace/feed?limit=10000")
        assert resp.status_code == 200
        assert len(resp.json()["feed"]) <= 500

    def test_feed_limit_minimum_one(self):
        client = self._client(v2=True)
        self.tasks_store.create(creator="op", assignee="a", title="t1")
        self.tasks_store.create(creator="op", assignee="a", title="t2")
        # Negative values clamp to the minimum (1) — bogus client input
        # shouldn't surface as a 500. ``limit=0`` falls back to the
        # default per ``or`` semantics, matching outputs/tasks routes.
        resp = client.get("/dashboard/api/workplace/feed?limit=-1")
        assert resp.status_code == 200
        assert len(resp.json()["feed"]) == 1

    def test_feed_respects_explicit_limit(self):
        client = self._client(v2=True)
        for i in range(5):
            self.tasks_store.create(
                creator="op", assignee="a", title=f"t{i}",
            )
        resp = client.get("/dashboard/api/workplace/feed?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["feed"]) == 2


class TestFeedEventSummarizer:
    """Direct unit coverage for ``_summarize_task_event``.

    Workplace task drill-in (PR 4) emits ``task_outcome`` events the feed
    needs to render in plain English. Exercise the summarizer directly so
    PR 5 doesn't depend on PR 4's ``Tasks.set_outcome`` API to verify the
    branch.
    """

    def test_task_outcome_first_rating(self):
        from src.dashboard.server import _summarize_task_event

        out = _summarize_task_event(
            event_kind="task_outcome",
            actor="operator",
            title="Draft hero",
            project_id="growth",
            assignee="writer-1",
            blocker_note="",
            payload={"outcome": "accepted", "previous_outcome": None},
        )
        assert "operator" in out
        assert "rated" in out
        assert "Draft hero" in out
        assert "accepted" in out

    def test_task_outcome_re_rating_uses_re_rated_verb(self):
        from src.dashboard.server import _summarize_task_event

        out = _summarize_task_event(
            event_kind="task_outcome",
            actor="operator",
            title="Edit copy",
            project_id="",
            assignee="reviewer",
            blocker_note="",
            payload={"outcome": "accepted", "previous_outcome": "rejected"},
        )
        assert "re-rated" in out
        assert "accepted" in out
