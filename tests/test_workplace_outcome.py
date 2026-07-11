"""Endpoint tests for the Workplace task outcome submission (PR 4).

Covers:

* POST /api/workplace/tasks/{id}/outcome happy path (accept / rework / reject)
* Rework spawns a linked task with the same assignee + project
* Validation: missing feedback for rework/reject, unknown outcome, oversize,
  non-terminal status, double-set
"""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.orchestration import Tasks
from src.host.traces import TraceStore
from src.host.track_record import TrackRecordStore


def _make_components(tmp_path: str) -> dict:
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = None
    runtime_mock.browser_auth_token = ""
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": {},
    }


class _CSRFTestClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            if "X-Requested-With" not in headers:
                headers["X-Requested-With"] = "XMLHttpRequest"
                kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


def _make_client(components: dict, tasks_store, track_record_store=None) -> TestClient:
    from src.dashboard.server import create_dashboard_router
    router = create_dashboard_router(
        **components,
        mesh_port=8420,
        tasks_store=tasks_store,
        track_record_store=track_record_store,
    )
    app = FastAPI()
    app.include_router(router)
    return _CSRFTestClient(app)


def _create_done_task(tasks: Tasks, **kw) -> dict:
    rec = tasks.create(**kw)
    tasks.update_status(rec["id"], "working", actor=rec["assignee"])
    tasks.update_status(rec["id"], "done", actor=rec["assignee"])
    return tasks.get(rec["id"])


class TestWorkplaceOutcome:

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.components = _make_components(self._tmp)
        self.tasks = Tasks(db_path=os.path.join(self._tmp, "tasks.db"))
        self.track_record = TrackRecordStore(":memory:")
        self.client = _make_client(
            self.components, self.tasks, track_record_store=self.track_record,
        )

    def teardown_method(self):
        try:
            self.tasks.close()
        except Exception:
            pass
        self.track_record.close()
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_outcome_accept_no_feedback(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["task"]["outcome"] == "accepted"
        assert "rework_task_id" not in body

    def test_outcome_rework_spawns_linked_task(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst",
            title="research X", team_id="research",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={
                "outcome": "rework",
                "feedback": "go deeper on the legal angle",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"]["outcome"] == "rework"
        assert body["rework_task_id"]
        assert body["rework_assignee"] == "analyst"
        # Verify the new task was actually persisted with the link.
        new = self.tasks.get(body["rework_task_id"])
        assert new is not None
        assert new["previous_task_id"] == rec["id"]
        assert new["assignee"] == "analyst"
        assert new["team_id"] == "research"
        assert new["title"].startswith("Rework: ")
        assert new["description"] == "go deeper on the legal angle"

    def test_outcome_rejected_with_feedback(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rejected", "feedback": "off-topic"},
        )
        assert resp.status_code == 200
        assert resp.json()["task"]["outcome"] == "rejected"

    def test_outcome_rework_requires_feedback(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rework", "feedback": "  "},
        )
        assert resp.status_code == 400

    def test_outcome_rejected_requires_feedback(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rejected", "feedback": ""},
        )
        assert resp.status_code == 400

    def test_outcome_unknown_value_rejected(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "meh", "feedback": "x"},
        )
        assert resp.status_code == 400

    def test_outcome_acknowledged_no_feedback_allowed(self):
        """The new 'acknowledged' outcome is the neutral ➖ rating: it
        records that the user reviewed the work without judgement,
        accepts an empty feedback string, and does NOT spawn a rework
        task."""
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "acknowledged", "feedback": ""},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["task"]["outcome"] == "acknowledged"
        assert "rework_task_id" not in body

    def test_outcome_acknowledged_does_not_spawn_rework(self):
        """Regression guard: only outcome=='rework' creates a follow-up
        task. Accept / acknowledged / rejected all stay leaf nodes.
        Without this we could silently regress to over-spawning when the
        endpoint logic is touched again."""
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst",
            title="research X", team_id="research",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "acknowledged", "feedback": "noted"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"]["outcome"] == "acknowledged"
        assert "rework_task_id" not in body
        assert "rework_assignee" not in body
        # No "Rework: ..." task was persisted (check pending inbox AND
        # terminal history so we'd see a stray rework either way).
        rework_titles = [
            t["title"] for t in self.tasks.list_inbox(
                "analyst", include_terminal=True,
            )
            if t["title"].startswith("Rework: ")
        ]
        assert rework_titles == []

    def test_rerate_from_rework_to_acknowledged_leaves_spawned_task(self):
        """Documents the per-question-2a deferred behavior: re-rating a
        previously-rework task as 👍 / ➖ does NOT auto-cancel the
        spawned rework task. The user must cancel it manually. This
        test exists so a future change that adds auto-cancel breaks
        this test loudly rather than silently changing semantics.
        """
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
            team_id="research",
        )
        # Initial rating: rework — spawns a follow-up task.
        first = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rework", "feedback": "tighten the lead"},
        )
        assert first.status_code == 200
        rework_id = first.json().get("rework_task_id")
        assert rework_id, "first rework rating should spawn a task"
        # Re-rate as acknowledged — should NOT cancel the spawned task.
        second = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "acknowledged", "feedback": ""},
        )
        assert second.status_code == 200
        assert second.json()["task"]["outcome"] == "acknowledged"
        # The previously-spawned rework task is still in the DB and in
        # a non-terminal state. (Auto-cancel-on-rerate is the deferred
        # follow-up.)
        rework_task = self.tasks.get(rework_id)
        assert rework_task is not None
        assert rework_task["status"] not in ("cancelled",)

    def test_outcome_non_terminal_returns_409(self):
        rec = self.tasks.create(creator="op", assignee="analyst", title="t")
        self.tasks.update_status(rec["id"], "working", actor="analyst")
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 409

    def test_outcome_re_rating_overwrites_latest(self):
        """Outcomes are write-many: a re-rating updates ``tasks.outcome``
        in place and appends a fresh ``task_outcome`` audit row so the
        full history stays inspectable."""
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        first = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rejected", "feedback": "wrong"},
        )
        assert first.status_code == 200
        assert first.json()["task"]["outcome"] == "rejected"
        second = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": "fine on reread"},
        )
        assert second.status_code == 200
        body = second.json()
        assert body["task"]["outcome"] == "accepted"
        assert body["task"]["feedback_text"] == "fine on reread"
        events = self.tasks.list_events(rec["id"])
        outcome_events = [e for e in events if e["event_kind"] == "task_outcome"]
        assert len(outcome_events) == 2

    def test_outcome_oversize_feedback_rejected(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": "x" * 5000},
        )
        assert resp.status_code == 400

    def test_outcome_404_for_missing_task(self):
        resp = self.client.post(
            "/dashboard/api/workplace/tasks/task_missing/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 404

    def test_outcome_non_dict_body_rejected(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json=["accepted"],
        )
        assert resp.status_code == 400

    # ── plan §8 #18 — durable track record write ──────────────────

    def test_outcome_writes_track_record_event_as_human(self):
        """The dashboard outcome endpoint is the HUMAN-driven path (plan
        §8 #18) — its track-record write must be tagged
        ``rater_kind="human"`` so it counts at full weight toward
        earned-autonomy scoring."""
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst",
            title="t", team_id="research",
        )
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 200
        events = self.track_record.recent_events("analyst")
        assert len(events) == 1
        event = events[0]
        assert event["source"] == "task_outcome"
        assert event["ref_id"] == rec["id"]
        assert event["outcome"] == "accepted"
        assert event["rater_kind"] == "human"
        assert event["team_id"] == "research"
        autonomy_counts = self.track_record.counts_for_agent(
            "analyst", rater_kinds=("human", "system"),
        )
        assert autonomy_counts == {"task_outcome": {"accepted": 1}}

    def test_outcome_track_record_write_failure_does_not_fail_request(self):
        """A ledger write failure must NEVER fail the source operation
        (plan §8 #18's best-effort contract)."""
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        self.track_record.close()  # subsequent .record() calls now raise
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["task"]["outcome"] == "accepted"

    def test_outcome_absent_track_record_store_is_a_noop(self):
        """Existing dashboard test constructions that don't wire a
        track_record_store (the default) keep working unchanged."""
        client = _make_client(self.components, self.tasks)  # no track_record_store
        rec = _create_done_task(
            self.tasks, creator="op", assignee="writer2", title="t",
        )
        resp = client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 200
