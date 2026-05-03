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

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.orchestration import Tasks
from src.host.traces import TraceStore


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


def _make_client(components: dict, tasks_store) -> TestClient:
    from src.dashboard.server import create_dashboard_router
    router = create_dashboard_router(
        **components, mesh_port=8420, tasks_store=tasks_store,
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
        os.environ["OPENLEGION_ORCHESTRATION_TASKS_V2"] = "1"
        self.components = _make_components(self._tmp)
        self.tasks = Tasks(db_path=os.path.join(self._tmp, "tasks.db"))
        self.client = _make_client(self.components, self.tasks)

    def teardown_method(self):
        try:
            self.tasks.close()
        except Exception:
            pass
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        os.environ.pop("OPENLEGION_ORCHESTRATION_TASKS_V2", None)
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
            title="research X", project_id="research",
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
        assert new["project_id"] == "research"
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

    def test_outcome_non_terminal_returns_409(self):
        rec = self.tasks.create(creator="op", assignee="analyst", title="t")
        self.tasks.update_status(rec["id"], "working", actor="analyst")
        resp = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert resp.status_code == 409

    def test_outcome_double_set_returns_409(self):
        rec = _create_done_task(
            self.tasks, creator="op", assignee="analyst", title="t",
        )
        first = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "accepted", "feedback": ""},
        )
        assert first.status_code == 200
        second = self.client.post(
            f"/dashboard/api/workplace/tasks/{rec['id']}/outcome",
            json={"outcome": "rework", "feedback": "actually no"},
        )
        assert second.status_code == 409

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
