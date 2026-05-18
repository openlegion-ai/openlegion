"""Endpoint tests for the Workplace task drill-in (PR 4).

Covers:

* GET /api/workplace/tasks/{id} — task + events + resolved artifacts
* GET /api/workplace/tasks/{id}/events — events-only polling endpoint
* 404 path when task is missing
* artifact resolution: text inlining, truncation cap, missing refs
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


class TestWorkplaceDrillIn:

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
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
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_drill_in_404_when_task_missing(self):
        resp = self.client.get("/dashboard/api/workplace/tasks/task_missing")
        assert resp.status_code == 404

    def test_drill_in_returns_task_events_artifacts(self):
        rec = self.tasks.create(
            creator="op", assignee="analyst", title="dig X",
            description="please research X",
        )
        # Drive to terminal so the events list has multiple entries.
        self.tasks.update_status(rec["id"], "working", actor="analyst")
        self.tasks.update_status(rec["id"], "done", actor="analyst")
        resp = self.client.get(
            f"/dashboard/api/workplace/tasks/{rec['id']}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["id"] == rec["id"]
        assert data["task"]["title"] == "dig X"
        assert data["task"]["status"] == "done"
        kinds = [e["event_kind"] for e in data["events"]]
        assert "created" in kinds
        assert "status_changed" in kinds
        assert isinstance(data["artifacts"], list)

    def test_drill_in_resolves_text_artifact_inline(self):
        rec = self.tasks.create(
            creator="op", assignee="analyst", title="t",
            artifact_refs=["output/analyst/ho1"],
        )
        self.components["blackboard"].write(
            "output/analyst/ho1",
            {"text": "the report body"},
            written_by="analyst",
        )
        resp = self.client.get(f"/dashboard/api/workplace/tasks/{rec['id']}")
        assert resp.status_code == 200
        artifacts = resp.json()["artifacts"]
        assert len(artifacts) == 1
        a = artifacts[0]
        assert a["ref"] == "output/analyst/ho1"
        assert a["kind"] == "text"
        assert a["content"] == "the report body"
        assert a["content_truncated"] is False

    def test_drill_in_resolves_dict_artifact_as_json(self):
        rec = self.tasks.create(
            creator="op", assignee="analyst", title="t",
            artifact_refs=["output/analyst/ho2"],
        )
        self.components["blackboard"].write(
            "output/analyst/ho2",
            {"summary": "ok", "score": 0.9},
            written_by="analyst",
        )
        resp = self.client.get(f"/dashboard/api/workplace/tasks/{rec['id']}")
        a = resp.json()["artifacts"][0]
        assert a["kind"] == "text"
        assert "summary" in a["content"]
        assert "0.9" in a["content"]

    def test_drill_in_marks_missing_artifact(self):
        rec = self.tasks.create(
            creator="op", assignee="analyst", title="t",
            artifact_refs=["output/never/written"],
        )
        resp = self.client.get(f"/dashboard/api/workplace/tasks/{rec['id']}")
        a = resp.json()["artifacts"][0]
        assert a["kind"] == "missing"

    def test_drill_in_truncates_huge_artifact(self):
        rec = self.tasks.create(
            creator="op", assignee="analyst", title="t",
            artifact_refs=["output/analyst/big"],
        )
        big_text = "x" * 50_000
        self.components["blackboard"].write(
            "output/analyst/big",
            {"text": big_text},
            written_by="analyst",
        )
        resp = self.client.get(f"/dashboard/api/workplace/tasks/{rec['id']}")
        a = resp.json()["artifacts"][0]
        assert a["kind"] == "text"
        assert a["content_truncated"] is True
        assert len(a["content"]) == 10_000

    def test_drill_in_404_when_unknown_task(self):
        resp = self.client.get("/dashboard/api/workplace/tasks/whatever")
        assert resp.status_code == 404

    def test_events_only_endpoint_returns_timeline(self):
        rec = self.tasks.create(creator="op", assignee="analyst", title="t")
        self.tasks.update_status(rec["id"], "working", actor="analyst")
        resp = self.client.get(
            f"/dashboard/api/workplace/tasks/{rec['id']}/events",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == rec["id"]
        kinds = [e["event_kind"] for e in data["events"]]
        assert "created" in kinds
        assert "status_changed" in kinds

    def test_events_only_endpoint_404_when_missing(self):
        resp = self.client.get("/dashboard/api/workplace/tasks/missing/events")
        assert resp.status_code == 404
