"""Tests for WebhookManager: add, remove, update, dispatch, persistence, instructions."""

from __future__ import annotations

import hashlib
import hmac
import shutil
import tempfile
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, HTTPException, Request
from starlette.testclient import TestClient

from src.host.webhooks import WebhookManager, _build_message


class TestBuildMessage:
    def test_default_suffix(self):
        hook = {"name": "gh-push"}
        msg = _build_message(hook, '{"action": "push"}')
        assert "Webhook 'gh-push' received:" in msg
        assert "Process this webhook payload." in msg
        assert '"action": "push"' in msg

    def test_custom_instructions(self):
        hook = {"name": "gh-push", "instructions": "Extract the commit SHA and update Jira."}
        msg = _build_message(hook, '{"sha": "abc123"}')
        assert "Extract the commit SHA and update Jira." in msg
        assert "Process this webhook payload." not in msg

    def test_test_label(self):
        hook = {"name": "gh-push"}
        msg = _build_message(hook, '{}', test=True)
        assert "(test)" in msg

    def test_truncates_body(self):
        hook = {"name": "x"}
        long_json = "a" * 5000
        msg = _build_message(hook, long_json)
        # 3000 chars of body + surrounding text
        assert len(long_json) > 3000
        assert "a" * 3000 in msg
        assert "a" * 3001 not in msg


class TestWebhookManager:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_hook(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="researcher", name="github-push")
        assert hook["agent"] == "researcher"
        assert hook["name"] == "github-push"
        assert "id" in hook
        assert len(mgr.hooks) == 1

    def test_add_hook_with_signature(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="signed", require_signature=True)
        assert "secret" in hook
        assert len(hook["secret"]) == 64  # 32 bytes hex

    def test_add_hook_with_instructions(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="instr", instructions="Do X then Y.")
        assert hook["instructions"] == "Do X then Y."

    def test_add_hook_blank_instructions_not_stored(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="blank", instructions="   ")
        assert "instructions" not in hook

    def test_persistence(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test-hook")

        mgr2 = WebhookManager(config_path=self.config_path)
        assert hook["id"] in mgr2.hooks

    def test_remove_hook(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test-hook")
        assert mgr.remove_hook(hook["id"])
        assert len(mgr.hooks) == 0

    def test_remove_nonexistent(self):
        mgr = WebhookManager(config_path=self.config_path)
        assert not mgr.remove_hook("nonexistent")

    def test_list_hooks(self):
        mgr = WebhookManager(config_path=self.config_path)
        mgr.add_hook(agent="a", name="hook1")
        mgr.add_hook(agent="b", name="hook2")
        hooks = mgr.list_hooks()
        assert len(hooks) == 2

    @pytest.mark.asyncio
    async def test_test_hook(self):
        dispatch = AsyncMock(return_value="processed")
        mgr = WebhookManager(config_path=self.config_path, dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="test", name="test-hook")

        result = await mgr.test_hook(hook["id"], {"event": "push"})
        assert result["status"] == "processed"
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_hook_with_instructions(self):
        dispatch = AsyncMock(return_value="done")
        mgr = WebhookManager(config_path=self.config_path, dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="test", name="instr-hook", instructions="Summarize the event.")

        await mgr.test_hook(hook["id"], {"event": "push"})
        msg = dispatch.call_args[0][1]
        assert "Summarize the event." in msg
        assert "Process this webhook payload." not in msg

    @pytest.mark.asyncio
    async def test_test_nonexistent_hook(self):
        mgr = WebhookManager(config_path=self.config_path)
        result = await mgr.test_hook("nonexistent", {})
        assert result is None


class TestWebhookManagerUpdate:
    """Tests for WebhookManager.update_hook()."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_update_name(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="old-name")
        result = mgr.update_hook(hook["id"], name="new-name")
        assert result["name"] == "new-name"
        assert mgr.hooks[hook["id"]]["name"] == "new-name"

    def test_update_agent(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="old-agent", name="test")
        result = mgr.update_hook(hook["id"], agent="new-agent")
        assert result["agent"] == "new-agent"
        assert mgr.hooks[hook["id"]]["agent"] == "new-agent"

    def test_set_instructions(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="test")
        assert "instructions" not in mgr.hooks[hook["id"]]
        result = mgr.update_hook(hook["id"], instructions="Do something.")
        assert result["instructions"] == "Do something."
        assert mgr.hooks[hook["id"]]["instructions"] == "Do something."

    def test_clear_instructions(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="test", instructions="Keep this.")
        assert mgr.hooks[hook["id"]]["instructions"] == "Keep this."
        result = mgr.update_hook(hook["id"], instructions="")
        assert "instructions" not in result
        assert "instructions" not in mgr.hooks[hook["id"]]

    def test_nonexistent_hook(self):
        mgr = WebhookManager(config_path=self.config_path)
        assert mgr.update_hook("nonexistent", name="x") is None

    def test_persistence(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="original")
        mgr.update_hook(hook["id"], name="updated")

        mgr2 = WebhookManager(config_path=self.config_path)
        assert mgr2.hooks[hook["id"]]["name"] == "updated"

    def test_return_is_copy(self):
        """Mutating the returned dict must not affect stored state."""
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="test")
        result = mgr.update_hook(hook["id"], name="updated")
        result["name"] = "tampered"
        assert mgr.hooks[hook["id"]]["name"] == "updated"

    def test_partial_update_preserves_other_fields(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="a", name="test", instructions="Keep me.")
        mgr.update_hook(hook["id"], name="renamed")
        stored = mgr.hooks[hook["id"]]
        assert stored["name"] == "renamed"
        assert stored["agent"] == "a"
        assert stored["instructions"] == "Keep me."


class TestWebhookRouter:
    """Test the FastAPI router created by WebhookManager."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_app(self, dispatch_fn=None):
        from fastapi import FastAPI

        mgr = WebhookManager(config_path=self.config_path, dispatch_fn=dispatch_fn)
        app = FastAPI()
        app.include_router(mgr.create_router())
        return app, mgr

    def test_receive_webhook_json(self):
        dispatch = AsyncMock(return_value="ok")
        app, mgr = self._make_app(dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="a", name="test")
        client = TestClient(app)

        resp = client.post(f"/webhook/hook/{hook['id']}", json={"foo": "bar"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processed"
        dispatch.assert_called_once()
        msg = dispatch.call_args[0][1]
        assert '"foo": "bar"' in msg

    def test_receive_webhook_unknown_id(self):
        app, _ = self._make_app()
        client = TestClient(app)
        resp = client.post("/webhook/hook/nonexistent", json={})
        assert resp.status_code == 404

    def test_receive_webhook_non_json_body(self):
        dispatch = AsyncMock(return_value="ok")
        app, mgr = self._make_app(dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="a", name="raw")
        client = TestClient(app)

        resp = client.post(
            f"/webhook/hook/{hook['id']}",
            content=b"not json",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 200
        msg = dispatch.call_args[0][1]
        assert "not json" in msg

    def test_hmac_signature_valid(self):
        dispatch = AsyncMock(return_value="ok")
        app, mgr = self._make_app(dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="a", name="signed", require_signature=True)
        secret = hook["secret"]
        client = TestClient(app)

        body = b'{"event": "push"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        resp = client.post(
            f"/webhook/hook/{hook['id']}",
            content=body,
            headers={"Content-Type": "application/json", "x-webhook-signature": sig},
        )
        assert resp.status_code == 200

    def test_hmac_signature_invalid(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="signed", require_signature=True)
        client = TestClient(app)

        resp = client.post(
            f"/webhook/hook/{hook['id']}",
            json={"event": "push"},
            headers={"x-webhook-signature": "bad"},
        )
        assert resp.status_code == 401

    def test_call_count_increments(self):
        dispatch = AsyncMock(return_value="ok")
        app, mgr = self._make_app(dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="a", name="counter")
        client = TestClient(app)

        client.post(f"/webhook/hook/{hook['id']}", json={})
        client.post(f"/webhook/hook/{hook['id']}", json={})
        assert mgr.hooks[hook["id"]]["call_count"] == 2

    def test_instructions_in_dispatched_message(self):
        dispatch = AsyncMock(return_value="ok")
        app, mgr = self._make_app(dispatch_fn=dispatch)
        hook = mgr.add_hook(agent="a", name="instr", instructions="Log the event to Slack.")
        client = TestClient(app)

        client.post(f"/webhook/hook/{hook['id']}", json={"type": "deploy"})
        msg = dispatch.call_args[0][1]
        assert "Log the event to Slack." in msg
        assert "Process this webhook payload." not in msg

    def test_no_dispatch_fn(self):
        app, mgr = self._make_app(dispatch_fn=None)
        hook = mgr.add_hook(agent="a", name="no-dispatch")
        client = TestClient(app)

        resp = client.post(f"/webhook/hook/{hook['id']}", json={"x": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" not in data  # response redacted for security (U14-4)
        assert mgr.hooks[hook["id"]]["call_count"] == 1

    def test_hmac_missing_header_when_required(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="signed", require_signature=True)
        client = TestClient(app)

        # No x-webhook-signature header at all
        resp = client.post(f"/webhook/hook/{hook['id']}", json={"event": "push"})
        assert resp.status_code == 401

    def test_list_hooks_does_not_mutate_stored_data(self):
        """Callers mutating returned dicts must not pollute the manager's state."""
        app, mgr = self._make_app()
        mgr.add_hook(agent="a", name="test")
        hooks = mgr.list_hooks()
        # Simulate what a caller might do
        for h in hooks:
            h["url"] = "http://example.com/webhook/hook/" + h["id"]

        # The stored hook should NOT have the url field
        stored = list(mgr.hooks.values())[0]
        assert "url" not in stored


class TestUpdateHook:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_update_name(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="old-name")
        updated = mgr.update_hook(hook["id"], name="new-name")
        assert updated["name"] == "new-name"
        assert mgr.hooks[hook["id"]]["name"] == "new-name"

    def test_update_agent(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="old-agent", name="test")
        updated = mgr.update_hook(hook["id"], agent="new-agent")
        assert updated["agent"] == "new-agent"

    def test_update_instructions_set_change_clear(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        # Set
        updated = mgr.update_hook(hook["id"], instructions="Do X.")
        assert updated["instructions"] == "Do X."
        # Change
        updated = mgr.update_hook(hook["id"], instructions="Do Y.")
        assert updated["instructions"] == "Do Y."
        # Clear
        updated = mgr.update_hook(hook["id"], instructions="")
        assert "instructions" not in updated

    def test_enable_signature(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        assert "secret" not in hook
        updated = mgr.update_hook(hook["id"], require_signature=True)
        assert "secret" in updated
        assert len(updated["secret"]) == 64

    def test_disable_signature(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test", require_signature=True)
        assert "secret" in hook
        updated = mgr.update_hook(hook["id"], require_signature=False)
        assert "secret" not in updated
        assert "secret" not in mgr.hooks[hook["id"]]

    def test_regenerate_secret(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test", require_signature=True)
        old_secret = hook["secret"]
        updated = mgr.update_hook(hook["id"], regenerate_secret=True)
        assert "secret" in updated
        assert updated["secret"] != old_secret

    def test_regenerate_secret_ignored_when_unsigned(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        updated = mgr.update_hook(hook["id"], regenerate_secret=True)
        assert "secret" not in updated
        assert "secret" not in mgr.hooks[hook["id"]]

    def test_update_preserves_call_count(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        mgr.hooks[hook["id"]]["call_count"] = 42
        updated = mgr.update_hook(hook["id"], name="renamed")
        assert updated["call_count"] == 42

    def test_update_nonexistent(self):
        mgr = WebhookManager(config_path=self.config_path)
        assert mgr.update_hook("nonexistent", name="x") is None

    def test_update_persists(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        mgr.update_hook(hook["id"], name="persisted", agent="new-agent")
        mgr2 = WebhookManager(config_path=self.config_path)
        assert mgr2.hooks[hook["id"]]["name"] == "persisted"
        assert mgr2.hooks[hook["id"]]["agent"] == "new-agent"

    def test_update_empty_name_rejected(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        with pytest.raises(ValueError):
            mgr.update_hook(hook["id"], name="")

    def test_update_empty_agent_rejected(self):
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="test", name="test")
        with pytest.raises(ValueError):
            mgr.update_hook(hook["id"], agent="")

    def test_validation_error_does_not_partially_mutate(self):
        """If agent validation fails, name should not be changed."""
        mgr = WebhookManager(config_path=self.config_path)
        hook = mgr.add_hook(agent="original-agent", name="original-name")
        with pytest.raises(ValueError):
            mgr.update_hook(hook["id"], name="new-name", agent="")
        # Name must remain unchanged
        assert mgr.hooks[hook["id"]]["name"] == "original-name"


class TestWebhookDashboardPatch:
    """Test the PATCH /api/webhooks/{hook_id} endpoint via create_dashboard_router."""

    def setup_method(self):
        import os
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self._tmpdir, "webhooks.json")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_app(self, dispatch_fn=None):
        import os
        from unittest.mock import MagicMock

        from fastapi import FastAPI

        from src.dashboard.events import EventBus
        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore

        tmp = self._tmpdir
        bb = Blackboard(db_path=os.path.join(tmp, "bb.db"))
        cost_tracker = CostTracker(db_path=os.path.join(tmp, "costs.db"))
        trace_store = TraceStore(db_path=os.path.join(tmp, "traces.db"))
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

        mgr = WebhookManager(config_path=self.config_path, dispatch_fn=dispatch_fn)
        dashboard_router = create_dashboard_router(
            blackboard=bb,
            health_monitor=health_monitor,
            cost_tracker=cost_tracker,
            trace_store=trace_store,
            event_bus=event_bus,
            agent_registry={},
            mesh_port=8420,
            webhook_manager=mgr,
        )
        app = FastAPI()
        app.include_router(dashboard_router)
        return app, mgr

    def test_patch_via_dashboard_success(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="original")
        client = TestClient(app)

        resp = client.patch(
            f"/dashboard/api/webhooks/{hook['id']}",
            json={"name": "updated", "instructions": "Do X."},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] is True
        assert data["hook"]["name"] == "updated"
        assert data["hook"]["instructions"] == "Do X."
        assert "url" in data["hook"]

    def test_patch_via_dashboard_not_found(self):
        app, _ = self._make_app()
        client = TestClient(app)

        resp = client.patch(
            "/dashboard/api/webhooks/nonexistent",
            json={"name": "x"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 404

    def test_patch_via_dashboard_empty_body_rejected(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test")
        client = TestClient(app)

        resp = client.patch(
            f"/dashboard/api/webhooks/{hook['id']}",
            json={},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400

    def test_patch_via_dashboard_empty_name_rejected(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test")
        client = TestClient(app)

        resp = client.patch(
            f"/dashboard/api/webhooks/{hook['id']}",
            json={"name": ""},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400

    def test_patch_via_dashboard_enable_signature(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test")
        assert "secret" not in hook
        client = TestClient(app)

        resp = client.patch(
            f"/dashboard/api/webhooks/{hook['id']}",
            json={"require_signature": True},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "secret" in data["hook"]
        assert len(data["hook"]["secret"]) == 64

    def test_patch_via_dashboard_no_manager(self):
        import os
        from unittest.mock import MagicMock

        from fastapi import FastAPI

        from src.dashboard.events import EventBus
        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore

        tmp = self._tmpdir
        bb = Blackboard(db_path=os.path.join(tmp, "bb2.db"))
        cost_tracker = CostTracker(db_path=os.path.join(tmp, "costs2.db"))
        trace_store = TraceStore(db_path=os.path.join(tmp, "traces2.db"))
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

        dashboard_router = create_dashboard_router(
            blackboard=bb,
            health_monitor=health_monitor,
            cost_tracker=cost_tracker,
            trace_store=trace_store,
            event_bus=event_bus,
            agent_registry={},
            mesh_port=8420,
            webhook_manager=None,
        )
        app = FastAPI()
        app.include_router(dashboard_router)
        client = TestClient(app)

        resp = client.patch(
            "/dashboard/api/webhooks/some-id",
            json={"name": "x"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 503


class TestWebhookUpdateRouter:
    """Test the PATCH /api/webhooks/{hook_id} endpoint."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_app(self):
        """Build a minimal FastAPI app with webhook update endpoints."""
        from fastapi import FastAPI

        mgr = WebhookManager(config_path=self.config_path)

        app = FastAPI()
        api = APIRouter()

        @api.get("/api/webhooks")
        async def api_webhooks_list(request: Request) -> dict:
            hooks = mgr.list_hooks()
            base = str(request.base_url).rstrip("/")
            result = []
            for h in hooks:
                entry = {k: v for k, v in h.items() if k != "secret"}
                entry["url"] = f"{base}/webhook/hook/{h['id']}"
                entry["has_secret"] = "secret" in h
                result.append(entry)
            return {"webhooks": result}

        @api.patch("/api/webhooks/{hook_id}")
        async def api_webhooks_update(hook_id: str, request: Request) -> dict:
            body = await request.json()
            fields = {}
            for key in ("name", "agent", "instructions"):
                if key in body:
                    fields[key] = body[key]
            if "require_signature" in body:
                fields["require_signature"] = bool(body["require_signature"])
            if body.get("regenerate_secret"):
                fields["regenerate_secret"] = True
            if not fields:
                raise HTTPException(status_code=400, detail="No valid fields provided")
            try:
                updated = mgr.update_hook(hook_id, **fields)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if updated is None:
                raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
            base = str(request.base_url).rstrip("/")
            updated["url"] = f"{base}/webhook/hook/{updated['id']}"
            return {"updated": True, "hook": updated}

        app.include_router(api)
        return app, mgr

    def test_patch_webhook_success(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="original")
        client = TestClient(app)

        resp = client.patch(
            f"/api/webhooks/{hook['id']}",
            json={"name": "renamed", "agent": "b"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] is True
        assert data["hook"]["name"] == "renamed"
        assert data["hook"]["agent"] == "b"
        assert "url" in data["hook"]

    def test_patch_webhook_not_found(self):
        app, _ = self._make_app()
        client = TestClient(app)
        resp = client.patch("/api/webhooks/nonexistent", json={"name": "x"})
        assert resp.status_code == 404

    def test_patch_webhook_new_secret_in_response(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test")
        client = TestClient(app)

        resp = client.patch(
            f"/api/webhooks/{hook['id']}",
            json={"require_signature": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "secret" in data["hook"]
        assert len(data["hook"]["secret"]) == 64

    def test_patch_webhook_no_secret_when_unchanged(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test", require_signature=True)
        client = TestClient(app)

        resp = client.patch(
            f"/api/webhooks/{hook['id']}",
            json={"name": "renamed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "secret" not in data["hook"]

    def test_patch_webhook_empty_body(self):
        app, mgr = self._make_app()
        hook = mgr.add_hook(agent="a", name="test")
        client = TestClient(app)
        resp = client.patch(f"/api/webhooks/{hook['id']}", json={})
        assert resp.status_code == 400

    def test_list_includes_has_secret(self):
        app, mgr = self._make_app()
        mgr.add_hook(agent="a", name="unsigned")
        mgr.add_hook(agent="b", name="signed", require_signature=True)
        client = TestClient(app)

        resp = client.get("/api/webhooks")
        hooks = resp.json()["webhooks"]
        unsigned = next(h for h in hooks if h["name"] == "unsigned")
        signed = next(h for h in hooks if h["name"] == "signed")
        assert unsigned["has_secret"] is False
        assert signed["has_secret"] is True
        assert "secret" not in signed  # value still stripped
