"""Tests for WebhookManager: add, remove, dispatch, persistence, instructions."""

from __future__ import annotations

import hashlib
import hmac
import shutil
import tempfile
from unittest.mock import AsyncMock

import pytest
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
