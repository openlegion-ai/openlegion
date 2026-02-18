"""Tests for WebhookManager: add, remove, dispatch, persistence."""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock

import pytest

from src.host.webhooks import WebhookManager


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
    async def test_test_nonexistent_hook(self):
        mgr = WebhookManager(config_path=self.config_path)
        result = await mgr.test_hook("nonexistent", {})
        assert result is None
