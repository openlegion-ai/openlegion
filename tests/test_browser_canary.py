"""Tests for the stealth canary (Phase 2 §5.4)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


def _enable_canary(monkeypatch):
    monkeypatch.setenv("BROWSER_CANARY_ENABLED", "true")
    import src.browser.flags as flags
    flags._operator_settings = None


def _disable_canary(monkeypatch):
    monkeypatch.delenv("BROWSER_CANARY_ENABLED", raising=False)
    import src.browser.flags as flags
    flags._operator_settings = None


def _make_manager_mock():
    """Manager that satisfies run_canary's touch points."""
    mgr = MagicMock()
    mgr.boot_id = "boot-1"
    mgr.navigate = AsyncMock(return_value={"success": True})
    mgr.screenshot = AsyncMock(return_value={
        "success": True,
        "data": {"image_base64": "", "format": "png"},
    })
    mgr.evaluate = AsyncMock(return_value={
        "success": True, "data": {"result": {"pass": 10, "fail": 0}},
    })
    mgr.stop = AsyncMock()
    return mgr


class TestFeatureGate:
    @pytest.mark.asyncio
    async def test_raises_when_disabled(self, monkeypatch, tmp_path):
        from src.browser.canary import CanaryDisabledError, run_canary
        _disable_canary(monkeypatch)
        mgr = _make_manager_mock()
        with pytest.raises(CanaryDisabledError):
            await run_canary(
                mgr,
                state_path=tmp_path / "state.json",
                report_dir=tmp_path / "reports",
            )

    @pytest.mark.asyncio
    async def test_runs_when_enabled(self, monkeypatch, tmp_path):
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)
        mgr = _make_manager_mock()
        report = await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        assert report["agent_id"] == "canary-probe"
        assert len(report["scanners"]) == 4
        assert all("name" in s and "status" in s for s in report["scanners"])


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_second_run_rate_limited(self, monkeypatch, tmp_path):
        from src.browser.canary import CanaryRateLimitedError, run_canary
        _enable_canary(monkeypatch)
        mgr = _make_manager_mock()

        await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        with pytest.raises(CanaryRateLimitedError) as ex:
            await run_canary(
                mgr,
                state_path=tmp_path / "state.json",
                report_dir=tmp_path / "reports",
            )
        assert ex.value.retry_after_s > 0

    @pytest.mark.asyncio
    async def test_force_bypasses_rate_limit(self, monkeypatch, tmp_path):
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)
        mgr = _make_manager_mock()

        await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        # force=True: must NOT raise
        report = await run_canary(
            mgr, force=True,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        assert report["scanners"]

    @pytest.mark.asyncio
    async def test_state_persists_across_calls(self, monkeypatch, tmp_path):
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)
        state_path = tmp_path / "state.json"
        mgr = _make_manager_mock()
        await run_canary(
            mgr, state_path=state_path, report_dir=tmp_path / "r",
        )
        # Simulate a service restart — load state from disk.
        state = json.loads(state_path.read_text())
        assert state["last_run_ts"] > 0
        # Overall score present even when no numeric scanner scores yet.
        assert "last_overall_score" in state


class TestPerScannerResilience:
    @pytest.mark.asyncio
    async def test_one_scanner_timeout_does_not_stop_others(
        self, monkeypatch, tmp_path,
    ):
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)

        mgr = MagicMock()
        mgr.boot_id = "b"
        mgr.stop = AsyncMock()
        mgr.screenshot = AsyncMock(return_value={
            "success": True, "data": {"image_base64": "", "format": "png"},
        })
        mgr.evaluate = AsyncMock(return_value={
            "success": True, "data": {"result": {"pass": 0, "fail": 0}},
        })
        # First scanner raises; rest succeed.
        call_count = {"n": 0}

        async def sometimes_fail(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return {"success": True}

        mgr.navigate = AsyncMock(side_effect=sometimes_fail)

        report = await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        assert len(report["scanners"]) == 4
        # Exactly one erroring, three ok.
        errored = [s for s in report["scanners"] if s["status"] == "error"]
        assert len(errored) == 1

    @pytest.mark.asyncio
    async def test_nav_failure_sets_status_but_continues(
        self, monkeypatch, tmp_path,
    ):
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)

        mgr = _make_manager_mock()
        mgr.navigate = AsyncMock(return_value={
            "success": False, "error": "DNS failure",
        })

        report = await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        assert all(s["status"] == "nav_failed" for s in report["scanners"])

    @pytest.mark.asyncio
    async def test_stops_canary_instance_after_run(self, monkeypatch, tmp_path):
        """Explicit stop() after the sweep frees the canary's profile lock."""
        from src.browser.canary import run_canary
        _enable_canary(monkeypatch)

        mgr = _make_manager_mock()
        await run_canary(
            mgr,
            state_path=tmp_path / "state.json",
            report_dir=tmp_path / "reports",
        )
        # Called at start (best-effort cleanup) AND at end
        assert mgr.stop.await_count >= 1


class TestCanaryEndpoint:
    """Browser service ``POST /browser/_canary`` gating & responses."""

    def _mk_app(self, monkeypatch, manager):
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        from src.browser.server import create_browser_app
        return create_browser_app(manager)

    def test_endpoint_403_when_flag_off(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        _disable_canary(monkeypatch)
        mgr = _make_manager_mock()
        app = self._mk_app(monkeypatch, mgr)
        with TestClient(app) as client:
            resp = client.post("/browser/_canary")
        assert resp.status_code == 403

    def test_endpoint_429_when_rate_limited(self, monkeypatch, tmp_path):
        """Simulate a canary rate-limit by pre-seeding state."""
        from fastapi.testclient import TestClient
        _enable_canary(monkeypatch)
        # Point the canary's default paths at tmp to avoid touching /data
        monkeypatch.setattr(
            "src.browser.canary._DEFAULT_STATE_PATH",
            tmp_path / "state.json",
        )
        monkeypatch.setattr(
            "src.browser.canary._DEFAULT_REPORT_DIR",
            tmp_path / "reports",
        )
        (tmp_path / "state.json").write_text(
            json.dumps({"last_run_ts": time.time()}),
        )
        mgr = _make_manager_mock()
        app = self._mk_app(monkeypatch, mgr)
        with TestClient(app) as client:
            resp = client.post("/browser/_canary")
        assert resp.status_code == 429
        body = resp.json()
        assert "retry_after_s" in body["detail"]

    def test_endpoint_requires_auth_when_configured(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        monkeypatch.setenv("BROWSER_AUTH_TOKEN", "t0k")
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        _enable_canary(monkeypatch)
        from src.browser.server import create_browser_app
        mgr = _make_manager_mock()
        app = create_browser_app(mgr)
        with TestClient(app) as client:
            unauth = client.post("/browser/_canary")
            assert unauth.status_code == 401
