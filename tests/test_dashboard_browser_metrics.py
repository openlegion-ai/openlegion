"""Tests for the per-agent browser-metrics dashboard panel (Phase 7 §10.1).

Covers the new ``GET /dashboard/api/agents/{id}/browser/metrics`` endpoint
plus regression coverage for the existing WS event publication path
(``_poll_browser_metrics_once`` already has its own integration test in
``tests/test_browser_metrics_ingest.py`` — we only assert the wiring point
here so we'd notice if the contract drifts).

Also covers Phase 7 §10.2: the env-var read at
``src/browser/__main__._resolve_max_browsers`` honors
``OPENLEGION_BROWSER_MAX_CONCURRENT`` with the legacy ``MAX_BROWSERS``
fallback and a default of 5.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore

# ── Test fixtures (mirror tests/test_dashboard.py patterns) ─────────────────


def _make_components(tmp_path: str) -> dict:
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    agent_registry: dict[str, str] = {
        "alpha": "http://localhost:8401",
        "beta": "http://localhost:8402",
    }
    runtime_mock = MagicMock()
    # The endpoint short-circuits to a `service_unavailable` envelope when
    # this is unset; tests that exercise the upstream-fetch path patch it.
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = None
    runtime_mock.browser_auth_token = ""
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    health_monitor.register("alpha")
    health_monitor.register("beta")
    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": agent_registry,
        "runtime": runtime_mock,
    }


class _CSRFTestClient(TestClient):
    """TestClient that auto-injects the X-Requested-With CSRF header."""

    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            if "X-Requested-With" not in headers:
                headers["X-Requested-With"] = "XMLHttpRequest"
                kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


def _make_client(components: dict) -> TestClient:
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return _CSRFTestClient(app)


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()


# ── Endpoint behaviour ─────────────────────────────────────────────────────


class TestBrowserMetricsEndpoint:
    """Phase 7 §10.1 — per-agent browser-metrics endpoint."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_unknown_agent_returns_404(self):
        resp = self.client.get(
            "/dashboard/api/agents/does-not-exist/browser/metrics",
        )
        assert resp.status_code == 404

    def test_no_browser_service_returns_envelope(self):
        # runtime.browser_service_url is None — the endpoint must not 500;
        # it must return the §2.3 error envelope so the panel can degrade.
        resp = self.client.get(
            "/dashboard/api/agents/alpha/browser/metrics",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "service_unavailable"
        assert "Browser service" in body["error"]["message"]

    def test_returns_filtered_metrics(self):
        """Endpoint returns only payloads for the requested agent."""
        runtime = self.components["runtime"]
        runtime.browser_service_url = "http://browser:8500"
        runtime.browser_auth_token = "tok"

        # Build a fake upstream response with two agents' metrics.
        upstream = {
            "current_seq": 7,
            "boot_id": "boot-1",
            "metrics": [
                {"agent_id": "alpha", "seq": 5, "click_success": 3,
                 "click_fail": 0, "snapshot_count": 2,
                 "snapshot_bytes_p50": 1000, "snapshot_bytes_p95": 2000,
                 "nav_timeout": 0, "click_success_rate_100": 1.0,
                 "click_window_size": 3, "ts": 1000.0},
                {"agent_id": "beta", "seq": 6, "click_success": 1,
                 "click_fail": 0, "snapshot_count": 1,
                 "snapshot_bytes_p50": 500, "snapshot_bytes_p95": 800,
                 "nav_timeout": 0, "click_success_rate_100": 1.0,
                 "click_window_size": 1, "ts": 1001.0},
                {"agent_id": "alpha", "seq": 7, "click_success": 2,
                 "click_fail": 1, "snapshot_count": 1,
                 "snapshot_bytes_p50": 1100, "snapshot_bytes_p95": 1500,
                 "nav_timeout": 0, "click_success_rate_100": 0.83,
                 "click_window_size": 6, "ts": 1002.0},
            ],
        }

        captured: dict = {}

        async def _fake_fetch(client, service_url, auth_token, since_seq):
            captured["service_url"] = service_url
            captured["auth_token"] = auth_token
            captured["since_seq"] = since_seq
            return {"success": True, "data": upstream}

        with mock.patch(
            "src.dashboard.server._fetch_browser_metrics_upstream",
            side_effect=_fake_fetch,
        ):
            resp = self.client.get(
                "/dashboard/api/agents/alpha/browser/metrics",
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["current_seq"] == 7
        assert body["boot_id"] == "boot-1"
        # Only alpha's payloads — beta filtered out.
        assert len(body["metrics"]) == 2
        assert all(p["agent_id"] == "alpha" for p in body["metrics"])
        seqs = [p["seq"] for p in body["metrics"]]
        assert seqs == [5, 7]
        # Confirm runtime values were forwarded.
        assert captured["service_url"] == "http://browser:8500"
        assert captured["auth_token"] == "tok"
        assert captured["since_seq"] == 0

    def test_pagination_via_since_param(self):
        """``since=N`` is forwarded to the upstream helper unchanged."""
        runtime = self.components["runtime"]
        runtime.browser_service_url = "http://browser:8500"

        captured: dict = {}

        async def _fake_fetch(client, service_url, auth_token, since_seq):
            captured["since_seq"] = since_seq
            return {
                "success": True,
                "data": {
                    "current_seq": 42, "boot_id": "b", "metrics": [],
                },
            }

        with mock.patch(
            "src.dashboard.server._fetch_browser_metrics_upstream",
            side_effect=_fake_fetch,
        ):
            resp = self.client.get(
                "/dashboard/api/agents/alpha/browser/metrics?since=12",
            )

        assert resp.status_code == 200
        assert captured["since_seq"] == 12
        assert resp.json()["current_seq"] == 42

    def test_negative_since_clamped_to_zero(self):
        runtime = self.components["runtime"]
        runtime.browser_service_url = "http://browser:8500"

        captured: dict = {}

        async def _fake_fetch(client, service_url, auth_token, since_seq):
            captured["since_seq"] = since_seq
            return {
                "success": True,
                "data": {"current_seq": 0, "boot_id": "b", "metrics": []},
            }

        with mock.patch(
            "src.dashboard.server._fetch_browser_metrics_upstream",
            side_effect=_fake_fetch,
        ):
            resp = self.client.get(
                "/dashboard/api/agents/alpha/browser/metrics?since=-99",
            )
        assert resp.status_code == 200
        assert captured["since_seq"] == 0

    def test_upstream_error_returns_envelope(self):
        runtime = self.components["runtime"]
        runtime.browser_service_url = "http://browser:8500"

        async def _fake_fetch(*args, **kwargs):
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service unreachable",
                    "retry_after_ms": 60_000,
                },
            }

        with mock.patch(
            "src.dashboard.server._fetch_browser_metrics_upstream",
            side_effect=_fake_fetch,
        ):
            resp = self.client.get(
                "/dashboard/api/agents/alpha/browser/metrics",
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "service_unavailable"
        assert body["error"]["retry_after_ms"] == 60_000

    def test_no_metrics_for_agent_returns_empty_list(self):
        """When the upstream payload has data for *other* agents only."""
        runtime = self.components["runtime"]
        runtime.browser_service_url = "http://browser:8500"

        async def _fake_fetch(*args, **kwargs):
            return {
                "success": True,
                "data": {
                    "current_seq": 99,
                    "boot_id": "b",
                    "metrics": [
                        {"agent_id": "beta", "seq": 50,
                         "click_success": 1, "click_fail": 0},
                    ],
                },
            }

        with mock.patch(
            "src.dashboard.server._fetch_browser_metrics_upstream",
            side_effect=_fake_fetch,
        ):
            resp = self.client.get(
                "/dashboard/api/agents/alpha/browser/metrics",
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["metrics"] == []
        # current_seq still bumps so the client paginates forward.
        assert body["current_seq"] == 99

    def test_get_does_not_require_csrf_header(self):
        """GET endpoints are safe methods; no CSRF requirement."""
        # Bypass the auto-injecting test client by going through the bare
        # FastAPI app — the dashboard router is GET-exempt for X-Requested-With.
        from src.dashboard.server import create_dashboard_router

        router = create_dashboard_router(
            **self.components, mesh_port=8420,
        )
        app = FastAPI()
        app.include_router(router)
        plain = TestClient(app)
        resp = plain.get("/dashboard/api/agents/alpha/browser/metrics")
        # No CSRF rejection — service envelope (no browser configured).
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False


# ── Upstream helper coverage (network / HTTP / JSON errors) ────────────────


class TestFetchBrowserMetricsUpstream:
    """Direct coverage for ``_fetch_browser_metrics_upstream`` — the helper
    the endpoint delegates to. Tested separately so the §2.3 envelope is
    pinned at the boundary rather than re-asserted in every endpoint test.
    """

    def _run(self, coro):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_network_failure_returns_envelope(self):
        from src.dashboard.server import _fetch_browser_metrics_upstream

        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = self._run(_fetch_browser_metrics_upstream(
            client, "http://browser:8500", "tok", 0,
        ))
        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"
        assert result["error"]["retry_after_ms"] == 60_000

    def test_http_error_returns_envelope(self):
        from src.dashboard.server import _fetch_browser_metrics_upstream

        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json = MagicMock(return_value={})
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_resp)
        result = self._run(_fetch_browser_metrics_upstream(
            client, "http://browser:8500", "tok", 0,
        ))
        assert result["success"] is False
        assert "502" in result["error"]["message"]

    def test_bad_json_returns_envelope(self):
        from src.dashboard.server import _fetch_browser_metrics_upstream

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(side_effect=ValueError("bad json"))
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_resp)
        result = self._run(_fetch_browser_metrics_upstream(
            client, "http://browser:8500", "", 0,
        ))
        assert result["success"] is False
        assert "Malformed" in result["error"]["message"]

    def test_success_includes_data(self):
        from src.dashboard.server import _fetch_browser_metrics_upstream

        upstream = {"current_seq": 5, "boot_id": "b", "metrics": []}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=upstream)
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_resp)
        result = self._run(_fetch_browser_metrics_upstream(
            client, "http://browser:8500", "tok", 7,
        ))
        assert result["success"] is True
        assert result["data"] == upstream
        # Auth header forwarded.
        called_kwargs = client.get.await_args.kwargs
        assert called_kwargs["headers"]["Authorization"] == "Bearer tok"
        assert called_kwargs["params"] == {"since": 7}

    def test_no_auth_token_omits_header(self):
        from src.dashboard.server import _fetch_browser_metrics_upstream

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "current_seq": 0, "boot_id": "", "metrics": [],
        })
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_resp)
        self._run(_fetch_browser_metrics_upstream(
            client, "http://browser:8500", "", 0,
        ))
        called_kwargs = client.get.await_args.kwargs
        # Empty token → no Authorization header.
        assert "Authorization" not in (called_kwargs.get("headers") or {})


# ── WS event publication regression check ──────────────────────────────────


class TestBrowserMetricsEventBusContract:
    """Make sure the dashboard renderer's input contract stays intact.

    The Alpine panel reads from ``browser_metrics`` events on the EventBus
    via the existing ``/ws/events`` channel — the panel does NOT add a new
    transport. This test pins that wiring so a future refactor that drops
    the event-name doesn't silently break the dashboard.
    """

    def test_event_bus_emit_browser_metrics_routes_to_subscribers(self):
        from src.dashboard.events import EventBus

        bus = EventBus()
        captured: list[dict] = []

        # Fake subscription via direct buffer inspection — we don't need
        # the WebSocket here, just to confirm the event-name plumbing.
        bus.emit("browser_metrics", agent="alpha", data={
            "agent_id": "alpha", "seq": 1, "click_success": 1,
            "click_fail": 0, "snapshot_count": 1,
            "snapshot_bytes_p50": 100, "snapshot_bytes_p95": 200,
            "nav_timeout": 0, "click_success_rate_100": 1.0,
            "click_window_size": 1,
        })
        captured = [
            e for e in bus.recent_events()
            if e["type"] == "browser_metrics"
        ]
        assert len(captured) == 1
        assert captured[0]["agent"] == "alpha"
        assert captured[0]["data"]["click_success"] == 1


# ── §10.2: max_concurrent env-var resolution ───────────────────────────────


class TestMaxConcurrentEnvVar:
    """Phase 7 §10.2 — startup-only ``OPENLEGION_BROWSER_MAX_CONCURRENT``."""

    def test_default_when_unset(self):
        # Importing inside the test so each invocation gets a fresh read
        # against the patched environment.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENLEGION_BROWSER_MAX_CONCURRENT", None)
            os.environ.pop("MAX_BROWSERS", None)
            from src.browser.__main__ import _resolve_max_browsers
            assert _resolve_max_browsers() == 5

    def test_canonical_var_wins(self):
        with mock.patch.dict(os.environ, {
            "OPENLEGION_BROWSER_MAX_CONCURRENT": "12",
            "MAX_BROWSERS": "3",
        }):
            from src.browser.__main__ import _resolve_max_browsers
            assert _resolve_max_browsers() == 12

    def test_legacy_var_used_when_canonical_unset(self):
        with mock.patch.dict(os.environ, {"MAX_BROWSERS": "9"}, clear=False):
            os.environ.pop("OPENLEGION_BROWSER_MAX_CONCURRENT", None)
            from src.browser.__main__ import _resolve_max_browsers
            assert _resolve_max_browsers() == 9

    def test_clamped_to_min(self):
        with mock.patch.dict(os.environ, {
            "OPENLEGION_BROWSER_MAX_CONCURRENT": "0",
        }):
            from src.browser.__main__ import _resolve_max_browsers
            assert _resolve_max_browsers() == 1  # clamped up

    def test_clamped_to_max(self):
        with mock.patch.dict(os.environ, {
            "OPENLEGION_BROWSER_MAX_CONCURRENT": "9999",
        }):
            from src.browser.__main__ import _resolve_max_browsers
            assert _resolve_max_browsers() == 64  # clamped down

    def test_garbage_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {
            "OPENLEGION_BROWSER_MAX_CONCURRENT": "not-a-number",
        }):
            os.environ.pop("MAX_BROWSERS", None)
            from src.browser.__main__ import _resolve_max_browsers
            # flags.get_int falls back to its default (legacy) which is 5
            # when MAX_BROWSERS is also absent.
            assert _resolve_max_browsers() == 5

    def test_env_var_documented_in_known_flags(self):
        from src.browser.flags import KNOWN_FLAGS
        assert "OPENLEGION_BROWSER_MAX_CONCURRENT" in KNOWN_FLAGS
