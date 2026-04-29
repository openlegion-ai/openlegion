"""Tests for Phase 10 §24 — per-tenant CAPTCHA cost rollup + alerts + CSV export.

Covers:
  * ``_tenant_for(agent_id)`` resolves via the existing project-membership
    map, caches LRU(256), and returns ``None`` for unprojected agents.
  * ``get_tenant_total`` sums correctly across multiple agents.
  * Cross-tenant isolation — tenant A's agents do NOT appear in tenant B's
    rollup, and vice versa.
  * ``since`` filter behaves as documented for in-memory state (current
    month falls through to live total; older months drop to zero).
  * CSV export endpoint shape — header row, per-agent rows in sorted
    order, ``__tenant_total__`` summary row, period-start column.
  * CSV endpoint requires auth (no ol_session cookie in production), GET
    is allowed without ``X-Requested-With`` (CSRF only on state changes).
  * ``record_tenant_threshold_alerts`` fires once per crossing per month
    at the 50/80/100% gates; subsequent calls in the same month do not
    re-fire.
  * Month rollover resets the fired-pct memory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.browser import captcha_cost_counter as cost


@pytest.fixture(autouse=True)
async def _isolate_state(tmp_path, monkeypatch):
    """Each test starts with fresh state + a tmp snapshot path."""
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    cost.reset_threshold_state()
    cost.reset_tenant_cache()
    yield
    await cost.reset()
    cost.reset_threshold_state()
    cost.reset_tenant_cache()


# ── Helper: stub the project membership lookup ─────────────────────────────


def _patch_projects(membership: dict[str, str]):
    """Return a context manager that swaps ``cli.config._load_config``.

    ``membership`` maps agent_id → project_name. ``_load_config`` returns
    a dict whose ``_agent_projects`` key matches the production shape
    (built by ``_load_config`` from ``config/projects/<name>/metadata.yaml``).
    """
    fake_cfg = {"_agent_projects": dict(membership)}
    return patch(
        "src.cli.config._load_config",
        return_value=fake_cfg,
    )


# ── _tenant_for ────────────────────────────────────────────────────────────


class TestTenantFor:
    def test_resolves_member_to_project(self):
        with _patch_projects({"alpha": "tenant-a", "beta": "tenant-b"}):
            cost.reset_tenant_cache()
            assert cost._tenant_for("alpha") == "tenant-a"
            assert cost._tenant_for("beta") == "tenant-b"

    def test_unknown_agent_returns_none(self):
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            assert cost._tenant_for("unknown") is None

    def test_empty_agent_id_returns_none(self):
        with _patch_projects({}):
            cost.reset_tenant_cache()
            assert cost._tenant_for("") is None

    def test_lru_cache_in_use(self):
        """Successive calls with the same agent_id hit the cache."""
        cost.reset_tenant_cache()
        call_count = {"n": 0}

        def fake_loader():
            call_count["n"] += 1
            return {"_agent_projects": {"alpha": "tenant-a"}}

        with patch("src.cli.config._load_config", side_effect=fake_loader):
            cost._tenant_for("alpha")
            cost._tenant_for("alpha")
            cost._tenant_for("alpha")
        # Exactly one underlying load — subsequent calls hit the LRU cache.
        assert call_count["n"] == 1

    def test_cache_invalidated_by_reset_tenant_cache(self):
        cost.reset_tenant_cache()
        with _patch_projects({"alpha": "tenant-a"}):
            assert cost._tenant_for("alpha") == "tenant-a"
        cost.reset_tenant_cache()
        with _patch_projects({"alpha": "tenant-b"}):
            assert cost._tenant_for("alpha") == "tenant-b"

    def test_loader_failure_returns_none(self):
        """If ``_load_config`` raises (missing config dir), tenant=None."""
        cost.reset_tenant_cache()
        with patch(
            "src.cli.config._load_config",
            side_effect=RuntimeError("no config"),
        ):
            assert cost._tenant_for("alpha") is None


# ── get_tenant_total ───────────────────────────────────────────────────────


class TestGetTenantTotal:
    @pytest.mark.asyncio
    async def test_sums_across_multiple_agents(self):
        with _patch_projects({
            "alpha": "tenant-a",
            "beta": "tenant-a",
            "gamma": "tenant-a",
        }):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("beta", 50)
            await cost.add_cost("gamma", 25)
            assert await cost.get_tenant_total("tenant-a") == 175

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(self):
        """Tenant A's agents do NOT contribute to tenant B's total."""
        with _patch_projects({
            "alpha": "tenant-a",
            "beta": "tenant-b",
            "gamma": "tenant-a",
        }):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("beta", 999)
            await cost.add_cost("gamma", 50)
            assert await cost.get_tenant_total("tenant-a") == 150
            assert await cost.get_tenant_total("tenant-b") == 999

    @pytest.mark.asyncio
    async def test_unknown_tenant_returns_zero(self):
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            assert await cost.get_tenant_total("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_unprojected_agents_excluded(self):
        """Agents whose ``_tenant_for`` returns ``None`` are not summed."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("orphan", 50)  # not in projects map
            assert await cost.get_tenant_total("tenant-a") == 100

    @pytest.mark.asyncio
    async def test_since_filter_current_month_returns_live_total(self):
        with _patch_projects({"alpha": "tenant-a", "beta": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("beta", 50)
            now = datetime.now(timezone.utc)
            seven_days_ago = now - timedelta(days=7)
            # ``since`` falling within the current month → live total.
            assert await cost.get_tenant_total(
                "tenant-a", since=seven_days_ago,
            ) == 150

    @pytest.mark.asyncio
    async def test_since_filter_past_month_returns_zero(self):
        """``since`` rooted in a past calendar month returns zero — the
        in-memory state is current-month only."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            now = datetime.now(timezone.utc)
            # Step back at least one full month.
            two_months_ago = (
                now.replace(day=1) - timedelta(days=45)
            )
            assert await cost.get_tenant_total(
                "tenant-a", since=two_months_ago,
            ) == 0

    @pytest.mark.asyncio
    async def test_empty_tenant_id_returns_zero(self):
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            assert await cost.get_tenant_total("") == 0


# ── get_tenant_breakdown ───────────────────────────────────────────────────


class TestGetTenantBreakdown:
    @pytest.mark.asyncio
    async def test_returns_per_agent_dict(self):
        with _patch_projects({
            "alpha": "tenant-a",
            "beta": "tenant-a",
            "gamma": "tenant-b",
        }):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("beta", 50)
            await cost.add_cost("gamma", 25)
            breakdown = await cost.get_tenant_breakdown("tenant-a")
            assert breakdown == {"alpha": 100, "beta": 50}
            # Cross-tenant: gamma is invisible from tenant-a.
            assert "gamma" not in breakdown

    @pytest.mark.asyncio
    async def test_empty_tenant_returns_empty_dict(self):
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            assert await cost.get_tenant_breakdown("nonexistent") == {}


# ── record_tenant_threshold_alerts ─────────────────────────────────────────


class TestThresholdAlerts:
    @pytest.mark.asyncio
    async def test_fires_at_50_80_100_pct(self):
        """Three crossings produce three single-fire events."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            cap_millicents = 1000
            captured: list[dict] = []

            def emit(payload: dict) -> None:
                captured.append(payload)

            # 50% crossing
            await cost.add_cost("alpha", 500)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, emit,
            )
            assert fired == [50]

            # 80% crossing
            await cost.add_cost("alpha", 300)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, emit,
            )
            assert fired == [80]

            # 100% crossing
            await cost.add_cost("alpha", 200)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, emit,
            )
            assert fired == [100]

            assert [p["pct"] for p in captured] == [50, 80, 100]
            assert all(p["tenant_id"] == "tenant-a" for p in captured)
            assert all(
                p["cap_millicents"] == cap_millicents for p in captured
            )

    @pytest.mark.asyncio
    async def test_single_fire_per_crossing(self):
        """Calling twice without further spend does not re-fire."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            cap_millicents = 1000
            captured: list[dict] = []

            await cost.add_cost("alpha", 500)
            fired_first = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, lambda p: captured.append(p),
            )
            fired_second = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, lambda p: captured.append(p),
            )
            assert fired_first == [50]
            assert fired_second == []  # no re-fire
            assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_jumps_emit_all_crossed_pcts(self):
        """A single big spend that vaults over 50 AND 80 fires both."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            cap_millicents = 1000
            captured: list[dict] = []

            await cost.add_cost("alpha", 850)  # 85% — crosses 50 + 80
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, lambda p: captured.append(p),
            )
            assert fired == [50, 80]
            assert {p["pct"] for p in captured} == {50, 80}

    @pytest.mark.asyncio
    async def test_disabled_when_cap_zero(self):
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            captured: list[dict] = []
            await cost.add_cost("alpha", 9999)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", 0, lambda p: captured.append(p),
            )
            assert fired == []
            assert captured == []

    @pytest.mark.asyncio
    async def test_month_rollover_resets_fired_pct(self):
        """Mutating the threshold bucket's month forgets previous crossings."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            cap_millicents = 1000
            captured: list[dict] = []

            await cost.add_cost("alpha", 600)
            await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, lambda p: captured.append(p),
            )
            assert len(captured) == 1
            # Simulate month rollover by reaching into the threshold state
            # and forcing a stale month — production gets this for free
            # when ``_threshold_bucket`` sees the month change.
            cost._threshold_state["tenant-a"]["month"] = "1970-01"
            # Need fresh spend or the spend bucket itself rolls over too.
            # Reset spend then add post-rollover.
            await cost.reset()
            await cost.add_cost("alpha", 600)

            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a", cap_millicents, lambda p: captured.append(p),
            )
            assert fired == [50]  # re-fired in the new month

    @pytest.mark.asyncio
    async def test_async_emit_callback_supported(self):
        """Coroutine emit callbacks are awaited."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            captured: list[dict] = []

            async def aemit(payload: dict) -> None:
                captured.append(payload)

            await cost.add_cost("alpha", 500)
            await cost.record_tenant_threshold_alerts(
                "tenant-a", 1000, aemit,
            )
            assert len(captured) == 1
            assert captured[0]["pct"] == 50


# ── CSV export endpoint ────────────────────────────────────────────────────


def _make_dashboard_client(tmp_path: str) -> TestClient:
    """Build a TestClient with the dashboard router mounted (auth-bypass).

    Auth-bypass: the dashboard's ``verify_session_cookie`` returns
    ``None`` (= pass) when no access-token file is present (dev mode).
    Tests run in dev mode by default, so we don't need to forge cookies.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

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
    health_monitor.register("alpha")
    health_monitor.register("beta")

    components = {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": {
            "alpha": "http://localhost:8401",
            "beta": "http://localhost:8402",
        },
    }
    # Suppress unused import warnings for AsyncMock
    _ = AsyncMock
    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), components


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()


class TestCSVExportEndpoint:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        # Force dev-mode auth (no access token file).
        self._auth_patch = patch(
            "src.dashboard.auth._ACCESS_TOKEN_PATH",
            str(Path(self._tmpdir) / "nonexistent_token"),
        )
        self._auth_patch.start()
        from src.dashboard.auth import reset_cache
        reset_cache()
        self.client, self.components = _make_dashboard_client(self._tmpdir)

    def teardown_method(self):
        _teardown(self.components)
        self._auth_patch.stop()
        from src.dashboard.auth import reset_cache
        reset_cache()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_csv_endpoint_returns_correct_shape(self):
        """Endpoint returns CSV with header + per-agent rows + total row."""
        import asyncio

        with _patch_projects({
            "alpha": "tenant-a",
            "beta": "tenant-a",
        }):
            cost.reset_tenant_cache()
            asyncio.run(cost.add_cost("alpha", 100))
            asyncio.run(cost.add_cost("beta", 50))

            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "monthly"},
            )

        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "captcha-rollup-tenant-a-monthly.csv" in (
            resp.headers.get("content-disposition", "")
        )
        lines = resp.text.strip().split("\n")
        assert lines[0] == (
            "period_start,agent_id,millicents,dollars,data_scope"
        )
        # Sorted agent rows then the synthetic total. ``data_scope`` is
        # ``monthly_actual`` for monthly because the in-memory state
        # IS the current month — the number is correct for the period.
        assert ",alpha,100,0.00100,monthly_actual" in lines[1]
        assert ",beta,50,0.00050,monthly_actual" in lines[2]
        assert "__tenant_total__,150,0.00150,monthly_actual" in lines[3]

    def test_csv_endpoint_period_daily(self):
        import asyncio

        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            asyncio.run(cost.add_cost("alpha", 100))
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "daily"},
            )
        assert resp.status_code == 200
        # Daily period_start is today UTC at midnight.
        first_data_line = resp.text.strip().split("\n")[1]
        cells = first_data_line.split(",")
        assert cells[0].endswith("T00:00:00Z")
        # Billing-honesty: ``daily`` (and ``weekly``) report month-to-date
        # data because the in-memory state is current-month only — the
        # ``data_scope`` column flags this so finance reconciliation
        # tooling doesn't accept the "daily" CSV as a daily-correct number.
        assert cells[-1] == "current_month_aggregate"

    def test_csv_endpoint_period_weekly(self):
        import asyncio

        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            asyncio.run(cost.add_cost("alpha", 100))
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "weekly"},
            )
        assert resp.status_code == 200
        first_data_line = resp.text.strip().split("\n")[1]
        cells = first_data_line.split(",")
        assert cells[0].endswith("T00:00:00Z")
        assert cells[-1] == "current_month_aggregate"

    def test_csv_missing_tenant_returns_400(self):
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"period": "monthly"},
        )
        assert resp.status_code == 400

    def test_csv_invalid_period_returns_400(self):
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "yearly"},
        )
        assert resp.status_code == 400

    def test_csv_get_does_not_require_csrf_header(self):
        """CSRF check exempts GET — the endpoint works without the header."""
        import asyncio

        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            asyncio.run(cost.add_cost("alpha", 100))
            # No X-Requested-With header — should still pass CSRF gate
            # because GET is in the exempt-method set.
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "monthly"},
            )
        assert resp.status_code == 200

    def test_csv_requires_auth_when_token_present(self):
        """When a real access-token file exists, missing/invalid cookie 401s."""
        # Stop the dev-mode patch and install a real token file.
        self._auth_patch.stop()
        token_file = Path(self._tmpdir) / "real_token"
        token_file.write_text("real-secret-token")
        production_patch = patch(
            "src.dashboard.auth._ACCESS_TOKEN_PATH", str(token_file),
        )
        production_patch.start()
        from src.dashboard.auth import reset_cache
        reset_cache()
        try:
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "monthly"},
            )
            assert resp.status_code == 401
        finally:
            production_patch.stop()
            # Re-instate the dev-mode patch so teardown_method's stop() is balanced.
            self._auth_patch.start()
            reset_cache()

    def test_csv_tenant_with_no_spend_returns_total_zero(self):
        """An empty tenant still emits the header + a zero-total row."""
        with _patch_projects({}):
            cost.reset_tenant_cache()
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "ghost", "period": "monthly"},
            )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert lines[0] == (
            "period_start,agent_id,millicents,dollars,data_scope"
        )
        # No agent rows, just header + total.
        assert len(lines) == 2
        assert "__tenant_total__,0,0.00000,monthly_actual" in lines[1]
