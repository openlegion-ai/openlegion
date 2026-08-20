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
  * CSV export reads per-agent spend from the BROWSER SERVICE over HTTP
    (``GET /browser/captcha-costs``) rather than from the mesh process's
    own copy of the counter globals, groups it by team mesh-side, and
    503s instead of exporting zeros when the service is unreachable.
  * CSV endpoint requires auth (no ol_session cookie in production), GET
    is allowed without ``X-Requested-With`` (CSRF only on state changes).
  * ``record_tenant_threshold_alerts`` fires once per crossing per month
    at the 50/80/100% gates; subsequent calls in the same month do not
    re-fire.
  * Month rollover resets the fired-pct memory.
"""

from __future__ import annotations

import contextlib
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
        "CAPTCHA_COST_COUNTER_PATH",
        str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    cost.reset_threshold_state()
    cost.reset_tenant_cache()
    yield
    await cost.reset()
    cost.reset_threshold_state()
    cost.reset_tenant_cache()


# ── Helper: seed the team membership lookup ────────────────────────────────


@contextlib.contextmanager
def _patch_projects(membership: dict[str, str]):
    """Seed a real teams.db in a temp dir and point the resolver at it.

    ``membership`` maps agent_id → team_name. ``_tenant_for`` opens the
    TeamStore at ``OPENLEGION_TEAMS_DB`` in pure-DB mode — this seeds
    the production shape (one team per agent, membership rows).
    """
    from src.host.teams import TeamStore

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "teams.db")
        store = TeamStore(db_path=db)
        for agent, team in membership.items():
            if not store.team_exists(team):
                store.create_team(team)
            store.add_member(team, agent)
        with patch.dict(os.environ, {"OPENLEGION_TEAMS_DB": db}):
            cost.reset_tenant_cache()
            yield


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
        call_count = {"n": 0}
        real_read = cost._read_team_of

        def counting_read(db, agent_id):
            call_count["n"] += 1
            return real_read(db, agent_id)

        with _patch_projects({"alpha": "tenant-a"}):
            with patch.object(cost, "_read_team_of", counting_read):
                cost._tenant_for("alpha")
                cost._tenant_for("alpha")
                cost._tenant_for("alpha")
        # Exactly one underlying lookup — subsequent calls hit the LRU cache.
        assert call_count["n"] == 1

    def test_cache_invalidated_by_reset_tenant_cache(self):
        cost.reset_tenant_cache()
        with _patch_projects({"alpha": "tenant-a"}):
            assert cost._tenant_for("alpha") == "tenant-a"
        cost.reset_tenant_cache()
        with _patch_projects({"alpha": "tenant-b"}):
            assert cost._tenant_for("alpha") == "tenant-b"

    def test_loader_failure_returns_none(self):
        """If the DB read raises (corrupt DB), tenant=None."""
        with _patch_projects({"alpha": "tenant-a"}):
            with patch.object(
                cost,
                "_read_team_of",
                side_effect=RuntimeError("boom"),
            ):
                assert cost._tenant_for("alpha") is None

    def test_missing_db_returns_none(self, tmp_path):
        """No teams.db on disk (browser container) → tenant=None, no error."""
        cost.reset_tenant_cache()
        with patch.dict(
            os.environ,
            {"OPENLEGION_TEAMS_DB": str(tmp_path / "nope" / "teams.db")},
        ):
            assert cost._tenant_for("alpha") is None


# ── get_tenant_total ───────────────────────────────────────────────────────


class TestGetTenantTotal:
    @pytest.mark.asyncio
    async def test_sums_across_multiple_agents(self):
        with _patch_projects(
            {
                "alpha": "tenant-a",
                "beta": "tenant-a",
                "gamma": "tenant-a",
            }
        ):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            await cost.add_cost("beta", 50)
            await cost.add_cost("gamma", 25)
            assert await cost.get_tenant_total("tenant-a") == 175

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(self):
        """Tenant A's agents do NOT contribute to tenant B's total."""
        with _patch_projects(
            {
                "alpha": "tenant-a",
                "beta": "tenant-b",
                "gamma": "tenant-a",
            }
        ):
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
            # Anchor ``since`` to the first instant of the current
            # calendar month. The previous ``now - timedelta(days=7)``
            # crossed the month boundary on days 1–7, so the helper
            # (which only stores current-month state in memory)
            # correctly returned 0 — the assertion was the bug.
            month_start = now.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            assert (
                await cost.get_tenant_total(
                    "tenant-a",
                    since=month_start,
                )
                == 150
            )

    @pytest.mark.asyncio
    async def test_since_filter_past_month_returns_zero(self):
        """``since`` rooted in a past calendar month returns zero — the
        in-memory state is current-month only."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            await cost.add_cost("alpha", 100)
            now = datetime.now(timezone.utc)
            # Step back at least one full month.
            two_months_ago = now.replace(day=1) - timedelta(days=45)
            assert (
                await cost.get_tenant_total(
                    "tenant-a",
                    since=two_months_ago,
                )
                == 0
            )

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
        with _patch_projects(
            {
                "alpha": "tenant-a",
                "beta": "tenant-a",
                "gamma": "tenant-b",
            }
        ):
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
                "tenant-a",
                cap_millicents,
                emit,
            )
            assert fired == [50]

            # 80% crossing
            await cost.add_cost("alpha", 300)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a",
                cap_millicents,
                emit,
            )
            assert fired == [80]

            # 100% crossing
            await cost.add_cost("alpha", 200)
            fired = await cost.record_tenant_threshold_alerts(
                "tenant-a",
                cap_millicents,
                emit,
            )
            assert fired == [100]

            assert [p["pct"] for p in captured] == [50, 80, 100]
            assert all(p["tenant_id"] == "tenant-a" for p in captured)
            assert all(p["cap_millicents"] == cap_millicents for p in captured)

    @pytest.mark.asyncio
    async def test_single_fire_per_crossing(self):
        """Calling twice without further spend does not re-fire."""
        with _patch_projects({"alpha": "tenant-a"}):
            cost.reset_tenant_cache()
            cap_millicents = 1000
            captured: list[dict] = []

            await cost.add_cost("alpha", 500)
            fired_first = await cost.record_tenant_threshold_alerts(
                "tenant-a",
                cap_millicents,
                lambda p: captured.append(p),
            )
            fired_second = await cost.record_tenant_threshold_alerts(
                "tenant-a",
                cap_millicents,
                lambda p: captured.append(p),
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
                "tenant-a",
                cap_millicents,
                lambda p: captured.append(p),
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
                "tenant-a",
                0,
                lambda p: captured.append(p),
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
                "tenant-a",
                cap_millicents,
                lambda p: captured.append(p),
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
                "tenant-a",
                cap_millicents,
                lambda p: captured.append(p),
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
                "tenant-a",
                1000,
                aemit,
            )
            assert len(captured) == 1
            assert captured[0]["pct"] == 50


# ── CSV export endpoint ────────────────────────────────────────────────────
#
# The rollup does NOT read the cost counter in-process. ``_state`` is a
# process-local module global written in the BROWSER SERVICE process
# (:8500), while the dashboard router runs inside the mesh process
# (:8420) — an in-process import there reads a copy nothing ever writes.
# The endpoint therefore fetches per-agent spend over HTTP from the
# browser service and groups it by team on the mesh side, where the
# TeamStore lives. These tests wire a stub browser service through
# ``httpx.MockTransport`` so the real fetch helper is exercised end to end.

_BROWSER_SERVICE_URL = "http://browser.test:8500"


def _make_dashboard_client(
    tmp_path: str,
    teams: dict[str, str] | None = None,
    browser_service_url: str = _BROWSER_SERVICE_URL,
):
    """Build a TestClient with the dashboard router mounted (auth-bypass).

    Auth-bypass: the dashboard's ``verify_session_cookie`` returns
    ``None`` (= pass) when no access-token file is present (dev mode).
    Tests run in dev mode by default, so we don't need to forge cookies.

    ``teams`` maps agent_id → team name and seeds the real TeamStore the
    router groups by. ``browser_service_url`` is what the runtime reports;
    pass ``""`` to simulate a browser service that never came up.
    """
    from unittest.mock import MagicMock

    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.teams import TeamStore
    from src.host.traces import TraceStore

    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()

    teams_store = TeamStore(db_path=os.path.join(tmp_path, "teams.db"))
    for agent, team in (teams or {}).items():
        if not teams_store.team_exists(team):
            teams_store.create_team(team)
        teams_store.add_member(team, agent)

    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = browser_service_url
    runtime_mock.browser_auth_token = "browser-token"
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock,
        transport=transport_mock,
        router=router_mock,
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
        "runtime": runtime_mock,
        "teams_store": teams_store,
    }
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

        # Stub browser service. ``upstream`` is the per-agent spend the
        # service reports; ``upstream_status`` lets a test simulate a
        # failing / unreachable service.
        self.upstream: dict[str, int] = {}
        self.upstream_status = 200
        self.upstream_auth: list[str] = []

        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/browser/captcha-costs"
            self.upstream_auth.append(request.headers.get("Authorization", ""))
            if self.upstream_status != 200:
                return httpx.Response(
                    self.upstream_status,
                    json={"detail": "browser service error"},
                )
            return httpx.Response(
                200,
                json={
                    "month": datetime.now(timezone.utc).strftime("%Y-%m"),
                    "agents": dict(self.upstream),
                },
            )

        original_async_client = httpx.AsyncClient

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        # The dashboard builds its browser client at router-construction
        # time, so the patch only has to be live across the build.
        httpx_patch = patch.object(httpx, "AsyncClient", patched_async_client)
        httpx_patch.start()
        try:
            self.client, self.components = _make_dashboard_client(
                self._tmpdir,
                teams={"alpha": "tenant-a", "beta": "tenant-a"},
            )
        finally:
            httpx_patch.stop()

    def teardown_method(self):
        _teardown(self.components)
        self._auth_patch.stop()
        from src.dashboard.auth import reset_cache

        reset_cache()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_csv_endpoint_returns_correct_shape(self):
        """Endpoint returns CSV with header + per-agent rows + total row."""
        self.upstream = {"alpha": 100, "beta": 50}

        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "monthly"},
        )

        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "captcha-rollup-tenant-a-monthly.csv" in (resp.headers.get("content-disposition", ""))
        lines = resp.text.strip().split("\n")
        assert lines[0] == ("period_start,agent_id,millicents,dollars,data_scope")
        # Sorted agent rows then the synthetic total. ``data_scope`` is
        # ``monthly_actual`` for monthly because the upstream state
        # IS the current month — the number is correct for the period.
        assert ",alpha,100,0.00100,monthly_actual" in lines[1]
        assert ",beta,50,0.00050,monthly_actual" in lines[2]
        assert "__tenant_total__,150,0.00150,monthly_actual" in lines[3]

    def test_csv_reads_browser_service_not_in_process_counter(self):
        """Regression: the rollup must NOT read the mesh process's counter.

        ``captcha_cost_counter._state`` is process-local. The dashboard
        runs in the mesh process where that dict is always empty in production,
        so the export used to report 0 for every tenant. Seed the local
        counter with a value the browser service does NOT report: the CSV
        must carry the browser service's number, never the local one.
        """
        import asyncio

        with _patch_projects({"alpha": "tenant-a", "beta": "tenant-a"}):
            cost.reset_tenant_cache()
            asyncio.run(cost.add_cost("alpha", 999))
            asyncio.run(cost.add_cost("beta", 777))

            self.upstream = {"alpha": 100, "beta": 50}
            resp = self.client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "monthly"},
            )

        assert resp.status_code == 200
        body = resp.text
        assert ",alpha,100," in body
        assert ",beta,50," in body
        assert "__tenant_total__,150," in body
        # The in-process numbers must not leak into the export at all.
        assert "999" not in body
        assert "777" not in body

    def test_csv_forwards_browser_auth_token(self):
        """The upstream read carries the runtime's browser bearer token."""
        self.upstream = {"alpha": 100}
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "monthly"},
        )
        assert resp.status_code == 200
        assert self.upstream_auth == ["Bearer browser-token"]

    def test_csv_includes_team_member_with_no_spend(self):
        """Team members the browser service never charged still get a row."""
        self.upstream = {"alpha": 100}
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "monthly"},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert ",alpha,100,0.00100,monthly_actual" in lines[1]
        assert ",beta,0,0.00000,monthly_actual" in lines[2]
        assert "__tenant_total__,100,0.00100,monthly_actual" in lines[3]

    def test_csv_excludes_agents_outside_the_tenant(self):
        """Spend from agents outside the requested team is not rolled up."""
        self.upstream = {"alpha": 100, "stranger": 5000}
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "monthly"},
        )
        assert resp.status_code == 200
        assert "stranger" not in resp.text
        assert "__tenant_total__,100," in resp.text

    def test_csv_503_when_browser_service_not_configured(self):
        """No browser service → 503, never a zero-filled CSV.

        Silently exporting zeros into finance tooling is the exact
        failure mode this endpoint exists to avoid.
        """
        client, components = _make_dashboard_client(
            self._tmpdir,
            teams={"alpha": "tenant-a"},
            browser_service_url="",
        )
        try:
            resp = client.get(
                "/dashboard/api/billing/captcha-rollup",
                params={"tenant": "tenant-a", "period": "monthly"},
            )
            assert resp.status_code == 503
            assert "0.00000" not in resp.text
        finally:
            _teardown(components)

    def test_csv_503_when_browser_service_errors(self):
        """Upstream HTTP failure surfaces as 503, not a zero-filled CSV."""
        self.upstream_status = 500
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "tenant-a", "period": "monthly"},
        )
        assert resp.status_code == 503
        assert "0.00000" not in resp.text

    def test_csv_endpoint_period_daily(self):
        self.upstream = {"alpha": 100}
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
        # data because the upstream state is current-month only — the
        # ``data_scope`` column flags this so finance reconciliation
        # tooling doesn't accept the "daily" CSV as a daily-correct number.
        assert cells[-1] == "current_month_aggregate"

    def test_csv_endpoint_period_weekly(self):
        self.upstream = {"alpha": 100}
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
        self.upstream = {"alpha": 100}
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
            "src.dashboard.auth._ACCESS_TOKEN_PATH",
            str(token_file),
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
        self.upstream = {"alpha": 100}
        resp = self.client.get(
            "/dashboard/api/billing/captcha-rollup",
            params={"tenant": "ghost", "period": "monthly"},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert lines[0] == ("period_start,agent_id,millicents,dollars,data_scope")
        # No agent rows, just header + total.
        assert len(lines) == 2
        assert "__tenant_total__,0,0.00000,monthly_actual" in lines[1]


# ── Browser-service read surface ───────────────────────────────────────────


class TestSpendByAgent:
    """``spend_by_agent`` + its ``GET /browser/captcha-costs`` route.

    This is the seam the mesh-side rollup reads across the process
    boundary, so both the payload shape and the route have to hold.
    """

    @pytest.mark.asyncio
    async def test_returns_every_current_month_bucket(self):
        await cost.add_cost("alpha", 100)
        await cost.add_cost("beta", 50)
        payload = await cost.spend_by_agent()
        assert payload["month"] == datetime.now(timezone.utc).strftime("%Y-%m")
        assert payload["agents"] == {"alpha": 100, "beta": 50}

    @pytest.mark.asyncio
    async def test_skips_stale_month_buckets(self):
        await cost.add_cost("alpha", 100)
        # Age the bucket into a previous month — it contributes nothing to
        # the current month and must not be reported as if it did.
        cost._state["alpha"]["month"] = "1999-01"
        payload = await cost.spend_by_agent()
        assert payload["agents"] == {}

    @pytest.mark.asyncio
    async def test_empty_state_returns_empty_mapping(self):
        payload = await cost.spend_by_agent()
        assert payload["agents"] == {}

    def test_route_returns_counter_state(self, monkeypatch):
        """The browser service exposes the counter over HTTP."""
        import asyncio
        from unittest.mock import MagicMock

        from src.browser.server import create_browser_app

        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        asyncio.run(cost.add_cost("alpha", 100))

        app = create_browser_app(MagicMock())
        with TestClient(app) as client:
            resp = client.get("/browser/captcha-costs")

        assert resp.status_code == 200
        assert resp.json()["agents"] == {"alpha": 100}

    def test_route_requires_auth_when_token_set(self, monkeypatch):
        """The read is bearer-gated like every other service endpoint."""
        from unittest.mock import MagicMock

        from src.browser.server import create_browser_app

        monkeypatch.setenv("BROWSER_AUTH_TOKEN", "svc-token")

        app = create_browser_app(MagicMock())
        with TestClient(app) as client:
            assert client.get("/browser/captcha-costs").status_code == 401
            ok = client.get(
                "/browser/captcha-costs",
                headers={"Authorization": "Bearer svc-token"},
            )
        assert ok.status_code == 200
