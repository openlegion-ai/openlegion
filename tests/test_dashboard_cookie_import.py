"""Tests for Phase 6 §9.2: operator cookie/session import.

Coverage:
1.  Audit log NEVER contains cookie values, JWTs, Bearer tokens, or
    SigV4-shaped strings (the most important security property).
2.  Playwright + Netscape happy paths.
3.  CSRF rejection (missing X-Requested-With).
4.  Auth rejection (missing ol_session in hosted mode).
5.  Rate-limit overflow → 429 with retry_after_ms.
6.  Kill-switch (BROWSER_COOKIE_IMPORT_DISABLED=1).
7.  Per-cookie 4 KiB value cap and 256 KiB total payload cap and
    1000-cookie list-length cap.
8.  Drop reasons: __Host-with-domain, IP-literal domain, invalid SameSite.
9.  Invalid JSON / unknown shape → 400.
10. Idempotency-ish: two overlapping imports succeed.
11. Browser-service down → 503.
12. Parser unit tests for ``_parse_netscape``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.dashboard.server import (
    _is_ip_literal_domain,
    _parse_netscape,
    _validate_cookies,
    create_dashboard_router,
)
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore

# A short corpus of value shapes the audit log MUST NEVER echo. Each
# string is also wrapped into a Playwright cookie payload below.
_SECRET_VALUES = [
    "raw-secretvalue-12345",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0In0.abc123",  # JWT-shape
    "Bearer abc123def456ghi789jkl012mno345pqr678stu901vwx234",          # Bearer
    "AKIAIOSFODNN7EXAMPLEAWS4-HMAC-SHA256-Credential-AKIAIOSFODNN7EXAMPLE-1234567890abcdef",  # SigV4-ish
]


class _CSRFTestClient(TestClient):
    """Auto-injects X-Requested-With on state-changing methods."""

    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            if "X-Requested-With" not in headers:
                headers["X-Requested-With"] = "XMLHttpRequest"
                kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


def _make_components(tmp_path: str) -> dict:
    """Minimal components for the dashboard router, with a runtime mock
    pointing at a non-existent browser_service_url so the test can
    intercept httpx calls via the shared client."""
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    agent_registry: dict[str, str] = {
        "alpha": "http://localhost:8401",
    }

    runtime_mock = MagicMock()
    runtime_mock.browser_service_url = "http://browser-svc:8500"
    runtime_mock.browser_auth_token = "test-token"
    runtime_mock.browser_vnc_url = None

    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    health_monitor.register("alpha")
    health_monitor.agents["alpha"].status = "healthy"

    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": agent_registry,
        "runtime": runtime_mock,
    }


def _make_client(components: dict) -> TestClient:
    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return _CSRFTestClient(app)


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()


def _stub_browser_service(monkeypatch, *, success: bool = True,
                          imported_count: int | None = None,
                          status_code: int = 200,
                          raise_exc: Exception | None = None) -> dict:
    """Patch the dashboard's shared httpx client so we can intercept
    posts to the browser service. Returns a dict where ``calls`` collects
    every (url, body) tuple."""
    tracker: dict = {"calls": []}

    async def _fake_post(url, *args, **kwargs):
        tracker["calls"].append({"url": url, "json": kwargs.get("json")})
        if raise_exc is not None:
            raise raise_exc
        body = kwargs.get("json") or {}
        cookies = body.get("cookies") or []
        count = imported_count if imported_count is not None else len(cookies)

        class _Resp:
            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        if success:
            payload = {"success": True, "data": {"imported": count}}
        else:
            payload = {
                "success": False,
                "error": {
                    "code": "invalid_input",
                    "message": "fake failure",
                    "retry_after_ms": None,
                },
            }
        return _Resp(status_code, payload)

    # Patch by mutating the module-level client used inside the router.
    # The client is created fresh per `create_dashboard_router` call —
    # we monkeypatch httpx.AsyncClient.post via the instance the router
    # captured. Easiest path: stash a reference into the router state.
    import src.dashboard.server as _srv

    real_async_client = _srv._httpx.AsyncClient if hasattr(_srv, "_httpx") else None
    if real_async_client is None:
        # The httpx import is local to create_dashboard_router. Patch
        # the real httpx module instead so all clients route through.
        import httpx as _httpx_mod

        async def _proxy_post(self, url, *args, **kwargs):
            return await _fake_post(url, *args, **kwargs)

        monkeypatch.setattr(_httpx_mod.AsyncClient, "post", _proxy_post)
    return tracker


# ────────────────────────────────────────────────────────────────────────
# Pure-function tests (no HTTP)
# ────────────────────────────────────────────────────────────────────────


class TestParseNetscape:
    def test_basic_line(self):
        text = ".example.com\tTRUE\t/\tTRUE\t1735689600\tsid\tabcdef\n"
        result = _parse_netscape(text)
        assert len(result) == 1
        assert result[0]["name"] == "sid"
        assert result[0]["value"] == "abcdef"
        assert result[0]["domain"] == ".example.com"
        assert result[0]["secure"] is True
        assert result[0]["httpOnly"] is False
        assert result[0]["expires"] == 1735689600
        assert result[0]["path"] == "/"

    def test_comment_lines_skipped(self):
        text = (
            "# this is a comment\n"
            "# another comment\n"
            ".example.com\tTRUE\t/\tFALSE\t100\tname\tvalue\n"
        )
        result = _parse_netscape(text)
        assert len(result) == 1
        assert result[0]["secure"] is False

    def test_httponly_prefix(self):
        text = "#HttpOnly_.example.com\tTRUE\t/\tTRUE\t100\tsid\tabc\n"
        result = _parse_netscape(text)
        assert len(result) == 1
        assert result[0]["domain"] == ".example.com"
        assert result[0]["httpOnly"] is True

    def test_missing_fields_dropped(self):
        text = "# header\n.example.com\tTRUE\t/\tTRUE\n"  # too few fields
        assert _parse_netscape(text) == []

    def test_unicode_domain_accepted(self):
        text = ".пример.com\tTRUE\t/\tTRUE\t100\tsid\txyz\n"
        result = _parse_netscape(text)
        assert len(result) == 1
        assert result[0]["domain"] == ".пример.com"

    def test_non_numeric_expires_dropped(self):
        text = ".example.com\tTRUE\t/\tTRUE\tnot-a-number\tsid\txyz\n"
        assert _parse_netscape(text) == []

    def test_blank_lines_skipped(self):
        text = "\n\n\n.example.com\tTRUE\t/\tTRUE\t100\tsid\tabc\n\n"
        assert len(_parse_netscape(text)) == 1


class TestValidateCookies:
    def test_playwright_happy_path(self):
        accepted, dropped, fmt = _validate_cookies(
            [{"name": "sid", "value": "abc", "domain": ".example.com"}],
        )
        assert fmt == "playwright"
        assert len(accepted) == 1
        assert dropped == []

    def test_drops_value_too_large(self):
        accepted, dropped, fmt = _validate_cookies(
            [{"name": "sid", "value": "x" * 5000, "domain": ".example.com"}],
        )
        assert accepted == []
        assert dropped == [{"reason": "value_too_large", "count": 1}]

    def test_drops_host_prefix_with_domain(self):
        accepted, dropped, _ = _validate_cookies(
            [{"name": "__Host-sess", "value": "abc", "domain": ".example.com"}],
        )
        assert accepted == []
        assert dropped == [{"reason": "host_prefix_with_domain", "count": 1}]

    def test_allows_host_prefix_without_domain(self):
        # __Host- without domain (empty domain) is rejected by empty_domain rule.
        # The valid form has domain absent entirely; we just verify host_prefix
        # only fires when domain is set.
        accepted, dropped, _ = _validate_cookies(
            [{"name": "__Host-sess", "value": "abc", "domain": ""}],
        )
        # empty_domain triggers first; not host_prefix
        assert "host_prefix_with_domain" not in [d["reason"] for d in dropped]

    def test_drops_ip_literal_domain(self):
        for domain in ["127.0.0.1", "[::1]", "192.168.1.1"]:
            accepted, dropped, _ = _validate_cookies(
                [{"name": "sid", "value": "x", "domain": domain}],
            )
            assert accepted == []
            assert dropped == [{"reason": "ip_domain_unsupported", "count": 1}]

    def test_drops_invalid_samesite(self):
        accepted, dropped, _ = _validate_cookies(
            [{"name": "sid", "value": "x", "domain": ".example.com",
              "sameSite": "BadValue"}],
        )
        assert accepted == []
        assert dropped == [{"reason": "invalid_samesite", "count": 1}]

    def test_normalizes_samesite_case_insensitive(self):
        accepted, _, _ = _validate_cookies(
            [{"name": "sid", "value": "x", "domain": ".example.com",
              "sameSite": "lax"}],
        )
        assert accepted[0]["sameSite"] == "Lax"

    def test_rejects_string_expires(self):
        accepted, dropped, _ = _validate_cookies(
            [{"name": "sid", "value": "x", "domain": ".example.com",
              "expires": "not-a-number"}],
        )
        assert accepted == []
        assert dropped == [{"reason": "invalid_expires", "count": 1}]

    def test_unknown_format_returns_none(self):
        # An int payload doesn't match either format.
        _, _, fmt = _validate_cookies(42)
        assert fmt is None


class TestIsIpLiteralDomain:
    def test_ipv4(self):
        assert _is_ip_literal_domain("127.0.0.1")
        assert _is_ip_literal_domain("8.8.8.8")
        assert _is_ip_literal_domain(".10.0.0.1")  # leading dot stripped

    def test_ipv6_bracketed(self):
        assert _is_ip_literal_domain("[::1]")
        assert _is_ip_literal_domain("[2001:db8::1]")

    def test_normal_domain(self):
        assert not _is_ip_literal_domain(".example.com")
        assert not _is_ip_literal_domain("not.an.ip")

    def test_octet_out_of_range_rejected(self):
        # 999 is not a valid octet — must NOT be classified as IP.
        assert not _is_ip_literal_domain("999.999.999.999")


# ────────────────────────────────────────────────────────────────────────
# HTTP-level tests (full router)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def setup(monkeypatch):
    """Build a fresh dashboard router + browser-service stub per test."""
    tmpdir = tempfile.mkdtemp()
    components = _make_components(tmpdir)
    tracker = _stub_browser_service(monkeypatch)
    client = _make_client(components)
    yield {
        "client": client,
        "components": components,
        "tracker": tracker,
        "tmpdir": tmpdir,
    }
    _teardown(components)
    shutil.rmtree(tmpdir, ignore_errors=True)


# Reset the cookie-import rate-limit dict between tests by tearing
# down and rebuilding the router fresh in each test (the fixture does this).


class TestAuditLogValueRedaction:
    """The most critical property: audit log MUST NEVER contain cookie values."""

    def test_audit_log_no_value(self, setup, caplog):
        """Captured log text contains domain + name but NEVER value."""
        client = setup["client"]
        # Build a payload where each cookie's VALUE is one of the secret shapes.
        cookies = [
            {"name": f"sid_{i}", "value": secret,
             "domain": f".example{i}.com"}
            for i, secret in enumerate(_SECRET_VALUES)
        ]
        with caplog.at_level(logging.INFO, logger="dashboard.cookie_import"):
            resp = client.post(
                "/dashboard/api/agents/alpha/browser/import_cookies",
                json={"format": "playwright", "cookies": cookies},
            )
        assert resp.status_code == 200, resp.text
        # Concatenate every captured record.
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        # Domains AND names should appear (proves audit hit and we have
        # the right log messages).
        for c in cookies:
            assert c["domain"] in log_text
            assert c["name"] in log_text
        # NONE of the cookie values should appear in any form.
        for secret in _SECRET_VALUES:
            assert secret not in log_text, (
                f"Cookie value leaked into audit log: {secret!r}"
            )
        # Also assert structural negatives — common patterns within values.
        assert "raw-secretvalue" not in log_text
        assert "eyJhbGciOiJIUzI1NiIs" not in log_text
        assert "Bearer abc123" not in log_text

    def test_audit_log_no_value_in_netscape_path(self, setup, caplog):
        """Same property when the input format is Netscape TSV."""
        client = setup["client"]
        # Build a Netscape blob whose VALUE column carries each secret.
        ns = "\n".join(
            f".example{i}.com\tTRUE\t/\tTRUE\t1735689600\tsid_{i}\t{secret}"
            for i, secret in enumerate(_SECRET_VALUES)
        )
        with caplog.at_level(logging.INFO, logger="dashboard.cookie_import"):
            resp = client.post(
                "/dashboard/api/agents/alpha/browser/import_cookies",
                json={"format": "netscape", "cookies": ns},
            )
        assert resp.status_code == 200, resp.text
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        for secret in _SECRET_VALUES:
            assert secret not in log_text


class TestHappyPaths:
    def test_playwright_format_pushes_to_browser_service(self, setup):
        client = setup["client"]
        tracker = setup["tracker"]
        cookies = [
            {"name": "sid", "value": "abc",
             "domain": ".example.com", "path": "/",
             "expires": 1735689600,
             "httpOnly": True, "secure": True, "sameSite": "Lax"},
        ]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": cookies},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["imported"] == 1
        assert data["data"]["format"] == "playwright"
        # Browser service was actually called.
        assert any(
            c["url"].endswith("/browser/alpha/import_cookies")
            for c in tracker["calls"]
        )
        # The list pushed contains the normalized shape (sameSite still "Lax").
        pushed = next(
            c for c in tracker["calls"]
            if c["url"].endswith("/browser/alpha/import_cookies")
        )
        assert pushed["json"]["cookies"][0]["sameSite"] == "Lax"

    def test_netscape_format_with_httponly_prefix(self, setup):
        client = setup["client"]
        tracker = setup["tracker"]
        ns = (
            "# Netscape\n"
            "#HttpOnly_.example.com\tTRUE\t/\tTRUE\t1735689600\tsid\tabc\n"
        )
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "netscape", "cookies": ns},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["imported"] == 1
        # Verify the pushed cookie has httpOnly=True.
        pushed = next(
            c for c in tracker["calls"]
            if c["url"].endswith("/browser/alpha/import_cookies")
        )
        cookie = pushed["json"]["cookies"][0]
        assert cookie["httpOnly"] is True
        assert cookie["domain"] == ".example.com"


class TestCSRFAndAuth:
    def test_csrf_rejection_no_x_requested_with(self, setup):
        client = TestClient(_make_client(setup["components"]).app)
        # Bare TestClient — no X-Requested-With injection.
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": []},
        )
        assert resp.status_code == 403
        assert "X-Requested-With" in resp.text

    def test_auth_rejection_in_hosted_mode(self, setup, monkeypatch, tmp_path):
        """When access token file is present, missing ol_session → 401."""
        from src.dashboard import auth
        # Create a fake access token + hosted indicator so verify_session_cookie
        # enters hosted-mode behavior.
        token_path = tmp_path / "access_token"
        token_path.write_text("test-access-token")
        hosted_indicator = tmp_path / ".subdomain"
        hosted_indicator.write_text("test")
        monkeypatch.setattr(auth, "_ACCESS_TOKEN_PATH", str(token_path))
        monkeypatch.setattr(auth, "_HOSTED_INDICATOR", str(hosted_indicator))
        auth.reset_cache()
        # Reset _is_hosted cache.
        monkeypatch.setattr(auth, "_is_hosted", None)
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": []},
        )
        assert resp.status_code == 401
        # Cleanup auth cache so other tests in this session aren't affected.
        auth.reset_cache()


class TestRateLimit:
    def test_eleventh_call_within_window_returns_429(self, setup):
        client = setup["client"]
        # Send 10 requests — all should succeed.
        for i in range(10):
            resp = client.post(
                "/dashboard/api/agents/alpha/browser/import_cookies",
                json={"format": "playwright",
                      "cookies": [{"name": f"sid{i}", "value": "x",
                                   "domain": ".example.com"}]},
            )
            assert resp.status_code == 200, f"call {i}: {resp.text}"
        # 11th must be rate-limited.
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright",
                  "cookies": [{"name": "sidX", "value": "x",
                               "domain": ".example.com"}]},
        )
        assert resp.status_code == 429
        body = resp.json()
        # FastAPI nests the structured body under "detail" when raising HTTPException.
        err = body.get("detail", {}).get("error") or body.get("error") or {}
        assert err.get("code") == "conflict"
        assert err.get("retry_after_ms") and err["retry_after_ms"] > 0


class TestKillSwitch:
    def test_disabled_flag_returns_403(self, setup, monkeypatch):
        from src.browser import flags as _flags
        monkeypatch.setenv("BROWSER_COOKIE_IMPORT_DISABLED", "1")
        # Reload settings cache so the flag picks up.
        _flags.reload_operator_settings()
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": []},
        )
        assert resp.status_code == 403
        body = resp.json()
        err = body.get("detail", {}).get("error") or {}
        assert err.get("code") == "forbidden"


class TestSizeCaps:
    def test_per_cookie_value_too_large(self, setup):
        client = setup["client"]
        # 5 KiB > 4 KiB cap; cookie should be DROPPED (not import-fail).
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright",
                  "cookies": [{"name": "sid", "value": "x" * 5000,
                               "domain": ".example.com"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["imported"] == 0
        reasons = [d["reason"] for d in data["data"]["dropped"]]
        assert "value_too_large" in reasons

    def test_total_payload_too_large(self, setup):
        client = setup["client"]
        # 257 KiB raw bytes — must be 413 BEFORE we even parse JSON.
        big = "x" * (257 * 1024)
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            content=big,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        err = body.get("detail", {}).get("error") or {}
        assert err.get("code") == "size_limit"

    def test_list_length_cap_exceeded(self, setup):
        """1001 cookies → 413 envelope."""
        client = setup["client"]
        # Use minimal payload per cookie to stay under 256 KiB total.
        cookies = [
            {"name": f"s{i}", "value": "v", "domain": ".example.com"}
            for i in range(1001)
        ]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": cookies},
        )
        assert resp.status_code == 413
        body = resp.json()
        err = body.get("detail", {}).get("error") or {}
        assert err.get("code") == "size_limit"


class TestDropReasons:
    def test_host_prefix_with_domain_dropped(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright",
                  "cookies": [{"name": "__Host-sess", "value": "x",
                               "domain": ".example.com"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["imported"] == 0
        reasons = [d["reason"] for d in data["data"]["dropped"]]
        assert "host_prefix_with_domain" in reasons

    def test_ip_literal_dropped(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright",
                  "cookies": [{"name": "sid", "value": "x",
                               "domain": "127.0.0.1"}]},
        )
        data = resp.json()
        reasons = [d["reason"] for d in data["data"]["dropped"]]
        assert "ip_domain_unsupported" in reasons

    def test_invalid_samesite_dropped(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright",
                  "cookies": [{"name": "sid", "value": "x",
                               "domain": ".example.com",
                               "sameSite": "RandomGarbage"}]},
        )
        data = resp.json()
        reasons = [d["reason"] for d in data["data"]["dropped"]]
        assert "invalid_samesite" in reasons


class TestInvalidInput:
    def test_invalid_json_returns_400(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        err = body.get("detail", {}).get("error") or {}
        assert err.get("code") == "invalid_input"

    def test_unknown_shape_returns_400(self, setup):
        client = setup["client"]
        # Payload is neither a list (playwright) nor a TSV string (netscape).
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"cookies": 42},
        )
        assert resp.status_code == 400
        body = resp.json()
        err = body.get("detail", {}).get("error") or {}
        assert err.get("code") == "invalid_input"

    def test_unknown_format_explicit(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "curl-cookie-jar", "cookies": []},
        )
        assert resp.status_code == 400


class TestIdempotencyLike:
    def test_two_overlapping_imports_succeed(self, setup):
        client = setup["client"]
        cookies1 = [{"name": "sid", "value": "v1", "domain": ".example.com"}]
        cookies2 = [
            {"name": "sid", "value": "v2", "domain": ".example.com"},
            {"name": "tok", "value": "v3", "domain": ".example.com"},
        ]
        r1 = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": cookies1},
        )
        r2 = client.post(
            "/dashboard/api/agents/alpha/browser/import_cookies",
            json={"format": "playwright", "cookies": cookies2},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["data"]["imported"] == 1
        assert r2.json()["data"]["imported"] == 2


class TestBrowserServiceDown:
    def test_browser_service_raises_returns_503(self, monkeypatch):
        tmpdir = tempfile.mkdtemp()
        try:
            components = _make_components(tmpdir)
            # Make any httpx post raise.
            _stub_browser_service(monkeypatch, raise_exc=RuntimeError("svc down"))
            client = _make_client(components)
            resp = client.post(
                "/dashboard/api/agents/alpha/browser/import_cookies",
                json={"format": "playwright",
                      "cookies": [{"name": "sid", "value": "x",
                                   "domain": ".example.com"}]},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert resp.status_code == 503
            body = resp.json()
            err = body.get("detail", {}).get("error") or {}
            assert err.get("code") == "service_unavailable"
        finally:
            _teardown(components)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_browser_service_returns_500_returns_503(self, monkeypatch):
        tmpdir = tempfile.mkdtemp()
        try:
            components = _make_components(tmpdir)
            _stub_browser_service(monkeypatch, status_code=502)
            client = _make_client(components)
            resp = client.post(
                "/dashboard/api/agents/alpha/browser/import_cookies",
                json={"format": "playwright",
                      "cookies": [{"name": "sid", "value": "x",
                                   "domain": ".example.com"}]},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert resp.status_code == 503
        finally:
            _teardown(components)
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestUnknownAgent:
    def test_404_when_agent_not_in_registry(self, setup):
        client = setup["client"]
        resp = client.post(
            "/dashboard/api/agents/nonexistent/browser/import_cookies",
            json={"format": "playwright", "cookies": []},
        )
        assert resp.status_code == 404
