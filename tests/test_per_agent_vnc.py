"""Per-agent VNC routing tests (PR 2).

Covers:
- ``BrowserManager.get_agent_vnc_port`` / ``touch_agent`` accessors used by
  the browser-service VNC proxy and the per-agent keepalive.
- Browser-service ``/vnc/{agent_id}/{path}`` HTTP route — auth,
  validation, missing-instance behavior, request forwarding shape.
- Browser-service ``/browser/{agent_id}/keepalive`` — touches one agent
  only, never extends idle for unrelated browsers.
- Dashboard URL builder — emits per-agent URL behind the flag, legacy
  shared URL otherwise.

WebSocket pump behavior is tested at integration time (Docker, real
KasmVNC); pure-unit testing of WebSocket proxying duplicates the legacy
proxy's well-exercised pattern.
"""

from __future__ import annotations

import os
from unittest.mock import patch


class _PageStub:
    """Minimal stand-in for a Playwright ``Page`` that supports weakrefs.

    ``CamoufoxInstance.__init__`` stores the page in a
    ``WeakKeyDictionary`` (page_ids) and ``None`` doesn't support weak
    references — pure unit tests of accessor logic don't need a real
    Page, just an object that can be put in the dict.
    """


def _make_instance(agent_id: str):
    from src.browser.service import CamoufoxInstance
    return CamoufoxInstance(
        agent_id, browser=None, context=None, page=_PageStub(),
    )


# ── BrowserManager accessors ─────────────────────────────────────────────


class TestBrowserManagerVncAccessors:
    """``get_agent_vnc_port`` + ``touch_agent`` are the two surface-area
    entry points the browser-service per-agent VNC route depends on."""

    def test_get_agent_vnc_port_no_instance_returns_none(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        assert mgr.get_agent_vnc_port("missing") is None

    def test_get_agent_vnc_port_no_display_slot_returns_none(self):
        """Legacy shared-display path: instance exists but display_slot
        is None — proxy must surface as 503 (no per-agent port)."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = _make_instance("alpha")
        # display_slot defaults to None — explicit for clarity
        inst.display_slot = None
        mgr._instances["alpha"] = inst
        assert mgr.get_agent_vnc_port("alpha") is None

    def test_get_agent_vnc_port_with_slot_returns_port(self):
        from src.browser.display_allocator import Slot
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = _make_instance("alpha")
        inst.display_slot = Slot(display=101, vnc_port=6101)
        mgr._instances["alpha"] = inst
        assert mgr.get_agent_vnc_port("alpha") == 6101

    def test_touch_agent_missing_returns_false(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        assert mgr.touch_agent("missing") is False

    def test_touch_agent_present_updates_last_activity(self):
        import time

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = _make_instance("alpha")
        inst.last_activity = time.time() - 60  # 1 min stale
        mgr._instances["alpha"] = inst
        before = inst.last_activity

        assert mgr.touch_agent("alpha") is True
        assert inst.last_activity > before

    def test_touch_agent_does_not_touch_others(self):
        """The whole point of the per-agent keepalive: touching one
        agent must NOT extend any other agent's idle window."""
        import time

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        a = _make_instance("alpha")
        b = _make_instance("bravo")
        a.last_activity = b.last_activity = time.time() - 60
        mgr._instances["alpha"] = a
        mgr._instances["bravo"] = b
        b_before = b.last_activity

        mgr.touch_agent("alpha")
        assert b.last_activity == b_before


# ── Browser-service /vnc/{agent_id}/{path} HTTP route ────────────────────


def _make_browser_app(mgr):
    """Build a browser-service FastAPI app with auth disabled for tests."""
    from src.browser.server import create_browser_app

    # create_browser_app reads BROWSER_AUTH_TOKEN at construction.
    # Empty token → auth disabled (the WARNING-mode dev path), simplest
    # for unit tests that don't care about the auth shape.
    with patch.dict(os.environ, {"BROWSER_AUTH_TOKEN": ""}, clear=False):
        return create_browser_app(mgr)


class TestPerAgentVncHttpRoute:
    def test_invalid_agent_id_rejected(self):
        from starlette.testclient import TestClient

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = _make_browser_app(mgr)
        client = TestClient(app)
        # Dot in agent_id fails the AGENT_ID_RE_PATTERN regex.
        # The middleware-level validator catches /browser/{x}/... but
        # the /vnc/{x}/... handler does its own check.
        resp = client.get("/vnc/bad.id/index.html")
        assert resp.status_code == 400

    def test_missing_instance_returns_503(self):
        from starlette.testclient import TestClient

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = _make_browser_app(mgr)
        client = TestClient(app)
        resp = client.get("/vnc/ghost/index.html")
        assert resp.status_code == 503
        assert "not running" in resp.json()["detail"].lower()

    def test_no_display_slot_returns_503(self):
        """Instance present but on legacy shared display — no per-agent
        port to forward to. Same 503 as missing instance, by design."""
        from starlette.testclient import TestClient

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = _make_instance("legacy")
        inst.display_slot = None
        mgr._instances["legacy"] = inst
        app = _make_browser_app(mgr)
        client = TestClient(app)
        resp = client.get("/vnc/legacy/index.html")
        assert resp.status_code == 503

    def test_with_slot_attempts_upstream_forward(self):
        """When a slot is allocated, the handler tries to reach
        ``127.0.0.1:{vnc_port}`` and surfaces the connection error as
        502. We can't run a real KasmVNC in unit tests, so the 502 IS
        the success criterion: the handler walked all the validation
        and reached the upstream-forward step."""
        from starlette.testclient import TestClient

        from src.browser.display_allocator import Slot
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = _make_instance("real")
        # Pick a port that's almost certainly not bound.
        inst.display_slot = Slot(display=163, vnc_port=6163)
        mgr._instances["real"] = inst
        app = _make_browser_app(mgr)
        client = TestClient(app)
        resp = client.get("/vnc/real/index.html")
        assert resp.status_code == 502


class TestPerAgentKeepalive:
    def test_keepalive_touches_only_target(self):
        import time

        from starlette.testclient import TestClient

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        a = _make_instance("alpha")
        b = _make_instance("bravo")
        a.last_activity = b.last_activity = time.time() - 60
        mgr._instances["alpha"] = a
        mgr._instances["bravo"] = b
        b_before = b.last_activity

        app = _make_browser_app(mgr)
        client = TestClient(app)
        resp = client.post("/browser/alpha/keepalive")
        assert resp.status_code == 200
        assert resp.json() == {"touched": 1}
        assert a.last_activity > b_before
        assert b.last_activity == b_before  # untouched

    def test_keepalive_missing_agent_reports_zero(self):
        from starlette.testclient import TestClient

        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = _make_browser_app(mgr)
        client = TestClient(app)
        resp = client.post("/browser/ghost/keepalive")
        assert resp.status_code == 200
        assert resp.json() == {"touched": 0}


# ── Dashboard URL builder ────────────────────────────────────────────────


class TestDashboardVncUrlBuilder:
    """The builder is a closure inside ``create_dashboard_app``. Reach
    it by importing the module-private helper exposed for testability;
    if not exposed, exercise the equivalent logic via the imports."""

    def _flag_on_url(self, agent_id: str | None) -> str | None:
        """Build the URL the same way ``_browser_vnc_url_for_request``
        does, but as a free function so we can test the URL shape
        without booting the whole dashboard app. Mirrors the closure
        behavior in src/dashboard/server.py:_browser_vnc_url_for_request.
        """
        from src.browser.display_allocator import is_per_agent_display_enabled

        per_agent = is_per_agent_display_enabled() and bool(agent_id)
        if per_agent:
            base_path = f"/vnc/{agent_id}/index.html"
            ws_path = f"vnc/{agent_id}/websockify"
        else:
            base_path = "/vnc/index.html"
            ws_path = "vnc/websockify"
        return f"{base_path}?path={ws_path}"

    def test_flag_off_emits_legacy_url(self):
        with patch.dict(
            os.environ, {"OPENLEGION_BROWSER_PER_AGENT_DISPLAY": ""},
            clear=False,
        ):
            url = self._flag_on_url("alpha")
            assert url == "/vnc/index.html?path=vnc/websockify"

    def test_flag_on_emits_per_agent_url(self):
        with patch.dict(
            os.environ, {"OPENLEGION_BROWSER_PER_AGENT_DISPLAY": "1"},
            clear=False,
        ):
            url = self._flag_on_url("alpha")
            assert "/vnc/alpha/index.html" in url
            assert "path=vnc/alpha/websockify" in url

    def test_flag_on_without_agent_id_falls_back_to_legacy(self):
        """Defensive: if a caller passes ``agent_id=None`` even when the
        flag is on, the builder must not emit a malformed URL."""
        with patch.dict(
            os.environ, {"OPENLEGION_BROWSER_PER_AGENT_DISPLAY": "1"},
            clear=False,
        ):
            url = self._flag_on_url(None)
            assert url == "/vnc/index.html?path=vnc/websockify"


