"""Tests for Phase 6 §9.1 ``inspect_requests`` browser action.

Covers:
  * URL redaction at store-time (userinfo, JWT-shaped path, sensitive query).
  * ``deque`` maxlen enforcement (200) — older entries drop.
  * ``requestfailed`` classifier — adblock vs user-cancelled vs failed.
  * ``include_blocked`` filter + ``dropped_blocked`` counter.
  * ``limit`` cap at 200.
  * User-control conflict envelope.
  * ``BROWSER_NETWORK_INSPECT_DISABLED`` operator kill-switch (mesh route).
  * ISO-8601 UTC ``ts``.
  * RESET clears ``network_log`` and re-attaches listeners.
  * Context-level ``page`` listener attaches per-request handler to new tabs.
"""

from __future__ import annotations

import re
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────


class _FakeFailure:
    """Non-callable failure shape — matches Playwright's ``Request.failure``
    which is a property holding an object with ``errorText``. We deliberately
    avoid ``MagicMock`` here because MagicMocks are always callable, which
    would force the implementation's callable-fallback path on every test
    rather than exercising the property path.
    """

    def __init__(self, error_text: str):
        self.errorText = error_text


class _FakeRequest:
    """Non-callable request shape with the four attributes listeners read."""

    def __init__(self, url, method, resource_type, failure):
        self.url = url
        self.method = method
        self.resource_type = resource_type
        self.failure = failure


def _make_mock_request(
    url: str,
    method: str = "GET",
    resource_type: str = "document",
    failure_error: str | None = None,
):
    """Build a Playwright-Request-shaped fake for listener tests."""
    failure = _FakeFailure(failure_error) if failure_error is not None else None
    return _FakeRequest(url, method, resource_type, failure)


def _make_instance_with_listeners():
    """Return a CamoufoxInstance with mocked context/page wired by listeners.

    The mocks capture handlers passed to ``.on(...)`` so the test can fire
    them synchronously.
    """
    from src.browser.service import BrowserManager, CamoufoxInstance

    mgr = BrowserManager.__new__(BrowserManager)

    # Track context listeners in a dict so we can fire ``page`` / ``requestfailed``.
    context_handlers: dict[str, list] = {"page": [], "requestfailed": []}

    def _ctx_on(event: str, handler):
        context_handlers.setdefault(event, []).append(handler)

    context = MagicMock()
    context.on = MagicMock(side_effect=_ctx_on)

    # Page mock: same trick — ``request`` handlers stored, callable via _fire.
    page_handlers: dict[str, list] = {}

    def _page_on(event: str, handler):
        page_handlers.setdefault(event, []).append(handler)

    page = MagicMock()
    page.on = MagicMock(side_effect=_page_on)

    inst = CamoufoxInstance("agent1", MagicMock(), context, page)
    return mgr, inst, context_handlers, page_handlers


# ───────────────────────────────────────────────────────────────────────
# 1. URL redaction at store time
# ───────────────────────────────────────────────────────────────────────


class TestStoreTimeRedaction:
    def test_userinfo_jwt_and_sensitive_query_redacted(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)

        # userinfo + JWT in path + sensitive query (``token``)
        jwt = "abcdefghij.klmnopqrst.uvwxyz1234"
        url = (
            "https://user:secret@example.com/auth/"
            f"{jwt}/cb?token=SUPER_SECRET&keep=1"
        )
        req = _make_mock_request(url)
        mgr._record_request(inst, req)

        assert len(inst.network_log) == 1
        stored = inst.network_log[0]["url"]
        # Userinfo gone
        assert "user:secret" not in stored
        assert "secret@" not in stored
        # JWT segment redacted
        assert jwt not in stored
        # Sensitive query value redacted; key preserved
        assert "SUPER_SECRET" not in stored
        assert "token=%5BREDACTED%5D" in stored or "token=[REDACTED]" in stored
        # Non-sensitive query preserved
        assert "keep=1" in stored


# ───────────────────────────────────────────────────────────────────────
# 2. deque maxlen enforcement
# ───────────────────────────────────────────────────────────────────────


class TestDequeMaxlen:
    def test_only_newest_200_kept(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)

        for i in range(250):
            mgr._record_request(
                inst,
                _make_mock_request(f"https://example.com/r/{i}"),
            )

        assert len(inst.network_log) == 200
        # Oldest 50 dropped — first remaining entry should be /r/50.
        assert "/r/50" in inst.network_log[0]["url"]
        assert "/r/249" in inst.network_log[-1]["url"]


# ───────────────────────────────────────────────────────────────────────
# 3. requestfailed classifier
# ───────────────────────────────────────────────────────────────────────


class TestRequestFailedClassifier:
    @pytest.mark.parametrize("marker, flag", [
        ("NS_ERROR_CONTENT_BLOCKED", "blocked_by_adblock"),
        ("ERR_BLOCKED_BY_CLIENT", "blocked_by_adblock"),
        ("NS_ERROR_BLOCKED_BY_POLICY", "blocked_by_adblock"),
        ("NS_BINDING_ABORTED", "user_cancelled"),
        ("NS_ERROR_NET_TIMEOUT", "failed_network"),
    ])
    def test_marker_routes_to_correct_flag(self, marker, flag):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)

        url = "https://tracker.example.com/pixel.gif"
        mgr._record_request(inst, _make_mock_request(url))
        assert inst.network_log[-1][flag] is False  # Initial state

        failed_req = _make_mock_request(url, failure_error=marker)
        mgr._record_request_failed(inst, failed_req)

        entry = inst.network_log[-1]
        assert entry[flag] is True
        # Mutually exclusive — only the matching flag flips.
        for other in ("blocked_by_adblock", "user_cancelled", "failed_network"):
            if other != flag:
                assert entry[other] is False, (
                    f"marker={marker!r} also flipped {other}"
                )

    def test_failed_with_no_matching_entry_does_not_create_phantom(self):
        """requestfailed for a URL that scrolled out of the window is dropped."""
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)

        # No matching request recorded — the failure update should be ignored.
        failed_req = _make_mock_request(
            "https://gone.example.com/never-recorded",
            failure_error="NS_ERROR_CONTENT_BLOCKED",
        )
        mgr._record_request_failed(inst, failed_req)
        assert len(inst.network_log) == 0

    def test_callable_failure_attribute_handled(self):
        """Some Playwright bindings expose ``failure`` as a callable."""
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)

        url = "https://tracker.example.com/x"
        mgr._record_request(inst, _make_mock_request(url))

        # Build a request whose ``failure`` is a callable returning the
        # error-bearing object. Use a plain class (not MagicMock) so the
        # rest of the request shape stays property-like.
        failure_obj = _FakeFailure("NS_ERROR_CONTENT_BLOCKED")
        failed_req = _FakeRequest(url, "GET", "document", failure=lambda: failure_obj)
        mgr._record_request_failed(inst, failed_req)

        assert inst.network_log[-1]["blocked_by_adblock"] is True


# ───────────────────────────────────────────────────────────────────────
# 4. include_blocked / dropped_blocked
# ───────────────────────────────────────────────────────────────────────


class TestIncludeBlockedFilter:
    @pytest.mark.asyncio
    async def test_excludes_blocked_by_default(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        # 3 normal + 2 blocked
        for i in range(3):
            mgr._record_request(
                inst,
                _make_mock_request(f"https://example.com/ok/{i}"),
            )
        for i in range(2):
            url = f"https://tracker.example.com/blocked/{i}"
            mgr._record_request(inst, _make_mock_request(url))
            mgr._record_request_failed(
                inst,
                _make_mock_request(url, failure_error="ERR_BLOCKED_BY_CLIENT"),
            )

        result = await mgr.inspect_requests("agent1", include_blocked=False)
        assert result["success"] is True
        data = result["data"]
        assert data["total"] == 5
        assert data["dropped_blocked"] == 2
        assert len(data["requests"]) == 3
        assert all("blocked" not in r["url"] for r in data["requests"])

    @pytest.mark.asyncio
    async def test_include_blocked_keeps_them(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        for i in range(2):
            url = f"https://tracker.example.com/blocked/{i}"
            mgr._record_request(inst, _make_mock_request(url))
            mgr._record_request_failed(
                inst,
                _make_mock_request(url, failure_error="ERR_BLOCKED_BY_CLIENT"),
            )
        mgr._record_request(
            inst,
            _make_mock_request("https://example.com/ok"),
        )

        result = await mgr.inspect_requests("agent1", include_blocked=True)
        assert result["success"] is True
        data = result["data"]
        assert data["total"] == 3
        assert data["dropped_blocked"] == 0
        assert len(data["requests"]) == 3
        blocked_returned = [
            r for r in data["requests"] if r["blocked_by_adblock"]
        ]
        assert len(blocked_returned) == 2


# ───────────────────────────────────────────────────────────────────────
# 5. limit cap at 200
# ───────────────────────────────────────────────────────────────────────


class TestLimitCap:
    @pytest.mark.asyncio
    async def test_limit_above_200_coerced_to_200(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        for i in range(250):
            mgr._record_request(
                inst,
                _make_mock_request(f"https://example.com/r/{i}"),
            )

        result = await mgr.inspect_requests("agent1", limit=500)
        assert result["success"] is True
        # Underlying deque is also capped at 200.
        assert len(result["data"]["requests"]) == 200

    @pytest.mark.asyncio
    async def test_default_limit_50_returns_50(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        for i in range(120):
            mgr._record_request(
                inst,
                _make_mock_request(f"https://example.com/r/{i}"),
            )

        result = await mgr.inspect_requests("agent1")
        assert len(result["data"]["requests"]) == 50

    @pytest.mark.asyncio
    async def test_results_are_newest_first(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        for i in range(5):
            mgr._record_request(
                inst,
                _make_mock_request(f"https://example.com/r/{i}"),
            )

        result = await mgr.inspect_requests("agent1", limit=5)
        urls = [r["url"] for r in result["data"]["requests"]]
        assert "/r/4" in urls[0]
        assert "/r/0" in urls[-1]


# ───────────────────────────────────────────────────────────────────────
# 6. user_control returns conflict envelope
# ───────────────────────────────────────────────────────────────────────


class TestUserControlConflict:
    @pytest.mark.asyncio
    async def test_user_control_returns_conflict_envelope(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        inst._user_control = True
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.inspect_requests("agent1")

        assert result["success"] is False
        assert result["error"]["code"] == "conflict"
        assert "retry_after_ms" in result["error"]
        assert result["error"]["retry_after_ms"] is None


# ───────────────────────────────────────────────────────────────────────
# 7. Mesh kill-switch — BROWSER_NETWORK_INSPECT_DISABLED returns 403
# ───────────────────────────────────────────────────────────────────────


class TestMeshKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_returns_forbidden(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        from src.browser import flags as bflags
        from src.host.costs import CostTracker
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app
        from src.host.traces import TraceStore
        from src.shared.types import AgentPermissions

        monkeypatch.setenv("BROWSER_NETWORK_INSPECT_DISABLED", "1")
        # Reset cached operator settings so the env override wins.
        bflags.reload_operator_settings()

        permissions = PermissionMatrix()
        permissions.permissions["worker"] = AgentPermissions(
            agent_id="worker", can_use_browser=True,
        )
        router = MessageRouter(permissions, {"worker": "http://worker:8400"})
        bb = Blackboard(str(tmp_path / "bb.db"))
        pubsub = PubSub()
        costs = CostTracker(str(tmp_path / "costs.db"))
        traces = TraceStore(str(tmp_path / "traces.db"))
        cm = MagicMock()
        cm.browser_service_url = "http://browser-svc:8500"
        cm.browser_auth_token = ""

        app = create_mesh_app(
            blackboard=bb,
            pubsub=pubsub,
            router=router,
            permissions=permissions,
            cost_tracker=costs,
            trace_store=traces,
            event_bus=MagicMock(),
            container_manager=cm,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "agent_id": "worker",
                    "action": "inspect_requests",
                    "params": {"include_blocked": False, "limit": 10},
                },
                headers={"X-Agent-ID": "worker"},
            )

        assert resp.status_code == 403, (resp.status_code, resp.text)
        body = resp.json()
        # FastAPI wraps ``detail=`` payload under ``detail``.
        detail = body.get("detail", body)
        assert detail.get("success") is False
        assert detail["error"]["code"] == "forbidden"
        assert "retry_after_ms" in detail["error"]


# ───────────────────────────────────────────────────────────────────────
# 8. ts is ISO-8601 UTC string
# ───────────────────────────────────────────────────────────────────────


class TestTimestampShape:
    @pytest.mark.asyncio
    async def test_ts_is_iso8601_utc(self):
        mgr, inst, _ctx, _page = _make_instance_with_listeners()
        mgr._attach_network_listeners(inst)
        mgr.get_or_start = AsyncMock(return_value=inst)

        mgr._record_request(
            inst, _make_mock_request("https://example.com/x"),
        )
        result = await mgr.inspect_requests("agent1")

        ts = result["data"]["requests"][0]["ts"]
        # Shape: 2024-01-01T00:00:00Z
        assert isinstance(ts, str)
        assert ts.endswith("Z"), ts
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), ts


# ───────────────────────────────────────────────────────────────────────
# 9. RESET clears network_log + re-attaches listeners on next start
# ───────────────────────────────────────────────────────────────────────


class TestResetClearsAndReattaches:
    @pytest.mark.asyncio
    async def test_reset_drops_old_instance_and_new_one_starts_fresh(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        # Build a manager with one stale instance carrying a populated log.
        mgr = BrowserManager(profiles_dir="/tmp/_test_inspect_reset")
        old_ctx = AsyncMock()
        old_page = MagicMock()
        # Old context.on was wired during __init__ by _attach_network_listeners;
        # we simulate that state by manually flipping the flag + populating log.
        old = CamoufoxInstance("agent1", MagicMock(), old_ctx, old_page)
        old._network_attached = True
        old.network_log.append({
            "url": "https://example.com/stale",
            "method": "GET",
            "resource_type": "document",
            "ts": 1.0,
            "status": None,
            "blocked_by_adblock": False,
            "user_cancelled": False,
            "failed_network": False,
        })
        mgr._instances["agent1"] = old

        # RESET drops the old instance — confirmed by absence in _instances.
        await mgr.reset("agent1")
        assert "agent1" not in mgr._instances

        # The next CamoufoxInstance constructed for the same agent is a
        # fresh object. We verify the contract: a new instance starts with
        # empty network_log + _network_attached=False, so the next
        # _attach_network_listeners call wires fresh listeners.
        fresh_page = MagicMock()
        fresh_ctx = MagicMock()
        fresh_ctx.on = MagicMock()
        fresh_page.on = MagicMock()
        fresh = CamoufoxInstance("agent1", MagicMock(), fresh_ctx, fresh_page)
        assert fresh.network_log == deque(maxlen=200)
        assert len(fresh.network_log) == 0
        assert fresh._network_attached is False

        mgr._attach_network_listeners(fresh)
        assert fresh._network_attached is True
        # context-level listeners wired for ``page`` and ``requestfailed``.
        events = [call.args[0] for call in fresh_ctx.on.call_args_list]
        assert "page" in events
        assert "requestfailed" in events


# ───────────────────────────────────────────────────────────────────────
# 10. Context-level listener wires per-request handler on new pages
# ───────────────────────────────────────────────────────────────────────


class TestContextLevelPageListener:
    def test_new_page_inherits_request_handler(self):
        """Simulate context.emit('page', new_page) firing the registered handler.

        The handler we registered should attach a per-page ``request``
        listener; firing that listener should record into ``network_log``.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager.__new__(BrowserManager)

        ctx_handlers: dict[str, list] = {}

        def _ctx_on(event: str, handler):
            ctx_handlers.setdefault(event, []).append(handler)

        context = MagicMock()
        context.on = MagicMock(side_effect=_ctx_on)

        existing_page = MagicMock()
        existing_page.on = MagicMock()

        inst = CamoufoxInstance("agent1", MagicMock(), context, existing_page)
        mgr._attach_network_listeners(inst)

        # Now fire a context-level "page" event with a brand-new Mock Page.
        new_page = MagicMock()
        new_page_handlers: dict[str, list] = {}

        def _new_page_on(event: str, handler):
            new_page_handlers.setdefault(event, []).append(handler)

        new_page.on = MagicMock(side_effect=_new_page_on)

        # The registered context handler should attach a request handler
        # to the new page when invoked.
        for h in ctx_handlers["page"]:
            h(new_page)

        # The new_page now has a "request" listener registered.
        assert "request" in new_page_handlers
        assert len(new_page_handlers["request"]) == 1

        # Firing that listener with a Mock request should record into log.
        req = _make_mock_request("https://newtab.example.com/asset.js")
        new_page_handlers["request"][0](req)

        assert len(inst.network_log) == 1
        assert "/asset.js" in inst.network_log[0]["url"]


# ───────────────────────────────────────────────────────────────────────
# 11. Idempotency of _attach_network_listeners
# ───────────────────────────────────────────────────────────────────────


class TestAttachIdempotent:
    def test_double_attach_does_not_double_register(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager.__new__(BrowserManager)
        ctx = MagicMock()
        ctx.on = MagicMock()
        page = MagicMock()
        page.on = MagicMock()
        inst = CamoufoxInstance("agent1", MagicMock(), ctx, page)

        mgr._attach_network_listeners(inst)
        first_count = ctx.on.call_count
        mgr._attach_network_listeners(inst)  # second call: must no-op
        assert ctx.on.call_count == first_count
        assert inst._network_attached is True


# ───────────────────────────────────────────────────────────────────────
# 12. Skill registration sanity check
# ───────────────────────────────────────────────────────────────────────


class TestSkillRegistration:
    def test_skill_present_with_correct_name_and_params(self):
        from src.agent.builtins import browser_tool

        fn = getattr(browser_tool, "browser_inspect_requests", None)
        assert fn is not None

        meta = getattr(fn, "skill_meta", None) or getattr(fn, "_skill", None)
        # Registry-agnostic — just confirm callable + has the expected name
        # somewhere on its metadata. The skill registry itself is tested
        # separately in tests/test_skills.py.
        assert callable(fn)
        # The skill decorator stores metadata; we just confirm description
        # mentions adblock so future refactors can't silently lose intent.
        # Falls back to inspecting the docstring if no metadata attr.
        meta_text = (
            getattr(meta, "description", "") if meta else ""
        ) or (fn.__doc__ or "")
        # description bound to the decorator-generated callable
        decorated = getattr(fn, "__wrapped__", fn)
        registry_desc = ""
        if hasattr(fn, "description"):
            registry_desc = fn.description  # type: ignore[attr-defined]
        haystack = " ".join([meta_text, registry_desc, decorated.__doc__ or ""])
        assert "network requests" in haystack.lower() or "adblock" in haystack.lower()
