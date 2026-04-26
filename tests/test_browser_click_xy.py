"""Tests for Phase 6 §9.3 ``click_xy`` — coordinate-based click with
``document.elementFromPoint`` overlay pre-check.

Style follows ``tests/test_browser_service.py``: Playwright is mocked
end-to-end, so the real engine never runs. The element-from-point JS
also stays in the manager — we mock ``inst.page.evaluate`` to return
canned dicts and verify the manager's classification + dispatch path.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_inst(
    *, viewport: dict | None = None, user_control: bool = False,
):
    """Build a CamoufoxInstance with the bits ``click_xy`` reads."""
    from src.browser.service import CamoufoxInstance
    mock_page = MagicMock()
    mock_page.viewport_size = (
        {"width": 1280, "height": 800} if viewport is None else viewport
    )
    # mouse.click is awaited; AsyncMock for the mouse facade.
    mock_page.mouse = MagicMock()
    mock_page.mouse.click = AsyncMock()
    # locator() is used by _check_captcha — return zero matches by default.
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_page.locator.return_value = mock_locator
    inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
    inst._user_control = user_control
    return inst


def _make_mgr_with(inst):
    from src.browser.service import BrowserManager
    mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
    mgr._instances[inst.agent_id] = inst
    return mgr


class TestClickXyHappyPath:
    """A clean hit on an unmasked element should dispatch the click."""

    @pytest.mark.asyncio
    async def test_click_xy_success_returns_actual_element(self):
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "button",
            "role": "button",
            "name": "Save",
            "masked_by": None,
            "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=100, y=200)
        assert result["success"] is True
        assert result["data"]["clicked_at"] == {"x": 100.0, "y": 200.0}
        actual = result["data"]["actual_element"]
        assert actual["tag"] == "button"
        assert actual["role"] == "button"
        assert actual["name"] == "Save"

    @pytest.mark.asyncio
    async def test_click_xy_dispatches_mouse_click_exactly_once(self):
        """Spec test #10 — mouse.click invoked once with the supplied coords."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "div", "role": None, "name": "x",
            "masked_by": None, "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        await mgr.click_xy("a1", x=42, y=43)
        inst.page.mouse.click.assert_awaited_once_with(42.0, 43.0)

    @pytest.mark.asyncio
    async def test_click_xy_calls_check_captcha_post_click(self):
        """Spec test #11 — post-click CAPTCHA re-detection runs."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "a", "role": None, "name": "Link",
            "masked_by": None, "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)
        mgr._check_captcha = AsyncMock(return_value=None)

        await mgr.click_xy("a1", x=1, y=1)
        mgr._check_captcha.assert_awaited_once_with(inst)

    @pytest.mark.asyncio
    async def test_click_xy_success_increments_metrics(self):
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "button", "role": "button", "name": "ok",
            "masked_by": None, "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        await mgr.click_xy("a1", x=10, y=10)
        assert inst.m_click_success == 1
        assert list(inst.click_window) == [True]


class TestClickXyBoundsValidation:
    """§2.3 ``invalid_input`` envelope on out-of-range coordinates."""

    @pytest.mark.asyncio
    async def test_negative_x_returns_invalid_input(self):
        """Spec test #2 — x=-1 → invalid_input.

        Negative coords are rejected before viewport_size is even read,
        so the message doesn't carry viewport dimensions — it's a
        coordinate-domain bug, not a viewport-mismatch.
        """
        inst = _make_inst()
        inst.page.evaluate = AsyncMock()
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=-1, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "out of viewport bounds" in result["error"]["message"]
        assert "non-negative" in result["error"]["message"]
        # Must not have evaluated or clicked.
        inst.page.evaluate.assert_not_awaited()
        inst.page.mouse.click.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_x_at_or_above_width_carries_viewport_dims(self):
        """Upper-bound rejection includes the viewport dims so the agent
        can re-target. Negative-coord rejection (above) skips them."""
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=2000, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "1280" in result["error"]["message"]
        assert "800" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_x_at_or_above_width_returns_invalid_input(self):
        """Spec test #2 — x=10000 → invalid_input."""
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=10000, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_x_equal_to_width_is_rejected(self):
        """``x == width`` is out of bounds (0-indexed)."""
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=1280, y=400)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_negative_y_returns_invalid_input(self):
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=10, y=-5)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_zero_zero_is_in_bounds(self):
        """(0, 0) is valid — top-left pixel of the viewport."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "html", "role": None, "name": "",
            "masked_by": None, "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=0, y=0)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_viewport_size_none_skips_upper_bound_check(self):
        """``page.viewport_size`` is ``None`` in some Playwright configs.
        The manager must not return ``service_unavailable`` — it should
        skip the upper bound check and let Playwright reject out-of-window
        coords with its own error if needed. Negative coords are still
        rejected unconditionally."""
        inst = _make_inst(viewport=None)
        # Override _make_inst's default to actually set viewport_size=None
        inst.page.viewport_size = None
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "div", "role": None, "name": "ok",
            "masked_by": None, "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        # Positive coords with no viewport metadata: click proceeds.
        result = await mgr.click_xy("a1", x=100, y=200)
        assert result["success"] is True
        inst.page.mouse.click.assert_awaited_once_with(100.0, 200.0)

    @pytest.mark.asyncio
    async def test_viewport_size_none_still_rejects_negative(self):
        """Even when viewport_size is unknown, negative coords are
        always invalid — clamp before reading viewport_size at all."""
        inst = _make_inst()
        inst.page.viewport_size = None
        inst.page.evaluate = AsyncMock()
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=-1, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        inst.page.evaluate.assert_not_awaited()


class TestClickXyTypeValidation:
    """Reject NaN/inf/bool — these are caller bugs."""

    @pytest.mark.asyncio
    async def test_nan_x_returns_invalid_input(self):
        """Spec test #3 — NaN/inf → invalid_input."""
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=float("nan"), y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "finite" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_inf_y_returns_invalid_input(self):
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=10, y=float("inf"))
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_bool_x_rejected(self):
        """Python ``True == 1`` would pass the (int, float) check —
        booleans are rejected explicitly so a typo doesn't silently click."""
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x=True, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "bool" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_string_x_rejected(self):
        inst = _make_inst()
        mgr = _make_mgr_with(inst)
        result = await mgr.click_xy("a1", x="100", y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


class TestClickXyElementFromPoint:
    """Pre-check classification: null hit, masking detection."""

    @pytest.mark.asyncio
    async def test_no_element_at_point(self):
        """Spec test #4 — ``elementFromPoint`` returns null."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value=None)
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=10, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "no_element_at_point"
        inst.page.mouse.click.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_masked_by_pointer_events(self):
        """Spec test #5 — ancestor ``pointer-events: none`` → invalid_input
        with ``masked_by`` set."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "button",
            "role": "button",
            "name": "Hidden",
            "masked_by": "div.overlay",
            "mask_reason": "pointer-events",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=10, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        data = result["error"]["data"]
        assert data["masked_by"] == "div.overlay"
        assert data["mask_reason"] == "pointer-events"
        assert data["actual"]["tag"] == "button"
        assert data["actual"]["role"] == "button"
        assert data["actual"]["name"] == "Hidden"
        # No click should be dispatched on a masked hit.
        inst.page.mouse.click.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_masked_by_visibility_hidden(self):
        """Spec test #6 — ancestor ``visibility: hidden`` → mask_reason=visibility."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "span", "role": None, "name": "x",
            "masked_by": "div.hidden",
            "mask_reason": "visibility",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=10, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["data"]["mask_reason"] == "visibility"

    @pytest.mark.asyncio
    async def test_masked_by_display_none(self):
        """Spec test #7 — ancestor ``display: none`` → mask_reason=display."""
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "a", "role": None, "name": "y",
            "masked_by": "section#hide",
            "mask_reason": "display",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=10, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["data"]["mask_reason"] == "display"

    @pytest.mark.asyncio
    async def test_inner_pointer_events_auto_overrides_outer_none(self):
        """Spec test #8 — inner ``pointer-events: auto`` cascades over outer
        ``pointer-events: none``; the click MUST proceed.

        The element-from-point walker is in JS; we represent the "auto
        boundary detected before any none ancestor" outcome with
        ``masked_by=null`` (the unmasked hit shape). The click then
        dispatches normally.
        """
        inst = _make_inst()
        inst.page.evaluate = AsyncMock(return_value={
            "tag": "button",
            "role": "button",
            "name": "Reachable",
            "masked_by": None,
            "mask_reason": "",
        })
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=50, y=60)
        assert result["success"] is True
        inst.page.mouse.click.assert_awaited_once_with(50.0, 60.0)


class TestClickXyConcurrency:
    """User control + lock invariants."""

    @pytest.mark.asyncio
    async def test_user_control_returns_conflict(self):
        """Spec test #9 — ``inst._user_control = True`` → conflict envelope."""
        inst = _make_inst(user_control=True)
        inst.page.evaluate = AsyncMock()
        mgr = _make_mgr_with(inst)

        result = await mgr.click_xy("a1", x=10, y=10)
        assert result["success"] is False
        assert result["error"]["code"] == "conflict"
        # Pre-check JS and the click must not have run.
        inst.page.evaluate.assert_not_awaited()
        inst.page.mouse.click.assert_not_awaited()


class TestClickXyJsHelper:
    """The element-from-point JS string is a key integration surface —
    we don't run it (no real browser) but we verify it is the helper that
    the manager evaluates and that it carries the cascade comment so a
    future maintainer doesn't naively rewrite the walk."""

    def test_js_helper_referenced_by_click_xy(self):
        """``click_xy`` must pass the helper to ``page.evaluate``."""
        from src.browser import service as svc
        assert hasattr(svc, "_JS_ELEMENT_FROM_POINT")
        assert "elementFromPoint" in svc._JS_ELEMENT_FROM_POINT
        # Pointer-events cascade comment must remain — explicit test that
        # the inside-out walk is documented.
        assert "pointer_events_decided" in svc._JS_ELEMENT_FROM_POINT
        assert "auto" in svc._JS_ELEMENT_FROM_POINT
        assert "none" in svc._JS_ELEMENT_FROM_POINT


class TestClickXyServerRoute:
    """``POST /browser/{agent_id}/click_xy`` route validation."""

    def test_route_accepts_numeric_xy(self):
        from starlette.testclient import TestClient

        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        async def _fake_click_xy(agent_id, x, y):
            return {
                "success": True,
                "data": {"clicked_at": {"x": x, "y": y},
                         "actual_element": {"tag": "div", "role": None, "name": ""}},
            }
        mgr.click_xy = _fake_click_xy
        app = create_browser_app(mgr)

        client = TestClient(app)
        resp = client.post(
            "/browser/test_agent/click_xy",
            json={"x": 100, "y": 200},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["clicked_at"] == {"x": 100, "y": 200}

    def test_route_rejects_missing_coords(self):
        from starlette.testclient import TestClient

        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        client = TestClient(app)
        resp = client.post("/browser/test_agent/click_xy", json={})
        # Missing keys are rejected via the type guard (None → not number).
        assert resp.status_code == 400

    def test_route_rejects_string_coords(self):
        from starlette.testclient import TestClient

        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        client = TestClient(app)
        resp = client.post(
            "/browser/test_agent/click_xy",
            json={"x": "10", "y": "20"},
        )
        assert resp.status_code == 400

    def test_route_rejects_bool_coords(self):
        from starlette.testclient import TestClient

        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        client = TestClient(app)
        resp = client.post(
            "/browser/test_agent/click_xy",
            json={"x": True, "y": 10},
        )
        assert resp.status_code == 400


class TestClickXyMeshAction:
    """The action name must be in ``KNOWN_BROWSER_ACTIONS`` so the mesh
    accepts it (input validation), AND ``can_browser_action`` must
    default-allow it for any agent with ``can_use_browser=true``."""

    def test_click_xy_in_known_browser_actions(self):
        """Spec test #12 — KNOWN_BROWSER_ACTIONS contains click_xy."""
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "click_xy" in KNOWN_BROWSER_ACTIONS

    def test_default_allow_grants_click_xy(self, tmp_path):
        """Spec test #13 — agent with ``browser_actions=None`` is granted."""
        from src.host.permissions import PermissionMatrix

        cfg = {
            "permissions": {
                "default-grant": {"can_use_browser": True},
            },
        }
        path = tmp_path / "permissions.json"
        path.write_text(json.dumps(cfg))
        matrix = PermissionMatrix(config_path=str(path))
        assert matrix.can_browser_action("default-grant", "click_xy") is True

    def test_restricted_actions_denies_click_xy(self, tmp_path):
        """Spec test #13 — ``browser_actions=['navigate']`` denies click_xy."""
        from src.host.permissions import PermissionMatrix

        cfg = {
            "permissions": {
                "restricted": {
                    "can_use_browser": True,
                    "browser_actions": ["navigate"],
                },
            },
        }
        path = tmp_path / "permissions.json"
        path.write_text(json.dumps(cfg))
        matrix = PermissionMatrix(config_path=str(path))
        assert matrix.can_browser_action("restricted", "click_xy") is False

    def test_can_use_browser_false_denies_click_xy(self, tmp_path):
        from src.host.permissions import PermissionMatrix

        cfg = {
            "permissions": {
                "no-browser": {"can_use_browser": False},
            },
        }
        path = tmp_path / "permissions.json"
        path.write_text(json.dumps(cfg))
        matrix = PermissionMatrix(config_path=str(path))
        assert matrix.can_browser_action("no-browser", "click_xy") is False


class TestClickXyAgentSkill:
    """Agent-side skill registration + invocation."""

    @pytest.mark.asyncio
    async def test_browser_click_xy_skill_calls_mesh_with_click_xy_action(self):
        from src.agent.builtins import browser_tool

        mesh = MagicMock()
        mesh.browser_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await browser_tool.browser_click_xy(
            x=12, y=34, mesh_client=mesh,
        )
        mesh.browser_command.assert_awaited_once_with(
            "click_xy", {"x": 12, "y": 34},
        )
        # Skills wrap responses in deep_redact — the success envelope round-trips.
        assert result["success"] is True

    def test_browser_click_xy_registered_as_skill(self):
        """Skill metadata must surface to the staging registry so the LLM
        sees it. Importing ``browser_tool`` triggers the @skill decorator
        which populates ``_skill_staging``."""
        import src.agent.builtins.browser_tool  # noqa: F401
        from src.agent.skills import _skill_staging

        assert "browser_click_xy" in _skill_staging
        meta = _skill_staging["browser_click_xy"]
        # Match plan §9.3 contract — parallel_safe=False because clicks
        # mutate shared browser state.
        assert meta["_parallel_safe"] is False
        assert "x" in meta["parameters"]
        assert "y" in meta["parameters"]

    def test_skill_description_documents_review_caveats(self):
        """Third-pass review locked these caveats into the description so
        agents are warned about the non-obvious failure modes:
          - CSS pixels (not device pixels — DPR irrelevant)
          - viewport-relative (not document-absolute; invalid across scrolls)
          - cross-origin iframe / shadow DOM limitations
          - post-click CAPTCHA detection is best-effort
        """
        import src.agent.builtins.browser_tool  # noqa: F401
        from src.agent.skills import _skill_staging

        desc = _skill_staging["browser_click_xy"]["description"].lower()
        assert "css pixel" in desc, "DPR clarification missing"
        assert "scroll" in desc, "scroll-position warning missing"
        assert "iframe" in desc, "cross-origin iframe limitation missing"
        assert "shadow" in desc, "shadow DOM limitation missing"
        assert "captcha" in desc, "post-click captcha caveat missing"
        # Specific viewport literals that lie about the fleet pool must
        # not be present — fleet uses 1920x1080, 1366x768, etc.
        assert "1280" not in desc, "stale viewport literal in description"
