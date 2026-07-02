"""Behavior tests for the human-parity browser actions.

Covers the three actions added on top of the base browser surface:

  * ``set_dialog_policy`` / ``_on_dialog`` — native JS dialog handling. The
    handler must ALWAYS resolve the dialog (accept or dismiss) — a pending
    dialog wedges the page's JS loop — and record the message either way.
  * ``grant_permissions`` — Firefox-scoped permission grants. Camera /
    microphone / any non-Firefox string must be rejected with a structured
    error BEFORE Playwright can raise a raw ``Unknown permission``.
  * ``set_geolocation`` — sets context geolocation AND grants the
    geolocation permission (Firefox needs both).
  * ``drag`` — trusted-X11 drag when an X11 window exists, CDP mouse
    fallback otherwise. ``_x11_drag`` presses button 1 down, moves with it
    held, then releases.

``BrowserManager._start_browser`` imports ``camoufox`` on its first line
(not installed in CI), so we NEVER go through it — every test drives the
handler methods directly against a fake instance, patching
``get_or_start`` to return it (mirrors ``tests/test_binding_cookie_coherence``).
"""

from __future__ import annotations

import tempfile
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import _FIREFOX_GRANTABLE_PERMISSIONS, BrowserManager


def _make_manager() -> BrowserManager:
    root = tempfile.mkdtemp(prefix="ol_parity_")
    return BrowserManager(profiles_dir=root)


class _FakeInstance:
    """Minimal CamoufoxInstance duck-type for the parity handlers."""

    def __init__(self, *, x11_wid=None):
        import asyncio

        self.agent_id = "agent-parity"
        self.x11_wid = x11_wid
        self._user_control = False
        self.dialog_policy = {"action": "dismiss", "text": ""}
        self.last_dialog_message = None
        self.refs: dict = {}
        self._lock = asyncio.Lock()
        # Context / page are MagicMocks with AsyncMock async methods.
        self.context = MagicMock()
        self.context.grant_permissions = AsyncMock()
        self.context.set_geolocation = AsyncMock()
        self.page = MagicMock()
        self.page.mouse = MagicMock()
        self.page.mouse.move = AsyncMock()
        self.page.mouse.down = AsyncMock()
        self.page.mouse.up = AsyncMock()

    @property
    def lock(self):
        return self._lock

    def touch(self):
        pass

    def subprocess_env(self):
        # xdotool env for the X11 drag path; value is irrelevant here
        # because subprocess.run is monkeypatched in the drag test.
        return None


def _patch_get_or_start(mgr: BrowserManager, inst: _FakeInstance) -> None:
    mgr.get_or_start = AsyncMock(return_value=inst)  # type: ignore[assignment]


# ── Native JS dialog handling ─────────────────────────────────────────────


class _FakeDialog:
    def __init__(self, message: str, dtype: str):
        self.message = message
        self.type = dtype
        self.accept = AsyncMock()
        self.dismiss = AsyncMock()


class TestOnDialog:
    @pytest.mark.asyncio
    async def test_accept_policy_accepts_and_records(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.dialog_policy = {"action": "accept", "text": ""}
        dialog = _FakeDialog("Delete this item?", "confirm")

        await mgr._on_dialog(inst, dialog)

        dialog.accept.assert_awaited_once()
        dialog.dismiss.assert_not_called()
        assert inst.last_dialog_message == {
            "message": "Delete this item?",
            "type": "confirm",
        }

    @pytest.mark.asyncio
    async def test_dismiss_default_dismisses_but_still_records(self):
        mgr = _make_manager()
        inst = _FakeInstance()  # default policy is dismiss
        dialog = _FakeDialog("Leave site? Changes may not be saved.", "beforeunload")

        await mgr._on_dialog(inst, dialog)

        dialog.dismiss.assert_awaited_once()
        dialog.accept.assert_not_called()
        # Message stored even though we dismissed it.
        assert inst.last_dialog_message["message"].startswith("Leave site?")
        assert inst.last_dialog_message["type"] == "beforeunload"

    @pytest.mark.asyncio
    async def test_respond_passes_prompt_text(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.dialog_policy = {"action": "respond", "text": "Ada Lovelace"}
        dialog = _FakeDialog("Your name?", "prompt")

        await mgr._on_dialog(inst, dialog)

        dialog.accept.assert_awaited_once_with("Ada Lovelace")

    @pytest.mark.asyncio
    async def test_always_resolves_even_when_accept_raises(self):
        """A pending dialog wedges the page — a failed accept must fall back
        to a dismiss so the dialog is never left pending."""
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.dialog_policy = {"action": "accept", "text": ""}
        dialog = _FakeDialog("boom", "alert")
        dialog.accept = AsyncMock(side_effect=RuntimeError("target closed"))

        await mgr._on_dialog(inst, dialog)

        dialog.dismiss.assert_awaited_once()


class TestSetDialogPolicy:
    @pytest.mark.asyncio
    async def test_mutates_policy_and_returns_last_message(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.last_dialog_message = {"message": "prior", "type": "confirm"}
        _patch_get_or_start(mgr, inst)

        result = await mgr.set_dialog_policy("agent-parity", action="accept")

        assert result["success"] is True
        assert inst.dialog_policy == {"action": "accept", "text": ""}
        assert result["data"]["last_dialog"] == {"message": "prior", "type": "confirm"}

    @pytest.mark.asyncio
    async def test_rejects_bad_action(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.set_dialog_policy("agent-parity", action="explode")

        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        # get_or_start must not even run for a pure input error.
        mgr.get_or_start.assert_not_called()


# ── Permission grants + geolocation ───────────────────────────────────────


class TestGrantPermissions:
    @pytest.mark.asyncio
    async def test_valid_firefox_permissions_granted(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.grant_permissions(
            "agent-parity",
            permissions=["geolocation", "notifications"],
            origin="https://example.com",
        )

        assert result["success"] is True
        inst.context.grant_permissions.assert_awaited_once_with(
            ["geolocation", "notifications"],
            origin="https://example.com",
        )

    @pytest.mark.asyncio
    async def test_camera_rejected_without_calling_playwright(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.grant_permissions(
            "agent-parity",
            permissions=["camera"],
        )

        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "camera" in result["error"]["message"]
        # Never reached Playwright — no context call at all.
        mgr.get_or_start.assert_not_called()
        inst.context.grant_permissions.assert_not_called()

    @pytest.mark.asyncio
    async def test_microphone_rejected(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.grant_permissions(
            "agent-parity",
            permissions=["geolocation", "microphone"],
        )

        assert result["success"] is False
        inst.context.grant_permissions.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_rejected(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.grant_permissions("agent-parity", permissions=[])
        assert result["success"] is False

    def test_allowlist_matches_firefox_reality(self):
        # These are the ONLY four web permissions Playwright's Firefox engine
        # maps (ffBrowser.js ``webPermissionToProtocol``): geolocation, push,
        # persistent-storage, and notifications (→ desktop-notification).
        # screen-wake-lock is Chromium-only — it is NOT in the FF map and
        # ``grant_permissions`` raises "Unknown permission" for it, so it must
        # never be in the allowlist.
        assert _FIREFOX_GRANTABLE_PERMISSIONS == frozenset(
            {
                "geolocation",
                "notifications",
                "persistent-storage",
                "push",
            }
        )
        assert "screen-wake-lock" not in _FIREFOX_GRANTABLE_PERMISSIONS
        assert "camera" not in _FIREFOX_GRANTABLE_PERMISSIONS
        assert "microphone" not in _FIREFOX_GRANTABLE_PERMISSIONS


class TestSetGeolocation:
    @pytest.mark.asyncio
    async def test_sets_coords_and_grants_geolocation(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.set_geolocation(
            "agent-parity",
            latitude=37.7749,
            longitude=-122.4194,
            accuracy=50,
        )

        assert result["success"] is True
        inst.context.set_geolocation.assert_awaited_once_with(
            {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "accuracy": 50.0,
            }
        )
        # Firefox needs the geolocation permission granted too.
        inst.context.grant_permissions.assert_awaited_once_with(["geolocation"])
        # Truth-in-advertising note about geo.enabled=False is surfaced.
        assert "geo.enabled" in result["data"]["note"]

    @pytest.mark.asyncio
    async def test_rejects_non_numeric_coords(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.set_geolocation(
            "agent-parity",
            latitude="north",
            longitude=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        mgr.get_or_start.assert_not_called()


# ── Drag-and-drop ─────────────────────────────────────────────────────────


class TestDragDispatch:
    @pytest.mark.asyncio
    async def test_x11_path_used_when_window_present(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=123)
        _patch_get_or_start(mgr, inst)
        mgr._x11_drag = AsyncMock()  # type: ignore[assignment]

        result = await mgr.drag(
            "agent-parity",
            source_x=10,
            source_y=20,
            target_x=100,
            target_y=200,
        )

        assert result["success"] is True
        assert result["data"]["method"] == "x11"
        mgr._x11_drag.assert_awaited_once_with(inst, 10, 20, 100, 200)
        # CDP mouse untouched.
        inst.page.mouse.down.assert_not_called()

    @pytest.mark.asyncio
    async def test_cdp_fallback_when_no_window(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)

        result = await mgr.drag(
            "agent-parity",
            source_x=5,
            source_y=6,
            target_x=7,
            target_y=8,
        )

        assert result["success"] is True
        assert result["data"]["method"] == "cdp"
        inst.page.mouse.move.assert_any_await(5, 6)
        inst.page.mouse.down.assert_awaited_once()
        inst.page.mouse.move.assert_any_await(7, 8)
        inst.page.mouse.up.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ref_pair_resolves_bounding_boxes(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)

        src_loc = MagicMock()
        src_loc.bounding_box = AsyncMock(
            return_value={"x": 0, "y": 0, "width": 40, "height": 40},
        )
        tgt_loc = MagicMock()
        tgt_loc.bounding_box = AsyncMock(
            return_value={"x": 200, "y": 200, "width": 40, "height": 40},
        )
        locators = {"e1": src_loc, "e2": tgt_loc}
        mgr._locator_from_ref = AsyncMock(  # type: ignore[assignment]
            side_effect=lambda inst, ref: locators.get(ref),
        )

        result = await mgr.drag(
            "agent-parity",
            source_ref="e1",
            target_ref="e2",
        )

        assert result["success"] is True
        # Points land inside each element's bbox (Fitts' sampler).
        ax, ay = result["data"]["from"]
        bx, by = result["data"]["to"]
        assert 0 <= ax <= 40 and 0 <= ay <= 40
        assert 200 <= bx <= 240 and 200 <= by <= 240
        inst.page.mouse.up.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_args_rejected(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.drag("agent-parity")  # nothing supplied
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


class TestX11Drag:
    @pytest.mark.asyncio
    async def test_press_move_release_sequence(self, monkeypatch):
        """``_x11_drag`` must move to source, press button 1, move to target
        with it held, then release — verified via the xdotool call log."""
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=777)

        moves: list[tuple[int, int]] = []

        async def _fake_move(_inst, x, y):
            moves.append((x, y))

        monkeypatch.setattr(mgr, "_x11_move_to", _fake_move)

        calls: list[list[str]] = []

        class _FakeCompleted:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return _FakeCompleted()

        fake_subprocess = types.SimpleNamespace(run=_fake_run)
        monkeypatch.setattr("src.browser.service.subprocess", fake_subprocess)

        await mgr._x11_drag(inst, 10, 20, 300, 400)

        # Two moves: source then target.
        assert moves == [(10, 20), (300, 400)]
        # xdotool: mousedown 1 THEN mouseup 1, in that order.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "1" and calls[1][-1] == "1"


# ── Agent-side @tool forwarding ────────────────────────────────────────────


def _mesh(action_capture: dict):
    mc = MagicMock()

    async def _cmd(action, params):
        action_capture["action"] = action
        action_capture["params"] = params
        return {"success": True}

    mc.browser_command = AsyncMock(side_effect=_cmd)
    return mc


class TestAgentToolForwarding:
    @pytest.mark.asyncio
    async def test_set_dialog_policy_tool(self):
        from src.agent.builtins.browser_tool import browser_set_dialog_policy

        cap: dict = {}
        await browser_set_dialog_policy(
            action="respond",
            text="hi",
            mesh_client=_mesh(cap),
        )
        assert cap["action"] == "set_dialog_policy"
        assert cap["params"] == {"action": "respond", "text": "hi"}

    @pytest.mark.asyncio
    async def test_drag_tool_prefers_refs(self):
        from src.agent.builtins.browser_tool import browser_drag

        cap: dict = {}
        await browser_drag(
            source_ref="e1",
            target_ref="e2",
            mesh_client=_mesh(cap),
        )
        assert cap["action"] == "drag"
        assert cap["params"] == {"source_ref": "e1", "target_ref": "e2"}

    @pytest.mark.asyncio
    async def test_drag_tool_coords(self):
        from src.agent.builtins.browser_tool import browser_drag

        cap: dict = {}
        await browser_drag(
            source_x=1,
            source_y=2,
            target_x=3,
            target_y=4,
            mesh_client=_mesh(cap),
        )
        assert cap["action"] == "drag"
        assert cap["params"] == {
            "source_x": 1,
            "source_y": 2,
            "target_x": 3,
            "target_y": 4,
        }

    @pytest.mark.asyncio
    async def test_grant_permissions_tool_omits_empty_origin(self):
        from src.agent.builtins.browser_tool import browser_grant_permissions

        cap: dict = {}
        await browser_grant_permissions(
            permissions=["geolocation"],
            mesh_client=_mesh(cap),
        )
        assert cap["action"] == "grant_permissions"
        assert cap["params"] == {"permissions": ["geolocation"]}

    @pytest.mark.asyncio
    async def test_set_geolocation_tool_includes_accuracy(self):
        from src.agent.builtins.browser_tool import browser_set_geolocation

        cap: dict = {}
        await browser_set_geolocation(
            latitude=1.5,
            longitude=2.5,
            accuracy=10,
            mesh_client=_mesh(cap),
        )
        assert cap["action"] == "set_geolocation"
        assert cap["params"] == {
            "latitude": 1.5,
            "longitude": 2.5,
            "accuracy": 10,
        }
