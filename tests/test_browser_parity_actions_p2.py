"""Behavior tests for the P2 human-parity browser actions.

Covers the actions stacked on top of the P1 parity surface:

  * ``right_click`` — trusted X11 button-3 click when an X11 window exists
    (``_x11_right_click_xy``), CDP ``page.mouse.click(..., button="right")``
    fallback otherwise. Accepts a ref or (x, y) coords.
  * ``read_clipboard`` / ``write_clipboard`` — the async Clipboard API driven
    via ``page.evaluate``. A rejected evaluate (insecure context / permission)
    must surface a structured error rather than hanging.
  * ``wait_for_network_idle`` — ``page.wait_for_load_state("networkidle")``;
    a timeout returns a NON-FATAL ``idle=False`` result instead of raising.

``BrowserManager._start_browser`` imports ``camoufox`` (not installed in CI),
so we NEVER go through it — every test drives the handler methods directly
against a fake instance, patching ``get_or_start`` to return it (mirrors
``tests/test_browser_parity_actions``).
"""

from __future__ import annotations

import asyncio
import tempfile
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager


def _make_manager() -> BrowserManager:
    root = tempfile.mkdtemp(prefix="ol_parity_p2_")
    return BrowserManager(profiles_dir=root)


class _FakeInstance:
    """Minimal CamoufoxInstance duck-type for the P2 parity handlers."""

    def __init__(self, *, x11_wid=None):
        self.agent_id = "agent-parity"
        self.x11_wid = x11_wid
        self._user_control = False
        self.refs: dict = {}
        self._lock = asyncio.Lock()
        self.page = MagicMock()
        self.page.mouse = MagicMock()
        self.page.mouse.click = AsyncMock()
        self.page.evaluate = AsyncMock()
        self.page.wait_for_load_state = AsyncMock()

    @property
    def lock(self):
        return self._lock

    def touch(self):
        pass

    def subprocess_env(self):
        # xdotool env for the X11 path; value is irrelevant here because
        # subprocess.run is monkeypatched in the X11 test.
        return None


def _patch_get_or_start(mgr: BrowserManager, inst: _FakeInstance) -> None:
    mgr.get_or_start = AsyncMock(return_value=inst)  # type: ignore[assignment]


# ── Right-click ────────────────────────────────────────────────────────────


class TestRightClick:
    @pytest.mark.asyncio
    async def test_x11_path_used_when_window_present(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=123)
        _patch_get_or_start(mgr, inst)
        mgr._x11_right_click_xy = AsyncMock()  # type: ignore[assignment]

        result = await mgr.right_click("agent-parity", x=42, y=84)

        assert result["success"] is True
        assert result["data"]["method"] == "x11"
        assert result["data"]["at"] == [42, 84]
        mgr._x11_right_click_xy.assert_awaited_once_with(inst, 42, 84)
        # CDP mouse untouched on the trusted path.
        inst.page.mouse.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_cdp_fallback_uses_right_button(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)

        result = await mgr.right_click("agent-parity", x=5, y=6)

        assert result["success"] is True
        assert result["data"]["method"] == "cdp"
        inst.page.mouse.click.assert_awaited_once_with(5, 6, button="right")

    @pytest.mark.asyncio
    async def test_x11_failure_falls_back_to_cdp(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=999)
        _patch_get_or_start(mgr, inst)
        mgr._x11_right_click_xy = AsyncMock(  # type: ignore[assignment]
            side_effect=RuntimeError("xdotool missing"),
        )

        result = await mgr.right_click("agent-parity", x=1, y=2)

        # Fell back to CDP with the secondary button.
        assert result["success"] is True
        inst.page.mouse.click.assert_awaited_once_with(1, 2, button="right")

    @pytest.mark.asyncio
    async def test_ref_resolves_bounding_box(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)

        loc = MagicMock()
        loc.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 100, "width": 40, "height": 40},
        )
        mgr._locator_from_ref = AsyncMock(  # type: ignore[assignment]
            return_value=loc,
        )

        result = await mgr.right_click("agent-parity", ref="e3")

        assert result["success"] is True
        px, py = result["data"]["at"]
        # Point lands inside the element's bbox (Fitts' sampler).
        assert 100 <= px <= 140 and 100 <= py <= 140
        inst.page.mouse.click.assert_awaited_once_with(px, py, button="right")

    @pytest.mark.asyncio
    async def test_stale_ref_rejected(self):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)
        mgr._locator_from_ref = AsyncMock(  # type: ignore[assignment]
            return_value=None,
        )

        result = await mgr.right_click("agent-parity", ref="gone")

        assert result["success"] is False
        assert result["error"]["code"] == "ref_stale"

    @pytest.mark.asyncio
    async def test_missing_args_rejected(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.right_click("agent-parity")  # no ref, no coords

        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


class TestX11RightClickXy:
    @pytest.mark.asyncio
    async def test_press_release_uses_button_3(self, monkeypatch):
        """``_x11_right_click_xy`` must move to the point, then press + release
        xdotool button 3 (the secondary/context-menu button)."""
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

        await mgr._x11_right_click_xy(inst, 30, 60)

        assert moves == [(30, 60)]
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        # Both xdotool calls target button 3, not 1.
        assert calls[0][-1] == "3" and calls[1][-1] == "3"


# ── Clipboard ──────────────────────────────────────────────────────────────


class TestClipboard:
    @pytest.mark.asyncio
    async def test_write_clipboard_calls_evaluate(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.write_clipboard("agent-parity", text="hello world")

        assert result["success"] is True
        assert result["data"]["written"] == len("hello world")
        inst.page.evaluate.assert_awaited_once_with(
            "(t) => navigator.clipboard.writeText(t)", "hello world",
        )

    @pytest.mark.asyncio
    async def test_read_clipboard_returns_text(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.evaluate = AsyncMock(return_value="copied text")
        _patch_get_or_start(mgr, inst)

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is True
        assert result["data"]["text"] == "copied text"
        inst.page.evaluate.assert_awaited_once_with(
            "() => navigator.clipboard.readText()",
        )

    @pytest.mark.asyncio
    async def test_read_clipboard_none_becomes_empty(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.evaluate = AsyncMock(return_value=None)
        _patch_get_or_start(mgr, inst)

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is True
        assert result["data"]["text"] == ""

    @pytest.mark.asyncio
    async def test_read_clipboard_rejection_surfaces_error(self):
        """An insecure-context / permission rejection must be a structured
        error, never a hang."""
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.evaluate = AsyncMock(
            side_effect=RuntimeError("Clipboard read blocked (insecure context)"),
        )
        _patch_get_or_start(mgr, inst)

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_write_clipboard_rejection_surfaces_error(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.evaluate = AsyncMock(
            side_effect=RuntimeError("writeText blocked"),
        )
        _patch_get_or_start(mgr, inst)

        result = await mgr.write_clipboard("agent-parity", text="x")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"


# ── Network-idle wait ──────────────────────────────────────────────────────


class TestWaitForNetworkIdle:
    @pytest.mark.asyncio
    async def test_idle_reached(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.wait_for_network_idle("agent-parity", timeout=5000)

        assert result["success"] is True
        assert result["data"]["idle"] is True
        assert result["data"]["timeout_ms"] == 5000
        inst.page.wait_for_load_state.assert_awaited_once_with(
            "networkidle", timeout=5000,
        )

    @pytest.mark.asyncio
    async def test_timeout_is_non_fatal(self):
        """A Playwright timeout (message contains 'Timeout ... exceeded') must
        return a structured idle=False, NOT raise."""
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.wait_for_load_state = AsyncMock(
            side_effect=RuntimeError("Timeout 10000ms exceeded"),
        )
        _patch_get_or_start(mgr, inst)

        result = await mgr.wait_for_network_idle("agent-parity")

        assert result["success"] is True
        assert result["data"]["idle"] is False
        assert "note" in result["data"]

    @pytest.mark.asyncio
    async def test_timeout_capped_at_30s(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)

        result = await mgr.wait_for_network_idle("agent-parity", timeout=99999)

        assert result["success"] is True
        assert result["data"]["timeout_ms"] == 30000
        inst.page.wait_for_load_state.assert_awaited_once_with(
            "networkidle", timeout=30000,
        )

    @pytest.mark.asyncio
    async def test_non_timeout_error_is_fatal(self):
        mgr = _make_manager()
        inst = _FakeInstance()
        inst.page.wait_for_load_state = AsyncMock(
            side_effect=RuntimeError("target crashed"),
        )
        _patch_get_or_start(mgr, inst)

        result = await mgr.wait_for_network_idle("agent-parity")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"


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
    async def test_right_click_prefers_ref(self):
        from src.agent.builtins.browser_tool import browser_right_click

        cap: dict = {}
        await browser_right_click(ref="e9", x=1, y=2, mesh_client=_mesh(cap))
        assert cap["action"] == "right_click"
        assert cap["params"] == {"ref": "e9"}

    @pytest.mark.asyncio
    async def test_right_click_coords(self):
        from src.agent.builtins.browser_tool import browser_right_click

        cap: dict = {}
        await browser_right_click(x=11, y=22, mesh_client=_mesh(cap))
        assert cap["action"] == "right_click"
        assert cap["params"] == {"x": 11, "y": 22}

    @pytest.mark.asyncio
    async def test_write_clipboard_tool(self):
        from src.agent.builtins.browser_tool import browser_write_clipboard

        cap: dict = {}
        await browser_write_clipboard(text="paste me", mesh_client=_mesh(cap))
        assert cap["action"] == "write_clipboard"
        assert cap["params"] == {"text": "paste me"}

    @pytest.mark.asyncio
    async def test_read_clipboard_tool(self):
        from src.agent.builtins.browser_tool import browser_read_clipboard

        cap: dict = {}
        await browser_read_clipboard(mesh_client=_mesh(cap))
        assert cap["action"] == "read_clipboard"
        assert cap["params"] == {}

    @pytest.mark.asyncio
    async def test_wait_for_network_idle_tool(self):
        from src.agent.builtins.browser_tool import browser_wait_for_network_idle

        cap: dict = {}
        await browser_wait_for_network_idle(timeout=8000, mesh_client=_mesh(cap))
        assert cap["action"] == "wait_for_network_idle"
        assert cap["params"] == {"timeout": 8000}
