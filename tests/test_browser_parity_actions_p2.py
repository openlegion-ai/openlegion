"""Behavior tests for the P2 human-parity browser actions.

Covers the actions stacked on top of the P1 parity surface:

  * ``right_click`` — trusted X11 button-3 click when an X11 window exists
    (``_x11_right_click_xy``), CDP ``page.mouse.click(..., button="right")``
    fallback otherwise. Accepts a ref or (x, y) coords.
  * ``read_clipboard`` / ``write_clipboard`` — the agent's X11 clipboard driven
    via ``xclip`` (NOT the page's ``navigator.clipboard`` API, which would let
    any visited page read/write the clipboard). A missing / failed / empty
    ``xclip`` must surface a structured error rather than hanging.
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
        # The reported method must reflect the path ACTUALLY taken (cdp),
        # not the x11 preference that raised and fell back.
        assert result["data"]["method"] == "cdp"

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
    async def test_ref_stale_exception_returns_ref_stale(self):
        """A RefStale raised while resolving the ref (DOM shifted since
        snapshot) must surface the ref_stale recovery path, not a misleading
        service_unavailable from the outer handler."""
        from src.browser.ref_handle import RefStale

        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=None)
        _patch_get_or_start(mgr, inst)
        mgr._locator_from_ref = AsyncMock(  # type: ignore[assignment]
            side_effect=RefStale("shadow host missing", ref="e3"),
        )

        result = await mgr.right_click("agent-parity", ref="e3")

        assert result["success"] is False
        assert result["error"]["code"] == "ref_stale"
        inst.page.mouse.click.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_mouseup_fires_in_finally_on_cancel(self, monkeypatch):
        """A CancelledError during the click dwell (after button 3 is DOWN)
        must still release the button — otherwise button 3 stays physically
        held for the rest of the session (X11 corruption), the same class of
        bug as the drag held-button leak."""
        import src.browser.service as svc

        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=555)

        async def _fake_move(_inst, x, y):
            return None

        monkeypatch.setattr(mgr, "_x11_move_to", _fake_move)

        calls: list[list[str]] = []

        class _FakeCompletedProc:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return _FakeCompletedProc()

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        # Raise CancelledError on the 2nd asyncio.sleep — the dwell between
        # mousedown (button held) and mouseup. The 1st (pre_click_settle)
        # sleeps normally.
        real_sleep = asyncio.sleep
        state = {"n": 0}

        async def _sleep(*a, **k):
            state["n"] += 1
            if state["n"] == 2:
                raise asyncio.CancelledError()
            return await real_sleep(0)

        monkeypatch.setattr(svc.asyncio, "sleep", _sleep)

        with pytest.raises(asyncio.CancelledError):
            await mgr._x11_right_click_xy(inst, 10, 20)

        # Despite the cancel mid-dwell, mouseup 3 MUST have fired.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "3" and calls[1][-1] == "3"


# ── Left-click (button 1) release + xdotool-failure guards ──────────────────


_DWELL_SENTINEL = 0.123456  # distinctive value so a patched sleep can pick the dwell


def _make_locator():
    loc = MagicMock()
    loc.bounding_box = AsyncMock(
        return_value={"x": 100, "y": 100, "width": 40, "height": 40},
    )
    return loc


async def _noop(*a, **k):
    return None


class TestX11ClickButtonRelease:
    """Guard tests for ``_x11_click`` (ref path), the primary-button analogue
    of ``TestX11RightClickXy``:

      * a CancelledError mid-dwell (uvicorn cancels the request task on client
        disconnect) must still release button 1 — otherwise it stays physically
        held for the rest of the session (X11 corruption);
      * a dropped xdotool mousedown/mouseup (wedged X server / lost WID) must
        surface as a RuntimeError so the caller's CDP fallback engages, never a
        silent success.
    """

    def _prep(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_x11_ensure_in_viewport", _noop)
        monkeypatch.setattr(mgr, "_x11_move_to", _noop)

    @pytest.mark.asyncio
    async def test_mouseup_fires_in_finally_on_cancel(self, monkeypatch):
        import src.browser.service as svc

        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=321)
        self._prep(mgr, monkeypatch)

        calls: list[list[str]] = []

        class _Ok:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return _Ok()

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        # Raise CancelledError from INSIDE the dwell sleep (button 1 is DOWN):
        # tag the dwell via a sentinel click_dwell() so it is picked out
        # regardless of how many settle-sleeps precede it.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(svc, "click_dwell", lambda: _DWELL_SENTINEL)

        async def _sleep(*a, **k):
            if a and a[0] == _DWELL_SENTINEL:
                raise asyncio.CancelledError()
            return await real_sleep(0)

        monkeypatch.setattr(svc.asyncio, "sleep", _sleep)

        with pytest.raises(asyncio.CancelledError):
            await mgr._x11_click(inst, _make_locator())

        # Despite the cancel mid-dwell, button 1 was released exactly once.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "1" and calls[1][-1] == "1"

    @pytest.mark.asyncio
    async def test_mouseup_fires_in_finally_on_cancel_during_mousedown(self, monkeypatch):
        """A CancelledError during the MOUSEDOWN executor await (the pressed
        window BEFORE the dwell) must still release button 1: the executor
        thread runs xdotool mousedown to completion (button pressed) even as
        the awaiting coroutine unwinds, so the mousedown must sit INSIDE the
        guarded block for the finally to cover it."""
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=321)
        self._prep(mgr, monkeypatch)

        calls: list[list[str]] = []

        class _Ok:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            # Record the call (the thread ran the subprocess), then simulate the
            # awaiting future being cancelled on the mousedown await only.
            calls.append(cmd)
            if cmd[1] == "mousedown":
                raise asyncio.CancelledError()
            return _Ok()

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(asyncio.CancelledError):
            await mgr._x11_click(inst, _make_locator())

        # Button pressed (thread ran) + await cancelled → the finally released
        # it anyway. Without the mousedown inside the try this would be
        # ["mousedown"] only (button stranded).
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "1" and calls[1][-1] == "1"

    @pytest.mark.asyncio
    async def test_nonzero_mousedown_raises(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=321)
        self._prep(mgr, monkeypatch)

        calls: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=1)

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(RuntimeError):
            await mgr._x11_click(inst, _make_locator())

        # A dropped mousedown raises so the caller falls back to CDP. The
        # finally still issues a best-effort mouseup — a harmless no-op (the
        # button was never pressed), strictly safer than skipping it.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]

    @pytest.mark.asyncio
    async def test_nonzero_mouseup_raises_after_best_effort_release(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=321)
        self._prep(mgr, monkeypatch)

        calls: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            # mousedown succeeds; every mouseup reports failure.
            rc = 0 if cmd[1] == "mousedown" else 1
            return types.SimpleNamespace(returncode=rc)

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(RuntimeError):
            await mgr._x11_click(inst, _make_locator())

        # A non-zero normal mouseup may not have released — the finally issues a
        # best-effort SECOND release, then the RuntimeError propagates so the
        # caller falls back to CDP.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup", "mouseup"]
        assert all(c[-1] == "1" for c in calls)


class TestX11ClickXyButtonRelease:
    """Same guards for ``_x11_click_xy`` (coord path)."""

    @pytest.mark.asyncio
    async def test_mouseup_fires_in_finally_on_cancel(self, monkeypatch):
        import src.browser.service as svc

        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=654)
        monkeypatch.setattr(mgr, "_x11_move_to", _noop)

        calls: list[list[str]] = []

        class _Ok:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return _Ok()

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        real_sleep = asyncio.sleep
        monkeypatch.setattr(svc, "click_dwell", lambda: _DWELL_SENTINEL)

        async def _sleep(*a, **k):
            if a and a[0] == _DWELL_SENTINEL:
                raise asyncio.CancelledError()
            return await real_sleep(0)

        monkeypatch.setattr(svc.asyncio, "sleep", _sleep)

        with pytest.raises(asyncio.CancelledError):
            await mgr._x11_click_xy(inst, 10, 20)

        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "1" and calls[1][-1] == "1"

    @pytest.mark.asyncio
    async def test_mouseup_fires_in_finally_on_cancel_during_mousedown(self, monkeypatch):
        """CancelledError during the mousedown await must still release button
        1 (coord path). See the ``_x11_click`` analogue for the full rationale."""
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=654)
        monkeypatch.setattr(mgr, "_x11_move_to", _noop)

        calls: list[list[str]] = []

        class _Ok:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[1] == "mousedown":
                raise asyncio.CancelledError()
            return _Ok()

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(asyncio.CancelledError):
            await mgr._x11_click_xy(inst, 10, 20)

        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]
        assert calls[0][-1] == "1" and calls[1][-1] == "1"

    @pytest.mark.asyncio
    async def test_nonzero_mousedown_raises(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=654)
        monkeypatch.setattr(mgr, "_x11_move_to", _noop)

        calls: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=1)

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(RuntimeError):
            await mgr._x11_click_xy(inst, 10, 20)

        # Best-effort emergency mouseup fires (harmless no-op) then the raise
        # propagates so the caller falls back to CDP.
        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup"]

    @pytest.mark.asyncio
    async def test_nonzero_mouseup_raises_after_best_effort_release(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance(x11_wid=654)
        monkeypatch.setattr(mgr, "_x11_move_to", _noop)

        calls: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            rc = 0 if cmd[1] == "mousedown" else 1
            return types.SimpleNamespace(returncode=rc)

        monkeypatch.setattr(
            "src.browser.service.subprocess",
            types.SimpleNamespace(run=_fake_run),
        )

        with pytest.raises(RuntimeError):
            await mgr._x11_click_xy(inst, 10, 20)

        verbs = [c[1] for c in calls]
        assert verbs == ["mousedown", "mouseup", "mouseup"]
        assert all(c[-1] == "1" for c in calls)


# ── Clipboard ──────────────────────────────────────────────────────────────


class _FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_subprocess_run(monkeypatch, calls: list, result):
    """Patch ONLY ``subprocess.run`` on the service module (keeping DEVNULL /
    TimeoutExpired intact). ``result`` is returned, or raised if it is an
    exception instance."""

    def _run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("src.browser.service.subprocess.run", _run)


class TestClipboard:
    @pytest.mark.asyncio
    async def test_write_clipboard_pipes_to_xclip(self, monkeypatch):
        """write_clipboard must pipe the text to ``xclip -selection clipboard
        -i`` — the X clipboard, NOT the page's navigator.clipboard API."""
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(monkeypatch, calls, _FakeCompleted(returncode=0))

        result = await mgr.write_clipboard("agent-parity", text="hello world")

        assert result["success"] is True
        assert result["data"]["written"] == len("hello world")
        # navigator.clipboard was NOT used.
        inst.page.evaluate.assert_not_called()
        assert len(calls) == 1
        assert calls[0]["cmd"] == ["xclip", "-selection", "clipboard", "-i"]
        assert calls[0]["kwargs"]["input"] == b"hello world"

    @pytest.mark.asyncio
    async def test_read_clipboard_reads_from_xclip(self, monkeypatch):
        """read_clipboard must return the stdout of ``xclip -selection
        clipboard -o``."""
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(
            monkeypatch, calls,
            _FakeCompleted(returncode=0, stdout=b"copied text"),
        )

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is True
        assert result["data"]["text"] == "copied text"
        inst.page.evaluate.assert_not_called()
        assert calls[0]["cmd"] == ["xclip", "-selection", "clipboard", "-o"]

    @pytest.mark.asyncio
    async def test_read_clipboard_missing_xclip_is_structured_error(self, monkeypatch):
        """A missing xclip binary must fail safe with a structured error, not
        raise (fail-closed for the feature)."""
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(
            monkeypatch, calls, FileNotFoundError("xclip"),
        )

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_read_clipboard_empty_is_structured_error(self, monkeypatch):
        """xclip exits non-zero on an empty clipboard (no STRING target) — that
        must surface as an error, never a phantom empty string or a hang."""
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(
            monkeypatch, calls,
            _FakeCompleted(returncode=1, stderr=b"Error: target STRING not available"),
        )

        result = await mgr.read_clipboard("agent-parity")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_write_clipboard_missing_xclip_is_structured_error(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(
            monkeypatch, calls, FileNotFoundError("xclip"),
        )

        result = await mgr.write_clipboard("agent-parity", text="x")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_write_clipboard_nonzero_is_structured_error(self, monkeypatch):
        mgr = _make_manager()
        inst = _FakeInstance()
        _patch_get_or_start(mgr, inst)
        calls: list = []
        _patch_subprocess_run(
            monkeypatch, calls, _FakeCompleted(returncode=1),
        )

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
