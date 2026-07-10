"""Tests for the §11.5 in-house DataDome image-slider detect + solve.

Covers three layers, all behind the DEFAULT-OFF
``CAPTCHA_INHOUSE_SLIDER_ENABLED`` flag:

  * ``slider_solver.compute_slider_offset`` — pure CV gap detection on
    synthesized images (Pillow only, no browser).
  * ``captcha._classify_slider`` — positive detection of the SOLVABLE
    slider vs the ``/blocker`` behavioral wall (which must stay
    ``datadome-behavioral``).
  * ``BrowserManager._solve_slider`` — the gated solve dispatch: flag off
    or no X11 ⇒ ``None`` (escalate); flag on + X11 + CV offset ⇒ trusted
    ``_x11_drag`` + success re-probe ⇒ solved envelope.
  * ``BrowserManager._check_captcha`` integration — a solvable slider with
    the flag OFF still escalates via ``request_captcha_help`` (no
    regression to the current escalate-to-human behavior).
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image, ImageDraw

from src.browser import captcha as captcha_mod
from src.browser import service as svc
from src.browser.service import BrowserManager, CamoufoxInstance
from src.browser.slider_solver import compute_slider_offset

# ── helpers ────────────────────────────────────────────────────────────────


def _png_with_notch(
    *, width: int = 200, height: int = 100, notch_x: int, notch_w: int = 4,
    bg: int = 205, notch: int = 25,
) -> bytes:
    """Synthesize a light-gray grayscale PNG with a dark vertical bar.

    The bar's vertical edges are the strongest horizontal-gradient columns,
    so ``compute_slider_offset`` should peak within ~``notch_w`` px of
    ``notch_x``.
    """
    img = Image.new("L", (width, height), bg)
    draw = ImageDraw.Draw(img)
    left = notch_x - notch_w // 2
    draw.rectangle([left, 0, left + notch_w - 1, height - 1], fill=notch)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _flat_png(*, width: int = 200, height: int = 100, value: int = 205) -> bytes:
    img = Image.new("L", (width, height), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mk_inst(*, agent_id: str = "agent-1", x11_wid: int | None = None,
             page_url: str = "https://shop.example.com") -> CamoufoxInstance:
    page = MagicMock()
    page.url = page_url
    inst = CamoufoxInstance(agent_id, MagicMock(), MagicMock(), page)
    inst.x11_wid = x11_wid
    return inst


def _mk_element(*, box: dict | None = None, screenshot: bytes | None = None):
    el = MagicMock()
    el.bounding_box = AsyncMock(return_value=box)
    el.screenshot = AsyncMock(return_value=screenshot)
    return el


# ── §11.5 CV: compute_slider_offset ──────────────────────────────────────────


class TestComputeSliderOffset:
    def test_detects_known_notch(self):
        known_x = 130
        width = 200
        png = _png_with_notch(width=width, notch_x=known_x, notch_w=4)
        res = compute_slider_offset(png)
        assert res is not None
        gap_fraction, conf = res
        # Contract: a DPR-independent width fraction in [0, 1] that maps
        # back to the known column via ``fraction * (width - 1)``.
        assert 0.0 <= gap_fraction <= 1.0
        assert abs(gap_fraction * (width - 1) - known_x) <= 4
        assert conf > 0.7

    def test_dpr_independent_fraction(self):
        # The SAME logical notch at two image widths (1x and 2x) must yield
        # ~the same fraction — this is exactly the property that makes the
        # result safe to combine with CSS-pixel bounding boxes under any
        # ``deviceScaleFactor`` (element screenshots capture at DPR).
        res1 = compute_slider_offset(
            _png_with_notch(width=200, notch_x=130, notch_w=4),
        )
        res2 = compute_slider_offset(
            _png_with_notch(width=400, notch_x=260, notch_w=8),
        )
        assert res1 is not None and res2 is not None
        assert abs(res1[0] - res2[0]) < 0.02

    def test_flat_image_returns_none(self):
        assert compute_slider_offset(_flat_png()) is None

    def test_garbage_bytes_returns_none(self):
        assert compute_slider_offset(b"not-a-png") is None

    def test_left_margin_ignored(self):
        # A notch INSIDE the resting margin must not be reported — the piece
        # lives there. Only the right-of-margin region is searched.
        png = _png_with_notch(notch_x=3, notch_w=2)
        assert compute_slider_offset(png, left_margin=8) is None


# ── §11.5 classifier: _classify_slider ───────────────────────────────────────


def _mk_page_for_classify(*, probe: dict, frame_url: str = "",
                          handle_present: bool = False) -> MagicMock:
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=probe)
    if frame_url:
        frame = MagicMock()
        frame.url = frame_url
        frame.query_selector = AsyncMock(
            return_value=(MagicMock() if handle_present else None),
        )
        page.frames = [frame]
    else:
        page.frames = []
    return page


class TestClassifySlider:
    @pytest.mark.asyncio
    async def test_solvable_slider_present(self):
        page = _mk_page_for_classify(
            probe={"dd_iframe": True, "dd_blocker": False},
            frame_url="https://geo.captcha-delivery.com/captcha/?initialCid=x",
            handle_present=True,
        )
        assert await captcha_mod._classify_slider(page) == "datadome-slider"

    @pytest.mark.asyncio
    async def test_blocker_wall_stays_behavioral(self):
        # The /blocker wall must NOT be claimed as a solvable slider — it
        # stays datadome-behavioral (handled by _classify_behavioral).
        page = _mk_page_for_classify(
            probe={"dd_iframe": True, "dd_blocker": True},
            frame_url="https://geo.captcha-delivery.com/captcha/?initialCid=x",
            handle_present=True,
        )
        assert await captcha_mod._classify_slider(page) is None

    @pytest.mark.asyncio
    async def test_no_datadome_iframe(self):
        page = _mk_page_for_classify(probe={"dd_iframe": False, "dd_blocker": False})
        assert await captcha_mod._classify_slider(page) is None

    @pytest.mark.asyncio
    async def test_iframe_but_no_handle(self):
        page = _mk_page_for_classify(
            probe={"dd_iframe": True, "dd_blocker": False},
            frame_url="https://geo.captcha-delivery.com/captcha/?initialCid=x",
            handle_present=False,
        )
        assert await captcha_mod._classify_slider(page) is None

    @pytest.mark.asyncio
    async def test_evaluate_failure_returns_none(self):
        page = MagicMock()
        page.evaluate = AsyncMock(side_effect=RuntimeError("cross-origin"))
        assert await captcha_mod._classify_slider(page) is None


# ── §11.5 solve dispatch: _solve_slider ──────────────────────────────────────


@pytest.fixture()
def mgr(tmp_path):
    return BrowserManager(profiles_dir=str(tmp_path / "profiles"))


def _wire_frame(monkeypatch, *, bg_el, handle_el) -> None:
    """Point the service-side frame lookup at a mock puzzle frame."""
    frame = MagicMock()
    frame.url = "https://geo.captcha-delivery.com/captcha/?initialCid=x"

    async def _qs(sel):
        if "slider" in sel or "slide-button" in sel:
            return handle_el
        return bg_el

    frame.query_selector = _qs
    monkeypatch.setattr(svc, "_datadome_slider_frame", lambda page: frame)


class TestSolveSlider:
    @pytest.mark.asyncio
    async def test_flag_off_returns_none(self, mgr, monkeypatch):
        monkeypatch.delenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", raising=False)
        inst = _mk_inst(x11_wid=12345)
        mgr._x11_drag = AsyncMock()
        assert await mgr._solve_slider(inst, inst.page.url) is None
        mgr._x11_drag.assert_not_called()

    @pytest.mark.asyncio
    async def test_flag_on_but_no_x11_returns_none(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", "true")
        inst = _mk_inst(x11_wid=None)
        mgr._x11_drag = AsyncMock()
        assert await mgr._solve_slider(inst, inst.page.url) is None
        mgr._x11_drag.assert_not_called()

    @pytest.mark.asyncio
    async def test_cv_none_returns_none(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", "true")
        inst = _mk_inst(x11_wid=12345)
        mgr._x11_drag = AsyncMock()
        bg_el = _mk_element(
            box={"x": 10, "y": 60, "width": 260, "height": 160},
            screenshot=b"PNG",
        )
        handle_el = _mk_element(box={"x": 20, "y": 100, "width": 40, "height": 40})
        _wire_frame(monkeypatch, bg_el=bg_el, handle_el=handle_el)
        monkeypatch.setattr(
            "src.browser.slider_solver.compute_slider_offset",
            lambda *a, **k: None,
        )
        assert await mgr._solve_slider(inst, inst.page.url) is None
        mgr._x11_drag.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_drags_and_returns_envelope(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", "true")
        monkeypatch.setattr(svc.asyncio, "sleep", AsyncMock())
        inst = _mk_inst(x11_wid=12345)
        mgr._x11_drag = AsyncMock()

        bg_el = _mk_element(
            box={"x": 10, "y": 60, "width": 260, "height": 160},
            screenshot=b"PNGBYTES",
        )
        handle_el = _mk_element(box={"x": 20, "y": 100, "width": 40, "height": 40})
        _wire_frame(monkeypatch, bg_el=bg_el, handle_el=handle_el)
        # CV returns a WIDTH FRACTION (DPR-independent), not a pixel column.
        monkeypatch.setattr(
            "src.browser.slider_solver.compute_slider_offset",
            lambda *a, **k: (0.5, 0.9),
        )
        # Re-probe: slider gone ⇒ success.
        monkeypatch.setattr(svc, "_classify_slider", AsyncMock(return_value=None))
        monkeypatch.setattr(svc, "_record_captcha_audit_event", AsyncMock())

        envelope = await mgr._solve_slider(inst, inst.page.url)

        # Geometry: hx=40, hy=120; target_css_x=fraction(0.5)*bg_w(260)=130;
        # piece_resting_x=hx-bg_x=30; delta=130-30=100; end_x=40+100=140.
        mgr._x11_drag.assert_awaited_once_with(inst, 40, 120, 140, 120)
        assert envelope is not None
        assert envelope["kind"] == "datadome-slider"
        assert envelope["solver_attempted"] is True
        assert envelope["solver_outcome"] == "solved"
        assert envelope["next_action"] == "solved"
        assert envelope["solver_confidence"] == "high"

    @pytest.mark.asyncio
    async def test_success_reprobe_still_present_returns_none(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", "true")
        monkeypatch.setattr(svc.asyncio, "sleep", AsyncMock())
        inst = _mk_inst(x11_wid=12345)
        mgr._x11_drag = AsyncMock()
        bg_el = _mk_element(
            box={"x": 10, "y": 60, "width": 260, "height": 160},
            screenshot=b"PNGBYTES",
        )
        handle_el = _mk_element(box={"x": 20, "y": 100, "width": 40, "height": 40})
        _wire_frame(monkeypatch, bg_el=bg_el, handle_el=handle_el)
        monkeypatch.setattr(
            "src.browser.slider_solver.compute_slider_offset",
            lambda *a, **k: (0.5, 0.9),
        )
        # Re-probe: slider STILL present ⇒ DataDome rejected the drag.
        monkeypatch.setattr(
            svc, "_classify_slider", AsyncMock(return_value="datadome-slider"),
        )
        # Drag fired, but the result is "not solved" (escalate).
        assert await mgr._solve_slider(inst, inst.page.url) is None
        mgr._x11_drag.assert_awaited_once()


# ── §11.5 integration: _check_captcha preserves escalation with flag off ─────


class TestCheckCaptchaEscalationPreserved:
    @pytest.mark.asyncio
    async def test_slider_detected_flag_off_escalates(self, mgr, monkeypatch):
        monkeypatch.delenv("CAPTCHA_INHOUSE_SLIDER_ENABLED", raising=False)
        # A captcha selector matches so we enter classification.
        page = MagicMock()
        page.url = "https://shop.example.com"
        locator = MagicMock()
        locator.count = AsyncMock(return_value=1)
        page.locator = MagicMock(return_value=locator)
        inst = CamoufoxInstance("agent-1", MagicMock(), MagicMock(), page)
        inst.x11_wid = 12345

        # Positively classify a solvable slider; keep audit side-effect quiet.
        monkeypatch.setattr(
            svc, "_classify_slider", AsyncMock(return_value="datadome-slider"),
        )
        monkeypatch.setattr(svc, "_record_captcha_audit_event", AsyncMock())
        mgr._x11_drag = AsyncMock()

        envelope = await mgr._check_captcha(inst)

        # Flag off ⇒ _solve_slider returns None ⇒ identical escalation.
        assert envelope["captcha_found"] is True
        assert envelope["kind"] == "datadome-slider"
        assert envelope["solver_attempted"] is False
        assert envelope["next_action"] == "request_captcha_help"
        mgr._x11_drag.assert_not_called()
