"""Tests for the set-of-marks screenshot annotation feature.

Two layers, both hermetic (no Docker, no real browser):

1. The pure ``_draw_set_of_marks`` helper — valid-PNG output, graceful
   no-op identity, and DPR-correct coordinate math.
2. ``BrowserManager.screenshot_marks`` with the a11y/locator machinery
   mocked, plus the ``browser_screenshot_marks`` agent tool wrapper's
   ``_image`` extraction and param plumbing.
"""

import atexit
import base64
import shutil
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.ref_handle import RefHandle
from src.browser.service import (
    BrowserManager,
    CamoufoxInstance,
    _draw_set_of_marks,
)

_PROFILES_ROOT = tempfile.mkdtemp(prefix="ol_test_set_of_marks_")
atexit.register(shutil.rmtree, _PROFILES_ROOT, ignore_errors=True)


def _make_png_bytes(width: int = 200, height: int = 200) -> bytes:
    """Render a real PNG of a known pixel size with deterministic content."""
    from PIL import Image
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 3 % 256, y * 3 % 256, (x + y) % 256)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _open(png_bytes: bytes):
    from PIL import Image
    return Image.open(BytesIO(png_bytes)).convert("RGB")


class TestDrawSetOfMarks:
    """The pure ``_draw_set_of_marks`` helper."""

    def test_valid_png_output_differs_from_input(self):
        """With marks + Pillow present, output is a NEW valid PNG object."""
        png = _make_png_bytes(200, 200)
        marks = [{"mark": 1, "box": (10, 10, 20, 20)}]
        out = _draw_set_of_marks(png, marks, css_w=200, css_h=200, agent_id="a1")
        # New object (identity differs — annotation happened).
        assert out is not png
        # Still a valid PNG of the same dimensions.
        img = _open(out)
        assert img.size == (200, 200)

    def test_empty_marks_returns_same_object(self):
        """No marks → nothing to draw → SAME object (annotated=False signal)."""
        png = _make_png_bytes(64, 48)
        out = _draw_set_of_marks(png, [], css_w=64, css_h=48, agent_id="a1")
        assert out is png

    def test_zero_css_dims_returns_same_object(self):
        """Unusable CSS viewport (0) → SAME object, no divide-by-zero."""
        png = _make_png_bytes(64, 48)
        marks = [{"mark": 1, "box": (1, 1, 2, 2)}]
        assert _draw_set_of_marks(png, marks, 0, 0, agent_id="a1") is png

    def test_dpr_correct_coordinate_scaling(self):
        """A retina 2× PNG (image 200×200, CSS viewport 100×100) must scale
        each CSS box up by 2× before drawing. A CSS box at (10,10,20,20)
        lands its outline at image pixels (20,20)-(60,60); assert the drawn
        red edge is there and a far region is untouched."""
        png = _make_png_bytes(200, 200)
        original = _open(png)
        marks = [{"mark": 1, "box": (10, 10, 20, 20)}]
        out = _draw_set_of_marks(png, marks, css_w=100, css_h=100, agent_id="a1")
        assert out is not png
        img = _open(out)

        # Top edge of the DPR-scaled rectangle (y≈20, x within 20..60) is red.
        assert img.getpixel((40, 20)) == (255, 0, 0)
        # A pixel well outside the box + label is untouched vs the original.
        assert img.getpixel((150, 150)) == original.getpixel((150, 150))
        # Sanity: if scaling were 1× (bug), (40,20) would NOT be on the
        # rectangle (which would sit at 10..30) — the assertion above pins it.

    def test_bad_box_shape_skipped_not_fatal(self):
        """A malformed box entry is skipped; a good one still draws."""
        png = _make_png_bytes(200, 200)
        marks = [
            {"mark": 1, "box": (10, 10)},          # wrong arity → skipped
            {"mark": 2, "box": (50, 50, 20, 20)},  # good → drawn
        ]
        out = _draw_set_of_marks(png, marks, css_w=200, css_h=200)
        assert out is not png
        assert _open(out).size == (200, 200)

    def test_outline_thicker_at_downscale(self):
        """Marks are size-compensated: at scale<1 the outline (and label) are
        drawn thicker on the native image so they survive the downscale. Assert
        the scale=0.5 render lays down strictly more red than scale=1.0."""
        png = _make_png_bytes(200, 200)
        marks = [{"mark": 1, "box": (50, 50, 100, 60)}]

        def _red_count(scale: float) -> int:
            out = _draw_set_of_marks(
                png, marks, css_w=200, css_h=200, scale=scale, agent_id="a1",
            )
            assert out is not png
            img = _open(out)
            return sum(
                1
                for y in range(img.height)
                for x in range(img.width)
                if img.getpixel((x, y)) == (255, 0, 0)
            )

        assert _red_count(0.5) > _red_count(1.0)


def _fake_locator(box, visible=True):
    loc = AsyncMock()
    loc.is_visible = AsyncMock(return_value=visible)
    loc.bounding_box = AsyncMock(return_value=box)
    return loc


def _ref(role, name):
    return RefHandle.light_dom(
        page_id="p", scope_root=None, role=role, name=name,
        occurrence=0, disabled=False,
    )


def _make_mgr(agent_id, png_bytes, viewport):
    mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/profiles")
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=png_bytes)
    mock_page.viewport_size = viewport
    inst = CamoufoxInstance(agent_id, MagicMock(), MagicMock(), mock_page)
    mgr._instances[agent_id] = inst
    # Snapshot is a no-op here — refs are pre-populated by the caller.
    mgr._snapshot_impl = AsyncMock(return_value={"success": True})
    return mgr, inst


class TestScreenshotMarksManager:
    """``BrowserManager.screenshot_marks`` mark-building + coordinate output."""

    @pytest.mark.asyncio
    async def test_builds_marks_for_actionable_visible_refs(self):
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {
            "e0": _ref("button", "Play"),
            "e1": _ref("heading", "Title"),   # not actionable → skipped
            "e2": _ref("link", "Home"),
        }
        locators = {
            "e0": _fake_locator({"x": 100, "y": 200, "width": 50, "height": 20}),
            "e2": _fake_locator({"x": 300, "y": 400, "width": 80, "height": 30}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png", scale=1.0)

        assert result["success"] is True
        data = result["data"]
        assert data["marks_shown"] == 2
        assert data["marks_truncated"] is False
        assert data["marks_max"] == 50
        assert data["annotated"] is True
        assert data["viewport"] == {"width": 800, "height": 600}
        assert data["image"] == {"width": 800, "height": 600}
        assert data["click_scale"] == {"x": 1.0, "y": 1.0}
        # Mark 1 → e0 (button), centre = (100+25, 200+10).
        assert data["marks"][0] == {
            "mark": 1, "ref": "e0", "role": "button",
            "name": "Play", "x": 125.0, "y": 210.0,
        }
        assert data["marks"][1]["ref"] == "e2"
        assert data["marks"][1]["mark"] == 2
        # Internal CSS box tuple must be stripped from the wire shape.
        assert all("box" not in m for m in data["marks"])
        # Image is real base64 and byte count matches.
        assert len(base64.b64decode(data["image_base64"])) == data["bytes"]

    @pytest.mark.asyncio
    async def test_out_of_viewport_ref_skipped(self):
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {
            "e0": _ref("button", "In"),
            "e1": _ref("button", "Off"),
        }
        locators = {
            "e0": _fake_locator({"x": 10, "y": 10, "width": 40, "height": 20}),
            # x >= viewport width → fully off-screen.
            "e1": _fake_locator({"x": 900, "y": 10, "width": 40, "height": 20}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png")
        marks = result["data"]["marks"]
        assert [m["ref"] for m in marks] == ["e0"]

    @pytest.mark.asyncio
    async def test_edge_element_click_point_strictly_inside_viewport(self):
        """An element touching the right edge must yield x < vw (click_xy
        rejects x>=vw). The click point is the centre of the visible
        intersection, clamped strictly inside — NOT the raw centre clamped
        to the edge."""
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {"e0": _ref("button", "Edge")}
        # x=790,width=100 in an 800px viewport → raw centre 840 (off-screen).
        locators = {
            "e0": _fake_locator({"x": 790, "y": 10, "width": 100, "height": 20}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png")
        mark = result["data"]["marks"][0]
        assert mark["ref"] == "e0"
        # Strictly clickable: x < viewport width, and y in bounds.
        assert mark["x"] < 800.0
        assert 0.0 <= mark["y"] < 600.0
        # Centre of the visible slice [790..800] → 795.
        assert mark["x"] == 795.0

    @pytest.mark.asyncio
    async def test_snapshot_truncation_propagates(self):
        """When the a11y snapshot itself dropped elements past its cap,
        ``snapshot_truncated`` must be True and stay DISTINCT from
        ``marks_truncated``."""
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        mgr._snapshot_impl = AsyncMock(
            return_value={"success": True, "data": {"truncated": True}},
        )
        inst.refs = {"e0": _ref("button", "One")}
        locators = {
            "e0": _fake_locator({"x": 10, "y": 10, "width": 40, "height": 20}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png")
        data = result["data"]
        assert data["snapshot_truncated"] is True
        # marks_truncated tracks the max_marks cap only — untouched here.
        assert data["marks_truncated"] is False

    @pytest.mark.asyncio
    async def test_snapshot_not_truncated_by_default(self):
        """No upstream truncation → ``snapshot_truncated`` is False."""
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {"e0": _ref("button", "One")}
        locators = {
            "e0": _fake_locator({"x": 10, "y": 10, "width": 40, "height": 20}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png")
        assert result["data"]["snapshot_truncated"] is False

    @pytest.mark.asyncio
    async def test_max_marks_truncates(self):
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {
            "e0": _ref("button", "One"),
            "e1": _ref("button", "Two"),
        }
        locators = {
            "e0": _fake_locator({"x": 10, "y": 10, "width": 40, "height": 20}),
            "e1": _fake_locator({"x": 60, "y": 60, "width": 40, "height": 20}),
        }

        async def _resolve(_inst, ref):
            return locators.get(ref)
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png", max_marks=1)
        data = result["data"]
        assert data["marks_shown"] == 1
        assert data["marks_truncated"] is True
        assert data["marks_max"] == 1

    @pytest.mark.asyncio
    async def test_stale_ref_does_not_abort_capture(self):
        """A ref whose resolution raises must be skipped, not fatal."""
        from src.browser.ref_handle import RefStale
        png = _make_png_bytes(800, 600)
        mgr, inst = _make_mgr("a1", png, {"width": 800, "height": 600})
        inst.refs = {
            "e0": _ref("button", "Bad"),
            "e1": _ref("button", "Good"),
        }
        good = _fake_locator({"x": 10, "y": 10, "width": 40, "height": 20})

        async def _resolve(_inst, ref):
            if ref == "e0":
                raise RefStale("tab closed", ref=ref)
            return good
        mgr._locator_from_ref = _resolve

        result = await mgr.screenshot_marks("a1", format="png")
        assert result["success"] is True
        assert [m["ref"] for m in result["data"]["marks"]] == ["e1"]

    @pytest.mark.asyncio
    async def test_unsupported_format_rejected(self):
        png = _make_png_bytes(100, 100)
        mgr, _inst = _make_mgr("a1", png, {"width": 100, "height": 100})
        result = await mgr.screenshot_marks("a1", format="gif")
        assert result["success"] is False
        assert "gif" in result["error"].lower()


class TestScreenshotMarksTool:
    """The ``browser_screenshot_marks`` agent-tool wrapper."""

    @pytest.mark.asyncio
    async def test_emits_action_and_extracts_image(self):
        from src.agent.builtins.browser_tool import browser_screenshot_marks

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "success": True,
            "data": {
                "image_base64": base64.b64encode(b"IMG").decode(),
                "format": "webp",
                "marks": [{"mark": 1, "ref": "e0", "x": 10.0, "y": 20.0}],
                "marks_shown": 1,
            },
        })

        result = await browser_screenshot_marks(
            format="png", quality=90, scale=0.8, max_marks=10, mesh_client=mc,
        )

        mc.browser_command.assert_awaited_once_with(
            "screenshot_marks",
            {"format": "png", "quality": 90, "scale": 0.8, "max_marks": 10},
        )
        # Image extracted into _image, popped from data before redaction.
        assert result["_image"] == {
            "data": base64.b64encode(b"IMG").decode(),
            "media_type": "image/webp",
        }
        assert "image_base64" not in result["data"]
        # The marks map survives redaction.
        assert result["data"]["marks"][0]["ref"] == "e0"

    @pytest.mark.asyncio
    async def test_defaults(self):
        from src.agent.builtins.browser_tool import browser_screenshot_marks

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"data": {}})
        await browser_screenshot_marks(mesh_client=mc)
        mc.browser_command.assert_awaited_once_with(
            "screenshot_marks",
            {"format": "webp", "quality": 75, "scale": 1.0, "max_marks": 50},
        )

    @pytest.mark.asyncio
    async def test_status_reflects_annotation_happened(self):
        """When the service reports ``annotated=True`` the status says so and
        includes the mark count."""
        from src.agent.builtins.browser_tool import browser_screenshot_marks

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "success": True,
            "data": {
                "image_base64": base64.b64encode(b"IMG").decode(),
                "format": "webp",
                "marks": [{"mark": 1, "ref": "e0", "x": 1.0, "y": 2.0}],
                "annotated": True,
            },
        })
        result = await browser_screenshot_marks(mesh_client=mc)
        assert "annotated screenshot captured" in result["status"]
        assert "1 marks" in result["status"]

    @pytest.mark.asyncio
    async def test_status_does_not_claim_annotated_when_draw_failed(self):
        """An image exists even when annotation didn't happen (Pillow missing /
        draw failed / no marks). Status must NOT claim 'annotated'."""
        from src.agent.builtins.browser_tool import browser_screenshot_marks

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "success": True,
            "data": {
                "image_base64": base64.b64encode(b"IMG").decode(),
                "format": "png",
                "marks": [{"mark": 1, "ref": "e0", "x": 1.0, "y": 2.0}],
                "annotated": False,
            },
        })
        result = await browser_screenshot_marks(mesh_client=mc)
        # An image is still attached.
        assert result["_image"]["media_type"] == "image/png"
        # But the status is honest about annotation being unavailable.
        assert "annotated screenshot captured" not in result["status"]
        assert "annotation unavailable" in result["status"]
        assert "1 marks mapped" in result["status"]

    @pytest.mark.asyncio
    async def test_no_mesh_client(self):
        from src.agent.builtins.browser_tool import browser_screenshot_marks

        result = await browser_screenshot_marks(mesh_client=None)
        assert "error" in result
        assert "mesh" in result["error"].lower()
