"""Tests for the screenshot coordinate-mapping fields (``viewport``,
``image``, ``click_scale``) added to ``BrowserManager.screenshot``.

These fields let a caller convert an image pixel the agent SEES in a
``browser_screenshot`` result into a ``browser_click_xy`` viewport
CSS-pixel coordinate — needed whenever the captured/encoded image
dimensions diverge from the viewport (device-scale-factor fingerprints,
or ``scale<1.0`` Pillow downscaling).
"""

import atexit
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager, CamoufoxInstance

_PROFILES_ROOT = tempfile.mkdtemp(prefix="ol_test_screenshot_mapping_")
atexit.register(shutil.rmtree, _PROFILES_ROOT, ignore_errors=True)


def _make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    """Render a real PNG via Pillow of a known pixel size."""
    from io import BytesIO

    from PIL import Image
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _make_mgr_and_instance(agent_id: str, png_bytes: bytes, viewport):
    mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/test_profiles")
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=png_bytes)
    # ``viewport_size`` is a plain attribute (not awaited) on Playwright's
    # Page — set it directly rather than as an AsyncMock return value.
    mock_page.viewport_size = viewport
    inst = CamoufoxInstance(agent_id, MagicMock(), MagicMock(), mock_page)
    mgr._instances[agent_id] = inst
    return mgr


class TestScreenshotCoordinateMapping:
    """``BrowserManager.screenshot`` result includes viewport/image/click_scale."""

    @pytest.mark.asyncio
    async def test_scale_1_0_matches_viewport_click_scale_is_1(self):
        """PNG dims == viewport dims at scale=1.0 → click_scale is 1:1."""
        png = _make_png_bytes(width=800, height=600)
        mgr = _make_mgr_and_instance(
            "a1", png, {"width": 800, "height": 600},
        )

        result = await mgr.screenshot("a1", format="png", scale=1.0)

        assert result["success"] is True
        data = result["data"]
        assert data["viewport"] == {"width": 800, "height": 600}
        assert data["image"] == {"width": 800, "height": 600}
        assert data["click_scale"] == {"x": 1.0, "y": 1.0}

    @pytest.mark.asyncio
    async def test_scale_0_5_halves_image_click_scale_is_2(self):
        """Pillow downscale to half size → click_scale is 2.0 (image px * 2 = css px)."""
        png = _make_png_bytes(width=800, height=600)
        mgr = _make_mgr_and_instance(
            "a1", png, {"width": 800, "height": 600},
        )

        result = await mgr.screenshot("a1", format="webp", scale=0.5)

        assert result["success"] is True
        data = result["data"]
        assert data["viewport"] == {"width": 800, "height": 600}
        assert data["image"] == {"width": 400, "height": 300}
        assert data["click_scale"] == {"x": 2.0, "y": 2.0}

    @pytest.mark.asyncio
    async def test_viewport_none_yields_null_viewport_no_click_scale(self):
        """``viewport_size`` returning None → ``viewport`` is null and
        ``click_scale`` is omitted entirely (no divide-by-zero, no lie)."""
        png = _make_png_bytes(width=800, height=600)
        mgr = _make_mgr_and_instance("a1", png, None)

        result = await mgr.screenshot("a1", format="png", scale=1.0)

        assert result["success"] is True
        data = result["data"]
        assert data["viewport"] is None
        assert data["image"] == {"width": 800, "height": 600}
        assert "click_scale" not in data

    @pytest.mark.asyncio
    async def test_existing_keys_unchanged(self):
        """Additive only — image_base64/format/bytes are still present
        and correct alongside the new fields."""
        import base64

        png = _make_png_bytes(width=200, height=150)
        mgr = _make_mgr_and_instance(
            "a1", png, {"width": 200, "height": 150},
        )

        result = await mgr.screenshot("a1", format="png", scale=1.0)

        data = result["data"]
        assert set(data.keys()) >= {"image_base64", "format", "bytes", "viewport", "image", "click_scale"}
        decoded = base64.b64decode(data["image_base64"])
        assert len(decoded) == data["bytes"]
        assert decoded == png

    @pytest.mark.asyncio
    async def test_full_page_omits_click_scale(self):
        """``full_page=True`` captures the whole scrollable document, so the
        image height is the full page height (not the viewport height). Its
        y-mapping to the viewport-based click_xy space is invalid, so
        ``click_scale`` must be omitted entirely — but ``viewport`` and
        ``image`` dims stay present (they're informative)."""
        # Image far taller than the viewport → a full-page capture.
        png = _make_png_bytes(width=800, height=3200)
        mgr = _make_mgr_and_instance(
            "a1", png, {"width": 800, "height": 600},
        )

        result = await mgr.screenshot("a1", full_page=True, format="png", scale=1.0)

        assert result["success"] is True
        data = result["data"]
        assert data["viewport"] == {"width": 800, "height": 600}
        assert data["image"] == {"width": 800, "height": 3200}
        assert "click_scale" not in data

    @pytest.mark.asyncio
    async def test_asymmetric_ratio_click_scale_computed_per_axis(self):
        """x and y click_scale are computed independently, not coupled to
        a single scalar — a viewport/image mismatch with different x vs y
        ratios must produce different x vs y click_scale factors."""
        png = _make_png_bytes(width=400, height=200)
        mgr = _make_mgr_and_instance(
            "a1", png, {"width": 800, "height": 600},
        )

        result = await mgr.screenshot("a1", format="png", scale=1.0)

        data = result["data"]
        assert data["image"] == {"width": 400, "height": 200}
        assert data["click_scale"] == {"x": 2.0, "y": 3.0}
