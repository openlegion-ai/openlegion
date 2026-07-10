"""Pure-CV gap detection for the DataDome image-slider puzzle (§11.5).

This module is deliberately dependency-thin and side-effect free: it takes
the raw PNG bytes of a puzzle background and returns the horizontal offset
of the notch (the "gap" the puzzle piece must slide into). It imports NO
Playwright and NO browser-service code, so it is unit-testable against
synthesized images.

**Best-effort by design.** DataDome binds solve *success* to behavioral
telemetry (the shape / timing / jitter of the drag), NOT just the final
offset. A geometrically correct offset can still be rejected. The trusted
X11 drag (``_x11_drag``) supplies the human-like motion; this module only
supplies the target. Treat a returned offset as a hint, and always keep the
human-escalation fallback intact.

**Scope.** Only the DataDome IMAGE-SLIDER (a solid puzzle piece sliding into
a notch in a background raster) is handled here. GeeTest / canvas-rendered
sliders and FunCaptcha rotate/tile challenges are OUT of scope: they render
the puzzle to a ``<canvas>`` whose pixels are only reachable via
``toDataURL()`` extraction, not a plain element screenshot.  # deferred
"""

from __future__ import annotations

import io

from src.shared.utils import setup_logging

logger = setup_logging("browser.slider_solver")


# Minimum normalized prominence for the detected peak to be trusted. Below
# this the "gap" is indistinguishable from image texture and we return None
# so the caller escalates rather than dragging to a guessed offset.
#
# KNOWN best-effort limitation (documented follow-up, NOT fixed here): the
# score is a single peak-vertical-edge measure over the WHOLE image with a
# fixed ``left_margin``. It can be fooled by any high-contrast vertical edge
# that isn't the puzzle notch, and the margin doesn't scale with DPR / piece
# width or restrict to a vertical band of interest (ROI). A real paired-edge
# / gap-shape / ROI detector is deferred; the confidence threshold + the
# default-off flag + the fail-safe re-probe in ``_solve_slider`` keep a
# false peak from ever producing a false "solved".
_MIN_CONFIDENCE = 0.35

# Hard pixel-count cap. The service can hand us a large element screenshot
# (broad ``img`` / ``canvas`` selectors), and the decode + O(w·h) scan run on
# a worker thread but still allocate memory and burn CPU. A real slider bg is
# ~300×200; a few-million-pixel ceiling rejects anything pathological while
# leaving a comfortable margin.
_MAX_IMAGE_PIXELS = 4_000_000


def compute_slider_offset(
    background_png: bytes,
    *,
    left_margin: int = 8,
) -> tuple[float, float] | None:
    """Locate the puzzle gap in a slider-captcha background image.

    Args:
        background_png: Raw PNG (or any Pillow-decodable) bytes of the
            puzzle background — typically an element screenshot of the
            DataDome puzzle canvas/image.
        left_margin: Number of leading columns to ignore. The puzzle piece
            rests against the left edge, so the strongest vertical edges
            there belong to the piece itself, not the gap.

    Returns:
        ``(gap_fraction, confidence)`` where ``gap_fraction`` is the gap
        column expressed as a fraction of the image width in ``[0.0, 1.0]``
        (``peak_x / (width - 1)`` — 0.0 = left edge, 1.0 = right edge) and
        ``confidence`` is a saturating ``0.0..1.0`` score, or ``None`` when
        no sufficiently prominent gap is found.

        The result is a WIDTH FRACTION, not a pixel column, on purpose:
        Playwright element screenshots capture at the page's
        ``deviceScaleFactor`` (so the screenshot may be 2× the element's CSS
        width under a HiDPI profile), while the caller drives the drag in
        CSS pixels from ``bounding_box()``. A fraction is device-pixel-ratio
        independent — the caller multiplies it by the element's CSS width to
        recover the correct CSS-pixel target regardless of DPR.

    Approach (kept intentionally simple + deterministic): grayscale the
    image, accumulate per-column horizontal-gradient energy
    (``sum_y |p(x, y) - p(x-1, y)|`` — the signature of a vertical edge),
    ignore the resting margin, take the peak column as the gap, normalize it
    to a width fraction, and score prominence as ``peak / mean`` folded
    through a saturating curve.

    Pillow-only: no numpy (not a guaranteed dependency). Never raises —
    returns ``None`` on any decode/processing failure so callers can treat
    "couldn't solve" and "no gap" identically (both escalate).
    """
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - Pillow is a hard dep
        logger.debug("compute_slider_offset: Pillow unavailable", exc_info=True)
        return None

    try:
        img = Image.open(io.BytesIO(background_png))
        # ``.size`` reads the header only — cheap, before we load pixels.
        width, height = img.size
    except Exception:
        logger.debug("compute_slider_offset: image decode failed", exc_info=True)
        return None

    # Need enough columns past the resting margin to have a search region.
    if width <= left_margin + 2 or height < 1:
        return None
    # Reject pathologically large images BEFORE the pixel load + O(w·h) scan.
    if width * height > _MAX_IMAGE_PIXELS:
        logger.debug(
            "compute_slider_offset: image %dx%d exceeds %d px cap; skipping",
            width, height, _MAX_IMAGE_PIXELS,
        )
        return None

    try:
        img = img.convert("L")
    except Exception:
        logger.debug("compute_slider_offset: grayscale convert failed", exc_info=True)
        return None

    try:
        # ``tobytes()`` returns a flat, row-major bytes buffer for mode "L"
        # (one int per pixel). Preferred over the now-deprecated
        # ``getdata()`` and available on the whole ``Pillow>=10.3`` range.
        pixels = img.tobytes()
    except Exception:
        logger.debug("compute_slider_offset: tobytes failed", exc_info=True)
        return None
    if len(pixels) < width * height:
        return None

    # Per-column horizontal-gradient energy. Column x accumulates the
    # absolute difference from column x-1 over every row; a vertical notch
    # edge lights up a single column strongly.
    col_energy = [0.0] * width
    for y in range(height):
        row_base = y * width
        prev = pixels[row_base]
        for x in range(1, width):
            cur = pixels[row_base + x]
            col_energy[x] += abs(cur - prev)
            prev = cur

    # The piece rests in the first ``left_margin`` columns — its own edges
    # would otherwise win. Search strictly to the right of it.
    search_start = max(1, left_margin)
    if search_start >= width:
        return None
    region = col_energy[search_start:]
    if not region:
        return None

    # Peak column (first max wins on ties — deterministic).
    peak_val = region[0]
    peak_idx = 0
    for i, val in enumerate(region):
        if val > peak_val:
            peak_val = val
            peak_idx = i
    peak_x = search_start + peak_idx

    mean = sum(region) / len(region)
    eps = 1e-6
    ratio = peak_val / (mean + eps)

    # Saturating normalization: ratio == 1 (peak == mean, i.e. flat/uniform
    # gradient) → confidence 0; a dominant peak → confidence → 1.
    if ratio <= 1.0:
        confidence = 0.0
    else:
        confidence = 1.0 - (1.0 / ratio)

    if confidence < _MIN_CONFIDENCE:
        return None
    # Normalize to a DPR-independent width fraction in [0.0, 1.0].
    gap_fraction = peak_x / max(1, width - 1)
    return gap_fraction, confidence
