"""Human-like timing helpers for browser interactions.

Uses clamped Gaussian distributions to produce natural variance in
action delays, keystroke timing, and scroll behavior. No third-party
dependencies — built on stdlib ``random``.

A global *speed* value scales all time-based values (higher = faster):
  - 4.0  = lightning (minimal delays, for testing / impatient tasks)
  - 1.0  = normal (default human-like timing)
  - 0.25 = stealth (maximum human emulation, slower and more cautious)

Pixel-based values (scroll_increment) are unaffected — scroll speed is
already governed by scroll_pause timing.
"""

from __future__ import annotations

import asyncio
import math
import random

from src.browser import flags

# ── Speed ─────────────────────────────────────────────────────

_speed: float = 1.0
_SPEED_MIN: float = 0.25
_SPEED_MAX: float = 4.0


def get_speed() -> float:
    """Return the current speed (0.25–4.0, default 1.0). Higher = faster."""
    return _speed


def set_speed(speed: float) -> None:
    """Set the global speed, clamped to [0.25, 4.0]. Higher = faster."""
    global _speed
    _speed = max(_SPEED_MIN, min(_SPEED_MAX, float(speed)))


# ── Inter-action delay ───────────────────────────────────────

_delay: float = 0.0
_DELAY_MIN: float = 0.0
_DELAY_MAX: float = 10.0


def get_delay() -> float:
    """Return the current inter-action delay mean (0.0–30.0 seconds, default 0.0)."""
    return _delay


def set_delay(delay: float) -> None:
    """Set the inter-action delay mean, clamped to [0.0, 10.0]. 0 = disabled."""
    global _delay
    _delay = max(_DELAY_MIN, min(_DELAY_MAX, float(delay)))


# ── Internal ──────────────────────────────────────────────────


def _clamped_gauss(mean: float, stddev: float, low: float, high: float) -> float:
    """Sample from a Gaussian distribution, clamped to [low, high]."""
    return max(low, min(high, random.gauss(mean, stddev)))


def _scaled(mean: float, stddev: float, low: float, high: float) -> float:
    """Sample a time value inversely scaled by speed (higher speed → shorter delays)."""
    f = 1.0 / _speed  # Convert speed to delay factor
    return _clamped_gauss(mean * f, stddev * f, low * f, high * f)


# ── Action timing ──────────────────────────────────────────────


def action_delay() -> float:
    """Post-click pause (seconds). Base: μ=0.18, σ=0.05, range 0.08–0.30."""
    return _scaled(0.18, 0.05, 0.08, 0.30)


def navigation_jitter() -> float:
    """Extra jitter added on top of wait_ms after navigation (seconds).

    Base: μ=0.08, σ=0.04, range 0.0–0.20.
    """
    return _scaled(0.08, 0.04, 0.0, 0.20)


def keystroke_delay(char: str) -> float:
    """Per-key delay (seconds). Symbols/digits are slower than letters.

    Base values (scaled by speed factor):
    Letters: μ=0.090, σ=0.030, range 0.040–0.180.
    Symbols/digits: μ=0.130, σ=0.040, range 0.050–0.220.
    Spaces: μ=0.070, σ=0.020, range 0.036–0.140 (faster, word boundary rhythm).

    At speed=1.0, this is ≈135 CPM / 27 WPM — a moderate-to-slow typist
    that better matches the real-user population (StatCounter / Typing.com
    benchmarks put the modal user at 30–40 WPM). Pre-2026-04 values
    were halved (μ=0.045 → 54 WPM); operators reported the typing-vs-
    mouse speed mismatch was an obvious behavioral signal — keystrokes
    fired in ~500ms while a click+settle takes ~300ms, so an "agent
    types 11 chars in the time of one mouse click" pattern was visible
    in VNC playback. Halving brings the ratio into a believable range.
    """
    if char == " ":
        return _scaled(0.070, 0.020, 0.036, 0.140)
    if char.isalpha():
        return _scaled(0.090, 0.030, 0.040, 0.180)
    return _scaled(0.130, 0.040, 0.050, 0.220)


def think_pause() -> float:
    """Mid-typing hesitation (seconds).

    Simulates the natural pauses humans make while composing text —
    e.g. after finishing a clause or before a difficult word.
    Apply stochastically (≈1.5 % mid-word, ≈8 % at word boundaries).

    Base: μ=0.40, σ=0.15, range 0.20–0.90.
    """
    return _scaled(0.40, 0.15, 0.20, 0.90)


# ── Scroll timing ─────────────────────────────────────────────


def scroll_pause() -> float:
    """Pause between scroll increments (seconds). Base: μ=0.08, σ=0.03, range 0.03–0.15."""
    return _scaled(0.08, 0.03, 0.03, 0.15)


def x11_step_delay() -> float:
    """Inter-step delay for X11 mouse trajectory (seconds).

    Base: μ=0.008, σ=0.003, range 0.003–0.014.
    Scaled by speed — at 3x this is ~1-5ms per step.
    """
    return _scaled(0.008, 0.003, 0.003, 0.014)


def x11_settle_delay() -> float:
    """Brief settle before/after X11 actions (seconds).

    Base: μ=0.035, σ=0.012, range 0.015–0.060.
    Scaled by speed — at 3x this is ~5-20ms.
    """
    return _scaled(0.035, 0.012, 0.015, 0.060)


def click_dwell() -> float:
    """Mousedown-to-mouseup hold time (seconds).

    Base: μ=0.085, σ=0.025, range 0.045–0.140.
    Scaled by speed — at 3x this is ~15-47ms.
    """
    return _scaled(0.085, 0.025, 0.045, 0.140)


def pre_click_settle() -> float:
    """Pause after reaching click target, before pressing (seconds).

    Models the human hesitation between landing the cursor on a target
    and confirming it visually before clicking.  Longer than
    x11_settle_delay which governs motor-control timing during
    the movement trajectory.

    Base: μ=0.15, σ=0.04, range 0.08-0.28.
    """
    return _scaled(0.15, 0.04, 0.08, 0.28)


def scroll_increment() -> int:
    """Per-step scroll distance (pixels). μ=140, σ=30, range 80–200.

    Not scaled by speed factor — scroll speed is governed by scroll_pause timing.
    """
    return int(_clamped_gauss(140, 30, 80, 200))


def scroll_ramp(progress: float) -> float:
    """Scroll step multiplier for momentum effect (0.15–1.0).

    *progress* is the scroll progress from 0.0 (start) to 1.0 (end).
    Returns a multiplier that scales scroll step size to model
    wheel inertia: smaller steps at the start and end of a scroll,
    full-size steps in the middle.  Uses a sine curve for smooth ramp.
    """
    return max(0.15, math.sin(max(0.0, min(1.0, progress)) * math.pi))


# ── Inter-action delay sampling ──────────────────────────────


def inter_action_delay() -> float:
    """Sample an inter-action delay (seconds).

    Applied after stateful browser actions (click, type, navigate, etc.)
    to simulate the natural human pause between actions — reading the page,
    deciding what to do next, moving eyes to the target element.

    Returns 0.0 when disabled.  Otherwise samples from a Gaussian centred
    on the configured mean with 40 % relative stddev, clamped to
    [mean * 0.3, mean * 2.0].  This produces natural variance: most pauses
    cluster near the mean, with occasional short bursts and long dwells.

    Not scaled by the speed setting — speed governs *intra*-action timing
    (keystroke pace, click dwell) while delay governs *inter*-action pacing.
    They are independent knobs.
    """
    if _delay <= 0:
        return 0.0
    stddev = _delay * 0.4
    low = _delay * 0.3
    high = _delay * 2.0
    return _clamped_gauss(_delay, stddev, low, high)


# ── CAPTCHA solve pacing (§11.11) ─────────────────────────────


_CAPTCHA_PACING_DEFAULTS = {
    "mu_ms": 6000,
    "sigma_ms": 2500,
    "min_ms": 3000,
    "max_ms": 12000,
}
# Hard ceiling on any individual captcha-pacing knob — 10 minutes is well
# beyond any sane operator override and prevents a misconfigured env var
# from stalling a solve indefinitely.
_CAPTCHA_PACING_HARD_MAX_MS = 600_000


async def captcha_solve_delay() -> None:
    """Gaussian-with-clamp delay between solver token retrieval and DOM injection.

    Real users take 5-15s between captcha appearing and form submit.
    Instant token injection is a low-but-real anti-bot signal. μ=6000ms,
    σ=2500ms, clamped to [3000, 12000]. Operator override via env vars:

    * ``CAPTCHA_SOLVE_PACING_MU_MS`` — Gaussian mean (default 6000).
    * ``CAPTCHA_SOLVE_PACING_SIGMA_MS`` — Gaussian stddev (default 2500).
    * ``CAPTCHA_PACING_MS_MIN`` — clamp lower bound (default 3000).
    * ``CAPTCHA_PACING_MS_MAX`` — clamp upper bound (default 12000).

    Each env var is bounded to ``[0, 600_000]`` (10 minutes) at the
    flag-loader layer to prevent a typo from producing a multi-hour
    sleep. We additionally validate ``min_ms < max_ms`` — an inverted
    pair (operator typo, e.g. ``MIN=20000, MAX=10000``) would silently
    produce a degenerate clamp where every sample lands at the lower
    bound; we log a warning and fall back to defaults instead.

    The existing ``human_delay`` / ``keystroke_delay`` helpers operate on
    millisecond-scale clamps and are unsuitable for the 3-12s scale of
    post-solve pacing — this helper exists specifically for that range.
    Not scaled by the global ``_speed`` factor: the goal is to mimic real
    human reading time, which is independent of the operator's "go fast"
    knob for intra-action timing.
    """
    mu_ms = flags.get_int(
        "CAPTCHA_SOLVE_PACING_MU_MS",
        _CAPTCHA_PACING_DEFAULTS["mu_ms"],
        min_value=0, max_value=_CAPTCHA_PACING_HARD_MAX_MS,
    )
    sigma_ms = flags.get_int(
        "CAPTCHA_SOLVE_PACING_SIGMA_MS",
        _CAPTCHA_PACING_DEFAULTS["sigma_ms"],
        min_value=0, max_value=_CAPTCHA_PACING_HARD_MAX_MS,
    )
    min_ms = flags.get_int(
        "CAPTCHA_PACING_MS_MIN",
        _CAPTCHA_PACING_DEFAULTS["min_ms"],
        min_value=0, max_value=_CAPTCHA_PACING_HARD_MAX_MS,
    )
    max_ms = flags.get_int(
        "CAPTCHA_PACING_MS_MAX",
        _CAPTCHA_PACING_DEFAULTS["max_ms"],
        min_value=0, max_value=_CAPTCHA_PACING_HARD_MAX_MS,
    )
    if min_ms >= max_ms:
        # Inverted / equal clamp — operator misconfig. Log once-shaped
        # warning (no rate gate; this fires per-call but only when
        # misconfigured) and fall back to the documented defaults so
        # the solve still paces realistically rather than landing on
        # a degenerate bound.
        import logging
        logging.getLogger("browser.timing").warning(
            "CAPTCHA pacing clamp is inverted (MIN=%d, MAX=%d); "
            "falling back to defaults [%d, %d].",
            min_ms, max_ms,
            _CAPTCHA_PACING_DEFAULTS["min_ms"],
            _CAPTCHA_PACING_DEFAULTS["max_ms"],
        )
        mu_ms = _CAPTCHA_PACING_DEFAULTS["mu_ms"]
        sigma_ms = _CAPTCHA_PACING_DEFAULTS["sigma_ms"]
        min_ms = _CAPTCHA_PACING_DEFAULTS["min_ms"]
        max_ms = _CAPTCHA_PACING_DEFAULTS["max_ms"]
    delay_ms = _clamped_gauss(float(mu_ms), float(sigma_ms),
                              float(min_ms), float(max_ms))
    await asyncio.sleep(delay_ms / 1000.0)
