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

import random

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
    Letters: μ=0.045, σ=0.015, range 0.020–0.090.
    Symbols/digits: μ=0.065, σ=0.020, range 0.025–0.110.
    Spaces: μ=0.035, σ=0.010, range 0.018–0.070 (faster, word boundary rhythm).

    At speed=1.0, this is ≈270 CPM / 54 WPM — a moderate typist.
    """
    if char == " ":
        return _scaled(0.035, 0.010, 0.018, 0.070)
    if char.isalpha():
        return _scaled(0.045, 0.015, 0.020, 0.090)
    return _scaled(0.065, 0.020, 0.025, 0.110)


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


def scroll_increment() -> int:
    """Per-step scroll distance (pixels). μ=140, σ=30, range 80–200.

    Not scaled by speed factor — scroll speed is governed by scroll_pause timing.
    """
    return int(_clamped_gauss(140, 30, 80, 200))


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
