"""Human-like timing helpers for browser interactions.

Uses clamped Gaussian distributions to produce natural variance in
action delays, keystroke timing, and scroll behavior. No third-party
dependencies — built on stdlib ``random``.

A global *speed factor* scales all time-based values:
  - 0.25 = fast (minimal delays, for testing / impatient tasks)
  - 1.0  = normal (default human-like timing)
  - 3.0  = careful (maximum human emulation, slower and more cautious)

Pixel-based values (scroll_increment) are unaffected — scroll speed is
already governed by scroll_pause timing.
"""

from __future__ import annotations

import random

# ── Speed factor ──────────────────────────────────────────────

_speed_factor: float = 1.0
_SPEED_MIN: float = 0.25
_SPEED_MAX: float = 3.0


def get_speed_factor() -> float:
    """Return the current speed factor (0.25–3.0, default 1.0)."""
    return _speed_factor


def set_speed_factor(factor: float) -> None:
    """Set the global speed factor, clamped to [0.25, 3.0]."""
    global _speed_factor
    _speed_factor = max(_SPEED_MIN, min(_SPEED_MAX, float(factor)))


# ── Internal ──────────────────────────────────────────────────


def _clamped_gauss(mean: float, stddev: float, low: float, high: float) -> float:
    """Sample from a Gaussian distribution, clamped to [low, high]."""
    return max(low, min(high, random.gauss(mean, stddev)))


def _scaled(mean: float, stddev: float, low: float, high: float) -> float:
    """Sample a time value scaled by the current speed factor."""
    f = _speed_factor
    return _clamped_gauss(mean * f, stddev * f, low * f, high * f)


# ── Action timing ──────────────────────────────────────────────


def action_delay() -> float:
    """Post-click pause (seconds). Base: μ=0.30, σ=0.08, range 0.15–0.50."""
    return _scaled(0.30, 0.08, 0.15, 0.50)


def navigation_jitter() -> float:
    """Extra jitter added on top of wait_ms after navigation (seconds).

    Base: μ=0.20, σ=0.10, range 0.0–0.50.
    """
    f = _speed_factor
    return _clamped_gauss(0.20 * f, 0.10 * f, 0.0, 0.50 * f)


def keystroke_delay(char: str) -> float:
    """Per-key delay (seconds). Symbols/digits are slower than letters.

    Base values (scaled by speed factor):
    Letters: μ=0.08, σ=0.025, range 0.04–0.20.
    Symbols/digits: μ=0.11, σ=0.03, range 0.04–0.20.
    Spaces: μ=0.06, σ=0.02, range 0.03–0.12 (faster, word boundary rhythm).
    """
    if char == " ":
        return _scaled(0.06, 0.02, 0.03, 0.12)
    if char.isalpha():
        return _scaled(0.08, 0.025, 0.04, 0.20)
    return _scaled(0.11, 0.03, 0.04, 0.20)


def think_pause() -> float:
    """Mid-typing hesitation (seconds).

    Simulates the natural 0.3–1.5 s pauses humans make while composing text —
    e.g. after finishing a clause or before a difficult word.
    Apply stochastically (≈5 % of characters) rather than on every keystroke.

    Base: μ=0.65, σ=0.25, range 0.30–1.50.
    """
    return _scaled(0.65, 0.25, 0.30, 1.50)


# ── Scroll timing ─────────────────────────────────────────────


def scroll_pause() -> float:
    """Pause between scroll increments (seconds). Base: μ=0.08, σ=0.03, range 0.03–0.15."""
    return _scaled(0.08, 0.03, 0.03, 0.15)


def scroll_increment() -> int:
    """Per-step scroll distance (pixels). μ=140, σ=30, range 80–200.

    Not scaled by speed factor — scroll speed is governed by scroll_pause timing.
    """
    return int(_clamped_gauss(140, 30, 80, 200))
