"""Shared pytest configuration.

Zeroes the human-pacing helpers in ``src.browser.timing`` for tests that
don't care about real wall-clock behavior. Production code calls these on
every keystroke / scroll step / click to look human (clamped Gaussians,
μ≈80ms scroll pause × hundreds of actions per test). Logic-level tests
of the surrounding browser code pay that pacing incidentally and
dominate CI wall time — the slowest five test files account for ~43 %
of the serial run.

Tests that *do* assert on the helpers themselves (range checks, pacing
distributions) opt out via ``@pytest.mark.real_timing``. Apply at module
level with ``pytestmark = pytest.mark.real_timing`` for whole-file opt-out
(see ``tests/test_captcha_solve_pacing.py``).

Why patch bound names on the consumer modules instead of patching
``src.browser.timing`` directly: ``src/browser/service.py`` does
``from src.browser.timing import action_delay`` (and friends), so the
bound names live on the service module — patching the timing module
would have no effect on those existing await sites. Tests in
``TestHumanTiming`` re-import directly from ``src.browser.timing`` and
must keep seeing the real distributions; patching only the service
module leaves those tests untouched.

``captcha_solve_delay`` is the one exception: it's awaited via
``timing.captcha_solve_delay()`` (dotted access from
``src/browser/captcha.py``), so we patch it on the timing module and
rely on the ``real_timing`` opt-out for the dedicated pacing tests.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

# Bypass the mesh's fail-closed trust-tier startup gate for in-process
# test fixtures. The gate refuses to boot when
# ``OPENLEGION_TEAM_SCOPE_MODE=enforce`` (the default) and ``auth_tokens``
# is empty — the production "no tokens were ever configured" scenario.
# Tests legitimately construct ``create_mesh_app`` without tokens and
# drive identity via ``X-Agent-ID`` for their own assertions, which is
# exactly the forgery the production gate guards against. The dedicated
# env var (not an ambient ``"pytest" in sys.modules`` check) keeps the
# bypass explicit and out of reach of any production import-graph
# accident that drags pytest into a live mesh process.
os.environ.setdefault("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE", "1")

# Time-returning helpers consumed by ``src.browser.service`` via
# ``from src.browser.timing import X``. ``scroll_increment`` and
# ``scroll_ramp`` are intentionally excluded — they return pixel counts
# and ramp multipliers, not durations, and zeroing them would deadlock
# the scroll loop.
_SYNC_TIMING_FUNCS = (
    "action_delay",
    "navigation_jitter",
    "keystroke_delay",
    "think_pause",
    "scroll_pause",
    "x11_step_delay",
    "x11_settle_delay",
    "click_dwell",
    "pre_click_settle",
)


def _zero(*_args: Any, **_kwargs: Any) -> float:
    return 0.0


async def _async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_timing: opt out of the autouse fast-timing fixture and use "
        "the real human-pacing delays from src.browser.timing.",
    )


@pytest.fixture(autouse=True)
def _fast_browser_timing(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_timing"):
        return

    service = sys.modules.get("src.browser.service")
    if service is not None:
        for name in _SYNC_TIMING_FUNCS:
            if hasattr(service, name):
                monkeypatch.setattr(service, name, _zero, raising=False)

    timing = sys.modules.get("src.browser.timing")
    if timing is not None:
        monkeypatch.setattr(timing, "captcha_solve_delay", _async_noop, raising=False)
