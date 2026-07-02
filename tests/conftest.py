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
#
# Hard assignment (not ``setdefault``) — if a CI runner has pre-set
# the var to ``""`` or ``"0"`` the gate would fire and break the
# whole session. The test session unconditionally needs the bypass.
os.environ["OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE"] = "1"

# Keep litellm from autoloading the repo .env during collection.
# ``import litellm`` (pulled in transitively when test modules import)
# calls ``load_dotenv()`` unless ``LITELLM_MODE=PRODUCTION``. On a dev
# machine with a real .env (OAuth creds, wallet seed) that leaks the
# developer's real credentials into ``os.environ`` before any test
# runs — ``CredentialVault._load_credentials()`` then picks them up,
# making local runs diverge from CI (which has no .env) and letting
# real OAuth-only creds flip vault-behavior tests (e.g. the BYOK
# provider-validation gate).
#
# ``setdefault`` (not hard assignment) — a runner that deliberately
# wants litellm's dev-mode dotenv loading can still override. The e2e
# files call ``load_dotenv()`` themselves at module level, so explicit
# e2e runs with real keys keep working.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

# Redirect the durable fingerprint-state sidecar off the real ``data/``
# path for the whole session. The browser service now snapshots fingerprint
# burn + binding-signature state on every low-frequency mutation
# (``_record_fingerprint_outcome`` / ``_force_fingerprint_burn`` /
# ``_reset_fingerprint_window`` and the launch-time binding-coherence check),
# so any test that drives those would otherwise write ``data/fingerprint_state.json``
# in the repo root. A per-pid temp path keeps each xdist/shard worker isolated;
# ``setdefault`` so a test that needs a specific path can still override.
import tempfile as _tempfile  # noqa: E402

os.environ.setdefault(
    "FINGERPRINT_STATE_PATH",
    os.path.join(
        _tempfile.gettempdir(), f"ol_test_fingerprint_state_{os.getpid()}.json",
    ),
)

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


# Credential-bearing env prefixes (see src/host/credentials.py). Snapshotted
# and restored around every test by ``_isolate_credential_env`` below.
_CREDENTIAL_ENV_PREFIXES = (
    "OPENLEGION_SYSTEM_", "OPENLEGION_CRED_", "OPENLEGION_CONN_",
)


@pytest.fixture(autouse=True)
def _isolate_credential_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Isolate every test from the developer's real .env — reads AND writes.

    Tests must never touch the real PROJECT_ROOT/.env — it holds real
    credentials (OAuth tokens, wallet seed), and a crash mid-test
    corrupts it. Three seams:

    1. ``_default_env_file`` — ``_persist_to_env`` / ``_remove_from_env``
       default to PROJECT_ROOT/.env when called without an explicit
       ``env_file``; redirect to a per-test temp file. Tests that pass an
       explicit ``env_file`` are unaffected.
    2. ``ENV_FILE`` — the CLI group callback (``src/cli/main.py``) calls
       ``load_dotenv(cli_config.ENV_FILE)`` on EVERY CliRunner invocation
       that reaches a subcommand, loading the developer's real .env into
       ``os.environ`` mid-run and flipping later credential/template tests
       (the LITELLM_MODE guard above only covers litellm's import-time
       load). ``cli/config.py`` also WRITES through ``str(ENV_FILE)``.
       Patched on BOTH modules that bind it — ``src.cli.config`` and
       ``src.cli.runtime`` (``from``-import copies the binding).
    3. Belt-and-braces: snapshot the credential-prefixed ``os.environ``
       keys before the test and restore after, so any pollution path not
       covered above (e.g. ``_persist_to_env`` setting ``os.environ``)
       stays confined to the test that caused it. Ambient pre-session
       values (a developer's deliberate exports) are preserved.
    """
    from src.cli import config as cli_config
    from src.cli import runtime as cli_runtime
    from src.host import credentials as cred_mod

    monkeypatch.setattr(
        cred_mod, "_default_env_file", lambda: str(tmp_path / "test.env")
    )
    fake_env_file = tmp_path / "cli.env"
    monkeypatch.setattr(cli_config, "ENV_FILE", fake_env_file)
    monkeypatch.setattr(cli_runtime, "ENV_FILE", fake_env_file)

    saved = {
        k: v for k, v in os.environ.items()
        if k.startswith(_CREDENTIAL_ENV_PREFIXES)
    }
    yield
    for key in [
        k for k in os.environ if k.startswith(_CREDENTIAL_ENV_PREFIXES)
    ]:
        if key not in saved:
            os.environ.pop(key, None)
    os.environ.update(saved)


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
