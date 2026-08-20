"""Browser-service persistence defaults must live on the mounted volume.

``Dockerfile.browser`` sets ``WORKDIR /app`` and the only durable mount the
browser container gets is the ``openlegion_browser_data`` Docker volume at
``/data`` (``src/host/runtime.py`` ``_start_browser_container``). Any
persistence default expressed as a RELATIVE path therefore resolves to
``/app/<path>`` — the container's ephemeral write layer, which is discarded
whenever the engine recreates the container (runtime force-removes the stale
``openlegion_browser`` container on every start).

Three sidecars regressed on exactly that: session storage_state, fingerprint
burn/binding state, and the CAPTCHA spend ledger. Their writes succeeded
(``/app`` is chown'd to the browser user, so nothing errored) and the data
silently evaporated on restart — defeating the whole point of each module.

This file pins the invariant for all three: with no env override set, the
resolved default is absolute and lives under ``/data``, matching the
convention already used by ``service.BrowserManager.profiles_dir``
(``/data/profiles``), ``canary`` (``/data/canary``) and ``recorder``
(``/data/debug``). The env overrides remain the escape hatch for tests and
custom volume layouts, so those are pinned too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.browser import captcha_cost_counter as ccc
from src.browser import fingerprint_state as fp
from src.browser import session_persistence as sp

# (env var, zero-arg resolver) for every browser-service persistence sidecar.
_SIDECARS = [
    pytest.param("BROWSER_SESSION_DIR", sp._sessions_dir, id="sessions"),
    pytest.param("FINGERPRINT_STATE_PATH", fp._state_path, id="fingerprint"),
    pytest.param("CAPTCHA_COST_COUNTER_PATH", ccc._state_path, id="captcha_costs"),
]

# The browser container's durable mount point (Dockerfile.browser ``VOLUME
# /data``; runtime.py binds ``openlegion_browser_data`` here).
_VOLUME = Path("/data")


class TestDefaultsLandOnTheVolume:
    @pytest.mark.parametrize(("env_var", "resolver"), _SIDECARS)
    def test_default_is_under_data_volume(self, env_var, resolver, monkeypatch):
        monkeypatch.delenv(env_var, raising=False)
        resolved = resolver()
        assert resolved.is_absolute(), (
            f"{env_var} default {resolved} is relative — it would resolve "
            f"under the container's WORKDIR (/app), not the /data volume."
        )
        assert resolved == _VOLUME or _VOLUME in resolved.parents, (
            f"{env_var} default {resolved} is not on the {_VOLUME} volume — "
            f"it would not survive a container restart."
        )

    @pytest.mark.parametrize(("env_var", "resolver"), _SIDECARS)
    def test_env_override_still_wins(self, env_var, resolver, tmp_path, monkeypatch):
        target = tmp_path / "custom"
        monkeypatch.setenv(env_var, str(target))
        assert resolver() == target


class TestSessionPathUsesTheVolume:
    def test_per_agent_sidecar_path(self, monkeypatch):
        """The public path builder inherits the volume default."""
        monkeypatch.delenv("BROWSER_SESSION_DIR", raising=False)
        assert sp.session_path("agent-a") == Path("/data/sessions/agent-a.json")
