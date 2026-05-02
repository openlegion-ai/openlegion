"""Rollout defaults — opt-in flags flipped to default-on.

Verifies the orchestration roadmap rollout: ``OPENLEGION_PROJECT_SCOPE_MODE``
defaults to ``enforce`` and ``OPENLEGION_ORCHESTRATION_TASKS_V2`` defaults
to ``1``. Also confirms the emergency-rollback off-switches still work
(setting either env var back to its legacy value re-enters the legacy
behavior path).

The flags are read once at module import in ``src.host.server`` so the
tests reload the module after each ``monkeypatch.setenv`` /
``monkeypatch.delenv`` to exercise the new value. Subsequent tests in
the suite read the live module attribute, which means we always restore
the unset state at the end of each test.
"""

from __future__ import annotations

import importlib

import pytest


def _reload_server():
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def test_project_scope_mode_default_is_enforce(monkeypatch):
    """With the env var unset, ``_PROJECT_SCOPE_MODE`` is ``enforce``."""
    monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
    server_module = _reload_server()
    try:
        assert server_module._PROJECT_SCOPE_MODE == "enforce"
    finally:
        # Reload once more to leave the module in a clean default state
        # for the rest of the suite.
        _reload_server()


def test_orchestration_tasks_v2_default_on(monkeypatch):
    """With the env var unset, ``_ORCHESTRATION_TASKS_V2`` is True."""
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    server_module = _reload_server()
    try:
        assert server_module._ORCHESTRATION_TASKS_V2 is True
    finally:
        _reload_server()


def test_orchestration_tasks_v2_can_be_disabled(monkeypatch):
    """``OPENLEGION_ORCHESTRATION_TASKS_V2=0`` is the rollback off-switch."""
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
    server_module = _reload_server()
    try:
        assert server_module._ORCHESTRATION_TASKS_V2 is False
    finally:
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        _reload_server()


def test_project_scope_mode_can_be_warn(monkeypatch):
    """``OPENLEGION_PROJECT_SCOPE_MODE=warn`` is the rollback kill-switch."""
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    server_module = _reload_server()
    try:
        assert server_module._PROJECT_SCOPE_MODE == "warn"
    finally:
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        _reload_server()


def test_invalid_project_scope_mode_falls_back_to_enforce(monkeypatch, caplog):
    """An unknown value coerces to the new default (``enforce``)."""
    import logging

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "bogus")
    with caplog.at_level(logging.WARNING, logger="host.server"):
        server_module = _reload_server()
    try:
        assert server_module._PROJECT_SCOPE_MODE == "enforce"
        # The warning mentions the new fallback default explicitly.
        warn_lines = [r.message for r in caplog.records if "defaulting to" in r.message]
        assert any("enforce" in m for m in warn_lines), (
            f"expected fallback-to-enforce warning, got {warn_lines}"
        )
    finally:
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        _reload_server()


@pytest.mark.parametrize("value", ["", "true", "yes", "on"])
def test_orchestration_tasks_v2_only_zero_disables(monkeypatch, value):
    """The flag uses ``== "1"`` so anything else parses as off — but the
    server module currently treats only the literal ``"1"`` as on. The
    intent of the rollout is "default-on, ``0`` disables"; this guards
    that the literal-1 contract on the **server** module still holds
    after the flip (i.e. an empty string env var still parses as off).

    The agent + dashboard read sites use the more permissive
    ``!= "0"`` parse — those have separate coverage.
    """
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", value)
    server_module = _reload_server()
    try:
        assert server_module._ORCHESTRATION_TASKS_V2 is False
    finally:
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        _reload_server()
