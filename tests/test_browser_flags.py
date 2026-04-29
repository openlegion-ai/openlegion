"""Tests for :mod:`src.browser.flags` — unified flag loader (Phase 1.6).

Verifies the precedence chain (per-agent → operator settings → env →
default) and the typed accessors (bool / int / float / str). Operator-
settings layer is exercised via a temporary JSON file + env override of
``OPENLEGION_SETTINGS_PATH``.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from src.browser import flags


@pytest.fixture(autouse=True)
def _isolate_overrides():
    """Ensure tests don't leak agent overrides or operator settings across each other."""
    saved_agents = dict(flags._agent_overrides)
    flags._agent_overrides.clear()
    flags.reload_operator_settings()
    yield
    flags._agent_overrides.clear()
    flags._agent_overrides.update(saved_agents)
    flags.reload_operator_settings()


@pytest.fixture
def settings_file(tmp_path):
    """Return a callable that writes a JSON settings file and points the env var at it."""

    def _write(body: dict):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(body))
        return str(path)

    return _write


# ── Precedence chain ────────────────────────────────────────────────────────


class TestPrecedence:
    def test_default_returned_when_nothing_set(self):
        assert flags.get_bool("BROWSER_DOWNLOADS_DISABLED", False) is False
        assert flags.get_str("CAPTCHA_SOLVER_PROVIDER", "") == ""
        assert flags.get_int("OPENLEGION_BROWSER_MAX_CONCURRENT", 5) == 5

    def test_env_wins_over_default(self):
        with mock.patch.dict(os.environ, {"BROWSER_DOWNLOADS_DISABLED": "true"}):
            assert flags.get_bool("BROWSER_DOWNLOADS_DISABLED", False) is True

    def test_operator_wins_over_env(self, settings_file):
        path = settings_file({"browser_flags": {"BROWSER_DOWNLOADS_DISABLED": "true"}})
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": path,
            "BROWSER_DOWNLOADS_DISABLED": "false",
        }):
            flags.reload_operator_settings()
            assert flags.get_bool("BROWSER_DOWNLOADS_DISABLED", False) is True

    def test_agent_wins_over_operator_and_env(self, settings_file):
        path = settings_file({"browser_flags": {"BROWSER_DOWNLOADS_DISABLED": "true"}})
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": path,
            "BROWSER_DOWNLOADS_DISABLED": "true",
        }):
            flags.reload_operator_settings()
            flags.set_agent_override("agent-X", "BROWSER_DOWNLOADS_DISABLED", "false")
            assert flags.get_bool(
                "BROWSER_DOWNLOADS_DISABLED", default=True, agent_id="agent-X",
            ) is False
            # Other agents see operator-level True
            assert flags.get_bool(
                "BROWSER_DOWNLOADS_DISABLED", default=False, agent_id="agent-Y",
            ) is True

    def test_set_agent_override_none_clears(self):
        flags.set_agent_override("a", "BROWSER_CANARY_ENABLED", "true")
        assert flags.get_bool(
            "BROWSER_CANARY_ENABLED", default=False, agent_id="a",
        ) is True
        flags.set_agent_override("a", "BROWSER_CANARY_ENABLED", None)
        assert flags.get_bool(
            "BROWSER_CANARY_ENABLED", default=False, agent_id="a",
        ) is False


# ── Typed accessors ────────────────────────────────────────────────────────


class TestBoolAccessor:
    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("FALSE", False), ("0", False), ("no", False),
        ("off", False), ("", False),
    ])
    def test_common_strings_coerce(self, raw, expected):
        with mock.patch.dict(os.environ, {"BROWSER_CANARY_ENABLED": raw}):
            assert flags.get_bool("BROWSER_CANARY_ENABLED", not expected) is expected

    def test_garbage_falls_back_to_default(self, caplog):
        with mock.patch.dict(os.environ, {"BROWSER_CANARY_ENABLED": "maybe"}):
            assert flags.get_bool("BROWSER_CANARY_ENABLED", True) is True
        # Warning emitted so operators know their env var is malformed.
        assert any("non-boolean" in r.message for r in caplog.records)

    def test_bad_agent_override_falls_through_to_operator(
        self, settings_file, caplog,
    ):
        path = settings_file({"browser_flags": {"BROWSER_CANARY_ENABLED": "true"}})
        with mock.patch.dict(os.environ, {"OPENLEGION_SETTINGS_PATH": path}):
            flags.reload_operator_settings()
            flags.set_agent_override("a1", "BROWSER_CANARY_ENABLED", "maybe")

            assert flags.get_bool(
                "BROWSER_CANARY_ENABLED", False, agent_id="a1",
            ) is True
        assert any("non-boolean" in r.message for r in caplog.records)

    def test_bad_operator_value_falls_through_to_env(
        self, settings_file, caplog,
    ):
        path = settings_file({"browser_flags": {"BROWSER_CANARY_ENABLED": "maybe"}})
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": path,
            "BROWSER_CANARY_ENABLED": "true",
        }):
            flags.reload_operator_settings()

            assert flags.get_bool("BROWSER_CANARY_ENABLED", False) is True
        assert any("non-boolean" in r.message for r in caplog.records)


class TestIntAccessor:
    def test_valid_int(self):
        with mock.patch.dict(os.environ, {"OPENLEGION_BROWSER_MAX_CONCURRENT": "12"}):
            assert flags.get_int("OPENLEGION_BROWSER_MAX_CONCURRENT", 5) == 12

    def test_bounds_clamp(self):
        with mock.patch.dict(os.environ, {"FLAG": "999"}):
            assert flags.get_int("FLAG", 5, max_value=100) == 100
        with mock.patch.dict(os.environ, {"FLAG": "-10"}):
            assert flags.get_int("FLAG", 5, min_value=0) == 0

    def test_garbage_falls_back_to_default(self, caplog):
        with mock.patch.dict(os.environ, {"FLAG": "twelve"}):
            assert flags.get_int("FLAG", 5) == 5
        assert any("non-integer" in r.message for r in caplog.records)

    def test_bad_operator_int_falls_through_to_env(
        self, settings_file, caplog,
    ):
        path = settings_file({"browser_flags": {"FLAG": "twelve"}})
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": path,
            "FLAG": "12",
        }):
            flags.reload_operator_settings()

            assert flags.get_int("FLAG", 5) == 12
        assert any("non-integer" in r.message for r in caplog.records)


class TestFloatAccessor:
    def test_valid_float(self):
        with mock.patch.dict(os.environ, {"FLAG": "0.8"}):
            assert flags.get_float("FLAG", 0.5) == 0.8

    def test_bounds(self):
        with mock.patch.dict(os.environ, {"FLAG": "2.5"}):
            assert flags.get_float("FLAG", 0.5, min_value=0.1, max_value=0.9) == 0.9

    def test_bad_agent_float_falls_through_to_env(self, caplog):
        with mock.patch.dict(os.environ, {"FLAG": "0.75"}):
            flags.set_agent_override("a1", "FLAG", "high")

            assert flags.get_float("FLAG", 0.5, agent_id="a1") == 0.75
        assert any("non-float" in r.message for r in caplog.records)


class TestStrAccessor:
    def test_default_when_absent(self):
        assert flags.get_str("UNSET_FLAG", "fallback") == "fallback"

    def test_agent_override_string(self):
        flags.set_agent_override("a", "FLAG", "value-for-a")
        assert flags.get_str("FLAG", "def", agent_id="a") == "value-for-a"


# ── Operator settings file handling ────────────────────────────────────────


class TestOperatorSettings:
    def test_missing_file_uses_env_only(self):
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": "/tmp/does-not-exist.json",
            "FLAG": "hello",
        }):
            flags.reload_operator_settings()
            assert flags.get_str("FLAG") == "hello"

    def test_malformed_json_does_not_crash(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json")
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": str(bad),
            "FLAG": "fallthrough",
        }):
            flags.reload_operator_settings()
            assert flags.get_str("FLAG") == "fallthrough"

    def test_non_dict_browser_flags_ignored(self, tmp_path):
        weird = tmp_path / "w.json"
        weird.write_text(json.dumps({"browser_flags": [1, 2, 3]}))
        with mock.patch.dict(os.environ, {
            "OPENLEGION_SETTINGS_PATH": str(weird),
            "FLAG": "from-env",
        }):
            flags.reload_operator_settings()
            assert flags.get_str("FLAG") == "from-env"

    def test_reload_picks_up_changes(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"browser_flags": {"FLAG": "v1"}}))
        with mock.patch.dict(os.environ, {"OPENLEGION_SETTINGS_PATH": str(path)}):
            flags.reload_operator_settings()
            assert flags.get_str("FLAG") == "v1"
            path.write_text(json.dumps({"browser_flags": {"FLAG": "v2"}}))
            # Not reloaded yet — still cached
            assert flags.get_str("FLAG") == "v1"
            flags.reload_operator_settings()
            assert flags.get_str("FLAG") == "v2"


class TestSnapshotAll:
    def test_returns_known_flags(self):
        result = flags.snapshot_all()
        for name in flags.KNOWN_FLAGS:
            assert name in result

    def test_agent_override_appears_in_snapshot(self):
        flags.set_agent_override("a", "BROWSER_DOWNLOADS_DISABLED", "true")
        snap = flags.snapshot_all(agent_id="a")
        assert snap["BROWSER_DOWNLOADS_DISABLED"] == "true"
