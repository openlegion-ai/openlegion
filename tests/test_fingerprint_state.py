"""Tests for the durable fingerprint-state sidecar (burn + binding signatures).

Covers:
  * snapshot → restore round-trip preserves the rolling window (values AND
    the ``maxlen`` bound), the hard-burned set, and the hard-burn reasons.
  * binding-signature map round-trips.
  * a ``deque`` longer than ``maxlen`` restores keeping only the last N.
  * missing file restores an empty state (no raise).
  * corrupt JSON / wrong-shape / wrong-version restores empty (no raise).
  * the sidecar is written 0o600 (opaque state, but matches the codebase's
    atomic-write posture).
"""

from __future__ import annotations

import json
import os
import stat
from collections import deque

from src.browser import fingerprint_state as fp


def _snapshot(path, **overrides):
    """Snapshot with sensible empty defaults; overrides fill in fields."""
    kwargs = dict(
        window={},
        last_signal={},
        hard_burned=set(),
        hard_burn_reason={},
        binding_signatures={},
        path=path,
    )
    kwargs.update(overrides)
    return fp.snapshot(**kwargs)


class TestRoundTrip:
    def test_window_values_and_maxlen_survive(self, tmp_path):
        target = tmp_path / "fp.json"
        window = {
            "agent-a": deque([True, False, True], maxlen=10),
            "agent-b": deque([False], maxlen=10),
        }
        assert _snapshot(target, window=window) is True

        loaded = fp.restore(target, window_maxlen=10)
        assert list(loaded.window["agent-a"]) == [True, False, True]
        assert list(loaded.window["agent-b"]) == [False]
        # Rebuilt as a bounded deque with the same maxlen — appending an
        # 11th element evicts the oldest rather than growing unbounded.
        restored = loaded.window["agent-a"]
        assert isinstance(restored, deque)
        assert restored.maxlen == 10
        for _ in range(12):
            restored.append(True)
        assert len(restored) == 10

    def test_full_window_at_maxlen_keeps_last_n(self, tmp_path):
        target = tmp_path / "fp.json"
        # 10 entries = a full window: 6 rejects then 4 accepts.
        full = deque([True] * 6 + [False] * 4, maxlen=10)
        assert _snapshot(target, window={"agent-a": full}) is True

        loaded = fp.restore(target, window_maxlen=10)
        assert list(loaded.window["agent-a"]) == [True] * 6 + [False] * 4
        assert loaded.window["agent-a"].maxlen == 10

    def test_oversized_list_truncates_to_maxlen_last_n(self, tmp_path):
        # A hand-tampered / legacy file could hold more entries than maxlen;
        # deque(iterable, maxlen=N) keeps the LAST N, so restore is safe.
        target = tmp_path / "fp.json"
        target.write_text(
            json.dumps(
                {
                    "version": 1,
                    "saved_at": 0,
                    "window": {"agent-a": [True, False, True, True, False]},
                    "last_signal": {},
                    "hard_burned": [],
                    "hard_burn_reason": {},
                    "binding_signatures": {},
                }
            )
        )
        loaded = fp.restore(target, window_maxlen=3)
        assert list(loaded.window["agent-a"]) == [True, True, False]
        assert loaded.window["agent-a"].maxlen == 3

    def test_hard_burned_and_reasons_survive(self, tmp_path):
        target = tmp_path / "fp.json"
        ok = _snapshot(
            target,
            hard_burned={"agent-a", "agent-b"},
            hard_burn_reason={
                "agent-a": ("cloudflare", "cf-mitigated=block"),
                "agent-b": ("perimeterx", "x-px-block-type=1"),
            },
            last_signal={"agent-a": 1234.5},
        )
        assert ok is True

        loaded = fp.restore(target)
        assert loaded.hard_burned == {"agent-a", "agent-b"}
        assert loaded.hard_burn_reason["agent-a"] == ("cloudflare", "cf-mitigated=block")
        assert loaded.hard_burn_reason["agent-b"] == ("perimeterx", "x-px-block-type=1")
        # Reasons restore as a 2-tuple (list-in-JSON → tuple in memory).
        assert isinstance(loaded.hard_burn_reason["agent-a"], tuple)
        assert loaded.last_signal["agent-a"] == 1234.5

    def test_binding_signatures_round_trip(self, tmp_path):
        target = tmp_path / "fp.json"
        sigs = {"agent-a": "0011223344556677", "agent-b": "aabbccddeeff0011"}
        assert _snapshot(target, binding_signatures=sigs) is True

        loaded = fp.restore(target)
        assert loaded.binding_signatures == sigs

    def test_combined_round_trip(self, tmp_path):
        target = tmp_path / "fp.json"
        ok = _snapshot(
            target,
            window={"agent-a": deque([True, True], maxlen=10)},
            last_signal={"agent-a": 42.0},
            hard_burned={"agent-a"},
            hard_burn_reason={"agent-a": ("datadome", "x-datadome=protected")},
            binding_signatures={"agent-a": "cafebabecafebabe"},
        )
        assert ok is True

        loaded = fp.restore(target, window_maxlen=10)
        assert list(loaded.window["agent-a"]) == [True, True]
        assert loaded.last_signal["agent-a"] == 42.0
        assert loaded.hard_burned == {"agent-a"}
        assert loaded.hard_burn_reason["agent-a"] == ("datadome", "x-datadome=protected")
        assert loaded.binding_signatures["agent-a"] == "cafebabecafebabe"


class TestNonFatalRestore:
    def test_missing_file_restores_empty(self, tmp_path):
        loaded = fp.restore(tmp_path / "does-not-exist.json")
        assert loaded.window == {}
        assert loaded.last_signal == {}
        assert loaded.hard_burned == set()
        assert loaded.hard_burn_reason == {}
        assert loaded.binding_signatures == {}

    def test_corrupt_json_restores_empty(self, tmp_path):
        target = tmp_path / "fp.json"
        target.write_text("{not valid json at all")
        loaded = fp.restore(target)
        assert loaded.hard_burned == set()
        assert loaded.binding_signatures == {}

    def test_wrong_shape_restores_empty(self, tmp_path):
        target = tmp_path / "fp.json"
        target.write_text(json.dumps(["not", "a", "dict"]))
        loaded = fp.restore(target)
        assert loaded.window == {}

    def test_wrong_version_restores_empty(self, tmp_path):
        target = tmp_path / "fp.json"
        target.write_text(
            json.dumps(
                {
                    "version": 99,
                    "hard_burned": ["agent-a"],
                    "binding_signatures": {"agent-a": "sig"},
                }
            )
        )
        loaded = fp.restore(target)
        assert loaded.hard_burned == set()
        assert loaded.binding_signatures == {}

    def test_malformed_entries_skipped_not_fatal(self, tmp_path):
        # Individual bad rows are dropped; the well-formed ones survive.
        target = tmp_path / "fp.json"
        target.write_text(
            json.dumps(
                {
                    "version": 1,
                    "window": {"agent-a": [True], "bad": "not-a-list"},
                    "last_signal": {"agent-a": 1.0, "bad": "nan-string"},
                    "hard_burned": ["agent-a", 123],
                    "hard_burn_reason": {"agent-a": ["cf", "block"], "bad": ["only-one"]},
                    "binding_signatures": {"agent-a": "sig", "bad": 12345},
                }
            )
        )
        loaded = fp.restore(target, window_maxlen=10)
        assert list(loaded.window["agent-a"]) == [True]
        assert "bad" not in loaded.window
        assert loaded.last_signal == {"agent-a": 1.0}
        assert loaded.hard_burned == {"agent-a"}
        assert loaded.hard_burn_reason == {"agent-a": ("cf", "block")}
        assert loaded.binding_signatures == {"agent-a": "sig"}


class TestFilePermissions:
    def test_sidecar_is_owner_only(self, tmp_path):
        target = tmp_path / "fp.json"
        assert _snapshot(target, hard_burned={"agent-a"}) is True
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600


class TestEnvOverride:
    def test_default_path_env_override(self, tmp_path, monkeypatch):
        target = tmp_path / "custom_fp.json"
        monkeypatch.setenv("FINGERPRINT_STATE_PATH", str(target))
        # No explicit path → falls back to the env-configured location.
        assert (
            fp.snapshot(
                window={},
                last_signal={},
                hard_burned={"agent-z"},
                hard_burn_reason={},
                binding_signatures={},
            )
            is True
        )
        assert target.exists()
        loaded = fp.restore()
        assert loaded.hard_burned == {"agent-z"}
