"""Tests for the offline behavior analyzer (Phase 6 §9.5)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tools import behavior_analyze as ba

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_recording(
    path: Path,
    *,
    agent: str = "agent-test",
    events: list[dict] | None = None,
    omit_header: bool = False,
) -> Path:
    """Write a JSONL recording matching the recorder's on-disk format."""
    events = events or []
    lines = []
    if not omit_header:
        lines.append(json.dumps({
            "schema": "openlegion.browser.recorder/v1",
            "agent": agent,
            "reason": "test",
            "event_count": len(events),
        }))
    for ev in events:
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def _make_event(
    ts: float,
    interval_s: float | None,
    type_: str,
    **fields,
) -> dict:
    return {"ts": ts, "interval_s": interval_s, "type": type_, **fields}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_header_only_file_reports_no_events(tmp_path):
    """Test 1: header-only file → no crash, reported as zero events."""
    p = _write_recording(tmp_path / "rec.jsonl", events=[])
    fr = ba.load_file(p)
    assert not fr.skipped
    assert fr.events == []
    metrics = ba.compute_metrics(fr.events)
    assert metrics["event_count"] == 0
    assert metrics["interval_summary_seconds"]["count"] == 0
    assert metrics["cadence_entropy"]["sample_count"] == 0


def test_three_files_aggregated_across_agents(tmp_path):
    """Test 2: three valid files → metrics aggregated."""
    files = []
    for i, agent in enumerate(("a", "b", "c")):
        events = [
            _make_event(1000.0 + i * 100, None, "navigate", host="x"),
            _make_event(1001.0 + i * 100, 1.0, "click", method="x11", success=True),
            _make_event(1002.5 + i * 100, 1.5, "keystrokes", char_count=5,
                        fast=False, method="x11"),
        ]
        files.append(_write_recording(
            tmp_path / f"rec-{agent}.jsonl",
            agent=agent,
            events=events,
        ))
    results = [ba.load_file(p) for p in files]
    report = ba.aggregate(results)
    assert report["files_loaded"] == 3
    assert report["files_skipped"] == 0
    assert sorted(report["agents"]) == ["a", "b", "c"]
    # Global: 9 events; intervals: per file (None,1.0,1.5) -> 6 valid intervals.
    assert report["global"]["event_count"] == 9
    assert report["global"]["interval_summary_seconds"]["count"] == 6


def test_filter_agent_restricts_events(tmp_path):
    """Test 3: --filter-agent narrows to one agent's events."""
    f1 = _write_recording(tmp_path / "a.jsonl", agent="alice", events=[
        _make_event(1.0, None, "click", method="x11", success=True),
        _make_event(2.0, 1.0, "click", method="x11", success=True),
    ])
    f2 = _write_recording(tmp_path / "b.jsonl", agent="bob", events=[
        _make_event(1.0, None, "click", method="cdp", success=True),
    ])
    results = [ba.load_file(f1), ba.load_file(f2)]
    report = ba.aggregate(results, filter_agent="alice")
    assert report["agents"] == ["alice"]
    assert report["global"]["event_count"] == 2


def test_since_drops_older_events(tmp_path):
    """Test 4: --since drops events older than threshold."""
    p = _write_recording(tmp_path / "r.jsonl", events=[
        _make_event(100.0, None, "click", method="x11", success=True),
        _make_event(200.0, 100.0, "click", method="x11", success=True),
        _make_event(300.0, 100.0, "click", method="x11", success=True),
    ])
    fr = ba.load_file(p)
    report = ba.aggregate([fr], since=250.0)
    # Only the third event survives.
    assert report["global"]["event_count"] == 1


def test_cadence_entropy_metronomic_is_low():
    """Test 5a: identical intervals → entropy near zero (one bin)."""
    intervals = [0.5] * 50
    out = ba._shannon_entropy_bits(intervals)
    assert out["entropy_bits"] == pytest.approx(0.0, abs=1e-9)
    # All samples in one bin.
    assert max(out["bin_counts"]) == 50


def test_cadence_entropy_uniform_is_high():
    """Test 5b: roughly uniform across bins → entropy near log2(N_BINS)."""
    # Pick one value per bin's center so each bin gets equal mass.
    log_min = math.log(0.01)
    log_max = math.log(60.0)
    n = 10
    centers = []
    for i in range(n):
        frac = (i + 0.5) / n
        centers.append(math.exp(log_min + frac * (log_max - log_min)))
    intervals = centers * 100  # 1000 samples, equal weight per bin
    out = ba._shannon_entropy_bits(intervals)
    assert out["entropy_bits"] == pytest.approx(math.log2(n), abs=1e-6)


def test_malformed_line_warning_does_not_crash(tmp_path, capsys):
    """Test 6: malformed JSONL line → warning to stderr, line skipped."""
    p = tmp_path / "bad.jsonl"
    p.write_text("\n".join([
        json.dumps({
            "schema": "openlegion.browser.recorder/v1",
            "agent": "x",
            "reason": "t",
            "event_count": 0,
        }),
        json.dumps({"ts": 1.0, "interval_s": None, "type": "click"}),
        "this is not json {{{",
        json.dumps({"ts": 2.0, "interval_s": 1.0, "type": "click"}),
    ]) + "\n")
    fr = ba.load_file(p)
    assert not fr.skipped
    assert len(fr.events) == 2
    assert fr.malformed_lines == 1
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "malformed" in captured.err.lower()


def test_missing_baseline_path_does_not_crash(tmp_path, capsys):
    """Test 7: missing baseline → returns None, warning to stderr."""
    nonexistent = tmp_path / "nope.jsonl"
    out = ba.load_baseline(nonexistent)
    assert out is None
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_json_output_is_valid(tmp_path, capsys):
    """Test 8: --json emits a single valid JSON object."""
    p = _write_recording(tmp_path / "r.jsonl", events=[
        _make_event(1.0, None, "click", method="x11", success=True),
        _make_event(2.0, 1.0, "click", method="x11", success=True),
    ])
    code = ba.main([
        "--file", str(p),
        "--baseline", "none",
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "report" in parsed
    assert "baseline" in parsed
    assert parsed["baseline"] is None
    assert parsed["report"]["global"]["event_count"] == 2


def test_per_event_type_counts_match_input(tmp_path):
    """Test 9: type breakdown counts exactly mirror input."""
    events = [
        _make_event(1.0, None, "click", method="x11", success=True),
        _make_event(2.0, 1.0, "click", method="x11", success=True),
        _make_event(3.0, 1.0, "keystrokes", char_count=5, fast=False, method="x11"),
        _make_event(4.0, 1.0, "scroll", direction="down", delta=100, method="x11"),
        _make_event(5.0, 1.0, "navigate", host="x", wait_until="load"),
        _make_event(6.0, 1.0, "navigate", host="y", wait_until="load"),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    tb = report["global"]["type_breakdown"]
    assert tb["by_type"] == {
        "click": 2, "keystrokes": 1, "scroll": 1, "navigate": 2,
    }
    assert tb["total"] == 6
    # Fractions sum to 1.
    assert sum(tb["fractions"].values()) == pytest.approx(1.0)


def test_empty_directory_exits_nonzero(tmp_path, capsys):
    """Test 10: no input files → exit code 1, 'No data found' on stderr."""
    code = ba.main([
        "--dir", str(tmp_path),
        "--baseline", "none",
    ])
    assert code == 1
    captured = capsys.readouterr()
    assert "No data found" in captured.err


def test_click_method_fractions_sum_to_one(tmp_path):
    """Test 11: click method distribution sums to 1.0 when clicks exist."""
    events = [
        _make_event(1.0, None, "click", method="x11", success=True),
        _make_event(2.0, 1.0, "click", method="x11", success=True),
        _make_event(3.0, 1.0, "click", method="cdp", success=True),
        _make_event(4.0, 1.0, "click", method="auto", success=True),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    cm = report["global"]["click_method_distribution"]
    assert cm["total"] == 4
    assert cm["by_method"]["x11"] == pytest.approx(0.5)
    assert sum(cm["by_method"].values()) == pytest.approx(1.0, abs=1e-9)


def test_click_method_distribution_empty_when_no_clicks(tmp_path):
    """Test 11b: click method table empty when no clicks."""
    events = [
        _make_event(1.0, None, "scroll", direction="down", delta=100, method="x11"),
        _make_event(2.0, 1.0, "navigate", host="x", wait_until="load"),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    cm = report["global"]["click_method_distribution"]
    assert cm["total"] == 0
    assert cm["by_method"] == {}


def test_navigate_cadence_three_events(tmp_path):
    """Test 12: 3 nav events → median interval matches expected."""
    events = [
        _make_event(100.0, None, "navigate", host="a", wait_until="load"),
        # other event between navs (should not affect nav cadence)
        _make_event(105.0, 5.0, "click", method="x11", success=True),
        _make_event(110.0, 5.0, "navigate", host="b", wait_until="load"),
        _make_event(130.0, 20.0, "navigate", host="c", wait_until="load"),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    nc = report["global"]["navigate_cadence"]
    assert nc["navigate_count"] == 3
    # nav timestamps: 100, 110, 130 -> deltas: 10, 20 -> median 15.0
    assert nc["median_interval_s"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# End-to-end / extra coverage
# ---------------------------------------------------------------------------


def test_end_to_end_subprocess_runs_on_baseline():
    """Spot-check the script runs as ``python -m tools.behavior_analyze``."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline = repo_root / "tools" / "behavior_baseline.jsonl"
    assert baseline.exists(), "synthetic baseline must ship in-repo"
    proc = subprocess.run(
        [
            sys.executable, "-m", "tools.behavior_analyze",
            "--file", str(baseline),
            "--baseline", "none",
            "--json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["report"]["global"]["event_count"] > 0


def test_keystroke_per_char_proxy(tmp_path):
    """Per-char interval = interval_s / char_count."""
    events = [
        _make_event(1.0, None, "keystrokes", char_count=5,
                    fast=False, method="x11"),
        _make_event(2.5, 1.5, "keystrokes", char_count=5,
                    fast=False, method="x11"),  # 0.30 s/char
        _make_event(4.5, 2.0, "keystrokes", char_count=10,
                    fast=False, method="x11"),  # 0.20 s/char
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    kp = report["global"]["keystroke_per_char"]
    # Two valid samples (first event has interval=None).
    assert kp["summary_seconds_per_char"]["count"] == 2
    assert kp["summary_seconds_per_char"]["mean"] == pytest.approx(0.25)


def test_skipped_files_counted(tmp_path, capsys):
    """Files with bad headers are reported as skipped but don't crash."""
    good = _write_recording(tmp_path / "good.jsonl", events=[
        _make_event(1.0, None, "click", method="x11", success=True),
    ])
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not even json\n")
    results = [ba.load_file(good), ba.load_file(bad)]
    report = ba.aggregate(results)
    assert report["files_loaded"] == 1
    assert report["files_skipped"] == 1


def test_baseline_deviation_zero_when_self_compared(tmp_path):
    """Comparing a recording to itself should yield 0% deltas."""
    events = [
        _make_event(1.0, None, "click", method="x11", success=True),
        _make_event(2.0, 1.0, "keystrokes", char_count=5,
                    fast=False, method="x11"),
        _make_event(4.0, 2.0, "click", method="x11", success=True),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    baseline = ba.load_baseline(p)
    assert baseline is not None
    text = ba.render_text(report, baseline)
    # Mean of intervals appears in summary table; matched -> 0%.
    assert "+0.0%" in text or "(0.0%" in text


def test_render_text_handles_no_baseline(tmp_path):
    """Text render works without baseline (regression for None handling)."""
    events = [
        _make_event(1.0, None, "click", method="x11", success=True),
    ]
    p = _write_recording(tmp_path / "r.jsonl", events=events)
    fr = ba.load_file(p)
    report = ba.aggregate([fr])
    text = ba.render_text(report, None)
    assert "Baseline: not loaded" in text


def test_log_bin_index_extremes():
    """Boundary handling of log-spaced bin index."""
    assert ba._log_bin_index(0.001) == 0
    assert ba._log_bin_index(120.0) == ba._ENTROPY_BIN_COUNT - 1
    # mid-range value lands strictly inside.
    idx = ba._log_bin_index(1.0)
    assert 0 < idx < ba._ENTROPY_BIN_COUNT - 1
