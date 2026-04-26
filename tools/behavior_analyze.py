"""Offline behavioral entropy analyzer (Phase 6 §9.5).

Reads behavioral recordings produced by ``src/browser/recorder.py`` (one
JSONL file per dump, header line + one event per subsequent line) and
emits a jerk/snap entropy report. Intended for operator review of
whether the browser-automation timing distributions look human.

What this is
------------
A *purely offline* operator tool. There are no runtime hooks, no agent
surface (this is not a ``@skill``), and no mesh routes. Operators run
this script against ``/data/debug/*.jsonl`` (the recorder's dump
directory) after a session to inspect timing distributions and compare
against a baseline corpus.

What this is not
----------------
- Not a real-time monitor. The recorder writes on browser-instance
  teardown; this analyzer reads finished files.
- Not a measurement of literal mechanical jerk/snap. We don't capture
  per-key positions, only per-event timestamps and char_counts. The
  "jerk/snap" framing is a *proxy* via inter-keystroke timing
  variability — see ``_keystroke_per_char_metrics``.

Baseline
--------
Default baseline (``tools/behavior_baseline.jsonl``) is a SYNTHETIC
PLACEHOLDER. The numbers are realistic-looking but were not collected
from a human. Replace with a real corpus collected from internal
volunteers (per your operator consent policy) before relying on the
deviation metrics in production review.

Usage
-----
    python -m tools.behavior_analyze --dir /data/debug
    python -m tools.behavior_analyze --file recording.jsonl --json
    python -m tools.behavior_analyze --filter-agent agent-7 --since 1714000000
    python -m tools.behavior_analyze --baseline none  # skip comparison
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.shared.utils import setup_logging

logger = setup_logging("tools.behavior_analyze")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASELINE = _REPO_ROOT / "tools" / "behavior_baseline.jsonl"
_DEFAULT_DUMP_DIR = Path("/data/debug")

# Log-spaced bins for cadence-entropy buckets, 10ms .. 60s. Human input
# intervals span four-plus orders of magnitude (rapid keystrokes vs
# pauses to read), so linear bins concentrate everything in one bucket.
# Log spacing gives each order of magnitude proportional weight.
_ENTROPY_BIN_MIN_S = 0.01
_ENTROPY_BIN_MAX_S = 60.0
_ENTROPY_BIN_COUNT = 10


@dataclass
class Event:
    ts: float
    interval_s: float | None
    type: str
    raw: dict[str, Any]


@dataclass
class FileLoadResult:
    """Outcome of loading a single recorder file."""

    path: Path
    header: dict[str, Any] | None
    events: list[Event] = field(default_factory=list)
    malformed_lines: int = 0
    skipped: bool = False
    skip_reason: str | None = None


def _iter_jsonl(path: Path) -> Iterable[tuple[int, str]]:
    """Yield ``(line_number, raw)`` for non-empty lines."""
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                yield n, stripped


def load_file(path: Path) -> FileLoadResult:
    """Load a recorder JSONL file. Never raises; reports issues inline."""
    res = FileLoadResult(path=path, header=None)
    try:
        lines = list(_iter_jsonl(path))
    except OSError as e:
        res.skipped = True
        res.skip_reason = f"unreadable: {e}"
        return res

    if not lines:
        res.skipped = True
        res.skip_reason = "empty file"
        return res

    # First non-empty line is the header. Fail soft if it's not.
    header_n, header_raw = lines[0]
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError as e:
        res.skipped = True
        res.skip_reason = f"header line {header_n} is not valid JSON: {e}"
        return res
    if not isinstance(header, dict) or header.get("schema") != "openlegion.browser.recorder/v1":
        # Could be an event-only file; treat header as event if it has 'type' field.
        if isinstance(header, dict) and "type" in header and "ts" in header:
            res.header = None
            event_lines = lines
        else:
            res.skipped = True
            res.skip_reason = "missing or unrecognized header"
            return res
    else:
        res.header = header
        event_lines = lines[1:]

    for n, raw in event_lines:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            res.malformed_lines += 1
            print(
                f"warning: {path}:{n}: malformed JSON, skipping line",
                file=sys.stderr,
            )
            continue
        if not isinstance(ev, dict) or "ts" not in ev or "type" not in ev:
            res.malformed_lines += 1
            print(
                f"warning: {path}:{n}: missing required fields, skipping line",
                file=sys.stderr,
            )
            continue
        try:
            ts = float(ev["ts"])
        except (TypeError, ValueError):
            res.malformed_lines += 1
            print(
                f"warning: {path}:{n}: non-numeric ts, skipping line",
                file=sys.stderr,
            )
            continue
        interval = ev.get("interval_s")
        if interval is not None:
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                interval = None
        res.events.append(
            Event(ts=ts, interval_s=interval, type=str(ev["type"]), raw=ev)
        )
    return res


def discover_files(directory: Path) -> list[Path]:
    """Return ``*.jsonl`` files in ``directory``, sorted; ignore .partial."""
    if not directory.exists() or not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix == ".jsonl"
    ]
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_pct(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    # Linear interpolation between closest ranks.
    k = (len(s) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "p90": None,
            "p99": None,
            "iqr": None,
        }
    p25 = _safe_pct(values, 0.25)
    p75 = _safe_pct(values, 0.75)
    iqr = (p75 - p25) if (p25 is not None and p75 is not None) else None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p90": _safe_pct(values, 0.90),
        "p99": _safe_pct(values, 0.99),
        "iqr": iqr,
    }


def _log_bin_index(value: float) -> int:
    """Return bin index in [0, _ENTROPY_BIN_COUNT)."""
    if value <= _ENTROPY_BIN_MIN_S:
        return 0
    if value >= _ENTROPY_BIN_MAX_S:
        return _ENTROPY_BIN_COUNT - 1
    log_min = math.log(_ENTROPY_BIN_MIN_S)
    log_max = math.log(_ENTROPY_BIN_MAX_S)
    frac = (math.log(value) - log_min) / (log_max - log_min)
    idx = int(frac * _ENTROPY_BIN_COUNT)
    return min(idx, _ENTROPY_BIN_COUNT - 1)


def _shannon_entropy_bits(values: list[float]) -> dict[str, Any]:
    """Shannon entropy (bits) of values binned into log-spaced buckets.

    Returned ``entropy_bits`` is comparable across runs because the bin
    set is fixed. ``entropy_bits_max = log2(N_BINS)`` is the upper bound
    when every bin is equally populated; we report it so deviation is
    interpretable.
    """
    if not values:
        return {
            "entropy_bits": None,
            "entropy_bits_max": math.log2(_ENTROPY_BIN_COUNT),
            "bin_counts": [0] * _ENTROPY_BIN_COUNT,
            "sample_count": 0,
        }
    counts = [0] * _ENTROPY_BIN_COUNT
    for v in values:
        counts[_log_bin_index(v)] += 1
    total = sum(counts)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return {
        "entropy_bits": entropy,
        "entropy_bits_max": math.log2(_ENTROPY_BIN_COUNT),
        "bin_counts": counts,
        "sample_count": total,
    }


def _keystroke_per_char_metrics(events: list[Event]) -> dict[str, Any]:
    """Per-character interval distribution for keystroke events.

    True jerk/snap (3rd/4th derivatives of position) requires per-key
    positional samples we never collect (privacy: see recorder docstring).
    What we *can* report is the variability of per-character timing,
    which is a useful proxy for "is this typing rhythm too regular":
    a metronomic per-char interval is a stealth red flag, even without
    geometric jerk.
    """
    per_char: list[float] = []
    for ev in events:
        if ev.type != "keystrokes":
            continue
        char_count = ev.raw.get("char_count")
        interval = ev.interval_s
        try:
            cc = int(char_count) if char_count is not None else 0
        except (TypeError, ValueError):
            cc = 0
        if cc <= 0 or interval is None or interval <= 0:
            continue
        per_char.append(interval / cc)
    return {
        "summary_seconds_per_char": _summary(per_char),
        "samples": per_char,
    }


def _navigate_cadence(events: list[Event]) -> dict[str, Any]:
    nav_ts = sorted(ev.ts for ev in events if ev.type == "navigate")
    if len(nav_ts) < 2:
        return {
            "navigate_count": len(nav_ts),
            "median_interval_s": None,
            "mean_interval_s": None,
        }
    deltas = [b - a for a, b in zip(nav_ts, nav_ts[1:]) if b > a]
    return {
        "navigate_count": len(nav_ts),
        "median_interval_s": statistics.median(deltas) if deltas else None,
        "mean_interval_s": statistics.fmean(deltas) if deltas else None,
    }


def _click_method_distribution(events: list[Event]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for ev in events:
        if ev.type != "click":
            continue
        method = ev.raw.get("method")
        if method is None:
            continue
        counts[str(method)] += 1
    total = sum(counts.values())
    if total == 0:
        return {"total": 0, "by_method": {}}
    return {
        "total": total,
        "by_method": {m: c / total for m, c in counts.most_common()},
        "raw_counts": dict(counts),
    }


def _type_breakdown(events: list[Event]) -> dict[str, Any]:
    counts: Counter[str] = Counter(ev.type for ev in events)
    total = sum(counts.values())
    if total == 0:
        return {"total": 0, "by_type": {}, "fractions": {}}
    return {
        "total": total,
        "by_type": dict(counts),
        "fractions": {t: c / total for t, c in counts.most_common()},
    }


def compute_metrics(events: list[Event]) -> dict[str, Any]:
    """All metrics for an event stream. ``events`` may be empty."""
    intervals = [
        ev.interval_s
        for ev in events
        if ev.interval_s is not None and ev.interval_s > 0
    ]
    return {
        "event_count": len(events),
        "interval_summary_seconds": _summary(intervals),
        "cadence_entropy": _shannon_entropy_bits(intervals),
        "keystroke_per_char": _keystroke_per_char_metrics(events),
        "type_breakdown": _type_breakdown(events),
        "click_method_distribution": _click_method_distribution(events),
        "navigate_cadence": _navigate_cadence(events),
    }


# ---------------------------------------------------------------------------
# Aggregation across files / agents
# ---------------------------------------------------------------------------


def aggregate(
    files: list[FileLoadResult],
    *,
    filter_agent: str | None = None,
    since: float | None = None,
) -> dict[str, Any]:
    """Aggregate events across files into per-agent + global metrics."""
    by_agent: dict[str, list[Event]] = {}
    files_used = 0
    for fr in files:
        if fr.skipped:
            continue
        # Header is informational; agent comes from header if present,
        # else "unknown".
        agent = (fr.header or {}).get("agent", "unknown")
        if filter_agent is not None and agent != filter_agent:
            continue
        kept_events = [
            ev for ev in fr.events
            if since is None or ev.ts >= since
        ]
        if not kept_events and not fr.events:
            # Header-only file. Still register the agent so a "no events"
            # report is produced rather than swallowing the file silently.
            by_agent.setdefault(agent, [])
            files_used += 1
            continue
        if kept_events:
            by_agent.setdefault(agent, []).extend(kept_events)
            files_used += 1

    per_agent_metrics = {
        agent: compute_metrics(evs) for agent, evs in by_agent.items()
    }
    all_events: list[Event] = []
    for evs in by_agent.values():
        all_events.extend(evs)
    return {
        "files_loaded": files_used,
        "files_skipped": sum(1 for f in files if f.skipped),
        "malformed_lines_total": sum(
            f.malformed_lines for f in files if not f.skipped
        ),
        "agents": list(by_agent.keys()),
        "per_agent": per_agent_metrics,
        "global": compute_metrics(all_events),
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Compute baseline metrics from a single JSONL file. None on miss."""
    if not path.exists():
        print(
            f"warning: baseline {path} does not exist; skipping deviation",
            file=sys.stderr,
        )
        return None
    fr = load_file(path)
    if fr.skipped:
        print(
            f"warning: baseline {path} unusable ({fr.skip_reason}); skipping deviation",
            file=sys.stderr,
        )
        return None
    return compute_metrics(fr.events)


def _pct_delta(actual: float | None, baseline: float | None) -> float | None:
    if actual is None or baseline is None or baseline == 0:
        return None
    return (actual - baseline) / baseline * 100.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(v: Any, places: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{places}f}"
    return str(v)


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return ""
    sign = "+" if d >= 0 else ""
    return f" ({sign}{d:.1f}% vs base)"


def _render_summary_table(label: str, summary: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    lines = [f"#### {label}", ""]
    if summary["count"] == 0:
        lines.append("No samples.")
        lines.append("")
        return lines
    lines.append("| Stat | Value | Delta |")
    lines.append("|---|---|---|")
    for key in ("count", "mean", "median", "stddev", "p90", "p99", "iqr"):
        actual = summary.get(key)
        delta: float | None = None
        if baseline is not None:
            delta = _pct_delta(actual, baseline.get(key))
        places = 0 if key == "count" else 4
        delta_text = _fmt_delta(delta).strip() if delta is not None else ""
        lines.append(f"| {key} | {_fmt(actual, places)} | {delta_text} |")
    lines.append("")
    return lines


def render_text(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    """Markdown-flavored text report."""
    out: list[str] = []
    out.append("# Behavior analyzer report")
    out.append("")
    out.append(f"- Files loaded: {report['files_loaded']}")
    out.append(f"- Files skipped: {report['files_skipped']}")
    out.append(f"- Malformed lines skipped: {report['malformed_lines_total']}")
    out.append(f"- Agents: {', '.join(report['agents']) if report['agents'] else '(none)'}")
    if baseline is not None:
        out.append("- Baseline: loaded; deltas shown as `(±X% vs base)`")
    else:
        out.append("- Baseline: not loaded (no deviation column)")
    out.append("")

    sections: list[tuple[str, dict[str, Any]]] = [("global", report["global"])]
    for agent, metrics in sorted(report["per_agent"].items()):
        sections.append((f"agent: {agent}", metrics))

    for label, m in sections:
        out.append(f"## {label}")
        out.append("")
        out.append(f"Event count: **{m['event_count']}**")
        out.append("")
        if m["event_count"] == 0:
            out.append("_No events; nothing to report._")
            out.append("")
            continue

        # Inter-event interval summary.
        out.extend(_render_summary_table(
            "Inter-event interval (seconds)",
            m["interval_summary_seconds"],
            (baseline or {}).get("interval_summary_seconds"),
        ))

        # Cadence entropy.
        ce = m["cadence_entropy"]
        out.append("#### Cadence entropy (Shannon, log-spaced bins 10ms..60s)")
        out.append("")
        if ce["sample_count"] == 0:
            out.append("No interval samples.")
            out.append("")
        else:
            base_ce = (baseline or {}).get("cadence_entropy") or {}
            delta = _pct_delta(ce["entropy_bits"], base_ce.get("entropy_bits"))
            out.append(
                f"- entropy_bits: {_fmt(ce['entropy_bits'])} / "
                f"max {_fmt(ce['entropy_bits_max'])}{_fmt_delta(delta)}"
            )
            out.append(f"- bin_counts: {ce['bin_counts']}")
            out.append(
                "  (higher entropy = more variable cadence; metronomic input -> 0)"
            )
            out.append("")

        # Keystroke per-char proxy.
        kp = m["keystroke_per_char"]["summary_seconds_per_char"]
        out.extend(_render_summary_table(
            "Keystroke per-char interval proxy (seconds; jerk/snap surrogate)",
            kp,
            ((baseline or {}).get("keystroke_per_char") or {}).get(
                "summary_seconds_per_char"
            ),
        ))
        out.append(
            "_Per-character interval is a *proxy* for jerk/snap — we don't capture "
            "per-key positions (privacy), so this measures rhythm regularity rather "
            "than literal mechanical derivatives._"
        )
        out.append("")

        # Type breakdown.
        tb = m["type_breakdown"]
        out.append("#### Event type breakdown")
        out.append("")
        if tb["total"] == 0:
            out.append("No events.")
        else:
            out.append("| Type | Count | Fraction |")
            out.append("|---|---|---|")
            for t in sorted(tb["by_type"]):
                out.append(
                    f"| {t} | {tb['by_type'][t]} | "
                    f"{_fmt(tb['fractions'][t], 3)} |"
                )
        out.append("")

        # Click method.
        cm = m["click_method_distribution"]
        if cm["total"] > 0:
            out.append("#### Click method distribution")
            out.append("")
            out.append("| Method | Fraction |")
            out.append("|---|---|")
            for method, frac in cm["by_method"].items():
                out.append(f"| {method} | {_fmt(frac, 3)} |")
            out.append("")
        else:
            out.append("#### Click method distribution")
            out.append("")
            out.append("_No click events._")
            out.append("")

        # Navigate cadence.
        nc = m["navigate_cadence"]
        out.append("#### Navigate cadence")
        out.append("")
        out.append(f"- count: {nc['navigate_count']}")
        out.append(f"- median interval (s): {_fmt(nc['median_interval_s'])}")
        out.append(f"- mean interval (s):   {_fmt(nc['mean_interval_s'])}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="behavior_analyze",
        description=(
            "Offline analyzer for browser behavioral recordings. "
            "Reads JSONL dumps from src/browser/recorder.py and reports "
            "interval / entropy / cadence metrics."
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--dir",
        type=Path,
        help=f"Directory of *.jsonl recordings (default: {_DEFAULT_DUMP_DIR})",
    )
    src.add_argument(
        "--file",
        type=Path,
        help="Single recording file to analyze",
    )
    p.add_argument(
        "--baseline",
        default=str(_DEFAULT_BASELINE),
        help=(
            "Path to baseline JSONL for deviation column. "
            "Pass 'none' to disable. "
            f"Default: {_DEFAULT_BASELINE}"
        ),
    )
    p.add_argument(
        "--filter-agent",
        type=str,
        help="Only include events from this agent id",
    )
    p.add_argument(
        "--since",
        type=float,
        help="Drop events with ts older than this Unix timestamp",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object instead of a text report",
    )
    return p


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.file is not None:
        if not args.file.exists():
            print(f"error: {args.file}: not found", file=sys.stderr)
            return []
        return [args.file]
    directory = args.dir if args.dir is not None else _DEFAULT_DUMP_DIR
    return discover_files(directory)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    paths = _resolve_inputs(args)
    if not paths:
        print("No data found.", file=sys.stderr)
        return 1

    file_results = [load_file(p) for p in paths]
    report = aggregate(
        file_results,
        filter_agent=args.filter_agent,
        since=args.since,
    )

    # Edge case: every file got skipped.
    if report["files_loaded"] == 0:
        print("No data found.", file=sys.stderr)
        return 1

    baseline_metrics: dict[str, Any] | None = None
    if str(args.baseline).lower() != "none":
        baseline_metrics = load_baseline(Path(args.baseline))

    if args.json:
        out = {
            "report": report,
            "baseline": baseline_metrics,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print(render_text(report, baseline_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
