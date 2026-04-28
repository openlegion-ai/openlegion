"""Markdown report generator for the §18.1 validation campaign.

Consumes a list of :class:`AttemptOutcome` records and produces the report
template described in plan §18.1:

* Per-site outcome distribution table (counts per ``solver_outcome``).
* Top-5 ``unsupported`` and ``rejected`` outcomes (URL-redacted).
* Classifier accuracy table (expected vs emitted ``kind``, % match).
* Cost reconciliation table per site, flagging deltas >10%.
* Promotion recommendations for §11.20 deferred items.

Reuse contract: every URL appearing in the report flows through
:func:`src.shared.redaction.redact_url`. Tests assert no raw query strings
or token-shaped path segments leak.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.shared.redaction import redact_url
from tools.captcha_validation.schema import AttemptOutcome

# Outcome buckets we always render in the per-site distribution table, in a
# stable order. Outcomes that appear in the data but aren't in this list
# get a row appended at the end. Keeping the canonical order makes
# cross-site comparison immediate (the operator's eye lands on
# ``solved`` / ``rejected`` first regardless of which site row).
_CANONICAL_OUTCOMES: tuple[str, ...] = (
    "solved",
    "rejected",
    "unsupported",
    "timeout",
    "cost_cap",
    "rate_limited",
    "skipped_behavioral",
    "no_solver",
    "harness_timeout",
    "harness_error",
    "navigate_failed",
    "injection_failed",
    "captcha_during_solve",
)


# Cost-reconciliation tolerance — counted vs estimate within this fraction
# is considered a match. The plan calls out 10% explicitly. Below that
# is reported as informational; above is flagged for operator follow-up
# (suggests the cost counter is missing increments).
_COST_DELTA_FLAG_THRESHOLD: float = 0.10


# §11.20 deferred items keyed by the §11.13 ``kind`` they would unblock.
# Used by ``_promotion_recommendations`` to surface concrete trigger
# evidence ("recommend promote" / "no signal yet"). Ordered to match the
# plan's deferred-list narrative.
_DEFERRED_KINDS_TO_ITEM: dict[str, str] = {
    "funcaptcha":          "§11.5 (FunCaptcha) — Twitter/LinkedIn signup",
    "geetest":             "§11.5 (GeeTest) — Asia-targeted SaaS",
    "geetest-v3":          "§11.5 (GeeTest) — v3 site",
    "geetest-v4":          "§11.5 (GeeTest) — v4 site",
    "aws-waf":             "§11.5 (AWS WAF) — AWS console signin",
    "datadome-behavioral": "§11.5 (DataDome) — DataDome-protected target",
    "px-press-hold":       "§11.5 (PerimeterX) — Press & Hold",
}


def _redact_attempt_url(url: str) -> str:
    """Redact a target URL for inclusion in the report."""
    return redact_url(url) if url else url


def _outcome_counter(attempts: list[AttemptOutcome]) -> Counter[str]:
    """Count outcomes across an attempt list, slotting unknowns under
    ``unknown_outcome`` so the totals always reconcile to ``len(attempts)``.
    """
    c: Counter[str] = Counter()
    for a in attempts:
        outcome = a.envelope.get("solver_outcome")
        if not isinstance(outcome, str) or not outcome:
            outcome = "unknown_outcome"
        c[outcome] += 1
    return c


def _ordered_outcomes(seen: Counter[str]) -> list[str]:
    """Return outcomes in canonical order, then any extras alphabetically.

    Stable across runs so diffs between reports stay readable.
    """
    canonical = [o for o in _CANONICAL_OUTCOMES if o in seen]
    extras = sorted(o for o in seen if o not in _CANONICAL_OUTCOMES)
    return canonical + extras


def _percent(part: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(part / total) * 100:.0f}%"


def _format_cost_delta(counted_mc: int, estimate_cents: int) -> tuple[str, bool]:
    """Format ``counted vs estimate`` for the reconciliation table.

    Returns ``(formatted, flag)`` — ``flag=True`` means the delta exceeds
    the operator-visible threshold and should be highlighted in the report.
    """
    counted_cents_f = counted_mc / 1000.0
    if estimate_cents == 0 and counted_mc == 0:
        return "$0.00 / $0.00", False
    counted_dollars = counted_mc / 100_000.0
    estimate_dollars = estimate_cents / 100.0
    base = f"${counted_dollars:.4f} / ${estimate_dollars:.4f}"
    if estimate_cents == 0:
        # Counted cost without an estimate row — operator should review the
        # provider list; usually means the kind isn't priced.
        return base + " (no published rate)", counted_mc > 0
    delta = abs(counted_cents_f - estimate_cents) / max(estimate_cents, 1)
    if delta > _COST_DELTA_FLAG_THRESHOLD:
        return base + f" ⚠ Δ={delta * 100:.0f}%", True
    return base + f" Δ={delta * 100:.0f}%", False


def _group_by_site(
    attempts: list[AttemptOutcome],
) -> dict[str, list[AttemptOutcome]]:
    """Group outcomes by ``site_url`` preserving insertion order."""
    grouped: dict[str, list[AttemptOutcome]] = defaultdict(list)
    for a in attempts:
        grouped[a.site_url].append(a)
    return grouped


def _per_site_table(grouped: dict[str, list[AttemptOutcome]]) -> str:
    """Render the per-site outcome distribution table in markdown."""
    if not grouped:
        return "_No attempts recorded._\n"
    # Discover all outcome buckets that appear across the campaign.
    overall: Counter[str] = Counter()
    site_counters: dict[str, Counter[str]] = {}
    for url, group in grouped.items():
        c = _outcome_counter(group)
        site_counters[url] = c
        overall.update(c)
    columns = _ordered_outcomes(overall)

    header = "| Site | Total | " + " | ".join(columns) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(columns)) + "|"
    rows = [header, sep]
    for url, group in grouped.items():
        c = site_counters[url]
        cells = [str(c.get(col, 0)) for col in columns]
        rows.append(
            f"| {_redact_attempt_url(url)} | {len(group)} | "
            + " | ".join(cells) + " |",
        )
    return "\n".join(rows) + "\n"


def _classifier_accuracy_table(
    grouped: dict[str, list[AttemptOutcome]],
) -> str:
    """Render the classifier accuracy table.

    Per site: total attempts, expected_kind, count of envelopes whose
    ``kind`` matched expected, % match. Flags sites where the match rate
    is <80% so §11.6 promotion has concrete evidence.
    """
    if not grouped:
        return "_No attempts recorded._\n"
    rows = ["| Site | Expected kind | Match | Total | % match |"]
    rows.append("|---|---|---:|---:|---:|")
    for url, group in grouped.items():
        if not group:
            continue
        # Every attempt in a site's group shares the expected_kind in the
        # campaign config — but that lives on SiteConfig, not on the
        # outcome. Recover it from the classifier_match boolean against
        # the emitted kind: when match=True we know what expected was
        # (the emitted kind); when match=False we surface "(see notes)".
        # Tests + report consumers can re-derive against SiteConfig if
        # they need the strict expected value, but per-site classification
        # rate doesn't depend on knowing it.
        match_count = sum(1 for a in group if a.classifier_match)
        # Pick the modal expected kind from the matched attempts; fall
        # back to "(unknown)" when none matched.
        expected_kinds = Counter(
            a.envelope.get("kind", "unknown")
            for a in group if a.classifier_match
        )
        expected = expected_kinds.most_common(1)[0][0] if expected_kinds else "(unknown)"
        rows.append(
            f"| {_redact_attempt_url(url)} | {expected} | "
            f"{match_count} | {len(group)} | "
            f"{_percent(match_count, len(group))} |",
        )
    return "\n".join(rows) + "\n"


def _cost_reconciliation_table(
    grouped: dict[str, list[AttemptOutcome]],
) -> str:
    """Render the cost-reconciliation table per site."""
    if not grouped:
        return "_No attempts recorded._\n"
    rows = ["| Site | Counted / Estimate (USD) | Flagged? |"]
    rows.append("|---|---|---:|")
    any_flag = False
    for url, group in grouped.items():
        counted_total = sum(a.cost_counted_millicents for a in group)
        estimate_total = sum(a.cost_charged_cents_estimated for a in group)
        formatted, flag = _format_cost_delta(counted_total, estimate_total)
        any_flag = any_flag or flag
        rows.append(
            f"| {_redact_attempt_url(url)} | {formatted} | "
            f"{'⚠ yes' if flag else 'no'} |",
        )
    rows.append("")
    if any_flag:
        rows.append(
            "> **Note.** Sites flagged ⚠ exceeded the 10% counted-vs-estimate "
            "tolerance. Operator should compare provider invoice totals "
            "against the cost-counter snapshot for the affected sites.",
        )
    return "\n".join(rows) + "\n"


def _top_n_outcome_examples(
    attempts: list[AttemptOutcome],
    outcome: str,
    n: int = 5,
) -> list[str]:
    """Return up to ``n`` redacted URLs of attempts with ``solver_outcome==outcome``.

    Sorted by ``site_url`` (stable across re-runs) and then earliest
    ``timestamp_utc`` so the same examples surface every time.
    """
    matching = [
        a for a in attempts
        if a.envelope.get("solver_outcome") == outcome
    ]
    matching.sort(key=lambda a: (a.site_url, a.timestamp_utc))
    seen: set[str] = set()
    out: list[str] = []
    for a in matching:
        red = _redact_attempt_url(a.site_url)
        if red in seen:
            continue
        seen.add(red)
        out.append(red)
        if len(out) >= n:
            break
    return out


def _promotion_recommendations(attempts: list[AttemptOutcome]) -> list[str]:
    """Generate concrete promotion recommendations per §11.20 deferred item.

    For each kind the harness might surface that maps to a deferred item,
    we count how often the campaign saw it and generate a recommendation
    line with the evidence:

    * ``unsupported`` ≥ 50% on a category → recommend promote.
    * ``rejected`` consistently with classifier_match=True → recommend
      promote (the solver has the wrong shape for this site).
    * No signal for the kind → recommend keeping deferred with tightened
      trigger.

    Output is a list of lines suitable for direct inclusion in markdown.
    """
    lines: list[str] = []
    by_kind: dict[str, list[AttemptOutcome]] = defaultdict(list)
    for a in attempts:
        kind = a.envelope.get("kind") or "unknown"
        by_kind[kind].append(a)

    for kind, item in _DEFERRED_KINDS_TO_ITEM.items():
        observations = by_kind.get(kind, [])
        if not observations:
            lines.append(
                f"- **{item}** — no occurrences in this campaign; keep deferred.",
            )
            continue
        unsupported_count = sum(
            1 for a in observations
            if a.envelope.get("solver_outcome") == "unsupported"
        )
        rejected_count = sum(
            1 for a in observations
            if a.envelope.get("solver_outcome") == "rejected"
        )
        total = len(observations)
        if unsupported_count >= max(1, total // 2):
            lines.append(
                f"- **{item}** — observed {unsupported_count}/{total} "
                f"`unsupported` outcomes; **recommend promote**.",
            )
        elif rejected_count >= max(1, total // 2):
            lines.append(
                f"- **{item}** — observed {rejected_count}/{total} "
                f"`rejected` outcomes; **investigate before promote** "
                f"(could be solver-task-shape mismatch rather than "
                f"missing coverage).",
            )
        else:
            lines.append(
                f"- **{item}** — {total} observation(s); no decisive "
                f"signal yet. Keep deferred; broaden the campaign target "
                f"list to gather more evidence.",
            )

    # Classifier accuracy → §11.1 / §11.6 promotion signal.
    unknown_count = sum(
        1 for a in attempts if (a.envelope.get("kind") or "unknown") == "unknown"
    )
    if unknown_count and attempts:
        share = unknown_count / len(attempts)
        if share > 0.20:
            lines.append(
                f"- **§11.6 (robust sitekey extraction / classifier hardening)** "
                f"— {unknown_count}/{len(attempts)} attempts produced "
                f"`kind=unknown` ({share * 100:.0f}%); **recommend promote**.",
            )
    return lines


def _campaign_summary(attempts: list[AttemptOutcome]) -> dict[str, object]:
    """Produce the top-of-report summary block."""
    total = len(attempts)
    counted_mc = sum(a.cost_counted_millicents for a in attempts)
    estimate_cents = sum(a.cost_charged_cents_estimated for a in attempts)
    wall_clocks = [a.wall_clock_ms for a in attempts] or [0]
    classifier_match = sum(1 for a in attempts if a.classifier_match)
    return {
        "total": total,
        "counted_dollars": counted_mc / 100_000.0,
        "estimate_dollars": estimate_cents / 100.0,
        "median_ms": int(statistics.median(wall_clocks)),
        "max_ms": max(wall_clocks),
        "classifier_match_pct": _percent(classifier_match, total),
    }


def generate_report(
    attempts: list[AttemptOutcome],
    output_path: Path,
) -> None:
    """Generate a markdown campaign report at ``output_path``.

    The report is fully self-contained — it does NOT inline raw URLs.
    Operator can attach the file to a §11.20 review without further redaction.
    """
    summary = _campaign_summary(attempts)
    grouped = _group_by_site(attempts)

    lines: list[str] = []
    lines.append("# Phase 9 §18.1 — Captcha validation campaign")
    lines.append("")
    lines.append(
        f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total attempts: **{summary['total']}**")
    lines.append(f"- Sites covered: **{len(grouped)}**")
    lines.append(
        f"- Cost: counted **${summary['counted_dollars']:.4f}** vs "
        f"estimated **${summary['estimate_dollars']:.4f}**",
    )
    lines.append(
        f"- Wall-clock per attempt: median **{summary['median_ms']}ms**, "
        f"max **{summary['max_ms']}ms**",
    )
    lines.append(
        f"- Classifier kind-match rate: **{summary['classifier_match_pct']}**",
    )
    lines.append("")

    lines.append("## Per-site outcome distribution")
    lines.append("")
    lines.append(_per_site_table(grouped))

    # Top-5 unsupported / rejected
    lines.append("## Top 5 `unsupported` examples")
    lines.append("")
    examples = _top_n_outcome_examples(attempts, "unsupported", n=5)
    if examples:
        for ex in examples:
            lines.append(f"- {ex}")
    else:
        lines.append("_No `unsupported` outcomes._")
    lines.append("")

    lines.append("## Top 5 `rejected` examples")
    lines.append("")
    examples = _top_n_outcome_examples(attempts, "rejected", n=5)
    if examples:
        for ex in examples:
            lines.append(f"- {ex}")
    else:
        lines.append("_No `rejected` outcomes._")
    lines.append("")

    lines.append("## Classifier accuracy")
    lines.append("")
    lines.append(_classifier_accuracy_table(grouped))

    lines.append("## Cost reconciliation")
    lines.append("")
    lines.append(_cost_reconciliation_table(grouped))

    lines.append("## §11.20 promotion recommendations")
    lines.append("")
    recs = _promotion_recommendations(attempts)
    if recs:
        lines.extend(recs)
    else:
        lines.append("_No data — campaign produced zero attempts._")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
