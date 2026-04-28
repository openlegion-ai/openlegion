"""Phase 9 §18.1 validation campaign runner.

Drives a real :class:`BrowserManager` end-to-end against operator-curated
target sites, captures the §11.13 envelope per attempt, reconciles cost
counter delta against the published estimate, and produces a JSONL outcome
ledger plus a markdown report.

NEVER auto-runs at import time. Entry point:

  python -m tools.captcha_validation.runner config.local.yaml [output_dir]
  python -m tools.captcha_validation.runner config.local.yaml --dry-run

Threading model: a single ``BrowserManager`` is created per campaign run and
re-used across all sites. A *fresh* ``CamoufoxInstance`` is opened per site
(via ``mgr.stop(site_agent_id)`` between sites) so the fingerprint resets —
critical for the validity of the campaign data; otherwise burn signals
correlate across sites and corrupt the per-site outcome distributions.

Per-attempt budget: every attempt is wrapped in ``asyncio.wait_for(180s)`` so
a stuck Playwright page can never stall the campaign. Wall-clock failures
become harness errors (``error`` field on the ``AttemptOutcome``) so the
report can flag flaky targets.

Cost reconciliation: before/after each attempt the runner reads
``captcha_cost_counter.get_millicents(agent_id)`` to derive
``cost_counted_millicents``. The ``cost_charged_cents_estimated`` is
computed from the envelope's emitted ``kind`` + the solver's provider via
``estimate_millicents``. Discrepancy >10% per site is flagged in the
report — drives §11.10 promotion decisions.

Cost-budget abort: cumulative counted spend is checked AFTER every attempt;
crossing the campaign's ``cost_budget_usd`` raises :class:`BudgetExceeded`
which terminates the loop with a clear log line.

Fingerprint-burn abort: 5 consecutive ``solver_outcome="rejected"`` (or any
of the burn-shaped outcomes) for one site marks that site as burned and
skips its remaining attempts. Other sites continue.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.browser import captcha_cost_counter as cost
from tools.captcha_validation.schema import (
    AttemptOutcome,
    CampaignConfig,
    SiteConfig,
)

if TYPE_CHECKING:  # pragma: no cover - type-only
    from src.browser.service import BrowserManager

logger = logging.getLogger("tools.captcha_validation.runner")


# Per-attempt hard timeout. Even if the production solver disagrees with the
# kind's documented timeout (§11.9), the harness must not block the whole
# campaign on one attempt. Generous default; tests override this directly.
ATTEMPT_TIMEOUT_S: float = 180.0

# Pacing — how long to wait between same-site attempts in a non-test run.
# Defaults to a short jitter so attempts within a single same-day session
# don't fire back-to-back. Tests set this to 0 via the ``pace_seconds`` arg
# on :func:`run_campaign`.
DEFAULT_PACE_SECONDS: float = 1.5

# Number of consecutive "rejected"-shaped outcomes before we declare the
# fingerprint burned for that site and skip the rest of its attempts.
BURN_STREAK_THRESHOLD: int = 5

# Outcomes that strongly suggest the fingerprint is burned (token rejected
# by target server; repeated challenge after solve). These count toward the
# ``BURN_STREAK_THRESHOLD``. ``rate_limited``/``cost_cap``/``timeout`` do NOT
# — those are operational, not fingerprint-driven.
_BURN_SHAPED_OUTCOMES: frozenset[str] = frozenset({
    "rejected",
    "injection_failed",
    "captcha_during_solve",
})


class BudgetExceeded(RuntimeError):
    """Raised when the cumulative counted spend crosses ``cost_budget_usd``."""


@dataclasses.dataclass
class CampaignReport:
    """Return value of :func:`run_campaign` — drives the markdown report.

    ``attempts`` is the in-memory mirror of ``output_dir/attempts.jsonl``
    (one :class:`AttemptOutcome` per attempt, in execution order).
    ``aborted_sites`` lists site URLs that hit the burn-streak threshold.
    ``budget_exceeded`` is True when the campaign aborted on cost cap.
    ``output_dir`` is the directory containing the JSONL ledger + the
    generated markdown report.
    """

    attempts: list[AttemptOutcome]
    aborted_sites: list[str]
    budget_exceeded: bool
    output_dir: Path
    report_path: Path | None


def load_campaign(config_path: Path) -> CampaignConfig:
    """Load and validate the YAML config file.

    Raises ``pydantic.ValidationError`` on malformed configs — the exception
    message is the operator's error UX. We never fail open on bad input.
    """
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ValueError(f"{config_path} is empty or contains only whitespace")
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} top-level must be a mapping, got {type(raw).__name__}")
    return CampaignConfig(**raw)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_burn_shaped(envelope: dict) -> bool:
    """True iff the envelope's ``solver_outcome`` is fingerprint-burn-shaped."""
    outcome = envelope.get("solver_outcome")
    if not isinstance(outcome, str):
        return False
    return outcome in _BURN_SHAPED_OUTCOMES


def _envelope_kind(envelope: dict) -> str:
    """Return the envelope's ``kind`` string, defaulting to 'unknown'."""
    k = envelope.get("kind")
    return k if isinstance(k, str) and k else "unknown"


def _solver_provider(mgr: "BrowserManager") -> str:
    """Return the provider name attached to the solver, or 'unknown'.

    The solver may be ``None`` when the operator hasn't configured a solver
    — in that case every attempt's ``solver_outcome`` will be ``no_solver``
    and the cost reconciliation is trivially zero on both sides.
    """
    solver = getattr(mgr, "_captcha_solver", None)
    if solver is None:
        return "unknown"
    provider = getattr(solver, "provider", "")
    return provider if isinstance(provider, str) and provider else "unknown"


def _estimate_charge_cents(provider: str, envelope: dict) -> int:
    """Compute ``cost_charged_cents_estimated`` from the envelope.

    Returns 0 when the variant has no published rate, when the solver
    short-circuited, or when the envelope reported no successful charge.
    """
    outcome = envelope.get("solver_outcome")
    if outcome != "solved":
        # Provider only charges on a delivered token. Other outcomes
        # (rejected, timeout, cost_cap, rate_limited, no_solver) cost zero.
        return 0
    kind = _envelope_kind(envelope)
    proxy_aware = bool(envelope.get("solver_used_proxy_aware", False))
    mc = cost.estimate_millicents(provider, kind, proxy_aware=proxy_aware)
    if mc is None:
        return 0
    # millicents → cents (rounded — operators want a whole-cent estimate).
    return int(round(mc / 1000))


async def _ensure_fresh_instance(mgr: "BrowserManager", agent_id: str) -> None:
    """Tear down any prior instance for ``agent_id`` so the next nav opens
    a fresh Camoufox profile (clean fingerprint for the new site).

    This is safe to call when no instance exists — :meth:`BrowserManager.stop`
    is a no-op in that case.
    """
    with contextlib.suppress(Exception):
        await mgr.stop(agent_id)


async def _drive_attempt(
    mgr: "BrowserManager",
    agent_id: str,
    site: SiteConfig,
) -> dict:
    """Run a single attempt: navigate → optionally click → solve_captcha.

    Returns the §11.13 envelope dict (the ``data`` field of the
    ``solve_captcha`` response). Caller wraps this in ``asyncio.wait_for``
    so a stuck Playwright page can't pin the campaign.
    """
    nav_result = await mgr.navigate(
        agent_id, site.url,
        wait_ms=2000, wait_until="load",
    )
    if not nav_result.get("success"):
        # Surface the navigate error as a synthetic envelope so the report
        # can group it under "harness-side failures" without a special
        # branch in the outcome aggregator.
        return {
            "captcha_found": False,
            "kind": "unknown",
            "solver_outcome": "navigate_failed",
            "solver_confidence": "low",
            "next_action": "operator_review",
            "error": nav_result.get("error", "navigate failed"),
        }

    if site.interaction == "navigate_then_click_submit":
        if not site.interaction_selector:
            return {
                "captcha_found": False,
                "kind": "unknown",
                "solver_outcome": "config_error",
                "solver_confidence": "low",
                "next_action": "operator_review",
                "error": "interaction_then_click requires interaction_selector",
            }
        click_result = await mgr.click(
            agent_id, selector=site.interaction_selector,
        )
        if not click_result.get("success"):
            # Clicks failing isn't necessarily a captcha failure — the page
            # may have changed shape. Record the harness-side error and let
            # solve_captcha probe whatever the current page looks like.
            logger.warning(
                "click %r on attempt for %s failed: %s",
                site.interaction_selector,
                site.category,
                click_result.get("error"),
            )

    solve_result = await mgr.solve_captcha(
        agent_id,
        retry_previous=True,
    )
    if not solve_result.get("success"):
        return {
            "captcha_found": False,
            "kind": "unknown",
            "solver_outcome": "harness_error",
            "solver_confidence": "low",
            "next_action": "operator_review",
            "error": solve_result.get("error", "solve_captcha returned success=false"),
        }
    return solve_result.get("data", {}) or {}


async def _run_one_attempt(
    mgr: "BrowserManager",
    agent_id: str,
    site: SiteConfig,
    attempt_index: int,
    *,
    timeout_s: float,
    cumulative_counted_millicents_before: int,
) -> tuple[AttemptOutcome, int]:
    """Execute one attempt and produce an :class:`AttemptOutcome`.

    Returns ``(outcome, cumulative_counted_millicents_after)`` where the
    cumulative number folds in the per-attempt cost-counter delta. Caller
    persists the outcome to the JSONL ledger AND THEN runs
    :func:`_budget_check` to decide whether to abort — keeping budget
    enforcement out of this routine guarantees every attempt the harness
    actually runs reaches the ledger, even when the campaign aborts.
    """
    started = _utc_now_iso()
    t0 = time.monotonic()
    counted_before = await cost.get_millicents(agent_id)

    error: str | None = None
    envelope: dict[str, Any] = {}
    try:
        envelope = await asyncio.wait_for(
            _drive_attempt(mgr, agent_id, site),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        error = f"attempt timed out after {timeout_s:.0f}s"
        envelope = {
            "captcha_found": False,
            "kind": "unknown",
            "solver_outcome": "harness_timeout",
            "solver_confidence": "low",
            "next_action": "operator_review",
        }
    except Exception as e:  # noqa: BLE001 — defensive: any exception from
        # navigate/click/solve must NOT stall the campaign loop.
        error = f"{type(e).__name__}: {e}"
        envelope = {
            "captcha_found": False,
            "kind": "unknown",
            "solver_outcome": "harness_error",
            "solver_confidence": "low",
            "next_action": "operator_review",
        }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    counted_after = await cost.get_millicents(agent_id)
    counted_delta = max(0, counted_after - counted_before)

    cumulative = cumulative_counted_millicents_before + counted_delta

    provider = _solver_provider(mgr)
    estimate_cents = _estimate_charge_cents(provider, envelope)

    outcome = AttemptOutcome(
        site_url=site.url,
        attempt_index=attempt_index,
        wall_clock_ms=elapsed_ms,
        envelope=envelope,
        cost_charged_cents_estimated=estimate_cents,
        cost_counted_millicents=counted_delta,
        fingerprint_burn_signal=_is_burn_shaped(envelope),
        classifier_match=(_envelope_kind(envelope) == site.expected_kind),
        error=error,
        timestamp_utc=started,
    )
    return outcome, cumulative


def _budget_check(
    *,
    cumulative_millicents: int,
    budget_millicents: int,
    site_category: str,
    attempt_index: int,
) -> None:
    """Raise :class:`BudgetExceeded` when ``cumulative >= budget``.

    Called by the campaign loop AFTER the attempt's outcome has been
    persisted. Keeping the budget check separate from outcome construction
    means the budget abort never strands an attempt's data — every
    attempt the harness ran reaches the JSONL ledger regardless of the
    abort fate of the campaign.
    """
    if budget_millicents > 0 and cumulative_millicents >= budget_millicents:
        raise BudgetExceeded(
            f"Campaign aborted: cumulative counted spend "
            f"{cumulative_millicents} millicents "
            f"(≈${cumulative_millicents / 100_000:.4f}) reached budget "
            f"{budget_millicents} millicents "
            f"(≈${budget_millicents / 100_000:.2f}) "
            f"after attempt {attempt_index} on {site_category}",
        )


def _write_outcome_jsonl(path: Path, outcome: AttemptOutcome) -> None:
    """Append one outcome record to the JSONL ledger.

    Each record is a single JSON line so the file is streamable and
    durable across mid-campaign aborts.
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(outcome.model_dump_json() + "\n")


def _site_agent_id(site: SiteConfig) -> str:
    """Synthesize a per-site agent id used by ``BrowserManager``.

    Embedding the category + a short uuid suffix means a re-run of the same
    config doesn't collide with leftover ledger state from a prior run.
    """
    short = uuid.uuid4().hex[:8]
    return f"validation-{site.category}-{short}"


async def run_campaign(
    config: CampaignConfig,
    output_dir: Path,
    *,
    timeout_s: float = ATTEMPT_TIMEOUT_S,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    browser_manager_factory=None,
    write_report: bool = True,
) -> CampaignReport:
    """Run the campaign end-to-end.

    Args:
        config: validated :class:`CampaignConfig`.
        output_dir: directory for ``attempts.jsonl`` + the markdown report.
            Created if missing.
        timeout_s: per-attempt hard timeout (asyncio.wait_for).
        pace_seconds: sleep between attempts on the same site. Tests pass 0.
        browser_manager_factory: zero-arg async callable that returns a
            ready ``BrowserManager``. Defaults to a function that imports
            and instantiates :class:`BrowserManager` directly. Tests inject
            a mock manager to avoid spinning real Playwright.
        write_report: when True (default) generate the markdown report at
            campaign end. Tests can pass False to inspect ``CampaignReport``
            without disk-writing the report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "attempts.jsonl"

    if browser_manager_factory is None:
        async def _default_factory() -> "BrowserManager":
            from src.browser.service import BrowserManager
            mgr = BrowserManager(profiles_dir=str(output_dir / "profiles"))
            return mgr
        browser_manager_factory = _default_factory

    mgr = await browser_manager_factory()

    cost_budget_millicents = int(round(config.cost_budget_usd * 100_000))
    attempts: list[AttemptOutcome] = []
    aborted_sites: list[str] = []
    budget_exceeded = False
    cumulative_counted_millicents = 0

    try:
        for site in config.sites:
            agent_id = _site_agent_id(site)
            await _ensure_fresh_instance(mgr, agent_id)

            burn_streak = 0
            attempts_today = 0

            for attempt_index in range(site.attempt_count):
                # Pacing: sleep between attempts to avoid hammering the
                # site within a single second. Tests pass 0 to skip.
                if attempt_index > 0 and pace_seconds > 0:
                    await asyncio.sleep(pace_seconds)

                # ``attempts_per_day`` is the operator's stated comfort
                # level — when we'd cross it we abort the SITE rather
                # than block the campaign on a long sleep. The runbook
                # tells operators to schedule a follow-up run on the
                # next day to pick up the remaining attempts.
                if attempts_today >= config.attempts_per_day:
                    logger.info(
                        "Daily quota %d reached for site=%s; "
                        "skipping remaining %d attempt(s)",
                        config.attempts_per_day,
                        site.category,
                        site.attempt_count - attempt_index,
                    )
                    break

                outcome, cumulative_counted_millicents = await _run_one_attempt(
                    mgr,
                    agent_id,
                    site,
                    attempt_index,
                    timeout_s=timeout_s,
                    cumulative_counted_millicents_before=cumulative_counted_millicents,
                )
                attempts.append(outcome)
                _write_outcome_jsonl(jsonl_path, outcome)
                attempts_today += 1

                # Budget gate runs AFTER the outcome is persisted so a
                # campaign that hits the cap mid-flight still has every
                # attempt it actually ran in the ledger. The exception
                # exits the outer loop via the surrounding try/except.
                try:
                    _budget_check(
                        cumulative_millicents=cumulative_counted_millicents,
                        budget_millicents=cost_budget_millicents,
                        site_category=site.category,
                        attempt_index=attempt_index,
                    )
                except BudgetExceeded as bx:
                    logger.warning("%s", bx)
                    budget_exceeded = True
                    raise

                if outcome.fingerprint_burn_signal:
                    burn_streak += 1
                else:
                    burn_streak = 0
                if burn_streak >= BURN_STREAK_THRESHOLD:
                    logger.warning(
                        "Fingerprint burn detected on category=%s after %d "
                        "consecutive burn-shaped outcomes; skipping "
                        "remaining %d attempt(s)",
                        site.category, burn_streak,
                        site.attempt_count - (attempt_index + 1),
                    )
                    aborted_sites.append(site.url)
                    break

            await _ensure_fresh_instance(mgr, agent_id)
    except BudgetExceeded:
        # Caller-visible as a flag on CampaignReport; not re-raised.
        pass
    finally:
        with contextlib.suppress(Exception):
            await mgr.stop_all()

    report_path: Path | None = None
    if write_report:
        # Local import — avoids circular ``runner ↔ report`` imports during
        # type-stub evaluation and keeps ``--dry-run`` strictly free of any
        # report-writer dependencies.
        from tools.captcha_validation.report import generate_report
        report_path = output_dir / (
            f"validation-report-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
        )
        generate_report(attempts, report_path)

    return CampaignReport(
        attempts=attempts,
        aborted_sites=aborted_sites,
        budget_exceeded=budget_exceeded,
        output_dir=output_dir,
        report_path=report_path,
    )


# ── CLI entry point ────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.captcha_validation.runner",
        description=(
            "Phase 9 §18.1 captcha-validation campaign runner. Drives the "
            "production browser-service codepath against operator-curated "
            "target sites and produces a §11.20-promotion-grade evidence "
            "report.\n\n"
            "Reads a YAML config (see config.example.yaml). NEVER auto-runs "
            "against live sites at import time; --dry-run loads + validates "
            "the config without making any HTTP calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the campaign YAML config (e.g. config.local.yaml).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/captcha_validation"),
        help=(
            "Directory for attempts.jsonl + the markdown report. "
            "Created if missing. Default: data/captcha_validation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load + validate the config and print a summary; do not "
            "instantiate a BrowserManager and do not make any HTTP "
            "calls. Use this to verify config.local.yaml before a real run."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=ATTEMPT_TIMEOUT_S,
        help=f"Per-attempt hard timeout in seconds (default: {ATTEMPT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=(
            f"Sleep between attempts on the same site (default: "
            f"{DEFAULT_PACE_SECONDS}). Operators MAY raise this to spread "
            "attempts further within a session."
        ),
    )
    return parser


def _print_dry_run_summary(config: CampaignConfig) -> None:
    """Print a human-readable summary of the loaded config.

    Crucially: NEVER prints the full URLs (only host + path-prefix) and
    NEVER prints creds-refs. Operator can sanity-check categories +
    attempt counts before a real run.
    """
    print(f"Loaded {len(config.sites)} site(s)")
    print(f"Cost budget: ${config.cost_budget_usd:.2f}")
    print(f"Attempts per day per site: {config.attempts_per_day}")
    print()
    print("Sites:")
    for site in config.sites:
        # Print only the netloc + first path segment so casual scrollback
        # never reveals OAuth callback paths or test-account user-handles.
        from urllib.parse import urlsplit
        parts = urlsplit(site.url)
        loc = parts.netloc or "(unknown)"
        head = parts.path.split("/", 2)
        first_seg = "/" + head[1] if len(head) > 1 and head[1] else ""
        print(
            f"  - [{site.category:>20}] {loc}{first_seg}/...  "
            f"expected_kind={site.expected_kind} "
            f"attempts={site.attempt_count} "
            f"interaction={site.interaction}",
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("OPENLEGION_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        config = load_campaign(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry-run: loaded {args.config}; no HTTP calls will be made.")
        _print_dry_run_summary(config)
        return 0

    asyncio.run(
        run_campaign(
            config,
            args.output_dir,
            timeout_s=args.timeout_seconds,
            pace_seconds=args.pace_seconds,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
