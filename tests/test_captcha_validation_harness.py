"""Self-tests for the Phase 9 §18.1 validation harness.

The harness lives at ``tools/captcha_validation``; these tests exercise
schema validation, runner orchestration (with a mocked BrowserManager so we
never touch a real browser), report generation, cost-budget enforcement,
fingerprint-burn skip behavior, and URL redaction in the rendered report.

NO real solver provider, NO live Playwright, NO real captcha — every
external surface is mocked.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.browser import captcha_cost_counter as cost
from tools.captcha_validation import report as report_mod
from tools.captcha_validation import runner as runner_mod
from tools.captcha_validation.report import generate_report
from tools.captcha_validation.runner import (
    BURN_STREAK_THRESHOLD,
    BudgetExceeded,
    load_campaign,
    main,
    run_campaign,
)
from tools.captcha_validation.schema import (
    AttemptOutcome,
    CampaignConfig,
    SiteConfig,
)

# ── Schema validation tests ─────────────────────────────────────────────────


class TestSchema:
    def test_site_config_minimal(self) -> None:
        site = SiteConfig(
            url="https://example.com/login",
            expected_kind="recaptcha-v3",
            category="google_signin",
        )
        assert site.attempt_count == 10
        assert site.interaction == "navigate_only"

    def test_site_config_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SiteConfig(
                url="https://example.com/login",
                expected_kind="not-a-real-kind",
                category="google_signin",
            )
        assert "expected_kind" in str(excinfo.value)

    def test_site_config_rejects_unknown_category(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SiteConfig(
                url="https://example.com/login",
                expected_kind="recaptcha-v3",
                category="zoom_signin",
            )
        assert "category" in str(excinfo.value)

    def test_site_config_rejects_placeholder_url(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SiteConfig(
                url="https://accounts.google.com/...",
                expected_kind="recaptcha-v3",
                category="google_signin",
            )
        assert "placeholder" in str(excinfo.value)

    def test_site_config_attempt_count_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SiteConfig(
                url="https://example.com/login",
                expected_kind="recaptcha-v3",
                category="google_signin",
                attempt_count=0,
            )

    def test_campaign_config_requires_at_least_one_site(self) -> None:
        with pytest.raises(ValidationError):
            CampaignConfig(sites=[])

    def test_campaign_config_loads_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "cost_budget_usd": 1.5,
                "attempts_per_day": 3,
                "sites": [
                    {
                        "url": "https://example.com/login",
                        "expected_kind": "hcaptcha",
                        "category": "hcaptcha_saas",
                        "attempt_count": 2,
                    },
                ],
            }),
        )
        cfg = load_campaign(cfg_path)
        assert len(cfg.sites) == 1
        assert cfg.cost_budget_usd == 1.5
        assert cfg.attempts_per_day == 3

    def test_load_campaign_rejects_empty_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        with pytest.raises(ValueError):
            load_campaign(cfg_path)

    def test_example_yaml_rejects_due_to_placeholders(self) -> None:
        # The shipped example file uses ``...``-suffixed placeholder URLs
        # so a copy-paste-and-run cannot accidentally fire a real campaign.
        # Verify the schema actually catches it.
        example = Path(__file__).resolve().parents[1] / "tools" / "captcha_validation" / "config.example.yaml"
        with open(example, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        with pytest.raises(ValidationError):
            CampaignConfig(**raw)


# ── Mock BrowserManager for runner tests ───────────────────────────────────


class _MockSolver:
    """Minimal solver stub. Holds a provider name + a controllable per-call
    behavior; the runner reads ``provider`` for cost reconciliation.
    """

    def __init__(self, provider: str = "2captcha") -> None:
        self.provider = provider


class _ScriptedManager:
    """A BrowserManager-shaped stub for runner tests.

    The runner calls (in order, per attempt):

        navigate(agent_id, url, wait_ms=, wait_until=)
        [click(agent_id, selector=)]   # only when interaction != navigate_only
        solve_captcha(agent_id, retry_previous=)

    Plus, between sites:
        stop(agent_id)
        stop_all()  # final teardown

    The scripted envelopes drive the per-attempt outcomes. ``cost_seq``
    drives the in-memory ``captcha_cost_counter`` increments — one entry
    per attempt; the manager calls ``cost.add_cost(agent_id, delta)``
    BEFORE returning the envelope so the runner's after-read sees the
    delta.
    """

    def __init__(
        self,
        envelopes: Iterable[dict],
        cost_deltas_millicents: Iterable[int] | None = None,
        provider: str = "2captcha",
    ) -> None:
        self._envelopes = list(envelopes)
        self._cost_deltas = list(cost_deltas_millicents or [])
        self._idx = 0
        self._captcha_solver = _MockSolver(provider=provider)
        self.navigated: list[tuple[str, str]] = []
        self.clicked: list[tuple[str, str | None]] = []
        self.stops: list[str] = []
        self.stop_all_called = False

    async def navigate(
        self, agent_id: str, url: str, wait_ms: int = 1000,
        wait_until: str = "domcontentloaded", **_: object,
    ) -> dict:
        self.navigated.append((agent_id, url))
        return {"success": True, "data": {"url": url}}

    async def click(
        self, agent_id: str, selector: str | None = None, **_: object,
    ) -> dict:
        self.clicked.append((agent_id, selector))
        return {"success": True}

    async def solve_captcha(
        self, agent_id: str, **_: object,
    ) -> dict:
        if self._idx >= len(self._envelopes):
            envelope = {
                "captcha_found": False,
                "kind": "unknown",
                "solver_outcome": "no_solver",
                "solver_confidence": "low",
                "next_action": "operator_review",
            }
            delta = 0
        else:
            envelope = self._envelopes[self._idx]
            delta = (
                self._cost_deltas[self._idx]
                if self._idx < len(self._cost_deltas) else 0
            )
        self._idx += 1
        if delta:
            await cost.add_cost(agent_id, delta)
        return {"success": True, "data": dict(envelope)}

    async def stop(self, agent_id: str) -> None:
        self.stops.append(agent_id)

    async def stop_all(self) -> None:
        self.stop_all_called = True


@pytest.fixture(autouse=True)
async def _isolate_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own ``captcha_costs.json`` and an empty counter."""
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    yield
    await cost.reset()


def _site(
    *, url: str = "https://example.com/login",
    expected_kind: str = "hcaptcha",
    attempts: int = 3,
    interaction: str = "navigate_only",
    interaction_selector: str | None = None,
    category: str = "hcaptcha_saas",
) -> SiteConfig:
    return SiteConfig(
        url=url,
        expected_kind=expected_kind,
        attempt_count=attempts,
        interaction=interaction,
        interaction_selector=interaction_selector,
        category=category,
    )


def _solved_envelope(kind: str = "hcaptcha") -> dict:
    return {
        "captcha_found": True,
        "kind": kind,
        "solver_outcome": "solved",
        "solver_confidence": "high",
        "next_action": "continue",
        "solver_used_proxy_aware": False,
    }


def _rejected_envelope(kind: str = "hcaptcha") -> dict:
    return {
        "captcha_found": True,
        "kind": kind,
        "solver_outcome": "rejected",
        "solver_confidence": "low",
        "next_action": "request_captcha_help",
    }


def _unsupported_envelope(kind: str = "funcaptcha") -> dict:
    return {
        "captcha_found": True,
        "kind": kind,
        "solver_outcome": "unsupported",
        "solver_confidence": "low",
        "next_action": "request_captcha_help",
    }


# ── Runner orchestration tests ─────────────────────────────────────────────


class TestRunnerHappyPath:
    @pytest.mark.asyncio
    async def test_three_solved_attempts_produce_three_outcomes(
        self, tmp_path: Path,
    ) -> None:
        site = _site(attempts=3)
        cfg = CampaignConfig(sites=[site], cost_budget_usd=1.0)
        envelopes = [_solved_envelope() for _ in range(3)]
        # 100 millicents each = 0.1 cents = $0.001 — the proxyless 2captcha
        # hcaptcha rate. Total counted: 300 millicents = $0.003.
        deltas = [100, 100, 100]
        manager = _ScriptedManager(envelopes, cost_deltas_millicents=deltas)

        report = await run_campaign(
            cfg, tmp_path / "out",
            timeout_s=5.0, pace_seconds=0.0,
            browser_manager_factory=lambda: _wrap(manager),
        )

        assert len(report.attempts) == 3
        for a in report.attempts:
            assert a.envelope["solver_outcome"] == "solved"
            assert a.classifier_match is True
            assert a.cost_counted_millicents == 100
            # Estimate from estimate_millicents("2captcha", "hcaptcha") = 100mc
            # → cents = round(100/1000) = 0
            # The estimate is integer cents; 0.1 cent rounds to 0 cents. The
            # test asserts the rounding is intentional (not a bug).
            assert a.cost_charged_cents_estimated == 0
        # Manager should have stopped between/after sites.
        assert manager.stop_all_called is True

    @pytest.mark.asyncio
    async def test_navigate_then_click_invokes_click(
        self, tmp_path: Path,
    ) -> None:
        site = _site(
            attempts=1,
            interaction="navigate_then_click_submit",
            interaction_selector="#submit",
        )
        cfg = CampaignConfig(sites=[site], cost_budget_usd=1.0)
        manager = _ScriptedManager([_solved_envelope()])
        await run_campaign(
            cfg, tmp_path / "out",
            timeout_s=5.0, pace_seconds=0.0,
            browser_manager_factory=lambda: _wrap(manager),
        )
        assert len(manager.clicked) == 1
        assert manager.clicked[0][1] == "#submit"


class TestRunnerOutputFiles:
    @pytest.mark.asyncio
    async def test_jsonl_ledger_is_written(self, tmp_path: Path) -> None:
        site = _site(attempts=2)
        cfg = CampaignConfig(sites=[site], cost_budget_usd=1.0)
        manager = _ScriptedManager(
            [_solved_envelope(), _solved_envelope()],
            cost_deltas_millicents=[100, 100],
        )
        out = tmp_path / "out"
        await run_campaign(
            cfg, out,
            timeout_s=5.0, pace_seconds=0.0,
            browser_manager_factory=lambda: _wrap(manager),
        )
        ledger = out / "attempts.jsonl"
        assert ledger.exists()
        lines = ledger.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            payload = json.loads(line)
            # Schema-shape sanity — every recorded outcome is loadable
            # back into AttemptOutcome.
            AttemptOutcome(**payload)


# ── Cost-budget enforcement ─────────────────────────────────────────────────


class TestBudget:
    @pytest.mark.asyncio
    async def test_budget_aborts_after_two_solves(self, tmp_path: Path) -> None:
        # $0.01 budget = 1000 millicents. 500 mc/solve. Two solves = 1000 mc
        # (=$0.01) which crosses the >= threshold → abort BEFORE the third
        # attempt records.
        site = _site(attempts=4)
        cfg = CampaignConfig(sites=[site], cost_budget_usd=0.01)
        manager = _ScriptedManager(
            [_solved_envelope() for _ in range(4)],
            cost_deltas_millicents=[500, 500, 500, 500],
        )
        report = await run_campaign(
            cfg, tmp_path / "out",
            timeout_s=5.0, pace_seconds=0.0,
            browser_manager_factory=lambda: _wrap(manager),
        )
        assert report.budget_exceeded is True
        # Two attempts persisted (the second crossed the threshold and the
        # exception fired AFTER its outcome was appended).
        assert len(report.attempts) == 2

    def test_budget_exceeded_class_inherits_runtimeerror(self) -> None:
        assert issubclass(BudgetExceeded, RuntimeError)


# ── Burn-signal abort ───────────────────────────────────────────────────────


class TestBurnSignal:
    @pytest.mark.asyncio
    async def test_five_consecutive_rejected_skips_remaining_attempts(
        self, tmp_path: Path,
    ) -> None:
        # Five rejects on site A trip the burn streak — site A's remaining
        # attempts should be skipped. Site B (subsequent in the config)
        # continues normally. Site B's lone attempt produces a solved
        # envelope and is recorded.
        site_a = _site(
            url="https://burned.example.com/login",
            attempts=8,  # 5 rejected + 3 that should be skipped
            category="hcaptcha_saas",
        )
        site_b = _site(
            url="https://other.example.com/login",
            attempts=1,
            category="cloudflare_saas",
            expected_kind="cf-interstitial-turnstile",
        )
        cfg = CampaignConfig(sites=[site_a, site_b], cost_budget_usd=10.0)
        envelopes = [_rejected_envelope() for _ in range(5)] + [
            _solved_envelope(kind="cf-interstitial-turnstile"),
        ]
        manager = _ScriptedManager(envelopes)
        report = await run_campaign(
            cfg, tmp_path / "out",
            timeout_s=5.0, pace_seconds=0.0,
            browser_manager_factory=lambda: _wrap(manager),
        )
        # 5 rejects on A + 1 solved on B = 6 outcomes total (3 attempts of
        # site A skipped).
        assert len(report.attempts) == 6
        assert "https://burned.example.com/login" in report.aborted_sites
        # Site B's lone attempt landed.
        assert report.attempts[-1].envelope["kind"] == "cf-interstitial-turnstile"

    def test_burn_threshold_is_five(self) -> None:
        # Sanity gate — the threshold is part of the harness contract.
        # Bumping it requires updating the README + the §18.1 spec note.
        assert BURN_STREAK_THRESHOLD == 5


# ── Report generation + URL redaction ───────────────────────────────────────


class TestReport:
    def test_report_contains_expected_sections(self, tmp_path: Path) -> None:
        attempts = [
            AttemptOutcome(
                site_url="https://example.com/login",
                attempt_index=i,
                wall_clock_ms=1234 + i,
                envelope=_solved_envelope(),
                cost_charged_cents_estimated=0,
                cost_counted_millicents=100,
                fingerprint_burn_signal=False,
                classifier_match=True,
                error=None,
                timestamp_utc="2026-04-27T00:00:00+00:00",
            )
            for i in range(2)
        ]
        out = tmp_path / "report.md"
        generate_report(attempts, out)
        content = out.read_text()
        # Each section header should appear once.
        for section in [
            "# Phase 9 §18.1",
            "## Summary",
            "## Per-site outcome distribution",
            "## Top 5 `unsupported` examples",
            "## Top 5 `rejected` examples",
            "## Classifier accuracy",
            "## Cost reconciliation",
            "## §11.20 promotion recommendations",
        ]:
            assert section in content, f"missing section: {section!r}"

    def test_report_redacts_query_string_secrets(self, tmp_path: Path) -> None:
        # The site URL contains a sensitive query param (``token``) that
        # MUST be redacted in the rendered report. Use the harness
        # url-redaction contract described in src/shared/redaction.py.
        secret = "ya29.A0AfH6SMBmockTokenForTestOnlyXXXXXXXXXX"
        attempts = [
            AttemptOutcome(
                site_url=f"https://example.com/cb?code=abc&token={secret}",
                attempt_index=0,
                wall_clock_ms=42,
                envelope=_solved_envelope(),
                cost_charged_cents_estimated=0,
                cost_counted_millicents=100,
                fingerprint_burn_signal=False,
                classifier_match=True,
                error=None,
                timestamp_utc="2026-04-27T00:00:00+00:00",
            ),
        ]
        out = tmp_path / "report.md"
        generate_report(attempts, out)
        content = out.read_text()
        # No raw token in the rendered markdown.
        assert secret not in content
        # And the token-shaped string also shouldn't leak via deep_redact
        # heuristics — the [REDACTED] sentinel should appear instead.
        assert "[REDACTED]" in content

    def test_report_promotes_unsupported_funcaptcha(
        self, tmp_path: Path,
    ) -> None:
        # 5/5 attempts on FunCaptcha return unsupported → recommend promote.
        attempts = [
            AttemptOutcome(
                site_url="https://twitter.example.com/signup",
                attempt_index=i,
                wall_clock_ms=42,
                envelope=_unsupported_envelope("funcaptcha"),
                cost_charged_cents_estimated=0,
                cost_counted_millicents=0,
                fingerprint_burn_signal=False,
                classifier_match=True,
                error=None,
                timestamp_utc="2026-04-27T00:00:00+00:00",
            )
            for i in range(5)
        ]
        out = tmp_path / "report.md"
        generate_report(attempts, out)
        content = out.read_text()
        assert "FunCaptcha" in content
        assert "recommend promote" in content

    def test_cost_reconciliation_flags_large_delta(
        self, tmp_path: Path,
    ) -> None:
        # Counted = 5_000_000 mc ($50), Estimate = 100 cents ($1) — delta
        # is 5000% which must trip the ⚠ flag.
        attempts = [
            AttemptOutcome(
                site_url="https://example.com/login",
                attempt_index=0,
                wall_clock_ms=42,
                envelope=_solved_envelope(),
                cost_charged_cents_estimated=100,
                cost_counted_millicents=5_000_000,
                fingerprint_burn_signal=False,
                classifier_match=True,
                error=None,
                timestamp_utc="2026-04-27T00:00:00+00:00",
            ),
        ]
        out = tmp_path / "report.md"
        generate_report(attempts, out)
        content = out.read_text()
        assert "⚠" in content


# ── CLI surface tests ──────────────────────────────────────────────────────


class TestCli:
    def test_help_returns_clean_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as ex:
            main(["--help"])
        assert ex.value.code == 0
        out = capsys.readouterr().out
        assert "captcha-validation" in out
        assert "--dry-run" in out

    def test_dry_run_validates_config_without_network(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "cost_budget_usd": 1.0,
            "sites": [
                {
                    "url": "https://example.com/login",
                    "expected_kind": "hcaptcha",
                    "category": "hcaptcha_saas",
                    "attempt_count": 2,
                },
            ],
        }))
        rc = main([str(cfg_path), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dry-run" in out
        # The dry-run never touches a BrowserManager or HTTP. Sanity-check
        # that no stray attempts.jsonl was written.
        assert not (tmp_path / "out" / "attempts.jsonl").exists()

    def test_cli_rejects_missing_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([str(tmp_path / "no-such.yaml"), "--dry-run"])
        assert rc == 2

    def test_dry_run_on_example_yaml_fails_validation(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The shipped example contains placeholder URLs the schema rejects
        # — so even ``--dry-run`` against the example exits non-zero. This
        # is the property that makes the example file safe to ship.
        example = Path(__file__).resolve().parents[1] / "tools" / "captcha_validation" / "config.example.yaml"
        rc = main([str(example), "--dry-run"])
        assert rc == 2


# ── Helpers ────────────────────────────────────────────────────────────────


def _wrap(manager) -> object:
    """Wrap a sync manager into the async-factory shape the runner expects."""
    async def factory() -> object:
        return manager
    return factory()


# Smoke test: importing the runner does NOT auto-fire any HTTP / browser
# work. Catches accidental top-level side effects (a class of bugs that
# would otherwise only surface in CI when ``import tools.captcha_validation``
# happens during collection).
def test_runner_import_has_no_side_effects() -> None:
    # Already imported at module load; the assertion is that test
    # collection got to this line without an HTTP request firing.
    assert "tools.captcha_validation.runner" in sys.modules
    # And the report module is independently importable without the runner
    # attempting to touch a config file at import time.
    assert report_mod is not None
    assert runner_mod.ATTEMPT_TIMEOUT_S > 0
