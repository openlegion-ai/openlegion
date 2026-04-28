"""Typed contracts for the Phase 9 §18.1 validation harness.

Three Pydantic models:

* :class:`SiteConfig` — one row of an operator-curated target list. The URL,
  expected §11.13 ``kind``, and how to provoke the captcha (navigate-only or
  navigate-then-click).
* :class:`CampaignConfig` — top-level YAML shape: a list of sites plus
  campaign-wide caps (cost budget, attempts-per-day pacing, optional
  solver-proxy creds reference).
* :class:`AttemptOutcome` — one row of the JSONL outcome ledger. The §11.13
  envelope is preserved verbatim; cost and timing are reconciled at the
  harness layer.

The schema deliberately treats credentials as opaque references — the YAML
contains *names* (``vault://test_google_account``) that the operator
resolves out-of-band, never raw secrets. Defends against accidental commits
of ``config.local.yaml`` containing real tokens.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# §11.13 envelope ``kind`` enum. Must stay in sync with
# ``src/browser/captcha.py:_VALID_CAPTCHA_KINDS`` plus the broader list of
# kinds the §18.1 campaign deliberately exercises (FunCaptcha, GeeTest, AWS
# WAF, DataDome, PerimeterX) — those are §11.20 deferred items that the
# campaign is supposed to surface as ``unsupported`` outcomes for evidence.
# Validating expected_kind against this set catches typos early; the runner
# still records whatever the production envelope emits, even if the live
# kind isn't in this set.
KNOWN_CAPTCHA_KINDS: frozenset[str] = frozenset({
    # Currently supported (Phase 8 merged)
    "recaptcha-v2-checkbox",
    "recaptcha-v2-invisible",
    "recaptcha-v3",
    "recaptcha-enterprise-v2",
    "recaptcha-enterprise-v3",
    "hcaptcha",
    "turnstile",
    "cf-interstitial-turnstile",
    "cf-interstitial-auto",
    "cf-interstitial-behavioral",
    # §11.20 deferred — campaign expects ``unsupported`` outcome here
    "funcaptcha",
    "geetest",
    "geetest-v3",
    "geetest-v4",
    "aws-waf",
    "datadome-behavioral",
    "px-press-hold",
    # Catchall when the classifier hasn't picked a variant
    "unknown",
})


# 10 operator-facing categories from the plan §18.1 site list. Used for
# report grouping and to enforce that the campaign covers each category at
# most once (the spec calls for one URL per category, more attempts in any
# category that turns out flaky).
KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "google_signin",
    "twitter_signup",
    "linkedin_auth",
    "cloudflare_saas",
    "aws_waf",
    "geetest",
    "invisible_v2",
    "hcaptcha_saas",
    "human_perimeterx",
    "datadome",
    # Operator escape hatch for one-off targets that don't fit a category.
    "other",
})


class SiteConfig(BaseModel):
    """One operator-curated target site."""

    url: str = Field(
        ...,
        description="Target URL. Operator-curated; never logged un-redacted in the report.",
    )
    expected_kind: str = Field(
        ...,
        description="§11.13 envelope kind enum the operator expects this site to emit.",
    )
    attempt_count: int = Field(
        10,
        ge=1,
        le=100,
        description="How many solve attempts to make against this site in the campaign.",
    )
    interaction: Literal["navigate_only", "navigate_then_click_submit"] = Field(
        "navigate_only",
        description=(
            "How to provoke the captcha. ``navigate_only`` = the captcha "
            "fires on page load. ``navigate_then_click_submit`` = navigate, "
            "then click ``interaction_selector`` to surface the captcha."
        ),
    )
    interaction_selector: str | None = Field(
        None,
        description=(
            "CSS selector for the click in ``navigate_then_click_submit``. "
            "Required iff interaction == 'navigate_then_click_submit'."
        ),
    )
    account_creds_ref: str | None = Field(
        None,
        description=(
            "Opaque reference to a vault entry (e.g. ``vault://test_google``). "
            "The harness NEVER reads creds from the YAML — operator resolves "
            "the reference out-of-band before running. Stored only to "
            "annotate the report with which test account was used."
        ),
    )
    category: str = Field(
        ...,
        description=(
            "One of the 10 §18.1 categories (e.g. ``google_signin``, "
            "``twitter_signup``). See KNOWN_CATEGORIES."
        ),
    )
    notes: str | None = Field(
        None,
        description="Operator-facing free-form notes; appears in the report.",
    )

    @field_validator("expected_kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        if v not in KNOWN_CAPTCHA_KINDS:
            raise ValueError(
                f"expected_kind={v!r} is not a known §11.13 kind. "
                f"Known: {sorted(KNOWN_CAPTCHA_KINDS)}",
            )
        return v

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if v not in KNOWN_CATEGORIES:
            raise ValueError(
                f"category={v!r} is not a known §18.1 category. "
                f"Known: {sorted(KNOWN_CATEGORIES)}",
            )
        return v

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must be non-empty")
        # Reject obvious placeholder URLs that operators forgot to fill in.
        # These are the literal strings used in ``config.example.yaml``.
        if v.endswith("/...") or v.startswith("vault://"):
            raise ValueError(
                f"url={v!r} looks like an unfilled placeholder — "
                f"replace with a real target URL before running",
            )
        return v


class CampaignConfig(BaseModel):
    """Top-level YAML shape — one campaign run."""

    sites: list[SiteConfig] = Field(
        ...,
        description="The curated target list. One row per site.",
    )
    solver_proxy_creds_ref: str | None = Field(
        None,
        description=(
            "Opaque reference to a vault entry holding the dedicated "
            "solver-proxy creds (NOT the agent's primary egress proxy — "
            "see §11.2 security note). The harness expects the operator to "
            "have already exported these as ``CAPTCHA_SOLVER_PROXY_*`` env "
            "vars before invoking the runner; this field is documentary."
        ),
    )
    cost_budget_usd: float = Field(
        5.0,
        gt=0,
        le=100.0,
        description=(
            "Hard ceiling on the campaign's solver spend in USD. The runner "
            "aborts the entire campaign once the cumulative counted spend "
            "(via ``captcha_cost_counter``) crosses this threshold."
        ),
    )
    attempts_per_day: int = Field(
        5,
        ge=1,
        le=100,
        description=(
            "Per-site pacing — how many attempts to make on a given calendar "
            "day for any single site. The runner spreads ``attempt_count`` "
            "across multiple days when ``attempt_count > attempts_per_day`` "
            "by sleeping until the next day before resuming. In practice "
            "operators run two short same-day sessions."
        ),
    )

    @field_validator("sites")
    @classmethod
    def _validate_non_empty(cls, v: list[SiteConfig]) -> list[SiteConfig]:
        if not v:
            raise ValueError("sites must contain at least one entry")
        return v


class AttemptOutcome(BaseModel):
    """One row of the per-attempt outcome ledger.

    Persisted as JSONL under ``output_dir/attempts.jsonl``. The §11.13
    envelope is captured verbatim under ``envelope`` so the report writer
    can group outcomes without re-deriving anything; cost reconciliation
    fields live alongside.
    """

    site_url: str = Field(
        ..., description="Source URL of the attempt (will be redacted in the report).",
    )
    attempt_index: int = Field(
        ..., ge=0, description="0-based index of this attempt within its site.",
    )
    wall_clock_ms: int = Field(
        ..., ge=0, description="Wall-clock ms from detection start to envelope return.",
    )
    envelope: dict = Field(
        ...,
        description=(
            "§11.13 envelope verbatim. Common keys: ``kind``, "
            "``solver_outcome``, ``solver_confidence``, ``next_action``, "
            "``captcha_found``."
        ),
    )
    cost_charged_cents_estimated: int = Field(
        0,
        ge=0,
        description=(
            "Estimated provider charge in cents (=millicents/1000) computed "
            "from ``estimate_millicents(provider, kind, proxy_aware=...)``. "
            "Zero when the variant has no published rate or when the solve "
            "short-circuited before any provider charge."
        ),
    )
    cost_counted_millicents: int = Field(
        0,
        ge=0,
        description=(
            "Per-attempt delta from ``captcha_cost_counter.get_millicents`` "
            "(after - before). Zero when no cost was counted."
        ),
    )
    fingerprint_burn_signal: bool = Field(
        False,
        description=(
            "True when the envelope's ``solver_outcome`` indicates the "
            "site has flagged the fingerprint (token rejected, repeated "
            "challenge after solve). Five consecutive True values for one "
            "site triggers the runner's per-site abort."
        ),
    )
    classifier_match: bool = Field(
        ...,
        description="True iff envelope.kind == site.expected_kind.",
    )
    error: str | None = Field(
        None,
        description=(
            "Harness-side exception summary (e.g. timeout reaching the "
            "browser, invalid_input from solve_captcha). Distinct from "
            "envelope-level errors — those live inside ``envelope``."
        ),
    )
    timestamp_utc: str = Field(
        ...,
        description="ISO-8601 UTC timestamp at attempt start.",
    )
