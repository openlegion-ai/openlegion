# Phase 9 §18.1 — Captcha validation harness (operator runbook)

Reproducible tooling to drive the production browser-service codepath
end-to-end against operator-curated protected sites and capture structured
outcome evidence. Drives §11.20 promotion decisions: which deferred items
should ship, which stay deferred, and what trigger to attach to each.

The harness is **infrastructure**. It ships ready-to-use; live runs require
an operator-provided target URL list + solver creds + test accounts.

## Pre-requisites

Before a real run, the operator must have:

1. **Solver provider account** with a funded balance — either
   [2captcha](https://2captcha.com) or [CapSolver](https://capsolver.com).
   The harness reads the same provider config the production browser
   service uses (env vars consumed by `src/browser/captcha.py:get_solver`).

2. **Solver-side proxy credentials** — a *separate* proxy from the agent's
   primary egress proxy. See plan §11.2: handing your scraping-proxy creds
   to a third-party solver is a credential-leak vector. Set ALL FIVE:

   ```
   export CAPTCHA_SOLVER_PROXY_TYPE=socks5
   export CAPTCHA_SOLVER_PROXY_ADDRESS=...
   export CAPTCHA_SOLVER_PROXY_PORT=...
   export CAPTCHA_SOLVER_PROXY_LOGIN=...
   export CAPTCHA_SOLVER_PROXY_PASSWORD=...
   ```

   The harness verifies the env-var family is configured at startup; partial
   config falls back to proxyless (the captcha module logs a once-per-session
   warning).

3. **Test accounts** for every site whose category is a signin / signup
   target (`google_signin`, `twitter_signup`, `linkedin_auth`,
   `aws_waf`). Don't use production accounts — the campaign deliberately
   stresses fingerprint detection.

4. **Curated target URLs** for each of the 10 categories from plan §18.1.
   The example config (`config.example.yaml`) lists categories with
   placeholder URLs (`https://accounts.google.com/...`); the schema
   validator REJECTS placeholder URLs so a copy-paste-and-run can never
   accidentally exercise the example file.

## Quick start

```bash
# 1. Copy the example to a local config (gitignored).
cp tools/captcha_validation/config.example.yaml config.local.yaml

# 2. Edit config.local.yaml — fill in real URLs + creds-refs per site.

# 3. Export solver-proxy env vars from your vault.
source ~/.openlegion/captcha_solver_proxy.env

# 4. Validate the config without making HTTP calls.
python -m tools.captcha_validation.runner config.local.yaml --dry-run

# 5. Run the full campaign.
python -m tools.captcha_validation.runner config.local.yaml \
       data/captcha_validation
```

Outputs land in `data/captcha_validation/`:

* `attempts.jsonl` — per-attempt outcome ledger (one record per line).
* `validation-report-YYYY-MM-DD.md` — the markdown report described in
  plan §18.1.
* `profiles/` — one Camoufox profile directory per agent-id, retained for
  forensics. Safe to delete between runs.

## Interpreting the report

The report contains five sections:

* **Per-site outcome distribution** — for each site, the count of each
  `solver_outcome` value (solved / rejected / unsupported / timeout /
  cost_cap / rate_limited / skipped_behavioral / no_solver / harness_*).
  Wide variance across sites of the same category is the operator's first
  signal that fingerprint hygiene needs work.

* **Top 5 unsupported / rejected examples** — URL-redacted (via
  `src.shared.redaction.redact_url`). Useful for pasting into §11.20
  trigger-evidence lines.

* **Classifier accuracy** — % of attempts where the envelope's `kind`
  matched the configured `expected_kind`. Sub-80% on a category is a
  §11.6 (sitekey extraction hardening) promotion signal.

* **Cost reconciliation** — counted (from `captcha_cost_counter`) vs
  estimated (from `estimate_millicents`). A delta exceeding 10% is
  flagged ⚠ and means the cost counter is missing increments — file a
  §11.10 follow-up.

* **§11.20 promotion recommendations** — concrete `recommend promote /
  keep deferred / investigate` lines per deferred item, derived from the
  campaign data. Carry these directly into your §11.20 trigger update.

## Cost expectations

A full 10-site × 10-attempt campaign at default solver rates costs
**roughly $1–$5** depending on which kinds the operator targets:

* recaptcha-v2 / hcaptcha: ~$0.10 per 100 attempts (proxyless)
* recaptcha-enterprise / turnstile: ~$0.20 per 100 attempts (proxyless)
* proxy-aware tasks: ~3× the proxyless rate

The campaign's `cost_budget_usd` (default `$5.00`) is a HARD ceiling. The
runner aborts the campaign once the cumulative *counted* spend (read from
`captcha_cost_counter`) reaches the budget. This is checked AFTER every
attempt — it is not possible for the harness to overshoot the budget by
more than one solve's published rate.

## Fingerprint hygiene

The campaign deliberately hammers protected sites; doing so back-to-back
on a single fingerprint corrupts the per-site outcome distributions. The
harness mitigates this two ways:

* **Per-site fresh profile.** Between sites the harness calls
  `BrowserManager.stop(agent_id)` so the next site opens with a clean
  Camoufox instance. The profile lives under `output_dir/profiles/`; the
  next site uses a new agent-id (UUID-suffixed) and therefore a separate
  profile dir.

* **Per-site burn detection.** If 5 consecutive attempts on a single site
  return a burn-shaped outcome (`rejected`, `injection_failed`,
  `captcha_during_solve`), the runner skips the rest of that site's
  attempts and continues with the next site. The burned site URL is
  recorded in `CampaignReport.aborted_sites`.

* **Daily quota.** `attempts_per_day` (default 5) caps how many attempts
  the harness will make against any one site within a single same-day
  session. To complete a 10-attempt site in one calendar day, schedule
  two runs separated by a few hours (the harness preserves `attempts.jsonl`
  across runs — the report aggregates the lot).

If a site burns mid-campaign and you want to retry, change the agent-id
prefix (it's UUID-suffixed automatically) by re-running the campaign
against just that one site in a fresh `output_dir`. There's no surgical
"retry one site" flag — by design, the harness pushes operators toward
fresh fingerprints rather than partial retries.

## Legal note

Some sites' Terms of Service forbid automated access. Verifying that the
operator's chosen target list is acceptable for *their* deployment is
**the operator's responsibility**. The harness does not maintain an
allowlist of "OK to test" sites. When in doubt, target self-owned staging
infrastructure or vendor sandbox environments (most major captcha
providers offer test pages — those are always safe to target).

## Self-tests

```bash
pytest tests/test_captcha_validation_harness.py -v
```

The self-tests use mock Camoufox + a local `http.server` serving the
fixture pages under `fixtures/`. No real solver provider is contacted; no
real network captcha is encountered. CI can run these even without the
optional `playwright` dependency installed.
