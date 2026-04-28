"""Phase 9 §18.1 captcha-validation harness.

Operator-driven tooling that drives the production browser-service codepath
end-to-end against curated protected sites and captures structured outcome
records. The harness is *infrastructure* — the live runs that drive §11.20
promotion decisions are kicked off by an operator with their own target list
and solver creds. NEVER auto-runs against live sites at import time.

See ``tools/captcha_validation/README.md`` for the operator runbook.
"""
