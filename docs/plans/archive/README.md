# Plans Archive

This folder holds **completed/shipped or superseded design docs**, kept for their
historical rationale. They are no longer active work, but the "why" behind decisions
that are now live in the codebase is often worth re-reading. (Git has the full history
for everything here — nothing was deleted, just moved.)

**Active plans live one level up, in [`docs/plans/`](../).**

## Status-header convention

Every plan doc starts with a one-line `**Status:**` marker near its header:

- `**Status:** SHIPPED` — implemented and live (these are the ones archived here).
- `**Status:** ACTIVE` — accepted/in-progress direction (stays in `docs/plans/`).
- `**Status:** PROPOSED` — drafted for review, not yet accepted (stays in `docs/plans/`).
- `**Status:** SUPERSEDED` — replaced by a later plan (archived, with a pointer where useful).

When a plan ships, add/normalize its `**Status:**` line to `SHIPPED` and `git mv` it here.
