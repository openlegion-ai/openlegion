"""Cost tracking and budget enforcement for LLM usage.

Intercepts at the CredentialVault layer — every LLM call already routes
through it, so this is a single integration point covering all agents.

Storage: SQLite (lightweight, no external services).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.shared.models import estimate_cost, get_model_cost  # noqa: F401 — re-export
from src.shared.sqlite_helpers import open_db
from src.shared.utils import setup_logging

logger = setup_logging("host.costs")

# Single source of truth for the per-agent budget defaults (B-pre #3):
# the dashboard's Settings page (`_SYSTEM_SETTINGS_DEFAULTS`) reads these
# same constants, so the advertised default and the enforced default can
# never diverge again (previously the file-absent code path silently
# enforced $10/day while the dashboard advertised $50).
DEFAULT_DAILY_BUDGET_USD = 50.0
DEFAULT_MONTHLY_BUDGET_USD = 200.0


def _default_budget() -> dict:
    """Read default budget from dashboard system settings, falling back to hardcoded defaults."""
    try:
        p = Path("config/settings.json")
        if p.exists():
            data = json.loads(p.read_text())
            return {
                "daily_usd": data.get("default_daily_budget", DEFAULT_DAILY_BUDGET_USD),
                "monthly_usd": data.get("default_monthly_budget", DEFAULT_MONTHLY_BUDGET_USD),
            }
    except (json.JSONDecodeError, OSError):
        pass
    return {"daily_usd": DEFAULT_DAILY_BUDGET_USD, "monthly_usd": DEFAULT_MONTHLY_BUDGET_USD}


class CostTracker:
    """Tracks token usage and cost per agent, enforces budgets."""

    def __init__(
        self,
        db_path: str = "data/costs.db",
        budgets_path: str | None = None,
    ):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = open_db(db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        # Per-agent budget overrides. Persisted to ``budgets_path`` so caps
        # raised/lowered by the operator survive a mesh restart (otherwise
        # they silently revert to the global default after a redeploy).
        #
        # When the caller doesn't pass an explicit ``budgets_path``, derive it
        # from ``db_path``: the production default (``data/costs.db``) maps to
        # the canonical ``config/agent_budgets.json``, while any other db path
        # (e.g. a test tmp dir) co-locates the budgets file alongside the db so
        # tests and tools never read/write the real ``config/`` file.
        if budgets_path is None:
            if db_path == "data/costs.db":
                budgets_path = "config/agent_budgets.json"
            else:
                budgets_path = str(Path(db_path).parent / "agent_budgets.json")
        self.budgets_path = Path(budgets_path)
        # Serializes the in-memory mutation + ``_save_budgets()`` so concurrent
        # ``set_budget`` / ``cleanup_agent`` calls can't clobber the file (one
        # writer's ``os.replace`` landing last would otherwise drop another's
        # change).
        self._budget_lock = threading.Lock()
        self.budgets: dict[str, dict[str, float]] = {}
        self._load_budgets()
        # Team-envelope source (the TeamStore) — wired post-construction by
        # the runtime via set_team_store(). None (e.g. bare tests) disables
        # the envelope layer; the per-agent budget still applies.
        self._team_store = None

    def set_team_store(self, team_store) -> None:
        """Wire the TeamStore so team envelopes can be resolved per call."""
        self._team_store = team_store

    def _load_budgets(self) -> None:
        """Load per-agent budget overrides from disk (tolerant of missing/corrupt)."""
        if not self.budgets_path.exists():
            return
        try:
            data = json.loads(self.budgets_path.read_text())
            if not isinstance(data, dict):
                logger.warning(
                    "agent_budgets file %s is not a JSON object; ignoring",
                    self.budgets_path,
                )
                return
            for agent, budget in data.items():
                try:
                    if (
                        isinstance(budget, dict)
                        and "daily_usd" in budget
                        and "monthly_usd" in budget
                    ):
                        self.budgets[agent] = {
                            "daily_usd": float(budget["daily_usd"]),
                            "monthly_usd": float(budget["monthly_usd"]),
                        }
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Skipping malformed agent budget for %s in %s: %s",
                        agent, self.budgets_path, e,
                    )
                    continue
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load agent budgets from %s: %s", self.budgets_path, e
            )

    def _save_budgets(self) -> None:
        """Persist per-agent budget overrides atomically (temp file + rename).

        Tolerates write failures (logs, does not raise) so a transient disk
        error never crashes the cost-tracking / LLM-proxy path.
        """
        try:
            self.budgets_path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(self.budgets, indent=2) + "\n"
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.budgets_path.parent), suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(content)
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass  # already closed by fdopen
                Path(tmp_path).unlink(missing_ok=True)
                raise
            try:
                os.replace(tmp_path, self.budgets_path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except Exception as e:
            logger.warning(
                "Failed to persist agent budgets to %s: %s", self.budgets_path, e
            )

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT (datetime('now')),
                trace_id TEXT,
                kind TEXT NOT NULL DEFAULT 'work'
            );
            CREATE INDEX IF NOT EXISTS idx_usage_agent_ts ON usage(agent, timestamp);
        """)
        # Session observability (Phase 1) — additive, nullable trace_id so
        # a cost row JOINs to its task/trace by the per-turn correlation
        # id. ALTER ... ADD COLUMN is a metadata-only op in SQLite, fast
        # and safe to re-run (we filter on the introspected column set so
        # an already-migrated DB is a no-op). Existing rows keep NULL.
        existing = {
            row[1] for row in self.db.execute("PRAGMA table_info(usage)").fetchall()
        }
        if "trace_id" not in existing:
            self.db.execute("ALTER TABLE usage ADD COLUMN trace_id TEXT")
        # B2 spend split (Phase-3 unit 1): coordination vs work ledger.
        # Same introspection-migration pattern as trace_id; legacy rows
        # take the 'work' default, which preserves their enforcement
        # semantics exactly (everything was work before the split).
        if "kind" not in existing:
            self.db.execute(
                "ALTER TABLE usage ADD COLUMN kind TEXT NOT NULL DEFAULT 'work'"
            )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def cleanup_agent(self, agent_id: str) -> int:
        """Delete all cost records for an agent. Returns rows deleted."""
        cursor = self.db.execute("DELETE FROM usage WHERE agent = ?", (agent_id,))
        self.db.commit()
        with self._budget_lock:
            if self.budgets.pop(agent_id, None) is not None:
                self._save_budgets()
        return cursor.rowcount

    def set_budget(self, agent: str, daily_usd: float | None = None, monthly_usd: float | None = None) -> None:
        if daily_usd is None or monthly_usd is None:
            defaults = _default_budget()
            if daily_usd is None:
                daily_usd = defaults["daily_usd"]
            if monthly_usd is None:
                monthly_usd = defaults["monthly_usd"]
        with self._budget_lock:
            self.budgets[agent] = {"daily_usd": daily_usd, "monthly_usd": monthly_usd}
            self._save_budgets()

    def _check_budget_post_hoc(self, agent: str) -> bool:
        """Check if agent exceeded daily/monthly budget. Returns True if over budget."""
        budget = self.budgets.get(agent)
        if not budget:
            return False
        over = False
        daily_spent = self.get_spend(agent, "today", kind="work").get("total_cost", 0)
        if daily_spent > budget["daily_usd"]:
            logger.warning(
                "Agent '%s' exceeded daily budget: $%.4f / $%.2f",
                agent, daily_spent, budget["daily_usd"],
            )
            over = True
        monthly_spent = self.get_spend(agent, "month", kind="work").get("total_cost", 0)
        if monthly_spent > budget["monthly_usd"]:
            logger.warning(
                "Agent '%s' exceeded monthly budget: $%.4f / $%.2f",
                agent, monthly_spent, budget["monthly_usd"],
            )
            over = True
        return over

    def track(
        self, agent: str, model: str, prompt_tokens: int, completion_tokens: int,
        *, bill: bool = True, kind: str = "work",
    ) -> dict:
        """Record a single LLM call. Returns {"cost": float, "over_budget": bool}.

        ``bill=False`` records the token usage with ``cost_usd=0`` — used for
        OAuth (subscription) traffic, which consumes tokens but incurs no
        per-call dollar cost. Operators keep token visibility, while the
        spend total stays $0 so subscription usage never accrues cost and
        never trips a daily/monthly budget cap (the cap reads SUM(cost_usd)).

        ``kind`` is the B2 spend-split ledger tag: ``"work"`` (default) or
        ``"coordination"`` (utility-model traffic, classified mesh-side at
        the LLM proxy). Enforcement reads filter ``kind='work'``; reporting
        stays spend-inclusive.
        """
        total = prompt_tokens + completion_tokens
        cost = (
            estimate_cost(model, input_tokens=prompt_tokens, output_tokens=completion_tokens)
            if bill else 0.0
        )

        # Session observability (Phase 1) — stamp the active per-turn
        # trace_id (seeded from the inbound X-Trace-Id header on the mesh
        # proxy path) so spend attributes to a task/session. NULL when no
        # trace is active.
        from src.shared.trace import current_trace_id
        trace_id = current_trace_id.get()
        self.db.execute(
            "INSERT INTO usage (agent, model, prompt_tokens, completion_tokens, "
            "total_tokens, cost_usd, trace_id, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (agent, model, prompt_tokens, completion_tokens, total, cost, trace_id, kind),
        )
        self.db.commit()

        # A non-billed (OAuth) row adds $0, so it can never push the agent
        # over budget — short-circuit the post-hoc check to keep that
        # guarantee explicit (and avoid attributing a metered overage to a
        # subscription call).
        over_budget = self._check_budget_post_hoc(agent) if bill else False
        return {"cost": cost, "over_budget": over_budget}

    def track_fixed_cost(self, agent: str, model: str, cost_usd: float) -> dict:
        """Record a fixed-cost API call (e.g. image generation).

        Inserts a row with zero tokens and the given USD cost.
        Returns {"cost": float, "over_budget": bool}.
        """
        from src.shared.trace import current_trace_id
        trace_id = current_trace_id.get()
        self.db.execute(
            "INSERT INTO usage (agent, model, prompt_tokens, completion_tokens, total_tokens, cost_usd, trace_id) "
            "VALUES (?, ?, 0, 0, 0, ?, ?)",
            (agent, model, cost_usd, trace_id),
        )
        self.db.commit()

        return {"cost": cost_usd, "over_budget": self._check_budget_post_hoc(agent)}

    def check_budget(self, agent: str) -> dict:
        """Check if agent is within budget (WORK ledger only — B2 split).

        Returns {"allowed": bool, "daily_used": float, "daily_limit": float, ...}.
        """
        budget = self.budgets.get(agent) or _default_budget()
        daily_used = self.get_spend(agent, "today", kind="work").get("total_cost", 0)
        monthly_used = self.get_spend(agent, "month", kind="work").get("total_cost", 0)

        allowed = (daily_used < budget["daily_usd"]) and (monthly_used < budget["monthly_usd"])

        return {
            "allowed": allowed,
            "daily_used": round(daily_used, 4),
            "daily_limit": budget["daily_usd"],
            "monthly_used": round(monthly_used, 4),
            "monthly_limit": budget["monthly_usd"],
        }

    def preflight_check(self, agent: str, model: str, estimated_tokens: int = 4096) -> dict:
        """Pre-flight budget check before an LLM call.

        Estimates the cost of the upcoming call and checks if the agent can
        afford it. Returns {"allowed": bool, "estimated_cost": float, ...}.
        """
        estimated_cost = estimate_cost(model, total_tokens=estimated_tokens)
        budget = self.budgets.get(agent) or _default_budget()
        daily_used = self.get_spend(agent, "today", kind="work").get("total_cost", 0)
        monthly_used = self.get_spend(agent, "month", kind="work").get("total_cost", 0)

        daily_ok = (daily_used + estimated_cost) <= budget["daily_usd"]
        monthly_ok = (monthly_used + estimated_cost) <= budget["monthly_usd"]
        allowed = daily_ok and monthly_ok

        if not allowed:
            logger.warning(
                "Pre-flight budget check failed for agent '%s': "
                "estimated $%.4f would exceed limits (daily: $%.4f/$%.2f, monthly: $%.4f/$%.2f)",
                agent, estimated_cost, daily_used, budget["daily_usd"],
                monthly_used, budget["monthly_usd"],
            )

        return {
            "allowed": allowed,
            "estimated_cost": round(estimated_cost, 6),
            "daily_used": round(daily_used, 4),
            "daily_limit": budget["daily_usd"],
            "monthly_used": round(monthly_used, 4),
            "monthly_limit": budget["monthly_usd"],
        }

    # ── Coordination ledger (B2 spend split, plan §8 #11) ─────────────
    #
    # Utility-model LLM traffic (heartbeats, summarization, agenda
    # ticks) is classified COORDINATION at the mesh proxy and lands in
    # the usage ledger with kind='coordination'. It is exempt from the
    # per-agent work preflight and the team envelope — the whole point
    # of B2 is that coordination churn can never starve real work — and
    # is instead gated by its own per-agent daily cap
    # (limits.coordination_daily_cap_usd; 0 = tier blocked).

    def coordination_preflight_check(
        self, agent: str, model: str, estimated_tokens: int = 4096,
    ) -> dict:
        """Pre-flight check for a coordination (utility-model) LLM call.

        Same estimate convention as ``preflight_check``. A cap ≤ 0 blocks
        the coordination tier outright (operator kill-switch).
        """
        from src.shared.limits import coordination_daily_cap_usd

        cap = coordination_daily_cap_usd()
        estimated_cost = estimate_cost(model, total_tokens=estimated_tokens)
        daily_used = self.get_spend(agent, "today", kind="coordination").get("total_cost", 0)
        allowed = cap > 0 and (daily_used + estimated_cost) <= cap
        if not allowed:
            logger.warning(
                "Coordination pre-flight check failed for agent '%s': "
                "estimated $%.4f would exceed the daily coordination cap ($%.4f/$%.2f)",
                agent, estimated_cost, daily_used, cap,
            )
        return {
            "allowed": allowed,
            "estimated_cost": round(estimated_cost, 6),
            "daily_used": round(daily_used, 4),
            "daily_limit": cap,
        }

    def get_coordination_spend(self, agent: str) -> dict:
        """Today's coordination spend + the daily cap (introspect breakout)."""
        from src.shared.limits import coordination_daily_cap_usd

        cap = coordination_daily_cap_usd()
        daily_used = self.get_spend(agent, "today", kind="coordination").get("total_cost", 0)
        return {"daily_used": round(daily_used, 4), "daily_limit": cap}

    def get_spend(
        self, agent: str | None = None, period: str = "today",
        kind: str | None = None,
    ) -> dict:
        """Get spend breakdown for an agent or all agents.

        ``kind=None`` (default) includes ALL rows — reporting surfaces
        (dashboard, get_team_spend, introspect display) stay
        spend-inclusive across the B2 split. Enforcement callers pass
        ``kind="work"`` so coordination traffic never counts against the
        work budget (and vice versa).
        """
        since = _period_to_since(period)

        query = (
            "SELECT model, SUM(prompt_tokens), SUM(completion_tokens), "
            "SUM(total_tokens), SUM(cost_usd) FROM usage WHERE timestamp >= ?"
        )
        params: list = [since]
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " GROUP BY model"

        rows = self.db.execute(query, params).fetchall()

        total_cost = sum(r[4] or 0 for r in rows)
        total_tokens = sum(r[3] or 0 for r in rows)
        by_model = {
            r[0]: {
                "prompt": r[1] or 0,
                "completion": r[2] or 0,
                "total": r[3] or 0,
                "cost": round(r[4] or 0, 4),
            }
            for r in rows
        }

        return {
            "agent": agent or "all",
            "period": period,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "by_model": by_model,
        }

    # ── Team-level budget enforcement (durable envelope, plan B4) ─────
    #
    # The envelope lives on the team row in the TeamStore (data/teams.db),
    # not here — costs.py only reads it. SEMANTICS (the B4 flip): an
    # unset/NULL/0 envelope means UNLIMITED, the opposite of the per-agent
    # ledger's arithmetic where a 0 budget blocks everything. A brand-new
    # enforced team layer defaulting to "block all members" would be the
    # exact failure mode B4 warns about.

    def _members_spend_totals(self, members: list[str], since: str) -> tuple[float, int]:
        """Sum WORK (cost, tokens) across member agents since a timestamp.

        Feeds ``team_envelope_check`` — enforcement, so coordination rows
        are excluded (B2): utility-model churn must never eat the team's
        work envelope. Display aggregation (``get_team_spend``) stays
        spend-inclusive via ``get_spend``.
        """
        if not members:
            return 0.0, 0
        placeholders = ",".join("?" for _ in members)
        row = self.db.execute(
            "SELECT SUM(cost_usd), SUM(total_tokens) FROM usage "
            f"WHERE timestamp >= ? AND kind = 'work' AND agent IN ({placeholders})",
            [since, *members],
        ).fetchone()
        return float(row[0] or 0.0), int(row[1] or 0)

    def team_envelope_check(self, agent: str, model: str, estimated_tokens: int = 4096) -> dict:
        """Pre-flight team-envelope check for one member's upcoming call.

        Returns ``{"allowed": True, "team": None}`` when the agent has no
        team, no store is wired, or the team's envelope is unset/0
        (= unlimited) — those are legitimate "no envelope" cases, not
        errors, and are unaffected by the posture knob below.

        On a store READ ERROR the posture is CONFIGURABLE
        (``limits.team_envelope_fail_closed`` /
        ``OPENLEGION_TEAM_ENVELOPE_FAIL_CLOSED``). The default (False) fails
        OPEN — the envelope is an additional governor on top of the
        always-on per-agent budget, and a storage hiccup must not take down
        the whole LLM path. Set True (Phase-0 safety substrate,
        docs/plans/2026-07-16-autonomous-team-delivery.md §0) to fail CLOSED:
        the read error BLOCKS the call, returning an ``{"allowed": False,
        "reason": "envelope_check_unavailable", ...}`` dict shaped like a
        real envelope-exceeded result, so an unattended fleet cannot keep
        spending through a governor it can no longer read.
        """
        store = self._team_store
        if store is None or not agent:
            return {"allowed": True, "team": None}
        try:
            team = store.team_of(agent)
            if not team:
                return {"allowed": True, "team": None}
            trow = store.get_team(team)
        except Exception as e:
            from src.shared.limits import team_envelope_fail_closed

            if team_envelope_fail_closed():
                logger.warning("Team envelope check FAILED CLOSED (store read failed): %s", e)
                # Shape-compatible with the real envelope-exceeded return so
                # the credentials.py blocked path (which subscripts
                # daily_used/monthly_used/estimated_cost) renders cleanly.
                return {
                    "allowed": False,
                    "team": None,
                    "reason": "envelope_check_unavailable",
                    "estimated_cost": 0.0,
                    "daily_used": 0.0,
                    "daily_limit": None,
                    "monthly_used": 0.0,
                    "monthly_limit": None,
                }
            logger.warning("Team envelope check skipped (store read failed): %s", e)
            return {"allowed": True, "team": None}
        if trow is None:
            return {"allowed": True, "team": None}
        daily_env = trow.get("budget_daily_usd") or 0.0
        monthly_env = trow.get("budget_monthly_usd") or 0.0
        if daily_env <= 0 and monthly_env <= 0:
            # Unset / 0 / negative = unlimited (B4).
            return {"allowed": True, "team": team, "daily_limit": None, "monthly_limit": None}

        members = trow.get("members") or [agent]
        estimated_cost = estimate_cost(model, total_tokens=estimated_tokens)
        daily_used, _ = self._members_spend_totals(members, _period_to_since("today"))
        monthly_used, _ = self._members_spend_totals(members, _period_to_since("month"))

        daily_ok = daily_env <= 0 or (daily_used + estimated_cost) <= daily_env
        monthly_ok = monthly_env <= 0 or (monthly_used + estimated_cost) <= monthly_env
        allowed = daily_ok and monthly_ok
        if not allowed:
            logger.warning(
                "Team envelope check failed for agent '%s' (team '%s'): "
                "estimated $%.4f would exceed envelope (daily: $%.4f/$%.2f, monthly: $%.4f/$%.2f)",
                agent, team, estimated_cost, daily_used, daily_env, monthly_used, monthly_env,
            )
        return {
            "allowed": allowed,
            "team": team,
            "estimated_cost": round(estimated_cost, 6),
            "daily_used": round(daily_used, 4),
            "daily_limit": daily_env if daily_env > 0 else None,
            "monthly_used": round(monthly_used, 4),
            "monthly_limit": monthly_env if monthly_env > 0 else None,
        }

    def get_team_spend(self, team: str, period: str = "today") -> dict:
        """Aggregate spend across a team's members, with its envelope.

        Unknown team (or no store wired) keeps the historical error-dict
        contract — callers guard on ``"error" not in result``.
        ``daily_limit``/``monthly_limit`` are None when the envelope is
        unset/0 (= unlimited, plan B4).
        """
        store = self._team_store
        trow = None
        if store is not None:
            try:
                trow = store.get_team(team)
            except Exception as e:
                logger.warning("get_team_spend: store read failed: %s", e)
        if trow is None:
            return {"team": team, "error": "Unknown team (or no team store wired)"}
        members = trow.get("members") or []
        total_cost = 0.0
        total_tokens = 0
        agent_breakdown = []
        for member in members:
            spend = self.get_spend(member, period)
            total_cost += spend.get("total_cost", 0)
            total_tokens += spend.get("total_tokens", 0)
            agent_breakdown.append({
                "agent": member,
                "cost": spend.get("total_cost", 0),
                "tokens": spend.get("total_tokens", 0),
            })
        daily_env = trow.get("budget_daily_usd") or 0.0
        monthly_env = trow.get("budget_monthly_usd") or 0.0
        # ENFORCED figure: ``team_envelope_check`` counts only kind='work',
        # but ``total_cost`` above is spend-INCLUSIVE (work + coordination).
        # Surface ``work_cost`` — the number that actually trips the
        # daily/monthly limit — so the operator isn't misled by a total that
        # sits above the enforced amount.
        try:
            work_cost, _work_tokens = self._members_spend_totals(
                members, _period_to_since(period),
            )
        except Exception as e:  # noqa: BLE001 - display degrades to the total
            logger.warning("get_team_spend: work-total read failed: %s", e)
            work_cost = total_cost
        return {
            "team": team,
            "period": period,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "work_cost": round(work_cost, 4),
            "coordination_cost": round(max(0.0, total_cost - work_cost), 4),
            "daily_limit": daily_env if daily_env > 0 else None,
            "monthly_limit": monthly_env if monthly_env > 0 else None,
            "agents": agent_breakdown,
        }

    def get_all_agents_spend(self, period: str = "today") -> list[dict]:
        since = _period_to_since(period)
        rows = self.db.execute(
            "SELECT agent, SUM(total_tokens), SUM(cost_usd) FROM usage "
            "WHERE timestamp >= ? GROUP BY agent ORDER BY SUM(cost_usd) DESC",
            (since,),
        ).fetchall()
        return [
            {"agent": r[0], "tokens": r[1] or 0, "cost": round(r[2] or 0, 4)}
            for r in rows
        ]

    def get_all_agents_last_worked(self) -> dict[str, float]:
        """Map of agent → unix timestamp (UTC) of its most recent LLM call.

        This is an accurate "last actually worked" signal: every billed LLM
        call is recorded in the usage ledger, so the most recent row marks
        the last time the agent did real work — distinct from the health
        monitor's "last seen", which is just a container liveness probe.
        Returns an empty dict if nothing is recorded yet.
        """
        rows = self.db.execute(
            "SELECT agent, MAX(timestamp) FROM usage GROUP BY agent"
        ).fetchall()
        out: dict[str, float] = {}
        for agent, ts in rows:
            if not agent or not ts:
                continue
            try:
                # usage.timestamp is "YYYY-MM-DD HH:MM:SS" in UTC
                # (SQLite datetime('now')) — match _period_to_since's format.
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                out[agent] = dt.timestamp()
            except (ValueError, TypeError):
                continue
        return out

    def get_spend_by_model(self, period: str = "today") -> list[dict]:
        """Get cost breakdown by model across all agents."""
        since = _period_to_since(period)
        rows = self.db.execute(
            "SELECT model, SUM(prompt_tokens), SUM(completion_tokens), "
            "SUM(total_tokens), SUM(cost_usd) FROM usage WHERE timestamp >= ? "
            "GROUP BY model ORDER BY SUM(cost_usd) DESC",
            (since,),
        ).fetchall()
        return [
            {
                "model": r[0],
                "prompt_tokens": r[1] or 0,
                "completion_tokens": r[2] or 0,
                "tokens": r[3] or 0,
                "cost": round(r[4] or 0, 4),
            }
            for r in rows
        ]


def _period_to_since(period: str) -> str:
    now = datetime.now(timezone.utc)
    if period == "today":
        return now.strftime("%Y-%m-%d 00:00:00")
    if period == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    if period == "month":
        return now.strftime("%Y-%m-01 00:00:00")
    if period == "week":
        return (now - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
    return period
