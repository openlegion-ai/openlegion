"""Cost tracking and budget enforcement for LLM usage.

Intercepts at the CredentialVault layer — every LLM call already routes
through it, so this is a single integration point covering all agents.

Storage: SQLite (lightweight, no external services).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.shared.models import estimate_cost, get_model_cost  # noqa: F401 — re-export
from src.shared.utils import setup_logging

logger = setup_logging("host.costs")

def _default_budget() -> dict:
    """Read default budget from dashboard system settings, falling back to hardcoded defaults."""
    try:
        p = Path("config/settings.json")
        if p.exists():
            data = json.loads(p.read_text())
            return {
                "daily_usd": data.get("default_daily_budget", 10.0),
                "monthly_usd": data.get("default_monthly_budget", 200.0),
            }
    except (json.JSONDecodeError, OSError):
        pass
    return {"daily_usd": 10.0, "monthly_usd": 200.0}


class CostTracker:
    """Tracks token usage and cost per agent, enforces budgets."""

    def __init__(self, db_path: str = "data/costs.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._init_schema()
        self.budgets: dict[str, dict[str, float]] = {}
        self._project_budgets: dict[str, dict] = {}

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
                timestamp TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_usage_agent_ts ON usage(agent, timestamp);
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def set_budget(self, agent: str, daily_usd: float | None = None, monthly_usd: float | None = None) -> None:
        if daily_usd is None or monthly_usd is None:
            defaults = _default_budget()
            if daily_usd is None:
                daily_usd = defaults["daily_usd"]
            if monthly_usd is None:
                monthly_usd = defaults["monthly_usd"]
        self.budgets[agent] = {"daily_usd": daily_usd, "monthly_usd": monthly_usd}

    def _check_budget_post_hoc(self, agent: str) -> bool:
        """Check if agent exceeded daily/monthly budget. Returns True if over budget."""
        budget = self.budgets.get(agent)
        if not budget:
            return False
        over = False
        daily_spent = self.get_spend(agent, "today").get("total_cost", 0)
        if daily_spent > budget["daily_usd"]:
            logger.warning(
                "Agent '%s' exceeded daily budget: $%.4f / $%.2f",
                agent, daily_spent, budget["daily_usd"],
            )
            over = True
        monthly_spent = self.get_spend(agent, "month").get("total_cost", 0)
        if monthly_spent > budget["monthly_usd"]:
            logger.warning(
                "Agent '%s' exceeded monthly budget: $%.4f / $%.2f",
                agent, monthly_spent, budget["monthly_usd"],
            )
            over = True
        return over

    def track(
        self, agent: str, model: str, prompt_tokens: int, completion_tokens: int,
    ) -> dict:
        """Record a single LLM call. Returns {"cost": float, "over_budget": bool}."""
        total = prompt_tokens + completion_tokens
        cost = estimate_cost(model, input_tokens=prompt_tokens, output_tokens=completion_tokens)

        self.db.execute(
            "INSERT INTO usage (agent, model, prompt_tokens, completion_tokens, total_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent, model, prompt_tokens, completion_tokens, total, cost),
        )
        self.db.commit()

        return {"cost": cost, "over_budget": self._check_budget_post_hoc(agent)}

    def track_fixed_cost(self, agent: str, model: str, cost_usd: float) -> dict:
        """Record a fixed-cost API call (e.g. image generation).

        Inserts a row with zero tokens and the given USD cost.
        Returns {"cost": float, "over_budget": bool}.
        """
        self.db.execute(
            "INSERT INTO usage (agent, model, prompt_tokens, completion_tokens, total_tokens, cost_usd) "
            "VALUES (?, ?, 0, 0, 0, ?)",
            (agent, model, cost_usd),
        )
        self.db.commit()

        return {"cost": cost_usd, "over_budget": self._check_budget_post_hoc(agent)}

    def check_budget(self, agent: str) -> dict:
        """Check if agent is within budget.

        Returns {"allowed": bool, "daily_used": float, "daily_limit": float, ...}.
        """
        budget = self.budgets.get(agent) or _default_budget()
        daily_used = self.get_spend(agent, "today").get("total_cost", 0)
        monthly_used = self.get_spend(agent, "month").get("total_cost", 0)

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
        daily_used = self.get_spend(agent, "today").get("total_cost", 0)
        monthly_used = self.get_spend(agent, "month").get("total_cost", 0)

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

    def get_spend(self, agent: str | None = None, period: str = "today") -> dict:
        """Get spend breakdown for an agent or all agents."""
        since = _period_to_since(period)

        query = (
            "SELECT model, SUM(prompt_tokens), SUM(completion_tokens), "
            "SUM(total_tokens), SUM(cost_usd) FROM usage WHERE timestamp >= ?"
        )
        params: list = [since]
        if agent:
            query += " AND agent = ?"
            params.append(agent)
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

    # ── Project-level budget enforcement ──────────────────────────

    def set_project_budget(
        self,
        project: str,
        members: list[str],
        daily_usd: float = 50.0,
        monthly_usd: float = 1000.0,
    ) -> None:
        """Set an aggregate budget for a project (sum of member agents)."""
        self._project_budgets[project] = {
            "members": members,
            "daily_usd": daily_usd,
            "monthly_usd": monthly_usd,
        }

    def get_project_spend(self, project: str, period: str = "today") -> dict:
        """Aggregate spend across all member agents for a project."""
        pbudget = self._project_budgets.get(project)
        if pbudget is None:
            return {"project": project, "error": "No project budget configured"}
        members = pbudget["members"]
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
        return {
            "project": project,
            "period": period,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "daily_limit": pbudget["daily_usd"],
            "monthly_limit": pbudget["monthly_usd"],
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
    if period == "month":
        return now.strftime("%Y-%m-01 00:00:00")
    if period == "week":
        return (now - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
    return period
