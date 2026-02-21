"""Cost tracking and budget enforcement for LLM usage.

Intercepts at the CredentialVault layer — every LLM call already routes
through it, so this is a single integration point covering all agents.

Storage: SQLite (lightweight, no external services).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("host.costs")

# Cost per 1K tokens (input, output). Update as pricing changes.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (0.0025, 0.01),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "openai/gpt-4.1": (0.002, 0.008),
    "openai/gpt-4.1-mini": (0.0004, 0.0016),
    "openai/gpt-4.1-nano": (0.0001, 0.0004),
    "openai/o3-mini": (0.0011, 0.0044),
    "anthropic/claude-sonnet-4-5-20250929": (0.003, 0.015),
    "anthropic/claude-haiku-4-5-20251001": (0.0008, 0.004),
    "text-embedding-3-small": (0.00002, 0.0),
}

_DEFAULT_COST = (0.003, 0.015)  # Conservative fallback


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
) -> float:
    """Estimate USD cost for an LLM call.

    If input/output split is unavailable, falls back to 70/30 split of total_tokens.
    """
    ir, or_ = MODEL_COSTS.get(model, _DEFAULT_COST)
    pt = input_tokens or int(total_tokens * 0.7)
    ct = output_tokens or (total_tokens - pt)
    return round((pt / 1000 * ir) + (ct / 1000 * or_), 6)


class CostTracker:
    """Tracks token usage and cost per agent, enforces budgets."""

    def __init__(self, db_path: str = "data/costs.db", event_bus=None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._event_bus = event_bus
        self._init_schema()
        self.budgets: dict[str, dict[str, float]] = {}

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

    def set_budget(self, agent: str, daily_usd: float = 10.0, monthly_usd: float = 200.0) -> None:
        self.budgets[agent] = {"daily_usd": daily_usd, "monthly_usd": monthly_usd}

    def track(
        self, agent: str, model: str, prompt_tokens: int, completion_tokens: int,
    ) -> float:
        """Record a single LLM call. Returns the cost in USD."""
        total = prompt_tokens + completion_tokens
        cost = estimate_cost(model, input_tokens=prompt_tokens, output_tokens=completion_tokens)

        self.db.execute(
            "INSERT INTO usage (agent, model, prompt_tokens, completion_tokens, total_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent, model, prompt_tokens, completion_tokens, total, cost),
        )
        self.db.commit()

        return cost

    def check_budget(self, agent: str) -> dict:
        """Check if agent is within budget.

        Returns {"allowed": bool, "daily_used": float, "daily_limit": float, ...}.
        """
        budget = self.budgets.get(agent, {"daily_usd": 10.0, "monthly_usd": 200.0})
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


def _period_to_since(period: str) -> str:
    now = datetime.now(UTC)
    if period == "today":
        return now.strftime("%Y-%m-%d 00:00:00")
    if period == "month":
        return now.strftime("%Y-%m-01 00:00:00")
    if period == "week":
        return (now - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
    return period
