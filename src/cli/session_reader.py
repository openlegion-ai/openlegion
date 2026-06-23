"""Read-only session reconstruction reader (Phase 3 of session observability).

This module assembles a *session* — a single human-rooted interaction — from
the host-side SQLite stores so an engineer can answer "what happened in this
session?" after the fact, typically over SSH on a hosted VPS.

Design decisions (see docs/plans/2026-06-18-session-observability.md, Phase 3):

* **Read host SQLite DIRECTLY, read-only.** Each DB is opened with
  ``sqlite3.connect("file:<path>?mode=ro", uri=True)`` and queried with raw
  ``SELECT``s. We deliberately do NOT instantiate the store classes
  (IntentStore / TraceStore / Tasks / CostTracker): they open read-write and
  may run GC / migrations on a prod DB. Read-only + raw SQL is the whole point
  — the reader must work *even when the mesh is down* (forensics happen exactly
  then) and must never mutate prod data.

  This couples the reader to the on-disk schema of those stores. That coupling
  is acceptable and intentional: the alternative (going through the live store
  classes or the mesh HTTP API) sacrifices the down-mesh and never-mutate
  guarantees that are the reason this tool exists. The columns read here are the
  stable, long-lived ones; the per-store ``CREATE TABLE`` is the source of truth
  (src/host/intent.py, traces.py, orchestration.py, costs.py). Missing DBs /
  missing columns degrade gracefully rather than crashing.

* A *session* is, for now, the per-turn ``trace_id`` (the derived-grouping
  decision in the plan — no stored ``session_id`` yet). ``session <trace_id>``
  reconstructs one; ``sessions --since`` lists recent trace_ids to drill into.

Known limitation (surfaced in --help and output): chat transcripts are
container-local (per-agent ``chat_transcript.jsonl``), so the host-side timeline
= intent + traces + tasks + costs. Full per-turn transcript text needs a
separate per-container fetch.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

# One-line caveat reused in --help text and the rendered footer.
TRANSCRIPT_CAVEAT = (
    "Note: per-turn chat transcript text is container-local; this host-side "
    "timeline = intent + traces + tasks + costs (not the full conversation)."
)


# ── time parsing ─────────────────────────────────────────────


def parse_since(since: str | None) -> float:
    """Parse a ``--since`` value into an epoch-seconds floor.

    Accepts (case-insensitive):
    * ``""`` / ``None`` → last 7 days.
    * ``today`` → local midnight today.
    * a duration ``Nh`` / ``Nd`` / ``Nm`` / ``Ns`` → now minus that span.
    * an ISO date / timestamp (``2026-06-18`` or ``2026-06-18T12:00:00``).

    Mirrors the mesh-side ``_parse_since`` (src/host/server.py) so flag
    semantics stay consistent across the CLI; unparseable input falls back to
    the 7-day default rather than erroring.
    """
    default = time.time() - (7 * 24 * 60 * 60)
    if not since:
        return default
    s = since.strip().lower()
    if not s:
        return default
    if s == "today":
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()
    # Duration form: Ns / Nm / Nh / Nd
    if s[-1] in {"s", "m", "h", "d"} and s[:-1].isdigit():
        n = int(s[:-1])
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[s[-1]]
        return time.time() - (n * mult)
    # ISO date / timestamp
    try:
        dt = datetime.fromisoformat(s.replace("z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return default


def _fmt_ts(ts: float | None) -> str:
    """Format an epoch-seconds timestamp as a UTC ``YYYY-MM-DD HH:MM:SS`` string."""
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError, TypeError):
        return str(ts)


# ── read-only DB access ──────────────────────────────────────


def _connect_ro(path: str | Path) -> sqlite3.Connection | None:
    """Open a SQLite DB read-only (``mode=ro``). Returns None if it can't be opened.

    ``mode=ro`` guarantees the reader never creates or mutates the DB — opening
    a *missing* file with ``mode=ro`` raises (unlike default connect, which would
    create it), which is exactly the safety we want: a missing store degrades to
    "this layer is unavailable", never to a freshly-created empty file in the
    prod data dir.
    """
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
        return None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the column names of *table*, or an empty set if it doesn't exist."""
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.OperationalError:
        return set()


# ── per-store fetchers (each degrades gracefully) ────────────


def _fetch_intent(data_dir: Path, trace_id: str) -> list[dict]:
    conn = _connect_ro(data_dir / "intent.db")
    if conn is None:
        return []
    try:
        cols = _columns(conn, "intent")
        if not cols:
            return []
        rows = conn.execute(
            "SELECT timestamp, origin_kind, origin_channel, origin_user, agent, message "
            "FROM intent WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "origin_kind": r["origin_kind"],
                "origin_channel": r["origin_channel"],
                "origin_user": r["origin_user"],
                "agent": r["agent"],
                "message": r["message"],
            }
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _fetch_actions(data_dir: Path, trace_id: str) -> list[dict]:
    conn = _connect_ro(data_dir / "traces.db")
    if conn is None:
        return []
    try:
        cols = _columns(conn, "traces")
        if not cols:
            return []
        # Read only columns that exist (older DBs predate status/error).
        wanted = ["timestamp", "source", "agent", "event_type", "detail", "duration_ms", "status", "error"]
        present = [c for c in wanted if c in cols]
        rows = conn.execute(
            f"SELECT {', '.join(present)} FROM traces WHERE trace_id = ? ORDER BY timestamp, rowid",
            (trace_id,),
        ).fetchall()
        return [{c: r[c] for c in present} for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _fetch_tasks(data_dir: Path, trace_id: str) -> list[dict]:
    conn = _connect_ro(data_dir / "tasks.db")
    if conn is None:
        return []
    try:
        cols = _columns(conn, "tasks")
        if not cols or "trace_id" not in cols:
            # Pre-Phase-1 DB without the trace_id column — can't correlate.
            return []
        wanted = [
            "id", "title", "status", "assignee", "blocker_note", "result_summary",
            "outcome", "parent_task_id", "previous_task_id", "created_at", "updated_at",
            "completed_at",
        ]
        present = [c for c in wanted if c in cols]
        rows = conn.execute(
            f"SELECT {', '.join(present)} FROM tasks WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        ).fetchall()
        return [{c: r[c] for c in present} for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _fetch_cost(data_dir: Path, trace_id: str) -> dict:
    """Return a cost rollup for a trace_id: totals + per-model breakdown."""
    empty = {"total_tokens": 0, "total_cost_usd": 0.0, "by_model": []}
    conn = _connect_ro(data_dir / "costs.db")
    if conn is None:
        return empty
    try:
        cols = _columns(conn, "usage")
        if not cols or "trace_id" not in cols:
            return empty
        rows = conn.execute(
            "SELECT model, "
            "SUM(COALESCE(total_tokens, 0)), "
            "SUM(COALESCE(cost_usd, 0.0)) "
            "FROM usage WHERE trace_id = ? GROUP BY model ORDER BY SUM(cost_usd) DESC",
            (trace_id,),
        ).fetchall()
        by_model = [
            {"model": r[0], "tokens": int(r[1] or 0), "cost_usd": round(float(r[2] or 0.0), 6)}
            for r in rows
        ]
        return {
            "total_tokens": sum(m["tokens"] for m in by_model),
            "total_cost_usd": round(sum(m["cost_usd"] for m in by_model), 6),
            "by_model": by_model,
        }
    except sqlite3.OperationalError:
        return empty
    finally:
        conn.close()


# ── assembly ─────────────────────────────────────────────────


def assemble_session(data_dir: str | Path, trace_id: str) -> dict:
    """Assemble a full session object for one ``trace_id`` from the four stores.

    Each layer is fetched independently; a missing DB or missing column yields an
    empty/zeroed layer rather than raising. The ``found`` flag is False only when
    *no* store has any row for the trace_id.
    """
    data_dir = Path(data_dir)
    intent = _fetch_intent(data_dir, trace_id)
    actions = _fetch_actions(data_dir, trace_id)
    tasks = _fetch_tasks(data_dir, trace_id)
    cost = _fetch_cost(data_dir, trace_id)
    found = bool(intent or actions or tasks or cost["by_model"])
    return {
        "trace_id": trace_id,
        "found": found,
        "intent": intent,
        "actions": actions,
        "tasks": tasks,
        "cost": cost,
    }


def list_sessions(
    data_dir: str | Path,
    *,
    since: float,
    user: str | None = None,
    agent: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return recent sessions (one summary row per intent), newest first.

    The ``intent`` table is the human-rooted index, so we drive the listing off
    it. For each row we add a cheap per-trace outcome + cost lookup so the
    summary is useful without drilling in. A missing intent.db yields an empty
    list (the reader still works for direct ``session <trace_id>`` lookups
    against the other stores).
    """
    data_dir = Path(data_dir)
    conn = _connect_ro(data_dir / "intent.db")
    if conn is None:
        return []
    try:
        cols = _columns(conn, "intent")
        if not cols:
            return []
        conditions = ["timestamp >= ?"]
        params: list = [since]
        if user is not None:
            conditions.append("origin_user = ?")
            params.append(user)
        if agent is not None:
            conditions.append("agent = ?")
            params.append(agent)
        where = " WHERE " + " AND ".join(conditions)
        params.append(max(1, min(limit, 1000)))
        rows = conn.execute(
            "SELECT trace_id, timestamp, origin_kind, origin_channel, origin_user, agent, message "
            f"FROM intent{where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    summaries: list[dict] = []
    for r in rows:
        trace_id = r["trace_id"]
        tasks = _fetch_tasks(data_dir, trace_id)
        cost = _fetch_cost(data_dir, trace_id)
        # The terminal task's status/outcome is the most useful one-line signal.
        outcome = ""
        if tasks:
            last = tasks[-1]
            outcome = last.get("status") or ""
            if last.get("outcome"):
                outcome = f"{outcome}/{last['outcome']}"
        message = (r["message"] or "").replace("\n", " ").strip()
        summaries.append(
            {
                "trace_id": trace_id,
                "timestamp": r["timestamp"],
                "origin_kind": r["origin_kind"],
                "origin_channel": r["origin_channel"],
                "origin_user": r["origin_user"],
                "agent": r["agent"],
                "preview": message[:80],
                "outcome": outcome,
                "total_cost_usd": cost["total_cost_usd"],
                "total_tokens": cost["total_tokens"],
            }
        )
    return summaries


# ── rendering ────────────────────────────────────────────────


def render_session(session: dict) -> str:
    """Render one assembled session as a human-readable chronological timeline."""
    trace_id = session["trace_id"]
    if not session["found"]:
        return f"No session found for {trace_id}."

    lines: list[str] = []
    lines.append(f"Session {trace_id}")
    lines.append("=" * (8 + len(trace_id)))
    lines.append("")

    # Intent
    lines.append("INTENT")
    if session["intent"]:
        for it in session["intent"]:
            who = it.get("origin_user") or "?"
            chan = it.get("origin_channel") or it.get("origin_kind") or "?"
            lines.append(f"  [{_fmt_ts(it.get('timestamp'))}] {who} via {chan} → {it.get('agent') or '?'}")
            lines.append(f"    {it.get('message') or ''}")
    else:
        lines.append("  (no verbatim intent captured for this trace)")
    lines.append("")

    # Chronological actions + task transitions, merged on timestamp.
    lines.append("TIMELINE")
    events: list[tuple[float, str]] = []
    for a in session["actions"]:
        ts = a.get("timestamp") or 0
        detail = a.get("detail") or ""
        status = a.get("status") or ""
        dur = a.get("duration_ms") or 0
        agent = a.get("agent") or ""
        bits = [a.get("event_type") or "event"]
        if agent:
            bits.append(f"agent={agent}")
        if detail:
            bits.append(detail[:120])
        tail = []
        if dur:
            tail.append(f"{dur}ms")
        if status:
            tail.append(status)
        if a.get("error"):
            tail.append(f"error={a['error'][:80]}")
        suffix = f" ({', '.join(tail)})" if tail else ""
        events.append((ts, f"  [{_fmt_ts(ts)}] {' '.join(bits)}{suffix}"))
    for t in session["tasks"]:
        ts = t.get("created_at") or 0
        line = (
            f"  [{_fmt_ts(ts)}] task {t.get('id') or '?'} → {t.get('status') or '?'} "
            f"(assignee={t.get('assignee') or '?'})"
        )
        events.append((ts, line))
    if events:
        for _ts, line in sorted(events, key=lambda e: e[0]):
            lines.append(line)
    else:
        lines.append("  (no actions or task transitions recorded)")
    lines.append("")

    # Outcome
    lines.append("OUTCOME")
    if session["tasks"]:
        for t in session["tasks"]:
            lines.append(
                f"  task {t.get('id') or '?'}: status={t.get('status') or '?'}"
                + (f", outcome={t['outcome']}" if t.get("outcome") else "")
            )
            if t.get("blocker_note"):
                lines.append(f"    blocker: {t['blocker_note']}")
            if t.get("result_summary"):
                lines.append(f"    result: {t['result_summary']}")
    else:
        lines.append("  (no task records for this trace)")
    lines.append("")

    # Cost rollup
    cost = session["cost"]
    lines.append("COST")
    lines.append(
        f"  total: {cost['total_tokens']} tokens, ${cost['total_cost_usd']:.6f}"
    )
    for m in cost["by_model"]:
        lines.append(f"    {m['model']}: {m['tokens']} tokens, ${m['cost_usd']:.6f}")
    lines.append("")
    lines.append(TRANSCRIPT_CAVEAT)
    return "\n".join(lines)


def render_sessions(summaries: list[dict]) -> str:
    """Render the ``sessions --since`` listing as a one-row-per-session table."""
    if not summaries:
        return "No sessions found."
    lines: list[str] = []
    header = f"{'Trace':<16} {'When (UTC)':<20} {'User':<14} {'Outcome':<14} {'Cost':<10} Preview"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{(s['trace_id'] or '')[:16]:<16} "
            f"{_fmt_ts(s['timestamp']):<20} "
            f"{(s['origin_user'] or '-')[:14]:<14} "
            f"{(s['outcome'] or '-')[:14]:<14} "
            f"${s['total_cost_usd']:<9.4f} "
            f"{s['preview']}"
        )
    lines.append("")
    lines.append(TRANSCRIPT_CAVEAT)
    return "\n".join(lines)
