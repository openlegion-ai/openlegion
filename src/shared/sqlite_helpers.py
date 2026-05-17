"""Shared SQLite helpers — common connect + pragma boilerplate."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def open_db(
    path: str | Path,
    *,
    busy_timeout_ms: int = 30000,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open a sqlite3 connection with the engine's standard pragmas.

    Caller is responsible for any additional pragmas (journal_mode,
    foreign_keys, etc.) and for closing the connection.
    """
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    return conn
