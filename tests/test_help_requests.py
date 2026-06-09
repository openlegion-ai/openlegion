"""Unit tests for the persistent HelpRequests registry.

Covers the guarantees the "Needs you" panel depends on:
  * persistence across a (simulated) mesh restart — the reason this stopped
    being an in-memory dict
  * atomic single-claim on resolve (a save racing a cancel resolves once)
  * list_open as the authoritative feed
  * field hoisting (service/name/description/url) for the feed
"""

from __future__ import annotations

import pytest

from src.host.help_requests import HelpRequests


def test_record_and_get_roundtrip(tmp_path):
    store = HelpRequests(db_path=str(tmp_path / "hr.db"))
    rid = store.record(
        "credential_request", "agent-1",
        {"name": "stripe", "service": "stripe", "description": "for checkout"},
    )
    rec = store.get(rid)
    assert rec is not None
    assert rec["kind"] == "credential_request"
    assert rec["agent_id"] == "agent-1"
    assert rec["service"] == "stripe"
    assert rec["name"] == "stripe"
    assert rec["description"] == "for checkout"
    assert rec["status"] == "open"


def test_list_open_orders_oldest_first(tmp_path):
    store = HelpRequests(db_path=str(tmp_path / "hr.db"))
    a = store.record("credential_request", "a", {"name": "one"})
    b = store.record("browser_login_request", "b", {"service": "two"})
    ids = [r["request_id"] for r in store.list_open()]
    assert ids == [a, b]


def test_resolve_pops_and_is_idempotent(tmp_path):
    store = HelpRequests(db_path=str(tmp_path / "hr.db"))
    rid = store.record("credential_request", "a", {"name": "x"})
    claimed = store.resolve(rid, status="resolved")
    assert claimed is not None
    assert claimed["status"] == "resolved"
    assert store.get(rid) is None
    # Second resolve loses the claim — returns None, NOT an error. This is the
    # save/cancel race guard: only one caller fires its side effect.
    assert store.resolve(rid, status="resolved") is None
    assert store.list_open() == []


def test_resolve_expected_kind_mismatch_is_noop(tmp_path):
    store = HelpRequests(db_path=str(tmp_path / "hr.db"))
    rid = store.record("credential_request", "a", {"name": "x"})
    # A browser-login cancel must not claim a credential request.
    assert store.resolve(rid, expected_kind="browser_login_request") is None
    # Still open for its real kind.
    assert store.resolve(rid, expected_kind="credential_request") is not None


def test_persists_across_restart(tmp_path):
    db = str(tmp_path / "hr.db")
    store = HelpRequests(db_path=db)
    rid = store.record("browser_login_request", "scout", {"service": "linkedin"})
    store.close()
    # Simulate a mesh restart: a fresh store on the same file must still see
    # the open request. This is the whole reason the registry is persisted —
    # an empty panel after restart would falsely read as "nothing needs you".
    store2 = HelpRequests(db_path=db)
    open_ids = [r["request_id"] for r in store2.list_open()]
    assert rid in open_ids


def test_reap_old_drops_only_aged_rows(tmp_path):
    store = HelpRequests(db_path=str(tmp_path / "hr.db"))
    rid = store.record("credential_request", "a", {"name": "fresh"})
    # Nothing aged out at a 1-day cutoff.
    assert store.reap_old(max_age_sec=86400) == 0
    # Everything older than 0s is reaped.
    assert store.reap_old(max_age_sec=0) == 1
    assert store.get(rid) is None
