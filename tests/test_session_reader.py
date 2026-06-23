"""Tests for the Phase 3 read-only session reader (openlegion session/sessions).

Seeds temp DBs by WRITING with the real store classes (IntentStore, Tasks,
TraceStore, CostTracker) into a tmp data dir, then invokes the reader pointed at
that dir via ``--data-dir`` and asserts the assembled timeline / --json shape.

The reader opens the host DBs read-only (mode=ro) and degrades gracefully when a
DB file is missing — both correctness-critical bits are asserted here.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from click.testing import CliRunner

from src.cli import cli
from src.cli.session_reader import assemble_session, parse_since
from src.host.costs import CostTracker
from src.host.intent import IntentStore
from src.host.orchestration import Tasks
from src.host.traces import TraceStore
from src.shared.trace import current_trace_id

TRACE_A = "tr_aaaaaaaaaaaa"
TRACE_B = "tr_bbbbbbbbbbbb"


def _seed(data_dir: str) -> None:
    """Write rows for TRACE_A and TRACE_B across all four stores."""
    intent = IntentStore(db_path=os.path.join(data_dir, "intent.db"))
    traces = TraceStore(db_path=os.path.join(data_dir, "traces.db"))
    tasks = Tasks(os.path.join(data_dir, "tasks.db"))
    costs = CostTracker(db_path=os.path.join(data_dir, "costs.db"))

    # TRACE_A — a full human-rooted session.
    intent.record(
        trace_id=TRACE_A, origin_kind="human", origin_channel="dashboard",
        origin_user="alice", agent="alpha", message="build me a sales report",
    )
    traces.record(TRACE_A, "mesh", "alpha", "chat", detail="dispatch to alpha")
    traces.record(TRACE_A, "mesh", "alpha", "llm_call", detail="opus call", duration_ms=1200)

    token = current_trace_id.set(TRACE_A)
    try:
        tasks.create(creator="alice", assignee="alpha", title="Sales report",
                     origin={"kind": "human", "channel": "dashboard", "user": "alice"})
        costs.track("alpha", "anthropic/claude-opus-4-8", 1000, 500)
        costs.track("alpha", "anthropic/claude-opus-4-8", 200, 100)
    finally:
        current_trace_id.reset(token)

    # TRACE_B — a different session; must be excluded from TRACE_A's view.
    intent.record(
        trace_id=TRACE_B, origin_kind="human", origin_channel="telegram",
        origin_user="bob", agent="beta", message="unrelated request",
    )
    traces.record(TRACE_B, "mesh", "beta", "chat", detail="other dispatch")
    token = current_trace_id.set(TRACE_B)
    try:
        tasks.create(creator="bob", assignee="beta", title="Other task",
                     origin={"kind": "human", "channel": "telegram", "user": "bob"})
        costs.track("beta", "anthropic/claude-opus-4-8", 50, 50)
    finally:
        current_trace_id.reset(token)

    intent.close()
    traces.close()
    tasks.close()
    costs.close()


class _Dir:
    def __init__(self):
        self.path = tempfile.mkdtemp()

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)


class TestParseSince:
    def test_duration_and_today_and_iso(self):
        import time as _t
        now = _t.time()
        assert parse_since("1h") <= now - 3500
        assert parse_since("2d") <= now - (2 * 86400) + 5
        assert parse_since("today") <= now
        assert parse_since("2020-01-01") < now
        # Empty / garbage → 7-day default (within a small window).
        default = now - (7 * 86400)
        assert abs(parse_since("") - default) < 5
        assert abs(parse_since("nonsense") - default) < 5


class TestAssemble:
    def setup_method(self):
        self.d = _Dir()
        _seed(self.d.path)

    def teardown_method(self):
        self.d.cleanup()

    def test_session_assembles_all_layers(self):
        s = assemble_session(self.d.path, TRACE_A)
        assert s["found"] is True
        assert len(s["intent"]) == 1
        assert s["intent"][0]["message"] == "build me a sales report"
        assert s["intent"][0]["origin_user"] == "alice"
        # two trace events
        assert len(s["actions"]) == 2
        # one task
        assert len(s["tasks"]) == 1
        assert s["tasks"][0]["assignee"] == "alpha"
        # cost rollup summed across the two usage rows
        assert s["cost"]["total_tokens"] == 1800
        assert s["cost"]["total_cost_usd"] > 0
        assert s["cost"]["by_model"][0]["model"] == "anthropic/claude-opus-4-8"

    def test_other_trace_excluded(self):
        s = assemble_session(self.d.path, TRACE_A)
        users = {i["origin_user"] for i in s["intent"]}
        assert "bob" not in users
        assignees = {t["assignee"] for t in s["tasks"]}
        assert "beta" not in assignees

    def test_not_found(self):
        s = assemble_session(self.d.path, "tr_does_not_exist")
        assert s["found"] is False
        assert s["intent"] == []
        assert s["actions"] == []
        assert s["tasks"] == []
        assert s["cost"]["total_tokens"] == 0


class TestGracefulDegradation:
    def setup_method(self):
        self.d = _Dir()
        _seed(self.d.path)

    def teardown_method(self):
        self.d.cleanup()

    def test_missing_intent_db_still_assembles_other_layers(self):
        os.remove(os.path.join(self.d.path, "intent.db"))
        s = assemble_session(self.d.path, TRACE_A)
        # intent gone, but actions/tasks/cost still present → found is True
        assert s["intent"] == []
        assert len(s["actions"]) == 2
        assert len(s["tasks"]) == 1
        assert s["found"] is True

    def test_all_dbs_missing_does_not_crash(self):
        empty = tempfile.mkdtemp()
        try:
            s = assemble_session(empty, TRACE_A)
            assert s["found"] is False
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestReadOnly:
    def test_reader_does_not_mutate_or_create_dbs(self):
        d = _Dir()
        try:
            _seed(d.path)
            # mtimes + size before
            before = {}
            for name in ("intent.db", "traces.db", "tasks.db", "costs.db"):
                p = os.path.join(d.path, name)
                st = os.stat(p)
                before[name] = (st.st_mtime_ns, st.st_size)
            # Run a missing-db scenario too: ensure no file is created.
            assemble_session(d.path, TRACE_A)
            assemble_session(d.path, "tr_missing")
            for name, (mtime, size) in before.items():
                st = os.stat(os.path.join(d.path, name))
                assert st.st_mtime_ns == mtime, f"{name} mtime changed"
                assert st.st_size == size, f"{name} size changed"
            # The reader must NOT create a file for a non-existent DB.
            assemble_session(d.path, TRACE_A)
            assert not os.path.exists(os.path.join(d.path, "nonexistent.db"))
        finally:
            d.cleanup()

    def test_missing_db_not_created_by_reader(self):
        empty = tempfile.mkdtemp()
        try:
            assemble_session(empty, TRACE_A)
            # mode=ro must never create the files
            assert os.listdir(empty) == []
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestCli:
    def setup_method(self):
        self.d = _Dir()
        _seed(self.d.path)
        self.runner = CliRunner()

    def teardown_method(self):
        self.d.cleanup()

    def test_session_text_output(self):
        r = self.runner.invoke(cli, ["session", TRACE_A, "--data-dir", self.d.path])
        assert r.exit_code == 0, r.output
        assert "build me a sales report" in r.output
        assert "alice" in r.output
        assert "COST" in r.output
        assert "container-local" in r.output  # the surfaced caveat

    def test_session_json_shape(self):
        r = self.runner.invoke(cli, ["session", TRACE_A, "--data-dir", self.d.path, "--json"])
        assert r.exit_code == 0, r.output
        obj = json.loads(r.output)
        assert obj["trace_id"] == TRACE_A
        assert obj["found"] is True
        assert {"intent", "actions", "tasks", "cost"} <= set(obj.keys())
        assert obj["cost"]["total_tokens"] == 1800

    def test_session_not_found_exits_zero(self):
        r = self.runner.invoke(cli, ["session", "tr_missing", "--data-dir", self.d.path])
        assert r.exit_code == 0
        assert "No session found" in r.output

    def test_sessions_lists_newest_first(self):
        r = self.runner.invoke(
            cli, ["sessions", "--since", "1d", "--data-dir", self.d.path],
        )
        assert r.exit_code == 0, r.output
        # TRACE_B was seeded last → newest → appears before TRACE_A
        assert r.output.index(TRACE_B[:16]) < r.output.index(TRACE_A[:16])

    def test_sessions_filter_by_user(self):
        r = self.runner.invoke(
            cli, ["sessions", "--since", "1d", "--user", "alice", "--data-dir", self.d.path],
        )
        assert r.exit_code == 0, r.output
        assert TRACE_A[:16] in r.output
        assert TRACE_B[:16] not in r.output

    def test_sessions_filter_by_agent(self):
        r = self.runner.invoke(
            cli, ["sessions", "--since", "1d", "--agent", "beta", "--data-dir", self.d.path],
        )
        assert r.exit_code == 0, r.output
        assert TRACE_B[:16] in r.output
        assert TRACE_A[:16] not in r.output

    def test_sessions_limit(self):
        r = self.runner.invoke(
            cli, ["sessions", "--since", "1d", "--limit", "1", "--data-dir", self.d.path],
        )
        assert r.exit_code == 0, r.output
        # newest only
        assert TRACE_B[:16] in r.output
        assert TRACE_A[:16] not in r.output

    def test_sessions_json_array(self):
        r = self.runner.invoke(
            cli, ["sessions", "--since", "1d", "--data-dir", self.d.path, "--json"],
        )
        assert r.exit_code == 0, r.output
        obj = json.loads(r.output)
        assert "sessions" in obj
        traces = {s["trace_id"] for s in obj["sessions"]}
        assert TRACE_A in traces and TRACE_B in traces
        # outcome + cost surfaced in the summary
        a = next(s for s in obj["sessions"] if s["trace_id"] == TRACE_A)
        assert a["origin_user"] == "alice"
        assert a["total_tokens"] == 1800
