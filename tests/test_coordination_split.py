"""B2 coordination-vs-work spend split (Phase-3 unit 1, plan §8 #11).

An LLM call is COORDINATION iff its REQUESTED model equals the
deployment-configured ``llm.utility_model`` (prefix-insensitive,
mesh-held config — never container headers). Coordination traffic lands
in the usage ledger with ``kind='coordination'``, skips the per-agent
work preflight, the team envelope AND the ask billing redirect, and is
gated only by its own daily cap (``OPENLEGION_COORDINATION_DAILY_CAP_USD``,
0 = tier blocked). Enforcement reads filter ``kind='work'``; reporting
surfaces stay spend-inclusive.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.teams import TeamStore
from src.shared.limits import coordination_daily_cap_usd

WORK_MODEL = "openai/gpt-4o"
UTILITY_MODEL = "openai/gpt-4o-mini"


def _vault_and_tracker(tmp_path, monkeypatch, *, utility_model=UTILITY_MODEL, team_store=None):
    from src.host.credentials import CredentialVault

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
    if team_store is not None:
        tracker.set_team_store(team_store)
    vault = CredentialVault(cost_tracker=tracker)
    vault.set_utility_model_provider(lambda: utility_model)
    return vault, tracker


def _chat_request(model):
    from src.shared.types import APIProxyRequest

    return APIProxyRequest(
        service="llm",
        action="chat",
        params={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def _mock_acompletion_ok(model, messages, api_key, **kwargs):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].message.tool_calls = None
    resp.usage = MagicMock()
    resp.usage.total_tokens = 10
    resp.usage.prompt_tokens = 6
    resp.usage.completion_tokens = 4
    return resp


async def _mock_acompletion(model, messages, api_key, **kwargs):
    return _mock_acompletion_ok(model, messages, api_key, **kwargs)


def _usage_rows(tracker, agent):
    return tracker.db.execute(
        "SELECT model, kind, cost_usd FROM usage WHERE agent = ? ORDER BY id",
        (agent,),
    ).fetchall()


class TestClassification:
    """Model-keyed kind stamping at the non-streaming proxy chokepoint."""

    @pytest.mark.asyncio
    async def test_utility_model_call_tracked_as_coordination(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        rows = _usage_rows(tracker, "worker")
        assert [r[1] for r in rows] == ["coordination"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_non_utility_model_call_tracked_as_work(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(WORK_MODEL), agent_id="worker")
        assert result.success is True
        rows = _usage_rows(tracker, "worker")
        assert [r[1] for r in rows] == ["work"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_prefix_insensitive_match(self, tmp_path, monkeypatch):
        """Configured bare 'gpt-4o-mini' matches requested 'openai/gpt-4o-mini'."""
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch, utility_model="gpt-4o-mini")
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["coordination"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_unset_utility_model_means_no_coordination(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch, utility_model="")
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["work"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_unwired_provider_means_no_coordination(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        vault.set_utility_model_provider(None)
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["work"]
        tracker.close()


class TestWorkLedgerIsolation:
    """B2's whole point: an exhausted work ledger never blocks coordination."""

    @pytest.mark.asyncio
    async def test_work_budget_exhausted_utility_call_still_allowed(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        tracker.set_budget("worker", 0.0, 0.0)

        # Work call is hard-blocked (existing contract).
        result = await vault.execute_api_call(_chat_request(WORK_MODEL), agent_id="worker")
        assert result.success is False
        assert result.error.startswith("Budget exceeded")

        # Utility-model call sails through on its own ledger.
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["coordination"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_team_envelope_exhausted_coordination_still_allowed(self, tmp_path, monkeypatch):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        store.add_member("alpha", "burner")
        store.set_budget("alpha", 0.001, None)
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch, team_store=store)
        tracker.track("burner", WORK_MODEL, 100_000, 50_000)

        # Work call blocked by the envelope (existing contract).
        result = await vault.execute_api_call(_chat_request(WORK_MODEL), agent_id="worker")
        assert result.success is False
        assert "Team budget exceeded" in result.error

        # Coordination is exempt from the envelope.
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        tracker.close()

    def test_coordination_spend_excluded_from_envelope_sums(self, tmp_path, monkeypatch):
        """_members_spend_totals (feeds team_envelope_check) is work-only."""
        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        store.set_budget("alpha", 0.001, None)
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.set_team_store(store)
        # Burn far past the envelope — but on the coordination ledger.
        tracker.track("worker", WORK_MODEL, 100_000, 50_000, kind="coordination")

        envelope = tracker.team_envelope_check("worker", "openai/gpt-4o-mini", estimated_tokens=1)
        assert envelope["allowed"] is True
        assert envelope["daily_used"] == 0.0
        tracker.close()


class TestCoordinationCapEnforcement:
    @pytest.mark.asyncio
    async def test_cap_exceeded_blocks_with_distinct_error(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "0.000001")

        result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is False
        assert result.error.startswith("Coordination budget exceeded")

        # Reverse isolation: work traffic is never touched by the cap.
        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(WORK_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["work"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_zero_cap_kill_switch_blocks_coordination(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "0")

        result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is False
        assert result.error.startswith("Coordination budget exceeded")
        tracker.close()

    def test_work_spend_does_not_eat_coordination_cap(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("worker", WORK_MODEL, 1_000_000, 500_000)  # way past $2 in work spend
        check = tracker.coordination_preflight_check("worker", UTILITY_MODEL)
        assert check["allowed"] is True
        assert check["daily_used"] == 0.0
        tracker.close()

    def test_coordination_preflight_blocks_at_cap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "0.01")
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("worker", WORK_MODEL, 10_000, 5_000, kind="coordination")
        check = tracker.coordination_preflight_check("worker", UTILITY_MODEL)
        assert check["allowed"] is False
        assert check["daily_limit"] == 0.01
        tracker.close()


class TestCapResolution:
    """limits.coordination_daily_cap_usd: default / kill-switch / clamp."""

    def test_default(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        assert coordination_daily_cap_usd() == 2.0

    def test_zero_is_valid_not_clamped_up(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "0")
        assert coordination_daily_cap_usd() == 0.0

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "not-a-number")
        assert coordination_daily_cap_usd() == 2.0

    def test_clamped_high(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "500")
        assert coordination_daily_cap_usd() == 100.0

    def test_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "-3")
        assert coordination_daily_cap_usd() == 0.0


class _FakeBroker:
    """Duck-typed ask billing resolver: always names an open window."""

    def __init__(self, asker="asker"):
        self.asker = asker
        self.noted = []

    def active_billing_for(self, agent_id):
        return self.asker

    def note_billed_cost(self, agent_id, cost):
        self.noted.append((agent_id, cost))


class TestAskWindowPrecedence:
    """Coordination classification WINS over an open ask window."""

    @pytest.mark.asyncio
    async def test_coordination_bills_caller_not_asker(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        broker = _FakeBroker()
        vault.set_bill_resolver(broker)

        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(UTILITY_MODEL), agent_id="worker")
        assert result.success is True
        assert _usage_rows(tracker, "asker") == []
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["coordination"]
        assert broker.noted == []
        tracker.close()

    @pytest.mark.asyncio
    async def test_work_call_still_redirects_to_asker(self, tmp_path, monkeypatch):
        """Contrast: the ask redirect itself is unchanged for work traffic."""
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        broker = _FakeBroker()
        vault.set_bill_resolver(broker)

        with patch("litellm.acompletion", side_effect=_mock_acompletion):
            result = await vault.execute_api_call(_chat_request(WORK_MODEL), agent_id="worker")
        assert result.success is True
        assert [r[1] for r in _usage_rows(tracker, "asker")] == ["work"]
        assert _usage_rows(tracker, "worker") == []
        assert len(broker.noted) == 1
        tracker.close()


class _MockStream:
    """Async-iterable stream carrying a final ``usage`` (litellm shape)."""

    def __init__(self):
        self.usage = MagicMock()
        self.usage.total_tokens = 10
        self.usage.prompt_tokens = 6
        self.usage.completion_tokens = 4

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "streamed"
        chunk.choices[0].delta.tool_calls = None
        yield chunk


class TestStreamingPath:
    """The same split applies to the stream_llm proxy path."""

    @pytest.mark.asyncio
    async def test_stream_coordination_blocked_with_distinct_error(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        monkeypatch.setenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", "0")

        events = []
        async for event in vault.stream_llm(_chat_request(UTILITY_MODEL), agent_id="worker"):
            events.append(event)
        assert any("Coordination budget exceeded" in e for e in events)
        assert _usage_rows(tracker, "worker") == []
        tracker.close()

    @pytest.mark.asyncio
    async def test_stream_work_budget_exhausted_coordination_allowed(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        tracker.set_budget("worker", 0.0, 0.0)

        # Work stream blocked.
        events = []
        async for event in vault.stream_llm(_chat_request(WORK_MODEL), agent_id="worker"):
            events.append(event)
        assert any("Budget exceeded" in e for e in events)

        # Coordination stream succeeds and stamps the coordination kind.
        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(UTILITY_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["coordination"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_stream_non_utility_tracked_as_work(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)

        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(WORK_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["work"]
        tracker.close()

    @pytest.mark.asyncio
    async def test_stream_work_blocked_by_exhausted_team_envelope(self, tmp_path, monkeypatch):
        """M1: the team envelope is enforced on the STREAMING work fork —
        the default production transport (``llm.chat_collect`` streams).
        Regression: the envelope lived only on ``execute_api_call`` so real
        (streaming) traffic silently ignored it."""
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        store.add_member("alpha", "burner")
        store.set_budget("alpha", 0.001, None)
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch, team_store=store)
        tracker.track("burner", WORK_MODEL, 100_000, 50_000)

        # Streaming WORK call is blocked by the exhausted envelope, and never
        # opens a stream (no usage row lands).
        events = []
        async for event in vault.stream_llm(_chat_request(WORK_MODEL), agent_id="worker"):
            events.append(event)
        assert any("Team budget exceeded" in e for e in events)
        assert _usage_rows(tracker, "worker") == []

        # Negative control: coordination is exempt from the envelope even on
        # the streaming fork (B2 precedence preserved).
        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(UTILITY_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        tracker.close()

    @pytest.mark.asyncio
    async def test_stream_unset_team_envelope_does_not_block(self, tmp_path, monkeypatch):
        """M1 pin: an UNSET team envelope is UNLIMITED (B4) — the streaming
        envelope check must not over-block a team that hasn't been given a
        budget. Mirrors the sync ``test_zero_envelope_does_not_block``."""
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        # Deliberately NO set_budget — unset envelope == unlimited.
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch, team_store=store)

        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(WORK_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["work"]
        tracker.close()


class TestStreamingAskBilling:
    """M2: the ask_teammate asker-billing window applies on the STREAMING
    path too — an ask delivered to an IDLE recipient runs as a streaming
    followup-lane work turn. Mirrors ``TestAskWindowPrecedence`` (sync)."""

    @pytest.mark.asyncio
    async def test_stream_work_redirects_billing_to_asker(self, tmp_path, monkeypatch):
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        broker = _FakeBroker()
        vault.set_bill_resolver(broker)

        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(WORK_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        # The ASKER pays for the ask turn; the recipient's ledger is untouched.
        assert [r[1] for r in _usage_rows(tracker, "asker")] == ["work"]
        assert _usage_rows(tracker, "worker") == []
        # Ask-cap accrual is noted, keyed by the RECIPIENT (verified caller).
        assert len(broker.noted) == 1
        assert broker.noted[0][0] == "worker"
        tracker.close()

    @pytest.mark.asyncio
    async def test_stream_coordination_bills_caller_not_asker(self, tmp_path, monkeypatch):
        """Coordination classification WINS over an open ask window on the
        streaming fork too — the redirect is skipped, the caller pays."""
        vault, tracker = _vault_and_tracker(tmp_path, monkeypatch)
        broker = _FakeBroker()
        vault.set_bill_resolver(broker)

        async def mock_stream_acompletion(model, messages, api_key, stream=False, **kwargs):
            return _MockStream()

        with patch("litellm.acompletion", side_effect=mock_stream_acompletion):
            events = []
            async for event in vault.stream_llm(_chat_request(UTILITY_MODEL), agent_id="worker"):
                events.append(event)
        assert any("done" in e for e in events)
        assert [r[1] for r in _usage_rows(tracker, "worker")] == ["coordination"]
        assert _usage_rows(tracker, "asker") == []
        assert broker.noted == []
        tracker.close()


class TestReportingInclusive:
    """Reporting surfaces keep including coordination rows (money is money)."""

    def test_get_spend_default_includes_both_kinds(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("worker", WORK_MODEL, 1000, 500)
        tracker.track("worker", UTILITY_MODEL, 1000, 500, kind="coordination")

        all_spend = tracker.get_spend("worker", "today")
        work_spend = tracker.get_spend("worker", "today", kind="work")
        coord_spend = tracker.get_spend("worker", "today", kind="coordination")
        assert all_spend["total_tokens"] == 3000
        assert work_spend["total_tokens"] == 1500
        assert coord_spend["total_tokens"] == 1500
        # get_spend rounds each total to 4 decimals — allow one ulp of
        # rounding drift between the whole and the sum of the parts.
        assert all_spend["total_cost"] == pytest.approx(
            work_spend["total_cost"] + coord_spend["total_cost"], abs=2e-4,
        )
        tracker.close()

    def test_get_team_spend_includes_coordination(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.set_team_store(store)
        tracker.track("worker", UTILITY_MODEL, 1000, 500, kind="coordination")

        spend = tracker.get_team_spend("alpha", "today")
        assert spend["total_tokens"] == 1500
        assert spend["total_cost"] > 0
        tracker.close()

    def test_all_agents_and_by_model_include_coordination(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("worker", UTILITY_MODEL, 1000, 500, kind="coordination")

        agents = tracker.get_all_agents_spend("today")
        assert agents and agents[0]["agent"] == "worker" and agents[0]["tokens"] == 1500
        by_model = tracker.get_spend_by_model("today")
        assert any(r["model"] == UTILITY_MODEL and r["tokens"] == 1500 for r in by_model)
        tracker.close()

    def test_get_coordination_spend_breakout(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("worker", UTILITY_MODEL, 1000, 500, kind="coordination")
        tracker.track("worker", WORK_MODEL, 1000, 500)

        breakout = tracker.get_coordination_spend("worker")
        assert breakout["daily_limit"] == 2.0
        assert breakout["daily_used"] == tracker.get_spend(
            "worker", "today", kind="coordination",
        )["total_cost"]
        tracker.close()


class TestKindMigration:
    def test_kind_migration_idempotent_on_legacy_db(self, tmp_path):
        """A pre-existing usage table without ``kind`` gets the column added
        once (legacy rows read 'work'); re-opening is a no-op."""
        legacy_path = str(tmp_path / "legacy_costs.db")
        conn = sqlite3.connect(legacy_path)
        conn.executescript(
            """
            CREATE TABLE usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT (datetime('now')),
                trace_id TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO usage (agent, model, total_tokens, cost_usd) VALUES (?, ?, ?, ?)",
            ("legacy", WORK_MODEL, 100, 0.5),
        )
        conn.commit()
        conn.close()

        t1 = CostTracker(db_path=legacy_path)
        cols = {r[1] for r in t1.db.execute("PRAGMA table_info(usage)").fetchall()}
        assert "kind" in cols
        # Legacy rows take the 'work' default — their enforcement
        # semantics are unchanged by the split.
        row = t1.db.execute("SELECT kind FROM usage WHERE agent = 'legacy'").fetchone()
        assert row[0] == "work"
        assert t1.get_spend("legacy", "today", kind="work")["total_cost"] == 0.5
        t1.close()

        t2 = CostTracker(db_path=legacy_path)
        kind_cols = [r for r in t2.db.execute("PRAGMA table_info(usage)").fetchall() if r[1] == "kind"]
        assert len(kind_cols) == 1
        t2.close()


class TestIntrospectCoordinationBlock:
    def test_budget_section_includes_coordination(self, tmp_path, monkeypatch):
        import src.cli.config as _cfg
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        # Finding 4(a): the coordination sub-dict is surfaced ONLY when a
        # utility model is configured (otherwise the tier is structurally
        # inert). Configure one so this correct-semantics case holds.
        monkeypatch.setattr(
            _cfg, "_load_config",
            lambda *a, **k: {"llm": {"utility_model": UTILITY_MODEL}},
        )
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        router = MessageRouter(permissions=perms, agent_registry={})
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("alice", UTILITY_MODEL, 1000, 500, kind="coordination")

        app = create_mesh_app(bb, PubSub(), router, perms, cost_tracker=tracker)
        response = TestClient(app).get(
            "/mesh/introspect",
            params={"section": "budget"},
            headers={"X-Agent-ID": "alice"},
        )
        assert response.status_code == 200
        budget = response.json()["budget"]
        assert budget["coordination"]["daily_limit"] == 2.0
        assert budget["coordination"]["daily_used"] > 0
        # The work budget excludes the coordination row (enforcement split).
        assert budget["daily_used"] == 0.0
        bb.close()
        tracker.close()

    def test_no_utility_model_omits_coordination(self, tmp_path, monkeypatch):
        """Finding 4(a): with no ``llm.utility_model``, nothing can classify
        as coordination (every call bills WORK), so the introspect budget
        omits the ``coordination`` sub-dict entirely — no misleading
        permanently-$0.00 tier in the runtime context."""
        import src.cli.config as _cfg
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        monkeypatch.delenv("OPENLEGION_COORDINATION_DAILY_CAP_USD", raising=False)
        monkeypatch.setattr(_cfg, "_load_config", lambda *a, **k: {"llm": {}})
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        router = MessageRouter(permissions=perms, agent_registry={})
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.track("alice", UTILITY_MODEL, 1000, 500, kind="coordination")

        app = create_mesh_app(bb, PubSub(), router, perms, cost_tracker=tracker)
        response = TestClient(app).get(
            "/mesh/introspect",
            params={"section": "budget"},
            headers={"X-Agent-ID": "alice"},
        )
        assert response.status_code == 200
        budget = response.json()["budget"]
        assert "coordination" not in budget
        bb.close()
        tracker.close()
