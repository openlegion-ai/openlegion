"""Tests for CLI commands and REPL functionality."""

import json
import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.cli import cli


class TestVersion:
    def test_version_flag(self):
        """--version outputs version string and exits 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "." in result.output


class _MockCtx:
    """Lightweight mock of RuntimeContext for REPL tests."""

    def __init__(self, agent_urls=None, *, blackboard=None, lane_manager=None,
                 cron_scheduler=None):
        self.agent_urls = agent_urls or {}
        self.cfg = {"mesh": {"port": 8420}, "agents": {}, "llm": {"default_model": "openai/gpt-4o-mini"}}
        # In-memory team store so REPL team commands never touch the
        # on-disk data/teams.db during tests.
        from src.host.teams import TeamStore
        self.teams_store = TeamStore(db_path=":memory:")
        self.blackboard = blackboard
        self.lane_manager = lane_manager
        self.cron_scheduler = cron_scheduler
        self.cost_tracker = None
        self.credential_vault = None
        self.runtime = None
        self.transport = None
        self.router = None
        self.permissions = None
        self.pubsub = None
        self.health_monitor = None
        self.event_bus = None
        self.trace_store = None
        # Set up a real event loop in a background thread for async cron methods
        import asyncio
        import threading
        self._dispatch_loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._dispatch_loop.run_forever, daemon=True)
        t.start()

    @property
    def agents(self):
        return self.agent_urls

    @property
    def dispatch_loop(self):
        return self._dispatch_loop


class TestCostsBarChart:
    def test_bar_chart_function(self):
        """Bar chart renders correctly."""
        from src.cli.repl import _bar

        assert len(_bar(5, 10)) == 20
        assert "\u2588" in _bar(5, 10)
        assert _bar(0, 10) == "\u2591" * 20
        assert _bar(10, 10) == "\u2588" * 20

    def test_bar_zero_max(self):
        """Zero max value returns all empty blocks."""
        from src.cli.repl import _bar

        assert _bar(5, 0) == "\u2591" * 20

    def test_bar_custom_width(self):
        """Custom width is respected."""
        from src.cli.repl import _bar

        assert len(_bar(5, 10, width=10)) == 10

    def test_bar_negative_value(self):
        """_bar handles negative values by returning empty bar."""
        from src.cli.repl import _bar

        assert _bar(-5, 10) == "\u2591" * 20


class TestREPLZeroAgents:
    """REPLSession with no agents configured."""

    def test_init_sets_current_none(self):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        assert repl.current is None

    def test_help_works_with_no_agents(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_help("")
        out = capsys.readouterr().out
        assert "/add" in out
        assert "/quit" in out
        assert "/blackboard" in out

    def test_status_with_no_agents(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_status("")
        out = capsys.readouterr().out
        assert "No agents" in out
        assert "/add" in out

    def test_broadcast_with_no_agents(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_broadcast("hello everyone")
        out = capsys.readouterr().out
        assert "No agents" in out

    def test_steer_with_no_current_agent(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_steer("change direction")
        out = capsys.readouterr().out
        assert "No active agent" in out

    def test_reset_with_no_current_agent(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_reset("")
        out = capsys.readouterr().out
        assert "No active agent" in out

    def test_edit_with_no_current_agent(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_edit("")
        out = capsys.readouterr().out
        assert "No active agent" in out

    def test_commands_dispatch_with_no_current(self, capsys):
        """Slash commands should work even when self.current is None."""
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        # Dispatch /status — should not crash, should print "No agents"
        result = repl._dispatch_command("/status")
        assert result is None  # not "quit"
        out = capsys.readouterr().out
        assert "No agents" in out

    def test_quit_works_with_no_current(self):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        result = repl._dispatch_command("/quit")
        assert result == "quit"


class TestREPLBlackboard:
    def _make_repl(self, bb):
        from src.cli.repl import REPLSession
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"}, blackboard=bb)
        return REPLSession(ctx)

    def test_list_empty(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        repl = self._make_repl(bb)
        repl._cmd_blackboard("")
        out = capsys.readouterr().out
        assert "No entries" in out
        bb.close()

    def test_set_and_get(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        repl = self._make_repl(bb)

        repl._cmd_blackboard('set test/key {"val": 1}')
        out = capsys.readouterr().out
        assert "Written: test/key" in out

        repl._cmd_blackboard("get test/key")
        out = capsys.readouterr().out
        assert "test/key" in out
        assert '"val": 1' in out or '"val":1' in out
        assert "Written by: cli" in out
        bb.close()

    def test_list_with_prefix(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("tasks/a", {"done": False}, written_by="agent1")
        bb.write("tasks/b", {"done": True}, written_by="agent2")
        bb.write("context/x", {"info": "hi"}, written_by="agent1")
        repl = self._make_repl(bb)

        repl._cmd_blackboard("list tasks/")
        out = capsys.readouterr().out
        assert "tasks/a" in out
        assert "tasks/b" in out
        assert "context/x" not in out
        bb.close()

    def test_prefix_without_subcommand(self, tmp_path, capsys):
        """Typing `/blackboard tasks/` should list entries, not show usage."""
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("tasks/a", {"done": False}, written_by="agent1")
        repl = self._make_repl(bb)

        repl._cmd_blackboard("tasks/")
        out = capsys.readouterr().out
        assert "tasks/a" in out
        bb.close()

    def test_delete(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("test/del", {"x": 1}, written_by="cli")
        repl = self._make_repl(bb)

        with patch("src.cli.repl.click.confirm", return_value=True):
            repl._cmd_blackboard("del test/del")
        out = capsys.readouterr().out
        assert "Deleted: test/del" in out
        assert bb.read("test/del") is None
        bb.close()

    def test_delete_cancelled(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("test/keep", {"x": 1}, written_by="cli")
        repl = self._make_repl(bb)

        with patch("src.cli.repl.click.confirm", return_value=False):
            repl._cmd_blackboard("del test/keep")
        out = capsys.readouterr().out
        assert "Deleted" not in out
        assert bb.read("test/keep") is not None
        bb.close()

    def test_delete_history_blocked(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("history/x", {"x": 1}, written_by="cli")
        repl = self._make_repl(bb)

        with patch("src.cli.repl.click.confirm", return_value=True):
            repl._cmd_blackboard("del history/x")
        out = capsys.readouterr().out
        assert "Error" in out
        bb.close()

    def test_get_not_found(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        repl = self._make_repl(bb)
        repl._cmd_blackboard("get nonexistent")
        out = capsys.readouterr().out
        assert "not found" in out
        bb.close()

    def test_set_invalid_json(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        repl = self._make_repl(bb)
        repl._cmd_blackboard("set test/key {bad json}")
        out = capsys.readouterr().out
        assert "Invalid JSON" in out
        bb.close()

    def test_set_wraps_non_dict(self, tmp_path, capsys):
        """Non-dict JSON (e.g. a bare string) is wrapped in {"value": ...}."""
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        repl = self._make_repl(bb)
        repl._cmd_blackboard('set test/key "hello"')
        capsys.readouterr()

        entry = bb.read("test/key")
        assert entry is not None
        assert entry.value == {"value": "hello"}
        bb.close()

    def test_not_available(self, capsys):
        from src.cli.repl import REPLSession
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        repl = REPLSession(ctx)
        repl._cmd_blackboard("")
        out = capsys.readouterr().out
        assert "not available" in out


class TestREPLQueue:
    def test_queue_shows_status(self, capsys):
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession
        lm = MagicMock()
        lm.get_status.return_value = {
            "agent1": {"busy": True, "queued": 2},
            "agent2": {"busy": False, "queued": 0},
        }
        ctx = _MockCtx(agent_urls={"agent1": "url1", "agent2": "url2"}, lane_manager=lm)
        repl = REPLSession(ctx)

        repl._cmd_queue("")
        out = capsys.readouterr().out
        assert "agent1" in out
        assert "busy" in out
        assert "2 queued" in out
        assert "agent2" in out
        assert "idle" in out

    def test_queue_empty(self, capsys):
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession
        lm = MagicMock()
        lm.get_status.return_value = {}
        ctx = _MockCtx(lane_manager=lm)
        repl = REPLSession(ctx)

        repl._cmd_queue("")
        out = capsys.readouterr().out
        assert "No agent queues" in out

    def test_queue_not_available(self, capsys):
        from src.cli.repl import REPLSession
        ctx = _MockCtx()
        repl = REPLSession(ctx)
        repl._cmd_queue("")
        out = capsys.readouterr().out
        assert "not available" in out


class TestREPLCronExtended:
    def test_pause_job(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check")
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron(f"pause {job.id}")
        out = capsys.readouterr().out
        assert "Paused" in out
        assert not sched.jobs[job.id].enabled

    def test_resume_job(self, tmp_path, capsys):
        import asyncio

        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check")
        asyncio.run(sched.pause_job(job.id))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron(f"resume {job.id}")
        out = capsys.readouterr().out
        assert "Resumed" in out
        assert sched.jobs[job.id].enabled

    def test_pause_not_found(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("pause nonexistent_id")
        out = capsys.readouterr().out
        assert "not found" in out

    def test_run_not_found(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("run nonexistent_id")
        out = capsys.readouterr().out
        assert "not found" in out

    def test_pause_missing_id(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("pause")
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_resume_missing_id(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("resume")
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_run_missing_id(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("run")
        out = capsys.readouterr().out
        assert "Usage:" in out


class TestREPLBareTextNoAgent:
    """Bare text input gives feedback when no active agent."""

    def test_bare_text_shows_feedback(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        # _parse_input with no current agent returns (None, "hello")
        target, msg = repl._parse_input("hello world")
        assert target is None
        assert msg == "hello world"

    def test_at_mention_unknown_agent(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        target, msg = repl._parse_input("@ghost do something")
        assert target is None
        assert msg == ""
        out = capsys.readouterr().out
        assert "Unknown agent" in out

    def test_at_mention_valid_agent_no_current(self, capsys):
        """@mention a valid agent should work even with no current agent."""
        from src.cli.repl import REPLSession
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        repl = REPLSession(ctx)
        # Force current to None (simulating zero-agent start then /add)
        repl.current = None
        target, msg = repl._parse_input("@bot do stuff")
        assert target == "bot"
        assert msg == "do stuff"


class TestREPLUseNoAgents:
    """The /use command with no agents available."""

    def test_use_no_agents(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_use("")
        out = capsys.readouterr().out
        assert "No agents available" in out
        assert "/add" in out

    def test_use_no_agents_with_name(self, capsys):
        from src.cli.repl import REPLSession
        repl = REPLSession(_MockCtx())
        repl._cmd_use("ghost")
        out = capsys.readouterr().out
        assert "No agents available" in out


class TestREPLBlackboardSetMissingArgs:
    """Blackboard set with insufficient arguments."""

    def test_set_missing_value(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"}, blackboard=bb)
        repl = REPLSession(ctx)

        repl._cmd_blackboard("set mykey")
        out = capsys.readouterr().out
        assert "Usage:" in out
        bb.close()

    def test_set_missing_key_and_value(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"}, blackboard=bb)
        repl = REPLSession(ctx)

        repl._cmd_blackboard("set")
        out = capsys.readouterr().out
        assert "Usage:" in out
        bb.close()


class TestREPLCronListFormatting:
    """Cron list shows last_run and proper formatting."""

    def test_cron_list_shows_last_run(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check status")
        # Simulate a past execution
        sched.jobs[job.id].last_run = "2026-02-22T10:30:00+00:00"
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("")
        out = capsys.readouterr().out
        assert "Last run" in out
        assert "2026-02-22 10:30:00" in out
        assert "bot" in out
        assert "every 5m" in out
        assert "check status" in out

    def test_cron_list_never_run(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        sched.add_job(agent="bot", schedule="every 1h", message="sweep")
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("")
        out = capsys.readouterr().out
        assert "never" in out

    def test_cron_list_paused_job(self, tmp_path, capsys):
        import asyncio

        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 10m", message="ping")
        asyncio.run(sched.pause_job(job.id))
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("")
        out = capsys.readouterr().out
        assert "paused" in out


class TestREPLBlackboardGetMultiline:
    """Blackboard get with multiline values are properly indented."""

    def test_multiline_value_indented(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("test/multi", {"a": 1, "b": {"nested": "value"}}, written_by="cli")
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"}, blackboard=bb)
        repl = REPLSession(ctx)

        repl._cmd_blackboard("get test/multi")
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        # Find lines after "Value:" — continuation lines should be indented
        value_started = False
        continuation_lines = []
        for line in lines:
            if "Value:" in line:
                value_started = True
                continue
            if value_started and line.strip():
                continuation_lines.append(line)
        # All continuation lines should start with spaces (indented)
        for cl in continuation_lines:
            assert cl.startswith("  "), f"Continuation line not indented: {cl!r}"
        bb.close()


class TestChatNoMesh:
    def test_chat_fails_gracefully_when_mesh_not_running(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "testbot", "--port", "19999"])
        assert result.exit_code == 1
        assert "not running" in result.output


class TestPermissionsDefault:
    def test_new_agent_gets_default_permissions(self):
        """Verify the permissions module falls back to 'default' template."""
        from src.host.permissions import PermissionMatrix

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "permissions": {
                    "default": {
                        "can_message": ["mesh"],
                        "allowed_apis": ["llm"],
                        "blackboard_read": ["context/*"],
                    }
                }
            }, f)
            f.flush()
            pm = PermissionMatrix(config_path=f.name)

        perms = pm.get_permissions("some_unknown_agent")
        assert perms.agent_id == "some_unknown_agent"
        assert "mesh" in perms.can_message
        assert "llm" in perms.allowed_apis
        assert pm.can_use_api("some_unknown_agent", "llm")
        os.unlink(f.name)


class TestEnsureAllAgentPermissions:
    def test_backfills_missing_agents(self, tmp_path):
        """_ensure_all_agent_permissions adds permissions for agents not yet in
        permissions.json and forward-migrates missing flags on existing ones."""
        from unittest.mock import patch

        from src.cli.config import _ensure_all_agent_permissions

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("agents:\n  alice:\n    role: helper\n  bob:\n    role: coder\n")
        perms_file = tmp_path / "permissions.json"
        # alice exists but predates can_manage_cron; bob is absent entirely.
        perms_file.write_text(json.dumps({"permissions": {"alice": {"can_message": ["mesh"]}}}))

        with patch("src.cli.config.AGENTS_FILE", agents_yaml), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())
        # Both agents present
        assert "alice" in perms["permissions"]
        assert "bob" in perms["permissions"]
        # New agent gets full defaults
        assert perms["permissions"]["bob"]["allowed_credentials"] == ["*"]
        assert perms["permissions"]["bob"]["can_manage_cron"] is True
        # Existing agent gets forward-migrated flag
        assert perms["permissions"]["alice"]["can_manage_cron"] is True

    def test_forward_migrates_can_manage_cron_for_existing_agents(self, tmp_path):
        """Agents already in permissions.json but missing can_manage_cron get it added."""
        from unittest.mock import patch

        from src.cli.config import _ensure_all_agent_permissions

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("agents:\n  charlie:\n    role: helper\n")
        perms_file = tmp_path / "permissions.json"
        # charlie exists with full-looking perms but no can_manage_cron (legacy entry)
        legacy = {
            "can_message": ["mesh"],
            "allowed_credentials": ["*"],
            "allowed_apis": ["llm"],
        }
        perms_file.write_text(json.dumps({"permissions": {"charlie": legacy}}))

        with patch("src.cli.config.AGENTS_FILE", agents_yaml), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["charlie"]["can_manage_cron"] is True
        # Existing keys are preserved
        assert perms["permissions"]["charlie"]["allowed_credentials"] == ["*"]

    def test_idempotent_when_flag_already_present(self, tmp_path):
        """Running _ensure_all_agent_permissions twice does not change an
        already-migrated permissions file."""
        from unittest.mock import patch

        from src.cli.config import _ensure_all_agent_permissions

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("agents:\n  dave:\n    role: helper\n")
        perms_file = tmp_path / "permissions.json"
        already_migrated = {
            "can_message": ["mesh"],
            "can_manage_cron": True,
        }
        perms_file.write_text(json.dumps({"permissions": {"dave": already_migrated}}))

        with patch("src.cli.config.AGENTS_FILE", agents_yaml), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _ensure_all_agent_permissions()
            _ensure_all_agent_permissions()  # second run should be a no-op write

        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["dave"]["can_manage_cron"] is True

    def test_default_entry_is_not_migrated(self, tmp_path):
        """The 'default' key in permissions.json is a template, not an agent;
        it must not be touched by the forward-migration loop."""
        from unittest.mock import patch

        from src.cli.config import _ensure_all_agent_permissions

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("agents: {}\n")
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {"default": {"can_message": ["*"]}}}))

        with patch("src.cli.config.AGENTS_FILE", agents_yaml), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())
        # 'default' must not have been modified
        assert "can_manage_cron" not in perms["permissions"]["default"]


class TestAddAgentPermissionsDefaults:
    def test_new_agent_gets_can_manage_cron(self, tmp_path):
        """_add_agent_permissions sets can_manage_cron:True for new agents."""
        from unittest.mock import patch

        from src.cli.config import _add_agent_permissions

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {}}))

        with patch("src.cli.config.AGENTS_FILE", tmp_path / "agents.yaml"), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_agent_permissions("new_agent")

        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["new_agent"]["can_manage_cron"] is True


class TestAddAgentToConfigInitialInstructions:
    def test_initial_instructions_persisted(self, tmp_path):
        """_add_agent_to_config writes initial_instructions to agents.yaml when provided."""
        from src.cli.config import _add_agent_to_config

        agents_file = tmp_path / "agents.yaml"

        with patch("src.cli.config.AGENTS_FILE", agents_file):
            _add_agent_to_config(
                name="writer", role="content", model="openai/gpt-4o-mini",
                initial_instructions="You write blog posts.",
            )

        data = yaml.safe_load(agents_file.read_text())
        assert data["agents"]["writer"]["initial_instructions"] == "You write blog posts."

    def test_no_initial_instructions_omitted(self, tmp_path):
        """_add_agent_to_config omits initial_instructions when empty."""
        from src.cli.config import _add_agent_to_config

        agents_file = tmp_path / "agents.yaml"

        with patch("src.cli.config.AGENTS_FILE", agents_file):
            _add_agent_to_config(
                name="helper", role="assistant", model="openai/gpt-4o-mini",
            )

        data = yaml.safe_load(agents_file.read_text())
        assert "initial_instructions" not in data["agents"]["helper"]


class TestREPLProjectCommand:
    def _make_repl(self, tmp_path):
        from src.cli.repl import REPLSession
        from src.host.teams import TeamStore

        ctx = _MockCtx(agent_urls={"bot1": "http://bot1:8400", "bot2": "http://bot2:8400"})
        ctx.cfg["agents"] = {"bot1": {"role": "a"}, "bot2": {"role": "b"}}
        ctx.teams_store = TeamStore(db_path=":memory:")
        ctx.teams_store.create_team("alpha", description="Alpha project")
        ctx.teams_store.add_member("alpha", "bot1")

        repl = REPLSession(ctx)
        return repl

    def test_project_list(self, tmp_path, capsys):
        repl = self._make_repl(tmp_path)
        repl._cmd_team("list")
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "bot1" in out
        assert "bot2" in out  # standalone

    def test_project_use_and_clear(self, tmp_path, capsys):
        repl = self._make_repl(tmp_path)
        repl._cmd_team("use alpha")
        assert repl._active_team == "alpha"

        repl._cmd_team("use none")
        assert repl._active_team is None

    def test_project_use_unknown(self, tmp_path, capsys):
        repl = self._make_repl(tmp_path)
        repl._cmd_team("use nonexistent")
        out = capsys.readouterr().out
        assert "Unknown team" in out
        assert repl._active_team is None

    def test_project_info(self, tmp_path, capsys):
        repl = self._make_repl(tmp_path)
        repl._cmd_team("info alpha")
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "Alpha project" in out


class TestREPLBlackboardProjectScoping:
    """Blackboard commands are scoped to active project."""

    def _make_repl_with_bb(self, tmp_path):
        from src.cli.repl import REPLSession
        from src.host.mesh import Blackboard

        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        ctx = _MockCtx(agent_urls={"bot1": "http://bot1:8400"}, blackboard=bb)
        repl = REPLSession(ctx)
        return repl, bb

    def test_set_and_get_with_project(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        repl._active_team = "alpha"

        repl._cmd_blackboard('set mykey {"val": 1}')
        out = capsys.readouterr().out
        assert "teams/alpha/mykey" in out

        repl._cmd_blackboard("get mykey")
        out = capsys.readouterr().out
        assert "teams/alpha/mykey" in out
        bb.close()

    def test_list_with_project_scoping(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        bb.write("teams/alpha/ctx", {"v": 1}, written_by="test")
        bb.write("global/other", {"v": 2}, written_by="test")

        repl._active_team = "alpha"
        repl._cmd_blackboard("list")
        out = capsys.readouterr().out
        assert "teams/alpha/ctx" in out
        assert "global/other" not in out
        bb.close()

    def test_list_bypass_with_all_flag(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        bb.write("teams/alpha/ctx", {"v": 1}, written_by="test")
        bb.write("global/other", {"v": 2}, written_by="test")

        repl._active_team = "alpha"
        repl._cmd_blackboard("list --all")
        out = capsys.readouterr().out
        assert "teams/alpha/ctx" in out
        assert "global/other" in out
        bb.close()


class TestJsonOutput:
    def test_status_json(self, tmp_path):
        """status --json outputs valid JSON with mesh_online key."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"bot1": {"role": "test", "model": "openai/gpt-4.1"}}
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.TEAMS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["status", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "mesh_online" in data
            assert "agents" in data


class TestREPLCompleter:
    def test_completer_commands(self):
        """Tab completer completes /st to /status and /steer."""
        from unittest.mock import MagicMock

        from src.cli.repl import _REPLCompleter

        session = MagicMock()
        session.ctx.agents = {
            "researcher": "http://localhost:8401",
            "coder": "http://localhost:8402",
        }
        session._commands = {
            "/status": None,
            "/steer": None,
            "/quit": None,
            "/help": None,
            "/use": None,
            "/add": None,
        }

        completer = _REPLCompleter(session)

        import readline

        with patch.object(readline, "get_line_buffer", return_value="/st"):
            result0 = completer.complete("/st", 0)
            assert result0 is not None
            assert result0.startswith("/st")
            # Should match /status and /steer
            result1 = completer.complete("/st", 1)
            assert result1 is not None

    def test_completer_agents(self):
        """Tab completer completes @res to @researcher."""
        from unittest.mock import MagicMock

        from src.cli.repl import _REPLCompleter

        session = MagicMock()
        session.ctx.agents = {
            "researcher": "http://localhost:8401",
            "coder": "http://localhost:8402",
        }
        session._commands = {}

        completer = _REPLCompleter(session)

        import readline

        with patch.object(readline, "get_line_buffer", return_value="@res"):
            result = completer.complete("@res", 0)
            assert result == "@researcher "

    def test_completer_use_subcommand(self):
        """Tab completer completes agent names after /use."""
        from unittest.mock import MagicMock

        from src.cli.repl import _REPLCompleter

        session = MagicMock()
        session.ctx.agents = {
            "researcher": "http://localhost:8401",
            "coder": "http://localhost:8402",
        }
        session._commands = {}

        completer = _REPLCompleter(session)

        import readline

        with patch.object(readline, "get_line_buffer", return_value="/use c"):
            result = completer.complete("c", 0)
            assert result == "coder "

    def test_completer_no_match(self):
        """Tab completer returns None when no match."""
        from unittest.mock import MagicMock

        from src.cli.repl import _REPLCompleter

        session = MagicMock()
        session.ctx.agents = {}
        session._commands = {}

        completer = _REPLCompleter(session)

        import readline

        with patch.object(readline, "get_line_buffer", return_value="hello"):
            result = completer.complete("hello", 0)
            assert result is None


class TestVersionCommand:
    def test_version_basic(self):
        """version shows version string."""
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "OpenLegion" in result.output

    def test_version_verbose(self):
        """version -v shows Python version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["version", "-v"])
        assert result.exit_code == 0
        assert "Python" in result.output


class TestStatusEnhanced:
    def test_status_wide(self, tmp_path):
        """status --wide shows extra columns."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"bot1": {"role": "test", "model": "openai/gpt-4.1"}}
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.TEAMS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["status", "--wide"])
            assert result.exit_code == 0
            assert "Tasks" in result.output
            assert "Cost" in result.output

    def test_status_json(self, tmp_path):
        """status --json outputs valid JSON."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"bot1": {"role": "test", "model": "openai/gpt-4.1"}}
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.TEAMS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["status", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "agents" in data
            assert "mesh_online" in data


class TestREPLHistory:
    """Tests for /history command."""

    def test_history_no_agent(self, capsys):
        from src.cli.repl import REPLSession

        repl = REPLSession(_MockCtx())
        repl._cmd_history("")
        out = capsys.readouterr().out
        assert "no active agent" in out.lower()

    def test_history_unknown_agent(self, capsys):
        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        repl = REPLSession(ctx)
        repl._cmd_history("nonexistent")
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_history_shows_messages(self, capsys):
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession

        transport = MagicMock()
        transport.request_sync.return_value = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        ctx.transport = transport
        repl = REPLSession(ctx)
        repl._cmd_history("bot")
        out = capsys.readouterr().out
        assert "Hello" in out
        assert "Hi there!" in out

    def test_history_truncates_long_content(self, capsys):
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession

        transport = MagicMock()
        transport.request_sync.return_value = {
            "messages": [
                {"role": "user", "content": "A" * 200},
            ]
        }
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        ctx.transport = transport
        repl = REPLSession(ctx)
        repl._cmd_history("bot")
        out = capsys.readouterr().out
        assert "..." in out


class TestREPLLogs:
    """Tests for /logs command."""

    def test_logs_no_file(self, capsys, tmp_path):
        from src.cli.repl import REPLSession

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            repl = REPLSession(_MockCtx())
            repl._cmd_logs("")
            out = capsys.readouterr().out
            assert "no log file" in out.lower()

    def test_logs_shows_lines(self, capsys, tmp_path):
        from src.cli.repl import REPLSession

        log_file = tmp_path / ".openlegion.log"
        log_file.write_text("INFO line1\nERROR line2\nDEBUG line3\n")
        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            repl = REPLSession(_MockCtx())
            repl._cmd_logs("")
            out = capsys.readouterr().out
            assert "line1" in out
            assert "line2" in out
            assert "line3" in out

    def test_logs_level_filter(self, capsys, tmp_path):
        from src.cli.repl import REPLSession

        log_file = tmp_path / ".openlegion.log"
        log_file.write_text("INFO ok\nERROR bad\nINFO also ok\n")
        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            repl = REPLSession(_MockCtx())
            repl._cmd_logs("error")
            out = capsys.readouterr().out
            assert "bad" in out
            assert "ok" not in out


class TestCronDeleteConfirmation:
    """Cron delete requires confirmation."""

    def test_cron_delete_confirmed(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler

        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check")
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        with patch("src.cli.repl.click.confirm", return_value=True):
            repl._cmd_cron(f"del {job.id}")
        out = capsys.readouterr().out
        assert "Deleted" in out

    def test_cron_delete_cancelled(self, tmp_path, capsys):
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler

        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check")
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        with patch("src.cli.repl.click.confirm", return_value=False):
            repl._cmd_cron(f"del {job.id}")
        out = capsys.readouterr().out
        assert "Deleted" not in out
        # Job should still exist
        assert job.id in sched.jobs


class TestTracesAlias:
    """/traces dispatches to the same handler as /debug."""

    def test_traces_alias_registered(self):
        from src.cli.repl import REPLSession

        repl = REPLSession(_MockCtx())
        assert "/traces" in repl._commands
        assert repl._commands["/traces"][0] == repl._commands["/debug"][0]


class TestRestartCommand:
    """/restart command in REPL."""

    def test_restart_no_agent(self, capsys):
        from src.cli.repl import REPLSession

        repl = REPLSession(_MockCtx())
        repl.current = None
        repl._cmd_restart("")
        out = capsys.readouterr().out
        assert "No active agent" in out

    def test_restart_unknown_agent(self, capsys):
        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        repl = REPLSession(ctx)
        repl._cmd_restart("ghost")
        out = capsys.readouterr().out
        assert "not found" in out

    def test_restart_dispatches_to_restart_agent(self):
        """Empty arg with current agent dispatches to _restart_agent."""
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        repl = REPLSession(ctx)
        repl.current = "bot"
        repl._restart_agent = MagicMock()
        repl._cmd_restart("")
        repl._restart_agent.assert_called_once_with("bot")

    def test_restart_explicit_name_dispatches(self):
        """Explicit name dispatches to _restart_agent with that name."""
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400", "coder": "http://coder:8400"})
        repl = REPLSession(ctx)
        repl.current = "bot"
        repl._restart_agent = MagicMock()
        repl._cmd_restart("coder")
        repl._restart_agent.assert_called_once_with("coder")


class TestAddkeyNoInlineKey:
    """Inline key values are no longer accepted by /addkey."""

    def test_addkey_ignores_inline_value(self, capsys):
        """Even if arg contains spaces, only first word is used as service name."""
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        ctx.credential_vault = MagicMock()
        ctx.credential_vault.add_credential = MagicMock()
        repl = REPLSession(ctx)

        # Passing "myservice sk-secret-key" — the old code would use "sk-secret-key"
        # as the key value. New code should only use "myservice" as the service name
        # and prompt for the key.
        with (
            patch("src.cli.repl.click.prompt", return_value="prompted-key"),
            patch("src.cli.repl.click.confirm", return_value=False),
        ):
            repl._cmd_addkey("myservice sk-secret-key")

        # Should have stored the prompted key, not the inline one
        ctx.credential_vault.add_credential.assert_called_once()
        call_args = ctx.credential_vault.add_credential.call_args
        assert call_args[0][1] == "prompted-key"


class TestRestartCompleter:
    """Tab completer completes agent names after /restart."""

    def test_completer_restart_agents(self):
        from unittest.mock import MagicMock

        from src.cli.repl import _REPLCompleter

        session = MagicMock()
        session.ctx.agents = {
            "researcher": "http://localhost:8401",
            "coder": "http://localhost:8402",
        }
        session._commands = {"/restart": None}

        completer = _REPLCompleter(session)

        import readline

        with patch.object(readline, "get_line_buffer", return_value="/restart c"):
            result = completer.complete("c", 0)
            assert result == "coder "


# ── WU3: stop crash when Docker unavailable ──────────────────


class TestStopNoDocker:
    """stop should not crash when Docker is unavailable."""

    def test_stop_no_docker(self, tmp_path):
        # No PID file, no Docker → should exit cleanly
        with (
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.main.cli_config.PROJECT_ROOT", tmp_path),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["stop"])
            assert result.exit_code == 0, result.output
            assert "Could not connect to Docker" in result.output or "No OpenLegion" in result.output


class TestJsonGlobalFlag:
    """--json global flag should be accepted."""

    def test_json_flag_with_status(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({"agents": {"bot1": {"role": "test"}}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.TEAMS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--json", "status"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "agents" in data


# ── WU12: --verbose/--quiet flags ────────────────────────────


class TestVerboseQuietFlags:
    """--verbose and --quiet flags should be accepted."""

    def test_verbose_flag_accepted(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_quiet_flag_accepted(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-q", "--help"])
        assert result.exit_code == 0


# ── WU16: shell completion ───────────────────────────────────


class TestShellCompletion:
    """Shell completion functions return valid results."""

    def test_complete_agent_names(self, tmp_path):
        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(yaml.dump({"agents": {"coder": {}, "researcher": {}}}))
        with (
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"),
        ):
            from src.cli.main import _complete_agent_names
            result = _complete_agent_names(None, None, "c")
            assert "coder" in result
            assert "researcher" not in result


# ── Task 2b: CLI REPL stamps human/cli origin on dispatch ──


class TestREPLOriginStamp:
    """``_send_message_streaming`` must stamp ``X-Origin: kind=human``
    on every CLI-initiated request so the agent / lane sees a typed
    origin instead of a missing or legacy header.
    """

    def test_streaming_dispatch_sends_typed_human_origin_header(self):
        import asyncio
        import threading
        from unittest.mock import AsyncMock, MagicMock

        from src.cli.repl import _send_message_streaming
        from src.host.transport import HttpTransport
        from src.shared.types import MessageOrigin

        captured_headers: list = []

        async def _fake_stream(*_args, **kwargs):
            captured_headers.append(kwargs.get("headers"))
            yield {"type": "done", "response": "ok"}

        transport = MagicMock(spec=HttpTransport)
        transport.stream_request = _fake_stream
        transport.request = AsyncMock(return_value={"response": "ok"})

        # _send_message_streaming uses run_coroutine_threadsafe — needs
        # a real running event loop in a background thread.
        dispatch_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run():
            asyncio.set_event_loop(dispatch_loop)
            dispatch_loop.call_soon(ready.set)
            dispatch_loop.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        ready.wait()

        try:
            with patch.dict(os.environ, {"USER": "alice"}):
                _send_message_streaming(
                    transport, "researcher", "do thing",
                    dispatch_loop=dispatch_loop,
                )
        finally:
            dispatch_loop.call_soon_threadsafe(dispatch_loop.stop)
            t.join(timeout=2)
            dispatch_loop.close()

        assert captured_headers, "stream_request was never invoked"
        hdrs = captured_headers[0] or {}
        assert "X-Origin" in hdrs
        parsed = MessageOrigin.from_header_value(hdrs["X-Origin"], trust_kind=True)
        assert parsed is not None
        assert parsed.kind == "human"
        assert parsed.channel == "cli"
        assert parsed.user == "alice"


# === Task 9 — Workplace + pending-action CLI commands ===


class TestWorkplaceCLICommands:
    """``openlegion teams / team / tasks / pending / confirm /
    cancel`` shell out to the local mesh over httpx — patch the client
    to assert the right path and shape are returned to stdout."""

    def _patched_get(self, payload, status=200):
        """Return a context-managed ``patch`` that fakes ``httpx.get``."""
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return patch("httpx.get", return_value=resp)

    def _patched_post(self, payload, status=200):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return patch("httpx.post", return_value=resp)

    def test_teams_command_renders_table(self):
        runner = CliRunner()
        with self._patched_get({"teams": [
            {"name": "research", "status": "active", "total": 3,
             "members": ["alpha", "beta"], "counts": {"working": 2}},
        ]}):
            result = runner.invoke(cli, ["teams"])
        assert result.exit_code == 0
        assert "research" in result.output
        assert "active" in result.output

    def test_teams_command_json_mode(self):
        runner = CliRunner()
        with self._patched_get({"teams": [{"name": "research", "status": "active", "total": 0}]}):
            result = runner.invoke(cli, ["--json", "teams"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["teams"][0]["name"] == "research"

    def test_team_command_shows_one(self):
        runner = CliRunner()
        body = {"teams": [
            {"name": "research", "status": "active", "total": 1,
             "description": "deep dive", "members": ["alpha"],
             "counts": {"pending": 1}, "blockers": []},
        ]}
        with self._patched_get(body):
            result = runner.invoke(cli, ["team", "research"])
        assert result.exit_code == 0
        assert "research" in result.output
        assert "deep dive" in result.output

    def test_team_command_unknown_exits_1(self):
        runner = CliRunner()
        with self._patched_get({"teams": []}):
            result = runner.invoke(cli, ["team", "missing"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_tasks_command_table_render(self):
        runner = CliRunner()
        body = {"enabled": True, "tasks": [
            {"id": "task_abc", "status": "working", "assignee": "alpha",
             "team_id": "research", "title": "dig"},
        ]}
        with self._patched_get(body):
            result = runner.invoke(cli, ["tasks"])
        assert result.exit_code == 0
        assert "task_abc" in result.output
        assert "working" in result.output

    def test_pending_command_table(self):
        runner = CliRunner()
        body = {"pending": [
            {"nonce": "n1", "actor": "operator", "action_kind": "model",
             "target_kind": "agent", "target_id": "alpha"},
        ]}
        with self._patched_get(body):
            result = runner.invoke(cli, ["pending"])
        assert result.exit_code == 0
        assert "n1" in result.output
        assert "model" in result.output

    def test_pending_command_empty(self):
        runner = CliRunner()
        with self._patched_get({"pending": []}):
            result = runner.invoke(cli, ["pending"])
        assert result.exit_code == 0
        assert "No pending" in result.output

    def test_confirm_command_calls_mesh(self):
        runner = CliRunner()
        with self._patched_post({"success": True}) as m:
            result = runner.invoke(cli, ["confirm", "abc-nonce"])
        assert result.exit_code == 0
        assert "Confirmed" in result.output
        assert "abc-nonce" in result.output
        # URL must be the dashboard proxy, not the raw mesh endpoint — the
        # mesh endpoint requires bearer/internal + human-origin headers a
        # bare CLI POST can't provide and 401s/403s (plan §8 #24).
        called_url = m.call_args.args[0]
        assert "/dashboard/api/workplace/pending/abc-nonce/confirm" in called_url

    def test_confirm_command_propagates_error(self):
        runner = CliRunner()
        with self._patched_post({"detail": "expired"}, status=400):
            result = runner.invoke(cli, ["confirm", "abc-nonce"])
        assert result.exit_code == 1

    def test_cancel_command_calls_mesh(self):
        runner = CliRunner()
        with self._patched_post({"ok": True, "nonce": "abc"}) as m:
            result = runner.invoke(cli, ["cancel", "abc"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        called_url = m.call_args.args[0]
        assert "/dashboard/api/workplace/pending/abc/cancel" in called_url

    def test_cancel_command_includes_csrf_header(self):
        """The CLI commands must inject X-Requested-With so the dashboard
        CSRF gate doesn't reject the request."""
        runner = CliRunner()
        with self._patched_post({"ok": True, "nonce": "abc"}) as m:
            runner.invoke(cli, ["cancel", "abc"])
        kwargs = m.call_args.kwargs
        headers = kwargs.get("headers") or {}
        assert headers.get("X-Requested-With") == "XMLHttpRequest"
        # And X-Origin so the human-confirmation gate accepts it
        assert "kind=human" in headers.get("X-Origin", "")
        assert "channel=cli" in headers.get("X-Origin", "")


class TestRestartHealthReregistration:
    """Archive deregisters an agent from the health monitor; a manual
    ``/restart`` must re-register it (same guard as the dashboard restart
    endpoint) or the revived agent runs permanently unmonitored."""

    def _restart(self, health_monitor, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from src.cli import repl as repl_mod
        from src.cli.repl import REPLSession

        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"})
        ctx.runtime = MagicMock()
        ctx.runtime.start_agent.return_value = "http://bot:8400"
        ctx.runtime.wait_for_agent = AsyncMock(return_value=True)
        ctx.router = MagicMock()
        ctx.transport = MagicMock()
        ctx.health_monitor = health_monitor
        monkeypatch.setattr(
            repl_mod, "_load_config",
            lambda: {"agents": {"bot": {}}, "llm": {"default_model": "m"}},
        )
        repl = REPLSession(ctx)
        repl._restart_agent("bot")
        return ctx

    def test_restart_reregisters_deregistered_agent(self, monkeypatch):
        from unittest.mock import MagicMock

        health = MagicMock()
        health.agents = {}  # archived → deregistered
        self._restart(health, monkeypatch)
        health.register.assert_called_once_with("bot")

    def test_restart_leaves_registered_agent_alone(self, monkeypatch):
        from unittest.mock import MagicMock

        health = MagicMock()
        health.agents = {"bot": object()}  # still monitored
        self._restart(health, monkeypatch)
        health.register.assert_not_called()

    def test_restart_tolerates_missing_health_monitor(self, monkeypatch):
        # ctx.health_monitor is None in embedded/test setups — no crash.
        self._restart(None, monkeypatch)
