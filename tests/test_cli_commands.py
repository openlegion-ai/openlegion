"""Tests for CLI commands: agent add/list/edit/remove, setup helpers."""

import json
import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.cli import cli


class TestAgentAdd:
    def test_add_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot"],
                input="Code review specialist\n\n\n",  # description + model + browser
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert "mybot" in agents_cfg["agents"]
            assert agents_cfg["agents"]["mybot"]["role"] == "Code review specialist"
            assert "openai/" in agents_cfg["agents"]["mybot"]["model"]
            assert agents_cfg["agents"]["mybot"].get("browser_backend", "basic") in ("basic", "")

            perms = json.loads(perms_file.read_text())
            assert "mybot" in perms["permissions"]
            assert "llm" in perms["permissions"]["mybot"]["allowed_apis"]

    def test_add_agent_with_model_flag(self, tmp_path):
        """--model flag skips interactive model selection."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot", "--model", "anthropic/claude-haiku-4-5-20251001"],
                input="Code review specialist\n\n",  # description + browser
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["model"] == "anthropic/claude-haiku-4-5-20251001"

    def test_add_agent_with_browser_flag(self, tmp_path):
        """--browser flag skips interactive browser selection."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot", "--model", "openai/gpt-4.1", "--browser", "stealth"],
                input="Web scraper agent\n",  # only description needed
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output
            assert "stealth" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["browser_backend"] == "stealth"

    def test_add_duplicate_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"existing": {"role": "test"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "add", "existing"])
            assert "already exists" in result.output


class TestAgentEdit:
    def test_edit_model_flag(self, tmp_path):
        """--model flag updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot", "--model", "anthropic/claude-sonnet-4-6"],
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["model"] == "anthropic/claude-sonnet-4-6"

    def test_edit_browser_flag(self, tmp_path):
        """--browser flag updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "openai/gpt-4.1",
                "browser_backend": "basic",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot", "--browser", "stealth"],
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["browser_backend"] == "stealth"

    def test_edit_description_flag(self, tmp_path):
        """--description flag updates agents.yaml role field."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "openai/gpt-4.1",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot", "--description", "Web research specialist"],
            )
            assert result.exit_code == 0, result.output
            assert "description updated" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Web research specialist"

    def test_edit_interactive(self, tmp_path):
        """Interactive menu flow picks model via property selector."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot"],
                input="1\n2\n",  # select model property, then pick second model
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

    def test_edit_nonexistent_agent(self, tmp_path):
        """Nonexistent agent name shows error."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))
        agents_file.write_text(yaml.dump({
            "agents": {"other": {"role": "test", "model": "openai/gpt-4.1"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "edit", "ghost", "--model", "x"])
            assert "not found" in result.output

    def test_edit_no_change(self, tmp_path):
        """Same model value should report no change."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot", "--model", "anthropic/claude-haiku-4-5-20251001"],
            )
            assert result.exit_code == 0
            assert "already uses" in result.output


class TestAgentList:
    def test_list_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {
                "bot1": {"role": "assistant", "model": "openai/gpt-4o-mini"},
                "bot2": {"role": "coder", "model": "anthropic/claude-sonnet-4-5-20250929"},
            },
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert result.exit_code == 0
            assert "bot1" in result.output
            assert "bot2" in result.output
            assert "openai/gpt-4o-mini" in result.output
            assert "anthropic/claude-sonnet-4-5-20250929" in result.output
            assert "Browser" in result.output
            assert "basic" in result.output

    def test_list_no_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert "No agents" in result.output


class TestAgentRemove:
    def test_remove_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {"role": "test"}},
        }))
        perms_file.write_text(json.dumps({
            "permissions": {"mybot": {"allowed_apis": ["llm"]}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "remove", "mybot", "--yes"])
            assert result.exit_code == 0
            assert "Removed" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert "mybot" not in agents_cfg.get("agents", {})

            perms = json.loads(perms_file.read_text())
            assert "mybot" not in perms["permissions"]

    def test_remove_nonexistent_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))
        agents_file.write_text(yaml.dump({
            "agents": {"other": {"role": "test"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "remove", "ghost"])
            assert "not found" in result.output


class _MockCtx:
    """Lightweight mock of RuntimeContext for REPL tests."""

    def __init__(self, agent_urls=None, *, blackboard=None, lane_manager=None,
                 cron_scheduler=None, orchestrator=None):
        self.agent_urls = agent_urls or {}
        self.cfg = {"mesh": {"port": 8420}, "agents": {}, "llm": {"default_model": "openai/gpt-4o-mini"}}
        self.blackboard = blackboard
        self.lane_manager = lane_manager
        self.cron_scheduler = cron_scheduler
        self.orchestrator = orchestrator
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
        self._dispatch_loop = None

    @property
    def agents(self):
        return self.agent_urls

    @property
    def dispatch_loop(self):
        return self._dispatch_loop


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

        repl._cmd_blackboard("del test/del")
        out = capsys.readouterr().out
        assert "Deleted: test/del" in out
        assert bb.read("test/del") is None
        bb.close()

    def test_delete_history_blocked(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        bb.write("history/x", {"x": 1}, written_by="cli")
        repl = self._make_repl(bb)

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


class TestREPLWorkflow:
    def test_list_no_workflows(self, capsys):
        from unittest.mock import MagicMock
        from src.cli.repl import REPLSession
        orch = MagicMock()
        orch.workflows = {}
        orch.active_executions = {}
        ctx = _MockCtx(orchestrator=orch)
        repl = REPLSession(ctx)

        repl._cmd_workflow("")
        out = capsys.readouterr().out
        assert "No workflows" in out

    def test_list_with_workflows(self, capsys):
        from unittest.mock import MagicMock
        from src.cli.repl import REPLSession
        wf = MagicMock()
        wf.steps = [1, 2, 3]
        orch = MagicMock()
        orch.workflows = {"deploy": wf}
        orch.active_executions = {}
        ctx = _MockCtx(orchestrator=orch)
        repl = REPLSession(ctx)

        repl._cmd_workflow("")
        out = capsys.readouterr().out
        assert "deploy" in out
        assert "3 steps" in out

    def test_run_missing_name(self, capsys):
        from unittest.mock import MagicMock
        from src.cli.repl import REPLSession
        orch = MagicMock()
        ctx = _MockCtx(orchestrator=orch)
        repl = REPLSession(ctx)

        repl._cmd_workflow("run")
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_not_available(self, capsys):
        from src.cli.repl import REPLSession
        ctx = _MockCtx()
        repl = REPLSession(ctx)
        repl._cmd_workflow("")
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
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 5m", message="check")
        sched.pause_job(job.id)
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


class TestREPLWorkflowActive:
    """Workflow list with active executions."""

    def test_list_with_active_executions(self, capsys):
        from unittest.mock import MagicMock
        from src.cli.repl import REPLSession
        wf = MagicMock()
        wf.steps = [1, 2]
        ex = MagicMock()
        ex.status = "running"
        ex.workflow.name = "deploy"
        orch = MagicMock()
        orch.workflows = {"deploy": wf}
        orch.active_executions = {"exec-001": ex}
        ctx = _MockCtx(orchestrator=orch)
        repl = REPLSession(ctx)

        repl._cmd_workflow("")
        out = capsys.readouterr().out
        assert "deploy" in out
        assert "2 steps" in out
        assert "Active executions" in out
        assert "exec-001" in out
        assert "running" in out


class TestREPLBlackboardSetMissingArgs:
    """Blackboard set with insufficient arguments."""

    def test_set_missing_value(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        from src.cli.repl import REPLSession
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        ctx = _MockCtx(agent_urls={"bot": "http://bot:8400"}, blackboard=bb)
        repl = REPLSession(ctx)

        repl._cmd_blackboard("set mykey")
        out = capsys.readouterr().out
        assert "Usage:" in out
        bb.close()

    def test_set_missing_key_and_value(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        from src.cli.repl import REPLSession
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
        from src.cli.repl import REPLSession
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = sched.add_job(agent="bot", schedule="every 10m", message="ping")
        sched.pause_job(job.id)
        ctx = _MockCtx(cron_scheduler=sched)
        repl = REPLSession(ctx)

        repl._cmd_cron("")
        out = capsys.readouterr().out
        assert "paused" in out


class TestREPLBlackboardGetMultiline:
    """Blackboard get with multiline values are properly indented."""

    def test_multiline_value_indented(self, tmp_path, capsys):
        from src.host.mesh import Blackboard
        from src.cli.repl import REPLSession
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
        assert result.exit_code == 0
        assert "not running" in result.output


class TestPermissionsDefault:
    def test_new_agent_gets_default_permissions(self):
        """Verify the permissions module falls back to 'default' template."""
        from src.host.permissions import PermissionMatrix

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "permissions": {
                    "default": {
                        "can_message": ["orchestrator"],
                        "allowed_apis": ["llm"],
                        "blackboard_read": ["context/*"],
                    }
                }
            }, f)
            f.flush()
            pm = PermissionMatrix(config_path=f.name)

        perms = pm.get_permissions("some_unknown_agent")
        assert perms.agent_id == "some_unknown_agent"
        assert "orchestrator" in perms.can_message
        assert "llm" in perms.allowed_apis
        assert pm.can_use_api("some_unknown_agent", "llm")
        os.unlink(f.name)
