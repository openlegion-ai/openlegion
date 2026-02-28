"""Tests for CLI commands: agent add/list/edit/remove, project CRUD, setup helpers."""

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
                input="Code review specialist\n\n",  # description + model
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert "mybot" in agents_cfg["agents"]
            assert agents_cfg["agents"]["mybot"]["role"] == "Code review specialist"
            assert "openai/" in agents_cfg["agents"]["mybot"]["model"]

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
                input="Code review specialist\n",  # description
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["model"] == "anthropic/claude-haiku-4-5-20251001"

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
            assert result.exit_code == 1
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

    def test_edit_description_flag(self, tmp_path):
        """--role flag updates agents.yaml role field."""
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
                ["agent", "edit", "mybot", "--role", "Web research specialist"],
            )
            assert result.exit_code == 0, result.output
            assert "description updated" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Web research specialist"

    def test_edit_description_flag_compat(self, tmp_path):
        """--description (hidden compat) still updates agents.yaml role field."""
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
                ["agent", "edit", "mybot", "--description", "Legacy flag test"],
            )
            assert result.exit_code == 0, result.output
            assert "description updated" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Legacy flag test"

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
            assert result.exit_code == 1
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
            assert result.exit_code == 1
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


class TestConfigExportImport:
    def test_config_export_creates_file(self, tmp_path):
        """Export creates a tarball containing config files."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        projects_dir = tmp_path / "projects"

        config_file.write_text(yaml.dump({"mesh": {"host": "0.0.0.0", "port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {"bot": {"role": "test"}}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        archive = tmp_path / "backup.tar.gz"

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "export", str(archive)])
            assert result.exit_code == 0, result.output
            assert archive.exists()

    def test_config_import_restores_files(self, tmp_path):
        """Import restores config from tarball."""
        import tarfile

        # Create a source tarball with config content
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "mesh.yaml").write_text(yaml.dump({"mesh": {"port": 9999}}))
        (src_dir / "agents.yaml").write_text(yaml.dump({"agents": {"bot": {"role": "test"}}}))

        archive = tmp_path / "backup.tar.gz"
        with tarfile.open(str(archive), "w:gz") as tar:
            tar.add(str(src_dir / "mesh.yaml"), arcname="mesh.yaml")
            tar.add(str(src_dir / "agents.yaml"), arcname="agents.yaml")

        # Set up destination config dir
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        config_file = dest_dir / "mesh.yaml"
        agents_file = dest_dir / "agents.yaml"
        perms_file = dest_dir / "permissions.json"
        projects_dir = dest_dir / "projects"

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "import", str(archive), "--yes"])
            assert result.exit_code == 0, result.output
            assert config_file.exists()
            restored = yaml.safe_load(config_file.read_text())
            assert restored["mesh"]["port"] == 9999

    def test_config_import_rejects_unsafe_path(self, tmp_path):
        """Import rejects archives with path traversal."""
        import tarfile

        archive = tmp_path / "evil.tar.gz"
        with tarfile.open(str(archive), "w:gz") as tar:
            import io
            data = b"evil"
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        config_file = tmp_path / "mesh.yaml"
        projects_dir = tmp_path / "projects"

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "import", str(archive), "--yes"])
            assert result.exit_code != 0

    def test_config_import_rejects_symlinks(self, tmp_path):
        """config import rejects archives containing symlinks."""
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="evil_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        buf.seek(0)

        archive = tmp_path / "evil.tar.gz"
        archive.write_bytes(buf.read())

        config_file = tmp_path / "mesh.yaml"
        projects_dir = tmp_path / "projects"

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "import", str(archive), "--yes"])
            assert result.exit_code == 1
            assert "Unsafe link" in result.output


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


class TestEnsureAllAgentPermissions:
    def test_backfills_missing_agents(self, tmp_path):
        """_ensure_all_agent_permissions adds permissions for agents not yet in permissions.json."""
        from unittest.mock import patch

        from src.cli.config import _ensure_all_agent_permissions

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("agents:\n  alice:\n    role: helper\n  bob:\n    role: coder\n")
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {"alice": {"can_message": ["orchestrator"]}}}))

        with patch("src.cli.config.AGENTS_FILE", agents_yaml), \
             patch("src.cli.config.CONFIG_FILE", tmp_path / "mesh.yaml"), \
             patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())
        assert "alice" in perms["permissions"]
        assert "bob" in perms["permissions"]
        assert perms["permissions"]["bob"]["allowed_credentials"] == ["*"]


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


# ── Project CLI commands ─────────────────────────────────────


class TestProjectCreate:
    def test_create_project_with_flags(self, tmp_path):
        projects_dir = tmp_path / "projects"
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({
            "agents": {"bot1": {"role": "test"}, "bot2": {"role": "test2"}},
        }))
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot1": {"blackboard_read": [], "blackboard_write": []},
                "bot2": {"blackboard_read": [], "blackboard_write": []},
            }
        }))

        with (
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["project", "create", "test-proj", "-D", "Test project", "-a", "bot1,bot2"],
            )
            assert result.exit_code == 0
            assert "test-proj" in result.output
            assert "bot1" in result.output

            meta_file = projects_dir / "test-proj" / "metadata.yaml"
            assert meta_file.exists()
            data = yaml.safe_load(meta_file.read_text())
            assert data["name"] == "test-proj"
            assert "bot1" in data["members"]
            assert "bot2" in data["members"]

    def test_create_project_rejects_invalid_name(self, tmp_path):
        projects_dir = tmp_path / "projects"
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {}}))

        with (
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "create", "../escape", "-D", "bad"])
            assert result.exit_code == 1
            assert "Invalid project name" in result.output

    def test_create_project_unknown_agents(self, tmp_path):
        projects_dir = tmp_path / "projects"
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {"bot1": {"role": "test"}}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["project", "create", "proj", "-D", "x", "-a", "nonexistent"],
            )
            assert result.exit_code == 1
            assert "Unknown agent" in result.output


class TestProjectList:
    def test_list_with_projects(self, tmp_path):
        projects_dir = tmp_path / "projects"
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({
            "agents": {"bot1": {"role": "test"}, "bot2": {"role": "test2"}},
        }))

        d = projects_dir / "my-proj"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(yaml.dump({
            "name": "my-proj", "description": "Desc", "members": ["bot1"],
        }))

        with (
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "list"])
            assert result.exit_code == 0
            assert "my-proj" in result.output
            assert "bot1" in result.output
            # bot2 is standalone
            assert "bot2" in result.output

    def test_list_empty(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {}}))

        with (
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "nonexistent"),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "list"])
            assert "No projects" in result.output


class TestProjectDelete:
    def test_delete_project(self, tmp_path):
        projects_dir = tmp_path / "projects"
        perms_file = tmp_path / "permissions.json"
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        d = projects_dir / "doomed"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(yaml.dump({
            "name": "doomed", "members": ["bot1"],
        }))
        (d / "project.md").write_text("# doomed")

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {"bot1": {"role": "test"}}}))
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot1": {
                    "blackboard_read": ["projects/doomed/*"],
                    "blackboard_write": ["projects/doomed/*"],
                },
            }
        }))

        with (
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "delete", "doomed", "--yes"])
            assert result.exit_code == 0
            assert "Deleted" in result.output
            assert not d.exists()


# ── REPL project commands ────────────────────────────────────


class TestREPLProjectCommand:
    def _make_repl(self, tmp_path):
        from src.cli.repl import REPLSession

        projects_dir = tmp_path / "projects"
        d = projects_dir / "alpha"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(yaml.dump({
            "name": "alpha", "description": "Alpha project",
            "members": ["bot1"], "settings": {},
        }))

        ctx = _MockCtx(agent_urls={"bot1": "http://bot1:8400", "bot2": "http://bot2:8400"})
        ctx.cfg["agents"] = {"bot1": {"role": "a"}, "bot2": {"role": "b"}}
        ctx.cfg["_agent_projects"] = {"bot1": "alpha"}

        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            repl = REPLSession(ctx)
        return repl, projects_dir

    def test_project_list(self, tmp_path, capsys):
        repl, projects_dir = self._make_repl(tmp_path)
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            repl._cmd_project("list")
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "bot1" in out
        assert "bot2" in out  # standalone

    def test_project_use_and_clear(self, tmp_path, capsys):
        repl, projects_dir = self._make_repl(tmp_path)
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            repl._cmd_project("use alpha")
        assert repl._active_project == "alpha"

        repl._cmd_project("use none")
        assert repl._active_project is None

    def test_project_use_unknown(self, tmp_path, capsys):
        repl, projects_dir = self._make_repl(tmp_path)
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            repl._cmd_project("use nonexistent")
        out = capsys.readouterr().out
        assert "Unknown project" in out
        assert repl._active_project is None

    def test_project_info(self, tmp_path, capsys):
        repl, projects_dir = self._make_repl(tmp_path)
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            repl._cmd_project("info alpha")
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
        ctx.cfg["_agent_projects"] = {"bot1": "alpha"}
        repl = REPLSession(ctx)
        return repl, bb

    def test_set_and_get_with_project(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        repl._active_project = "alpha"

        repl._cmd_blackboard('set mykey {"val": 1}')
        out = capsys.readouterr().out
        assert "projects/alpha/mykey" in out

        repl._cmd_blackboard("get mykey")
        out = capsys.readouterr().out
        assert "projects/alpha/mykey" in out
        bb.close()

    def test_list_with_project_scoping(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        bb.write("projects/alpha/ctx", {"v": 1}, written_by="test")
        bb.write("global/other", {"v": 2}, written_by="test")

        repl._active_project = "alpha"
        repl._cmd_blackboard("list")
        out = capsys.readouterr().out
        assert "projects/alpha/ctx" in out
        assert "global/other" not in out
        bb.close()

    def test_list_bypass_with_all_flag(self, tmp_path, capsys):
        repl, bb = self._make_repl_with_bb(tmp_path)
        bb.write("projects/alpha/ctx", {"v": 1}, written_by="test")
        bb.write("global/other", {"v": 2}, written_by="test")

        repl._active_project = "alpha"
        repl._cmd_blackboard("list --all")
        out = capsys.readouterr().out
        assert "projects/alpha/ctx" in out
        assert "global/other" in out
        bb.close()


class TestExitCodes:
    def test_agent_add_duplicate_exits_nonzero(self, tmp_path):
        """Adding an existing agent exits with code 1."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({"agents": {"mybot": {"role": "test", "model": "openai/gpt-4.1"}}}))
        perms_file.write_text(json.dumps({"permissions": {"mybot": {}}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "add", "mybot"], input="test\n1\n")
            assert result.exit_code != 0

    def test_channels_add_unknown_exits_nonzero(self):
        """Adding unknown channel type exits with code 1."""
        runner = CliRunner()
        result = runner.invoke(cli, ["channels", "add", "fakechannel"])
        assert result.exit_code != 0

    def test_channels_remove_unknown_exits_nonzero(self):
        """Removing unknown channel type exits with code 1."""
        runner = CliRunner()
        result = runner.invoke(cli, ["channels", "remove", "fakechannel"])
        assert result.exit_code != 0


class TestBudgetValidation:
    def test_agent_edit_negative_budget_rejected(self, tmp_path):
        """Negative budget via --budget flag is rejected by Click."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({"agents": {"mybot": {"role": "test", "model": "openai/gpt-4.1"}}}))
        perms_file.write_text(json.dumps({"permissions": {"mybot": {}}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "edit", "mybot", "--budget", "-5"])
            assert result.exit_code != 0


class TestFlagConflicts:
    def test_project_create_uppercase_d_for_description(self, tmp_path):
        """project create -D sets description (not detach conflict)."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        projects_dir = tmp_path / "projects"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", projects_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "create", "testproj", "-D", "My description"])
            assert result.exit_code == 0, result.output
            assert "testproj" in result.output


class TestREPLWorkflowProjectScoping:
    """Workflow commands filter by active project."""

    def _make_repl_with_orch(self):
        from unittest.mock import MagicMock

        from src.cli.repl import REPLSession
        from src.shared.types import WorkflowDefinition

        orch = MagicMock()
        orch.workflows = {
            "alpha/build": WorkflowDefinition(name="alpha/build", trigger="webhook", steps=[]),
            "alpha/deploy": WorkflowDefinition(name="alpha/deploy", trigger="webhook", steps=[]),
            "global-wf": WorkflowDefinition(name="global-wf", trigger="webhook", steps=[]),
        }
        orch.active_executions = {}

        ctx = _MockCtx(agent_urls={"bot1": "http://bot1:8400"}, orchestrator=orch)
        ctx.cfg["_agent_projects"] = {"bot1": "alpha"}
        repl = REPLSession(ctx)
        return repl

    def test_workflow_list_filtered(self, capsys):
        repl = self._make_repl_with_orch()
        repl._active_project = "alpha"
        repl._cmd_workflow("list")
        out = capsys.readouterr().out
        assert "alpha/build" in out
        assert "alpha/deploy" in out
        assert "global-wf" not in out

    def test_workflow_list_unfiltered(self, capsys):
        repl = self._make_repl_with_orch()
        repl._cmd_workflow("list")
        out = capsys.readouterr().out
        assert "alpha/build" in out
        assert "global-wf" in out


class TestJsonOutput:
    def test_agent_list_json(self, tmp_path):
        """agent list --json outputs valid JSON with agents key."""
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
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "agents" in data
            assert len(data["agents"]) == 1
            assert data["agents"][0]["name"] == "bot1"

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
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["status", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "mesh_online" in data
            assert "agents" in data

    def test_skill_list_json(self, tmp_path):
        """skill list --json outputs valid JSON."""
        with patch("src.cli.config.MARKETPLACE_DIR", tmp_path / "marketplace"):
            runner = CliRunner()
            result = runner.invoke(cli, ["skill", "list", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "skills" in data

    def test_project_list_json(self, tmp_path):
        """project list --json outputs valid JSON."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["project", "list", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "projects" in data
            assert "standalone" in data


    def test_channels_list_json(self, tmp_path):
        """channels list --json returns valid JSON with channels key."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({"agents": {}}))
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["channels", "list", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert "channels" in data


class TestCompletion:
    def test_completion_bash(self):
        """completion bash outputs eval string."""
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])
        assert result.exit_code == 0
        assert "OPENLEGION_COMPLETE" in result.output

    def test_completion_unknown_shell(self):
        """completion with unknown shell exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "powershell"])
        assert result.exit_code != 0


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


# ── WP5: logs, config, health, version commands ─────────────


class TestLogsCommand:
    def test_logs_reads_logfile(self, tmp_path):
        """logs command reads from .openlegion.log."""
        log_file = tmp_path / ".openlegion.log"
        log_file.write_text("line1\nline2\nline3\n")

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "-n", "2"])
            assert result.exit_code == 0
            assert "line2" in result.output
            assert "line3" in result.output

    def test_logs_level_filter(self, tmp_path):
        """logs --level filters by level."""
        log_file = tmp_path / ".openlegion.log"
        log_file.write_text(
            '{"level":"INFO","msg":"ok"}\n{"level":"ERROR","msg":"bad"}\n'
        )

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "--level", "error"])
            assert result.exit_code == 0
            assert "ERROR" in result.output
            assert "INFO" not in result.output

    def test_logs_no_file(self, tmp_path):
        """logs exits non-zero when no log file exists."""
        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs"])
            assert result.exit_code != 0


class TestConfigCommand:
    def test_config_show(self, tmp_path):
        """config show outputs YAML."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(
            yaml.dump({"mesh": {"host": "0.0.0.0", "port": 8420}})
        )
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code == 0
            assert "mesh" in result.output

    def test_config_show_json(self, tmp_path):
        """config show --json outputs valid JSON."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(
            yaml.dump({"mesh": {"host": "0.0.0.0", "port": 8420}})
        )
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "show", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "mesh" in data

    def test_config_validate_ok(self, tmp_path):
        """config validate exits 0 with valid config."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        env_file = tmp_path / ".env"

        config_file.write_text(
            yaml.dump({"mesh": {"host": "0.0.0.0", "port": 8420}})
        )
        agents_file.write_text(yaml.dump({"agents": {}}))
        perms_file.write_text(json.dumps({"permissions": {}}))
        env_file.write_text("")

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.ENV_FILE", env_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["config", "validate"])
            assert result.exit_code == 0

    def test_config_path(self):
        """config path shows file locations."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "path"])
        assert result.exit_code == 0
        assert "Project root" in result.output


class TestHealthCommand:
    def test_health_no_docker(self):
        """health reports Docker status."""
        with patch("src.cli.config._check_docker_running", return_value=False):
            runner = CliRunner()
            result = runner.invoke(cli, ["health"])
            # Should exit non-zero if docker is down
            assert result.exit_code != 0


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


# ── WP7: fleet operations ───────────────────────────────────


class TestAgentRestart:
    def test_agent_restart_all_no_mesh(self):
        """agent restart --all exits non-zero when mesh is down."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "restart", "--all"])
            assert result.exit_code != 0
            assert "not running" in result.output.lower() or "mesh" in result.output.lower()

    def test_agent_restart_no_names_no_flag(self):
        """agent restart with no names or --all exits non-zero."""
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"bot1": "http://bot1:8400"}
        with patch("httpx.get", return_value=mock_resp):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "restart"])
            assert result.exit_code != 0
            assert "no agents specified" in result.output.lower()


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
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
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
            patch("src.cli.config.PROJECTS_DIR", tmp_path / "projects"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["status", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "agents" in data
            assert "mesh_online" in data


class TestWebhookCommand:
    def test_webhook_list_no_mesh(self):
        """webhook list exits non-zero when mesh is down."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            runner = CliRunner()
            result = runner.invoke(cli, ["webhook", "list"])
            assert result.exit_code != 0

    def test_webhook_test_no_mesh(self):
        """webhook test exits non-zero when mesh is down."""
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
            runner = CliRunner()
            result = runner.invoke(cli, ["webhook", "test", "my-hook"])
            assert result.exit_code != 0


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


class TestAgentAddWithRole:
    """agent add --role skips the interactive description prompt."""

    def test_add_agent_with_role_flag(self, tmp_path):
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
                ["agent", "add", "mybot", "--role", "Code reviewer"],
                input="\n",  # model selection only
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output
            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Code reviewer"


class TestChannelsRemoveConfirmation:
    """channels remove prompts for confirmation and supports interactive picker."""

    def test_channels_remove_with_confirmation(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "channels": {"telegram_bot": {"enabled": True}},
        }))

        with patch("src.cli.config.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["channels", "remove", "telegram"],
                input="y\n",
            )
            assert result.exit_code == 0, result.output
            assert "Removed" in result.output

    def test_channels_remove_abort(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "channels": {"telegram_bot": {"enabled": True}},
        }))

        with patch("src.cli.config.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["channels", "remove", "telegram"],
                input="n\n",
            )
            assert result.exit_code != 0  # click.confirm abort

    def test_channels_remove_yes_flag(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "channels": {"telegram_bot": {"enabled": True}},
        }))

        with patch("src.cli.config.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["channels", "remove", "telegram", "-y"],
            )
            assert result.exit_code == 0, result.output
            assert "Removed" in result.output


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


class TestAgentAddFullyNonInteractive:
    """agent add with --role and --model requires no stdin."""

    def test_add_no_prompts(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", tmp_path),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot", "--role", "Coder", "--model", "openai/gpt-4.1"],
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output
            assert "openai/gpt-4.1" in result.output
            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Coder"
            assert agents_cfg["agents"]["mybot"]["model"] == "openai/gpt-4.1"


class TestAgentEditRolePrecedence:
    """When both --role and --description are passed, --role wins."""

    def test_role_takes_precedence(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {"role": "test", "model": "openai/gpt-4.1"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "edit", "mybot", "--role", "Winner", "--description", "Loser"],
            )
            assert result.exit_code == 0, result.output
            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["role"] == "Winner"


class TestChannelsRemoveInteractive:
    """channels remove with no argument triggers interactive picker."""

    def test_interactive_picker(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "channels": {"telegram_bot": {"enabled": True}},
        }))

        with patch("src.cli.config.CONFIG_FILE", config_file):
            runner = CliRunner()
            # Select first channel (telegram), then confirm
            result = runner.invoke(
                cli,
                ["channels", "remove"],
                input="1\ny\n",
            )
            assert result.exit_code == 0, result.output
            assert "Removed" in result.output
            assert "Telegram" in result.output


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
