"""Boot must not resurrect archived agents.

Regression guard (plan §8 #24 prereq i): ``RuntimeContext._start_agents``
iterated ``agents_cfg`` with NO status filter — an archived agent (torn
down intentionally by ``_archive_agent_core``: container stopped, health
deregistered) got its container restarted and health re-registered on
every mesh boot, fighting the archive. Mirrors the archive-endpoint
regression guard in ``test_archive_health_dereg.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from src.cli.runtime import RuntimeContext
from src.host.transport import HttpTransport


class _StubRuntimeBackend:
    """Cheap stand-in for DockerBackend/SandboxBackend — tracks which
    agent ids actually got a ``start_agent`` call."""

    def __init__(self):
        self.extra_env: dict[str, str] = {}
        self.started: list[str] = []

    def start_agent(self, *, agent_id, role, tools_dir, model, thinking, env_overrides):
        self.started.append(agent_id)
        return f"http://agent-{agent_id}"


def _build_ctx(fake_cfg: dict) -> RuntimeContext:
    ctx = RuntimeContext.__new__(RuntimeContext)
    ctx.cfg = fake_cfg
    ctx.cost_tracker = MagicMock()
    ctx.runtime = _StubRuntimeBackend()
    ctx.router = MagicMock()
    ctx.transport = HttpTransport()
    ctx.health_monitor = MagicMock()
    ctx.permissions = MagicMock()
    ctx.credential_vault = MagicMock()
    ctx.connector_store = MagicMock()
    return ctx


def test_start_agents_skips_archived(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_config
    import src.cli.runtime as runtime_mod

    fake_cfg = {
        "agents": {
            "worker-active": {"role": "worker", "status": "active"},
            "worker-archived": {"role": "worker", "status": "archived"},
        },
        "llm": {},
        "mesh": {"port": 8400},
        "network": {},
    }
    monkeypatch.setattr(runtime_mod, "_load_config", lambda: fake_cfg)
    monkeypatch.setattr(cli_config, "_ensure_operator_agent", lambda **k: None)

    ctx = _build_ctx(fake_cfg)

    with caplog.at_level(logging.INFO):
        RuntimeContext._start_agents(ctx)

    # No container start for the archived agent — only the active one.
    assert ctx.runtime.started == ["worker-active"]
    # No router registration for the archived agent.
    registered_ids = [c.args[0] for c in ctx.router.register_agent.call_args_list]
    assert registered_ids == ["worker-active"]
    # No transport registration (HttpTransport.register) either.
    assert "worker-active" in ctx.transport._urls
    assert "worker-archived" not in ctx.transport._urls
    # No health monitor registration.
    health_registered = [c.args[0] for c in ctx.health_monitor.register.call_args_list]
    assert health_registered == ["worker-active"]
    # One info line logged for the skipped agent.
    assert any("worker-archived" in r.message for r in caplog.records)


def test_start_agents_missing_status_defaults_active(tmp_path, monkeypatch):
    """Legacy rows with no ``status`` field must still boot (default active)."""
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_config
    import src.cli.runtime as runtime_mod

    fake_cfg = {
        "agents": {"worker-legacy": {"role": "worker"}},
        "llm": {}, "mesh": {"port": 8400}, "network": {},
    }
    monkeypatch.setattr(runtime_mod, "_load_config", lambda: fake_cfg)
    monkeypatch.setattr(cli_config, "_ensure_operator_agent", lambda **k: None)

    ctx = _build_ctx(fake_cfg)
    RuntimeContext._start_agents(ctx)

    assert ctx.runtime.started == ["worker-legacy"]
