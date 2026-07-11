"""Configuration loading, Docker helpers, and agent management."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import click
import yaml

from src.shared.limits import THINKING_LEVELS
from src.shared.operator_playbooks import _OPERATOR_CORE
from src.shared.types import RESERVED_AGENT_IDS
from src.shared.utils import atomic_write_text, truncate

logger = logging.getLogger("cli")

# L4: irreversible-grant ceiling for fleet templates. Booleans here can never
# be set true from a template's ``permissions`` dict — the wallet is an
# operator/user-only grant (spending money) that must not be mintable by
# template application. ``can_spawn`` is intentionally NOT here: ephemeral
# fleet-spawn is now a default-on capability for new agents (see
# ``_add_agent_permissions`` base), so templates may set it freely. Mirrors
# ``_OPERATOR_PERMISSION_CEILING`` in ``src/shared/operator_ceiling.py``.
_TEMPLATE_PERMISSION_CEILING = frozenset({"can_use_wallet"})

# ── Path constants ──────────────────────────────────────────


def _find_project_root() -> Path:
    """Find the project root directory.

    When running from source, __file__ is in src/cli/config.py so
    parent.parent.parent gives the repo root. When pip-installed into
    site-packages, that path is wrong — fall back to CWD and walk up
    looking for pyproject.toml.
    """
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    # Installed as package — walk up from CWD
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return cwd


PROJECT_ROOT = _find_project_root()
# Canonical on-box data directory holding the SQLite stores (tasks.db,
# traces.db, costs.db, intent.db, …). Historically DB paths were bare
# strings (``data/<name>.db``) relative to CWD in runtime.py/server.py;
# this constant makes the location resolvable for read-only tooling like
# the session reader (``openlegion session``).
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh.yaml"
AGENTS_FILE = PROJECT_ROOT / "config" / "agents.yaml"
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"
TEAMS_DIR = PROJECT_ROOT / "config" / "teams"
NETWORK_FILE = PROJECT_ROOT / "config" / "network.yaml"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DOCKER_IMAGE = "openlegion-agent:latest"
BROWSER_IMAGE = "openlegion-browser:latest"

# ── Provider data ───────────────────────────────────────────


def _get_providers() -> list[dict[str, str]]:
    """Load provider list from the model registry (single source of truth)."""
    from src.shared.models import get_all_providers

    return get_all_providers()


# Lazy-evaluated: built on first access so litellm discovery works.
class _LazyProviders:
    """List-like wrapper that loads providers on first access."""

    def __init__(self) -> None:
        self._data: list[dict[str, str]] | None = None

    def _ensure(self) -> None:
        if self._data is None:
            self._data = _get_providers()

    def __getitem__(self, key):
        self._ensure()
        return self._data[key]

    def __iter__(self):
        self._ensure()
        return iter(self._data)

    def __len__(self):
        self._ensure()
        return len(self._data)


_PROVIDERS = _LazyProviders()


class _LazyProviderModels:
    """Dict-like wrapper that builds from litellm on first access."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] | None = None

    def _ensure(self) -> None:
        if self._data is None:
            from src.shared.models import get_provider_models

            self._data = {p["name"]: get_provider_models(p["name"]) for p in _PROVIDERS}

    def __getitem__(self, key: str) -> list[str]:
        self._ensure()
        return self._data[key]  # type: ignore[index]

    def items(self):
        self._ensure()
        return self._data.items()  # type: ignore[union-attr]

    def values(self):
        self._ensure()
        return self._data.values()  # type: ignore[union-attr]

    def keys(self):
        self._ensure()
        return self._data.keys()  # type: ignore[union-attr]

    def get(self, key: str, default=None):
        self._ensure()
        return self._data.get(key, default)  # type: ignore[union-attr]

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return key in self._data  # type: ignore[operator]

    def __iter__(self):
        self._ensure()
        return iter(self._data)  # type: ignore[arg-type]


_PROVIDER_MODELS = _LazyProviderModels()

# ── Channel metadata ────────────────────────────────────────

CHANNEL_TYPES = {
    "telegram": {
        "label": "Telegram",
        "env_key": "telegram_bot_token",
        "config_section": "telegram",
        "token_help": "Get one from @BotFather on Telegram: https://t.me/BotFather",
    },
    "discord": {
        "label": "Discord",
        "env_key": "discord_bot_token",
        "config_section": "discord",
        "token_help": "Create one at: https://discord.com/developers/applications",
    },
    "slack": {
        "label": "Slack",
        "env_key": "slack_bot_token",
        "config_section": "slack",
        "token_help": "Create a Slack app at https://api.slack.com/apps -- enable Socket Mode",
    },
    "whatsapp": {
        "label": "WhatsApp",
        "env_key": "whatsapp_access_token",
        "config_section": "whatsapp",
        "token_help": "Set up at https://developers.facebook.com/apps -- WhatsApp Business API",
    },
}

# Permission flags that every agent should have. Entries here are
# forward-migrated onto existing agents at startup by
# _ensure_all_agent_permissions — add new flags here when introducing them.
_AGENT_PERMISSION_DEFAULTS: dict[str, object] = {
    "can_use_browser": True,
    "can_manage_cron": True,
}

# ── Config loading ──────────────────────────────────────────


def _load_config(mesh_path: Path | None = None) -> dict:
    """Load mesh config, agent definitions, and team metadata."""
    path = mesh_path or CONFIG_FILE
    cfg: dict = {
        "mesh": {"host": "0.0.0.0", "port": 8420},
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "agents": {},
    }
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            cfg.update(data)
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_data = yaml.safe_load(f) or {}
            cfg.setdefault("agents", {}).update(agents_data.get("agents", {}))

    # Load network config
    network_cfg = {}
    if NETWORK_FILE.exists():
        with open(NETWORK_FILE) as f:
            network_cfg = yaml.safe_load(f) or {}
    cfg["network"] = network_cfg
    return cfg


# ── Cross-thread / cross-process config lock (B-pre #2) ────────
#
# Recon widened B-pre #2: every ``config/agents.yaml`` writer was a bare
# ``open(..., "w")`` truncate-and-write — no lock, no atomic rename — and
# ``config/permissions.json``'s load->mutate->save was spread across
# helpers that each independently re-read the file, so two concurrent
# writers could both load the same baseline and the second save would
# silently clobber the first (the lost-update race this module used to
# just document below, unfixed). ``_config_lock()`` closes both gaps: ONE
# sidecar lock file guards the FULL load->mutate->save critical section
# for BOTH agents.yaml and permissions.json. A single shared lock (not one
# per file) because several flows write both in one logical operation
# (creating an agent touches agents.yaml, then permissions.json) — two
# separate locks would need a fixed acquisition order to avoid deadlock;
# one lock sidesteps that entirely.
#
# Built on ``fcntl.flock(LOCK_EX)`` (blocking), not a ``threading`` /
# ``asyncio`` primitive, because the CLI REPL, the dashboard process, the
# setup-wizard process, and the mesh host can all mutate these files from
# DIFFERENT PROCESSES — a Python-level lock only ever serializes within
# one process. Mirrors the flock context-manager precedent in
# ``src/browser/profile_schema.py`` (``_LockHandle``), with two
# deliberate differences: this lock BLOCKS (not ``LOCK_NB`` — callers need
# the critical section to eventually run, not to skip it under
# contention), and it is REENTRANT within a single thread via a
# ``threading.local`` depth counter. Reentrancy is required because
# several of the locked helpers below call each other synchronously
# (e.g. ``_remove_agent`` calls ``_remove_team_blackboard_permissions``,
# ``_ensure_all_agent_permissions`` calls ``_add_agent_permissions``) —
# flock is scoped to the OPEN FILE DESCRIPTION, so acquiring a second,
# independent fd for the same path from the same thread would
# self-deadlock without the depth guard. Only the OUTERMOST
# ``with _config_lock():`` in a thread's call stack opens an fd and
# flocks it; nested acquisitions from that same thread reuse it and are
# free. A different thread (or process) entering concurrently still opens
# its OWN fresh fd and genuinely blocks on the OS-level lock until the
# outermost holder's fd is unlocked — so cross-thread and cross-process
# mutual exclusion is real (enforced by the kernel), not just Python-level
# bookkeeping.
#
# The lock path is derived from ``AGENTS_FILE.parent`` at ACQUISITION time
# (not frozen into a module-level constant at import time) so it always
# sits next to whichever agents.yaml/permissions.json pair is actually in
# play — tests redirect ``AGENTS_FILE``/``PERMISSIONS_FILE`` to a temp
# ``config/`` dir per-test, and a frozen constant would silently keep
# locking the real repo's config dir instead of following that redirect.


def _config_lock_path() -> Path:
    """Return the sidecar lock file path, next to the live agents.yaml."""
    return AGENTS_FILE.parent / ".config.lock"


_config_lock_state = threading.local()


class _ConfigLock:
    """Context manager for the shared agents.yaml + permissions.json lock.

    Obtain via the ``_config_lock()`` factory: ``with _config_lock(): ...``.
    See the module comment above for the full design rationale (blocking +
    reentrant fcntl.flock on a fresh fd per outermost acquisition).
    """

    def __enter__(self) -> "_ConfigLock":
        depth = getattr(_config_lock_state, "depth", 0)
        if depth == 0:
            lock_path = _config_lock_path()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until any holder releases
            except BaseException:
                # flock raised after the fd was opened — close it before
                # propagating so a failed acquisition can't leak a descriptor
                # (repeated failures would otherwise exhaust the fd table).
                os.close(fd)
                raise
            _config_lock_state.fd = fd
        _config_lock_state.depth = depth + 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        depth = _config_lock_state.depth - 1
        _config_lock_state.depth = depth
        if depth == 0:
            fd = _config_lock_state.fd
            _config_lock_state.fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _config_lock() -> _ConfigLock:
    """Return the shared agents.yaml/permissions.json lock as a context manager.

    Every load->mutate->save helper for either file acquires this for its
    full critical section — see the module comment above ``_config_lock_path``.
    """
    return _ConfigLock()


def _load_agents_yaml() -> dict:
    """Load the raw agents.yaml document (``{"agents": {...}, ...}``).

    Read-only I/O helper — does NOT itself acquire ``_config_lock()``.
    Callers doing load->mutate->save MUST wrap the whole sequence in
    ``with _config_lock(): ...`` (locking only the read half wouldn't
    close the lost-update race — mirrors why ``_load_permissions``
    alone was never sufficient either).
    """
    if not AGENTS_FILE.exists():
        return {"agents": {}}
    with open(AGENTS_FILE) as f:
        return yaml.safe_load(f) or {"agents": {}}


def _save_agents_yaml(agents_cfg: dict) -> None:
    """Persist agents.yaml atomically (tempfile in the same dir + fsync +
    os.replace, via the shared ``atomic_write_text`` helper) — a concurrent
    reader always sees the old or the new complete file, never a torn one.

    Does NOT itself acquire ``_config_lock()`` — callers doing
    load->mutate->save MUST hold it across the whole sequence; this
    function alone only protects against torn reads, not lost updates.
    """
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.dump(agents_cfg, default_flow_style=False, sort_keys=False)
    atomic_write_text(AGENTS_FILE, content)


# Guards individual reads/writes of permissions.json against torn reads —
# held internally by _load_permissions and _save_permissions so a save
# never interleaves with a read, pairing with the atomic os.replace below
# so a reader never observes a half-written file (finding M14).
#
# This is now belt-and-suspenders: the real lost-update fix is the shared
# ``_config_lock()`` above, which every load->mutate->save call site holds
# across its FULL critical section (e.g. ``with _config_lock(): perms =
# _load_permissions(); ...mutate...; _save_permissions(perms)``) — and,
# being fcntl-based, also serializes across PROCESSES, which this
# in-process-only ``RLock`` never could. ``_PERMISSIONS_LOCK`` stays
# in place rather than being removed (B-pre #2's ratified decision keeps
# it as harmless belt-and-suspenders for this unit).
_PERMISSIONS_LOCK = threading.RLock()


def _load_permissions() -> dict:
    if not PERMISSIONS_FILE.exists():
        return {"permissions": {}}
    with _PERMISSIONS_LOCK:
        with open(PERMISSIONS_FILE) as f:
            return json.load(f)


def _save_permissions(perms: dict) -> None:
    """Persist permissions.json atomically (temp file + os.replace).

    The atomic rename guarantees a concurrent reader sees either the old
    or the new complete file, never a truncated one — so a crash or
    interleaved read can never produce a corrupt ACL (which fail-closed
    loading would then treat as deny-all).
    """
    PERMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(perms, indent=2) + "\n"
    with _PERMISSIONS_LOCK:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(PERMISSIONS_FILE.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass  # already closed by fdopen
            Path(tmp_path).unlink(missing_ok=True)
            raise
        try:
            os.replace(tmp_path, PERMISSIONS_FILE)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise


def _set_env_key(name: str, value: str, *, system: bool = False) -> None:
    """Set or update a key in the .env file.

    Args:
        system: If True, uses the ``OPENLEGION_SYSTEM_`` prefix (for
                provider keys). Otherwise uses ``OPENLEGION_CRED_``.
    """
    from src.host.credentials import AGENT_PREFIX, SYSTEM_PREFIX, _persist_to_env

    prefix = SYSTEM_PREFIX if system else AGENT_PREFIX
    env_key = f"{prefix}{name.upper()}"
    _persist_to_env(env_key, value, env_file=str(ENV_FILE))


# ── Docker helpers ──────────────────────────────────────────


def _check_docker_running() -> bool:
    """Verify Docker daemon is running and accessible."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        logger.debug("Docker daemon check failed: %s", e)
        return False


def _check_docker_image() -> bool:
    """Check if the agent Docker image exists."""
    try:
        import docker

        client = docker.from_env()
        client.images.get(DOCKER_IMAGE)
        return True
    except Exception as e:
        logger.debug("Docker image check failed: %s", e)
        return False


def _docker_image_is_stale(
    image_name: str = DOCKER_IMAGE,
    src_dirs: list | None = None,
    dockerfile_name: str = "Dockerfile.agent",
) -> bool:
    """Check if source files are newer than a Docker image."""
    try:
        from datetime import datetime, timezone

        import docker

        client = docker.from_env()
        image = client.images.get(image_name)
        created_str = image.attrs.get("Created", "")
        if not created_str:
            return True
        image_time = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

        check_dirs = src_dirs or [PROJECT_ROOT / "src" / "agent", PROJECT_ROOT / "src" / "shared"]
        for src_dir in check_dirs:
            if not src_dir.exists():
                continue
            for py_file in src_dir.rglob("*.py"):
                file_mtime = datetime.fromtimestamp(py_file.stat().st_mtime, tz=timezone.utc)
                if file_mtime > image_time:
                    return True

        dockerfile = PROJECT_ROOT / dockerfile_name
        if dockerfile.exists():
            df_mtime = datetime.fromtimestamp(dockerfile.stat().st_mtime, tz=timezone.utc)
            if df_mtime > image_time:
                return True
        return False
    except Exception as e:
        logger.debug("Docker image staleness check failed: %s", e)
        return False


def _ensure_docker_image() -> None:
    """Build the agent and browser Docker images if missing or stale."""
    if not _check_docker_image():
        _build_docker_image()
    elif _docker_image_is_stale():
        click.echo("Source code changed since last Docker build.")
        _build_docker_image()
    _ensure_browser_image()


def _check_browser_image() -> bool:
    """Check if the browser Docker image exists."""
    try:
        import docker

        client = docker.from_env()
        client.images.get(BROWSER_IMAGE)
        return True
    except Exception:
        return False


def _ensure_browser_image() -> None:
    """Build the browser Docker image if missing or stale."""
    if not _check_browser_image():
        _build_browser_image()
    elif _docker_image_is_stale(
        image_name=BROWSER_IMAGE,
        src_dirs=[PROJECT_ROOT / "src" / "browser", PROJECT_ROOT / "src" / "shared"],
        dockerfile_name="Dockerfile.browser",
    ):
        click.echo("Browser source changed since last build.")
        _build_browser_image()


def _build_browser_image() -> None:
    """Build the browser Docker image with visible progress."""
    click.echo("Building browser service Docker image...")
    click.echo("  First build downloads Camoufox + KasmVNC (~3 min). Rebuilds are fast.\n")
    proc = subprocess.Popen(
        [
            "docker",
            "build",
            "-t",
            BROWSER_IMAGE,
            "-f",
            "Dockerfile.browser",
            ".",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("Step ") or line.startswith("#") or "Downloading" in line or "Installing" in line:
            click.echo(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        click.echo("Browser image build failed. Run manually for full output:", err=True)
        click.echo(f"  docker build -t {BROWSER_IMAGE} -f Dockerfile.browser .", err=True)
        sys.exit(1)
    click.echo("\n  Browser Docker image built successfully.")


def _build_docker_image() -> None:
    """Build the agent Docker image with visible progress."""
    click.echo("Building agent Docker image...")
    click.echo("  First build downloads base image (~1 min). Rebuilds are fast.\n")
    proc = subprocess.Popen(
        [
            "docker",
            "build",
            "-t",
            DOCKER_IMAGE,
            "-f",
            "Dockerfile.agent",
            ".",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("Step ") or line.startswith("#") or "Downloading" in line or "Installing" in line:
            click.echo(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        click.echo("Docker build failed. Run manually for full output:", err=True)
        click.echo(f"  docker build -t {DOCKER_IMAGE} -f Dockerfile.agent .", err=True)
        sys.exit(1)
    click.echo("\n  Docker image built successfully.")


# ── Agent management ────────────────────────────────────────


def _add_agent_to_config(
    name: str,
    role: str,
    model: str,
    initial_instructions: str = "",
    initial_soul: str = "",
    initial_heartbeat: str = "",
    initial_interface: str = "",
    thinking: str = "",
    budget: dict | None = None,
    capabilities: list[str] | None = None,
    preferred_inputs: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    escalation_to: str | None = None,
    forbidden: list[str] | None = None,
) -> None:
    """Add an agent entry to agents.yaml.

    Task 8 adds five structured routing fields (``capabilities``,
    ``preferred_inputs``, ``expected_outputs``, ``escalation_to``,
    ``forbidden``). Each defaults to its empty form so existing
    callers that don't pass them continue to work unchanged. They are
    only persisted when non-empty so untouched yaml entries don't grow
    noisy default keys.
    """
    entry: dict = {
        "role": role,
        "model": model,
        "tools_dir": f"./agent_tools/{name}",
    }
    if initial_instructions:
        entry["initial_instructions"] = initial_instructions
    if initial_soul:
        entry["initial_soul"] = initial_soul
    if initial_heartbeat:
        entry["initial_heartbeat"] = initial_heartbeat
    if initial_interface:
        entry["initial_interface"] = initial_interface
    if thinking:
        entry["thinking"] = thinking
    if budget:
        entry["budget"] = budget
    # Task 8 — structured routing fields. Persist only when non-empty
    # so existing minimal yaml entries don't grow empty default keys.
    if capabilities:
        entry["capabilities"] = list(capabilities)
    if preferred_inputs:
        entry["preferred_inputs"] = list(preferred_inputs)
    if expected_outputs:
        entry["expected_outputs"] = list(expected_outputs)
    if escalation_to:
        entry["escalation_to"] = escalation_to
    if forbidden:
        entry["forbidden"] = list(forbidden)

    with _config_lock():
        agents_cfg = _load_agents_yaml()
        if "agents" not in agents_cfg:
            agents_cfg["agents"] = {}
        agents_cfg["agents"][name] = entry
        _save_agents_yaml(agents_cfg)


# Coordination grants layered on top of the base defaults for a fully-fledged
# created agent. Applied by EVERY human create path (`_create_agent`: setup
# wizard, REPL, dashboard no-template) AND the operator/mesh create endpoint,
# so a manually-created agent is no more limited than an operator- or
# template-created one. Capability toggles (browser/internet/cron/spawn/
# wallet) live in the base dict, not here. The mesh `create_custom_agent`
# endpoint imports this same constant rather than duplicating it.
#
# blackboard_read carries NO wildcard (ratified #5): the base defaults grant
# the agent's private ``teams/{name}/*`` namespace (which also satisfies the
# /mesh/register auto-watch gate on blackboard_read being truthy), and team
# membership adds ``teams/{team}/*`` on join. With the agent-side standalone
# guards deleted, the host ACL is THE read boundary — a ``*`` here would hand
# every never-teamed worker fleet-wide blackboard reads, strictly wider than
# a team member (whose join strips ``*``). can_publish/can_subscribe keep the
# wildcard because pubsub is additionally prefix-gated host-side per caller.
_DEFAULT_AGENT_COORDINATION_PERMS: dict = {
    "blackboard_read": [],
    # ``output/*`` + ``artifacts/*`` removed in Phase-2 unit 4 — those
    # payload flows moved to the Team Drive; the blackboard is signals-only.
    "blackboard_write": ["tasks/*", "context/*", "status/*"],
    "can_publish": ["*"],
    "can_subscribe": ["*"],
}


def _add_agent_permissions(name: str, permissions: dict | None = None, *, from_template: bool = False) -> None:
    """Add default permissions for a new agent.

    If collaboration mode is enabled in mesh.yaml, agents can message
    all other agents. Otherwise they can only message mesh.

    If *permissions* is provided (from a template), its ``blackboard_read``,
    ``blackboard_write``, ``can_publish``, and ``can_subscribe`` entries
    are merged into the defaults.
    """
    with _config_lock():
        cfg = _load_config()
        collab = cfg.get("collaboration", True)

        perms = _load_permissions()
        if collab:
            other_agents = [a for a in cfg.get("agents", {}) if a != name]
            can_message = ["*"] if other_agents else []
        else:
            can_message = []

        # Self pattern: every WORKER always holds ``teams/{name}/*`` on both
        # blackboard fields — its private team-of-one namespace (ratified #5).
        # Invariant: "self always; team while member". The operator keeps its
        # explicit ["*"] grants from _ensure_operator_agent instead.
        self_patterns = [f"teams/{name}/*"] if name != _OPERATOR_AGENT_ID else []

        agent_perms: dict = {
            "can_message": can_message,
            "can_publish": ["*"] if collab else [f"{name}_complete"],
            "can_subscribe": ["*"] if collab else [],
            "blackboard_read": list(self_patterns),
            "blackboard_write": list(self_patterns),
            "allowed_apis": ["llm", "image_gen"],
            "allowed_credentials": ["*"],
            # Default capability set for every new agent: browser + internet +
            # schedules (cron) + ephemeral fleet-spawn ON; wallet OFF (the sole
            # privileged grant — spending money requires explicit user setup).
            # NOTE: ``can_spawn=True`` applies to NAMED agents created here. The
            # ephemeral ``spawn-*`` agents they create resolve permissions via
            # ``PermissionMatrix.get_permissions`` ("default" record / bare
            # fallback → field default False), so a spawned agent CANNOT itself
            # spawn — the recursion wall is bounded at one level. A template may
            # flip any of these off via the merge below.
            "can_use_browser": True,
            "can_use_internet": True,
            "can_manage_cron": True,
            "can_spawn": True,
            "can_use_wallet": False,
        }

        # Merge template permissions into defaults
        if permissions:
            for key in ("blackboard_read", "blackboard_write", "can_publish", "can_subscribe"):
                tpl_values = permissions.get(key)
                if tpl_values and isinstance(tpl_values, list):
                    existing = set(agent_perms.get(key, []))
                    existing.update(tpl_values)
                    agent_perms[key] = sorted(existing)
            # Defense in depth (ratified #5): no create path may hand a WORKER a
            # blackboard wildcard — the host ACL is the read boundary now that
            # the agent-side standalone guards are gone, and team joins strip
            # ``*`` anyway (an agent must never start wider than a team member
            # ends up). The operator's explicit ["*"] comes from
            # _ensure_operator_agent, not this path.
            if name != _OPERATOR_AGENT_ID:
                for key in ("blackboard_read", "blackboard_write"):
                    agent_perms[key] = [p for p in agent_perms.get(key, []) if p != "*"]
            # Boolean flags — template can override defaults. Includes the
            # six control-plane permissions from Task 3 so the operator's
            # explicit grants in ``_ensure_operator_agent`` get persisted to
            # permissions.json instead of being silently dropped.
            for key in (
                "can_use_browser",
                "can_use_internet",
                "can_spawn",
                "can_manage_cron",
                "can_manage_fleet",
                "can_manage_teams",
                "can_edit_agent_config",
                "can_view_fleet_metrics",
                "can_request_user_credentials",
                "can_use_wallet",
            ):
                if key in permissions:
                    value = bool(permissions[key])
                    # L4: clamp the irreversible-grant ceiling. A fleet template
                    # must never mint an agent that can spawn other agents or
                    # spend from the wallet — those are operator/user-only grants
                    # (mirrors _OPERATOR_PERMISSION_CEILING in operator_tools.py).
                    # None of the shipped templates set these true, so legitimate
                    # template application is unaffected.
                    if from_template and key in _TEMPLATE_PERMISSION_CEILING and value:
                        logger.warning(
                            "Template for agent '%s' tried to grant '%s'=true; "
                            "clamping to false (irreversible-grant ceiling)",
                            name,
                            key,
                        )
                        value = False
                    agent_perms[key] = value

        perms["permissions"][name] = agent_perms
        _save_permissions(perms)


def _ensure_all_agent_permissions() -> None:
    """Backfill permissions for agents missing from permissions.json, and
    forward-migrate any missing boolean capability flags for existing agents."""
    with _config_lock():
        cfg = _load_config()
        perms = _load_permissions()
        existing = set(perms.get("permissions", {}).keys())
        for name in cfg.get("agents", {}):
            if name not in existing:
                _add_agent_permissions(name)

        # Reload after potentially writing new agents so the migration sees all
        # agents, including ones just added above.
        perms = _load_permissions()

        # Forward-migrate: add any permission flags introduced after an agent's
        # initial permissions entry was created (keyed in _AGENT_PERMISSION_DEFAULTS).
        changed = False
        for agent_id, agent_perms in perms.get("permissions", {}).items():
            if agent_id == "default":
                continue
            for flag, default_val in _AGENT_PERMISSION_DEFAULTS.items():
                if flag not in agent_perms:
                    agent_perms[flag] = default_val
                    changed = True

        # Solo self-pattern default (ratified #5 — solo agent = team-of-one):
        # every TEAMLESS worker holds its private ``teams/{agent_id}/*``
        # pattern on both blackboard fields ("self always"). Additive only —
        # existing patterns are never removed except the untouched pre-#5
        # default read wildcard (below). The live enforcement is the
        # resolution-time carve-out in PermissionMatrix._is_own_namespace;
        # this keeps the FILE state matching the invariant.
        try:
            team_map = _open_teams_store().agent_team_map()
        except Exception:
            team_map = {}
        for agent_id, agent_perms in perms.get("permissions", {}).items():
            if agent_id in ("default", _OPERATOR_AGENT_ID):
                continue
            if team_map.get(agent_id):
                continue
            self_pattern = f"teams/{agent_id}/*"
            # Narrow the pre-#5 default read wildcard: a teamless worker whose
            # blackboard_read is exactly the untouched old default shape
            # (["*"], or ["*"] + its own self pattern) drops the ``*`` — with
            # the agent-side standalone guards gone, leaving it would grant
            # fleet-wide reads no team member has. Any other (human-customized)
            # pattern list is left alone.
            reads = agent_perms.get("blackboard_read") or []
            if set(reads) in ({"*"}, {"*", self_pattern}):
                agent_perms["blackboard_read"] = [self_pattern]
                changed = True
            for field in ("blackboard_read", "blackboard_write"):
                patterns = agent_perms.get(field) or []
                if self_pattern not in patterns:
                    agent_perms[field] = [*patterns, self_pattern]
                    changed = True
        if changed:
            _save_permissions(perms)


def _set_collaborative_permissions() -> None:
    """Update all agent permissions to allow inter-agent messaging and pub/sub."""
    with _config_lock():
        perms = _load_permissions()
        for name, p in perms.get("permissions", {}).items():
            if name == "default":
                continue
            if "*" not in p.get("can_message", []):
                p["can_message"] = list({*p.get("can_message", []), "*"})
            if "*" not in p.get("can_publish", []):
                p["can_publish"] = list({*p.get("can_publish", []), "*"})
            if "*" not in p.get("can_subscribe", []):
                p["can_subscribe"] = list({*p.get("can_subscribe", []), "*"})
            p["allowed_credentials"] = ["*"]
        _save_permissions(perms)


def _validate_agent_name(name: str) -> str:
    """Validate and return a safe agent name.

    Rejects path traversal, slashes, non-alphanumeric chars
    (aside from hyphens and underscores), and reserved internal names.
    """
    if not name or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError(
            f"Invalid agent name '{name}': must be 1–64 alphanumeric chars, "
            "hyphens, or underscores (must start with a letter or digit)."
        )
    if name in RESERVED_AGENT_IDS:
        raise ValueError(f"Agent name '{name}' is reserved for internal use")
    return name


def _reject_agent_team_collision(name: str) -> None:
    """Cross-namespace collision guard (ratified #5 makes it load-bearing).

    Agents and teams share one namespace downstream (``teams/{id}/*``
    blackboard prefixes, ``TEAM_NAME`` env): a solo agent's private
    team-of-one scope is ``teams/{agent_id}/*``, so an agent named after
    an existing team would silently gain that team's namespace. Reject at
    every agent-create path; the team-create endpoints enforce the mirror
    direction. Best-effort on store errors: an unreadable teams DB must
    not block agent creation (the mirror guard still holds forward).
    """
    try:
        exists = _open_teams_store().team_exists(name)
    except Exception as e:
        logger.warning("Team-collision check skipped for '%s': %s", name, e)
        return
    if exists:
        raise ValueError(
            f"Agent name '{name}' conflicts with an existing team — "
            "teams and agents share one namespace. Pick a different name."
        )


def _create_agent(
    name: str,
    description: str,
    model: str,
) -> None:
    """Create an agent: config, permissions, tools directory."""
    name = _validate_agent_name(name)
    _reject_agent_team_collision(name)
    _add_agent_to_config(name, description, model)
    # Apply the same coordination defaults the operator/mesh create path uses
    # so a human-created (no-template) agent isn't second-class — empty
    # blackboard ACLs would lock it out of the coordination protocol.
    _add_agent_permissions(name, permissions=_DEFAULT_AGENT_COORDINATION_PERMS)
    tools_dir = PROJECT_ROOT / "agent_tools" / name
    tools_dir.mkdir(parents=True, exist_ok=True)


def _suppress_host_logs() -> None:
    """Set host-side loggers to WARNING for clean CLI output."""
    for name in [
        "host",
        "host.credentials",
        "host.mesh",
        "host.costs",
        "host.permissions",
        "host.cron",
        "host.webhooks",
        "host.health",
        "host.lanes",
        "host.runtime",
        "channels",
        "channels.base",
        "channels.telegram",
        "channels.discord",
        "channels.slack",
        "channels.whatsapp",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Silence third-party library internal loggers — channel adapters
    # handle all error reporting via their own loggers.  Without this,
    # python-telegram-bot's default error callback prints full tracebacks
    # for transient Conflict errors (duplicate polling sessions).
    for name in ["telegram", "httpx", "httpcore"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _ensure_pairing_code(pairing_path: Path) -> str | None:
    """Ensure a pairing file exists with a code. Returns code if unpaired, None if already paired."""
    data: dict = {}
    if pairing_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            data = json.loads(pairing_path.read_text())
    if data.get("owner"):
        return None
    code = data.get("pairing_code")
    if not code:
        code = secrets.token_hex(8)
        data = {"owner": None, "allowed": [], "pairing_code": code}
        pairing_path.parent.mkdir(parents=True, exist_ok=True)
        pairing_path.write_text(json.dumps(data, indent=2) + "\n")
    return code


def _get_default_model() -> str:
    cfg = _load_config()
    return cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")


# ── Team management ──────────────────────────────────────
#
# Team identity, metadata, and membership live in the SQLite-backed
# ``TeamStore`` (src/host/teams.py) — THE single authority since the
# Phase-1 "team as first-class" repoint. The helpers kept here are the
# permissions.json wiring invoked by the endpoint layer around store
# membership ops (``_add_team_blackboard_permissions`` /
# ``_remove_team_blackboard_permissions``) plus ``_open_teams_store``
# for CLI paths that run without a Runtime object in scope.


def _open_teams_store():
    """Construct a TeamStore handle on demand (no module-level singleton).

    SQLite WAL makes cross-process access safe, so detached CLI paths
    (``_remove_agent``, repl listings) open their own handle instead of
    threading the runtime's instance through every call chain.
    """
    from src.host.teams import TeamStore

    return TeamStore(
        db_path=os.environ.get("OPENLEGION_TEAMS_DB", "data/teams.db"),
        teams_dir=TEAMS_DIR,
    )


def _backfill_capabilities_for_existing_agents() -> None:
    """One-shot Task 8 back-fill of structured routing fields on agents.yaml.

    For any agent that has an empty ``capabilities`` list AND a non-empty
    ``initial_interface`` block, parse the markdown headings and persist
    the derived structured fields back to ``agents.yaml``. Skips agents
    that already declare ``capabilities`` (idempotent: a populated field
    is the source of truth).

    This runs once at startup. Failure to parse any single agent never
    raises — the field stays empty and routing falls back to existing
    behaviour.
    """
    if not AGENTS_FILE.exists():
        return

    from src.agent.workspace import _parse_interface_text

    with _config_lock():
        try:
            agents_cfg = _load_agents_yaml()
        except Exception:
            return

        agents = agents_cfg.get("agents", {})
        if not isinstance(agents, dict) or not agents:
            return

        changed = False
        for _agent_name, entry in agents.items():
            if not isinstance(entry, dict):
                continue
            # Already declared — structured field wins, do nothing.
            if entry.get("capabilities"):
                continue
            interface_text = entry.get("initial_interface") or ""
            if not interface_text:
                continue

            try:
                derived = _parse_interface_text(interface_text)
            except Exception:
                continue

            # Only persist when the derivation produced something useful.
            if not (
                derived.get("capabilities")
                or derived.get("preferred_inputs")
                or derived.get("expected_outputs")
                or derived.get("escalation_to")
                or derived.get("forbidden")
            ):
                continue

            if derived.get("capabilities"):
                entry["capabilities"] = list(derived["capabilities"])
            if derived.get("preferred_inputs"):
                entry["preferred_inputs"] = list(derived["preferred_inputs"])
            if derived.get("expected_outputs"):
                entry["expected_outputs"] = list(derived["expected_outputs"])
            if derived.get("escalation_to"):
                entry["escalation_to"] = derived["escalation_to"]
            if derived.get("forbidden"):
                entry["forbidden"] = list(derived["forbidden"])
            changed = True

        if changed:
            try:
                _save_agents_yaml(agents_cfg)
            except Exception:
                logger.exception("Failed to persist back-filled agent capabilities")


def _set_agent_status(name: str, status: str) -> None:
    """Persist a per-agent ``status`` flag in agents.yaml.

    ``status="archived"`` stops scheduling without deleting workspace
    or history. Live containers continue running until the next restart;
    operator tools that archive an agent should also stop the container
    via the runtime backend (the host endpoint handles that).
    """
    with _config_lock():
        if not AGENTS_FILE.exists():
            raise ValueError("Agents config not found")
        agents_cfg = _load_agents_yaml()
        agents = agents_cfg.get("agents", {})
        if name not in agents:
            raise ValueError(f"Agent '{name}' not found")
        agents[name]["status"] = status
        _save_agents_yaml(agents_cfg)


def _archive_agent(name: str) -> None:
    """Mark an agent as archived (stop scheduling, retain workspace + history)."""
    _set_agent_status(name, "archived")


def _unarchive_agent(name: str) -> None:
    """Re-activate an archived agent."""
    _set_agent_status(name, "active")


def _agent_status(name: str) -> str:
    """Read an agent's status. Returns ``"active"`` for legacy rows missing the field."""
    if not AGENTS_FILE.exists():
        raise ValueError("Agents config not found")
    with open(AGENTS_FILE) as f:
        agents_cfg = yaml.safe_load(f) or {"agents": {}}
    agents = agents_cfg.get("agents", {})
    if name not in agents:
        raise ValueError(f"Agent '{name}' not found")
    return agents[name].get("status", "active") or "active"


def _add_team_blackboard_permissions(agent: str, team: str) -> None:
    """Grant blackboard access for a team member.

    Grants ``teams/{team}/*`` and strips any pre-existing ``*``
    wildcard so an agent cannot accumulate fleet-wide reach via team
    moves (Task 5). The MeshClient auto-prefixes all blackboard keys
    with the team namespace, so agents use natural keys
    (``context/market``) which are transparently stored as
    ``teams/{team}/context/market``. Each team's data lives
    under its own namespace.

    Pre-Task-5 fleets had ``["*"]`` ACLs that survived membership churn;
    actively-managed fleets are narrowed here at every join. Legacy
    in-place ``*`` ACLs that haven't been touched stay intact until
    enforce mode tightens the read/write hot path.
    """
    with _config_lock():
        perms = _load_permissions()
        agent_perms = perms.get("permissions", {}).get(agent)
        if agent_perms is None:
            return
        team_pattern = f"teams/{team}/*"
        self_pattern = f"teams/{agent}/*"
        for field in ("blackboard_read", "blackboard_write"):
            patterns = agent_perms.get(field, [])
            # Strip any wildcard — replaced by the explicit team pattern
            # below. This is the active narrowing for Task 5: an agent that
            # previously had ``["*"]`` and is then added to a team ends
            # up with its self + team patterns.
            narrowed = [p for p in patterns if p != "*"]
            if team_pattern not in narrowed:
                narrowed.append(team_pattern)
            # Invariant (ratified #5): "self always; team while member" — the
            # private team-of-one pattern rides alongside the team pattern.
            if self_pattern not in narrowed:
                narrowed.append(self_pattern)
            agent_perms[field] = narrowed
        _save_permissions(perms)


def _remove_team_blackboard_permissions(agent: str, team: str) -> None:
    """Revoke team blackboard access when an agent leaves a team.

    Strips the team namespace pattern AND any pre-existing ``*``
    wildcard, then restores the agent's self pattern
    (``teams/{agent}/*``) so a leaver lands in a working private
    team-of-one namespace instead of an empty-ACL lockout (ratified #5:
    "self always; team while member"). Wildcard stripping matches the
    add-side narrowing so an agent that ever had ``*`` does not
    reacquire fleet-wide reach by being added to and then removed from
    a team (Task 5).
    """
    with _config_lock():
        perms = _load_permissions()
        agent_perms = perms.get("permissions", {}).get(agent)
        if agent_perms is None:
            return
        team_pattern = f"teams/{team}/*"
        self_pattern = f"teams/{agent}/*"
        for field in ("blackboard_read", "blackboard_write"):
            patterns = agent_perms.get(field, [])
            # Drop both the target team pattern and any surviving ``*``.
            # The wildcard strip is the safety net for an agent whose ACL
            # was never re-narrowed (e.g. config-edited directly): once the
            # membership system touches it on remove, the wildcard goes.
            remaining = [p for p in patterns if p != team_pattern and p != "*"]
            if self_pattern not in remaining:
                remaining.append(self_pattern)
            agent_perms[field] = remaining
        _save_permissions(perms)


def _remove_agent(name: str, stop_container: bool = False) -> None:
    """Remove an agent from config, permissions, and any team membership.

    If *stop_container* is True, attempt to stop and remove the Docker
    container named ``openlegion_{name}``.  Failures are silently ignored
    so that config removal always succeeds.
    """
    if stop_container:
        try:
            import docker

            client = docker.from_env()
            container = client.containers.get(f"openlegion_{name}")
            container.stop(timeout=10)
            container.remove()
        except Exception as e:
            logger.warning("Failed to stop/remove container for '%s': %s", name, e)

    # Remove from team if member (also clears standing goals). Wraps the
    # blackboard-permission narrowing AND the agents.yaml/permissions.json
    # removal below in ONE lock acquisition so a concurrent writer can't
    # observe (or clobber) a half-removed agent — reentrant on
    # ``_remove_team_blackboard_permissions``'s own inner acquisition.
    with _config_lock():
        with contextlib.suppress(Exception):
            team = _open_teams_store().remove_agent(name)
            if team:
                _remove_team_blackboard_permissions(name, team)

        # Remove from agents.yaml
        if AGENTS_FILE.exists():
            agents_cfg = _load_agents_yaml()
            agents_cfg.get("agents", {}).pop(name, None)
            _save_agents_yaml(agents_cfg)

        # Remove from permissions
        perms = _load_permissions()
        perms.get("permissions", {}).pop(name, None)
        _save_permissions(perms)


def _pick_model_interactive(
    default_model: str,
    label: str = "current",
    credential_vault: object | None = None,
) -> str:
    """Show model picker for the default model's provider. Returns selected model.

    When *credential_vault* is provided, only shows models from providers
    that have credentials configured.
    """
    provider = default_model.split("/")[0] if "/" in default_model else "anthropic"

    # Filter to providers with credentials if vault is available
    active_providers: set[str] | None = None
    if credential_vault is not None and hasattr(credential_vault, "get_providers_with_credentials"):
        active_providers = credential_vault.get_providers_with_credentials()
        if active_providers and provider not in active_providers:
            # Current provider has no credentials — switch to first that does
            for p in _PROVIDER_MODELS:
                if p in active_providers:
                    provider = p
                    break

    if active_providers:
        available = {p: m for p, m in _PROVIDER_MODELS.items() if p in active_providers}
    else:
        available = _PROVIDER_MODELS

    models = available.get(provider, [default_model])
    default_idx = 1
    for i, m in enumerate(models, 1):
        marker = f" ({label})" if m == default_model else ""
        click.echo(f"  {i}. {m}{marker}")
        if m == default_model:
            default_idx = i
    model_choice = click.prompt(
        "Model",
        type=click.IntRange(1, len(models)),
        default=default_idx,
    )
    return models[model_choice - 1]


def _validate_agent_template(template: dict) -> list[str]:
    """Lightweight validator for fleet templates (Task 8).

    Returns a list of human-readable warnings — informational, not
    rejection. Callers (``_load_templates``) log them at info level so
    template authors notice missing routing metadata without breaking
    existing workflows.

    Validates:
    - If ``capabilities`` is empty AND there is no ``initial_interface``
      / ``interface`` block to derive from, warn naming the agent.
    - If ``escalation_to`` references an agent ID that doesn't exist in
      the same template, warn naming both.

    Empty templates / missing ``agents`` keys produce no warnings —
    that's an unrelated structural issue surfaced elsewhere.
    """
    warnings: list[str] = []
    tpl_name = template.get("name", "<unnamed>")
    agents = template.get("agents") or {}
    if not isinstance(agents, dict):
        return warnings

    agent_ids = set(agents.keys())
    for agent_name, agent_def in agents.items():
        if not isinstance(agent_def, dict):
            continue
        capabilities = agent_def.get("capabilities") or []
        interface_text = agent_def.get("initial_interface") or agent_def.get("interface") or ""
        if not capabilities and not interface_text:
            warnings.append(
                f"template '{tpl_name}' agent '{agent_name}': no "
                "structured 'capabilities' and no INTERFACE.md to derive "
                "from — operator routing will see an empty interface."
            )

        escalation = agent_def.get("escalation_to")
        if escalation and isinstance(escalation, str):
            if escalation not in agent_ids:
                warnings.append(
                    f"template '{tpl_name}' agent '{agent_name}': "
                    f"escalation_to='{escalation}' is not defined in "
                    "this template (cross-template escalations should "
                    "use a fleet-global agent like 'operator')."
                )
    return warnings


def _load_templates() -> dict[str, dict]:
    """Load available team templates from src/templates/.

    Each template is run through ``_validate_agent_template`` for
    lightweight informational validation (Task 8). Warnings are logged
    at info level — they never reject a template.
    """
    available: dict[str, dict] = {}
    if not TEMPLATES_DIR.exists():
        return available
    for tpl_file in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(tpl_file) as f:
            tpl = yaml.safe_load(f) or {}
        name = tpl.get("name", tpl_file.stem)
        available[name] = tpl
        for warning in _validate_agent_template(tpl):
            logger.info("template-validate: %s", warning)
    return available


def _apply_template(
    template_name: str,
    tpl: dict,
    agent_overrides: dict[str, dict] | None = None,
) -> list[str]:
    """Apply a team template, creating all agents. Returns list of agent names.

    ``agent_overrides`` (PR-N) is an optional ``{agent_name: {field: value}}``
    map. v2 supports per-agent ``model``, ``instructions``, ``soul``,
    ``heartbeat``, ``interface``, and ``role`` overrides applied on top of
    the template's defaults. The mesh route validates the shape and
    contents UPFRONT — by the time we get here, the input is trusted.
    """
    cfg = _load_config()
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    tpl_agents = tpl.get("agents", {})
    overrides = agent_overrides or {}
    created: list[str] = []

    # Cross-namespace collision guard — UPFRONT over every slot name so a
    # collision rejects the whole apply before any agent is created.
    for slot_name in tpl_agents:
        _reject_agent_team_collision(_validate_agent_name(slot_name))

    # Load existing agents to avoid silent overwrites
    existing_agents: set[str] = set(_load_agents_yaml().get("agents", {}).keys())

    # Lazy import — keeps the CLI-only path from pulling in shared/models
    # at module-import time (template loader runs from setup wizard too).
    from src.shared.models import resolve_slot_model

    for agent_name, agent_def in tpl_agents.items():
        agent_name = _validate_agent_name(agent_name)
        if agent_name in existing_agents:
            click.echo(f"  Skipping '{agent_name}' — agent already exists")
            continue
        per_agent_override = overrides.get(agent_name) or {}
        # P1.2 — route through the shared helper so a slot with
        # ``model: null`` (explicit None) coerces to the config default
        # instead of crashing on ``.replace()``. ``_apply_template`` has
        # no top-level model_override channel (the mesh route does), so
        # pass an empty string for that argument; precedence is
        # slot override > template default > config default.
        model = resolve_slot_model(
            agent_name,
            agent_def,
            overrides,
            "",
            default_model,
        )
        instructions = agent_def.get("instructions", "") or agent_def.get("system_prompt", "")
        if "instructions" in per_agent_override:
            instructions = per_agent_override["instructions"]
        soul = agent_def.get("soul", "")
        if "soul" in per_agent_override:
            soul = per_agent_override["soul"]
        heartbeat = agent_def.get("heartbeat", "")
        if "heartbeat" in per_agent_override:
            heartbeat = per_agent_override["heartbeat"]
        interface = agent_def.get("initial_interface", "") or agent_def.get("interface", "")
        if "interface" in per_agent_override:
            interface = per_agent_override["interface"]
        role = agent_def.get("role", agent_name)
        if "role" in per_agent_override:
            role = per_agent_override["role"]
        thinking = agent_def.get("thinking", "")
        budget = agent_def.get("budget")
        agent_permissions = agent_def.get("permissions")

        _add_agent_to_config(
            name=agent_name,
            role=role,
            model=model,
            initial_instructions=instructions,
            initial_soul=soul,
            initial_heartbeat=heartbeat,
            initial_interface=interface,
            thinking=thinking,
            budget=budget,
            capabilities=agent_def.get("capabilities") or [],
            preferred_inputs=agent_def.get("preferred_inputs") or [],
            expected_outputs=agent_def.get("expected_outputs") or [],
            escalation_to=agent_def.get("escalation_to"),
            forbidden=agent_def.get("forbidden") or [],
        )
        _add_agent_permissions(agent_name, permissions=agent_permissions, from_template=True)
        tools_dir = PROJECT_ROOT / "agent_tools" / agent_name
        tools_dir.mkdir(parents=True, exist_ok=True)
        created.append(agent_name)

    return created


def _load_tool_templates() -> list[dict]:
    """Load individual agent role templates from team templates.

    Returns a flat list of agent definitions extracted from all team templates,
    each identified by ``"id": "team/agent_name"``.
    """
    templates = _load_templates()
    result: list[dict] = []
    for tpl_name, tpl in templates.items():
        tpl_desc = tpl.get("description", "")
        for agent_name, agent_def in tpl.get("agents", {}).items():
            result.append(
                {
                    "id": f"{tpl_name}/{agent_name}",
                    "name": agent_name,
                    "source": tpl_name,
                    "source_description": tpl_desc,
                    "role": agent_def.get("role", agent_name),
                    "has_instructions": bool(agent_def.get("instructions") or agent_def.get("system_prompt")),
                    "has_soul": bool(agent_def.get("soul")),
                    "has_heartbeat": bool(agent_def.get("heartbeat")),
                    "thinking": agent_def.get("thinking", ""),
                }
            )
    return result


def _create_agent_from_template(
    name: str,
    template_id: str,
    model: str,
) -> None:
    """Create an agent applying a tool template's config.

    *template_id* has the form ``"team/agent_name"`` (e.g. ``"devteam/engineer"``).
    """
    parts = template_id.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid template id: {template_id}")
    tpl_source, tpl_agent = parts
    templates = _load_templates()
    tpl = templates.get(tpl_source)
    if not tpl:
        raise ValueError(f"Template source not found: {tpl_source}")
    agent_def = tpl.get("agents", {}).get(tpl_agent)
    if not agent_def:
        raise ValueError(f"Agent '{tpl_agent}' not found in template '{tpl_source}'")

    name = _validate_agent_name(name)
    _reject_agent_team_collision(name)
    cfg = _load_config()
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    # P1.2 — coerce explicit ``model: null`` (None) to the config
    # default before ``.replace()``. ``dict.get(key, default)`` returns
    # the key's actual value when it exists EVEN IF that value is None;
    # the default only applies when the key is ABSENT.
    resolved_model = model or agent_def.get("model") or default_model
    resolved_model = resolved_model.replace("{default_model}", default_model)

    # BYOK-safety: reject up front if the resolved model's provider
    # has no credentials in env. Mirrors the mesh-side check in
    # ``create_custom_agent`` so ``apply_template`` doesn't silently
    # mint dead-on-arrival agents (Bug 5).
    from src.shared.models import (
        get_available_providers,
        missing_provider_key_message,
        model_not_compatible_message,
        resolve_provider_for_model,
    )

    provider = resolve_provider_for_model(resolved_model)
    if provider:
        available = get_available_providers()
        if provider not in available:
            raise ValueError(
                missing_provider_key_message(resolved_model, provider, available),
            )

    # Credential-kind-aware check (Fix 2 in seam follow-up): OAuth-only
    # providers only accept specific models. Use a fresh vault so we
    # pick up env-configured OAuth state without depending on a running
    # mesh process — this path is CLI-side at fleet apply time.
    try:
        from src.host.credentials import CredentialVault

        _vault = CredentialVault()
        _vault._load_credentials()
        _compatible, _reason = _vault.is_model_compatible(resolved_model)
        if not _compatible:
            raise ValueError(_reason or model_not_compatible_message(resolved_model))
    except ImportError:
        # ``host`` package may not be importable in trimmed test harnesses;
        # the mesh-side check still fires on the create_agent path.
        pass

    instructions = agent_def.get("instructions", "") or agent_def.get("system_prompt", "")
    soul = agent_def.get("soul", "")
    heartbeat = agent_def.get("heartbeat", "")
    interface = agent_def.get("initial_interface", "") or agent_def.get("interface", "")
    thinking = agent_def.get("thinking", "")
    budget = agent_def.get("budget")
    agent_permissions = agent_def.get("permissions")

    _add_agent_to_config(
        name=name,
        role=agent_def.get("role", name),
        model=resolved_model,
        initial_instructions=instructions,
        initial_soul=soul,
        initial_heartbeat=heartbeat,
        initial_interface=interface,
        thinking=thinking,
        budget=budget,
        capabilities=agent_def.get("capabilities") or [],
        preferred_inputs=agent_def.get("preferred_inputs") or [],
        expected_outputs=agent_def.get("expected_outputs") or [],
        escalation_to=agent_def.get("escalation_to"),
        forbidden=agent_def.get("forbidden") or [],
    )
    _add_agent_permissions(name, permissions=agent_permissions, from_template=True)
    tools_dir = PROJECT_ROOT / "agent_tools" / name
    tools_dir.mkdir(parents=True, exist_ok=True)


def _default_description(name: str) -> str:
    """Return a default agent description."""
    return f"General-purpose {name} agent"


def _update_agent_field(name: str, field: str, value) -> None:
    """Update a single field in agents.yaml for an agent."""
    with _config_lock():
        agents_cfg = _load_agents_yaml()
        if name in agents_cfg.get("agents", {}):
            agents_cfg["agents"][name][field] = value
            _save_agents_yaml(agents_cfg)


def _update_network_config(field: str, value) -> None:
    """Update a field in network.yaml."""
    if NETWORK_FILE.exists():
        with open(NETWORK_FILE) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data[field] = value
    NETWORK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NETWORK_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


_THINKING_LEVELS = list(THINKING_LEVELS)
_THINKING_LABELS = {
    "off": "off",
    "low": "low (5K tokens)",
    "medium": "medium (10K)",
    "high": "high (25K)",
}


def _edit_agent_interactive(
    name: str,
    credential_vault: object | None = None,
) -> str | None:
    """Interactive property editor for an agent. Reads fresh config.

    Returns the field name that was changed (``"model"``, ``"role"``,
    ``"budget"``, ``"thinking"``), or ``None`` if nothing changed.
    Callers decide how to apply the change (restart hint, live
    restart, cost-tracker update, etc.). MCP servers are managed on
    the dashboard Connectors page, not here.
    """
    cfg = _load_config()
    agent_cfg = cfg.get("agents", {}).get(name, {})
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

    current_model = agent_cfg.get("model", default_model)
    current_desc = agent_cfg.get("role", "")
    budget_cfg = agent_cfg.get("budget", {})
    current_budget = budget_cfg.get("daily_usd") if budget_cfg else None
    current_thinking = agent_cfg.get("thinking", "off") or "off"

    click.echo(f"\n  {name}")
    click.echo(f"  Model:       {current_model}")
    click.echo(f"  Description: {current_desc or '(none)'}")
    if current_budget is not None:
        click.echo(f"  Budget:      ${current_budget:.2f}/day")
    click.echo(f"  Thinking:    {current_thinking}")
    click.echo("\n  What to change?\n")

    options = [
        ("model", current_model),
        ("description", truncate(current_desc, 50) or "(none)"),
        ("budget", f"${current_budget:.2f}/day" if current_budget is not None else "(none)"),
        ("thinking", current_thinking),
    ]

    for i, (label, val) in enumerate(options, 1):
        click.echo(f"  {i}. {label:<16} {val}")

    choice = click.prompt(
        "\n  Select",
        type=click.IntRange(1, len(options)),
        default=1,
    )

    if choice == 1:  # model
        new_model = _pick_model_interactive(
            current_model,
            label="current",
            credential_vault=credential_vault,
        )
        if new_model == current_model:
            click.echo(f"Agent '{name}' already uses {current_model}.")
            return None
        _update_agent_field(name, "model", new_model)
        click.echo(f"Agent '{name}' model: {current_model} -> {new_model}")
        return "model"

    elif choice == 2:  # description
        new_desc = click.prompt("  Description", default=current_desc)
        if new_desc != current_desc:
            _update_agent_field(name, "role", new_desc)
            click.echo(f"Agent '{name}' description updated.")
            return "role"
        click.echo("No change.")
        return None

    elif choice == 3:  # budget
        default_budget = str(current_budget) if current_budget is not None else ""
        new_budget_str = click.prompt("  Daily budget (USD)", default=default_budget)
        if not new_budget_str.strip():
            click.echo("No change.")
            return None
        try:
            new_budget = float(new_budget_str)
            if new_budget < 0:
                click.echo("Budget cannot be negative.")
                return None
            _update_agent_field(name, "budget", {"daily_usd": new_budget})
            click.echo(f"Agent '{name}' budget: ${new_budget:.2f}/day")
            return "budget"
        except ValueError:
            click.echo("Invalid number. Budget not changed.")
            return None

    elif choice == 4:  # thinking
        click.echo()
        for i, level in enumerate(_THINKING_LEVELS, 1):
            marker = " (current)" if level == current_thinking else ""
            click.echo(f"  {i}. {_THINKING_LABELS[level]}{marker}")
        idx = click.prompt(
            "\n  Thinking level",
            type=click.IntRange(1, len(_THINKING_LEVELS)),
            default=_THINKING_LEVELS.index(current_thinking) + 1,
        )
        new_thinking = _THINKING_LEVELS[idx - 1]
        if new_thinking == current_thinking:
            click.echo("No change.")
            return None
        _update_agent_field(name, "thinking", new_thinking)
        click.echo(f"Agent '{name}' thinking: {current_thinking} -> {new_thinking}")
        return "thinking"

    return None


def _setup_agent_wizard(model: str) -> str:
    """Interactive agent creation for setup. Returns agent name."""
    agent_name = click.prompt("  Agent name", default="assistant")
    description = click.prompt(
        "  What should this agent do?",
        default=_default_description(agent_name),
    )
    _create_agent(agent_name, description, model)
    click.echo(f"  Created agent '{agent_name}'.")
    return agent_name


# ── Operator agent ─────────────────────────────────────────

_OPERATOR_AGENT_ID = "operator"

_OPERATOR_ALLOWED_TOOLS: list[str] = [
    # Monitoring + heartbeat
    "get_system_status",
    "notify_user",
    "inspect_agents",
    "inspect_teams",
    "list_agent_queue",
    "get_team_outputs",
    "summarize_team_progress",
    # Composes and persists a work summary card for the Work tab.
    # Backed by ``WorkSummariesStore``; the daily cron invokes this
    # tool directly (no LLM cost per fire). User rates via dashboard.
    "compose_work_summary",
    # Canonical inverse of edit_agent — operator reads the current
    # config surface before mutating. Pair with list_peer_artifacts /
    # read_peer_artifact for deeper inspection of peer-written files.
    "read_agent_config",
    "list_peer_artifacts",
    "read_peer_artifact",
    # Peer FILE reads — full /data volume, not just artifacts/. Lets the
    # operator locate + relay a worker's deliverable (CSV, data.md) the
    # user asked for, instead of reporting it unreachable.
    "list_peer_files",
    "read_peer_file",
    # Observation log of agent→user notifications — PULL-only, NOT a
    # message channel (agents can't address the operator). Lets the
    # operator answer "what's blocking?" from what workers already told
    # the human. Sanitized + display_only at the tool boundary.
    "read_user_notifications",
    # Coordination + chat
    "list_templates",
    "apply_template",
    "hand_off",
    "check_inbox",
    # Inline teammate Q&A (Phase 2 unit 3). ``ask_teammate`` lets the
    # operator answer a blocked worker's question inline (the worker
    # resumes the SAME task — no re-hand_off duplicate) and pull quick
    # facts from workers; ``answer_ask`` is the reply side.
    "ask_teammate",
    "answer_ask",
    # Skill-pack discovery (SKILL.md procedures) — read-only, so the
    # operator can see which skills its workers can draw on. install_skill /
    # remove_skill are operator-gated + user-origin-gated mutations.
    "skills_list",
    "skill_view",
    "install_skill",
    "remove_skill",
    "list_skill_assignments",
    "assign_skill",
    # Workflow awareness — operator-only chain inspection + single-task
    # blocking primitive (see _HEARTBEAT_TOOLS in src/agent/loop.py for
    # the heartbeat surface; both tools self-reject for non-operators).
    "workflow_snapshot",
    "await_task_event",
    # Per-task execution diagnostics (B5) — answers "why did this come
    # out shallow/wrong": thinking level, token/LLM-call counts, trace
    # errors, status timeline. Pairs with rate_delivery on rework.
    "inspect_task_run",
    # Configuration edits — edit_agent applies every field immediately
    # and emits an undo receipt (5min for soft fields, 30min for hard).
    # undo_change lets the operator self-revert within the TTL.
    "edit_agent",
    "undo_change",
    # Credential-aware model discovery — operator calls this BEFORE
    # edit_agent / create_agent so it doesn't have to memorize which
    # models are usable with the active credential setup (OAuth-allowed
    # subsets vs full API-key catalog). See Fix 2 in the seam follow-up.
    "list_available_models",
    # Creation
    "create_agent",
    "create_team",
    # Team membership + context
    "add_agents_to_team",
    "remove_agents_from_team",
    "update_team_context",
    # P2 — section-scoped TEAM.md updates (fleet-wide knowledge
    # propagation; canonical use: '## User Preferences').
    "update_team_brief",
    # PR 5 — north-star setter is no-confirmation meta-config.
    "set_team_goal",
    # User-visible business goals on the Work tab (separate from
    # per-team north stars). Backed by GOALS.json + rendered GOALS.md
    # in the operator's workspace; dashboard reads via
    # /api/workplace/goals. Operator-only — _is_operator() is enforced
    # at the tool boundary as well.
    "manage_goals",
    # Per-agent standing goals — writes the Team-store goals record
    # (PUT /mesh/agents/{id}/goals) every worker injects into its
    # prompts. Deliberate planning act: intentionally NOT in
    # _OPERATOR_HEARTBEAT_TOOLS.
    "set_agent_goals",
    # Per-task outcome rating (PR 2 of Work tab rewrite). Replaces
    # the per-task 👍/➖/👎 buttons deleted from the Work tab —
    # operator scores tasks programmatically from the heartbeat
    # loop, preserving the agent-feedback machine loop (memory
    # writes + auto-rework spawn).
    "rate_delivery",
    # Lifecycle (consolidated archive/delete)
    "manage_team",
    "manage_agent",
    "manage_task",
    # Self-cleanup — operator can clear stale pending actions and prune
    # the audit log without waiting for TTL. ``list_pending`` lets the
    # operator find the nonce before calling cancel_pending_action.
    "list_pending",
    "cancel_pending_action",
    "archive_audit_before",
    # Credential + browser handoff. ``vault_list`` returns names only
    # (never values) so the operator can check what credentials already
    # exist before calling request_credential.
    "vault_list",
    "request_credential",
    "request_browser_login",
    # Operator self-notes + workspace management. Workspace file caps
    # already enforce safety on writes; write_file is intentionally NOT
    # granted (operator orchestrates, doesn't author arbitrary files).
    "memory_save",
    "memory_search",
    "memory_think",
    "update_workspace",
    "read_file",
    # Internet access (gated by ``can_use_internet`` permission — the
    # agent's runtime filters these out of the effective allowlist when
    # the Operator Settings → Internet access toggle is OFF).
    "http_request",
    "web_search",
    # Browser access (gated by ``can_use_browser`` permission — the
    # agent's runtime filters these out of the effective allowlist when
    # the Operator Settings → Browser access toggle is OFF). Mirrors the
    # full worker browser surface from ``builtins/browser_tool.py`` so
    # the operator can navigate the web directly. ``request_browser_login``
    # (above) remains the delegation path for landing a worker's cookies.
    "browser_navigate",
    "browser_warmup",
    "browser_get_elements",
    "browser_wait_for",
    "browser_screenshot",
    "browser_screenshot_marks",
    "browser_click",
    "browser_click_xy",
    "browser_type",
    "browser_hover",
    "browser_scroll",
    "browser_reset",
    "browser_press_key",
    "browser_go_back",
    "browser_go_forward",
    "browser_switch_tab",
    "browser_find_text",
    "browser_fill_form",
    "browser_open_tab",
    "browser_inspect_requests",
    "browser_detect_captcha",
    "browser_upload_file",
    "browser_solve_captcha",
    "browser_download",
    "browser_set_dialog_policy",
    "browser_drag",
    "browser_grant_permissions",
    "browser_set_geolocation",
    "browser_right_click",
    "browser_read_clipboard",
    "browser_write_clipboard",
    "browser_wait_for_network_idle",
    "browser_select_option",
]

# Grouped Tool Search bridge — pulls a deferred capability group's full schemas
# into context for the next turn. ``execute_code`` runs a Python snippet in the
# agent container (code-as-action speed lever). Both are part of the operator's
# standard surface.
_OPERATOR_ALLOWED_TOOLS.append("load_tools")
_OPERATOR_ALLOWED_TOOLS.append("execute_code")

# Reference list documenting the tools operator uses on heartbeat. The
# loop's actual gate is ``_HEARTBEAT_TOOLS`` in ``src/agent/loop.py`` —
# this list is kept in sync as documentation but is not itself consulted
# by the runtime. New tools should be added to BOTH places (and the
# operator HEARTBEAT.md prompt) to take effect.
_OPERATOR_HEARTBEAT_TOOLS: list[str] = [
    # v1 baseline (read-only)
    "list_agents",
    "get_agent_profile",
    "get_system_status",
    "notify_user",
    # v2 workflow-awareness — back-edge events + chain inspection +
    # single-task blocking so the heartbeat can drive multi-stage
    # chains without dropping out to a full /chat turn.
    "check_inbox",
    "workflow_snapshot",
    "await_task_event",
    # v3 Work-tab rewrite — heartbeat grades up to 10 oldest unrated
    # done tasks per cycle and stewards goal staleness.
    "rate_delivery",
    "manage_goals",
    # v4 — ``inspect_agents`` was already prompted in step 5 of the
    # heartbeat procedure but missing from this allowlist; added so
    # the runtime gate stops denying the prompted call.
    "inspect_agents",
    # B5 — read-only per-task run diagnostics (tokens, LLM calls,
    # thinking, trace errors) so rating decisions can be informed.
    "inspect_task_run",
    # A5 — the heartbeat procedure says "re-read GOALS.json fresh each
    # cycle"; without read_file the prompted call was denied.
    "read_file",
    # v7 (Phase 2 unit 3) — step 1 answers a task_blocked worker's
    # clarifying question inline so it resumes the SAME task (replaces
    # the blocked→re-hand_off dance).
    "ask_teammate",
]

_OPERATOR_SOUL = """\
You are sharp, proactive, and action-oriented. Users should feel like they \
have a competent team lead from the first message.

Do things — don't list things you could do. Ask only what you need, then act. \
One question is better than four. One confirmation covers an entire setup flow.

Speak in short, direct sentences. No filler. No "I can help you with..." — \
just help. When presenting a plan, use bullet points with agent names bolded. \
When confirming completion, state what's ready, not what you did.

You build the workforce and step back. Users work with agents directly.
"""

_OPERATOR_HEARTBEAT = """\
<!-- heartbeat_v2_workflow_aware -->
<!-- heartbeat_v3_rate_delivery -->
<!-- heartbeat_v4_goal_seeding -->
<!-- heartbeat_v5_fleet_health -->
<!-- heartbeat_v6_agent_retro -->
<!-- heartbeat_v7_ask_teammate -->
You are running an autonomous fleet health check. You have access ONLY to monitoring tools.

Step budget: stay at or under 8 tool calls per cycle (HEARTBEAT_MAX_ITERATIONS=12
in the loop; 8 leaves headroom for the final assistant turn).

1. Call check_inbox() FIRST so any task_failed / task_blocked back-edge events
   from active workflows are surfaced before you do fleet-wide work. Skim the
   ``events[]`` array:
   - For each task_failed / task_blocked event note the task_id and recipient.
   - For a task_blocked event whose blocker_note is a clarifying question you
     can answer from context, answer it inline with
     ask_teammate(to=<recipient>, question='<your answer / clarification>') —
     the worker resumes the SAME task. Do NOT re-hand_off a duplicate. If you
     cannot answer it, surface it via notify_user in step 5.
   - For each event whose task is part of an orchestration you started, drop a
     ``workflow_snapshot(root_task_id)`` call (step 3) to see chain state.

2. Read the ## Fleet Health block already in your context (refreshed every
   ~5 min) — it carries the non-zero failure / rejected / stale / chain-break
   counts, your own stale-inbox depth, and the top cost outliers. Call
   get_system_status() ONLY when you need to drill past the digest (full
   per-agent breakdowns, plan limits, cost trend vs yesterday).
   - If the digest (or get_system_status's inbox_stale_count) shows stale
     inbox tasks, triage the oldest operator-inbox handoffs first via
     check_inbox — that count is YOUR own untriaged backlog.
   - Before rating a suspicious delivery or re-dispatching failed work, call
     inspect_task_run(task_id) to see HOW it ran (tokens, LLM calls, thinking,
     trace errors).

3. Workflow awareness — for any active orchestration you kicked off (a task you
   created as a root that fanned out via hand_off), call
   ``workflow_snapshot(root_task_id)`` to see all stages. Read the response:
   - ``summary.failed`` > 0 OR ``summary.blocked`` > 0: surface to the user
     in step 5 with the offending stage's assignee, title, and blocker_note
     (the snapshot includes it inline — no follow-up get_task needed).
   - Any stage with ``status == "working"`` AND ``age_in_state_seconds > 300``:
     mention it in your notify_user message — the agent is genuinely slow.
     DO NOT mark the work failed; the lane watchdog handles the actual cap.
   - Skip ``workflow_snapshot`` entirely if you have no active orchestrations.
     One call covers an entire chain — do NOT loop.
   - Cap at 3 snapshot calls per heartbeat: if you have more than 3 active
     workflows, snapshot the 3 most concerning (most-recent failed events
     wins, then most-recent task_started) and defer the rest to next cycle.

4. Call inspect_agents() for the roster summary if you haven't already this
   cycle. Then drill into at most THREE most-concerning agents. PREFER one
   targeted call per drill — don't fan out across the whole fleet:
   - Candidates are agents that appear in agents_needing_attention OR have
     per_agent_cost_vs_yesterday_ratio is not None AND > 2.0 OR
     outcome_rejected_24h_count[agent] > 5 OR
     outcome_rework_24h_count[agent] >= 3 (a rework streak means the agent's
     instructions or briefs need tightening — offer the user a tune-up) OR
     execution_failures_24h_count[agent] > 3 OR
     chain_breaks_24h_count[agent] > 0.
   - If more than 3 agents trigger any of the above thresholds, focus on
     the top-3 worst (highest cost outlier, highest rejected count, longest
     stale duration) and defer the rest to the next cycle.
   - For each selected agent, call inspect_agents(agent_id, depth="profile").
   - If stale_tasks_24h_count has any non-zero entry, call
     inspect_agents(stale_threshold_hours=24) ONCE to pull the offending
     task IDs (the result annotates every roster entry — don't loop).

5. Call notify_user() if ANY of the following triggered this cycle:
   - A workflow stage reached task_failed or task_blocked (include task_id +
     recipient + the kickoff root_task_id).
   - An agent is CRITICAL (failed state, budget exceeded, >5 rejected outcomes).
   - A workflow stage has been ``working`` for > 5 minutes (inform, don't kill).
   Do not re-notify on issues you already alerted on last cycle unless severity
   has increased.

If any tool call fails, continue with the remaining steps.
Do not hallucinate data you could not retrieve.

## Per-task rating cadence

Each heartbeat, rate up to 10 oldest unrated done tasks via
`rate_delivery`. Prefer keeping one team's tasks contiguous if it
helps focus. Outcomes:
- accepted: matches the ask cleanly → rating signal only
- acknowledged: unsure or low-confidence (the safe default when you
  can't confidently defend the judgment)
- rework: fixable miss → spawns follow-up task; your feedback is
  pushed into the agent's learnings; feedback required
- rejected: needs starting over → feedback pushed to the agent's
  learnings; feedback required, no spawn

DEFAULT TO `acknowledged` WHEN UNCERTAIN. Never guess.

## Agent retro (at most ONE agent per heartbeat)

If the Fleet Health digest or rating history shows an agent with
rework >= 3 or rejected > 0 in the window: drill in with
`inspect_agents` and read the repeated feedback. If the SAME
correction keeps appearing, do not edit anything from the heartbeat —
instead `notify_user` with a one-line proposed instruction delta,
e.g. 'research-1 keeps omitting source links (3x rework). Proposed
INSTRUCTIONS.md addition: "Always end reports with a Sources
section." Reply to approve, or tell me to apply it.' Apply via
`edit_agent(field="instructions")` only in a normal (non-heartbeat)
turn, as ONE appended dated bullet — never a rewrite. Skip this step
entirely when no agent crosses the thresholds.

## Workspace-as-source-of-truth

Re-read GOALS.json, AGENTS.md fresh at the start of each cycle. Treat
working memory as ephemeral — anything load-bearing must come from
workspace files.

## Surface ambiguity, don't guess

If you cannot identify which team a task belongs to, or two goals
conflict, call `notify_user` and stop. Do not guess your way through.

## Goal seeding (cold start)

Procedure when GOALS.json is empty AND ≥1 team has active agents:

1. Call `manage_goals(action="list")`. The response carries both
   `goals` (the current list) and `seed_ask` (your throttle record:
   `{last_ts, team_names}` or `null` if you've never asked).
2. If `seed_ask` is `null` OR `seed_ask.last_ts` is more than 7 days
   old, you may re-ask. Otherwise skip — you've already pinged the
   user this cycle's window.
3. To ask: call `notify_user` with a short message naming the
   visible teams and inviting the user to state business outcomes.
   Immediately follow with
   `manage_goals(action="record_seed_ask", team_names=[…])` so the
   throttle is stamped. Don't guess goals from a heartbeat —
   surface the question and stop.

The structured `seed_ask` block on GOALS.json replaces the
freeform-notes throttle Codex flagged during PR 972 review; the
LLM no longer has to scan free-form prose to decide whether it
already asked.

During CHAT (not heartbeat), draft goals with `manage_goals` action
"set" / "add" whenever the user states outcomes freely or after
team creation reveals measurable goals — that's the primary seeding
path; the heartbeat ask is the safety net.

## Goal stewardship

On each cycle scan goals for staleness (no related task activity in
14 days). Surface stale candidates in your next summary's "What I'm
watching" section. DO NOT auto-retire goals — only the user decides
to drop a goal.

## Tool semantics (don't confuse these)

- `manage_goals` — workplace-wide user-stated business outcomes.
  Source of truth is GOALS.json. Use when user states/changes
  direction in chat. (NEW — added in PR 1.)
- `set_team_goal` — each team's mission statement. Rarely changes;
  usually set on team creation.
- `rate_delivery` — your per-task outcome judgment after a task
  completes. Not user-facing. (NEW — added in PR 2.)
"""


def _ensure_operator_agent(config_path: Path | None = None, default_model: str = "") -> None:
    """Create the operator agent if it doesn't exist. Handles concierge->operator migration.

    Thin locked wrapper: the whole function body (several load->mutate->save
    sites across agents.yaml AND permissions.json, with multiple early
    returns) runs inside ONE ``_config_lock()`` acquisition via
    ``_ensure_operator_agent_locked`` — delegating rather than reindenting
    every branch keeps this diff reviewable and avoids a mass-reindent of a
    ~250-line function.
    """
    with _config_lock():
        _ensure_operator_agent_locked(config_path, default_model)


def _ensure_operator_agent_locked(config_path: Path | None, default_model: str) -> None:
    """Body of ``_ensure_operator_agent`` — ONLY called with the lock held."""
    agents_cfg: dict = _load_agents_yaml()
    if "agents" not in agents_cfg:
        agents_cfg["agents"] = {}

    has_concierge = "concierge" in agents_cfg["agents"]
    has_operator = _OPERATOR_AGENT_ID in agents_cfg["agents"]

    # Migration: rename concierge -> operator
    if has_concierge and not has_operator:
        agents_cfg["agents"][_OPERATOR_AGENT_ID] = agents_cfg["agents"].pop("concierge")
        _save_agents_yaml(agents_cfg)
        # Also rename in permissions
        perms = _load_permissions()
        if "concierge" in perms.get("permissions", {}):
            perms["permissions"][_OPERATOR_AGENT_ID] = perms["permissions"].pop("concierge")
            _save_permissions(perms)
        return
    elif has_concierge and has_operator:
        # Both exist — keep operator, remove concierge
        del agents_cfg["agents"]["concierge"]
        _save_agents_yaml(agents_cfg)
        return

    if has_operator:
        # Refresh the operator's heartbeat field whenever the canonical
        # template gains a new sentinel marker. Mirrors the workspace-
        # side refresh in ``WorkspaceManager._ensure_scaffold`` — the
        # latest sentinel in ``HEARTBEAT_SENTINELS`` is the current
        # expectation; an operator carrying only earlier markers (e.g.
        # ``heartbeat_v2_workflow_aware`` after we add v3) needs a
        # refresh to receive the new instructions.
        op_entry = agents_cfg["agents"].get(_OPERATOR_AGENT_ID, {}) or {}
        existing_heartbeat = op_entry.get("heartbeat") or ""
        # ``initial_heartbeat`` is the key the runtime actually consumes:
        # cli/runtime.py builds the INITIAL_HEARTBEAT container env from it,
        # and the workspace-side sentinel refresh fires only when that env
        # carries the latest marker. Refreshing ``heartbeat`` alone never
        # reaches deployed containers — the v2→v5 bumps silently missed
        # every pre-existing operator (found live on cake 2026-06-11; same
        # class of gap as the playbook initial_instructions roll-forward
        # fixed in #1110). Both keys must be rolled forward together.
        existing_initial = op_entry.get("initial_heartbeat") or ""
        from src.shared.types import HEARTBEAT_SENTINELS

        latest_sentinel = HEARTBEAT_SENTINELS[-1] if HEARTBEAT_SENTINELS else None
        new_has_latest = latest_sentinel is not None and f"<!-- {latest_sentinel} -->" in _OPERATOR_HEARTBEAT
        # Current only when BOTH copies carry the latest marker — a stale
        # ``initial_heartbeat`` next to a current ``heartbeat`` still needs
        # the refresh (that exact split is what the consumed-key gap left
        # behind on deployed boxes).
        old_has_latest = (
            latest_sentinel is not None
            and f"<!-- {latest_sentinel} -->" in existing_heartbeat
            and f"<!-- {latest_sentinel} -->" in existing_initial
        )
        # Roll forward ONLY when the existing heartbeat carries at
        # least one prior sentinel — proves it's a system-managed
        # template we have rights to refresh. A user-customised
        # heartbeat carries NO marker because the user replaced the
        # template; without this guard every sentinel bump would
        # silently overwrite their customisation.
        #
        # Trade-off (Codex pre-merge review): pre-sentinel system
        # installs also lack any marker, so they look identical to
        # user customisation and stay on their old template. The
        # operator can manually clear `heartbeat:` in agents.yaml to
        # opt in to the fresh template on next startup. We log a
        # warn so the situation is visible.
        old_has_any_sentinel = any(f"<!-- {s} -->" in existing_heartbeat for s in HEARTBEAT_SENTINELS)
        # An EMPTY heartbeat is the operator's documented opt-in path
        # for re-bootstrap (the WARN below for no-sentinel files
        # instructs operators to clear the field; the refresh has to
        # actually fire when they do). Treated as "fresh install" and
        # rewritten from the canonical template.
        existing_is_empty = not existing_heartbeat.strip()
        if new_has_latest and ((not old_has_latest and old_has_any_sentinel) or existing_is_empty):
            op_entry["heartbeat"] = _OPERATOR_HEARTBEAT
            op_entry["initial_heartbeat"] = _OPERATOR_HEARTBEAT
            agents_cfg["agents"][_OPERATOR_AGENT_ID] = op_entry
            _save_agents_yaml(agents_cfg)
            logger.info(
                "Refreshed operator heartbeat to versioned template%s",
                " (was empty — bootstrapped)" if existing_is_empty else "",
            )
        elif new_has_latest and not old_has_latest and not old_has_any_sentinel and not existing_is_empty:
            logger.warning(
                "operator heartbeat carries no known sentinel — "
                "treating as user-customised and skipping refresh. "
                "To opt in to the fresh template: clear the `heartbeat:` "
                "field in agents.yaml (set to empty/null) AND delete "
                "the operator's HEARTBEAT.md — startup will then write "
                "both layers fresh from the canonical template."
            )

        # Refresh the operator's ``initial_instructions`` payload the same
        # way. The workspace migrator appends a new playbook addendum to
        # the live INSTRUCTIONS.md ONLY when the container's
        # INITIAL_INSTRUCTIONS payload carries the new sentinel — and that
        # payload comes verbatim from agents.yaml, which was written once
        # at creation and never updated. Without this roll-forward,
        # existing operators never receive any playbook addendum (live
        # finding 2026-06-11: a production operator predating every
        # sentinel had silently missed two playbook generations). Note
        # this only refreshes the PAYLOAD — the workspace-side migrator
        # still appends to the live INSTRUCTIONS.md, so the operator's
        # self-evolved content is never rewritten.
        from src.shared.types import PLAYBOOK_SENTINELS

        existing_instructions = op_entry.get("initial_instructions") or ""
        pb_latest = PLAYBOOK_SENTINELS[-1] if PLAYBOOK_SENTINELS else None
        pb_new_has_latest = pb_latest is not None and f"<!-- {pb_latest} -->" in _OPERATOR_CORE
        pb_old_has_latest = pb_latest is not None and f"<!-- {pb_latest} -->" in existing_instructions
        pb_old_has_any = any(f"<!-- {s} -->" in existing_instructions for s in PLAYBOOK_SENTINELS)
        pb_existing_empty = not existing_instructions.strip()
        if pb_new_has_latest and ((not pb_old_has_latest and pb_old_has_any) or pb_existing_empty):
            op_entry["initial_instructions"] = _OPERATOR_CORE
            agents_cfg["agents"][_OPERATOR_AGENT_ID] = op_entry
            _save_agents_yaml(agents_cfg)
            logger.info(
                "Refreshed operator instructions payload to versioned playbook%s",
                " (was empty — bootstrapped)" if pb_existing_empty else "",
            )
        elif pb_new_has_latest and not pb_old_has_latest and not pb_old_has_any and not pb_existing_empty:
            logger.warning(
                "operator instructions payload carries no known sentinel — "
                "treating as user-customised and skipping refresh. To opt "
                "in: clear the `initial_instructions:` field in agents.yaml "
                "— startup rewrites it from the canonical playbook. The "
                "live INSTRUCTIONS.md is never rewritten; new guidance is "
                "appended to it."
            )

        # Ensure permissions are correct even for existing operator
        # (handles upgrades where operator was created before permissions were set)
        perms = _load_permissions()
        op_perms = perms.get("permissions", {}).get(_OPERATOR_AGENT_ID, {})
        needs_update = False
        if not op_perms.get("allowed_apis"):
            op_perms["allowed_apis"] = ["llm"]
            needs_update = True
        if "can_spawn" not in op_perms or not op_perms["can_spawn"]:
            op_perms["can_spawn"] = True
            needs_update = True
        # Operator must always be able to message all agents (it's created first,
        # before other agents exist, so the default logic sets can_message=[])
        if op_perms.get("can_message") != ["*"]:
            op_perms["can_message"] = ["*"]
            needs_update = True
        if op_perms.get("can_publish") != ["*"]:
            op_perms["can_publish"] = ["*"]
            needs_update = True
        if op_perms.get("can_subscribe") != ["*"]:
            op_perms["can_subscribe"] = ["*"]
            needs_update = True
        if not op_perms.get("blackboard_read"):
            op_perms["blackboard_read"] = ["*"]
            needs_update = True
        if not op_perms.get("blackboard_write"):
            op_perms["blackboard_write"] = ["*"]
            needs_update = True
        # Control-plane permissions (Task 3). Idempotent: only set when
        # missing/falsy, so running this block twice is a no-op.
        if not op_perms.get("can_manage_fleet", False):
            op_perms["can_manage_fleet"] = True
            needs_update = True
        if not op_perms.get("can_manage_teams", False):
            op_perms["can_manage_teams"] = True
            needs_update = True
        if not op_perms.get("can_edit_agent_config", False):
            op_perms["can_edit_agent_config"] = True
            needs_update = True
        if not op_perms.get("can_view_fleet_metrics", False):
            op_perms["can_view_fleet_metrics"] = True
            needs_update = True
        # ``can_route_tasks`` retired in favour of unified ``can_message``
        # gate on /mesh/tasks (worker→worker handoffs need exactly one
        # permission, not two). The field stays on AgentPermissions for
        # back-compat with existing permissions.json files but is no
        # longer read; we stop grant-on-backfill so new operators don't
        # accumulate dead configuration.
        if not op_perms.get("can_request_user_credentials", False):
            op_perms["can_request_user_credentials"] = True
            needs_update = True
        # Internet access: backfill True for existing operators that
        # predate the field. Idempotent — once set (True OR False), the
        # ``not in`` guard skips so the user's explicit OFF choice is
        # preserved across restarts.
        if "can_use_internet" not in op_perms:
            op_perms["can_use_internet"] = True
            needs_update = True
        # Browser access: backfill True for existing operators that
        # predate the field. Idempotent — once set (True OR False), the
        # ``not in`` guard skips so the user's explicit OFF choice in
        # Operator Settings is preserved across restarts.
        if "can_use_browser" not in op_perms:
            op_perms["can_use_browser"] = True
            needs_update = True
        if needs_update:
            perms.setdefault("permissions", {})[_OPERATOR_AGENT_ID] = op_perms
            _save_permissions(perms)
        return

    # Create new operator
    if not default_model:
        cfg = _load_config(config_path)
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

    _add_agent_to_config(
        _OPERATOR_AGENT_ID,
        role="Operator — builds and manages your agent workforce",
        model=default_model,
        initial_instructions=_OPERATOR_CORE,
        initial_soul=_OPERATOR_SOUL,
        initial_heartbeat=_OPERATOR_HEARTBEAT,
    )
    _add_agent_permissions(
        _OPERATOR_AGENT_ID,
        permissions={
            "can_spawn": True,
            # Operator gets ``can_use_browser=True`` by default so it can
            # browse the web directly to help users navigate and answer
            # questions without first spinning up a worker. The dashboard
            # surfaces a toggle in Operator Settings → "Browser access"
            # that flips this off when the user wants the operator
            # sandboxed (mirrors the internet-access toggle). The
            # delegation path (``request_browser_login`` with
            # ``agent_id=target_worker``) still works for landing a
            # worker's session cookies — see
            # ``src/shared/operator_playbooks.py``.
            "can_use_browser": True,
            # Operator gets internet (HTTPS + web search) by default so
            # it can fetch reference material and answer factual
            # questions without delegating to a worker. The dashboard
            # surfaces a toggle in Operator Settings → "Internet
            # access" that flips this off when the user wants the
            # operator sandboxed.
            "can_use_internet": True,
            "blackboard_read": ["*"],
            "blackboard_write": ["*"],
            "can_publish": ["*"],
            "can_subscribe": ["*"],
            # Control-plane permissions (Task 3) — operator gets all six.
            "can_manage_fleet": True,
            "can_manage_teams": True,
            "can_edit_agent_config": True,
            "can_view_fleet_metrics": True,
            "can_request_user_credentials": True,
        },
    )
    # Force can_message=["*"] — _add_agent_permissions sets it to []
    # because operator is always the first agent created (no peers yet).
    perms = _load_permissions()
    perms["permissions"][_OPERATOR_AGENT_ID]["can_message"] = ["*"]
    _save_permissions(perms)
    tools_dir = PROJECT_ROOT / "agent_tools" / _OPERATOR_AGENT_ID
    tools_dir.mkdir(parents=True, exist_ok=True)
