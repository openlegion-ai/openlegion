"""Configuration loading, Docker helpers, and agent management."""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import sys
from pathlib import Path

import click
import yaml

from src.shared.utils import truncate

logger = logging.getLogger("cli")

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
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh.yaml"
AGENTS_FILE = PROJECT_ROOT / "config" / "agents.yaml"
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"
PROJECTS_DIR = PROJECT_ROOT / "config" / "projects"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DOCKER_IMAGE = "openlegion-agent:latest"
BROWSER_IMAGE = "openlegion-browser:latest"

MARKETPLACE_DIR = PROJECT_ROOT / "skills" / "_marketplace"

# ── Provider data ───────────────────────────────────────────

_PROVIDERS = [
    {"name": "anthropic", "label": "Anthropic (recommended)"},
    {"name": "moonshot", "label": "Moonshot / Kimi (recommended)"},
    {"name": "deepseek", "label": "DeepSeek"},
    {"name": "openai", "label": "OpenAI"},
    {"name": "gemini", "label": "Google Gemini"},
    {"name": "xai", "label": "xAI (Grok)"},
    {"name": "groq", "label": "Groq"},
    {"name": "minimax", "label": "MiniMax"},
    {"name": "zai", "label": "Z.AI (GLM)"},
]


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

# ── Config loading ──────────────────────────────────────────


def _load_config(mesh_path: Path | None = None) -> dict:
    """Load mesh config, agent definitions, and project metadata."""
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

    # Load projects and build reverse map (agent → project)
    projects = _load_projects()
    cfg["projects"] = projects
    agent_projects: dict[str, str] = {}
    for pname, pdata in projects.items():
        for member in pdata.get("members", []):
            agent_projects[member] = pname
    cfg["_agent_projects"] = agent_projects
    return cfg


def _load_permissions() -> dict:
    if not PERMISSIONS_FILE.exists():
        return {"permissions": {}}
    with open(PERMISSIONS_FILE) as f:
        return json.load(f)


def _save_permissions(perms: dict) -> None:
    PERMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PERMISSIONS_FILE, "w") as f:
        json.dump(perms, f, indent=2)
        f.write("\n")


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
            "docker", "build",
            "-t", BROWSER_IMAGE,
            "-f", "Dockerfile.browser", ".",
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
            "docker", "build",
            "-t", DOCKER_IMAGE,
            "-f", "Dockerfile.agent", ".",
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
    name: str, role: str, model: str,
    initial_instructions: str = "",
) -> None:
    """Add an agent entry to agents.yaml."""
    agents_cfg: dict = {"agents": {}}
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {"agents": {}}
    if "agents" not in agents_cfg:
        agents_cfg["agents"] = {}

    entry: dict = {
        "role": role,
        "model": model,
        "skills_dir": f"./skills/{name}",
    }
    if initial_instructions:
        entry["initial_instructions"] = initial_instructions
    agents_cfg["agents"][name] = entry
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_FILE, "w") as f:
        yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)


def _add_agent_permissions(name: str) -> None:
    """Add default permissions for a new agent.

    If collaboration mode is enabled in mesh.yaml, agents can message
    all other agents. Otherwise they can only message the orchestrator.
    """
    cfg = _load_config()
    collab = cfg.get("collaboration", True)

    perms = _load_permissions()
    if collab:
        other_agents = [a for a in cfg.get("agents", {}) if a != name]
        can_message = ["orchestrator", "*"] if other_agents else ["orchestrator"]
    else:
        can_message = ["orchestrator"]

    perms["permissions"][name] = {
        "can_message": can_message,
        "can_publish": ["*"] if collab else [f"{name}_complete"],
        "can_subscribe": ["*"] if collab else [],
        "blackboard_read": [],
        "blackboard_write": [],
        "allowed_apis": ["llm"],
        "allowed_credentials": ["*"],
    }
    _save_permissions(perms)


def _ensure_all_agent_permissions() -> None:
    """Backfill permissions for agents missing from permissions.json."""
    cfg = _load_config()
    perms = _load_permissions()
    existing = set(perms.get("permissions", {}).keys())
    for name in cfg.get("agents", {}):
        if name not in existing:
            _add_agent_permissions(name)


def _set_collaborative_permissions() -> None:
    """Update all agent permissions to allow inter-agent messaging and pub/sub."""
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

    Rejects path traversal, slashes, and non-alphanumeric chars
    (aside from hyphens and underscores).
    """
    import re

    if not name or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError(
            f"Invalid agent name '{name}': must be 1–64 alphanumeric chars, "
            "hyphens, or underscores (must start with a letter or digit)."
        )
    return name


def _create_agent(
    name: str, description: str, model: str,
) -> None:
    """Create an agent: config, permissions, skills directory."""
    name = _validate_agent_name(name)
    _add_agent_to_config(name, description, model)
    _add_agent_permissions(name)
    skills_dir = PROJECT_ROOT / "skills" / name
    skills_dir.mkdir(parents=True, exist_ok=True)


def _suppress_host_logs() -> None:
    """Set host-side loggers to WARNING for clean CLI output."""
    for name in [
        "host", "host.containers", "host.credentials", "host.orchestrator",
        "host.mesh", "host.costs", "host.permissions", "host.cron", "host.webhooks",
        "host.health", "host.lanes", "host.runtime", "host.watchers",
        "channels", "channels.base", "channels.telegram", "channels.discord",
        "channels.slack", "channels.whatsapp",
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
        try:
            data = json.loads(pairing_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
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


# ── Project management ──────────────────────────────────────


def _validate_project_name(name: str) -> str:
    """Validate and return a safe project name (same rules as agent names)."""
    try:
        return _validate_agent_name(name)
    except ValueError:
        raise ValueError(
            f"Invalid project name '{name}': must be 1–64 alphanumeric chars, "
            "hyphens, or underscores (must start with a letter or digit)."
        )


def _load_projects() -> dict[str, dict]:
    """Scan config/projects/*/metadata.yaml and return {name: metadata}."""
    from src.shared.types import ProjectMetadata

    projects: dict[str, dict] = {}
    if not PROJECTS_DIR.exists():
        return projects
    for meta_file in sorted(PROJECTS_DIR.glob("*/metadata.yaml")):
        try:
            dir_name = meta_file.parent.name
            with open(meta_file) as f:
                data = yaml.safe_load(f) or {}
            pm = ProjectMetadata(**data)
            projects[dir_name] = pm.model_dump()
        except Exception as e:
            logger.warning("Failed to load project %s: %s", meta_file, e)
    return projects


def _get_agent_project(agent_name: str) -> str | None:
    """Return the project name an agent belongs to, or None if standalone."""
    projects = _load_projects()
    for pname, pdata in projects.items():
        if agent_name in pdata.get("members", []):
            return pname
    return None


def _create_project(
    name: str, description: str = "", members: list[str] | None = None,
) -> None:
    """Create a new project: directory, metadata.yaml, scaffold project.md."""
    name = _validate_project_name(name)
    project_dir = PROJECTS_DIR / name
    if project_dir.exists():
        raise ValueError(f"Project '{name}' already exists")

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "workflows").mkdir(exist_ok=True)

    from datetime import UTC, datetime

    from src.shared.types import ProjectMetadata

    pm = ProjectMetadata(
        name=name,
        description=description,
        created_at=datetime.now(UTC).isoformat(),
        members=[],  # members added via _add_agent_to_project below
    )
    with open(project_dir / "metadata.yaml", "w") as f:
        yaml.dump(pm.model_dump(), f, default_flow_style=False, sort_keys=False)

    # Scaffold project.md
    (project_dir / "project.md").write_text(
        f"# {name}\n\n{description}\n\n"
        "<!-- Shared context for all agents in this project -->\n"
    )

    # Add initial members (handles removing from old projects + permissions)
    for agent in (members or []):
        _add_agent_to_project(name, agent)


def _delete_project(name: str) -> None:
    """Delete a project directory and clean up agent permissions."""
    import shutil

    project_dir = PROJECTS_DIR / name
    if not project_dir.exists():
        raise ValueError(f"Project '{name}' not found")

    # Read members before deleting
    meta_file = project_dir / "metadata.yaml"
    members: list[str] = []
    if meta_file.exists():
        with open(meta_file) as f:
            data = yaml.safe_load(f) or {}
        members = data.get("members", [])

    # Remove project blackboard permissions from all members
    for agent in members:
        _remove_project_blackboard_permissions(agent, name)

    shutil.rmtree(project_dir)


def _add_agent_to_project(project: str, agent: str) -> None:
    """Assign an agent to a project. Removes from old project if any."""
    project_dir = PROJECTS_DIR / project
    meta_file = project_dir / "metadata.yaml"
    if not meta_file.exists():
        raise ValueError(f"Project '{project}' not found")

    # Remove from old project first
    old_project = _get_agent_project(agent)
    if old_project and old_project != project:
        _remove_agent_from_project(old_project, agent)

    # Add to new project
    with open(meta_file) as f:
        data = yaml.safe_load(f) or {}
    members = data.get("members", [])
    if agent not in members:
        members.append(agent)
        data["members"] = members
        with open(meta_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    _add_project_blackboard_permissions(agent, project)


def _remove_agent_from_project(project: str, agent: str) -> None:
    """Remove an agent from a project (becomes standalone)."""
    project_dir = PROJECTS_DIR / project
    meta_file = project_dir / "metadata.yaml"
    if not meta_file.exists():
        raise ValueError(f"Project '{project}' not found")

    with open(meta_file) as f:
        data = yaml.safe_load(f) or {}
    members = data.get("members", [])
    if agent in members:
        members.remove(agent)
        data["members"] = members
        with open(meta_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    _remove_project_blackboard_permissions(agent, project)


def _add_project_blackboard_permissions(agent: str, project: str) -> None:
    """Grant blackboard access for a project member.

    Only grants ``projects/{project}/*``.  The MeshClient auto-prefixes
    all blackboard keys with the project namespace, so agents use natural
    keys (``context/market``) which are transparently stored as
    ``projects/{project}/context/market``.  This prevents cross-project
    leakage — each project's data lives under its own namespace.
    """
    perms = _load_permissions()
    agent_perms = perms.get("permissions", {}).get(agent)
    if agent_perms is None:
        return
    project_pattern = f"projects/{project}/*"
    for field in ("blackboard_read", "blackboard_write"):
        patterns = agent_perms.get(field, [])
        if project_pattern not in patterns:
            patterns.append(project_pattern)
        agent_perms[field] = patterns
    _save_permissions(perms)


def _remove_project_blackboard_permissions(agent: str, project: str) -> None:
    """Revoke all blackboard access when an agent leaves a project.

    Clears the project namespace pattern, restoring the agent to
    standalone state (no blackboard access).
    """
    perms = _load_permissions()
    agent_perms = perms.get("permissions", {}).get(agent)
    if agent_perms is None:
        return
    agent_perms["blackboard_read"] = []
    agent_perms["blackboard_write"] = []
    _save_permissions(perms)


def _remove_agent(name: str, stop_container: bool = False) -> None:
    """Remove an agent from config, permissions, and any project membership.

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
        except Exception:
            pass  # Docker unavailable or container already gone

    # Remove from project if member
    project = _get_agent_project(name)
    if project:
        try:
            _remove_agent_from_project(project, name)
        except ValueError:
            pass

    # Remove from agents.yaml
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {}
        agents_cfg.get("agents", {}).pop(name, None)
        with open(AGENTS_FILE, "w") as f:
            yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)

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


def _load_templates() -> dict[str, dict]:
    """Load available team templates from src/templates/."""
    available: dict[str, dict] = {}
    if not TEMPLATES_DIR.exists():
        return available
    for tpl_file in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(tpl_file) as f:
            tpl = yaml.safe_load(f) or {}
        name = tpl.get("name", tpl_file.stem)
        available[name] = tpl
    return available


def _apply_template(template_name: str, tpl: dict) -> list[str]:
    """Apply a team template, creating all agents. Returns list of agent names."""
    cfg = _load_config()
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    tpl_agents = tpl.get("agents", {})
    created: list[str] = []

    for agent_name, agent_def in tpl_agents.items():
        model = agent_def.get("model", default_model).replace("{default_model}", default_model)
        instructions = agent_def.get("instructions", "") or agent_def.get("system_prompt", "")
        _add_agent_to_config(
            name=agent_name,
            role=agent_def.get("role", agent_name),
            model=model,
            initial_instructions=instructions,
        )
        if "resources" in agent_def:
            agents_cfg: dict = {"agents": {}}
            if AGENTS_FILE.exists():
                with open(AGENTS_FILE) as f:
                    agents_cfg = yaml.safe_load(f) or {"agents": {}}
            if agent_name in agents_cfg.get("agents", {}):
                agents_cfg["agents"][agent_name]["resources"] = agent_def["resources"]
                with open(AGENTS_FILE, "w") as f:
                    yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)
        _add_agent_permissions(agent_name)
        skills_dir = PROJECT_ROOT / "skills" / agent_name
        skills_dir.mkdir(parents=True, exist_ok=True)
        created.append(agent_name)

    return created


def _default_description(name: str) -> str:
    """Return a default agent description."""
    return f"General-purpose {name} agent"


def _update_agent_field(name: str, field: str, value) -> None:
    """Update a single field in agents.yaml for an agent."""
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {"agents": {}}
    else:
        agents_cfg = {"agents": {}}
    if name in agents_cfg.get("agents", {}):
        agents_cfg["agents"][name][field] = value
        with open(AGENTS_FILE, "w") as f:
            yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)


_THINKING_LEVELS = ["off", "low", "medium", "high"]
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
    ``"budget"``, ``"thinking"``, ``"mcp_servers"``), or ``None``
    if nothing changed.  Callers decide how to apply the change
    (restart hint, live restart, cost-tracker update, etc.).
    """
    cfg = _load_config()
    agent_cfg = cfg.get("agents", {}).get(name, {})
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

    current_model = agent_cfg.get("model", default_model)
    current_desc = agent_cfg.get("role", "")
    budget_cfg = agent_cfg.get("budget", {})
    current_budget = budget_cfg.get("daily_usd") if budget_cfg else None
    current_thinking = agent_cfg.get("thinking", "off") or "off"
    current_mcp = agent_cfg.get("mcp_servers") or []

    click.echo(f"\n  {name}")
    click.echo(f"  Model:       {current_model}")
    click.echo(f"  Description: {current_desc or '(none)'}")
    if current_budget is not None:
        click.echo(f"  Budget:      ${current_budget:.2f}/day")
    click.echo(f"  Thinking:    {current_thinking}")
    mcp_summary = f"{len(current_mcp)} server{'s' if len(current_mcp) != 1 else ''}" if current_mcp else "(none)"
    if current_mcp:
        mcp_names = ", ".join(s.get("name", "?") for s in current_mcp)
        mcp_summary += f" ({mcp_names})"
    click.echo(f"  MCP servers: {mcp_summary}")
    click.echo("\n  What to change?\n")

    options = [
        ("model", current_model),
        ("description", truncate(current_desc, 50) or "(none)"),
        ("budget", f"${current_budget:.2f}/day" if current_budget is not None else "(none)"),
        ("thinking", current_thinking),
        ("MCP servers", mcp_summary),
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
            current_model, label="current",
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

    elif choice == 5:  # MCP servers
        servers = list(current_mcp)
        changed = False
        while True:
            click.echo()
            if servers:
                for i, s in enumerate(servers, 1):
                    click.echo(f"  {i}. {s.get('name', '?')} — {s.get('command', '?')}")
            else:
                click.echo("  (no servers)")
            click.echo("\n  1. add server  2. remove server  3. done")
            action = click.prompt("  Action", type=click.IntRange(1, 3), default=3)
            if action == 3:
                break
            elif action == 1:
                srv_name = click.prompt("  Server name")
                srv_command = click.prompt("  Command (e.g. npx)")
                srv_args_str = click.prompt("  Args (space-separated)", default="")
                srv_args = srv_args_str.split() if srv_args_str.strip() else []
                servers.append({"name": srv_name, "command": srv_command, "args": srv_args})
                changed = True
            elif action == 2:
                if not servers:
                    click.echo("  No servers to remove.")
                    continue
                for i, s in enumerate(servers, 1):
                    click.echo(f"  {i}. {s.get('name', '?')}")
                idx = click.prompt("  Remove", type=click.IntRange(1, len(servers)))
                removed = servers.pop(idx - 1)
                click.echo(f"  Removed '{removed.get('name', '?')}'")
                changed = True
        if changed:
            _update_agent_field(name, "mcp_servers", servers if servers else None)
            click.echo(f"Agent '{name}' MCP servers updated.")
            return "mcp_servers"
        click.echo("No change.")
        return None

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
