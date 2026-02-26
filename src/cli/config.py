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

logger = logging.getLogger("cli")

# ── Path constants ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh.yaml"
AGENTS_FILE = PROJECT_ROOT / "config" / "agents.yaml"
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"
PROJECTS_DIR = PROJECT_ROOT / "config" / "projects"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DOCKER_IMAGE = "openlegion-agent:latest"

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

_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-5.2",
        "openai/gpt-5.2-pro",
        "openai/gpt-5.1-codex",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/o3",
        "openai/o4-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ],
    "anthropic": [
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-5-20250929",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "gemini": [
        "gemini/gemini-3-pro-preview",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
    ],
    "xai": [
        "xai/grok-4-1-fast-reasoning",
        "xai/grok-4",
        "xai/grok-3",
        "xai/grok-3-mini",
    ],
    "deepseek": [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
    "moonshot": [
        "moonshot/kimi-k2.5",
        "moonshot/kimi-k2",
        "moonshot/moonshot-v1-128k",
    ],
    "groq": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/llama-3-groq-70b-tool-use",
    ],
    "minimax": [
        "minimax/MiniMax-M2.5",
        "minimax/MiniMax-M2.5-Lightning",
        "minimax/MiniMax-M2.1",
        "minimax/MiniMax-M2.1-lightning",
        "minimax/MiniMax-M2",
    ],
    "zai": [
        "zai/glm-5",
    ],
}

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
    except Exception:
        return False


def _check_docker_image() -> bool:
    """Check if the agent Docker image exists."""
    try:
        import docker
        client = docker.from_env()
        client.images.get(DOCKER_IMAGE)
        return True
    except Exception:
        return False


def _docker_image_is_stale() -> bool:
    """Check if source files are newer than the Docker image."""
    try:
        from datetime import datetime, timezone

        import docker
        client = docker.from_env()
        image = client.images.get(DOCKER_IMAGE)
        created_str = image.attrs.get("Created", "")
        if not created_str:
            return True
        image_time = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

        src_dirs = [PROJECT_ROOT / "src" / "agent", PROJECT_ROOT / "src" / "shared"]
        for src_dir in src_dirs:
            if not src_dir.exists():
                continue
            for py_file in src_dir.rglob("*.py"):
                file_mtime = datetime.fromtimestamp(py_file.stat().st_mtime, tz=timezone.utc)
                if file_mtime > image_time:
                    return True

        dockerfile = PROJECT_ROOT / "Dockerfile.agent"
        if dockerfile.exists():
            df_mtime = datetime.fromtimestamp(dockerfile.stat().st_mtime, tz=timezone.utc)
            if df_mtime > image_time:
                return True
        return False
    except Exception:
        return False


def _ensure_docker_image() -> None:
    """Build the Docker image if missing or stale."""
    if not _check_docker_image():
        _build_docker_image()
    elif _docker_image_is_stale():
        click.echo("Source code changed since last Docker build.")
        _build_docker_image()


def _build_docker_image() -> None:
    """Build the agent Docker image with visible progress."""
    click.echo("Building Docker image...")
    click.echo("  First build downloads base image + Chromium (~2 min). Rebuilds are fast.\n")
    proc = subprocess.Popen(
        [
            "docker", "build",
            "--platform", "linux/amd64",
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
        click.echo(f"  docker build --platform linux/amd64 -t {DOCKER_IMAGE} -f Dockerfile.agent .", err=True)
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
        "blackboard_read": ["context/*", "tasks/*", "goals/*", "signals/*", "artifacts/*"],
        "blackboard_write": ["context/*", "goals/*", "signals/*", "artifacts/*"],
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
    """Add projects/<name>/* patterns to agent's blackboard permissions."""
    perms = _load_permissions()
    agent_perms = perms.get("permissions", {}).get(agent)
    if agent_perms is None:
        return
    pattern = f"projects/{project}/*"
    for field in ("blackboard_read", "blackboard_write"):
        patterns = agent_perms.get(field, [])
        if pattern not in patterns:
            patterns.append(pattern)
            agent_perms[field] = patterns
    _save_permissions(perms)


def _remove_project_blackboard_permissions(agent: str, project: str) -> None:
    """Remove projects/<name>/* patterns from agent's blackboard permissions."""
    perms = _load_permissions()
    agent_perms = perms.get("permissions", {}).get(agent)
    if agent_perms is None:
        return
    pattern = f"projects/{project}/*"
    for field in ("blackboard_read", "blackboard_write"):
        patterns = agent_perms.get(field, [])
        if pattern in patterns:
            patterns.remove(pattern)
            agent_perms[field] = patterns
    _save_permissions(perms)


def _remove_agent(name: str) -> None:
    """Remove an agent from config, permissions, and any project membership."""
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


def _pick_model_interactive(default_model: str, label: str = "current") -> str:
    """Show model picker for the default model's provider. Returns selected model."""
    provider = default_model.split("/")[0] if "/" in default_model else "anthropic"
    models = _PROVIDER_MODELS.get(provider, [default_model])
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


def _edit_agent_interactive(name: str) -> str | None:
    """Interactive property editor for an agent. Reads fresh config.

    Returns the field name that was changed (``"model"``, ``"role"``,
    ``"budget"``), or ``None`` if nothing changed.  Callers
    decide how to apply the change (restart hint, live restart,
    cost-tracker update, etc.).
    """
    cfg = _load_config()
    agent_cfg = cfg.get("agents", {}).get(name, {})
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

    current_model = agent_cfg.get("model", default_model)
    current_desc = agent_cfg.get("role", "")
    budget_cfg = agent_cfg.get("budget", {})
    current_budget = budget_cfg.get("daily_usd") if budget_cfg else None

    click.echo(f"\n  {name}")
    click.echo(f"  Model:       {current_model}")
    click.echo(f"  Description: {current_desc or '(none)'}")
    if current_budget is not None:
        click.echo(f"  Budget:      ${current_budget:.2f}/day")
    click.echo("\n  What to change?\n")

    options = [
        ("model", current_model),
        ("description", _truncate(current_desc, 50) or "(none)"),
        ("budget", f"${current_budget:.2f}/day" if current_budget is not None else "(none)"),
    ]

    for i, (label, val) in enumerate(options, 1):
        click.echo(f"  {i}. {label:<16} {val}")

    choice = click.prompt(
        "\n  Select",
        type=click.IntRange(1, len(options)),
        default=1,
    )

    if choice == 1:  # model
        new_model = _pick_model_interactive(current_model, label="current")
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
            _update_agent_field(name, "budget", {"daily_usd": new_budget})
            click.echo(f"Agent '{name}' budget: ${new_budget:.2f}/day")
            return "budget"
        except ValueError:
            click.echo("Invalid number. Budget not changed.")
            return None

    return None


def _truncate(text: str, length: int) -> str:
    """Truncate text with ellipsis if longer than length."""
    if len(text) > length:
        return text[:length] + "..."
    return text


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
