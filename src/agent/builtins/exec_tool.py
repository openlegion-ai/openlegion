"""Shell command execution inside the agent container.

The container IS the sandbox -- resource limits, no host access,
no credentials. Commands run as the container user.
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile

from src.agent.tools import tool

_MAX_OUTPUT = 100_000
_MAX_TIMEOUT = 300

# ── execute_code (code-as-action) — Phase 2 of the operator memory/context
# overhaul (docs/plans/2026-06-09-operator-memory-context-overhaul.md §7, C2).
#
# Lets the model write ONE Python block and get back only what it ``print()``s,
# collapsing several shell/tool rounds into a single turn. Intermediate values
# never enter context. v1 is deliberately a pure Python-execution + stdout
# capture tool — the hermes-style "call agent tools from inside the code" bridge
# is OUT of scope (see _DEFERRED note below). No recursion into execute_code or
# delegate.
#
# Ships dormant: gated behind OPENLEGION_EXECUTE_CODE (default off). When off
# the schema is hidden from the worker surface (see _EXECUTE_CODE_TOOLS in
# loop.py) AND the handler self-rejects (defense-in-depth for the operator
# allowlist path).
EXECUTE_CODE_ENABLED_ENV = "OPENLEGION_EXECUTE_CODE"

# Tighter cap than run_command's 100 KB: this is "what the model printed",
# meant to flow straight back into context, so keep it lean.
_CODE_MAX_OUTPUT = 20_000
_CODE_MAX_TIMEOUT = 120

# Env-var name substrings that mark a value as sensitive. Matched
# case-insensitively against each var NAME (never the value). Agents already
# never hold API keys (those live mesh-side), but the child gets a scrubbed
# env regardless — defense-in-depth, and it future-proofs against any var that
# does leak into the container environment.
_SENSITIVE_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CRED")
# Whole-name prefixes that are stripped outright (the OpenLegion namespace
# carries config + per-agent identity we don't want a code snippet reading).
_SENSITIVE_PREFIXES = ("OPENLEGION_", "OL_")
# Minimal allowlist of harmless vars the child genuinely needs to behave like a
# normal Python process (PATH for subprocess lookups, locale, tmp). Everything
# else is dropped — the child does NOT inherit the agent's full environment.
_SAFE_ENV_KEYS = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "TERM",
    "PYTHONIOENCODING", "PYTHONHASHSEED",
)


def execute_code_enabled() -> bool:
    """True iff the code-as-action ``execute_code`` tool is opted in."""
    return os.environ.get(EXECUTE_CODE_ENABLED_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _is_sensitive_env_name(name: str) -> bool:
    """True iff an env-var name should be scrubbed from the child process."""
    upper = name.upper()
    if any(upper.startswith(prefix) for prefix in _SENSITIVE_PREFIXES):
        return True
    return any(part in upper for part in _SENSITIVE_NAME_PARTS)


def _scrubbed_env() -> dict[str, str]:
    """Build a minimal, credential-free environment for the child process.

    Strategy is allowlist-first (only ``_SAFE_ENV_KEYS`` pass through) with a
    sensitive-name denylist applied on top as belt-and-suspenders. The agent's
    full env is never forwarded.

    SCOPE: this scrubs ENVIRONMENT VARIABLES only. It does NOT sandbox the
    filesystem — a snippet can still read any file the container user owns
    (e.g. ``~/.netrc``). That is by design and identical to ``run_command``;
    the Docker container hardening (no host mounts, no host credentials, agents
    hold no API keys) is the actual boundary, not this scrub.
    """
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None and not _is_sensitive_env_name(key):
            env[key] = val
    # Force unbuffered stdout so print() output is captured even on early exit.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child AND any grandchildren it spawned.

    The child is started in its own session (``start_new_session=True``), so it
    leads a process group. Killing the group reaps any subprocesses the snippet
    forked — otherwise a snippet that spawns a long-running child leaves orphans
    that accumulate against the container ``pids_limit``.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


@tool(
    name="run_command",
    description=(
        "Run a shell command in your Linux environment and return stdout+stderr. "
        "You have Python, Node.js, curl, git, and standard tools. "
        "Use for: installing packages, running scripts, processing data, "
        "downloading files, or any system operation. "
        "Do NOT use run_command to modify workspace identity files "
        "(SOUL.md, INSTRUCTIONS.md, HEARTBEAT.md, USER.md, MEMORY.md) — "
        "use update_workspace instead."
    ),
    parameters={
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Max seconds to wait (default 30)",
            "default": 30,
        },
        "workdir": {
            "type": "string",
            "description": "Working directory (default /data)",
            "default": "/data",
        },
    },
)
async def exec_command(command: str, timeout: int = 30, workdir: str = "/data") -> dict:
    """Run a shell command and return exit code + output."""
    timeout = min(timeout, _MAX_TIMEOUT)
    # Inside the container, confine workdir to /data to prevent directory escape.
    # Outside containers (dev/test), /data won't exist — skip the check.
    if os.path.isdir("/data"):
        resolved = os.path.realpath(workdir)
        if resolved != "/data" and not resolved.startswith("/data/"):
            return {"exit_code": -1, "stdout": "", "stderr": f"workdir must be under /data, got: {workdir}"}
        workdir = resolved
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        stdout = stdout_bytes.decode(errors="replace")[:_MAX_OUTPUT]
        stderr = stderr_bytes.decode(errors="replace")[:_MAX_OUTPUT]
        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


@tool(
    name="execute_code",
    description=(
        "Run a Python snippet in your Linux environment and get back ONLY what "
        "you print(). Use this to collapse several steps into one turn: fetch, "
        "compute, filter, and print just the final answer — intermediate values "
        "never come back to you, only your printed output. The snippet runs as a "
        "fresh `python3` process (no state carries over between calls). On "
        "failure you get the error (stderr) and a non-zero exit code. "
        "print() what you need; everything else is discarded."
    ),
    parameters={
        "code": {"type": "string", "description": "Python source to execute"},
        "timeout": {
            "type": "integer",
            "description": "Max seconds to wait (default 30)",
            "default": 30,
        },
    },
)
async def execute_code(code: str, timeout: int = 30) -> dict:
    """Run a Python snippet and return its printed stdout (stderr on failure).

    Reuses run_command's container-exec mechanism (the Docker container IS the
    security boundary). The snippet is written to a temp file and run with
    ``python3 <file>`` — temp-file over ``-c`` to sidestep shell/quoting issues
    — under a SCRUBBED environment (no credentials, no OpenLegion namespace; see
    ``_scrubbed_env``). Intermediate values stay out of context: only what the
    snippet print()s is returned.
    """
    # Ships dormant. The worker surface hides the schema when the flag is off
    # (see _EXECUTE_CODE_TOOLS in loop.py); this self-reject also covers the
    # operator allowlist path, which doesn't go through that exclusion.
    if not execute_code_enabled():
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                "execute_code is disabled. Set OPENLEGION_EXECUTE_CODE=1 to enable "
                "it, or use run_command instead."
            ),
        }
    if not isinstance(code, str) or not code.strip():
        return {"exit_code": -1, "stdout": "", "stderr": "code must be a non-empty string"}

    timeout = min(timeout, _CODE_MAX_TIMEOUT)
    workdir = "/data" if os.path.isdir("/data") else None
    path = None
    proc: asyncio.subprocess.Process | None = None
    try:
        # Write the snippet to a temp file so we exec `python3 <file>` with no
        # shell interpolation of the (untrusted, possibly quote-laden) source.
        fd, path = tempfile.mkstemp(suffix=".py", prefix="execcode_")
        with os.fdopen(fd, "w") as fh:
            fh.write(code)
        # Read-only before exec — closes the (already narrow, container-only)
        # TOCTOU window between write and run.
        os.chmod(path, 0o400)

        # start_new_session: the child leads its own process group so a timeout
        # can SIGKILL the whole group (child + any snippet-spawned grandchildren).
        proc = await asyncio.create_subprocess_exec(
            "python3",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=_scrubbed_env(),
            start_new_session=True,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        stdout = stdout_bytes.decode(errors="replace")[:_CODE_MAX_OUTPUT]
        stderr = stderr_bytes.decode(errors="replace")[:_CODE_MAX_OUTPUT]
        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            # Only surface stderr on failure — keeps the success path to just
            # what the model printed.
            "stderr": stderr if proc.returncode != 0 else "",
        }
    except asyncio.TimeoutError:
        if proc is not None:
            _kill_process_group(proc)
            await proc.wait()
        return {"exit_code": -1, "stdout": "", "stderr": f"Code timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass


# DEFERRED (v1 → follow-up): the hermes-style tool-call bridge — letting the
# executed snippet invoke other agent tools (e.g. a `call_tool(...)` helper in
# scope) to truly collapse N tool rounds into one — is intentionally NOT built
# here. It is a larger security surface (tool whitelist, no recursion into
# execute_code/delegate) and ships as its own PR. v1 is pure Python + stdout.
