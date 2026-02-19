"""Shell command execution inside the agent container.

The container IS the sandbox -- resource limits, no host access,
no credentials. Commands run as the container user.
"""

from __future__ import annotations

import asyncio

from src.agent.skills import skill

_MAX_OUTPUT = 100_000


@skill(
    name="exec",
    description=(
        "Run a shell command and return stdout+stderr. You have a full Linux "
        "environment with Python, Node.js, curl, git, and standard tools. "
        "Use for: installing packages, running scripts, processing data, "
        "downloading files, or any system operation."
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
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}
