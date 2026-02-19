# Contributing to OpenLegion

Thanks for your interest in OpenLegion. This guide covers everything you need to go from idea to merged PR.

## Quick Start

```bash
git clone https://github.com/openlegion/openlegion.git && cd openlegion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x
```

If the tests pass, you're ready.

## What We're Looking For

**High-priority contributions:**
- New channels (Slack, WhatsApp, Matrix, etc.)
- MCP server/tool integrations
- Built-in skills (new `@skill` functions in `src/agent/builtins/`)
- Bug fixes with reproduction steps
- Test coverage improvements
- Documentation fixes and examples

**We'll probably decline:**
- Framework dependencies (LangChain, Redis, Kubernetes, etc.)
- Changes that weaken container isolation or security boundaries
- Features that require agents to hold API keys directly
- Large refactors without prior discussion

When in doubt, open an issue first.

## Development Workflow

### 1. Pick or create an issue

- Check [open issues](https://github.com/openlegion/openlegion/issues) for something you'd like to work on
- Comment on the issue to claim it — prevents duplicate effort
- For new features, open an issue describing your proposed change before writing code

### 2. Branch and develop

```bash
git checkout -b your-branch-name
# Make your changes
```

Follow the existing patterns in the codebase:
- **Async by default** — use `async def` for any function that does I/O
- **Small modules** — no file over ~800 lines
- **Pydantic at boundaries** — cross-component messages use models from `src/shared/types.py`
- **SQLite for state** — no external databases
- **`setup_logging(name)`** — for all new modules

### 3. Write tests

Every change needs tests. Match the existing patterns:

| You changed | Add tests in |
|---|---|
| `src/agent/builtins/*.py` | `tests/test_builtins.py` |
| `src/agent/loop.py` | `tests/test_loop.py` |
| `src/agent/memory.py` | `tests/test_memory.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/host/credentials.py` | `tests/test_credentials.py` |
| `src/channels/*.py` | `tests/test_channels.py` |
| New built-in skill | `tests/test_builtins.py` |

```bash
# Run the test suite (no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Run a specific test file
pytest tests/test_builtins.py -x -v
```

### 4. Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### 5. Submit a PR

- Keep PRs focused — one feature or fix per PR
- Write a clear description of what changed and why
- Reference the issue number if applicable
- All tests must pass in CI before merge

## Code Style

- **Python 3.12+** features are fine (type unions with `|`, etc.)
- **Type hints** on function signatures, especially public APIs
- **Docstrings** on public classes and non-obvious functions — skip them for self-explanatory helpers
- **No wildcard imports** — always import explicitly
- **`TYPE_CHECKING` guards** for imports that would cause circular dependencies

## Architecture Rules

These are non-negotiable. PRs that violate them will be rejected:

1. **Agents never hold API keys.** External calls go through the mesh credential vault.
2. **No `eval()` or `exec()` on untrusted input.** Extend the safe parser if you need conditional logic.
3. **Permission checks on every cross-boundary operation.** New mesh endpoints must enforce the permission matrix.
4. **Container isolation is not optional.** Don't weaken resource limits, network restrictions, or privilege settings.
5. **Skills over core changes.** New agent capabilities should be `@skill` functions, not changes to the agent loop.

See `CLAUDE.md` for the full engineering guide.

## Adding a New Built-in Skill

The most common contribution. Here's the template:

```python
# src/agent/builtins/your_tool.py

from src.agent.skills import skill

@skill(
    name="your_tool",
    description="Clear description for the LLM — when should it use this tool?",
    parameters={
        "param1": {"type": "string", "description": "What this param is for"},
        "param2": {"type": "integer", "description": "Optional param", "default": 10},
    },
)
async def your_tool(param1: str, param2: int = 10, *, mesh_client=None) -> dict:
    # mesh_client, workspace_manager, and memory_store are auto-injected
    # when declared as keyword-only args
    return {"result": "value"}
```

Then add tests in `tests/test_builtins.py` following the existing patterns.

## Adding a New Channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. Message routing is handled by the base class — you just need transport logic
4. Add startup in `src/cli.py:_start_channels()`
5. Add tests in `tests/test_channels.py`

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version, OS, Docker version
- Relevant logs (redact any API keys)

## Security Vulnerabilities

**Do not open a public issue for security vulnerabilities.** Email security@openlegion.dev with details. We'll coordinate a fix and disclosure timeline.

## Developer Certificate of Origin

By contributing to OpenLegion, you agree that your contributions are submitted under the project's MIT License and that you have the right to submit them.

Every commit must be signed off with your real name and email, certifying the [Developer Certificate of Origin (DCO)](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Add `Signed-off-by` to your commits:

```bash
git commit -s -m "Add new feature"
# Produces: Signed-off-by: Your Name <your@email.com>
```

## License

All contributions are licensed under the [MIT License](LICENSE). By submitting a pull request, you agree that your contribution falls under this license.

The OpenLegion maintainers reserve the right to offer commercial products and services built on top of this open-source project. Your contributions to the open-source project will always remain available under the MIT License.
