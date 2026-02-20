# Contributing to OpenLegion

We're glad you're here. OpenLegion is better when more people shape it.

Whether it's a typo fix, a new channel integration, or a wild idea for a feature — contributions of all sizes are welcome. This guide helps you get started quickly.

## Getting Set Up

```bash
git clone https://github.com/openlegion-ai/openlegion.git && cd openlegion
python3 -m venv .venv && source .venv/bin/activate
pip3 install -e ".[dev]"
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x
```

Tests pass? You're ready.

## Ways to Contribute

**Great first contributions:**
- Fix a bug (check [open issues](https://github.com/openlegion-ai/openlegion/issues))
- Improve documentation or add examples
- Add test coverage to an existing module
- Build a new `@skill` function ([see template below](#adding-a-new-skill))

**Bigger contributions we'd love help with:**
- New channels — Slack, WhatsApp, Matrix, etc.
- MCP server/tool integrations
- Performance improvements
- Internationalization

**Not sure where to start?** Open an issue and say hi — we'll point you in the right direction.

## How to Submit Changes

1. **Fork and branch** — work on a feature branch, not main
2. **Write tests** — every change should have test coverage
3. **Run the suite** — `pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x`
4. **Lint** — `ruff check src/ tests/ && ruff format src/ tests/`
5. **Open a PR** — describe what you changed and why. Link to the issue if there is one.

Keep PRs focused — one feature or fix per PR is easiest to review.

## Code Patterns

Follow what's already in the codebase:

- **Async by default** — `async def` for anything that does I/O
- **Pydantic at boundaries** — cross-component messages use models from `src/shared/types.py`
- **SQLite for state** — no external databases
- **`setup_logging(name)`** — for new modules

Where to put tests:

| You changed | Tests go in |
|---|---|
| `src/agent/builtins/*.py` | `tests/test_builtins.py` |
| `src/agent/loop.py` | `tests/test_loop.py` |
| `src/agent/memory.py` | `tests/test_memory.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/channels/*.py` | `tests/test_channels.py` |

## Adding a New Skill

The most common contribution — and a great first one. Here's the full template:

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

Add tests in `tests/test_builtins.py`, then open a PR.

## Adding a New Channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. Message routing is handled by the base class — you just need transport logic
4. Add startup in `src/cli.py:_start_channels()`
5. Add tests in `tests/test_channels.py`

## Architecture Boundaries

These keep agents secure. They're the few things we can't be flexible on:

1. **Agents never hold API keys** — external calls go through the mesh credential vault
2. **No `eval()` or `exec()` on untrusted input** — extend the safe parser instead
3. **Permission checks on cross-boundary operations** — new mesh endpoints must enforce the permission matrix
4. **Container isolation stays intact** — don't weaken resource limits or network restrictions
5. **New capabilities = new skills** — add `@skill` functions, not changes to the core loop

See `CLAUDE.md` for the full engineering guide.

## Reporting Bugs

Open an issue with:
- What you expected vs what happened
- Steps to reproduce
- Python version, OS, Docker version
- Relevant logs (redact API keys)

## Security Issues

Don't open a public issue. Email security@openlegion.dev — we'll coordinate a fix and responsible disclosure.

## Contributor License Agreement

By submitting a pull request to this repository, you agree that your contribution may be relicensed by the project maintainers under any open-source license, including future license changes.
