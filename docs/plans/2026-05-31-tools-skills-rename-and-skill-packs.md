# Tools / Skills Vocabulary Split + Pluggable Skill Packs

**Goal:** Expand agent use-cases without forcing agents to author their own code. Free up the word **"Skills"** for a user-pluggable, ecosystem-standard capability layer (`SKILL.md`, per [agentskills.io](https://agentskills.io)), and rename our existing executable `@skill` functions to **"Tools"** — the name the LLM API, MCP, and our own permission system already use for them.

**Core distinction (the whole decision in one line):**

- **Tools = code.** Executable functions the agent calls — built-in Python (`@tool`) or external via MCP. One verb each: `web_search`, `http_request`, `browser_navigate`.
- **Skills = instructions.** Packaged know-how (`SKILL.md`: YAML frontmatter + markdown body, optionally bundling helper scripts) that tells the agent *how to do a job using the tools it already has*. **Adds no new code to the trusted runtime.**

**Core invariant:** A Skill can only orchestrate Tools the agent is already permissioned for. The per-agent ACL in `src/host/permissions.py` (glob patterns over tool names) stays the single capability chokepoint. Installing a Skill never widens the permission surface — it composes existing grants.

**Mental model for users:** Skills are recipes; Tools are the kitchen appliances. A recipe can bundle a script, but the script still runs through the appliances the agent's permissions allow. **Skills are the product surface (browse / install / author / share); Tools are infrastructure (visible only in permissions + advanced views).** They are layers, not peers.

---

## Why this, and why now

1. **"Skill" was misnamed from day one.** Our `@skill`-decorated functions are executable code. The LLM function-calling spec, MCP, and our own plumbing already call these **tools** (`get_tool_definitions`, `ALLOWED_TOOLS`, `_HEARTBEAT_TOOLS`, `get_tool_sources`). The rename aligns the registration/storage layer with what the execution layer already says.
2. **The ecosystem standardized on the split.** Claude Agent Skills and Nous Hermes both implement the agentskills.io `SKILL.md` standard: *Tools* = executable functions, *Skills* = procedural packs. Adopting the standard's vocabulary means a skill written for Claude/Hermes runs on us unmodified, and our users' mental models transfer in both directions.
3. **Skills are the safe extension surface.** Today the only "add a capability" path is `create_skill` → agent writes Python → AST-denylist sandbox (`skill_tool.py`) → loaded into the **trusted** agent process. Denylist sandboxes around in-process code are fragile by nature. A `SKILL.md` adds zero code to the runtime — it's data that orchestrates already-permissioned tools. This becomes the default extension path; agent-authored Python gets demoted to advanced/operator-only.
4. **No backwards-compat burden.** The custom-tool user base is tiny, so we do a clean cutover — no shim layer, no dual-naming.

---

## Three-layer extension model (target state)

```
SKILLS   ← browse · install · author · share        (product surface)
  │        provenance: built-in / custom / imported
  │        = SKILL.md (markdown + frontmatter + optional scripts)
  ▼
TOOLS    ← capability substrate, mostly invisible    (infrastructure)
  │        provenance: built-in / MCP
  │        = executable functions (@tool Python, or MCP protocol)
```

For genuinely new *executable* third-party capability, the answer is **MCP (out-of-process), not in-process Python plugins** — it fits container isolation far better. We already have the client (`src/agent/mcp_client.py`).

---

## Tasks

| Task | Description | PR |
|---|---|---|
| Task 1 | **The rename.** `@skill`→`@tool`, `SkillRegistry`→`ToolRegistry`, `skills.py`→`tools.py`, `create_skill`/`reload_skills`→`create_tool`/`reload_tools`, storage dirs + env vars, dashboard labels, `src/shared/types.py` contract, docs. No behavior change. | PR 1 |
| Task 2 | **`SKILL.md` loader + store.** Parse frontmatter, directory-scan a skills store, model the pack (name/description/body/scripts/refs/config/env). | PR 2 |
| Task 3 | **Progressive disclosure tools.** `skills_list` (Level 0: names+descriptions) / `skill_view(name)` (Level 1: full body) / `skill_view(name, path)` (Level 2: reference file). | PR 2 |
| Task 4 | **Invocation.** Slash-command routing (`/competitor-research ...`) + natural-language match against descriptions. Inject resolved config values + declared env into the agent context on load. | PR 3 |
| Task 5 | **Install / distribution.** Operator-gated install path through the existing `src/marketplace.py`; per-team scoping via permissions (mirror `fleet_tool.apply_template`). Dashboard surface with built-in / custom / imported provenance. | PR 3 |
| Task 6 | **Demote agent-authored code.** Make `create_tool` (formerly `create_skill`) operator/advanced-only now that markdown Skills cover most "teach my agent X" cases. | PR 4 |

---

## Task 1 — the rename (PR 1, no behavior change)

**Blast radius:** ~462 `skill` mentions across 39 `.py` files, but most are mechanical (`@skill(` decorator calls + description strings). Structural work concentrates in:

| File | Hits | Work |
|---|---|---|
| `src/agent/skills.py` | 74 | Core: `@skill`→`@tool`, `SkillRegistry`→`ToolRegistry`; file → `tools.py`. Keep `_skill_staging` global semantics (threading-lock protected) — rename to `_tool_staging`. |
| `src/agent/builtins/skill_tool.py` | 50 | `create_skill`/`reload_skills`→`create_tool`/`reload_tools`; module → `tool_authoring_tool.py`. AST validation logic unchanged. |
| `src/marketplace.py` | 40 | Audit — this becomes the distribution hub for **both** Tools (MCP) and the new Skills. Disambiguate naming carefully here. |
| `src/agent/loop.py` | 40 | `self.skills`→`self.tools`, `_skill_filter_kw`. Most of loop.py already says "tool". |
| `src/agent/builtins/operator_tools.py` | 34 | Mostly description strings. |
| `src/host/runtime.py` | 25 | `SKILLS_DIR` env + container mount paths. |
| 33 more files | ≤28 each | Overwhelmingly `@skill(` decorator usages — find/replace. |

**Storage + env renames:**

| Before | After |
|---|---|
| `/data/custom_skills` | `/data/custom_tools` |
| `/app/marketplace_skills` | `/app/marketplace_tools` |
| `SKILLS_DIR` (default `/app/skills`) | `TOOLS_DIR` (default `/app/tools`) |
| `SKILL_AUTHORING.md` | `TOOL_AUTHORING.md` |

**Already correct (no change — proof the rename aligns with reality):** `get_tool_definitions`, `ALLOWED_TOOLS`, `_HEARTBEAT_TOOLS`, `_BLACKBOARD_TOOLS`, `get_tool_sources`, the per-tool permission globs in `permissions.py`.

**Contract surface:** check `src/shared/types.py` (2 hits) and `DashboardEvent.type` for any `skill*` literals — these are cross-component and need coordinated rename. Dashboard JS (`src/dashboard/`) labels + any tool-provenance UI tags.

---

## SKILL.md format we adopt (Task 2)

Wire-compatible with agentskills.io so ecosystem packs drop in unmodified:

```
skills/<category>/<skill-name>/
├── SKILL.md          # required
├── scripts/          # optional helper scripts (run via existing tools)
├── references/       # optional, loaded on demand (Level 2)
└── templates/
```

```markdown
---
name: competitor-research
description: Research a competitor across web + filings, produce a cited brief
version: 1.0.0
author: ...
license: MIT
metadata:
  hermes:                          # ecosystem namespace; map to our own on ingest
    requires_toolsets: [web_search, http_request]
    config:
      - {key: research.max_sources, default: "10"}
required_environment_variables:
  - {name: SOME_API_KEY, prompt: "API key", required_for: "..."}
---
# When to Use
...trigger conditions...
# Procedure
1. Use `web_search` on these angles ...
2. Run `${SKILL_DIR}/scripts/dedupe.py` ...
# Pitfalls / Verification
```

- **Progressive disclosure** keeps token cost proportional to *use*, not catalog size.
- **Config injection:** resolved config values appended to the skill message on load.
- **Env passthrough:** declared env vars routed to the exec/terminal sandbox so bundled scripts can use them — gated by our existing credential vault rules, never exposing SYSTEM_* tier.
- **Template tokens** (`${SKILL_DIR}`, `${SESSION_ID}`) substituted at load.

---

## Open decisions (resolve before PR 2)

1. **Wire-compatibility level** — full agentskills.io drop-in (constrains us to their frontmatter, incl. the `metadata.hermes.*` namespace) vs. our own dialect with an import shim. *Lean: full compat, ingest-time mapping of the vendor namespace.*
2. **Skill scope** — per-team store vs. global store with permission grants. *Lean: per-team, mirroring fleet templates.*
3. **Self-authoring** — keep `create_tool` (agent writes Python) at all, or humans/MCP only? *Lean: keep but operator-gated (Task 6).*
4. **Marketplace shape** — is `src/marketplace.py` the home for Skill distribution, and does it serve Tools (MCP) too?

---

## Test plan

- **Task 1:** existing `tests/test_skills.py` → `tests/test_tools.py`; assert the registry, decorator, discovery, dependency injection, and parallel-safety behavior are unchanged under the new names. Grep-guard test: no `@skill`/`SkillRegistry` left in `src/`. CI ruff + pytest green with zero behavior diff.
- **Task 2–3:** SKILL.md parse (valid + malformed frontmatter), directory scan, progressive-disclosure levels, token-budget assertion on Level 0.
- **Task 4–5:** slash-command + NL-match routing; config/env injection; operator-gated install (mirror `apply_template` permission tests); a real agentskills.io pack installs and runs end-to-end using only already-granted tools.
- **Security:** a Skill cannot invoke a tool the agent lacks permission for; bundled-script env passthrough never leaks SYSTEM_* credentials.
```

