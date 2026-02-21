# Security Model

OpenLegion is designed assuming agents will be compromised. Defense-in-depth with five layers prevents a compromised agent from accessing credentials, escaping isolation, or affecting other agents.

## Five Security Layers

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| 1. Runtime isolation | Docker containers or Sandbox microVMs | Agent escape, kernel exploits |
| 2. Container hardening | Non-root user, no-new-privileges, memory/CPU limits | Privilege escalation, resource abuse |
| 3. Credential separation | Vault holds keys, agents call via proxy | Key leakage, unauthorized API use |
| 4. Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs | Unauthorized data access |
| 5. Input validation | Path traversal prevention, safe condition eval, token budgets, iteration limits | Injection, runaway loops |
| 6. Unicode sanitization | Invisible character stripping at three choke points | Prompt injection via invisible Unicode |

## Runtime Isolation

### Docker Containers (Default)

Agents run as non-root (UID 1000) with:
- `no-new-privileges` security option
- 512MB memory limit
- 50% CPU quota
- No host filesystem access (only `/data` volume)
- Bridge network with port mapping (no direct host network on macOS/Windows)

```bash
openlegion start  # Default container isolation
```

### Docker Sandbox MicroVMs

Each agent gets its own Linux kernel via hypervisor isolation:
- Apple Virtualization.framework (macOS) or Hyper-V (Windows)
- Full kernel boundary between agents
- Communication only via `docker sandbox exec` transport
- Even code execution inside the agent cannot see other agents or the host

```bash
openlegion start --sandbox  # MicroVM isolation (Docker Desktop 4.58+)
```

## Credential Vault

Agents **never** hold API keys. The credential vault (`src/host/credentials.py`) works as follows:

1. Credentials are loaded from `OPENLEGION_CRED_*` environment variables on the host
2. Agents make API calls by POSTing to `/mesh/api` on the mesh host
3. The vault injects the appropriate credentials server-side
4. The response is relayed back to the agent
5. Budget limits are enforced before dispatching, token usage recorded after

### Adding New Service Integrations

New external services are added as vault handlers, not as agent-side code:

```python
# In src/host/credentials.py
# 1. Add provider detection in _detect_provider()
# 2. Add credential injection in _inject_credentials()
# 3. The agent calls it like any other API through the mesh proxy
```

## Permission Matrix

Every inter-agent operation checks per-agent ACLs defined in `config/permissions.json`:

```json
{
  "researcher": {
    "can_message": ["orchestrator"],
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"],
    "blackboard_read": ["tasks/*", "context/*"],
    "blackboard_write": ["context/prospect_*"],
    "allowed_apis": ["llm", "brave_search"]
  }
}
```

- **Glob patterns** for blackboard paths (`tasks/*` matches `tasks/abc123`)
- **Explicit allowlists** for messaging, pub/sub, and API access
- **Default deny** -- if not listed, it's blocked
- Enforced at the mesh host before every operation

## Input Validation

### Path Traversal Prevention

Agent file tools (`src/agent/builtins/file_tool.py`) validate all paths are within `/data`:
- Resolves symlinks before checking
- Rejects `../` traversal attempts
- All file operations are scoped to the container's `/data` volume

### Safe Condition Evaluation

Workflow conditions (`src/host/orchestrator.py`) use a regex-based parser:
- No `eval()` or `exec()` -- ever
- Supports comparisons (`>=`, `==`, `!=`, `<`, `>`, `<=`)
- Dot-notation variable access only (`step.result.score`)
- Single comparison per condition (no boolean operators)

### Bounded Execution

- Task mode: 20 iterations maximum (`AgentLoop.MAX_ITERATIONS`)
- Chat mode: 30 tool rounds maximum (`CHAT_MAX_TOOL_ROUNDS`)
- Per-agent token budgets enforced at the vault layer
- Prevents runaway loops and unbounded spend

## Unicode Sanitization (Prompt Injection Defense)

Agents process untrusted text from user messages, web pages, HTTP responses, tool outputs, blackboard data, and MCP servers. Attackers can embed invisible instructions using tag characters (U+E0001-E007F), RTL overrides (U+202A-202E), zero-width spaces, variation selectors, and other invisible codepoints that LLM tokenizers decode while being invisible to humans.

`sanitize_for_prompt()` in `src/shared/utils.py` strips these at three choke points:

| Choke Point | File | What It Covers |
|-------------|------|----------------|
| User input | `src/agent/server.py` | All user messages from all channels/CLI |
| Tool results | `src/agent/loop.py` | All tool outputs (browser, web search, HTTP, file, exec, memory, MCP) |
| System prompt context | `src/agent/loop.py` | Workspace bootstrap, blackboard goals, memory facts, learnings, tool history |

### What Gets Stripped

- **Dangerous categories** (Cc, Cf, Co, Cs, Cn) except TAB/LF/CR, ZWNJ/ZWJ, VS15/VS16
- **Data smuggling vectors**: VS1-14, VS17-256, Combining Grapheme Joiner, Hangul fillers, Object Replacement
- **Normalization**: U+2028/U+2029 (line/paragraph separator) to `\n`

### What Is Preserved

Normal text in all scripts (Arabic, Hebrew, CJK, Devanagari, etc.), emoji with ZWJ sequences, ZWNJ for Persian/Arabic, tabs, newlines, and VS15/VS16 for emoji presentation.

### Adding New Paths to LLM Context

If you add a new path where untrusted text reaches LLM context (new tool, new system prompt section, new message source), wrap it with `sanitize_for_prompt()`. See `tests/test_sanitize.py` for the full test suite.

## Mesh Authentication

Each agent receives a unique auth token at startup (`MESH_AUTH_TOKEN`). All requests from agents to the mesh include this token for verification. This prevents:
- Spoofed agent requests
- Container-to-container communication bypassing the mesh
- Unauthorized access to mesh endpoints
