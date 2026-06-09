"""Tests for C3 — cache-prefix stabilization.

Always-on: per-turn-VOLATILE prompt fragments (context/round warnings, operator
playbooks, 5-min runtime context, the ``## Recent`` memory slice) are relocated
OUT of the cached system block and re-injected AFTER the mesh-side cache
breakpoint (into the last message). Goals:

  * volatile fragments are NOT in the system prompt but ARE present later in the
    message list; role-alternation stays valid.
  * the STABLE prefix is byte-identical across two builds that differ ONLY in
    volatile content (the whole point — prove the prefix is stable).

These exercise the prompt BUILDERS directly (no LLM round-trip needed).
"""

from src.agent.context import ContextManager
from src.agent.workspace import WorkspaceManager
from src.shared.types import TaskAssignment
from tests.test_loop import _make_loop

# A compiled head + an append-only log so get_memory_injection() produces a
# real ``## Recent`` slice. The head is the STABLE part; the log tail is the
# VOLATILE ``## Recent`` slice that must move out of the cached prefix.
_HEAD = "The user prefers concise answers. The live fleet is content-seo."
_MEMORY_WITH_RECENT = (
    "# Long-Term Memory\n\n"
    "<!-- compiled:begin -->\n"
    f"{_HEAD}\n"
    "<!-- compiled:end -->\n\n"
    "## 2026-06-09T10:00 flush\n\n"
    "- learned: the user is named Jeff and wants SEO work\n"
)
_MEMORY_HEAD_ONLY = (
    "# Long-Term Memory\n\n"
    "<!-- compiled:begin -->\n"
    f"{_HEAD}\n"
    "<!-- compiled:end -->\n"
)

_RECENT_MARKER = "## Recent"
_RECENT_BODY = "the user is named Jeff and wants SEO work"


def _operator_loop(tmp_path):
    """A loop with a real workspace + context manager, operator role."""
    ws = WorkspaceManager(workspace_dir=str(tmp_path / "ws"))
    (ws.root / "SOUL.md").write_text("# Identity\n\nI am the operator.\n")
    (ws.root / "INSTRUCTIONS.md").write_text("# Instructions\n\nDo the thing.\n")
    (ws.root / "MEMORY.md").write_text(_MEMORY_WITH_RECENT)

    loop = _make_loop()
    loop.workspace = ws
    # Operator: allowed_tools is the signal _is_operator keys off.
    loop._is_operator = True
    loop._allowed_tools = frozenset({"notify_user"})
    loop.context_manager = ContextManager(max_tokens=200, model="test-model")
    return loop


def _chat_prompt(loop, **kw):
    """Build a chat system prompt with a default introspect payload."""
    introspect = kw.pop("introspect_data", {"permissions": {}, "budget": None})
    return loop._build_chat_system_prompt(introspect_data=introspect, **kw)


# ── Volatile moves out, lands in the message list ─────────────────────


def test_recent_slice_leaves_system_prompt(tmp_path):
    loop = _operator_loop(tmp_path)
    prompt = _chat_prompt(loop)

    # The volatile ``## Recent`` slice is NO LONGER in the system prompt …
    assert _RECENT_MARKER not in prompt
    assert _RECENT_BODY not in prompt
    # … but the STABLE head still is.
    assert _HEAD in prompt
    # … and the recent slice is stashed for re-injection.
    assert _RECENT_BODY in loop._volatile_prompt_suffix


def test_context_warning_leaves_system_prompt(tmp_path):
    loop = _operator_loop(tmp_path)
    loop._chat_messages = [{"role": "user", "content": "x " * 400}]

    prompt = _chat_prompt(loop)

    assert "CONTEXT WARNING" not in prompt
    assert "CONTEXT WARNING" in loop._volatile_prompt_suffix


def test_volatile_reinjected_into_last_message(tmp_path):
    """Relocated content must still REACH the model — appended to the last
    message, after the cache breakpoint, without adding a message."""
    loop = _operator_loop(tmp_path)
    loop._chat_messages = [{"role": "user", "content": "Hello operator " * 60}]
    _chat_prompt(loop)

    out = loop._messages_with_volatile(loop._chat_messages)
    # No new message added (role-alternation preserved).
    assert len(out) == len(loop._chat_messages)
    assert out[-1]["role"] == "user"
    # Persistent list NOT mutated.
    assert "Live context" not in loop._chat_messages[-1]["content"]
    # Volatile content present in the transient copy's last message.
    assert _RECENT_BODY in out[-1]["content"]
    assert "CONTEXT WARNING" in out[-1]["content"]


def test_reinjection_handles_multimodal_last_message(tmp_path):
    """A list-content (multimodal) last message gets a trailing text block, not
    a clobbered string — and no extra message."""
    loop = _operator_loop(tmp_path)
    loop._chat_messages = [
        {"role": "user", "content": [{"type": "text", "text": "pic " * 30}]},
    ]
    _chat_prompt(loop)

    out = loop._messages_with_volatile(loop._chat_messages)
    assert len(out) == len(loop._chat_messages)
    last = out[-1]["content"]
    assert isinstance(last, list)
    assert last[-1]["type"] == "text"
    assert _RECENT_BODY in last[-1]["text"]
    # Original blocks untouched.
    assert loop._chat_messages[-1]["content"][0]["text"].startswith("pic ")
    assert len(loop._chat_messages[-1]["content"]) == 1


# ── The whole point: the STABLE prefix is stable ──────────────────────


def test_stable_prefix_identical_across_volatile_change(tmp_path):
    """Two builds that differ ONLY in volatile content must yield a
    byte-identical system prefix.

    Build A: low context (no warning). Build B: high context (CONTEXT WARNING
    fires) AND the recent log grows. Both must produce the SAME system prompt —
    that's what makes #1073's cached prefix hit.
    """
    loop = _operator_loop(tmp_path)

    # Build A — minimal volatile content.
    loop._chat_messages = [{"role": "user", "content": "hi"}]
    prefix_a = _chat_prompt(loop)

    # Build B — change ONLY volatile inputs: blow up the context (→ warning)
    # and append a NEW recent log entry (→ different ## Recent slice).
    loop._chat_messages = [{"role": "user", "content": "x " * 400}]
    loop.workspace.append_memory("brand-new volatile fact " * 5)
    prefix_b = _chat_prompt(loop)

    assert prefix_a == prefix_b, "stable prefix drifted across volatile change"
    # Sanity: the volatile suffixes DID differ (so we actually changed volatile
    # inputs, not nothing).
    # Rebuild A's suffix to compare against B's stash.
    assert "CONTEXT WARNING" in loop._volatile_prompt_suffix  # B had a warning


# ── Task-path builder ─────────────────────────────────────────────────


def test_task_builder_relocates_runtime_and_recent(tmp_path):
    loop = _operator_loop(tmp_path)
    assignment = TaskAssignment(
        workflow_id="wf", step_id="s", task_type="research", input_data={},
    )
    introspect = {"permissions": {}, "budget": None, "fleet": [{"id": "a"}]}
    prompt = loop._build_system_prompt(assignment, introspect_data=introspect)
    # Volatile out of the system prompt …
    assert _RECENT_MARKER not in prompt
    assert "## Runtime Context" not in prompt
    # … stashed for re-injection.
    assert _RECENT_BODY in loop._volatile_prompt_suffix
    assert "## Runtime Context" in loop._volatile_prompt_suffix
    # Stable identity content stays.
    assert "Operating Rules" in prompt


# ── Workspace memory-injection split ──────────────────────────────────


def test_get_memory_injection_head_only(tmp_path):
    ws = WorkspaceManager(workspace_dir=str(tmp_path / "ws"))
    (ws.root / "MEMORY.md").write_text(_MEMORY_WITH_RECENT)
    full = ws.get_memory_injection(include_recent=True)
    head_only = ws.get_memory_injection(include_recent=False)
    assert _RECENT_MARKER in full
    assert _RECENT_MARKER not in head_only
    assert _HEAD in head_only
    # The relocated slice is exactly what get_recent_memory_slice returns.
    slice_ = ws.get_recent_memory_slice()
    assert slice_.startswith(_RECENT_MARKER)
    assert _RECENT_BODY in slice_


def test_bootstrap_cache_slots_independent(tmp_path):
    """The head-only bootstrap variant is memoized separately from the default
    (recent-inclusive) one — calling one must not poison the other."""
    ws = WorkspaceManager(workspace_dir=str(tmp_path / "ws"))
    (ws.root / "MEMORY.md").write_text(_MEMORY_WITH_RECENT)
    with_recent = ws.get_bootstrap_content(include_recent=True)
    head_only = ws.get_bootstrap_content(include_recent=True) is not None and \
        ws.get_bootstrap_content(include_recent=False)
    assert _RECENT_MARKER in with_recent
    assert _RECENT_MARKER not in head_only
    # Re-fetch from cache — still correct (no cross-contamination).
    assert ws.get_bootstrap_content(include_recent=True) == with_recent
    assert ws.get_bootstrap_content(include_recent=False) == head_only


def test_bootstrap_cache_both_slots_detect_external_mtime_change(tmp_path):
    """Regression: an mtime-triggered rebuild of ONE variant must not mark the
    OTHER variant fresh. Each slot owns its own mtime snapshot, so an external
    write to MEMORY.md (one that does NOT explicitly clear the caches) is picked
    up by BOTH variants on their next call — neither serves stale content."""
    import os

    ws = WorkspaceManager(workspace_dir=str(tmp_path / "ws"))
    memory = ws.root / "MEMORY.md"
    memory.write_text(_MEMORY_WITH_RECENT)

    # Warm BOTH cache slots against the original content.
    assert _HEAD in ws.get_bootstrap_content(include_recent=True)
    assert _HEAD in ws.get_bootstrap_content(include_recent=False)

    # Simulate an EXTERNAL change: write directly to the file and bump its
    # mtime, bypassing append_memory/write_compiled_memory/update_file (the
    # paths that explicitly clear the caches). Only the lazy mtime check applies.
    new_head = "The user now prefers verbose answers. The live fleet is sales."
    memory.write_text(
        "# Long-Term Memory\n\n"
        "<!-- compiled:begin -->\n"
        f"{new_head}\n"
        "<!-- compiled:end -->\n\n"
        "## 2026-06-09T11:00 flush\n\n"
        "- learned: the user switched preferences\n"
    )
    future = memory.stat().st_mtime + 10
    os.utime(memory, (future, future))

    # BOTH variants must observe the new content (no stale slot).
    assert new_head in ws.get_bootstrap_content(include_recent=True)
    assert new_head in ws.get_bootstrap_content(include_recent=False)
    assert _HEAD not in ws.get_bootstrap_content(include_recent=True)
    assert _HEAD not in ws.get_bootstrap_content(include_recent=False)


def test_no_recent_slice_when_log_empty(tmp_path):
    """A head-only MEMORY (no log) stashes nothing extra for the recent slice
    and the head is still present in the system prompt."""
    ws = WorkspaceManager(workspace_dir=str(tmp_path / "ws"))
    (ws.root / "MEMORY.md").write_text(_MEMORY_HEAD_ONLY)
    assert ws.get_recent_memory_slice() == ""
    loop = _make_loop()
    loop.workspace = ws
    loop._is_operator = False
    loop.context_manager = None
    prompt = loop._build_chat_system_prompt(
        introspect_data=None,
    )
    assert _HEAD in prompt
    assert _RECENT_MARKER not in loop._volatile_prompt_suffix
