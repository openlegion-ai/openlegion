"""Memory tools: search and append to persistent workspace memory.

The workspace_manager and memory_store are injected at call time via the
tools system (same pattern as mesh_client injection).
"""

from __future__ import annotations

from src.agent.tools import tool
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.memory_tool")


async def _search_with_fallback(memory_store, query: str, top_k: int):
    """Try hierarchical search, fall back to flat search on failure."""
    try:
        return await memory_store.search_hierarchical(query, top_k=top_k)
    except Exception as e:
        logger.debug("Hierarchical search failed, falling back to flat: %s", e)
    try:
        return await memory_store.search(query, top_k=top_k)
    except Exception as e2:
        logger.warning("Flat memory search also failed: %s", e2)
        return None


def _parse_fact(content: str) -> tuple[str, str]:
    """Parse free-text content into a (key, value) pair for structured storage.

    Heuristic: if content has a colon or dash separator, split there.
    Otherwise use the first ~60 chars as the key and full text as value.
    """
    content = content.strip()
    # Try "key: value" format
    for sep in [":", " - ", " — "]:
        if sep in content:
            parts = content.split(sep, 1)
            if len(parts[0]) <= 80 and parts[1].strip():
                return parts[0].strip(), parts[1].strip()
    # Fallback: truncate for key, full text for value
    key = content[:60].rstrip()
    if len(content) > 60:
        key = key.rsplit(" ", 1)[0] if " " in key else key
    return key, content


@tool(
    name="memory_search",
    description=(
        "Search your long-term memory. By default searches both workspace files "
        "and your structured fact database. Provide a category to search only "
        "the fact database filtered to that category. Use this to recall facts, "
        "preferences, decisions, or past events before answering questions."
    ),
    parameters={
        "query": {"type": "string", "description": "What to search for"},
        "category": {
            "type": "string",
            "description": (
                "Optional: filter to a fact category (e.g. 'user_preferences', "
                "'decisions'). When set, searches only the structured fact database."
            ),
            "default": "",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 5)",
            "default": 5,
        },
    },
)
async def memory_search(
    query: str, category: str = "", max_results: int = 5,
    *, workspace_manager=None, memory_store=None,
) -> dict:
    """Search workspace memory files and structured fact database."""
    results = []

    # Category-filtered search: only the structured fact DB
    if category:
        if memory_store is None:
            return {"error": "No memory_store available for category search", "results": []}
        # Over-fetch when filtering by category since post-fetch filtering
        # may discard many results
        fetch_k = max_results * 3
        facts = await _search_with_fallback(memory_store, query, fetch_k)
        if facts is None:
            return {"error": "Memory search failed", "results": []}
        for fact in facts:
            if fact.category.lower() != category.lower():
                continue
            results.append({
                "key": fact.key,
                "value": fact.value,
                "category": fact.category,
                "confidence": fact.confidence,
                "access_count": fact.access_count,
                "source": "memory_db",
            })
        return {"results": results, "count": len(results)}

    # Default: search both workspace and DB
    if workspace_manager is not None:
        ws_hits = workspace_manager.search(query, max_results=max_results)
        for hit in ws_hits:
            hit["source"] = "workspace"
            results.append(hit)

    if memory_store is not None:
        db_facts = await _search_with_fallback(memory_store, query, max_results)
        if db_facts:
            for fact in db_facts:
                results.append({
                    "key": fact.key,
                    "value": fact.value,
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "source": "memory_db",
                })

    if not results and workspace_manager is None and memory_store is None:
        return {"error": "No memory backends available", "results": []}

    return {"results": results, "count": len(results)}


def _gather_evidence(query: str, max_facts: int, workspace_manager, memory_store, db_facts):
    """Build a numbered evidence list from DB facts + workspace hits.

    The numbers are the citation anchors the LLM cites inline as [n].
    Each item is a dict carrying enough metadata to return as a citation.
    """
    evidence: list[dict] = []

    if db_facts:
        for fact in db_facts:
            evidence.append({
                "source": "memory_db",
                "key": fact.key,
                "value": fact.value,
                "category": fact.category,
                "confidence": fact.confidence,
            })

    if workspace_manager is not None:
        try:
            ws_hits = workspace_manager.search(query, max_results=max_facts)
        except Exception as e:
            logger.warning("Workspace search failed in memory_think: %s", e)
            ws_hits = []
        for hit in ws_hits:
            evidence.append({
                "source": "workspace",
                "file": hit.get("file"),
                "snippet": hit.get("snippet"),
                "score": hit.get("score"),
            })

    return evidence


def _render_evidence(evidence: list[dict]) -> str:
    """Render the numbered evidence list for the LLM prompt.

    Facts come from our own DB / workspace, so this is light; we still trim
    whitespace to keep the prompt compact.
    """
    lines = []
    for i, item in enumerate(evidence, start=1):
        if item["source"] == "memory_db":
            lines.append(f"[{i}] (memory_db) {item['key']}: {item['value']}".strip())
        else:
            snippet = (item.get("snippet") or "").strip()
            lines.append(f"[{i}] (workspace:{item.get('file')}) {snippet}")
    return "\n".join(lines)


_THINK_SYSTEM_PROMPT = (
    "You are synthesizing an answer strictly from a numbered list of memory "
    "evidence. Rules: (1) Answer ONLY using the numbered evidence below — do "
    "not invent facts. (2) Cite each claim inline with its source number, e.g. "
    "[1] or [2][3]. (3) Be concise: under 150 words. (4) End with a final line "
    "beginning exactly 'Unknown / not in memory:' that states what the "
    "question asks for which the evidence does NOT cover (write 'nothing "
    "obvious' if the evidence fully covers it)."
)


@tool(
    name="memory_think",
    description=(
        "Synthesize an answer from your long-term memory. Unlike memory_search "
        "(which returns raw hits), this retrieves the most relevant facts and "
        "produces a SHORT cited answer plus an explicit note about what your "
        "memory does NOT yet cover. Use when you want a reasoned recall rather "
        "than a list of matches."
    ),
    parameters={
        "query": {"type": "string", "description": "The question to answer from memory"},
        "max_facts": {
            "type": "integer",
            "description": "Maximum facts to retrieve as evidence (default 8)",
            "default": 8,
        },
    },
)
async def memory_think(
    query: str, max_facts: int = 8,
    *, workspace_manager=None, memory_store=None, mesh_client=None,
) -> dict:
    """Retrieve relevant memory and synthesize a cited answer with a gap note."""
    if memory_store is None and workspace_manager is None:
        return {"error": "No memory backends available"}

    # 1. Retrieve evidence from both backends.
    db_facts = None
    if memory_store is not None:
        db_facts = await _search_with_fallback(memory_store, query, max_facts)
    evidence = _gather_evidence(query, max_facts, workspace_manager, memory_store, db_facts)

    if not evidence:
        return {"answer": "", "note": "No relevant memory found.", "evidence_count": 0}

    # 2. Resolve the parent LLM via the established registry (tools are not
    #    injected an LLM directly — same pattern as spawn_subagent).
    from src.agent.builtins.subagent_tool import _get_parent_llm

    agent_id = getattr(mesh_client, "agent_id", None)
    llm = _get_parent_llm(agent_id) if agent_id else None

    # 3. Graceful degradation: no LLM → return raw evidence without synthesis.
    if llm is None:
        return {
            "answer": None,
            "note": "Synthesis unavailable; returning raw matches.",
            "results": evidence,
            "evidence_count": len(evidence),
        }

    # 4. Synthesize. Any failure falls back to raw evidence — never crash.
    rendered = _render_evidence(evidence)
    user_msg = f"Question: {query}\n\nNumbered evidence:\n{rendered}"
    try:
        resp = await llm.chat(
            system=_THINK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
            temperature=0.2,
        )
        answer = (resp.content or "").strip()
    except Exception as e:
        logger.warning("memory_think synthesis failed, returning raw matches: %s", e)
        return {
            "answer": None,
            "note": "Synthesis unavailable; returning raw matches.",
            "results": evidence,
            "evidence_count": len(evidence),
        }

    return {"answer": answer, "citations": evidence, "evidence_count": len(evidence)}


@tool(
    name="memory_save",
    description=(
        "Save a fact or note to your memory database, searchable later via "
        "memory_search. Use for discrete facts: user preferences, decisions, "
        "findings, contact info. Do NOT use this for your operating instructions "
        "or identity — use update_workspace for that (SOUL.md, INSTRUCTIONS.md, "
        "USER.md)."
    ),
    parameters={
        "content": {
            "type": "string",
            "description": "The fact or note to save",
        },
    },
)
async def memory_save(content: str, *, workspace_manager=None, memory_store=None) -> dict:
    """Save a fact to both the daily log and structured memory DB."""
    saved_workspace = False
    saved_db = False

    # 1. Workspace daily log (human-readable markdown)
    if workspace_manager is not None:
        workspace_manager.append_daily_log(content)
        saved_workspace = True

    # 2. Structured memory DB (searchable via memory_search)
    if memory_store is not None:
        try:
            # Parse content into key/value — use first sentence or clause as key
            key, value = _parse_fact(content)
            await memory_store.store_fact(
                key=key, value=value, source="memory_save", confidence=0.9,
            )
            saved_db = True
        except Exception as e:
            logger.warning("Failed to store fact in memory DB: %s", e)

    if not saved_workspace and not saved_db:
        return {"error": "No memory backends available"}

    return {"saved": True, "saved_workspace": saved_workspace, "saved_db": saved_db, "content": content}
