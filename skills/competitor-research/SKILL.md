---
name: competitor-research
description: Research a competitor across the web and produce a cited, structured brief.
version: 1.0.0
license: MIT
metadata:
  requires_toolsets: [web_search, http_request]
---
# When to Use

Use this skill when asked to "research", "look into", or "produce a brief on"
a specific competitor or company. It assumes you already have `web_search`
and `http_request`; it adds no new capability — only a procedure.

# Procedure

1. Establish the basics: official site, what they sell, who they sell to.
   Run `web_search` for the company name plus "pricing", "product", "funding".
2. Gather 5–10 distinct sources. Prefer primary sources (the company's own
   pages, filings, docs) over secondary commentary. Record the URL for each
   claim as you go — every fact in the brief must be traceable.
3. Cross-check anything surprising against a second source before stating it.
   If two sources conflict, say so in the brief rather than picking one.
4. De-duplicate your source URLs before writing — run
   `python ${SKILL_DIR}/scripts/dedupe_sources.py url1 url2 ...`; it prints the
   unique set so you don't cite the same page twice.
5. Assemble the brief using the structure in
   `references/brief-template.md` (read it with skill_view if unsure).

# Pitfalls / Verification

- Do not state pricing, headcount, or funding numbers without a dated source.
- Marketing copy is not a source for capability claims — verify in docs.
- Before reporting done, confirm every section of the template is filled and
  every non-obvious claim has a citation.
