"""Browser-service redaction — backward-compat shim over :mod:`src.shared.redaction`.

Historically carried its own copy of the secret-pattern list. Phase 1.3
consolidated patterns into :mod:`src.shared.redaction` so there's one source
of truth, URL-aware redaction ships here too, and updates don't risk the two
copies drifting.

Keeps the :class:`CredentialRedactor` API so existing call sites in
``src/browser/service.py`` continue to work unchanged.
"""

from __future__ import annotations

from src.shared.redaction import (
    _REDACT_PATTERNS,  # noqa: F401  — re-export for old imports
    deep_redact,
    redact_string,
)


class CredentialRedactor:
    """Per-agent credential redaction state.

    Thin shim: ``agent_id`` is accepted for future per-agent allowlisting but
    not yet used. The underlying pattern set is global.
    """

    def redact(self, agent_id: str, text: str) -> str:
        """Apply pattern-based redaction to ``text``."""
        return redact_string(text)

    def deep_redact(self, agent_id: str, obj):
        """Recursively redact credential values from a JSON-shaped structure."""
        return deep_redact(obj)
