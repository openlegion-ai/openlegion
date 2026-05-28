"""Built-in tools available to every agent.

These are auto-discovered by SkillRegistry alongside custom skills.
They provide core capabilities: shell execution, file I/O, HTTP, and browser.
"""

# Re-exported from src.shared.types — that's the canonical location for
# the credential handle syntax shared by agent-side (http_tool) and
# mesh-side (host.credentials, host.runtime) resolvers. Re-export kept
# here so existing ``from src.agent.builtins import CRED_HANDLE_RE``
# callers in http_tool.py don't need to be touched.
from src.shared.types import CRED_HANDLE_RE

__all__ = ["CRED_HANDLE_RE"]
