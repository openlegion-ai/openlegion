"""Built-in tools available to every agent.

These are auto-discovered by SkillRegistry alongside custom skills.
They provide core capabilities: shell execution, file I/O, HTTP, and browser.
"""

import re

# Shared pattern for credential handle syntax: $CRED{name}
CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")
