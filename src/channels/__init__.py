"""Channel adapters for messaging platforms."""

import re

# Shared regex for @agent routing across all channel types that support it.
AT_MENTION_RE = re.compile(r"^@(\w+)\s+(.+)$", re.DOTALL)
