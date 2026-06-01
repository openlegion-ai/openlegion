#!/usr/bin/env python3
"""Print the unique set of source URLs, order-preserving.

Bundled with the competitor-research skill; the agent runs it via its existing
shell tool (it is NOT imported into the runtime). Usage:

    python dedupe_sources.py <url> [<url> ...]
"""

import sys


def main(argv: list[str]) -> int:
    seen: set[str] = set()
    for url in argv:
        u = url.strip()
        if u and u not in seen:
            seen.add(u)
            print(u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
