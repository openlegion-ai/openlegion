#!/usr/bin/env bash
set -e

# OpenLegion installer — checks dependencies and sets up the project.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }

echo ""
echo "  OpenLegion Installer"
echo "  ────────────────────"
echo ""

# ── Check Python ──────────────────────────────────────────────

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3.12+ is required but not found.
    Install it:
      macOS:   brew install python@3.12
      Ubuntu:  sudo apt install python3
      Windows: https://python.org/downloads
      Other:   https://github.com/pyenv/pyenv"
fi
info "Python $version ($PYTHON)"

# ── Check Docker ──────────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    fail "Docker is not installed.
    Install it:
      macOS/Windows: https://docker.com/products/docker-desktop
      Linux:         sudo apt install docker.io"
fi

if ! docker info &>/dev/null 2>&1; then
    fail "Docker is installed but not running.
    Start it:
      macOS/Windows: Open Docker Desktop
      Linux:         sudo systemctl start docker"
fi
info "Docker is running"

# ── Check Git ─────────────────────────────────────────────────

if ! command -v git &>/dev/null; then
    fail "Git is not installed.
    Install it:
      macOS:   xcode-select --install
      Ubuntu:  sudo apt install git
      Windows: https://git-scm.com"
fi
info "Git available"

# ── Create virtual environment and install ────────────────────

if [ ! -d ".venv" ]; then
    echo ""
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

echo "  Installing dependencies (this may take a minute)..."
pip install -q -e ".[dev]" 2>&1 | tail -1

info "OpenLegion installed"

# ── Done ──────────────────────────────────────────────────────

echo ""
echo -e "  ${GREEN}Ready!${NC} Next steps:"
echo ""
echo "    source .venv/bin/activate    # activate the environment"
echo "    openlegion setup             # configure API key + agents"
echo "    openlegion start             # launch and start chatting"
echo ""
