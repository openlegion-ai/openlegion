#!/usr/bin/env bash
set -e

# OpenLegion installer — checks dependencies and sets up the project.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}!${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; exit 1; }

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
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3.10+ is required but not found.
    Install it:
      macOS:   brew install python@3
      Ubuntu:  sudo apt install python3
      Windows: https://python.org/downloads
      Other:   https://github.com/pyenv/pyenv"
fi
info "Python $version ($PYTHON)"

# ── Check pip ─────────────────────────────────────────────────

if ! "$PYTHON" -m pip --version &>/dev/null; then
    fail "pip is not installed.
    Install it:
      macOS:   python3 -m ensurepip --upgrade
      Ubuntu:  sudo apt install python3-pip
      Other:   https://pip.pypa.io/en/stable/installation/"
fi
info "pip available"

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

# ── Create virtual environment ────────────────────────────────

echo ""
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv .venv
    info "Virtual environment created"
else
    info "Virtual environment exists"
fi

source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────

echo ""
echo -e "  ${BOLD}Installing dependencies...${NC}"
echo -e "  ${YELLOW}First install takes 2-3 minutes (downloads ~70 packages).${NC}"
echo -e "  ${YELLOW}Subsequent installs are fast (cached).${NC}"
echo ""

pip install -e ".[dev]" 2>&1 | while IFS= read -r line; do
    # Show download/install progress, skip noise
    case "$line" in
        *Collecting*|*Downloading*|*Installing*|*Building*|*Successfully*)
            echo "  $line"
            ;;
    esac
done

echo ""
info "OpenLegion installed"

# ── Done ──────────────────────────────────────────────────────

echo ""
echo -e "  ${GREEN}Ready!${NC} Next steps:"
echo ""
echo "    source .venv/bin/activate    # activate the environment"
echo "    openlegion setup             # configure API key + agents"
echo "    openlegion start             # launch and start chatting"
echo ""
