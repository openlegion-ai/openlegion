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

# ── Sudo guard ───────────────────────────────────────────────
# Running the whole script as root causes venv/config ownership issues.
# Only Docker may need elevated access — handled separately below.

if [ "$(id -u)" -eq 0 ]; then
    echo -e "  ${YELLOW}!${NC} Running as root. The virtual environment and config files"
    echo -e "  ${YELLOW} ${NC} will be owned by root, which causes permission issues later."
    echo -e "  ${YELLOW} ${NC} Run without sudo instead:"
    echo -e "  ${YELLOW} ${NC}   ./install.sh"
    echo ""
    if [ "$(uname)" = "Linux" ]; then
        echo -e "  ${YELLOW} ${NC} If Docker requires sudo, add your user to the docker group:"
        echo -e "  ${YELLOW} ${NC}   sudo usermod -aG docker \$USER && newgrp docker"
        echo ""
    fi
    read -rp "  Continue anyway? [y/N] " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo ""
fi

# ── Check all prerequisites ──────────────────────────────────
# Collect ALL failures before exiting so users can fix everything in one pass.

ERRORS=()

# Python
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
    ERRORS+=("Python 3.10+ is required but not found.
    Install it:
      macOS:   brew install python@3
      Ubuntu:  sudo apt install python3
      Windows: https://python.org/downloads
      Other:   https://github.com/pyenv/pyenv")
else
    info "Python $version ($PYTHON)"

    # pip (only check if Python was found)
    if ! "$PYTHON" -m pip --version &>/dev/null; then
        ERRORS+=("pip is not installed.
    Install it:
      macOS:   python3 -m ensurepip --upgrade
      Ubuntu:  sudo apt install python3-pip
      Other:   https://pip.pypa.io/en/stable/installation/")
    else
        info "pip available"
    fi

    # venv module (only check if Python was found)
    if ! "$PYTHON" -m venv --help &>/dev/null 2>&1; then
        if [ -f /etc/debian_version ]; then
            ERRORS+=("python3-venv is not installed (required on Debian/Ubuntu).
    Install it:
      sudo apt install python3-venv")
        else
            ERRORS+=("Python venv module is not available.
    Install it for your platform and retry.")
        fi
    fi
fi

# Docker
if ! command -v docker &>/dev/null; then
    ERRORS+=("Docker is not installed.
    Install it:
      macOS/Windows: https://docker.com/products/docker-desktop
      Linux:         sudo apt install docker.io")
elif ! docker info &>/dev/null 2>&1; then
    if docker info 2>&1 | grep -qi "permission denied"; then
        if [ "$(uname)" = "Linux" ]; then
            ERRORS+=("Docker permission denied. Add your user to the docker group:
      sudo usermod -aG docker \$USER && newgrp docker
    Then run this installer again (without sudo).")
        else
            ERRORS+=("Docker permission denied. Make sure Docker Desktop is running.")
        fi
    else
        ERRORS+=("Docker is installed but not running.
    Start it:
      macOS/Windows: Open Docker Desktop
      Linux:         sudo systemctl start docker")
    fi
else
    info "Docker is running"
fi

# Git
if ! command -v git &>/dev/null; then
    ERRORS+=("Git is not installed.
    Install it:
      macOS:   xcode-select --install
      Ubuntu:  sudo apt install git
      Windows: https://git-scm.com")
else
    info "Git available"
fi

# Report all failures at once
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${RED}Found ${#ERRORS[@]} issue(s):${NC}"
    echo ""
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}✗${NC} $err"
        echo ""
    done
    echo -e "  Fix the above and run ${BOLD}./install.sh${NC} again."
    exit 1
fi

# ── Detect platform for venv paths ───────────────────────────
# Windows (Git Bash/MSYS2) uses Scripts/, Unix uses bin/

if [ -d ".venv/Scripts" ] || [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]]; then
    VENV_BIN_DIR=".venv/Scripts"
else
    VENV_BIN_DIR=".venv/bin"
fi

# ── Create virtual environment ────────────────────────────────

echo ""

# If .venv exists but is broken (missing python or pip), recreate it
VENV_BROKEN=false
if [ -d ".venv" ]; then
    if [ ! -x "$VENV_BIN_DIR/python" ] && [ ! -x "$VENV_BIN_DIR/python.exe" ]; then
        VENV_BROKEN=true
    elif [ ! -x "$VENV_BIN_DIR/pip" ] && [ ! -x "$VENV_BIN_DIR/pip.exe" ]; then
        VENV_BROKEN=true
    fi
fi
if [ "$VENV_BROKEN" = true ]; then
    warn "Existing .venv is broken — recreating..."
    rm -rf .venv 2>/dev/null || true
    # If rm failed (e.g. root-owned .venv), try with sudo
    if [ -d ".venv" ]; then
        warn ".venv is owned by another user — need sudo to remove it"
        sudo rm -rf .venv || fail "Cannot remove broken .venv. Delete it manually:
      sudo rm -rf .venv"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv .venv
    info "Virtual environment created"
else
    info "Virtual environment exists"
fi

# ── Install dependencies ──────────────────────────────────────

echo ""
echo -e "  ${BOLD}Installing dependencies...${NC}"
echo -e "  ${YELLOW}First install takes 2-3 minutes (downloads ~70 packages).${NC}"
echo -e "  ${YELLOW}Subsequent installs are fast (cached).${NC}"
echo ""

"$VENV_BIN_DIR/pip" install -e ".[dev,channels]" 2>&1 | while IFS= read -r line; do
    # Show download/install progress, skip noise
    case "$line" in
        *Collecting*|*Downloading*|*Installing*|*Building*|*Successfully*)
            echo "  $line"
            ;;
    esac
done

# Check pip actually succeeded (pipe can mask failures with set -e)
if [ ! -x "$VENV_BIN_DIR/openlegion" ] && [ ! -x "$VENV_BIN_DIR/openlegion.exe" ]; then
    # Retry without filtering to show the real error
    echo ""
    warn "Install may have failed. Retrying with full output..."
    echo ""
    "$VENV_BIN_DIR/pip" install -e ".[dev,channels]"
fi

echo ""
info "OpenLegion installed"

# ── Global CLI access ────────────────────────────────────────

echo ""

# Resolve the real user's home when running under sudo
if [ -n "$SUDO_USER" ]; then
    # getent works on Linux; fall back to HOME-based lookup or ~user expansion
    REAL_HOME=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)
    if [ -z "$REAL_HOME" ]; then
        # macOS / systems without getent
        REAL_HOME=$(eval echo "~$SUDO_USER")
    fi
else
    REAL_HOME="$HOME"
fi

LINK_DIR="$REAL_HOME/.local/bin"
VENV_BIN="$(cd "$(dirname "$0")" && pwd)/$VENV_BIN_DIR/openlegion"
PATH_UPDATED=""

if [ -f "$VENV_BIN" ]; then
    mkdir -p "$LINK_DIR"
    ln -sf "$VENV_BIN" "$LINK_DIR/openlegion"

    # Fix ownership if running as root
    if [ -n "$SUDO_USER" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$LINK_DIR" 2>/dev/null || true
    fi

    if echo "$PATH" | tr ':' '\n' | grep -qx "$LINK_DIR"; then
        info "openlegion available globally"
    else
        # Auto-add to shell rc file (same approach as rustup, nvm, etc.)
        SHELL_NAME="$(basename "${SHELL:-/bin/bash}")"
        PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
        RC_FILE=""
        if [ "$SHELL_NAME" = "zsh" ]; then
            RC_FILE="$REAL_HOME/.zshrc"
        elif [ "$SHELL_NAME" = "fish" ]; then
            RC_FILE=""  # fish uses fish_add_path, handled separately
        else
            RC_FILE="$REAL_HOME/.bashrc"
        fi

        if [ -n "$RC_FILE" ]; then
            # Only append if not already present
            if ! grep -qF '/.local/bin' "$RC_FILE" 2>/dev/null; then
                echo "" >> "$RC_FILE"
                echo "# Added by OpenLegion installer" >> "$RC_FILE"
                echo "$PATH_LINE" >> "$RC_FILE"
                # Fix ownership of rc file if running as root
                if [ -n "$SUDO_USER" ]; then
                    chown "$SUDO_USER:$SUDO_USER" "$RC_FILE" 2>/dev/null || true
                fi
                info "Added ~/.local/bin to PATH in $(basename "$RC_FILE")"
            fi
        elif [ "$SHELL_NAME" = "fish" ]; then
            fish -c "fish_add_path $LINK_DIR" 2>/dev/null || true
            info "Added ~/.local/bin to fish PATH"
        fi

        # Update PATH for the current script session
        export PATH="$LINK_DIR:$PATH"
        PATH_UPDATED="yes"
        info "openlegion available globally"
    fi
else
    warn "Could not create global symlink — run from the project directory"
fi

# ── Done ──────────────────────────────────────────────────────

echo ""
echo -e "  ${GREEN}Ready!${NC} Next steps:"
echo ""
if [ -n "$PATH_UPDATED" ]; then
    if [ -n "$RC_FILE" ]; then
        echo "    source ~/$(basename "$RC_FILE")   # reload PATH in this terminal"
    else
        echo "    Open a new terminal to pick up the PATH change, then:"
    fi
fi
echo "    openlegion setup             # configure API key + agents"
echo "    openlegion start             # launch and start chatting"
echo ""
