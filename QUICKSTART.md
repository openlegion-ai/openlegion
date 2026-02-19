# Quick Start Guide

Step-by-step setup for macOS, Linux, and Windows. From zero to running agents in under 10 minutes.

---

## Step 1: Install Python 3.12+

### macOS

**Option A — Homebrew (recommended):**

```bash
brew install python@3.12
```

**Option B — Installer:**

Download from [python.org/downloads](https://www.python.org/downloads/). Run the `.pkg` installer. Check "Add Python to PATH" during install.

**Verify:**

```bash
python3 --version
# Python 3.12.x or higher
```

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

### Linux (Fedora / RHEL)

```bash
sudo dnf install python3 python3-pip git
```

**Verify:**

```bash
python3 --version
# Python 3.12.x or higher
```

> If your distro ships Python 3.11 or older, use [pyenv](https://github.com/pyenv/pyenv) or the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) (Ubuntu) to install 3.12+.

### Windows

1. Download Python 3.12+ from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. **Check "Add python.exe to PATH"** at the bottom of the first screen — this is critical
4. Click "Install Now"

**Verify (open PowerShell or Command Prompt):**

```
python --version
# Python 3.12.x or higher
```

> On Windows, the commands are `python` and `pip` (not `python3` / `pip3`).

---

## Step 2: Install Docker

Docker runs each agent in an isolated container. It must be installed **and running** before you continue.

### macOS

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Open the `.dmg` and drag Docker to Applications
3. Launch **Docker Desktop** from Applications
4. Wait for the whale icon in the menu bar to stop animating (this means the daemon is ready)

**Verify:**

```bash
docker info
# Should print "Server: Docker Desktop" and version info
# If you see "Cannot connect to the Docker daemon" — Docker Desktop isn't running yet
```

### Linux (Ubuntu / Debian)

```bash
# Install Docker Engine
sudo apt update
sudo apt install docker.io

# Start Docker and enable on boot
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (avoids needing sudo)
sudo usermod -aG docker $USER
```

**Log out and log back in** for the group change to take effect.

**Verify:**

```bash
docker info
# Should print server info without permission errors
```

### Linux (Fedora / RHEL)

```bash
sudo dnf install docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**Log out and log back in** for the group change to take effect.

### Windows

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer (requires admin rights)
3. **Enable WSL 2 backend** when prompted (recommended)
4. Restart your computer if prompted
5. Launch **Docker Desktop** from the Start menu
6. Wait for "Docker Desktop is running" in the system tray

**Verify (PowerShell):**

```
docker info
```

> **Windows troubleshooting:** If Docker won't start, ensure "Virtual Machine Platform" and "Windows Subsystem for Linux" are enabled in Windows Features. Docker Desktop will prompt you to install these if missing.

---

## Step 3: Get an LLM API Key

You need at least one API key from a supported LLM provider. Pick one:

| Provider | Get a key | Notes |
|---|---|---|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | GPT-4o-mini is cheap and fast for testing |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) | Claude is recommended for browser automation |
| **Groq** | [console.groq.com](https://console.groq.com/) | Free tier available, very fast inference |

Keep the key handy — the setup wizard will ask for it.

> **100+ providers supported** via LiteLLM. See [litellm.ai](https://litellm.ai) for the full list. The three above are the most common starting points.

---

## Step 4: Clone and Install OpenLegion

### macOS / Linux

```bash
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python3 -m venv .venv
source .venv/bin/activate
pip3 install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> If PowerShell blocks the activation script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

### Windows (Command Prompt)

```cmd
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

**Verify the install worked:**

```bash
openlegion --help
# Should print available commands: setup, start, stop, status, agent, chat
```

---

## Step 5: Run Setup

**Make sure Docker is running first** (Docker Desktop open on macOS/Windows, `sudo systemctl start docker` on Linux).

```bash
openlegion setup
```

The wizard walks you through:

1. **LLM provider** — pick OpenAI, Anthropic, or Groq and paste your API key
2. **Project description** — one line about what your agents will do
3. **Team template** — starter (1 agent), sales (3 agents), devteam (3 agents), or content (3 agents)
4. **Docker image build** — builds the agent container image (takes 1-2 minutes the first time)

> **"Docker is not running" error?** Open Docker Desktop (macOS/Windows) or run `sudo systemctl start docker` (Linux) and try again.

---

## Step 6: Start Your Agents

```bash
openlegion start
```

You'll see containers starting with their IDs, then the interactive REPL appears. You're in.

### Talk to an agent

```
@researcher Research the latest developments in AI agent frameworks.
```

### Check costs

```
/costs
```

### See all agents

```
/agents
```

### Broadcast to the fleet

```
/broadcast What is your role and current status?
```

### Stop everything

```
/quit
```

---

## What's Next

```bash
# Add another agent
openlegion agent add analyst

# Run in the background
openlegion start -d

# Connect from another terminal
openlegion chat researcher

# Stop all agents
openlegion stop
```

### Connect Telegram (optional)

Add Telegram channel config to `config/mesh.yaml` with your bot token. When you run `openlegion start`, a pairing code will appear — send it to your bot in Telegram to link your account.

### Connect Discord (optional)

Same pattern — add Discord bot token to your config. The bot joins your server and routes messages to agents.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: openlegion` | Make sure your virtual environment is activated: `source .venv/bin/activate` |
| `Docker is not running` | Open Docker Desktop (macOS/Windows) or `sudo systemctl start docker` (Linux) |
| `permission denied` on Docker (Linux) | Add yourself to the docker group: `sudo usermod -aG docker $USER` then log out and back in |
| `python3: command not found` | Install Python 3.12+ — see Step 1 above |
| `pip3 install` fails with permission error | Make sure you're inside the virtual environment (you should see `(.venv)` in your prompt) |
| `ModuleNotFoundError` when running | You're outside the venv. Run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows) |
| Docker image build is slow | First build downloads base image + Chromium (~2 min). Subsequent builds use cache and are fast. |
| Agent not responding | Check `openlegion status`. If unhealthy, the health monitor will auto-restart it. Check your API key has credits. |
| `openlegion setup` hangs at "Building Docker image" | This is normal on first run — it's downloading and building. Wait 2-3 minutes. |
| PowerShell won't activate venv (Windows) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
