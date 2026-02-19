# Quick Start Guide

Get OpenLegion running on macOS, Linux, or Windows.

**You need three things:** Python 3.12+, Docker, and an LLM API key.

---

## 1. Install Python 3.12+

<details>
<summary><strong>macOS</strong></summary>

```bash
# Option A: Homebrew (recommended)
brew install python@3.12

# Option B: Download installer from https://python.org/downloads
```

Verify:

```bash
python3 --version   # should print 3.12 or higher
```

</details>

<details>
<summary><strong>Linux (Ubuntu / Debian)</strong></summary>

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

Verify:

```bash
python3 --version   # should print 3.12 or higher
```

> Distro ships 3.11 or older? Use [pyenv](https://github.com/pyenv/pyenv) or the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) to get 3.12+.

</details>

<details>
<summary><strong>Linux (Fedora / RHEL)</strong></summary>

```bash
sudo dnf install python3 python3-pip git
```

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Download Python 3.12+ from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. **Check "Add python.exe to PATH"** — this is critical
4. Click "Install Now"

Verify (PowerShell):

```
python --version
```

> On Windows, use `python` and `pip` (not `python3` / `pip3`).

</details>

---

## 2. Install and Start Docker

Docker isolates each agent in its own container. It must be **running** before setup.

<details>
<summary><strong>macOS</strong></summary>

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Open the `.dmg` and drag Docker to Applications
3. Launch Docker Desktop from Applications
4. **Skip the sign-in** — you do not need a Docker account. Click "Continue without signing in" or close the login prompt.
5. Wait for the whale icon in the menu bar to stop animating (daemon is ready)

Verify:

```bash
docker info   # should print server info
```

> **"Cannot connect to the Docker daemon"** = Docker Desktop isn't open yet. Launch it from Applications and wait for the whale icon to settle.

</details>

<details>
<summary><strong>Linux (Ubuntu / Debian)</strong></summary>

```bash
sudo apt update && sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**Log out and back in** for the group change to take effect.

Verify:

```bash
docker info
```

</details>

<details>
<summary><strong>Linux (Fedora / RHEL)</strong></summary>

```bash
sudo dnf install docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**Log out and back in** for the group change to take effect.

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer — **enable WSL 2 backend** when prompted
3. Restart if prompted
4. Launch Docker Desktop — **skip the sign-in** (no account needed)
5. Wait for "Docker Desktop is running" in the system tray

Verify (PowerShell):

```
docker info
```

> Docker won't start? Enable "Virtual Machine Platform" and "Windows Subsystem for Linux" in Windows Features.

</details>

---

## 3. Get an LLM API Key

You need one API key. Pick a provider:

| Provider | Best for | Get a key |
|---|---|---|
| **Anthropic** | Agentic tasks, browser automation, tool use | [console.anthropic.com](https://console.anthropic.com/) |
| **Moonshot / Kimi** | Agentic tasks, long context, tool use | [platform.moonshot.cn](https://platform.moonshot.cn/) |
| **DeepSeek** | Cost-effective reasoning | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenAI** | General purpose | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Groq** | Fast inference, free tier | [console.groq.com](https://console.groq.com/) |

> **Recommended:** Anthropic Claude or Moonshot Kimi for agentic workloads. They have built-in computer use training and strong tool-calling support. 100+ providers supported via [LiteLLM](https://litellm.ai).

---

## 4. Install OpenLegion

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python3 -m venv .venv
source .venv/bin/activate
pip3 install -e ".[dev]"
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> If PowerShell blocks the script: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

</details>

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```cmd
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

</details>

Verify:

```bash
openlegion --help
```

---

## 5. Run Setup

**Docker must be running.** If you skipped step 2, go back.

```bash
openlegion setup
```

The wizard asks for:

1. **LLM provider + API key** — pick one, paste your key
2. **Project description** — one line about what your agents will do
3. **Team template** — starter (1 agent), sales (3), devteam (3), or content (3)
4. **Docker image build** — automatic, takes 1-2 minutes on first run

---

## 6. Start

```bash
openlegion start
```

Containers start, then the interactive REPL appears. Try these:

```
@researcher Research the latest developments in AI agent frameworks.
```

```
/costs              # see per-agent LLM spend
/agents             # list running agents
/broadcast Hello!   # message all agents
/quit               # stop everything
```

---

## After Setup

```bash
openlegion agent add analyst    # add an agent
openlegion start -d             # run in background
openlegion chat researcher      # connect from another terminal
openlegion stop                 # shut down
```

### Connect Telegram (optional)

Add your Telegram bot token to `config/mesh.yaml`. On next `openlegion start`, a pairing code appears — send it to your bot to link.

### Connect Discord (optional)

Same pattern with a Discord bot token.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: openlegion` | Activate the venv: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows) |
| `Docker is not running` | Open Docker Desktop (macOS/Windows) or `sudo systemctl start docker` (Linux) |
| `permission denied` on Docker | Linux only: `sudo usermod -aG docker $USER` then log out and back in |
| `python3: command not found` | Install Python 3.12+ (see step 1) |
| `pip install` permission error | You're outside the venv — activate it first |
| `ModuleNotFoundError` | Same — activate the venv |
| Docker build is slow | First build downloads base image + Chromium (~2 min). Rebuilds use cache. |
| Agent not responding | Run `openlegion status`. Check your API key has credits. |
| Setup hangs at "Building Docker image" | Normal on first run. Wait 2-3 minutes. |
| PowerShell blocks venv activation | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
