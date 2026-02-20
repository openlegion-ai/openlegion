# Quick Start Guide

## Fast Path (macOS / Linux)

**Prerequisites:** [Docker Desktop](https://docker.com/products/docker-desktop) running, Python 3.10+, an LLM API key.

```bash
git clone https://github.com/openlegion-ai/openlegion.git && cd openlegion
./install.sh         # checks dependencies, creates venv, installs everything
source .venv/bin/activate
openlegion setup     # API key, project description, team template
openlegion start     # launch agents and start chatting
```

That's it. If `install.sh` passes, you're good. If it fails, it tells you exactly what's missing.

You can also use `make`:

```bash
make install         # same as ./install.sh
make setup           # install + openlegion setup
make start           # openlegion start
make test            # run the test suite
```

## Fast Path (Windows)

**Prerequisites:** [Docker Desktop](https://docker.com/products/docker-desktop) running, Python 3.10+, an LLM API key.

```powershell
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
powershell -ExecutionPolicy Bypass -File install.ps1
.venv\Scripts\Activate.ps1
openlegion setup     # API key, project description, team template
openlegion start     # launch agents and start chatting
```

> PowerShell blocks the script? Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Prerequisites

You need three things installed before running OpenLegion:

### 1. Python 3.10+

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install python@3
python3 --version   # verify: 3.10 or higher
```

Or download from [python.org/downloads](https://www.python.org/downloads/).

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv git

# Fedora / RHEL
sudo dnf install python3 python3-pip git
```

```bash
python3 --version   # verify: 3.10 or higher
```

> Distro ships 3.9 or older? Use [pyenv](https://github.com/pyenv/pyenv) to get 3.10+.

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Download Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/)
2. Run installer — **check "Add python.exe to PATH"**
3. Verify in PowerShell: `python --version`

> On Windows, use `python` and `pip` (not `python3` / `pip3`).

</details>

### 2. Docker

Docker must be **installed and running** before setup.

<details>
<summary><strong>macOS</strong></summary>

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Drag to Applications and launch
3. **Skip the sign-in** — no account needed
4. Wait for the whale icon in the menu bar to settle

```bash
docker info   # verify: should print server info
```

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install docker.io
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker $USER   # then log out and back in
```

```bash
docker info   # verify
```

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Install — **enable WSL 2 backend** when prompted
3. Restart if prompted
4. Launch Docker Desktop — **skip the sign-in**
5. Wait for "Docker Desktop is running"

```
docker info   # verify in PowerShell
```

</details>

### 3. LLM API Key

You need one key. Pick a provider:

| Provider | Best for | Get a key |
|---|---|---|
| **Anthropic** | Agentic tasks, browser automation, tool use | [console.anthropic.com](https://console.anthropic.com/) |
| **Moonshot / Kimi** | Agentic tasks, long context, tool use | [platform.moonshot.cn](https://platform.moonshot.cn/) |
| **DeepSeek** | Cost-effective reasoning | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenAI** | General purpose | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Groq** | Fast inference, free tier | [console.groq.com](https://console.groq.com/) |

> 100+ providers supported via [LiteLLM](https://litellm.ai). Anthropic Claude or Moonshot Kimi recommended for agentic workloads.

---

## Install (Manual)

If you prefer not to use the install scripts:

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

> PowerShell blocks the script? Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

</details>

> First install downloads ~70 packages and takes 2-3 minutes. Subsequent installs use pip's cache and are much faster.

Verify: `openlegion --help`

---

## Setup and Start

```bash
openlegion setup     # walks you through API key, project, team template, Docker build
openlegion start     # launches agents and drops you into the chat REPL
```

The setup wizard asks for:
1. **LLM provider + API key** — pick one, paste your key
2. **Project description** — one line about what your agents will do (optional)
3. **Team template** — starter (1 agent), sales (3), devteam (3), or content (3)

The Docker image builds automatically on your first `openlegion start` (~2 min).

---

## Using the REPL

```
@researcher Research the latest developments in AI agent frameworks.
/costs              # per-agent LLM spend
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

### Telegram / Discord (optional)

```bash
openlegion channels add telegram   # prompts for bot token
openlegion channels add discord    # prompts for bot token
openlegion channels list           # check what's connected
```

On next `openlegion start`, a pairing code appears — send it to your bot to link.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: openlegion` | Activate the venv: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows) |
| `Docker is not running` | Open Docker Desktop (macOS/Windows) or `sudo systemctl start docker` (Linux) |
| `permission denied` on Docker | Linux: `sudo usermod -aG docker $USER` then log out/in |
| `python3: command not found` | Install Python 3.10+ (see above). On Windows, use `python` instead of `python3`. |
| `pip install` is slow | First install downloads ~70 packages (2-3 min). This is normal. Subsequent installs are fast. |
| `pip install` permission error | Activate the venv first — don't install globally. |
| Docker build is slow | First build downloads base image + Chromium (~2 min). Rebuilds are fast. |
| Agent not responding | `openlegion status` to check health. Verify API key has credits. |
| PowerShell blocks scripts | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
