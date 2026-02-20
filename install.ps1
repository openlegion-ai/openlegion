# OpenLegion Installer for Windows (PowerShell)
# Run: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

function Write-OK($msg)   { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  OpenLegion Installer"
Write-Host "  ────────────────────"
Write-Host ""

# ── Check Python ──────────────────────────────────────────────

$python = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        $major = & $cmd -c "import sys; print(sys.version_info.major)" 2>$null
        $minor = & $cmd -c "import sys; print(sys.version_info.minor)" 2>$null
        if ([int]$major -ge 3 -and [int]$minor -ge 10) {
            $python = $cmd
            break
        }
    } catch { }
}

if (-not $python) {
    Write-Fail "Python 3.10+ is required but not found.
    Download from: https://python.org/downloads
    IMPORTANT: Check 'Add python.exe to PATH' during install."
}
Write-OK "Python $ver ($python)"

# ── Check pip ─────────────────────────────────────────────────

try {
    & $python -m pip --version 2>$null | Out-Null
    Write-OK "pip available"
} catch {
    Write-Fail "pip is not installed.
    Run: $python -m ensurepip --upgrade"
}

# ── Check Docker ──────────────────────────────────────────────

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail "Docker is not installed.
    Download Docker Desktop: https://docker.com/products/docker-desktop"
}

try {
    docker info 2>$null | Out-Null
    Write-OK "Docker is running"
} catch {
    Write-Fail "Docker is installed but not running.
    Open Docker Desktop and wait for 'Docker Desktop is running'."
}

# ── Check Git ─────────────────────────────────────────────────

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "Git is not installed.
    Download from: https://git-scm.com"
}
Write-OK "Git available"

# ── Create virtual environment ────────────────────────────────

Write-Host ""
if (-not (Test-Path ".venv")) {
    Write-Host "  Creating virtual environment..."
    & $python -m venv .venv
    Write-OK "Virtual environment created"
} else {
    Write-OK "Virtual environment exists"
}

& .venv\Scripts\Activate.ps1

# ── Install dependencies ──────────────────────────────────────

Write-Host ""
Write-Host "  Installing dependencies..." -NoNewline:$false
Write-Host "  First install takes 2-3 minutes (downloads ~70 packages)." -ForegroundColor Yellow
Write-Host "  Subsequent installs are fast (cached)." -ForegroundColor Yellow
Write-Host ""

pip install -e ".[dev]" 2>&1 | ForEach-Object {
    if ($_ -match "Collecting|Downloading|Installing|Building|Successfully") {
        Write-Host "  $_"
    }
}

Write-Host ""
Write-OK "OpenLegion installed"

# ── Global CLI access ────────────────────────────────────────

Write-Host ""
$LinkDir = Join-Path $env:LOCALAPPDATA "OpenLegion"
$VenvExe = Join-Path $PWD ".venv\Scripts\openlegion.exe"

$NeedRestart = $false
if (Test-Path $VenvExe) {
    if (-not (Test-Path $LinkDir)) { New-Item -ItemType Directory -Path $LinkDir | Out-Null }

    # Create a .cmd wrapper that invokes the venv entry point
    # Includes existence check so users get a clear error if the project is moved
    $WrapperPath = Join-Path $LinkDir "openlegion.cmd"
    $WrapperContent = @"
@echo off
if not exist "$VenvExe" (
    echo openlegion: installation not found at $VenvExe
    echo Re-run install.ps1 from the project directory to fix.
    exit /b 1
)
"$VenvExe" %*
"@
    Set-Content -Path $WrapperPath -Value $WrapperContent

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$LinkDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$LinkDir;$UserPath", "User")
        $env:Path = "$LinkDir;$env:Path"
        Write-OK "Added $LinkDir to user PATH"
        $NeedRestart = $true
    } else {
        Write-OK "openlegion available globally"
    }
} else {
    Write-Warn "Could not create global wrapper — run from the project directory"
}

# ── Done ──────────────────────────────────────────────────────

Write-Host ""
Write-Host "  Ready!" -ForegroundColor Green -NoNewline
Write-Host " Next steps:"
Write-Host ""
if ($NeedRestart) {
    Write-Host "    Restart your terminal, then:" -ForegroundColor Yellow
}
Write-Host "    openlegion setup             # configure API key + agents"
Write-Host "    openlegion start             # launch and start chatting"
Write-Host ""
