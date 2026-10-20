# One-time setup for Windows.
# Run from the project root (the folder containing src):
#     .\scripts\setup.ps1
#
# If PowerShell refuses to run it:
#     powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
#
# This file is intentionally ASCII-only. PowerShell 5.1 reads .ps1 files as
# ANSI when there is no BOM, which turns any non-ASCII text into garbage and
# breaks parsing. English-only messages avoid that entirely.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== TrendRadar setup ===" -ForegroundColor Cyan
Write-Host ""

# --- check we are in the right folder -------------------------------------
if (-not (Test-Path ".\requirements.txt")) {
    Write-Host "ERROR: wrong folder." -ForegroundColor Red
    Write-Host "There is no requirements.txt here, so this is not the project root."
    Write-Host ""
    Write-Host "Current folder: $(Get-Location)"
    $nested = Get-ChildItem -Directory -ErrorAction SilentlyContinue |
              Where-Object { Test-Path (Join-Path $_.FullName "requirements.txt") }
    if ($nested) {
        Write-Host ""
        Write-Host "The project looks like it is inside: .\$($nested[0].Name)" -ForegroundColor Yellow
        Write-Host "Run these two commands:" -ForegroundColor Yellow
        Write-Host "    cd $($nested[0].Name)" -ForegroundColor Cyan
        Write-Host "    .\scripts\setup.ps1" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "Find the folder that contains src, then run it from there."
    }
    Write-Host ""
    exit 1
}
Write-Host "Folder looks right." -ForegroundColor Green

# --- python ---------------------------------------------------------------
Write-Host ""
Write-Host "--- Python ---" -ForegroundColor Cyan
$py = $null
foreach ($cand in @("python", "py", "python3")) {
    try {
        $v = & $cand --version 2>&1
        if ($LASTEXITCODE -eq 0) { $py = $cand; Write-Host "$cand -> $v"; break }
    } catch { }
}
if (-not $py) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host "Install it from python.org and tick 'Add Python to PATH'."
    Write-Host "Then CLOSE this window and open PowerShell again."
    exit 1
}

# --- venv -----------------------------------------------------------------
Write-Host ""
Write-Host "--- Virtual environment ---" -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: could not create .venv" -ForegroundColor Red
        exit 1
    }
    Write-Host "created .venv"
} else {
    Write-Host ".venv already exists"
}
$vpy  = ".\.venv\Scripts\python.exe"
$vpip = ".\.venv\Scripts\pip.exe"

# --- dependencies ---------------------------------------------------------
Write-Host ""
Write-Host "--- Dependencies (this takes 2-4 minutes) ---" -ForegroundColor Cyan
& $vpip install --upgrade pip --quiet
& $vpip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: could not install dependencies." -ForegroundColor Red
    Write-Host "Read the message above. Common causes:"
    Write-Host "  - no internet"
    Write-Host "  - corporate proxy / antivirus blocking pip"
    Write-Host "  - disk full"
    exit 1
}
Write-Host "dependencies installed" -ForegroundColor Green

# --- database -------------------------------------------------------------
Write-Host ""
Write-Host "--- Database ---" -ForegroundColor Cyan
& $vpy src\db.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: could not create the database." -ForegroundColor Red
    exit 1
}

# --- done -----------------------------------------------------------------
Write-Host ""
Write-Host "=== Ready ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next, run these from this same folder:"
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "    python src\collect.py" -ForegroundColor Cyan
Write-Host "    python src\youtube_collect.py --seed" -ForegroundColor Cyan
Write-Host "    python src\origins.py" -ForegroundColor Cyan
Write-Host "    python src\report.py" -ForegroundColor Cyan
Write-Host "    python -m streamlit run dashboard\app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "If Activate.ps1 is blocked by policy, use the long form instead:"
Write-Host "    .\.venv\Scripts\python.exe src\collect.py" -ForegroundColor DarkGray
Write-Host ""
