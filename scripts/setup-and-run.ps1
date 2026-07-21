# One-shot setup for running Better Backgrounds from a source checkout on Windows.
# Installs dependencies, downloads the three mandatory models, and starts the app.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$missing = @()
foreach ($tool in @("uv", "node", "make")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $missing += $tool }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing required tools: $($missing -join ', ')"
    Write-Host ""
    Write-Host "  uv    https://docs.astral.sh/uv/getting-started/installation/"
    Write-Host "  node  https://nodejs.org/ (version 20 or newer)"
    Write-Host "  make  winget install GnuWin32.Make  (or use Git Bash with scripts/setup-and-run.sh)"
    exit 1
}

Write-Host "==> Installing dependencies (this downloads PyTorch and can take a while)"
make setup
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Downloading the three mandatory models (~3.1 GiB, once)"
Write-Host "    SHARP is licensed for non-commercial scientific research only."
uv run better-backgrounds prepare-models --accept-model-license
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Starting Better Backgrounds"
make desktop
