# Mandelscope - one-time environment setup for Windows (PowerShell)
#   Right-click > "Run with PowerShell", or:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1. uv (fast Python package / venv manager from Astral)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv ..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

# 2. Python 3.12 managed by uv (does not touch any system Python)
uv python install 3.12

# 3. Create .venv and install the locked dependencies (+ pytest)
uv sync --all-extras

# 4. Smoke test: GPU, OpenGL version, video encoders (NVENC on GeForce)
uv run python -m mandelscope info

Write-Host ""
Write-Host "Setup done." -ForegroundColor Green
Write-Host "Start the explorer with the Mandelscope shortcut (made by Install Mandelscope.bat) or:  .\run.bat" -ForegroundColor Green
Write-Host "Run the test-suite with:        uv run pytest -q" -ForegroundColor Green
