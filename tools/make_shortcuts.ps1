# Creates "Mandelscope" shortcuts (Desktop, Start menu, this folder) that start the app
# without a console window:  .venv\Scripts\pythonw.exe -m mandelscope.launcher
param([switch]$NoDesktop)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "Missing $pyw - run 'Install Mandelscope.bat' first." }
$icon = Join-Path $root "mandelscope\assets\mandelscope.ico"
$ws = New-Object -ComObject WScript.Shell
$targets = @($root, [Environment]::GetFolderPath("Programs"))
if (-not $NoDesktop) { $targets += [Environment]::GetFolderPath("Desktop") }
foreach ($dir in $targets) {
    $path = Join-Path $dir "Mandelscope.lnk"
    $lnk = $ws.CreateShortcut($path)
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "-m mandelscope.launcher"
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "Mandelscope - GPU deep-zoom Mandelbrot explorer"
    $lnk.Save()
    Write-Host "  shortcut: $path"
}
