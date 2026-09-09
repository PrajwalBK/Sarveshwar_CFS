[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$BuildFrontend
)

$ErrorActionPreference = 'Stop'
$launcherRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $launcherRoot 'backend'
$frontendRoot = Join-Path $launcherRoot 'frontend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'
$frontendIndex = Join-Path $frontendRoot 'dist\gate\browser\index.html'

if (-not (Test-Path -LiteralPath $backendPython -PathType Leaf)) {
    throw 'Backend environment is missing. Create backend\.venv and install backend\requirements.txt first.'
}

if ($BuildFrontend -or -not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        throw 'The Angular build is missing and npm.cmd is unavailable. Install Node.js, run npm ci in frontend, then retry.'
    }
    Push-Location $frontendRoot
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Angular production build failed.' }
    }
    finally {
        Pop-Location
    }
}

Write-Host ("GateVision starting at http://{0}:{1}" -f $HostAddress, $Port)
Write-Host 'Press Ctrl+C to stop the complete application.'
Set-Location $backendRoot
& $backendPython -m uvicorn app.main:app --host $HostAddress --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
