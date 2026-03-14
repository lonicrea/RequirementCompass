param(
  [string]$BackendDir = ".\\backend",
  [int]$BackendPort = 5000
)

$ErrorActionPreference = "Stop"

function Get-CloudflaredPath {
  $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $fallback = "C:\\Users\\$env:USERNAME\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\\cloudflared.exe"
  if (Test-Path $fallback) { return $fallback }
  throw "cloudflared not found. Install first: winget install --id Cloudflare.cloudflared -e"
}

function Start-Backend {
  param([string]$Dir, [int]$Port)

  $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Host "Backend already running on port $Port, skip start."
    return
  }

  $venvPython = Join-Path $Dir ".venv\\Scripts\\python.exe"
  $pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

  $outLog = Join-Path $Dir "uvicorn.public.log"
  $errLog = Join-Path $Dir "uvicorn.public.err"

  Write-Host "Starting backend..."
  Start-Process -FilePath $pythonExe `
    -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port $Port" `
    -WorkingDirectory $Dir `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog | Out-Null

  Start-Sleep -Seconds 3
  $check = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if (-not $check) {
    throw "Backend failed to start. Check: $errLog"
  }
}

function Start-QuickTunnel {
  param([string]$CfPath, [int]$Port, [string]$LogPath)

  Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
  if (Test-Path $LogPath) { Remove-Item $LogPath -Force }

  Write-Host "Starting Cloudflare Quick Tunnel..."
  Start-Process -FilePath $CfPath `
    -ArgumentList "tunnel --url http://localhost:$Port --no-autoupdate --logfile `"$LogPath`" --loglevel info" | Out-Null
}

function Wait-TunnelUrl {
  param([string]$LogPath, [int]$TimeoutSec = 90)

  $pattern = "https://[a-z0-9-]+\.trycloudflare\.com"
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    if (Test-Path $LogPath) {
      $content = Get-Content $LogPath -Raw
      $m = [regex]::Match($content, $pattern)
      if ($m.Success) { return $m.Value }
    }
    Start-Sleep -Milliseconds 700
  }
  throw "Timeout waiting for tunnel URL. Check cloudflare network access."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Resolve-Path (Join-Path $root $BackendDir)
$cfPath = Get-CloudflaredPath

Start-Backend -Dir $backendPath -Port $BackendPort

$tunnelLog = Join-Path $backendPath "cloudflared.quick.log"
Start-QuickTunnel -CfPath $cfPath -Port $BackendPort -LogPath $tunnelLog
$tunnelUrl = Wait-TunnelUrl -LogPath $tunnelLog

Write-Host ""
Write-Host "==== Deployment Info ===="
Write-Host "Set frontend env NEXT_PUBLIC_API_BASE_URL to:"
Write-Host "$tunnelUrl/api"
Write-Host ""
Write-Host "Quick Tunnel URL: $tunnelUrl"
Write-Host "Backend health check: $tunnelUrl/api/health"
Write-Host ""
Write-Host "Stop tunnel: Stop-Process -Name cloudflared -Force"
