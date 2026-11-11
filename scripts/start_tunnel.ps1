param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "OpenSSH client not found. Enable it from Windows optional features first."
}

Write-Host "Starting localhost.run tunnel for http://localhost:$Port ..." -ForegroundColor Cyan
Write-Host "The public HTTPS URL will appear below. Keep this window open." -ForegroundColor DarkGray
Write-Host "If the tunnel drops, this script reconnects automatically and prints a new URL." -ForegroundColor DarkGray
Write-Host ""

while ($true) {
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R "80:localhost:$Port" nokey@localhost.run
    Write-Host ""
    Write-Host "Tunnel disconnected. Reconnecting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
