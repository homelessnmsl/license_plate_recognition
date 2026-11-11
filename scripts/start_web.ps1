param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [string]$Token = "",
    [switch]$NoWarmup,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = "E:\venvs\lpr311\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

if (-not $Token) { $Token = $env:LPR_WEB_TOKEN }
if (-not $Token) {
    $Token = -join ((48..57) + (65..90) | Get-Random -Count 8 | ForEach-Object { [char]$_ })
}

$OnnxModel = Join-Path $Root "deploy\models\yolov8n_plate.onnx"
$OcrModel = Join-Path $Root "deploy\models\ocr_rec"
if (Test-Path -LiteralPath $OnnxModel) {
    $env:LPR_YOLO_WEIGHTS = $OnnxModel
}
if (Test-Path -LiteralPath (Join-Path $OcrModel "inference.pdmodel")) {
    $env:LPR_OCR_REC_MODEL_DIR = $OcrModel
}
$env:LPR_WEB_TOKEN = $Token
$env:LPR_HOST = $BindHost
$env:LPR_PORT = "$Port"
$env:PYTHONUTF8 = "1"
if ($NoWarmup) { $env:LPR_WARMUP = "false" } else { $env:LPR_WARMUP = "true" }

$Ipv4 = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notmatch '^(127\.|169\.254\.|172\.(1[6-9]|2[0-9]|3[01])\.)' -and
        $_.InterfaceAlias -notmatch 'VMware|vEthernet|Loopback'
    } |
    Select-Object -First 1 -ExpandProperty IPAddress
$LanUrl = if ($Ipv4) { "http://${Ipv4}:$Port/?key=$Token" } else { "http://localhost:$Port/?key=$Token" }

Write-Host ""
Write-Host "Plate Lens web service" -ForegroundColor Cyan
Write-Host "Local: $LanUrl"
Write-Host "Access code: $Token" -ForegroundColor Yellow
Write-Host "Health: http://localhost:$Port/api/health"
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

Push-Location $Root
try {
    & $Python -m uvicorn webapp.app:app --host $BindHost --port $Port --workers 1
} finally {
    Pop-Location
}
