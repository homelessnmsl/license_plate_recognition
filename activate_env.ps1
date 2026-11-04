# Dot-source this file so activation persists in the current shell:
#   . .\activate_env.ps1
$VenvPython = "E:\venvs\lpr311\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found: $VenvPython"
}
& "E:\venvs\lpr311\Scripts\Activate.ps1"
Write-Host "Activated: E:\venvs\lpr311" -ForegroundColor Green
$env:LPR_DEVICE = "0"
$env:LPR_OCR_USE_GPU = "false"