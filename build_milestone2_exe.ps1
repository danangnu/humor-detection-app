$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Humor Bot Milestone 2 - Final Windows Build" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$Python = ".\venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found: .\venv"
}

$Required = @(
    ".\app.py",
    ".\core.py",
    ".\generator.py",
    ".\feedback_store.py",
    ".\model_admin.py",
    ".\desktop_launcher.py",
    ".\HumorBot_Milestone2.spec",

    ".\training\correction_dataset.py",
    ".\training\candidate_gate.py",
    ".\training\candidate_trainer.py",

    ".\static\index.html",
    ".\static\style.css",
    ".\static\script.js",
    ".\static\admin.html",
    ".\static\admin.js",
    ".\static\admin_m2_3.css",

    ".\models\humor_transformer\config.json",
    ".\models\humor_transformer\thresholds.json",

    ".\data\processed\train.csv",
    ".\data\processed\validation.csv",
    ".\data\processed\test.csv"
)

foreach ($File in $Required) {
    if (-not (Test-Path $File)) {
        throw "Required file missing: $File"
    }
}

$WeightsFound =
    (Test-Path ".\models\humor_transformer\model.safetensors") -or
    (Test-Path ".\models\humor_transformer\pytorch_model.bin")

if (-not $WeightsFound) {
    throw "No trained model weights found in models\humor_transformer."
}

Write-Host "Python:" -ForegroundColor Gray
& $Python --version

Write-Host ""
Write-Host "Running source-level safety tests..." -ForegroundColor Yellow

if (Test-Path ".\tests\test_model_admin.py") {
    & $Python -m pytest ".\tests\test_model_admin.py" -q

    if ($LASTEXITCODE -ne 0) {
        throw "Model-admin tests failed."
    }
}

if (Test-Path ".\tests\test_candidate_gate.py") {
    & $Python -m pytest ".\tests\test_candidate_gate.py" -q

    if ($LASTEXITCODE -ne 0) {
        throw "Candidate-gate tests failed."
    }
}

Write-Host ""
Write-Host "Checking the local active Transformer..." -ForegroundColor Yellow

$env:ALLOW_BOOTSTRAP_MODEL = "false"
$env:HUMOR_MODEL_PATH = (Resolve-Path ".\models\humor_transformer").Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

& $Python -c "from core import classifier; classifier.load(); print('Model:', classifier.model_source); print('Thresholds:', classifier.thresholds.low, classifier.thresholds.high); assert classifier.model_source == 'local-trained-model'"

if ($LASTEXITCODE -ne 0) {
    throw "Local trained model pre-build check failed."
}

Write-Host ""
Write-Host "Installing/updating PyInstaller tools..." -ForegroundColor Yellow

& $Python -m pip install --upgrade `
    pyinstaller `
    pyinstaller-hooks-contrib

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller installation failed."
}

Write-Host ""
Write-Host "Cleaning previous build..." -ForegroundColor Yellow

Remove-Item ".\build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\dist\HumorBot" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\release\HumorBot_Milestone2.zip" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Building..." -ForegroundColor Yellow
Write-Host "PyTorch + Transformers + retraining dependencies make this a large package."
Write-Host ""

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    ".\HumorBot_Milestone2.spec"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

$Dist = Resolve-Path ".\dist\HumorBot"
$Exe = Join-Path $Dist "HumorBot.exe"

if (-not (Test-Path $Exe)) {
    throw "Build finished but HumorBot.exe was not found."
}

if (-not (Test-Path (Join-Path $Dist "i\bootstrap\models\humor_transformer\config.json"))) {
    throw "Bootstrap active model was not packaged."
}

if (-not (Test-Path (Join-Path $Dist "i\bootstrap\data\processed\train.csv"))) {
    throw "Bootstrap training dataset was not packaged."
}

if (Test-Path ".\.env.example") {
    Copy-Item ".\.env.example" `
        (Join-Path $Dist ".env.example") `
        -Force
}

@'
HUMOR BOT - MILESTONE 2
========================

INSTALL
-------
1. Extract the full HumorBot folder to a short path, for example:

   C:\HumorBot

2. Run:

   HumorBot.exe

Do not copy HumorBot.exe by itself. The "i" directory is required.

PERSISTENT DATA
---------------
Writable application/model data is NOT stored inside the "i" folder.

Humor Bot uses:

  %LOCALAPPDATA%\HumorBot

That directory contains the runtime copy of:

  humor_feedback.db
  models\humor_transformer
  models\candidates
  models\model_history
  data\processed
  data\retraining
  data\admin
  logs\HumorBot.log

This lets candidate retraining, promotion and rollback work without modifying
PyInstaller's internal files.

FIRST RUN
---------
On first startup, Humor Bot copies the packaged baseline model and protected
training/evaluation data into %LOCALAPPDATA%\HumorBot.

Later launches preserve promoted models and correction history.

ADMIN
-----
Open:

  http://127.0.0.1:8000/admin

The application can:

  - save and approve corrections
  - train candidate Transformers
  - review validation metrics
  - promote a passing candidate
  - back up the previous model
  - roll back a promotion

After PROMOTION or ROLLBACK:
  close Humor Bot and run HumorBot.exe again.

GEMINI
------
Gemini is optional.

To enable it, create a .env file beside HumorBot.exe:

  GEMINI_API_KEY=your_key_here
  GEMINI_MODEL=gemini-2.5-flash

Do not distribute a developer's personal API key.

RESET / CLEAN TEST
------------------
To simulate a first-time installation, close Humor Bot and rename:

  %LOCALAPPDATA%\HumorBot

For example:

  HumorBot_backup

Do not delete it if it contains real client corrections/models you need.

TROUBLESHOOTING
---------------
Log:

  %LOCALAPPDATA%\HumorBot\logs\HumorBot.log
'@ | Set-Content `
    (Join-Path $Dist "README_CLIENT.txt") `
    -Encoding UTF8

Write-Host ""
Write-Host "Creating client ZIP..." -ForegroundColor Yellow

New-Item ".\release" -ItemType Directory -Force | Out-Null

Compress-Archive `
    -Path ".\dist\HumorBot" `
    -DestinationPath ".\release\HumorBot_Milestone2.zip" `
    -CompressionLevel Optimal `
    -Force

$Bytes = (
    Get-ChildItem $Dist -Recurse -File |
    Measure-Object Length -Sum
).Sum

$MB = [math]::Round(
    $Bytes / 1MB,
    1
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " BUILD SUCCESSFUL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Executable:"
Write-Host "  $Exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "Client ZIP:"
Write-Host "  .\release\HumorBot_Milestone2.zip" -ForegroundColor Cyan
Write-Host ""
Write-Host "Uncompressed package size: $MB MB"
Write-Host ""
Write-Host "Next:"
Write-Host "  .\smoke_test_milestone2_exe.ps1" -ForegroundColor Yellow
