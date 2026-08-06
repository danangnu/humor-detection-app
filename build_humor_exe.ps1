$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Building Humor Bot Windows executable..." -ForegroundColor Cyan

$Required = @(
  ".\app.py", ".\core.py", ".\generator.py",
  ".\desktop_launcher.py", ".\HumorBot.spec",
  ".\static\index.html", ".\static\style.css", ".\static\script.js",
  ".\models\humor_transformer\config.json",
  ".\models\humor_transformer\thresholds.json"
)
foreach ($f in $Required) {
  if (-not (Test-Path $f)) { throw "Required file missing: $f" }
}

if (-not ((Test-Path ".\models\humor_transformer\model.safetensors") -or (Test-Path ".\models\humor_transformer\pytorch_model.bin"))) {
  throw "Trained model weights were not found."
}
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
  throw "Virtual environment not found at .\venv"
}

$Python = ".\venv\Scripts\python.exe"

$env:ALLOW_BOOTSTRAP_MODEL = "false"
$env:HUMOR_MODEL_PATH = "models/humor_transformer"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Checking trained local model..." -ForegroundColor Yellow
& $Python -c "from core import classifier; classifier.load(); print('Model:', classifier.model_source); print('Thresholds:', classifier.thresholds.low, classifier.thresholds.high); assert classifier.model_source == 'local-trained-model'"
if ($LASTEXITCODE -ne 0) { throw "Local model pre-build check failed." }

Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
& $Python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
if ($LASTEXITCODE -ne 0) { throw "PyInstaller installation failed." }

Remove-Item ".\build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\dist\HumorBot" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Packaging PyTorch + Transformer model. This may take several minutes..." -ForegroundColor Yellow
& $Python -m PyInstaller --noconfirm --clean ".\HumorBot.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }

$Dist = Resolve-Path ".\dist\HumorBot"
$Exe = Join-Path $Dist "HumorBot.exe"
if (-not (Test-Path $Exe)) { throw "HumorBot.exe was not produced." }

if (Test-Path ".\.env.example") {
  Copy-Item ".\.env.example" (Join-Path $Dist ".env.example") -Force
}

@'
HUMOR BOT - WINDOWS BUILD

Run:
  HumorBot.exe

The application starts a local server and opens your default browser.

Important:
- The trained local Transformer is bundled.
- The bootstrap model is disabled.
- Gemini is optional.
- Without a .env file the local fallback response is used.

To enable Gemini, create .env beside HumorBot.exe:
  GEMINI_API_KEY=your_key_here
  GEMINI_MODEL=gemini-2.5-flash

Send the ENTIRE HumorBot folder, not only HumorBot.exe.

If startup fails, check HumorBot.log.
'@ | Set-Content (Join-Path $Dist "README_CLIENT.txt") -Encoding UTF8

$Bytes = (Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum
$MB = [math]::Round($Bytes / 1MB, 1)

Write-Host ""
Write-Host "BUILD SUCCESSFUL" -ForegroundColor Green
Write-Host "Executable: $Exe"
Write-Host "Deliver folder: $Dist"
Write-Host "Package size: $MB MB"
Write-Host ""
Write-Host "Test with:" -ForegroundColor Yellow
Write-Host ".\dist\HumorBot\HumorBot.exe"
