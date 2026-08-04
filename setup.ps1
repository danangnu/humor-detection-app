$ErrorActionPreference = "Stop"
if (-not (Test-Path ".\venv")) { py -3.11 -m venv venv }
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path ".\.env")) { Copy-Item .\.env.example .\.env }
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next: python training\prepare_socket_dataset.py"
