$ErrorActionPreference = "Stop"
$Exe = ".\dist\HumorBot\HumorBot.exe"
if (-not (Test-Path $Exe)) { throw "Build the EXE first." }

$Existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($Existing) {
  throw "Port 8000 is already in use. Stop the existing server before this smoke test."
}

$Process = Start-Process $Exe -PassThru
$Ready = $false

for ($i=0; $i -lt 60; $i++) {
  Start-Sleep -Milliseconds 500
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 1
    if ($h.status -eq "ok") {
      $Ready = $true
      break
    }
  } catch {}
  if ($Process.HasExited) { break }
}

if (-not $Ready) {
  throw "Packaged app health check failed. Check dist\HumorBot\HumorBot.log."
}

Write-Host "PASS" -ForegroundColor Green
Write-Host "Model source: $($h.model_source)"
Write-Host "Local model ready: $($h.local_model_ready)"
Write-Host "Thresholds: $($h.thresholds.low) / $($h.thresholds.high)"

if (-not $h.local_model_ready) { throw "Bundled local model is not ready." }
