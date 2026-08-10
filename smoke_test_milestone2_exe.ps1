$ErrorActionPreference = "Stop"

$Exe = ".\dist\HumorBot\HumorBot.exe"

if (-not (Test-Path $Exe)) {
    throw "HumorBot.exe not found. Run .\build_milestone2_exe.ps1 first."
}

$Existing = Get-NetTCPConnection `
    -LocalPort 8000 `
    -State Listen `
    -ErrorAction SilentlyContinue

if ($Existing) {
    throw "Port 8000 is already in use. Stop the existing Humor Bot/server first."
}

Write-Host "Starting packaged Humor Bot..." -ForegroundColor Cyan

$Process = Start-Process $Exe -PassThru
$Health = $null

for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 500

    try {
        $Health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8000/health" `
            -TimeoutSec 1

        if ($Health.status -eq "ok") {
            break
        }
    }
    catch {
    }

    if ($Process.HasExited) {
        break
    }
}

if (-not $Health -or $Health.status -ne "ok") {
    throw "Packaged app failed health check. Review %LOCALAPPDATA%\HumorBot\logs\HumorBot.log"
}

Write-Host ""
Write-Host "Health check: PASS" -ForegroundColor Green
Write-Host "Version:          $($Health.version)"
Write-Host "Model source:     $($Health.model_source)"
Write-Host "Local model:      $($Health.local_model_ready)"
Write-Host "Runtime root:     $($Health.runtime_root)"
Write-Host "Low threshold:    $($Health.thresholds.low)"
Write-Host "High threshold:   $($Health.thresholds.high)"

$Admin = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/admin/status" `
    -TimeoutSec 5

Write-Host ""
Write-Host "Admin API check: PASS" -ForegroundColor Green
Write-Host "Active model:     $($Admin.active_model.model_version)"
Write-Host "Retraining ready: $($Admin.retraining_available)"
Write-Host "Rollback ready:   $($Admin.rollback_available)"

if (-not $Health.local_model_ready) {
    throw "Packaged app did not report local_model_ready=true."
}

Write-Host ""
Write-Host "Milestone 2 packaged smoke test: PASS" -ForegroundColor Green
Write-Host ""
Write-Host "Manual checks still required:"
Write-Host "  1. Analyze text"
Write-Host "  2. Save/approve a correction"
Write-Host "  3. Train Candidate from /admin"
Write-Host "  4. Promote + restart"
Write-Host "  5. Rollback + restart"
