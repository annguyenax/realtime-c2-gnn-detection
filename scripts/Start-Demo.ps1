param(
    [string]$Threshold = "0.9118",
    [int]$MaxFlows = 0,
    [int]$RealtimeFactor = 50
)

$ROOT = Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH = "$ROOT\src"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=== C2GNN Demo ===" -ForegroundColor Cyan
Write-Host "Root: $ROOT"

# Start FastAPI
Write-Host "[1/3] Starting FastAPI on port 8000..." -ForegroundColor Yellow
$apiArgs = "-NoExit", "-Command", "`$env:PYTHONPATH='$ROOT\src'; `$env:PYTHONIOENCODING='utf-8'; Set-Location '$ROOT'; python -m uvicorn c2gnn.api.server:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList $apiArgs

# Start Streamlit
Write-Host "[2/3] Starting Streamlit on port 8501..." -ForegroundColor Yellow
$dashArgs = "-NoExit", "-Command", "`$env:PYTHONPATH='$ROOT\src'; `$env:PYTHONIOENCODING='utf-8'; Set-Location '$ROOT'; python -m streamlit run src/c2gnn/dashboard/app.py --server.port 8501"
Start-Process powershell -ArgumentList $dashArgs

# Wait for servers to start
Write-Host "Waiting 5s for servers to be ready..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Open dashboard in browser
Write-Host "Opening dashboard in browser..." -ForegroundColor Gray
Start-Process "http://localhost:8501"

# Run pipeline in current window
Write-Host "[3/3] Running pipeline (threshold=$Threshold, factor=$RealtimeFactor)..." -ForegroundColor Yellow
$pipelineCmd = "python -m c2gnn.realtime.pipeline " +
    "--data data/processed/scenario10_test.parquet " +
    "--model models/artifacts/graphsage_best.pt " +
    "--model-type graphsage " +
    "--threshold $Threshold " +
    "--window-size 60 " +
    "--realtime-factor $RealtimeFactor " +
    "--api-url http://localhost:8000"

if ($MaxFlows -gt 0) {
    $pipelineCmd += " --max-flows $MaxFlows"
}

Set-Location $ROOT
Invoke-Expression $pipelineCmd

Write-Host ""
Write-Host "=== Pipeline complete. Dashboard: http://localhost:8501 ===" -ForegroundColor Cyan
