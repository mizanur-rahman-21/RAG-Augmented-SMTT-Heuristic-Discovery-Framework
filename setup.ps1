# ==============================================================================
# SMTT RAG Heuristic Discovery - Windows Setup Script
# ==============================================================================
# Sets up Python virtual environment, installs dependencies, and verifies setup.
# Note: This script prepares your environment but does NOT launch costly experiments.
# ==============================================================================

[CmdletBinding()]
param (
    [switch]$SkipVenv = $false,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SMTT RAG Heuristic Discovery - Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python was not found in PATH. Please install Python 3.10+ and add it to PATH."
}

# 2. Virtual Environment
if (-not $SkipVenv) {
    Write-Host "`n[2/5] Setting up virtual environment at '$VenvPath'..." -ForegroundColor Yellow
    if (-not (Test-Path $VenvPath)) {
        Write-Host "  Creating virtual environment..."
        python -m venv $VenvPath
    } else {
        Write-Host "  Virtual environment already exists."
    }

    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "  Activating virtual environment..."
        & $activateScript
    } else {
        Write-Warning "Could not find $activateScript; continuing with current environment."
    }
} else {
    Write-Host "`n[2/5] Skipping virtual environment creation (-SkipVenv specified)." -ForegroundColor Yellow
}

# 3. Install Dependencies
Write-Host "`n[3/5] Installing dependencies from requirements-lock.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Error "Failed to install dependencies."
}

# 4. Check Ollama Status
Write-Host "`n[4/5] Checking Ollama service..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "  Ollama service is ONLINE." -ForegroundColor Green
    
    $modelName = "qwen2.5-coder:7b-instruct-q4_K_M"
    $models = $response.models | ForEach-Object { $_.name }
    $hasModel = $models -contains $modelName -or ($models | Where-Object { $_ -like "*$modelName*" })

    if ($hasModel) {
        Write-Host "  Found target model '$modelName'." -ForegroundColor Green
    } else {
        Write-Host "  Target model '$modelName' is not downloaded yet." -ForegroundColor DarkYellow
        Write-Host "  To download it, run:" -ForegroundColor Cyan
        Write-Host "    ollama pull $modelName" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  Ollama service is currently OFFLINE or unreachable at http://localhost:11434." -ForegroundColor DarkYellow
    Write-Host "  To run LLM discovery experiments, install and start Ollama:" -ForegroundColor Cyan
    Write-Host "    1. Download Ollama from https://ollama.com" -ForegroundColor Cyan
    Write-Host "    2. Start Ollama: ollama serve" -ForegroundColor Cyan
    Write-Host "    3. Pull model:   ollama pull qwen2.5-coder:7b-instruct-q4_K_M" -ForegroundColor Cyan
}

# 5. Run Verification Unit Tests
Write-Host "`n[5/5] Running baseline and solver verification unit tests..." -ForegroundColor Yellow
python -m unittest discover -s tests -p "test_*.py"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  All test suites passed!" -ForegroundColor Green
} else {
    Write-Warning "Some tests failed. Please review the output above."
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Copy configuration:  Copy-Item config.example.yaml config.yaml"
Write-Host "  2. Run preflight check: python run_preflight.py"
Write-Host "  3. Run pilot study:     python run_pilot.py"
Write-Host "============================================================" -ForegroundColor Cyan
