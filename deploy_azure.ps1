<#
.SYNOPSIS
    One-shot deployment of the Regulatory Impact & Readiness Streamlit cockpit
    to an Azure App Service (Linux, Python) -- no GitHub, no Docker.

.DESCRIPTION
    This script:
      1. Verifies Azure CLI is installed and you are signed in.
      2. Creates (or reuses) a Resource Group, App Service Plan, and Web App.
      3. Configures the startup command, build behaviour, and App Settings
         (mapped from your local .env file).
      4. Packages the project into a ZIP (excluding secrets, __pycache__,
         venv, local data files, IDE folders) and pushes it via
         `az webapp deploy`.
      5. Prints the live URL.

.PARAMETER ResourceGroup
    Resource Group name. Created if it does not exist.

.PARAMETER Location
    Azure region (e.g. eastus, westeurope, uksouth).

.PARAMETER PlanName
    App Service Plan name. Created if it does not exist.

.PARAMETER PlanSku
    Plan SKU. B1 (Basic, ~$13/mo) is the minimum practical for this app.
    Use P1v3 for production. F1 (free) has 60 min/day CPU quota and no
    "Always On", so Streamlit's websocket may drop -- not recommended.

.PARAMETER AppName
    Web App name. Must be globally unique across *.azurewebsites.net.

.PARAMETER PythonVersion
    Python runtime version. 3.11 matches what the project has been tested on.

.PARAMETER EnvFile
    Path to the local .env file used to seed App Settings. Defaults to .\.env
    next to the script. Set to "" to skip App Settings sync.

.EXAMPLE
    # First-time deploy
    .\deploy_azure.ps1 `
        -ResourceGroup rg-reg-impact `
        -Location eastus `
        -PlanName plan-reg-impact `
        -AppName reg-impact-cockpit-demo `
        -PlanSku B1

.EXAMPLE
    # Redeploy code only (RG/plan/app already exist, App Settings already set)
    .\deploy_azure.ps1 `
        -ResourceGroup rg-reg-impact `
        -Location eastus `
        -PlanName plan-reg-impact `
        -AppName reg-impact-cockpit-demo `
        -EnvFile ""

.NOTES
    Requires: Azure CLI 2.60+ ( https://aka.ms/installazurecliwindows ).
    Run `az login` before invoking this script.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $ResourceGroup,
    [Parameter(Mandatory = $true)] [string] $Location,
    [Parameter(Mandatory = $true)] [string] $PlanName,
    [Parameter(Mandatory = $true)] [string] $AppName,
    [ValidateSet('F1', 'B1', 'B2', 'B3', 'P0v3', 'P1v3', 'P2v3', 'P3v3')]
    [string] $PlanSku = 'B1',
    [string] $PythonVersion = '3.11',
    [string] $EnvFile = ".\.env"
)

$ErrorActionPreference = 'Stop'

function Write-Section {
    param([string] $Title)
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
Write-Section "0. Preflight checks"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is not installed. Install from https://aka.ms/installazurecliwindows and re-run."
}

$account = az account show --output json 2>$null | ConvertFrom-Json
if (-not $account) {
    throw "Not signed in. Run 'az login' first."
}
Write-Host "Signed in as       : $($account.user.name)"
Write-Host "Subscription       : $($account.name) ($($account.id))"
Write-Host "Tenant             : $($account.tenantId)"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

if (-not (Test-Path ".\app.py")) {
    throw "app.py not found in $scriptRoot. Run this script from the project root."
}
if (-not (Test-Path ".\requirements.txt")) {
    throw "requirements.txt not found in $scriptRoot."
}

# ---------------------------------------------------------------------------
# 1. Resource Group
# ---------------------------------------------------------------------------
Write-Section "1. Resource Group: $ResourceGroup"

$rgExists = az group exists --name $ResourceGroup | ConvertFrom-Json
if ($rgExists) {
    Write-Host "Resource Group already exists -- reusing."
} else {
    az group create --name $ResourceGroup --location $Location --output none
    Write-Host "Created Resource Group $ResourceGroup in $Location."
}

# ---------------------------------------------------------------------------
# 2. App Service Plan (Linux)
# ---------------------------------------------------------------------------
Write-Section "2. App Service Plan: $PlanName ($PlanSku, Linux)"

$plan = az appservice plan show --name $PlanName --resource-group $ResourceGroup --output json 2>$null | ConvertFrom-Json
if ($plan) {
    Write-Host "Plan already exists -- reusing (SKU: $($plan.sku.name))."
} else {
    az appservice plan create `
        --name $PlanName `
        --resource-group $ResourceGroup `
        --location $Location `
        --is-linux `
        --sku $PlanSku `
        --output none
    Write-Host "Created Linux plan $PlanName ($PlanSku)."
}

# ---------------------------------------------------------------------------
# 3. Web App (Python)
# ---------------------------------------------------------------------------
Write-Section "3. Web App: $AppName (Python $PythonVersion)"

$app = az webapp show --name $AppName --resource-group $ResourceGroup --output json 2>$null | ConvertFrom-Json
if ($app) {
    Write-Host "Web App already exists -- reusing."
} else {
    az webapp create `
        --resource-group $ResourceGroup `
        --plan $PlanName `
        --name $AppName `
        --runtime "PYTHON:$PythonVersion" `
        --output none
    Write-Host "Created Web App $AppName."
}

# Always On keeps the Streamlit process warm so websocket connections stay alive.
# F1/Free tier does NOT support Always On -- skip it there.
if ($PlanSku -ne 'F1') {
    az webapp config set `
        --resource-group $ResourceGroup `
        --name $AppName `
        --always-on true `
        --output none | Out-Null
    Write-Host "Enabled Always On."
}

# HTTPS-only + startup command (Streamlit via startup.sh).
az webapp update `
    --resource-group $ResourceGroup `
    --name $AppName `
    --https-only true `
    --output none | Out-Null

az webapp config set `
    --resource-group $ResourceGroup `
    --name $AppName `
    --startup-file "bash /home/site/wwwroot/startup.sh" `
    --output none | Out-Null
Write-Host "Configured startup command and HTTPS-only."

# ---------------------------------------------------------------------------
# 4. App Settings (env vars) -- seeded from .env, plus platform toggles.
# ---------------------------------------------------------------------------
Write-Section "4. App Settings"

$platformSettings = @(
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true",
    "ENABLE_ORYX_BUILD=true",
    "WEBSITES_PORT=8000",
    "PYTHONUNBUFFERED=1"
)

$envSettings = @()
if ($EnvFile -and (Test-Path $EnvFile)) {
    Write-Host "Reading env vars from $EnvFile ..."
    foreach ($line in Get-Content $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        if ($trimmed -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { continue }
        $key   = $Matches[1]
        $value = $Matches[2]
        # Strip surrounding quotes if present.
        if ($value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $envSettings += "$key=$value"
    }
    Write-Host "Parsed $($envSettings.Count) env vars from $EnvFile."
} elseif ($EnvFile) {
    Write-Warning "EnvFile '$EnvFile' not found -- skipping .env sync. Set App Settings manually in the portal."
}

$allSettings = $platformSettings + $envSettings
if ($allSettings.Count -gt 0) {
    # `--settings` accepts KEY=VALUE pairs, one per argument.
    az webapp config appsettings set `
        --resource-group $ResourceGroup `
        --name $AppName `
        --settings @allSettings `
        --output none | Out-Null
    Write-Host "Applied $($allSettings.Count) App Settings."
}

# ---------------------------------------------------------------------------
# 5. Build & push ZIP
# ---------------------------------------------------------------------------
Write-Section "5. Package & deploy code"

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$zipPath = Join-Path $env:TEMP "reg-impact-$stamp.zip"

# Files/dirs to exclude from the deployment package.
$excludeDirs = @(
    '__pycache__', '.venv', 'venv', '.git', '.github', '.idea', '.vscode',
    '.cursor', 'outputs', 'uploads', 'data'
)
$excludeFiles = @(
    '*.pyc', '*.pyo', '*.pyd', '*.db', '*.db-journal', '*.log', '*.zip',
    '.env', '.env.*'
)

Write-Host "Staging deployment payload ..."
$staging = Join-Path $env:TEMP "reg-impact-stage-$stamp"
New-Item -ItemType Directory -Path $staging -Force | Out-Null

$robocopyExcludeDirs  = $excludeDirs  | ForEach-Object { $_ }
$robocopyExcludeFiles = $excludeFiles | ForEach-Object { $_ }

# robocopy exit codes 0-7 are success; 8+ are failures. We ignore success codes.
$robocopyArgs = @(
    ".", $staging, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
    "/XD"
) + $robocopyExcludeDirs + @("/XF") + $robocopyExcludeFiles

$rc = & robocopy @robocopyArgs
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed with exit code $LASTEXITCODE while staging deployment."
}

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -Force
$zipSizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "Built $zipPath ($zipSizeMB MB)."

Write-Host "Uploading and building on Azure (this can take 3-8 minutes) ..."
az webapp deploy `
    --resource-group $ResourceGroup `
    --name $AppName `
    --src-path $zipPath `
    --type zip `
    --async false `
    --output none

Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
Write-Section "6. Done"
$hostName = (az webapp show --resource-group $ResourceGroup --name $AppName --query defaultHostName --output tsv).Trim()
Write-Host "App URL            : https://$hostName" -ForegroundColor Green
Write-Host "Log stream (Ctrl+C to stop) :"
Write-Host "    az webapp log tail --resource-group $ResourceGroup --name $AppName"
Write-Host ""
Write-Host "Redeploy code only :"
Write-Host "    .\deploy_azure.ps1 -ResourceGroup $ResourceGroup -Location $Location -PlanName $PlanName -AppName $AppName -EnvFile """""
