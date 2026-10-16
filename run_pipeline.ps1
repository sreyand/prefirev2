param(
    [switch]$SkipAcquisition,
    [switch]$SkipWeatherAudit,
    [switch]$SkipOperationalData
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install requirements.txt first.'
}

function Invoke-PipelineScript {
    param([Parameter(Mandatory = $true)][string]$Name)
    $taskScriptPath = Join-Path (Join-Path $PSScriptRoot 'scripts') $Name
    if (-not (Test-Path -LiteralPath $taskScriptPath)) {
        throw "Required pipeline script is missing: $Name"
    }
    Write-Output "Running $Name"
    & $taskPython $taskScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed in $Name. Cached source downloads are retained."
    }
}

function Confirm-CachedReport {
    param([string]$Report, [string]$SwitchName)
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $Report))) {
        throw "$SwitchName requires an existing $Report from the same source/cohort version. Run without that switch to build it."
    }
    Write-Output "Keeping cached $Report ($SwitchName)"
}

if (-not $SkipAcquisition) {
    & $taskPython scripts\acquire_data.py --fire-zip data\raw\S_USA.Fire_FPA_FOD_7th_Fires.gdb.zip --nifc-csv data\raw\nifc_suppression_costs.csv
    if ($LASTEXITCODE -ne 0) {
        throw 'Data acquisition failed. Cached downloads are retained.'
    }
}

foreach ($taskScript in @('prepare_features.py', 'validate_data.py', 'audit_reporting.py')) {
    Invoke-PipelineScript -Name $taskScript
}

if ($SkipWeatherAudit) {
    Confirm-CachedReport -Report 'reports\weather_audit.json' -SwitchName '-SkipWeatherAudit'
} else {
    Invoke-PipelineScript -Name 'audit_weather.py'
}

if ($SkipOperationalData) {
    Confirm-CachedReport -Report 'reports\data_upgrade.json' -SwitchName '-SkipOperationalData'
} else {
    Invoke-PipelineScript -Name 'acquire_operational_data.py'
}

foreach ($taskScript in @(
    'build_dashboard_summary.py',
    'train_risk.py',
    'analyze_load.py',
    'audit_model_support.py',
    'calibrate_risk.py',
    'paper_benchmark.py',
    'audit_alert_precision.py',
    'check_app.py'
)) {
    Invoke-PipelineScript -Name $taskScript
}

Write-Output 'Pipeline complete. Run .\launch.ps1 to open the workbench.'
