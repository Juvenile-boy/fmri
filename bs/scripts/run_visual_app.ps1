param(
  [string]$RepoRoot = "",
  [string]$PythonExe = "py",
  [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $RepoRoot = Split-Path -Path $PSScriptRoot -Parent
}

$appScript = Join-Path $RepoRoot "python\app\fmri_visualization_app.py"
if (!(Test-Path $appScript)) {
  throw "App script not found: $appScript"
}

$oldEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $PythonExe -c "import streamlit" *> $null
$importCode = $LASTEXITCODE
$ErrorActionPreference = $oldEap

if ($importCode -ne 0) {
  throw "Missing dependency: streamlit. Run: py -m pip install -r requirements_mature.txt"
}

Write-Host "Launching interactive fMRI visualization system..."
& $PythonExe -m streamlit run $appScript --server.port $Port --server.headless true
