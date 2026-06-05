param(
  [string]$RepoRoot = "",
  [string]$MatlabExe = "matlab",
  [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $RepoRoot = Split-Path -Path $PSScriptRoot -Parent
}

$configPath = Join-Path $RepoRoot "configs\study_config.json"
$subjectsPath = Join-Path $RepoRoot "configs\subjects.tsv"

if (!(Test-Path $configPath)) {
  throw "Missing $configPath. Copy from configs\study_config.example.json"
}
if (!(Test-Path $subjectsPath)) {
  throw "Missing $subjectsPath. Copy from configs\subjects.example.tsv"
}

$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($cfg.spm_dir) -or !(Test-Path $cfg.spm_dir)) {
  throw "Invalid spm_dir in config: $($cfg.spm_dir)"
}
if ([string]::IsNullOrWhiteSpace($cfg.derivatives_root)) {
  throw "Invalid derivatives_root in config."
}
if (!(Test-Path $cfg.derivatives_root)) {
  New-Item -ItemType Directory -Path $cfg.derivatives_root -Force | Out-Null
}

Write-Host "Step 1/2: MATLAB preprocessing..."
$runFile = (Join-Path $RepoRoot "matlab\preproc\run_preproc_batch.m").Replace("\", "/")
$matlabCmd = "run('$runFile');exit;"
& $MatlabExe -batch $matlabCmd
if ($LASTEXITCODE -ne 0) {
  throw "MATLAB preprocessing failed (exit code: $LASTEXITCODE)."
}

Write-Host "Step 2/2: Python QC report..."
$derivativesRoot = $cfg.derivatives_root
$qcCsv = Join-Path $derivativesRoot "qc\preproc_qc_metrics.csv"
$statusCsv = Join-Path $derivativesRoot "logs\preproc_status.csv"
$outHtml = Join-Path $derivativesRoot "qc\qc_report.html"
$reportScript = Join-Path $RepoRoot "python\report\generate_qc_report.py"

& $PythonExe $reportScript --qc-csv $qcCsv --status-csv $statusCsv --out-html $outHtml
if ($LASTEXITCODE -ne 0) {
  throw "Python QC report generation failed (exit code: $LASTEXITCODE)."
}

Write-Host "Done. Report: $outHtml"
