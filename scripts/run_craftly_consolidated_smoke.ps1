param(
    [string]$Prompt = "build secure login api",
    [string]$Workspace = ".",
    [ValidateSet("strict", "development")]
    [string]$PolicyMode = "development",
    [ValidateSet("auto", "active", "mock")]
    [string]$AgentBackendMode = "mock",
    [int]$MaxAgentNodes = 2,
    [switch]$RunVerifier,
    [switch]$RunSecurity,
    [string]$OutputPath = "artifacts\craftly\consolidated-smoke.json"
)

$ErrorActionPreference = "Stop"

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Force $outputDirectory | Out-Null
}

$arguments = @(
    "-m", "src.craftly.cli",
    "--prompt", $Prompt,
    "--workspace", $Workspace,
    "--policy-mode", $PolicyMode,
    "--agent-backend-mode", $AgentBackendMode,
    "--max-agent-nodes", "$MaxAgentNodes"
)

if (-not $RunVerifier) {
    $arguments += "--no-verifier"
}

if (-not $RunSecurity) {
    $arguments += "--no-security"
}

$json = & python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
[System.IO.File]::WriteAllText($resolvedOutputPath, $json, [System.Text.UTF8Encoding]::new($false))

$report = $json | ConvertFrom-Json
$planGoal = $report.plan.goal
$routeIntent = $report.route.intent

Write-Host "Craftly consolidated smoke complete"
Write-Host "status=$($report.status)"
Write-Host "run_id=$($report.run_id)"
Write-Host "confidence=$($report.confidence)"
Write-Host "intent=$routeIntent"
Write-Host "goal=$planGoal"
Write-Host "artifact=$OutputPath"

