param(
    [string]$Pattern = "tests.test_craftly_mvp_foundation"
)

$ErrorActionPreference = "Stop"

python -m unittest $Pattern
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python -m compileall -q src\craftly
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Craftly MVP foundation checks passed"
