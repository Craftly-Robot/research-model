param(
    [string]$Pattern = "tests.test_aeitron_mvp_foundation"
)

$ErrorActionPreference = "Stop"

python -m unittest $Pattern
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python -m compileall -q src\aeitron
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Craftly MVP foundation checks passed"
