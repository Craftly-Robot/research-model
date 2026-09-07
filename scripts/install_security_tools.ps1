$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$root = Get-Location
$toolsDir = Join-Path $root "tools"
$artifactsDir = Join-Path $root "artifacts\craftly\security-tools"
New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null

$summary = [ordered]@{
    semgrep = [ordered]@{ ok = $false; mode = $null; message = $null }
    codeql = [ordered]@{ ok = $false; version = $null; path = $null; qlpacks = @(); message = $null }
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    cmd /c "docker image inspect semgrep/semgrep >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        docker pull semgrep/semgrep
    }
    $semgrepVersion = docker run --rm --entrypoint semgrep semgrep/semgrep --version
    $summary.semgrep.ok = $LASTEXITCODE -eq 0
    $summary.semgrep.mode = "docker"
    $summary.semgrep.message = ($semgrepVersion -join "`n")
} else {
    $summary.semgrep.message = "Docker is unavailable; install Semgrep with pipx or uv."
}

$codeqlExe = Join-Path $toolsDir "codeql\codeql.exe"
if (-not (Test-Path $codeqlExe)) {
    $release = Invoke-RestMethod -Headers @{ "User-Agent" = "craftly-security-tools" } -Uri "https://api.github.com/repos/github/codeql-cli-binaries/releases/latest"
    $asset = $release.assets | Where-Object { $_.name -eq "codeql-win64.zip" } | Select-Object -First 1
    $checksumAsset = $release.assets | Where-Object { $_.name -eq "codeql-win64.zip.checksum.txt" } | Select-Object -First 1
    if (-not $asset) {
        throw "Could not find codeql-win64.zip in latest CodeQL release."
    }
    $zipPath = Join-Path $toolsDir "codeql-win64.zip"
    $checksumPath = Join-Path $toolsDir "codeql-win64.zip.checksum.txt"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath
    if ($checksumAsset) {
        Invoke-WebRequest -Uri $checksumAsset.browser_download_url -OutFile $checksumPath
        $expected = ((Get-Content $checksumPath -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
        $actual = (Get-FileHash -Algorithm SHA256 $zipPath).Hash.ToLowerInvariant()
        if ($expected -ne $actual) {
            throw "CodeQL checksum mismatch. expected=$expected actual=$actual"
        }
    }
    Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
}

if (Test-Path $codeqlExe) {
    $versionJson = & $codeqlExe version --format=json
    $summary.codeql.ok = $LASTEXITCODE -eq 0
    $summary.codeql.path = $codeqlExe
    $summary.codeql.message = ($versionJson -join "`n")
    try {
        $parsedVersion = ($versionJson -join "`n") | ConvertFrom-Json
        $summary.codeql.version = $parsedVersion.version
    } catch {
        $summary.codeql.version = $null
    }
    $packs = @("codeql/python-queries", "codeql/javascript-queries", "codeql/cpp-queries")
    foreach ($pack in $packs) {
        & $codeqlExe pack download $pack | Out-Host
    }
    $resolvedPacks = & $codeqlExe resolve qlpacks --format=json --additional-packs "$env:USERPROFILE\.codeql\packages"
    $summary.codeql.qlpacks = ($resolvedPacks -join "`n")
} else {
    $summary.codeql.message = "CodeQL executable was not found after install."
}

$summaryPath = Join-Path $artifactsDir "security-tools-install.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
$summary | ConvertTo-Json -Depth 6

