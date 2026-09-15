param(
    [string]$OpenSearchHome = $env:OPENSEARCH_HOME,
    [switch]$OpenSearchOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenSearchUrl = "http://127.0.0.1:9200"
$OpenSearchBatch = Join-Path $OpenSearchHome "bin\opensearch.bat"
$BundledJava = Join-Path $OpenSearchHome "jdk"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Test-OpenSearchHealth {
    & curl.exe --silent --fail --noproxy "*" --max-time 2 $OpenSearchUrl *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $OpenSearchBatch)) {
    throw "OpenSearch was not found at $OpenSearchHome"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "The project virtual environment was not found at $Python"
}

if (-not (Test-OpenSearchHealth)) {
    $env:OPENSEARCH_JAVA_HOME = $BundledJava
    Start-Process `
        -FilePath $OpenSearchBatch `
        -ArgumentList @(
            "-Ediscovery.type=single-node",
            "-Eplugins.security.disabled=true",
            "-Enetwork.host=127.0.0.1"
        ) `
        -WorkingDirectory $OpenSearchHome `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $ProjectRoot "opensearch-local.stdout.log") `
        -RedirectStandardError (Join-Path $ProjectRoot "opensearch-local.stderr.log") | Out-Null

    Write-Host "Starting OpenSearch from $OpenSearchHome ..."
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 2
        if (Test-OpenSearchHealth) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "OpenSearch did not become healthy. Check opensearch-local.stdout.log."
    }
}

$env:OPENSEARCH_URL = $OpenSearchUrl
$env:OPENSEARCH_VERIFY_CERTS = "false"

Write-Host "OpenSearch is healthy at $OpenSearchUrl"
if ($OpenSearchOnly) {
    return
}

Write-Host "Starting Streamlit at http://127.0.0.1:8501"
& $Python -m streamlit run (Join-Path $ProjectRoot "app2.py")
