param(
    [string]$Sources = "config/data_sources.production.sample.json",
    [string]$DatabaseUrl = $env:CRAFTLY_DATABASE_URL,
    [string]$ObjectStoreUri = "s3://craftly-datasets/pretraining",
    [int]$WorkerReplicas = 8,
    [int]$AsyncWorkers = 64
)

$ErrorActionPreference = "Stop"

if (-not $DatabaseUrl) {
    throw "CRAFTLY_DATABASE_URL or -DatabaseUrl is required"
}

python -m src.craftly.learning.production_check `
    --sources $Sources `
    --frontier-backend postgres `
    --postgres-dsn $DatabaseUrl `
    --object-store-uri $ObjectStoreUri `
    --production `
    --worker-replicas $WorkerReplicas `
    --async-workers $AsyncWorkers

