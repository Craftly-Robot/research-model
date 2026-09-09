"""Production readiness gate for large Craftly data-platform runs."""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlparse

from pydantic import Field

from src.craftly.db.migration_runner import load_migrations
from src.craftly.learning.benchmark_contamination_filter import load_patterns
from src.craftly.learning.source_registry import SourceRegistry
from src.craftly.shared.schemas import print_report
from src.craftly.shared.schemas import StrictModel, print_report


class ReadinessCheck(StrictModel):
    name: str
    status: str
    message: str


class DataPlatformReadinessConfig(StrictModel):
    sources_path: str
    frontier_backend: str = "sqlite"
    postgres_dsn: str | None = None
    object_store_uri: str = "local://artifacts/craftly/object-store"
    contamination_patterns_path: str | None = None
    block_contamination: bool = True
    upload_artifacts: bool = True
    production_mode: bool = False
    worker_replicas: int = Field(default=1, ge=1)
    async_workers: int = Field(default=8, ge=1)


class DataPlatformReadinessReport(StrictModel):
    status: str
    production_mode: bool
    checks: list[ReadinessCheck]

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _check(name: str, condition: bool, message: str, *, warn: bool = False) -> ReadinessCheck:
    if condition:
        return ReadinessCheck(name=name, status="pass", message=message)
    return ReadinessCheck(name=name, status="warn" if warn else "fail", message=message)


def run_readiness_check(config: DataPlatformReadinessConfig) -> DataPlatformReadinessReport:
    checks: list[ReadinessCheck] = []
    registry = SourceRegistry.from_file(config.sources_path)
    registry_report = registry.validate()
    checks.append(
        _check(
            "source_registry",
            registry_report.source_count > 0
            and registry_report.url_count > 0
            and not registry_report.warnings,
            f"{registry_report.source_count} sources, {registry_report.url_count} seed urls, warnings={len(registry_report.warnings)}",
        )
    )

    is_postgres = config.frontier_backend == "postgres" and bool(config.postgres_dsn)
    checks.append(
        _check(
            "distributed_frontier",
            is_postgres or not config.production_mode,
            "production runs require --frontier-backend postgres and a Postgres DSN",
        )
    )

    parsed_store = urlparse(config.object_store_uri)
    is_distributed_store = parsed_store.scheme == "s3"
    checks.append(
        _check(
            "object_storage",
            (config.upload_artifacts and is_distributed_store) or not config.production_mode,
            "production runs require artifact upload to S3/MinIO object storage",
        )
    )

    patterns = load_patterns(config.contamination_patterns_path)
    checks.append(
        _check(
            "contamination_gate",
            config.block_contamination and len(patterns) > 0,
            f"contamination gate enabled with {len(patterns)} benchmark/holdout patterns",
        )
    )

    migrations = load_migrations()
    has_data_platform_migration = any(
        "data_platform" in migration.version for migration in migrations
    )
    checks.append(
        _check(
            "postgres_migrations",
            has_data_platform_migration,
            "data-platform Postgres migration must be present",
        )
    )

    checks.append(
        _check(
            "worker_scale",
            (config.worker_replicas >= 2 and config.async_workers >= 16)
            or not config.production_mode,
            f"worker_replicas={config.worker_replicas}, async_workers={config.async_workers}",
            warn=not config.production_mode,
        )
    )

    has_fail = any(item.status == "fail" for item in checks)
    has_warn = any(item.status == "warn" for item in checks)
    status = "block" if has_fail else "warn" if has_warn else "pass"
    return DataPlatformReadinessReport(
        status=status, production_mode=config.production_mode, checks=checks
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Craftly data-platform production readiness."
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument("--frontier-backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--postgres-dsn")
    parser.add_argument("--object-store-uri", default="local://artifacts/craftly/object-store")
    parser.add_argument("--contamination-patterns")
    parser.add_argument("--allow-contamination-hits", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--worker-replicas", type=int, default=1)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DataPlatformReadinessConfig:
    return DataPlatformReadinessConfig(
        sources_path=args.sources,
        frontier_backend=args.frontier_backend,
        postgres_dsn=args.postgres_dsn,
        object_store_uri=args.object_store_uri,
        contamination_patterns_path=args.contamination_patterns,
        block_contamination=not args.allow_contamination_hits,
        upload_artifacts=not args.no_upload,
        production_mode=args.production,
        worker_replicas=args.worker_replicas,
        async_workers=args.async_workers,
    )


def main() -> None:
    report = run_readiness_check(config_from_args(parse_args()))
    print_report(report)
    if report.status == "block":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
