"""Generate an executable first-run plan for serious Craftly data collection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.capacity import CapacityPlanConfig, build_capacity_plan
from src.craftly.learning.production_check import DataPlatformReadinessConfig, run_readiness_check
from src.craftly.learning.resource_catalog import ResourceCatalogReport, write_resource_catalog_report
from src.craftly.learning.source_registry import SourceRegistry
from src.craftly.shared.schemas import StrictModel


class DataRunPlanConfig(StrictModel):
    source_paths: list[str]
    output_dir: str = "artifacts/craftly/data-runs/first-serious-run"
    merged_registry_path: str | None = None
    dataset_id: str = "craftly-defensive-coding-corpus"
    target_documents: int = Field(default=1_000_000, ge=1)
    target_days: float = Field(default=7.0, gt=0.0)
    frontier_backend: str = "postgres"
    postgres_dsn: str = "postgresql://user:pass@postgres:5432/craftly"
    object_store_uri: str = "s3://craftly-datasets/pretraining"
    object_store_endpoint_url: str | None = None
    worker_replicas: int = Field(default=8, ge=1)
    async_workers: int = Field(default=64, ge=1)
    max_depth: int = Field(default=2, ge=0, le=20)
    sequence_length: int = Field(default=2048, ge=16)
    skip_train: bool = True


class DataRunPlan(StrictModel):
    status: str
    created_at_unix: float = Field(default_factory=time.time)
    output_dir: str
    merged_registry_path: str
    source_registry: dict[str, Any]
    readiness: dict[str, Any]
    capacity: dict[str, Any]
    resource_catalog: dict[str, Any] | None = None
    resource_catalog_path: str | None = None
    commands: dict[str, str]


def build_data_run_plan(config: DataRunPlanConfig) -> DataRunPlan:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry.from_files(config.source_paths)
    registry_report = registry.validate()
    merged_registry_path = Path(config.merged_registry_path) if config.merged_registry_path else root / "sources.merged.json"
    registry.write(merged_registry_path)
    resource_catalog: ResourceCatalogReport | None = None
    resource_catalog_path = root / "resource_catalog_report.json"
    if len(config.source_paths) == 1:
        resource_catalog = write_resource_catalog_report(config.source_paths[0], resource_catalog_path)

    readiness = run_readiness_check(
        DataPlatformReadinessConfig(
            sources_path=str(merged_registry_path),
            frontier_backend=config.frontier_backend,
            postgres_dsn=config.postgres_dsn,
            object_store_uri=config.object_store_uri,
            production_mode=True,
            worker_replicas=config.worker_replicas,
            async_workers=config.async_workers,
        )
    )
    capacity = build_capacity_plan(
        CapacityPlanConfig(
            target_documents=config.target_documents,
            target_days=config.target_days,
            worker_replicas=config.worker_replicas,
            async_workers_per_replica=config.async_workers,
        )
    )
    skip_train = " --skip-train" if config.skip_train else ""
    endpoint = f" --object-store-endpoint-url {config.object_store_endpoint_url}" if config.object_store_endpoint_url else ""
    commands = {
        "readiness": (
            "python -m src.craftly.learning.production_check "
            f"--sources {merged_registry_path} --frontier-backend {config.frontier_backend} "
            f"--postgres-dsn {config.postgres_dsn} --object-store-uri {config.object_store_uri} "
            f"--production --worker-replicas {config.worker_replicas} --async-workers {config.async_workers}"
        ),
        "pipeline": (
            "python -m src.craftly.learning.data_pipeline "
            f"--sources {merged_registry_path} --dataset-id {config.dataset_id} --work-dir {root / 'pipeline'} "
            f"--frontier-backend {config.frontier_backend} --postgres-dsn {config.postgres_dsn} "
            f"--object-store-uri {config.object_store_uri}{endpoint} --max-docs {config.target_documents} "
            f"--workers {config.async_workers} --max-depth {config.max_depth} --sequence-length {config.sequence_length}{skip_train}"
        ),
        "capacity": (
            "python -m src.craftly.learning.capacity "
            f"--target-documents {config.target_documents} --target-days {config.target_days} "
            f"--worker-replicas {config.worker_replicas} --async-workers-per-replica {config.async_workers}"
        ),
        "resource_catalog": (
            "python -m src.craftly.learning.resource_catalog "
            f"--catalog {config.source_paths[0]} --output {resource_catalog_path}"
        ),
    }
    plan = DataRunPlan(
        status="ready" if readiness.status == "pass" else "blocked",
        output_dir=str(root),
        merged_registry_path=str(merged_registry_path),
        source_registry=registry_report.model_dump(),
        readiness=readiness.model_dump(),
        capacity=capacity.model_dump(),
        resource_catalog=resource_catalog.model_dump() if resource_catalog else None,
        resource_catalog_path=str(resource_catalog_path) if resource_catalog else None,
        commands=commands,
    )
    (root / "run_plan.json").write_text(json.dumps(plan.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    (root / "commands.ps1").write_text("\n".join(commands.values()) + "\n", encoding="utf-8")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a serious Craftly data collection run.")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output-dir", default="artifacts/craftly/data-runs/first-serious-run")
    parser.add_argument("--dataset-id", default="craftly-defensive-coding-corpus")
    parser.add_argument("--target-documents", type=int, default=1_000_000)
    parser.add_argument("--target-days", type=float, default=7.0)
    parser.add_argument("--frontier-backend", default="postgres", choices=["sqlite", "postgres"])
    parser.add_argument("--postgres-dsn", default="postgresql://user:pass@postgres:5432/craftly")
    parser.add_argument("--object-store-uri", default="s3://craftly-datasets/pretraining")
    parser.add_argument("--object-store-endpoint-url")
    parser.add_argument("--worker-replicas", type=int, default=8)
    parser.add_argument("--async-workers", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DataRunPlanConfig:
    return DataRunPlanConfig(
        source_paths=args.sources,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        target_documents=args.target_documents,
        target_days=args.target_days,
        frontier_backend=args.frontier_backend,
        postgres_dsn=args.postgres_dsn,
        object_store_uri=args.object_store_uri,
        object_store_endpoint_url=args.object_store_endpoint_url,
        worker_replicas=args.worker_replicas,
        async_workers=args.async_workers,
        max_depth=args.max_depth,
        sequence_length=args.sequence_length,
        skip_train=not args.train,
    )


def main() -> None:
    plan = build_data_run_plan(config_from_args(parse_args()))
    print(json.dumps(plan.model_dump(), indent=2, sort_keys=True))
    if plan.status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

