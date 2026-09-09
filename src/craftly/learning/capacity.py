"""Capacity planning for billion-scale Craftly data collection."""

from __future__ import annotations

import argparse
import json
import math

from pydantic import Field

from src.craftly.shared.schemas import StrictModel


class CapacityPlanConfig(StrictModel):
    target_documents: int = Field(default=1_000_000_000, ge=1)
    avg_document_bytes: int = Field(default=64_000, ge=1)
    avg_clean_acceptance_rate: float = Field(default=0.35, gt=0.0, le=1.0)
    worker_replicas: int = Field(default=32, ge=1)
    async_workers_per_replica: int = Field(default=32, ge=1)
    docs_per_async_worker_per_minute: float = Field(default=6.0, gt=0.0)
    compression_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    target_days: float = Field(default=30.0, gt=0.0)


class CapacityPlan(StrictModel):
    target_documents: int
    raw_storage_tb: float
    compressed_storage_tb: float
    expected_clean_documents: int
    docs_per_minute_capacity: float
    estimated_days: float
    required_docs_per_second_for_target_days: float
    required_bandwidth_mbps_for_target_days: float
    recommended_worker_replicas_for_target_days: int


def build_capacity_plan(config: CapacityPlanConfig) -> CapacityPlan:
    raw_bytes = config.target_documents * config.avg_document_bytes
    compressed_bytes = raw_bytes * config.compression_ratio
    docs_per_minute = (
        config.worker_replicas
        * config.async_workers_per_replica
        * config.docs_per_async_worker_per_minute
    )
    estimated_days = config.target_documents / docs_per_minute / 60 / 24
    required_docs_per_second = config.target_documents / (config.target_days * 24 * 60 * 60)
    required_bandwidth_mbps = (required_docs_per_second * config.avg_document_bytes * 8) / 1_000_000
    docs_per_replica_per_day = (
        config.async_workers_per_replica * config.docs_per_async_worker_per_minute * 60 * 24
    )
    recommended_replicas = math.ceil(
        config.target_documents / (config.target_days * docs_per_replica_per_day)
    )
    return CapacityPlan(
        target_documents=config.target_documents,
        raw_storage_tb=round(raw_bytes / 1_000_000_000_000, 3),
        compressed_storage_tb=round(compressed_bytes / 1_000_000_000_000, 3),
        expected_clean_documents=int(config.target_documents * config.avg_clean_acceptance_rate),
        docs_per_minute_capacity=round(docs_per_minute, 3),
        estimated_days=round(estimated_days, 3),
        required_docs_per_second_for_target_days=round(required_docs_per_second, 3),
        required_bandwidth_mbps_for_target_days=round(required_bandwidth_mbps, 3),
        recommended_worker_replicas_for_target_days=recommended_replicas,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Craftly data-platform cluster capacity.")
    parser.add_argument("--target-documents", type=int, default=1_000_000_000)
    parser.add_argument("--avg-document-bytes", type=int, default=64_000)
    parser.add_argument("--acceptance-rate", type=float, default=0.35)
    parser.add_argument("--worker-replicas", type=int, default=32)
    parser.add_argument("--async-workers-per-replica", type=int, default=32)
    parser.add_argument("--docs-per-worker-minute", type=float, default=6.0)
    parser.add_argument("--compression-ratio", type=float, default=0.35)
    parser.add_argument("--target-days", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_capacity_plan(
        CapacityPlanConfig(
            target_documents=args.target_documents,
            avg_document_bytes=args.avg_document_bytes,
            avg_clean_acceptance_rate=args.acceptance_rate,
            worker_replicas=args.worker_replicas,
            async_workers_per_replica=args.async_workers_per_replica,
            docs_per_async_worker_per_minute=args.docs_per_worker_minute,
            compression_ratio=args.compression_ratio,
            target_days=args.target_days,
        )
    )
    print(json.dumps(plan.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
