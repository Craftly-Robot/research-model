"""Distributed Craftly crawler worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from pydantic import Field

from src.craftly.learning.data_engine import DataEngine, DataEngineConfig, PostgresFrontierStore
from src.craftly.learning.source_registry import SourceRegistry
from src.craftly.shared.schemas import StrictModel


class CrawlerWorkerConfig(StrictModel):
    sources_path: str
    postgres_dsn: str
    raw_output_dir: str
    clean_output_dir: str
    worker_id: str = "worker-0"
    batch_docs: int = Field(default=10_000, ge=1)
    loops: int = Field(default=1, ge=1)
    workers: int = Field(default=16, ge=1, le=256)
    max_depth: int = Field(default=2, ge=0, le=20)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    shard_rows: int = Field(default=10_000, ge=1)


async def run_crawler_worker(config: CrawlerWorkerConfig) -> dict[str, object]:
    registry = SourceRegistry.from_file(config.sources_path)
    registry_report = registry.validate()
    reports = []
    started = time.perf_counter()
    store = await PostgresFrontierStore.create(config.postgres_dsn, max_size=max(2, config.workers))
    engine_config = DataEngineConfig(
        frontier_backend="postgres",
        postgres_dsn=config.postgres_dsn,
        output_dir=config.raw_output_dir,
        clean_output_dir=config.clean_output_dir,
        max_docs=config.batch_docs,
        workers=config.workers,
        max_depth=config.max_depth,
        delay_seconds=config.delay_seconds,
        shard_rows=config.shard_rows,
    )
    engine = DataEngine(engine_config, store=store, owns_store=True)
    try:
        for _ in range(config.loops):
            report = await engine.run(registry.to_sources())
            reports.append(report.model_dump())
            if report.fetched == 0:
                break
    finally:
        await engine.aclose()
    return {
        "status": "complete",
        "worker_id": config.worker_id,
        "duration_ms": (time.perf_counter() - started) * 1000,
        "source_registry": registry_report.model_dump(),
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a distributed Craftly crawler worker.")
    parser.add_argument("--sources", default=os.environ.get("CRAFTLY_DATA_SOURCES", "config/data_sources.ultimate.json"))
    parser.add_argument("--postgres-dsn", default=os.environ.get("CRAFTLY_DATABASE_URL", ""))
    parser.add_argument("--raw-output-dir", default=os.environ.get("CRAFTLY_RAW_OUTPUT_DIR", "artifacts/craftly/data-engine/raw"))
    parser.add_argument("--clean-output-dir", default=os.environ.get("CRAFTLY_CLEAN_OUTPUT_DIR", "artifacts/craftly/data-engine/clean"))
    parser.add_argument("--worker-id", default=os.environ.get("CRAFTLY_WORKER_ID", "worker-0"))
    parser.add_argument("--batch-docs", type=int, default=int(os.environ.get("CRAFTLY_BATCH_DOCS", "10000")))
    parser.add_argument("--loops", type=int, default=int(os.environ.get("CRAFTLY_WORKER_LOOPS", "1")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CRAFTLY_ASYNC_WORKERS", "16")))
    parser.add_argument("--max-depth", type=int, default=int(os.environ.get("CRAFTLY_CRAWL_MAX_DEPTH", "2")))
    parser.add_argument("--delay-seconds", type=float, default=float(os.environ.get("CRAFTLY_CRAWL_DELAY_SECONDS", "1.0")))
    parser.add_argument("--shard-rows", type=int, default=int(os.environ.get("CRAFTLY_SHARD_ROWS", "10000")))
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CrawlerWorkerConfig:
    if not args.postgres_dsn:
        raise ValueError("CRAFTLY_DATABASE_URL or --postgres-dsn is required")
    return CrawlerWorkerConfig(
        sources_path=args.sources,
        postgres_dsn=args.postgres_dsn,
        raw_output_dir=args.raw_output_dir,
        clean_output_dir=args.clean_output_dir,
        worker_id=args.worker_id,
        batch_docs=args.batch_docs,
        loops=args.loops,
        workers=args.workers,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        shard_rows=args.shard_rows,
    )


def main() -> None:
    result = asyncio.run(run_crawler_worker(config_from_args(parse_args())))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


