"""Long-running data worker supervision with heartbeat and backoff."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.production_check import DataPlatformReadinessConfig, run_readiness_check
from src.craftly.learning.worker import CrawlerWorkerConfig, run_crawler_worker
from src.craftly.shared.schemas import StrictModel


class SupervisorConfig(StrictModel):
    sources_path: str
    postgres_dsn: str
    raw_output_dir: str
    clean_output_dir: str
    object_store_uri: str = "s3://craftly-datasets/pretraining"
    heartbeat_path: str = "artifacts/craftly/supervisor/heartbeat.json"
    status_path: str = "artifacts/craftly/supervisor/status.json"
    worker_replicas: int = Field(default=1, ge=1)
    async_workers: int = Field(default=16, ge=1, le=256)
    batch_docs: int = Field(default=10_000, ge=1)
    max_cycles: int = Field(default=100, ge=1)
    sleep_seconds: float = Field(default=30.0, ge=0.0)
    max_failures: int = Field(default=5, ge=1)


class SupervisorReport(StrictModel):
    status: str
    cycles: int
    failures: int
    last_error: str | None = None
    last_worker_report: dict[str, Any] | None = None
    duration_ms: float


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


async def run_supervisor(config: SupervisorConfig) -> SupervisorReport:
    readiness = run_readiness_check(
        DataPlatformReadinessConfig(
            sources_path=config.sources_path,
            frontier_backend="postgres",
            postgres_dsn=config.postgres_dsn,
            object_store_uri=config.object_store_uri,
            production_mode=True,
            worker_replicas=config.worker_replicas,
            async_workers=config.async_workers,
        )
    )
    if readiness.status == "block":
        raise RuntimeError(f"data platform readiness blocked supervisor: {readiness.model_dump()}")

    started = time.perf_counter()
    failures = 0
    last_error: str | None = None
    last_worker_report: dict[str, Any] | None = None
    for cycle in range(1, config.max_cycles + 1):
        _write_json(
            config.heartbeat_path,
            {
                "status": "running",
                "cycle": cycle,
                "failures": failures,
                "updated_at_unix": time.time(),
            },
        )
        try:
            worker_report = await run_crawler_worker(
                CrawlerWorkerConfig(
                    sources_path=config.sources_path,
                    postgres_dsn=config.postgres_dsn,
                    raw_output_dir=config.raw_output_dir,
                    clean_output_dir=config.clean_output_dir,
                    worker_id=f"supervisor-{os.getpid()}-{cycle}",
                    batch_docs=config.batch_docs,
                    loops=1,
                    workers=config.async_workers,
                )
            )
            last_worker_report = worker_report
            last_error = None
            fetched = sum(int(item.get("fetched", 0)) for item in worker_report.get("reports", []))
            if fetched == 0:
                break
        except Exception as exc:
            failures += 1
            last_error = str(exc)
            if failures >= config.max_failures:
                break
        await asyncio.sleep(config.sleep_seconds * max(1, failures + 1))

    status = "failed" if failures >= config.max_failures else "complete"
    report = SupervisorReport(
        status=status,
        cycles=cycle,
        failures=failures,
        last_error=last_error,
        last_worker_report=last_worker_report,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    _write_json(config.status_path, report.model_dump())
    _write_json(
        config.heartbeat_path,
        {
            "status": status,
            "cycle": report.cycles,
            "failures": report.failures,
            "updated_at_unix": time.time(),
        },
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised Craftly data crawl cycles.")
    parser.add_argument("--sources", default=os.environ.get("CRAFTLY_DATA_SOURCES", "config/data_sources.ultimate.json"))
    parser.add_argument("--postgres-dsn", default=os.environ.get("CRAFTLY_DATABASE_URL", ""))
    parser.add_argument("--raw-output-dir", default=os.environ.get("CRAFTLY_RAW_OUTPUT_DIR", "artifacts/craftly/data-engine/raw"))
    parser.add_argument("--clean-output-dir", default=os.environ.get("CRAFTLY_CLEAN_OUTPUT_DIR", "artifacts/craftly/data-engine/clean"))
    parser.add_argument("--object-store-uri", default=os.environ.get("CRAFTLY_OBJECT_STORE_URI", "s3://craftly-datasets/pretraining"))
    parser.add_argument("--heartbeat-path", default=os.environ.get("CRAFTLY_HEARTBEAT_PATH", "artifacts/craftly/supervisor/heartbeat.json"))
    parser.add_argument("--status-path", default=os.environ.get("CRAFTLY_STATUS_PATH", "artifacts/craftly/supervisor/status.json"))
    parser.add_argument("--worker-replicas", type=int, default=int(os.environ.get("CRAFTLY_WORKER_REPLICAS", "1")))
    parser.add_argument("--async-workers", type=int, default=int(os.environ.get("CRAFTLY_ASYNC_WORKERS", "16")))
    parser.add_argument("--batch-docs", type=int, default=int(os.environ.get("CRAFTLY_BATCH_DOCS", "10000")))
    parser.add_argument("--max-cycles", type=int, default=int(os.environ.get("CRAFTLY_SUPERVISOR_MAX_CYCLES", "100")))
    parser.add_argument("--sleep-seconds", type=float, default=float(os.environ.get("CRAFTLY_SUPERVISOR_SLEEP_SECONDS", "30")))
    parser.add_argument("--max-failures", type=int, default=int(os.environ.get("CRAFTLY_SUPERVISOR_MAX_FAILURES", "5")))
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> SupervisorConfig:
    if not args.postgres_dsn:
        raise ValueError("CRAFTLY_DATABASE_URL or --postgres-dsn is required")
    return SupervisorConfig(
        sources_path=args.sources,
        postgres_dsn=args.postgres_dsn,
        raw_output_dir=args.raw_output_dir,
        clean_output_dir=args.clean_output_dir,
        object_store_uri=args.object_store_uri,
        heartbeat_path=args.heartbeat_path,
        status_path=args.status_path,
        worker_replicas=args.worker_replicas,
        async_workers=args.async_workers,
        batch_docs=args.batch_docs,
        max_cycles=args.max_cycles,
        sleep_seconds=args.sleep_seconds,
        max_failures=args.max_failures,
    )


def main() -> None:
    report = asyncio.run(run_supervisor(_config_from_args(_parse_args())))
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

