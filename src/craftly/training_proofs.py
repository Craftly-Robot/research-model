"""Measured production-proof harness for the Craftly training workspace.

The harness never converts a missing external dependency into a pass. Local
Docker can prove the Postgres, Redis, and MinIO lifecycle. Kubernetes, Slurm,
multi-node GPU, long soak, and 60B resume proofs require their real targets.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed proof commands, no shell execution
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field

from src.craftly.db.migration_runner import apply_migrations
from src.craftly.learning.storage import ObjectStoreConfig, S3ObjectStore, verify_object_store_lifecycle
from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file
from src.craftly.training_workspace import (
    JobStatus,
    PostgresTrainingStore,
    QualificationCampaignRegistry,
    RedisEventBus,
    SchedulerAdapter,
    SchedulerBinding,
    TrainingAttempt,
    TrainingController,
    TrainingJob,
    TrainingEventBatch,
    TrainingEventInput,
    TrainingJobCreateRequest,
    TrainingProfileRegistry,
    TrainingWorkspaceService,
)


ProofStatus = Literal["passed", "failed", "blocked"]


class ProofResult(StrictModel):
    name: str
    status: ProofStatus
    started_at: str
    duration_seconds: float = Field(ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)


class ProductionProofReport(StrictModel):
    schema_version: int = 1
    generated_at: str
    environment: dict[str, Any]
    proofs: list[ProofResult]
    passed: int
    failed: int
    blocked: int
    status: ProofStatus


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_evidence(argv: list[str], *, timeout: int = 30) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"required executable is missing: {argv[0]}")
    completed = subprocess.run(  # nosec B603 - argv is internally constructed
        [executable, *argv[1:]],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


async def infrastructure_lifecycle_proof(
    *,
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    output_dir: Path,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    service: TrainingWorkspaceService | None = None
    try:
        import boto3

        bucket = urlparse(object_store_uri).netloc
        s3_client = boto3.client("s3", endpoint_url=object_store_endpoint_url)
        try:
            await asyncio.to_thread(s3_client.head_bucket, Bucket=bucket)
        except Exception:
            await asyncio.to_thread(s3_client.create_bucket, Bucket=bucket)
        migration = await apply_migrations(database_url)
        object_report = await asyncio.to_thread(
            verify_object_store_lifecycle,
            config=ObjectStoreConfig(uri=object_store_uri, endpoint_url=object_store_endpoint_url),
            work_dir=output_dir / "object-lifecycle",
            key=f"proofs/{uuid.uuid4()}/lifecycle.json",
        )
        store = PostgresTrainingStore(database_url)
        events = RedisEventBus(redis_url, retention_events=2_000_000)
        object_store = S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url)
        service = TrainingWorkspaceService(
            store=store,
            events=events,
            object_store=object_store,
            profiles=TrainingProfileRegistry.from_file(),
            campaigns=QualificationCampaignRegistry.from_file(),
        )
        owner = f"proof-{uuid.uuid4()}"
        from src.craftly.training_workspace import TrainingJobCreateRequest

        job = await service.create_job(
            TrainingJobCreateRequest(
                profile_id="defensive-1k",
                idempotency_key=f"proof-{uuid.uuid4()}",
                git_commit="abcdef1",
                container_digest="sha256:" + "a" * 64,
                metadata={"proof": "postgres-redis-minio-lifecycle"},
            ),
            owner_id=owner,
        )
        running, attempt = await service.claim_notebook_job(job.job_id)
        accepted = await service.ingest_events(
            running.job_id,
            TrainingEventBatch(
                attempt_id=attempt.attempt_id,
                events=[
                    TrainingEventInput(
                        source_sequence=1,
                        kind="heartbeat",
                        stage="proof",
                        status="running",
                        payload={"probe": True},
                    )
                ],
            ),
        )
        replay = await service.ingest_events(
            running.job_id,
            TrainingEventBatch(
                attempt_id=attempt.attempt_id,
                events=[
                    TrainingEventInput(
                        source_sequence=1,
                        kind="heartbeat",
                        stage="proof",
                        status="running",
                    )
                ],
            ),
        )
        persisted_events = await events.read(job.job_id, after_sequence=0, limit=10)
        cancelled = await service.cancel_job(job.job_id, actor_id="proof-runner")
        import asyncpg

        connection = await asyncpg.connect(database_url)
        try:
            database_evidence = {
                "job_rows": await connection.fetchval("SELECT count(*) FROM training_jobs WHERE id=$1", uuid.UUID(job.job_id)),
                "attempt_rows": await connection.fetchval("SELECT count(*) FROM training_attempts WHERE job_id=$1", uuid.UUID(job.job_id)),
                "ingress_rows": await connection.fetchval(
                    "SELECT count(*) FROM training_event_ingress WHERE attempt_id=$1", uuid.UUID(attempt.attempt_id)
                ),
            }
        finally:
            await connection.close()
        passed = (
            object_report.status == "passed"
            and len(accepted) == 1
            and not replay
            and len(persisted_events) == 1
            and cancelled.status == JobStatus.CANCELLED
            and all(value == 1 for value in database_evidence.values())
        )
        return ProofResult(
            name="postgres_redis_minio_lifecycle",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence={
                "migration": migration,
                "object_lifecycle": object_report.model_dump(mode="json"),
                "job_id": job.job_id,
                "attempt_id": attempt.attempt_id,
                "redis_event_count": len(persisted_events),
                "duplicate_event_rejected": not replay,
                "database": database_evidence,
            },
        )
    except Exception as exc:
        return ProofResult(
            name="postgres_redis_minio_lifecycle",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if service is not None:
            await service.close()


async def million_event_proof(
    *,
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    event_count: int,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    service: TrainingWorkspaceService | None = None
    try:
        store = PostgresTrainingStore(database_url)
        events = RedisEventBus(redis_url, retention_events=max(1_100_000, event_count + 1000))
        service = TrainingWorkspaceService(
            store=store,
            events=events,
            object_store=S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url),
        )
        from src.craftly.training_workspace import TrainingJobCreateRequest

        job = await service.create_job(
            TrainingJobCreateRequest(
                profile_id="defensive-1k",
                idempotency_key=f"event-proof-{uuid.uuid4()}",
                git_commit="abcdef1",
                container_digest="sha256:" + "b" * 64,
            ),
            owner_id=f"event-proof-{uuid.uuid4()}",
        )
        running, attempt = await service.claim_notebook_job(job.job_id)
        batch_size = 100
        for offset in range(0, event_count, batch_size):
            size = min(batch_size, event_count - offset)
            batch = TrainingEventBatch(
                attempt_id=attempt.attempt_id,
                events=[
                    TrainingEventInput(
                        source_sequence=offset + index + 1,
                        kind="metric",
                        stage="stress",
                        status="running",
                        step=offset + index + 1,
                        max_steps=event_count,
                        loss=1.0,
                    )
                    for index in range(size)
                ],
            )
            accepted = await service.ingest_events(running.job_id, batch)
            if len(accepted) != size:
                raise RuntimeError(f"event batch loss at offset {offset}: accepted={len(accepted)}, expected={size}")
        final_job = await service.get_job(job.job_id)
        tail = await events.read(job.job_id, after_sequence=max(0, event_count - 1), limit=2)
        duration = time.perf_counter() - started
        passed = final_job.event_sequence == event_count and len(tail) == 1 and tail[0].sequence == event_count
        await service.cancel_job(job.job_id, actor_id="proof-runner")
        return ProofResult(
            name=f"ordered_event_stress_{event_count}",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=duration,
            evidence={
                "job_id": job.job_id,
                "event_count": event_count,
                "final_sequence": final_job.event_sequence,
                "events_per_second": event_count / max(duration, 1e-9),
                "tail_sequence": tail[0].sequence if tail else None,
            },
        )
    except Exception as exc:
        return ProofResult(
            name=f"ordered_event_stress_{event_count}",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if service is not None:
            await service.close()


async def soak_proof(
    *,
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    duration_seconds: int,
    interval_seconds: int,
    minimum_availability: float = 0.999,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    checks = 0
    failures: list[str] = []
    latencies: dict[str, list[float]] = {"postgres": [], "redis": [], "object_store": [], "cycle": []}
    consecutive_failures = 0
    maximum_consecutive_failures = 0
    try:
        import asyncpg
        import redis.asyncio as redis

        redis_client = redis.from_url(redis_url, decode_responses=True)
        object_store = S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url)
        deadline = time.monotonic() + duration_seconds
        with tempfile.TemporaryDirectory(prefix="craftly-soak-") as temp_dir:
            download_path = Path(temp_dir) / "roundtrip.json"
            while time.monotonic() < deadline:
                cycle_started = time.perf_counter()
                audit_id = uuid.uuid4()
                redis_key = f"craftly:proof:soak:{audit_id}"
                object_key = f"proofs/soak/{audit_id}.json"
                try:
                    operation_started = time.perf_counter()
                    connection = await asyncpg.connect(database_url)
                    try:
                        async with connection.transaction():
                            await connection.execute(
                                """
                                INSERT INTO training_audit_events(id,actor_id,action,outcome,metadata)
                                VALUES($1,'soak-proof','infrastructure.soak','accepted',$2::jsonb)
                                """,
                                audit_id,
                                json.dumps({"started_at": started_at}),
                            )
                            stored = await connection.fetchval(
                                "SELECT count(*) FROM training_audit_events WHERE id=$1", audit_id
                            )
                            if stored != 1:
                                raise RuntimeError("Postgres soak transaction was not readable")
                            await connection.execute("DELETE FROM training_audit_events WHERE id=$1", audit_id)
                    finally:
                        await connection.close()
                    latencies["postgres"].append(time.perf_counter() - operation_started)

                    operation_started = time.perf_counter()
                    await redis_client.set(redis_key, str(audit_id), ex=max(60, interval_seconds * 4))
                    if await redis_client.get(redis_key) != str(audit_id):
                        raise RuntimeError("Redis soak round-trip mismatch")
                    await redis_client.delete(redis_key)
                    latencies["redis"].append(time.perf_counter() - operation_started)

                    operation_started = time.perf_counter()
                    payload = {"proof_id": str(audit_id), "timestamp": now_iso()}
                    stored_object = await asyncio.to_thread(object_store.put_json, payload, key=object_key)
                    await asyncio.to_thread(object_store.get_file, object_key, download_path)
                    if sha256_file(download_path) != stored_object.sha256:
                        raise RuntimeError("object-store soak checksum mismatch")
                    await asyncio.to_thread(object_store.delete, object_key)
                    download_path.unlink(missing_ok=True)
                    latencies["object_store"].append(time.perf_counter() - operation_started)
                    checks += 1
                    consecutive_failures = 0
                except Exception as exc:
                    failures.append(f"{now_iso()} {type(exc).__name__}: {exc}")
                    consecutive_failures += 1
                    maximum_consecutive_failures = max(maximum_consecutive_failures, consecutive_failures)
                    with contextlib.suppress(Exception):
                        await redis_client.delete(redis_key)
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(object_store.delete, object_key)
                latencies["cycle"].append(time.perf_counter() - cycle_started)
                await asyncio.sleep(min(interval_seconds, max(0.0, deadline - time.monotonic())))
        await redis_client.aclose()
        measured = time.perf_counter() - started
        full_duration = measured >= duration_seconds * 0.99
        total_checks = checks + len(failures)
        availability = checks / max(1, total_checks)
        latency_report = {
            name: {
                "p50_ms": round(percentile(values, 0.50) * 1000, 3),
                "p95_ms": round(percentile(values, 0.95) * 1000, 3),
                "p99_ms": round(percentile(values, 0.99) * 1000, 3),
                "max_ms": round(max(values, default=0.0) * 1000, 3),
            }
            for name, values in latencies.items()
        }
        passed = full_duration and checks > 0 and availability >= minimum_availability and maximum_consecutive_failures <= 1
        return ProofResult(
            name=f"infrastructure_soak_{duration_seconds}s",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=measured,
            evidence={
                "requested_seconds": duration_seconds,
                "successful_transactions": checks,
                "failed_transactions": len(failures),
                "availability": availability,
                "minimum_availability": minimum_availability,
                "maximum_consecutive_failures": maximum_consecutive_failures,
                "latency": latency_report,
            },
            blockers=failures[-20:],
        )
    except Exception as exc:
        return ProofResult(
            name=f"infrastructure_soak_{duration_seconds}s",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )


def external_scheduler_proofs() -> list[ProofResult]:
    results: list[ProofResult] = []
    started_at = now_iso()
    started = time.perf_counter()
    try:
        cluster = command_evidence(["kubectl", "cluster-info"], timeout=20)
        nodes = command_evidence(["kubectl", "get", "nodes", "-o", "json"], timeout=20)
        crd = command_evidence(["kubectl", "get", "crd", "pytorchjobs.kubeflow.org", "-o", "name"], timeout=20)
        passed = all(item["exit_code"] == 0 for item in (cluster, nodes, crd))
        results.append(
            ProofResult(
                name="kubernetes_pytorchjob_cluster",
                status="passed" if passed else "blocked",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
                evidence={"cluster": cluster, "nodes": nodes, "pytorchjob_crd": crd},
                blockers=[] if passed else ["reachable Kubernetes cluster with Kubeflow PyTorchJob CRD is required"],
            )
        )
    except Exception as exc:
        results.append(
            ProofResult(
                name="kubernetes_pytorchjob_cluster",
                status="blocked",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
                blockers=[f"{type(exc).__name__}: {exc}"],
            )
        )

    started_at = now_iso()
    started = time.perf_counter()
    try:
        sinfo = command_evidence(["sinfo", "--json"], timeout=20)
        squeue = command_evidence(["squeue", "--json"], timeout=20)
        passed = sinfo["exit_code"] == 0 and squeue["exit_code"] == 0
        results.append(
            ProofResult(
                name="slurm_cluster",
                status="passed" if passed else "blocked",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
                evidence={"sinfo": sinfo, "squeue": squeue},
                blockers=[] if passed else ["reachable Slurm controller and authenticated account are required"],
            )
        )
    except Exception as exc:
        results.append(
            ProofResult(
                name="slurm_cluster",
                status="blocked",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
                blockers=[f"{type(exc).__name__}: {exc}"],
            )
        )
    return results


async def live_scheduler_execution_proof(
    *,
    profile_id: str,
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    tokenizer_uri: str,
    tokenizer_sha256: str,
    git_commit: str,
    container_digest: str,
    timeout_seconds: int,
    poll_seconds: int,
    inject_worker_loss: bool = False,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    service: TrainingWorkspaceService | None = None
    controller: TrainingController | None = None
    job: TrainingJob | None = None
    failure_injected = False
    injection_sequence = 0
    megatron_checkpoint_files = 0
    try:
        service = TrainingWorkspaceService(
            store=PostgresTrainingStore(database_url),
            events=RedisEventBus(redis_url, retention_events=2_000_000),
            object_store=S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url),
            profiles=TrainingProfileRegistry.from_file(),
            campaigns=QualificationCampaignRegistry.from_file(),
            production_mode=True,
        )
        profile = service.profiles.latest(profile_id)
        if profile.scheduler not in {"kubernetes", "kubernetes_pytorch", "slurm"}:
            raise ValueError("live scheduler proof requires a cluster scheduler profile")
        request = TrainingJobCreateRequest(
            profile_id=profile_id,
            idempotency_key=f"live-proof-{profile_id}-{uuid.uuid4().hex[:16]}",
            dataset_manifest_uri=dataset_manifest_uri,
            dataset_manifest_sha256=dataset_manifest_sha256,
            tokenizer_uri=tokenizer_uri,
            tokenizer_sha256=tokenizer_sha256,
            git_commit=git_commit,
            container_digest=container_digest,
            metadata={"dataset_promotion": "promoted", "proof": "live-scheduler-execution"},
        )
        job = await service.create_job(request, owner_id="production-proof")
        controller = TrainingController(service)
        deadline = time.monotonic() + timeout_seconds
        observed: list[str] = []
        while time.monotonic() < deadline:
            job = await controller.reconcile_job(job)
            if not observed or observed[-1] != job.status.value:
                observed.append(job.status.value)
            if job.status.value in {"succeeded", "failed", "blocked", "cancelled"}:
                break
            if inject_worker_loss and not failure_injected and job.status == JobStatus.RUNNING:
                committed = await service.list_checkpoints(job.job_id)
                checkpoint_ready = bool(committed and any(item.reload_verified for item in committed))
                if profile.distributed_strategy == "megatron":
                    shared_root_value = os.environ.get("CRAFTLY_SHARED_CHECKPOINT_ROOT", "")
                    checkpoint_root = Path(shared_root_value).expanduser().resolve() / "jobs" / job.job_id / "training" / "checkpoints"
                    megatron_checkpoint_files = sum(1 for item in checkpoint_root.rglob("*") if item.is_file()) if checkpoint_root.is_dir() else 0
                    checkpoint_ready = megatron_checkpoint_files > 1
                if checkpoint_ready:
                    binding = SchedulerBinding.model_validate(job.scheduler_binding)
                    if profile.scheduler in {"kubernetes", "kubernetes_pytorch"}:
                        pods = command_evidence(
                            [
                                "kubectl",
                                "get",
                                "pods",
                                "-n",
                                binding.namespace or profile.scheduler_policy.namespace,
                                "-l",
                                f"job-id={job.job_id}",
                                "-o",
                                "json",
                            ],
                            timeout=30,
                        )
                        if pods["exit_code"] != 0:
                            raise RuntimeError("Kubernetes worker inventory failed before chaos injection")
                        pod_payload = json.loads(pods["stdout"])
                        candidates = [
                            item["metadata"]["name"]
                            for item in pod_payload.get("items", [])
                            if item.get("status", {}).get("phase") == "Running"
                        ]
                        if not candidates:
                            raise RuntimeError("no running Kubernetes worker pod is available for chaos injection")
                        deleted = command_evidence(
                            [
                                "kubectl",
                                "delete",
                                "pod",
                                sorted(candidates)[-1],
                                "-n",
                                binding.namespace or profile.scheduler_policy.namespace,
                                "--grace-period=0",
                                "--wait=false",
                            ],
                            timeout=30,
                        )
                        if deleted["exit_code"] != 0:
                            raise RuntimeError("Kubernetes worker deletion failed")
                    else:
                        requeued = command_evidence(["scontrol", "requeue", binding.external_id], timeout=30)
                        if requeued["exit_code"] != 0:
                            raise RuntimeError("Slurm job requeue failed during chaos injection")
                    current = await service.get_job(job.job_id)
                    injection_sequence = current.event_sequence
                    failure_injected = True
            await asyncio.sleep(poll_seconds)
        else:
            await service.cancel_job(
                job.job_id,
                scheduler=controller.schedulers.get(job.spec.scheduler),
                actor_id="production-proof-timeout",
            )
            raise TimeoutError(f"live scheduler proof exceeded {timeout_seconds} seconds")
        events = []
        cursor = 0
        while True:
            page = await service.events.read(job.job_id, after_sequence=cursor, limit=1000)
            if not page:
                break
            events.extend(page)
            cursor = page[-1].sequence
        checkpoints = await service.list_checkpoints(job.job_id)
        attempts = await service.store.list_attempts(job.job_id)
        if profile.distributed_strategy == "megatron":
            shared_root_value = os.environ.get("CRAFTLY_SHARED_CHECKPOINT_ROOT", "")
            checkpoint_root = Path(shared_root_value).expanduser().resolve() / "jobs" / job.job_id / "training" / "checkpoints"
            megatron_checkpoint_files = sum(1 for item in checkpoint_root.rglob("*") if item.is_file()) if checkpoint_root.is_dir() else 0
        checkpoint_proven = (
            megatron_checkpoint_files > 1
            if profile.distributed_strategy == "megatron"
            else any(item.reload_verified for item in checkpoints)
        )
        passed = (
            job.status == JobStatus.SUCCEEDED
            and bool(attempts)
            and any(event.kind == "heartbeat" for event in events)
            and checkpoint_proven
            and (not inject_worker_loss or (failure_injected and any(event.sequence > injection_sequence for event in events)))
        )
        return ProofResult(
            name=f"live_{profile.scheduler}_{profile_id}",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence={
                "job_id": job.job_id,
                "scheduler": profile.scheduler,
                "distributed_strategy": profile.distributed_strategy,
                "observed_states": observed,
                "attempts": len(attempts),
                "events": len(events),
                "heartbeats": sum(event.kind == "heartbeat" for event in events),
                "checkpoints": len(checkpoints),
                "reload_verified_checkpoints": sum(item.reload_verified for item in checkpoints),
                "megatron_checkpoint_files": megatron_checkpoint_files,
                "terminal_status": job.status.value,
                "worker_loss_requested": inject_worker_loss,
                "worker_loss_injected": failure_injected,
                "post_injection_events": sum(event.sequence > injection_sequence for event in events),
            },
            blockers=[] if passed else ["job did not finish with heartbeat and reload-verified checkpoint evidence"],
        )
    except (FileNotFoundError, ConnectionError, TimeoutError, RuntimeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        dependency_markers = ["required", "unavailable", "missing", "cluster", "topology", "connection", "timed out"]
        blocked = any(marker in message.lower() for marker in dependency_markers)
        return ProofResult(
            name=f"live_scheduler_{profile_id}",
            status="blocked" if blocked else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence={"job_id": job.job_id if job else None},
            blockers=[message],
        )
    except Exception as exc:
        return ProofResult(
            name=f"live_scheduler_{profile_id}",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence={"job_id": job.job_id if job else None},
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if service is not None:
            await service.close()


def docker_disaster_recovery_proof(
    *,
    project_name: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    qdrant_url: str,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    evidence: dict[str, Any] = {}
    try:
        import boto3
        import docker
        import redis

        client = docker.from_env()

        def service_container(service: str) -> Any:
            matches = client.containers.list(
                all=True,
                filters={
                    "label": [
                        f"com.docker.compose.project={project_name}",
                        f"com.docker.compose.service={service}",
                    ]
                },
            )
            if len(matches) != 1:
                raise RuntimeError(f"expected one {service} proof container, found {len(matches)}")
            return matches[0]

        postgres = service_container("postgres")
        redis_container = service_container("redis")
        minio = service_container("minio")
        qdrant = service_container("qdrant")
        qdrant_endpoint = urlparse(qdrant_url.rstrip("/"))
        if (
            qdrant_endpoint.scheme != "http"
            or qdrant_endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or qdrant_endpoint.username
            or qdrant_endpoint.password
        ):
            raise ValueError(
                "Docker disaster-recovery proof only accepts a loopback Qdrant URL"
            )

        def exec_checked(container: Any, command: list[str]) -> str:
            result = container.exec_run(command)
            output = result.output.decode("utf-8", "replace")
            if result.exit_code != 0:
                raise RuntimeError(f"container command failed ({container.name}): {output[-2000:]}")
            return output

        dump_path = f"/var/lib/postgresql/data/craftly-proof-dr-{uuid.uuid4().hex}.dump"
        try:
            exec_checked(
                postgres,
                ["pg_dump", "-U", "craftly", "-d", "craftly_proof", "-Fc", "-f", dump_path],
            )
            exec_checked(postgres, ["dropdb", "-U", "craftly", "--if-exists", "craftly_proof_restore"])
            exec_checked(postgres, ["createdb", "-U", "craftly", "craftly_proof_restore"])
            exec_checked(
                postgres,
                ["pg_restore", "-U", "craftly", "-d", "craftly_proof_restore", "--no-owner", dump_path],
            )
            restored_count = int(
                exec_checked(
                    postgres,
                    ["psql", "-U", "craftly", "-d", "craftly_proof_restore", "-Atc", "SELECT count(*) FROM schema_migrations"],
                ).strip()
            )
        finally:
            postgres.exec_run(["rm", "-f", "--", dump_path])
        if restored_count < 1:
            raise RuntimeError("restored Postgres database has no migration history")

        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_key = f"craftly:proof:restart:{uuid.uuid4()}"
        redis_client.set(redis_key, "persisted")
        redis_client.bgsave()
        deadline = time.monotonic() + 30
        while redis_client.info("persistence").get("rdb_bgsave_in_progress") and time.monotonic() < deadline:
            time.sleep(0.2)
        redis_container.restart(timeout=15)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if redis_client.ping() and redis_client.get(redis_key) == "persisted":
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Redis did not recover the persisted proof key after restart")
        redis_client.delete(redis_key)

        bucket = urlparse(object_store_uri).netloc
        prefix = urlparse(object_store_uri).path.strip("/")
        object_key = f"{prefix}/proofs/restart-{uuid.uuid4()}.json".strip("/")
        s3 = boto3.client("s3", endpoint_url=object_store_endpoint_url)
        body = json.dumps({"proof": "minio-restart"}, sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        s3.put_object(Bucket=bucket, Key=object_key, Body=body, Metadata={"sha256": digest})
        minio.restart(timeout=15)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                recovered = s3.get_object(Bucket=bucket, Key=object_key)["Body"].read()
                if hashlib.sha256(recovered).hexdigest() == digest:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("MinIO object did not recover after restart")
        s3.delete_object(Bucket=bucket, Key=object_key)

        qdrant_collection = f"craftly_dr_{uuid.uuid4().hex}"
        qdrant_point = str(uuid.uuid4())
        qdrant_marker = uuid.uuid4().hex
        qdrant_recovered = False
        qdrant_created = False
        qdrant_cleanup_error = ""
        try:
            with httpx.Client(timeout=15.0) as qdrant_client:
                created = qdrant_client.put(
                    f"{qdrant_url.rstrip('/')}/collections/{qdrant_collection}",
                    json={"vectors": {"size": 4, "distance": "Cosine"}},
                )
                created.raise_for_status()
                qdrant_created = True
                inserted = qdrant_client.put(
                    f"{qdrant_url.rstrip('/')}/collections/{qdrant_collection}/points",
                    params={"wait": "true"},
                    json={
                        "points": [
                            {
                                "id": qdrant_point,
                                "vector": [1.0, 0.0, 0.0, 0.0],
                                "payload": {"proof_marker": qdrant_marker},
                            }
                        ]
                    },
                )
                inserted.raise_for_status()
            qdrant.restart(timeout=15)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    with httpx.Client(timeout=5.0) as qdrant_client:
                        queried = qdrant_client.post(
                            f"{qdrant_url.rstrip('/')}/collections/{qdrant_collection}/points/query",
                            json={
                                "query": [1.0, 0.0, 0.0, 0.0],
                                "limit": 1,
                                "with_payload": True,
                            },
                        )
                        queried.raise_for_status()
                        points = queried.json().get("result", {}).get("points", [])
                        qdrant_recovered = bool(
                            points
                            and str(points[0].get("id")) == qdrant_point
                            and points[0].get("payload", {}).get("proof_marker")
                            == qdrant_marker
                        )
                        if qdrant_recovered:
                            break
                except Exception:
                    time.sleep(0.5)
            if not qdrant_recovered:
                raise RuntimeError("Qdrant point did not recover after restart")
        finally:
            if qdrant_created:
                try:
                    with httpx.Client(timeout=10.0) as qdrant_client:
                        deleted = qdrant_client.delete(
                            f"{qdrant_url.rstrip('/')}/collections/{qdrant_collection}"
                        )
                        if deleted.status_code not in {200, 404}:
                            qdrant_cleanup_error = (
                                f"Qdrant cleanup returned HTTP {deleted.status_code}"
                            )
                except Exception as exc:
                    qdrant_cleanup_error = f"Qdrant cleanup failed: {exc}"
        if qdrant_cleanup_error:
            raise RuntimeError(qdrant_cleanup_error)

        postgres.restart(timeout=15)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = postgres.exec_run(
                ["pg_isready", "-U", "craftly", "-d", "craftly_proof"],
            )
            if result.exit_code == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Postgres did not become ready after restart")
        evidence = {
            "postgres_dump_restore": True,
            "postgres_restored_migrations": restored_count,
            "postgres_restart": True,
            "redis_aof_restart": True,
            "minio_volume_restart": True,
            "qdrant_volume_restart": True,
            "containers": {
                "postgres": postgres.name,
                "redis": redis_container.name,
                "minio": minio.name,
                "qdrant": qdrant.name,
            },
        }
        return ProofResult(
            name="docker_disaster_recovery_drill",
            status="passed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
        )
    except Exception as exc:
        return ProofResult(
            name="docker_disaster_recovery_drill",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )


KUBERNETES_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
KUBERNETES_ROLLOUT_KINDS = {"deployment", "statefulset"}


def parse_dr_workload(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("DR workload must use namespace:kind:name")
    namespace, kind, name = parts
    kind = kind.lower()
    if kind not in KUBERNETES_ROLLOUT_KINDS:
        raise ValueError(f"unsupported DR workload kind: {kind}")
    for label, item in (("namespace", namespace), ("name", name)):
        if len(item) > 253 or KUBERNETES_NAME_RE.fullmatch(item) is None:
            raise ValueError(f"invalid Kubernetes {label}: {item!r}")
    return namespace, kind, name


async def kubernetes_disaster_recovery_proof(
    *,
    workloads: list[str],
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
    rollout_timeout_seconds: int,
) -> ProofResult:
    """Disrupt self-hosted control-plane services and prove durable recovery.

    This is intentionally opt-in. It never accepts arbitrary commands: callers
    provide only validated Kubernetes namespace, workload kind, and DNS name.
    """

    started = time.perf_counter()
    started_at = now_iso()
    service: TrainingWorkspaceService | None = None
    marker_dir: Path | None = None
    evidence: dict[str, Any] = {"workloads": workloads, "restarts": []}
    try:
        if shutil.which("kubectl") is None:
            raise FileNotFoundError("kubectl is required for the Kubernetes disaster-recovery proof")
        parsed = [parse_dr_workload(value) for value in workloads]
        if not parsed:
            raise ValueError("at least one Kubernetes DR workload is required")

        for namespace, kind, name in parsed:
            preflight = command_evidence(["kubectl", "get", kind, name, "-n", namespace, "-o", "name"], timeout=30)
            if preflight["exit_code"] != 0:
                raise RuntimeError(f"Kubernetes DR preflight failed for {namespace}:{kind}:{name}: {preflight['stderr'][-500:]}")

        service = TrainingWorkspaceService(
            store=PostgresTrainingStore(database_url),
            events=RedisEventBus(redis_url, retention_events=2_000_000),
            object_store=S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url),
            profiles=TrainingProfileRegistry.from_file(),
            campaigns=QualificationCampaignRegistry.from_file(),
        )
        job = await service.create_job(
            TrainingJobCreateRequest(
                profile_id="defensive-1k",
                idempotency_key=f"cluster-dr-{uuid.uuid4()}",
                git_commit="abcdef1",
                container_digest="sha256:" + "d" * 64,
                metadata={"proof": "kubernetes-disaster-recovery"},
            ),
            owner_id="cluster-dr-proof",
        )
        running, attempt = await service.claim_notebook_job(job.job_id)
        await service.ingest_events(
            running.job_id,
            TrainingEventBatch(
                attempt_id=attempt.attempt_id,
                events=[
                    TrainingEventInput(
                        source_sequence=1,
                        kind="checkpoint",
                        stage="disaster_recovery",
                        status="prepared",
                        payload={"durable_marker": True},
                    )
                ],
            ),
        )
        marker = json.dumps({"job_id": job.job_id, "attempt_id": attempt.attempt_id}, sort_keys=True).encode("utf-8")
        marker_digest = hashlib.sha256(marker).hexdigest()
        marker_dir = Path(tempfile.mkdtemp(prefix="craftly-cluster-dr-"))
        marker_path = marker_dir / "marker.json"
        marker_path.write_bytes(marker)
        marker_key = f"proofs/cluster-dr/{job.job_id}/marker.json"
        stored_marker = await asyncio.to_thread(
            service.object_store.put_file,
            marker_path,
            key=marker_key,
        )
        marker_uri = stored_marker.uri
        await service.close()
        service = None

        for namespace, kind, name in parsed:
            target = f"{kind}/{name}"
            restarted = command_evidence(["kubectl", "rollout", "restart", target, "-n", namespace], timeout=30)
            if restarted["exit_code"] != 0:
                raise RuntimeError(f"failed to restart {namespace}:{target}: {restarted['stderr'][-500:]}")
            recovered = command_evidence(
                ["kubectl", "rollout", "status", target, "-n", namespace, f"--timeout={rollout_timeout_seconds}s"],
                timeout=rollout_timeout_seconds + 30,
            )
            evidence["restarts"].append(
                {
                    "namespace": namespace,
                    "target": target,
                    "restart_exit_code": restarted["exit_code"],
                    "recovery_exit_code": recovered["exit_code"],
                }
            )
            if recovered["exit_code"] != 0:
                raise RuntimeError(f"rollout recovery failed for {namespace}:{target}: {recovered['stderr'][-500:]}")

        service = TrainingWorkspaceService(
            store=PostgresTrainingStore(database_url),
            events=RedisEventBus(redis_url, retention_events=2_000_000),
            object_store=S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url),
            profiles=TrainingProfileRegistry.from_file(),
            campaigns=QualificationCampaignRegistry.from_file(),
        )
        recovered_job = await service.get_job(job.job_id)
        recovered_events = await service.events.read(job.job_id, after_sequence=0, limit=10)
        marker_download = marker_path.with_name("recovered-marker.json")
        await asyncio.to_thread(service.object_store.get_file, marker_key, marker_download)
        recovered_digest = sha256_file(marker_download)
        passed = (
            recovered_job.job_id == job.job_id
            and recovered_job.status == JobStatus.RUNNING
            and len(recovered_events) == 1
            and recovered_events[0].kind == "checkpoint"
            and recovered_digest == marker_digest
            and all(item["recovery_exit_code"] == 0 for item in evidence["restarts"])
        )
        evidence.update(
            {
                "job_id": job.job_id,
                "attempt_id": attempt.attempt_id,
                "postgres_job_recovered": recovered_job.job_id == job.job_id,
                "redis_event_recovered": len(recovered_events) == 1,
                "object_uri": marker_uri,
                "object_checksum_expected": marker_digest,
                "object_checksum_recovered": recovered_digest,
            }
        )
        return ProofResult(
            name="kubernetes_control_plane_disaster_recovery",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
            blockers=[] if passed else ["one or more durable records did not survive the Kubernetes recovery drill"],
        )
    except (FileNotFoundError, ConnectionError, TimeoutError, RuntimeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        dependency_failure = any(term in message.lower() for term in ("required", "preflight", "connect", "not found", "unavailable"))
        return ProofResult(
            name="kubernetes_control_plane_disaster_recovery",
            status="blocked" if dependency_failure else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
            blockers=[message],
        )
    except Exception as exc:
        return ProofResult(
            name="kubernetes_control_plane_disaster_recovery",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if service is not None:
            await service.close()
        if marker_dir is not None:
            shutil.rmtree(marker_dir, ignore_errors=True)


async def worker_loss_recovery_proof(
    *,
    database_url: str,
    redis_url: str,
    object_store_uri: str,
    object_store_endpoint_url: str,
) -> ProofResult:
    started = time.perf_counter()
    started_at = now_iso()
    service: TrainingWorkspaceService | None = None
    container: Any | None = None
    try:
        import docker

        docker_client = docker.from_env()

        class DockerLossScheduler(SchedulerAdapter):
            name = "kubernetes"

            async def validate(self, spec: Any) -> dict[str, Any]:
                return {"status": "ready", "proof_scheduler": "docker-process"}

            async def submit(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding:
                nonlocal container
                container = await asyncio.to_thread(
                    docker_client.containers.run,
                    "alpine:3.20",
                    ["sleep", "300"],
                    detach=True,
                    network_disabled=True,
                    read_only=True,
                    mem_limit="64m",
                    nano_cpus=100_000_000,
                    labels={"craftly.proof": "worker-loss", "craftly.job-id": job.job_id},
                )
                return SchedulerBinding(
                    scheduler=self.name,
                    external_id=container.id,
                    metadata={"failure_class": "node_loss"},
                )

            async def status(self, binding: SchedulerBinding) -> str:
                active = docker_client.containers.get(binding.external_id)
                await asyncio.to_thread(active.reload)
                if active.status in {"created", "restarting"}:
                    return "provisioning"
                if active.status == "running":
                    return "running"
                result = await asyncio.to_thread(active.wait)
                return "succeeded" if int(result.get("StatusCode", 1)) == 0 else "failed"

            async def cancel(self, binding: SchedulerBinding) -> None:
                active = docker_client.containers.get(binding.external_id)
                await asyncio.to_thread(active.remove, force=True)

        service = TrainingWorkspaceService(
            store=PostgresTrainingStore(database_url),
            events=RedisEventBus(redis_url),
            object_store=S3ObjectStore(object_store_uri, endpoint_url=object_store_endpoint_url),
        )
        from src.craftly.training_workspace import TrainingJobCreateRequest

        job = await service.create_job(
            TrainingJobCreateRequest(
                profile_id="defensive-10k",
                idempotency_key=f"worker-loss-{uuid.uuid4()}",
                git_commit="abcdef1",
                container_digest="sha256:" + "c" * 64,
            ),
            owner_id=f"worker-loss-{uuid.uuid4()}",
        )
        controller = TrainingController(service, schedulers={"kubernetes": DockerLossScheduler()})
        submitted = await controller.reconcile_job(job)
        running = await controller.reconcile_job(submitted)
        if running.status != JobStatus.RUNNING or container is None:
            raise RuntimeError(f"proof worker did not reach running state: {running.status.value}")
        await asyncio.to_thread(container.kill)
        requeued = await controller.reconcile_job(running)
        attempts = await service.store.list_attempts(job.job_id)
        passed = (
            requeued.status == JobStatus.QUEUED
            and len(attempts) == 1
            and requeued.scheduler_binding.get("retry_failure_class") == "node_loss"
            and float(requeued.scheduler_binding.get("retry_not_before_unix", 0)) > time.time()
        )
        return ProofResult(
            name="worker_loss_retry_recovery",
            status="passed" if passed else "failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence={
                "job_id": job.job_id,
                "status_after_loss": requeued.status.value,
                "attempt_count": len(attempts),
                "retry_failure_class": requeued.scheduler_binding.get("retry_failure_class"),
                "retry_not_before_unix": requeued.scheduler_binding.get("retry_not_before_unix"),
            },
        )
    except Exception as exc:
        return ProofResult(
            name="worker_loss_retry_recovery",
            status="failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if container is not None:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception:
                pass
        if service is not None:
            await service.close()


def distributed_gpu_proofs() -> list[ProofResult]:
    started_at = now_iso()
    started = time.perf_counter()
    evidence: dict[str, Any] = {}
    blockers: list[str] = []
    try:
        import torch

        evidence = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "torch_version": torch.__version__,
        }
        if torch.cuda.device_count() < 2:
            blockers.append("at least two visible CUDA GPUs are required for multi-GPU FSDP proof")
        if shutil.which("deepspeed") is None:
            blockers.append("DeepSpeed executable is required for ZeRO-3 proof")
        megatron_root = os.environ.get("CRAFTLY_MEGATRON_ROOT", "")
        if not megatron_root or not Path(megatron_root).is_dir():
            blockers.append("CRAFTLY_MEGATRON_ROOT must point to a validated Megatron-LM checkout")
        blockers.append("multi-node and 60B checkpoint/resume require a real scheduler allocation and promoted dataset")
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}: {exc}")
    return [
        ProofResult(
            name="multi_node_fsdp_zero3_megatron_60b_resume",
            status="blocked" if blockers else "passed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
            blockers=blockers,
        )
    ]


def build_report(results: list[ProofResult]) -> ProductionProofReport:
    counts = {status: sum(item.status == status for item in results) for status in ("passed", "failed", "blocked")}
    status: ProofStatus = "failed" if counts["failed"] else "blocked" if counts["blocked"] else "passed"
    return ProductionProofReport(
        generated_at=now_iso(),
        environment={
            "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown",
            "python": os.sys.version,
            "git_commit": command_evidence(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        },
        proofs=results,
        passed=counts["passed"],
        failed=counts["failed"],
        blocked=counts["blocked"],
        status=status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measured Craftly training-workspace production proofs.")
    parser.add_argument("--database-url", default=os.environ.get("CRAFTLY_DATABASE_URL", "postgresql://craftly:craftly_proof_only_change_me@127.0.0.1:55432/craftly_proof"))
    parser.add_argument("--redis-url", default=os.environ.get("CRAFTLY_REDIS_URL", "redis://127.0.0.1:56379/0"))
    parser.add_argument("--object-store-uri", default=os.environ.get("CRAFTLY_OBJECT_STORE_URI", "s3://craftly-proof/training-workspace"))
    parser.add_argument("--object-store-endpoint-url", default=os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL", "http://127.0.0.1:59000"))
    parser.add_argument("--qdrant-url", default=os.environ.get("CRAFTLY_QDRANT_URL", "http://127.0.0.1:56333"))
    parser.add_argument("--output-dir", default="artifacts/craftly/production-proofs")
    parser.add_argument("--event-count", type=int, default=0, help="Set to 1000000 for the full ordered-event proof.")
    parser.add_argument("--soak-seconds", type=int, default=0, help="Set to 86400 for the required 24-hour proof.")
    parser.add_argument("--soak-interval-seconds", type=int, default=30)
    parser.add_argument("--skip-infrastructure", action="store_true")
    parser.add_argument("--skip-capability-probes", action="store_true", help="Omit unrelated scheduler/GPU probes from a targeted proof report.")
    parser.add_argument("--docker-project", default="craftly-proof")
    parser.add_argument("--skip-disaster-recovery", action="store_true")
    parser.add_argument(
        "--inject-kubernetes-disaster-recovery",
        action="store_true",
        help="Disrupt the explicitly listed Kubernetes workloads and verify Postgres, Redis, and S3 recovery.",
    )
    parser.add_argument(
        "--dr-workload",
        action="append",
        default=[],
        help="Kubernetes workload as namespace:kind:name; repeat for each control-plane dependency.",
    )
    parser.add_argument("--dr-rollout-timeout-seconds", type=int, default=600)
    parser.add_argument("--live-profile", action="append", default=[], help="Submit and watch this real cluster profile; repeatable.")
    parser.add_argument("--dataset-manifest-uri")
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--tokenizer-uri")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--git-commit", default=os.environ.get("CRAFTLY_TRAINING_GIT_COMMIT", ""))
    parser.add_argument("--container-digest", default=os.environ.get("CRAFTLY_TRAINING_IMAGE_DIGEST", ""))
    parser.add_argument("--live-timeout-seconds", type=int, default=86400)
    parser.add_argument("--live-poll-seconds", type=int, default=15)
    parser.add_argument("--inject-worker-loss", action="store_true", help="After a verified checkpoint, kill/requeue one real worker and require recovery.")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "craftly-proof")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "craftly-proof-password-change-me")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    results: list[ProofResult] = []
    if not args.skip_infrastructure:
        results.append(
            await infrastructure_lifecycle_proof(
                database_url=args.database_url,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                output_dir=output_dir,
            )
        )
    if args.event_count:
        results.append(
            await million_event_proof(
                database_url=args.database_url,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                event_count=args.event_count,
            )
        )
    if args.soak_seconds:
        results.append(
            await soak_proof(
                database_url=args.database_url,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                duration_seconds=args.soak_seconds,
                interval_seconds=args.soak_interval_seconds,
            )
        )
    if not args.skip_infrastructure and not args.skip_disaster_recovery:
        results.append(
            await asyncio.to_thread(
                docker_disaster_recovery_proof,
                project_name=args.docker_project,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                qdrant_url=args.qdrant_url,
            )
        )
        results.append(
            await worker_loss_recovery_proof(
                database_url=args.database_url,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
            )
        )
    if args.inject_kubernetes_disaster_recovery:
        workloads = args.dr_workload or [
            "default:statefulset:craftly-postgres",
            "default:statefulset:craftly-redis",
            "default:statefulset:craftly-minio",
            "craftly-training:deployment:craftly-training-controller",
        ]
        results.append(
            await kubernetes_disaster_recovery_proof(
                workloads=workloads,
                database_url=args.database_url,
                redis_url=args.redis_url,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                rollout_timeout_seconds=args.dr_rollout_timeout_seconds,
            )
        )
    if args.live_profile:
        required_live = {
            "dataset_manifest_uri": args.dataset_manifest_uri,
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "tokenizer_uri": args.tokenizer_uri,
            "tokenizer_sha256": args.tokenizer_sha256,
            "git_commit": args.git_commit,
            "container_digest": args.container_digest,
        }
        missing_live = [name for name, value in required_live.items() if not value]
        if missing_live:
            raise ValueError("live scheduler proofs require: " + ", ".join(missing_live))
        for profile_id in args.live_profile:
            results.append(
                await live_scheduler_execution_proof(
                    profile_id=profile_id,
                    database_url=args.database_url,
                    redis_url=args.redis_url,
                    object_store_uri=args.object_store_uri,
                    object_store_endpoint_url=args.object_store_endpoint_url,
                    dataset_manifest_uri=args.dataset_manifest_uri,
                    dataset_manifest_sha256=args.dataset_manifest_sha256,
                    tokenizer_uri=args.tokenizer_uri,
                    tokenizer_sha256=args.tokenizer_sha256,
                    git_commit=args.git_commit,
                    container_digest=args.container_digest,
                    timeout_seconds=args.live_timeout_seconds,
                    poll_seconds=args.live_poll_seconds,
                    inject_worker_loss=args.inject_worker_loss,
                )
            )
    if not args.skip_capability_probes:
        results.extend(external_scheduler_proofs())
        results.extend(distributed_gpu_proofs())
    report = build_report(results)
    report_path = output_dir / "production_proof_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), flush=True)
    return 2 if report.failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
