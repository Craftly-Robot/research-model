"""Live production proof runner for Craftly deployments.

This module does not fake production readiness. It verifies live dependencies
when URLs/secrets are supplied and marks missing dependencies as skipped in
validation mode or failed in strict mode.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field

from src.craftly.db.migration_runner import apply_migrations
from src.craftly.db import PostgresRAGDispatcher, PostgresRAGStore
from src.craftly.identity.quota import RedisQuotaStore
from src.craftly.indexing.repository_indexer import RAGIndexCoordinator
from src.craftly.learning.storage import ObjectStoreConfig, verify_object_store_lifecycle
from src.craftly.security.audit import run_security_audit
from src.craftly.shared.config_contracts import load_active_model_contract
from src.craftly.shared.integrity import canonical_json_bytes, sha256_file
from src.craftly.shared.schemas import StrictModel


class ProofCheckResult(StrictModel):
    name: str
    status: str
    required: bool
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class NativeServingLoadReport(StrictModel):
    status: str
    endpoint: str
    requests: int
    concurrency: int
    passed: int
    failed: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float = 0.0
    max_latency_ms: float
    duration_seconds: float = 0.0
    throughput_rps: float = 0.0
    response_bytes: int = Field(default=0, ge=0)
    status_codes: dict[str, int] = Field(default_factory=dict)
    timeout_count: int
    streaming_requests: int = Field(default=0, ge=0)
    streaming_passed: int = Field(default=0, ge=0)
    content_validation_failures: int = Field(default=0, ge=0)
    error_samples: list[str] = Field(default_factory=list)


class ProductionProofConfig(StrictModel):
    strict: bool = False
    output_dir: str = "artifacts/craftly/production-proof"
    postgres_url: str | None = None
    apply_postgres_migrations: bool = False
    redis_url: str | None = None
    object_store_uri: str | None = None
    object_store_endpoint_url: str | None = None
    qdrant_url: str | None = None
    run_rag_control_plane: bool = False
    allowed_insecure_service_hosts: list[str] = Field(default_factory=list)
    serving_url: str | None = None
    serving_api_key: str | None = None
    serving_model: str = "craftly-scratch"
    load_test_requests: int = Field(default=0, ge=0, le=10_000)
    load_test_concurrency: int = Field(default=4, ge=1, le=512)
    load_test_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    load_test_streaming_requests: int = Field(default=0, ge=0, le=1_000)
    benchmark_dir: str | None = None
    executable_benchmark_report: str | None = None
    scorecard_report: str | None = None
    active_model_profile: str | None = None
    run_security_audit: bool = False
    strict_security_tools: bool = False


class ProductionProofReport(StrictModel):
    authority: Literal["evidence_only"] = "evidence_only"
    status: str
    mode: str
    checks: list[ProofCheckResult]
    recommendations: list[str]
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "production_proof_report.json"
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "production_proof_report.md")
        return path


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _validated_service_url(
    value: str,
    *,
    label: str,
    allowed_insecure_hosts: list[str],
) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} URL must not contain a query string or fragment")
    allowed = {host.lower() for host in allowed_insecure_hosts}
    if parsed.scheme != "https" and parsed.hostname.lower() not in {
        "127.0.0.1",
        "localhost",
        "::1",
        *allowed,
    }:
        raise ValueError(
            f"remote {label} URL must use HTTPS unless its exact host is explicitly "
            "listed with --allow-insecure-service-host"
        )
    return normalized


def config_from_env(args: argparse.Namespace) -> ProductionProofConfig:
    return ProductionProofConfig(
        strict=args.strict,
        output_dir=args.output_dir,
        postgres_url=args.postgres_url or _env("CRAFTLY_DATABASE_URL"),
        apply_postgres_migrations=args.apply_postgres_migrations,
        redis_url=args.redis_url or _env("CRAFTLY_REDIS_URL"),
        object_store_uri=args.object_store_uri or _env("CRAFTLY_OBJECT_STORE_URI"),
        object_store_endpoint_url=args.object_store_endpoint_url or _env("CRAFTLY_OBJECT_STORE_ENDPOINT_URL"),
        qdrant_url=args.qdrant_url or _env("CRAFTLY_QDRANT_URL"),
        run_rag_control_plane=args.run_rag_control_plane,
        allowed_insecure_service_hosts=args.allow_insecure_service_host,
        serving_url=args.serving_url or _env("CRAFTLY_SERVING_URL"),
        serving_api_key=args.serving_api_key or _env("CRAFTLY_MODEL_API_KEY"),
        serving_model=args.serving_model,
        load_test_requests=args.load_test_requests,
        load_test_concurrency=args.load_test_concurrency,
        load_test_timeout_seconds=args.load_test_timeout_seconds,
        load_test_streaming_requests=args.load_test_streaming_requests,
        benchmark_dir=args.benchmark_dir or _env("CRAFTLY_BENCHMARK_DIR"),
        executable_benchmark_report=(
            args.executable_benchmark_report or _env("CRAFTLY_EXECUTABLE_BENCHMARK_REPORT")
        ),
        scorecard_report=args.scorecard_report or _env("CRAFTLY_AGENT_SCORECARD_REPORT"),
        active_model_profile=args.active_model_profile or _env("CRAFTLY_ACTIVE_MODEL_PROFILE_PATH"),
        run_security_audit=args.run_security_audit,
        strict_security_tools=args.strict_security_tools,
    )


async def _check_postgres(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.postgres_url:
        return _missing("postgres_migrations", config.strict, started, "CRAFTLY_DATABASE_URL or --postgres-url is required")
    if config.strict and not config.apply_postgres_migrations:
        return _result(
            "postgres_migrations",
            "failed",
            True,
            started,
            {"reason": "strict proof requires --apply-postgres-migrations; a dry-run is not deployment evidence"},
        )
    try:
        result = await apply_migrations(config.postgres_url, dry_run=not config.apply_postgres_migrations)
        return _result("postgres_migrations", "passed", True, started, result)
    except Exception as exc:
        return _result("postgres_migrations", "failed", True, started, error=str(exc))


async def _check_redis_quota(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.redis_url:
        return _missing("redis_quota", config.strict, started, "CRAFTLY_REDIS_URL or --redis-url is required")
    try:
        store = RedisQuotaStore(config.redis_url)
        allowed, remaining = await store.consume(
            "production-proof",
            now=time.time(),
            rate=1.0,
            capacity=5.0,
            cost=1.0,
        )
        return _result("redis_quota", "passed" if allowed else "failed", True, started, {"allowed": allowed, "remaining": remaining})
    except Exception as exc:
        return _result("redis_quota", "failed", True, started, error=str(exc))


async def _check_object_store(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.object_store_uri:
        return _missing("object_store_lifecycle", config.strict, started, "CRAFTLY_OBJECT_STORE_URI or --object-store-uri is required")
    try:
        report = verify_object_store_lifecycle(
            config=ObjectStoreConfig(uri=config.object_store_uri, endpoint_url=config.object_store_endpoint_url),
            work_dir=Path(config.output_dir) / "object-store",
            key=f"production-proof/{int(time.time())}.json",
        )
        return _result("object_store_lifecycle", report.status, True, started, report.model_dump())
    except Exception as exc:
        return _result("object_store_lifecycle", "failed", True, started, error=str(exc))


async def _check_qdrant(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.qdrant_url:
        return _missing("qdrant_round_trip", config.strict, started, "CRAFTLY_QDRANT_URL or --qdrant-url is required")
    collection = f"craftly_proof_{uuid.uuid4().hex}"
    endpoint = ""
    marker = uuid.uuid4().hex
    point_id = str(uuid.uuid4())
    organization_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    cleanup_error = ""
    primary_error = ""
    collection_created = False
    matched = False
    try:
        endpoint = _validated_service_url(
            config.qdrant_url,
            label="Qdrant",
            allowed_insecure_hosts=config.allowed_insecure_service_hosts,
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            created = await client.put(
                f"{endpoint}/collections/{collection}",
                json={"vectors": {"size": 4, "distance": "Cosine"}},
            )
            created.raise_for_status()
            collection_created = True
            upserted = await client.put(
                f"{endpoint}/collections/{collection}/points",
                params={"wait": "true"},
                json={
                    "points": [
                        {
                            "id": point_id,
                            "vector": [1.0, 0.0, 0.0, 0.0],
                            "payload": {
                                "proof_marker": marker,
                                "organization_id": organization_id,
                                "project_id": project_id,
                            },
                        }
                    ]
                },
            )
            upserted.raise_for_status()
            queried = await client.post(
                f"{endpoint}/collections/{collection}/points/query",
                json={
                    "query": [1.0, 0.0, 0.0, 0.0],
                    "limit": 1,
                    "with_payload": True,
                    "filter": {
                        "must": [
                            {"key": "organization_id", "match": {"value": organization_id}},
                            {"key": "project_id", "match": {"value": project_id}},
                        ]
                    },
                },
            )
            queried.raise_for_status()
            payload = queried.json()
            points = payload.get("result", {}).get("points", [])
            matched = bool(
                points
                and str(points[0].get("id")) == point_id
                and points[0].get("payload", {}).get("proof_marker") == marker
            )
            if not matched:
                raise RuntimeError("Qdrant query did not return the inserted proof point")
            rejected = await client.post(
                f"{endpoint}/collections/{collection}/points/query",
                json={
                    "query": [1.0, 0.0, 0.0, 0.0],
                    "limit": 1,
                    "with_payload": True,
                    "filter": {
                        "must": [
                            {"key": "organization_id", "match": {"value": str(uuid.uuid4())}},
                            {"key": "project_id", "match": {"value": project_id}},
                        ]
                    },
                },
            )
            rejected.raise_for_status()
            if rejected.json().get("result", {}).get("points", []):
                raise RuntimeError("Qdrant tenant filter returned a cross-tenant point")
    except Exception as exc:
        primary_error = str(exc)
    finally:
        if endpoint and collection_created:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    deleted = await client.delete(f"{endpoint}/collections/{collection}")
                    if deleted.status_code not in {200, 404}:
                        cleanup_error = (
                            "Qdrant proof collection cleanup returned "
                            f"HTTP {deleted.status_code}"
                        )
            except Exception as exc:
                cleanup_error = f"Qdrant proof collection cleanup failed: {exc}"
        if cleanup_error:
            primary_error = primary_error or cleanup_error
    if primary_error:
        return _result(
            "qdrant_round_trip",
            "failed",
            True,
            started,
            {
                "collection": collection,
                "created": collection_created,
                "query_verified": matched,
                "cleanup_verified": not cleanup_error,
            },
            error=primary_error,
        )
    return _result(
        "qdrant_round_trip",
        "passed",
        True,
        started,
        {
            "collection": collection,
            "created": True,
            "upsert_verified": True,
            "query_verified": True,
            "tenant_filter_verified": True,
            "cleanup_verified": True,
        },
    )


async def _check_rag_control_plane(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.run_rag_control_plane:
        return _result(
            "rag_control_plane_lifecycle",
            "failed" if config.strict else "skipped",
            config.strict,
            started,
            {"reason": "--run-rag-control-plane was not set"},
        )
    missing = [
        name
        for name, value in (
            ("Postgres", config.postgres_url),
            ("Redis", config.redis_url),
            ("S3/MinIO", config.object_store_uri),
        )
        if not value
    ]
    if missing:
        return _result(
            "rag_control_plane_lifecycle",
            "failed",
            True,
            started,
            {"reason": "missing " + ", ".join(missing)},
        )
    organization_id = str(uuid.uuid4())
    store: PostgresRAGStore | None = None
    dispatcher: PostgresRAGDispatcher | None = None
    coordinator: RAGIndexCoordinator | None = None
    cleanup_error = ""
    primary_error = ""
    details: dict[str, Any] = {}
    wake_task: asyncio.Task[bool] | None = None
    previous_uri = os.environ.get("CRAFTLY_RAG_OBJECT_STORE_URI")
    previous_endpoint = os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL")
    try:
        assert config.postgres_url and config.redis_url and config.object_store_uri
        if not config.object_store_uri.startswith("s3://"):
            raise RuntimeError("RAG control-plane proof requires S3/MinIO snapshot storage")
        os.environ["CRAFTLY_RAG_OBJECT_STORE_URI"] = config.object_store_uri.rstrip("/") + "/rag-proof-snapshots"
        if config.object_store_endpoint_url:
            os.environ["CRAFTLY_OBJECT_STORE_ENDPOINT_URL"] = config.object_store_endpoint_url
        with tempfile.TemporaryDirectory(prefix="craftly-rag-proof-") as temporary:
            repository = Path(temporary)
            (repository / "lib.rs").write_text(
                "fn helper(v: i32) -> i32 { v + 1 }\nfn run() -> i32 { helper(4) }\n",
                encoding="utf-8",
            )
            store = PostgresRAGStore(config.postgres_url, organization_id=organization_id, max_pool_size=4)
            store.ensure_organization(organization_id, name="Craftly RAG lifecycle proof")
            project = store.create_project(
                name="rag-lifecycle-proof",
                repo_path=str(repository),
                organization_id=organization_id,
                owner_user_id="production-proof-operator",
            )
            coordinator = RAGIndexCoordinator(
                store,
                redis_url=config.redis_url,
                production_mode=True,
                worker_id=f"rag-proof-{uuid.uuid4().hex[:12]}",
            )
            if coordinator.redis is None:
                raise RuntimeError("RAG proof did not initialize Redis control")
            await coordinator.redis.healthcheck()
            wake_task = asyncio.create_task(
                coordinator.redis.wait_for_work(coordinator.worker_id, timeout_seconds=5.0)
            )
            await asyncio.sleep(0.05)
            idempotency_key = f"rag-proof-{uuid.uuid4().hex}"
            first = await coordinator.submit(
                organization_id=organization_id,
                project_id=str(project["id"]),
                idempotency_key=idempotency_key,
            )
            if not await wake_task:
                raise RuntimeError("Redis consumer-group wake signal was not delivered")
            wake_task = None
            duplicate = await coordinator.submit(
                organization_id=organization_id,
                project_id=str(project["id"]),
                idempotency_key=idempotency_key,
            )
            if first["id"] != duplicate["id"]:
                raise RuntimeError("RAG index idempotency proof failed")
            dispatcher = PostgresRAGDispatcher(config.postgres_url)
            claimed = await asyncio.to_thread(
                dispatcher.claim_index_job,
                worker_id=coordinator.worker_id,
                lease_seconds=120,
            )
            if claimed is None or claimed["id"] != first["id"]:
                raise RuntimeError("global RAG dispatcher did not claim the submitted job")
            completed = await coordinator.execute_claimed_index_job(claimed)
            if not completed or completed.get("status") != "succeeded":
                raise RuntimeError(f"RAG index job did not succeed: {completed}")
            index_status = store.index_status(str(project["id"]))
            if int(index_status["chunk_count"]) < 2:
                raise RuntimeError("RAG index did not persist expected Tree-sitter chunks")
            event = await asyncio.to_thread(
                dispatcher.claim_outbox,
                worker_id=coordinator.worker_id,
                lease_seconds=120,
            )
            if event is None:
                raise RuntimeError("RAG vector-sync outbox event is missing")
            store.fail_rag_outbox(
                str(event["id"]),
                worker_id=coordinator.worker_id,
                error="injected vector backend outage",
                retry_delay_seconds=0,
            )
            replay_worker = f"rag-replay-{uuid.uuid4().hex[:12]}"
            replay = store.claim_rag_outbox(worker_id=replay_worker)
            if replay is None or int(replay["attempt"]) != 2:
                raise RuntimeError("RAG outbox did not replay after injected outage")
            store.complete_rag_outbox(str(replay["id"]), worker_id=replay_worker)
            details = {
                "organization_id": organization_id,
                "project_id": str(project["id"]),
                "job_id": str(completed["id"]),
                "idempotency_verified": True,
                "redis_consumer_group_wake_verified": True,
                "tree_sitter_chunk_count": int(index_status["chunk_count"]),
                "outbox_retry_attempt": int(replay["attempt"]),
                "snapshot_storage": "s3_or_minio",
            }
    except Exception as exc:
        primary_error = str(exc)
    finally:
        if wake_task is not None and not wake_task.done():
            wake_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wake_task
        if previous_uri is None:
            os.environ.pop("CRAFTLY_RAG_OBJECT_STORE_URI", None)
        else:
            os.environ["CRAFTLY_RAG_OBJECT_STORE_URI"] = previous_uri
        if previous_endpoint is None:
            os.environ.pop("CRAFTLY_OBJECT_STORE_ENDPOINT_URL", None)
        else:
            os.environ["CRAFTLY_OBJECT_STORE_ENDPOINT_URL"] = previous_endpoint
        if coordinator is not None:
            await coordinator.close()
        if store is not None:
            store.close()
        if dispatcher is not None:
            dispatcher.close()
        if config.postgres_url:
            try:
                import asyncpg

                connection = await asyncpg.connect(config.postgres_url)
                try:
                    async with connection.transaction():
                        await connection.execute(
                            "SELECT set_config('craftly.organization_id', $1, true)",
                            organization_id,
                        )
                        await connection.execute(
                            "UPDATE projects SET active_index_revision=NULL WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM rag_outbox_events WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM rag_index_jobs WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM code_chunks WHERE project_id IN "
                            "(SELECT id FROM projects WHERE organization_id=$1::uuid)",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM workspace_files WHERE project_id IN "
                            "(SELECT id FROM projects WHERE organization_id=$1::uuid)",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM rag_index_revisions WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM project_members WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM projects WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM organization_members WHERE organization_id=$1::uuid",
                            organization_id,
                        )
                        await connection.execute(
                            "DELETE FROM organizations WHERE id=$1::uuid",
                            organization_id,
                        )
                finally:
                    await connection.close()
            except Exception as exc:
                cleanup_error = str(exc)
    if cleanup_error:
        primary_error = primary_error or f"RAG proof cleanup failed: {cleanup_error}"
    return _result(
        "rag_control_plane_lifecycle",
        "failed" if primary_error else "passed",
        True,
        started,
        {**details, "cleanup_verified": not cleanup_error},
        error=primary_error,
    )


async def _check_serving_health(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.serving_url:
        return _missing("native_serving_health", config.strict, started, "CRAFTLY_SERVING_URL or --serving-url is required")
    try:
        if config.strict and not config.serving_api_key:
            raise RuntimeError("strict serving proof requires an authenticated service token")
        endpoint = _validated_service_url(
            config.serving_url,
            label="serving",
            allowed_insecure_hosts=config.allowed_insecure_service_hosts,
        )
        if config.strict and urlparse(endpoint).path not in {"", "/"}:
            raise RuntimeError("strict serving proof URL must identify the service root")
        headers = _auth_headers(config)
        async with httpx.AsyncClient(timeout=10.0) as client:
            ready = await client.get(f"{endpoint}/health/ready", headers=headers)
            models = await client.get(f"{endpoint}/v1/models", headers=headers)
        ready.raise_for_status()
        models.raise_for_status()
        readiness = ready.json()
        model_payload = models.json()
        model_ids = {
            str(item.get("id") or "")
            for item in model_payload.get("data", [])
            if isinstance(item, dict)
        }
        if readiness.get("status") != "ready" or readiness.get("scratch_only") is not True:
            raise RuntimeError("serving readiness does not prove a loaded scratch checkpoint")
        if readiness.get("model_name") != config.serving_model or config.serving_model not in model_ids:
            raise RuntimeError("serving model identity does not match the requested Craftly model")
        checkpoint_manifest = str(readiness.get("checkpoint_manifest") or "")
        if not checkpoint_manifest:
            raise RuntimeError("serving readiness is missing checkpoint identity")
        for key in ("checkpoint_manifest_sha256", "tokenizer_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(readiness.get(key) or "")):
                raise RuntimeError(f"serving readiness is missing {key}")
        if config.strict:
            if not config.active_model_profile:
                raise RuntimeError("strict serving proof requires --active-model-profile")
            active = load_active_model_contract(
                Path(config.active_model_profile).expanduser().resolve(strict=True)
            )
            expected_endpoint = f"{endpoint.rstrip('/')}/v1"
            if active.profile.endpoint.rstrip("/") != expected_endpoint:
                raise RuntimeError("active model profile endpoint does not match the live service")
            if (
                readiness["checkpoint_manifest_sha256"]
                != active.profile.evidence.get("checkpoint_manifest_sha256")
                or readiness["tokenizer_sha256"]
                != active.profile.evidence.get("tokenizer_sha256")
            ):
                raise RuntimeError("live serving hashes do not match the active model profile")
        return _result(
            "native_serving_health",
            "passed",
            True,
            started,
            {
                "ready": readiness,
                "models": model_payload,
                "model_identity_verified": True,
                "checkpoint_hash_verified": True,
                "tokenizer_hash_verified": True,
                "authenticated": bool(config.serving_api_key),
            },
        )
    except Exception as exc:
        return _result("native_serving_health", "failed", True, started, error=str(exc))


async def _check_serving_load(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.serving_url:
        return _missing("native_serving_load", config.strict, started, "CRAFTLY_SERVING_URL or --serving-url is required")
    if config.load_test_requests <= 0:
        return _result("native_serving_load", "skipped" if not config.strict else "failed", config.strict, started, {"reason": "--load-test-requests is 0"})
    try:
        endpoint = _validated_service_url(
            config.serving_url,
            label="serving",
            allowed_insecure_hosts=config.allowed_insecure_service_hosts,
        )
        if config.strict and config.load_test_streaming_requests < 1:
            raise RuntimeError(
                "strict serving proof requires at least one streaming SSE request"
            )
        report = await run_native_serving_load_test(
            endpoint=endpoint,
            model=config.serving_model,
            api_key=config.serving_api_key,
            requests=config.load_test_requests,
            concurrency=config.load_test_concurrency,
            timeout_seconds=config.load_test_timeout_seconds,
            streaming_requests=config.load_test_streaming_requests,
        )
        return _result("native_serving_load", report.status, True, started, report.model_dump())
    except Exception as exc:
        return _result("native_serving_load", "failed", True, started, error=str(exc))


def _check_benchmarks(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if config.strict:
        required_paths = {
            "executable benchmark report": config.executable_benchmark_report,
            "repository scorecard report": config.scorecard_report,
            "active model profile": config.active_model_profile,
        }
        missing = [name for name, value in required_paths.items() if not value]
        if missing:
            return _result(
                "benchmark_and_model_evidence",
                "failed",
                True,
                started,
                {"reason": "strict proof requires " + ", ".join(missing)},
            )
        try:
            from src.craftly.evaluation.agent_scorecard import AgentScorecardReport
            from src.craftly.evaluation.benchmark_suites import BenchmarkSuitesReport

            assert config.executable_benchmark_report
            assert config.scorecard_report
            assert config.active_model_profile
            evaluation_path = Path(config.executable_benchmark_report).expanduser().resolve(strict=True)
            scorecard_path = Path(config.scorecard_report).expanduser().resolve(strict=True)
            profile_path = Path(config.active_model_profile).expanduser().resolve(strict=True)
            evaluation = BenchmarkSuitesReport.model_validate_json(
                evaluation_path.read_text(encoding="utf-8-sig")
            )
            scorecard = AgentScorecardReport.model_validate_json(
                scorecard_path.read_text(encoding="utf-8-sig")
            )
            active = load_active_model_contract(profile_path)
            if (
                evaluation.status != "passed"
                or evaluation.evaluation_mode != "executable_model"
                or not evaluation.suites
                or any(
                    suite.status != "passed"
                    or suite.total < 1
                    or suite.pass_at_k.get("pass@1", 0.0) <= 0.0
                    for suite in evaluation.suites
                )
            ):
                raise ValueError("executable benchmark report did not pass every measured suite")
            if (
                scorecard.status != "passed"
                or scorecard.policy_mode != "strict"
                or scorecard.task_count < 50
                or len(scorecard.tasks) != scorecard.task_count
            ):
                raise ValueError("repository scorecard is not a passed strict 50-task run")
            if active.production_blockers:
                raise ValueError("active model profile still has production blockers")
            expected = {
                "checkpoint_manifest_sha256": active.profile.evidence.get(
                    "checkpoint_manifest_sha256", ""
                ),
                "tokenizer_sha256": active.profile.evidence.get("tokenizer_sha256", ""),
                "evaluation_report_sha256": sha256_file(evaluation_path),
                "scorecard_report_sha256": sha256_file(scorecard_path),
            }
            if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in expected.values()):
                raise ValueError("active model profile contains incomplete evidence hashes")
            if active.profile.evidence != expected:
                raise ValueError("active model profile evidence does not exactly match current artifacts")
            if any(
                scorecard.model_evidence.get(key) != expected[key]
                for key in (
                    "checkpoint_manifest_sha256",
                    "tokenizer_sha256",
                    "evaluation_report_sha256",
                )
            ):
                raise ValueError("scorecard evidence does not match the active checkpoint")
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                scorecard.model_evidence.get("serving_identity_sha256", ""),
            ):
                raise ValueError("scorecard is missing live serving identity evidence")
            return _result(
                "benchmark_and_model_evidence",
                "passed",
                True,
                started,
                {
                    "evaluation_report_sha256": expected["evaluation_report_sha256"],
                    "scorecard_report_sha256": expected["scorecard_report_sha256"],
                    "active_profile_sha256": sha256_file(profile_path),
                    "suite_count": len(evaluation.suites),
                    "scorecard_tasks": scorecard.task_count,
                },
            )
        except Exception as exc:
            return _result(
                "benchmark_and_model_evidence",
                "failed",
                True,
                started,
                error=str(exc),
            )
    if not config.benchmark_dir:
        return _missing("benchmark_pack", config.strict, started, "CRAFTLY_BENCHMARK_DIR or --benchmark-dir is required")
    root = Path(config.benchmark_dir)
    try:
        from src.craftly.evaluation.benchmark_pack import BenchmarkPackConfig, run_benchmark_pack

        report = run_benchmark_pack(
            BenchmarkPackConfig(
                human_eval_path=str(root / "humaneval.jsonl"),
                mbpp_path=str(root / "mbpp.jsonl"),
                swe_bench_path=str(root / "swe_bench_style.jsonl"),
                cyberseceval_path=str(root / "cyberseceval_style.jsonl"),
                custom_security_path=str(root / "craftly_security.jsonl"),
                strict=config.strict,
                production=config.strict,
            ),
            output_dir=Path(config.output_dir) / "benchmarks",
        )
        return _result("benchmark_pack", report.status, True, started, report.model_dump())
    except Exception as exc:
        return _result("benchmark_pack", "failed", True, started, error=str(exc))


def _check_security_audit(config: ProductionProofConfig) -> ProofCheckResult:
    started = time.perf_counter()
    if not config.run_security_audit:
        return _result("security_audit", "skipped" if not config.strict else "failed", config.strict, started, {"reason": "--run-security-audit not set"})
    try:
        report = run_security_audit(
            output_dir=Path(config.output_dir) / "security-audit",
            strict_external_tools=config.strict_security_tools,
        )
        return _result("security_audit", report.status, True, started, report.model_dump())
    except Exception as exc:
        return _result("security_audit", "failed", True, started, error=str(exc))


async def run_production_proof(config: ProductionProofConfig) -> ProductionProofReport:
    checks = [
        await _check_postgres(config),
        await _check_redis_quota(config),
        await _check_object_store(config),
        await _check_qdrant(config),
        await _check_rag_control_plane(config),
        await _check_serving_health(config),
        await _check_serving_load(config),
        _check_benchmarks(config),
        _check_security_audit(config),
    ]
    failed = [item for item in checks if item.status == "failed"]
    skipped_required = [item for item in checks if item.required and item.status == "skipped"]
    recommendations = []
    for item in failed + skipped_required:
        recommendations.append(f"{item.name}: {item.error or item.details.get('reason') or 'check did not pass'}")
    report = ProductionProofReport(
        status="passed" if not failed and not skipped_required else "failed",
        mode="strict" if config.strict else "validation",
        checks=checks,
        recommendations=recommendations,
    )
    report.write(config.output_dir)
    return report


async def run_native_serving_load_test(
    *,
    endpoint: str,
    model: str,
    api_key: str | None,
    requests: int,
    concurrency: int,
    timeout_seconds: float,
    streaming_requests: int = 0,
) -> NativeServingLoadReport:
    if streaming_requests < 0 or streaming_requests > requests:
        raise ValueError("streaming_requests must be between zero and total requests")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list[str] = []
    timeout_count = 0
    streaming_passed = 0
    content_validation_failures = 0
    response_bytes = 0
    status_codes: dict[str, int] = {}
    headers = _auth_headers_value(api_key)
    timeout = httpx.Timeout(timeout_seconds)
    test_started = time.perf_counter()

    async def one(client: httpx.AsyncClient, index: int) -> None:
        nonlocal timeout_count, streaming_passed, content_validation_failures, response_bytes
        async with semaphore:
            started = time.perf_counter()
            try:
                request_payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Return a concise safe coding checklist. Request {index}.",
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 32,
                    "stream": index < streaming_requests,
                }
                if index < streaming_requests:
                    done = False
                    content_parts: list[str] = []
                    async with client.stream(
                        "POST",
                        f"{endpoint.rstrip('/')}/v1/chat/completions",
                        headers=headers,
                        json=request_payload,
                    ) as response:
                        status_codes[str(response.status_code)] = status_codes.get(
                            str(response.status_code), 0
                        ) + 1
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            response_bytes += len(line.encode("utf-8", "replace"))
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                done = True
                                break
                            payload = json.loads(data)
                            if payload.get("model") != model:
                                raise RuntimeError("streaming response model identity mismatch")
                            choices = payload.get("choices") or []
                            if choices:
                                content = choices[0].get("delta", {}).get("content")
                                if content:
                                    content_parts.append(str(content))
                    if not done or not "".join(content_parts).strip():
                        content_validation_failures += 1
                        raise RuntimeError("streaming response did not contain content and [DONE]")
                    streaming_passed += 1
                else:
                    response = await client.post(
                        f"{endpoint.rstrip('/')}/v1/chat/completions",
                        headers=headers,
                        json=request_payload,
                    )
                    status_codes[str(response.status_code)] = status_codes.get(
                        str(response.status_code), 0
                    ) + 1
                    response.raise_for_status()
                    payload = response.json()
                    raw_content = getattr(response, "content", None)
                    response_bytes += (
                        len(raw_content)
                        if isinstance(raw_content, bytes)
                        else len(canonical_json_bytes(payload))
                    )
                    choices = payload.get("choices") or []
                    content = (
                        choices[0].get("message", {}).get("content")
                        if choices and isinstance(choices[0], dict)
                        else ""
                    )
                    if (
                        payload.get("model") != model
                        or not str(content or "").strip()
                        or payload.get("craftly", {}).get("scratch_only") is not True
                    ):
                        content_validation_failures += 1
                        raise RuntimeError("non-streaming response failed model/content/scratch validation")
                latencies.append((time.perf_counter() - started) * 1000)
            except httpx.TimeoutException as exc:
                timeout_count += 1
                errors.append(f"timeout:{exc}")
            except Exception as exc:
                errors.append(str(exc)[:240])

    limits = httpx.Limits(
        max_connections=max(1, concurrency),
        max_keepalive_connections=max(1, min(concurrency, 256)),
        keepalive_expiry=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(*(one(client, index) for index in range(requests)))
    duration_seconds = max(time.perf_counter() - test_started, 1e-9)
    passed = len(latencies)
    sorted_latencies = sorted(latencies)
    p50 = statistics.median(sorted_latencies) if sorted_latencies else 0.0
    p95 = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))] if sorted_latencies else 0.0
    p99 = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.99))] if sorted_latencies else 0.0
    status = (
        "passed"
        if passed == requests
        and timeout_count == 0
        and streaming_passed == streaming_requests
        and content_validation_failures == 0
        else "failed"
    )
    return NativeServingLoadReport(
        status=status,
        endpoint=endpoint,
        requests=requests,
        concurrency=concurrency,
        passed=passed,
        failed=requests - passed,
        latency_ms_p50=round(p50, 3),
        latency_ms_p95=round(p95, 3),
        latency_ms_p99=round(p99, 3),
        max_latency_ms=round(max(sorted_latencies) if sorted_latencies else 0.0, 3),
        duration_seconds=round(duration_seconds, 6),
        throughput_rps=round(passed / duration_seconds, 6),
        response_bytes=response_bytes,
        status_codes=dict(sorted(status_codes.items())),
        timeout_count=timeout_count,
        streaming_requests=streaming_requests,
        streaming_passed=streaming_passed,
        content_validation_failures=content_validation_failures,
        error_samples=errors[:10],
    )


def _auth_headers(config: ProductionProofConfig) -> dict[str, str]:
    return _auth_headers_value(config.serving_api_key)


def _auth_headers_value(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _missing(name: str, strict: bool, started: float, reason: str) -> ProofCheckResult:
    return _result(name, "failed" if strict else "skipped", strict, started, {"reason": reason})


def _result(
    name: str,
    status: str,
    required: bool,
    started: float,
    details: dict[str, Any] | None = None,
    *,
    error: str = "",
) -> ProofCheckResult:
    return ProofCheckResult(
        name=name,
        status=status,
        required=required,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        details=details or {},
        error=error,
    )


def write_markdown(report: ProductionProofReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Production Proof Report",
        "",
        f"- status: {report.status}",
        f"- mode: {report.mode}",
        "",
        "| check | status | required | duration_ms | detail |",
        "|---|---|---:|---:|---|",
    ]
    for check in report.checks:
        detail = check.error or check.details.get("reason") or ""
        lines.append(f"| {check.name} | {check.status} | {str(check.required).lower()} | {check.duration_ms:.3f} | {detail} |")
    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in report.recommendations:
            lines.append(f"- {item}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live Craftly production proof checks.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/craftly/production-proof")
    parser.add_argument("--postgres-url")
    parser.add_argument("--apply-postgres-migrations", action="store_true")
    parser.add_argument("--redis-url")
    parser.add_argument("--object-store-uri")
    parser.add_argument("--object-store-endpoint-url")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--run-rag-control-plane", action="store_true")
    parser.add_argument(
        "--allow-insecure-service-host",
        action="append",
        default=[],
        help="Exact private service hostname allowed to use HTTP; repeat for each host.",
    )
    parser.add_argument("--serving-url")
    parser.add_argument("--serving-api-key")
    parser.add_argument("--serving-model", default="craftly-scratch")
    parser.add_argument("--load-test-requests", type=int, default=0)
    parser.add_argument("--load-test-concurrency", type=int, default=4)
    parser.add_argument("--load-test-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--load-test-streaming-requests", type=int, default=0)
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--executable-benchmark-report")
    parser.add_argument("--scorecard-report")
    parser.add_argument("--active-model-profile")
    parser.add_argument("--run-security-audit", action="store_true")
    parser.add_argument("--strict-security-tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = config_from_env(parse_args())
    report = asyncio.run(run_production_proof(config))
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
