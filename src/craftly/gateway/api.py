"""Consolidated Craftly Gateway API."""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.craftly.db import LocalStore, PostgresRAGStore, PostgresRAGStoreFactory
from src.craftly.evaluation.benchmarks import BenchmarkHarness, built_in_security_tasks
from src.craftly.identity import AuthError, auth_status, create_jwt, install_auth, install_quota, validate_token_issue_request
from src.craftly.indexing import ContextBuilder, RepositoryIndexer, VectorBackendConfig, create_vector_index, vector_capabilities
from src.craftly.indexing.repository_indexer import RAGIndexCoordinator
from src.craftly.learning.dataset_authority import (
    DatasetAuthorityService,
    PromotionEvidenceCreate,
    ReviewAdjudicationCreate,
    ReviewDecisionCreate,
    ReviewItemCreate,
    SourceSnapshotCreate,
)
from src.craftly.learning.production_dataset import (
    ProductionDatasetManifest,
    validate_dataset_manifest_for_promotion,
)
from src.craftly.learning.versioning import DatasetLedger
from src.craftly.memory import MemoryIngestRequest, UnifiedMemoryManager
from src.craftly.model_ops.backends import active_model_health, list_model_profiles
from src.craftly.model_ops.foundation import PretrainingRunSpec, foundation_status
from src.craftly.observability import METRICS, install_observability
from src.craftly.patches import PatchPreviewRequest, PatchService, PatchVerifyRequest
from src.craftly.runtime.collaboration import AgentMessage, BlackboardKind, BlackboardWrite, CollaborationRuntime, FailureIntelligence
from src.craftly.runtime.engine import CraftlyRuntime
from src.craftly.runtime.execution import AgentExecutionReport, AgentExecutionRequest, AgentExecutionService
from src.craftly.runtime.taskgraph import AgentRunCreateRequest, TaskCompleteRequest, TaskFailRequest, TaskGraphRuntime
from src.craftly.shared.schemas import CraftlyRunRequest, CraftlyRunReport
from src.craftly.tools import DockerSandboxRunner, HardenedToolExecutor, SandboxRunRequest, ToolExecuteRequest
from src.craftly.training_workspace import (
    ArtifactUploadRequest,
    CheckpointCommitRequest,
    EvaluationCommitRequest,
    JobStatus,
    ServiceAccount,
    TrainingArtifact,
    TrainingEventBatch,
    TrainingJob,
    TrainingJobCreateRequest,
    TrainingWorkspaceService,
    workspace_readiness,
)
from src.craftly.verifier import VerificationRequest, VerifierRuntime

app = FastAPI(title="Craftly Consolidated Gateway", version="2.0.0")
QUOTA_CONFIG = install_quota(app)
AUTH_CONFIG = install_auth(app)
install_observability(app)
STORE = LocalStore()
RAG_STORE_FACTORY: PostgresRAGStoreFactory | None = None
RAG_STORE_FACTORY_DSN = ""
RAG_STORE_FACTORY_LOCK = threading.RLock()
TRAINING_WORKSPACE = TrainingWorkspaceService.from_environment()
DATASET_AUTHORITY = DatasetAuthorityService.from_environment()


class AuthTokenRequest(BaseModel):
    user_id: str = Field(default="local-user", min_length=1)
    organization_id: str = Field(default="local", min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: ["api"])
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    issue_key: str | None = Field(default=None, min_length=1)


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    repo_path: str = Field(min_length=1)
    default_branch: str = Field(default="main", min_length=1, max_length=100)


SAFE_REPO_PATH_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,254}$")


def resolve_registered_repo_path(value: str) -> Path:
    """Resolve an API repository path beneath an administrator-owned root."""

    if "\x00" in value or len(value) > 4096:
        raise ValueError("repo_path contains invalid data")
    normalized = value.replace("\\", "/").strip()
    if normalized in {"", "."}:
        parts: list[str] = []
    else:
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
            raise ValueError("repo_path must be relative to an allowed project root")
        parts = normalized.split("/")
        if any(part in {"", ".", ".."} or SAFE_REPO_PATH_PART.fullmatch(part) is None for part in parts):
            raise ValueError("repo_path contains an unsafe path component")

    configured = [item for item in os.environ.get("CRAFTLY_PROJECT_ROOTS", "").split(os.pathsep) if item]
    roots = [Path(item).expanduser().resolve() for item in configured] if configured else [Path.cwd().resolve()]
    for root in roots:
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_dir():
            return candidate
    raise ValueError("repo_path is not a directory beneath an allowed project root")


class IndexProjectRequest(BaseModel):
    force: bool = False
    include_suffixes: list[str] | None = None
    max_file_bytes: int = Field(default=1_000_000, ge=10_000, le=10_000_000)
    max_chunk_lines: int = Field(default=120, ge=20, le=400)
    max_chunk_tokens: int = Field(default=512, ge=128, le=4096)
    overlap_tokens: int = Field(default=64, ge=0, le=1024)
    max_attempts: int = Field(default=3, ge=1, le=20)


class ContextBuildRequest(BaseModel):
    project_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    token_budget: int = Field(default=24_000, ge=1_000, le=200_000)
    pinned_files: list[str] = Field(default_factory=list)
    max_chunks: int = Field(default=24, ge=1, le=100)


class VectorSearchRequest(BaseModel):
    project_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=12, ge=1, le=100)
    backend: str = Field(default="local_hashing", pattern="^(local_hashing|faiss|hnsw|qdrant|pgvector)$")
    dims: int = Field(default=384, ge=64, le=4096)


class VectorSyncRequest(BaseModel):
    project_id: str = Field(min_length=1)
    backend: str = Field(default="qdrant", pattern="^(local_hashing|qdrant)$")
    dims: int = Field(default=384, ge=64, le=4096)
    batch_size: int = Field(default=64, ge=1, le=256)


class MemoryRetrieveRequest(BaseModel):
    project_id: str | None = None
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    layers: list[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    project_id: str = Field(min_length=1)
    title: str = Field(default="Craftly Session", min_length=1, max_length=200)


class TrainingServiceAccountRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: list[str] = Field(min_length=1, max_length=16)


class TrainingTokenExchangeRequest(BaseModel):
    bootstrap_token: str = Field(min_length=32, max_length=512)


class TrainingTokenRefreshRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    refresh_token: str = Field(min_length=32, max_length=512)


class TrainingArtifactRegistrationRequest(BaseModel):
    upload: ArtifactUploadRequest
    uri: str = Field(min_length=1, max_length=4096)


class DatasetReviewClaimRequest(BaseModel):
    lease_seconds: int = Field(default=3_600, ge=300, le=14_400)


SAFE_DATASET_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,159}$")


def require_scope(request: Request, scope: str) -> None:
    if not AUTH_CONFIG.enabled:
        return
    claims = getattr(request.state, "jwt_claims", {}) or {}
    scopes = set(str(item) for item in claims.get("scopes", []) if item)
    namespace = scope.split(":", 1)[0]
    if namespace in {"training", "data", "tools", "agents", "context", "memory"}:
        allowed = scope in scopes or f"{namespace}:admin" in scopes
    else:
        allowed = scope in scopes or "api" in scopes
    if not allowed:
        raise PermissionError(f"missing required scope: {scope}")


def request_scopes(request: Request) -> set[str]:
    claims = getattr(request.state, "jwt_claims", {}) or {}
    return {str(item) for item in claims.get("scopes", []) if item}


def request_owner(request: Request) -> str:
    return str(getattr(request.state, "user_id", "") or "local-user")


def request_organization(request: Request) -> str:
    if not AUTH_CONFIG.enabled:
        return "local"
    claims = getattr(request.state, "jwt_claims", {}) or {}
    organization_id = str(claims.get("organization_id") or "")
    if not organization_id or len(organization_id) > 128:
        raise PermissionError("JWT requires an immutable organization_id claim")
    return organization_id


def rag_store(organization_id: str) -> LocalStore | PostgresRAGStore:
    global RAG_STORE_FACTORY, RAG_STORE_FACTORY_DSN
    if os.environ.get("CRAFTLY_ENV", "development").lower() != "production":
        return STORE
    dsn = os.environ.get("CRAFTLY_DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("production RAG requires CRAFTLY_DATABASE_URL")
    with RAG_STORE_FACTORY_LOCK:
        if RAG_STORE_FACTORY is None:
            RAG_STORE_FACTORY = PostgresRAGStoreFactory(
                dsn,
                min_pool_size=int(os.environ.get("CRAFTLY_RAG_DB_POOL_MIN", "1")),
                max_pool_size=int(os.environ.get("CRAFTLY_RAG_DB_POOL_MAX", "20")),
            )
            RAG_STORE_FACTORY_DSN = dsn
        elif dsn != RAG_STORE_FACTORY_DSN:
            raise RuntimeError("CRAFTLY_DATABASE_URL changed after the shared RAG pool was initialized")
        return RAG_STORE_FACTORY.for_organization(organization_id)


def require_project_access(request: Request, project_id: str, scope: str) -> dict[str, Any]:
    require_scope(request, scope)
    organization_id = request_organization(request)
    return rag_store(organization_id).require_project_access(
        project_id,
        organization_id,
        user_id=request_owner(request) if AUTH_CONFIG.enabled else None,
    )


def production_vector_config() -> VectorBackendConfig:
    production = os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
    return VectorBackendConfig(
        backend="qdrant",
        dims=int(os.environ.get("CRAFTLY_EMBEDDING_DIMS", "768")),
        qdrant_url=os.environ.get("CRAFTLY_QDRANT_URL"),
        embedding_url=os.environ.get("CRAFTLY_EMBEDDING_URL"),
        embedding_model=os.environ.get("CRAFTLY_EMBEDDING_MODEL", "Craftly-Code-Embed-v1"),
        embedding_manifest_path=os.environ.get("CRAFTLY_EMBEDDING_MANIFEST"),
        production_mode=production,
    )


def _dataset_version_manifest_path(version_id: str) -> Path:
    if SAFE_DATASET_VERSION.fullmatch(version_id) is None:
        raise ValueError("invalid dataset version identifier")
    root = Path(os.environ.get("CRAFTLY_DATASET_VERSIONS_ROOT", "data/production")).expanduser().resolve()
    candidate = (root / version_id / "dataset_version_manifest.json").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset version path escapes configured root") from exc
    if not candidate.is_file():
        direct = (root / "dataset_version_manifest.json").resolve()
        if direct.is_file():
            payload = ProductionDatasetManifest.model_validate_json(direct.read_text(encoding="utf-8"))
            if payload.version_id == version_id:
                return direct
        raise FileNotFoundError("dataset version manifest not found")
    return candidate


async def require_training_job_access(request: Request, job_id: str, scope: str) -> TrainingJob:
    require_scope(request, scope)
    job = await TRAINING_WORKSPACE.get_job(job_id)
    claims = getattr(request.state, "jwt_claims", {}) or {}
    job_bound = str(claims.get("job_id") or "") == job_id
    if AUTH_CONFIG.enabled and "training:admin" not in request_scopes(request) and not job_bound and job.owner_id != request_owner(request):
        raise PermissionError("training job belongs to a different workspace identity")
    return job


@app.get("/health/ready")
async def health_ready() -> dict[str, object]:
    training = workspace_readiness(TRAINING_WORKSPACE)
    return {
        "status": "ready",
        "model_ops": active_model_health(),
        "auth": auth_status(AUTH_CONFIG),
        "quota": {"enabled": QUOTA_CONFIG.enabled, "capacity": QUOTA_CONFIG.capacity},
        "database": {"ok": True, "engine": "sqlite-local", "path": str(STORE.path)},
        "training_workspace": training,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return METRICS.render_prometheus()


@app.get("/v1/auth/status")
async def auth_status_endpoint() -> dict[str, object]:
    return auth_status(AUTH_CONFIG)


@app.post("/v1/auth/token")
async def issue_auth_token(request: AuthTokenRequest) -> dict[str, object]:
    if not AUTH_CONFIG.jwt_secret:
        raise HTTPException(status_code=503, detail="CRAFTLY_JWT_SECRET is required")
    try:
        validate_token_issue_request(AUTH_CONFIG, request.issue_key)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    token = create_jwt(
        subject=request.user_id,
        secret=AUTH_CONFIG.jwt_secret,
        issuer=AUTH_CONFIG.issuer,
        audience=AUTH_CONFIG.audience,
        scopes=request.scopes,
        ttl_seconds=request.ttl_seconds,
        extra_claims={"organization_id": request.organization_id},
    )
    return {
        "token_type": "Bearer",  # nosec B105 - OAuth token type label, not a credential.
        "access_token": token,
        "expires_in": request.ttl_seconds,
        "user_id": request.user_id,
    }


def training_access_token(account: ServiceAccount) -> str:
    if not AUTH_CONFIG.jwt_secret:
        raise HTTPException(status_code=503, detail="CRAFTLY_JWT_SECRET is required for workspace tokens")
    return create_jwt(
        subject=account.service_account_id,
        secret=AUTH_CONFIG.jwt_secret,
        issuer=AUTH_CONFIG.issuer,
        audience=AUTH_CONFIG.audience,
        scopes=account.scopes,
        ttl_seconds=900,
    )


@app.get("/v1/training/readiness")
async def training_workspace_readiness() -> dict[str, Any]:
    return workspace_readiness(TRAINING_WORKSPACE)


@app.post("/v1/training/service-accounts")
async def create_training_service_account(request: TrainingServiceAccountRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:admin")
        credential = await TRAINING_WORKSPACE.create_service_account(name=request.name, scopes=request.scopes)
        return credential.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/training/token/exchange")
async def exchange_training_token(request: TrainingTokenExchangeRequest) -> dict[str, Any]:
    account = await TRAINING_WORKSPACE.authenticate_bootstrap(request.bootstrap_token)
    if account is None:
        raise HTTPException(status_code=401, detail="invalid workspace bootstrap credential")
    session = await TRAINING_WORKSPACE.create_refresh_session(account)
    return {
        "token_type": "Bearer",  # nosec B105 - OAuth token type label, not a credential.
        "access_token": training_access_token(account),
        "expires_in": 900,
        "session_id": session.session_id,
        "refresh_token": session.refresh_token,
        "refresh_expires_at": session.expires_at.isoformat(),
        "service_account_id": account.service_account_id,
        "scopes": account.scopes,
    }


@app.post("/v1/training/token/refresh")
async def refresh_training_token(request: TrainingTokenRefreshRequest) -> dict[str, Any]:
    account = await TRAINING_WORKSPACE.authenticate_refresh(request.session_id, request.refresh_token)
    if account is None:
        raise HTTPException(status_code=401, detail="refresh session is invalid, expired, or revoked")
    return {
        "token_type": "Bearer",  # nosec B105 - OAuth token type label, not a credential.
        "access_token": training_access_token(account),
        "expires_in": 900,
        "service_account_id": account.service_account_id,
        "scopes": account.scopes,
    }


@app.post("/v1/training/token/revoke")
async def revoke_training_token(request: TrainingTokenRefreshRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:jobs:read")
        revoked = await TRAINING_WORKSPACE.revoke_refresh(request.session_id, request.refresh_token)
        if not revoked:
            raise HTTPException(status_code=404, detail="refresh session was not found")
        return {"revoked": True}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v1/training/profiles")
async def training_profiles(http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:jobs:read")
        return {"profiles": TRAINING_WORKSPACE.profile_report()}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v1/training/campaigns")
async def training_campaigns(http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:jobs:read")
        return {
            "campaigns": [
                {
                    **campaign.model_dump(mode="json"),
                    "milestones": [item.model_dump(mode="json") for item in campaign.milestones],
                }
                for campaign in TRAINING_WORKSPACE.campaigns.campaigns
            ]
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v1/training/jobs")
async def create_training_job(request: TrainingJobCreateRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:jobs:create")
        job = await TRAINING_WORKSPACE.create_job(request, owner_id=request_owner(http_request))
        METRICS.inc("craftly_training_jobs_total", profile=job.spec.profile_id, status=job.status.value)
        return job.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/training/jobs")
async def list_training_jobs(http_request: Request, limit: int = 100) -> dict[str, Any]:
    try:
        require_scope(http_request, "training:jobs:read")
        owner = None if "training:admin" in request_scopes(http_request) else request_owner(http_request)
        jobs = await TRAINING_WORKSPACE.list_jobs(owner_id=owner, limit=limit)
        return {"jobs": [job.model_dump(mode="json") for job in jobs]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v1/training/jobs/{job_id}")
async def get_training_job(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        job = await require_training_job_access(http_request, job_id, "training:jobs:read")
        return job.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:jobs:cancel")
        job = await TRAINING_WORKSPACE.cancel_job(job_id, actor_id=request_owner(http_request))
        METRICS.inc("craftly_training_job_transitions_total", target="cancelled")
        return job.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/resume")
async def resume_training_job(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:jobs:create")
        job = await TRAINING_WORKSPACE.resume_job(job_id, actor_id=request_owner(http_request))
        METRICS.inc("craftly_training_job_transitions_total", target="queued")
        return job.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/worker-token")
async def issue_training_worker_token(job_id: str, http_request: Request, ttl_seconds: int = 21_600) -> dict[str, Any]:
    try:
        job = await require_training_job_access(http_request, job_id, "training:jobs:create")
        if not AUTH_CONFIG.jwt_secret:
            raise HTTPException(status_code=503, detail="CRAFTLY_JWT_SECRET is required for worker tokens")
        ttl = min(max(ttl_seconds, 900), 86_400)
        token = create_jwt(
            subject=f"worker:{job.job_id}",
            secret=AUTH_CONFIG.jwt_secret,
            issuer=AUTH_CONFIG.issuer,
            audience=AUTH_CONFIG.audience,
            scopes=["training:events:write", "training:artifacts:write", "training:jobs:read"],
            ttl_seconds=ttl,
            extra_claims={"job_id": job.job_id, "token_class": "training_worker"},
        )
        return {"token_type": "Bearer", "access_token": token, "expires_in": ttl, "job_id": job.job_id}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/claim")
async def claim_notebook_training_job(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:jobs:create")
        job, attempt = await TRAINING_WORKSPACE.claim_notebook_job(job_id)
        if not AUTH_CONFIG.jwt_secret:
            raise HTTPException(status_code=503, detail="CRAFTLY_JWT_SECRET is required for notebook worker tokens")
        token = create_jwt(
            subject=f"worker:{job.job_id}",
            secret=AUTH_CONFIG.jwt_secret,
            issuer=AUTH_CONFIG.issuer,
            audience=AUTH_CONFIG.audience,
            scopes=["training:events:write", "training:artifacts:write", "training:jobs:read"],
            ttl_seconds=21_600,
            extra_claims={"job_id": job.job_id, "token_class": "notebook_worker"},
        )
        return {
            "job": job.model_dump(mode="json"),
            "attempt": attempt.model_dump(mode="json"),
            "worker_access_token": token,
            "worker_token_expires_in": 21_600,
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/events:batch")
async def ingest_training_events(job_id: str, request: TrainingEventBatch, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:events:write")
        events = await TRAINING_WORKSPACE.ingest_events(job_id, request)
        METRICS.inc("craftly_training_events_total", value=float(len(events)), kind="batch")
        if not events:
            METRICS.inc("craftly_training_event_duplicates_total", value=float(len(request.events)))
            return {"accepted": 0, "duplicates": len(request.events), "first_sequence": None, "last_sequence": None, "last_event_id": None}
        profile_id = (await TRAINING_WORKSPACE.get_job(job_id)).spec.profile_id
        for event in events:
            if event.loss is not None:
                METRICS.observe("craftly_training_loss", event.loss, profile=profile_id)
            if event.validation_loss is not None:
                METRICS.observe("craftly_training_validation_loss", event.validation_loss)
            if event.tokens_per_second is not None:
                METRICS.observe("craftly_training_tokens_per_second", event.tokens_per_second)
        return {
            "accepted": len(events),
            "duplicates": len(request.events) - len(events),
            "first_sequence": events[0].sequence,
            "last_sequence": events[-1].sequence,
            "last_event_id": events[-1].event_id,
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/training/jobs/{job_id}/audit")
async def list_training_audit(job_id: str, http_request: Request, limit: int = 100) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:jobs:read")
        events = await TRAINING_WORKSPACE.list_audit_events(job_id, limit=limit)
        return {"audit_events": [event.model_dump(mode="json") for event in events]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/training/jobs/{job_id}/events")
async def stream_training_events(job_id: str, http_request: Request, after_sequence: int = 0) -> StreamingResponse:
    try:
        await require_training_job_access(http_request, job_id, "training:jobs:read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    header_cursor = http_request.headers.get("last-event-id", "").strip()
    if header_cursor:
        try:
            after_sequence = max(after_sequence, int(header_cursor))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer sequence") from exc

    async def generate() -> Any:
        async for event in TRAINING_WORKSPACE.stream_events(job_id, after_sequence=after_sequence):
            if await http_request.is_disconnected():
                return
            if event is None:
                yield ": heartbeat\n\n"
                continue
            payload = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            yield f"id: {event.sequence}\nevent: {event.kind}\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/training/jobs/{job_id}/artifacts")
async def list_training_artifacts(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:read")
        artifacts = await TRAINING_WORKSPACE.list_artifacts(job_id)
        return {"artifacts": [artifact.model_dump(mode="json") for artifact in artifacts]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/artifacts/presign")
async def presign_training_artifact(job_id: str, request: ArtifactUploadRequest, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:write")
        return await TRAINING_WORKSPACE.presign_artifact_upload(job_id, request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/artifacts/register")
async def register_training_artifact(
    job_id: str,
    request: TrainingArtifactRegistrationRequest,
    http_request: Request,
) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:write")
        artifact = await TRAINING_WORKSPACE.verify_and_register_artifact(job_id, request.upload, uri=request.uri)
        return artifact.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/checkpoints")
async def commit_training_checkpoint(job_id: str, request: CheckpointCommitRequest, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:write")
        checkpoint = await TRAINING_WORKSPACE.commit_checkpoint(job_id, request)
        METRICS.inc("craftly_training_checkpoints_total", status="committed")
        return checkpoint.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/training/jobs/{job_id}/checkpoints")
async def list_training_checkpoints(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:read")
        rows = await TRAINING_WORKSPACE.list_checkpoints(job_id)
        return {"checkpoints": [item.model_dump(mode="json") for item in rows]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/evaluations")
async def commit_training_evaluation(job_id: str, request: EvaluationCommitRequest, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:write")
        evaluation = await TRAINING_WORKSPACE.commit_evaluation(job_id, request)
        METRICS.inc("craftly_training_evaluations_total", decision=evaluation.decision)
        return evaluation.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/training/jobs/{job_id}/evaluations")
async def list_training_evaluations(job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:artifacts:read")
        rows = await TRAINING_WORKSPACE.list_evaluations(job_id)
        return {"evaluations": [item.model_dump(mode="json") for item in rows]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/promote")
async def promote_training_checkpoint(job_id: str, checkpoint_id: str, http_request: Request) -> dict[str, Any]:
    try:
        await require_training_job_access(http_request, job_id, "training:admin")
        checkpoint = await TRAINING_WORKSPACE.promote_checkpoint(
            job_id,
            checkpoint_id,
            actor_id=request_owner(http_request),
        )
        METRICS.inc("craftly_training_checkpoints_total", status="promoted")
        return checkpoint.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/model/profiles")
async def model_profiles() -> dict[str, object]:
    return {"profiles": list_model_profiles()}


@app.post("/v1/evaluation/security-static")
async def run_security_static_benchmark() -> dict[str, object]:
    return BenchmarkHarness().run_static(built_in_security_tasks()).model_dump()


@app.get("/v1/model/foundation/status")
async def model_foundation_status() -> dict[str, object]:
    return foundation_status()


@app.get("/v1/data/platform/status")
async def data_platform_status() -> dict[str, object]:
    root = Path(os.environ.get("CRAFTLY_DATA_PIPELINE_DIR", "artifacts/craftly/data-pipeline"))
    latest = DatasetLedger(root / "versions" / "ledger.jsonl").latest()
    dashboard = root / "dashboard.html"
    return {
        "status": "ready" if latest else "no_dataset_versions",
        "pipeline_dir": str(root),
        "latest_version": latest.model_dump() if latest else None,
        "dashboard_path": str(dashboard) if dashboard.exists() else None,
    }


@app.post("/v1/data/reviews")
async def enqueue_dataset_review(request: ReviewItemCreate, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:review")
        return (await DATASET_AUTHORITY.enqueue(request)).model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/data/source-snapshots")
async def record_dataset_source_snapshot(request: SourceSnapshotCreate, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:audit")
        return (await DATASET_AUTHORITY.record_source_snapshot(request)).model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/data/reviews")
async def list_dataset_reviews(
    http_request: Request,
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:review")
        items = await DATASET_AUTHORITY.list_items(
            status=status,
            limit=max(1, min(500, limit)),
            reviewer_id=request_owner(http_request),
        )
        return {"items": [item.model_dump() for item in items]}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/data/reviews/evidence")
async def dataset_review_evidence(http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:audit")
        return (await DATASET_AUTHORITY.review_evidence()).model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v1/data/reviews/{record_id}/claim")
async def claim_dataset_review(
    record_id: str,
    request: DatasetReviewClaimRequest,
    http_request: Request,
) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:review")
        assignment = await DATASET_AUTHORITY.claim(record_id, request_owner(http_request), request.lease_seconds)
        return assignment.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/data/reviews/{record_id}/decision")
async def decide_dataset_review(
    record_id: str,
    request: ReviewDecisionCreate,
    http_request: Request,
) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:review")
        item = await DATASET_AUTHORITY.decide(record_id, request_owner(http_request), request)
        return item.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/data/reviews/{record_id}/adjudicate")
async def adjudicate_dataset_review(
    record_id: str,
    request: ReviewAdjudicationCreate,
    http_request: Request,
) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:adjudicate")
        item = await DATASET_AUTHORITY.adjudicate(record_id, request_owner(http_request), request)
        return item.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/data/versions/{version_id}")
async def get_dataset_version(version_id: str, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:audit")
        path = _dataset_version_manifest_path(version_id)
        return ProductionDatasetManifest.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/data/versions/{version_id}/promote")
async def promote_dataset_version(version_id: str, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "data:promote")
        path = _dataset_version_manifest_path(version_id)
        policy_path = os.environ.get("CRAFTLY_DATASET_TRUST_POLICY", "config/dataset_trust_policy.json")
        manifest = validate_dataset_manifest_for_promotion(path, trust_policy_path=policy_path)
        manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        event = await DATASET_AUTHORITY.record_promotion(
            request_owner(http_request),
            PromotionEvidenceCreate(
                dataset_id=manifest.dataset_id,
                version_id=manifest.version_id,
                manifest_sha256=manifest_sha256,
                policy_sha256=manifest.policy_sha256,
                decision="promoted",
                evidence={
                    "artifact_sha256": manifest.artifact_sha256,
                    "promotion_decision": manifest.promotion_decision,
                },
            ),
        )
        return event.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/model/foundation/pretraining/readiness")
async def model_pretraining_readiness(request: PretrainingRunSpec) -> dict[str, object]:
    return request.readiness_report()


@app.post("/v1/projects")
async def create_project(payload: ProjectCreateRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "context:index")
        organization_id = request_organization(http_request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        repo_path = resolve_registered_repo_path(payload.repo_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await asyncio.to_thread(
        rag_store(organization_id).create_project,
        name=payload.name,
        repo_path=str(repo_path),
        default_branch=payload.default_branch,
        organization_id=organization_id,
        owner_user_id=request_owner(http_request),
    )


@app.get("/v1/projects/{project_id}")
async def get_project(project_id: str, request: Request) -> dict[str, Any]:
    try:
        return require_project_access(request, project_id, "context:read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.post("/v1/sessions")
async def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    try:
        return STORE.create_session(project_id=request.project_id, title=request.title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.post("/v1/projects/{project_id}/index")
async def index_project(project_id: str, payload: IndexProjectRequest, http_request: Request) -> dict[str, Any]:
    try:
        project = require_project_access(http_request, project_id, "context:index")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    organization_id = request_organization(http_request)
    active_store = rag_store(organization_id)
    if project["index_status"] == "ready" and not payload.force:
        return await asyncio.to_thread(active_store.index_status, project_id)
    if payload.overlap_tokens >= payload.max_chunk_tokens:
        raise HTTPException(status_code=400, detail="overlap_tokens must be smaller than max_chunk_tokens")
    suffixes = set(payload.include_suffixes) if payload.include_suffixes else None
    try:
        production = os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
        if production:
            idempotency_key = http_request.headers.get("Idempotency-Key", "")
            if not idempotency_key:
                raise ValueError("production index submission requires Idempotency-Key")
            coordinator = RAGIndexCoordinator(active_store, production_mode=True)
            try:
                return await coordinator.submit(
                    organization_id=organization_id,
                    project_id=project_id,
                    idempotency_key=idempotency_key,
                    max_attempts=payload.max_attempts,
                    request={
                        "include_suffixes": sorted(suffixes) if suffixes else None,
                        "max_file_bytes": payload.max_file_bytes,
                        "max_chunk_lines": payload.max_chunk_lines,
                        "max_chunk_tokens": payload.max_chunk_tokens,
                        "overlap_tokens": payload.overlap_tokens,
                    },
                )
            finally:
                await coordinator.close()
        report = await asyncio.to_thread(
            RepositoryIndexer(active_store).index_project,
            project_id=project_id,
            organization_id=organization_id,
            include_suffixes=suffixes,
            max_file_bytes=payload.max_file_bytes,
            max_chunk_lines=payload.max_chunk_lines,
            max_chunk_tokens=payload.max_chunk_tokens,
            overlap_tokens=payload.overlap_tokens,
        )
    except Exception as exc:
        status = 400 if isinstance(exc, (ValueError, PermissionError)) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return report.model_dump()


@app.get("/v1/projects/{project_id}/index/status")
async def get_project_index_status(project_id: str, http_request: Request) -> dict[str, Any]:
    try:
        require_project_access(http_request, project_id, "context:read")
        organization_id = request_organization(http_request)
        return await asyncio.to_thread(rag_store(organization_id).index_status, project_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.get("/v1/projects/{project_id}/index/jobs")
async def list_project_index_jobs(
    project_id: str,
    http_request: Request,
    limit: int = 100,
) -> dict[str, Any]:
    try:
        require_project_access(http_request, project_id, "context:read")
        organization_id = request_organization(http_request)
        jobs = await asyncio.to_thread(
            rag_store(organization_id).list_index_jobs,
            organization_id=organization_id,
            project_id=project_id,
            limit=limit,
        )
        return {"project_id": project_id, "jobs": jobs}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.get("/v1/projects/{project_id}/index/jobs/{job_id}")
async def get_project_index_job(project_id: str, job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        require_project_access(http_request, project_id, "context:read")
        organization_id = request_organization(http_request)
        job = await asyncio.to_thread(
            rag_store(organization_id).get_index_job,
            job_id,
            organization_id=organization_id,
        )
        if job is None or str(job["project_id"]) != project_id:
            raise KeyError("index job not found")
        return job
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/projects/{project_id}/index/jobs/{job_id}/cancel")
async def cancel_project_index_job(project_id: str, job_id: str, http_request: Request) -> dict[str, Any]:
    try:
        require_project_access(http_request, project_id, "context:index")
        organization_id = request_organization(http_request)
        active_store = rag_store(organization_id)
        job = await asyncio.to_thread(active_store.get_index_job, job_id, organization_id=organization_id)
        if job is None or str(job["project_id"]) != project_id:
            raise KeyError("index job not found")
        return await asyncio.to_thread(
            active_store.request_index_job_cancel,
            job_id,
            organization_id=organization_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/agent/runs")
async def create_agent_run(request: AgentRunCreateRequest) -> dict[str, Any]:
    try:
        return TaskGraphRuntime(STORE).create_agent_run(request).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project or session not found") from exc


@app.post("/v1/agent/execute", response_model=AgentExecutionReport)
async def execute_agent_workflow(
    payload: AgentExecutionRequest,
    http_request: Request,
) -> AgentExecutionReport:
    """Run architect-to-verifier workflow and mutate only an accepted patch."""

    started = time.perf_counter()
    try:
        require_scope(http_request, "agents:execute")
        report = await AgentExecutionService(STORE).execute(payload)
        METRICS.inc(
            "craftly_agent_executions_total",
            status=report.status,
            applied=str(report.applied).lower(),
        )
        METRICS.observe(
            "craftly_agent_execution_duration_ms",
            report.total_duration_ms,
            status=report.status,
        )
        METRICS.observe(
            "craftly_agent_revision_attempts",
            float(len(report.attempts)),
            status=report.status,
        )
        return report
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        METRICS.observe(
            "craftly_agent_gateway_duration_ms",
            (time.perf_counter() - started) * 1000,
        )


@app.get("/v1/agent/runs/{run_id}")
async def get_agent_run(run_id: str) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/v1/taskgraphs/{task_graph_id}")
async def get_task_graph(task_graph_id: str) -> dict[str, Any]:
    graph = STORE.get_task_graph(task_graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="task graph not found")
    return graph


@app.post("/v1/taskgraphs/{task_graph_id}/advance")
async def advance_task_graph(task_graph_id: str) -> dict[str, Any]:
    try:
        return TaskGraphRuntime(STORE).advance(task_graph_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task graph not found") from exc


@app.post("/v1/taskgraphs/{task_graph_id}/cancel")
async def cancel_task_graph(task_graph_id: str, request: Request) -> dict[str, Any]:
    try:
        require_scope(request, "agents:control")
        return TaskGraphRuntime(STORE).cancel(task_graph_id).model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task graph not found") from exc


@app.post("/v1/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: TaskCompleteRequest) -> dict[str, Any]:
    try:
        return TaskGraphRuntime(STORE).complete_task(task_id, request).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@app.post("/v1/tasks/{task_id}/fail")
async def fail_task(task_id: str, request: TaskFailRequest) -> dict[str, Any]:
    try:
        return TaskGraphRuntime(STORE).fail_task(task_id, request).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@app.post("/v1/agent/messages")
async def publish_agent_message(message: AgentMessage, request: Request) -> dict[str, Any]:
    try:
        require_scope(request, "agents:messages:write")
        return CollaborationRuntime(STORE).publish(message).model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/agent/runs/{run_id}/messages")
async def list_agent_messages(
    run_id: str,
    request: Request,
    correlation_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    try:
        require_scope(request, "agents:messages:read")
        return [
            item.model_dump(mode="json")
            for item in CollaborationRuntime(STORE).history(run_id, correlation_id=correlation_id, limit=limit)
        ]
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/v1/agent/blackboard")
async def write_agent_blackboard(entry: BlackboardWrite, request: Request) -> dict[str, Any]:
    try:
        require_scope(request, "agents:blackboard:write")
        return CollaborationRuntime(STORE).write_blackboard(entry).model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/agent/runs/{run_id}/blackboard")
async def read_agent_blackboard(
    run_id: str,
    request: Request,
    kind: BlackboardKind | None = None,
) -> list[dict[str, Any]]:
    try:
        require_scope(request, "agents:blackboard:read")
        return [item.model_dump(mode="json") for item in CollaborationRuntime(STORE).blackboard(run_id, kind=kind)]
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/projects/{project_id}/failure-clusters")
async def list_failure_clusters(project_id: str, request: Request) -> list[dict[str, Any]]:
    try:
        require_scope(request, "agents:failures:read")
        if STORE.get_project(project_id) is None:
            raise KeyError(project_id)
        return FailureIntelligence(STORE).clusters(project_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.get("/v1/projects/{project_id}/symbols")
async def project_symbols(project_id: str, request: Request) -> dict[str, Any]:
    try:
        require_project_access(request, project_id, "context:read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    organization_id = request_organization(request)
    chunks = await asyncio.to_thread(rag_store(organization_id).list_chunks, project_id)
    symbols: list[dict[str, Any]] = []
    dependencies: dict[str, set[str]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        path = str(chunk.get("path") or "")
        dependency_values = metadata.get("dependencies", [])
        if not isinstance(dependency_values, list):
            dependency_values = [dependency_values] if dependency_values else []
        if path:
            dependencies.setdefault(path, set()).update(str(item) for item in dependency_values if item)
        if not chunk.get("symbol_name"):
            continue
        symbols.append(
            {
                "path": path,
                "language": chunk.get("language"),
                "symbol_name": chunk["symbol_name"],
                "kind": chunk["kind"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "signature": metadata.get("signature", ""),
                "calls": metadata.get("calls", []),
                "state_mutations": metadata.get("state_mutations", []),
            }
        )
    return {
        "project_id": project_id,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "dependencies": {path: sorted(values) for path, values in sorted(dependencies.items())},
    }


@app.post("/v1/context/build")
async def build_context(payload: ContextBuildRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_project_access(http_request, payload.project_id, "context:read")
        organization_id = request_organization(http_request)
        report = await asyncio.to_thread(
            ContextBuilder(rag_store(organization_id)).build,
            project_id=payload.project_id,
            organization_id=organization_id,
            query=payload.query,
            token_budget=payload.token_budget,
            pinned_files=payload.pinned_files,
            max_chunks=payload.max_chunks,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    return report.model_dump()


@app.post("/v1/context/vector-search")
async def vector_search(payload: VectorSearchRequest, http_request: Request, response: Response) -> dict[str, Any]:
    try:
        project = require_project_access(http_request, payload.project_id, "context:read")
        production = os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
        config = production_vector_config() if production else VectorBackendConfig(backend=payload.backend, dims=payload.dims)  # type: ignore[arg-type]
        organization_id = request_organization(http_request)
        index = create_vector_index(rag_store(organization_id), config)
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "Wed, 31 Dec 2026 23:59:59 GMT"
        report = await asyncio.to_thread(
            index.search,
            organization_id=organization_id,
            project_id=payload.project_id,
            revision_id=str(project.get("active_index_revision") or ""),
            query=payload.query,
            top_k=payload.top_k,
        )
        return report.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/context/vector-sync")
async def vector_sync(request: VectorSyncRequest, http_request: Request) -> dict[str, Any]:
    try:
        project = require_project_access(http_request, request.project_id, "context:index")
        production = os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
        organization_id = request_organization(http_request)
        index = create_vector_index(
            rag_store(organization_id),
            production_vector_config() if production else VectorBackendConfig(backend=request.backend, dims=request.dims),  # type: ignore[arg-type]
        )
        report = await asyncio.to_thread(
            index.sync_project,
            organization_id=organization_id,
            project_id=request.project_id,
            revision_id=str(project.get("active_index_revision") or ""),
            batch_size=request.batch_size,
        )
        return report.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/context/vector-capabilities")
async def get_vector_capabilities() -> dict[str, Any]:
    return {"capabilities": [item.model_dump() for item in vector_capabilities()]}


@app.post("/v1/memory/ingest")
async def ingest_memory(request: MemoryIngestRequest, http_request: Request, project_id: str | None = None) -> dict[str, Any]:
    try:
        require_scope(http_request, "memory:write")
        if AUTH_CONFIG.enabled and project_id is None:
            raise PermissionError("authenticated memory ingestion requires a tenant-bound project_id")
        if project_id is not None:
            require_project_access(http_request, project_id, "memory:write")
        organization_id = request_organization(http_request)
        report = await asyncio.to_thread(
            UnifiedMemoryManager(project_id=project_id, store=rag_store(organization_id)).ingest,
            request,
        )
        return report.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/memory/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "memory:read")
        if AUTH_CONFIG.enabled and request.project_id is None:
            raise PermissionError("authenticated memory retrieval requires a tenant-bound project_id")
        if request.project_id is not None:
            require_project_access(http_request, request.project_id, "memory:read")
        organization_id = request_organization(http_request)
        report = await asyncio.to_thread(
            UnifiedMemoryManager(project_id=request.project_id, store=rag_store(organization_id)).retrieve_report,
            request.query,
            limit=request.limit,
            layers=request.layers or None,  # type: ignore[arg-type]
        )
        return report.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/tools/execute")
async def execute_tool(request: ToolExecuteRequest, http_request: Request) -> dict[str, Any]:
    try:
        require_scope(http_request, "tools:execute")
        return HardenedToolExecutor(STORE).execute(request).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/sandbox/run")
async def run_sandbox(request: SandboxRunRequest) -> dict[str, Any]:
    return DockerSandboxRunner().run(request).model_dump()


@app.post("/v1/patches/preview")
async def preview_patch(request: PatchPreviewRequest) -> dict[str, Any]:
    try:
        return PatchService(STORE).preview(request).model_dump()
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/patches/verify")
async def verify_patch_loop(request: PatchVerifyRequest) -> dict[str, Any]:
    try:
        return PatchService(STORE).preview_apply_verify(request).model_dump()
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/patches/{patch_id}/apply")
async def apply_patch(patch_id: str) -> dict[str, Any]:
    try:
        return PatchService(STORE).apply(patch_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/patches/{patch_id}/rollback")
async def rollback_patch(patch_id: str) -> dict[str, Any]:
    try:
        return PatchService(STORE).rollback(patch_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/verifier/run")
async def run_verifier(request: VerificationRequest) -> dict[str, Any]:
    try:
        return VerifierRuntime(STORE).run(request).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


@app.post("/v1/agent/run", response_model=CraftlyRunReport)
async def run_agent(request: CraftlyRunRequest) -> CraftlyRunReport:
    return await CraftlyRuntime().run(request)

