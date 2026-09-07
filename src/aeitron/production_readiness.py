"""Production readiness contract for Aeitron.

This module is the single source of truth for honest deployment status. It does
not fake external infrastructure: services that require Redis, Postgres, object
storage, Qdrant, GPU clusters, benchmark files, or scanner CLIs are marked with
explicit dependency states until those dependencies are present and tested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.aeitron.evaluation.benchmark_pack import validate_protected_benchmark_manifest
from src.aeitron.identity.auth import AuthConfig
from src.aeitron.identity.quota import QuotaConfig
from src.aeitron.indexing.vector_index import VectorBackendConfig, validate_embedding_manifest
from src.aeitron.model_ops.backends import active_model_health
from src.aeitron.model_ops.foundation import model_profile
from src.aeitron.shared.config import load_active_profile
from src.aeitron.shared.schemas import StrictModel
from src.aeitron.training_workspace import TrainingProfileRegistry


ReadinessStatus = Literal[
    "production_ready",
    "production_ready_requires_external_service",
    "built_not_cluster_proven",
    "blocked_missing_dependency",
    "not_implemented",
]


class ReadinessCheck(StrictModel):
    subsystem: str
    status: ReadinessStatus
    summary: str
    required_dependencies: list[str] = Field(default_factory=list)
    missing_dependencies: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    production_blocker: bool = False


class ProductionReadinessReport(StrictModel):
    mode: Literal["dev", "production"]
    status: str
    checks: list[ReadinessCheck]
    created_at_unix: float = Field(default_factory=time.time)

    @property
    def blockers(self) -> list[ReadinessCheck]:
        return [check for check in self.checks if check.production_blocker]

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "production_readiness_report.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "production_readiness_report.md")
        return target


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _external_dependency(name: str, *, env: str | None = None, executable: str | None = None, path: str | None = None) -> tuple[bool, str]:
    if env:
        value = os.environ.get(env)
        return bool(value), f"env:{env}"
    if executable:
        return shutil.which(executable) is not None, f"executable:{executable}"
    if path:
        return Path(path).exists(), f"path:{path}"
    return False, name


def _python_tool_present(executable: str, module: str | None = None) -> bool:
    if shutil.which(executable) is not None:
        return True
    scripts_candidate = Path(sys.executable).resolve().parent / "Scripts" / f"{executable}.exe"
    if scripts_candidate.exists():
        return True
    return bool(module and importlib.util.find_spec(module) is not None)


def _codeql_present() -> bool:
    env_path = os.environ.get("AEITRON_CODEQL_BIN", "")
    if env_path and Path(env_path).expanduser().exists():
        return True
    if shutil.which("codeql") is not None:
        return True
    return any(
        candidate.exists()
        for candidate in [
            Path.home() / ".aeitron" / "tools" / "codeql" / "codeql" / "codeql.exe",
            Path.home() / ".aeitron" / "tools" / "codeql" / "codeql" / "codeql",
        ]
    )


def _check_auth(mode: str) -> ReadinessCheck:
    config = AuthConfig.from_env()
    missing = []
    if not config.enabled:
        missing.append("AEITRON_AUTH_ENABLED=1")
    if not config.jwt_secret or len(config.jwt_secret) < 32:
        missing.append("AEITRON_JWT_SECRET length >= 32")
    if config.allow_token_issue:
        missing.append("AEITRON_ALLOW_TOKEN_ISSUE=0")
    return ReadinessCheck(
        subsystem="auth",
        status="production_ready" if not missing else "blocked_missing_dependency",
        summary="JWT auth is enforced for protected routes." if not missing else "JWT auth is not production-enforced.",
        required_dependencies=["AEITRON_AUTH_ENABLED", "AEITRON_JWT_SECRET"],
        missing_dependencies=missing,
        evidence={"enabled": config.enabled, "token_issue_allowed": config.allow_token_issue, "secret_present": bool(config.jwt_secret)},
        production_blocker=mode == "production" and bool(missing),
    )


def _check_quota(mode: str) -> ReadinessCheck:
    config = QuotaConfig.from_env()
    missing = []
    if not config.enabled:
        missing.append("AEITRON_QUOTA_ENABLED=1")
    if not config.redis_url:
        missing.append("AEITRON_REDIS_URL")
    return ReadinessCheck(
        subsystem="quota",
        status="production_ready_requires_external_service" if not missing else "blocked_missing_dependency",
        summary="Redis-backed regenerative quota is configured." if not missing else "Quota is missing Redis production configuration.",
        required_dependencies=["AEITRON_QUOTA_ENABLED", "AEITRON_REDIS_URL"],
        missing_dependencies=missing,
        evidence={"enabled": config.enabled, "redis_url_present": bool(config.redis_url)},
        production_blocker=mode == "production" and bool(missing),
    )


def _check_model_backend(mode: str) -> ReadinessCheck:
    active = active_model_health()
    profile = load_active_profile()
    backend = str(active.get("backend") or "mock")
    missing = []
    if backend == "mock":
        missing.append("non-mock AEITRON_MODEL_BACKEND")
    if backend in {"craftly_serving", "aeitron_serving", "active"} and not active.get("endpoint"):
        missing.append("AEITRON_MODEL_ENDPOINT")
    checkpoint_manifest = os.environ.get("CRAFTLY_CHECKPOINT_MANIFEST") or os.environ.get("AEITRON_CHECKPOINT_MANIFEST", "")
    tokenizer_path = os.environ.get("CRAFTLY_TOKENIZER_PATH") or os.environ.get("AEITRON_TOKENIZER_PATH", "")
    if backend in {"craftly_serving", "aeitron_serving", "active"}:
        if not checkpoint_manifest or not Path(checkpoint_manifest).exists():
            missing.append("AEITRON_CHECKPOINT_MANIFEST existing file")
        if not tokenizer_path or not Path(tokenizer_path).exists():
            missing.append("AEITRON_TOKENIZER_PATH existing file")
    return ReadinessCheck(
        subsystem="serving",
        status="production_ready" if not missing else "blocked_missing_dependency",
        summary="Native Craftly serving backend is selected." if not missing else "Serving is still using mock/test-double configuration.",
        required_dependencies=["AEITRON_MODEL_BACKEND", "AEITRON_MODEL_ENDPOINT", "AEITRON_CHECKPOINT_MANIFEST", "AEITRON_TOKENIZER_PATH"],
        missing_dependencies=missing,
        evidence={
            "backend": backend,
            "model_name": active.get("model_name"),
            "checkpoint_manifest": checkpoint_manifest,
            "tokenizer_path": tokenizer_path,
            "active_profile": profile.get("profile", {}),
        },
        production_blocker=mode == "production" and bool(missing),
    )


def _check_external_services(mode: str) -> list[ReadinessCheck]:
    specs = [
        ("postgres", "AEITRON_DATABASE_URL", "Postgres persistence/migrations"),
        ("object_storage", "AEITRON_OBJECT_STORE_URI", "S3/MinIO dataset/checkpoint artifact storage"),
        ("otel", "AEITRON_OTEL_EXPORTER_OTLP_ENDPOINT", "OpenTelemetry exporter"),
    ]
    checks: list[ReadinessCheck] = []
    for subsystem, env, summary in specs:
        present, dep = _external_dependency(subsystem, env=env)
        checks.append(
            ReadinessCheck(
                subsystem=subsystem,
                status="production_ready_requires_external_service" if present else "blocked_missing_dependency",
                summary=summary if present else f"{summary} is not configured.",
                required_dependencies=[dep],
                missing_dependencies=[] if present else [dep],
                evidence={"env_present": present},
                production_blocker=mode == "production" and not present,
            )
        )
    qdrant_dependencies = {
        "AEITRON_QDRANT_URL": bool(os.environ.get("AEITRON_QDRANT_URL")),
        "AEITRON_EMBEDDING_URL": bool(os.environ.get("AEITRON_EMBEDDING_URL")),
        "AEITRON_EMBEDDING_MANIFEST": bool(os.environ.get("AEITRON_EMBEDDING_MANIFEST")),
    }
    missing_qdrant = [
        f"env:{name}"
        for name, present in qdrant_dependencies.items()
        if not present
    ]
    manifest_evidence: dict[str, Any] = {}
    if not missing_qdrant:
        try:
            manifest = validate_embedding_manifest(
                VectorBackendConfig(
                    backend="qdrant",
                    dims=int(os.environ.get("AEITRON_EMBEDDING_DIMS", "768")),
                    qdrant_url=os.environ.get("AEITRON_QDRANT_URL"),
                    embedding_url=os.environ.get("AEITRON_EMBEDDING_URL"),
                    embedding_model=os.environ.get("AEITRON_EMBEDDING_MODEL", "Aeitron-Code-Embed-v1"),
                    embedding_manifest_path=os.environ.get("AEITRON_EMBEDDING_MANIFEST"),
                    production_mode=True,
                )
            )
            manifest_evidence = {
                "scratch_only": manifest.get("scratch_only"),
                "borrowed_weights": manifest.get("borrowed_weights"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            }
        except (RuntimeError, ValueError) as exc:
            missing_qdrant.append(f"embedding_manifest:{exc}")
    checks.append(
        ReadinessCheck(
            subsystem="qdrant",
            status=(
                "production_ready_requires_external_service"
                if not missing_qdrant
                else "blocked_missing_dependency"
            ),
            summary=(
                "Qdrant and semantic embedding endpoints are configured."
                if not missing_qdrant
                else "Qdrant semantic indexing dependencies are incomplete."
            ),
            required_dependencies=[f"env:{name}" for name in qdrant_dependencies],
            missing_dependencies=missing_qdrant,
            evidence={
                "qdrant_url_present": qdrant_dependencies["AEITRON_QDRANT_URL"],
                "embedding_url_present": qdrant_dependencies["AEITRON_EMBEDDING_URL"],
                "embedding_manifest_present": qdrant_dependencies["AEITRON_EMBEDDING_MANIFEST"],
                "embedding_manifest": manifest_evidence,
            },
            production_blocker=mode == "production" and bool(missing_qdrant),
        )
    )
    return checks


def _check_rag_runtime(mode: str) -> ReadinessCheck:
    required_files = [
        Path("src/aeitron/indexing/repository_indexer.py"),
        Path("src/aeitron/indexing/context_builder.py"),
        Path("src/aeitron/indexing/vector_index.py"),
        Path("src/aeitron/db/migrations/0008_rag_operations.sql"),
        Path("config/rag_embedding_training.json"),
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    proof_path_value = os.environ.get("AEITRON_RAG_PROOF_REPORT", "")
    evaluation_path_value = os.environ.get("AEITRON_RAG_EVALUATION_REPORT", "")
    load_path_value = os.environ.get("AEITRON_RAG_LOAD_REPORT", "")
    proof_evidence: dict[str, Any] = {}
    proof_passed = False
    if proof_path_value:
        try:
            proof_path = Path(proof_path_value).resolve(strict=True)
            proof_evidence = json.loads(proof_path.read_text(encoding="utf-8-sig"))
            checks = {
                str(item.get("name")): str(item.get("status"))
                for item in proof_evidence.get("checks", [])
                if isinstance(item, dict)
            }
            proof_passed = all(
                checks.get(name) == "passed"
                for name in (
                    "postgres_migrations",
                    "redis_quota",
                    "object_store_lifecycle",
                    "qdrant_round_trip",
                    "rag_control_plane_lifecycle",
                )
            )
            if not proof_passed:
                missing.append("passed real RAG dependency lifecycle checks")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            missing.append(f"valid AEITRON_RAG_PROOF_REPORT: {exc}")
    elif mode == "production":
        missing.append("env:AEITRON_RAG_PROOF_REPORT")
    evaluation_passed = False
    if evaluation_path_value:
        try:
            evaluation = json.loads(Path(evaluation_path_value).resolve(strict=True).read_text(encoding="utf-8-sig"))
            evaluation_passed = (
                evaluation.get("status") == "passed"
                and int(evaluation.get("task_count", 0)) >= 500
                and float(evaluation.get("hybrid_gain_percentage_points", 0.0)) >= 5.0
                and int(evaluation.get("stale_revision_results", -1)) == 0
                and int(evaluation.get("cross_tenant_results", -1)) == 0
                and bool(evaluation.get("governance_sha256"))
            )
            if not evaluation_passed:
                missing.append("passed governed 500-task RAG evaluation")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            missing.append(f"valid AEITRON_RAG_EVALUATION_REPORT: {exc}")
    elif mode == "production":
        missing.append("env:AEITRON_RAG_EVALUATION_REPORT")
    load_passed = False
    if load_path_value:
        try:
            load_report = json.loads(Path(load_path_value).resolve(strict=True).read_text(encoding="utf-8-sig"))
            stages = load_report.get("stages", [])
            load_passed = (
                load_report.get("status") == "passed"
                and int(load_report.get("target_chunks", 0)) >= 100_000_000
                and any(int(item.get("concurrency", 0)) >= 1000 for item in stages if isinstance(item, dict))
            )
            if not load_passed:
                missing.append("passed 100M-chunk/1000-concurrency RAG load report")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            missing.append(f"valid AEITRON_RAG_LOAD_REPORT: {exc}")
    elif mode == "production":
        missing.append("env:AEITRON_RAG_LOAD_REPORT")
    status: ReadinessStatus
    if missing:
        status = "blocked_missing_dependency" if mode == "production" else "built_not_cluster_proven"
    elif proof_passed and evaluation_passed and load_passed:
        status = "production_ready_requires_external_service"
    else:
        status = "built_not_cluster_proven"
    return ReadinessCheck(
        subsystem="hybrid_rag_runtime",
        status=status,
        summary=(
            "Durable Postgres dispatch, Redis wake-up, immutable S3 snapshots, Qdrant isolation, "
            "Tree-sitter graph retrieval, and scratch embedding contracts are implemented."
        ),
        required_dependencies=[str(path) for path in required_files]
        + ["AEITRON_RAG_PROOF_REPORT", "AEITRON_RAG_EVALUATION_REPORT", "AEITRON_RAG_LOAD_REPORT"],
        missing_dependencies=missing,
        evidence={
            "real_dependency_proof": proof_passed,
            "proof_report": proof_path_value,
            "governed_evaluation_passed": evaluation_passed,
            "scale_load_passed": load_passed,
            "proof_authority": proof_evidence.get("authority"),
            "governed_500_task_eval_required": True,
            "100m_chunk_load_proof_required": True,
        },
        production_blocker=mode == "production" and bool(missing),
    )


def _check_cli_tools(mode: str) -> list[ReadinessCheck]:
    tools = [
        ("semgrep", "semgrep", "semgrep", "Semgrep static security scan"),
        ("codeql", "codeql", None, "CodeQL semantic security scan"),
        ("bandit", "bandit", "bandit", "Bandit Python security scan"),
        ("pip_audit", "pip-audit", "pip_audit", "Dependency vulnerability scan"),
        ("docker", "docker", None, "Docker sandbox runtime"),
        ("kubectl", "kubectl", None, "Kubernetes server-side deployment validation"),
    ]
    checks: list[ReadinessCheck] = []
    for subsystem, executable, module, summary in tools:
        present = _codeql_present() if subsystem == "codeql" else (_python_tool_present(executable, module) if module else shutil.which(executable) is not None)
        dep = f"executable:{executable}" if module is None else f"executable:{executable} or python module:{module}"
        checks.append(
            ReadinessCheck(
                subsystem=subsystem,
                status="production_ready_requires_external_service" if present else "blocked_missing_dependency",
                summary=summary if present else f"{summary} executable is missing.",
                required_dependencies=[dep],
                missing_dependencies=[] if present else [dep],
                evidence={"executable_present": present},
                production_blocker=mode == "production" and not present,
            )
        )
    return checks


def _check_training_stack(mode: str) -> list[ReadinessCheck]:
    cuda_available = False
    torch_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    deepspeed_available = importlib.util.find_spec("deepspeed") is not None
    vllm_available = importlib.util.find_spec("vllm") is not None
    megatron_root = os.environ.get("MEGATRON_LM_ROOT", "")
    megatron_available = bool(megatron_root and Path(megatron_root).exists())
    hf_export_dir = os.environ.get("AEITRON_HF_EXPORT_DIR", "")
    hf_export_ready = bool(
        hf_export_dir
        and all((Path(hf_export_dir) / name).exists() for name in ["config.json", "tokenizer.json"])
        and ((Path(hf_export_dir) / "model.safetensors").exists() or (Path(hf_export_dir) / "pytorch_model.bin").exists())
    )
    trt_build = shutil.which("trtllm-build") is not None
    vllm_trt_missing = []
    if not hf_export_ready:
        vllm_trt_missing.append("AEITRON_HF_EXPORT_DIR with config.json/model.safetensors/tokenizer.json")
    if not vllm_available:
        vllm_trt_missing.append("python module: vllm")
    if not trt_build:
        vllm_trt_missing.append("executable:trtllm-build")
    deepspeed_missing = []
    if not deepspeed_available:
        deepspeed_missing.append("python module: deepspeed")
    if not megatron_available:
        deepspeed_missing.append("MEGATRON_LM_ROOT existing checkout")
    four_t_profile = model_profile("4t_moe")
    four_t_parameters = four_t_profile.parameter_report()
    return [
        ReadinessCheck(
            subsystem="canonical_4t_architecture",
            status="built_not_cluster_proven",
            summary=(
                "The canonical 4T-total/128B-active MLA-MoE contract passes deterministic "
                "parameter accounting; no cluster training, checkpoint, context, or serving proof exists yet."
            ),
            required_dependencies=[
                "governed evidence-selected tokenizer",
                "promoted immutable dataset",
                "Megatron-Core TP/PP/DP/CP/EP cluster",
                "distributed checkpoint reload proof",
                "1M native-context benchmark proof",
                "converted-serving logit parity",
            ],
            missing_dependencies=[
                "real 4T distributed training/checkpoint proof",
                "real 1M native-context evaluation",
                "real MLA/MoE serving conversion parity",
            ],
            evidence={
                "profile": four_t_profile.name,
                "architecture_version": four_t_profile.architecture_version,
                "contract_sha256": four_t_profile.contract_sha256(),
                "parameter_report": four_t_parameters,
                "native_context_tokens": four_t_profile.max_sequence_length,
                "effective_context_tokens": four_t_profile.effective_context_length,
                "runtime_backend": four_t_profile.runtime_backend,
                "cluster_proven": False,
            },
            production_blocker=mode == "production",
        ),
        ReadinessCheck(
            subsystem="pretraining",
            status="built_not_cluster_proven" if cuda_available else "blocked_missing_dependency",
            summary=(
                "A CUDA runtime is available for technical validation, but serious distributed pretraining is not proven."
                if cuda_available
                else "CUDA training runtime is not available on this host."
            ),
            required_dependencies=["CUDA-capable GPU", "PyTorch CUDA build", "measured distributed training proof"],
            missing_dependencies=(
                ["measured distributed training proof"]
                if cuda_available
                else ["CUDA-capable GPU or compatible PyTorch CUDA build", "measured distributed training proof"]
            ),
            evidence={"torch_version": torch_version, "cuda_available": cuda_available, "cluster_proven": False},
            production_blocker=mode == "production",
        ),
        ReadinessCheck(
            subsystem="fsdp",
            status="built_not_cluster_proven",
            summary="Native PyTorch FSDP runtime is wired but must pass a real multi-GPU cluster gate.",
            required_dependencies=["torchrun", "multi-GPU cluster", "NCCL"],
            missing_dependencies=[],
            evidence={"runtime_path": "src.aeitron.model_ops.pretrain_loop"},
            production_blocker=False,
        ),
        ReadinessCheck(
            subsystem="deepspeed_megatron",
            status="built_not_cluster_proven" if not deepspeed_missing else "blocked_missing_dependency",
            summary=(
                "DeepSpeed and Megatron launch contracts are wired; both still require measured cluster gates."
                if not deepspeed_missing
                else "DeepSpeed/Megatron dependencies are missing."
            ),
            required_dependencies=["deepspeed", "Megatron-LM checkout", "cluster release gate"],
            missing_dependencies=deepspeed_missing,
            evidence={"deepspeed_module": deepspeed_available, "megatron_root": megatron_root},
            production_blocker=mode == "production",
        ),
        ReadinessCheck(
            subsystem="vllm_tensorrt",
            status="built_not_cluster_proven" if not vllm_trt_missing else "blocked_missing_dependency",
            summary=(
                "Dense HF export dependencies exist; runtime decode parity is still required and MLA/MoE conversion is not implemented."
                if hf_export_ready
                else "HF/vLLM/TensorRT export artifacts or runtime dependencies are missing."
            ),
            required_dependencies=[
                "dense Aeitron-to-HF converter",
                "custom MLA/MoE vLLM or TensorRT-LLM conversion",
                "fixed-prompt logit parity proof",
            ],
            missing_dependencies=[
                *vllm_trt_missing,
                "custom MLA/MoE serving conversion and fixed-prompt parity proof",
            ],
            evidence={
                "hf_export_dir": hf_export_dir,
                "hf_export_ready": hf_export_ready,
                "vllm_module": vllm_available,
                "trtllm_build": trt_build,
                "mla_moe_conversion_implemented": False,
                "decode_parity_proven": False,
            },
            production_blocker=mode == "production",
        ),
    ]


def _check_training_workspace(mode: str) -> ReadinessCheck:
    database_url = os.environ.get("AEITRON_DATABASE_URL", "")
    redis_url = os.environ.get("AEITRON_REDIS_URL", "")
    object_uri = os.environ.get("AEITRON_OBJECT_STORE_URI", "")
    jwt_secret = os.environ.get("AEITRON_JWT_SECRET", "")
    missing = []
    if not database_url.startswith(("postgres://", "postgresql://")):
        missing.append("AEITRON_DATABASE_URL=postgresql://...")
    if not redis_url.startswith(("redis://", "rediss://")):
        missing.append("AEITRON_REDIS_URL=redis[s]://...")
    if not object_uri.startswith("s3://"):
        missing.append("AEITRON_OBJECT_STORE_URI=s3://...")
    if len(jwt_secret) < 32:
        missing.append("AEITRON_JWT_SECRET length >= 32")
    profile_count = 0
    try:
        profile_count = len(TrainingProfileRegistry.from_file().profiles)
    except (FileNotFoundError, ValueError):
        missing.append("valid config/training_profiles.json")
    kubernetes_client = importlib.util.find_spec("kubernetes") is not None or shutil.which("kubectl") is not None
    if not kubernetes_client:
        missing.append("kubernetes Python client or kubectl")
    return ReadinessCheck(
        subsystem="training_workspace",
        status="built_not_cluster_proven" if not missing else "blocked_missing_dependency",
        summary=(
            "Durable training control-plane dependencies are configured; scheduler paths still require live cluster proof."
            if not missing
            else "Training workspace production dependencies are incomplete."
        ),
        required_dependencies=["Postgres", "Redis Streams", "S3/MinIO", "JWT signing key", "Kubernetes scheduler client"],
        missing_dependencies=missing,
        evidence={
            "profile_count": profile_count,
            "postgres_configured": database_url.startswith(("postgres://", "postgresql://")),
            "redis_configured": redis_url.startswith(("redis://", "rediss://")),
            "object_storage_configured": object_uri.startswith("s3://"),
            "scheduler_client_present": kubernetes_client,
            "cluster_proof": False,
        },
        production_blocker=mode == "production" and bool(missing),
    )


def _check_agent_collaboration(mode: str) -> ReadinessCheck:
    migration = Path("src/aeitron/db/migrations/0005_agent_collaboration.sql")
    database_url = os.environ.get("AEITRON_DATABASE_URL", "")
    postgres_configured = database_url.startswith(("postgres://", "postgresql://"))
    proof_path = Path(
        os.environ.get(
            "AEITRON_AGENT_COLLABORATION_PROOF_REPORT",
            "artifacts/aeitron/agent-collaboration-proof/agent_collaboration_postgres_proof.json",
        )
    )
    live_proven = False
    if proof_path.is_file():
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            live_proven = (
                proof.get("status") == "passed"
                and proof.get("migration") == "0005_agent_collaboration"
                and proof.get("atomic_claim_winner_count") == 1
                and proof.get("blackboard_stale_update_rejected") is True
                and proof.get("durable_message_round_trip") is True
            )
        except (OSError, json.JSONDecodeError):
            live_proven = False
    missing = []
    if not migration.exists():
        missing.append(str(migration))
    if mode == "production" and not postgres_configured:
        missing.append("AEITRON_DATABASE_URL=postgresql://...")
    if mode == "production" and not live_proven:
        missing.append(f"valid live Postgres lifecycle proof: {proof_path}")
    status: ReadinessStatus
    if missing:
        status = "blocked_missing_dependency"
    elif live_proven:
        status = "production_ready_requires_external_service"
    else:
        status = "built_not_cluster_proven"
    return ReadinessCheck(
        subsystem="agent_collaboration",
        status=status,
        summary=(
            "Concurrent TaskGraph leases, typed messages, blackboard CAS, bounded reflection, and failure intelligence are wired."
            if not missing
            else "Agent collaboration persistence is missing a production dependency."
        ),
        required_dependencies=[
            "migration 0005_agent_collaboration",
            "Postgres in production",
            "live concurrent lifecycle proof",
        ],
        missing_dependencies=missing,
        evidence={
            "migration_present": migration.exists(),
            "postgres_configured": postgres_configured,
            "live_postgres_lifecycle_proven": live_proven,
            "proof_report": str(proof_path),
            "max_reflection_revisions": 3,
            "typed_message_kinds": ["proposal", "evidence", "challenge", "review", "decision"],
        },
        production_blocker=mode == "production" and bool(missing),
    )


def _check_agent_execution(mode: str) -> ReadinessCheck:
    engine = Path("src/aeitron/runtime/execution.py")
    scorecard_runner = Path("src/aeitron/evaluation/agent_scorecard.py")
    qualification_runner = Path("src/aeitron/evaluation/qualification_campaign.py")
    report_path = Path(
        os.environ.get(
            "AEITRON_AGENT_SCORECARD_REPORT",
            "artifacts/aeitron/agent-scorecard/agent_scorecard.json",
        )
    )
    pack_path = Path(
        os.environ.get(
            "AEITRON_QUALIFICATION_PACK_MANIFEST",
            "artifacts/aeitron/qualification-pack/qualification_pack_manifest.json",
        )
    )
    pack_proven = False
    pack_evidence: dict[str, Any] = {}
    pack_task_ids: set[str] = set()
    if pack_path.is_file():
        try:
            pack_evidence = json.loads(pack_path.read_text(encoding="utf-8-sig"))
            pack_task_ids = {str(item) for item in pack_evidence.get("task_ids") or []}
            category_counts = dict(pack_evidence.get("category_counts") or {})
            pack_proven = (
                int(pack_evidence.get("task_count") or 0) == 50
                and len(pack_task_ids) == 50
                and all(int(category_counts.get(name) or 0) == 10 for name in [
                    "coding",
                    "debugging",
                    "defensive_security",
                    "patch_generation",
                    "long_context",
                ])
                and bool(pack_evidence.get("evaluation_only"))
                and bool(pack_evidence.get("benchmark_holdout"))
                and not bool(pack_evidence.get("ground_truth_in_prompts"))
                and len(str(pack_evidence.get("source_commit") or "")) >= 40
                and len(str(pack_evidence.get("legal_approval_sha256") or "")) == 64
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pack_proven = False
    scorecard_proven = False
    scorecard_evidence: dict[str, Any] = {}
    if report_path.is_file():
        try:
            scorecard_evidence = json.loads(report_path.read_text(encoding="utf-8"))
            scorecard_task_ids = {
                str(item.get("task_id") or "")
                for item in scorecard_evidence.get("tasks") or []
                if isinstance(item, dict)
            }
            scorecard_proven = (
                scorecard_evidence.get("status") == "passed"
                and scorecard_evidence.get("policy_mode") == "strict"
                and 50 <= int(scorecard_evidence.get("task_count") or 0) <= 100
                and float(scorecard_evidence.get("architecture_reliability_score") or 0.0) >= 0.95
                and float(scorecard_evidence.get("workflow_completion_score") or 0.0) >= 0.80
                and float(scorecard_evidence.get("sandbox_test_pass_rate") or 0.0) >= 0.80
                and int(scorecard_evidence.get("regression_count") or 0) == 0
                and pack_proven
                and scorecard_task_ids == pack_task_ids
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            scorecard_proven = False
    missing = [str(path) for path in [engine, scorecard_runner, qualification_runner] if not path.is_file()]
    if mode == "production" and not pack_proven:
        missing.append(f"governed immutable 50-task qualification pack: {pack_path}")
    if mode == "production" and not scorecard_proven:
        missing.append(f"strict 50-100 repository task scorecard proof: {report_path}")
    if missing:
        status: ReadinessStatus = "blocked_missing_dependency"
    elif scorecard_proven:
        status = "production_ready"
    else:
        status = "built_not_cluster_proven"
    return ReadinessCheck(
        subsystem="verified_agent_execution",
        status=status,
        summary=(
            "Architect, coder, sandbox tester, defensive reviewers, critic, verifier, bounded revision, and patch transaction are wired."
            if not missing
            else "Verified agent execution is missing code or a required real-repository proof."
        ),
        required_dependencies=[
            "non-mock Aeitron scratch serving backend",
            "Docker sandbox",
            "Semgrep or CodeQL",
            "governed pinned 50-task qualification pack",
            "strict 50-100 repository task scorecard",
        ],
        missing_dependencies=missing,
        evidence={
            "execution_engine_present": engine.is_file(),
            "scorecard_runner_present": scorecard_runner.is_file(),
            "qualification_runner_present": qualification_runner.is_file(),
            "qualification_pack_manifest": str(pack_path),
            "qualification_pack_proven": pack_proven,
            "qualification_pack_id": pack_evidence.get("pack_id"),
            "scorecard_report": str(report_path),
            "scorecard_proven": scorecard_proven,
            "scorecard_summary": {
                key: scorecard_evidence.get(key)
                for key in [
                    "status",
                    "policy_mode",
                    "task_count",
                    "architecture_reliability_score",
                    "workflow_completion_score",
                    "sandbox_test_pass_rate",
                    "regression_count",
                ]
            },
            "max_patch_revisions": 3,
            "original_mutation_before_accept": False,
        },
        production_blocker=mode == "production" and bool(missing),
    )


def _check_benchmark_files(mode: str, benchmark_dir: str | Path) -> ReadinessCheck:
    root = Path(benchmark_dir)
    repository_root = Path(__file__).resolve().parents[2]
    protected_manifest = root / "protected" / "protected_benchmark_manifest.json"
    protected_config = repository_root / "config" / "protected_benchmarks.json"
    executable_report_value = os.environ.get("AEITRON_EXECUTABLE_BENCHMARK_REPORT", "")
    executable_report = Path(executable_report_value).expanduser() if executable_report_value else None
    missing: list[str] = []
    evidence: dict[str, Any] = {
        "benchmark_dir": str(root),
        "protected_manifest": str(protected_manifest),
        "executable_report": str(executable_report or ""),
    }
    try:
        manifest = validate_protected_benchmark_manifest(protected_config, protected_manifest)
        evidence["protected_pack_id"] = manifest.pack_id
        evidence["protected_artifact_count"] = len(manifest.artifacts)
    except (OSError, ValueError) as exc:
        missing.append(f"valid protected benchmark manifest: {exc}")

    executable_proven = False
    if executable_report is not None and executable_report.is_file():
        try:
            payload = json.loads(executable_report.read_text(encoding="utf-8-sig"))
            suites = payload.get("suites") if isinstance(payload, dict) else None
            executable_proven = (
                payload.get("status") == "passed"
                and payload.get("evaluation_mode") == "executable_model"
                and isinstance(suites, list)
                and bool(suites)
                and all(
                    isinstance(item, dict)
                    and item.get("status") == "passed"
                    and isinstance(item.get("pass_at_k"), dict)
                    and "pass@1" in item["pass_at_k"]
                    for item in suites
                )
            )
        except (OSError, json.JSONDecodeError):
            executable_proven = False
    evidence["executable_model_proven"] = executable_proven
    if mode == "production" and not executable_proven:
        missing.append("passed AEITRON_EXECUTABLE_BENCHMARK_REPORT with measured pass@1")

    if missing:
        status: ReadinessStatus = "blocked_missing_dependency"
        summary = "Protected benchmark or executable-model evidence is missing or invalid."
    elif executable_proven:
        status = "production_ready"
        summary = "Protected benchmark bindings and executable-model evaluation are verified."
    else:
        status = "built_not_cluster_proven"
        summary = "Protected benchmark bindings are verified; executable-model proof is pending."
    return ReadinessCheck(
        subsystem="benchmark_eval",
        status=status,
        summary=summary,
        required_dependencies=[
            str(protected_config),
            str(protected_manifest),
            "AEITRON_EXECUTABLE_BENCHMARK_REPORT",
        ],
        missing_dependencies=missing,
        evidence=evidence,
        production_blocker=mode == "production" and bool(missing),
    )


def run_production_readiness(
    *,
    mode: Literal["dev", "production"] = "dev",
    benchmark_dir: str | Path = "data/eval",
) -> ProductionReadinessReport:
    checks = [
        _check_auth(mode),
        _check_quota(mode),
        _check_model_backend(mode),
        *_check_external_services(mode),
        _check_rag_runtime(mode),
        *_check_cli_tools(mode),
        *_check_training_stack(mode),
        _check_training_workspace(mode),
        _check_agent_collaboration(mode),
        _check_agent_execution(mode),
        _check_benchmark_files(mode, benchmark_dir),
    ]
    failed = any(check.production_blocker for check in checks)
    return ProductionReadinessReport(mode=mode, status="failed" if failed else "passed", checks=checks)


def write_markdown(report: ProductionReadinessReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Aeitron Production Readiness Report",
        "",
        f"- mode: {report.mode}",
        f"- status: {report.status}",
        f"- blockers: {len(report.blockers)}",
        "",
        "| subsystem | status | blocker | missing | summary |",
        "|---|---|---:|---|---|",
    ]
    for check in report.checks:
        missing = ", ".join(check.missing_dependencies)
        lines.append(f"| {check.subsystem} | {check.status} | {check.production_blocker} | {missing} | {check.summary} |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aeitron production readiness gate.")
    parser.add_argument("--mode", choices=["dev", "production"], default="dev")
    parser.add_argument("--benchmark-dir", default="data/eval")
    parser.add_argument("--output-dir", default="artifacts/aeitron/production-readiness")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_production_readiness(mode=args.mode, benchmark_dir=args.benchmark_dir)
    report.write(args.output_dir)
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

