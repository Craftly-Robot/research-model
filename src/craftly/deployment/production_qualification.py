"""Authoritative production qualification control plane for Craftly.

This module turns measured evidence into one tamper-evident release decision.
It does not replace live probes, training proofs, scanners, or canary systems;
it validates and binds their outputs so stale or partial reports cannot be
mistaken for current production evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import platform
import re
import subprocess  # nosec B404 - every subprocess argv is fixed internally
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, field_validator, model_validator

from src.craftly.deployment.production_proof import (
    NativeServingLoadReport,
    ProductionProofConfig,
    ProductionProofReport,
    _validated_service_url,
    run_native_serving_load_test,
    run_production_proof,
)
from src.craftly.learning.ablation_runner import verify_promotion_chain
from src.craftly.learning.calibration_gate import (
    CalibrationDecision,
    CalibrationPreflightReport,
)
from src.craftly.learning.production_dataset import (
    ProductionDatasetManifest,
    validate_dataset_manifest_for_promotion,
)
from src.craftly.model_ops.tokenizer_pipeline import TokenizerAuditReport
from src.craftly.shared.integrity import canonical_json_bytes, sha256_file
from src.craftly.shared.schemas import StrictModel

QualificationStatus = Literal["passed", "failed", "blocked", "not_run"]
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LoadStagePolicy(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    concurrency: int = Field(ge=1, le=10_000)
    requests: int = Field(ge=1, le=100_000)
    streaming_requests: int = Field(ge=0, le=10_000)
    maximum_error_rate: float = Field(ge=0.0, le=1.0)
    maximum_p95_latency_ms: float = Field(gt=0.0, le=3_600_000.0)
    minimum_throughput_rps: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_stage(self) -> "LoadStagePolicy":
        if self.streaming_requests > self.requests:
            raise ValueError("streaming_requests cannot exceed requests")
        if self.requests < self.concurrency:
            raise ValueError("requests must be at least concurrency")
        return self


class QualificationPolicy(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    evidence_max_age_seconds: int = Field(ge=60, le=31_536_000)
    minimum_security_reviewers: int = Field(ge=2, le=20)
    require_report_signature_in_production: bool = True
    require_operator_notification_proof: bool = True
    load_stages: list[LoadStagePolicy] = Field(min_length=1)
    required_training_proofs: list[str] = Field(min_length=1)
    required_soak_seconds: list[int] = Field(min_length=2)
    required_security_domains: list[str] = Field(min_length=1)
    required_canary_percentages: list[int] = Field(min_length=1)
    minimum_internal_canary_users: int = Field(ge=1)
    maximum_internal_canary_users: int = Field(ge=1)
    canary_maximum_error_rate: float = Field(ge=0.0, le=1.0)
    canary_maximum_p95_latency_ms: float = Field(gt=0.0)
    canary_rollback_error_rate: float = Field(gt=0.0, le=1.0)
    canary_maximum_rollback_seconds: float = Field(gt=0.0, le=3600.0)
    minimum_promoted_dataset_records: int = Field(default=100_000, ge=100_000)
    tokenizer_selection_mode: Literal["fixed", "evidence_selected"] = "fixed"
    required_tokenizer_vocab_size: int | None = Field(default=128_000, ge=1_000)
    require_family_safe_tokenizer_split: bool = True
    t4_minimum_hidden_size: int = Field(default=512, ge=128)
    t4_minimum_layers: int = Field(default=8, ge=2)
    t4_minimum_sequence_length: int = Field(default=256, ge=64)
    maximum_checkpoint_reload_logit_difference: float = Field(default=0.005, ge=0.0)

    @model_validator(mode="after")
    def validate_policy(self) -> "QualificationPolicy":
        if self.minimum_internal_canary_users > self.maximum_internal_canary_users:
            raise ValueError("minimum_internal_canary_users exceeds maximum")
        if sorted(self.required_canary_percentages) != self.required_canary_percentages:
            raise ValueError("required_canary_percentages must be ordered")
        if len(set(self.required_canary_percentages)) != len(
            self.required_canary_percentages
        ):
            raise ValueError("required_canary_percentages contains duplicates")
        if self.required_canary_percentages[-1] != 100:
            raise ValueError("canary ladder must end at 100 percent")
        if self.canary_rollback_error_rate <= self.canary_maximum_error_rate:
            raise ValueError(
                "rollback error threshold must exceed normal canary error threshold"
            )
        if len(set(stage.concurrency for stage in self.load_stages)) != len(
            self.load_stages
        ):
            raise ValueError("load stage concurrency values must be unique")
        if (
            self.tokenizer_selection_mode == "fixed"
            and self.required_tokenizer_vocab_size is None
        ):
            raise ValueError(
                "fixed tokenizer selection requires required_tokenizer_vocab_size"
            )
        if (
            self.tokenizer_selection_mode == "evidence_selected"
            and self.required_tokenizer_vocab_size is not None
        ):
            raise ValueError(
                "evidence-selected tokenizer policy cannot hard-code a vocabulary size"
            )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "QualificationPolicy":
        source = Path(path).expanduser().resolve(strict=True)
        if source.stat().st_size > 1_048_576:
            raise ValueError("qualification policy exceeds 1 MiB")
        return cls.model_validate_json(source.read_text(encoding="utf-8-sig"))


class EvidenceBinding(StrictModel):
    evidence_id: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    modified_at_unix: float
    age_seconds: float = Field(ge=0.0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256 digest")
        return value


class QualificationCheck(StrictModel):
    subsystem: str
    status: QualificationStatus
    summary: str
    evidence: list[EvidenceBinding] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)


class ScratchAdvancementReport(StrictModel):
    schema_version: Literal[1] = 1
    authority: Literal["scratch_training_advancement"] = "scratch_training_advancement"
    report_id: str
    created_at: str
    git_commit: str
    check: QualificationCheck
    report_sha256: str = ""

    @field_validator("report_sha256")
    @classmethod
    def validate_report_sha256(cls, value: str) -> str:
        if value and HEX_SHA256.fullmatch(value) is None:
            raise ValueError("scratch advancement report digest must be SHA-256")
        return value


class LoadStageResult(StrictModel):
    name: str
    status: QualificationStatus
    concurrency: int
    requests: int
    error_rate: float = Field(ge=0.0, le=1.0)
    throughput_rps: float = Field(ge=0.0)
    latency_ms_p50: float = Field(ge=0.0)
    latency_ms_p95: float = Field(ge=0.0)
    latency_ms_p99: float = Field(ge=0.0)
    maximum_latency_ms: float = Field(ge=0.0)
    streaming_passed: int = Field(ge=0)
    blockers: list[str] = Field(default_factory=list)


class ProductionQualificationReport(StrictModel):
    schema_version: int = 1
    authority: Literal["production_release_decision"] = "production_release_decision"
    report_id: str
    status: QualificationStatus
    mode: Literal["validation", "production"]
    created_at: str
    git_commit: str
    policy_sha256: str
    previous_report_sha256: str | None = None
    environment: dict[str, str]
    checks: list[QualificationCheck]
    load_stages: list[LoadStageResult] = Field(default_factory=list)
    report_sha256: str = ""
    signature_algorithm: str | None = None
    signature: str | None = None

    @field_validator("policy_sha256", "report_sha256")
    @classmethod
    def validate_optional_digest(cls, value: str) -> str:
        if value and HEX_SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256 digest")
        return value


class SecurityReviewEvidence(StrictModel):
    schema_version: int = 1
    review_id: str = Field(min_length=1, max_length=256)
    reviewed_at: str
    reviewers: list[str] = Field(min_length=2)
    domains: dict[str, Literal["passed", "failed"]]
    critical_findings: int = Field(ge=0)
    high_findings: int = Field(ge=0)
    unresolved_findings: list[str] = Field(default_factory=list)
    scanner_report_sha256: str
    decision: Literal["approved", "rejected"]

    @field_validator("scanner_report_sha256")
    @classmethod
    def validate_scanner_digest(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("scanner_report_sha256 must be a SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_independence(self) -> "SecurityReviewEvidence":
        normalized = {reviewer.strip().lower() for reviewer in self.reviewers}
        if len(normalized) != len(self.reviewers):
            raise ValueError("security reviewers must be distinct")
        return self


class CanaryStageEvidence(StrictModel):
    percentage: int = Field(ge=1, le=100)
    requests: int = Field(ge=1)
    error_rate: float = Field(ge=0.0, le=1.0)
    p95_latency_ms: float = Field(ge=0.0)
    rollback_trigger_tested: bool
    rollback_trigger_error_rate: float = Field(ge=0.0, le=1.0)
    rollback_succeeded: bool
    rollback_completed_seconds: float = Field(ge=0.0)


class CanaryEvidence(StrictModel):
    schema_version: int = 1
    canary_id: str = Field(min_length=1, max_length=256)
    created_at: str
    internal_user_count: int = Field(ge=1)
    stages: list[CanaryStageEvidence] = Field(min_length=1)
    status: Literal["passed", "failed"]


class OperatorNotificationEvidence(StrictModel):
    schema_version: int = 1
    proof_id: str = Field(min_length=1, max_length=256)
    created_at: str
    provider: str = Field(min_length=1, max_length=128)
    channel_type: Literal["email", "pager", "slack", "teams", "webhook"]
    recipient_reference_sha256: str
    firing_delivery_id: str = Field(min_length=1, max_length=512)
    recovery_delivery_id: str = Field(min_length=1, max_length=512)
    firing_delivered_at: str
    recovery_delivered_at: str
    firing_delivered: bool
    recovery_delivered: bool
    status: Literal["passed", "failed"]

    @field_validator("recipient_reference_sha256")
    @classmethod
    def validate_recipient_digest(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("recipient_reference_sha256 must be a SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_deliveries(self) -> "OperatorNotificationEvidence":
        if hmac.compare_digest(self.firing_delivery_id, self.recovery_delivery_id):
            raise ValueError("firing and recovery delivery IDs must be distinct")
        return self


class ImmutableQualificationStore:
    def __init__(self, root: str | Path, *, signing_key: str | None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.signing_key = signing_key

    def write(self, report: ProductionQualificationReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        previous_digest = self._latest_digest()
        unsigned = report.model_copy(
            update={
                "previous_report_sha256": previous_digest,
                "report_sha256": "",
                "signature_algorithm": None,
                "signature": None,
            }
        )
        digest = hashlib.sha256(canonical_json_bytes(unsigned.model_dump())).hexdigest()
        signature = None
        algorithm = None
        if self.signing_key:
            signature = hmac.new(
                self.signing_key.encode("utf-8"),
                digest.encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            algorithm = "HMAC-SHA256"
        final = unsigned.model_copy(
            update={
                "report_sha256": digest,
                "signature_algorithm": algorithm,
                "signature": signature,
            }
        )
        run_dir = self.root / "runs" / final.report_id
        run_dir.mkdir(parents=True, exist_ok=False)
        report_path = run_dir / "production_qualification_report.json"
        self._write_exclusive(
            report_path, canonical_json_bytes(final.model_dump()) + b"\n"
        )
        self._write_exclusive(
            report_path.with_suffix(".json.sha256"),
            f"{sha256_file(report_path)}  {report_path.name}\n".encode("ascii"),
        )
        pointer = {
            "schema_version": 1,
            "report_id": final.report_id,
            "report_path": str(report_path),
            "file_sha256": sha256_file(report_path),
            "report_sha256": final.report_sha256,
            "updated_at": utc_now(),
        }
        self._write_atomic(
            self.root / "latest.json", canonical_json_bytes(pointer) + b"\n"
        )
        return report_path

    def _latest_digest(self) -> str | None:
        pointer_path = self.root / "latest.json"
        if not pointer_path.exists():
            return None
        payload = json.loads(pointer_path.read_text(encoding="utf-8-sig"))
        report_path = (
            Path(str(payload["report_path"])).expanduser().resolve(strict=True)
        )
        actual = sha256_file(report_path)
        if not hmac.compare_digest(actual, str(payload["file_sha256"])):
            raise ValueError("latest qualification report file hash mismatch")
        report = verify_qualification_report(report_path, signing_key=self.signing_key)
        if not report.report_sha256:
            raise ValueError("latest qualification report has no report digest")
        return report.report_sha256

    @staticmethod
    def _write_exclusive(path: Path, content: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_atomic(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def verify_qualification_report(
    path: str | Path,
    *,
    signing_key: str | None,
) -> ProductionQualificationReport:
    source = Path(path).expanduser().resolve(strict=True)
    report = ProductionQualificationReport.model_validate_json(
        source.read_text(encoding="utf-8-sig")
    )
    unsigned = report.model_copy(
        update={
            "report_sha256": "",
            "signature_algorithm": None,
            "signature": None,
        }
    )
    expected_digest = hashlib.sha256(
        canonical_json_bytes(unsigned.model_dump())
    ).hexdigest()
    if not hmac.compare_digest(expected_digest, report.report_sha256):
        raise ValueError("qualification report digest verification failed")
    if report.signature_algorithm or report.signature:
        if report.signature_algorithm != "HMAC-SHA256" or not report.signature:
            raise ValueError("qualification report signature metadata is incomplete")
        if not signing_key:
            raise ValueError(
                "qualification report signature cannot be verified without signing key"
            )
        expected_signature = hmac.new(
            signing_key.encode("utf-8"),
            report.report_sha256.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, report.signature):
            raise ValueError("qualification report signature verification failed")
    return report


def bind_evidence(
    path: str | Path,
    *,
    evidence_id: str,
    maximum_age_seconds: int,
) -> EvidenceBinding:
    candidate = Path(path).expanduser().absolute()
    current = candidate
    while True:
        if current.exists() and current.is_symlink():
            raise ValueError(f"{evidence_id} path cannot contain symlinks")
        if current.parent == current:
            break
        current = current.parent
    source = candidate.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{evidence_id} must be a regular non-symlink file")
    stat = source.stat()
    if stat.st_size > 128 * 1024 * 1024:
        raise ValueError(f"{evidence_id} exceeds the 128 MiB evidence limit")
    age = max(0.0, time.time() - stat.st_mtime)
    if age > maximum_age_seconds:
        raise ValueError(
            f"{evidence_id} is stale: age={age:.0f}s, maximum={maximum_age_seconds}s"
        )
    return EvidenceBinding(
        evidence_id=evidence_id,
        path=str(source),
        sha256=sha256_file(source),
        size_bytes=stat.st_size,
        modified_at_unix=stat.st_mtime,
        age_seconds=round(age, 3),
    )


def load_json_evidence(binding: EvidenceBinding) -> Any:
    source = Path(binding.path)
    if not hmac.compare_digest(sha256_file(source), binding.sha256):
        raise ValueError(f"{binding.evidence_id} changed after it was bound")
    return json.loads(source.read_text(encoding="utf-8-sig"))


def require_fresh_timestamp(
    value: str,
    *,
    label: str,
    maximum_age_seconds: int,
) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp must include a timezone")
    age = (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()
    if age < -300:
        raise ValueError(f"{label} timestamp is more than five minutes in the future")
    if age > maximum_age_seconds:
        raise ValueError(
            f"{label} timestamp is stale: age={age:.0f}s, maximum={maximum_age_seconds}s"
        )
    return parsed


def _git_commit() -> str:
    completed = subprocess.run(  # nosec B603 - fixed git argv, no shell
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = completed.stdout.strip().lower()
    return (
        value
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value)
        else "unavailable"
    )


def run_fixed_functional_gates(
    output_dir: Path, *, timeout_seconds: int
) -> QualificationCheck:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src/craftly",
            "tests",
            "deploy/gpu",
        ],
        [sys.executable, "-m", "unittest"],
        [sys.executable, "-m", "src.craftly.evaluation.release_gate", "--skip-tests"],
    ]
    evidence: list[EvidenceBinding] = []
    failures: list[str] = []
    command_reports: list[dict[str, Any]] = []
    for index, argv in enumerate(commands, start=1):
        started = time.perf_counter()
        try:
            completed = subprocess.run(  # nosec B603 - fixed internal argv, no shell
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout_path = output_dir / f"gate-{index}.stdout.log"
            stderr_path = output_dir / f"gate-{index}.stderr.log"
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            evidence.extend(
                [
                    bind_evidence(
                        stdout_path,
                        evidence_id=f"functional-gate-{index}-stdout",
                        maximum_age_seconds=60,
                    ),
                    bind_evidence(
                        stderr_path,
                        evidence_id=f"functional-gate-{index}-stderr",
                        maximum_age_seconds=60,
                    ),
                ]
            )
            command_reports.append(
                {
                    "argv": argv,
                    "exit_code": completed.returncode,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
            )
            if completed.returncode != 0:
                failures.append(
                    f"functional gate {index} exited {completed.returncode}"
                )
        except subprocess.TimeoutExpired:
            failures.append(f"functional gate {index} exceeded {timeout_seconds}s")
    return QualificationCheck(
        subsystem="functional_hardening",
        status="passed" if not failures else "failed",
        summary="Fixed compile, full unit/integration suite, and release gate.",
        evidence=evidence,
        metrics={"commands": command_reports},
        blockers=failures,
    )


async def run_load_ladder(
    *,
    policy: QualificationPolicy,
    endpoint: str | None,
    model: str,
    api_key: str | None,
    timeout_seconds: float,
) -> list[LoadStageResult]:
    if not endpoint:
        return [
            LoadStageResult(
                name=stage.name,
                status="blocked",
                concurrency=stage.concurrency,
                requests=stage.requests,
                error_rate=1.0,
                throughput_rps=0.0,
                latency_ms_p50=0.0,
                latency_ms_p95=0.0,
                latency_ms_p99=0.0,
                maximum_latency_ms=0.0,
                streaming_passed=0,
                blockers=["serving endpoint is not configured"],
            )
            for stage in policy.load_stages
        ]
    results: list[LoadStageResult] = []
    for stage in policy.load_stages:
        started = time.perf_counter()
        raw: NativeServingLoadReport = await run_native_serving_load_test(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            requests=stage.requests,
            concurrency=stage.concurrency,
            timeout_seconds=timeout_seconds,
            streaming_requests=stage.streaming_requests,
        )
        duration = max(time.perf_counter() - started, 1e-9)
        error_rate = raw.failed / max(1, raw.requests)
        throughput = raw.passed / duration
        blockers: list[str] = []
        if error_rate > stage.maximum_error_rate:
            blockers.append("error rate exceeded stage SLO")
        if raw.latency_ms_p95 > stage.maximum_p95_latency_ms:
            blockers.append("p95 latency exceeded stage SLO")
        if throughput < stage.minimum_throughput_rps:
            blockers.append("throughput was below stage SLO")
        if raw.streaming_passed != stage.streaming_requests:
            blockers.append("not every SSE request completed")
        results.append(
            LoadStageResult(
                name=stage.name,
                status="passed"
                if raw.status == "passed" and not blockers
                else "failed",
                concurrency=stage.concurrency,
                requests=stage.requests,
                error_rate=round(error_rate, 6),
                throughput_rps=round(throughput, 6),
                latency_ms_p50=raw.latency_ms_p50,
                latency_ms_p95=raw.latency_ms_p95,
                latency_ms_p99=raw.latency_ms_p99,
                maximum_latency_ms=raw.max_latency_ms,
                streaming_passed=raw.streaming_passed,
                blockers=blockers + raw.error_samples[:5],
            )
        )
        if results[-1].status != "passed":
            break
    return results


def validate_training_proofs(
    path: str | None,
    *,
    policy: QualificationPolicy,
) -> tuple[QualificationCheck, QualificationCheck]:
    if not path:
        blocked = QualificationCheck(
            subsystem="failure_chaos",
            status="blocked",
            summary="No measured training/failure proof report was supplied.",
            blockers=["--training-proof-report is required"],
        )
        soak = QualificationCheck(
            subsystem="soak_recovery",
            status="blocked",
            summary="24-hour and 7-day soak evidence is absent.",
            blockers=[
                "a current training proof report containing both soak durations is required"
            ],
        )
        return blocked, soak
    try:
        binding = bind_evidence(
            path,
            evidence_id="training-proof-report",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        payload = load_json_evidence(binding)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("training proof report schema_version must be 1")
        require_fresh_timestamp(
            str(payload.get("generated_at") or ""),
            label="training proof report",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        rows = payload.get("proofs", [])
        if not isinstance(rows, list):
            raise ValueError("training proof report proofs must be a list")
        statuses = {
            str(row.get("name")): str(row.get("status"))
            for row in rows
            if isinstance(row, dict)
        }
        missing_failure = [
            name
            for name in policy.required_training_proofs
            if statuses.get(name) != "passed"
        ]
        proof_by_name = {
            str(row.get("name")): row for row in rows if isinstance(row, dict)
        }
        semantic_errors = [
            error
            for name in policy.required_training_proofs
            for error in _validate_training_proof_semantics(
                name,
                proof_by_name.get(name),
            )
        ]
        soak_names = [
            f"infrastructure_soak_{seconds}s"
            for seconds in policy.required_soak_seconds
        ]
        missing_soak = [name for name in soak_names if statuses.get(name) != "passed"]
        for seconds, name in zip(policy.required_soak_seconds, soak_names, strict=True):
            row = proof_by_name.get(name)
            if not isinstance(row, dict):
                continue
            evidence = (
                row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            )
            if float(row.get("duration_seconds", 0.0)) < seconds * 0.99:
                missing_soak.append(f"{name}: measured duration was too short")
            if int(evidence.get("requested_seconds", 0)) != seconds:
                missing_soak.append(f"{name}: requested duration evidence mismatch")
            if int(evidence.get("successful_transactions", 0)) < 1:
                missing_soak.append(f"{name}: no successful transaction evidence")
        failure = QualificationCheck(
            subsystem="failure_chaos",
            status="passed"
            if not missing_failure and not semantic_errors
            else "failed",
            summary="Controlled dependency restart, worker loss, ordered events, and durable recovery.",
            evidence=[binding],
            metrics={"required": policy.required_training_proofs, "statuses": statuses},
            blockers=[
                f"required proof did not pass: {name}" for name in missing_failure
            ]
            + semantic_errors,
        )
        soak = QualificationCheck(
            subsystem="soak_recovery",
            status="passed" if not missing_soak else "failed",
            summary="Required continuous infrastructure soak durations.",
            evidence=[binding],
            metrics={"required": soak_names},
            blockers=[f"required soak did not pass: {name}" for name in missing_soak],
        )
        return failure, soak
    except Exception as exc:
        failed = QualificationCheck(
            subsystem="failure_chaos",
            status="failed",
            summary="Training/failure proof evidence was invalid.",
            blockers=[str(exc)],
        )
        soak = QualificationCheck(
            subsystem="soak_recovery",
            status="failed",
            summary="Soak evidence was invalid.",
            blockers=[str(exc)],
        )
        return failed, soak


def _verify_bound_file(
    path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    if not path:
        raise ValueError(f"{label} path is missing")
    if HEX_SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} SHA-256 is missing or invalid")
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} is not a regular file")
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return source


def _validate_dataset_artifacts(dataset: ProductionDatasetManifest) -> list[str]:
    blockers: list[str] = []
    required = {
        "train",
        "val",
        "test",
        "split_manifest",
        "promotion_decision",
        "dedup_report",
        "contamination_report",
        "review_report",
        "verified_patch_report",
    }
    for name in sorted(required):
        path = str(dataset.artifacts.get(name) or "")
        expected = str(dataset.artifact_sha256.get(name) or "")
        try:
            _verify_bound_file(path, expected, label=f"dataset artifact {name}")
        except Exception as exc:
            blockers.append(str(exc))
    reports = dataset.reports if isinstance(dataset.reports, dict) else {}
    trust = reports.get("dataset_trust_metrics")
    split = reports.get("split_manifest")
    if not isinstance(trust, dict):
        blockers.append("dataset trust metrics are missing")
    else:
        required_values = {
            "license_coverage": 1.0,
            "provenance_coverage": 1.0,
            "high_value_review_coverage": 1.0,
            "verified_patch_evidence_coverage": 1.0,
        }
        for name, expected in required_values.items():
            if float(trust.get(name, -1.0)) != expected:
                blockers.append(f"dataset trust metric {name} must equal {expected}")
        if float(trust.get("average_quality", 0.0)) < 0.80:
            blockers.append("dataset average quality is below 0.80")
        if float(trust.get("p10_quality", 0.0)) < 0.70:
            blockers.append("dataset p10 quality is below 0.70")
        if int(trust.get("secret_or_pii_hits", -1)) != 0:
            blockers.append("dataset contains secret or disallowed PII findings")
        if float(trust.get("maximum_source_token_fraction", 1.0)) > 0.20:
            blockers.append("dataset single-source token share exceeds 20 percent")
        if float(trust.get("maximum_source_family_token_fraction", 1.0)) > 0.35:
            blockers.append("dataset source-family token share exceeds 35 percent")
    if (
        not isinstance(split, dict)
        or int(split.get("cross_split_group_collisions", -1)) != 0
    ):
        blockers.append("dataset split is not proven family-safe")
    promotion = dataset.promotion_decision
    checks = promotion.get("checks") if isinstance(promotion, dict) else None
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        blockers.append("dataset promotion checks are missing or contain a failure")
    return blockers


def validate_scratch_training_chain(
    *,
    calibration_preflight_report: str | None = None,
    calibration_200_decision: str | None,
    calibration_5k_decision: str | None,
    production_dataset_manifest: str | None,
    tokenizer_selection_promotion: str | None = None,
    tokenizer_audit_report: str | None,
    overfit_sanity_report: str | None,
    t4_1k_training_report: str | None,
    t4_10k_training_report: str | None,
    maximum_age_seconds: int,
    policy: QualificationPolicy | None = None,
) -> QualificationCheck:
    active_policy = policy or QualificationPolicy.model_validate(
        {
            "evidence_max_age_seconds": maximum_age_seconds,
            "minimum_security_reviewers": 2,
            "load_stages": [
                {
                    "name": "default",
                    "concurrency": 1,
                    "requests": 1,
                    "streaming_requests": 0,
                    "maximum_error_rate": 0.0,
                    "maximum_p95_latency_ms": 30_000,
                    "minimum_throughput_rps": 0.001,
                }
            ],
            "required_training_proofs": ["required"],
            "required_soak_seconds": [86_400, 604_800],
            "required_security_domains": ["required"],
            "required_canary_percentages": [100],
            "minimum_internal_canary_users": 1,
            "maximum_internal_canary_users": 1,
            "canary_maximum_error_rate": 0.01,
            "canary_maximum_p95_latency_ms": 30_000,
            "canary_rollback_error_rate": 0.02,
            "canary_maximum_rollback_seconds": 120,
        }
    )
    paths = {
        "calibration-200-decision": calibration_200_decision,
        "calibration-5k-decision": calibration_5k_decision,
        "production-dataset-manifest": production_dataset_manifest,
        "tokenizer-audit-report": tokenizer_audit_report,
        "overfit-sanity-report": overfit_sanity_report,
        "t4-1k-training-report": t4_1k_training_report,
        "t4-10k-training-report": t4_10k_training_report,
    }
    if active_policy.tokenizer_selection_mode == "evidence_selected":
        paths["tokenizer-selection-promotion"] = tokenizer_selection_promotion
    if calibration_200_decision is None and calibration_preflight_report:
        try:
            preflight_binding = bind_evidence(
                calibration_preflight_report,
                evidence_id="calibration-preflight-report",
                maximum_age_seconds=maximum_age_seconds,
            )
            preflight = CalibrationPreflightReport.model_validate(
                load_json_evidence(preflight_binding)
            )
            if preflight.status != "ready":
                if any(
                    item.startswith("reviewer_qualification:")
                    for item in preflight.blockers
                ):
                    next_stage = "reviewer-qualification-report"
                elif any(
                    item.startswith("source_legal_approval:")
                    for item in preflight.blockers
                ):
                    next_stage = "source-legal-approvals"
                else:
                    next_stage = "governance-preflight"
                return QualificationCheck(
                    subsystem="scratch_training_chain",
                    status="blocked",
                    summary="Reviewer/legal governance must pass before calibration collection.",
                    evidence=[preflight_binding],
                    metrics={"next_required_stage": next_stage},
                    blockers=list(preflight.blockers),
                )
        except Exception as exc:
            return QualificationCheck(
                subsystem="scratch_training_chain",
                status="failed",
                summary="Calibration preflight evidence is invalid or stale.",
                metrics={"next_required_stage": "governance-preflight"},
                blockers=[str(exc)],
            )
    missing = [name for name, path in paths.items() if not path]
    if missing:
        return QualificationCheck(
            subsystem="scratch_training_chain",
            status="blocked",
            summary="Governed 200 -> 5k -> 100k -> tokenizer -> T4 qualification chain is incomplete.",
            metrics={"next_required_stage": missing[0]},
            blockers=[f"missing evidence: {name}" for name in missing],
        )

    bindings: dict[str, EvidenceBinding] = {}
    blockers: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        for name, path in paths.items():
            bindings[name] = bind_evidence(
                str(path),
                evidence_id=name,
                maximum_age_seconds=maximum_age_seconds,
            )

        calibration_200 = CalibrationDecision.model_validate(
            load_json_evidence(bindings["calibration-200-decision"])
        )
        calibration_5k = CalibrationDecision.model_validate(
            load_json_evidence(bindings["calibration-5k-decision"])
        )
        dataset = ProductionDatasetManifest.model_validate(
            load_json_evidence(bindings["production-dataset-manifest"])
        )
        active_trust_policy = dataset.governance_paths.get("trust_policy_path")
        if not active_trust_policy:
            blockers.append(
                "production dataset manifest is missing its active trust-policy binding"
            )
        else:
            try:
                replayed_dataset = validate_dataset_manifest_for_promotion(
                    bindings["production-dataset-manifest"].path,
                    trust_policy_path=active_trust_policy,
                )
                replay_identity = (
                    replayed_dataset.dataset_id,
                    replayed_dataset.version_id,
                    replayed_dataset.status,
                    replayed_dataset.advancement_decision_sha256,
                )
                parsed_identity = (
                    dataset.dataset_id,
                    dataset.version_id,
                    dataset.status,
                    dataset.advancement_decision_sha256,
                )
                if replay_identity != parsed_identity:
                    blockers.append(
                        "production dataset authority replay returned a different manifest identity"
                    )
            except Exception as exc:
                blockers.append(f"production dataset authority replay failed: {exc}")
        tokenizer = TokenizerAuditReport.model_validate(
            load_json_evidence(bindings["tokenizer-audit-report"])
        )
        overfit = load_json_evidence(bindings["overfit-sanity-report"])
        t4_reports = {
            "t4_1k": load_json_evidence(bindings["t4-1k-training-report"]),
            "t4_10k": load_json_evidence(bindings["t4-10k-training-report"]),
        }

        if (
            calibration_200.stage != "calibration_200"
            or calibration_200.status != "passed"
            or calibration_200.dev_test
            or calibration_200.next_stage != "5k_calibration_allowed"
        ):
            blockers.append("calibration_200 decision did not unlock calibration_5k")
        if (
            calibration_5k.stage != "calibration_5k"
            or calibration_5k.status != "passed"
            or calibration_5k.dev_test
            or calibration_5k.next_stage != "100k_dataset_build_allowed"
        ):
            blockers.append("calibration_5k decision did not unlock production_100k")
        if (
            calibration_5k.prior_decision_sha256
            != bindings["calibration-200-decision"].sha256
        ):
            blockers.append(
                "calibration_5k is not hash-bound to the supplied calibration_200 decision"
            )

        if dataset.status != "promoted" or dataset.dev_smoke:
            blockers.append(
                "production dataset manifest is not a non-smoke promoted version"
            )
        if (
            int(dataset.metrics.get("promoted_records", 0))
            < active_policy.minimum_promoted_dataset_records
        ):
            blockers.append(
                "production dataset has fewer than "
                f"{active_policy.minimum_promoted_dataset_records:,} promoted records"
            )
        if (
            dataset.advancement_decision_sha256
            != bindings["calibration-5k-decision"].sha256
        ):
            blockers.append(
                "production dataset is not hash-bound to the supplied calibration_5k decision"
            )
        if dataset.promotion_decision.get("status") != "promoted":
            blockers.append("dataset promotion decision is not promoted")
        blockers.extend(_validate_dataset_artifacts(dataset))

        selected_vocab_size = active_policy.required_tokenizer_vocab_size
        if active_policy.tokenizer_selection_mode == "evidence_selected":
            try:
                promotion, decision, comparison, experiment_manifest = (
                    verify_promotion_chain(
                        bindings["tokenizer-selection-promotion"].path
                    )
                )
                if (
                    promotion.status != "promoted"
                    or decision.status != "passed"
                    or comparison.status != "passed"
                    or experiment_manifest.campaign.experiment_type
                    != "tokenizer_selection"
                ):
                    raise ValueError(
                        "tokenizer scientific promotion chain did not pass"
                    )
                selected_vocab_size = int(str(promotion.promoted_candidate))
                if (
                    selected_vocab_size
                    not in experiment_manifest.campaign.candidate_vocab_sizes
                ):
                    raise ValueError("promoted tokenizer is not a planned candidate")
                if (
                    experiment_manifest.bindings["dataset_manifest"].sha256
                    != bindings["production-dataset-manifest"].sha256
                ):
                    raise ValueError(
                        "tokenizer experiment is not bound to the production dataset"
                    )
                metrics["tokenizer_experiment_id"] = experiment_manifest.experiment_id
                metrics["tokenizer_promotion_sha256"] = promotion.promotion_sha256
            except Exception as exc:
                blockers.append(
                    f"tokenizer evidence-selection verification failed: {exc}"
                )
        if (
            tokenizer.status != "passed"
            or selected_vocab_size is None
            or tokenizer.vocab_size_requested != selected_vocab_size
            or tokenizer.vocab_size_actual != selected_vocab_size
            or tokenizer.special_tokens_missing
            or tokenizer.audit_failures
        ):
            blockers.append("evidence-selected tokenizer qualification did not pass")
        if (
            tokenizer.dataset_manifest_sha256
            != bindings["production-dataset-manifest"].sha256
        ):
            blockers.append(
                "tokenizer is not hash-bound to the supplied production dataset manifest"
            )
        if active_policy.require_family_safe_tokenizer_split and (
            not tokenizer.family_safe_split
            or tokenizer.split_strategy != "pre_split_family_safe"
        ):
            blockers.append("tokenizer corpus split is not family-safe")
        tokenizer_bound_files = {
            "tokenizer": (tokenizer.tokenizer_path, tokenizer.tokenizer_sha256),
            "tokenizer manifest": (
                tokenizer.tokenizer_manifest_path,
                tokenizer.tokenizer_manifest_sha256,
            ),
            "token shard manifest": (
                tokenizer.shard_manifest_path,
                tokenizer.shard_manifest_sha256,
            ),
            "token efficiency report": (
                tokenizer.efficiency_report_path,
                tokenizer.efficiency_report_sha256,
            ),
        }
        for label, (path, expected) in tokenizer_bound_files.items():
            try:
                _verify_bound_file(path, expected, label=label)
            except Exception as exc:
                blockers.append(str(exc))
        for source_path, expected in tokenizer.source_sha256.items():
            try:
                _verify_bound_file(source_path, expected, label="tokenizer source")
            except Exception as exc:
                blockers.append(str(exc))
        shard_payload = tokenizer.shard_manifest
        if (
            str(shard_payload.get("tokenizer_sha256") or "")
            != tokenizer.tokenizer_sha256
        ):
            blockers.append("token shard manifest tokenizer hash mismatch")
        if (
            str(shard_payload.get("dataset_manifest_sha256") or "")
            != tokenizer.dataset_manifest_sha256
        ):
            blockers.append("token shard manifest dataset hash mismatch")
        if str(shard_payload.get("split_strategy") or "") != tokenizer.split_strategy:
            blockers.append("token shard manifest split strategy mismatch")

        if not isinstance(overfit, dict) or overfit.get("status") != "passed":
            blockers.append("controlled overfit sanity did not pass")
        else:
            relative_drop = float(overfit.get("relative_loss_drop", -1.0))
            required_drop = float(overfit.get("required_relative_loss_drop", 0.20))
            if not math.isfinite(relative_drop) or relative_drop < max(
                0.20, required_drop
            ):
                blockers.append(
                    "controlled overfit loss did not drop by the required amount"
                )
            overfit_training = overfit.get("training_report")
            if (
                not isinstance(overfit_training, dict)
                or overfit_training.get("scratch_only") is not True
            ):
                blockers.append(
                    "controlled overfit report is not bound to a scratch training run"
                )
            elif (
                overfit_training.get("checkpoint_reload_verified") is not True
                or overfit_training.get("checkpoint_reload_logit_parity") is not True
            ):
                blockers.append(
                    "controlled overfit checkpoint reload parity did not pass"
                )

        required_steps = {"t4_1k": 1_000, "t4_10k": 10_000}
        for name, payload in t4_reports.items():
            if not isinstance(payload, dict):
                blockers.append(f"{name} training report must be an object")
                continue
            model = (
                payload.get("model_config")
                if isinstance(payload.get("model_config"), dict)
                else {}
            )
            if payload.get("status") not in {"passed", "early_stopped"}:
                blockers.append(f"{name} training status did not pass")
            if payload.get("scratch_only") is not True:
                blockers.append(f"{name} is not scratch-only")
            if int(payload.get("steps", 0)) < required_steps[name]:
                blockers.append(
                    f"{name} did not complete {required_steps[name]} measured steps"
                )
            if payload.get("checkpoint_reload_verified") is not True:
                blockers.append(f"{name} checkpoint reload was not verified")
            if payload.get("checkpoint_reload_logit_parity") is not True:
                blockers.append(
                    f"{name} checkpoint reload logit parity was not verified"
                )
            reload_difference = float(
                payload.get("checkpoint_reload_max_abs_logit_difference", math.inf)
            )
            if (
                not math.isfinite(reload_difference)
                or reload_difference
                > active_policy.maximum_checkpoint_reload_logit_difference
            ):
                blockers.append(
                    f"{name} checkpoint reload logit difference exceeded policy"
                )
            if (
                int(model.get("hidden_size", 0)) < active_policy.t4_minimum_hidden_size
                or int(model.get("num_layers", 0)) < active_policy.t4_minimum_layers
                or int(model.get("max_sequence_length", 0))
                < active_policy.t4_minimum_sequence_length
            ):
                blockers.append(
                    f"{name} model is below the T4 technical qualification profile"
                )
            if (
                selected_vocab_size is None
                or int(model.get("vocab_size", 0)) != selected_vocab_size
            ):
                blockers.append(
                    f"{name} model vocabulary does not match the qualified tokenizer"
                )
            if (
                payload.get("dataset_manifest_sha256")
                != tokenizer.shard_manifest_sha256
            ):
                blockers.append(
                    f"{name} training dataset does not match the qualified shard manifest"
                )
            if payload.get("shard_manifest_sha256") != tokenizer.shard_manifest_sha256:
                blockers.append(f"{name} shard manifest lineage mismatch")
            if payload.get("tokenizer_sha256") != tokenizer.tokenizer_sha256:
                blockers.append(f"{name} tokenizer lineage mismatch")
            git_revision = str(payload.get("git_commit") or "")
            if re.fullmatch(r"[0-9a-f]{40}", git_revision) is None:
                blockers.append(f"{name} Git commit binding is missing or invalid")
            for field in ("checkpoint_manifest", "best_checkpoint_manifest"):
                digest_field = f"{field}_sha256"
                try:
                    _verify_bound_file(
                        str(payload.get(field) or ""),
                        str(payload.get(digest_field) or ""),
                        label=f"{name} {field.replace('_', ' ')}",
                    )
                except Exception as exc:
                    blockers.append(str(exc))
            losses = payload.get("train_losses")
            if (
                not isinstance(losses, list)
                or not losses
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in losses
                )
            ):
                blockers.append(
                    f"{name} training loss evidence is missing or non-finite"
                )
            validation_losses = payload.get("validation_losses")
            if not isinstance(validation_losses, list) or not validation_losses:
                blockers.append(f"{name} validation loss evidence is missing")
            elif any(
                not isinstance(row, dict)
                or not isinstance(row.get("loss"), (int, float))
                or not math.isfinite(float(row["loss"]))
                for row in validation_losses
            ):
                blockers.append(f"{name} validation loss evidence is non-finite")

        metrics.update(
            {
                "calibration_200": calibration_200.status,
                "calibration_5k": calibration_5k.status,
                "promoted_records": int(dataset.metrics.get("promoted_records", 0)),
                "dataset_manifest_sha256": bindings[
                    "production-dataset-manifest"
                ].sha256,
                "tokenizer_vocab_size": tokenizer.vocab_size_actual,
                "tokenizer_sha256": tokenizer.tokenizer_sha256,
                "token_shard_manifest_sha256": tokenizer.shard_manifest_sha256,
                "overfit_relative_loss_drop": float(
                    overfit.get("relative_loss_drop", -1.0)
                ),
                "t4_1k_steps": int(t4_reports["t4_1k"].get("steps", 0)),
                "t4_10k_steps": int(t4_reports["t4_10k"].get("steps", 0)),
            }
        )
    except Exception as exc:
        blockers.append(str(exc))

    return QualificationCheck(
        subsystem="scratch_training_chain",
        status="passed" if not blockers else "failed",
        summary="Hash-bound governed data, tokenizer, and T4 scratch-training advancement.",
        evidence=list(bindings.values()),
        metrics=metrics,
        blockers=blockers,
    )


def write_scratch_advancement_report(
    check: QualificationCheck,
    *,
    output_dir: str | Path,
) -> tuple[ScratchAdvancementReport, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_id = f"scratch-advancement-{uuid.uuid4()}"
    unsigned = ScratchAdvancementReport(
        report_id=report_id,
        created_at=utc_now(),
        git_commit=_git_commit(),
        check=check,
    )
    digest = hashlib.sha256(
        canonical_json_bytes(unsigned.model_dump(mode="json"))
    ).hexdigest()
    report = unsigned.model_copy(update={"report_sha256": digest})
    target = root / f"{report_id}.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_json_bytes(report.model_dump(mode="json")) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report, target


def verify_scratch_advancement_report(path: str | Path) -> ScratchAdvancementReport:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > 10_000_000:
        raise ValueError(
            "scratch advancement report must be a regular file no larger than 10 MiB"
        )
    report = ScratchAdvancementReport.model_validate_json(
        source.read_text(encoding="utf-8-sig")
    )
    unsigned = report.model_copy(update={"report_sha256": ""})
    expected = hashlib.sha256(
        canonical_json_bytes(unsigned.model_dump(mode="json"))
    ).hexdigest()
    if not hmac.compare_digest(report.report_sha256, expected):
        raise ValueError("scratch advancement report digest verification failed")
    return report


def _validate_training_proof_semantics(
    name: str,
    row: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(row, dict) or row.get("status") != "passed":
        return []
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        return [f"{name}: evidence must be an object"]
    errors: list[str] = []
    if name == "postgres_redis_minio_lifecycle":
        if int(evidence.get("redis_event_count", 0)) < 1:
            errors.append(f"{name}: Redis event round-trip is missing")
        if evidence.get("duplicate_event_rejected") is not True:
            errors.append(f"{name}: idempotent duplicate rejection is missing")
        database = evidence.get("database")
        if not isinstance(database, dict) or any(
            int(database.get(key, 0)) < 1
            for key in ("job_rows", "attempt_rows", "ingress_rows")
        ):
            errors.append(f"{name}: durable Postgres lifecycle evidence is incomplete")
    elif name == "ordered_event_stress_1000000":
        if (
            int(evidence.get("event_count", 0)) < 1_000_000
            or int(evidence.get("final_sequence", 0)) != 1_000_000
            or int(evidence.get("tail_sequence", 0)) != 1_000_000
        ):
            errors.append(f"{name}: one-million ordered event evidence is incomplete")
    elif name == "docker_disaster_recovery_drill":
        required = (
            "postgres_dump_restore",
            "postgres_restart",
            "redis_aof_restart",
            "minio_volume_restart",
            "qdrant_volume_restart",
        )
        if any(evidence.get(key) is not True for key in required):
            errors.append(f"{name}: one or more dependency recovery checks are absent")
    elif name == "worker_loss_retry_recovery":
        if (
            evidence.get("status_after_loss") != "queued"
            or evidence.get("retry_failure_class") != "node_loss"
            or int(evidence.get("attempt_count", 0)) < 1
        ):
            errors.append(f"{name}: bounded worker-loss retry evidence is incomplete")
    return errors


def validate_security_review(
    path: str | None,
    *,
    policy: QualificationPolicy,
    expected_scanner_report_sha256: str | None = None,
) -> QualificationCheck:
    if not path:
        return QualificationCheck(
            subsystem="independent_security_review",
            status="blocked",
            summary="Independent manual security review is not supplied.",
            blockers=["--manual-security-review is required"],
        )
    try:
        binding = bind_evidence(
            path,
            evidence_id="manual-security-review",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        review = SecurityReviewEvidence.model_validate(load_json_evidence(binding))
        require_fresh_timestamp(
            review.reviewed_at,
            label="security review",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        missing_domains = [
            domain
            for domain in policy.required_security_domains
            if review.domains.get(domain) != "passed"
        ]
        blockers = [
            f"security domain did not pass: {domain}" for domain in missing_domains
        ]
        if len(review.reviewers) < policy.minimum_security_reviewers:
            blockers.append("independent reviewer count is below policy")
        if (
            review.critical_findings
            or review.high_findings
            or review.unresolved_findings
        ):
            blockers.append("security review has unresolved critical/high findings")
        if review.decision != "approved":
            blockers.append("security review decision is not approved")
        if expected_scanner_report_sha256 and not hmac.compare_digest(
            review.scanner_report_sha256,
            expected_scanner_report_sha256,
        ):
            blockers.append(
                "manual review is not bound to the current automated scanner report"
            )
        return QualificationCheck(
            subsystem="independent_security_review",
            status="passed" if not blockers else "failed",
            summary="Independent threat-model and manual security review.",
            evidence=[binding],
            metrics={
                "review_id": review.review_id,
                "reviewer_count": len(review.reviewers),
                "domains": review.domains,
                "scanner_report_sha256": review.scanner_report_sha256,
            },
            blockers=blockers,
        )
    except Exception as exc:
        return QualificationCheck(
            subsystem="independent_security_review",
            status="failed",
            summary="Manual security review evidence was invalid.",
            blockers=[str(exc)],
        )


def validate_canary(
    path: str | None,
    *,
    policy: QualificationPolicy,
) -> QualificationCheck:
    if not path:
        return QualificationCheck(
            subsystem="canary_release",
            status="blocked",
            summary="No measured internal-user and staged traffic canary is supplied.",
            blockers=["--canary-report is required"],
        )
    try:
        binding = bind_evidence(
            path,
            evidence_id="canary-report",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        canary = CanaryEvidence.model_validate(load_json_evidence(binding))
        require_fresh_timestamp(
            canary.created_at,
            label="canary",
            maximum_age_seconds=policy.evidence_max_age_seconds,
        )
        by_percentage = {stage.percentage: stage for stage in canary.stages}
        blockers: list[str] = []
        if not (
            policy.minimum_internal_canary_users
            <= canary.internal_user_count
            <= policy.maximum_internal_canary_users
        ):
            blockers.append("internal canary user count is outside policy")
        for percentage in policy.required_canary_percentages:
            stage = by_percentage.get(percentage)
            if stage is None:
                blockers.append(f"missing {percentage}% canary stage")
                continue
            if stage.error_rate > policy.canary_maximum_error_rate:
                blockers.append(f"{percentage}% canary error rate exceeded policy")
            if stage.p95_latency_ms > policy.canary_maximum_p95_latency_ms:
                blockers.append(f"{percentage}% canary p95 latency exceeded policy")
            if not stage.rollback_trigger_tested or not stage.rollback_succeeded:
                blockers.append(f"{percentage}% canary rollback proof is incomplete")
            if stage.rollback_trigger_error_rate < policy.canary_rollback_error_rate:
                blockers.append(
                    f"{percentage}% canary rollback was not triggered at the policy threshold"
                )
            if (
                stage.rollback_completed_seconds
                > policy.canary_maximum_rollback_seconds
            ):
                blockers.append(f"{percentage}% canary rollback exceeded recovery SLO")
        if canary.status != "passed":
            blockers.append("canary report status is not passed")
        return QualificationCheck(
            subsystem="canary_release",
            status="passed" if not blockers else "failed",
            summary="Internal users and progressive 1/10/50/100 percent traffic canary.",
            evidence=[binding],
            metrics={
                "canary_id": canary.canary_id,
                "stages": [row.model_dump() for row in canary.stages],
            },
            blockers=blockers,
        )
    except Exception as exc:
        return QualificationCheck(
            subsystem="canary_release",
            status="failed",
            summary="Canary evidence was invalid.",
            blockers=[str(exc)],
        )


async def check_observability(
    *,
    metrics_url: str | None,
    prometheus_url: str | None,
    grafana_url: str | None,
    otel_health_url: str | None,
    alertmanager_url: str | None,
    operator_notification_report: str | None,
    policy: QualificationPolicy,
    allowed_insecure_hosts: list[str],
) -> QualificationCheck:
    endpoints = {
        "metrics": metrics_url,
        "prometheus": prometheus_url,
        "grafana": grafana_url,
        "otel": otel_health_url,
        "alertmanager": alertmanager_url,
    }
    if any(not value for value in endpoints.values()):
        missing = [name for name, value in endpoints.items() if not value]
        return QualificationCheck(
            subsystem="observability",
            status="blocked",
            summary="Live logs/metrics/tracing stack is not fully configured.",
            blockers=[f"missing observability endpoint: {name}" for name in missing],
        )
    if policy.require_operator_notification_proof and not operator_notification_report:
        return QualificationCheck(
            subsystem="observability",
            status="blocked",
            summary="Live observability endpoints are configured, but operator delivery proof is absent.",
            blockers=["--operator-notification-report is required"],
        )
    try:
        notification_binding: EvidenceBinding | None = None
        notification: OperatorNotificationEvidence | None = None
        if operator_notification_report:
            notification_binding = bind_evidence(
                operator_notification_report,
                evidence_id="operator-notification-report",
                maximum_age_seconds=policy.evidence_max_age_seconds,
            )
            notification = OperatorNotificationEvidence.model_validate(
                load_json_evidence(notification_binding)
            )
            require_fresh_timestamp(
                notification.created_at,
                label="operator notification proof",
                maximum_age_seconds=policy.evidence_max_age_seconds,
            )
            firing_at = require_fresh_timestamp(
                notification.firing_delivered_at,
                label="operator firing notification",
                maximum_age_seconds=policy.evidence_max_age_seconds,
            )
            recovery_at = require_fresh_timestamp(
                notification.recovery_delivered_at,
                label="operator recovery notification",
                maximum_age_seconds=policy.evidence_max_age_seconds,
            )
            if recovery_at < firing_at:
                raise ValueError(
                    "operator recovery notification predates firing notification"
                )
            if (
                notification.status != "passed"
                or not notification.firing_delivered
                or not notification.recovery_delivered
            ):
                raise ValueError(
                    "operator firing and recovery notifications were not both delivered"
                )
        validated = {
            name: _validated_service_url(
                str(value),
                label=name,
                allowed_insecure_hosts=allowed_insecure_hosts,
            )
            for name, value in endpoints.items()
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            (
                metric_response,
                prometheus_response,
                grafana_response,
                otel_response,
                alertmanager_response,
            ) = await asyncio.gather(
                client.get(validated["metrics"]),
                client.get(validated["prometheus"]),
                client.get(validated["grafana"]),
                client.get(validated["otel"]),
                client.get(f"{validated['alertmanager'].rstrip('/')}/-/ready"),
            )
            alert_lifecycle = await prove_alert_lifecycle(
                client,
                validated["alertmanager"],
            )
        for response in (
            metric_response,
            prometheus_response,
            grafana_response,
            otel_response,
            alertmanager_response,
        ):
            response.raise_for_status()
        if "craftly_http_requests_total" not in metric_response.text:
            raise ValueError("metrics endpoint does not expose Craftly HTTP metrics")
        grafana_payload = grafana_response.json()
        if str(grafana_payload.get("database", "")).lower() != "ok":
            raise ValueError("Grafana health did not report database=ok")
        return QualificationCheck(
            subsystem="observability",
            status="passed",
            summary="Live metrics, tracing, alert lifecycle, and operator delivery are proven.",
            evidence=[notification_binding] if notification_binding else [],
            metrics={
                "prometheus_ready": prometheus_response.status_code == 200,
                "grafana_database": grafana_payload.get("database"),
                "otel_ready": otel_response.status_code == 200,
                "alertmanager_ready": alertmanager_response.status_code == 200,
                "alert_lifecycle": alert_lifecycle,
                "operator_notification": (
                    {
                        "proof_id": notification.proof_id,
                        "provider": notification.provider,
                        "channel_type": notification.channel_type,
                        "firing_delivered": notification.firing_delivered,
                        "recovery_delivered": notification.recovery_delivered,
                    }
                    if notification
                    else {"required": False}
                ),
            },
        )
    except Exception as exc:
        return QualificationCheck(
            subsystem="observability",
            status="failed",
            summary="Live observability proof failed.",
            blockers=[str(exc)],
        )


async def prove_alert_lifecycle(
    client: httpx.AsyncClient,
    endpoint: str,
) -> dict[str, Any]:
    proof_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    labels = {
        "alertname": "CraftlyQualificationSynthetic",
        "service": "craftly-qualification",
        "severity": "info",
        "craftly_qualification_proof": "true",
        "craftly_proof_id": proof_id,
    }
    firing = [
        {
            "labels": labels,
            "annotations": {
                "summary": "Synthetic qualification alert; routed to null receiver"
            },
            "startsAt": now.isoformat(),
            "endsAt": datetime.fromtimestamp(now.timestamp() + 300, UTC).isoformat(),
            "generatorURL": "https://craftly.invalid/qualification",
        }
    ]
    posted = await client.post(
        f"{endpoint.rstrip('/')}/api/v2/alerts",
        json=firing,
    )
    posted.raise_for_status()

    async def is_active() -> bool:
        response = await client.get(
            f"{endpoint.rstrip('/')}/api/v2/alerts",
            params={
                "active": "true",
                "silenced": "true",
                "inhibited": "true",
                "unprocessed": "true",
            },
        )
        response.raise_for_status()
        return any(
            isinstance(row, dict)
            and row.get("labels", {}).get("craftly_proof_id") == proof_id
            for row in response.json()
        )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not await is_active():
        await asyncio.sleep(0.25)
    if not await is_active():
        raise RuntimeError("synthetic Alertmanager alert did not enter active state")

    resolved = [
        {
            "labels": labels,
            "annotations": firing[0]["annotations"],
            "startsAt": firing[0]["startsAt"],
            "endsAt": datetime.now(UTC).isoformat(),
            "generatorURL": firing[0]["generatorURL"],
        }
    ]
    posted_resolution = await client.post(
        f"{endpoint.rstrip('/')}/api/v2/alerts",
        json=resolved,
    )
    posted_resolution.raise_for_status()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not await is_active():
            return {"proof_id": proof_id, "fired": True, "resolved": True}
        await asyncio.sleep(0.5)
    raise RuntimeError("synthetic Alertmanager alert did not resolve")


def _map_live_proof(
    report: ProductionProofReport,
) -> tuple[QualificationCheck, QualificationCheck]:
    by_name = {check.name: check for check in report.checks}
    dependency_names = {
        "postgres_migrations",
        "redis_quota",
        "object_store_lifecycle",
        "qdrant_round_trip",
    }
    serving_names = {"native_serving_health", "benchmark_and_model_evidence"}

    def mapped(subsystem: str, names: set[str], summary: str) -> QualificationCheck:
        rows = [by_name[name] for name in names if name in by_name]
        missing = sorted(names - set(by_name))
        failed = [row for row in rows if row.status == "failed"]
        blocked = [
            row for row in rows if row.status in {"skipped", "blocked", "not_run"}
        ]
        status: QualificationStatus = (
            "failed" if failed else "blocked" if blocked or missing else "passed"
        )
        return QualificationCheck(
            subsystem=subsystem,
            status=status,
            summary=summary,
            metrics={row.name: row.details for row in rows},
            blockers=[
                f"{row.name}: {row.error or row.details.get('reason', row.status)}"
                for row in failed + blocked
            ]
            + [f"missing live proof check: {name}" for name in missing],
        )

    return (
        mapped(
            "real_dependency_e2e",
            dependency_names,
            "Live Postgres, Redis, object storage, and Qdrant lifecycle.",
        ),
        mapped(
            "native_model_serving",
            serving_names,
            "Authenticated Craftly scratch checkpoint identity and executable evaluation.",
        ),
    )


async def run_qualification(
    args: argparse.Namespace,
) -> tuple[ProductionQualificationReport, Path]:
    policy_path = Path(args.policy).expanduser().resolve(strict=True)
    policy = QualificationPolicy.from_file(policy_path)
    signing_key = os.environ.get("CRAFTLY_PROOF_SIGNING_KEY")
    if args.production and policy.require_report_signature_in_production:
        if not signing_key or len(signing_key.encode("utf-8")) < 32:
            raise ValueError(
                "production qualification requires CRAFTLY_PROOF_SIGNING_KEY with at least 32 bytes"
            )
    output_dir = Path(args.output_dir).expanduser().resolve()
    live_config = ProductionProofConfig(
        strict=args.production,
        output_dir=str(output_dir / "live-proof"),
        postgres_url=args.postgres_url or os.environ.get("CRAFTLY_DATABASE_URL"),
        apply_postgres_migrations=args.apply_postgres_migrations,
        redis_url=args.redis_url or os.environ.get("CRAFTLY_REDIS_URL"),
        object_store_uri=args.object_store_uri
        or os.environ.get("CRAFTLY_OBJECT_STORE_URI"),
        object_store_endpoint_url=(
            args.object_store_endpoint_url
            or os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL")
        ),
        qdrant_url=args.qdrant_url or os.environ.get("CRAFTLY_QDRANT_URL"),
        allowed_insecure_service_hosts=args.allow_insecure_service_host,
        serving_url=args.serving_url or os.environ.get("CRAFTLY_SERVING_URL"),
        serving_api_key=args.serving_api_key or os.environ.get("CRAFTLY_MODEL_API_KEY"),
        serving_model=args.serving_model,
        load_test_requests=max(1, policy.load_stages[0].requests),
        load_test_concurrency=policy.load_stages[0].concurrency,
        load_test_timeout_seconds=args.request_timeout_seconds,
        load_test_streaming_requests=max(1, policy.load_stages[0].streaming_requests),
        executable_benchmark_report=args.executable_benchmark_report,
        scorecard_report=args.scorecard_report,
        active_model_profile=args.active_model_profile,
        run_security_audit=args.run_security_audit,
        strict_security_tools=args.strict_security_tools,
    )
    live = await run_production_proof(live_config)
    dependency_check, serving_check = _map_live_proof(live)
    if args.run_functional_gates:
        functional = await asyncio.to_thread(
            run_fixed_functional_gates,
            output_dir / "functional-gates",
            timeout_seconds=args.functional_timeout_seconds,
        )
    else:
        functional = QualificationCheck(
            subsystem="functional_hardening",
            status="not_run",
            summary="Fixed functional gates were not requested.",
            blockers=["use --run-functional-gates"],
        )
    load_stages = await run_load_ladder(
        policy=policy,
        endpoint=live_config.serving_url,
        model=live_config.serving_model,
        api_key=live_config.serving_api_key,
        timeout_seconds=args.request_timeout_seconds,
    )
    load_status: QualificationStatus = (
        "failed"
        if any(stage.status == "failed" for stage in load_stages)
        else "blocked"
        if any(stage.status == "blocked" for stage in load_stages)
        else "passed"
    )
    load_check = QualificationCheck(
        subsystem="load_capacity",
        status=load_status,
        summary="Progressive 10/100/500/1000 concurrency capacity ladder.",
        metrics={"stages": [stage.model_dump() for stage in load_stages]},
        blockers=[item for stage in load_stages for item in stage.blockers],
    )
    failure, soak = validate_training_proofs(args.training_proof_report, policy=policy)
    scratch_training_chain = validate_scratch_training_chain(
        calibration_200_decision=args.calibration_200_decision,
        calibration_5k_decision=args.calibration_5k_decision,
        production_dataset_manifest=args.production_dataset_manifest,
        tokenizer_selection_promotion=args.tokenizer_selection_promotion,
        tokenizer_audit_report=args.tokenizer_audit_report,
        overfit_sanity_report=args.overfit_sanity_report,
        t4_1k_training_report=args.t4_1k_training_report,
        t4_10k_training_report=args.t4_10k_training_report,
        maximum_age_seconds=policy.evidence_max_age_seconds,
        policy=policy,
    )
    observability = await check_observability(
        metrics_url=args.metrics_url,
        prometheus_url=args.prometheus_url,
        grafana_url=args.grafana_url,
        otel_health_url=args.otel_health_url,
        alertmanager_url=args.alertmanager_url,
        operator_notification_report=args.operator_notification_report,
        policy=policy,
        allowed_insecure_hosts=args.allow_insecure_service_host,
    )
    scanner_report_path = (
        output_dir / "live-proof" / "security-audit" / "security_audit_report.json"
    )
    scanner_hash = (
        sha256_file(scanner_report_path) if scanner_report_path.exists() else None
    )
    security = validate_security_review(
        args.manual_security_review,
        policy=policy,
        expected_scanner_report_sha256=scanner_hash,
    )
    live_security = next(
        (check for check in live.checks if check.name == "security_audit"),
        None,
    )
    if live_security is None or live_security.status != "passed":
        reason = (
            "automated security audit did not run"
            if live_security is None
            else live_security.error
            or str(live_security.details.get("reason") or live_security.status)
        )
        security.status = (
            "failed"
            if live_security and live_security.status == "failed"
            else "blocked"
        )
        security.blockers.append(f"automated security audit: {reason}")
    canary = validate_canary(args.canary_report, policy=policy)
    checks = [
        QualificationCheck(
            subsystem="production_proof_baseline",
            status="passed",
            summary="Current run uses versioned, hash-chained, tamper-evident evidence.",
            metrics={"live_proof_status": live.status},
        ),
        dependency_check,
        functional,
        scratch_training_chain,
        serving_check,
        load_check,
        failure,
        observability,
        soak,
        security,
        canary,
    ]
    status: QualificationStatus = (
        "failed"
        if any(check.status == "failed" for check in checks)
        else "blocked"
        if any(check.status == "blocked" for check in checks)
        else "not_run"
        if any(check.status == "not_run" for check in checks)
        else "passed"
    )
    report = ProductionQualificationReport(
        report_id=str(uuid.uuid4()),
        status=status,
        mode="production" if args.production else "validation",
        created_at=utc_now(),
        git_commit=_git_commit(),
        policy_sha256=sha256_file(policy_path),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname_sha256": hashlib.sha256(
                platform.node().encode("utf-8")
            ).hexdigest(),
        },
        checks=checks,
        load_stages=load_stages,
    )
    store = ImmutableQualificationStore(output_dir, signing_key=signing_key)
    path = store.write(report)
    persisted = ProductionQualificationReport.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )
    return persisted, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the authoritative Craftly production qualification decision."
    )
    parser.add_argument("--policy", default="config/production_qualification.json")
    parser.add_argument(
        "--output-dir", default="artifacts/craftly/production-qualification"
    )
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--postgres-url")
    parser.add_argument("--apply-postgres-migrations", action="store_true")
    parser.add_argument("--redis-url")
    parser.add_argument("--object-store-uri")
    parser.add_argument("--object-store-endpoint-url")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--allow-insecure-service-host", action="append", default=[])
    parser.add_argument("--serving-url")
    parser.add_argument("--serving-api-key")
    parser.add_argument("--serving-model", default="craftly-scratch")
    parser.add_argument("--active-model-profile")
    parser.add_argument("--executable-benchmark-report")
    parser.add_argument("--scorecard-report")
    parser.add_argument("--training-proof-report")
    parser.add_argument("--calibration-preflight-report")
    parser.add_argument("--calibration-200-decision")
    parser.add_argument("--calibration-5k-decision")
    parser.add_argument("--production-dataset-manifest")
    parser.add_argument("--tokenizer-selection-promotion")
    parser.add_argument("--tokenizer-audit-report")
    parser.add_argument("--overfit-sanity-report")
    parser.add_argument("--t4-1k-training-report")
    parser.add_argument("--t4-10k-training-report")
    parser.add_argument("--manual-security-review")
    parser.add_argument("--canary-report")
    parser.add_argument("--metrics-url")
    parser.add_argument("--prometheus-url")
    parser.add_argument("--grafana-url")
    parser.add_argument("--otel-health-url")
    parser.add_argument("--alertmanager-url")
    parser.add_argument("--operator-notification-report")
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--run-functional-gates", action="store_true")
    parser.add_argument("--functional-timeout-seconds", type=int, default=1800)
    parser.add_argument("--run-security-audit", action="store_true")
    parser.add_argument("--strict-security-tools", action="store_true")
    parser.add_argument(
        "--scratch-chain-only",
        action="store_true",
        help="Validate and persist only the governed 200 -> 5k -> 100k -> tokenizer -> T4 evidence ladder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scratch_chain_only:
        policy = QualificationPolicy.from_file(args.policy)
        check = validate_scratch_training_chain(
            calibration_preflight_report=args.calibration_preflight_report,
            calibration_200_decision=args.calibration_200_decision,
            calibration_5k_decision=args.calibration_5k_decision,
            production_dataset_manifest=args.production_dataset_manifest,
            tokenizer_selection_promotion=args.tokenizer_selection_promotion,
            tokenizer_audit_report=args.tokenizer_audit_report,
            overfit_sanity_report=args.overfit_sanity_report,
            t4_1k_training_report=args.t4_1k_training_report,
            t4_10k_training_report=args.t4_10k_training_report,
            maximum_age_seconds=policy.evidence_max_age_seconds,
            policy=policy,
        )
        advancement, path = write_scratch_advancement_report(
            check, output_dir=args.output_dir
        )
        print(
            json.dumps(
                {
                    "status": advancement.check.status,
                    "report_id": advancement.report_id,
                    "report_path": str(path),
                    "report_sha256": advancement.report_sha256,
                    "next_required_stage": advancement.check.metrics.get(
                        "next_required_stage"
                    ),
                    "blockers": advancement.check.blockers,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if advancement.check.status != "passed":
            raise SystemExit(1)
        return

    report, path = asyncio.run(run_qualification(args))
    print(
        json.dumps(
            {
                "status": report.status,
                "report_id": report.report_id,
                "report_path": str(path),
                "report_sha256": report.report_sha256,
                "checks": {check.subsystem: check.status for check in report.checks},
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
