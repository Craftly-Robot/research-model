"""Governed calibration ladder for Craftly dataset advancement.

This module coordinates existing authorities; it does not create another data
promotion implementation. It refuses to crawl until legal/source contracts and
the protected benchmark pack are valid. Production advancement is fixed at
200 -> 5,000 -> 100,000 records. A successful crawl remains ``awaiting_review``
until two independent humans complete the bound sample.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, model_validator

from src.craftly.evaluation.benchmark_pack import (
    ProtectedBenchmarkManifest,
    validate_protected_benchmark_manifest,
)
from src.craftly.learning.benchmark_contamination_filter import (
    BenchmarkContaminationFilterReport,
    filter_benchmark_contamination_jsonl,
)
from src.craftly.learning.data_engine import DataEngine, DataEngineConfig, FrontierStore
from src.craftly.learning.dataset_authority import (
    ReviewItemCreate,
    ReviewerQualificationEvaluationReport,
    ReviewerRoster,
    SQLiteDatasetAuthorityStore,
    SourceSnapshotCreate,
    load_reviewer_roster,
)
from src.craftly.learning.license_filter import LicenseFilterReport, filter_jsonl_by_license
from src.craftly.learning.near_dedup import NearDedupReport, deduplicate_jsonl
from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.learning.quality_inspector import QualityInspectionReport, write_quality_report
from src.craftly.learning.source_registry import (
    SourceRegistry,
    SourceRegistryReport,
    source_registry_entry_sha256,
)
from src.craftly.shared.config_contracts import DatasetTrustPolicyContract, load_dataset_trust_policy
from src.craftly.shared.progress import ProgressReporter, progress_from_options
from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file as _sha256_file


CalibrationStage = Literal["calibration_200", "calibration_5k"]
CalibrationNextStage = Literal[
    "repeat_current_stage",
    "5k_calibration_allowed",
    "100k_dataset_build_allowed",
]
STAGE_RECORD_COUNTS: dict[str, int] = {
    "calibration_200": 200,
    "calibration_5k": 5_000,
}
STAGE_PRIOR_REQUIREMENTS: dict[str, tuple[str, str] | None] = {
    "calibration_200": None,
    "calibration_5k": ("calibration_200", "5k_calibration_allowed"),
}


class CalibrationPreflightReport(StrictModel):
    status: Literal["ready", "blocked"]
    source_registry: dict[str, Any] | None = None
    protected_benchmark_manifest: dict[str, Any] | None = None
    reviewer_roster: dict[str, Any] | None = None
    reviewer_qualification: dict[str, Any] | None = None
    approval_request_dir: str
    legal_evidence_dir: str
    blockers: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)


class CalibrationReviewBinding(StrictModel):
    review_item_id: str
    content_hash: str
    source_id: str
    source_snapshot_sha256: str
    data_type: str
    actual_high_value: bool


class CalibrationManifest(StrictModel):
    schema_version: Literal[2] = 2
    calibration_id: str
    stage: CalibrationStage
    status: Literal["awaiting_review"]
    dev_test: bool = False
    target_records: int
    final_records: int
    prior_decision_path: str | None = None
    prior_decision_sha256: str | None = None
    source_registry_sha256: str
    trust_policy_sha256: str
    reviewer_roster_sha256: str
    reviewer_qualification_path: str
    reviewer_qualification_sha256: str
    legal_evidence_sha256: str
    protected_manifest_sha256: str
    authority_database: str
    artifacts: dict[str, str]
    artifact_sha256: dict[str, str]
    source_fractions: dict[str, float]
    review_bindings: list[CalibrationReviewBinding]
    crawl_report: dict[str, Any]
    license_report: dict[str, Any]
    contamination_report: dict[str, Any]
    dedup_report: dict[str, Any]
    quality_report: dict[str, Any]
    created_at_unix: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_stage_contract(self) -> "CalibrationManifest":
        required_count = STAGE_RECORD_COUNTS[self.stage]
        if not self.dev_test and self.target_records != required_count:
            raise ValueError(f"{self.stage} requires exactly {required_count} target records")
        if not self.dev_test and self.final_records > self.target_records:
            raise ValueError("calibration final_records cannot exceed the stage target")
        prior = STAGE_PRIOR_REQUIREMENTS[self.stage]
        has_prior = bool(self.prior_decision_path and self.prior_decision_sha256)
        if prior is None and has_prior:
            raise ValueError("calibration_200 must not bind a prior decision")
        if prior is not None and not has_prior and not self.dev_test:
            raise ValueError(f"{self.stage} requires a prior-stage decision")
        if bool(self.prior_decision_path) != bool(self.prior_decision_sha256):
            raise ValueError("prior decision path and SHA-256 must be supplied together")
        return self


class CalibrationDecision(StrictModel):
    schema_version: Literal[2] = 2
    calibration_id: str
    stage: CalibrationStage
    status: Literal["passed", "blocked", "failed"]
    dev_test: bool = False
    manifest_path: str
    manifest_sha256: str
    source_registry_sha256: str
    trust_policy_sha256: str
    reviewer_roster_sha256: str
    reviewer_qualification_sha256: str
    legal_evidence_sha256: str
    protected_manifest_sha256: str
    authority_evidence_sha256: str
    prior_decision_sha256: str | None = None
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    issues: list[str] = Field(default_factory=list)
    next_stage: CalibrationNextStage
    created_at_unix: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_decision_contract(self) -> "CalibrationDecision":
        for name in (
            "manifest_sha256",
            "source_registry_sha256",
            "trust_policy_sha256",
            "reviewer_roster_sha256",
            "reviewer_qualification_sha256",
            "legal_evidence_sha256",
            "protected_manifest_sha256",
            "authority_evidence_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.prior_decision_sha256 is not None and (
            len(self.prior_decision_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.prior_decision_sha256)
        ):
            raise ValueError("prior_decision_sha256 must be a lowercase SHA-256")
        if self.dev_test and self.next_stage != "repeat_current_stage":
            raise ValueError("dev-test decisions cannot unlock governed advancement")
        return self


def _write_json_atomic(path: str | Path, payload: StrictModel | dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    data = payload.model_dump(mode="json") if isinstance(payload, StrictModel) else payload
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _fresh_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"calibration work directory must be new and empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _registry_sha256(registry: SourceRegistry) -> str:
    payload = json.dumps(
        [source.model_dump(mode="json") for source in sorted(registry.to_sources(), key=lambda item: item.source_id or "")],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legal_evidence_sha256(path: str | Path) -> str:
    root = Path(path).resolve()
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name in {"approval.json", "license.txt"}
    ) if root.is_dir() else []
    payload = [
        {
            "path": file.relative_to(root).as_posix(),
            "sha256": _sha256_file(file),
        }
        for file in files
    ]
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _validate_reviewer_qualification(
    report_path: str | Path,
    reviewer_roster_path: str | Path,
) -> ReviewerQualificationEvaluationReport:
    source = Path(report_path).expanduser().resolve(strict=True)
    report = ReviewerQualificationEvaluationReport.model_validate_json(
        source.read_text(encoding="utf-8-sig")
    )
    roster = load_reviewer_roster(reviewer_roster_path)
    if report.status != "passed" or report.blockers:
        raise ValueError("reviewer qualification report did not pass")
    if report.roster_id != roster.roster_id:
        raise ValueError("reviewer qualification roster_id does not match the active roster")
    if report.roster_sha256 != _sha256_file(reviewer_roster_path):
        raise ValueError("reviewer qualification report is not bound to the active roster")
    active_reviewers = {
        identity.reviewer_id
        for identity in roster.identities
        if identity.active and "reviewer" in identity.roles
    }
    evaluated_reviewers = {item.reviewer_id for item in report.reviewers}
    if evaluated_reviewers != active_reviewers:
        raise ValueError("reviewer qualification identities do not match the active reviewers")
    if any(
        not item.passed or item.accuracy < report.minimum_accuracy
        for item in report.reviewers
    ):
        raise ValueError("reviewer qualification accuracy threshold was not met")
    if report.reviewer_agreement_kappa < report.minimum_kappa:
        raise ValueError("reviewer qualification agreement threshold was not met")
    return report


def _authority_evidence(manifest: CalibrationManifest) -> list[dict[str, Any]]:
    """Return only authority records bound to this calibration manifest."""

    database = Path(manifest.authority_database).resolve(strict=True)
    uri = f"{database.as_uri()}?mode=ro"
    evidence: list[dict[str, Any]] = []
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for binding in sorted(manifest.review_bindings, key=lambda item: item.review_item_id):
            item = connection.execute(
                """
                SELECT id,content_hash,source_snapshot_sha256,source_id,data_type,
                       high_value,status,version
                FROM dataset_review_items WHERE id=?
                """,
                (binding.review_item_id,),
            ).fetchone()
            if item is None:
                raise ValueError(f"authority review item is missing: {binding.review_item_id}")
            decisions = [
                {
                    key: row[key]
                    for key in (
                        "reviewer_id",
                        "decision",
                        "rationale",
                        "content_hash",
                        "source_snapshot_sha256",
                        "evidence_json",
                    )
                }
                for row in connection.execute(
                    """
                    SELECT reviewer_id,decision,rationale,content_hash,
                           source_snapshot_sha256,evidence_json
                    FROM dataset_review_decisions
                    WHERE review_item_id=?
                    ORDER BY reviewer_id
                    """,
                    (binding.review_item_id,),
                )
            ]
            adjudication_row = connection.execute(
                """
                SELECT adjudicator_id,decision,rationale,evidence_json
                FROM dataset_review_adjudications WHERE review_item_id=?
                """,
                (binding.review_item_id,),
            ).fetchone()
            snapshot = connection.execute(
                """
                SELECT source_id,source_family,immutable_revision,registry_sha256,
                       license_evidence_sha256,legal_approval_sha256,snapshot_sha256,status
                FROM data_source_snapshots WHERE snapshot_sha256=?
                """,
                (binding.source_snapshot_sha256,),
            ).fetchone()
            evidence.append(
                {
                    "binding": binding.model_dump(mode="json"),
                    "item": dict(item),
                    "decisions": decisions,
                    "adjudication": dict(adjudication_row) if adjudication_row is not None else None,
                    "source_snapshot": dict(snapshot) if snapshot is not None else None,
                }
            )
    return evidence


def _authority_evidence_sha256(manifest: CalibrationManifest) -> str:
    evidence = _authority_evidence(manifest)
    return stable_hash(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _replay_manifest_evidence(
    manifest: CalibrationManifest,
    policy: DatasetTrustPolicyContract,
) -> tuple[dict[str, bool], dict[str, float | int], list[str], str]:
    """Recalculate stage gates from immutable rows and authority records."""

    issues: list[str] = []
    checks: dict[str, bool] = {}
    for name, path in manifest.artifacts.items():
        checks[f"artifact_{name}_untampered"] = (
            Path(path).is_file() and _sha256_file(path) == manifest.artifact_sha256.get(name)
        )
    rows_path = manifest.artifacts.get("calibration_rows")
    rows = list(iter_jsonl(rows_path)) if rows_path and Path(rows_path).is_file() else []
    row_hashes = {
        str(row.get("content_hash") or stable_hash(_row_text(row)))
        for row in rows
    }
    high_value_types = set(policy.high_value_data_types)
    high_value_hashes = {
        str(row.get("content_hash") or stable_hash(_row_text(row)))
        for row in rows
        if _row_data_type(row) in high_value_types
    }
    binding_by_id = {binding.review_item_id: binding for binding in manifest.review_bindings}
    bound_hashes = {binding.content_hash for binding in manifest.review_bindings}
    bound_routine = sum(not binding.actual_high_value for binding in manifest.review_bindings)
    routine_rows = len(rows) - len(high_value_hashes)
    minimum_routine_sample = math.ceil(max(0, routine_rows) * policy.review.routine_sample_fraction)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_name = str(row.get("source") or "unknown")
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
    source_fractions = {
        source_name: round(count / max(1, len(rows)), 6)
        for source_name, count in sorted(source_counts.items())
    }
    checks.update(
        {
            "manifest_row_count_matches_artifact": len(rows) == manifest.final_records,
            "review_bindings_reference_calibration_rows": bound_hashes.issubset(row_hashes),
            "all_high_value_rows_bound_for_review": high_value_hashes.issubset(bound_hashes),
            "routine_review_sample_fraction_met": bound_routine >= minimum_routine_sample,
            "source_fractions_match_rows": source_fractions == manifest.source_fractions,
        }
    )

    evidence = _authority_evidence(manifest)
    approved = rejected = pending = paired = 0
    pairs: list[tuple[str, str]] = []
    bindings_valid = True
    adjudicator_separation_valid = True
    for record in evidence:
        item = record["item"]
        binding = binding_by_id[str(item["id"])]
        if (
            item["content_hash"] != binding.content_hash
            or item["source_snapshot_sha256"] != binding.source_snapshot_sha256
            or item["source_id"] != binding.source_id
            or item["data_type"] != binding.data_type
        ):
            bindings_valid = False
        decisions = record["decisions"]
        reviewer_ids = {str(decision["reviewer_id"]) for decision in decisions}
        decision_values = [str(decision["decision"]) for decision in decisions]
        decision_bindings_valid = all(
            decision["content_hash"] == binding.content_hash
            and decision["source_snapshot_sha256"] == binding.source_snapshot_sha256
            for decision in decisions
        )
        bindings_valid = bindings_valid and decision_bindings_valid
        if len(decision_values) >= 2 and len(reviewer_ids) >= 2:
            paired += 1
            pairs.append((decision_values[0], decision_values[1]))
        adjudication = record["adjudication"]
        if adjudication is not None and str(adjudication["adjudicator_id"]) in reviewer_ids:
            adjudicator_separation_valid = False
        if item["status"] == "approved":
            approved += 1
        elif item["status"] == "rejected":
            rejected += 1
        else:
            pending += 1

    sample_count = len(manifest.review_bindings)
    acceptance_rate = approved / max(1, approved + rejected)
    kappa = _cohen_kappa(pairs)
    maximum_source_fraction = max(manifest.source_fractions.values(), default=1.0)
    average_quality = float(manifest.quality_report.get("avg_quality_score", 0.0))
    contamination_hits = int(manifest.contamination_report.get("rejected", 0))
    metrics: dict[str, float | int] = {
        "target_records": manifest.target_records,
        "final_records": manifest.final_records,
        "review_sample_records": sample_count,
        "paired_review_records": paired,
        "approved_review_records": approved,
        "rejected_review_records": rejected,
        "pending_review_records": pending,
        "review_acceptance_rate": round(acceptance_rate, 6),
        "reviewer_agreement_kappa": round(kappa, 6),
        "average_quality_score": round(average_quality, 6),
        "maximum_source_fraction": round(maximum_source_fraction, 6),
        "protected_contamination_hits": contamination_hits,
    }
    checks.update(
        {
            "stage_record_count_exact": (
                manifest.dev_test
                or (
                    manifest.target_records == STAGE_RECORD_COUNTS[manifest.stage]
                    and manifest.final_records == STAGE_RECORD_COUNTS[manifest.stage]
                )
            ),
            "authority_review_bindings_valid": bindings_valid,
            "adjudicator_identity_separate": adjudicator_separation_valid,
            "all_sample_records_have_two_reviews": paired == sample_count and pending == 0,
            "review_acceptance_threshold": acceptance_rate >= policy.review.sampled_acceptance_minimum,
            "reviewer_agreement_threshold": kappa >= policy.review.reviewer_agreement_minimum,
            "average_quality_threshold": average_quality >= policy.promotion.minimum_average_quality,
            "source_fraction_threshold": maximum_source_fraction <= policy.source_limits.source_max_token_fraction,
            "protected_contamination_zero": contamination_hits == 0,
        }
    )
    issues.extend(f"failed replay check: {name}" for name, passed in checks.items() if not passed)
    authority_hash = stable_hash(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return checks, metrics, issues, authority_hash


def _validate_current_governance(
    manifest: CalibrationManifest,
    *,
    sources_path: str | Path,
    protected_config_path: str | Path,
    protected_manifest_path: str | Path,
    reviewer_roster_path: str | Path,
    legal_evidence_dir: str | Path,
    trust_policy_path: str | Path,
) -> list[str]:
    issues: list[str] = []
    registry = SourceRegistry.from_file(sources_path)
    try:
        registry.validate(production=True)
    except ValueError as exc:
        issues.append(f"source registry is not production-valid: {exc}")
    if _registry_sha256(registry) != manifest.source_registry_sha256:
        issues.append("source registry hash changed")
    legal_blockers = registry.verify_approval_evidence_directory(legal_evidence_dir)
    issues.extend(f"legal evidence invalid: {blocker}" for blocker in legal_blockers)
    if _legal_evidence_sha256(legal_evidence_dir) != manifest.legal_evidence_sha256:
        issues.append("legal evidence directory hash changed")
    roster = load_reviewer_roster(reviewer_roster_path)
    issues.extend(f"reviewer roster invalid: {blocker}" for blocker in roster.readiness_blockers())
    if _sha256_file(reviewer_roster_path) != manifest.reviewer_roster_sha256:
        issues.append("reviewer roster hash changed")
    try:
        _validate_reviewer_qualification(
            manifest.reviewer_qualification_path,
            reviewer_roster_path,
        )
        if (
            _sha256_file(manifest.reviewer_qualification_path)
            != manifest.reviewer_qualification_sha256
        ):
            issues.append("reviewer qualification report hash changed")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        issues.append(f"reviewer qualification invalid: {exc}")
    if _sha256_file(trust_policy_path) != manifest.trust_policy_sha256:
        issues.append("dataset trust policy hash changed")
    validate_protected_benchmark_manifest(protected_config_path, protected_manifest_path)
    if _sha256_file(protected_manifest_path) != manifest.protected_manifest_sha256:
        issues.append("protected benchmark manifest hash changed")
    return issues


def validate_advancement_decision(
    decision_path: str | Path,
    *,
    expected_stage: CalibrationStage,
    expected_next_stage: CalibrationNextStage,
    sources_path: str | Path,
    protected_config_path: str | Path,
    protected_manifest_path: str | Path,
    reviewer_roster_path: str | Path,
    legal_evidence_dir: str | Path,
    trust_policy_path: str | Path,
    _visited: set[Path] | None = None,
) -> CalibrationDecision:
    """Replay the complete immutable evidence chain for an advancement decision."""

    source = Path(decision_path).resolve(strict=True)
    visited = set() if _visited is None else _visited
    if source in visited:
        raise ValueError("calibration advancement decision chain contains a cycle")
    visited.add(source)
    decision = CalibrationDecision.model_validate_json(source.read_text(encoding="utf-8"))
    if decision.stage != expected_stage:
        raise ValueError(f"expected {expected_stage} decision, received {decision.stage}")
    if decision.status != "passed" or not decision.checks or not all(decision.checks.values()):
        raise ValueError(f"{expected_stage} decision did not pass every governed check")
    if decision.dev_test:
        raise ValueError("dev-test calibration decisions cannot authorize production advancement")
    if decision.next_stage != expected_next_stage:
        raise ValueError(
            f"{expected_stage} decision does not authorize {expected_next_stage}: {decision.next_stage}"
        )

    manifest_path = Path(decision.manifest_path).resolve(strict=True)
    if _sha256_file(manifest_path) != decision.manifest_sha256:
        raise ValueError("calibration manifest hash does not match the advancement decision")
    manifest = CalibrationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.calibration_id != decision.calibration_id or manifest.stage != decision.stage:
        raise ValueError("calibration decision identity does not match its manifest")
    required_count = STAGE_RECORD_COUNTS[manifest.stage]
    if manifest.target_records != required_count or manifest.final_records != required_count:
        raise ValueError(f"{manifest.stage} did not produce exactly {required_count} final rows")

    bound_hashes = {
        "source_registry_sha256": manifest.source_registry_sha256,
        "trust_policy_sha256": manifest.trust_policy_sha256,
        "reviewer_roster_sha256": manifest.reviewer_roster_sha256,
        "reviewer_qualification_sha256": manifest.reviewer_qualification_sha256,
        "legal_evidence_sha256": manifest.legal_evidence_sha256,
        "protected_manifest_sha256": manifest.protected_manifest_sha256,
    }
    for name, expected in bound_hashes.items():
        if getattr(decision, name) != expected:
            raise ValueError(f"decision {name} does not match its calibration manifest")
    for name, path in manifest.artifacts.items():
        artifact = Path(path)
        if not artifact.is_file() or _sha256_file(artifact) != manifest.artifact_sha256.get(name):
            raise ValueError(f"calibration artifact is missing or tampered: {name}")
    governance_issues = _validate_current_governance(
        manifest,
        sources_path=sources_path,
        protected_config_path=protected_config_path,
        protected_manifest_path=protected_manifest_path,
        reviewer_roster_path=reviewer_roster_path,
        legal_evidence_dir=legal_evidence_dir,
        trust_policy_path=trust_policy_path,
    )
    if governance_issues:
        raise ValueError("calibration governance binding is stale: " + "; ".join(governance_issues))
    policy = load_dataset_trust_policy(trust_policy_path)
    replay_checks, replay_metrics, replay_issues, replay_authority_sha256 = _replay_manifest_evidence(
        manifest,
        policy,
    )
    if replay_issues or not all(replay_checks.values()):
        raise ValueError("calibration evidence replay failed: " + "; ".join(replay_issues))
    if replay_metrics != decision.metrics:
        raise ValueError("calibration decision metrics do not match replayed evidence")
    for check_name, replayed in replay_checks.items():
        if decision.checks.get(check_name) is not replayed:
            raise ValueError(f"calibration decision check does not match replayed evidence: {check_name}")
    if replay_authority_sha256 != decision.authority_evidence_sha256:
        raise ValueError("dataset authority evidence changed after calibration finalization")

    prior_requirement = STAGE_PRIOR_REQUIREMENTS[manifest.stage]
    if prior_requirement is None:
        if manifest.prior_decision_path or manifest.prior_decision_sha256 or decision.prior_decision_sha256:
            raise ValueError("calibration_200 must not contain prior-stage evidence")
    else:
        if not manifest.prior_decision_path or not manifest.prior_decision_sha256:
            raise ValueError(f"{manifest.stage} is missing its prior-stage decision")
        prior_path = Path(manifest.prior_decision_path).resolve(strict=True)
        actual_prior_sha256 = _sha256_file(prior_path)
        if (
            actual_prior_sha256 != manifest.prior_decision_sha256
            or actual_prior_sha256 != decision.prior_decision_sha256
        ):
            raise ValueError("prior-stage decision hash does not match the advancement chain")
        validate_advancement_decision(
            prior_path,
            expected_stage=prior_requirement[0],  # type: ignore[arg-type]
            expected_next_stage=prior_requirement[1],  # type: ignore[arg-type]
            sources_path=sources_path,
            protected_config_path=protected_config_path,
            protected_manifest_path=protected_manifest_path,
            reviewer_roster_path=reviewer_roster_path,
            legal_evidence_dir=legal_evidence_dir,
            trust_policy_path=trust_policy_path,
            _visited=visited,
        )
    return decision


def preflight_calibration(
    *,
    sources_path: str | Path,
    protected_config_path: str | Path,
    protected_manifest_path: str | Path,
    reviewer_roster_path: str | Path,
    reviewer_qualification_report_path: str | Path | None,
    legal_evidence_dir: str | Path,
    approval_request_dir: str | Path,
) -> CalibrationPreflightReport:
    blockers: list[str] = []
    source_report: SourceRegistryReport | None = None
    protected: ProtectedBenchmarkManifest | None = None
    reviewer_roster: ReviewerRoster | None = None
    reviewer_qualification: ReviewerQualificationEvaluationReport | None = None
    registry = SourceRegistry.from_file(sources_path)
    registry.prepare_approval_requests(approval_request_dir)
    try:
        source_report = registry.validate()
        blockers.extend(f"source_legal_approval:{issue}" for issue in registry.production_blockers())
        blockers.extend(
            f"source_legal_evidence:{issue}"
            for issue in registry.verify_approval_evidence_directory(legal_evidence_dir)
        )
    except (ValueError, FileNotFoundError) as exc:
        blockers.append(f"source_legal_approval:{exc}")
    try:
        protected = validate_protected_benchmark_manifest(protected_config_path, protected_manifest_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        blockers.append(f"protected_benchmarks:{exc}")
    try:
        reviewer_roster = load_reviewer_roster(reviewer_roster_path)
        blockers.extend(f"reviewer_roster:{issue}" for issue in reviewer_roster.readiness_blockers())
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        blockers.append(f"reviewer_roster:{exc}")
    if reviewer_qualification_report_path is None:
        blockers.append("reviewer_qualification:passed qualification report is required")
    else:
        try:
            reviewer_qualification = _validate_reviewer_qualification(
                reviewer_qualification_report_path,
                reviewer_roster_path,
            )
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            blockers.append(f"reviewer_qualification:{exc}")
    return CalibrationPreflightReport(
        status="blocked" if blockers else "ready",
        source_registry=source_report.model_dump(mode="json") if source_report else None,
        protected_benchmark_manifest=protected.model_dump(mode="json") if protected else None,
        reviewer_roster=reviewer_roster.model_dump(mode="json") if reviewer_roster else None,
        reviewer_qualification=(
            {
                "status": reviewer_qualification.status,
                "roster_id": reviewer_qualification.roster_id,
                "minimum_accuracy": reviewer_qualification.minimum_accuracy,
                "minimum_kappa": reviewer_qualification.minimum_kappa,
                "reviewer_agreement_kappa": reviewer_qualification.reviewer_agreement_kappa,
                "reviewer_count": len(reviewer_qualification.reviewers),
            }
            if reviewer_qualification
            else None
        ),
        approval_request_dir=str(Path(approval_request_dir).resolve()),
        legal_evidence_dir=str(Path(legal_evidence_dir).resolve()),
        blockers=blockers,
    )


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or "")


def _row_data_type(row: dict[str, Any]) -> str:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    return str(quality.get("data_type") or row.get("category") or "unknown")


def _source_snapshot_sha256(source_id: str, revision: str, content_hashes: list[str]) -> str:
    payload = json.dumps(
        {
            "source_id": source_id,
            "immutable_revision": revision,
            "content_hashes": sorted(content_hashes),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_calibration_rows(input_path: Path, output_path: Path, target_records: int) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(input_path))
    rows.sort(key=lambda row: (str(row.get("source") or ""), str(row.get("content_hash") or "")))
    selected = rows[:target_records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return selected


async def _bind_review_sample(
    *,
    rows: list[dict[str, Any]],
    registry: SourceRegistry,
    authority: SQLiteDatasetAuthorityStore,
    calibration_id: str,
    policy: DatasetTrustPolicyContract,
    minimum_sample_size: int,
    seed: int,
) -> list[CalibrationReviewBinding]:
    sources_by_name = {source.name: source for source in registry.to_sources()}
    sources_by_id = {source.source_id: source for source in registry.to_sources()}
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_key = str(row.get("source") or "")
        source = sources_by_name.get(source_key) or sources_by_id.get(source_key)
        if source is None or source.source_id is None:
            raise ValueError(f"calibration row references unknown source: {source_key!r}")
        rows_by_source.setdefault(source.source_id, []).append(row)

    snapshot_by_source: dict[str, str] = {}
    for source_id, source_rows in sorted(rows_by_source.items()):
        source = sources_by_id[source_id]
        assert source.source_family is not None
        assert source.license_evidence_sha256 is not None
        assert source.legal_approval_sha256 is not None
        content_hashes = [
            str(row.get("content_hash") or stable_hash(_row_text(row)))
            for row in source_rows
        ]
        snapshot_sha256 = _source_snapshot_sha256(source_id, source.immutable_revision, content_hashes)
        await authority.record_source_snapshot(
            SourceSnapshotCreate(
                source_id=source_id,
                source_family=source.source_family,
                immutable_revision=source.immutable_revision,
                registry_sha256=source_registry_entry_sha256(source),
                license_evidence_sha256=source.license_evidence_sha256,
                legal_approval_sha256=source.legal_approval_sha256,
                snapshot_sha256=snapshot_sha256,
                status="approved",
                metadata={
                    "calibration_id": calibration_id,
                    "record_count": len(source_rows),
                    "content_hashes_sha256": stable_hash(":".join(sorted(content_hashes))),
                },
            )
        )
        snapshot_by_source[source_id] = snapshot_sha256

    high_value_types = set(policy.high_value_data_types)
    high_value: list[dict[str, Any]] = []
    routine: list[dict[str, Any]] = []
    for row in rows:
        (high_value if _row_data_type(row) in high_value_types else routine).append(row)
    routine_target = max(
        minimum_sample_size - len(high_value),
        math.ceil(len(routine) * policy.review.routine_sample_fraction),
        0,
    )
    rng = random.Random(seed)
    routine.sort(key=lambda row: stable_hash(f"{seed}:{row.get('content_hash') or _row_text(row)}"))
    rng.shuffle(routine)
    selected = high_value + routine[: min(len(routine), routine_target)]
    selected.sort(key=lambda row: str(row.get("content_hash") or stable_hash(_row_text(row))))

    bindings: list[CalibrationReviewBinding] = []
    for row in selected:
        source_key = str(row.get("source") or "")
        source = sources_by_name.get(source_key) or sources_by_id.get(source_key)
        assert source is not None and source.source_id is not None
        text = _row_text(row)
        content_hash = str(row.get("content_hash") or stable_hash(text))
        data_type = _row_data_type(row)
        actual_high_value = data_type in high_value_types
        item = await authority.enqueue(
            ReviewItemCreate(
                content_hash=content_hash,
                source_snapshot_sha256=snapshot_by_source[source.source_id],
                source_id=source.source_id,
                data_type=data_type,
                # Calibration samples always require two independent decisions.
                high_value=True,
                payload={
                    "calibration_id": calibration_id,
                    "actual_high_value": actual_high_value,
                    "text": text,
                    "url": row.get("url"),
                    "license": row.get("license"),
                    "quality": row.get("quality"),
                },
            )
        )
        bindings.append(
            CalibrationReviewBinding(
                review_item_id=item.review_item_id,
                content_hash=content_hash,
                source_id=source.source_id,
                source_snapshot_sha256=snapshot_by_source[source.source_id],
                data_type=data_type,
                actual_high_value=actual_high_value,
            )
        )
    return bindings


async def run_calibration(
    *,
    stage: CalibrationStage,
    sources_path: str | Path,
    protected_config_path: str | Path,
    protected_manifest_path: str | Path,
    reviewer_roster_path: str | Path,
    reviewer_qualification_report_path: str | Path | None,
    legal_evidence_dir: str | Path,
    trust_policy_path: str | Path,
    work_dir: str | Path,
    authority_database: str | Path,
    prior_decision_path: str | Path | None = None,
    dev_test: bool = False,
    dev_test_target_records: int | None = None,
    crawl_multiplier: int = 2,
    minimum_review_sample: int = 30,
    workers: int = 6,
    max_depth: int = 2,
    max_bytes_per_doc: int = 250_000,
    delay_seconds: float = 0.35,
    seed: int = 1337,
    progress_to_stdout: bool = True,
    client: httpx.AsyncClient | None = None,
) -> CalibrationPreflightReport | CalibrationManifest:
    if dev_test_target_records is not None and not dev_test:
        raise ValueError("custom calibration counts require explicit dev-test mode")
    target_records = dev_test_target_records if dev_test_target_records is not None else STAGE_RECORD_COUNTS[stage]
    if target_records <= 0:
        raise ValueError("calibration target must be positive")
    prior_requirement = STAGE_PRIOR_REQUIREMENTS[stage]
    prior_sha256: str | None = None
    resolved_prior: Path | None = None
    if stage == "calibration_200" and prior_decision_path is not None:
        raise ValueError("calibration_200 cannot accept a prior decision")
    if prior_requirement is not None and not dev_test:
        if prior_decision_path is None:
            raise ValueError(f"{stage} requires a passed calibration_200 decision")
        resolved_prior = Path(prior_decision_path).resolve(strict=True)
        validate_advancement_decision(
            resolved_prior,
            expected_stage="calibration_200",
            expected_next_stage="5k_calibration_allowed",
            sources_path=sources_path,
            protected_config_path=protected_config_path,
            protected_manifest_path=protected_manifest_path,
            reviewer_roster_path=reviewer_roster_path,
            legal_evidence_dir=legal_evidence_dir,
            trust_policy_path=trust_policy_path,
        )
        prior_sha256 = _sha256_file(resolved_prior)
    elif prior_decision_path is not None:
        resolved_prior = Path(prior_decision_path).resolve(strict=True)
        prior_sha256 = _sha256_file(resolved_prior)

    root = _fresh_directory(work_dir)
    approval_dir = root / "legal-approval-requests"
    preflight = preflight_calibration(
        sources_path=sources_path,
        protected_config_path=protected_config_path,
        protected_manifest_path=protected_manifest_path,
        reviewer_roster_path=reviewer_roster_path,
        reviewer_qualification_report_path=reviewer_qualification_report_path,
        legal_evidence_dir=legal_evidence_dir,
        approval_request_dir=approval_dir,
    )
    _write_json_atomic(root / "calibration_preflight_report.json", preflight)
    if preflight.status != "ready":
        return preflight
    assert reviewer_qualification_report_path is not None
    reviewer_qualification_path = Path(reviewer_qualification_report_path).expanduser().resolve(strict=True)

    registry = SourceRegistry.from_file(sources_path)
    registry_report = registry.validate(production=True)
    policy = load_dataset_trust_policy(trust_policy_path)
    protected_manifest_file = Path(protected_manifest_path).resolve()
    protected_manifest = validate_protected_benchmark_manifest(protected_config_path, protected_manifest_file)
    protected_index = (protected_manifest_file.parent / protected_manifest.fingerprint_index_path).resolve()
    progress: ProgressReporter = progress_from_options(
        path=str(root / "progress.jsonl"),
        to_stdout=progress_to_stdout,
    )
    frontier = FrontierStore(root / "frontier.sqlite3")
    engine = DataEngine(
        DataEngineConfig(
            frontier_path=str(root / "frontier.sqlite3"),
            output_dir=str(root / "raw"),
            clean_output_dir=str(root / "clean"),
            max_docs=target_records * crawl_multiplier,
            max_bytes_per_doc=max_bytes_per_doc,
            max_depth=max_depth,
            workers=workers,
            shard_rows=max(target_records, 100),
            respect_robots=True,
            delay_seconds=delay_seconds,
        ),
        store=frontier,
        owns_store=True,
    )
    try:
        crawl_report = await engine.run(
            registry.to_sources(),
            client=client,
            progress=progress,
            progress_every_docs=max(1, target_records // 10),
        )
    finally:
        await engine.aclose()
        progress.close()
    clean_files = sorted((root / "clean").glob("clean-*.jsonl"))
    if not clean_files:
        raise RuntimeError("calibration crawl produced no quality-accepted records")

    license_path = root / "filtered" / "license.jsonl"
    license_report: LicenseFilterReport = filter_jsonl_by_license(
        clean_files,
        license_path,
        strict_unknown=True,
    )
    protected_clean_path = root / "filtered" / "protected-clean.jsonl"
    contamination_report: BenchmarkContaminationFilterReport = filter_benchmark_contamination_jsonl(
        [license_path],
        protected_clean_path,
        protected_index_path=protected_index,
    )
    dedup_path = root / "dedup" / "dedup-clean.jsonl"
    dedup_report: NearDedupReport = deduplicate_jsonl(
        [protected_clean_path],
        dedup_path,
        index_path=root / "dedup" / "fingerprints.sqlite3",
        hamming_threshold=3,
    )
    calibration_rows_path = root / "calibration" / f"{stage}.jsonl"
    rows = _write_calibration_rows(dedup_path, calibration_rows_path, target_records)
    if len(rows) != target_records:
        raise RuntimeError(
            f"{stage} requires exactly {target_records} final rows, but only {len(rows)} survived governance gates"
        )
    quality_report: QualityInspectionReport = write_quality_report(
        [calibration_rows_path],
        root / "reports" / "quality_report.json",
    )
    calibration_id = f"cal-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    authority_path = Path(authority_database).resolve()
    authority = SQLiteDatasetAuthorityStore(authority_path)
    bindings = await _bind_review_sample(
        rows=rows,
        registry=registry,
        authority=authority,
        calibration_id=calibration_id,
        policy=policy,
        minimum_sample_size=minimum_review_sample,
        seed=seed,
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    source_fractions = {
        source: round(count / max(1, len(rows)), 6)
        for source, count in sorted(source_counts.items())
    }
    artifacts = {
        "calibration_rows": str(calibration_rows_path),
        "quality_report": str(root / "reports" / "quality_report.json"),
        "preflight_report": str(root / "calibration_preflight_report.json"),
        "progress_log": str(root / "progress.jsonl"),
    }
    manifest = CalibrationManifest(
        calibration_id=calibration_id,
        stage=stage,
        status="awaiting_review",
        dev_test=dev_test,
        target_records=target_records,
        final_records=len(rows),
        prior_decision_path=str(resolved_prior) if resolved_prior is not None else None,
        prior_decision_sha256=prior_sha256,
        source_registry_sha256=_registry_sha256(registry),
        trust_policy_sha256=_sha256_file(trust_policy_path),
        reviewer_roster_sha256=_sha256_file(reviewer_roster_path),
        reviewer_qualification_path=str(reviewer_qualification_path),
        reviewer_qualification_sha256=_sha256_file(reviewer_qualification_path),
        legal_evidence_sha256=_legal_evidence_sha256(legal_evidence_dir),
        protected_manifest_sha256=_sha256_file(protected_manifest_file),
        authority_database=str(authority_path),
        artifacts=artifacts,
        artifact_sha256={name: _sha256_file(path) for name, path in artifacts.items()},
        source_fractions=source_fractions,
        review_bindings=bindings,
        crawl_report=crawl_report.model_dump(mode="json"),
        license_report=license_report.model_dump(mode="json"),
        contamination_report=contamination_report.model_dump(mode="json"),
        dedup_report=dedup_report.model_dump(mode="json"),
        quality_report=quality_report.model_dump(mode="json"),
    )
    _write_json_atomic(root / "calibration_manifest.json", manifest)
    return manifest


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_approve = sum(left == "approve" for left, _ in pairs) / len(pairs)
    right_approve = sum(right == "approve" for _, right in pairs) / len(pairs)
    expected = left_approve * right_approve + (1 - left_approve) * (1 - right_approve)
    return 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)


async def finalize_calibration(
    manifest_path: str | Path,
    *,
    sources_path: str | Path,
    protected_config_path: str | Path,
    protected_manifest_path: str | Path,
    reviewer_roster_path: str | Path,
    legal_evidence_dir: str | Path,
    trust_policy_path: str | Path,
    output_path: str | Path | None = None,
) -> CalibrationDecision:
    manifest_file = Path(manifest_path).resolve()
    manifest = CalibrationManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    policy = load_dataset_trust_policy(trust_policy_path)
    roster = load_reviewer_roster(reviewer_roster_path)
    issues: list[str] = []
    checks: dict[str, bool] = {}
    registry = SourceRegistry.from_file(sources_path)
    try:
        registry.validate(production=True)
        checks["source_contracts_unchanged"] = _registry_sha256(registry) == manifest.source_registry_sha256
    except ValueError as exc:
        checks["source_contracts_unchanged"] = False
        issues.append(f"source registry is no longer production-valid: {exc}")
    legal_blockers = registry.verify_approval_evidence_directory(legal_evidence_dir)
    checks["legal_evidence_valid"] = not legal_blockers
    checks["legal_evidence_unchanged"] = (
        _legal_evidence_sha256(legal_evidence_dir) == manifest.legal_evidence_sha256
    )
    issues.extend(f"legal evidence invalid: {blocker}" for blocker in legal_blockers)
    checks["trust_policy_unchanged"] = _sha256_file(trust_policy_path) == manifest.trust_policy_sha256
    checks["reviewer_roster_ready"] = not roster.readiness_blockers()
    checks["reviewer_roster_unchanged"] = (
        _sha256_file(reviewer_roster_path) == manifest.reviewer_roster_sha256
    )
    try:
        _validate_reviewer_qualification(
            manifest.reviewer_qualification_path,
            reviewer_roster_path,
        )
        checks["reviewer_qualification_valid"] = True
        checks["reviewer_qualification_unchanged"] = (
            _sha256_file(manifest.reviewer_qualification_path)
            == manifest.reviewer_qualification_sha256
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        checks["reviewer_qualification_valid"] = False
        checks["reviewer_qualification_unchanged"] = False
        issues.append(f"reviewer qualification is invalid: {exc}")
    try:
        validate_protected_benchmark_manifest(protected_config_path, protected_manifest_path)
        checks["protected_holdout_unchanged"] = (
            _sha256_file(protected_manifest_path) == manifest.protected_manifest_sha256
        )
    except (ValueError, FileNotFoundError) as exc:
        checks["protected_holdout_unchanged"] = False
        issues.append(f"protected benchmark pack is invalid: {exc}")
    for name, path in manifest.artifacts.items():
        valid = Path(path).is_file() and _sha256_file(path) == manifest.artifact_sha256.get(name)
        checks[f"artifact_{name}_untampered"] = valid
    calibration_rows_path = manifest.artifacts.get("calibration_rows")
    calibration_rows = (
        list(iter_jsonl(calibration_rows_path))
        if calibration_rows_path and Path(calibration_rows_path).is_file()
        else []
    )
    row_hashes = {
        str(row.get("content_hash") or stable_hash(_row_text(row)))
        for row in calibration_rows
    }
    high_value_hashes = {
        str(row.get("content_hash") or stable_hash(_row_text(row)))
        for row in calibration_rows
        if _row_data_type(row) in set(policy.high_value_data_types)
    }
    bound_hashes = {binding.content_hash for binding in manifest.review_bindings}
    bound_routine = sum(not binding.actual_high_value for binding in manifest.review_bindings)
    routine_rows = len(calibration_rows) - len(high_value_hashes)
    minimum_routine_sample = math.ceil(
        max(0, routine_rows) * policy.review.routine_sample_fraction
    )
    calculated_source_counts: dict[str, int] = {}
    for row in calibration_rows:
        source_name = str(row.get("source") or "unknown")
        calculated_source_counts[source_name] = calculated_source_counts.get(source_name, 0) + 1
    calculated_source_fractions = {
        source_name: round(count / max(1, len(calibration_rows)), 6)
        for source_name, count in sorted(calculated_source_counts.items())
    }
    checks["manifest_row_count_matches_artifact"] = (
        len(calibration_rows) == manifest.final_records
    )
    checks["review_bindings_reference_calibration_rows"] = bound_hashes.issubset(row_hashes)
    checks["all_high_value_rows_bound_for_review"] = high_value_hashes.issubset(bound_hashes)
    checks["routine_review_sample_fraction_met"] = bound_routine >= minimum_routine_sample
    checks["source_fractions_match_rows"] = calculated_source_fractions == manifest.source_fractions
    prior_requirement = STAGE_PRIOR_REQUIREMENTS[manifest.stage]
    if prior_requirement is None:
        checks["prior_stage_chain_valid"] = (
            manifest.prior_decision_path is None and manifest.prior_decision_sha256 is None
        )
    elif manifest.dev_test:
        checks["prior_stage_chain_valid"] = True
    else:
        try:
            if not manifest.prior_decision_path or not manifest.prior_decision_sha256:
                raise ValueError("prior decision is missing")
            prior_path = Path(manifest.prior_decision_path).resolve(strict=True)
            if _sha256_file(prior_path) != manifest.prior_decision_sha256:
                raise ValueError("prior decision hash changed")
            validate_advancement_decision(
                prior_path,
                expected_stage="calibration_200",
                expected_next_stage="5k_calibration_allowed",
                sources_path=sources_path,
                protected_config_path=protected_config_path,
                protected_manifest_path=protected_manifest_path,
                reviewer_roster_path=reviewer_roster_path,
                legal_evidence_dir=legal_evidence_dir,
                trust_policy_path=trust_policy_path,
            )
            checks["prior_stage_chain_valid"] = True
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            checks["prior_stage_chain_valid"] = False
            issues.append(f"prior stage evidence invalid: {exc}")

    authority = SQLiteDatasetAuthorityStore(manifest.authority_database)
    approved = rejected = pending = paired = 0
    pairs: list[tuple[str, str]] = []
    for binding in manifest.review_bindings:
        item = await authority.get_item(binding.review_item_id, reviewer_id=None)
        if (
            item.content_hash != binding.content_hash
            or item.source_snapshot_sha256 != binding.source_snapshot_sha256
        ):
            issues.append(f"review binding changed: {binding.review_item_id}")
            pending += 1
            continue
        independent_reviewers = {decision.reviewer_id for decision in item.decisions}
        decisions = [decision.decision for decision in item.decisions]
        if len(decisions) >= 2 and len(independent_reviewers) >= 2:
            paired += 1
            pairs.append((decisions[0], decisions[1]))
        if item.status == "approved":
            approved += 1
        elif item.status == "rejected":
            rejected += 1
        else:
            pending += 1
    sample_count = len(manifest.review_bindings)
    acceptance_rate = approved / max(1, approved + rejected)
    kappa = _cohen_kappa(pairs)
    maximum_source_fraction = max(manifest.source_fractions.values(), default=1.0)
    average_quality = float(manifest.quality_report.get("avg_quality_score", 0.0))
    contamination_hits = int(manifest.contamination_report.get("rejected", 0))
    metrics: dict[str, float | int] = {
        "target_records": manifest.target_records,
        "final_records": manifest.final_records,
        "review_sample_records": sample_count,
        "paired_review_records": paired,
        "approved_review_records": approved,
        "rejected_review_records": rejected,
        "pending_review_records": pending,
        "review_acceptance_rate": round(acceptance_rate, 6),
        "reviewer_agreement_kappa": round(kappa, 6),
        "average_quality_score": round(average_quality, 6),
        "maximum_source_fraction": round(maximum_source_fraction, 6),
        "protected_contamination_hits": contamination_hits,
    }
    checks.update(
        {
            "stage_record_count_exact": (
                manifest.dev_test
                or (
                    manifest.target_records == STAGE_RECORD_COUNTS[manifest.stage]
                    and manifest.final_records == STAGE_RECORD_COUNTS[manifest.stage]
                )
            ),
            "all_sample_records_have_two_reviews": paired == sample_count and pending == 0,
            "review_acceptance_threshold": acceptance_rate >= policy.review.sampled_acceptance_minimum,
            "reviewer_agreement_threshold": kappa >= policy.review.reviewer_agreement_minimum,
            "average_quality_threshold": average_quality >= policy.promotion.minimum_average_quality,
            "source_fraction_threshold": maximum_source_fraction <= policy.source_limits.source_max_token_fraction,
            "protected_contamination_zero": contamination_hits == 0,
        }
    )
    replay_checks, replay_metrics, replay_issues, authority_evidence_sha256 = (
        _replay_manifest_evidence(manifest, policy)
    )
    checks.update(replay_checks)
    metrics = replay_metrics
    issues.extend(issue for issue in replay_issues if issue not in issues)
    for check, passed in checks.items():
        if not passed and not any(check in issue for issue in issues):
            issues.append(f"failed check: {check}")
    passed = all(checks.values())
    status: Literal["passed", "blocked", "failed"]
    status = "passed" if passed else ("blocked" if pending > 0 else "failed")
    if passed and not manifest.dev_test:
        next_stage: CalibrationNextStage = (
            "5k_calibration_allowed"
            if manifest.stage == "calibration_200"
            else "100k_dataset_build_allowed"
        )
    else:
        next_stage = "repeat_current_stage"
    decision = CalibrationDecision(
        calibration_id=manifest.calibration_id,
        stage=manifest.stage,
        status=status,
        dev_test=manifest.dev_test,
        manifest_path=str(manifest_file),
        manifest_sha256=_sha256_file(manifest_file),
        source_registry_sha256=manifest.source_registry_sha256,
        trust_policy_sha256=manifest.trust_policy_sha256,
        reviewer_roster_sha256=manifest.reviewer_roster_sha256,
        reviewer_qualification_sha256=manifest.reviewer_qualification_sha256,
        legal_evidence_sha256=manifest.legal_evidence_sha256,
        protected_manifest_sha256=manifest.protected_manifest_sha256,
        authority_evidence_sha256=authority_evidence_sha256,
        prior_decision_sha256=manifest.prior_decision_sha256,
        checks=checks,
        metrics=metrics,
        issues=issues,
        next_stage=next_stage,
    )
    target = Path(output_path).resolve() if output_path else manifest_file.parent / "calibration_decision.json"
    _write_json_atomic(target, decision)
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the governed Craftly 200 -> 5k calibration ladder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--sources", default="config/data_sources.ultimate.json")
        command.add_argument("--protected-config", default="config/protected_benchmarks.json")
        command.add_argument("--protected-manifest", default="data/eval/protected/protected_benchmark_manifest.json")
        command.add_argument("--reviewer-roster", default="config/data_reviewers.json")
        command.add_argument(
            "--reviewer-qualification-report",
            default=None,
            help="Passed, immutable reviewer qualification evaluation report.",
        )
        command.add_argument("--legal-evidence-dir", default="governance/source-approvals")
    prepare = subparsers.choices["prepare"]
    prepare.add_argument("--output-dir", default="artifacts/craftly/calibration-preflight")
    run = subparsers.choices["run"]
    run.add_argument("--trust-policy", default="config/dataset_trust_policy.json")
    run.add_argument("--work-dir", required=True)
    run.add_argument("--authority-db", default="artifacts/craftly/dataset-authority.sqlite3")
    run.add_argument("--stage", choices=sorted(STAGE_RECORD_COUNTS), required=True)
    run.add_argument("--prior-decision")
    run.add_argument("--dev-test", action="store_true")
    run.add_argument("--dev-test-target-records", type=int)
    run.add_argument("--crawl-multiplier", type=int, default=2)
    run.add_argument("--minimum-review-sample", type=int, default=30)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--max-depth", type=int, default=2)
    run.add_argument("--max-bytes-per-doc", type=int, default=250_000)
    run.add_argument("--delay-seconds", type=float, default=0.35)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--no-live-progress", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--sources", default="config/data_sources.ultimate.json")
    finalize.add_argument("--protected-config", default="config/protected_benchmarks.json")
    finalize.add_argument("--protected-manifest", default="data/eval/protected/protected_benchmark_manifest.json")
    finalize.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    finalize.add_argument("--legal-evidence-dir", default="governance/source-approvals")
    finalize.add_argument("--trust-policy", default="config/dataset_trust_policy.json")
    finalize.add_argument("--output")
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> StrictModel:
    if args.command == "prepare":
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        report = preflight_calibration(
            sources_path=args.sources,
            protected_config_path=args.protected_config,
            protected_manifest_path=args.protected_manifest,
            reviewer_roster_path=args.reviewer_roster,
            reviewer_qualification_report_path=args.reviewer_qualification_report,
            legal_evidence_dir=args.legal_evidence_dir,
            approval_request_dir=output / "legal-approval-requests",
        )
        _write_json_atomic(output / "calibration_preflight_report.json", report)
        return report
    if args.command == "run":
        return await run_calibration(
            stage=args.stage,
            sources_path=args.sources,
            protected_config_path=args.protected_config,
            protected_manifest_path=args.protected_manifest,
            reviewer_roster_path=args.reviewer_roster,
            reviewer_qualification_report_path=args.reviewer_qualification_report,
            legal_evidence_dir=args.legal_evidence_dir,
            trust_policy_path=args.trust_policy,
            work_dir=args.work_dir,
            authority_database=args.authority_db,
            prior_decision_path=args.prior_decision,
            dev_test=args.dev_test,
            dev_test_target_records=args.dev_test_target_records,
            crawl_multiplier=args.crawl_multiplier,
            minimum_review_sample=args.minimum_review_sample,
            workers=args.workers,
            max_depth=args.max_depth,
            max_bytes_per_doc=args.max_bytes_per_doc,
            delay_seconds=args.delay_seconds,
            seed=args.seed,
            progress_to_stdout=not args.no_live_progress,
        )
    return await finalize_calibration(
        args.manifest,
        sources_path=args.sources,
        protected_config_path=args.protected_config,
        protected_manifest_path=args.protected_manifest,
        reviewer_roster_path=args.reviewer_roster,
        legal_evidence_dir=args.legal_evidence_dir,
        trust_policy_path=args.trust_policy,
        output_path=args.output,
    )


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_main_async(args))
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if getattr(report, "status", "") in {"blocked", "failed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
