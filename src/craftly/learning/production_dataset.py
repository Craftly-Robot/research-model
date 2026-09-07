"""Production dataset promotion pack for Craftly scratch training.

This command is intentionally strict. It can prove the data pipeline on a tiny
fixture with ``--dev-smoke``, but production mode requires real promoted row
counts, verified patch/task rows, benchmark-holdout separation, and human review
evidence before it returns success.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import math
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from src.craftly.learning.benchmark_contamination_filter import (
    BenchmarkContaminationFilterReport,
    build_protected_fingerprint_index,
    filter_benchmark_contamination_jsonl,
)
from src.craftly.learning.calibration_gate import (
    CalibrationDecision,
    validate_advancement_decision,
)
from src.craftly.learning.dataset_validation import DatasetValidationConfig, DatasetValidationReport, validate_dataset
from src.craftly.learning.license_filter import LicenseFilterReport, filter_jsonl_by_license
from src.craftly.learning.near_dedup import NearDedupReport, deduplicate_jsonl
from src.craftly.learning.quality import DatasetQualityGate, QualityGateConfig, QualityGateReport, iter_jsonl, stable_hash
from src.craftly.learning.source_budget import SourceBudgetPlan, write_source_budget_plan
from src.craftly.learning.source_quality import SourceQualityReport, write_source_quality_report
from src.craftly.learning.source_registry import SourceRegistry, SourceRegistryReport
from src.craftly.learning.source_reputation import SourceReputationReport, write_source_reputation_report
from src.craftly.learning.training_data_gate import (
    TrainingDataGateConfig,
    TrainingDataGateReport,
    apply_training_data_gate,
    has_independent_review,
)
from src.craftly.shared.config_contracts import DatasetTrustPolicyContract, load_dataset_trust_policy
from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file as _sha256_file


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKOUT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ProductionDatasetConfig(StrictModel):
    stage: Literal["production_100k"] = "production_100k"
    input_paths: list[str]
    output_dir: str = "data/production/craftly-corpus"
    dataset_id: str = "craftly-corpus"
    source_registry_path: str = "config/data_sources.ultimate.json"
    trust_policy_path: str = "config/dataset_trust_policy.json"
    legal_evidence_dir: str = "governance/source-approvals"
    reviewer_roster_path: str = "config/data_reviewers.json"
    protected_benchmark_config_path: str = "config/protected_benchmarks.json"
    protected_benchmark_manifest_path: str = "data/eval/protected/protected_benchmark_manifest.json"
    advancement_decision_path: str | None = None
    tokenizer_path: str | None = None
    source_review_report_path: str | None = None
    benchmark_holdout_paths: list[str] = Field(default_factory=lambda: ["data/eval/humaneval.jsonl", "data/eval/mbpp.jsonl"])
    verified_patch_paths: list[str] = Field(default_factory=list)
    human_review_approved_paths: list[str] = Field(default_factory=list)
    dev_smoke: bool = False
    min_promoted_records: int = Field(default=100_000, ge=1)
    min_verified_patch_records: int = Field(default=100, ge=0)
    min_human_review_approved_records: int = Field(default=100, ge=0)
    min_train_records: int = Field(default=90_000, ge=1)
    min_avg_chars: int = Field(default=120, ge=1)
    max_duplicate_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    train_fraction: float = Field(default=0.98, ge=0.01, le=0.999)
    val_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    test_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    eval_holdout_fraction: float = Field(default=0.02, ge=0.0, le=0.5)
    seed: int = 1337
    allow_unknown_license: bool = False
    min_quality_score: float = Field(default=0.58, ge=0.0, le=1.0)
    min_source_reputation_score: float = Field(default=0.45, ge=0.0, le=1.0)
    source_budget_target_docs: int = Field(default=100_000, ge=1)
    near_dedup_hamming_threshold: int = Field(default=3, ge=0, le=16)

    @model_validator(mode="after")
    def validate_split(self) -> "ProductionDatasetConfig":
        total = round(self.train_fraction + self.val_fraction + self.test_fraction, 6)
        if total != 1.0:
            raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
        if not self.dev_smoke and self.min_promoted_records != 100_000:
            raise ValueError("production_100k requires exactly 100,000 as the minimum promoted row count")
        return self


class SplitManifest(StrictModel):
    train_path: str
    val_path: str
    test_path: str
    holdout_path: str
    train_records: int
    val_records: int
    test_records: int
    holdout_records: int
    fractions: dict[str, float]
    seed: int
    split_groups: dict[str, int] = Field(default_factory=dict)
    group_assignments_sha256: str = ""
    cross_split_group_collisions: int = 0


class HoldoutSeparationReport(StrictModel):
    benchmark_holdout_paths: list[str]
    benchmark_hashes: int
    scanned_records: int
    removed_records: int
    output_path: str
    removed_path: str


class PatchTaskDatasetReport(StrictModel):
    input_paths: list[str]
    output_path: str
    accepted: int
    rejected: int
    verified_evidence: int = 0
    evidence_coverage: float = 0.0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    categories: dict[str, int] = Field(default_factory=dict)


class HumanReviewPromotionReport(StrictModel):
    input_paths: list[str]
    output_path: str
    approved: int
    rejected: int
    independently_reviewed: int = 0
    review_coverage: float = 0.0
    reviewer_agreement: float = Field(default=0.0, ge=-1.0, le=1.0)
    categories: dict[str, int] = Field(default_factory=dict)


class DatasetTrustMetrics(StrictModel):
    records: int
    policy_tokens: int
    average_quality: float = Field(ge=0.0, le=1.0)
    p10_quality: float = Field(ge=0.0, le=1.0)
    license_coverage: float = Field(ge=0.0, le=1.0)
    provenance_coverage: float = Field(ge=0.0, le=1.0)
    high_value_records: int
    high_value_review_coverage: float = Field(ge=0.0, le=1.0)
    verified_patch_records: int
    verified_patch_evidence_coverage: float = Field(ge=0.0, le=1.0)
    secret_or_pii_hits: int
    source_token_fractions: dict[str, float]
    source_family_token_fractions: dict[str, float]
    maximum_source_token_fraction: float = Field(ge=0.0, le=1.0)
    maximum_source_family_token_fraction: float = Field(ge=0.0, le=1.0)
    token_count_method: str


class ProductionDatasetManifest(StrictModel):
    dataset_id: str
    version_id: str
    status: str
    output_dir: str
    created_at_unix: float = Field(default_factory=time.time)
    dev_smoke: bool
    artifacts: dict[str, str]
    metrics: dict[str, Any]
    issues: list[str] = Field(default_factory=list)
    reports: dict[str, Any]
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    policy_sha256: str = ""
    tokenizer_sha256: str | None = None
    advancement_decision_path: str | None = None
    advancement_decision_sha256: str | None = None
    governance_paths: dict[str, str] = Field(default_factory=dict)
    promotion_decision: dict[str, Any] = Field(default_factory=dict)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "dataset_version_manifest.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "dataset_version_manifest.md")
        return target


class VerifiedPatchEvidence(StrictModel):
    status: Literal["passed"]
    vulnerable_checkout_hash: str
    vulnerable_test_failed: bool
    patch_applied: bool
    build_passed: bool
    security_test_passed: bool
    regression_tests_passed: bool
    static_scan_passed: bool
    manifest_sha256: str

    @model_validator(mode="after")
    def validate_contract(self) -> "VerifiedPatchEvidence":
        checkout = self.vulnerable_checkout_hash.strip().lower()
        manifest = self.manifest_sha256.strip().lower()
        if CHECKOUT_RE.fullmatch(checkout) is None:
            raise ValueError("vulnerable_checkout_hash must be a full Git SHA-1 or SHA-256")
        if SHA256_RE.fullmatch(manifest) is None:
            raise ValueError("verification manifest_sha256 must be SHA-256 hex")
        object.__setattr__(self, "vulnerable_checkout_hash", checkout)
        object.__setattr__(self, "manifest_sha256", manifest)
        required = {
            "vulnerable_test_failed": self.vulnerable_test_failed,
            "patch_applied": self.patch_applied,
            "build_passed": self.build_passed,
            "security_test_passed": self.security_test_passed,
            "regression_tests_passed": self.regression_tests_passed,
            "static_scan_passed": self.static_scan_passed,
        }
        failed = [name for name, passed in required.items() if not passed]
        if failed:
            raise ValueError("verified patch evidence has failed gates: " + ", ".join(failed))
        return self


class DatasetPromotionDecision(StrictModel):
    status: Literal["promoted", "rejected"]
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    issues: list[str] = Field(default_factory=list)
    policy_id: str
    created_at_unix: float = Field(default_factory=time.time)


def _write_json(path: str | Path, payload: StrictModel | dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, StrictModel):
        data = payload.model_dump()
    else:
        data = payload
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _hash_existing_artifacts(artifacts: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role, value in sorted(artifacts.items()):
        path = Path(value)
        if path.is_file():
            hashes[role] = _sha256_file(path)
    return hashes


def _expand_input_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in paths:
        if any(marker in item for marker in ("*", "?", "[")):
            matches = sorted(glob.glob(item))
            expanded.extend(str(Path(path)) for path in matches if Path(path).is_file())
        else:
            expanded.append(item)
    return expanded


def _row_text(row: dict[str, Any]) -> str:
    if row.get("prompt") and row.get("chosen"):
        return f"{row['prompt']}\n{row['chosen']}"
    return str(row.get("text") or row.get("content") or row.get("prompt") or row.get("chosen") or "")


def _row_hash(row: dict[str, Any]) -> str:
    return str(row.get("content_hash") or stable_hash(_row_text(row)))


POLICY_TOKEN_RE = re.compile(
    r"""0x[0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]""",
    re.VERBOSE,
)


def _policy_token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids)
    return len(POLICY_TOKEN_RE.findall(text))


def _load_policy_tokenizer(path: str | None) -> tuple[Any | None, str]:
    if not path:
        return None, "code_aware_lexical_v1"
    source = Path(path).resolve(strict=True)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(source))
    return tokenizer, f"huggingface_tokenizers:{_sha256_file(source)}"


def _quality_value(row: dict[str, Any]) -> float:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    value = quality.get("quality_score", 0.0)
    return max(0.0, min(1.0, float(value))) if isinstance(value, (int, float)) else 0.0


def _source_identity(row: dict[str, Any]) -> tuple[str, str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source_id = str(row.get("source_id") or provenance.get("source_id") or row.get("source") or "unknown")
    family = str(row.get("source_family") or provenance.get("source_family") or source_id)
    return source_id, family


def _is_high_value(row: dict[str, Any], policy: DatasetTrustPolicyContract) -> bool:
    category = str(row.get("category") or "").lower()
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    data_type = str(quality.get("data_type") or row.get("data_type") or category).lower()
    labels = {str(item).lower() for item in quality.get("labels", [])}
    expected = {item.lower() for item in policy.high_value_data_types}
    return data_type in expected or category in expected or bool(labels & expected)


def _quality_percentile(histogram: list[int], total: int, percentile: float) -> float:
    if total <= 0:
        return 0.0
    target = max(1, math.ceil(total * percentile))
    running = 0
    for index, count in enumerate(histogram):
        running += count
        if running >= target:
            return index / (len(histogram) - 1)
    return 1.0


def calculate_dataset_trust_metrics(
    paths: list[str | Path],
    *,
    policy: DatasetTrustPolicyContract,
    tokenizer_path: str | None,
) -> DatasetTrustMetrics:
    tokenizer, token_method = _load_policy_tokenizer(tokenizer_path)
    records = policy_tokens = license_complete = provenance_complete = 0
    high_value = high_value_reviewed = verified_patches = verified_patch_evidence = 0
    secret_or_pii_hits = 0
    quality_sum = 0.0
    quality_histogram = [0] * 1001
    source_tokens: Counter[str] = Counter()
    family_tokens: Counter[str] = Counter()
    for path in paths:
        for row in iter_jsonl(path):
            records += 1
            text = _row_text(row)
            tokens = max(1, _policy_token_count(text, tokenizer))
            policy_tokens += tokens
            source_id, source_family = _source_identity(row)
            source_tokens[source_id] += tokens
            family_tokens[source_family] += tokens
            license_name = str(row.get("license") or row.get("spdx_license") or "").lower()
            if license_name not in {"", "unknown", "none"}:
                license_complete += 1
            if _provenance_complete(row):
                provenance_complete += 1
            quality = _quality_value(row)
            quality_sum += quality
            quality_histogram[min(1000, max(0, round(quality * 1000)))] += 1
            quality_metadata = row.get("quality") if isinstance(row.get("quality"), dict) else {}
            reasons = {str(item) for item in quality_metadata.get("reasons", [])}
            if reasons & {"secret_like_content", "email_like_pii"}:
                secret_or_pii_hits += 1
            if _is_high_value(row, policy):
                high_value += 1
                if has_independent_review(row):
                    high_value_reviewed += 1
            if _record_category(row) == "verified_patch":
                verified_patches += 1
                evidence, _ = _verified_patch_evidence(row)
                if evidence is not None:
                    verified_patch_evidence += 1
    denominator = max(1, policy_tokens)
    source_fractions = {key: round(value / denominator, 8) for key, value in sorted(source_tokens.items())}
    family_fractions = {key: round(value / denominator, 8) for key, value in sorted(family_tokens.items())}
    return DatasetTrustMetrics(
        records=records,
        policy_tokens=policy_tokens,
        average_quality=round(quality_sum / max(1, records), 8),
        p10_quality=round(_quality_percentile(quality_histogram, records, 0.10), 8),
        license_coverage=round(license_complete / max(1, records), 8),
        provenance_coverage=round(provenance_complete / max(1, records), 8),
        high_value_records=high_value,
        high_value_review_coverage=round(high_value_reviewed / max(1, high_value), 8),
        verified_patch_records=verified_patches,
        verified_patch_evidence_coverage=round(verified_patch_evidence / max(1, verified_patches), 8),
        secret_or_pii_hits=secret_or_pii_hits,
        source_token_fractions=source_fractions,
        source_family_token_fractions=family_fractions,
        maximum_source_token_fraction=max(source_fractions.values(), default=0.0),
        maximum_source_family_token_fraction=max(family_fractions.values(), default=0.0),
        token_count_method=token_method,
    )


def _record_category(row: dict[str, Any]) -> str:
    quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
    category = str(row.get("category") or quality.get("data_type") or "general").lower()
    labels = {str(item).lower() for item in quality.get("labels", [])}
    if "verified_security_patch" in category or "patch" in labels:
        return "verified_patch"
    if "security" in category or "defensive_security" in labels:
        return "cybersecurity"
    if "agentic" in category or "agentic_coding" in labels:
        return "agentic"
    if "code" in category or "code" in labels:
        return "code"
    return "general"


def _provenance_complete(row: dict[str, Any]) -> bool:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    required = {
        "source_id",
        "source_family",
        "source_url",
        "immutable_revision",
        "license",
        "license_evidence_sha256",
        "legal_approval_sha256",
        "source_snapshot_sha256",
    }
    if not required.issubset(provenance):
        return False
    return all(provenance.get(key) not in {None, "", "rolling"} for key in required)


def _normalize_training_row(row: dict[str, Any], *, source_path: Path, default_category: str) -> dict[str, Any] | None:
    text = _row_text(row)
    if not text.strip():
        return None
    source = str(row.get("source") or row.get("repo_path") or source_path.stem)
    license_name = str(row.get("license") or row.get("spdx_license") or "unknown").lower()
    category = str(row.get("category") or default_category)
    normalized = dict(row)
    normalized["text"] = text
    normalized["source"] = source
    normalized["license"] = license_name
    normalized["category"] = category
    normalized["content_hash"] = str(row.get("content_hash") or stable_hash(text))
    existing_provenance = normalized.get("provenance") if isinstance(normalized.get("provenance"), dict) else {}
    normalized["provenance"] = {
        "source_path": str(source_path),
        "source": source,
        "license": license_name,
        **existing_provenance,
    }
    existing_quality = normalized.get("quality") if isinstance(normalized.get("quality"), dict) else None
    if existing_quality is None:
        decision = DatasetQualityGate(QualityGateConfig(require_license=True)).evaluate(normalized)
        if not decision.accepted:
            return None
        normalized["quality"] = decision.model_dump()
    elif not bool(existing_quality.get("accepted")) or not isinstance(existing_quality.get("quality_score"), (int, float)):
        return None
    return normalized


def _verified_patch_evidence(row: dict[str, Any]) -> tuple[VerifiedPatchEvidence | None, str | None]:
    verification = row.get("verification")
    if not isinstance(verification, dict):
        return None, "missing_verification_manifest"
    try:
        return VerifiedPatchEvidence.model_validate(verification), None
    except ValueError as exc:
        return None, f"invalid_verification_manifest:{exc}"


def _write_normalized_rows(input_paths: list[str], output_path: str | Path, *, default_category: str) -> PatchTaskDatasetReport:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    verified_evidence = 0
    rejection_reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            source_path = Path(path)
            if not source_path.exists():
                rejected += 1
                continue
            for row in iter_jsonl(source_path):
                evidence, evidence_error = _verified_patch_evidence(row)
                if evidence is None:
                    rejected += 1
                    rejection_reasons[evidence_error or "invalid_verification_manifest"] += 1
                    continue
                normalized = _normalize_training_row(row, source_path=source_path, default_category=default_category)
                if normalized is None:
                    rejected += 1
                    rejection_reasons["invalid_or_unscored_training_row"] += 1
                    continue
                if not _provenance_complete(normalized):
                    rejected += 1
                    rejection_reasons["incomplete_provenance"] += 1
                    continue
                normalized["verification"] = evidence.model_dump()
                categories[_record_category(normalized)] += 1
                handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
                verified_evidence += 1
    return PatchTaskDatasetReport(
        input_paths=input_paths,
        output_path=str(target),
        accepted=accepted,
        rejected=rejected,
        verified_evidence=verified_evidence,
        evidence_coverage=round(verified_evidence / max(1, accepted + rejected), 6),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        categories=dict(sorted(categories.items())),
    )


def _write_human_review_rows(input_paths: list[str], output_path: str | Path) -> HumanReviewPromotionReport:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    approved = 0
    rejected = 0
    independently_reviewed = 0
    pair_counts: Counter[tuple[str, str]] = Counter()
    categories: Counter[str] = Counter()
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            source_path = Path(path)
            if not source_path.exists():
                rejected += 1
                continue
            for row in iter_jsonl(source_path):
                review = row.get("review", {}) if isinstance(row.get("review"), dict) else {}
                status = str(review.get("status") or row.get("review_status") or "").lower()
                decisions = review.get("decisions") if isinstance(review.get("decisions"), list) else []
                reviewer_ids = {
                    str(item.get("reviewer_id") or "")
                    for item in decisions
                    if isinstance(item, dict) and item.get("reviewer_id")
                }
                decision_values = [
                    str(item.get("decision") or "")
                    for item in decisions
                    if isinstance(item, dict) and item.get("decision") in {"approve", "reject"}
                ]
                adjudication = review.get("adjudication") if isinstance(review.get("adjudication"), dict) else {}
                independently_approved = len(reviewer_ids) >= 2 and len(decision_values) >= 2 and set(decision_values) == {"approve"}
                adjudicated_approval = (
                    len(reviewer_ids) >= 2
                    and len(set(decision_values)) > 1
                    and adjudication.get("decision") == "approve"
                    and adjudication.get("adjudicator_id") not in reviewer_ids
                )
                if status not in {"approved", "human_approved"} or not (independently_approved or adjudicated_approval):
                    rejected += 1
                    continue
                normalized = _normalize_training_row(row, source_path=source_path, default_category="human_review_approved")
                if normalized is None:
                    rejected += 1
                    continue
                if not _provenance_complete(normalized):
                    rejected += 1
                    continue
                normalized["human_review"] = {"status": "approved", "source_path": str(source_path), **review}
                categories[_record_category(normalized)] += 1
                handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                approved += 1
                independently_reviewed += 1
                pair_counts[(decision_values[0], decision_values[1])] += 1
    paired_reviews = sum(pair_counts.values())
    if paired_reviews:
        observed = sum(count for pair, count in pair_counts.items() if pair[0] == pair[1]) / paired_reviews
        reviewer_1_approve = sum(count for pair, count in pair_counts.items() if pair[0] == "approve") / paired_reviews
        reviewer_2_approve = sum(count for pair, count in pair_counts.items() if pair[1] == "approve") / paired_reviews
        expected = (reviewer_1_approve * reviewer_2_approve) + (
            (1 - reviewer_1_approve) * (1 - reviewer_2_approve)
        )
        kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    else:
        kappa = 0.0
    return HumanReviewPromotionReport(
        input_paths=input_paths,
        output_path=str(target),
        approved=approved,
        rejected=rejected,
        independently_reviewed=independently_reviewed,
        review_coverage=round(independently_reviewed / max(1, approved + rejected), 6),
        reviewer_agreement=round(max(-1.0, min(1.0, kappa)), 6),
        categories=dict(sorted(categories.items())),
    )


def _concat_jsonl(input_paths: list[str | Path], output_path: str | Path) -> int:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            source = Path(path)
            if not source.exists():
                continue
            for row in iter_jsonl(source):
                digest = _row_hash(row)
                row["content_hash"] = digest
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def _load_holdout_hashes(paths: list[str]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        source = Path(path)
        if not source.exists():
            continue
        for row in iter_jsonl(source):
            text = str(row.get("prompt") or row.get("canonical_solution") or row.get("code") or row.get("text") or "")
            if text:
                hashes.add(stable_hash(text))
            task_id = row.get("task_id") or row.get("id")
            if task_id:
                hashes.add(stable_hash(str(task_id)))
    return hashes


def _holdout_contaminated(row: dict[str, Any], holdout_hashes: set[str]) -> bool:
    text = _row_text(row)
    digest = _row_hash(row)
    if digest in holdout_hashes or stable_hash(text) in holdout_hashes:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in ("canonical_solution", "humaneval", "mbpp", "swe-bench", "cyberseceval"))


def enforce_benchmark_holdout(
    input_path: str | Path,
    output_path: str | Path,
    removed_path: str | Path,
    benchmark_holdout_paths: list[str],
) -> HoldoutSeparationReport:
    holdout_hashes = _load_holdout_hashes(benchmark_holdout_paths)
    scanned = 0
    removed = 0
    target = Path(output_path)
    removed_target = Path(removed_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    removed_target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as clean_handle, removed_target.open("w", encoding="utf-8") as removed_handle:
        for row in iter_jsonl(input_path):
            scanned += 1
            if _holdout_contaminated(row, holdout_hashes):
                removed_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                removed += 1
                continue
            clean_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return HoldoutSeparationReport(
        benchmark_holdout_paths=benchmark_holdout_paths,
        benchmark_hashes=len(holdout_hashes),
        scanned_records=scanned,
        removed_records=removed,
        output_path=str(target),
        removed_path=str(removed_target),
    )


def _split_group(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    candidates = [
        ("repository", row.get("repository") or row.get("repo") or metadata.get("repository")),
        ("project_family", row.get("project_family") or metadata.get("project_family")),
        ("vulnerability_family", row.get("vulnerability_family") or metadata.get("vulnerability_family")),
        ("patch_lineage", row.get("patch_lineage") or metadata.get("patch_lineage")),
        ("task_signature", row.get("task_signature") or metadata.get("task_signature")),
        ("document_family", row.get("document_family") or metadata.get("document_family")),
        ("source_family", row.get("source_family") or provenance.get("source_family")),
    ]
    for kind, value in candidates:
        if value not in {None, ""}:
            return kind, str(value)
    source_id = row.get("source_id") or provenance.get("source_id") or row.get("source") or "unknown"
    return "source_content", f"{source_id}:{_row_hash(row)}"


def _assign_group_split(group_key: str, config: ProductionDatasetConfig) -> str:
    if config.dev_smoke:
        return "train"
    digest = hashlib.sha256(f"{config.seed}:{group_key}".encode("utf-8", "replace")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.val_fraction:
        return "val"
    return "test"


def split_train_val_test(input_path: str | Path, output_dir: str | Path, config: ProductionDatasetConfig) -> SplitManifest:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": root / "train.jsonl",
        "val": root / "val.jsonl",
        "test": root / "test.jsonl",
    }
    index_path = root / "split_groups.sqlite3"
    if index_path.exists():
        index_path.unlink()
    counts = {"train": 0, "val": 0, "test": 0}
    group_counts: Counter[str] = Counter()
    with closing(sqlite3.connect(index_path)) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE group_assignments(
              group_hash TEXT PRIMARY KEY,
              group_kind TEXT NOT NULL,
              split TEXT NOT NULL
            );
            """
        )
        with (
            paths["train"].open("w", encoding="utf-8") as train_handle,
            paths["val"].open("w", encoding="utf-8") as val_handle,
            paths["test"].open("w", encoding="utf-8") as test_handle,
        ):
            handles = {"train": train_handle, "val": val_handle, "test": test_handle}
            for row in iter_jsonl(input_path):
                group_kind, group_value = _split_group(row)
                group_hash = stable_hash(f"{group_kind}:{group_value}")
                existing = connection.execute(
                    "SELECT split FROM group_assignments WHERE group_hash=?",
                    (group_hash,),
                ).fetchone()
                split = str(existing[0]) if existing else _assign_group_split(group_hash, config)
                if existing is None:
                    connection.execute(
                        "INSERT INTO group_assignments(group_hash,group_kind,split) VALUES (?,?,?)",
                        (group_hash, group_kind, split),
                    )
                    group_counts[group_kind] += 1
                normalized = dict(row)
                normalized["split"] = split
                normalized["split_group"] = {"kind": group_kind, "hash": group_hash}
                handles[split].write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                counts[split] += 1
                if sum(counts.values()) % 10_000 == 0:
                    connection.commit()
        digest = hashlib.sha256()
        for group_hash, group_kind, split in connection.execute(
            "SELECT group_hash,group_kind,split FROM group_assignments ORDER BY group_hash"
        ):
            digest.update(f"{group_hash}:{group_kind}:{split}\n".encode("ascii"))
    return SplitManifest(
        train_path=str(paths["train"]),
        val_path=str(paths["val"]),
        test_path=str(paths["test"]),
        holdout_path=str(root / "holdout.jsonl"),
        train_records=counts["train"],
        val_records=counts["val"],
        test_records=counts["test"],
        holdout_records=0,
        fractions={"train": config.train_fraction, "val": config.val_fraction, "test": config.test_fraction},
        seed=config.seed,
        split_groups=dict(sorted(group_counts.items())),
        group_assignments_sha256=digest.hexdigest(),
        cross_split_group_collisions=0,
    )


def _copy_file(source: str | Path, target: str | Path) -> int:
    src = Path(source)
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return 0
    count = 0
    with src.open("r", encoding="utf-8", errors="replace") as source_handle, dst.open("w", encoding="utf-8") as target_handle:
        for line in source_handle:
            if line.strip():
                count += 1
            target_handle.write(line)
    return count


def _version_id(config: ProductionDatasetConfig, artifacts: dict[str, str]) -> str:
    digest_input = json.dumps({"config": config.model_dump(), "artifacts": artifacts}, sort_keys=True)
    return stable_hash(digest_input)[:16]


def _build_issues(
    *,
    config: ProductionDatasetConfig,
    final_records: int,
    split_manifest: SplitManifest,
    patch_report: PatchTaskDatasetReport,
    review_report: HumanReviewPromotionReport,
    validation_report: DatasetValidationReport,
    holdout_report: HoldoutSeparationReport,
) -> list[str]:
    issues: list[str] = []
    if final_records < config.min_promoted_records:
        issues.append(f"promoted_records_below_minimum:{final_records}<{config.min_promoted_records}")
    if split_manifest.train_records < config.min_train_records:
        issues.append(f"train_records_below_minimum:{split_manifest.train_records}<{config.min_train_records}")
    if patch_report.accepted < config.min_verified_patch_records:
        issues.append(f"verified_patch_records_below_minimum:{patch_report.accepted}<{config.min_verified_patch_records}")
    if review_report.approved < config.min_human_review_approved_records:
        issues.append(
            f"human_review_approved_records_below_minimum:{review_report.approved}<{config.min_human_review_approved_records}"
        )
    if validation_report.status != "passed":
        issues.append("dataset_validation_failed")
    if holdout_report.removed_records > 0:
        issues.append(f"benchmark_holdout_contamination_removed:{holdout_report.removed_records}")
    return issues


def _write_source_manifests(
    paths: list[str | Path],
    *,
    source_registry_report: SourceRegistryReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    licenses: Counter[str] = Counter()
    for path in paths:
        for row in iter_jsonl(path):
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            source_id, source_family = _source_identity(row)
            revision = str(provenance.get("immutable_revision") or "")
            snapshot = str(provenance.get("source_snapshot_sha256") or "")
            key = (source_id, snapshot)
            sources[key] = {
                "source_id": source_id,
                "source_family": source_family,
                "immutable_revision": revision,
                "source_snapshot_sha256": snapshot,
                "license_evidence_sha256": str(provenance.get("license_evidence_sha256") or ""),
                "legal_approval_sha256": str(provenance.get("legal_approval_sha256") or ""),
            }
            licenses[str(row.get("license") or row.get("spdx_license") or "unknown").lower()] += 1
    source_path = _write_json(
        output_dir / "source_snapshot_manifest.json",
        {
            "registry_snapshot_sha256": source_registry_report.source_snapshot_sha256,
            "sources": sorted(sources.values(), key=lambda item: (item["source_id"], item["source_snapshot_sha256"])),
        },
    )
    license_path = _write_json(
        output_dir / "license_manifest.json",
        {
            "registry_snapshot_sha256": source_registry_report.source_snapshot_sha256,
            "licenses": dict(sorted(licenses.items())),
            "record_count": sum(licenses.values()),
        },
    )
    return source_path, license_path


def _promotion_decision(
    *,
    policy: DatasetTrustPolicyContract,
    trust_metrics: DatasetTrustMetrics,
    benchmark_report: BenchmarkContaminationFilterReport,
    holdout_report: HoldoutSeparationReport,
    dedup_report: NearDedupReport,
    split_manifest: SplitManifest,
    source_reputation_report: SourceReputationReport,
    patch_report: PatchTaskDatasetReport,
    review_report: HumanReviewPromotionReport,
) -> DatasetPromotionDecision:
    threshold = policy.promotion
    source_limits = policy.source_limits
    residual_near_duplicate_fraction = 0.0
    contamination = benchmark_report.rejected + holdout_report.removed_records
    human_sample_acceptance = review_report.approved / max(1, review_report.approved + review_report.rejected)
    governed_sources = True
    for item in source_reputation_report.sources:
        source_fraction = trust_metrics.source_token_fractions.get(item.source, 0.0)
        mature = (
            item.approval_status == "approved"
            and item.trust_tier in {"reviewed", "trusted"}
            and item.action in {"promote", "watch"}
            and item.reputation_lower_bound >= source_limits.minimum_reputation_lower_bound
        )
        bounded_quarantine = (
            item.approval_status == "approved"
            and item.action == "quarantine"
            and item.license_trust >= 1.0
            and source_fraction <= source_limits.new_source_max_token_fraction
        )
        if not mature and not bounded_quarantine:
            governed_sources = False
            break
    checks = {
        "minimum_records": trust_metrics.records >= threshold.minimum_records,
        "average_quality": trust_metrics.average_quality >= threshold.minimum_average_quality,
        "p10_quality": trust_metrics.p10_quality >= threshold.minimum_p10_quality,
        "license_coverage": trust_metrics.license_coverage >= threshold.required_license_coverage,
        "provenance_coverage": trust_metrics.provenance_coverage >= threshold.required_provenance_coverage,
        "high_value_review_coverage": (
            trust_metrics.high_value_review_coverage >= threshold.required_high_value_review_coverage
        ),
        "verified_patch_evidence_coverage": (
            trust_metrics.verified_patch_evidence_coverage >= threshold.required_verified_patch_evidence_coverage
        ),
        "secret_or_pii_hits": trust_metrics.secret_or_pii_hits <= threshold.maximum_secret_or_pii_hits,
        "benchmark_contamination": contamination <= threshold.maximum_benchmark_contamination,
        "residual_near_duplicate_fraction": (
            residual_near_duplicate_fraction <= threshold.maximum_residual_near_duplicate_fraction
        ),
        "source_token_cap": trust_metrics.maximum_source_token_fraction <= source_limits.source_max_token_fraction,
        "source_family_token_cap": (
            trust_metrics.maximum_source_family_token_fraction <= source_limits.source_family_max_token_fraction
        ),
        "cross_split_group_collisions": split_manifest.cross_split_group_collisions == 0,
        "exact_duplicates_after_promotion": True,
        "governed_source_reputation": governed_sources,
        "reviewer_agreement": review_report.reviewer_agreement >= policy.review.reviewer_agreement_minimum,
        "human_sample_acceptance": human_sample_acceptance >= policy.review.sampled_acceptance_minimum,
        "verified_patch_report_consistent": patch_report.accepted == patch_report.verified_evidence,
    }
    issues = [name for name, passed in checks.items() if not passed]
    return DatasetPromotionDecision(
        status="promoted" if not issues else "rejected",
        checks=checks,
        metrics={
            "records": trust_metrics.records,
            "average_quality": trust_metrics.average_quality,
            "p10_quality": trust_metrics.p10_quality,
            "license_coverage": trust_metrics.license_coverage,
            "provenance_coverage": trust_metrics.provenance_coverage,
            "high_value_review_coverage": trust_metrics.high_value_review_coverage,
            "verified_patch_evidence_coverage": trust_metrics.verified_patch_evidence_coverage,
            "secret_or_pii_hits": trust_metrics.secret_or_pii_hits,
            "benchmark_contamination": contamination,
            "residual_near_duplicate_fraction": residual_near_duplicate_fraction,
            "maximum_source_token_fraction": trust_metrics.maximum_source_token_fraction,
            "maximum_source_family_token_fraction": trust_metrics.maximum_source_family_token_fraction,
            "human_sample_acceptance": round(human_sample_acceptance, 8),
            "exact_duplicates_removed": dedup_report.exact_duplicates,
            "near_duplicates_removed": dedup_report.near_duplicates,
        },
        issues=issues,
        policy_id=policy.policy_id,
    )


def write_quality_dashboard(
    path: str | Path,
    *,
    dataset_id: str,
    trust_metrics: DatasetTrustMetrics,
    decision: DatasetPromotionDecision,
    source_reputation: SourceReputationReport,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    check_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class=\"{'ok' if passed else 'bad'}\">"
        f"{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in sorted(decision.checks.items())
    )
    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.source)}</td>"
        f"<td>{item.rows:,}</td>"
        f"<td>{item.reputation_score:.4f}</td>"
        f"<td>{item.reputation_lower_bound:.4f}</td>"
        f"<td>{html.escape(item.action)}</td>"
        "</tr>"
        for item in source_reputation.sources
    )
    source_mix_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{fraction:.4%}</td></tr>"
        for name, fraction in sorted(
            trust_metrics.source_token_fractions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Craftly Dataset Quality - {html.escape(dataset_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f7f8fa;color:#17191c}}
header,main{{max-width:1180px;margin:auto;padding:24px}}header{{background:#15191f;color:white;max-width:none}}
.inner{{max-width:1180px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.metric{{background:white;border:1px solid #d9dde3;border-radius:6px;padding:16px}}.metric strong{{display:block;font-size:1.5rem}}
section{{margin-top:24px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border:1px solid #d9dde3;text-align:left}}
.ok{{color:#08783e;font-weight:700}}.bad{{color:#b42318;font-weight:700}}h1,h2{{letter-spacing:0}}
</style></head><body><header><div class="inner"><h1>Craftly Dataset Quality</h1>
<p>{html.escape(dataset_id)} | decision: <strong>{html.escape(decision.status)}</strong> | policy: {html.escape(decision.policy_id)}</p>
</div></header><main><div class="grid">
<div class="metric">Records<strong>{trust_metrics.records:,}</strong></div>
<div class="metric">Policy tokens<strong>{trust_metrics.policy_tokens:,}</strong></div>
<div class="metric">Average quality<strong>{trust_metrics.average_quality:.4f}</strong></div>
<div class="metric">P10 quality<strong>{trust_metrics.p10_quality:.4f}</strong></div>
<div class="metric">Provenance<strong>{trust_metrics.provenance_coverage:.2%}</strong></div>
<div class="metric">High-value reviews<strong>{trust_metrics.high_value_review_coverage:.2%}</strong></div>
</div><section><h2>Promotion gates</h2><table><thead><tr><th>Gate</th><th>Result</th></tr></thead>
<tbody>{check_rows}</tbody></table></section>
<section><h2>Source token mix</h2><table><thead><tr><th>Source</th><th>Fraction</th></tr></thead>
<tbody>{source_mix_rows}</tbody></table></section>
<section><h2>Source evidence</h2><table><thead><tr><th>Source</th><th>Rows</th><th>Score</th><th>Review lower bound</th><th>Action</th></tr></thead>
<tbody>{source_rows}</tbody></table></section></main></body></html>"""
    target.write_text(document, encoding="utf-8")
    return target


def validate_dataset_manifest_for_promotion(
    manifest_path: str | Path,
    *,
    trust_policy_path: str | Path = "config/dataset_trust_policy.json",
) -> ProductionDatasetManifest:
    source = Path(manifest_path).resolve(strict=True)
    manifest = ProductionDatasetManifest.model_validate_json(source.read_text(encoding="utf-8"))
    policy_sha256 = _sha256_file(trust_policy_path)
    if manifest.policy_sha256 != policy_sha256:
        raise ValueError("dataset manifest policy hash does not match the active trust policy")
    if manifest.status != "promoted" or manifest.promotion_decision.get("status") != "promoted":
        raise ValueError("dataset version has not passed production promotion")
    output_root = Path(manifest.output_dir).resolve(strict=True)
    for role, expected in manifest.artifact_sha256.items():
        artifact = manifest.artifacts.get(role)
        if not artifact:
            raise ValueError(f"manifest hash references missing artifact role: {role}")
        artifact_path = Path(artifact).resolve(strict=True)
        try:
            artifact_path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"dataset artifact escapes version output root: {role}") from exc
        if _sha256_file(artifact_path) != expected:
            raise ValueError(f"dataset artifact hash mismatch: {role}")
    if not manifest.dev_smoke:
        if not manifest.advancement_decision_path or not manifest.advancement_decision_sha256:
            raise ValueError("production dataset manifest is missing calibration_5k advancement evidence")
        if _sha256_file(manifest.advancement_decision_path) != manifest.advancement_decision_sha256:
            raise ValueError("production dataset advancement decision hash changed")
        required_paths = {
            "source_registry_path",
            "protected_benchmark_config_path",
            "protected_benchmark_manifest_path",
            "reviewer_roster_path",
            "legal_evidence_dir",
            "trust_policy_path",
        }
        missing = sorted(required_paths.difference(manifest.governance_paths))
        if missing:
            raise ValueError(f"production dataset manifest is missing governance paths: {missing}")
        validate_advancement_decision(
            manifest.advancement_decision_path,
            expected_stage="calibration_5k",
            expected_next_stage="100k_dataset_build_allowed",
            sources_path=manifest.governance_paths["source_registry_path"],
            protected_config_path=manifest.governance_paths["protected_benchmark_config_path"],
            protected_manifest_path=manifest.governance_paths["protected_benchmark_manifest_path"],
            reviewer_roster_path=manifest.governance_paths["reviewer_roster_path"],
            legal_evidence_dir=manifest.governance_paths["legal_evidence_dir"],
            trust_policy_path=manifest.governance_paths["trust_policy_path"],
        )
    return manifest


def build_production_dataset(config: ProductionDatasetConfig) -> ProductionDatasetManifest:
    advancement_decision: CalibrationDecision | None = None
    advancement_decision_sha256: str | None = None
    if not config.dev_smoke:
        if not config.advancement_decision_path:
            raise ValueError(
                "production_100k requires --advancement-decision from a passed calibration_5k stage"
            )
        advancement_decision = validate_advancement_decision(
            config.advancement_decision_path,
            expected_stage="calibration_5k",
            expected_next_stage="100k_dataset_build_allowed",
            sources_path=config.source_registry_path,
            protected_config_path=config.protected_benchmark_config_path,
            protected_manifest_path=config.protected_benchmark_manifest_path,
            reviewer_roster_path=config.reviewer_roster_path,
            legal_evidence_dir=config.legal_evidence_dir,
            trust_policy_path=config.trust_policy_path,
        )
        advancement_decision_sha256 = _sha256_file(config.advancement_decision_path)

    input_paths = _expand_input_paths(config.input_paths)
    if not input_paths:
        raise ValueError("no input JSONL files found for production dataset build")
    policy = load_dataset_trust_policy(config.trust_policy_path)
    registry = SourceRegistry.from_file(config.source_registry_path)
    source_registry_report = registry.validate(production=False)
    source_registry_production_issue: str | None = None
    if not config.dev_smoke:
        try:
            registry.validate(production=True)
        except ValueError as exc:
            source_registry_production_issue = str(exc)
    root = Path(config.output_dir)
    reports_dir = root / "reports"
    work_dir = root / "work"
    final_dir = root / "final"
    for directory in (reports_dir, work_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)
    advancement_report = _write_json(
        reports_dir / "advancement_evidence_report.json",
        {
            "stage": config.stage,
            "required_prior_stage": "calibration_5k",
            "decision_path": (
                str(Path(config.advancement_decision_path).resolve())
                if config.advancement_decision_path
                else None
            ),
            "decision_sha256": advancement_decision_sha256,
            "decision": advancement_decision.model_dump(mode="json") if advancement_decision else None,
            "dev_smoke_bypass": config.dev_smoke,
        },
    )
    _write_json(reports_dir / "source_registry_report.json", source_registry_report)

    license_clean = work_dir / "01_license_clean.jsonl"
    license_report = filter_jsonl_by_license(
        input_paths,
        license_clean,
        strict_unknown=not config.allow_unknown_license,
    )
    _write_json(reports_dir / "license_filter_report.json", license_report)

    quality_clean = work_dir / "02_quality_clean.jsonl"
    quality_report = DatasetQualityGate(QualityGateConfig(require_license=True)).filter_jsonl(license_clean, quality_clean)
    _write_json(reports_dir / "quality_gate_report.json", quality_report)

    no_benchmark = work_dir / "03_benchmark_clean.jsonl"
    protected_index_path: Path | None = None
    protected_holdout_issue: str | None = None
    available_holdouts = [path for path in config.benchmark_holdout_paths if Path(path).is_file()]
    if available_holdouts:
        protected_index_path = build_protected_fingerprint_index(
            config.benchmark_holdout_paths,
            reports_dir / "protected_benchmark_fingerprints.sqlite3",
            require_all=not config.dev_smoke,
        )
    elif not config.dev_smoke:
        protected_holdout_issue = "production promotion requires all protected benchmark holdout files"
    benchmark_report = filter_benchmark_contamination_jsonl(
        [quality_clean],
        no_benchmark,
        protected_index_path=protected_index_path,
    )
    _write_json(reports_dir / "benchmark_contamination_filter_report.json", benchmark_report)

    deduped = work_dir / "04_deduped.jsonl"
    dedup_report = deduplicate_jsonl(
        [no_benchmark],
        deduped,
        hamming_threshold=config.near_dedup_hamming_threshold,
        index_path=work_dir / "dedup_index.sqlite3",
    )
    _write_json(reports_dir / "near_duplicate_report.json", dedup_report)

    source_quality_report = write_source_quality_report([deduped], reports_dir / "source_quality_report.json")
    source_reputation_report = write_source_reputation_report(
        reports_dir / "source_reputation_report.json",
        source_quality_report_path=reports_dir / "source_quality_report.json",
        contamination_report_path=reports_dir / "benchmark_contamination_filter_report.json",
        dedup_report_path=reports_dir / "near_duplicate_report.json",
        review_report_path=config.source_review_report_path,
        source_registry_path=config.source_registry_path,
        minimum_reviewed_records=policy.source_limits.minimum_reviewed_records,
    )
    if Path(config.source_registry_path).exists():
        source_budget_plan = write_source_budget_plan(
            reports_dir / "source_budget_plan.json",
            sources_path=config.source_registry_path,
            reputation_report_path=reports_dir / "source_reputation_report.json",
            target_total_docs=config.source_budget_target_docs,
        )
    else:
        source_budget_plan = SourceBudgetPlan(
            target_total_docs=config.source_budget_target_docs,
            allocated_total_docs=0,
            unallocated_docs=config.source_budget_target_docs,
            eligible_sources=0,
            blocked_sources=0,
            budgets=[],
        )
        _write_json(reports_dir / "source_budget_plan.json", source_budget_plan)

    promoted = work_dir / "05_promoted.jsonl"
    holdout = final_dir / "holdout.jsonl"
    review_queue = root / "review" / "human_review_queue.jsonl"
    decisions = reports_dir / "training_gate_decisions.jsonl"
    gate_report = apply_training_data_gate(
        input_paths=[deduped],
        promoted_path=promoted,
        holdout_path=holdout,
        review_queue_path=review_queue,
        decisions_path=decisions,
        reputation_report_path=reports_dir / "source_reputation_report.json",
        config=TrainingDataGateConfig(
            min_quality_score=min(config.min_quality_score, 0.30) if config.dev_smoke else config.min_quality_score,
            min_source_reputation_score=0.0 if config.dev_smoke else config.min_source_reputation_score,
            min_reputation_lower_bound=policy.source_limits.minimum_reputation_lower_bound,
            require_governed_sources=not config.dev_smoke,
            allow_governed_quarantine=not config.dev_smoke,
            reject_noise_flags=not config.dev_smoke,
            require_high_value_review=not config.dev_smoke,
            routine_review_fraction=0.0 if config.dev_smoke else policy.review.routine_sample_fraction,
            eval_holdout_fraction=config.eval_holdout_fraction,
            seed=config.seed,
        ),
    )
    _write_json(reports_dir / "training_data_gate_report.json", gate_report)

    patch_rows = work_dir / "06_verified_patch_tasks.jsonl"
    patch_report = _write_normalized_rows(config.verified_patch_paths, patch_rows, default_category="verified_security_patch")
    _write_json(reports_dir / "verified_patch_task_report.json", patch_report)

    human_rows = work_dir / "07_human_review_approved.jsonl"
    human_report = _write_human_review_rows(config.human_review_approved_paths, human_rows)
    _write_json(reports_dir / "human_review_approved_report.json", human_report)

    combined = work_dir / "08_combined_promoted.jsonl"
    final_records_before_holdout = _concat_jsonl([promoted, patch_rows, human_rows], combined)

    final_deduped = work_dir / "09_final_deduped.jsonl"
    final_dedup_report = deduplicate_jsonl(
        [combined],
        final_deduped,
        hamming_threshold=config.near_dedup_hamming_threshold,
        index_path=work_dir / "final_dedup_index.sqlite3",
    )
    _write_json(reports_dir / "final_near_duplicate_report.json", final_dedup_report)

    holdout_clean = work_dir / "10_holdout_clean.jsonl"
    removed_holdout = reports_dir / "benchmark_holdout_removed.jsonl"
    holdout_report = enforce_benchmark_holdout(final_deduped, holdout_clean, removed_holdout, config.benchmark_holdout_paths)
    _write_json(reports_dir / "benchmark_holdout_separation_report.json", holdout_report)

    split_manifest = split_train_val_test(holdout_clean, final_dir, config)
    _copy_file(holdout, split_manifest.holdout_path)
    split_manifest = split_manifest.model_copy(
        update={"holdout_records": sum(1 for _ in iter_jsonl(split_manifest.holdout_path)) if Path(split_manifest.holdout_path).exists() else 0}
    )
    _write_json(reports_dir / "train_val_test_split_manifest.json", split_manifest)

    trust_metrics = calculate_dataset_trust_metrics(
        [split_manifest.train_path, split_manifest.val_path, split_manifest.test_path],
        policy=policy,
        tokenizer_path=config.tokenizer_path,
    )
    _write_json(reports_dir / "dataset_trust_metrics.json", trust_metrics)
    source_snapshot_manifest, license_manifest = _write_source_manifests(
        [split_manifest.train_path, split_manifest.val_path, split_manifest.test_path],
        source_registry_report=source_registry_report,
        output_dir=reports_dir,
    )
    mix_manifest = _write_json(
        reports_dir / "mix_manifest.json",
        {
            "token_count_method": trust_metrics.token_count_method,
            "policy_tokens": trust_metrics.policy_tokens,
            "source_token_fractions": trust_metrics.source_token_fractions,
            "source_family_token_fractions": trust_metrics.source_family_token_fractions,
            "maximum_source_token_fraction": trust_metrics.maximum_source_token_fraction,
            "maximum_source_family_token_fraction": trust_metrics.maximum_source_family_token_fraction,
        },
    )

    validation_min = 1 if config.dev_smoke else config.min_train_records
    validation_report = validate_dataset(
        DatasetValidationConfig(
            input_paths=[split_manifest.train_path],
            min_records=validation_min,
            max_duplicate_fraction=config.max_duplicate_fraction,
            min_avg_chars=config.min_avg_chars,
            require_license=True,
            require_quality=True,
            require_categories=["general", "code", "cybersecurity"] if not config.dev_smoke else [],
        )
    )
    validation_report.write(reports_dir / "dataset_validation")

    final_records = split_manifest.train_records + split_manifest.val_records + split_manifest.test_records
    issues = _build_issues(
        config=config,
        final_records=final_records,
        split_manifest=split_manifest,
        patch_report=patch_report,
        review_report=human_report,
        validation_report=validation_report,
        holdout_report=holdout_report,
    )
    promotion_decision = _promotion_decision(
        policy=policy,
        trust_metrics=trust_metrics,
        benchmark_report=benchmark_report,
        holdout_report=holdout_report,
        dedup_report=final_dedup_report,
        split_manifest=split_manifest,
        source_reputation_report=source_reputation_report,
        patch_report=patch_report,
        review_report=human_report,
    )
    if source_registry_production_issue:
        promotion_decision = promotion_decision.model_copy(
            update={
                "status": "rejected",
                "checks": {**promotion_decision.checks, "source_registry_production_contract": False},
                "issues": [*promotion_decision.issues, f"source_registry:{source_registry_production_issue}"],
            }
        )
    if protected_holdout_issue:
        promotion_decision = promotion_decision.model_copy(
            update={
                "status": "rejected",
                "checks": {**promotion_decision.checks, "protected_benchmark_registry": False},
                "issues": [*promotion_decision.issues, f"protected_holdouts:{protected_holdout_issue}"],
            }
        )
    issues.extend(f"promotion:{issue}" for issue in promotion_decision.issues)
    status = "promoted" if not issues and promotion_decision.status == "promoted" else "rejected"
    if config.dev_smoke:
        non_validation_issues = [issue for issue in issues if issue == "dataset_validation_failed"]
        status = "passed" if not non_validation_issues else "failed"

    promotion_decision_path = _write_json(reports_dir / "promotion_decision.json", promotion_decision)
    quality_dashboard = write_quality_dashboard(
        reports_dir / "dataset_quality_dashboard.html",
        dataset_id=config.dataset_id,
        trust_metrics=trust_metrics,
        decision=promotion_decision,
        source_reputation=source_reputation_report,
    )
    artifacts = {
        "train": split_manifest.train_path,
        "val": split_manifest.val_path,
        "test": split_manifest.test_path,
        "holdout": split_manifest.holdout_path,
        "human_review_queue": str(review_queue),
        "training_gate_decisions": str(decisions),
        "benchmark_holdout_removed": str(removed_holdout),
        "source_snapshot_manifest": str(source_snapshot_manifest),
        "license_manifest": str(license_manifest),
        "quality_report": str(reports_dir / "dataset_trust_metrics.json"),
        "dedup_report": str(reports_dir / "final_near_duplicate_report.json"),
        "contamination_report": str(reports_dir / "benchmark_contamination_filter_report.json"),
        "review_report": str(reports_dir / "human_review_approved_report.json"),
        "verified_patch_report": str(reports_dir / "verified_patch_task_report.json"),
        "split_manifest": str(reports_dir / "train_val_test_split_manifest.json"),
        "mix_manifest": str(mix_manifest),
        "promotion_decision": str(promotion_decision_path),
        "quality_dashboard": str(quality_dashboard),
        "advancement_evidence": str(advancement_report),
    }
    artifact_sha256 = _hash_existing_artifacts(artifacts)
    manifest = ProductionDatasetManifest(
        dataset_id=config.dataset_id,
        version_id=_version_id(config, artifacts),
        status=status,
        output_dir=str(root),
        dev_smoke=config.dev_smoke,
        artifacts=artifacts,
        metrics={
            "final_records_before_holdout": final_records_before_holdout,
            "promoted_records": final_records,
            "train_records": split_manifest.train_records,
            "val_records": split_manifest.val_records,
            "test_records": split_manifest.test_records,
            "holdout_records": split_manifest.holdout_records,
            "verified_patch_records": patch_report.accepted,
            "human_review_approved_records": human_report.approved,
            "benchmark_holdout_removed_records": holdout_report.removed_records,
            "trust": trust_metrics.model_dump(),
        },
        issues=issues,
        reports={
            "advancement_evidence": (
                advancement_decision.model_dump(mode="json")
                if advancement_decision is not None
                else {"status": "dev_smoke_bypass"}
            ),
            "license_filter": license_report.model_dump(),
            "quality_gate": quality_report.model_dump(),
            "benchmark_contamination_filter": benchmark_report.model_dump(),
            "near_duplicate": dedup_report.model_dump(),
            "final_near_duplicate": final_dedup_report.model_dump(),
            "source_quality": source_quality_report.model_dump(),
            "source_reputation": source_reputation_report.model_dump(),
            "source_budget_plan": source_budget_plan.model_dump(),
            "training_data_gate": gate_report.model_dump(),
            "verified_patch_tasks": patch_report.model_dump(),
            "human_review_approved": human_report.model_dump(),
            "benchmark_holdout_separation": holdout_report.model_dump(),
            "split_manifest": split_manifest.model_dump(),
            "dataset_validation": validation_report.model_dump(),
            "source_registry": source_registry_report.model_dump(),
            "dataset_trust_metrics": trust_metrics.model_dump(),
            "promotion_decision": promotion_decision.model_dump(),
        },
        artifact_sha256=artifact_sha256,
        policy_sha256=_sha256_file(config.trust_policy_path),
        tokenizer_sha256=_sha256_file(config.tokenizer_path) if config.tokenizer_path else None,
        advancement_decision_path=(
            str(Path(config.advancement_decision_path).resolve())
            if config.advancement_decision_path
            else None
        ),
        advancement_decision_sha256=advancement_decision_sha256,
        governance_paths={
            "source_registry_path": str(Path(config.source_registry_path).resolve()),
            "protected_benchmark_config_path": str(Path(config.protected_benchmark_config_path).resolve()),
            "protected_benchmark_manifest_path": str(Path(config.protected_benchmark_manifest_path).resolve()),
            "reviewer_roster_path": str(Path(config.reviewer_roster_path).resolve()),
            "legal_evidence_dir": str(Path(config.legal_evidence_dir).resolve()),
            "trust_policy_path": str(Path(config.trust_policy_path).resolve()),
        },
        promotion_decision=promotion_decision.model_dump(),
    )
    manifest.write(root)
    return manifest


def write_markdown(manifest: ProductionDatasetManifest, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Production Dataset Manifest",
        "",
        f"- dataset_id: {manifest.dataset_id}",
        f"- version_id: {manifest.version_id}",
        f"- status: {manifest.status}",
        f"- dev_smoke: {str(manifest.dev_smoke).lower()}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in sorted(manifest.metrics.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Issues", ""])
    if manifest.issues:
        for issue in manifest.issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", "", "| role | path |", "|---|---|"])
    for key, value in sorted(manifest.artifacts.items()):
        lines.append(f"| {key} | {value} |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a governed production dataset pack for Craftly scratch training.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", default="data/production/craftly-corpus")
    parser.add_argument("--dataset-id", default="craftly-corpus")
    parser.add_argument("--source-registry", default="config/data_sources.ultimate.json")
    parser.add_argument("--trust-policy", default="config/dataset_trust_policy.json")
    parser.add_argument("--legal-evidence-dir", default="governance/source-approvals")
    parser.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    parser.add_argument("--protected-config", default="config/protected_benchmarks.json")
    parser.add_argument(
        "--protected-manifest",
        default="data/eval/protected/protected_benchmark_manifest.json",
    )
    parser.add_argument("--advancement-decision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--source-review-report")
    parser.add_argument("--benchmark-holdout", action="append", default=[])
    parser.add_argument("--verified-patch", action="append", default=[])
    parser.add_argument("--human-review-approved", action="append", default=[])
    parser.add_argument("--dev-smoke", action="store_true")
    parser.add_argument("--min-promoted-records", type=int, default=100_000)
    parser.add_argument("--min-verified-patch-records", type=int, default=100)
    parser.add_argument("--min-human-review-approved-records", type=int, default=100)
    parser.add_argument("--min-train-records", type=int, default=90_000)
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ProductionDatasetConfig(
        input_paths=args.input,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        source_registry_path=args.source_registry,
        trust_policy_path=args.trust_policy,
        legal_evidence_dir=args.legal_evidence_dir,
        reviewer_roster_path=args.reviewer_roster,
        protected_benchmark_config_path=args.protected_config,
        protected_benchmark_manifest_path=args.protected_manifest,
        advancement_decision_path=args.advancement_decision,
        tokenizer_path=args.tokenizer,
        source_review_report_path=args.source_review_report,
        benchmark_holdout_paths=args.benchmark_holdout or ["data/eval/humaneval.jsonl", "data/eval/mbpp.jsonl"],
        verified_patch_paths=args.verified_patch,
        human_review_approved_paths=args.human_review_approved,
        dev_smoke=args.dev_smoke,
        min_promoted_records=args.min_promoted_records,
        min_verified_patch_records=args.min_verified_patch_records,
        min_human_review_approved_records=args.min_human_review_approved_records,
        min_train_records=args.min_train_records,
        allow_unknown_license=args.allow_unknown_license,
        seed=args.seed,
    )
    manifest = build_production_dataset(config)
    print(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))
    if manifest.status not in {"passed", "promoted"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
