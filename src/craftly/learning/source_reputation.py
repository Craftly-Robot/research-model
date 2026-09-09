"""Source reputation scoring for production data acquisition."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel


class SourceReputationScore(StrictModel):
    source: str
    rows: int
    avg_quality_score: float = Field(ge=0.0, le=1.0)
    task_coverage: float = Field(ge=0.0, le=1.0)
    security_coverage: float = Field(ge=0.0, le=1.0)
    code_coverage: float = Field(ge=0.0, le=1.0)
    review_approval_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    contamination_rate: float = Field(ge=0.0, le=1.0)
    license_trust: float = Field(ge=0.0, le=1.0)
    benchmark_feedback_score: float = Field(ge=0.0, le=1.0)
    reputation_score: float = Field(ge=0.0, le=1.0)
    reputation_lower_bound: float = Field(default=0.0, ge=0.0, le=1.0)
    reviewed_records: int = Field(default=0, ge=0)
    trust_tier: str = "quarantine"
    approval_status: str = "pending"
    action: str
    reasons: list[str] = Field(default_factory=list)


class SourceReputationReport(StrictModel):
    sources: list[SourceReputationScore]
    created_at_unix: float = Field(default_factory=time.time)


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _wilson_lower_bound(approved: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = approved / total
    denominator = 1 + (z * z / total)
    centre = proportion + (z * z / (2 * total))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return _clamp((centre - margin) / denominator)


def _registry_evidence(path: str | Path | None) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    evidence: dict[str, dict[str, Any]] = {}
    for item in payload.get("sources", []):
        name = str(item.get("name") or "")
        source_id = str(item.get("source_id") or name)
        record = {
            "trust_tier": str(item.get("trust_tier") or "quarantine"),
            "approval_status": str(item.get("approval_status") or "pending"),
            "approved_use": str(item.get("approved_use") or "defensive"),
            "license_evidence": bool(item.get("license_evidence_sha256")),
            "legal_approval": bool(item.get("legal_approval_sha256")),
        }
        evidence[name] = record
        evidence[source_id] = record
    return evidence


def _review_counts(review_report: dict[str, Any], source: str) -> tuple[int, int]:
    by_source = review_report.get("by_source", {}) if review_report else {}
    item = by_source.get(source, {}) if isinstance(by_source, dict) else {}
    approved = int(item.get("approved", 0))
    rejected = int(item.get("rejected", 0))
    return approved, approved + rejected


def _duplicate_rate(dedup_report: dict[str, Any], source: str) -> float:
    by_source = dedup_report.get("by_source", {}) if dedup_report else {}
    item = by_source.get(source, {}) if isinstance(by_source, dict) else {}
    accepted = int(item.get("accepted", 0))
    duplicates = sum(
        int(item.get(key, 0))
        for key in (
            "exact_duplicates",
            "structural_duplicates",
            "lineage_duplicates",
            "near_duplicates",
        )
    )
    return _clamp(duplicates / max(1, accepted + duplicates))


def _contamination_rate(
    contamination_report: dict[str, Any], source: str, rows: int
) -> float:
    hits = contamination_report.get("hits", []) if contamination_report else []
    source_hits = sum(
        1
        for hit in hits
        if str(hit.get("source") or hit.get("source_id") or "unknown") == source
    )
    return _clamp(source_hits / max(1, rows))


def build_source_reputation_report(
    *,
    source_quality_report_path: str | Path,
    task_report_path: str | Path | None = None,
    review_report_path: str | Path | None = None,
    feedback_report_path: str | Path | None = None,
    contamination_report_path: str | Path | None = None,
    dedup_report_path: str | Path | None = None,
    source_registry_path: str | Path | None = None,
    minimum_reviewed_records: int = 100,
) -> SourceReputationReport:
    source_quality = _load_json(source_quality_report_path)
    task_report = _load_json(task_report_path)
    review_report = _load_json(review_report_path)
    feedback_report = _load_json(feedback_report_path)
    contamination_report = _load_json(contamination_report_path)
    dedup_report = _load_json(dedup_report_path)

    registry = _registry_evidence(source_registry_path)
    task_by_source = task_report.get("by_source", {}) if task_report else {}
    feedback_by_source = feedback_report.get("by_source", {}) if feedback_report else {}

    scores: list[SourceReputationScore] = []
    for item in source_quality.get("sources", []):
        source_name = str(item.get("source") or "unknown")
        rows = int(item.get("rows", 0))
        security_coverage = _clamp(
            float(item.get("defensive_security_rows", 0)) / max(1, rows)
        )
        code_coverage = _clamp(float(item.get("code_rows", 0)) / max(1, rows))
        task_evidence = (
            task_by_source.get(source_name, {})
            if isinstance(task_by_source, dict)
            else {}
        )
        if isinstance(task_evidence, dict):
            task_count = int(
                task_evidence.get("extracted", task_evidence.get("tasks", 0))
            )
            task_coverage = _clamp(task_count / max(1, rows))
        else:
            task_coverage = 0.0
        feedback_evidence = (
            feedback_by_source.get(source_name, {})
            if isinstance(feedback_by_source, dict)
            else {}
        )
        benchmark_feedback_score = _clamp(
            float(feedback_evidence.get("score", 0.0))
            if isinstance(feedback_evidence, dict)
            else 0.0
        )
        avg_quality = _clamp(float(item.get("avg_quality_score", 0.0)))
        source_evidence = registry.get(source_name, {})
        approval_status = str(source_evidence.get("approval_status") or "pending")
        trust_tier = str(source_evidence.get("trust_tier") or "quarantine")
        license_trust = (
            1.0
            if approval_status == "approved"
            and bool(source_evidence.get("license_evidence"))
            and bool(source_evidence.get("legal_approval"))
            else 0.0
        )
        approved, reviewed_records = _review_counts(review_report, source_name)
        review_approval_rate = (
            _clamp(approved / reviewed_records) if reviewed_records else 0.0
        )
        review_lower_bound = _wilson_lower_bound(approved, reviewed_records)
        duplicate_rate = _duplicate_rate(dedup_report, source_name)
        contamination_rate = _contamination_rate(
            contamination_report, source_name, rows
        )
        reputation = _clamp(
            (0.42 * avg_quality)
            + (0.20 * security_coverage)
            + (0.14 * code_coverage)
            + (0.10 * task_coverage)
            + (0.06 * review_lower_bound)
            + (0.05 * license_trust)
            + (0.03 * benchmark_feedback_score)
            - (0.20 * contamination_rate)
            - (0.16 * duplicate_rate)
        )
        reasons: list[str] = []
        if avg_quality < 0.60:
            reasons.append("low_average_quality")
        if security_coverage < 0.12:
            reasons.append("low_security_coverage")
        if code_coverage < 0.08:
            reasons.append("low_code_coverage")
        if contamination_rate > 0:
            reasons.append("contamination_seen_in_run")
        if duplicate_rate > 0.15:
            reasons.append("high_duplicate_pressure")
        if approval_status != "approved" or trust_tier == "quarantine":
            reasons.append("source_not_governance_approved")
        if reviewed_records < minimum_reviewed_records:
            reasons.append("insufficient_independent_review_evidence")
        if approval_status != "approved":
            action = "block"
        elif trust_tier == "quarantine" or reviewed_records < minimum_reviewed_records:
            action = "quarantine"
        else:
            action = (
                "promote"
                if reputation >= 0.82
                else "watch"
                if reputation >= 0.64
                else "throttle"
                if reputation >= 0.46
                else "block"
            )
        scores.append(
            SourceReputationScore(
                source=source_name,
                rows=rows,
                avg_quality_score=round(avg_quality, 6),
                task_coverage=round(task_coverage, 6),
                security_coverage=round(security_coverage, 6),
                code_coverage=round(code_coverage, 6),
                review_approval_rate=round(review_approval_rate, 6),
                duplicate_rate=round(duplicate_rate, 6),
                contamination_rate=round(contamination_rate, 6),
                license_trust=round(license_trust, 6),
                benchmark_feedback_score=round(benchmark_feedback_score, 6),
                reputation_score=round(reputation, 6),
                reputation_lower_bound=round(review_lower_bound, 6),
                reviewed_records=reviewed_records,
                trust_tier=trust_tier,
                approval_status=approval_status,
                action=action,
                reasons=reasons,
            )
        )
    scores.sort(key=lambda item: item.reputation_score, reverse=True)
    return SourceReputationReport(sources=scores)


def write_source_reputation_report(
    output_path: str | Path, **kwargs: object
) -> SourceReputationReport:
    report = build_source_reputation_report(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Craftly source reputation report."
    )
    parser.add_argument("--source-quality-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-report")
    parser.add_argument("--review-report")
    parser.add_argument("--feedback-report")
    parser.add_argument("--contamination-report")
    parser.add_argument("--dedup-report")
    parser.add_argument("--source-registry")
    parser.add_argument("--minimum-reviewed-records", type=int, default=100)
    args = parser.parse_args()
    report = write_source_reputation_report(
        args.output,
        source_quality_report_path=args.source_quality_report,
        task_report_path=args.task_report,
        review_report_path=args.review_report,
        feedback_report_path=args.feedback_report,
        contamination_report_path=args.contamination_report,
        dedup_report_path=args.dedup_report,
        source_registry_path=args.source_registry,
        minimum_reviewed_records=args.minimum_reviewed_records,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
