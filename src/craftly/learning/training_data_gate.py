"""Strict promotion gate for Craftly training data.

This module sits after license/contamination/dedup filtering and before
tokenizer/shard construction. It promotes only high-signal rows into training,
separates evaluation holdout rows, and writes a human-review queue for valuable
but risky/uncertain material.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.shared.schemas import print_report
from src.craftly.shared.schemas import StrictModel, print_report

PATCH_DATA_TYPES = {"patch", "debug_trace"}
SECURITY_DATA_TYPES = {"security_advisory", "security_reference"}
HIGH_VALUE_LABELS = {
    "defensive_security",
    "code",
    "patch",
    "tests",
    "runtime_trace",
    "agentic_coding",
}
MANDATORY_REVIEW_LABELS = {"defensive_security", "patch", "tests", "runtime_trace"}
NOISE_FLAGS = {"navigation_or_boilerplate_noise", "repeated_lines", "heavy_boilerplate_noise"}


class TrainingDataGateConfig(StrictModel):
    min_quality_score: float = Field(default=0.58, ge=0.0, le=1.0)
    min_source_reputation_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_reputation_lower_bound: float = Field(default=0.70, ge=0.0, le=1.0)
    require_governed_sources: bool = False
    allow_governed_quarantine: bool = False
    reject_noise_flags: bool = True
    eval_holdout_fraction: float = Field(default=0.02, ge=0.0, le=0.5)
    seed: int = 1337
    patch_priority_bonus: float = Field(default=0.12, ge=0.0, le=0.5)
    security_priority_bonus: float = Field(default=0.08, ge=0.0, le=0.5)
    high_value_review_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    min_review_queue_score: float = Field(default=0.62, ge=0.0, le=1.0)
    routine_review_fraction: float = Field(default=0.03, ge=0.0, le=0.25)
    require_high_value_review: bool = False


class GateDecision(StrictModel):
    content_hash: str
    source: str
    status: str
    score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    source_reputation_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    priority_labels: list[str] = Field(default_factory=list)


class TrainingDataGateReport(StrictModel):
    input_paths: list[str]
    promoted_path: str
    holdout_path: str
    review_queue_path: str
    decisions_path: str
    scanned: int
    promoted: int
    holdout: int
    review_queue: int
    rejected: int
    by_status: dict[str, int] = Field(default_factory=dict)
    by_reason: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    by_data_type: dict[str, int] = Field(default_factory=dict)
    avg_promoted_score: float = 0.0
    created_at_unix: float = Field(default_factory=time.time)


def _load_reputation(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {str(item.get("source") or "unknown"): dict(item) for item in payload.get("sources", [])}


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    quality = row.get("quality", {})
    return quality if isinstance(quality, dict) else {}


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or row.get("prompt") or "")


def _inc(bucket: dict[str, int], key: str) -> None:
    bucket[key] = bucket.get(key, 0) + 1


def _priority_labels(quality: dict[str, Any]) -> list[str]:
    labels = {str(item) for item in quality.get("labels", [])}
    data_type = str(quality.get("data_type") or "")
    priorities = sorted(labels.intersection(HIGH_VALUE_LABELS))
    if data_type in PATCH_DATA_TYPES:
        priorities.append("patch_or_debug_trace")
    if data_type in SECURITY_DATA_TYPES:
        priorities.append("security_reference")
    return sorted(set(priorities))


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def has_independent_review(row: dict[str, Any]) -> bool:
    """Return whether two independent reviewers approved, directly or by adjudication."""
    review = row.get("human_review") if isinstance(row.get("human_review"), dict) else {}
    decisions = review.get("decisions") if isinstance(review.get("decisions"), list) else []
    reviewer_ids = {
        str(item.get("reviewer_id"))
        for item in decisions
        if isinstance(item, dict) and item.get("reviewer_id")
    }
    values = [
        str(item.get("decision"))
        for item in decisions
        if isinstance(item, dict) and item.get("decision") in {"approve", "reject"}
    ]
    if len(reviewer_ids) < 2 or len(values) < 2:
        return False
    if set(values) == {"approve"}:
        return True
    adjudication = (
        review.get("adjudication") if isinstance(review.get("adjudication"), dict) else {}
    )
    return (
        len(set(values)) > 1
        and adjudication.get("decision") == "approve"
        and adjudication.get("adjudicator_id") not in reviewer_ids
    )


def _deterministic_sample(content_hash: str, fraction: float, seed: int, stratum: str) -> bool:
    if fraction <= 0:
        return False
    value = int(stable_hash(f"{seed}:{stratum}:{content_hash}")[:16], 16) / float(1 << 64)
    return value < fraction


def score_row(
    row: dict[str, Any],
    *,
    reputation_by_source: dict[str, dict[str, Any]],
    config: TrainingDataGateConfig,
) -> GateDecision:
    quality = _quality(row)
    source = str(row.get("source") or "unknown")
    text = _text(row)
    digest = str(row.get("content_hash") or quality.get("content_hash") or stable_hash(text))
    quality_score = _bounded(float(quality.get("quality_score", 0.0)))
    source_evidence = reputation_by_source.get(source, {})
    reputation = _bounded(float(source_evidence.get("reputation_score", 0.5)))
    reputation_lower_bound = _bounded(float(source_evidence.get("reputation_lower_bound", 0.0)))
    source_action = str(source_evidence.get("action") or "unknown")
    source_approval = str(source_evidence.get("approval_status") or "pending")
    source_license_trust = _bounded(float(source_evidence.get("license_trust", 0.0)))
    data_type = str(quality.get("data_type") or "unknown")
    labels = {str(item) for item in quality.get("labels", [])}
    risk_flags = {str(item) for item in quality.get("risk_flags", [])}
    priorities = _priority_labels(quality)
    mandatory_review = data_type in PATCH_DATA_TYPES | SECURITY_DATA_TYPES or bool(
        labels & MANDATORY_REVIEW_LABELS
    )
    reasons: list[str] = []

    if quality_score < config.min_quality_score:
        reasons.append("quality_below_training_threshold")
    if reputation < config.min_source_reputation_score:
        reasons.append("source_reputation_below_threshold")
    if config.require_governed_sources:
        quarantine_allowed = (
            config.allow_governed_quarantine
            and source_action == "quarantine"
            and source_approval == "approved"
            and source_license_trust >= 1.0
        )
        if source_action not in {"promote", "watch"} and not quarantine_allowed:
            reasons.append(f"source_governance_action:{source_action}")
        if reputation_lower_bound < config.min_reputation_lower_bound and not quarantine_allowed:
            reasons.append("source_reputation_lower_bound_below_threshold")
    if (
        config.reject_noise_flags
        and risk_flags.intersection(NOISE_FLAGS)
        and quality_score < config.high_value_review_threshold
    ):
        reasons.append("boilerplate_or_low_signal_noise")
    if not text.strip():
        reasons.append("empty_text")

    score = (0.70 * quality_score) + (0.20 * reputation) + (0.10 * min(1.0, len(priorities) / 3.0))
    if data_type in PATCH_DATA_TYPES or "patch" in labels:
        score += config.patch_priority_bonus
    if data_type in SECURITY_DATA_TYPES or "defensive_security" in labels:
        score += config.security_priority_bonus
    if risk_flags.intersection(NOISE_FLAGS):
        score -= 0.10
    score = _bounded(score)

    if reasons:
        high_value_uncertain = score >= config.min_review_queue_score and bool(priorities)
        status = "review_queue" if high_value_uncertain else "rejected"
    elif (
        (config.require_high_value_review or config.require_governed_sources)
        and mandatory_review
        and not has_independent_review(row)
    ):
        status = "review_queue"
        reasons.append("independent_high_value_review_required")
    elif _deterministic_sample(
        digest,
        config.routine_review_fraction,
        config.seed,
        f"{source}:{data_type}",
    ) and not has_independent_review(row):
        status = "review_queue"
        reasons.append("routine_stratified_review_sample")
    else:
        status = "promoted"

    return GateDecision(
        content_hash=digest,
        source=source,
        status=status,
        score=round(score, 6),
        quality_score=round(quality_score, 6),
        source_reputation_score=round(reputation, 6),
        reasons=reasons,
        priority_labels=priorities,
    )


def apply_training_data_gate(
    *,
    input_paths: list[str | Path],
    promoted_path: str | Path,
    holdout_path: str | Path,
    review_queue_path: str | Path,
    decisions_path: str | Path,
    reputation_report_path: str | Path | None = None,
    config: TrainingDataGateConfig | None = None,
) -> TrainingDataGateReport:
    active_config = config or TrainingDataGateConfig()
    reputation_by_source = _load_reputation(reputation_report_path)
    rng = random.Random(active_config.seed)
    promoted_target = Path(promoted_path)
    holdout_target = Path(holdout_path)
    review_target = Path(review_queue_path)
    decisions_target = Path(decisions_path)
    for target in (promoted_target, holdout_target, review_target, decisions_target):
        target.parent.mkdir(parents=True, exist_ok=True)

    scanned = promoted = holdout = review_queue = rejected = 0
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_data_type: dict[str, int] = {}
    promoted_scores: list[float] = []
    seen: set[str] = set()

    with (
        promoted_target.open("w", encoding="utf-8") as promoted_handle,
        holdout_target.open("w", encoding="utf-8") as holdout_handle,
        review_target.open("w", encoding="utf-8") as review_handle,
        decisions_target.open("w", encoding="utf-8") as decisions_handle,
    ):
        for path in input_paths:
            for row in iter_jsonl(path):
                scanned += 1
                decision = score_row(
                    row, reputation_by_source=reputation_by_source, config=active_config
                )
                if decision.content_hash in seen:
                    decision = decision.model_copy(
                        update={
                            "status": "rejected",
                            "reasons": [*decision.reasons, "duplicate_after_gate"],
                        }
                    )
                seen.add(decision.content_hash)
                _inc(by_status, decision.status)
                _inc(by_source, decision.source)
                quality = _quality(row)
                _inc(by_data_type, str(quality.get("data_type") or "unknown"))
                for reason in decision.reasons:
                    _inc(by_reason, reason)
                row = dict(row)
                row["content_hash"] = decision.content_hash
                row["training_gate"] = decision.model_dump()
                decisions_handle.write(
                    json.dumps(decision.model_dump(), ensure_ascii=False, sort_keys=True) + "\n"
                )
                if decision.status == "promoted":
                    if (
                        active_config.eval_holdout_fraction > 0
                        and rng.random() < active_config.eval_holdout_fraction
                    ):
                        row["train_policy"] = "eval_holdout"
                        holdout_handle.write(
                            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        holdout += 1
                    else:
                        row["train_policy"] = "train"
                        promoted_handle.write(
                            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        promoted += 1
                        promoted_scores.append(decision.score)
                elif decision.status == "review_queue":
                    row["train_policy"] = "human_review_required"
                    review_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    review_queue += 1
                else:
                    rejected += 1

    return TrainingDataGateReport(
        input_paths=[str(path) for path in input_paths],
        promoted_path=str(promoted_target),
        holdout_path=str(holdout_target),
        review_queue_path=str(review_target),
        decisions_path=str(decisions_target),
        scanned=scanned,
        promoted=promoted,
        holdout=holdout,
        review_queue=review_queue,
        rejected=rejected,
        by_status=dict(sorted(by_status.items())),
        by_reason=dict(sorted(by_reason.items())),
        by_source=dict(sorted(by_source.items())),
        by_data_type=dict(sorted(by_data_type.items())),
        avg_promoted_score=round(sum(promoted_scores) / max(1, len(promoted_scores)), 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote only high-quality Craftly rows into training."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--promoted-out", required=True)
    parser.add_argument("--holdout-out", required=True)
    parser.add_argument("--review-out", required=True)
    parser.add_argument("--decisions-out", required=True)
    parser.add_argument("--reputation-report")
    parser.add_argument("--min-quality-score", type=float, default=0.58)
    parser.add_argument("--min-source-reputation-score", type=float, default=0.45)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.02)
    args = parser.parse_args()
    report = apply_training_data_gate(
        input_paths=args.input,
        promoted_path=args.promoted_out,
        holdout_path=args.holdout_out,
        review_queue_path=args.review_out,
        decisions_path=args.decisions_out,
        reputation_report_path=args.reputation_report,
        config=TrainingDataGateConfig(
            min_quality_score=args.min_quality_score,
            min_source_reputation_score=args.min_source_reputation_score,
            eval_holdout_fraction=args.eval_holdout_fraction,
        ),
    )
    print_report(report)


if __name__ == "__main__":
    main()
