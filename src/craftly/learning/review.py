"""Automated policy screening for extracted Craftly task candidates.

This module does not perform human review and cannot promote training data.
Passing rows remain candidates until the Dataset Trust Authority completes the
configured independent human-review and evidence gates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.shared.schemas import StrictModel


HIGH_RISK_ACTION_TERMS = {
    "deploy malware",
    "steal",
    "credential theft",
    "phishing",
    "ransomware",
    "persistence",
    "evade detection",
}


class ReviewDecision(StrictModel):
    task_id: str
    status: Literal["automated_pass", "needs_human_review", "rejected"]
    reviewer: str = "automated-policy"
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=time.time)


class ReviewReport(StrictModel):
    input_path: str
    decisions_path: str
    automated_pass_path: str
    review_mode: Literal["automated_policy"] = "automated_policy"
    total: int
    automated_pass: int
    human_approved: int = 0
    needs_human_review: int
    rejected: int
    by_type: dict[str, int] = Field(default_factory=dict)
    automated_pass_by_type: dict[str, int] = Field(default_factory=dict)
    avg_score: float = 0.0
    created_at_unix: float = Field(default_factory=time.time)


def review_task(task: dict[str, Any]) -> ReviewDecision:
    prompt = str(task.get("prompt") or "")
    lowered = prompt.lower()
    reasons: list[str] = []
    score = 0.5
    if len(prompt) < 120:
        reasons.append("prompt_too_short")
        score -= 0.2
    if "safe" in lowered or "defensive" in lowered or "secure" in lowered:
        score += 0.2
    if task.get("source_url"):
        score += 0.1
    if task.get("language"):
        score += 0.1
    if any(term in lowered for term in HIGH_RISK_ACTION_TERMS):
        reasons.append("high_risk_action_request")
        score -= 0.6
    task_type = str(task.get("task_type") or "")
    metadata = dict(task.get("metadata") or {})
    training_priority = str(metadata.get("training_priority") or "normal")
    if task_type in {
        "debugging_from_error_trace",
        "implementation_planning",
        "regression_test_generation",
        "secure_code_review",
        "security_patch_generation",
        "security_vulnerability_identification",
    }:
        score += 0.1
    score = max(0.0, min(1.0, score))
    if "high_risk_action_request" in reasons:
        status: Literal["automated_pass", "needs_human_review", "rejected"] = "rejected"
    elif score >= 0.75 and not reasons:
        status = "automated_pass"
    elif training_priority == "critical" and score >= 0.50:
        status = "needs_human_review"
    elif score >= 0.55:
        status = "needs_human_review"
    else:
        status = "rejected"
    return ReviewDecision(
        task_id=str(task.get("task_id") or f"task-{stable_hash(prompt)[:16]}"),
        status=status,
        score=score,
        reasons=reasons,
        source_url=task.get("source_url"),
        metadata={"task_type": task_type, "language": task.get("language"), **metadata},
    )


def review_tasks(
    input_path: str | Path,
    decisions_path: str | Path,
    automated_pass_path: str | Path,
) -> ReviewReport:
    decisions_target = Path(decisions_path)
    automated_pass_target = Path(automated_pass_path)
    decisions_target.parent.mkdir(parents=True, exist_ok=True)
    automated_pass_target.parent.mkdir(parents=True, exist_ok=True)
    total = automated_pass = human = rejected = 0
    by_type: dict[str, int] = {}
    automated_pass_by_type: dict[str, int] = {}
    scores: list[float] = []
    with (
        decisions_target.open("w", encoding="utf-8") as decisions_handle,
        automated_pass_target.open("w", encoding="utf-8") as automated_pass_handle,
    ):
        for task in iter_jsonl(input_path):
            total += 1
            task_type = str(task.get("task_type") or "unknown")
            by_type[task_type] = by_type.get(task_type, 0) + 1
            decision = review_task(task)
            scores.append(decision.score)
            decisions_handle.write(json.dumps(decision.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")
            if decision.status == "automated_pass":
                automated_pass += 1
                automated_pass_by_type[task_type] = automated_pass_by_type.get(task_type, 0) + 1
                task["automated_policy_review"] = decision.model_dump()
                automated_pass_handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
            elif decision.status == "needs_human_review":
                human += 1
            else:
                rejected += 1
    return ReviewReport(
        input_path=str(input_path),
        decisions_path=str(decisions_target),
        automated_pass_path=str(automated_pass_target),
        total=total,
        automated_pass=automated_pass,
        needs_human_review=human,
        rejected=rejected,
        by_type=dict(sorted(by_type.items())),
        automated_pass_by_type=dict(sorted(automated_pass_by_type.items())),
        avg_score=round(sum(scores) / max(1, len(scores)), 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply automated policy screening to extracted Craftly task candidates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--decisions-out", required=True)
    parser.add_argument("--automated-pass-out", required=True)
    parser.add_argument("--report-out")
    args = parser.parse_args()
    report = review_tasks(args.input, args.decisions_out, args.automated_pass_out)
    if args.report_out:
        report_target = Path(args.report_out)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

