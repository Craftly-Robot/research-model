"""Benchmark feedback loop for dataset quality and task promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel, write_report, print_report


class FeedbackRecommendation(StrictModel):
    kind: str
    severity: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkFeedbackReport(StrictModel):
    benchmark_score: float
    benchmark_total: int
    benchmark_passed: int
    task_automated_pass_rate: float
    avg_quality_score: float
    recommendations: list[FeedbackRecommendation] = Field(default_factory=list)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_feedback_report(
    *,
    benchmark_report_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
    review_report_path: str | Path | None = None,
) -> BenchmarkFeedbackReport:
    benchmark = load_json(benchmark_report_path) if benchmark_report_path else {}
    quality = load_json(quality_report_path) if quality_report_path else {}
    review = load_json(review_report_path) if review_report_path else {}
    benchmark_score = float(benchmark.get("score", 0.0))
    benchmark_total = int(benchmark.get("total", 0))
    benchmark_passed = int(benchmark.get("passed", 0))
    avg_quality_score = float(quality.get("avg_quality_score", 0.0))
    review_total = int(review.get("total", 0))
    automated_pass = int(review.get("automated_pass", 0))
    task_automated_pass_rate = automated_pass / max(1, review_total)
    automated_pass_by_type = dict(review.get("automated_pass_by_type") or {})
    task_type_count = len([value for value in automated_pass_by_type.values() if int(value) > 0])
    components = dict(quality.get("avg_component_scores") or {})

    recommendations: list[FeedbackRecommendation] = []
    if avg_quality_score < 0.65:
        recommendations.append(
            FeedbackRecommendation(
                kind="quality_gate",
                severity="high",
                message="Average quality score is low; tighten source allowlist and minimum quality thresholds.",
                metadata={"avg_quality_score": avg_quality_score},
            )
        )
    if review_total and task_automated_pass_rate < 0.5:
        recommendations.append(
            FeedbackRecommendation(
                kind="task_extraction",
                severity="high",
                message="Automated task pass rate is low; improve extraction rules and reject noisy sources.",
                metadata={"task_automated_pass_rate": task_automated_pass_rate},
            )
        )
    if automated_pass >= 20 and task_type_count < 3:
        recommendations.append(
            FeedbackRecommendation(
                kind="task_diversity",
                severity="medium",
                message="Automated-pass task mix is narrow; add sources that produce security, patch, debug, test, and implementation tasks.",
                metadata={"automated_pass_by_type": automated_pass_by_type},
            )
        )
    if (
        components
        and max(
            float(components.get("security_signal", 0.0)),
            float(components.get("agentic_signal", 0.0)),
        )
        < 0.2
    ):
        recommendations.append(
            FeedbackRecommendation(
                kind="signal_quality",
                severity="medium",
                message="Corpus component scores show weak security/agentic signal; increase high-value defensive and coding sources.",
                metadata={"avg_component_scores": components},
            )
        )
    if benchmark_total and benchmark_score < 0.75:
        recommendations.append(
            FeedbackRecommendation(
                kind="benchmark",
                severity="critical",
                message="Benchmark score is below promotion threshold; do not promote this dataset version to training.",
                metadata={"benchmark_score": benchmark_score, "benchmark_total": benchmark_total},
            )
        )
    if benchmark_total == 0:
        recommendations.append(
            FeedbackRecommendation(
                kind="benchmark_missing",
                severity="high",
                message="No governed benchmark result is bound to this dataset; training promotion is prohibited.",
            )
        )
    if automated_pass:
        recommendations.append(
            FeedbackRecommendation(
                kind="human_review_required",
                severity="high",
                message=(
                    "Automated policy checks are not human approval. Candidate tasks require the configured "
                    "independent review and Dataset Trust Authority promotion gates."
                ),
                metadata={"automated_pass_candidates": automated_pass},
            )
        )
    if not recommendations:
        recommendations.append(
            FeedbackRecommendation(
                kind="governance",
                severity="info",
                message="Automated feedback found no additional issue; only the Dataset Trust Authority can promote this version.",
            )
        )
    return BenchmarkFeedbackReport(
        benchmark_score=benchmark_score,
        benchmark_total=benchmark_total,
        benchmark_passed=benchmark_passed,
        task_automated_pass_rate=round(task_automated_pass_rate, 6),
        avg_quality_score=avg_quality_score,
        recommendations=recommendations,
    )


def write_feedback_report(
    *,
    output_path: str | Path,
    benchmark_report_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
    review_report_path: str | Path | None = None,
) -> BenchmarkFeedbackReport:
    report = build_feedback_report(
        benchmark_report_path=benchmark_report_path,
        quality_report_path=quality_report_path,
        review_report_path=review_report_path,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Craftly benchmark/data feedback report.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-report")
    parser.add_argument("--quality-report")
    parser.add_argument("--review-report")
    args = parser.parse_args()
    report = write_feedback_report(
        output_path=args.output,
        benchmark_report_path=args.benchmark_report,
        quality_report_path=args.quality_report,
        review_report_path=args.review_report,
    )
    print_report(report)


if __name__ == "__main__":
    main()
