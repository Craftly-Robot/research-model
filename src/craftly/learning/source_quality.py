"""Source quality scoring from clean corpus inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl
from src.craftly.shared.schemas import write_report, print_report
from src.craftly.shared.schemas import StrictModel, write_report, print_report


class SourceQualityScore(StrictModel):
    source: str
    rows: int
    avg_quality_score: float
    defensive_security_rows: int = 0
    code_rows: int = 0
    score: float = Field(ge=0.0, le=1.0)
    action: str


class SourceQualityReport(StrictModel):
    input_paths: list[str]
    sources: list[SourceQualityScore]


def build_source_quality_report(paths: list[str | Path]) -> SourceQualityReport:
    buckets: dict[str, dict[str, float]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            source = str(row.get("source") or "unknown")
            quality = row.get("quality", {})
            labels = set(quality.get("labels", []))
            bucket = buckets.setdefault(
                source, {"rows": 0.0, "quality": 0.0, "security": 0.0, "code": 0.0}
            )
            bucket["rows"] += 1
            bucket["quality"] += float(quality.get("quality_score", 0.0))
            bucket["security"] += 1 if "defensive_security" in labels else 0
            bucket["code"] += 1 if "code" in labels else 0
    scores: list[SourceQualityScore] = []
    for source, bucket in sorted(buckets.items()):
        rows = int(bucket["rows"])
        avg = bucket["quality"] / max(1, rows)
        coverage_bonus = min(0.2, ((bucket["security"] + bucket["code"]) / max(1, rows)) * 0.2)
        score = max(0.0, min(1.0, avg + coverage_bonus))
        action = (
            "promote" if score >= 0.75 and rows >= 1 else "watch" if score >= 0.55 else "demote"
        )
        scores.append(
            SourceQualityScore(
                source=source,
                rows=rows,
                avg_quality_score=round(avg, 6),
                defensive_security_rows=int(bucket["security"]),
                code_rows=int(bucket["code"]),
                score=round(score, 6),
                action=action,
            )
        )
    return SourceQualityReport(input_paths=[str(path) for path in paths], sources=scores)


def write_source_quality_report(
    paths: list[str | Path], output_path: str | Path
) -> SourceQualityReport:
    report = build_source_quality_report(paths)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Craftly data sources from clean JSONL shards."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_source_quality_report(args.input, args.output)
    print_report(report)


if __name__ == "__main__":
    main()
