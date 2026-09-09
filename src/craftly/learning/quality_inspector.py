"""Dataset quality inspection reports for clean Craftly JSONL shards."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl
from src.craftly.shared.schemas import StrictModel


class QualityInspectionReport(StrictModel):
    input_paths: list[str]
    rows: int
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    by_label: dict[str, int] = Field(default_factory=dict)
    by_language: dict[str, int] = Field(default_factory=dict)
    by_data_type: dict[str, int] = Field(default_factory=dict)
    by_license: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    by_risk_flag: dict[str, int] = Field(default_factory=dict)
    avg_component_scores: dict[str, float] = Field(default_factory=dict)


def _inc(bucket: dict[str, int], key: str | None) -> None:
    active = key or "unknown"
    bucket[active] = bucket.get(active, 0) + 1


def inspect_clean_jsonl(paths: list[str | Path]) -> QualityInspectionReport:
    scores: list[float] = []
    by_label: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_data_type: dict[str, int] = {}
    by_license: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_risk_flag: dict[str, int] = {}
    component_totals: dict[str, float] = {}
    component_counts: dict[str, int] = {}
    rows = 0
    for path in paths:
        for row in iter_jsonl(path):
            rows += 1
            quality = row.get("quality", {})
            scores.append(float(quality.get("quality_score", 0.0)))
            for label in quality.get("labels", []):
                _inc(by_label, str(label))
            for risk_flag in quality.get("risk_flags", []):
                _inc(by_risk_flag, str(risk_flag))
            for name, value in dict(quality.get("component_scores") or {}).items():
                component_totals[str(name)] = component_totals.get(
                    str(name), 0.0
                ) + float(value)
                component_counts[str(name)] = component_counts.get(str(name), 0) + 1
            _inc(by_language, quality.get("language_hint"))
            _inc(by_data_type, quality.get("data_type"))
            _inc(by_license, str(row.get("license") or "unknown"))
            _inc(by_source, str(row.get("source") or "unknown"))
    if not scores:
        scores = [0.0]
    return QualityInspectionReport(
        input_paths=[str(path) for path in paths],
        rows=rows,
        avg_quality_score=round(statistics.mean(scores), 6),
        min_quality_score=round(min(scores), 6),
        max_quality_score=round(max(scores), 6),
        by_label=dict(sorted(by_label.items())),
        by_language=dict(sorted(by_language.items())),
        by_data_type=dict(sorted(by_data_type.items())),
        by_license=dict(sorted(by_license.items())),
        by_source=dict(sorted(by_source.items())),
        by_risk_flag=dict(sorted(by_risk_flag.items())),
        avg_component_scores={
            key: round(component_totals[key] / max(1, component_counts.get(key, 0)), 6)
            for key in sorted(component_totals)
        },
    )


def write_quality_report(
    paths: list[str | Path], output_path: str | Path
) -> QualityInspectionReport:
    report = inspect_clean_jsonl(paths)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect clean Craftly dataset JSONL shards."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_quality_report(args.input, args.output)
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
