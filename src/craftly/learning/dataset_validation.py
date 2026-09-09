"""Streaming dataset validation for large Craftly training corpora."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.shared.schemas import StrictModel


class DatasetValidationConfig(StrictModel):
    input_paths: list[str]
    min_records: int = Field(default=100_000, ge=1)
    max_duplicate_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    min_avg_chars: int = Field(default=80, ge=1)
    require_license: bool = True
    require_quality: bool = True
    require_categories: list[str] = Field(
        default_factory=lambda: ["general", "code", "cybersecurity"]
    )
    holdout_policies: list[str] = Field(
        default_factory=lambda: ["eval_holdout", "benchmark_holdout"]
    )


class DatasetValidationIssue(StrictModel):
    severity: str
    code: str
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class DatasetValidationReport(StrictModel):
    status: str
    input_paths: list[str]
    total_records: int
    duplicate_records: int
    duplicate_fraction: float
    avg_chars: float
    categories: dict[str, int]
    licenses: dict[str, int]
    holdout_rows: int
    train_rows: int
    issue_count: int
    issues: list[DatasetValidationIssue]
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "dataset_validation_report.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "dataset_validation_report.md")
        return target


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or row.get("prompt") or "")


def _row_category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
    category = str(
        row.get("category") or metadata.get("category") or quality.get("data_type") or "general"
    ).lower()
    labels = {str(item).lower() for item in quality.get("labels", [])}
    if "cyber" in category or "security" in category or "defensive_security" in labels:
        return "cybersecurity"
    if (
        "code" in category
        or "code" in labels
        or any(marker in _row_text(row) for marker in ("def ", "class ", "fn ", "function "))
    ):
        return "code"
    if "agentic" in category or "agentic" in labels:
        return "agentic"
    return "general"


def validate_dataset(config: DatasetValidationConfig) -> DatasetValidationReport:
    seen: set[str] = set()
    duplicate_records = 0
    total_records = 0
    total_chars = 0
    missing_license = 0
    missing_quality = 0
    categories: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    train_rows = 0
    holdout_rows = 0
    holdout_policies = {item.lower() for item in config.holdout_policies}
    for path in config.input_paths:
        for row in iter_jsonl(path):
            total_records += 1
            text = _row_text(row)
            total_chars += len(text)
            digest = str(row.get("content_hash") or stable_hash(text))
            if digest in seen:
                duplicate_records += 1
            else:
                seen.add(digest)
            category = _row_category(row)
            categories[category] += 1
            license_name = str(row.get("license") or row.get("spdx_license") or "unknown").lower()
            licenses[license_name] += 1
            if config.require_license and license_name in {"", "unknown", "none"}:
                missing_license += 1
            if config.require_quality and not isinstance(row.get("quality"), dict):
                missing_quality += 1
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            policy = str(row.get("train_policy") or metadata.get("train_policy") or "").lower()
            if policy in holdout_policies:
                holdout_rows += 1
            else:
                train_rows += 1
    avg_chars = total_chars / max(1, total_records)
    duplicate_fraction = duplicate_records / max(1, total_records)
    issues: list[DatasetValidationIssue] = []
    if total_records < config.min_records:
        issues.append(
            DatasetValidationIssue(
                severity="fail",
                code="record_count_below_minimum",
                message=f"dataset has {total_records} records, below required {config.min_records}",
            )
        )
    if duplicate_fraction > config.max_duplicate_fraction:
        issues.append(
            DatasetValidationIssue(
                severity="fail",
                code="duplicate_fraction_too_high",
                message="duplicate fraction exceeds configured maximum",
                metrics={
                    "duplicate_fraction": duplicate_fraction,
                    "limit": config.max_duplicate_fraction,
                },
            )
        )
    if avg_chars < config.min_avg_chars:
        issues.append(
            DatasetValidationIssue(
                severity="warn",
                code="average_text_too_short",
                message="average text length is low for pretraining quality",
                metrics={"avg_chars": avg_chars, "minimum": config.min_avg_chars},
            )
        )
    if missing_license:
        issues.append(
            DatasetValidationIssue(
                severity="fail",
                code="missing_license",
                message="rows are missing explicit license metadata",
                metrics={"missing_license": missing_license},
            )
        )
    if missing_quality:
        issues.append(
            DatasetValidationIssue(
                severity="warn",
                code="missing_quality_metadata",
                message="rows are missing quality metadata",
                metrics={"missing_quality": missing_quality},
            )
        )
    missing_categories = [
        category for category in config.require_categories if categories.get(category, 0) == 0
    ]
    if missing_categories:
        issues.append(
            DatasetValidationIssue(
                severity="fail",
                code="missing_required_categories",
                message="dataset is missing required category coverage",
                metrics={"missing_categories": missing_categories},
            )
        )
    status = "failed" if any(issue.severity == "fail" for issue in issues) else "passed"
    return DatasetValidationReport(
        status=status,
        input_paths=config.input_paths,
        total_records=total_records,
        duplicate_records=duplicate_records,
        duplicate_fraction=round(duplicate_fraction, 6),
        avg_chars=round(avg_chars, 3),
        categories=dict(sorted(categories.items())),
        licenses=dict(sorted(licenses.items())),
        holdout_rows=holdout_rows,
        train_rows=train_rows,
        issue_count=len(issues),
        issues=issues,
    )


def write_markdown(report: DatasetValidationReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Dataset Validation Report",
        "",
        f"- status: {report.status}",
        f"- total_records: {report.total_records}",
        f"- duplicate_fraction: {report.duplicate_fraction:.4f}",
        f"- avg_chars: {report.avg_chars:.1f}",
        f"- train_rows: {report.train_rows}",
        f"- holdout_rows: {report.holdout_rows}",
        "",
        "| severity | code | message |",
        "|---|---|---|",
    ]
    for issue in report.issues:
        lines.append(f"| {issue.severity} | {issue.code} | {issue.message} |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a large Craftly JSONL training corpus.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-records", type=int, default=100_000)
    parser.add_argument("--max-duplicate-fraction", type=float, default=0.02)
    parser.add_argument("--min-avg-chars", type=int, default=80)
    parser.add_argument("--allow-missing-license", action="store_true")
    parser.add_argument("--allow-missing-quality", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_dataset(
        DatasetValidationConfig(
            input_paths=args.inputs,
            min_records=args.min_records,
            max_duplicate_fraction=args.max_duplicate_fraction,
            min_avg_chars=args.min_avg_chars,
            require_license=not args.allow_missing_license,
            require_quality=not args.allow_missing_quality,
        )
    )
    report.write(args.output_dir)
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
