"""Strict license filtering for Craftly training corpora."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl
from src.craftly.shared.schemas import StrictModel

DEFAULT_ALLOWED_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "mit",
    "mpl-2.0",
    "postgresql",
    "psf-2.0",
    "public-domain",
}


class LicenseDecision(StrictModel):
    accepted: bool
    license: str
    reason: str


class LicenseFilterReport(StrictModel):
    input_paths: list[str]
    output_path: str
    accepted: int
    rejected: int
    rejected_by_license: dict[str, int] = Field(default_factory=dict)
    allowed_licenses: list[str]
    strict_unknown: bool
    created_at_unix: float = Field(default_factory=time.time)


def normalize_license(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "unknown"}:
        return "unknown"
    return text


def decide_license(
    license_name: object,
    *,
    allowed_licenses: set[str] | None = None,
    strict_unknown: bool = True,
) -> LicenseDecision:
    allowed = allowed_licenses or DEFAULT_ALLOWED_LICENSES
    normalized = normalize_license(license_name)
    if normalized in allowed:
        return LicenseDecision(
            accepted=True, license=normalized, reason="approved_license"
        )
    if normalized in {"unknown", "unknown-ok"} and not strict_unknown:
        return LicenseDecision(
            accepted=True, license=normalized, reason="unknown_allowed_by_policy"
        )
    return LicenseDecision(
        accepted=False, license=normalized, reason="license_not_approved"
    )


def filter_jsonl_by_license(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    allowed_licenses: set[str] | None = None,
    strict_unknown: bool = True,
) -> LicenseFilterReport:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    rejected_by_license: dict[str, int] = {}
    allowed = allowed_licenses or DEFAULT_ALLOWED_LICENSES
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            for row in iter_jsonl(path):
                decision = decide_license(
                    row.get("license"),
                    allowed_licenses=allowed,
                    strict_unknown=strict_unknown,
                )
                if not decision.accepted:
                    rejected += 1
                    rejected_by_license[decision.license] = (
                        rejected_by_license.get(decision.license, 0) + 1
                    )
                    continue
                row["license"] = decision.license
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
    return LicenseFilterReport(
        input_paths=[str(path) for path in input_paths],
        output_path=str(target),
        accepted=accepted,
        rejected=rejected,
        rejected_by_license=dict(sorted(rejected_by_license.items())),
        allowed_licenses=sorted(allowed),
        strict_unknown=strict_unknown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Craftly JSONL rows by approved training-data license."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()
    report = filter_jsonl_by_license(
        args.input, args.output, strict_unknown=not args.allow_unknown_license
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
