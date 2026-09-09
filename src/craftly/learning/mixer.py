"""Token-aware data mixing controller for Craftly training corpora."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.model_ops.tokenizer_pipeline import (
    ShardBuildConfig,
    ShardManifest,
    build_token_shards,
    load_tokenizer,
)
from src.craftly.shared.config_contracts import (
    MixRatiosContract as MixConfig,
)
from src.craftly.shared.config_contracts import (
    load_mix_ratios_contract,
)
from src.craftly.shared.schemas import StrictModel

SCRATCH_INSTRUCTION_RATIOS = {
    "instruction_security_coding": 0.40,
    "verified_patch_tests": 0.30,
    "high_quality_docs_code": 0.20,
    "debugging_error_logs": 0.10,
}

CURRICULUM_RATIOS = {
    "balanced": SCRATCH_INSTRUCTION_RATIOS,
    "fundamentals_only": {
        "instruction_security_coding": 0.0,
        "verified_patch_tests": 0.0,
        "high_quality_docs_code": 1.0,
        "debugging_error_logs": 0.0,
    },
    "defensive_security_only": {
        "instruction_security_coding": 1.0,
        "verified_patch_tests": 0.0,
        "high_quality_docs_code": 0.0,
        "debugging_error_logs": 0.0,
    },
    "debug_patch_only": {
        "instruction_security_coding": 0.0,
        "verified_patch_tests": 0.65,
        "high_quality_docs_code": 0.0,
        "debugging_error_logs": 0.35,
    },
    "agentic_coding_only": {
        "instruction_security_coding": 0.45,
        "verified_patch_tests": 0.35,
        "high_quality_docs_code": 0.20,
        "debugging_error_logs": 0.0,
    },
}

OFFENSIVE_MISUSE_PATTERNS = [
    r"\breverse\s+shell\b",
    r"\bshellcode\b",
    r"\bmetasploit\b",
    r"\bweaponiz(?:e|ed|ation)\b",
    r"\bexfiltrat(?:e|ion)\b",
    r"\bsteal\s+(?:cookies|tokens|passwords|credentials)\b",
    r"\bcredential\s+dump(?:ing)?\b",
    r"\bc2\s+(?:server|callback|beacon)\b",
    r"\bransomware\b",
    r"\bpersistence\s+mechanism\b",
    r"\bexploit\s+(?:chain|payload|code)\b",
    r"\bpayload\s+(?:to|that)\s+(?:execute|run|spawn|bypass)\b",
    r"\bbypass\s+(?:edr|antivirus|av|waf)\b",
]


class MixBucketReport(StrictModel):
    bucket: str
    input_rows: int
    output_rows: int
    output_tokens: int
    target_ratio: float
    actual_ratio: float


class MixManifest(StrictModel):
    experiment: str
    input_paths: list[str]
    output_jsonl: str
    shard_manifest: dict[str, Any] | None = None
    tokenizer_path: str | None = None
    total_rows: int
    total_tokens: int
    buckets: list[MixBucketReport]
    excluded_holdout_rows: int
    created_at_unix: float = Field(default_factory=time.time)


class ScratchMixConfig(StrictModel):
    ratios: dict[str, float] = Field(
        default_factory=lambda: SCRATCH_INSTRUCTION_RATIOS.copy()
    )
    curriculum_mode: str = Field(
        default="balanced",
        pattern="^(balanced|fundamentals_only|defensive_security_only|debug_patch_only|agentic_coding_only)$",
    )
    seed: int = 1337
    max_rows: int | None = Field(default=None, ge=1)
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    preserve_raw_docs: bool = True
    strict_offensive_filter: bool = True

    @model_validator(mode="after")
    def validate_scratch_ratios(self) -> "ScratchMixConfig":
        missing = set(SCRATCH_INSTRUCTION_RATIOS) - set(self.ratios)
        if missing:
            raise ValueError(f"missing scratch mix buckets: {sorted(missing)}")
        total = sum(float(value) for value in self.ratios.values())
        if total <= 0:
            raise ValueError("scratch mix ratios must sum to a positive value")
        return self


class ScratchMixBucketReport(StrictModel):
    bucket: str
    input_rows: int
    output_rows: int
    output_tokens: int
    target_ratio: float
    actual_ratio: float


class ScratchInstructionMixReport(StrictModel):
    status: str
    input_paths: list[str]
    output_jsonl: str
    curriculum_mode: str
    total_rows: int
    total_tokens: int
    buckets: list[ScratchMixBucketReport]
    rejected_rows: int
    offensive_rejected_rows: int = 0
    strict_offensive_filter: bool = True
    transformed_instruction_rows: int
    raw_preserved_rows: int
    target_ratios: dict[str, float]
    recommendations: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)


def load_mix_config(path: str | Path) -> MixConfig:
    return load_mix_ratios_contract(path)


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or row.get("prompt") or "")


def classify_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    policy = str(row.get("train_policy") or metadata.get("train_policy") or "").lower()
    if policy in {"eval_holdout", "benchmark_holdout"}:
        return "holdout"
    category = str(row.get("category") or metadata.get("category") or "").lower()
    quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
    labels = {str(item).lower() for item in quality.get("labels", [])}
    text = _row_text(row).lower()
    if "agentic" in category or "agentic" in labels or "task_type" in row:
        return "agentic"
    if "cyber" in category or "security" in category or "defensive_security" in labels:
        return "cybersecurity"
    if (
        "code" in category
        or "code" in labels
        or any(
            token in text
            for token in ("def ", "class ", "function ", "fn ", "package ")
        )
    ):
        return "code"
    return "general"


def _estimate_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return max(1, len(tokenizer.encode(text).ids))
    return max(1, len(text.split()))


def _load_rows(
    input_paths: list[str | Path],
    tokenizer: Any | None,
    holdout_policies: set[str],
    *,
    min_quality_score: float,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "general": [],
        "code": [],
        "cybersecurity": [],
        "agentic": [],
    }
    excluded = 0
    seen: set[str] = set()
    for path in input_paths:
        for row in iter_jsonl(path):
            metadata = (
                row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            )
            policy = str(
                row.get("train_policy") or metadata.get("train_policy") or ""
            ).lower()
            bucket = classify_row(row)
            if bucket == "holdout" or policy in holdout_policies:
                excluded += 1
                continue
            quality = (
                row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
            )
            training_gate = (
                row.get("training_gate", {})
                if isinstance(row.get("training_gate"), dict)
                else {}
            )
            quality_score = max(
                float(quality.get("quality_score", 0.0)),
                float(training_gate.get("score", 0.0)),
            )
            if quality_score < min_quality_score:
                excluded += 1
                continue
            text = _row_text(row)
            if not text:
                continue
            digest = str(row.get("content_hash") or stable_hash(text))
            if digest in seen:
                continue
            seen.add(digest)
            row = dict(row)
            row["content_hash"] = digest
            row["_mix_bucket"] = bucket
            row["_estimated_tokens"] = _estimate_tokens(text, tokenizer)
            buckets.setdefault(bucket, []).append(row)
    return buckets, excluded


def _normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in ratios.values()) or 1.0
    return {key: max(0.0, float(value)) / total for key, value in ratios.items()}


def curriculum_ratios(
    mode: str, custom_ratios: dict[str, float] | None = None
) -> dict[str, float]:
    if mode == "balanced":
        return custom_ratios or SCRATCH_INSTRUCTION_RATIOS.copy()
    if mode not in CURRICULUM_RATIOS:
        raise ValueError(f"unknown curriculum mode: {mode}")
    return CURRICULUM_RATIOS[mode].copy()


def contains_offensive_misuse(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in OFFENSIVE_MISUSE_PATTERNS)


def bucket_allowed_for_curriculum(bucket: str, mode: str) -> bool:
    ratios = curriculum_ratios(mode)
    return ratios.get(bucket, 0.0) > 0.0


def _sample_mixed_rows(
    *,
    buckets: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    rng: random.Random,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    normalized = _normalize_ratios(ratios)
    for rows in buckets.values():
        rng.shuffle(rows)
    cursors = {bucket: 0 for bucket in buckets}
    token_totals = {bucket: 0 for bucket in buckets}
    output: list[dict[str, Any]] = []
    total_available = sum(len(rows) for rows in buckets.values())
    limit = min(max_rows or total_available, total_available)
    active_buckets = [bucket for bucket, rows in buckets.items() if rows]
    while len(output) < limit and active_buckets:
        choices = [
            bucket
            for bucket in active_buckets
            if cursors[bucket] < len(buckets[bucket])
        ]
        if not choices:
            break
        current_total_tokens = max(1, sum(token_totals.values()))
        deficits = []
        for candidate in choices:
            target_ratio = normalized.get(candidate, 0.0)
            actual_ratio = token_totals[candidate] / current_total_tokens
            deficits.append((target_ratio - actual_ratio, rng.random(), candidate))
        deficits.sort(reverse=True)
        bucket = deficits[0][2]
        row = buckets[bucket][cursors[bucket]]
        cursors[bucket] += 1
        token_totals[bucket] += int(row.get("_estimated_tokens", 1))
        output.append(row)
        active_buckets = [
            item for item in active_buckets if cursors[item] < len(buckets[item])
        ]
    return output


def build_mix(
    *,
    input_paths: list[str | Path],
    config_path: str | Path,
    experiment: str,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    shard_token_count: int = 1_000_000,
    sequence_length: int = 2048,
    validation_fraction: float = 0.01,
) -> MixManifest:
    config = load_mix_config(config_path)
    active_experiment = next(
        (item for item in config.experiments if item.name == experiment), None
    )
    if active_experiment is None:
        raise ValueError(f"unknown mix experiment: {experiment}")
    active_tokenizer_path = str(tokenizer_path or config.tokenizer_path or "") or None
    tokenizer = load_tokenizer(active_tokenizer_path) if active_tokenizer_path else None
    buckets, excluded = _load_rows(
        input_paths,
        tokenizer,
        set(config.holdout_policies),
        min_quality_score=config.min_quality_score,
    )
    rng = random.Random(config.seed)
    mixed_rows = _sample_mixed_rows(
        buckets=buckets,
        ratios=active_experiment.ratios,
        rng=rng,
        max_rows=config.max_rows,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_jsonl = root / f"{experiment}.mixed.jsonl"
    bucket_stats: dict[str, dict[str, float]] = {
        bucket: {
            "input_rows": float(len(rows)),
            "output_rows": 0.0,
            "output_tokens": 0.0,
        }
        for bucket, rows in buckets.items()
    }
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in mixed_rows:
            bucket = str(row.pop("_mix_bucket"))
            tokens = int(row.pop("_estimated_tokens"))
            bucket_stats[bucket]["output_rows"] += 1
            bucket_stats[bucket]["output_tokens"] += tokens
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    total_tokens = int(sum(item["output_tokens"] for item in bucket_stats.values()))
    normalized = _normalize_ratios(active_experiment.ratios)
    reports = [
        MixBucketReport(
            bucket=bucket,
            input_rows=int(stats["input_rows"]),
            output_rows=int(stats["output_rows"]),
            output_tokens=int(stats["output_tokens"]),
            target_ratio=round(normalized.get(bucket, 0.0), 6),
            actual_ratio=round(stats["output_tokens"] / max(1, total_tokens), 6),
        )
        for bucket, stats in sorted(bucket_stats.items())
    ]
    shard_manifest: ShardManifest | None = None
    if active_tokenizer_path:
        shard_manifest = build_token_shards(
            input_paths=[output_jsonl],
            tokenizer_path=active_tokenizer_path,
            output_dir=root / "shards",
            config=ShardBuildConfig(
                shard_token_count=shard_token_count,
                sequence_length=sequence_length,
                validation_fraction=validation_fraction,
            ),
            dataset_id=f"craftly-mix-{experiment}",
        )
    manifest = MixManifest(
        experiment=experiment,
        input_paths=[str(path) for path in input_paths],
        output_jsonl=str(output_jsonl),
        shard_manifest=shard_manifest.model_dump() if shard_manifest else None,
        tokenizer_path=active_tokenizer_path,
        total_rows=len(mixed_rows),
        total_tokens=total_tokens,
        buckets=reports,
        excluded_holdout_rows=excluded,
    )
    (root / "mix_manifest.json").write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    quality = row.get("quality", {})
    return quality if isinstance(quality, dict) else {}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _training_gate(row: dict[str, Any]) -> dict[str, Any]:
    gate = row.get("training_gate", {})
    return gate if isinstance(gate, dict) else {}


def _labels(row: dict[str, Any]) -> set[str]:
    quality = _quality(row)
    metadata = _metadata(row)
    labels = {str(item).lower() for item in quality.get("labels", [])}
    labels.update(
        str(item).lower()
        for item in metadata.get("labels", [])
        if isinstance(metadata.get("labels", []), list)
    )
    category = str(row.get("category") or metadata.get("category") or "").lower()
    if category:
        labels.add(category)
    return labels


def _data_type(row: dict[str, Any]) -> str:
    return str(
        _quality(row).get("data_type") or _metadata(row).get("data_type") or ""
    ).lower()


def _source(row: dict[str, Any]) -> str:
    return str(row.get("source") or _metadata(row).get("source") or "approved-source")


def _license(row: dict[str, Any]) -> str:
    return str(row.get("license") or _metadata(row).get("license") or "unknown")


def _quality_score(row: dict[str, Any]) -> float:
    quality = _quality(row)
    gate = _training_gate(row)
    return max(
        float(quality.get("quality_score", 0.0) or 0.0),
        float(gate.get("score", 0.0) or 0.0),
    )


def classify_scratch_bucket(row: dict[str, Any]) -> str:
    text = _row_text(row).lower()
    labels = _labels(row)
    data_type = _data_type(row)
    if (
        "defensive_security" in labels
        or data_type in {"security_advisory", "security_reference"}
        or any(
            term in text
            for term in (
                "cwe-",
                "cve-",
                "vulnerability",
                "secure coding",
                "owasp",
                "mitigation",
                "authentication",
                "authorization",
            )
        )
    ):
        return "instruction_security_coding"
    if data_type == "debug_trace" or any(
        term in text
        for term in (
            "traceback",
            "compile error",
            "undefined reference",
            "stack trace",
            "panic:",
            "exception",
        )
    ):
        return "debugging_error_logs"
    if data_type in {"patch", "test"} or any(
        term in text
        for term in (
            "diff --git",
            "\n+++ ",
            "\n--- ",
            "regression test",
            "pytest",
            "unittest",
            "assert ",
        )
    ):
        return "verified_patch_tests"
    if "agentic_coding" in labels:
        return "instruction_security_coding"
    return "high_quality_docs_code"


def _sentence_excerpt(text: str, *, limit: int = 1400) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]


def convert_to_instruction_row(row: dict[str, Any], *, bucket: str) -> dict[str, Any]:
    source_text = _row_text(row)
    excerpt = _sentence_excerpt(source_text)
    source = _source(row)
    digest = str(row.get("content_hash") or stable_hash(source_text))
    language = str(
        _quality(row).get("language_hint") or row.get("language") or "unknown"
    )
    prompt = "Analyze this approved source and produce a defensive coding answer."
    correct_answer = "Summarize the risk or engineering task, propose a safe implementation direction, and include verification."
    patch = "No direct patch is present. Provide a patch plan tied to the context."
    tests = "Verify with regression tests, static checks, and compile/runtime checks where applicable."
    verification = (
        "expected: safe defensive reasoning with no live-target attack workflow"
    )

    if bucket == "instruction_security_coding":
        prompt = "Turn this approved cybersecurity/coding context into a defensive instruction response."
        correct_answer = (
            "Identify vulnerability surfaces or implementation requirements, explain the defensive rationale, "
            "avoid unsupported claims, and propose safe code or architecture changes."
        )
        patch = "Patch plan: validate inputs, preserve behavior, add narrow checks, and document security assumptions."
        tests = "Tests: include malicious-input regression, normal behavior regression, and static-analysis verification."
    elif bucket == "verified_patch_tests":
        prompt = "Extract a patch-generation training example from this approved patch/test context."
        correct_answer = "Explain the bug, the intended fix, and why the tests prove the regression is closed."
        patch = source_text[:2500]
        tests = "Tests: run the included or implied assertions; add a failing-before/passing-after regression case."
        verification = "expected: patch and tests are treated as supervised scratch pretraining text"
    elif bucket == "debugging_error_logs":
        prompt = "Debug this runtime or compilation failure and produce the smallest safe fix plan."
        correct_answer = "Identify the likely failing line or invariant, explain the cause, and propose a minimal safe correction."
        patch = "Patch plan: add the missing guard, dependency, type check, or compile fix indicated by the trace."
        tests = "Tests: reproduce the failure, apply the fix, rerun the failing command, and add a regression assertion."
    elif bucket == "high_quality_docs_code":
        prompt = "Convert this high-quality documentation or code context into implementation guidance."
        correct_answer = (
            "Extract reusable design rules, APIs, constraints, and verification steps."
        )
        if (
            "def " in source_text
            or "class " in source_text
            or "function " in source_text
            or "fn " in source_text
        ):
            patch = "Implementation plan: preserve API behavior, add tests, and keep changes local to the owning module."

    text = (
        "<|thought_start|>\n"
        f"Prompt: {prompt}\n"
        f"Source: {source}\n"
        f"Language: {language}\n"
        f"Context:\n{excerpt}\n\n"
        f"Answer:\n{correct_answer}\n"
        "<|thought_end|>\n"
        "<|patch_start|>\n"
        f"{patch}\n"
        "<|patch_end|>\n"
        f"Tests: {tests}\n"
        f"Verification: {verification}\n"
    )
    metadata = {
        **_metadata(row),
        "scratch_mix_bucket": bucket,
        "source_content_hash": digest,
        "converted_to_instruction": True,
    }
    return {
        "text": text,
        "content_hash": stable_hash(text),
        "source": source,
        "url": row.get("url"),
        "license": _license(row),
        "category": bucket,
        "quality": _quality(row),
        "training_gate": _training_gate(row),
        "metadata": metadata,
        "train_policy": "train",
    }


def _scratch_estimated_tokens(row: dict[str, Any], tokenizer: Any | None) -> int:
    text = _row_text(row)
    if tokenizer is not None:
        return max(1, len(tokenizer.encode(text).ids))
    return max(1, len(text) // 4)


def build_scratch_instruction_mix(
    *,
    input_paths: list[str | Path],
    output_path: str | Path,
    report_path: str | Path,
    config: ScratchMixConfig | None = None,
    tokenizer_path: str | Path | None = None,
) -> ScratchInstructionMixReport:
    active = config or ScratchMixConfig()
    selected_ratios = curriculum_ratios(active.curriculum_mode, active.ratios)
    normalized = _normalize_ratios(selected_ratios)
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else None
    rng = random.Random(active.seed)
    buckets: dict[str, list[dict[str, Any]]] = {
        bucket: [] for bucket in SCRATCH_INSTRUCTION_RATIOS
    }
    rejected = 0
    offensive_rejected = 0
    seen: set[str] = set()
    for path in input_paths:
        for row in iter_jsonl(path):
            text = _row_text(row)
            digest = str(row.get("content_hash") or stable_hash(text))
            if not text or digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            if _quality_score(row) < active.min_quality_score:
                rejected += 1
                continue
            bucket = classify_scratch_bucket(row)
            if not bucket_allowed_for_curriculum(bucket, active.curriculum_mode):
                rejected += 1
                continue
            if (
                active.curriculum_mode == "defensive_security_only"
                and active.strict_offensive_filter
                and contains_offensive_misuse(text)
            ):
                rejected += 1
                offensive_rejected += 1
                continue
            converted = convert_to_instruction_row(row, bucket=bucket)
            converted["_scratch_bucket"] = bucket
            converted["_estimated_tokens"] = _scratch_estimated_tokens(
                converted, tokenizer
            )
            buckets[bucket].append(converted)

    for rows in buckets.values():
        rng.shuffle(rows)

    cursors = {bucket: 0 for bucket in buckets}
    token_totals: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    total_available = sum(len(rows) for rows in buckets.values())
    limit = min(active.max_rows or total_available, total_available)
    while len(selected) < limit:
        choices = [
            bucket for bucket, rows in buckets.items() if cursors[bucket] < len(rows)
        ]
        if not choices:
            break
        current_total = max(1, sum(token_totals.values()))
        ranked = []
        for bucket in choices:
            actual = token_totals[bucket] / current_total
            ranked.append((normalized.get(bucket, 0.0) - actual, rng.random(), bucket))
        ranked.sort(reverse=True)
        bucket = ranked[0][2]
        row = buckets[bucket][cursors[bucket]]
        cursors[bucket] += 1
        token_totals[bucket] += int(row["_estimated_tokens"])
        selected.append(row)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    transformed = 0
    raw_preserved = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in selected:
            bucket = str(row.pop("_scratch_bucket"))
            tokens = int(row.pop("_estimated_tokens"))
            if row.get("metadata", {}).get("converted_to_instruction"):
                transformed += 1
            else:
                raw_preserved += 1
                if active.preserve_raw_docs:
                    original_text = _row_text(row)
                    row["text"] = original_text
            row["scratch_mix"] = {
                "bucket": bucket,
                "estimated_tokens": tokens,
                "target_ratio": normalized.get(bucket, 0.0),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    total_tokens = sum(token_totals.values())
    bucket_reports = [
        ScratchMixBucketReport(
            bucket=bucket,
            input_rows=len(rows),
            output_rows=sum(
                1
                for row in selected
                if row.get("scratch_mix", {}).get("bucket") == bucket
            ),
            output_tokens=int(token_totals[bucket]),
            target_ratio=round(normalized.get(bucket, 0.0), 6),
            actual_ratio=round(token_totals[bucket] / max(1, total_tokens), 6),
        )
        for bucket, rows in sorted(buckets.items())
    ]
    recommendations: list[str] = []
    for item in bucket_reports:
        if item.output_rows == 0:
            recommendations.append(
                f"bucket {item.bucket} has zero rows; add higher-yield approved sources for this class"
            )
        elif item.actual_ratio < item.target_ratio * 0.5:
            recommendations.append(
                f"bucket {item.bucket} is under target; available source data is scarce"
            )
    status = (
        "passed"
        if selected and not any(item.output_rows == 0 for item in bucket_reports)
        else "warning"
    )
    report = ScratchInstructionMixReport(
        status=status,
        input_paths=[str(path) for path in input_paths],
        output_jsonl=str(target),
        curriculum_mode=active.curriculum_mode,
        total_rows=len(selected),
        total_tokens=int(total_tokens),
        buckets=bucket_reports,
        rejected_rows=rejected,
        offensive_rejected_rows=offensive_rejected,
        strict_offensive_filter=active.strict_offensive_filter,
        transformed_instruction_rows=transformed,
        raw_preserved_rows=raw_preserved,
        target_ratios={
            key: round(value, 6) for key, value in sorted(normalized.items())
        },
        recommendations=recommendations,
    )
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Craftly token-ratio mixed corpus."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--config", default="config/mix_ratios.json")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch-instruction-mix", action="store_true")
    parser.add_argument(
        "--curriculum-mode",
        default="balanced",
        choices=[
            "balanced",
            "fundamentals_only",
            "defensive_security_only",
            "debug_patch_only",
            "agentic_coding_only",
        ],
    )
    parser.add_argument("--allow-offensive-misuse-rows", action="store_true")
    parser.add_argument(
        "--output-jsonl", help="Output path for --scratch-instruction-mix."
    )
    parser.add_argument(
        "--report-path", help="Report path for --scratch-instruction-mix."
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scratch_instruction_mix:
        output_path = args.output_jsonl or str(
            Path(args.output_dir) / "scratch_instruction_mix.jsonl"
        )
        report_path = args.report_path or str(
            Path(args.output_dir) / "scratch_instruction_mix_report.json"
        )
        report = build_scratch_instruction_mix(
            input_paths=args.inputs,
            output_path=output_path,
            report_path=report_path,
            tokenizer_path=args.tokenizer_path,
            config=ScratchMixConfig(
                max_rows=args.max_rows,
                min_quality_score=args.min_quality_score,
                curriculum_mode=args.curriculum_mode,
                strict_offensive_filter=not args.allow_offensive_misuse_rows,
            ),
        )
        print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
        raise SystemExit(0 if report.status in {"passed", "warning"} else 1)
    manifest = build_mix(
        input_paths=args.inputs,
        config_path=args.config,
        experiment=args.experiment,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        shard_token_count=args.shard_token_count,
        sequence_length=args.sequence_length,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
