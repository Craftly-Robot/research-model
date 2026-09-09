"""Dataset quality gates for scratch pretraining corpora."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel

SECRET_RE = re.compile(
    r"(?i)(-----BEGIN .*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,})"
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
HEX_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
CWE_RE = re.compile(r"\bCWE-\d{1,5}\b", re.IGNORECASE)

DEFENSIVE_SECURITY_TERMS = {
    "authentication",
    "authorization",
    "buffer overflow",
    "cve",
    "cwe",
    "csrf",
    "deserialization",
    "encryption",
    "hardcoded secret",
    "injection",
    "mitigation",
    "owasp",
    "patch",
    "path traversal",
    "sanitize",
    "sanitization",
    "secure",
    "ssrf",
    "validation",
    "vulnerability",
    "xss",
}
AGENTIC_CODING_TERMS = {
    "debug",
    "failing test",
    "fix",
    "implementation",
    "pull request",
    "refactor",
    "regression",
    "repository",
    "stack trace",
    "test suite",
    "traceback",
}
CODE_TOKENS = {
    "class ",
    "def ",
    "fn ",
    "func ",
    "function ",
    "import ",
    "package ",
    "pub ",
    "return ",
    "struct ",
}
CONFIG_TOKENS = {
    "dockerfile",
    "docker-compose",
    "github actions",
    "kubernetes",
    "nginx",
    "terraform",
    "yaml",
}
NOISE_TERMS = {
    "accept cookies",
    "all rights reserved",
    "cookie policy",
    "enable javascript",
    "main navigation",
    "privacy policy",
    "skip to content",
    "subscribe to newsletter",
    "table of contents",
    "terms of service",
}
BINARY_CONTENT_TYPES = {
    "application/octet-stream",
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/svg+xml",
    "image/webp",
}


class QualityGateConfig(StrictModel):
    min_chars: int = Field(default=200, ge=1)
    max_chars: int = Field(default=2_000_000, ge=1)
    require_license: bool = True
    allowed_licenses: set[str] = Field(
        default_factory=lambda: {
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
            "unknown-ok",
        }
    )
    reject_emails: bool = True
    reject_secrets: bool = True
    min_alpha_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    min_unique_word_ratio: float = Field(default=0.08, ge=0.0, le=1.0)


class QualityDecision(StrictModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    language_hint: str | None = None
    data_type: str = "unknown"
    content_hash: str
    component_scores: dict[str, float] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)


class QualityGateReport(StrictModel):
    input_path: str
    output_path: str
    accepted: int
    rejected: int
    duplicate: int
    created_at_unix: float = Field(default_factory=time.time)


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    snippet = line[:240].replace("\n", "\\n")
                    raise ValueError(
                        f"invalid JSONL in {source} at line {line_number}: {exc.msg}; snippet={snippet!r}"
                    ) from exc


def infer_language(text: str, url: str = "") -> str | None:
    lowered_url = url.lower()
    suffix_map = {
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".go": "go",
        ".h": "c_cpp_header",
        ".hpp": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".py": "python",
        ".rs": "rust",
        ".sh": "bash",
        ".sol": "solidity",
        ".ts": "typescript",
        ".tsx": "typescript",
    }
    for suffix, language in suffix_map.items():
        if lowered_url.endswith(suffix):
            return language
    lowered = text.lower()
    if "pragma solidity" in lowered or "contract " in lowered and "solidity" in lowered:
        return "solidity"
    if "fn main(" in lowered or "cargo.toml" in lowered or "pub struct" in lowered:
        return "rust"
    if "package main" in lowered and "func " in lowered:
        return "go"
    if "#include <" in lowered or "int main(" in lowered or "strcpy(" in lowered:
        return "c_cpp"
    if "public class " in lowered or "private static" in lowered:
        return "java"
    if (
        "def " in text
        or "import " in text
        and "python" in lowered
        or "traceback (most recent call last)" in lowered
    ):
        return "python"
    if "#!/bin/bash" in lowered or "set -euo pipefail" in lowered:
        return "bash"
    if (
        "interface " in lowered
        and "typescript" in lowered
        or ": string" in lowered
        and "const " in lowered
    ):
        return "typescript"
    if "function " in lowered or "const " in lowered and "=>" in lowered:
        return "javascript"
    return None


def infer_data_type(text: str, labels: list[str]) -> str:
    lowered = text.lower()
    if (
        "diff --git" in lowered
        or lowered.startswith("--- ")
        and "\n+++ " in lowered
        or "patch" in labels
    ):
        return "patch"
    if (
        "traceback (most recent call last)" in lowered
        or "compile error" in lowered
        or "stack trace" in lowered
    ):
        return "debug_trace"
    if "test_" in lowered or "pytest" in lowered or "unittest" in lowered or "assert " in lowered:
        return "test"
    if CVE_RE.search(text) or CWE_RE.search(text) or "known exploited vulnerabilities" in lowered:
        return "security_advisory"
    if "vulnerability" in lowered or "mitigation" in lowered or "owasp" in lowered:
        return "security_reference"
    if any(token in lowered for token in CONFIG_TOKENS):
        return "config_or_infra"
    if "code" in labels:
        return "code"
    if "api" in lowered or "reference" in lowered:
        return "api_documentation"
    if "tutorial" in lowered or "guide" in lowered:
        return "tutorial"
    return "documentation"


def _count_matches(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _text_metrics(normalized: str) -> dict[str, float]:
    chars = len(normalized)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_+-]*", normalized)
    unique_words = len({word.lower() for word in words})
    alpha_chars = sum(1 for char in normalized if char.isalpha())
    lines = max(1, normalized.count("\n") + 1)
    repeated_line_count = 0
    line_counts: dict[str, int] = {}
    for line in normalized.splitlines():
        key = line.strip()
        if len(key) < 12:
            continue
        line_counts[key] = line_counts.get(key, 0) + 1
        if line_counts[key] == 2:
            repeated_line_count += 1
    return {
        "alpha_ratio": alpha_chars / max(1, chars),
        "unique_word_ratio": unique_words / max(1, len(words)),
        "line_count": float(lines),
        "repeated_line_ratio": repeated_line_count / max(1, lines),
        "word_count": float(len(words)),
    }


def build_labels(text: str) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []
    if (
        any(term in lowered for term in DEFENSIVE_SECURITY_TERMS)
        or CVE_RE.search(text)
        or CWE_RE.search(text)
    ):
        labels.append("defensive_security")
    if any(term in lowered for term in AGENTIC_CODING_TERMS):
        labels.append("agentic_coding")
    if any(token in lowered for token in CODE_TOKENS):
        labels.append("code")
    if "diff --git" in lowered or "\n+++ " in lowered and "\n--- " in lowered:
        labels.append("patch")
    if any(token in lowered for token in ["test_", "pytest", "unittest", "assert "]):
        labels.append("tests")
    if any(token in lowered for token in CONFIG_TOKENS):
        labels.append("infrastructure")
    if HEX_ADDRESS_RE.search(text):
        labels.append("runtime_trace")
    return labels


def quality_score(
    *, text: str, labels: list[str], reasons: list[str]
) -> tuple[float, dict[str, float], list[str]]:
    if reasons:
        return 0.0, {}, []
    normalized = text.strip()
    metrics = _text_metrics(normalized)
    length = len(" ".join(normalized.split()))
    security_hits = _count_matches(normalized, DEFENSIVE_SECURITY_TERMS)
    agentic_hits = _count_matches(normalized, AGENTIC_CODING_TERMS)
    code_hits = _count_matches(normalized, CODE_TOKENS)
    noise_hits = _count_matches(normalized, NOISE_TERMS)

    components = {
        "length": _bounded((length - 180) / 2500),
        "security_signal": _bounded(security_hits / 5),
        "agentic_signal": _bounded(agentic_hits / 4),
        "code_signal": _bounded((code_hits + (1 if "code" in labels else 0)) / 5),
        "test_signal": 1.0 if "tests" in labels else 0.0,
        "structure": _bounded((metrics["line_count"] / 80) + (0.3 if "patch" in labels else 0.0)),
        "low_noise": _bounded(1.0 - (noise_hits * 0.18) - metrics["repeated_line_ratio"]),
        "lexical_diversity": _bounded(metrics["unique_word_ratio"] * 4),
    }
    score = (
        0.14 * components["length"]
        + 0.20 * components["security_signal"]
        + 0.14 * components["agentic_signal"]
        + 0.18 * components["code_signal"]
        + 0.08 * components["test_signal"]
        + 0.09 * components["structure"]
        + 0.09 * components["low_noise"]
        + 0.08 * components["lexical_diversity"]
    )
    if "defensive_security" in labels and "code" in labels:
        score += 0.08
    if "patch" in labels or "runtime_trace" in labels:
        score += 0.05

    risk_flags: list[str] = []
    if noise_hits:
        risk_flags.append("navigation_or_boilerplate_noise")
    if noise_hits >= 3:
        risk_flags.append("heavy_boilerplate_noise")
    if metrics["repeated_line_ratio"] > 0.18:
        risk_flags.append("repeated_lines")
    if metrics["unique_word_ratio"] < 0.12:
        risk_flags.append("low_lexical_diversity")
    return (
        round(_bounded(score), 6),
        {key: round(value, 6) for key, value in components.items()},
        risk_flags,
    )


class DatasetQualityGate:
    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(self, row: dict[str, Any], *, seen: set[str] | None = None) -> QualityDecision:
        text = str(row.get("text") or row.get("content") or "")
        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        compact = " ".join(normalized.split())
        digest = stable_hash(compact)
        reasons: list[str] = []
        risk_flags: list[str] = []
        if len(compact) < self.config.min_chars:
            reasons.append("too_short")
        if len(compact) > self.config.max_chars:
            reasons.append("too_large")
        license_name = str(
            row.get("license") or row.get("metadata", {}).get("license") or ""
        ).lower()
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        content_type = (
            str(metadata.get("content_type") or row.get("content_type") or "")
            .split(";")[0]
            .strip()
            .lower()
        )
        if content_type in BINARY_CONTENT_TYPES:
            reasons.append("non_text_content_type")
        if self.config.require_license and license_name not in self.config.allowed_licenses:
            reasons.append("license_not_allowed")
        if self.config.reject_secrets and SECRET_RE.search(text):
            reasons.append("secret_like_content")
        if self.config.reject_emails and EMAIL_RE.search(text):
            reasons.append("email_like_pii")
        if seen is not None and digest in seen:
            reasons.append("duplicate")
        metrics = _text_metrics(compact)
        if (
            len(compact) >= self.config.min_chars
            and metrics["alpha_ratio"] < self.config.min_alpha_ratio
        ):
            reasons.append("low_text_signal")
        if (
            metrics["word_count"] >= 80
            and metrics["unique_word_ratio"] < self.config.min_unique_word_ratio
        ):
            risk_flags.append("low_unique_word_ratio")
            if metrics["unique_word_ratio"] < 0.025:
                reasons.append("low_unique_word_ratio")
        if metrics["repeated_line_ratio"] > 0.35:
            reasons.append("excessive_repeated_lines")

        labels = build_labels(normalized)
        language_hint = infer_language(text, str(row.get("url") or ""))
        data_type = infer_data_type(text, labels)
        score, component_scores, score_flags = quality_score(
            text=normalized, labels=labels, reasons=reasons
        )
        risk_flags.extend(score_flags)
        if (
            "heavy_boilerplate_noise" in risk_flags
            and "code" not in labels
            and "defensive_security" not in labels
        ):
            reasons.append("heavy_boilerplate_noise")
            score, component_scores, score_flags = quality_score(
                text=normalized, labels=labels, reasons=reasons
            )
            risk_flags.extend(score_flags)
        return QualityDecision(
            accepted=not reasons,
            reasons=reasons,
            labels=labels,
            quality_score=score,
            language_hint=language_hint,
            data_type=data_type,
            content_hash=digest,
            component_scores=component_scores,
            risk_flags=sorted(set(risk_flags)),
        )

    def filter_jsonl(self, input_path: str | Path, output_path: str | Path) -> QualityGateReport:
        seen: set[str] = set()
        accepted = rejected = duplicate = 0
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in iter_jsonl(input_path):
                decision = self.evaluate(row, seen=seen)
                if "duplicate" in decision.reasons:
                    duplicate += 1
                if not decision.accepted:
                    rejected += 1
                    continue
                seen.add(decision.content_hash)
                row["quality"] = decision.model_dump()
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
        return QualityGateReport(
            input_path=str(input_path),
            output_path=str(target),
            accepted=accepted,
            rejected=rejected,
            duplicate=duplicate,
        )
