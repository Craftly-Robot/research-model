"""Extract supervised/evaluation task candidates from clean corpus shards."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.quality import CVE_RE, CWE_RE, iter_jsonl, stable_hash
from src.craftly.shared.schemas import StrictModel

CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-.#]*)\n(.*?)```", re.DOTALL)
DIFF_RE = re.compile(r"(?ms)^diff --git .*?(?=^diff --git |\Z)")
ERROR_RE = re.compile(
    r"(?i)(traceback \(most recent call last\)|compile error|undefined reference|segmentation fault|stack trace|exception|panic:)"
)
SECURITY_CATEGORIES = {
    "buffer_overflow": (
        "buffer overflow",
        "strcpy",
        "memcpy",
        "out-of-bounds",
        "cwe-120",
        "cwe-787",
    ),
    "command_injection": ("command injection", "shell=true", "os.system", "subprocess", "cwe-78"),
    "dependency_confusion": (
        "dependency confusion",
        "typosquatting",
        "package takeover",
        "malicious package",
    ),
    "deserialization": (
        "deserialization",
        "pickle.loads",
        "yaml.load",
        "objectinputstream",
        "cwe-502",
    ),
    "hardcoded_secret": ("hardcoded secret", "api_key", "secret", "password =", "cwe-798"),
    "insecure_auth_session": (
        "session fixation",
        "jwt none",
        "missing authorization",
        "broken access control",
        "cwe-287",
        "cwe-862",
    ),
    "insecure_solidity": (
        "reentrancy",
        "unchecked call",
        "tx.origin",
        "integer overflow",
        "solidity",
    ),
    "path_traversal": ("path traversal", "../", "cwe-22"),
    "prototype_pollution": ("prototype pollution", "__proto__", "constructor.prototype"),
    "race_condition": ("race condition", "time-of-check", "time of check", "toctou", "cwe-362"),
    "sql_injection": ("sql injection", "select * from", "cursor.execute", "cwe-89"),
    "ssrf": ("ssrf", "server-side request forgery", "cwe-918"),
    "supply_chain": ("slsa", "scorecard", "sigstore", "provenance", "software supply chain"),
    "weak_crypto": ("md5", "sha1", "weak crypto", "insecure random", "cwe-327"),
    "xss": ("cross-site scripting", "xss", "innerhtml", "cwe-79"),
}
IMPLEMENTATION_TERMS = (
    "build",
    "implement",
    "api",
    "database",
    "deployment",
    "architecture",
    "workflow",
)
TEST_TERMS = ("pytest", "unittest", "assert ", "test_", "regression")
PATCH_TERMS = ("patch", "fix", "mitigation", "diff --git", "+++ ", "--- ")


class ExtractedTask(StrictModel):
    task_id: str
    task_type: str
    prompt: str
    source_url: str | None = None
    language: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskExtractionReport(StrictModel):
    output_path: str
    scanned_rows: int
    rows_with_tasks: int
    capped_rows: int
    extracted: int
    average_tasks_per_row: float = Field(ge=0.0)
    by_type: dict[str, int]
    by_language: dict[str, int] = Field(default_factory=dict)
    by_security_category: dict[str, int] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=time.time)


def _security_categories(text: str, source_url: str | None = None) -> list[str]:
    lowered = text.lower()
    primary_context = lowered[:300]
    normalized_url = (source_url or "").lower().replace("-", "_")
    scores: list[tuple[int, str]] = []
    for category, terms in SECURITY_CATEGORIES.items():
        score = 0
        if category in normalized_url:
            score += 10
        for term in terms:
            count = lowered.count(term)
            if term in primary_context:
                score += 5
            if count >= 2:
                score += min(count, 4)
            elif count == 1 and any(marker in term for marker in (".", "=", "_", "/", "-")):
                score += 2
        if score >= 4:
            scores.append((score, category))
    if CVE_RE.search(text) or CWE_RE.search(text):
        scores.append((10, "vulnerability_taxonomy"))
    return [
        category for _score, category in sorted(scores, key=lambda item: (-item[0], item[1]))[:3]
    ]


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or "")


def _source(row: dict[str, Any]) -> str:
    return str(row.get("url") or "approved corpus")


def _language(row: dict[str, Any], fallback: str | None = None) -> str | None:
    quality = row.get("quality", {})
    language = fallback or quality.get("language_hint") or row.get("language")
    return str(language) if language else None


def _base_constraints() -> list[str]:
    return [
        "Keep the answer defensive and authorized.",
        "Do not provide exploit execution steps.",
        "Do not invent source facts that are not present in the context.",
    ]


def _task(
    *,
    row: dict[str, Any],
    task_type: str,
    prompt: str,
    language: str | None,
    success_criteria: list[str],
    metadata: dict[str, Any],
) -> ExtractedTask:
    digest = stable_hash(task_type + "\n" + prompt)
    quality = row.get("quality", {})
    data_type = str(quality.get("data_type") or "")
    priority = "normal"
    if task_type in {"security_patch_generation", "security_vulnerability_identification"}:
        priority = "high"
    if data_type in {"patch", "security_advisory", "security_reference", "debug_trace"}:
        priority = "critical"
    return ExtractedTask(
        task_id=f"task-{digest[:16]}",
        task_type=task_type,
        prompt=prompt,
        source_url=row.get("url"),
        language=language,
        success_criteria=success_criteria,
        negative_constraints=_base_constraints(),
        metadata={
            "source": row.get("source"),
            "license": row.get("license"),
            "content_hash": row.get("content_hash") or quality.get("content_hash"),
            "quality_score": quality.get("quality_score"),
            "data_type": data_type,
            "training_priority": priority,
            **metadata,
        },
    )


def _security_finding_task(row: dict[str, Any], text: str, language: str | None) -> ExtractedTask:
    categories = _security_categories(text, row.get("url"))
    excerpt = text[:9000]
    prompt = (
        f"Analyze the approved defensive source context from {_source(row)}.\n\n"
        "Task: identify vulnerability surfaces, CWE/CVE references if present, affected code patterns, "
        "and safe remediation guidance. Return only defensive analysis.\n\n"
        f"Context:\n{excerpt}"
    )
    return _task(
        row=row,
        task_type="security_vulnerability_identification",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Names the likely weakness category.",
            "Explains why the pattern is risky.",
            "Provides defensive mitigation or safe coding guidance.",
            "Avoids exploit instructions.",
        ],
        metadata={"security_categories": categories},
    )


def _patch_generation_task(
    row: dict[str, Any], text: str, language: str | None, code: str | None = None
) -> ExtractedTask:
    body = code or text[:9000]
    prompt = (
        f"Using the approved defensive source context from {_source(row)}, produce a safe patch plan and patch-shaped answer.\n\n"
        "Task: fix the vulnerability or unsafe coding pattern while preserving behavior. Include tests or verification steps "
        "when they are implied by the context.\n\n"
        f"Artifact:\n{body[:9000]}"
    )
    return _task(
        row=row,
        task_type="security_patch_generation",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Identifies the unsafe behavior.",
            "Produces a defensive fix.",
            "Mentions validation or regression checks.",
            "Does not execute or weaponize the issue.",
        ],
        metadata={
            "security_categories": _security_categories(text + "\n" + (code or ""), row.get("url"))
        },
    )


def _code_review_task(
    row: dict[str, Any], text: str, language: str | None, code: str
) -> ExtractedTask:
    prompt = (
        f"Review this approved {language or 'code'} artifact from {_source(row)} for correctness, maintainability, "
        "and defensive security. Prioritize concrete findings and safe fixes.\n\n"
        f"Code:\n{code[:9000]}"
    )
    return _task(
        row=row,
        task_type="secure_code_review",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Finds correctness or security risks when present.",
            "Separates confirmed issues from assumptions.",
            "Suggests safe minimal changes.",
        ],
        metadata={"security_categories": _security_categories(text + "\n" + code, row.get("url"))},
    )


def _test_generation_task(
    row: dict[str, Any], text: str, language: str | None, code: str | None = None
) -> ExtractedTask:
    artifact = code or text[:9000]
    prompt = (
        f"Create regression and security-oriented tests from the approved source context at {_source(row)}.\n\n"
        "Task: derive meaningful test cases for the behavior or vulnerability described. Prefer small, deterministic tests.\n\n"
        f"Context or code:\n{artifact[:9000]}"
    )
    return _task(
        row=row,
        task_type="regression_test_generation",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Includes positive and negative cases when applicable.",
            "Targets the described bug or security behavior.",
            "Avoids brittle external dependencies.",
        ],
        metadata={},
    )


def _debugging_task(row: dict[str, Any], text: str, language: str | None) -> ExtractedTask:
    prompt = (
        f"Debug the following approved runtime or compilation context from {_source(row)}.\n\n"
        "Task: infer the likely root cause, propose a safe fix, and list verification commands.\n\n"
        f"Trace/context:\n{text[:9000]}"
    )
    return _task(
        row=row,
        task_type="debugging_from_error_trace",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Identifies the likely failure point.",
            "Proposes a minimal fix.",
            "Includes a verification path.",
        ],
        metadata={"contains_hex_address": "0x" in text.lower()},
    )


def _implementation_task(row: dict[str, Any], text: str, language: str | None) -> ExtractedTask:
    prompt = (
        f"Turn this approved technical source into an implementation plan for an agentic coding system.\n\n"
        f"Source: {_source(row)}\n\n"
        "Task: extract the engineering requirements, dependencies, failure modes, and verification checklist.\n\n"
        f"Context:\n{text[:9000]}"
    )
    return _task(
        row=row,
        task_type="implementation_planning",
        prompt=prompt,
        language=language,
        success_criteria=[
            "Lists concrete implementation requirements.",
            "Identifies dependencies and risks.",
            "Defines testable success criteria.",
        ],
        metadata={},
    )


def _code_candidates(text: str, language: str | None) -> list[tuple[str, str | None]]:
    candidates: list[tuple[str, str | None]] = []
    for match in CODE_FENCE_RE.finditer(text):
        code = match.group(2).strip()
        if len(code) >= 80:
            candidates.append((code, match.group(1).strip() or language))
    for match in DIFF_RE.finditer(text):
        diff = match.group(0).strip()
        if len(diff) >= 80:
            candidates.append((diff, language))
    return candidates[:5]


def _tasks_from_row(row: dict[str, Any]) -> list[ExtractedTask]:
    text = _row_text(row)
    quality = row.get("quality", {})
    language = _language(row)
    labels = set(quality.get("labels") or [])
    lowered = text.lower()
    tasks: list[ExtractedTask] = []
    code_candidates = _code_candidates(text, language)
    security_signal = (
        bool(_security_categories(text, row.get("url"))) or "defensive_security" in labels
    )
    patch_signal = any(term in lowered for term in PATCH_TERMS) or "patch" in labels
    test_signal = any(term in lowered for term in TEST_TERMS) or "tests" in labels
    implementation_signal = any(term in lowered for term in IMPLEMENTATION_TERMS)

    if security_signal:
        tasks.append(_security_finding_task(row, text, language))
    if patch_signal or quality.get("data_type") == "patch" or security_signal and code_candidates:
        tasks.append(
            _patch_generation_task(
                row, text, language, code_candidates[0][0] if code_candidates else None
            )
        )
    if ERROR_RE.search(text):
        tasks.append(_debugging_task(row, text, language))
    if test_signal:
        tasks.append(
            _test_generation_task(
                row, text, language, code_candidates[0][0] if code_candidates else None
            )
        )
    for code, candidate_language in code_candidates[:2]:
        tasks.append(_code_review_task(row, text, _language(row, candidate_language), code))
    if implementation_signal or not tasks:
        tasks.append(_implementation_task(row, text, language))
    return tasks


TASK_TYPE_PRIORITY = {
    "security_patch_generation": 0,
    "security_vulnerability_identification": 1,
    "debugging_from_error_trace": 2,
    "regression_test_generation": 3,
    "secure_code_review": 4,
    "implementation_planning": 5,
}


def extract_tasks(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    max_tasks: int = 10_000,
    max_tasks_per_row: int = 3,
) -> TaskExtractionReport:
    if max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    if not 1 <= max_tasks_per_row <= 20:
        raise ValueError("max_tasks_per_row must be between 1 and 20")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scanned = rows_with_tasks = capped_rows = extracted = 0
    by_type: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_security_category: dict[str, int] = {}
    seen: set[str] = set()
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            for row in iter_jsonl(path):
                scanned += 1
                candidates = sorted(
                    _tasks_from_row(row),
                    key=lambda task: (TASK_TYPE_PRIORITY.get(task.task_type, 99), task.task_id),
                )
                if len(candidates) > max_tasks_per_row:
                    capped_rows += 1
                selected = candidates[:max_tasks_per_row]
                emitted_for_row = 0
                for task in selected:
                    if task.task_id in seen:
                        continue
                    seen.add(task.task_id)
                    handle.write(
                        json.dumps(task.model_dump(), ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    extracted += 1
                    emitted_for_row += 1
                    by_type[task.task_type] = by_type.get(task.task_type, 0) + 1
                    by_language[task.language or "unknown"] = (
                        by_language.get(task.language or "unknown", 0) + 1
                    )
                    for category in task.metadata.get("security_categories") or []:
                        by_security_category[category] = by_security_category.get(category, 0) + 1
                    if extracted >= max_tasks:
                        return TaskExtractionReport(
                            output_path=str(target),
                            scanned_rows=scanned,
                            rows_with_tasks=rows_with_tasks + (1 if emitted_for_row else 0),
                            capped_rows=capped_rows,
                            extracted=extracted,
                            average_tasks_per_row=round(extracted / max(1, scanned), 6),
                            by_type=dict(sorted(by_type.items())),
                            by_language=dict(sorted(by_language.items())),
                            by_security_category=dict(sorted(by_security_category.items())),
                        )
                if emitted_for_row:
                    rows_with_tasks += 1
    return TaskExtractionReport(
        output_path=str(target),
        scanned_rows=scanned,
        rows_with_tasks=rows_with_tasks,
        capped_rows=capped_rows,
        extracted=extracted,
        average_tasks_per_row=round(extracted / max(1, scanned), 6),
        by_type=dict(sorted(by_type.items())),
        by_language=dict(sorted(by_language.items())),
        by_security_category=dict(sorted(by_security_category.items())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract bounded task candidates from governed Craftly JSONL."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL path; repeat for multiple shards.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tasks", type=int, default=10_000)
    parser.add_argument("--max-tasks-per-row", type=int, default=3)
    parser.add_argument("--report-out")
    args = parser.parse_args()
    report = extract_tasks(
        args.input,
        args.output,
        max_tasks=args.max_tasks,
        max_tasks_per_row=args.max_tasks_per_row,
    )
    if args.report_out:
        report_target = Path(args.report_out)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(
            json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
