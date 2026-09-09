"""Extract defensive patch tasks from approved local Git repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from pydantic import Field

from src.craftly.learning.quality import stable_hash
from src.craftly.shared.schemas import StrictModel

SECURITY_PATCH_TERMS = (
    "auth",
    "buffer",
    "cve",
    "csrf",
    "cwe",
    "injection",
    "overflow",
    "path traversal",
    "sanitize",
    "security",
    "ssrf",
    "validation",
    "vulnerability",
    "xss",
)


class ExtractedPatchTask(StrictModel):
    task_id: str
    repo_path: str
    commit: str
    subject: str
    prompt: str
    patch: str
    license: str
    source: str
    category: str = "security_patch"
    content_hash: str


class RepoPatchExtractionReport(StrictModel):
    repo_path: str
    output_path: str
    scanned_commits: int
    extracted: int
    skipped: int
    created_at_unix: float = Field(default_factory=time.time)


def _run_git(repo_path: Path, args: list[str], *, max_bytes: int = 2_000_000) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return result.stdout[:max_bytes]


def _looks_security_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SECURITY_PATCH_TERMS)


def extract_security_patch_tasks(
    repo_path: str | Path,
    output_path: str | Path,
    *,
    license_name: str,
    max_commits: int = 200,
    max_patch_chars: int = 80_000,
) -> RepoPatchExtractionReport:
    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"not a git repository: {repo}")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    commits_raw = _run_git(
        repo, ["log", f"--max-count={max_commits}", "--format=%H%x1f%s"]
    )
    scanned = 0
    extracted = 0
    skipped = 0
    with target.open("w", encoding="utf-8") as handle:
        for line in commits_raw.splitlines():
            if not line.strip():
                continue
            scanned += 1
            commit, _, subject = line.partition("\x1f")
            if not _looks_security_relevant(subject):
                skipped += 1
                continue
            patch = _run_git(
                repo,
                ["show", "--format=fuller", "--patch", "--find-renames", commit],
                max_bytes=max_patch_chars,
            )
            if not _looks_security_relevant(patch):
                skipped += 1
                continue
            prompt = (
                "Given this approved permissive-license security patch, learn the defensive task shape. "
                "Identify the vulnerability class, explain the safe fix, and propose verification tests.\n\n"
                f"Commit subject: {subject}\n\nPatch:\n{patch[:max_patch_chars]}"
            )
            digest = stable_hash(commit + "\n" + subject + "\n" + patch)
            task = ExtractedPatchTask(
                task_id=f"repo-patch-{digest[:16]}",
                repo_path=str(repo),
                commit=commit,
                subject=subject,
                prompt=prompt,
                patch=patch,
                license=license_name.lower(),
                source=repo.name,
                content_hash=digest,
            )
            handle.write(
                json.dumps(task.model_dump(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            extracted += 1
    return RepoPatchExtractionReport(
        repo_path=str(repo),
        output_path=str(target),
        scanned_commits=scanned,
        extracted=extracted,
        skipped=skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract defensive security patch tasks from an approved local repo."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--max-commits", type=int, default=200)
    args = parser.parse_args()
    report = extract_security_patch_tasks(
        args.repo, args.output, license_name=args.license, max_commits=args.max_commits
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
