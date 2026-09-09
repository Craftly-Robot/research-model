"""Build verified defensive security patch datasets from approved Git history."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.quality import stable_hash
from src.craftly.learning.repo_patch_extraction import SECURITY_PATCH_TERMS
from src.craftly.shared.schemas import StrictModel

DEFAULT_ALLOWED_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "mit",
    "mpl-2.0",
    "postgresql",
    "psf-2.0",
}


class VerifiedPatchRecord(StrictModel):
    record_id: str
    repo_path: str
    source: str
    license: str
    commit: str
    parent_commit: str
    subject: str
    files_changed: list[str]
    vulnerability_categories: list[str]
    prompt: str
    chosen: str
    patch: str
    before_after: dict[str, dict[str, str]]
    verification: dict[str, Any]
    content_hash: str
    category: str = "verified_security_patch"
    train_policy: str = "train"
    created_at_unix: float = Field(default_factory=time.time)


class VerifiedPatchDatasetReport(StrictModel):
    output_path: str
    scanned_repos: int
    scanned_commits: int
    extracted: int
    skipped: int
    failed_verification: int
    by_category: dict[str, int] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=time.time)


def run_git(
    repo: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 30,
    max_bytes: int = 4_000_000,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _must_git(
    repo: Path, args: list[str], *, timeout: int = 30, max_bytes: int = 4_000_000
) -> str:
    result = run_git(repo, args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[:500]}")
    return result.stdout[:max_bytes]


def security_categories(text: str) -> list[str]:
    lowered = text.lower()
    categories = []
    mapping = {
        "auth": ("auth", "authorization", "authentication", "session", "jwt"),
        "buffer": ("buffer", "overflow", "out-of-bounds", "memcpy", "strcpy"),
        "injection": ("injection", "sql", "command injection", "xss"),
        "path_traversal": ("path traversal", "../", "directory traversal"),
        "ssrf": ("ssrf", "server-side request forgery"),
        "crypto": ("crypto", "md5", "sha1", "random", "tls"),
        "cve_cwe": ("cve-", "cwe-"),
        "validation": ("sanitize", "validation", "validate", "escaping"),
    }
    for category, terms in mapping.items():
        if any(term in lowered for term in terms):
            categories.append(category)
    return sorted(set(categories or ["security_fix"]))


def looks_security_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SECURITY_PATCH_TERMS) or bool(
        security_categories(text)
    )


def changed_files(repo: Path, parent: str, commit: str) -> list[str]:
    raw = _must_git(repo, ["diff", "--name-only", parent, commit])
    return [line.strip() for line in raw.splitlines() if line.strip()]


def file_at(repo: Path, revision: str, path: str, *, max_chars: int) -> str:
    result = run_git(repo, ["show", f"{revision}:{path}"], timeout=20)
    if result.returncode != 0:
        return ""
    return result.stdout[:max_chars]


def verify_patch_applies(repo: Path, parent: str, patch: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        worktree = Path(temp) / "worktree"
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(worktree)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if clone.returncode != 0:
            return {
                "status": "failed",
                "method": "git_clone",
                "stderr": clone.stderr[:1000],
            }
        checkout = run_git(worktree, ["checkout", "--quiet", parent], timeout=30)
        if checkout.returncode != 0:
            return {
                "status": "failed",
                "method": "git_checkout_parent",
                "stderr": checkout.stderr[:1000],
            }
        check = run_git(
            worktree,
            ["apply", "--check", "--whitespace=nowarn", "-"],
            input_text=patch,
            timeout=30,
        )
        return {
            "status": "passed" if check.returncode == 0 else "failed",
            "method": "git_apply_check_on_parent",
            "returncode": check.returncode,
            "stderr": check.stderr[:1000],
        }


def build_prompt(
    *, subject: str, files: list[str], before_after: dict[str, dict[str, str]]
) -> str:
    context_parts = [
        "You are given an approved defensive security-fix commit from a permissive-license repository.",
        "Task: identify the vulnerability class, explain the safe fix, and produce a patch-shaped remediation answer.",
        "Do not provide live-target attack instructions.",
        f"Commit subject: {subject}",
        f"Files changed: {', '.join(files[:20])}",
    ]
    for path, versions in list(before_after.items())[:3]:
        context_parts.append(
            f'\n<file path="{path}" before>\n{versions.get("before", "")[:4000]}\n</file>'
        )
    return "\n".join(context_parts)


def build_chosen(*, categories: list[str], patch: str) -> str:
    return (
        "<|thought_start|>"
        f"Security categories: {', '.join(categories)}. The safe response is defensive patch analysis with regression verification."
        "<|thought_end|>"
        "<|patch_start|>\n"
        f"{patch}"
        "\n<|patch_end|>"
    )


def iter_security_commits(repo: Path, *, max_commits: int) -> list[tuple[str, str]]:
    raw = _must_git(repo, ["log", f"--max-count={max_commits}", "--format=%H%x1f%s"])
    output: list[tuple[str, str]] = []
    for line in raw.splitlines():
        commit, _, subject = line.partition("\x1f")
        if commit and looks_security_relevant(subject):
            output.append((commit, subject))
    return output


def build_verified_patch_dataset(
    *,
    repo_paths: list[str | Path],
    output_path: str | Path,
    license_name: str,
    max_commits_per_repo: int = 500,
    max_patch_chars: int = 120_000,
    max_file_chars: int = 12_000,
    allowed_licenses: set[str] | None = None,
) -> VerifiedPatchDatasetReport:
    if shutil.which("git") is None:
        raise RuntimeError("git executable is required")
    allowed = allowed_licenses or DEFAULT_ALLOWED_LICENSES
    normalized_license = license_name.lower()
    if normalized_license not in allowed:
        raise ValueError(f"license is not approved for training: {license_name}")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scanned_repos = scanned_commits = extracted = skipped = failed_verification = 0
    by_category: dict[str, int] = {}
    seen: set[str] = set()
    with target.open("w", encoding="utf-8") as handle:
        for repo_path in repo_paths:
            repo = Path(repo_path).resolve()
            scanned_repos += 1
            if not (repo / ".git").exists():
                raise ValueError(f"not a git repository: {repo}")
            for commit, subject in iter_security_commits(
                repo, max_commits=max_commits_per_repo
            ):
                scanned_commits += 1
                parent_result = run_git(repo, ["rev-parse", f"{commit}^"], timeout=20)
                if parent_result.returncode != 0:
                    skipped += 1
                    continue
                parent = parent_result.stdout.strip()
                patch = _must_git(
                    repo,
                    [
                        "show",
                        "--format=",
                        "--patch",
                        "--find-renames",
                        "--binary",
                        commit,
                    ],
                    max_bytes=max_patch_chars,
                )
                if not patch.strip() or not looks_security_relevant(
                    subject + "\n" + patch
                ):
                    skipped += 1
                    continue
                verification = verify_patch_applies(repo, parent, patch)
                if verification["status"] != "passed":
                    failed_verification += 1
                    continue
                files = changed_files(repo, parent, commit)
                before_after = {
                    path: {
                        "before": file_at(repo, parent, path, max_chars=max_file_chars),
                        "after": file_at(repo, commit, path, max_chars=max_file_chars),
                    }
                    for path in files[:8]
                }
                categories = security_categories(subject + "\n" + patch)
                digest = stable_hash(str(repo) + commit + patch)
                if digest in seen:
                    skipped += 1
                    continue
                seen.add(digest)
                for category in categories:
                    by_category[category] = by_category.get(category, 0) + 1
                record = VerifiedPatchRecord(
                    record_id=f"verified-patch-{digest[:16]}",
                    repo_path=str(repo),
                    source=repo.name,
                    license=normalized_license,
                    commit=commit,
                    parent_commit=parent,
                    subject=subject,
                    files_changed=files,
                    vulnerability_categories=categories,
                    prompt=build_prompt(
                        subject=subject, files=files, before_after=before_after
                    ),
                    chosen=build_chosen(categories=categories, patch=patch),
                    patch=patch,
                    before_after=before_after,
                    verification=verification,
                    content_hash=digest,
                )
                handle.write(
                    json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                extracted += 1
    return VerifiedPatchDatasetReport(
        output_path=str(target),
        scanned_repos=scanned_repos,
        scanned_commits=scanned_commits,
        extracted=extracted,
        skipped=skipped,
        failed_verification=failed_verification,
        by_category=dict(sorted(by_category.items())),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verified defensive security patch JSONL from approved Git repos."
    )
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Approved local Git repository path. Repeatable.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--max-commits-per-repo", type=int, default=500)
    parser.add_argument("--max-patch-chars", type=int, default=120_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_verified_patch_dataset(
        repo_paths=args.repo,
        output_path=args.output,
        license_name=args.license,
        max_commits_per_repo=args.max_commits_per_repo,
        max_patch_chars=args.max_patch_chars,
    )
    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
