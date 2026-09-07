"""Repository-level scorecard for the verified Craftly agent workflow.

Strict runs require 50-100 real repository tasks and a non-mock model backend.
Every task executes against an isolated copy; source repositories are hashed
before and after the run to prove the scorecard did not mutate its fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from src.craftly.db import LocalStore
from src.craftly.model_ops.backends import ModelBackend, build_active_backend
from src.craftly.model_ops.foundation import sha256_file
from src.craftly.runtime.execution import (
    AgentExecutionRequest,
    AgentExecutionService,
    stage_repository_copy,
)
from src.craftly.shared.config import active_profile_path, load_active_profile
from src.craftly.shared.schemas import StrictModel


class RepositoryAgentTask(StrictModel):
    task_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    repository: str = Field(min_length=1, max_length=4096)
    prompt: str = Field(min_length=1, max_length=32_000)
    category: Literal["coding", "debugging", "security", "patch", "long_context"]
    verification_commands: list[list[str]] = Field(min_length=1, max_length=20)
    expected_changed_files: list[str] = Field(default_factory=list, max_length=100)
    required_substrings: dict[str, list[str]] = Field(default_factory=dict)
    forbidden_substrings: dict[str, list[str]] = Field(default_factory=dict)
    pinned_files: list[str] = Field(default_factory=list, max_length=100)
    short_prompt: bool = False
    run_semgrep: bool = True
    run_codeql: bool = False

    @field_validator("repository")
    @classmethod
    def reject_ambiguous_repository_path(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("repository path contains a NUL byte")
        return value

    @field_validator("verification_commands")
    @classmethod
    def validate_commands(cls, commands: list[list[str]]) -> list[list[str]]:
        AgentExecutionRequest.validate_commands(commands)
        return commands

    @model_validator(mode="after")
    def validate_assertion_paths(self) -> "RepositoryAgentTask":
        paths = [
            *self.expected_changed_files,
            *self.required_substrings,
            *self.forbidden_substrings,
            *self.pinned_files,
        ]
        for value in paths:
            normalized = value.replace("\\", "/")
            parts = normalized.split("/")
            if (
                normalized.startswith("/")
                or Path(normalized).drive
                or (len(parts[0]) >= 2 and parts[0][1] == ":")
                or any(part in {"", ".", ".."} for part in parts)
                or any(ord(character) < 32 for character in normalized)
            ):
                raise ValueError(f"unsafe task assertion path: {value}")
        return self


class AgentTaskScore(StrictModel):
    task_id: str
    category: str
    accepted: bool
    applied: bool
    source_immutable: bool
    expected_files_changed: bool
    content_assertions_passed: bool
    tests_passed: bool
    security_passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    attempts: int
    duration_ms: float
    score: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class AgentScorecardReport(StrictModel):
    status: Literal["passed", "failed"]
    policy_mode: Literal["strict", "development"]
    task_count: int
    architecture_reliability_score: float
    workflow_completion_score: float
    security_detection_fix_score: float
    short_prompt_understanding_score: float
    sandbox_test_pass_rate: float
    regression_count: int
    average_confidence: float
    average_score: float
    model_backend: str = ""
    model_evidence: dict[str, str] = Field(default_factory=dict)
    task_suite_sha256: str = ""
    tasks: list[AgentTaskScore]
    failures: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)


class AgentScorecardRunner:
    """Execute repository tasks through the same production agent service."""

    def __init__(
        self,
        *,
        backend: ModelBackend | None = None,
        policy_mode: Literal["strict", "development"] = "strict",
        max_revisions: int = 3,
        confidence_threshold: float = 0.85,
        concurrency: int = 4,
    ) -> None:
        if not 1 <= concurrency <= 16:
            raise ValueError("scorecard concurrency must be between 1 and 16")
        self.backend = backend
        self.policy_mode = policy_mode
        self.max_revisions = max_revisions
        self.confidence_threshold = confidence_threshold
        self.concurrency = concurrency

    async def run(
        self,
        *,
        tasks_path: Path,
        output_dir: Path,
        repository_root: Path | None = None,
    ) -> AgentScorecardReport:
        tasks_path = tasks_path.resolve(strict=True)
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = self._load_tasks(tasks_path)
        self._short_task_ids = {task.task_id for task in tasks if task.short_prompt}
        self._validate_suite(tasks)
        root = (repository_root or tasks_path.parent).resolve(strict=True)
        allowed_roots = self._allowed_roots(root)
        backend = self.backend or build_active_backend()
        owns_backend = self.backend is None
        if self.policy_mode == "strict" and backend.name == "mock":
            if owns_backend:
                await backend.aclose()
            raise RuntimeError("strict scorecard requires a non-mock Craftly scratch model backend")
        model_evidence = await self._model_evidence(
            backend,
            require_complete=self.policy_mode == "strict",
        )

        scores: list[AgentTaskScore] = []
        failures: list[str] = []
        try:
            semaphore = asyncio.Semaphore(self.concurrency)

            async def guarded(task: RepositoryAgentTask) -> AgentTaskScore:
                source = self._resolve_repository(task.repository, root, allowed_roots)
                source_before = self._tree_hash(source)
                async with semaphore:
                    try:
                        return await self._run_task(task, source, backend, output_dir)
                    except Exception as exc:
                        return AgentTaskScore(
                            task_id=task.task_id,
                            category=task.category,
                            accepted=False,
                            applied=False,
                            source_immutable=source_before == self._tree_hash(source),
                            expected_files_changed=False,
                            content_assertions_passed=False,
                            tests_passed=False,
                            security_passed=False,
                            confidence=0.0,
                            attempts=0,
                            duration_ms=0.0,
                            score=0.0,
                            errors=[f"{type(exc).__name__}: {exc}"],
                        )

            scores = list(await asyncio.gather(*(guarded(task) for task in tasks)))
            for task, score in zip(tasks, scores, strict=True):
                if score.errors:
                    failures.append(f"{task.task_id}: {'; '.join(score.errors)}")
        finally:
            if owns_backend:
                await backend.aclose()

        report = self._aggregate(
            scores,
            failures,
            model_backend=backend.name,
            model_evidence=model_evidence,
            task_suite_sha256=sha256_file(tasks_path),
        )
        self._write_reports(output_dir, report)
        return report

    async def _run_task(
        self,
        task: RepositoryAgentTask,
        source: Path,
        backend: ModelBackend,
        output_dir: Path,
    ) -> AgentTaskScore:
        started = time.perf_counter()
        source_before = self._tree_hash(source)
        with tempfile.TemporaryDirectory(prefix=f"craftly-score-{task.task_id[:24]}-") as temporary:
            worktree = Path(temporary) / "repository"
            await asyncio.to_thread(stage_repository_copy, source, worktree)
            before = self._file_hashes(worktree)
            store = LocalStore(Path(temporary) / "scorecard.sqlite3")
            project: dict[str, Any] | None = None
            try:
                project = store.create_project(
                    name=f"scorecard-{task.task_id}",
                    repo_path=str(worktree),
                    default_branch="main",
                )
                request = AgentExecutionRequest(
                    project_id=str(project["id"]),
                    prompt=task.prompt,
                    policy_mode=self.policy_mode,
                    verification_commands=task.verification_commands,
                    pinned_files=task.pinned_files,
                    max_revisions=self.max_revisions,
                    apply_on_accept=True,
                    require_sandbox=self.policy_mode == "strict",
                    allow_local_test_fallback=self.policy_mode == "development",
                    run_semgrep=task.run_semgrep,
                    run_codeql=task.run_codeql,
                    fail_on_scanner_unavailable=self.policy_mode == "strict",
                    confidence_threshold=self.confidence_threshold,
                )
                report = await AgentExecutionService(store, backend).execute(request)
                task_output = output_dir / "tasks" / task.task_id
                task_output.mkdir(parents=True, exist_ok=True)
                self._atomic_text(
                    task_output / "execution_report.json",
                    json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                )
                after = self._file_hashes(worktree)
                changed = {
                    path
                    for path in set(before) | set(after)
                    if before.get(path) != after.get(path)
                }
                expected_files = set(task.expected_changed_files)
                expected_files_changed = not expected_files or expected_files.issubset(changed)
                content_passed, assertion_errors = self._content_assertions(task, worktree)
                final_attempt = report.attempts[-1] if report.attempts else None
                tests_passed = bool(final_attempt and final_attempt.test_passed)
                security_passed = bool(final_attempt and final_attempt.security_passed)
                source_immutable = source_before == self._tree_hash(source)
                errors = list(assertion_errors)
                if not source_immutable:
                    errors.append("scorecard mutated the source repository")
                if report.accepted and not expected_files_changed:
                    errors.append("accepted patch did not change every expected file")
                objective = [
                    report.accepted,
                    report.applied,
                    source_immutable,
                    expected_files_changed,
                    content_passed,
                    tests_passed,
                    security_passed,
                ]
                numeric_score = sum(1.0 for value in objective if value) / len(objective)
                return AgentTaskScore(
                    task_id=task.task_id,
                    category=task.category,
                    accepted=report.accepted,
                    applied=report.applied,
                    source_immutable=source_immutable,
                    expected_files_changed=expected_files_changed,
                    content_assertions_passed=content_passed,
                    tests_passed=tests_passed,
                    security_passed=security_passed,
                    confidence=report.confidence,
                    attempts=len(report.attempts),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    score=numeric_score,
                    errors=errors,
                )
            finally:
                if project is not None:
                    store.delete_project(str(project["id"]))
                store.close()

    def _aggregate(
        self,
        scores: list[AgentTaskScore],
        failures: list[str],
        *,
        model_backend: str,
        model_evidence: dict[str, str],
        task_suite_sha256: str,
    ) -> AgentScorecardReport:
        def rate(values: list[bool]) -> float:
            return sum(1 for value in values if value) / len(values) if values else 0.0

        security = [item for item in scores if item.category in {"security", "patch"}]
        short = [item for item in scores if self._task_short(item.task_id)]
        reliability = rate([item.source_immutable and not item.errors for item in scores])
        workflow = rate([item.accepted and item.applied for item in scores])
        security_score = rate([item.security_passed and item.tests_passed for item in security])
        short_score = rate([item.accepted and item.tests_passed for item in short])
        sandbox_rate = rate([item.tests_passed for item in scores])
        regressions = sum(
            1
            for item in scores
            if item.accepted and (not item.tests_passed or not item.security_passed or not item.content_assertions_passed)
        )
        average_score = sum(item.score for item in scores) / len(scores) if scores else 0.0
        average_confidence = sum(item.confidence for item in scores) / len(scores) if scores else 0.0
        required = 0.80 if self.policy_mode == "strict" else 0.0
        passed = (
            bool(scores)
            and not failures
            and regressions == 0
            and reliability >= (0.95 if self.policy_mode == "strict" else 0.0)
            and workflow >= required
            and security_score >= required
            and short_score >= required
            and sandbox_rate >= required
        )
        return AgentScorecardReport(
            status="passed" if passed else "failed",
            policy_mode=self.policy_mode,
            task_count=len(scores),
            architecture_reliability_score=reliability,
            workflow_completion_score=workflow,
            security_detection_fix_score=security_score,
            short_prompt_understanding_score=short_score,
            sandbox_test_pass_rate=sandbox_rate,
            regression_count=regressions,
            average_confidence=average_confidence,
            average_score=average_score,
            model_backend=model_backend,
            model_evidence=model_evidence,
            task_suite_sha256=task_suite_sha256,
            tasks=scores,
            failures=failures,
        )

    @staticmethod
    async def _model_evidence(
        backend: ModelBackend,
        *,
        require_complete: bool,
    ) -> dict[str, str]:
        profile_payload = load_active_profile()
        profile = profile_payload.get("profile") if isinstance(profile_payload.get("profile"), dict) else {}
        evidence = dict(profile.get("evidence") or {}) if isinstance(profile, dict) else {}
        profile_source = active_profile_path()
        checkpoint_value = str(profile.get("checkpoint_manifest") or "") if isinstance(profile, dict) else ""
        tokenizer_value = str(profile.get("tokenizer_path") or "") if isinstance(profile, dict) else ""

        if profile_source.is_file():
            evidence["active_profile_sha256"] = sha256_file(profile_source)
        if checkpoint_value:
            checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
            actual_checkpoint_hash = sha256_file(checkpoint)
            if evidence.get("checkpoint_manifest_sha256") != actual_checkpoint_hash:
                raise RuntimeError("active model profile checkpoint evidence does not match the checkpoint manifest")
        if tokenizer_value:
            tokenizer = Path(tokenizer_value).expanduser().resolve(strict=True)
            actual_tokenizer_hash = sha256_file(tokenizer)
            if evidence.get("tokenizer_sha256") != actual_tokenizer_hash:
                raise RuntimeError("active model profile tokenizer evidence does not match the tokenizer")

        if require_complete:
            required = {
                "active_profile_sha256",
                "checkpoint_manifest_sha256",
                "tokenizer_sha256",
                "evaluation_report_sha256",
            }
            missing = sorted(
                key
                for key in required
                if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get(key) or ""))
            )
            if missing:
                raise RuntimeError(
                    "strict scorecard requires an evidence-bound validation model profile; missing "
                    + ", ".join(missing)
                )
            if backend.name != "craftly_serving":
                raise RuntimeError("strict scorecard requires the evidence-bound Craftly serving backend")
            identity = await backend.identity()
            if identity.get("checkpoint_manifest_sha256") != evidence["checkpoint_manifest_sha256"]:
                raise RuntimeError("serving checkpoint identity does not match the active model profile")
            if identity.get("tokenizer_sha256") != evidence["tokenizer_sha256"]:
                raise RuntimeError("serving tokenizer identity does not match the active model profile")
            identity_evidence = {
                "status": identity.get("status"),
                "model_name": identity.get("model_name"),
                "checkpoint_manifest_sha256": identity.get("checkpoint_manifest_sha256"),
                "tokenizer_sha256": identity.get("tokenizer_sha256"),
                "scratch_only": identity.get("scratch_only"),
            }
            evidence["serving_identity_sha256"] = hashlib.sha256(
                json.dumps(
                    identity_evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        return {str(key): str(value) for key, value in evidence.items()}

    def _load_tasks(self, path: Path) -> list[RepositoryAgentTask]:
        tasks: list[RepositoryAgentTask] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                tasks.append(RepositoryAgentTask.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid scorecard task at {path}:{line_number}: {exc}") from exc
        return tasks

    def _validate_suite(self, tasks: list[RepositoryAgentTask]) -> None:
        minimum = 50 if self.policy_mode == "strict" else 1
        if not minimum <= len(tasks) <= 100:
            raise ValueError(f"{self.policy_mode} scorecard requires {minimum}-100 repository tasks")
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("scorecard task IDs must be unique")
        categories = {task.category for task in tasks}
        if self.policy_mode == "strict" and not {"coding", "debugging", "security", "patch", "long_context"}.issubset(categories):
            raise ValueError("strict scorecard must cover coding, debugging, security, patch, and long_context")
        if self.policy_mode == "strict" and any(not task.run_semgrep and not task.run_codeql for task in tasks):
            raise ValueError("every strict scorecard task must enable Semgrep or CodeQL")
        if self.policy_mode == "strict":
            for category in ["coding", "debugging", "security", "patch", "long_context"]:
                if sum(task.category == category for task in tasks) < 10:
                    raise ValueError(f"strict scorecard requires at least 10 {category} tasks")
            if sum(task.short_prompt for task in tasks) < 10:
                raise ValueError("strict scorecard requires at least 10 short-prompt tasks")
            if any(not task.expected_changed_files for task in tasks):
                raise ValueError("strict scorecard tasks require expected_changed_files")
            if any(not task.required_substrings and not task.forbidden_substrings for task in tasks):
                raise ValueError("strict scorecard tasks require at least one content assertion")

    @staticmethod
    def _allowed_roots(default: Path) -> list[Path]:
        configured = [item for item in os.environ.get("CRAFTLY_SCORECARD_REPO_ROOTS", "").split(os.pathsep) if item]
        return [Path(item).expanduser().resolve(strict=True) for item in configured] or [default]

    @staticmethod
    def _resolve_repository(value: str, default_root: Path, allowed_roots: list[Path]) -> Path:
        candidate = Path(value).expanduser()
        candidate = candidate.resolve(strict=True) if candidate.is_absolute() else (default_root / candidate).resolve(strict=True)
        if not candidate.is_dir():
            raise ValueError(f"scorecard repository is not a directory: {candidate}")
        if not any(candidate == root or root in candidate.parents for root in allowed_roots):
            raise ValueError(f"scorecard repository is outside CRAFTLY_SCORECARD_REPO_ROOTS: {candidate}")
        return candidate

    @staticmethod
    def _file_hashes(root: Path) -> dict[str, str]:
        output: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(root).as_posix()
                output[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return output

    @classmethod
    def _tree_hash(cls, root: Path) -> str:
        digest = hashlib.sha256()
        for path, value in sorted(cls._file_hashes(root).items()):
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _content_assertions(task: RepositoryAgentTask, root: Path) -> tuple[bool, list[str]]:
        errors: list[str] = []
        for relative, required in task.required_substrings.items():
            path = root / relative
            if not path.is_file():
                errors.append(f"required assertion file missing: {relative}")
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            for value in required:
                if value not in content:
                    errors.append(f"{relative} is missing required content: {value[:120]}")
        for relative, forbidden in task.forbidden_substrings.items():
            path = root / relative
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            for value in forbidden:
                if value in content:
                    errors.append(f"{relative} contains forbidden content: {value[:120]}")
        return not errors, errors

    def _task_short(self, task_id: str) -> bool:
        # The task flag is folded into the ID set during aggregation by callers.
        return task_id in self._short_task_ids

    @property
    def _short_task_ids(self) -> set[str]:
        return getattr(self, "__short_task_ids", set())

    @_short_task_ids.setter
    def _short_task_ids(self, value: set[str]) -> None:
        setattr(self, "__short_task_ids", value)

    def _write_reports(self, output_dir: Path, report: AgentScorecardReport) -> None:
        payload = report.model_dump(mode="json")
        self._atomic_text(output_dir / "agent_scorecard.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Craftly Repository Agent Scorecard",
            "",
            f"- Status: **{report.status}**",
            f"- Tasks: {report.task_count}",
            f"- Architecture reliability: {report.architecture_reliability_score:.2%}",
            f"- Workflow completion: {report.workflow_completion_score:.2%}",
            f"- Security detection/fix: {report.security_detection_fix_score:.2%}",
            f"- Short prompt understanding: {report.short_prompt_understanding_score:.2%}",
            f"- Sandbox/test pass rate: {report.sandbox_test_pass_rate:.2%}",
            f"- Regressions: {report.regression_count}",
            "",
            "| Task | Category | Accepted | Tests | Security | Score |",
            "|---|---|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {item.task_id} | {item.category} | {item.accepted} | {item.tests_passed} | "
            f"{item.security_passed} | {item.score:.2%} |"
            for item in report.tasks
        )
        self._atomic_text(output_dir / "agent_scorecard.md", "\n".join(lines) + "\n")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)


async def _run_cli(args: argparse.Namespace) -> AgentScorecardReport:
    runner = AgentScorecardRunner(
        policy_mode=args.policy_mode,
        max_revisions=args.max_revisions,
        confidence_threshold=args.confidence_threshold,
        concurrency=args.concurrency,
    )
    tasks = Path(args.tasks).resolve(strict=True)
    return await runner.run(
        tasks_path=tasks,
        output_dir=Path(args.output_dir),
        repository_root=Path(args.repository_root) if args.repository_root else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-100 real repository tasks through the Craftly agent")
    parser.add_argument("--tasks", required=True, help="JSONL repository task suite")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root")
    parser.add_argument("--policy-mode", choices=["strict", "development"], default="strict")
    parser.add_argument("--max-revisions", type=int, default=3, choices=range(0, 4))
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    report = asyncio.run(_run_cli(args))
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
