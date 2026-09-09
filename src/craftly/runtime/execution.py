"""End-to-end role worker execution for Craftly coding-agent runs.

Candidate changes are evaluated in an ephemeral repository copy. The original
workspace is changed only after tests, defensive scanners, critic review, and
the pure verifier all accept the same patch revision.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from src.craftly.db import LocalStore
from src.craftly.indexing import ContextBuilder, RepositoryIndexer
from src.craftly.model_ops.backends import ModelBackend, build_active_backend
from src.craftly.patches import FileEdit, PatchPreviewRequest, PatchService
from src.craftly.planning.engine import IntentPlanningEngine, PlanningResult
from src.craftly.runtime.collaboration import CriticScore as CriticArtifact
from src.craftly.runtime.collaboration import FailureIntelligence
from src.craftly.runtime.taskgraph import AgentRunCreateRequest, TaskGraphRuntime
from src.craftly.shared.schemas import StrictModel
from src.craftly.tools import (
    DockerSandboxRunner,
    HardenedSandboxPolicy,
    SandboxRunRequest,
    SecurityScanner,
)
from src.craftly.tools.policy import project_root
from src.craftly.verifier import VerificationRequest, VerifierRuntime

if TYPE_CHECKING:
    from src.craftly.runtime.engine import AgentWorkerPool


MAX_PATCH_FILES = 50
MAX_PATCH_BYTES = 5_000_000
MAX_STAGE_FILES = 20_000
MAX_STAGE_BYTES = 256_000_000
IGNORED_STAGE_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "artifacts",
}


def stage_repository_copy(source: Path, destination: Path) -> None:
    """Copy a bounded repository tree without links or generated directories."""

    source = source.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=False)
    file_count = 0
    total_bytes = 0
    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        directories[:] = [
            name
            for name in directories
            if name not in IGNORED_STAGE_DIRECTORIES
            and not (root_path / name).is_symlink()
        ]
        relative_root = root_path.relative_to(source)
        target_root = destination / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            source_file = root_path / name
            if source_file.is_symlink() or not source_file.is_file():
                continue
            file_count += 1
            total_bytes += source_file.stat().st_size
            if file_count > MAX_STAGE_FILES or total_bytes > MAX_STAGE_BYTES:
                raise ValueError("repository exceeds secure staging limits")
            shutil.copy2(source_file, target_root / name)


class AgentExecutionRequest(StrictModel):
    project_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=32_000)
    mode: Literal["code_edit", "debug", "explain", "security_review"] = "code_edit"
    policy_mode: Literal["strict", "development"] = "strict"
    verification_commands: list[list[str]] = Field(default_factory=list, max_length=20)
    pinned_files: list[str] = Field(default_factory=list, max_length=100)
    context_token_budget: int = Field(default=24_000, ge=2_000, le=200_000)
    max_context_chunks: int = Field(default=32, ge=1, le=100)
    max_revisions: int = Field(default=3, ge=0, le=3)
    concurrency: int = Field(default=4, ge=1, le=8)
    task_timeout_seconds: float = Field(default=300.0, ge=5.0, le=1800.0)
    apply_on_accept: bool = True
    require_sandbox: bool = True
    allow_local_test_fallback: bool = False
    sandbox_image: str = Field(default="python:3.12-slim", min_length=1, max_length=200)
    sandbox_timeout_ms: int = Field(default=120_000, ge=1_000, le=300_000)
    run_semgrep: bool = True
    run_codeql: bool = False
    fail_on_scanner_unavailable: bool = True
    confidence_threshold: float = Field(default=0.85, ge=0.6, le=1.0)
    model_max_tokens: int = Field(default=4096, ge=512, le=32_768)
    max_patch_files: int = Field(default=20, ge=1, le=MAX_PATCH_FILES)
    max_patch_bytes: int = Field(default=1_000_000, ge=1_000, le=MAX_PATCH_BYTES)

    @field_validator("verification_commands")
    @classmethod
    def validate_commands(cls, commands: list[list[str]]) -> list[list[str]]:
        for command in commands:
            if not command or len(command) > 100:
                raise ValueError(
                    "each verification command must contain 1-100 argv items"
                )
            if any(
                not isinstance(item, str)
                or not item
                or "\x00" in item
                or len(item) > 4096
                for item in command
            ):
                raise ValueError("verification command contains an invalid argv item")
        return commands

    @field_validator("sandbox_image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        allowed = os.environ.get(
            "CRAFTLY_SANDBOX_IMAGES", "python:3.12-slim,ubuntu:24.04"
        )
        allowlist = {item.strip() for item in allowed.split(",") if item.strip()}
        if value not in allowlist:
            raise ValueError(f"sandbox image is not allowlisted: {value}")
        if (
            "@sha256:" not in value
            and os.environ.get("CRAFTLY_REQUIRE_PINNED_SANDBOX_IMAGE", "0") == "1"
        ):
            raise ValueError("production sandbox image must be pinned by sha256 digest")
        return value

    @model_validator(mode="after")
    def enforce_strict_policy(self) -> "AgentExecutionRequest":
        if self.policy_mode == "strict":
            if not self.require_sandbox:
                raise ValueError("strict execution requires the Docker sandbox")
            if self.allow_local_test_fallback:
                raise ValueError(
                    "strict execution cannot fall back to host test execution"
                )
            if not self.run_semgrep and not self.run_codeql:
                raise ValueError(
                    "strict execution requires at least one defensive scanner"
                )
        return self


class PatchProposal(StrictModel):
    summary: str = Field(min_length=1, max_length=4_000)
    edits: list[FileEdit] = Field(min_length=1, max_length=MAX_PATCH_FILES)
    test_commands: list[list[str]] = Field(default_factory=list, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=50)


class SecurityReviewArtifact(StrictModel):
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list, max_length=100)
    analysis: list[str] = Field(default_factory=list, max_length=100)


class PerformanceReviewArtifact(StrictModel):
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list, max_length=100)


class FinalSummaryArtifact(StrictModel):
    summary: str = Field(min_length=1, max_length=8_000)
    changed_files: list[str] = Field(default_factory=list, max_length=MAX_PATCH_FILES)
    verification_status: Literal["accepted", "rejected"]
    limitations: list[str] = Field(default_factory=list, max_length=100)


class AgentExecutionAttempt(StrictModel):
    attempt: int = Field(ge=0, le=3)
    run_id: str
    task_graph_id: str
    status: str
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    patch_id: str | None = None
    patch_status: str | None = None
    test_passed: bool = False
    security_passed: bool = False
    verifier_reasons: list[str] = Field(default_factory=list)
    failure_record_ids: list[str] = Field(default_factory=list)
    test_evidence: dict[str, Any] = Field(default_factory=dict)
    security_evidence: dict[str, Any] = Field(default_factory=dict)
    critic_review: dict[str, Any] = Field(default_factory=dict)
    verifier_decision: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float


class AgentExecutionReport(StrictModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    prompt: str
    status: str
    accepted: bool
    applied: bool
    final_run_id: str
    final_task_graph_id: str
    final_patch_id: str | None = None
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    attempts: list[AgentExecutionAttempt]
    total_duration_ms: float
    created_at_unix: float = Field(default_factory=time.time)


class JsonModelClient:
    """Serialize model calls and enforce exact Pydantic JSON responses."""

    def __init__(self, backend: ModelBackend, *, max_tokens: int) -> None:
        self.backend = backend
        self.max_tokens = max_tokens
        self._lock = asyncio.Lock()

    async def generate_model(
        self,
        prompt: str,
        schema: type[StrictModel],
        *,
        temperature: float = 0.0,
    ) -> StrictModel:
        async with self._lock:
            raw = await self.backend.generate(
                prompt, temperature=temperature, max_tokens=self.max_tokens
            )
        if raw.lstrip().startswith("```"):
            raise ValueError(
                "model returned a Markdown fence; strict workers require raw JSON"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return schema.model_validate(payload)


class _AttemptState:
    def __init__(
        self,
        *,
        request: AgentExecutionRequest,
        run_id: str,
        graph_id: str,
        original_project_id: str,
        stage_project_id: str,
        stage_root: Path,
        model: JsonModelClient,
        previous_feedback: list[str],
        baseline_security: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.request = request
        self.run_id = run_id
        self.graph_id = graph_id
        self.original_project_id = original_project_id
        self.stage_project_id = stage_project_id
        self.stage_root = stage_root
        self.model = model
        self.previous_feedback = previous_feedback
        self.baseline_security = baseline_security
        self.plan: PlanningResult | None = None
        self.context_prompt = ""
        self.context_report: dict[str, Any] = {}
        self.patch: PatchProposal | None = None
        self.original_patch_id: str | None = None
        self.stage_patch_id: str | None = None
        self.test_evidence: dict[str, Any] = {}
        self.security_evidence: dict[str, Any] = {}
        self.performance_review: dict[str, Any] = {}
        self.critic: dict[str, Any] = {}
        self.verifier: dict[str, Any] = {}
        self.final_answer = ""


class ProductionRoleWorkers:
    """Concrete role workers over existing Craftly services."""

    def __init__(self, store: LocalStore, state: _AttemptState) -> None:
        self.store = store
        self.state = state
        self.planner = IntentPlanningEngine()

    def register(self, pool: AgentWorkerPool) -> None:
        timeout = self.state.request.task_timeout_seconds
        pool.register("understand", self.understand, timeout_seconds=timeout)
        pool.register("planner", self.plan, timeout_seconds=timeout)
        pool.register(
            "retrieve_context", self.retrieve_context, timeout_seconds=timeout
        )
        pool.register("edit", self.edit, timeout_seconds=timeout)
        pool.register("test", self.test, timeout_seconds=timeout)
        pool.register("security_review", self.security_review, timeout_seconds=timeout)
        pool.register(
            "performance_review", self.performance_review, timeout_seconds=timeout
        )
        pool.register("critic_review", self.critic_review, timeout_seconds=timeout)
        pool.register("verify", self.verify, timeout_seconds=timeout)
        pool.register("summarize", self.summarize, timeout_seconds=timeout)

    async def understand(self, _task: dict[str, Any]) -> dict[str, Any]:
        return {
            "intent": self.state.request.mode,
            "requirements": [
                "minimal repository patch",
                "measured tests",
                "defensive security verification",
            ],
            "constraints": {
                "original_workspace_mutation_before_accept": False,
                "max_revisions": self.state.request.max_revisions,
                "sandbox_required": self.state.request.require_sandbox,
            },
        }

    async def plan(self, _task: dict[str, Any]) -> dict[str, Any]:
        self.state.plan = await self.planner.plan_structured(
            self.state.request.prompt,
            backend=self.state.model.backend,
            run_id=self.state.run_id,
            allow_dev_fallback=self.state.request.policy_mode == "development",
        )
        return self.state.plan.model_dump()

    async def retrieve_context(self, _task: dict[str, Any]) -> dict[str, Any]:
        report = await asyncio.to_thread(
            ContextBuilder(self.store).build,
            project_id=self.state.original_project_id,
            query=self.state.request.prompt,
            token_budget=self.state.request.context_token_budget,
            pinned_files=self.state.request.pinned_files,
            max_chunks=self.state.request.max_context_chunks,
        )
        self.state.context_prompt = report.prompt_context
        self.state.context_report = report.model_dump(mode="json")
        return {
            "context_id": report.context_id,
            "context_report_sha256": report.report_sha256,
            "index_revision": report.index_revision,
            "retrieval_mode": report.retrieval_mode,
            "degraded": report.degraded,
            "degraded_reason": report.degraded_reason,
            "estimated_tokens": report.estimated_tokens,
            "files": report.files,
            "chunk_ids": [chunk.chunk_id for chunk in report.chunks],
            "evidence_ids": [chunk.evidence_id for chunk in report.chunks],
        }

    async def edit(self, _task: dict[str, Any]) -> dict[str, Any]:
        if self.state.plan is None or not self.state.context_prompt:
            raise RuntimeError("coder requires completed plan and repository context")
        feedback = (
            "\n".join(f"- {item}" for item in self.state.previous_feedback) or "- none"
        )
        prompt = (
            "You are the Craftly Coder worker. Return only JSON matching:\n"
            '{"summary":"string","edits":[{"path":"relative/path","new_content":"complete file content"}],'
            '"test_commands":[["executable","arg"]],"assumptions":["string"]}\n'
            "Produce minimal complete-file edits. Do not claim tests passed. Do not edit .git or escape the repository.\n\n"
            f"PLAN:\n{json.dumps(self.state.plan.model_dump(), sort_keys=True)}\n"
            f"PREVIOUS VERIFIED FEEDBACK:\n{feedback}\n"
            f"REPOSITORY CONTEXT:\n{self.state.context_prompt}"
        )
        proposal = await self.state.model.generate_model(
            prompt, PatchProposal, temperature=0.1
        )
        if not isinstance(proposal, PatchProposal):
            raise TypeError("coder schema validation failed")
        self._validate_patch(proposal)
        self.state.patch = proposal
        original_patch = await asyncio.to_thread(
            PatchService(self.store).preview,
            PatchPreviewRequest(
                project_id=self.state.original_project_id,
                run_id=self.state.run_id,
                edits=proposal.edits,
            ),
        )
        self.state.original_patch_id = original_patch.patch_id
        stage_patch = await asyncio.to_thread(
            PatchService(self.store).preview,
            PatchPreviewRequest(
                project_id=self.state.stage_project_id,
                run_id=None,
                edits=proposal.edits,
            ),
        )
        await asyncio.to_thread(PatchService(self.store).apply, stage_patch.patch_id)
        self.state.stage_patch_id = stage_patch.patch_id
        await asyncio.to_thread(
            RepositoryIndexer(self.store).index_project,
            project_id=self.state.stage_project_id,
        )
        return {
            "summary": proposal.summary,
            "patch_id": original_patch.patch_id,
            "stage_patch_id": stage_patch.patch_id,
            "files_changed": original_patch.files_changed,
            "diff": original_patch.diff,
            "assumptions": proposal.assumptions,
        }

    async def test(self, _task: dict[str, Any]) -> dict[str, Any]:
        if self.state.patch is None:
            raise RuntimeError("tester requires a staged patch")
        commands = (
            self.state.request.verification_commands or self.state.patch.test_commands
        )
        if not commands:
            self.state.test_evidence = {
                "passed": False,
                "status": "failed",
                "reason": "no verification commands were supplied or generated",
                "results": [],
                "evidence_refs": [],
            }
            return self.state.test_evidence
        results: list[dict[str, Any]] = []
        if self.state.request.require_sandbox:
            files = await asyncio.to_thread(self._sandbox_files)
            for command in commands:
                result = await asyncio.to_thread(
                    DockerSandboxRunner().run,
                    SandboxRunRequest(
                        command=command,
                        files=files,
                        policy=HardenedSandboxPolicy(
                            image=self.state.request.sandbox_image,
                            timeout_ms=self.state.request.sandbox_timeout_ms,
                        ),
                    ),
                )
                results.append(result.model_dump())
            unavailable = any(item["status"] == "unavailable" for item in results)
            if unavailable and self.state.request.allow_local_test_fallback:
                results = await self._run_local_tests(commands)
        else:
            results = await self._run_local_tests(commands)
        passed = bool(results) and all(
            item.get("status") in {"ok", "passed"} and item.get("exit_code") == 0
            for item in results
        )
        refs = [str(uuid.uuid4()) for _ in results]
        self.state.test_evidence = {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "results": results,
            "evidence_refs": refs,
        }
        return self.state.test_evidence

    async def security_review(self, _task: dict[str, Any]) -> dict[str, Any]:
        scans = await self._scan_workspace(self.state.stage_root)
        new_findings: list[dict[str, Any]] = []
        unavailable: list[str] = []
        for tool, results in scans.items():
            baseline = {
                self._finding_fingerprint(finding)
                for scan in self.state.baseline_security.get(tool, [])
                for finding in scan.get("findings", [])
            }
            for result in results:
                status = str(result.get("status") or "")
                execution_failed = (
                    status == "failed"
                    and not result.get("findings")
                    and result.get("exit_code") not in {0, None}
                )
                if status in {"skipped", "timeout", "unavailable"} or execution_failed:
                    unavailable.append(f"{tool}:{status}")
                for finding in result.get("findings", []):
                    if self._finding_fingerprint(finding) not in baseline:
                        new_findings.append({"tool": tool, **finding})
        model_prompt = (
            "You are the Craftly defensive Security Reviewer. Analyze measured scanner output and the patch. "
            "Return only JSON: "
            '{"accepted":bool,"confidence":0.0,"issues":["string"],"analysis":["string"]}. '
            "Never claim a scan passed if its measured status is skipped, timeout, or failed.\n\n"
            f"PATCH:\n{json.dumps(self.state.patch.model_dump() if self.state.patch else {}, sort_keys=True)}\n"
            f"SCANS:\n{json.dumps(scans, sort_keys=True)[:80_000]}"
        )
        review = await self.state.model.generate_model(
            model_prompt, SecurityReviewArtifact
        )
        if not isinstance(review, SecurityReviewArtifact):
            raise TypeError("security reviewer schema validation failed")
        scanner_gate = not new_findings and not (
            unavailable and self.state.request.fail_on_scanner_unavailable
        )
        accepted = review.accepted and scanner_gate
        issues = list(review.issues)
        if new_findings:
            issues.append(f"{len(new_findings)} new scanner finding(s) introduced")
        if unavailable and self.state.request.fail_on_scanner_unavailable:
            issues.append(f"required scanner unavailable: {', '.join(unavailable)}")
        refs = [str(uuid.uuid4()) for _ in scans]
        self.state.security_evidence = {
            "accepted": accepted,
            "confidence": min(review.confidence, 0.99 if scanner_gate else 0.0),
            "issues": issues,
            "analysis": review.analysis,
            "new_findings": new_findings,
            "scanner_results": scans,
            "evidence_refs": refs,
        }
        return self.state.security_evidence

    async def performance_review(self, _task: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are the Craftly performance and maintainability reviewer. Return only JSON: "
            '{"accepted":bool,"confidence":0.0,"issues":["string"]}. '
            "Reject unnecessary complexity, obvious unbounded work, resource leaks, and incompatible public API changes.\n\n"
            f"PATCH:\n{json.dumps(self.state.patch.model_dump() if self.state.patch else {}, sort_keys=True)[:80_000]}"
        )
        review = await self.state.model.generate_model(
            prompt, PerformanceReviewArtifact
        )
        if not isinstance(review, PerformanceReviewArtifact):
            raise TypeError("performance reviewer schema validation failed")
        self.state.performance_review = review.model_dump()
        return self.state.performance_review

    async def critic_review(self, _task: dict[str, Any]) -> dict[str, Any]:
        evidence = {
            "context": {
                "context_id": self.state.context_report.get("context_id"),
                "report_sha256": self.state.context_report.get("report_sha256"),
                "index_revision": self.state.context_report.get("index_revision"),
                "degraded": self.state.context_report.get("degraded"),
                "evidence_ids": [
                    chunk.get("evidence_id")
                    for chunk in self.state.context_report.get("chunks", [])
                ],
            },
            "test": self.state.test_evidence,
            "security": self.state.security_evidence,
            "performance": self.state.performance_review,
        }
        prompt = (
            "You are the Craftly Critic. Return only JSON matching: "
            '{"confidence":0.0,"flaws":["string"],"assumptions_wrong":["string"],'
            '"failure_modes":["string"],"security_risks":["string"],"unverified_evidence":["string"]}. '
            "Do not provide a replacement patch. Score only the proposal and measured evidence.\n\n"
            f"EVIDENCE:\n{json.dumps(evidence, sort_keys=True)[:100_000]}"
        )
        review = await self.state.model.generate_model(prompt, CriticArtifact)
        if not isinstance(review, CriticArtifact):
            raise TypeError("critic schema validation failed")
        deterministic_confidence = 1.0
        if not self.state.test_evidence.get("passed"):
            deterministic_confidence = 0.0
        if not self.state.security_evidence.get("accepted"):
            deterministic_confidence = min(deterministic_confidence, 0.4)
        if not self.state.performance_review.get("accepted"):
            deterministic_confidence = min(deterministic_confidence, 0.6)
        payload = review.model_dump()
        payload["confidence"] = min(review.confidence, deterministic_confidence)
        self.state.critic = payload
        return payload

    async def verify(self, _task: dict[str, Any]) -> dict[str, Any]:
        criteria = {
            "context_evidence_bound": bool(
                self.state.context_report.get("report_sha256")
                and self.state.context_report.get("index_revision")
            ),
            "patch_exists": bool(self.state.original_patch_id and self.state.patch),
            "tests_passed": self.state.test_evidence.get("passed") is True,
            "security_passed": self.state.security_evidence.get("accepted") is True,
            "performance_passed": self.state.performance_review.get("accepted") is True,
            "critic_confident": float(self.state.critic.get("confidence", 0.0))
            >= self.state.request.confidence_threshold,
        }
        accepted = all(criteria.values())
        refs = [
            *self.state.test_evidence.get("evidence_refs", []),
            *self.state.security_evidence.get("evidence_refs", []),
        ]
        if accepted and not refs:
            accepted = False
            criteria["evidence_present"] = False
        else:
            criteria["evidence_present"] = bool(refs)
        self.state.verifier = {
            "accepted": accepted,
            "criteria": criteria,
            "criteria_passed": [key for key, value in criteria.items() if value],
            "criteria_failed": [key for key, value in criteria.items() if not value],
            "confidence": float(self.state.critic.get("confidence", 0.0)),
            "evidence_refs": refs,
        }
        return self.state.verifier

    async def summarize(self, _task: dict[str, Any]) -> dict[str, Any]:
        accepted = bool(self.state.verifier.get("accepted"))
        prompt = (
            "Return only JSON matching "
            '{"summary":"string","changed_files":["string"],"verification_status":"accepted|rejected",'
            '"limitations":["string"]}. Include only measured verification status. Never claim tests or scanners passed '
            "unless the supplied verifier says so.\n\n"
            f"PATCH SUMMARY: {self.state.patch.summary if self.state.patch else 'none'}\n"
            f"VERIFIER: {json.dumps(self.state.verifier, sort_keys=True)}"
        )
        summary = await self.state.model.generate_model(prompt, FinalSummaryArtifact)
        if not isinstance(summary, FinalSummaryArtifact):
            raise TypeError("summary schema validation failed")
        expected_status = "accepted" if accepted else "rejected"
        if summary.verification_status != expected_status:
            raise ValueError(
                "summary verification status contradicts verifier evidence"
            )
        self.state.final_answer = summary.summary
        return {
            "accepted": accepted,
            **summary.model_dump(),
            "verification": self.state.verifier,
        }

    def _validate_patch(self, proposal: PatchProposal) -> None:
        if len(proposal.edits) > self.state.request.max_patch_files:
            raise ValueError("patch exceeds configured file count")
        total_bytes = sum(
            len(edit.new_content.encode("utf-8")) for edit in proposal.edits
        )
        if total_bytes > self.state.request.max_patch_bytes:
            raise ValueError("patch exceeds configured byte limit")
        paths = [edit.path for edit in proposal.edits]
        if len(set(paths)) != len(paths):
            raise ValueError("patch contains duplicate file paths")
        AgentExecutionRequest.validate_commands(proposal.test_commands)

    async def _run_local_tests(self, commands: list[list[str]]) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(
            VerifierRuntime(self.store).run,
            VerificationRequest(
                project_id=self.state.stage_project_id,
                run_id=self.state.run_id,
                patch_id=self.state.stage_patch_id,
                commands=commands,
                run_secret_scan=False,
                run_semgrep=False,
                run_codeql=False,
                timeout_ms=self.state.request.sandbox_timeout_ms,
            ),
        )
        return response.test_results

    async def _scan_workspace(self, root: Path) -> dict[str, list[dict[str, Any]]]:
        scanner = SecurityScanner(root)
        output: dict[str, list[dict[str, Any]]] = {}
        if self.state.request.run_semgrep:
            result = await asyncio.to_thread(
                scanner.run_semgrep,
                timeout_ms=int(self.state.request.task_timeout_seconds * 1000),
            )
            output["semgrep"] = [result.model_dump()]
        if self.state.request.run_codeql:
            result = await asyncio.to_thread(
                scanner.run_codeql_source,
                timeout_ms=int(self.state.request.task_timeout_seconds * 1000),
            )
            output["codeql"] = [result.model_dump()]
        return output

    def _sandbox_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        total_bytes = 0
        for path in sorted(self.state.stage_root.rglob("*")):
            if len(files) >= MAX_STAGE_FILES:
                raise ValueError(f"sandbox stage exceeds {MAX_STAGE_FILES} files")
            relative = path.relative_to(self.state.stage_root)
            if any(part in IGNORED_STAGE_DIRECTORIES for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            payload = path.read_bytes()
            if b"\x00" in payload[:8192]:
                continue
            total_bytes += len(payload)
            if total_bytes > MAX_STAGE_BYTES:
                raise ValueError(f"sandbox stage exceeds {MAX_STAGE_BYTES} bytes")
            files[relative.as_posix()] = payload.decode("utf-8", errors="replace")
        return files

    @staticmethod
    def _finding_fingerprint(finding: dict[str, Any]) -> str:
        stable = {
            "rule_id": finding.get("rule_id") or finding.get("check_id") or "",
            "path": str(finding.get("path") or "").replace("\\", "/"),
            "line": finding.get("line"),
            "message": finding.get("message") or "",
        }
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True).encode("utf-8")
        ).hexdigest()


class AgentExecutionService:
    """One-command, bounded, verified coding-agent execution service."""

    def __init__(
        self, store: LocalStore | None = None, backend: ModelBackend | None = None
    ) -> None:
        self.store = store or LocalStore()
        self.backend = backend
        self._owns_backend = backend is None

    async def execute(self, request: AgentExecutionRequest) -> AgentExecutionReport:
        started = time.perf_counter()
        project = self.store.get_project(request.project_id)
        if project is None:
            raise KeyError(f"unknown project: {request.project_id}")
        original_root = project_root(self.store, request.project_id)
        await asyncio.to_thread(
            RepositoryIndexer(self.store).index_project, project_id=request.project_id
        )
        backend = self.backend or build_active_backend()
        if request.policy_mode == "strict" and backend.name == "mock":
            if self._owns_backend:
                await backend.aclose()
            raise RuntimeError(
                "strict agent execution requires a non-mock Craftly scratch model backend"
            )
        await asyncio.to_thread(self._preflight, request, original_root)
        model = JsonModelClient(backend, max_tokens=request.model_max_tokens)
        attempts: list[AgentExecutionAttempt] = []
        feedback: list[str] = []
        unresolved_failures: list[str] = []
        final_answer = ""
        final_run_id = ""
        final_graph_id = ""
        final_patch_id: str | None = None
        final_confidence = 0.0
        accepted = applied = False

        try:
            baseline_security = await self._baseline_security(request, original_root)
            if request.fail_on_scanner_unavailable:
                failed_scanners = self._unavailable_scanners(baseline_security)
                if failed_scanners:
                    raise RuntimeError(
                        "required baseline scanner did not produce trustworthy results: "
                        + ", ".join(failed_scanners)
                    )
            for revision in range(request.max_revisions + 1):
                attempt_started = time.perf_counter()
                run = TaskGraphRuntime(self.store).create_agent_run(
                    AgentRunCreateRequest(
                        project_id=request.project_id,
                        session_id=request.session_id,
                        prompt=request.prompt,
                        mode=request.mode,
                        max_steps=12,
                        apply_patch=False,
                        model_profile=backend.name,
                    )
                )
                final_run_id = run.run_id
                final_graph_id = run.task_graph_id
                with tempfile.TemporaryDirectory(
                    prefix=f"craftly-agent-{run.run_id[:8]}-"
                ) as stage_dir:
                    stage_root = Path(stage_dir) / "repo"
                    await asyncio.to_thread(
                        stage_repository_copy, original_root, stage_root
                    )
                    stage_project = self.store.create_project(
                        name=f"stage-{run.run_id}",
                        repo_path=str(stage_root),
                        default_branch=str(project.get("default_branch") or "main"),
                    )
                    state = _AttemptState(
                        request=request,
                        run_id=run.run_id,
                        graph_id=run.task_graph_id,
                        original_project_id=request.project_id,
                        stage_project_id=str(stage_project["id"]),
                        stage_root=stage_root,
                        model=model,
                        previous_feedback=feedback,
                        baseline_security=baseline_security,
                    )
                    try:
                        await asyncio.to_thread(
                            RepositoryIndexer(self.store).index_project,
                            project_id=state.stage_project_id,
                        )
                        from src.craftly.runtime.engine import AgentWorkerPool

                        pool = AgentWorkerPool(
                            TaskGraphRuntime(self.store),
                            concurrency=request.concurrency,
                            lease_seconds=min(
                                600.0, request.task_timeout_seconds + 60.0
                            ),
                        )
                        workers = ProductionRoleWorkers(self.store, state)
                        workers.register(pool)
                        pool_report = await pool.run_until_blocked_or_complete(
                            run.task_graph_id
                        )
                        verify_task = next(
                            (
                                task
                                for task in self.store.list_tasks(run.task_graph_id)
                                if task["kind"] == "verify"
                            ),
                            None,
                        )
                        verifier = (verify_task or {}).get("outputs") or state.verifier
                        accepted = (
                            bool(verifier.get("accepted"))
                            and pool_report.status == "completed"
                        )
                        final_confidence = float(verifier.get("confidence") or 0.0)
                        final_answer = state.final_answer or self._failure_summary(
                            pool_report.status, state
                        )
                        final_patch_id = state.original_patch_id
                        failure_ids = self._record_attempt_failures(request, state)
                        unresolved_failures.extend(failure_ids)
                        patch_status = None
                        if state.original_patch_id:
                            if accepted and request.apply_on_accept:
                                patch = await asyncio.to_thread(
                                    PatchService(self.store).apply,
                                    state.original_patch_id,
                                )
                                applied = True
                                patch_status = patch.status
                                await asyncio.to_thread(
                                    RepositoryIndexer(self.store).index_project,
                                    project_id=request.project_id,
                                )
                                self._resolve_failures(
                                    unresolved_failures,
                                    state,
                                    verification_ref=str(
                                        (verify_task or {})
                                        .get("outputs", {})
                                        .get("agent_message_id")
                                        or run.run_id
                                    ),
                                )
                            elif accepted:
                                patch_status = "preview"
                            else:
                                patch = await asyncio.to_thread(
                                    PatchService(self.store).rollback,
                                    state.original_patch_id,
                                )
                                patch_status = patch.status
                        feedback = self._revision_feedback(state, pool_report.status)
                        status = "accepted" if accepted else "rejected"
                        self.store.update_run_status(
                            run.run_id,
                            "completed" if accepted else "failed",
                            summary=final_answer[:4_000],
                            confidence=final_confidence,
                            finished_at=time.time(),
                        )
                        attempts.append(
                            AgentExecutionAttempt(
                                attempt=revision,
                                run_id=run.run_id,
                                task_graph_id=run.task_graph_id,
                                status=status,
                                accepted=accepted,
                                confidence=final_confidence,
                                patch_id=state.original_patch_id,
                                patch_status=patch_status,
                                test_passed=state.test_evidence.get("passed") is True,
                                security_passed=state.security_evidence.get("accepted")
                                is True,
                                verifier_reasons=list(
                                    state.verifier.get("criteria_failed", [])
                                ),
                                failure_record_ids=failure_ids,
                                test_evidence=state.test_evidence,
                                security_evidence=state.security_evidence,
                                critic_review=state.critic,
                                verifier_decision=state.verifier,
                                duration_ms=(time.perf_counter() - attempt_started)
                                * 1000,
                            )
                        )
                    finally:
                        self.store.delete_project(str(stage_project["id"]))
                if accepted:
                    break
        finally:
            if self._owns_backend:
                await backend.aclose()

        return AgentExecutionReport(
            project_id=request.project_id,
            prompt=request.prompt,
            status="accepted" if accepted else "rejected",
            accepted=accepted,
            applied=applied,
            final_run_id=final_run_id,
            final_task_graph_id=final_graph_id,
            final_patch_id=final_patch_id,
            final_answer=final_answer,
            confidence=final_confidence,
            attempts=attempts,
            total_duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def _baseline_security(
        self,
        request: AgentExecutionRequest,
        root: Path,
    ) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        scanner = SecurityScanner(root)
        if request.run_semgrep:
            result = await asyncio.to_thread(
                scanner.run_semgrep,
                timeout_ms=int(request.task_timeout_seconds * 1000),
            )
            output["semgrep"] = [result.model_dump()]
        if request.run_codeql:
            result = await asyncio.to_thread(
                scanner.run_codeql_source,
                timeout_ms=int(request.task_timeout_seconds * 1000),
            )
            output["codeql"] = [result.model_dump()]
        return output

    @staticmethod
    def _unavailable_scanners(
        scans: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        unavailable: list[str] = []
        for tool, results in scans.items():
            for result in results:
                status = str(result.get("status") or "")
                execution_failed = (
                    status == "failed"
                    and not result.get("findings")
                    and result.get("exit_code") not in {0, None}
                )
                if status in {"skipped", "timeout", "unavailable"} or execution_failed:
                    unavailable.append(f"{tool}:{status}")
        return unavailable

    @staticmethod
    def _preflight(request: AgentExecutionRequest, root: Path) -> None:
        if request.policy_mode != "strict":
            return
        if request.require_sandbox:
            try:
                import docker
            except ImportError as exc:
                raise RuntimeError(
                    "strict agent execution requires the Docker Python SDK"
                ) from exc
            try:
                client = docker.from_env()
                client.ping()
            except Exception as exc:
                raise RuntimeError(
                    f"strict agent execution requires a reachable Docker engine: {exc}"
                ) from exc
            finally:
                if "client" in locals():
                    client.close()
        if (
            request.fail_on_scanner_unavailable
            and request.run_semgrep
            and shutil.which("semgrep") is None
        ):
            raise RuntimeError("strict agent execution requires the Semgrep CLI")
        if (
            request.fail_on_scanner_unavailable
            and request.run_codeql
            and SecurityScanner(root)._codeql_executable() is None
        ):
            raise RuntimeError("strict agent execution requires the CodeQL CLI")

    def _record_attempt_failures(
        self, request: AgentExecutionRequest, state: _AttemptState
    ) -> list[str]:
        failures = []
        details = []
        for result in state.test_evidence.get("results", []):
            if result.get("status") not in {"ok", "passed"}:
                details.append(
                    str(result.get("stderr") or result.get("reason") or "test failed")
                )
        details.extend(str(item) for item in state.security_evidence.get("issues", []))
        details.extend(str(item) for item in state.critic.get("flaws", []))
        intelligence = FailureIntelligence(self.store)
        for detail in details[:20]:
            if not detail.strip():
                continue
            record = intelligence.observe(
                detail,
                project_id=request.project_id,
                run_id=state.run_id,
                task_id=None,
                metadata={"source": "verified_agent_attempt"},
            )
            failures.append(str(record["id"]))
        return failures

    def _resolve_failures(
        self,
        failure_ids: list[str],
        state: _AttemptState,
        *,
        verification_ref: str,
    ) -> None:
        if not state.original_patch_id:
            return
        root_cause = "; ".join(state.critic.get("assumptions_wrong", []))
        if not root_cause:
            root_cause = state.patch.summary if state.patch else "verified repair"
        for failure_id in sorted(set(failure_ids)):
            try:
                FailureIntelligence(self.store).resolve(
                    failure_id,
                    root_cause=root_cause,
                    patch_id=state.original_patch_id,
                    verification_ref=verification_ref,
                    verification_passed=True,
                )
            except (KeyError, ValueError):
                continue

    @staticmethod
    def _revision_feedback(state: _AttemptState, pool_status: str) -> list[str]:
        feedback = [f"worker pool status: {pool_status}"]
        feedback.extend(str(item) for item in state.critic.get("flaws", []))
        feedback.extend(str(item) for item in state.critic.get("assumptions_wrong", []))
        feedback.extend(str(item) for item in state.critic.get("failure_modes", []))
        feedback.extend(str(item) for item in state.critic.get("security_risks", []))
        feedback.extend(
            str(item) for item in state.critic.get("unverified_evidence", [])
        )
        feedback.extend(str(item) for item in state.security_evidence.get("issues", []))
        feedback.extend(
            f"verification failed: {item}"
            for item in state.verifier.get("criteria_failed", [])
        )
        return feedback[:100]

    @staticmethod
    def _failure_summary(pool_status: str, state: _AttemptState) -> str:
        reasons = (
            state.verifier.get("criteria_failed")
            or state.critic.get("flaws")
            or ["worker execution failed"]
        )
        return f"Craftly rejected the patch ({pool_status}): {', '.join(str(item) for item in reasons)}"
