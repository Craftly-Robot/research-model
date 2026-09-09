"""MVP verifier runtime for tests and defensive static checks."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.shared.config import ROOT
from src.craftly.shared.config_contracts import (
    VerifierPolicyContract,
    load_verifier_policy_contract,
)
from src.craftly.shared.schemas import StrictModel
from src.craftly.tools import HardenedToolExecutor, SecurityScanner, ToolExecuteRequest
from src.craftly.tools.policy import project_root

SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
]


class VerificationRequest(StrictModel):
    project_id: str
    run_id: str | None = None
    patch_id: str | None = None
    policy_profile: str | None = None
    commands: list[list[str]] = Field(default_factory=list)
    run_secret_scan: bool = True
    run_semgrep: bool = False
    run_codeql: bool = False
    codeql_database: str | None = None
    fail_on_tool_unavailable: bool = False
    timeout_ms: int = Field(default=60_000, ge=1_000, le=300_000)


class VerificationResponse(StrictModel):
    verification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    run_id: str | None = None
    patch_id: str | None = None
    status: str
    verdict: str
    reason: str
    test_results: list[dict[str, Any]]
    security_results: list[dict[str, Any]]
    duration_ms: float


class GuardrailReview(StrictModel):
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    risks: list[str]
    issues: list[str] = Field(default_factory=list)
    engine: str = "native"


class VerifierRuntime:
    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store or LocalStore()

    def run(self, request: VerificationRequest) -> VerificationResponse:
        started = time.perf_counter()
        request = self._apply_policy_profile(request)
        tool = HardenedToolExecutor(self.store)
        test_results = []
        for command in request.commands:
            result = tool.execute(
                ToolExecuteRequest(
                    project_id=request.project_id,
                    run_id=request.run_id,
                    tool="test",
                    command=command,
                    timeout_ms=request.timeout_ms,
                )
            )
            test_results.append(result.model_dump())
        security_results = []
        if request.run_secret_scan:
            security_results.append(self.secret_scan(request.project_id))
        scanner = SecurityScanner(project_root(self.store, request.project_id))
        if request.run_semgrep:
            security_results.append(scanner.run_semgrep(timeout_ms=request.timeout_ms).model_dump())
        if request.run_codeql:
            security_results.append(
                scanner.run_codeql(
                    database=request.codeql_database,
                    timeout_ms=request.timeout_ms,
                ).model_dump()
            )
        failed_tests = [item for item in test_results if item["status"] != "ok"]
        failed_security = [item for item in security_results if item["status"] == "failed"]
        unavailable_security = [
            item
            for item in security_results
            if item["status"] in {"skipped", "timeout"} and request.fail_on_tool_unavailable
        ]
        status = (
            "passed"
            if not failed_tests and not failed_security and not unavailable_security
            else "failed"
        )
        return VerificationResponse(
            project_id=request.project_id,
            run_id=request.run_id,
            patch_id=request.patch_id,
            status=status,
            verdict="accept" if status == "passed" else "reject",
            reason="all configured verification checks passed"
            if status == "passed"
            else "one or more verification checks failed",
            test_results=test_results,
            security_results=security_results,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _apply_policy_profile(self, request: VerificationRequest) -> VerificationRequest:
        if not request.policy_profile:
            return request
        policy = load_verifier_policy()
        profile = policy.profiles.get(request.policy_profile)
        if profile is None:
            raise ValueError(f"unknown verifier policy profile: {request.policy_profile}")
        updates = {
            "run_secret_scan": profile.run_secret_scan,
            "run_semgrep": profile.run_semgrep,
            "run_codeql": profile.run_codeql,
            "fail_on_tool_unavailable": profile.fail_on_tool_unavailable,
            "timeout_ms": min(request.timeout_ms, profile.timeout_ms),
        }
        if profile.run_sandbox and not request.commands:
            updates["commands"] = [profile.sandbox_command.split()]
        return request.model_copy(update=updates)

    def secret_scan(self, project_id: str) -> dict[str, Any]:
        if self.store.get_project(project_id) is None:
            raise KeyError(f"unknown project: {project_id}")
        findings = []
        for chunk in self.store.list_chunks(project_id):
            for line_no, line in enumerate(
                str(chunk["content"]).splitlines(), start=chunk["start_line"]
            ):
                if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                    findings.append(
                        {
                            "path": chunk["path"],
                            "line": line_no,
                            "title": "possible secret",
                            "evidence": line[:200],
                        }
                    )
        return {
            "tool": "secret_scan",
            "status": "failed" if findings else "passed",
            "findings": findings,
            "finding_count": len(findings),
        }

    def strict_review(self, prompt: str) -> GuardrailReview:
        lowered = prompt.lower()
        risks = [
            term
            for term in [
                "delete",
                "secret",
                "password",
                "token",
                "unsafe",
                "eval",
                "exploit",
                "malware",
            ]
            if term in lowered
        ]
        confidence = 0.9 if not risks else 0.65
        return GuardrailReview(accepted=confidence >= 0.6, confidence=confidence, risks=risks)

    def critic_review(self, artifact: str, *, prompt: str = "") -> GuardrailReview:
        issues: list[str] = []
        if "TODO" in artifact:
            issues.append("artifact contains TODO")
        if len(artifact.strip()) < 20:
            issues.append("artifact is too small to validate")
        if prompt and "security" in prompt.lower() and "test" not in artifact.lower():
            issues.append("security-related artifact does not mention tests")
        confidence = 0.9 if not issues else 0.55
        return GuardrailReview(
            accepted=confidence >= 0.6, confidence=confidence, risks=[], issues=issues
        )


def load_verifier_policy(path: str | None = None) -> VerifierPolicyContract:
    return load_verifier_policy_contract(
        Path(path) if path else ROOT / "config" / "verifier_policy.json"
    )
