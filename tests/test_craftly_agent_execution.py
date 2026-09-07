from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.craftly.db import LocalStore
from src.craftly.evaluation.agent_scorecard import AgentScorecardRunner, RepositoryAgentTask
from src.craftly.gateway import api as gateway_api
from src.craftly.model_ops.backends import ModelBackend
from src.craftly.patches import FileEdit, PatchPreviewRequest, PatchService
from src.craftly.runtime.execution import AgentExecutionReport, AgentExecutionRequest, AgentExecutionService
from src.craftly.tools import HardenedSandboxPolicy, SandboxRunRequest, SecurityScanner


class DeterministicAgentBackend(ModelBackend):
    name = "craftly-test-checkpoint"

    def __init__(self) -> None:
        self.coder_calls = 0

    async def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if "Return only valid JSON for an Craftly coding-agent plan" in prompt:
            return json.dumps(
                {
                    "goal": "Correct add and verify the repository",
                    "requirements": ["repair calc.py", "run verify.py"],
                    "risks": ["incorrect arithmetic"],
                    "success_criteria": ["verification exits zero"],
                    "expansion": {"intent": "debug"},
                }
            )
        if "Craftly Coder worker" in prompt:
            self.coder_calls += 1
            implementation = "def add(a, b):\n    return a - b\n" if self.coder_calls == 1 else "def add(a, b):\n    return a + b\n"
            return json.dumps(
                {
                    "summary": "Repair addition behavior",
                    "edits": [{"path": "calc.py", "new_content": implementation}],
                    "test_commands": [["python", "-S", "verify.py"]],
                    "assumptions": [],
                }
            )
        if "defensive Security Reviewer" in prompt:
            return json.dumps(
                {
                    "accepted": True,
                    "confidence": 0.98,
                    "issues": [],
                    "analysis": ["No measured defensive regression"],
                }
            )
        if "performance and maintainability reviewer" in prompt:
            return json.dumps({"accepted": True, "confidence": 0.98, "issues": []})
        if "Craftly Critic" in prompt:
            return json.dumps(
                {
                    "confidence": 0.97,
                    "flaws": [],
                    "assumptions_wrong": [],
                    "failure_modes": [],
                    "security_risks": [],
                    "unverified_evidence": [],
                }
            )
        return json.dumps(
            {
                "summary": "Craftly applied the measured patch after repository verification.",
                "changed_files": ["calc.py"],
                "verification_status": "accepted",
                "limitations": [],
            }
        )


class FixtureVerifierRuntime:
    """Fast deterministic verifier; hardened executor has separate real tests."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store

    def run(self, request: object) -> SimpleNamespace:
        project = self.store.get_project(str(getattr(request, "project_id")))
        content = (Path(str(project["repo_path"])) / "calc.py").read_text(encoding="utf-8")
        passed = "return a + b" in content
        return SimpleNamespace(
            test_results=[
                {
                    "status": "ok" if passed else "failed",
                    "exit_code": 0 if passed else 1,
                    "stdout": "verified" if passed else "",
                    "stderr": "" if passed else "assertion failed",
                    "duration_ms": 1.0,
                }
            ]
        )


def make_repository(root: Path) -> None:
    (root / "calc.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")
    (root / "verify.py").write_text(
        "from calc import add\nassert add(2, 3) == 5\nprint('verified')\n",
        encoding="utf-8",
    )


class CraftlyAgentExecutionTest(unittest.TestCase):
    def test_revision_loop_applies_only_verified_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            make_repository(root)
            with LocalStore(Path(temporary) / "state.sqlite3") as store:
                project = store.create_project(name="fixture", repo_path=str(root))
                with patch("src.craftly.runtime.execution.VerifierRuntime", FixtureVerifierRuntime):
                    report = asyncio.run(
                        AgentExecutionService(store, DeterministicAgentBackend()).execute(
                            AgentExecutionRequest(
                                project_id=str(project["id"]),
                                prompt="fix add",
                                policy_mode="development",
                                verification_commands=[["python", "-S", "verify.py"]],
                                max_revisions=3,
                                require_sandbox=False,
                                allow_local_test_fallback=True,
                                run_semgrep=False,
                                run_codeql=False,
                                fail_on_scanner_unavailable=False,
                            )
                        )
                    )
                self.assertTrue(report.accepted)
                self.assertTrue(report.applied)
                self.assertEqual(len(report.attempts), 2)
                self.assertFalse(report.attempts[0].test_passed)
                self.assertEqual(report.attempts[0].patch_status, "rolled_back")
                self.assertTrue(report.attempts[1].test_passed)
                self.assertEqual((root / "calc.py").read_text(encoding="utf-8"), "def add(a, b):\n    return a + b\n")

    def test_patch_rollback_removes_new_file_and_apply_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            with LocalStore(Path(temporary) / "state.sqlite3") as store:
                project = store.create_project(name="fixture", repo_path=str(root))
                service = PatchService(store)
                preview = service.preview(
                    PatchPreviewRequest(
                        project_id=str(project["id"]),
                        edits=[FileEdit(path="nested/new.py", new_content="VALUE = 1\n")],
                    )
                )
                first = service.apply(preview.patch_id)
                second = service.apply(preview.patch_id)
                self.assertEqual(first.status, "applied")
                self.assertEqual(second.status, "applied")
                self.assertTrue((root / "nested" / "new.py").exists())
                rolled_back = service.rollback(preview.patch_id)
                self.assertEqual(rolled_back.status, "rolled_back")
                self.assertFalse((root / "nested" / "new.py").exists())
                self.assertFalse((root / "nested").exists())
                with self.assertRaises(ValidationError):
                    FileEdit(path="C:/outside.py", new_content="")

    def test_rejecting_preview_does_not_touch_original_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            target = root / "stable.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            before = target.stat()
            with LocalStore(Path(temporary) / "state.sqlite3") as store:
                project = store.create_project(name="fixture", repo_path=str(root))
                service = PatchService(store)
                preview = service.preview(
                    PatchPreviewRequest(
                        project_id=str(project["id"]),
                        edits=[FileEdit(path="stable.py", new_content="VALUE = 2\n")],
                    )
                )
                rejected = service.rollback(preview.patch_id)
            after = target.stat()
            self.assertEqual(rejected.status, "rolled_back")
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(after.st_ino, before.st_ino)

    def test_scanner_execution_errors_are_fail_closed(self) -> None:
        self.assertEqual(
            AgentExecutionService._unavailable_scanners(
                {"semgrep": [{"status": "failed", "findings": [], "exit_code": 2}]}
            ),
            ["semgrep:failed"],
        )
        self.assertEqual(
            AgentExecutionService._unavailable_scanners(
                {
                    "semgrep": [
                        {
                            "status": "failed",
                            "findings": [{"rule_id": "measured"}],
                            "exit_code": 0,
                        }
                    ]
                }
            ),
            [],
        )

    def test_codeql_language_detection_and_official_suite_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["main.py", "server.ts", "native.cpp", "Program.cs", "Main.java", "main.go", "app.rb", "App.swift", "lib.rs"]:
                (root / name).write_text("", encoding="utf-8")
            scanner = SecurityScanner(root)
            self.assertEqual(
                set(scanner._detect_codeql_languages()),
                {"python", "javascript-typescript", "cpp", "csharp", "java-kotlin", "go", "ruby", "swift", "rust"},
            )
            self.assertEqual(
                scanner._codeql_suite("python", "security-and-quality"),
                "codeql/python-queries:codeql-suites/python-security-and-quality.qls",
            )

    def test_sandbox_policy_cannot_be_weakened_by_api_input(self) -> None:
        with self.assertRaises(ValidationError):
            HardenedSandboxPolicy(network_mode="host")
        with self.assertRaises(ValidationError):
            HardenedSandboxPolicy(cap_drop=[])
        with self.assertRaises(ValidationError):
            HardenedSandboxPolicy(read_only=False)
        with self.assertRaises(ValidationError):
            SandboxRunRequest(command=["python3", "-c", "print(1)"], workdir="/tmp")
        with self.assertRaises(ValidationError):
            SandboxRunRequest(command=["python3"], files={"../escape.py": "bad"})
        with self.assertRaises(ValidationError):
            SandboxRunRequest(command=["python3"], files={"/etc/passwd": "bad"})
        with self.assertRaises(ValidationError):
            SandboxRunRequest(command=["python3"], files={"C:\\outside.py": "bad"})

    def test_strict_scorecard_requires_real_coverage_and_oracles(self) -> None:
        runner = AgentScorecardRunner(backend=DeterministicAgentBackend(), policy_mode="strict")
        with self.assertRaises(ValidationError):
            RepositoryAgentTask(
                task_id="unsafe-oracle",
                repository=".",
                prompt="fix",
                category="coding",
                verification_commands=[["python", "-S", "verify.py"]],
                expected_changed_files=["C:\\outside.py"],
            )
        task = RepositoryAgentTask(
            task_id="only-one",
            repository=".",
            prompt="fix",
            category="coding",
            verification_commands=[["python", "-S", "verify.py"]],
            expected_changed_files=["calc.py"],
            required_substrings={"calc.py": ["return"]},
        )
        with self.assertRaisesRegex(ValueError, "50-100"):
            runner._validate_suite([task])

    def test_strict_scorecard_replays_live_serving_identity(self) -> None:
        class IdentityBackend(ModelBackend):
            name = "craftly_serving"

            def __init__(self, checkpoint_hash: str, tokenizer_hash: str) -> None:
                self.checkpoint_hash = checkpoint_hash
                self.tokenizer_hash = tokenizer_hash

            async def identity(self) -> dict[str, object]:
                return {
                    "status": "ready",
                    "model_name": "craftly-scratch",
                    "checkpoint_manifest_sha256": self.checkpoint_hash,
                    "tokenizer_sha256": self.tokenizer_hash,
                    "scratch_only": True,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.json"
            tokenizer = root / "tokenizer.json"
            profile_path = root / "active-profile.json"
            checkpoint.write_text("{}\n", encoding="utf-8")
            tokenizer.write_text("{}\n", encoding="utf-8")
            profile_path.write_text('{"profile":"validation"}\n', encoding="utf-8")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            profile = {
                "profile": {
                    "checkpoint_manifest": str(checkpoint),
                    "tokenizer_path": str(tokenizer),
                    "evidence": {
                        "checkpoint_manifest_sha256": checkpoint_hash,
                        "tokenizer_sha256": tokenizer_hash,
                        "evaluation_report_sha256": "a" * 64,
                    },
                }
            }
            backend = IdentityBackend(checkpoint_hash, tokenizer_hash)
            with (
                patch(
                    "src.craftly.evaluation.agent_scorecard.load_active_profile",
                    return_value=profile,
                ),
                patch(
                    "src.craftly.evaluation.agent_scorecard.active_profile_path",
                    return_value=profile_path,
                ),
            ):
                evidence = asyncio.run(
                    AgentScorecardRunner._model_evidence(
                        backend,
                        require_complete=True,
                    )
                )
                self.assertRegex(evidence["serving_identity_sha256"], r"^[0-9a-f]{64}$")
                backend.checkpoint_hash = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "serving checkpoint identity"):
                    asyncio.run(
                        AgentScorecardRunner._model_evidence(
                            backend,
                            require_complete=True,
                        )
                    )

    def test_gateway_one_command_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replacement = LocalStore(Path(temporary) / "state.sqlite3")
            root = Path(temporary) / "repo"
            root.mkdir()
            project = replacement.create_project(name="gateway", repo_path=str(root))
            report = AgentExecutionReport(
                project_id=str(project["id"]),
                prompt="fix",
                status="accepted",
                accepted=True,
                applied=True,
                final_run_id="run",
                final_task_graph_id="graph",
                final_answer="done",
                confidence=0.95,
                attempts=[],
                total_duration_ms=1.0,
            )

            class FakeService:
                def __init__(self, store: LocalStore) -> None:
                    self.store = store

                async def execute(self, request: AgentExecutionRequest) -> AgentExecutionReport:
                    self.request = request
                    return report

            original_store = gateway_api.STORE
            gateway_api.STORE = replacement
            try:
                with patch.object(gateway_api, "AgentExecutionService", FakeService):
                    response = TestClient(gateway_api.app).post(
                        "/v1/agent/execute",
                        json={
                            "project_id": str(project["id"]),
                            "prompt": "fix",
                            "policy_mode": "development",
                            "require_sandbox": False,
                            "allow_local_test_fallback": True,
                            "run_semgrep": False,
                            "run_codeql": False,
                        },
                    )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["accepted"])
            finally:
                gateway_api.STORE = original_store
                replacement.close()

    def test_development_scorecard_measures_real_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "fixture"
            repository.mkdir()
            make_repository(repository)
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "task_id": "debug-short-001",
                        "repository": "fixture",
                        "prompt": "fix add",
                        "category": "debugging",
                        "verification_commands": [["python", "-S", "verify.py"]],
                        "expected_changed_files": ["calc.py"],
                        "required_substrings": {"calc.py": ["return a + b"]},
                        "forbidden_substrings": {"calc.py": ["return a - b"]},
                        "short_prompt": True,
                        "run_semgrep": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("src.craftly.runtime.execution.VerifierRuntime", FixtureVerifierRuntime):
                report = asyncio.run(
                    AgentScorecardRunner(
                        backend=DeterministicAgentBackend(),
                        policy_mode="development",
                    ).run(
                        tasks_path=tasks,
                        output_dir=root / "reports",
                        repository_root=root,
                    )
                )
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.task_count, 1)
            self.assertEqual(report.workflow_completion_score, 1.0)
            self.assertEqual(report.short_prompt_understanding_score, 1.0)
            self.assertTrue((root / "reports" / "agent_scorecard.md").is_file())


if __name__ == "__main__":
    unittest.main()
