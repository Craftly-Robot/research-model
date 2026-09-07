from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.craftly.db import LocalStore
from src.craftly.gateway import api as gateway_api
from src.craftly.memory import UnifiedMemoryManager
from src.craftly.runtime.collaboration import (
    AgentMessage,
    AgentRole,
    BlackboardKind,
    BlackboardWrite,
    CollaborationRuntime,
    CriticScore,
    FailureIntelligence,
    MessageKind,
    PeerReviewResult,
    VerifierDecision,
)
from src.craftly.runtime.engine import AgentWorkerPool
from src.craftly.runtime.taskgraph import AgentRunCreateRequest, TaskGraphRuntime


class CraftlyAgentCollaborationTest(unittest.TestCase):
    def make_runtime(self, root: str, store: LocalStore) -> tuple[dict[str, object], object, TaskGraphRuntime]:
        project = store.create_project(name="collaboration", repo_path=root)
        runtime = TaskGraphRuntime(store)
        run = runtime.create_agent_run(
            AgentRunCreateRequest(project_id=str(project["id"]), prompt="securely repair and verify the service")
        )
        return project, run, runtime

    def test_role_separation_rejects_architect_code_and_critic_decision(self) -> None:
        common = {
            "run_id": "run",
            "task_graph_id": "graph",
            "recipient_role": AgentRole.CODER,
        }
        with self.assertRaises(ValidationError):
            AgentMessage(
                **common,
                sender_role=AgentRole.ARCHITECT,
                kind=MessageKind.PROPOSAL,
                payload={"artifact_type": "patch", "content": "code"},
            )
        with self.assertRaises(ValidationError):
            AgentMessage(
                **common,
                sender_role=AgentRole.CRITIC,
                kind=MessageKind.DECISION,
                payload={"artifact_type": "critic_review", "accepted": True},
            )

    def test_durable_messages_blackboard_cas_and_immutable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with LocalStore(Path(root) / "state.sqlite3") as store:
                _project, run, _runtime = self.make_runtime(root, store)
                collaboration = CollaborationRuntime(store)
                message = collaboration.publish(
                    AgentMessage(
                        run_id=run.run_id,
                        task_graph_id=run.task_graph_id,
                        sender_role=AgentRole.TESTER,
                        recipient_role=AgentRole.CRITIC,
                        kind=MessageKind.EVIDENCE,
                        payload={"artifact_type": "test_result", "passed": True, "api_key": "must-not-persist"},
                    )
                )
                self.assertEqual(message.payload["api_key"], "[REDACTED]")
                self.assertEqual(len(collaboration.history(run.run_id)), 1)
                entry = collaboration.write_blackboard(
                    BlackboardWrite(
                        run_id=run.run_id,
                        task_graph_id=run.task_graph_id,
                        entry_key="evidence/test-1",
                        kind=BlackboardKind.EVIDENCE,
                        value={"exit_code": 0},
                        source_message_id=message.message_id,
                    )
                )
                self.assertTrue(entry.immutable)
                self.assertEqual(entry.version, 1)
                with self.assertRaises(RuntimeError):
                    collaboration.write_blackboard(
                        BlackboardWrite(
                            run_id=run.run_id,
                            task_graph_id=run.task_graph_id,
                            entry_key="evidence/test-1",
                            kind=BlackboardKind.EVIDENCE,
                            value={"exit_code": 1},
                            expected_version=1,
                            source_message_id=message.message_id,
                        )
                    )
                mutable = collaboration.write_blackboard(
                    BlackboardWrite(
                        run_id=run.run_id,
                        task_graph_id=run.task_graph_id,
                        entry_key="question/open",
                        kind=BlackboardKind.QUESTION,
                        value={"text": "is the migration verified?"},
                        expected_version=0,
                    )
                )
                with self.assertRaises(RuntimeError):
                    collaboration.write_blackboard(
                        BlackboardWrite(
                            run_id=run.run_id,
                            task_graph_id=run.task_graph_id,
                            entry_key="question/open",
                            kind=BlackboardKind.FACT,
                            value={"answer": "yes"},
                            expected_version=mutable.version + 1,
                        )
                    )
                updated = collaboration.write_blackboard(
                    BlackboardWrite(
                        run_id=run.run_id,
                        task_graph_id=run.task_graph_id,
                        entry_key="question/open",
                        kind=BlackboardKind.FACT,
                        value={"answer": "yes"},
                        expected_version=mutable.version,
                    )
                )
                self.assertEqual(updated.version, 2)

    def test_negotiation_reflects_at_most_three_times_and_promotes_only_verified(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as root:
                with LocalStore(Path(root) / "negotiation.sqlite3") as store:
                    project, run, _runtime = self.make_runtime(root, store)
                    collaboration = CollaborationRuntime(store, confidence_threshold=0.85, max_revisions=3)
                    memory = UnifiedMemoryManager(project_id=str(project["id"]), store=store)
                    critic_calls = 0

                    async def peer(_proposal: AgentMessage) -> PeerReviewResult:
                        return PeerReviewResult(accepted=True, confidence=0.9)

                    async def evidence(_proposal: AgentMessage, _review: PeerReviewResult) -> dict[str, object]:
                        return {"artifact_type": "patch", "test_report": "verification-1"}

                    async def critic(_proposal: AgentMessage, _review: PeerReviewResult) -> CriticScore:
                        nonlocal critic_calls
                        critic_calls += 1
                        if critic_calls == 1:
                            return CriticScore(
                                confidence=0.5,
                                assumptions_wrong=["input was assumed trusted"],
                                failure_modes=["invalid input"],
                                security_risks=["injection"],
                                unverified_evidence=["integration test"],
                            )
                        return CriticScore(confidence=0.92)

                    async def verifier(_proposal: AgentMessage, score: CriticScore) -> VerifierDecision:
                        return VerifierDecision(
                            accepted=score.confidence >= 0.85,
                            criteria_passed=["tests", "security"] if score.confidence >= 0.85 else [],
                            criteria_failed=[] if score.confidence >= 0.85 else ["confidence"],
                            verification_refs=["verification-1"] if score.confidence >= 0.85 else [],
                        )

                    async def revise(
                        proposal: AgentMessage,
                        _score: CriticScore,
                        questions: list[str],
                    ) -> dict[str, object]:
                        self.assertEqual(len(questions), 4)
                        return {**proposal.payload, "revision": 1, "assumptions_checked": True}

                    report = await collaboration.negotiate(
                        AgentMessage(
                            run_id=run.run_id,
                            task_graph_id=run.task_graph_id,
                            sender_role=AgentRole.CODER,
                            recipient_role=AgentRole.SECURITY_REVIEWER,
                            kind=MessageKind.PROPOSAL,
                            payload={"artifact_type": "patch", "summary": "validate user input"},
                        ),
                        peer_role=AgentRole.SECURITY_REVIEWER,
                        peer_reviewer=peer,
                        evidence_responder=evidence,
                        critic=critic,
                        verifier=verifier,
                        reviser=revise,
                        memory=memory,
                    )
                    self.assertTrue(report.accepted)
                    self.assertEqual(report.revision_count, 1)
                    self.assertTrue(report.memory_promoted)
                    promoted = store.list_memory_entries(str(project["id"]))
                    self.assertEqual(len(promoted), 1)
                    self.assertTrue(promoted[0]["metadata"]["verified"])

        asyncio.run(scenario())

    def test_worker_pool_executes_dependency_ready_reviews_concurrently(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as root:
                with LocalStore(Path(root) / "workers.sqlite3") as store:
                    _project, run, runtime = self.make_runtime(root, store)
                    pool = AgentWorkerPool(runtime, concurrency=4)
                    active = 0
                    peak = 0
                    lock = asyncio.Lock()

                    def handler_for(kind: str):
                        async def handler(_task: dict[str, object]) -> dict[str, object]:
                            nonlocal active, peak
                            async with lock:
                                active += 1
                                peak = max(peak, active)
                            if kind in {"test", "security_review", "performance_review"}:
                                await asyncio.sleep(0.05)
                            async with lock:
                                active -= 1
                            if kind == "verify":
                                return {"accepted": True, "evidence_refs": ["verification-1"]}
                            if kind in {"critic_review", "performance_review"}:
                                return {"confidence": 0.9, "issues": []}
                            return {"passed": True, "kind": kind}

                        return handler

                    for kind in [
                        "understand",
                        "planner",
                        "retrieve_context",
                        "edit",
                        "test",
                        "critic_review",
                        "security_review",
                        "performance_review",
                        "verify",
                        "summarize",
                    ]:
                        pool.register(kind, handler_for(kind))
                    report = await pool.run_until_blocked_or_complete(run.task_graph_id)
                    self.assertEqual(report.status, "completed")
                    self.assertEqual(report.completed, 10)
                    self.assertGreaterEqual(report.peak_parallelism, 3)
                    self.assertGreaterEqual(peak, 3)
                    self.assertEqual(report.message_count, 10)
                    kinds = {entry.kind for entry in pool.collaboration.blackboard(run.run_id)}
                    self.assertIn(BlackboardKind.EVIDENCE, kinds)
                    self.assertIn(BlackboardKind.DECISION, kinds)

        asyncio.run(scenario())

    def test_timeout_retry_cancellation_and_failure_candidate_gate(self) -> None:
        async def timeout_scenario() -> None:
            with tempfile.TemporaryDirectory() as root:
                with LocalStore(Path(root) / "timeout.sqlite3") as store:
                    project, run, runtime = self.make_runtime(root, store)
                    pool = AgentWorkerPool(runtime, concurrency=1)

                    async def slow(_task: dict[str, object]) -> dict[str, object]:
                        await asyncio.sleep(0.2)
                        return {"passed": True}

                    pool.register("understand", slow, timeout_seconds=0.1)
                    report = await pool.run_until_blocked_or_complete(run.task_graph_id)
                    self.assertEqual(report.status, "failed")
                    self.assertEqual(report.timed_out, 2)
                    clusters = pool.failure_intelligence.clusters(str(project["id"]))
                    self.assertEqual(clusters[0]["occurrence_count"], 2)

                    _project2, run2, runtime2 = self.make_runtime(root, store)
                    event = asyncio.Event()
                    event.set()
                    cancelled = await AgentWorkerPool(runtime2).run_until_blocked_or_complete(
                        run2.task_graph_id,
                        cancellation_event=event,
                    )
                    self.assertEqual(cancelled.status, "cancelled")

        asyncio.run(timeout_scenario())

        with tempfile.TemporaryDirectory() as root:
            with LocalStore(Path(root) / "failures.sqlite3") as store:
                project, run, _runtime = self.make_runtime(root, store)
                intelligence = FailureIntelligence(store, candidate_threshold=2)
                first = intelligence.observe(
                    "ValueError at C:\\repo\\api.py line=174 address 0x7ffd001122",
                    project_id=str(project["id"]),
                    run_id=run.run_id,
                    task_id=None,
                )
                second = intelligence.observe(
                    "ValueError at D:\\build\\api.py line=991 address 0x7fff998877",
                    project_id=str(project["id"]),
                    run_id=run.run_id,
                    task_id=None,
                )
                self.assertEqual(first["id"], second["id"])
                patch = store.create_patch_record(
                    project_id=str(project["id"]),
                    run_id=run.run_id,
                    status="verified",
                    diff="--- a/api.py\n+++ b/api.py\n",
                    files_changed=["api.py"],
                    backup={},
                )
                resolved = intelligence.resolve(
                    str(second["id"]),
                    root_cause="unvalidated input reached the parser",
                    patch_id=str(patch["id"]),
                    verification_ref="verification-report-1",
                    verification_passed=True,
                )
                self.assertEqual(resolved["status"], "verified")
                self.assertIsNotNone(resolved["dataset_candidate_id"])
                self.assertEqual(resolved["dataset_candidate"]["status"], "pending_review")

    def test_gateway_exposes_durable_messages_blackboard_and_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            original_store = gateway_api.STORE
            replacement = LocalStore(Path(root) / "gateway-collaboration.sqlite3")
            gateway_api.STORE = replacement
            try:
                project, run, _runtime = self.make_runtime(root, replacement)
                client = TestClient(gateway_api.app)
                response = client.post(
                    "/v1/agent/messages",
                    json={
                        "run_id": run.run_id,
                        "task_graph_id": run.task_graph_id,
                        "sender_role": "tester",
                        "recipient_role": "critic",
                        "kind": "evidence",
                        "payload": {"artifact_type": "test_result", "passed": True},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                message_id = response.json()["message_id"]
                board = client.put(
                    "/v1/agent/blackboard",
                    json={
                        "run_id": run.run_id,
                        "task_graph_id": run.task_graph_id,
                        "entry_key": "evidence/api-test",
                        "kind": "evidence",
                        "value": {"exit_code": 0},
                        "source_message_id": message_id,
                    },
                )
                self.assertEqual(board.status_code, 200, board.text)
                history = client.get(f"/v1/agent/runs/{run.run_id}/messages")
                self.assertEqual(history.status_code, 200, history.text)
                self.assertEqual(len(history.json()), 1)
                entries = client.get(f"/v1/agent/runs/{run.run_id}/blackboard?kind=evidence")
                self.assertEqual(entries.status_code, 200, entries.text)
                self.assertTrue(entries.json()[0]["immutable"])
                clusters = client.get(f"/v1/projects/{project['id']}/failure-clusters")
                self.assertEqual(clusters.status_code, 200, clusters.text)
                cancelled = client.post(f"/v1/taskgraphs/{run.task_graph_id}/cancel")
                self.assertEqual(cancelled.status_code, 200, cancelled.text)
                self.assertEqual(cancelled.json()["status"], "cancelled")
            finally:
                replacement.close()
                gateway_api.STORE = original_store

    def test_expired_worker_lease_consumes_retry_budget_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with LocalStore(Path(root) / "lease.sqlite3") as store:
                _project, run, runtime = self.make_runtime(root, store)
                first = runtime.claim_ready_tasks(
                    run.task_graph_id,
                    limit=1,
                    worker_id="worker-a",
                    lease_seconds=5,
                )[0]
                store.connection.execute(
                    "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
                    (time.time() - 1, first["id"]),
                )
                store.connection.commit()
                retrying = runtime.report(run.task_graph_id)
                retried_task = store.get_task(str(first["id"]))
                self.assertEqual(retrying.status, "running")
                self.assertEqual(retried_task["attempt"], 1)
                self.assertEqual(retried_task["status"], "queued")
                runtime.claim_ready_tasks(
                    run.task_graph_id,
                    limit=1,
                    worker_id="worker-b",
                    lease_seconds=5,
                )
                store.connection.execute(
                    "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
                    (time.time() - 1, first["id"]),
                )
                store.connection.commit()
                final = runtime.report(run.task_graph_id)
                self.assertEqual(final.status, "failed")
                self.assertEqual(store.get_task(str(first["id"]))["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
