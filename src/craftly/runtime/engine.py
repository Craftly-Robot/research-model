"""Native Craftly agent runtime and dependency-aware worker pool."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.gateway.api import resolve_registered_repo_path
from src.craftly.indexing import ContextBuilder, RepositoryIndexer
from src.craftly.model_ops.backends import build_active_backend
from src.craftly.planning.engine import IntentPlanningEngine
from src.craftly.runtime.collaboration import (
    AgentMessage,
    AgentRole,
    BlackboardKind,
    BlackboardWrite,
    CollaborationRuntime,
    FailureIntelligence,
    MessageKind,
)
from src.craftly.runtime.taskgraph import (
    AgentRunCreateRequest,
    TaskCompleteRequest,
    TaskFailRequest,
    TaskGraphRuntime,
)
from src.craftly.shared.schemas import CraftlyRunReport, CraftlyRunRequest, StrictModel

AgentWorker = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

TASK_ROLE = {
    "understand": AgentRole.ARCHITECT,
    "planner": AgentRole.ARCHITECT,
    "retrieve_context": AgentRole.ARCHITECT,
    "edit": AgentRole.CODER,
    "test": AgentRole.TESTER,
    "critic_review": AgentRole.CRITIC,
    "security_review": AgentRole.SECURITY_REVIEWER,
    "performance_review": AgentRole.CRITIC,
    "verify": AgentRole.VERIFIER,
    "summarize": AgentRole.ORCHESTRATOR,
}
TASK_ARTIFACT_TYPE = {
    "understand": "plan",
    "planner": "task_graph",
    "retrieve_context": "architecture",
    "edit": "patch",
    "test": "test_result",
    "critic_review": "critic_review",
    "security_review": "security_review",
    "performance_review": "critic_review",
    "verify": "verification_decision",
    "summarize": "coordination",
}
TASK_RECIPIENT = {
    "understand": AgentRole.ORCHESTRATOR,
    "planner": AgentRole.CODER,
    "retrieve_context": AgentRole.CODER,
    "edit": AgentRole.TESTER,
    "test": AgentRole.CRITIC,
    "critic_review": AgentRole.VERIFIER,
    "security_review": AgentRole.VERIFIER,
    "performance_review": AgentRole.VERIFIER,
    "verify": AgentRole.ORCHESTRATOR,
    "summarize": AgentRole.BROADCAST,
}


class AgentWorkerPoolReport(StrictModel):
    task_graph_id: str
    status: str
    completed: int = 0
    failed: int = 0
    iterations: int = 0
    concurrency: int = Field(default=1, ge=1, le=64)
    missing_handlers: list[str] = Field(default_factory=list)
    timed_out: int = 0
    cancelled: int = 0
    peak_parallelism: int = 0
    message_count: int = 0


class WorkerRegistration(StrictModel):
    kind: str
    role: AgentRole
    timeout_seconds: float = Field(default=120.0, ge=0.1, le=3600.0)


class AgentWorkerPool:
    """Concurrent, dependency-aware TaskGraph workers with durable leases."""

    def __init__(
        self,
        runtime: TaskGraphRuntime,
        *,
        concurrency: int = 1,
        lease_seconds: float = 300.0,
        collaboration: CollaborationRuntime | None = None,
        failure_intelligence: FailureIntelligence | None = None,
    ) -> None:
        if not 1 <= concurrency <= 64:
            raise ValueError("concurrency must be between 1 and 64")
        if not 5.0 <= lease_seconds <= 3600.0:
            raise ValueError("lease_seconds must be between 5 and 3600")
        self.runtime = runtime
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.handlers: dict[str, AgentWorker] = {}
        self.registrations: dict[str, WorkerRegistration] = {}
        self.collaboration = collaboration or CollaborationRuntime(runtime.store)
        self.failure_intelligence = failure_intelligence or FailureIntelligence(runtime.store)
        self.worker_id = f"pool-{uuid.uuid4()}"

    def register(
        self,
        kind: str,
        handler: AgentWorker,
        *,
        role: AgentRole | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        expected_role = TASK_ROLE.get(kind)
        if expected_role is None:
            raise ValueError(f"unknown TaskGraph task kind: {kind}")
        selected_role = role or expected_role
        if selected_role != expected_role:
            raise ValueError(
                f"role mixing is forbidden: task kind {kind} requires {expected_role.value}, got {selected_role.value}"
            )
        self.handlers[kind] = handler
        self.registrations[kind] = WorkerRegistration(
            kind=kind,
            role=selected_role,
            timeout_seconds=timeout_seconds,
        )

    async def run_until_blocked_or_complete(
        self,
        task_graph_id: str,
        *,
        max_iterations: int = 100,
        cancellation_event: asyncio.Event | None = None,
    ) -> AgentWorkerPoolReport:
        if not 1 <= max_iterations <= 100_000:
            raise ValueError("max_iterations must be between 1 and 100000")
        graph = self.runtime.store.get_task_graph(task_graph_id)
        if graph is None:
            raise KeyError(f"unknown task graph: {task_graph_id}")
        run_id = str(graph["run_id"])
        initial_messages = len(self.collaboration.history(run_id))
        completed = failed = timed_out = cancelled = peak_parallelism = 0
        missing: list[str] = []

        for iteration in range(1, max_iterations + 1):
            if cancellation_event is not None and cancellation_event.is_set():
                state = self.runtime.cancel(task_graph_id)
                cancelled += sum(
                    1
                    for task in self.runtime.store.list_tasks(task_graph_id)
                    if task["status"] == "cancelled"
                )
                return self._report(
                    state.status,
                    task_graph_id,
                    run_id,
                    completed,
                    failed,
                    iteration,
                    missing,
                    timed_out,
                    cancelled,
                    peak_parallelism,
                    initial_messages,
                )
            state = self.runtime.report(task_graph_id)
            if state.status in {"completed", "failed", "cancelled"}:
                return self._report(
                    state.status,
                    task_graph_id,
                    run_id,
                    completed,
                    failed,
                    iteration,
                    missing,
                    timed_out,
                    cancelled,
                    peak_parallelism,
                    initial_messages,
                )
            claimed = self.runtime.claim_ready_tasks(
                task_graph_id,
                limit=self.concurrency,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if not claimed:
                return self._report(
                    "blocked",
                    task_graph_id,
                    run_id,
                    completed,
                    failed,
                    iteration,
                    missing,
                    timed_out,
                    cancelled,
                    peak_parallelism,
                    initial_messages,
                )
            peak_parallelism = max(peak_parallelism, len(claimed))
            outcomes = await asyncio.gather(
                *(
                    self._execute_claimed(task, cancellation_event=cancellation_event)
                    for task in claimed
                )
            )
            for outcome, task_kind in outcomes:
                if outcome == "completed":
                    completed += 1
                elif outcome == "timeout":
                    timed_out += 1
                    failed += 1
                elif outcome == "cancelled":
                    cancelled += 1
                else:
                    failed += 1
                    if outcome == "missing":
                        missing.append(task_kind)

        return self._report(
            "blocked",
            task_graph_id,
            run_id,
            completed,
            failed,
            max_iterations,
            missing,
            timed_out,
            cancelled,
            peak_parallelism,
            initial_messages,
        )

    async def _execute_claimed(
        self,
        task: dict[str, Any],
        *,
        cancellation_event: asyncio.Event | None,
    ) -> tuple[str, str]:
        kind = str(task.get("kind") or "")
        handler = self.handlers.get(kind)
        registration = self.registrations.get(kind)
        if handler is None or registration is None:
            self.runtime.fail_task(
                task["id"],
                TaskFailRequest(error=f"no worker registered for task kind: {kind}"),
                claim_next=False,
            )
            return "missing", kind
        if cancellation_event is not None and cancellation_event.is_set():
            self.runtime.cancel(str(task["task_graph_id"]))
            return "cancelled", kind
        handler_task: asyncio.Task[dict[str, Any]] | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        cancellation_task: asyncio.Task[bool] | None = None
        try:
            handler_task = asyncio.create_task(
                handler(task), name=f"craftly-agent-{kind}-{task['id']}"
            )
            heartbeat_task = asyncio.create_task(
                self._renew_lease(str(task["id"])),
                name=f"craftly-lease-{task['id']}",
            )
            if cancellation_event is not None:
                cancellation_task = asyncio.create_task(
                    cancellation_event.wait(),
                    name=f"craftly-cancel-{task['id']}",
                )
                done, _pending = await asyncio.wait(
                    {handler_task, cancellation_task},
                    timeout=registration.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancellation_task in done and cancellation_event.is_set():
                    handler_task.cancel()
                    await asyncio.gather(handler_task, return_exceptions=True)
                    self.runtime.cancel(str(task["task_graph_id"]))
                    return "cancelled", kind
                if handler_task not in done:
                    raise TimeoutError
                outputs = handler_task.result()
            else:
                outputs = await asyncio.wait_for(handler_task, timeout=registration.timeout_seconds)
            if not isinstance(outputs, dict):
                raise TypeError(f"worker {kind} must return a dictionary")
            message = self._record_worker_output(task, outputs, registration.role)
            self.runtime.complete_task(
                task["id"],
                TaskCompleteRequest(outputs={**outputs, "agent_message_id": message.message_id}),
                claim_next=False,
            )
            return "completed", kind
        except TimeoutError:
            if handler_task is not None and not handler_task.done():
                handler_task.cancel()
                await asyncio.gather(handler_task, return_exceptions=True)
            error = f"worker {kind} exceeded timeout of {registration.timeout_seconds:.3f}s"
            self._record_failure(task, error, failure_kind="timeout")
            self.runtime.fail_task(task["id"], TaskFailRequest(error=error), claim_next=False)
            return "timeout", kind
        except asyncio.CancelledError:
            if handler_task is not None:
                handler_task.cancel()
                await asyncio.gather(handler_task, return_exceptions=True)
            self.runtime.cancel(str(task["task_graph_id"]))
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._record_failure(task, error, failure_kind="worker_exception")
            self.runtime.fail_task(task["id"], TaskFailRequest(error=error), claim_next=False)
            return "failed", kind
        finally:
            for background in (heartbeat_task, cancellation_task):
                if background is not None:
                    background.cancel()
            await asyncio.gather(
                *(task for task in (heartbeat_task, cancellation_task) if task is not None),
                return_exceptions=True,
            )

    async def _renew_lease(self, task_id: str) -> None:
        interval = max(1.0, min(30.0, self.lease_seconds / 3.0))
        while True:
            await asyncio.sleep(interval)
            renewed = self.runtime.store.renew_task_lease(
                task_id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if not renewed:
                return

    def _record_worker_output(
        self,
        task: dict[str, Any],
        outputs: dict[str, Any],
        role: AgentRole,
    ) -> AgentMessage:
        task_kind = str(task["kind"])
        message_kind = (
            MessageKind.DECISION
            if task_kind == "verify"
            else MessageKind.REVIEW
            if task_kind in {"critic_review", "security_review", "performance_review"}
            else MessageKind.EVIDENCE
            if task_kind == "test"
            else MessageKind.PROPOSAL
        )
        payload = dict(outputs)
        payload.setdefault("artifact_type", TASK_ARTIFACT_TYPE[task_kind])
        evidence_refs = [str(item) for item in payload.pop("evidence_refs", []) if str(item)][:128]
        message = self.collaboration.publish(
            AgentMessage(
                run_id=str(task["run_id"]),
                task_graph_id=str(task["task_graph_id"]),
                task_id=str(task["id"]),
                correlation_id=str(task["id"]),
                sender_role=role,
                recipient_role=TASK_RECIPIENT[task_kind],
                kind=message_kind,
                payload=payload,
                evidence_refs=evidence_refs,
            )
        )
        board_kind = (
            BlackboardKind.EVIDENCE
            if message_kind == MessageKind.EVIDENCE
            else BlackboardKind.DECISION
            if message_kind == MessageKind.DECISION
            else BlackboardKind.ARTIFACT
            if message_kind == MessageKind.PROPOSAL
            else BlackboardKind.FACT
        )
        verified = (
            bool(payload.get("accepted") or payload.get("passed"))
            if board_kind == BlackboardKind.DECISION
            else False
        )
        self.collaboration.write_blackboard(
            BlackboardWrite(
                run_id=message.run_id,
                task_graph_id=message.task_graph_id,
                entry_key=f"task/{task['id']}/{board_kind.value}",
                kind=board_kind,
                value={
                    "message_id": message.message_id,
                    "task_kind": task_kind,
                    "payload": message.payload,
                },
                expected_version=0,
                verified=verified,
                source_message_id=message.message_id,
            )
        )
        return message

    def _record_failure(self, task: dict[str, Any], error: str, *, failure_kind: str) -> None:
        run = self.runtime.store.get_run(str(task["run_id"])) or {}
        self.failure_intelligence.observe(
            error,
            project_id=run.get("project_id"),
            run_id=str(task["run_id"]),
            task_id=str(task["id"]),
            metadata={"task_kind": task.get("kind"), "failure_kind": failure_kind},
        )

    def _report(
        self,
        status: str,
        task_graph_id: str,
        run_id: str,
        completed: int,
        failed: int,
        iterations: int,
        missing: list[str],
        timed_out: int,
        cancelled: int,
        peak_parallelism: int,
        initial_messages: int,
    ) -> AgentWorkerPoolReport:
        return AgentWorkerPoolReport(
            task_graph_id=task_graph_id,
            status=status,
            completed=completed,
            failed=failed,
            iterations=iterations,
            concurrency=self.concurrency,
            missing_handlers=sorted(set(missing)),
            timed_out=timed_out,
            cancelled=cancelled,
            peak_parallelism=peak_parallelism,
            message_count=len(self.collaboration.history(run_id)) - initial_messages,
        )


class AgentRouter:
    """Deterministic role router used before a learned scratch router exists."""

    def route(self, prompt: str, *, top_k: int = 4) -> dict[str, object]:
        lowered = prompt.lower()
        candidates: list[dict[str, object]] = []
        if any(
            term in lowered
            for term in ["security", "vulnerability", "cve", "secret", "xss", "sql injection"]
        ):
            candidates.append({"role": "security", "score": 0.95})
        if any(term in lowered for term in ["test", "pytest", "fail", "bug", "regression"]):
            candidates.append({"role": "testing", "score": 0.9})
        if any(term in lowered for term in ["code", "build", "implement", "fix", "refactor"]):
            candidates.append({"role": "coding", "score": 0.88})
        if any(term in lowered for term in ["design", "architecture", "plan", "system"]):
            candidates.append({"role": "architect", "score": 0.84})
        candidates.append({"role": "planner", "score": 0.75})
        return {"route": candidates[:top_k], "top_role": candidates[0]["role"], "router": "native"}


class CraftlyRuntime:
    def __init__(self) -> None:
        self.planner = IntentPlanningEngine()
        self.router = AgentRouter()

    async def run(self, request: CraftlyRunRequest) -> CraftlyRunReport:
        started = time.perf_counter()
        backend = build_active_backend()
        try:
            plan = await self.planner.plan_structured(
                request.prompt,
                backend=backend,
                allow_dev_fallback=backend.name == "mock" or request.policy_mode == "development",
            )
            route = self.router.route(request.prompt, top_k=request.max_agent_nodes or 4)
            store = LocalStore()
            workspace = str(resolve_registered_repo_path(request.workspace))
            project = store.create_project(name=f"runtime-{time.time_ns()}", repo_path=workspace)
            index_report = RepositoryIndexer(store).index_project(project_id=project["id"])
            context = ContextBuilder(store).build(
                project_id=project["id"], query=request.prompt, token_budget=8000
            )
            agent_run = TaskGraphRuntime(store).create_agent_run(
                AgentRunCreateRequest(
                    project_id=project["id"],
                    prompt=request.prompt,
                    mode=plan.expansion.get("intent", "code_edit"),
                    max_steps=request.max_agent_nodes or 6,
                )
            )
            graph = store.get_task_graph(agent_run.task_graph_id)
            answer = await backend.generate(
                f"{context.prompt_context}\n\nUser request:\n{request.prompt}\n\n"
                "Return a concise implementation plan and patch guidance.",
                temperature=0.2,
                max_tokens=1024,
            )
        finally:
            await backend.aclose()
        return CraftlyRunReport(
            run_id=agent_run.run_id,
            status="complete",
            summary="Craftly completed planning, indexing, context packing, and model response.",
            confidence=plan.confidence,
            prompt=request.prompt,
            workspace=workspace,
            final_answer=answer,
            route={"intent": plan.expansion.get("intent"), "runtime": "native-taskgraph", **route},
            plan=plan.model_dump(),
            memory={
                "context_id": context.context_id,
                "chunks": [chunk.model_dump(exclude={"content"}) for chunk in context.chunks],
            },
            verification=None,
            security=None,
            artifacts={
                "project": project,
                "index": index_report.model_dump(),
                "task_graph": graph,
            },
            duration_ms=(time.perf_counter() - started) * 1000,
        )
