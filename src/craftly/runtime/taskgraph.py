"""Durable MVP TaskGraph runtime."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.shared.schemas import StrictModel

TASK_NODE_ORDER = [
    ("understand", "Understand request and repository target"),
    ("planner", "Build durable implementation plan"),
    ("retrieve_context", "Retrieve ranked repository context"),
    ("edit", "Generate patch proposal"),
    ("test", "Run targeted tests"),
    ("security_review", "Review defensive security constraints"),
    ("performance_review", "Review performance and maintainability"),
    ("critic_review", "Review correctness and reasoning risks"),
    ("verify", "Verify patch and security constraints"),
    ("summarize", "Return final answer and evidence"),
]


class TaskNode(StrictModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: str
    title: str
    instructions: str
    status: str = "queued"
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 2
    started_at: float | None = None
    finished_at: float | None = None


class TaskGraph(StrictModel):
    task_graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    run_id: str
    goal: str
    status: str = "queued"
    nodes: list[TaskNode]
    edges: list[dict[str, str]]
    success_criteria: list[str]
    created_at_unix: float = Field(default_factory=time.time)
    updated_at_unix: float = Field(default_factory=time.time)


class AgentRunCreateRequest(StrictModel):
    project_id: str
    session_id: str | None = None
    prompt: str = Field(min_length=1)
    mode: str = Field(
        default="code_edit", pattern="^(code_edit|debug|explain|security_review)$"
    )
    max_steps: int = Field(default=12, ge=1, le=50)
    apply_patch: bool = False
    model_profile: str = "mock"


class AgentRunCreateResponse(StrictModel):
    run_id: str
    project_id: str
    session_id: str | None
    status: str
    task_graph_id: str


class TaskAdvanceResponse(StrictModel):
    task_graph_id: str
    status: str
    active_task: dict[str, Any] | None = None
    active_tasks: list[dict[str, Any]] = Field(default_factory=list)
    ready_task_count: int
    completed_task_count: int
    failed_task_count: int


class TaskCompleteRequest(StrictModel):
    outputs: dict[str, Any] = Field(default_factory=dict)


class TaskFailRequest(StrictModel):
    error: str = Field(min_length=1)
    outputs: dict[str, Any] = Field(default_factory=dict)


class TaskGraphRuntime:
    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store or LocalStore()

    def create_agent_run(
        self, request: AgentRunCreateRequest
    ) -> AgentRunCreateResponse:
        run = self.store.create_run(
            project_id=request.project_id,
            session_id=request.session_id,
            prompt=request.prompt,
            mode=request.mode,
            model_profile=request.model_profile,
            status="queued",
        )
        graph = self.build_graph(
            project_id=request.project_id,
            run_id=run["id"],
            prompt=request.prompt,
            mode=request.mode,
            max_steps=request.max_steps,
            apply_patch=request.apply_patch,
        )
        self.store.create_task_graph(
            project_id=request.project_id,
            run_id=run["id"],
            goal=graph.goal,
            status=graph.status,
            graph=graph.model_dump(),
        )
        return AgentRunCreateResponse(
            run_id=run["id"],
            project_id=request.project_id,
            session_id=request.session_id,
            status=run["status"],
            task_graph_id=graph.task_graph_id,
        )

    def build_graph(
        self,
        *,
        project_id: str,
        run_id: str,
        prompt: str,
        mode: str,
        max_steps: int,
        apply_patch: bool,
    ) -> TaskGraph:
        nodes: list[TaskNode] = []
        node_ids: dict[str, str] = {}
        dependency_kinds: dict[str, list[str]] = {
            "understand": [],
            "planner": ["understand"],
            "retrieve_context": ["planner"],
            "edit": ["retrieve_context"],
            "test": ["edit"],
            "critic_review": ["test", "security_review", "performance_review"],
            "security_review": ["edit"],
            "performance_review": ["edit"],
            "verify": [
                "test",
                "critic_review",
                "security_review",
                "performance_review",
            ],
            "summarize": ["verify"],
        }
        for kind, title in TASK_NODE_ORDER:
            dependencies = [node_ids[item] for item in dependency_kinds[kind]]
            node = TaskNode(
                kind=kind,
                title=title,
                instructions=self.instructions_for(
                    kind, prompt=prompt, mode=mode, apply_patch=apply_patch
                ),
                depends_on=dependencies,
                inputs={"prompt": prompt, "mode": mode, "max_steps": max_steps}
                if kind == "understand"
                else {},
            )
            nodes.append(node)
            node_ids[kind] = node.node_id
        edges = [
            {"from": dependency_id, "to": node.node_id, "condition": "success"}
            for node in nodes
            for dependency_id in node.depends_on
        ]
        return TaskGraph(
            project_id=project_id,
            run_id=run_id,
            goal=f"Complete Craftly agent request: {prompt}",
            nodes=nodes,
            edges=edges,
            success_criteria=[
                "context is relevant to the prompt",
                "patch is previewed before apply",
                "tests or verification commands complete",
                "failed patches are rejected or rolled back",
            ],
        )

    def instructions_for(
        self, kind: str, *, prompt: str, mode: str, apply_patch: bool
    ) -> str:
        instructions = {
            "understand": "Classify intent, target files, risks, and expected verification evidence.",
            "planner": "Create a durable dependency-aware task plan. Do not write executable code in this stage.",
            "retrieve_context": "Build a ranked context pack from indexed repository chunks.",
            "edit": "Generate minimal file edits as structured patch operations.",
            "test": "Run targeted tests and capture stdout, stderr, exit code, and duration.",
            "critic_review": "Review executor outputs for logic gaps, missing requirements, and incorrect assumptions. Do not produce the final solution.",
            "security_review": "Review patch and evidence for defensive security risks, secret exposure, unsafe execution, and vulnerability regressions.",
            "performance_review": "Review patch for unnecessary complexity, slow paths, resource usage, and maintainability risk.",
            "verify": "Accept only if patch applies cleanly and verification passes.",
            "summarize": "Return concise final answer with changed files and verification evidence.",
        }
        base = instructions[kind]
        return f"{base} mode={mode}; apply_patch={apply_patch}; prompt={prompt}"

    def advance(self, task_graph_id: str) -> TaskAdvanceResponse:
        graph = self.store.get_task_graph(task_graph_id)
        if graph is None:
            raise KeyError(f"unknown task graph: {task_graph_id}")
        tasks = self.store.list_tasks(task_graph_id)
        failed = [task for task in tasks if task["status"] == "failed"]
        if failed:
            self.store.update_task_graph_status(task_graph_id, "failed")
            return self.advance_report(task_graph_id, "failed", None, tasks)
        self.store.recover_expired_task_leases(task_graph_id)
        tasks = self.store.list_tasks(task_graph_id)
        failed = [task for task in tasks if task["status"] == "failed"]
        if failed:
            self.store.update_task_graph_status(task_graph_id, "failed")
            return self.advance_report(task_graph_id, "failed", None, tasks)
        cancelled = [task for task in tasks if task["status"] == "cancelled"]
        if cancelled:
            self.store.update_task_graph_status(task_graph_id, "cancelled")
            return self.advance_report(task_graph_id, "cancelled", None, tasks)
        running = [task for task in tasks if task["status"] == "running"]
        if running:
            return self.advance_report(task_graph_id, "running", running[0], tasks)
        claimed = self.claim_ready_tasks(
            task_graph_id,
            limit=1,
            worker_id=f"api-{uuid.uuid4()}",
            lease_seconds=300.0,
        )
        if claimed:
            started = claimed[0]
            return self.advance_report(
                task_graph_id, "running", started, self.store.list_tasks(task_graph_id)
            )
        if tasks and all(task["status"] == "completed" for task in tasks):
            self.store.update_task_graph_status(task_graph_id, "completed")
            return self.advance_report(task_graph_id, "completed", None, tasks)
        return self.advance_report(
            task_graph_id, str(graph.get("status") or "queued"), None, tasks
        )

    def complete_task(
        self,
        task_id: str,
        request: TaskCompleteRequest | None = None,
        *,
        claim_next: bool = True,
    ) -> TaskAdvanceResponse:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        self.store.update_task_state(
            task_id,
            status="completed",
            outputs=(request.outputs if request else {}),
            error=None,
            finished=True,
        )
        if claim_next:
            return self.advance(task["task_graph_id"])
        return self.report(task["task_graph_id"])

    def fail_task(
        self, task_id: str, request: TaskFailRequest, *, claim_next: bool = True
    ) -> TaskAdvanceResponse:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        next_attempt = int(task.get("attempt", 0)) + 1
        max_attempts = int(task.get("max_attempts", 2))
        retry_outputs = {
            **request.outputs,
            "last_error": request.error,
            "attempt": next_attempt,
            "max_attempts": max_attempts,
            "retrying": next_attempt < max_attempts,
        }
        if next_attempt < max_attempts:
            self.store.update_task_state(
                task_id,
                status="queued",
                outputs=retry_outputs,
                error=request.error,
                finished=False,
            )
            self.store.update_task_attempt(
                task_id,
                attempt=next_attempt,
                outputs=retry_outputs,
                error=request.error,
            )
            self.store.update_task_graph_status(task["task_graph_id"], "running")
            if claim_next:
                return self.advance(task["task_graph_id"])
            return self.report(task["task_graph_id"])
        self.store.update_task_state(
            task_id,
            status="failed",
            outputs=retry_outputs,
            error=request.error,
            finished=True,
        )
        self.store.update_task_attempt(
            task_id, attempt=next_attempt, outputs=retry_outputs, error=request.error
        )
        self.store.update_task_graph_status(task["task_graph_id"], "failed")
        return self.report(task["task_graph_id"])

    def claim_ready_tasks(
        self,
        task_graph_id: str,
        *,
        limit: int,
        worker_id: str,
        lease_seconds: float,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 64:
            raise ValueError("claim limit must be between 1 and 64")
        graph = self.store.get_task_graph(task_graph_id)
        if graph is None:
            raise KeyError(f"unknown task graph: {task_graph_id}")
        if str(graph.get("status")) in {"completed", "failed", "cancelled"}:
            return []
        self.store.recover_expired_task_leases(task_graph_id)
        tasks = self.store.list_tasks(task_graph_id)
        if any(task["status"] == "failed" for task in tasks):
            self.store.update_task_graph_status(task_graph_id, "failed")
            return []
        ready = self.ready_tasks(tasks)[:limit]
        return self.store.claim_tasks(
            task_graph_id,
            [str(task["id"]) for task in ready],
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    def cancel(self, task_graph_id: str) -> TaskAdvanceResponse:
        if self.store.get_task_graph(task_graph_id) is None:
            raise KeyError(f"unknown task graph: {task_graph_id}")
        self.store.cancel_task_graph(task_graph_id)
        return self.report(task_graph_id)

    def report(self, task_graph_id: str) -> TaskAdvanceResponse:
        graph = self.store.get_task_graph(task_graph_id)
        if graph is None:
            raise KeyError(f"unknown task graph: {task_graph_id}")
        self.store.recover_expired_task_leases(task_graph_id)
        tasks = self.store.list_tasks(task_graph_id)
        failed = any(task["status"] == "failed" for task in tasks)
        cancelled = any(task["status"] == "cancelled" for task in tasks)
        complete = bool(tasks) and all(task["status"] == "completed" for task in tasks)
        running = [task for task in tasks if task["status"] == "running"]
        status = (
            "failed"
            if failed
            else "cancelled"
            if cancelled
            else "completed"
            if complete
            else "running"
            if running or self.ready_tasks(tasks)
            else str(graph.get("status") or "queued")
        )
        if status != graph.get("status"):
            self.store.update_task_graph_status(task_graph_id, status)
        return self.advance_report(
            task_graph_id, status, running[0] if running else None, tasks
        )

    def ready_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed = {task["id"] for task in tasks if task["status"] == "completed"}
        return [
            task
            for task in tasks
            if task["status"] == "queued"
            and all(dependency in completed for dependency in task["depends_on"])
        ]

    def advance_report(
        self,
        task_graph_id: str,
        status: str,
        active_task: dict[str, Any] | None,
        tasks: list[dict[str, Any]],
    ) -> TaskAdvanceResponse:
        ready = self.ready_tasks(tasks)
        return TaskAdvanceResponse(
            task_graph_id=task_graph_id,
            status=status,
            active_task=active_task,
            active_tasks=[task for task in tasks if task["status"] == "running"],
            ready_task_count=len(ready),
            completed_task_count=sum(
                1 for task in tasks if task["status"] == "completed"
            ),
            failed_task_count=sum(1 for task in tasks if task["status"] == "failed"),
        )
