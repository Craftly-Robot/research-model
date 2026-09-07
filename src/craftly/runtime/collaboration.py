"""Durable collaboration, negotiation, reflection, and failure intelligence.

This module is the single ownership boundary for agent-to-agent coordination.
It deliberately does not execute tools, mutate repository files, or provide
long-term memory retrieval. Workers retain those responsibilities; this layer
validates communication, records evidence, and decides whether a verified
outcome is eligible for promotion.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.craftly.db import LocalStore
from src.craftly.memory import MemoryIngestRequest, UnifiedMemoryManager
from src.craftly.shared.schemas import StrictModel


MAX_MESSAGE_BYTES = 64 * 1024
MAX_EVIDENCE_REFS = 128
REFLECTION_QUESTIONS = (
    "What assumptions are wrong?",
    "What can fail?",
    "What security risks exist?",
    "What was not verified?",
)
SECRET_KEY_PATTERN = re.compile(r"(?i)(authorization|api[_-]?key|access[_-]?token|password|secret)")
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9._~+/=-]{16,}|"
    r"(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


class AgentRole(StrEnum):
    ARCHITECT = "architect"
    CODER = "coder"
    TESTER = "tester"
    SECURITY_REVIEWER = "security_reviewer"
    CRITIC = "critic"
    VERIFIER = "verifier"
    ORCHESTRATOR = "orchestrator"
    BROADCAST = "broadcast"


class MessageKind(StrEnum):
    PROPOSAL = "proposal"
    EVIDENCE = "evidence"
    CHALLENGE = "challenge"
    REVIEW = "review"
    DECISION = "decision"


class BlackboardKind(StrEnum):
    FACT = "fact"
    ARTIFACT = "artifact"
    DECISION = "decision"
    QUESTION = "question"
    EVIDENCE = "evidence"


ROLE_MESSAGE_POLICY: dict[AgentRole, frozenset[MessageKind]] = {
    AgentRole.ARCHITECT: frozenset({MessageKind.PROPOSAL, MessageKind.EVIDENCE}),
    AgentRole.CODER: frozenset({MessageKind.PROPOSAL, MessageKind.EVIDENCE}),
    AgentRole.TESTER: frozenset({MessageKind.EVIDENCE, MessageKind.CHALLENGE, MessageKind.REVIEW}),
    AgentRole.SECURITY_REVIEWER: frozenset({MessageKind.EVIDENCE, MessageKind.CHALLENGE, MessageKind.REVIEW}),
    AgentRole.CRITIC: frozenset({MessageKind.CHALLENGE, MessageKind.REVIEW}),
    AgentRole.VERIFIER: frozenset({MessageKind.DECISION}),
    AgentRole.ORCHESTRATOR: frozenset(
        {MessageKind.PROPOSAL, MessageKind.EVIDENCE, MessageKind.CHALLENGE, MessageKind.REVIEW, MessageKind.DECISION}
    ),
}

ROLE_ARTIFACT_POLICY: dict[AgentRole, frozenset[str]] = {
    AgentRole.ARCHITECT: frozenset({"plan", "architecture", "task_graph"}),
    AgentRole.CODER: frozenset({"code", "patch", "implementation"}),
    AgentRole.TESTER: frozenset({"test_result", "test_plan", "runtime_evidence"}),
    AgentRole.SECURITY_REVIEWER: frozenset({"security_review", "security_evidence"}),
    AgentRole.CRITIC: frozenset({"critic_review"}),
    AgentRole.VERIFIER: frozenset({"verification_decision"}),
    AgentRole.ORCHESTRATOR: frozenset({"coordination", "revision_request"}),
}
REVIEW_ARTIFACT_TYPE = {
    AgentRole.TESTER: "test_plan",
    AgentRole.SECURITY_REVIEWER: "security_review",
    AgentRole.CRITIC: "critic_review",
}


def _redact_payload(value: Any, *, key: str = "") -> Any:
    if SECRET_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _redact_payload(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item, key=key) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    return value


def _payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


class AgentMessage(StrictModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(min_length=1, max_length=128)
    task_graph_id: str = Field(min_length=1, max_length=128)
    task_id: str | None = Field(default=None, min_length=1, max_length=128)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=128)
    sender_role: AgentRole
    recipient_role: AgentRole
    kind: MessageKind
    payload: dict[str, Any]
    evidence_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)
    created_at_unix: float = Field(default_factory=time.time)

    @field_validator("message_id", "run_id", "task_graph_id", "correlation_id")
    @classmethod
    def validate_required_uuid(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("identifier must be a canonical UUID") from exc

    @field_validator("task_id")
    @classmethod
    def validate_optional_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("task_id must be a canonical UUID") from exc

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = _redact_payload(value)
        if _payload_size(redacted) > MAX_MESSAGE_BYTES:
            raise ValueError(f"agent message payload exceeds {MAX_MESSAGE_BYTES} bytes")
        return redacted

    @field_validator("evidence_refs")
    @classmethod
    def unique_evidence_refs(cls, values: list[str]) -> list[str]:
        if any(not item or len(item) > 128 for item in values):
            raise ValueError("evidence references must contain 1-128 characters")
        if len(set(values)) != len(values):
            raise ValueError("evidence references must be unique")
        return values

    @model_validator(mode="after")
    def enforce_role_separation(self) -> "AgentMessage":
        if self.sender_role == AgentRole.BROADCAST:
            raise ValueError("broadcast is a recipient only")
        allowed = ROLE_MESSAGE_POLICY[self.sender_role]
        if self.kind not in allowed:
            raise ValueError(f"{self.sender_role.value} cannot emit {self.kind.value} messages")
        artifact_type = str(self.payload.get("artifact_type") or "")
        if not artifact_type:
            raise ValueError("agent messages require a typed artifact_type")
        if artifact_type and artifact_type not in ROLE_ARTIFACT_POLICY[self.sender_role]:
            raise ValueError(
                f"{self.sender_role.value} cannot publish artifact_type={artifact_type}; role mixing is forbidden"
            )
        if self.sender_role == self.recipient_role and self.kind in {
            MessageKind.CHALLENGE,
            MessageKind.REVIEW,
            MessageKind.DECISION,
        }:
            raise ValueError("review, challenge, and decision messages require a distinct recipient")
        forbidden_reasoning_keys = {"chain_of_thought", "raw_thoughts", "reasoning_steps"}
        if forbidden_reasoning_keys.intersection(self.payload):
            raise ValueError("raw private reasoning must not be persisted in agent messages")
        if self.sender_role == AgentRole.CRITIC and self.kind == MessageKind.REVIEW:
            confidence = self.payload.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0:
                raise ValueError("critic review requires confidence between 0 and 1")
        if self.sender_role == AgentRole.VERIFIER:
            accepted = self.payload.get("accepted")
            if not isinstance(accepted, bool):
                raise ValueError("verifier decision requires a boolean accepted field")
            if accepted and not self.evidence_refs:
                raise ValueError("accepted verifier decisions require verification evidence references")
        return self


class BlackboardWrite(StrictModel):
    run_id: str = Field(min_length=1, max_length=128)
    task_graph_id: str = Field(min_length=1, max_length=128)
    entry_key: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    kind: BlackboardKind
    value: dict[str, Any]
    expected_version: int | None = Field(default=None, ge=0)
    verified: bool = False
    source_message_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("run_id", "task_graph_id")
    @classmethod
    def validate_run_uuid(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("run and task graph identifiers must be canonical UUIDs") from exc

    @field_validator("source_message_id")
    @classmethod
    def validate_source_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("source_message_id must be a canonical UUID") from exc

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = _redact_payload(value)
        if _payload_size(redacted) > MAX_MESSAGE_BYTES:
            raise ValueError(f"blackboard value exceeds {MAX_MESSAGE_BYTES} bytes")
        return redacted


class BlackboardEntry(StrictModel):
    entry_id: str
    run_id: str
    task_graph_id: str
    entry_key: str
    kind: BlackboardKind
    value: dict[str, Any]
    version: int = Field(ge=1)
    immutable: bool
    verified: bool
    source_message_id: str | None = None
    created_at_unix: float
    updated_at_unix: float


class PeerReviewResult(StrictModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list, max_length=100)
    evidence_requests: list[str] = Field(default_factory=list, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)


class CriticScore(StrictModel):
    confidence: float = Field(ge=0.0, le=1.0)
    flaws: list[str] = Field(default_factory=list, max_length=100)
    assumptions_wrong: list[str] = Field(default_factory=list, max_length=100)
    failure_modes: list[str] = Field(default_factory=list, max_length=100)
    security_risks: list[str] = Field(default_factory=list, max_length=100)
    unverified_evidence: list[str] = Field(default_factory=list, max_length=100)


class VerifierDecision(StrictModel):
    accepted: bool
    criteria_passed: list[str] = Field(default_factory=list, max_length=100)
    criteria_failed: list[str] = Field(default_factory=list, max_length=100)
    verification_refs: list[str] = Field(default_factory=list, max_length=128)


class NegotiationReport(StrictModel):
    run_id: str
    correlation_id: str
    accepted: bool
    revision_count: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    final_proposal_message_id: str
    decision_message_id: str
    message_count: int
    memory_promoted: bool = False
    reflection_questions: list[str] = Field(default_factory=lambda: list(REFLECTION_QUESTIONS))


PeerReviewer = Callable[[AgentMessage], Awaitable[PeerReviewResult]]
EvidenceResponder = Callable[[AgentMessage, PeerReviewResult], Awaitable[dict[str, Any]]]
Critic = Callable[[AgentMessage, PeerReviewResult], Awaitable[CriticScore]]
Verifier = Callable[[AgentMessage, CriticScore], Awaitable[VerifierDecision]]
Reviser = Callable[[AgentMessage, CriticScore, list[str]], Awaitable[dict[str, Any]]]


class CollaborationRuntime:
    """Validated durable message bus, blackboard, and bounded negotiation."""

    def __init__(self, store: LocalStore, *, confidence_threshold: float = 0.85, max_revisions: int = 3) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= max_revisions <= 3:
            raise ValueError("max_revisions must be between 0 and 3")
        self.store = store
        self.confidence_threshold = confidence_threshold
        self.max_revisions = max_revisions

    def publish(self, message: AgentMessage) -> AgentMessage:
        self._validate_run_binding(message.run_id, message.task_graph_id, message.task_id)
        return AgentMessage.model_validate(self.store.insert_agent_message(message.model_dump(mode="json")))

    def history(self, run_id: str, *, correlation_id: str | None = None, limit: int = 500) -> list[AgentMessage]:
        if self.store.get_run(run_id) is None:
            raise KeyError(f"unknown run: {run_id}")
        return [
            AgentMessage.model_validate(item)
            for item in self.store.list_agent_messages(run_id, correlation_id=correlation_id, limit=limit)
        ]

    def write_blackboard(self, request: BlackboardWrite) -> BlackboardEntry:
        self._validate_run_binding(request.run_id, request.task_graph_id, None)
        if request.source_message_id:
            source = self.store.get_agent_message(request.source_message_id)
            if source is None or source["run_id"] != request.run_id:
                raise ValueError("blackboard source message must belong to the same run")
        immutable = request.kind == BlackboardKind.EVIDENCE
        stored = self.store.put_blackboard_entry(
            entry_id=str(uuid.uuid4()),
            run_id=request.run_id,
            task_graph_id=request.task_graph_id,
            entry_key=request.entry_key,
            kind=request.kind.value,
            value=request.value,
            immutable=immutable,
            verified=request.verified,
            source_message_id=request.source_message_id,
            expected_version=request.expected_version,
        )
        return BlackboardEntry.model_validate(stored)

    def blackboard(self, run_id: str, *, kind: BlackboardKind | None = None) -> list[BlackboardEntry]:
        if self.store.get_run(run_id) is None:
            raise KeyError(f"unknown run: {run_id}")
        return [
            BlackboardEntry.model_validate(item)
            for item in self.store.list_blackboard_entries(run_id, kind=kind.value if kind else None)
        ]

    async def negotiate(
        self,
        proposal: AgentMessage,
        *,
        peer_role: AgentRole,
        peer_reviewer: PeerReviewer,
        evidence_responder: EvidenceResponder,
        critic: Critic,
        verifier: Verifier,
        reviser: Reviser,
        memory: UnifiedMemoryManager | None = None,
    ) -> NegotiationReport:
        if proposal.kind != MessageKind.PROPOSAL:
            raise ValueError("negotiation must start with a proposal message")
        if peer_role == proposal.sender_role or peer_role not in REVIEW_ARTIFACT_TYPE:
            raise ValueError("peer reviewer must be a distinct tester, security reviewer, or critic")
        correlation_id = proposal.correlation_id
        current = self.publish(proposal)
        self.write_blackboard(
            BlackboardWrite(
                run_id=current.run_id,
                task_graph_id=current.task_graph_id,
                entry_key=f"artifact/{correlation_id}",
                kind=BlackboardKind.ARTIFACT,
                value={"proposal_message_id": current.message_id, "payload": current.payload, "revision": 0},
                expected_version=0,
                source_message_id=current.message_id,
            )
        )
        decision_message: AgentMessage | None = None
        final_score = 0.0
        accepted = False
        revision = 0
        board_version = 1

        while True:
            peer_result = await peer_reviewer(current)
            peer_kind = MessageKind.REVIEW if peer_result.accepted else MessageKind.CHALLENGE
            peer_message = self.publish(
                AgentMessage(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    task_id=current.task_id,
                    correlation_id=correlation_id,
                    sender_role=peer_role,
                    recipient_role=current.sender_role,
                    kind=peer_kind,
                    payload={
                        "artifact_type": REVIEW_ARTIFACT_TYPE[peer_role],
                        **peer_result.model_dump(),
                    },
                    evidence_refs=current.evidence_refs,
                )
            )
            evidence_payload = await evidence_responder(current, peer_result)
            evidence_message = self.publish(
                AgentMessage(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    task_id=current.task_id,
                    correlation_id=correlation_id,
                    sender_role=current.sender_role,
                    recipient_role=peer_role,
                    kind=MessageKind.EVIDENCE,
                    payload=evidence_payload,
                    evidence_refs=current.evidence_refs,
                )
            )
            self.write_blackboard(
                BlackboardWrite(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    entry_key=f"evidence/{evidence_message.message_id}",
                    kind=BlackboardKind.EVIDENCE,
                    value={"payload": evidence_message.payload, "peer_message_id": peer_message.message_id},
                    verified=False,
                    source_message_id=evidence_message.message_id,
                )
            )
            critic_score = await critic(current, peer_result)
            final_score = critic_score.confidence
            critic_message = self.publish(
                AgentMessage(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    task_id=current.task_id,
                    correlation_id=correlation_id,
                    sender_role=AgentRole.CRITIC,
                    recipient_role=AgentRole.VERIFIER,
                    kind=MessageKind.REVIEW,
                    payload={"artifact_type": "critic_review", **critic_score.model_dump()},
                    evidence_refs=[evidence_message.message_id],
                )
            )
            verifier_result = await verifier(current, critic_score)
            accepted = (
                peer_result.accepted
                and critic_score.confidence >= self.confidence_threshold
                and verifier_result.accepted
                and bool(verifier_result.verification_refs)
            )
            decision_message = self.publish(
                AgentMessage(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    task_id=current.task_id,
                    correlation_id=correlation_id,
                    sender_role=AgentRole.VERIFIER,
                    recipient_role=AgentRole.ORCHESTRATOR,
                    kind=MessageKind.DECISION,
                    payload={
                        "artifact_type": "verification_decision",
                        **verifier_result.model_dump(),
                        "accepted": accepted,
                        "critic_confidence": critic_score.confidence,
                        "critic_message_id": critic_message.message_id,
                    },
                    evidence_refs=verifier_result.verification_refs,
                )
            )
            if accepted or revision >= self.max_revisions:
                break
            revision += 1
            revised_payload = await reviser(current, critic_score, list(REFLECTION_QUESTIONS))
            current = self.publish(
                AgentMessage(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    task_id=current.task_id,
                    correlation_id=correlation_id,
                    sender_role=current.sender_role,
                    recipient_role=peer_role,
                    kind=MessageKind.PROPOSAL,
                    payload=revised_payload,
                    evidence_refs=current.evidence_refs,
                )
            )
            board_version += 1
            self.write_blackboard(
                BlackboardWrite(
                    run_id=current.run_id,
                    task_graph_id=current.task_graph_id,
                    entry_key=f"artifact/{correlation_id}",
                    kind=BlackboardKind.ARTIFACT,
                    value={
                        "proposal_message_id": current.message_id,
                        "payload": current.payload,
                        "revision": revision,
                        "reflection_questions": list(REFLECTION_QUESTIONS),
                    },
                    expected_version=board_version - 1,
                    source_message_id=current.message_id,
                )
            )

        if decision_message is None:
            raise RuntimeError("negotiation ended without a verifier decision")
        self.write_blackboard(
            BlackboardWrite(
                run_id=current.run_id,
                task_graph_id=current.task_graph_id,
                entry_key=f"decision/{correlation_id}",
                kind=BlackboardKind.DECISION,
                value={
                    "accepted": accepted,
                    "confidence": final_score,
                    "decision_message_id": decision_message.message_id,
                    "revision_count": revision,
                },
                expected_version=0,
                verified=accepted,
                source_message_id=decision_message.message_id,
            )
        )
        promoted = False
        if accepted and memory is not None:
            memory.ingest(
                MemoryIngestRequest(
                    layer="episodic",
                    kind="successful_plan",
                    content={
                        "proposal": current.payload,
                        "decision_message_id": decision_message.message_id,
                        "verification_refs": decision_message.evidence_refs,
                    },
                    relevance=0.85,
                    success_rate=1.0,
                    source_run_id=current.run_id,
                    metadata={"correlation_id": correlation_id, "verified": True},
                )
            )
            promoted = True
        return NegotiationReport(
            run_id=current.run_id,
            correlation_id=correlation_id,
            accepted=accepted,
            revision_count=revision,
            confidence=final_score,
            final_proposal_message_id=current.message_id,
            decision_message_id=decision_message.message_id,
            message_count=len(self.history(current.run_id, correlation_id=correlation_id)),
            memory_promoted=promoted,
        )

    def _validate_run_binding(self, run_id: str, task_graph_id: str, task_id: str | None) -> None:
        run = self.store.get_run(run_id)
        graph = self.store.get_task_graph(task_graph_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if graph is None or graph.get("run_id") != run_id:
            raise ValueError("task graph does not belong to the supplied run")
        if task_id:
            task = self.store.get_task(task_id)
            if task is None or task["run_id"] != run_id or task["task_graph_id"] != task_graph_id:
                raise ValueError("task does not belong to the supplied run and graph")


PATH_PATTERN = re.compile(r"(?:[A-Za-z]:)?(?:[/\\][^/\\\s:]+){2,}")
ADDRESS_PATTERN = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
LINE_PATTERN = re.compile(r"(?i)\b(?:line|column|offset|pid|thread)\s*[=:]?\s*\d+\b")
NUMBER_PATTERN = re.compile(r"\b\d{2,}\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


class FailureIntelligence:
    """Normalize, cluster, resolve, and safely promote repeated failures."""

    def __init__(self, store: LocalStore, *, candidate_threshold: int = 2) -> None:
        if not 2 <= candidate_threshold <= 100:
            raise ValueError("candidate_threshold must be between 2 and 100")
        self.store = store
        self.candidate_threshold = candidate_threshold

    @staticmethod
    def normalize(error: str) -> tuple[str, str]:
        if not error.strip():
            raise ValueError("failure error must not be empty")
        normalized = error[:32_768]
        normalized = SECRET_VALUE_PATTERN.sub("[REDACTED]", normalized)
        normalized = PATH_PATTERN.sub("<path>", normalized)
        normalized = ADDRESS_PATTERN.sub("<address>", normalized)
        normalized = UUID_PATTERN.sub("<uuid>", normalized)
        normalized = LINE_PATTERN.sub("<location>", normalized)
        normalized = NUMBER_PATTERN.sub("<number>", normalized)
        normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip().lower()
        signature = normalized[:2_000]
        cluster_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        return signature, cluster_key

    def observe(
        self,
        error: str,
        *,
        project_id: str | None,
        run_id: str | None,
        task_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        signature, cluster_key = self.normalize(error)
        return self.store.record_failure(
            project_id=project_id,
            run_id=run_id,
            task_id=task_id,
            signature=signature,
            cluster_key=cluster_key,
            raw_error=_redact_payload({"error": error[:32_768]})["error"],
            metadata=metadata,
        )

    def resolve(
        self,
        failure_id: str,
        *,
        root_cause: str,
        patch_id: str,
        verification_ref: str,
        verification_passed: bool,
    ) -> dict[str, Any]:
        if not root_cause.strip() or not patch_id.strip() or not verification_ref.strip():
            raise ValueError("root cause, patch, and verification reference are required")
        result = self.store.link_failure_resolution(
            failure_id,
            root_cause=str(_redact_payload({"root_cause": root_cause[:8_000]})["root_cause"]),
            patch_id=patch_id,
            verification_ref=verification_ref[:2_000],
            verified=verification_passed,
        )
        if verification_passed and int(result["occurrence_count"]) >= self.candidate_threshold:
            candidate = self.store.insert_learning_candidate(
                project_id=result.get("project_id"),
                run_id=result.get("run_id"),
                patch_id=patch_id,
                kind="verified_failure_repair",
                prompt=f"Diagnose and repair this normalized failure:\n{result['signature']}",
                chosen=f"Root cause: {root_cause}\nVerified patch reference: {patch_id}",
                verification={
                    "passed": True,
                    "verification_ref": verification_ref,
                    "cluster_key": result["cluster_key"],
                    "occurrence_count": result["occurrence_count"],
                },
                score=1.0,
            )
            self.store.attach_failure_candidate(failure_id, str(candidate["id"]))
            result = self.store.get_failure(failure_id) or result
            result["dataset_candidate"] = candidate
        return result

    def clusters(self, project_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_failure_clusters(project_id)


async def prove_postgres_collaboration(database_url: str, output_dir: str | Path) -> dict[str, Any]:
    """Run a destructive-safe, namespaced Postgres lifecycle and contention proof."""

    if not database_url.startswith(("postgres://", "postgresql://")):
        raise ValueError("Postgres proof requires a postgres:// or postgresql:// URL")
    try:
        import asyncpg
    except ImportError as exc:
        raise RuntimeError("asyncpg is required for the Postgres collaboration proof") from exc

    proof_id = uuid.uuid4()
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    graph_id = uuid.uuid4()
    task_id = uuid.uuid4()
    message_id = uuid.uuid4()
    blackboard_id = uuid.uuid4()
    started = time.perf_counter()
    pool = await asyncpg.create_pool(database_url, min_size=2, max_size=4, command_timeout=30)
    stale_update_rejected = False
    claim_winners: list[str] = []

    async def claim(worker: str) -> str | None:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT id FROM tasks
                    WHERE id = $1 AND status = 'queued' AND cancel_requested = false
                    FOR UPDATE SKIP LOCKED
                    """,
                    task_id,
                )
                if row is None:
                    return None
                await connection.execute(
                    """
                    UPDATE tasks SET status = 'running', lease_owner = $1,
                      lease_expires_at = now() + interval '60 seconds', started_at = now()
                    WHERE id = $2
                    """,
                    worker,
                    task_id,
                )
                return worker

    try:
        async with pool.acquire() as connection:
            required_tables = ["projects", "runs", "task_graphs", "tasks", "agent_messages", "blackboard_entries", "failure_records"]
            for table in required_tables:
                present = await connection.fetchval("SELECT to_regclass($1)", f"public.{table}")
                if present is None:
                    raise RuntimeError(f"required Postgres table is missing: {table}; apply migration 0005")
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO projects(id, name, repo_path, created_at, updated_at)
                    VALUES($1, $2, $3, now(), now())
                    """,
                    project_id,
                    f"collaboration-proof-{proof_id}",
                    f"/proof/{proof_id}",
                )
                await connection.execute(
                    """
                    INSERT INTO runs(id, project_id, prompt, mode, status, model_profile, created_at)
                    VALUES($1, $2, 'proof', 'code_edit', 'queued', 'proof', now())
                    """,
                    run_id,
                    project_id,
                )
                await connection.execute(
                    """
                    INSERT INTO task_graphs(id, project_id, run_id, goal, status, graph_json)
                    VALUES($1, $2, $3, 'proof', 'queued', '{}'::jsonb)
                    """,
                    graph_id,
                    project_id,
                    run_id,
                )
                await connection.execute(
                    """
                    INSERT INTO tasks(id, task_graph_id, run_id, kind, title, status)
                    VALUES($1, $2, $3, 'test', 'contention proof', 'queued')
                    """,
                    task_id,
                    graph_id,
                    run_id,
                )
                await connection.execute(
                    """
                    INSERT INTO agent_messages(
                      id, run_id, task_graph_id, task_id, correlation_id, sender_role,
                      recipient_role, kind, payload_json, evidence_refs
                    ) VALUES($1, $2, $3, $4, $5, 'tester', 'critic', 'evidence',
                      '{"artifact_type":"test_result","passed":true}'::jsonb, ARRAY['proof'])
                    """,
                    message_id,
                    run_id,
                    graph_id,
                    task_id,
                    proof_id,
                )
                await connection.execute(
                    """
                    INSERT INTO blackboard_entries(
                      id, run_id, task_graph_id, entry_key, kind, value_json,
                      immutable, verified, source_message_id
                    ) VALUES($1, $2, $3, 'proof/fact', 'fact', '{"value":1}'::jsonb,
                      false, false, $4)
                    """,
                    blackboard_id,
                    run_id,
                    graph_id,
                    message_id,
                )

        winners = await asyncio.gather(claim("worker-a"), claim("worker-b"))
        claim_winners = [item for item in winners if item is not None]
        if len(claim_winners) != 1:
            raise RuntimeError(f"atomic claim proof expected one winner, got {claim_winners}")

        async with pool.acquire() as connection:
            updated = await connection.execute(
                """
                UPDATE blackboard_entries SET value_json = '{"value":2}'::jsonb, version = version + 1
                WHERE id = $1 AND version = 1
                """,
                blackboard_id,
            )
            if updated != "UPDATE 1":
                raise RuntimeError("blackboard compare-and-swap proof did not update version 1")
            stale = await connection.execute(
                """
                UPDATE blackboard_entries SET value_json = '{"value":3}'::jsonb, version = version + 1
                WHERE id = $1 AND version = 1
                """,
                blackboard_id,
            )
            stale_update_rejected = stale == "UPDATE 0"
            if not stale_update_rejected:
                raise RuntimeError("stale blackboard update was not rejected")
            stored_message = await connection.fetchval(
                "SELECT payload_json->>'artifact_type' FROM agent_messages WHERE id = $1",
                message_id,
            )
            if stored_message != "test_result":
                raise RuntimeError("durable message round-trip failed")
    finally:
        try:
            async with pool.acquire() as connection:
                await connection.execute("DELETE FROM projects WHERE id = $1", project_id)
        finally:
            await pool.close()

    report = {
        "status": "passed",
        "proof_id": str(proof_id),
        "migration": "0005_agent_collaboration",
        "atomic_claim_winner_count": len(claim_winners),
        "blackboard_stale_update_rejected": stale_update_rejected,
        "durable_message_round_trip": True,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "created_at_unix": time.time(),
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "agent_collaboration_postgres_proof.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(target.resolve())
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Craftly agent collaboration production proof")
    parser.add_argument("--postgres-proof", action="store_true")
    parser.add_argument("--database-url", default=os.environ.get("CRAFTLY_DATABASE_URL", ""))
    parser.add_argument("--output-dir", default="artifacts/craftly/agent-collaboration-proof")
    args = parser.parse_args()
    if not args.postgres_proof:
        parser.error("--postgres-proof is required")
    report = asyncio.run(prove_postgres_collaboration(args.database_url, args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
