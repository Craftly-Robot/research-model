"""Durable, blind, two-reviewer dataset quality authority.

The authority never scores or rewrites content. It owns only identity-bound
review decisions and immutable promotion evidence. Dataset construction stays
in ``production_dataset`` as the single promotion path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from src.craftly.shared.integrity import canonical_json_text as _json
from src.craftly.shared.integrity import sha256_file as _hash_file
from src.craftly.shared.schemas import StrictModel

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVIEW_RUBRIC_ID = "craftly-review-rubric-v1"
PLACEHOLDER_MARKERS = (
    "placeholder",
    "real_reviewer",
    "real_adjudicator",
    "real_oidc",
    "example",
    "replace_me",
    "todo",
)
REVIEW_STATUSES = {
    "pending",
    "in_review",
    "approved",
    "rejected",
    "conflict",
    "adjudication_required",
}
SQLITE_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS data_source_snapshots (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  source_family TEXT NOT NULL,
  immutable_revision TEXT NOT NULL,
  registry_sha256 TEXT NOT NULL,
  license_evidence_sha256 TEXT NOT NULL,
  legal_approval_sha256 TEXT NOT NULL,
  snapshot_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(source_id, immutable_revision, snapshot_sha256)
);
CREATE TABLE IF NOT EXISTS dataset_review_items (
  id TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  source_id TEXT NOT NULL,
  data_type TEXT NOT NULL,
  high_value INTEGER NOT NULL,
  status TEXT NOT NULL,
  version INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(content_hash, source_snapshot_sha256)
);
CREATE TABLE IF NOT EXISTS dataset_review_assignments (
  id TEXT PRIMARY KEY,
  review_item_id TEXT NOT NULL REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  reviewer_id TEXT NOT NULL,
  reviewer_slot INTEGER NOT NULL CHECK(reviewer_slot IN (1,2)),
  status TEXT NOT NULL,
  claimed_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  UNIQUE(review_item_id, reviewer_id),
  UNIQUE(review_item_id, reviewer_slot)
);
CREATE TABLE IF NOT EXISTS dataset_review_decisions (
  id TEXT PRIMARY KEY,
  review_item_id TEXT NOT NULL REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  reviewer_id TEXT NOT NULL,
  decision TEXT NOT NULL CHECK(decision IN ('approve','reject')),
  rationale TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(review_item_id, reviewer_id)
);
CREATE TABLE IF NOT EXISTS dataset_review_adjudications (
  id TEXT PRIMARY KEY,
  review_item_id TEXT NOT NULL UNIQUE REFERENCES dataset_review_items(id) ON DELETE CASCADE,
  adjudicator_id TEXT NOT NULL,
  decision TEXT NOT NULL CHECK(decision IN ('approve','reject')),
  rationale TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_promotion_events (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL,
  decision TEXT NOT NULL CHECK(decision IN ('promoted','rejected')),
  evidence_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(dataset_id,version_id,manifest_sha256)
);
CREATE INDEX IF NOT EXISTS idx_dataset_review_status ON dataset_review_items(status,created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_review_source ON dataset_review_items(source_id,data_type,status);
"""


def _validate_hash(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be SHA-256 hex")
    return normalized


class ReviewItemCreate(StrictModel):
    content_hash: str
    source_snapshot_sha256: str
    source_id: str = Field(min_length=2, max_length=128)
    data_type: str = Field(min_length=2, max_length=80)
    high_value: bool
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> ReviewItemCreate:
        object.__setattr__(self, "content_hash", _validate_hash(self.content_hash, "content_hash"))
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _validate_hash(self.source_snapshot_sha256, "source_snapshot_sha256"),
        )
        return self


class SourceSnapshotCreate(StrictModel):
    source_id: str = Field(min_length=2, max_length=128)
    source_family: str = Field(min_length=2, max_length=128)
    immutable_revision: str = Field(min_length=1, max_length=256)
    registry_sha256: str
    license_evidence_sha256: str
    legal_approval_sha256: str
    snapshot_sha256: str
    status: Literal["quarantine", "approved", "revoked"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> SourceSnapshotCreate:
        if self.immutable_revision == "rolling":
            raise ValueError("source snapshot revision must be immutable")
        for name in (
            "registry_sha256",
            "license_evidence_sha256",
            "legal_approval_sha256",
            "snapshot_sha256",
        ):
            object.__setattr__(self, name, _validate_hash(getattr(self, name), name))
        return self


class SourceSnapshotView(SourceSnapshotCreate):
    snapshot_id: str
    created_at: float


class ReviewDecisionCreate(StrictModel):
    decision: Literal["approve", "reject"]
    rationale: str = Field(min_length=20, max_length=4_000)
    content_hash: str
    source_snapshot_sha256: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> ReviewDecisionCreate:
        object.__setattr__(self, "content_hash", _validate_hash(self.content_hash, "content_hash"))
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _validate_hash(self.source_snapshot_sha256, "source_snapshot_sha256"),
        )
        return self


class ReviewAdjudicationCreate(StrictModel):
    decision: Literal["approve", "reject"]
    rationale: str = Field(min_length=30, max_length=4_000)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReviewAssignment(StrictModel):
    assignment_id: str
    review_item_id: str
    reviewer_id: str
    reviewer_slot: int = Field(ge=1, le=2)
    status: str
    claimed_at: float
    expires_at: float


class ReviewDecisionView(StrictModel):
    reviewer_id: str
    decision: Literal["approve", "reject"]
    rationale: str
    evidence: dict[str, Any]
    created_at: float


class ReviewItemView(StrictModel):
    review_item_id: str
    content_hash: str
    source_snapshot_sha256: str
    source_id: str
    data_type: str
    high_value: bool
    status: str
    version: int
    payload: dict[str, Any]
    assignments: list[ReviewAssignment] = Field(default_factory=list)
    decisions: list[ReviewDecisionView] = Field(default_factory=list)
    created_at: float
    updated_at: float


class PromotionEvidenceCreate(StrictModel):
    dataset_id: str = Field(min_length=2, max_length=160)
    version_id: str = Field(min_length=2, max_length=160)
    manifest_sha256: str
    policy_sha256: str
    decision: Literal["promoted", "rejected"]
    evidence: dict[str, Any]

    @model_validator(mode="after")
    def validate_contract(self) -> PromotionEvidenceCreate:
        object.__setattr__(
            self, "manifest_sha256", _validate_hash(self.manifest_sha256, "manifest_sha256")
        )
        object.__setattr__(
            self, "policy_sha256", _validate_hash(self.policy_sha256, "policy_sha256")
        )
        return self


class PromotionEvidenceView(PromotionEvidenceCreate):
    promotion_id: str
    actor_id: str
    created_at: float


class SourceReviewEvidence(StrictModel):
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    pending: int = Field(ge=0)
    paired_reviews: int = Field(ge=0)
    reviewer_agreement_kappa: float = Field(ge=-1.0, le=1.0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)


class ReviewEvidenceReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["empty", "in_progress", "complete"]
    total_items: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    pending: int = Field(ge=0)
    paired_reviews: int = Field(ge=0)
    by_source: dict[str, SourceReviewEvidence]
    created_at_unix: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_aggregates(self) -> ReviewEvidenceReport:
        if self.total_items != self.approved + self.rejected + self.pending:
            raise ValueError("review evidence item totals are inconsistent")
        if self.source_count != len(self.by_source):
            raise ValueError("review evidence source_count is inconsistent")
        if self.approved != sum(item.approved for item in self.by_source.values()):
            raise ValueError("review evidence approved total is inconsistent")
        if self.rejected != sum(item.rejected for item in self.by_source.values()):
            raise ValueError("review evidence rejected total is inconsistent")
        if self.pending != sum(item.pending for item in self.by_source.values()):
            raise ValueError("review evidence pending total is inconsistent")
        if self.paired_reviews != sum(item.paired_reviews for item in self.by_source.values()):
            raise ValueError("review evidence paired_reviews total is inconsistent")
        expected_status = (
            "empty"
            if self.total_items == 0
            else ("complete" if self.pending == 0 else "in_progress")
        )
        if self.status != expected_status:
            raise ValueError(f"review evidence status must be {expected_status!r}")
        return self


class ReviewerIdentity(StrictModel):
    reviewer_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    identity_provider_subject: str = Field(min_length=3, max_length=256)
    roles: set[Literal["reviewer", "adjudicator"]] = Field(min_length=1)
    active: bool = True
    approved_by: str = Field(min_length=3, max_length=256)
    approved_at: str

    @field_validator("approved_at")
    @classmethod
    def validate_approval_timestamp(cls, value: str) -> str:
        from datetime import datetime

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewer approved_at must be RFC3339") from exc
        if parsed.tzinfo is None:
            raise ValueError("reviewer approved_at must include timezone")
        return value


class ReviewerRoster(StrictModel):
    schema_version: Literal[1] = 1
    roster_id: str = Field(min_length=3, max_length=120)
    identities: list[ReviewerIdentity]

    @model_validator(mode="after")
    def validate_unique_identities(self) -> ReviewerRoster:
        reviewer_ids = [identity.reviewer_id for identity in self.identities]
        subjects = [identity.identity_provider_subject for identity in self.identities]
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError("reviewer roster contains duplicate reviewer_id")
        if len(subjects) != len(set(subjects)):
            raise ValueError("reviewer roster contains duplicate identity-provider subject")
        return self

    def active_identity(
        self, reviewer_id: str, role: Literal["reviewer", "adjudicator"]
    ) -> ReviewerIdentity:
        identity = next(
            (
                item
                for item in self.identities
                if item.reviewer_id == reviewer_id and item.active and role in item.roles
            ),
            None,
        )
        if identity is None:
            raise PermissionError(f"{reviewer_id!r} is not an active {role} in the governed roster")
        return identity

    def readiness_blockers(self) -> list[str]:
        reviewers = {
            identity.identity_provider_subject
            for identity in self.identities
            if identity.active and "reviewer" in identity.roles
        }
        adjudicators = {
            identity.identity_provider_subject
            for identity in self.identities
            if identity.active and "adjudicator" in identity.roles
        }
        blockers: list[str] = []
        if len(reviewers) < 2:
            blockers.append(
                "at least two active reviewers with distinct identity-provider subjects are required"
            )
        if not adjudicators:
            blockers.append("at least one active adjudicator is required")
        if adjudicators and not any(subject not in reviewers for subject in adjudicators):
            blockers.append(
                "an adjudicator identity independent of both reviewer identities is required"
            )
        return blockers


class ReviewerRosterSlot(StrictModel):
    slot: Literal["reviewer_1", "reviewer_2", "adjudicator"]
    required_role: Literal["reviewer", "adjudicator"]
    required_fields: list[str]


class ReviewerRosterOnboardingTemplate(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["awaiting_human_identities"] = "awaiting_human_identities"
    roster_id: str
    required_slots: list[ReviewerRosterSlot]
    final_roster_path: str
    warning: str


class ReviewerRosterReadinessReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["ready", "blocked"]
    roster_id: str
    active_reviewer_count: int = Field(ge=0)
    active_adjudicator_count: int = Field(ge=0)
    independent_subject_count: int = Field(ge=0)
    blockers: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> ReviewerRosterReadinessReport:
        expected = "blocked" if self.blockers else "ready"
        if self.status != expected:
            raise ValueError(f"reviewer roster readiness status must be {expected!r}")
        return self


class ReviewerGovernanceBundleReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["awaiting_human_identities", "ready_for_qualification"] = (
        "awaiting_human_identities"
    )
    roster_id: str
    rubric_id: str
    output_dir: str
    roster_path: str
    rubric_path: str
    onboarding_path: str
    roster_sha256: str
    rubric_sha256: str
    warning: str

    @model_validator(mode="after")
    def validate_hashes(self) -> ReviewerGovernanceBundleReport:
        object.__setattr__(
            self, "roster_sha256", _validate_hash(self.roster_sha256, "roster_sha256")
        )
        object.__setattr__(
            self, "rubric_sha256", _validate_hash(self.rubric_sha256, "rubric_sha256")
        )
        return self


class ReviewerDeliveryArtifact(StrictModel):
    reviewer_id: str
    package_path: str
    package_sha256: str
    item_count: Literal[20] = 20
    answer_key_included: Literal[False] = False

    @field_validator("package_sha256")
    @classmethod
    def validate_package_hash(cls, value: str) -> str:
        return _validate_hash(value, "package_sha256")


class ReviewerDeliveryReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["ready_for_secure_delivery"] = "ready_for_secure_delivery"
    roster_id: str
    rubric_id: Literal["craftly-review-rubric-v1"] = REVIEW_RUBRIC_ID
    rubric_sha256: str
    qualification_pack_sha256: str
    packages: list[ReviewerDeliveryArtifact]
    warning: str

    @model_validator(mode="after")
    def validate_contract(self) -> ReviewerDeliveryReport:
        object.__setattr__(
            self, "rubric_sha256", _validate_hash(self.rubric_sha256, "rubric_sha256")
        )
        object.__setattr__(
            self,
            "qualification_pack_sha256",
            _validate_hash(self.qualification_pack_sha256, "qualification_pack_sha256"),
        )
        if len(self.packages) != 2:
            raise ValueError("reviewer delivery requires exactly two reviewer packages")
        if len({package.reviewer_id for package in self.packages}) != 2:
            raise ValueError("reviewer delivery packages must target distinct reviewers")
        return self


class ReviewerQualificationItem(StrictModel):
    schema_version: Literal[1] = 1
    item_id: str
    rubric_id: Literal["craftly-review-rubric-v1"] = REVIEW_RUBRIC_ID
    data_type: Literal["security_document", "code_example", "patch", "debug_trace"]
    title: str
    candidate_summary: str
    evidence_summary: str


class ReviewerQualificationAnswer(StrictModel):
    item_id: str
    expected_decision: Literal["approve", "reject"]
    reason_code: str | None = None
    rationale: str = Field(min_length=20, max_length=1_000)


class ReviewerQualificationPackReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["ready_for_blind_qualification"] = "ready_for_blind_qualification"
    rubric_id: Literal["craftly-review-rubric-v1"] = REVIEW_RUBRIC_ID
    item_count: Literal[20] = 20
    expected_approve_count: Literal[10] = 10
    expected_reject_count: Literal[10] = 10
    pack_path: str
    answer_key_path: str
    pack_sha256: str
    answer_key_sha256: str
    handling_warning: str

    @model_validator(mode="after")
    def validate_hashes(self) -> ReviewerQualificationPackReport:
        object.__setattr__(self, "pack_sha256", _validate_hash(self.pack_sha256, "pack_sha256"))
        object.__setattr__(
            self,
            "answer_key_sha256",
            _validate_hash(self.answer_key_sha256, "answer_key_sha256"),
        )
        return self


class ReviewerQualificationResponse(StrictModel):
    item_id: str = Field(pattern=r"^reviewer-qualification-\d{2}$")
    reviewer_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    decision: Literal["approve", "reject"]
    rationale: str = Field(min_length=30, max_length=1_000)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 30:
            raise ValueError(
                "qualification rationale must contain at least 30 non-whitespace characters"
            )
        if normalized.lower() in {"looks good", "approve", "reject", "not good"}:
            raise ValueError("qualification rationale must be concrete")
        return normalized


class ReviewerQualificationScore(StrictModel):
    reviewer_id: str
    response_sha256: str
    item_count: Literal[20] = 20
    correct_count: int = Field(ge=0, le=20)
    accuracy: float = Field(ge=0.0, le=1.0)
    passed: bool

    @field_validator("response_sha256")
    @classmethod
    def validate_response_hash(cls, value: str) -> str:
        return _validate_hash(value, "response_sha256")


class ReviewerQualificationEvaluationReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["passed", "failed"]
    roster_id: str
    rubric_id: Literal["craftly-review-rubric-v1"] = REVIEW_RUBRIC_ID
    roster_sha256: str
    rubric_sha256: str
    qualification_pack_sha256: str
    answer_key_sha256: str
    minimum_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_kappa: float = Field(default=0.80, ge=-1.0, le=1.0)
    reviewer_agreement_kappa: float = Field(ge=-1.0, le=1.0)
    reviewers: list[ReviewerQualificationScore]
    blockers: list[str]
    created_at_unix: float = Field(default_factory=time.time)
    handling_warning: str

    @model_validator(mode="after")
    def validate_contract(self) -> ReviewerQualificationEvaluationReport:
        for field_name in (
            "roster_sha256",
            "rubric_sha256",
            "qualification_pack_sha256",
            "answer_key_sha256",
        ):
            object.__setattr__(
                self, field_name, _validate_hash(getattr(self, field_name), field_name)
            )
        if len(self.reviewers) != 2 or len({item.reviewer_id for item in self.reviewers}) != 2:
            raise ValueError("qualification evaluation requires exactly two distinct reviewers")
        expected_status = "failed" if self.blockers else "passed"
        if self.status != expected_status:
            raise ValueError(f"qualification evaluation status must be {expected_status!r}")
        if not self.blockers:
            if any(not item.passed for item in self.reviewers):
                raise ValueError("passing qualification report contains a failed reviewer")
            if self.reviewer_agreement_kappa < self.minimum_kappa:
                raise ValueError("passing qualification report is below the kappa threshold")
        return self


def build_reviewer_rubric_markdown() -> str:
    return """# Craftly Reviewer Rubric v1

Rubric ID: `craftly-review-rubric-v1`

This rubric governs the independent review of coding, debugging, and defensive-security training rows.
It does not replace source-license approval, automated contamination checks, or verification evidence.

## Approve only when every condition is satisfied

- Content matches its bound source snapshot and provenance.
- Content is relevant to coding, debugging, or defensive-security learning.
- Content is technically coherent and sufficiently complete to teach from without hidden context.
- Content is not boilerplate, navigation, advertising, or duplicated filler.
- Content contains no credential, secret, private key, or disallowed personal information.
- Content is not a protected benchmark prompt, answer, patch, or solution.
- Content does not provide live-target attack, persistence, credential theft, or unbounded malware workflow.
- Code examples have a clear intent and do not recommend an obvious insecure pattern.
- Patch examples bind vulnerable-before, changed-after, and verification evidence.
- CVE, CWE, compiler, test, and security-scan claims are supported by bound evidence.

## Reject when any condition applies

- `incorrect_or_misleading`
- `low_signal_or_boilerplate`
- `incomplete_context`
- `duplicate_or_near_duplicate`
- `missing_provenance`
- `secret_or_pii`
- `protected_benchmark`
- `unsafe_live_target_content`
- `fabricated_verification`
- `unverified_patch`
- `license_scope_mismatch`

## Decision rationale

Every decision must contain at least one concrete sentence tied to the reviewed row.

Approve example:

> Contains a complete parameterized-query example with clear defensive rationale and no unsupported verification claim.

Reject example:

> Claims regression tests passed but provides no bound test execution or verification manifest.

Do not submit only a reason code, a generic phrase such as "looks good", or another reviewer's rationale.

## Independence and adjudication

- Two reviewers use this same rubric and submit decisions independently.
- Neither reviewer sees the other decision before both decisions are submitted.
- A disagreement becomes a conflict; reviewers do not rewrite decisions to manufacture agreement.
- An independent adjudicator reviews the row and evidence under this rubric.
- Rubric wording may be clarified before production review starts. Once a calibration round starts, its rubric hash is fixed.

## Qualification targets

- Policy-floor Cohen's kappa: `>= 0.80`
- Stretch-target Cohen's kappa: `>= 0.85`
- Policy-floor sampled acceptance rate: `>= 0.95`
- Stretch-target sampled acceptance rate: `>= 0.97`

The stretch target never replaces the policy floor, and thresholds are never lowered after observing results.
"""


def build_reviewer_roster_onboarding_template(
    *,
    roster_id: str = "craftly-data-reviewers-v1",
    final_roster_path: str = "config/data_reviewers.json",
) -> ReviewerRosterOnboardingTemplate:
    required_fields = [
        "reviewer_id",
        "identity_provider_subject",
        "roles",
        "active",
        "approved_by",
        "approved_at",
    ]
    return ReviewerRosterOnboardingTemplate(
        roster_id=roster_id,
        required_slots=[
            ReviewerRosterSlot(
                slot="reviewer_1",
                required_role="reviewer",
                required_fields=required_fields,
            ),
            ReviewerRosterSlot(
                slot="reviewer_2",
                required_role="reviewer",
                required_fields=required_fields,
            ),
            ReviewerRosterSlot(
                slot="adjudicator",
                required_role="adjudicator",
                required_fields=required_fields,
            ),
        ],
        final_roster_path=final_roster_path,
        warning=(
            "This template contains no identities and cannot unlock calibration. "
            "A governance operator must register two independent reviewers and one independent adjudicator."
        ),
    )


def reviewer_roster_readiness(roster: ReviewerRoster) -> ReviewerRosterReadinessReport:
    active_reviewers = {
        identity.identity_provider_subject
        for identity in roster.identities
        if identity.active and "reviewer" in identity.roles
    }
    active_adjudicators = {
        identity.identity_provider_subject
        for identity in roster.identities
        if identity.active and "adjudicator" in identity.roles
    }
    blockers = roster.readiness_blockers()
    for identity in roster.identities:
        identity_values = {
            "reviewer_id": identity.reviewer_id,
            "identity_provider_subject": identity.identity_provider_subject,
            "approved_by": identity.approved_by,
        }
        for field_name, value in identity_values.items():
            normalized = value.strip().lower()
            if any(marker in normalized for marker in PLACEHOLDER_MARKERS):
                blockers.append(
                    f"{identity.reviewer_id}: {field_name} contains a placeholder marker and is not a real identity"
                )
    return ReviewerRosterReadinessReport(
        status="blocked" if blockers else "ready",
        roster_id=roster.roster_id,
        active_reviewer_count=len(active_reviewers),
        active_adjudicator_count=len(active_adjudicators),
        independent_subject_count=len(active_reviewers | active_adjudicators),
        blockers=blockers,
    )


def initialize_reviewer_governance_bundle(
    output_dir: str | Path,
    *,
    roster_id: str = "craftly-data-reviewers-v1",
) -> ReviewerGovernanceBundleReport:
    root = Path(output_dir).expanduser().resolve()
    roster_path = root / "data_reviewers.json"
    rubric_path = root / "reviewer-rubric-v1.md"
    onboarding_path = root / "reviewer-onboarding.json"
    manifest_path = root / "reviewer-governance-manifest.json"
    protected_paths = (roster_path, rubric_path, onboarding_path, manifest_path)
    existing = [str(path) for path in protected_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite reviewer governance artifacts: " + ", ".join(existing)
        )

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    rubric = build_reviewer_rubric_markdown()
    empty_roster = ReviewerRoster(roster_id=roster_id, identities=[])
    onboarding = build_reviewer_roster_onboarding_template(
        roster_id=roster_id,
        final_roster_path=str(roster_path),
    )
    _write_text_atomically(rubric_path, rubric)
    _write_model_atomically(empty_roster, roster_path)
    _write_model_atomically(onboarding, onboarding_path)
    for path in (roster_path, rubric_path, onboarding_path):
        os.chmod(path, 0o600)
    os.chmod(root, 0o700)

    report = ReviewerGovernanceBundleReport(
        roster_id=roster_id,
        rubric_id=REVIEW_RUBRIC_ID,
        output_dir=str(root),
        roster_path=str(roster_path),
        rubric_path=str(rubric_path),
        onboarding_path=str(onboarding_path),
        roster_sha256=_hash_file(roster_path),
        rubric_sha256=_hash_file(rubric_path),
        warning=(
            "The roster is intentionally empty and cannot unlock calibration. "
            "A governance operator must add two real independent OIDC reviewer subjects "
            "and one real independent adjudicator subject, then run roster validation."
        ),
    )
    _write_model_atomically(report, manifest_path)
    os.chmod(manifest_path, 0o600)
    return report


def _reviewer_qualification_cases() -> list[
    tuple[ReviewerQualificationItem, ReviewerQualificationAnswer]
]:
    cases = [
        (
            "security_document",
            "Parameterized SQL guidance",
            "Explains placeholders and parameter binding with a complete defensive example.",
            "Official source snapshot and license evidence are bound.",
            "approve",
            None,
            "The guidance is complete, defensive, technically coherent, and supported by provenance.",
        ),
        (
            "code_example",
            "Canonical path containment",
            "Resolves a requested path and verifies containment beneath a fixed project root.",
            "Unit tests cover traversal and valid nested paths.",
            "approve",
            None,
            "The example has a clear secure intent and includes relevant positive and negative tests.",
        ),
        (
            "security_document",
            "Password storage guidance",
            "Documents memory-hard password hashing, salts, and parameter migration.",
            "Official defensive documentation snapshot is bound.",
            "approve",
            None,
            "The row is focused, technically useful, and contains no unsupported security claim.",
        ),
        (
            "code_example",
            "Safe process invocation",
            "Uses an argument vector, fixed executable allowlist, bounded timeout, and a reduced environment.",
            "Tests cover rejected shell metacharacters and timeout behavior.",
            "approve",
            None,
            "The process example is complete and demonstrates bounded defensive execution.",
        ),
        (
            "code_example",
            "Checked integer arithmetic",
            "Uses checked arithmetic and handles overflow explicitly in a systems-language example.",
            "Compiler and unit-test evidence are bound.",
            "approve",
            None,
            "The code and evidence form a coherent secure-coding example with explicit failure handling.",
        ),
        (
            "code_example",
            "HTTP client resource bounds",
            "Configures connect and response timeouts, response-size limits, and cancellation.",
            "Tests use an isolated local server and contain no live target.",
            "approve",
            None,
            "The example teaches bounded network behavior with isolated verification evidence.",
        ),
        (
            "security_document",
            "Container least privilege",
            "Explains non-root execution, dropped capabilities, read-only filesystems, and temporary storage.",
            "Official container documentation snapshot is bound.",
            "approve",
            None,
            "The content is structured, defensive, and sufficiently complete for training.",
        ),
        (
            "security_document",
            "Restricted workload policy",
            "Explains restricted pod security controls and provides a valid declarative example.",
            "Schema validation and official documentation provenance are bound.",
            "approve",
            None,
            "The row combines authoritative provenance, valid configuration, and defensive rationale.",
        ),
        (
            "patch",
            "Verified input-validation patch",
            "Changes validation logic and includes the vulnerable-before and corrected-after diff.",
            "The original security test fails, then build, security, and regression tests pass after the patch.",
            "approve",
            None,
            "The patch has complete before-and-after evidence and makes no unsupported verification claim.",
        ),
        (
            "debug_trace",
            "Reproducible authentication regression",
            "Links a traceback to a minimal failing test and a bounded corrective change.",
            "The failing and passing test runs are bound to the same repository snapshot.",
            "approve",
            None,
            "The debugging record is reproducible, evidence-grounded, and useful without hidden context.",
        ),
        (
            "security_document",
            "Incorrect cryptographic claim",
            "Claims that an obsolete fast hash is appropriate for password storage.",
            "No authoritative evidence supports the claim.",
            "reject",
            "incorrect_or_misleading",
            "The central security recommendation is technically incorrect and would train unsafe behavior.",
        ),
        (
            "security_document",
            "Navigation-only page",
            "Contains menus, footer text, cookie controls, and no substantive technical content.",
            "A source URL exists but no meaningful body content remains.",
            "reject",
            "low_signal_or_boilerplate",
            "The candidate is boilerplate-heavy and contains no useful standalone training signal.",
        ),
        (
            "code_example",
            "Truncated validation helper",
            "Shows a partial function ending before error handling or return behavior is defined.",
            "No tests or surrounding call context are present.",
            "reject",
            "incomplete_context",
            "The incomplete function cannot be interpreted or verified as a standalone example.",
        ),
        (
            "security_document",
            "Near-identical duplicate",
            "Repeats a previously accepted paragraph with only heading and whitespace changes.",
            "Normalized-content and MinHash evidence identify the earlier record.",
            "reject",
            "duplicate_or_near_duplicate",
            "The row adds no new signal and is a verified near duplicate of an accepted record.",
        ),
        (
            "code_example",
            "Unknown repository origin",
            "Presents a substantial code sample with no source repository, revision, or license binding.",
            "Provenance fields and source snapshot are absent.",
            "reject",
            "missing_provenance",
            "The code may be useful but cannot enter governed training without complete provenance.",
        ),
        (
            "debug_trace",
            "Credential-bearing log",
            "Contains a captured production log with an unredacted authentication credential.",
            "Secret scanning flags the credential class.",
            "reject",
            "secret_or_pii",
            "The record contains disallowed sensitive material and must be rejected rather than normalized.",
        ),
        (
            "code_example",
            "Protected evaluation solution",
            "Contains a solution associated with a protected benchmark task identifier.",
            "The holdout fingerprint index reports an exact protected match.",
            "reject",
            "protected_benchmark",
            "Protected benchmark material is absolutely excluded from training regardless of apparent quality.",
        ),
        (
            "security_document",
            "Live-target intrusion workflow",
            "Describes operational steps against a third-party system without authorization or isolation.",
            "No lab authorization, synthetic environment, or defensive framing is bound.",
            "reject",
            "unsafe_live_target_content",
            "The content is an unbounded live-target workflow outside the governed defensive scope.",
        ),
        (
            "patch",
            "Claimed successful patch",
            "States that compilation, security tests, and regression tests passed.",
            "No command output, test manifest, repository hash, or scanner result is attached.",
            "reject",
            "fabricated_verification",
            "A text claim cannot substitute for immutable verification evidence.",
        ),
        (
            "patch",
            "Unverified patch with scope mismatch",
            "Provides a plausible security diff sourced under an unapproved training scope.",
            "Before-and-after tests are absent and legal scope is evaluation-only.",
            "reject",
            "unverified_patch",
            "The patch lacks verification and its source is not approved for training collection.",
        ),
    ]
    return [
        (
            ReviewerQualificationItem(
                item_id=f"reviewer-qualification-{index:02d}",
                data_type=data_type,
                title=title,
                candidate_summary=candidate,
                evidence_summary=evidence,
            ),
            ReviewerQualificationAnswer(
                item_id=f"reviewer-qualification-{index:02d}",
                expected_decision=decision,
                reason_code=reason_code,
                rationale=rationale,
            ),
        )
        for index, (
            data_type,
            title,
            candidate,
            evidence,
            decision,
            reason_code,
            rationale,
        ) in enumerate(
            cases,
            start=1,
        )
    ]


def initialize_reviewer_qualification_pack(
    output_dir: str | Path,
) -> ReviewerQualificationPackReport:
    root = Path(output_dir).expanduser().resolve()
    pack_path = root / "reviewer-qualification-v1.jsonl"
    answer_key_path = root / "reviewer-qualification-answer-key-v1.json"
    manifest_path = root / "reviewer-qualification-manifest.json"
    existing = [str(path) for path in (pack_path, answer_key_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite reviewer qualification artifacts: " + ", ".join(existing)
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cases = _reviewer_qualification_cases()
    items = [item for item, _ in cases]
    answers = [answer for _, answer in cases]
    if len(items) != 20:
        raise RuntimeError("reviewer qualification pack must contain exactly 20 items")
    approve_count = sum(answer.expected_decision == "approve" for answer in answers)
    reject_count = sum(answer.expected_decision == "reject" for answer in answers)
    if (approve_count, reject_count) != (10, 10):
        raise RuntimeError(
            "reviewer qualification answer key must contain ten approvals and ten rejections"
        )
    _write_text_atomically(
        pack_path,
        "".join(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n" for item in items),
    )
    _write_text_atomically(
        answer_key_path,
        json.dumps(
            {
                "schema_version": 1,
                "rubric_id": REVIEW_RUBRIC_ID,
                "answers": [answer.model_dump(mode="json") for answer in answers],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    for path in (pack_path, answer_key_path):
        os.chmod(path, 0o600)
    report = ReviewerQualificationPackReport(
        pack_path=str(pack_path),
        answer_key_path=str(answer_key_path),
        pack_sha256=_hash_file(pack_path),
        answer_key_sha256=_hash_file(answer_key_path),
        handling_warning=(
            "Give reviewers only the qualification pack. Keep the answer key restricted until both "
            "independent decisions are submitted; qualification records never enter training."
        ),
    )
    _write_model_atomically(report, manifest_path)
    os.chmod(manifest_path, 0o600)
    return report


def finalize_reviewer_governance_bundle(output_dir: str | Path) -> ReviewerGovernanceBundleReport:
    root = Path(output_dir).expanduser().resolve(strict=True)
    roster_path = root / "data_reviewers.json"
    rubric_path = root / "reviewer-rubric-v1.md"
    onboarding_path = root / "reviewer-onboarding.json"
    manifest_path = root / "reviewer-governance-manifest.json"
    for required_path in (roster_path, rubric_path, onboarding_path, manifest_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"reviewer governance artifact is missing: {required_path}")

    roster = load_reviewer_roster(roster_path)
    readiness = reviewer_roster_readiness(roster)
    if readiness.status != "ready":
        raise ValueError("reviewer roster is not ready: " + "; ".join(readiness.blockers))
    expected_rubric = build_reviewer_rubric_markdown()
    if rubric_path.read_text(encoding="utf-8") != expected_rubric:
        raise ValueError("reviewer rubric does not match the code-governed v1 rubric")

    report = ReviewerGovernanceBundleReport(
        status="ready_for_qualification",
        roster_id=roster.roster_id,
        rubric_id=REVIEW_RUBRIC_ID,
        output_dir=str(root),
        roster_path=str(roster_path),
        rubric_path=str(rubric_path),
        onboarding_path=str(onboarding_path),
        roster_sha256=_hash_file(roster_path),
        rubric_sha256=_hash_file(rubric_path),
        warning=(
            "The identity roster is ready. Phase 0 remains incomplete until both reviewers finish "
            "the blind qualification pack and the qualification result is evaluated."
        ),
    )
    _write_model_atomically(report, manifest_path)
    os.chmod(manifest_path, 0o600)
    return report


def prepare_reviewer_delivery_packages(
    governance_dir: str | Path,
    output_dir: str | Path,
) -> ReviewerDeliveryReport:
    governance_root = Path(governance_dir).expanduser().resolve(strict=True)
    output_root = Path(output_dir).expanduser().resolve()
    governance_manifest_path = governance_root / "reviewer-governance-manifest.json"
    qualification_manifest_path = governance_root / "reviewer-qualification-manifest.json"
    governance = ReviewerGovernanceBundleReport.model_validate_json(
        governance_manifest_path.read_text(encoding="utf-8")
    )
    qualification = ReviewerQualificationPackReport.model_validate_json(
        qualification_manifest_path.read_text(encoding="utf-8")
    )
    if governance.status != "ready_for_qualification":
        raise ValueError("reviewer governance manifest must be finalized before delivery")

    roster_path = governance_root / "data_reviewers.json"
    rubric_path = governance_root / "reviewer-rubric-v1.md"
    pack_path = governance_root / "reviewer-qualification-v1.jsonl"
    if _hash_file(roster_path) != governance.roster_sha256:
        raise ValueError("reviewer roster changed after governance finalization")
    if _hash_file(rubric_path) != governance.rubric_sha256:
        raise ValueError("reviewer rubric changed after governance finalization")
    if _hash_file(pack_path) != qualification.pack_sha256:
        raise ValueError("reviewer qualification pack changed after initialization")

    roster = load_reviewer_roster(roster_path)
    readiness = reviewer_roster_readiness(roster)
    if readiness.status != "ready":
        raise ValueError("reviewer roster is no longer ready: " + "; ".join(readiness.blockers))
    reviewers = sorted(
        (
            identity
            for identity in roster.identities
            if identity.active and "reviewer" in identity.roles
        ),
        key=lambda identity: identity.reviewer_id,
    )
    if len(reviewers) != 2:
        raise ValueError("delivery requires exactly two active reviewer identities")

    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = output_root / "reviewer-delivery-report.json"
    targets = [output_root / f"{reviewer.reviewer_id}-qualification.zip" for reviewer in reviewers]
    existing = [str(path) for path in [*targets, report_path] if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite reviewer delivery artifacts: " + ", ".join(existing)
        )

    pack_rows = [
        json.loads(line)
        for line in pack_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(pack_rows) != 20:
        raise ValueError("reviewer qualification pack must contain exactly twenty rows")
    packages: list[ReviewerDeliveryArtifact] = []
    for reviewer, target in zip(reviewers, targets, strict=True):
        responses = "".join(
            json.dumps(
                {
                    "item_id": row["item_id"],
                    "reviewer_id": reviewer.reviewer_id,
                    "decision": None,
                    "rationale": None,
                },
                sort_keys=True,
            )
            + "\n"
            for row in pack_rows
        )
        package_manifest = {
            "schema_version": 1,
            "reviewer_id": reviewer.reviewer_id,
            "rubric_id": REVIEW_RUBRIC_ID,
            "rubric_sha256": governance.rubric_sha256,
            "qualification_pack_sha256": qualification.pack_sha256,
            "item_count": 20,
            "answer_key_included": False,
            "instructions": [
                "Read reviewer-rubric-v1.md before reviewing.",
                "Complete every decision and write a concrete one-sentence rationale.",
                "Do not discuss decisions with the other reviewer before both submissions are complete.",
                "Return only reviewer-responses.jsonl through the approved secure channel.",
            ],
        }
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                _write_deterministic_zip_entry(
                    archive,
                    "reviewer-rubric-v1.md",
                    rubric_path.read_bytes(),
                )
                _write_deterministic_zip_entry(
                    archive,
                    "reviewer-qualification-v1.jsonl",
                    pack_path.read_bytes(),
                )
                _write_deterministic_zip_entry(
                    archive,
                    "reviewer-responses.jsonl",
                    responses.encode("utf-8"),
                )
                _write_deterministic_zip_entry(
                    archive,
                    "delivery-manifest.json",
                    (json.dumps(package_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        os.chmod(target, 0o600)
        with zipfile.ZipFile(target, mode="r") as archive:
            names = set(archive.namelist())
        if "reviewer-qualification-answer-key-v1.json" in names or len(names) != 4:
            raise RuntimeError("reviewer delivery package content contract was violated")
        packages.append(
            ReviewerDeliveryArtifact(
                reviewer_id=reviewer.reviewer_id,
                package_path=str(target),
                package_sha256=_hash_file(target),
            )
        )

    report = ReviewerDeliveryReport(
        roster_id=roster.roster_id,
        rubric_sha256=governance.rubric_sha256,
        qualification_pack_sha256=qualification.pack_sha256,
        packages=packages,
        warning=(
            "Packages contain no answer key or OIDC subject. Deliver each package only to its named "
            "reviewer through an authenticated secure channel."
        ),
    )
    _write_model_atomically(report, report_path)
    os.chmod(report_path, 0o600)
    os.chmod(output_root, 0o700)
    return report


def evaluate_reviewer_qualification(
    *,
    governance_dir: str | Path,
    response_paths: list[str | Path],
    output_dir: str | Path,
    minimum_accuracy: float = 0.95,
    minimum_kappa: float = 0.80,
) -> ReviewerQualificationEvaluationReport:
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be between 0 and 1")
    if not -1.0 <= minimum_kappa <= 1.0:
        raise ValueError("minimum_kappa must be between -1 and 1")
    if len(response_paths) != 2:
        raise ValueError("qualification evaluation requires exactly two response files")

    root = Path(governance_dir).expanduser().resolve(strict=True)
    governance_path = root / "reviewer-governance-manifest.json"
    qualification_path = root / "reviewer-qualification-manifest.json"
    roster_path = root / "data_reviewers.json"
    rubric_path = root / "reviewer-rubric-v1.md"
    pack_path = root / "reviewer-qualification-v1.jsonl"
    answer_key_path = root / "reviewer-qualification-answer-key-v1.json"
    for required_path in (
        governance_path,
        qualification_path,
        roster_path,
        rubric_path,
        pack_path,
        answer_key_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"qualification governance artifact is missing: {required_path}"
            )

    governance = ReviewerGovernanceBundleReport.model_validate_json(
        governance_path.read_text(encoding="utf-8")
    )
    qualification = ReviewerQualificationPackReport.model_validate_json(
        qualification_path.read_text(encoding="utf-8")
    )
    if governance.status != "ready_for_qualification":
        raise ValueError("reviewer governance is not finalized")
    integrity_checks = {
        "reviewer roster": (_hash_file(roster_path), governance.roster_sha256),
        "reviewer rubric": (_hash_file(rubric_path), governance.rubric_sha256),
        "qualification pack": (_hash_file(pack_path), qualification.pack_sha256),
        "qualification answer key": (_hash_file(answer_key_path), qualification.answer_key_sha256),
    }
    changed = [name for name, (actual, expected) in integrity_checks.items() if actual != expected]
    if changed:
        raise ValueError(
            "qualification governance artifact changed after binding: " + ", ".join(changed)
        )

    roster = load_reviewer_roster(roster_path)
    readiness = reviewer_roster_readiness(roster)
    if readiness.status != "ready":
        raise ValueError("reviewer roster is no longer ready: " + "; ".join(readiness.blockers))
    expected_reviewers = {
        identity.reviewer_id
        for identity in roster.identities
        if identity.active and "reviewer" in identity.roles
    }
    if len(expected_reviewers) != 2:
        raise ValueError("qualification requires exactly two active reviewers")

    pack_items = [
        ReviewerQualificationItem.model_validate_json(line)
        for line in pack_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_item_ids = {item.item_id for item in pack_items}
    if len(pack_items) != 20 or len(expected_item_ids) != 20:
        raise ValueError("qualification pack must contain twenty unique items")

    answer_payload = json.loads(answer_key_path.read_text(encoding="utf-8"))
    if (
        answer_payload.get("schema_version") != 1
        or answer_payload.get("rubric_id") != REVIEW_RUBRIC_ID
    ):
        raise ValueError("qualification answer key contract is invalid")
    answers = [
        ReviewerQualificationAnswer.model_validate(item)
        for item in answer_payload.get("answers", [])
    ]
    answer_by_id = {item.item_id: item for item in answers}
    if len(answers) != 20 or set(answer_by_id) != expected_item_ids:
        raise ValueError("qualification answer key does not match the governed pack")

    submissions: dict[str, tuple[Path, dict[str, ReviewerQualificationResponse]]] = {}
    resolved_paths: set[Path] = set()
    for raw_path in response_paths:
        response_path = Path(raw_path).expanduser().resolve(strict=True)
        if response_path in resolved_paths:
            raise ValueError("qualification response files must be distinct")
        resolved_paths.add(response_path)
        if not response_path.is_file() or response_path.stat().st_size > 1_048_576:
            raise ValueError(
                f"qualification response must be a regular file no larger than 1 MiB: {response_path}"
            )
        rows: list[ReviewerQualificationResponse] = []
        for line_number, line in enumerate(
            response_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                rows.append(ReviewerQualificationResponse.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid qualification response in {response_path} at line {line_number}: {exc}"
                ) from exc
        if len(rows) != 20:
            raise ValueError(f"{response_path} must contain exactly twenty completed responses")
        reviewer_ids = {row.reviewer_id for row in rows}
        if len(reviewer_ids) != 1:
            raise ValueError(f"{response_path} mixes reviewer identities")
        reviewer_id = next(iter(reviewer_ids))
        if reviewer_id not in expected_reviewers:
            raise PermissionError(f"{reviewer_id!r} is not an active governed reviewer")
        if reviewer_id in submissions:
            raise ValueError(f"duplicate qualification submission for reviewer {reviewer_id!r}")
        by_item = {row.item_id: row for row in rows}
        if len(by_item) != 20 or set(by_item) != expected_item_ids:
            raise ValueError(
                f"{response_path} contains duplicate, missing, or unexpected qualification items"
            )
        submissions[reviewer_id] = (response_path, by_item)
    if set(submissions) != expected_reviewers:
        raise ValueError("qualification submissions do not cover both active reviewers")

    scores: list[ReviewerQualificationScore] = []
    for reviewer_id in sorted(submissions):
        response_path, by_item = submissions[reviewer_id]
        correct = sum(
            by_item[item_id].decision == answer_by_id[item_id].expected_decision
            for item_id in sorted(expected_item_ids)
        )
        accuracy = correct / 20
        scores.append(
            ReviewerQualificationScore(
                reviewer_id=reviewer_id,
                response_sha256=_hash_file(response_path),
                correct_count=correct,
                accuracy=round(accuracy, 6),
                passed=accuracy >= minimum_accuracy,
            )
        )

    reviewer_ids = sorted(submissions)
    pairs = [
        (
            submissions[reviewer_ids[0]][1][item_id].decision,
            submissions[reviewer_ids[1]][1][item_id].decision,
        )
        for item_id in sorted(expected_item_ids)
    ]
    kappa = round(_cohen_kappa(pairs), 6)
    blockers = [
        f"{score.reviewer_id} accuracy {score.accuracy:.4f} is below {minimum_accuracy:.4f}"
        for score in scores
        if not score.passed
    ]
    if kappa < minimum_kappa:
        blockers.append(f"reviewer agreement kappa {kappa:.4f} is below {minimum_kappa:.4f}")

    report = ReviewerQualificationEvaluationReport(
        status="failed" if blockers else "passed",
        roster_id=roster.roster_id,
        roster_sha256=governance.roster_sha256,
        rubric_sha256=governance.rubric_sha256,
        qualification_pack_sha256=qualification.pack_sha256,
        answer_key_sha256=qualification.answer_key_sha256,
        minimum_accuracy=minimum_accuracy,
        minimum_kappa=minimum_kappa,
        reviewer_agreement_kappa=kappa,
        reviewers=scores,
        blockers=blockers,
        handling_warning=(
            "This report contains aggregate scores and hashes only. Reviewer rationales, decisions, "
            "OIDC subjects, and answer-key content remain restricted governance material."
        ),
    )
    output_root = Path(output_dir).expanduser().resolve()
    report_path = output_root / "reviewer-qualification-evaluation.json"
    markdown_path = output_root / "reviewer-qualification-evaluation.md"
    existing = [str(path) for path in (report_path, markdown_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite qualification evaluation artifacts: " + ", ".join(existing)
        )
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_model_atomically(report, report_path)
    lines = [
        "# Craftly Reviewer Qualification Evaluation",
        "",
        f"- Status: `{report.status}`",
        f"- Roster: `{report.roster_id}`",
        f"- Minimum accuracy: `{report.minimum_accuracy:.2%}`",
        f"- Minimum Cohen's kappa: `{report.minimum_kappa:.4f}`",
        f"- Measured Cohen's kappa: `{report.reviewer_agreement_kappa:.4f}`",
        "",
        "| Reviewer | Correct | Accuracy | Passed | Response SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for score in report.reviewers:
        lines.append(
            f"| {score.reviewer_id} | {score.correct_count}/20 | {score.accuracy:.2%} | "
            f"{score.passed} | `{score.response_sha256}` |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {blocker}" for blocker in report.blockers)
    if not report.blockers:
        lines.append("- None")
    lines.extend(["", report.handling_warning, ""])
    _write_text_atomically(markdown_path, "\n".join(lines))
    for path in (report_path, markdown_path):
        os.chmod(path, 0o600)
    os.chmod(output_root, 0o700)
    return report


def load_reviewer_roster(path: str | Path) -> ReviewerRoster:
    return ReviewerRoster.model_validate_json(Path(path).read_text(encoding="utf-8"))


class DatasetAuthorityStore(Protocol):
    async def record_source_snapshot(self, request: SourceSnapshotCreate) -> SourceSnapshotView: ...
    async def enqueue(self, request: ReviewItemCreate) -> ReviewItemView: ...
    async def list_items(
        self, *, status: str | None, limit: int, reviewer_id: str | None
    ) -> list[ReviewItemView]: ...
    async def get_item(self, review_item_id: str, *, reviewer_id: str | None) -> ReviewItemView: ...
    async def claim(
        self, review_item_id: str, reviewer_id: str, lease_seconds: int
    ) -> ReviewAssignment: ...
    async def decide(
        self, review_item_id: str, reviewer_id: str, request: ReviewDecisionCreate
    ) -> ReviewItemView: ...
    async def adjudicate(
        self, review_item_id: str, adjudicator_id: str, request: ReviewAdjudicationCreate
    ) -> ReviewItemView: ...
    async def record_promotion(
        self, actor_id: str, request: PromotionEvidenceCreate
    ) -> PromotionEvidenceView: ...
    async def review_evidence(self) -> ReviewEvidenceReport: ...


def _terminal_or_visible(status: str, reviewer_id: str | None, decision_reviewer: str) -> bool:
    return (
        status in {"approved", "rejected", "conflict", "adjudication_required"}
        or reviewer_id == decision_reviewer
    )


def _status_from_decisions(high_value: bool, decisions: list[str]) -> str:
    required = 2 if high_value else 1
    if len(decisions) < required:
        return "in_review"
    unique = set(decisions)
    if len(unique) > 1:
        return "adjudication_required"
    return "approved" if decisions[0] == "approve" else "rejected"


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    observed = sum(left == right for left, right in pairs) / len(pairs)
    first_approve = sum(left == "approve" for left, _ in pairs) / len(pairs)
    second_approve = sum(right == "approve" for _, right in pairs) / len(pairs)
    expected = first_approve * second_approve + (1 - first_approve) * (1 - second_approve)
    return 1.0 if expected >= 1.0 else max(-1.0, min(1.0, (observed - expected) / (1 - expected)))


def _review_evidence_from_rows(
    items: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> ReviewEvidenceReport:
    decisions_by_item: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        decisions_by_item.setdefault(str(decision["review_item_id"]), []).append(decision)
    accumulators: dict[str, dict[str, Any]] = {}
    for item in items:
        source = str(item["source_id"])
        bucket = accumulators.setdefault(
            source,
            {"approved": 0, "rejected": 0, "pending": 0, "pairs": []},
        )
        status = str(item["status"])
        if status == "approved":
            bucket["approved"] += 1
        elif status == "rejected":
            bucket["rejected"] += 1
        else:
            bucket["pending"] += 1
        item_decisions = sorted(
            decisions_by_item.get(str(item["id"]), []),
            key=lambda row: float(row["created_at"]),
        )
        if len(item_decisions) >= 2:
            bucket["pairs"].append(
                (str(item_decisions[0]["decision"]), str(item_decisions[1]["decision"]))
            )
    by_source: dict[str, SourceReviewEvidence] = {}
    for source, bucket in sorted(accumulators.items()):
        total_terminal = int(bucket["approved"]) + int(bucket["rejected"])
        pairs = list(bucket["pairs"])
        by_source[source] = SourceReviewEvidence(
            approved=int(bucket["approved"]),
            rejected=int(bucket["rejected"]),
            pending=int(bucket["pending"]),
            paired_reviews=len(pairs),
            reviewer_agreement_kappa=round(_cohen_kappa(pairs), 6),
            acceptance_rate=round(int(bucket["approved"]) / max(1, total_terminal), 6),
        )
    approved = sum(item.approved for item in by_source.values())
    rejected = sum(item.rejected for item in by_source.values())
    pending = sum(item.pending for item in by_source.values())
    total_items = approved + rejected + pending
    return ReviewEvidenceReport(
        status="empty" if total_items == 0 else ("complete" if pending == 0 else "in_progress"),
        total_items=total_items,
        decision_count=len(decisions),
        source_count=len(by_source),
        approved=approved,
        rejected=rejected,
        pending=pending,
        paired_reviews=sum(item.paired_reviews for item in by_source.values()),
        by_source=by_source,
    )


class SQLiteDatasetAuthorityStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with closing(self._connect()) as connection:
            connection.executescript(SQLITE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def record_source_snapshot(self, request: SourceSnapshotCreate) -> SourceSnapshotView:
        def operation() -> SourceSnapshotView:
            with self.lock, closing(self._connect()) as connection:
                now = time.time()
                snapshot_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT OR IGNORE INTO data_source_snapshots(
                      id,source_id,source_family,immutable_revision,registry_sha256,
                      license_evidence_sha256,legal_approval_sha256,snapshot_sha256,
                      status,metadata_json,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        snapshot_id,
                        request.source_id,
                        request.source_family,
                        request.immutable_revision,
                        request.registry_sha256,
                        request.license_evidence_sha256,
                        request.legal_approval_sha256,
                        request.snapshot_sha256,
                        request.status,
                        _json(request.metadata),
                        now,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM data_source_snapshots
                    WHERE source_id=? AND immutable_revision=? AND snapshot_sha256=?
                    """,
                    (request.source_id, request.immutable_revision, request.snapshot_sha256),
                ).fetchone()
                assert row is not None
                connection.commit()
                return SourceSnapshotView(
                    snapshot_id=row["id"],
                    source_id=row["source_id"],
                    source_family=row["source_family"],
                    immutable_revision=row["immutable_revision"],
                    registry_sha256=row["registry_sha256"],
                    license_evidence_sha256=row["license_evidence_sha256"],
                    legal_approval_sha256=row["legal_approval_sha256"],
                    snapshot_sha256=row["snapshot_sha256"],
                    status=row["status"],
                    metadata=json.loads(row["metadata_json"]),
                    created_at=float(row["created_at"]),
                )

        return await asyncio.to_thread(operation)

    def _view(
        self, connection: sqlite3.Connection, review_item_id: str, reviewer_id: str | None
    ) -> ReviewItemView:
        row = connection.execute(
            "SELECT * FROM dataset_review_items WHERE id=?", (review_item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(review_item_id)
        assignments = [
            ReviewAssignment(
                assignment_id=item["id"],
                review_item_id=item["review_item_id"],
                reviewer_id=item["reviewer_id"],
                reviewer_slot=int(item["reviewer_slot"]),
                status=item["status"],
                claimed_at=float(item["claimed_at"]),
                expires_at=float(item["expires_at"]),
            )
            for item in connection.execute(
                "SELECT * FROM dataset_review_assignments WHERE review_item_id=? ORDER BY reviewer_slot",
                (review_item_id,),
            )
        ]
        decisions = [
            ReviewDecisionView(
                reviewer_id=item["reviewer_id"],
                decision=item["decision"],
                rationale=item["rationale"],
                evidence=json.loads(item["evidence_json"]),
                created_at=float(item["created_at"]),
            )
            for item in connection.execute(
                "SELECT * FROM dataset_review_decisions WHERE review_item_id=? ORDER BY created_at",
                (review_item_id,),
            )
            if _terminal_or_visible(row["status"], reviewer_id, item["reviewer_id"])
        ]
        return ReviewItemView(
            review_item_id=row["id"],
            content_hash=row["content_hash"],
            source_snapshot_sha256=row["source_snapshot_sha256"],
            source_id=row["source_id"],
            data_type=row["data_type"],
            high_value=bool(row["high_value"]),
            status=row["status"],
            version=int(row["version"]),
            payload=json.loads(row["payload_json"]),
            assignments=assignments,
            decisions=decisions,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    async def enqueue(self, request: ReviewItemCreate) -> ReviewItemView:
        def operation() -> ReviewItemView:
            with self.lock, closing(self._connect()) as connection:
                now = time.time()
                review_item_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT OR IGNORE INTO dataset_review_items(
                      id,content_hash,source_snapshot_sha256,source_id,data_type,high_value,
                      status,version,payload_json,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,1,?,?,?)
                    """,
                    (
                        review_item_id,
                        request.content_hash,
                        request.source_snapshot_sha256,
                        request.source_id,
                        request.data_type,
                        int(request.high_value),
                        "pending",
                        _json(request.payload),
                        now,
                        now,
                    ),
                )
                existing = connection.execute(
                    "SELECT id FROM dataset_review_items WHERE content_hash=? AND source_snapshot_sha256=?",
                    (request.content_hash, request.source_snapshot_sha256),
                ).fetchone()
                assert existing is not None
                connection.commit()
                return self._view(connection, existing["id"], None)

        return await asyncio.to_thread(operation)

    async def list_items(
        self, *, status: str | None, limit: int, reviewer_id: str | None
    ) -> list[ReviewItemView]:
        if status is not None and status not in REVIEW_STATUSES:
            raise ValueError("invalid review status")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")

        def operation() -> list[ReviewItemView]:
            with self.lock, closing(self._connect()) as connection:
                if status:
                    rows = connection.execute(
                        "SELECT id FROM dataset_review_items WHERE status=? ORDER BY created_at LIMIT ?",
                        (status, limit),
                    )
                else:
                    rows = connection.execute(
                        "SELECT id FROM dataset_review_items ORDER BY created_at LIMIT ?", (limit,)
                    )
                return [self._view(connection, row["id"], reviewer_id) for row in rows]

        return await asyncio.to_thread(operation)

    async def get_item(self, review_item_id: str, *, reviewer_id: str | None) -> ReviewItemView:
        return await asyncio.to_thread(self._get_sync, review_item_id, reviewer_id)

    def _get_sync(self, review_item_id: str, reviewer_id: str | None) -> ReviewItemView:
        with self.lock, closing(self._connect()) as connection:
            return self._view(connection, review_item_id, reviewer_id)

    async def claim(
        self, review_item_id: str, reviewer_id: str, lease_seconds: int
    ) -> ReviewAssignment:
        if not reviewer_id.strip():
            raise ValueError("reviewer_id cannot be empty")
        if not 60 <= lease_seconds <= 86_400:
            raise ValueError("review lease must be between 60 and 86400 seconds")

        def operation() -> ReviewAssignment:
            with self.lock, closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                connection.execute(
                    "DELETE FROM dataset_review_assignments WHERE status='claimed' AND expires_at<?",
                    (now,),
                )
                item = connection.execute(
                    "SELECT * FROM dataset_review_items WHERE id=?", (review_item_id,)
                ).fetchone()
                if item is None:
                    raise KeyError(review_item_id)
                if item["status"] in {"approved", "rejected"}:
                    raise ValueError("terminal review item cannot be claimed")
                duplicate = connection.execute(
                    "SELECT 1 FROM dataset_review_assignments WHERE review_item_id=? AND reviewer_id=?",
                    (review_item_id, reviewer_id),
                ).fetchone()
                if duplicate:
                    raise PermissionError("same reviewer cannot occupy both review slots")
                required = 2 if bool(item["high_value"]) else 1
                occupied = {
                    int(row["reviewer_slot"])
                    for row in connection.execute(
                        "SELECT reviewer_slot FROM dataset_review_assignments WHERE review_item_id=?",
                        (review_item_id,),
                    )
                }
                slot = next(
                    (
                        candidate
                        for candidate in range(1, required + 1)
                        if candidate not in occupied
                    ),
                    None,
                )
                if slot is None:
                    raise ValueError("all review slots are already claimed")
                assignment_id = str(uuid.uuid4())
                expires_at = now + lease_seconds
                connection.execute(
                    """
                    INSERT INTO dataset_review_assignments(
                      id,review_item_id,reviewer_id,reviewer_slot,status,claimed_at,expires_at
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (assignment_id, review_item_id, reviewer_id, slot, "claimed", now, expires_at),
                )
                connection.execute(
                    "UPDATE dataset_review_items SET status='in_review',version=version+1,updated_at=? WHERE id=?",
                    (now, review_item_id),
                )
                connection.commit()
                return ReviewAssignment(
                    assignment_id=assignment_id,
                    review_item_id=review_item_id,
                    reviewer_id=reviewer_id,
                    reviewer_slot=slot,
                    status="claimed",
                    claimed_at=now,
                    expires_at=expires_at,
                )

        return await asyncio.to_thread(operation)

    async def decide(
        self, review_item_id: str, reviewer_id: str, request: ReviewDecisionCreate
    ) -> ReviewItemView:
        def operation() -> ReviewItemView:
            with self.lock, closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                item = connection.execute(
                    "SELECT * FROM dataset_review_items WHERE id=?", (review_item_id,)
                ).fetchone()
                if item is None:
                    raise KeyError(review_item_id)
                if (
                    item["content_hash"] != request.content_hash
                    or item["source_snapshot_sha256"] != request.source_snapshot_sha256
                ):
                    raise ValueError("review evidence no longer matches content/source snapshot")
                assignment = connection.execute(
                    """
                    SELECT * FROM dataset_review_assignments
                    WHERE review_item_id=? AND reviewer_id=? AND status='claimed'
                    """,
                    (review_item_id, reviewer_id),
                ).fetchone()
                if assignment is None:
                    raise PermissionError("reviewer does not hold this review item")
                if float(assignment["expires_at"]) < now:
                    raise PermissionError("review assignment has expired")
                connection.execute(
                    """
                    INSERT INTO dataset_review_decisions(
                      id,review_item_id,reviewer_id,decision,rationale,content_hash,
                      source_snapshot_sha256,evidence_json,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(uuid.uuid4()),
                        review_item_id,
                        reviewer_id,
                        request.decision,
                        request.rationale,
                        request.content_hash,
                        request.source_snapshot_sha256,
                        _json(request.evidence),
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE dataset_review_assignments SET status='submitted' WHERE id=?",
                    (assignment["id"],),
                )
                decisions = [
                    row["decision"]
                    for row in connection.execute(
                        "SELECT decision FROM dataset_review_decisions WHERE review_item_id=? ORDER BY created_at",
                        (review_item_id,),
                    )
                ]
                status = _status_from_decisions(bool(item["high_value"]), decisions)
                connection.execute(
                    "UPDATE dataset_review_items SET status=?,version=version+1,updated_at=? WHERE id=?",
                    (status, now, review_item_id),
                )
                connection.commit()
                return self._view(connection, review_item_id, reviewer_id)

        return await asyncio.to_thread(operation)

    async def adjudicate(
        self, review_item_id: str, adjudicator_id: str, request: ReviewAdjudicationCreate
    ) -> ReviewItemView:
        def operation() -> ReviewItemView:
            with self.lock, closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                item = connection.execute(
                    "SELECT * FROM dataset_review_items WHERE id=?", (review_item_id,)
                ).fetchone()
                if item is None:
                    raise KeyError(review_item_id)
                if item["status"] not in {"conflict", "adjudication_required"}:
                    raise ValueError("only conflicting reviews can be adjudicated")
                reviewer = connection.execute(
                    "SELECT 1 FROM dataset_review_decisions WHERE review_item_id=? AND reviewer_id=?",
                    (review_item_id, adjudicator_id),
                ).fetchone()
                if reviewer:
                    raise PermissionError("a reviewer cannot adjudicate the same record")
                now = time.time()
                connection.execute(
                    """
                    INSERT INTO dataset_review_adjudications(
                      id,review_item_id,adjudicator_id,decision,rationale,evidence_json,created_at
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        str(uuid.uuid4()),
                        review_item_id,
                        adjudicator_id,
                        request.decision,
                        request.rationale,
                        _json(request.evidence),
                        now,
                    ),
                )
                status = "approved" if request.decision == "approve" else "rejected"
                connection.execute(
                    "UPDATE dataset_review_items SET status=?,version=version+1,updated_at=? WHERE id=?",
                    (status, now, review_item_id),
                )
                connection.commit()
                return self._view(connection, review_item_id, adjudicator_id)

        return await asyncio.to_thread(operation)

    async def record_promotion(
        self, actor_id: str, request: PromotionEvidenceCreate
    ) -> PromotionEvidenceView:
        def operation() -> PromotionEvidenceView:
            with self.lock, closing(self._connect()) as connection:
                now = time.time()
                promotion_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT OR IGNORE INTO dataset_promotion_events(
                      id,dataset_id,version_id,actor_id,manifest_sha256,policy_sha256,
                      decision,evidence_json,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        promotion_id,
                        request.dataset_id,
                        request.version_id,
                        actor_id,
                        request.manifest_sha256,
                        request.policy_sha256,
                        request.decision,
                        _json(request.evidence),
                        now,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM dataset_promotion_events
                    WHERE dataset_id=? AND version_id=? AND manifest_sha256=?
                    """,
                    (request.dataset_id, request.version_id, request.manifest_sha256),
                ).fetchone()
                assert row is not None
                connection.commit()
                return PromotionEvidenceView(
                    promotion_id=row["id"],
                    dataset_id=row["dataset_id"],
                    version_id=row["version_id"],
                    actor_id=row["actor_id"],
                    manifest_sha256=row["manifest_sha256"],
                    policy_sha256=row["policy_sha256"],
                    decision=row["decision"],
                    evidence=json.loads(row["evidence_json"]),
                    created_at=float(row["created_at"]),
                )

        return await asyncio.to_thread(operation)

    async def review_evidence(self) -> ReviewEvidenceReport:
        def operation() -> ReviewEvidenceReport:
            with self.lock, closing(self._connect()) as connection:
                items = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT id,source_id,status FROM dataset_review_items"
                    )
                ]
                decisions = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT review_item_id,decision,created_at FROM dataset_review_decisions ORDER BY created_at"
                    )
                ]
                return _review_evidence_from_rows(items, decisions)

        return await asyncio.to_thread(operation)


class PostgresDatasetAuthorityStore:
    """Async Postgres implementation using migration ``0006_dataset_trust``."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    import asyncpg

                    self._pool = await asyncpg.create_pool(
                        self.dsn, min_size=1, max_size=10, command_timeout=30
                    )
        return self._pool

    async def record_source_snapshot(self, request: SourceSnapshotCreate) -> SourceSnapshotView:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                INSERT INTO data_source_snapshots(
                  id,source_id,source_family,immutable_revision,registry_sha256,
                  license_evidence_sha256,legal_approval_sha256,snapshot_sha256,status,metadata
                ) VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)
                ON CONFLICT(source_id,immutable_revision,snapshot_sha256)
                DO UPDATE SET snapshot_sha256=data_source_snapshots.snapshot_sha256
                RETURNING *
                """,
                str(uuid.uuid4()),
                request.source_id,
                request.source_family,
                request.immutable_revision,
                request.registry_sha256,
                request.license_evidence_sha256,
                request.legal_approval_sha256,
                request.snapshot_sha256,
                request.status,
                _json(request.metadata),
            )
            return SourceSnapshotView(
                snapshot_id=str(row["id"]),
                source_id=row["source_id"],
                source_family=row["source_family"],
                immutable_revision=row["immutable_revision"],
                registry_sha256=row["registry_sha256"],
                license_evidence_sha256=row["license_evidence_sha256"],
                legal_approval_sha256=row["legal_approval_sha256"],
                snapshot_sha256=row["snapshot_sha256"],
                status=row["status"],
                metadata=dict(row["metadata"]),
                created_at=row["created_at"].timestamp(),
            )

    async def review_evidence(self) -> ReviewEvidenceReport:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            item_rows = await connection.fetch(
                "SELECT id,source_id,status FROM dataset_review_items"
            )
            decision_rows = await connection.fetch(
                "SELECT review_item_id,decision,created_at FROM dataset_review_decisions ORDER BY created_at"
            )
        items = [
            {"id": str(row["id"]), "source_id": row["source_id"], "status": row["status"]}
            for row in item_rows
        ]
        decisions = [
            {
                "review_item_id": str(row["review_item_id"]),
                "decision": row["decision"],
                "created_at": row["created_at"].timestamp(),
            }
            for row in decision_rows
        ]
        return _review_evidence_from_rows(items, decisions)

    @staticmethod
    def _view_from_rows(
        item: Any, assignments: list[Any], decisions: list[Any], reviewer_id: str | None
    ) -> ReviewItemView:
        visible = [
            ReviewDecisionView(
                reviewer_id=row["reviewer_id"],
                decision=row["decision"],
                rationale=row["rationale"],
                evidence=dict(row["evidence"]),
                created_at=row["created_at"].timestamp(),
            )
            for row in decisions
            if _terminal_or_visible(item["status"], reviewer_id, row["reviewer_id"])
        ]
        return ReviewItemView(
            review_item_id=str(item["id"]),
            content_hash=item["content_hash"],
            source_snapshot_sha256=item["source_snapshot_sha256"],
            source_id=item["source_id"],
            data_type=item["data_type"],
            high_value=item["high_value"],
            status=item["status"],
            version=item["version"],
            payload=dict(item["payload"]),
            assignments=[
                ReviewAssignment(
                    assignment_id=str(row["id"]),
                    review_item_id=str(row["review_item_id"]),
                    reviewer_id=row["reviewer_id"],
                    reviewer_slot=row["reviewer_slot"],
                    status=row["status"],
                    claimed_at=row["claimed_at"].timestamp(),
                    expires_at=row["expires_at"].timestamp(),
                )
                for row in assignments
            ],
            decisions=visible,
            created_at=item["created_at"].timestamp(),
            updated_at=item["updated_at"].timestamp(),
        )

    async def _view(
        self, connection: Any, review_item_id: str, reviewer_id: str | None
    ) -> ReviewItemView:
        item = await connection.fetchrow(
            "SELECT * FROM dataset_review_items WHERE id=$1::uuid", review_item_id
        )
        if item is None:
            raise KeyError(review_item_id)
        assignments = await connection.fetch(
            "SELECT * FROM dataset_review_assignments WHERE review_item_id=$1::uuid ORDER BY reviewer_slot",
            review_item_id,
        )
        decisions = await connection.fetch(
            "SELECT * FROM dataset_review_decisions WHERE review_item_id=$1::uuid ORDER BY created_at",
            review_item_id,
        )
        return self._view_from_rows(item, list(assignments), list(decisions), reviewer_id)

    async def enqueue(self, request: ReviewItemCreate) -> ReviewItemView:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                INSERT INTO dataset_review_items(
                  id,content_hash,source_snapshot_sha256,source_id,data_type,high_value,status,payload
                ) VALUES ($1::uuid,$2,$3,$4,$5,$6,'pending',$7::jsonb)
                ON CONFLICT(content_hash,source_snapshot_sha256)
                DO UPDATE SET updated_at=dataset_review_items.updated_at
                RETURNING id
                """,
                str(uuid.uuid4()),
                request.content_hash,
                request.source_snapshot_sha256,
                request.source_id,
                request.data_type,
                request.high_value,
                _json(request.payload),
            )
            return await self._view(connection, str(row["id"]), None)

    async def list_items(
        self, *, status: str | None, limit: int, reviewer_id: str | None
    ) -> list[ReviewItemView]:
        if status is not None and status not in REVIEW_STATUSES:
            raise ValueError("invalid review status")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            if status:
                rows = await connection.fetch(
                    "SELECT id FROM dataset_review_items WHERE status=$1 ORDER BY created_at LIMIT $2",
                    status,
                    limit,
                )
            else:
                rows = await connection.fetch(
                    "SELECT id FROM dataset_review_items ORDER BY created_at LIMIT $1", limit
                )
            return [await self._view(connection, str(row["id"]), reviewer_id) for row in rows]

    async def get_item(self, review_item_id: str, *, reviewer_id: str | None) -> ReviewItemView:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            return await self._view(connection, review_item_id, reviewer_id)

    async def claim(
        self, review_item_id: str, reviewer_id: str, lease_seconds: int
    ) -> ReviewAssignment:
        if not 60 <= lease_seconds <= 86_400:
            raise ValueError("review lease must be between 60 and 86400 seconds")
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            item = await connection.fetchrow(
                "SELECT * FROM dataset_review_items WHERE id=$1::uuid FOR UPDATE",
                review_item_id,
            )
            if item is None:
                raise KeyError(review_item_id)
            await connection.execute(
                "DELETE FROM dataset_review_assignments WHERE review_item_id=$1::uuid AND status='claimed' AND expires_at<now()",
                review_item_id,
            )
            if await connection.fetchval(
                "SELECT 1 FROM dataset_review_assignments WHERE review_item_id=$1::uuid AND reviewer_id=$2",
                review_item_id,
                reviewer_id,
            ):
                raise PermissionError("same reviewer cannot occupy both review slots")
            required = 2 if item["high_value"] else 1
            occupied = {
                row["reviewer_slot"]
                for row in await connection.fetch(
                    "SELECT reviewer_slot FROM dataset_review_assignments WHERE review_item_id=$1::uuid",
                    review_item_id,
                )
            }
            slot = next(
                (candidate for candidate in range(1, required + 1) if candidate not in occupied),
                None,
            )
            if slot is None:
                raise ValueError("all review slots are already claimed")
            assignment_id = str(uuid.uuid4())
            row = await connection.fetchrow(
                """
                INSERT INTO dataset_review_assignments(
                  id,review_item_id,reviewer_id,reviewer_slot,status,expires_at
                ) VALUES ($1::uuid,$2::uuid,$3,$4,'claimed',now()+($5 || ' seconds')::interval)
                RETURNING *
                """,
                assignment_id,
                review_item_id,
                reviewer_id,
                slot,
                str(lease_seconds),
            )
            await connection.execute(
                "UPDATE dataset_review_items SET status='in_review',version=version+1,updated_at=now() WHERE id=$1::uuid",
                review_item_id,
            )
            return ReviewAssignment(
                assignment_id=str(row["id"]),
                review_item_id=str(row["review_item_id"]),
                reviewer_id=row["reviewer_id"],
                reviewer_slot=row["reviewer_slot"],
                status=row["status"],
                claimed_at=row["claimed_at"].timestamp(),
                expires_at=row["expires_at"].timestamp(),
            )

    async def decide(
        self, review_item_id: str, reviewer_id: str, request: ReviewDecisionCreate
    ) -> ReviewItemView:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            item = await connection.fetchrow(
                "SELECT * FROM dataset_review_items WHERE id=$1::uuid FOR UPDATE",
                review_item_id,
            )
            if item is None:
                raise KeyError(review_item_id)
            if (
                item["content_hash"] != request.content_hash
                or item["source_snapshot_sha256"] != request.source_snapshot_sha256
            ):
                raise ValueError("review evidence no longer matches content/source snapshot")
            assignment = await connection.fetchrow(
                """
                SELECT * FROM dataset_review_assignments
                WHERE review_item_id=$1::uuid AND reviewer_id=$2 AND status='claimed' AND expires_at>=now()
                """,
                review_item_id,
                reviewer_id,
            )
            if assignment is None:
                raise PermissionError("reviewer does not hold an active assignment")
            await connection.execute(
                """
                INSERT INTO dataset_review_decisions(
                  id,review_item_id,reviewer_id,decision,rationale,content_hash,
                  source_snapshot_sha256,evidence
                ) VALUES ($1::uuid,$2::uuid,$3,$4,$5,$6,$7,$8::jsonb)
                """,
                str(uuid.uuid4()),
                review_item_id,
                reviewer_id,
                request.decision,
                request.rationale,
                request.content_hash,
                request.source_snapshot_sha256,
                _json(request.evidence),
            )
            await connection.execute(
                "UPDATE dataset_review_assignments SET status='submitted' WHERE id=$1::uuid",
                str(assignment["id"]),
            )
            decisions = [
                row["decision"]
                for row in await connection.fetch(
                    "SELECT decision FROM dataset_review_decisions WHERE review_item_id=$1::uuid ORDER BY created_at",
                    review_item_id,
                )
            ]
            status = _status_from_decisions(item["high_value"], decisions)
            await connection.execute(
                "UPDATE dataset_review_items SET status=$1,version=version+1,updated_at=now() WHERE id=$2::uuid",
                status,
                review_item_id,
            )
            return await self._view(connection, review_item_id, reviewer_id)

    async def adjudicate(
        self, review_item_id: str, adjudicator_id: str, request: ReviewAdjudicationCreate
    ) -> ReviewItemView:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            item = await connection.fetchrow(
                "SELECT * FROM dataset_review_items WHERE id=$1::uuid FOR UPDATE",
                review_item_id,
            )
            if item is None:
                raise KeyError(review_item_id)
            if item["status"] not in {"conflict", "adjudication_required"}:
                raise ValueError("only conflicting reviews can be adjudicated")
            if await connection.fetchval(
                "SELECT 1 FROM dataset_review_decisions WHERE review_item_id=$1::uuid AND reviewer_id=$2",
                review_item_id,
                adjudicator_id,
            ):
                raise PermissionError("a reviewer cannot adjudicate the same record")
            await connection.execute(
                """
                INSERT INTO dataset_review_adjudications(
                  id,review_item_id,adjudicator_id,decision,rationale,evidence
                ) VALUES ($1::uuid,$2::uuid,$3,$4,$5,$6::jsonb)
                """,
                str(uuid.uuid4()),
                review_item_id,
                adjudicator_id,
                request.decision,
                request.rationale,
                _json(request.evidence),
            )
            status = "approved" if request.decision == "approve" else "rejected"
            await connection.execute(
                "UPDATE dataset_review_items SET status=$1,version=version+1,updated_at=now() WHERE id=$2::uuid",
                status,
                review_item_id,
            )
            return await self._view(connection, review_item_id, adjudicator_id)

    async def record_promotion(
        self, actor_id: str, request: PromotionEvidenceCreate
    ) -> PromotionEvidenceView:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                INSERT INTO dataset_promotion_events(
                  id,dataset_id,version_id,actor_id,manifest_sha256,policy_sha256,decision,evidence
                ) VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8::jsonb)
                ON CONFLICT(dataset_id,version_id,manifest_sha256)
                DO UPDATE SET manifest_sha256=dataset_promotion_events.manifest_sha256
                RETURNING *
                """,
                str(uuid.uuid4()),
                request.dataset_id,
                request.version_id,
                actor_id,
                request.manifest_sha256,
                request.policy_sha256,
                request.decision,
                _json(request.evidence),
            )
            return PromotionEvidenceView(
                promotion_id=str(row["id"]),
                dataset_id=row["dataset_id"],
                version_id=row["version_id"],
                actor_id=row["actor_id"],
                manifest_sha256=row["manifest_sha256"],
                policy_sha256=row["policy_sha256"],
                decision=row["decision"],
                evidence=dict(row["evidence"]),
                created_at=row["created_at"].timestamp(),
            )


class DatasetAuthorityService:
    def __init__(self, store: DatasetAuthorityStore) -> None:
        self.store = store

    @classmethod
    def from_environment(cls) -> DatasetAuthorityService:
        dsn = os.environ.get("CRAFTLY_DATABASE_URL", "").strip()
        if dsn.startswith(("postgres://", "postgresql://")):
            return cls(PostgresDatasetAuthorityStore(dsn))
        local_path = os.environ.get(
            "CRAFTLY_DATA_AUTHORITY_DB", "artifacts/craftly/dataset-authority.sqlite3"
        )
        return cls(SQLiteDatasetAuthorityStore(local_path))

    async def record_source_snapshot(self, request: SourceSnapshotCreate) -> SourceSnapshotView:
        return await self.store.record_source_snapshot(request)

    async def enqueue(self, request: ReviewItemCreate) -> ReviewItemView:
        return await self.store.enqueue(request)

    async def list_items(
        self, *, status: str | None, limit: int, reviewer_id: str | None
    ) -> list[ReviewItemView]:
        return await self.store.list_items(status=status, limit=limit, reviewer_id=reviewer_id)

    async def get_item(self, review_item_id: str, *, reviewer_id: str | None) -> ReviewItemView:
        return await self.store.get_item(review_item_id, reviewer_id=reviewer_id)

    async def claim(
        self, review_item_id: str, reviewer_id: str, lease_seconds: int = 3_600
    ) -> ReviewAssignment:
        return await self.store.claim(review_item_id, reviewer_id, lease_seconds)

    async def decide(
        self, review_item_id: str, reviewer_id: str, request: ReviewDecisionCreate
    ) -> ReviewItemView:
        return await self.store.decide(review_item_id, reviewer_id, request)

    async def adjudicate(
        self,
        review_item_id: str,
        adjudicator_id: str,
        request: ReviewAdjudicationCreate,
    ) -> ReviewItemView:
        return await self.store.adjudicate(review_item_id, adjudicator_id, request)

    async def record_promotion(
        self, actor_id: str, request: PromotionEvidenceCreate
    ) -> PromotionEvidenceView:
        return await self.store.record_promotion(actor_id, request)

    async def review_evidence(self) -> ReviewEvidenceReport:
        return await self.store.review_evidence()


def content_snapshot_hash(content: str, source_snapshot_sha256: str) -> str:
    _validate_hash(source_snapshot_sha256, "source_snapshot_sha256")
    return hashlib.sha256(
        f"{source_snapshot_sha256}:{content}".encode("utf-8", "replace")
    ).hexdigest()


def new_bootstrap_review_token() -> str:
    """Return a high-entropy value for out-of-band reviewer bootstrap workflows."""

    return secrets.token_urlsafe(32)


async def _write_review_evidence_report(output_path: str | Path) -> ReviewEvidenceReport:
    service = DatasetAuthorityService.from_environment()
    report = await service.review_evidence()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def _cli_service(database: str | None) -> DatasetAuthorityService:
    if database:
        return DatasetAuthorityService(SQLiteDatasetAuthorityStore(Path(database).resolve()))
    return DatasetAuthorityService.from_environment()


def _validate_cli_identity(args: argparse.Namespace) -> None:
    roster_path = getattr(args, "reviewer_roster", None)
    if not roster_path or args.command in {
        "finalize-reviewer-governance",
        "initialize-reviewer-governance",
        "initialize-reviewer-qualification",
        "list",
        "prepare-reviewer-deliveries",
        "evaluate-reviewer-qualification",
        "review-report",
        "reviewer-roster-template",
        "validate-reviewer-roster",
    }:
        return
    roster = load_reviewer_roster(roster_path)
    if args.command in {"claim", "decide"}:
        roster.active_identity(args.reviewer_id, "reviewer")
    elif args.command == "adjudicate":
        roster.active_identity(args.adjudicator_id, "adjudicator")


def _parse_cli_evidence(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--evidence-json must contain a JSON object")
    return payload


async def _run_cli_command(args: argparse.Namespace) -> StrictModel | list[StrictModel]:
    _validate_cli_identity(args)
    if args.command == "initialize-reviewer-governance":
        return initialize_reviewer_governance_bundle(
            args.output_dir,
            roster_id=args.roster_id,
        )
    if args.command == "initialize-reviewer-qualification":
        return initialize_reviewer_qualification_pack(args.output_dir)
    if args.command == "finalize-reviewer-governance":
        return finalize_reviewer_governance_bundle(args.output_dir)
    if args.command == "prepare-reviewer-deliveries":
        return prepare_reviewer_delivery_packages(
            args.governance_dir,
            args.output_dir,
        )
    if args.command == "evaluate-reviewer-qualification":
        return evaluate_reviewer_qualification(
            governance_dir=args.governance_dir,
            response_paths=args.response,
            output_dir=args.output_dir,
            minimum_accuracy=args.minimum_accuracy,
            minimum_kappa=args.minimum_kappa,
        )
    if args.command == "reviewer-roster-template":
        template = build_reviewer_roster_onboarding_template(
            roster_id=args.roster_id,
            final_roster_path=args.final_roster_path,
        )
        _write_model_atomically(template, args.output)
        return template
    if args.command == "validate-reviewer-roster":
        return reviewer_roster_readiness(load_reviewer_roster(args.reviewer_roster))
    service = _cli_service(getattr(args, "database", None))
    if args.command == "review-report":
        report = await service.review_evidence()
        _write_model_atomically(report, args.output)
        return report
    if args.command == "list":
        return await service.list_items(
            status=args.status,
            limit=args.limit,
            reviewer_id=args.reviewer_id,
        )
    if args.command == "claim":
        return await service.claim(args.review_item_id, args.reviewer_id, args.lease_seconds)
    if args.command == "decide":
        item = await service.get_item(args.review_item_id, reviewer_id=args.reviewer_id)
        return await service.decide(
            args.review_item_id,
            args.reviewer_id,
            ReviewDecisionCreate(
                decision=args.decision,
                rationale=args.rationale,
                content_hash=item.content_hash,
                source_snapshot_sha256=item.source_snapshot_sha256,
                evidence=_parse_cli_evidence(args.evidence_json),
            ),
        )
    if args.command == "adjudicate":
        return await service.adjudicate(
            args.review_item_id,
            args.adjudicator_id,
            ReviewAdjudicationCreate(
                decision=args.decision,
                rationale=args.rationale,
                evidence=_parse_cli_evidence(args.evidence_json),
            ),
        )
    raise ValueError(f"unsupported command: {args.command}")


def _write_model_atomically(model: StrictModel, output_path: str | Path) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_text_atomically(output_path: str | Path, payload: str) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_deterministic_zip_entry(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("reviewer package entry must be a safe relative path")
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export immutable review evidence from the configured Craftly Dataset Authority.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize_parser = subparsers.add_parser(
        "initialize-reviewer-governance",
        help="Create a fail-closed external reviewer roster, rubric, and hash manifest.",
    )
    initialize_parser.add_argument("--output-dir", required=True)
    initialize_parser.add_argument("--roster-id", default="craftly-data-reviewers-v1")
    qualification_parser = subparsers.add_parser(
        "initialize-reviewer-qualification",
        help="Create a deterministic 20-item blind qualification pack and restricted answer key.",
    )
    qualification_parser.add_argument("--output-dir", required=True)
    finalize_parser = subparsers.add_parser(
        "finalize-reviewer-governance",
        help="Rebind the manifest to a ready real-identity roster and code-governed rubric.",
    )
    finalize_parser.add_argument("--output-dir", required=True)
    delivery_parser = subparsers.add_parser(
        "prepare-reviewer-deliveries",
        help="Build two answer-key-free reviewer delivery packages.",
    )
    delivery_parser.add_argument("--governance-dir", required=True)
    delivery_parser.add_argument("--output-dir", required=True)
    evaluation_parser = subparsers.add_parser(
        "evaluate-reviewer-qualification",
        help="Validate two blind reviewer submissions and emit a hash-bound aggregate qualification decision.",
    )
    evaluation_parser.add_argument("--governance-dir", required=True)
    evaluation_parser.add_argument("--response", action="append", required=True)
    evaluation_parser.add_argument("--output-dir", required=True)
    evaluation_parser.add_argument("--minimum-accuracy", type=float, default=0.95)
    evaluation_parser.add_argument("--minimum-kappa", type=float, default=0.80)
    report_parser = subparsers.add_parser(
        "review-report",
        help="Export aggregate independent-review evidence for dataset promotion.",
    )
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--database")
    roster_template_parser = subparsers.add_parser(
        "reviewer-roster-template",
        help="Create a non-authorizing onboarding template for real reviewer identities.",
    )
    roster_template_parser.add_argument("--output", required=True)
    roster_template_parser.add_argument("--roster-id", default="craftly-data-reviewers-v1")
    roster_template_parser.add_argument("--final-roster-path", default="config/data_reviewers.json")
    roster_validate_parser = subparsers.add_parser(
        "validate-reviewer-roster",
        help="Validate reviewer independence and report calibration readiness.",
    )
    roster_validate_parser.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    list_parser = subparsers.add_parser("list", help="List blinded review records.")
    list_parser.add_argument("--database")
    list_parser.add_argument("--reviewer-id")
    list_parser.add_argument("--status", choices=sorted(REVIEW_STATUSES))
    list_parser.add_argument("--limit", type=int, default=100)
    claim_parser = subparsers.add_parser("claim", help="Claim one reviewer slot.")
    claim_parser.add_argument("--database")
    claim_parser.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    claim_parser.add_argument("--review-item-id", required=True)
    claim_parser.add_argument("--reviewer-id", required=True)
    claim_parser.add_argument("--lease-seconds", type=int, default=3_600)
    decide_parser = subparsers.add_parser("decide", help="Submit a blinded reviewer decision.")
    decide_parser.add_argument("--database")
    decide_parser.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    decide_parser.add_argument("--review-item-id", required=True)
    decide_parser.add_argument("--reviewer-id", required=True)
    decide_parser.add_argument("--decision", choices=["approve", "reject"], required=True)
    decide_parser.add_argument("--rationale", required=True)
    decide_parser.add_argument("--evidence-json")
    adjudicate_parser = subparsers.add_parser(
        "adjudicate", help="Resolve a conflicting review with a third identity."
    )
    adjudicate_parser.add_argument("--database")
    adjudicate_parser.add_argument("--reviewer-roster", default="config/data_reviewers.json")
    adjudicate_parser.add_argument("--review-item-id", required=True)
    adjudicate_parser.add_argument("--adjudicator-id", required=True)
    adjudicate_parser.add_argument("--decision", choices=["approve", "reject"], required=True)
    adjudicate_parser.add_argument("--rationale", required=True)
    adjudicate_parser.add_argument("--evidence-json")
    args = parser.parse_args()
    result = asyncio.run(_run_cli_command(args))
    if isinstance(result, list):
        payload = [item.model_dump(mode="json") for item in result]
    else:
        payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if isinstance(result, ReviewerRosterReadinessReport) and result.status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
