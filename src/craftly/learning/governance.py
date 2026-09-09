"""Source governance, license approval, and human review operations."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.craftly.shared.schemas import StrictModel

ApprovalStatus = Literal["pending", "approved", "rejected"]
ReviewStatus = Literal["queued", "approved", "rejected"]


class SourceApprovalRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    urls: list[str] = Field(default_factory=list)
    proposed_license: str = Field(min_length=1)
    evidence_url: str = Field(min_length=1)
    requested_by: str = Field(default="system", min_length=1)
    justification: str = Field(min_length=1)
    status: ApprovalStatus = "pending"
    decision_reason: str | None = None
    decided_by: str | None = None
    created_at_unix: float = Field(default_factory=time.time)
    decided_at_unix: float | None = None


class HumanReviewItem(StrictModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = Field(min_length=1)
    priority: int = Field(default=5, ge=1, le=10)
    payload: dict[str, Any]
    status: ReviewStatus = "queued"
    reviewer: str | None = None
    decision_reason: str | None = None
    created_at_unix: float = Field(default_factory=time.time)
    decided_at_unix: float | None = None


class GovernanceReport(StrictModel):
    approvals_pending: int
    approvals_approved: int
    approvals_rejected: int
    review_queued: int
    review_approved: int
    review_rejected: int
    high_priority_review: int
    created_at_unix: float = Field(default_factory=time.time)


class GovernanceStore:
    """Append-only governance store for auditable data-source decisions."""

    def __init__(self, root: str | Path = "artifacts/craftly/governance") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.approvals_path = self.root / "source_approvals.jsonl"
        self.review_path = self.root / "human_review_queue.jsonl"

    def submit_source_approval(self, request: SourceApprovalRequest) -> SourceApprovalRequest:
        self._append(self.approvals_path, request.model_dump())
        return request

    def decide_source_approval(
        self,
        request_id: str,
        *,
        status: Literal["approved", "rejected"],
        decided_by: str,
        reason: str,
    ) -> SourceApprovalRequest:
        rows = [
            SourceApprovalRequest.model_validate(row)
            for row in self._read_jsonl(self.approvals_path)
        ]
        updated: list[SourceApprovalRequest] = []
        selected: SourceApprovalRequest | None = None
        for row in rows:
            if row.request_id == request_id:
                row.status = status
                row.decided_by = decided_by
                row.decision_reason = reason
                row.decided_at_unix = time.time()
                selected = row
            updated.append(row)
        if selected is None:
            raise KeyError(f"source approval request not found: {request_id}")
        self._rewrite(self.approvals_path, [row.model_dump() for row in updated])
        return selected

    def enqueue_review(self, item: HumanReviewItem) -> HumanReviewItem:
        self._append(self.review_path, item.model_dump())
        return item

    def decide_review(
        self,
        item_id: str,
        *,
        status: Literal["approved", "rejected"],
        reviewer: str,
        reason: str,
    ) -> HumanReviewItem:
        rows = [HumanReviewItem.model_validate(row) for row in self._read_jsonl(self.review_path)]
        updated: list[HumanReviewItem] = []
        selected: HumanReviewItem | None = None
        for row in rows:
            if row.item_id == item_id:
                row.status = status
                row.reviewer = reviewer
                row.decision_reason = reason
                row.decided_at_unix = time.time()
                selected = row
            updated.append(row)
        if selected is None:
            raise KeyError(f"human review item not found: {item_id}")
        self._rewrite(self.review_path, [row.model_dump() for row in updated])
        return selected

    def report(self) -> GovernanceReport:
        approvals = [
            SourceApprovalRequest.model_validate(row)
            for row in self._read_jsonl(self.approvals_path)
        ]
        reviews = [
            HumanReviewItem.model_validate(row) for row in self._read_jsonl(self.review_path)
        ]
        return GovernanceReport(
            approvals_pending=sum(1 for item in approvals if item.status == "pending"),
            approvals_approved=sum(1 for item in approvals if item.status == "approved"),
            approvals_rejected=sum(1 for item in approvals if item.status == "rejected"),
            review_queued=sum(1 for item in reviews if item.status == "queued"),
            review_approved=sum(1 for item in reviews if item.status == "approved"),
            review_rejected=sum(1 for item in reviews if item.status == "rejected"),
            high_priority_review=sum(
                1 for item in reviews if item.status == "queued" and item.priority >= 8
            ),
        )

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _rewrite(self, path: Path, rows: list[dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(path)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage Craftly source governance and human review queues."
    )
    parser.add_argument("--store", default="artifacts/craftly/governance")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit-source")
    submit.add_argument("--source-name", required=True)
    submit.add_argument("--category", required=True)
    submit.add_argument("--url", action="append", default=[])
    submit.add_argument("--license", required=True)
    submit.add_argument("--evidence-url", required=True)
    submit.add_argument("--requested-by", default="system")
    submit.add_argument("--justification", required=True)

    decide = sub.add_parser("decide-source")
    decide.add_argument("--request-id", required=True)
    decide.add_argument("--status", choices=["approved", "rejected"], required=True)
    decide.add_argument("--decided-by", required=True)
    decide.add_argument("--reason", required=True)

    enqueue = sub.add_parser("enqueue-review")
    enqueue.add_argument("--kind", required=True)
    enqueue.add_argument("--priority", type=int, default=5)
    enqueue.add_argument("--payload-json", required=True)

    review = sub.add_parser("decide-review")
    review.add_argument("--item-id", required=True)
    review.add_argument("--status", choices=["approved", "rejected"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)

    sub.add_parser("report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store = GovernanceStore(args.store)
    if args.command == "submit-source":
        result = store.submit_source_approval(
            SourceApprovalRequest(
                source_name=args.source_name,
                category=args.category,
                urls=args.url,
                proposed_license=args.license,
                evidence_url=args.evidence_url,
                requested_by=args.requested_by,
                justification=args.justification,
            )
        )
    elif args.command == "decide-source":
        result = store.decide_source_approval(
            args.request_id,
            status=args.status,
            decided_by=args.decided_by,
            reason=args.reason,
        )
    elif args.command == "enqueue-review":
        result = store.enqueue_review(
            HumanReviewItem(
                kind=args.kind, priority=args.priority, payload=json.loads(args.payload_json)
            )
        )
    elif args.command == "decide-review":
        result = store.decide_review(
            args.item_id, status=args.status, reviewer=args.reviewer, reason=args.reason
        )
    else:
        result = store.report()
    print(json.dumps(result.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
