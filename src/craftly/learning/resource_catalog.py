"""Training resource catalog and priority planning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.craftly.shared.schemas import StrictModel

TrainPolicy = Literal[
    "pretrain",
    "agentic_task",
    "eval_holdout",
    "research_reference",
    "governance_review",
]


class TrainingResource(StrictModel):
    resource_id: int = Field(ge=1)
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    section: str = Field(min_length=1)
    resource_type: str = Field(min_length=1)
    train_policy: TrainPolicy
    priority_rank: int = Field(default=999, ge=1)
    ingestion_mode: str = Field(min_length=1)
    license_status: str = Field(min_length=1)
    safety_policy: str = Field(min_length=1)
    notes: str = ""


class PriorityGroup(StrictModel):
    rank: int = Field(ge=1)
    name: str = Field(min_length=1)
    resource_ids: list[int] = Field(min_length=1)
    purpose: str = Field(min_length=1)


class ResourceCatalogReport(StrictModel):
    catalog_path: str
    total_resources: int
    priority_groups: list[PriorityGroup]
    train_first_resources: list[TrainingResource]
    eval_holdout_resources: list[TrainingResource]
    review_required_resources: list[TrainingResource]
    created_at_unix: float = Field(default_factory=time.time)


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_training_resources(path: str | Path) -> list[TrainingResource]:
    payload = _load_payload(path)
    return [
        TrainingResource.model_validate(item)
        for item in payload.get("training_resources", [])
    ]


def load_priority_groups(path: str | Path) -> list[PriorityGroup]:
    payload = _load_payload(path)
    return [
        PriorityGroup.model_validate(item)
        for item in payload.get("training_priority_groups", [])
    ]


def build_resource_catalog_report(
    path: str | Path, *, train_first_limit: int = 24
) -> ResourceCatalogReport:
    resources = load_training_resources(path)
    groups = load_priority_groups(path)
    ids = {item.resource_id for item in resources}
    missing = [
        resource_id
        for group in groups
        for resource_id in group.resource_ids
        if resource_id not in ids
    ]
    if missing:
        raise ValueError(
            f"priority groups reference unknown resources: {sorted(set(missing))}"
        )

    trainable = [
        item for item in resources if item.train_policy in {"pretrain", "agentic_task"}
    ]
    trainable.sort(key=lambda item: (item.priority_rank, item.resource_id))
    eval_holdout = [item for item in resources if item.train_policy == "eval_holdout"]
    eval_holdout.sort(key=lambda item: (item.priority_rank, item.resource_id))
    review_required = [
        item for item in resources if item.train_policy == "governance_review"
    ]
    review_required.sort(key=lambda item: (item.priority_rank, item.resource_id))

    return ResourceCatalogReport(
        catalog_path=str(path),
        total_resources=len(resources),
        priority_groups=sorted(groups, key=lambda item: item.rank),
        train_first_resources=trainable[:train_first_limit],
        eval_holdout_resources=eval_holdout,
        review_required_resources=review_required,
    )


def write_resource_catalog_report(
    path: str | Path, output_path: str | Path, *, train_first_limit: int = 24
) -> ResourceCatalogReport:
    report = build_resource_catalog_report(path, train_first_limit=train_first_limit)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Craftly training resource priority catalog."
    )
    parser.add_argument("--catalog", default="config/data_sources.ultimate.json")
    parser.add_argument("--output")
    parser.add_argument("--train-first-limit", type=int, default=24)
    args = parser.parse_args()
    report = build_resource_catalog_report(
        args.catalog, train_first_limit=args.train_first_limit
    )
    if args.output:
        write_resource_catalog_report(
            args.catalog, args.output, train_first_limit=args.train_first_limit
        )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
