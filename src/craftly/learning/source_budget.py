"""Source budget allocation driven by reputation scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pydantic import Field

from src.craftly.learning.source_registry import SourceRegistry
from src.craftly.shared.schemas import StrictModel


class SourceBudget(StrictModel):
    source: str
    category: str
    action: str
    reputation_score: float = Field(ge=0.0, le=1.0)
    target_docs: int
    max_depth: int
    delay_seconds: float
    reason: str


class SourceBudgetPlan(StrictModel):
    target_total_docs: int
    allocated_total_docs: int
    unallocated_docs: int
    eligible_sources: int
    blocked_sources: int
    budgets: list[SourceBudget]
    created_at_unix: float = Field(default_factory=time.time)


def _load_reputation(path: str | Path | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {str(item["source"]): item for item in payload.get("sources", [])}


def _weighted_allocations(
    weights: dict[str, float],
    caps: dict[str, int],
    target_total_docs: int,
    min_docs_per_source: int,
) -> dict[str, int]:
    allocations = {name: 0 for name in weights}
    ordered = sorted(weights, key=lambda name: (-weights[name], name))
    if not ordered or target_total_docs <= 0:
        return allocations

    minimum = min_docs_per_source if target_total_docs >= min_docs_per_source * len(ordered) else 0
    if minimum:
        for name in ordered:
            granted = min(minimum, caps[name])
            allocations[name] = granted
    remaining = max(0, target_total_docs - sum(allocations.values()))

    while remaining:
        active = [name for name in ordered if allocations[name] < caps[name]]
        if not active:
            break
        total_weight = sum(weights[name] for name in active)
        if total_weight <= 0:
            break
        raw_shares = {name: remaining * weights[name] / total_weight for name in active}
        granted_this_round = 0
        for name in active:
            grant = min(caps[name] - allocations[name], int(raw_shares[name]))
            if grant:
                allocations[name] += grant
                remaining -= grant
                granted_this_round += grant
        if remaining <= 0:
            break
        if granted_this_round == 0:
            ranked = sorted(
                active,
                key=lambda name: (
                    -(raw_shares[name] - int(raw_shares[name])),
                    -weights[name],
                    name,
                ),
            )
            for name in ranked:
                if remaining <= 0:
                    break
                if allocations[name] < caps[name]:
                    allocations[name] += 1
                    remaining -= 1
    return allocations


def build_source_budget_plan(
    *,
    sources_path: str | Path,
    reputation_report_path: str | Path | None,
    target_total_docs: int,
    min_docs_per_source: int = 10,
    max_docs_per_source: int = 5000,
) -> SourceBudgetPlan:
    registry = SourceRegistry.from_file(sources_path)
    reputation_by_source = _load_reputation(reputation_report_path)
    weights: dict[str, float] = {}
    source_meta: dict[str, tuple[str, str, float, int, bool]] = {}
    for source in registry.to_sources():
        rep = reputation_by_source.get(source.name)
        has_evidence = rep is not None
        score = float(rep.get("reputation_score", 0.0)) if rep else 0.0
        action = str(rep.get("action", "block")) if rep else "block"
        category_multiplier = (
            1.25 if source.category in {"defensive_security", "vulnerability_database"} else 1.0
        )
        action_multiplier = {
            "promote": 1.4,
            "watch": 1.0,
            "throttle": 0.35,
            "quarantine": 0.20,
            "block": 0.0,
        }.get(action, 0.0)
        weight = max(0.0, score * category_multiplier * action_multiplier)
        weights[source.name] = weight
        source_cap = min(max_docs_per_source, source.collection_budget)
        if action == "quarantine":
            source_cap = min(source_cap, max(1, int(target_total_docs * 0.01)))
        source_meta[source.name] = (source.category, action, score, source_cap, has_evidence)

    eligible_weights = {name: weight for name, weight in weights.items() if weight > 0}
    caps = {name: source_meta[name][3] for name in eligible_weights}
    allocations = _weighted_allocations(
        eligible_weights,
        caps,
        target_total_docs,
        min_docs_per_source,
    )
    budgets: list[SourceBudget] = []
    for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        category, action, score, _cap, has_evidence = source_meta[name]
        docs = allocations.get(name, 0)
        if not has_evidence:
            reason = "missing_reputation_evidence"
        elif action == "block" or weight <= 0:
            reason = "blocked_by_reputation"
        elif docs == 0:
            reason = "eligible_but_budget_exhausted"
        elif action == "quarantine":
            reason = "allocated_under_quarantine_cap"
        else:
            reason = "allocated_by_reputation_weight"
        budgets.append(
            SourceBudget(
                source=name,
                category=category,
                action=action,
                reputation_score=round(score, 6),
                target_docs=docs,
                max_depth=2 if score >= 0.70 else 1,
                delay_seconds=0.25 if score >= 0.78 else 0.75 if score >= 0.55 else 1.5,
                reason=reason,
            )
        )
    allocated = sum(item.target_docs for item in budgets)
    return SourceBudgetPlan(
        target_total_docs=target_total_docs,
        allocated_total_docs=allocated,
        unallocated_docs=max(0, target_total_docs - allocated),
        eligible_sources=len(eligible_weights),
        blocked_sources=len(weights) - len(eligible_weights),
        budgets=budgets,
    )


def write_source_budget_plan(output_path: str | Path, **kwargs: object) -> SourceBudgetPlan:
    plan = build_source_budget_plan(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocate Craftly crawl budget from source reputation."
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument("--reputation-report")
    parser.add_argument("--target-total-docs", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = write_source_budget_plan(
        args.output,
        sources_path=args.sources,
        reputation_report_path=args.reputation_report,
        target_total_docs=args.target_total_docs,
    )
    print(json.dumps(plan.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
