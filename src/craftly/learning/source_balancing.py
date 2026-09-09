"""Source balancing for training corpora.

The crawler intentionally keeps provenance. This module uses that provenance to
prevent a single source from dominating tokenizer/shard training.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl
from src.craftly.shared.schemas import StrictModel


class SourceBalanceItem(StrictModel):
    source: str
    input_rows: int
    output_rows: int
    action: str
    avg_quality_score: float


class SourceBalanceReport(StrictModel):
    input_paths: list[str]
    output_path: str
    input_rows: int
    output_rows: int
    max_source_fraction: float
    sources: list[SourceBalanceItem]
    created_at_unix: float = Field(default_factory=time.time)


def _quality(row: dict[str, Any]) -> float:
    quality = row.get("quality")
    if isinstance(quality, dict):
        return float(quality.get("quality_score", 0.0))
    return 0.0


def _row_sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
    return (
        -_quality(row),
        str(row.get("content_hash") or ""),
        str(row.get("url") or ""),
    )


def _cap_for_source(
    *, source_count: int, other_count: int, max_source_fraction: float, min_source_rows: int
) -> int:
    if other_count <= 0:
        return source_count
    cap = math.floor((max_source_fraction / max(1e-9, 1.0 - max_source_fraction)) * other_count)
    return min(source_count, max(min_source_rows, cap))


def balance_clean_jsonl(
    *,
    input_paths: list[str | Path],
    output_path: str | Path,
    max_source_fraction: float = 0.35,
    min_source_rows: int = 25,
) -> SourceBalanceReport:
    if not 0.05 <= max_source_fraction <= 1.0:
        raise ValueError("max_source_fraction must be between 0.05 and 1.0")
    source_counts: dict[str, int] = defaultdict(int)
    quality_totals: dict[str, float] = defaultdict(float)
    for path in input_paths:
        for row in iter_jsonl(path):
            source = str(row.get("source") or "unknown")
            source_counts[source] += 1
            quality_totals[source] += _quality(row)
    all_rows = sum(source_counts.values())
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    items: list[SourceBalanceItem] = []
    caps: dict[str, int] = {}
    for source, source_count in sorted(source_counts.items()):
        other_count = all_rows - source_count
        cap = (
            source_count
            if max_source_fraction >= 1.0
            else _cap_for_source(
                source_count=source_count,
                other_count=other_count,
                max_source_fraction=max_source_fraction,
                min_source_rows=min_source_rows,
            )
        )
        caps[source] = cap
        avg_quality = quality_totals[source] / max(1, source_count)
        items.append(
            SourceBalanceItem(
                source=source,
                input_rows=source_count,
                output_rows=cap,
                action="kept" if cap == source_count else "capped",
                avg_quality_score=round(avg_quality, 6),
            )
        )

    output_rows = 0
    written_by_source: dict[str, int] = defaultdict(int)
    with target.open("w", encoding="utf-8") as handle:
        for path in input_paths:
            for row in iter_jsonl(path):
                source = str(row.get("source") or "unknown")
                if written_by_source[source] >= caps.get(source, 0):
                    continue
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                written_by_source[source] += 1
                output_rows += 1

    return SourceBalanceReport(
        input_paths=[str(path) for path in input_paths],
        output_path=str(target),
        input_rows=all_rows,
        output_rows=output_rows,
        max_source_fraction=max_source_fraction,
        sources=items,
    )
