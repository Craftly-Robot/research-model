"""Dataset version manifests and append-only local ledger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.storage import StoredObject, file_sha256
from src.craftly.shared.schemas import write_report
from src.craftly.shared.schemas import StrictModel, write_report


class DatasetArtifact(StrictModel):
    path: str
    role: str
    size_bytes: int
    sha256: str


class DatasetVersionManifest(StrictModel):
    dataset_id: str
    version_id: str
    created_at_unix: float = Field(default_factory=time.time)
    source_registry: dict[str, Any]
    crawl_report: dict[str, Any]
    license_filter_report: dict[str, Any] | None = None
    benchmark_contamination_filter_report: dict[str, Any] | None = None
    near_dedup_report: dict[str, Any] | None = None
    contamination_report: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    training_quality_report: dict[str, Any] | None = None
    source_quality_report: dict[str, Any] | None = None
    source_reputation_report: dict[str, Any] | None = None
    source_budget_plan: dict[str, Any] | None = None
    training_data_gate_report: dict[str, Any] | None = None
    source_balance_report: dict[str, Any] | None = None
    instruction_mix_report: dict[str, Any] | None = None
    task_report: dict[str, Any] | None = None
    review_report: dict[str, Any] | None = None
    feedback_report: dict[str, Any] | None = None
    checkpoint_eval_report: dict[str, Any] | None = None
    checkpoint_comparison_report: dict[str, Any] | None = None
    tokenizer_path: str
    shard_manifest: dict[str, Any]
    artifacts: list[DatasetArtifact]
    uploaded_objects: list[StoredObject] = Field(default_factory=list)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_report(self, target)
        return target


class DatasetLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, manifest: DatasetVersionManifest) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest.model_dump(), sort_keys=True) + "\n")

    def latest(self, dataset_id: str | None = None) -> DatasetVersionManifest | None:
        if not self.path.exists():
            return None
        latest_payload: dict[str, Any] | None = None
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if dataset_id is None or payload.get("dataset_id") == dataset_id:
                latest_payload = payload
        return DatasetVersionManifest.model_validate(latest_payload) if latest_payload else None


def artifact_from_path(path: str | Path, *, role: str) -> DatasetArtifact:
    source = Path(path)
    return DatasetArtifact(
        path=str(source), role=role, size_bytes=source.stat().st_size, sha256=file_sha256(source)
    )


def build_version_id(dataset_id: str, artifact_hashes: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(dataset_id.encode("utf-8"))
    for item in sorted(artifact_hashes):
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()[:16]
