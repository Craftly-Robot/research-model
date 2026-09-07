"""Unified memory system for Craftly."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.indexing.vector_index import cosine, hashed_embedding
from src.craftly.shared.schemas import StrictModel


MemoryLayer = Literal["working", "project", "episodic", "semantic", "user", "verified_fix"]
ALLOWED_MEMORY_LAYERS: set[str] = {"working", "project", "episodic", "semantic", "user", "verified_fix"}
ALLOWED_INGESTION_KINDS: set[str] = {
    "benchmark_pass",
    "project_fact",
    "security_finding",
    "successful_plan",
    "user_preference",
    "verified_fix",
}
REJECTED_INGESTION_KINDS: set[str] = {"failed_guess", "raw_thought", "transient_output"}


class MemoryEntry(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str | None = None
    layer: MemoryLayer
    kind: str
    content: dict[str, Any]
    relevance: float = Field(ge=0.0, le=1.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    usage_count: int = 0
    last_used: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=time.time)

    def text(self) -> str:
        return " ".join([self.layer, self.kind, str(self.content), str(self.metadata)])


class MemoryIngestRequest(StrictModel):
    layer: MemoryLayer
    kind: str
    content: dict[str, Any]
    relevance: float = Field(default=0.7, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalHit(StrictModel):
    entry: MemoryEntry
    vector_similarity: float
    success_rate: float
    recency_weight: float
    usage_count_weight: float
    final_score: float


class MemoryRetrievalReport(StrictModel):
    project_id: str | None
    query: str
    layers: list[str]
    hits: list[MemoryRetrievalHit]


def recency_weight(created_at: float, *, now: float | None = None, half_life_seconds: float = 30 * 24 * 60 * 60) -> float:
    active_now = now or time.time()
    age = max(0.0, active_now - created_at)
    return 0.5 ** (age / half_life_seconds)


def usage_count_weight(usage_count: int) -> float:
    return min(1.0, usage_count / 20.0)


def memory_rank_score(
    *,
    vector_similarity: float,
    success_rate: float,
    recency: float,
    usage: float,
) -> float:
    return (0.4 * vector_similarity) + (0.3 * success_rate) + (0.2 * recency) + (0.1 * usage)


class UnifiedMemoryManager:
    """Typed memory manager with strict anti-pollution ingestion rules."""

    def __init__(self, *, project_id: str | None = "default", store: LocalStore | None = None, dims: int = 384) -> None:
        self.project_id = project_id
        self.store = store
        self.dims = dims
        self.entries: list[MemoryEntry] = []

    def ingest(self, request: MemoryIngestRequest) -> MemoryEntry:
        if request.layer not in ALLOWED_MEMORY_LAYERS:
            raise ValueError(f"unsupported memory layer: {request.layer}")
        if request.kind in REJECTED_INGESTION_KINDS or request.kind not in ALLOWED_INGESTION_KINDS:
            raise ValueError(f"memory kind is not allowed for ingestion: {request.kind}")
        if request.layer == "working":
            # Working memory is session-local. Keep it in-process only.
            entry = self._entry_from_request(request)
            self.entries.append(entry)
            return entry
        if self.store is not None:
            stored = self.store.insert_memory_entry(
                project_id=self.project_id,
                kind=f"{request.layer}:{request.kind}",
                content=request.content,
                source_run_id=request.source_run_id,
                relevance=request.relevance,
                success_rate=request.success_rate,
                metadata={"layer": request.layer, "kind": request.kind, **request.metadata},
            )
            return self._entry_from_store(stored)
        entry = self._entry_from_request(request)
        self.entries.append(entry)
        return entry

    def remember_verified_fix(self, failure: str, fix: str, context: str) -> dict[str, Any]:
        entry = self.ingest(
            MemoryIngestRequest(
                layer="verified_fix",
                kind="verified_fix",
                content={"failure": failure, "fix": fix, "context": context},
                relevance=0.85,
                success_rate=1.0,
            )
        )
        return entry.model_dump()

    def remember_project_fact(self, module_name: str, path: str, tech_stack: str) -> dict[str, Any]:
        entry = self.ingest(
            MemoryIngestRequest(
                layer="project",
                kind="project_fact",
                content={"module_name": module_name, "path": path, "tech_stack": tech_stack},
                relevance=0.75,
                success_rate=0.9,
            )
        )
        return entry.model_dump()

    def remember_user_preference(self, preference: str, context: str = "") -> dict[str, Any]:
        entry = self.ingest(
            MemoryIngestRequest(
                layer="user",
                kind="user_preference",
                content={"preference": preference, "context": context},
                relevance=0.7,
                success_rate=0.85,
            )
        )
        return entry.model_dump()

    def retrieve(self, query: str, *, limit: int = 5, layers: list[MemoryLayer] | None = None) -> dict[str, Any]:
        return self.retrieve_report(query, limit=limit, layers=layers).model_dump()

    def retrieve_report(self, query: str, *, limit: int = 5, layers: list[MemoryLayer] | None = None) -> MemoryRetrievalReport:
        active_layers = layers or ["verified_fix", "project", "episodic", "semantic", "user", "working"]
        unknown_layers = [layer for layer in active_layers if layer not in ALLOWED_MEMORY_LAYERS]
        if unknown_layers:
            raise ValueError(f"unsupported memory layer(s): {', '.join(str(layer) for layer in unknown_layers)}")
        entries = self._load_entries(active_layers)
        query_vector = hashed_embedding(query, dims=self.dims)
        hits: list[MemoryRetrievalHit] = []
        now = time.time()
        for entry in entries:
            vector_similarity = max(0.0, cosine(query_vector, hashed_embedding(entry.text(), dims=self.dims)))
            recency = recency_weight(entry.created_at_unix, now=now)
            usage = usage_count_weight(entry.usage_count)
            final_score = memory_rank_score(
                vector_similarity=vector_similarity,
                success_rate=entry.success_rate,
                recency=recency,
                usage=usage,
            )
            if final_score <= 0:
                continue
            hits.append(
                MemoryRetrievalHit(
                    entry=entry,
                    vector_similarity=round(vector_similarity, 6),
                    success_rate=entry.success_rate,
                    recency_weight=round(recency, 6),
                    usage_count_weight=round(usage, 6),
                    final_score=round(final_score, 6),
                )
            )
        hits.sort(key=lambda item: item.final_score, reverse=True)
        selected = hits[:limit]
        if self.store is not None:
            for hit in selected:
                try:
                    self.store.mark_memory_used(hit.entry.id)
                except Exception:
                    pass
        else:
            selected_ids = {hit.entry.id for hit in selected}
            for entry in self.entries:
                if entry.id in selected_ids:
                    entry.usage_count += 1
                    entry.last_used = now
        return MemoryRetrievalReport(
            project_id=self.project_id,
            query=query,
            layers=[str(layer) for layer in active_layers],
            hits=selected,
        )

    def archive_low_quality(self, *, min_success_rate: float = 0.25, min_usage_count: int = 0) -> list[MemoryEntry]:
        archived: list[MemoryEntry] = []
        kept: list[MemoryEntry] = []
        for entry in self.entries:
            if entry.success_rate < min_success_rate and entry.usage_count <= min_usage_count:
                entry.metadata["archived"] = True
                archived.append(entry)
            else:
                kept.append(entry)
        self.entries = kept
        return archived

    def _entry_from_request(self, request: MemoryIngestRequest) -> MemoryEntry:
        return MemoryEntry(
            project_id=self.project_id,
            layer=request.layer,
            kind=request.kind,
            content=request.content,
            relevance=request.relevance,
            success_rate=request.success_rate,
            metadata=request.metadata,
        )

    def _entry_from_store(self, stored: dict[str, Any]) -> MemoryEntry:
        metadata = stored.get("metadata") or {}
        kind = str(metadata.get("kind") or stored["kind"]).split(":", 1)[-1]
        layer = str(metadata.get("layer") or str(stored["kind"]).split(":", 1)[0])
        if layer not in ALLOWED_MEMORY_LAYERS:
            layer = "semantic"
        return MemoryEntry(
            id=stored["id"],
            project_id=stored.get("project_id"),
            layer=layer,  # type: ignore[arg-type]
            kind=kind,
            content=stored["content"],
            relevance=float(stored["relevance"]),
            success_rate=float(stored["success_rate"]),
            usage_count=int(stored["usage_count"]),
            last_used=stored.get("last_used_at"),
            metadata=metadata,
            created_at_unix=float(stored["created_at"]),
        )

    def _load_entries(self, layers: list[MemoryLayer]) -> list[MemoryEntry]:
        in_memory = [entry for entry in self.entries if entry.layer in layers]
        if self.store is None:
            return in_memory
        prefixes = [f"{layer}:" for layer in layers]
        stored_entries = []
        for row in self.store.list_memory_entries(self.project_id):
            if any(str(row["kind"]).startswith(prefix) for prefix in prefixes):
                stored_entries.append(self._entry_from_store(row))
        return [*in_memory, *stored_entries]


CraftlyMemory = UnifiedMemoryManager

