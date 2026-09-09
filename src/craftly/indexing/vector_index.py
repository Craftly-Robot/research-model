"""Repository vector index contracts and local/production backend adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import httpx
from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.learning.quality import SECRET_RE
from src.craftly.shared.schemas import StrictModel

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - dependency readiness handles this path.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|0x[0-9A-Fa-f]+|[./\\\w-]+\.[A-Za-z0-9]+")
VectorBackendName = Literal["local_hashing", "faiss", "hnsw", "qdrant", "pgvector"]


class ScratchEmbeddingConfig(StrictModel):
    model_name: str = "Craftly-Code-Embed-v1"
    vocab_size: int = Field(default=128_000, ge=256)
    hidden_size: int = Field(default=768, ge=64)
    num_layers: int = Field(default=12, ge=1)
    num_attention_heads: int = Field(default=12, ge=1)
    projection_dimension: int = Field(default=768, ge=64)
    intermediate_size: int = Field(default=3072, ge=128)
    max_sequence_length: int = Field(default=4096, ge=32)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    temperature: float = Field(default=0.05, gt=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")


if nn is not None:

    class ScratchCodeEmbeddingModel(nn.Module):
        """Craftly-owned random-initialized dual-encoder backbone."""

        def __init__(self, config: ScratchEmbeddingConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.output_norm = nn.LayerNorm(config.hidden_size)
            self.projection = nn.Linear(config.hidden_size, config.projection_dimension, bias=False)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "input_ids and attention_mask must have matching [batch, sequence] shapes"
                )
            if input_ids.shape[1] > self.config.max_sequence_length:
                raise ValueError("embedding input exceeds configured maximum sequence length")
            if input_ids.numel() and (
                int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
            ):
                raise ValueError("embedding input contains an out-of-vocabulary token ID")
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
            hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
            hidden = self.output_norm(hidden)
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return F.normalize(self.projection(pooled).float(), p=2, dim=-1)

        def contrastive_loss(
            self,
            query_ids: torch.Tensor,
            query_mask: torch.Tensor,
            positive_ids: torch.Tensor,
            positive_mask: torch.Tensor,
        ) -> torch.Tensor:
            query = self(query_ids, query_mask)
            positive = self(positive_ids, positive_mask)
            logits = query @ positive.transpose(0, 1) / self.config.temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            return 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
            )

        def training_loss(
            self,
            query_ids: torch.Tensor,
            query_mask: torch.Tensor,
            positive_ids: torch.Tensor,
            positive_mask: torch.Tensor,
            *,
            hard_negative_ids: torch.Tensor | None = None,
            hard_negative_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            query = self(query_ids, query_mask)
            positive = self(positive_ids, positive_mask)
            candidates = positive
            if hard_negative_ids is not None:
                if hard_negative_mask is None or hard_negative_ids.ndim != 3:
                    raise ValueError(
                        "hard negatives require matching [batch, negatives, sequence] tensors"
                    )
                batch, negatives, sequence = hard_negative_ids.shape
                hard = self(
                    hard_negative_ids.reshape(batch * negatives, sequence),
                    hard_negative_mask.reshape(batch * negatives, sequence),
                )
                candidates = torch.cat([positive, hard], dim=0)
            labels = torch.arange(query.shape[0], device=query.device)
            query_loss = F.cross_entropy(
                query @ candidates.transpose(0, 1) / self.config.temperature, labels
            )
            positive_loss = F.cross_entropy(
                positive @ query.transpose(0, 1) / self.config.temperature, labels
            )
            return 0.5 * (query_loss + positive_loss)
else:

    class ScratchCodeEmbeddingModel:  # type: ignore[no-redef]
        def __init__(self, _config: ScratchEmbeddingConfig) -> None:
            raise RuntimeError("PyTorch is required for the Craftly scratch embedding model")


def save_scratch_embedding_checkpoint(
    model: ScratchCodeEmbeddingModel,
    output_dir: str | Path,
    *,
    tokenizer_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to save an embedding checkpoint")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for secure embedding checkpoint serialization"
        ) from exc
    for name, value in {
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }.items():
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "embedding_model.safetensors"
    temporary = target / ".embedding_model.safetensors.tmp"
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    save_file(tensors, str(temporary), metadata={"format": "craftly-scratch-embedding-v1"})
    os.replace(temporary, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "model_name": model.config.model_name,
        "scratch_only": True,
        "borrowed_weights": False,
        "projection_dimension": model.config.projection_dimension,
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config": model.config.model_dump(),
        "status": "built_not_gpu_proven",
    }
    manifest_path = target / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path), "checkpoint_path": str(checkpoint)}


class EmbeddingPair(StrictModel):
    pair_id: str = Field(min_length=16, max_length=128)
    query: str = Field(min_length=1, max_length=200_000)
    positive: str = Field(min_length=1, max_length=2_000_000)
    hard_negatives: list[str] = Field(default_factory=list, max_length=8)
    category: Literal[
        "function_doc",
        "symbol_call",
        "code_test",
        "error_fix",
        "task_evidence",
        "security_patch",
    ]
    source_revision: str = Field(min_length=1, max_length=512)
    lineage_key: str = Field(min_length=1, max_length=1024)
    verified: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingPairBuildReport(StrictModel):
    status: Literal["passed", "blocked"]
    project_id: str
    index_revision: str
    pair_count: int
    categories: dict[str, int]
    output_path: str
    output_sha256: str
    blockers: list[str] = Field(default_factory=list)


class EmbeddingTrainingConfig(StrictModel):
    model: ScratchEmbeddingConfig = Field(default_factory=ScratchEmbeddingConfig)
    steps: int = Field(default=10_000, ge=1)
    batch_size: int = Field(default=32, ge=2, le=1024)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=1024)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1e-2)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=500, ge=0)
    validation_interval: int = Field(default=250, ge=1)
    checkpoint_interval: int = Field(default=1000, ge=1)
    validation_fraction: float = Field(default=0.05, gt=0.0, lt=0.5)
    max_validation_pairs: int = Field(default=2000, ge=10, le=100_000)
    hard_negatives_per_query: int = Field(default=1, ge=0, le=8)
    seed: int = 1337
    device: str = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    early_stopping_patience: int = Field(default=10, ge=1, le=100)
    max_gradient_norm: float = Field(default=1.0, gt=0.0, le=100.0)

    def model_post_init(self, __context: Any) -> None:
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")


class EmbeddingRetrievalMetrics(StrictModel):
    pair_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_20: float
    mrr_at_10: float
    mean_positive_similarity: float
    mean_off_diagonal_similarity: float
    embedding_dimension_std: float
    collapse_detected: bool


class EmbeddingTrainingReport(StrictModel):
    status: Literal["passed", "failed", "blocked"]
    steps_completed: int
    best_step: int
    best_validation_loss: float | None = None
    train_loss_last: float | None = None
    metrics: EmbeddingRetrievalMetrics | None = None
    tokenizer_sha256: str
    dataset_sha256: str
    checkpoint_manifest: str | None = None
    resumed_from_step: int = 0
    duration_seconds: float
    blockers: list[str] = Field(default_factory=list)


class EmbeddingCheckpointState(StrictModel):
    schema_version: Literal[1] = 1
    step: int = Field(ge=0)
    best_step: int = Field(ge=0)
    best_validation_loss: float | None = None
    stale_validations: int = Field(default=0, ge=0)
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_random_state: str


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[EmbeddingPair]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            payload = (
                json.dumps(row.model_dump(mode="json"), sort_keys=True, ensure_ascii=True) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, digest.hexdigest()


def build_embedding_pairs(
    store: LocalStore,
    *,
    project_id: str,
    output_path: str | Path,
    minimum_pairs: int = 1,
) -> EmbeddingPairBuildReport:
    """Build evidence-bound retrieval pairs from a committed repository index."""

    project = store.get_project(project_id)
    if project is None:
        raise KeyError(f"unknown project: {project_id}")
    revision = store.active_index_revision(project_id)
    if revision is None:
        raise RuntimeError("embedding pairs require a committed index revision")
    chunks = store.list_chunks(project_id)
    by_id = {str(chunk["id"]): chunk for chunk in chunks}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        symbol = str(chunk.get("symbol_name") or "")
        if symbol:
            by_symbol.setdefault(symbol.lower(), []).append(chunk)
    pairs: dict[str, EmbeddingPair] = {}

    def add(
        query: str, positive: str, category: str, lineage: str, metadata: dict[str, Any]
    ) -> None:
        if not query.strip() or not positive.strip():
            return
        identity = hashlib.sha256(
            "\x1f".join([str(revision["id"]), category, lineage, query, positive]).encode(
                "utf-8", "surrogatepass"
            )
        ).hexdigest()
        candidates = [
            str(item["content"])
            for item in chunks
            if str(item["id"]) != str(metadata.get("chunk_id"))
            and item.get("language") == metadata.get("language")
            and str(item.get("path")) != str(metadata.get("path"))
        ]
        candidates.sort(
            key=lambda text: hashlib.sha256((identity + text).encode("utf-8")).hexdigest()
        )
        pairs[identity] = EmbeddingPair(
            pair_id=identity,
            query=query,
            positive=positive,
            hard_negatives=candidates[:2],
            category=category,  # type: ignore[arg-type]
            source_revision=str(revision["source_revision"]),
            lineage_key=lineage,
            verified=True,
            metadata={**metadata, "index_revision": str(revision["id"])},
        )

    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        symbol = str(chunk.get("symbol_name") or "")
        path = str(chunk["path"])
        lineage = f"{path}:{symbol or chunk['start_line']}"
        signature = str(metadata.get("signature") or symbol)
        docstring = str(metadata.get("docstring") or "").strip()
        if symbol and docstring:
            add(
                f"Find the implementation described as: {docstring}",
                chunk_search_text(chunk),
                "function_doc",
                lineage,
                {"chunk_id": chunk["id"], "path": path, "language": chunk.get("language")},
            )
        if symbol:
            add(
                f"Locate symbol {signature} in {path}",
                chunk_search_text(chunk),
                "task_evidence",
                lineage,
                {"chunk_id": chunk["id"], "path": path, "language": chunk.get("language")},
            )
        for edge in metadata.get("resolved_calls", []):
            if not isinstance(edge, dict):
                continue
            target = by_id.get(str(edge.get("target_chunk_id") or ""))
            if target is None:
                continue
            add(
                f"Find the implementation called by {symbol or path}: {edge.get('call')}",
                chunk_search_text(target),
                "symbol_call",
                f"{lineage}->{target['path']}:{target.get('symbol_name')}",
                {
                    "chunk_id": target["id"],
                    "path": target["path"],
                    "language": target.get("language"),
                },
            )
        if "test" in path.lower() or "spec" in path.lower():
            for called in metadata.get("calls", []):
                targets = by_symbol.get(str(called).split(".")[-1].lower(), [])
                if len(targets) == 1:
                    target = targets[0]
                    add(
                        f"Find code covered by test {symbol or path}",
                        chunk_search_text(target),
                        "code_test",
                        f"{path}->{target['path']}:{target.get('symbol_name')}",
                        {
                            "chunk_id": target["id"],
                            "path": target["path"],
                            "language": target.get("language"),
                        },
                    )

    ordered = [pairs[key] for key in sorted(pairs)]
    path = Path(output_path).resolve()
    count, digest = _atomic_jsonl(path, ordered)
    categories = Counter(pair.category for pair in ordered)
    blockers = (
        [] if count >= minimum_pairs else [f"pair count {count} is below required {minimum_pairs}"]
    )
    return EmbeddingPairBuildReport(
        status="blocked" if blockers else "passed",
        project_id=project_id,
        index_revision=str(revision["id"]),
        pair_count=count,
        categories=dict(sorted(categories.items())),
        output_path=str(path),
        output_sha256=digest,
        blockers=blockers,
    )


def load_embedding_pairs(path: str | Path) -> tuple[list[EmbeddingPair], str]:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    pairs: list[EmbeddingPair] = []
    seen: set[str] = set()
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                pair = EmbeddingPair.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(f"invalid embedding pair at line {line_number}") from exc
            if not pair.verified:
                raise ValueError(f"unverified embedding pair at line {line_number}")
            if pair.pair_id in seen:
                raise ValueError(f"duplicate embedding pair ID at line {line_number}")
            seen.add(pair.pair_id)
            pairs.append(pair)
    if not pairs:
        raise ValueError("embedding pair dataset is empty")
    return pairs, digest.hexdigest()


def split_embedding_pairs(
    pairs: list[EmbeddingPair],
    *,
    validation_fraction: float,
) -> tuple[list[EmbeddingPair], list[EmbeddingPair]]:
    validation: list[EmbeddingPair] = []
    training: list[EmbeddingPair] = []
    threshold = int(validation_fraction * 10_000)
    for pair in pairs:
        bucket = int(hashlib.sha256(pair.lineage_key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        (validation if bucket < threshold else training).append(pair)
    if not validation and len(training) > 1:
        validation.append(training.pop())
    if not training or not validation:
        raise ValueError(
            "embedding dataset cannot produce non-empty lineage-safe train and validation splits"
        )
    train_lineages = {pair.lineage_key for pair in training}
    if train_lineages.intersection(pair.lineage_key for pair in validation):
        raise RuntimeError("embedding lineage leaked across train and validation")
    return training, validation


def _tokenize_embedding_texts(
    tokenizer: Any, texts: list[str], *, max_length: int, device: Any
) -> tuple[Any, Any]:
    encoded = [tokenizer.encode(text).ids[:max_length] for text in texts]
    if not encoded or any(not item for item in encoded):
        raise ValueError("tokenizer produced an empty embedding sequence")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = 0
    width = max(len(item) for item in encoded)
    ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for row, values in enumerate(encoded):
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = 1
    return ids, mask


def _save_optimizer_safely(optimizer: Any, output_dir: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for optimizer checkpoints") from exc
    state = optimizer.state_dict()
    tensors: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    for parameter_id, values in state["state"].items():
        for name, value in values.items():
            key = f"state.{parameter_id}.{name}"
            if torch.is_tensor(value):
                tensors[key] = value.detach().cpu().contiguous()
            elif isinstance(value, (bool, int, float, str)) or value is None:
                scalars[key] = value
            else:
                raise TypeError(f"unsupported optimizer state value: {key}")
    save_file(
        tensors, str(output_dir / "optimizer.safetensors"), metadata={"format": "craftly-adamw-v1"}
    )
    (output_dir / "optimizer.json").write_text(
        json.dumps(
            {"param_groups": state["param_groups"], "scalars": scalars}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def _load_optimizer_safely(optimizer: Any, output_dir: Path) -> None:
    from safetensors.torch import load_file

    metadata = json.loads((output_dir / "optimizer.json").read_text(encoding="utf-8"))
    tensors = load_file(str(output_dir / "optimizer.safetensors"), device="cpu")
    state: dict[int, dict[str, Any]] = {}
    for key, value in {**metadata.get("scalars", {}), **tensors}.items():
        _, parameter_id, name = key.split(".", 2)
        state.setdefault(int(parameter_id), {})[name] = value
    optimizer.load_state_dict({"state": state, "param_groups": metadata["param_groups"]})


def evaluate_embedding_model(
    model: ScratchCodeEmbeddingModel,
    tokenizer: Any,
    pairs: list[EmbeddingPair],
    *,
    device: Any,
    max_pairs: int = 2000,
) -> EmbeddingRetrievalMetrics:
    selected = pairs[:max_pairs]
    model.eval()
    query_vectors: list[Any] = []
    positive_vectors: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            batch = selected[start : start + 64]
            query_ids, query_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.query for item in batch],
                max_length=model.config.max_sequence_length,
                device=device,
            )
            positive_ids, positive_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.positive for item in batch],
                max_length=model.config.max_sequence_length,
                device=device,
            )
            query_vectors.append(model(query_ids, query_mask).cpu())
            positive_vectors.append(model(positive_ids, positive_mask).cpu())
    queries = torch.cat(query_vectors)
    positives = torch.cat(positive_vectors)
    similarities = queries @ positives.transpose(0, 1)
    order = similarities.argsort(dim=1, descending=True)
    ranks = []
    for index in range(len(selected)):
        rank = int((order[index] == index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    diagonal = similarities.diag()
    off_diagonal = (
        similarities[~torch.eye(len(selected), dtype=torch.bool)]
        if len(selected) > 1
        else torch.zeros(1)
    )
    dimension_std = float(torch.cat([queries, positives]).std(dim=0).mean().item())
    off_mean = float(off_diagonal.mean().item())
    collapse = dimension_std < 1e-4 or off_mean > 0.98 or not math.isfinite(dimension_std)
    return EmbeddingRetrievalMetrics(
        pair_count=len(selected),
        recall_at_1=sum(rank <= 1 for rank in ranks) / len(ranks),
        recall_at_5=sum(rank <= 5 for rank in ranks) / len(ranks),
        recall_at_20=sum(rank <= 20 for rank in ranks) / len(ranks),
        mrr_at_10=sum((1.0 / rank) if rank <= 10 else 0.0 for rank in ranks) / len(ranks),
        mean_positive_similarity=float(diagonal.mean().item()),
        mean_off_diagonal_similarity=off_mean,
        embedding_dimension_std=dimension_std,
        collapse_detected=collapse,
    )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _load_embedding_checkpoint(
    model: ScratchCodeEmbeddingModel,
    checkpoint_dir: Path,
    *,
    tokenizer_sha256: str,
    dataset_sha256: str,
) -> EmbeddingCheckpointState:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for secure checkpoint resume") from exc
    manifest_path = checkpoint_dir / "embedding_manifest.json"
    state_path = checkpoint_dir / "training_state.json"
    weights_path = checkpoint_dir / "embedding_model.safetensors"
    for path in (
        manifest_path,
        state_path,
        weights_path,
        checkpoint_dir / "optimizer.json",
        checkpoint_dir / "optimizer.safetensors",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"resume checkpoint is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = EmbeddingCheckpointState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if (
        manifest.get("tokenizer_sha256") != tokenizer_sha256
        or state.tokenizer_sha256 != tokenizer_sha256
    ):
        raise RuntimeError("resume checkpoint tokenizer hash mismatch")
    if (
        manifest.get("dataset_manifest_sha256") != dataset_sha256
        or state.dataset_sha256 != dataset_sha256
    ):
        raise RuntimeError("resume checkpoint embedding dataset hash mismatch")
    actual = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if actual != manifest.get("checkpoint_sha256"):
        raise RuntimeError("resume checkpoint weight checksum mismatch")
    expected_config = model.config.model_dump(mode="json")
    if manifest.get("config") != expected_config:
        raise RuntimeError("resume checkpoint model configuration mismatch")
    incompatible = model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("resume checkpoint model state is incompatible")
    return state


def _save_training_checkpoint(
    model: ScratchCodeEmbeddingModel,
    optimizer: Any,
    output_dir: Path,
    *,
    state: EmbeddingCheckpointState,
) -> dict[str, Any]:
    manifest = save_scratch_embedding_checkpoint(
        model,
        output_dir,
        tokenizer_sha256=state.tokenizer_sha256,
        dataset_manifest_sha256=state.dataset_sha256,
    )
    _save_optimizer_safely(optimizer, output_dir)
    _atomic_json(output_dir / "training_state.json", state.model_dump(mode="json"))
    return manifest


def _learning_rate(step: int, config: EmbeddingTrainingConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    decay_steps = max(1, config.steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _validation_loss(
    model: ScratchCodeEmbeddingModel,
    tokenizer: Any,
    pairs: list[EmbeddingPair],
    *,
    device: Any,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    observations = 0
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if len(batch) < 2:
                continue
            query_ids, query_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.query for item in batch],
                max_length=model.config.max_sequence_length,
                device=device,
            )
            positive_ids, positive_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.positive for item in batch],
                max_length=model.config.max_sequence_length,
                device=device,
            )
            loss = model.contrastive_loss(query_ids, query_mask, positive_ids, positive_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("embedding validation produced non-finite loss")
            total += float(loss.item()) * len(batch)
            observations += len(batch)
    if not observations:
        raise RuntimeError("embedding validation requires at least one batch with two pairs")
    return total / observations


def _training_batch(
    pairs: list[EmbeddingPair],
    *,
    batch_size: int,
    cursor: int,
    hard_negatives_per_query: int,
) -> tuple[list[EmbeddingPair], list[list[str]], int]:
    selected = [pairs[(cursor + offset) % len(pairs)] for offset in range(batch_size)]
    negatives: list[list[str]] = []
    for row, pair in enumerate(selected):
        values = list(pair.hard_negatives[:hard_negatives_per_query])
        offset = 1
        while len(values) < hard_negatives_per_query:
            candidate = selected[(row + offset) % len(selected)].positive
            offset += 1
            if candidate != pair.positive and candidate not in values:
                values.append(candidate)
            if offset > len(selected) * 2:
                raise RuntimeError("training batch cannot construct distinct hard negatives")
        negatives.append(values)
    return selected, negatives, (cursor + batch_size) % len(pairs)


def train_scratch_embedding_model(
    *,
    pairs_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    config: EmbeddingTrainingConfig,
    resume_from: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> EmbeddingTrainingReport:
    """Train Craftly-Code-Embed from random initialization with evidence-bound resume.

    The function intentionally accepts only the Craftly pair schema and a local
    tokenizer artifact. It never downloads or initializes borrowed weights.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for scratch embedding training")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Hugging Face tokenizers is required for embedding training") from exc

    started = time.monotonic()
    pairs, dataset_sha256 = load_embedding_pairs(pairs_path)
    training_pairs, validation_pairs = split_embedding_pairs(
        pairs,
        validation_fraction=config.validation_fraction,
    )
    validation_pairs = validation_pairs[: config.max_validation_pairs]
    if len(training_pairs) < config.batch_size:
        raise ValueError("embedding training set must contain at least one full batch")
    tokenizer_file = Path(tokenizer_path).resolve(strict=True)
    tokenizer_sha256 = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab_size != config.model.vocab_size:
        raise ValueError(
            f"embedding model/tokenizer vocabulary mismatch: {config.model.vocab_size} != {tokenizer_vocab_size}"
        )
    if config.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = config.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA embedding training requested but CUDA is unavailable")
    if config.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 embedding training requires CUDA")
    if config.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 embedding training requires a bf16-capable CUDA device")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = ScratchCodeEmbeddingModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    config_sha256 = _canonical_sha256(config.model_dump(mode="json"))
    step = best_step = stale_validations = 0
    best_validation_loss: float | None = None
    resume_step = 0
    if resume_from is not None:
        resume_dir = Path(resume_from).resolve(strict=True)
        state = _load_embedding_checkpoint(
            model,
            resume_dir,
            tokenizer_sha256=tokenizer_sha256,
            dataset_sha256=dataset_sha256,
        )
        if state.config_sha256 != config_sha256:
            raise RuntimeError("resume checkpoint training configuration mismatch")
        _load_optimizer_safely(optimizer, resume_dir)
        random.setstate(_nested_tuple(json.loads(state.python_random_state)))
        step = resume_step = state.step
        best_step = state.best_step
        best_validation_loss = state.best_validation_loss
        stale_validations = state.stale_validations
        if step >= config.steps:
            raise ValueError("resume checkpoint has already reached the requested training steps")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cursor = (step * config.batch_size * config.gradient_accumulation_steps) % len(training_pairs)
    shuffled = list(training_pairs)
    random.Random(config.seed + step).shuffle(shuffled)
    use_autocast = device.type == "cuda" and config.precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    last_loss: float | None = None
    last_metrics: EmbeddingRetrievalMetrics | None = None
    stopped_early = False

    while step < config.steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch, negatives, cursor = _training_batch(
                shuffled,
                batch_size=config.batch_size,
                cursor=cursor,
                hard_negatives_per_query=config.hard_negatives_per_query,
            )
            query_ids, query_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.query for item in batch],
                max_length=config.model.max_sequence_length,
                device=device,
            )
            positive_ids, positive_mask = _tokenize_embedding_texts(
                tokenizer,
                [item.positive for item in batch],
                max_length=config.model.max_sequence_length,
                device=device,
            )
            hard_ids = hard_mask = None
            if config.hard_negatives_per_query:
                flat = [value for row in negatives for value in row]
                flat_ids, flat_mask = _tokenize_embedding_texts(
                    tokenizer, flat, max_length=config.model.max_sequence_length, device=device
                )
                hard_ids = flat_ids.reshape(config.batch_size, config.hard_negatives_per_query, -1)
                hard_mask = flat_mask.reshape(
                    config.batch_size, config.hard_negatives_per_query, -1
                )
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
            ):
                loss = model.training_loss(
                    query_ids,
                    query_mask,
                    positive_ids,
                    positive_mask,
                    hard_negative_ids=hard_ids,
                    hard_negative_mask=hard_mask,
                )
                scaled_loss = loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite embedding loss at optimizer step {step + 1}")
            scaler.scale(scaled_loss).backward()
            accumulated += float(loss.detach().item())
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite embedding gradient at optimizer step {step + 1}")
        scaler.step(optimizer)
        scaler.update()
        step += 1
        lr = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        last_loss = accumulated / config.gradient_accumulation_steps
        event = {
            "event": "craftly_embedding_train",
            "step": step,
            "max_steps": config.steps,
            "loss": last_loss,
            "learning_rate": lr,
            "gradient_norm": float(gradient_norm),
            "device": str(device),
        }
        if progress is not None:
            progress(event)

        should_validate = step % config.validation_interval == 0 or step == config.steps
        if should_validate:
            validation_loss = _validation_loss(
                model,
                tokenizer,
                validation_pairs,
                device=device,
                batch_size=config.batch_size,
            )
            last_metrics = evaluate_embedding_model(
                model,
                tokenizer,
                validation_pairs,
                device=device,
                max_pairs=config.max_validation_pairs,
            )
            improved = best_validation_loss is None or validation_loss < best_validation_loss - 1e-6
            if improved:
                best_validation_loss = validation_loss
                best_step = step
                stale_validations = 0
            else:
                stale_validations += 1
            if progress is not None:
                progress(
                    {
                        **event,
                        "event": "craftly_embedding_validation",
                        "validation_loss": validation_loss,
                        "recall_at_20": last_metrics.recall_at_20,
                        "mrr_at_10": last_metrics.mrr_at_10,
                        "collapse_detected": last_metrics.collapse_detected,
                        "improved": improved,
                    }
                )
            state = EmbeddingCheckpointState(
                step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                stale_validations=stale_validations,
                tokenizer_sha256=tokenizer_sha256,
                dataset_sha256=dataset_sha256,
                config_sha256=config_sha256,
                python_random_state=json.dumps(random.getstate(), separators=(",", ":")),
            )
            _save_training_checkpoint(model, optimizer, root / "latest", state=state)
            if improved:
                _save_training_checkpoint(model, optimizer, root / "best", state=state)
            if stale_validations >= config.early_stopping_patience:
                stopped_early = True
                break
        elif step % config.checkpoint_interval == 0:
            state = EmbeddingCheckpointState(
                step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                stale_validations=stale_validations,
                tokenizer_sha256=tokenizer_sha256,
                dataset_sha256=dataset_sha256,
                config_sha256=config_sha256,
                python_random_state=json.dumps(random.getstate(), separators=(",", ":")),
            )
            _save_training_checkpoint(model, optimizer, root / "latest", state=state)

    blockers: list[str] = []
    if best_validation_loss is None:
        blockers.append("no validation checkpoint was produced")
    if last_metrics is None:
        blockers.append("no retrieval evaluation was produced")
    elif last_metrics.collapse_detected:
        blockers.append("embedding collapse detector failed")
    best_manifest = root / "best" / "embedding_manifest.json"
    if not best_manifest.is_file():
        blockers.append("best checkpoint manifest is missing")
    report = EmbeddingTrainingReport(
        status="failed" if blockers else "passed",
        steps_completed=step,
        best_step=best_step,
        best_validation_loss=best_validation_loss,
        train_loss_last=last_loss,
        metrics=last_metrics,
        tokenizer_sha256=tokenizer_sha256,
        dataset_sha256=dataset_sha256,
        checkpoint_manifest=str(best_manifest) if best_manifest.is_file() else None,
        resumed_from_step=resume_step,
        duration_seconds=round(time.monotonic() - started, 3),
        blockers=blockers,
    )
    _atomic_json(root / "embedding_training_report.json", report.model_dump(mode="json"))
    markdown = [
        "# Craftly Scratch Embedding Training",
        "",
        f"- Status: `{report.status}`",
        f"- Steps: `{report.steps_completed}`",
        f"- Best step: `{report.best_step}`",
        f"- Best validation loss: `{report.best_validation_loss}`",
        f"- Early stop: `{stopped_early}`",
        f"- Tokenizer SHA-256: `{report.tokenizer_sha256}`",
        f"- Dataset SHA-256: `{report.dataset_sha256}`",
    ]
    if report.metrics is not None:
        markdown.extend(
            [
                f"- Recall@20: `{report.metrics.recall_at_20:.6f}`",
                f"- MRR@10: `{report.metrics.mrr_at_10:.6f}`",
                f"- Collapse detected: `{report.metrics.collapse_detected}`",
            ]
        )
    if report.blockers:
        markdown.extend(["", "## Blockers", *[f"- {item}" for item in report.blockers]])
    (root / "embedding_training_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


class VectorBackendConfig(StrictModel):
    backend: VectorBackendName = "local_hashing"
    dims: int = Field(default=384, ge=64, le=4096)
    qdrant_url: str | None = None
    qdrant_collection: str = "craftly_code_chunks"
    postgres_dsn: str | None = None
    hnsw_space: str = "cosine"
    embedding_url: str | None = None
    embedding_model: str = "Craftly-Code-Embed-v1"
    embedding_manifest_path: str | None = None
    production_mode: bool = False
    qdrant_alias: str = "craftly-rag-current"
    qdrant_expected_points: int = Field(default=100_000_000, ge=1)
    qdrant_replication_factor: int = Field(default=2, ge=1, le=9)


class VectorIndexCapability(StrictModel):
    backend: VectorBackendName
    available: bool
    reason: str
    production_grade: bool
    notes: list[str] = Field(default_factory=list)


class VectorSearchResult(StrictModel):
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    score: float
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchReport(StrictModel):
    organization_id: str = "local"
    project_id: str
    revision_id: str = ""
    query: str
    backend: VectorBackendName = "local_hashing"
    dims: int = 384
    results: list[VectorSearchResult]


class VectorSyncReport(StrictModel):
    organization_id: str = "local"
    project_id: str
    revision_id: str = ""
    backend: VectorBackendName
    collection: str
    indexed_chunks: int = Field(ge=0)
    deleted_stale_chunks: int = Field(ge=0)
    embedding_dimensions: int = Field(ge=1)
    revision_sha256: str


class VectorIndexBackend(Protocol):
    config: VectorBackendConfig

    def search(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        query: str,
        top_k: int = 12,
    ) -> VectorSearchReport: ...

    def sync_project(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        batch_size: int = 64,
    ) -> VectorSyncReport: ...


class EmbeddingProvider(Protocol):
    dims: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashingEmbeddingProvider:
    def __init__(self, *, dims: int = 384) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        return hashed_embedding(text, dims=self.dims)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class HttpEmbeddingProvider:
    """Production embedding provider contract.

    The endpoint must return either {"embedding": [...]} or OpenAI-style
    {"data": [{"embedding": [...]}]}. Missing or malformed vectors fail fast.
    """

    def __init__(self, *, endpoint: str, model: str, dims: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.dims = dims
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("embedding endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("embedding endpoint must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("embedding endpoint must not contain a query string or fragment")
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("remote embedding endpoints must use HTTPS")
        self._lock = threading.Lock()
        self._failure_count = 0
        self._circuit_open_until = 0.0
        headers = {"User-Agent": "Craftly-RAG/1"}
        api_key = os.environ.get("CRAFTLY_EMBEDDING_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        cert: str | tuple[str, str] | None = None
        client_cert = os.environ.get("CRAFTLY_EMBEDDING_CLIENT_CERT")
        client_key = os.environ.get("CRAFTLY_EMBEDDING_CLIENT_KEY")
        if bool(client_cert) != bool(client_key):
            raise ValueError("embedding mTLS requires both client certificate and key")
        if client_cert and client_key:
            cert = (client_cert, client_key)
        self.client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
            verify=os.environ.get("CRAFTLY_EMBEDDING_CA_BUNDLE", True),
            cert=cert,
            headers=headers,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts or len(texts) > 256:
            raise ValueError("embedding batch size must be between 1 and 256")
        if any(
            not isinstance(text, str) or not text or len(text.encode("utf-8")) > 2_000_000
            for text in texts
        ):
            raise ValueError("embedding inputs must be non-empty strings no larger than 2MB")
        with self._lock:
            if time.monotonic() < self._circuit_open_until:
                raise RuntimeError("embedding provider circuit is open")
        request_input: str | list[str] = texts[0] if len(texts) == 1 else texts
        try:
            response = self.client.post(
                self.endpoint, json={"model": self.model, "input": request_input}
            )
            response.raise_for_status()
            if len(response.content) > 64 * 1024 * 1024:
                raise RuntimeError("embedding provider response exceeded 64MB")
            payload = response.json()
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            with self._lock:
                self._failure_count += 1
                if self._failure_count >= 3:
                    self._circuit_open_until = time.monotonic() + 30.0
            raise RuntimeError(f"embedding provider request failed: {type(exc).__name__}") from exc
        if len(texts) == 1 and isinstance(payload.get("embedding"), list):
            vectors = [payload["embedding"]]
        else:
            data = payload.get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise RuntimeError("embedding provider returned an invalid batch")
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors = [item.get("embedding") for item in ordered]
        validated = [_validated_embedding(vector, dims=self.dims) for vector in vectors]
        with self._lock:
            self._failure_count = 0
            self._circuit_open_until = 0.0
        return validated


def validate_embedding_manifest(config: VectorBackendConfig) -> dict[str, Any]:
    path_value = config.embedding_manifest_path or os.environ.get("CRAFTLY_EMBEDDING_MANIFEST")
    if not path_value:
        if config.production_mode:
            raise RuntimeError("production embeddings require CRAFTLY_EMBEDDING_MANIFEST")
        return {"status": "not_bound", "scratch_only": False}
    path = os.path.realpath(os.path.expanduser(path_value))
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("embedding model manifest is unreadable") from exc
    if manifest.get("scratch_only") is not True or manifest.get("borrowed_weights") is not False:
        raise RuntimeError("embedding manifest does not prove Craftly scratch-only weights")
    if str(manifest.get("model_name")) != config.embedding_model:
        raise RuntimeError("embedding manifest model name mismatch")
    if int(manifest.get("projection_dimension", 0)) != config.dims:
        raise RuntimeError("embedding manifest dimension mismatch")
    tokenizer_hash = str(manifest.get("tokenizer_sha256") or "")
    checkpoint_hash = str(manifest.get("checkpoint_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", tokenizer_hash) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_hash
    ):
        raise RuntimeError("embedding manifest requires tokenizer and checkpoint SHA-256 bindings")
    return manifest


def create_embedding_provider(
    config: VectorBackendConfig, *, allow_local_hashing: bool
) -> EmbeddingProvider:
    endpoint = config.embedding_url or os.environ.get("CRAFTLY_EMBEDDING_URL")
    if endpoint:
        validate_embedding_manifest(config)
        return HttpEmbeddingProvider(
            endpoint=endpoint, model=config.embedding_model, dims=config.dims
        )
    if allow_local_hashing:
        return LocalHashingEmbeddingProvider(dims=config.dims)
    raise RuntimeError("production vector backend requires embedding_url/CRAFTLY_EMBEDDING_URL")


def _validated_embedding(value: Any, *, dims: int) -> list[float]:
    if not isinstance(value, list) or len(value) != dims:
        raise RuntimeError(
            f"embedding provider returned invalid vector dimensions; expected {dims}"
        )
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise RuntimeError("embedding provider returned a non-finite vector")
    return vector


def text_terms(text: str) -> Counter[str]:
    return Counter(term.lower() for term in TOKEN_RE.findall(text))


def hashed_embedding(text: str, *, dims: int = 384) -> list[float]:
    vector = [0.0] * dims
    terms = text_terms(text)
    for term, count in terms.items():
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def chunk_search_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return " ".join(
        [
            str(chunk.get("path") or ""),
            str(chunk.get("language") or ""),
            str(chunk.get("kind") or ""),
            str(chunk.get("symbol_name") or ""),
            str(metadata.get("signature") or ""),
            " ".join(str(item) for item in metadata.get("imports", []) if item),
            " ".join(str(item) for item in metadata.get("calls", []) if item),
            " ".join(str(item) for item in metadata.get("dependencies", []) if item),
            str(chunk.get("content") or ""),
        ]
    )


class LocalVectorIndex:
    """Deterministic local vector index.

    This backend is a deterministic development fallback for small repositories.
    It is not semantic enough or scalable enough for production retrieval.
    """

    def __init__(
        self,
        store: LocalStore | None = None,
        *,
        dims: int = 384,
        config: VectorBackendConfig | None = None,
    ) -> None:
        self.store = store or LocalStore()
        self.config = config or VectorBackendConfig(backend="local_hashing", dims=dims)
        self.dims = self.config.dims
        self.embedding_provider = create_embedding_provider(self.config, allow_local_hashing=True)

    def search(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        query: str,
        top_k: int = 12,
    ) -> VectorSearchReport:
        self.store.require_project_access(project_id, organization_id)
        query_vector = self.embedding_provider.embed(query)
        scored: list[VectorSearchResult] = []
        for chunk in self.store.list_chunks(project_id):
            score = cosine(query_vector, self.embedding_provider.embed(chunk_search_text(chunk)))
            if score <= 0:
                continue
            scored.append(
                VectorSearchResult(
                    chunk_id=chunk["id"],
                    path=chunk["path"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    symbol_name=chunk.get("symbol_name"),
                    score=round(score, 6),
                    content=chunk["content"],
                    metadata=chunk.get("metadata") or {},
                )
            )
        return VectorSearchReport(
            organization_id=organization_id,
            project_id=project_id,
            revision_id=revision_id,
            query=query,
            backend=self.config.backend,
            dims=self.dims,
            results=sorted(scored, key=lambda item: item.score, reverse=True)[:top_k],
        )

    def sync_project(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        batch_size: int = 64,
    ) -> VectorSyncReport:
        self.store.require_project_access(project_id, organization_id)
        if batch_size < 1 or batch_size > 256:
            raise ValueError("vector sync batch_size must be between 1 and 256")
        chunks = self.store.list_chunks(project_id)
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(str(chunk["id"]).encode("utf-8"))
            digest.update(hashlib.sha256(chunk_search_text(chunk).encode("utf-8")).digest())
        return VectorSyncReport(
            organization_id=organization_id,
            project_id=project_id,
            revision_id=revision_id,
            backend=self.config.backend,
            collection="local-exact-scan",
            indexed_chunks=len(chunks),
            deleted_stale_chunks=0,
            embedding_dimensions=self.dims,
            revision_sha256=digest.hexdigest(),
        )


class FaissVectorIndex(LocalVectorIndex):
    """FAISS adapter contract.

    The current repository does not persist a FAISS sidecar yet. This class
    validates dependency availability and falls back to exact local scoring only
    when explicitly allowed by using `local_hashing`.
    """

    def __init__(
        self, store: LocalStore | None = None, *, config: VectorBackendConfig | None = None
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("FAISS backend requested but faiss is not installed") from exc
        super().__init__(store, config=config or VectorBackendConfig(backend="faiss"))


class HnswVectorIndex(LocalVectorIndex):
    def __init__(
        self, store: LocalStore | None = None, *, config: VectorBackendConfig | None = None
    ) -> None:
        try:
            import hnswlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("HNSW backend requested but hnswlib is not installed") from exc
        super().__init__(store, config=config or VectorBackendConfig(backend="hnsw"))


class QdrantVectorIndex(LocalVectorIndex):
    def __init__(
        self, store: LocalStore | None = None, *, config: VectorBackendConfig | None = None
    ) -> None:
        active = config or VectorBackendConfig(backend="qdrant")
        active = active.model_copy(
            update={
                "qdrant_url": active.qdrant_url or os.environ.get("CRAFTLY_QDRANT_URL"),
                "embedding_url": active.embedding_url or os.environ.get("CRAFTLY_EMBEDDING_URL"),
            }
        )
        if active.qdrant_collection == "craftly_code_chunks":
            version_suffix = hashlib.sha256(active.embedding_model.encode("utf-8")).hexdigest()[:12]
            active = active.model_copy(
                update={"qdrant_collection": f"craftly_code_chunks_{version_suffix}"}
            )
        if not active.qdrant_url:
            raise RuntimeError(
                "Qdrant backend requested but qdrant_url/CRAFTLY_QDRANT_URL is not configured"
            )
        self.embedding_provider = create_embedding_provider(active, allow_local_hashing=False)
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant backend requested but qdrant-client is not installed"
            ) from exc
        self.store = store or LocalStore()
        self.config = active
        self.dims = active.dims
        self.client = QdrantClient(url=active.qdrant_url)

    def _promote_alias(self) -> None:
        if not self.config.production_mode:
            return
        if not hasattr(self.client, "update_collection_aliases"):
            raise RuntimeError("Qdrant client does not support atomic alias promotion")
        from qdrant_client import models

        operations: list[Any] = []
        try:
            aliases = self.client.get_aliases().aliases
            if any(
                getattr(item, "alias_name", None) == self.config.qdrant_alias for item in aliases
            ):
                operations.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=self.config.qdrant_alias)
                    )
                )
        except Exception as exc:
            raise RuntimeError(f"Qdrant alias discovery failed: {exc}") from exc
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self.config.qdrant_collection,
                    alias_name=self.config.qdrant_alias,
                )
            )
        )
        try:
            self.client.update_collection_aliases(change_aliases_operations=operations)
        except Exception as exc:
            raise RuntimeError(f"Qdrant alias promotion failed: {exc}") from exc

    @staticmethod
    def _point_id(organization_id: str, project_id: str, revision_id: str, chunk_id: str) -> str:
        del revision_id
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"craftly:{organization_id}:{project_id}:{chunk_id}")
        )

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        try:
            exists = bool(self.client.collection_exists(self.config.qdrant_collection))
        except AttributeError:
            try:
                self.client.get_collection(self.config.qdrant_collection)
                exists = True
            except Exception:
                exists = False
        if not exists:
            shard_count = (
                max(16, min(256, math.ceil(self.config.qdrant_expected_points / 10_000_000)))
                if self.config.production_mode
                else 1
            )
            self.client.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=self.dims,
                    distance=models.Distance.COSINE,
                    on_disk=self.config.production_mode,
                ),
                shard_number=shard_count,
                replication_factor=self.config.qdrant_replication_factor
                if self.config.production_mode
                else 1,
                write_consistency_factor=(self.config.qdrant_replication_factor // 2) + 1
                if self.config.production_mode
                else 1,
            )
        for field_name in ("organization_id", "project_id", "revision_id", "content_hash"):
            try:
                self.client.create_payload_index(
                    collection_name=self.config.qdrant_collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise RuntimeError(
                        f"Qdrant payload index creation failed for {field_name}: {exc}"
                    ) from exc

    @staticmethod
    def _tenant_filter(models: Any, organization_id: str, project_id: str, revision_id: str) -> Any:
        must = [
            models.FieldCondition(
                key="organization_id", match=models.MatchValue(value=organization_id)
            ),
            models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
        ]
        if revision_id:
            must.append(
                models.FieldCondition(key="revision_id", match=models.MatchValue(value=revision_id))
            )
        return models.Filter(must=must)

    def sync_project(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        batch_size: int = 64,
    ) -> VectorSyncReport:
        from qdrant_client import models

        if batch_size < 1 or batch_size > 256:
            raise ValueError("Qdrant batch_size must be between 1 and 256")
        project = self.store.require_project_access(project_id, organization_id)
        active_revision = str(project.get("active_index_revision") or "")
        revision_id = revision_id or active_revision
        if not revision_id or revision_id != active_revision:
            raise RuntimeError("Qdrant sync requires the active committed index revision")
        chunks = self.store.list_chunks(project_id)
        if any(str(chunk.get("index_revision") or "") != revision_id for chunk in chunks):
            raise RuntimeError(
                "local chunks are not consistently bound to the active index revision"
            )
        sensitive = [
            str(chunk["id"])
            for chunk in chunks
            if SECRET_RE.search(str(chunk.get("content") or ""))
            or "-----BEGIN PRIVATE KEY-----" in str(chunk.get("content") or "")
        ]
        if sensitive:
            raise RuntimeError(f"secret policy blocked embedding for {len(sensitive)} chunk(s)")
        self._ensure_collection()
        digest = hashlib.sha256()
        desired_ids: set[str] = set()
        for chunk in chunks:
            digest.update(str(chunk["id"]).encode("utf-8"))
            digest.update(str(chunk.get("chunk_hash") or "").encode("ascii"))
        existing_hashes: dict[str, str] = {}
        existing_ids: set[str] = set()
        offset: Any = None
        project_filter = self._tenant_filter(models, organization_id, project_id, "")
        while True:
            records, offset = self.client.scroll(
                collection_name=self.config.qdrant_collection,
                scroll_filter=project_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                point_id = str(record.id)
                existing_ids.add(point_id)
                payload = dict(getattr(record, "payload", {}) or {})
                existing_hashes[point_id] = str(payload.get("content_hash") or "")
            if offset is None:
                break
        reused_ids: list[str] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            changed_batch: list[dict[str, Any]] = []
            for chunk in batch:
                point_id = self._point_id(
                    organization_id, project_id, revision_id, str(chunk["id"])
                )
                desired_ids.add(point_id)
                if existing_hashes.get(point_id) == str(chunk.get("chunk_hash") or ""):
                    reused_ids.append(point_id)
                else:
                    changed_batch.append(chunk)
            if not changed_batch:
                continue
            texts = [chunk_search_text(chunk) for chunk in changed_batch]
            vectors = self.embedding_provider.embed_many(texts)
            if len(vectors) != len(changed_batch):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            points = []
            for chunk, text, vector in zip(changed_batch, texts, vectors, strict=True):
                point_id = self._point_id(
                    organization_id, project_id, revision_id, str(chunk["id"])
                )
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=_validated_embedding(vector, dims=self.dims),
                        payload={
                            "organization_id": organization_id,
                            "project_id": project_id,
                            "revision_id": revision_id,
                            "chunk_id": str(chunk["id"]),
                            "path": str(chunk["path"]),
                            "start_line": int(chunk["start_line"]),
                            "end_line": int(chunk["end_line"]),
                            "symbol_name": chunk.get("symbol_name"),
                            "language": str(chunk.get("language") or ""),
                            "source_kind": "repository",
                            "content_hash": str(chunk.get("chunk_hash") or text_hash),
                        },
                    )
                )
            try:
                self.client.upsert(
                    collection_name=self.config.qdrant_collection,
                    points=points,
                    wait=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Qdrant upsert failed: {exc}") from exc
        if reused_ids:
            if not hasattr(self.client, "set_payload"):
                if self.config.production_mode:
                    raise RuntimeError(
                        "Qdrant client cannot update revision payload for reused vectors"
                    )
            else:
                self.client.set_payload(
                    collection_name=self.config.qdrant_collection,
                    payload={"revision_id": revision_id},
                    points=reused_ids,
                    wait=True,
                )
        stale_ids = sorted(existing_ids - desired_ids)
        if stale_ids:
            self.client.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )
        self._promote_alias()
        return VectorSyncReport(
            organization_id=organization_id,
            project_id=project_id,
            revision_id=revision_id,
            backend="qdrant",
            collection=self.config.qdrant_collection,
            indexed_chunks=len(chunks),
            deleted_stale_chunks=len(stale_ids),
            embedding_dimensions=self.dims,
            revision_sha256=digest.hexdigest(),
        )

    def search(
        self,
        *,
        organization_id: str = "local",
        project_id: str,
        revision_id: str = "",
        query: str,
        top_k: int = 12,
    ) -> VectorSearchReport:
        if top_k < 1 or top_k > 500:
            raise ValueError("top_k must be between 1 and 500")
        project = self.store.require_project_access(project_id, organization_id)
        active_revision = str(project.get("active_index_revision") or "")
        revision_id = revision_id or active_revision
        if not revision_id or revision_id != active_revision:
            raise RuntimeError("Qdrant search requires the active committed index revision")
        vector = self.embedding_provider.embed(query)
        try:
            from qdrant_client import models

            query_filter = self._tenant_filter(models, organization_id, project_id, revision_id)
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.config.qdrant_alias
                    if self.config.production_mode
                    else self.config.qdrant_collection,
                    query=vector,
                    limit=top_k,
                    query_filter=query_filter,
                )
                hits = response.points
            else:
                hits = self.client.search(
                    collection_name=self.config.qdrant_alias
                    if self.config.production_mode
                    else self.config.qdrant_collection,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=query_filter,
                )
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc
        results: list[VectorSearchResult] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or getattr(hit, "id", ""))
            chunk = self.store.get_chunk(chunk_id, project_id=project_id)
            if chunk is None or str(chunk.get("index_revision") or "") != revision_id:
                continue
            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    path=str(chunk.get("path") or ""),
                    start_line=int(chunk.get("start_line") or 0),
                    end_line=int(chunk.get("end_line") or 0),
                    symbol_name=chunk.get("symbol_name"),
                    score=round(float(getattr(hit, "score", 0.0)), 6),
                    content=str(chunk.get("content") or ""),
                    metadata=dict(chunk.get("metadata") or {}),
                )
            )
        return VectorSearchReport(
            organization_id=organization_id,
            project_id=project_id,
            revision_id=revision_id,
            query=query,
            backend="qdrant",
            dims=self.dims,
            results=results,
        )


class PgVectorIndex(LocalVectorIndex):
    def __init__(
        self, store: LocalStore | None = None, *, config: VectorBackendConfig | None = None
    ) -> None:
        active = config or VectorBackendConfig(backend="pgvector")
        active = active.model_copy(
            update={
                "postgres_dsn": active.postgres_dsn or os.environ.get("CRAFTLY_DATABASE_URL"),
                "embedding_url": active.embedding_url or os.environ.get("CRAFTLY_EMBEDDING_URL"),
            }
        )
        if not active.postgres_dsn:
            raise RuntimeError(
                "pgvector backend requested but postgres_dsn/CRAFTLY_DATABASE_URL is not configured"
            )
        create_embedding_provider(active, allow_local_hashing=False)
        super().__init__(store, config=active)


def create_vector_index(
    store: LocalStore | None = None, config: VectorBackendConfig | None = None
) -> VectorIndexBackend:
    active = config or VectorBackendConfig()
    if active.backend == "local_hashing":
        return LocalVectorIndex(store, config=active)
    if active.backend == "faiss":
        return FaissVectorIndex(store, config=active)
    if active.backend == "hnsw":
        return HnswVectorIndex(store, config=active)
    if active.backend == "qdrant":
        return QdrantVectorIndex(store, config=active)
    if active.backend == "pgvector":
        return PgVectorIndex(store, config=active)
    raise ValueError(f"unsupported vector backend: {active.backend}")


def vector_capabilities() -> list[VectorIndexCapability]:
    capabilities: list[VectorIndexCapability] = [
        VectorIndexCapability(
            backend="local_hashing",
            available=True,
            reason="built-in deterministic hashed embeddings",
            production_grade=False,
            notes=[
                "dev and validation fallback only",
                "exact scan",
                "not semantic enough for production",
            ],
        )
    ]
    for backend, package, production_notes in [
        (
            "faiss",
            "faiss",
            ["dependency available", "persistent ANN implementation is not complete"],
        ),
        (
            "hnsw",
            "hnswlib",
            ["dependency available", "persistent ANN implementation is not complete"],
        ),
    ]:
        try:
            __import__(package)
            available = True
            reason = f"{package} installed"
        except ImportError:
            available = False
            reason = f"{package} not installed"
        capabilities.append(
            VectorIndexCapability(
                backend=backend,  # type: ignore[arg-type]
                available=available,
                reason=reason,
                production_grade=False,
                notes=production_notes,
            )
        )
    capabilities.append(
        VectorIndexCapability(
            backend="qdrant",
            available=bool(
                os.environ.get("CRAFTLY_QDRANT_URL") and os.environ.get("CRAFTLY_EMBEDDING_URL")
            ),
            reason="CRAFTLY_QDRANT_URL and CRAFTLY_EMBEDDING_URL configured"
            if os.environ.get("CRAFTLY_QDRANT_URL") and os.environ.get("CRAFTLY_EMBEDDING_URL")
            else "CRAFTLY_QDRANT_URL or CRAFTLY_EMBEDDING_URL not configured",
            production_grade=bool(
                os.environ.get("CRAFTLY_QDRANT_URL") and os.environ.get("CRAFTLY_EMBEDDING_URL")
            ),
            notes=["distributed vector database", "best for long-term memory and many projects"],
        )
    )
    capabilities.append(
        VectorIndexCapability(
            backend="pgvector",
            available=bool(
                os.environ.get("CRAFTLY_DATABASE_URL") and os.environ.get("CRAFTLY_EMBEDDING_URL")
            ),
            reason="CRAFTLY_DATABASE_URL and CRAFTLY_EMBEDDING_URL configured"
            if os.environ.get("CRAFTLY_DATABASE_URL") and os.environ.get("CRAFTLY_EMBEDDING_URL")
            else "CRAFTLY_DATABASE_URL or CRAFTLY_EMBEDDING_URL not configured",
            production_grade=False,
            notes=[
                "configuration contract only",
                "native pgvector persistence/query implementation is not complete",
            ],
        )
    )
    return capabilities


def _embedding_progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True, separators=(",", ":")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and train the Craftly scratch code embedding model"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser(
        "build-pairs", help="Build verified retrieval pairs from a committed local index"
    )
    pairs.add_argument("--sqlite-path", required=True)
    pairs.add_argument("--project-id", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--minimum-pairs", type=int, default=500)
    train = commands.add_parser("train", help="Train Craftly-Code-Embed from random initialization")
    train.add_argument("--pairs", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--resume-from")
    args = parser.parse_args()
    if args.command == "build-pairs":
        with LocalStore(args.sqlite_path) as store:
            report = build_embedding_pairs(
                store,
                project_id=args.project_id,
                output_path=args.output,
                minimum_pairs=args.minimum_pairs,
            )
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        raise SystemExit(0 if report.status == "passed" else 2)
    config_path = Path(args.config).resolve(strict=True)
    config = EmbeddingTrainingConfig.model_validate_json(
        config_path.read_text(encoding="utf-8-sig")
    )
    report = train_scratch_embedding_model(
        pairs_path=args.pairs,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        config=config,
        resume_from=args.resume_from,
        progress=_embedding_progress,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 2)


if __name__ == "__main__":
    main()
