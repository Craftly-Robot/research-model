"""Craftly training workspace control plane.

This module owns the canonical scratch-training job contract, durable state
transitions, live event transport, scheduler adapters, and controller loop.
It deliberately does not accept arbitrary shell commands from API clients.
Kaggle and Colab are validation clients; production jobs are scheduled only
onto trusted Kubernetes or Slurm workers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import socket
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Protocol
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from src.craftly.learning.storage import ObjectStore, ObjectStoreConfig, S3ObjectStore, create_object_store
from src.craftly.model_ops.foundation import (
    TRAINING_STEP_SEMANTICS,
    ParallelismPlan,
    ScratchDecoderConfig,
    TrainingBatchContract,
    model_profile as get_model_profile,
)
from src.craftly.model_ops.production_adapters import (
    megatron_model_arguments,
    resolve_megatron_data_prefix,
)
from src.craftly.shared.config_contracts import ScientificExperimentCampaignContract
from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import canonical_json_text as canonical_json, sha256_file as sha256_path


PROFILE_PATH = Path("config/training_profiles.json")
CAMPAIGN_PATH = Path("config/training_qualification_campaigns.json")
MAX_EVENT_BYTES = 64 * 1024
MAX_EVENT_BATCH = 100
SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
SAFE_GIT_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
SAFE_CONTAINER_DIGEST = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)
TERMINAL_STATES = {"succeeded", "failed", "blocked", "cancelled"}


class JobStatus(str, Enum):
    VALIDATING = "validating"
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.VALIDATING: {JobStatus.QUEUED, JobStatus.BLOCKED, JobStatus.CANCELLED},
    JobStatus.QUEUED: {JobStatus.PROVISIONING, JobStatus.CANCELLED, JobStatus.BLOCKED},
    JobStatus.PROVISIONING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.CHECKPOINTING,
        JobStatus.EVALUATING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.BLOCKED,
        JobStatus.CANCELLED,
    },
    JobStatus.CHECKPOINTING: {JobStatus.RUNNING, JobStatus.EVALUATING, JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED},
    JobStatus.EVALUATING: {JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.BLOCKED, JobStatus.CANCELLED},
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: {JobStatus.QUEUED},
    JobStatus.BLOCKED: set(),
    JobStatus.CANCELLED: {JobStatus.QUEUED},
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def new_ulid() -> str:
    """Return a sortable ULID without adding a runtime dependency."""

    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    output = []
    for _ in range(26):
        output.append(alphabet[value & 31])
        value >>= 5
    return "".join(reversed(output))


def redact_text(value: str) -> str:
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def redact_payload(value: Any, *, key: str = "") -> Any:
    secret_key = re.search(r"(?i)(authorization|api[_-]?key|token|password|secret)", key) is not None
    if secret_key:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): redact_payload(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_payload(item, key=key) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def rank_from_environment() -> tuple[int, int, str]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, world_size, socket.gethostname()


class OverrideRange(StrictModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "OverrideRange":
        if self.maximum < self.minimum:
            raise ValueError("override maximum must be >= minimum")
        return self


class ResourceRequest(StrictModel):
    nodes: int = Field(ge=1, le=1024)
    gpus_per_node: int = Field(ge=0, le=32)
    gpu_memory_gib: int = Field(ge=0, le=1024)
    cpu_cores: int = Field(ge=1, le=1024)
    memory_gib: int = Field(ge=1, le=8192)
    rdma_required: bool = False


class OptimizerPolicy(StrictModel):
    name: Literal["adamw"] = "adamw"
    learning_rate: float = Field(gt=0.0, le=1.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0, le=1e-2)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)


class LearningRatePolicy(StrictModel):
    schedule: Literal["constant", "linear", "cosine"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    minimum_learning_rate_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_warmup(self) -> "LearningRatePolicy":
        if self.warmup_steps and self.warmup_ratio:
            raise ValueError("set either warmup_steps or warmup_ratio, not both")
        return self


class TokenBudgetPolicy(StrictModel):
    target_tokens: int = Field(gt=0)
    maximum_tokens: int = Field(gt=0)
    global_batch_sequences: int = Field(gt=0)
    accounting: Literal["non_padding_tokens"] = "non_padding_tokens"

    @model_validator(mode="after")
    def validate_budget(self) -> "TokenBudgetPolicy":
        if self.maximum_tokens < self.target_tokens:
            raise ValueError("maximum_tokens must be >= target_tokens")
        return self


class CheckpointPolicy(StrictModel):
    interval_steps: int = Field(gt=0)
    keep_last: int = Field(default=3, ge=1, le=100)
    keep_best: int = Field(default=1, ge=1, le=20)
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    require_checksum_manifest: bool = True
    require_reload_verification: bool = True


class EvaluationPolicy(StrictModel):
    interval_steps: int = Field(gt=0)
    validation_batches: int = Field(default=8, ge=1, le=10_000)
    suite_path: str = Field(min_length=1, max_length=2048)
    strict: bool = True
    early_stopping_patience: int = Field(default=0, ge=0, le=1000)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    maximum_validation_regression: float = Field(default=0.03, ge=0.0, le=1.0)


class RetentionPolicy(StrictModel):
    hot_days: int = Field(default=30, ge=1, le=3650)
    archive_days: int = Field(default=365, ge=1, le=36500)
    delete_after_days: int = Field(default=730, ge=1, le=36500)
    preserve_promoted: bool = True

    @model_validator(mode="after")
    def validate_order(self) -> "RetentionPolicy":
        if not self.hot_days <= self.archive_days <= self.delete_after_days:
            raise ValueError("retention must satisfy hot_days <= archive_days <= delete_after_days")
        return self


class PromotionPolicy(StrictModel):
    require_reload_verification: bool = True
    require_passing_evaluation: bool = True
    require_no_regression: bool = True
    minimum_completed_evaluations: int = Field(default=1, ge=1, le=100)
    minimum_step: int = Field(default=1, ge=1)


class RetryPolicy(StrictModel):
    maximum_attempts: int = Field(default=3, ge=1, le=20)
    initial_backoff_seconds: int = Field(default=30, ge=1, le=86_400)
    maximum_backoff_seconds: int = Field(default=900, ge=1, le=86_400)
    retryable_failures: list[Literal["infrastructure", "preemption", "node_loss", "network", "object_store"]]
    preemption_action: Literal["checkpoint_then_requeue", "fail"] = "checkpoint_then_requeue"

    @model_validator(mode="after")
    def validate_backoff(self) -> "RetryPolicy":
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum retry backoff must be >= initial backoff")
        return self


class RuntimeLimitPolicy(StrictModel):
    maximum_wall_time_seconds: int = Field(ge=300, le=90 * 86_400)
    heartbeat_interval_seconds: int = Field(default=5, ge=1, le=300)
    heartbeat_timeout_seconds: int = Field(default=60, ge=5, le=3600)
    graceful_shutdown_seconds: int = Field(default=120, ge=5, le=3600)

    @model_validator(mode="after")
    def validate_heartbeat(self) -> "RuntimeLimitPolicy":
        if self.heartbeat_timeout_seconds < self.heartbeat_interval_seconds * 2:
            raise ValueError("heartbeat timeout must allow at least two heartbeat intervals")
        return self


class DataLoaderPolicy(StrictModel):
    workers_per_rank: int = Field(default=4, ge=0, le=128)
    prefetch_batches: int = Field(default=4, ge=1, le=128)
    pin_memory: bool = True
    persistent_workers: bool = True
    deterministic_shuffle: bool = True
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    local_cache_gib: int = Field(default=100, ge=0, le=100_000)
    cache_path: str = Field(default="/cache/craftly", min_length=1, max_length=1024)
    verify_shard_checksums: bool = True


class RuntimeImagePolicy(StrictModel):
    repository: str = Field(min_length=1, max_length=512)
    python_version: str = Field(pattern=r"^3\.\d{1,2}(?:\.\d{1,2})?$")
    pytorch_version: str = Field(min_length=1, max_length=64)
    cuda_version: str = Field(min_length=1, max_length=64)
    require_digest_pin: bool = True

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        if any(character.isspace() for character in value) or value.startswith(("http://", "https://")):
            raise ValueError("container repository must be an OCI image name")
        return value


class SchedulerPolicy(StrictModel):
    namespace: str = Field(min_length=1, max_length=128)
    queue: str = Field(min_length=1, max_length=128)
    account: str | None = Field(default=None, max_length=128)
    storage_class: str = Field(min_length=1, max_length=128)
    shared_checkpoint_claim: str | None = Field(default=None, max_length=253)
    priority_class: str = Field(min_length=1, max_length=128)
    gang_scheduling: bool = False
    topology_key: str = Field(default="kubernetes.io/hostname", min_length=1, max_length=253)

    @field_validator("namespace", "queue", "account", "storage_class", "priority_class")
    @classmethod
    def validate_scheduler_identifier(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}", value):
            raise ValueError("scheduler policy contains an unsafe identifier")
        return value

    @field_validator("shared_checkpoint_claim")
    @classmethod
    def validate_shared_checkpoint_claim(cls, value: str | None) -> str | None:
        if value and not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?", value):
            raise ValueError("shared checkpoint claim must be a DNS-compatible name")
        return value


class CostQuotaPolicy(StrictModel):
    maximum_gpu_hours: float = Field(gt=0.0, le=100_000_000.0)
    maximum_cost_usd: float = Field(gt=0.0, le=1_000_000_000.0)
    estimated_gpu_hour_cost_usd: float = Field(gt=0.0, le=100_000.0)
    maximum_storage_gib: int = Field(gt=0, le=100_000_000)
    maximum_egress_gib: int = Field(default=0, ge=0, le=100_000_000)
    maximum_concurrent_jobs_per_owner: int = Field(default=1, ge=1, le=1000)


class ImmutableInputPolicy(StrictModel):
    require_promoted_dataset: bool = True
    require_dataset_sha256: bool = True
    require_tokenizer_sha256: bool = True
    allowed_uri_schemes: list[Literal["s3", "file"]] = Field(default_factory=lambda: ["s3"])


def training_data_parallel_size(
    *,
    resources: ResourceRequest,
    distributed_strategy: str,
    parallelism: ParallelismPlan,
) -> int:
    """Return ranks consuming distinct samples, excluding model-parallel ranks."""
    if distributed_strategy == "megatron":
        return parallelism.data_parallel
    return resources.nodes * max(1, resources.gpus_per_node)


class TrainingProfile(StrictModel):
    profile_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=500)
    run_type: Literal["data_pipeline", "pretrain", "evaluation"]
    dev_only: bool = False
    model_profile: str = Field(min_length=1, max_length=64)
    curriculum_mode: Literal["balanced", "fundamentals_only", "defensive_security_only", "debug_patch_only", "agentic_coding_only"]
    scheduler: Literal["notebook", "kubernetes", "kubernetes_pytorch", "slurm"]
    distributed_strategy: Literal["none", "fsdp", "deepspeed_zero2", "deepspeed_zero3", "megatron"]
    parallelism: ParallelismPlan = Field(default_factory=ParallelismPlan)
    steps: int = Field(ge=1)
    sequence_length: int = Field(ge=32, le=1_048_576)
    batch_size: int = Field(ge=1, le=1024)
    gradient_accumulation_steps: int = Field(ge=1, le=65536)
    dtype: Literal["bf16", "fp16", "fp32"]
    resources: ResourceRequest
    optimizer: OptimizerPolicy
    learning_rate: LearningRatePolicy
    token_budget: TokenBudgetPolicy
    checkpoint: CheckpointPolicy
    evaluation: EvaluationPolicy
    retention: RetentionPolicy
    promotion: PromotionPolicy
    retry: RetryPolicy
    runtime_limits: RuntimeLimitPolicy
    dataloader: DataLoaderPolicy
    runtime_image: RuntimeImagePolicy
    scheduler_policy: SchedulerPolicy
    cost_quota: CostQuotaPolicy
    immutable_inputs: ImmutableInputPolicy
    allowed_overrides: dict[str, OverrideRange] = Field(default_factory=dict)
    requirements: list[str] = Field(default_factory=list)
    secret_references: list[str] = Field(default_factory=list)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if not SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("profile_id contains unsafe characters")
        return value

    @field_validator("secret_references")
    @classmethod
    def validate_secret_references(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", value):
                raise ValueError("secret references must be DNS-compatible names")
        return sorted(set(values))

    @model_validator(mode="after")
    def validate_distributed_topology(self) -> "TrainingProfile":
        if self.distributed_strategy != "none" and self.resources.gpus_per_node < 1:
            raise ValueError("distributed profiles require GPUs")
        if self.scheduler in {"kubernetes", "kubernetes_pytorch"} and not self.scheduler_policy.shared_checkpoint_claim:
            raise ValueError("Kubernetes training profiles require a shared RWX checkpoint claim")
        if self.scheduler == "notebook" and not self.dev_only:
            raise ValueError("notebook profiles must be dev_only validation profiles")
        if self.model_profile in {"32b", "62b", "4t_moe"} and not self.resources.rdma_required:
            raise ValueError("32B and larger profiles require RDMA")
        world_size = self.resources.nodes * max(1, self.resources.gpus_per_node)
        if self.distributed_strategy == "megatron":
            model = get_model_profile(self.model_profile)
            topology = model.distributed_topology_report(
                tensor_parallel=self.parallelism.tensor_parallel,
                pipeline_parallel=self.parallelism.pipeline_parallel,
                data_parallel=self.parallelism.data_parallel,
                context_parallel=self.parallelism.context_parallel,
                expert_parallel=self.parallelism.expert_parallel,
                gpus_per_node=self.resources.gpus_per_node,
            )
            if not topology["passed"]:
                raise ValueError("invalid Megatron profile topology: " + "; ".join(topology["failures"]))
            if topology["world_size"] != world_size:
                raise ValueError("Megatron parallelism world size must match requested GPU resources")
        data_parallel = training_data_parallel_size(
            resources=self.resources,
            distributed_strategy=self.distributed_strategy,
            parallelism=self.parallelism,
        )
        batch_contract = TrainingBatchContract(
            optimizer_steps=self.steps,
            sequence_length=self.sequence_length,
            micro_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            data_parallel_size=data_parallel,
        )
        if self.token_budget.global_batch_sequences != batch_contract.global_batch_sequences:
            raise ValueError(
                "global_batch_sequences must equal micro_batch_size * gradient_accumulation_steps * "
                "data_parallel_size"
            )
        if self.token_budget.target_tokens != batch_contract.target_tokens:
            raise ValueError(
                "target_tokens must equal optimizer_steps * sequence_length * global_batch_sequences"
            )
        if self.learning_rate.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be lower than total steps")
        if self.checkpoint.interval_steps > self.steps or self.evaluation.interval_steps > self.steps:
            raise ValueError("checkpoint/evaluation intervals cannot exceed total steps")
        if self.promotion.minimum_step > self.steps:
            raise ValueError("promotion minimum_step cannot exceed total steps")
        return self

    @property
    def immutable_hash(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class TrainingProfileRegistry(StrictModel):
    schema_version: int = Field(ge=1)
    profiles: list[TrainingProfile]

    @model_validator(mode="after")
    def validate_unique_profiles(self) -> "TrainingProfileRegistry":
        identities = [(item.profile_id, item.version) for item in self.profiles]
        if len(identities) != len(set(identities)):
            raise ValueError("training profile identifiers and versions must be unique")
        return self

    def latest(self, profile_id: str) -> TrainingProfile:
        candidates = [item for item in self.profiles if item.profile_id == profile_id]
        if not candidates:
            raise KeyError(f"training profile not found: {profile_id}")
        return max(candidates, key=lambda item: item.version)

    @classmethod
    def from_file(cls, path: str | Path = PROFILE_PATH) -> "TrainingProfileRegistry":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"training profile registry not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"training profile registry is invalid JSON: {source}") from exc
        defaults = payload.pop("defaults", {})
        if not isinstance(defaults, dict) or not isinstance(payload.get("profiles"), list):
            raise ValueError("training profile registry defaults/profiles have invalid types")
        payload["profiles"] = [deep_merge(defaults, item) for item in payload["profiles"]]
        return cls.model_validate(payload)


class QualificationRange(StrictModel):
    start_steps: int = Field(gt=0)
    stop_steps: int = Field(gt=0)
    increment_steps: int = Field(gt=0)
    profile_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_range(self) -> "QualificationRange":
        if self.stop_steps < self.start_steps:
            raise ValueError("qualification stop_steps must be >= start_steps")
        if (self.stop_steps - self.start_steps) % self.increment_steps:
            raise ValueError("qualification range must end on an exact increment")
        return self


class QualificationGate(StrictModel):
    require_succeeded_job: bool = True
    require_reload_verified_checkpoint: bool = True
    require_passing_evaluation: bool = True
    require_same_dataset_and_tokenizer: bool = True
    maximum_validation_regression: float = Field(default=0.03, ge=0.0, le=1.0)


class QualificationMilestone(StrictModel):
    milestone_id: str
    ordinal: int = Field(ge=1)
    profile_id: str
    steps: int = Field(gt=0)
    previous_milestone_id: str | None = None


class QualificationCampaign(StrictModel):
    campaign_id: str
    version: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=1000)
    ranges: list[QualificationRange] = Field(min_length=1)
    gate: QualificationGate = Field(default_factory=QualificationGate)

    @property
    def milestones(self) -> list[QualificationMilestone]:
        result: list[QualificationMilestone] = []
        previous: str | None = None
        ordinal = 1
        seen_steps: set[int] = set()
        for item in self.ranges:
            for steps in range(item.start_steps, item.stop_steps + 1, item.increment_steps):
                if steps in seen_steps:
                    raise ValueError(f"qualification campaign repeats step milestone: {steps}")
                seen_steps.add(steps)
                milestone_id = f"steps-{steps:07d}"
                result.append(
                    QualificationMilestone(
                        milestone_id=milestone_id,
                        ordinal=ordinal,
                        profile_id=item.profile_id,
                        steps=steps,
                        previous_milestone_id=previous,
                    )
                )
                previous = milestone_id
                ordinal += 1
        return result

    def milestone(self, milestone_id: str) -> QualificationMilestone:
        for item in self.milestones:
            if item.milestone_id == milestone_id:
                return item
        raise KeyError(f"qualification milestone not found: {milestone_id}")


class QualificationCampaignRegistry(StrictModel):
    schema_version: int = Field(ge=1)
    campaigns: list[QualificationCampaign]
    scientific_experiments: list[ScientificExperimentCampaignContract] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_campaigns(self) -> "QualificationCampaignRegistry":
        identities = [(item.campaign_id, item.version) for item in self.campaigns]
        if len(identities) != len(set(identities)):
            raise ValueError("qualification campaign identifiers and versions must be unique")
        for campaign in self.campaigns:
            campaign.milestones
        scientific_identities = [
            (item.campaign_id, item.version) for item in self.scientific_experiments
        ]
        if len(scientific_identities) != len(set(scientific_identities)):
            raise ValueError("scientific campaign identifiers and versions must be unique")
        return self

    def latest(self, campaign_id: str) -> QualificationCampaign:
        matches = [item for item in self.campaigns if item.campaign_id == campaign_id]
        if not matches:
            raise KeyError(f"qualification campaign not found: {campaign_id}")
        return max(matches, key=lambda item: item.version)

    @classmethod
    def from_file(cls, path: str | Path = CAMPAIGN_PATH) -> "QualificationCampaignRegistry":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"training qualification registry not found: {source}")
        return cls.model_validate_json(source.read_text(encoding="utf-8-sig"))


class TrainingJobCreateRequest(StrictModel):
    profile_id: str = Field(min_length=1, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=128)
    overrides: dict[str, int] = Field(default_factory=dict)
    dataset_manifest_uri: str | None = Field(default=None, max_length=2048)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tokenizer_uri: str | None = Field(default=None, max_length=2048)
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_progression_decision_uri: str | None = Field(default=None, max_length=2048)
    model_progression_decision_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    git_commit: str = Field(default="0000000", min_length=7, max_length=64)
    container_digest: str = Field(default="sha256:" + ("0" * 64), min_length=71, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)
    qualification_campaign_id: str | None = Field(default=None, max_length=128)
    qualification_milestone_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_qualification_binding(self) -> "TrainingJobCreateRequest":
        if bool(self.qualification_campaign_id) != bool(self.qualification_milestone_id):
            raise ValueError("qualification campaign and milestone must be supplied together")
        if bool(self.model_progression_decision_uri) != bool(
            self.model_progression_decision_sha256
        ):
            raise ValueError("model progression decision URI and SHA-256 must be supplied together")
        return self

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        if not SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("idempotency_key contains unsafe characters")
        return value

    @field_validator("git_commit")
    @classmethod
    def validate_git_commit(cls, value: str) -> str:
        if not SAFE_GIT_COMMIT.fullmatch(value):
            raise ValueError("git_commit must be a hexadecimal commit identifier")
        return value

    @field_validator("container_digest")
    @classmethod
    def validate_container_digest(cls, value: str) -> str:
        if not SAFE_CONTAINER_DIGEST.fullmatch(value):
            raise ValueError("container_digest must be an immutable sha256 digest")
        return value


class TrainingJobSpec(StrictModel):
    schema_version: int = Field(default=3, ge=1, le=3)
    scratch_only: Literal[True] = True
    step_semantics: Literal["optimizer_update_v2"] = TRAINING_STEP_SEMANTICS
    validation_only: bool = False
    profile_id: str
    profile_version: int
    profile_hash: str
    project_id: str | None = None
    run_type: Literal["data_pipeline", "pretrain", "evaluation"]
    model_profile: str
    model_contract: dict[str, Any] | None = None
    model_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_parameter_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    curriculum_mode: str
    scheduler: str
    distributed_strategy: str
    parallelism: ParallelismPlan = Field(default_factory=ParallelismPlan)
    steps: int
    sequence_length: int
    batch_size: int
    gradient_accumulation_steps: int
    dtype: str
    resources: ResourceRequest
    optimizer: OptimizerPolicy
    learning_rate: LearningRatePolicy
    token_budget: TokenBudgetPolicy
    checkpoint: CheckpointPolicy
    evaluation: EvaluationPolicy
    retention: RetentionPolicy
    promotion: PromotionPolicy
    retry: RetryPolicy
    runtime_limits: RuntimeLimitPolicy
    dataloader: DataLoaderPolicy
    runtime_image: RuntimeImagePolicy
    scheduler_policy: SchedulerPolicy
    cost_quota: CostQuotaPolicy
    immutable_inputs: ImmutableInputPolicy
    dataset_manifest_uri: str | None = None
    dataset_manifest_sha256: str | None = None
    tokenizer_uri: str | None = None
    tokenizer_sha256: str | None = None
    model_progression_decision_uri: str | None = None
    model_progression_decision_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    git_commit: str
    container_digest: str
    requirements: list[str] = Field(default_factory=list)
    secret_references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    qualification_campaign_id: str | None = None
    qualification_campaign_version: int | None = None
    qualification_milestone_id: str | None = None
    spec_hash: str = ""

    @model_validator(mode="after")
    def validate_and_hash(self) -> "TrainingJobSpec":
        if self.schema_version >= 2:
            if not self.model_contract or not self.model_contract_sha256 or not self.model_parameter_report_sha256:
                raise ValueError("schema v2+ training jobs require an immutable canonical model contract")
            bound_model = ScratchDecoderConfig.model_validate(self.model_contract)
            if not hmac.compare_digest(bound_model.contract_sha256(), self.model_contract_sha256):
                raise ValueError("training job model contract hash mismatch")
            parameter_report_sha256 = sha256_text(
                canonical_json(bound_model.parameter_report())
            )
            if not hmac.compare_digest(parameter_report_sha256, self.model_parameter_report_sha256):
                raise ValueError("training job model parameter report hash mismatch")
        if self.run_type == "pretrain":
            missing = [
                name
                for name, value in [
                    ("dataset_manifest_uri", self.dataset_manifest_uri),
                    ("dataset_manifest_sha256", self.dataset_manifest_sha256),
                    ("tokenizer_uri", self.tokenizer_uri),
                    ("tokenizer_sha256", self.tokenizer_sha256),
                ]
                if not value
            ]
            if missing:
                raise ValueError("pretraining job is missing immutable inputs: " + ", ".join(missing))
        if bool(self.model_progression_decision_uri) != bool(
            self.model_progression_decision_sha256
        ):
            raise ValueError("model progression decision URI and SHA-256 must be bound together")
        if "model_progression_decision" in self.requirements and not self.model_progression_decision_uri:
            raise ValueError("selected model profile requires an authorized model progression decision")
        data_parallel = training_data_parallel_size(
            resources=self.resources,
            distributed_strategy=self.distributed_strategy,
            parallelism=self.parallelism,
        )
        batch_contract = TrainingBatchContract(
            optimizer_steps=self.steps,
            sequence_length=self.sequence_length,
            micro_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            data_parallel_size=data_parallel,
        )
        if self.token_budget.global_batch_sequences != batch_contract.global_batch_sequences:
            raise ValueError("resolved global batch does not match the job topology")
        if self.token_budget.target_tokens != batch_contract.target_tokens:
            raise ValueError("resolved token target does not match steps and topology")
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        calculated = sha256_text(canonical_json(payload))
        if self.spec_hash and not hmac.compare_digest(self.spec_hash, calculated):
            raise ValueError("training job spec hash mismatch")
        object.__setattr__(self, "spec_hash", calculated)
        return self

    def assert_current_model_contract(self) -> ScratchDecoderConfig:
        """Reject execution if the named runtime contract drifted after submission."""
        if self.schema_version < 2:
            raise ValueError("legacy schema v1 jobs cannot execute without a bound model contract")
        bound = ScratchDecoderConfig.model_validate(self.model_contract)
        current = get_model_profile(self.model_profile).model_copy(
            update={"vocab_size": bound.vocab_size}
        )
        if not self.model_contract_sha256 or not hmac.compare_digest(
            current.contract_sha256(), self.model_contract_sha256
        ):
            raise ValueError(
                "current canonical model contract differs from the immutable training job binding"
            )
        return current


class TrainingAttempt(StrictModel):
    attempt_id: str
    job_id: str
    attempt_number: int = Field(ge=1)
    scheduler: str
    scheduler_binding: dict[str, Any] = Field(default_factory=dict)
    checkpoint_uri: str | None = None
    status: JobStatus
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TrainingJob(StrictModel):
    job_id: str
    owner_id: str
    idempotency_key: str
    spec: TrainingJobSpec
    status: JobStatus
    version: int = Field(ge=1)
    event_sequence: int = Field(default=0, ge=0)
    archived_event_sequence: int = Field(default=0, ge=0)
    current_attempt_id: str | None = None
    scheduler_binding: dict[str, Any] = Field(default_factory=dict)
    failure_code: str | None = None
    failure_detail: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TrainingEventInput(StrictModel):
    source_sequence: int = Field(ge=0)
    kind: Literal["status", "metric", "log", "heartbeat", "checkpoint", "evaluation", "error"]
    stage: str = Field(min_length=1, max_length=128)
    status: str = Field(default="running", min_length=1, max_length=64)
    rank: int = Field(default=0, ge=0, le=1_000_000)
    world_size: int = Field(default=1, ge=1, le=1_000_000)
    node: str = Field(default_factory=socket.gethostname, min_length=1, max_length=255)
    step: int | None = Field(default=None, ge=0)
    max_steps: int | None = Field(default=None, ge=1)
    loss: float | None = None
    validation_loss: float | None = None
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    gpu_memory_bytes: int | None = Field(default=None, ge=0)
    message: str | None = Field(default=None, max_length=60_000)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_finite_metrics(self) -> "TrainingEventInput":
        for name in ("loss", "validation_loss"):
            value = getattr(self, name)
            if value is not None and (value != value or value in {float("inf"), float("-inf")}):
                raise ValueError(f"{name} must be finite")
        encoded = canonical_json(self.model_dump(mode="json")).encode("utf-8")
        if len(encoded) > MAX_EVENT_BYTES:
            raise ValueError(f"event exceeds {MAX_EVENT_BYTES} bytes")
        return self


class TrainingEvent(TrainingEventInput):
    schema_version: int = 1
    event_id: str
    job_id: str
    attempt_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)


class TrainingEventBatch(StrictModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    events: list[TrainingEventInput] = Field(min_length=1, max_length=MAX_EVENT_BATCH)


class TrainingArtifact(StrictModel):
    artifact_id: str
    job_id: str
    attempt_id: str | None = None
    kind: Literal["spec", "log", "checkpoint", "evaluation", "report", "dataset", "tokenizer"]
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    promoted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CheckpointVersion(StrictModel):
    checkpoint_id: str
    job_id: str
    attempt_id: str | None = None
    step: int = Field(ge=0)
    manifest_uri: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    reload_verified: bool = False
    promoted: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class CheckpointCommitRequest(StrictModel):
    attempt_id: str
    step: int = Field(ge=0)
    manifest_uri: str = Field(min_length=1, max_length=4096)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    reload_verified: bool = False


class EvaluationRun(StrictModel):
    evaluation_id: str
    job_id: str
    checkpoint_id: str | None = None
    status: str
    report_uri: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["pass", "fail", "release"]
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationCommitRequest(StrictModel):
    checkpoint_id: str | None = None
    status: str = Field(min_length=1, max_length=64)
    report_uri: str = Field(min_length=1, max_length=4096)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["pass", "fail", "release"]
    metrics: dict[str, Any] = Field(default_factory=dict)


class ArtifactUploadRequest(StrictModel):
    attempt_id: str | None = Field(default=None, max_length=128)
    kind: Literal["log", "checkpoint", "evaluation", "report", "dataset", "tokenizer"]
    filename: str = Field(min_length=1, max_length=255)
    relative_path: str | None = Field(default=None, max_length=1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0, le=10 * 1024**4)
    content_type: str = Field(default="application/octet-stream", min_length=1, max_length=128)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."} or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}", value):
            raise ValueError("artifact filename is unsafe")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/").strip("/")
        parts = normalized.split("/")
        if not normalized or len(parts) > 32 or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("artifact relative_path is unsafe")
        if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}", part) for part in parts):
            raise ValueError("artifact relative_path contains an unsafe component")
        return normalized


class SchedulerBinding(StrictModel):
    scheduler: str
    external_id: str
    namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceAccount(StrictModel):
    service_account_id: str
    name: str
    scopes: list[str]
    token_prefix: str
    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None


class ServiceAccountCredential(StrictModel):
    account: ServiceAccount
    bootstrap_token: str


class RefreshSession(StrictModel):
    session_id: str
    service_account_id: str
    refresh_token: str
    expires_at: datetime


class TrainingAuditEvent(StrictModel):
    audit_id: str
    actor_id: str = Field(min_length=1, max_length=255)
    job_id: str | None = None
    action: str = Field(min_length=1, max_length=128)
    outcome: Literal["accepted", "succeeded", "denied", "failed"]
    request_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TrainingStore(Protocol):
    async def sync_profiles(self, profiles: list[TrainingProfile]) -> None: ...
    async def create_job(self, job: TrainingJob) -> TrainingJob: ...
    async def get_job(self, job_id: str) -> TrainingJob | None: ...
    async def get_job_by_idempotency(self, owner_id: str, idempotency_key: str) -> TrainingJob | None: ...
    async def list_jobs(self, *, owner_id: str | None = None, limit: int = 100) -> list[TrainingJob]: ...
    async def transition_job(self, job_id: str, target: JobStatus, *, expected_version: int, detail: str | None = None, binding: dict[str, Any] | None = None) -> TrainingJob: ...
    async def update_binding(self, job_id: str, *, expected_version: int, binding: dict[str, Any]) -> TrainingJob: ...
    async def update_archive_cursor(self, job_id: str, sequence: int) -> TrainingJob: ...
    async def allocate_event_sequences(
        self,
        job_id: str,
        attempt_id: str,
        source_keys: list[tuple[int, int]],
    ) -> list[int | None]: ...
    async def create_attempt(self, attempt: TrainingAttempt) -> TrainingAttempt: ...
    async def get_attempt(self, attempt_id: str) -> TrainingAttempt | None: ...
    async def list_attempts(self, job_id: str) -> list[TrainingAttempt]: ...
    async def update_attempt_status(self, attempt_id: str, status: JobStatus) -> TrainingAttempt: ...
    async def add_artifact(self, artifact: TrainingArtifact) -> TrainingArtifact: ...
    async def list_artifacts(self, job_id: str) -> list[TrainingArtifact]: ...
    async def add_checkpoint(self, checkpoint: CheckpointVersion) -> CheckpointVersion: ...
    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointVersion | None: ...
    async def list_checkpoints(self, job_id: str) -> list[CheckpointVersion]: ...
    async def promote_checkpoint(self, checkpoint_id: str) -> CheckpointVersion: ...
    async def add_evaluation(self, evaluation: EvaluationRun) -> EvaluationRun: ...
    async def list_evaluations(self, job_id: str) -> list[EvaluationRun]: ...
    async def create_service_account(self, account: ServiceAccount, token_hash: str) -> None: ...
    async def ensure_service_account(self, account: ServiceAccount, token_hash: str) -> None: ...
    async def authenticate_service_account(self, bootstrap_token: str) -> ServiceAccount | None: ...
    async def create_refresh_session(self, service_account_id: str, refresh_hash: str, expires_at: datetime) -> str: ...
    async def consume_refresh_session(self, session_id: str, refresh_token: str) -> ServiceAccount | None: ...
    async def revoke_refresh_session(self, session_id: str, refresh_token: str) -> bool: ...
    async def add_audit_event(self, event: TrainingAuditEvent) -> TrainingAuditEvent: ...
    async def list_audit_events(self, job_id: str, *, limit: int = 100) -> list[TrainingAuditEvent]: ...
    async def close(self) -> None: ...


class InMemoryTrainingStore:
    """Deterministic development/test store; never production-ready."""

    def __init__(self) -> None:
        self.jobs: dict[str, TrainingJob] = {}
        self.idempotency: dict[tuple[str, str], str] = {}
        self.attempts: dict[str, TrainingAttempt] = {}
        self.artifacts: dict[str, list[TrainingArtifact]] = defaultdict(list)
        self.checkpoints: dict[str, CheckpointVersion] = {}
        self.evaluations: dict[str, EvaluationRun] = {}
        self.accounts: dict[str, tuple[ServiceAccount, str]] = {}
        self.refresh_sessions: dict[str, tuple[str, str, datetime]] = {}
        self.audit_events: list[TrainingAuditEvent] = []
        self.profile_hashes: dict[tuple[str, int], str] = {}
        self.event_ingress: dict[tuple[str, int, int], int] = {}
        self.lock = asyncio.Lock()

    async def sync_profiles(self, profiles: list[TrainingProfile]) -> None:
        async with self.lock:
            for profile in profiles:
                identity = (profile.profile_id, profile.version)
                existing = self.profile_hashes.get(identity)
                if existing and not hmac.compare_digest(existing, profile.immutable_hash):
                    raise ValueError(f"immutable training profile changed without a version bump: {profile.profile_id}@{profile.version}")
                self.profile_hashes[identity] = profile.immutable_hash

    async def create_job(self, job: TrainingJob) -> TrainingJob:
        async with self.lock:
            key = (job.owner_id, job.idempotency_key)
            existing_id = self.idempotency.get(key)
            if existing_id:
                return self.jobs[existing_id].model_copy(deep=True)
            self.jobs[job.job_id] = job.model_copy(deep=True)
            self.idempotency[key] = job.job_id
            return job.model_copy(deep=True)

    async def get_job(self, job_id: str) -> TrainingJob | None:
        async with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def get_job_by_idempotency(self, owner_id: str, idempotency_key: str) -> TrainingJob | None:
        async with self.lock:
            job_id = self.idempotency.get((owner_id, idempotency_key))
            return self.jobs[job_id].model_copy(deep=True) if job_id else None

    async def list_jobs(self, *, owner_id: str | None = None, limit: int = 100) -> list[TrainingJob]:
        async with self.lock:
            rows = [item for item in self.jobs.values() if owner_id is None or item.owner_id == owner_id]
            rows.sort(key=lambda item: item.created_at, reverse=True)
            return [item.model_copy(deep=True) for item in rows[:limit]]

    async def transition_job(
        self,
        job_id: str,
        target: JobStatus,
        *,
        expected_version: int,
        detail: str | None = None,
        binding: dict[str, Any] | None = None,
    ) -> TrainingJob:
        async with self.lock:
            current = self.jobs.get(job_id)
            if current is None:
                raise KeyError(f"training job not found: {job_id}")
            if current.version != expected_version:
                raise RuntimeError("training job version conflict")
            if target not in ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(f"invalid training job transition: {current.status.value} -> {target.value}")
            now = utc_now()
            update: dict[str, Any] = {"status": target, "version": current.version + 1, "updated_at": now}
            if binding is not None:
                update["scheduler_binding"] = binding
            if target == JobStatus.RUNNING and current.started_at is None:
                update["started_at"] = now
            if target.value in TERMINAL_STATES:
                update["finished_at"] = now
            if target in {JobStatus.FAILED, JobStatus.BLOCKED}:
                update["failure_detail"] = redact_text(detail or "")[:4000]
            updated = current.model_copy(update=update)
            self.jobs[job_id] = updated
            return updated.model_copy(deep=True)

    async def allocate_event_sequences(
        self,
        job_id: str,
        attempt_id: str,
        source_keys: list[tuple[int, int]],
    ) -> list[int | None]:
        if not source_keys or len(source_keys) > MAX_EVENT_BATCH:
            raise ValueError("event sequence allocation count is invalid")
        async with self.lock:
            current = self.jobs.get(job_id)
            if current is None:
                raise KeyError(f"training job not found: {job_id}")
            sequence = current.event_sequence
            allocated: list[int | None] = []
            for rank, source_sequence in source_keys:
                identity = (attempt_id, rank, source_sequence)
                if identity in self.event_ingress:
                    allocated.append(None)
                    continue
                sequence += 1
                self.event_ingress[identity] = sequence
                allocated.append(sequence)
            if sequence != current.event_sequence:
                self.jobs[job_id] = current.model_copy(update={"event_sequence": sequence})
            return allocated

    async def update_binding(self, job_id: str, *, expected_version: int, binding: dict[str, Any]) -> TrainingJob:
        async with self.lock:
            current = self.jobs.get(job_id)
            if current is None:
                raise KeyError(f"training job not found: {job_id}")
            if current.version != expected_version:
                raise RuntimeError("training job version conflict")
            updated = current.model_copy(
                update={"scheduler_binding": binding, "version": current.version + 1, "updated_at": utc_now()}
            )
            self.jobs[job_id] = updated
            return updated.model_copy(deep=True)

    async def update_archive_cursor(self, job_id: str, sequence: int) -> TrainingJob:
        async with self.lock:
            current = self.jobs.get(job_id)
            if current is None:
                raise KeyError(f"training job not found: {job_id}")
            if sequence < current.archived_event_sequence or sequence > current.event_sequence:
                raise ValueError("invalid event archive cursor")
            updated = current.model_copy(update={"archived_event_sequence": sequence, "updated_at": utc_now()})
            self.jobs[job_id] = updated
            return updated.model_copy(deep=True)

    async def create_attempt(self, attempt: TrainingAttempt) -> TrainingAttempt:
        async with self.lock:
            self.attempts[attempt.attempt_id] = attempt.model_copy(deep=True)
            job = self.jobs[attempt.job_id]
            self.jobs[attempt.job_id] = job.model_copy(update={"current_attempt_id": attempt.attempt_id})
            return attempt.model_copy(deep=True)

    async def get_attempt(self, attempt_id: str) -> TrainingAttempt | None:
        async with self.lock:
            attempt = self.attempts.get(attempt_id)
            return attempt.model_copy(deep=True) if attempt else None

    async def list_attempts(self, job_id: str) -> list[TrainingAttempt]:
        async with self.lock:
            rows = [item for item in self.attempts.values() if item.job_id == job_id]
            rows.sort(key=lambda item: item.attempt_number)
            return [item.model_copy(deep=True) for item in rows]

    async def update_attempt_status(self, attempt_id: str, status: JobStatus) -> TrainingAttempt:
        async with self.lock:
            current = self.attempts.get(attempt_id)
            if current is None:
                raise KeyError(f"training attempt not found: {attempt_id}")
            now = utc_now()
            updated = current.model_copy(
                update={
                    "status": status,
                    "started_at": now if status == JobStatus.RUNNING and current.started_at is None else current.started_at,
                    "finished_at": now if status.value in TERMINAL_STATES else current.finished_at,
                }
            )
            self.attempts[attempt_id] = updated
            return updated.model_copy(deep=True)

    async def add_artifact(self, artifact: TrainingArtifact) -> TrainingArtifact:
        async with self.lock:
            self.artifacts[artifact.job_id].append(artifact.model_copy(deep=True))
            return artifact.model_copy(deep=True)

    async def list_artifacts(self, job_id: str) -> list[TrainingArtifact]:
        async with self.lock:
            return [item.model_copy(deep=True) for item in self.artifacts.get(job_id, [])]

    async def add_checkpoint(self, checkpoint: CheckpointVersion) -> CheckpointVersion:
        async with self.lock:
            duplicate = next(
                (item for item in self.checkpoints.values() if item.job_id == checkpoint.job_id and item.step == checkpoint.step),
                None,
            )
            if duplicate:
                if duplicate.manifest_sha256 != checkpoint.manifest_sha256:
                    raise ValueError("checkpoint step already exists with a different manifest")
                return duplicate.model_copy(deep=True)
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint.model_copy(deep=True)
            return checkpoint.model_copy(deep=True)

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointVersion | None:
        async with self.lock:
            item = self.checkpoints.get(checkpoint_id)
            return item.model_copy(deep=True) if item else None

    async def list_checkpoints(self, job_id: str) -> list[CheckpointVersion]:
        async with self.lock:
            rows = [item for item in self.checkpoints.values() if item.job_id == job_id]
            rows.sort(key=lambda item: item.step, reverse=True)
            return [item.model_copy(deep=True) for item in rows]

    async def promote_checkpoint(self, checkpoint_id: str) -> CheckpointVersion:
        async with self.lock:
            current = self.checkpoints.get(checkpoint_id)
            if current is None:
                raise KeyError(f"checkpoint not found: {checkpoint_id}")
            updated = current.model_copy(update={"promoted": True})
            self.checkpoints[checkpoint_id] = updated
            return updated.model_copy(deep=True)

    async def add_evaluation(self, evaluation: EvaluationRun) -> EvaluationRun:
        async with self.lock:
            self.evaluations[evaluation.evaluation_id] = evaluation.model_copy(deep=True)
            return evaluation.model_copy(deep=True)

    async def list_evaluations(self, job_id: str) -> list[EvaluationRun]:
        async with self.lock:
            rows = [item for item in self.evaluations.values() if item.job_id == job_id]
            rows.sort(key=lambda item: item.created_at, reverse=True)
            return [item.model_copy(deep=True) for item in rows]

    async def create_service_account(self, account: ServiceAccount, token_hash: str) -> None:
        async with self.lock:
            if any(existing.name == account.name for existing, _ in self.accounts.values()):
                raise ValueError("service account name already exists")
            self.accounts[account.service_account_id] = (account.model_copy(deep=True), token_hash)

    async def ensure_service_account(self, account: ServiceAccount, token_hash: str) -> None:
        async with self.lock:
            existing = self.accounts.get(account.service_account_id)
            if existing and existing[0].name != account.name:
                raise ValueError("reserved bootstrap service-account identity conflict")
            if existing and not hmac.compare_digest(existing[1], token_hash):
                self.refresh_sessions = {
                    session_id: row
                    for session_id, row in self.refresh_sessions.items()
                    if row[0] != account.service_account_id
                }
            self.accounts[account.service_account_id] = (account.model_copy(deep=True), token_hash)

    async def authenticate_service_account(self, bootstrap_token: str) -> ServiceAccount | None:
        digest = sha256_text(bootstrap_token)
        async with self.lock:
            for account, expected in self.accounts.values():
                if account.enabled and hmac.compare_digest(digest, expected):
                    updated = account.model_copy(update={"last_used_at": utc_now()})
                    self.accounts[account.service_account_id] = (updated, expected)
                    return updated.model_copy(deep=True)
        return None

    async def create_refresh_session(self, service_account_id: str, refresh_hash: str, expires_at: datetime) -> str:
        session_id = str(uuid.uuid4())
        async with self.lock:
            self.refresh_sessions[session_id] = (service_account_id, refresh_hash, expires_at)
        return session_id

    async def consume_refresh_session(self, session_id: str, refresh_token: str) -> ServiceAccount | None:
        digest = sha256_text(refresh_token)
        async with self.lock:
            row = self.refresh_sessions.get(session_id)
            if not row:
                return None
            account_id, expected, expires_at = row
            if expires_at < utc_now() or not hmac.compare_digest(digest, expected):
                self.refresh_sessions.pop(session_id, None)
                return None
            account = self.accounts.get(account_id)
            return account[0].model_copy(deep=True) if account and account[0].enabled else None

    async def revoke_refresh_session(self, session_id: str, refresh_token: str) -> bool:
        digest = sha256_text(refresh_token)
        async with self.lock:
            row = self.refresh_sessions.get(session_id)
            if not row or not hmac.compare_digest(row[1], digest):
                return False
            self.refresh_sessions.pop(session_id, None)
            return True

    async def add_audit_event(self, event: TrainingAuditEvent) -> TrainingAuditEvent:
        async with self.lock:
            stored = event.model_copy(update={"metadata": redact_payload(event.metadata)})
            self.audit_events.append(stored)
            return stored.model_copy(deep=True)

    async def list_audit_events(self, job_id: str, *, limit: int = 100) -> list[TrainingAuditEvent]:
        async with self.lock:
            rows = [item for item in self.audit_events if item.job_id == job_id]
            rows.sort(key=lambda item: item.created_at, reverse=True)
            return [item.model_copy(deep=True) for item in rows[:limit]]

    async def close(self) -> None:
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(value)
    return value


class PostgresTrainingStore:
    """Async Postgres source of truth for production training jobs."""

    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 10) -> None:
        if not database_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("training store requires a Postgres database URL")
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                import asyncpg

                self._pool = await asyncpg.create_pool(self.database_url, min_size=self.min_size, max_size=self.max_size)
        return self._pool

    @staticmethod
    def _job(row: Any) -> TrainingJob:
        payload = dict(row)
        payload["job_id"] = str(payload.pop("id"))
        payload["current_attempt_id"] = str(payload["current_attempt_id"]) if payload.get("current_attempt_id") else None
        payload["spec"] = _json_value(payload["spec_json"])
        payload["scheduler_binding"] = _json_value(payload.get("scheduler_binding") or {})
        for key in ["spec_json", "profile_id", "profile_version", "spec_hash"]:
            payload.pop(key, None)
        return TrainingJob.model_validate(payload)

    async def sync_profiles(self, profiles: list[TrainingProfile]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            for profile in profiles:
                await connection.execute(
                    """
                    INSERT INTO training_profiles(profile_id,version,profile_hash,profile_json,enabled)
                    VALUES($1,$2,$3,$4::jsonb,true)
                    ON CONFLICT(profile_id,version) DO NOTHING
                    """,
                    profile.profile_id,
                    profile.version,
                    profile.immutable_hash,
                    canonical_json(profile.model_dump(mode="json")),
                )
                stored_hash = await connection.fetchval(
                    "SELECT profile_hash FROM training_profiles WHERE profile_id=$1 AND version=$2",
                    profile.profile_id,
                    profile.version,
                )
                if not stored_hash or not hmac.compare_digest(str(stored_hash), profile.immutable_hash):
                    raise ValueError(
                        f"immutable training profile changed without a version bump: {profile.profile_id}@{profile.version}"
                    )

    async def create_job(self, job: TrainingJob) -> TrainingJob:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO training_jobs(
                  id, owner_id, idempotency_key, profile_id, profile_version,
                  spec_hash, spec_json, status, version, event_sequence,
                  scheduler_binding, created_at, updated_at
                ) VALUES($1::uuid,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11::jsonb,$12,$13)
                ON CONFLICT(owner_id, idempotency_key) DO UPDATE SET updated_at=training_jobs.updated_at
                RETURNING *
                """,
                job.job_id,
                job.owner_id,
                job.idempotency_key,
                job.spec.profile_id,
                job.spec.profile_version,
                job.spec.spec_hash,
                canonical_json(job.spec.model_dump(mode="json")),
                job.status.value,
                job.version,
                job.event_sequence,
                canonical_json(job.scheduler_binding),
                job.created_at,
                job.updated_at,
            )
        return self._job(row)

    async def get_job(self, job_id: str) -> TrainingJob | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow("SELECT * FROM training_jobs WHERE id=$1::uuid", job_id)
        return self._job(row) if row else None

    async def get_job_by_idempotency(self, owner_id: str, idempotency_key: str) -> TrainingJob | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM training_jobs WHERE owner_id=$1 AND idempotency_key=$2",
                owner_id,
                idempotency_key,
            )
        return self._job(row) if row else None

    async def list_jobs(self, *, owner_id: str | None = None, limit: int = 100) -> list[TrainingJob]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            if owner_id:
                rows = await connection.fetch(
                    "SELECT * FROM training_jobs WHERE owner_id=$1 ORDER BY created_at DESC LIMIT $2",
                    owner_id,
                    limit,
                )
            else:
                rows = await connection.fetch("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT $1", limit)
        return [self._job(row) for row in rows]

    async def transition_job(
        self,
        job_id: str,
        target: JobStatus,
        *,
        expected_version: int,
        detail: str | None = None,
        binding: dict[str, Any] | None = None,
    ) -> TrainingJob:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            current_row = await connection.fetchrow("SELECT * FROM training_jobs WHERE id=$1::uuid FOR UPDATE", job_id)
            if not current_row:
                raise KeyError(f"training job not found: {job_id}")
            current = self._job(current_row)
            if current.version != expected_version:
                raise RuntimeError("training job version conflict")
            if target not in ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(f"invalid training job transition: {current.status.value} -> {target.value}")
            now = utc_now()
            started_at = now if target == JobStatus.RUNNING and current.started_at is None else current.started_at
            finished_at = now if target.value in TERMINAL_STATES else current.finished_at
            failure_detail = redact_text(detail or "")[:4000] if target in {JobStatus.FAILED, JobStatus.BLOCKED} else current.failure_detail
            row = await connection.fetchrow(
                """
                UPDATE training_jobs SET status=$2, version=version+1, updated_at=$3,
                  started_at=$4, finished_at=$5, failure_detail=$6,
                  scheduler_binding=COALESCE($7::jsonb, scheduler_binding)
                WHERE id=$1::uuid AND version=$8 RETURNING *
                """,
                job_id,
                target.value,
                now,
                started_at,
                finished_at,
                failure_detail,
                canonical_json(binding) if binding is not None else None,
                expected_version,
            )
            if not row:
                raise RuntimeError("training job version conflict")
        return self._job(row)

    async def allocate_event_sequences(
        self,
        job_id: str,
        attempt_id: str,
        source_keys: list[tuple[int, int]],
    ) -> list[int | None]:
        if not source_keys or len(source_keys) > MAX_EVENT_BATCH:
            raise ValueError("event sequence allocation count is invalid")
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT event_sequence FROM training_jobs WHERE id=$1::uuid FOR UPDATE",
                job_id,
            )
            if not row:
                raise KeyError(f"training job not found: {job_id}")
            sequence = int(row["event_sequence"])
            ranks = [item[0] for item in source_keys]
            source_sequences = [item[1] for item in source_keys]
            existing_rows = await connection.fetch(
                """
                WITH incoming AS (
                  SELECT rank,source_sequence
                  FROM unnest($2::integer[],$3::bigint[]) AS source(rank,source_sequence)
                )
                SELECT ingress.rank,ingress.source_sequence,ingress.assigned_sequence
                FROM training_event_ingress AS ingress
                JOIN incoming USING(rank,source_sequence)
                WHERE ingress.attempt_id=$1::uuid
                """,
                attempt_id,
                ranks,
                source_sequences,
            )
            existing = {
                (int(item["rank"]), int(item["source_sequence"])): int(item["assigned_sequence"])
                for item in existing_rows
            }
            allocated: list[int | None] = []
            insert_ranks: list[int] = []
            insert_sources: list[int] = []
            insert_assigned: list[int] = []
            seen_in_batch: set[tuple[int, int]] = set()
            for rank, source_sequence in source_keys:
                identity = (rank, source_sequence)
                if identity in existing or identity in seen_in_batch:
                    allocated.append(None)
                else:
                    seen_in_batch.add(identity)
                    sequence += 1
                    allocated.append(sequence)
                    insert_ranks.append(rank)
                    insert_sources.append(source_sequence)
                    insert_assigned.append(sequence)
            if insert_ranks:
                await connection.execute(
                    """
                    INSERT INTO training_event_ingress(attempt_id,rank,source_sequence,assigned_sequence)
                    SELECT $1::uuid,rank,source_sequence,assigned_sequence
                    FROM unnest($2::integer[],$3::bigint[],$4::bigint[])
                      AS source(rank,source_sequence,assigned_sequence)
                    """,
                    attempt_id,
                    insert_ranks,
                    insert_sources,
                    insert_assigned,
                )
            await connection.execute(
                "UPDATE training_jobs SET event_sequence=$2 WHERE id=$1::uuid",
                job_id,
                sequence,
            )
        return allocated

    async def update_binding(self, job_id: str, *, expected_version: int, binding: dict[str, Any]) -> TrainingJob:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE training_jobs SET scheduler_binding=$3::jsonb,version=version+1,updated_at=now()
                WHERE id=$1::uuid AND version=$2 RETURNING *
                """,
                job_id,
                expected_version,
                canonical_json(binding),
            )
        if not row:
            raise RuntimeError("training job version conflict")
        return self._job(row)

    async def update_archive_cursor(self, job_id: str, sequence: int) -> TrainingJob:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE training_jobs SET archived_event_sequence=$2,updated_at=now()
                WHERE id=$1::uuid AND archived_event_sequence<=$2 AND event_sequence>=$2 RETURNING *
                """,
                job_id,
                sequence,
            )
        if not row:
            raise ValueError("invalid event archive cursor or training job not found")
        return self._job(row)

    async def create_attempt(self, attempt: TrainingAttempt) -> TrainingAttempt:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO training_attempts(id,job_id,attempt_number,scheduler,scheduler_binding,checkpoint_uri,status,created_at,started_at,finished_at)
                VALUES($1::uuid,$2::uuid,$3,$4,$5::jsonb,$6,$7,$8,$9,$10)
                """,
                attempt.attempt_id,
                attempt.job_id,
                attempt.attempt_number,
                attempt.scheduler,
                canonical_json(attempt.scheduler_binding),
                attempt.checkpoint_uri,
                attempt.status.value,
                attempt.created_at,
                attempt.started_at,
                attempt.finished_at,
            )
            await connection.execute(
                "UPDATE training_jobs SET current_attempt_id=$2::uuid WHERE id=$1::uuid",
                attempt.job_id,
                attempt.attempt_id,
            )
        return attempt

    @staticmethod
    def _attempt(row: Any) -> TrainingAttempt:
        payload = dict(row)
        payload["attempt_id"] = str(payload.pop("id"))
        payload["job_id"] = str(payload["job_id"])
        payload["scheduler_binding"] = _json_value(payload.get("scheduler_binding") or {})
        return TrainingAttempt.model_validate(payload)

    async def get_attempt(self, attempt_id: str) -> TrainingAttempt | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow("SELECT * FROM training_attempts WHERE id=$1::uuid", attempt_id)
        return self._attempt(row) if row else None

    async def list_attempts(self, job_id: str) -> list[TrainingAttempt]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM training_attempts WHERE job_id=$1::uuid ORDER BY attempt_number",
                job_id,
            )
        return [self._attempt(row) for row in rows]

    async def update_attempt_status(self, attempt_id: str, status: JobStatus) -> TrainingAttempt:
        pool = await self._get_pool()
        now = utc_now()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE training_attempts SET status=$2,
                  started_at=CASE WHEN $2='running' AND started_at IS NULL THEN $3 ELSE started_at END,
                  finished_at=CASE WHEN $2=ANY($4::text[]) THEN $3 ELSE finished_at END
                WHERE id=$1::uuid RETURNING *
                """,
                attempt_id,
                status.value,
                now,
                sorted(TERMINAL_STATES),
            )
        if not row:
            raise KeyError(f"training attempt not found: {attempt_id}")
        return self._attempt(row)

    async def add_artifact(self, artifact: TrainingArtifact) -> TrainingArtifact:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO training_artifacts(id,job_id,attempt_id,kind,uri,sha256,size_bytes,promoted,metadata,created_at)
                VALUES($1::uuid,$2::uuid,$3::uuid,$4,$5,$6,$7,$8,$9::jsonb,$10)
                ON CONFLICT(job_id,uri) DO UPDATE SET sha256=EXCLUDED.sha256,size_bytes=EXCLUDED.size_bytes,metadata=EXCLUDED.metadata
                """,
                artifact.artifact_id,
                artifact.job_id,
                artifact.attempt_id,
                artifact.kind,
                artifact.uri,
                artifact.sha256,
                artifact.size_bytes,
                artifact.promoted,
                canonical_json(artifact.metadata),
                artifact.created_at,
            )
        return artifact

    async def list_artifacts(self, job_id: str) -> list[TrainingArtifact]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch("SELECT * FROM training_artifacts WHERE job_id=$1::uuid ORDER BY created_at", job_id)
        result = []
        for row in rows:
            payload = dict(row)
            payload["artifact_id"] = str(payload.pop("id"))
            payload["job_id"] = str(payload["job_id"])
            payload["attempt_id"] = str(payload["attempt_id"]) if payload.get("attempt_id") else None
            payload["metadata"] = _json_value(payload.get("metadata") or {})
            result.append(TrainingArtifact.model_validate(payload))
        return result

    @staticmethod
    def _checkpoint(row: Any) -> CheckpointVersion:
        payload = dict(row)
        payload["checkpoint_id"] = str(payload.pop("id"))
        payload["job_id"] = str(payload["job_id"])
        payload["attempt_id"] = str(payload["attempt_id"]) if payload.get("attempt_id") else None
        payload["topology"] = _json_value(payload.pop("topology_json"))
        payload["metrics"] = _json_value(payload.pop("metrics_json"))
        return CheckpointVersion.model_validate(payload)

    async def add_checkpoint(self, checkpoint: CheckpointVersion) -> CheckpointVersion:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO checkpoint_versions(
                  id,job_id,attempt_id,step,manifest_uri,manifest_sha256,dataset_sha256,
                  tokenizer_sha256,topology_json,metrics_json,reload_verified,promoted,created_at
                ) VALUES($1::uuid,$2::uuid,$3::uuid,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb,$11,$12,$13)
                ON CONFLICT(job_id,step) DO UPDATE SET manifest_uri=checkpoint_versions.manifest_uri
                RETURNING *
                """,
                checkpoint.checkpoint_id,
                checkpoint.job_id,
                checkpoint.attempt_id,
                checkpoint.step,
                checkpoint.manifest_uri,
                checkpoint.manifest_sha256,
                checkpoint.dataset_sha256,
                checkpoint.tokenizer_sha256,
                canonical_json(checkpoint.topology),
                canonical_json(checkpoint.metrics),
                checkpoint.reload_verified,
                checkpoint.promoted,
                checkpoint.created_at,
            )
        stored = self._checkpoint(row)
        if not hmac.compare_digest(stored.manifest_sha256, checkpoint.manifest_sha256):
            raise ValueError("checkpoint step already exists with a different manifest")
        return stored

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointVersion | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow("SELECT * FROM checkpoint_versions WHERE id=$1::uuid", checkpoint_id)
        return self._checkpoint(row) if row else None

    async def list_checkpoints(self, job_id: str) -> list[CheckpointVersion]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM checkpoint_versions WHERE job_id=$1::uuid ORDER BY step DESC",
                job_id,
            )
        return [self._checkpoint(row) for row in rows]

    async def promote_checkpoint(self, checkpoint_id: str) -> CheckpointVersion:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "UPDATE checkpoint_versions SET promoted=true WHERE id=$1::uuid RETURNING *",
                checkpoint_id,
            )
        if not row:
            raise KeyError(f"checkpoint not found: {checkpoint_id}")
        return self._checkpoint(row)

    async def add_evaluation(self, evaluation: EvaluationRun) -> EvaluationRun:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO evaluation_runs(id,job_id,checkpoint_id,status,report_uri,report_sha256,decision,metrics_json,created_at)
                VALUES($1::uuid,$2::uuid,$3::uuid,$4,$5,$6,$7,$8::jsonb,$9)
                """,
                evaluation.evaluation_id,
                evaluation.job_id,
                evaluation.checkpoint_id,
                evaluation.status,
                evaluation.report_uri,
                evaluation.report_sha256,
                evaluation.decision,
                canonical_json(evaluation.metrics),
                evaluation.created_at,
            )
        return evaluation

    async def list_evaluations(self, job_id: str) -> list[EvaluationRun]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM evaluation_runs WHERE job_id=$1::uuid ORDER BY created_at DESC",
                job_id,
            )
        return [
            EvaluationRun(
                evaluation_id=str(row["id"]),
                job_id=str(row["job_id"]),
                checkpoint_id=str(row["checkpoint_id"]) if row["checkpoint_id"] else None,
                status=row["status"],
                report_uri=row["report_uri"],
                report_sha256=row["report_sha256"],
                decision=row["decision"],
                metrics=_json_value(row["metrics_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def create_service_account(self, account: ServiceAccount, token_hash: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO service_accounts(id,name,scopes,token_prefix,token_hash,enabled,created_at)
                VALUES($1::uuid,$2,$3::text[],$4,$5,$6,$7)
                """,
                account.service_account_id,
                account.name,
                account.scopes,
                account.token_prefix,
                token_hash,
                account.enabled,
                account.created_at,
            )

    async def ensure_service_account(self, account: ServiceAccount, token_hash: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            existing_hash = await connection.fetchval(
                "SELECT token_hash FROM service_accounts WHERE id=$1::uuid FOR UPDATE",
                account.service_account_id,
            )
            await connection.execute(
                """
                INSERT INTO service_accounts(id,name,scopes,token_prefix,token_hash,enabled,created_at)
                VALUES($1::uuid,$2,$3::text[],$4,$5,true,$6)
                ON CONFLICT(id) DO UPDATE SET
                  scopes=EXCLUDED.scopes,
                  token_prefix=EXCLUDED.token_prefix,
                  token_hash=EXCLUDED.token_hash,
                  enabled=true
                """,
                account.service_account_id,
                account.name,
                account.scopes,
                account.token_prefix,
                token_hash,
                account.created_at,
            )
            if existing_hash and not hmac.compare_digest(str(existing_hash), token_hash):
                await connection.execute(
                    "UPDATE service_account_sessions SET revoked_at=now() WHERE service_account_id=$1::uuid AND revoked_at IS NULL",
                    account.service_account_id,
                )

    async def authenticate_service_account(self, bootstrap_token: str) -> ServiceAccount | None:
        digest = sha256_text(bootstrap_token)
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow("SELECT * FROM service_accounts WHERE token_hash=$1 AND enabled=true", digest)
            if not row:
                return None
            now = utc_now()
            await connection.execute("UPDATE service_accounts SET last_used_at=$2 WHERE id=$1", row["id"], now)
        return ServiceAccount(
            service_account_id=str(row["id"]),
            name=row["name"],
            scopes=list(row["scopes"]),
            token_prefix=row["token_prefix"],
            enabled=row["enabled"],
            created_at=row["created_at"],
            last_used_at=now,
        )

    async def create_refresh_session(self, service_account_id: str, refresh_hash: str, expires_at: datetime) -> str:
        session_id = str(uuid.uuid4())
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO service_account_sessions(id,service_account_id,refresh_hash,expires_at) VALUES($1::uuid,$2::uuid,$3,$4)",
                session_id,
                service_account_id,
                refresh_hash,
                expires_at,
            )
        return session_id

    async def consume_refresh_session(self, session_id: str, refresh_token: str) -> ServiceAccount | None:
        digest = sha256_text(refresh_token)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT sa.* FROM service_account_sessions s
                JOIN service_accounts sa ON sa.id=s.service_account_id
                WHERE s.id=$1::uuid AND s.refresh_hash=$2 AND s.expires_at>now() AND s.revoked_at IS NULL AND sa.enabled=true
                """,
                session_id,
                digest,
            )
        if not row:
            return None
        return ServiceAccount(
            service_account_id=str(row["id"]),
            name=row["name"],
            scopes=list(row["scopes"]),
            token_prefix=row["token_prefix"],
            enabled=row["enabled"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )

    async def revoke_refresh_session(self, session_id: str, refresh_token: str) -> bool:
        digest = sha256_text(refresh_token)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            result = await connection.execute(
                """
                UPDATE service_account_sessions SET revoked_at=now()
                WHERE id=$1::uuid AND refresh_hash=$2 AND revoked_at IS NULL
                """,
                session_id,
                digest,
            )
        return result == "UPDATE 1"

    async def add_audit_event(self, event: TrainingAuditEvent) -> TrainingAuditEvent:
        pool = await self._get_pool()
        metadata = redact_payload(event.metadata)
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO training_audit_events(id,actor_id,job_id,action,outcome,request_id,metadata,created_at)
                VALUES($1::uuid,$2,$3::uuid,$4,$5,$6,$7::jsonb,$8)
                """,
                event.audit_id,
                event.actor_id,
                event.job_id,
                event.action,
                event.outcome,
                event.request_id,
                canonical_json(metadata),
                event.created_at,
            )
        return event.model_copy(update={"metadata": metadata})

    async def list_audit_events(self, job_id: str, *, limit: int = 100) -> list[TrainingAuditEvent]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM training_audit_events WHERE job_id=$1::uuid ORDER BY created_at DESC LIMIT $2",
                job_id,
                min(max(limit, 1), 500),
            )
        return [
            TrainingAuditEvent(
                audit_id=str(row["id"]),
                actor_id=row["actor_id"],
                job_id=str(row["job_id"]) if row["job_id"] else None,
                action=row["action"],
                outcome=row["outcome"],
                request_id=row["request_id"],
                metadata=_json_value(row["metadata"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class EventBus(Protocol):
    async def append(self, events: list[TrainingEvent]) -> None: ...
    async def read(self, job_id: str, *, after_sequence: int = 0, limit: int = 100, block_ms: int = 0) -> list[TrainingEvent]: ...
    async def close(self) -> None: ...


class InMemoryEventBus:
    def __init__(self, *, max_events_per_job: int = 100_000) -> None:
        self.events: dict[str, list[TrainingEvent]] = defaultdict(list)
        self.max_events_per_job = max_events_per_job
        self.condition = asyncio.Condition()

    async def append(self, events: list[TrainingEvent]) -> None:
        async with self.condition:
            for event in events:
                rows = self.events[event.job_id]
                rows.append(event.model_copy(deep=True))
                if len(rows) > self.max_events_per_job:
                    del rows[: len(rows) - self.max_events_per_job]
            self.condition.notify_all()

    async def read(self, job_id: str, *, after_sequence: int = 0, limit: int = 100, block_ms: int = 0) -> list[TrainingEvent]:
        def available() -> list[TrainingEvent]:
            return [item for item in self.events.get(job_id, []) if item.sequence > after_sequence][:limit]

        async with self.condition:
            rows = available()
            if not rows and block_ms > 0:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.condition.wait(), timeout=block_ms / 1000)
                rows = available()
            return [item.model_copy(deep=True) for item in rows]

    async def close(self) -> None:
        return None


class RedisEventBus:
    def __init__(self, redis_url: str, *, retention_events: int = 250_000, retention_seconds: int = 86_400) -> None:
        if not redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("Redis event bus requires redis:// or rediss:// URL")
        self.redis_url = redis_url
        self.retention_events = retention_events
        self.retention_seconds = min(max(retention_seconds, 3600), 7 * 86_400)
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=5, health_check_interval=30)
        return self._client

    @staticmethod
    def _key(job_id: str) -> str:
        return f"craftly:training:events:{job_id}"

    async def append(self, events: list[TrainingEvent]) -> None:
        client = await self._get_client()
        async with client.pipeline(transaction=True) as pipe:
            for event in events:
                key = self._key(event.job_id)
                pipe.xadd(
                    key,
                    {"event": canonical_json(event.model_dump(mode="json"))},
                    id=f"{event.sequence}-0",
                    maxlen=self.retention_events,
                    approximate=True,
                )
                pipe.expire(key, self.retention_seconds)
            await pipe.execute()

    async def read(self, job_id: str, *, after_sequence: int = 0, limit: int = 100, block_ms: int = 0) -> list[TrainingEvent]:
        client = await self._get_client()
        key = self._key(job_id)
        if block_ms > 0:
            response = await client.xread({key: f"{after_sequence}-0"}, count=limit, block=block_ms)
            rows = response[0][1] if response else []
        else:
            rows = await client.xrange(key, min=f"({after_sequence}-0", max="+", count=limit)
        return [TrainingEvent.model_validate_json(fields["event"]) for _, fields in rows]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_training_command(spec: TrainingJobSpec, *, output_dir: str) -> list[str]:
    """Build argv from validated fields only; no user-provided command is accepted."""

    spec.assert_current_model_contract()
    if spec.run_type == "data_pipeline":
        command = [
            sys.executable,
            "-u",
            "deploy/gpu/run_real_data_training_pipeline.py",
            "--sources",
            "config/data_sources.ultimate.json",
            "--work-dir",
            output_dir,
            "--model-profile",
            spec.model_profile,
            "--curriculum-mode",
            spec.curriculum_mode,
            "--steps",
            str(spec.steps),
            "--sequence-length",
            str(spec.sequence_length),
            "--batch-size",
            str(spec.batch_size),
            "--gradient-accumulation-steps",
            str(spec.gradient_accumulation_steps),
            "--dtype",
            spec.dtype,
            "--learning-rate",
            str(spec.optimizer.learning_rate),
            "--optimizer-beta1",
            str(spec.optimizer.beta1),
            "--optimizer-beta2",
            str(spec.optimizer.beta2),
            "--optimizer-epsilon",
            str(spec.optimizer.epsilon),
            "--weight-decay",
            str(spec.optimizer.weight_decay),
            "--gradient-clip-norm",
            str(spec.optimizer.gradient_clip_norm),
            "--learning-rate-schedule",
            spec.learning_rate.schedule,
            "--warmup-steps",
            str(spec.learning_rate.warmup_steps),
            "--warmup-ratio",
            str(spec.learning_rate.warmup_ratio),
            "--minimum-learning-rate-ratio",
            str(spec.learning_rate.minimum_learning_rate_ratio),
            "--validation-interval",
            str(spec.evaluation.interval_steps),
            "--validation-batches",
            str(spec.evaluation.validation_batches),
            "--checkpoint-every",
            str(spec.checkpoint.interval_steps),
            "--early-stopping-patience",
            str(spec.evaluation.early_stopping_patience),
            "--early-stopping-min-delta",
            str(spec.evaluation.early_stopping_min_delta),
            "--expected-python-version",
            spec.runtime_image.python_version,
            "--expected-pytorch-version",
            spec.runtime_image.pytorch_version,
            "--expected-cuda-version",
            spec.runtime_image.cuda_version,
            "--progress-to-stdout",
        ]
        if spec.validation_only:
            command.append("--kaggle-validation")
        else:
            command.extend(["--production", "--frontier-backend", "postgres"])
        return command
    if spec.run_type == "pretrain":
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.craftly.model_ops.pretrain_loop",
            "--manifest",
            str(spec.dataset_manifest_uri),
            "--tokenizer-path",
            str(spec.tokenizer_uri),
            "--manifest-sha256",
            str(spec.dataset_manifest_sha256),
            "--tokenizer-sha256",
            str(spec.tokenizer_sha256),
            "--optimizer-policy",
            "config/training_profiles.json",
            "--artifact-cache-dir",
            spec.dataloader.cache_path,
            "--output-dir",
            output_dir,
            "--device",
            "cuda",
            "--steps",
            str(spec.steps),
            "--sequence-length",
            str(spec.sequence_length),
            "--batch-size",
            str(spec.batch_size),
            "--gradient-accumulation-steps",
            str(spec.gradient_accumulation_steps),
            "--dtype",
            spec.dtype,
            "--learning-rate",
            str(spec.optimizer.learning_rate),
            "--optimizer-beta1",
            str(spec.optimizer.beta1),
            "--optimizer-beta2",
            str(spec.optimizer.beta2),
            "--optimizer-epsilon",
            str(spec.optimizer.epsilon),
            "--weight-decay",
            str(spec.optimizer.weight_decay),
            "--gradient-clip-norm",
            str(spec.optimizer.gradient_clip_norm),
            "--learning-rate-schedule",
            spec.learning_rate.schedule,
            "--warmup-steps",
            str(spec.learning_rate.warmup_steps),
            "--warmup-ratio",
            str(spec.learning_rate.warmup_ratio),
            "--minimum-learning-rate-ratio",
            str(spec.learning_rate.minimum_learning_rate_ratio),
            "--target-tokens",
            str(spec.token_budget.target_tokens),
            "--validate-every",
            str(spec.evaluation.interval_steps),
            "--validation-batches",
            str(spec.evaluation.validation_batches),
            "--checkpoint-every",
            str(spec.checkpoint.interval_steps),
            "--early-stopping-patience",
            str(spec.evaluation.early_stopping_patience),
            "--early-stopping-min-delta",
            str(spec.evaluation.early_stopping_min_delta),
            "--dataloader-prefetch-batches",
            str(spec.dataloader.prefetch_batches),
            "--dataloader-seed",
            str(spec.dataloader.seed),
            "--model-seed",
            str(spec.dataloader.seed),
            "--expected-python-version",
            spec.runtime_image.python_version,
            "--expected-pytorch-version",
            spec.runtime_image.pytorch_version,
            "--expected-cuda-version",
            spec.runtime_image.cuda_version,
            "--model-profile",
            spec.model_profile,
            "--distributed-strategy",
            spec.distributed_strategy,
            "--gradient-checkpointing",
        ]
        if spec.model_progression_decision_uri and spec.model_progression_decision_sha256:
            command.extend(
                [
                    "--model-progression-decision",
                    spec.model_progression_decision_uri,
                    "--model-progression-decision-sha256",
                    spec.model_progression_decision_sha256,
                ]
            )
        return command
    raise ValueError(f"scheduler does not execute run_type={spec.run_type}")


def scheduler_worker_command(spec: TrainingJobSpec, *, output_dir: str, scheduler: Literal["kubernetes", "slurm"]) -> list[str]:
    command = build_training_command(spec, output_dir=output_dir)
    prefix = [sys.executable, "-u", "-m", "src.craftly.model_ops.pretrain_loop"]
    if command[: len(prefix)] != prefix:
        raise ValueError("distributed scheduler workers only support the native Craftly pretraining module")
    return [
        sys.executable,
        "-u",
        "-m",
        "src.craftly.model_ops.distributed_worker",
        "--scheduler",
        scheduler,
        "--nodes",
        str(spec.resources.nodes),
        "--processes-per-node",
        str(spec.resources.gpus_per_node),
        "--",
        *command[len(prefix) :],
    ]


async def _run_argv(argv: list[str], *, stdin_text: str | None = None, timeout: float = 30.0) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(f"scheduler command timed out after {timeout}s: {argv[0]}")
    return process.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


class SchedulerAdapter(ABC):
    name: str

    @abstractmethod
    async def validate(self, spec: TrainingJobSpec) -> dict[str, Any]: ...

    @abstractmethod
    async def submit(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding: ...

    @abstractmethod
    async def status(self, binding: SchedulerBinding) -> str: ...

    @abstractmethod
    async def cancel(self, binding: SchedulerBinding) -> None: ...

    async def resume(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding:
        if not attempt.checkpoint_uri:
            raise ValueError("resume requires a verified checkpoint URI")
        return await self.submit(job, attempt)

    async def collect_runtime_metadata(self, binding: SchedulerBinding) -> dict[str, Any]:
        return binding.metadata

    async def rotate_credentials(self, job: TrainingJob, binding: SchedulerBinding) -> SchedulerBinding:
        return binding


def job_worker_token(job_id: str, *, ttl_seconds: int = 7200) -> tuple[str, int]:
    from src.craftly.identity.auth import create_jwt

    secret = os.environ.get("CRAFTLY_JWT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("CRAFTLY_JWT_SECRET length >= 32 is required to provision training workers")
    expires_at = int(time.time()) + ttl_seconds
    token = create_jwt(
        subject=f"worker:{job_id}",
        secret=secret,
        issuer=os.environ.get("CRAFTLY_JWT_ISSUER", "craftly-local"),
        audience=os.environ.get("CRAFTLY_JWT_AUDIENCE", "craftly-api"),
        scopes=["training:events:write", "training:artifacts:write", "training:jobs:read"],
        ttl_seconds=ttl_seconds,
        extra_claims={"job_id": job_id, "token_class": "training_worker"},
    )
    return token, expires_at


class NotebookValidationAdapter(SchedulerAdapter):
    name = "notebook"

    def __init__(self) -> None:
        self.processes: dict[str, asyncio.subprocess.Process] = {}

    async def validate(self, spec: TrainingJobSpec) -> dict[str, Any]:
        if spec.scheduler != self.name:
            raise ValueError("notebook adapter received a non-notebook job")
        if spec.distributed_strategy != "none" or spec.resources.nodes != 1:
            raise ValueError("notebook validation cannot run distributed jobs")
        return {"status": "ready", "production_ready": False, "reason": "validation client only"}

    async def submit(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding:
        await self.validate(job.spec)
        output_dir = f"artifacts/craftly/workspace/jobs/{job.job_id}/attempts/{attempt.attempt_number}"
        command = build_training_command(job.spec, output_dir=output_dir)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=None,
            stderr=None,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "CRAFTLY_TRAINING_JOB_ID": job.job_id,
                "CRAFTLY_TRAINING_ATTEMPT_ID": attempt.attempt_id,
            },
        )
        external_id = str(process.pid)
        self.processes[external_id] = process
        return SchedulerBinding(scheduler=self.name, external_id=external_id, metadata={"argv": command, "output_dir": output_dir})

    async def status(self, binding: SchedulerBinding) -> str:
        process = self.processes.get(binding.external_id)
        if process is None:
            return "unknown"
        result = process.returncode
        if result is None:
            return "running"
        return "succeeded" if result == 0 else "failed"

    async def cancel(self, binding: SchedulerBinding) -> None:
        process = self.processes.get(binding.external_id)
        if process and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=15)
            if process.returncode is None:
                process.kill()


class KubernetesSchedulerAdapter(SchedulerAdapter):
    name = "kubernetes"

    def __init__(self, *, namespace: str = "craftly-training") -> None:
        if not SAFE_IDENTIFIER.fullmatch(namespace):
            raise ValueError("unsafe Kubernetes namespace")
        self.namespace = namespace

    @staticmethod
    def node_selector(spec: TrainingJobSpec) -> dict[str, str]:
        selector = {"craftly.ai/gpu-memory-gib": str(spec.resources.gpu_memory_gib)} if spec.resources.gpus_per_node else {}
        if spec.resources.rdma_required:
            selector["craftly.ai/rdma"] = "true"
        return selector

    async def validate(self, spec: TrainingJobSpec) -> dict[str, Any]:
        if spec.scheduler != self.name:
            raise ValueError("Kubernetes Job adapter received an incompatible scheduler")
        if spec.resources.nodes != 1:
            raise ValueError("standard Kubernetes Job adapter supports one node; use kubernetes_pytorch for multi-node")
        if not shutil_which("kubectl"):
            try:
                await asyncio.to_thread(self._python_clients)
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("Kubernetes Python client or kubectl is required for Kubernetes scheduling") from exc
        topology = await self._validate_cluster_capacity(spec)
        return {
            "status": "ready",
            "namespace": self.namespace,
            "client": "python" if not shutil_which("kubectl") else "kubectl",
            "topology": topology,
        }

    @staticmethod
    def _eligible_nodes(payload: dict[str, Any], spec: TrainingJobSpec) -> list[str]:
        eligible: list[str] = []
        for node in payload.get("items", []):
            metadata = node.get("metadata", {})
            labels = metadata.get("labels", {}) or {}
            conditions = node.get("status", {}).get("conditions", [])
            ready = any(item.get("type") == "Ready" and item.get("status") == "True" for item in conditions)
            unschedulable = bool(node.get("spec", {}).get("unschedulable"))
            allocatable = node.get("status", {}).get("allocatable", {})
            gpu_count = int(allocatable.get("nvidia.com/gpu", 0))
            try:
                gpu_memory = int(labels.get("craftly.ai/gpu-memory-gib", 0))
            except (TypeError, ValueError):
                gpu_memory = 0
            rdma = str(labels.get("craftly.ai/rdma", "false")).lower() == "true"
            if not ready or unschedulable or gpu_count < spec.resources.gpus_per_node:
                continue
            if spec.resources.gpus_per_node and gpu_memory < spec.resources.gpu_memory_gib:
                continue
            if spec.resources.rdma_required and not rdma:
                continue
            eligible.append(str(metadata.get("name", "unknown")))
        return eligible

    async def _validate_cluster_capacity(self, spec: TrainingJobSpec) -> dict[str, Any]:
        if shutil_which("kubectl"):
            code, stdout, stderr = await _run_argv(["kubectl", "get", "nodes", "-o", "json"])
            if code != 0:
                raise RuntimeError(f"Kubernetes node inventory failed: {redact_text(stderr)[-2000:]}")
            payload = json.loads(stdout)
        else:
            core, _, _ = await asyncio.to_thread(self._python_clients)
            response = await asyncio.to_thread(core.list_node)
            from kubernetes.client import ApiClient

            payload = ApiClient().sanitize_for_serialization(response)
        eligible = self._eligible_nodes(payload, spec)
        if len(eligible) < spec.resources.nodes:
            requirements = (
                f"nodes={spec.resources.nodes}, gpus_per_node={spec.resources.gpus_per_node}, "
                f"gpu_memory_gib={spec.resources.gpu_memory_gib}, rdma={spec.resources.rdma_required}"
            )
            raise RuntimeError(
                f"cluster topology preflight found {len(eligible)} eligible nodes but requires {requirements}; "
                "label GPU nodes with craftly.ai/gpu-memory-gib and craftly.ai/rdma"
            )
        return {"eligible_nodes": eligible[: spec.resources.nodes], "eligible_count": len(eligible)}

    @staticmethod
    def _python_clients() -> tuple[Any, Any, Any]:
        try:
            from kubernetes import client, config
            from kubernetes.config.config_exception import ConfigException
        except ImportError as exc:
            raise RuntimeError("kubernetes Python package is not installed") from exc
        try:
            config.load_incluster_config()
        except ConfigException:
            try:
                config.load_kube_config()
            except ConfigException as exc:
                raise RuntimeError("Kubernetes client configuration is unavailable") from exc
        return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()

    def _python_apply_secret(self, manifest: dict[str, Any]) -> None:
        from kubernetes.client.exceptions import ApiException

        core, _, _ = self._python_clients()
        name = manifest["metadata"]["name"]
        namespace = manifest["metadata"]["namespace"]
        try:
            core.create_namespaced_secret(namespace=namespace, body=manifest)
        except ApiException as exc:
            if exc.status != 409:
                raise
            current = core.read_namespaced_secret(name=name, namespace=namespace)
            manifest["metadata"]["resourceVersion"] = current.metadata.resource_version
            core.replace_namespaced_secret(name=name, namespace=namespace, body=manifest)

    def _python_create_job(self, manifest: dict[str, Any]) -> None:
        from kubernetes.client.exceptions import ApiException

        _, batch, _ = self._python_clients()
        namespace = manifest["metadata"]["namespace"]
        name = manifest["metadata"]["name"]
        try:
            batch.create_namespaced_job(namespace=namespace, body=manifest)
        except ApiException as exc:
            if exc.status != 409:
                raise
            existing = batch.read_namespaced_job(name=name, namespace=namespace)
            labels = existing.metadata.labels or {}
            if labels.get("job-id") != manifest["metadata"]["labels"]["job-id"]:
                raise RuntimeError("existing Kubernetes Job has a different Craftly job identity") from exc

    def _python_job_status(self, binding: SchedulerBinding) -> str:
        _, batch, _ = self._python_clients()
        job = batch.read_namespaced_job(name=binding.external_id, namespace=binding.namespace or self.namespace)
        status = job.status
        if status.succeeded:
            return "succeeded"
        if status.failed:
            return "failed"
        if status.active:
            return "running"
        return "provisioning"

    def _python_delete_job(self, binding: SchedulerBinding) -> None:
        from kubernetes.client.exceptions import ApiException

        _, batch, _ = self._python_clients()
        try:
            batch.delete_namespaced_job(
                name=binding.external_id,
                namespace=binding.namespace or self.namespace,
                propagation_policy="Foreground",
            )
        except ApiException as exc:
            if exc.status != 404:
                raise

    def manifest(self, job: TrainingJob, attempt: TrainingAttempt) -> dict[str, Any]:
        name = f"craftly-{job.job_id[:8]}-a{attempt.attempt_number}"
        output_dir = f"/workspace/jobs/{job.job_id}/training"
        command = build_training_command(job.spec, output_dir=output_dir)
        image = job.spec.container_digest
        if "@" not in image:
            image = job.spec.runtime_image.repository + "@" + image
        namespace = job.spec.scheduler_policy.namespace
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": namespace, "labels": {"app": "craftly-training", "job-id": job.job_id}},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "activeDeadlineSeconds": job.spec.runtime_limits.maximum_wall_time_seconds,
                "template": {
                    "metadata": {"labels": {"app": "craftly-training", "job-id": job.job_id}},
                    "spec": {
                        "restartPolicy": "Never",
                        "serviceAccountName": "craftly-training-worker",
                        "priorityClassName": job.spec.scheduler_policy.priority_class,
                        "terminationGracePeriodSeconds": job.spec.runtime_limits.graceful_shutdown_seconds,
                        "nodeSelector": self.node_selector(job.spec),
                        "containers": [
                            {
                                "name": "trainer",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "command": command,
                                "env": [
                                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                                    {"name": "HOME", "value": "/workspace/home"},
                                    {"name": "TORCH_EXTENSIONS_DIR", "value": "/workspace/torch-extensions"},
                                    {"name": "CRAFTLY_TRAINING_JOB_ID", "value": job.job_id},
                                    {"name": "CRAFTLY_TRAINING_ATTEMPT_ID", "value": attempt.attempt_id},
                                    {"name": "CRAFTLY_WORKSPACE_URL", "valueFrom": {"configMapKeyRef": {"name": "craftly-training-config", "key": "workspace-url"}}},
                                    {"name": "CRAFTLY_WORKSPACE_TOKEN_FILE", "value": "/var/run/secrets/craftly/token"},
                                ],
                                "envFrom": [
                                    {"secretRef": {"name": name}}
                                    for name in job.spec.secret_references
                                ],
                                "resources": {
                                    "requests": {"cpu": str(job.spec.resources.cpu_cores), "memory": f"{job.spec.resources.memory_gib}Gi", "nvidia.com/gpu": str(job.spec.resources.gpus_per_node)},
                                    "limits": {"cpu": str(job.spec.resources.cpu_cores), "memory": f"{job.spec.resources.memory_gib}Gi", "nvidia.com/gpu": str(job.spec.resources.gpus_per_node)},
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "volumeMounts": [
                                    {"name": "workspace-token", "mountPath": "/var/run/secrets/craftly", "readOnly": True},
                                    {"name": "workspace", "mountPath": "/workspace"},
                                    {"name": "cache", "mountPath": job.spec.dataloader.cache_path},
                                    {"name": "tmp", "mountPath": "/tmp"},  # nosec B108 - memory-backed emptyDir
                                ],
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 10001,
                            "runAsGroup": 10001,
                            "fsGroup": 10001,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "volumes": [
                            {"name": "workspace-token", "secret": {"secretName": f"craftly-job-{job.job_id[:8]}"}},
                            {
                                "name": "workspace",
                                "persistentVolumeClaim": {
                                    "claimName": job.spec.scheduler_policy.shared_checkpoint_claim,
                                    "readOnly": False,
                                },
                            },
                            {"name": "cache", "emptyDir": {"sizeLimit": f"{job.spec.dataloader.local_cache_gib}Gi"}},
                            {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
                        ],
                    },
                },
            },
        }

    async def submit(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding:
        await self.validate(job.spec)
        token, expires_at = job_worker_token(job.job_id)
        secret_name = f"craftly-job-{job.job_id[:8]}"
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": job.spec.scheduler_policy.namespace, "labels": {"app": "craftly-training", "job-id": job.job_id}},
            "type": "Opaque",
            "stringData": {"token": token},
        }
        manifest = self.manifest(job, attempt)
        if shutil_which("kubectl"):
            secret_code, _, secret_error = await _run_argv(
                ["kubectl", "apply", "-f", "-"], stdin_text=canonical_json(secret_manifest)
            )
            if secret_code != 0:
                raise RuntimeError(f"kubectl training-token secret apply failed: {redact_text(secret_error)[-2000:]}")
            code, stdout, stderr = await _run_argv(["kubectl", "apply", "-f", "-"], stdin_text=canonical_json(manifest))
            if code != 0:
                raise RuntimeError(f"kubectl apply failed: {redact_text(stderr)[-2000:]}")
            apply_output = stdout.strip()
        else:
            await asyncio.to_thread(self._python_apply_secret, secret_manifest)
            await asyncio.to_thread(self._python_create_job, manifest)
            apply_output = "created-via-kubernetes-python-client"
        return SchedulerBinding(
            scheduler=self.name,
            external_id=manifest["metadata"]["name"],
            namespace=job.spec.scheduler_policy.namespace,
            metadata={"apply_output": apply_output, "token_secret": secret_name, "token_expires_at": expires_at},
        )

    async def rotate_credentials(self, job: TrainingJob, binding: SchedulerBinding) -> SchedulerBinding:
        expires_at = int(binding.metadata.get("token_expires_at", 0))
        if expires_at - int(time.time()) > 1800:
            return binding
        token, new_expiry = job_worker_token(job.job_id)
        secret_name = str(binding.metadata.get("token_secret") or f"craftly-job-{job.job_id[:8]}")
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": binding.namespace or self.namespace, "labels": {"app": "craftly-training", "job-id": job.job_id}},
            "type": "Opaque",
            "stringData": {"token": token},
        }
        if shutil_which("kubectl"):
            code, _, stderr = await _run_argv(["kubectl", "apply", "-f", "-"], stdin_text=canonical_json(secret_manifest))
            if code != 0:
                raise RuntimeError(f"worker token rotation failed: {redact_text(stderr)[-2000:]}")
        else:
            await asyncio.to_thread(self._python_apply_secret, secret_manifest)
        metadata = dict(binding.metadata)
        metadata["token_expires_at"] = new_expiry
        return binding.model_copy(update={"metadata": metadata})

    async def status(self, binding: SchedulerBinding) -> str:
        if not shutil_which("kubectl"):
            return await asyncio.to_thread(self._python_job_status, binding)
        code, stdout, stderr = await _run_argv(
            ["kubectl", "get", "job", binding.external_id, "-n", binding.namespace or self.namespace, "-o", "json"]
        )
        if code != 0:
            raise RuntimeError(f"kubectl get failed: {redact_text(stderr)[-2000:]}")
        status = json.loads(stdout).get("status", {})
        if status.get("succeeded"):
            return "succeeded"
        if status.get("failed"):
            return "failed"
        if status.get("active"):
            return "running"
        return "provisioning"

    async def cancel(self, binding: SchedulerBinding) -> None:
        if not shutil_which("kubectl"):
            await asyncio.to_thread(self._python_delete_job, binding)
            return
        code, _, stderr = await _run_argv(
            ["kubectl", "delete", "job", binding.external_id, "-n", binding.namespace or self.namespace, "--ignore-not-found=true"]
        )
        if code != 0:
            raise RuntimeError(f"kubectl delete failed: {redact_text(stderr)[-2000:]}")


class KubernetesPyTorchAdapter(KubernetesSchedulerAdapter):
    name = "kubernetes_pytorch"

    async def validate(self, spec: TrainingJobSpec) -> dict[str, Any]:
        if spec.scheduler != self.name:
            raise ValueError("PyTorchJob adapter received an incompatible scheduler")
        if spec.resources.nodes < 2:
            raise ValueError("PyTorchJob multi-node profile requires at least two nodes")
        if spec.distributed_strategy == "none":
            raise ValueError("PyTorchJob requires a distributed strategy")
        if shutil_which("kubectl"):
            code, stdout, _ = await _run_argv(["kubectl", "api-resources", "--api-group=kubeflow.org", "-o", "name"])
            if code != 0 or "pytorchjobs" not in stdout.lower():
                raise RuntimeError("Kubeflow Training Operator is required for multi-node PyTorchJob scheduling")
        else:
            try:
                _, _, custom = await asyncio.to_thread(self._python_clients)
                await asyncio.to_thread(
                    custom.list_cluster_custom_object,
                    group="kubeflow.org",
                    version="v1",
                    plural="pytorchjobs",
                    limit=1,
                )
            except Exception as exc:
                raise RuntimeError("Kubeflow Training Operator PyTorchJob CRD is unavailable") from exc
        topology = await self._validate_cluster_capacity(spec)
        return {
            "status": "ready",
            "namespace": self.namespace,
            "operator": "kubeflow-training-operator",
            "topology": topology,
        }

    def _python_create_job(self, manifest: dict[str, Any]) -> None:
        from kubernetes.client.exceptions import ApiException

        _, _, custom = self._python_clients()
        namespace = manifest["metadata"]["namespace"]
        name = manifest["metadata"]["name"]
        try:
            custom.create_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                body=manifest,
            )
        except ApiException as exc:
            if exc.status != 409:
                raise
            existing = custom.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                name=name,
            )
            labels = existing.get("metadata", {}).get("labels", {})
            if labels.get("job-id") != manifest["metadata"]["labels"]["job-id"]:
                raise RuntimeError("existing PyTorchJob has a different Craftly job identity") from exc

    def _python_job_status(self, binding: SchedulerBinding) -> str:
        _, _, custom = self._python_clients()
        payload = custom.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=binding.namespace or self.namespace,
            plural="pytorchjobs",
            name=binding.external_id,
        )
        conditions = payload.get("status", {}).get("conditions", [])
        active = {item.get("type") for item in conditions if item.get("status") == "True"}
        if "Succeeded" in active:
            return "succeeded"
        if "Failed" in active:
            return "failed"
        if "Running" in active:
            return "running"
        return "provisioning"

    def _python_delete_job(self, binding: SchedulerBinding) -> None:
        from kubernetes.client.exceptions import ApiException

        _, _, custom = self._python_clients()
        try:
            custom.delete_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=binding.namespace or self.namespace,
                plural="pytorchjobs",
                name=binding.external_id,
                body={"propagationPolicy": "Foreground"},
            )
        except ApiException as exc:
            if exc.status != 404:
                raise

    def manifest(self, job: TrainingJob, attempt: TrainingAttempt) -> dict[str, Any]:
        name = f"craftly-{job.job_id[:8]}-a{attempt.attempt_number}"
        output_dir = f"/workspace/jobs/{job.job_id}/training"
        base_command = scheduler_worker_command(job.spec, output_dir=output_dir, scheduler="kubernetes")
        image = job.spec.container_digest
        if "@" not in image:
            image = job.spec.runtime_image.repository + "@" + image
        container = {
            "name": "pytorch",
            "image": image,
            "command": base_command,
            "env": [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                {"name": "HOME", "value": "/workspace/home"},
                {"name": "TORCH_EXTENSIONS_DIR", "value": "/workspace/torch-extensions"},
                {"name": "CRAFTLY_TRAINING_JOB_ID", "value": job.job_id},
                {"name": "CRAFTLY_TRAINING_ATTEMPT_ID", "value": attempt.attempt_id},
                {"name": "CRAFTLY_WORKSPACE_URL", "valueFrom": {"configMapKeyRef": {"name": "craftly-training-config", "key": "workspace-url"}}},
                {"name": "CRAFTLY_WORKSPACE_TOKEN_FILE", "value": "/var/run/secrets/craftly/token"},
            ],
            "envFrom": [{"secretRef": {"name": secret_name}} for secret_name in job.spec.secret_references],
            "resources": {
                "requests": {"cpu": str(job.spec.resources.cpu_cores), "memory": f"{job.spec.resources.memory_gib}Gi", "nvidia.com/gpu": str(job.spec.resources.gpus_per_node)},
                "limits": {"cpu": str(job.spec.resources.cpu_cores), "memory": f"{job.spec.resources.memory_gib}Gi", "nvidia.com/gpu": str(job.spec.resources.gpus_per_node)},
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
            "volumeMounts": [
                {"name": "workspace-token", "mountPath": "/var/run/secrets/craftly", "readOnly": True},
                {"name": "workspace", "mountPath": "/workspace"},
                {"name": "cache", "mountPath": job.spec.dataloader.cache_path},
                {"name": "tmp", "mountPath": "/tmp"},  # nosec B108 - memory-backed emptyDir
            ],
        }
        replica = {
            "replicas": 1,
            "restartPolicy": "OnFailure",
            "template": {
                "spec": {
                    "serviceAccountName": "craftly-training-worker",
                    "priorityClassName": job.spec.scheduler_policy.priority_class,
                    "terminationGracePeriodSeconds": job.spec.runtime_limits.graceful_shutdown_seconds,
                    "nodeSelector": self.node_selector(job.spec),
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "fsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [container],
                    "volumes": [
                        {"name": "workspace-token", "secret": {"secretName": f"craftly-job-{job.job_id[:8]}"}},
                        {
                            "name": "workspace",
                            "persistentVolumeClaim": {
                                "claimName": job.spec.scheduler_policy.shared_checkpoint_claim,
                                "readOnly": False,
                            },
                        },
                        {
                            "name": "cache",
                            "emptyDir": {"sizeLimit": f"{job.spec.dataloader.local_cache_gib}Gi"},
                        },
                        {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
                    ],
                }
            },
        }
        worker = json.loads(canonical_json(replica))
        worker["replicas"] = job.spec.resources.nodes - 1
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {"name": name, "namespace": job.spec.scheduler_policy.namespace, "labels": {"app": "craftly-training", "job-id": job.job_id}},
            "spec": {
                "runPolicy": {
                    "cleanPodPolicy": "Running",
                    "activeDeadlineSeconds": job.spec.runtime_limits.maximum_wall_time_seconds,
                    "schedulingPolicy": {"minAvailable": job.spec.resources.nodes},
                },
                "pytorchReplicaSpecs": {"Master": replica, "Worker": worker},
            },
        }

    async def status(self, binding: SchedulerBinding) -> str:
        code, stdout, stderr = await _run_argv(
            ["kubectl", "get", "pytorchjob", binding.external_id, "-n", binding.namespace or self.namespace, "-o", "json"]
        )
        if code != 0:
            raise RuntimeError(f"kubectl get PyTorchJob failed: {redact_text(stderr)[-2000:]}")
        conditions = json.loads(stdout).get("status", {}).get("conditions", [])
        active = {item.get("type") for item in conditions if item.get("status") == "True"}
        if "Succeeded" in active:
            return "succeeded"
        if "Failed" in active:
            return "failed"
        if "Running" in active:
            return "running"
        return "provisioning"

    async def cancel(self, binding: SchedulerBinding) -> None:
        code, _, stderr = await _run_argv(
            ["kubectl", "delete", "pytorchjob", binding.external_id, "-n", binding.namespace or self.namespace, "--ignore-not-found=true"]
        )
        if code != 0:
            raise RuntimeError(f"kubectl delete PyTorchJob failed: {redact_text(stderr)[-2000:]}")


class SlurmSchedulerAdapter(SchedulerAdapter):
    name = "slurm"

    def __init__(self, *, work_dir: str | Path = "artifacts/craftly/slurm") -> None:
        self.work_dir = Path(work_dir)

    async def validate(self, spec: TrainingJobSpec) -> dict[str, Any]:
        if spec.scheduler != self.name:
            raise ValueError("Slurm adapter received an incompatible scheduler")
        missing = [name for name in ["sbatch", "sacct", "scancel", "sinfo"] if not shutil_which(name)]
        if missing:
            raise RuntimeError("Slurm commands are missing: " + ", ".join(missing))
        if spec.resources.nodes < 2 or not spec.resources.rdma_required:
            raise ValueError("60B-class Slurm profiles require multi-node RDMA topology")
        workspace_url = os.environ.get("CRAFTLY_WORKSPACE_URL", "")
        if not workspace_url.startswith("https://"):
            raise RuntimeError("Slurm production workers require an HTTPS CRAFTLY_WORKSPACE_URL")
        shared_root_value = os.environ.get("CRAFTLY_SHARED_CHECKPOINT_ROOT", "")
        shared_root = Path(shared_root_value).expanduser().resolve() if shared_root_value else None
        if shared_root is None or not shared_root.is_absolute() or not shared_root.is_dir():
            raise RuntimeError("CRAFTLY_SHARED_CHECKPOINT_ROOT must be a mounted shared filesystem directory")
        if not os.access(shared_root, os.W_OK):
            raise RuntimeError("CRAFTLY_SHARED_CHECKPOINT_ROOT is not writable")
        code, stdout, stderr = await _run_argv(["sinfo", "-N", "-h", "-o", "%N|%t|%G|%f"])
        if code != 0:
            raise RuntimeError(f"Slurm topology inventory failed: {redact_text(stderr)[-2000:]}")
        eligible = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            node, state, gres, features = parts
            gpu_counts = [int(value) for value in re.findall(r"gpu(?::[^:,()]+)?:(\d+)", gres.lower())]
            feature_set = {value.strip().lower() for value in features.split(",") if value.strip()}
            memory_labels = {
                f"gpu-mem-{spec.resources.gpu_memory_gib}g",
                f"gpu-memory-{spec.resources.gpu_memory_gib}g",
            }
            if state.lower().rstrip("*") not in {"idle", "mix", "alloc"}:
                continue
            if max(gpu_counts, default=0) < spec.resources.gpus_per_node:
                continue
            if not memory_labels.intersection(feature_set) or "rdma" not in feature_set:
                continue
            eligible.append(node)
        if len(set(eligible)) < spec.resources.nodes:
            raise RuntimeError(
                f"Slurm preflight found {len(set(eligible))} eligible nodes; requires {spec.resources.nodes} nodes with "
                f"{spec.resources.gpus_per_node} GPUs, gpu-memory-{spec.resources.gpu_memory_gib}g and rdma features"
            )
        return {"status": "ready", "commands": ["sbatch", "sacct", "scancel", "sinfo"], "eligible_nodes": sorted(set(eligible))}

    def script(self, job: TrainingJob, attempt: TrainingAttempt) -> str:
        shared_root = Path(os.environ.get("CRAFTLY_SHARED_CHECKPOINT_ROOT", "")).expanduser().resolve()
        output_dir = str(shared_root / "jobs" / job.job_id / "training")
        if job.spec.distributed_strategy == "megatron":
            megatron_command = self._megatron_command(job, output_dir=output_dir)
            command = [
                sys.executable,
                "-u",
                "-m",
                "src.craftly.model_ops.distributed_worker",
                "--scheduler",
                "slurm",
                "--target",
                "megatron",
                "--nodes",
                str(job.spec.resources.nodes),
                "--processes-per-node",
                str(job.spec.resources.gpus_per_node),
                "--",
                *megatron_command[1:],
            ]
        else:
            command = scheduler_worker_command(job.spec, output_dir=output_dir, scheduler="slurm")
        quoted = " ".join(shlex.quote(item) for item in command)
        token_path = self.work_dir / f"{job.job_id}-token"
        wall_seconds = job.spec.runtime_limits.maximum_wall_time_seconds
        wall_days, remainder = divmod(wall_seconds, 86_400)
        wall_hours, remainder = divmod(remainder, 3600)
        wall_minutes, wall_seconds_remainder = divmod(remainder, 60)
        slurm_time = f"{wall_days}-{wall_hours:02d}:{wall_minutes:02d}:{wall_seconds_remainder:02d}"
        scheduler_lines = [
            f"#SBATCH --partition={job.spec.scheduler_policy.queue}",
            f"#SBATCH --time={slurm_time}",
        ]
        if job.spec.scheduler_policy.account:
            scheduler_lines.append(f"#SBATCH --account={job.spec.scheduler_policy.account}")
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"#SBATCH --job-name=craftly-{job.job_id[:8]}",
                f"#SBATCH --nodes={job.spec.resources.nodes}",
                f"#SBATCH --ntasks-per-node={job.spec.resources.gpus_per_node}",
                f"#SBATCH --gpus-per-node={job.spec.resources.gpus_per_node}",
                f"#SBATCH --cpus-per-task={max(1, job.spec.resources.cpu_cores // max(1, job.spec.resources.gpus_per_node))}",
                f"#SBATCH --output={shlex.quote(output_dir)}/slurm-%j.out",
                *scheduler_lines,
                "export PYTHONUNBUFFERED=1",
                "export NCCL_ASYNC_ERROR_HANDLING=1",
                f"export CRAFTLY_TRAINING_JOB_ID={shlex.quote(job.job_id)}",
                f"export CRAFTLY_TRAINING_ATTEMPT_ID={shlex.quote(attempt.attempt_id)}",
                f"export CRAFTLY_WORKSPACE_TOKEN_FILE={shlex.quote(str(token_path))}",
                f"export CRAFTLY_WORKSPACE_URL={shlex.quote(os.environ.get('CRAFTLY_WORKSPACE_URL', ''))}",
                "export MASTER_PORT=${MASTER_PORT:-29500}",
                f"srun {quoted}",
                "",
            ]
        )

    def _megatron_command(self, job: TrainingJob, *, output_dir: str) -> list[str]:
        root_value = os.environ.get("CRAFTLY_MEGATRON_ROOT", "")
        if not root_value:
            raise RuntimeError("CRAFTLY_MEGATRON_ROOT is required for Megatron jobs")
        root = Path(root_value).expanduser().resolve()
        pretrain = (root / "pretrain_gpt.py").resolve()
        if not pretrain.is_file() or root not in pretrain.parents:
            raise RuntimeError("CRAFTLY_MEGATRON_ROOT does not contain pretrain_gpt.py")
        manifest_value = str(job.spec.dataset_manifest_uri or "")
        tokenizer = str(job.spec.tokenizer_uri or "")
        if (
            (urlparse(manifest_value).scheme and not Path(manifest_value).is_absolute())
            or (urlparse(tokenizer).scheme and not Path(tokenizer).is_absolute())
        ):
            raise ValueError("Megatron Slurm jobs require cluster-mounted immutable dataset and tokenizer paths")
        dataset_manifest = Path(manifest_value).expanduser().resolve()
        tokenizer_path = Path(tokenizer).expanduser().resolve()
        if not dataset_manifest.is_file() or sha256_path(dataset_manifest) != job.spec.dataset_manifest_sha256:
            raise ValueError("Megatron dataset manifest is missing or does not match its immutable checksum")
        data_path = resolve_megatron_data_prefix(dataset_manifest)
        if not tokenizer_path.is_file() or sha256_path(tokenizer_path) != job.spec.tokenizer_sha256:
            raise ValueError("Megatron tokenizer is missing or does not match its immutable checksum")
        profile = job.spec.assert_current_model_contract()
        parallelism = job.spec.parallelism
        world_size = job.spec.resources.nodes * job.spec.resources.gpus_per_node
        topology = profile.distributed_topology_report(
            tensor_parallel=parallelism.tensor_parallel,
            pipeline_parallel=parallelism.pipeline_parallel,
            data_parallel=parallelism.data_parallel,
            context_parallel=parallelism.context_parallel,
            expert_parallel=parallelism.expert_parallel,
            gpus_per_node=job.spec.resources.gpus_per_node,
        )
        if not topology["passed"] or topology["world_size"] != world_size:
            raise ValueError("Megatron job topology does not match the canonical model contract")
        return [
            sys.executable,
            "-u",
            str(pretrain),
            "--tensor-model-parallel-size",
            str(parallelism.tensor_parallel),
            "--pipeline-model-parallel-size",
            str(parallelism.pipeline_parallel),
            "--context-parallel-size",
            str(parallelism.context_parallel),
            *(
                ["--expert-model-parallel-size", str(parallelism.expert_parallel)]
                if profile.feed_forward_architecture == "moe"
                else []
            ),
            *megatron_model_arguments(profile),
            "--sequence-parallel",
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
            "--ckpt-format",
            "torch_dist",
            "--dist-ckpt-optim-fully-reshardable",
            "--seq-length",
            str(job.spec.sequence_length),
            "--max-position-embeddings",
            str(job.spec.sequence_length),
            "--micro-batch-size",
            str(job.spec.batch_size),
            "--global-batch-size",
            str(job.spec.token_budget.global_batch_sequences),
            "--train-iters",
            str(job.spec.steps),
            "--lr",
            str(job.spec.optimizer.learning_rate),
            "--min-lr",
            str(job.spec.optimizer.learning_rate * job.spec.learning_rate.minimum_learning_rate_ratio),
            "--lr-decay-style",
            job.spec.learning_rate.schedule,
            "--lr-warmup-iters",
            str(job.spec.learning_rate.warmup_steps),
            "--weight-decay",
            str(job.spec.optimizer.weight_decay),
            "--clip-grad",
            str(job.spec.optimizer.gradient_clip_norm),
            "--bf16",
            "--data-path",
            str(data_path),
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            str(tokenizer_path),
            "--save",
            str(Path(output_dir).resolve() / "checkpoints"),
            "--load",
            str(Path(output_dir).resolve() / "checkpoints"),
            "--save-interval",
            str(job.spec.checkpoint.interval_steps),
            "--eval-interval",
            str(job.spec.evaluation.interval_steps),
            "--eval-iters",
            str(job.spec.evaluation.validation_batches),
        ]

    async def submit(self, job: TrainingJob, attempt: TrainingAttempt) -> SchedulerBinding:
        await self.validate(job.spec)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        token, expires_at = job_worker_token(job.job_id)
        token_path = self.work_dir / f"{job.job_id}-token"
        token_path.write_text(token, encoding="utf-8")
        with contextlib.suppress(OSError):
            token_path.chmod(0o600)
        script_path = self.work_dir / f"{job.job_id}-attempt-{attempt.attempt_number}.sbatch"
        script_path.write_text(self.script(job, attempt), encoding="utf-8", newline="\n")
        code, stdout, stderr = await _run_argv(["sbatch", "--parsable", str(script_path)])
        if code != 0:
            raise RuntimeError(f"sbatch failed: {redact_text(stderr)[-2000:]}")
        external_id = stdout.strip().split(";", 1)[0]
        if not external_id.isdigit():
            raise RuntimeError(f"sbatch returned invalid job id: {redact_text(stdout)}")
        return SchedulerBinding(
            scheduler=self.name,
            external_id=external_id,
            metadata={"script_path": str(script_path), "token_path": str(token_path), "token_expires_at": expires_at},
        )

    async def rotate_credentials(self, job: TrainingJob, binding: SchedulerBinding) -> SchedulerBinding:
        expires_at = int(binding.metadata.get("token_expires_at", 0))
        if expires_at - int(time.time()) > 1800:
            return binding
        token_path = Path(str(binding.metadata.get("token_path") or self.work_dir / f"{job.job_id}-token"))
        token, new_expiry = job_worker_token(job.job_id)
        temporary = token_path.with_suffix(".tmp")
        temporary.write_text(token, encoding="utf-8")
        with contextlib.suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, token_path)
        metadata = dict(binding.metadata)
        metadata["token_expires_at"] = new_expiry
        return binding.model_copy(update={"metadata": metadata})

    async def status(self, binding: SchedulerBinding) -> str:
        code, stdout, stderr = await _run_argv(["sacct", "-j", binding.external_id, "--noheader", "--parsable2", "--format=State"])
        if code != 0:
            raise RuntimeError(f"sacct failed: {redact_text(stderr)[-2000:]}")
        state = stdout.strip().splitlines()[0].split("|", 1)[0].split("+", 1)[0].upper() if stdout.strip() else "UNKNOWN"
        if state in {"COMPLETED"}:
            return "succeeded"
        if state in {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"}:
            return "failed"
        if state in {"CANCELLED", "PREEMPTED"}:
            return "cancelled"
        if state in {"RUNNING", "COMPLETING"}:
            return "running"
        return "provisioning"

    async def cancel(self, binding: SchedulerBinding) -> None:
        code, _, stderr = await _run_argv(["scancel", binding.external_id])
        if code != 0:
            raise RuntimeError(f"scancel failed: {redact_text(stderr)[-2000:]}")


def shutil_which(executable: str) -> str | None:
    import shutil

    return shutil.which(executable)


class TrainingWorkspaceService:
    def __init__(
        self,
        *,
        store: TrainingStore,
        events: EventBus,
        object_store: ObjectStore,
        profiles: TrainingProfileRegistry | None = None,
        campaigns: QualificationCampaignRegistry | None = None,
        production_mode: bool = False,
    ) -> None:
        self.store = store
        self.events = events
        self.object_store = object_store
        self.profiles = profiles or TrainingProfileRegistry.from_file()
        self.campaigns = campaigns or QualificationCampaignRegistry.from_file()
        self.production_mode = production_mode

    @classmethod
    def from_environment(cls) -> "TrainingWorkspaceService":
        database_url = os.environ.get("CRAFTLY_DATABASE_URL")
        redis_url = os.environ.get("CRAFTLY_REDIS_URL")
        object_uri = os.environ.get("CRAFTLY_OBJECT_STORE_URI", "local://artifacts/craftly/training-workspace")
        production_mode = os.environ.get("CRAFTLY_ENV", "development") == "production"
        if production_mode:
            missing = [name for name, value in [("CRAFTLY_DATABASE_URL", database_url), ("CRAFTLY_REDIS_URL", redis_url), ("CRAFTLY_OBJECT_STORE_URI", object_uri if object_uri.startswith("s3://") else None)] if not value]
            if missing:
                raise RuntimeError("production training workspace dependencies missing: " + ", ".join(missing))
        store: TrainingStore = PostgresTrainingStore(database_url) if database_url else InMemoryTrainingStore()
        event_bus: EventBus = RedisEventBus(redis_url) if redis_url else InMemoryEventBus()
        object_store = create_object_store(
            ObjectStoreConfig(uri=object_uri, endpoint_url=os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL"))
        )
        return cls(store=store, events=event_bus, object_store=object_store, production_mode=production_mode)

    def profile_report(self) -> list[dict[str, Any]]:
        return [
            {
                **profile.model_dump(mode="json"),
                "profile_hash": profile.immutable_hash,
                "readiness": (
                    "validation_ready"
                    if profile.dev_only
                    else "built_not_cluster_proven"
                    if profile.resources.nodes > 1
                    else "production_ready_requires_external_service"
                ),
            }
            for profile in self.profiles.profiles
        ]

    def resolve_spec(self, request: TrainingJobCreateRequest) -> TrainingJobSpec:
        profile = self.profiles.latest(request.profile_id)
        campaign: QualificationCampaign | None = None
        milestone: QualificationMilestone | None = None
        if request.qualification_campaign_id and request.qualification_milestone_id:
            campaign = self.campaigns.latest(request.qualification_campaign_id)
            milestone = campaign.milestone(request.qualification_milestone_id)
            if milestone.profile_id != profile.profile_id:
                raise ValueError(
                    f"qualification milestone requires profile {milestone.profile_id}, not {profile.profile_id}"
                )
            if request.overrides.get("steps") != milestone.steps:
                raise ValueError(
                    f"qualification milestone {milestone.milestone_id} requires immutable steps={milestone.steps}"
                )
        git_commit = request.git_commit
        container_digest = request.container_digest
        if self.production_mode:
            git_commit = os.environ.get("CRAFTLY_TRAINING_GIT_COMMIT", git_commit)
            container_digest = os.environ.get("CRAFTLY_TRAINING_IMAGE_DIGEST", container_digest)
            if git_commit == "0000000" or not SAFE_GIT_COMMIT.fullmatch(git_commit):
                raise ValueError("production jobs require CRAFTLY_TRAINING_GIT_COMMIT with a real immutable commit")
            if container_digest == "sha256:" + ("0" * 64) or not SAFE_CONTAINER_DIGEST.fullmatch(container_digest):
                raise ValueError("production jobs require CRAFTLY_TRAINING_IMAGE_DIGEST with a real immutable image digest")
        values = {
            "steps": profile.steps,
            "sequence_length": profile.sequence_length,
            "batch_size": profile.batch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        }
        resources = profile.resources.model_copy(deep=True)
        for key, value in request.overrides.items():
            bounds = profile.allowed_overrides.get(key)
            if bounds is None:
                raise ValueError(f"profile does not permit override: {key}")
            if not bounds.minimum <= value <= bounds.maximum:
                raise ValueError(f"override {key} must be between {bounds.minimum} and {bounds.maximum}")
            if key == "nodes":
                resources = resources.model_copy(update={"nodes": value})
            elif key in values:
                values[key] = value
            elif key == "max_docs":
                continue
            else:
                raise ValueError(f"unsupported bounded override: {key}")
        if self.production_mode and profile.dev_only:
            raise ValueError("dev-only notebook profile cannot be submitted in production mode")
        data_parallel = training_data_parallel_size(
            resources=resources,
            distributed_strategy=profile.distributed_strategy,
            parallelism=profile.parallelism,
        )
        batch_contract = TrainingBatchContract(
            optimizer_steps=values["steps"],
            sequence_length=values["sequence_length"],
            micro_batch_size=values["batch_size"],
            gradient_accumulation_steps=values["gradient_accumulation_steps"],
            data_parallel_size=data_parallel,
        )
        global_batch_sequences = batch_contract.global_batch_sequences
        target_tokens = batch_contract.target_tokens
        token_budget = profile.token_budget.model_copy(
            update={
                "global_batch_sequences": global_batch_sequences,
                "target_tokens": target_tokens,
                "maximum_tokens": max(
                    profile.token_budget.maximum_tokens,
                    target_tokens,
                ),
            }
        )
        if self.production_mode and profile.run_type == "pretrain":
            for name, uri in (
                ("dataset_manifest_uri", request.dataset_manifest_uri),
                ("tokenizer_uri", request.tokenizer_uri),
            ):
                scheme = urlparse(uri or "").scheme or "file"
                if scheme not in profile.immutable_inputs.allowed_uri_schemes:
                    raise ValueError(f"{name} scheme {scheme!r} is not allowed by the immutable input policy")
            if profile.immutable_inputs.require_promoted_dataset and request.metadata.get("dataset_promotion") != "promoted":
                raise ValueError("production pretraining requires metadata.dataset_promotion='promoted'")
        progression = None
        if "model_progression_decision" in profile.requirements:
            if not request.model_progression_decision_uri or not request.model_progression_decision_sha256:
                raise ValueError(
                    f"profile {profile.profile_id} requires an immutable model progression decision"
                )
            raw_decision_uri = request.model_progression_decision_uri
            decision_uri = urlparse(raw_decision_uri)
            windows_path = re.match(r"^[A-Za-z]:[\\/]", raw_decision_uri) is not None
            if decision_uri.scheme not in {"", "file"} and not windows_path:
                raise ValueError(
                    "model progression decision must be locally mounted for semantic verification"
                )
            decision_path = Path(
                decision_uri.path if decision_uri.scheme == "file" else raw_decision_uri
            ).expanduser().resolve(strict=True)
            if not hmac.compare_digest(
                sha256_path(decision_path),
                request.model_progression_decision_sha256,
            ):
                raise ValueError("model progression decision SHA-256 mismatch")
            from src.craftly.learning.ablation_runner import verify_model_progression_decision

            progression = verify_model_progression_decision(decision_path)
            if progression.status != "authorized" or progression.target_model_profile != profile.model_profile:
                raise ValueError("model progression decision does not authorize this model profile")
        metadata = dict(request.metadata)
        metadata["requested_overrides"] = request.overrides
        canonical_model = get_model_profile(profile.model_profile)
        if progression is not None:
            canonical_model = canonical_model.model_copy(
                update={"vocab_size": progression.selected_tokenizer_vocab_size}
            )
            if not hmac.compare_digest(
                canonical_model.contract_sha256(),
                str(progression.target_model_contract_sha256 or ""),
            ):
                raise ValueError("model progression target contract differs from the current profile")
        model_contract = canonical_model.model_dump(mode="json")
        parameter_report_sha256 = sha256_text(
            canonical_json(canonical_model.parameter_report())
        )
        return TrainingJobSpec(
            profile_id=profile.profile_id,
            profile_version=profile.version,
            profile_hash=profile.immutable_hash,
            project_id=request.project_id,
            run_type=profile.run_type,
            validation_only=profile.dev_only,
            model_profile=profile.model_profile,
            model_contract=model_contract,
            model_contract_sha256=canonical_model.contract_sha256(),
            model_parameter_report_sha256=parameter_report_sha256,
            curriculum_mode=profile.curriculum_mode,
            scheduler=profile.scheduler,
            distributed_strategy=profile.distributed_strategy,
            parallelism=profile.parallelism,
            steps=values["steps"],
            sequence_length=values["sequence_length"],
            batch_size=values["batch_size"],
            gradient_accumulation_steps=values["gradient_accumulation_steps"],
            dtype=profile.dtype,
            resources=resources,
            optimizer=profile.optimizer,
            learning_rate=profile.learning_rate,
            token_budget=token_budget,
            checkpoint=profile.checkpoint,
            evaluation=profile.evaluation,
            retention=profile.retention,
            promotion=profile.promotion,
            retry=profile.retry,
            runtime_limits=profile.runtime_limits,
            dataloader=profile.dataloader,
            runtime_image=profile.runtime_image,
            scheduler_policy=profile.scheduler_policy,
            cost_quota=profile.cost_quota,
            immutable_inputs=profile.immutable_inputs,
            dataset_manifest_uri=request.dataset_manifest_uri,
            dataset_manifest_sha256=request.dataset_manifest_sha256,
            tokenizer_uri=request.tokenizer_uri,
            tokenizer_sha256=request.tokenizer_sha256,
            model_progression_decision_uri=request.model_progression_decision_uri,
            model_progression_decision_sha256=request.model_progression_decision_sha256,
            git_commit=git_commit,
            container_digest=container_digest,
            requirements=profile.requirements,
            secret_references=profile.secret_references,
            metadata=metadata,
            qualification_campaign_id=campaign.campaign_id if campaign else None,
            qualification_campaign_version=campaign.version if campaign else None,
            qualification_milestone_id=milestone.milestone_id if milestone else None,
        )

    async def record_audit(
        self,
        *,
        actor_id: str,
        action: str,
        outcome: Literal["accepted", "succeeded", "denied", "failed"],
        job_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingAuditEvent:
        return await self.store.add_audit_event(
            TrainingAuditEvent(
                audit_id=str(uuid.uuid4()),
                actor_id=actor_id,
                job_id=job_id,
                action=action,
                outcome=outcome,
                request_id=request_id,
                metadata=redact_payload(metadata or {}),
            )
        )

    async def create_job(self, request: TrainingJobCreateRequest, *, owner_id: str) -> TrainingJob:
        existing = await self.store.get_job_by_idempotency(owner_id, request.idempotency_key)
        spec = self.resolve_spec(request)
        if existing:
            if not hmac.compare_digest(existing.spec.spec_hash, spec.spec_hash):
                raise ValueError("idempotency key already exists with a different immutable job spec")
            return existing
        await self._validate_qualification_predecessor(spec, owner_id=owner_id)
        active_jobs = [
            item
            for item in await self.store.list_jobs(owner_id=owner_id, limit=500)
            if item.status.value not in TERMINAL_STATES
        ]
        if len(active_jobs) >= spec.cost_quota.maximum_concurrent_jobs_per_owner:
            raise ValueError(
                "owner training concurrency quota exceeded: "
                f"{len(active_jobs)} active, limit={spec.cost_quota.maximum_concurrent_jobs_per_owner}"
            )
        requested_gpu_hours = (
            spec.runtime_limits.maximum_wall_time_seconds
            * spec.resources.nodes
            * spec.resources.gpus_per_node
            / 3600.0
        )
        if requested_gpu_hours > spec.cost_quota.maximum_gpu_hours:
            raise ValueError(
                f"job requests up to {requested_gpu_hours:.2f} GPU-hours but profile quota allows "
                f"{spec.cost_quota.maximum_gpu_hours:.2f}"
            )
        requested_cost_usd = requested_gpu_hours * spec.cost_quota.estimated_gpu_hour_cost_usd
        if requested_cost_usd > spec.cost_quota.maximum_cost_usd:
            raise ValueError(
                f"job projected cost is ${requested_cost_usd:.2f} but profile quota allows "
                f"${spec.cost_quota.maximum_cost_usd:.2f}"
            )
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            owner_id=owner_id,
            idempotency_key=request.idempotency_key,
            spec=spec,
            status=JobStatus.VALIDATING,
            version=1,
        )
        await self.store.sync_profiles(self.profiles.profiles)
        created = await self.store.create_job(job)
        if not hmac.compare_digest(created.spec.spec_hash, spec.spec_hash):
            raise ValueError("idempotency key was concurrently created with a different immutable job spec")
        stored = await asyncio.to_thread(
            self.object_store.put_json,
            spec.model_dump(mode="json"),
            key=f"jobs/{job.job_id}/spec.json",
        )
        await self.store.add_artifact(
            TrainingArtifact(
                artifact_id=str(uuid.uuid4()),
                job_id=job.job_id,
                kind="spec",
                uri=stored.uri,
                sha256=stored.sha256,
                size_bytes=stored.size_bytes,
            )
        )
        queued = await self.store.transition_job(created.job_id, JobStatus.QUEUED, expected_version=created.version)
        await self.record_audit(
            actor_id=owner_id,
            job_id=queued.job_id,
            action="training.job.create",
            outcome="accepted",
            metadata={"profile_id": queued.spec.profile_id, "spec_hash": queued.spec.spec_hash},
        )
        return queued

    async def _validate_qualification_predecessor(self, spec: TrainingJobSpec, *, owner_id: str) -> None:
        if not spec.qualification_campaign_id or not spec.qualification_milestone_id:
            return
        campaign = self.campaigns.latest(spec.qualification_campaign_id)
        milestone = campaign.milestone(spec.qualification_milestone_id)
        if milestone.previous_milestone_id is None:
            return
        candidates = [
            item
            for item in await self.store.list_jobs(owner_id=owner_id, limit=500)
            if item.spec.qualification_campaign_id == campaign.campaign_id
            and item.spec.qualification_campaign_version == campaign.version
            and item.spec.qualification_milestone_id == milestone.previous_milestone_id
        ]
        if not candidates:
            raise ValueError(
                f"qualification milestone {milestone.milestone_id} is locked; "
                f"previous milestone {milestone.previous_milestone_id} has no job"
            )
        previous = max(candidates, key=lambda item: item.created_at)
        gate = campaign.gate
        if gate.require_succeeded_job and previous.status != JobStatus.SUCCEEDED:
            raise ValueError(
                f"qualification milestone is locked; previous job status={previous.status.value}, expected=succeeded"
            )
        if gate.require_same_dataset_and_tokenizer:
            for name in ("dataset_manifest_sha256", "tokenizer_sha256"):
                previous_value = getattr(previous.spec, name)
                current_value = getattr(spec, name)
                if not previous_value or not current_value or not hmac.compare_digest(previous_value, current_value):
                    raise ValueError(f"qualification milestone changed immutable {name}")
        checkpoints = await self.store.list_checkpoints(previous.job_id)
        if gate.require_reload_verified_checkpoint and not any(item.reload_verified for item in checkpoints):
            raise ValueError("qualification milestone is locked; previous checkpoint reload was not verified")
        evaluations = await self.store.list_evaluations(previous.job_id)
        passing = [item for item in evaluations if item.decision in {"pass", "release"}]
        if gate.require_passing_evaluation and not passing:
            raise ValueError("qualification milestone is locked; previous evaluation did not pass")
        for evaluation in passing:
            regression = abs(float(evaluation.metrics.get("validation_regression", 0.0)))
            if bool(evaluation.metrics.get("regression_detected")) or regression > gate.maximum_validation_regression:
                raise ValueError(
                    f"qualification milestone is locked by validation regression {regression:.6f}"
                )

    async def get_job(self, job_id: str) -> TrainingJob:
        job = await self.store.get_job(job_id)
        if not job:
            raise KeyError(f"training job not found: {job_id}")
        return job

    async def list_jobs(self, *, owner_id: str | None = None, limit: int = 100) -> list[TrainingJob]:
        return await self.store.list_jobs(owner_id=owner_id, limit=min(max(limit, 1), 500))

    async def ingest_events(self, job_id: str, batch: TrainingEventBatch) -> list[TrainingEvent]:
        job = await self.get_job(job_id)
        attempt = await self.store.get_attempt(batch.attempt_id)
        if not attempt or attempt.job_id != job_id:
            raise ValueError("event attempt does not belong to the training job")
        sequences = await self.store.allocate_event_sequences(
            job_id,
            batch.attempt_id,
            [(event.rank, event.source_sequence) for event in batch.events],
        )
        events = []
        for sequence, source in zip(sequences, batch.events, strict=True):
            if sequence is None:
                continue
            payload = source.model_dump(mode="json")
            if payload.get("message"):
                payload["message"] = redact_text(str(payload["message"]))
            payload["payload"] = redact_payload(payload.get("payload") or {})
            events.append(
                TrainingEvent(
                    **payload,
                    event_id=new_ulid(),
                    job_id=job_id,
                    attempt_id=batch.attempt_id,
                    sequence=sequence,
                )
            )
        await self.events.append(events)
        await self._apply_event_state(job, events)
        return events

    async def _apply_event_state(self, initial_job: TrainingJob, events: list[TrainingEvent]) -> None:
        job = initial_job
        for event in events:
            target: JobStatus | None = None
            if event.kind == "checkpoint" and job.status == JobStatus.RUNNING:
                target = JobStatus.CHECKPOINTING
            elif event.kind == "evaluation" and job.status in {JobStatus.RUNNING, JobStatus.CHECKPOINTING}:
                target = JobStatus.EVALUATING
            elif event.kind == "metric" and event.stage.lower() == "training" and job.status in {
                JobStatus.CHECKPOINTING,
                JobStatus.EVALUATING,
            }:
                target = JobStatus.RUNNING
            elif event.kind == "status" and event.status == "running" and job.status == JobStatus.PROVISIONING:
                target = JobStatus.RUNNING
            elif event.kind == "status" and event.status in {"complete", "completed", "succeeded"} and job.status in {
                JobStatus.RUNNING,
                JobStatus.CHECKPOINTING,
                JobStatus.EVALUATING,
            }:
                target = JobStatus.SUCCEEDED
            elif event.kind == "error":
                fatal_class = str(event.payload.get("failure_class", "runtime"))
                target = JobStatus.BLOCKED if fatal_class in {"data", "tokenizer", "nan_loss", "quality_gate", "security"} else JobStatus.FAILED
            if target is not None and target in ALLOWED_TRANSITIONS[job.status]:
                job = await self.store.transition_job(job.job_id, target, expected_version=job.version, detail=event.message)
                await self.store.update_attempt_status(event.attempt_id, target)
                await self.record_audit(
                    actor_id=f"worker:{event.attempt_id}",
                    job_id=job.job_id,
                    action="training.job.transition",
                    outcome="failed" if target in {JobStatus.FAILED, JobStatus.BLOCKED} else "succeeded",
                    metadata={"target": target.value, "event_id": event.event_id, "sequence": event.sequence},
                )

    async def stream_events(
        self,
        job_id: str,
        *,
        after_sequence: int = 0,
        heartbeat_seconds: float = 5.0,
    ) -> AsyncIterator[TrainingEvent | None]:
        await self.get_job(job_id)
        cursor = after_sequence
        while True:
            rows = await self.events.read(job_id, after_sequence=cursor, limit=100, block_ms=int(heartbeat_seconds * 1000))
            if not rows:
                yield None
            for event in rows:
                cursor = max(cursor, event.sequence)
                yield event
            job = await self.get_job(job_id)
            if job.status.value in TERMINAL_STATES and not rows:
                return

    async def create_attempt(self, job: TrainingJob, *, checkpoint_uri: str | None = None) -> TrainingAttempt:
        attempts = await self.store.list_attempts(job.job_id)
        return await self.store.create_attempt(
            TrainingAttempt(
                attempt_id=str(uuid.uuid4()),
                job_id=job.job_id,
                attempt_number=len(attempts) + 1,
                scheduler=job.spec.scheduler,
                checkpoint_uri=checkpoint_uri,
                status=JobStatus.PROVISIONING,
            )
        )

    async def claim_notebook_job(self, job_id: str) -> tuple[TrainingJob, TrainingAttempt]:
        job = await self.get_job(job_id)
        if job.spec.scheduler != "notebook":
            raise ValueError("only notebook validation profiles can be claimed by a client worker")
        if job.status != JobStatus.QUEUED:
            raise ValueError(f"notebook job cannot be claimed from status={job.status.value}")
        provisioning = await self.store.transition_job(job_id, JobStatus.PROVISIONING, expected_version=job.version)
        attempt = await self.create_attempt(provisioning)
        running = await self.store.transition_job(job_id, JobStatus.RUNNING, expected_version=provisioning.version)
        attempt = await self.store.update_attempt_status(attempt.attempt_id, JobStatus.RUNNING)
        return running, attempt

    async def cancel_job(
        self,
        job_id: str,
        scheduler: SchedulerAdapter | None = None,
        *,
        actor_id: str = "system",
    ) -> TrainingJob:
        job = await self.get_job(job_id)
        if job.status.value in TERMINAL_STATES:
            return job
        if scheduler and job.scheduler_binding:
            await scheduler.cancel(SchedulerBinding.model_validate(job.scheduler_binding))
        cancelled = await self.store.transition_job(job_id, JobStatus.CANCELLED, expected_version=job.version)
        if job.current_attempt_id:
            await self.store.update_attempt_status(job.current_attempt_id, JobStatus.CANCELLED)
        await self.record_audit(
            actor_id=actor_id,
            job_id=job_id,
            action="training.job.cancel",
            outcome="succeeded",
        )
        return cancelled

    async def resume_job(self, job_id: str, *, actor_id: str = "system") -> TrainingJob:
        job = await self.get_job(job_id)
        if job.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise ValueError("only failed infrastructure attempts or cancelled jobs can resume")
        artifacts = await self.store.list_artifacts(job_id)
        checkpoints = [item for item in artifacts if item.kind == "checkpoint" and item.promoted]
        if not checkpoints:
            raise ValueError("resume requires a promoted, checksum-verified checkpoint")
        queued = await self.store.transition_job(job_id, JobStatus.QUEUED, expected_version=job.version)
        await self.record_audit(
            actor_id=actor_id,
            job_id=job_id,
            action="training.job.resume",
            outcome="accepted",
            metadata={"checkpoint_uri": checkpoints[-1].uri},
        )
        return queued

    async def list_audit_events(self, job_id: str, *, limit: int = 100) -> list[TrainingAuditEvent]:
        await self.get_job(job_id)
        return await self.store.list_audit_events(job_id, limit=min(max(limit, 1), 500))

    async def register_artifact(self, artifact: TrainingArtifact) -> TrainingArtifact:
        await self.get_job(artifact.job_id)
        return await self.store.add_artifact(artifact)

    async def presign_artifact_upload(self, job_id: str, request: ArtifactUploadRequest) -> dict[str, Any]:
        job = await self.get_job(job_id)
        if request.attempt_id:
            attempt = await self.store.get_attempt(request.attempt_id)
            if not attempt or attempt.job_id != job_id:
                raise ValueError("artifact attempt does not belong to the training job")
        if not isinstance(self.object_store, S3ObjectStore):
            raise RuntimeError("presigned artifact uploads require production S3/MinIO object storage")
        attempt_segment = request.attempt_id or job.current_attempt_id or "job"
        object_name = request.relative_path or request.filename
        key = f"jobs/{job_id}/attempts/{attempt_segment}/{request.kind}/{object_name}"
        return await asyncio.to_thread(
            self.object_store.presign_put,
            key=key,
            sha256=request.sha256,
            content_type=request.content_type,
            expires_seconds=900,
        )

    async def verify_and_register_artifact(
        self,
        job_id: str,
        request: ArtifactUploadRequest,
        *,
        uri: str,
    ) -> TrainingArtifact:
        await self.get_job(job_id)
        if not isinstance(self.object_store, S3ObjectStore):
            raise RuntimeError("artifact verification requires production S3/MinIO object storage")
        parsed_prefix = f"s3://{self.object_store.bucket}/"
        if not uri.startswith(parsed_prefix):
            raise ValueError("artifact URI does not belong to the configured object store")
        object_key = uri[len(parsed_prefix) :]
        configured_prefix = self.object_store.prefix.strip("/")
        relative_key = object_key[len(configured_prefix) + 1 :] if configured_prefix and object_key.startswith(configured_prefix + "/") else object_key
        stored = await asyncio.to_thread(self.object_store.head, relative_key)
        if stored.size_bytes != request.size_bytes:
            raise ValueError("uploaded artifact size does not match the declared size")
        if not stored.sha256 or not hmac.compare_digest(stored.sha256, request.sha256):
            raise ValueError("uploaded artifact SHA-256 metadata is missing or mismatched")
        return await self.store.add_artifact(
            TrainingArtifact(
                artifact_id=str(uuid.uuid4()),
                job_id=job_id,
                attempt_id=request.attempt_id,
                kind=request.kind,
                uri=stored.uri,
                sha256=request.sha256,
                size_bytes=request.size_bytes,
            )
        )

    async def list_artifacts(self, job_id: str) -> list[TrainingArtifact]:
        await self.get_job(job_id)
        return await self.store.list_artifacts(job_id)

    async def commit_checkpoint(self, job_id: str, request: CheckpointCommitRequest) -> CheckpointVersion:
        job = await self.get_job(job_id)
        attempt = await self.store.get_attempt(request.attempt_id)
        if not attempt or attempt.job_id != job_id:
            raise ValueError("checkpoint attempt does not belong to the training job")
        if job.spec.dataset_manifest_sha256 and not hmac.compare_digest(
            job.spec.dataset_manifest_sha256, request.dataset_sha256
        ):
            raise ValueError("checkpoint dataset hash does not match the immutable job spec")
        if job.spec.tokenizer_sha256 and not hmac.compare_digest(job.spec.tokenizer_sha256, request.tokenizer_sha256):
            raise ValueError("checkpoint tokenizer hash does not match the immutable job spec")
        artifacts = await self.store.list_artifacts(job_id)
        verified_manifest = next(
            (
                item
                for item in artifacts
                if item.kind == "checkpoint"
                and item.attempt_id == request.attempt_id
                and item.uri == request.manifest_uri
                and hmac.compare_digest(item.sha256, request.manifest_sha256)
            ),
            None,
        )
        if verified_manifest is None:
            raise ValueError("checkpoint manifest must be uploaded and checksum-verified before commit")
        if isinstance(self.object_store, S3ObjectStore):
            await self._verify_checkpoint_objects(request, artifacts)
        checkpoint = await self.store.add_checkpoint(
            CheckpointVersion(
                checkpoint_id=str(uuid.uuid4()),
                job_id=job_id,
                attempt_id=request.attempt_id,
                step=request.step,
                manifest_uri=request.manifest_uri,
                manifest_sha256=request.manifest_sha256,
                dataset_sha256=request.dataset_sha256,
                tokenizer_sha256=request.tokenizer_sha256,
                topology=redact_payload(request.topology),
                metrics=redact_payload(request.metrics),
                reload_verified=request.reload_verified,
            )
        )
        await self.record_audit(
            actor_id=f"worker:{request.attempt_id}",
            job_id=job_id,
            action="training.checkpoint.commit",
            outcome="succeeded",
            metadata={"checkpoint_id": checkpoint.checkpoint_id, "step": checkpoint.step},
        )
        return checkpoint

    async def _verify_checkpoint_objects(
        self,
        request: CheckpointCommitRequest,
        artifacts: list[TrainingArtifact],
    ) -> None:
        assert isinstance(self.object_store, S3ObjectStore)
        bucket_prefix = f"s3://{self.object_store.bucket}/"
        if not request.manifest_uri.startswith(bucket_prefix):
            raise ValueError("checkpoint manifest is outside the configured object-store bucket")
        object_key = request.manifest_uri[len(bucket_prefix) :]
        configured_prefix = self.object_store.prefix.strip("/")
        if configured_prefix:
            expected_prefix = configured_prefix + "/"
            if not object_key.startswith(expected_prefix):
                raise ValueError("checkpoint manifest is outside the configured object-store prefix")
            relative_key = object_key[len(expected_prefix) :]
        else:
            relative_key = object_key
        with tempfile.TemporaryDirectory(prefix="craftly-checkpoint-manifest-") as temp_dir:
            local_manifest = Path(temp_dir) / "checkpoint_manifest.json"
            downloaded = await asyncio.to_thread(self.object_store.get_file, relative_key, local_manifest)
            if not hmac.compare_digest(downloaded.sha256, request.manifest_sha256):
                raise ValueError("downloaded checkpoint manifest checksum mismatch")
            try:
                payload = json.loads(local_manifest.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("checkpoint manifest is not valid UTF-8 JSON") from exc
        files = payload.get("files")
        if not isinstance(files, list) or not files or len(files) > 100_000:
            raise ValueError("checkpoint manifest must contain between 1 and 100000 file entries")
        artifact_index = {item.uri: item for item in artifacts if item.kind == "checkpoint"}
        base_uri = request.manifest_uri.rsplit("/", 1)[0]
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("checkpoint manifest file entry must be an object")
            relative = str(entry.get("path") or "")
            digest = str(entry.get("sha256") or "")
            size = entry.get("size_bytes")
            relative_path = Path(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ValueError("checkpoint manifest contains an unsafe or invalid file entry")
            uri = f"{base_uri}/{relative_path.as_posix()}"
            artifact = artifact_index.get(uri)
            if artifact is None:
                raise ValueError(f"checkpoint object was not uploaded and verified: {relative}")
            if artifact.size_bytes != size or not hmac.compare_digest(artifact.sha256, digest):
                raise ValueError(f"checkpoint object checksum or size mismatch: {relative}")

    async def commit_evaluation(self, job_id: str, request: EvaluationCommitRequest) -> EvaluationRun:
        await self.get_job(job_id)
        if request.checkpoint_id:
            checkpoint = await self.store.get_checkpoint(request.checkpoint_id)
            if not checkpoint or checkpoint.job_id != job_id:
                raise ValueError("evaluation checkpoint does not belong to the training job")
        artifacts = await self.store.list_artifacts(job_id)
        verified_report = next(
            (
                item
                for item in artifacts
                if item.kind == "evaluation"
                and item.uri == request.report_uri
                and hmac.compare_digest(item.sha256, request.report_sha256)
            ),
            None,
        )
        if verified_report is None:
            raise ValueError("evaluation report must be uploaded and checksum-verified before commit")
        evaluation = await self.store.add_evaluation(
            EvaluationRun(
                evaluation_id=str(uuid.uuid4()),
                job_id=job_id,
                checkpoint_id=request.checkpoint_id,
                status=request.status,
                report_uri=request.report_uri,
                report_sha256=request.report_sha256,
                decision=request.decision,
                metrics=redact_payload(request.metrics),
            )
        )
        await self.record_audit(
            actor_id="evaluation-worker",
            job_id=job_id,
            action="training.evaluation.commit",
            outcome="succeeded" if request.decision in {"pass", "release"} else "failed",
            metadata={"evaluation_id": evaluation.evaluation_id, "decision": evaluation.decision},
        )
        return evaluation

    async def promote_checkpoint(self, job_id: str, checkpoint_id: str, *, actor_id: str) -> CheckpointVersion:
        job = await self.get_job(job_id)
        checkpoint = await self.store.get_checkpoint(checkpoint_id)
        if not checkpoint or checkpoint.job_id != job_id:
            raise KeyError(f"checkpoint not found for job: {checkpoint_id}")
        if job.spec.validation_only:
            raise ValueError("validation-only checkpoints cannot be promoted as release checkpoints")
        policy = job.spec.promotion
        if checkpoint.step < policy.minimum_step:
            raise ValueError(f"checkpoint step {checkpoint.step} is below promotion minimum {policy.minimum_step}")
        if policy.require_reload_verification and not checkpoint.reload_verified:
            raise ValueError("checkpoint promotion requires a successful reload smoke verification")
        evaluations = await self.store.list_evaluations(job_id)
        passing_evaluations = [
            item
            for item in evaluations
            if item.checkpoint_id == checkpoint_id and item.decision in {"pass", "release"}
        ]
        if policy.require_passing_evaluation and len(passing_evaluations) < policy.minimum_completed_evaluations:
            raise ValueError("checkpoint promotion requires a passing committed evaluation")
        if policy.require_no_regression and any(
            bool(item.metrics.get("regression_detected")) for item in passing_evaluations
        ):
            raise ValueError("checkpoint promotion blocked by evaluation regression")
        passing = passing_evaluations[-1] if passing_evaluations else None
        promoted = await self.store.promote_checkpoint(checkpoint_id)
        latest = await asyncio.to_thread(
            self.object_store.put_json,
            {
                "schema_version": 1,
                "job_id": job_id,
                "checkpoint_id": checkpoint_id,
                "step": checkpoint.step,
                "manifest_uri": checkpoint.manifest_uri,
                "manifest_sha256": checkpoint.manifest_sha256,
                "evaluation_id": passing.evaluation_id if passing else None,
                "promoted_at": utc_now().isoformat(),
            },
            key=f"jobs/{job_id}/checkpoints/latest.json",
        )
        await self.record_audit(
            actor_id=actor_id,
            job_id=job_id,
            action="training.checkpoint.promote",
            outcome="succeeded",
            metadata={"checkpoint_id": checkpoint_id, "latest_pointer": latest.uri},
        )
        return promoted

    async def list_checkpoints(self, job_id: str) -> list[CheckpointVersion]:
        await self.get_job(job_id)
        return await self.store.list_checkpoints(job_id)

    async def list_evaluations(self, job_id: str) -> list[EvaluationRun]:
        await self.get_job(job_id)
        return await self.store.list_evaluations(job_id)

    async def create_service_account(self, *, name: str, scopes: list[str]) -> ServiceAccountCredential:
        if not SAFE_IDENTIFIER.fullmatch(name):
            raise ValueError("service account name contains unsafe characters")
        allowed = {
            "training:jobs:create",
            "training:jobs:read",
            "training:jobs:cancel",
            "training:events:write",
            "training:artifacts:write",
            "training:artifacts:read",
            "training:admin",
        }
        unknown = set(scopes) - allowed
        if unknown:
            raise ValueError("unknown service-account scopes: " + ", ".join(sorted(unknown)))
        account_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)
        token = f"craftly_sa_{account_id}_{secret}"
        account = ServiceAccount(
            service_account_id=account_id,
            name=name,
            scopes=sorted(set(scopes)),
            token_prefix=token[:24],
        )
        await self.store.create_service_account(account, sha256_text(token))
        return ServiceAccountCredential(account=account, bootstrap_token=token)

    async def authenticate_bootstrap(self, token: str) -> ServiceAccount | None:
        env_token = os.environ.get("CRAFTLY_WORKSPACE_BOOTSTRAP_TOKEN")
        if env_token and hmac.compare_digest(token, env_token):
            account = ServiceAccount(
                service_account_id="00000000-0000-0000-0000-000000000001",
                name="environment-bootstrap",
                scopes=[
                    "training:jobs:create",
                    "training:jobs:read",
                    "training:jobs:cancel",
                    "training:events:write",
                    "training:artifacts:read",
                ],
                token_prefix="environment",
            )
            await self.store.ensure_service_account(account, sha256_text(token))
            return account
        account = await self.store.authenticate_service_account(token)
        if env_token and account and account.service_account_id == "00000000-0000-0000-0000-000000000001":
            return None
        return account

    async def create_refresh_session(self, account: ServiceAccount, *, ttl_seconds: int = 43_200) -> RefreshSession:
        refresh_token = secrets.token_urlsafe(48)
        expires = datetime.fromtimestamp(time.time() + ttl_seconds, tz=timezone.utc)
        session_id = await self.store.create_refresh_session(account.service_account_id, sha256_text(refresh_token), expires)
        return RefreshSession(session_id=session_id, service_account_id=account.service_account_id, refresh_token=refresh_token, expires_at=expires)

    async def authenticate_refresh(self, session_id: str, refresh_token: str) -> ServiceAccount | None:
        return await self.store.consume_refresh_session(session_id, refresh_token)

    async def revoke_refresh(self, session_id: str, refresh_token: str) -> bool:
        return await self.store.revoke_refresh_session(session_id, refresh_token)

    async def close(self) -> None:
        await self.events.close()
        await self.store.close()


class TrainingController:
    """Idempotent reconciler between queued jobs and trusted schedulers."""

    def __init__(self, service: TrainingWorkspaceService, schedulers: dict[str, SchedulerAdapter] | None = None) -> None:
        self.service = service
        self.schedulers = schedulers or {
            "notebook": NotebookValidationAdapter(),
            "kubernetes": KubernetesSchedulerAdapter(),
            "kubernetes_pytorch": KubernetesPyTorchAdapter(),
            "slurm": SlurmSchedulerAdapter(),
        }

    async def reconcile_job(self, job: TrainingJob) -> TrainingJob:
        scheduler = self.schedulers.get(job.spec.scheduler)
        if not scheduler:
            return await self.service.store.transition_job(
                job.job_id,
                JobStatus.BLOCKED,
                expected_version=job.version,
                detail=f"scheduler adapter is not registered: {job.spec.scheduler}",
            )
        if job.status == JobStatus.QUEUED:
            retry_not_before = float(job.scheduler_binding.get("retry_not_before_unix", 0.0))
            if retry_not_before > time.time():
                return job
            try:
                await scheduler.validate(job.spec)
                provisioning = await self.service.store.transition_job(job.job_id, JobStatus.PROVISIONING, expected_version=job.version)
                attempt = await self.service.create_attempt(provisioning)
                binding = await scheduler.submit(provisioning, attempt)
                submitted = await self.service.store.update_binding(
                    job.job_id,
                    expected_version=provisioning.version,
                    binding=binding.model_dump(mode="json"),
                )
                await self.service.record_audit(
                    actor_id="training-controller",
                    job_id=job.job_id,
                    action="training.scheduler.submit",
                    outcome="accepted",
                    metadata={"scheduler": scheduler.name, "attempt_id": attempt.attempt_id, "external_id": binding.external_id},
                )
                return submitted
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                current = await self.service.get_job(job.job_id)
                if JobStatus.BLOCKED in ALLOWED_TRANSITIONS[current.status]:
                    blocked = await self.service.store.transition_job(
                        job.job_id,
                        JobStatus.BLOCKED,
                        expected_version=current.version,
                        detail=str(exc),
                    )
                    if blocked.current_attempt_id:
                        await self.service.store.update_attempt_status(blocked.current_attempt_id, JobStatus.BLOCKED)
                    await self.service.record_audit(
                        actor_id="training-controller",
                        job_id=blocked.job_id,
                        action="training.scheduler.validate",
                        outcome="failed",
                        metadata={"reason": str(exc)},
                    )
                    return blocked
                raise
        if job.status in {JobStatus.PROVISIONING, JobStatus.RUNNING, JobStatus.CHECKPOINTING, JobStatus.EVALUATING} and job.scheduler_binding:
            binding = SchedulerBinding.model_validate(job.scheduler_binding)
            rotated = await scheduler.rotate_credentials(job, binding)
            if rotated != binding:
                job = await self.service.store.update_binding(
                    job.job_id,
                    expected_version=job.version,
                    binding=rotated.model_dump(mode="json"),
                )
                binding = rotated
            runtime_status = await scheduler.status(binding)
            mapping = {
                "running": JobStatus.RUNNING,
                "succeeded": JobStatus.SUCCEEDED,
                "failed": JobStatus.FAILED,
                "cancelled": JobStatus.CANCELLED,
            }
            target = mapping.get(runtime_status)
            if target and target != job.status and target in ALLOWED_TRANSITIONS[job.status]:
                transitioned = await self.service.store.transition_job(job.job_id, target, expected_version=job.version)
                if transitioned.current_attempt_id:
                    await self.service.store.update_attempt_status(transitioned.current_attempt_id, target)
                await self.service.record_audit(
                    actor_id="training-controller",
                    job_id=transitioned.job_id,
                    action="training.scheduler.reconcile",
                    outcome="failed" if target == JobStatus.FAILED else "succeeded",
                    metadata={"runtime_status": runtime_status, "target": target.value},
                )
                if target == JobStatus.FAILED:
                    attempts = await self.service.store.list_attempts(job.job_id)
                    failure_class = str(binding.metadata.get("failure_class") or "infrastructure")
                    if (
                        failure_class in transitioned.spec.retry.retryable_failures
                        and len(attempts) < transitioned.spec.retry.maximum_attempts
                    ):
                        delay = min(
                            transitioned.spec.retry.maximum_backoff_seconds,
                            transitioned.spec.retry.initial_backoff_seconds * (2 ** max(0, len(attempts) - 1)),
                        )
                        retry_binding = dict(transitioned.scheduler_binding)
                        retry_binding.update(
                            {
                                "retry_not_before_unix": time.time() + delay,
                                "retry_failure_class": failure_class,
                                "previous_external_id": binding.external_id,
                            }
                        )
                        transitioned = await self.service.store.update_binding(
                            transitioned.job_id,
                            expected_version=transitioned.version,
                            binding=retry_binding,
                        )
                        transitioned = await self.service.store.transition_job(
                            transitioned.job_id,
                            JobStatus.QUEUED,
                            expected_version=transitioned.version,
                            detail=f"automatic retry after {failure_class}; backoff={delay}s",
                        )
                        await self.service.record_audit(
                            actor_id="training-controller",
                            job_id=transitioned.job_id,
                            action="training.scheduler.retry",
                            outcome="accepted",
                            metadata={"attempts": len(attempts), "failure_class": failure_class, "backoff_seconds": delay},
                        )
                return transitioned
        return job

    async def reconcile_once(self, *, limit: int = 100) -> list[TrainingJob]:
        jobs = await self.service.list_jobs(limit=limit)
        active = [
            item
            for item in jobs
            if item.status.value not in TERMINAL_STATES and item.spec.scheduler != "notebook"
        ]
        results = []
        for job in active:
            try:
                results.append(await self.reconcile_job(job))
            except Exception as exc:
                current = await self.service.get_job(job.job_id)
                if JobStatus.FAILED in ALLOWED_TRANSITIONS[current.status]:
                    failed = await self.service.store.transition_job(
                        current.job_id,
                        JobStatus.FAILED,
                        expected_version=current.version,
                        detail=f"controller reconciliation failed: {exc}",
                    )
                    if failed.current_attempt_id:
                        await self.service.store.update_attempt_status(failed.current_attempt_id, JobStatus.FAILED)
                    results.append(failed)
                else:
                    raise
        return results

    async def run_forever(self, *, poll_seconds: float = 5.0) -> None:
        while True:
            await self.reconcile_once()
            await asyncio.sleep(poll_seconds)


class TrainingEventArchiver:
    """Copies ordered Redis events into immutable compressed object chunks."""

    def __init__(self, service: TrainingWorkspaceService, *, chunk_events: int = 5000) -> None:
        self.service = service
        self.chunk_events = min(max(chunk_events, 100), 5000)

    async def archive_job(self, job: TrainingJob) -> list[TrainingArtifact]:
        cursor = job.archived_event_sequence
        archived: list[TrainingArtifact] = []
        while True:
            events = await self.service.events.read(
                job.job_id,
                after_sequence=cursor,
                limit=self.chunk_events,
                block_ms=0,
            )
            if not events:
                break
            start = events[0].sequence
            end = events[-1].sequence
            with tempfile.TemporaryDirectory(prefix="craftly-events-") as temp_dir:
                source = Path(temp_dir) / f"events-{start:012d}-{end:012d}.jsonl.gz"
                with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                    for event in events:
                        handle.write(canonical_json(event.model_dump(mode="json")) + "\n")
                stored = await asyncio.to_thread(
                    self.service.object_store.put_file,
                    source,
                    key=f"jobs/{job.job_id}/attempts/{events[0].attempt_id}/events/{source.name}",
                )
            artifact = await self.service.store.add_artifact(
                TrainingArtifact(
                    artifact_id=str(uuid.uuid4()),
                    job_id=job.job_id,
                    attempt_id=events[0].attempt_id,
                    kind="log",
                    uri=stored.uri,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                    metadata={"first_sequence": start, "last_sequence": end, "event_count": len(events), "compression": "gzip"},
                )
            )
            archived.append(artifact)
            cursor = end
            job = await self.service.store.update_archive_cursor(job.job_id, cursor)
            if len(events) < self.chunk_events:
                break
        return archived

    async def archive_once(self, *, limit: int = 100) -> list[TrainingArtifact]:
        jobs = await self.service.list_jobs(limit=limit)
        archived = []
        for job in jobs:
            if job.event_sequence > job.archived_event_sequence:
                archived.extend(await self.archive_job(job))
        return archived


def workspace_readiness(service: TrainingWorkspaceService) -> dict[str, Any]:
    production_store = isinstance(service.store, PostgresTrainingStore)
    production_events = isinstance(service.events, RedisEventBus)
    object_store_name = type(service.object_store).__name__
    production_objects = object_store_name == "S3ObjectStore"
    missing = []
    if not production_store:
        missing.append("PostgresTrainingStore")
    if not production_events:
        missing.append("RedisEventBus")
    if not production_objects:
        missing.append("S3ObjectStore")
    return {
        "status": "production_ready_requires_external_service" if not missing else "blocked_missing_dependency",
        "production_mode": service.production_mode,
        "profile_count": len(service.profiles.profiles),
        "store": type(service.store).__name__,
        "event_bus": type(service.events).__name__,
        "object_store": object_store_name,
        "missing_dependencies": missing,
        "cluster_status": "built_not_cluster_proven",
    }


def parse_workspace_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Craftly Training Workspace controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    controller = subparsers.add_parser("controller")
    controller.add_argument("--poll-seconds", type=float, default=5.0)
    controller.add_argument("--once", action="store_true")
    subparsers.add_parser("profiles")
    subparsers.add_parser("campaigns")
    subparsers.add_parser("readiness")
    return parser.parse_args()


async def workspace_main() -> None:
    args = parse_workspace_args()
    service = TrainingWorkspaceService.from_environment()
    try:
        if args.command == "profiles":
            print(json.dumps({"profiles": service.profile_report()}, indent=2, sort_keys=True), flush=True)
            return
        if args.command == "campaigns":
            print(
                json.dumps(
                    {
                        "campaigns": [
                            {
                                **campaign.model_dump(mode="json"),
                                "milestones": [item.model_dump(mode="json") for item in campaign.milestones],
                            }
                            for campaign in service.campaigns.campaigns
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if args.command == "readiness":
            print(json.dumps(workspace_readiness(service), indent=2, sort_keys=True), flush=True)
            return
        controller = TrainingController(service)
        archiver = TrainingEventArchiver(service)
        if args.once:
            jobs = await controller.reconcile_once()
            archived = await archiver.archive_once()
            print(
                json.dumps(
                    {
                        "reconciled": [job.model_dump(mode="json") for job in jobs],
                        "archived_artifacts": [item.model_dump(mode="json") for item in archived],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        while True:
            await controller.reconcile_once()
            await archiver.archive_once()
            await asyncio.sleep(max(0.5, args.poll_seconds))
    finally:
        await service.close()


def main() -> None:
    asyncio.run(workspace_main())


if __name__ == "__main__":
    main()
