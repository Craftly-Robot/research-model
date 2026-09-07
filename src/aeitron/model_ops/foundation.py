"""Scratch-model foundation contracts for Aeitron.

This module is intentionally not a fine-tuning wrapper. It defines the stable
contracts Aeitron will use for tokenizer assets, decoder-only architecture
profiles, pretraining plans, and checkpoint manifests before GPU training code
is attached.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from src.aeitron.shared.schemas import StrictModel
from src.aeitron.shared.integrity import canonical_json_bytes, sha256_file


TRAINING_STEP_SEMANTICS = "optimizer_update_v2"


class TrainingBatchContract(StrictModel):
    """Canonical optimizer-step, global-batch, and token-accounting contract."""

    step_semantics: Literal["optimizer_update_v2"] = TRAINING_STEP_SEMANTICS
    optimizer_steps: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    data_parallel_size: int = Field(ge=1)

    @property
    def global_batch_sequences(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps * self.data_parallel_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.sequence_length * self.global_batch_sequences

    @property
    def target_tokens(self) -> int:
        return self.optimizer_steps * self.tokens_per_optimizer_step

    @property
    def completed_micro_batches_per_rank(self) -> int:
        return self.optimizer_steps * self.gradient_accumulation_steps

    def report(self) -> dict[str, int | str]:
        return {
            "step_semantics": self.step_semantics,
            "optimizer_steps": self.optimizer_steps,
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.data_parallel_size,
            "global_batch_sequences": self.global_batch_sequences,
            "tokens_per_optimizer_step": self.tokens_per_optimizer_step,
            "target_tokens": self.target_tokens,
            "completed_micro_batches_per_rank": self.completed_micro_batches_per_rank,
        }


class TokenizerContract(StrictModel):
    tokenizer_type: Literal["bpe", "unigram", "sentencepiece"] = "bpe"
    vocab_size: int = Field(default=128_000, ge=8_000, le=512_000)
    tokenizer_path: str | None = None
    special_tokens: list[str] = Field(
        default_factory=lambda: [
            "<|thought_start|>",
            "<|thought_end|>",
            "<|patch_start|>",
            "<|patch_end|>",
            "<|compile_error|>",
            "<|tool_call|>",
            "<|tool_result|>",
        ]
    )

    def required_asset_paths(self) -> list[Path]:
        return [Path(self.tokenizer_path)] if self.tokenizer_path else []


class ScratchDecoderConfig(StrictModel):
    """Canonical architecture contract shared by every Aeitron model runtime."""

    architecture_version: int = Field(default=2, ge=1)
    name: str = "aeitron-tiny-smoke"
    vocab_size: int = Field(default=4096, ge=256)
    max_sequence_length: int = Field(default=512, ge=16)
    effective_context_length: int | None = Field(default=None, ge=16)
    hidden_size: int = Field(default=256, ge=64)
    num_layers: int = Field(default=4, ge=1)
    num_attention_heads: int = Field(default=4, ge=1)
    num_key_value_heads: int = Field(default=4, ge=1)
    intermediate_size: int = Field(default=1024, ge=128)
    attention_architecture: Literal["gqa", "mla"] = "gqa"
    q_lora_rank: int | None = Field(default=None, ge=16)
    kv_lora_rank: int | None = Field(default=None, ge=16)
    qk_nope_head_dim: int | None = Field(default=None, ge=8)
    qk_rope_head_dim: int | None = Field(default=None, ge=8)
    v_head_dim: int | None = Field(default=None, ge=8)
    feed_forward_architecture: Literal["dense", "moe"] = "dense"
    num_dense_layers: int | None = Field(default=None, ge=0)
    num_routed_experts: int = Field(default=0, ge=0)
    num_shared_experts: int = Field(default=0, ge=0)
    experts_per_token: int = Field(default=0, ge=0)
    moe_intermediate_size: int | None = Field(default=None, ge=64)
    router_bias_update_rate: float = Field(default=0.0, ge=0.0, le=0.1)
    router_load_limit: float = Field(default=1.20, ge=1.0)
    router_drop_tokens: Literal[False] = False
    mtp_num_layers: int = Field(default=0, ge=0, le=1)
    mtp_loss_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    rope_theta: float = Field(default=1_000_000.0, gt=0)
    rope_scaling_type: Literal["none", "linear", "yarn", "longrope"] = "none"
    rope_scaling_factor: float = Field(default=1.0, ge=1.0)
    norm_eps: float = Field(default=1e-6, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    tie_word_embeddings: bool = True
    attention_impl: Literal["auto", "sdpa", "eager"] = "auto"
    attention_window: int | None = Field(default=None, ge=16)
    initializer_range: float = Field(default=0.02, gt=0.0, le=1.0)
    use_cache: bool = True
    gradient_checkpointing: bool = False
    logits_soft_cap: float | None = Field(default=None, gt=0.0)
    runtime_backend: Literal["native_reference", "megatron_core"] = "native_reference"
    target_total_parameters: int | None = Field(default=None, ge=1)
    target_active_parameters: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_shape(self) -> "ScratchDecoderConfig":
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.effective_context_length is not None and self.effective_context_length < self.max_sequence_length:
            raise ValueError("effective_context_length cannot be smaller than max_sequence_length")
        if self.attention_architecture == "gqa":
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even for rotary embeddings")
        else:
            required = {
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("MLA requires " + ", ".join(missing))
            if int(self.qk_rope_head_dim or 0) % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even")
        dense_layers = self.num_layers if self.num_dense_layers is None else self.num_dense_layers
        if dense_layers > self.num_layers:
            raise ValueError("num_dense_layers cannot exceed num_layers")
        if self.feed_forward_architecture == "moe":
            if dense_layers >= self.num_layers:
                raise ValueError("MoE architecture requires at least one routed layer")
            if self.num_routed_experts < 2 or self.num_shared_experts < 1:
                raise ValueError("MoE architecture requires routed and shared experts")
            if not 1 <= self.experts_per_token <= self.num_routed_experts:
                raise ValueError("experts_per_token must be between one and num_routed_experts")
            if self.moe_intermediate_size is None:
                raise ValueError("MoE architecture requires moe_intermediate_size")
        else:
            if dense_layers != self.num_layers:
                raise ValueError("dense architecture must use dense feed-forward blocks in every layer")
            if any((self.num_routed_experts, self.num_shared_experts, self.experts_per_token)):
                raise ValueError("dense architecture cannot configure MoE experts")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def dense_layer_count(self) -> int:
        return self.num_layers if self.num_dense_layers is None else self.num_dense_layers

    @property
    def moe_layer_count(self) -> int:
        return self.num_layers - self.dense_layer_count

    def _attention_parameters_per_layer(self) -> int:
        if self.attention_architecture == "gqa":
            q = self.hidden_size * self.hidden_size
            kv = 2 * self.hidden_size * self.head_dim * self.num_key_value_heads
            out = self.hidden_size * self.hidden_size
            return q + kv + out
        q_rank = int(self.q_lora_rank or 0)
        kv_rank = int(self.kv_lora_rank or 0)
        nope = int(self.qk_nope_head_dim or 0)
        rope = int(self.qk_rope_head_dim or 0)
        value = int(self.v_head_dim or 0)
        return (
            self.hidden_size * q_rank
            + q_rank
            + q_rank * self.num_attention_heads * (nope + rope)
            + self.hidden_size * (kv_rank + rope)
            + kv_rank
            + kv_rank * self.num_attention_heads * (nope + value)
            + self.num_attention_heads * value * self.hidden_size
        )

    def _mtp_parameters_per_layer(self) -> int:
        """Parameters in one training-only multi-token prediction block.

        The block fuses the main decoder hidden state with the embedding of the
        next observed token, then applies one dense transformer block and an
        output norm. Token embeddings and the language-model head are shared
        with the main decoder and are therefore not counted a second time.
        """
        if not self.mtp_num_layers:
            return 0
        fusion_projection = 2 * self.hidden_size * self.hidden_size
        dense_transformer = (
            self._attention_parameters_per_layer()
            + 3 * self.hidden_size * self.intermediate_size
        )
        # Hidden input norm, next-token embedding norm, two block norms, and
        # the prediction output norm.
        normalization = 5 * self.hidden_size
        return fusion_projection + dense_transformer + normalization

    def parameter_report(self) -> dict[str, Any]:
        embedding = self.vocab_size * self.hidden_size
        attention = self.num_layers * self._attention_parameters_per_layer()
        norms = self.num_layers * 2 * self.hidden_size + self.hidden_size
        dense_mlp = self.dense_layer_count * 3 * self.hidden_size * self.intermediate_size
        expert_size = 0
        routed_experts = shared_experts = routers = active_experts = 0
        if self.moe_layer_count:
            expert_size = 3 * self.hidden_size * int(self.moe_intermediate_size or 0)
            routed_experts = self.moe_layer_count * self.num_routed_experts * expert_size
            shared_experts = self.moe_layer_count * self.num_shared_experts * expert_size
            routers = self.moe_layer_count * self.hidden_size * self.num_routed_experts
            active_experts = self.moe_layer_count * (
                self.experts_per_token + self.num_shared_experts
            ) * expert_size
        mtp = self.mtp_num_layers * self._mtp_parameters_per_layer()
        lm_head = 0 if self.tie_word_embeddings else embedding
        total = embedding + attention + norms + dense_mlp + routed_experts + shared_experts + routers + mtp + lm_head
        active = embedding + attention + norms + dense_mlp + active_experts + routers + mtp + lm_head
        total_delta = None if self.target_total_parameters is None else (total - self.target_total_parameters) / self.target_total_parameters
        active_delta = None if self.target_active_parameters is None else (active - self.target_active_parameters) / self.target_active_parameters
        if self.attention_architecture == "mla":
            compressed_cache_elements = int(self.kv_lora_rank or 0) + int(self.qk_rope_head_dim or 0)
            expanded_cache_elements = self.num_attention_heads * (
                int(self.qk_nope_head_dim or 0)
                + int(self.qk_rope_head_dim or 0)
                + int(self.v_head_dim or 0)
            )
        else:
            compressed_cache_elements = 2 * self.num_key_value_heads * self.head_dim
            expanded_cache_elements = 2 * self.num_attention_heads * self.head_dim
        cache_compression_ratio = compressed_cache_elements / max(expanded_cache_elements, 1)
        return {
            "embedding": embedding,
            "attention": attention,
            "dense_mlp": dense_mlp,
            "routed_experts": routed_experts,
            "shared_experts": shared_experts,
            "routers": routers,
            "mtp": mtp,
            "norms": norms,
            "lm_head": lm_head,
            "expert_size": expert_size,
            "total": total,
            "active": active,
            "total_billions": round(total / 1_000_000_000, 6),
            "active_billions": round(active / 1_000_000_000, 6),
            "total_target_relative_delta": total_delta,
            "active_target_relative_delta": active_delta,
            "total_target_passed": total_delta is None or abs(total_delta) <= 0.005,
            "active_target_passed": active_delta is None or abs(active_delta) <= 0.05,
            "kv_cache_elements_per_token_per_layer": compressed_cache_elements,
            "expanded_kv_elements_per_token_per_layer": expanded_cache_elements,
            "kv_cache_compression_ratio": cache_compression_ratio,
            "kv_cache_bf16_bytes_per_token_all_layers": (
                compressed_cache_elements * self.num_layers * 2
            ),
            "bf16_parameter_bytes": total * 2,
            "fp32_parameter_bytes": total * 4,
            # BF16 parameters and gradients, FP32 master parameters, and two
            # FP32 Adam moments. Activations and communication buffers vary by
            # topology and are intentionally not hidden inside this estimate.
            "adamw_mixed_precision_state_bytes_lower_bound": total * 16,
        }

    def parameter_estimate(self) -> int:
        return int(self.parameter_report()["total"])

    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()

    def distributed_topology_report(
        self,
        *,
        tensor_parallel: int,
        pipeline_parallel: int,
        data_parallel: int,
        context_parallel: int = 1,
        expert_parallel: int = 1,
        gpus_per_node: int | None = None,
    ) -> dict[str, Any]:
        dimensions = {
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": pipeline_parallel,
            "data_parallel": data_parallel,
            "context_parallel": context_parallel,
            "expert_parallel": expert_parallel,
        }
        failures = [f"{name} must be positive" for name, value in dimensions.items() if value < 1]
        if failures:
            return {"passed": False, "failures": failures, "dimensions": dimensions}
        if self.hidden_size % tensor_parallel:
            failures.append("hidden_size is not divisible by tensor_parallel")
        if self.num_attention_heads % tensor_parallel:
            failures.append("num_attention_heads is not divisible by tensor_parallel")
        if self.num_layers % pipeline_parallel:
            failures.append("num_layers is not divisible by pipeline_parallel")
        if data_parallel % expert_parallel:
            failures.append("data_parallel is not divisible by expert_parallel")
        if self.feed_forward_architecture == "dense" and expert_parallel != 1:
            failures.append("dense architecture cannot use expert_parallel > 1")
        if self.feed_forward_architecture == "moe" and self.num_routed_experts % expert_parallel:
            failures.append("num_routed_experts is not divisible by expert_parallel")
        world_size = tensor_parallel * pipeline_parallel * context_parallel * data_parallel
        if gpus_per_node is not None and gpus_per_node < 1:
            failures.append("gpus_per_node must be positive")
        nodes = None if gpus_per_node is None else (world_size + gpus_per_node - 1) // gpus_per_node
        return {
            "passed": not failures,
            "cluster_proven": False,
            "failures": failures,
            "dimensions": dimensions,
            "world_size": world_size,
            "nodes_required": nodes,
            "gpus_per_node": gpus_per_node,
            "layers_per_pipeline_stage": self.num_layers // pipeline_parallel,
            "attention_heads_per_tensor_rank": self.num_attention_heads // tensor_parallel,
            "routed_experts_per_expert_rank": (
                self.num_routed_experts // expert_parallel
                if self.feed_forward_architecture == "moe"
                else 0
            ),
            "architecture_sha256": self.contract_sha256(),
            "proof_required": [
                "node GPU memory inventory",
                "NVLink or NVSwitch topology",
                "RDMA bandwidth and NCCL health",
                "expert p99 load at or below configured limit",
                "distributed checkpoint save and reload",
            ],
        }


class ContextCurriculumStage(StrictModel):
    name: str
    sequence_length: int = Field(ge=32_768, le=5_000_000)
    training_mode: Literal["pretrain", "mixed_length", "context_parallel", "evaluation"]
    minimum_short_context_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_ruler_score: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    required_evaluations: tuple[str, ...] = ()
    production_claim: str


def context_curriculum() -> tuple[ContextCurriculumStage, ...]:
    return (
        ContextCurriculumStage(
            name="native-32k",
            sequence_length=32_768,
            training_mode="pretrain",
            required_evaluations=("short_context_retention",),
            production_claim="training_baseline",
        ),
        ContextCurriculumStage(
            name="native-64k",
            sequence_length=65_536,
            training_mode="pretrain",
            required_evaluations=("short_context_retention", "repo_dependency_trace"),
            production_claim="training_baseline",
        ),
        ContextCurriculumStage(
            name="native-128k",
            sequence_length=131_072,
            training_mode="mixed_length",
            required_evaluations=("ruler", "helmet", "repoqa", "context_order"),
            production_claim="verified_only_after_evaluation",
        ),
        ContextCurriculumStage(
            name="native-256k",
            sequence_length=262_144,
            training_mode="context_parallel",
            minimum_ruler_score=0.80,
            required_evaluations=("ruler", "helmet", "repoqa", "cross_file_patch_localization"),
            production_claim="built_not_cluster_proven",
        ),
        ContextCurriculumStage(
            name="native-1m",
            sequence_length=1_000_000,
            training_mode="context_parallel",
            minimum_ruler_score=0.80,
            required_evaluations=("ruler", "helmet", "repoqa", "context_order", "unsupported_claims"),
            production_claim="built_not_cluster_proven",
        ),
        ContextCurriculumStage(
            name="effective-5m",
            sequence_length=5_000_000,
            training_mode="evaluation",
            minimum_ruler_score=0.80,
            required_evaluations=("hierarchical_retrieval", "evidence_recall", "unsupported_claims"),
            production_claim="hierarchical_effective_context_not_full_attention",
        ),
    )


class TrainingDataContract(StrictModel):
    manifest_path: str
    token_count_estimate: int = Field(ge=0)
    domains: list[str] = Field(default_factory=list)
    contamination_checked: bool = False
    license_checked: bool = False
    pii_scrubbed: bool = False

    def required_asset_paths(self) -> list[Path]:
        return [Path(self.manifest_path)]


class ParallelismPlan(StrictModel):
    tensor_parallel: int = Field(default=1, ge=1)
    pipeline_parallel: int = Field(default=1, ge=1)
    data_parallel: int = Field(default=1, ge=1)
    context_parallel: int = Field(default=1, ge=1)
    expert_parallel: int = Field(default=1, ge=1)
    sequence_parallel: bool = True
    zero_stage: int = Field(default=2, ge=0, le=3)

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel
            * self.pipeline_parallel
            * self.context_parallel
            * self.data_parallel
        )

    @model_validator(mode="after")
    def validate_parallelism(self) -> "ParallelismPlan":
        if self.data_parallel % self.expert_parallel:
            raise ValueError("data_parallel must be divisible by expert_parallel")
        return self


class PretrainingRunSpec(StrictModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    architecture: ScratchDecoderConfig
    tokenizer: TokenizerContract = Field(default_factory=TokenizerContract)
    data: TrainingDataContract
    parallelism: ParallelismPlan = Field(default_factory=ParallelismPlan)
    sequence_length: int = Field(default=32_768, ge=2_048)
    global_batch_tokens: int = Field(default=4_194_304, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    min_learning_rate: float = Field(default=3e-5, gt=0)
    warmup_tokens: int = Field(default=10_000_000_000, ge=0)
    target_train_tokens: int = Field(default=1_000_000_000_000, ge=1)
    checkpoint_interval_tokens: int = Field(default=10_000_000_000, ge=1)
    output_dir: str = "artifacts/aeitron/pretraining"

    @model_validator(mode="after")
    def validate_training_shape(self) -> "PretrainingRunSpec":
        if self.sequence_length > self.architecture.max_sequence_length:
            raise ValueError("sequence_length cannot exceed architecture.max_sequence_length")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        canonical_hashes = {profile.contract_sha256() for profile in model_profiles().values()}
        if self.architecture.contract_sha256() not in canonical_hashes:
            raise ValueError("architecture must match an immutable canonical Aeitron model profile")
        return self

    def readiness_report(self) -> dict[str, Any]:
        missing_assets = []
        for asset in [*self.tokenizer.required_asset_paths(), *self.data.required_asset_paths()]:
            if not asset.exists():
                missing_assets.append(str(asset))
        policy_failures = []
        if not self.data.contamination_checked:
            policy_failures.append("data contamination check is not complete")
        if not self.data.license_checked:
            policy_failures.append("data license check is not complete")
        if not self.data.pii_scrubbed:
            policy_failures.append("data PII scrub is not complete")
        return {
            "run_id": self.run_id,
            "ready": not missing_assets and not policy_failures,
            "missing_assets": missing_assets,
            "policy_failures": policy_failures,
            "parameter_estimate": self.architecture.parameter_report(),
            "world_size": self.parallelism.world_size,
            "scratch_training": True,
            "external_model_training": False,
            "scratch_only": True,
        }


class CheckpointManifest(StrictModel):
    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    architecture_name: str
    run_id: str
    step: int = Field(ge=0)
    trained_tokens: int = Field(ge=0)
    checkpoint_dir: str
    files: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=time.time)

    @classmethod
    def from_directory(
        cls,
        *,
        architecture_name: str,
        run_id: str,
        step: int,
        trained_tokens: int,
        checkpoint_dir: str | Path,
        metrics: dict[str, float] | None = None,
    ) -> "CheckpointManifest":
        root = Path(checkpoint_dir).resolve()
        files = []
        if root.exists():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return cls(
            architecture_name=architecture_name,
            run_id=run_id,
            step=step,
            trained_tokens=trained_tokens,
            checkpoint_dir=str(root),
            files=files,
            metrics=metrics or {},
        )

    def write_atomic(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.model_dump(), indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            handle.write(payload)
            handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, target)
        return target


def tiny_smoke_config() -> ScratchDecoderConfig:
    return ScratchDecoderConfig(
        name="aeitron-tiny-gpu-smoke",
        vocab_size=2048,
        max_sequence_length=128,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
    )


def model_profiles() -> dict[str, ScratchDecoderConfig]:
    """Return canonical immutable-by-construction model profiles."""
    return {
        "tiny": tiny_smoke_config(),
        "tiny_moe": ScratchDecoderConfig(
            name="aeitron-tiny-mla-moe",
            vocab_size=512,
            max_sequence_length=128,
            effective_context_length=512,
            hidden_size=64,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=192,
            attention_architecture="mla",
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            feed_forward_architecture="moe",
            num_dense_layers=1,
            num_routed_experts=4,
            num_shared_experts=1,
            experts_per_token=2,
            moe_intermediate_size=64,
            mtp_num_layers=1,
        ),
        "t4_validation": ScratchDecoderConfig(
            name="aeitron-t4-validation",
            vocab_size=128_000,
            max_sequence_length=2048,
            hidden_size=512,
            num_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=2048,
            gradient_checkpointing=True,
        ),
        "50m": ScratchDecoderConfig(
            name="aeitron-50m-scaling-pilot",
            vocab_size=128_000,
            max_sequence_length=2_048,
            hidden_size=320,
            num_layers=8,
            num_attention_heads=10,
            num_key_value_heads=2,
            intermediate_size=960,
            gradient_checkpointing=True,
        ),
        "100m": ScratchDecoderConfig(
            name="aeitron-100m-dense-ab",
            vocab_size=128_000,
            max_sequence_length=2_048,
            hidden_size=512,
            num_layers=12,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1_536,
            gradient_checkpointing=True,
        ),
        "100m_moe": ScratchDecoderConfig(
            name="aeitron-100m-active-mla-moe-ab",
            vocab_size=128_000,
            max_sequence_length=2_048,
            hidden_size=512,
            num_layers=12,
            num_attention_heads=8,
            num_key_value_heads=1,
            intermediate_size=1_536,
            attention_architecture="mla",
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            v_head_dim=64,
            feed_forward_architecture="moe",
            num_dense_layers=2,
            num_routed_experts=8,
            num_shared_experts=1,
            experts_per_token=2,
            moe_intermediate_size=480,
            router_bias_update_rate=0.001,
            mtp_num_layers=1,
            gradient_checkpointing=True,
        ),
        "300m": ScratchDecoderConfig(
            name="aeitron-300m-dense-ab",
            vocab_size=128_000,
            max_sequence_length=2_048,
            hidden_size=1_024,
            num_layers=14,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=3_072,
            gradient_checkpointing=True,
        ),
        "300m_moe": ScratchDecoderConfig(
            name="aeitron-300m-active-mla-moe-ab",
            vocab_size=128_000,
            max_sequence_length=2_048,
            hidden_size=1_024,
            num_layers=14,
            num_attention_heads=16,
            num_key_value_heads=1,
            intermediate_size=3_072,
            attention_architecture="mla",
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            v_head_dim=64,
            feed_forward_architecture="moe",
            num_dense_layers=2,
            num_routed_experts=8,
            num_shared_experts=1,
            experts_per_token=2,
            moe_intermediate_size=986,
            router_bias_update_rate=0.001,
            mtp_num_layers=1,
            gradient_checkpointing=True,
        ),
        "1b": ScratchDecoderConfig(
            name="aeitron-1b-scratch",
            vocab_size=128_000,
            max_sequence_length=65_536,
            effective_context_length=5_000_000,
            hidden_size=2048,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=5504,
            rope_scaling_type="yarn",
            rope_scaling_factor=2.0,
        ),
        "1b_moe": ScratchDecoderConfig(
            name="aeitron-1b-active-mla-moe-ab",
            vocab_size=128_000,
            max_sequence_length=65_536,
            effective_context_length=5_000_000,
            hidden_size=2048,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=1,
            intermediate_size=5504,
            attention_architecture="mla",
            q_lora_rank=512,
            kv_lora_rank=128,
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=128,
            feed_forward_architecture="moe",
            num_dense_layers=4,
            num_routed_experts=32,
            num_shared_experts=1,
            experts_per_token=4,
            moe_intermediate_size=1152,
            router_bias_update_rate=0.001,
            mtp_num_layers=1,
            rope_scaling_type="yarn",
            rope_scaling_factor=2.0,
            gradient_checkpointing=True,
            runtime_backend="megatron_core",
            target_active_parameters=1_325_500_000,
        ),
        "7b": ScratchDecoderConfig(
            name="aeitron-7b-scratch",
            vocab_size=128_000,
            max_sequence_length=131_072,
            effective_context_length=5_000_000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=11008,
            rope_scaling_type="yarn",
            rope_scaling_factor=4.0,
        ),
        "7b_moe": ScratchDecoderConfig(
            name="aeitron-7b-active-mla-moe-candidate",
            vocab_size=128_000,
            max_sequence_length=131_072,
            effective_context_length=5_000_000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=1,
            intermediate_size=11008,
            attention_architecture="mla",
            q_lora_rank=1024,
            kv_lora_rank=256,
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=128,
            feed_forward_architecture="moe",
            num_dense_layers=4,
            num_routed_experts=32,
            num_shared_experts=1,
            experts_per_token=4,
            moe_intermediate_size=2368,
            router_bias_update_rate=0.001,
            mtp_num_layers=1,
            rope_scaling_type="yarn",
            rope_scaling_factor=4.0,
            gradient_checkpointing=True,
            runtime_backend="megatron_core",
            target_active_parameters=6_195_253_248,
        ),
        "32b": ScratchDecoderConfig(
            name="aeitron-32b-scratch",
            vocab_size=128_000,
            max_sequence_length=131_072,
            effective_context_length=5_000_000,
            hidden_size=6656,
            num_layers=60,
            num_attention_heads=52,
            num_key_value_heads=4,
            intermediate_size=17920,
            rope_scaling_type="yarn",
            rope_scaling_factor=4.0,
        ),
        "62b": ScratchDecoderConfig(
            name="aeitron-62b-scratch",
            vocab_size=128_000,
            max_sequence_length=262_144,
            effective_context_length=5_000_000,
            hidden_size=8192,
            num_layers=72,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=28672,
            rope_scaling_type="longrope",
            rope_scaling_factor=8.0,
            gradient_checkpointing=True,
        ),
        "4t_moe": ScratchDecoderConfig(
            name="aeitron-4t-moe-128b-active",
            vocab_size=128_000,
            max_sequence_length=1_000_000,
            effective_context_length=5_000_000,
            hidden_size=16_384,
            num_layers=96,
            num_attention_heads=128,
            num_key_value_heads=1,
            intermediate_size=53_248,
            attention_architecture="mla",
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            feed_forward_architecture="moe",
            num_dense_layers=4,
            num_routed_experts=256,
            num_shared_experts=1,
            experts_per_token=4,
            moe_intermediate_size=3392,
            router_bias_update_rate=0.001,
            mtp_num_layers=1,
            rope_scaling_type="longrope",
            rope_scaling_factor=32.0,
            gradient_checkpointing=True,
            runtime_backend="megatron_core",
            target_total_parameters=4_000_000_000_000,
            target_active_parameters=128_000_000_000,
        ),
    }


def model_profile(name: str) -> ScratchDecoderConfig:
    normalized = name.lower()
    key = normalized.removeprefix("craftly-").removeprefix("aeitron-").replace("-", "_")
    profiles = model_profiles()
    if key in profiles:
        return profiles[key]
    if normalized in profiles:
        return profiles[normalized]
    raise ValueError(f"unknown model profile: {name}")


def architecture_presets() -> dict[str, ScratchDecoderConfig]:
    """Public compatibility view backed by the canonical model contracts."""
    presets = {
        f"craftly-{key.replace('_', '-')}": model_profile(key)
        for key in ("7b", "7b_moe", "32b", "62b", "4t_moe")
    }
    presets.update({
        f"aeitron-{key.replace('_', '-')}": model_profile(key)
        for key in ("7b", "7b_moe", "32b", "62b", "4t_moe")
    })
    return presets


def foundation_status() -> dict[str, Any]:
    presets = architecture_presets()
    final_profile = model_profile("4t_moe")
    return {
        "scratch_first": True,
        "scratch_only": True,
        "external_model_training": False,
        "presets": {name: spec.parameter_report() for name, spec in presets.items()},
        "canonical_4t_contract": {
            "architecture_sha256": final_profile.contract_sha256(),
            "parameter_report": final_profile.parameter_report(),
            "runtime_backend": final_profile.runtime_backend,
            "status": "built_not_cluster_proven",
        },
        "context_curriculum": [stage.model_dump(mode="json") for stage in context_curriculum()],
        "required_before_training": [
            "tokenizer asset",
            "deduplicated dataset manifest",
            "license and PII gates",
            "contamination gate",
            "GPU cluster plan",
            "checkpoint storage plan",
            "evaluation baseline",
        ],
    }

