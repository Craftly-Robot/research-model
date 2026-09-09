"""Checkpoint-resumable scratch pretraining loop for Craftly decoder models."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hmac
import json
import math
import os
import random
import shutil
import subprocess  # nosec B404
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Literal

from src.craftly.model_ops.data_loader import ArtifactCache, TokenShardStream, count_batches, load_manifest
from src.craftly.model_ops.foundation import (
    TRAINING_STEP_SEMANTICS,
    CheckpointManifest,
    TrainingBatchContract,
    model_profiles,
    sha256_file,
)
from src.craftly.model_ops.tokenizer_pipeline import ShardBuildConfig, ShardManifest, build_token_shards, load_tokenizer, read_uint32_tokens
from src.craftly.model_ops.torch_decoder import (
    DecoderBlock,
    CraftlyDecoderLM,
    ScratchDecoderConfig,
    load_trusted_checkpoint,
    model_profile,
    require_torch,
    save_trusted_checkpoint,
    select_torch_device,
    tiny_smoke_config,
)
from src.craftly.shared.progress import NullProgressReporter, ProgressReporter

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from src.craftly.learning.continual import ElasticWeightConsolidation


DistributedStrategy = Literal["none", "fsdp", "deepspeed_zero2", "deepspeed_zero3", "megatron"]
LearningRateSchedule = Literal["constant", "linear", "cosine"]


def build_learning_rate_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: LearningRateSchedule,
    minimum_learning_rate_ratio: float,
) -> Any:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if not 0.0 <= minimum_learning_rate_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be between 0 and 1")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(minimum_learning_rate_ratio, float(step + 1) / float(warmup_steps))
        if schedule == "constant":
            return 1.0
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        if schedule == "linear":
            factor = 1.0 - progress
        elif schedule == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported learning-rate schedule: {schedule}")
        return minimum_learning_rate_ratio + (1.0 - minimum_learning_rate_ratio) * factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def validate_runtime_versions(
    *,
    expected_python: str | None,
    expected_pytorch: str | None,
    expected_cuda: str | None,
    device: Any,
) -> dict[str, str]:
    actual = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": str(torch.__version__).split("+", 1)[0],
        "cuda": str(torch.version.cuda or "none"),
    }
    expected = {"python": expected_python, "pytorch": expected_pytorch, "cuda": expected_cuda}
    for name, expected_value in expected.items():
        if not expected_value:
            continue
        if name == "cuda" and device.type != "cuda":
            raise RuntimeError(f"runtime contract requires CUDA {expected_value}, but selected device is {device.type}")
        if not actual[name].startswith(expected_value):
            raise RuntimeError(
                f"runtime {name} mismatch: expected prefix {expected_value!r}, actual={actual[name]!r}"
            )
    return actual


def is_deepspeed_strategy(strategy: DistributedStrategy) -> bool:
    return strategy in {"deepspeed_zero2", "deepspeed_zero3"}


def default_deepspeed_config_path(strategy: DistributedStrategy) -> Path:
    if strategy == "deepspeed_zero2":
        return Path("deploy/gpu/deepspeed_zero2.json")
    if strategy == "deepspeed_zero3":
        return Path("deploy/gpu/deepspeed_zero3.json")
    raise ValueError(f"strategy does not use DeepSpeed: {strategy}")


def load_deepspeed_config(
    *,
    strategy: DistributedStrategy,
    config_path: str | Path | None,
    batch_size: int,
    gradient_accumulation_steps: int,
    dtype: str,
) -> dict[str, Any]:
    path = Path(config_path) if config_path else default_deepspeed_config_path(strategy)
    if not path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    world_size = max(1, distributed_world_size())
    payload["train_micro_batch_size_per_gpu"] = batch_size
    payload["gradient_accumulation_steps"] = gradient_accumulation_steps
    payload["train_batch_size"] = batch_size * gradient_accumulation_steps * world_size
    payload["bf16"] = {"enabled": dtype == "bf16"}
    payload["fp16"] = {"enabled": dtype == "fp16"}
    return payload


def build_cluster_training_plan(
    *,
    output_dir: str | Path,
    manifest: str | Path,
    model_profile_name: str,
    strategy: DistributedStrategy,
    num_nodes: int,
    gpus_per_node: int,
    sequence_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    steps: int,
    dtype: str,
    attention_impl: str = "auto",
    gradient_checkpointing: bool = True,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    deepspeed_config: str | Path | None = None,
    megatron_root: str | Path | None = None,
) -> dict[str, Any]:
    if strategy == "none":
        raise ValueError("cluster training plan requires a distributed strategy")
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be >= 1")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("dtype must be one of bf16, fp16, fp32")
    if not Path(manifest).exists():
        raise FileNotFoundError(f"shard manifest not found: {manifest}")
    profile = model_profile(model_profile_name) if model_profile_name != "tiny" else tiny_smoke_config()
    total_gpus = num_nodes * gpus_per_node
    global_sequences = total_gpus * batch_size * gradient_accumulation_steps
    tokens_per_optimizer_step = global_sequences * sequence_length
    base_train_args = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--dtype",
        dtype,
        "--model-profile",
        model_profile_name,
        "--attention-impl",
        attention_impl,
    ]
    if gradient_checkpointing:
        base_train_args.append("--gradient-checkpointing")

    warnings: list[str] = []
    required_env = {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "NODE_RANK": str(node_rank),
        "WORLD_SIZE": str(total_gpus),
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    if model_profile_name in {"32b", "62b"} and total_gpus < 8:
        warnings.append("32B/62B profiles normally need >=8 high-memory GPUs; this plan may OOM on smaller clusters.")
    if sequence_length > 8192 and strategy not in {"deepspeed_zero3", "megatron"}:
        warnings.append("Long-context runs above 8k tokens should usually use ZeRO-3/Megatron-style partitioning.")
    if dtype == "fp32" and model_profile_name != "tiny":
        warnings.append("fp32 is not practical for large-profile training; prefer bf16 on Ampere/Hopper or fp16 where required.")

    if strategy == "fsdp":
        launcher = "torchrun"
        command = [
            "torchrun",
            "--nnodes",
            str(num_nodes),
            "--nproc_per_node",
            str(gpus_per_node),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            master_addr,
            "--master_port",
            str(master_port),
            "-m",
            "src.craftly.model_ops.pretrain_loop",
            "--distributed-strategy",
            "fsdp",
            *base_train_args,
        ]
    elif strategy in {"deepspeed_zero2", "deepspeed_zero3"}:
        launcher = "deepspeed"
        if deepspeed_config is None:
            stage = "2" if strategy == "deepspeed_zero2" else "3"
            warnings.append(f"No DeepSpeed JSON path provided; generate/use a ZeRO-{stage} config before launching.")
        elif not Path(deepspeed_config).exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_config}")
        command = [
            "deepspeed",
            "--num_nodes",
            str(num_nodes),
            "--num_gpus",
            str(gpus_per_node),
            "-m",
            "src.craftly.model_ops.pretrain_loop",
            "--distributed-strategy",
            strategy,
            *base_train_args,
        ]
        if deepspeed_config is not None:
            command.extend(["--deepspeed-config", str(deepspeed_config)])
    else:
        launcher = "megatron"
        if megatron_root is None:
            warnings.append("Megatron root path is not set; set --megatron-root to the checked-out Megatron-LM repository.")
        elif not Path(megatron_root).exists():
            raise FileNotFoundError(f"Megatron root not found: {megatron_root}")
        command = [
            "torchrun",
            "--nnodes",
            str(num_nodes),
            "--nproc_per_node",
            str(gpus_per_node),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            master_addr,
            "--master_port",
            str(master_port),
            "-m",
            "src.craftly.model_ops.pretrain_loop",
            "--distributed-strategy",
            "megatron",
            *base_train_args,
        ]
    return {
        "status": "ready_with_warnings" if warnings else "ready",
        "scratch_only": True,
        "strategy": strategy,
        "launcher": launcher,
        "command": command,
        "env": required_env,
        "model_profile": profile.model_dump(),
        "estimated_parameter_count": profile.parameter_estimate(),
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "total_gpus": total_gpus,
        "batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "global_sequences_per_optimizer_step": global_sequences,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "target_train_tokens": tokens_per_optimizer_step * steps,
        "warnings": warnings,
        "required_release_gates": [
            "tokenizer audit passes on real corpus",
            "dataset contamination gate passes",
            "10k-step single-node GPU validation passes",
            "distributed dry-run initializes every rank",
            "first cluster checkpoint reloads and evaluates",
        ],
    }


def write_cluster_training_plan(*, output_path: str | Path, plan: dict[str, Any]) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return target


def distributed_is_initialized() -> bool:
    require_torch()
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def distributed_rank() -> int:
    if distributed_is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def distributed_world_size() -> int:
    if distributed_is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def distributed_barrier() -> None:
    if distributed_is_initialized():
        torch.distributed.barrier()


def initialize_distributed_runtime(strategy: DistributedStrategy, requested_device: "torch.device") -> dict[str, Any]:
    require_torch()
    if strategy == "none":
        return {"enabled": False, "strategy": "none", "rank": 0, "world_size": 1, "local_rank": 0}
    if is_deepspeed_strategy(strategy):
        try:
            import deepspeed
        except ImportError as exc:
            raise RuntimeError("DeepSpeed strategy requested but deepspeed is not installed") from exc
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if requested_device.type == "cuda":
            torch.cuda.set_device(local_rank)
        return {
            "enabled": True,
            "strategy": strategy,
            "rank": distributed_rank(),
            "world_size": distributed_world_size(),
            "local_rank": local_rank,
            "backend": torch.distributed.get_backend() if torch.distributed.is_initialized() else "",
        }
    if strategy != "fsdp":
        raise RuntimeError(
            f"{strategy} requires its dedicated engine adapter and cluster release gate. "
            "Use --cluster-plan-only to validate launch math; use --distributed-strategy fsdp for native torch FSDP runtime."
        )
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if not torch.distributed.is_initialized():
        backend = "nccl" if requested_device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    if requested_device.type == "cuda":
        torch.cuda.set_device(local_rank)
    return {
        "enabled": True,
        "strategy": strategy,
        "rank": distributed_rank(),
        "world_size": distributed_world_size(),
        "local_rank": local_rank,
        "backend": torch.distributed.get_backend(),
    }


def wrap_for_distributed(
    model: "CraftlyDecoderLM",
    *,
    strategy: DistributedStrategy,
    dtype: str,
    device: "torch.device",
) -> "torch.nn.Module":
    require_torch()
    if strategy == "none":
        return model
    if strategy != "fsdp":
        raise RuntimeError(f"unsupported distributed runtime strategy: {strategy}")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    mixed_precision = None
    if device.type == "cuda" and dtype in {"bf16", "fp16"}:
        active_dtype = autocast_dtype(dtype)
        mixed_precision = MixedPrecision(param_dtype=active_dtype, reduce_dtype=active_dtype, buffer_dtype=active_dtype)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={DecoderBlock})
    return FSDP(model, auto_wrap_policy=auto_wrap_policy, mixed_precision=mixed_precision, use_orig_params=True)


def checkpoint_model_state_dict(model: "torch.nn.Module") -> dict[str, Any]:
    require_torch()
    if hasattr(model, "module") and model.__class__.__name__.lower().endswith("engine"):
        return model.module.state_dict()
    try:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
    except ImportError:
        return model.state_dict()
    if isinstance(model, FSDP):
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            return model.state_dict()
    return model.state_dict()


def wrap_for_deepspeed(
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler",
    *,
    strategy: DistributedStrategy,
    config_path: str | Path | None,
    batch_size: int,
    gradient_accumulation_steps: int,
    dtype: str,
) -> tuple["torch.nn.Module", "torch.optim.Optimizer", "torch.optim.lr_scheduler.LRScheduler"]:
    if not is_deepspeed_strategy(strategy):
        return model, optimizer, scheduler
    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("DeepSpeed strategy requested but deepspeed is not installed") from exc
    ds_config = load_deepspeed_config(
        strategy=strategy,
        config_path=config_path,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dtype=dtype,
    )
    engine, active_optimizer, _loader, active_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    return engine, active_optimizer, active_scheduler


def autocast_dtype(name: str) -> "torch.dtype":
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for directory in root.glob("checkpoint-step-*"):
        try:
            step = int(directory.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (directory / "dcp" / ".metadata").is_file():
            candidates.append((step, directory / "dcp"))
        elif (directory / "deepspeed" / "latest").is_file():
            candidates.append((step, directory / "deepspeed"))
        elif (directory / "model.pt").is_file():
            candidates.append((step, directory / "model.pt"))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def checkpoint_payload_from_manifest(manifest_path: str | Path) -> Path:
    """Resolve and verify a single-process checkpoint from its signed-by-hash manifest."""

    source = Path(manifest_path).resolve(strict=True)
    manifest = CheckpointManifest.model_validate_json(source.read_text(encoding="utf-8-sig"))
    root = Path(manifest.checkpoint_dir).resolve(strict=True)
    if not manifest.files:
        raise ValueError("initial checkpoint manifest contains no files")
    for entry in manifest.files:
        relative = str(entry.get("path") or "")
        candidate = (root / relative).resolve(strict=True)
        if candidate != root and root not in candidate.parents:
            raise ValueError("initial checkpoint manifest contains a path outside checkpoint_dir")
        expected_size = int(entry.get("size_bytes", -1))
        expected_hash = str(entry.get("sha256") or "")
        if candidate.stat().st_size != expected_size or sha256_file(candidate) != expected_hash:
            raise ValueError(f"initial checkpoint integrity check failed: {relative}")
    payload = root / "model.pt"
    if not payload.is_file():
        raise ValueError("external checkpoint initialization currently requires a single-process model.pt checkpoint")
    return payload


def _json_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"checkpoint state contains unsupported JSON value: {type(value).__name__}")


def _validate_checkpoint_training_invariants(
    trainer_state: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    if expected is None:
        return
    saved = trainer_state.get("training_args")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint has no immutable training argument contract")
    mismatched = {
        key: {"checkpoint": saved.get(key), "expected": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatched:
        names = ", ".join(sorted(mismatched))
        raise ValueError(f"checkpoint training semantics are incompatible: {names}")
    step = int(trainer_state.get("step", 0))
    dataloader_state = trainer_state.get("dataloader_state")
    if not isinstance(dataloader_state, dict):
        raise ValueError("checkpoint has no deterministic dataloader cursor")
    accumulation = int(expected["gradient_accumulation_steps"])
    batches_per_epoch = int(expected["batches_per_rank_per_epoch"])
    expected_micro_batches = step * accumulation
    if int(dataloader_state.get("completed_micro_batches_per_rank", -1)) != expected_micro_batches:
        raise ValueError("checkpoint dataloader cursor does not match optimizer step and accumulation contract")
    expected_cursor = {
        "semantics": expected["step_semantics"],
        "completed_optimizer_steps": step,
        "epoch": expected_micro_batches // batches_per_epoch,
        "batch_offset_within_epoch": expected_micro_batches % batches_per_epoch,
        "batches_per_rank_per_epoch": batches_per_epoch,
        "world_size": expected["data_parallel_world_size"],
    }
    mismatched_cursor = [
        key for key, value in expected_cursor.items() if dataloader_state.get(key) != value
    ]
    if mismatched_cursor:
        raise ValueError(
            "checkpoint deterministic dataloader cursor is incompatible: "
            + ", ".join(sorted(mismatched_cursor))
        )


def _save_dcp_checkpoint(
    *,
    checkpoint_dir: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None",
    trainer_state: dict[str, Any],
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".incomplete")
    if distributed_rank() == 0:
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=False)
    distributed_barrier()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(temporary / "dcp"),
    )
    rank_state = {
        "rank": distributed_rank(),
        "world_size": distributed_world_size(),
        "torch_rng_state": torch.get_rng_state().tolist(),
        "cuda_rng_states": [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }
    (temporary / f"rank-{distributed_rank():05d}.json").write_text(
        json.dumps(rank_state, sort_keys=True), encoding="utf-8"
    )
    distributed_barrier()
    if distributed_rank() == 0:
        common = dict(trainer_state)
        common["scheduler"] = _json_state(scheduler.state_dict()) if scheduler is not None else {}
        (temporary / "trainer_state.json").write_text(
            json.dumps(common, indent=2, sort_keys=True), encoding="utf-8"
        )
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        os.replace(temporary, checkpoint_dir)
    distributed_barrier()


def _load_dcp_checkpoint(
    *,
    checkpoint_path: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None",
    expected_config: ScratchDecoderConfig | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_training_invariants: dict[str, Any] | None = None,
) -> tuple[int, int]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(checkpoint_path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    root = checkpoint_path.parent
    trainer_state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    _validate_checkpoint_training_invariants(trainer_state, expected_training_invariants)
    if expected_config is not None:
        try:
            saved_config = ScratchDecoderConfig.model_validate(trainer_state.get("config") or {})
        except Exception as exc:
            raise ValueError(f"distributed checkpoint model config is invalid: {exc}") from exc
        if saved_config.model_dump() != expected_config.model_dump():
            mismatched = sorted(
                key
                for key, value in expected_config.model_dump().items()
                if saved_config.model_dump().get(key) != value
            )
            raise ValueError(f"distributed checkpoint architecture config is incompatible: {', '.join(mismatched)}")
    if expected_dataset_manifest_sha256 is not None:
        if str(trainer_state.get("dataset_manifest_sha256") or "") != expected_dataset_manifest_sha256:
            raise ValueError("distributed checkpoint dataset manifest hash does not match immutable training input")
    if expected_tokenizer_sha256 is not None:
        if str(trainer_state.get("tokenizer_sha256") or "") != expected_tokenizer_sha256:
            raise ValueError("distributed checkpoint tokenizer hash does not match immutable training input")
    if scheduler is not None and trainer_state.get("scheduler"):
        scheduler.load_state_dict(trainer_state["scheduler"])
    rank_path = root / f"rank-{distributed_rank():05d}.json"
    if not rank_path.is_file():
        raise FileNotFoundError(f"distributed checkpoint has no RNG state for rank {distributed_rank()}")
    rank_state = json.loads(rank_path.read_text(encoding="utf-8"))
    torch.set_rng_state(torch.tensor(rank_state["torch_rng_state"], dtype=torch.uint8))
    if torch.cuda.is_available() and rank_state.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8, device="cpu") for value in rank_state["cuda_rng_states"]]
        )
    return int(trainer_state.get("step", 0)), int(trainer_state.get("trained_tokens", 0))


def publish_checkpoint_to_workspace(
    *,
    manifest_path: Path,
    attempt_id: str,
    dataset_sha256: str,
    tokenizer_sha256: str,
    topology: dict[str, Any],
    metrics: dict[str, Any],
    reload_verified: bool,
    maximum_workers: int = 8,
) -> dict[str, Any] | None:
    workspace_url = os.environ.get("CRAFTLY_WORKSPACE_URL", "").rstrip("/")
    job_id = os.environ.get("CRAFTLY_TRAINING_JOB_ID", "")
    token_file = os.environ.get("CRAFTLY_WORKSPACE_TOKEN_FILE", "")
    access_token = os.environ.get("CRAFTLY_WORKSPACE_ACCESS_TOKEN", "")
    if not workspace_url or not job_id or not attempt_id or not (token_file or access_token):
        return None
    token = Path(token_file).read_text(encoding="utf-8").strip() if token_file else access_token
    if not token:
        raise RuntimeError("workspace checkpoint publication token is empty")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    checkpoint_dir = Path(payload["checkpoint_dir"]).resolve()
    step = int(payload["step"])
    files = list(payload.get("files") or [])
    if not files or len(files) > 100_000:
        raise ValueError("checkpoint manifest file count is outside publication limits")
    headers = {"Authorization": f"Bearer {token}"}
    prefix = f"step-{step:08d}"

    def upload(path: Path, relative_path: str, digest: str, size_bytes: int) -> str:
        import httpx

        request = {
            "attempt_id": attempt_id,
            "kind": "checkpoint",
            "filename": path.name,
            "relative_path": f"{prefix}/{relative_path}",
            "sha256": digest,
            "size_bytes": size_bytes,
            "content_type": "application/json" if path.suffix == ".json" else "application/octet-stream",
        }
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
            response = client.post(
                f"{workspace_url}/v1/training/jobs/{job_id}/artifacts/presign",
                headers=headers,
                json=request,
            )
            response.raise_for_status()
            presigned = response.json()
            with path.open("rb") as handle:
                uploaded = client.put(presigned["url"], headers=presigned["headers"], content=handle)
            uploaded.raise_for_status()
            registered = client.post(
                f"{workspace_url}/v1/training/jobs/{job_id}/artifacts/register",
                headers=headers,
                json={"upload": request, "uri": presigned["uri"]},
            )
            registered.raise_for_status()
            return str(presigned["uri"])

    manifest_digest = sha256_file(manifest_path)
    manifest_uri = upload(
        manifest_path,
        "checkpoint_manifest.json",
        manifest_digest,
        manifest_path.stat().st_size,
    )
    work: list[tuple[Path, str, str, int]] = []
    for entry in files:
        relative = str(entry["path"])
        source = (checkpoint_dir / relative).resolve()
        if checkpoint_dir != source and checkpoint_dir not in source.parents:
            raise ValueError("checkpoint manifest path escapes checkpoint directory")
        if not source.is_file() or sha256_file(source) != str(entry["sha256"]):
            raise ValueError(f"checkpoint file failed publication preflight: {relative}")
        work.append((source, relative.replace("\\", "/"), str(entry["sha256"]), int(entry["size_bytes"])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(maximum_workers, len(work))) as executor:
        futures = [executor.submit(upload, *item) for item in work]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    import httpx

    with httpx.Client(timeout=60.0) as client:
        committed = client.post(
            f"{workspace_url}/v1/training/jobs/{job_id}/checkpoints",
            headers=headers,
            json={
                "attempt_id": attempt_id,
                "step": step,
                "manifest_uri": manifest_uri,
                "manifest_sha256": manifest_digest,
                "dataset_sha256": dataset_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "topology": topology,
                "metrics": metrics,
                "reload_verified": reload_verified,
            },
        )
        committed.raise_for_status()
        return committed.json()


def git_commit(root: str | Path = ".") -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def training_environment_report(*, device: "torch.device", dtype: str, distributed_strategy: DistributedStrategy) -> dict[str, Any]:
    require_torch()
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": __import__("sys").version.split()[0],
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and cuda_available else "",
        "dtype": dtype,
        "distributed_strategy": distributed_strategy,
        "distributed_rank": distributed_rank(),
        "distributed_world_size": distributed_world_size(),
        "env": {
            "LOCAL_RANK": os.environ.get("LOCAL_RANK", ""),
            "RANK": os.environ.get("RANK", ""),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE", ""),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", ""),
            "MASTER_PORT": os.environ.get("MASTER_PORT", ""),
        },
    }


def prune_old_checkpoints(
    output_dir: Path,
    keep_last_n: int = 3,
    preserve_dirs: set[Path] | None = None,
) -> list[Path]:
    """Delete older step checkpoint directories to prevent disk overflow while preserving recent checkpoints."""
    if keep_last_n <= 0:
        return []
    preserve = {p.resolve() for p in (preserve_dirs or set())}
    checkpoint_dirs = sorted(
        [d for d in output_dir.glob("checkpoint-step-*") if d.is_dir()],
        key=lambda d: d.name,
    )
    if len(checkpoint_dirs) <= keep_last_n:
        return []
    to_delete = checkpoint_dirs[:-keep_last_n]
    deleted = []
    for d in to_delete:
        if d.resolve() in preserve:
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(d)
        except Exception:
            pass
    return deleted


def save_training_checkpoint(
    *,
    output_dir: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    config: ScratchDecoderConfig,
    step: int,
    trained_tokens: int,
    metrics: dict[str, float],
    training_args: dict[str, Any] | None = None,
    dataset_manifest_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    environment: dict[str, Any] | None = None,
    dataloader_state: dict[str, Any] | None = None,
    manifest_filename: str = "checkpoint_manifest.json",
    keep_last_n_checkpoints: int = 3,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-step-{step:08d}"
    manifest_path = output_dir / manifest_filename
    dataset_manifest_hash = sha256_file(Path(dataset_manifest_path)) if dataset_manifest_path and Path(dataset_manifest_path).exists() else ""
    tokenizer_hash = sha256_file(Path(tokenizer_path)) if tokenizer_path and Path(tokenizer_path).exists() else ""
    trainer_state = {
        "config": config.model_dump(),
        "step": step,
        "trained_tokens": trained_tokens,
        "metrics": metrics,
        "training_args": training_args or {},
        "dataset_manifest_path": str(dataset_manifest_path or ""),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "tokenizer_path": str(tokenizer_path or ""),
        "tokenizer_sha256": tokenizer_hash,
        "git_commit": git_commit(),
        "environment": environment or {},
        "dataloader_state": dataloader_state or {},
        "distributed_world_size": distributed_world_size(),
    }
    is_fsdp = model.__class__.__name__ == "FullyShardedDataParallel"
    is_deepspeed = hasattr(model, "save_checkpoint") and model.__class__.__name__.lower().endswith("engine")
    if is_fsdp:
        _save_dcp_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
    elif is_deepspeed:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        saved = model.save_checkpoint(
            str(checkpoint_dir / "deepspeed"),
            tag=f"step-{step:08d}",
            client_state=trainer_state,
        )
        if saved is False:
            raise RuntimeError("DeepSpeed rejected distributed checkpoint save")
        distributed_barrier()
    elif distributed_world_size() > 1:
        raise RuntimeError("distributed checkpointing requires FSDP or DeepSpeed engine sharding")
    elif distributed_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_state = checkpoint_model_state_dict(model)
        try:
            save_trusted_checkpoint(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else {"type": "constant", "state": {}},
                    **trainer_state,
                },
                checkpoint_dir / "model.pt",
            )
        except Exception as err:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed writing checkpoint at step {step} to {checkpoint_dir / 'model.pt'}. "
                f"Disk space exhaustion or I/O error: {err}"
            ) from err
    if distributed_rank() == 0:
        (checkpoint_dir / "config.json").write_text(json.dumps(config.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        manifest = CheckpointManifest.from_directory(
            architecture_name=config.name,
            run_id="scratch-pretrain-loop",
            step=step,
            trained_tokens=trained_tokens,
            checkpoint_dir=checkpoint_dir,
            metrics=metrics,
        )
        manifest.write_atomic(manifest_path)
        prune_old_checkpoints(output_dir, keep_last_n=keep_last_n_checkpoints, preserve_dirs={checkpoint_dir})
    distributed_barrier()
    return manifest_path


def checkpoint_logit_probe(
    model: "torch.nn.Module",
    *,
    config: ScratchDecoderConfig,
    device: "torch.device",
) -> "torch.Tensor":
    """Capture deterministic logits used to prove a checkpoint reload is exact enough."""
    sequence_length = min(8, config.max_sequence_length)
    token_ids = torch.arange(sequence_length, dtype=torch.long, device=device).remainder(
        config.vocab_size
    )
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            output = model(token_ids.unsqueeze(0), use_cache=False)
            logits = output.logits.detach().float().cpu().clone()
    finally:
        model.train(was_training)
    if not torch.isfinite(logits).all():
        raise RuntimeError("checkpoint parity probe produced non-finite logits")
    return logits


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    device: "torch.device",
    expected_config: ScratchDecoderConfig | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_training_invariants: dict[str, Any] | None = None,
) -> tuple[int, int]:
    if checkpoint_path.is_dir() and (checkpoint_path / ".metadata").is_file():
        return _load_dcp_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=expected_config,
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            expected_training_invariants=expected_training_invariants,
        )
    if checkpoint_path.is_dir() and (checkpoint_path / "latest").exists() and hasattr(model, "load_checkpoint"):
        loaded_path, client_state = model.load_checkpoint(str(checkpoint_path))
        if not loaded_path:
            raise RuntimeError("DeepSpeed checkpoint reload failed")
        if expected_config is not None:
            saved_config = ScratchDecoderConfig.model_validate(client_state.get("config") or {})
            if saved_config.model_dump() != expected_config.model_dump():
                raise ValueError("DeepSpeed checkpoint architecture config is incompatible")
        if expected_dataset_manifest_sha256 is not None:
            if str(client_state.get("dataset_manifest_sha256") or "") != expected_dataset_manifest_sha256:
                raise ValueError("DeepSpeed checkpoint dataset manifest hash does not match immutable training input")
        if expected_tokenizer_sha256 is not None:
            if str(client_state.get("tokenizer_sha256") or "") != expected_tokenizer_sha256:
                raise ValueError("DeepSpeed checkpoint tokenizer hash does not match immutable training input")
        _validate_checkpoint_training_invariants(client_state, expected_training_invariants)
        return int(client_state.get("step", 0)), int(client_state.get("trained_tokens", 0))
    payload = load_trusted_checkpoint(checkpoint_path, map_location=device)
    _validate_checkpoint_training_invariants(payload, expected_training_invariants)
    if expected_config is not None:
        try:
            saved_config = ScratchDecoderConfig.model_validate(payload.get("config") or {})
        except Exception as exc:
            raise ValueError(f"checkpoint model config is missing or invalid: {exc}") from exc
        expected_payload = expected_config.model_dump()
        saved_payload = saved_config.model_dump()
        mismatched = {
            key: {"checkpoint": saved_payload.get(key), "expected": value}
            for key, value in expected_payload.items()
            if saved_payload.get(key) != value
        }
        if mismatched:
            names = ", ".join(sorted(mismatched))
            raise ValueError(f"checkpoint architecture config is incompatible: {names}")
    if expected_dataset_manifest_sha256 is not None:
        actual_dataset_hash = str(payload.get("dataset_manifest_sha256") or "")
        if actual_dataset_hash != expected_dataset_manifest_sha256:
            raise ValueError("checkpoint dataset manifest hash does not match immutable training input")
    if expected_tokenizer_sha256 is not None:
        actual_tokenizer_hash = str(payload.get("tokenizer_sha256") or "")
        if actual_tokenizer_hash != expected_tokenizer_sha256:
            raise ValueError("checkpoint tokenizer hash does not match immutable training input")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and isinstance(payload.get("scheduler"), dict):
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except Exception as exc:
            raise ValueError(f"checkpoint scheduler state is incompatible: {exc}") from exc
    return int(payload.get("step", 0)), int(payload.get("trained_tokens", 0))


def validate_production_training_args(
    *,
    production_mode: bool,
    dev_smoke: bool,
    model_profile_name: str,
    manifest: str | Path | None,
    tokenizer_path: str | Path | None,
    active_manifest: ShardManifest,
    validate_every: int,
    checkpoint_every: int,
    run_steps: int,
) -> None:
    if not production_mode:
        return
    failures = []
    if model_profile_name == "tiny" and not dev_smoke:
        failures.append("production mode cannot use model-profile=tiny without --dev-smoke")
    if not manifest:
        failures.append("production mode requires an explicit shard manifest")
    if not active_manifest.tokenizer_path or not Path(active_manifest.tokenizer_path).exists():
        failures.append("production mode requires a tokenizer path in the shard manifest")
    if tokenizer_path and not Path(tokenizer_path).exists():
        failures.append("production mode tokenizer_path does not exist")
    if not active_manifest.val_shards or active_manifest.val_tokens < active_manifest.sequence_length:
        failures.append("production mode requires a non-empty validation shard set")
    if active_manifest.split_strategy != "pre_split_family_safe":
        failures.append("production mode requires a pre-split family-safe token corpus")
    if active_manifest.schema_version < 2:
        failures.append("production mode requires document-boundary shard schema v2")
    if (
        active_manifest.document_boundary_token != "<|document_end|>"
        or active_manifest.document_boundary_token_id is None
        or active_manifest.boundary_token_count
        != active_manifest.train_documents + active_manifest.val_documents
        or active_manifest.boundary_token_count < 1
    ):
        failures.append("production mode requires a verified document boundary token for every source row")
    if not active_manifest.dataset_manifest_path or not active_manifest.dataset_manifest_sha256:
        failures.append("production mode requires a promoted dataset manifest binding")
    else:
        dataset_manifest = Path(active_manifest.dataset_manifest_path)
        if not dataset_manifest.is_file() or sha256_file(dataset_manifest) != active_manifest.dataset_manifest_sha256:
            failures.append("production dataset manifest binding failed integrity verification")
    tokenizer_source = Path(active_manifest.tokenizer_path)
    if (
        not active_manifest.tokenizer_sha256
        or not tokenizer_source.is_file()
        or sha256_file(tokenizer_source) != active_manifest.tokenizer_sha256
    ):
        failures.append("production tokenizer binding failed integrity verification")
    if not active_manifest.source_sha256:
        failures.append("production token shards require immutable source bindings")
    for source_path, expected_sha256 in active_manifest.source_sha256.items():
        source = Path(source_path)
        if not source.is_file() or sha256_file(source) != expected_sha256:
            failures.append(f"production tokenizer source binding failed: {source_path}")
    if validate_every <= 0 or validate_every > run_steps:
        failures.append("production mode requires validation to run inside the requested step count")
    if checkpoint_every <= 0:
        failures.append("production mode requires checkpointing")
    if failures:
        raise ValueError("production training validation failed: " + "; ".join(failures))


def tensor_batch(batch: list[list[int]], *, device: "torch.device") -> "torch.Tensor":
    return torch.tensor(batch, dtype=torch.long, device=device)


def validate_training_shards(
    *,
    train_shards: list[str],
    sequence_length: int,
    batch_size: int,
    artifact_cache: ArtifactCache | None = None,
    expected_sha256: dict[str, str] | None = None,
) -> int:
    if not train_shards:
        raise ValueError("manifest has no training shards; provide a corpus that produces at least one train shard")
    required_tokens = sequence_length * batch_size
    available_batches = count_batches(
        train_shards,
        sequence_length=sequence_length,
        batch_size=batch_size,
        artifact_cache=artifact_cache,
        expected_sha256=expected_sha256,
    )
    if available_batches < 1:
        total_tokens = sum(
            len((artifact_cache.materialize(path) if artifact_cache else Path(path)).read_bytes()) // 4
            for path in train_shards
        )
        raise ValueError(
            "not enough training tokens for one batch: "
            f"train_tokens={total_tokens}, required_tokens={required_tokens} "
            f"(batch_size={batch_size} * sequence_length={sequence_length}). "
            "Use a larger corpus, reduce --batch-size, or reduce --sequence-length."
        )
    return available_batches


def max_token_id(shard_paths: list[str], *, artifact_cache: ArtifactCache | None = None) -> int:
    maximum = -1
    for path in shard_paths:
        tokens = read_uint32_tokens(artifact_cache.materialize(path) if artifact_cache else path)
        if tokens:
            maximum = max(maximum, max(tokens))
    return maximum


def tokenizer_vocab_size(tokenizer_path: str | Path) -> int:
    return int(load_tokenizer(tokenizer_path).get_vocab_size(with_added_tokens=True))


def build_training_config(
    active_manifest: ShardManifest,
    *,
    sequence_length: int,
    model_profile_name: str = "tiny",
    attention_impl: str = "auto",
    gradient_checkpointing: bool = False,
    artifact_cache: ArtifactCache | None = None,
) -> ScratchDecoderConfig:
    base = model_profile(model_profile_name) if model_profile_name != "tiny" else tiny_smoke_config()
    vocab_size = base.vocab_size
    tokenizer_path = Path(active_manifest.tokenizer_path)
    if tokenizer_path.exists():
        vocab_size = max(256, tokenizer_vocab_size(tokenizer_path))
    highest_token_id = max_token_id(
        active_manifest.train_shards + active_manifest.val_shards,
        artifact_cache=artifact_cache,
    )
    if highest_token_id >= vocab_size:
        vocab_size = highest_token_id + 1
    return base.model_copy(
        update={
            "vocab_size": vocab_size,
            "max_sequence_length": max(base.max_sequence_length, sequence_length),
            "attention_impl": attention_impl,
            "gradient_checkpointing": gradient_checkpointing or base.gradient_checkpointing,
        }
    )


@torch.no_grad() if torch is not None else (lambda fn: fn)
def validation_loss(
    *,
    model: "CraftlyDecoderLM",
    stream: TokenShardStream,
    device: "torch.device",
    max_batches: int,
    dtype: str,
) -> float:
    model.eval()
    losses: list[float] = []
    use_autocast = device.type == "cuda" and dtype in {"bf16", "fp16"}
    try:
        for index, batch in enumerate(stream.batches(epoch=0)):
            if index >= max_batches:
                break
            input_ids = tensor_batch(batch, device=device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype(dtype), enabled=use_autocast):
                output = model(input_ids, labels=input_ids)
            if output.loss is not None:
                losses.append(float(output.loss.detach().cpu()))
    finally:
        model.train()
    local_sum = sum(losses)
    local_count = len(losses)
    if distributed_is_initialized():
        aggregate = torch.tensor([local_sum, float(local_count)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(aggregate, op=torch.distributed.ReduceOp.SUM)
        local_sum, local_count = float(aggregate[0].item()), int(aggregate[1].item())
    if local_count < 1:
        raise RuntimeError("validation stream produced no complete batches")
    mean_loss = local_sum / local_count
    if not math.isfinite(mean_loss):
        raise FloatingPointError("validation stream produced a non-finite loss")
    return mean_loss


def _ewc_dataloader(
    stream: TokenShardStream,
    batch_size: int,
    sequence_length: int,
    device: "torch.device",
) -> Any:
    """Yield (input_ids, labels) tensors for EWC Fisher computation."""
    for batch in stream.batches(epoch=0):
        input_ids = tensor_batch(batch, device=device)
        yield input_ids, input_ids


def run_pretraining_loop(
    *,
    output_dir: str | Path,
    manifest: str | Path | None = None,
    token_file: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    manifest_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
    optimizer_policy_path: str | Path | None = None,
    model_progression_decision_path: str | Path | None = None,
    model_progression_decision_sha256: str | None = None,
    artifact_cache_dir: str | Path | None = None,
    object_store_endpoint_url: str | None = None,
    device: str = "auto",
    steps: int = 100,
    batch_size: int = 2,
    sequence_length: int = 64,
    learning_rate: float = 1e-3,
    optimizer_beta1: float = 0.9,
    optimizer_beta2: float = 0.95,
    optimizer_epsilon: float = 1e-8,
    weight_decay: float = 0.1,
    gradient_clip_norm: float = 1.0,
    learning_rate_schedule: LearningRateSchedule = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    minimum_learning_rate_ratio: float = 0.1,
    target_tokens: int | None = None,
    gradient_accumulation_steps: int = 1,
    dtype: str = "bf16",
    validate_every: int = 25,
    validation_batches: int = 4,
    checkpoint_every: int = 50,
    keep_last_n_checkpoints: int = 3,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    resume: bool = True,
    initial_checkpoint_manifest: str | Path | None = None,
    allow_initial_dataset_rebind: bool = False,
    progress: ProgressReporter | None = None,
    progress_every_steps: int = 25,
    model_profile_name: str = "tiny",
    attention_impl: str = "auto",
    gradient_checkpointing: bool = False,
    distributed_strategy: DistributedStrategy = "none",
    deepspeed_config: str | Path | None = None,
    production_mode: bool = False,
    dev_smoke: bool = False,
    max_training_loss: float = 10_000.0,
    dataloader_prefetch_batches: int = 4,
    dataloader_seed: int = 1337,
    model_seed: int = 1337,
    expected_python_version: str | None = None,
    expected_pytorch_version: str | None = None,
    expected_cuda_version: str | None = None,
    # Elastic Weight Consolidation (EWC) anti-forgetting parameters
    ewc_lambda: float = 0.0,
    ewc_fisher_samples: int = 256,
) -> dict[str, Any]:
    require_torch()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be between 0 and 1")
    if not (0.0 < optimizer_beta1 < 1.0 and 0.0 < optimizer_beta2 < 1.0):
        raise ValueError("optimizer beta values must be between 0 and 1")
    if optimizer_epsilon <= 0.0 or weight_decay < 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("optimizer epsilon/gradient clip must be positive and weight decay non-negative")
    if warmup_steps and warmup_ratio:
        raise ValueError("set either warmup_steps or warmup_ratio, not both")
    resolved_warmup_steps = warmup_steps or int(steps * warmup_ratio)
    if resolved_warmup_steps >= steps:
        raise ValueError("warmup must be shorter than the training run")
    if ewc_lambda < 0.0:
        raise ValueError("ewc_lambda must be non-negative")
    if ewc_lambda > 0.0 and ewc_fisher_samples < 1:
        raise ValueError("ewc_fisher_samples must be >= 1 when EWC is enabled")
    selected = select_torch_device(device)
    runtime_versions = validate_runtime_versions(
        expected_python=expected_python_version,
        expected_pytorch=expected_pytorch_version,
        expected_cuda=expected_cuda_version,
        device=selected,
    )
    distributed_report = initialize_distributed_runtime(distributed_strategy, selected)
    if distributed_report["enabled"] and selected.type == "cuda":
        selected = torch.device(f"cuda:{distributed_report['local_rank']}")
    active_progress = progress or NullProgressReporter()
    if distributed_rank() != 0:
        active_progress = NullProgressReporter()
    optimizer_policy_source: Path | None = None
    optimizer_policy_digest = ""
    if optimizer_policy_path is not None:
        optimizer_policy_source = Path(optimizer_policy_path).expanduser().resolve(strict=True)
        if not optimizer_policy_source.is_file() or optimizer_policy_source.stat().st_size < 1:
            raise ValueError("optimizer policy must be a non-empty regular file")
        optimizer_policy_digest = sha256_file(optimizer_policy_source)
    if bool(model_progression_decision_path) != bool(model_progression_decision_sha256):
        raise ValueError("model progression decision path and SHA-256 must be supplied together")
    progression_source: Path | None = None
    progression_digest = ""
    progression = None
    if model_progression_decision_path is not None:
        progression_source = Path(model_progression_decision_path).expanduser().resolve(strict=True)
        progression_digest = sha256_file(progression_source)
        if not hmac.compare_digest(progression_digest, str(model_progression_decision_sha256)):
            raise ValueError("model progression decision SHA-256 mismatch")
        from src.craftly.learning.ablation_runner import verify_model_progression_decision

        progression = verify_model_progression_decision(progression_source)
        if progression.status != "authorized" or progression.target_model_profile != model_profile_name:
            raise ValueError("model progression decision does not authorize this model profile")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_cache = ArtifactCache(
        artifact_cache_dir or os.environ.get("CRAFTLY_ARTIFACT_CACHE_DIR", root / "artifact-cache"),
        s3_endpoint_url=object_store_endpoint_url or os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL"),
    )
    active_manifest_path: Path | None = (
        artifact_cache.materialize(manifest, expected_sha256=manifest_sha256) if manifest else None
    )
    if active_manifest_path is None and token_file and tokenizer_path:
        active_manifest = build_token_shards(
            input_paths=[token_file],
            tokenizer_path=tokenizer_path,
            output_dir=root / "generated-shards",
            config=ShardBuildConfig(
                shard_token_count=max(128, batch_size * sequence_length * 8),
                sequence_length=sequence_length,
                validation_fraction=0.1,
            ),
        )
        active_manifest_path = (root / "generated-shards" / "manifest.json").resolve()
    elif active_manifest_path is not None:
        active_manifest = load_manifest(active_manifest_path)
    else:
        raise ValueError("provide --manifest, or both --token-file and --tokenizer-path")
    immutable_tokenizer = tokenizer_path or active_manifest.tokenizer_path
    local_tokenizer = artifact_cache.materialize(immutable_tokenizer, expected_sha256=tokenizer_sha256)
    active_manifest = active_manifest.model_copy(update={"tokenizer_path": str(local_tokenizer)})
    validate_production_training_args(
        production_mode=production_mode,
        dev_smoke=dev_smoke,
        model_profile_name=model_profile_name,
        manifest=manifest,
        tokenizer_path=tokenizer_path,
        active_manifest=active_manifest,
        validate_every=validate_every,
        checkpoint_every=checkpoint_every,
        run_steps=steps,
    )

    config = build_training_config(
        active_manifest,
        sequence_length=sequence_length,
        model_profile_name=model_profile_name,
        attention_impl=attention_impl,
        gradient_checkpointing=gradient_checkpointing,
        artifact_cache=artifact_cache,
    )
    if progression is not None and not hmac.compare_digest(
        config.contract_sha256(),
        str(progression.target_model_contract_sha256 or ""),
    ):
        raise ValueError("runtime model contract differs from the authorized progression target")
    available_batches = validate_training_shards(
        train_shards=active_manifest.train_shards,
        sequence_length=sequence_length,
        batch_size=batch_size,
        artifact_cache=artifact_cache,
        expected_sha256=active_manifest.shard_sha256,
    )
    data_parallel_rank = distributed_rank()
    data_parallel_world_size = distributed_world_size()
    available_batches_per_rank = available_batches // data_parallel_world_size
    if available_batches_per_rank < 1:
        raise ValueError(
            "training corpus cannot provide one non-overlapping batch per data-parallel rank: "
            f"global_batches={available_batches}, world_size={data_parallel_world_size}"
        )
    available_validation_batches = 0
    if active_manifest.val_shards:
        available_validation_batches = count_batches(
            active_manifest.val_shards,
            sequence_length=sequence_length,
            batch_size=batch_size,
            artifact_cache=artifact_cache,
            expected_sha256=active_manifest.shard_sha256,
        )
        if validate_every > 0 and available_validation_batches < data_parallel_world_size:
            raise ValueError(
                "validation shards cannot provide one non-overlapping batch per data-parallel rank: "
                f"validation_tokens={active_manifest.val_tokens}, "
                f"required_tokens={batch_size * sequence_length * data_parallel_world_size}"
            )
    batch_contract = TrainingBatchContract(
        optimizer_steps=steps,
        sequence_length=sequence_length,
        micro_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_size=data_parallel_world_size,
    )
    tokens_per_optimizer_step = batch_contract.tokens_per_optimizer_step
    expected_target_tokens = batch_contract.target_tokens
    if target_tokens is not None and target_tokens != expected_target_tokens:
        raise ValueError(
            "target_tokens must equal optimizer steps * micro batch * sequence length * "
            "gradient accumulation * data-parallel world size: "
            f"target_tokens={target_tokens}, expected={expected_target_tokens}"
        )
    active_progress.emit(
        "training",
        "started",
        device=str(selected),
        dtype=dtype,
        requested_steps=steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_shards=len(active_manifest.train_shards),
        val_shards=len(active_manifest.val_shards),
        available_batches=available_batches,
        available_batches_per_rank=available_batches_per_rank,
        available_validation_batches=available_validation_batches,
        tokens_per_optimizer_step=tokens_per_optimizer_step,
        vocab_size=config.vocab_size,
        parameter_count=config.parameter_estimate(),
    )
    if not 0 <= model_seed <= 2**31 - 1:
        raise ValueError("model_seed must be between 0 and 2**31-1")
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    train_stream = TokenShardStream(
        active_manifest.train_shards,
        sequence_length=sequence_length,
        batch_size=batch_size,
        seed=dataloader_seed,
        shuffle=True,
        prefetch_batches=dataloader_prefetch_batches,
        artifact_cache=artifact_cache,
        expected_sha256=active_manifest.shard_sha256,
        rank=data_parallel_rank,
        world_size=data_parallel_world_size,
    )
    val_stream = (
        TokenShardStream(
            active_manifest.val_shards,
            sequence_length=sequence_length,
            batch_size=batch_size,
            seed=dataloader_seed + 5994,
            shuffle=False,
            prefetch_batches=dataloader_prefetch_batches,
            artifact_cache=artifact_cache,
            expected_sha256=active_manifest.shard_sha256,
            rank=data_parallel_rank,
            world_size=data_parallel_world_size,
        )
        if active_manifest.val_shards
        else None
    )

    # Every rank must initialize identical parameters before distributed
    # wrapping. Runtime stochastic operations receive a rank-specific stream
    # only after initialization, while checkpoint resume restores exact RNG
    # state from the selected checkpoint.
    model = CraftlyDecoderLM(config).to(selected)
    runtime_seed = model_seed + distributed_rank() * 1_000_003
    random.seed(runtime_seed)
    torch.manual_seed(runtime_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime_seed)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if not is_deepspeed_strategy(distributed_strategy):
        model = wrap_for_distributed(model, strategy=distributed_strategy, dtype=dtype, device=selected)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(optimizer_beta1, optimizer_beta2),
        eps=optimizer_epsilon,
        weight_decay=weight_decay,
    )
    scheduler = build_learning_rate_scheduler(
        optimizer,
        total_steps=steps,
        warmup_steps=resolved_warmup_steps,
        schedule=learning_rate_schedule,
        minimum_learning_rate_ratio=minimum_learning_rate_ratio,
    )
    if is_deepspeed_strategy(distributed_strategy):
        model, optimizer, scheduler = wrap_for_deepspeed(
            model,
            optimizer,
            scheduler,
            strategy=distributed_strategy,
            config_path=deepspeed_config,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dtype=dtype,
        )
    checkpoint_environment = training_environment_report(device=selected, dtype=dtype, distributed_strategy=distributed_strategy)
    checkpoint_args = {
        "objective": "causal_language_modeling",
        "optimizer_policy_path": str(optimizer_policy_source or ""),
        "optimizer_policy_sha256": optimizer_policy_digest,
        "model_progression_decision_path": str(progression_source or ""),
        "model_progression_decision_sha256": progression_digest,
        "step_semantics": TRAINING_STEP_SEMANTICS,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "optimizer_beta1": optimizer_beta1,
        "optimizer_beta2": optimizer_beta2,
        "optimizer_epsilon": optimizer_epsilon,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "learning_rate_schedule": learning_rate_schedule,
        "warmup_steps": resolved_warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "target_tokens": target_tokens,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "dtype": dtype,
        "validate_every": validate_every,
        "validation_batches": validation_batches,
        "checkpoint_every": checkpoint_every,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "model_profile_name": model_profile_name,
        "attention_impl": attention_impl,
        "gradient_checkpointing": gradient_checkpointing,
        "distributed_strategy": distributed_strategy,
        "deepspeed_config": str(deepspeed_config or default_deepspeed_config_path(distributed_strategy)) if is_deepspeed_strategy(distributed_strategy) else "",
        "production_mode": production_mode,
        "dev_smoke": dev_smoke,
        "max_training_loss": max_training_loss,
        "dataloader_prefetch_batches": dataloader_prefetch_batches,
        "dataloader_seed": dataloader_seed,
        "model_seed": model_seed,
        "runtime_seed": runtime_seed,
        "distributed_rank": distributed_rank(),
        "data_parallel_world_size": data_parallel_world_size,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "batches_per_rank_per_epoch": available_batches_per_rank,
        "runtime_versions": runtime_versions,
        "environment": checkpoint_environment,
        "ewc_lambda": ewc_lambda,
        "ewc_fisher_samples": ewc_fisher_samples,
    }
    checkpoint_training_invariants = {
        name: checkpoint_args[name]
        for name in (
            "step_semantics",
            "batch_size",
            "sequence_length",
            "gradient_accumulation_steps",
            "dataloader_seed",
            "model_seed",
            "runtime_seed",
            "distributed_rank",
            "data_parallel_world_size",
            "tokens_per_optimizer_step",
            "batches_per_rank_per_epoch",
        )
    }
    start_step = 0
    trained_tokens = 0
    resume_source = ""
    if resume:
        checkpoint = latest_checkpoint(root)
        if checkpoint is not None:
            start_step, trained_tokens = load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=selected,
                expected_config=config,
                expected_dataset_manifest_sha256=sha256_file(active_manifest_path) if active_manifest_path else None,
                expected_tokenizer_sha256=sha256_file(local_tokenizer),
                expected_training_invariants=checkpoint_training_invariants,
            )
            resume_source = str(checkpoint)
        elif initial_checkpoint_manifest:
            initial_checkpoint = checkpoint_payload_from_manifest(initial_checkpoint_manifest)
            start_step, trained_tokens = load_checkpoint(
                initial_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=selected,
                expected_config=config,
                expected_dataset_manifest_sha256=(
                    None
                    if allow_initial_dataset_rebind
                    else sha256_file(active_manifest_path) if active_manifest_path else None
                ),
                expected_tokenizer_sha256=sha256_file(local_tokenizer),
                expected_training_invariants=checkpoint_training_invariants,
            )
            resume_source = str(initial_checkpoint)
    elif initial_checkpoint_manifest:
        raise ValueError("initial_checkpoint_manifest requires resume=True")
    if start_step >= steps:
        raise ValueError(
            f"requested final step {steps} must be greater than resumed checkpoint step {start_step}"
        )

    # Initialize EWC (Elastic Weight Consolidation) if enabled
    ewc: ElasticWeightConsolidation | None = None
    if ewc_lambda > 0.0:
        ewc = ElasticWeightConsolidation(model, lambda_=ewc_lambda)

    model.train()
    started = time.perf_counter()
    train_losses: list[float] = []
    val_losses: list[dict[str, float]] = []
    mtp_losses: list[float] = []
    router_load_ratios: list[float] = []
    router_dropped_assignments = 0
    use_autocast = selected.type == "cuda" and dtype in {"bf16", "fp16"}
    optimizer.zero_grad(set_to_none=True)
    current_step = start_step
    expected_resumed_tokens = start_step * tokens_per_optimizer_step
    if trained_tokens != expected_resumed_tokens:
        raise ValueError(
            "checkpoint token cursor is incompatible with optimizer-step semantics: "
            f"trained_tokens={trained_tokens}, expected={expected_resumed_tokens}"
        )
    completed_micro_batches_per_rank = start_step * gradient_accumulation_steps
    epoch = completed_micro_batches_per_rank // available_batches_per_rank
    epoch_batch_offset = completed_micro_batches_per_rank % available_batches_per_rank
    train_iterator = iter(train_stream.batches(epoch=epoch, start_batch=epoch_batch_offset))

    def next_training_batch() -> list[list[int]]:
        nonlocal epoch, train_iterator
        try:
            return next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = iter(train_stream.batches(epoch=epoch))
            try:
                return next(train_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "rank-partitioned training stream produced no batches after preflight validation"
                ) from exc

    def current_dataloader_state(step: int) -> dict[str, int | str]:
        completed = step * gradient_accumulation_steps
        return {
            "semantics": TRAINING_STEP_SEMANTICS,
            "completed_optimizer_steps": step,
            "completed_micro_batches_per_rank": completed,
            "epoch": completed // available_batches_per_rank,
            "batch_offset_within_epoch": completed % available_batches_per_rank,
            "batches_per_rank_per_epoch": available_batches_per_rank,
            "rank": data_parallel_rank,
            "world_size": data_parallel_world_size,
        }

    best_val_loss = float("inf")
    best_val_step = 0
    best_checkpoint_manifest: Path | None = None
    validations_without_improvement = 0
    early_stopped = False
    early_stop_reason = ""

    # Compute EWC Fisher information on initial data before training starts
    if ewc is not None and start_step == 0:
        ewc.compute_fisher(
            _ewc_dataloader(train_stream, batch_size, sequence_length, selected),
            max_samples=ewc_fisher_samples,
            device=str(selected),
        )
        active_progress.emit(
            "ewc",
            "fisher_computed",
            lambda_=ewc_lambda,
            fisher_samples=ewc_fisher_samples,
        )

    while current_step < steps:
        optimizer_step_losses: list[float] = []
        optimizer_step_mtp_losses: list[float] = []
        for accumulation_index in range(gradient_accumulation_steps):
            batch = next_training_batch()
            input_ids = tensor_batch(batch, device=selected)
            synchronize = accumulation_index == gradient_accumulation_steps - 1
            sync_context = (
                model.no_sync()
                if not synchronize
                and not is_deepspeed_strategy(distributed_strategy)
                and hasattr(model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type=selected.type, dtype=autocast_dtype(dtype), enabled=use_autocast):
                    output = model(input_ids, labels=input_ids)
                    if output.loss is None:
                        raise RuntimeError("loss missing")
                    observed_loss = float(output.loss.detach().cpu())
                    if not math.isfinite(observed_loss):
                        raise FloatingPointError(
                            f"non-finite training loss at optimizer step {current_step + 1}, "
                            f"microbatch {accumulation_index + 1}: {observed_loss}"
                        )
                    if observed_loss > max_training_loss:
                        raise FloatingPointError(
                            f"catastrophic training loss at optimizer step {current_step + 1}: "
                            f"{observed_loss} > {max_training_loss}"
                        )
                    optimizer_step_losses.append(observed_loss)
                    if output.mtp_loss is not None:
                        observed_mtp_loss = float(output.mtp_loss.detach().cpu())
                        optimizer_step_mtp_losses.append(observed_mtp_loss)
                        mtp_losses.append(observed_mtp_loss)
                    for router_metric in output.router_metrics:
                        dropped = int(router_metric.get("dropped_tokens", 0.0))
                        router_dropped_assignments += dropped
                        router_load_ratios.append(float(router_metric["p99_to_mean_load"]))
                    if router_dropped_assignments:
                        raise RuntimeError("dropless MoE invariant failed: one or more token assignments were dropped")
                    # Add EWC penalty to loss if enabled
                    ewc_penalty = ewc.penalty() if ewc is not None else torch.tensor(0.0)
                    total_loss = output.loss + ewc_penalty
                    backward_loss = (
                        total_loss
                        if is_deepspeed_strategy(distributed_strategy)
                        else total_loss / gradient_accumulation_steps
                    )
                if is_deepspeed_strategy(distributed_strategy):
                    model.backward(backward_loss)
                else:
                    backward_loss.backward()
            if is_deepspeed_strategy(distributed_strategy):
                model.step()
            trained_tokens += batch_size * sequence_length * data_parallel_world_size
            if target_tokens is not None and trained_tokens > target_tokens:
                raise RuntimeError(
                    f"training token accounting exceeded immutable target: {trained_tokens} > {target_tokens}"
                )
        if is_deepspeed_strategy(distributed_strategy):
            grad_norm = torch.tensor(0.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        current_step += 1
        train_losses.append(sum(optimizer_step_losses) / len(optimizer_step_losses))
        if current_step == 1 or current_step % max(1, progress_every_steps) == 0 or current_step >= steps:
            active_progress.emit(
                "training",
                "running",
                step=current_step,
                requested_steps=steps,
                step_semantics=TRAINING_STEP_SEMANTICS,
                completed_micro_batches_per_rank=current_step * gradient_accumulation_steps,
                loss=round(train_losses[-1], 6),
                grad_norm=round(float(grad_norm.detach().cpu()), 6),
                trained_tokens=trained_tokens,
                epoch=epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                mtp_loss=(
                    round(sum(optimizer_step_mtp_losses) / len(optimizer_step_mtp_losses), 6)
                    if optimizer_step_mtp_losses
                    else None
                ),
                router_p99_to_mean=round(router_load_ratios[-1], 6) if router_load_ratios else None,
            )

        if val_stream is not None and validate_every > 0 and current_step % validate_every == 0:
            current_val_loss = validation_loss(
                model=model,
                stream=val_stream,
                device=selected,
                max_batches=validation_batches,
                dtype=dtype,
            )
            val_losses.append({"step": float(current_step), "loss": current_val_loss})
            active_progress.emit(
                "validation",
                "complete",
                step=current_step,
                validation_loss=round(current_val_loss, 6),
                best_validation_loss=round(best_val_loss, 6) if best_val_loss != float("inf") else None,
            )
            if current_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = current_val_loss
                best_val_step = current_step
                validations_without_improvement = 0
                best_checkpoint_manifest = save_training_checkpoint(
                    output_dir=root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    step=current_step,
                    trained_tokens=trained_tokens,
                    metrics={"train_loss": train_losses[-1], "val_loss": current_val_loss, "best_val_loss": current_val_loss},
                    training_args=checkpoint_args,
                    dataset_manifest_path=active_manifest_path,
                    tokenizer_path=active_manifest.tokenizer_path,
                    environment=checkpoint_environment,
                    dataloader_state=current_dataloader_state(current_step),
                    manifest_filename="best_checkpoint_manifest.json",
                    keep_last_n_checkpoints=keep_last_n_checkpoints,
                )
                active_progress.emit(
                    "checkpoint",
                    "best_saved",
                    step=current_step,
                    validation_loss=round(current_val_loss, 6),
                    checkpoint_manifest=str(best_checkpoint_manifest),
                )
            else:
                validations_without_improvement += 1
                if early_stopping_patience > 0 and validations_without_improvement >= early_stopping_patience:
                    early_stopped = True
                    early_stop_reason = (
                        f"validation loss did not improve by {early_stopping_min_delta} "
                        f"for {validations_without_improvement} validation checks"
                    )
                    active_progress.emit("training", "early_stopping", step=current_step, reason=early_stop_reason)
        if early_stopped:
            break
        if checkpoint_every > 0 and current_step % checkpoint_every == 0:
            checkpoint_manifest = save_training_checkpoint(
                output_dir=root,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                step=current_step,
                trained_tokens=trained_tokens,
                metrics={"train_loss": train_losses[-1], "val_loss": val_losses[-1]["loss"] if val_losses else -1.0},
                training_args=checkpoint_args,
                dataset_manifest_path=active_manifest_path,
                tokenizer_path=active_manifest.tokenizer_path,
                environment=checkpoint_environment,
                dataloader_state=current_dataloader_state(current_step),
                keep_last_n_checkpoints=keep_last_n_checkpoints,
            )
            active_progress.emit(
                "checkpoint",
                "saved",
                step=current_step,
                checkpoint_manifest=str(checkpoint_manifest),
                train_loss=round(train_losses[-1], 6),
                validation_loss=round(val_losses[-1]["loss"], 6) if val_losses else None,
            )

    expected_completed_tokens = current_step * tokens_per_optimizer_step
    if trained_tokens != expected_completed_tokens:
        raise RuntimeError(
            "optimizer-step token accounting invariant failed: "
            f"trained_tokens={trained_tokens}, expected={expected_completed_tokens}"
        )
    final_val_loss = val_losses[-1]["loss"] if val_losses else -1.0
    manifest_path = save_training_checkpoint(
        output_dir=root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=current_step,
        trained_tokens=trained_tokens,
        metrics={"train_loss": train_losses[-1], "val_loss": final_val_loss, "best_val_loss": best_val_loss if val_losses else -1.0},
        training_args=checkpoint_args,
        dataset_manifest_path=active_manifest_path,
        tokenizer_path=active_manifest.tokenizer_path,
        environment=checkpoint_environment,
        dataloader_state=current_dataloader_state(current_step),
        manifest_filename="checkpoint_manifest.json",
    )
    if best_checkpoint_manifest is None:
        best_checkpoint_manifest = save_training_checkpoint(
            output_dir=root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=current_step,
            trained_tokens=trained_tokens,
            metrics={"train_loss": train_losses[-1], "val_loss": final_val_loss, "best_val_loss": final_val_loss},
            training_args=checkpoint_args,
            dataset_manifest_path=active_manifest_path,
            tokenizer_path=active_manifest.tokenizer_path,
            environment=checkpoint_environment,
            dataloader_state=current_dataloader_state(current_step),
            manifest_filename="best_checkpoint_manifest.json",
        )
        best_val_loss = final_val_loss
        best_val_step = current_step
    pre_reload_logits = checkpoint_logit_probe(model, config=config, device=selected)
    reload_source = latest_checkpoint(root)
    if reload_source is None:
        raise RuntimeError("final checkpoint was not discoverable for reload verification")
    reloaded_step, reloaded_tokens = load_checkpoint(
        reload_source,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=selected,
        expected_config=config,
        expected_dataset_manifest_sha256=sha256_file(active_manifest_path) if active_manifest_path else None,
        expected_tokenizer_sha256=sha256_file(local_tokenizer),
        expected_training_invariants=checkpoint_training_invariants,
    )
    if reloaded_step != current_step or reloaded_tokens != trained_tokens:
        raise RuntimeError(
            "checkpoint reload state mismatch: "
            f"step={reloaded_step}/{current_step}, tokens={reloaded_tokens}/{trained_tokens}"
        )
    post_reload_logits = checkpoint_logit_probe(model, config=config, device=selected)
    reload_max_abs_logit_difference = float(
        torch.max(torch.abs(pre_reload_logits - post_reload_logits)).item()
    )
    reload_tolerance = 1e-6 if dtype == "fp32" else 5e-3
    checkpoint_reload_logit_parity = (
        math.isfinite(reload_max_abs_logit_difference)
        and reload_max_abs_logit_difference <= reload_tolerance
    )
    if not checkpoint_reload_logit_parity:
        raise RuntimeError(
            "checkpoint reload logit parity failed: "
            f"max_abs_difference={reload_max_abs_logit_difference}, tolerance={reload_tolerance}"
        )
    distributed_barrier()
    checkpoint_publication = None
    if distributed_rank() == 0:
        checkpoint_publication = publish_checkpoint_to_workspace(
            manifest_path=manifest_path,
            attempt_id=os.environ.get("CRAFTLY_TRAINING_ATTEMPT_ID", ""),
            dataset_sha256=sha256_file(active_manifest_path) if active_manifest_path else "",
            tokenizer_sha256=sha256_file(local_tokenizer),
            topology=distributed_report,
            metrics={"train_loss": train_losses[-1], "validation_loss": final_val_loss},
            reload_verified=True,
        )
    report = {
        "status": "early_stopped" if early_stopped else "passed",
        "scratch_only": True,
        "objective": "causal_language_modeling",
        "steps": current_step,
        "step_semantics": TRAINING_STEP_SEMANTICS,
        "optimizer_steps": current_step,
        "completed_micro_batches_per_rank": current_step * gradient_accumulation_steps,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "available_batches_per_rank": available_batches_per_rank,
        "dataloader_state": current_dataloader_state(current_step),
        "requested_steps": steps,
        "start_step": start_step,
        "resume_source": resume_source,
        "initial_checkpoint_manifest": str(initial_checkpoint_manifest or ""),
        "initial_dataset_rebound": bool(initial_checkpoint_manifest and allow_initial_dataset_rebind),
        "device": str(selected),
        "dtype": dtype,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "validate_every": validate_every,
        "validation_batches": validation_batches,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "model_config": config.model_dump(),
        "model_profile_name": model_profile_name,
        "attention_impl": attention_impl,
        "distributed_strategy": distributed_strategy,
        "distributed": distributed_report,
        "deepspeed_config": str(deepspeed_config or default_deepspeed_config_path(distributed_strategy)) if is_deepspeed_strategy(distributed_strategy) else "",
        "production_mode": production_mode,
        "dev_smoke": dev_smoke,
        "max_training_loss": max_training_loss,
        "git_commit": git_commit(),
        "dataset_manifest_path": str(active_manifest_path or ""),
        "dataset_manifest_sha256": (
            sha256_file(active_manifest_path) if active_manifest_path else ""
        ),
        "tokenizer_path": str(local_tokenizer),
        "tokenizer_sha256": sha256_file(local_tokenizer),
        "shard_manifest_path": str(active_manifest_path or ""),
        "shard_manifest_sha256": (
            sha256_file(active_manifest_path) if active_manifest_path else ""
        ),
        "train_losses": train_losses,
        "mtp_losses": mtp_losses,
        "router_metrics": {
            "sample_count": len(router_load_ratios),
            "dropped_assignments": router_dropped_assignments,
            "maximum_p99_to_mean_load": max(router_load_ratios, default=0.0),
            "configured_p99_to_mean_limit": config.router_load_limit,
            "load_balance_passed": (
                not router_load_ratios
                or max(router_load_ratios) <= config.router_load_limit
            ),
        },
        "validation_losses": val_losses,
        "best_validation_loss": best_val_loss,
        "best_validation_step": best_val_step,
        "best_checkpoint_manifest": str(best_checkpoint_manifest),
        "trained_tokens": trained_tokens,
        "target_tokens": target_tokens,
        "runtime_versions": runtime_versions,
        "training_args": checkpoint_args,
        "optimizer_policy_path": str(optimizer_policy_source or ""),
        "optimizer_policy_sha256": optimizer_policy_digest,
        "model_progression_decision_path": str(progression_source or ""),
        "model_progression_decision_sha256": progression_digest,
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "best_checkpoint_manifest_sha256": sha256_file(best_checkpoint_manifest),
        "checkpoint_reload_verified": True,
        "checkpoint_reload_logit_parity": checkpoint_reload_logit_parity,
        "checkpoint_reload_max_abs_logit_difference": reload_max_abs_logit_difference,
        "checkpoint_reload_tolerance": reload_tolerance,
        "checkpoint_publication": checkpoint_publication,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if distributed_rank() == 0:
        (root / "pretrain_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    active_progress.emit(
        "training",
        "complete",
        training_status=report["status"],
        steps=current_step,
        trained_tokens=trained_tokens,
        final_loss=round(train_losses[-1], 6),
        best_validation_loss=round(best_val_loss, 6) if best_val_loss != float("inf") else None,
        checkpoint_manifest=str(manifest_path),
        duration_ms=report["duration_ms"],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly scratch pretraining loop.")
    parser.add_argument("--output-dir", default="artifacts/craftly/pretrain-loop")
    parser.add_argument("--manifest")
    parser.add_argument("--token-file")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--optimizer-policy")
    parser.add_argument("--model-progression-decision")
    parser.add_argument("--model-progression-decision-sha256")
    parser.add_argument("--artifact-cache-dir")
    parser.add_argument("--object-store-endpoint-url")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.95)
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate-schedule", choices=["constant", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--initial-checkpoint-manifest")
    parser.add_argument("--allow-initial-dataset-rebind", action="store_true")
    parser.add_argument(
        "--model-profile",
        default="tiny",
        choices=sorted(name for name in model_profiles() if name != "4t_moe"),
    )
    parser.add_argument("--attention-impl", default="auto", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--production", action="store_true", help="Enable strict production training validation.")
    parser.add_argument("--dev-smoke", action="store_true", help="Explicitly allow tiny/dev smoke behavior under production validation.")
    parser.add_argument("--max-training-loss", type=float, default=10_000.0)
    parser.add_argument("--dataloader-prefetch-batches", type=int, default=4)
    parser.add_argument("--dataloader-seed", type=int, default=1337)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--expected-python-version")
    parser.add_argument("--expected-pytorch-version")
    parser.add_argument("--expected-cuda-version")
    parser.add_argument(
        "--distributed-strategy",
        default="none",
        choices=["none", "fsdp", "deepspeed_zero2", "deepspeed_zero3", "megatron"],
        help="Validated distributed strategy contract. Use --cluster-plan-only before cluster execution.",
    )
    parser.add_argument("--cluster-plan-only", action="store_true", help="Write/print a distributed training launch plan without training.")
    parser.add_argument("--cluster-plan-out", default="artifacts/craftly/cluster_training_plan.json")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--megatron-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cluster_plan_only:
        if not args.manifest:
            raise SystemExit("--manifest is required for --cluster-plan-only")
        plan = build_cluster_training_plan(
            output_dir=args.output_dir,
            manifest=args.manifest,
            model_profile_name=args.model_profile,
            strategy=args.distributed_strategy,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            steps=args.steps,
            dtype=args.dtype,
            attention_impl=args.attention_impl,
            gradient_checkpointing=args.gradient_checkpointing,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            deepspeed_config=args.deepspeed_config,
            megatron_root=args.megatron_root,
        )
        write_cluster_training_plan(output_path=args.cluster_plan_out, plan=plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    report = run_pretraining_loop(
        output_dir=args.output_dir,
        manifest=args.manifest,
        token_file=args.token_file,
        tokenizer_path=args.tokenizer_path,
        manifest_sha256=args.manifest_sha256,
        tokenizer_sha256=args.tokenizer_sha256,
        optimizer_policy_path=args.optimizer_policy,
        model_progression_decision_path=args.model_progression_decision,
        model_progression_decision_sha256=args.model_progression_decision_sha256,
        artifact_cache_dir=args.artifact_cache_dir,
        object_store_endpoint_url=args.object_store_endpoint_url,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_epsilon=args.optimizer_epsilon,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        target_tokens=args.target_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dtype=args.dtype,
        validate_every=args.validate_every,
        validation_batches=args.validation_batches,
        checkpoint_every=args.checkpoint_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        resume=not args.no_resume,
        initial_checkpoint_manifest=args.initial_checkpoint_manifest,
        allow_initial_dataset_rebind=args.allow_initial_dataset_rebind,
        model_profile_name=args.model_profile,
        attention_impl=args.attention_impl,
        gradient_checkpointing=args.gradient_checkpointing,
        distributed_strategy=args.distributed_strategy,
        deepspeed_config=args.deepspeed_config,
        production_mode=args.production,
        dev_smoke=args.dev_smoke,
        max_training_loss=args.max_training_loss,
        dataloader_prefetch_batches=args.dataloader_prefetch_batches,
        dataloader_seed=args.dataloader_seed,
        model_seed=args.model_seed,
        expected_python_version=args.expected_python_version,
        expected_pytorch_version=args.expected_pytorch_version,
        expected_cuda_version=args.expected_cuda_version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

