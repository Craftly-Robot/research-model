"""Production adapter exports for Craftly scratch checkpoints.

This module owns the bridge from the native Craftly checkpoint format to external
large-model runtimes. It intentionally separates "artifact generated" from
"runtime proven": vLLM/TensorRT/Megatron still require their external packages,
GPU hardware, and release gates before production promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.model_ops.foundation import (
    CheckpointManifest,
    sha256_file,
)
from src.craftly.model_ops.foundation import (
    model_profile as get_model_profile,
)
from src.craftly.model_ops.torch_decoder import (
    ScratchDecoderConfig,
    load_trusted_checkpoint,
    require_torch,
)
from src.craftly.shared.schemas import StrictModel, write_report, print_report

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class AdapterReport(StrictModel):
    status: str
    adapter: str
    output_dir: str
    artifacts: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    missing_dependencies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path, name: str) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        write_report(self, target)
        return target


def resolve_megatron_data_prefix(manifest: str | Path) -> Path:
    """Resolve and validate a Megatron indexed-dataset prefix from a manifest.

    The prefix is constrained to the immutable manifest directory. Both
    ``<prefix>.bin`` and ``<prefix>.idx`` must exist, preventing a launch plan
    from silently passing a JSON manifest where Megatron expects indexed data.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Megatron dataset manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Megatron dataset manifest is invalid JSON") from exc
    prefix_value = payload.get("megatron_data_prefix")
    if not isinstance(prefix_value, str) or not prefix_value.strip():
        raise ValueError("Megatron dataset manifest must define megatron_data_prefix")
    raw_prefix = Path(prefix_value).expanduser()
    prefix = (
        raw_prefix.resolve()
        if raw_prefix.is_absolute()
        else (manifest_path.parent / raw_prefix).resolve()
    )
    allowed_root = manifest_path.parent.resolve()
    if prefix != allowed_root and allowed_root not in prefix.parents:
        raise ValueError("Megatron data prefix must remain inside the immutable dataset directory")
    if not prefix.with_suffix(".bin").is_file() or not prefix.with_suffix(".idx").is_file():
        raise FileNotFoundError("Megatron data prefix must resolve to existing .bin and .idx files")
    return prefix


def megatron_model_arguments(profile: ScratchDecoderConfig) -> list[str]:
    """Translate one canonical model contract into Megatron-Core argv."""
    arguments = [
        "--num-layers",
        str(profile.num_layers),
        "--hidden-size",
        str(profile.hidden_size),
        "--ffn-hidden-size",
        str(profile.intermediate_size),
        "--num-attention-heads",
        str(profile.num_attention_heads),
        "--num-query-groups",
        str(profile.num_key_value_heads),
        "--vocab-size",
        str(profile.vocab_size),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(profile.norm_eps),
        "--swiglu",
        "--disable-bias-linear",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        str(profile.rope_theta),
        "--rotary-percent",
        "1.0",
    ]
    if profile.rope_scaling_factor != 1.0:
        arguments.extend(["--rotary-scaling-factor", str(profile.rope_scaling_factor)])
    if not profile.tie_word_embeddings:
        arguments.append("--untie-embeddings-and-output-weights")
    if profile.attention_architecture == "mla":
        arguments.extend(
            [
                "--multi-latent-attention",
                "--qk-layernorm",
                "--q-lora-rank",
                str(profile.q_lora_rank),
                "--kv-lora-rank",
                str(profile.kv_lora_rank),
                "--qk-head-dim",
                str(profile.qk_nope_head_dim),
                "--qk-pos-emb-head-dim",
                str(profile.qk_rope_head_dim),
                "--v-head-dim",
                str(profile.v_head_dim),
            ]
        )
    if profile.feed_forward_architecture == "moe":
        arguments.extend(
            [
                "--num-experts",
                str(profile.num_routed_experts),
                "--moe-router-topk",
                str(profile.experts_per_token),
                "--moe-layer-freq",
                f"([0]*{profile.dense_layer_count}+[1]*{profile.moe_layer_count})",
                "--moe-ffn-hidden-size",
                str(profile.moe_intermediate_size),
                "--moe-shared-expert-intermediate-size",
                str(profile.num_shared_experts * int(profile.moe_intermediate_size or 0)),
                "--moe-grouped-gemm",
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-router-load-balancing-type",
                "none",
                "--moe-router-score-function",
                "sigmoid",
                "--moe-aux-loss-coeff",
                "0.0",
                "--moe-router-enable-expert-bias",
                "--moe-router-bias-update-rate",
                str(profile.router_bias_update_rate),
            ]
        )
    if profile.mtp_num_layers:
        arguments.extend(
            [
                "--mtp-num-layers",
                str(profile.mtp_num_layers),
                "--mtp-loss-scaling-factor",
                str(profile.mtp_loss_weight),
            ]
        )
    return arguments


def _load_checkpoint(
    manifest_path: str | Path,
) -> tuple[CheckpointManifest, dict[str, Any], ScratchDecoderConfig]:
    require_torch()
    manifest = CheckpointManifest.model_validate(
        json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    )
    checkpoint_path = Path(manifest.checkpoint_dir) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint model file not found: {checkpoint_path}")
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    config = ScratchDecoderConfig.model_validate(payload["config"])
    return manifest, payload, config


def _hf_llama_config(config: ScratchDecoderConfig, *, torch_dtype: str) -> dict[str, Any]:
    if config.attention_architecture != "gqa" or config.feed_forward_architecture != "dense":
        raise ValueError(
            "HF Llama export supports only canonical dense/GQA checkpoints; "
            "MLA/MoE requires a native custom converter with fixed-token logit parity"
        )
    payload = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_sequence_length,
        "initializer_range": config.initializer_range,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": torch_dtype,
        "use_cache": config.use_cache,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    if config.rope_scaling_factor > 1.0:
        payload["rope_scaling"] = {"type": "linear", "factor": config.rope_scaling_factor}
    return payload


def _convert_state_dict_to_hf_llama(
    state: dict[str, Any], config: ScratchDecoderConfig
) -> dict[str, Any]:
    if config.attention_architecture != "gqa" or config.feed_forward_architecture != "dense":
        raise ValueError("MLA/MoE state cannot be represented by the Llama checkpoint schema")
    converted: dict[str, Any] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"],
        "model.norm.weight": state["norm.weight"],
    }
    if "lm_head.weight" in state:
        converted["lm_head.weight"] = state["lm_head.weight"].clone()
    elif config.tie_word_embeddings:
        converted["lm_head.weight"] = state["embed_tokens.weight"].clone()
    for layer_index in range(config.num_layers):
        source = f"layers.{layer_index}"
        target = f"model.layers.{layer_index}"
        mapping = {
            f"{source}.input_norm.weight": f"{target}.input_layernorm.weight",
            f"{source}.post_attention_norm.weight": f"{target}.post_attention_layernorm.weight",
            f"{source}.attention.q_proj.weight": f"{target}.self_attn.q_proj.weight",
            f"{source}.attention.k_proj.weight": f"{target}.self_attn.k_proj.weight",
            f"{source}.attention.v_proj.weight": f"{target}.self_attn.v_proj.weight",
            f"{source}.attention.o_proj.weight": f"{target}.self_attn.o_proj.weight",
            f"{source}.mlp.gate_proj.weight": f"{target}.mlp.gate_proj.weight",
            f"{source}.mlp.up_proj.weight": f"{target}.mlp.up_proj.weight",
            f"{source}.mlp.down_proj.weight": f"{target}.mlp.down_proj.weight",
        }
        for old_key, new_key in mapping.items():
            if old_key not in state:
                raise KeyError(f"missing Craftly checkpoint tensor: {old_key}")
            converted[new_key] = state[old_key]
    return converted


def export_hf_llama_package(
    *,
    checkpoint_manifest: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    torch_dtype: str = "float32",
) -> AdapterReport:
    require_torch()
    manifest, payload, config = _load_checkpoint(checkpoint_manifest)
    tokenizer = Path(tokenizer_path)
    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer file not found: {tokenizer}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    state = _convert_state_dict_to_hf_llama(payload["model"], config)
    try:
        from safetensors.torch import save_file as save_safetensors
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for production HF/vLLM export") from exc
    save_safetensors(state, target / "model.safetensors")
    (target / "config.json").write_text(
        json.dumps(_hf_llama_config(config, torch_dtype=torch_dtype), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copy2(tokenizer, target / "tokenizer.json")
    (target / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": config.max_sequence_length,
                "tokenizer_file": "tokenizer.json",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (target / "special_tokens_map.json").write_text(
        json.dumps(
            {"unk_token": "<unk>", "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    conversion = {
        "format": "hf_llama_compatible",
        "source_checkpoint_manifest": str(checkpoint_manifest),
        "source_checkpoint_sha256": sha256_file(Path(manifest.checkpoint_dir) / "model.pt"),
        "source_tokenizer_sha256": sha256_file(tokenizer),
        "parameter_count": config.parameter_estimate(),
        "serving_targets": {
            "vllm": "load output_dir as a local Hugging Face Llama-compatible model after validation",
            "tensorrt_llm": "use this package as the source checkpoint for TensorRT-LLM conversion tooling",
        },
        "required_validation": [
            "load with transformers LlamaForCausalLM",
            "compare logits against native Craftly on fixed token ids",
            "run vLLM smoke generation",
            "run TensorRT-LLM engine build and decode parity test",
        ],
    }
    (target / "craftly_conversion_manifest.json").write_text(
        json.dumps(conversion, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifacts = [
        str(target / name)
        for name in [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "craftly_conversion_manifest.json",
        ]
    ]
    report = AdapterReport(
        status="built_not_runtime_proven",
        adapter="hf_llama_vllm",
        output_dir=str(target),
        artifacts=artifacts,
    )
    report.write(target, "hf_export_report.json")
    return report


def validate_vllm_package(*, hf_model_dir: str | Path) -> AdapterReport:
    root = Path(hf_model_dir)
    missing = [
        str(root / name) for name in ["config.json", "tokenizer.json"] if not (root / name).exists()
    ]
    if not (root / "model.safetensors").exists() and not (root / "pytorch_model.bin").exists():
        missing.append(str(root / "model.safetensors"))
    if shutil.which("python") is None:
        missing.append("python executable")
    try:
        import vllm  # noqa: F401

        vllm_present = True
    except ImportError:
        vllm_present = False
        missing.append("python module: vllm")
    command = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(root)]
    return AdapterReport(
        status="production_ready_requires_external_service"
        if not missing
        else "blocked_missing_dependency",
        adapter="vllm",
        output_dir=str(root),
        command=command,
        missing_dependencies=missing,
        notes=["vLLM module present" if vllm_present else "vLLM module missing"],
    )


def build_tensorrt_llm_plan(
    *, hf_model_dir: str | Path, output_dir: str | Path, dtype: str = "bfloat16"
) -> AdapterReport:
    source = Path(hf_model_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    missing = [
        str(source / name)
        for name in ["config.json", "tokenizer.json"]
        if not (source / name).exists()
    ]
    if not (source / "model.safetensors").exists() and not (source / "pytorch_model.bin").exists():
        missing.append(str(source / "model.safetensors"))
    if shutil.which("trtllm-build") is None:
        missing.append("executable:trtllm-build")
    if shutil.which("python") is None:
        missing.append("python executable")
    command = [
        "trtllm-build",
        "--checkpoint_dir",
        str(source),
        "--output_dir",
        str(target / "engine"),
        "--gpt_attention_plugin",
        dtype,
        "--gemm_plugin",
        dtype,
    ]
    report = AdapterReport(
        status="production_ready_requires_external_service"
        if not missing
        else "blocked_missing_dependency",
        adapter="tensorrt_llm",
        output_dir=str(target),
        command=command,
        missing_dependencies=missing,
        notes=["Run TensorRT-LLM build/decode parity on NVIDIA GPU before production promotion."],
    )
    report.write(target, "tensorrt_llm_plan.json")
    return report


def build_megatron_launch_plan(
    *,
    manifest: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    model_profile: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    data_parallel: int,
    context_parallel: int = 1,
    expert_parallel: int = 1,
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    train_iters: int,
    num_nodes: int = 1,
    gpus_per_node: int | None = None,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    megatron_root: str | Path | None = None,
    execute: bool = False,
) -> AdapterReport:
    numeric_values = {
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": pipeline_parallel,
        "data_parallel": data_parallel,
        "context_parallel": context_parallel,
        "expert_parallel": expert_parallel,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "train_iters": train_iters,
        "num_nodes": num_nodes,
        "master_port": master_port,
    }
    invalid = [
        name for name, value in numeric_values.items() if not isinstance(value, int) or value <= 0
    ]
    if invalid:
        raise ValueError("Megatron integer settings must be positive: " + ", ".join(invalid))
    if node_rank < 0 or node_rank >= num_nodes:
        raise ValueError("node_rank must be between zero and num_nodes - 1")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", master_addr):
        raise ValueError("master_addr must be a bounded hostname or IP literal")
    if num_nodes > 1 and master_addr in {"127.0.0.1", "localhost"}:
        raise ValueError("multi-node Megatron plans require a non-loopback master_addr")
    if not 1024 <= master_port <= 65535:
        raise ValueError("master_port must be between 1024 and 65535")
    profile = get_model_profile(model_profile)
    if sequence_length > profile.max_sequence_length:
        raise ValueError(
            f"sequence_length={sequence_length} exceeds profile context={profile.max_sequence_length}"
        )
    world_size = tensor_parallel * pipeline_parallel * context_parallel * data_parallel
    if gpus_per_node is None:
        if world_size % num_nodes:
            raise ValueError("world_size must be divisible by num_nodes")
        gpus_per_node = world_size // num_nodes
    topology = profile.distributed_topology_report(
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        context_parallel=context_parallel,
        expert_parallel=expert_parallel,
        gpus_per_node=gpus_per_node,
    )
    if not topology["passed"]:
        raise ValueError("invalid Megatron topology: " + "; ".join(topology["failures"]))
    if num_nodes * gpus_per_node != world_size:
        raise ValueError("num_nodes * gpus_per_node must equal the parallel world_size")
    if global_batch_size % (micro_batch_size * data_parallel) != 0:
        raise ValueError("global_batch_size must be divisible by micro_batch_size * data_parallel")
    if tensor_parallel * pipeline_parallel > world_size:
        raise ValueError("invalid Megatron parallel topology")

    root_value = megatron_root or os.environ.get("MEGATRON_LM_ROOT", "")
    root = (
        Path(root_value).expanduser().resolve()
        if root_value
        else Path("__missing_megatron_root__").resolve()
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    manifest_path = Path(manifest).expanduser().resolve()
    tokenizer_file = Path(tokenizer_path).expanduser().resolve()
    data_prefix: Path | None = None
    try:
        data_prefix = resolve_megatron_data_prefix(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        missing.append(f"dataset:{exc}")
    if not tokenizer_file.is_file():
        missing.append(f"tokenizer:{tokenizer_path}")
    pretrain = (root / "pretrain_gpt.py").resolve()
    if not root.is_dir() or not pretrain.is_file() or root not in pretrain.parents:
        missing.append("MEGATRON_LM_ROOT/pretrain_gpt.py")
    torchrun = shutil.which("torchrun")
    if not torchrun:
        missing.append("executable:torchrun")
    command = [
        str(Path(torchrun).resolve()) if torchrun else "torchrun",
        "--nproc_per_node",
        str(gpus_per_node),
        "--nnodes",
        str(num_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        str(pretrain),
        "--tensor-model-parallel-size",
        str(tensor_parallel),
        "--pipeline-model-parallel-size",
        str(pipeline_parallel),
        "--context-parallel-size",
        str(context_parallel),
        *megatron_model_arguments(profile),
        "--seq-length",
        str(sequence_length),
        "--max-position-embeddings",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--train-iters",
        str(train_iters),
        "--save",
        str(output / "checkpoints"),
        "--load",
        str(output / "checkpoints"),
        "--data-path",
        str(data_prefix) if data_prefix is not None else "__missing_megatron_data_prefix__",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_file),
    ]
    if profile.feed_forward_architecture == "moe":
        command.extend(["--expert-model-parallel-size", str(expert_parallel)])
    report = AdapterReport(
        status="blocked_missing_dependency" if missing else "built_not_cluster_proven",
        adapter="megatron_lm",
        output_dir=str(output),
        command=command,
        missing_dependencies=missing,
        notes=[
            f"model_profile={model_profile}",
            f"parameter_report={json.dumps(profile.parameter_report(), sort_keys=True)}",
            f"topology_report={json.dumps(topology, sort_keys=True)}",
            "Megatron execution requires preprocessed indexed dataset and cluster validation.",
        ],
    )
    report.write(output, "megatron_launch_plan.json")
    if execute:
        if missing:
            raise RuntimeError(
                "cannot execute Megatron plan with missing dependencies: " + ", ".join(missing)
            )
        # Every path, integer, and executable above is canonicalized and bounded; argv is a list and shell stays disabled.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
        completed = subprocess.run(  # nosec B603
            command,  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            cwd=root,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Megatron command failed with exit code {completed.returncode}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Craftly production adapter utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    hf = sub.add_parser(
        "export-hf", help="Export native Craftly checkpoint as HF Llama-compatible package."
    )
    hf.add_argument("--checkpoint-manifest", required=True)
    hf.add_argument("--tokenizer-path", required=True)
    hf.add_argument("--output-dir", required=True)
    hf.add_argument("--torch-dtype", default="float32")
    vllm = sub.add_parser("validate-vllm", help="Validate local vLLM package prerequisites.")
    vllm.add_argument("--hf-model-dir", required=True)
    trt = sub.add_parser("plan-tensorrt", help="Write TensorRT-LLM build plan.")
    trt.add_argument("--hf-model-dir", required=True)
    trt.add_argument("--output-dir", required=True)
    trt.add_argument("--dtype", default="bfloat16")
    mega = sub.add_parser("plan-megatron", help="Write Megatron-LM launch plan.")
    mega.add_argument("--manifest", required=True)
    mega.add_argument("--tokenizer-path", required=True)
    mega.add_argument("--output-dir", required=True)
    mega.add_argument("--model-profile", default="7b")
    mega.add_argument("--tensor-parallel", type=int, default=1)
    mega.add_argument("--pipeline-parallel", type=int, default=1)
    mega.add_argument("--data-parallel", type=int, default=1)
    mega.add_argument("--context-parallel", type=int, default=1)
    mega.add_argument("--expert-parallel", type=int, default=1)
    mega.add_argument("--sequence-length", type=int, default=2048)
    mega.add_argument("--micro-batch-size", type=int, default=1)
    mega.add_argument("--global-batch-size", type=int, default=8)
    mega.add_argument("--train-iters", type=int, default=1000)
    mega.add_argument("--num-nodes", type=int, default=1)
    mega.add_argument("--gpus-per-node", type=int)
    mega.add_argument("--node-rank", type=int, default=0)
    mega.add_argument("--master-addr", default="127.0.0.1")
    mega.add_argument("--master-port", type=int, default=29500)
    mega.add_argument("--megatron-root")
    mega.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export-hf":
        report = export_hf_llama_package(
            checkpoint_manifest=args.checkpoint_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "validate-vllm":
        report = validate_vllm_package(hf_model_dir=args.hf_model_dir)
    elif args.command == "plan-tensorrt":
        report = build_tensorrt_llm_plan(
            hf_model_dir=args.hf_model_dir, output_dir=args.output_dir, dtype=args.dtype
        )
    else:
        report = build_megatron_launch_plan(
            manifest=args.manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            model_profile=args.model_profile,
            tensor_parallel=args.tensor_parallel,
            pipeline_parallel=args.pipeline_parallel,
            data_parallel=args.data_parallel,
            context_parallel=args.context_parallel,
            expert_parallel=args.expert_parallel,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            train_iters=args.train_iters,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            megatron_root=args.megatron_root,
            execute=args.execute,
        )
    print_report(report)
    if report.status == "blocked_missing_dependency":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
