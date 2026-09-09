"""Authoritative full-parameter Supervised Fine-Tuning (SFT) loop for Craftly models.

Maintains strict scratch-origin provenance by resuming exclusively from Craftly
foundation checkpoints. Updates all parameters without LoRA or adapters, using
masked cross-entropy loss focused strictly on defensive completion tokens.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.continual import ExperienceReplayBuffer, ReferenceModelKLLoss
from src.craftly.learning.sft_dataset import SFTDatasetEngine, encode_sft_record
from src.craftly.model_ops.foundation import CheckpointManifest, ScratchDecoderConfig
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    load_trusted_checkpoint,
    require_torch,
    save_trusted_checkpoint,
    select_torch_device,
)
from src.craftly.shared.progress import NullProgressReporter, ProgressReporter
from src.craftly.shared.schemas import StrictModel

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class SFTRunConfig(StrictModel):
    """Configuration for an SFT instruction-tuning execution."""

    base_checkpoint: str = Field(description="Path to base pretrained checkpoint or manifest")
    tokenizer_path: str = Field(description="Path to tokenizer.json")
    output_dir: str = Field(description="Output directory for SFT checkpoints and logs")
    dataset_path: str | None = Field(default=None, description="Optional path to existing SFT JSONL dataset")
    steps: int = Field(default=1000, ge=1, le=100_000)
    batch_size: int = Field(default=4, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=3e-5, gt=0.0, le=1e-2)
    warmup_steps: int = Field(default=50, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_sequence_length: int = Field(default=1024, ge=64, le=4096)
    validate_every: int = Field(default=50, ge=1)
    checkpoint_every: int = Field(default=250, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    keep_last_n_checkpoints: int = Field(default=3, ge=1)
    seed: int = 1337
    dtype: str = Field(default="bf16", pattern=r"^(bf16|fp16|fp32)$")
    device: str = Field(default="auto", description="PyTorch device ('auto', 'cuda', 'cpu')")
    replay_ratio: float = Field(default=0.0, ge=0.0, le=0.8, description="Proportion of pretraining code replay")
    kl_penalty_weight: float = Field(default=0.0, ge=0.0, le=10.0, description="Weight for reference model KL penalty")


def resolve_base_model_assets(base_checkpoint_ref: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve model weights state_dict and architecture config from a checkpoint or manifest."""
    ref_path = Path(base_checkpoint_ref).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found at: {ref_path}")

    # Case 0: Zip archive containing model bundle
    if ref_path.suffix == ".zip":
        import zipfile
        extract_dir = ref_path.parent / f"_extracted_{ref_path.stem}"
        if not (extract_dir / "model.pt").exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ref_path, "r") as zf:
                zf.extractall(extract_dir)
        model_file = extract_dir / "model.pt"
        config_file = extract_dir / "config.json"
    # Case 1: Manifest JSON
    elif ref_path.name.endswith(".json") and "manifest" in ref_path.name.lower():
        with ref_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        ckpt_dir = Path(manifest_data.get("checkpoint_dir", ref_path.parent))
        model_file = ckpt_dir / "model.pt"
        config_file = ckpt_dir / "config.json"
    elif ref_path.is_dir():
        model_file = ref_path / "model.pt"
        config_file = ref_path / "config.json"
    else:
        model_file = ref_path
        config_file = ref_path.parent / "config.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Missing model.pt in checkpoint asset tree: {model_file}")

    raw_payload = load_trusted_checkpoint(model_file, map_location="cpu")

    if isinstance(raw_payload, dict) and "model" in raw_payload:
        state_dict = raw_payload["model"]
        config_dict = raw_payload.get("config", {})
    else:
        state_dict = raw_payload
        config_dict = {}

    if not config_dict and config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

    if not config_dict:
        raise ValueError(f"Unable to resolve architecture configuration for model at: {model_file}")

    return model_file, state_dict, config_dict


def build_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def prune_sft_checkpoints(output_dir: Path, keep_last_n: int = 3, preserve_dirs: set[Path] | None = None) -> list[Path]:
    """Prune older intermediate checkpoints to preserve disk space."""
    preserve = preserve_dirs or set()
    pruned: list[Path] = []
    checkpoint_dirs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-sft-step-")],
        key=lambda p: p.name,
    )
    if len(checkpoint_dirs) > keep_last_n:
        for obsolete in checkpoint_dirs[:-keep_last_n]:
            if obsolete not in preserve:
                shutil.rmtree(obsolete, ignore_errors=True)
                pruned.append(obsolete)
    return pruned


def run_sft(
    run_config: SFTRunConfig,
    progress_reporter: ProgressReporter | None = None,
    on_checkpoint: Any = None,
) -> dict[str, Any]:
    """Execute full-parameter SFT instruction fine-tuning."""
    require_torch()
    reporter = progress_reporter or NullProgressReporter()
    start_time = time.time()

    # 1. Setup paths and directories
    out_dir = Path(run_config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Resolve base model assets
    base_file, state_dict, config_dict = resolve_base_model_assets(run_config.base_checkpoint)
    arch_config = ScratchDecoderConfig(**config_dict)

    # 3. Initialize Model and Load Scratch Weights
    device = select_torch_device(run_config.device)
    model = CraftlyDecoderLM(arch_config)
    model.load_state_dict(state_dict)
    model.train()
    model = model.to(device)

    # 4. Load Tokenizer
    tok_path = Path(run_config.tokenizer_path).resolve()
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer = load_tokenizer(tok_path)

    # 5. Prepare SFT Dataset
    if run_config.dataset_path and Path(run_config.dataset_path).exists():
        engine = SFTDatasetEngine.load_jsonl(run_config.dataset_path)
    else:
        engine = SFTDatasetEngine()
        engine.expand_dataset(target_count=max(600, run_config.steps * run_config.batch_size), seed=run_config.seed)
        dataset_save_file = out_dir / "sft_training_corpus.jsonl"
        engine.save_jsonl(dataset_save_file)

    train_records, val_records = engine.partition(val_ratio=0.1, seed=run_config.seed)

    # 6. Optimizer and Precision
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": run_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=run_config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    use_autocast = device.type == "cuda" and run_config.dtype in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if run_config.dtype == "bf16" else torch.float16

    # 7. Training Loop State
    train_losses: list[float] = []
    val_losses: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    step = 0
    train_record_idx = 0
    trained_tokens = 0

    replay_buffer = None
    if run_config.replay_ratio > 0.0:
        replay_buffer = ExperienceReplayBuffer(replay_ratio=run_config.replay_ratio, seed=run_config.seed)

    ref_model = None
    kl_loss_fn = None
    if run_config.kl_penalty_weight > 0.0:
        ref_model = CraftlyDecoderLM(arch_config)
        ref_model.load_state_dict(state_dict)
        ref_model.eval()
        ref_model = ref_model.to(device)
        for p in ref_model.parameters():
            p.requires_grad = False
        kl_loss_fn = ReferenceModelKLLoss()

    reporter.emit(
        stage="sft",
        status="running",
        step=0,
        requested_steps=run_config.steps,
        message=f"Starting SFT on {device} ({arch_config.name}, {len(train_records)} train records, replay: {run_config.replay_ratio:.0%}, KL: {run_config.kl_penalty_weight})",
    )

    optimizer.zero_grad(set_to_none=True)

    while step < run_config.steps:
        step_loss_acc = 0.0

        for _ in range(run_config.gradient_accumulation_steps):
            # Form micro-batch with optional pretraining replay
            batch_input_ids = []
            batch_labels = []

            replay_count = 0
            if replay_buffer and run_config.replay_ratio > 0.0:
                replay_count = max(1, int(round(run_config.batch_size * run_config.replay_ratio)))
                replay_count = min(replay_count, max(1, run_config.batch_size - 1))

            sft_count = max(1, run_config.batch_size - replay_count)
            for _ in range(sft_count):
                rec = train_records[train_record_idx % len(train_records)]
                train_record_idx += 1
                inp_ids, lbls, _ = encode_sft_record(
                    rec,
                    tokenizer,
                    max_sequence_length=run_config.max_sequence_length,
                )
                batch_input_ids.append(inp_ids)
                batch_labels.append(lbls)

            if replay_count > 0 and replay_buffer:
                r_inps, r_lbls = replay_buffer.sample_batch(
                    count=replay_count,
                    tokenizer=tokenizer,
                    max_sequence_length=run_config.max_sequence_length,
                )
                batch_input_ids.extend(r_inps)
                batch_labels.extend(r_lbls)

            inp_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
            lbl_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            # Count unmasked target completion tokens
            unmasked = int((lbl_tensor != -100).sum().item())
            trained_tokens += unmasked

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                output = model(inp_tensor, labels=lbl_tensor)
                if output.loss is None:
                    raise RuntimeError("SFT model forward pass did not return loss")
                raw_loss = output.loss
                if kl_loss_fn and ref_model and run_config.kl_penalty_weight > 0.0:
                    with torch.no_grad():
                        ref_out = ref_model(inp_tensor)
                    kl_val = kl_loss_fn.compute(output.logits, ref_out.logits, mask=lbl_tensor)
                    raw_loss = raw_loss + (run_config.kl_penalty_weight * kl_val)

                loss = raw_loss / run_config.gradient_accumulation_steps

            loss.backward()
            step_loss_acc += float(output.loss.detach().cpu())

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        curr_lr = build_cosine_lr(step, run_config.steps, run_config.warmup_steps, run_config.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        avg_loss = step_loss_acc / run_config.gradient_accumulation_steps
        train_losses.append(avg_loss)

        # Progress reporting
        reporter.emit(
            stage="sft",
            status="running",
            step=step,
            requested_steps=run_config.steps,
            loss=round(avg_loss, 4),
            learning_rate=curr_lr,
            trained_tokens=trained_tokens,
        )

        # 8. Validation check
        if step % run_config.validate_every == 0 or step == run_config.steps:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                val_sample_records = val_records[: min(len(val_records), 16)]
                for i in range(0, len(val_sample_records), run_config.batch_size):
                    chunk = val_sample_records[i : i + run_config.batch_size]
                    v_inps, v_lbls = [], []
                    for r in chunk:
                        in_ids, lbs, _ = encode_sft_record(
                            r,
                            tokenizer,
                            max_sequence_length=run_config.max_sequence_length,
                        )
                        v_inps.append(in_ids)
                        v_lbls.append(lbs)
                    v_inp_t = torch.tensor(v_inps, dtype=torch.long, device=device)
                    v_lbl_t = torch.tensor(v_lbls, dtype=torch.long, device=device)

                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        val_out = model(v_inp_t, labels=v_lbl_t)
                        if val_out.loss is not None:
                            val_loss_sum += float(val_out.loss.detach().cpu())
                            val_batches += 1

            eval_loss = val_loss_sum / max(1, val_batches)
            val_losses.append({"step": step, "loss": eval_loss})
            model.train()

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                best_step = step
                patience_counter = 0
            else:
                patience_counter += 1

            reporter.emit(
                stage="sft_validation",
                status="running",
                step=step,
                validation_loss=round(eval_loss, 4),
                best_val_loss=round(best_val_loss, 4),
                best_step=best_step,
            )

            # Early stopping check
            if patience_counter >= run_config.early_stopping_patience:
                reporter.emit(
                    stage="sft",
                    status="early_stopped",
                    step=step,
                    message=f"Early stopping triggered at step {step} (Best Val Loss: {best_val_loss:.4f})",
                )
                break

        # 9. Scheduled checkpoint save
        if step % run_config.checkpoint_every == 0 or step == run_config.steps:
            ckpt_dir = out_dir / f"checkpoint-sft-step-{step:08d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_trusted_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "trained_tokens": trained_tokens,
                    "config": arch_config.model_dump(),
                },
                ckpt_dir / "model.pt",
            )
            (ckpt_dir / "config.json").write_text(json.dumps(arch_config.model_dump(), indent=2), encoding="utf-8")

            manifest = CheckpointManifest.from_directory(
                architecture_name=arch_config.name,
                run_id="sft-instruction-tuning",
                step=step,
                trained_tokens=trained_tokens,
                checkpoint_dir=ckpt_dir,
                metrics={"train_loss": avg_loss, "best_val_loss": best_val_loss},
            )
            manifest.write_atomic(out_dir / "checkpoint_manifest.json")
            prune_sft_checkpoints(out_dir, keep_last_n=run_config.keep_last_n_checkpoints, preserve_dirs={ckpt_dir})

            if callable(on_checkpoint):
                try:
                    on_checkpoint(step=step, tokens=trained_tokens, ckpt_path=ckpt_dir)
                except Exception:
                    pass

    # Final Save
    final_ckpt_dir = out_dir / f"checkpoint-sft-step-{step:08d}"
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_trusted_checkpoint(
        {
            "model": model.state_dict(),
            "step": step,
            "trained_tokens": trained_tokens,
            "config": arch_config.model_dump(),
        },
        final_ckpt_dir / "model.pt",
    )
    (final_ckpt_dir / "config.json").write_text(json.dumps(arch_config.model_dump(), indent=2), encoding="utf-8")

    manifest = CheckpointManifest.from_directory(
        architecture_name=arch_config.name,
        run_id="sft-instruction-tuning",
        step=step,
        trained_tokens=trained_tokens,
        checkpoint_dir=final_ckpt_dir,
        metrics={"final_train_loss": train_losses[-1] if train_losses else 0.0, "best_val_loss": best_val_loss},
    )
    manifest_path = manifest.write_atomic(out_dir / "checkpoint_manifest.json")

    duration = time.time() - start_time
    summary = {
        "status": "completed",
        "steps_completed": step,
        "final_train_loss": train_losses[-1] if train_losses else 0.0,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "trained_tokens": trained_tokens,
        "duration_seconds": round(duration, 2),
        "checkpoint_manifest": str(manifest_path),
        "latest_checkpoint_dir": str(final_ckpt_dir),
    }

    reporter.emit(stage="sft", **summary)
    return summary
