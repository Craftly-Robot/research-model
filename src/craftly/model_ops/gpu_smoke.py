"""GPU smoke runner for Craftly scratch decoder training path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.craftly.model_ops.foundation import CheckpointManifest
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    save_trusted_checkpoint,
    select_torch_device,
    tiny_smoke_config,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def gpu_name(device: "torch.device") -> str:
    require_torch()
    assert torch is not None
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return "cpu"


def run_scratch_gpu_smoke(
    *,
    device: str = "auto",
    output_dir: str | Path = "artifacts/craftly/gpu-smoke",
    batch_size: int = 2,
    sequence_length: int = 64,
    steps: int = 2,
    seed: int = 1337,
    dtype: str = "bf16",
) -> dict[str, Any]:
    require_torch()
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    started = time.perf_counter()
    selected_device = select_torch_device(device)
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = tiny_smoke_config()
    if sequence_length > config.max_sequence_length:
        raise ValueError(
            "sequence_length exceeds tiny smoke config max_sequence_length"
        )
    model = CraftlyDecoderLM(config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1
    )
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    use_autocast = selected_device.type == "cuda" and dtype in {"bf16", "fp16"}
    losses: list[float] = []
    model.train()
    for _step in range(steps):
        tokens = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, sequence_length),
            device=selected_device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=selected_device.type, dtype=autocast_dtype, enabled=use_autocast
        ):
            output = model(tokens, labels=tokens)
            if output.loss is None:
                raise RuntimeError("scratch decoder did not produce a loss")
            loss = output.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    root = Path(output_dir).resolve()
    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_trusted_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.model_dump(),
            "step": steps,
        },
        checkpoint_dir / "model.pt",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = CheckpointManifest.from_directory(
        architecture_name=config.name,
        run_id="gpu-smoke",
        step=steps,
        trained_tokens=batch_size * sequence_length * steps,
        checkpoint_dir=checkpoint_dir,
        metrics={"loss": losses[-1], "grad_norm": float(grad_norm.detach().cpu())},
    )
    manifest_path = manifest.write_atomic(root / "checkpoint_manifest.json")
    max_memory_mb = 0.0
    if selected_device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(selected_device) / (1024 * 1024)
    report = {
        "status": "passed",
        "scratch_only": True,
        "borrowed_model_used": False,
        "device": str(selected_device),
        "device_name": gpu_name(selected_device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "config": config.model_dump(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "steps": steps,
        "losses": losses,
        "max_memory_allocated_mb": round(max_memory_mb, 3),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_manifest": str(manifest_path),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    (root / "gpu_smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Craftly scratch decoder GPU smoke test."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="artifacts/craftly/gpu-smoke")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_scratch_gpu_smoke(
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        steps=args.steps,
        seed=args.seed,
        dtype=args.dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
