"""Run a serious Craftly scratch GPU training validation job.

This script is intentionally separate from the quick smoke tests. It is meant
for Colab/Kaggle or a real GPU node and defaults to 10,000 optimizer steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.craftly.evaluation.checkpoint_eval import evaluate_checkpoint
from src.craftly.model_ops.pretrain_loop import run_pretraining_loop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly 10k-step scratch GPU validation.")
    parser.add_argument("--manifest", required=True, help="Token shard manifest from tokenizer_pipeline/mixer.")
    parser.add_argument("--output-dir", default="artifacts/craftly/gpu-10k-validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--model-profile", default="tiny", choices=["tiny", "t4_validation", "1b", "7b", "32b", "62b"])
    parser.add_argument("--attention-impl", default="auto", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steps < 10_000:
        raise SystemExit("--steps must be at least 10000 for production GPU validation")
    root = Path(args.output_dir)
    train_dir = root / "train"
    report = run_pretraining_loop(
        manifest=args.manifest,
        output_dir=train_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        dtype=args.dtype,
        validate_every=args.validate_every,
        validation_batches=args.validation_batches,
        early_stopping_patience=args.early_stopping_patience,
        model_profile_name=args.model_profile,
        attention_impl=args.attention_impl,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    checkpoint_eval = evaluate_checkpoint(
        checkpoint_manifest_path=report["checkpoint_manifest"],
        training_report=report,
        output_dir=root / "checkpoint_eval",
    )
    payload = {
        "status": "passed" if report["status"] == "passed" and checkpoint_eval.status == "passed" else "failed",
        "training": report,
        "checkpoint_eval": checkpoint_eval.model_dump(),
        "minimum_steps_enforced": 10_000,
    }
    (root / "gpu_10k_validation_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

