"""Kaggle/Colab entrypoint for Craftly scratch tokenizer->shards->pretrain pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.model_ops.pretrain_loop import run_pretraining_loop  # noqa: E402
from src.craftly.model_ops.tokenizer_pipeline import ShardBuildConfig, TokenizerTrainConfig, build_token_shards, train_bpe_tokenizer  # noqa: E402
from src.craftly.shared.progress import ProgressReporter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly scratch tokenizer, shard builder, and pretraining loop.")
    parser.add_argument("--input", required=True, help="Clean JSONL/text corpus path.")
    parser.add_argument("--output-dir", default="artifacts/craftly/pretraining-pipeline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vocab-size", type=int, default=128_000)
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--progress-path")
    parser.add_argument("--no-progress-stdout", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    progress = ProgressReporter(path=args.progress_path or root / "progress.jsonl", to_stdout=not args.no_progress_stdout)
    progress.emit("pretraining_pipeline", "started", input=args.input, output_dir=str(root), steps=args.steps)
    progress.emit("tokenizer", "started", vocab_size=args.vocab_size)
    tokenizer_path = train_bpe_tokenizer(
        [args.input],
        root / "tokenizer" / "tokenizer.json",
        TokenizerTrainConfig(vocab_size=args.vocab_size),
    )
    progress.emit("tokenizer", "complete", tokenizer_path=str(tokenizer_path))
    progress.emit("sharding", "started", shard_token_count=args.shard_token_count, sequence_length=args.sequence_length)
    manifest = build_token_shards(
        input_paths=[args.input],
        tokenizer_path=tokenizer_path,
        output_dir=root / "shards",
        config=ShardBuildConfig(
            shard_token_count=args.shard_token_count,
            sequence_length=args.sequence_length,
            validation_fraction=args.validation_fraction,
        ),
    )
    progress.emit(
        "sharding",
        "complete",
        train_tokens=manifest.train_tokens,
        val_tokens=manifest.val_tokens,
        train_shards=len(manifest.train_shards),
        val_shards=len(manifest.val_shards),
    )
    report = run_pretraining_loop(
        output_dir=root / "train",
        manifest=root / "shards" / "manifest.json",
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dtype=args.dtype,
        progress=progress,
        progress_every_steps=args.progress_every_steps,
    )
    progress.emit("pretraining_pipeline", "complete", training_status=report["status"], checkpoint_manifest=report["checkpoint_manifest"])
    print(json.dumps({"tokenizer": str(tokenizer_path), "manifest": manifest.model_dump(), "training": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

