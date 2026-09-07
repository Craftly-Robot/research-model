"""Run a real approved-source Craftly data -> GPU training -> eval job.

This entrypoint is intended for Kaggle/Colab smoke runs and single-node GPU
jobs. For production-scale collection, use the same pipeline with Postgres
frontier workers and object storage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.data_pipeline import DataPipelineConfig, run_data_pipeline  # noqa: E402


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_recent_progress(path: Path, *, limit: int = 80) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def _append_progress_blocked(progress_path: str, reason: str) -> None:
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts_unix": time.time(),
        "stage": "run",
        "status": "blocked",
        "block_reason": reason,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _partial_report_state(output_dir: str, progress_path: str) -> dict[str, object]:
    reports = Path(output_dir) / "reports"
    report_files = {
        "license_filter_report": reports / "license_filter_report.json",
        "benchmark_contamination_filter_report": reports / "benchmark_contamination_filter_report.json",
        "near_dedup_report": reports / "near_dedup_report.json",
        "contamination_report": reports / "contamination_report.json",
        "quality_report": reports / "quality_report.json",
        "task_report": reports / "task_extraction_report.json",
        "review_report": reports / "task_review_report.json",
        "source_reputation_report": reports / "source_reputation_report.json",
        "source_budget_plan": reports / "source_budget_plan.json",
        "training_data_gate_report": reports / "training_data_gate_report.json",
        "source_balance_report": reports / "source_balance_report.json",
        "instruction_mix_report": reports / "instruction_mix_report.json",
        "training_quality_report": reports / "training_quality_report.json",
        "checkpoint_comparison_report": reports / "checkpoint_compare" / "checkpoint_comparison_report.json",
    }
    return {
        key: value
        for key, value in {name: _read_json(path) for name, path in report_files.items()}.items()
        if value
    } | {"recent_progress": _read_recent_progress(Path(progress_path))}


def _write_run_report(output_dir: str, payload: dict[str, object]) -> None:
    report_dir = Path(output_dir, "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(report_dir, "real_data_training_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _blocked_payload(args: argparse.Namespace, *, progress_path: str, reason: str) -> dict[str, object]:
    _append_progress_blocked(progress_path, reason)
    recommendation = (
        "Strict production gates stayed closed. Increase --max-docs/source yield for production, "
        "or rerun validation intentionally with --kaggle-validation."
    )
    payload: dict[str, object] = {
        "status": "blocked",
        "dataset_id": args.dataset_id,
        "work_dir": args.output_dir,
        "block_reason": reason,
        "progress_path": progress_path,
        "validation_profile": "kaggle" if args.kaggle_validation else "custom",
        "thresholds": {
            "min_clean_records": args.min_clean_records,
            "min_training_rows": args.min_training_rows,
            "min_train_tokens": args.min_train_tokens,
            "min_training_quality_score": args.min_training_quality_score,
            "min_training_average_quality_score": args.min_training_average_quality_score,
            "min_source_reputation_score": args.min_source_reputation_score,
            "max_source_fraction": args.max_source_fraction,
        },
        "recommendation": recommendation,
        "next_commands": {
            "inspect": f"python deploy/gpu/inspect_real_data_run.py --work-dir {Path(args.output_dir).as_posix()}",
            "kaggle_validation_retry": (
                "PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py "
                f"--sources {args.sources} --work-dir {Path(args.output_dir).as_posix()} "
                "--kaggle-validation --progress-to-stdout"
            ),
        },
    }
    payload.update(_partial_report_state(args.output_dir, progress_path))
    _write_run_report(args.output_dir, payload)
    return payload


def apply_kaggle_validation_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not args.kaggle_validation:
        return args
    if args.production:
        raise ValueError("--kaggle-validation cannot be combined with --production")
    args.min_clean_records = min(args.min_clean_records, 800)
    args.min_training_rows = min(args.min_training_rows, 800)
    args.min_train_tokens = min(args.min_train_tokens, 100_000)
    args.max_docs = max(args.max_docs, 16_000)
    args.max_bytes_per_doc = min(args.max_bytes_per_doc, 250_000)
    args.workers = min(args.workers, 6)
    args.progress_to_stdout = True
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real approved-source crawl, shard, scratch training, and checkpoint eval.")
    parser.add_argument("--sources", default="config/data_sources.ultimate.json")
    parser.add_argument("--dataset-id", default="craftly-real-approved-corpus")
    parser.add_argument("--output-dir", "--work-dir", dest="output_dir", default="artifacts/craftly/real-data-training")
    parser.add_argument("--frontier-backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--postgres-dsn", default=os.environ.get("CRAFTLY_DATABASE_URL"))
    parser.add_argument("--max-docs", type=int, default=10_000)
    parser.add_argument(
        "--max-bytes-per-doc",
        type=int,
        default=300_000,
        help="Kaggle-safe document text cap. Increase on larger machines.",
    )
    parser.add_argument("--min-clean-records", "--target-records", dest="min_clean_records", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--vocab-size", type=int, default=128_000)
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--train-steps", "--steps", dest="train_steps", type=int, default=1_000)
    parser.add_argument("--train-batch-size", "--batch-size", dest="train_batch_size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
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
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--model-profile", default="tiny", choices=["tiny", "t4_validation", "1b", "7b", "32b", "62b"])
    parser.add_argument("--attention-impl", default="auto", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validate-every", "--validation-interval", dest="validate_every", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-training-data-gate", action="store_true")
    parser.add_argument("--min-training-quality-score", type=float, default=0.62)
    parser.add_argument("--min-training-average-quality-score", type=float, default=0.62)
    parser.add_argument("--min-training-rows", type=int, default=5_000)
    parser.add_argument("--min-train-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-source-reputation-score", type=float, default=0.50)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--no-source-balancing", action="store_true")
    parser.add_argument("--max-source-fraction", type=float, default=0.25)
    parser.add_argument("--min-source-rows", type=int, default=25)
    parser.add_argument("--no-instruction-mix", action="store_true")
    parser.add_argument("--instruction-mix-max-rows", type=int)
    parser.add_argument(
        "--curriculum-mode",
        default="balanced",
        choices=["balanced", "fundamentals_only", "defensive_security_only", "debug_patch_only", "agentic_coding_only"],
    )
    parser.add_argument("--allow-offensive-misuse-rows", action="store_true")
    parser.add_argument("--object-store-uri", default=os.environ.get("CRAFTLY_OBJECT_STORE_URI", "local://artifacts/craftly/object-store"))
    parser.add_argument("--object-store-endpoint-url", default=os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL"))
    parser.add_argument("--checkpoint-compare-prompt-suite")
    parser.add_argument("--checkpoint-compare-min-score", type=float, default=0.0)
    parser.add_argument("--checkpoint-compare-max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-compare-repetition-penalty", type=float, default=1.12)
    parser.add_argument("--checkpoint-compare-no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--checkpoint-compare-max-repetition-ratio", type=float, default=0.72)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--progress-path")
    parser.add_argument("--progress-to-stdout", action="store_true", help="Explicitly stream structured progress events to stdout.")
    parser.add_argument("--no-progress-stdout", action="store_true")
    parser.add_argument("--progress-every-docs", type=int, default=10)
    parser.add_argument("--progress-every-steps", type=int, default=10)
    parser.add_argument("--dataloader-prefetch-batches", type=int, default=4)
    parser.add_argument("--dataloader-seed", type=int, default=1337)
    parser.add_argument("--expected-python-version")
    parser.add_argument("--expected-pytorch-version")
    parser.add_argument("--expected-cuda-version")
    parser.add_argument("--production", action="store_true", help="Enable strict production data/training validation.")
    parser.add_argument("--dev-smoke", action="store_true", help="Explicitly allow tiny/dev smoke behavior under production validation.")
    parser.add_argument("--kaggle-validation", action="store_true", help="Use explicit Kaggle-safe validation thresholds; not for production.")
    args = parser.parse_args()
    try:
        return apply_kaggle_validation_profile(args)
    except ValueError as exc:
        parser.error(str(exc))


def clean_record_count(report: dict[str, object]) -> int:
    crawl = report.get("crawl")
    if isinstance(crawl, dict):
        return int(crawl.get("accepted", 0))
    return 0


async def run(args: argparse.Namespace) -> dict[str, object]:
    progress_path = args.progress_path or str(Path(args.output_dir) / "progress.jsonl")
    try:
        report = await run_data_pipeline(
            DataPipelineConfig(
                sources_path=args.sources,
                dataset_id=args.dataset_id,
                work_dir=args.output_dir,
                frontier_backend=args.frontier_backend,
                postgres_dsn=args.postgres_dsn,
                max_docs=args.max_docs,
                max_bytes_per_doc=args.max_bytes_per_doc,
                workers=args.workers,
                max_depth=args.max_depth,
                delay_seconds=args.delay_seconds,
                shard_rows=10_000,
                vocab_size=args.vocab_size,
                tokenizer_min_frequency=2,
                shard_token_count=args.shard_token_count,
                sequence_length=args.sequence_length,
                validation_fraction=args.validation_fraction,
                skip_train=args.skip_train,
                train_steps=args.train_steps,
                train_device=args.device,
                train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
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
                dtype=args.dtype,
                model_profile_name=args.model_profile,
                attention_impl=args.attention_impl,
                gradient_checkpointing=args.gradient_checkpointing,
                validate_every=min(args.validate_every, args.train_steps) if args.validate_every > 0 else 0,
                validation_batches=args.validation_batches,
                checkpoint_every=args.checkpoint_every,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                apply_training_data_gate=not args.no_training_data_gate,
                min_training_quality_score=args.min_training_quality_score,
                min_training_average_quality_score=args.min_training_average_quality_score,
                min_training_rows=args.min_training_rows,
                min_train_tokens=args.min_train_tokens,
                min_source_reputation_score=args.min_source_reputation_score,
                eval_holdout_fraction=args.eval_holdout_fraction,
                balance_sources=not args.no_source_balancing,
                max_source_fraction=args.max_source_fraction,
                min_source_rows=args.min_source_rows,
                instruction_mix=not args.no_instruction_mix,
                instruction_mix_max_rows=args.instruction_mix_max_rows,
                curriculum_mode=args.curriculum_mode,
                strict_offensive_filter=not args.allow_offensive_misuse_rows,
                run_checkpoint_eval=not args.skip_train,
                object_store_uri=args.object_store_uri,
                object_store_endpoint_url=args.object_store_endpoint_url,
                checkpoint_compare_prompt_suite=args.checkpoint_compare_prompt_suite,
                checkpoint_compare_min_score=args.checkpoint_compare_min_score,
                checkpoint_compare_max_new_tokens=args.checkpoint_compare_max_new_tokens,
                checkpoint_compare_repetition_penalty=args.checkpoint_compare_repetition_penalty,
                checkpoint_compare_no_repeat_ngram_size=args.checkpoint_compare_no_repeat_ngram_size,
                checkpoint_compare_max_repetition_ratio=args.checkpoint_compare_max_repetition_ratio,
                upload_artifacts=True,
                progress_path=progress_path,
                progress_to_stdout=args.progress_to_stdout or not args.no_progress_stdout,
                progress_every_docs=args.progress_every_docs,
                progress_every_steps=args.progress_every_steps,
                dataloader_prefetch_batches=args.dataloader_prefetch_batches,
                dataloader_seed=args.dataloader_seed,
                expected_python_version=args.expected_python_version,
                expected_pytorch_version=args.expected_pytorch_version,
                expected_cuda_version=args.expected_cuda_version,
                production_mode=args.production,
                dev_smoke=args.dev_smoke,
            )
        )
    except RuntimeError as exc:
        return _blocked_payload(args, progress_path=progress_path, reason=str(exc))
    payload = report.model_dump()
    accepted = clean_record_count(payload)
    if accepted < args.min_clean_records:
        payload["status"] = "blocked"
        payload["block_reason"] = (
            f"accepted clean records {accepted} below required minimum {args.min_clean_records}; "
            "add approved sources, increase --max-docs, or lower --min-clean-records for a smoke run"
        )
        payload["validation_profile"] = "kaggle" if args.kaggle_validation else "custom"
        payload["next_commands"] = {
            "inspect": f"python deploy/gpu/inspect_real_data_run.py --work-dir {Path(args.output_dir).as_posix()}",
            "kaggle_validation_retry": (
                "PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py "
                f"--sources {args.sources} --work-dir {Path(args.output_dir).as_posix()} "
                "--kaggle-validation --progress-to-stdout"
            ),
        }
        _write_run_report(args.output_dir, payload)
        return payload
    payload["status"] = "complete"
    payload["accepted_clean_records"] = accepted
    _write_run_report(args.output_dir, payload)
    return payload


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "event": "craftly_real_data_training_start",
                "work_dir": args.output_dir,
                "progress_path": args.progress_path or str(Path(args.output_dir) / "progress.jsonl"),
                "progress_stdout": not args.no_progress_stdout,
                "train_steps": args.train_steps,
                "max_docs": args.max_docs,
                "min_clean_records": args.min_clean_records,
                "min_training_rows": args.min_training_rows,
                "min_training_quality_score": args.min_training_quality_score,
                "min_training_average_quality_score": args.min_training_average_quality_score,
                "min_train_tokens": args.min_train_tokens,
                "instruction_mix_enabled": not args.no_instruction_mix,
                "curriculum_mode": args.curriculum_mode,
                "checkpoint_compare_prompt_suite": args.checkpoint_compare_prompt_suite or "",
                "validation_profile": "kaggle" if args.kaggle_validation else "custom",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    payload = asyncio.run(run(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload.get("status") == "complete" else 1)


if __name__ == "__main__":
    main()


