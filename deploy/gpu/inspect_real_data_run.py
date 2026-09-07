"""Inspect an Craftly real-data training run and recommend the next action.

This script is intentionally stdlib-only so it can run inside Kaggle/Colab even
when the training dependencies failed to install. It reads the structured report
and progress stream produced by ``run_real_data_training_pipeline.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_progress(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _last_stage(progress: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    for row in reversed(progress):
        if row.get("stage") == stage:
            return row
    return {}


def _training_payload(report: dict[str, Any]) -> dict[str, Any]:
    training = report.get("training")
    if isinstance(training, dict):
        merged = dict(report)
        merged.update(training)
        return merged
    return report


def _command_block(lines: list[str]) -> str:
    return "\n".join(lines)


def inspect_run(*, work_dir: Path, report_path: Path | None = None, progress_path: Path | None = None) -> dict[str, Any]:
    report_file = report_path or work_dir / "reports" / "real_data_training_report.json"
    progress_file = progress_path or work_dir / "progress.jsonl"
    report = _read_json(report_file)
    progress = _read_progress(progress_file)
    training = _training_payload(report)

    status = str(report.get("status") or training.get("status") or "missing_report")
    block_reason = str(report.get("block_reason") or "")
    crawl = report.get("crawl") if isinstance(report.get("crawl"), dict) else {}
    training_gate = report.get("training_gate_report") if isinstance(report.get("training_gate_report"), dict) else {}
    source_balance = report.get("source_balance_report") if isinstance(report.get("source_balance_report"), dict) else {}
    training_quality = report.get("training_quality_report") if isinstance(report.get("training_quality_report"), dict) else {}

    crawl_progress = _last_stage(progress, "crawl")
    gate_progress = _last_stage(progress, "training_data_gate")
    balance_progress = _last_stage(progress, "source_balancing")
    train_progress = _last_stage(progress, "pretraining")
    token_progress = _last_stage(progress, "tokenizer")

    accepted_clean = int(report.get("accepted_clean_records") or crawl.get("accepted") or crawl_progress.get("accepted") or 0)
    promoted = int(
        training_gate.get("promoted")
        or gate_progress.get("promoted")
        or source_balance.get("output_rows")
        or balance_progress.get("output_rows")
        or 0
    )
    balanced_rows = int(source_balance.get("output_rows") or balance_progress.get("output_rows") or promoted)
    avg_quality = training_quality.get("avg_quality_score") or _nested(report, "training_quality_report", "avg_quality_score")
    train_tokens = _nested(report, "manifest", "train_tokens") or _nested(report, "shard_manifest", "train_tokens") or 0
    train_status = str(training.get("status") or "")
    train_steps = training.get("steps")
    best_manifest = str(training.get("best_checkpoint_manifest") or training.get("checkpoint_manifest") or "")
    tokenizer_path = str(training.get("tokenizer_path") or _nested(report, "manifest", "tokenizer_path") or "")

    if not report_file.exists():
        decision = "missing_report"
        recommendation = "The run has not produced a final report yet, or work-dir/path is wrong."
    elif status == "complete" and best_manifest:
        decision = "ready_for_checkpoint_comparison"
        recommendation = "Run checkpoint comparison and benchmark eval on the produced checkpoint."
    elif "below required minimum" in block_reason or balanced_rows < 800:
        decision = "data_gate_blocked_low_promoted_rows"
        recommendation = "Increase source yield/max-docs, or lower Kaggle validation thresholds intentionally."
    elif report_file.exists() and status == "blocked":
        decision = "blocked"
        recommendation = "Read block_reason and inspect the last progress stages before retrying."
    else:
        decision = "needs_manual_review"
        recommendation = "Inspect report/progress for a non-standard state."

    compare_command = ""
    if best_manifest or status == "complete":
        compare_command = _command_block(
            [
                "python deploy/gpu/run_checkpoint_comparison.py \\",
                f"  --training-report {report_file.as_posix()} \\",
                f"  --output-dir {(work_dir / 'reports' / 'checkpoint_compare').as_posix()} \\",
                "  --device cuda",
            ]
        )

    validation_retry_command = _command_block(
        [
            "PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py \\",
            "  --sources config/data_sources.ultimate.json \\",
            f"  --work-dir {work_dir.as_posix()} \\",
            "  --kaggle-validation \\",
            "  --max-docs 24000 \\",
            "  --max-bytes-per-doc 250000 \\",
            "  --workers 6 \\",
            "  --max-depth 2 \\",
            "  --delay-seconds 0.35 \\",
            "  --steps 1000 \\",
            "  --sequence-length 128 \\",
            "  --batch-size 2 \\",
            "  --gradient-accumulation-steps 8 \\",
            "  --validation-interval 100 \\",
            "  --validation-batches 4 \\",
            "  --early-stopping-patience 5 \\",
            "  --progress-to-stdout",
        ]
    )

    return {
        "decision": decision,
        "status": status,
        "recommendation": recommendation,
        "work_dir": str(work_dir),
        "report_path": str(report_file),
        "progress_path": str(progress_file),
        "block_reason": block_reason,
        "accepted_clean_records": accepted_clean,
        "training_gate_promoted_rows": promoted,
        "balanced_training_rows": balanced_rows,
        "training_average_quality_score": avg_quality,
        "train_tokens": train_tokens,
        "training_status": train_status,
        "training_steps": train_steps,
        "best_checkpoint_manifest": best_manifest,
        "tokenizer_path": tokenizer_path,
        "latest_stages": {
            "crawl": crawl_progress,
            "tokenizer": token_progress,
            "training_data_gate": gate_progress,
            "source_balancing": balance_progress,
            "pretraining": train_progress,
        },
        "next_commands": {
            "checkpoint_comparison": compare_command,
            "validation_retry": validation_retry_command,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an Craftly Kaggle/Colab real-data training run.")
    parser.add_argument("--work-dir", default="artifacts/craftly/real-data-validation-v1")
    parser.add_argument("--report")
    parser.add_argument("--progress")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = inspect_run(
        work_dir=Path(args.work_dir),
        report_path=Path(args.report) if args.report else None,
        progress_path=Path(args.progress) if args.progress else None,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_blocked and payload["decision"] not in {"ready_for_checkpoint_comparison"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
