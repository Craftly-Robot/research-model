"""Post-checkpoint evaluation gates for scratch pretraining runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.craftly.evaluation.benchmarks import (
    BenchmarkHarness,
    BenchmarkRunReport,
    built_in_security_tasks,
)
from src.craftly.model_ops.foundation import CheckpointManifest, sha256_file
from src.craftly.shared.schemas import EvaluationGate as EvalGate
from src.craftly.shared.schemas import StrictModel


class CheckpointEvalReport(StrictModel):
    status: str
    checkpoint_manifest: str
    output_dir: str
    gates: list[EvalGate]
    benchmark_report: dict[str, Any]
    recommendations: list[str]

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "checkpoint_eval_report.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def _load_training_report(
    path_or_payload: str | Path | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if path_or_payload is None:
        return None
    if isinstance(path_or_payload, dict):
        return path_or_payload
    path = Path(path_or_payload)
    if not path.exists():
        raise FileNotFoundError(f"training report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _checkpoint_integrity_gate(manifest: CheckpointManifest) -> EvalGate:
    root = Path(manifest.checkpoint_dir)
    missing: list[str] = []
    mismatched: list[str] = []
    for file_info in manifest.files:
        relative = str(file_info["path"])
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        expected = str(file_info.get("sha256", ""))
        actual = sha256_file(path)
        if expected and actual != expected:
            mismatched.append(relative)
    status = "pass" if not missing and not mismatched and bool(manifest.files) else "fail"
    return EvalGate(
        name="checkpoint_integrity",
        status=status,
        reason="checkpoint files exist and hashes match"
        if status == "pass"
        else "checkpoint files missing or hash mismatch",
        metrics={"files": len(manifest.files), "missing": missing, "mismatched": mismatched},
    )


def _training_loss_gate(training_report: dict[str, Any] | None) -> EvalGate:
    if training_report is None:
        return EvalGate(
            name="training_loss",
            status="warn",
            reason="no training report provided; checkpoint integrity can be checked but loss trend cannot",
        )
    losses = [float(item) for item in training_report.get("train_losses", [])]
    finite = all(math.isfinite(item) for item in losses)
    if not losses:
        return EvalGate(
            name="training_loss", status="fail", reason="training report has no train_losses"
        )
    if not finite:
        return EvalGate(
            name="training_loss", status="fail", reason="training loss contains NaN or infinity"
        )
    first = losses[0]
    final = losses[-1]
    improvement = (first - final) / max(abs(first), 1e-9)
    if len(losses) == 1:
        status = "pass"
        reason = "single-step run produced a finite training loss"
    elif final <= first * 1.5:
        status = "pass"
        reason = "training loss is finite and non-exploding"
    else:
        status = "fail"
        reason = "training loss increased beyond the non-explosion threshold"
    return EvalGate(
        name="training_loss",
        status=status,
        reason=reason,
        metrics={
            "count": len(losses),
            "first": first,
            "final": final,
            "relative_improvement": improvement,
            "best": min(losses),
        },
    )


def _validation_loss_gate(training_report: dict[str, Any] | None) -> EvalGate:
    if training_report is None:
        return EvalGate(name="validation_loss", status="warn", reason="no training report provided")
    losses = training_report.get("validation_losses", [])
    if not losses:
        validate_every = int(training_report.get("validate_every", 0) or 0)
        steps = int(training_report.get("steps", 0) or 0)
        if validate_every <= 0:
            reason = "validation was disabled for this training run"
            recommendation = (
                "set --validate-every to a positive value for checkpoint quality evaluation"
            )
        elif steps < validate_every:
            reason = "no validation interval was reached during this training run"
            recommendation = "set --validate-every less than or equal to --train-steps"
        else:
            reason = "no validation batches were produced; increase corpus size or validation_fraction for quality evaluation"
            recommendation = (
                "increase corpus size or validation_fraction so validation loss is measured"
            )
        return EvalGate(
            name="validation_loss",
            status="warn",
            reason=reason,
            metrics={
                "steps": steps,
                "validate_every": validate_every,
                "recommendation": recommendation,
            },
        )
    values = [float(item["loss"]) for item in losses]
    finite = all(math.isfinite(item) for item in values)
    if not finite:
        return EvalGate(
            name="validation_loss", status="fail", reason="validation loss contains NaN or infinity"
        )
    first = values[0]
    final = values[-1]
    status = "pass" if len(values) == 1 or final <= first * 1.5 else "fail"
    reason = (
        "validation loss is finite and non-exploding"
        if status == "pass"
        else "validation loss exploded"
    )
    return EvalGate(
        name="validation_loss",
        status=status,
        reason=reason,
        metrics={"count": len(values), "first": first, "final": final, "best": min(values)},
    )


def _checkpoint_selection_gate(
    training_report: dict[str, Any] | None, checkpoint_manifest_path: Path
) -> EvalGate:
    if training_report is None:
        return EvalGate(
            name="checkpoint_selection", status="warn", reason="no training report provided"
        )
    best_manifest = str(training_report.get("best_checkpoint_manifest") or "")
    final_manifest = str(training_report.get("checkpoint_manifest") or "")
    best_loss = float(training_report.get("best_validation_loss", -1.0))
    best_step = int(training_report.get("best_validation_step", 0) or 0)
    selected = str(checkpoint_manifest_path)
    status = "pass" if best_manifest and selected == best_manifest else "warn"
    reason = (
        "validation-best checkpoint selected"
        if status == "pass"
        else "final or unknown checkpoint selected"
    )
    return EvalGate(
        name="checkpoint_selection",
        status=status,
        reason=reason,
        metrics={
            "selected_manifest": selected,
            "best_manifest": best_manifest,
            "final_manifest": final_manifest,
            "best_validation_loss": best_loss,
            "best_validation_step": best_step,
        },
    )


def evaluate_checkpoint(
    *,
    checkpoint_manifest_path: str | Path,
    training_report: str | Path | dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/craftly/checkpoint-eval",
) -> CheckpointEvalReport:
    manifest_path = Path(checkpoint_manifest_path)
    manifest = CheckpointManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    )
    active_training_report = _load_training_report(training_report)

    benchmark_dir = Path(output_dir) / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_report: BenchmarkRunReport = BenchmarkHarness().run_static(built_in_security_tasks())
    benchmark_report.write_markdown(benchmark_dir / "built_in_security_benchmark.md")

    gates = [
        _checkpoint_integrity_gate(manifest),
        _checkpoint_selection_gate(active_training_report, manifest_path),
        _training_loss_gate(active_training_report),
        _validation_loss_gate(active_training_report),
        EvalGate(
            name="built_in_security_benchmark",
            status="pass" if benchmark_report.status == "passed" else "fail",
            reason="built-in defensive security benchmark passed"
            if benchmark_report.status == "passed"
            else "built-in defensive benchmark failed",
            metrics={
                "score": benchmark_report.score,
                "total": benchmark_report.total,
                "passed": benchmark_report.passed,
            },
        ),
    ]
    blocking = [gate for gate in gates if gate.status == "fail"]
    recommendations: list[str] = []
    if any(gate.name == "validation_loss" and gate.status == "warn" for gate in gates):
        recommendations.append(
            "increase corpus size or validation_fraction so validation loss is measured"
        )
    if any(gate.name == "training_loss" and gate.status == "fail" for gate in gates):
        recommendations.append(
            "reduce learning rate, inspect data quality, or lower batch/sequence settings"
        )
    if any(gate.name == "checkpoint_integrity" and gate.status == "fail" for gate in gates):
        recommendations.append(
            "treat checkpoint as invalid and rerun training from a verified manifest"
        )

    report = CheckpointEvalReport(
        status="passed" if not blocking else "failed",
        checkpoint_manifest=str(manifest_path),
        output_dir=str(output_dir),
        gates=gates,
        benchmark_report=benchmark_report.model_dump(),
        recommendations=recommendations,
    )
    report.write(output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Craftly scratch training checkpoint.")
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--training-report")
    parser.add_argument("--output-dir", default="artifacts/craftly/checkpoint-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_checkpoint(
        checkpoint_manifest_path=args.checkpoint_manifest,
        training_report=args.training_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


if __name__ == "__main__":
    main()
