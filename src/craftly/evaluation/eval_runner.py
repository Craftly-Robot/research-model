"""Config-driven deterministic checkpoint evaluation runner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.evaluation.benchmarks import (
    BenchmarkHarness,
    BenchmarkRunReport,
    BenchmarkTask,
    built_in_security_tasks,
)
from src.craftly.model_ops.checkpoint_compare import (
    DEFAULT_PROMPTS,
    GenerationConfig,
    PromptCase,
    _load_model,
    _score_output,
    generate_text,
)
from src.craftly.model_ops.foundation import CheckpointManifest
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import select_torch_device
from src.craftly.shared.config_contracts import (
    EvalBenchmarkContract as EvalBenchmarkSpec,
)
from src.craftly.shared.config_contracts import (
    EvalScheduleContract as EvalSchedule,
    load_eval_schedule_contract,
)
from src.craftly.shared.schemas import EvaluationGate as RegressionFlag, StrictModel, print_report

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class EvalBenchmarkResult(StrictModel):
    name: str
    kind: str
    status: str
    score: float = Field(ge=0.0, le=1.0)
    total: int = 0
    passed: int = 0
    category: str = "general"
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class EvalRunReport(StrictModel):
    status: str
    checkpoint_manifest: str
    checkpoint_step: int
    trained_tokens: int
    output_dir: str
    benchmarks: list[EvalBenchmarkResult]
    aggregate_scores: dict[str, float]
    regression_flags: list[RegressionFlag]
    recommendations: list[str]
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "eval_report.json"
        json_path.write_text(
            json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
        )
        write_markdown_report(self, root / "eval_report.md")
        return json_path


def load_schedule(path: str | Path) -> EvalSchedule:
    return load_eval_schedule_contract(path)


def _load_manifest(path: str | Path) -> CheckpointManifest:
    return CheckpointManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _load_prompt_cases(path: str | Path | None, *, default_category: str) -> list[PromptCase]:
    if path is None:
        return [PromptCase.model_validate(item) for item in DEFAULT_PROMPTS]
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if source.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = (
            payload.get("prompts", payload.get("tasks", payload))
            if isinstance(payload, dict)
            else payload
        )
    cases = []
    for index, row in enumerate(rows):
        cases.append(
            PromptCase(
                task_id=str(row.get("task_id") or row.get("id") or f"case-{index}"),
                category=str(row.get("category") or default_category),
                prompt=str(row.get("prompt") or row.get("question") or ""),
                expected_terms=[
                    str(item) for item in row.get("expected_terms", row.get("expected", []))
                ],
                forbidden_terms=[str(item) for item in row.get("forbidden_terms", [])],
            )
        )
    return cases


def _score_mcq_rows(path: str | Path) -> EvalBenchmarkResult:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    total = 0
    for row in rows:
        total += 1
        prediction = (
            str(row.get("prediction") or row.get("model_answer") or row.get("chosen_option") or "")
            .strip()
            .lower()
        )
        expected = (
            str(row.get("answer") or row.get("gold") or row.get("expected") or "").strip().lower()
        )
        if prediction and expected and prediction == expected:
            passed += 1
    score = passed / max(1, total)
    return EvalBenchmarkResult(
        name=Path(path).stem,
        kind="mcq_jsonl",
        status="passed" if passed == total else "failed",
        score=round(score, 6),
        total=total,
        passed=passed,
        category="general",
        reason="MCQ predictions compared against expected answers",
    )


def _score_generation_cases(
    *,
    checkpoint_manifest: str | Path,
    tokenizer_path: str | Path,
    cases: list[PromptCase],
    generation: GenerationConfig,
    device: str,
    name: str,
    kind: str,
    category: str,
) -> EvalBenchmarkResult:
    selected = select_torch_device(device)
    model, _manifest = _load_model(checkpoint_manifest, device=selected)
    tokenizer = load_tokenizer(tokenizer_path)
    scores = []
    details = []
    for case in cases:
        output, token_count = generate_text(
            model=model, tokenizer=tokenizer, prompt=case.prompt, device=selected, config=generation
        )
        score, expected_hits, missing, forbidden_hits, repetition = _score_output(output, case)
        scores.append(score)
        details.append(
            {
                "task_id": case.task_id,
                "category": case.category,
                "score": score,
                "expected_hits": expected_hits,
                "missing_expected_terms": missing,
                "forbidden_hits": forbidden_hits,
                "repetition_ratio": repetition,
                "token_count": token_count,
            }
        )
    average = sum(scores) / max(1, len(scores))
    passed = sum(1 for item in scores if item >= 0.60)
    return EvalBenchmarkResult(
        name=name,
        kind=kind,
        status="passed" if average >= 0.60 else "failed",
        score=round(average, 6),
        total=len(cases),
        passed=passed,
        category=category,
        reason="deterministic generation suite scored from expected/forbidden terms",
        details={"results": details},
    )


def _run_benchmark(
    *,
    spec: EvalBenchmarkSpec,
    checkpoint_manifest: str | Path,
    tokenizer_path: str | Path | None,
    generation: GenerationConfig,
    device: str,
) -> EvalBenchmarkResult:
    if spec.path and not Path(spec.path).exists():
        status = "failed" if spec.required else "skipped"
        return EvalBenchmarkResult(
            name=spec.name,
            kind=spec.kind,
            status=status,
            score=0.0,
            category=spec.category,
            reason=f"benchmark file missing: {spec.path}",
        )
    if spec.kind == "built_in_security":
        report: BenchmarkRunReport = BenchmarkHarness().run_static(built_in_security_tasks())
        return EvalBenchmarkResult(
            name=spec.name,
            kind=spec.kind,
            status="passed" if report.status == "passed" else "failed",
            score=report.score,
            total=report.total,
            passed=report.passed,
            category="domain",
            reason="built-in defensive security benchmark",
            details=report.model_dump(),
        )
    if spec.kind == "static_jsonl":
        tasks = [
            BenchmarkTask.model_validate(json.loads(line))
            for line in Path(str(spec.path)).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = BenchmarkHarness().run_static(tasks)
        return EvalBenchmarkResult(
            name=spec.name,
            kind=spec.kind,
            status="passed" if report.status == "passed" else "failed",
            score=report.score,
            total=report.total,
            passed=report.passed,
            category=spec.category,
            reason="static JSONL benchmark",
            details=report.model_dump(),
        )
    if spec.kind == "mcq_jsonl":
        return _score_mcq_rows(str(spec.path))
    if spec.kind in {"generation_suite", "jsonl_generation"}:
        if tokenizer_path is None:
            return EvalBenchmarkResult(
                name=spec.name,
                kind=spec.kind,
                status="failed",
                score=0.0,
                category=spec.category,
                reason="tokenizer_path is required for generation benchmarks",
            )
        cases = _load_prompt_cases(spec.path, default_category=spec.category)
        return _score_generation_cases(
            checkpoint_manifest=checkpoint_manifest,
            tokenizer_path=tokenizer_path,
            cases=cases,
            generation=generation,
            device=device,
            name=spec.name,
            kind=spec.kind,
            category=spec.category,
        )
    raise ValueError(f"unsupported benchmark kind: {spec.kind}")


def aggregate_scores(results: list[EvalBenchmarkResult]) -> dict[str, float]:
    active = [item for item in results if item.status != "skipped"]
    by_category: dict[str, list[float]] = {}
    for item in active:
        by_category.setdefault(item.category, []).append(item.score)
    payload = {"overall": round(sum(item.score for item in active) / max(1, len(active)), 6)}
    for category, values in by_category.items():
        payload[category] = round(sum(values) / max(1, len(values)), 6)
    return payload


def regression_flags(
    *,
    current: dict[str, float],
    previous_report_path: str | Path | None,
    warn_threshold: float,
    fail_threshold: float,
) -> list[RegressionFlag]:
    if not previous_report_path:
        return [
            RegressionFlag(
                name="baseline", status="pass", reason="no previous eval report supplied"
            )
        ]
    source = Path(previous_report_path)
    if not source.exists():
        return [
            RegressionFlag(
                name="baseline", status="warn", reason=f"previous eval report missing: {source}"
            )
        ]
    previous = EvalRunReport.model_validate(json.loads(source.read_text(encoding="utf-8")))
    flags: list[RegressionFlag] = []
    for key in sorted(set(previous.aggregate_scores) | set(current)):
        old = float(previous.aggregate_scores.get(key, 0.0))
        new = float(current.get(key, 0.0))
        drop = old - new
        status = "fail" if drop > fail_threshold else "warn" if drop > warn_threshold else "pass"
        flags.append(
            RegressionFlag(
                name=f"regression_{key}",
                status=status,
                reason=f"{key} score delta {new - old:+.4f}",
                metrics={"previous": old, "current": new, "drop": drop},
            )
        )
    return flags


def evaluate_checkpoint_with_schedule(
    *,
    checkpoint_manifest: str | Path,
    schedule_path: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    previous_report: str | Path | None = None,
    device: str = "auto",
) -> EvalRunReport:
    schedule = load_schedule(schedule_path)
    manifest = _load_manifest(checkpoint_manifest)
    generation = GenerationConfig(
        max_new_tokens=schedule.max_new_tokens,
        temperature=schedule.temperature,
        seed=schedule.seed,
    )
    benchmarks = [
        _run_benchmark(
            spec=spec,
            checkpoint_manifest=checkpoint_manifest,
            tokenizer_path=tokenizer_path,
            generation=generation,
            device=device,
        )
        for spec in schedule.benchmarks
    ]
    aggregates = aggregate_scores(benchmarks)
    flags = regression_flags(
        current=aggregates,
        previous_report_path=previous_report,
        warn_threshold=schedule.regression_threshold_warn,
        fail_threshold=schedule.regression_threshold_fail,
    )
    failures = [item for item in benchmarks if item.status == "failed"] + [
        item for item in flags if item.status == "fail"
    ]
    recommendations: list[str] = []
    if any(item.status == "skipped" for item in benchmarks):
        recommendations.append("add local benchmark files for skipped optional tasks")
    if any(item.status == "failed" for item in benchmarks):
        recommendations.append("inspect failed benchmark categories before promoting checkpoint")
    if any(item.status == "fail" for item in flags):
        recommendations.append(
            "catastrophic regression detected; keep previous checkpoint as promotion candidate"
        )
    report = EvalRunReport(
        status="failed" if failures else "passed",
        checkpoint_manifest=str(checkpoint_manifest),
        checkpoint_step=manifest.step,
        trained_tokens=manifest.trained_tokens,
        output_dir=str(output_dir),
        benchmarks=benchmarks,
        aggregate_scores=aggregates,
        regression_flags=flags,
        recommendations=recommendations,
    )
    report.write(output_dir)
    return report


def write_markdown_report(report: EvalRunReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Checkpoint Eval Report",
        "",
        f"- status: {report.status}",
        f"- checkpoint: `{report.checkpoint_manifest}`",
        f"- step: {report.checkpoint_step}",
        f"- trained_tokens: {report.trained_tokens}",
        "",
        "## Aggregate Scores",
        "",
        "| metric | score |",
        "|---|---:|",
    ]
    for key, value in sorted(report.aggregate_scores.items()):
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Benchmarks",
            "",
            "| name | kind | status | score | reason |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in report.benchmarks:
        lines.append(
            f"| {item.name} | {item.kind} | {item.status} | {item.score:.4f} | {item.reason} |"
        )
    lines.extend(["", "## Regression Flags", "", "| name | status | reason |", "|---|---|---|"])
    for item in report.regression_flags:
        lines.append(f"| {item.name} | {item.status} | {item.reason} |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Craftly checkpoint evaluation.")
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--schedule", default="config/eval_schedule.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--previous-report")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = evaluate_checkpoint_with_schedule(
        checkpoint_manifest=args.checkpoint_manifest,
        schedule_path=args.schedule,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        previous_report=args.previous_report,
        device=args.device,
    )
    print_report(report)
    if report.status == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
