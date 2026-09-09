"""End-to-end Craftly data pipeline: crawl -> clean -> shard -> train."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field

from src.craftly.evaluation.checkpoint_eval import CheckpointEvalReport, evaluate_checkpoint
from src.craftly.learning.benchmark_contamination_filter import (
    BenchmarkContaminationFilterReport,
    ContaminationDetector,
    filter_benchmark_contamination_jsonl,
    load_patterns,
)
from src.craftly.learning.dashboard import write_dashboard
from src.craftly.learning.data_engine import (
    DataEngine,
    DataEngineConfig,
    FrontierStore,
    PostgresFrontierStore,
)
from src.craftly.learning.feedback import BenchmarkFeedbackReport, write_feedback_report
from src.craftly.learning.license_filter import LicenseFilterReport, filter_jsonl_by_license
from src.craftly.learning.mixer import (
    ScratchInstructionMixReport,
    ScratchMixConfig,
    build_scratch_instruction_mix,
)
from src.craftly.learning.near_dedup import NearDedupReport, deduplicate_jsonl
from src.craftly.learning.production_dataset import validate_dataset_manifest_for_promotion
from src.craftly.learning.quality_inspector import QualityInspectionReport, write_quality_report
from src.craftly.learning.review import ReviewReport, review_tasks
from src.craftly.learning.source_balancing import SourceBalanceReport, balance_clean_jsonl
from src.craftly.learning.source_budget import SourceBudgetPlan, write_source_budget_plan
from src.craftly.learning.source_quality import SourceQualityReport, write_source_quality_report
from src.craftly.learning.source_registry import SourceRegistry, SourceRegistryReport
from src.craftly.learning.source_reputation import (
    SourceReputationReport,
    write_source_reputation_report,
)
from src.craftly.learning.storage import ObjectStoreConfig, create_object_store, upload_paths
from src.craftly.learning.task_extraction import TaskExtractionReport, extract_tasks
from src.craftly.learning.training_data_gate import (
    TrainingDataGateConfig,
    TrainingDataGateReport,
    apply_training_data_gate,
)
from src.craftly.learning.versioning import (
    DatasetArtifact,
    DatasetLedger,
    DatasetVersionManifest,
    artifact_from_path,
    build_version_id,
)
from src.craftly.model_ops.checkpoint_compare import (
    CheckpointComparisonReport,
    GenerationConfig,
    compare_checkpoints,
)
from src.craftly.model_ops.learning_validation import TokenDominance, audit_tokenizer_dominance
from src.craftly.model_ops.pretrain_loop import run_pretraining_loop
from src.craftly.model_ops.tokenizer_pipeline import (
    ShardBuildConfig,
    ShardManifest,
    TokenizerTrainConfig,
    build_token_shards,
    train_bpe_tokenizer,
)
from src.craftly.shared.progress import ProgressReporter, progress_from_options
from src.craftly.shared.schemas import StrictModel


class DataPipelineConfig(StrictModel):
    sources_path: str
    dataset_id: str = "craftly-defensive-coding-corpus"
    work_dir: str = "artifacts/craftly/data-pipeline"
    frontier_backend: str = "sqlite"
    postgres_dsn: str | None = None
    max_docs: int = Field(default=10_000, ge=1)
    max_bytes_per_doc: int = Field(default=2_000_000, ge=1000)
    workers: int = Field(default=8, ge=1, le=256)
    max_depth: int = Field(default=2, ge=0, le=20)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    shard_rows: int = Field(default=10_000, ge=1)
    respect_robots: bool = True
    vocab_size: int = Field(default=128_000, ge=1_000)
    tokenizer_min_frequency: int = Field(default=2, ge=1)
    shard_token_count: int = Field(default=1_000_000, ge=128)
    sequence_length: int = Field(default=2048, ge=16)
    validation_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    skip_train: bool = False
    train_steps: int = Field(default=100, ge=1)
    train_device: str = "auto"
    train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    optimizer_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    optimizer_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    optimizer_epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    learning_rate_schedule: str = Field(default="cosine", pattern="^(constant|linear|cosine)$")
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    minimum_learning_rate_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    target_tokens: int | None = Field(default=None, gt=0)
    dtype: str = "bf16"
    model_profile_name: str = "tiny"
    attention_impl: str = "auto"
    gradient_checkpointing: bool = False
    validate_every: int = Field(default=25, ge=0)
    validation_batches: int = Field(default=4, ge=1)
    checkpoint_every: int = Field(default=50, ge=1)
    run_checkpoint_eval: bool = True
    early_stopping_patience: int = Field(default=0, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    filter_licenses: bool = True
    strict_unknown_licenses: bool = True
    filter_benchmark_contamination: bool = True
    near_dedup: bool = True
    near_dedup_hamming_threshold: int = Field(default=3, ge=0, le=16)
    build_source_reputation: bool = True
    build_source_budget: bool = True
    source_budget_target_docs: int | None = Field(default=None, ge=1)
    apply_training_data_gate: bool = True
    min_training_quality_score: float = Field(default=0.58, ge=0.0, le=1.0)
    min_training_average_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_training_rows: int = Field(default=1, ge=1)
    min_train_tokens: int = Field(default=128, ge=1)
    min_source_reputation_score: float = Field(default=0.45, ge=0.0, le=1.0)
    eval_holdout_fraction: float = Field(default=0.02, ge=0.0, le=0.5)
    balance_sources: bool = True
    max_source_fraction: float = Field(default=0.35, ge=0.05, le=1.0)
    min_source_rows: int = Field(default=25, ge=1)
    instruction_mix: bool = True
    instruction_mix_max_rows: int | None = Field(default=None, ge=1)
    curriculum_mode: str = Field(
        default="balanced",
        pattern="^(balanced|fundamentals_only|defensive_security_only|debug_patch_only|agentic_coding_only)$",
    )
    strict_offensive_filter: bool = True
    contamination_patterns_path: str | None = None
    block_contamination: bool = True
    extract_tasks: bool = True
    review_tasks: bool = True
    max_extracted_tasks: int = Field(default=50_000, ge=1)
    max_tasks_per_source_row: int = Field(default=3, ge=1, le=20)
    object_store_uri: str = "local://artifacts/craftly/object-store"
    object_store_endpoint_url: str | None = None
    upload_artifacts: bool = True
    checkpoint_compare_prompt_suite: str | None = None
    checkpoint_compare_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    checkpoint_compare_max_new_tokens: int = Field(default=96, ge=1, le=2048)
    checkpoint_compare_repetition_penalty: float = Field(default=1.12, ge=1.0, le=5.0)
    checkpoint_compare_no_repeat_ngram_size: int = Field(default=4, ge=0, le=20)
    checkpoint_compare_max_repetition_ratio: float = Field(default=0.72, ge=0.0, le=1.0)
    progress_path: str | None = None
    progress_to_stdout: bool = False
    progress_every_docs: int = Field(default=25, ge=1)
    progress_every_steps: int = Field(default=25, ge=1)
    dataloader_prefetch_batches: int = Field(default=4, ge=0, le=128)
    dataloader_seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    expected_python_version: str | None = None
    expected_pytorch_version: str | None = None
    expected_cuda_version: str | None = None
    production_mode: bool = False
    dev_smoke: bool = False
    promoted_dataset_manifest_path: str | None = None
    dataset_trust_policy_path: str = "config/dataset_trust_policy.json"


class DataPipelineReport(StrictModel):
    status: str
    dataset_id: str
    version_id: str
    work_dir: str
    source_registry: SourceRegistryReport
    crawl: dict[str, Any]
    license_filter_report: dict[str, Any] | None = None
    benchmark_contamination_filter_report: dict[str, Any] | None = None
    near_dedup_report: dict[str, Any] | None = None
    contamination_report: dict[str, Any] | None
    quality_report: dict[str, Any] | None
    training_quality_report: dict[str, Any] | None = None
    source_quality_report: dict[str, Any] | None
    task_report: dict[str, Any] | None
    review_report: dict[str, Any] | None
    feedback_report: dict[str, Any] | None
    source_reputation_report: dict[str, Any] | None = None
    source_budget_plan: dict[str, Any] | None = None
    training_data_gate_report: dict[str, Any] | None = None
    source_balance_report: dict[str, Any] | None = None
    instruction_mix_report: dict[str, Any] | None = None
    clean_files: list[str]
    training_files: list[str]
    tokenizer_path: str
    shard_manifest: dict[str, Any]
    version_manifest_path: str
    dashboard_path: str
    uploaded_objects: list[dict[str, Any]]
    training: dict[str, Any] | None = None
    checkpoint_eval: dict[str, Any] | None = None
    checkpoint_comparison: dict[str, Any] | None = None
    tokenizer_audit_report: dict[str, Any] | None = None
    report_artifacts: dict[str, str] = Field(default_factory=dict)


class PipelineRunLock:
    def __init__(self, path: str | Path, *, stale_after_seconds: int = 24 * 60 * 60) -> None:
        self.path = Path(path)
        self.stale_after_seconds = stale_after_seconds
        self.fd: int | None = None

    def __enter__(self) -> PipelineRunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        if self.path.exists() and now - self.path.stat().st_mtime > self.stale_after_seconds:
            self.path.unlink()
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = (
                self.path.read_text(encoding="utf-8", errors="replace")
                if self.path.exists()
                else ""
            )
            raise RuntimeError(
                f"data pipeline work_dir is already locked: {self.path}. "
                "Use a fresh --output-dir or wait for the running job to finish. "
                f"lock_details={details!r}"
            ) from exc
        payload = json.dumps({"pid": os.getpid(), "created_at_unix": now}, sort_keys=True) + "\n"
        os.write(self.fd, payload.encode("utf-8"))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


async def _build_store(config: DataPipelineConfig) -> Any:
    if config.frontier_backend == "postgres":
        if not config.postgres_dsn:
            raise ValueError("postgres_dsn is required when frontier_backend='postgres'")
        return await PostgresFrontierStore.create(config.postgres_dsn)
    if config.frontier_backend != "sqlite":
        raise ValueError("frontier_backend must be 'sqlite' or 'postgres'")
    return FrontierStore(Path(config.work_dir) / "frontier.sqlite3")


async def run_data_pipeline(
    config: DataPipelineConfig, *, client: httpx.AsyncClient | None = None
) -> DataPipelineReport:
    validate_data_pipeline_production_config(config)
    root = Path(config.work_dir)
    raw_dir = root / "raw"
    clean_dir = root / "clean"
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    shards_dir = root / "shards"
    train_dir = root / "train"
    tasks_path = root / "tasks" / "tasks.jsonl"
    reports_dir = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_path = config.progress_path or str(root / "progress.jsonl")
    progress = progress_from_options(path=progress_path, to_stdout=config.progress_to_stdout)

    with PipelineRunLock(root / ".pipeline.lock"):
        return await _run_data_pipeline_locked(
            config,
            client=client,
            root=root,
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            tokenizer_path=tokenizer_path,
            shards_dir=shards_dir,
            train_dir=train_dir,
            tasks_path=tasks_path,
            reports_dir=reports_dir,
            progress=progress,
        )


def validate_data_pipeline_production_config(config: DataPipelineConfig) -> None:
    if not config.production_mode:
        return
    failures = []
    if config.frontier_backend != "postgres":
        failures.append("production mode requires frontier_backend='postgres'")
    if not config.postgres_dsn:
        failures.append("production mode requires postgres_dsn")
    if config.object_store_uri.startswith("local://"):
        failures.append("production mode requires S3/MinIO object storage, not local://")
    if config.skip_train:
        failures.append("production mode cannot skip training")
    if config.model_profile_name == "tiny" and not config.dev_smoke:
        failures.append("production mode cannot use model_profile_name='tiny' without dev_smoke")
    if not config.filter_licenses or not config.strict_unknown_licenses:
        failures.append("production mode requires strict license filtering")
    if not config.filter_benchmark_contamination:
        failures.append("production mode requires benchmark contamination filtering")
    if not config.near_dedup:
        failures.append("production mode requires near-duplicate removal")
    if not config.promoted_dataset_manifest_path:
        failures.append("production mode requires a V2 promoted_dataset_manifest_path")
    if not config.run_checkpoint_eval:
        failures.append("production mode requires checkpoint evaluation")
    if not config.checkpoint_compare_prompt_suite:
        failures.append("production mode requires checkpoint_compare_prompt_suite")
    if config.validate_every <= 0 or config.validate_every > config.train_steps:
        failures.append("production mode requires validation during the training run")
    if config.min_training_average_quality_score < 0.60:
        failures.append("production mode requires min_training_average_quality_score >= 0.60")
    if config.min_training_rows < 10_000 and not config.dev_smoke:
        failures.append("production mode requires min_training_rows >= 10000")
    if config.min_train_tokens < 1_000_000 and not config.dev_smoke:
        failures.append("production mode requires min_train_tokens >= 1000000")
    if config.promoted_dataset_manifest_path:
        try:
            validate_dataset_manifest_for_promotion(
                config.promoted_dataset_manifest_path,
                trust_policy_path=config.dataset_trust_policy_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"promoted dataset manifest failed trust validation: {exc}")
    if failures:
        raise ValueError("production data pipeline validation failed: " + "; ".join(failures))


async def _run_data_pipeline_locked(
    config: DataPipelineConfig,
    *,
    client: httpx.AsyncClient | None,
    root: Path,
    raw_dir: Path,
    clean_dir: Path,
    tokenizer_path: Path,
    shards_dir: Path,
    train_dir: Path,
    tasks_path: Path,
    reports_dir: Path,
    progress: ProgressReporter,
) -> DataPipelineReport:
    progress.emit(
        "pipeline",
        "started",
        dataset_id=config.dataset_id,
        work_dir=str(root),
        sources_path=config.sources_path,
        max_docs=config.max_docs,
        train_steps=config.train_steps,
        skip_train=config.skip_train,
    )
    registry = SourceRegistry.from_file(config.sources_path)
    registry_report = registry.validate()
    progress.emit(
        "source_registry",
        "complete",
        source_count=registry_report.source_count,
        url_count=registry_report.url_count,
        warnings=len(registry_report.warnings),
    )
    store = await _build_store(config)
    engine_config = DataEngineConfig(
        frontier_backend=config.frontier_backend,
        postgres_dsn=config.postgres_dsn,
        frontier_path=str(root / "frontier.sqlite3"),
        output_dir=str(raw_dir),
        clean_output_dir=str(clean_dir),
        max_docs=config.max_docs,
        max_bytes_per_doc=config.max_bytes_per_doc,
        max_depth=config.max_depth,
        workers=config.workers,
        shard_rows=config.shard_rows,
        respect_robots=config.respect_robots,
        delay_seconds=config.delay_seconds,
    )
    engine = DataEngine(engine_config, store=store, owns_store=True)
    try:
        crawl_report = await engine.run(
            registry.to_sources(),
            client=client,
            progress=progress,
            progress_every_docs=config.progress_every_docs,
        )
    finally:
        await engine.aclose()

    clean_files = sorted(str(path) for path in clean_dir.glob("clean-*.jsonl"))
    if not clean_files:
        raise RuntimeError(
            "crawler produced zero clean shards; check sources, quality gate, robots policy, and allowlist"
        )

    license_filter_report: LicenseFilterReport | None = None
    license_filter_path = root / "filtered" / "license-clean-000000.jsonl"
    if config.filter_licenses:
        progress.emit("license_filter", "started", input_files=len(clean_files))
        license_filter_report = filter_jsonl_by_license(
            clean_files,
            license_filter_path,
            strict_unknown=config.strict_unknown_licenses,
        )
        progress.emit(
            "license_filter",
            "complete",
            accepted=license_filter_report.accepted,
            rejected=license_filter_report.rejected,
        )
        (reports_dir / "license_filter_report.json").write_text(
            json.dumps(license_filter_report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        clean_files = [str(license_filter_path)]
        if license_filter_report.accepted == 0:
            raise RuntimeError(
                "license filter rejected every row; review source licenses or disable strict unknown handling"
            )

    benchmark_filter_report: BenchmarkContaminationFilterReport | None = None
    benchmark_filter_path = root / "filtered" / "benchmark-clean-000000.jsonl"
    if config.filter_benchmark_contamination:
        progress.emit("benchmark_contamination_filter", "started", input_files=len(clean_files))
        benchmark_filter_report = filter_benchmark_contamination_jsonl(
            clean_files, benchmark_filter_path
        )
        progress.emit(
            "benchmark_contamination_filter",
            "complete",
            accepted=benchmark_filter_report.accepted,
            rejected=benchmark_filter_report.rejected,
        )
        (reports_dir / "benchmark_contamination_filter_report.json").write_text(
            json.dumps(benchmark_filter_report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        clean_files = [str(benchmark_filter_path)]
        if benchmark_filter_report.accepted == 0:
            raise RuntimeError("benchmark contamination filter rejected every row")

    near_dedup_report: NearDedupReport | None = None
    near_dedup_path = root / "dedup" / "dedup-clean-000000.jsonl"
    if config.near_dedup:
        progress.emit("near_dedup", "started", input_files=len(clean_files))
        near_dedup_report = deduplicate_jsonl(
            clean_files,
            near_dedup_path,
            hamming_threshold=config.near_dedup_hamming_threshold,
        )
        progress.emit(
            "near_dedup",
            "complete",
            accepted=near_dedup_report.accepted,
            exact_duplicates=near_dedup_report.exact_duplicates,
            near_duplicates=near_dedup_report.near_duplicates,
        )
        (reports_dir / "near_dedup_report.json").write_text(
            json.dumps(near_dedup_report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        clean_files = [str(near_dedup_path)]
        if near_dedup_report.accepted == 0:
            raise RuntimeError(
                "deduplication rejected every row; lower --near-dedup-hamming-threshold"
            )

    progress.emit("contamination_scan", "started", input_files=len(clean_files))
    contamination_report = ContaminationDetector(
        load_patterns(config.contamination_patterns_path)
    ).scan_jsonl(
        clean_files,
        block_on_hit=config.block_contamination,
    )
    progress.emit(
        "contamination_scan",
        "complete",
        blocked=contamination_report.blocked,
        hits=len(contamination_report.hits),
    )
    contamination_path = reports_dir / "contamination_report.json"
    contamination_path.parent.mkdir(parents=True, exist_ok=True)
    contamination_path.write_text(
        json.dumps(contamination_report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    if contamination_report.blocked:
        raise RuntimeError(
            f"contamination detector blocked dataset: {len(contamination_report.hits)} hits"
        )

    quality_report_path = reports_dir / "quality_report.json"
    progress.emit("quality_inspection", "started", input_files=len(clean_files))
    quality_report: QualityInspectionReport = write_quality_report(clean_files, quality_report_path)
    source_quality_path = reports_dir / "source_quality_report.json"
    source_quality_report: SourceQualityReport = write_source_quality_report(
        clean_files, source_quality_path
    )
    progress.emit(
        "quality_inspection",
        "complete",
        rows=quality_report.rows,
        avg_quality_score=quality_report.avg_quality_score,
        sources=len(source_quality_report.sources),
    )

    task_report: TaskExtractionReport | None = None
    review_report: ReviewReport | None = None
    automated_pass_tasks_path = root / "tasks" / "automated_pass_tasks.jsonl"
    task_report_path = reports_dir / "task_extraction_report.json"
    if config.extract_tasks:
        progress.emit(
            "task_extraction",
            "started",
            max_tasks=config.max_extracted_tasks,
            max_tasks_per_source_row=config.max_tasks_per_source_row,
        )
        task_report = extract_tasks(
            clean_files,
            tasks_path,
            max_tasks=config.max_extracted_tasks,
            max_tasks_per_row=config.max_tasks_per_source_row,
        )
        task_report_path.write_text(
            json.dumps(task_report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
        )
        progress.emit(
            "task_extraction",
            "complete",
            extracted=task_report.extracted,
            scanned_rows=task_report.scanned_rows,
            capped_rows=task_report.capped_rows,
            average_tasks_per_row=task_report.average_tasks_per_row,
        )
        if config.review_tasks:
            progress.emit("task_review", "started")
            review_report = review_tasks(
                tasks_path,
                reports_dir / "task_review_decisions.jsonl",
                automated_pass_tasks_path,
            )
            (reports_dir / "task_review_report.json").write_text(
                json.dumps(review_report.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
            )
            progress.emit(
                "task_review",
                "complete",
                automated_pass=review_report.automated_pass,
                human_approved=review_report.human_approved,
                rejected=review_report.rejected,
            )
    feedback_path = reports_dir / "feedback_report.json"
    review_report_path = (
        reports_dir / "task_review_report.json" if review_report is not None else None
    )
    feedback_report: BenchmarkFeedbackReport = write_feedback_report(
        output_path=feedback_path,
        quality_report_path=quality_report_path,
        review_report_path=review_report_path,
    )
    progress.emit(
        "feedback_report", "complete", recommendations=len(feedback_report.recommendations)
    )
    source_reputation_report: SourceReputationReport | None = None
    source_budget_plan: SourceBudgetPlan | None = None
    source_reputation_path = reports_dir / "source_reputation_report.json"
    source_budget_path = reports_dir / "source_budget_plan.json"
    if config.build_source_reputation:
        progress.emit("source_reputation", "started")
        source_reputation_report = write_source_reputation_report(
            source_reputation_path,
            source_quality_report_path=source_quality_path,
            task_report_path=task_report_path if task_report is not None else None,
            review_report_path=review_report_path,
            feedback_report_path=feedback_path,
            contamination_report_path=contamination_path,
            dedup_report_path=reports_dir / "near_dedup_report.json"
            if near_dedup_report is not None
            else None,
        )
        progress.emit(
            "source_reputation", "complete", sources=len(source_reputation_report.sources)
        )
    if config.build_source_budget:
        progress.emit(
            "source_budget",
            "started",
            target_total_docs=config.source_budget_target_docs or config.max_docs,
        )
        source_budget_plan = write_source_budget_plan(
            source_budget_path,
            sources_path=config.sources_path,
            reputation_report_path=source_reputation_path
            if source_reputation_report is not None
            else None,
            target_total_docs=config.source_budget_target_docs or config.max_docs,
        )
        progress.emit(
            "source_budget",
            "complete",
            budgets=len(source_budget_plan.budgets),
            allocated_total_docs=source_budget_plan.allocated_total_docs,
        )
    training_data_gate_report: TrainingDataGateReport | None = None
    gated_train_path = root / "gated" / "training-promoted.jsonl"
    gated_holdout_path = root / "gated" / "eval-holdout.jsonl"
    gated_review_path = root / "gated" / "human-review-queue.jsonl"
    gated_decisions_path = reports_dir / "training_data_gate_decisions.jsonl"
    if config.apply_training_data_gate and not config.production_mode:
        progress.emit(
            "training_data_gate",
            "started",
            min_training_quality_score=config.min_training_quality_score,
            min_source_reputation_score=config.min_source_reputation_score,
            eval_holdout_fraction=config.eval_holdout_fraction,
        )
        training_data_gate_report = apply_training_data_gate(
            input_paths=clean_files,
            promoted_path=gated_train_path,
            holdout_path=gated_holdout_path,
            review_queue_path=gated_review_path,
            decisions_path=gated_decisions_path,
            reputation_report_path=source_reputation_path
            if source_reputation_report is not None
            else None,
            config=TrainingDataGateConfig(
                min_quality_score=config.min_training_quality_score,
                min_source_reputation_score=config.min_source_reputation_score,
                eval_holdout_fraction=config.eval_holdout_fraction,
            ),
        )
        (reports_dir / "training_data_gate_report.json").write_text(
            json.dumps(training_data_gate_report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        progress.emit(
            "training_data_gate",
            "complete",
            promoted=training_data_gate_report.promoted,
            holdout=training_data_gate_report.holdout,
            review_queue=training_data_gate_report.review_queue,
            rejected=training_data_gate_report.rejected,
        )
        if training_data_gate_report.promoted == 0:
            raise RuntimeError(
                "training data gate promoted zero rows; lower thresholds or improve source allowlist"
            )
        clean_files = [str(gated_train_path)]
    source_balance_report: SourceBalanceReport | None = None
    training_files = clean_files
    if config.production_mode:
        governed_manifest = validate_dataset_manifest_for_promotion(
            config.promoted_dataset_manifest_path or "",
            trust_policy_path=config.dataset_trust_policy_path,
        )
        training_files = [governed_manifest.artifacts["train"]]
    balanced_clean_path = root / "balanced" / "balanced-clean-000000.jsonl"
    if config.balance_sources and not config.production_mode:
        progress.emit("source_balancing", "started", max_source_fraction=config.max_source_fraction)
        source_balance_report = balance_clean_jsonl(
            input_paths=clean_files,
            output_path=balanced_clean_path,
            max_source_fraction=config.max_source_fraction,
            min_source_rows=config.min_source_rows,
        )
        progress.emit(
            "source_balancing",
            "complete",
            input_rows=source_balance_report.input_rows,
            output_rows=source_balance_report.output_rows,
            capped_sources=sum(
                1 for item in source_balance_report.sources if item.action == "capped"
            ),
        )
        (reports_dir / "source_balance_report.json").write_text(
            json.dumps(source_balance_report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        training_files = [str(balanced_clean_path)]

    instruction_mix_report: ScratchInstructionMixReport | None = None
    instruction_mix_path = root / "mixed" / "scratch-instruction-mix.jsonl"
    instruction_mix_report_path = reports_dir / "instruction_mix_report.json"
    if config.instruction_mix and not config.production_mode:
        progress.emit(
            "instruction_mix",
            "started",
            input_files=len(training_files),
            instruction_security_coding=0.40,
            verified_patch_tests=0.30,
            high_quality_docs_code=0.20,
            debugging_error_logs=0.10,
        )
        instruction_mix_report = build_scratch_instruction_mix(
            input_paths=[Path(path) for path in training_files],
            output_path=instruction_mix_path,
            report_path=instruction_mix_report_path,
            config=ScratchMixConfig(
                max_rows=config.instruction_mix_max_rows,
                min_quality_score=config.min_training_quality_score,
                curriculum_mode=config.curriculum_mode,
                strict_offensive_filter=config.strict_offensive_filter,
            ),
        )
        if instruction_mix_report.total_rows == 0:
            raise RuntimeError(
                "instruction mix produced zero rows; improve approved source yield before training"
            )
        training_files = [str(instruction_mix_path)]
        progress.emit(
            "instruction_mix",
            "complete",
            mix_status=instruction_mix_report.status,
            rows=instruction_mix_report.total_rows,
            tokens=instruction_mix_report.total_tokens,
            output_jsonl=str(instruction_mix_path),
        )

    training_quality_path = reports_dir / "training_quality_report.json"
    progress.emit("training_quality_inspection", "started", input_files=len(training_files))
    training_quality_report: QualityInspectionReport = write_quality_report(
        training_files, training_quality_path
    )
    progress.emit(
        "training_quality_inspection",
        "complete",
        rows=training_quality_report.rows,
        avg_quality_score=training_quality_report.avg_quality_score,
        minimum_required=config.min_training_average_quality_score,
    )
    if training_quality_report.rows < config.min_training_rows:
        raise RuntimeError(
            f"training corpus rows {training_quality_report.rows} below required minimum {config.min_training_rows}; "
            "increase --max-docs, lower strict gates for validation runs, or add higher-yield approved sources"
        )
    if training_quality_report.avg_quality_score < config.min_training_average_quality_score:
        raise RuntimeError(
            f"training corpus average quality {training_quality_report.avg_quality_score} below required minimum "
            f"{config.min_training_average_quality_score}; raise source quality or lower threshold only for dev validation"
        )

    progress.emit(
        "tokenizer", "started", vocab_size=config.vocab_size, input_files=len(training_files)
    )
    trained_tokenizer = train_bpe_tokenizer(
        training_files,
        tokenizer_path,
        TokenizerTrainConfig(
            vocab_size=config.vocab_size, min_frequency=config.tokenizer_min_frequency
        ),
    )
    progress.emit("tokenizer", "complete", tokenizer_path=str(trained_tokenizer))
    tokenizer_audit_report: TokenDominance = audit_tokenizer_dominance(
        tokenizer_path=trained_tokenizer,
        corpus_path=training_files[0],
        output_path=reports_dir / "tokenizer_audit_report.json",
    )
    progress.emit(
        "tokenizer_audit",
        tokenizer_audit_report.status,
        dot_fraction=tokenizer_audit_report.dot_fraction,
        whitespace_fraction=tokenizer_audit_report.whitespace_fraction,
        single_char_fraction=tokenizer_audit_report.single_char_fraction,
        warnings=len(tokenizer_audit_report.warnings),
    )
    progress.emit(
        "sharding",
        "started",
        shard_token_count=config.shard_token_count,
        sequence_length=config.sequence_length,
    )
    manifest: ShardManifest = build_token_shards(
        input_paths=training_files,
        tokenizer_path=trained_tokenizer,
        output_dir=shards_dir,
        config=ShardBuildConfig(
            shard_token_count=config.shard_token_count,
            sequence_length=config.sequence_length,
            validation_fraction=config.validation_fraction,
        ),
        dataset_id=config.dataset_id,
    )
    progress.emit(
        "sharding",
        "complete",
        train_tokens=manifest.train_tokens,
        val_tokens=manifest.val_tokens,
        train_shards=len(manifest.train_shards),
        val_shards=len(manifest.val_shards),
    )
    if manifest.train_tokens < config.min_train_tokens:
        raise RuntimeError(
            f"training tokens {manifest.train_tokens} below required minimum {config.min_train_tokens}; "
            "increase corpus size, reduce filters only for dev validation, or lower --min-train-tokens for a smoke run"
        )
    training_report = None
    checkpoint_eval_report: CheckpointEvalReport | None = None
    checkpoint_comparison_report: CheckpointComparisonReport | None = None
    if not config.skip_train:
        training_report = run_pretraining_loop(
            output_dir=train_dir,
            manifest=shards_dir / "manifest.json",
            device=config.train_device,
            steps=config.train_steps,
            batch_size=config.train_batch_size,
            sequence_length=config.sequence_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_epsilon=config.optimizer_epsilon,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            learning_rate_schedule=config.learning_rate_schedule,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
            target_tokens=config.target_tokens,
            dtype=config.dtype,
            model_profile_name=config.model_profile_name,
            attention_impl=config.attention_impl,
            gradient_checkpointing=config.gradient_checkpointing,
            checkpoint_every=min(config.checkpoint_every, config.train_steps),
            validate_every=config.validate_every,
            validation_batches=config.validation_batches,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            resume=True,
            progress=progress,
            progress_every_steps=config.progress_every_steps,
            production_mode=config.production_mode,
            dev_smoke=config.dev_smoke,
            dataloader_prefetch_batches=config.dataloader_prefetch_batches,
            dataloader_seed=config.dataloader_seed,
            expected_python_version=config.expected_python_version,
            expected_pytorch_version=config.expected_pytorch_version,
            expected_cuda_version=config.expected_cuda_version,
        )
        if config.run_checkpoint_eval:
            progress.emit("checkpoint_eval", "started")
            checkpoint_eval_report = evaluate_checkpoint(
                checkpoint_manifest_path=training_report.get("best_checkpoint_manifest")
                or training_report["checkpoint_manifest"],
                training_report=training_report,
                output_dir=reports_dir / "checkpoint_eval",
            )
            progress.emit(
                "checkpoint_eval",
                "complete",
                eval_status=checkpoint_eval_report.status,
                gates=len(checkpoint_eval_report.gates),
            )
        if config.checkpoint_compare_prompt_suite:
            progress.emit(
                "checkpoint_comparison",
                "started",
                prompt_suite=config.checkpoint_compare_prompt_suite,
                min_score=config.checkpoint_compare_min_score,
            )
            candidate_manifest = (
                training_report.get("best_checkpoint_manifest")
                or training_report["checkpoint_manifest"]
            )
            checkpoint_comparison_report = compare_checkpoints(
                baseline_manifest=training_report["checkpoint_manifest"],
                candidate_manifest=candidate_manifest,
                tokenizer_path=trained_tokenizer,
                prompt_suite=config.checkpoint_compare_prompt_suite,
                output_dir=reports_dir / "checkpoint_compare",
                device=config.train_device,
                generation_config=GenerationConfig(
                    max_new_tokens=config.checkpoint_compare_max_new_tokens,
                    temperature=0.0,
                    top_k=20,
                    repetition_penalty=config.checkpoint_compare_repetition_penalty,
                    no_repeat_ngram_size=config.checkpoint_compare_no_repeat_ngram_size,
                    max_repetition_ratio=config.checkpoint_compare_max_repetition_ratio,
                ),
            )
            progress.emit(
                "checkpoint_comparison",
                "complete",
                comparison_status=checkpoint_comparison_report.status,
                candidate_average_score=checkpoint_comparison_report.candidate.average_score,
                score_delta=checkpoint_comparison_report.score_delta,
            )
            if checkpoint_comparison_report.status in {
                "regressed",
                "failed_generation_collapse",
                "failed_hallucination_guardrail",
            }:
                raise RuntimeError(
                    f"checkpoint comparison gate failed: {checkpoint_comparison_report.status}"
                )
            if (
                checkpoint_comparison_report.candidate.average_score
                < config.checkpoint_compare_min_score
            ):
                raise RuntimeError(
                    f"checkpoint comparison gate failed: candidate average score "
                    f"{checkpoint_comparison_report.candidate.average_score:.4f} below minimum "
                    f"{config.checkpoint_compare_min_score:.4f}"
                )

    artifacts: list[DatasetArtifact] = []
    for path in clean_files:
        artifacts.append(artifact_from_path(path, role="clean_jsonl"))
    artifacts.append(artifact_from_path(trained_tokenizer, role="tokenizer"))
    artifacts.append(artifact_from_path(shards_dir / "manifest.json", role="shard_manifest"))
    if license_filter_report is not None:
        artifacts.append(
            artifact_from_path(
                reports_dir / "license_filter_report.json", role="license_filter_report"
            )
        )
    if benchmark_filter_report is not None:
        artifacts.append(
            artifact_from_path(
                reports_dir / "benchmark_contamination_filter_report.json",
                role="benchmark_contamination_filter_report",
            )
        )
    if near_dedup_report is not None:
        artifacts.append(
            artifact_from_path(reports_dir / "near_dedup_report.json", role="near_dedup_report")
        )
    artifacts.append(artifact_from_path(contamination_path, role="contamination_report"))
    artifacts.append(artifact_from_path(quality_report_path, role="quality_report"))
    artifacts.append(artifact_from_path(training_quality_path, role="training_quality_report"))
    artifacts.append(
        artifact_from_path(
            reports_dir / "tokenizer_audit_report.json", role="tokenizer_audit_report"
        )
    )
    artifacts.append(artifact_from_path(source_quality_path, role="source_quality_report"))
    artifacts.append(artifact_from_path(feedback_path, role="feedback_report"))
    if source_reputation_report is not None:
        artifacts.append(
            artifact_from_path(source_reputation_path, role="source_reputation_report")
        )
    if source_budget_plan is not None:
        artifacts.append(artifact_from_path(source_budget_path, role="source_budget_plan"))
    if training_data_gate_report is not None:
        artifacts.append(
            artifact_from_path(
                reports_dir / "training_data_gate_report.json", role="training_data_gate_report"
            )
        )
        artifacts.append(artifact_from_path(gated_train_path, role="promoted_training_jsonl"))
        artifacts.append(artifact_from_path(gated_holdout_path, role="eval_holdout_jsonl"))
        artifacts.append(artifact_from_path(gated_review_path, role="human_review_queue_jsonl"))
    if source_balance_report is not None:
        artifacts.append(
            artifact_from_path(
                reports_dir / "source_balance_report.json", role="source_balance_report"
            )
        )
        artifacts.append(artifact_from_path(balanced_clean_path, role="balanced_clean_jsonl"))
    if instruction_mix_report is not None:
        artifacts.append(
            artifact_from_path(instruction_mix_report_path, role="instruction_mix_report")
        )
        artifacts.append(
            artifact_from_path(instruction_mix_path, role="scratch_instruction_mix_jsonl")
        )
    checkpoint_eval_path = reports_dir / "checkpoint_eval" / "checkpoint_eval_report.json"
    if checkpoint_eval_report is not None:
        artifacts.append(artifact_from_path(checkpoint_eval_path, role="checkpoint_eval_report"))
    checkpoint_comparison_path = (
        reports_dir / "checkpoint_compare" / "checkpoint_comparison_report.json"
    )
    if checkpoint_comparison_report is not None:
        artifacts.append(
            artifact_from_path(checkpoint_comparison_path, role="checkpoint_comparison_report")
        )
    if task_report is not None:
        artifacts.append(artifact_from_path(tasks_path, role="extracted_tasks"))
        artifacts.append(artifact_from_path(task_report_path, role="task_extraction_report"))
    if review_report is not None:
        artifacts.append(
            artifact_from_path(reports_dir / "task_review_report.json", role="task_review_report")
        )
        artifacts.append(
            artifact_from_path(automated_pass_tasks_path, role="automated_task_candidates")
        )
    for shard_path in manifest.train_shards + manifest.val_shards:
        artifacts.append(artifact_from_path(shard_path, role="token_shard"))
    version_id = build_version_id(config.dataset_id, [artifact.sha256 for artifact in artifacts])
    uploaded_objects = []
    store = None
    if config.upload_artifacts:
        progress.emit(
            "artifact_upload",
            "started",
            object_store_uri=config.object_store_uri,
            artifact_count=len(artifacts),
        )
        store = create_object_store(
            ObjectStoreConfig(
                uri=config.object_store_uri, endpoint_url=config.object_store_endpoint_url
            )
        )
        upload_prefix = f"{config.dataset_id}/{version_id}"
        upload_candidates = clean_files + [
            str(trained_tokenizer),
            str(shards_dir / "manifest.json"),
            str(contamination_path),
            str(quality_report_path),
            str(training_quality_path),
            str(source_quality_path),
            str(feedback_path),
        ]
        if license_filter_report is not None:
            upload_candidates.append(str(reports_dir / "license_filter_report.json"))
        if benchmark_filter_report is not None:
            upload_candidates.append(
                str(reports_dir / "benchmark_contamination_filter_report.json")
            )
        if near_dedup_report is not None:
            upload_candidates.append(str(reports_dir / "near_dedup_report.json"))
        if source_reputation_report is not None:
            upload_candidates.append(str(source_reputation_path))
        if source_budget_plan is not None:
            upload_candidates.append(str(source_budget_path))
        if training_data_gate_report is not None:
            upload_candidates.extend(
                [
                    str(reports_dir / "training_data_gate_report.json"),
                    str(gated_train_path),
                    str(gated_holdout_path),
                    str(gated_review_path),
                    str(gated_decisions_path),
                ]
            )
        if task_report is not None:
            upload_candidates.extend([str(tasks_path), str(task_report_path)])
        if review_report is not None:
            upload_candidates.extend(
                [str(reports_dir / "task_review_report.json"), str(automated_pass_tasks_path)]
            )
        if checkpoint_eval_report is not None:
            upload_candidates.append(str(checkpoint_eval_path))
        if source_balance_report is not None:
            upload_candidates.extend(
                [str(reports_dir / "source_balance_report.json"), str(balanced_clean_path)]
            )
        if instruction_mix_report is not None:
            upload_candidates.extend([str(instruction_mix_report_path), str(instruction_mix_path)])
        if checkpoint_comparison_report is not None:
            upload_candidates.append(str(checkpoint_comparison_path))
        upload_candidates.append(str(reports_dir / "tokenizer_audit_report.json"))
        upload_candidates.extend(manifest.train_shards + manifest.val_shards)
        uploaded_objects = upload_paths(store, upload_candidates, prefix=upload_prefix)
        progress.emit("artifact_upload", "complete", uploaded_objects=len(uploaded_objects))
    version_manifest = DatasetVersionManifest(
        dataset_id=config.dataset_id,
        version_id=version_id,
        source_registry=registry_report.model_dump(),
        crawl_report=crawl_report.model_dump(),
        license_filter_report=license_filter_report.model_dump() if license_filter_report else None,
        benchmark_contamination_filter_report=benchmark_filter_report.model_dump()
        if benchmark_filter_report
        else None,
        near_dedup_report=near_dedup_report.model_dump() if near_dedup_report else None,
        contamination_report=contamination_report.model_dump(),
        quality_report=quality_report.model_dump(),
        training_quality_report=training_quality_report.model_dump(),
        source_quality_report=source_quality_report.model_dump(),
        task_report=task_report.model_dump() if task_report else None,
        review_report=review_report.model_dump() if review_report else None,
        feedback_report=feedback_report.model_dump(),
        source_reputation_report=source_reputation_report.model_dump()
        if source_reputation_report
        else None,
        source_budget_plan=source_budget_plan.model_dump() if source_budget_plan else None,
        training_data_gate_report=training_data_gate_report.model_dump()
        if training_data_gate_report
        else None,
        checkpoint_eval_report=checkpoint_eval_report.model_dump()
        if checkpoint_eval_report
        else None,
        source_balance_report=source_balance_report.model_dump() if source_balance_report else None,
        instruction_mix_report=instruction_mix_report.model_dump()
        if instruction_mix_report
        else None,
        checkpoint_comparison_report=checkpoint_comparison_report.model_dump()
        if checkpoint_comparison_report
        else None,
        tokenizer_path=str(trained_tokenizer),
        shard_manifest=manifest.model_dump(),
        artifacts=artifacts,
        uploaded_objects=uploaded_objects,
    )
    manifest_path = version_manifest.write(root / "versions" / f"{version_id}.json")
    DatasetLedger(root / "versions" / "ledger.jsonl").append(version_manifest)
    if store is not None:
        uploaded_objects.append(
            store.put_file(
                manifest_path, key=f"{config.dataset_id}/{version_id}/version_manifest.json"
            )
        )
        progress.emit(
            "version_manifest_upload", "complete", version_manifest_path=str(manifest_path)
        )

    report_payload = DataPipelineReport(
        status="complete",
        dataset_id=config.dataset_id,
        version_id=version_id,
        work_dir=str(root),
        source_registry=registry_report,
        crawl=crawl_report.model_dump(),
        license_filter_report=license_filter_report.model_dump() if license_filter_report else None,
        benchmark_contamination_filter_report=benchmark_filter_report.model_dump()
        if benchmark_filter_report
        else None,
        near_dedup_report=near_dedup_report.model_dump() if near_dedup_report else None,
        contamination_report=contamination_report.model_dump(),
        quality_report=quality_report.model_dump(),
        training_quality_report=training_quality_report.model_dump(),
        source_quality_report=source_quality_report.model_dump(),
        task_report=task_report.model_dump() if task_report else None,
        review_report=review_report.model_dump() if review_report else None,
        feedback_report=feedback_report.model_dump(),
        source_reputation_report=source_reputation_report.model_dump()
        if source_reputation_report
        else None,
        source_budget_plan=source_budget_plan.model_dump() if source_budget_plan else None,
        training_data_gate_report=training_data_gate_report.model_dump()
        if training_data_gate_report
        else None,
        source_balance_report=source_balance_report.model_dump() if source_balance_report else None,
        instruction_mix_report=instruction_mix_report.model_dump()
        if instruction_mix_report
        else None,
        clean_files=clean_files,
        training_files=training_files,
        tokenizer_path=str(trained_tokenizer),
        shard_manifest=manifest.model_dump(),
        version_manifest_path=str(manifest_path),
        dashboard_path=str(root / "dashboard.html"),
        uploaded_objects=[item.model_dump() for item in uploaded_objects],
        training=training_report,
        checkpoint_eval=checkpoint_eval_report.model_dump() if checkpoint_eval_report else None,
        checkpoint_comparison=checkpoint_comparison_report.model_dump()
        if checkpoint_comparison_report
        else None,
        tokenizer_audit_report=tokenizer_audit_report.model_dump(),
        report_artifacts={
            "pipeline_report": str(reports_dir / "pipeline_report.json"),
            "dashboard": str(root / "dashboard.html"),
            "quality_report": str(reports_dir / "quality_report.json"),
            "training_quality_report": str(reports_dir / "training_quality_report.json"),
            "source_reputation_report": str(reports_dir / "source_reputation_report.json"),
            "source_budget_plan": str(reports_dir / "source_budget_plan.json"),
            "training_data_gate_report": str(reports_dir / "training_data_gate_report.json"),
            "instruction_mix_report": str(reports_dir / "instruction_mix_report.json"),
            "tokenizer_audit_report": str(reports_dir / "tokenizer_audit_report.json"),
            "tokenizer_audit_markdown": str(reports_dir / "tokenizer_audit_report.md"),
            "checkpoint_eval_report": str(
                reports_dir / "checkpoint_eval" / "checkpoint_eval_report.json"
            ),
            "checkpoint_comparison_report": str(
                reports_dir / "checkpoint_compare" / "checkpoint_comparison_report.json"
            ),
            "progress_log": str(progress.path) if progress.path else "",
        },
    )
    report_path = reports_dir / "pipeline_report.json"
    report_path.write_text(
        json.dumps(report_payload.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_dashboard(report_payload.model_dump(), root / "dashboard.html")
    progress.emit(
        "pipeline",
        "complete",
        dataset_id=config.dataset_id,
        version_id=version_id,
        pipeline_status=report_payload.status,
        clean_files=len(clean_files),
        training_files=len(training_files),
        dashboard_path=str(root / "dashboard.html"),
    )
    progress.close()
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Craftly crawl -> clean -> shard -> train pipeline."
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument("--dataset-id", default="craftly-defensive-coding-corpus")
    parser.add_argument("--work-dir", default="artifacts/craftly/data-pipeline")
    parser.add_argument("--frontier-backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--postgres-dsn")
    parser.add_argument("--max-docs", type=int, default=10_000)
    parser.add_argument("--max-bytes-per-doc", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--shard-rows", type=int, default=10_000)
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128_000)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--model-profile",
        default="tiny",
        choices=["tiny", "t4_validation", "1b", "7b", "32b", "62b"],
    )
    parser.add_argument("--attention-impl", default="auto", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--no-checkpoint-eval", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-license-filter", action="store_true")
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--no-benchmark-contamination-filter", action="store_true")
    parser.add_argument("--no-near-dedup", action="store_true")
    parser.add_argument("--near-dedup-hamming-threshold", type=int, default=3)
    parser.add_argument("--no-source-reputation", action="store_true")
    parser.add_argument("--no-source-budget", action="store_true")
    parser.add_argument("--source-budget-target-docs", type=int)
    parser.add_argument("--no-training-data-gate", action="store_true")
    parser.add_argument("--min-training-quality-score", type=float, default=0.58)
    parser.add_argument("--min-training-average-quality-score", type=float, default=0.0)
    parser.add_argument("--min-training-rows", type=int, default=1)
    parser.add_argument("--min-train-tokens", type=int, default=128)
    parser.add_argument("--min-source-reputation-score", type=float, default=0.45)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--no-source-balancing", action="store_true")
    parser.add_argument("--max-source-fraction", type=float, default=0.35)
    parser.add_argument("--min-source-rows", type=int, default=25)
    parser.add_argument("--no-instruction-mix", action="store_true")
    parser.add_argument("--instruction-mix-max-rows", type=int)
    parser.add_argument(
        "--curriculum-mode",
        default="balanced",
        choices=[
            "balanced",
            "fundamentals_only",
            "defensive_security_only",
            "debug_patch_only",
            "agentic_coding_only",
        ],
    )
    parser.add_argument("--allow-offensive-misuse-rows", action="store_true")
    parser.add_argument("--contamination-patterns")
    parser.add_argument("--allow-contamination-hits", action="store_true")
    parser.add_argument("--no-task-extraction", action="store_true")
    parser.add_argument("--no-task-review", action="store_true")
    parser.add_argument("--max-extracted-tasks", type=int, default=50_000)
    parser.add_argument("--max-tasks-per-source-row", type=int, default=3)
    parser.add_argument("--object-store-uri", default="local://artifacts/craftly/object-store")
    parser.add_argument("--object-store-endpoint-url")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--checkpoint-compare-prompt-suite")
    parser.add_argument("--checkpoint-compare-min-score", type=float, default=0.0)
    parser.add_argument("--checkpoint-compare-max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-compare-repetition-penalty", type=float, default=1.12)
    parser.add_argument("--checkpoint-compare-no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--checkpoint-compare-max-repetition-ratio", type=float, default=0.72)
    parser.add_argument("--progress-path")
    parser.add_argument("--progress-to-stdout", action="store_true")
    parser.add_argument("--progress-every-docs", type=int, default=25)
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--production-mode", action="store_true")
    parser.add_argument("--dev-smoke", action="store_true")
    parser.add_argument("--promoted-dataset-manifest")
    parser.add_argument("--dataset-trust-policy", default="config/dataset_trust_policy.json")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DataPipelineConfig:
    return DataPipelineConfig(
        sources_path=args.sources,
        dataset_id=args.dataset_id,
        work_dir=args.work_dir,
        frontier_backend=args.frontier_backend,
        postgres_dsn=args.postgres_dsn,
        max_docs=args.max_docs,
        max_bytes_per_doc=args.max_bytes_per_doc,
        workers=args.workers,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        shard_rows=args.shard_rows,
        respect_robots=not args.ignore_robots,
        vocab_size=args.vocab_size,
        tokenizer_min_frequency=args.tokenizer_min_frequency,
        shard_token_count=args.shard_token_count,
        sequence_length=args.sequence_length,
        validation_fraction=args.validation_fraction,
        skip_train=args.skip_train,
        train_steps=args.train_steps,
        train_device=args.train_device,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dtype=args.dtype,
        model_profile_name=args.model_profile,
        attention_impl=args.attention_impl,
        gradient_checkpointing=args.gradient_checkpointing,
        validate_every=args.validate_every,
        validation_batches=args.validation_batches,
        run_checkpoint_eval=not args.no_checkpoint_eval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        filter_licenses=not args.no_license_filter,
        strict_unknown_licenses=not args.allow_unknown_license,
        filter_benchmark_contamination=not args.no_benchmark_contamination_filter,
        near_dedup=not args.no_near_dedup,
        near_dedup_hamming_threshold=args.near_dedup_hamming_threshold,
        build_source_reputation=not args.no_source_reputation,
        build_source_budget=not args.no_source_budget,
        source_budget_target_docs=args.source_budget_target_docs,
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
        contamination_patterns_path=args.contamination_patterns,
        block_contamination=not args.allow_contamination_hits,
        extract_tasks=not args.no_task_extraction,
        review_tasks=not args.no_task_review,
        max_extracted_tasks=args.max_extracted_tasks,
        max_tasks_per_source_row=args.max_tasks_per_source_row,
        object_store_uri=args.object_store_uri,
        object_store_endpoint_url=args.object_store_endpoint_url,
        upload_artifacts=not args.no_upload,
        checkpoint_compare_prompt_suite=args.checkpoint_compare_prompt_suite,
        checkpoint_compare_min_score=args.checkpoint_compare_min_score,
        checkpoint_compare_max_new_tokens=args.checkpoint_compare_max_new_tokens,
        checkpoint_compare_repetition_penalty=args.checkpoint_compare_repetition_penalty,
        checkpoint_compare_no_repeat_ngram_size=args.checkpoint_compare_no_repeat_ngram_size,
        checkpoint_compare_max_repetition_ratio=args.checkpoint_compare_max_repetition_ratio,
        progress_path=args.progress_path,
        progress_to_stdout=args.progress_to_stdout,
        progress_every_docs=args.progress_every_docs,
        progress_every_steps=args.progress_every_steps,
        production_mode=args.production_mode,
        dev_smoke=args.dev_smoke,
        promoted_dataset_manifest_path=args.promoted_dataset_manifest,
        dataset_trust_policy_path=args.dataset_trust_policy,
    )


def main() -> None:
    report = asyncio.run(run_data_pipeline(config_from_args(parse_args())))
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
