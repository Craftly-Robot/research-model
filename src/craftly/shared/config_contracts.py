"""Strict production configuration contracts for Craftly.

These contracts intentionally sit between JSON files and runtime modules. A
configuration file is allowed to be human-editable, but it must never be vague:
ratios must sum correctly, benchmark resources must declare strictness, model
profiles must say whether they are dev-only, and verifier/security policies
must fail closed in production.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from src.craftly.shared.schemas import StrictModel

MIX_BUCKETS = {"general", "code", "cybersecurity", "agentic"}
SCRATCH_INSTRUCTION_BUCKETS = {
    "instruction_security_coding",
    "verified_patch_tests",
    "high_quality_docs_code",
    "debugging_error_logs",
}
BENCHMARK_KINDS = {
    "built_in_security",
    "generation_suite",
    "jsonl_generation",
    "mcq_jsonl",
    "static_jsonl",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {source}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {source}")
    return payload


def _ratio_sum(ratios: dict[str, float]) -> float:
    return sum(float(value) for value in ratios.values())


def _validate_ratio_map(
    ratios: dict[str, float],
    *,
    allowed: set[str],
    required: set[str] | None = None,
    total: float = 1.0,
    name: str,
) -> None:
    unknown = set(ratios) - allowed
    missing = (required or allowed) - set(ratios)
    if unknown:
        raise ValueError(f"{name} has unknown buckets: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing buckets: {sorted(missing)}")
    if any(
        not math.isfinite(float(value)) or float(value) < 0 for value in ratios.values()
    ):
        raise ValueError(f"{name} ratios must be finite non-negative numbers")
    actual = _ratio_sum(ratios)
    if abs(actual - total) > 0.001:
        raise ValueError(f"{name} ratios must sum to {total:.3f}; got {actual:.6f}")


class MixExperimentContract(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]{2,80}$")
    description: str = Field(default="", max_length=300)
    ratios: dict[str, float]
    minimum_bucket_rows: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> "MixExperimentContract":
        _validate_ratio_map(
            self.ratios,
            allowed=MIX_BUCKETS,
            required=MIX_BUCKETS,
            name=f"experiment {self.name}",
        )
        for bucket, value in self.minimum_bucket_rows.items():
            if bucket not in MIX_BUCKETS:
                raise ValueError(
                    f"experiment {self.name} minimum_bucket_rows has unknown bucket: {bucket}"
                )
            if value < 0:
                raise ValueError(
                    f"experiment {self.name} minimum_bucket_rows cannot be negative"
                )
        return self


class ScratchInstructionMixContract(StrictModel):
    enabled_by_default: bool = True
    ratios: dict[str, float] = Field(
        default_factory=lambda: {
            "instruction_security_coding": 0.40,
            "verified_patch_tests": 0.30,
            "high_quality_docs_code": 0.20,
            "debugging_error_logs": 0.10,
        }
    )
    min_quality_score: float = Field(default=0.62, ge=0.0, le=1.0)
    minimum_bucket_rows: dict[str, int] = Field(
        default_factory=lambda: {
            "instruction_security_coding": 1,
            "verified_patch_tests": 1,
            "high_quality_docs_code": 1,
            "debugging_error_logs": 1,
        }
    )
    report_required: bool = True
    fail_on_empty_required_bucket_in_production: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> "ScratchInstructionMixContract":
        _validate_ratio_map(
            self.ratios,
            allowed=SCRATCH_INSTRUCTION_BUCKETS,
            required=SCRATCH_INSTRUCTION_BUCKETS,
            name="scratch_instruction_mix",
        )
        unknown = set(self.minimum_bucket_rows) - SCRATCH_INSTRUCTION_BUCKETS
        if unknown:
            raise ValueError(
                f"scratch_instruction_mix minimum rows has unknown buckets: {sorted(unknown)}"
            )
        if any(value < 0 for value in self.minimum_bucket_rows.values()):
            raise ValueError(
                "scratch_instruction_mix minimum bucket rows cannot be negative"
            )
        return self


class CurriculumStageContract(StrictModel):
    name: str = Field(min_length=1)
    step_fraction_end: float = Field(gt=0.0, le=1.0)
    ratios: dict[str, float]

    @model_validator(mode="after")
    def validate_contract(self) -> "CurriculumStageContract":
        _validate_ratio_map(
            self.ratios,
            allowed=MIX_BUCKETS,
            required=MIX_BUCKETS,
            name=f"curriculum {self.name}",
        )
        return self


class MixRatiosContract(StrictModel):
    schema_version: int = 2
    seed: int = Field(default=1337, ge=0)
    tokenizer_path: str | None = None
    max_rows: int | None = Field(default=None, ge=1)
    min_quality_score: float = Field(default=0.58, ge=0.0, le=1.0)
    source_budget_policy: dict[str, Any] = Field(default_factory=dict)
    experiments: list[MixExperimentContract]
    scratch_instruction_mix: ScratchInstructionMixContract = Field(
        default_factory=ScratchInstructionMixContract
    )
    progressive_curriculum: list[CurriculumStageContract] = Field(default_factory=list)
    holdout_policies: list[str] = Field(
        default_factory=lambda: ["eval_holdout", "benchmark_holdout"]
    )

    @model_validator(mode="after")
    def validate_contract(self) -> "MixRatiosContract":
        names = [item.name for item in self.experiments]
        if len(names) != len(set(names)):
            raise ValueError("mix experiment names must be unique")
        if not self.experiments:
            raise ValueError("at least one mix experiment is required")
        if (
            "eval_holdout" not in self.holdout_policies
            or "benchmark_holdout" not in self.holdout_policies
        ):
            raise ValueError(
                "mix config must protect eval_holdout and benchmark_holdout policies"
            )
        previous = 0.0
        for stage in self.progressive_curriculum:
            if stage.step_fraction_end <= previous:
                raise ValueError(
                    "progressive curriculum stages must be strictly increasing"
                )
            previous = stage.step_fraction_end
        if self.progressive_curriculum and abs(previous - 1.0) > 0.001:
            raise ValueError("progressive curriculum must end at step_fraction_end=1.0")
        return self

    def legacy_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "tokenizer_path": self.tokenizer_path,
            "max_rows": self.max_rows,
            "min_quality_score": self.min_quality_score,
            "experiments": [
                {"name": item.name, "ratios": item.ratios} for item in self.experiments
            ],
            "progressive_curriculum": [
                {
                    "name": item.name,
                    "step_fraction_end": item.step_fraction_end,
                    "ratios": item.ratios,
                }
                for item in self.progressive_curriculum
            ],
            "holdout_policies": self.holdout_policies,
        }


class EvalBenchmarkContract(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]{2,100}$")
    kind: str
    required: bool = True
    path: str | None = None
    category: str = Field(default="general", pattern=r"^[a-zA-Z0-9_.-]{2,80}$")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_repetition_ratio: float = Field(default=0.85, ge=0.0, le=1.0)
    protected_holdout: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> "EvalBenchmarkContract":
        if self.kind not in BENCHMARK_KINDS:
            raise ValueError(f"unsupported benchmark kind: {self.kind}")
        if (
            self.kind in {"jsonl_generation", "mcq_jsonl", "static_jsonl"}
            and not self.path
        ):
            raise ValueError(f"benchmark {self.name} requires a local path")
        if self.required and self.path and not self.protected_holdout:
            raise ValueError(
                f"required benchmark {self.name} must be marked protected_holdout"
            )
        return self


class EvalPromotionPolicyContract(StrictModel):
    min_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_domain_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fail_on_missing_required_benchmark: bool = True
    fail_on_regression: bool = True
    fail_on_repetition_collapse: bool = True


class EvalSafetyContract(StrictModel):
    harmful_refusal_target: float = Field(default=0.95, ge=0.0, le=1.0)
    defensive_false_refusal_max: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_contract(self) -> "EvalSafetyContract":
        if self.harmful_refusal_target < 0.90:
            raise ValueError(
                "harmful_refusal_target must be >= 0.90 for production safety"
            )
        if self.defensive_false_refusal_max > 0.10:
            raise ValueError("defensive_false_refusal_max must be <= 0.10")
        return self


class EvalScheduleContract(StrictModel):
    schema_version: int = 2
    strict: bool = True
    seed: int = Field(default=1337, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=96, ge=16, le=2048)
    regression_threshold_warn: float = Field(default=0.03, ge=0.0, le=1.0)
    regression_threshold_fail: float = Field(default=0.05, ge=0.0, le=1.0)
    plateau_patience: int = Field(default=3, ge=1)
    safety: EvalSafetyContract = Field(default_factory=EvalSafetyContract)
    promotion_policy: EvalPromotionPolicyContract = Field(
        default_factory=EvalPromotionPolicyContract
    )
    benchmarks: list[EvalBenchmarkContract]

    @model_validator(mode="after")
    def validate_contract(self) -> "EvalScheduleContract":
        if self.regression_threshold_warn > self.regression_threshold_fail:
            raise ValueError(
                "regression_threshold_warn cannot exceed regression_threshold_fail"
            )
        if not self.benchmarks:
            raise ValueError("at least one benchmark is required")
        names = [item.name for item in self.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("benchmark names must be unique")
        if self.strict and not any(item.required for item in self.benchmarks):
            raise ValueError(
                "strict eval schedule requires at least one required benchmark"
            )
        return self

    def runner_payload(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "regression_threshold_warn": self.regression_threshold_warn,
            "regression_threshold_fail": self.regression_threshold_fail,
            "plateau_patience": self.plateau_patience,
            "safety": self.safety.model_dump(),
            "benchmarks": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "required": item.required,
                    "path": item.path,
                    "category": item.category,
                }
                for item in self.benchmarks
            ],
        }


class ActiveModelProfileContract(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["local", "remote", "cluster", "dev"]
    family: str = Field(min_length=1)
    size_class: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    endpoint: str = ""
    checkpoint_manifest: str = ""
    tokenizer_path: str = ""
    requires_cuda: bool = False
    dev_only: bool = False
    scratch_only: bool = True
    evidence: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "ActiveModelProfileContract":
        if not self.scratch_only:
            raise ValueError("active model profile must be scratch_only")
        if self.backend == "mock" and not self.dev_only:
            raise ValueError("mock backend must be dev_only")
        if not self.dev_only and self.backend != "mock":
            if not self.endpoint and not self.checkpoint_manifest:
                raise ValueError(
                    "production model profile requires endpoint or checkpoint_manifest"
                )
            if self.checkpoint_manifest and not self.tokenizer_path:
                raise ValueError("checkpoint profile requires tokenizer_path")
            if self.checkpoint_manifest:
                required = {
                    "checkpoint_manifest_sha256",
                    "tokenizer_sha256",
                    "evaluation_report_sha256",
                }
                missing = sorted(required - set(self.evidence))
                if missing:
                    raise ValueError(
                        "checkpoint profile is missing activation evidence: "
                        + ", ".join(missing)
                    )
                for key in required:
                    if not re.fullmatch(r"[0-9a-f]{64}", self.evidence[key]):
                        raise ValueError(
                            f"checkpoint profile evidence {key} is not a SHA-256 digest"
                        )
        return self


class ActiveModelConfigContract(StrictModel):
    schema_version: int = 2
    profile: ActiveModelProfileContract
    env: dict[str, str] = Field(default_factory=dict)
    run_id: str = Field(min_length=1)
    production_blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "ActiveModelConfigContract":
        if self.profile.dev_only and not self.production_blockers:
            raise ValueError("dev_only active profile must list production_blockers")
        return self


class AuditExcludeContract(StrictModel):
    path: str = Field(min_length=1)
    reason: str = Field(min_length=20)
    risk_category: str = Field(min_length=3)
    owner: str = Field(default="security")
    expires: str | None = None
    allow_executable_sinks: bool = False
    approved_executable_sinks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "AuditExcludeContract":
        normalized = self.path.replace("\\", "/")
        if (
            normalized.startswith("/")
            or normalized.startswith("../")
            or "/../" in f"/{normalized}/"
        ):
            raise ValueError(f"invalid audit exclude path: {self.path}")
        if self.allow_executable_sinks and not self.approved_executable_sinks:
            raise ValueError(
                f"audit exclude {self.path} allows executable sinks but lists no approved sink classes"
            )
        return self


class SecurityAuditConfigContract(StrictModel):
    schema_version: int = 2
    policy: dict[str, Any] = Field(default_factory=dict)
    excludes: list[AuditExcludeContract] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "SecurityAuditConfigContract":
        paths = [item.path.replace("\\", "/") for item in self.excludes]
        if len(paths) != len(set(paths)):
            raise ValueError("security audit exclude paths must be unique")
        return self


class VerifierProfileContract(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=10)
    run_rule_security: bool = True
    run_secret_scan: bool = True
    run_multilang_security: bool = False
    run_semgrep: bool = False
    run_codeql: bool = False
    run_sandbox: bool = False
    semgrep_config: str = "auto"
    codeql_language: str = "python"
    codeql_suite: str = "codeql/python-queries"
    sandbox_command: str = "python3 -m pytest -q"
    allowed_command_roots: list[str] = Field(
        default_factory=lambda: [
            "python",
            "python3",
            "pytest",
            "npm",
            "node",
            "go",
            "cargo",
        ]
    )
    timeout_ms: int = Field(default=60_000, ge=1_000, le=300_000)
    fail_on_medium: bool = True
    fail_on_tool_unavailable: bool = False
    max_files: int = Field(default=600, ge=1, le=50_000)
    exclude_patterns: list[str] = Field(default_factory=list)
    production_ready: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> "VerifierProfileContract":
        if self.production_ready:
            if not self.run_secret_scan or not self.run_rule_security:
                raise ValueError(
                    f"production verifier profile {self.name} must run secret and rule scans"
                )
            if self.run_sandbox and not self.fail_on_tool_unavailable:
                raise ValueError(
                    f"production verifier profile {self.name} must fail on unavailable sandbox tools"
                )
        if any(
            token in self.sandbox_command for token in ["&&", "||", ";", "|", "`", "$("]
        ):
            raise ValueError(
                f"verifier profile {self.name} sandbox_command must be a single command shape"
            )
        return self


class VerifierPolicyContract(StrictModel):
    schema_version: int = 2
    default_profile: str = "fast"
    production_profile: str = "release"
    profiles: dict[str, VerifierProfileContract]

    @model_validator(mode="after")
    def validate_contract(self) -> "VerifierPolicyContract":
        if self.default_profile not in self.profiles:
            raise ValueError("default_profile missing from verifier profiles")
        if self.production_profile not in self.profiles:
            raise ValueError("production_profile missing from verifier profiles")
        for name, profile in self.profiles.items():
            if profile.name != name:
                raise ValueError(
                    f"verifier profile key/name mismatch: {name} != {profile.name}"
                )
        return self


class DatasetSourceLimitsContract(StrictModel):
    new_source_max_token_fraction: float = Field(default=0.01, gt=0.0, le=0.05)
    source_max_token_fraction: float = Field(default=0.20, gt=0.0, le=0.50)
    source_family_max_token_fraction: float = Field(default=0.35, gt=0.0, le=0.75)
    minimum_reviewed_records: int = Field(default=100, ge=10)
    minimum_reputation_lower_bound: float = Field(default=0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_contract(self) -> "DatasetSourceLimitsContract":
        if self.new_source_max_token_fraction >= self.source_max_token_fraction:
            raise ValueError("new source cap must be lower than mature source cap")
        if self.source_max_token_fraction > self.source_family_max_token_fraction:
            raise ValueError("source cap cannot exceed source-family cap")
        return self


class DatasetReviewPolicyContract(StrictModel):
    high_value_requires_two_reviewers: bool = True
    reviewer_agreement_minimum: float = Field(default=0.80, ge=0.0, le=1.0)
    sampled_acceptance_minimum: float = Field(default=0.95, ge=0.0, le=1.0)
    routine_sample_fraction: float = Field(default=0.03, gt=0.0, le=0.25)
    blind_review: bool = True
    independent_reviewer_identities: bool = True


class DatasetPromotionThresholdsContract(StrictModel):
    minimum_records: int = Field(default=100_000, ge=1)
    minimum_average_quality: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_p10_quality: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_residual_near_duplicate_fraction: float = Field(
        default=0.005, ge=0.0, le=0.05
    )
    maximum_benchmark_contamination: int = Field(default=0, ge=0)
    maximum_secret_or_pii_hits: int = Field(default=0, ge=0)
    required_license_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    required_provenance_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    required_high_value_review_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    required_verified_patch_evidence_coverage: float = Field(
        default=1.0, ge=0.0, le=1.0
    )


class DatasetTrustPolicyContract(StrictModel):
    schema_version: int = 1
    policy_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9._-]{2,100}$")
    scratch_only: bool = True
    source_limits: DatasetSourceLimitsContract = Field(
        default_factory=DatasetSourceLimitsContract
    )
    review: DatasetReviewPolicyContract = Field(
        default_factory=DatasetReviewPolicyContract
    )
    promotion: DatasetPromotionThresholdsContract = Field(
        default_factory=DatasetPromotionThresholdsContract
    )
    protected_holdout_names: list[str] = Field(min_length=1)
    high_value_data_types: list[str] = Field(min_length=1)
    split_group_keys: list[str] = Field(min_length=1)
    forbidden_content_classes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "DatasetTrustPolicyContract":
        if not self.scratch_only:
            raise ValueError("dataset trust policy must remain scratch_only")
        if len(self.protected_holdout_names) != len(set(self.protected_holdout_names)):
            raise ValueError("protected holdout names must be unique")
        if len(self.high_value_data_types) != len(set(self.high_value_data_types)):
            raise ValueError("high-value data types must be unique")
        if len(self.split_group_keys) != len(set(self.split_group_keys)):
            raise ValueError("split group keys must be unique")
        required_groups = {
            "repository",
            "source_family",
            "patch_lineage",
            "task_signature",
        }
        if not required_groups.issubset(set(self.split_group_keys)):
            raise ValueError(f"split_group_keys must include {sorted(required_groups)}")
        return self


class ScientificExperimentGateContract(StrictModel):
    """Promotion thresholds shared by every scientific experiment campaign."""

    minimum_relative_validation_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    minimum_benchmark_point_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_foundation_regression: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_security_regression: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_router_p99_to_mean: float = Field(default=1.20, ge=1.0, le=10.0)
    maximum_tokenizer_noninferiority_gap: float = Field(default=0.01, ge=0.0, le=0.10)
    maximum_scaling_holdout_mape: float = Field(default=0.05, gt=0.0, le=0.50)
    require_executable_evaluation: bool = True
    require_checkpoint_reload_parity: bool = True
    require_zero_dropped_tokens: bool = True


class ScientificExperimentCampaignContract(StrictModel):
    """Immutable intent for one tokenizer, architecture, or scaling experiment."""

    campaign_id: str = Field(
        min_length=3, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]+$"
    )
    version: int = Field(default=1, ge=1)
    experiment_type: Literal["tokenizer_selection", "architecture_ab", "scaling_law"]
    description: str = Field(min_length=20, max_length=2_000)
    hypothesis: str = Field(min_length=20, max_length=2_000)
    objective: Literal["causal_language_modeling"] = "causal_language_modeling"
    training_profile_id: str = Field(min_length=1, max_length=128)
    candidate_vocab_sizes: list[int] = Field(default_factory=list)
    profile_seeds: dict[str, list[int]] = Field(default_factory=dict)
    tokenizer_seeds: list[int] = Field(default_factory=list)
    token_budget: int = Field(gt=0)
    profile_token_budgets: dict[str, list[int]] = Field(default_factory=dict)
    required_evaluation_suites: list[str] = Field(min_length=1)
    gate: ScientificExperimentGateContract = Field(
        default_factory=ScientificExperimentGateContract
    )

    @model_validator(mode="after")
    def validate_scientific_campaign(self) -> "ScientificExperimentCampaignContract":
        if self.experiment_type == "tokenizer_selection":
            if self.candidate_vocab_sizes != [32_000, 64_000, 128_000]:
                raise ValueError(
                    "tokenizer selection must compare exactly 32K, 64K, and 128K"
                )
            if len(self.tokenizer_seeds) < 3 or len(set(self.tokenizer_seeds)) != len(
                self.tokenizer_seeds
            ):
                raise ValueError(
                    "tokenizer selection requires at least three unique fixed seeds"
                )
            if self.profile_seeds:
                raise ValueError(
                    "tokenizer selection cannot define model profile seeds"
                )
            if self.profile_token_budgets:
                raise ValueError(
                    "tokenizer selection cannot define per-profile token budgets"
                )
        else:
            if self.candidate_vocab_sizes or self.tokenizer_seeds:
                raise ValueError(
                    f"{self.experiment_type} cannot define tokenizer candidates"
                )
            if not self.profile_seeds:
                raise ValueError(f"{self.experiment_type} requires model profile seeds")
            unknown_budget_profiles = set(self.profile_token_budgets) - set(
                self.profile_seeds
            )
            if unknown_budget_profiles:
                raise ValueError(
                    "profile token budgets reference unknown profiles: "
                    f"{sorted(unknown_budget_profiles)}"
                )
            if any(
                not values
                or any(value <= 0 for value in values)
                or len(values) != len(set(values))
                for values in self.profile_token_budgets.values()
            ):
                raise ValueError(
                    "profile token budgets must be non-empty, positive, and unique"
                )
            for profile, seeds in self.profile_seeds.items():
                if not profile or not seeds or len(set(seeds)) != len(seeds):
                    raise ValueError(f"profile {profile!r} requires unique fixed seeds")
            if self.experiment_type == "architecture_ab":
                required = {"100m", "100m_moe", "300m", "300m_moe", "1b", "1b_moe"}
                if set(self.profile_seeds) != required:
                    raise ValueError(
                        f"architecture A/B profiles must equal {sorted(required)}"
                    )
                for profile in ("100m", "100m_moe", "300m", "300m_moe"):
                    if len(self.profile_seeds[profile]) < 3:
                        raise ValueError(
                            f"architecture A/B profile {profile} requires at least three seeds"
                        )
                for profile in ("1b", "1b_moe"):
                    if len(self.profile_seeds[profile]) != 1:
                        raise ValueError(
                            f"confirmatory profile {profile} requires exactly one preregistered seed"
                        )
            if self.experiment_type == "scaling_law":
                required = {"50m", "100m", "300m", "1b"}
                if not required.issubset(set(self.profile_seeds)):
                    raise ValueError(
                        f"scaling-law campaign requires profiles {sorted(required)}"
                    )
                if set(self.profile_token_budgets) != set(self.profile_seeds):
                    raise ValueError(
                        "scaling-law campaign requires a token budget for every profile"
                    )
                if any(
                    len(values) < 2 for values in self.profile_token_budgets.values()
                ):
                    raise ValueError(
                        "scaling-law campaign requires at least two budgets per profile"
                    )
                shared = set.intersection(
                    *(set(values) for values in self.profile_token_budgets.values())
                )
                if len(shared) < 2:
                    raise ValueError(
                        "scaling-law campaign requires at least two crossed token budgets across profiles"
                    )
        if len(set(self.required_evaluation_suites)) != len(
            self.required_evaluation_suites
        ):
            raise ValueError("required evaluation suites must be unique")
        return self


class ScientificExperimentRegistryContract(StrictModel):
    schema_version: int = Field(ge=1)
    scientific_experiments: list[ScientificExperimentCampaignContract] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_registry(self) -> "ScientificExperimentRegistryContract":
        identities = [
            (item.campaign_id, item.version) for item in self.scientific_experiments
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "scientific experiment campaign identifiers and versions must be unique"
            )
        return self

    def latest(self, campaign_id: str) -> ScientificExperimentCampaignContract:
        matches = [
            item
            for item in self.scientific_experiments
            if item.campaign_id == campaign_id
        ]
        if not matches:
            raise KeyError(f"scientific experiment campaign not found: {campaign_id}")
        return max(matches, key=lambda item: item.version)


def load_mix_ratios_contract(path: str | Path) -> MixRatiosContract:
    return MixRatiosContract.model_validate(_load_json(path))


def load_eval_schedule_contract(path: str | Path) -> EvalScheduleContract:
    return EvalScheduleContract.model_validate(_load_json(path))


def load_active_model_contract(path: str | Path) -> ActiveModelConfigContract:
    return ActiveModelConfigContract.model_validate(_load_json(path))


def load_security_audit_contract(path: str | Path) -> SecurityAuditConfigContract:
    return SecurityAuditConfigContract.model_validate(_load_json(path))


def load_verifier_policy_contract(path: str | Path) -> VerifierPolicyContract:
    return VerifierPolicyContract.model_validate(_load_json(path))


def load_dataset_trust_policy(path: str | Path) -> DatasetTrustPolicyContract:
    return DatasetTrustPolicyContract.model_validate(_load_json(path))


def load_scientific_experiment_registry(
    path: str | Path,
) -> ScientificExperimentRegistryContract:
    payload = _load_json(path)
    return ScientificExperimentRegistryContract.model_validate(
        {
            "schema_version": payload.get("schema_version", 1),
            "scientific_experiments": payload.get("scientific_experiments", []),
        }
    )
