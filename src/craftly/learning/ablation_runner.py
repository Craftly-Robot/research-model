"""Authoritative scientific experiment and model-selection control plane.

This module owns experiment identity, immutable input binding, evidence
admission, statistical comparison, and scientific promotion. It deliberately
does not implement another tokenizer, trainer, scheduler, or evaluator. Those
artifacts are produced by their existing authorities and admitted here only
after integrity and semantic checks pass.

The legacy data-mix ablation entrypoint remains available for compatibility,
but its output is explicitly preparatory and cannot promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess  # nosec B404 - fixed git argv only
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field, field_validator, model_validator

from src.craftly.learning.mixer import MixManifest, build_mix, load_mix_config
from src.craftly.model_ops.foundation import ScratchDecoderConfig, model_profile
from src.craftly.model_ops.tokenizer_pipeline import (
    SPECIAL_TOKENS,
    TokenizerArtifactManifest,
)
from src.craftly.shared.config_contracts import (
    ScientificExperimentCampaignContract,
    load_scientific_experiment_registry,
)
from src.craftly.shared.integrity import canonical_json_bytes, sha256_file
from src.craftly.shared.schemas import StrictModel

ExperimentStatus = Literal["passed", "failed", "blocked", "not_run"]
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_CONTAINER_DIGEST = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
PROMOTION_EVALUATION_AUTHORITY = "executable_model"


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            content = canonical_json_bytes(payload) + b"\n"
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _exclusive_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _json_object(
    path: str | Path, *, maximum_bytes: int = 64 * 1024 * 1024
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"JSON evidence is not a regular file: {source}")
    if source.stat().st_size > maximum_bytes:
        raise ValueError(f"JSON evidence exceeds {maximum_bytes} bytes: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON evidence {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {source}")
    return payload


def _git_commit() -> str:
    result = subprocess.run(  # nosec B603
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        },
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
        raise RuntimeError(
            "a readable immutable Git commit is required for an experiment"
        )
    return commit


def _digest_model(model: StrictModel, *, omitted: set[str] | None = None) -> str:
    payload = model.model_dump(mode="json")
    for name in omitted or set():
        payload.pop(name, None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class BoundArtifact(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    path: str
    sha256: str
    size_bytes: int = Field(ge=1)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("artifact SHA-256 is invalid")
        return value

    @classmethod
    def bind(cls, name: str, path: str | Path) -> "BoundArtifact":
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file() or source.stat().st_size < 1:
            raise ValueError(f"bound artifact is missing or empty: {source}")
        return cls(
            name=name,
            path=str(source),
            sha256=sha256_file(source),
            size_bytes=source.stat().st_size,
        )

    def verify(self) -> None:
        source = Path(self.path).expanduser().resolve(strict=True)
        if not source.is_file() or source.stat().st_size != self.size_bytes:
            raise ValueError(f"bound artifact size changed: {self.name}")
        if not hmac.compare_digest(sha256_file(source), self.sha256):
            raise ValueError(f"bound artifact hash changed: {self.name}")


def _bind_evaluation_inputs(
    evaluation_manifest: str | Path,
    *,
    required_suites: Sequence[str],
) -> dict[str, BoundArtifact]:
    """Bind every file that can influence a scientific evaluation result."""

    payload = _json_object(evaluation_manifest)
    policy = payload.get("executable_evaluation")
    if not isinstance(policy, dict):
        raise ValueError("evaluation manifest requires an executable_evaluation object")
    bindings: dict[str, BoundArtifact] = {}
    for name in ("protected_config", "protected_manifest"):
        raw_path = policy.get(name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"evaluation manifest is missing {name}")
        artifact = BoundArtifact.bind(name, raw_path)
        declared = policy.get(f"{name}_sha256")
        if declared is not None and not hmac.compare_digest(
            str(declared), artifact.sha256
        ):
            raise ValueError(f"evaluation manifest {name} hash mismatch")
        bindings[name] = artifact

    raw_suites = policy.get("suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError("evaluation manifest requires non-empty executable suites")
    seen: set[str] = set()
    for row in raw_suites:
        if not isinstance(row, dict):
            raise ValueError("evaluation suite entries must be objects")
        name = str(row.get("name") or "").strip()
        path = row.get("path")
        if not name or name in seen:
            raise ValueError("evaluation suite names must be non-empty and unique")
        if row.get("required") is not True:
            raise ValueError(f"scientific evaluation suite must be required: {name}")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"evaluation suite path is missing: {name}")
        artifact = BoundArtifact.bind(name, path)
        declared = row.get("sha256")
        if declared is not None and not hmac.compare_digest(
            str(declared), artifact.sha256
        ):
            raise ValueError(f"evaluation suite hash mismatch: {name}")
        bindings[f"suite:{name}"] = artifact
        seen.add(name)
    missing = sorted(set(required_suites) - seen)
    if missing:
        raise ValueError(
            f"evaluation manifest is missing required campaign suites: {missing}"
        )
    return bindings


class ExperimentArmPlan(StrictModel):
    arm_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,160}$")
    seed: int = Field(ge=0)
    model_profile: str
    model_contract: dict[str, Any]
    model_contract_sha256: str
    total_parameters: int = Field(gt=0)
    active_parameters: int = Field(gt=0)
    canonical_training_flops: float = Field(gt=0.0)
    vocab_size: int = Field(gt=0)
    token_budget: int = Field(gt=0)
    training_profile_id: str
    required_evaluation_suites: list[str] = Field(min_length=1)

    @field_validator("model_contract_sha256")
    @classmethod
    def validate_model_digest(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("model contract SHA-256 is invalid")
        return value

    @model_validator(mode="after")
    def verify_model_contract(self) -> "ExperimentArmPlan":
        contract = ScratchDecoderConfig.model_validate(self.model_contract)
        if not hmac.compare_digest(
            contract.contract_sha256(), self.model_contract_sha256
        ):
            raise ValueError("experiment arm model contract hash mismatch")
        report = contract.parameter_report()
        if (
            int(report["total"]) != self.total_parameters
            or int(report["active"]) != self.active_parameters
        ):
            raise ValueError("experiment arm parameter accounting mismatch")
        if contract.vocab_size != self.vocab_size:
            raise ValueError("experiment arm vocabulary does not match model contract")
        expected_flops = float(6 * self.active_parameters * self.token_budget)
        if not math.isclose(
            self.canonical_training_flops, expected_flops, rel_tol=1e-12
        ):
            raise ValueError("experiment arm canonical training FLOPs mismatch")
        return self


class ScientificArmExecutionRequest(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    arm_id: str
    status: Literal["not_run"] = "not_run"
    scheduler: Literal["notebook", "kubernetes", "kubernetes_pytorch", "slurm"]
    distributed_strategy: str
    training_profile_id: str
    model_profile: str
    model_contract_sha256: str
    total_parameters: int = Field(gt=0)
    active_parameters: int = Field(gt=0)
    canonical_training_flops: float = Field(gt=0.0)
    model_seed: int = Field(ge=0, le=2**31 - 1)
    dataloader_seed: int = Field(ge=0, le=2**31 - 1)
    world_size: int = Field(ge=1)
    optimizer_steps: int = Field(ge=1)
    token_budget: int = Field(ge=1)
    tokens_per_optimizer_step: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    dtype: str
    tokenizer_manifest_path: str
    tokenizer_manifest_sha256: str
    shard_manifest_path: str
    shard_manifest_sha256: str
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    split_manifest_path: str
    split_manifest_sha256: str
    optimizer_policy_path: str
    optimizer_policy_sha256: str
    evaluation_manifest_path: str
    evaluation_manifest_sha256: str
    container_digest: str
    required_evaluation_suites: list[str]

    @model_validator(mode="after")
    def validate_execution_shape(self) -> "ScientificArmExecutionRequest":
        if self.optimizer_steps * self.tokens_per_optimizer_step != self.token_budget:
            raise ValueError("scientific arm token budget is not exactly executable")
        expected_flops = float(6 * self.active_parameters * self.token_budget)
        if not math.isclose(
            self.canonical_training_flops, expected_flops, rel_tol=1e-12
        ):
            raise ValueError(
                "scientific execution request FLOPs do not match its active compute"
            )
        for value in (
            self.tokenizer_manifest_sha256,
            self.shard_manifest_sha256,
            self.dataset_manifest_sha256,
            self.split_manifest_sha256,
            self.model_contract_sha256,
            self.optimizer_policy_sha256,
            self.evaluation_manifest_sha256,
        ):
            if HEX_SHA256.fullmatch(value) is None:
                raise ValueError(
                    "scientific execution request contains an invalid SHA-256"
                )
        if SAFE_CONTAINER_DIGEST.fullmatch(self.container_digest) is None:
            raise ValueError(
                "scientific execution request requires a pinned container digest"
            )
        return self


class ExperimentManifest(StrictModel):
    schema_version: Literal[2] = 2
    authority: Literal["craftly_scientific_experiment"] = (
        "craftly_scientific_experiment"
    )
    experiment_id: str
    campaign: ScientificExperimentCampaignContract
    campaign_sha256: str
    git_commit: str
    container_digest: str
    objective: Literal["causal_language_modeling"]
    bindings: dict[str, BoundArtifact]
    evaluation_inputs: dict[str, BoundArtifact]
    tokenizers: dict[str, BoundArtifact]
    arms: list[ExperimentArmPlan] = Field(min_length=1)
    environment: dict[str, str]
    created_at_unix: float = Field(default_factory=time.time)
    manifest_sha256: str = ""

    @field_validator("campaign_sha256", "manifest_sha256")
    @classmethod
    def validate_manifest_digest(cls, value: str) -> str:
        if value and HEX_SHA256.fullmatch(value) is None:
            raise ValueError("experiment manifest digest is invalid")
        return value

    @field_validator("container_digest")
    @classmethod
    def validate_container_digest(cls, value: str) -> str:
        if SAFE_CONTAINER_DIGEST.fullmatch(value) is None:
            raise ValueError("experiment requires an immutable sha256 container digest")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> "ExperimentManifest":
        identities = [arm.arm_id for arm in self.arms]
        if len(identities) != len(set(identities)):
            raise ValueError("experiment arm IDs must be unique")
        expected_campaign = hashlib.sha256(
            canonical_json_bytes(self.campaign.model_dump(mode="json"))
        ).hexdigest()
        if not hmac.compare_digest(expected_campaign, self.campaign_sha256):
            raise ValueError("scientific campaign hash mismatch")
        expected_tokenizer_keys = (
            {str(value) for value in self.campaign.candidate_vocab_sizes}
            if self.campaign.experiment_type == "tokenizer_selection"
            else {"selected"}
        )
        if set(self.tokenizers) != expected_tokenizer_keys:
            raise ValueError(
                "experiment tokenizer bindings do not match the campaign: "
                f"expected={sorted(expected_tokenizer_keys)} actual={sorted(self.tokenizers)}"
            )
        required_evaluation_keys = {
            f"suite:{name}" for name in self.campaign.required_evaluation_suites
        }
        if not {
            "protected_config",
            "protected_manifest",
            *required_evaluation_keys,
        }.issubset(self.evaluation_inputs):
            raise ValueError("experiment evaluation inputs are incomplete")
        return self

    def sealed(self) -> "ExperimentManifest":
        return self.model_copy(
            update={"manifest_sha256": _digest_model(self, omitted={"manifest_sha256"})}
        )

    def verify(self) -> None:
        expected = _digest_model(self, omitted={"manifest_sha256"})
        if not self.manifest_sha256 or not hmac.compare_digest(
            expected, self.manifest_sha256
        ):
            raise ValueError("experiment manifest has been modified")
        for artifact in self.bindings.values():
            artifact.verify()
        for artifact in self.evaluation_inputs.values():
            artifact.verify()
        contracts = {
            key: _verified_tokenizer_contract(
                artifact,
                dataset_manifest_sha256=self.bindings["dataset_manifest"].sha256,
            )
            for key, artifact in self.tokenizers.items()
        }
        for key, contract in contracts.items():
            if key != "selected" and contract.vocab_size != int(key):
                raise ValueError(
                    f"tokenizer candidate {key} contains vocabulary {contract.vocab_size}"
                )
        selected_vocab = contracts.get("selected")
        if selected_vocab is not None and any(
            arm.vocab_size != selected_vocab.vocab_size for arm in self.arms
        ):
            raise ValueError("model arms do not use the selected tokenizer vocabulary")


def _verified_tokenizer_contract(
    artifact: BoundArtifact,
    *,
    dataset_manifest_sha256: str,
) -> TokenizerArtifactManifest:
    artifact.verify()
    contract = TokenizerArtifactManifest.model_validate(_json_object(artifact.path))
    if contract.status != "passed":
        raise ValueError(f"tokenizer manifest did not pass: {artifact.name}")
    if (
        not contract.family_safe_split
        or contract.split_strategy != "pre_split_family_safe"
    ):
        raise ValueError(f"tokenizer corpus is not family-safe: {artifact.name}")
    missing_control_tokens = sorted(
        {"<unk>", *SPECIAL_TOKENS} - set(contract.special_tokens)
    )
    if missing_control_tokens:
        raise ValueError(
            f"tokenizer manifest is missing required control tokens: {missing_control_tokens}"
        )
    if not hmac.compare_digest(
        contract.dataset_manifest_sha256, dataset_manifest_sha256
    ):
        raise ValueError(f"tokenizer dataset binding mismatch: {artifact.name}")
    for label, path, expected in (
        ("tokenizer", contract.tokenizer_path, contract.tokenizer_sha256),
        ("token shards", contract.shard_manifest_path, contract.shard_manifest_sha256),
    ):
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file() or not hmac.compare_digest(
            sha256_file(source), expected
        ):
            raise ValueError(f"{label} integrity verification failed: {artifact.name}")
    return contract


def _manifest_tokenizer_contract(
    manifest: ExperimentManifest,
    arm: ExperimentArmPlan,
) -> TokenizerArtifactManifest:
    key = (
        str(arm.vocab_size)
        if manifest.campaign.experiment_type == "tokenizer_selection"
        else "selected"
    )
    return _verified_tokenizer_contract(
        manifest.tokenizers[key],
        dataset_manifest_sha256=manifest.bindings["dataset_manifest"].sha256,
    )


class ArmEvidence(StrictModel):
    schema_version: Literal[1] = 1
    arm_id: str
    status: ExperimentStatus
    seed: int = Field(ge=0)
    objective: Literal["causal_language_modeling"]
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    optimizer_policy_sha256: str
    evaluation_manifest_sha256: str
    model_contract_sha256: str
    tokenizer_sha256: str
    tokenizer_vocab_size: int = Field(gt=0)
    trained_tokens: int = Field(gt=0)
    training_flops: float = Field(gt=0.0)
    total_parameters: int = Field(gt=0)
    active_parameters: int = Field(gt=0)
    validation_loss: float = Field(gt=0.0)
    executable_benchmark_score: float = Field(ge=0.0, le=1.0)
    foundation_score: float = Field(ge=0.0, le=1.0)
    security_score: float = Field(ge=0.0, le=1.0)
    tokens_per_byte: float = Field(gt=0.0)
    checkpoint_reload_parity: bool
    generation_collapsed: bool
    dropped_tokens: int = Field(ge=0)
    router_p99_to_mean: float | None = Field(default=None, ge=1.0)
    evaluation_authority: str
    training_report: BoundArtifact
    evaluation_report: BoundArtifact
    generation_audit: BoundArtifact
    checkpoint_manifest: BoundArtifact
    tokenizer_audit: BoundArtifact
    blockers: list[str] = Field(default_factory=list)

    @field_validator(
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "optimizer_policy_sha256",
        "evaluation_manifest_sha256",
        "model_contract_sha256",
        "tokenizer_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("arm evidence contains an invalid SHA-256")
        return value

    @model_validator(mode="after")
    def enforce_real_evidence(self) -> "ArmEvidence":
        if self.status == "passed" and self.blockers:
            raise ValueError("passed arm evidence cannot contain blockers")
        if self.evaluation_authority != PROMOTION_EVALUATION_AUTHORITY:
            raise ValueError(
                "only executable_model evaluation is admissible for promotion"
            )
        if not math.isfinite(self.validation_loss) or not math.isfinite(
            self.training_flops
        ):
            raise ValueError("training evidence contains non-finite metrics")
        return self

    def verify_artifacts(
        self,
        *,
        arm: ExperimentArmPlan,
        evaluation_inputs: dict[str, BoundArtifact],
    ) -> None:
        self.training_report.verify()
        self.evaluation_report.verify()
        self.generation_audit.verify()
        self.checkpoint_manifest.verify()
        self.tokenizer_audit.verify()
        training = _json_object(self.training_report.path)
        evaluation = _json_object(self.evaluation_report.path)
        generation = _json_object(self.generation_audit.path)
        tokenizer = _json_object(self.tokenizer_audit.path)
        if training.get("status") not in {"passed", "early_stopped"}:
            raise ValueError("bound training report did not pass")
        if training.get("scratch_only") is not True:
            raise ValueError("bound training report is not scratch-origin")
        if training.get("objective") != self.objective:
            raise ValueError(
                "training report objective does not match the preregistered objective"
            )
        if not hmac.compare_digest(
            str(training.get("optimizer_policy_sha256") or ""),
            self.optimizer_policy_sha256,
        ):
            raise ValueError(
                "training report optimizer policy does not match the experiment"
            )
        training_args = training.get("training_args")
        if not isinstance(training_args, dict):
            raise ValueError("training report is missing immutable training arguments")
        if training_args.get("step_semantics") != "optimizer_update_v2":
            raise ValueError("training report does not use optimizer-step semantics")
        if int(training_args.get("model_seed", -1)) != arm.seed:
            raise ValueError(
                "training report model seed does not match the preregistered arm seed"
            )
        expected_runtime_seed = (
            arm.seed + int(training_args.get("distributed_rank", 0)) * 1_000_003
        )
        if int(training_args.get("runtime_seed", -1)) != expected_runtime_seed:
            raise ValueError(
                "training report runtime seed is not derived from the preregistered arm seed"
            )
        if int(training_args.get("target_tokens", -1)) != self.trained_tokens:
            raise ValueError(
                "training arguments do not bind the preregistered token budget"
            )
        if int(training.get("trained_tokens", -1)) != self.trained_tokens:
            raise ValueError(
                "trained token count is not supported by the training report"
            )
        config = ScratchDecoderConfig.model_validate(training.get("model_config"))
        if config.contract_sha256() != self.model_contract_sha256:
            raise ValueError("training report model contract does not match the arm")
        if (
            training.get("checkpoint_reload_verified")
            is not self.checkpoint_reload_parity
        ):
            raise ValueError(
                "checkpoint reload evidence does not match the arm summary"
            )
        if (
            training.get("checkpoint_reload_logit_parity")
            is not self.checkpoint_reload_parity
        ):
            raise ValueError("checkpoint logit parity does not match the arm summary")
        reported_validation = float(training.get("best_validation_loss", math.inf))
        if not math.isclose(
            reported_validation, self.validation_loss, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError("validation loss is not supported by the training report")
        expected_flops = float(6 * self.active_parameters * self.trained_tokens)
        if not math.isclose(self.training_flops, expected_flops, rel_tol=1e-9):
            raise ValueError(
                "training FLOPs must use canonical 6*N_active*tokens accounting"
            )
        checkpoint_path = str(
            training.get("best_checkpoint_manifest")
            or training.get("checkpoint_manifest")
            or ""
        )
        checkpoint_sha = str(
            training.get("best_checkpoint_manifest_sha256")
            or training.get("checkpoint_manifest_sha256")
            or ""
        )
        if Path(checkpoint_path).expanduser().resolve(strict=True) != Path(
            self.checkpoint_manifest.path
        ):
            raise ValueError(
                "bound checkpoint differs from the selected training checkpoint"
            )
        if checkpoint_sha != self.checkpoint_manifest.sha256:
            raise ValueError("selected training checkpoint hash mismatch")
        router = (
            training.get("router_metrics")
            if isinstance(training.get("router_metrics"), dict)
            else {}
        )
        if int(router.get("dropped_assignments", 0)) != self.dropped_tokens:
            raise ValueError("router token-drop summary differs from training evidence")
        reported_router = router.get("maximum_p99_to_mean_load")
        if self.router_p99_to_mean is not None and not math.isclose(
            float(reported_router), self.router_p99_to_mean, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError("router load summary differs from training evidence")
        if (
            evaluation.get("status") != "passed"
            or evaluation.get("evaluation_mode") != "executable_model"
        ):
            raise ValueError("bound evaluation is not a passed executable-model report")
        if not hmac.compare_digest(
            str(evaluation.get("evaluation_manifest_sha256") or ""),
            self.evaluation_manifest_sha256,
        ):
            raise ValueError(
                "executable evaluation is not bound to the experiment evaluation manifest"
            )
        if not hmac.compare_digest(
            str(evaluation.get("checkpoint_manifest_sha256") or ""),
            self.checkpoint_manifest.sha256,
        ):
            raise ValueError(
                "executable evaluation checkpoint hash differs from the selected checkpoint"
            )
        if not hmac.compare_digest(
            str(evaluation.get("tokenizer_sha256") or ""),
            self.tokenizer_sha256,
        ):
            raise ValueError(
                "executable evaluation tokenizer hash differs from the selected tokenizer"
            )
        reported_suite_hashes = evaluation.get("suite_artifact_sha256")
        if not isinstance(reported_suite_hashes, dict):
            raise ValueError("executable evaluation is missing suite artifact hashes")
        expected_suite_hashes = {
            key.removeprefix("suite:"): artifact.sha256
            for key, artifact in evaluation_inputs.items()
            if key.startswith("suite:")
        }
        for name, expected_hash in expected_suite_hashes.items():
            if not hmac.compare_digest(
                str(reported_suite_hashes.get(name) or ""), expected_hash
            ):
                raise ValueError(f"executable evaluation suite hash mismatch: {name}")
        if not math.isclose(
            float(evaluation.get("aggregate_score", math.nan)),
            self.executable_benchmark_score,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "benchmark score differs from executable evaluation evidence"
            )
        suites = (
            evaluation.get("suites")
            if isinstance(evaluation.get("suites"), list)
            else []
        )
        suite_names = {str(row.get("name")) for row in suites if isinstance(row, dict)}
        missing_suites = sorted(set(arm.required_evaluation_suites) - suite_names)
        if missing_suites:
            raise ValueError(
                f"executable evaluation is missing required suites: {missing_suites}"
            )
        foundation_kinds = {
            "human_eval_style",
            "mbpp_style",
            "swe_bench_style",
            "repoqa_style",
        }
        security_kinds = {"cyberseceval_style", "custom_security"}
        foundation_scores = [
            float(row["score"])
            for row in suites
            if isinstance(row, dict)
            and row.get("kind") in foundation_kinds
            and row.get("status") == "passed"
        ]
        security_scores = [
            float(row["score"])
            for row in suites
            if isinstance(row, dict)
            and row.get("kind") in security_kinds
            and row.get("status") == "passed"
        ]
        if not foundation_scores or not security_scores:
            raise ValueError(
                "executable evaluation lacks passed foundation or security suites"
            )
        if not math.isclose(
            statistics.fmean(foundation_scores),
            self.foundation_score,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("foundation score differs from executable suite evidence")
        if not math.isclose(
            statistics.fmean(security_scores),
            self.security_score,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("security score differs from executable suite evidence")
        if (
            generation.get("evaluation_authority") != "diagnostic_keyword"
            or generation.get("promotion_eligible") is not False
        ):
            raise ValueError(
                "generation audit must be explicitly diagnostic and non-promotable"
            )
        candidate = (
            generation.get("candidate")
            if isinstance(generation.get("candidate"), dict)
            else {}
        )
        collapsed = int(candidate.get("collapsed_count", -1)) > 0
        if collapsed is not self.generation_collapsed:
            raise ValueError(
                "generation-collapse summary differs from diagnostic evidence"
            )
        if float(candidate.get("hallucination_rate", 1.0)) > 0.0:
            raise ValueError("diagnostic generation audit contains hallucination flags")
        candidate_checkpoint = (
            Path(str(candidate.get("checkpoint_manifest") or ""))
            .expanduser()
            .resolve(strict=True)
        )
        if candidate_checkpoint != Path(self.checkpoint_manifest.path):
            raise ValueError(
                "generation audit checkpoint differs from the selected checkpoint"
            )
        if (
            tokenizer.get("status") != "passed"
            or tokenizer.get("audit_failures")
            or tokenizer.get("special_tokens_missing")
        ):
            raise ValueError("bound tokenizer audit did not pass")
        if int(tokenizer.get("vocab_size_actual", -1)) != self.tokenizer_vocab_size:
            raise ValueError("tokenizer vocabulary differs from tokenizer audit")
        if str(tokenizer.get("tokenizer_sha256") or "") != self.tokenizer_sha256:
            raise ValueError("tokenizer hash differs from tokenizer audit")
        token_statistics = tokenizer.get("token_statistics")
        if not isinstance(token_statistics, dict) or not math.isclose(
            float(token_statistics.get("tokens_per_byte", math.nan)),
            self.tokens_per_byte,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("token efficiency differs from tokenizer audit evidence")


class MetricSummary(StrictModel):
    count: int = Field(ge=1)
    mean: float
    standard_deviation: float = Field(ge=0.0)
    confidence_low: float
    confidence_high: float


class CandidateComparison(StrictModel):
    candidate: str
    validation_loss: MetricSummary
    executable_benchmark_score: MetricSummary
    foundation_score: MetricSummary
    security_score: MetricSummary
    tokens_per_byte: MetricSummary
    downstream_score_summary: MetricSummary
    downstream_score: float
    composite_score: float
    hard_gate_passed: bool
    blockers: list[str] = Field(default_factory=list)


class ArchitecturePairComparison(StrictModel):
    scale: str
    dense_profile: str
    moe_profile: str
    active_parameter_delta: float
    training_flop_delta: float
    relative_validation_improvement: float
    benchmark_point_improvement: float
    foundation_regression: float
    security_regression: float
    passed: bool
    blockers: list[str] = Field(default_factory=list)


class ScalingLawReport(StrictModel):
    status: ExperimentStatus
    parameter_exponent: float | None = None
    token_exponent: float | None = None
    irreducible_loss: float | None = None
    parameter_coefficient: float | None = None
    token_coefficient: float | None = None
    holdout_mape: float | None = None
    parameter_exponent_ci: tuple[float, float] | None = None
    token_exponent_ci: tuple[float, float] | None = None
    predictions: dict[str, dict[str, float]] = Field(default_factory=dict)
    compute_optimal: dict[str, dict[str, float]] = Field(default_factory=dict)
    distinct_shapes: int = Field(default=0, ge=0)
    recommended_next_experiment: str
    blockers: list[str] = Field(default_factory=list)


class StatisticalComparisonReport(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    experiment_type: Literal["tokenizer_selection", "architecture_ab", "scaling_law"]
    status: ExperimentStatus
    candidate_comparisons: list[CandidateComparison] = Field(default_factory=list)
    architecture_pairs: list[ArchitecturePairComparison] = Field(default_factory=list)
    scaling_law: ScalingLawReport | None = None
    selected_candidate: str | None = None
    blockers: list[str] = Field(default_factory=list)
    arm_reports_sha256: str = ""
    report_sha256: str = ""

    @field_validator("arm_reports_sha256")
    @classmethod
    def validate_arm_reports_digest(cls, value: str) -> str:
        if value and HEX_SHA256.fullmatch(value) is None:
            raise ValueError("arm report digest is invalid")
        return value

    def sealed(self) -> "StatisticalComparisonReport":
        return self.model_copy(
            update={"report_sha256": _digest_model(self, omitted={"report_sha256"})}
        )


class ExperimentDecision(StrictModel):
    schema_version: Literal[1] = 1
    authority: Literal["craftly_experiment_decision"] = "craftly_experiment_decision"
    experiment_id: str
    status: ExperimentStatus
    selected_candidate: str | None = None
    manifest_sha256: str
    comparison_sha256: str
    rationale: list[str] = Field(min_length=1)
    blockers: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)
    decision_sha256: str = ""

    def sealed(self) -> "ExperimentDecision":
        return self.model_copy(
            update={"decision_sha256": _digest_model(self, omitted={"decision_sha256"})}
        )


class PromotionDecision(StrictModel):
    schema_version: Literal[1] = 1
    authority: Literal["craftly_scientific_promotion"] = "craftly_scientific_promotion"
    experiment_id: str
    status: Literal["promoted", "blocked"]
    promoted_candidate: str | None = None
    experiment_decision_sha256: str
    production_qualification_required: Literal[True] = True
    blockers: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)
    promotion_sha256: str = ""

    def sealed(self) -> "PromotionDecision":
        return self.model_copy(
            update={
                "promotion_sha256": _digest_model(self, omitted={"promotion_sha256"})
            }
        )


class ModelProgressionDecision(StrictModel):
    """Composite scientific authorization for the next scratch-model scale.

    This is not a deployment or production promotion. It proves that the
    tokenizer, architecture, and scaling decisions all passed on one immutable
    evidence lineage. Production qualification remains the final authority.
    """

    schema_version: Literal[2] = 2
    authority: Literal["craftly_model_progression"] = "craftly_model_progression"
    status: Literal["authorized", "blocked"]
    target_scale: Literal["7b"] = "7b"
    selected_tokenizer_vocab_size: int | None = Field(default=None, gt=0)
    selected_architecture: str | None = None
    target_model_profile: str | None = None
    target_model_contract_sha256: str | None = None
    tokenizer_promotion_sha256: str
    architecture_promotion_sha256: str
    scaling_promotion_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    evaluation_manifest_sha256: str
    selected_tokenizer_manifest_sha256: str | None = None
    blockers: list[str] = Field(default_factory=list)
    production_qualification_required: Literal[True] = True
    created_at_unix: float = Field(default_factory=time.time)
    decision_sha256: str = ""

    @field_validator(
        "tokenizer_promotion_sha256",
        "architecture_promotion_sha256",
        "scaling_promotion_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "evaluation_manifest_sha256",
    )
    @classmethod
    def validate_progression_digest(cls, value: str) -> str:
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError("model progression contains an invalid SHA-256")
        return value

    @field_validator(
        "selected_tokenizer_manifest_sha256", "target_model_contract_sha256"
    )
    @classmethod
    def validate_optional_progression_digest(cls, value: str | None) -> str | None:
        if value is not None and HEX_SHA256.fullmatch(value) is None:
            raise ValueError("selected tokenizer manifest SHA-256 is invalid")
        return value

    @model_validator(mode="after")
    def validate_progression(self) -> "ModelProgressionDecision":
        if self.status == "authorized":
            if self.blockers:
                raise ValueError("authorized model progression cannot contain blockers")
            if (
                self.selected_tokenizer_vocab_size is None
                or not self.selected_architecture
            ):
                raise ValueError(
                    "authorized model progression requires tokenizer and architecture choices"
                )
            if self.selected_tokenizer_manifest_sha256 is None:
                raise ValueError(
                    "authorized model progression requires the selected tokenizer manifest hash"
                )
            if self.target_model_contract_sha256 is None:
                raise ValueError(
                    "authorized model progression requires the exact target model contract hash"
                )
            expected_profile = (
                "7b_moe" if self.selected_architecture == "1b_moe" else "7b"
            )
            if self.target_model_profile != expected_profile:
                raise ValueError(
                    "7B target profile does not match the selected 1B architecture"
                )
        return self

    def sealed(self) -> "ModelProgressionDecision":
        return self.model_copy(
            update={"decision_sha256": _digest_model(self, omitted={"decision_sha256"})}
        )


class AblationReport(StrictModel):
    """Legacy mix preparation report; never a model-promotion decision."""

    status: Literal["prepared_not_evaluated"] = "prepared_not_evaluated"
    mix_config: str
    input_paths: list[str]
    output_dir: str
    experiments: list[dict[str, object]]
    recommendation: str
    created_at_unix: float = Field(default_factory=time.time)


def _arm_from_profile(
    *,
    arm_id: str,
    profile_name: str,
    seed: int,
    token_budget: int,
    training_profile_id: str,
    required_suites: list[str],
    vocab_size: int | None = None,
) -> ExperimentArmPlan:
    contract = model_profile(profile_name)
    if vocab_size is not None:
        contract = contract.model_copy(update={"vocab_size": vocab_size})
    report = contract.parameter_report()
    return ExperimentArmPlan(
        arm_id=arm_id,
        seed=seed,
        model_profile=profile_name,
        model_contract=contract.model_dump(mode="json"),
        model_contract_sha256=contract.contract_sha256(),
        total_parameters=int(report["total"]),
        active_parameters=int(report["active"]),
        canonical_training_flops=float(6 * int(report["active"]) * token_budget),
        vocab_size=contract.vocab_size,
        token_budget=token_budget,
        training_profile_id=training_profile_id,
        required_evaluation_suites=required_suites,
    )


def _campaign_arms(
    campaign: ScientificExperimentCampaignContract,
    *,
    selected_vocab_size: int | None = None,
) -> list[ExperimentArmPlan]:
    arms: list[ExperimentArmPlan] = []
    if campaign.experiment_type == "tokenizer_selection":
        for vocab_size in campaign.candidate_vocab_sizes:
            for seed in campaign.tokenizer_seeds:
                arms.append(
                    _arm_from_profile(
                        arm_id=f"vocab-{vocab_size}-seed-{seed}",
                        profile_name="t4_validation",
                        seed=seed,
                        token_budget=campaign.token_budget,
                        training_profile_id=campaign.training_profile_id,
                        required_suites=campaign.required_evaluation_suites,
                        vocab_size=vocab_size,
                    )
                )
    else:
        if selected_vocab_size is None:
            raise ValueError(
                "architecture and scaling campaigns require an evidence-selected tokenizer"
            )
        for profile_name, seeds in sorted(campaign.profile_seeds.items()):
            budgets = campaign.profile_token_budgets.get(
                profile_name, [campaign.token_budget]
            )
            for budget in budgets:
                for seed in seeds:
                    arms.append(
                        _arm_from_profile(
                            arm_id=(
                                f"{profile_name.replace('_', '-')}-tokens-{budget}-seed-{seed}"
                                if len(budgets) > 1
                                else f"{profile_name.replace('_', '-')}-seed-{seed}"
                            ),
                            profile_name=profile_name,
                            seed=seed,
                            token_budget=budget,
                            training_profile_id=campaign.training_profile_id,
                            required_suites=campaign.required_evaluation_suites,
                            vocab_size=selected_vocab_size,
                        )
                    )
    return arms


def create_experiment_manifest(
    *,
    campaign: ScientificExperimentCampaignContract,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
    optimizer_policy: str | Path,
    evaluation_manifest: str | Path,
    tokenizer_manifests: dict[str, str | Path],
    container_digest: str,
) -> ExperimentManifest:
    bindings = {
        "dataset_manifest": BoundArtifact.bind("dataset_manifest", dataset_manifest),
        "split_manifest": BoundArtifact.bind("split_manifest", split_manifest),
        "optimizer_policy": BoundArtifact.bind("optimizer_policy", optimizer_policy),
        "evaluation_manifest": BoundArtifact.bind(
            "evaluation_manifest", evaluation_manifest
        ),
    }
    evaluation_inputs = _bind_evaluation_inputs(
        evaluation_manifest,
        required_suites=campaign.required_evaluation_suites,
    )
    campaign_sha256 = hashlib.sha256(
        canonical_json_bytes(campaign.model_dump(mode="json"))
    ).hexdigest()
    expected_tokenizer_keys = (
        {str(value) for value in campaign.candidate_vocab_sizes}
        if campaign.experiment_type == "tokenizer_selection"
        else {"selected"}
    )
    if set(tokenizer_manifests) != expected_tokenizer_keys:
        raise ValueError(
            "tokenizer manifest arguments do not match the campaign: "
            f"expected={sorted(expected_tokenizer_keys)} actual={sorted(tokenizer_manifests)}"
        )
    tokenizer_bindings = {
        key: BoundArtifact.bind(f"tokenizer_manifest_{key}", path)
        for key, path in sorted(tokenizer_manifests.items())
    }
    tokenizer_contracts = {
        key: _verified_tokenizer_contract(
            artifact,
            dataset_manifest_sha256=bindings["dataset_manifest"].sha256,
        )
        for key, artifact in tokenizer_bindings.items()
    }
    for key, contract in tokenizer_contracts.items():
        if key != "selected" and contract.vocab_size != int(key):
            raise ValueError(
                f"tokenizer candidate {key} contains vocabulary {contract.vocab_size}"
            )
    selected_vocab_size = (
        tokenizer_contracts["selected"].vocab_size
        if campaign.experiment_type != "tokenizer_selection"
        else None
    )
    identity_payload = {
        "campaign": campaign_sha256,
        "bindings": {name: item.sha256 for name, item in sorted(bindings.items())},
        "evaluation_inputs": {
            name: item.sha256 for name, item in sorted(evaluation_inputs.items())
        },
        "tokenizers": {
            name: item.sha256 for name, item in sorted(tokenizer_bindings.items())
        },
        "git_commit": _git_commit(),
        "container_digest": container_digest,
    }
    experiment_id = f"{campaign.campaign_id}-{hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()[:16]}"
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        campaign=campaign,
        campaign_sha256=campaign_sha256,
        git_commit=identity_payload["git_commit"],
        container_digest=container_digest,
        objective=campaign.objective,
        bindings=bindings,
        evaluation_inputs=evaluation_inputs,
        tokenizers=tokenizer_bindings,
        arms=_campaign_arms(campaign, selected_vocab_size=selected_vocab_size),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    ).sealed()
    manifest.verify()
    return manifest


def _load_manifest(experiment_dir: str | Path) -> ExperimentManifest:
    root = Path(experiment_dir).expanduser().resolve(strict=True)
    manifest = ExperimentManifest.model_validate(
        _json_object(root / "experiment_manifest.json")
    )
    manifest.verify()
    return manifest


def _load_arm_evidence(
    manifest: ExperimentManifest, evidence_dir: str | Path
) -> tuple[list[ArmEvidence], list[str]]:
    root = Path(evidence_dir).expanduser().resolve()
    planned = {arm.arm_id: arm for arm in manifest.arms}
    evidence: list[ArmEvidence] = []
    blockers: list[str] = []
    for arm_id, arm in planned.items():
        path = root / f"{arm_id}.json"
        if not path.exists():
            blockers.append(f"missing arm evidence: {arm_id}")
            continue
        try:
            item = ArmEvidence.model_validate(_json_object(path))
            if item.arm_id != arm.arm_id or item.seed != arm.seed:
                raise ValueError(
                    "arm identity or seed does not match the immutable plan"
                )
            if item.objective != manifest.objective:
                raise ValueError("training objective differs from the immutable plan")
            if item.model_contract_sha256 != arm.model_contract_sha256:
                raise ValueError("model contract differs from the immutable plan")
            if item.tokenizer_vocab_size != arm.vocab_size:
                raise ValueError("tokenizer vocabulary differs from the immutable plan")
            tokenizer_contract = _manifest_tokenizer_contract(manifest, arm)
            if not hmac.compare_digest(
                item.tokenizer_sha256, tokenizer_contract.tokenizer_sha256
            ):
                raise ValueError(
                    "tokenizer hash differs from the immutable experiment binding"
                )
            if item.trained_tokens != arm.token_budget:
                raise ValueError(
                    "trained token count differs from the immutable token budget"
                )
            if (
                item.total_parameters != arm.total_parameters
                or item.active_parameters != arm.active_parameters
            ):
                raise ValueError(
                    "arm parameter evidence differs from canonical accounting"
                )
            item.verify_artifacts(arm=arm, evaluation_inputs=manifest.evaluation_inputs)
            expected_bindings = {
                "dataset_manifest_sha256": manifest.bindings["dataset_manifest"].sha256,
                "split_manifest_sha256": manifest.bindings["split_manifest"].sha256,
                "optimizer_policy_sha256": manifest.bindings["optimizer_policy"].sha256,
                "evaluation_manifest_sha256": manifest.bindings[
                    "evaluation_manifest"
                ].sha256,
            }
            for field, expected in expected_bindings.items():
                if not hmac.compare_digest(str(getattr(item, field)), expected):
                    raise ValueError(f"arm evidence binding mismatch: {field}")
            if item.status != "passed":
                blockers.append(f"arm did not pass: {arm_id}")
            evidence.append(item)
        except Exception as exc:
            blockers.append(f"invalid arm evidence {arm_id}: {exc}")
    unexpected = sorted(
        path.stem for path in root.glob("*.json") if path.stem not in planned
    )
    if unexpected:
        blockers.append(f"unexpected arm evidence files: {unexpected}")
    return evidence, blockers


def _bootstrap_summary(values: Sequence[float], *, seed: int) -> MetricSummary:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric summary requires finite values")
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    if len(samples) == 1:
        low = high = mean
    else:
        rng = random.Random(seed)
        means = sorted(
            statistics.fmean(rng.choice(samples) for _ in samples) for _ in range(2_000)
        )
        low = means[int(0.025 * (len(means) - 1))]
        high = means[int(0.975 * (len(means) - 1))]
    return MetricSummary(
        count=len(samples),
        mean=mean,
        standard_deviation=deviation,
        confidence_low=low,
        confidence_high=high,
    )


def _minmax_quality(
    value: float, values: Sequence[float], *, lower_is_better: bool
) -> float:
    low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return 1.0
    normalized = (value - low) / (high - low)
    return 1.0 - normalized if lower_is_better else normalized


def _common_hard_gate_blockers(
    items: Sequence[ArmEvidence],
    campaign: ScientificExperimentCampaignContract,
) -> list[str]:
    blockers: list[str] = []
    if any(item.status != "passed" for item in items):
        blockers.append("one or more runs did not pass")
    if campaign.gate.require_checkpoint_reload_parity and any(
        not item.checkpoint_reload_parity for item in items
    ):
        blockers.append("checkpoint reload parity failed")
    if any(item.generation_collapsed for item in items):
        blockers.append("generation collapse detected")
    if campaign.gate.require_zero_dropped_tokens and any(
        item.dropped_tokens != 0 for item in items
    ):
        blockers.append("token drops detected")
    if campaign.gate.require_executable_evaluation and any(
        item.evaluation_authority != PROMOTION_EVALUATION_AUTHORITY for item in items
    ):
        blockers.append("non-executable evaluation evidence detected")
    return blockers


def compare_tokenizer_candidates(
    manifest: ExperimentManifest,
    evidence: Sequence[ArmEvidence],
) -> StatisticalComparisonReport:
    grouped: dict[int, list[ArmEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.tokenizer_vocab_size, []).append(item)
    expected = set(manifest.campaign.candidate_vocab_sizes)
    blockers: list[str] = []
    if set(grouped) != expected:
        blockers.append("tokenizer evidence does not cover every planned vocabulary")
    all_loss = [item.validation_loss for item in evidence]
    all_efficiency = [item.tokens_per_byte for item in evidence]
    comparisons: list[CandidateComparison] = []
    for vocab_size in sorted(grouped):
        items = grouped[vocab_size]
        hard = _common_hard_gate_blockers(items, manifest.campaign)
        expected_seed_count = len(manifest.campaign.tokenizer_seeds)
        if len(items) != expected_seed_count:
            hard.append(f"expected {expected_seed_count} seeds, found {len(items)}")
        validation = _bootstrap_summary(
            [item.validation_loss for item in items], seed=vocab_size + 1
        )
        executable = _bootstrap_summary(
            [item.executable_benchmark_score for item in items], seed=vocab_size + 2
        )
        foundation = _bootstrap_summary(
            [item.foundation_score for item in items], seed=vocab_size + 3
        )
        security = _bootstrap_summary(
            [item.security_score for item in items], seed=vocab_size + 4
        )
        efficiency = _bootstrap_summary(
            [item.tokens_per_byte for item in items], seed=vocab_size + 5
        )
        downstream_values = [
            0.50 * item.executable_benchmark_score
            + 0.30 * item.security_score
            + 0.20 * item.foundation_score
            for item in items
        ]
        downstream_summary = _bootstrap_summary(downstream_values, seed=vocab_size + 6)
        validation_quality = _minmax_quality(
            validation.mean, all_loss, lower_is_better=True
        )
        efficiency_quality = _minmax_quality(
            efficiency.mean, all_efficiency, lower_is_better=True
        )
        composite = (
            0.40 * executable.mean
            + 0.25 * security.mean
            + 0.15 * foundation.mean
            + 0.15 * validation_quality
            + 0.05 * efficiency_quality
        )
        downstream = (
            0.50 * executable.mean + 0.30 * security.mean + 0.20 * foundation.mean
        )
        comparisons.append(
            CandidateComparison(
                candidate=str(vocab_size),
                validation_loss=validation,
                executable_benchmark_score=executable,
                foundation_score=foundation,
                security_score=security,
                tokens_per_byte=efficiency,
                downstream_score_summary=downstream_summary,
                downstream_score=round(downstream, 8),
                composite_score=round(composite, 8),
                hard_gate_passed=not hard,
                blockers=hard,
            )
        )
    eligible = [item for item in comparisons if item.hard_gate_passed]
    selected: str | None = None
    if not blockers and eligible:
        best = max(item.downstream_score for item in eligible)
        gap = manifest.campaign.gate.maximum_tokenizer_noninferiority_gap
        noninferior = [item for item in eligible if best - item.downstream_score <= gap]
        candidate = min(noninferior, key=lambda item: int(item.candidate))
        smaller = [
            item for item in eligible if int(item.candidate) < int(candidate.candidate)
        ]
        if smaller and any(
            candidate.downstream_score_summary.confidence_low
            <= item.downstream_score_summary.confidence_high
            for item in smaller
        ):
            blockers.append(
                "larger tokenizer exceeds the one-percent boundary but is not statistically superior"
            )
        else:
            selected = candidate.candidate
    else:
        blockers.extend(
            "candidate hard gate failed: " + item.candidate
            for item in comparisons
            if not item.hard_gate_passed
        )
    return StatisticalComparisonReport(
        experiment_id=manifest.experiment_id,
        experiment_type="tokenizer_selection",
        status="passed" if selected else "blocked",
        candidate_comparisons=comparisons,
        selected_candidate=selected,
        blockers=sorted(set(blockers)),
    ).sealed()


def _profile_mean(items: Sequence[ArmEvidence], attribute: str) -> float:
    return statistics.fmean(float(getattr(item, attribute)) for item in items)


def compare_architecture_candidates(
    manifest: ExperimentManifest,
    evidence: Sequence[ArmEvidence],
) -> StatisticalComparisonReport:
    by_profile: dict[str, list[ArmEvidence]] = {}
    arm_profiles = {arm.arm_id: arm.model_profile for arm in manifest.arms}
    for item in evidence:
        by_profile.setdefault(arm_profiles[item.arm_id], []).append(item)
    pair_names = [
        ("100m", "100m", "100m_moe"),
        ("300m", "300m", "300m_moe"),
        ("1b", "1b", "1b_moe"),
    ]
    pairs: list[ArchitecturePairComparison] = []
    authority_blockers: list[str] = []
    for scale, dense_name, moe_name in pair_names:
        dense = by_profile.get(dense_name, [])
        moe = by_profile.get(moe_name, [])
        pair_blockers: list[str] = []
        if not dense or not moe:
            pair_blockers.append("paired dense/MoE evidence is incomplete")
            pairs.append(
                ArchitecturePairComparison(
                    scale=scale,
                    dense_profile=dense_name,
                    moe_profile=moe_name,
                    active_parameter_delta=1.0,
                    training_flop_delta=1.0,
                    relative_validation_improvement=-1.0,
                    benchmark_point_improvement=-1.0,
                    foundation_regression=1.0,
                    security_regression=1.0,
                    passed=False,
                    blockers=pair_blockers,
                )
            )
            authority_blockers.append(
                f"{scale}: paired dense/MoE evidence is incomplete"
            )
            continue
        dense_blockers = _common_hard_gate_blockers(dense, manifest.campaign)
        authority_blockers.extend(
            f"{scale} dense baseline: {item}" for item in dense_blockers
        )
        pair_blockers.extend(
            _common_hard_gate_blockers([*dense, *moe], manifest.campaign)
        )
        dense_active = _profile_mean(dense, "active_parameters")
        moe_active = _profile_mean(moe, "active_parameters")
        active_delta = abs(moe_active - dense_active) / dense_active
        dense_flops = _profile_mean(dense, "training_flops")
        moe_flops = _profile_mean(moe, "training_flops")
        flop_delta = abs(moe_flops - dense_flops) / dense_flops
        if active_delta > 0.01:
            pair_blockers.append("active parameter difference exceeds one percent")
        if flop_delta > 0.01:
            pair_blockers.append("training FLOP difference exceeds one percent")
        dense_loss = _profile_mean(dense, "validation_loss")
        moe_loss = _profile_mean(moe, "validation_loss")
        relative_improvement = (dense_loss - moe_loss) / dense_loss
        benchmark_delta = _profile_mean(
            moe, "executable_benchmark_score"
        ) - _profile_mean(dense, "executable_benchmark_score")
        foundation_regression = _profile_mean(
            dense, "foundation_score"
        ) - _profile_mean(moe, "foundation_score")
        security_regression = _profile_mean(dense, "security_score") - _profile_mean(
            moe, "security_score"
        )
        if foundation_regression > manifest.campaign.gate.maximum_foundation_regression:
            pair_blockers.append("foundation regression exceeds policy")
        if security_regression > manifest.campaign.gate.maximum_security_regression:
            pair_blockers.append("security regression exceeds policy")
        if any(
            item.router_p99_to_mean is None
            or item.router_p99_to_mean
            > manifest.campaign.gate.maximum_router_p99_to_mean
            for item in moe
        ):
            pair_blockers.append("MoE router p99 load exceeds policy or is missing")
        quality_gate = (
            relative_improvement
            >= manifest.campaign.gate.minimum_relative_validation_improvement
            or benchmark_delta
            >= manifest.campaign.gate.minimum_benchmark_point_improvement
        )
        if not quality_gate:
            pair_blockers.append(
                "MoE did not meet validation-loss or benchmark improvement gate"
            )
        pairs.append(
            ArchitecturePairComparison(
                scale=scale,
                dense_profile=dense_name,
                moe_profile=moe_name,
                active_parameter_delta=active_delta,
                training_flop_delta=flop_delta,
                relative_validation_improvement=relative_improvement,
                benchmark_point_improvement=benchmark_delta,
                foundation_regression=foundation_regression,
                security_regression=security_regression,
                passed=not pair_blockers,
                blockers=pair_blockers,
            )
        )
    selected = (
        None
        if authority_blockers
        else "1b_moe"
        if all(pair.passed for pair in pairs)
        else "1b"
    )
    return StatisticalComparisonReport(
        experiment_id=manifest.experiment_id,
        experiment_type="architecture_ab",
        status="passed" if selected else "failed",
        architecture_pairs=pairs,
        selected_candidate=selected,
        blockers=authority_blockers,
    ).sealed()


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("scaling-law design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


ScalingCoefficients = tuple[float, float, float, float, float]


def _aggregate_scaling_shapes(
    rows: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for parameters, tokens, loss in rows:
        grouped.setdefault((parameters, tokens), []).append(loss)
    return [
        (parameters, tokens, statistics.fmean(losses))
        for (parameters, tokens), losses in sorted(grouped.items())
    ]


def _linear_scaling_coefficients(
    rows: Sequence[tuple[float, float, float]],
    *,
    parameter_exponent: float,
    token_exponent: float,
) -> tuple[float, float, float, float]:
    # Weighted least squares minimizes relative prediction error, preventing the
    # highest-loss pilot from dominating the scientific decision.
    design = []
    target = []
    for parameters, tokens, loss in rows:
        if parameters <= 0 or tokens <= 0 or loss <= 0 or not math.isfinite(loss):
            raise ValueError("scaling measurements must be finite and positive")
        weight = 1.0 / loss
        design.append(
            [
                weight,
                (parameters**-parameter_exponent) * weight,
                (tokens**-token_exponent) * weight,
            ]
        )
        target.append(1.0)
    xtx = [[sum(row[i] * row[j] for row in design) for j in range(3)] for i in range(3)]
    xty = [sum(row[i] * value for row, value in zip(design, target)) for i in range(3)]
    irreducible, parameter_coefficient, token_coefficient = _solve_linear_system(
        xtx, xty
    )
    if irreducible < 0 or parameter_coefficient <= 0 or token_coefficient <= 0:
        raise ValueError("non-physical scaling coefficients")
    errors = []
    for parameters, tokens, loss in rows:
        predicted = (
            irreducible
            + parameter_coefficient * parameters**-parameter_exponent
            + token_coefficient * tokens**-token_exponent
        )
        if predicted <= 0 or not math.isfinite(predicted):
            raise ValueError("non-finite scaling prediction")
        errors.append(((predicted - loss) / loss) ** 2)
    return (
        irreducible,
        parameter_coefficient,
        token_coefficient,
        statistics.fmean(errors),
    )


def _fit_scaling_law(rows: Sequence[tuple[float, float, float]]) -> ScalingCoefficients:
    shapes = _aggregate_scaling_shapes(rows)
    if len(shapes) < 8:
        raise ValueError(
            "additive scaling-law fit requires at least eight distinct crossed shapes"
        )
    candidates: list[tuple[float, float, float, float, float, float]] = []
    for parameter_step in range(1, 16):
        parameter_exponent = parameter_step * 0.02
        for token_step in range(1, 16):
            token_exponent = token_step * 0.02
            try:
                irreducible, parameter_coefficient, token_coefficient, error = (
                    _linear_scaling_coefficients(
                        shapes,
                        parameter_exponent=parameter_exponent,
                        token_exponent=token_exponent,
                    )
                )
            except ValueError:
                continue
            candidates.append(
                (
                    error,
                    irreducible,
                    parameter_coefficient,
                    token_coefficient,
                    parameter_exponent,
                    token_exponent,
                )
            )
    if not candidates:
        raise ValueError("no physically valid additive scaling-law fit was found")
    coarse = min(candidates, key=lambda item: item[0])
    refined: list[tuple[float, float, float, float, float, float]] = [coarse]
    for parameter_offset in range(-10, 11):
        parameter_exponent = coarse[4] + parameter_offset * 0.002
        if parameter_exponent <= 0:
            continue
        for token_offset in range(-10, 11):
            token_exponent = coarse[5] + token_offset * 0.002
            if token_exponent <= 0:
                continue
            try:
                irreducible, parameter_coefficient, token_coefficient, error = (
                    _linear_scaling_coefficients(
                        shapes,
                        parameter_exponent=parameter_exponent,
                        token_exponent=token_exponent,
                    )
                )
            except ValueError:
                continue
            refined.append(
                (
                    error,
                    irreducible,
                    parameter_coefficient,
                    token_coefficient,
                    parameter_exponent,
                    token_exponent,
                )
            )
    (
        _,
        irreducible,
        parameter_coefficient,
        token_coefficient,
        parameter_exponent,
        token_exponent,
    ) = min(refined, key=lambda item: item[0])
    return (
        irreducible,
        parameter_coefficient,
        token_coefficient,
        parameter_exponent,
        token_exponent,
    )


def _scaling_prediction(
    coefficients: ScalingCoefficients, parameters: float, tokens: float
) -> float:
    (
        irreducible,
        parameter_coefficient,
        token_coefficient,
        parameter_exponent,
        token_exponent,
    ) = coefficients
    return (
        irreducible
        + parameter_coefficient * parameters**-parameter_exponent
        + token_coefficient * tokens**-token_exponent
    )


def _leave_one_shape_out_mape(rows: Sequence[tuple[float, float, float]]) -> float:
    shapes = _aggregate_scaling_shapes(rows)
    errors: list[float] = []
    for index, (_, _, actual) in enumerate(shapes):
        train = [row for position, row in enumerate(shapes) if position != index]
        try:
            coefficients = _fit_scaling_law(train)
        except ValueError:
            continue
        predicted = _scaling_prediction(
            coefficients, shapes[index][0], shapes[index][1]
        )
        errors.append(abs(predicted - actual) / actual)
    if len(errors) != len(shapes):
        raise ValueError(
            "scaling-law leave-one-shape-out validation was not identifiable"
        )
    return statistics.fmean(errors)


def _compute_optimal_shape(
    coefficients: ScalingCoefficients, training_flops: float
) -> dict[str, float]:
    _, parameter_coefficient, token_coefficient, parameter_exponent, token_exponent = (
        coefficients
    )
    compute_constant = training_flops / 6.0
    parameters = (
        (parameter_coefficient * parameter_exponent)
        / (token_coefficient * token_exponent)
        * compute_constant**token_exponent
    ) ** (1.0 / (parameter_exponent + token_exponent))
    tokens = compute_constant / parameters
    return {
        "training_flops": training_flops,
        "parameters": parameters,
        "tokens": tokens,
        "predicted_validation_loss": _scaling_prediction(
            coefficients, parameters, tokens
        ),
    }


def fit_scaling_law(
    manifest: ExperimentManifest,
    evidence: Sequence[ArmEvidence],
) -> ScalingLawReport:
    blockers = _common_hard_gate_blockers(evidence, manifest.campaign)
    rows = [
        (
            float(item.active_parameters),
            float(item.trained_tokens),
            item.validation_loss,
        )
        for item in evidence
    ]
    unique_shapes = {(parameters, tokens) for parameters, tokens, _ in rows}
    parameter_values = {parameters for parameters, _, _ in rows}
    token_values = {tokens for _, tokens, _ in rows}
    crossed_pairs = {
        tokens: {
            parameters
            for parameters, candidate_tokens, _ in rows
            if candidate_tokens == tokens
        }
        for tokens in token_values
    }
    if len(unique_shapes) < 8:
        blockers.append(
            "scaling-law fit requires at least eight distinct parameter/token shapes"
        )
    if (
        len(parameter_values) < 4
        or sum(len(values) >= 2 for values in crossed_pairs.values()) < 2
    ):
        blockers.append(
            "scaling-law design does not independently identify parameter and data effects"
        )
    if blockers:
        return ScalingLawReport(
            status="blocked",
            recommended_next_experiment="repair missing or invalid scaling evidence",
            blockers=blockers,
        )
    try:
        coefficients = _fit_scaling_law(rows)
        mape = _leave_one_shape_out_mape(rows)
        rng = random.Random(1337)
        shapes = _aggregate_scaling_shapes(rows)
        bootstraps: list[ScalingCoefficients] = []
        for _ in range(400):
            sampled = [rng.choice(shapes) for _ in shapes]
            if len({(item[0], item[1]) for item in sampled}) < 8:
                continue
            try:
                fitted = _fit_scaling_law(sampled)
            except ValueError:
                continue
            if all(math.isfinite(value) for value in fitted):
                bootstraps.append(fitted)
        if len(bootstraps) < 100:
            raise ValueError("insufficient stable bootstrap fits")
        parameter_samples = sorted(item[3] for item in bootstraps)
        token_samples = sorted(item[4] for item in bootstraps)
        low_index = int(0.025 * (len(bootstraps) - 1))
        high_index = int(0.975 * (len(bootstraps) - 1))
        predictions: dict[str, dict[str, float]] = {}
        targets = {"7b": (7e9, 175e9), "32b": (32e9, 700e9)}
        for name, (parameters, tokens) in targets.items():
            bootstrap_predictions = sorted(
                _scaling_prediction(item, parameters, tokens) for item in bootstraps
            )
            predictions[name] = {
                "parameters": parameters,
                "tokens": tokens,
                "training_flops": 6.0 * parameters * tokens,
                "predicted_validation_loss": _scaling_prediction(
                    coefficients, parameters, tokens
                ),
                "prediction_ci_low": bootstrap_predictions[low_index],
                "prediction_ci_high": bootstrap_predictions[high_index],
            }
        compute_optimal = {
            name: _compute_optimal_shape(coefficients, values["training_flops"])
            for name, values in predictions.items()
        }
        passed = mape <= manifest.campaign.gate.maximum_scaling_holdout_mape
        return ScalingLawReport(
            status="passed" if passed else "failed",
            irreducible_loss=coefficients[0],
            parameter_coefficient=coefficients[1],
            token_coefficient=coefficients[2],
            parameter_exponent=coefficients[3],
            token_exponent=coefficients[4],
            holdout_mape=mape,
            parameter_exponent_ci=(
                parameter_samples[low_index],
                parameter_samples[high_index],
            ),
            token_exponent_ci=(token_samples[low_index], token_samples[high_index]),
            predictions=predictions,
            compute_optimal=compute_optimal,
            distinct_shapes=len(unique_shapes),
            recommended_next_experiment=(
                "run the evidence-bound 7B scratch campaign"
                if passed
                else "add intermediate scaling pilots; extrapolation error exceeds five percent"
            ),
            blockers=[] if passed else ["scaling-law holdout MAPE exceeds policy"],
        )
    except Exception as exc:
        return ScalingLawReport(
            status="failed",
            recommended_next_experiment="add scientifically identifiable scaling measurements",
            blockers=[str(exc)],
        )


def compare_scaling_campaign(
    manifest: ExperimentManifest,
    evidence: Sequence[ArmEvidence],
) -> StatisticalComparisonReport:
    report = fit_scaling_law(manifest, evidence)
    return StatisticalComparisonReport(
        experiment_id=manifest.experiment_id,
        experiment_type="scaling_law",
        status=report.status,
        scaling_law=report,
        selected_candidate="7b" if report.status == "passed" else None,
        blockers=report.blockers,
    ).sealed()


def compare_experiment(
    manifest: ExperimentManifest,
    evidence: Sequence[ArmEvidence],
    admission_blockers: Sequence[str] = (),
) -> StatisticalComparisonReport:
    if admission_blockers:
        return StatisticalComparisonReport(
            experiment_id=manifest.experiment_id,
            experiment_type=manifest.campaign.experiment_type,
            status="blocked",
            blockers=list(admission_blockers),
        ).sealed()
    if len(evidence) != len(manifest.arms):
        return StatisticalComparisonReport(
            experiment_id=manifest.experiment_id,
            experiment_type=manifest.campaign.experiment_type,
            status="blocked",
            blockers=["admitted evidence count does not match planned arm count"],
        ).sealed()
    if manifest.campaign.experiment_type == "tokenizer_selection":
        return compare_tokenizer_candidates(manifest, evidence)
    if manifest.campaign.experiment_type == "architecture_ab":
        return compare_architecture_candidates(manifest, evidence)
    return compare_scaling_campaign(manifest, evidence)


def decide_experiment(
    manifest: ExperimentManifest,
    comparison: StatisticalComparisonReport,
) -> ExperimentDecision:
    expected_comparison = _digest_model(comparison, omitted={"report_sha256"})
    if not hmac.compare_digest(expected_comparison, comparison.report_sha256):
        raise ValueError("statistical comparison report has been modified")
    if comparison.experiment_id != manifest.experiment_id:
        raise ValueError("comparison belongs to a different experiment")
    passed = comparison.status == "passed" and bool(comparison.selected_candidate)
    rationale = [
        f"campaign={manifest.campaign.campaign_id}",
        f"experiment_type={manifest.campaign.experiment_type}",
        (
            f"selected={comparison.selected_candidate}"
            if passed
            else "no candidate satisfied every immutable scientific gate"
        ),
    ]
    if manifest.campaign.experiment_type == "architecture_ab" and passed:
        rationale.append(
            "all MoE advancement gates passed"
            if comparison.selected_candidate == "1b_moe"
            else "MoE advancement gates were not all satisfied; the valid dense baseline was selected"
        )
    return ExperimentDecision(
        experiment_id=manifest.experiment_id,
        status="passed" if passed else comparison.status,
        selected_candidate=comparison.selected_candidate if passed else None,
        manifest_sha256=manifest.manifest_sha256,
        comparison_sha256=comparison.report_sha256,
        rationale=rationale,
        blockers=[] if passed else comparison.blockers,
    ).sealed()


def promote_experiment(decision: ExperimentDecision) -> PromotionDecision:
    expected = _digest_model(decision, omitted={"decision_sha256"})
    if not decision.decision_sha256 or not hmac.compare_digest(
        expected, decision.decision_sha256
    ):
        raise ValueError("experiment decision has been modified")
    promoted = decision.status == "passed" and bool(decision.selected_candidate)
    return PromotionDecision(
        experiment_id=decision.experiment_id,
        status="promoted" if promoted else "blocked",
        promoted_candidate=decision.selected_candidate if promoted else None,
        experiment_decision_sha256=decision.decision_sha256,
        blockers=[]
        if promoted
        else [*decision.blockers, "scientific experiment did not pass"],
    ).sealed()


def verify_promotion_chain(
    promotion_path: str | Path,
) -> tuple[
    PromotionDecision,
    ExperimentDecision,
    StatisticalComparisonReport,
    ExperimentManifest,
]:
    """Replay a colocated promotion chain and verify every semantic digest."""

    promotion_source = Path(promotion_path).expanduser().resolve(strict=True)
    root = promotion_source.parent
    promotion = PromotionDecision.model_validate(_json_object(promotion_source))
    expected_promotion = _digest_model(promotion, omitted={"promotion_sha256"})
    if not promotion.promotion_sha256 or not hmac.compare_digest(
        expected_promotion, promotion.promotion_sha256
    ):
        raise ValueError("scientific promotion decision has been modified")
    decision = ExperimentDecision.model_validate(
        _json_object(root / "experiment_decision.json")
    )
    expected_decision = _digest_model(decision, omitted={"decision_sha256"})
    if not decision.decision_sha256 or not hmac.compare_digest(
        expected_decision, decision.decision_sha256
    ):
        raise ValueError("scientific experiment decision has been modified")
    if not hmac.compare_digest(
        promotion.experiment_decision_sha256, decision.decision_sha256
    ):
        raise ValueError("promotion is not bound to the colocated experiment decision")
    comparison = StatisticalComparisonReport.model_validate(
        _json_object(root / "statistical_comparison.json")
    )
    expected_comparison = _digest_model(comparison, omitted={"report_sha256"})
    if not comparison.report_sha256 or not hmac.compare_digest(
        expected_comparison, comparison.report_sha256
    ):
        raise ValueError("scientific comparison report has been modified")
    if not hmac.compare_digest(decision.comparison_sha256, comparison.report_sha256):
        raise ValueError("experiment decision is not bound to the comparison report")
    manifest = ExperimentManifest.model_validate(
        _json_object(root / "experiment_manifest.json")
    )
    manifest.verify()
    if not hmac.compare_digest(decision.manifest_sha256, manifest.manifest_sha256):
        raise ValueError("experiment decision is not bound to the experiment manifest")
    identities = {
        promotion.experiment_id,
        decision.experiment_id,
        comparison.experiment_id,
        manifest.experiment_id,
    }
    if len(identities) != 1:
        raise ValueError(
            "scientific promotion chain contains mixed experiment identities"
        )
    arm_reports_path = root / "arm_reports.json"
    if not comparison.arm_reports_sha256:
        raise ValueError("scientific comparison is not bound to admitted arm evidence")
    if not arm_reports_path.is_file() or not hmac.compare_digest(
        sha256_file(arm_reports_path),
        comparison.arm_reports_sha256,
    ):
        raise ValueError("scientific arm report binding changed")
    arm_reports = _json_object(arm_reports_path)
    if arm_reports.get("experiment_id") != manifest.experiment_id:
        raise ValueError("arm reports belong to a different experiment")
    raw_bindings = arm_reports.get("evidence")
    if not isinstance(raw_bindings, dict):
        raise ValueError("arm reports do not contain evidence bindings")
    planned = {arm.arm_id: arm for arm in manifest.arms}
    if set(raw_bindings) != set(planned):
        raise ValueError("arm report evidence set differs from the preregistered arms")
    for arm_id, raw_binding in raw_bindings.items():
        artifact = BoundArtifact.model_validate(raw_binding)
        artifact.verify()
        evidence = ArmEvidence.model_validate(_json_object(artifact.path))
        if evidence.arm_id != arm_id:
            raise ValueError(f"arm evidence identity mismatch: {arm_id}")
        evidence.verify_artifacts(
            arm=planned[arm_id],
            evaluation_inputs=manifest.evaluation_inputs,
        )
    return promotion, decision, comparison, manifest


def build_model_progression_decision(
    *,
    tokenizer_promotion_path: str | Path,
    architecture_promotion_path: str | Path,
    scaling_promotion_path: str | Path,
) -> ModelProgressionDecision:
    """Authorize a 7B experiment only from three compatible promotion chains."""

    chains = {
        "tokenizer_selection": verify_promotion_chain(tokenizer_promotion_path),
        "architecture_ab": verify_promotion_chain(architecture_promotion_path),
        "scaling_law": verify_promotion_chain(scaling_promotion_path),
    }
    blockers: list[str] = []
    by_type: dict[
        str,
        tuple[
            PromotionDecision,
            ExperimentDecision,
            StatisticalComparisonReport,
            ExperimentManifest,
        ],
    ] = {}
    for supplied_name, chain in chains.items():
        actual_name = chain[3].campaign.experiment_type
        if actual_name != supplied_name:
            blockers.append(
                f"{supplied_name} input contains {actual_name} promotion evidence"
            )
        if actual_name in by_type:
            blockers.append(f"duplicate scientific promotion type: {actual_name}")
        by_type[actual_name] = chain
        if chain[0].status != "promoted":
            blockers.append(f"{supplied_name} scientific promotion did not pass")

    tokenizer_chain = by_type.get("tokenizer_selection")
    architecture_chain = by_type.get("architecture_ab")
    scaling_chain = by_type.get("scaling_law")
    if tokenizer_chain is None or architecture_chain is None or scaling_chain is None:
        raise ValueError(
            "model progression requires tokenizer, architecture, and scaling promotion chains"
        )

    manifests = [tokenizer_chain[3], architecture_chain[3], scaling_chain[3]]
    for binding_name in ("dataset_manifest", "split_manifest", "evaluation_manifest"):
        digests = {manifest.bindings[binding_name].sha256 for manifest in manifests}
        if len(digests) != 1:
            blockers.append(
                f"scientific campaigns do not share one {binding_name} lineage"
            )

    tokenizer_candidate = tokenizer_chain[0].promoted_candidate
    architecture_candidate = architecture_chain[0].promoted_candidate
    scaling_candidate = scaling_chain[0].promoted_candidate
    selected_tokenizer_artifact: BoundArtifact | None = None
    selected_vocab_size: int | None = None
    if (
        tokenizer_candidate is None
        or tokenizer_candidate not in tokenizer_chain[3].tokenizers
    ):
        blockers.append("tokenizer promotion does not identify a bound candidate")
    else:
        selected_vocab_size = int(tokenizer_candidate)
        selected_tokenizer_artifact = tokenizer_chain[3].tokenizers[tokenizer_candidate]
        for label, manifest in (
            ("architecture", architecture_chain[3]),
            ("scaling", scaling_chain[3]),
        ):
            selected = manifest.tokenizers.get("selected")
            if selected is None or not hmac.compare_digest(
                selected.sha256,
                selected_tokenizer_artifact.sha256,
            ):
                blockers.append(
                    f"{label} campaign is not bound to the tokenizer selected by the tokenizer campaign"
                )

    if architecture_candidate not in {"1b", "1b_moe"}:
        blockers.append(
            "architecture promotion did not select a supported 1B confirmation"
        )
    if scaling_candidate != "7b":
        blockers.append("scaling-law promotion did not authorize the 7B experiment")
    target_profile = (
        "7b_moe"
        if architecture_candidate == "1b_moe"
        else "7b"
        if architecture_candidate == "1b"
        else None
    )
    target_contract_sha256: str | None = None
    if target_profile is not None:
        try:
            target = model_profile(target_profile)
            if selected_vocab_size is not None:
                target = target.model_copy(update={"vocab_size": selected_vocab_size})
            target.parameter_report()
            target_contract_sha256 = target.contract_sha256()
        except Exception as exc:
            blockers.append(f"target model profile is invalid: {exc}")

    decision = ModelProgressionDecision(
        status="authorized" if not blockers else "blocked",
        selected_tokenizer_vocab_size=selected_vocab_size if not blockers else None,
        selected_architecture=architecture_candidate if not blockers else None,
        target_model_profile=target_profile if not blockers else None,
        target_model_contract_sha256=target_contract_sha256 if not blockers else None,
        tokenizer_promotion_sha256=tokenizer_chain[0].promotion_sha256,
        architecture_promotion_sha256=architecture_chain[0].promotion_sha256,
        scaling_promotion_sha256=scaling_chain[0].promotion_sha256,
        dataset_manifest_sha256=tokenizer_chain[3].bindings["dataset_manifest"].sha256,
        split_manifest_sha256=tokenizer_chain[3].bindings["split_manifest"].sha256,
        evaluation_manifest_sha256=tokenizer_chain[3]
        .bindings["evaluation_manifest"]
        .sha256,
        selected_tokenizer_manifest_sha256=(
            selected_tokenizer_artifact.sha256
            if selected_tokenizer_artifact and not blockers
            else None
        ),
        blockers=sorted(set(blockers)),
    ).sealed()
    return decision


def verify_model_progression_decision(path: str | Path) -> ModelProgressionDecision:
    source = Path(path).expanduser().resolve(strict=True)
    decision = ModelProgressionDecision.model_validate(_json_object(source))
    expected = _digest_model(decision, omitted={"decision_sha256"})
    if not decision.decision_sha256 or not hmac.compare_digest(
        expected, decision.decision_sha256
    ):
        raise ValueError("model progression decision has been modified")
    return decision


def build_arm_execution_requests(
    manifest: ExperimentManifest,
) -> list[ScientificArmExecutionRequest]:
    """Translate preregistered arms into exact, scheduler-ready workloads."""

    from src.craftly.training_workspace import TrainingProfileRegistry

    policy_path = Path(manifest.bindings["optimizer_policy"].path)
    try:
        registry = TrainingProfileRegistry.from_file(policy_path)
        profile = registry.latest(manifest.campaign.training_profile_id)
    except Exception as exc:
        raise ValueError(
            "scientific optimizer policy must be a valid training profile registry "
            f"containing {manifest.campaign.training_profile_id!r}: {exc}"
        ) from exc
    world_size = (
        1
        if profile.scheduler == "notebook"
        else profile.resources.nodes * profile.resources.gpus_per_node
    )
    tokens_per_step = (
        profile.sequence_length
        * profile.batch_size
        * profile.gradient_accumulation_steps
        * world_size
    )
    requests: list[ScientificArmExecutionRequest] = []
    for arm in manifest.arms:
        if arm.token_budget % tokens_per_step:
            raise ValueError(
                f"arm {arm.arm_id} token budget {arm.token_budget} is not divisible by "
                f"the immutable training shape {tokens_per_step} tokens/optimizer-step"
            )
        tokenizer = _manifest_tokenizer_contract(manifest, arm)
        requests.append(
            ScientificArmExecutionRequest(
                experiment_id=manifest.experiment_id,
                arm_id=arm.arm_id,
                scheduler=profile.scheduler,
                distributed_strategy=profile.distributed_strategy,
                training_profile_id=arm.training_profile_id,
                model_profile=arm.model_profile,
                model_contract_sha256=arm.model_contract_sha256,
                total_parameters=arm.total_parameters,
                active_parameters=arm.active_parameters,
                canonical_training_flops=arm.canonical_training_flops,
                model_seed=arm.seed,
                dataloader_seed=arm.seed,
                world_size=world_size,
                optimizer_steps=arm.token_budget // tokens_per_step,
                token_budget=arm.token_budget,
                tokens_per_optimizer_step=tokens_per_step,
                sequence_length=profile.sequence_length,
                batch_size=profile.batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                dtype=profile.dtype,
                tokenizer_manifest_path=manifest.tokenizers[
                    str(arm.vocab_size)
                    if manifest.campaign.experiment_type == "tokenizer_selection"
                    else "selected"
                ].path,
                tokenizer_manifest_sha256=manifest.tokenizers[
                    str(arm.vocab_size)
                    if manifest.campaign.experiment_type == "tokenizer_selection"
                    else "selected"
                ].sha256,
                shard_manifest_path=tokenizer.shard_manifest_path,
                shard_manifest_sha256=tokenizer.shard_manifest_sha256,
                dataset_manifest_path=manifest.bindings["dataset_manifest"].path,
                dataset_manifest_sha256=manifest.bindings["dataset_manifest"].sha256,
                split_manifest_path=manifest.bindings["split_manifest"].path,
                split_manifest_sha256=manifest.bindings["split_manifest"].sha256,
                optimizer_policy_path=manifest.bindings["optimizer_policy"].path,
                optimizer_policy_sha256=manifest.bindings["optimizer_policy"].sha256,
                evaluation_manifest_path=manifest.bindings["evaluation_manifest"].path,
                evaluation_manifest_sha256=manifest.bindings[
                    "evaluation_manifest"
                ].sha256,
                container_digest=manifest.container_digest,
                required_evaluation_suites=arm.required_evaluation_suites,
            )
        )
    return requests


def admit_arm_evidence_from_reports(
    *,
    experiment_dir: str | Path,
    arm_id: str,
    training_report_path: str | Path,
    evaluation_report_path: str | Path,
    generation_audit_path: str | Path,
    tokenizer_audit_path: str | Path,
    output_dir: str | Path | None = None,
) -> ArmEvidence:
    """Derive immutable arm evidence from reports owned by existing authorities."""

    manifest = _load_manifest(experiment_dir)
    arms = {arm.arm_id: arm for arm in manifest.arms}
    arm = arms.get(arm_id)
    if arm is None:
        raise ValueError(
            f"arm is not present in the immutable experiment plan: {arm_id}"
        )
    training_artifact = BoundArtifact.bind("training", training_report_path)
    evaluation_artifact = BoundArtifact.bind("evaluation", evaluation_report_path)
    generation_artifact = BoundArtifact.bind("generation_audit", generation_audit_path)
    tokenizer_artifact = BoundArtifact.bind("tokenizer_audit", tokenizer_audit_path)
    training = _json_object(training_artifact.path)
    evaluation = _json_object(evaluation_artifact.path)
    generation = _json_object(generation_artifact.path)
    tokenizer = _json_object(tokenizer_artifact.path)

    checkpoint_path = str(
        training.get("best_checkpoint_manifest")
        or training.get("checkpoint_manifest")
        or ""
    )
    if not checkpoint_path:
        raise ValueError(
            "training report does not identify its selected checkpoint manifest"
        )
    checkpoint_artifact = BoundArtifact.bind("checkpoint", checkpoint_path)
    suites = evaluation.get("suites")
    if not isinstance(suites, list):
        raise ValueError("evaluation report does not contain suite results")
    foundation_kinds = {
        "human_eval_style",
        "mbpp_style",
        "swe_bench_style",
        "repoqa_style",
    }
    security_kinds = {"cyberseceval_style", "custom_security"}
    foundation_scores = [
        float(row["score"])
        for row in suites
        if isinstance(row, dict)
        and row.get("kind") in foundation_kinds
        and row.get("status") == "passed"
    ]
    security_scores = [
        float(row["score"])
        for row in suites
        if isinstance(row, dict)
        and row.get("kind") in security_kinds
        and row.get("status") == "passed"
    ]
    if not foundation_scores or not security_scores:
        raise ValueError(
            "evaluation report requires passed foundation and security suites"
        )
    token_statistics = tokenizer.get("token_statistics")
    if not isinstance(token_statistics, dict):
        raise ValueError("tokenizer audit is missing token statistics")
    router = training.get("router_metrics")
    if not isinstance(router, dict):
        router = {}
    raw_router_ratio = router.get("maximum_p99_to_mean_load")
    router_ratio = (
        float(raw_router_ratio)
        if raw_router_ratio is not None and float(raw_router_ratio) >= 1.0
        else None
    )
    candidate = generation.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("generation audit is missing candidate results")
    tokenizer_contract = _manifest_tokenizer_contract(manifest, arm)
    evidence = ArmEvidence(
        arm_id=arm.arm_id,
        status="passed",
        seed=arm.seed,
        objective=manifest.objective,
        dataset_manifest_sha256=manifest.bindings["dataset_manifest"].sha256,
        split_manifest_sha256=manifest.bindings["split_manifest"].sha256,
        optimizer_policy_sha256=manifest.bindings["optimizer_policy"].sha256,
        evaluation_manifest_sha256=manifest.bindings["evaluation_manifest"].sha256,
        model_contract_sha256=arm.model_contract_sha256,
        tokenizer_sha256=tokenizer_contract.tokenizer_sha256,
        tokenizer_vocab_size=arm.vocab_size,
        trained_tokens=int(training.get("trained_tokens", -1)),
        training_flops=arm.canonical_training_flops,
        total_parameters=arm.total_parameters,
        active_parameters=arm.active_parameters,
        validation_loss=float(training.get("best_validation_loss", math.inf)),
        executable_benchmark_score=float(evaluation.get("aggregate_score", math.nan)),
        foundation_score=statistics.fmean(foundation_scores),
        security_score=statistics.fmean(security_scores),
        tokens_per_byte=float(token_statistics.get("tokens_per_byte", math.nan)),
        checkpoint_reload_parity=(
            training.get("checkpoint_reload_verified") is True
            and training.get("checkpoint_reload_logit_parity") is True
        ),
        generation_collapsed=int(candidate.get("collapsed_count", -1)) > 0,
        dropped_tokens=int(router.get("dropped_assignments", 0)),
        router_p99_to_mean=router_ratio,
        evaluation_authority=PROMOTION_EVALUATION_AUTHORITY,
        training_report=training_artifact,
        evaluation_report=evaluation_artifact,
        generation_audit=generation_artifact,
        checkpoint_manifest=checkpoint_artifact,
        tokenizer_audit=tokenizer_artifact,
    )
    evidence.verify_artifacts(arm=arm, evaluation_inputs=manifest.evaluation_inputs)
    destination = (
        Path(output_dir or (Path(experiment_dir) / "arm-evidence"))
        .expanduser()
        .resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    _exclusive_json(
        destination / f"{arm.arm_id}.json", evidence.model_dump(mode="json")
    )
    return evidence


def assemble_scientific_evaluation_report(
    *,
    experiment_dir: str | Path,
    code_benchmark_report_path: str | Path,
    repository_scorecard_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Combine executable code and repository evidence without re-scoring either."""

    manifest = _load_manifest(experiment_dir)
    code_artifact = BoundArtifact.bind(
        "code_benchmark_report", code_benchmark_report_path
    )
    scorecard_artifact = BoundArtifact.bind(
        "repository_scorecard_report",
        repository_scorecard_report_path,
    )
    code = _json_object(code_artifact.path)
    scorecard = _json_object(scorecard_artifact.path)
    if (
        code.get("status") != "passed"
        or code.get("evaluation_mode") != "executable_model"
    ):
        raise ValueError(
            "code benchmark report is not passed executable-model evidence"
        )
    if not hmac.compare_digest(
        str(code.get("evaluation_manifest_sha256") or ""),
        manifest.bindings["evaluation_manifest"].sha256,
    ):
        raise ValueError("code benchmark report uses a different evaluation manifest")
    checkpoint_sha256 = str(code.get("checkpoint_manifest_sha256") or "")
    tokenizer_sha256 = str(code.get("tokenizer_sha256") or "")
    if (
        HEX_SHA256.fullmatch(checkpoint_sha256) is None
        or HEX_SHA256.fullmatch(tokenizer_sha256) is None
    ):
        raise ValueError("code benchmark report lacks checkpoint or tokenizer identity")
    if scorecard.get("status") != "passed" or scorecard.get("policy_mode") != "strict":
        raise ValueError("repository scorecard is not a passed strict run")
    if int(scorecard.get("task_count", 0)) < 50:
        raise ValueError("repository scorecard contains fewer than 50 governed tasks")
    model_evidence = scorecard.get("model_evidence")
    if not isinstance(model_evidence, dict):
        raise ValueError("repository scorecard lacks model evidence")
    if not hmac.compare_digest(
        str(model_evidence.get("checkpoint_manifest_sha256") or ""),
        checkpoint_sha256,
    ):
        raise ValueError("repository scorecard checkpoint differs from code evaluation")
    if not hmac.compare_digest(
        str(model_evidence.get("tokenizer_sha256") or ""),
        tokenizer_sha256,
    ):
        raise ValueError("repository scorecard tokenizer differs from code evaluation")

    expected_suite_hashes = {
        key.removeprefix("suite:"): artifact.sha256
        for key, artifact in manifest.evaluation_inputs.items()
        if key.startswith("suite:")
    }
    reported_code_hashes = code.get("suite_artifact_sha256")
    if not isinstance(reported_code_hashes, dict):
        raise ValueError("code benchmark report lacks suite hashes")
    raw_suites = code.get("suites")
    if not isinstance(raw_suites, list):
        raise ValueError("code benchmark report lacks suite results")
    suites = [dict(row) for row in raw_suites if isinstance(row, dict)]
    code_names = {str(row.get("name") or "") for row in suites}
    for name in code_names:
        expected = expected_suite_hashes.get(name)
        if expected is None or not hmac.compare_digest(
            str(reported_code_hashes.get(name) or ""),
            expected,
        ):
            raise ValueError(
                f"code benchmark suite hash differs from the experiment: {name}"
            )

    scorecard_hash = str(scorecard.get("task_suite_sha256") or "")
    scorecard_suite_rows = {
        "CraftlyDefensiveSecurity": {
            "name": "CraftlyDefensiveSecurity",
            "kind": "custom_security",
            "status": "passed",
            "score": float(scorecard.get("security_detection_fix_score", math.nan)),
            "total": sum(
                1
                for row in scorecard.get("tasks", [])
                if isinstance(row, dict)
                and row.get("category") in {"security", "patch"}
            ),
            "passed": sum(
                1
                for row in scorecard.get("tasks", [])
                if isinstance(row, dict)
                and row.get("category") in {"security", "patch"}
                and row.get("security_passed") is True
                and row.get("tests_passed") is True
            ),
            "reason": "passed strict governed repository security scorecard",
            "report": {"source_report_sha256": scorecard_artifact.sha256},
        },
        "CraftlyRepositoryScorecard": {
            "name": "CraftlyRepositoryScorecard",
            "kind": "swe_bench_style",
            "status": "passed",
            "score": float(scorecard.get("average_score", math.nan)),
            "total": int(scorecard.get("task_count", 0)),
            "passed": sum(
                1
                for row in scorecard.get("tasks", [])
                if isinstance(row, dict)
                and row.get("accepted") is True
                and row.get("tests_passed") is True
            ),
            "reason": "passed strict governed repository workflow scorecard",
            "report": {"source_report_sha256": scorecard_artifact.sha256},
        },
    }
    for name, row in scorecard_suite_rows.items():
        if name not in manifest.campaign.required_evaluation_suites:
            continue
        if not hmac.compare_digest(scorecard_hash, expected_suite_hashes.get(name, "")):
            raise ValueError(
                f"repository scorecard task-suite hash differs from the experiment: {name}"
            )
        suites.append(row)

    suite_names = {str(row.get("name") or "") for row in suites}
    missing = sorted(set(manifest.campaign.required_evaluation_suites) - suite_names)
    if missing:
        raise ValueError(f"assembled evaluation is missing required suites: {missing}")
    scores = [float(row["score"]) for row in suites if row.get("status") == "passed"]
    if len(scores) != len(suites) or any(not math.isfinite(score) for score in scores):
        raise ValueError(
            "assembled evaluation contains failed or non-finite suite evidence"
        )
    payload = {
        "schema_version": 2,
        "status": "passed",
        "evaluation_mode": "executable_model",
        "suites": suites,
        "aggregate_score": round(statistics.fmean(scores), 6),
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "evaluation_manifest_sha256": manifest.bindings["evaluation_manifest"].sha256,
        "suite_artifact_sha256": expected_suite_hashes,
        "source_reports": {
            "code_benchmark": code_artifact.model_dump(mode="json"),
            "repository_scorecard": scorecard_artifact.model_dump(mode="json"),
        },
        "created_at_unix": time.time(),
    }
    target = Path(output_path).expanduser().resolve()
    return _exclusive_json(target, payload)


class ExperimentAuthority:
    def __init__(self, experiment_dir: str | Path) -> None:
        self.root = Path(experiment_dir).expanduser().resolve()

    def plan(
        self,
        *,
        campaign: ScientificExperimentCampaignContract,
        dataset_manifest: str | Path,
        split_manifest: str | Path,
        optimizer_policy: str | Path,
        evaluation_manifest: str | Path,
        tokenizer_manifests: dict[str, str | Path],
        container_digest: str,
    ) -> ExperimentManifest:
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("experiment directory must be new and empty")
        manifest = create_experiment_manifest(
            campaign=campaign,
            dataset_manifest=dataset_manifest,
            split_manifest=split_manifest,
            optimizer_policy=optimizer_policy,
            evaluation_manifest=evaluation_manifest,
            tokenizer_manifests=tokenizer_manifests,
            container_digest=container_digest,
        )
        execution_requests = build_arm_execution_requests(manifest)
        requests = {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "status": "not_run",
            "arms": [request.model_dump(mode="json") for request in execution_requests],
        }
        _atomic_json(
            self.root / "experiment_manifest.json", manifest.model_dump(mode="json")
        )
        _atomic_json(self.root / "arm_execution_requests.json", requests)
        return manifest

    def inspect(self, *, evidence_dir: str | Path | None = None) -> dict[str, Any]:
        manifest = _load_manifest(self.root)
        evidence, blockers = _load_arm_evidence(
            manifest, evidence_dir or (self.root / "arm-evidence")
        )
        return {
            "experiment_id": manifest.experiment_id,
            "campaign_id": manifest.campaign.campaign_id,
            "experiment_type": manifest.campaign.experiment_type,
            "status": "ready_to_compare"
            if not blockers and len(evidence) == len(manifest.arms)
            else "blocked",
            "planned_arms": len(manifest.arms),
            "admitted_arms": len(evidence),
            "blockers": blockers,
        }

    def run(
        self, *, evidence_dir: str | Path | None = None
    ) -> StatisticalComparisonReport:
        if (self.root / "experiment_decision.json").exists() or (
            self.root / "promotion_decision.json"
        ).exists():
            raise RuntimeError(
                "a decided experiment is immutable; create a new experiment for new evidence"
            )
        manifest = _load_manifest(self.root)
        evidence, blockers = _load_arm_evidence(
            manifest, evidence_dir or (self.root / "arm-evidence")
        )
        evidence_bindings = {
            item.arm_id: BoundArtifact.bind(
                item.arm_id,
                Path(evidence_dir or (self.root / "arm-evidence"))
                / f"{item.arm_id}.json",
            ).model_dump(mode="json")
            for item in evidence
        }
        arm_reports_path = _atomic_json(
            self.root / "arm_reports.json",
            {
                "schema_version": 1,
                "experiment_id": manifest.experiment_id,
                "status": "passed" if not blockers else "blocked",
                "evidence": evidence_bindings,
                "blockers": blockers,
            },
        )
        comparison = compare_experiment(manifest, evidence, blockers)
        comparison = comparison.model_copy(
            update={
                "arm_reports_sha256": sha256_file(arm_reports_path),
                "report_sha256": "",
            }
        ).sealed()
        _atomic_json(
            self.root / "statistical_comparison.json",
            comparison.model_dump(mode="json"),
        )
        if comparison.scaling_law is not None:
            _atomic_json(
                self.root / "scaling_law_report.json",
                comparison.scaling_law.model_dump(mode="json"),
            )
        return comparison

    def decide(self) -> ExperimentDecision:
        if (self.root / "experiment_decision.json").exists():
            raise FileExistsError("experiment decision is immutable and already exists")
        manifest = _load_manifest(self.root)
        comparison = StatisticalComparisonReport.model_validate(
            _json_object(self.root / "statistical_comparison.json")
        )
        decision = decide_experiment(manifest, comparison)
        _exclusive_json(
            self.root / "experiment_decision.json", decision.model_dump(mode="json")
        )
        return decision

    def promote(self) -> PromotionDecision:
        if (self.root / "promotion_decision.json").exists():
            raise FileExistsError("promotion decision is immutable and already exists")
        decision = ExperimentDecision.model_validate(
            _json_object(self.root / "experiment_decision.json")
        )
        promotion = promote_experiment(decision)
        _exclusive_json(
            self.root / "promotion_decision.json", promotion.model_dump(mode="json")
        )
        return promotion


def run_ablation(
    *,
    input_paths: list[str | Path],
    mix_config: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    sequence_length: int = 2048,
) -> AblationReport:
    config = load_mix_config(mix_config)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifests: list[MixManifest] = []
    for experiment in config.experiments:
        manifests.append(
            build_mix(
                input_paths=input_paths,
                config_path=mix_config,
                experiment=experiment.name,
                output_dir=root / experiment.name,
                tokenizer_path=tokenizer_path,
                sequence_length=sequence_length,
            )
        )
    experiments = [
        {
            "name": manifest.experiment,
            "total_rows": manifest.total_rows,
            "total_tokens": manifest.total_tokens,
            "output_jsonl": manifest.output_jsonl,
            "shard_manifest": manifest.shard_manifest,
            "buckets": [bucket.model_dump() for bucket in manifest.buckets],
        }
        for manifest in manifests
    ]
    report = AblationReport(
        mix_config=str(mix_config),
        input_paths=[str(path) for path in input_paths],
        output_dir=str(root),
        experiments=experiments,
        recommendation=(
            "This report only prepares data mixes. Register measured scratch-training arms "
            "with ExperimentAuthority before any scientific or production promotion."
        ),
    )
    _atomic_json(root / "ablation_report.json", report.model_dump(mode="json"))
    write_markdown(report, root / "ablation_report.md")
    return report


def write_markdown(report: AblationReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Data Mix Preparation Report",
        "",
        f"- status: {report.status}",
        f"- recommendation: {report.recommendation}",
        "",
        "| experiment | rows | tokens |",
        "|---|---:|---:|",
    ]
    for item in report.experiments:
        lines.append(
            f"| {item['name']} | {item['total_rows']} | {item['total_tokens']} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _legacy_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Craftly data-mix ablations (legacy compatibility)."
    )
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--mix-config", default="config/mix_ratios.json")
    parser.add_argument("--base-run-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--sequence-length", type=int, default=2048)
    return parser.parse_args()


def _authority_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Craftly scientific experiment authority"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--campaign", required=True)
    plan.add_argument(
        "--registry", default="config/training_qualification_campaigns.json"
    )
    plan.add_argument("--dataset-manifest", required=True)
    plan.add_argument("--split-manifest", required=True)
    plan.add_argument("--optimizer-policy", required=True)
    plan.add_argument("--evaluation-manifest", required=True)
    plan.add_argument(
        "--tokenizer-manifest",
        action="append",
        required=True,
        metavar="KEY=PATH",
        help="Repeat as 32000=PATH, 64000=PATH, 128000=PATH for tokenizer selection; use selected=PATH otherwise.",
    )
    plan.add_argument("--container-digest", required=True)
    plan.add_argument("--output-dir", required=True)
    for name in ("run", "resume", "inspect", "compare", "decide", "promote"):
        command = commands.add_parser(name)
        command.add_argument("--experiment-dir", required=True)
        if name in {"run", "resume", "inspect", "compare"}:
            command.add_argument("--evidence-dir")
    admit = commands.add_parser("admit-arm")
    admit.add_argument("--experiment-dir", required=True)
    admit.add_argument("--arm-id", required=True)
    admit.add_argument("--training-report", required=True)
    admit.add_argument("--evaluation-report", required=True)
    admit.add_argument("--generation-audit", required=True)
    admit.add_argument("--tokenizer-audit", required=True)
    admit.add_argument("--evidence-dir")
    assemble = commands.add_parser("assemble-evaluation")
    assemble.add_argument("--experiment-dir", required=True)
    assemble.add_argument("--code-benchmark-report", required=True)
    assemble.add_argument("--repository-scorecard-report", required=True)
    assemble.add_argument("--output", required=True)
    progression = commands.add_parser("advance-7b")
    progression.add_argument("--tokenizer-promotion", required=True)
    progression.add_argument("--architecture-promotion", required=True)
    progression.add_argument("--scaling-promotion", required=True)
    progression.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _parse_keyed_paths(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        key = key.strip()
        path = path.strip()
        if separator != "=" or not key or not path:
            raise ValueError(f"expected KEY=PATH, received {value!r}")
        if key in parsed:
            raise ValueError(f"duplicate keyed path: {key}")
        parsed[key] = path
    return parsed


def _run_authority_cli(args: argparse.Namespace) -> int:
    if args.command == "plan":
        campaign = load_scientific_experiment_registry(args.registry).latest(
            args.campaign
        )
        manifest = ExperimentAuthority(args.output_dir).plan(
            campaign=campaign,
            dataset_manifest=args.dataset_manifest,
            split_manifest=args.split_manifest,
            optimizer_policy=args.optimizer_policy,
            evaluation_manifest=args.evaluation_manifest,
            tokenizer_manifests=_parse_keyed_paths(args.tokenizer_manifest),
            container_digest=args.container_digest,
        )
        payload: Any = manifest.model_dump(mode="json")
        code = 0
    elif args.command == "advance-7b":
        decision = build_model_progression_decision(
            tokenizer_promotion_path=args.tokenizer_promotion,
            architecture_promotion_path=args.architecture_promotion,
            scaling_promotion_path=args.scaling_promotion,
        )
        _exclusive_json(
            Path(args.output).expanduser().resolve(), decision.model_dump(mode="json")
        )
        payload = decision.model_dump(mode="json")
        code = 0 if decision.status == "authorized" else 2
    elif args.command == "admit-arm":
        evidence = admit_arm_evidence_from_reports(
            experiment_dir=args.experiment_dir,
            arm_id=args.arm_id,
            training_report_path=args.training_report,
            evaluation_report_path=args.evaluation_report,
            generation_audit_path=args.generation_audit,
            tokenizer_audit_path=args.tokenizer_audit,
            output_dir=args.evidence_dir,
        )
        payload = evidence.model_dump(mode="json")
        code = 0
    elif args.command == "assemble-evaluation":
        target = assemble_scientific_evaluation_report(
            experiment_dir=args.experiment_dir,
            code_benchmark_report_path=args.code_benchmark_report,
            repository_scorecard_report_path=args.repository_scorecard_report,
            output_path=args.output,
        )
        payload = {"status": "passed", "evaluation_report": str(target)}
        code = 0
    else:
        authority = ExperimentAuthority(args.experiment_dir)
        if args.command == "inspect":
            payload = authority.inspect(evidence_dir=args.evidence_dir)
            code = 0 if payload["status"] == "ready_to_compare" else 2
        elif args.command in {"run", "resume", "compare"}:
            report = authority.run(evidence_dir=args.evidence_dir)
            payload = report.model_dump(mode="json")
            code = 0 if report.status == "passed" else 2
        elif args.command == "decide":
            decision = authority.decide()
            payload = decision.model_dump(mode="json")
            code = 0 if decision.status == "passed" else 2
        else:
            promotion = authority.promote()
            payload = promotion.model_dump(mode="json")
            code = 0 if promotion.status == "promoted" else 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


def main() -> None:
    commands = {
        "plan",
        "run",
        "resume",
        "inspect",
        "compare",
        "decide",
        "promote",
        "admit-arm",
        "assemble-evaluation",
        "advance-7b",
    }
    # Scientific experiment control is the authoritative public interface. The
    # legacy flag-only mix invocation remains supported for compatibility, but
    # must not hide the evidence-gated commands from ``--help``.
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"} or sys.argv[1] in commands:
        raise SystemExit(_run_authority_cli(_authority_args()))
    args = _legacy_args()
    inputs = args.inputs
    if not inputs and args.base_run_dir:
        base = Path(args.base_run_dir)
        inputs = [
            str(path)
            for path in sorted(
                list(base.rglob("clean-*.jsonl"))
                + list(base.rglob("*clean*.jsonl"))
                + list(base.rglob("*.mixed.jsonl"))
            )
        ]
    if not inputs:
        raise SystemExit("--inputs or --base-run-dir containing JSONL data is required")
    report = run_ablation(
        input_paths=inputs,
        mix_config=args.mix_config,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
