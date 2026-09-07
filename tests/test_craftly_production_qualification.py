from __future__ import annotations

import json
import os
import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from hypothesis import given, settings, strategies as st

from src.craftly.deployment.production_qualification import (
    CanaryEvidence,
    ImmutableQualificationStore,
    LoadStagePolicy,
    OperatorNotificationEvidence,
    ProductionQualificationReport,
    QualificationCheck,
    QualificationPolicy,
    SecurityReviewEvidence,
    bind_evidence,
    check_observability,
    load_json_evidence,
    validate_canary,
    validate_scratch_training_chain,
    validate_security_review,
    validate_training_proofs,
    verify_scratch_advancement_report,
    write_scratch_advancement_report,
    utc_now,
)
from src.craftly.model_ops.foundation import sha256_file
from src.craftly.learning.production_dataset import ProductionDatasetManifest
from src.craftly.model_ops.native_serving import (
    ChatCompletionRequest,
    NativeServingState,
)


def policy_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_max_age_seconds": 3600,
        "minimum_security_reviewers": 2,
        "require_report_signature_in_production": True,
        "require_operator_notification_proof": True,
        "load_stages": [
            {
                "name": "small",
                "concurrency": 2,
                "requests": 4,
                "streaming_requests": 1,
                "maximum_error_rate": 0.01,
                "maximum_p95_latency_ms": 30000,
                "minimum_throughput_rps": 0.01,
            }
        ],
        "required_training_proofs": [
            "postgres_redis_minio_lifecycle",
            "ordered_event_stress_1000000",
            "docker_disaster_recovery_drill",
            "worker_loss_retry_recovery",
        ],
        "required_soak_seconds": [86400, 604800],
        "required_security_domains": [
            "authentication_authorization",
            "ssrf",
            "path_traversal",
            "sandbox_escape",
            "secrets_and_iam",
            "dependency_supply_chain",
            "container_kubernetes",
        ],
        "required_canary_percentages": [1, 10, 50, 100],
        "minimum_internal_canary_users": 1,
        "maximum_internal_canary_users": 5,
        "canary_maximum_error_rate": 0.01,
        "canary_maximum_p95_latency_ms": 30000,
        "canary_rollback_error_rate": 0.02,
        "canary_maximum_rollback_seconds": 120,
    }


def build_policy() -> QualificationPolicy:
    return QualificationPolicy.model_validate(policy_payload())


def build_report(report_id: str) -> ProductionQualificationReport:
    return ProductionQualificationReport(
        report_id=report_id,
        status="blocked",
        mode="validation",
        created_at="2026-07-20T00:00:00+00:00",
        git_commit="a" * 40,
        policy_sha256="b" * 64,
        environment={"python": "test"},
        checks=[
            QualificationCheck(
                subsystem="production_proof_baseline",
                status="passed",
                summary="test",
            )
        ],
    )


class CraftlyProductionQualificationTest(unittest.TestCase):
    def test_scratch_advancement_report_is_hash_bound_and_names_next_stage(self) -> None:
        check = validate_scratch_training_chain(
            calibration_200_decision=None,
            calibration_5k_decision=None,
            production_dataset_manifest=None,
            tokenizer_audit_report=None,
            overfit_sanity_report=None,
            t4_1k_training_report=None,
            t4_10k_training_report=None,
            maximum_age_seconds=3600,
        )
        self.assertEqual(check.metrics["next_required_stage"], "calibration-200-decision")
        with tempfile.TemporaryDirectory() as temp_dir:
            preflight_path = Path(temp_dir) / "calibration-preflight.json"
            preflight_path.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "approval_request_dir": "approval-requests",
                        "legal_evidence_dir": "source-approvals",
                        "blockers": [
                            "reviewer_qualification:passed qualification report is required",
                            "source_legal_approval:official-docs requires human approval",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            governed = validate_scratch_training_chain(
                calibration_preflight_report=str(preflight_path),
                calibration_200_decision=None,
                calibration_5k_decision=None,
                production_dataset_manifest=None,
                tokenizer_audit_report=None,
                overfit_sanity_report=None,
                t4_1k_training_report=None,
                t4_10k_training_report=None,
                maximum_age_seconds=3600,
            )
            self.assertEqual(governed.metrics["next_required_stage"], "reviewer-qualification-report")
            report, path = write_scratch_advancement_report(check, output_dir=temp_dir)
            verified = verify_scratch_advancement_report(path)
            self.assertEqual(verified.report_sha256, report.report_sha256)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["check"]["status"] = "passed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest verification failed"):
                verify_scratch_advancement_report(path)

    def test_scratch_training_chain_blocks_missing_and_rejects_tampering(self) -> None:
        missing = validate_scratch_training_chain(
            calibration_200_decision=None,
            calibration_5k_decision=None,
            production_dataset_manifest=None,
            tokenizer_audit_report=None,
            overfit_sanity_report=None,
            t4_1k_training_report=None,
            t4_10k_training_report=None,
            maximum_age_seconds=3600,
        )
        self.assertEqual(missing.status, "blocked")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "schema_version": 2,
                "calibration_id": "calibration",
                "status": "passed",
                "dev_test": False,
                "manifest_path": "manifest.json",
                "manifest_sha256": "a" * 64,
                "source_registry_sha256": "b" * 64,
                "trust_policy_sha256": "c" * 64,
                "reviewer_roster_sha256": "d" * 64,
                "reviewer_qualification_sha256": "e" * 64,
                "legal_evidence_sha256": "f" * 64,
                "protected_manifest_sha256": "1" * 64,
                "authority_evidence_sha256": "2" * 64,
                "checks": {"all": True},
                "metrics": {"records": 200},
                "issues": [],
                "created_at_unix": time.time(),
            }
            calibration_200 = root / "calibration-200.json"
            calibration_200.write_text(
                json.dumps(
                    {
                        **common,
                        "stage": "calibration_200",
                        "next_stage": "5k_calibration_allowed",
                        "prior_decision_sha256": None,
                    }
                ),
                encoding="utf-8",
            )
            calibration_5k = root / "calibration-5k.json"
            calibration_5k.write_text(
                json.dumps(
                    {
                        **common,
                        "calibration_id": "calibration-5k",
                        "stage": "calibration_5k",
                        "next_stage": "100k_dataset_build_allowed",
                        "prior_decision_sha256": sha256_file(calibration_200),
                        "metrics": {"records": 5_000},
                    }
                ),
                encoding="utf-8",
            )
            artifact_names = {
                "train",
                "val",
                "test",
                "split_manifest",
                "promotion_decision",
                "dedup_report",
                "contamination_report",
                "review_report",
                "verified_patch_report",
            }
            dataset_artifacts = {}
            dataset_artifact_hashes = {}
            for name in artifact_names:
                artifact = root / f"dataset-{name}.json"
                artifact.write_text(json.dumps({"name": name}), encoding="utf-8")
                dataset_artifacts[name] = str(artifact)
                dataset_artifact_hashes[name] = sha256_file(artifact)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "dataset_id": "foundation",
                        "version_id": "v1",
                        "status": "promoted",
                        "output_dir": str(root),
                        "dev_smoke": False,
                        "artifacts": dataset_artifacts,
                        "artifact_sha256": dataset_artifact_hashes,
                        "metrics": {"promoted_records": 100_000},
                        "issues": [],
                        "reports": {
                            "dataset_trust_metrics": {
                                "license_coverage": 1.0,
                                "provenance_coverage": 1.0,
                                "high_value_review_coverage": 1.0,
                                "verified_patch_evidence_coverage": 1.0,
                                "average_quality": 0.90,
                                "p10_quality": 0.80,
                                "secret_or_pii_hits": 0,
                                "maximum_source_token_fraction": 0.20,
                                "maximum_source_family_token_fraction": 0.35,
                            },
                            "split_manifest": {"cross_split_group_collisions": 0},
                        },
                        "governance_paths": {"trust_policy_path": str(root / "trust-policy.json")},
                        "advancement_decision_sha256": sha256_file(calibration_5k),
                        "promotion_decision": {"status": "promoted", "checks": {"all": True}},
                    }
                ),
                encoding="utf-8",
            )
            tokenizer_model = root / "tokenizer.json"
            tokenizer_model.write_text('{"model":"test"}', encoding="utf-8")
            tokenizer_manifest = root / "tokenizer-manifest.json"
            tokenizer_manifest.write_text('{"status":"passed"}', encoding="utf-8")
            shard_manifest = root / "shards.json"
            shard_payload = {
                "tokenizer_sha256": sha256_file(tokenizer_model),
                "dataset_manifest_sha256": sha256_file(dataset),
                "split_strategy": "pre_split_family_safe",
            }
            shard_manifest.write_text(json.dumps(shard_payload), encoding="utf-8")
            efficiency = root / "efficiency.json"
            efficiency.write_text('{"status":"passed"}', encoding="utf-8")
            tokenizer_source = root / "tokenizer-source.jsonl"
            tokenizer_source.write_text('{"text":"secure code"}\n', encoding="utf-8")
            tokenizer = root / "tokenizer-audit.json"
            tokenizer.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "tokenizer_path": str(tokenizer_model),
                        "tokenizer_sha256": sha256_file(tokenizer_model),
                        "tokenizer_manifest_path": str(tokenizer_manifest),
                        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest),
                        "shard_manifest_path": str(shard_manifest),
                        "shard_manifest_sha256": sha256_file(shard_manifest),
                        "efficiency_report_path": str(efficiency),
                        "efficiency_report_sha256": sha256_file(efficiency),
                        "dataset_manifest_path": str(dataset),
                        "dataset_manifest_sha256": sha256_file(dataset),
                        "source_sha256": {str(tokenizer_source): sha256_file(tokenizer_source)},
                        "split_strategy": "pre_split_family_safe",
                        "family_safe_split": True,
                        "vocab_size_requested": 128_000,
                        "vocab_size_actual": 128_000,
                        "special_tokens_missing": [],
                        "audit_failures": [],
                        "sample_token_counts": {},
                        "source_rows": 100_000,
                        "source_chars": 1_000_000,
                        "shard_manifest": shard_payload,
                    }
                ),
                encoding="utf-8",
            )
            overfit = root / "overfit-sanity.json"
            overfit.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "relative_loss_drop": 0.30,
                        "required_relative_loss_drop": 0.20,
                        "training_report": {
                            "scratch_only": True,
                            "checkpoint_reload_verified": True,
                            "checkpoint_reload_logit_parity": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def write_training(path: Path, steps: int) -> None:
                checkpoint = root / f"checkpoint-{steps}.json"
                checkpoint.write_text(json.dumps({"step": steps}), encoding="utf-8")
                best_checkpoint = root / f"best-checkpoint-{steps}.json"
                best_checkpoint.write_text(json.dumps({"step": steps, "best": True}), encoding="utf-8")
                path.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "scratch_only": True,
                            "steps": steps,
                            "checkpoint_reload_verified": True,
                            "checkpoint_reload_logit_parity": True,
                            "checkpoint_reload_max_abs_logit_difference": 0.0,
                            "dataset_manifest_sha256": sha256_file(shard_manifest),
                            "shard_manifest_sha256": sha256_file(shard_manifest),
                            "tokenizer_sha256": sha256_file(tokenizer_model),
                            "git_commit": "a" * 40,
                            "checkpoint_manifest": str(checkpoint),
                            "checkpoint_manifest_sha256": sha256_file(checkpoint),
                            "best_checkpoint_manifest": str(best_checkpoint),
                            "best_checkpoint_manifest_sha256": sha256_file(best_checkpoint),
                            "model_config": {
                                "hidden_size": 512,
                                "num_layers": 8,
                                "max_sequence_length": 256,
                                "vocab_size": 128_000,
                            },
                            "train_losses": [7.0, 6.0],
                            "validation_losses": [{"step": steps, "loss": 6.1}],
                        }
                    ),
                    encoding="utf-8",
                )

            t4_1k = root / "t4-1k.json"
            t4_10k = root / "t4-10k.json"
            write_training(t4_1k, 1_000)
            write_training(t4_10k, 10_000)
            kwargs = {
                "calibration_200_decision": str(calibration_200),
                "calibration_5k_decision": str(calibration_5k),
                "production_dataset_manifest": str(dataset),
                "tokenizer_audit_report": str(tokenizer),
                "overfit_sanity_report": str(overfit),
                "t4_1k_training_report": str(t4_1k),
                "t4_10k_training_report": str(t4_10k),
                "maximum_age_seconds": 3600,
            }
            dataset_contract = json.loads(dataset.read_text(encoding="utf-8"))
            with patch(
                "src.craftly.deployment.production_qualification.validate_dataset_manifest_for_promotion",
                return_value=ProductionDatasetManifest.model_validate(dataset_contract),
            ) as replay:
                valid = validate_scratch_training_chain(**kwargs)
            self.assertEqual(valid.status, "passed", valid.model_dump())
            replay.assert_called_once_with(str(dataset), trust_policy_path=str(root / "trust-policy.json"))
            tokenizer_promotion = root / "tokenizer-promotion.json"
            tokenizer_promotion.write_text('{"status":"promoted"}', encoding="utf-8")
            evidence_policy = build_policy().model_copy(
                update={
                    "tokenizer_selection_mode": "evidence_selected",
                    "required_tokenizer_vocab_size": None,
                }
            )
            experiment_manifest = SimpleNamespace(
                experiment_id="tokenizer-selection-test",
                campaign=SimpleNamespace(
                    experiment_type="tokenizer_selection",
                    candidate_vocab_sizes=[32_000, 64_000, 128_000],
                ),
                bindings={
                    "dataset_manifest": SimpleNamespace(sha256=sha256_file(dataset))
                },
            )
            with (
                patch(
                    "src.craftly.deployment.production_qualification.validate_dataset_manifest_for_promotion",
                    return_value=ProductionDatasetManifest.model_validate(dataset_contract),
                ),
                patch(
                    "src.craftly.deployment.production_qualification.verify_promotion_chain",
                    return_value=(
                        SimpleNamespace(
                            status="promoted",
                            promoted_candidate="128000",
                            promotion_sha256="9" * 64,
                        ),
                        SimpleNamespace(status="passed"),
                        SimpleNamespace(status="passed"),
                        experiment_manifest,
                    ),
                ),
            ):
                selected = validate_scratch_training_chain(
                    **kwargs,
                    tokenizer_selection_promotion=str(tokenizer_promotion),
                    policy=evidence_policy,
                )
            self.assertEqual(selected.status, "passed", selected.model_dump())
            self.assertEqual(selected.metrics["tokenizer_experiment_id"], "tokenizer-selection-test")
            with patch(
                "src.craftly.deployment.production_qualification.validate_dataset_manifest_for_promotion",
                side_effect=ValueError("stale legal approval evidence"),
            ):
                stale_governance = validate_scratch_training_chain(**kwargs)
            self.assertEqual(stale_governance.status, "failed")
            self.assertTrue(
                any("authority replay failed" in item for item in stale_governance.blockers),
                stale_governance.model_dump(),
            )
            tokenizer_source.write_text('{"text":"tampered tokenizer source"}\n', encoding="utf-8")
            with patch(
                "src.craftly.deployment.production_qualification.validate_dataset_manifest_for_promotion",
                return_value=ProductionDatasetManifest.model_validate(dataset_contract),
            ):
                source_tampered = validate_scratch_training_chain(**kwargs)
            self.assertEqual(source_tampered.status, "failed")
            self.assertTrue(
                any("tokenizer source SHA-256 mismatch" in item for item in source_tampered.blockers),
                source_tampered.model_dump(),
            )
            tokenizer_source.write_text('{"text":"secure code"}\n', encoding="utf-8")
            calibration_200.write_text(calibration_200.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with patch(
                "src.craftly.deployment.production_qualification.validate_dataset_manifest_for_promotion",
                return_value=ProductionDatasetManifest.model_validate(dataset_contract),
            ):
                tampered = validate_scratch_training_chain(**kwargs)
            self.assertEqual(tampered.status, "failed")
            self.assertTrue(any("hash-bound" in item for item in tampered.blockers))

    def test_observability_requires_operator_delivery_evidence(self) -> None:
        result = asyncio.run(
            check_observability(
                metrics_url="http://127.0.0.1:1/metrics",
                prometheus_url="http://127.0.0.1:1/-/ready",
                grafana_url="http://127.0.0.1:1/api/health",
                otel_health_url="http://127.0.0.1:1/",
                alertmanager_url="http://127.0.0.1:1",
                operator_notification_report=None,
                policy=build_policy(),
                allowed_insecure_hosts=["127.0.0.1"],
            )
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("--operator-notification-report is required", result.blockers)

    def test_operator_delivery_ids_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            OperatorNotificationEvidence(
                proof_id="notification-proof",
                created_at=utc_now(),
                provider="test-provider",
                channel_type="webhook",
                recipient_reference_sha256="a" * 64,
                firing_delivery_id="same",
                recovery_delivery_id="same",
                firing_delivered_at=utc_now(),
                recovery_delivered_at=utc_now(),
                firing_delivered=True,
                recovery_delivered=True,
                status="passed",
            )

    def test_immutable_report_is_versioned_signed_and_hash_chained(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ImmutableQualificationStore(root, signing_key="s" * 32)
            first_path = store.write(build_report("run-1"))
            first = ProductionQualificationReport.model_validate_json(
                first_path.read_text(encoding="utf-8")
            )
            self.assertEqual(first.signature_algorithm, "HMAC-SHA256")
            self.assertEqual(len(first.report_sha256), 64)
            self.assertEqual(
                sha256_file(first_path),
                json.loads((root / "latest.json").read_text(encoding="utf-8"))[
                    "file_sha256"
                ],
            )

            second_path = store.write(build_report("run-2"))
            second = ProductionQualificationReport.model_validate_json(
                second_path.read_text(encoding="utf-8")
            )
            self.assertEqual(second.previous_report_sha256, first.report_sha256)
            self.assertNotEqual(first_path, second_path)
            with self.assertRaises(FileExistsError):
                store.write(build_report("run-2"))

    def test_tampered_latest_report_blocks_next_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ImmutableQualificationStore(root, signing_key=None)
            report_path = store.write(build_report("run-1"))
            report_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                store.write(build_report("run-2"))

    def test_stale_and_changed_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            path.write_text('{"status":"passed"}', encoding="utf-8")
            old = time.time() - 7200
            os.utime(path, (old, old))
            with self.assertRaisesRegex(ValueError, "stale"):
                bind_evidence(
                    path,
                    evidence_id="stale",
                    maximum_age_seconds=60,
                )

            os.utime(path, None)
            binding = bind_evidence(
                path,
                evidence_id="current",
                maximum_age_seconds=60,
            )
            path.write_text('{"status":"failed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_json_evidence(binding)

    def test_training_proof_requires_failure_and_both_soak_durations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.json"
            proofs = [
                {
                    "name": "postgres_redis_minio_lifecycle",
                    "status": "passed",
                    "evidence": {
                        "redis_event_count": 1,
                        "duplicate_event_rejected": True,
                        "database": {
                            "job_rows": 1,
                            "attempt_rows": 1,
                            "ingress_rows": 1,
                        },
                    },
                },
                {
                    "name": "ordered_event_stress_1000000",
                    "status": "passed",
                    "evidence": {
                        "event_count": 1_000_000,
                        "final_sequence": 1_000_000,
                        "tail_sequence": 1_000_000,
                    },
                },
                {
                    "name": "docker_disaster_recovery_drill",
                    "status": "passed",
                    "evidence": {
                        "postgres_dump_restore": True,
                        "postgres_restart": True,
                        "redis_aof_restart": True,
                        "minio_volume_restart": True,
                        "qdrant_volume_restart": True,
                    },
                },
                {
                    "name": "worker_loss_retry_recovery",
                    "status": "passed",
                    "evidence": {
                        "status_after_loss": "queued",
                        "retry_failure_class": "node_loss",
                        "attempt_count": 1,
                    },
                },
                {
                    "name": "infrastructure_soak_86400s",
                    "status": "passed",
                    "duration_seconds": 86400,
                    "evidence": {
                        "requested_seconds": 86400,
                        "successful_transactions": 2880,
                    },
                },
                {
                    "name": "infrastructure_soak_604800s",
                    "status": "passed",
                    "duration_seconds": 604800,
                    "evidence": {
                        "requested_seconds": 604800,
                        "successful_transactions": 20160,
                    },
                },
            ]
            for proof in proofs:
                proof.setdefault("started_at", utc_now())
                proof.setdefault("duration_seconds", 1.0)
                proof.setdefault("blockers", [])
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": utc_now(),
                        "environment": {},
                        "proofs": proofs,
                        "passed": len(proofs),
                        "failed": 0,
                        "blocked": 0,
                        "status": "passed",
                    }
                ),
                encoding="utf-8",
            )
            failure, soak = validate_training_proofs(
                str(path),
                policy=build_policy(),
            )
            self.assertEqual(failure.status, "passed")
            self.assertEqual(soak.status, "passed")

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["proofs"][-1]["status"] = "failed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, failed_soak = validate_training_proofs(
                str(path),
                policy=build_policy(),
            )
            self.assertEqual(failed_soak.status, "failed")

    def test_security_review_requires_independence_domains_and_no_open_highs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "security.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_id": "manual-1",
                        "reviewed_at": utc_now(),
                        "reviewers": ["reviewer-a", "reviewer-b"],
                        "domains": {
                            domain: "passed"
                            for domain in build_policy().required_security_domains
                        },
                        "critical_findings": 0,
                        "high_findings": 0,
                        "unresolved_findings": [],
                        "scanner_report_sha256": "c" * 64,
                        "decision": "approved",
                    }
                ),
                encoding="utf-8",
            )
            result = validate_security_review(str(path), policy=build_policy())
            self.assertEqual(result.status, "passed")

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["reviewers"] = ["reviewer-a", "REVIEWER-A"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            failed = validate_security_review(str(path), policy=build_policy())
            self.assertEqual(failed.status, "failed")
            self.assertIn("distinct", failed.blockers[0])

    def test_canary_requires_every_stage_and_rollback_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "canary.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "canary_id": "canary-1",
                        "created_at": utc_now(),
                        "internal_user_count": 3,
                        "status": "passed",
                        "stages": [
                            {
                                "percentage": percentage,
                                "requests": 100,
                                "error_rate": 0.0,
                                "p95_latency_ms": 50,
                                "rollback_trigger_tested": True,
                                "rollback_trigger_error_rate": 0.02,
                                "rollback_succeeded": True,
                                "rollback_completed_seconds": 5,
                            }
                            for percentage in [1, 10, 50, 100]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = validate_canary(str(path), policy=build_policy())
            self.assertEqual(result.status, "passed")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["stages"][2]["rollback_succeeded"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            failed = validate_canary(str(path), policy=build_policy())
            self.assertEqual(failed.status, "failed")
            self.assertTrue(any("50%" in blocker for blocker in failed.blockers))

    @settings(max_examples=100, deadline=None)
    @given(
        concurrency=st.integers(min_value=1, max_value=500),
        multiplier=st.integers(min_value=1, max_value=10),
        streaming_fraction=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_load_stage_property_preserves_bounded_invariants(
        self,
        concurrency: int,
        multiplier: int,
        streaming_fraction: float,
    ) -> None:
        requests = concurrency * multiplier
        streaming = min(requests, int(requests * streaming_fraction))
        stage = LoadStagePolicy(
            name="fuzz-stage",
            concurrency=concurrency,
            requests=requests,
            streaming_requests=streaming,
            maximum_error_rate=0.01,
            maximum_p95_latency_ms=30000,
            minimum_throughput_rps=0.01,
        )
        self.assertLessEqual(stage.streaming_requests, stage.requests)
        self.assertGreaterEqual(stage.requests, stage.concurrency)

    @settings(max_examples=100, deadline=None)
    @given(st.binary(min_size=0, max_size=4096))
    def test_malformed_json_evidence_never_silently_passes(self, payload: bytes) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fuzz.json"
            path.write_bytes(payload)
            binding = bind_evidence(
                path,
                evidence_id="fuzz",
                maximum_age_seconds=60,
            )
            try:
                parsed = load_json_evidence(binding)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            self.assertIsInstance(parsed, (dict, list, str, int, float, bool, type(None)))

    def test_strict_models_reject_unknown_evidence_fields(self) -> None:
        payload = {
            "schema_version": 1,
            "canary_id": "x",
            "created_at": "2026-07-20T00:00:00+00:00",
            "internal_user_count": 1,
            "status": "passed",
            "stages": [],
            "unexpected": "not allowed",
        }
        with self.assertRaises(ValueError):
            CanaryEvidence.model_validate(payload)
        with self.assertRaises(ValueError):
            SecurityReviewEvidence.model_validate({"unexpected": True})

    def test_timed_out_generation_keeps_capacity_until_worker_exits(self) -> None:
        state = object.__new__(NativeServingState)
        state.config = SimpleNamespace(
            max_concurrent_generations=1,
            max_queue_depth=1,
            queue_timeout_seconds=0.01,
            generation_timeout_seconds=0.01,
        )
        state._generation_slots = asyncio.Semaphore(1)
        state._state_lock = asyncio.Lock()
        state._queued = 0
        state._active = 0
        state._completed = 0
        state._failed = 0
        state._timed_out = 0
        active_workers = 0
        maximum_workers = 0
        guard = threading.Lock()

        def slow_generate(
            self: NativeServingState,
            request: ChatCompletionRequest,
        ) -> tuple[str, int, float]:
            nonlocal active_workers, maximum_workers
            with guard:
                active_workers += 1
                maximum_workers = max(maximum_workers, active_workers)
            try:
                time.sleep(0.08)
                return "done", 1, 80.0
            finally:
                with guard:
                    active_workers -= 1

        state.generate = MethodType(slow_generate, state)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "safe"}],
            max_tokens=1,
        )

        async def scenario() -> None:
            with self.assertRaises(HTTPException) as first:
                await state.generate_async(request)
            self.assertEqual(first.exception.status_code, 504)
            with self.assertRaises(HTTPException) as second:
                await state.generate_async(request)
            self.assertEqual(second.exception.status_code, 503)
            deadline = asyncio.get_running_loop().time() + 1.0
            while state._active and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            self.assertEqual(state._active, 0)

        asyncio.run(scenario())
        self.assertEqual(maximum_workers, 1)


if __name__ == "__main__":
    unittest.main()
