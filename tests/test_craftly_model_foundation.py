from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.craftly.gateway import api as gateway_api
from src.craftly.model_ops.backends import list_model_profiles, promote_scratch_checkpoint
from src.craftly.model_ops.foundation import (
    CheckpointManifest,
    PretrainingRunSpec,
    TokenizerContract,
    TrainingDataContract,
    architecture_presets,
    foundation_status,
    sha256_file,
)


class CraftlyModelFoundationTest(unittest.TestCase):
    def test_architecture_presets_are_valid_scratch_specs(self) -> None:
        status = foundation_status()
        self.assertTrue(status["scratch_first"])
        self.assertTrue(status["scratch_only"])
        self.assertFalse(status["external_model_training"])
        presets = architecture_presets()
        self.assertIn("craftly-7b", presets)
        self.assertIn("craftly-62b", presets)
        self.assertIn("craftly-4t-moe", presets)
        for name, spec in presets.items():
            estimate = spec.parameter_report()
            self.assertTrue(spec.name.startswith("craftly-"))
            self.assertGreater(estimate["total"], 0)
            self.assertGreater(estimate["total_billions"], 1)
            self.assertEqual(len(spec.contract_sha256()), 64)
        final = presets["craftly-4t-moe"].parameter_report()
        self.assertTrue(final["total_target_passed"])
        self.assertTrue(final["active_target_passed"])
        self.assertEqual(final["total"], 3_994_840_025_600)
        self.assertEqual(final["active"], 129_520_454_144)
        self.assertEqual(final["kv_cache_elements_per_token_per_layer"], 576)
        self.assertLess(final["kv_cache_compression_ratio"], 0.02)

    def test_pretraining_readiness_requires_real_assets_and_data_gates(self) -> None:
        spec = PretrainingRunSpec(
            architecture=architecture_presets()["craftly-7b"],
            tokenizer=TokenizerContract(tokenizer_path="missing-tokenizer.json"),
            data=TrainingDataContract(manifest_path="missing-dataset.jsonl", token_count_estimate=0),
        )
        report = spec.readiness_report()
        self.assertFalse(report["ready"])
        self.assertTrue(report["missing_assets"])
        self.assertIn("data contamination check is not complete", report["policy_failures"])
        self.assertTrue(report["scratch_training"])
        self.assertTrue(report["scratch_only"])
        self.assertFalse(report["external_model_training"])

    def test_pretraining_readiness_rejects_noncanonical_architecture(self) -> None:
        tampered = architecture_presets()["craftly-7b"].model_copy(
            update={"hidden_size": 8192}
        )
        with self.assertRaisesRegex(ValueError, "immutable canonical"):
            PretrainingRunSpec(
                architecture=tampered,
                tokenizer=TokenizerContract(),
                data=TrainingDataContract(
                    manifest_path="missing-dataset.jsonl",
                    token_count_estimate=0,
                ),
            )

    def test_checkpoint_manifest_hashes_files_and_writes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "ckpt"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("weights", encoding="utf-8")
            manifest = CheckpointManifest.from_directory(
                architecture_name="craftly-7b",
                run_id="run-1",
                step=10,
                trained_tokens=1024,
                checkpoint_dir=checkpoint,
                metrics={"loss": 1.25},
            )
            self.assertEqual(len(manifest.files), 1)
            self.assertEqual(manifest.files[0]["path"], "model.safetensors")
            output = manifest.write_atomic(root / "manifest.json")
            self.assertTrue(output.exists())

    def test_gateway_foundation_status_contract(self) -> None:
        client = TestClient(gateway_api.app)
        response = client.get("/v1/model/foundation/status")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["scratch_first"])
        self.assertIn("craftly-7b", payload["presets"])

    def test_model_profiles_are_scratch_only_or_test_double(self) -> None:
        profiles = list_model_profiles()
        self.assertEqual(set(profiles), {"mock", "craftly-scratch-local"})
        self.assertEqual(profiles["craftly-scratch-local"]["backend"], "craftly_serving")
        forbidden = " ".join(str(value) for profile in profiles.values() for value in profile.values()).lower()
        for term in ["qw" + "en", "deep" + "seek", "lla" + "ma", "openai" + "_compatible"]:
            self.assertNotIn(term, forbidden)

    def _promotion_evidence(self, root: Path) -> tuple[Path, Path, Path, Path]:
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.pt").write_bytes(b"trusted-scratch-checkpoint")
        (checkpoint / "config.json").write_text('{"name":"craftly-test"}', encoding="utf-8")
        manifest = CheckpointManifest.from_directory(
            architecture_name="craftly-test",
            run_id="run-promotion",
            step=100,
            trained_tokens=4096,
            checkpoint_dir=checkpoint,
            metrics={"validation_loss": 1.5},
        )
        manifest_path = manifest.write_atomic(root / "checkpoint_manifest.json").resolve()
        tokenizer_path = root / "tokenizer.json"
        tokenizer_path.write_text('{"version":"1.0"}', encoding="utf-8")
        evaluation_path = root / "benchmark_suites_report.json"
        evaluation_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "evaluation_mode": "executable_model",
                    "aggregate_score": 0.25,
                    "suites": [
                        {
                            "name": "humaneval",
                            "kind": "human_eval_style",
                            "status": "passed",
                            "score": 0.25,
                            "total": 1,
                            "passed": 1,
                            "pass_at_k": {"pass@1": 0.25},
                            "report": {
                                "checkpoint_manifest": str(manifest_path),
                                "tokenizer_path": str(tokenizer_path.resolve()),
                                "tasks": [
                                    {
                                        "task_id": "task-1",
                                        "candidate_count": 1,
                                        "passed_candidates": 1,
                                        "pass_at_k": {"pass@1": 1.0},
                                        "candidates": [
                                            {
                                                "candidate_index": 0,
                                                "passed": True,
                                                "status": "ok",
                                                "exit_code": 0,
                                                "duration_ms": 1.0,
                                                "generated_tokens": 10,
                                                "output_sha256": "a" * 64,
                                                "failure": "",
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        scorecard_path = root / "scorecard.json"
        score_task = {
            "task_id": "task",
            "category": "coding",
            "accepted": True,
            "applied": True,
            "source_immutable": True,
            "expected_files_changed": True,
            "content_assertions_passed": True,
            "tests_passed": True,
            "security_passed": True,
            "confidence": 0.95,
            "attempts": 1,
            "duration_ms": 1.0,
            "score": 1.0,
            "errors": [],
        }
        scorecard_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "policy_mode": "strict",
                    "task_count": 50,
                    "architecture_reliability_score": 1.0,
                    "workflow_completion_score": 1.0,
                    "security_detection_fix_score": 1.0,
                    "short_prompt_understanding_score": 1.0,
                    "sandbox_test_pass_rate": 1.0,
                    "regression_count": 0,
                    "average_confidence": 0.95,
                    "average_score": 1.0,
                    "model_backend": "craftly_serving",
                    "model_evidence": {
                        "checkpoint_manifest_sha256": sha256_file(manifest_path),
                        "tokenizer_sha256": sha256_file(tokenizer_path),
                        "evaluation_report_sha256": sha256_file(evaluation_path),
                        "active_profile_sha256": "f" * 64,
                        "serving_identity_sha256": "e" * 64,
                    },
                    "tasks": [{**score_task, "task_id": f"task-{index}"} for index in range(50)],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, tokenizer_path, evaluation_path, scorecard_path

    def test_checkpoint_promotion_is_evidence_bound_and_profile_is_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, tokenizer, evaluation, _scorecard = self._promotion_evidence(root)
            output = root / "active-profile.json"
            contract = promote_scratch_checkpoint(
                checkpoint_manifest=manifest,
                tokenizer_path=tokenizer,
                evaluation_report=evaluation,
                output_path=output,
                endpoint="http://127.0.0.1:8010/v1",
                promotion_mode="validation",
            )
            self.assertEqual(contract.profile.backend, "craftly_serving")
            self.assertTrue(contract.profile.scratch_only)
            self.assertTrue(contract.production_blockers)
            with patch.dict(os.environ, {"CRAFTLY_ACTIVE_MODEL_PROFILE_PATH": str(output)}):
                from src.craftly.shared.config import load_active_profile

                loaded = load_active_profile()
            self.assertEqual(loaded["profile"]["checkpoint_manifest"], str(manifest))
            with self.assertRaises(FileExistsError):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=evaluation,
                    output_path=output,
                    endpoint="http://127.0.0.1:8010/v1",
                )

    def test_production_promotion_requires_scorecard_and_secure_remote_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, tokenizer, evaluation, scorecard = self._promotion_evidence(root)
            with self.assertRaisesRegex(ValueError, "scorecard"):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=evaluation,
                    output_path=root / "missing-scorecard.json",
                    endpoint="https://serving.example/v1",
                    promotion_mode="production",
                )
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=evaluation,
                    scorecard_report=scorecard,
                    output_path=root / "insecure.json",
                    endpoint="http://serving.example/v1",
                    promotion_mode="production",
                )
            contract = promote_scratch_checkpoint(
                checkpoint_manifest=manifest,
                tokenizer_path=tokenizer,
                evaluation_report=evaluation,
                scorecard_report=scorecard,
                output_path=root / "production.json",
                endpoint="https://serving.example/v1",
                promotion_mode="production",
            )
            self.assertEqual(contract.production_blockers, [])

    def test_production_promotion_rejects_scorecard_from_another_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, tokenizer, evaluation, scorecard = self._promotion_evidence(root)
            payload = json.loads(scorecard.read_text(encoding="utf-8"))
            payload["model_evidence"]["checkpoint_manifest_sha256"] = "0" * 64
            scorecard.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scorecard is not bound"):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=evaluation,
                    scorecard_report=scorecard,
                    output_path=root / "mismatched-scorecard.json",
                    endpoint="https://serving.example/v1",
                    promotion_mode="production",
                )

    def test_checkpoint_promotion_rejects_static_evaluation_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, tokenizer, evaluation, _scorecard = self._promotion_evidence(root)
            payload = json.loads(evaluation.read_text(encoding="utf-8"))
            payload["evaluation_mode"] = "dataset_validation"
            evaluation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable_model"):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=evaluation,
                    output_path=root / "static.json",
                    endpoint="http://127.0.0.1:8010/v1",
                )
            checkpoint_payload = json.loads(manifest.read_text(encoding="utf-8"))
            checkpoint_path = Path(checkpoint_payload["checkpoint_dir"]) / "model.pt"
            checkpoint_path.write_bytes(b"tampered")
            valid_evaluation = root / "valid-eval.json"
            valid_evaluation.write_text(
                evaluation.read_text(encoding="utf-8").replace("dataset_validation", "executable_model"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                promote_scratch_checkpoint(
                    checkpoint_manifest=manifest,
                    tokenizer_path=tokenizer,
                    evaluation_report=valid_evaluation,
                    output_path=root / "tampered.json",
                    endpoint="http://127.0.0.1:8010/v1",
                )


if __name__ == "__main__":
    unittest.main()

