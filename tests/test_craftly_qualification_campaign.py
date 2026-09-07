from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.craftly.evaluation.qualification_campaign import (
    CATEGORIES,
    QualificationBaselineReport,
    QualificationGatePolicy,
    build_repository_qualification_pack,
    decide_stage,
    import_secrepobench_evidence,
    run_checkpoint_baseline,
    validate_defensive_dataset_binding,
    write_approval_template,
    write_campaign_plan,
)
from src.craftly.evaluation.benchmark_suites import (
    BenchmarkSuiteResult,
    BenchmarkSuitesReport,
)
from src.craftly.evaluation.benchmark_pack import materialize_protected_benchmark_pack
from src.craftly.model_ops.checkpoint_compare import (
    CandidateResult,
    CheckpointComparisonReport,
    CheckpointSideReport,
)
from src.craftly.model_ops.foundation import sha256_file
from src.craftly.model_ops.pretrain_loop import run_pretraining_loop
from src.craftly.model_ops.tokenizer_pipeline import (
    ShardBuildConfig,
    TokenizerTrainConfig,
    build_token_shards,
    train_bpe_tokenizer,
)


class CraftlyQualificationCampaignTest(unittest.TestCase):
    def _benchmark_fixture(self, root: Path) -> tuple[Path, Path, str]:
        source = root / "SecRepoBench"
        source.mkdir()
        metadata = {}
        reports = {}
        repositories = []
        crash_types = [
            "Heap-buffer-overflow READ 1",
            "Heap-buffer-overflow WRITE 4",
            "Heap-use-after-free READ 8",
            "Use-of-uninitialized-value",
            "Negative-size-param",
        ]
        for index in range(60):
            task_id = str(1_000 + index)
            project = f"project-{index % 6}"
            metadata[task_id] = {
                "project_name": project,
                "fixing_commit": f"{index + 1:040x}",
                "changed_file": f"src/file_{index}.c",
                "changed_function": [f"parse_{index}"],
                "diff": {"added": [[10, "if (length > capacity) return ERROR;"]], "deleted": []},
                "source_code_before": (
                    f"int parse_{index}(const unsigned char *data, size_t length) {{\n"
                    "    unsigned char local[16];\n"
                    "    memcpy(local, data, length);\n"
                    "    return local[0];\n"
                    "}\n"
                ),
                "source_code": "sealed fixed source",
                "crash_type": crash_types[index % len(crash_types)],
            }
            reports[task_id] = {
                "testcase_vul": "crash",
                "testcase_sec": "pass",
                "unittest_sec": {"pass": ["unit-a", "unit-b"], "fail": []},
            }
        for index in range(6):
            repositories.append(
                {
                    "project": f"project-{index}",
                    "repo_addr": f"https://github.com/example/project-{index}.git",
                }
            )
        with gzip.open(source / "sample_metadata.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        with gzip.open(source / "report.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(reports, handle)
        (source / "github_repos.json").write_text(json.dumps(repositories), encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "tests@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Craftly Tests"], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        registry = root / "sources.json"
        registry.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sources": [
                        {
                            "source_id": "fixture-security-benchmark",
                            "adapter": "secrepobench",
                            "repository_url": "https://github.com/example/security-benchmark.git",
                            "pinned_commit": commit,
                            "evaluation_only": True,
                            "approval_required": True,
                            "license_review_status": "required_before_use",
                            "required_files": [
                                "sample_metadata.json.gz",
                                "report.json.gz",
                                "github_repos.json",
                            ],
                            "allowed_repository_hosts": ["github.com"],
                            "notes": "test-only historical repository task metadata",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return source, registry, commit

    def _approved_record(self, source: Path, registry: Path, root: Path) -> Path:
        approval = root / "approval.json"
        write_approval_template(
            source_root=source,
            source_id="fixture-security-benchmark",
            registry_path=registry,
            output_path=approval,
        )
        payload = json.loads(approval.read_text(encoding="utf-8"))
        payload.update(
            {
                "decision": "approved",
                "approved_by": "Authorized Test Reviewer",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "rationale": "Approved for isolated automated evaluation tests only; redistribution is prohibited.",
            }
        )
        approval.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return approval

    def test_pack_requires_approval_and_builds_exact_real_task_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, registry, _commit = self._benchmark_fixture(root)
            approval = root / "approval.json"
            write_approval_template(
                source_root=source,
                source_id="fixture-security-benchmark",
                registry_path=registry,
                output_path=approval,
            )
            with self.assertRaisesRegex(PermissionError, "not approved"):
                build_repository_qualification_pack(
                    source_root=source,
                    approval_path=approval,
                    output_dir=root / "pack-rejected",
                    source_id="fixture-security-benchmark",
                    registry_path=registry,
                )

            approved = self._approved_record(source, registry, root)
            manifest = build_repository_qualification_pack(
                source_root=source,
                approval_path=approved,
                output_dir=root / "pack",
                source_id="fixture-security-benchmark",
                registry_path=registry,
            )
            self.assertEqual(manifest.task_count, 50)
            self.assertEqual(manifest.category_counts, {category: 10 for category in CATEGORIES})
            self.assertEqual(len(set(manifest.task_ids)), 50)
            self.assertFalse(manifest.ground_truth_in_prompts)
            prompt_text = Path(manifest.prompt_suite_path).read_text(encoding="utf-8")
            self.assertNotIn("sealed fixed source", prompt_text)
            self.assertNotIn('"diff"', prompt_text)
            self.assertEqual(sha256_file(Path(manifest.prompt_suite_path)), manifest.prompt_suite_sha256)

    def test_pack_rejects_source_changed_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, registry, _commit = self._benchmark_fixture(root)
            approval = self._approved_record(source, registry, root)
            with gzip.open(source / "sample_metadata.json.gz", "at", encoding="utf-8") as handle:
                handle.write("tamper")
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                build_repository_qualification_pack(
                    source_root=source,
                    approval_path=approval,
                    output_dir=root / "pack",
                    source_id="fixture-security-benchmark",
                    registry_path=registry,
                )

    def test_baseline_reports_prompt_metrics_without_inventing_repository_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, registry, _commit = self._benchmark_fixture(root)
            approval = self._approved_record(source, registry, root)
            pack = build_repository_qualification_pack(
                source_root=source,
                approval_path=approval,
                output_dir=root / "pack",
                source_id="fixture-security-benchmark",
                registry_path=registry,
            )
            checkpoint = root / "checkpoint_manifest.json"
            tokenizer = root / "tokenizer.json"
            checkpoint.write_text("{}\n", encoding="utf-8")
            tokenizer.write_text("{}\n", encoding="utf-8")
            prompt_report = self._side_report("baseline", total=50, pass_count=12, score=0.4)
            with patch(
                "src.craftly.evaluation.qualification_campaign.evaluate_checkpoint_prompt_suite",
                return_value=prompt_report,
            ):
                report: QualificationBaselineReport = run_checkpoint_baseline(
                    pack_manifest_path=root / "pack" / "qualification_pack_manifest.json",
                    checkpoint_manifest=checkpoint,
                    tokenizer_path=tokenizer,
                    output_dir=root / "baseline",
                    device="cpu",
                )
            self.assertEqual(report.status, "measured_prompt_only")
            self.assertEqual(report.solved_tasks, 12)
            self.assertEqual(report.repository_evidence.status, "blocked_missing_evidence")
            self.assertIsNone(report.repository_evidence.test_pass_rate)
            self.assertIsNone(report.repository_evidence.security_pass_rate)

    def test_official_evidence_import_requires_all_exact_pack_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, registry, _commit = self._benchmark_fixture(root)
            approval = self._approved_record(source, registry, root)
            pack = build_repository_qualification_pack(
                source_root=source,
                approval_path=approval,
                output_dir=root / "pack",
                source_id="fixture-security-benchmark",
                registry_path=registry,
            )
            catalog = [
                json.loads(line)
                for line in Path(pack.task_catalog_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            evaluation = {}
            for task in catalog:
                source_task_id = task["provenance"]["source_task_id"]
                evaluation[source_task_id] = {
                    "agent": {
                        "model": {
                            "context": {
                                "prompt": {
                                    "mode": {
                                        "testcase": "pass",
                                        "unittest": {"pass": ["unit-a", "unit-b"], "fail": []},
                                    }
                                }
                            }
                        }
                    }
                }
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            evidence_path = import_secrepobench_evidence(
                pack_manifest_path=root / "pack" / "qualification_pack_manifest.json",
                benchmark_source_root=source,
                evaluation_report_path=evaluation_path,
                agent_key="agent",
                model_key="model",
                context_key="context",
                prompt_key="prompt",
                mode_key="mode",
                output_path=root / "repository_evidence.json",
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["task_count"], 50)
            self.assertEqual(evidence["sandbox_test_pass_rate"], 1.0)
            self.assertEqual(evidence["security_detection_fix_score"], 1.0)
            del evaluation[next(iter(evaluation))]
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 1 qualification tasks"):
                import_secrepobench_evidence(
                    pack_manifest_path=root / "pack" / "qualification_pack_manifest.json",
                    benchmark_source_root=source,
                    evaluation_report_path=evaluation_path,
                    agent_key="agent",
                    model_key="model",
                    context_key="context",
                    prompt_key="prompt",
                    mode_key="mode",
                    output_path=root / "repository_evidence-incomplete.json",
                )

    def test_stage_gate_requires_improvement_from_10k(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = root / "best.json"
            selected.write_text("{}\n", encoding="utf-8")
            training = {
                "status": "passed",
                "steps": 1_000,
                "checkpoint_reload_verified": True,
                "train_losses": [7.0, 6.0],
                "best_validation_loss": 5.9,
                "best_checkpoint_manifest": str(selected),
            }
            neutral = self._comparison(root, delta=0.0)
            stage_1k = decide_stage(
                target_curriculum_steps=1_000,
                global_target_steps=1_000,
                policy=QualificationGatePolicy(),
                training_report=training,
                checkpoint_eval_status="passed",
                tokenizer_audit_status="passed",
                comparison=neutral,
                training_report_path=root / "train.json",
                checkpoint_eval_path=root / "eval.json",
                checkpoint_comparison_path=root / "compare.json",
                tokenizer_audit_path=root / "audit.json",
            )
            self.assertTrue(stage_1k.promotion_allowed)
            training["steps"] = 10_000
            stage_10k = decide_stage(
                target_curriculum_steps=10_000,
                global_target_steps=10_000,
                policy=QualificationGatePolicy(),
                training_report=training,
                checkpoint_eval_status="passed",
                tokenizer_audit_status="passed",
                comparison=neutral,
                training_report_path=root / "train.json",
                checkpoint_eval_path=root / "eval.json",
                checkpoint_comparison_path=root / "compare.json",
                tokenizer_audit_path=root / "audit.json",
            )
            self.assertFalse(stage_10k.promotion_allowed)
            self.assertIn("minimum_score_improvement_not_met", stage_10k.gate_failures)
            self.assertIn("executable_benchmark_missing", stage_10k.gate_failures)

    def test_10k_stage_requires_passing_executable_humaneval_and_mbpp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = root / "best.json"
            selected.write_text("{}\n", encoding="utf-8")
            training = {
                "status": "passed",
                "steps": 10_000,
                "checkpoint_reload_verified": True,
                "train_losses": [7.0, 5.0],
                "best_validation_loss": 4.9,
                "best_checkpoint_manifest": str(selected),
            }
            benchmark_path = root / "benchmark.json"
            report = BenchmarkSuitesReport(
                status="passed",
                evaluation_mode="executable_model",
                suites=[
                    BenchmarkSuiteResult(
                        name="humaneval",
                        kind="human_eval_style",
                        status="passed",
                        score=0.10,
                        total=25,
                        passed=2,
                        pass_at_k={"pass@1": 0.08, "pass@5": 0.20},
                    ),
                    BenchmarkSuiteResult(
                        name="mbpp",
                        kind="mbpp_style",
                        status="passed",
                        score=0.04,
                        total=25,
                        passed=1,
                        pass_at_k={"pass@1": 0.04, "pass@5": 0.12},
                    ),
                ],
                aggregate_score=0.07,
            )
            benchmark_path.write_text(report.model_dump_json(), encoding="utf-8")
            record = decide_stage(
                target_curriculum_steps=10_000,
                global_target_steps=10_000,
                policy=QualificationGatePolicy(),
                training_report=training,
                checkpoint_eval_status="passed",
                tokenizer_audit_status="passed",
                comparison=self._comparison(root, delta=0.02),
                training_report_path=root / "train.json",
                checkpoint_eval_path=root / "eval.json",
                checkpoint_comparison_path=root / "compare.json",
                tokenizer_audit_path=root / "audit.json",
                executable_benchmark=report,
                executable_benchmark_path=benchmark_path,
            )
            self.assertTrue(record.promotion_allowed)
            self.assertEqual(record.executable_pass_at_1, 0.04)
            self.assertEqual(record.executable_benchmark_sha256, sha256_file(benchmark_path))

            failed_report = report.model_copy(
                update={
                    "status": "failed",
                    "suites": [
                        report.suites[0],
                        report.suites[1].model_copy(
                            update={
                                "status": "failed",
                                "passed": 0,
                                "pass_at_k": {"pass@1": 0.0, "pass@5": 0.0},
                            }
                        ),
                    ],
                }
            )
            failed = decide_stage(
                target_curriculum_steps=10_000,
                global_target_steps=10_000,
                policy=QualificationGatePolicy(),
                training_report=training,
                checkpoint_eval_status="passed",
                tokenizer_audit_status="passed",
                comparison=self._comparison(root, delta=0.02),
                training_report_path=root / "train.json",
                checkpoint_eval_path=root / "eval.json",
                checkpoint_comparison_path=root / "compare.json",
                tokenizer_audit_path=root / "audit.json",
                executable_benchmark=failed_report,
                executable_benchmark_path=benchmark_path,
            )
            self.assertFalse(failed.promotion_allowed)
            self.assertIn("executable_benchmark_failed", failed.gate_failures)

    def test_defensive_dataset_binding_rejects_wrong_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard = {
                "dataset_id": "defensive",
                "tokenizer_path": "tokenizer.json",
                "train_shards": ["train.bin"],
                "val_shards": ["val.bin"],
                "train_tokens": 1000,
                "val_tokens": 100,
                "sequence_length": 128,
                "shard_sha256": {"train.bin": "a" * 64, "val.bin": "b" * 64},
            }
            shard_path = root / "shards.json"
            version_path = root / "version.json"
            shard_path.write_text(json.dumps(shard), encoding="utf-8")
            version = {
                "shard_manifest": shard,
                "instruction_mix_report": {
                    "status": "passed",
                    "curriculum_mode": "defensive_security_only",
                    "strict_offensive_filter": True,
                    "total_rows": 100,
                    "total_tokens": 1000,
                },
                "training_data_gate_report": {"promoted": 100},
                "benchmark_contamination_filter_report": {"accepted": 100},
                "near_dedup_report": {"accepted": 100},
            }
            version_path.write_text(json.dumps(version), encoding="utf-8")
            validate_defensive_dataset_binding(
                shard_manifest_path=shard_path,
                version_manifest_path=version_path,
            )
            version["instruction_mix_report"]["curriculum_mode"] = "agentic_coding_only"
            version_path.write_text(json.dumps(version), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not defensive_security_only"):
                validate_defensive_dataset_binding(
                    shard_manifest_path=shard_path,
                    version_manifest_path=version_path,
                )

    def test_campaign_plan_is_exact_1k_to_100k_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected_config = root / "protected.json"
            protected_config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "pack_id": "qualification-plan-test-pack",
                        "sources": [
                            {
                                "name": "humaneval",
                                "kind": "builtin_craftly",
                                "revision": "test-contract-v1",
                                "license": "Proprietary-Evaluation",
                                "output_file": "humaneval.jsonl",
                                "minimum_rows": 25,
                                "train_policy": "eval_holdout",
                            },
                            {
                                "name": "mbpp",
                                "kind": "builtin_craftly",
                                "revision": "test-contract-v1",
                                "license": "Proprietary-Evaluation",
                                "output_file": "mbpp.jsonl",
                                "minimum_rows": 25,
                                "train_policy": "eval_holdout",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            protected_root = root / "protected"
            materialize_protected_benchmark_pack(protected_config, protected_root)
            campaign_payload = json.loads(
                Path("config/defensive_checkpoint_qualification.json").read_text(
                    encoding="utf-8"
                )
            )
            campaign_payload["executable_evaluation"]["protected_config"] = str(
                protected_config
            )
            campaign_payload["executable_evaluation"]["protected_manifest"] = str(
                protected_root / "protected_benchmark_manifest.json"
            )
            for suite in campaign_payload["executable_evaluation"]["suites"]:
                suite["path"] = str(protected_root / f"{suite['name']}.jsonl")
            campaign_config = root / "campaign.json"
            campaign_config.write_text(
                json.dumps(campaign_payload),
                encoding="utf-8",
            )
            path = write_campaign_plan(
                config_path=campaign_config,
                output_dir=root / "plan",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["target_curriculum_steps"] for item in payload["stages"]],
                [1_000, 10_000, 20_000, 50_000, 100_000],
            )
            self.assertTrue(payload["scratch_only"])
            self.assertFalse(payload["stages"][0]["requires_executable_benchmark"])
            self.assertTrue(payload["stages"][1]["requires_executable_benchmark"])
            self.assertEqual(
                payload["executable_evaluation"]["protected_manifest_sha256"],
                sha256_file(protected_root / "protected_benchmark_manifest.json"),
            )
            self.assertEqual(
                {suite["name"] for suite in payload["executable_evaluation"]["suites"]},
                {"humaneval", "mbpp"},
            )

    def test_external_checkpoint_continuation_is_hash_verified_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"text": "validate bounds defensive patch regression test " * 400}) + "\n",
                encoding="utf-8",
            )
            tokenizer = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer,
                output_dir=root / "shards",
                config=ShardBuildConfig(
                    shard_token_count=256,
                    sequence_length=16,
                    validation_fraction=0.2,
                ),
            )
            initial = run_pretraining_loop(
                output_dir=root / "initial",
                manifest=root / "shards" / "manifest.json",
                tokenizer_path=tokenizer,
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                dtype="fp32",
                validate_every=1,
                validation_batches=1,
                checkpoint_every=1,
                resume=False,
            )
            continued = run_pretraining_loop(
                output_dir=root / "continued",
                manifest=root / "shards" / "manifest.json",
                tokenizer_path=tokenizer,
                device="cpu",
                steps=2,
                batch_size=1,
                sequence_length=16,
                dtype="fp32",
                validate_every=1,
                validation_batches=1,
                checkpoint_every=1,
                resume=True,
                initial_checkpoint_manifest=initial["checkpoint_manifest"],
            )
            self.assertEqual(continued["start_step"], 1)
            self.assertEqual(continued["steps"], 2)
            self.assertEqual(continued["initial_checkpoint_manifest"], initial["checkpoint_manifest"])
            self.assertTrue(continued["checkpoint_reload_verified"])

    @staticmethod
    def _candidate(task_id: str, score: float) -> CandidateResult:
        return CandidateResult(
            task_id=task_id,
            category="defensive_security",
            prompt="Review this defensive security code.",
            output="Validate bounds, implement the fix, and run regression tests.",
            score=score,
            expected_hits=["validate", "test"],
            repetition_ratio=0.1,
            token_count=12,
            latency_ms=1.0,
        )

    @classmethod
    def _side_report(
        cls,
        label: str,
        *,
        total: int,
        pass_count: int,
        score: float,
    ) -> CheckpointSideReport:
        results = [cls._candidate(f"task-{index}", score) for index in range(total)]
        return CheckpointSideReport(
            label=label,
            checkpoint_manifest=f"{label}.json",
            checkpoint_step=1,
            trained_tokens=100,
            average_score=score,
            pass_count=pass_count,
            total=total,
            pass_rate=pass_count / total,
            hallucination_count=0,
            hallucination_rate=0.0,
            collapsed_count=0,
            collapse_rate=0.0,
            failure_categories={},
            results=results,
        )

    @classmethod
    def _comparison(cls, root: Path, *, delta: float) -> CheckpointComparisonReport:
        baseline = cls._side_report("baseline", total=1, pass_count=1, score=0.7)
        candidate = cls._side_report("candidate", total=1, pass_count=1, score=0.7 + delta)
        return CheckpointComparisonReport(
            status="improved" if delta > 0.03 else "neutral",
            tokenizer_path=str(root / "tokenizer.json"),
            device="cpu",
            generation={},
            baseline=baseline,
            candidate=candidate,
            score_delta=delta,
            pass_delta=0,
            improved_tasks=["task-0"] if delta > 0.05 else [],
            regressed_tasks=[],
            unchanged_tasks=[] if delta > 0.05 else ["task-0"],
            recommendation="candidate_improved" if delta > 0.03 else "candidate_neutral",
        )


if __name__ == "__main__":
    unittest.main()
