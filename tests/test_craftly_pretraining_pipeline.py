from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.craftly.learning.quality import DatasetQualityGate, QualityGateConfig
from src.craftly.learning.mixer import ScratchMixConfig, build_scratch_instruction_mix
from src.craftly.evaluation.checkpoint_eval import evaluate_checkpoint
from src.craftly.learning.web_ingest import allowed_url, text_from_html
from src.craftly.model_ops.data_loader import ArtifactCache, TokenShardStream, load_manifest
from src.craftly.model_ops.pretrain_loop import (
    _load_dcp_checkpoint,
    _save_dcp_checkpoint,
    build_cluster_training_plan,
    load_deepspeed_config,
    run_pretraining_loop,
    validate_production_training_args,
)
from src.craftly.model_ops.native_serving import NativeServingConfig, create_app
from src.craftly.model_ops.foundation import sha256_file
from src.craftly.model_ops.production_adapters import build_megatron_launch_plan, build_tensorrt_llm_plan, export_hf_llama_package, validate_vllm_package
from src.craftly.model_ops.checkpoint_compare import GenerationConfig, compare_checkpoints
from src.craftly.model_ops.checkpoint_compare import PromptCase, hallucination_flags_for_output
from src.craftly.model_ops.learning_validation import (
    audit_tokenizer_dominance,
    run_learning_validation,
    write_expanded_eval_suite,
    write_instruction_corpus,
)
from src.craftly.model_ops.tokenizer_pipeline import (
    _load_production_dataset_binding,
    RealCorpusTokenizerConfig,
    ShardBuildConfig,
    ShardManifest,
    TokenizerTrainConfig,
    build_token_shards,
    train_real_corpus_tokenizer,
    train_bpe_tokenizer,
    write_uint32_tokens,
)
from src.craftly.model_ops.torch_decoder import model_profile


class CraftlyPretrainingPipelineTest(unittest.TestCase):
    def test_token_shard_stream_partitions_batches_across_data_parallel_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.bin"
            write_uint32_tokens(shard, list(range(64)))
            rank_zero = list(
                TokenShardStream(
                    [str(shard)],
                    sequence_length=4,
                    batch_size=1,
                    shuffle=False,
                    rank=0,
                    world_size=2,
                ).batches()
            )
            rank_one = list(
                TokenShardStream(
                    [str(shard)],
                    sequence_length=4,
                    batch_size=1,
                    shuffle=False,
                    rank=1,
                    world_size=2,
                ).batches()
            )
            rank_zero_starts = {batch[0][0] for batch in rank_zero}
            rank_one_starts = {batch[0][0] for batch in rank_one}
            self.assertEqual(len(rank_zero), len(rank_one))
            self.assertTrue(rank_zero_starts.isdisjoint(rank_one_starts))
            self.assertEqual(rank_zero_starts | rank_one_starts, set(range(0, 64, 4)))
            resumed = list(
                TokenShardStream(
                    [str(shard)],
                    sequence_length=4,
                    batch_size=1,
                    shuffle=False,
                    rank=0,
                    world_size=2,
                ).batches(start_batch=3)
            )
            self.assertEqual(resumed, rank_zero[3:])

    def test_gradient_accumulation_counts_optimizer_steps_and_persists_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(
                json.dumps({"text": "def verify(value): return value is not None " * 100, "license": "mit"}) + "\n",
                encoding="utf-8",
            )
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=256, sequence_length=16, validation_fraction=0.0),
            )
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=2,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=2,
                target_tokens=64,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=1,
                resume=False,
            )
            self.assertEqual(report["step_semantics"], "optimizer_update_v2")
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["completed_micro_batches_per_rank"], 4)
            self.assertEqual(report["trained_tokens"], 64)
            self.assertEqual(len(report["train_losses"]), 2)
            self.assertEqual(report["dataloader_state"]["completed_micro_batches_per_rank"], 4)

            import torch

            manifest = json.loads(Path(report["checkpoint_manifest"]).read_text(encoding="utf-8"))
            payload = torch.load(Path(manifest["checkpoint_dir"]) / "model.pt", map_location="cpu")
            self.assertEqual(payload["training_args"]["step_semantics"], "optimizer_update_v2")
            self.assertEqual(payload["dataloader_state"]["batch_offset_within_epoch"], 4)

    def test_pretraining_rejects_inconsistent_immutable_token_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(
                json.dumps({"text": "secure code and tests " * 100, "license": "mit"}) + "\n",
                encoding="utf-8",
            )
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            with self.assertRaisesRegex(ValueError, "target_tokens must equal optimizer steps"):
                run_pretraining_loop(
                    output_dir=root / "train",
                    manifest=root / "shards" / "manifest.json",
                    device="cpu",
                    steps=2,
                    batch_size=1,
                    sequence_length=16,
                    gradient_accumulation_steps=2,
                    target_tokens=32,
                    dtype="fp32",
                    validate_every=0,
                    checkpoint_every=0,
                    resume=False,
                )

    def test_production_tokenizer_requires_hash_bound_family_safe_dataset_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            split = root / "split.json"
            promotion = root / "promotion.json"
            train.write_text('{"text":"secure train"}\n', encoding="utf-8")
            val.write_text('{"text":"secure validation"}\n', encoding="utf-8")
            split.write_text('{"cross_split_group_collisions":0}', encoding="utf-8")
            promotion.write_text('{"status":"promoted"}', encoding="utf-8")
            artifacts = {
                "train": str(train),
                "val": str(val),
                "split_manifest": str(split),
                "promotion_decision": str(promotion),
            }
            manifest = root / "dataset_version_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "promoted",
                        "dev_smoke": False,
                        "metrics": {"promoted_records": 100_000},
                        "artifacts": artifacts,
                        "artifact_sha256": {
                            name: sha256_file(path) for name, path in artifacts.items()
                        },
                        "reports": {
                            "split_manifest": {"cross_split_group_collisions": 0}
                        },
                        "advancement_decision_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            config = RealCorpusTokenizerConfig(
                input_paths=[str(train)],
                validation_input_paths=[str(val)],
                output_dir=str(root / "tokenizer"),
                production_mode=True,
                include_stress_samples=False,
                dataset_manifest_path=str(manifest),
            )
            digest, payload = _load_production_dataset_binding(config)
            self.assertEqual(digest, sha256_file(manifest))
            self.assertEqual(payload["status"], "promoted")
            train.write_text('{"text":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact integrity"):
                _load_production_dataset_binding(config)

    def test_artifact_cache_accepts_absolute_file_and_enforces_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "manifest.json"
            source.write_text('{"dataset_id":"test"}', encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            cache = ArtifactCache(root / "cache")
            self.assertEqual(cache.materialize(source, expected_sha256=digest), source.resolve())
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cache.materialize(source, expected_sha256="0" * 64)

    def test_distributed_checkpoint_roundtrip_preserves_training_state(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(4, 4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            checkpoint = Path(temp_dir) / "checkpoint-step-00000001"
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            _save_dcp_checkpoint(
                checkpoint_dir=checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state={
                    "step": 1,
                    "trained_tokens": 64,
                    "config": {"vocab_size": 16, "hidden_size": 4},
                },
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            step, tokens = _load_dcp_checkpoint(
                checkpoint_path=checkpoint / "dcp",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self.assertEqual((step, tokens), (1, 64))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_learning_validation_generates_instruction_corpus_and_tokenizer_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = write_instruction_corpus(root / "instruction.jsonl", count=24, repeats=2)
            suite = write_expanded_eval_suite(root / "suite.jsonl", count=50)
            self.assertEqual(len(corpus.read_text(encoding="utf-8").splitlines()), 48)
            self.assertEqual(len(suite.read_text(encoding="utf-8").splitlines()), 50)
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            audit = audit_tokenizer_dominance(
                tokenizer_path=tokenizer_path,
                corpus_path=corpus,
                output_path=root / "tokenizer_audit.json",
            )
            self.assertGreater(audit.total_tokens, 0)
            self.assertFalse(audit.special_token_missing)
            self.assertLess(audit.dot_fraction, 0.08)
            self.assertTrue(Path(root / "tokenizer_audit.json").exists())

    def test_learning_validation_report_can_skip_expensive_overfit_for_local_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_learning_validation(
                output_dir=Path(temp_dir) / "learning_validation",
                instruction_count=30,
                run_overfit=False,
                device="cpu",
            )
            self.assertEqual(report.status, "passed")
            self.assertTrue(Path(report.instruction_corpus_path).exists())
            self.assertTrue(Path(report.expanded_eval_suite_path).exists())
            self.assertIn("--model-profile t4_validation", report.t4_validation_command)
            self.assertIn("kaggle_1k_validation", report.staged_validation_commands)
            self.assertIn("kaggle_10k_validation", report.staged_validation_commands)

    def test_t4_validation_profile_is_registered_for_non_tiny_gpu_checks(self) -> None:
        profile = model_profile("t4_validation")
        self.assertEqual(profile.hidden_size, 512)
        self.assertEqual(profile.num_layers, 8)
        self.assertEqual(profile.max_sequence_length, 2048)
        self.assertTrue(profile.gradient_checkpointing)

    def test_scratch_instruction_mix_converts_real_rows_and_reports_bucket_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "promoted.jsonl"
            rows = [
                {
                    "text": "CVE-style SQL injection defensive analysis. Use parameterized queries and verify CWE mitigation.",
                    "source": "security-reference",
                    "license": "cc-by-4.0",
                    "category": "cybersecurity",
                    "quality": {"quality_score": 0.92, "labels": ["defensive_security"], "data_type": "security_reference"},
                    "training_gate": {"decision": "train"},
                },
                {
                    "text": "diff --git a/auth.py b/auth.py\n+assert password\npytest test_auth_empty_password regression verification",
                    "source": "verified-patch",
                    "license": "mit",
                    "category": "agentic_coding",
                    "quality": {"quality_score": 0.91, "labels": ["patch", "tests"], "data_type": "patch"},
                    "training_gate": {"decision": "train"},
                },
                {
                    "text": "def safe_parse(value):\n    return value.strip()\nArchitecture notes for clean repository coding.",
                    "source": "docs-code",
                    "license": "apache-2.0",
                    "category": "general_docs_code",
                    "quality": {"quality_score": 0.88, "labels": ["code"], "data_type": "code"},
                    "training_gate": {"decision": "train"},
                },
                {
                    "text": "Traceback (most recent call last): AttributeError: 'NoneType' object has no attribute 'name'. compile error stack trace",
                    "source": "debug-log",
                    "license": "mit",
                    "category": "debugging",
                    "quality": {"quality_score": 0.87, "labels": ["runtime_trace"], "data_type": "debug_trace"},
                    "training_gate": {"decision": "train"},
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = build_scratch_instruction_mix(
                input_paths=[source],
                output_path=root / "scratch_instruction_mix.jsonl",
                report_path=root / "scratch_instruction_mix_report.json",
                config=ScratchMixConfig(min_quality_score=0.6),
            )
            self.assertEqual(report.total_rows, 4)
            self.assertTrue(Path(report.output_jsonl).exists())
            buckets = {item.bucket: item.output_rows for item in report.buckets}
            self.assertEqual(buckets["instruction_security_coding"], 1)
            self.assertEqual(buckets["verified_patch_tests"], 1)
            self.assertEqual(buckets["high_quality_docs_code"], 1)
            self.assertEqual(buckets["debugging_error_logs"], 1)
            mixed_text = Path(report.output_jsonl).read_text(encoding="utf-8")
            self.assertIn("<|thought_start|>", mixed_text)
            self.assertIn("<|patch_start|>", mixed_text)
            self.assertIn("Tests:", mixed_text)
            self.assertIn("Verification:", mixed_text)
            self.assertIn("target_ratios", Path(root / "scratch_instruction_mix_report.json").read_text(encoding="utf-8"))

    def test_defensive_curriculum_filters_offensive_rows_and_keeps_security_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "promoted.jsonl"
            rows = [
                {
                    "text": "CWE-89 SQL injection defensive mitigation. Use parameterized queries and regression tests.",
                    "source": "security-reference",
                    "license": "cc-by-4.0",
                    "category": "cybersecurity",
                    "quality": {"quality_score": 0.92, "labels": ["defensive_security"], "data_type": "security_reference"},
                    "training_gate": {"decision": "train"},
                },
                {
                    "text": "Exploit payload that spawns a reverse shell and bypass edr.",
                    "source": "blocked-offensive",
                    "license": "review-required",
                    "category": "cybersecurity",
                    "quality": {"quality_score": 0.95, "labels": ["defensive_security"], "data_type": "security_reference"},
                    "training_gate": {"decision": "train"},
                },
                {
                    "text": "def helper(value): return value.strip()",
                    "source": "fundamentals",
                    "license": "mit",
                    "category": "general_docs_code",
                    "quality": {"quality_score": 0.9, "labels": ["code"], "data_type": "code"},
                    "training_gate": {"decision": "train"},
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = build_scratch_instruction_mix(
                input_paths=[source],
                output_path=root / "defensive.jsonl",
                report_path=root / "defensive_report.json",
                config=ScratchMixConfig(
                    curriculum_mode="defensive_security_only",
                    min_quality_score=0.6,
                ),
            )
            self.assertEqual(report.curriculum_mode, "defensive_security_only")
            self.assertEqual(report.total_rows, 1)
            self.assertEqual(report.offensive_rejected_rows, 1)
            text = Path(report.output_jsonl).read_text(encoding="utf-8").lower()
            self.assertIn("defensive", text)
            self.assertNotIn("reverse shell", text)

    def test_defensive_eval_suite_adds_hallucination_guard_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = write_expanded_eval_suite(root / "defensive_suite.jsonl", count=100, curriculum_mode="defensive_security_only")
            rows = [json.loads(line) for line in Path(suite).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertTrue(all(row["defensive_only"] for row in rows))
            self.assertTrue(any(row["require_uncertainty_if_evidence_missing"] for row in rows))
            self.assertTrue(all(not row["allow_cve_claims"] for row in rows))

    def test_hallucination_guard_flags_invented_cve_and_unverified_test_claims(self) -> None:
        case = PromptCase(
            task_id="defensive-missing-evidence",
            category="defensive_security",
            prompt="Only note available: security review pending.",
            expected_terms=["cannot", "confirm"],
            require_uncertainty_if_evidence_missing=True,
            allow_cve_claims=False,
            allow_verified_test_claims=False,
            defensive_only=True,
        )
        flags = hallucination_flags_for_output(
            output="This is CVE-2026-12345 and tests passed. Run metasploit with a reverse shell.",
            prompt=case.prompt,
            case=case,
        )
        self.assertIn("missing_required_uncertainty", flags)
        self.assertIn("invented_cve_id", flags)
        self.assertIn("unverified_test_pass_claim", flags)
        self.assertIn("offensive_steps_in_defensive_eval", flags)

    def test_quality_gate_filters_duplicates_and_secret_like_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw.jsonl"
            clean = root / "clean.jsonl"
            good_text = "Secure coding guidance for CWE mitigation. " * 20
            rows = [
                {"text": good_text, "license": "mit"},
                {"text": good_text, "license": "mit"},
                {"text": "api_key = 'abcdefghijklmnopqrstuvwxyz123456'", "license": "mit"},
            ]
            raw.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = DatasetQualityGate(QualityGateConfig(min_chars=20)).filter_jsonl(raw, clean)
            self.assertEqual(report.accepted, 1)
            self.assertEqual(report.duplicate, 1)
            self.assertEqual(len(clean.read_text(encoding="utf-8").splitlines()), 1)

    def test_web_ingest_helpers_are_allowlist_and_html_safe(self) -> None:
        self.assertTrue(allowed_url("https://docs.example.org/a", ["example.org"]))
        self.assertFalse(allowed_url("https://evil.example.net/a", ["example.org"]))
        self.assertEqual(text_from_html("<html><script>x()</script><body>Hello <b>world</b></body></html>"), "Hello world")

    def test_tokenizer_shards_stream_and_pretrain_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            text = "def secure_login(user, password): assert password CWE mitigation patch " * 200
            corpus.write_text(json.dumps({"text": text, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            manifest = build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            loaded = load_manifest(root / "shards" / "manifest.json")
            self.assertTrue(loaded.train_shards)
            self.assertEqual(set(loaded.shard_sha256), set(loaded.train_shards + loaded.val_shards))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                next(
                    TokenShardStream(
                        loaded.train_shards,
                        sequence_length=16,
                        batch_size=1,
                        shuffle=False,
                        expected_sha256={loaded.train_shards[0]: "0" * 64},
                    ).batches()
                )
            batch = next(
                TokenShardStream(
                    loaded.train_shards,
                    sequence_length=16,
                    batch_size=1,
                    expected_sha256=loaded.shard_sha256,
                ).batches()
            )
            self.assertEqual(len(batch[0]), 16)
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=2,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=1,
                resume=False,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["steps"], 2)
            self.assertEqual(report["validate_every"], 0)
            self.assertTrue(Path(report["checkpoint_manifest"]).exists())
            self.assertTrue(Path(report["best_checkpoint_manifest"]).exists())
            self.assertIn("best_validation_loss", report)
            eval_report = evaluate_checkpoint(
                checkpoint_manifest_path=report["best_checkpoint_manifest"],
                training_report=report,
                output_dir=root / "checkpoint_eval",
            )
            self.assertEqual(eval_report.status, "passed")
            self.assertTrue(Path(root / "checkpoint_eval" / "checkpoint_eval_report.json").exists())

    def test_real_corpus_tokenizer_audits_special_tokens_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(
                "\n".join(
                    json.dumps({"text": "def secure_query(value):\n    return validate(value) 0x7ffd00ff <|compile_error|> " * 20})
                    for _ in range(4)
                )
                + "\n",
                encoding="utf-8",
            )
            report = train_real_corpus_tokenizer(
                RealCorpusTokenizerConfig(
                    input_paths=[str(corpus)],
                    output_dir=str(root / "tokenizer_run"),
                    vocab_size=1200,
                    min_frequency=1,
                    shard_token_count=128,
                    sequence_length=16,
                    validation_fraction=0.0,
                    require_exact_vocab_size=False,
                )
            )
            self.assertEqual(report.status, "passed")
            self.assertFalse(report.special_tokens_missing)
            self.assertTrue(Path(report.tokenizer_path).exists())
            self.assertTrue(report.shard_manifest["train_shards"])
            self.assertEqual(report.shard_manifest["schema_version"], 2)
            self.assertEqual(report.shard_manifest["document_boundary_token"], "<|document_end|>")
            self.assertGreaterEqual(report.shard_manifest["boundary_token_count"], 4)
            self.assertEqual(
                report.shard_manifest["boundary_token_count"],
                report.shard_manifest["train_documents"] + report.shard_manifest["val_documents"],
            )

    def test_real_corpus_tokenizer_fails_when_exact_vocab_cannot_be_trained(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "small.jsonl"
            corpus.write_text(
                json.dumps({"text": "def secure(value):\n    return value\n"}) + "\n",
                encoding="utf-8",
            )
            report = train_real_corpus_tokenizer(
                RealCorpusTokenizerConfig(
                    input_paths=[str(corpus)],
                    output_dir=str(root / "tokenizer_run"),
                    vocab_size=64_000,
                    min_frequency=1,
                    shard_token_count=128,
                    sequence_length=16,
                    validation_fraction=0.0,
                    include_stress_samples=False,
                )
            )
            self.assertEqual(report.status, "failed")
            self.assertLess(report.vocab_size_actual, report.vocab_size_requested)
            self.assertTrue(any("vocabulary size mismatch" in failure for failure in report.audit_failures))

    def test_checkpoint_eval_reports_validation_interval_not_reached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            text = "def secure_parser(value): assert value CWE mitigation patch " * 200
            corpus.write_text(json.dumps({"text": text, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.2),
            )
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=25,
                checkpoint_every=0,
                resume=False,
            )
            eval_report = evaluate_checkpoint(
                checkpoint_manifest_path=report["checkpoint_manifest"],
                training_report=report,
                output_dir=root / "checkpoint_eval",
            )
            validation_gate = next(gate for gate in eval_report.gates if gate.name == "validation_loss")
            self.assertEqual(validation_gate.status, "warn")
            self.assertIn("no validation interval", validation_gate.reason)

    def test_pretrain_loop_saves_best_checkpoint_and_can_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(json.dumps({"text": "def secure_parser(value): assert value", "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            train_shard = root / "shards" / "train" / "shard-000000.bin"
            val_shard = root / "shards" / "val" / "shard-000000.bin"
            write_uint32_tokens(train_shard, [0, 1, 2, 3] * 128)
            write_uint32_tokens(val_shard, [0, 1, 2, 3] * 64)
            manifest = ShardManifest(
                dataset_id="unit-test",
                tokenizer_path=str(tokenizer_path),
                output_dir=str(root / "shards"),
                train_shards=[str(train_shard)],
                val_shards=[str(val_shard)],
                train_tokens=512,
                val_tokens=256,
                sequence_length=16,
            )
            (root / "shards" / "manifest.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=20,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=1,
                validation_batches=1,
                early_stopping_patience=1,
                early_stopping_min_delta=10_000.0,
                checkpoint_every=0,
                resume=False,
            )
            self.assertEqual(report["status"], "early_stopped")
            self.assertLess(report["steps"], report["requested_steps"])
            self.assertTrue(Path(report["best_checkpoint_manifest"]).exists())

    def test_pretrain_loop_reports_small_shards_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(json.dumps({"text": "tiny corpus for one shard", "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=64, validation_fraction=0.0),
            )
            with self.assertRaisesRegex(ValueError, "not enough training tokens for one batch"):
                run_pretraining_loop(
                    output_dir=root / "train",
                    manifest=root / "shards" / "manifest.json",
                    device="cpu",
                    steps=1,
                    batch_size=4,
                    sequence_length=64,
                    gradient_accumulation_steps=1,
                    dtype="fp32",
                    validate_every=0,
                    checkpoint_every=0,
                    resume=False,
                )

    def test_pretrain_loop_expands_model_vocab_for_large_tokenizer_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(json.dumps({"text": "small tokenizer asset", "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            shard = root / "shards" / "train" / "shard-000000.bin"
            write_uint32_tokens(shard, [0, 1, 2, 3000] * 16)
            manifest = ShardManifest(
                dataset_id="unit-test",
                tokenizer_path=str(tokenizer_path),
                output_dir=str(root / "shards"),
                train_shards=[str(shard)],
                val_shards=[],
                train_tokens=64,
                val_tokens=0,
                sequence_length=16,
            )
            (root / "shards").mkdir(parents=True, exist_ok=True)
            (root / "shards" / "manifest.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
            )
            self.assertEqual(report["status"], "passed")
            self.assertGreaterEqual(report["model_config"]["vocab_size"], 3001)

    def test_pretrain_loop_accepts_attention_and_checkpointing_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(json.dumps({"text": "def secure(value): return validate(value) " * 80, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            report = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
                attention_impl="eager",
                gradient_checkpointing=True,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["attention_impl"], "eager")
            self.assertTrue(report["model_config"]["gradient_checkpointing"])
            import torch

            manifest_payload = json.loads(Path(report["checkpoint_manifest"]).read_text(encoding="utf-8"))
            checkpoint_payload = torch.load(Path(manifest_payload["checkpoint_dir"]) / "model.pt", map_location="cpu")
            self.assertIn("scheduler", checkpoint_payload)
            self.assertIn("training_args", checkpoint_payload)
            self.assertIn("dataset_manifest_sha256", checkpoint_payload)
            self.assertIn("tokenizer_sha256", checkpoint_payload)
            self.assertIn("git_commit", checkpoint_payload)
            self.assertIn("environment", checkpoint_payload)

    def test_checkpoint_comparison_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            text = "def secure_login(user_input): parameterized query validation test patch " * 300
            corpus.write_text(json.dumps({"text": text, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            training = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
            )
            comparison = compare_checkpoints(
                baseline_manifest=training["checkpoint_manifest"],
                candidate_manifest=training["checkpoint_manifest"],
                tokenizer_path=tokenizer_path,
                output_dir=root / "compare",
                device="cpu",
                generation_config=GenerationConfig(max_new_tokens=8, max_repetition_ratio=1.0),
            )
            self.assertEqual(comparison.status, "neutral")
            self.assertEqual(comparison.baseline.total, 5)
            self.assertTrue(Path(root / "compare" / "checkpoint_comparison_report.json").exists())
            self.assertTrue(Path(root / "compare" / "checkpoint_comparison_report.md").exists())

    def test_checkpoint_comparison_flags_generation_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            corpus.write_text(json.dumps({"text": "'''''''''''''''''''''''''''''''' " * 100, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            training = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
            )
            comparison = compare_checkpoints(
                baseline_manifest=training["checkpoint_manifest"],
                candidate_manifest=training["checkpoint_manifest"],
                tokenizer_path=tokenizer_path,
                output_dir=root / "compare",
                device="cpu",
                generation_config=GenerationConfig(max_new_tokens=16, max_repetition_ratio=0.05),
            )
            self.assertEqual(comparison.status, "failed_generation_collapse")
            self.assertTrue(any(item.collapsed for item in comparison.candidate.results))

    def test_native_serving_loads_scratch_checkpoint_and_returns_chat_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            text = "def validate(value): return value.strip() secure patch test " * 250
            corpus.write_text(json.dumps({"text": text, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            training = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
            )
            app = create_app(
                NativeServingConfig(
                    checkpoint_manifest=training["checkpoint_manifest"],
                    tokenizer_path=str(tokenizer_path),
                    model_name="craftly-test",
                    device="cpu",
                    auth_enabled=False,
                    quota_enabled=False,
                    reject_context_truncation=False,
                    max_prompt_characters=1024,
                )
            )
            client = TestClient(app)
            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200, ready.text)
            self.assertEqual(
                ready.json()["checkpoint_manifest_sha256"],
                sha256_file(Path(training["checkpoint_manifest"])),
            )
            self.assertEqual(
                ready.json()["tokenizer_sha256"],
                sha256_file(Path(tokenizer_path)),
            )
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "craftly-test",
                    "messages": [{"role": "user", "content": "Write a safe validation patch."}],
                    "max_tokens": 2,
                    "temperature": 0.0,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["model"], "craftly-test")
            self.assertIn("choices", payload)
            self.assertTrue(payload["craftly"]["scratch_only"])
            self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f-]{36}$")
            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            self.assertIn("craftly_http_requests_total", metrics.text)
            empty = client.post(
                "/v1/chat/completions",
                json={
                    "model": "craftly-test",
                    "messages": [{"role": "user", "content": ""}],
                },
            )
            self.assertEqual(empty.status_code, 422)
            invalid_role = client.post(
                "/v1/chat/completions",
                json={
                    "model": "craftly-test",
                    "messages": [{"role": "tool", "content": "not allowed"}],
                },
            )
            self.assertEqual(invalid_role.status_code, 422)
            huge = client.post(
                "/v1/chat/completions",
                json={
                    "model": "craftly-test",
                    "messages": [{"role": "user", "content": "界" * 2048}],
                    "max_tokens": 1,
                },
            )
            self.assertEqual(huge.status_code, 413)

    def test_hf_export_and_external_runtime_plans_are_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "clean.jsonl"
            text = "def secure_patch(value): return value.strip() tokenizer export " * 250
            corpus.write_text(json.dumps({"text": text, "license": "mit"}) + "\n", encoding="utf-8")
            tokenizer_path = train_bpe_tokenizer(
                [corpus],
                root / "tokenizer.json",
                TokenizerTrainConfig(vocab_size=1200, min_frequency=1),
            )
            build_token_shards(
                input_paths=[corpus],
                tokenizer_path=tokenizer_path,
                output_dir=root / "shards",
                config=ShardBuildConfig(shard_token_count=128, sequence_length=16, validation_fraction=0.0),
            )
            training = run_pretraining_loop(
                output_dir=root / "train",
                manifest=root / "shards" / "manifest.json",
                device="cpu",
                steps=1,
                batch_size=1,
                sequence_length=16,
                dtype="fp32",
                validate_every=0,
                checkpoint_every=0,
                resume=False,
            )
            hf_report = export_hf_llama_package(
                checkpoint_manifest=training["checkpoint_manifest"],
                tokenizer_path=tokenizer_path,
                output_dir=root / "hf",
            )
            self.assertEqual(hf_report.status, "built_not_runtime_proven")
            self.assertTrue((root / "hf" / "model.safetensors").exists())
            self.assertEqual(json.loads((root / "hf" / "config.json").read_text(encoding="utf-8"))["model_type"], "llama")
            vllm = validate_vllm_package(hf_model_dir=root / "hf")
            self.assertIn(vllm.status, {"blocked_missing_dependency", "production_ready_requires_external_service"})
            trt = build_tensorrt_llm_plan(hf_model_dir=root / "hf", output_dir=root / "trt")
            self.assertTrue((root / "trt" / "tensorrt_llm_plan.json").exists())
            self.assertIn("trtllm-build", trt.command)
            megatron = build_megatron_launch_plan(
                manifest=root / "shards" / "manifest.json",
                tokenizer_path=tokenizer_path,
                output_dir=root / "mega",
                model_profile="tiny",
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=1,
                sequence_length=16,
                micro_batch_size=1,
                global_batch_size=1,
                train_iters=1,
                megatron_root=root / "missing-megatron",
            )
            self.assertEqual(megatron.status, "blocked_missing_dependency")
            sparse_plan = build_megatron_launch_plan(
                manifest=root / "shards" / "manifest.json",
                tokenizer_path=tokenizer_path,
                output_dir=root / "mega-4t",
                model_profile="4t_moe",
                tensor_parallel=8,
                pipeline_parallel=12,
                data_parallel=32,
                context_parallel=8,
                expert_parallel=32,
                sequence_length=32_768,
                micro_batch_size=1,
                global_batch_size=32,
                train_iters=1,
                num_nodes=3072,
                gpus_per_node=8,
                master_addr="controller.internal",
                megatron_root=root / "missing-megatron",
            )
            self.assertEqual(sparse_plan.status, "blocked_missing_dependency")
            self.assertIn("--multi-latent-attention", sparse_plan.command)
            self.assertIn("--moe-router-topk", sparse_plan.command)
            self.assertIn("--expert-model-parallel-size", sparse_plan.command)
            self.assertIn("--qk-layernorm", sparse_plan.command)
            self.assertIn("--moe-layer-freq", sparse_plan.command)
            self.assertIn("([0]*4+[1]*92)", sparse_plan.command)
            self.assertIn("--moe-shared-expert-intermediate-size", sparse_plan.command)
            self.assertIn("--moe-router-score-function", sparse_plan.command)
            self.assertIn("sigmoid", sparse_plan.command)
            self.assertIn("--mtp-num-layers", sparse_plan.command)
            self.assertIn("--mtp-loss-scaling-factor", sparse_plan.command)
            model_flags = [item for item in sparse_plan.command if item.startswith("--")]
            self.assertEqual(len(model_flags), len(set(model_flags)))
            self.assertTrue(any(note.startswith("topology_report=") for note in sparse_plan.notes))
            with self.assertRaisesRegex(ValueError, "must be positive"):
                build_megatron_launch_plan(
                    manifest=root / "shards" / "manifest.json",
                    tokenizer_path=tokenizer_path,
                    output_dir=root / "mega-invalid",
                    model_profile="tiny",
                    tensor_parallel=0,
                    pipeline_parallel=1,
                    data_parallel=1,
                    sequence_length=16,
                    micro_batch_size=1,
                    global_batch_size=1,
                    train_iters=1,
                    megatron_root=root / "missing-megatron",
                )

    def test_cluster_training_plan_validates_manifest_and_batch_math(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard = root / "train.bin"
            write_uint32_tokens(shard, [0, 1, 2, 3] * 64)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            manifest = ShardManifest(
                dataset_id="cluster-plan-test",
                tokenizer_path=str(tokenizer_path),
                output_dir=str(root / "shards"),
                train_shards=[str(shard)],
                val_shards=[],
                train_tokens=256,
                val_tokens=0,
                sequence_length=32,
            )
            (root / "shards").mkdir(parents=True, exist_ok=True)
            manifest_path = root / "shards" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
            plan = build_cluster_training_plan(
                output_dir=root / "cluster-run",
                manifest=manifest_path,
                model_profile_name="7b",
                strategy="fsdp",
                num_nodes=2,
                gpus_per_node=4,
                sequence_length=128,
                batch_size=2,
                gradient_accumulation_steps=8,
                steps=100,
                dtype="bf16",
            )
            self.assertEqual(plan["strategy"], "fsdp")
            self.assertEqual(plan["total_gpus"], 8)
            self.assertEqual(plan["tokens_per_optimizer_step"], 8 * 2 * 8 * 128)
            self.assertIn("torchrun", plan["command"][0])

    def test_deepspeed_config_loader_patches_runtime_batch_fields(self) -> None:
        config = load_deepspeed_config(
            strategy="deepspeed_zero3",
            config_path="deploy/gpu/deepspeed_zero3.json",
            batch_size=2,
            gradient_accumulation_steps=4,
            dtype="fp16",
        )
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 2)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["train_batch_size"], 8)
        self.assertTrue(config["fp16"]["enabled"])
        self.assertFalse(config["bf16"]["enabled"])

    def test_production_training_args_reject_tiny_without_dev_smoke(self) -> None:
        manifest = ShardManifest(
            dataset_id="production-validation",
            tokenizer_path="missing-tokenizer.json",
            output_dir="shards",
            train_shards=["train.bin"],
            val_shards=[],
            train_tokens=1024,
            val_tokens=0,
            sequence_length=128,
        )
        with self.assertRaisesRegex(ValueError, "document-boundary shard schema v2"):
            validate_production_training_args(
                production_mode=True,
                dev_smoke=False,
                model_profile_name="tiny",
                manifest="manifest.json",
                tokenizer_path=None,
                active_manifest=manifest,
                validate_every=10,
                checkpoint_every=10,
                run_steps=100,
            )


if __name__ == "__main__":
    unittest.main()

