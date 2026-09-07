from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.craftly.identity.auth import AuthConfig
from src.craftly.learning.storage import LocalObjectStore
from src.craftly.learning.ablation_runner import ModelProgressionDecision
from src.craftly.training_client import _validate_workspace_url, format_event
from src.craftly.model_ops.distributed_worker import normalize_slurm_environment, parse_args
from src.craftly.model_ops.foundation import model_profile
from src.craftly.training_proofs import parse_dr_workload
from src.craftly.training_workspace import (
    ALLOWED_TRANSITIONS,
    InMemoryEventBus,
    InMemoryTrainingStore,
    JobStatus,
    KubernetesPyTorchAdapter,
    KubernetesSchedulerAdapter,
    QualificationCampaignRegistry,
    SlurmSchedulerAdapter,
    TrainingArtifact,
    CheckpointCommitRequest,
    EvaluationCommitRequest,
    TrainingController,
    TrainingEventArchiver,
    TrainingEventBatch,
    TrainingEventInput,
    TrainingJobCreateRequest,
    TrainingJobSpec,
    TrainingProfileRegistry,
    TrainingWorkspaceService,
    build_training_command,
    workspace_readiness,
)


class CraftlyTrainingWorkspaceTest(unittest.TestCase):
    def test_disaster_recovery_workload_parser_rejects_command_shapes(self) -> None:
        self.assertEqual(
            parse_dr_workload("craftly-training:deployment:craftly-training-controller"),
            ("craftly-training", "deployment", "craftly-training-controller"),
        )
        for unsafe in [
            "default:job:runner",
            "default:deployment:controller;whoami",
            "default/deployment/controller",
            "default:deployment:../controller",
        ]:
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                parse_dr_workload(unsafe)

    def test_slurm_environment_normalization_rejects_invalid_rank_topology(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SLURM_PROCID": "8",
                "SLURM_LOCALID": "0",
                "SLURM_NTASKS": "8",
                "MASTER_ADDR": "trainer-0",
                "MASTER_PORT": "29500",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "rank topology"):
                normalize_slurm_environment()

    def test_distributed_worker_rejects_control_characters_in_arguments(self) -> None:
        argv = [
            "distributed_worker",
            "--scheduler",
            "slurm",
            "--nodes",
            "1",
            "--processes-per-node",
            "1",
            "--",
            "--output-dir",
            "safe\ninjected",
        ]
        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            parse_args()

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = InMemoryTrainingStore()
        self.events = InMemoryEventBus()
        self.service = TrainingWorkspaceService(
            store=self.store,
            events=self.events,
            object_store=LocalObjectStore(self.root / "objects"),
            profiles=TrainingProfileRegistry.from_file(),
        )
        progression = ModelProgressionDecision(
            status="authorized",
            selected_tokenizer_vocab_size=64_000,
            selected_architecture="1b",
            target_model_profile="7b",
            target_model_contract_sha256=model_profile("7b").model_copy(
                update={"vocab_size": 64_000}
            ).contract_sha256(),
            tokenizer_promotion_sha256="1" * 64,
            architecture_promotion_sha256="2" * 64,
            scaling_promotion_sha256="3" * 64,
            dataset_manifest_sha256="4" * 64,
            split_manifest_sha256="5" * 64,
            evaluation_manifest_sha256="6" * 64,
            selected_tokenizer_manifest_sha256="7" * 64,
        ).sealed()
        self.progression_path = self.root / "model_progression.json"
        self.progression_path.write_text(
            json.dumps(progression.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        self.progression_sha256 = hashlib.sha256(self.progression_path.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        asyncio.run(self.service.close())
        self.temp_dir.cleanup()

    def request(self, **overrides: object) -> TrainingJobCreateRequest:
        payload = {
            "profile_id": "defensive-1k",
            "idempotency_key": "workspace-test-0001",
            "git_commit": "abcdef1",
            "container_digest": "sha256:" + ("a" * 64),
        }
        payload.update(overrides)
        if payload["profile_id"] == "craftly-7b-fsdp":
            payload.setdefault("model_progression_decision_uri", str(self.progression_path))
            payload.setdefault("model_progression_decision_sha256", self.progression_sha256)
        return TrainingJobCreateRequest.model_validate(payload)

    def test_registry_is_immutable_and_has_scale_profiles(self) -> None:
        registry = TrainingProfileRegistry.from_file()
        self.assertEqual(registry.schema_version, 3)
        self.assertEqual(registry.latest("defensive-1k").scheduler, "notebook")
        target = registry.latest("craftly-60b-hybrid")
        self.assertEqual(target.distributed_strategy, "megatron")
        self.assertTrue(target.resources.rdma_required)
        self.assertEqual(len(target.immutable_hash), 64)
        self.assertEqual(target.optimizer.name, "adamw")
        self.assertEqual(target.learning_rate.schedule, "cosine")
        self.assertEqual(target.token_budget.global_batch_sequences, 256)
        self.assertEqual(target.parallelism.data_parallel, 4)
        final = registry.latest("craftly-4t-moe")
        self.assertEqual(final.model_profile, "4t_moe")
        self.assertEqual(final.token_budget.global_batch_sequences, 2048)
        self.assertGreaterEqual(final.token_budget.target_tokens, 30_000_000_000_000)
        self.assertEqual(final.parallelism.expert_parallel, 16)
        self.assertEqual(final.resources.nodes * final.resources.gpus_per_node, 6144)
        self.assertTrue(final.checkpoint.require_reload_verification)
        self.assertTrue(final.promotion.require_no_regression)
        self.assertTrue(final.scheduler_policy.gang_scheduling)
        self.assertGreater(final.cost_quota.estimated_gpu_hour_cost_usd, 0.0)

    def test_qualification_campaign_has_dense_gated_milestones(self) -> None:
        campaign = QualificationCampaignRegistry.from_file().latest("defensive-staircase-v1")
        milestones = campaign.milestones
        self.assertEqual(len(milestones), 37)
        self.assertEqual([item.steps for item in milestones[:12]], list(range(1000, 13000, 1000)))
        self.assertEqual(milestones[-1].steps, 1_000_000)
        self.assertEqual(milestones[-1].previous_milestone_id, "steps-0900000")

    def test_resolved_policy_recomputes_tokens_and_reaches_runtime_command(self) -> None:
        request = self.request(overrides={"steps": 2000})
        spec = self.service.resolve_spec(request)
        self.assertEqual(spec.schema_version, 3)
        self.assertEqual(len(spec.model_contract_sha256 or ""), 64)
        self.assertEqual(len(spec.model_parameter_report_sha256 or ""), 64)
        self.assertEqual(spec.model_contract["name"], "craftly-t4-validation")
        self.assertEqual(spec.token_budget.target_tokens, 2_048_000)
        command = build_training_command(spec, output_dir="artifacts/craftly/test")
        self.assertIn("--learning-rate-schedule", command)
        self.assertIn("--checkpoint-every", command)
        self.assertIn("--optimizer-beta2", command)

    def test_training_job_rejects_tampered_or_drifted_model_contract(self) -> None:
        spec = self.service.resolve_spec(self.request())
        tampered = spec.model_dump(mode="json", exclude={"spec_hash"})
        tampered["model_contract"]["hidden_size"] = 128
        with self.assertRaisesRegex(ValueError, "contract hash mismatch"):
            TrainingJobSpec.model_validate(tampered)

        drifted = spec.model_copy(
            update={"model_contract_sha256": "0" * 64}
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            build_training_command(drifted, output_dir="artifacts/craftly/test")

    def test_synced_profile_version_is_append_only(self) -> None:
        profiles = self.service.profiles.profiles
        asyncio.run(self.store.sync_profiles(profiles))
        mutated = profiles[0].model_copy(update={"description": "mutated without version bump"})
        with self.assertRaisesRegex(ValueError, "version bump"):
            asyncio.run(self.store.sync_profiles([mutated]))

    def test_job_creation_is_idempotent_and_rejects_spec_reuse(self) -> None:
        first = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        repeated = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        self.assertEqual(first.job_id, repeated.job_id)
        self.assertEqual(first.status, JobStatus.QUEUED)
        with self.assertRaisesRegex(ValueError, "different immutable job spec"):
            asyncio.run(
                self.service.create_job(
                    self.request(overrides={"steps": 1500}),
                    owner_id="researcher-1",
                )
            )

    def test_job_admission_enforces_projected_gpu_cost(self) -> None:
        profiles = list(self.service.profiles.profiles)
        source = self.service.profiles.latest("defensive-1k")
        expensive = source.model_copy(
            update={
                "profile_id": "defensive-cost-blocked",
                "cost_quota": source.cost_quota.model_copy(
                    update={"maximum_cost_usd": 1.0, "estimated_gpu_hour_cost_usd": 100.0}
                ),
            }
        )
        self.service.profiles = TrainingProfileRegistry(schema_version=2, profiles=[*profiles, expensive])
        request = self.request(profile_id=expensive.profile_id, idempotency_key="workspace-cost-quota-1")
        with self.assertRaisesRegex(ValueError, "projected cost"):
            asyncio.run(self.service.create_job(request, owner_id="researcher-cost"))

    def test_profile_override_bounds_and_pretraining_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "between"):
            self.service.resolve_spec(self.request(overrides={"steps": 999999}))
        with self.assertRaisesRegex(ValueError, "does not permit"):
            self.service.resolve_spec(self.request(overrides={"learning_rate": 1}))
        with self.assertRaisesRegex(ValueError, "immutable inputs"):
            self.service.resolve_spec(
                self.request(profile_id="craftly-7b-fsdp", idempotency_key="workspace-pretrain-1")
            )

    def test_7b_profile_requires_untampered_matching_model_progression(self) -> None:
        common = {
            "profile_id": "craftly-7b-fsdp",
            "dataset_manifest_uri": "/data/manifest.json",
            "dataset_manifest_sha256": "b" * 64,
            "tokenizer_uri": "/data/tokenizer.json",
            "tokenizer_sha256": "c" * 64,
        }
        with self.assertRaisesRegex(ValueError, "model progression"):
            self.service.resolve_spec(
                self.request(
                    **common,
                    idempotency_key="progression-missing",
                    model_progression_decision_uri=None,
                    model_progression_decision_sha256=None,
                )
            )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.service.resolve_spec(
                self.request(
                    **common,
                    idempotency_key="progression-tampered",
                    model_progression_decision_sha256="f" * 64,
                )
            )

        wrong = ModelProgressionDecision(
            status="authorized",
            selected_tokenizer_vocab_size=64_000,
            selected_architecture="1b_moe",
            target_model_profile="7b_moe",
            target_model_contract_sha256=model_profile("7b_moe").model_copy(
                update={"vocab_size": 64_000}
            ).contract_sha256(),
            tokenizer_promotion_sha256="1" * 64,
            architecture_promotion_sha256="2" * 64,
            scaling_promotion_sha256="3" * 64,
            dataset_manifest_sha256="4" * 64,
            split_manifest_sha256="5" * 64,
            evaluation_manifest_sha256="6" * 64,
            selected_tokenizer_manifest_sha256="7" * 64,
        ).sealed()
        wrong_path = self.root / "wrong-model-progression.json"
        wrong_path.write_text(json.dumps(wrong.model_dump(mode="json")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not authorize this model profile"):
            self.service.resolve_spec(
                self.request(
                    **common,
                    idempotency_key="progression-wrong-target",
                    model_progression_decision_uri=str(wrong_path),
                    model_progression_decision_sha256=hashlib.sha256(wrong_path.read_bytes()).hexdigest(),
                )
            )

    def test_qualification_next_milestone_is_locked_until_previous_proof(self) -> None:
        common = {
            "dataset_manifest_uri": "file:///data/manifest.json",
            "dataset_manifest_sha256": "b" * 64,
            "tokenizer_uri": "file:///data/tokenizer.json",
            "tokenizer_sha256": "c" * 64,
            "qualification_campaign_id": "defensive-staircase-v1",
        }
        first = self.request(
            profile_id="defensive-staircase-notebook",
            idempotency_key="qualification-first",
            overrides={"steps": 1000},
            qualification_milestone_id="steps-0001000",
            **common,
        )
        asyncio.run(self.service.create_job(first, owner_id="qualification-owner"))
        second = self.request(
            profile_id="defensive-staircase-notebook",
            idempotency_key="qualification-second",
            overrides={"steps": 2000},
            qualification_milestone_id="steps-0002000",
            **common,
        )
        with self.assertRaisesRegex(ValueError, "previous job status=queued"):
            asyncio.run(self.service.create_job(second, owner_id="qualification-owner"))

    def test_notebook_claim_events_redaction_ordering_and_terminal_state(self) -> None:
        job = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        running, attempt = asyncio.run(self.service.claim_notebook_job(job.job_id))
        self.assertEqual(running.status, JobStatus.RUNNING)
        events = asyncio.run(
            self.service.ingest_events(
                job.job_id,
                TrainingEventBatch(
                    attempt_id=attempt.attempt_id,
                    events=[
                        TrainingEventInput(
                            source_sequence=1,
                            kind="metric",
                            stage="training",
                            status="running",
                            step=1,
                            max_steps=1000,
                            loss=7.1,
                            payload={"api_key": "must-not-leak", "tokens": 128},
                        ),
                        TrainingEventInput(
                            source_sequence=2,
                            kind="status",
                            stage="pipeline",
                            status="complete",
                            message="authorization=must-not-leak",
                        ),
                    ],
                ),
            )
        )
        self.assertEqual([item.sequence for item in events], [1, 2])
        self.assertEqual(events[0].payload["api_key"], "[REDACTED]")
        self.assertNotIn("must-not-leak", events[1].message or "")
        completed = asyncio.run(self.service.get_job(job.job_id))
        self.assertEqual(completed.status, JobStatus.SUCCEEDED)
        audit = asyncio.run(self.service.list_audit_events(job.job_id))
        self.assertEqual(audit[0].action, "training.job.transition")
        self.assertEqual(audit[-1].action, "training.job.create")

    def test_event_archiver_writes_gzip_and_advances_cursor(self) -> None:
        job = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        _, attempt = asyncio.run(self.service.claim_notebook_job(job.job_id))
        asyncio.run(
            self.service.ingest_events(
                job.job_id,
                TrainingEventBatch(
                    attempt_id=attempt.attempt_id,
                    events=[
                        TrainingEventInput(source_sequence=index, kind="metric", stage="training", step=index, loss=7.0 - index / 10)
                        for index in range(1, 6)
                    ],
                ),
            )
        )
        artifacts = asyncio.run(TrainingEventArchiver(self.service, chunk_events=100).archive_once())
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].kind, "log")
        self.assertTrue(artifacts[0].uri.endswith(".jsonl.gz"))
        updated = asyncio.run(self.service.get_job(job.job_id))
        self.assertEqual(updated.archived_event_sequence, 5)

    def test_worker_wal_replay_is_deduplicated_by_rank_and_source_sequence(self) -> None:
        job = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        _, attempt = asyncio.run(self.service.claim_notebook_job(job.job_id))
        batch = TrainingEventBatch(
            attempt_id=attempt.attempt_id,
            events=[TrainingEventInput(source_sequence=9, rank=0, kind="metric", stage="training", step=9, loss=6.1)],
        )
        first = asyncio.run(self.service.ingest_events(job.job_id, batch))
        replay = asyncio.run(self.service.ingest_events(job.job_id, batch))
        self.assertEqual(len(first), 1)
        self.assertEqual(replay, [])
        current = asyncio.run(self.service.get_job(job.job_id))
        self.assertEqual(current.event_sequence, 1)

    def test_service_account_bootstrap_and_refresh_session(self) -> None:
        credential = asyncio.run(
            self.service.create_service_account(
                name="kaggle-validation",
                scopes=["training:jobs:create", "training:jobs:read"],
            )
        )
        account = asyncio.run(self.service.authenticate_bootstrap(credential.bootstrap_token))
        self.assertIsNotNone(account)
        assert account is not None
        session = asyncio.run(self.service.create_refresh_session(account, ttl_seconds=3600))
        refreshed = asyncio.run(self.service.authenticate_refresh(session.session_id, session.refresh_token))
        self.assertEqual(refreshed.service_account_id if refreshed else None, account.service_account_id)
        self.assertIsNone(asyncio.run(self.service.authenticate_refresh(session.session_id, "wrong-refresh-token")))
        second_session = asyncio.run(self.service.create_refresh_session(account, ttl_seconds=3600))
        self.assertTrue(asyncio.run(self.service.revoke_refresh(second_session.session_id, second_session.refresh_token)))
        self.assertIsNone(asyncio.run(self.service.authenticate_refresh(second_session.session_id, second_session.refresh_token)))

    def test_cancel_and_resume_require_verified_promoted_checkpoint(self) -> None:
        job = asyncio.run(self.service.create_job(self.request(), owner_id="researcher-1"))
        cancelled = asyncio.run(self.service.cancel_job(job.job_id))
        self.assertEqual(cancelled.status, JobStatus.CANCELLED)
        with self.assertRaisesRegex(ValueError, "promoted"):
            asyncio.run(self.service.resume_job(job.job_id))
        asyncio.run(
            self.service.register_artifact(
                TrainingArtifact(
                    artifact_id="11111111-1111-1111-1111-111111111111",
                    job_id=job.job_id,
                    kind="checkpoint",
                    uri="s3://test/checkpoint",
                    sha256="a" * 64,
                    size_bytes=100,
                    promoted=True,
                )
            )
        )
        resumed = asyncio.run(self.service.resume_job(job.job_id))
        self.assertEqual(resumed.status, JobStatus.QUEUED)

    def test_checkpoint_promotion_requires_verified_reload_and_passing_evaluation(self) -> None:
        request = self.request(
            profile_id="craftly-7b-fsdp",
            idempotency_key="workspace-checkpoint-1",
            dataset_manifest_uri="s3://dataset/manifest.json",
            dataset_manifest_sha256="b" * 64,
            tokenizer_uri="s3://dataset/tokenizer.json",
            tokenizer_sha256="c" * 64,
        )
        job = asyncio.run(self.service.create_job(request, owner_id="admin"))
        provisioning = asyncio.run(
            self.store.transition_job(job.job_id, JobStatus.PROVISIONING, expected_version=job.version)
        )
        attempt = asyncio.run(self.service.create_attempt(provisioning))
        manifest_uri = "s3://checkpoints/step-100/manifest.json"
        asyncio.run(
            self.service.register_artifact(
                TrainingArtifact(
                    artifact_id="33333333-3333-3333-3333-333333333333",
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    kind="checkpoint",
                    uri=manifest_uri,
                    sha256="d" * 64,
                    size_bytes=100,
                )
            )
        )
        checkpoint = asyncio.run(
            self.service.commit_checkpoint(
                job.job_id,
                CheckpointCommitRequest(
                    attempt_id=attempt.attempt_id,
                    step=100,
                    manifest_uri=manifest_uri,
                    manifest_sha256="d" * 64,
                    dataset_sha256="b" * 64,
                    tokenizer_sha256="c" * 64,
                    topology={"world_size": 16},
                    reload_verified=True,
                ),
            )
        )
        report_uri = "s3://checkpoints/step-100/eval.json"
        asyncio.run(
            self.service.register_artifact(
                TrainingArtifact(
                    artifact_id="44444444-4444-4444-4444-444444444444",
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    kind="evaluation",
                    uri=report_uri,
                    sha256="e" * 64,
                    size_bytes=100,
                )
            )
        )
        asyncio.run(
            self.service.commit_evaluation(
                job.job_id,
                EvaluationCommitRequest(
                    checkpoint_id=checkpoint.checkpoint_id,
                    status="complete",
                    report_uri=report_uri,
                    report_sha256="e" * 64,
                    decision="pass",
                    metrics={"validation_loss": 4.2},
                ),
            )
        )
        promoted = asyncio.run(self.service.promote_checkpoint(job.job_id, checkpoint.checkpoint_id, actor_id="admin"))
        self.assertTrue(promoted.promoted)

    def test_scheduler_manifests_are_structured_and_digest_pinned(self) -> None:
        registry = TrainingProfileRegistry.from_file()
        request = self.request(
            profile_id="craftly-7b-fsdp",
            idempotency_key="workspace-cluster-1",
            dataset_manifest_uri="/data/manifest.json",
            dataset_manifest_sha256="b" * 64,
            tokenizer_uri="/data/tokenizer.json",
            tokenizer_sha256="c" * 64,
        )
        spec = self.service.resolve_spec(request)
        job = asyncio.run(
            self.store.create_job(
                self._job_from_spec(spec, owner="admin", key="workspace-cluster-1")
            )
        )
        attempt = asyncio.run(self.service.create_attempt(job))
        manifest = KubernetesPyTorchAdapter().manifest(job, attempt)
        self.assertEqual(manifest["kind"], "PyTorchJob")
        container = manifest["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
        self.assertIn("@sha256:", container["image"])
        self.assertIsInstance(container["command"], list)
        self.assertNotIn("shell", json.dumps(manifest).lower())
        self.assertIn("CRAFTLY_WORKSPACE_TOKEN_FILE", json.dumps(manifest))
        command = container["command"]
        self.assertIn("src.craftly.model_ops.distributed_worker", command)
        volumes = manifest["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["volumes"]
        workspace = next(item for item in volumes if item["name"] == "workspace")
        self.assertEqual(workspace["persistentVolumeClaim"]["claimName"], "craftly-training-checkpoints")

        standard_spec = self.service.resolve_spec(
            self.request(
                profile_id="defensive-10k",
                idempotency_key="workspace-kubernetes-1",
                dataset_manifest_uri="/data/manifest.json",
                dataset_manifest_sha256="b" * 64,
                tokenizer_uri="/data/tokenizer.json",
                tokenizer_sha256="c" * 64,
            )
        )
        standard_job = self._job_from_spec(standard_spec, owner="admin", key="workspace-kubernetes-1")
        standard_manifest = KubernetesSchedulerAdapter().manifest(standard_job, attempt)
        standard_volumes = standard_manifest["spec"]["template"]["spec"]["volumes"]
        standard_workspace = next(item for item in standard_volumes if item["name"] == "workspace")
        self.assertEqual(standard_workspace["persistentVolumeClaim"]["claimName"], "craftly-training-checkpoints")

    @staticmethod
    def _job_from_spec(spec: object, *, owner: str, key: str):
        from src.craftly.training_workspace import TrainingJob

        return TrainingJob(
            job_id="22222222-2222-2222-2222-222222222222",
            owner_id=owner,
            idempotency_key=key,
            spec=spec,
            status=JobStatus.QUEUED,
            version=1,
        )

    def test_slurm_script_uses_fixed_srun_path_and_strict_shell(self) -> None:
        data_dir = self.root / "megatron-data"
        data_dir.mkdir()
        prefix = data_dir / "corpus"
        prefix.with_suffix(".bin").write_bytes(b"tokens")
        prefix.with_suffix(".idx").write_bytes(b"index")
        manifest = data_dir / "manifest.json"
        manifest.write_text(json.dumps({"megatron_data_prefix": str(prefix)}), encoding="utf-8")
        tokenizer = data_dir / "tokenizer.json"
        tokenizer.write_text("{}", encoding="utf-8")
        megatron_root = self.root / "Megatron-LM"
        megatron_root.mkdir()
        (megatron_root / "pretrain_gpt.py").write_text("# test entrypoint\n", encoding="utf-8")
        request = self.request(
            profile_id="craftly-60b-hybrid",
            idempotency_key="workspace-slurm-1",
            dataset_manifest_uri=str(manifest),
            dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            tokenizer_uri=str(tokenizer),
            tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        )
        spec = self.service.resolve_spec(request)
        job = self._job_from_spec(spec, owner="admin", key="workspace-slurm-1")
        attempt = asyncio.run(self.service.create_attempt(asyncio.run(self.store.create_job(job))))
        with patch.dict(
            "os.environ",
            {"CRAFTLY_MEGATRON_ROOT": str(megatron_root), "CRAFTLY_SHARED_CHECKPOINT_ROOT": str(self.root)},
        ):
            script = SlurmSchedulerAdapter(work_dir=self.root / "slurm").script(job, attempt)
        self.assertIn("set -euo pipefail", script)
        self.assertIn("srun", script)
        self.assertIn("CRAFTLY_WORKSPACE_TOKEN_FILE", script)
        self.assertNotIn("eval ", script)

    def test_controller_blocks_missing_cluster_dependency_honestly(self) -> None:
        request = self.request(
            profile_id="craftly-7b-fsdp",
            idempotency_key="workspace-controller-1",
            dataset_manifest_uri="/data/manifest.json",
            dataset_manifest_sha256="b" * 64,
            tokenizer_uri="/data/tokenizer.json",
            tokenizer_sha256="c" * 64,
        )
        job = asyncio.run(self.service.create_job(request, owner_id="admin"))

        class MissingScheduler(KubernetesSchedulerAdapter):
            name = "kubernetes_pytorch"

            async def validate(self, spec):
                raise RuntimeError("cluster is unavailable")

        controller = TrainingController(self.service, schedulers={"kubernetes_pytorch": MissingScheduler()})
        blocked = asyncio.run(controller.reconcile_job(job))
        self.assertEqual(blocked.status, JobStatus.BLOCKED)
        self.assertIn("cluster is unavailable", blocked.failure_detail or "")

    def test_workspace_readiness_never_marks_dev_fallbacks_ready(self) -> None:
        report = workspace_readiness(self.service)
        self.assertEqual(report["status"], "blocked_missing_dependency")
        self.assertIn("PostgresTrainingStore", report["missing_dependencies"])
        self.assertEqual(report["cluster_status"], "built_not_cluster_proven")

    def test_client_rejects_insecure_remote_url_and_formats_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            _validate_workspace_url("http://training.example.com")
        self.assertEqual(_validate_workspace_url("http://127.0.0.1:8090"), "http://127.0.0.1:8090")
        line = format_event({"sequence": 1, "stage": "training", "status": "running", "step": 10, "loss": 3.2})
        self.assertIn("step=10", line)
        self.assertIn("loss=3.2", line)

    def test_state_machine_has_no_live_self_transition(self) -> None:
        for state, targets in ALLOWED_TRANSITIONS.items():
            self.assertNotIn(state, targets)
        self.assertEqual(ALLOWED_TRANSITIONS[JobStatus.SUCCEEDED], set())
        self.assertEqual(ALLOWED_TRANSITIONS[JobStatus.BLOCKED], set())


class CraftlyTrainingWorkspaceGatewayTest(unittest.TestCase):
    def test_gateway_job_lifecycle_and_token_exchange(self) -> None:
        from src.craftly.gateway import api as gateway_api

        with tempfile.TemporaryDirectory() as temp_dir:
            original_workspace = gateway_api.TRAINING_WORKSPACE
            original_auth = gateway_api.AUTH_CONFIG
            service = TrainingWorkspaceService(
                store=InMemoryTrainingStore(),
                events=InMemoryEventBus(),
                object_store=LocalObjectStore(Path(temp_dir) / "objects"),
                profiles=TrainingProfileRegistry.from_file(),
            )
            gateway_api.TRAINING_WORKSPACE = service
            gateway_api.AUTH_CONFIG = AuthConfig(enabled=False, jwt_secret="x" * 64)
            try:
                client = TestClient(gateway_api.app)
                credential = client.post(
                    "/v1/training/service-accounts",
                    json={"name": "gateway-client", "scopes": ["training:jobs:create", "training:jobs:read"]},
                )
                self.assertEqual(credential.status_code, 200, credential.text)
                exchange = client.post(
                    "/v1/training/token/exchange",
                    json={"bootstrap_token": credential.json()["bootstrap_token"]},
                )
                self.assertEqual(exchange.status_code, 200, exchange.text)
                self.assertEqual(exchange.json()["expires_in"], 900)
                create = client.post(
                    "/v1/training/jobs",
                    json={
                        "profile_id": "defensive-1k",
                        "idempotency_key": "gateway-workspace-1",
                        "git_commit": "abcdef1",
                        "container_digest": "sha256:" + ("a" * 64),
                    },
                )
                self.assertEqual(create.status_code, 200, create.text)
                job_id = create.json()["job_id"]
                self.assertEqual(client.get(f"/v1/training/jobs/{job_id}").status_code, 200)
                claim = client.post(f"/v1/training/jobs/{job_id}/claim")
                self.assertEqual(claim.status_code, 200, claim.text)
                self.assertEqual(claim.json()["job"]["status"], "running")
                self.assertIn("worker_access_token", claim.json())
                audit = client.get(f"/v1/training/jobs/{job_id}/audit")
                self.assertEqual(audit.status_code, 200, audit.text)
                self.assertEqual(audit.json()["audit_events"][0]["action"], "training.job.create")
            finally:
                asyncio.run(service.close())
                gateway_api.TRAINING_WORKSPACE = original_workspace
                gateway_api.AUTH_CONFIG = original_auth


if __name__ == "__main__":
    unittest.main()
