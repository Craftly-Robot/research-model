from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.craftly.architecture_integrity import run_architecture_integrity
from src.craftly.db import LocalStore, PostgresRAGStore
from src.craftly.db.migration_runner import expand_psql_includes, load_migrations
from src.craftly.evaluation.benchmarks import BenchmarkHarness, built_in_security_tasks
from src.craftly.gateway import api as gateway_api
from src.craftly.identity.auth import AuthConfig, AuthError, validate_token_issue_request
from src.craftly.identity.quota import AsyncLocalQuotaStore, LocalQuotaStore
from src.craftly.indexing import (
    ContextBuilder,
    EmbeddingPair,
    EmbeddingTrainingConfig,
    RAGEvaluationGovernance,
    RAGEvaluationTask,
    ScratchCodeEmbeddingModel,
    ScratchEmbeddingConfig,
    build_rag_scale_plan,
    save_scratch_embedding_checkpoint,
    train_scratch_embedding_model,
)
from src.craftly.indexing import LocalVectorIndex, RepositoryIndexer, VectorBackendConfig, create_vector_index, vector_capabilities
from src.craftly.indexing.repository_indexer import RAGIndexCoordinator, RedisRAGControl
from src.craftly.indexing.vector_index import QdrantVectorIndex, VectorSearchReport, VectorSearchResult
from src.craftly.learning.capacity import CapacityPlanConfig, build_capacity_plan
from src.craftly.learning.storage import LocalObjectStore
from src.craftly.memory import MemoryIngestRequest, UnifiedMemoryManager
from src.craftly.model_ops.backends import MockModelBackend
from src.craftly.patches import PatchVerifyRequest
from src.craftly.patches.service import PatchService
from src.craftly.planning.engine import IntentPlanningEngine
from src.craftly.patches.verified_loop import RepositoryPatchLoopRequest, run_repository_patch_loop
from src.craftly.runtime.taskgraph import AgentRunCreateRequest, TaskFailRequest, TaskGraphRuntime
from src.craftly.security.audit import run_security_audit
from src.craftly.shared.config_contracts import (
    load_active_model_contract,
    load_eval_schedule_contract,
    load_mix_ratios_contract,
    load_security_audit_contract,
    load_verifier_policy_contract,
)
from src.craftly.tools import HardenedToolExecutor, ToolExecuteRequest
from src.craftly.tools.sandbox import HardenedSandboxPolicy, SandboxRunRequest
from src.craftly.verifier.runtime import VerificationRequest, VerifierRuntime, load_verifier_policy


class CraftlyProductionHardeningTest(unittest.TestCase):
    def test_postgres_tenant_facades_share_pool_without_closing_it(self) -> None:
        class SharedPool:
            def __init__(self) -> None:
                self.closed = False

            def connection(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        pool = SharedPool()
        first = PostgresRAGStore.from_shared_pool(pool, organization_id="d7d9fc25-8a27-4133-b420-c0290e0ec074")
        second = PostgresRAGStore.from_shared_pool(pool, organization_id="6eb798b7-e88b-4d50-81d0-f15a14020cf3")
        self.assertIs(first._pool, second._pool)
        first.close()
        second.close()
        self.assertFalse(pool.closed)
        with self.assertRaisesRegex(ValueError, "organization_id must be a UUID"):
            PostgresRAGStore.from_shared_pool(pool, organization_id="tenant-name")

    def test_redis_worker_wake_uses_consumer_group_and_acknowledges_signal(self) -> None:
        class FakeRedis:
            def __init__(self) -> None:
                self.acked: list[str] = []

            async def xgroup_create(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("BUSYGROUP Consumer Group name already exists")

            async def xreadgroup(self, *_args: object, **_kwargs: object) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
                return [("craftly:rag:index-jobs", [("171-0", {"job_id": "job"})])]

            async def xack(self, _stream: str, _group: str, *message_ids: str) -> None:
                self.acked.extend(message_ids)

        async def scenario() -> None:
            control = RedisRAGControl("redis://127.0.0.1:6379/0")
            fake = FakeRedis()
            control._client = fake
            self.assertTrue(await control.wait_for_work("worker-001", timeout_seconds=0.1))
            self.assertEqual(fake.acked, ["171-0"])
            with self.assertRaisesRegex(ValueError, "unsafe characters"):
                await control.wait_for_work("bad worker", timeout_seconds=0.1)

        asyncio.run(scenario())

    def test_failed_index_generation_preserves_previous_searchable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            source = workspace / "service.py"
            source.write_text("def stable():\n    return 'v1'\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "rag.sqlite3") as store:
                project = store.create_project(name="revision", repo_path=str(workspace))
                first = RepositoryIndexer(store).index_project(project_id=project["id"])
                old_chunks = store.list_chunks(project["id"])
                source.write_text("def stable():\n    return 'v2'\n", encoding="utf-8")
                indexer = RepositoryIndexer(store)
                with patch.object(indexer, "chunk_file", side_effect=RuntimeError("forced build failure")):
                    with self.assertRaisesRegex(RuntimeError, "forced build failure"):
                        indexer.index_project(project_id=project["id"])
                self.assertEqual(store.get_project(project["id"])["active_index_revision"], first.revision_id)
                self.assertEqual(store.list_chunks(project["id"]), old_chunks)
                self.assertEqual(store.index_status(project["id"])["status"], "failed")

    def test_production_indexing_rejects_missing_tree_sitter_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "rag.sqlite3") as store:
                project = store.create_project(name="strict-parser", repo_path=str(workspace))
                indexer = RepositoryIndexer(store, production_mode=False)
                indexer.production_mode = True
                with patch(
                    "src.craftly.indexing.repository_indexer.tree_sitter_parser",
                    side_effect=RuntimeError("tree-sitter-language-pack is unavailable"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "tree-sitter-language-pack is unavailable"):
                        indexer.index_project(project_id=project["id"])
                self.assertEqual(store.index_status(project["id"])["status"], "failed")

    def test_index_snapshot_is_immutable_and_hash_bound_in_object_storage(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace_dir,
            tempfile.TemporaryDirectory() as db_dir,
            tempfile.TemporaryDirectory() as object_dir,
        ):
            workspace = Path(workspace_dir)
            (workspace / "main.py").write_text("print('snapshot')\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "snapshot.sqlite3") as store:
                project = store.create_project(name="snapshot", repo_path=str(workspace))
                report = RepositoryIndexer(store, object_store=LocalObjectStore(object_dir)).index_project(
                    project_id=project["id"]
                )
                revision = store.get_index_revision(report.revision_id)
                binding = revision["manifest"]["source_snapshot_object"]
                self.assertEqual(binding["snapshot_sha256"], report.source_snapshot_sha256)
                self.assertEqual(len(binding["sha256"]), 64)
                self.assertTrue(Path(binding["uri"]).is_file())

    def test_project_access_is_bound_to_organization_and_membership(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            with LocalStore(Path(db_dir) / "tenant.sqlite3") as store:
                project = store.create_project(
                    name="tenant",
                    repo_path=workspace_dir,
                    organization_id="org-a",
                    owner_user_id="owner-a",
                )
                self.assertEqual(
                    store.require_project_access(project["id"], "org-a", user_id="owner-a")["id"],
                    project["id"],
                )
                with self.assertRaises(KeyError):
                    store.require_project_access(project["id"], "org-b")
                with self.assertRaises(PermissionError):
                    store.require_project_access(project["id"], "org-a", user_id="stranger")
                store.insert_memory_entry(
                    project_id=None,
                    kind="project:project_fact",
                    content={"secret": "local-only"},
                )
                self.assertEqual(store.list_memory_entries(project["id"]), [])

    def test_hybrid_context_fuses_semantic_evidence_and_reports_outage(self) -> None:
        class FakeVectorIndex:
            config = VectorBackendConfig(backend="qdrant", dims=64, embedding_model="Craftly-Code-Embed-v1")

            def __init__(self, chunk_id: str) -> None:
                self.chunk_id = chunk_id

            def search(self, **kwargs: object) -> VectorSearchReport:
                return VectorSearchReport(
                    organization_id=str(kwargs["organization_id"]),
                    project_id=str(kwargs["project_id"]),
                    revision_id=str(kwargs["revision_id"]),
                    query=str(kwargs["query"]),
                    backend="qdrant",
                    dims=64,
                    results=[
                        VectorSearchResult(
                            chunk_id=self.chunk_id,
                            path="auth.py",
                            start_line=1,
                            end_line=2,
                            score=0.99,
                        )
                    ],
                )

            def sync_project(self, **_kwargs: object) -> object:
                raise NotImplementedError

        class FailingVectorIndex(FakeVectorIndex):
            def search(self, **_kwargs: object) -> VectorSearchReport:
                raise RuntimeError("qdrant unavailable at https://secret.internal")

        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "auth.py").write_text("def authorize(user):\n    return bool(user)\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "hybrid.sqlite3") as store:
                project = store.create_project(name="hybrid", repo_path=str(workspace))
                RepositoryIndexer(store).index_project(project_id=project["id"])
                chunk = store.list_chunks(project["id"])[0]
                report = ContextBuilder(store, vector_index=FakeVectorIndex(chunk["id"])).build(
                    project_id=project["id"], query="authorization", token_budget=2000
                )
                self.assertEqual(report.retrieval_mode, "hybrid")
                self.assertFalse(report.degraded)
                self.assertGreater(report.candidate_counts["semantic"], 0)
                self.assertTrue(report.report_sha256)
                degraded = ContextBuilder(store, vector_index=FailingVectorIndex(chunk["id"])).build(
                    project_id=project["id"], query="authorization", token_budget=2000
                )
                self.assertTrue(degraded.degraded)
                self.assertEqual(degraded.retrieval_mode, "degraded_lexical_graph")
                self.assertNotIn("secret.internal", degraded.degraded_reason or "")
                self.assertTrue(degraded.chunks)

    def test_rag_production_evaluation_cannot_bypass_500_task_gate(self) -> None:
        report = ContextBuilder().evaluate(
            [
                RAGEvaluationTask(
                    task_id="one",
                    project_id="missing",
                    query="find symbol",
                    relevant_chunk_ids=["expected"],
                )
            ],
            strict=True,
        )
        self.assertEqual(report.status, "blocked")
        self.assertIn("500", report.blockers[0])

    def test_rag_scale_plan_requires_real_100m_target_for_production_claim(self) -> None:
        small = build_rag_scale_plan(target_chunks=1_000_000)
        production = build_rag_scale_plan(target_chunks=100_000_000)
        self.assertFalse(small.production_scale_target_met)
        self.assertTrue(production.production_scale_target_met)
        self.assertGreaterEqual(production.shard_count, 16)
        self.assertLessEqual(production.maximum_points_per_shard, 10_000_000)

    def test_rag_job_idempotency_retry_cancellation_and_outbox_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "jobs.sqlite3") as store:
                project = store.create_project(name="jobs", repo_path=str(workspace))
                first = store.create_index_job(
                    organization_id="local",
                    project_id=project["id"],
                    idempotency_key="index-jobs-0001",
                    request={"max_chunk_tokens": 512},
                    max_attempts=2,
                )
                duplicate = store.create_index_job(
                    organization_id="local",
                    project_id=project["id"],
                    idempotency_key="index-jobs-0001",
                    request={"max_chunk_tokens": 512},
                    max_attempts=2,
                )
                self.assertEqual(first["id"], duplicate["id"])
                claimed = store.claim_index_job(worker_id="worker-test-1")
                self.assertIsNotNone(claimed)
                retried = store.fail_index_job(
                    first["id"],
                    worker_id="worker-test-1",
                    error="temporary dependency outage",
                    retry_delay_seconds=0,
                )
                self.assertEqual(retried["status"], "queued")
                claimed_again = store.claim_index_job(worker_id="worker-test-2")
                self.assertEqual(claimed_again["attempt"], 2)
                dead = store.fail_index_job(
                    first["id"],
                    worker_id="worker-test-2",
                    error="dependency outage repeated",
                    retry_delay_seconds=0,
                )
                self.assertEqual(dead["status"], "dead_letter")
                second = store.create_index_job(
                    organization_id="local",
                    project_id=project["id"],
                    idempotency_key="index-jobs-0002",
                    request={},
                    max_attempts=2,
                )
                cancelled = store.request_index_job_cancel(second["id"], organization_id="local")
                self.assertEqual(cancelled["status"], "cancelled")

    def test_tree_sitter_multilingual_chunks_resolve_interprocedural_edges(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "lib.rs").write_text(
                "fn helper(value: i32) -> i32 { value + 1 }\nfn run() -> i32 { helper(4) }\n",
                encoding="utf-8",
            )
            (workspace / "main.go").write_text(
                "package main\nfunc helperGo(v int) int { return v + 1 }\nfunc runGo() int { return helperGo(4) }\n",
                encoding="utf-8",
            )
            (workspace / "main.js").write_text(
                "function helperJs(v) { return v + 1; }\nfunction runJs() { return helperJs(4); }\n",
                encoding="utf-8",
            )
            with LocalStore(Path(db_dir) / "trees.sqlite3") as store:
                project = store.create_project(name="trees", repo_path=str(workspace))
                report = RepositoryIndexer(store).index_project(project_id=project["id"])
                chunks = store.list_chunks(project["id"])
                self.assertEqual(report.status, "ready")
                parsers = {str(chunk["metadata"].get("parser")) for chunk in chunks}
                self.assertIn("tree_sitter", parsers)
                resolved = [
                    edge
                    for chunk in chunks
                    for edge in chunk["metadata"].get("resolved_calls", [])
                ]
                self.assertGreaterEqual(len(resolved), 3)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
    def test_scratch_embedding_model_has_finite_loss_and_hash_bound_manifest(self) -> None:
        import torch

        config = ScratchEmbeddingConfig(
            vocab_size=256,
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            projection_dimension=64,
            intermediate_size=128,
            max_sequence_length=32,
            dropout=0.0,
        )
        model = ScratchCodeEmbeddingModel(config)
        tokens = torch.randint(0, 256, (2, 16))
        mask = torch.ones_like(tokens)
        loss = model.contrastive_loss(tokens, mask, tokens.flip(1), mask)
        self.assertTrue(torch.isfinite(loss).item())
        with tempfile.TemporaryDirectory() as output_dir:
            manifest = save_scratch_embedding_checkpoint(
                model,
                output_dir,
                tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
                dataset_manifest_sha256=hashlib.sha256(b"dataset").hexdigest(),
            )
            self.assertTrue(manifest["scratch_only"])
            self.assertFalse(manifest["borrowed_weights"])
            self.assertEqual(len(manifest["checkpoint_sha256"]), 64)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tokenizers") is not None,
        "PyTorch/tokenizers not installed",
    )
    def test_scratch_embedding_trainer_produces_best_hash_bound_checkpoint(self) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocab = {"<|pad|>": 0, "<|unk|>": 1}
            for index in range(2, 256):
                vocab[f"tok{index}"] = index
            tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            pairs_path = root / "pairs.jsonl"
            with pairs_path.open("w", encoding="utf-8") as handle:
                for index in range(40):
                    pair = EmbeddingPair(
                        pair_id=hashlib.sha256(f"pair-{index}".encode()).hexdigest(),
                        query=f"tok{2 + index} tok100",
                        positive=f"tok{2 + index} tok101",
                        hard_negatives=[f"tok{120 + (index % 20)} tok102"],
                        category="task_evidence",
                        source_revision="revision-1",
                        lineage_key=f"repository-{index}",
                    )
                    handle.write(pair.model_dump_json() + "\n")
            config = EmbeddingTrainingConfig(
                model=ScratchEmbeddingConfig(
                    vocab_size=256,
                    hidden_size=64,
                    num_layers=1,
                    num_attention_heads=4,
                    projection_dimension=64,
                    intermediate_size=128,
                    max_sequence_length=32,
                    dropout=0.0,
                ),
                steps=1,
                batch_size=2,
                warmup_steps=0,
                validation_interval=1,
                checkpoint_interval=1,
                validation_fraction=0.25,
                max_validation_pairs=20,
                precision="fp32",
                device="cpu",
            )
            report = train_scratch_embedding_model(
                pairs_path=pairs_path,
                tokenizer_path=tokenizer_path,
                output_dir=root / "output",
                config=config,
            )
            self.assertEqual(report.status, "passed", report.model_dump())
            self.assertEqual(report.best_step, 1)
            self.assertTrue(Path(report.checkpoint_manifest or "").is_file())

    def test_architecture_integrity_passes_repository_and_detects_synthetic_drift(self) -> None:
        current = run_architecture_integrity()
        self.assertEqual(current.status, "passed", current.model_dump())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "src" / "craftly"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            repeated = """
def repeated(value):
    first = value + 1
    second = first * 2
    third = second - 3
    fourth = third / 4
    if fourth:
        return fourth
    return 0
"""
            (package / "a.py").write_text(
                "from src.craftly.b import repeated\n" + repeated,
                encoding="utf-8",
            )
            (package / "b.py").write_text(
                "from src.craftly.a import repeated\n" + repeated,
                encoding="utf-8",
            )
            drift = run_architecture_integrity(repository_root=root)
            self.assertEqual(drift.status, "failed")
            self.assertTrue(drift.duplicate_function_bodies)
            self.assertTrue(drift.import_cycles)

    def test_migrations_load_and_expand_schema_include(self) -> None:
        migrations = load_migrations()
        self.assertTrue(migrations)
        self.assertTrue(all(not migration.sql.startswith("\ufeff") for migration in migrations))
        expanded = expand_psql_includes(migrations[0].sql, base_dir=Path.cwd())
        self.assertIn("CREATE TABLE IF NOT EXISTS projects", expanded)
        self.assertTrue(any(migration.version == "0003_task_retry" for migration in migrations))
        rag_operations = next(migration for migration in migrations if migration.version == "0008_rag_operations")
        normalized_sql = " ".join(rag_operations.sql.upper().split())
        self.assertIn("SECURITY DEFINER", normalized_sql)
        self.assertIn("SET SEARCH_PATH = PUBLIC, PG_TEMP", normalized_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", normalized_sql)
        self.assertIn("REVOKE ALL ON FUNCTION CRAFTLY_CLAIM_RAG_INDEX_JOB", normalized_sql)
        self.assertIn("REVOKE ALL ON FUNCTION CRAFTLY_CLAIM_RAG_OUTBOX", normalized_sql)

    def test_existing_sqlite_db_auto_adds_task_retry_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "old.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE tasks (
                  id TEXT PRIMARY KEY,
                  task_graph_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  title TEXT NOT NULL,
                  status TEXT NOT NULL,
                  depends_on_json TEXT NOT NULL DEFAULT '[]',
                  input_json TEXT NOT NULL DEFAULT '{}',
                  output_json TEXT NOT NULL DEFAULT '{}',
                  error TEXT,
                  started_at REAL,
                  finished_at REAL,
                  created_at REAL NOT NULL
                )
                """
            )
            connection.commit()
            connection.close()
            with LocalStore(db_path) as store:
                columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(tasks)").fetchall()}
            self.assertIn("attempt", columns)
            self.assertIn("max_attempts", columns)

    def test_hardened_executor_rejects_pathlike_executable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            fake = workspace / "python.exe"
            fake.write_text("not real", encoding="utf-8")
            with LocalStore(Path(db_dir) / "tool.sqlite3") as store:
                project = store.create_project(name="tool", repo_path=str(workspace))
                request = ToolExecuteRequest(
                    project_id=project["id"],
                    tool="test",
                    command=[str(fake), "-c", "print('bad')"],
                )
                with self.assertRaisesRegex(ValueError, "basename"):
                    HardenedToolExecutor(store).execute(request)

    def test_gateway_rejects_unsafe_tool_command_shape(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            original_store = gateway_api.STORE
            gateway_api.STORE = LocalStore(Path(db_dir) / "gateway.sqlite3")
            try:
                project = gateway_api.STORE.create_project(name="gw", repo_path=str(workspace))
                client = TestClient(gateway_api.app)
                response = client.post(
                    "/v1/tools/execute",
                    json={"project_id": project["id"], "tool": "git_diff", "command": ["git", "status"]},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("git diff", response.text)
            finally:
                gateway_api.STORE.close()
                gateway_api.STORE = original_store

    def test_gateway_tools_route_requires_scope_when_auth_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            original_store = gateway_api.STORE
            original_auth = gateway_api.AUTH_CONFIG
            gateway_api.STORE = LocalStore(Path(db_dir) / "gateway-auth.sqlite3")
            gateway_api.AUTH_CONFIG = AuthConfig(enabled=True, jwt_secret="x" * 32)
            try:
                project = gateway_api.STORE.create_project(name="gw-auth", repo_path=str(workspace))
                client = TestClient(gateway_api.app)
                response = client.post(
                    "/v1/tools/execute",
                    json={"project_id": project["id"], "tool": "git_diff", "command": ["git", "diff"]},
                )
                self.assertEqual(response.status_code, 403, response.text)
                self.assertIn("tools:execute", response.text)
            finally:
                gateway_api.STORE.close()
                gateway_api.STORE = original_store
                gateway_api.AUTH_CONFIG = original_auth

    def test_verifier_uses_hardened_executor_and_preserves_output(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "ok.py").write_text("print('verifier-ok')\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "verify.sqlite3") as store:
                project = store.create_project(name="verify", repo_path=str(workspace))
                RepositoryIndexer(store).index_project(project_id=project["id"])
                response = PatchService(store).preview_apply_verify(
                    PatchVerifyRequest(
                        project_id=project["id"],
                        edits=[{"path": "ok.py", "new_content": "print('verifier-ok')\n"}],
                        commands=[["python", "ok.py"]],
                        apply_on_accept=False,
                    )
                )
                self.assertEqual(response.verdict, "accept")
                self.assertIn("verifier-ok", response.verification["test_results"][0]["stdout"])

    def test_taskgraph_retries_before_final_failure(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            with LocalStore(Path(db_dir) / "retry.sqlite3") as store:
                project = store.create_project(name="retry", repo_path=workspace_dir)
                runtime = TaskGraphRuntime(store)
                run = runtime.create_agent_run(AgentRunCreateRequest(project_id=project["id"], prompt="fix retry"))
                first = runtime.advance(run.task_graph_id)
                task_id = first.active_task["id"]
                retry = runtime.fail_task(task_id, TaskFailRequest(error="transient"))
                self.assertEqual(retry.status, "running")
                task = store.get_task(task_id)
                self.assertEqual(task["attempt"], 1)
                self.assertEqual(task["status"], "running")
                final = runtime.fail_task(task_id, TaskFailRequest(error="still broken"))
                self.assertEqual(final.status, "failed")
                self.assertEqual(store.get_task(task_id)["attempt"], 2)

    def test_audit_exclude_config_blocks_unapproved_executable_sink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "config").mkdir()
            (root / "src" / "excluded.py").write_text("eval('1+1')\n", encoding="utf-8")
            (root / "config" / "security_audit_excludes.json").write_text(
                json.dumps(
                    {
                        "excludes": [
                            {
                                "path": "src/excluded.py",
                                "reason": "test fixture exclude must still scan executable sinks",
                                "risk_category": "test_fixture",
                                "allow_executable_sinks": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = run_security_audit(
                root=root,
                run_bandit=False,
                validate_k8s=False,
                run_semgrep=False,
                run_codeql=False,
                run_pip_audit=False,
            )
            self.assertEqual(report.status, "failed")
            self.assertTrue(any(finding.check == "audit_exclude_executable_sink" for finding in report.findings))

    def test_production_config_contracts_validate_current_files(self) -> None:
        mix = load_mix_ratios_contract("config/mix_ratios.json")
        self.assertAlmostEqual(sum(mix.scratch_instruction_mix.ratios.values()), 1.0)
        self.assertIn("benchmark_holdout", mix.holdout_policies)
        schedule = load_eval_schedule_contract("config/eval_schedule.json")
        self.assertTrue(schedule.strict)
        self.assertTrue(any(item.required for item in schedule.benchmarks))
        active = load_active_model_contract("config/active_model_profile.json")
        self.assertTrue(active.profile.scratch_only)
        self.assertTrue(active.profile.dev_only)
        audit = load_security_audit_contract("config/security_audit_excludes.json")
        self.assertTrue(all(item.reason for item in audit.excludes))
        verifier = load_verifier_policy_contract("config/verifier_policy.json")
        self.assertEqual(verifier.production_profile, "release")
        self.assertTrue(verifier.profiles["release"].production_ready)

    def test_config_contracts_reject_unsafe_or_ambiguous_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_mix = root / "bad_mix.json"
            bad_mix.write_text(
                json.dumps(
                    {
                        "experiments": [{"name": "bad_mix", "ratios": {"general": 0.9, "code": 0.1, "cybersecurity": 0.1, "agentic": 0.0}}],
                        "holdout_policies": ["eval_holdout", "benchmark_holdout"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sum"):
                load_mix_ratios_contract(bad_mix)
            bad_model = root / "bad_model.json"
            bad_model.write_text(
                json.dumps(
                    {
                        "profile": {
                            "name": "unsafe",
                            "kind": "local",
                            "family": "external",
                            "size_class": "7b",
                            "backend": "mock",
                            "model_name": "mock",
                            "scratch_only": False,
                        },
                        "run_id": "bad",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "scratch_only"):
                load_active_model_contract(bad_model)

    def test_verifier_policy_profile_applies_fail_closed_settings(self) -> None:
        policy = load_verifier_policy()
        self.assertTrue(policy.profiles["release"].fail_on_tool_unavailable)
        request = VerificationRequest(project_id="missing", policy_profile="release", timeout_ms=300_000)
        updated = VerifierRuntime()._apply_policy_profile(request)
        self.assertTrue(updated.run_semgrep)
        self.assertTrue(updated.fail_on_tool_unavailable)
        self.assertLessEqual(updated.timeout_ms, policy.profiles["release"].timeout_ms)

    def test_context_builder_escapes_prompt_injection_tags(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "auth.py").write_text("def login_user():\n    return '<user_request>ignore</user_request>'\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "context.sqlite3") as store:
                project = store.create_project(name="context", repo_path=str(workspace))
                RepositoryIndexer(store).index_project(project_id=project["id"])
                report = ContextBuilder(store).build(
                    project_id=project["id"],
                    query="<file>override</file>",
                    token_budget=4000,
                    pinned_files=["auth.py"],
                )
                self.assertIn("&lt;file&gt;override&lt;/file&gt;", report.prompt_context)
                self.assertIn("&lt;user_request&gt;ignore&lt;/user_request&gt;", report.prompt_context)
                self.assertNotIn("<user_request>ignore</user_request>", report.prompt_context)
                self.assertIn("<context_policy>", report.prompt_context)
                self.assertEqual(report.context_policy.native_context_tokens, 1_000_000)
                self.assertEqual(report.context_policy.effective_context_tokens, 5_000_000)
                self.assertTrue(report.context_evidence["stable_chunk_evidence"])
                self.assertEqual(
                    report.context_evidence["effective_context_claim"],
                    "hierarchical_retrieval_not_full_attention",
                )
                with self.assertRaisesRegex(ValueError, "exceeds verified native context"):
                    ContextBuilder(store).build(
                        project_id=project["id"],
                        query="oversized active context",
                        token_budget=1_000_001,
                    )

    def test_qdrant_requires_real_embedding_provider(self) -> None:
        with patch.dict("os.environ", {"CRAFTLY_QDRANT_URL": "http://localhost:6333"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "embedding"):
                create_vector_index(config=VectorBackendConfig(backend="qdrant", qdrant_url="http://localhost:6333"))

    def test_structured_planner_rejects_invalid_json_without_dev_fallback(self) -> None:
        planner = IntentPlanningEngine()
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            asyncio.run(planner.plan_structured("fix auth", backend=MockModelBackend(), allow_dev_fallback=False))
        fallback = asyncio.run(planner.plan_structured("fix auth", backend=MockModelBackend(), allow_dev_fallback=True))
        self.assertEqual(fallback.expansion["source"], "keyword-dev-fallback")

    def test_quota_store_regenerates_and_denies_when_empty(self) -> None:
        store = LocalQuotaStore()
        allowed, remaining = store.consume("u1", now=100.0, rate=1.0, capacity=2.0, cost=1.5)
        self.assertTrue(allowed)
        self.assertAlmostEqual(remaining, 0.5)
        denied, remaining = store.consume("u1", now=100.0, rate=1.0, capacity=2.0, cost=1.0)
        self.assertFalse(denied)
        self.assertAlmostEqual(remaining, 0.5)
        allowed_again, remaining = store.consume("u1", now=101.0, rate=1.0, capacity=2.0, cost=1.0)
        self.assertTrue(allowed_again)
        self.assertAlmostEqual(remaining, 0.5)

    def test_vector_search_and_patch_verify_loop(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "auth.py").write_text(
                "def login_user(username, password):\n    return password == 'secret'\n",
                encoding="utf-8",
            )
            with LocalStore(Path(db_dir) / "craftly.sqlite3") as store:
                project = store.create_project(name="prod", repo_path=str(workspace))
                RepositoryIndexer(store).index_project(project_id=project["id"])
                search = LocalVectorIndex(store).search(project_id=project["id"], query="login password", top_k=3)
                self.assertTrue(search.results)
                selected = create_vector_index(store, VectorBackendConfig(backend="local_hashing", dims=256))
                selected_search = selected.search(project_id=project["id"], query="login password", top_k=1)
                self.assertEqual(selected_search.backend, "local_hashing")
                self.assertEqual(selected_search.dims, 256)
                sync = selected.sync_project(project_id=project["id"])
                self.assertEqual(sync.indexed_chunks, store.index_status(project["id"])["chunk_count"])
                self.assertEqual(sync.backend, "local_hashing")
                response = PatchService(store).preview_apply_verify(
                    PatchVerifyRequest(
                        project_id=project["id"],
                        edits=[
                            {
                                "path": "auth.py",
                                "new_content": "def login_user(username, password):\n    return bool(password)\n",
                            }
                        ],
                        commands=[["python", "-c", "import auth; raise SystemExit(0 if auth.login_user('a','b') else 1)"]],
                        apply_on_accept=False,
                    )
                )
                self.assertEqual(response.verdict, "accept")
                self.assertTrue(response.rolled_back)
                self.assertIn("password == 'secret'", (workspace / "auth.py").read_text(encoding="utf-8"))

    def test_qdrant_sync_upserts_chunks_and_removes_stale_points(self) -> None:
        class Model:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class FakeClient:
            def __init__(self, **_kwargs: object) -> None:
                self.points: list[object] = []
                self.deleted: list[str] = []
                self.created = False

            def collection_exists(self, _name: str) -> bool:
                return False

            def create_collection(self, **_kwargs: object) -> None:
                self.created = True

            def create_payload_index(self, **_kwargs: object) -> None:
                return None

            def upsert(self, *, points: list[object], **_kwargs: object) -> None:
                self.points.extend(points)

            def scroll(self, **_kwargs: object) -> tuple[list[object], None]:
                return [Model(id="00000000-0000-0000-0000-000000000001")], None

            def delete(self, *, points_selector: object, **_kwargs: object) -> None:
                self.deleted.extend(points_selector.points)

        fake_client = FakeClient()
        qdrant_module = types.ModuleType("qdrant_client")
        models_module = types.ModuleType("qdrant_client.models")
        models_module.Distance = Model(COSINE="cosine")
        models_module.PayloadSchemaType = Model(KEYWORD="keyword")
        for name in (
            "VectorParams",
            "PointStruct",
            "Filter",
            "FieldCondition",
            "MatchValue",
            "PointIdsList",
        ):
            setattr(models_module, name, Model)
        qdrant_module.models = models_module
        qdrant_module.QdrantClient = lambda **_kwargs: fake_client

        class FakeEmbeddings:
            dims = 64

            def embed(self, _text: str) -> list[float]:
                return [0.1] * self.dims

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * self.dims for _ in texts]

        with (
            tempfile.TemporaryDirectory() as workspace_dir,
            tempfile.TemporaryDirectory() as db_dir,
            patch.dict(
                sys.modules,
                {
                    "qdrant_client": qdrant_module,
                    "qdrant_client.models": models_module,
                },
            ),
        ):
            workspace = Path(workspace_dir)
            (workspace / "service.py").write_text("def health():\n    return True\n", encoding="utf-8")
            with LocalStore(Path(db_dir) / "craftly.sqlite3") as store:
                project = store.create_project(name="vector", repo_path=str(workspace))
                RepositoryIndexer(store).index_project(project_id=project["id"])
                index = QdrantVectorIndex(
                    store,
                    config=VectorBackendConfig(
                        backend="qdrant",
                        dims=64,
                        qdrant_url="http://qdrant.internal:6333",
                        embedding_url="https://embedding.internal/v1/embeddings",
                    ),
                )
                index.embedding_provider = FakeEmbeddings()
                report = index.sync_project(project_id=project["id"], batch_size=2)
                self.assertTrue(fake_client.created)
                self.assertEqual(report.indexed_chunks, store.index_status(project["id"])["chunk_count"])
                self.assertEqual(len(fake_client.points), report.indexed_chunks)
                for point in fake_client.points:
                    self.assertNotIn("content", point.payload)
                    self.assertEqual(point.payload["organization_id"], "local")
                    self.assertEqual(point.payload["revision_id"], report.revision_id)
                self.assertEqual(report.deleted_stale_chunks, 1)
                self.assertEqual(fake_client.deleted, ["00000000-0000-0000-0000-000000000001"])

    def test_repository_patch_loop_indexes_context_verifies_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "auth.py").write_text(
                "def login_user(username, password):\n    return bool(username) and password == 'secret'\n",
                encoding="utf-8",
            )
            report = run_repository_patch_loop(
                RepositoryPatchLoopRequest(
                    repo_path=str(workspace),
                    goal="fix authentication validation",
                    edits=[
                        {
                            "path": "auth.py",
                            "new_content": "def login_user(username, password):\n    return bool(username) and bool(password)\n",
                        }
                    ],
                    commands=[["python", "-c", "import auth; raise SystemExit(0 if auth.login_user('a','b') else 1)"]],
                    store_path=str(Path(db_dir) / "loop.sqlite3"),
                    apply_on_accept=False,
                )
            )
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.verdict, "accept")
            self.assertTrue(report.pre_patch_context["chunks"])
            self.assertTrue(report.post_patch_context["chunks"])
            self.assertIn("password == 'secret'", (workspace / "auth.py").read_text(encoding="utf-8"))

    def test_vector_capabilities_and_unified_memory_manager(self) -> None:
        capabilities = vector_capabilities()
        self.assertTrue(any(item.backend == "local_hashing" and item.available for item in capabilities))
        with tempfile.TemporaryDirectory() as db_dir:
            with LocalStore(Path(db_dir) / "memory.sqlite3") as store:
                project = store.create_project(name="memory-demo", repo_path=str(Path(db_dir)))
                manager = UnifiedMemoryManager(project_id=project["id"], store=store)
                manager.ingest(
                    MemoryIngestRequest(
                        layer="project",
                        kind="project_fact",
                        content={"module_name": "auth", "path": "auth.py", "tech_stack": "python"},
                        relevance=0.8,
                        success_rate=0.95,
                    )
                )
                manager.remember_verified_fix("empty password accepted", "reject empty password", "auth login")
                report = manager.retrieve_report("fix auth empty password", limit=2)
                self.assertTrue(report.hits)
                self.assertGreaterEqual(report.hits[0].final_score, report.hits[-1].final_score)
                with self.assertRaises(ValueError):
                    manager.ingest(
                        MemoryIngestRequest(
                            layer="semantic",
                            kind="raw_thought",
                            content={"thought": "maybe this works"},
                        )
                    )

    def test_benchmark_harness_and_gateway_metrics_contract(self) -> None:
        report = BenchmarkHarness().run_static(built_in_security_tasks())
        self.assertEqual(report.status, "passed")
        self.assertGreaterEqual(report.total, 10)
        self.assertEqual(report.score, 1.0)
        client = TestClient(gateway_api.app)
        health = client.get("/health/ready")
        self.assertEqual(health.status_code, 200)
        metrics = client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("craftly_http_requests_total", metrics.text)
        self.assertNotIn("}{", metrics.text)

    def test_sandbox_policy_is_hardened_by_default(self) -> None:
        request = SandboxRunRequest(command=["python3", "-c", "print(1)"])
        policy = request.policy
        self.assertEqual(policy.network_mode, "none")
        self.assertEqual(policy.mem_limit, "512m")
        self.assertTrue(policy.read_only)
        self.assertIn("ALL", policy.cap_drop)
        self.assertIn("/tmp", policy.tmpfs)

    def test_token_issue_is_blocked_when_auth_enabled_without_explicit_permission(self) -> None:
        config = AuthConfig(enabled=True, jwt_secret="x" * 32, allow_token_issue=False)
        with self.assertRaises(AuthError):
            validate_token_issue_request(config, None)
        allowed = AuthConfig(enabled=True, jwt_secret="x" * 32, allow_token_issue=True, token_issue_key="issue-secret")
        with self.assertRaises(AuthError):
            validate_token_issue_request(allowed, "wrong")
        validate_token_issue_request(allowed, "issue-secret")

    def test_async_local_quota_store_matches_regenerative_contract(self) -> None:
        async def run_case() -> tuple[bool, float]:
            return await AsyncLocalQuotaStore().consume("async-u1", now=200.0, rate=1.0, capacity=2.0, cost=1.0)

        import asyncio

        allowed, remaining = asyncio.run(run_case())
        self.assertTrue(allowed)
        self.assertAlmostEqual(remaining, 1.0)

    def test_data_platform_cluster_manifests_and_capacity_plan_exist(self) -> None:
        postgres = Path("deploy/k8s/postgres-redis.yaml").read_text(encoding="utf-8")
        minio = Path("deploy/k8s/minio.yaml").read_text(encoding="utf-8")
        worker = Path("deploy/k8s/data-worker.yaml").read_text(encoding="utf-8")
        hpa = Path("deploy/k8s/data-worker-hpa.yaml").read_text(encoding="utf-8")
        network = Path("deploy/k8s/data-network-policy.yaml").read_text(encoding="utf-8")
        compose = Path("deploy/prod/docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("kind: StatefulSet", postgres)
        self.assertIn("volumeClaimTemplates", postgres)
        self.assertIn("craftly-minio", minio)
        self.assertIn("craftly-datasets", minio)
        self.assertIn("craftly-data-worker", worker)
        self.assertIn("HorizontalPodAutoscaler", hpa)
        self.assertIn("NetworkPolicy", network)
        self.assertIn("qdrant:", compose)
        self.assertIn("CRAFTLY_QDRANT_URL", compose)
        self.assertIn("CRAFTLY_OBJECT_STORE_URI", compose)
        plan = build_capacity_plan(CapacityPlanConfig(target_documents=1_000_000, worker_replicas=8, async_workers_per_replica=16))
        self.assertGreater(plan.raw_storage_tb, 0.0)
        self.assertGreater(plan.recommended_worker_replicas_for_target_days, 0)


if __name__ == "__main__":
    unittest.main()

