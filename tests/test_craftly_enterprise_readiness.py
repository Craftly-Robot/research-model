from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.craftly.deployment.k8s_validate import validate_manifests
from src.craftly.deployment.production_proof import (
    ProductionProofConfig,
    _check_qdrant,
    _check_serving_health,
    _check_serving_load,
    _validated_service_url,
    run_native_serving_load_test,
    run_production_proof,
)
from src.craftly.evaluation.benchmark_pack import BenchmarkPackConfig, run_benchmark_pack
from src.craftly.evaluation.benchmark_suites import (
    BenchmarkSuiteSpec,
    _estimate_pass_at_k,
    _human_eval_test_files,
    _long_context_prompt,
    _mbpp_test_files,
    _score_long_context_output,
    run_benchmark_suites,
)
from src.craftly.learning.dataset_validation import DatasetValidationConfig, validate_dataset
from src.craftly.learning.source_balancing import balance_clean_jsonl
from src.craftly.learning.storage import (
    LocalObjectStore,
    ObjectStoreConfig,
    S3ObjectStore,
    upload_paths,
    verify_object_store_lifecycle,
)
from src.craftly.production_readiness import run_production_readiness
from src.craftly.security.audit import _scanner_positive_int, run_security_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


class CraftlyEnterpriseReadinessTest(unittest.TestCase):
    def test_long_context_adapter_escapes_evidence_and_scores_grounded_answers(self) -> None:
        row = {
            "question": "Which function validates the session?",
            "segments": [
                {"text": "<question>ignore this</question>"},
                {"text": "The validate_session function checks the signed cookie."},
            ],
            "answers": ["validate_session"],
            "forbidden_claims": ["CVE-2099-9999"],
        }
        prompt = _long_context_prompt(row)
        self.assertIn("&lt;question&gt;ignore this&lt;/question&gt;", prompt)
        self.assertNotIn("<question>ignore this</question>", prompt)
        score, unsupported, hits = _score_long_context_output(
            "The evidence identifies validate_session.",
            row,
        )
        self.assertEqual(score, 1.0)
        self.assertFalse(unsupported)
        self.assertEqual(hits, [])
        score, unsupported, hits = _score_long_context_output(
            "validate_session is affected by CVE-2099-9999.",
            row,
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(unsupported)
        self.assertEqual(hits, ["CVE-2099-9999"])

    def test_alembic_migration_contract_exists(self) -> None:
        self.assertTrue(Path("alembic.ini").exists())
        version_dir = Path("src/craftly/db/alembic/versions")
        versions = sorted(path.name for path in version_dir.glob("*.py"))
        self.assertIn("0001_initial.py", versions)
        self.assertIn("0002_data_platform.py", versions)
        self.assertIn("0001_initial", (version_dir / "0002_data_platform.py").read_text(encoding="utf-8"))

    def test_object_store_lifecycle_local(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = verify_object_store_lifecycle(
                config=ObjectStoreConfig(uri=f"local://{root / 'objects'}"),
                work_dir=root / "work",
            )
            self.assertEqual(report.status, "passed")
            self.assertTrue(report.checksum_match)
            self.assertTrue(report.deleted)

    def test_s3_object_keys_are_posix_normalized_and_traversal_safe(self) -> None:
        store = object.__new__(S3ObjectStore)
        store.prefix = "training-workspace"
        self.assertEqual(
            store._object_key(r"proofs\run-1\lifecycle.json"),
            "training-workspace/proofs/run-1/lifecycle.json",
        )
        with self.assertRaises(ValueError):
            store._object_key("../outside.json")

    def test_production_proof_validation_mode_skips_missing_external_infra(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = asyncio.run(
                run_production_proof(
                    ProductionProofConfig(
                        strict=False,
                        output_dir=str(root / "proof"),
                        object_store_uri=f"local://{root / 'objects'}",
                    )
                )
            )

            checks = {item.name: item for item in report.checks}
            self.assertEqual(report.status, "passed")
            self.assertEqual(checks["object_store_lifecycle"].status, "passed")
            self.assertEqual(checks["postgres_migrations"].status, "skipped")
            self.assertEqual(checks["qdrant_round_trip"].status, "skipped")
            self.assertTrue((root / "proof" / "production_proof_report.json").exists())

    def test_production_proof_strict_mode_fails_without_required_infra(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = asyncio.run(
                run_production_proof(
                    ProductionProofConfig(
                        strict=True,
                        output_dir=str(Path(temp_dir) / "proof"),
                    )
                )
            )

            self.assertEqual(report.status, "failed")
            self.assertTrue(any(item.status == "failed" and item.required for item in report.checks))

    def test_native_serving_load_test_scores_mocked_openai_compatible_endpoint(self) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "model": "craftly-scratch",
                    "choices": [{"message": {"content": "ok"}}],
                    "craftly": {"scratch_only": True},
                }

        class FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                return None

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def post(self, *args: object, **kwargs: object) -> FakeResponse:
                return FakeResponse()

        with patch("src.craftly.deployment.production_proof.httpx.AsyncClient", FakeClient):
            report = asyncio.run(
                run_native_serving_load_test(
                    endpoint="http://serving.test",
                    model="craftly-scratch",
                    api_key="token",
                    requests=5,
                    concurrency=2,
                    timeout_seconds=5.0,
                )
            )

        self.assertEqual(report.status, "passed")
        self.assertEqual(report.passed, 5)
        self.assertEqual(report.failed, 0)

    def test_qdrant_proof_is_transactional_and_cleans_up(self) -> None:
        state: dict[str, object] = {"deleted": False}

        class FakeResponse:
            def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
                self.payload = payload or {"result": True}
                self.status_code = status_code

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self) -> dict[str, object]:
                return self.payload

        class FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                return None

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def put(self, url: str, **kwargs: object) -> FakeResponse:
                payload = kwargs.get("json")
                if isinstance(payload, dict) and "points" in payload:
                    point = payload["points"][0]
                    state["point"] = point
                return FakeResponse()

            async def post(self, url: str, **kwargs: object) -> FakeResponse:
                point = state["point"]
                request = kwargs.get("json")
                if isinstance(request, dict):
                    filters = request.get("filter", {}).get("must", [])  # type: ignore[union-attr]
                    expected_organization = next(
                        (
                            item.get("match", {}).get("value")
                            for item in filters
                            if item.get("key") == "organization_id"
                        ),
                        None,
                    )
                    if expected_organization != point["payload"]["organization_id"]:
                        return FakeResponse({"result": {"points": []}})
                return FakeResponse({"result": {"points": [point]}})

            async def delete(self, url: str, **kwargs: object) -> FakeResponse:
                state["deleted"] = True
                return FakeResponse(status_code=int(state.get("delete_status", 200)))

        with patch("src.craftly.deployment.production_proof.httpx.AsyncClient", FakeClient):
            result = asyncio.run(
                _check_qdrant(
                    ProductionProofConfig(
                        qdrant_url="http://qdrant.internal:6333",
                        allowed_insecure_service_hosts=["qdrant.internal"],
                    )
                )
            )
        self.assertEqual(result.status, "passed")
        self.assertTrue(state["deleted"])
        self.assertTrue(result.details["query_verified"])
        state["delete_status"] = 500
        with patch("src.craftly.deployment.production_proof.httpx.AsyncClient", FakeClient):
            cleanup_failed = asyncio.run(
                _check_qdrant(
                    ProductionProofConfig(
                        qdrant_url="http://qdrant.internal:6333",
                        allowed_insecure_service_hosts=["qdrant.internal"],
                    )
                )
            )
        self.assertEqual(cleanup_failed.status, "failed")
        self.assertIn("cleanup returned HTTP 500", cleanup_failed.error)

    def test_native_serving_load_test_validates_sse_completion(self) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "model": "craftly-scratch",
                    "choices": [{"message": {"content": "safe output"}}],
                    "craftly": {"scratch_only": True},
                }

        class FakeStream(FakeResponse):
            async def __aenter__(self) -> "FakeStream":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def aiter_lines(self):
                yield (
                    'data: {"model":"craftly-scratch","choices":'
                    '[{"delta":{"content":"safe "}}]}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                return None

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def post(self, *args: object, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def stream(self, *args: object, **kwargs: object) -> FakeStream:
                return FakeStream()

        with patch("src.craftly.deployment.production_proof.httpx.AsyncClient", FakeClient):
            report = asyncio.run(
                run_native_serving_load_test(
                    endpoint="http://127.0.0.1:8001",
                    model="craftly-scratch",
                    api_key="token",
                    requests=3,
                    streaming_requests=2,
                    concurrency=2,
                    timeout_seconds=5.0,
                )
            )
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.streaming_passed, 2)
        self.assertEqual(report.content_validation_failures, 0)

    def test_production_service_urls_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            _validated_service_url(
                "http://qdrant.internal:6333",
                label="Qdrant",
                allowed_insecure_hosts=[],
            )

    def test_malformed_qdrant_url_is_reported_not_raised(self) -> None:
        result = asyncio.run(
            _check_qdrant(
                ProductionProofConfig(
                    strict=True,
                    qdrant_url="http://user:secret@qdrant.internal:6333",
                    allowed_insecure_service_hosts=["qdrant.internal"],
                )
            )
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("embedded credentials", result.error)

    def test_strict_serving_load_requires_streaming_proof(self) -> None:
        result = asyncio.run(
            _check_serving_load(
                ProductionProofConfig(
                    strict=True,
                    serving_url="http://127.0.0.1:8001",
                    serving_api_key="test-token",
                    load_test_requests=2,
                    load_test_streaming_requests=0,
                )
            )
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("streaming SSE request", result.error)

    def test_strict_serving_health_matches_active_profile_hashes(self) -> None:
        checkpoint_hash = "a" * 64
        tokenizer_hash = "b" * 64

        class FakeResponse:
            status_code = 200

            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return self.payload

        class FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                return None

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def get(self, url: str, **kwargs: object) -> FakeResponse:
                if url.endswith("/health/ready"):
                    return FakeResponse(
                        {
                            "status": "ready",
                            "model_name": "craftly-scratch",
                            "checkpoint_manifest": "/models/checkpoint.json",
                            "checkpoint_manifest_sha256": checkpoint_hash,
                            "tokenizer_sha256": tokenizer_hash,
                            "scratch_only": True,
                        }
                    )
                return FakeResponse({"data": [{"id": "craftly-scratch"}]})

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = root / "active.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "profile": {
                            "name": "craftly-production",
                            "kind": "local",
                            "family": "craftly-scratch",
                            "size_class": "test",
                            "backend": "craftly_serving",
                            "model_name": "craftly-scratch",
                            "endpoint": "http://127.0.0.1:8001/v1",
                            "checkpoint_manifest": "/models/checkpoint.json",
                            "tokenizer_path": "/models/tokenizer.json",
                            "scratch_only": True,
                            "evidence": {
                                "checkpoint_manifest_sha256": checkpoint_hash,
                                "tokenizer_sha256": tokenizer_hash,
                                "evaluation_report_sha256": "c" * 64,
                                "scorecard_report_sha256": "d" * 64,
                            },
                        },
                        "env": {},
                        "run_id": "test-production-profile",
                        "production_blockers": [],
                    }
                ),
                encoding="utf-8",
            )
            config = ProductionProofConfig(
                strict=True,
                serving_url="http://127.0.0.1:8001",
                serving_api_key="test-token",
                active_model_profile=str(profile),
            )
            with patch(
                "src.craftly.deployment.production_proof.httpx.AsyncClient",
                FakeClient,
            ):
                result = asyncio.run(_check_serving_health(config))
            self.assertEqual(result.status, "passed")
            self.assertTrue(result.details["checkpoint_hash_verified"])

            payload = json.loads(profile.read_text(encoding="utf-8"))
            payload["profile"]["evidence"]["checkpoint_manifest_sha256"] = "0" * 64
            profile.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "src.craftly.deployment.production_proof.httpx.AsyncClient",
                FakeClient,
            ):
                failed = asyncio.run(_check_serving_health(config))
            self.assertEqual(failed.status, "failed")
            self.assertIn("live serving hashes", failed.error)
        with self.assertRaisesRegex(ValueError, "embedded credentials"):
            _validated_service_url(
                "https://user:secret@qdrant.example",
                label="Qdrant",
                allowed_insecure_hosts=[],
            )

    def test_object_store_upload_paths_keeps_duplicate_shard_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "shards" / "train" / "shard-000000.bin"
            val = root / "shards" / "val" / "shard-000000.bin"
            train.parent.mkdir(parents=True)
            val.parent.mkdir(parents=True)
            train.write_bytes(b"train")
            val.write_bytes(b"validation")

            objects = upload_paths(
                LocalObjectStore(root / "objects"),
                [train, val],
                prefix="runs/example/shards",
            )

            self.assertEqual(len(objects), 2)
            self.assertEqual(len({item.uri for item in objects}), 2)
            self.assertTrue(all(Path(item.uri).exists() for item in objects))
            self.assertIn("train", Path(objects[0].uri).read_text(encoding="utf-8"))
            self.assertIn("validation", Path(objects[1].uri).read_text(encoding="utf-8"))

    def test_dataset_validation_passes_and_blocks_bad_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = write_jsonl(
                root / "good.jsonl",
                [
                    {
                        "text": "General software architecture documentation with enough detail for language modeling.",
                        "license": "mit",
                        "category": "general",
                        "quality": {"labels": ["documentation"]},
                    },
                    {
                        "text": "def secure_query(cursor, value): return cursor.execute('select * from t where id=?', [value])",
                        "license": "apache-2.0",
                        "category": "code",
                        "quality": {"labels": ["code"]},
                    },
                    {
                        "text": "Defensive security analysis: SQL injection should be fixed with parameterized queries and tests.",
                        "license": "cc-by-4.0",
                        "category": "cybersecurity",
                        "quality": {"labels": ["defensive_security"]},
                    },
                ],
            )
            report = validate_dataset(
                DatasetValidationConfig(
                    input_paths=[str(good)],
                    min_records=3,
                    min_avg_chars=20,
                    max_duplicate_fraction=0.50,
                )
            )
            self.assertEqual(report.status, "passed")

            bad = write_jsonl(
                root / "bad.jsonl",
                [
                    {"text": "dup", "category": "general"},
                    {"text": "dup", "category": "general"},
                ],
            )
            bad_report = validate_dataset(
                DatasetValidationConfig(
                    input_paths=[str(bad)],
                    min_records=3,
                    max_duplicate_fraction=0.10,
                )
            )
            self.assertEqual(bad_report.status, "failed")

    def test_source_balancing_report_never_claims_more_rows_than_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [
                {
                    "text": f"dominant source record {index}",
                    "source": "dominant",
                    "quality": {"quality_score": 0.6},
                }
                for index in range(10)
            ]
            rows.append({"text": "tiny source record", "source": "tiny", "quality": {"quality_score": 0.9}})
            corpus = write_jsonl(root / "clean.jsonl", rows)

            report = balance_clean_jsonl(
                input_paths=[corpus],
                output_path=root / "balanced.jsonl",
                max_source_fraction=0.30,
                min_source_rows=25,
            )

            self.assertEqual(report.input_rows, 11)
            self.assertLessEqual(report.output_rows, report.input_rows)
            for item in report.sources:
                self.assertLessEqual(item.output_rows, item.input_rows)

    def test_k8s_manifests_validate_without_blocking_failures(self) -> None:
        report = validate_manifests(sorted(Path("deploy/k8s").glob("*.yaml")))
        self.assertEqual(report.status, "passed")
        self.assertGreaterEqual(report.resources.get("Deployment", 0), 1)
        self.assertGreaterEqual(report.resources.get("NetworkPolicy", 0), 1)

    def test_benchmark_suite_adapters_execute_local_holdout_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            swe = write_jsonl(
                root / "swe.jsonl",
                [
                    {
                        "instance_id": "swe-1",
                        "problem_statement": "Fix auth validation",
                        "patch": "+ def test_auth():\n+     assert True",
                        "expected_terms": ["test_auth", "assert"],
                    }
                ],
            )
            cyber = write_jsonl(
                root / "cyber.jsonl",
                [
                    {
                        "id": "sec-1",
                        "code": "cursor.execute('select * from users where name=' + user_input)",
                        "expected_findings": ["user_input"],
                    }
                ],
            )
            report = run_benchmark_suites(
                [
                    BenchmarkSuiteSpec(name="swe", kind="swe_bench_style", path=str(swe)),
                    BenchmarkSuiteSpec(name="cyber", kind="cyberseceval_style", path=str(cyber)),
                ]
            )
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.aggregate_score, 1.0)
            self.assertEqual(report.evaluation_mode, "dataset_validation")
            self.assertTrue(
                all("not a model capability score" in suite.reason for suite in report.suites)
            )

    def test_executable_code_benchmark_contract_builds_safe_tests_and_pass_at_k(self) -> None:
        human_files = _human_eval_test_files(
            {
                "prompt": "def add(a, b):\n    \"\"\"Return a plus b.\"\"\"\n",
                "entry_point": "add",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
            },
            "    return a + b",
        )
        self.assertIn("def add", human_files["candidate.py"])
        self.assertIn("check(candidate)", human_files["runner.py"])
        mbpp_files = _mbpp_test_files(
            {"test_list": ["assert reverse('ab') == 'ba'"]},
            "def reverse(value):\n    return value[::-1]",
        )
        self.assertIn("from candidate import *", mbpp_files["runner.py"])
        self.assertAlmostEqual(_estimate_pass_at_k(10, 3, 1), 0.3)
        self.assertAlmostEqual(
            _estimate_pass_at_k(10, 3, 5),
            1.0 - (21 / 252),
        )
        with self.assertRaisesRegex(ValueError, "entry point"):
            _human_eval_test_files(
                {
                    "prompt": "def bad(): pass",
                    "entry_point": "bad;import os",
                    "test": "def check(candidate): pass",
                },
                "pass",
            )

    def test_benchmark_pack_runs_required_and_optional_local_suites(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            human = write_jsonl(
                root / "human.jsonl",
                [{"task_id": "h1", "prompt": "Write add", "canonical_solution": "def add(a,b): return a+b", "expected_terms": ["def"]}],
            )
            mbpp = write_jsonl(
                root / "mbpp.jsonl",
                [{"task_id": "m1", "prompt": "Reverse string", "code": "def reverse(s): return s[::-1]", "expected_terms": ["return"]}],
            )
            security = write_jsonl(
                root / "security.jsonl",
                [{"id": "s1", "code": "cursor.execute('select * from t where x=' + x)", "expected_findings": ["cursor.execute"]}],
            )
            report = run_benchmark_pack(
                BenchmarkPackConfig(
                    human_eval_path=str(human),
                    mbpp_path=str(mbpp),
                    cyberseceval_path=str(security),
                    strict=True,
                ),
                output_dir=root / "bench",
            )
            self.assertEqual(report.status, "passed")
            self.assertIn("humaneval", report.required_suites)
            self.assertTrue((root / "bench" / "benchmark_pack_report.json").exists())

            production_report = run_benchmark_pack(
                BenchmarkPackConfig(
                    human_eval_path=str(human),
                    mbpp_path=str(mbpp),
                    cyberseceval_path=str(security),
                    strict=True,
                    production=True,
                ),
                output_dir=root / "bench-production",
            )
            self.assertEqual(production_report.status, "failed")
            self.assertTrue(production_report.recommendations)

    def test_security_audit_detects_secret_and_can_pass_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean = root / "src" / "craftly"
            clean.mkdir(parents=True)
            (clean / "app.py").write_text("def ok():\n    return True\n", encoding="utf-8")
            report = run_security_audit(root=root, run_bandit=False, validate_k8s=False, run_semgrep=False, run_codeql=False, run_pip_audit=False)
            self.assertEqual(report.status, "passed")
            self.assertIn("python_tools", report.scanner_install_plan)
            self.assertIn("strict_audit_command", report.scanner_install_plan)

            (clean / "bad.py").write_text("API_KEY = 'abcdefghijklmnopqrstuvwxyz123456'\n", encoding="utf-8")
            bad = run_security_audit(root=root, run_bandit=False, validate_k8s=False, run_semgrep=False, run_codeql=False, run_pip_audit=False)
            self.assertEqual(bad.status, "failed")
            self.assertTrue(any(item.check == "secret_pattern" for item in bad.findings))

    def test_security_scanner_resource_limits_are_validated(self) -> None:
        with patch.dict("os.environ", {"CRAFTLY_CODEQL_THREADS": "3"}, clear=False):
            self.assertEqual(
                _scanner_positive_int("codeql", "threads", 2, minimum=1, maximum=64),
                3,
            )
        with patch.dict("os.environ", {"CRAFTLY_CODEQL_THREADS": "0"}, clear=False):
            with self.assertRaisesRegex(ValueError, "between 1 and 64"):
                _scanner_positive_int("codeql", "threads", 2, minimum=1, maximum=64)
        with patch.dict("os.environ", {"CRAFTLY_CODEQL_RAM_MB": "not-an-integer"}, clear=False):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                _scanner_positive_int("codeql", "ram_mb", 4096, minimum=1024, maximum=524_288)

    def test_production_readiness_is_honest_about_missing_external_dependencies(self) -> None:
        cleared = {
            "CRAFTLY_AUTH_ENABLED": "",
            "CRAFTLY_JWT_SECRET": "",
            "CRAFTLY_QUOTA_ENABLED": "",
            "CRAFTLY_REDIS_URL": "",
            "CRAFTLY_MODEL_BACKEND": "",
            "CRAFTLY_MODEL_ENDPOINT": "",
            "CRAFTLY_DATABASE_URL": "",
            "CRAFTLY_OBJECT_STORE_URI": "",
            "CRAFTLY_QDRANT_URL": "",
        }
        with patch.dict("os.environ", cleared, clear=False):
            report = run_production_readiness(mode="production", benchmark_dir="definitely-missing-eval-dir")
            self.assertEqual(report.status, "failed")
            statuses = {check.subsystem: check.status for check in report.checks}
            self.assertEqual(statuses["serving"], "blocked_missing_dependency")
            self.assertEqual(statuses["benchmark_eval"], "blocked_missing_dependency")
            self.assertEqual(statuses["canonical_4t_architecture"], "built_not_cluster_proven")
            architecture = next(
                check for check in report.checks if check.subsystem == "canonical_4t_architecture"
            )
            self.assertEqual(
                architecture.evidence["parameter_report"]["total"],
                3_994_840_025_600,
            )
            self.assertTrue(architecture.production_blocker)
            self.assertTrue(any(check.production_blocker for check in report.checks))

            dev_report = run_production_readiness(mode="dev", benchmark_dir="definitely-missing-eval-dir")
            self.assertEqual(dev_report.status, "passed")
            self.assertTrue(any(check.status == "blocked_missing_dependency" for check in dev_report.checks))
            dev_architecture = next(
                check for check in dev_report.checks if check.subsystem == "canonical_4t_architecture"
            )
            self.assertFalse(dev_architecture.production_blocker)


if __name__ == "__main__":
    unittest.main()

