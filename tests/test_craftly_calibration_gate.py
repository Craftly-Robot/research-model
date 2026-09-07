from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from src.craftly.evaluation.benchmark_pack import (
    materialize_public_benchmark_pack,
    materialize_protected_benchmark_pack,
    validate_protected_benchmark_manifest,
)
from src.craftly.learning.calibration_gate import (
    CalibrationDecision,
    CalibrationManifest,
    CalibrationReviewBinding,
    _legal_evidence_sha256,
    _registry_sha256,
    _sha256_file,
    finalize_calibration,
    preflight_calibration,
    run_calibration,
    validate_advancement_decision,
)
from src.craftly.learning.dataset_authority import (
    ReviewDecisionCreate,
    ReviewItemCreate,
    SQLiteDatasetAuthorityStore,
)
from src.craftly.learning.source_registry import SourceRegistry, source_registry_entry_sha256
from src.craftly.learning.web_ingest import SourceSpec


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _protected_config(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "pack_id": "test-protected-pack",
            "sources": [
                {
                    "name": "craftly_security",
                    "kind": "builtin_craftly",
                    "revision": "builtin-task-contract-v1",
                    "license": "Proprietary-Evaluation",
                    "output_file": "craftly_security.jsonl",
                    "minimum_rows": 25,
                    "maximum_bytes": 1000000,
                    "train_policy": "eval_holdout",
                }
            ],
        },
    )


def _approved_source(index: int, evidence_root: Path) -> SourceSpec:
    pending = SourceSpec(
        name=f"source-{index}",
        source_id=f"source-{index}",
        source_family=f"family-{index}",
        urls=[f"https://source{index}.example/docs"],
        allowed_domains=[f"source{index}.example"],
        license="mit",
        category="defensive_security",
        trust_tier="quarantine",
        approved_use="defensive",
        approval_status="pending",
        immutable_revision="rolling",
    )
    source_root = evidence_root / f"source-{index}"
    source_root.mkdir(parents=True, exist_ok=True)
    license_path = source_root / "license.txt"
    license_path.write_text(f"Reviewed MIT license evidence for source {index}", encoding="utf-8")
    license_hash = _sha256_file(license_path)
    request_hash = source_registry_entry_sha256(pending)
    legal_path = source_root / "approval.json"
    _write_json(
        legal_path,
        {
            "schema_version": 1,
            "approval_id": f"approval-source-{index}",
            "decision": "approved",
            "source_id": f"source-{index}",
            "registry_entry_sha256": request_hash,
            "immutable_revision": f"snapshot-{index}-20260719",
            "license": "mit",
            "license_evidence_sha256": license_hash,
            "approved_use": "defensive",
            "approved_by": "test-legal-governance",
            "approved_at": "2026-07-19T12:00:00+06:00",
            "scope": "training_collection",
            "rationale": "The source license and defensive training scope were independently reviewed and approved.",
        },
    )
    return SourceSpec.model_validate(
        pending.model_copy(
            update={
                "trust_tier": "reviewed",
                "approval_status": "approved",
                "immutable_revision": f"snapshot-{index}-20260719",
                "license_evidence_sha256": license_hash,
                "legal_approval_sha256": _sha256_file(legal_path),
                "approval_request_sha256": request_hash,
            }
        ).model_dump()
    )


def _reviewer_roster(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "roster_id": "test-reviewers-v1",
            "identities": [
                {
                    "reviewer_id": "reviewer-one",
                    "identity_provider_subject": "oidc:test:reviewer-one",
                    "roles": ["reviewer"],
                    "active": True,
                    "approved_by": "test-governance",
                    "approved_at": "2026-07-19T12:00:00+06:00",
                },
                {
                    "reviewer_id": "reviewer-two",
                    "identity_provider_subject": "oidc:test:reviewer-two",
                    "roles": ["reviewer"],
                    "active": True,
                    "approved_by": "test-governance",
                    "approved_at": "2026-07-19T12:00:00+06:00",
                },
                {
                    "reviewer_id": "review-adjudicator",
                    "identity_provider_subject": "oidc:test:adjudicator",
                    "roles": ["adjudicator"],
                    "active": True,
                    "approved_by": "test-governance",
                    "approved_at": "2026-07-19T12:00:00+06:00",
                },
            ],
        },
    )


def _reviewer_qualification(path: Path, roster_path: Path) -> Path:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    reviewer_ids = [
        identity["reviewer_id"]
        for identity in roster["identities"]
        if identity["active"] and "reviewer" in identity["roles"]
    ]
    return _write_json(
        path,
        {
            "schema_version": 1,
            "status": "passed",
            "roster_id": roster["roster_id"],
            "rubric_id": "craftly-review-rubric-v1",
            "roster_sha256": _sha256_file(roster_path),
            "rubric_sha256": "a" * 64,
            "qualification_pack_sha256": "b" * 64,
            "answer_key_sha256": "c" * 64,
            "minimum_accuracy": 0.95,
            "minimum_kappa": 0.80,
            "reviewer_agreement_kappa": 1.0,
            "reviewers": [
                {
                    "reviewer_id": reviewer_id,
                    "response_sha256": f"{index + 1:064x}",
                    "item_count": 20,
                    "correct_count": 20,
                    "accuracy": 1.0,
                    "passed": True,
                }
                for index, reviewer_id in enumerate(reviewer_ids)
            ],
            "blockers": [],
            "handling_warning": "Synthetic unit-test evidence only; never use this fixture for production governance.",
        },
    )


class ProtectedBenchmarkGovernanceTest(unittest.TestCase):
    def test_public_pack_is_revision_hash_and_holdout_bound(self) -> None:
        human_rows = [{"task_id": f"HumanEval/{index}", "prompt": "pass"} for index in range(164)]
        mbpp_rows = [{"task_id": index, "text": "write code"} for index in range(374)]

        def download(url: str, *, max_bytes: int = 20_000_000) -> bytes:
            self.assertGreater(max_bytes, 0)
            if "human-eval" in url:
                self.assertIn("6d43fb980f9fee3c892a914eda09951f772ad10d", url)
                payload = "\n".join(json.dumps(row) for row in human_rows).encode("utf-8")
                return gzip.compress(payload)
            self.assertIn("95e3a1da2d27cb9c8289f6fd3076cfed608c3c94", url)
            return "\n".join(json.dumps(row) for row in mbpp_rows).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.craftly.evaluation.benchmark_pack._download_bytes",
            side_effect=download,
        ):
            report = materialize_public_benchmark_pack(temp_dir)
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.rows, {"humaneval": 164, "mbpp": 374})
            self.assertEqual(set(report.sha256), {"humaneval", "mbpp"})
            self.assertTrue(all(len(digest) == 64 for digest in report.sha256.values()))
            self.assertEqual(
                report.train_policy,
                {"humaneval": "eval_holdout", "mbpp": "eval_holdout"},
            )

    def test_materialized_pack_is_hash_bound_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _protected_config(root / "protected.json")
            output = root / "pack"
            manifest = materialize_protected_benchmark_pack(config, output)
            manifest_path = output / "protected_benchmark_manifest.json"
            validated = validate_protected_benchmark_manifest(config, manifest_path)
            self.assertEqual(validated.pack_id, manifest.pack_id)
            artifact = output / manifest.artifacts[0].path
            artifact.write_text(artifact.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_protected_benchmark_manifest(config, manifest_path)

    def test_preflight_blocks_pending_legal_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _protected_config(root / "protected.json")
            pack = root / "pack"
            materialize_protected_benchmark_pack(config, pack)
            sources = _write_json(
                root / "sources.json",
                {
                    "sources": [
                        {
                            "name": "pending-source",
                            "source_id": "pending-source",
                            "source_family": "pending-family",
                            "urls": ["https://example.com/docs"],
                            "allowed_domains": ["example.com"],
                            "license": "mit",
                            "approval_status": "pending",
                            "immutable_revision": "rolling",
                        }
                    ]
                },
            )
            report = preflight_calibration(
                sources_path=sources,
                protected_config_path=config,
                protected_manifest_path=pack / "protected_benchmark_manifest.json",
                reviewer_roster_path=_reviewer_roster(root / "reviewers.json"),
                reviewer_qualification_report_path=None,
                legal_evidence_dir=root / "legal-evidence",
                approval_request_dir=root / "requests",
            )
            self.assertEqual(report.status, "blocked")
            self.assertTrue(any("source_legal_approval" in blocker for blocker in report.blockers))
            self.assertTrue((root / "requests" / "pending-source.approval-request.json").is_file())


class CalibrationFinalizationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protected_config = _protected_config(self.root / "protected.json")
        self.protected_dir = self.root / "protected-pack"
        materialize_protected_benchmark_pack(self.protected_config, self.protected_dir)
        self.protected_manifest = self.protected_dir / "protected_benchmark_manifest.json"
        self.policy = Path("config/dataset_trust_policy.json").resolve()
        self.reviewers = _reviewer_roster(self.root / "reviewers.json")
        self.reviewer_qualification = _reviewer_qualification(
            self.root / "reviewer-qualification.json",
            self.reviewers,
        )
        self.legal_evidence = self.root / "legal-evidence"
        self.sources = [_approved_source(index, self.legal_evidence) for index in range(5)]
        self.registry = SourceRegistry(self.sources)
        self.sources_path = _write_json(
            self.root / "sources.json",
            {"sources": [source.model_dump(mode="json") for source in self.sources]},
        )
        self.authority_path = self.root / "authority.sqlite3"
        self.authority = SQLiteDatasetAuthorityStore(self.authority_path)

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def _manifest(
        self,
        *,
        stage: str = "calibration_200",
        prior_decision: Path | None = None,
    ) -> tuple[Path, list[CalibrationReviewBinding]]:
        bindings: list[CalibrationReviewBinding] = []
        row_count = 200 if stage == "calibration_200" else 5_000
        binding_count = 11 if stage == "calibration_200" else 155
        hash_base = 201 if stage == "calibration_200" else 20_001
        for index in range(binding_count):
            source = self.sources[index % len(self.sources)]
            content_hash = f"{index + hash_base:064x}"
            snapshot_hash = f"{index + hash_base + 1_000:064x}"
            high_value = index < 5
            item = await self.authority.enqueue(
                ReviewItemCreate(
                    content_hash=content_hash,
                    source_snapshot_sha256=snapshot_hash,
                    source_id=source.source_id or "",
                    data_type="security_reference" if high_value else "documentation",
                    high_value=True,
                    payload={"text": f"review payload {index} with enough evidence"},
                )
            )
            bindings.append(
                CalibrationReviewBinding(
                    review_item_id=item.review_item_id,
                    content_hash=content_hash,
                    source_id=source.source_id or "",
                    source_snapshot_sha256=snapshot_hash,
                    data_type="security_reference" if high_value else "documentation",
                    actual_high_value=high_value,
                )
            )
        rows = self.root / f"{stage}.jsonl"
        with rows.open("w", encoding="utf-8") as handle:
            for index in range(row_count):
                source = self.sources[index % len(self.sources)]
                content_hash = (
                    f"{index + hash_base:064x}"
                    if index < len(bindings)
                    else f"{index + hash_base + 100_000:064x}"
                )
                handle.write(
                    json.dumps(
                        {
                            "source": source.name,
                            "content_hash": content_hash,
                            "text": f"governed calibration row {index} with deterministic evidence",
                            "quality": {
                                "data_type": "security_reference" if index < 5 else "documentation",
                                "score": 0.9,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        artifacts = {"calibration_rows": str(rows)}
        manifest = CalibrationManifest(
            calibration_id=f"{stage}-test-001",
            stage=stage,
            status="awaiting_review",
            target_records=200 if stage == "calibration_200" else 5_000,
            final_records=200 if stage == "calibration_200" else 5_000,
            prior_decision_path=str(prior_decision.resolve()) if prior_decision else None,
            prior_decision_sha256=_sha256_file(prior_decision) if prior_decision else None,
            source_registry_sha256=_registry_sha256(self.registry),
            trust_policy_sha256=_sha256_file(self.policy),
            reviewer_roster_sha256=_sha256_file(self.reviewers),
            reviewer_qualification_path=str(self.reviewer_qualification),
            reviewer_qualification_sha256=_sha256_file(self.reviewer_qualification),
            legal_evidence_sha256=_legal_evidence_sha256(self.legal_evidence),
            protected_manifest_sha256=_sha256_file(self.protected_manifest),
            authority_database=str(self.authority_path),
            artifacts=artifacts,
            artifact_sha256={"calibration_rows": _sha256_file(rows)},
            source_fractions={source.name: 0.2 for source in self.sources},
            review_bindings=bindings,
            crawl_report={"status": "complete"},
            license_report={"accepted": 5, "rejected": 0},
            contamination_report={"accepted": 5, "rejected": 0},
            dedup_report={"accepted": 5},
            quality_report={"avg_quality_score": 0.9},
        )
        path = self.root / f"{stage}_manifest.json"
        _write_json(path, manifest.model_dump(mode="json"))
        return path, bindings

    async def _approve(self, bindings: list[CalibrationReviewBinding]) -> None:
        for binding in bindings:
            for reviewer in ("reviewer-one", "reviewer-two"):
                await self.authority.claim(binding.review_item_id, reviewer, 3600)
                await self.authority.decide(
                    binding.review_item_id,
                    reviewer,
                    ReviewDecisionCreate(
                        decision="approve",
                        rationale="The record is licensed, relevant, accurate, and contains sufficient verifiable evidence.",
                        content_hash=binding.content_hash,
                        source_snapshot_sha256=binding.source_snapshot_sha256,
                        evidence={"checklist_version": 1},
                    ),
                )

    async def test_preflight_requires_passed_reviewer_qualification(self) -> None:
        blocked = preflight_calibration(
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            reviewer_qualification_report_path=None,
            legal_evidence_dir=self.legal_evidence,
            approval_request_dir=self.root / "requests-blocked",
        )
        self.assertEqual(blocked.status, "blocked")
        self.assertTrue(any("reviewer_qualification" in issue for issue in blocked.blockers))
        ready = preflight_calibration(
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            reviewer_qualification_report_path=self.reviewer_qualification,
            legal_evidence_dir=self.legal_evidence,
            approval_request_dir=self.root / "requests-ready",
        )
        self.assertEqual(ready.status, "ready")
        self.assertEqual(ready.reviewer_qualification["reviewer_count"], 2)

    async def test_finalize_remains_blocked_before_independent_reviews(self) -> None:
        manifest, _ = await self._manifest()
        decision = await finalize_calibration(
            manifest,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        self.assertEqual(decision.status, "blocked")
        self.assertFalse(decision.checks["all_sample_records_have_two_reviews"])
        decision_path = manifest.parent / "calibration_decision.json"
        forged = json.loads(decision_path.read_text(encoding="utf-8"))
        forged["status"] = "passed"
        forged["checks"] = {name: True for name in forged["checks"]}
        forged["next_stage"] = "5k_calibration_allowed"
        _write_json(decision_path, forged)
        with self.assertRaisesRegex(ValueError, "evidence replay failed"):
            validate_advancement_decision(
                decision_path,
                expected_stage="calibration_200",
                expected_next_stage="5k_calibration_allowed",
                sources_path=self.sources_path,
                protected_config_path=self.protected_config,
                protected_manifest_path=self.protected_manifest,
                reviewer_roster_path=self.reviewers,
                legal_evidence_dir=self.legal_evidence,
                trust_policy_path=self.policy,
            )

    async def test_finalize_unlocks_5k_only_after_two_bound_approvals(self) -> None:
        manifest, bindings = await self._manifest()
        await self._approve(bindings)
        decision = await finalize_calibration(
            manifest,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        self.assertEqual(decision.status, "passed")
        self.assertEqual(decision.next_stage, "5k_calibration_allowed")
        self.assertTrue(all(decision.checks.values()))

    async def test_5k_requires_passed_200_decision_before_creating_work_dir(self) -> None:
        work_dir = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "requires a passed calibration_200"):
            await run_calibration(
                stage="calibration_5k",
                sources_path=self.sources_path,
                protected_config_path=self.protected_config,
                protected_manifest_path=self.protected_manifest,
                reviewer_roster_path=self.reviewers,
                reviewer_qualification_report_path=self.reviewer_qualification,
                legal_evidence_dir=self.legal_evidence,
                trust_policy_path=self.policy,
                work_dir=work_dir,
                authority_database=self.authority_path,
                progress_to_stdout=False,
            )
        self.assertFalse(work_dir.exists())

    async def test_stage_count_mismatch_is_rejected_outside_dev_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly 200"):
            CalibrationManifest(
                calibration_id="invalid-count",
                stage="calibration_200",
                status="awaiting_review",
                target_records=201,
                final_records=201,
                source_registry_sha256="1" * 64,
                trust_policy_sha256="2" * 64,
                reviewer_roster_sha256="3" * 64,
                reviewer_qualification_path="missing-qualification.json",
                reviewer_qualification_sha256="6" * 64,
                legal_evidence_sha256="4" * 64,
                protected_manifest_sha256="5" * 64,
                authority_database=str(self.authority_path),
                artifacts={},
                artifact_sha256={},
                source_fractions={},
                review_bindings=[],
                crawl_report={},
                license_report={},
                contamination_report={},
                dedup_report={},
                quality_report={},
            )

    async def test_tampered_200_manifest_invalidates_advancement(self) -> None:
        manifest_path, bindings = await self._manifest()
        await self._approve(bindings)
        decision = await finalize_calibration(
            manifest_path,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        decision_path = manifest_path.parent / "calibration_decision.json"
        validated = validate_advancement_decision(
            decision_path,
            expected_stage="calibration_200",
            expected_next_stage="5k_calibration_allowed",
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        self.assertEqual(validated.calibration_id, decision.calibration_id)
        manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            validate_advancement_decision(
                decision_path,
                expected_stage="calibration_200",
                expected_next_stage="5k_calibration_allowed",
                sources_path=self.sources_path,
                protected_config_path=self.protected_config,
                protected_manifest_path=self.protected_manifest,
                reviewer_roster_path=self.reviewers,
                legal_evidence_dir=self.legal_evidence,
                trust_policy_path=self.policy,
            )

    async def test_passed_5k_emits_100k_dataset_authorization(self) -> None:
        manifest_200, bindings = await self._manifest()
        await self._approve(bindings)
        await finalize_calibration(
            manifest_200,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        decision_200 = manifest_200.parent / "calibration_decision.json"
        manifest_5k, bindings_5k = await self._manifest(
            stage="calibration_5k",
            prior_decision=decision_200,
        )
        await self._approve(bindings_5k)
        decision = await finalize_calibration(
            manifest_5k,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
            output_path=self.root / "calibration_5k_decision.json",
        )
        self.assertEqual(decision.status, "passed")
        self.assertEqual(decision.next_stage, "100k_dataset_build_allowed")
        validated = validate_advancement_decision(
            self.root / "calibration_5k_decision.json",
            expected_stage="calibration_5k",
            expected_next_stage="100k_dataset_build_allowed",
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        self.assertIsInstance(validated, CalibrationDecision)

    async def test_active_governance_and_authority_drift_invalidate_decision(self) -> None:
        manifest_path, bindings = await self._manifest()
        await self._approve(bindings)
        await finalize_calibration(
            manifest_path,
            sources_path=self.sources_path,
            protected_config_path=self.protected_config,
            protected_manifest_path=self.protected_manifest,
            reviewer_roster_path=self.reviewers,
            legal_evidence_dir=self.legal_evidence,
            trust_policy_path=self.policy,
        )
        decision_path = manifest_path.parent / "calibration_decision.json"

        def validate() -> None:
            validate_advancement_decision(
                decision_path,
                expected_stage="calibration_200",
                expected_next_stage="5k_calibration_allowed",
                sources_path=self.sources_path,
                protected_config_path=self.protected_config,
                protected_manifest_path=self.protected_manifest,
                reviewer_roster_path=self.reviewers,
                legal_evidence_dir=self.legal_evidence,
                trust_policy_path=self.policy,
            )

        mutations: list[tuple[Path, Callable[[dict[str, Any]], dict[str, Any]]]] = [
            (
                self.sources_path,
                lambda payload: {
                    **payload,
                    "sources": [
                        {**payload["sources"][0], "collection_budget": 999},
                        *payload["sources"][1:],
                    ],
                },
            ),
            (
                self.reviewers,
                lambda payload: {
                    **payload,
                    "identities": [
                        {**payload["identities"][0], "approved_at": "2026-07-20T12:00:00+06:00"},
                        *payload["identities"][1:],
                    ],
                },
            ),
            (
                self.reviewer_qualification,
                lambda payload: {
                    **payload,
                    "created_at_unix": float(payload.get("created_at_unix", 0.0)) + 1.0,
                },
            ),
            (
                self.protected_manifest,
                lambda payload: {**payload, "created_at_unix": float(payload["created_at_unix"]) + 1.0},
            ),
        ]
        for path, mutate in mutations:
            original = path.read_bytes()
            payload = json.loads(original.decode("utf-8"))
            _write_json(path, mutate(payload))
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                validate()
            path.write_bytes(original)

        legal_path = self.legal_evidence / "source-0" / "license.txt"
        legal_original = legal_path.read_bytes()
        legal_path.write_bytes(legal_original + b"\nchanged")
        with self.assertRaises(ValueError):
            validate()
        legal_path.write_bytes(legal_original)

        policy_copy = self.root / "policy.json"
        policy_copy.write_bytes(self.policy.read_bytes())
        decision_payload = CalibrationDecision.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
        manifest_payload = CalibrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        # The baseline decision remains bound to the original policy; changing
        # only the active policy must therefore invalidate it.
        policy_json = json.loads(policy_copy.read_text(encoding="utf-8"))
        policy_json["policy_id"] = "craftly-dataset-trust-v1-drifted"
        _write_json(policy_copy, policy_json)
        with self.assertRaises(ValueError):
            validate_advancement_decision(
                decision_path,
                expected_stage="calibration_200",
                expected_next_stage="5k_calibration_allowed",
                sources_path=self.sources_path,
                protected_config_path=self.protected_config,
                protected_manifest_path=self.protected_manifest,
                reviewer_roster_path=self.reviewers,
                legal_evidence_dir=self.legal_evidence,
                trust_policy_path=policy_copy,
            )
        self.assertEqual(decision_payload.trust_policy_sha256, manifest_payload.trust_policy_sha256)

        with closing(sqlite3.connect(self.authority_path)) as connection:
            connection.execute(
                "UPDATE dataset_review_items SET status='rejected' WHERE id=?",
                (bindings[0].review_item_id,),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "evidence replay failed|authority evidence changed"):
            validate()


if __name__ == "__main__":
    unittest.main()
