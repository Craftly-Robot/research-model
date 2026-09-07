from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

from src.craftly.gateway import api as gateway_api
from src.craftly.identity import AuthConfig
from src.craftly.learning.benchmark_contamination_filter import (
    build_protected_fingerprint_index,
    filter_benchmark_contamination_jsonl,
)
from src.craftly.learning.dataset_authority import (
    ReviewAdjudicationCreate,
    ReviewDecisionCreate,
    ReviewItemCreate,
    ReviewerIdentity,
    ReviewerRoster,
    SQLiteDatasetAuthorityStore,
    SourceSnapshotCreate,
    build_reviewer_roster_onboarding_template,
    evaluate_reviewer_qualification,
    finalize_reviewer_governance_bundle,
    initialize_reviewer_governance_bundle,
    initialize_reviewer_qualification_pack,
    prepare_reviewer_delivery_packages,
    reviewer_roster_readiness,
)
from src.craftly.learning.data_pipeline import (
    DataPipelineConfig,
    validate_data_pipeline_production_config,
)
from src.craftly.learning.near_dedup import (
    PostgresLSHDedupIndex,
    deduplicate_jsonl,
    run_dedup_scale_validation,
)
from src.craftly.learning.production_dataset import (
    ProductionDatasetConfig,
    ProductionDatasetManifest,
    SplitManifest,
    _sha256_file,
    split_train_val_test,
    validate_dataset_manifest_for_promotion,
)
from src.craftly.learning.source_registry import (
    SourceEvidenceOrigin,
    SourceRegistry,
    load_evidence_origin_registry,
    source_registry_entry_sha256,
)
from src.craftly.learning.source_reputation import build_source_reputation_report
from src.craftly.learning.training_data_gate import TrainingDataGateConfig, score_row
from src.craftly.learning.web_ingest import SourceSpec


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class DatasetAuthorityTest(unittest.IsolatedAsyncioTestCase):
    async def test_governance_bundle_is_versioned_hash_bound_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = initialize_reviewer_governance_bundle(Path(temp_dir) / "governance")
            roster_path = Path(bundle.roster_path)
            rubric_path = Path(bundle.rubric_path)
            onboarding_path = Path(bundle.onboarding_path)
            self.assertEqual(bundle.status, "awaiting_human_identities")
            self.assertEqual(bundle.rubric_id, "craftly-review-rubric-v1")
            self.assertTrue(roster_path.is_file())
            self.assertTrue(rubric_path.is_file())
            self.assertTrue(onboarding_path.is_file())
            self.assertEqual(
                hashlib.sha256(roster_path.read_bytes()).hexdigest(),
                bundle.roster_sha256,
            )
            self.assertEqual(
                hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
                bundle.rubric_sha256,
            )
            roster = ReviewerRoster.model_validate_json(roster_path.read_text(encoding="utf-8"))
            self.assertEqual(roster.identities, [])
            rubric = rubric_path.read_text(encoding="utf-8")
            self.assertIn("incorrect_or_misleading", rubric)
            self.assertIn("Policy-floor Cohen's kappa: `>= 0.80`", rubric)
            with self.assertRaises(FileExistsError):
                initialize_reviewer_governance_bundle(Path(temp_dir) / "governance")

    async def test_reviewer_qualification_pack_is_balanced_sealed_and_non_production(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = initialize_reviewer_qualification_pack(Path(temp_dir) / "governance")
            pack_path = Path(report.pack_path)
            answer_key_path = Path(report.answer_key_path)
            rows = [
                json.loads(line)
                for line in pack_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
            self.assertEqual(report.item_count, 20)
            self.assertEqual(len(rows), 20)
            self.assertTrue(all("expected_decision" not in row for row in rows))
            self.assertEqual(
                sum(item["expected_decision"] == "approve" for item in answer_key["answers"]),
                10,
            )
            self.assertEqual(
                sum(item["expected_decision"] == "reject" for item in answer_key["answers"]),
                10,
            )
            self.assertIn("never enter training", report.handling_warning)
            with self.assertRaises(FileExistsError):
                initialize_reviewer_qualification_pack(Path(temp_dir) / "governance")

    async def test_ready_roster_is_rebound_and_delivery_packages_exclude_identity_and_answer_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governance_dir = Path(temp_dir) / "governance"
            initialize_reviewer_governance_bundle(governance_dir)
            initialize_reviewer_qualification_pack(governance_dir)
            roster = ReviewerRoster(
                roster_id="craftly-data-reviewers-v1",
                identities=[
                    ReviewerIdentity(
                        reviewer_id="reviewer-one",
                        identity_provider_subject="oidc:subject-one",
                        roles={"reviewer"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                    ReviewerIdentity(
                        reviewer_id="reviewer-two",
                        identity_provider_subject="oidc:subject-two",
                        roles={"reviewer"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                    ReviewerIdentity(
                        reviewer_id="adjudicator-one",
                        identity_provider_subject="oidc:subject-three",
                        roles={"adjudicator"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                ],
            )
            roster_path = governance_dir / "data_reviewers.json"
            roster_path.write_text(roster.model_dump_json(indent=2) + "\n", encoding="utf-8")
            finalized = finalize_reviewer_governance_bundle(governance_dir)
            self.assertEqual(finalized.status, "ready_for_qualification")
            self.assertEqual(
                finalized.roster_sha256,
                hashlib.sha256(roster_path.read_bytes()).hexdigest(),
            )

            delivery = prepare_reviewer_delivery_packages(
                governance_dir,
                Path(temp_dir) / "deliveries",
            )
            self.assertEqual(delivery.status, "ready_for_secure_delivery")
            self.assertEqual(len(delivery.packages), 2)
            for package in delivery.packages:
                with zipfile.ZipFile(package.package_path, mode="r") as archive:
                    names = set(archive.namelist())
                    self.assertEqual(
                        names,
                        {
                            "delivery-manifest.json",
                            "reviewer-qualification-v1.jsonl",
                            "reviewer-responses.jsonl",
                            "reviewer-rubric-v1.md",
                        },
                    )
                    combined = b"\n".join(archive.read(name) for name in sorted(names))
                self.assertNotIn(b"answer-key", combined)
                self.assertNotIn(b"oidc:subject", combined)
                self.assertFalse(package.answer_key_included)
            with self.assertRaises(FileExistsError):
                prepare_reviewer_delivery_packages(
                    governance_dir,
                    Path(temp_dir) / "deliveries",
                )

    async def test_qualification_evaluator_is_hash_bound_blind_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            governance_dir = root / "governance"
            initialize_reviewer_governance_bundle(governance_dir)
            qualification = initialize_reviewer_qualification_pack(governance_dir)
            roster = ReviewerRoster(
                roster_id="craftly-data-reviewers-v1",
                identities=[
                    ReviewerIdentity(
                        reviewer_id="reviewer-one",
                        identity_provider_subject="oidc:reviewer-one",
                        roles={"reviewer"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                    ReviewerIdentity(
                        reviewer_id="reviewer-two",
                        identity_provider_subject="oidc:reviewer-two",
                        roles={"reviewer"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                    ReviewerIdentity(
                        reviewer_id="adjudicator-one",
                        identity_provider_subject="oidc:adjudicator-one",
                        roles={"adjudicator"},
                        approved_by="governance-owner",
                        approved_at="2026-07-20T01:00:00+06:00",
                    ),
                ],
            )
            (governance_dir / "data_reviewers.json").write_text(
                roster.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            finalize_reviewer_governance_bundle(governance_dir)
            answers = json.loads(
                Path(qualification.answer_key_path).read_text(encoding="utf-8")
            )["answers"]
            response_paths: list[Path] = []
            for reviewer_id in ("reviewer-one", "reviewer-two"):
                response_path = root / f"{reviewer_id}-responses.jsonl"
                _write_jsonl(
                    response_path,
                    [
                        {
                            "item_id": answer["item_id"],
                            "reviewer_id": reviewer_id,
                            "decision": answer["expected_decision"],
                            "rationale": (
                                "This decision follows the governed rubric and the supplied evidence "
                                f"for {answer['item_id']}."
                            ),
                        }
                        for answer in answers
                    ],
                )
                response_paths.append(response_path)

            report = evaluate_reviewer_qualification(
                governance_dir=governance_dir,
                response_paths=response_paths,
                output_dir=root / "evaluation",
            )
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.reviewer_agreement_kappa, 1.0)
            self.assertTrue(all(item.accuracy == 1.0 for item in report.reviewers))
            serialized = (root / "evaluation" / "reviewer-qualification-evaluation.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("expected_decision", serialized)
            self.assertNotIn("oidc:", serialized)
            self.assertNotIn("This decision follows", serialized)
            with self.assertRaises(FileExistsError):
                evaluate_reviewer_qualification(
                    governance_dir=governance_dir,
                    response_paths=response_paths,
                    output_dir=root / "evaluation",
                )

            bad_rows = [
                {
                    "item_id": answer["item_id"],
                    "reviewer_id": "reviewer-two",
                    "decision": (
                        "reject"
                        if index < 2 and answer["expected_decision"] == "approve"
                        else "approve"
                        if index < 2
                        else answer["expected_decision"]
                    ),
                    "rationale": (
                        "This independent decision records a concrete evidence assessment under "
                        "the governed reviewer rubric."
                    ),
                }
                for index, answer in enumerate(answers)
            ]
            _write_jsonl(response_paths[1], bad_rows)
            failed = evaluate_reviewer_qualification(
                governance_dir=governance_dir,
                response_paths=response_paths,
                output_dir=root / "failed-evaluation",
            )
            self.assertEqual(failed.status, "failed")
            self.assertTrue(any("accuracy" in blocker for blocker in failed.blockers))

            answer_key_path = Path(qualification.answer_key_path)
            answer_key_path.write_text(
                answer_key_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "answer key"):
                evaluate_reviewer_qualification(
                    governance_dir=governance_dir,
                    response_paths=response_paths,
                    output_dir=root / "tampered-evaluation",
                )

    async def test_reviewer_onboarding_never_fabricates_identity_and_reports_readiness(self) -> None:
        template = build_reviewer_roster_onboarding_template()
        self.assertEqual(template.status, "awaiting_human_identities")
        self.assertEqual(
            [slot.required_role for slot in template.required_slots],
            ["reviewer", "reviewer", "adjudicator"],
        )
        empty = ReviewerRoster(roster_id=template.roster_id, identities=[])
        blocked = reviewer_roster_readiness(empty)
        self.assertEqual(blocked.status, "blocked")
        self.assertEqual(blocked.active_reviewer_count, 0)
        self.assertEqual(blocked.active_adjudicator_count, 0)

        ready = ReviewerRoster(
            roster_id=template.roster_id,
            identities=[
                ReviewerIdentity(
                    reviewer_id="reviewer-one",
                    identity_provider_subject="idp:reviewer-one",
                    roles={"reviewer"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
                ReviewerIdentity(
                    reviewer_id="reviewer-two",
                    identity_provider_subject="idp:reviewer-two",
                    roles={"reviewer"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
                ReviewerIdentity(
                    reviewer_id="adjudicator-one",
                    identity_provider_subject="idp:adjudicator-one",
                    roles={"adjudicator"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
            ],
        )
        readiness = reviewer_roster_readiness(ready)
        self.assertEqual(readiness.status, "ready")
        self.assertEqual(readiness.independent_subject_count, 3)
        self.assertEqual(readiness.blockers, [])

        placeholder = ReviewerRoster(
            roster_id=template.roster_id,
            identities=[
                ReviewerIdentity(
                    reviewer_id="placeholder-reviewer-one",
                    identity_provider_subject="oidc:placeholder-one",
                    roles={"reviewer"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
                ReviewerIdentity(
                    reviewer_id="reviewer-two",
                    identity_provider_subject="oidc:reviewer-two",
                    roles={"reviewer"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
                ReviewerIdentity(
                    reviewer_id="adjudicator-one",
                    identity_provider_subject="oidc:adjudicator-one",
                    roles={"adjudicator"},
                    approved_by="governance-owner",
                    approved_at="2026-07-19T12:00:00+06:00",
                ),
            ],
        )
        placeholder_report = reviewer_roster_readiness(placeholder)
        self.assertEqual(placeholder_report.status, "blocked")
        self.assertTrue(any("placeholder marker" in blocker for blocker in placeholder_report.blockers))

    async def test_empty_review_evidence_is_explicit_and_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteDatasetAuthorityStore(Path(temp_dir) / "authority.sqlite3")
            report = await store.review_evidence()
            self.assertEqual(report.schema_version, 1)
            self.assertEqual(report.status, "empty")
            self.assertEqual(report.total_items, 0)
            self.assertEqual(report.decision_count, 0)
            self.assertEqual(report.source_count, 0)
            self.assertEqual(report.approved, 0)
            self.assertEqual(report.rejected, 0)
            self.assertEqual(report.pending, 0)
            self.assertEqual(report.paired_reviews, 0)
            self.assertEqual(report.by_source, {})

    async def test_blind_two_reviewer_conflict_requires_independent_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteDatasetAuthorityStore(Path(temp_dir) / "authority.sqlite3")
            request = ReviewItemCreate(
                content_hash="a" * 64,
                source_snapshot_sha256="b" * 64,
                source_id="trusted-source",
                data_type="verified_security_patch",
                high_value=True,
                payload={"text": "reviewed defensive patch"},
            )
            item = await store.enqueue(request)
            await store.claim(item.review_item_id, "reviewer-one", 600)
            first = await store.decide(
                item.review_item_id,
                "reviewer-one",
                ReviewDecisionCreate(
                    decision="approve",
                    rationale="The patch evidence and regression coverage are complete.",
                    content_hash=request.content_hash,
                    source_snapshot_sha256=request.source_snapshot_sha256,
                    evidence={"manifest": "verified"},
                ),
            )
            self.assertEqual(first.status, "in_review")
            await store.claim(item.review_item_id, "reviewer-two", 600)
            blinded = await store.get_item(item.review_item_id, reviewer_id="reviewer-two")
            self.assertEqual(blinded.decisions, [])
            with self.assertRaises(PermissionError):
                await store.claim(item.review_item_id, "reviewer-one", 600)
            conflict = await store.decide(
                item.review_item_id,
                "reviewer-two",
                ReviewDecisionCreate(
                    decision="reject",
                    rationale="The static-analysis evidence does not cover the modified sink.",
                    content_hash=request.content_hash,
                    source_snapshot_sha256=request.source_snapshot_sha256,
                    evidence={"missing_check": "static-sink"},
                ),
            )
            self.assertEqual(conflict.status, "adjudication_required")
            with self.assertRaises(PermissionError):
                await store.adjudicate(
                    item.review_item_id,
                    "reviewer-one",
                    ReviewAdjudicationCreate(
                        decision="approve",
                        rationale="A reviewer cannot adjudicate their own disputed record.",
                    ),
                )
            resolved = await store.adjudicate(
                item.review_item_id,
                "adjudicator-three",
                ReviewAdjudicationCreate(
                    decision="reject",
                    rationale="The missing static-analysis evidence is material and blocks promotion.",
                    evidence={"decision": "return_to_verification"},
                ),
            )
            self.assertEqual(resolved.status, "rejected")
            report = await store.review_evidence()
            self.assertEqual(report.by_source["trusted-source"].rejected, 1)
            self.assertEqual(report.by_source["trusted-source"].paired_reviews, 1)
            self.assertEqual(report.status, "complete")
            self.assertEqual(report.total_items, 1)
            self.assertEqual(report.decision_count, 2)
            self.assertEqual(report.rejected, 1)

    async def test_source_drift_creates_new_review_and_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteDatasetAuthorityStore(Path(temp_dir) / "authority.sqlite3")
            snapshot = await store.record_source_snapshot(
                SourceSnapshotCreate(
                    source_id="source-a",
                    source_family="family-a",
                    immutable_revision="commit-123",
                    registry_sha256="1" * 64,
                    license_evidence_sha256="2" * 64,
                    legal_approval_sha256="3" * 64,
                    snapshot_sha256="4" * 64,
                    status="approved",
                )
            )
            repeated = await store.record_source_snapshot(
                SourceSnapshotCreate(
                    source_id="source-a",
                    source_family="family-a",
                    immutable_revision="commit-123",
                    registry_sha256="1" * 64,
                    license_evidence_sha256="2" * 64,
                    legal_approval_sha256="3" * 64,
                    snapshot_sha256="4" * 64,
                    status="approved",
                )
            )
            self.assertEqual(snapshot.snapshot_id, repeated.snapshot_id)
            first = await store.enqueue(
                ReviewItemCreate(
                    content_hash="5" * 64,
                    source_snapshot_sha256="4" * 64,
                    source_id="source-a",
                    data_type="code",
                    high_value=False,
                )
            )
            changed = await store.enqueue(
                ReviewItemCreate(
                    content_hash="5" * 64,
                    source_snapshot_sha256="6" * 64,
                    source_id="source-a",
                    data_type="code",
                    high_value=False,
                )
            )
            self.assertNotEqual(first.review_item_id, changed.review_item_id)
            with self.assertRaises(ValueError):
                SourceSnapshotCreate(
                    source_id="source-a",
                    source_family="family-a",
                    immutable_revision="rolling",
                    registry_sha256="1" * 64,
                    license_evidence_sha256="2" * 64,
                    legal_approval_sha256="3" * 64,
                    snapshot_sha256="4" * 64,
                    status="approved",
                )


class DatasetFingerprintAndManifestTest(unittest.TestCase):
    def test_data_governance_scopes_are_not_satisfied_by_generic_api_scope(self) -> None:
        original = gateway_api.AUTH_CONFIG
        gateway_api.AUTH_CONFIG = AuthConfig(enabled=True, jwt_secret="x" * 32)
        try:
            request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
            request.state.jwt_claims = {"scopes": ["api"]}
            with self.assertRaises(PermissionError):
                gateway_api.require_scope(request, "data:promote")
            request.state.jwt_claims = {"scopes": ["data:promote"]}
            gateway_api.require_scope(request, "data:promote")
        finally:
            gateway_api.AUTH_CONFIG = original

    def test_high_value_rows_enter_review_queue_without_durable_approval(self) -> None:
        row = {
            "source": "approved-source",
            "text": "Defensive patch and regression test evidence " * 20,
            "content_hash": "a" * 64,
            "quality": {
                "quality_score": 0.90,
                "data_type": "patch",
                "labels": ["patch", "tests", "defensive_security"],
                "risk_flags": [],
            },
        }
        decision = score_row(
            row,
            reputation_by_source={
                "approved-source": {
                    "reputation_score": 0.90,
                    "reputation_lower_bound": 0.80,
                    "action": "promote",
                    "approval_status": "approved",
                    "license_trust": 1.0,
                }
            },
            config=TrainingDataGateConfig(require_governed_sources=True),
        )
        self.assertEqual(decision.status, "review_queue")
        self.assertIn("independent_high_value_review_required", decision.reasons)

    def test_source_reputation_never_inherits_other_source_task_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quality = root / "quality.json"
            tasks = root / "tasks.json"
            feedback = root / "feedback.json"
            registry = root / "registry.json"
            quality.write_text(
                json.dumps(
                    {
                        "sources": [
                            {"source": "source-a", "rows": 100, "avg_quality_score": 0.8, "defensive_security_rows": 50, "code_rows": 20},
                            {"source": "source-b", "rows": 100, "avg_quality_score": 0.8, "defensive_security_rows": 50, "code_rows": 20},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tasks.write_text(json.dumps({"by_source": {"source-a": {"extracted": 40}}}), encoding="utf-8")
            feedback.write_text(json.dumps({"by_source": {"source-a": {"score": 0.9}}}), encoding="utf-8")
            registry.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": source,
                                "source_id": source,
                                "trust_tier": "reviewed",
                                "approval_status": "approved",
                                "approved_use": "foundation",
                                "license_evidence_sha256": "1" * 64,
                                "legal_approval_sha256": "2" * 64,
                            }
                            for source in ("source-a", "source-b")
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = build_source_reputation_report(
                source_quality_report_path=quality,
                task_report_path=tasks,
                feedback_report_path=feedback,
                source_registry_path=registry,
            )
            scores = {item.source: item for item in report.sources}
            self.assertGreater(scores["source-a"].task_coverage, 0.0)
            self.assertEqual(scores["source-b"].task_coverage, 0.0)
            self.assertEqual(scores["source-b"].benchmark_feedback_score, 0.0)

    def test_postgres_lsh_contract_rejects_non_postgres_dsn(self) -> None:
        with self.assertRaises(ValueError):
            PostgresLSHDedupIndex("sqlite:///unsafe.db", dataset_version="version-1")

    def test_source_selection_is_exact_deterministic_and_hash_bound(self) -> None:
        sources = [
            SourceSpec(
                name=f"source-{index}",
                source_id=f"source-{index}",
                source_family=f"family-{index}",
                urls=[f"https://example.com/{index}"],
                allowed_domains=["example.com"],
                license="mit",
            )
            for index in range(3)
        ]
        registry = SourceRegistry(sources)
        selected, manifest = registry.select_sources(
            ["source-2", "source-0"],
            expected_count=2,
        )
        self.assertEqual([source.source_id for source in selected.sources], ["source-0", "source-2"])
        self.assertEqual(manifest.source_ids, ["source-0", "source-2"])
        self.assertEqual(manifest.source_count, 2)
        self.assertEqual(
            [entry.registry_entry_sha256 for entry in manifest.entries],
            [
                source_registry_entry_sha256(sources[0]),
                source_registry_entry_sha256(sources[2]),
            ],
        )
        self.assertEqual(len(registry.sources), 3)
        with self.assertRaisesRegex(ValueError, "duplicate selected"):
            registry.select_sources(["source-0", "source-0"], expected_count=2)
        with self.assertRaisesRegex(ValueError, "unknown selected"):
            registry.select_sources(["source-missing"], expected_count=1)
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            registry.select_sources(["source-0"], expected_count=2)

    def test_tracked_eight_source_staging_registry_is_exact_and_quarantined(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ultimate = SourceRegistry.from_file(root / "config" / "data_sources.ultimate.json")
        staging = SourceRegistry.from_file(root / "config" / "data_sources.governed.staging.json")
        selection_payload = json.loads(
            (root / "config" / "data_sources.governed.selection.json").read_text(encoding="utf-8")
        )
        selected, expected_manifest = ultimate.select_sources(
            selection_payload["source_ids"],
            expected_count=8,
        )
        self.assertEqual(
            [source.model_dump(mode="json") for source in staging.sources],
            [source.model_dump(mode="json") for source in selected.sources],
        )
        self.assertEqual(selection_payload, expected_manifest.model_dump(mode="json"))
        self.assertTrue(all(source.approval_status == "pending" for source in staging.sources))
        self.assertTrue(all(source.trust_tier == "quarantine" for source in staging.sources))
        licenses = {source.source_id: source.license for source in staging.sources}
        self.assertEqual(licenses["owasp-cheat-sheet-series"], "cc-by-sa-4.0")
        evidence_origins = load_evidence_origin_registry(
            root / "config" / "governed_source_evidence_origins.json"
        )
        self.assertEqual(
            {source.source_id for source in evidence_origins.sources},
            {source.source_id for source in staging.sources},
        )
        self.assertEqual(
            {source.source_id: source.expected_license for source in evidence_origins.sources},
            licenses,
        )

    def test_evidence_candidate_materialization_is_atomic_and_never_approves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = SourceSpec(
                name="official-docs",
                source_id="official-docs",
                source_family="official-docs",
                urls=["https://docs.example.com/security"],
                allowed_domains=["docs.example.com"],
                license="apache-2.0",
            )
            origins = root / "origins.json"
            origins.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": [
                            {
                                "source_id": "official-docs",
                                "origin_kind": "git_repository",
                                "expected_license": "apache-2.0",
                                "repository_url": "https://github.com/example/docs.git",
                                "git_ref": "HEAD",
                                "license_path": "LICENSE",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "evidence"
            with (
                patch(
                    "src.craftly.learning.source_registry._resolve_public_git_revision",
                    return_value="a" * 40,
                ),
                patch(
                    "src.craftly.learning.source_registry._fetch_governed_evidence_url",
                    return_value=(b"Apache License 2.0\n", {"content-type": "text/plain"}),
                ),
            ):
                report = SourceRegistry([source]).materialize_evidence_candidates(
                    origin_registry_path=origins,
                    output_dir=output,
                )
            self.assertFalse(report.production_collection_authorized)
            self.assertEqual(report.status, "awaiting_human_legal_approval")
            self.assertTrue((output / "evidence-candidate-manifest.json").is_file())
            self.assertTrue((output / "official-docs" / "license.txt").is_file())
            self.assertTrue((output / "official-docs" / "candidate.json").is_file())
            template = json.loads(
                (output / "official-docs" / "approval.template.json").read_text(encoding="utf-8")
            )
            self.assertEqual(template["decision"], "pending_human_review")
            self.assertEqual(template["immutable_revision"], "a" * 40)
            self.assertFalse((output / "official-docs" / "approval.json").exists())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                SourceRegistry([source]).materialize_evidence_candidates(
                    origin_registry_path=origins,
                    output_dir=output,
                )

    def test_evidence_origin_rejects_ssrf_and_path_escape(self) -> None:
        with self.assertRaisesRegex(ValueError, "official HTTPS GitHub"):
            SourceEvidenceOrigin(
                source_id="unsafe-source",
                origin_kind="git_repository",
                expected_license="apache-2.0",
                repository_url="https://127.0.0.1/repository.git",
                git_ref="HEAD",
                license_path="LICENSE",
            )
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            SourceEvidenceOrigin(
                source_id="unsafe-source",
                origin_kind="git_repository",
                expected_license="apache-2.0",
                repository_url="https://github.com/example/repository.git",
                git_ref="HEAD",
                license_path="../LICENSE",
            )

    def test_registry_write_refuses_approved_source_removal_or_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "governed.json"
            approved = SourceSpec(
                name="approved-source",
                source_id="approved-source",
                source_family="approved-family",
                urls=["https://example.com/docs"],
                allowed_domains=["example.com"],
                license="mit",
                trust_tier="reviewed",
                approval_status="approved",
                immutable_revision="commit-abc123",
                license_evidence_sha256="1" * 64,
                legal_approval_sha256="2" * 64,
                approval_request_sha256="3" * 64,
            )
            SourceRegistry([approved]).write(target)
            with self.assertRaisesRegex(ValueError, "remove previously approved"):
                SourceRegistry(
                    [
                        SourceSpec(
                            name="different-source",
                            source_id="different-source",
                            source_family="different-family",
                            urls=["https://example.com/other"],
                            allowed_domains=["example.com"],
                            license="mit",
                        )
                    ]
                ).write(target)
            changed = approved.model_copy(update={"immutable_revision": "commit-def456"})
            with self.assertRaisesRegex(ValueError, "alter previously approved"):
                SourceRegistry([changed]).write(target)
            self.assertEqual(
                SourceRegistry.from_file(target).sources[0].immutable_revision,
                "commit-abc123",
            )

    def test_approval_request_manifest_uses_portable_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SourceRegistry(
                [
                    SourceSpec(
                        name="approval-source",
                        source_id="approval-source",
                        source_family="approval-family",
                        urls=["https://example.com/docs"],
                        allowed_domains=["example.com"],
                        license="mit",
                    )
                ]
            )
            output = Path(temp_dir) / "requests"
            registry.prepare_approval_requests(output)
            manifest = json.loads((output / "approval-request-manifest.json").read_text(encoding="utf-8"))
            request = json.loads(
                (output / "approval-source.approval-request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["requests"][0]["path"], "approval-source.approval-request.json")
            self.assertFalse(Path(manifest["requests"][0]["path"]).is_absolute())
            self.assertEqual(request["status"], "awaiting_legal_decision")
            self.assertEqual(request["approval_template"]["decision"], "pending_human_review")
            self.assertIsNone(request["approval_template"]["approved_by"])
            self.assertIsNone(request["approval_template"]["immutable_revision"])

    def test_source_approval_is_bound_to_real_evidence_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            license_evidence = root / "license.txt"
            legal_approval = root / "approval.json"
            license_evidence.write_text("Reviewed MIT license evidence", encoding="utf-8")
            registry = SourceRegistry(
                [
                    SourceSpec(
                        name="approved-test-source",
                        urls=["https://example.com/docs"],
                        allowed_domains=["example.com"],
                        license="mit",
                        source_id="approved-test-source",
                        source_family="approved-family",
                    )
                ]
            )
            source = registry.to_sources()[0]
            legal_approval.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approval_id": "approval-test-001",
                        "decision": "approved",
                        "source_id": "approved-test-source",
                        "registry_entry_sha256": source_registry_entry_sha256(source),
                        "immutable_revision": "commit-abc123",
                        "license": "mit",
                        "license_evidence_sha256": _sha256_file(license_evidence),
                        "approved_use": "defensive",
                        "approved_by": "legal-test-reviewer",
                        "approved_at": "2026-07-19T12:00:00+06:00",
                        "scope": "training_collection",
                        "rationale": "The source license and intended defensive training use were reviewed and approved.",
                    }
                ),
                encoding="utf-8",
            )
            approved = registry.approve_source(
                source_id="approved-test-source",
                immutable_revision="commit-abc123",
                license_evidence_path=license_evidence,
                legal_approval_path=legal_approval,
            )
            self.assertEqual(approved.approval_status, "approved")
            self.assertEqual(approved.license_evidence_sha256, _sha256_file(license_evidence))
            self.assertEqual(approved.legal_approval_sha256, _sha256_file(legal_approval))
            self.assertEqual(registry.validate(production=True).approved_sources, 1)

    def test_source_approval_rejects_evidence_bound_to_another_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            license_evidence = root / "license.txt"
            legal_approval = root / "approval.json"
            license_evidence.write_text("Reviewed MIT license evidence", encoding="utf-8")
            legal_approval.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approval_id": "approval-test-002",
                        "decision": "approved",
                        "source_id": "approved-test-source",
                        "registry_entry_sha256": "0" * 64,
                        "immutable_revision": "commit-abc123",
                        "license": "mit",
                        "license_evidence_sha256": _sha256_file(license_evidence),
                        "approved_use": "defensive",
                        "approved_by": "legal-test-reviewer",
                        "approved_at": "2026-07-19T12:00:00+06:00",
                        "scope": "training_collection",
                        "rationale": "The source license and intended defensive training use were reviewed and approved.",
                    }
                ),
                encoding="utf-8",
            )
            registry = SourceRegistry(
                [
                    SourceSpec(
                        name="approved-test-source",
                        urls=["https://example.com/docs"],
                        allowed_domains=["example.com"],
                        license="mit",
                        source_id="approved-test-source",
                        source_family="approved-family",
                    )
                ]
            )
            with self.assertRaisesRegex(ValueError, "registry_entry_sha256"):
                registry.approve_source(
                    source_id="approved-test-source",
                    immutable_revision="commit-abc123",
                    license_evidence_path=license_evidence,
                    legal_approval_path=legal_approval,
                )

    def test_scale_validation_uses_bounded_index_and_subquadratic_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_dedup_scale_validation(record_count=2_000, output_dir=temp_dir)
            self.assertEqual(report.status, "passed")
            self.assertTrue(report.memory_bounded)
            self.assertLess(report.comparisons_per_record, 128.0)
            self.assertTrue(Path(report.index_path).exists())

    def test_streaming_lsh_removes_exact_structural_lineage_and_near_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "rows.jsonl"
            rows = [
                {
                    "source": "one",
                    "language": "python",
                    "text": "def validate(value):\n    return value.strip() if isinstance(value, str) else ''\n",
                    "patch_lineage": "patch-a",
                },
                {
                    "source": "one",
                    "language": "python",
                    "text": "def validate(value):\n    return value.strip() if isinstance(value, str) else ''\n",
                    "patch_lineage": "patch-b",
                },
                {
                    "source": "two",
                    "language": "python",
                    "text": "def validate(value):\n    return value.strip() if isinstance(value, str) else None\n",
                    "patch_lineage": "patch-a",
                },
                {
                    "source": "three",
                    "language": "python",
                    "text": "def authorize(subject, action):\n    return bool(subject and action)\n",
                    "patch_lineage": "patch-c",
                },
            ]
            _write_jsonl(source, rows)
            report = deduplicate_jsonl([source], root / "clean.jsonl", index_path=root / "dedup.sqlite3")
            self.assertEqual(report.accepted, 2)
            self.assertGreaterEqual(report.exact_duplicates + report.structural_duplicates + report.lineage_duplicates, 2)
            self.assertLess(report.candidate_comparisons, len(rows) * len(rows))
            self.assertTrue(report.memory_bounded)

    def test_protected_fingerprints_block_exact_task_and_near_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            holdout = root / "protected.jsonl"
            candidate = root / "candidate.jsonl"
            protected_text = (
                "Implement a function that validates an authorization token, checks expiry, "
                "and returns a structured defensive error without exposing credentials."
            )
            _write_jsonl(holdout, [{"task_id": "secure-001", "text": protected_text}])
            _write_jsonl(
                candidate,
                [
                    {"source": "exact", "text": protected_text},
                    {"source": "task", "task_id": "secure-001", "text": "different task body with enough detail"},
                    {
                        "source": "clean",
                        "task_id": "independent-002",
                        "text": "Document deterministic build caching for a Rust workspace and its tests.",
                    },
                ],
            )
            index = build_protected_fingerprint_index([holdout], root / "protected.sqlite3")
            report = filter_benchmark_contamination_jsonl(
                [candidate],
                root / "clean.jsonl",
                protected_index_path=index,
            )
            self.assertEqual(report.accepted, 1)
            self.assertEqual(report.rejected, 2)
            self.assertEqual(report.exact_hits, 1)
            self.assertEqual(report.task_id_hits, 1)

    def test_group_split_never_crosses_repository_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "rows.jsonl"
            rows = [
                {"text": f"repository example {index}", "repository": "org/repo-a", "license": "mit"}
                for index in range(20)
            ]
            rows.extend(
                {"text": f"other family {index}", "repository": "org/repo-b", "license": "mit"}
                for index in range(20)
            )
            _write_jsonl(source, rows)
            manifest = split_train_val_test(
                source,
                root / "split",
                ProductionDatasetConfig(input_paths=[str(source)], output_dir=str(root / "out")),
            )
            observed: dict[str, set[str]] = {}
            for split_name, path in (("train", manifest.train_path), ("val", manifest.val_path), ("test", manifest.test_path)):
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    observed.setdefault(row["repository"], set()).add(split_name)
            self.assertTrue(all(len(splits) == 1 for splits in observed.values()))
            self.assertEqual(manifest.cross_split_group_collisions, 0)

    def test_manifest_tampering_invalidates_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "train.jsonl"
            artifact.write_text('{"text":"trusted"}\n', encoding="utf-8")
            policy = root / "policy.json"
            policy.write_text(Path("config/dataset_trust_policy.json").read_text(encoding="utf-8"), encoding="utf-8")
            manifest = ProductionDatasetManifest(
                dataset_id="craftly-test",
                version_id="version-1",
                status="promoted",
                output_dir=str(root),
                dev_smoke=True,
                artifacts={"train": str(artifact)},
                metrics={},
                reports={},
                artifact_sha256={"train": _sha256_file(artifact)},
                policy_sha256=_sha256_file(policy),
                promotion_decision={"status": "promoted"},
            )
            manifest_path = manifest.write(root)
            validated = validate_dataset_manifest_for_promotion(manifest_path, trust_policy_path=policy)
            self.assertEqual(validated.version_id, "version-1")
            artifact.write_text('{"text":"tampered"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dataset_manifest_for_promotion(manifest_path, trust_policy_path=policy)

    def test_production_training_requires_promoted_dataset_manifest(self) -> None:
        config = DataPipelineConfig(
            sources_path="config/data_sources.ultimate.json",
            frontier_backend="postgres",
            postgres_dsn="postgresql://craftly:secret@postgres/craftly",
            object_store_uri="s3://craftly-training",
            model_profile_name="craftly-1b",
            checkpoint_compare_prompt_suite="data/eval/qualification.jsonl",
            min_training_average_quality_score=0.80,
            min_training_rows=100_000,
            min_train_tokens=10_000_000,
            production_mode=True,
        )
        with self.assertRaisesRegex(ValueError, "promoted_dataset_manifest_path"):
            validate_data_pipeline_production_config(config)


if __name__ == "__main__":
    unittest.main()
