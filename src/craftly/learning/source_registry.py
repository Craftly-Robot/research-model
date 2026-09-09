"""Allowlisted source registry for defensive Craftly data collection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed git argv, validated public HTTPS repository
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx
from pydantic import Field, field_validator, model_validator

from src.craftly.learning.web_ingest import SourceSpec, allowed_url, load_sources
from src.craftly.shared.integrity import sha256_file
from src.craftly.shared.schemas import StrictModel

APPROVED_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "mit",
    "mpl-2.0",
    "postgresql",
    "psf-2.0",
    "public-domain",
    "unknown-ok",
}


class SourceRegistryReport(StrictModel):
    source_count: int
    url_count: int
    domains: list[str]
    categories: list[str]
    approved_sources: int = 0
    quarantine_sources: int = 0
    source_snapshot_sha256: str
    warnings: list[str] = Field(default_factory=list)


class LegalApprovalEvidence(StrictModel):
    schema_version: Literal[1] = 1
    approval_id: str = Field(min_length=8, max_length=160)
    decision: Literal["approved"]
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    registry_entry_sha256: str
    immutable_revision: str = Field(min_length=2, max_length=256)
    license: str = Field(min_length=2, max_length=80)
    license_evidence_sha256: str
    approved_use: Literal[
        "foundation", "defensive", "authorized_lab", "evaluation_only"
    ]
    approved_by: str = Field(min_length=3, max_length=256)
    approved_at: str
    scope: Literal["training_collection", "evaluation_only"]
    rationale: str = Field(min_length=30, max_length=4_000)

    @field_validator("registry_entry_sha256", "license_evidence_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("approval hashes must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def validate_timestamp_and_scope(self) -> "LegalApprovalEvidence":
        try:
            parsed = datetime.fromisoformat(self.approved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("approved_at must be an RFC3339 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("approved_at must include a timezone")
        if self.approved_use == "evaluation_only" and self.scope != "evaluation_only":
            raise ValueError("evaluation-only sources require evaluation_only scope")
        if (
            self.approved_use != "evaluation_only"
            and self.scope != "training_collection"
        ):
            raise ValueError("training sources require training_collection scope")
        return self


class SourceApprovalRequestArtifact(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["awaiting_legal_decision"] = "awaiting_legal_decision"
    source_id: str
    source_name: str
    source_family: str
    registry_entry_sha256: str
    current_license: str
    approved_use: str
    seed_urls: list[str]
    allowed_domains: list[str]
    required_legal_fields: list[str]
    evidence_paths: dict[str, str]
    approval_template: dict[str, object]
    immutable_revision_requirements: list[str]
    warning: str


class SourceEvidenceOrigin(StrictModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    origin_kind: Literal["git_repository", "web_snapshot"]
    expected_license: str = Field(min_length=2, max_length=80)
    repository_url: str | None = None
    git_ref: str | None = Field(default=None, min_length=1, max_length=160)
    license_path: str | None = Field(default=None, min_length=1, max_length=512)
    license_url: str | None = Field(default=None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_origin(self) -> "SourceEvidenceOrigin":
        if self.origin_kind == "git_repository":
            if not self.repository_url or not self.git_ref or not self.license_path:
                raise ValueError(
                    "git evidence origin requires repository_url, git_ref, and license_path"
                )
            parsed = urlparse(self.repository_url)
            if parsed.scheme != "https" or parsed.hostname != "github.com":
                raise ValueError(
                    "git evidence repository must be an official HTTPS GitHub URL"
                )
            if parsed.query or parsed.fragment or parsed.username or parsed.password:
                raise ValueError(
                    "git evidence repository URL cannot contain credentials, query, or fragment"
                )
            if (
                re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", parsed.path)
                is None
            ):
                raise ValueError(
                    "git evidence repository URL must contain exactly owner/repository"
                )
            relative = Path(self.license_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\x00" in self.license_path
            ):
                raise ValueError(
                    "git evidence license_path must be a safe relative path"
                )
            if self.license_url is not None:
                raise ValueError("git evidence origin cannot also define license_url")
        else:
            if not self.license_url:
                raise ValueError("web evidence origin requires license_url")
            parsed = urlparse(self.license_url)
            if parsed.scheme != "https" or parsed.hostname not in {"www.nist.gov"}:
                raise ValueError(
                    "web evidence URL must use an explicitly governed HTTPS host"
                )
            if parsed.query or parsed.fragment or parsed.username or parsed.password:
                raise ValueError(
                    "web evidence URL cannot contain credentials, query, or fragment"
                )
            if any(
                value is not None
                for value in (self.repository_url, self.git_ref, self.license_path)
            ):
                raise ValueError("web evidence origin cannot define git fields")
        return self


class SourceEvidenceOriginRegistry(StrictModel):
    schema_version: Literal[1] = 1
    sources: list[SourceEvidenceOrigin] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "SourceEvidenceOriginRegistry":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "source evidence origin registry contains duplicate source IDs"
            )
        return self


class SourceEvidenceCandidate(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["awaiting_human_legal_approval"] = "awaiting_human_legal_approval"
    source_id: str
    origin_kind: Literal["git_repository", "web_snapshot"]
    registry_entry_sha256: str
    expected_license: str
    requested_revision: str
    resolved_immutable_revision: str
    license_source_url: str
    license_evidence_path: str
    license_evidence_sha256: str
    license_evidence_bytes: int = Field(ge=1, le=1_048_576)
    response_metadata: dict[str, str] = Field(default_factory=dict)
    materialized_at_unix: float
    warning: str

    @field_validator("registry_entry_sha256", "license_evidence_sha256")
    @classmethod
    def validate_candidate_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("source evidence candidate hashes must be SHA-256 hex")
        return normalized


class SourceEvidenceCandidateEntry(StrictModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    candidate_path: str
    candidate_sha256: str
    license_path: str
    license_sha256: str
    approval_template_path: str
    approval_template_sha256: str

    @field_validator("candidate_sha256", "license_sha256", "approval_template_sha256")
    @classmethod
    def validate_entry_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("evidence candidate entry hashes must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def validate_relative_paths(self) -> "SourceEvidenceCandidateEntry":
        for value in (
            self.candidate_path,
            self.license_path,
            self.approval_template_path,
        ):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or "\x00" in value:
                raise ValueError(
                    "evidence candidate manifest paths must be safe and relative"
                )
            if not relative.parts or relative.parts[0] != self.source_id:
                raise ValueError(
                    "evidence candidate manifest path does not belong to its source"
                )
        return self


class SourceEvidenceCandidateBundleReport(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["awaiting_human_legal_approval"] = "awaiting_human_legal_approval"
    source_count: int = Field(ge=1)
    source_registry_sha256: str
    origin_registry_sha256: str
    output_dir: str
    candidates: list[SourceEvidenceCandidateEntry]
    created_at_unix: float
    production_collection_authorized: Literal[False] = False

    @field_validator("source_registry_sha256", "origin_registry_sha256")
    @classmethod
    def validate_bundle_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("evidence candidate bundle hashes must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def validate_candidate_set(self) -> "SourceEvidenceCandidateBundleReport":
        source_ids = [entry.source_id for entry in self.candidates]
        if self.source_count != len(source_ids) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError(
                "evidence candidate bundle count or source uniqueness is invalid"
            )
        if source_ids != sorted(source_ids):
            raise ValueError(
                "evidence candidate bundle sources must be deterministically sorted"
            )
        return self


class SourceSelectionEntry(StrictModel):
    source_id: str
    registry_entry_sha256: str


class SourceSelectionManifest(StrictModel):
    schema_version: Literal[1] = 1
    source_count: int = Field(ge=1)
    source_ids: list[str] = Field(min_length=1)
    input_registry_sha256: str
    selected_registry_sha256: str
    entries: list[SourceSelectionEntry] = Field(min_length=1)

    @field_validator("input_registry_sha256", "selected_registry_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("selection manifest hashes must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def validate_selection(self) -> "SourceSelectionManifest":
        if self.source_count != len(self.source_ids) or self.source_count != len(
            self.entries
        ):
            raise ValueError("selection manifest count does not match selected sources")
        if self.source_ids != sorted(self.source_ids):
            raise ValueError("selection manifest source_ids must be sorted")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("selection manifest contains duplicate source IDs")
        if [entry.source_id for entry in self.entries] != self.source_ids:
            raise ValueError("selection manifest entries do not match source_ids")
        return self


def source_registry_entry_sha256(source: SourceSpec) -> str:
    payload = json.dumps(
        source.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_registry_snapshot_sha256(sources: list[SourceSpec]) -> str:
    payload = json.dumps(
        [
            source.model_dump(mode="json")
            for source in sorted(sources, key=lambda item: item.source_id or "")
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_evidence_origin_registry(path: str | Path) -> SourceEvidenceOriginRegistry:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > 1_048_576:
        raise ValueError(
            "source evidence origin registry must be a regular file no larger than 1 MiB"
        )
    return SourceEvidenceOriginRegistry.model_validate_json(
        source.read_text(encoding="utf-8-sig")
    )


def _resolve_public_git_revision(
    repository_url: str, revision: str, *, timeout_seconds: float = 30.0
) -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "git executable is required to resolve governed evidence revisions"
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"}
    }
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ASKPASS": "",
        }
    )
    completed = subprocess.run(  # nosec B603 - validated URL and fixed argv without shell
        [
            git,
            "-c",
            "credential.helper=",
            "ls-remote",
            "--exit-code",
            repository_url,
            revision,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=environment,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:500]
        raise RuntimeError(
            f"failed to resolve governed repository revision: {detail or completed.returncode}"
        )
    revisions = {
        line.split()[0].lower()
        for line in completed.stdout.splitlines()
        if line.strip() and len(line.split()) >= 2
    }
    if len(revisions) != 1:
        raise RuntimeError(
            "governed repository revision did not resolve to exactly one commit"
        )
    resolved = next(iter(revisions))
    if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise RuntimeError("governed repository revision is not a full SHA-1 commit")
    return resolved


def _fetch_governed_evidence_url(
    url: str,
    *,
    maximum_bytes: int = 1_048_576,
    timeout_seconds: float = 30.0,
) -> tuple[bytes, dict[str, str]]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "raw.githubusercontent.com",
        "www.nist.gov",
    }:
        raise ValueError("governed evidence fetch attempted an unapproved host")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError(
            "governed evidence URL cannot contain credentials, query, or fragment"
        )
    headers = {
        "Accept": "text/plain,text/html;q=0.9",
        "User-Agent": "Craftly-Governance-Evidence/1.0",
    }
    with httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout_seconds),
        trust_env=False,
        headers=headers,
    ) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"governed evidence fetch returned HTTP {response.status_code}"
                )
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > maximum_bytes:
                raise ValueError("governed evidence response exceeds the maximum size")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(
                        "governed evidence response exceeds the maximum size"
                    )
                chunks.append(chunk)
            payload = b"".join(chunks)
            if not payload or b"\x00" in payload:
                raise ValueError("governed evidence response is empty or binary")
            metadata = {
                key: value[:512]
                for key in ("content-type", "etag", "last-modified")
                if (value := response.headers.get(key))
            }
    return payload, metadata


def _raw_github_url(repository_url: str, revision: str, relative_path: str) -> str:
    parsed = urlparse(repository_url)
    segments = parsed.path.removesuffix(".git").strip("/").split("/")
    if len(segments) != 2:
        raise ValueError("governed GitHub repository URL must contain owner/repository")
    owner, repository = (quote(segment, safe="._-") for segment in segments)
    safe_path = "/".join(quote(part, safe="._-") for part in Path(relative_path).parts)
    return (
        f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{safe_path}"
    )


class SourceRegistry:
    """Validates and summarizes approved crawl sources.

    This keeps large data jobs governed: every URL must match the source
    allowlist, duplicate seeds are rejected, and license/category metadata is
    auditable before a crawler process starts.
    """

    def __init__(self, sources: list[SourceSpec]) -> None:
        self.sources = sources

    @classmethod
    def from_file(cls, path: str | Path) -> "SourceRegistry":
        return cls(load_sources(path))

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> "SourceRegistry":
        sources: list[SourceSpec] = []
        for path in paths:
            sources.extend(load_sources(path))
        return cls(sources)

    def select_sources(
        self,
        source_ids: list[str],
        *,
        expected_count: int | None = None,
    ) -> tuple["SourceRegistry", SourceSelectionManifest]:
        requested = [source_id.strip().lower() for source_id in source_ids]
        if not requested or any(not source_id for source_id in requested):
            raise ValueError(
                "source selection requires at least one non-empty source ID"
            )
        duplicates = sorted(
            {source_id for source_id in requested if requested.count(source_id) > 1}
        )
        if duplicates:
            raise ValueError("duplicate selected source IDs: " + ", ".join(duplicates))
        if expected_count is not None and expected_count != len(requested):
            raise ValueError(
                f"selected source count mismatch: expected {expected_count}, received {len(requested)}"
            )

        available: dict[str, SourceSpec] = {}
        for source in self.sources:
            assert source.source_id is not None
            if source.source_id in available:
                raise ValueError(
                    f"duplicate source_id in input registry: {source.source_id}"
                )
            available[source.source_id] = source
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError("unknown selected source IDs: " + ", ".join(unknown))

        selected_ids = sorted(requested)
        selected_sources = [available[source_id] for source_id in selected_ids]
        selected_registry = SourceRegistry(selected_sources)
        selected_registry.validate()
        manifest = SourceSelectionManifest(
            source_count=len(selected_sources),
            source_ids=selected_ids,
            input_registry_sha256=source_registry_snapshot_sha256(self.sources),
            selected_registry_sha256=source_registry_snapshot_sha256(selected_sources),
            entries=[
                SourceSelectionEntry(
                    source_id=source_id,
                    registry_entry_sha256=source_registry_entry_sha256(
                        available[source_id]
                    ),
                )
                for source_id in selected_ids
            ],
        )
        return selected_registry, manifest

    def validate(self, *, production: bool = False) -> SourceRegistryReport:
        warnings: list[str] = []
        seen_urls: set[str] = set()
        seen_ids: set[str] = set()
        domains: set[str] = set()
        categories: set[str] = set()
        url_count = 0
        approved_sources = 0
        quarantine_sources = 0

        for source in self.sources:
            assert source.source_id is not None
            if source.source_id in seen_ids:
                raise ValueError(f"duplicate source_id: {source.source_id}")
            seen_ids.add(source.source_id)
            if source.license.lower() not in APPROVED_LICENSES:
                warnings.append(
                    f"{source.name}: license '{source.license}' needs explicit legal approval"
                )
            if source.approval_status == "approved":
                approved_sources += 1
            else:
                quarantine_sources += 1
                warnings.append(
                    f"{source.name}: source remains {source.approval_status}/{source.trust_tier}"
                )
            if not source.allowed_domains:
                raise ValueError(f"{source.name}: allowed_domains cannot be empty")
            categories.add(source.category)
            for domain in source.allowed_domains:
                domains.add(domain.lower())
            for url in source.urls:
                url_count += 1
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError(f"{source.name}: unsupported URL scheme in {url}")
                if not allowed_url(url, source.allowed_domains):
                    raise ValueError(
                        f"{source.name}: URL outside allowed domains: {url}"
                    )
                if url in seen_urls:
                    warnings.append(
                        f"{source.name}: duplicate seed URL ignored by frontier: {url}"
                    )
                seen_urls.add(url)

        report = SourceRegistryReport(
            source_count=len(self.sources),
            url_count=url_count,
            domains=sorted(domains),
            categories=sorted(categories),
            approved_sources=approved_sources,
            quarantine_sources=quarantine_sources,
            source_snapshot_sha256=source_registry_snapshot_sha256(self.sources),
            warnings=warnings,
        )
        if production:
            blockers = self.production_blockers()
            if blockers:
                raise ValueError("; ".join(blockers))
        return report

    def production_blockers(self) -> list[str]:
        blockers: list[str] = []
        for source in self.sources:
            if source.approval_status != "approved":
                blockers.append(
                    f"{source.name}: production collection requires approval_status='approved'"
                )
            if source.trust_tier not in {"reviewed", "trusted"}:
                blockers.append(
                    f"{source.name}: production training source must be reviewed or trusted"
                )
            if source.approved_use == "evaluation_only":
                blockers.append(
                    f"{source.name}: evaluation-only source cannot enter production training collection"
                )
            if source.immutable_revision == "rolling":
                blockers.append(
                    f"{source.name}: production source requires an immutable_revision"
                )
            if (
                source.license_evidence_sha256 is None
                or source.legal_approval_sha256 is None
                or source.approval_request_sha256 is None
            ):
                blockers.append(
                    f"{source.name}: production source requires request, license, and legal approval evidence hashes"
                )
        return blockers

    def to_sources(self) -> list[SourceSpec]:
        return self.sources

    def approve_source(
        self,
        *,
        source_id: str,
        immutable_revision: str,
        license_evidence_path: str | Path,
        legal_approval_path: str | Path,
        trust_tier: str = "reviewed",
    ) -> SourceSpec:
        if immutable_revision.strip() in {"", "rolling"}:
            raise ValueError("approval requires an immutable source revision")
        if trust_tier not in {"reviewed", "trusted"}:
            raise ValueError("approved source trust tier must be reviewed or trusted")
        license_path = Path(license_evidence_path).resolve(strict=True)
        legal_path = Path(legal_approval_path).resolve(strict=True)
        if not license_path.is_file() or not legal_path.is_file():
            raise ValueError("approval evidence must be regular files")

        index = next(
            (
                position
                for position, source in enumerate(self.sources)
                if source.source_id == source_id
            ),
            None,
        )
        if index is None:
            raise KeyError(source_id)
        source = self.sources[index]
        license_evidence_sha256 = sha256_file(license_path)
        try:
            legal_payload = json.loads(legal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"legal approval evidence must be valid JSON: {exc.msg}"
            ) from exc
        legal_evidence = LegalApprovalEvidence.model_validate(legal_payload)
        expected_entry_sha256 = source_registry_entry_sha256(source)
        expected = {
            "source_id": source.source_id,
            "registry_entry_sha256": expected_entry_sha256,
            "immutable_revision": immutable_revision.strip(),
            "license": source.license.lower(),
            "license_evidence_sha256": license_evidence_sha256,
            "approved_use": source.approved_use,
        }
        actual = {
            "source_id": legal_evidence.source_id,
            "registry_entry_sha256": legal_evidence.registry_entry_sha256,
            "immutable_revision": legal_evidence.immutable_revision,
            "license": legal_evidence.license.lower(),
            "license_evidence_sha256": legal_evidence.license_evidence_sha256,
            "approved_use": legal_evidence.approved_use,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]
        if mismatches:
            raise ValueError(
                "legal approval evidence does not match source contract: "
                + ", ".join(mismatches)
            )
        approved = source.model_copy(
            update={
                "immutable_revision": immutable_revision.strip(),
                "license_evidence_sha256": license_evidence_sha256,
                "legal_approval_sha256": sha256_file(legal_path),
                "approval_request_sha256": expected_entry_sha256,
                "trust_tier": trust_tier,
                "approval_status": "approved",
            }
        )
        self.sources[index] = SourceSpec.model_validate(approved.model_dump())
        return self.sources[index]

    def verify_approval_evidence_directory(self, evidence_dir: str | Path) -> list[str]:
        """Re-verify stored approval claims against durable evidence files."""

        root = Path(evidence_dir).resolve()
        blockers: list[str] = []
        for source in self.sources:
            if source.approval_status != "approved" or source.source_id is None:
                continue
            source_root = (root / source.source_id).resolve()
            if root not in source_root.parents:
                blockers.append(f"{source.name}: evidence path escaped governance root")
                continue
            license_path = source_root / "license.txt"
            legal_path = source_root / "approval.json"
            if not license_path.is_file() or not legal_path.is_file():
                blockers.append(
                    f"{source.name}: expected evidence files {source.source_id}/license.txt and approval.json"
                )
                continue

            if sha256_file(license_path) != source.license_evidence_sha256:
                blockers.append(f"{source.name}: license evidence hash changed")
            if sha256_file(legal_path) != source.legal_approval_sha256:
                blockers.append(f"{source.name}: legal approval evidence hash changed")
                continue
            try:
                evidence = LegalApprovalEvidence.model_validate_json(
                    legal_path.read_text(encoding="utf-8")
                )
            except ValueError as exc:
                blockers.append(
                    f"{source.name}: invalid legal approval evidence: {exc}"
                )
                continue
            expected = {
                "source_id": source.source_id,
                "registry_entry_sha256": source.approval_request_sha256,
                "immutable_revision": source.immutable_revision,
                "license": source.license.lower(),
                "license_evidence_sha256": source.license_evidence_sha256,
                "approved_use": source.approved_use,
            }
            actual = {
                "source_id": evidence.source_id,
                "registry_entry_sha256": evidence.registry_entry_sha256,
                "immutable_revision": evidence.immutable_revision,
                "license": evidence.license.lower(),
                "license_evidence_sha256": evidence.license_evidence_sha256,
                "approved_use": evidence.approved_use,
            }
            mismatches = [
                key for key, value in expected.items() if actual[key] != value
            ]
            if mismatches:
                blockers.append(
                    f"{source.name}: legal approval contract mismatch: {', '.join(mismatches)}"
                )
        return blockers

    def prepare_approval_requests(self, output_dir: str | Path) -> list[Path]:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for source in sorted(self.sources, key=lambda item: item.source_id or ""):
            assert source.source_id is not None
            assert source.source_family is not None
            artifact = SourceApprovalRequestArtifact(
                source_id=source.source_id,
                source_name=source.name,
                source_family=source.source_family,
                registry_entry_sha256=source_registry_entry_sha256(source),
                current_license=source.license,
                approved_use=source.approved_use,
                seed_urls=source.urls,
                allowed_domains=source.allowed_domains,
                required_legal_fields=list(LegalApprovalEvidence.model_fields),
                evidence_paths={
                    "license": f"governance/source-approvals/{source.source_id}/license.txt",
                    "approval": f"governance/source-approvals/{source.source_id}/approval.json",
                },
                approval_template={
                    "schema_version": 1,
                    "approval_id": None,
                    "decision": "pending_human_review",
                    "source_id": source.source_id,
                    "registry_entry_sha256": source_registry_entry_sha256(source),
                    "immutable_revision": None,
                    "license": source.license,
                    "license_evidence_sha256": None,
                    "approved_use": source.approved_use,
                    "approved_by": None,
                    "approved_at": None,
                    "scope": (
                        "evaluation_only"
                        if source.approved_use == "evaluation_only"
                        else "training_collection"
                    ),
                    "rationale": None,
                },
                immutable_revision_requirements=[
                    "Use an upstream commit, release, publication revision, or immutable content snapshot.",
                    "Do not copy the literal value 'rolling' into the legal decision.",
                    "Verify that the reviewed content and license evidence correspond to that revision.",
                ],
                warning=(
                    "This request is not an approval. An authorized human must independently review "
                    "the license, immutable revision, intended use, and evidence before creating approval.json."
                ),
            )
            target = root / f"{source.source_id}.approval-request.json"
            target.write_text(
                json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            written.append(target)
        manifest = {
            "schema_version": 1,
            "request_count": len(written),
            "requests": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in written
            ],
        }
        (root / "approval-request-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return written

    def materialize_evidence_candidates(
        self,
        *,
        origin_registry_path: str | Path,
        output_dir: str | Path,
    ) -> SourceEvidenceCandidateBundleReport:
        """Fetch immutable official evidence without granting legal approval.

        The output deliberately contains ``approval.template.json`` rather than
        ``approval.json``. Only an authorized human can turn reviewed evidence
        into the latter file through the existing approval contract.
        """

        origins = load_evidence_origin_registry(origin_registry_path)
        source_by_id = {source.source_id: source for source in self.sources}
        origin_by_id = {origin.source_id: origin for origin in origins.sources}
        if None in source_by_id:
            raise ValueError(
                "every governed source requires a source_id before evidence materialization"
            )
        source_ids = set(source_by_id)
        origin_ids = set(origin_by_id)
        if source_ids != origin_ids:
            missing = sorted(source_ids - origin_ids)
            unexpected = sorted(origin_ids - source_ids)
            raise ValueError(
                "evidence origin registry must exactly match governed sources; "
                f"missing={missing}, unexpected={unexpected}"
            )

        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite governed evidence candidate directory: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        candidate_entries: list[SourceEvidenceCandidateEntry] = []
        try:
            for source_id in sorted(source_ids):
                source = source_by_id[source_id]
                assert source is not None
                origin = origin_by_id[source_id]
                if origin.expected_license.lower() != source.license.lower():
                    raise ValueError(
                        f"{source_id}: evidence origin license {origin.expected_license!r} "
                        f"does not match registry license {source.license!r}"
                    )

                if origin.origin_kind == "git_repository":
                    assert origin.repository_url is not None
                    assert origin.git_ref is not None
                    assert origin.license_path is not None
                    resolved_revision = _resolve_public_git_revision(
                        origin.repository_url, origin.git_ref
                    )
                    evidence_url = _raw_github_url(
                        origin.repository_url,
                        resolved_revision,
                        origin.license_path,
                    )
                    requested_revision = origin.git_ref
                else:
                    assert origin.license_url is not None
                    evidence_url = origin.license_url
                    requested_revision = "content-addressed-web-snapshot"
                    resolved_revision = ""

                evidence, response_metadata = _fetch_governed_evidence_url(evidence_url)
                evidence_sha256 = hashlib.sha256(evidence).hexdigest()
                if origin.origin_kind == "web_snapshot":
                    resolved_revision = f"sha256:{evidence_sha256}"

                final_source_dir = target / source_id
                temporary_source_dir = temporary / source_id
                temporary_source_dir.mkdir(parents=True, exist_ok=False)
                license_path = temporary_source_dir / "license.txt"
                license_path.write_bytes(evidence)
                candidate = SourceEvidenceCandidate(
                    source_id=source_id,
                    origin_kind=origin.origin_kind,
                    registry_entry_sha256=source_registry_entry_sha256(source),
                    expected_license=origin.expected_license,
                    requested_revision=requested_revision,
                    resolved_immutable_revision=resolved_revision,
                    license_source_url=evidence_url,
                    license_evidence_path=str(final_source_dir / "license.txt"),
                    license_evidence_sha256=evidence_sha256,
                    license_evidence_bytes=len(evidence),
                    response_metadata=response_metadata,
                    materialized_at_unix=time.time(),
                    warning=(
                        "Evidence candidate only. An authorized human must inspect source scope, third-party "
                        "content, attribution/share-alike obligations, and intended training use before approval."
                    ),
                )
                candidate_path = temporary_source_dir / "candidate.json"
                self._atomic_write_json(
                    candidate_path, candidate.model_dump(mode="json")
                )
                approval_template = {
                    "schema_version": 1,
                    "approval_id": None,
                    "decision": "pending_human_review",
                    "source_id": source_id,
                    "registry_entry_sha256": candidate.registry_entry_sha256,
                    "immutable_revision": resolved_revision,
                    "license": source.license,
                    "license_evidence_sha256": evidence_sha256,
                    "approved_use": source.approved_use,
                    "approved_by": None,
                    "approved_at": None,
                    "scope": "evaluation_only"
                    if source.approved_use == "evaluation_only"
                    else "training_collection",
                    "rationale": None,
                }
                template_path = temporary_source_dir / "approval.template.json"
                self._atomic_write_json(template_path, approval_template)
                candidate_entries.append(
                    SourceEvidenceCandidateEntry(
                        source_id=source_id,
                        candidate_path=f"{source_id}/candidate.json",
                        candidate_sha256=sha256_file(candidate_path),
                        license_path=f"{source_id}/license.txt",
                        license_sha256=evidence_sha256,
                        approval_template_path=f"{source_id}/approval.template.json",
                        approval_template_sha256=sha256_file(template_path),
                    )
                )

            origin_payload = json.dumps(
                origins.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            report = SourceEvidenceCandidateBundleReport(
                source_count=len(candidate_entries),
                source_registry_sha256=source_registry_snapshot_sha256(self.sources),
                origin_registry_sha256=hashlib.sha256(origin_payload).hexdigest(),
                output_dir=str(target),
                candidates=candidate_entries,
                created_at_unix=time.time(),
            )
            self._atomic_write_json(
                temporary / "evidence-candidate-manifest.json",
                report.model_dump(mode="json"),
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return report

    @staticmethod
    def write_selection_manifest(
        manifest: SourceSelectionManifest, path: str | Path
    ) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        SourceRegistry._atomic_write_json(target, manifest.model_dump(mode="json"))
        return target

    @staticmethod
    def _atomic_write_json(target: Path, payload: dict[str, object]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def write(
        self, path: str | Path, *, protect_existing_approvals: bool = True
    ) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if protect_existing_approvals and target.is_file():
            existing = SourceRegistry.from_file(target)
            incoming_by_id = {source.source_id: source for source in self.sources}
            for approved in existing.sources:
                if approved.approval_status != "approved":
                    continue
                incoming = incoming_by_id.get(approved.source_id)
                if incoming is None:
                    raise ValueError(
                        f"refusing to remove previously approved source: {approved.source_id}"
                    )
                if incoming.model_dump(mode="json") != approved.model_dump(mode="json"):
                    raise ValueError(
                        f"refusing to alter previously approved source: {approved.source_id}"
                    )
        payload = {"sources": [source.model_dump() for source in self.sources]}
        self._atomic_write_json(target, payload)
        return target


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate or merge Craftly defensive data source registries."
    )
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--approve-source")
    parser.add_argument("--immutable-revision")
    parser.add_argument("--license-evidence")
    parser.add_argument("--legal-approval")
    parser.add_argument(
        "--trust-tier", choices=["reviewed", "trusted"], default="reviewed"
    )
    parser.add_argument("--prepare-approval-dir")
    parser.add_argument("--evidence-origins")
    parser.add_argument("--materialize-evidence-candidates")
    parser.add_argument(
        "--select-source",
        action="append",
        default=[],
        help="Select an exact source ID into an isolated governed batch; repeat for each source.",
    )
    parser.add_argument("--expect-source-count", type=int)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--verify-evidence-dir")
    args = parser.parse_args()
    registry = SourceRegistry.from_files(args.sources)
    if args.select_source:
        registry, selection_manifest = registry.select_sources(
            args.select_source,
            expected_count=args.expect_source_count,
        )
        if args.selection_manifest:
            registry.write_selection_manifest(
                selection_manifest, args.selection_manifest
            )
    elif args.selection_manifest:
        parser.error("--selection-manifest requires at least one --select-source")
    elif (
        args.expect_source_count is not None
        and len(registry.sources) != args.expect_source_count
    ):
        parser.error(
            f"source count mismatch: expected {args.expect_source_count}, loaded {len(registry.sources)}"
        )
    if args.prepare_approval_dir:
        registry.prepare_approval_requests(args.prepare_approval_dir)
    candidate_bundle: SourceEvidenceCandidateBundleReport | None = None
    if args.materialize_evidence_candidates:
        if not args.evidence_origins:
            parser.error(
                "--materialize-evidence-candidates requires --evidence-origins"
            )
        candidate_bundle = registry.materialize_evidence_candidates(
            origin_registry_path=args.evidence_origins,
            output_dir=args.materialize_evidence_candidates,
        )
    elif args.evidence_origins:
        parser.error("--evidence-origins requires --materialize-evidence-candidates")
    if args.approve_source:
        required = {
            "--immutable-revision": args.immutable_revision,
            "--license-evidence": args.license_evidence,
            "--legal-approval": args.legal_approval,
            "--output": args.output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error("source approval requires " + ", ".join(missing))
        registry.approve_source(
            source_id=args.approve_source,
            immutable_revision=args.immutable_revision,
            license_evidence_path=args.license_evidence,
            legal_approval_path=args.legal_approval,
            trust_tier=args.trust_tier,
        )
    report = registry.validate(production=args.production)
    if args.production and not args.verify_evidence_dir:
        parser.error(
            "--production requires --verify-evidence-dir for durable evidence replay"
        )
    if args.verify_evidence_dir:
        evidence_blockers = registry.verify_approval_evidence_directory(
            args.verify_evidence_dir
        )
        if evidence_blockers:
            parser.error(
                "approval evidence verification failed: " + "; ".join(evidence_blockers)
            )
    if args.output:
        registry.write(args.output)
    payload: dict[str, Any] = report.model_dump(mode="json")
    if candidate_bundle is not None:
        payload["evidence_candidates"] = candidate_bundle.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
