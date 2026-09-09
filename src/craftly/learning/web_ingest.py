"""Allowlisted web corpus ingestion for defensive coding/security data."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import Field, field_validator, model_validator

from src.craftly.shared.schemas import StrictModel

TAG_RE = re.compile(r"<(script|style).*?</\1>", re.IGNORECASE | re.DOTALL)
HTML_RE = re.compile(r"<[^>]+>")


class SourceSpec(StrictModel):
    name: str
    source_id: str | None = None
    urls: list[str] = Field(min_length=1)
    allowed_domains: list[str] = Field(min_length=1)
    license: str = "unknown-ok"
    category: str = "defensive_security"
    source_family: str | None = None
    trust_tier: Literal["quarantine", "reviewed", "trusted", "protected_holdout"] = "quarantine"
    approved_use: Literal["foundation", "defensive", "authorized_lab", "evaluation_only"] = (
        "defensive"
    )
    approval_status: Literal["pending", "approved", "rejected", "expired"] = "pending"
    immutable_revision: str = "rolling"
    license_evidence_sha256: str | None = None
    legal_approval_sha256: str | None = None
    approval_request_sha256: str | None = None
    allowed_content_types: list[str] = Field(
        default_factory=lambda: [
            "application/json",
            "application/xml",
            "text/html",
            "text/markdown",
            "text/plain",
            "text/x-python",
        ]
    )
    collection_budget: int = Field(default=1_000, ge=1, le=10_000_000)

    @field_validator("source_id", "source_family")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,127}", normalized) is None:
            raise ValueError("source identifiers must be lowercase slugs")
        return normalized

    @field_validator("license_evidence_sha256", "legal_approval_sha256", "approval_request_sha256")
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
            raise ValueError("approval evidence hashes must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def finalize_identity(self) -> SourceSpec:
        if self.source_id is None:
            object.__setattr__(
                self,
                "source_id",
                re.sub(r"[^a-z0-9._-]+", "-", self.name.strip().lower()).strip("-"),
            )
        if self.source_family is None:
            object.__setattr__(self, "source_family", self.source_id)
        if not self.source_id or not self.source_family:
            raise ValueError("source_id and source_family cannot be empty")
        if self.trust_tier == "protected_holdout" and self.approved_use != "evaluation_only":
            raise ValueError("protected holdout sources must be evaluation_only")
        return self


class CrawlConfig(StrictModel):
    user_agent: str = "CraftlyResearchBot/0.1 defensive AI dataset builder"
    request_timeout_seconds: float = 20.0
    delay_seconds: float = Field(default=1.0, ge=0.0)
    max_docs: int = Field(default=100, ge=1)
    max_bytes_per_doc: int = Field(default=2_000_000, ge=1000)
    respect_robots: bool = True


class IngestReport(StrictModel):
    output_path: str
    fetched: int
    rejected: int
    errors: list[str] = Field(default_factory=list)
    created_at_unix: float = Field(default_factory=time.time)


def allowed_url(url: str, domains: list[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(host == domain.lower() or host.endswith("." + domain.lower()) for domain in domains)


def text_from_html(raw: str) -> str:
    without_blocks = TAG_RE.sub(" ", raw)
    text = HTML_RE.sub(" ", without_blocks)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class RobotsCache:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self.parsers:
            parser = RobotFileParser()
            parser.set_url(f"{root}/robots.txt")
            try:
                response = await client.get(f"{root}/robots.txt")
                parser.parse(response.text.splitlines())
            except Exception:
                parser.parse([])
            self.parsers[root] = parser
        return self.parsers[root].can_fetch(self.user_agent, url)


class WebCorpusIngestor:
    def __init__(self, config: CrawlConfig | None = None) -> None:
        self.config = config or CrawlConfig()

    async def ingest(self, sources: list[SourceSpec], output_path: str | Path) -> IngestReport:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fetched = rejected = 0
        errors: list[str] = []
        robots = RobotsCache(self.config.user_agent)
        headers = {"User-Agent": self.config.user_agent}
        async with httpx.AsyncClient(
            headers=headers, timeout=self.config.request_timeout_seconds, follow_redirects=True
        ) as client:
            with target.open("w", encoding="utf-8") as handle:
                for source in sources:
                    for url in source.urls:
                        if fetched >= self.config.max_docs:
                            break
                        if not allowed_url(url, source.allowed_domains):
                            rejected += 1
                            errors.append(f"domain_not_allowed:{url}")
                            continue
                        if self.config.respect_robots and not await robots.allowed(client, url):
                            rejected += 1
                            errors.append(f"robots_disallow:{url}")
                            continue
                        await asyncio.sleep(self.config.delay_seconds)
                        try:
                            row = await self.fetch_one(client, source, url)
                        except Exception as exc:
                            rejected += 1
                            errors.append(f"fetch_error:{url}:{exc}")
                            continue
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        fetched += 1
        return IngestReport(
            output_path=str(target), fetched=fetched, rejected=rejected, errors=errors
        )

    async def fetch_one(
        self, client: httpx.AsyncClient, source: SourceSpec, url: str
    ) -> dict[str, Any]:
        response = await client.get(url)
        response.raise_for_status()
        raw = response.text[: self.config.max_bytes_per_doc]
        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type and media_type not in source.allowed_content_types:
            raise ValueError(f"content_type_not_allowed:{media_type}")
        text = (
            text_from_html(raw) if "html" in content_type.lower() or "<html" in raw.lower() else raw
        )
        raw_hash = content_hash(raw)
        canonical_hash = content_hash(text)
        snapshot_payload = {
            "source_id": source.source_id,
            "source_family": source.source_family,
            "source_url": url,
            "immutable_revision": source.immutable_revision,
            "license_evidence_sha256": source.license_evidence_sha256,
            "legal_approval_sha256": source.legal_approval_sha256,
            "raw_content_sha256": raw_hash,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        }
        source_snapshot_sha256 = content_hash(
            json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        return {
            "source": source.name,
            "source_id": source.source_id,
            "source_family": source.source_family,
            "url": url,
            "license": source.license,
            "category": source.category,
            "text": text,
            "content_hash": canonical_hash,
            "fetched_at_unix": time.time(),
            "provenance": {
                "source_id": source.source_id,
                "source_family": source.source_family,
                "source_url": url,
                "immutable_revision": source.immutable_revision,
                "license": source.license,
                "license_evidence_sha256": source.license_evidence_sha256,
                "legal_approval_sha256": source.legal_approval_sha256,
                "approval_status": source.approval_status,
                "approved_use": source.approved_use,
                "trust_tier": source.trust_tier,
                "raw_content_sha256": raw_hash,
                "canonical_content_sha256": canonical_hash,
                "source_snapshot_sha256": source_snapshot_sha256,
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
            },
            "metadata": {"content_type": content_type, "status_code": response.status_code},
        }


def load_sources(path: str | Path) -> list[SourceSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [SourceSpec.model_validate(item) for item in payload["sources"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest allowlisted defensive web corpus.")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = WebCorpusIngestor(
        CrawlConfig(max_docs=args.max_docs, delay_seconds=args.delay_seconds)
    )
    report = asyncio.run(ingestor.ingest(load_sources(args.sources), args.output))
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
