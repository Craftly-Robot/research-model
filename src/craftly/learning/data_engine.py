"""Million-scale defensive data engine.

The engine is designed for large allowlisted crawls: persistent frontier,
resume/retry, per-domain throttling, URL discovery, provenance, content hash
deduplication, JSONL shard writing, and quality-gate integration.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from pydantic import Field

from src.craftly.learning.quality import DatasetQualityGate
from src.craftly.learning.web_ingest import (
    RobotsCache,
    SourceSpec,
    allowed_url,
    content_hash,
    load_sources,
    text_from_html,
)
from src.craftly.shared.progress import NullProgressReporter, ProgressReporter
from src.craftly.shared.schemas import StrictModel

LINK_RE = re.compile(r"""href=["']([^"'#]+)["']""", re.IGNORECASE)


FRONTIER_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS urls (
  url TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  allowed_domains_json TEXT NOT NULL,
  license TEXT NOT NULL,
  category TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  priority INTEGER NOT NULL DEFAULT 100,
  depth INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_fetch_at REAL NOT NULL DEFAULT 0,
  last_error TEXT,
  discovered_from TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  content_hash TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  source_name TEXT NOT NULL,
  license TEXT NOT NULL,
  category TEXT NOT NULL,
  text_chars INTEGER NOT NULL,
  accepted INTEGER NOT NULL,
  quality_json TEXT NOT NULL,
  shard_path TEXT,
  fetched_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_urls_status_next ON urls(status, next_fetch_at, priority);
CREATE INDEX IF NOT EXISTS idx_urls_source ON urls(source_name);
"""


POSTGRES_FRONTIER_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
  url TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  allowed_domains_json TEXT NOT NULL,
  license TEXT NOT NULL,
  category TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  priority INTEGER NOT NULL DEFAULT 100,
  depth INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_fetch_at DOUBLE PRECISION NOT NULL DEFAULT 0,
  last_error TEXT,
  discovered_from TEXT,
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  content_hash TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  source_name TEXT NOT NULL,
  license TEXT NOT NULL,
  category TEXT NOT NULL,
  text_chars INTEGER NOT NULL,
  accepted INTEGER NOT NULL,
  quality_json TEXT NOT NULL,
  shard_path TEXT,
  fetched_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_urls_status_next ON urls(status, next_fetch_at, priority);
CREATE INDEX IF NOT EXISTS idx_urls_source ON urls(source_name);
"""


class DataEngineConfig(StrictModel):
    frontier_backend: str = "sqlite"
    postgres_dsn: str | None = None
    frontier_path: str = "artifacts/craftly/data-engine/frontier.sqlite3"
    output_dir: str = "artifacts/craftly/data-engine/raw"
    clean_output_dir: str = "artifacts/craftly/data-engine/clean"
    max_docs: int = Field(default=10_000, ge=1)
    max_depth: int = Field(default=2, ge=0, le=20)
    workers: int = Field(default=8, ge=1, le=256)
    shard_rows: int = Field(default=10_000, ge=1)
    retry_limit: int = Field(default=3, ge=0, le=20)
    discover_links: bool = True
    respect_robots: bool = True
    delay_seconds: float = Field(default=1.0, ge=0.0)
    request_timeout_seconds: float = Field(default=20.0, ge=1.0)
    max_bytes_per_doc: int = Field(default=2_000_000, ge=1000)
    user_agent: str = "CraftlyResearchBot/0.2 defensive AI dataset builder"


class DataEngineReport(StrictModel):
    status: str
    frontier_path: str
    raw_output_dir: str
    clean_output_dir: str
    fetched: int
    accepted: int
    rejected: int
    discovered: int
    failed: int
    duplicate: int
    duration_ms: float


class ShardedJsonlWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        prefix: str,
        rows_per_shard: int,
        overwrite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in self.output_dir.glob(f"{self.prefix}-*.jsonl"):
                path.unlink()
        self.shard_index = 0
        self.row_count = 0
        self.handle = None
        self.current_path: Path | None = None
        self.lock = threading.RLock()

    def write(self, row: dict[str, Any]) -> Path:
        with self.lock:
            if self.handle is None or self.row_count >= self.rows_per_shard:
                self.close()
                path = self.output_dir / f"{self.prefix}-{self.shard_index:06d}.jsonl"
                self.handle = path.open("w", encoding="utf-8")
                self.current_path = path
                self.shard_index += 1
                self.row_count = 0
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            self.handle.write(line)
            self.handle.write("\n")
            self.handle.flush()
            self.row_count += 1
            if self.current_path is None:
                raise RuntimeError("writer current_path missing after write")
            return self.current_path

    def close(self) -> None:
        with self.lock:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


class FrontierStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(FRONTIER_SCHEMA)
        self.connection.commit()

    def seed(self, sources: list[SourceSpec]) -> int:
        inserted = 0
        now = time.time()
        for source in sources:
            for url in source.urls:
                normalized = normalize_url(url)
                if not allowed_url(normalized, source.allowed_domains):
                    continue
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO urls(
                      url, source_name, allowed_domains_json, license, category,
                      status, priority, depth, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'queued', 100, 0, ?, ?)
                    """,
                    (
                        normalized,
                        source.name,
                        json.dumps(source.allowed_domains, sort_keys=True),
                        source.license,
                        source.category,
                        now,
                        now,
                    ),
                )
                inserted += cursor.rowcount
        self.connection.commit()
        return inserted

    def claim(self, *, limit: int, now: float) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM urls
            WHERE status = 'queued' AND next_fetch_at <= ?
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        urls = [dict(row) for row in rows]
        for row in urls:
            self.connection.execute(
                "UPDATE urls SET status = 'in_progress', attempts = attempts + 1, updated_at = ? WHERE url = ?",
                (now, row["url"]),
            )
        self.connection.commit()
        return urls

    def enqueue_discovered(
        self, *, parent: dict[str, Any], urls: list[str], max_depth: int
    ) -> int:
        if int(parent["depth"]) >= max_depth:
            return 0
        allowed_domains = json.loads(parent["allowed_domains_json"])
        now = time.time()
        inserted = 0
        for url in urls:
            normalized = normalize_url(url)
            if not allowed_url(normalized, allowed_domains):
                continue
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO urls(
                  url, source_name, allowed_domains_json, license, category,
                  status, priority, depth, discovered_from, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    normalized,
                    parent["source_name"],
                    parent["allowed_domains_json"],
                    parent["license"],
                    parent["category"],
                    int(parent["priority"]) + 10,
                    int(parent["depth"]) + 1,
                    parent["url"],
                    now,
                    now,
                ),
            )
            inserted += cursor.rowcount
        self.connection.commit()
        return inserted

    def mark_done(self, url: str) -> None:
        self.connection.execute(
            "UPDATE urls SET status = 'done', updated_at = ? WHERE url = ?",
            (time.time(), url),
        )
        self.connection.commit()

    def mark_failed(
        self, row: dict[str, Any], error: str, *, retry_limit: int, delay_seconds: float
    ) -> None:
        attempts = int(row["attempts"]) + 1
        status = "failed" if attempts >= retry_limit else "queued"
        next_fetch_at = time.time() + delay_seconds * max(1, attempts)
        self.connection.execute(
            "UPDATE urls SET status = ?, last_error = ?, next_fetch_at = ?, updated_at = ? WHERE url = ?",
            (status, error[:1000], next_fetch_at, time.time(), row["url"]),
        )
        self.connection.commit()

    def record_document(
        self, row: dict[str, Any], *, quality: dict[str, Any], shard_path: str | None
    ) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO documents(
              content_hash, url, source_name, license, category, text_chars,
              accepted, quality_json, shard_path, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["content_hash"],
                row["url"],
                row["source"],
                row["license"],
                row["category"],
                len(row.get("text", "")),
                1 if quality.get("accepted") else 0,
                json.dumps(quality, sort_keys=True),
                shard_path,
                time.time(),
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM urls GROUP BY status"
        ):
            stats[f"urls_{row['status']}"] = int(row["count"])
        for row in self.connection.execute(
            "SELECT accepted, COUNT(*) AS count FROM documents GROUP BY accepted"
        ):
            stats[f"documents_{'accepted' if row['accepted'] else 'rejected'}"] = int(
                row["count"]
            )
        return stats

    def close(self) -> None:
        self.connection.close()


def _asyncpg_inserted(status: str) -> bool:
    return status.endswith(" 1")


class PostgresFrontierStore:
    """Postgres frontier for distributed production crawlers.

    Use `await PostgresFrontierStore.create(dsn)` and pass it into `DataEngine`.
    Claims use row locks with `SKIP LOCKED`, so many crawler processes can share
    one frontier without claiming the same URL twice.
    """

    def __init__(self, pool: Any, *, dsn_label: str = "postgres") -> None:
        self.pool = pool
        self.path = dsn_label

    @classmethod
    async def create(
        cls, dsn: str, *, min_size: int = 1, max_size: int = 10
    ) -> "PostgresFrontierStore":
        try:
            import asyncpg
        except (
            ImportError
        ) as exc:  # pragma: no cover - dependency is optional in CPU-only tests
            raise RuntimeError(
                "asyncpg is required for Postgres frontier support"
            ) from exc
        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        store = cls(pool)
        async with pool.acquire() as connection:
            await connection.execute(POSTGRES_FRONTIER_SCHEMA)
        return store

    async def seed(self, sources: list[SourceSpec]) -> int:
        inserted = 0
        now = time.time()
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                for source in sources:
                    for url in source.urls:
                        normalized = normalize_url(url)
                        if not allowed_url(normalized, source.allowed_domains):
                            continue
                        status = await connection.execute(
                            """
                            INSERT INTO urls(
                              url, source_name, allowed_domains_json, license, category,
                              status, priority, depth, created_at, updated_at
                            )
                            VALUES($1, $2, $3, $4, $5, 'queued', 100, 0, $6, $7)
                            ON CONFLICT(url) DO NOTHING
                            """,
                            normalized,
                            source.name,
                            json.dumps(source.allowed_domains, sort_keys=True),
                            source.license,
                            source.category,
                            now,
                            now,
                        )
                        inserted += 1 if _asyncpg_inserted(status) else 0
        return inserted

    async def claim(self, *, limit: int, now: float) -> list[dict[str, Any]]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                WITH claimed AS (
                  SELECT url FROM urls
                  WHERE status = 'queued' AND next_fetch_at <= $1
                  ORDER BY priority ASC, created_at ASC
                  FOR UPDATE SKIP LOCKED
                  LIMIT $2
                )
                UPDATE urls
                SET status = 'in_progress', attempts = attempts + 1, updated_at = $1
                FROM claimed
                WHERE urls.url = claimed.url
                RETURNING urls.*
                """,
                now,
                limit,
            )
        return [dict(row) for row in rows]

    async def enqueue_discovered(
        self, *, parent: dict[str, Any], urls: list[str], max_depth: int
    ) -> int:
        if int(parent["depth"]) >= max_depth:
            return 0
        allowed_domains = json.loads(parent["allowed_domains_json"])
        now = time.time()
        inserted = 0
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                for url in urls:
                    normalized = normalize_url(url)
                    if not allowed_url(normalized, allowed_domains):
                        continue
                    status = await connection.execute(  # nosemgrep: python.lang.security.audit.sqli.asyncpg-sqli.asyncpg-sqli
                        """
                        INSERT INTO urls(
                          url, source_name, allowed_domains_json, license, category,
                          status, priority, depth, discovered_from, created_at, updated_at
                        )
                        VALUES($1, $2, $3, $4, $5, 'queued', $6, $7, $8, $9, $10)
                        ON CONFLICT(url) DO NOTHING
                        """,
                        normalized,
                        parent["source_name"],
                        parent["allowed_domains_json"],
                        parent["license"],
                        parent["category"],
                        int(parent["priority"]) + 10,
                        int(parent["depth"]) + 1,
                        parent["url"],
                        now,
                        now,
                    )
                    inserted += 1 if _asyncpg_inserted(status) else 0
        return inserted

    async def mark_done(self, url: str) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                "UPDATE urls SET status = 'done', updated_at = $1 WHERE url = $2",
                time.time(),
                url,
            )

    async def mark_failed(
        self, row: dict[str, Any], error: str, *, retry_limit: int, delay_seconds: float
    ) -> None:
        attempts = int(row["attempts"]) + 1
        status = "failed" if attempts >= retry_limit else "queued"
        next_fetch_at = time.time() + delay_seconds * max(1, attempts)
        async with self.pool.acquire() as connection:
            await connection.execute(
                "UPDATE urls SET status = $1, last_error = $2, next_fetch_at = $3, updated_at = $4 WHERE url = $5",
                status,
                error[:1000],
                next_fetch_at,
                time.time(),
                row["url"],
            )

    async def record_document(
        self, row: dict[str, Any], *, quality: dict[str, Any], shard_path: str | None
    ) -> bool:
        async with self.pool.acquire() as connection:
            status = await connection.execute(  # nosemgrep: python.lang.security.audit.sqli.asyncpg-sqli.asyncpg-sqli
                """
                INSERT INTO documents(
                  content_hash, url, source_name, license, category, text_chars,
                  accepted, quality_json, shard_path, fetched_at
                )
                VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT(content_hash) DO NOTHING
                """,
                row["content_hash"],
                row["url"],
                row["source"],
                row["license"],
                row["category"],
                len(row.get("text", "")),
                1 if quality.get("accepted") else 0,
                json.dumps(quality, sort_keys=True),
                shard_path,
                time.time(),
            )
        return _asyncpg_inserted(status)

    async def stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        async with self.pool.acquire() as connection:
            for row in await connection.fetch(
                "SELECT status, COUNT(*) AS count FROM urls GROUP BY status"
            ):
                stats[f"urls_{row['status']}"] = int(row["count"])
            for row in await connection.fetch(
                "SELECT accepted, COUNT(*) AS count FROM documents GROUP BY accepted"
            ):
                stats[f"documents_{'accepted' if row['accepted'] else 'rejected'}"] = (
                    int(row["count"])
                )
        return stats

    async def close(self) -> None:
        await self.pool.close()


class DomainThrottle:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_fetch: dict[str, float] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with lock:
            elapsed = time.time() - self.last_fetch.get(host, 0.0)
            if elapsed < self.delay_seconds:
                await asyncio.sleep(self.delay_seconds - elapsed)
            self.last_fetch[host] = time.time()


def normalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url)
    return clean.strip()


def discover_links(base_url: str, html: str) -> list[str]:
    output: list[str] = []
    for match in LINK_RE.finditer(html):
        href = match.group(1).strip()
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        output.append(normalize_url(urljoin(base_url, href)))
    return output


def is_supported_text_response(content_type: str, url: str) -> bool:
    lowered_type = content_type.lower().split(";", 1)[0].strip()
    lowered_url = url.lower()
    if not lowered_type:
        return not lowered_url.endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".ico",
                ".pdf",
                ".zip",
                ".gz",
                ".tar",
                ".woff",
                ".woff2",
                ".ttf",
                ".mp4",
                ".mp3",
            )
        )
    if lowered_type.startswith("text/"):
        return True
    return lowered_type in {
        "application/json",
        "application/ld+json",
        "application/xml",
        "application/xhtml+xml",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "application/x-ndjson",
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _store_call(store: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(store, method_name)
    return await _maybe_await(method(*args, **kwargs))


class DataEngine:
    def __init__(
        self,
        config: DataEngineConfig | None = None,
        *,
        store: Any | None = None,
        owns_store: bool | None = None,
    ) -> None:
        self.config = config or DataEngineConfig()
        self.store = store or FrontierStore(self.config.frontier_path)
        self.owns_store = store is None if owns_store is None else owns_store
        self.quality_gate = DatasetQualityGate()
        self.raw_writer = ShardedJsonlWriter(
            self.config.output_dir, prefix="raw", rows_per_shard=self.config.shard_rows
        )
        self.clean_writer = ShardedJsonlWriter(
            self.config.clean_output_dir,
            prefix="clean",
            rows_per_shard=self.config.shard_rows,
        )

    def close(self) -> None:
        self.raw_writer.close()
        self.clean_writer.close()
        if self.owns_store:
            result = self.store.close()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)

    async def aclose(self) -> None:
        self.raw_writer.close()
        self.clean_writer.close()
        if self.owns_store:
            await _maybe_await(self.store.close())

    async def run(
        self,
        sources: list[SourceSpec],
        *,
        client: httpx.AsyncClient | None = None,
        progress: ProgressReporter | None = None,
        progress_every_docs: int = 25,
    ) -> DataEngineReport:
        started = time.perf_counter()
        active_progress = progress or NullProgressReporter()
        active_progress.emit(
            "crawl",
            "started",
            sources=len(sources),
            max_docs=self.config.max_docs,
            workers=self.config.workers,
            max_depth=self.config.max_depth,
        )
        await _store_call(self.store, "seed", sources)
        throttle = DomainThrottle(self.config.delay_seconds)
        headers = {"User-Agent": self.config.user_agent}
        own_client = client is None
        active_client = client or httpx.AsyncClient(
            headers=headers,
            timeout=self.config.request_timeout_seconds,
            follow_redirects=True,
        )
        robots = RobotsCache(self.config.user_agent)
        counters = {
            "fetched": 0,
            "accepted": 0,
            "rejected": 0,
            "discovered": 0,
            "failed": 0,
            "duplicate": 0,
        }
        queue_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        last_progress_docs = 0
        last_progress_time = 0.0

        async def emit_crawl_progress(force: bool = False) -> None:
            nonlocal last_progress_docs, last_progress_time
            async with progress_lock:
                now = time.time()
                if (
                    force
                    or counters["fetched"] - last_progress_docs
                    >= max(1, progress_every_docs)
                    or now - last_progress_time >= 10.0
                ):
                    last_progress_docs = counters["fetched"]
                    last_progress_time = now
                    active_progress.emit(
                        "crawl",
                        "running",
                        fetched=counters["fetched"],
                        accepted=counters["accepted"],
                        rejected=counters["rejected"],
                        discovered=counters["discovered"],
                        failed=counters["failed"],
                        duplicate=counters["duplicate"],
                        max_docs=self.config.max_docs,
                    )

        async def worker() -> None:
            while counters["fetched"] < self.config.max_docs:
                async with queue_lock:
                    claimed = await _store_call(
                        self.store, "claim", limit=1, now=time.time()
                    )
                if not claimed:
                    return
                item = claimed[0]
                try:
                    await throttle.wait(item["url"])
                    if self.config.respect_robots and not await robots.allowed(
                        active_client, item["url"]
                    ):
                        await _store_call(
                            self.store,
                            "mark_failed",
                            item,
                            "robots_disallow",
                            retry_limit=0,
                            delay_seconds=self.config.delay_seconds,
                        )
                        counters["rejected"] += 1
                        continue
                    response = await active_client.get(item["url"])
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if not is_supported_text_response(content_type, item["url"]):
                        await _store_call(self.store, "mark_done", item["url"])
                        counters["rejected"] += 1
                        counters["fetched"] += 1
                        continue
                    raw = response.text[: self.config.max_bytes_per_doc]
                    text = (
                        text_from_html(raw)
                        if "html" in content_type.lower() or "<html" in raw.lower()
                        else raw
                    )
                    row = {
                        "source": item["source_name"],
                        "url": item["url"],
                        "license": item["license"],
                        "category": item["category"],
                        "text": text,
                        "content_hash": content_hash(text),
                        "fetched_at_unix": time.time(),
                        "metadata": {
                            "content_type": content_type,
                            "status_code": response.status_code,
                            "depth": item["depth"],
                        },
                    }
                    self.raw_writer.write(row)
                    decision = self.quality_gate.evaluate(row)
                    clean_path = None
                    if decision.accepted:
                        row["quality"] = decision.model_dump()
                        clean_path = self.clean_writer.write(row)
                        counters["accepted"] += 1
                    else:
                        counters["rejected"] += 1
                    recorded = await _store_call(
                        self.store,
                        "record_document",
                        row,
                        quality=decision.model_dump(),
                        shard_path=str(clean_path) if clean_path else None,
                    )
                    if not recorded:
                        counters["duplicate"] += 1
                    if self.config.discover_links and "html" in content_type.lower():
                        counters["discovered"] += await _store_call(
                            self.store,
                            "enqueue_discovered",
                            parent=item,
                            urls=discover_links(item["url"], raw),
                            max_depth=self.config.max_depth,
                        )
                    await _store_call(self.store, "mark_done", item["url"])
                    counters["fetched"] += 1
                    await emit_crawl_progress()
                except Exception as exc:
                    counters["failed"] += 1
                    await _store_call(
                        self.store,
                        "mark_failed",
                        item,
                        str(exc),
                        retry_limit=self.config.retry_limit,
                        delay_seconds=self.config.delay_seconds,
                    )
                    await emit_crawl_progress()

        try:
            await asyncio.gather(*(worker() for _ in range(self.config.workers)))
        finally:
            self.raw_writer.close()
            self.clean_writer.close()
            if own_client:
                await active_client.aclose()

        active_progress.emit(
            "crawl",
            "complete",
            **counters,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return DataEngineReport(
            status="complete",
            frontier_path=str(self.store.path),
            raw_output_dir=self.config.output_dir,
            clean_output_dir=self.config.clean_output_dir,
            duration_ms=(time.perf_counter() - started) * 1000,
            **counters,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Craftly million-scale defensive data engine."
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument(
        "--frontier-backend", choices=["sqlite", "postgres"], default="sqlite"
    )
    parser.add_argument("--postgres-dsn")
    parser.add_argument(
        "--frontier", default="artifacts/craftly/data-engine/frontier.sqlite3"
    )
    parser.add_argument("--raw-output-dir", default="artifacts/craftly/data-engine/raw")
    parser.add_argument(
        "--clean-output-dir", default="artifacts/craftly/data-engine/clean"
    )
    parser.add_argument("--max-docs", type=int, default=10_000)
    parser.add_argument("--max-bytes-per-doc", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--shard-rows", type=int, default=10_000)
    parser.add_argument("--ignore-robots", action="store_true")
    return parser.parse_args()


async def build_store(args: argparse.Namespace) -> Any:
    if args.frontier_backend == "postgres":
        if not args.postgres_dsn:
            raise ValueError(
                "--postgres-dsn is required when --frontier-backend postgres"
            )
        return await PostgresFrontierStore.create(args.postgres_dsn)
    return FrontierStore(args.frontier)


async def run_cli(args: argparse.Namespace) -> DataEngineReport:
    store = await build_store(args)
    config = DataEngineConfig(
        frontier_backend=args.frontier_backend,
        postgres_dsn=args.postgres_dsn,
        frontier_path=args.frontier,
        output_dir=args.raw_output_dir,
        clean_output_dir=args.clean_output_dir,
        max_docs=args.max_docs,
        max_bytes_per_doc=args.max_bytes_per_doc,
        workers=args.workers,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        shard_rows=args.shard_rows,
        respect_robots=not args.ignore_robots,
    )
    engine = DataEngine(config, store=store, owns_store=True)
    try:
        return await engine.run(load_sources(args.sources))
    finally:
        await engine.aclose()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_cli(args))
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
