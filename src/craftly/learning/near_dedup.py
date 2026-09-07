"""Memory-bounded exact, structural, lineage, and near-duplicate filtering."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sqlite3
import time
import tracemalloc
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.shared.schemas import StrictModel


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+")
COMMENT_RE = re.compile(r"(?m)(^\s*#.*$|//.*$|/\*.*?\*/)", re.DOTALL)
LSH_BANDS = 4
LSH_BAND_BITS = 16
SQLITE_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;
CREATE TABLE IF NOT EXISTS exact_hashes (
  content_hash TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS structure_hashes (
  structure_hash TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS lineage_hashes (
  lineage_hash TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS fingerprints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  simhash_hex TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fingerprint_bands (
  band_key TEXT NOT NULL,
  fingerprint_id INTEGER NOT NULL REFERENCES fingerprints(id) ON DELETE CASCADE,
  PRIMARY KEY (band_key, fingerprint_id)
);
CREATE INDEX IF NOT EXISTS idx_fingerprint_bands_key ON fingerprint_bands(band_key);
"""


class NearDedupReport(StrictModel):
    input_paths: list[str]
    output_path: str
    index_path: str
    accepted: int
    exact_duplicates: int
    structural_duplicates: int = 0
    lineage_duplicates: int = 0
    near_duplicates: int
    candidate_comparisons: int = 0
    by_source: dict[str, dict[str, int]] = Field(default_factory=dict)
    hamming_threshold: int
    lsh_bands: int = LSH_BANDS
    memory_bounded: bool = True
    created_at_unix: float = Field(default_factory=time.time)


class DedupScaleValidationReport(StrictModel):
    requested_records: int
    indexed_records: int
    candidate_comparisons: int
    index_path: str
    peak_python_memory_bytes: int
    duration_seconds: float
    comparisons_per_record: float
    memory_bounded: bool
    status: str


class DistributedDedupDecision(StrictModel):
    accepted: bool
    reason: str
    candidate_comparisons: int = 0
    content_hash: str
    structure_hash: str | None = None
    lineage_hash: str | None = None
    simhash64: str


def _fingerprint_tokens(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return tokens or text.lower().split()


def simhash64(text: str) -> int:
    weights = [0] * 64
    for token in _fingerprint_tokens(text):
        digest = int.from_bytes(hashlib.blake2b(token.encode("utf-8", "replace"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def normalized_structure_hash(text: str, language_hint: str | None) -> str | None:
    """Return an exact structural fingerprint without executing or importing code."""

    normalized_language = (language_hint or "").lower()
    if normalized_language == "python":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, MemoryError):
            return None
        payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return stable_hash(f"python-ast:{payload}")
    if normalized_language in {"c", "cpp", "c_cpp", "c_cpp_header", "go", "java", "javascript", "rust", "solidity", "typescript"}:
        without_comments = COMMENT_RE.sub(" ", text)
        tokens = TOKEN_RE.findall(without_comments)
        if len(tokens) < 8:
            return None
        return stable_hash(f"{normalized_language}-tokens:{' '.join(tokens)}")
    return None


def _lineage_hash(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    values = {
        "patch_lineage": row.get("patch_lineage") or metadata.get("patch_lineage"),
        "task_signature": row.get("task_signature") or metadata.get("task_signature"),
        "generated_from": row.get("generated_from") or metadata.get("generated_from"),
        "source_revision": provenance.get("immutable_revision") or metadata.get("source_revision"),
    }
    material = {key: str(value) for key, value in values.items() if value not in {None, ""}}
    if not any(key in material for key in ("patch_lineage", "task_signature", "generated_from")):
        return None
    return stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":")))


def _band_keys(value: int) -> list[str]:
    mask = (1 << LSH_BAND_BITS) - 1
    return [f"{band}:{(value >> (band * LSH_BAND_BITS)) & mask:04x}" for band in range(LSH_BANDS)]


class SQLiteLSHDedupIndex:
    """Persistent streaming index; memory use does not grow with corpus size."""

    def __init__(self, path: str | Path, *, reset: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SQLITE_SCHEMA)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def begin_candidate(self) -> None:
        self.connection.execute("SAVEPOINT dedup_candidate")

    def accept_candidate(self) -> None:
        self.connection.execute("RELEASE SAVEPOINT dedup_candidate")

    def reject_candidate(self) -> None:
        self.connection.execute("ROLLBACK TO SAVEPOINT dedup_candidate")
        self.connection.execute("RELEASE SAVEPOINT dedup_candidate")

    def _insert_unique(self, table: str, column: str, value: str | None) -> bool:
        if value is None:
            return True
        statements = {
            ("exact_hashes", "content_hash"): "INSERT OR IGNORE INTO exact_hashes(content_hash) VALUES (?)",
            ("structure_hashes", "structure_hash"): "INSERT OR IGNORE INTO structure_hashes(structure_hash) VALUES (?)",
            ("lineage_hashes", "lineage_hash"): "INSERT OR IGNORE INTO lineage_hashes(lineage_hash) VALUES (?)",
        }
        try:
            statement = statements[(table, column)]
        except KeyError as exc:
            raise ValueError(f"unsupported deduplication index: {table}.{column}") from exc
        cursor = self.connection.execute(statement, (value,))
        return cursor.rowcount == 1

    def add_exact(self, digest: str) -> bool:
        return self._insert_unique("exact_hashes", "content_hash", digest)

    def add_structure(self, digest: str | None) -> bool:
        return self._insert_unique("structure_hashes", "structure_hash", digest)

    def add_lineage(self, digest: str | None) -> bool:
        return self._insert_unique("lineage_hashes", "lineage_hash", digest)

    def near_duplicate(self, value: int, threshold: int) -> tuple[bool, int]:
        if threshold >= 64:
            rows = self.connection.execute("SELECT simhash_hex FROM fingerprints")
        else:
            candidate_hashes: set[str] = set()
            for key in _band_keys(value):
                matches = self.connection.execute(
                    """
                SELECT DISTINCT f.simhash_hex
                FROM fingerprint_bands b
                JOIN fingerprints f ON f.id = b.fingerprint_id
                    WHERE b.band_key=?
                """,
                    (key,),
                )
                candidate_hashes.update(str(match[0]) for match in matches)
            rows = ((candidate_hash,) for candidate_hash in candidate_hashes)
        comparisons = 0
        for (stored_hex,) in rows:
            comparisons += 1
            if hamming_distance(value, int(stored_hex, 16)) <= threshold:
                return True, comparisons
        return False, comparisons

    def add_fingerprint(self, value: int) -> None:
        cursor = self.connection.execute("INSERT INTO fingerprints(simhash_hex) VALUES (?)", (f"{value:016x}",))
        fingerprint_id = int(cursor.lastrowid)
        self.connection.executemany(
            "INSERT INTO fingerprint_bands(band_key,fingerprint_id) VALUES (?,?)",
            [(key, fingerprint_id) for key in _band_keys(value)],
        )


class PostgresLSHDedupIndex:
    """Concurrency-safe distributed LSH index backed by migration 0006.

    Candidate bands are protected by transaction-scoped Postgres advisory
    locks. This prevents two workers with overlapping LSH buckets from racing
    through the near-duplicate check before either fingerprint is committed.
    """

    def __init__(self, dsn: str, *, dataset_version: str, pool_size: int = 10) -> None:
        if not dsn.startswith(("postgres://", "postgresql://")):
            raise ValueError("Postgres LSH index requires a PostgreSQL DSN")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,159}", dataset_version):
            raise ValueError("invalid dataset version for distributed dedup index")
        if not 1 <= pool_size <= 100:
            raise ValueError("pool_size must be between 1 and 100")
        self.dsn = dsn
        self.dataset_version = dataset_version
        self.pool_size = pool_size
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def check_and_add(self, row: dict[str, Any], *, hamming_threshold: int = 3) -> DistributedDedupDecision:
        if not 0 <= hamming_threshold <= 64:
            raise ValueError("hamming_threshold must be between 0 and 64")
        text = _row_text(row)
        exact = str(row.get("content_hash") or stable_hash(text))
        structure = normalized_structure_hash(text, _language_hint(row))
        lineage = _lineage_hash(row)
        fingerprint = simhash64(text)
        keys = _band_keys(fingerprint)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            transaction = connection.transaction(isolation="read_committed")
            await transaction.start()
            try:
                for key in sorted(keys):
                    await connection.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        f"{self.dataset_version}:{key}",
                    )
                inserted = await connection.fetchval(
                    """
                    INSERT INTO dataset_dedup_exact(dataset_version,content_hash)
                    VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING true
                    """,
                    self.dataset_version,
                    exact,
                )
                if not inserted:
                    await transaction.rollback()
                    return DistributedDedupDecision(
                        accepted=False,
                        reason="exact_duplicate",
                        content_hash=exact,
                        structure_hash=structure,
                        lineage_hash=lineage,
                        simhash64=f"{fingerprint:016x}",
                    )
                if structure:
                    inserted = await connection.fetchval(
                        """
                        INSERT INTO dataset_dedup_structure(dataset_version,structure_hash)
                        VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING true
                        """,
                        self.dataset_version,
                        structure,
                    )
                    if not inserted:
                        await transaction.rollback()
                        return DistributedDedupDecision(
                            accepted=False,
                            reason="structural_duplicate",
                            content_hash=exact,
                            structure_hash=structure,
                            lineage_hash=lineage,
                            simhash64=f"{fingerprint:016x}",
                        )
                if lineage:
                    inserted = await connection.fetchval(
                        """
                        INSERT INTO dataset_dedup_lineage(dataset_version,lineage_hash)
                        VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING true
                        """,
                        self.dataset_version,
                        lineage,
                    )
                    if not inserted:
                        await transaction.rollback()
                        return DistributedDedupDecision(
                            accepted=False,
                            reason="lineage_duplicate",
                            content_hash=exact,
                            structure_hash=structure,
                            lineage_hash=lineage,
                            simhash64=f"{fingerprint:016x}",
                        )
                if hamming_threshold >= 64:
                    candidates = await connection.fetch(
                        "SELECT simhash_hex FROM dataset_dedup_fingerprints WHERE dataset_version=$1",
                        self.dataset_version,
                    )
                else:
                    candidates = await connection.fetch(
                        """
                        SELECT DISTINCT f.simhash_hex
                        FROM dataset_dedup_bands b
                        JOIN dataset_dedup_fingerprints f ON f.id=b.fingerprint_id
                        WHERE b.dataset_version=$1 AND b.band_key=ANY($2::text[])
                        """,
                        self.dataset_version,
                        keys,
                    )
                comparisons = 0
                for candidate in candidates:
                    comparisons += 1
                    if hamming_distance(fingerprint, int(candidate["simhash_hex"], 16)) <= hamming_threshold:
                        await transaction.rollback()
                        return DistributedDedupDecision(
                            accepted=False,
                            reason="near_duplicate",
                            candidate_comparisons=comparisons,
                            content_hash=exact,
                            structure_hash=structure,
                            lineage_hash=lineage,
                            simhash64=f"{fingerprint:016x}",
                        )
                fingerprint_id = await connection.fetchval(
                    """
                    INSERT INTO dataset_dedup_fingerprints(dataset_version,simhash_hex)
                    VALUES ($1,$2) RETURNING id
                    """,
                    self.dataset_version,
                    f"{fingerprint:016x}",
                )
                await connection.executemany(
                    """
                    INSERT INTO dataset_dedup_bands(dataset_version,band_key,fingerprint_id)
                    VALUES ($1,$2,$3)
                    """,
                    [(self.dataset_version, key, fingerprint_id) for key in keys],
                )
                await transaction.commit()
                return DistributedDedupDecision(
                    accepted=True,
                    reason="unique",
                    candidate_comparisons=comparisons,
                    content_hash=exact,
                    structure_hash=structure,
                    lineage_hash=lineage,
                    simhash64=f"{fingerprint:016x}",
                )
            except BaseException:
                await transaction.rollback()
                raise


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or row.get("prompt") or "")


def _language_hint(row: dict[str, Any]) -> str | None:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    return row.get("language") or quality.get("language_hint")


def deduplicate_jsonl(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    hamming_threshold: int = 3,
    index_path: str | Path | None = None,
) -> NearDedupReport:
    if not 0 <= hamming_threshold <= 64:
        raise ValueError("hamming_threshold must be between 0 and 64")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_index = Path(index_path) if index_path is not None else target.with_suffix(target.suffix + ".dedup.sqlite3")
    temporary_output = target.with_suffix(target.suffix + ".partial")
    accepted = exact_duplicates = structural_duplicates = lineage_duplicates = near_duplicates = comparisons = 0
    by_source: dict[str, dict[str, int]] = {}
    index = SQLiteLSHDedupIndex(resolved_index, reset=True)
    try:
        with temporary_output.open("w", encoding="utf-8") as handle:
            for path in input_paths:
                for row in iter_jsonl(path):
                    text = _row_text(row)
                    source = str(row.get("source_id") or row.get("source") or "unknown")
                    stats = by_source.setdefault(
                        source,
                        {
                            "accepted": 0,
                            "exact_duplicates": 0,
                            "structural_duplicates": 0,
                            "lineage_duplicates": 0,
                            "near_duplicates": 0,
                        },
                    )
                    exact = str(row.get("content_hash") or stable_hash(text))
                    index.begin_candidate()
                    if not index.add_exact(exact):
                        index.reject_candidate()
                        exact_duplicates += 1
                        stats["exact_duplicates"] += 1
                        continue
                    structure = normalized_structure_hash(text, _language_hint(row))
                    if not index.add_structure(structure):
                        index.reject_candidate()
                        structural_duplicates += 1
                        stats["structural_duplicates"] += 1
                        continue
                    lineage = _lineage_hash(row)
                    if not index.add_lineage(lineage):
                        index.reject_candidate()
                        lineage_duplicates += 1
                        stats["lineage_duplicates"] += 1
                        continue
                    value = simhash64(text)
                    duplicate, compared = index.near_duplicate(value, hamming_threshold)
                    comparisons += compared
                    if duplicate:
                        index.reject_candidate()
                        near_duplicates += 1
                        stats["near_duplicates"] += 1
                        continue
                    index.add_fingerprint(value)
                    index.accept_candidate()
                    row["content_hash"] = exact
                    row["dedup"] = {
                        "simhash64": f"{value:016x}",
                        "structure_hash": structure,
                        "lineage_hash": lineage,
                        "policy": "sqlite_lsh_v1",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    accepted += 1
                    stats["accepted"] += 1
                    if accepted % 5_000 == 0:
                        index.connection.commit()
        temporary_output.replace(target)
    finally:
        index.close()
        if temporary_output.exists():
            temporary_output.unlink()
    return NearDedupReport(
        input_paths=[str(path) for path in input_paths],
        output_path=str(target),
        index_path=str(resolved_index),
        accepted=accepted,
        exact_duplicates=exact_duplicates,
        structural_duplicates=structural_duplicates,
        lineage_duplicates=lineage_duplicates,
        near_duplicates=near_duplicates,
        candidate_comparisons=comparisons,
        by_source=dict(sorted(by_source.items())),
        hamming_threshold=hamming_threshold,
    )


def iter_index_fingerprints(path: str | Path) -> Iterable[int]:
    with closing(sqlite3.connect(path)) as connection:
        for (value,) in connection.execute("SELECT simhash_hex FROM fingerprints ORDER BY id"):
            yield int(value, 16)


def run_dedup_scale_validation(
    *,
    record_count: int,
    output_dir: str | Path,
    hamming_threshold: int = 3,
    maximum_comparisons_per_record: float = 128.0,
) -> DedupScaleValidationReport:
    if record_count < 1:
        raise ValueError("record_count must be positive")
    if record_count > 10_000_000:
        raise ValueError("record_count exceeds the bounded validation limit")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / f"dedup-scale-{record_count}.sqlite3"
    started = time.perf_counter()
    tracemalloc.start()
    comparisons = 0
    indexed = 0
    index = SQLiteLSHDedupIndex(index_path, reset=True)
    try:
        for record_id in range(record_count):
            text = (
                f"repository_{record_id // 1000} function_{record_id} validates input "
                f"and returns deterministic result_{record_id:016x}"
            )
            exact = stable_hash(text)
            value = simhash64(text)
            index.begin_candidate()
            if not index.add_exact(exact):
                index.reject_candidate()
                continue
            duplicate, compared = index.near_duplicate(value, hamming_threshold)
            comparisons += compared
            if duplicate:
                index.reject_candidate()
                continue
            index.add_fingerprint(value)
            index.accept_candidate()
            indexed += 1
            if indexed % 10_000 == 0:
                index.connection.commit()
    finally:
        index.close()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    duration = time.perf_counter() - started
    comparisons_per_record = comparisons / max(1, record_count)
    passed = indexed > 0 and comparisons_per_record <= maximum_comparisons_per_record
    return DedupScaleValidationReport(
        requested_records=record_count,
        indexed_records=indexed,
        candidate_comparisons=comparisons,
        index_path=str(index_path),
        peak_python_memory_bytes=peak_memory,
        duration_seconds=round(duration, 6),
        comparisons_per_record=round(comparisons_per_record, 6),
        memory_bounded=True,
        status="passed" if passed else "failed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream exact/structural/lineage/near deduplication into a SQLite LSH index.")
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--index")
    parser.add_argument("--hamming-threshold", type=int, default=3)
    parser.add_argument("--scale-dry-records", type=int)
    parser.add_argument("--scale-output-dir", default="artifacts/craftly/dedup-scale")
    args = parser.parse_args()
    if args.scale_dry_records is not None:
        report = run_dedup_scale_validation(
            record_count=args.scale_dry_records,
            output_dir=args.scale_output_dir,
            hamming_threshold=args.hamming_threshold,
        )
        print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
        raise SystemExit(0 if report.status == "passed" else 2)
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --scale-dry-records is used")
    report = deduplicate_jsonl(
        args.input,
        args.output,
        hamming_threshold=args.hamming_threshold,
        index_path=args.index,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
