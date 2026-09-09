"""Benchmark contamination filtering for dataset construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.near_dedup import normalized_structure_hash
from src.craftly.learning.quality import iter_jsonl, stable_hash
from src.craftly.shared.schemas import print_report
from src.craftly.shared.schemas import StrictModel, print_report

DEFAULT_BENCHMARK_PATTERNS = [
    r"\bhumaneval\b",
    r"\bmbpp\b",
    r"\bswe-bench\b",
    r"\bcyberseceval\b",
    r"\bapps benchmark\b",
    r"\bcodecontests\b",
    r"\bpass@(?:1|5|10)\b",
    r"\bcanonical_solution\b",
    r"\btest_list\b",
    r"\bentry_point\b",
]
MINHASH_SIZE = 64
MINHASH_BANDS = 8
MINHASH_ROWS_PER_BAND = MINHASH_SIZE // MINHASH_BANDS
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+")
FINGERPRINT_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS protected_exact(hash TEXT PRIMARY KEY, benchmark TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS protected_task_id(task_hash TEXT PRIMARY KEY, benchmark TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS protected_structure(hash TEXT PRIMARY KEY, benchmark TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS protected_signatures(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  benchmark TEXT NOT NULL,
  signature_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS protected_bands(
  band_key TEXT NOT NULL,
  signature_id INTEGER NOT NULL REFERENCES protected_signatures(id) ON DELETE CASCADE,
  PRIMARY KEY(band_key, signature_id)
);
CREATE INDEX IF NOT EXISTS idx_protected_bands_key ON protected_bands(band_key);
"""


def load_patterns(path: str | Path | None) -> list[str]:
    if path is None:
        return DEFAULT_BENCHMARK_PATTERNS.copy()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return [str(item) for item in payload.get("patterns", [])]


class BenchmarkContaminationHit(StrictModel):
    input_path: str
    line_number: int
    source: str
    pattern: str
    content_hash: str | None = None


class ContaminationHit(StrictModel):
    source_path: str
    line_number: int
    url: str | None = None
    reason: str
    content_hash: str


class BenchmarkContaminationFilterReport(StrictModel):
    input_paths: list[str]
    output_path: str
    accepted: int
    rejected: int
    hits: list[BenchmarkContaminationHit] = Field(default_factory=list)
    exact_hits: int = 0
    task_id_hits: int = 0
    structural_hits: int = 0
    near_hits: int = 0
    protected_index_path: str | None = None
    created_at_unix: float = Field(default_factory=time.time)


class ContaminationReport(StrictModel):
    scanned_rows: int
    hits: list[ContaminationHit] = Field(default_factory=list)
    blocked: bool
    created_at_unix: float = Field(default_factory=time.time)


class BenchmarkContaminationFilter:
    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = patterns or DEFAULT_BENCHMARK_PATTERNS
        self.compiled = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]

    def find_pattern(self, text: str) -> str | None:
        for raw, compiled in zip(self.patterns, self.compiled):
            if compiled.search(text):
                return raw
        return None


def _text(row: dict[str, Any]) -> str:
    return str(
        row.get("text")
        or row.get("content")
        or row.get("prompt")
        or row.get("canonical_solution")
        or row.get("code")
        or ""
    )


def _protected_texts(row: dict[str, Any]) -> list[str]:
    """Collect every meaningful protected payload, not only the first field.

    Benchmark records commonly store the prompt, reference patch, tests and
    canonical solution in separate fields. Indexing only ``prompt`` would
    permit reference solutions to leak into training undetected.
    """

    texts: list[str] = []
    ignored_keys = {
        "content_hash",
        "created_at",
        "created_at_unix",
        "license",
        "metadata",
        "revision",
        "sha256",
        "source",
        "url",
    }

    def visit(value: Any, key: str = "") -> None:
        if key.lower() in ignored_keys:
            return
        if isinstance(value, str):
            normalized = " ".join(value.split())
            if len(normalized) >= 16:
                texts.append(value)
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)

    visit(row)
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        digest = stable_hash(" ".join(text.split()))
        if digest not in seen:
            seen.add(digest)
            unique.append(text)
    return unique


def _task_id(row: dict[str, Any]) -> str | None:
    value = row.get("task_id") or row.get("id") or row.get("name")
    return str(value).strip() if value not in {None, ""} else None


def _language(row: dict[str, Any]) -> str | None:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    return row.get("language") or quality.get("language_hint")


def _shingles(text: str, width: int = 5) -> set[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def minhash_signature(text: str) -> tuple[int, ...]:
    """Return a linear-time one-permutation MinHash signature.

    The previous implementation re-hashed every shingle 64 times. That was
    correct but too expensive for million-row contamination scans. Here one
    64-bit digest selects a bucket and value, followed by deterministic
    densification for empty buckets.
    """

    shingles = _shingles(text)
    if not shingles:
        return tuple([0] * MINHASH_SIZE)
    maximum = (1 << 64) - 1
    values = [maximum] * MINHASH_SIZE
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8", "replace"), digest_size=8).digest(),
            "big",
        )
        bucket = digest & (MINHASH_SIZE - 1)
        candidate = digest >> 6
        if candidate < values[bucket]:
            values[bucket] = candidate
    nonempty = [index for index, value in enumerate(values) if value != maximum]
    if not nonempty:
        return tuple([0] * MINHASH_SIZE)
    for bucket, value in enumerate(values):
        if value != maximum:
            continue
        source = min(nonempty, key=lambda index: (index - bucket) % MINHASH_SIZE)
        distance = (source - bucket) % MINHASH_SIZE
        values[bucket] = int.from_bytes(
            hashlib.blake2b(
                f"{distance}:{source}:{values[source]}".encode("ascii"),
                digest_size=8,
            ).digest(),
            "big",
        )
    return tuple(values)


def _signature_band_keys(signature: tuple[int, ...]) -> list[str]:
    keys: list[str] = []
    for band in range(MINHASH_BANDS):
        start = band * MINHASH_ROWS_PER_BAND
        values = signature[start : start + MINHASH_ROWS_PER_BAND]
        payload = ",".join(f"{value:016x}" for value in values)
        keys.append(f"{band}:{stable_hash(payload)[:20]}")
    return keys


class ProtectedBenchmarkFingerprintIndex:
    """Disk-backed benchmark fingerprint registry used only as a holdout guard."""

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(FINGERPRINT_SCHEMA)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def add(self, row: dict[str, Any], benchmark: str) -> None:
        texts = _protected_texts(row)
        if not texts:
            return
        task_id = _task_id(row)
        if task_id:
            self.connection.execute(
                "INSERT OR IGNORE INTO protected_task_id(task_hash,benchmark) VALUES (?,?)",
                (stable_hash(task_id.lower()), benchmark),
            )
        for text in texts:
            digest = stable_hash(" ".join(text.split()))
            self.connection.execute(
                "INSERT OR IGNORE INTO protected_exact(hash,benchmark) VALUES (?,?)",
                (digest, benchmark),
            )
            structure = normalized_structure_hash(text, _language(row))
            if structure:
                self.connection.execute(
                    "INSERT OR IGNORE INTO protected_structure(hash,benchmark) VALUES (?,?)",
                    (structure, benchmark),
                )
            signature = minhash_signature(text)
            cursor = self.connection.execute(
                "INSERT INTO protected_signatures(benchmark,signature_json) VALUES (?,?)",
                (benchmark, json.dumps(signature, separators=(",", ":"))),
            )
            signature_id = int(cursor.lastrowid)
            self.connection.executemany(
                "INSERT INTO protected_bands(band_key,signature_id) VALUES (?,?)",
                [(key, signature_id) for key in _signature_band_keys(signature)],
            )

    def match(
        self, row: dict[str, Any], *, minimum_similarity: float = 0.80
    ) -> tuple[str | None, str | None]:
        text = _text(row)
        if not text:
            return None, None
        exact = stable_hash(" ".join(text.split()))
        hit = self.connection.execute(
            "SELECT benchmark FROM protected_exact WHERE hash=?", (exact,)
        ).fetchone()
        if hit:
            return "exact_fingerprint", str(hit[0])
        task_id = _task_id(row)
        if task_id:
            hit = self.connection.execute(
                "SELECT benchmark FROM protected_task_id WHERE task_hash=?",
                (stable_hash(task_id.lower()),),
            ).fetchone()
            if hit:
                return "task_id_fingerprint", str(hit[0])
        structure = normalized_structure_hash(text, _language(row))
        if structure:
            hit = self.connection.execute(
                "SELECT benchmark FROM protected_structure WHERE hash=?", (structure,)
            ).fetchone()
            if hit:
                return "structural_fingerprint", str(hit[0])
        signature = minhash_signature(text)
        keys = _signature_band_keys(signature)
        candidates: set[tuple[str, str]] = set()
        for key in keys:
            matches = self.connection.execute(
                """
                SELECT DISTINCT s.benchmark,s.signature_json
                FROM protected_bands b
                JOIN protected_signatures s ON s.id=b.signature_id
                WHERE b.band_key=?
                """,
                (key,),
            )
            candidates.update(
                (str(benchmark), str(serialized)) for benchmark, serialized in matches
            )
        for benchmark, serialized in candidates:
            protected = tuple(int(value) for value in json.loads(serialized))
            similarity = (
                sum(left == right for left, right in zip(signature, protected)) / MINHASH_SIZE
            )
            if similarity >= minimum_similarity:
                return f"near_fingerprint:{similarity:.6f}", str(benchmark)
        return None, None


def build_protected_fingerprint_index(
    holdout_paths: list[str | Path],
    output_path: str | Path,
    *,
    require_all: bool = True,
) -> Path:
    if not holdout_paths:
        raise ValueError("at least one protected holdout is required")
    index = ProtectedBenchmarkFingerprintIndex(output_path, reset=True)
    loaded = 0
    try:
        for path in holdout_paths:
            source = Path(path)
            if not source.is_file():
                if require_all:
                    raise FileNotFoundError(f"protected benchmark is missing: {source}")
                continue
            benchmark = source.stem
            for row in iter_jsonl(source):
                index.add(row, benchmark)
                loaded += 1
        if loaded == 0:
            raise ValueError("protected benchmark registry contains no records")
    finally:
        index.close()
    return Path(output_path)


class ContaminationDetector:
    """Read-only benchmark/holdout leakage scanner.

    Filtering and scanning live in this module so Craftly has one authoritative
    benchmark-contamination policy instead of parallel basic/advanced versions.
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.filter = BenchmarkContaminationFilter(patterns)

    def scan_jsonl(
        self, paths: list[str | Path], *, block_on_hit: bool = True
    ) -> ContaminationReport:
        hits: list[ContaminationHit] = []
        scanned = 0
        for path in paths:
            source = Path(path)
            for line_number, row in enumerate(iter_jsonl(source), start=1):
                scanned += 1
                text = str(row.get("text") or row.get("content") or "")
                pattern = self.filter.find_pattern(text)
                if pattern is not None:
                    hits.append(
                        ContaminationHit(
                            source_path=str(source),
                            line_number=line_number,
                            url=row.get("url"),
                            reason=f"benchmark_pattern:{pattern}",
                            content_hash=str(row.get("content_hash") or stable_hash(text)),
                        )
                    )
        return ContaminationReport(
            scanned_rows=scanned, hits=hits, blocked=bool(hits and block_on_hit)
        )


def filter_benchmark_contamination_jsonl(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    patterns: list[str] | None = None,
    protected_index_path: str | Path | None = None,
    minimum_similarity: float = 0.80,
) -> BenchmarkContaminationFilterReport:
    detector = BenchmarkContaminationFilter(patterns)
    protected = (
        ProtectedBenchmarkFingerprintIndex(protected_index_path) if protected_index_path else None
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    exact_hits = task_id_hits = structural_hits = near_hits = 0
    hits: list[BenchmarkContaminationHit] = []
    try:
        with target.open("w", encoding="utf-8") as handle:
            for path in input_paths:
                for line_number, row in enumerate(iter_jsonl(path), start=1):
                    text = _text(row)
                    pattern = detector.find_pattern(text)
                    benchmark: str | None = None
                    if pattern is None and protected is not None:
                        match, benchmark = protected.match(
                            row, minimum_similarity=minimum_similarity
                        )
                        pattern = match
                    if pattern is not None:
                        rejected += 1
                        if pattern == "exact_fingerprint":
                            exact_hits += 1
                        elif pattern == "task_id_fingerprint":
                            task_id_hits += 1
                        elif pattern == "structural_fingerprint":
                            structural_hits += 1
                        elif pattern.startswith("near_fingerprint:"):
                            near_hits += 1
                        hits.append(
                            BenchmarkContaminationHit(
                                input_path=str(path),
                                line_number=line_number,
                                source=str(row.get("source") or "unknown"),
                                pattern=f"{pattern}:{benchmark}" if benchmark else pattern,
                                content_hash=row.get("content_hash"),
                            )
                        )
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    accepted += 1
    finally:
        if protected is not None:
            protected.close()
    return BenchmarkContaminationFilterReport(
        input_paths=[str(path) for path in input_paths],
        output_path=str(target),
        accepted=accepted,
        rejected=rejected,
        hits=hits,
        exact_hits=exact_hits,
        task_id_hits=task_id_hits,
        structural_hits=structural_hits,
        near_hits=near_hits,
        protected_index_path=str(protected_index_path) if protected_index_path else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove benchmark-contaminated rows from Craftly JSONL shards."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patterns")
    parser.add_argument("--protected-index")
    parser.add_argument("--build-protected-index", nargs="*")
    parser.add_argument("--minimum-similarity", type=float, default=0.80)
    args = parser.parse_args()
    if args.build_protected_index:
        if not args.protected_index:
            raise SystemExit("--protected-index is required with --build-protected-index")
        build_protected_fingerprint_index(args.build_protected_index, args.protected_index)
    report = filter_benchmark_contamination_jsonl(
        args.input,
        args.output,
        patterns=load_patterns(args.patterns),
        protected_index_path=args.protected_index,
        minimum_similarity=args.minimum_similarity,
    )
    print_report(report)


if __name__ == "__main__":
    main()
