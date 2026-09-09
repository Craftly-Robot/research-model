"""Canonical integrity primitives shared across Craftly subsystems."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return the SHA-256 digest of a regular file using bounded memory."""
    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"integrity source is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for hashing and signatures."""
    return canonical_json_text(payload).encode("utf-8")


def canonical_json_text(payload: Any) -> str:
    """Serialize JSON deterministically as text for storage and transport."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
