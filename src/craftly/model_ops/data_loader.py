"""Streaming token-shard dataloader for scratch pretraining."""

from __future__ import annotations

import json
import hashlib
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from src.craftly.model_ops.tokenizer_pipeline import ShardManifest, read_uint32_tokens
from src.craftly.shared.integrity import sha256_file


class ArtifactCache:
    """Checksum-verifying, process-safe materializer for immutable training assets."""

    def __init__(
        self,
        root: str | Path,
        *,
        s3_endpoint_url: str | None = None,
        lock_timeout_seconds: float = 600.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.s3_endpoint_url = s3_endpoint_url
        self.lock_timeout_seconds = lock_timeout_seconds

    @staticmethod
    def _local_path(uri: str) -> Path | None:
        direct = Path(uri).expanduser()
        if direct.is_absolute():
            return direct.resolve()
        parsed = urlparse(uri)
        if parsed.scheme == "":
            return Path(uri).expanduser().resolve()
        if parsed.scheme != "file":
            return None
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URI must not reference a remote host")
        value = unquote(parsed.path)
        if os.name == "nt" and value.startswith("/") and len(value) > 3 and value[2] == ":":
            value = value[1:]
        return Path(value).expanduser().resolve()

    def _target(self, uri: str) -> Path:
        parsed = urlparse(uri)
        suffix = Path(parsed.path).suffix[:16]
        return self.root / f"{hashlib.sha256(uri.encode('utf-8')).hexdigest()}{suffix}"

    def materialize(self, uri: str | Path, *, expected_sha256: str | None = None) -> Path:
        value = str(uri)
        local = self._local_path(value)
        if local is not None:
            if not local.is_file():
                raise FileNotFoundError(f"training artifact does not exist: {local}")
            if expected_sha256 and sha256_file(local) != expected_sha256:
                raise ValueError(f"training artifact checksum mismatch: {local}")
            return local
        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"unsupported training artifact URI: {value!r}")
        target = self._target(value)
        if target.is_file() and (not expected_sha256 or sha256_file(target) == expected_sha256):
            return target
        lock = target.with_suffix(target.suffix + ".lock")
        deadline = time.monotonic() + self.lock_timeout_seconds
        acquired = False
        while not acquired:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
                acquired = True
            except FileExistsError:
                if target.is_file() and (not expected_sha256 or sha256_file(target) == expected_sha256):
                    return target
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for training artifact cache lock: {lock}")
                time.sleep(0.2)
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        try:
            import boto3

            client = boto3.client("s3", endpoint_url=self.s3_endpoint_url)
            client.download_file(parsed.netloc, parsed.path.strip("/"), str(temporary))
            digest = sha256_file(temporary)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError(f"downloaded training artifact checksum mismatch: {value}")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


def load_manifest(
    path: str | Path,
    *,
    cache: ArtifactCache | None = None,
    expected_sha256: str | None = None,
) -> ShardManifest:
    source = cache.materialize(path, expected_sha256=expected_sha256) if cache else Path(path)
    return ShardManifest.model_validate(json.loads(source.read_text(encoding="utf-8-sig")))


class TokenShardStream:
    def __init__(
        self,
        shard_paths: list[str],
        *,
        sequence_length: int,
        batch_size: int,
        seed: int = 1337,
        shuffle: bool = True,
        prefetch_batches: int = 0,
        artifact_cache: ArtifactCache | None = None,
        expected_sha256: dict[str, str] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not shard_paths:
            raise ValueError("at least one shard path is required")
        self.shard_paths = shard_paths
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        if prefetch_batches < 0 or prefetch_batches > 128:
            raise ValueError("prefetch_batches must be between 0 and 128")
        self.prefetch_batches = prefetch_batches
        self.artifact_cache = artifact_cache
        self.expected_sha256 = expected_sha256 or {}
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self.rank = rank
        self.world_size = world_size

    def _materialize_shard(self, path: str) -> Path:
        if self.artifact_cache:
            return self.artifact_cache.materialize(path, expected_sha256=self.expected_sha256.get(path))
        local = Path(path)
        expected = self.expected_sha256.get(path)
        if expected and sha256_file(local) != expected:
            raise ValueError(f"training shard checksum mismatch: {path}")
        return local

    def _global_batches(self, *, epoch: int = 0) -> Iterator[list[list[int]]]:
        rng = random.Random(self.seed + epoch)
        paths = self.shard_paths.copy()
        if self.shuffle:
            rng.shuffle(paths)
        buffer: list[int] = []
        needed = self.batch_size * self.sequence_length
        for path in paths:
            tokens = read_uint32_tokens(self._materialize_shard(path))
            if self.shuffle:
                offset = rng.randrange(max(1, min(len(tokens), self.sequence_length)))
                tokens = tokens[offset:] + tokens[:offset]
            buffer.extend(tokens)
            while len(buffer) >= needed:
                chunk = buffer[:needed]
                del buffer[:needed]
                yield [
                    chunk[index : index + self.sequence_length]
                    for index in range(0, needed, self.sequence_length)
                ]

    def _batches(self, *, epoch: int = 0, start_batch: int = 0) -> Iterator[list[list[int]]]:
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        group: list[list[list[int]]] = []
        local_index = 0
        for batch in self._global_batches(epoch=epoch):
            group.append(batch)
            if len(group) < self.world_size:
                continue
            selected = group[self.rank]
            group.clear()
            if local_index >= start_batch:
                yield selected
            local_index += 1

    def batches(self, *, epoch: int = 0, start_batch: int = 0) -> Iterator[list[list[int]]]:
        if self.prefetch_batches == 0:
            yield from self._batches(epoch=epoch, start_batch=start_batch)
            return

        sentinel = object()
        buffered: queue.Queue[object] = queue.Queue(maxsize=self.prefetch_batches)
        stop = threading.Event()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    buffered.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for batch in self._batches(epoch=epoch, start_batch=start_batch):
                    if not put(batch):
                        return
            except BaseException as exc:  # propagated to the training thread
                put(exc)
            finally:
                put(sentinel)

        worker = threading.Thread(
            target=produce,
            name=f"craftly-shard-prefetch-rank-{self.rank}-epoch-{epoch}",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                item = buffered.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise RuntimeError("token shard prefetch failed") from item
                yield item  # type: ignore[misc]
        finally:
            stop.set()
            worker.join(timeout=1.0)


def count_batches(
    shard_paths: list[str],
    *,
    sequence_length: int,
    batch_size: int,
    artifact_cache: ArtifactCache | None = None,
    expected_sha256: dict[str, str] | None = None,
) -> int:
    total_tokens = sum(
        len(
            read_uint32_tokens(
                artifact_cache.materialize(path, expected_sha256=(expected_sha256 or {}).get(path))
                if artifact_cache
                else path
            )
        )
        for path in shard_paths
    )
    return total_tokens // (sequence_length * batch_size)

