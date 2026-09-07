"""Object storage adapters for Craftly dataset artifacts."""

from __future__ import annotations

import hashlib
import argparse
import json
import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import Field

from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file as file_sha256


class StoredObject(StrictModel):
    source_path: str
    uri: str
    size_bytes: int
    sha256: str


class ObjectStore(Protocol):
    def put_file(self, path: str | Path, *, key: str | None = None) -> StoredObject:
        ...

    def put_json(self, payload: dict[str, Any], *, key: str) -> StoredObject:
        ...

    def head(self, key: str) -> StoredObject:
        ...

    def get_file(self, key: str, target_path: str | Path) -> StoredObject:
        ...

    def delete(self, key: str) -> None:
        ...

    def list_objects(self, prefix: str = "") -> list[StoredObject]:
        ...


class LocalObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, path: str | Path, *, key: str | None = None) -> StoredObject:
        source = Path(path)
        target = self.root / (key or source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return StoredObject(
            source_path=str(source),
            uri=str(target),
            size_bytes=target.stat().st_size,
            sha256=file_sha256(target),
        )

    def put_json(self, payload: dict[str, Any], *, key: str) -> StoredObject:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return StoredObject(source_path=str(target), uri=str(target), size_bytes=target.stat().st_size, sha256=file_sha256(target))

    def _resolve_key(self, key: str) -> Path:
        target = (self.root / key).resolve()
        root = self.root.resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"object key escapes local object store root: {key}")
        return target

    def head(self, key: str) -> StoredObject:
        target = self._resolve_key(key)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"object not found: {key}")
        return StoredObject(source_path=str(target), uri=str(target), size_bytes=target.stat().st_size, sha256=file_sha256(target))

    def get_file(self, key: str, target_path: str | Path) -> StoredObject:
        source = self._resolve_key(key)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"object not found: {key}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return StoredObject(source_path=str(source), uri=str(target), size_bytes=target.stat().st_size, sha256=file_sha256(target))

    def delete(self, key: str) -> None:
        target = self._resolve_key(key)
        if target.exists():
            target.unlink()

    def list_objects(self, prefix: str = "") -> list[StoredObject]:
        base = self._resolve_key(prefix) if prefix else self.root.resolve()
        if base.is_file():
            paths = [base]
        elif base.exists():
            paths = sorted(item for item in base.rglob("*") if item.is_file())
        else:
            paths = []
        return [
            StoredObject(source_path=str(path), uri=str(path), size_bytes=path.stat().st_size, sha256=file_sha256(path))
            for path in paths
        ]


class S3ObjectStore:
    def __init__(self, uri: str, *, endpoint_url: str | None = None, max_retries: int = 3) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("boto3 is required for s3/minio object storage") from exc
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("S3 object store URI must look like s3://bucket/prefix")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint_url)
        self.max_retries = max_retries

    def _object_key(self, key: str) -> str:
        normalized = key.replace("\\", "/").strip("/")
        parts = PurePosixPath(normalized).parts if normalized else ()
        if any(part in {".", ".."} for part in parts):
            raise ValueError(f"unsafe S3 object key: {key}")
        return f"{self.prefix}/{normalized}".strip("/") if self.prefix else normalized

    def put_file(self, path: str | Path, *, key: str | None = None) -> StoredObject:
        source = Path(path)
        object_key = self._object_key(key or source.name)
        digest = file_sha256(source)
        retry_sync(
            lambda: self.client.upload_file(
                str(source),
                self.bucket,
                object_key,
                ExtraArgs={"Metadata": {"sha256": digest}},
            ),
            max_retries=self.max_retries,
        )
        return StoredObject(
            source_path=str(source),
            uri=f"s3://{self.bucket}/{object_key}",
            size_bytes=source.stat().st_size,
            sha256=digest,
        )

    def put_json(self, payload: dict[str, Any], *, key: str) -> StoredObject:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        object_key = self._object_key(key)
        retry_sync(
            lambda: self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=body,
                ContentType="application/json",
                Metadata={"sha256": hashlib.sha256(body).hexdigest()},
            ),
            max_retries=self.max_retries,
        )
        digest = hashlib.sha256(body).hexdigest()
        return StoredObject(source_path=f"memory:{key}", uri=f"s3://{self.bucket}/{object_key}", size_bytes=len(body), sha256=digest)

    def head(self, key: str) -> StoredObject:
        object_key = self._object_key(key)
        try:
            response = retry_sync(
                lambda: self.client.head_object(Bucket=self.bucket, Key=object_key),
                max_retries=self.max_retries,
            )
        except RuntimeError as exc:
            cause = exc.__cause__
            response_code = getattr(cause, "response", {}).get("Error", {}).get("Code", "")
            if str(response_code) in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"object not found: {key}") from exc
            raise
        return StoredObject(
            source_path=f"s3://{self.bucket}/{object_key}",
            uri=f"s3://{self.bucket}/{object_key}",
            size_bytes=int(response.get("ContentLength", 0)),
            sha256=str(response.get("Metadata", {}).get("sha256", "")),
        )

    def get_file(self, key: str, target_path: str | Path) -> StoredObject:
        object_key = self._object_key(key)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        retry_sync(
            lambda: self.client.download_file(self.bucket, object_key, str(target)),
            max_retries=self.max_retries,
        )
        return StoredObject(source_path=f"s3://{self.bucket}/{object_key}", uri=str(target), size_bytes=target.stat().st_size, sha256=file_sha256(target))

    def delete(self, key: str) -> None:
        object_key = self._object_key(key)
        retry_sync(lambda: self.client.delete_object(Bucket=self.bucket, Key=object_key), max_retries=self.max_retries)

    def list_objects(self, prefix: str = "") -> list[StoredObject]:
        object_prefix = self._object_key(prefix) if prefix else self.prefix
        objects = []
        continuation: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": self.bucket, "Prefix": object_prefix}
            if continuation:
                request["ContinuationToken"] = continuation
            response = retry_sync(
                lambda request=request: self.client.list_objects_v2(**request),
                max_retries=self.max_retries,
            )
            for item in response.get("Contents", []):
                key = item["Key"]
                objects.append(
                    StoredObject(
                        source_path=f"s3://{self.bucket}/{key}",
                        uri=f"s3://{self.bucket}/{key}",
                        size_bytes=int(item.get("Size", 0)),
                        sha256="",
                    )
                )
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                raise RuntimeError("S3 pagination response was truncated without a continuation token")
        return objects

    def presign_put(
        self,
        *,
        key: str,
        sha256: str,
        content_type: str = "application/octet-stream",
        expires_seconds: int = 900,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("presigned upload requires a SHA-256 digest")
        object_key = self._object_key(key)
        params = {
            "Bucket": self.bucket,
            "Key": object_key,
            "ContentType": content_type,
            "Metadata": {"sha256": sha256},
        }
        url = self.client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=min(max(expires_seconds, 60), 3600),
            HttpMethod="PUT",
        )
        return {
            "method": "PUT",
            "url": url,
            "uri": f"s3://{self.bucket}/{object_key}",
            "key": key,
            "expires_in": min(max(expires_seconds, 60), 3600),
            "headers": {"Content-Type": content_type, "x-amz-meta-sha256": sha256},
        }


class ObjectStoreConfig(StrictModel):
    uri: str = Field(default="local://artifacts/craftly/object-store", min_length=1)
    endpoint_url: str | None = None
    max_retries: int = Field(default=3, ge=1, le=10)


def retry_sync(operation: Any, *, max_retries: int, base_delay_seconds: float = 0.25) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - network retries are integration-tested in deployment
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(base_delay_seconds * (2**attempt))
    raise RuntimeError(f"object storage operation failed after {max_retries} attempts") from last_error


def create_object_store(config: ObjectStoreConfig) -> ObjectStore:
    parsed = urlparse(config.uri)
    if len(parsed.scheme) == 1 and config.uri[1:3] in {":\\", ":/"}:
        return LocalObjectStore(config.uri)
    if parsed.scheme in {"", "file"}:
        return LocalObjectStore(parsed.path or config.uri)
    if parsed.scheme == "local":
        root = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        return LocalObjectStore(root)
    if parsed.scheme == "s3":
        return S3ObjectStore(config.uri, endpoint_url=config.endpoint_url, max_retries=config.max_retries)
    raise ValueError(f"unsupported object store URI: {config.uri}")


def upload_paths(store: ObjectStore, paths: list[str | Path], *, prefix: str) -> list[StoredObject]:
    uploaded: list[StoredObject] = []
    used_keys: set[str] = set()
    for path in paths:
        source = Path(path)
        object_key = f"{prefix}/{source.name}"
        if object_key in used_keys:
            object_key = f"{prefix}/{source.parent.name}/{source.name}"
        counter = 2
        while object_key in used_keys:
            object_key = f"{prefix}/{source.parent.name}/{source.stem}-{counter}{source.suffix}"
            counter += 1
        used_keys.add(object_key)
        uploaded.append(store.put_file(source, key=object_key))
    return uploaded


class ObjectStoreLifecycleReport(StrictModel):
    status: str
    uri: str
    key: str
    uploaded: StoredObject
    downloaded: StoredObject
    listed_count: int
    checksum_match: bool
    deleted: bool


def verify_object_store_lifecycle(
    *,
    config: ObjectStoreConfig,
    work_dir: str | Path,
    key: str = "lifecycle/probe.json",
) -> ObjectStoreLifecycleReport:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "probe-source.json"
    downloaded = root / "probe-downloaded.json"
    payload = {"component": "craftly-object-store-lifecycle", "created_at_unix": time.time()}
    source.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    store = create_object_store(config)
    uploaded = store.put_file(source, key=key)
    store.head(key)
    downloaded_object = store.get_file(key, downloaded)
    listed = store.list_objects(str(PurePosixPath(key).parent))
    checksum_match = file_sha256(source) == file_sha256(downloaded)
    store.delete(key)
    deleted = True
    try:
        store.head(key)
        deleted = False
    except FileNotFoundError:
        deleted = True
    report = ObjectStoreLifecycleReport(
        status="passed" if checksum_match and deleted and len(listed) >= 1 else "failed",
        uri=config.uri,
        key=key,
        uploaded=uploaded,
        downloaded=downloaded_object,
        listed_count=len(listed),
        checksum_match=checksum_match,
        deleted=deleted,
    )
    (root / "object_store_lifecycle_report.json").write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Verify Craftly object storage lifecycle.")
    parser.add_argument("--uri", default="local://artifacts/craftly/object-store")
    parser.add_argument("--endpoint-url")
    parser.add_argument("--work-dir", default="artifacts/craftly/object-store-lifecycle")
    parser.add_argument("--key", default="lifecycle/probe.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = verify_object_store_lifecycle(
        config=ObjectStoreConfig(uri=args.uri, endpoint_url=args.endpoint_url),
        work_dir=args.work_dir,
        key=args.key,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

