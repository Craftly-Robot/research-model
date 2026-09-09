"""Structured progress reporting for long Craftly jobs."""

from __future__ import annotations

import atexit
import json
import os
import queue
import random
import socket
import sys
import time
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, TextIO
from urllib.parse import urlparse


class ProgressReporter:
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        to_stdout: bool = False,
        prefix: str = "craftly-progress",
        stream: TextIO | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.to_stdout = to_stdout
        self.prefix = prefix
        self.stream = stream or sys.stdout
        self.lock = RLock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, status: str = "running", **metrics: Any) -> dict[str, Any]:
        payload = {
            "ts_unix": time.time(),
            "stage": stage,
            "status": status,
            **{key: value for key, value in metrics.items() if value is not None},
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.lock:
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if self.to_stdout:
                self.stream.write(f"[{self.prefix}] {line}\n")
                self.stream.flush()
        return payload

    def close(self) -> None:
        return None


class NullProgressReporter(ProgressReporter):
    def __init__(self) -> None:
        super().__init__(path=None, to_stdout=False)

    def emit(self, stage: str, status: str = "running", **metrics: Any) -> dict[str, Any]:
        return {
            "ts_unix": time.time(),
            "stage": stage,
            "status": status,
            **{key: value for key, value in metrics.items() if value is not None},
        }


class WorkspaceProgressReporter(ProgressReporter):
    """Non-blocking workspace event delivery with heartbeat and local WAL."""

    def __init__(
        self,
        *,
        workspace_url: str,
        job_id: str,
        attempt_id: str,
        access_token: str,
        path: str | Path | None,
        to_stdout: bool,
        wal_path: str | Path | None = None,
        batch_size: int = 25,
        flush_interval_seconds: float = 1.0,
        heartbeat_seconds: float = 5.0,
        queue_capacity: int = 10_000,
    ) -> None:
        parsed = urlparse(workspace_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("CRAFTLY_WORKSPACE_URL must be an absolute HTTP(S) URL")
        if (
            parsed.scheme != "https"
            and not local
            and os.environ.get("CRAFTLY_ALLOW_INSECURE_WORKSPACE") != "1"
        ):
            raise ValueError("remote workspace progress endpoint must use HTTPS")
        super().__init__(path=path, to_stdout=to_stdout)
        self.workspace_url = workspace_url.rstrip("/")
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.access_token = access_token
        self.wal_path = (
            Path(wal_path)
            if wal_path
            else Path(path or "progress.jsonl").with_name("progress-wal.jsonl")
        )
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = min(max(batch_size, 1), 100)
        self.flush_interval_seconds = max(flush_interval_seconds, 0.1)
        self.heartbeat_seconds = max(heartbeat_seconds, 1.0)
        self.pending: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_capacity)
        self.stop_event = Event()
        self.sequence_lock = RLock()
        self.wal_lock = RLock()
        self.coalesced_lock = RLock()
        # A process-unique monotonic base prevents rank-local sequence reuse
        # after a worker restart while preserving exact WAL replay identity.
        self.source_sequence = time.time_ns() // 1_000
        self.dropped_metrics = 0
        self.coalesced_metrics: dict[tuple[str, int, str], dict[str, Any]] = {}
        self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        self.world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        self.node = socket.gethostname()
        self.worker = Thread(
            target=self._delivery_loop, name="craftly-progress-delivery", daemon=True
        )
        self.worker.start()
        atexit.register(self.close)

    def _next_source_sequence(self) -> int:
        with self.sequence_lock:
            self.source_sequence += 1
            return self.source_sequence

    @staticmethod
    def _kind(stage: str, status: str, metrics: dict[str, Any]) -> str:
        lowered = stage.lower()
        if status in {"failed", "blocked", "error"} or metrics.get("error"):
            return "error"
        if "checkpoint" in lowered:
            return "checkpoint"
        if "eval" in lowered or "validation" in lowered:
            return "evaluation"
        if lowered == "heartbeat":
            return "heartbeat"
        if any(
            key in metrics
            for key in ["step", "loss", "validation_loss", "tokens_per_second", "gpu_memory_bytes"]
        ):
            return "metric"
        return "status"

    def emit(self, stage: str, status: str = "running", **metrics: Any) -> dict[str, Any]:
        payload = super().emit(stage, status, **metrics)
        kind = self._kind(stage, status, metrics)
        if self.rank != 0 and kind not in {"heartbeat", "error", "checkpoint"}:
            return payload
        known = {
            "step",
            "max_steps",
            "steps",
            "loss",
            "validation_loss",
            "tokens_per_second",
            "gpu_memory_bytes",
            "message",
        }
        event = {
            "source_sequence": self._next_source_sequence(),
            "kind": kind,
            "stage": stage,
            "status": status,
            "rank": self.rank,
            "world_size": self.world_size,
            "node": self.node,
            "step": metrics.get("step"),
            "max_steps": metrics.get("max_steps", metrics.get("steps")),
            "loss": metrics.get("loss"),
            "validation_loss": metrics.get("validation_loss"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "gpu_memory_bytes": metrics.get("gpu_memory_bytes"),
            "message": metrics.get("message") or metrics.get("error"),
            "payload": {
                key: value
                for key, value in metrics.items()
                if key not in known and value is not None
            },
        }
        self._enqueue(event, durable=kind in {"error", "checkpoint", "evaluation"})
        return payload

    def _enqueue(self, event: dict[str, Any], *, durable: bool) -> None:
        try:
            self.pending.put_nowait(event)
        except queue.Full:
            if durable:
                self._write_wal([event])
            else:
                with self.coalesced_lock:
                    self.coalesced_metrics[
                        (str(event.get("stage")), int(event.get("rank", 0)), str(event.get("kind")))
                    ] = event
                self.dropped_metrics += 1

    def _write_wal(self, events: list[dict[str, Any]]) -> None:
        row = {"attempt_id": self.attempt_id, "events": events}
        with self.wal_lock, self.wal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _token(self) -> str:
        token_file = os.environ.get("CRAFTLY_WORKSPACE_TOKEN_FILE")
        if token_file:
            value = Path(token_file).read_text(encoding="utf-8").strip()
            if value:
                return value
        return self.access_token

    def _send(self, events: list[dict[str, Any]]) -> bool:
        try:
            import httpx

            response = httpx.post(
                f"{self.workspace_url}/v1/training/jobs/{self.job_id}/events:batch",
                headers={"Authorization": f"Bearer {self._token()}"},
                json={"attempt_id": self.attempt_id, "events": events},
                timeout=10.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _replay_wal(self) -> None:
        if not self.wal_path.exists() or self.wal_path.stat().st_size == 0:
            return
        with self.wal_lock:
            lines = self.wal_path.read_text(encoding="utf-8", errors="replace").splitlines()
            retained = []
            for line in lines:
                try:
                    row = json.loads(line)
                    delivered = row.get("attempt_id") == self.attempt_id and self._send(
                        list(row.get("events") or [])
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    delivered = False
                if not delivered:
                    retained.append(line)
            temporary = self.wal_path.with_suffix(".tmp")
            temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
            os.replace(temporary, self.wal_path)

    def _heartbeat(self) -> dict[str, Any]:
        return {
            "source_sequence": self._next_source_sequence(),
            "kind": "heartbeat",
            "stage": "heartbeat",
            "status": "running",
            "rank": self.rank,
            "world_size": self.world_size,
            "node": self.node,
            "message": None,
            "payload": {"coalesced_metric_updates": self.dropped_metrics},
        }

    def _delivery_loop(self) -> None:
        self._replay_wal()
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        last_heartbeat = time.monotonic()
        failure_streak = 0
        while (
            not self.stop_event.is_set()
            or not self.pending.empty()
            or batch
            or self.coalesced_metrics
        ):
            timeout = max(0.05, min(self.flush_interval_seconds, self.heartbeat_seconds) / 2)
            try:
                batch.append(self.pending.get(timeout=timeout))
            except queue.Empty:
                pass
            with self.coalesced_lock:
                while self.coalesced_metrics and len(batch) < self.batch_size:
                    _, latest = self.coalesced_metrics.popitem()
                    batch.append(latest)
            now = time.monotonic()
            if now - last_heartbeat >= self.heartbeat_seconds:
                batch.append(self._heartbeat())
                last_heartbeat = now
            if batch and (
                len(batch) >= self.batch_size
                or now - last_flush >= self.flush_interval_seconds
                or self.stop_event.is_set()
            ):
                sending = batch[:100]
                del batch[: len(sending)]
                if not self._send(sending):
                    self._write_wal(sending)
                    failure_streak += 1
                    backoff = min(30.0, self.flush_interval_seconds * (2 ** min(failure_streak, 6)))
                    time.sleep(backoff * random.uniform(0.75, 1.25))  # nosec B311 - retry jitter is non-cryptographic
                else:
                    failure_streak = 0
                last_flush = time.monotonic()
            if now - last_flush >= 30.0:
                self._replay_wal()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.worker.join(timeout=10)


def progress_from_options(*, path: str | Path | None, to_stdout: bool) -> ProgressReporter:
    workspace_url = os.environ.get("CRAFTLY_WORKSPACE_URL")
    job_id = os.environ.get("CRAFTLY_TRAINING_JOB_ID")
    attempt_id = os.environ.get("CRAFTLY_TRAINING_ATTEMPT_ID")
    access_token = os.environ.get("CRAFTLY_WORKSPACE_ACCESS_TOKEN")
    token_file = os.environ.get("CRAFTLY_WORKSPACE_TOKEN_FILE")
    if workspace_url and job_id and attempt_id and (access_token or token_file):
        return WorkspaceProgressReporter(
            workspace_url=workspace_url,
            job_id=job_id,
            attempt_id=attempt_id,
            access_token=access_token or "token-from-file",
            path=path,
            to_stdout=to_stdout,
            wal_path=os.environ.get("CRAFTLY_PROGRESS_WAL_PATH"),
        )
    if path or to_stdout:
        return ProgressReporter(path=path, to_stdout=to_stdout)
    return NullProgressReporter()
