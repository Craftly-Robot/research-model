"""Lightweight authenticated client for the Craftly Training Workspace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess  # nosec B404 - git metadata lookup uses fixed argv
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

TERMINAL_STATES = {"succeeded", "failed", "blocked", "cancelled"}


def _git_commit() -> str:
    try:
        completed = subprocess.run(  # nosec B603
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        value = completed.stdout.strip().lower()
        if (
            completed.returncode == 0
            and 7 <= len(value) <= 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            return value
    except (OSError, subprocess.SubprocessError):
        return "0000000"
    return "0000000"


def _container_digest() -> str:
    value = os.environ.get("CRAFTLY_TRAINING_IMAGE_DIGEST", "")
    return value or "sha256:" + ("0" * 64)


def _validate_workspace_url(url: str) -> str:
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("CRAFTLY_WORKSPACE_URL must be an absolute HTTP(S) URL")
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        parsed.scheme != "https"
        and not local
        and os.environ.get("CRAFTLY_ALLOW_INSECURE_WORKSPACE") != "1"
    ):
        raise ValueError("non-local Craftly workspace URLs must use HTTPS")
    return normalized


@dataclass
class TokenSession:
    access_token: str
    access_expires_at: float
    session_id: str
    refresh_token: str
    refresh_expires_at: float


class TrainingRun:
    def __init__(self, workspace: Workspace, payload: dict[str, Any]) -> None:
        self.workspace = workspace
        self.payload = payload
        self.job_id = str(payload["job_id"])

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "unknown"))

    async def refresh(self) -> dict[str, Any]:
        self.payload = await self.workspace.get_job(self.job_id)
        return self.payload

    async def follow(self, *, after_sequence: int = 0, print_events: bool = True) -> dict[str, Any]:
        async for event in self.workspace.events(self.job_id, after_sequence=after_sequence):
            after_sequence = max(after_sequence, int(event.get("sequence", 0)))
            if print_events:
                print(format_event(event), flush=True)
        return await self.refresh()

    async def cancel(self) -> dict[str, Any]:
        self.payload = await self.workspace.cancel(self.job_id)
        return self.payload

    async def resume(self) -> dict[str, Any]:
        self.payload = await self.workspace.resume(self.job_id)
        return self.payload


class Workspace:
    def __init__(
        self,
        *,
        url: str,
        bootstrap_token: str | None = None,
        timeout_seconds: float = 30.0,
        verify_tls: bool = True,
    ) -> None:
        self.url = _validate_workspace_url(url)
        self.bootstrap_token = bootstrap_token
        self.timeout_seconds = timeout_seconds
        self.verify_tls = verify_tls
        self.tokens: TokenSession | None = None
        self.client = httpx.AsyncClient(
            base_url=self.url,
            timeout=httpx.Timeout(timeout_seconds, read=None),
            verify=verify_tls,
            headers={"User-Agent": "craftly-client/1.0"},
        )

    @classmethod
    def from_environment(cls) -> Workspace:
        url = os.environ.get("CRAFTLY_WORKSPACE_URL", "")
        if not url:
            raise RuntimeError("CRAFTLY_WORKSPACE_URL is required")
        token = os.environ.get("CRAFTLY_BOOTSTRAP_TOKEN") or os.environ.get(
            "CRAFTLY_WORKSPACE_BOOTSTRAP_TOKEN"
        )
        if not token:
            raise RuntimeError("CRAFTLY_BOOTSTRAP_TOKEN is required")
        verify_tls = os.environ.get("CRAFTLY_WORKSPACE_VERIFY_TLS", "1") != "0"
        return cls(url=url, bootstrap_token=token, verify_tls=verify_tls)

    async def authenticate(self) -> TokenSession:
        if not self.bootstrap_token:
            raise RuntimeError("workspace bootstrap token is not configured")
        response = await self.client.post(
            "/v1/training/token/exchange",
            json={"bootstrap_token": self.bootstrap_token},
        )
        response.raise_for_status()
        payload = response.json()
        refresh_expiry = payload.get("refresh_expires_at")
        if isinstance(refresh_expiry, str):
            from datetime import datetime

            refresh_expires_at = datetime.fromisoformat(refresh_expiry).timestamp()
        else:
            refresh_expires_at = time.time() + 43_200
        self.tokens = TokenSession(
            access_token=payload["access_token"],
            access_expires_at=time.time() + int(payload.get("expires_in", 900)) - 30,
            session_id=payload["session_id"],
            refresh_token=payload["refresh_token"],
            refresh_expires_at=refresh_expires_at,
        )
        return self.tokens

    async def _refresh(self) -> None:
        if not self.tokens or self.tokens.refresh_expires_at <= time.time():
            await self.authenticate()
            return
        response = await self.client.post(
            "/v1/training/token/refresh",
            json={"session_id": self.tokens.session_id, "refresh_token": self.tokens.refresh_token},
        )
        if response.status_code == 401:
            await self.authenticate()
            return
        response.raise_for_status()
        payload = response.json()
        self.tokens.access_token = payload["access_token"]
        self.tokens.access_expires_at = time.time() + int(payload.get("expires_in", 900)) - 30

    async def _access_token(self) -> str:
        if not self.tokens:
            await self.authenticate()
        elif self.tokens.access_expires_at <= time.time():
            await self._refresh()
        assert self.tokens is not None
        return self.tokens.access_token

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        token = await self._access_token()
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {token}"
        response = await self.client.request(method, path, headers=headers, **kwargs)
        if response.status_code == 401:
            await self._refresh()
            assert self.tokens is not None
            headers["Authorization"] = f"Bearer {self.tokens.access_token}"
            response = await self.client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    async def profiles(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/training/profiles")).json()["profiles"]

    async def campaigns(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/training/campaigns")).json()["campaigns"]

    async def train(
        self,
        profile: str,
        *,
        follow: bool = False,
        project_id: str | None = None,
        idempotency_key: str | None = None,
        overrides: dict[str, int] | None = None,
        dataset_manifest_uri: str | None = None,
        dataset_manifest_sha256: str | None = None,
        tokenizer_uri: str | None = None,
        tokenizer_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
        qualification_campaign_id: str | None = None,
        qualification_milestone_id: str | None = None,
    ) -> TrainingRun:
        request = {
            "profile_id": profile,
            "project_id": project_id,
            "idempotency_key": idempotency_key or f"client-{uuid.uuid4()}",
            "overrides": overrides or {},
            "dataset_manifest_uri": dataset_manifest_uri,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_uri": tokenizer_uri,
            "tokenizer_sha256": tokenizer_sha256,
            "git_commit": _git_commit(),
            "container_digest": _container_digest(),
            "metadata": metadata or {},
            "qualification_campaign_id": qualification_campaign_id,
            "qualification_milestone_id": qualification_milestone_id,
        }
        response = await self._request("POST", "/v1/training/jobs", json=request)
        run = TrainingRun(self, response.json())
        print(
            f"[craftly-workspace] job={run.job_id} status={run.status} profile={profile}",
            flush=True,
        )
        if follow:
            await run.follow()
        return run

    async def jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/training/jobs", params={"limit": limit})).json()[
            "jobs"
        ]

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return (await self._request("GET", f"/v1/training/jobs/{job_id}")).json()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return (await self._request("POST", f"/v1/training/jobs/{job_id}/cancel")).json()

    async def resume(self, job_id: str) -> dict[str, Any]:
        return (await self._request("POST", f"/v1/training/jobs/{job_id}/resume")).json()

    async def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        return (await self._request("GET", f"/v1/training/jobs/{job_id}/artifacts")).json()[
            "artifacts"
        ]

    async def audit(self, job_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return (
            await self._request(
                "GET",
                f"/v1/training/jobs/{job_id}/audit",
                params={"limit": min(max(limit, 1), 500)},
            )
        ).json()["audit_events"]

    async def revoke(self) -> bool:
        if not self.tokens:
            return False
        response = await self._request(
            "POST",
            "/v1/training/token/revoke",
            json={"session_id": self.tokens.session_id, "refresh_token": self.tokens.refresh_token},
        )
        self.tokens = None
        return bool(response.json().get("revoked"))

    async def presign_artifact(self, job_id: str, upload: dict[str, Any]) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                f"/v1/training/jobs/{job_id}/artifacts/presign",
                json=upload,
            )
        ).json()

    async def register_artifact(
        self, job_id: str, upload: dict[str, Any], uri: str
    ) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                f"/v1/training/jobs/{job_id}/artifacts/register",
                json={"upload": upload, "uri": uri},
            )
        ).json()

    async def commit_checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
        return (
            await self._request("POST", f"/v1/training/jobs/{job_id}/checkpoints", json=checkpoint)
        ).json()

    async def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        return (await self._request("GET", f"/v1/training/jobs/{job_id}/checkpoints")).json()[
            "checkpoints"
        ]

    async def commit_evaluation(self, job_id: str, evaluation: dict[str, Any]) -> dict[str, Any]:
        return (
            await self._request("POST", f"/v1/training/jobs/{job_id}/evaluations", json=evaluation)
        ).json()

    async def evaluations(self, job_id: str) -> list[dict[str, Any]]:
        return (await self._request("GET", f"/v1/training/jobs/{job_id}/evaluations")).json()[
            "evaluations"
        ]

    async def worker_token(self, job_id: str, *, ttl_seconds: int = 21_600) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                f"/v1/training/jobs/{job_id}/worker-token",
                params={"ttl_seconds": ttl_seconds},
            )
        ).json()

    async def claim_notebook_job(self, job_id: str) -> dict[str, Any]:
        return (await self._request("POST", f"/v1/training/jobs/{job_id}/claim")).json()

    async def emit_events(
        self, job_id: str, attempt_id: str, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                f"/v1/training/jobs/{job_id}/events:batch",
                json={"attempt_id": attempt_id, "events": events},
            )
        ).json()

    async def events(
        self, job_id: str, *, after_sequence: int = 0
    ) -> AsyncIterator[dict[str, Any]]:
        backoff = 1.0
        while True:
            token = await self._access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
                "Last-Event-ID": str(after_sequence),
            }
            try:
                async with self.client.stream(
                    "GET",
                    f"/v1/training/jobs/{job_id}/events",
                    headers=headers,
                    params={"after_sequence": after_sequence},
                ) as response:
                    if response.status_code == 401:
                        await self._refresh()
                        continue
                    response.raise_for_status()
                    data_lines: list[str] = []
                    event_id: int | None = None
                    async for line in response.aiter_lines():
                        if line.startswith("id:"):
                            event_id = int(line.split(":", 1)[1].strip())
                        elif line.startswith("data:"):
                            data_lines.append(line.split(":", 1)[1].lstrip())
                        elif line == "" and data_lines:
                            event = json.loads("\n".join(data_lines))
                            if event_id is not None:
                                after_sequence = max(after_sequence, event_id)
                            yield event
                            data_lines = []
                            event_id = None
                    state = await self.get_job(job_id)
                    if state.get("status") in TERMINAL_STATES:
                        return
            except (httpx.TransportError, httpx.TimeoutException):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            backoff = 1.0

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Workspace:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


def format_event(event: dict[str, Any]) -> str:
    fields = [
        f"seq={event.get('sequence', 0)}",
        f"stage={event.get('stage', '')}",
        f"status={event.get('status', '')}",
    ]
    for key in [
        "step",
        "max_steps",
        "loss",
        "validation_loss",
        "tokens_per_second",
        "gpu_memory_bytes",
    ]:
        if event.get(key) is not None:
            fields.append(f"{key}={event[key]}")
    if event.get("message"):
        fields.append(f"message={event['message']}")
    return "[craftly-live] " + " ".join(fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Craftly Training Workspace client")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--profile", required=True)
    train.add_argument("--follow", action="store_true")
    train.add_argument("--dataset-manifest-uri")
    train.add_argument("--dataset-manifest-sha256")
    train.add_argument("--tokenizer-uri")
    train.add_argument("--tokenizer-sha256")
    train.add_argument("--steps", type=int)
    train.add_argument("--campaign")
    train.add_argument("--milestone")
    subparsers.add_parser("campaigns")
    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("action", choices=["list", "inspect", "cancel", "resume"])
    jobs.add_argument("job_id", nargs="?")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    async with Workspace.from_environment() as workspace:
        if args.command == "train":
            run = await workspace.train(
                args.profile,
                follow=args.follow,
                dataset_manifest_uri=args.dataset_manifest_uri,
                dataset_manifest_sha256=args.dataset_manifest_sha256,
                tokenizer_uri=args.tokenizer_uri,
                tokenizer_sha256=args.tokenizer_sha256,
                overrides={"steps": args.steps} if args.steps is not None else None,
                qualification_campaign_id=args.campaign,
                qualification_milestone_id=args.milestone,
            )
            print(json.dumps(run.payload, indent=2, sort_keys=True), flush=True)
            return
        if args.command == "campaigns":
            print(json.dumps(await workspace.campaigns(), indent=2, sort_keys=True), flush=True)
            return
        if args.action == "list":
            payload: Any = await workspace.jobs()
        else:
            if not args.job_id:
                raise SystemExit("job_id is required")
            if args.action == "inspect":
                payload = await workspace.get_job(args.job_id)
            elif args.action == "cancel":
                payload = await workspace.cancel(args.job_id)
            else:
                payload = await workspace.resume(args.job_id)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
