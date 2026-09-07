"""Hardened Docker sandbox runner for untrusted code execution."""

from __future__ import annotations

import os
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from src.craftly.shared.schemas import StrictModel


SANDBOX_TMPFS = {
    "/tmp": "rw,noexec,nosuid,nodev,size=32m,mode=1777",  # nosec B108 - isolated container tmpfs.
    "/workspace": "rw,nosuid,nodev,size=128m,mode=1777",
}
MAX_SANDBOX_FILES = 20_000
MAX_SANDBOX_INPUT_BYTES = 256_000_000
MAX_SANDBOX_OUTPUT_BYTES = 2_000_000


class SandboxOutputLimitExceeded(RuntimeError):
    pass


def _normalize_sandbox_path(relative: str) -> str:
    normalized = relative.replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or normalized.startswith("../")
        or "/../" in f"/{normalized}/"
        or "\x00" in normalized
    ):
        raise ValueError(f"unsafe sandbox file path: {relative}")
    return normalized


class HardenedSandboxPolicy(StrictModel):
    image: str = "ubuntu:24.04"
    user: Literal["65534:65534"] = "65534:65534"
    network_mode: Literal["none"] = "none"
    mem_limit: Literal["512m"] = "512m"
    nano_cpus: int = Field(default=1_000_000_000, ge=1_000_000_000, le=1_000_000_000)
    read_only: Literal[True] = True
    cap_drop: list[str] = Field(default_factory=lambda: ["ALL"])
    security_opt: list[str] = Field(default_factory=lambda: ["no-new-privileges:true"])
    tmpfs: dict[str, str] = Field(default_factory=lambda: dict(SANDBOX_TMPFS))
    pids_limit: int = Field(default=64, ge=16, le=64)
    max_output_bytes: int = Field(default=MAX_SANDBOX_OUTPUT_BYTES, ge=64_000, le=MAX_SANDBOX_OUTPUT_BYTES)
    timeout_ms: int = Field(default=5_000, ge=100, le=300_000)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        configured = os.environ.get("CRAFTLY_SANDBOX_IMAGES", "python:3.12-slim,ubuntu:24.04")
        allowed = {item.strip() for item in configured.split(",") if item.strip()}
        if value not in allowed:
            raise ValueError(f"sandbox image is not allowlisted: {value}")
        if os.environ.get("CRAFTLY_REQUIRE_PINNED_SANDBOX_IMAGE", "0") == "1" and "@sha256:" not in value:
            raise ValueError("sandbox image must be pinned by sha256 digest")
        return value

    @model_validator(mode="after")
    def enforce_kernel_policy(self) -> "HardenedSandboxPolicy":
        if self.cap_drop != ["ALL"]:
            raise ValueError("sandbox must drop every Linux capability")
        if self.security_opt != ["no-new-privileges:true"]:
            raise ValueError("sandbox must enforce no-new-privileges")
        if self.tmpfs != SANDBOX_TMPFS:
            raise ValueError("sandbox tmpfs policy cannot be weakened by callers")
        return self


class SandboxRunRequest(StrictModel):
    command: list[str] = Field(min_length=1)
    files: dict[str, str] = Field(default_factory=dict)
    workdir: Literal["/workspace"] = "/workspace"
    policy: HardenedSandboxPolicy = Field(default_factory=HardenedSandboxPolicy)

    @field_validator("command")
    @classmethod
    def validate_command(cls, command: list[str]) -> list[str]:
        if len(command) > 100:
            raise ValueError("sandbox command exceeds 100 argv items")
        if any(not item or "\x00" in item or len(item) > 4096 for item in command):
            raise ValueError("sandbox command contains an invalid argv item")
        return command

    @field_validator("files")
    @classmethod
    def validate_files(cls, files: dict[str, str]) -> dict[str, str]:
        if len(files) > MAX_SANDBOX_FILES:
            raise ValueError(f"sandbox input exceeds {MAX_SANDBOX_FILES} files")
        total = 0
        for relative, content in files.items():
            _normalize_sandbox_path(relative)
            total += len(content.encode("utf-8"))
            if total > MAX_SANDBOX_INPUT_BYTES:
                raise ValueError(f"sandbox input exceeds {MAX_SANDBOX_INPUT_BYTES} bytes")
        return files


class SandboxRunResult(StrictModel):
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float
    policy: dict[str, Any]
    reason: str = ""


class DockerSandboxRunner:
    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        started = time.perf_counter()
        try:
            import docker
        except ImportError:
            return self.unavailable(request, started, "docker SDK is not installed")
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            return self.unavailable(request, started, f"docker engine unavailable: {exc}")

        container = None
        try:
            container = client.containers.create(
                request.policy.image,
                command=["sleep", "300"],
                detach=True,
                network_mode=request.policy.network_mode,
                mem_limit=request.policy.mem_limit,
                nano_cpus=request.policy.nano_cpus,
                read_only=request.policy.read_only,
                cap_drop=request.policy.cap_drop,
                security_opt=request.policy.security_opt,
                tmpfs=request.policy.tmpfs,
                pids_limit=request.policy.pids_limit,
                ulimits=[
                    docker.types.Ulimit(name="nofile", soft=1024, hard=1024),
                    docker.types.Ulimit(name="nproc", soft=request.policy.pids_limit, hard=request.policy.pids_limit),
                ],
                init=True,
                ipc_mode="none",
                stdin_open=False,
                tty=False,
                environment={
                    "HOME": "/tmp",  # nosec B108 - isolated container tmpfs, not host storage.
                    "TMPDIR": "/tmp",  # nosec B108 - isolated container tmpfs, not host storage.
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                labels={"craftly.sandbox": "untrusted-execution"},
                user=request.policy.user,
                working_dir=request.workdir,
            )
            container.start()
            if request.files:
                container.put_archive(request.workdir, self.files_to_tar(request.files))
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._exec_bounded, client, container, request)
            try:
                exit_code, stdout_raw, stderr_raw = future.result(timeout=request.policy.timeout_ms / 1000)
            except TimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                try:
                    container.kill()
                except Exception:
                    pass
                return SandboxRunResult(
                    status="timeout",
                    stderr="<|timeout|>",
                    exit_code=None,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    policy=request.policy.model_dump(),
                    reason="sandbox timeout exceeded",
                )
            except SandboxOutputLimitExceeded as exc:
                try:
                    container.kill()
                except Exception:
                    pass
                return SandboxRunResult(
                    status="failed",
                    stderr=str(exc),
                    exit_code=None,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    policy=request.policy.model_dump(),
                    reason="sandbox output limit exceeded",
                )
            finally:
                if not future.cancelled():
                    executor.shutdown(wait=False, cancel_futures=True)
            return SandboxRunResult(
                status="ok" if exit_code == 0 else "failed",
                stdout=(stdout_raw or b"").decode("utf-8", errors="replace")[-20_000:],
                stderr=(stderr_raw or b"").decode("utf-8", errors="replace")[-20_000:],
                exit_code=exit_code,
                duration_ms=(time.perf_counter() - started) * 1000,
                policy=request.policy.model_dump(),
            )
        except Exception as exc:
            return SandboxRunResult(
                status="failed",
                stderr=str(exc),
                exit_code=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                policy=request.policy.model_dump(),
                reason="sandbox execution failed",
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass

    def unavailable(self, request: SandboxRunRequest, started: float, reason: str) -> SandboxRunResult:
        return SandboxRunResult(
            status="unavailable",
            duration_ms=(time.perf_counter() - started) * 1000,
            policy=request.policy.model_dump(),
            reason=reason,
        )

    def files_to_tar(self, files: dict[str, str]) -> bytes:
        stream = BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for relative, content in files.items():
                normalized = _normalize_sandbox_path(relative)
                data = content.encode("utf-8")
                info = tarfile.TarInfo(normalized)
                info.size = len(data)
                archive.addfile(info, BytesIO(data))
        stream.seek(0)
        return stream.read()

    @staticmethod
    def _exec_bounded(
        client: Any,
        container: Any,
        request: SandboxRunRequest,
    ) -> tuple[int, bytes, bytes]:
        execution = client.api.exec_create(
            container.id,
            request.command,
            stdout=True,
            stderr=True,
            stdin=False,
            tty=False,
            privileged=False,
            user=request.policy.user,
            workdir=request.workdir,
        )
        execution_id = str(execution["Id"])
        stdout = bytearray()
        stderr = bytearray()
        for stdout_chunk, stderr_chunk in client.api.exec_start(
            execution_id,
            detach=False,
            tty=False,
            stream=True,
            demux=True,
        ):
            if stdout_chunk:
                stdout.extend(stdout_chunk)
            if stderr_chunk:
                stderr.extend(stderr_chunk)
            if len(stdout) + len(stderr) > request.policy.max_output_bytes:
                raise SandboxOutputLimitExceeded(
                    f"sandbox output exceeded {request.policy.max_output_bytes} bytes"
                )
        inspected = client.api.exec_inspect(execution_id)
        exit_code = inspected.get("ExitCode")
        if not isinstance(exit_code, int):
            raise RuntimeError("Docker exec completed without a valid integer exit code")
        return exit_code, bytes(stdout), bytes(stderr)


def run_local_sandbox_smoke(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    request = SandboxRunRequest(command=["python3", "-c", "print('sandbox-ok')"])
    result = DockerSandboxRunner().run(request)
    report = result.model_dump()
    (root / "sandbox_report.json").write_text(str(report), encoding="utf-8")
    return report

