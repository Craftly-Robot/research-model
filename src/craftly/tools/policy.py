"""Hardened host tool execution policy.

This module is the only approved host-command execution path for gateway and
verification routes. It is intentionally smaller than a general shell: commands
are argv-only, executable names are resolved outside the project tree, and the
child process receives a scrubbed environment with no application secrets.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - subprocess is wrapped by strict argv policy.
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.shared.schemas import StrictModel


MAX_OUTPUT_CHARS = 20_000
PATHLIKE_CHARS = {"/", "\\"}

ALLOWED_TOOL_COMMANDS: dict[str, set[str]] = {
    "git_diff": {"git"},
    "test": {"python", "python.exe", "python3", "pytest", "pytest.exe", "npm", "npm.cmd", "node", "node.exe"},
    "shell": {"python", "python.exe", "python3", "pytest", "pytest.exe", "git", "npm", "npm.cmd", "node", "node.exe"},
}


class ToolExecuteRequest(StrictModel):
    project_id: str
    run_id: str | None = None
    tool: str = Field(pattern="^(shell|test|git_diff)$")
    command: list[str] = Field(min_length=1)
    timeout_ms: int = Field(default=30_000, ge=1_000, le=300_000)


class ToolExecuteResponse(StrictModel):
    tool_call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    run_id: str | None = None
    tool: str
    status: str
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: float


class SandboxPolicy(StrictModel):
    timeout_ms: int = 30_000
    network_disabled: bool = True


class SandboxRequest(StrictModel):
    command: list[str] = Field(min_length=1)


class ExecutionRequest(ToolExecuteRequest):
    pass


class ExecutionResult(ToolExecuteResponse):
    pass


def project_root(store: LocalStore, project_id: str) -> Path:
    project = store.get_project(project_id)
    if project is None:
        raise KeyError(f"unknown project: {project_id}")
    root = Path(str(project["repo_path"])).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"project repo_path is not a directory: {root}")
    return root


class ResolvedCommand(StrictModel):
    executable: str
    argv: list[str]
    cwd: str
    env_keys: list[str] = Field(default_factory=list)


class HardenedToolExecutor:
    """Execute allowlisted local tooling with deterministic security checks."""

    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store or LocalStore()

    def execute(self, request: ToolExecuteRequest) -> ToolExecuteResponse:
        root = project_root(self.store, request.project_id)
        resolved = self.resolve_command(request, root=root)
        started = time.perf_counter()
        try:
            completed = subprocess.run(  # nosec B603 - strict argv, shell disabled, resolved executable. # nosemgrep: python.django.security.injection.command.subprocess-injection.subprocess-injection
                resolved.argv,
                cwd=resolved.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=request.timeout_ms / 1000,
                check=False,
                env=self.secret_free_env(),
            )
            return ToolExecuteResponse(
                project_id=request.project_id,
                run_id=request.run_id,
                tool=request.tool,
                status="ok" if completed.returncode == 0 else "failed",
                stdout=(completed.stdout or "")[-MAX_OUTPUT_CHARS:],
                stderr=(completed.stderr or "")[-MAX_OUTPUT_CHARS:],
                exit_code=completed.returncode,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return ToolExecuteResponse(
                project_id=request.project_id,
                run_id=request.run_id,
                tool=request.tool,
                status="timeout",
                stdout=stdout[-MAX_OUTPUT_CHARS:],
                stderr=stderr[-MAX_OUTPUT_CHARS:],
                exit_code=None,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    def resolve_command(self, request: ToolExecuteRequest, *, root: Path) -> ResolvedCommand:
        if not request.command:
            raise ValueError("command must not be empty")
        for arg in request.command:
            if "\x00" in arg:
                raise ValueError("command arguments cannot contain NUL bytes")
        executable_name = request.command[0].lower()
        if any(char in request.command[0] for char in PATHLIKE_CHARS):
            raise ValueError("command executable must be a basename, not a path")
        allowed = ALLOWED_TOOL_COMMANDS.get(request.tool, set())
        if executable_name not in allowed:
            raise ValueError(f"command {executable_name!r} is not allowed for tool {request.tool!r}")
        self.validate_tool_shape(request)
        resolved = self.resolve_executable(request.command[0], root=root)
        return ResolvedCommand(
            executable=str(resolved),
            argv=[str(resolved), *request.command[1:]],
            cwd=str(root),
            env_keys=sorted(self.secret_free_env().keys()),
        )

    def resolve_executable(self, executable: str, *, root: Path) -> Path:
        if executable.lower() in {"python", "python.exe", "python3"}:
            interpreter = Path(sys.executable).resolve()
            if not interpreter.is_file():
                raise FileNotFoundError("running Python interpreter is not a regular file")
            if self.is_inside(interpreter, root):
                raise ValueError(
                    f"refusing project-local Python interpreter to prevent executable shadowing: {interpreter}"
                )
            return interpreter
        search_path = self.safe_path(root)
        resolved = shutil.which(executable, path=search_path)
        if resolved is None:
            raise FileNotFoundError(f"allowed executable not found on safe PATH: {executable}")
        path = Path(resolved).resolve()
        if self.is_inside(path, root):
            raise ValueError(f"refusing project-local executable to prevent PATH shadowing: {path}")
        return path

    def validate_tool_shape(self, request: ToolExecuteRequest) -> None:
        command = [item.lower() for item in request.command]
        if request.tool == "git_diff":
            if command[:2] != ["git", "diff"]:
                raise ValueError("git_diff tool may only run: git diff")
            for arg in request.command[2:]:
                if arg.startswith("--output") or arg.startswith("--ext-diff"):
                    raise ValueError("git diff output/ext-diff options are not allowed")
        if request.tool == "test" and command[0] in {"npm", "npm.cmd"}:
            if len(command) < 2 or command[1] not in {"test", "run"}:
                raise ValueError("npm test tool may only run npm test or npm run <script>")
        if request.tool in {"test", "shell"} and command[0] == "git":
            if len(command) > 1 and command[1] not in {"diff", "status", "show", "rev-parse"}:
                raise ValueError("git commands are limited to read-only inspection commands")

    def secret_free_env(self) -> dict[str, str]:
        temp_dir = tempfile.gettempdir()
        env: dict[str, str] = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "TMP": os.environ.get("TMP", temp_dir),
            "TEMP": os.environ.get("TEMP", temp_dir),
            "TMPDIR": os.environ.get("TMPDIR", temp_dir),
        }
        for key in ["SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "HOME", "USERPROFILE"]:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def safe_path(self, root: Path) -> str:
        safe_parts: list[str] = []
        cwd = Path.cwd().resolve()
        for raw in os.environ.get("PATH", "").split(os.pathsep):
            if not raw:
                continue
            try:
                path = Path(raw).resolve()
            except OSError:
                continue
            if path in {root, cwd} or self.is_inside(path, root):
                continue
            safe_parts.append(str(path))
        return os.pathsep.join(safe_parts)

    @staticmethod
    def is_inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def execute_hardened_tool(store: LocalStore | None, request: ToolExecuteRequest) -> dict[str, Any]:
    return HardenedToolExecutor(store).execute(request).model_dump()

